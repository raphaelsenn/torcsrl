from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_torcs import rules
from gym_torcs.constants import (
    DEFAULT_TORCS_EXECUTABLE,
    FAILURE_REWARD,
    LOOKAHEAD_DISTANCES,
    RACING_LINE_DIR,
    STALL_REWARD,
)
from gym_torcs.core.client import Action, Sensors, TorcsClient
from gym_torcs.core.server import TorcsServer, TrackSpec
from gym_torcs.core.vision import FRAME_HEIGHT, FRAME_WIDTH, FrameBuffer, resize
from gym_torcs.curvature import RacingLineCurvatureMap
from gym_torcs.observation import STATE_SIZE, state_vector

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT = 120.0
FIRST_SENSORS_TIMEOUT = 60.0
STEP_TIMEOUT = 10.0
MAX_WAIT_STEPS = 1_000          # for the first camera frame and the start countdown
MAX_APPROACH_STEPS = 50_000     # tita driving to a random start
IDLE = np.zeros(2, dtype=np.float32)


class TorcsEnv(gym.Env):
    """A TORCS race on one track, driven through the SCR server.

    The action is [steer, throttle]. Negative throttle brakes. Gears shift automatically.
    See the README for observations, reward and info.

    Args:
        render_mode: None runs TORCS headless. "human" opens the TORCS window.
        port: UDP port of the SCR driver, 3001 to 3010. Each env needs its own port.
        host: Address of the machine running TORCS.
        track: Track to race on. reset can switch it with options={"track": ...}.
        laps: Laps in the TORCS race. TORCS ends the race after that many, and the next
            step is truncated with end_reason "race_over".
        vision: Also return a camera image. Needs render_mode="human".
        image_shape: (height, width) of the camera image.
        observation_noise: Std of the Gaussian noise added to the state.
        max_steps: Truncate the episode after this many steps (20 ms each).
        truncate_on_lap: Truncate the episode when a lap is finished.
        random_start: Chance that an episode starts somewhere on the lap instead of
            on the grid. tita drives the car there first.
        failure_reward: Reward of the step in which the car crashes, leaves the track or
            turns backwards.
        stall_reward: Reward of the step in which the car is declared stalled. Lower than
            failure_reward so that stopping is never a way to avoid a crash (see constants.py).
        executable: Path to the TORCS start script.
        racing_line_dir: Folder with one racing line CSV per track, used for the
            curvature features and random starts. Without a CSV, tita's own curvature
            estimate is used and random starts are off.
        startup_sleep: Seconds to wait after starting TORCS before connecting.
        verbose: Show the TORCS output.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        port: int = 3001,
        host: str = "127.0.0.1",
        track: TrackSpec = TrackSpec("michigan", "oval"),
        laps: int = 20,
        vision: bool = False,
        image_shape: tuple[int, int] = (64, 64),
        observation_noise: float = 0.01,
        max_steps: int = 25_000,
        truncate_on_lap: bool = True,
        random_start: float = 0.0,
        failure_reward: float = FAILURE_REWARD,
        stall_reward: float = STALL_REWARD,
        executable: str = DEFAULT_TORCS_EXECUTABLE,
        racing_line_dir: str | Path = RACING_LINE_DIR,
        startup_sleep: float = 1.0,
        verbose: bool = False,
    ) -> None:
        if render_mode not in (None, "human"):
            raise ValueError(f"render_mode must be None or 'human', got {render_mode!r}.")
        if vision and render_mode != "human":
            raise ValueError("vision=True needs render_mode='human', frames come from the TORCS window.")
        if not 0.0 <= random_start <= 1.0:
            raise ValueError(f"random_start must be a probability, got {random_start}.")

        self.render_mode = render_mode
        self.host = host
        self.port = port
        self.track = track
        self.laps = laps
        self.image_shape = image_shape
        self.observation_noise = observation_noise
        self.max_steps = max_steps
        self.truncate_on_lap = truncate_on_lap
        self.random_start = random_start
        self.failure_reward = failure_reward
        self.stall_reward = stall_reward
        self.racing_line_dir = Path(racing_line_dir)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        state_space = spaces.Box(-1.0, 1.0, shape=(STATE_SIZE,), dtype=np.float32)
        if vision:
            image_space = spaces.Box(0, 255, shape=(*image_shape, 3), dtype=np.uint8)
            self.observation_space = spaces.Dict({"state": state_space, "image": image_space})
        else:
            self.observation_space = state_space

        self.server = TorcsServer(
            port,
            executable=executable,
            gui=render_mode == "human",
            window_size=(FRAME_WIDTH, FRAME_HEIGHT) if vision else None,
            startup_sleep=startup_sleep,
            verbose=verbose,
        )
        self.frames = FrameBuffer() if vision else None
        self.client: TorcsClient | None = None

        self._racing_lines: dict[str, RacingLineCurvatureMap | None] = {}
        self._racing_line: RacingLineCurvatureMap | None = None
        self._sensors: Sensors = {}
        self._last_obs: Any = None
        self._step_count = 0
        self._laps_completed = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Restart TORCS. Pass options={"track": ...} to change the track."""
        super().reset(seed=seed)
        if options and "track" in options:
            self.track = options["track"]
        self._racing_line = self._load_racing_line(self.track.name)

        self._disconnect()
        self.server.start(self.track, self.laps)
        if self.frames is not None:
            self.frames.start_capture()
        self.client = TorcsClient(self.host, self.port)
        self.client.connect(CONNECT_TIMEOUT)
        sensors = self.client.receive(FIRST_SENSORS_TIMEOUT, keepalive=Action())

        if self.frames is not None:
            sensors = self._drive_until(sensors, lambda s: self.frames.has_frame, MAX_WAIT_STEPS,
                                        error="TORCS sent no camera frame. Is the race running in a visible window?")
        # Wait out the start countdown.
        sensors = self._drive_until(sensors, lambda s: s["curLapTime"] >= 0.0, MAX_WAIT_STEPS,
                                    error="The race did not start: curLapTime stayed negative.")

        start_on_lap = self._racing_line is not None and self.np_random.random() < self.random_start
        if start_on_lap:
            target = self.np_random.uniform(0.0, self._racing_line.track_length)
            sensors = self._drive_expert_to(sensors, target)

        self._sensors = sensors
        self._step_count = 0
        self._laps_completed = 0
        self._last_obs = self._observe(sensors)
        info = {
            "track": asdict(self.track),
            "track_length": self._racing_line.track_length if self._racing_line else None,
            "random_start": start_on_lap,
            **self._info(sensors),
        }
        return self._last_obs, info

    def step(self, action: np.ndarray):
        if self.client is None:
            raise RuntimeError("Call reset() before step().")

        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        prev = self._sensors
        self._send(action, prev)
        try:
            sensors = self.client.receive(STEP_TIMEOUT)
        except (TimeoutError, ConnectionError) as error:
            # ConnectionError: TORCS ended the race, e.g. after its last lap (see `laps`).
            end_reason = "torcs_timeout" if isinstance(error, TimeoutError) else "race_over"
            logger.warning("TORCS on port %d stopped at step %d: %s", self.port, self._step_count, end_reason)
            self._disconnect()
            # Return the last state we have.
            return self._last_obs, 0.0, False, True, self._info(prev, end_reason=end_reason)

        self._sensors = sensors
        self._step_count += 1
        lap_completed = rules.lap_completed(sensors, prev)
        self._laps_completed += int(lap_completed)

        end_reason = rules.failure(sensors, prev, self._step_count)
        terminated = end_reason is not None
        if terminated:
            reward = self.stall_reward if end_reason == "stalled" else self.failure_reward
        else:
            reward = rules.progress_reward(sensors)
            if self.truncate_on_lap and lap_completed:
                end_reason = "lap_completed"
            elif self._step_count >= self.max_steps:
                end_reason = "max_steps"
        truncated = not terminated and end_reason is not None

        self._last_obs = self._observe(sensors)
        info = self._info(sensors, end_reason, lap_completed, sensors["damage"] - prev["damage"])
        return self._last_obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._disconnect()
        self.server.close()
        if self.frames is not None:
            self.frames.close()

    # ------------------------------------------------------------------ driving

    def _send(self, action: np.ndarray, prev: Sensors) -> None:
        self.client.send(rules.command(float(action[0]), float(action[1]), prev))

    def _drive_until(
        self,
        sensors: Sensors,
        done: Callable[[Sensors], bool],
        max_steps: int,
        policy: Callable[[Sensors], np.ndarray] = lambda s: IDLE,
        error: str | None = None,
    ) -> Sensors:
        """Drive with policy until done(sensors) is true.

        Gives up after max_steps: raises if error is set, else returns the last sensors.
        """
        for _ in range(max_steps):
            if done(sensors):
                return sensors
            self._send(policy(sensors), sensors)
            sensors = self.client.receive(STEP_TIMEOUT)
        if error is not None:
            raise RuntimeError(error)
        return sensors

    def _drive_expert_to(self, sensors: Sensors, distance: float) -> Sensors:
        """Let tita drive until the car has covered distance meters and is on the track."""
        def arrived(s: Sensors) -> bool:
            return s["distRaced"] >= distance and abs(s["trackPos"]) < 0.9
        return self._drive_until(sensors, arrived, MAX_APPROACH_STEPS, policy=rules.expert_action)

    def _disconnect(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    # ------------------------------------------------------------- observation

    def _observe(self, sensors: Sensors) -> np.ndarray | dict[str, np.ndarray]:
        state = self._state(sensors)
        if self.frames is None:
            return state
        return {"state": state, "image": resize(self.frames.read(), *self.image_shape)}

    def _state(self, s: Sensors) -> np.ndarray:
        if self._racing_line is not None:
            curvature = self._racing_line.lookahead(s["distFromStart"], LOOKAHEAD_DISTANCES)
        else:
            curvature = np.array([s[f"curv{d}"] for d in LOOKAHEAD_DISTANCES])  # estimated by tita
        state = state_vector(s, curvature)
        if self.observation_noise > 0.0:
            state += self.np_random.normal(0.0, self.observation_noise, size=state.shape).astype(np.float32)
        return np.clip(state, -1.0, 1.0)

    def _info(
        self,
        s: Sensors,
        end_reason: str | None = None,
        lap_completed: bool = False,
        damage_delta: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "end_reason": end_reason,
            "lap_completed": lap_completed,
            "laps_completed": self._laps_completed,
            "lap_time": s["lastLapTime"],
            "current_lap_time": s["curLapTime"],
            "episode_steps": self._step_count,
            "distance_raced": s["distRaced"],
            "distance_from_start": s["distFromStart"],
            "speed": s["speedX"],
            "track_position": s["trackPos"],
            "position_x": s["x"],
            "position_y": s["y"],
            "damage": s["damage"],
            "damage_delta": damage_delta,
            "expert_action": rules.expert_action(s),
        }

    def _load_racing_line(self, track_name: str) -> RacingLineCurvatureMap | None:
        if track_name not in self._racing_lines:
            path = self.racing_line_dir / f"{track_name}.csv"
            if path.exists():
                self._racing_lines[track_name] = RacingLineCurvatureMap.from_csv(path)
            else:
                logger.warning("No racing line at %s, using curvature from TORCS.", path)
                self._racing_lines[track_name] = None
        return self._racing_lines[track_name]
