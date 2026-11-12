from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_torcs.client import Action, Sensors, TorcsClient
from gym_torcs.constants import (
    DEFAULT_TORCS_EXECUTABLE,
    FAILURE_REWARD,
    LOOKAHEAD_DISTANCES,
    MAX_ACCEL,
    MAX_CURVATURE,
    MAX_RPM,
    MAX_SPEED,
    MAX_TRACK,
    MAX_WHEEL_SPIN_VEL,
    MIN_SPEED,
    RACING_LINE_DIR,
    STALL_GRACE_STEPS,
)
from gym_torcs.curvature import RacingLineCurvatureMap
from gym_torcs.server import TorcsServer, TrackSpec
from gym_torcs.vision import FRAME_HEIGHT, FRAME_WIDTH, FrameBuffer, resize

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT = 120.0
FIRST_SENSORS_TIMEOUT = 60.0
STEP_TIMEOUT = 10.0
MAX_STEPS_WITHOUT_FRAME = 500
STATE_SIZE = 45


class TorcsEnv(gym.Env):
    """TORCS race on a single track, driven through the SCR server robot.

    Actions are ``[steer, throttle]`` in ``[-1, 1]``. Negative throttle brakes,
    and gears are shifted automatically.

    The state is a vector of 45 values, each scaled to ``[-1, 1]``:

    ====  ===================================================================
    0-2   speed along the car's x, y, z axes
    3-5   acceleration along the car's x, y, z axes
    6     engine rpm
    7     angle between the car and the track axis
    8     lateral position on the track, -1 is the right edge, 1 the left
    9-27  track edge distances along ``TRACK_SENSOR_ANGLES``
    28-31 wheel spin velocities
    32    offset from the racing line of the ``tita`` robot
    33-44 racing line curvature 0, 25, ..., 275 m ahead
    ====  ===================================================================

    With ``vision=True`` the observation is a dict with the state under
    ``"state"`` and an RGB camera image of ``image_shape`` under ``"image"``.
    Vision needs the TORCS window, so it requires ``render_mode="human"``.

    The reward is the forward speed, discounted by the distance to the racing
    line. Leaving the track, damage, driving backwards or standing still end
    the episode with a reward of ``FAILURE_REWARD``.
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
        observation_noise: float = 0.025,
        max_steps: int = 25_000,
        truncate_on_lap: bool = True,
        executable: str = DEFAULT_TORCS_EXECUTABLE,
        racing_line_dir: str | Path = RACING_LINE_DIR,
        startup_sleep: float = 1.0,
        verbose: bool = False,
    ) -> None:
        if render_mode not in (None, "human"):
            raise ValueError(f"render_mode must be None or 'human', got {render_mode!r}.")
        if vision and render_mode != "human":
            raise ValueError("vision=True needs render_mode='human', frames come from the TORCS window.")

        self.render_mode = render_mode
        self.host = host
        self.port = port
        self.track = track
        self.laps = laps
        self.image_shape = image_shape
        self.observation_noise = observation_noise
        self.max_steps = max_steps
        self.truncate_on_lap = truncate_on_lap
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

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Relaunch TORCS. Pass ``options={"track": TrackSpec(...)}`` to change the track."""
        super().reset(seed=seed)
        if options and "track" in options:
            self.track = options["track"]

        self._disconnect()
        self.server.start(self.track, self.laps)
        if self.frames is not None:
            self.frames.start_capture()

        self.client = TorcsClient(self.host, self.port)
        self.client.connect(CONNECT_TIMEOUT)
        sensors = self.client.receive(FIRST_SENSORS_TIMEOUT, keepalive=Action())
        if self.frames is not None:
            sensors = self._wait_for_first_frame()

        self._sensors = sensors
        self._step_count = 0
        self._racing_line = self._load_racing_line(self.track.name)
        self._last_obs = self._observe(sensors)

        info = {
            "track": asdict(self.track),
            "track_length": self._racing_line.track_length if self._racing_line else None,
            "distFromStart": sensors["distFromStart"],
            "action_tita": _expert_action(sensors),
        }
        return self._last_obs, info

    def step(self, action: np.ndarray):
        if self.client is None:
            raise RuntimeError("Call reset() before step().")

        steer, throttle = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        prev = self._sensors
        self.client.send(Action(
            steer=float(steer),
            accel=max(float(throttle), 0.0),
            brake=max(-float(throttle), 0.0),
            gear=_shift_gear(int(prev["gear"]), prev["rpm"]),
        ))

        try:
            sensors = self.client.receive(STEP_TIMEOUT)
        except TimeoutError:
            logger.warning("TORCS on port %d stopped sending data at step %d.", self.port, self._step_count)
            self._disconnect()
            return self._last_obs, 0.0, False, True, {"torcs_timeout": True, "crashed": True}

        damage_delta = sensors["damage"] - prev["damage"]
        crashed = damage_delta > 0.0
        off_track = abs(sensors["trackPos"]) > 1.0 or min(sensors["track"]) < 0.0
        backwards = np.cos(sensors["angle"]) < 0.0
        stalled = self._step_count > STALL_GRACE_STEPS and sensors["speedX"] < MIN_SPEED
        lap_completed = sensors["lastLapTime"] > 0.0

        self._sensors = sensors
        self._step_count += 1

        terminated = crashed or off_track or backwards or stalled
        truncated = (self.truncate_on_lap and lap_completed) or self._step_count >= self.max_steps
        reward = FAILURE_REWARD if terminated else _progress_reward(sensors)
        self._last_obs = self._observe(sensors)

        info = {
            "successfulLap": lap_completed,
            "offTrack": off_track,
            "crashed": crashed,
            "damage": sensors["damage"],
            "damage_delta": damage_delta,
            "distRaced": sensors["distRaced"],
            "timeAlive": sensors["curLapTime"],
            "speedX": sensors["speedX"],
            "lastLapTime": sensors["lastLapTime"],
            "distFromStart": sensors["distFromStart"],
            "trackPos": sensors["trackPos"],
            "globalX": sensors["x"],
            "globalY": sensors["y"],
            "action_tita": _expert_action(sensors),
        }
        return self._last_obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._disconnect()
        self.server.close()
        if self.frames is not None:
            self.frames.close()

    def _disconnect(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def _wait_for_first_frame(self) -> Sensors:
        """Idle at the start line until TORCS has written the first camera frame."""
        assert self.client is not None and self.frames is not None
        idle = Action()
        for _ in range(MAX_STEPS_WITHOUT_FRAME):
            self.client.send(idle)
            sensors = self.client.receive(STEP_TIMEOUT)
            if self.frames.has_frame:
                return sensors
        raise RuntimeError("TORCS sent no camera frame. Is the race running in a visible window?")

    def _observe(self, sensors: Sensors) -> np.ndarray | dict[str, np.ndarray]:
        state = self._state(sensors)
        if self.frames is None:
            return state
        return {"state": state, "image": resize(self.frames.read(), *self.image_shape)}

    def _state(self, s: Sensors) -> np.ndarray:
        if self._racing_line is not None:
            curvature = self._racing_line.lookahead(s["distFromStart"], LOOKAHEAD_DISTANCES)
        else:
            # Curvature estimated by the tita robot inside TORCS.
            curvature = np.array([s[f"curv{d}"] for d in LOOKAHEAD_DISTANCES])

        racing_line_offset = np.clip((s["trackPos"] - _racing_line_position(s)) / 2.0, -1.0, 1.0)

        state = np.concatenate([
            np.array([s["speedX"], s["speedY"], s["speedZ"]]) / MAX_SPEED,
            np.array([s["accelX"], s["accelY"], s["accelZ"]]) / MAX_ACCEL,
            [s["rpm"] / MAX_RPM * 2.0 - 1.0],
            [s["angle"] / np.pi],
            [s["trackPos"]],
            np.array(s["track"]) / MAX_TRACK * 2.0 - 1.0,
            np.array(s["wheelSpinVel"]) / MAX_WHEEL_SPIN_VEL * 2.0 - 1.0,
            [racing_line_offset],
            curvature / MAX_CURVATURE,
        ]).astype(np.float32)

        if self.observation_noise > 0.0:
            state += self.np_random.normal(0.0, self.observation_noise, size=state.shape).astype(np.float32)
        return np.clip(state, -1.0, 1.0)

    def _load_racing_line(self, track_name: str) -> RacingLineCurvatureMap | None:
        if track_name not in self._racing_lines:
            path = self.racing_line_dir / f"{track_name}.csv"
            if path.exists():
                self._racing_lines[track_name] = RacingLineCurvatureMap.from_csv(path)
            else:
                logger.warning("No racing line at %s, using curvature from TORCS.", path)
                self._racing_lines[track_name] = None
        return self._racing_lines[track_name]


def _racing_line_position(s: Sensors) -> float:
    # tita reports its line with the opposite sign convention to trackPos.
    return -s["titaLineTrackPos"]


def _progress_reward(s: Sensors) -> float:
    distance_to_line = abs(s["trackPos"] - _racing_line_position(s)) / 2.0
    return s["speedX"] / MAX_SPEED * (1.0 - distance_to_line)


def _expert_action(s: Sensors) -> np.ndarray:
    """The action the tita robot would take, in this env's action format."""
    return np.array([s["titaSteer"], s["titaGas"] - s["titaBrake"]], dtype=np.float32)


def _shift_gear(gear: int, rpm: float) -> int:
    if gear < 1:
        return 1
    if rpm > 8000.0 and gear < 6:
        return gear + 1
    if rpm < 3500.0 and gear > 1:
        return gear - 1
    return gear
