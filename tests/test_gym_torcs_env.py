"""TorcsEnv tests that replace the UDP client, so TORCS does not need to run."""

from pathlib import Path

import numpy as np
import pytest

from gym_torcs import TorcsEnv, TrackSpec
from gym_torcs.core.client import Action
from gym_torcs.constants import FAILURE_REWARD, MAX_CURVATURE, MAX_SPEED, RACING_LINE_DIR, STALL_REWARD
from gym_torcs.observation import STATE_SIZE
from gym_torcs.curvature import RacingLineCurvatureMap
from gym_torcs.rules import expert_action, progress_reward, racing_line_offset, shift_gear
from gym_torcs.core.vision import resize
from tests.conftest import make_sensors


class FakeClient:
    """Answers every action with the next prepared sensor message."""

    def __init__(self, *messages: dict) -> None:
        self.messages = list(messages)
        self.sent: list[Action] = []

    def send(self, action: Action) -> None:
        self.sent.append(action)

    def receive(self, timeout: float, keepalive=None) -> dict:
        if not self.messages:
            raise TimeoutError
        return self.messages.pop(0)

    def close(self) -> None:
        pass


@pytest.fixture
def env(tmp_path: Path):
    env = TorcsEnv(track=TrackSpec("test-track"), observation_noise=0.0, racing_line_dir=tmp_path)
    env._sensors = make_sensors()
    yield env
    env.close()


def drive(env: TorcsEnv, *messages: dict, action=(0.0, 1.0)):
    """Step ``env`` once with a client that answers with ``messages``."""
    env.client = FakeClient(*messages)
    return env.step(np.array(action, dtype=np.float32))


class TestTorcsEnvSpaces:
    def test_action_is_steer_and_throttle(self, env: TorcsEnv) -> None:
        assert env.action_space.shape == (2,)
        assert env.action_space.low.tolist() == [-1.0, -1.0]
        assert env.action_space.high.tolist() == [1.0, 1.0]

    def test_state_has_state_size_values_in_minus_one_to_one(self, env: TorcsEnv) -> None:
        state = env._state(make_sensors())
        assert state.shape == (STATE_SIZE,)
        assert env.observation_space.contains(state)

    def test_extreme_sensors_are_clipped(self, env: TorcsEnv) -> None:
        state = env._state(make_sensors(speedX=10_000.0, rpm=50_000.0))
        assert state.min() >= -1.0 and state.max() <= 1.0

    def test_rejects_unknown_render_mode(self) -> None:
        with pytest.raises(ValueError):
            TorcsEnv(render_mode="rgb_array")

    def test_vision_needs_the_torcs_window(self) -> None:
        with pytest.raises(ValueError):
            TorcsEnv(vision=True)


class TestTorcsEnvStep:
    def test_step_before_reset_raises(self, env: TorcsEnv) -> None:
        with pytest.raises(RuntimeError):
            env.step(np.zeros(3, dtype=np.float32))

    def test_driving_on_the_racing_line_rewards_speed(self, env: TorcsEnv) -> None:
        _, reward, terminated, truncated, _ = drive(env, make_sensors(speedX=125.0))

        assert reward == pytest.approx(125.0 / MAX_SPEED)
        assert not terminated and not truncated

    def test_positive_throttle_accelerates_and_negative_throttle_brakes(self, env: TorcsEnv) -> None:
        drive(env, make_sensors(), action=(0.2, 0.4))
        sent = env.client.sent[-1]
        assert (sent.steer, sent.accel, sent.brake) == pytest.approx((0.2, 0.4, 0.0))

        drive(env, make_sensors(), action=(0.0, -0.6))
        sent = env.client.sent[-1]
        assert (sent.accel, sent.brake) == pytest.approx((0.0, 0.6))

    def test_zero_throttle_coasts(self, env: TorcsEnv) -> None:
        drive(env, make_sensors(), action=(0.0, 0.0))
        sent = env.client.sent[-1]
        assert (sent.accel, sent.brake) == (0.0, 0.0)

    def test_actions_outside_the_space_are_clipped(self, env: TorcsEnv) -> None:
        drive(env, make_sensors(), action=(5.0, -7.0))
        sent = env.client.sent[-1]
        assert (sent.steer, sent.accel, sent.brake) == (1.0, 0.0, 1.0)

    @pytest.mark.parametrize(("failure", "reason"), [
        (dict(trackPos=1.5), "off_track"),
        (dict(damage=10.0), "crashed"),
        (dict(angle=np.pi), "backwards"),
    ])
    def test_failures_end_the_episode_with_the_failure_reward(self, env: TorcsEnv, failure: dict, reason: str) -> None:
        _, reward, terminated, truncated, info = drive(env, make_sensors(**failure))

        assert terminated and not truncated
        assert reward == FAILURE_REWARD
        assert info["end_reason"] == reason

    def test_cutting_a_kerb_like_tita_is_not_off_track(self, env: TorcsEnv) -> None:
        # tita reaches |trackPos| = 1.002 in aalborg's last hairpin; its range finders read -1 there.
        _, _, terminated, _, _ = drive(env, make_sensors(trackPos=-1.05, track=[-1.0] * 19))
        assert not terminated

    def test_standing_still_ends_the_episode_with_the_stall_reward(self, env: TorcsEnv) -> None:
        env._step_count = 1_000
        _, reward, terminated, _, info = drive(env, make_sensors(speedX=0.0))
        assert terminated
        assert reward == STALL_REWARD
        assert info["end_reason"] == "stalled"

    def test_braking_to_a_stop_is_worse_than_crashing(self) -> None:
        # Stopping takes ~150 steps; even discounted with gamma = 0.997 it must cost more than a crash.
        assert 0.997**150 * STALL_REWARD < FAILURE_REWARD

    def test_running_episode_has_no_end_reason(self, env: TorcsEnv) -> None:
        _, _, terminated, truncated, info = drive(env, make_sensors())
        assert not (terminated or truncated)
        assert info["end_reason"] is None

    def test_standing_still_never_beats_driving(self, env: TorcsEnv) -> None:
        # A stall must not be a safe refuge: every forward step earns at least the failure reward.
        assert progress_reward(make_sensors(speedX=1.0, trackPos=0.9)) >= FAILURE_REWARD
        assert progress_reward(make_sensors(speedX=0.0)) == 0.0

    def test_reward_does_not_depend_on_the_steering(self, env: TorcsEnv) -> None:
        _, straight, *_ = drive(env, make_sensors(), action=(0.0, 1.0))
        _, steered, *_ = drive(env, make_sensors(), action=(0.5, 1.0))

        assert steered == pytest.approx(straight)

    def test_waits_through_the_start_countdown(self, env: TorcsEnv) -> None:
        env.client = FakeClient(make_sensors(curLapTime=-0.5), make_sensors(curLapTime=0.02))
        sensors = env._drive_until(make_sensors(curLapTime=-0.98), lambda s: s["curLapTime"] >= 0.0, max_steps=10)

        assert sensors["curLapTime"] == pytest.approx(0.02)
        assert len(env.client.sent) == 2                       # idled through two countdown steps
        assert all(a.accel == 0.0 and a.brake == 0.0 for a in env.client.sent)

    def test_expert_drives_to_the_random_start(self, env: TorcsEnv) -> None:
        env.client = FakeClient(*(make_sensors(distRaced=d, titaSteer=0.2, titaGas=1.0) for d in (10.0, 20.0, 30.0)))
        sensors = env._drive_expert_to(make_sensors(distRaced=0.0, titaSteer=0.2, titaGas=1.0), distance=25.0)

        assert sensors["distRaced"] == 30.0
        assert len(env.client.sent) == 3
        assert all((a.steer, a.accel, a.brake) == pytest.approx((0.2, 1.0, 0.0)) for a in env.client.sent)

    def test_expert_keeps_driving_until_the_car_is_inside_the_track(self, env: TorcsEnv) -> None:
        env.client = FakeClient(make_sensors(distRaced=40.0, trackPos=0.95), make_sensors(distRaced=41.0, trackPos=0.5))
        sensors = env._drive_expert_to(make_sensors(distRaced=30.0, trackPos=0.95), distance=25.0)
        assert sensors["trackPos"] == 0.5

    def test_random_start_must_be_a_probability(self) -> None:
        with pytest.raises(ValueError):
            TorcsEnv(random_start=1.5)

    def test_standing_still_at_the_start_is_allowed(self, env: TorcsEnv) -> None:
        _, _, terminated, _, _ = drive(env, make_sensors(speedX=0.0))
        assert not terminated

    def test_completing_a_lap_truncates_the_episode(self, env: TorcsEnv) -> None:
        _, _, terminated, truncated, info = drive(env, make_sensors(lastLapTime=95.3))

        assert truncated and not terminated
        assert info["lap_completed"]
        assert info["laps_completed"] == 1
        assert info["lap_time"] == pytest.approx(95.3)
        assert info["end_reason"] == "lap_completed"

    def test_a_lap_is_counted_once(self, env: TorcsEnv) -> None:
        env.truncate_on_lap = False
        drive(env, make_sensors(lastLapTime=95.3))
        _, _, _, truncated, info = drive(env, make_sensors(lastLapTime=95.3))

        assert not truncated
        assert not info["lap_completed"]
        assert info["laps_completed"] == 1

    def test_reaching_max_steps_truncates_the_episode(self, env: TorcsEnv) -> None:
        env._step_count = env.max_steps - 1
        _, _, _, truncated, info = drive(env, make_sensors())
        assert truncated
        assert info["end_reason"] == "max_steps"

    def test_torcs_timeout_truncates_and_disconnects(self, env: TorcsEnv) -> None:
        _, reward, terminated, truncated, info = drive(env)      # client has no messages

        assert truncated and not terminated
        assert reward == 0.0
        assert info["end_reason"] == "torcs_timeout"
        assert info["distance_raced"] == make_sensors()["distRaced"]   # last known state
        assert env.client is None

    def test_end_of_the_torcs_race_truncates_and_disconnects(self, env: TorcsEnv) -> None:
        class FinishedRace(FakeClient):
            def receive(self, timeout: float, keepalive=None) -> dict:
                raise ConnectionError("***shutdown***")

        env.client = FinishedRace()
        _, reward, terminated, truncated, info = env.step(np.array([0.0, 1.0], dtype=np.float32))

        assert truncated and not terminated
        assert reward == 0.0
        assert info["end_reason"] == "race_over"
        assert env.client is None

    def test_info_contains_the_expert_action(self, env: TorcsEnv) -> None:
        _, _, _, _, info = drive(env, make_sensors(titaSteer=-0.3, titaGas=0.2, titaBrake=0.5))
        assert info["expert_action"] == pytest.approx([-0.3, -0.3])


class TestTorcsEnvHelpers:
    def test_reward_shrinks_with_distance_to_the_racing_line(self) -> None:
        on_line = progress_reward(make_sensors(trackPos=0.0))
        off_line = progress_reward(make_sensors(trackPos=0.5))
        assert off_line == pytest.approx(on_line * 0.75)

    def test_reward_shrinks_when_pointing_away_from_the_track_axis(self) -> None:
        straight = progress_reward(make_sensors(angle=0.0))
        angled = progress_reward(make_sensors(angle=0.3))
        assert angled == pytest.approx(straight * (np.cos(0.3) - np.sin(0.3)))
        assert progress_reward(make_sensors(angle=-0.3)) == pytest.approx(angled)

    def test_racing_line_offset_is_signed_and_in_track_widths(self) -> None:
        # titaLineTrackPos is in trackPos units (patched scr_server, scrTrackPos).
        assert racing_line_offset(make_sensors(trackPos=0.5, titaLineTrackPos=-0.5)) == pytest.approx(0.5)
        assert racing_line_offset(make_sensors(trackPos=-0.5, titaLineTrackPos=0.5)) == pytest.approx(-0.5)
        assert racing_line_offset(make_sensors(trackPos=0.3, titaLineTrackPos=0.3)) == pytest.approx(0.0)

    @pytest.mark.parametrize("csv", sorted(RACING_LINE_DIR.glob("*.csv")), ids=lambda p: p.stem)
    def test_recorded_curvature_fits_the_observation_scale(self, csv: Path) -> None:
        curvature = RacingLineCurvatureMap.from_csv(csv).curvature
        assert np.abs(curvature).max() <= MAX_CURVATURE

    def test_expert_action_combines_gas_and_brake_into_throttle(self) -> None:
        action = expert_action(make_sensors(titaSteer=0.1, titaGas=0.0, titaBrake=1.0))
        assert action == pytest.approx([0.1, -1.0])

    @pytest.mark.parametrize("gear, rpm, expected", [
        (0, 1000.0, 1),         # neutral -> first gear
        (-1, 1000.0, 1),        # reverse -> first gear
        (3, 9000.0, 4),         # high rpm -> shift up
        (6, 9000.0, 6),         # already in top gear
        (3, 3000.0, 2),         # low rpm -> shift down
        (1, 3000.0, 1),         # already in first gear
        (3, 5000.0, 3),         # normal rpm -> keep gear
    ])
    def test_shift_gear(self, gear: int, rpm: float, expected: int) -> None:
        assert shift_gear(gear, rpm) == expected


class TestResize:
    def test_output_has_the_requested_shape(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        assert resize(image, 64, 64).shape == (64, 64, 3)

    def test_constant_image_stays_constant(self) -> None:
        image = np.full((480, 640, 3), 200, dtype=np.uint8)
        assert np.all(resize(image, 48, 64) == 200)

    def test_pixels_are_averaged_over_their_block(self) -> None:
        image = np.array([[0, 100], [100, 200]], dtype=np.uint8)[..., None]
        assert resize(image, 1, 1)[0, 0, 0] == 100
