"""TorcsEnv tests that replace the UDP client, so TORCS does not need to run."""

from pathlib import Path

import numpy as np
import pytest

from gym_torcs import TorcsEnv, TrackSpec
from gym_torcs.client import Action
from gym_torcs.constants import FAILURE_REWARD, MAX_SPEED
from gym_torcs.env import STATE_SIZE, _expert_action, _progress_reward, _shift_gear
from gym_torcs.vision import resize
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

    def test_state_has_45_values_in_minus_one_to_one(self, env: TorcsEnv) -> None:
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
            env.step(np.zeros(2, dtype=np.float32))

    def test_driving_on_the_racing_line_rewards_speed(self, env: TorcsEnv) -> None:
        _, reward, terminated, truncated, _ = drive(env, make_sensors(speedX=125.0))

        assert reward == pytest.approx(125.0 / MAX_SPEED)
        assert not terminated and not truncated

    def test_positive_throttle_accelerates_and_negative_throttle_brakes(self, env: TorcsEnv) -> None:
        drive(env, make_sensors(), action=(0.2, 0.7))
        sent = env.client.sent[-1]
        assert (sent.steer, sent.accel, sent.brake) == pytest.approx((0.2, 0.7, 0.0))

        drive(env, make_sensors(), action=(0.0, -0.4))
        sent = env.client.sent[-1]
        assert (sent.accel, sent.brake) == pytest.approx((0.0, 0.4))

    def test_actions_outside_the_space_are_clipped(self, env: TorcsEnv) -> None:
        drive(env, make_sensors(), action=(5.0, 3.0))
        sent = env.client.sent[-1]
        assert (sent.steer, sent.accel) == (1.0, 1.0)

    @pytest.mark.parametrize("failure", [
        dict(trackPos=1.5),                 # off track
        dict(damage=10.0),                  # crash
        dict(angle=np.pi),                  # driving backwards
    ])
    def test_failures_end_the_episode_with_a_penalty(self, env: TorcsEnv, failure: dict) -> None:
        _, reward, terminated, _, _ = drive(env, make_sensors(**failure))

        assert terminated
        assert reward == FAILURE_REWARD

    def test_standing_still_ends_the_episode_after_the_grace_period(self, env: TorcsEnv) -> None:
        env._step_count = 1_000
        _, _, terminated, _, _ = drive(env, make_sensors(speedX=0.0))
        assert terminated

    def test_standing_still_at_the_start_is_allowed(self, env: TorcsEnv) -> None:
        _, _, terminated, _, _ = drive(env, make_sensors(speedX=0.0))
        assert not terminated

    def test_completing_a_lap_truncates_the_episode(self, env: TorcsEnv) -> None:
        _, _, terminated, truncated, info = drive(env, make_sensors(lastLapTime=95.3))

        assert truncated and not terminated
        assert info["successfulLap"]

    def test_reaching_max_steps_truncates_the_episode(self, env: TorcsEnv) -> None:
        env._step_count = env.max_steps - 1
        _, _, _, truncated, _ = drive(env, make_sensors())
        assert truncated

    def test_torcs_timeout_truncates_and_disconnects(self, env: TorcsEnv) -> None:
        _, reward, terminated, truncated, info = drive(env)      # client has no messages

        assert truncated and not terminated
        assert reward == 0.0
        assert info["torcs_timeout"]
        assert env.client is None

    def test_info_contains_the_expert_action(self, env: TorcsEnv) -> None:
        _, _, _, _, info = drive(env, make_sensors(titaSteer=-0.3, titaGas=0.2, titaBrake=0.5))
        assert info["action_tita"] == pytest.approx([-0.3, -0.3])


class TestTorcsEnvHelpers:
    def test_reward_shrinks_with_distance_to_the_racing_line(self) -> None:
        on_line = _progress_reward(make_sensors(trackPos=0.0))
        off_line = _progress_reward(make_sensors(trackPos=0.5))
        assert off_line == pytest.approx(on_line * 0.75)

    def test_expert_action_combines_gas_and_brake(self) -> None:
        action = _expert_action(make_sensors(titaSteer=0.1, titaGas=0.0, titaBrake=1.0))
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
        assert _shift_gear(gear, rpm) == expected


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
