import numpy as np
import pytest

from gym_torcs import HistoryWrapper, TimedTrackSelectionWrapper, TrackSpec
from tests.conftest import ACTION_DIM, OBS_DIM, FakeEnv

TRACKS = [TrackSpec("alpine-1"), TrackSpec("forza"), TrackSpec("aalborg")]


class TestHistoryWrapper:
    def test_observation_space_holds_all_observations_and_actions(self) -> None:
        env = HistoryWrapper(FakeEnv(), horizon_obs=3, horizon_act=2)
        assert env.observation_space.shape == (3 * OBS_DIM + 2 * ACTION_DIM,)

    def test_reset_repeats_the_first_observation_and_zeros_the_actions(self) -> None:
        env = HistoryWrapper(FakeEnv(), horizon_obs=3, horizon_act=2)
        history, _ = env.reset(seed=0)

        observations = history[: 3 * OBS_DIM].reshape(3, OBS_DIM)
        actions = history[3 * OBS_DIM:]
        assert np.any(observations[2] != 0.0)
        assert np.all(observations == observations[2])
        assert np.all(actions == 0.0)

    def test_step_shifts_the_history_oldest_first(self) -> None:
        env = HistoryWrapper(FakeEnv(), horizon_obs=2, horizon_act=2)
        history, _ = env.reset(seed=0)
        first_obs = history[OBS_DIM: 2 * OBS_DIM].copy()

        action = np.array([0.5, -0.5, 0.25], dtype=np.float32)
        history, *_ = env.step(action)

        assert np.array_equal(history[:OBS_DIM], first_obs)         # moved one slot back
        assert np.array_equal(history[-ACTION_DIM:], action)        # newest action last
        assert np.all(history[-2 * ACTION_DIM:-ACTION_DIM] == 0.0)

    def test_history_is_inside_the_observation_space(self) -> None:
        env = HistoryWrapper(FakeEnv())
        history, _ = env.reset(seed=0)
        for _ in range(5):
            history, *_ = env.step(env.action_space.sample())
            assert env.observation_space.contains(history.astype(np.float32))


class TestTimedTrackSelectionWrapper:
    def test_first_reset_selects_the_first_track(self) -> None:
        env = TimedTrackSelectionWrapper(FakeEnv(), TRACKS, switch_every_steps=10)
        _, info = env.reset()
        assert info["options"]["track"] == TRACKS[0]

    def test_keeps_the_track_until_enough_steps_were_driven(self) -> None:
        env = TimedTrackSelectionWrapper(FakeEnv(), TRACKS, switch_every_steps=10)
        env.reset()
        for _ in range(9):
            env.step(np.zeros(ACTION_DIM))

        _, info = env.reset()

        assert env.track_idx == 0
        assert "track" not in info["options"]

    def test_switches_to_the_next_track_after_enough_steps(self) -> None:
        env = TimedTrackSelectionWrapper(FakeEnv(), TRACKS, switch_every_steps=10)
        env.reset()
        for _ in range(10):
            env.step(np.zeros(ACTION_DIM))

        _, info = env.reset()

        assert info["options"]["track"] == TRACKS[1]

    def test_wraps_around_after_the_last_track(self) -> None:
        env = TimedTrackSelectionWrapper(FakeEnv(), TRACKS, switch_every_steps=1)
        selected = []
        for _ in range(4):
            _, info = env.reset()
            selected.append(info["options"]["track"])
            env.step(np.zeros(ACTION_DIM))

        assert selected == [*TRACKS, TRACKS[0]]

    def test_needs_at_least_one_track(self) -> None:
        with pytest.raises(ValueError):
            TimedTrackSelectionWrapper(FakeEnv(), [], switch_every_steps=10)

    def test_needs_a_positive_switch_interval(self) -> None:
        with pytest.raises(ValueError):
            TimedTrackSelectionWrapper(FakeEnv(), TRACKS, switch_every_steps=0)
