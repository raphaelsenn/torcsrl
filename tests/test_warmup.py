"""The warm-up of every off-policy algorithm, on the fake env."""

import numpy as np
import pytest

from gym_torcs import TimedTrackSelectionWrapper, TrackSpec
from torcsrl import DDPG, SAC, TD3, TD7, NStepDDPG, NStepSAC, NStepTD3
from tests.conftest import ACTION_DIM, FakeEnv

TRACKS = [TrackSpec("alpine-1"), TrackSpec("forza"), TrackSpec("aalborg")]
MLP = {"ac_kwargs": {"h1_dim": 16, "h2_dim": 16}}
ALGORITHMS = [
    (DDPG, MLP), (TD3, MLP), (NStepDDPG, {**MLP, "n_steps": 3}), (NStepTD3, {**MLP, "n_steps": 3}),
    (SAC, {"ac_kwargs": {"hidden_dim": 16}}), (NStepSAC, {"ac_kwargs": {"hidden_dim": 16}, "n_steps": 3}),
    (TD7, {"ac_kwargs": {"hidden_dim": 16, "zs_dim": 16}}),
]


class ExpertEnv(FakeEnv):
    """FakeEnv with a fixed expert action. Remembers the track of each reset."""

    EXPERT = np.full(ACTION_DIM, 0.5, dtype=np.float32)

    def __init__(self) -> None:
        super().__init__()
        self.reset_tracks: list[str | None] = []
        self.actions: list[np.ndarray] = []

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.reset_tracks.append((options or {}).get("track", TrackSpec("-")).name)
        return obs, {**info, "expert_action": self.EXPERT}

    def step(self, action):
        self.actions.append(np.asarray(action))
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, {**info, "expert_action": self.EXPERT}


def make(algorithm_cls, kwargs, buffer_start_size=60, warmup="expert"):
    inner = ExpertEnv()
    env = TimedTrackSelectionWrapper(inner, TRACKS, switch_every_steps=1_000)
    agent = algorithm_cls(env, FakeEnv(), buffer_start_size=buffer_start_size, batch_size=8, warmup=warmup,
                          buffer_size=1_000, verbose=False, lr_actor=1e-3, lr_critic=1e-3, **kwargs)
    return agent, env, inner


@pytest.mark.parametrize(("algorithm_cls", "kwargs"), ALGORITHMS, ids=lambda a: getattr(a, "__name__", ""))
def test_warmup_drives_the_expert_on_every_training_track(algorithm_cls, kwargs) -> None:
    agent, env, inner = make(algorithm_cls, kwargs)
    warmup = agent.collect_n_step_rollouts if hasattr(agent, "n_steps") else agent.collect_rollouts
    obs, info, *_ = warmup()

    # 60 steps on 3 tracks, 20-step episodes: one episode per track.
    assert all(np.array_equal(a, ExpertEnv.EXPERT) for a in inner.actions)
    assert len(inner.actions) == 60
    assert inner.reset_tracks[:3] == ["alpine-1", "forza", "aalborg"]
    assert len(agent.replay_buffer) == 60
    assert np.allclose(agent.replay_buffer.actions[:60], ExpertEnv.EXPERT)

    # Training starts on the first track. The wrapper did not count the warm-up.
    assert inner.reset_tracks[-1] == "alpine-1"
    assert env._steps_on_track == 0


@pytest.mark.parametrize(("algorithm_cls", "kwargs"), [a for a in ALGORITHMS if "n_steps" in a[1]],
                         ids=lambda a: getattr(a, "__name__", ""))
def test_n_step_queues_do_not_span_the_cut_between_tracks(algorithm_cls, kwargs) -> None:
    # 30 steps per track: a full 20-step episode, then 10 steps of a cut-off one.
    agent, _, _ = make(algorithm_cls, kwargs, buffer_start_size=90)
    _, _, (s_queue, a_queue, r_queue) = agent.collect_n_step_rollouts()

    assert len(agent.replay_buffer) == 90            # every step stored exactly once
    assert len(s_queue) == 1 and not a_queue and not r_queue


@pytest.mark.parametrize(("algorithm_cls", "kwargs"), ALGORITHMS, ids=lambda a: getattr(a, "__name__", ""))
def test_random_warmup_samples_the_action_space_on_every_track(algorithm_cls, kwargs) -> None:
    agent, env, inner = make(algorithm_cls, kwargs, warmup="random")
    warmup = agent.collect_n_step_rollouts if hasattr(agent, "n_steps") else agent.collect_rollouts
    warmup()

    actions = np.array(inner.actions)
    assert len(actions) == 60
    assert not np.any(np.all(actions == ExpertEnv.EXPERT, axis=1))
    assert np.all(np.abs(actions) <= 1.0) and actions.std() > 0.3
    assert inner.reset_tracks[:3] == ["alpine-1", "forza", "aalborg"]
    assert len(agent.replay_buffer) == 60


def test_unknown_warmup_is_rejected() -> None:
    with pytest.raises(ValueError):
        make(SAC, {"ac_kwargs": {"hidden_dim": 16}}, warmup="tita")


class LoggingEnv(FakeEnv):
    """FakeEnv that terminates every other episode after 3 steps and logs every step.

    Observations count up (obs[0] is a global counter), so each one can be found in the buffer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.counter = 0
        self.episodes: list[dict] = []    # one per reset: first obs, then the steps

    def _obs(self) -> np.ndarray:
        self.counter += 1
        return np.full(self.observation_space.shape, self.counter, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.episodes.append({"obs": [obs], "rewards": [], "terminated": False})
        return obs, info

    def step(self, action):
        obs, _, _, truncated, info = super().step(action)
        reward = float(self.counter)
        terminated = self.resets % 2 == 0 and self.steps == 3
        episode = self.episodes[-1]
        episode["obs"].append(obs)
        episode["rewards"].append(reward)
        episode["terminated"] = terminated
        return obs, reward, terminated, truncated, info


def expected_n_step_transitions(episodes, n, gamma):
    """(obs, return, obs_next, done, steps) of every n-step transition, computed by hand."""
    expected = []
    for episode in episodes:
        T = len(episode["rewards"])
        for t in range(T):
            k = min(n, T - t)
            ret = sum(gamma**i * episode["rewards"][t + i] for i in range(k))
            done = episode["terminated"] and t + k == T
            expected.append((episode["obs"][t][0], ret, episode["obs"][t + k][0], float(done), float(k)))
    return expected


@pytest.mark.parametrize(("algorithm_cls", "kwargs"), [a for a in ALGORITHMS if a[0].__name__.startswith("NStep")],
                         ids=lambda a: getattr(a, "__name__", ""))
def test_n_step_warmup_stores_correct_returns(algorithm_cls, kwargs) -> None:
    # 3 tracks x 25 steps: terminated (3), truncated (20) and budget-cut episodes all occur.
    inner = LoggingEnv()
    env = TimedTrackSelectionWrapper(inner, TRACKS, switch_every_steps=1_000)
    agent = algorithm_cls(env, FakeEnv(), buffer_start_size=75, batch_size=8, buffer_size=1_000, gamma=0.9,
                          verbose=False, lr_actor=1e-3, lr_critic=1e-3, **kwargs)
    obs, _, (s_queue, a_queue, r_queue) = agent.collect_n_step_rollouts()

    b = agent.replay_buffer
    stored = sorted(zip(b.obs[:b.size, 0], b.rewards[:b.size], b.obs_next[:b.size, 0], b.dones[:b.size], b.steps[:b.size]))
    expected = sorted(expected_n_step_transitions(inner.episodes[:-1], agent.n_steps, agent.gamma))

    assert {e["terminated"] for e in inner.episodes[:-1]} == {True, False}
    assert len(stored) == len(expected) == 75                       # every step starts one transition
    np.testing.assert_allclose(np.array(stored), np.array(expected), rtol=1e-5)
    # Training continues from a fresh episode with an empty queue.
    assert list(s_queue) == [obs] and not a_queue and not r_queue
