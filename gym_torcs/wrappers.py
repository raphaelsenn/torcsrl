from __future__ import annotations

from typing import Any, Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_torcs.core.server import TrackSpec


class TimedTrackSelectionWrapper(gym.Wrapper):
    """Cycle through ``tracks``, moving on after ``switch_every_steps`` steps.

    A running episode is never interrupted; the switch happens on the next reset.
    """

    def __init__(self, env: gym.Env, tracks: Iterable[TrackSpec], switch_every_steps: int) -> None:
        super().__init__(env)
        self.tracks = tuple(tracks)
        if not self.tracks:
            raise ValueError("At least one track is required.")
        if switch_every_steps <= 0:
            raise ValueError("switch_every_steps must be positive.")

        self.switch_every_steps = switch_every_steps
        self.track_idx = -1
        self._steps_on_track = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        options = dict(options or {})
        if self.track_idx < 0 or self._steps_on_track >= self.switch_every_steps:
            self.track_idx = (self.track_idx + 1) % len(self.tracks)
            self._steps_on_track = 0
            options["track"] = self.tracks[self.track_idx]
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        self._steps_on_track += 1
        return self.env.step(action)


class HistoryWrapper(gym.Wrapper):
    """Observe the last ``horizon_obs`` observations and ``horizon_act`` actions, oldest first.

    Adapted from the HistoryWrapper of rl-baselines3-zoo.
    """

    def __init__(self, env: gym.Env, horizon_obs: int = 3, horizon_act: int = 2) -> None:
        super().__init__(env)
        obs_space, act_space = env.observation_space, env.action_space
        if not isinstance(obs_space, spaces.Box) or not isinstance(act_space, spaces.Box):
            raise TypeError("HistoryWrapper needs Box observation and action spaces.")

        self.observation_space = spaces.Box(
            low=np.concatenate([np.tile(obs_space.low, horizon_obs), np.tile(act_space.low, horizon_act)]),
            high=np.concatenate([np.tile(obs_space.high, horizon_obs), np.tile(act_space.high, horizon_act)]),
            dtype=np.float32,
        )
        self._observations = np.zeros((horizon_obs, *obs_space.shape), dtype=np.float32)
        self._actions = np.zeros((horizon_act, *act_space.shape), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        # Fill the history with the first observation. Zeros would look like a parked car,
        # which is wrong when the episode starts at speed.
        self._observations[:] = obs
        self._actions[:] = 0.0
        return self._history(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._observations = np.roll(self._observations, -1, axis=0)
        self._observations[-1] = obs
        self._actions = np.roll(self._actions, -1, axis=0)
        self._actions[-1] = action
        return self._history(), reward, terminated, truncated, info

    def _history(self) -> np.ndarray:
        return np.concatenate([self._observations.ravel(), self._actions.ravel()])
