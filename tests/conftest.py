"""Shared test helpers.

The RL algorithms only need a gymnasium env, so they are tested against a small
fake env instead of TORCS. The gym_torcs tests build fake sensor messages.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from gym_torcs.constants import LOOKAHEAD_DISTANCES

OBS_DIM = 6
ACTION_DIM = 2
EPISODE_LENGTH = 20


class FakeEnv(gym.Env):
    """Random observations, reward = -|action|, ends after ``episode_length`` steps."""

    def __init__(self, episode_length: int = EPISODE_LENGTH) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.episode_length = episode_length
        self.steps = 0
        self.resets = 0
        self.closed = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.steps = 0
        self.resets += 1
        info = {"action_tita": np.zeros(ACTION_DIM, dtype=np.float32), "options": options}
        return self._obs(), info

    def step(self, action):
        self.steps += 1
        reward = -float(np.abs(action).sum())
        truncated = self.steps >= self.episode_length
        info = {"action_tita": np.zeros(ACTION_DIM, dtype=np.float32), "speedX": 10.0}
        return self._obs(), reward, False, truncated, info

    def close(self) -> None:
        self.closed = True

    def _obs(self) -> np.ndarray:
        return self.np_random.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)


@pytest.fixture
def train_env() -> FakeEnv:
    return FakeEnv()


@pytest.fixture
def val_env() -> FakeEnv:
    return FakeEnv()


# Small settings so every algorithm trains in well under a second.
# eval_every/save_every are huge so no tests write checkpoints to disk.
FAST_SETTINGS = dict(
    buffer_size=1_000,
    buffer_start_size=64,
    batch_size=16,
    save_every=10**9,
    eval_every=10**9,
    verbose=False,
    seed=0,
    device="cpu",
)


def obs_batch(batch_size: int = 4) -> torch.Tensor:
    return torch.rand(batch_size, OBS_DIM) * 2.0 - 1.0


def parameters_equal(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    return all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def parameters_snapshot(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def parameters_changed(module: torch.nn.Module, snapshot: list[torch.Tensor]) -> bool:
    return any(not torch.equal(p, q) for p, q in zip(module.parameters(), snapshot))


def make_sensors(**overrides: Any) -> dict[str, Any]:
    """A sensor message of a car driving straight in the middle of the track."""
    sensors: dict[str, Any] = {
        "angle": 0.0,
        "trackPos": 0.0,
        "track": [10.0] * 19,
        "speedX": 100.0, "speedY": 0.0, "speedZ": 0.0,
        "accelX": 0.0, "accelY": 0.0, "accelZ": 0.0,
        "rpm": 5000.0,
        "gear": 3.0,
        "wheelSpinVel": [50.0] * 4,
        "damage": 0.0,
        "distFromStart": 100.0,
        "distRaced": 100.0,
        "curLapTime": 10.0,
        "lastLapTime": 0.0,
        "x": 0.0, "y": 0.0,
        "titaLineTrackPos": 0.0,
        "titaSteer": 0.1, "titaGas": 0.8, "titaBrake": 0.0,
    }
    sensors.update({f"curv{d}": 0.0 for d in LOOKAHEAD_DISTANCES})
    sensors.update(overrides)
    return sensors
