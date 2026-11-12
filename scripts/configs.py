"""Shared settings of the experiment in the README: tracks, env and hyperparameters."""

import gymnasium as gym

from gym_torcs import HistoryWrapper, TimedTrackSelectionWrapper, TorcsEnv, TrackSpec
from torcsrl import TD7, NStepDDPG, NStepSAC, NStepTD3

TRAIN_TRACKS = [TrackSpec(name) for name in ["alpine-1", "aalborg", "street-1", "forza", "wheel-1", "wheel-2"]]
VAL_TRACKS = [TrackSpec(name) for name in ["g-track-2", "brondehach", "alpine-2"]]

# The agent sees the last 3 observations and actions.
HISTORY_OBS = 3
HISTORY_ACT = 3


def make_env(tracks: list[TrackSpec], switch_every_steps: int, **env_kwargs) -> gym.Env:
    """TorcsEnv with history stacking that cycles through ``tracks``."""
    env = TorcsEnv(track=tracks[0], **env_kwargs)
    env = HistoryWrapper(env, HISTORY_OBS, HISTORY_ACT)
    return TimedTrackSelectionWrapper(env, tracks, switch_every_steps)


COMMON = {
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "gamma": 0.992,
    "buffer_size": 1_000_000,
    "buffer_start_size": 10_000,
    "batch_size": 256,
    "save_every": 10_000,
    "eval_every": 10_000,
}
TD_STYLE = {"exploration_noise": 0.1, "policy_delay": 2}
N_STEP = {"n_steps": 3, "tau_polyak": 0.005}

# CLI name -> (algorithm class, hyperparameters)
ALGORITHMS = {
    "ddpg": (NStepDDPG, {**COMMON, **TD_STYLE, **N_STEP, "ac_kwargs": {"h1_dim": 256, "h2_dim": 256}}),
    "td3": (NStepTD3, {**COMMON, **TD_STYLE, **N_STEP, "ac_kwargs": {"h1_dim": 256, "h2_dim": 256}}),
    "sac": (NStepSAC, {**COMMON, **N_STEP, "lr_alpha": 3e-4, "ac_kwargs": {"hidden_dim": 256}}),
    "td7": (TD7, {**COMMON, **TD_STYLE, "lr_encoder": 3e-4, "ac_kwargs": {"hidden_dim": 256, "zs_dim": 256}}),
}
