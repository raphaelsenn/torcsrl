"""Shared settings of the experiment in the README: tracks, env and hyperparameters."""

import gymnasium as gym

from gym_torcs import HistoryWrapper, TimedTrackSelectionWrapper, TorcsEnv, TrackSpec
from torcsrl import DDPG, SAC, TD3, TD7, NStepDDPG, NStepSAC, NStepTD3

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
ONE_STEP = {"tau_polyak": 0.005}
N_STEP = {"n_steps": 3, "tau_polyak": 0.005}
MLP = {"ac_kwargs": {"h1_dim": 256, "h2_dim": 256}}
SAC_MLP = {"lr_alpha": 3e-4, "ac_kwargs": {"hidden_dim": 256}}

# CLI name -> (algorithm class, hyperparameters)
# Keys equal the lowercased class name so evaluate.py can infer them from checkpoint names.
ALGORITHMS = {
    "ddpg": (DDPG, {**COMMON, **TD_STYLE, **ONE_STEP, **MLP}),
    "nstepddpg": (NStepDDPG, {**COMMON, **TD_STYLE, **N_STEP, **MLP}),
    "td3": (TD3, {**COMMON, **TD_STYLE, **ONE_STEP, **MLP}),
    "nsteptd3": (NStepTD3, {**COMMON, **TD_STYLE, **N_STEP, **MLP}),
    "sac": (SAC, {**COMMON, **ONE_STEP, **SAC_MLP}),
    "nstepsac": (NStepSAC, {**COMMON, **N_STEP, **SAC_MLP}),
    "td7": (TD7, {**COMMON, **TD_STYLE, "lr_encoder": 3e-4, "ac_kwargs": {"hidden_dim": 256, "zs_dim": 256}}),
}
