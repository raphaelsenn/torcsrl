"""Shared settings of the experiment in the README."""

import gymnasium as gym

from gym_torcs import ActionRepeatWrapper, HistoryWrapper, TimedTrackSelectionWrapper, TorcsEnv, TrackSpec
from torcsrl import DDPG, SAC, TD3, TD7, NStepDDPG, NStepSAC, NStepTD3


# Training tracks ordered by difficulty (NOTE: difficulty = my own experience playing TORCS :D)
# 12 training tracks
TRAIN_TRACKS = [TrackSpec(name) for name in [
    "e-track-4", "forza", "g-track-1", "wheel-1", "eroad", "g-track-3",
    "wheel-2", "corkscrew", "street-1", "e-track-2", "alpine-1", "aalborg",
]]

# Validation tracks ordered by difficulty
# 6 validation tracks
VAL_TRACKS = [TrackSpec(name) for name in [
    "michigan", "g-track-2", "e-track-6", "ruudskogen", "brondehach", "alpine-2",
]]

# Repeat actions for this many steps
# ACTION_REPEAT = 2 reduces TORCS at 50Hz to 25Hz
ACTION_REPEAT = 2

# NOTE: TORCS runs now at 25 Hz (ACTION_REPEAT = 2)
# A history helps ensure the markov property.
# For automonous driving/racing read this:
# https://medium.com/data-science/learning-to-drive-smoothly-in-minutes-450a7cdb35f4
HISTORY_OBS = 3
HISTORY_ACT = 3

# Agent drives one track at a time and switches to the next every this many steps
# I did a hyperparameter sweep over this parameter (25k steps works really good)
# 50k or 62.5k steps may also work good
TRAIN_SWITCH_EVERY_STEPS = 25_000

# Probability that the agent starts somewhere on the lap instead of the start position
# NOTE: This parameter really speeds training time!!!
# NOTE: Validation always starts on the grid!
TRAIN_RANDOM_START = 0.5

# Algorithm hyperparameters
COMMON = {
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "gamma": 0.994,                 # Horizon = 1 / (1 - gamma) = ~167 steps at 25 Hz = ~6.67s
    "buffer_size": 5_000_000,
    "buffer_start_size": 25_000,    # Warmup steps on training track
    "warmup": "expert_noisy",       # Warmup policy; "random", "expert" (tita) or "expert_noisy" (tita + gaussian noise)
    "batch_size": 256,
    "save_every": 10_000,
    "eval_every": 10_000,
}

TD_STYLE = {"exploration_noise": 0.1, "policy_delay": 2}
ONE_STEP = {"tau_polyak": 0.005}
N_STEP = {"n_steps": 3, "tau_polyak": 0.005}
MLP = {"ac_kwargs": {"h1_dim": 512, "h2_dim": 512}}
SAC_MLP = {"lr_alpha": 3e-4, "ac_kwargs": {"h1_dim": 512, "h2_dim": 512}}

ALGORITHMS = {
    "ddpg": (DDPG, {**COMMON, **TD_STYLE, **ONE_STEP, **MLP}),
    "nstepddpg": (NStepDDPG, {**COMMON, **TD_STYLE, **N_STEP, **MLP}),
    "td3": (TD3, {**COMMON, **TD_STYLE, **ONE_STEP, **MLP}),
    "nsteptd3": (NStepTD3, {**COMMON, **TD_STYLE, **N_STEP, **MLP}),
    "sac": (SAC, {**COMMON, **ONE_STEP, **SAC_MLP}),
    "nstepsac": (NStepSAC, {**COMMON, **N_STEP, **SAC_MLP}),
    "td7": (TD7, {**COMMON, **TD_STYLE, "lr_encoder": 3e-4, "ac_kwargs": {"hidden_dim": 256, "zs_dim": 256}}),
}


def wrap_env(env: gym.Env) -> gym.Env:
    """Action repeat, then history stacking of the last observations and actions."""
    return HistoryWrapper(ActionRepeatWrapper(env, ACTION_REPEAT), HISTORY_OBS, HISTORY_ACT)


def make_train_env(tracks: list[TrackSpec], **env_kwargs) -> gym.Env:
    """Wrapped TorcsEnv that cycles through `tracks`, switching every `TRAIN_SWITCH_EVERY_STEPS` steps."""
    env = wrap_env(TorcsEnv(track=tracks[0], random_start=TRAIN_RANDOM_START, **env_kwargs))
    return TimedTrackSelectionWrapper(env, tracks, switch_every_steps=TRAIN_SWITCH_EVERY_STEPS)


def make_val_env(tracks: list[TrackSpec], **env_kwargs) -> gym.Env:
    """Wrapped TorcsEnv that cycles through `tracks`, one episode per track."""
    env = wrap_env(TorcsEnv(track=tracks[0], **env_kwargs))
    return TimedTrackSelectionWrapper(env, tracks, switch_every_steps=1)
