from gymnasium.envs.registration import register

from gym_torcs.core.server import TrackSpec
from gym_torcs.env import TorcsEnv
from gym_torcs.wrappers import ActionRepeatWrapper, HistoryWrapper, TimedTrackSelectionWrapper

register(id="TorcsSCR-v0", entry_point="gym_torcs.env:TorcsEnv")

__all__ = ["ActionRepeatWrapper", "HistoryWrapper", "TimedTrackSelectionWrapper", "TorcsEnv", "TrackSpec"]
