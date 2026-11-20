"""Reward, failures, laps, the tita expert and gear shifting."""

from __future__ import annotations

import numpy as np

from gym_torcs.constants import MAX_SPEED, MIN_SPEED, OFF_TRACK_LIMIT, STALL_GRACE_STEPS
from gym_torcs.core.client import Action, Sensors


def racing_line_offset(s: Sensors) -> float:
    """Signed distance from tita's racing line, in track widths (trackPos spans 2 per width)."""
    return (s["trackPos"] - s["titaLineTrackPos"]) / 2.0


def progress_reward(s: Sensors) -> float:
    """
    Compute reward:
        reward = cos(angle) - |sin(angle)| is from Lillicrap et al. (2015). It pays less when the car

    Taken from:
    Formula RL: Deep Reinforcement Learning for Autonomous Racing using Telemetry Data, Remonda et al., 2022
    https://arxiv.org/abs/2104.11106 
    """
    heading = np.cos(s["angle"]) - abs(np.sin(s["angle"]))
    return float(s["speedX"] / MAX_SPEED * heading * (1.0 - abs(racing_line_offset(s))))


def failure(s: Sensors, prev: Sensors, steps: int) -> str | None:
    """Why the episode failed in this step, or None."""
    if s["damage"] > prev["damage"]:
        return "crashed"
    if abs(s["trackPos"]) > OFF_TRACK_LIMIT:
        return "off_track"
    if np.cos(s["angle"]) < 0.0:
        return "backwards"
    if steps > STALL_GRACE_STEPS and s["speedX"] < MIN_SPEED:
        return "stalled"
    return None


def lap_completed(s: Sensors, prev: Sensors) -> bool:
    # lastLapTime stays the same until the next lap is done, so look for a change.
    return s["lastLapTime"] > 0.0 and s["lastLapTime"] != prev["lastLapTime"]


def expert_action(s: Sensors) -> np.ndarray:
    """What tita would do, as [steer, throttle]."""
    return np.array([s["titaSteer"], s["titaGas"] - s["titaBrake"]], dtype=np.float32)


def command(steer: float, throttle: float, prev: Sensors) -> Action:
    """Turn [steer, throttle] into an SCR command."""
    return Action(
        steer=steer,
        accel=max(throttle, 0.0),
        brake=max(-throttle, 0.0),
        gear=shift_gear(int(prev["gear"]), prev["rpm"]),
    )


def shift_gear(gear: int, rpm: float) -> int:
    if gear < 1:
        return 1
    if rpm > 8000.0 and gear < 6:
        return gear + 1
    if rpm < 3500.0 and gear > 1:
        return gear - 1
    return gear
