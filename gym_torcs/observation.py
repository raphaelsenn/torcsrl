"""The state vector the agent sees.

====  ===================================================================
0-1   speed along the car's x, y axes
2-3   acceleration along the car's x, y axes
4     engine rpm
5     angle between the car and the track axis (clipped at ±30°)
6     lateral position on the track, -1 is the right edge, 1 the left
7-25  distance to the track edge along TRACK_SENSOR_ANGLES
26    offset from tita's racing line
27-38 racing line curvature 0, 25, ..., 275 m ahead
====  ===================================================================

Values are scaled to about [-1, 1] with the constants in constants.py.
The env adds noise and clips the result to [-1, 1].
"""

from __future__ import annotations

import numpy as np

from gym_torcs.constants import MAX_ACCEL, MAX_ANGLE, MAX_CURVATURE, MAX_RPM, MAX_SPEED, MAX_TRACK
from gym_torcs.core.client import Sensors
from gym_torcs.rules import racing_line_offset

STATE_SIZE = 39


def state_vector(s: Sensors, curvature: np.ndarray) -> np.ndarray:
    """State without noise or clipping. curvature holds the values at LOOKAHEAD_DISTANCES."""
    return np.concatenate([
        np.array([s["speedX"], s["speedY"]]) / MAX_SPEED,
        np.array([s["accelX"], s["accelY"]]) / MAX_ACCEL,
        [s["rpm"] / MAX_RPM * 2.0 - 1.0],
        [s["angle"] / MAX_ANGLE],
        [s["trackPos"]],
        np.array(s["track"]) / MAX_TRACK * 2.0 - 1.0,
        [racing_line_offset(s)],
        np.asarray(curvature) / MAX_CURVATURE,
    ]).astype(np.float32)
