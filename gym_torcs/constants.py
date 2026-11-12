from __future__ import annotations

from pathlib import Path

DEFAULT_TORCS_EXECUTABLE = "/usr/local/bin/torcs"
RACING_LINE_DIR = Path(__file__).resolve().parent.parent / "data"

# The SCR server robot provides ten driver slots; slot i listens on port 3001 + i.
SCR_FIRST_PORT = 3001
SCR_NUM_SLOTS = 10

# Scales used to bring the raw sensors to roughly [-1, 1].
MAX_SPEED = 250.0           # km/h
MAX_ACCEL = 50.0            # m/s^2
MAX_TRACK = 200.0           # m
MAX_RPM = 10_000.0
MAX_WHEEL_SPIN_VEL = 100.0  # rad/s
MAX_CURVATURE = 0.1         # 1/m

# An episode fails if the car is slower than MIN_SPEED after STALL_GRACE_STEPS.
STALL_GRACE_STEPS = 100
MIN_SPEED = 5.0             # km/h
FAILURE_REWARD = -10.0

# Distances ahead of the car (in m) at which the racing line curvature is sampled.
LOOKAHEAD_DISTANCES = tuple(range(0, 300, 25))

# Angles of the 19 track edge range finders, dense around the heading.
# See https://github.com/ugo-nama-kun/gym_torcs/blob/master/snakeoil3_gym.py
TRACK_SENSOR_ANGLES = (
    -45.0, -19.0, -12.0, -7.0, -4.0, -2.5, -1.7, -1.0, -0.5,
    0.0,
    0.5, 1.0, 1.7, 2.5, 4.0, 7.0, 12.0, 19.0, 45.0,
)
