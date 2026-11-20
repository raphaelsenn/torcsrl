from __future__ import annotations

from pathlib import Path

DEFAULT_TORCS_EXECUTABLE = "/usr/local/bin/torcs"
RACING_LINE_DIR = Path(__file__).resolve().parent.parent / "data"

# BUG: TORCS crashes on e-track-1...

# How long TORCS waits for each action, in µs (the -t flag). The default of 10 ms is too short
# when the agent trains between steps.
CLIENT_TIMEOUT_US = 1_000_000

# The SCR server robot provides ten driver slots; slot i listens on port 3001 + i.
SCR_FIRST_PORT = 3001
SCR_NUM_SLOTS = 10

# Scales used to bring the raw sensors to roughly [-1, 1].
MAX_SPEED = 250.0           # km/h
MAX_ACCEL = 50.0            # m/s^2
MAX_ANGLE = 0.5235987755982988  # rad (pi/6); beyond this the car is crashing anyway
MAX_TRACK = 200.0           # m
MAX_RPM = 10_000.0
MAX_CURVATURE = 0.08        # 1/m (12.5 m radius); the tightest hairpin, on aalborg, is 0.078

# The car is off the track when |trackPos| exceeds this. trackPos is +-1 when the car's center
# is on the edge of the asphalt. tita treats a kerb on the inside of a corner as 1.5 m of extra
# track and cuts across it (berniw/trackdesc.cpp), e.g. reaching |trackPos| = 1.002 in the last
# hairpin of aalborg. The margin keeps the expert's own line legal.
# Between 1 and the limit the SCR range finders read -1, the value for "outside the track".
OFF_TRACK_LIMIT = 1.1

# An episode fails if the car is slower than MIN_SPEED after STALL_GRACE_STEPS.
STALL_GRACE_STEPS = 100
MIN_SPEED = 5.0             # km/h

# Reward when the episode fails by crashing, leaving the track or turning backwards.
FAILURE_REWARD = -100.0

# Reward when the episode fails by standing still. Braking to a stop in front of a corner takes
# about 150 steps, and with gamma = 0.997 a penalty 150 steps away is worth only
# 0.997**150 = 0.64 of it. With FAILURE_REWARD a stall would look less bad than a crash, and
# agents learn to stop in front of corners they cannot take yet (seen on aalborg).
# Twice FAILURE_REWARD keeps stopping worse than crashing for braking phases up to ~4.6 s.
STALL_REWARD = 2.0 * FAILURE_REWARD

# Distances ahead of the car (in m) at which the racing line curvature is sampled.
LOOKAHEAD_DISTANCES = tuple(range(0, 300, 25))

# Angles of the 19 track edge range finders, dense around the heading.
# See https://github.com/ugo-nama-kun/gym_torcs/blob/master/snakeoil3_gym.py
TRACK_SENSOR_ANGLES = (
    -45.0, -19.0, -12.0, -7.0, -4.0, -2.5, -1.7, -1.0, -0.5,
    0.0,
    0.5, 1.0, 1.7, 2.5, 4.0, 7.0, 12.0, 19.0, 45.0,
)

# Track to category mapping
TRACK_CATEGORIES = {
    **dict.fromkeys([
        "aalborg", "alpine-1", "alpine-2", "brondehach", "corkscrew", "e-track-1", "e-track-2",
        "e-track-3", "e-track-4", "e-track-6", "eroad", "forza", "g-track-1", "g-track-2",
        "g-track-3", "ole-road-1", "ruudskogen", "spring", "street-1", "wheel-1", "wheel-2",
    ], "road"),
    **dict.fromkeys([
        "a-speedway", "b-speedway", "c-speedway", "d-speedway", "e-speedway", "e-track-5",
        "f-speedway", "g-speedway", "michigan",
    ], "oval"),
    **dict.fromkeys([
        "dirt-1", "dirt-2", "dirt-3", "dirt-4", "dirt-5", "dirt-6", "mixed-1", "mixed-2",
    ], "dirt"),
}
