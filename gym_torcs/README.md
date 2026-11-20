# gym_torcs

Gymnasium environment for TORCS (The Open Car Racing Simulator).

## Usage

```python
import gymnasium as gym
import gym_torcs
from gym_torcs import HistoryWrapper, TrackSpec

env = gym.make("TorcsSCR-v0", port=3001, track=TrackSpec("forza", "road"))
env = HistoryWrapper(env, horizon_obs=3, horizon_act=3)

obs, info = env.reset(seed=0)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

Each env drives one SCR slot (so every env needs its own port in `3001..3010`).

## Parameters

| Parameter           | Default            | Meaning                                                                 |
| ------------------- | ------------------ | ----------------------------------------------------------------------- |
| `render_mode`       | `None`             | `None` runs TORCS headless, `"human"` opens the window                  |
| `port`              | `3001`             | UDP port of the SCR driver slot                                         |
| `track`             | `michigan` (oval)  | `TrackSpec`; `reset(options={"track": ...})` switches it                |
| `laps`              | `20`               | laps of the TORCS race                                                  |
| `vision`            | `False`            | also return a camera image (needs `render_mode="human"`)                |
| `image_shape`       | `(64, 64)`         | (height, width) of the camera image                                     |
| `observation_noise` | `0.01`             | std of the Gaussian noise added to the state                            |
| `max_steps`         | `25_000`           | truncate after this many steps (20 ms each)                             |
| `truncate_on_lap`   | `True`             | truncate when a lap is completed                                        |
| `random_start`      | `0.0`              | chance that tita first drives the car to a random point on the lap      |
| `failure_reward`    | `-100`             | reward for crashing, leaving the track or turning backwards             |
| `stall_reward`      | `-200`             | reward for standing still                                               |
| `racing_line_dir`   | `data/`            | one racing line CSV per track for curvature and random starts           |

## Spaces

|                                    |                                                                                         |
| ---------------------------------- | --------------------------------------------------------------------------------------- |
| Action Space                       | `Box(-1.0, 1.0, (2,), float32)`                                                         |
| Observation Space                  | `Box(-1.0, 1.0, (39,), float32)`                                                        |
| Observation Space (`vision=True`)  | `Dict("state": Box(-1.0, 1.0, (39,), float32), "image": Box(0, 255, (H, W, 3), uint8))` |

### Actions

Gas and brake share one axis, so they are never pressed together. Gears are shifted automatically.

| Num | Action                                                          | Min  | Max |
| --- | --------------------------------------------------------------- | ---- | --- |
| 0   | Steering, `-1` is full right, `1` is full left                  | -1.0 | 1.0 |
| 1   | Throttle, `1` is full throttle, `0` coasts, `-1` is full brake  | -1.0 | 1.0 |

### Observations

All sensors are scaled to `[-1, 1]`. Gaussian noise (`observation_noise=0.01`) is added afterwards and the result is clipped back to `[-1, 1]`.

| Num   | Observation                                                                        | Scaled from          |
| ----- | ---------------------------------------------------------------------------------- | -------------------- |
| 0-1   | Speed along the car's x, y axes                                                    | ±250 km/h            |
| 2-3   | Acceleration along the car's x, y axes                                             | ±50 m/s²             |
| 4     | Engine rpm                                                                         | 0 – 10,000 rpm       |
| 5     | Angle between the car's heading and the track axis (clipped)                       | ±π/6 rad             |
| 6     | Lateral position on the track, `-1` is the right edge, `1` the left edge           | track width          |
| 7-25  | 19 track edge range finders at -45°, -19°, -12°, -7°, -4°, -2.5°, -1.7°, -1°, -0.5°, 0°, 0.5°, 1°, 1.7°, 2.5°, 4°, 7°, 12°, 19°, 45° | 0 – 200 m |
| 26    | Lateral offset from the racing line of the `tita` robot                            | ±2 track half-widths |
| 27-38 | Racing line curvature 0, 25, 50, ..., 275 m ahead of the car                       | ±0.08 1/m (12.5 m radius) |


## Vision

```python
env = gym.make("TorcsSCR-v0", render_mode="human", vision=True, image_shape=(64, 64))
obs, info = env.reset()
obs["state"]  # (39,) float32
obs["image"]  # (64, 64, 3) uint8 RGB
```

## Reward

The reward per TORCS step is

```python
# θ is the angle between the car and the track axis (in rad)
# d the distance to tita racing line (in track widths)
(speed / 250 km/h) * (cos θ - |sin θ|) * (1 - d) 
```


| Failure                                         | `end_reason` | Reward                         |
| ----------------------------------------------- | ------------ | ------------------------------ |
| damage taken                                    | `crashed`    | `failure_reward` (default -100) |
| car center beyond a track edge (`\|trackPos\| > 1.1`) | `off_track` | `failure_reward`        |
| car pointing more than 90° away from the track  | `backwards`  | `failure_reward`               |
| slower than 5 km/h after the first 100 steps    | `stalled`    | `stall_reward` (default -200)  |

A stall costs more than a crash because braking to a stop takes ~150 steps. Discounted with
γ = 0.997, a -100 stall would look like -64 when the agent starts braking, cheaper than
crashing, and agents learn to stop in front of corners they cannot take yet.

The off-track limit is 1.1 instead of 1 because tita counts a kerb on the inside of a corner as
1.5 m of extra track and cuts across it (it reaches 1.002 in aalborg's last hairpin). While
`1 < |trackPos| <= 1.1`, the range finders (observations 7-25) read `-1`.

## Info

`reset` and `step` return the same keys; `reset` also adds `track` and `track_length`.

| Key                   | Meaning                                                                                  |
| --------------------- | ---------------------------------------------------------------------------------------- |
| `end_reason`          | `"crashed"`, `"off_track"`, `"backwards"`, `"stalled"` (terminated), `"lap_completed"`, `"max_steps"`, `"torcs_timeout"`, `"race_over"` (truncated), `None` while running |
| `lap_completed`       | a lap was completed in this step                                                         |
| `laps_completed`      | laps completed in this episode                                                           |
| `lap_time`            | time of the last completed lap in s, 0 before the first                                  |
| `current_lap_time`    | time since the start of the current lap in s                                             |
| `episode_steps`       | steps taken in this episode                                                              |
| `distance_raced`      | distance driven since the start in m                                                     |
| `distance_from_start` | distance along the track from the start line in m                                        |
| `speed`               | forward speed in km/h                                                                    |
| `track_position`      | lateral position, `-1` is the right edge, `1` the left edge                              |
| `position_x`, `position_y` | global position of the car in m                                                     |
| `damage`, `damage_delta` | total damage, and damage taken in this step                                           |
| `expert_action`       | the action the `tita` robot would take, in this env's action format                      |

