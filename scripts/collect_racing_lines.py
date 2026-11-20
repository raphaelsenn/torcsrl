"""
Record tita's racing line, which the env uses for the look-ahead curvature, as done in the paper: 
    Formula RL: Deep Reinforcement Learning for Autonomous Racing using Telemetry Data, from Adrian Remonda et al. 2022
    https://arxiv.org/abs/2104.11106

Examples:
    python scripts/collect_racing_lines.py  # all experiment tracks
    python scripts/collect_racing_lines.py forza --laps 20 --overwrite
"""

import argparse
import csv
from pathlib import Path

from configs import TRAIN_TRACKS, VAL_TRACKS
from gym_torcs import TorcsEnv, TrackSpec
from gym_torcs.constants import RACING_LINE_DIR

MAX_FAILURES = 5


def record(track: TrackSpec, laps: int, port: int) -> list[tuple[float, float, float]] | None:
    """Let tita drive `laps` laps and return (x, y, distFromStart) of every step, or None if tita keeps failing."""
    env = TorcsEnv(port=port, track=track, laps=laps + 1, observation_noise=0.0, max_steps=10**9, truncate_on_lap=False)
    rows, completed, failures = [], 0, 0
    _, info = env.reset(seed=0)
    while completed < laps and failures <= MAX_FAILURES:
        _, _, _, _, info = env.step(info["expert_action"])
        if info["end_reason"] is not None:  # no lap or step limit, so this is a failure
            failures += 1
            print(f"  tita {info['end_reason']}, restarting")
            _, info = env.reset()
            continue
        rows.append((info["position_x"], info["position_y"], info["distance_from_start"]))
        completed += info["lap_completed"]
    env.close()
    return rows if completed == laps else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tracks", nargs="*", default=[t.name for t in TRAIN_TRACKS + VAL_TRACKS])
    parser.add_argument("--laps", type=int, default=10)
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--out-dir", type=Path, default=RACING_LINE_DIR)
    parser.add_argument("--overwrite", action="store_true", help="replace existing files")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.tracks:
        path = args.out_dir / f"{name}.csv"
        if path.exists() and not args.overwrite:
            print(f"{name}: {path} exists, skipping")
            continue
        rows = record(TrackSpec(name), args.laps, args.port)
        if rows is None:
            print(f"{name}: tita failed more than {MAX_FAILURES} times, skipping")
            continue
        with path.open("w", newline="") as f:
            csv.writer(f).writerows([("x", "y", "distFromStart"), *rows])
        print(f"{name}: wrote {len(rows)} samples to {path}")


if __name__ == "__main__":
    main()
