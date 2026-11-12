"""Record the racing line of the tita robot for the look-ahead curvature.

The car follows tita's actions for a number of laps, and its position is
written to data/<track>.csv with the columns x, y, distFromStart. The env reads
these files to compute the curvature ahead of the car.

Example:
    python scripts/collect_racing_lines.py                    # all experiment tracks
    python scripts/collect_racing_lines.py forza alpine-2 --laps 20
"""

import argparse
import csv
from pathlib import Path

from configs import TRAIN_TRACKS, VAL_TRACKS
from gym_torcs import TorcsEnv, TrackSpec
from gym_torcs.constants import RACING_LINE_DIR

# |trackPos| beyond which the car counts as off the track for recording.
OFF_TRACK_LIMIT = 1.2


def record_laps(env: TorcsEnv, laps: int, max_restarts: int | None = 5) -> list[tuple[float, float, float]]:
    """Drive with tita until ``laps`` laps are completed, restarting after failures.

    Raises ``RuntimeError`` once tita failed more than ``max_restarts`` times.
    """
    rows, completed, restarts = [], 0, 0
    _, info = env.reset(seed=0)
    last_lap_time = 0.0

    while completed < laps:
        _, _, _, truncated, info = env.step(info["action_tita"])

        # The env terminates as soon as a wheel touches the edge (|trackPos| > 1),
        # tita brushes it on some tracks, so only restart on real failures.
        if info.get("torcs_timeout"):
            reason = "TORCS timed out"
        elif info["crashed"]:
            reason = "tita crashed"
        elif abs(info["trackPos"]) > OFF_TRACK_LIMIT:
            reason = f"tita left the track (trackPos {info['trackPos']:.2f})"
        elif truncated:
            reason = "episode truncated"
        else:
            reason = None

        if reason is not None:
            restarts += 1
            if max_restarts is not None and restarts > max_restarts:
                raise RuntimeError(f"{reason}; gave up after {max_restarts} restarts, {completed}/{laps} laps")
            print(f"  {reason} at distFromStart {info.get('distFromStart', float('nan')):.0f}, restarting")
            _, info = env.reset()
            last_lap_time = 0.0
            continue

        rows.append((info["globalX"], info["globalY"], info["distFromStart"]))
        if info["lastLapTime"] != last_lap_time:
            last_lap_time = info["lastLapTime"]
            completed += 1
            print(f"  lap {completed}/{laps}: {last_lap_time:.2f} s")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tracks", nargs="*", help="road track names, defaults to all experiment tracks")
    parser.add_argument("--laps", type=int, default=10)
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--out-dir", type=Path, default=RACING_LINE_DIR)
    parser.add_argument("--overwrite", action="store_true", help="replace existing racing lines")
    args = parser.parse_args()

    tracks = [TrackSpec(name) for name in args.tracks] or TRAIN_TRACKS + VAL_TRACKS
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for track in tracks:
        path = args.out_dir / f"{track.name}.csv"
        if path.exists() and not args.overwrite:
            print(f"{track.name}: {path} exists, skipping (use --overwrite to replace)")
            continue

        print(f"{track.name}: recording {args.laps} laps")
        env = TorcsEnv(port=args.port, track=track, laps=args.laps + 1,
                       observation_noise=0.0, max_steps=10**9, truncate_on_lap=False)
        try:
            rows = record_laps(env, args.laps)
        except RuntimeError as e:
            print(f"{track.name}: {e}, skipping")
            continue
        finally:
            env.close()

        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "distFromStart"])
            writer.writerows(rows)
        print(f"{track.name}: wrote {len(rows)} samples to {path}")


if __name__ == "__main__":
    main()
