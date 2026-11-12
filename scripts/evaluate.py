"""Watch a trained agent drive in the TORCS window.

Examples:
    # Drive a checkpoint on every training and validation track
    python scripts/evaluate.py checkpoints/nsteptd3/<run>/actor-nsteptd3-4727461.pt

    # Only some tracks, a few laps each
    python scripts/evaluate.py <actor.pt> --tracks brondehach alpine-2 --laps 3

    # Watch the tita expert robot instead of an agent
    python scripts/evaluate.py --expert --tracks aalborg

The algorithm is inferred from the checkpoint name (actor-<class>-<step>.pt),
pass --algorithm if the file was renamed.
"""

import argparse
import re
from pathlib import Path

import torch

from configs import ALGORITHMS, HISTORY_ACT, HISTORY_OBS, TRAIN_TRACKS, VAL_TRACKS
from gym_torcs import HistoryWrapper, TorcsEnv, TrackSpec


def parse_track(spec: str) -> TrackSpec:
    """``"aalborg"`` or ``"road/aalborg"``."""
    category, _, name = spec.rpartition("/")
    return TrackSpec(name, category or "road")


def infer_algorithm(checkpoint: Path) -> str:
    """``actor-nsteptd3-123.pt`` -> ``"td3"``."""
    by_class = {cls.__name__.lower(): key for key, (cls, _) in ALGORITHMS.items()}
    match = re.match(r"actor-(\w+)-\d+\.pt$", checkpoint.name)
    if match is None or match.group(1) not in by_class:
        raise SystemExit(f"Cannot infer the algorithm from {checkpoint.name}, pass --algorithm.")
    return by_class[match.group(1)]


def load_agent(env, checkpoint: Path, algorithm: str, device: str):
    algorithm_cls, config = ALGORITHMS[algorithm]
    # Only the networks are needed, so keep the replay buffer tiny.
    config = {**config, "buffer_size": config["batch_size"], "verbose": False}
    agent = algorithm_cls(env, env, device=device, **config)
    agent.actor.load_state_dict(torch.load(checkpoint, map_location=device))
    agent.actor.eval()

    # TD7 acts through a state encoder, saved next to the actor.
    if hasattr(agent, "encoder_fixed"):
        encoder = checkpoint.with_name(checkpoint.name.replace("actor-", "encoder-", 1))
        agent.encoder_fixed.load_state_dict(torch.load(encoder, map_location=device))
        agent.encoder_fixed.eval()
    return agent


def end_reason(info: dict, terminated: bool) -> str:
    if info.get("torcs_timeout"):
        return "timeout"
    if not terminated:
        return "finished"
    if info.get("crashed"):
        return "crashed"
    if info.get("offTrack"):
        return "off track"
    return "backwards or stalled"


def run_episode(env, agent, track: TrackSpec, seed: int, max_laps: int) -> dict:
    """Drive one episode and return its statistics."""
    obs, info = env.reset(seed=seed, options={"track": track})
    total_reward, steps, lap_times = 0.0, 0, []
    terminated = truncated = False

    while not (terminated or truncated) and len(lap_times) < max_laps:
        action = info["action_tita"] if agent is None else agent.act_numpy(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        new_lap = info.get("successfulLap") and (not lap_times or info["lastLapTime"] != lap_times[-1])
        if new_lap:
            lap_times.append(info["lastLapTime"])
            print(f"\n  lap {len(lap_times)} in {info['lastLapTime']:.2f}s")

        if steps % 50 == 0:
            print(f"\r{track.name:<12s} step {steps:>6d}  speed {info.get('speedX', 0.0):6.1f} km/h  "
                  f"distance {info.get('distRaced', 0.0):8.1f} m  return {total_reward:9.2f}", end="", flush=True)
    print()

    return {
        "track": f"{track.category}/{track.name}",
        "reason": end_reason(info, terminated),
        "return": total_reward,
        "distance": info.get("distRaced", 0.0),
        "best_lap": f"{min(lap_times):.2f}s" if lap_times else "-",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, nargs="?", help="actor checkpoint (.pt)")
    parser.add_argument("--algorithm", choices=ALGORITHMS, help="default: inferred from the checkpoint name")
    parser.add_argument("--expert", action="store_true", help="drive with the tita robot instead of an agent")
    parser.add_argument("--tracks", nargs="+", type=parse_track, default=TRAIN_TRACKS + VAL_TRACKS,
                        help="track names, optionally as category/name (default: training + validation tracks)")
    parser.add_argument("--episodes", type=int, default=1, help="episodes per track")
    parser.add_argument("--laps", type=int, default=1, help="laps before the episode ends")
    parser.add_argument("--observation-noise", type=float, default=0.025, help="state noise std, training uses 0.025")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()

    if args.expert == (args.checkpoint is not None):
        parser.error("pass either a checkpoint or --expert")

    env = TorcsEnv(
        render_mode="human",
        port=args.port,
        track=args.tracks[0],
        laps=args.laps,
        truncate_on_lap=args.laps == 1,  # with one lap, stop as soon as it is completed
        observation_noise=args.observation_noise,
    )
    env = HistoryWrapper(env, HISTORY_OBS, HISTORY_ACT)

    if args.expert:
        agent = None
        print("Driving the tita expert")
    else:
        algorithm = args.algorithm or infer_algorithm(args.checkpoint)
        agent = load_agent(env, args.checkpoint, algorithm, args.device)
        print(f"Driving {algorithm} from {args.checkpoint}")

    results = []
    try:
        for track in args.tracks:
            for episode in range(args.episodes):
                result = run_episode(env, agent, track, args.seed + episode, args.laps)
                results.append(result)
                print(f"{result['track']} episode {episode + 1}: {result['reason']}, "
                      f"return {result['return']:.2f}, distance {result['distance']:.1f} m")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()

    print("\nSummary")
    for r in results:
        print(f"  {r['track']:<20s} {r['reason']:<20s} return {r['return']:9.2f}  "
              f"distance {r['distance']:8.1f} m  best lap {r['best_lap']}")


if __name__ == "__main__":
    main()
