"""
Watch a trained RL agent (or the tita expert bot) drivew.

Examples:
    python scripts/evaluate.py checkpoints/td3/<run>/actor-td3-100000.pt
    python scripts/evaluate.py <actor.pt> --tracks aalborg alpine-2 --laps 3
    python scripts/evaluate.py --expert --tracks aalborg
"""

import argparse
from pathlib import Path

import torch

from configs import ALGORITHMS, TRAIN_TRACKS, VAL_TRACKS, wrap_env
from gym_torcs import TorcsEnv, TrackSpec


def load_agent(env, checkpoint: Path, device: str):
    """Build the agent named in the checkpoint file (actor-<algorithm>-<step>.pt) and load its actor."""
    name = checkpoint.stem.split("-")[1]
    by_class = {cls.__name__.lower(): (cls, config) for cls, config in ALGORITHMS.values()}
    if name not in by_class:
        raise SystemExit(f"Unknown algorithm {name!r} in {checkpoint.name}.")
    cls, config = by_class[name]

    agent = cls(env, env, device=device, **{**config, "buffer_size": config["batch_size"], "verbose": False})
    agent.actor.load_state_dict(torch.load(checkpoint, map_location=device))
    agent.actor.eval()
    return agent


def run_episode(env, agent, track: TrackSpec, seed: int, laps: int) -> str:
    """Drive one episode and return a one-line summary."""
    obs, info = env.reset(seed=seed, options={"track": track})
    total_reward, lap_times, done = 0.0, [], False

    while not done and len(lap_times) < laps:
        action = info["expert_action"] if agent is None else agent.act_numpy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        if info["lap_completed"]:
            lap_times.append(info["lap_time"])
        if info["episode_steps"] % 50 == 0:
            print(f"\r{track.name:<12s} step {info['episode_steps']:>6d}  speed {info['speed']:6.1f} km/h  "
                  f"distance {info['distance_raced']:8.1f} m  return {total_reward:9.2f}", end="", flush=True)
    print("\r\033[K", end="")  # clear the progress line

    end = info["end_reason"] or f"{laps} laps"
    best_lap = f"{min(lap_times):.2f}s" if lap_times else "-"
    return (f"{track.name:<12s} {end:<14s} return {total_reward:9.2f}  "
            f"distance {info['distance_raced']:8.1f} m  best lap {best_lap}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, nargs="?", help="actor checkpoint, actor-<algorithm>-<step>.pt")
    parser.add_argument("--expert", action="store_true", help="drive with tita instead of an agent")
    parser.add_argument("--tracks", nargs="+", default=[t.name for t in TRAIN_TRACKS + VAL_TRACKS])
    parser.add_argument("--episodes", type=int, default=1, help="episodes per track")
    parser.add_argument("--laps", type=int, default=1, help="laps per episode")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()
    if args.expert == (args.checkpoint is not None):
        parser.error("pass either a checkpoint or --expert")

    # With more than one lap, the episode must not end at the first finish line.
    env = wrap_env(TorcsEnv(render_mode="human", port=args.port, laps=args.laps, truncate_on_lap=args.laps == 1))
    agent = None if args.expert else load_agent(env, args.checkpoint, args.device)

    for name in args.tracks:
        for episode in range(args.episodes):
            print(run_episode(env, agent(name), args.seed + episode, args.laps))
    env.close()


if __name__ == "__main__":
    main()
