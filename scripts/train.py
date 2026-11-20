"""
Train an RL agent on the training tracks and validate it on all tracks.

Example:
    python scripts/train.py --algorithm td3 --seed 0 --device cuda

Checkpoints and validation stats are written to the folder ./checkpoints.
"""

import argparse

from configs import ALGORITHMS, TRAIN_TRACKS, VAL_TRACKS, make_train_env, make_val_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="td3")
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--port", type=int, default=3001, help="training port, validation uses port + 1")
    args = parser.parse_args()

    train_env = make_train_env(TRAIN_TRACKS, port=args.port)
    val_env = make_val_env(TRAIN_TRACKS + VAL_TRACKS, port=args.port + 1)
    cls, config = ALGORITHMS[args.algorithm]
    agent = cls(train_env, val_env, seed=args.seed, device=args.device, **config)

    print(f"Training {cls.__name__} on {args.device}, seed {args.seed}, {args.timesteps:,} steps")
    print(f"Train tracks (port {args.port}): {', '.join(t.name for t in TRAIN_TRACKS)}")
    print(f"Val tracks   (port {args.port + 1}): {', '.join(t.name for t in VAL_TRACKS)} (+ train tracks)")
    print(f"Settings: {config}")
    try:
        agent.train(args.timesteps)
    finally:
        train_env.close()
        val_env.close()


if __name__ == "__main__":
    main()
