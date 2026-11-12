"""Train an agent on the training tracks and evaluate it on all tracks.

Example:
    python scripts/train.py --algorithm td3 --seed 0 --device cuda

Checkpoints and evaluation statistics are written to ./checkpoints.
"""

import argparse

from configs import ALGORITHMS, TRAIN_TRACKS, VAL_TRACKS, make_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="td3")
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--port", type=int, default=3001, help="training env port, validation uses port + 1")
    args = parser.parse_args()

    # Train on one track at a time, switching every 50k steps.
    train_env = make_env(TRAIN_TRACKS, switch_every_steps=50_000, port=args.port)
    # Each evaluation drives one episode per track, training tracks included.
    val_env = make_env(TRAIN_TRACKS + VAL_TRACKS, switch_every_steps=1, port=args.port + 1)

    algorithm_cls, config = ALGORITHMS[args.algorithm]
    agent = algorithm_cls(train_env, val_env, seed=args.seed, device=args.device, **config)
    try:
        agent.train(args.timesteps)
    finally:
        train_env.close()
        val_env.close()


if __name__ == "__main__":
    main()
