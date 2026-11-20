"""
Drive the TORCS env with random actions.

Example:
    python scripts/random_policy.py --track aalborg --episodes 3
"""

import argparse
import random

from gym_torcs import TorcsEnv, TrackSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", default="aalborg", help="road track name")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_start", type=float, default=0.0)
    args = parser.parse_args()

    env = TorcsEnv(render_mode="human", port=args.port, track=TrackSpec(args.track), random_start=args.random_start)
    env.action_space.seed(args.seed)
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        total_reward, steps, done = 0.0, 0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            print(f"Observation shape: {obs.shape}")
            total_reward += reward
            steps += 1
            done = terminated or truncated
        print(f"episode {episode + 1}: {steps} steps, return {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
