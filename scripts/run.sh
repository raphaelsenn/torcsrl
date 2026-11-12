#!/bin/bash
# Reproduce the experiment of the README: DDPG and TD3, three seeds each.
set -euo pipefail
cd "$(dirname "$0")/.."

for seed in 0 1 2; do
    for algorithm in ddpg td3; do
        python scripts/train.py --algorithm "$algorithm" --seed "$seed" --timesteps 5000000 "$@"
    done
done
