#!/bin/bash
# Reproduce the experiment of the README: N-step DDPG and N-step TD3, three seeds each.
set -euo pipefail
cd "$(dirname "$0")/.."

for seed in 0 1; do
    for algorithm in nstepddpg nsteptd3; do
        python scripts/train.py --algorithm "$algorithm" --device "cuda" --seed "$seed" --timesteps 5000000 "$@"
    done
done
