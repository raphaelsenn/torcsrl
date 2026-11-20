#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

for seed in 0 1; do
    for algorithm in td3; do
        python scripts/train.py --algorithm "$algorithm" --device "cuda" --seed "$seed" --timesteps 5000000 "$@"
    done
done
