#!/usr/bin/env bash
# On-box: PART 7 Control 3 -- confidence_vs_cave_direction (confidence/margin direction + cosine(cave,conf)
# + off-intersection necessity). SC-C1/C2/DISSOC: is the cave-direction a deference signal or a confidence
# axis? Imports rlhf_differential / headset_direction / misconception_pool / job_truthful_flip (scp'd flat).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python confidence_vs_cave_direction.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b confidence_vs_cave_direction (base vs -it) ==="
python confidence_vs_cave_direction.py --device cuda \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/confidence_vs_cave_9b.log 2>&1
echo "exit=$?"
echo "--- tail confidence_vs_cave_9b.log ---"; tail -45 out/confidence_vs_cave_9b.log
echo "ALLDONE_CONFCAVE"
