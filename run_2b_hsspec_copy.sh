#!/usr/bin/env bash
# On-box: HEAD-SET specificity + POWER for the COPY lead (2b/2b-it). The -it copy lead (knockout restores
# 0.5-0.67, low-conf more) was n=7. Here: copy head SET ranked by answer->W* attention, size sweep {1,3,5,10,20}
# jointly knocked out -> concentration; matched-random-K floor; big pool for faithful-n power. base + it (QA
# template). No content swap (W* cannot be swapped). CONCENTRATED_SET / DISTRIBUTED_SET / NO_RESTORE / INSUFFICIENT.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest (model-free) ==="
python cave_headset_specificity.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== ensure datasets (TruthfulQA --big-pool) ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"

echo "=== 2b head-set specificity, mode=copy, big pool (base qa + it qa) ==="
python cave_headset_specificity.py --device cuda --mode copy --big-pool \
  --name-base google/gemma-2-2b --name-it google/gemma-2-2b-it --tag 2b > out/cave_headset_specificity_copy_2b.log 2>&1
echo "exit=$?"
echo "--- tail ---"; tail -55 out/cave_headset_specificity_copy_2b.log
echo "ALLDONE_HSSPEC_COPY_2B"
