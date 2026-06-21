#!/usr/bin/env bash
# On-box: HEAD-SET specificity de-confound for the DOUBT-cue lead (9b). Top-K doubt-attending head SET knocked
# out jointly over a size sweep {1,3,5,10,20} -> concentration; matched-random-K floor; + CONTENT SWAP (doubt
# framing -> length-matched non-doubt phrase, same W* assertion) = the clean cue-content de-confound, no knockout
# artifact. base + it (QA template; big pool for power). CONCENTRATED_SET / DISTRIBUTED_SET / NO_RESTORE /
# INSUFFICIENT + CONTENT_SPECIFIC / CONTENT_NONSPECIFIC.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest (model-free) ==="
python cave_headset_specificity.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== ensure datasets (TruthfulQA --big-pool) ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"

echo "=== 9b head-set specificity, mode=doubt, big pool (base qa + it qa) ==="
python cave_headset_specificity.py --device cuda --mode doubt --big-pool \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/cave_headset_specificity_doubt_9b.log 2>&1
echo "exit=$?"
echo "--- tail ---"; tail -55 out/cave_headset_specificity_doubt_9b.log
echo "ALLDONE_HSSPEC_DOUBT_9B"
