#!/usr/bin/env bash
# On-box INTERVENTION (1) PHASE 1: judge-free multi-sample cave-rate on the CLEAN substrate (NO --big-pool ->
# factual misconceptions only, drops the whimsical sycophancy-pool items that produced 'playful' non-caves).
# More genuine caves for power; cleaner self-judge labels; saves all generations for the reader + the Phase-2
# external-judge panel. Files land FLAT in ~/latent_verify (scp convention).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_multisample_caverate.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
echo "=== ensure datasets ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"
echo "=== multisample cave-rate, CLEAN substrate (no big-pool), 9b base vs 9b-it ==="
python cave_multisample_caverate.py --device cuda \
  --base google/gemma-2-9b --it google/gemma-2-9b-it \
  > out/cave_multisample_caverate_clean_9b.log 2>&1; echo "exit=$?"
echo "--- tail ---"; tail -60 out/cave_multisample_caverate_clean_9b.log
echo "ALLDONE_MULTISAMPLE_CLEAN"
