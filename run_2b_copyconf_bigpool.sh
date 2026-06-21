#!/usr/bin/env bash
# On-box: POWERED re-run of the copy x confidence conditional test. --big-pool merges TruthfulQA (generation)
# + sycophancy_items_lowconf.json with ITEMS_WIDE to lift faithful-caving n at 2b base (the binding limit was
# n=4). Same control/decision as before; only the item pool grows. base primary + it.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest (model-free) ==="
python cave_copy_confidence_conditional.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== ensure datasets (for TruthfulQA) ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED (control proceeds without TruthfulQA)"

echo "=== 2b copy x confidence conditional, BIG POOL (base qa + it) ==="
python cave_copy_confidence_conditional.py --device cuda --big-pool \
  --name-base google/gemma-2-2b --name-it google/gemma-2-2b-it --tag 2b_bigpool > out/cave_copy_confidence_conditional_2b_bigpool.log 2>&1
echo "exit=$?"
echo "--- tail ---"; tail -60 out/cave_copy_confidence_conditional_2b_bigpool.log
echo "ALLDONE_COPYCONF_BIGPOOL"
