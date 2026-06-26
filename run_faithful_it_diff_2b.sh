#!/usr/bin/env bash
# On-box: faithful -it readout + base<->it doubt-circuit differential at 2b (A10-fits; 9b deferred to A100).
# 2b-it caves harder behaviorally than 9b-it, so the prefilled answer-set readout is more likely to clear the
# faithfulness gate here. Loads 2b base then 2b-it (one resident at a time; trivial on A10-24GB).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest (model-free) ==="
python cave_faithful_it_diff.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== ensure datasets (--big-pool) ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"

echo "=== faithful -it diff, 2b base vs 2b-it (prefilled answer-set) ==="
python cave_faithful_it_diff.py --device cuda --big-pool \
  --base google/gemma-2-2b --it google/gemma-2-2b-it --tag 2b \
  > out/cave_faithful_it_diff_2b.log 2>&1; echo "exit=$?"
echo "--- tail ---"; tail -45 out/cave_faithful_it_diff_2b.log

echo "ALLDONE_FAITHFUL_IT_DIFF_2B"
