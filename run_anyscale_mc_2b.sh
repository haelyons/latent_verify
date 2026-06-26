#!/usr/bin/env bash
# On-box: Job A (residstate ANYSCALE @ 2b -- the NEW arm; cave-axis layer auto = round(0.667*26)=17) + Job C
# (forced-choice MC -it readout @ 2b). 2b fits a 24GB A10. Selftests gate both.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftests (model-free gate) ==="
python cave_residstate_anyscale.py --selftest || { echo "SELFTEST_FAIL_ANYSCALE"; exit 1; }
python cave_faithful_it_mc.py        --selftest || { echo "SELFTEST_FAIL_MC"; exit 1; }
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -2 || echo "DATASETS_PIP_FAILED"

echo "=== Job A: residstate ANYSCALE, 2b base vs 2b-it (NEW arm) ==="
python cave_residstate_anyscale.py --device cuda --big-pool \
  --base google/gemma-2-2b --it google/gemma-2-2b-it --tag 2b \
  > out/anyscale_2b.log 2>&1; echo "exit=$?"
echo "--- tail anyscale 2b ---"; tail -55 out/anyscale_2b.log

echo "=== Job C: faithful-it MC, 2b base vs 2b-it ==="
python cave_faithful_it_mc.py --device cuda --big-pool \
  --base google/gemma-2-2b --it google/gemma-2-2b-it --tag 2b \
  > out/mc_2b.log 2>&1; echo "exit=$?"
echo "--- tail mc 2b ---"; tail -55 out/mc_2b.log

echo "ALLDONE_ANYSCALE_MC_2B"
