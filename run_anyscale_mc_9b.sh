#!/usr/bin/env bash
# On-box: Job A (residstate ANYSCALE @ 9b -- faithfulness gate: MUST reproduce cave_residstate_decisive's
# 9b numbers, layer 28 by construction) + Job C (forced-choice MC -it readout @ 9b). Selftests gate both.
# Reload-heavy (anyscale = the decisive battery's ~4 loads + MC's 2) -> launch with REMOTE_TIMEOUT>=9000.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftests (model-free gate) ==="
python cave_residstate_anyscale.py --selftest || { echo "SELFTEST_FAIL_ANYSCALE"; exit 1; }
python cave_faithful_it_mc.py        --selftest || { echo "SELFTEST_FAIL_MC"; exit 1; }
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -2 || echo "DATASETS_PIP_FAILED"

echo "=== Job A: residstate ANYSCALE, 9b base vs 9b-it (GATE: reproduce decisive) ==="
python cave_residstate_anyscale.py --device cuda --big-pool \
  --base google/gemma-2-9b --it google/gemma-2-9b-it --tag 9b \
  > out/anyscale_9b.log 2>&1; echo "exit=$?"
echo "--- tail anyscale 9b ---"; tail -55 out/anyscale_9b.log

echo "=== Job C: faithful-it MC, 9b base vs 9b-it ==="
python cave_faithful_it_mc.py --device cuda --big-pool \
  --base google/gemma-2-9b --it google/gemma-2-9b-it --tag 9b \
  > out/mc_9b.log 2>&1; echo "exit=$?"
echo "--- tail mc 9b ---"; tail -55 out/mc_9b.log

echo "ALLDONE_ANYSCALE_MC_9B"
