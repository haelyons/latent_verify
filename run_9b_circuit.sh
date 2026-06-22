#!/usr/bin/env bash
# On-box: SHAPE-AGNOSTIC + SET-AWARE caving-circuit finder. On a FIXED faithful caving-item set (selected once
# under the DOUBT framing), attribution-patch EVERY head + MLP (no span assumption) -> rank; CONFIRM set-aware
# (joint activation-patch size-sweep of the ATP-top-K patched TOGETHER + the DOUBT-classed top set as a
# reference); DESCRIBE the top components (attention target + DLA W*-C). CONCENTRATED / DISTRIBUTED + a
# component-class breakdown (are the causal components the doubt-attending heads?). 9b base, big pool.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest (model-free) ==="
python cave_circuit_patch.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== ensure datasets (--big-pool) ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"

echo "=== 9b shape-agnostic circuit finder, big pool (base qa) ==="
python cave_circuit_patch.py --device cuda --big-pool \
  --name google/gemma-2-9b --tag 9b_base > out/cave_circuit_patch_9b_base.log 2>&1
echo "exit=$?"
echo "--- tail ---"; tail -65 out/cave_circuit_patch_9b_base.log
echo "ALLDONE_CIRCUIT_9B"
