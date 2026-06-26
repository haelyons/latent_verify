#!/usr/bin/env bash
# On-box CAUSAL test of the cave-direction (held-out steer + matched-norm random placebo + per-item CI) PAIRED with
# DLA positive-localization (which heads + MLP layers WRITE the cave-axis). TruthfulQA substrate, 9b base+it.
# Forward-only (steer = resid hook; DLA = cached component outputs) + single free-gen for the -it label.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_causal_localize.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
echo "=== ensure datasets ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"
echo "=== causal steer + DLA localize, TruthfulQA, 9b base vs 9b-it ==="
python cave_causal_localize.py --device cuda --truthfulqa --n 60 \
  --base google/gemma-2-9b --it google/gemma-2-9b-it \
  --it-labels causal_it_labels.json \
  > out/cave_causal_localize_9b.log 2>&1; echo "exit=$?"
echo "--- tail ---"; tail -50 out/cave_causal_localize_9b.log
echo "ALLDONE_CAUSAL_LOCALIZE"
