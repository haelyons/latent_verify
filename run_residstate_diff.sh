#!/usr/bin/env bash
# On-box: residual-state doubt-head battery, base<->it (RLHF-on-the-doubt-circuit, readout = resid.cave-axis,
# NOT the emitted token). base label = realized argmax (faithful); it label = free-gen self-judge. Loads 9b base
# then 9b-it. AXIS_WEAK guard if the held-out cave-axis AUROC < 0.70 (then no verdict).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_residstate_diff.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
echo "=== ensure datasets ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"
echo "=== residual-state diff, 9b base vs 9b-it ==="
python cave_residstate_diff.py --device cuda --big-pool \
  --base google/gemma-2-9b --it google/gemma-2-9b-it \
  > out/cave_residstate_diff_9b.log 2>&1; echo "exit=$?"
echo "--- tail ---"; tail -50 out/cave_residstate_diff_9b.log
echo "ALLDONE_RESIDSTATE_DIFF"
