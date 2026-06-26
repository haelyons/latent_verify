#!/usr/bin/env bash
# On-box FOLD-vs-LISTEN (9b: the headline scale -- the clean causal doubt site). Is the span-ranked doubt
# circuit wrongness-SPECIFIC or a GENERIC move-to-asserted mechanism? base-primary; -it self-bracketed
# (positive control + all-attention upper bound). Verdict: SC-SHARED / SC-DIRECTION / SC-DISTINCT / AXIS_WEAK /
# INSTRUMENT_DEAD / MOVE_UNMATCHED / INSUFFICIENT. Files land FLAT in ~/latent_verify (scp convention).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_fold_vs_listen.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
echo "=== ensure datasets ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"
echo "=== fold-vs-listen, 9b base vs 9b-it ==="
python cave_fold_vs_listen.py --device cuda --big-pool \
  --base google/gemma-2-9b --it google/gemma-2-9b-it \
  > out/cave_fold_vs_listen_9b.log 2>&1; echo "exit=$?"
echo "--- tail ---"; tail -60 out/cave_fold_vs_listen_9b.log
echo "ALLDONE_FOLD_VS_LISTEN_9B"
