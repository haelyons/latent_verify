#!/usr/bin/env bash
# On-box (a) ROUTE: does the span-doubt-5 heads' restoration route THROUGH the ATP-top MLPs? On fixed 9b-base
# DOUBT faithful items: baseline = doubt-head attention-KO (~0.59); route test = same KO while FREEZING the
# top-K MLPs at their counter (un-restored) output; control = freeze a matched-random-K MLP set. ROUTES_THROUGH
# _MLPS / DIRECT_OR_OTHER / NONSPECIFIC / NO_BASELINE / INSUFFICIENT. big pool.
set -uo pipefail
cd ~/latent_verify; . .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== selftest ==="; python cave_doubt_route.py --selftest || { echo SELFTEST_FAIL; exit 1; }
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -2 || echo DATASETS_PIP_FAILED
echo "=== 9b doubt route, big pool ==="
python cave_doubt_route.py --device cuda --big-pool --name google/gemma-2-9b --tag 9b_base > out/cave_doubt_route_9b_base.log 2>&1
echo "exit=$?"; echo "--- tail ---"; tail -40 out/cave_doubt_route_9b_base.log; echo ALLDONE_DOUBTROUTE_9B
