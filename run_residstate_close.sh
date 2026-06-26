#!/usr/bin/env bash
# On-box CLOSE: matched union set + -it re-localize (DLA-onto-cave-axis writer heads). Resolves the v5 lead's
# two caveats (intersection-0 item-confound; -it localization mismatch). Reloads 9b a few times (matched
# batteries) -- reload-heavy but correct. Verdict: DISTRIBUTED_CONFIRMED / RELOCATED / BASE_NULL / AXIS_WEAK.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_residstate_close.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
echo "=== ensure datasets ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"
echo "=== residstate CLOSE, 9b base vs 9b-it ==="
python cave_residstate_close.py --device cuda --big-pool \
  --base google/gemma-2-9b --it google/gemma-2-9b-it \
  > out/cave_residstate_close_9b.log 2>&1; echo "exit=$?"
echo "--- tail ---"; tail -45 out/cave_residstate_close_9b.log
echo "ALLDONE_RESIDSTATE_CLOSE"
