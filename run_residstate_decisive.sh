#!/usr/bin/env bash
# On-box DECISIVE close-of-the-close (PART8 v7): the owed controls after latent_skeptic wf_f807a702 downgraded
# "RLHF -> non-attention/distributed" to OPEN. Matched-union set + cave-axis readout (= cave_residstate_close),
# new = intervention coverage: (A) ALL-attention upper bound, (B) ALL-MLP upper bound, (C) -it positive control
# (steer the cave-axis, read the output margin), (D) label-match (self-judge vs realized-argmax axis), + bootstrap CI.
# Reload-heavy (4 model loads: 2 _measure + 2 decisive batteries) -> bump the hard cap. Verdict:
# ATTENTION_CARRIES / MLP_CARRIES / BOTH_REDUNDANT / NEITHER_LOCALIZED / CHANNEL_INERT / INSUFFICIENT.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_residstate_decisive.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
echo "=== ensure datasets ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"
echo "=== residstate DECISIVE, 9b base vs 9b-it ==="
python cave_residstate_decisive.py --device cuda --big-pool \
  --base google/gemma-2-9b --it google/gemma-2-9b-it \
  > out/cave_residstate_decisive_9b.log 2>&1; echo "exit=$?"
echo "--- tail ---"; tail -60 out/cave_residstate_decisive_9b.log
echo "ALLDONE_RESIDSTATE_DECISIVE"
