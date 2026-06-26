#!/usr/bin/env bash
# On-box: faithful -it readout + base<->it doubt-circuit differential (RLHF-on-the-doubt-circuit).
# Assistant-prefill "The answer is" makes C/W* decidable at one slot in BOTH models (weights-only contrast);
# answer-SET readout; R2 generation-validator gates the prefill; R3 softening reported. Loads 9b base then
# 9b-it (one resident at a time on A100-40GB). Verdict = INSTALL / AMPLIFY / RESHAPE / DISTRIBUTED, or the
# honest READOUT_STILL_BLOCKED if -it stays a tail-ghost even under prefill.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest (model-free) ==="
python cave_faithful_it_diff.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== ensure datasets (--big-pool) ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"

echo "=== faithful -it diff, 9b base vs 9b-it (prefilled answer-set) ==="
python cave_faithful_it_diff.py --device cuda --big-pool \
  --base google/gemma-2-9b --it google/gemma-2-9b-it --tag 9b \
  > out/cave_faithful_it_diff_9b.log 2>&1; echo "exit=$?"
echo "--- tail ---"; tail -45 out/cave_faithful_it_diff_9b.log

echo "ALLDONE_FAITHFUL_IT_DIFF"
