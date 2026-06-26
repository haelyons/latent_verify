#!/usr/bin/env bash
# On-box: Job B -- the doubt circuit RE-LOCALIZED at 27b (the scale never tested). STAGED RE-RUN of two
# already-validated, scale-portable controls (investigator a617db7: heads localized at runtime via rank_heads;
# 0 hardcoded-layer blockers; K<=20 << 1472 heads). 27b bf16 ~54GB -> needs an 80GB H100. No new code.
# write-vs-read (base) resolves READ vs WRITE on the span-ranked doubt heads; head-specificity confirms the set.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftests (model-free gate) ==="
python cave_doubt_write_vs_read.py --selftest || { echo "SELFTEST_FAIL_DOUBTWVR"; exit 1; }
python cave_headset_specificity.py --selftest || { echo "SELFTEST_FAIL_HSSPEC"; exit 1; }
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -2 || echo "DATASETS_PIP_FAILED"

echo "=== Job B: doubt write-vs-read, 27b base ==="
python cave_doubt_write_vs_read.py --device cuda --big-pool \
  --name google/gemma-2-27b --tag 27b_base \
  > out/doubtwvr_27b.log 2>&1; echo "exit=$?"
echo "--- tail doubtwvr 27b ---"; tail -60 out/doubtwvr_27b.log

echo "=== Job B: head-set specificity (doubt mode), 27b base vs -it ==="
python cave_headset_specificity.py --mode doubt --device cuda \
  --name-base google/gemma-2-27b --name-it google/gemma-2-27b-it --tag 27b \
  > out/hsspec_doubt_27b.log 2>&1; echo "exit=$?"
echo "--- tail hsspec 27b ---"; tail -60 out/hsspec_doubt_27b.log

echo "ALLDONE_DOUBT_27B"
