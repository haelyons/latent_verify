#!/usr/bin/env bash
# On-box: the untested gap -- does gemma-2-27b CAVE on TruthfulQA misconceptions, and is attention-to-W*
# (copy of the asserted wrong answer) necessary? job_truthful_flip at 27b base (qa) + it (chat). --sweep-cap 0
# SKIPS the per-head knockout (1472 heads x items is too slow at 27b, and per-head is already DIFFUSE at
# 2b/9b); keeps the primary: flip counts, all-heads attn->W* knockout necessity + matched neutral-span control.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python job_truthful_flip.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== install datasets (TruthfulQA) ==="
pip install -q datasets || echo "DATASETS_INSTALL_FAILED"

echo "=== 27b BASE caving (qa, no per-head sweep) ==="
python job_truthful_flip.py --truthfulqa --name google/gemma-2-27b --tag 27b_base --sweep-cap 0 \
  > out/truthful_flip_27b_base.log 2>&1
echo "exit=$?"; echo "--- tail 27b_base ---"; tail -25 out/truthful_flip_27b_base.log

echo "=== 27b-IT caving (chat, no per-head sweep) ==="
python job_truthful_flip.py --truthfulqa --name google/gemma-2-27b-it --tag 27b_it --chat --sweep-cap 0 \
  > out/truthful_flip_27b_it.log 2>&1
echo "exit=$?"; echo "--- tail 27b_it ---"; tail -25 out/truthful_flip_27b_it.log

echo "ALLDONE_27B_CAVING"
