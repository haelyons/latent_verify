#!/usr/bin/env bash
# On-box: I1 truthful_flip, gemma-2-9b BASE only (one model per box -> parallel, under timeout).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
pip install -q datasets 2>/dev/null
echo "=== I1 truthful_flip 9b base ==="
python job_truthful_flip.py --truthfulqa --name google/gemma-2-9b --tag 9b_base > out/i1_9b_base.log 2>&1; echo "9b_base exit=$?"
echo "ALLDONE_9B_BASE"
