#!/usr/bin/env bash
# On-box: NEXT-2 -- realized attention-to-source on a content-copy input, gemma-2-27b base vs -it. Does
# the copy basket actually attend the source token it would copy (realizes copy), and does the realized
# attention change base->it despite W_QK weights being unchanged (2b-style realized collapse)? Forward +
# hook_pattern; needs an 80GB GPU (27b). realized_attention.py scp'd flat.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python realized_attention.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 27b realized attention-to-source (base vs -it, content-copy, GPU) ==="
python realized_attention.py --device cuda \
  --name-base google/gemma-2-27b --name-it google/gemma-2-27b-it --tag 27b > out/realized_attention_27b.log 2>&1
echo "exit=$?"
echo "--- tail realized_attention_27b.log ---"; tail -40 out/realized_attention_27b.log
echo "ALLDONE_REALATTN"
