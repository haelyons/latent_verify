#!/usr/bin/env bash
# On-box: D2 at the next model size -- gate_dont_delete --select copy on gemma-2-27b base vs -it.
# Forward-only (weights-only OV + induction-pattern QK), so no AtP-backward OOM. 27b bf16 ~54GB ->
# needs GH200 96GB (loads base, frees, loads -it; peak one model at a time). Script is model-agnostic
# (auto nL/nH/GQA); runner just retags the hardcoded "2b" output filename to 27b.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest (model-free) ==="
python gate_dont_delete.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 27b gate-dont-delete --select copy (base vs -it) ==="
python gate_dont_delete.py --select copy \
  --name-base google/gemma-2-27b --name-it google/gemma-2-27b-it > out/gdd_27b.log 2>&1
echo "exit=$?"
mv out/gate_dont_delete_2b_copy.json out/gate_dont_delete_27b_copy.json 2>/dev/null || true
echo "--- tail gdd_27b.log ---"; tail -55 out/gdd_27b.log
echo "ALLDONE_27B_MOTIF"
