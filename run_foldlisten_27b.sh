#!/usr/bin/env bash
# On-box FOLD vs LISTEN (auditable: full completions to EOS, stored prompts, self-judge on the elicited final
# answer) -- gemma-2-27b base THEN 27b-it (each ~54GB bf16, loaded one-at-a-time; needs an >=80GB box).
# Files land FLAT in ~/latent_verify; results to out/.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out
echo "=== selftest (model-free) ==="
python foldlisten_judge.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
echo "=== 27b base (qa) ==="
python foldlisten_judge.py --family verifier_family --name google/gemma-2-27b --tag fl_27bbase --device cuda \
  > out/foldlisten_27bbase.log 2>&1; echo "exit=$?"
tail -30 out/foldlisten_27bbase.log
echo "=== 27b-it (chat) ==="
python foldlisten_judge.py --family verifier_family --name google/gemma-2-27b-it --tag fl_27bit --device cuda --chat \
  > out/foldlisten_27bit.log 2>&1; echo "exit=$?"
tail -30 out/foldlisten_27bit.log
echo "ALLDONE_FOLDLISTEN_27B"
