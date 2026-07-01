#!/usr/bin/env bash
# On-box FOLD vs LISTEN (auditable: full completions to EOS, stored prompts, self-judge on the elicited final
# answer) -- gemma-2-9b base THEN 9b-it. Files land FLAT in ~/latent_verify; results to out/.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out
echo "=== selftest (model-free) ==="
python foldlisten_judge.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
echo "=== 9b base (qa) ==="
python foldlisten_judge.py --family verifier_family --name google/gemma-2-9b --tag fl_9bbase --device cuda \
  > out/foldlisten_9bbase.log 2>&1; echo "exit=$?"
tail -30 out/foldlisten_9bbase.log
echo "=== 9b-it (chat) ==="
python foldlisten_judge.py --family verifier_family --name google/gemma-2-9b-it --tag fl_9bit --device cuda --chat \
  > out/foldlisten_9bit.log 2>&1; echo "exit=$?"
tail -30 out/foldlisten_9bit.log
echo "ALLDONE_FOLDLISTEN_9B"
