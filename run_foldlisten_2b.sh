#!/usr/bin/env bash
# On-box FOLD vs LISTEN (auditable: full completions to EOS, stored prompts, self-judge on the elicited final
# answer) -- gemma-2-2b base THEN 2b-it. Files land FLAT in ~/latent_verify; results to out/.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out
echo "=== selftest (model-free) ==="
python foldlisten_judge.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
echo "=== 2b base (qa) ==="
python foldlisten_judge.py --family verifier_family --name google/gemma-2-2b --tag fl_2bbase --device cuda \
  > out/foldlisten_2bbase.log 2>&1; echo "exit=$?"
tail -30 out/foldlisten_2bbase.log
echo "=== 2b-it (chat) ==="
python foldlisten_judge.py --family verifier_family --name google/gemma-2-2b-it --tag fl_2bit --device cuda --chat \
  > out/foldlisten_2bit.log 2>&1; echo "exit=$?"
tail -30 out/foldlisten_2bit.log
echo "ALLDONE_FOLDLISTEN_2B"
