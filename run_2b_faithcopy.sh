#!/usr/bin/env bash
# On-box: STEP 1 -- the one crack. 2b-it was the lone CLEAN copy-on-M case (necessity 0.83, control -0.15).
# Faithful re-test: does knocking all-heads attention-to-W* reduce the REALIZED P(W*) / move the argmax off W*
# at 2b? If FAITHFUL_COPY_OF_WSTAR -> the program's first behavior-faithful caving mechanism (at 2b). If M_ONLY
# -> caving has no localizable single-token mechanism at ANY scale.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python faithful_copy_wstar.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 2b faithful_copy_wstar (base qa + it chat) ==="
python faithful_copy_wstar.py --device cuda --chat \
  --name-base google/gemma-2-2b --name-it google/gemma-2-2b-it --tag 2b > out/faithful_copy_wstar_2b.log 2>&1
echo "exit=$?"
echo "--- tail faithful_copy_wstar_2b.log ---"; tail -50 out/faithful_copy_wstar_2b.log
echo "ALLDONE_FAITHCOPY_2B"
