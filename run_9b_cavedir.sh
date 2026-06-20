#!/usr/bin/env bash
# On-box: PART 7 Control 2 -- cave_direction_heldout (held-out/LOO + cross-regime transfer + label-permuted
# null). SC-D1 / SC-D-XREG: is the 9b cave-direction real out-of-sample and base-intrinsic, or in-sample /
# regime-specific fit? Imports rlhf_differential / headset_direction / job_truthful_flip (scp'd flat).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_direction_heldout.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b cave_direction_heldout (base vs -it) ==="
python cave_direction_heldout.py \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/cave_direction_heldout_9b.log 2>&1
echo "exit=$?"
echo "--- tail cave_direction_heldout_9b.log ---"; tail -45 out/cave_direction_heldout_9b.log
echo "ALLDONE_CAVEDIR"
