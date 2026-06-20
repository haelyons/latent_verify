#!/usr/bin/env bash
# On-box: PART 7 de-confound -- cave_direction_xregime_deconfound. Decides whether Control 2's
# REGIME_SPECIFIC verdict (base/it cave-directions differ -> "RLHF reshapes the direction") is real or a
# proj_n residual-SCALE artifact. Splits cross-regime transfer into: WITHIN, CROSS_DONOR_PROJN (existing
# path), CROSS_HOST_PROJN (scale-corrected target), and MATCHED_ITEM (shared both-cave intersection fit).
# If host-projn cross/within >= 0.6 -> directions shared -> "base-intrinsic" reinstated; if still < 0.6 ->
# regime-specific real -> RLHF reshapes. Imports rlhf_differential/misconception_pool/job_truthful_flip.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_direction_xregime_deconfound.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b cave_direction_xregime_deconfound (base vs -it) ==="
python cave_direction_xregime_deconfound.py \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/cave_direction_xregime_9b.log 2>&1
echo "exit=$?"
echo "--- tail cave_direction_xregime_9b.log ---"; tail -50 out/cave_direction_xregime_9b.log
echo "ALLDONE_XREGIME"
