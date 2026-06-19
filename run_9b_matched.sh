#!/usr/bin/env bash
# On-box: matched-item de-confound on gemma-2-9b base + -it. Restricts to the caving intersection (both
# models gap>+0.5, sign-restricted), then re-measures the head-SET joint frac and the DIRECTION necessity
# frac on the SAME items, base vs it -- hardening (or overturning) the "amplified-not-installed" claim that
# both NEXT-1 prongs rest on. Forward-only -> 40GB A100. Imports rlhf_differential + atp_low_confirm.HEADS +
# headset_joint_patch + headset_direction (all scp'd flat by lambda_run.sh).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python matched_item_deconfound.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b matched-item de-confound (set + direction on the caving intersection, base vs it) ==="
python matched_item_deconfound.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it > out/matched_deconfound_9b.log 2>&1
echo "exit=$?"
echo "--- tail matched_deconfound_9b.log ---"; tail -40 out/matched_deconfound_9b.log
echo "ALLDONE_MATCHED"
