#!/usr/bin/env bash
# On-box: NEXT-1 head-SET joint activation-patch on gemma-2-9b base + -it. Tests whether the 9b-it
# misconception caving is carried by a JOINTLY-necessary head set that the per-head sweep (which returned
# NULL) structurally cannot see. Forward-only (no AtP backward) -> fits the 40GB A100. Imports the verified
# rlhf_differential._confirm pieces + ITEMS and atp_low_confirm.HEADS (all scp'd flat by lambda_run.sh).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python headset_joint_patch.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b head-set joint-patch (base + it): cumulative ramp + matched-random-K + super-additivity ==="
python headset_joint_patch.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it > out/headset_joint_9b.log 2>&1
echo "exit=$?"
echo "--- tail headset_joint_9b.log ---"; tail -50 out/headset_joint_9b.log
echo "ALLDONE_HEADSET"
