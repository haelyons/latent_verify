#!/usr/bin/env bash
# On-box: NEXT-1 direction prong on gemma-2-9b base + -it. Fits a rank-1 cave/defer direction (diff-of-
# means counter vs neutral_turn) over a layer sweep, tests necessity (ablate) + sufficiency (steer) +
# low-rank (SVD) + base differential + set<->direction unification (cosine of the head-set's write with
# the direction). Forward-only -> fits the 40GB A100. Imports rlhf_differential + atp_low_confirm.HEADS +
# headset_joint_patch (all scp'd flat by lambda_run.sh).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python headset_direction.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b cave-direction prong (base + it): necessity / sufficiency / low-rank / set-unification ==="
python headset_direction.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it > out/headset_direction_9b.log 2>&1
echo "exit=$?"
echo "--- tail headset_direction_9b.log ---"; tail -60 out/headset_direction_9b.log
echo "ALLDONE_DIRECTION"
