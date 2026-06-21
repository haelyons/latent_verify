#!/usr/bin/env bash
# On-box: PART 7 mechanism test A -- cave_direction_overlay. Is the cave-direction a MECHANISM or an OVERLAY
# (Makelov)? Ablate u_cave on caving items; MECHANISM_LIKE iff the effect is targeted (mass W*->C, target_frac
# >=0.5) AND off-regime-specific (neutral/non-caving items barely perturbed) AND beats a random direction;
# else OVERLAY_LIKE. Imports rlhf_differential/headset_direction/misconception_pool.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_direction_overlay.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b cave_direction_overlay (base vs -it) ==="
python cave_direction_overlay.py --device cuda \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/cave_direction_overlay_9b.log 2>&1
echo "exit=$?"
echo "--- tail cave_direction_overlay_9b.log ---"; tail -45 out/cave_direction_overlay_9b.log
echo "ALLDONE_OVERLAY"
