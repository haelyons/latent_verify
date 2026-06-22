#!/usr/bin/env bash
# On-box: PART 7 NEXT-(1) -- cave_direction_dla. Localize WHICH upstream components write the base->it
# change in the cave-direction (DLA per attn-head/MLP onto u_cave; delta_onbase = change along the fixed
# base cave-axis; c_it_ondelta = write along the RLHF-added axis). LOCALIZED_RESHAPE if top-10 carry >=50%
# of the differential, else DIFFUSE. Turns "RLHF reshapes the direction" into "RLHF changes [these senders]".
# Forward-only, last-token DLA cache -> light; imports rlhf_differential/headset_direction/etc.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_direction_dla.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b cave_direction_dla (base vs -it, L28/L32) ==="
python cave_direction_dla.py \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/cave_direction_dla_9b.log 2>&1
echo "exit=$?"
echo "--- tail cave_direction_dla_9b.log ---"; tail -55 out/cave_direction_dla_9b.log
echo "ALLDONE_DLA"
