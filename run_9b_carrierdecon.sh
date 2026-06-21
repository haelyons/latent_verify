#!/usr/bin/env bash
# On-box: STEP 3(b/c) -- harden "u_cave is THE carrier of the pushback effect" vs the 2 open cruxes.
# On base Q/A argmax-W* caving items, compare 5 ablations: (A) u_cave->neutral-mean (original), (B) u_cave->0
# (zero-abl, tests CIRCULARITY), (C) u_cave->shuffled-neutral (resample, tests circularity), (D) orthogonalized
# IN-SHIFT direction->neutral-mean (tests IN-SHIFT SPECIFICITY), (E) isotropic-random (floor). HARDENED_CARRIER
# iff zero/resample still restore neutral (NOT_CIRCULAR) AND the in-shift orthogonal dir does NOT (SPECIFIC).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_carrier_deconfound.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b cave_carrier_deconfound (base qa + it chat) ==="
python cave_carrier_deconfound.py --device cuda --chat \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/cave_carrier_deconfound_9b.log 2>&1
echo "exit=$?"
echo "--- tail cave_carrier_deconfound_9b.log ---"; tail -55 out/cave_carrier_deconfound_9b.log
echo "ALLDONE_CARRIERDECON"
