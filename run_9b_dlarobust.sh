#!/usr/bin/env bash
# On-box: PART 7 NEXT-(1) de-confound -- cave_direction_dla_robust. Hardens the MLP-band localization vs the
# DLA-at-resid_post[L] positional artifact: readout sweep L{28,32,36,40} (does a STABLE component set top the
# ranking ACROSS readouts vs a band that slides with L), magnitude-residualized ranking (|delta| residualized
# on |c_base|, removes the bigger-base->bigger-delta trend), and item-bootstrap CIs (n=9-10). POSITION_STABLE_
# LOCALIZATION iff a >=3-component set tops the normalized ranking across readouts with CI lower bound >= 0.5.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_direction_dla_robust.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b cave_direction_dla_robust (base vs -it, readout sweep + residualized + bootstrap) ==="
python cave_direction_dla_robust.py \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/cave_direction_dla_robust_9b.log 2>&1
echo "exit=$?"
echo "--- tail cave_direction_dla_robust_9b.log ---"; tail -55 out/cave_direction_dla_robust_9b.log
echo "ALLDONE_DLAROBUST"
