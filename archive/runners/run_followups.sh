#!/usr/bin/env bash
# On-box: follow-up controls. (1) content-gated reselection, then (2) headset-specificity decollide. Both
# across 2b/9b/27b base, big pool. Weights already cached. Selftest gate first.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HF_TOKEN=$(cat .hf_token); export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

echo "=== selftest (1) contentgate ==="; python cave_doubt_contentgate.py --selftest || { echo "SELFTEST_FAIL_1"; exit 1; }
echo "=== selftest (2) headset_decollide ==="; python cave_headset_specificity_decollide.py --selftest || { echo "SELFTEST_FAIL_2"; exit 1; }

for ctrl in cave_doubt_contentgate cave_headset_specificity_decollide; do
  for spec in "google/gemma-2-2b 2b_base" "google/gemma-2-9b 9b_base" "google/gemma-2-27b 27b_base"; do
    set -- $spec; NAME=$1; TAG=$2
    echo "=== $ctrl $TAG ($NAME) big pool base qa ==="
    python "$ctrl.py" --device cuda --big-pool --name "$NAME" --tag "$TAG" > "out/${ctrl}_${TAG}.log" 2>&1
    echo "exit=$? ($ctrl $TAG)"; echo "--- tail ---"; tail -20 "out/${ctrl}_${TAG}.log"
  done
  echo "DONE_CONTROL_${ctrl}"
done
echo "ALLDONE_FOLLOWUPS"
