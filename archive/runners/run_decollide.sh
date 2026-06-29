#!/usr/bin/env bash
# On-box wrapper: readout-robustness control (RA first-token vs RB seq-margin vs RC stripped-margin) on the
# SAME doubt-circuit pipeline, three scales. Flat-scp layout (~/latent_verify). Gate: --selftest first, then
# 2b -> 9b -> 27b base, big pool. RA must reproduce the committed cave_doubt_write_vs_read restorations.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

echo "=== selftest (model-free) ==="
python cave_doubt_decollide.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== ensure datasets (--big-pool) ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"

echo "=== torch / cuda sanity ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" || { echo "TORCH_FAIL"; exit 1; }

for spec in "google/gemma-2-2b 2b_base" "google/gemma-2-9b 9b_base" "google/gemma-2-27b 27b_base"; do
  set -- $spec; NAME=$1; TAG=$2
  echo "=== decollide $TAG ($NAME), big pool, base qa ==="
  python cave_doubt_decollide.py --device cuda --big-pool --name "$NAME" --tag "$TAG" \
    > "out/cave_doubt_decollide_${TAG}.log" 2>&1
  echo "exit=$? ($TAG)"
  echo "--- tail $TAG ---"; tail -25 "out/cave_doubt_decollide_${TAG}.log"
done
echo "ALLDONE_DECOLLIDE"
