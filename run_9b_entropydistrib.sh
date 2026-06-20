#!/usr/bin/env bash
# On-box: PART 7 Control 1 -- entropy_distributed_presoftcap (group-ablation ramp + pre/post-softcap
# entropy). SC-EN-D1/D2: does the entropy NULL survive the distributed (group) grain and a pre-softcap
# readout? Self-contained (re-implements the null-space screen); needs `datasets` for the long-context ref.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python entropy_distributed_presoftcap.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== install datasets (for the long-context WikiText reference) ==="
pip install -q datasets || echo "DATASETS_INSTALL_FAILED (control will fall back to the short reference)"

echo "=== 9b entropy_distributed_presoftcap (group ramp + pre/post softcap, --ref long, --k 50, base vs -it) ==="
python entropy_distributed_presoftcap.py --device cuda --ref long --k 50 \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/entropy_distributed_9b.log 2>&1
echo "exit=$?"
echo "--- tail entropy_distributed_9b.log ---"; tail -45 out/entropy_distributed_9b.log
echo "ALLDONE_ENTROPYDISTRIB"
