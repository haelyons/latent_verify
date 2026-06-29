#!/usr/bin/env bash
# On-box: Q1 causal confirmation (polarity-head ablation) at 9b. Selftest gate first.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HF_TOKEN=$(cat .hf_token); export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out
echo "=== selftest cave_polarity_causal ==="; python cave_polarity_causal.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))" || { echo TORCH_FAIL; exit 1; }
echo "=== cave_polarity_causal 9b_base ==="
python cave_polarity_causal.py --device cuda --big-pool --name google/gemma-2-9b --tag 9b_base > out/cave_polarity_causal_9b_base.log 2>&1
echo "exit=$?"; tail -25 out/cave_polarity_causal_9b_base.log
echo "ALLDONE_Q1"
