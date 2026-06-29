#!/usr/bin/env bash
# On-box: mechanism controls at 9b. (A) polarity isolation, then (B) defer direction. Selftest gate first.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HF_TOKEN=$(cat .hf_token); export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

echo "=== selftest A polarity_isolation ==="; python cave_polarity_isolation.py --selftest || { echo "SELFTEST_FAIL_A"; exit 1; }
echo "=== selftest B defer_direction ==="; python cave_defer_direction.py --selftest || { echo "SELFTEST_FAIL_B"; exit 1; }
python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))" || { echo TORCH_FAIL; exit 1; }

echo "=== (A) polarity_isolation 9b_base ==="
python cave_polarity_isolation.py --device cuda --big-pool --name google/gemma-2-9b --tag 9b_base > out/cave_polarity_isolation_9b_base.log 2>&1
echo "exit=$? (A)"; tail -22 out/cave_polarity_isolation_9b_base.log
echo "DONE_A"

echo "=== (B) defer_direction 9b_base ==="
python cave_defer_direction.py --device cuda --big-pool --name google/gemma-2-9b --tag 9b_base > out/cave_defer_direction_9b_base.log 2>&1
echo "exit=$? (B)"; tail -22 out/cave_defer_direction_9b_base.log
echo "DONE_B"
echo "ALLDONE_MECH"
