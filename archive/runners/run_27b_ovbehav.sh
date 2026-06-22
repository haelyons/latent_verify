#!/usr/bin/env bash
# On-box: NEXT-3b -- behavioral test of the 27b copy-head OV-gain change. Forward passes on -it 27b:
# per head, scale hook_z by 1/alpha (restore base OV magnitude) and knockout (z=0), on random-token
# induction prompts; measure delta logit of the copied token. Needs an 80GB GPU (27b forward).
# ov_behavioral_scale.py scp'd flat. alpha values are from results_27b_ovmag (committed).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python ov_behavioral_scale.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 27b OV-gain behavioral scale-ablation (-it, induction, GPU) ==="
python ov_behavioral_scale.py --device cuda --name-it google/gemma-2-27b-it --tag 27b > out/ov_behavioral_27b.log 2>&1
echo "exit=$?"
echo "--- tail ov_behavioral_27b.log ---"; tail -40 out/ov_behavioral_27b.log
echo "ALLDONE_OVBEHAV"
