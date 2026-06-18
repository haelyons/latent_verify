#!/usr/bin/env bash
# R-4 (DESIGN_9b_scale_probes.md): doubt-direction probe on the 9b-it bare-softening.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
echo "=== R-4 doubt-direction (9b-it) ==="
python scale9b_doubt_direction.py --name google/gemma-2-9b-it --tag 9b_it > out/r4_doubt.log 2>&1; echo "r4 exit=$?"
echo "ALLDONE_R4"
