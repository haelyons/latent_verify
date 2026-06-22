#!/usr/bin/env bash
# On-box: C3-A numeric_generality, gemma-2-2b. The cross-scale contrast for C3: is 2b numeric
# CONCENTRATED (a reader) where 9b is diffuse? Same battery as the 9b arm. 208 heads -> faster sweep.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
echo "=== C3-A numeric_generality 2b ==="
python scale9b_numeric_generality.py --name google/gemma-2-2b --tag 2b > out/c3_generality_2b.log 2>&1; echo "gen exit=$?"
echo "--- tail c3_generality_2b.log ---"; tail -50 out/c3_generality_2b.log
echo "ALLDONE_C3A_2B"
