#!/usr/bin/env bash
# On-box: C3 author-queue control A (numeric-generality), gemma-2-9b BASE. The 672-head sweep is
# the long part (~30-60 min). Built-in gate: the authority/discovery SC-1 cell must reproduce R2
# (nec_W ~0.9, ctrl ~0, out/scale9b_numeric_copy_9b_base.json) before trusting held-out/alt phrasings.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
echo "=== C3-A numeric_generality 9b base ==="
python scale9b_numeric_generality.py --name google/gemma-2-9b --tag 9b_base > out/c3_generality.log 2>&1; echo "gen exit=$?"
echo "--- tail c3_generality.log ---"; tail -50 out/c3_generality.log
echo "ALLDONE_C3A"
