#!/usr/bin/env bash
# SEGMENTED Round-2 9b probes (DESIGN_9b_scale_probes.md). scale9b_* -> out/scale9b_*.json.
# R-1 dose-response (9b base) + R-2 capability-margin counter/bare (9b-it chat + 9b base).
# Not set -e. HF_TOKEN from env.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
B=google/gemma-2-9b; IT=google/gemma-2-9b-it
echo "=== R-1 dose-response (9b base, large n) ==="
python scale9b_dose_response.py --name $B --tag 9b_base --sweep-n 20 > out/r2_dose.log 2>&1; echo "dose exit=$?"
echo "=== R-2 margin pushback (9b-it, chat) ==="
python scale9b_margin_pushback.py --name $IT --tag 9b_it --chat > out/r2_mpush_it.log 2>&1; echo "mpush_it exit=$?"
echo "=== R-2 margin pushback (9b base, fragment) ==="
python scale9b_margin_pushback.py --name $B --tag 9b_base > out/r2_mpush_base.log 2>&1; echo "mpush_base exit=$?"
echo "ALLDONE_R2"
