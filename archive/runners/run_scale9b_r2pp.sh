#!/usr/bin/env bash
# R-2'' (DESIGN_9b_scale_probes.md): scale9b_margin_pushback.py with robust answer extraction
# (first non-operand int -> confidence gate selects real low-margin items) + FORCED chat answer
# slot (it/base comparable, bare-caving decidable). it + base.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
echo "=== R-2'' selftest (model-free) ==="
python scale9b_margin_pushback.py --selftest; echo "selftest exit=$?"
echo "=== R-2'' margin pushback (9b-it, chat, forced slot) ==="
python scale9b_margin_pushback.py --name google/gemma-2-9b-it --tag 9b_it_v3 --chat > out/r2pp_it.log 2>&1; echo "it exit=$?"
echo "=== R-2'' margin pushback (9b base, fragment) ==="
python scale9b_margin_pushback.py --name google/gemma-2-9b    --tag 9b_base_v3       > out/r2pp_base.log 2>&1; echo "base exit=$?"
echo "ALLDONE_R2PP"
