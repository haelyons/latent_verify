#!/usr/bin/env bash
# R-2' (DESIGN_9b_scale_probes.md): clean re-run of the capability-margin counter/bare probe
# with the FIXED scale9b_margin_pushback.py (last-int extraction + external counter-W). it + base.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
echo "=== R-2' margin pushback (9b-it, chat) ==="
python scale9b_margin_pushback.py --name google/gemma-2-9b-it --tag 9b_it_v2 --chat > out/r2p_it.log 2>&1; echo "it exit=$?"
echo "=== R-2' margin pushback (9b base, fragment) ==="
python scale9b_margin_pushback.py --name google/gemma-2-9b    --tag 9b_base_v2       > out/r2p_base.log 2>&1; echo "base exit=$?"
echo "ALLDONE_R2P"
