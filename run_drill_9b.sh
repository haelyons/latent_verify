#!/usr/bin/env bash
# On-box batch: I1 headline -- truthful_flip 9b base + 9b-it (chat). Invoked via remote_run.sh.
# Re-localizes nothing hard-coded; the per-head concentration sweep (SC-B) runs inside the script
# on the caving items. HF_TOKEN from the env remote_run sets (gated gemma-2-9b{,-it}).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
pip install -q datasets 2>/dev/null
B=google/gemma-2-9b
echo "=== I1 truthful_flip 9b base ==="
python job_truthful_flip.py --truthfulqa --name $B --tag 9b_base > out/i1_9b_base.log 2>&1; echo "9b_base exit=$?"
echo "=== I1 truthful_flip 9b-it (chat) ==="
python job_truthful_flip.py --truthfulqa --name google/gemma-2-9b-it --tag 9b_it --chat > out/i1_9b_it.log 2>&1; echo "9b_it exit=$?"
echo "ALLDONE_9B"
