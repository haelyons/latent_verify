#!/usr/bin/env bash
# SEQUENCE_170626 runner: N-1 recurrence repro -> P-A/P-B lowconf base -> lowconf it.
# Not set -e: each stage should attempt even if a prior one fails. HF_TOKEN comes from env.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
echo "=== recurrence (N-1 repro) ==="
python job_recurrence.py > out/rec.log 2>&1; echo "rec exit=$?"
echo "=== lowconf base (P-A/P-B) ==="
python job_sycophancy.py --items sycophancy_items_lowconf.json --model base --tag lowconf_base > out/lc_base.log 2>&1; echo "lc_base exit=$?"
echo "=== lowconf it (P-A/P-B) ==="
python job_sycophancy.py --items sycophancy_items_lowconf.json --model it --tag lowconf_it > out/lc_it.log 2>&1; echo "lc_it exit=$?"
echo "ALLDONE"
