#!/usr/bin/env bash
# SEQUENCE 9b SCALE ARM (gemma-2-9b base + -9b-it). The faithfulness gate
# (job_scale_mechanism.py --name google/gemma-2-9b) is run SEPARATELY and verified to
# reproduce FRAMING sec-10.1 (mean salience effect ~+0.02, max-attn-to-anchor ~0.42)
# BEFORE this batch runs. Every job re-localizes the reader from scratch via --reader auto
# (the 2b reader L18.H5 does NOT transfer to 9b; 42x16=672 heads). Not set -e: each stage
# attempts even if a prior one fails. HF_TOKEN from env (gated gemma-2-9b{,-it}).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
B=google/gemma-2-9b
echo "=== copyscore 9b base (N-3/N-4, sweep ALL 672 heads -> any OV-copy reader) ==="
python job_copyscore.py --name $B --tag 9b_base --reader auto --sweep > out/cs9b.log 2>&1; echo "cs exit=$?"
echo "=== localize 9b base (P-C/N-2, all 672 heads + router) ==="
python job_localize208.py --name $B --tag 9b_base --reader auto > out/loc9b.log 2>&1; echo "loc exit=$?"
echo "=== recurrence 9b base (N-1, name-mover vs induction; position de-confound) ==="
python job_recurrence.py --name $B --tag 9b_base --reader auto > out/rec9b.log 2>&1; echo "rec exit=$?"
echo "=== sycophancy lowconf 9b base (P-A/P-B, fragment) ==="
python job_sycophancy.py --items sycophancy_items_lowconf.json --name $B --tag lowconf_9b_base \
  --reader auto --sweep-layers all > out/lc9b_base.log 2>&1; echo "lc_base exit=$?"
echo "=== sycophancy lowconf 9b it (P-A/P-B, chat) ==="
python job_sycophancy.py --items sycophancy_items_lowconf.json --name google/gemma-2-9b-it \
  --tag lowconf_9b_it --chat --reader auto --sweep-layers all > out/lc9b_it.log 2>&1; echo "lc_it exit=$?"
echo "ALLDONE3"
