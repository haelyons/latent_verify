#!/usr/bin/env bash
# SEGMENTED 9b scale-probe test (DESIGN_9b_scale_probes.md). scale9b_* scripts ->
# out/scale9b_*.json, separate from the job_*/§10 lineage. Faithfulness anchor for this
# cue = S-1's effect gate reproducing the numeric_boundary_9b picture (large asserted-W
# shift on hard products). Not set -e: each stage attempts regardless. HF_TOKEN from env.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
B=google/gemma-2-9b; IT=google/gemma-2-9b-it
echo "=== S-1 numeric-copy battery (9b base) ==="
python scale9b_numeric_copy.py --name $B --tag 9b_base --sweep-n 12 > out/s9b_s1.log 2>&1; echo "s1 exit=$?"
echo "=== S-2 arith pushback (9b-it, chat) ==="
python scale9b_arith_pushback.py --name $IT --tag 9b_it --chat > out/s9b_s2it.log 2>&1; echo "s2it exit=$?"
echo "=== S-2 arith pushback (9b base, fragment control) ==="
python scale9b_arith_pushback.py --name $B --tag 9b_base > out/s9b_s2base.log 2>&1; echo "s2base exit=$?"
echo "ALLDONE_S9B"
