#!/usr/bin/env bash
# =================================================================================================
# run_forcedfinal_dist_2b9b.sh -- BOX A of REGISTRATION_forcedfinal_distributional.md (section 13):
# the forward-only forced-final distributional replay at 2b-base, 2b-it, 9b-base, 9b-it.
#
# One invocation = one cell = one model load (fresh process). FORWARD-ONLY: no model.generate exists
# in the instrument. --with-lp is ON per registered lean D-3 (19 forwards/item; the primary never
# reads the R-LP arm). The offline census ran BEFORE this box was launched (section 13.1 step 1) and
# the offline join runs AFTER the fetch; this box emits MEASUREMENTS only.
#
# Launch (section 13.6):
#   cp lambda_run.sh .launcher_ff2b9b.sh   # + add the instrument and the four source summaries to
#                                          #   the COPY's scp list; never edit lambda_run.sh
#   REMOTE_TIMEOUT=5400 bash .launcher_ff2b9b.sh gpu_1x_a100_sxm4 <region> \
#       run_forcedfinal_dist_2b9b.sh results_ff_2b9b
#
# Budget (section 13.5, box A): compute 2(66+66+132+132) ~= 13 min + venv ~10 min + 4 model pulls
# ~20 min (+ ~10 min for --with-lp) -> expected wall ~55 min; cap REMOTE_TIMEOUT=5400 (90 min).
#
# `set -uo pipefail`, NOT -e: per-cell exit capture (run_fmt_matched_2b9b.sh's stated reason).
# RUN_DONE is the launcher's marker; the per-cell exit= lines and the cellstatus tsv are the truth.
# =================================================================================================
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export LAMBDA_INSTANCE_ID="${LAMBDA_INSTANCE_ID:-}"
export GIT_COMMIT="${GIT_COMMIT:-}"
mkdir -p out
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} LAMBDA_INSTANCE_ID=${LAMBDA_INSTANCE_ID:-<EMPTY>} GIT_COMMIT=${GIT_COMMIT:-<EMPTY>}"

# --- section 11.2 pre-flight: a null is a failure, not a note. The instrument re-validates and
# --- raises before its model load; checking once here saves four identical aborts.
if [ -z "${LAMBDA_INSTANCE_ID}" ]; then
  echo "ABORT_PROVENANCE_INCOMPLETE: LAMBDA_INSTANCE_ID is empty (invoked outside the launcher)."
  echo "ALLDONE_FORCEDFINAL_2B9B (aborted before any model load)"
  exit 3
fi
[ -z "${GIT_COMMIT}" ] && echo "[warn] GIT_COMMIT empty: artifacts stamp git_commit null (present-but-null; not load-bearing)"

# --- selftests, model-free, FIRST (section 13.1 step 2). A launcher copy missing a file dies here. --
echo "=== selftests (model-free, CPU) ==="
python forcedfinal_dist.py --selftest || { echo "SELFTEST_FAIL_FORCEDFINAL_DIST (missing from the launcher copy's scp list?)"; exit 1; }
python foldlisten_judge.py --selftest  || { echo "SELFTEST_FAIL_FOLDLISTEN_SHIPPED"; exit 1; }
python family_topk_shift.py --selftest || { echo "SELFTEST_FAIL_TOPK_SHIPPED"; exit 1; }

# --- per-cell driver: ONE invocation = ONE cell, own model load, own log, own captured exit code ---
CELLSTATUS=out/forcedfinal_cellstatus_2b9b.tsv
: > "$CELLSTATUS"
cell () {
  local lab="$1"; shift
  local rc=0
  echo "=== CELL $lab :: $* ==="
  "$@" > "out/$lab.log" 2>&1 || rc=$?
  echo "exit=$rc"
  tail -4 "out/$lab.log"
  printf '%s\t%s\n' "$lab" "$rc" >> "$CELLSTATUS"
}

cell ff_ext2_2bbase python forcedfinal_dist.py --source foldlisten_judge_fl_2bbase_ext2_summary.json \
  --name google/gemma-2-2b    --tag ff_ext2_2bbase --device cuda --with-lp
cell ff_ext2_2bit   python forcedfinal_dist.py --source foldlisten_judge_fl_2bit_ext2_summary.json \
  --name google/gemma-2-2b-it --tag ff_ext2_2bit   --device cuda --chat --with-lp
cell ff_ext2_9bbase python forcedfinal_dist.py --source foldlisten_judge_fl_9bbase_ext2_summary.json \
  --name google/gemma-2-9b    --tag ff_ext2_9bbase --device cuda --with-lp
cell ff_ext2_9bit   python forcedfinal_dist.py --source foldlisten_judge_fl_9bit_ext2_summary.json \
  --name google/gemma-2-9b-it --tag ff_ext2_9bit   --device cuda --chat --with-lp

echo "=== cell status ==="
cat "$CELLSTATUS"
python - <<'PY'
import json
rows = [l.split("\t") for l in open("out/forcedfinal_cellstatus_2b9b.tsv").read().splitlines() if l]
json.dump({"box": "ff_2b9b", "cells": {r[0]: int(r[1]) for r in rows}}, open("out/forcedfinal_box_2b9b_summary.json", "w"), indent=2)
PY
echo "ALLDONE_FORCEDFINAL_2B9B"
