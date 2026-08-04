#!/usr/bin/env bash
# =================================================================================================
# run_forcedfinal_dist_27b.sh -- BOX B of REGISTRATION_forcedfinal_distributional.md (section 13):
# the forward-only forced-final distributional replay at 27b-base and 27b-it. Single >=80GB GPU
# (gemma-2-27b bf16 ~54GB resident); gpu_1x_h100_sxm5 per section 13.5 / D-6.
#
# Every 27b number this box produces is quotable ONLY under section 10's four-part disclosure:
# (i) lambda_instance_id + started_utc (stamped per artifact, section 11.1); (ii) the box class --
# card AND driver (the nvidia-smi line below + the provenance stamp); (iii) the SOURCE run's hardware
# is unrecoverable (section 3.4, stamped in source_provenance); (iv) the measured cross-box lp spread
# (median 0.009-0.13, max 0.44-0.59 nats). The replay-fidelity gate is the SIGN-FLIP count, not a
# magnitude tolerance, for exactly this reason.
#
# Launch (section 13.6):
#   cp lambda_run.sh .launcher_ff27b.sh    # + add the instrument and the two 27b source summaries
#   REMOTE_TIMEOUT=7200 bash .launcher_ff27b.sh gpu_1x_h100_sxm5 <region> \
#       run_forcedfinal_dist_27b.sh results_ff_27b
#
# Budget (section 13.5, box B): compute 2(267+267) ~= 18 min (+ ~13 min --with-lp) + venv ~10 min +
# 2 model pulls ~110GB ~35 min -> expected wall ~78 min; cap REMOTE_TIMEOUT=7200 (120 min).
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

if [ -z "${LAMBDA_INSTANCE_ID}" ]; then
  echo "ABORT_PROVENANCE_INCOMPLETE: LAMBDA_INSTANCE_ID is empty (invoked outside the launcher)."
  echo "ALLDONE_FORCEDFINAL_27B (aborted before any model load)"
  exit 3
fi
[ -z "${GIT_COMMIT}" ] && echo "[warn] GIT_COMMIT empty: artifacts stamp git_commit null (present-but-null; not load-bearing)"

echo "=== selftests (model-free, CPU) ==="
python forcedfinal_dist.py --selftest || { echo "SELFTEST_FAIL_FORCEDFINAL_DIST (missing from the launcher copy's scp list?)"; exit 1; }
python foldlisten_judge.py --selftest  || { echo "SELFTEST_FAIL_FOLDLISTEN_SHIPPED"; exit 1; }
python family_topk_shift.py --selftest || { echo "SELFTEST_FAIL_TOPK_SHIPPED"; exit 1; }

CELLSTATUS=out/forcedfinal_cellstatus_27b.tsv
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

cell ff_ext2_27bbase python forcedfinal_dist.py --source foldlisten_judge_fl_27bbase_ext2_summary.json \
  --name google/gemma-2-27b    --tag ff_ext2_27bbase --device cuda --with-lp
cell ff_ext2_27bit   python forcedfinal_dist.py --source foldlisten_judge_fl_27bit_ext2_summary.json \
  --name google/gemma-2-27b-it --tag ff_ext2_27bit   --device cuda --chat --with-lp

echo "=== cell status ==="
cat "$CELLSTATUS"
python - <<'PY'
import json
rows = [l.split("\t") for l in open("out/forcedfinal_cellstatus_27b.tsv").read().splitlines() if l]
json.dump({"box": "ff_27b", "cells": {r[0]: int(r[1]) for r in rows}}, open("out/forcedfinal_box_27b_summary.json", "w"), indent=2)
PY
echo "ALLDONE_FORCEDFINAL_27B"
