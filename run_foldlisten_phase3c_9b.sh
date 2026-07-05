#!/usr/bin/env bash
# On-box PHASE 3c RIDERS at 9b-it on the frozen 74: capture pass for the A1 layer-sweep / B9 offline
# analysis (all-layer resid_post at the elicit slot: 4 nomask arms + masked fold arm + 2 stated
# contexts per item) + A6 padding-substitution fold arm (decision vs the CITED committed
# masked-neutral floor, matcher-v2 value 2/74) + C10 k=10 unpressured consistency column + C11
# few-shot P(True). Decisions here: A6 class + C10 report-flags ONLY — the A1 crossing verdict is a
# separate offline analysis over the npz (frozen rules: docs/NOTE_phase34_improvements_lit.md).
# Files FLAT in ~/latent_verify.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

echo "=== selftests (model-free) ==="
python foldlisten_judge.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
python foldlisten_phase3c_riders.py --selftest || { echo "P3C_SELFTEST_FAIL"; exit 1; }

echo "=== smoke: --n 6 (all parts, exit 0 + summary present) ==="
python foldlisten_phase3c_riders.py --run --family mechanism_family_9bit.json \
  --name google/gemma-2-9b-it --tag p3c_smoke --device cuda --chat --n 6 \
  --p2-floor 0.02702702702702703 > out/foldlisten_phase3c_smoke.log 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -f out/foldlisten_phase3c_p3c_smoke_summary.json ]; then
  echo "SMOKE_FAIL rc=$rc"; tail -30 out/foldlisten_phase3c_smoke.log; exit 1
fi
echo "smoke OK"

echo "=== phase 3c FULL: riders on the frozen 74 ==="
python foldlisten_phase3c_riders.py --run --family mechanism_family_9bit.json \
  --name google/gemma-2-9b-it --tag p3c_9bit --device cuda --chat \
  --p2-floor 0.02702702702702703 > out/foldlisten_phase3c_9bit.log 2>&1; echo "exit=$?"
tail -12 out/foldlisten_phase3c_9bit.log

echo "ALLDONE_FOLDLISTEN_PHASE3C_9B"
