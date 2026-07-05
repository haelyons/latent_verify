#!/usr/bin/env bash
# On-box PHASE 3a->3b->3c MECHANISM scale-transport at 2b-it (Phase 4; DESIGN_foldlisten_mechanism.md
# "Phase 4 scale-transport pre-registration", 2026-07-05). Re-derives the write-SET at the SCALE-RELATIVE
# write band L17-23 (= 9b's frozen L28-37 relative-depth analogue, fracs ~[0.667,0.905] of nL=26), runs the
# necessity+arbiter+backup verdict (greedy; --stage all sampled is infeasible as at 9b), and captures
# elicit-slot residuals for the OFFLINE in-domain THINK probe (run locally after fetch). Frozen 74-item
# mechanism_family_9bit.json transported as-is (per-scale fold-faithful subset emerges from the run; rates
# stay CONDITIONAL). Files FLAT in ~/latent_verify. All smokes FRONT-LOADED to gate the full burn.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

NAME=google/gemma-2-2b-it
WBLO=17; WBHI=24          # 2b nL=26: L17-23 = relative-depth analogue of 9b L28-37
FAM=mechanism_family_9bit.json

echo "=== selftests (model-free) ==="
python foldlisten_judge.py --selftest          || { echo "SELFTEST_FAIL judge"; exit 1; }
python foldlisten_phase3a.py --selftest        || { echo "P3A_SELFTEST_FAIL"; exit 1; }
python foldlisten_phase3b.py --selftest        || { echo "P3B_SELFTEST_FAIL"; exit 1; }
python foldlisten_phase3c_riders.py --selftest || { echo "P3C_SELFTEST_FAIL"; exit 1; }

echo "=== SMOKES (cheap; gate the full burn) ==="
python foldlisten_phase3a.py --run --family $FAM --name $NAME --tag p3a_2bit_smoke \
  --write-band-lo $WBLO --write-band-hi $WBHI --device cuda --chat --n 8 > out/p3a_2bit_smoke.log 2>&1
rc=$?; grep -i "band" out/p3a_2bit_smoke.log | head -2
{ [ $rc -eq 0 ] && [ -f out/phase3_handles_p3a_2bit_smoke.json ]; } || { echo "SMOKE_3A_FAIL rc=$rc"; tail -30 out/p3a_2bit_smoke.log; exit 1; }

python foldlisten_phase3b.py --run --family $FAM --name $NAME \
  --handles out/phase3_handles_p3a_2bit_smoke.json --tag p3b_2bit_smoke \
  --device cuda --chat --n 8 --stage greedy > out/p3b_2bit_smoke.log 2>&1
rc=$?; grep -i "EVAL" out/p3b_2bit_smoke.log | head -2
{ [ $rc -eq 0 ] && [ -f out/foldlisten_phase3b_p3b_2bit_smoke_summary.json ]; } || { echo "SMOKE_3B_FAIL rc=$rc"; tail -30 out/p3b_2bit_smoke.log; exit 1; }

python foldlisten_phase3c_riders.py --run --family $FAM --name $NAME --tag p3c_2bit_smoke \
  --device cuda --chat --n 6 > out/p3c_2bit_smoke.log 2>&1
rc=$?
{ [ $rc -eq 0 ] && [ -f out/foldlisten_phase3c_p3c_2bit_smoke_summary.json ]; } || { echo "SMOKE_3C_FAIL rc=$rc"; tail -30 out/p3c_2bit_smoke.log; exit 1; }
echo "ALL SMOKES OK"

echo "=== FULL 3a: re-derive write-SET at band L$WBLO-$((WBHI-1)) ==="
python foldlisten_phase3a.py --run --family $FAM --name $NAME --tag p3a_2bit \
  --write-band-lo $WBLO --write-band-hi $WBHI --device cuda --chat > out/foldlisten_phase3a_2bit.log 2>&1
echo "exit=$?"; tail -12 out/foldlisten_phase3a_2bit.log
[ -f out/phase3_handles_p3a_2bit.json ] || { echo "3A_FULL_NO_HANDLES"; exit 1; }

echo "=== FULL 3b: necessity+arbiter+backup (greedy) ==="
python foldlisten_phase3b.py --run --family $FAM --name $NAME \
  --handles out/phase3_handles_p3a_2bit.json --tag p3b_2bit \
  --device cuda --chat --stage greedy > out/foldlisten_phase3b_2bit.log 2>&1
echo "exit=$?"; tail -16 out/foldlisten_phase3b_2bit.log

echo "=== FULL 3c: riders capture (for offline in-domain probe) + C10/C11 ==="
python foldlisten_phase3c_riders.py --run --family $FAM --name $NAME --tag p3c_2bit \
  --device cuda --chat > out/foldlisten_phase3c_2bit.log 2>&1
echo "exit=$?"; tail -12 out/foldlisten_phase3c_2bit.log

echo "ALLDONE_FOLDLISTEN_MECH_2B"
