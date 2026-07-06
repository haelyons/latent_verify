#!/usr/bin/env bash
# On-box PHASE 3b->3c RESUME at 2b-it, reusing the BANKED full-3a handles (phase3_handles_p3a_2bit.json/.npz,
# scp'd flat into ~/latent_verify) so 3a's ~2h derivation is NOT repeated. 3a already completed + was fetched
# (write-band L17-23 derived, read-side WEAK_AT_DERIVE both arms = echoes 9b). This runs the decision-bearing
# 3b (necessity + arbiter + backup, greedy) + the 3c riders capture (for the offline in-domain THINK probe).
# DESIGN_foldlisten_mechanism.md Phase-4 pre-reg. Files FLAT in ~/latent_verify.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

NAME=google/gemma-2-2b-it
FAM=mechanism_family_9bit.json
HANDLES=phase3_handles_p3a_2bit.json
[ -f "$HANDLES" ] && [ -f "${HANDLES%.json}.npz" ] || { echo "RESUME_NO_HANDLES: $HANDLES(.npz) missing -- was it scp'd?"; exit 1; }
echo "[resume] reusing banked 3a handles: $HANDLES (+ .npz)"

echo "=== selftests (model-free) ==="
python foldlisten_phase3b.py --selftest        || { echo "P3B_SELFTEST_FAIL"; exit 1; }
python foldlisten_phase3c_riders.py --selftest || { echo "P3C_SELFTEST_FAIL"; exit 1; }

echo "=== 3b SMOKE (--n 8, reuse banked handles) ==="
python foldlisten_phase3b.py --run --family $FAM --name $NAME --handles $HANDLES --tag p3b_2bit_smoke \
  --device cuda --chat --n 8 --stage greedy > out/p3b_2bit_resume_smoke.log 2>&1
rc=$?; grep -i "EVAL" out/p3b_2bit_resume_smoke.log | head -2
{ [ $rc -eq 0 ] && [ -f out/foldlisten_phase3b_p3b_2bit_smoke_summary.json ]; } || { echo "SMOKE_3B_FAIL rc=$rc"; tail -30 out/p3b_2bit_resume_smoke.log; exit 1; }

echo "=== 3c SMOKE (--n 6) ==="
python foldlisten_phase3c_riders.py --run --family $FAM --name $NAME --tag p3c_2bit_smoke \
  --device cuda --chat --n 6 > out/p3c_2bit_resume_smoke.log 2>&1
rc=$?
{ [ $rc -eq 0 ] && [ -f out/foldlisten_phase3c_p3c_2bit_smoke_summary.json ]; } || { echo "SMOKE_3C_FAIL rc=$rc"; tail -30 out/p3c_2bit_resume_smoke.log; exit 1; }
echo "ALL SMOKES OK"

echo "=== FULL 3b: necessity+arbiter+backup (greedy) ==="
python foldlisten_phase3b.py --run --family $FAM --name $NAME --handles $HANDLES --tag p3b_2bit \
  --device cuda --chat --stage greedy > out/foldlisten_phase3b_2bit.log 2>&1
echo "exit=$?"; tail -16 out/foldlisten_phase3b_2bit.log

echo "=== FULL 3c: riders capture (for offline in-domain probe) + C10/C11 ==="
python foldlisten_phase3c_riders.py --run --family $FAM --name $NAME --tag p3c_2bit \
  --device cuda --chat > out/foldlisten_phase3c_2bit.log 2>&1
echo "exit=$?"; tail -12 out/foldlisten_phase3c_2bit.log

echo "ALLDONE_FOLDLISTEN_MECH_2B_RESUME"
