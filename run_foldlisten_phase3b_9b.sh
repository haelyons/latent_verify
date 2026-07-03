#!/usr/bin/env bash
# On-box PHASE 3b at 9b-it: the Phase-3 core decision on the frozen 74-item family, consuming the
# FROZEN 3a handles (never re-derived; EVAL half only). Greedy stage decides (cross-transport necessity
# vs matched random floors + direct==total arbiter + backup check + THINK/SAY); sampled stage (temp 0.8
# n=12) quantifies + ADD/ceiling clause. Post-greedy checkpoint banks the verdict before the multi-hour
# sampled stage. SMOKE FIRST: --n 8 --stage greedy exercises the new swap/arbiter paths on a few EVAL
# items before the full burn (GPU-lens launch condition). Files FLAT in ~/latent_verify.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

echo "=== selftests (model-free) ==="
python foldlisten_judge.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
python foldlisten_phase3b.py --selftest || { echo "P3B_SELFTEST_FAIL"; exit 1; }

echo "=== smoke: --n 8 --stage greedy (must exercise EVAL>0 and exit 0) ==="
python foldlisten_phase3b.py --run --family mechanism_family_9bit.json \
  --handles phase3_handles_p3a_9bit.json \
  --p2-summary foldlisten_phase2_p2_9bit_summary.json \
  --name google/gemma-2-9b-it \
  --tag p3b_smoke --device cuda --chat --n 8 --stage greedy > out/foldlisten_phase3b_smoke.log 2>&1
rc=$?
grep -i "EVAL" out/foldlisten_phase3b_smoke.log | head -2
if [ $rc -ne 0 ] || [ ! -f out/foldlisten_phase3b_p3b_smoke_summary.json ]; then
  echo "SMOKE_FAIL rc=$rc"; tail -30 out/foldlisten_phase3b_smoke.log; exit 1
fi
echo "smoke OK"

echo "=== phase 3b FULL: --stage all on the frozen 74 (EVAL=37) ==="
python foldlisten_phase3b.py --run --family mechanism_family_9bit.json \
  --handles phase3_handles_p3a_9bit.json \
  --p2-summary foldlisten_phase2_p2_9bit_summary.json \
  --name google/gemma-2-9b-it \
  --tag p3b_9bit --device cuda --chat --stage all > out/foldlisten_phase3b_9bit.log 2>&1; echo "exit=$?"
tail -12 out/foldlisten_phase3b_9bit.log

echo "ALLDONE_FOLDLISTEN_PHASE3B_9B"
