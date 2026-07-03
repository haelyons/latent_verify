#!/usr/bin/env bash
# On-box PHASE 3b GREEDY-ONLY at 9b-it: the decisive verdict pass on the frozen EVAL half (37), writing
# the FULL groundable summary (per-item records incl. prompts + verdict + THINK/SAY matrices +
# handle_identity). The --stage all sampled block is compute-infeasible as specced (16 cells x 12 samples
# x 37 items ~= 14k generations ~= 35h; killed at cap 2026-07-03, greedy verdict already banked =
# MONITOR_AGAIN). Greedy is decisive here (arbiter sign-disagreement + backup restoration are
# rate-ceiling-independent); the sampled FRAGILE/ADD confirmation is owed as a cheap targeted re-spec, not
# this monster. ~1-1.5h. Files FLAT in ~/latent_verify.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

echo "=== selftests (model-free) ==="
python foldlisten_judge.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
python foldlisten_phase3b.py --selftest || { echo "P3B_SELFTEST_FAIL"; exit 1; }

echo "=== phase 3b GREEDY on the frozen 74 (EVAL=37) ==="
python foldlisten_phase3b.py --run --family mechanism_family_9bit.json \
  --handles phase3_handles_p3a_9bit.json \
  --p2-summary foldlisten_phase2_p2_9bit_summary.json \
  --name google/gemma-2-9b-it \
  --tag p3b_9bit --device cuda --chat --stage greedy > out/foldlisten_phase3b_greedy_9bit.log 2>&1; echo "exit=$?"
tail -14 out/foldlisten_phase3b_greedy_9bit.log

echo "ALLDONE_FOLDLISTEN_PHASE3B_GREEDY_9B"
