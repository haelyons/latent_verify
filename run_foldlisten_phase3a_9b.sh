#!/usr/bin/env bash
# On-box PHASE 3a at 9b-it on the frozen 74-item mechanism family: the three owed instrument patches
# (A1 real-5-turn prefix assert + full prompt storage; A2 masked W*-stated neutral floor -> re-read the
# Phase-2 listen KO; A3 neutral-arm DLA baseline vs the committed fold/listen profiles) + handle
# DERIVATION ONLY (read-side greedy head subsets on the DERIVE half; write-side per-layer diff-of-means
# directions L28-37), frozen to out/phase3_handles_* for Phase 3b. NO cross-transport / one-lever
# decision is evaluated here (pre-registration boundary). Files FLAT in ~/latent_verify.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

echo "=== selftests (model-free) ==="
python foldlisten_judge.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
python foldlisten_phase3a.py --selftest || { echo "P3A_SELFTEST_FAIL"; exit 1; }

echo "=== phase 3a: patches + handle derivation on mechanism_family_9bit.json (n=74) ==="
python foldlisten_phase3a.py --run --family mechanism_family_9bit.json \
  --p2-summary foldlisten_phase2_p2_9bit_summary.json \
  --name google/gemma-2-9b-it \
  --tag p3a_9bit --device cuda --chat > out/foldlisten_phase3a_9bit.log 2>&1; echo "exit=$?"
tail -12 out/foldlisten_phase3a_9bit.log

echo "ALLDONE_FOLDLISTEN_PHASE3A_9B"
