#!/usr/bin/env bash
# On-box Phase-B scale-out, >=80GB box (27b bf16 ~54GB, loaded one-at-a-time): the ext2 (n=82) cells at
# 27b — base then -it. Ported judge records commit_* + faithful_* labels with scorer_provenance (see
# NOTE_faithful_matcher.md 2026-07-21 addendum). Gate v2 on the -it summary under BOTH --labels readings.
# Files FLAT in ~/latent_verify.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

echo "=== selftests (model-free) ==="
python foldlisten_judge.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
python faithful_rescore.py --selftest || { echo "RESCORE_SELFTEST_FAIL"; exit 1; }

echo "=== 27b base (qa) ext2 ==="
python foldlisten_judge.py --family verifier_family_ext2.json --name google/gemma-2-27b --tag fl_27bbase_ext2 \
  --device cuda > out/foldlisten_27bbase_ext2.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_27bbase_ext2.log

echo "=== 27b-it (chat) ext2 ==="
python foldlisten_judge.py --family verifier_family_ext2.json --name google/gemma-2-27b-it --tag fl_27bit_ext2 \
  --device cuda --chat > out/foldlisten_27bit_ext2.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_27bit_ext2.log

echo "=== GATE v2 (pure) on the fresh -it summary, both label readings ==="
python foldlisten_judge.py --gate out/foldlisten_judge_fl_27bit_ext2_summary.json --v2 \
  2>&1 | tee out/foldlisten_gate_ext2_27b_commit.log
python foldlisten_judge.py --gate out/foldlisten_judge_fl_27bit_ext2_summary.json --v2 --labels faithful \
  2>&1 | tee out/foldlisten_gate_ext2_27b_faithful.log

echo "ALLDONE_FOLDLISTEN_EXT2_27B"
