#!/usr/bin/env bash
# On-box Phase-B scale-out (>=80GB box): 27b-it ext2 only — the cell the first 27b box's 3.5h cap cut
# off (27b-base ext2 was banked by run_foldlisten_ext2_27b.sh). Ported judge (commit_* + faithful_*
# labels, scorer_provenance). Gate v2 under BOTH --labels readings. Files FLAT in ~/latent_verify.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

echo "=== selftests (model-free) ==="
python foldlisten_judge.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
python faithful_rescore.py --selftest || { echo "RESCORE_SELFTEST_FAIL"; exit 1; }

echo "=== 27b-it (chat) ext2 ==="
python foldlisten_judge.py --family verifier_family_ext2.json --name google/gemma-2-27b-it --tag fl_27bit_ext2 \
  --device cuda --chat > out/foldlisten_27bit_ext2.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_27bit_ext2.log

echo "=== GATE v2 (pure) on the fresh -it summary, both label readings ==="
python foldlisten_judge.py --gate out/foldlisten_judge_fl_27bit_ext2_summary.json --v2 \
  2>&1 | tee out/foldlisten_gate_ext2_27bit_commit.log
python foldlisten_judge.py --gate out/foldlisten_judge_fl_27bit_ext2_summary.json --v2 --labels faithful \
  2>&1 | tee out/foldlisten_gate_ext2_27bit_faithful.log

echo "ALLDONE_FOLDLISTEN_EXT2_27BIT"
