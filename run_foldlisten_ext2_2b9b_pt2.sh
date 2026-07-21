#!/usr/bin/env bash
# On-box Phase-B scale-out part 2 (A100 40GB): the two cells the first box's 2h cap cut off —
# 2b-it ext2 (first attempt died mid-cell) and 9b-base ext2 (never started). Anchor3 + 2b-base ext2
# were banked by run_foldlisten_ext2_2b9b.sh. Same ported judge (commit_* + faithful_* labels,
# scorer_provenance). Gate v2 on the fresh -it summary under BOTH --labels readings.
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

echo "=== 2b-it (chat) ext2 ==="
python foldlisten_judge.py --family verifier_family_ext2.json --name google/gemma-2-2b-it --tag fl_2bit_ext2 \
  --device cuda --chat > out/foldlisten_2bit_ext2.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_2bit_ext2.log

echo "=== 9b base (qa) ext2 ==="
python foldlisten_judge.py --family verifier_family_ext2.json --name google/gemma-2-9b --tag fl_9bbase_ext2 \
  --device cuda > out/foldlisten_9bbase_ext2.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_9bbase_ext2.log

echo "=== GATE v2 (pure) on the fresh -it summary, both label readings ==="
python foldlisten_judge.py --gate out/foldlisten_judge_fl_2bit_ext2_summary.json --v2 \
  2>&1 | tee out/foldlisten_gate_ext2_2bit_commit.log
python foldlisten_judge.py --gate out/foldlisten_judge_fl_2bit_ext2_summary.json --v2 --labels faithful \
  2>&1 | tee out/foldlisten_gate_ext2_2bit_faithful.log

echo "ALLDONE_FOLDLISTEN_EXT2_2B9B_PT2"
