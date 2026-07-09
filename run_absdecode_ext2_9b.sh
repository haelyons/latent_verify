#!/usr/bin/env bash
# BASE-DECODE the ext2 expansion family (82 items) at gemma-2-9b BASE -- closes two POST1 gaps at once:
# (1) the decoded-abstention result so far rests on the ORIGINAL 22 items only (ext/r2 ran 9b-it only);
# (2) family_cave_diagnose now persists the teacher-forced lp components (lpC_*/lpW_*), so the same run
# makes the margin decomposition (did P(C) fall or P(W*) rise?) auditable per item for the first time.
# Decision rules live in the controls (neutral, unchanged); this runner is thin. Files FLAT in ~/latent_verify.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out
echo "=== selftests (model-free) ==="
python family_cave_diagnose.py --selftest || { echo "SELFTEST_FAIL_DIAGNOSE"; exit 1; }
python family_generate_judge.py --selftest || { echo "SELFTEST_FAIL_JUDGE"; exit 1; }
python family_topk_shift.py --selftest || { echo "SELFTEST_FAIL_TOPK"; exit 1; }
echo "=== 9b base top-K shift (ORIGINAL 22 items: is the curated W* in the model's bare top-K, and what rises under the counter turn) ==="
python family_topk_shift.py --family verifier_family --name google/gemma-2-9b --tag vfam_9bbase --device cuda \
  > out/family_topk_shift_vfam_9bbase.log 2>&1; echo "exit=$?"
tail -10 out/family_topk_shift_vfam_9bbase.log
echo "=== 9b base top-K shift (ext2, 82 items) ==="
python family_topk_shift.py --family verifier_family_ext2.json --name google/gemma-2-9b --tag vfam_ext2_9bbase --device cuda \
  > out/family_topk_shift_vfam_ext2_9bbase.log 2>&1; echo "exit=$?"
tail -10 out/family_topk_shift_vfam_ext2_9bbase.log
echo "=== 9b base diagnose (ext2, 82 items; persists lp components) ==="
python family_cave_diagnose.py --family verifier_family_ext2.json --name google/gemma-2-9b --tag vfam_ext2_9bbase --device cuda \
  > out/family_cave_diagnose_vfam_ext2_9bbase.log 2>&1; echo "exit=$?"
tail -20 out/family_cave_diagnose_vfam_ext2_9bbase.log
echo "=== 9b base generate+judge (ext2, 82 items; decoded replies) ==="
python family_generate_judge.py --family verifier_family_ext2.json --name google/gemma-2-9b --tag vfam_ext2_9bbase --device cuda \
  > out/family_generate_judge_vfam_ext2_9bbase.log 2>&1; echo "exit=$?"
tail -20 out/family_generate_judge_vfam_ext2_9bbase.log
echo "ALLDONE_ABSDECODE_EXT2_9B"
