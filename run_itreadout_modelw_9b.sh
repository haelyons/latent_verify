#!/usr/bin/env bash
# Two jobs riding one A100 box:
# (1) 9b-IT content-margin DECOMPOSITION (diagnose --chat, orig-22 + ext2-82): does -it realized
#     adoption come with lp(C) FALLING (displacement) or lp(W*) RISING over an intact C (as base
#     does, results_absdecode_ext2)? VALIDITY SCOPE (pre-run audit, 2026-07-11): at -it the
#     teacher-forced RC/lpC/lpW leg is SAFE; the first-token RA leg (P_w_*, RA_effect, faithful_RA)
#     is the known chat tail-token ghost — read the COMPONENTS, not the mixed decision category
#     (CONTENT_CAVES is the only category outcome that is valid at -it). family_topk_shift is
#     NO-GO at -it (neutral/counter first-token slots are preamble) and deliberately not run.
# (2) modelw_candidates at 9b-BASE (orig-22 + ext2-82): the model-derived W* candidate table
#     (bare top-K + greedy expansions) feeding DESIGN_modelderived_wstar.md; table producer, no verdict.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out
echo "=== selftests (model-free) ==="
python family_cave_diagnose.py --selftest || { echo "SELFTEST_FAIL_DIAGNOSE"; exit 1; }
python modelw_candidates.py --selftest || { echo "SELFTEST_FAIL_MODELW"; exit 1; }
echo "=== 9b-IT diagnose --chat (ORIGINAL 22; lp components are the payload, RA leg is ghost-contaminated) ==="
python family_cave_diagnose.py --family verifier_family --name google/gemma-2-9b-it --chat --tag vfam_9bit --device cuda \
  > out/family_cave_diagnose_vfam_9bit.log 2>&1; echo "exit=$?"
tail -5 out/family_cave_diagnose_vfam_9bit.log
echo "=== 9b-IT diagnose --chat (ext2, 82 items) ==="
python family_cave_diagnose.py --family verifier_family_ext2.json --name google/gemma-2-9b-it --chat --tag vfam_ext2_9bit --device cuda \
  > out/family_cave_diagnose_vfam_ext2_9bit.log 2>&1; echo "exit=$?"
tail -5 out/family_cave_diagnose_vfam_ext2_9bit.log
echo "=== 9b-BASE model-derived W* candidates (ORIGINAL 22) ==="
python modelw_candidates.py --family verifier_family --name google/gemma-2-9b --tag vfam_9bbase --device cuda \
  > out/modelw_candidates_vfam_9bbase.log 2>&1; echo "exit=$?"
tail -5 out/modelw_candidates_vfam_9bbase.log
echo "=== 9b-BASE model-derived W* candidates (ext2, 82 items) ==="
python modelw_candidates.py --family verifier_family_ext2.json --name google/gemma-2-9b --tag vfam_ext2_9bbase --device cuda \
  > out/modelw_candidates_vfam_ext2_9bbase.log 2>&1; echo "exit=$?"
tail -5 out/modelw_candidates_vfam_ext2_9bbase.log
echo "ALLDONE_ITREADOUT_MODELW_9B"
