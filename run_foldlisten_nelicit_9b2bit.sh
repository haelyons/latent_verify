#!/usr/bin/env bash
# On-box NEUTRAL-ARM ELICITATION, box 2 (>=40GB tier, expect gpu_1x_a100_sxm4). Pre-registered
# DESIGN_neutral_elicit.md sec 3.3: the two -it ext2 (n=82) cells at this box's sizes -- 9b-it then 2b-it.
# These are the NEGATIVE control for the whole run: the -it push arm withholds on 0-1 of 82, so sec 2.3
# pre-registers the neutral-elicited -it columns as NO_EFFECT_TO_EXPLAIN, and a -it cell that DOES withhold
# in the neutral arm is the one way claim (iii) gets touched. 9b-it also earns its own repro debt: its
# committed twin (results_foldlisten_r2/out/foldlisten_judge_fl_9bit_ext2_summary.json) is PRE-PORT and
# carries no faithful_* fields, so this is 9b-it ext2's first natively dual-labelled run (its faithful side
# compares against out/faithful_rescore_fl_9bit_ext2.json, not against the summary).
# Patched judge (neutral-elicited 4th arm + push_attribution{,_faithful} per record/summary). Tags
# DELIBERATELY IDENTICAL to the committed ones so the sec 5 byte-identity diff is a same-filename
# comparison. Gate v2 on BOTH -it summaries under BOTH --labels readings -- including the 27b-it-style
# contest shape: a reading that silently changes is itself a repro failure. Files FLAT in ~/latent_verify.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

echo "=== selftests (model-free) ==="
python foldlisten_judge.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
python faithful_rescore.py --selftest || { echo "RESCORE_SELFTEST_FAIL"; exit 1; }

echo "=== 9b-it (chat) ext2 ==="
python foldlisten_judge.py --family verifier_family_ext2.json --name google/gemma-2-9b-it --tag fl_9bit_ext2 \
  --device cuda --chat > out/foldlisten_nelicit_9bit_ext2.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_nelicit_9bit_ext2.log

echo "=== 2b-it (chat) ext2 ==="
python foldlisten_judge.py --family verifier_family_ext2.json --name google/gemma-2-2b-it --tag fl_2bit_ext2 \
  --device cuda --chat > out/foldlisten_nelicit_2bit_ext2.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_nelicit_2bit_ext2.log

echo "=== GATE v2 (pure) on both fresh -it summaries, both label readings ==="
python foldlisten_judge.py --gate out/foldlisten_judge_fl_9bit_ext2_summary.json \
  out/foldlisten_judge_fl_2bit_ext2_summary.json --v2 \
  2>&1 | tee out/foldlisten_gate_nelicit_2b9bit_commit.log
python foldlisten_judge.py --gate out/foldlisten_judge_fl_9bit_ext2_summary.json \
  out/foldlisten_judge_fl_2bit_ext2_summary.json --v2 --labels faithful \
  2>&1 | tee out/foldlisten_gate_nelicit_2b9bit_faithful.log

echo "ALLDONE_FOLDLISTEN_NELICIT_9B2BIT"
