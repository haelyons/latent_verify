#!/usr/bin/env bash
# On-box NEUTRAL-ARM ELICITATION, box 1 (>=40GB tier, expect gpu_1x_a100_sxm4). Pre-registered
# DESIGN_neutral_elicit.md sec 3.3: (1) the faithfulness anchor fl_9bit_anchor4 (9b-it, verifier_family
# n=22) -- sec 5 step 2 makes this BLOCKING: it must reproduce the committed
# results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_9bit_anchor3_summary.json character-for-character
# (fold 13/9/0, fold_rate 0.591, listen 21/0/1, agreement 36/44) or the run is substrate drift, not a
# finding -- then the two BASE ext2 (n=82) cells at this box's sizes: 9b-base, 2b-base.
# The judge is the PATCHED instrument (controls/foldlisten_judge.py:481-485): every record additionally
# carries the neutral-arm elicited 4th arm (neutral_elicit_prompt/_gen, commit_neutral_elicit,
# faithful_neutral_elicit strict per NOTE_faithful_matcher.md Addendum 1) and each summary carries
# push_attribution{,_faithful}. The new decode is placed AFTER every pre-existing generate() under greedy
# decoding, so the legacy fields must come back byte-identical -- that is the sec 5 gate, not an assumption.
# Tags are DELIBERATELY IDENTICAL to the committed ext2 ones so the byte-identity diff is a same-filename
# comparison against results_foldlisten_ext2_2b9b/out/. Gate v2 runs on the anchor summary under BOTH
# --labels readings (the two readings are distinct decisions and each must survive as its own artifact).
# Cell order is the pre-registered one: the blocking anchor is banked FIRST, so a cap kill can only cost
# the cell in flight (each cell writes its own summary as it finishes). Files FLAT in ~/latent_verify.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

echo "=== selftests (model-free) ==="
python foldlisten_judge.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
python faithful_rescore.py --selftest || { echo "RESCORE_SELFTEST_FAIL"; exit 1; }

echo "=== faithfulness anchor: 9b-it verifier_family (n=22, patched judge) ==="
python foldlisten_judge.py --family verifier_family --name google/gemma-2-9b-it --tag fl_9bit_anchor4 \
  --device cuda --chat > out/foldlisten_nelicit_9bit_anchor4.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_nelicit_9bit_anchor4.log

echo "=== 9b base (qa) ext2 ==="
python foldlisten_judge.py --family verifier_family_ext2.json --name google/gemma-2-9b --tag fl_9bbase_ext2 \
  --device cuda > out/foldlisten_nelicit_9bbase_ext2.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_nelicit_9bbase_ext2.log

echo "=== 2b base (qa) ext2 ==="
python foldlisten_judge.py --family verifier_family_ext2.json --name google/gemma-2-2b --tag fl_2bbase_ext2 \
  --device cuda > out/foldlisten_nelicit_2bbase_ext2.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_nelicit_2bbase_ext2.log

echo "=== GATE v2 (pure) on the fresh anchor summary, both label readings ==="
python foldlisten_judge.py --gate out/foldlisten_judge_fl_9bit_anchor4_summary.json --v2 \
  2>&1 | tee out/foldlisten_gate_nelicit_anchor4_commit.log
python foldlisten_judge.py --gate out/foldlisten_judge_fl_9bit_anchor4_summary.json --v2 --labels faithful \
  2>&1 | tee out/foldlisten_gate_nelicit_anchor4_faithful.log

echo "ALLDONE_FOLDLISTEN_NELICIT_9B2B"
