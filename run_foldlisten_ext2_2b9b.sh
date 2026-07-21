#!/usr/bin/env bash
# On-box Phase-B scale-out, A100 40GB box: (1) faithfulness anchor (committed 9b-it n=22 repro, third
# anchor — confirms the PORTED judge reproduces the deterministic core byte-identically), then the ext2
# (n=82) cells absent from the matrix at this box's sizes: 2b-base, 2b-it, 9b-base. Judge now records BOTH
# label sets per item (commit_* at generation time + faithful_* via faithful_rescore.classify; elicit
# strict) with scorer_provenance embedded — see NOTE_faithful_matcher.md 2026-07-21 addendum. Gates v2 run
# on the -it summary under BOTH --labels readings. Files FLAT in ~/latent_verify.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

echo "=== selftests (model-free) ==="
python foldlisten_judge.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
python faithful_rescore.py --selftest || { echo "RESCORE_SELFTEST_FAIL"; exit 1; }

echo "=== faithfulness anchor: 9b-it verifier_family (n=22, ported judge) ==="
python foldlisten_judge.py --family verifier_family --name google/gemma-2-9b-it --tag fl_9bit_anchor3 \
  --device cuda --chat > out/foldlisten_9bit_anchor3.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_9bit_anchor3.log

echo "=== 2b base (qa) ext2 ==="
python foldlisten_judge.py --family verifier_family_ext2.json --name google/gemma-2-2b --tag fl_2bbase_ext2 \
  --device cuda > out/foldlisten_2bbase_ext2.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_2bbase_ext2.log

echo "=== 2b-it (chat) ext2 ==="
python foldlisten_judge.py --family verifier_family_ext2.json --name google/gemma-2-2b-it --tag fl_2bit_ext2 \
  --device cuda --chat > out/foldlisten_2bit_ext2.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_2bit_ext2.log

echo "=== 9b base (qa) ext2 ==="
python foldlisten_judge.py --family verifier_family_ext2.json --name google/gemma-2-9b --tag fl_9bbase_ext2 \
  --device cuda > out/foldlisten_9bbase_ext2.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_9bbase_ext2.log

echo "=== GATE v2 (pure) on the fresh -it summaries, both label readings ==="
python foldlisten_judge.py --gate out/foldlisten_judge_fl_9bit_anchor3_summary.json \
  out/foldlisten_judge_fl_2bit_ext2_summary.json --v2 2>&1 | tee out/foldlisten_gate_ext2_2b9b_commit.log
python foldlisten_judge.py --gate out/foldlisten_judge_fl_9bit_anchor3_summary.json \
  out/foldlisten_judge_fl_2bit_ext2_summary.json --v2 --labels faithful 2>&1 | tee out/foldlisten_gate_ext2_2b9b_faithful.log

echo "ALLDONE_FOLDLISTEN_EXT2_2B9B"
