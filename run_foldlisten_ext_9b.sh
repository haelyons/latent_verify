#!/usr/bin/env bash
# On-box screening + real-gate run for the EXPANDED fold/listen family, gemma-2-9b-it (chat).
# Order (README entry ritual first): model-free selftest -> FAITHFULNESS REPRODUCTION of the committed
# 9b-it verifier_family run (must reproduce 13/9/0 fold, 21/0/1 listen -> gate PASS 8/22, 36/44) -> the
# UNSEEN verifier_family_ext run -> --gate on BOTH fresh summaries. Greedy throughout (registered instrument).
# Files land FLAT in ~/latent_verify; results to out/.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

echo "=== selftest (model-free) ==="
python foldlisten_judge.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
python verifier_family_ext.py --selftest || { echo "EXT_SELFTEST_FAIL"; exit 1; }

echo "=== faithfulness reproduction: 9b-it verifier_family (n=22) ==="
python foldlisten_judge.py --family verifier_family --name google/gemma-2-9b-it --tag fl_9bit_repro \
  --device cuda --chat > out/foldlisten_9bit_repro.log 2>&1; echo "exit=$?"
tail -8 out/foldlisten_9bit_repro.log

echo "=== EXPANDED family (unseen): 9b-it verifier_family_ext (n=34) ==="
python foldlisten_judge.py --family verifier_family_ext --name google/gemma-2-9b-it --tag fl_9bit_ext \
  --device cuda --chat > out/foldlisten_9bit_ext.log 2>&1; echo "exit=$?"
tail -8 out/foldlisten_9bit_ext.log

echo "=== GATE (pure, no model) on both fresh summaries ==="
python foldlisten_judge.py --gate out/foldlisten_judge_fl_9bit_repro_summary.json \
  out/foldlisten_judge_fl_9bit_ext_summary.json 2>&1 | tee out/foldlisten_gate_run.log

echo "ALLDONE_FOLDLISTEN_EXT_9B"
