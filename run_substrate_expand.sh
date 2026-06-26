#!/usr/bin/env bash
# On-box SUBSTRATE EXPANSION + panel, ONE box. (1) multisample cave-rate on the FULL TruthfulQA (clean factual
# misconceptions, single dominant competitor -> many more near-tie items -> more genuine caves for tight CIs);
# saves all gens + resid + self-judge + matcher. (2) judge PANEL (Qwen+Mistral, different families) on those SAME
# gens (no gold yet -> reader curates offline afterwards). gemma freed before judges; sequential -> fits 40GB.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftests ==="
python cave_multisample_caverate.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
python cave_judge_panel.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"

echo "=== (1) multisample on full TruthfulQA, 9b base+it, item cap 80 ==="
python cave_multisample_caverate.py --device cuda --truthfulqa --n 80 \
  --base google/gemma-2-9b --it google/gemma-2-9b-it \
  > out/cave_multisample_caverate_tqa_9b.log 2>&1; echo "multisample exit=$?"
echo "--- multisample tail ---"; tail -25 out/cave_multisample_caverate_tqa_9b.log

echo "=== (2) judge panel on the expanded gens (Qwen + Mistral; gold added offline later) ==="
python cave_judge_panel.py --gens out/cave_multisample_caverate.json --gold __none__ --device cuda \
  --judges Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3 \
  > out/cave_judge_panel_tqa.log 2>&1; echo "panel exit=$?"
echo "--- panel tail ---"; tail -30 out/cave_judge_panel_tqa.log
echo "ALLDONE_SUBSTRATE_EXPAND"
