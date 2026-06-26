#!/usr/bin/env bash
# On-box INTERVENTION (1) PHASE 2: independent JUDGE PANEL on the SAVED big-pool generations (panel_gens.json) +
# reader-GOLD (panel_gold.json). Scores each saved reply with Qwen2.5-7B / Mistral-7B / Llama-3.1-8B (different
# families; loaded one at a time; skip-on-failure for gated models), builds the agreement matrix vs
# self-judge/matcher/gold, panel = majority of independent judges, re-AUROCs the cave-direction under each label.
# No gemma reload. Files land FLAT in ~/latent_verify (scp convention).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_judge_panel.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
echo "=== judge panel on saved gens (panel_gens.json) + reader-gold (panel_gold.json) ==="
python cave_judge_panel.py --gens panel_gens.json --gold panel_gold.json --device cuda \
  --judges Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3 \
  > out/cave_judge_panel.log 2>&1; echo "exit=$?"
echo "--- tail ---"; tail -45 out/cave_judge_panel.log
echo "ALLDONE_JUDGE_PANEL"
