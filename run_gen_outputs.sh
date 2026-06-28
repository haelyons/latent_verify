#!/usr/bin/env bash
# On-box: generate the OUTPUTS TABLE for the post (Sun item, neutral + counter turns) at
# 2b/9b/27b x base/-it. Faithful prompt format from job_truthful_flip.py. No new measurement,
# just greedy generation + first-token margin readout. Needs an >=80GB box for 27b bf16 (~54GB).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest (model-free gate) ==="
python gen_outputs_table.py --selftest || { echo "SELFTEST_FAIL_GEN"; exit 1; }

echo "=== generate 6 cells (2b/9b/27b x base/-it) ==="
python gen_outputs_table.py --device cuda --max-new-tokens 64 > out/gen_outputs_table.log 2>&1; echo "exit=$?"
echo "--- tail gen_outputs_table.log ---"; tail -100 out/gen_outputs_table.log

echo "ALLDONE_GEN_OUTPUTS"
