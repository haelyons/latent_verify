#!/usr/bin/env bash
# On-box runner: mechanism (gate->u + trajectory) + fine-dose steering, gemma-2-2b base.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest (model-free) ==="
python cave_dir_mechanism.py --selftest || { echo "SELFTEST_FAIL mechanism"; exit 1; }
python cave_dir_dose_finegrained.py --selftest || { echo "SELFTEST_FAIL dose"; exit 1; }

echo "=== ensure datasets ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"

echo "=== mechanism (gate->u + trajectory), 2b base ==="
python cave_dir_mechanism.py --device cuda --big-pool --name google/gemma-2-2b --tag 2b_base \
  > out/cave_dir_mechanism_2b_base.log 2>&1; echo "mech exit=$?"
echo "--- mech tail ---"; tail -60 out/cave_dir_mechanism_2b_base.log

echo "=== fine-dose steering, 2b base ==="
python cave_dir_dose_finegrained.py --device cuda --big-pool --name google/gemma-2-2b --tag 2b_base \
  > out/cave_dir_dose_finegrained_2b_base.log 2>&1; echo "dose exit=$?"
echo "--- dose tail ---"; tail -60 out/cave_dir_dose_finegrained_2b_base.log

echo "=== collect ==="
mkdir -p out; cp -r results_calib/out/* out/ 2>/dev/null || true
ls -la out/ | tail -20
echo "ALLDONE_MECH_2B"
