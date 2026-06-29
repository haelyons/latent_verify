#!/usr/bin/env bash
# On-box runner: calibration-vs-identity geometry + doubt-injection, gemma-2-27b base. Needs >=80GB GPU.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest (model-free) ==="
python cave_dir_calibration_geometry.py --selftest || { echo "SELFTEST_FAIL geometry"; exit 1; }
python cave_dir_doubt_injection.py --selftest || { echo "SELFTEST_FAIL injection"; exit 1; }

echo "=== ensure datasets ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"

echo "=== calibration geometry + regression, 27b base ==="
python cave_dir_calibration_geometry.py --device cuda --big-pool \
  --name google/gemma-2-27b --tag 27b_base \
  > out/cave_dir_calibration_geometry_27b_base.log 2>&1; echo "geom exit=$?"
echo "--- geom tail ---"; tail -50 out/cave_dir_calibration_geometry_27b_base.log

echo "=== doubt-injection specificity, 27b base ==="
python cave_dir_doubt_injection.py --device cuda --big-pool \
  --name google/gemma-2-27b --tag 27b_base \
  > out/cave_dir_doubt_injection_27b_base.log 2>&1; echo "inj exit=$?"
echo "--- inj tail ---"; tail -50 out/cave_dir_doubt_injection_27b_base.log

echo "=== collect ==="
mkdir -p out; cp -r results_calib/out/* out/ 2>/dev/null || true
ls -la out/ | tail -20
echo "ALLDONE_CALIB_27B"
