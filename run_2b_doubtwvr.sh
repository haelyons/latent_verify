#!/usr/bin/env bash
# On-box (c) CROSS-SCALE: the doubt write-vs-read resolution at 2b base. Does the head-specific doubt circuit
# (read attention-KO + write output-patch) replicate at 2b? Same control, --name gemma-2-2b. big pool.
set -uo pipefail
cd ~/latent_verify; . .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== selftest ==="; python cave_doubt_write_vs_read.py --selftest || { echo SELFTEST_FAIL; exit 1; }
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -2 || echo DATASETS_PIP_FAILED
echo "=== 2b doubt write-vs-read, big pool ==="
python cave_doubt_write_vs_read.py --device cuda --big-pool --name google/gemma-2-2b --tag 2b_base > out/cave_doubt_write_vs_read_2b_base.log 2>&1
echo "exit=$?"; echo "--- tail ---"; tail -40 out/cave_doubt_write_vs_read_2b_base.log; echo ALLDONE_DOUBTWVR_2B
