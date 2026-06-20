#!/usr/bin/env bash
# On-box: cross-scale caving check -- does gemma-2-2b (base + it) cave on the TruthfulQA misconception
# substrate the way 9b does (9b ref: 9-14 flips)? Answers "is caving a 9b phenomenon, or also at 2b?" and
# gates the attribution-graph route (circuit-tracer fully supports 2b; 9b needs sourced transcoders).
# job_truthful_flip.py reports per-item pre-margin, cap_counter (>0 = caved toward W*), parrot_state
# ("flipped" = caved C->W*), and the SC-B per-head knockout sweep on flips (does a 2b flip recruit L18.H5?).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python job_truthful_flip.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== install datasets (TruthfulQA substrate) ==="
pip install -q datasets || echo "DATASETS_INSTALL_FAILED"

echo "=== 2b BASE caving check (raw, no chat) ==="
python job_truthful_flip.py --truthfulqa --name google/gemma-2-2b --tag 2b_base > out/truthful_flip_2b_base.log 2>&1
echo "exit=$?"; echo "--- tail 2b_base ---"; tail -30 out/truthful_flip_2b_base.log

echo "=== 2b-IT caving check (chat template) ==="
python job_truthful_flip.py --truthfulqa --name google/gemma-2-2b-it --tag 2b_it --chat > out/truthful_flip_2b_it.log 2>&1
echo "exit=$?"; echo "--- tail 2b_it ---"; tail -30 out/truthful_flip_2b_it.log

echo "ALLDONE_2B_CAVECHECK"
