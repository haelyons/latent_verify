#!/usr/bin/env bash
# On-box: PART 7 -- confidence-mechanism + cave-direction hardening on gemma-2-9b (base + it).
# Runs the three pre-registered controls in PART-7 order on ONE box (forward-only, A100-40GB):
#   Control 2  cave_direction_heldout.py        -- held-out/LOO + cross-regime transfer + label-permuted
#                                                   null (SC-D / SC-D-XREG): is the cave-direction real
#                                                   out-of-sample and base-intrinsic, or in-sample/regime fit?
#   Control 3  confidence_vs_cave_direction.py  -- confidence/margin direction + cosine(cave,conf) +
#                                                   off-intersection necessity (SC-C1/C2/DISSOC): is the
#                                                   cave-direction deference or a confidence axis?
#   Control 1  entropy_distributed_presoftcap.py -- group-ablation ramp + pre/post-softcap entropy
#                                                   (SC-EN-D1/D2): does the entropy NULL survive the
#                                                   distributed grain and a pre-softcap readout?
# Files scp'd flat by lambda_run.sh; controls 2/3 import the repo modules (rlhf_differential, headset_direction,
# misconception_pool, job_truthful_flip), control 1 needs `datasets` for the long-context WikiText reference.
# Launch (heavier multi-load run -> raise the hard cap):
#   REMOTE_TIMEOUT=9000 bash lambda_run.sh gpu_1x_a100_sxm4 us-west-2 run_9b_confidence_arc.sh results_9b_confidence_arc
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftests (model-free; gate the run) ==="
python cave_direction_heldout.py --selftest        || { echo "SELFTEST_FAIL cave_direction_heldout"; exit 1; }
python confidence_vs_cave_direction.py --selftest  || { echo "SELFTEST_FAIL confidence_vs_cave_direction"; exit 1; }
python entropy_distributed_presoftcap.py --selftest || { echo "SELFTEST_FAIL entropy_distributed_presoftcap"; exit 1; }

echo "=== install datasets (for Control 1's long-context WikiText reference) ==="
pip install -q datasets || echo "DATASETS_INSTALL_FAILED (Control 1 will fall back to the short reference)"

echo "=== Control 2: cave_direction_heldout (held-out/LOO + cross-regime + label-permuted) ==="
python cave_direction_heldout.py \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/cave_direction_heldout_9b.log 2>&1
echo "exit=$?"; echo "--- tail cave_direction_heldout_9b.log ---"; tail -40 out/cave_direction_heldout_9b.log

echo "=== Control 3: confidence_vs_cave_direction (conf direction + cosine + off-intersection) ==="
python confidence_vs_cave_direction.py --device cuda \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/confidence_vs_cave_9b.log 2>&1
echo "exit=$?"; echo "--- tail confidence_vs_cave_9b.log ---"; tail -40 out/confidence_vs_cave_9b.log

echo "=== Control 1: entropy_distributed_presoftcap (group ramp + pre/post softcap) ==="
python entropy_distributed_presoftcap.py --device cuda --ref long --k 50 \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/entropy_distributed_9b.log 2>&1
echo "exit=$?"; echo "--- tail entropy_distributed_9b.log ---"; tail -40 out/entropy_distributed_9b.log

echo "ALLDONE_CONFIDENCE_ARC"
