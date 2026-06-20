#!/usr/bin/env bash
# On-box: PART 7 control A (skeptic-required) -- substrate_margin_grid. Separates "confidence/margin level"
# from "substrate kind" and template, the confound the latent_skeptic flagged in the 2b-caving causal claim.
# 4 cells: {capitals, misconceptions} x {high-margin, low-margin}, FIXED chat template, capitulation vs
# pre-margin. MARGIN_GATED within a substrate iff low-margin caves more than high-margin by >= GATE_DELTA;
# the substrate-controlled read is BOTH_SUBSTRATES_MARGIN_GATED. Needs datasets (misconception substrate) +
# sycophancy_items.json (capitals).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python substrate_margin_grid.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== install datasets ==="
pip install -q datasets || echo "DATASETS_INSTALL_FAILED"

echo "=== 2b substrate x margin grid (it chat + base Q/A, fixed template per regime) ==="
python substrate_margin_grid.py --name google/gemma-2-2b-it --chat --name-base google/gemma-2-2b --tag 2b \
  > out/substrate_margin_grid_2b.log 2>&1
echo "exit=$?"
echo "--- tail substrate_margin_grid_2b.log ---"; tail -50 out/substrate_margin_grid_2b.log
echo "ALLDONE_MARGINSWEEP"
