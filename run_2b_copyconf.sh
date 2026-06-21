#!/usr/bin/env bash
# On-box: tests the HIERARCHY hypothesis -- is the 2b attention-COPY head's causal effect on the FAITHFUL cave
# CONDITIONAL on the model's confidence? Selects faithful caving items (realized shift toward W* under pushback),
# per item computes neutral-answer confidence + the copy-head (L18.H5) faithful restoration (zero its attention
# to the W*-span, renormalize; realized readout, NOT M), median-splits low/high confidence, reports the
# INTERACTION. CONDITIONAL_COPY (copy causal only when unconfident) / UNCONDITIONAL_COPY / NO_COPY_EFFECT /
# INSUFFICIENT / MIXED. transformer_lens only (no circuit-tracer); 2b fits a10/24GB. base primary + it.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest (model-free) ==="
python cave_copy_confidence_conditional.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 2b copy x confidence conditional (base qa + it chat) ==="
python cave_copy_confidence_conditional.py --device cuda \
  --name-base google/gemma-2-2b --name-it google/gemma-2-2b-it --tag 2b > out/cave_copy_confidence_conditional_2b.log 2>&1
echo "exit=$?"
echo "--- tail cave_copy_confidence_conditional_2b.log ---"; tail -55 out/cave_copy_confidence_conditional_2b.log
echo "ALLDONE_COPYCONF_2B"
