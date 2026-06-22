#!/usr/bin/env bash
# On-box: DECISIVE resolution -- on the SAME span-attention-ranked top-5 DOUBT heads + SAME fixed faithful items
# (9b base, DOUBT framing), compare ATTENTION_KO (zero attn to doubt span = remove the READ) vs OUTPUT_PATCH
# (replace counter output with neutral = remove the WRITE) vs RANDOM_OUTPUT (head-specificity floor). Settles
# whether the doubt heads WRITE the cave (WRITE_CIRCUIT) or only GATE the read (READ_GATE_ONLY). big pool for n.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest (model-free) ==="
python cave_doubt_write_vs_read.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== ensure datasets (--big-pool) ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"

echo "=== 9b doubt write-vs-read, big pool (base qa) ==="
python cave_doubt_write_vs_read.py --device cuda --big-pool \
  --name google/gemma-2-9b --tag 9b_base > out/cave_doubt_write_vs_read_9b_base.log 2>&1
echo "exit=$?"
echo "--- tail ---"; tail -45 out/cave_doubt_write_vs_read_9b_base.log
echo "ALLDONE_DOUBTWVR_9B"
