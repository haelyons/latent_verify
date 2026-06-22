#!/usr/bin/env bash
# On-box: the TWO-STAGE LINK. Do the DOUBT-attending head set WRITE the distributed cave-direction u_cave
# (feeding the downstream deference computation) and/or write the caved output (W*-C logit) directly? Per doubt
# head: head_out = z@W_O projected onto u_cave (fit diff-of-means counter-neutral) + DLA onto the W*-C logit;
# summed over the top-5 doubt set; matched-random-K floor; QK-target token. 9b base + it, big pool.
# FEEDS_CAVE_DIR (writes the rep, two-stage) / DIRECT_WRITER (writes output) / BOTH / NO_DIRECT_WRITE / INSUFFICIENT.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest (model-free) ==="
python cave_doubt_writes_cavedir.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== ensure datasets (--big-pool) ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"

echo "=== 9b doubt-writes-cavedir, big pool (base qa + it qa) ==="
python cave_doubt_writes_cavedir.py --device cuda --big-pool \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/cave_doubt_writes_cavedir_9b.log 2>&1
echo "exit=$?"
echo "--- tail ---"; tail -55 out/cave_doubt_writes_cavedir_9b.log
echo "ALLDONE_DOUBTDLA_9B"
