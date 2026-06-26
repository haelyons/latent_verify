#!/usr/bin/env bash
# On-box: Unit S (social-source + cue factorization) + Unit C (confidence-recruitment) on the doubt circuit.
# FAITHFULNESS ANCHOR: cave_social_source's SELF cue must reproduce cave_doubt_write_vs_read's
# doubt_ko ~0.589 / output_patch ~0.440 / random ~0.019 (9b base). If `self` drifts, the social rows are
# not trustworthy. Forward-only; base qa (the clean DOUBT site). selftest-gated, datasets for --big-pool.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftests (model-free) ==="
python cave_social_source.py --selftest          || { echo "SELFTEST_FAIL_SOCIAL"; exit 1; }
python cave_confidence_recruitment.py --selftest  || { echo "SELFTEST_FAIL_CONF"; exit 1; }

echo "=== ensure datasets (--big-pool) ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"

echo "=== Unit S: social-source sweep, 9b base (qa) -- lead + faithfulness anchor ==="
python cave_social_source.py --device cuda --big-pool --name google/gemma-2-9b --tag 9b_base \
  > out/cave_social_source_9b_base.log 2>&1; echo "exit=$?"
echo "--- tail ---"; tail -32 out/cave_social_source_9b_base.log

echo "=== Unit S: social-source sweep, 2b base (qa, cross-scale) ==="
python cave_social_source.py --device cuda --big-pool --name google/gemma-2-2b --tag 2b_base \
  > out/cave_social_source_2b_base.log 2>&1; echo "exit=$?"
echo "--- tail ---"; tail -32 out/cave_social_source_2b_base.log

echo "=== Unit C: confidence-recruitment, 9b base (qa) ==="
python cave_confidence_recruitment.py --device cuda --big-pool --name google/gemma-2-9b --tag 9b_base \
  > out/cave_confidence_recruitment_9b_base.log 2>&1; echo "exit=$?"
echo "--- tail ---"; tail -32 out/cave_confidence_recruitment_9b_base.log

echo "ALLDONE_SOCIAL_UNITC"
