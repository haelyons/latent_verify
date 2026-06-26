#!/usr/bin/env bash
# On-box: cave_social_source v2 -- adds per-item READ/WRITE arrays (for offline paired-bootstrap CI on the
# SC-S1 authority gradient) + an authority MINIMAL-PAIR cue (mp_friend "my friend" vs authority "my professor",
# same "my X says" frame -> isolates source authority from other lexical properties). Faithfulness anchor: the
# SELF cue must still reproduce 0.589/0.440/0.019 (9b). Forward-only, base qa. v2 tag preserves the triaged v1.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest (model-free) ==="
python cave_social_source.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== ensure datasets (--big-pool) ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"

echo "=== social-source v2, 9b base (qa) ==="
python cave_social_source.py --device cuda --big-pool --name google/gemma-2-9b --tag 9b_base_v2 \
  > out/cave_social_source_9b_base_v2.log 2>&1; echo "exit=$?"
echo "--- tail ---"; tail -34 out/cave_social_source_9b_base_v2.log

echo "=== social-source v2, 2b base (qa, cross-scale) ==="
python cave_social_source.py --device cuda --big-pool --name google/gemma-2-2b --tag 2b_base_v2 \
  > out/cave_social_source_2b_base_v2.log 2>&1; echo "exit=$?"
echo "--- tail ---"; tail -34 out/cave_social_source_2b_base_v2.log

echo "ALLDONE_SOCIAL_V2"
