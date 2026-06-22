#!/usr/bin/env bash
# On-box: PART 7 control B-Step-A -- cave_direction_sae_decomp. Decomposes the hardened cave-direction into
# GemmaScope SAE features (residual SAE, mid-layers L28/L32) -> is the cave-direction a small interpretable
# feature set (SPARSE) or distributed? First step of "direction -> circuit" (the attribution-graph route).
# Needs sae_lens (GemmaScope canonical residual SAEs). Imports rlhf_differential/headset_direction.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_direction_sae_decomp.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== install sae_lens (GemmaScope) ==="
pip install -q sae_lens || echo "SAELENS_INSTALL_FAILED (run will record sae_loaded=false)"

echo "=== 9b cave_direction_sae_decomp (base vs -it, L28/L32) ==="
python cave_direction_sae_decomp.py \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/cave_direction_sae_decomp_9b.log 2>&1
echo "exit=$?"
echo "--- tail cave_direction_sae_decomp_9b.log ---"; tail -45 out/cave_direction_sae_decomp_9b.log
echo "ALLDONE_SAEDECOMP"
