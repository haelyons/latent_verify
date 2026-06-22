#!/usr/bin/env bash
# On-box: PART 7 mechanism test B -- mlp_stream_caving_patch. DLA showed MLPs WRITE the cave-direction change;
# does the MLP stream CAUSALLY DRIVE caving? On -it caving items, project u_cave OUT of every MLP output
# (in-distribution, no cross-model substitution); MLP_STREAM_DRIVES_CAVING iff recovery >=0.20 AND random
# <0.05 AND mlp>attn (the attn-stream + random are the controls). base run = RLHF differential.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python mlp_stream_caving_patch.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b mlp_stream_caving_patch (base vs -it) ==="
python mlp_stream_caving_patch.py \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/mlp_stream_caving_patch_9b.log 2>&1
echo "exit=$?"
echo "--- tail mlp_stream_caving_patch_9b.log ---"; tail -45 out/mlp_stream_caving_patch_9b.log
echo "ALLDONE_MLPPATCH"
