#!/usr/bin/env bash
# On-box: PART 7 control C -- confidence_direction_causal. Tries stronger confidence-direction fits
# (margin-quartile + entropy-quartile, layer sweep, held-out causal necessity/sufficiency + random control)
# and re-measures cos(u_cave, u_conf). Makes "cave perpendicular to confidence" rigorous (the C3 margin
# axis was non-causal). Imports rlhf_differential/misconception_pool/headset_direction/entropy_neuron_gemma2.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python confidence_direction_causal.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b confidence_direction_causal (base vs -it) ==="
python confidence_direction_causal.py --device cuda \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/confidence_direction_causal_9b.log 2>&1
echo "exit=$?"
echo "--- tail confidence_direction_causal_9b.log ---"; tail -45 out/confidence_direction_causal_9b.log
echo "ALLDONE_CONFDIR"
