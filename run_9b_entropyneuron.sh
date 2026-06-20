#!/usr/bin/env bash
# On-box: PART 6 -- entropy-neuron identification on gemma-2-9b (base + it). Weights-only null-space
# screen (bottom singular dirs of W_U) -> causal mean-ablation (dEntropy vs dLoss) on a fixed generic
# reference set, + matched-random control + base/-it differential. Porting Stolfo/Gurnee (NeurIPS 2024,
# 2406.16254) entropy neurons to Gemma-2 (the open gap). Forward-only -> fits a 40GB A100.
# entropy_neuron_gemma2.py scp'd flat by lambda_run.sh; self-contained (no repo-internal imports).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python entropy_neuron_gemma2.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b entropy-neuron identification + causal mean-ablation (base vs -it) ==="
python entropy_neuron_gemma2.py --device cuda \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/entropy_neuron_9b.log 2>&1
echo "exit=$?"
echo "--- tail entropy_neuron_9b.log ---"; tail -40 out/entropy_neuron_9b.log
echo "ALLDONE_ENTROPYNEURON"
