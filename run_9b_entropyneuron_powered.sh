#!/usr/bin/env bash
# On-box: PART 6 POWERED -- entropy-neuron identification on gemma-2-9b addressing the latent_skeptic
# cruxes: (1) BOTH mean- AND zero-ablation per neuron (zero removes the mean null-space write that mean-
# ablation preserves -- the top crux); (2) LONG-CONTEXT reference (WikiText-2 256-token windows, the
# entropy-neuron literature's regime) via --ref long; (3) larger null basis --k 50. base + it.
# entropy_neuron_gemma2.py scp'd flat by lambda_run.sh; needs `datasets` for the long reference.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python entropy_neuron_gemma2.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== install datasets (for the long-context WikiText reference) ==="
pip install -q datasets || echo "DATASETS_INSTALL_FAILED (control will fall back to short reference)"

echo "=== 9b entropy-neuron POWERED (mean+zero ablation, --ref long, --k 50, base vs -it) ==="
python entropy_neuron_gemma2.py --device cuda --ref long --k 50 \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b_powered > out/entropy_neuron_9b_powered.log 2>&1
echo "exit=$?"
echo "--- tail entropy_neuron_9b_powered.log ---"; tail -45 out/entropy_neuron_9b_powered.log
echo "ALLDONE_ENTROPYNEURON_POWERED"
