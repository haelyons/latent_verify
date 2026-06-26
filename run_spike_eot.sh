#!/usr/bin/env bash
# On-box SPIKE: is there a readable cave-STATE at the gemma-2-it turn-boundary template token (M1)?
# base = positive control (caving committed/faithful). Forward + short generation + local Yes/No judge.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python spike_eot_cavestate.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
echo "=== ensure datasets ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"
echo "=== spike: eot cave-state, 9b base + 9b-it ==="
python spike_eot_cavestate.py --device cuda --big-pool \
  --base google/gemma-2-9b --it google/gemma-2-9b-it \
  > out/spike_eot_cavestate.log 2>&1; echo "exit=$?"
echo "--- tail ---"; tail -50 out/spike_eot_cavestate.log
echo "ALLDONE_SPIKE_EOT"
