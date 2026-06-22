#!/usr/bin/env bash
# On-box: C1 OV-norm probe (gemma-2-2b base + -it). Weight-only; A10 is enough.
# Faithfulness gate is built in: base ov_pref/rank must reproduce I2 (0.9997 / 0).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
echo "=== C1 ov_norm_probe 2b base+it ==="
python ov_norm_probe.py > out/c1_ovnorm.log 2>&1; echo "ovnorm exit=$?"
echo "--- c1_ovnorm.log ---"; cat out/c1_ovnorm.log
echo "ALLDONE_C1"
