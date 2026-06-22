#!/usr/bin/env bash
# On-box batch: I2 (OV-vs-QK, 2b base+it) + I1 2b cross-scale arm. Invoked via remote_run.sh
# (which builds the venv + cu124 torch and execs this). HF_TOKEN comes from the env remote_run sets.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
pip install -q datasets 2>/dev/null   # truthful_flip --truthfulqa needs it (I2 does not)
echo "=== I2 OV-vs-QK decomposition (gemma-2-2b base & it) ==="
python job_rlhf_ovqk.py > out/i2_ovqk.log 2>&1; echo "i2 exit=$?"
echo "=== I1 truthful_flip 2b (SC-D cross-scale arm) ==="
python job_truthful_flip.py --truthfulqa --name google/gemma-2-2b --tag 2b > out/i1_2b.log 2>&1; echo "i1_2b exit=$?"
echo "ALLDONE_2B"
