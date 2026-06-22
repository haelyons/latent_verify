#!/usr/bin/env bash
# On-box: QK weight-vs-realized control for L18.H5 at 2b. Settles whether post-training changes the
# head's OWN W_QK weight magnitude, or only the realized attention pattern (input-driven) -- the §8
# "deletes at the weights" scope question. Weights-only QK Frobenius + a small forward pass for the
# realized attention, gemma-2-2b base vs -it. 2b is tiny -> --device cpu (the forward is seconds and
# it dodges the A10 cu124 torch-cuda trap). qk_weight_2b_l18h5.py is scp'd flat by lambda_run.sh.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

echo "=== selftest ==="
python qk_weight_2b_l18h5.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 2b QK weight-vs-realized (base vs -it, L18.H5 + L18.H6 control) ==="
python qk_weight_2b_l18h5.py --device cpu \
  --name-base google/gemma-2-2b --name-it google/gemma-2-2b-it --tag 2b > out/qk_weight_2b.log 2>&1
echo "exit=$?"
echo "--- tail qk_weight_2b.log ---"; tail -30 out/qk_weight_2b.log
echo "ALLDONE_QKWEIGHT"
