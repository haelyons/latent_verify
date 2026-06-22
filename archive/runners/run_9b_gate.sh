#!/usr/bin/env bash
# On-box: PART 7 -- confidence_caving_gate. The cross-intervention prior controls never ran: does steering
# the (causal) entropy-confidence direction UP on caving items SUPPRESS the cave? gate_up >= GATE_THR AND
# random < BASE_FLOOR -> CONFIDENCE_GATES_CAVING; else NO_GATE (confidence does not control caving). The
# direct test of whether there is a confidence gate at all. Imports rlhf_differential/headset_direction/etc.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python confidence_caving_gate.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b confidence_caving_gate (base vs -it) ==="
python confidence_caving_gate.py --device cuda \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/confidence_caving_gate_9b.log 2>&1
echo "exit=$?"
echo "--- tail confidence_caving_gate_9b.log ---"; tail -45 out/confidence_caving_gate_9b.log
echo "ALLDONE_GATE"
