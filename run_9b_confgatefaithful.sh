#!/usr/bin/env bash
# On-box: STEP 1 (faithful re-foundation) -- does steering the causal entropy-CONFIDENCE direction UP change
# the model's REALIZED caved answer? On base Q/A argmax-W* caving items: GATES_REALIZED_CAVE iff the steered
# argmax returns to the model's unpushed (neutral) answer >= THR AND KL moves toward neutral AND the matched-
# random direction does not; else INDEPENDENT_REALIZED. Plus cos(u_cave, u_conf) + bootstrap CI on the
# hardened base carrier (AXIS_ALIGNED / AXIS_ORTHOGONAL). Re-tests the confidence<->caving relation on the
# FAITHFUL realized readout (the cave_suppress_vs_install criterion), NOT the logp-difference M the prior
# confidence_caving_gate used. Forward-only. Imports rlhf_differential/headset_direction/job_truthful_flip/etc.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python confidence_caving_gate_faithful.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b confidence_caving_gate_faithful (base vs -it) ==="
python confidence_caving_gate_faithful.py --device cuda \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b \
  > out/confidence_caving_gate_faithful_9b.log 2>&1
echo "exit=$?"
echo "--- tail confidence_caving_gate_faithful_9b.log ---"; tail -45 out/confidence_caving_gate_faithful_9b.log
echo "ALLDONE_CONFGATEFAITHFUL"
