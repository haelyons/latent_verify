#!/usr/bin/env bash
# On-box: STEP 0+2 -- faithful caving metric (F1) + does the cave-direction control the REALIZED answer (F2).
# F1: per misconception item under counter, count M_FLIP (logp(C)-logp(W*) crosses, the OLD metric) vs
# REALIZED_FLIP (argmax/realized-P actually shifts C->W*); report overlap + realized P(W*) tail-fraction.
# F2: on M_FLIP (and REALIZED_FLIP) items, ablate u_cave; does it move M only, or the REALIZED argmax/P?
# base (qa, faithful substrate -- W* carries mass) + it (chat).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python faithful_caving.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b faithful_caving (base qa + it chat) ==="
python faithful_caving.py --device cuda --chat \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/faithful_caving_9b.log 2>&1
echo "exit=$?"
echo "--- tail faithful_caving_9b.log ---"; tail -55 out/faithful_caving_9b.log
echo "ALLDONE_FAITHCAVING"
