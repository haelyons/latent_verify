#!/usr/bin/env bash
# On-box: STEP 3(a) -- when ablating u_cave suppresses realized W*, where does the answer go? On base Q/A
# caving items (counter argmax = W*), ablate u_cave and classify the new realized argmax: == C (install-C),
# == the NEUTRAL/unpushed argmax (restore-to-neutral), or a third token (suppress-only). + KL(P3||neutral) vs
# KL(counter||neutral) and a matched-random floor. Defines what the cave-direction IS mechanistically. base+it.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_suppress_vs_install.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b cave_suppress_vs_install (base qa + it chat) ==="
python cave_suppress_vs_install.py --device cuda --chat \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/cave_suppress_vs_install_9b.log 2>&1
echo "exit=$?"
echo "--- tail cave_suppress_vs_install_9b.log ---"; tail -50 out/cave_suppress_vs_install_9b.log
echo "ALLDONE_SUPPINSTALL"
