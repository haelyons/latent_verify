#!/usr/bin/env bash
# On-box: GRAPH prerequisite -- does 2b BASE have a faithful caving target (argmax-W* items: the model would
# actually emit W*)? cave_suppress_vs_install at 2b reports n_argmaxW_cave + whether ablating u_cave restores
# the unpushed answer at 2b (RESTORES_NEUTRAL / etc). If n_argmaxW_cave >= a few -> 2b base is a valid,
# fully-tooled substrate for the attribution graph (circuit-tracer + GemmaScope-2b transcoders).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_suppress_vs_install.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 2b cave_suppress_vs_install (base qa + it chat) ==="
python cave_suppress_vs_install.py --device cuda --chat \
  --name-base google/gemma-2-2b --name-it google/gemma-2-2b-it --tag 2b > out/cave_suppress_vs_install_2b.log 2>&1
echo "exit=$?"
echo "--- tail cave_suppress_vs_install_2b.log ---"; tail -40 out/cave_suppress_vs_install_2b.log
echo "ALLDONE_SUPPINSTALL_2B"
