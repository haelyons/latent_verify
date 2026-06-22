#!/usr/bin/env bash
# On-box (d) -it INSTALLATION: the doubt write-vs-read resolution at 9b-IT (QA template -- the chat template
# gives a P(W*) ghost / 0 faithful; QA on -it tests whether the RLHF model has the doubt circuit under base-like
# framing). Same control, --name gemma-2-9b-it. big pool. (chat-template + flip-rate readout deferred.)
set -uo pipefail
cd ~/latent_verify; . .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== selftest ==="; python cave_doubt_write_vs_read.py --selftest || { echo SELFTEST_FAIL; exit 1; }
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -2 || echo DATASETS_PIP_FAILED
echo "=== 9b-it doubt write-vs-read, big pool (QA template) ==="
python cave_doubt_write_vs_read.py --device cuda --big-pool --name google/gemma-2-9b-it --tag 9b_it > out/cave_doubt_write_vs_read_9b_it.log 2>&1
echo "exit=$?"; echo "--- tail ---"; tail -40 out/cave_doubt_write_vs_read_9b_it.log; echo ALLDONE_DOUBTWVR_9BIT
