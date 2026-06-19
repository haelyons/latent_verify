#!/usr/bin/env bash
# On-box: the open control -- activation-patch (arbiter) sweep over the AtP-low 9b heads, to harden or
# overturn the R1-DIFF NULL. Forward-only (no AtP backward), so it fits the 40GB A100. Imports the
# verified rlhf_differential._confirm + ITEMS (both scp'd flat alongside job_truthful_flip).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python atp_low_confirm.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b AtP-low activation-patch sweep (base + it) ==="
python atp_low_confirm.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it > out/atp_low_9b.log 2>&1
echo "exit=$?"
echo "--- tail atp_low_9b.log ---"; tail -45 out/atp_low_9b.log
echo "ALLDONE_ATPLOW"
