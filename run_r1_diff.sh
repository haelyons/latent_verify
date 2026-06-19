#!/usr/bin/env bash
# On-box: R1-DIFF -- the RLHF-installed-component hunt (gemma-2-9b base vs -it), A100.
# Substrate = the 16 I1 9b-it caving items (TruthfulQA misconceptions), embedded in rlhf_differential.py
# (no datasets dep). Instrument per the user's call: AtP a WIDE pre-filter, activation-patch the ARBITER.
# Faithfulness gate is built into the script: it reports `faithfulness_it_flips` -- must reproduce the
# I1 9b-it flip count (~14). If it does not, the regime/prompt drifted -> distrust the differential.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation headroom on the 40GB A100

echo "=== selftest (model-free) ==="
python rlhf_differential.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== R1-DIFF base<->it AtP differential + activation-patch confirm + Genadi band (9b) ==="
python rlhf_differential.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it > out/r1_diff.log 2>&1
echo "exit=$?"
echo "--- tail r1_diff.log ---"; tail -80 out/r1_diff.log
echo "ALLDONE_R1DIFF"
