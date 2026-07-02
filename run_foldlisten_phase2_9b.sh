#!/usr/bin/env bash
# On-box PHASE 2 at 9b-it on the frozen 74-item mechanism family: ALL-attention-to-challenge KO on the
# realized readout (5 arms incl. nomask baselines + masked-neutral drift floor) + DLA layer-overlap
# pre-check. The fold_nomask arm doubles as the harness/faithfulness check (family is fold-faithful by
# construction -> nomask fold rate ~1.0; INSUFFICIENT otherwise). Files FLAT in ~/latent_verify.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

echo "=== selftests (model-free) ==="
python foldlisten_judge.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
python foldlisten_phase2.py --selftest || { echo "P2_SELFTEST_FAIL"; exit 1; }

echo "=== phase 2: KO + DLA on mechanism_family_9bit.json (n=74) ==="
python foldlisten_phase2.py --run --family mechanism_family_9bit.json --name google/gemma-2-9b-it \
  --tag p2_9bit --device cuda --chat > out/foldlisten_phase2_9bit.log 2>&1; echo "exit=$?"
tail -12 out/foldlisten_phase2_9bit.log

echo "ALLDONE_FOLDLISTEN_PHASE2_9B"
