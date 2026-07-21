#!/usr/bin/env bash
# On-box Phase-B scale-out (>=80GB box): 27b-base ext2 ONLY. Measured pace on the first 27b box
# (128/164 items in a 12600s cap) puts one 27b cell at ~4.3h — each cell gets its own box; the -it
# cell runs from run_foldlisten_ext2_27bit.sh afterwards. Ported judge (commit_* + faithful_* labels,
# scorer_provenance). Files FLAT in ~/latent_verify.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

echo "=== selftests (model-free) ==="
python foldlisten_judge.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
python faithful_rescore.py --selftest || { echo "RESCORE_SELFTEST_FAIL"; exit 1; }

echo "=== 27b base (qa) ext2 ==="
python foldlisten_judge.py --family verifier_family_ext2.json --name google/gemma-2-27b --tag fl_27bbase_ext2 \
  --device cuda > out/foldlisten_27bbase_ext2.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_27bbase_ext2.log

echo "ALLDONE_FOLDLISTEN_EXT2_27BBASE"
