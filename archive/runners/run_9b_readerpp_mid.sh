#!/usr/bin/env bash
# On-box: (A) de-confound -- reader path-patch at MID layers L28 + L24 (the layer-proximity crux). At L36
# (6 from the readout) DIRECT_WRITE was trivial; L28/L24 have many downstream layers, so if u_cave has READERS
# there is room for them. DIRECT_WRITE at a mid layer too -> no reader circuit (write-direction, robust);
# LOCALIZED_READERS -> a real reader-circuit stage. base only (it has 0 argmax-W*).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_reader_pathpatch.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b reader path-patch @ L28 ==="
python cave_reader_pathpatch.py --device cuda --chat --layer 28 \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b_L28 > out/cave_reader_pathpatch_9b_L28.log 2>&1
echo "exit=$?"; echo "--- tail L28 ---"; tail -20 out/cave_reader_pathpatch_9b_L28.log

echo "=== 9b reader path-patch @ L24 ==="
python cave_reader_pathpatch.py --device cuda --chat --layer 24 \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b_L24 > out/cave_reader_pathpatch_9b_L24.log 2>&1
echo "exit=$?"; echo "--- tail L24 ---"; tail -20 out/cave_reader_pathpatch_9b_L24.log
echo "ALLDONE_READERPP_MID"
