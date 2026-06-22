#!/usr/bin/env bash
# On-box: TOWARD A CIRCUIT (A) -- reader-side path-patch. When ablating u_cave at L restores the base unpushed
# answer, WHICH downstream components carry that restoration to the logits? DIRECT-path (delta->logits, downstream
# frozen) vs per-downstream-component path-patch effect(R). DIRECT_WRITE / LOCALIZED_READERS (top-5 carry >=50% +
# jointly reconstruct = circuit edge) / DISTRIBUTED_READERS (no clean reader). The decisive circuit-or-distributed
# test on the faithful base readout. base + it (it expected INSUFFICIENT).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_reader_pathpatch.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b cave_reader_pathpatch (base qa + it chat) ==="
python cave_reader_pathpatch.py --device cuda --chat \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/cave_reader_pathpatch_9b.log 2>&1
echo "exit=$?"
echo "--- tail cave_reader_pathpatch_9b.log ---"; tail -55 out/cave_reader_pathpatch_9b.log
echo "ALLDONE_READERPP"
