#!/usr/bin/env bash
# On-box FOLD-vs-LISTEN (2b: cross-scale + cheap A10 hook-path shakedown). Is the span-ranked doubt circuit
# wrongness-SPECIFIC (a "fold" organ) or a GENERIC move-to-asserted mechanism (recruited equally by a correct
# "listen" push)? base-primary; -it self-bracketed (positive control + all-attention upper bound). Verdict:
# SC-SHARED / SC-DIRECTION / SC-DISTINCT / AXIS_WEAK / INSTRUMENT_DEAD / MOVE_UNMATCHED / INSUFFICIENT.
# Files land FLAT in ~/latent_verify (scp convention) -> call the bare module name, not controls/.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_fold_vs_listen.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
echo "=== ensure datasets ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"
echo "=== fold-vs-listen, 2b base vs 2b-it ==="
python cave_fold_vs_listen.py --device cuda --big-pool \
  --base google/gemma-2-2b --it google/gemma-2-2b-it \
  > out/cave_fold_vs_listen_2b.log 2>&1; echo "exit=$?"
echo "--- tail ---"; tail -60 out/cave_fold_vs_listen_2b.log
echo "ALLDONE_FOLD_VS_LISTEN_2B"
