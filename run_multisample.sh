#!/usr/bin/env bash
# On-box INTERVENTION (1): judge-FREE multi-sample cave-rate. Audits the self-judge readout behind the -it
# cave-direction (AUROC 0.92 was fit on self-judge labels scored on UNSAVED generations). This run SAVES every
# generation (H3-auditable), labels caving by deterministic answer-string match (no judge), re-fits + AUROCs the
# cave-direction on the judge-free label, and records the old self-judge per sample for an agreement comparison.
# Files land FLAT in ~/latent_verify (scp convention).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_multisample_caverate.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
echo "=== ensure datasets ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"
echo "=== multisample cave-rate, 9b base vs 9b-it ==="
python cave_multisample_caverate.py --device cuda --big-pool \
  --base google/gemma-2-9b --it google/gemma-2-9b-it \
  > out/cave_multisample_caverate_9b.log 2>&1; echo "exit=$?"
echo "--- tail ---"; tail -60 out/cave_multisample_caverate_9b.log
echo "ALLDONE_MULTISAMPLE"
