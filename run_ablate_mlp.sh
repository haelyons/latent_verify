#!/usr/bin/env bash
# On-box CAUSAL ablation of the DLA-top late -it MLPs {23,27,29,30} (L28 dropped, tautological) -> realized
# free-gen cave-rate (self-judge primary; all gens saved -> H3) vs baseline + matched-random late-MLP floor.
# Runs BOTH modes on one box: mean (global NEUTRAL-condition mean baseline) + resample (each item's own
# NEUTRAL mlp_out, the self-repair-cleaner arm). Outputs namespaced by mode. 9b base+it, TruthfulQA.
# Each leg writes a tiny *_summary.json (numbers+decision) that survives a flaky fetch. Controls land flat in
# ~/latent_verify (scp), so call the flat filename like run_causal_localize.sh.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_ablate_late_mlp.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
echo "=== ensure datasets ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"

for MODE in mean resample; do
  echo "=== ablation MODE=$MODE, TruthfulQA, 9b base vs 9b-it ==="
  python cave_ablate_late_mlp.py --device cuda --truthfulqa --n 60 \
    --base google/gemma-2-9b --it google/gemma-2-9b-it \
    --layers 23,27,29,30 --k-rand 3 --mode "$MODE" \
    > "out/cave_ablate_late_mlp_${MODE}_9b.log" 2>&1; echo "mode=$MODE exit=$?"
  # namespace outputs so the resample leg does not overwrite the mean leg
  for f in cave_ablate_late_mlp.json cave_ablate_late_mlp_summary.json; do
    [ -f "out/$f" ] && mv "out/$f" "out/${f%.json}_${MODE}.json"
  done
  echo "--- $MODE tail ---"; tail -25 "out/cave_ablate_late_mlp_${MODE}_9b.log"
done
echo "ALLDONE_ABLATE_MLP"
