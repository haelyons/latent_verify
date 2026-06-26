#!/usr/bin/env bash
# On-box CAUSAL MEAN-ABLATION of the DLA-top late -it MLPs {23,27,29,30} (L28 dropped, tautological) ->
# realized free-gen cave-rate (self-judge primary; all gens saved -> H3) vs baseline + matched-random late-MLP
# floor (k=3). 9b base+it, TruthfulQA. Converts PART 11's correlational DLA into a causal claim.
# Controls land flat in ~/latent_verify (scp), so call the flat filename like run_causal_localize.sh.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python cave_ablate_late_mlp.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
echo "=== ensure datasets ==="
python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -3 || echo "DATASETS_PIP_FAILED"
echo "=== mean-ablation, TruthfulQA, 9b base vs 9b-it ==="
python cave_ablate_late_mlp.py --device cuda --truthfulqa --n 60 \
  --base google/gemma-2-9b --it google/gemma-2-9b-it \
  --layers 23,27,29,30 --k-rand 3 --mode mean \
  > out/cave_ablate_late_mlp_9b.log 2>&1; echo "exit=$?"
echo "--- tail ---"; tail -60 out/cave_ablate_late_mlp_9b.log
echo "ALLDONE_ABLATE_MLP"
