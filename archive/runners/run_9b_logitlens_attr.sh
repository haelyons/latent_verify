#!/usr/bin/env bash
# On-box: reworked-E de-confound -- attribute the matched it-minus-base early logit-lens margin gap to
# WEIGHTS vs (A) prompt FORMAT (run each model under native + cross format), (B) lens CALIBRATION
# (re-project each model's native resid through BOTH models' unembeds), plus (C) FAITHFULNESS (per-layer
# argmax==final-greedy) and (D) FRACTIONAL erosion. gemma-2-9b base vs -it, matched both-know set, paired
# CIs. Resolves the latent_skeptic cruxes (format/lens NEEDS_RUN; precomputation EXPLAINS) by running.
# logit_lens_attribution.py scp'd flat; defers repo imports (rlhf_differential/job_truthful_flip/
# misconception_pool, all scp'd) so they resolve on-box. Holds both 9b unembeds on CPU (~3.6GB host RAM),
# one model live on GPU at a time -> fits a 40GB A100.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python logit_lens_attribution.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b logit-lens attribution (format crossover + cross-lens + faithfulness + frac-erosion) ==="
python logit_lens_attribution.py --device cuda \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/logit_lens_attribution_9b.log 2>&1
echo "exit=$?"
echo "--- tail logit_lens_attribution_9b.log ---"; tail -40 out/logit_lens_attribution_9b.log
echo "ALLDONE_LOGITLENS_ATTR"
