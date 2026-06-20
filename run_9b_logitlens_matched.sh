#!/usr/bin/env bash
# On-box: reworked-E HARDENED -- matched-intersection logit-lens margin trajectory. Same per-layer
# logit-lens readout as run_9b_logitlens.sh, but keeps ONLY items both gemma-2-9b base AND -it know
# (neutral final-layer margin > 0 in both), aggregates over that shared set, and reports paired
# bootstrap CIs on the base-vs-it early/late/erosion differentials + an L0-excluded early metric.
# De-confounds the unmatched (44 vs 38) item sets + adds CIs + drops the L0 embedding-bias layer.
# logit_lens_margin_matched.py scp'd flat; defers repo imports (rlhf_differential/job_truthful_flip/
# misconception_pool, all scp'd) so they resolve on-box.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python logit_lens_margin_matched.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b matched logit-lens margin trajectory (both-models-know intersection, paired CIs) ==="
python logit_lens_margin_matched.py --device cuda \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/logit_lens_matched_9b.log 2>&1
echo "exit=$?"
echo "--- tail logit_lens_matched_9b.log ---"; tail -30 out/logit_lens_matched_9b.log
echo "ALLDONE_LOGITLENS_MATCHED"
