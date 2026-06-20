#!/usr/bin/env bash
# On-box: reworked-E -- per-layer logit-lens margin trajectory (correct vs misconception) on the 9b
# misconception substrate, gemma-2-9b base vs -it, neutral vs challenge turn, knowledge-gated. Tests
# WHERE the factual preference forms across layers and whether base/-it diverge early or late. Forward-only
# logit-lens (model's own ln_final + W_U + gemma softcap; no fitted direction). 9b fits a 40GB A100.
# logit_lens_margin_trajectory.py is scp'd flat by lambda_run.sh; it defers its repo-internal imports
# (rlhf_differential/job_truthful_flip/misconception_pool, all also scp'd) so they resolve on-box.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python logit_lens_margin_trajectory.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b logit-lens margin trajectory (base vs -it, neutral vs challenge, knowledge-gated) ==="
python logit_lens_margin_trajectory.py --device cuda \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/logit_lens_margin_9b.log 2>&1
echo "exit=$?"
echo "--- tail logit_lens_margin_9b.log ---"; tail -30 out/logit_lens_margin_9b.log
echo "ALLDONE_LOGITLENS"
