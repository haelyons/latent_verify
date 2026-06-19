#!/usr/bin/env bash
# On-box: the 27b QK-collapse de-confound (latent_skeptic author_queue, claim 2). The induction probe
# was uninformative for low-induction OV-copy heads; this reads the WEIGHT-ONLY W_QK Frobenius (and OV
# magnitudes) base->it for the 27b copy basket -- does post-training change the QK bilinear magnitude?
# Weights-only, forward-free -> fast once the 27b weights load. qk_collapse_metric.py is scp'd flat.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python qk_collapse_metric.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 27b QK-collapse on the copy basket (base vs -it) ==="
python qk_collapse_metric.py --heads "11,2;11,4;11,7;11,21;16,3;17,4;19,2;19,5;19,7;23,24" \
  --name-base google/gemma-2-27b --name-it google/gemma-2-27b-it --tag 27b > out/qk_collapse_27b.log 2>&1
echo "exit=$?"
echo "--- tail qk_collapse_27b.log ---"; tail -45 out/qk_collapse_27b.log
echo "ALLDONE_27B_QK"
