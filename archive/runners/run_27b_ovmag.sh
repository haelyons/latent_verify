#!/usr/bin/env bash
# On-box: NEXT-3 -- characterize the 27b RLHF OV-magnitude increase. Weights-only (W_OV scalar-fit, write
# direction cosine, copy-pref, top-vocab overlap) base vs -it, on the copy basket where qk_collapse found
# W_OV_fro CHANGED. Forward-FREE, so CPU on the 200GB-RAM box fits 27b (54GB bf16) with no 80GB GPU needed.
# ov_magnitude_characterize.py scp'd flat.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python ov_magnitude_characterize.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 27b OV-magnitude characterization (base vs -it, weights-only, CPU) ==="
python ov_magnitude_characterize.py --device cpu \
  --name-base google/gemma-2-27b --name-it google/gemma-2-27b-it --tag 27b > out/ov_magnitude_27b.log 2>&1
echo "exit=$?"
echo "--- tail ov_magnitude_27b.log ---"; tail -40 out/ov_magnitude_27b.log
echo "ALLDONE_OVMAG"
