#!/usr/bin/env bash
# On-box: re-evaluate the 9b attention-COPY-of-W* effect on a FAITHFUL readout + MULTIPLE specificity controls
# (the 9b-it neutral-span control was dirty at 0.305). Knock out all-heads attention TO the W* span; measure
# realized P(W* first-token) drop + argmax-off + full-softmax target_frac (not just M=logp(C)-logp(W*)); compare
# vs neutral/random-content/question/C-answer span controls. FAITHFUL_COPY iff realized W*-drop fires AND beats
# every control span; else M_ONLY (overlay) / NON_SPECIFIC / ABSENT. base (qa) + it (chat).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest ==="
python faithful_copy_wstar.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b faithful_copy_wstar (base qa + it chat) ==="
python faithful_copy_wstar.py --device cuda --chat \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/faithful_copy_wstar_9b.log 2>&1
echo "exit=$?"
echo "--- tail faithful_copy_wstar_9b.log ---"; tail -50 out/faithful_copy_wstar_9b.log
echo "ALLDONE_FAITHCOPY"
