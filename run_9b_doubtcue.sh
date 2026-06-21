#!/usr/bin/env bash
# On-box: the upstream "decision/input-cue" candidate. Is FAITHFUL caving driven by heads attending the user's
# DOUBT/CHALLENGE span ("Actually, I think... Are you sure?", EXCLUDING the W* answer token, so dissociated from
# the already-refuted attention-COPY-of-W*)? And is that attention RLHF-installed (-it >> base)? gemma-2-9b
# base + it. Per-head attention-to-doubt base vs it (Genadi L10-15 band reported), then knock out attention to
# the doubt span (top heads + band) and read the FAITHFUL restoration toward neutral. DOUBT_DRIVEN /
# DOUBT_PRESENT_NOT_CAUSAL / NOT_DOUBT_DRIVEN / INSUFFICIENT. transformer_lens only (no circuit-tracer).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== selftest (model-free) ==="
python cave_doubt_cue_attention.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }

echo "=== 9b doubt-cue attention (base qa + it chat) ==="
python cave_doubt_cue_attention.py --device cuda --chat \
  --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b > out/cave_doubt_cue_attention_9b.log 2>&1
echo "exit=$?"
echo "--- tail cave_doubt_cue_attention_9b.log ---"; tail -60 out/cave_doubt_cue_attention_9b.log
echo "ALLDONE_DOUBTCUE_9B"
