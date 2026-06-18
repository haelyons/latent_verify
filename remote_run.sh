#!/usr/bin/env bash
# Remote bootstrap for the Lambda A10 session (SEQUENCE_170626). Idempotent: sets up
# the venv on first call, then execs whatever job command is passed. Encodes the two
# FRAMING_NOTES sec-11 traps: plain venv (NOT --system-site-packages) and torch cu124
# (the default PyPI wheel is too new for the A10 driver -> silent CPU fallback).
#   usage:  HF_TOKEN=hf_... bash remote_run.sh python job_sycophancy.py --model base
set -euo pipefail
cd ~/latent_verify
TORCH_IDX=https://download.pytorch.org/whl/cu124
if [ ! -f .venv/.ready ]; then
  echo "[setup] venv + stack"
  python3 -m venv .venv
  . .venv/bin/activate
  pip install -q --upgrade pip
  pip install -q torch --index-url "$TORCH_IDX"
  pip install -q transformer_lens transformers
  pip install -q torch --index-url "$TORCH_IDX"   # re-pin: TL may bump torch to a non-cuda wheel
  python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA NOT available -> silent CPU fallback (FRAMING sec 11)"
print("[setup] torch", torch.__version__, "cuda", torch.cuda.is_available())
import transformer_lens, transformers  # import-ability check (fatal if broken)
try:
    from importlib.metadata import version
    print("[setup] transformer_lens", version("transformer_lens"), "transformers", version("transformers"))
except Exception as e:
    print("[setup] version probe skipped:", e)
PY
  touch .venv/.ready
else
  . .venv/bin/activate
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export HF_TOKEN="${HF_TOKEN:-}"
mkdir -p out
echo "[run] $*"
exec "$@"
