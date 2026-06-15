#!/usr/bin/env bash
# CPU-session execution harness for the attribution-graph PoC.
#
# Pivot context: no GPU session available. The PoC needs only short forward
# passes through gemma-2-2b (~60 total), which fit a 4-vCPU / 15 GB container
# in bf16. Differences from run_poc.sh (the Lambda GPU path):
#   - base weights come from the ungated unsloth/gemma-2-2b mirror (the
#     google/ repo is gated and no HF token is available here); config and
#     tokenizer verified identical to the official architecture, and the
#     loaded stack is validated against Neuronpedia activations before any
#     experiment stage runs (see smoke_test.py).
#   - torch is the CPU wheel; everything runs in bf16.
#
# Pinned circuit-tracer commit (same as run_poc.sh):
CT_COMMIT="041a9b2cbd7f3fe7e0a625a6794e66fc4aa5f883"
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p out
LOG="out/run.log"
exec > >(tee "$LOG") 2>&1

export GEMMA_LOCAL_DIR="${GEMMA_LOCAL_DIR:-hf_local/gemma-2-2b}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "===== [setup] $(date -u) ====="
echo "host: $(hostname) (CPU-only, $(nproc) vCPU)"
python3 --version

if [ ! -d .venv ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -c "import torch" 2>/dev/null || \
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu
if ! python -c "import circuit_tracer" 2>/dev/null; then
  echo "[setup] installing circuit-tracer@${CT_COMMIT}"
  pip install -q "git+https://github.com/safety-research/circuit-tracer@${CT_COMMIT}"
fi

# Local mirror weights (gated google/ repo unreachable without a token).
if [ ! -s "$GEMMA_LOCAL_DIR/model.safetensors" ]; then
  echo "[setup] downloading gemma-2-2b weights from ungated mirror"
  mkdir -p "$GEMMA_LOCAL_DIR"
  for f in config.json generation_config.json special_tokens_map.json \
           tokenizer.json tokenizer.model tokenizer_config.json model.safetensors; do
    curl -sL -o "$GEMMA_LOCAL_DIR/$f" \
      "https://huggingface.co/unsloth/gemma-2-2b/resolve/main/$f"
  done
fi

echo "===== [smoke: validate stack vs Neuronpedia] $(date -u) ====="
python smoke_test.py || { echo "[smoke] FAILED -- aborting before t0/t1"; exit 1; }
grep -q "\[smoke\] PASS" "$LOG" || { echo "[smoke] no PASS marker -- aborting"; exit 1; }

echo "===== [t0] $(date -u) ====="
python poc_minimal.py --stage t0 || { echo "[t0] FAILED"; }

echo "===== [t1] $(date -u) ====="
TM="${TRANSPORT_M:--2.0}"
python poc_minimal.py --stage t1 --transport-multiplier "$TM" || { echo "[t1] FAILED"; }

for f in out/t0.json out/t1.json; do
  echo "===== BEGIN ${f} ====="
  cat "$f" 2>/dev/null || echo "(missing: $f)"
  echo "===== END ${f} ====="
done
echo "===== [done] $(date -u) ====="
