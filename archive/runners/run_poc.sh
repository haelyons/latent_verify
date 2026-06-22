#!/usr/bin/env bash
# Environment setup + execution harness for the attribution-graph PoC.
# Runs on the Lambda GPU instance (HTTPS-only return path: results are printed
# to stdout AND, if the clone is authenticated, committed back to the branch).
#
# Pinned circuit-tracer commit (API verified against this exact tree):
CT_COMMIT="041a9b2cbd7f3fe7e0a625a6794e66fc4aa5f883"
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p out
LOG="out/run.log"
exec > >(tee "$LOG") 2>&1

echo "===== [setup] $(date -u) ====="
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python3 --version

# --- venv (idempotent) -----------------------------------------------------
if [ ! -d .venv ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip

# circuit-tracer pinned at the verified commit. torch is expected to be
# preinstalled on the Lambda image; install it only if missing.
python -c "import torch" 2>/dev/null || pip install -q torch
if ! python -c "import circuit_tracer" 2>/dev/null; then
  echo "[setup] installing circuit-tracer@${CT_COMMIT}"
  pip install -q "git+https://github.com/safety-research/circuit-tracer@${CT_COMMIT}"
fi
echo "[setup] circuit-tracer commit: $(python -c 'import circuit_tracer,os,subprocess;import importlib.util as u;print(os.path.dirname(u.find_spec("circuit_tracer").origin))')"
pip show circuit-tracer 2>/dev/null | grep -E "Version|Location" || true

# --- HF / Gemma access pre-check ------------------------------------------
python - <<'PY'
import sys
try:
    from huggingface_hub import whoami
    who = whoami()
    print(f"[setup] HF authenticated as: {who.get('name','?')}")
except Exception as e:
    print(f"[setup][WARN] HF token not detected ({e}).")
    print("[setup][WARN] If model/transcoder download fails, run in a cell:")
    print("[setup][WARN]   from huggingface_hub import login; login('hf_...')")
PY

# --- one-token target sanity check (cheap, before the heavy run) -----------
python - <<'PY'
from transformers import AutoTokenizer
try:
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    ids = tok.encode(" Austin", add_special_tokens=False)
    print(f"[check] ' Austin' -> token ids {ids} (n={len(ids)}) "
          f"{'OK single-token' if len(ids)==1 else 'FAIL: NOT single token'}")
except Exception as e:
    print(f"[check][WARN] tokenizer check skipped ({e}); poc asserts this at runtime.")
PY

echo "===== [t0] $(date -u) ====="
python poc_minimal.py --stage t0 || { echo "[t0] FAILED"; }

echo "===== [t1] $(date -u) ====="
# transport multiplier is overridable via env TRANSPORT_M (default -2.0, the
# Anthropic-style inhibition value; recalibrate from the t0 response curve).
TM="${TRANSPORT_M:--2.0}"
python poc_minimal.py --stage t1 --transport-multiplier "$TM" || { echo "[t1] FAILED"; }

# --- return channel 1: verbatim stdout dump -------------------------------
for f in out/t0.json out/t1.json; do
  echo "===== BEGIN ${f} ====="
  cat "$f" 2>/dev/null || echo "(missing: $f)"
  echo "===== END ${f} ====="
done

# --- return channel 2: commit results back if the clone is authenticated ---
if git rev-parse --git-dir >/dev/null 2>&1; then
  git add -f out/t0.json out/t1.json out/run.log 2>/dev/null
  git -c user.email="poc@instance" -c user.name="lambda-runner" \
      commit -q -m "results: t0/t1 JSON + run log from GPU run" 2>/dev/null \
    && (git push origin HEAD 2>&1 | tail -3 && echo "[return] pushed results to branch") \
    || echo "[return] no push (no auth or nothing to commit) -- copy the JSON above"
fi
echo "===== [done] $(date -u) ====="
