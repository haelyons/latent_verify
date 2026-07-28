#!/usr/bin/env bash
# ============================================================================================
# R6 + R12 + part of R14 — the 2b and 9b run-only holes. Ledger GAPS_RECONCILED.md 4.3 rows R6
# (2 claims), R12 (1 claim) and R14 (0 claims, flag riders that cost nothing once a box is loaded).
#
# NO NEW REGISTRATION. Every instrument carries its own frozen metric / thresholds / decision_rule;
# this runner is thin. These are cells that were never run, not measurements that were never designed
# — which is exactly why they need no registration and the K1/K2/K5/K7 rows do.
#
# 2b (~5 GB) and 9b (~18 GB) both fit a 40 GB A100 with room, so both scales share one cheap box.
#
# FAIL-SOFT, AND FOR A SPECIFIC REASON. These eight instruments have not been exercised in this
# lineage recently and their argparse contracts were read statically, not run. A wrong flag must cost
# one cell, not the box. So: every cell logs `--help` FIRST (so a next session gets the exact CLI even
# from a failed cell), then its selftest if it has one, then the cell itself; nothing aborts the run.
# This is deliberate — the alternative is discovering a flag error after a model load, which is how
# gen_outputs_table's missing `tag` field would have died on a GPU (docs/drafts/CODEBLOCKS_verified.md K3).
#
# Files land FLAT in ~/latent_verify.
# ============================================================================================
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
mkdir -p out

python - <<'PY' > out/provenance_r6r12.json 2>/dev/null || echo '{"provenance":"FAILED"}' > out/provenance_r6r12.json
import json, os, subprocess, sys, datetime
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
def ver(m):
    try:
        from importlib.metadata import version; return version(m)
    except Exception: return None
p = {"run": "r6r12_small_riders",
     "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
     "python": sys.version.split()[0],
     "lambda_instance_id": os.environ.get("LAMBDA_INSTANCE_ID"),
     "git_commit": os.environ.get("GIT_COMMIT"),
     "gpu_name": sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
     "gpu_count": sh("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"),
     "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
     "dtype": "bfloat16", "torch": ver("torch"), "transformers": ver("transformers"),
     "transformer_lens": ver("transformer_lens"), "datasets": ver("datasets")}
try: import torch; p["cuda_runtime"] = torch.version.cuda
except Exception: p["cuda_runtime"] = None
print(json.dumps(p, indent=2))
PY
cat out/provenance_r6r12.json

python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -2 || echo "DATASETS_PIP_FAILED"

cell () {  # $1 = label, $2 = module for --help/--selftest, rest = command
  local label="$1" mod="$2"; shift 2
  echo "=== CELL $label ($mod) ==="
  python "$mod" --help > "out/r6r12_help_${label}.log" 2>&1 || echo "  (--help failed)"
  head -25 "out/r6r12_help_${label}.log"
  python "$mod" --selftest > "out/r6r12_selftest_${label}.log" 2>&1 \
    && echo "  selftest OK" || echo "  selftest FAIL or absent (continuing)"
  "$@" > "out/r6r12_${label}.log" 2>&1
  echo "exit=$?  ($label)"
  tail -12 "out/r6r12_${label}.log"
  echo "--- end $label ---"
}

# ---- R6: 2b riders -------------------------------------------------------------------------
cell copyscore_sweep_2b job_copyscore.py \
  python job_copyscore.py --name google/gemma-2-2b --tag 2bbase_sweep --sweep
cell gate_induction_2b gate_dont_delete.py \
  python gate_dont_delete.py --name-base google/gemma-2-2b --name-it google/gemma-2-2b-it --select induction
cell multisample_2b controls/cave_multisample_caverate.py \
  python cave_multisample_caverate.py --base google/gemma-2-2b --it google/gemma-2-2b-it --big-pool --device cuda

# ---- R12: 9b mechanism holes ---------------------------------------------------------------
cell gate_9b gate_dont_delete.py \
  python gate_dont_delete.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it
cell forcedchoice_9b job_forcedchoice.py \
  python job_forcedchoice.py --name google/gemma-2-9b --tag 9bbase
cell numeric_mech_9b job_numeric_mechanism.py \
  python job_numeric_mechanism.py --name google/gemma-2-9b
cell distractor_9b_base job_distractor_task.py \
  python job_distractor_task.py --model base --name google/gemma-2-9b --tag 9bbase
cell distractor_9b_it job_distractor_task.py \
  python job_distractor_task.py --model it --name google/gemma-2-9b-it --tag 9bit
cell instr_triang_9b instr_triangulation.py \
  python instr_triangulation.py --name google/gemma-2-9b

echo "### what landed ###"
ls -la out/*.json 2>/dev/null | tail -25

python - <<'PY'
import json, datetime
try:
    p = json.load(open("out/provenance_r6r12.json"))
    p["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(p, open("out/provenance_r6r12.json", "w"), indent=2)
except Exception as e: print("provenance close failed:", e)
PY
echo "ALLDONE_R6R12_SMALL_RIDERS"
