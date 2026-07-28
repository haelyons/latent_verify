#!/usr/bin/env bash
# R1 (partial) + R2: the ext2-82 DISTRIBUTIONAL readout at the 2b cells and the 9b-it hole.
# GAPS_RECONCILED.md 4.3 R1 is the highest-claim GPU row in the ledger (11 claims): every
# teacher-forced / top-k artifact in the repo is 9b-only, so the whole probability story has no
# scale axis. Both instruments already take --family/--name/--chat, so NO code change was needed
# and no threshold moves: the decision rules are the controls' own, frozen, unchanged.
# Cells here: 2b-base, 2b-it, and topk at 9b-it (9b-base and 9b-it diagnose already exist).
# The 27b half runs on its own box (run_r1_dist_27b.sh) because 27b bf16 needs 80GB.
# Files land FLAT in ~/latent_verify.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

# ---- provenance stamp (REGISTRATION_provenance.md 1). Written FIRST so it exists even if a cell
# ---- dies: the whole point is that a future agent can attribute a number to a machine.
python - <<'PY' > out/provenance_r1_2b9b.json 2>/dev/null || echo '{"provenance":"FAILED"}' > out/provenance_r1_2b9b.json
import json, subprocess, sys, datetime
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
p = {"run": "r1_dist_2b9b", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
     "python": sys.version.split()[0], "git_commit": sh("git rev-parse HEAD"),
     "gpu_name": sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
     "gpu_count": sh("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"),
     "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
     "cuda_smi": sh("nvidia-smi | head -3 | tail -1"),
     "lambda_instance_id": None, "dtype": "bfloat16"}
for m in ("torch", "transformers", "transformer_lens"):
    try: p[m] = __import__(m).__version__
    except Exception: p[m] = None
try: import torch; p["cuda_runtime"] = torch.version.cuda
except Exception: p["cuda_runtime"] = None
print(json.dumps(p, indent=2))
PY
cat out/provenance_r1_2b9b.json

echo "=== selftests (model-free; a failure here stops the run before a box-hour is spent) ==="
python family_topk_shift.py --selftest    || { echo "SELFTEST_FAIL_TOPK";     exit 1; }
python family_cave_diagnose.py --selftest || { echo "SELFTEST_FAIL_DIAGNOSE"; exit 1; }

run_cell () {   # $1=script $2=model $3=tag $4=chat-flag-or-empty
  echo "=== $1 :: $3 ==="
  python "$1" --family verifier_family_ext2.json --name "$2" --tag "$3" --device cuda $4 \
    > "out/$(basename "$1" .py)_$3.log" 2>&1
  echo "exit=$?"; tail -12 "out/$(basename "$1" .py)_$3.log"
}

# 2b: both variants, both instruments (4 of R1's 32 absent cells)
run_cell family_topk_shift.py    google/gemma-2-2b    vfam_ext2_2bbase ""
run_cell family_cave_diagnose.py google/gemma-2-2b    vfam_ext2_2bbase ""
run_cell family_topk_shift.py    google/gemma-2-2b-it vfam_ext2_2bit   "--chat"
run_cell family_cave_diagnose.py google/gemma-2-2b-it vfam_ext2_2bit   "--chat"

# 9b-it: topk only. R2's hole -- family_cave_diagnose at 9b-it already exists
# (results_itreadout_modelw/out/family_cave_diagnose_vfam_ext2_9bit.json), topk does not.
run_cell family_topk_shift.py    google/gemma-2-9b-it vfam_ext2_9bit   "--chat"

python - <<'PY'
import json, datetime
p = json.load(open("out/provenance_r1_2b9b.json"))
p["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
json.dump(p, open("out/provenance_r1_2b9b.json", "w"), indent=2)
PY
echo "ALLDONE_R1_DIST_2B9B"
