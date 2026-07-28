#!/usr/bin/env bash
# ============================================================================================
# B1 — THE LISTEN-DIRECTION DISTRIBUTIONAL ARM, at 2b and 9b. Pre-registered in
# docs/drafts/REGISTRATION_listen_distributional.md (registration owed #1, ledger gap K1, 12 claims —
# the largest block in the code-first class, and per GAPS_B "the single widest readout gap").
#
# WHY IT EXISTED. Every teacher-forced / top-k / rank instrument in this repo plants the literal C in
# both arms (controls/family_cave_diagnose.py:214-215), so all of them measure the FOLD direction only.
# There is a generation-level listen result at every scale and no probability-level one anywhere.
#
# WHAT RUNS. controls/family_cave_diagnose_arms.py --arm both. It is a plant/target
# re-parameterisation that IMPORTS every threshold and pure helper from the shipped instrument rather
# than copying them, so there is exactly one definition of each and the shipped file is untouched.
#   fold:   plant=C,  target=W*   (algebraically the shipped code)
#   listen: plant=W*, target=C
#
# THE ACCEPTANCE GATE RIDES ALONG FOR FREE. --arm both emits the fold block too, so each of the four
# cells here doubles as a fold-identity check against an artifact already on disk:
#   9b-base -> results_absdecode_ext2/out/family_cave_diagnose_vfam_ext2_9bbase.json
#   9b-it   -> results_itreadout_modelw/out/family_cave_diagnose_vfam_ext2_9bit.json
#   2b-base -> results_r1_dist_2b9b/out/family_cave_diagnose_vfam_ext2_2bbase.json
#   2b-it   -> results_r1_dist_2b9b/out/family_cave_diagnose_vfam_ext2_2bit.json
# FOUR independent gate checks from four cells that had to run anyway. The comparison itself is
# OFFLINE on the workstation, where all four references live. Per the registration's amended gate:
# every PRE-EXISTING field identical item-for-item (the new `arm`/`stamp`/plant-target keys are
# additive and appended, so whole-file bytes cannot match and are not the test).
#
# IF THE GATE FAILS the listen numbers from this run are NOT usable — the re-parameterisation would not
# be algebraically neutral, and a listen number computed by drifted code measures nothing. Registered
# that way on purpose: the gate is not advisory.
#
# THE PREDICTION ON RECORD (registration §4), so it cannot become a post-hoc excuse: the listen arm
# plants W*, which the model mostly does not believe, so |M0| should be large and headroom_pass should
# reject far MORE listen items than fold. If it does, that is a FINDING — the listen direction has
# little near-tie regime at the probability level — and widening MARGIN_KEEP to recover items is
# PROHIBITED. n_headroom_pass per arm is reported either way.
#
# 2b (~5 GB) and 9b (~18 GB) share one 40 GB box. 27b is a separate box, deliberately not here.
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

python - <<'PY' > out/provenance_b1_2b9b.json 2>/dev/null || echo '{"provenance":"FAILED"}' > out/provenance_b1_2b9b.json
import json, os, subprocess, sys, datetime
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
def ver(m):
    try:
        from importlib.metadata import version; return version(m)
    except Exception: return None
p = {"run": "b1_listen_arm_2b9b",
     "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
     "python": sys.version.split()[0],
     "lambda_instance_id": os.environ.get("LAMBDA_INSTANCE_ID"),
     "git_commit": os.environ.get("GIT_COMMIT"),
     "gpu_name": sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
     "gpu_count": sh("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"),
     "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
     "dtype": "bfloat16", "torch": ver("torch"), "transformers": ver("transformers"),
     "transformer_lens": ver("transformer_lens")}
try: import torch; p["cuda_runtime"] = torch.version.cuda
except Exception: p["cuda_runtime"] = None
print(json.dumps(p, indent=2))
PY
cat out/provenance_b1_2b9b.json

echo "=== selftest: model-free, and it asserts the fold path's arithmetic against the shipped formulas ==="
python family_cave_diagnose_arms.py --selftest || { echo "SELFTEST_FAIL_ARMS"; exit 1; }
python family_cave_diagnose.py --selftest      || { echo "SELFTEST_FAIL_SHIPPED"; exit 1; }

cell () {  # $1 tag, $2 model, $3 chat-flag-or-empty
  echo "=== CELL $1 (arm=both) ==="
  python family_cave_diagnose_arms.py --family verifier_family_ext2.json --name "$2" --tag "$1" \
    --device cuda --arm both $3 > "out/b1_$1.log" 2>&1
  echo "exit=$?  ($1)"; tail -18 "out/b1_$1.log"; echo "--- end $1 ---"
}

cell vfam_ext2_9bbase google/gemma-2-9b    ""
cell vfam_ext2_9bit   google/gemma-2-9b-it "--chat"
cell vfam_ext2_2bbase google/gemma-2-2b    ""
cell vfam_ext2_2bit   google/gemma-2-2b-it "--chat"

echo "=== per-arm headroom + decision, printed on the box so it is in the log even if a fetch fails ==="
python - <<'PY'
import glob, json
for f in sorted(glob.glob("out/family_cave_diagnose_arms_*.json")):
    try: d = json.load(open(f))
    except Exception as e: print("UNREADABLE", f, e); continue
    r = d.get("result", d)
    print("==", f.split("/")[-1], "| model:", d.get("name"))
    for arm, blk in (r.get("per_arm") or {}).items():
        a = blk.get("aggregate") or {}
        print(f"   {arm:7s} n={blk.get('n_records')} headroom_pass={a.get('n_headroom_pass') or a.get('n_headroom')} "
              f"faithful_RA={a.get('n_faithful_RA')} faithful_RC={a.get('n_faithful_RC')} "
              f"decision={(blk.get('decision') or {}).get('category')} {blk.get('threshold_provenance','')}")
PY

python - <<'PY'
import json, datetime
try:
    p = json.load(open("out/provenance_b1_2b9b.json"))
    p["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(p, open("out/provenance_b1_2b9b.json", "w"), indent=2)
except Exception as e: print("provenance close failed:", e)
PY
echo "ALLDONE_B1_LISTEN_ARM_2B9B"
