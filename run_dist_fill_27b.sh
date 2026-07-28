#!/usr/bin/env bash
# ============================================================================================
# THE 27b HALF OF THE EXHAUSTIVE DISTRIBUTIONAL GRID. Three things, one 27b box, no new
# registration and no code change:
#
#   1. B1's LISTEN ARM at 27b-base and 27b-it — the two cells missing from
#      docs/drafts/REGISTRATION_listen_distributional.md's six-cell scope. The other four ran and
#      the acceptance gate PASSED on all of them (out/b1_fold_identity_gate.json: 23 pre-existing
#      fields identical over 82 items, 4/4 cells), so the instrument is already gated — these two
#      cells extend a validated arm rather than testing it.
#   2. family_generate_judge on ext2-82 at 27b-base and 27b-it — present at 9b-base ONLY.
#   3. verify_graph_poc's T3 gate at 27b-base and 27b-it — present at 9b-base ONLY, on VF22.
#
# THE GRID THIS IS FILLING, measured rather than asserted: across the six distributional instruments
# x six cells x two families, 24 of 72 combinations exist and only 4 carry a listen arm. This runner
# closes 6 of the absent ones.
#
# T_PRE IS NOT HERE ON PURPOSE. verify_graph_poc's first gate is MODEL-FREE (its own docstring: "runs
# with NO torch") and it is a property of the FAMILY, not the model — so it needs one run per family,
# not one per cell, and it already ran offline on the workstation for $0:
# out/verify_graph_poc_vfam_ext2_TPRE.json = VALID, n=82, collision_frac 0.000, 82/82 entity answers.
# Only --pre-only=false (the T3 half) needs a GPU. Spending a 27b box-hour on T_PRE would be waste.
#
# SELF-JUDGE CAVEAT, stated before the run rather than discovered after: family_generate_judge's
# same-model self-judge is on record as DEGENERATE at base (it has never emitted a WRONG label at a
# base cell), and the measurement layer's v2 stance is commit-only-faithful with the judge diagnostic.
# So the 27b-base judge_label column is expected to carry no information. What the cell is FOR is its
# generations and its commit_prog labels, which are what the v2 readout actually uses.
#
# 27b bf16 ~54 GB -> 80 GB card. Fail-soft per cell. Files land FLAT in ~/latent_verify.
# ============================================================================================
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
mkdir -p out

python - <<'PY' > out/provenance_dist_27b.json 2>/dev/null || echo '{"provenance":"FAILED"}' > out/provenance_dist_27b.json
import json, os, subprocess, sys, datetime
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
def ver(m):
    try:
        from importlib.metadata import version; return version(m)
    except Exception: return None
p = {"run": "dist_fill_27b", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
cat out/provenance_dist_27b.json

echo "=== selftests ==="
python family_cave_diagnose_arms.py --selftest || { echo "SELFTEST_FAIL_ARMS"; exit 1; }
python family_generate_judge.py --selftest     || { echo "SELFTEST_FAIL_JUDGE"; exit 1; }
python verify_graph_poc.py --selftest          || { echo "SELFTEST_FAIL_VGP"; exit 1; }

cell () {  # $1 label, rest = command
  local l="$1"; shift
  echo "=== CELL $l ==="; "$@" > "out/dist27b_$l.log" 2>&1
  echo "exit=$?  ($l)"; tail -15 "out/dist27b_$l.log"; echo "--- end $l ---"
}

B=google/gemma-2-27b
I=google/gemma-2-27b-it
FAM=verifier_family_ext2.json

# 1. the two missing listen cells (highest value: completes B1's registered six-cell scope)
cell arms_27bbase python family_cave_diagnose_arms.py --family $FAM --name $B --tag vfam_ext2_27bbase --device cuda --arm both
cell arms_27bit   python family_cave_diagnose_arms.py --family $FAM --name $I --tag vfam_ext2_27bit   --device cuda --arm both --chat
# 2. generate+judge
cell judge_27bbase python family_generate_judge.py --family $FAM --name $B --tag vfam_ext2_27bbase --device cuda
cell judge_27bit   python family_generate_judge.py --family $FAM --name $I --tag vfam_ext2_27bit   --device cuda --chat
# 3. verify_graph_poc T3 (T_PRE already done offline, family-level)
cell vgp_27bbase python verify_graph_poc.py --family $FAM --name $B --tag vfam_ext2_27bbase --device cuda
cell vgp_27bit   python verify_graph_poc.py --family $FAM --name $I --tag vfam_ext2_27bit   --device cuda --chat

echo "=== per-arm summary for the two arms cells (in the log even if the fetch fails) ==="
python - <<'PY'
import glob, json
for f in sorted(glob.glob("out/family_cave_diagnose_arms_vfam_ext2_27b*.json")):
    d=json.load(open(f)); r=d.get("result",d)
    print("==", f.split("/")[-1], "|", d.get("name"))
    for arm,blk in (r.get("per_arm") or {}).items():
        a=blk.get("aggregate") or {}
        print(f"   {arm:7s} n={blk.get('n_records')} headroom={a.get('n_headroom_pass') or a.get('n_headroom')} "
              f"fRA={a.get('n_faithful_RA')} fRC={a.get('n_faithful_RC')} "
              f"meanRC={a.get('mean_RC_effect_headroom')} -> {(blk.get('decision') or {}).get('category')}")
PY
python - <<'PY'
import json, datetime
try:
    p=json.load(open("out/provenance_dist_27b.json"))
    p["finished_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(p,open("out/provenance_dist_27b.json","w"),indent=2)
except Exception as e: print("provenance close failed:",e)
PY
echo "ALLDONE_DIST_FILL_27B"
