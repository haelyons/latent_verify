#!/usr/bin/env bash
# ============================================================================================
# THE 2b/9b HALF OF THE EXHAUSTIVE DISTRIBUTIONAL GRID. Run-only, no code change, no new
# registration — every instrument carries its own frozen decision rule.
#
#   family_generate_judge on ext2-82 at 2b-base, 2b-it, 9b-it   (present at 9b-base ONLY)
#   verify_graph_poc T3   on ext2-82 at 2b-base, 2b-it, 9b-base, 9b-it
#
# Measured grid this fills: 24 of 72 (instrument x cell x family) combinations exist across the six
# distributional instruments. This runner closes 7 absent ones; run_dist_fill_27b.sh closes 6 more.
#
# T_PRE IS DELIBERATELY ABSENT. verify_graph_poc's first gate is model-free (its docstring: "runs with
# NO torch") and is a property of the FAMILY not the model, so it needs ONE run per family and it
# already ran on the workstation for $0 -> out/verify_graph_poc_vfam_ext2_TPRE.json (VALID, n=82,
# collision_frac 0.000, 82/82 entity answers). Only the T3 half needs a model. A per-cell T_PRE would
# have been six identical answers bought with six model loads.
#
# SELF-JUDGE CAVEAT, on record before the run: family_generate_judge's same-model self-judge is
# DEGENERATE at base cells (no WRONG label has ever been emitted at one) and the v2 measurement layer
# treats the judge as diagnostic-only. The base cells here are FOR their generations and commit_prog
# labels, which is what the v2 readout uses; their judge_label column is expected to be uninformative.
#
# 2b (~5 GB) + 9b (~18 GB) share one 40 GB box. Fail-soft per cell. Files land FLAT in ~/latent_verify.
# ============================================================================================
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
mkdir -p out

python - <<'PY' > out/provenance_dist_small.json 2>/dev/null || echo '{"provenance":"FAILED"}' > out/provenance_dist_small.json
import json, os, subprocess, sys, datetime
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
def ver(m):
    try:
        from importlib.metadata import version; return version(m)
    except Exception: return None
p = {"run": "dist_fill_small", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
cat out/provenance_dist_small.json

echo "=== selftests ==="
python family_generate_judge.py --selftest || { echo "SELFTEST_FAIL_JUDGE"; exit 1; }
python verify_graph_poc.py --selftest      || { echo "SELFTEST_FAIL_VGP"; exit 1; }

cell () { local l="$1"; shift
  echo "=== CELL $l ==="; "$@" > "out/distsm_$l.log" 2>&1
  echo "exit=$?  ($l)"; tail -15 "out/distsm_$l.log"; echo "--- end $l ---"; }

FAM=verifier_family_ext2.json

cell judge_2bbase python family_generate_judge.py --family $FAM --name google/gemma-2-2b    --tag vfam_ext2_2bbase --device cuda
cell judge_2bit   python family_generate_judge.py --family $FAM --name google/gemma-2-2b-it --tag vfam_ext2_2bit   --device cuda --chat
cell judge_9bit   python family_generate_judge.py --family $FAM --name google/gemma-2-9b-it --tag vfam_ext2_9bit   --device cuda --chat

cell vgp_2bbase python verify_graph_poc.py --family $FAM --name google/gemma-2-2b    --tag vfam_ext2_2bbase --device cuda
cell vgp_2bit   python verify_graph_poc.py --family $FAM --name google/gemma-2-2b-it --tag vfam_ext2_2bit   --device cuda --chat
cell vgp_9bbase python verify_graph_poc.py --family $FAM --name google/gemma-2-9b    --tag vfam_ext2_9bbase --device cuda
cell vgp_9bit   python verify_graph_poc.py --family $FAM --name google/gemma-2-9b-it --tag vfam_ext2_9bit   --device cuda --chat

echo "=== decisions on the box, so they survive a failed fetch ==="
python - <<'PY'
import glob, json
for pat in ("out/family_generate_judge_vfam_ext2_*.json","out/verify_graph_poc_vfam_ext2_*.json"):
    for f in sorted(glob.glob(pat)):
        try: d=json.load(open(f))
        except Exception as e: print("UNREADABLE",f,e); continue
        r=d.get("result",d)
        dec=r.get("decision") or d.get("decision")
        print(f"  {f.split('/')[-1]:52s} {d.get('name')}  {json.dumps(dec)[:120]}")
PY
python - <<'PY'
import json, datetime
try:
    p=json.load(open("out/provenance_dist_small.json"))
    p["finished_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(p,open("out/provenance_dist_small.json","w"),indent=2)
except Exception as e: print("provenance close failed:",e)
PY
echo "ALLDONE_DIST_FILL_SMALL"
