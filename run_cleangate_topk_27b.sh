#!/usr/bin/env bash
# ============================================================================================
# THREE JOBS, ONE 80 GB BOX. The first is a test I should have designed before the 27b listen cells
# ran and did not.
#
# 1. THE CLEAN FOLD-IDENTITY TEST, which removes hardware from the comparison.
#    out/b1_fold_identity_gate_27b.json records the problem: the fold arm of
#    family_cave_diagnose_arms.py DIFFERS from the shipped instrument's committed 27b artifacts on
#    43-81 of 82 items, but the reference ran on an H100 PCIe at driver 570.148.08 and the re-run on
#    an H100 80GB HBM3 at driver 580.105.08 — so code drift and hardware are CONFOUNDED and that
#    comparison cannot separate them. Verdict on record: GATE_UNEVALUABLE_AT_27B_CONFOUNDED, with the
#    27b listen numbers HELD and not quotable.
#    The fix is to run BOTH instruments on ONE box, in one process order, same weights, same driver:
#      shipped  family_cave_diagnose.py            (fold by construction)
#      new      family_cave_diagnose_arms.py --arm fold
#    Diffed offline. Pre-existing fields equal -> the re-parameterisation IS algebraically neutral and
#    the earlier 27b failure was entirely the machine, which releases the 27b listen numbers AND
#    independently confirms the cross-box divergence. Any field differing -> the code is at fault, the
#    2b/9b passes were luck, and B1's listen numbers are withdrawn at every scale.
#    Note the asymmetry deliberately: this test can only be read one way per outcome, and the
#    2b/9b cells already PASSED against same-hardware references, so a difference here would be the
#    surprising result and would be believed over them.
#
# 2. THE topk_shift LISTEN ARM at 27b-base and 27b-it. controls/family_topk_shift_arms.py, claim-blind
#    authored, selftest PASSES locally, same plant/target pattern already gate-validated on the
#    diagnose sibling. This is the instrument that made the repo's rank / top-k / top_riser story
#    one-directional (docs/drafts/DIST_COVERAGE.md gap 1: no listen arm at ANY cell).
#    ITS OWN FOLD ARM RIDES THE SAME CLEAN TEST: it also runs shipped family_topk_shift.py here, so
#    both re-parameterisations get a same-box reference in one go.
#
# 3. instr_triangulation at 9b — the OWED A/E4 rider. It OOM'd on a 40 GB A100 holding 38.60 of
#    39.49 GiB alone, because it calls .backward(); no flag shrinks it without deleting a leg
#    (--no-knockout-sweep removes one). Full scope on 80 GB, riding a box already up, because it is
#    one claim and 80 GB capacity is scarce. It runs LAST so it cannot cost the 27b work.
#
# Fail-soft per cell. Files land FLAT in ~/latent_verify.
# ============================================================================================
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
mkdir -p out

python - <<'PY' > out/provenance_cleangate.json 2>/dev/null || echo '{"provenance":"FAILED"}' > out/provenance_cleangate.json
import json, os, subprocess, sys, datetime
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
def ver(m):
    try:
        from importlib.metadata import version; return version(m)
    except Exception: return None
p = {"run": "cleangate_topk_27b", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
     "python": sys.version.split()[0],
     "lambda_instance_id": os.environ.get("LAMBDA_INSTANCE_ID"),
     "git_commit": os.environ.get("GIT_COMMIT"),
     "gpu_name": sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
     "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
     "dtype": "bfloat16", "torch": ver("torch"), "transformers": ver("transformers"),
     "transformer_lens": ver("transformer_lens")}
try: import torch; p["cuda_runtime"] = torch.version.cuda
except Exception: p["cuda_runtime"] = None
print(json.dumps(p, indent=2))
PY
cat out/provenance_cleangate.json

echo "=== selftests ==="
python family_cave_diagnose.py --selftest       || { echo "SELFTEST_FAIL_SHIPPED_DIAG"; exit 1; }
python family_cave_diagnose_arms.py --selftest  || { echo "SELFTEST_FAIL_ARMS_DIAG"; exit 1; }
python family_topk_shift.py --selftest          || { echo "SELFTEST_FAIL_SHIPPED_TOPK"; exit 1; }
python family_topk_shift_arms.py --selftest     || { echo "SELFTEST_FAIL_ARMS_TOPK"; exit 1; }

cell () { local l="$1"; shift
  echo "=== CELL $l ==="; "$@" > "out/cg_$l.log" 2>&1
  echo "exit=$?  ($l)"; tail -12 "out/cg_$l.log"; echo "--- end $l ---"; }

FAM=verifier_family_ext2.json
B=google/gemma-2-27b

# --- 1. the clean same-box pair: shipped vs re-parameterised, fold, 27b-base -------------------
cell diag_shipped python family_cave_diagnose.py      --family $FAM --name $B --tag cleangate_27bbase_shipped --device cuda
cell diag_arms    python family_cave_diagnose_arms.py --family $FAM --name $B --tag cleangate_27bbase_arms    --device cuda --arm fold
cell topk_shipped python family_topk_shift.py         --family $FAM --name $B --tag cleangate_27bbase_shipped --device cuda
cell topk_arms    python family_topk_shift_arms.py    --family $FAM --name $B --tag cleangate_27bbase_arms    --device cuda --arm fold

# --- 2. the topk listen arm at both 27b cells --------------------------------------------------
cell topkarms_27bbase python family_topk_shift_arms.py --family $FAM --name $B --tag vfam_ext2_27bbase --device cuda --arm both
cell topkarms_27bit   python family_topk_shift_arms.py --family $FAM --name google/gemma-2-27b-it --tag vfam_ext2_27bit --device cuda --arm both --chat

# --- 3. the OOM rider, LAST so it cannot cost the 27b work -------------------------------------
cell instr_triang_9b python instr_triangulation.py --name google/gemma-2-9b

echo "=== the clean test, computed ON THE BOX so the verdict survives a failed fetch ==="
python - <<'PY'
import json, itertools
def load(p):
    d=json.load(open(p)); r=d.get("result",d); return d,r
for shipped,arms,label in (("out/family_cave_diagnose_cleangate_27bbase_shipped.json",
                            "out/family_cave_diagnose_arms_cleangate_27bbase_arms.json","diagnose"),
                           ("out/family_topk_shift_cleangate_27bbase_shipped.json",
                            "out/family_topk_shift_arms_cleangate_27bbase_arms.json","topk_shift")):
    try: ds,rs = load(shipped); da,ra = load(arms)
    except Exception as e:
        print(f"[cleangate {label}] UNAVAILABLE: {e}"); continue
    si = rs["items"]; ai=[x for x in ra["items"] if x.get("arm","fold")=="fold"]
    if [x["q"] for x in si] != [x["q"] for x in ai]:
        print(f"[cleangate {label}] ITEM ORDER DIFFERS -> unevaluable"); continue
    pre=list(si[0]); bad={}
    for a,b in zip(ai,si):
        for k in pre:
            if a.get(k)!=b.get(k): bad[k]=bad.get(k,0)+1
    print(f"[cleangate {label}] n={len(si)} pre_fields={len(pre)} differing={bad if bad else '{} -> ALGEBRAICALLY NEUTRAL'}")
PY

python - <<'PY'
import json, datetime
try:
    p=json.load(open("out/provenance_cleangate.json"))
    p["finished_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(p,open("out/provenance_cleangate.json","w"),indent=2)
except Exception as e: print("provenance close failed:",e)
PY
echo "ALLDONE_CLEANGATE_TOPK_27B"
