#!/usr/bin/env bash
# R1 (27b half) + a DETERMINISM RIDER that this session's evidence made necessary.
#
# R1: the ext2-82 distributional readout at 27b-base and 27b-it. Both instruments already take
# --family/--name/--chat, so no code change and no threshold moves.
#
# THE RIDER, and why it is here. docs/drafts/REDERIVE_20260728.md measured that the 27b-base column
# is ONE DRAW: 98/164 elicit_gens, 96/164 counter_gens and 41/164 labels differ between the committed
# ext2 decode and the neutral-elicit re-run, against 0/0/0 at 2b and 9b. RETRACTIONS.md R-1 had to
# WITHDRAW the two sentences that attributed that to hardware, because no artifact records a machine.
# What has never been measured is the layer UNDERNEATH the decode: whether 27b's teacher-forced
# forward pass is even reproducible WITHIN one box. So family_cave_diagnose runs TWICE at 27b-base,
# same box, same process order, same greedy/forward-only path, second pass tagged _rep2. Diffed
# offline afterwards. This is registered here, before the data, as a two-outcome test:
#   * the two passes are IDENTICAL  -> 27b forward numerics are deterministic within a box, so the
#     decode divergence is NOT explained by within-box nondeterminism, and the across-box hypothesis
#     survives as the live one (now attributable, because this run stamps its hardware).
#   * the two passes DIFFER         -> 27b is nondeterministic within a box, and NO amount of
#     hardware bookkeeping would ever have made the committed 27b column reproducible. That would
#     make R-1's retraction correct for a second, stronger reason.
# Either way it is decisive, and it costs one extra forward pass over 82 items.
#
# 27b bf16 needs an 80GB card, hence its own box. Files land FLAT in ~/latent_verify.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

python - <<'PY' > out/provenance_r1_27b.json 2>/dev/null || echo '{"provenance":"FAILED"}' > out/provenance_r1_27b.json
import json, subprocess, sys, datetime
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
p = {"run": "r1_dist_27b", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
cat out/provenance_r1_27b.json

echo "=== selftests (model-free) ==="
python family_topk_shift.py --selftest    || { echo "SELFTEST_FAIL_TOPK";     exit 1; }
python family_cave_diagnose.py --selftest || { echo "SELFTEST_FAIL_DIAGNOSE"; exit 1; }

run_cell () {
  echo "=== $1 :: $3 ==="
  python "$1" --family verifier_family_ext2.json --name "$2" --tag "$3" --device cuda $4 \
    > "out/$(basename "$1" .py)_$3.log" 2>&1
  echo "exit=$?"; tail -12 "out/$(basename "$1" .py)_$3.log"
}

run_cell family_cave_diagnose.py google/gemma-2-27b    vfam_ext2_27bbase      ""
run_cell family_cave_diagnose.py google/gemma-2-27b    vfam_ext2_27bbase_rep2 ""   # the determinism rider
run_cell family_topk_shift.py    google/gemma-2-27b    vfam_ext2_27bbase      ""
run_cell family_cave_diagnose.py google/gemma-2-27b-it vfam_ext2_27bit        "--chat"
run_cell family_topk_shift.py    google/gemma-2-27b-it vfam_ext2_27bit        "--chat"

echo "=== determinism rider: pass 1 vs pass 2, same box, 27b-base ==="
python - <<'PY'
import json
try:
    a = json.load(open("out/family_cave_diagnose_vfam_ext2_27bbase.json"))
    b = json.load(open("out/family_cave_diagnose_vfam_ext2_27bbase_rep2.json"))
except Exception as e:
    print("RIDER_UNAVAILABLE", e); raise SystemExit(0)
ia = (a.get("result") or a)["items"]; ib = (b.get("result") or b)["items"]
assert [x["q"] for x in ia] == [x["q"] for x in ib], "item order differs -- rider invalid"
fields = [k for k in ia[0] if isinstance(ia[0][k], (int, float)) and not isinstance(ia[0][k], bool)]
diff = {f: sum(1 for x, y in zip(ia, ib) if x[f] != y[f]) for f in fields}
nz = {f: n for f, n in diff.items() if n}
res = {"control": "r1_27b_determinism_rider", "n_items": len(ia), "numeric_fields": len(fields),
       "fields_with_any_difference": nz, "n_fields_differing": len(nz),
       "decision": "WITHIN_BOX_DETERMINISTIC" if not nz else "WITHIN_BOX_NONDETERMINISTIC",
       "decision_rule": ("Registered in run_r1_dist_27b.sh before the data: two identical "
                         "family_cave_diagnose passes at 27b-base on one box. Zero differing numeric "
                         "fields -> WITHIN_BOX_DETERMINISTIC (the decode divergence is not explained by "
                         "within-box nondeterminism). Any differing field -> WITHIN_BOX_NONDETERMINISTIC "
                         "(no hardware record could ever have made the committed 27b column reproducible).")}
json.dump(res, open("out/r1_27b_determinism_rider.json", "w"), indent=2)
print(json.dumps({k: res[k] for k in ("n_items", "numeric_fields", "n_fields_differing", "decision")}, indent=1))
PY

python - <<'PY'
import json, datetime
p = json.load(open("out/provenance_r1_27b.json"))
p["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
json.dump(p, open("out/provenance_r1_27b.json", "w"), indent=2)
PY
echo "ALLDONE_R1_DIST_27B"
