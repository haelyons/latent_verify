#!/usr/bin/env bash
# ============================================================================================
# REGISTRATION (house style: frozen in this header, committed BEFORE the run, both outcomes of
# every comparison written down in advance).
#
# THE QUESTION. The committed 27b-base fold/listen decode is not reproducible. Measured in
# docs/drafts/REDERIVE_20260728.md: between results_foldlisten_ext2_27b (the committed draw) and
# results_foldlisten_nelicit_27b (the re-run), 98/164 elicit_gens, 96/164 counter_gens and 41/164
# faithful_elicit labels differ, against 0/0/0 at 2b and 9b. RETRACTIONS.md R-1 had to WITHDRAW the
# two sentences that attributed this to hardware, because no artifact records a machine.
#
# WHAT IS ALREADY SETTLED, and what it does not settle. out/../results_r1_dist_27b/out/
# r1_27b_determinism_rider.json = WITHIN_BOX_DETERMINISTIC: two identical family_cave_diagnose
# passes at 27b-base on one H100 PCIe agreed on 0 of 14 numeric fields across 82 items. That is the
# TEACHER-FORCED FORWARD layer. It does NOT cover the DECODE path, which is where the divergence
# actually lives: model.generate with a KV cache, different kernels and different batch shapes.
# This run tests that layer.
#
# THE DESIGN. foldlisten_judge on the SAME family (verifier_family_ext2.json, 82 items) at
# 27b-base, TWICE, on ONE box, in one process order, greedy. Then three comparisons via the
# committed gate controls/foldlisten_repro_diff.py (selftest PASSES, 12 groups, thresholds already
# frozen; NOT modified for this run):
#
#   C1  pass A vs pass B          -> WITHIN-BOX decode determinism
#   C2  pass A vs the COMMITTED draw (results_foldlisten_ext2_27b)
#   C3  pass A vs the RE-RUN draw  (results_foldlisten_nelicit_27b)
#
# REGISTERED OUTCOMES. Every cell of this table was written before the run.
#
#   C1 BYTE_IDENTICAL + (C2 or C3) BYTE_IDENTICAL
#       -> the decode is deterministic within a box AND one of the two existing draws is
#          reproducible on this hardware. The OTHER draw is the anomaly, and it is identified.
#          This is the outcome that would let a 27b column be published, naming its draw.
#   C1 BYTE_IDENTICAL + C2 DIFF + C3 DIFF
#       -> deterministic within a box, reproducible on NEITHER existing draw => the 27b decode
#          varies ACROSS boxes and both committed draws are box-specific. The 27b column cannot be
#          published as a measurement of the model; it is a measurement of a machine. R-1's
#          retraction is then correct for a second, independent reason.
#   C1 DIFF
#       -> the decode is NOT deterministic even within one box. No hardware record could ever have
#          made the committed 27b column reproducible, C2/C3 become uninterpretable, and every 27b
#          decode number in the repo must be reported with a run-to-run spread or withdrawn.
#          THIS OUTCOME WOULD SUPERSEDE the WITHIN_BOX_DETERMINISTIC forward result rather than
#          contradict it: forward-deterministic and decode-nondeterministic are compatible.
#
# NO THRESHOLD IS CHOSEN HERE. foldlisten_repro_diff.py's own frozen decision rule governs all
# three comparisons, unchanged, and its verdict is whatever it is.
# PRE-DATA SCOPE NOTE: this run is 27b-base ONLY. 27b-it is identical between the two existing
# draws, so it has nothing to reproduce; extending to 27b-it would be a different question.
#
# COST. ~4.3 h per 27b foldlisten cell on an H100 PCIe (docs/lambda-gpu-access.md), so ~8.6 h for
# two passes plus the ~54 GB weight pull. Cap 10 h. Files land FLAT in ~/latent_verify.
# ============================================================================================
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

# provenance FIRST, so it survives a dead cell. LAMBDA_INSTANCE_ID and GIT_COMMIT now arrive from
# the launcher (lambda_run.sh) -- the three nulls in the previous run's stamp were this gap.
python - <<'PY' > out/provenance_27b_decode_det.json 2>/dev/null || echo '{"provenance":"FAILED"}' > out/provenance_27b_decode_det.json
import json, os, subprocess, sys, datetime
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
def ver(m):
    try:
        from importlib.metadata import version; return version(m)
    except Exception:
        try: return __import__(m).__version__
        except Exception: return None
p = {"run": "27b_decode_determinism",
     "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
     "python": sys.version.split()[0],
     "lambda_instance_id": os.environ.get("LAMBDA_INSTANCE_ID"),
     "git_commit": os.environ.get("GIT_COMMIT"),
     "gpu_name": sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
     "gpu_count": sh("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"),
     "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
     "dtype": "bfloat16",
     "torch": ver("torch"), "transformers": ver("transformers"),
     "transformer_lens": ver("transformer_lens")}
try: import torch; p["cuda_runtime"] = torch.version.cuda
except Exception: p["cuda_runtime"] = None
print(json.dumps(p, indent=2))
PY
cat out/provenance_27b_decode_det.json

echo "=== selftests (model-free; stop before a box-hour is spent) ==="
python foldlisten_judge.py --selftest      || { echo "SELFTEST_FAIL_JUDGE";   exit 1; }
python faithful_rescore.py --selftest      || { echo "SELFTEST_FAIL_RESCORE"; exit 1; }

echo "=== PASS A :: 27b-base fold/listen decode, ext2-82 ==="
python foldlisten_judge.py --family verifier_family_ext2.json --name google/gemma-2-27b \
  --tag fl_27bbase_ext2_passA --device cuda > out/foldlisten_judge_fl_27bbase_ext2_passA.log 2>&1
echo "exit=$?"; tail -20 out/foldlisten_judge_fl_27bbase_ext2_passA.log

echo "=== PASS B :: identical invocation, same box, same process order ==="
python foldlisten_judge.py --family verifier_family_ext2.json --name google/gemma-2-27b \
  --tag fl_27bbase_ext2_passB --device cuda > out/foldlisten_judge_fl_27bbase_ext2_passB.log 2>&1
echo "exit=$?"; tail -20 out/foldlisten_judge_fl_27bbase_ext2_passB.log

echo "=== the box's job ends with the two passes ==="
# ALL THREE comparisons (C1 pass A vs pass B, C2 vs the committed draw, C3 vs the re-run draw) are
# computed OFFLINE on the workstation with controls/foldlisten_repro_diff.py --committed/--new. That
# gate is not in the launcher's scp list and does not need to be: both committed draws live on the
# workstation, the gate is CPU-only, and keeping it off the box means no box-hour is spent on it and
# no untested file is shipped. The gate's own frozen decision rule governs, unchanged.
ls -la out/foldlisten_judge_fl_27bbase_ext2_pass*_summary.json

python - <<'PY'
import json, datetime
try:
    p = json.load(open("out/provenance_27b_decode_det.json"))
    p["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(p, open("out/provenance_27b_decode_det.json", "w"), indent=2)
except Exception as e:
    print("provenance close failed:", e)
PY
echo "ALLDONE_27B_DECODE_DETERMINISM"
# C2 and C3 (pass A against each committed draw) run OFFLINE on the workstation, where both
# committed draws live -- no box-hour is spent shipping them up.
