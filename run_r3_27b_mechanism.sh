#!/usr/bin/env bash
# ============================================================================================
# R3 — THE 27b MECHANISM COLUMN. Ledger GAPS_RECONCILED.md 4.3 row R3 (3 claims), plus the
# 27b x copy cell of cave_headset_specificity. 27b is "the missing column of the whole mechanism +
# margin program" (reconciled ledger 1.1 row 3), reached independently by all three blind audits.
#
# NO NEW REGISTRATION IS NEEDED AND NONE IS BEING MADE. Every instrument below carries its own
# frozen metric / thresholds / decision_rule and its own neutral decision; this runner is thin, per
# house style ("Decision rules live in the controls (neutral, unchanged); this runner is thin").
# The one thing an audit had to settle first was whether 27b was code-blocked here: it is not.
# docs/drafts/CODEBLOCKS_verified.md D7 — cave_fold_vs_listen.py:80 READ_LAYER = 28 is SUPERSEDED at
# run time by pick_read_layer(n_layers) at :91, and the artifacts confirm it (read_layer 28 at 9b,
# 17 at 2b). So this is GPU RUN, no code change.
#
# FAIL-SOFT BY DESIGN. Each instrument writes its own artifact, so a failure in one must not cost the
# others their box-hours. Every cell logs its own exit code and the run continues. Ordered by value,
# so that if the cap truncates the run the most valuable cells are already banked:
#   1. cave_fold_vs_listen   -- the fold/listen mechanism battery at the missing scale (D1/D7)
#   2. cave_residstate_*     -- the residual cave-state readout at 27b
#   3. cave_faithful_it_*    -- the -it faithful readout at 27b
#   4. faithful_copy_wstar / prompt_feature / headset_specificity(copy)
#
# THE `datasets` LINE IS NOW LOAD-BEARING, NOT DEFENSIVE. These controls build the 891-item big pool,
# 817 items of which are the TruthfulQA generation split. As of 2026-07-28
# controls/cave_copy_confidence_conditional.py:328-333 RAISES instead of silently returning a
# 74-item pool (OWED A3), so a missing `datasets` is now a hard stop rather than a silent
# substrate swap. The precedent runners (run_doubt_27b.sh, run_calib_27b.sh) already pip it in.
#
# 27b bf16 ~54 GB -> needs an 80 GB card. Files land FLAT in ~/latent_verify.
# ============================================================================================
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
mkdir -p out

python - <<'PY' > out/provenance_r3_27b.json 2>/dev/null || echo '{"provenance":"FAILED"}' > out/provenance_r3_27b.json
import json, os, subprocess, sys, datetime
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
def ver(m):
    try:
        from importlib.metadata import version; return version(m)
    except Exception: return None
p = {"run": "r3_27b_mechanism",
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
cat out/provenance_r3_27b.json

python -c "import datasets" 2>/dev/null || pip install -q datasets 2>&1 | tail -2 || echo "DATASETS_PIP_FAILED"
python -c "import datasets, importlib.metadata as m; print('[deps] datasets', m.version('datasets'))" \
  || echo "DATASETS_STILL_MISSING -- the big pool will RAISE (this is the A3 fix working, not a new bug)"

B=google/gemma-2-27b
I=google/gemma-2-27b-it

cell () {  # $1 = label, rest = command
  local label="$1"; shift
  echo "=== CELL $label ==="
  "$@" > "out/r3_${label}.log" 2>&1
  local rc=$?
  echo "exit=$rc  ($label)"
  tail -12 "out/r3_${label}.log"
  echo "--- end $label ---"
}

echo "### selftests first: a module that cannot import must not burn a 27b load ###"
for m in cave_fold_vs_listen cave_residstate_anyscale cave_residstate_decisive cave_faithful_it_diff \
         cave_faithful_it_mc faithful_copy_wstar cave_prompt_feature_mechanism cave_headset_specificity; do
  python "$m.py" --selftest > "out/r3_selftest_$m.log" 2>&1 && echo "  selftest OK   $m" \
    || echo "  selftest FAIL $m (cell will still be attempted; see out/r3_selftest_$m.log)"
done

cell foldlisten_27b   python cave_fold_vs_listen.py            --base "$B" --it "$I" --big-pool --device cuda
cell residstate_any   python cave_residstate_anyscale.py       --base "$B" --it "$I" --big-pool --device cuda --tag 27b
cell residstate_dec   python cave_residstate_decisive.py       --base "$B" --it "$I" --big-pool --device cuda
cell faithful_it_diff python cave_faithful_it_diff.py          --base "$B" --it "$I" --big-pool --device cuda --tag 27b
cell faithful_it_mc   python cave_faithful_it_mc.py            --base "$B" --it "$I" --big-pool --device cuda --tag 27b
cell copy_wstar       python faithful_copy_wstar.py            --name-base "$B" --name-it "$I" --device cuda --tag 27b
cell promptfeat       python cave_prompt_feature_mechanism.py  --name-base "$B" --name-it "$I" --big-pool --device cuda --tag 27b
cell hsspec_copy      python cave_headset_specificity.py --mode copy --name-base "$B" --name-it "$I" --big-pool --device cuda --tag 27b

echo "### what landed ###"
ls -la out/*27b*.json 2>/dev/null | tail -20

python - <<'PY'
import json, datetime
try:
    p = json.load(open("out/provenance_r3_27b.json"))
    p["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(p, open("out/provenance_r3_27b.json", "w"), indent=2)
except Exception as e: print("provenance close failed:", e)
PY
echo "ALLDONE_R3_27B_MECHANISM"
