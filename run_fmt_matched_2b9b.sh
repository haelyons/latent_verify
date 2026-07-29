#!/usr/bin/env bash
# =================================================================================================
# BOX A of TWO — the FORMAT-MATCHED base-vs-`-it` readout at the 2b and 9b cells.
#
# SPEC: docs/drafts/REGISTRATION_format_matched_readout.md — FROZEN, PRE-DATA, amended twice on
# 2026-07-29 (A1–A20, log at §0.2). This runner adds NO threshold, NO flag and NO decision of its
# own: the cell list below is DERIVED from §1 / §7 / §10 / §14 and nothing else, and every verdict
# is emitted OFFLINE (§14.2). What runs on this box is measurement plus raw diff COUNTS.
#
# -------------------------------------------------------------------------------------------------
# STEP 0 — THE LAUNCHER COPY MUST BE EDITED FIRST, OR THIS RUN DIES AT THE SELFTESTS.
# -------------------------------------------------------------------------------------------------
# `lambda_run.sh:93-135` is a HARDCODED scp list and it does NOT contain the two new instruments
# (§12/E2 says so explicitly). Files land FLAT in ~/latent_verify — the scp preserves no `controls/`
# prefix — which is why every invocation below is `python family_..._fmt.py`, never
# `python controls/family_..._fmt.py` (same convention as run_cleangate_topk_27b.sh:84-87).
#
# Per §12's launch discipline and OWED.md E1 (editing lambda_run.sh while a launcher executes it
# corrupts the launcher and its EXIT trap tears down a LIVE box — it cost a whole box), the edit is
# made to a PER-RUN IMMUTABLE COPY and NEVER to lambda_run.sh:
#
#     cp lambda_run.sh .launcher_fmt2b9b.sh          # .launcher_*.sh is gitignored
#
# Then, in `.launcher_fmt2b9b.sh`, the scp list currently reads (lambda_run.sh:119-123, verbatim):
#
#       controls/foldlisten_judge.py controls/family_generate_judge.py controls/verifier_family.py \
#       controls/faithful_rescore.py \
#       controls/family_cave_diagnose.py controls/family_topk_shift.py controls/modelw_candidates.py \
#       controls/verifier_family_ext.py controls/think_probe_identity.py \
#       verifier_family_ext2.json combined_family.json mechanism_family_9bit.json controls/foldlisten_phase2.py \
#
# ADD EXACTLY ONE LINE, immediately after the `family_cave_diagnose.py` / `family_topk_shift.py`
# line (`:121`) so the new instruments sit beside the shipped ones they anchor against:
#
#       controls/family_cave_diagnose.py controls/family_topk_shift.py controls/modelw_candidates.py \
# +++   controls/family_topk_shift_fmt.py controls/family_cave_diagnose_fmt.py \
#       controls/verifier_family_ext.py controls/think_probe_identity.py \
#
# The trailing ` \` is load-bearing: the whole `scp ... ubuntu@$IP:latent_verify/` is one continued
# command and a missing backslash truncates it mid-list.
#
# NOT added: `controls/fmt_matched_join.py` — offline-only, never ships to a box (§14.2, §12).
# NOTE: inserting a line SHIFTS every later line number in the copy by 1, so in
# `.launcher_fmt2b9b.sh` the citations below become :136 (remote_run.sh + RUNNER), :145 (backstop),
# :175/:178 (the LAMBDA_INSTANCE_ID + GIT_COMMIT export), :187 (poll deadline).
#
# A launcher copy missing the two files fails at THIS runner's first action — the selftests —
# before a single model load. That is intended (§12).
#
# -------------------------------------------------------------------------------------------------
# THE CELL LIST, DERIVED. 20 invocations, 4 cells, 5 draws per cell.
# -------------------------------------------------------------------------------------------------
# §1 fixes the six cells as gemma-2-{2b,9b,27b} × {base,-it}; §14.4 splits them across two boxes and
# gives THIS box the four small ones. Per cell, five draws are owed, and each is owed by a named
# clause — nothing here is discretionary:
#
#   1. `family_topk_shift_fmt.py`     tag fmt_ext2_<cell>     §14 row 1 — R-RANK, the NEW instrument.
#                                                             Carries §7b's anchor arm internally
#                                                             (slot=bare × key=space is bit-for-bit
#                                                             the shipped construction).
#   2. `family_cave_diagnose_fmt.py`  tag fmt_ext2_<cell>     §14 row 2 — R-PROB, the NEW instrument.
#   3. `family_topk_shift.py`         tag sbref_ext2_<cell>   §7 REQUIREMENT 2 — the SAME-BOX,
#                                                             SAME-SESSION shipped reference. This is
#                                                             the control `clean_test_owed`
#                                                             (out/b1_fold_identity_gate_27b.json:145)
#                                                             says should have been designed and was
#                                                             not. For ranks it is the ONLY
#                                                             exact-gated comparison that exists (A4),
#                                                             and at 2b/9b it is the FIRST same-box
#                                                             rank repeat ever taken (A5: zero
#                                                             repeated family_topk_shift artifacts
#                                                             exist at any 2b or 9b cell).
#   4. `family_cave_diagnose.py`      tag sbref_ext2_<cell>   §7 requirement 2, lp side.
#   5. `family_cave_diagnose.py`      tag sbref2_ext2_<cell>  A18 — the SECOND shipped draw, at EVERY
#                                                             cell. LOAD-BEARING, not optional: it is
#                                                             the only source of the per-cell
#                                                             within-box run-to-run flip count, and
#                                                             §9.5 branch 1
#                                                             (KEY_MATERIALITY_UNEVALUABLE_NO_NOISE_
#                                                             CONTEXT) fires without it — NEITHER a
#                                                             material NOR an immaterial call. 8
#                                                             forwards/item buys the rule at 4 cells.
#
# ORDER: 2bbase, 2bit, 9bbase, 9bit — BASE CELL FIRST WITHIN EACH SCALE, same box, same session
# (§1 as amended by A8). A8 relaxed "same process" to "same box, same session" because same-process
# pairing is structurally impossible: one `--name` and one `is_chat` per invocation, and the model is
# freed inside the measurement call (controls/family_cave_diagnose.py:260-262). So each invocation
# below is ONE cell that loads and frees its own model; cells are NEVER batched into one process.
# §1: "a run producing `-it` cells without their same-box base twins is not a run under this
# registration and yields no §9 verdict."
#
# -------------------------------------------------------------------------------------------------
# EVERY OUTCOME, STATED BEFORE ANY DATA EXISTS — and none of them is decided here.
# -------------------------------------------------------------------------------------------------
# §14.2 is explicit: verdict emission is OFFLINE-ONLY and single-sourced. This runner prints raw diff
# COUNTS for the pairs whose BOTH sides it produces, and no verdict at all. The frozen outcome tables
# it feeds, so that what this box is buying is on the record before it runs:
#
#   §9.1 slot comparability   SLOT_DEGENERATE (onset == 0 at either arm; SUPPRESSES) →
#                             SLOT_UNMATCHED (|Δonset| > 0.10; emitted but DOWNGRADED) → SLOT_MATCHED.
#                             ONSET_FLOOR is WITHDRAWN (A15) — the level is reported, never gated.
#   §9.2 rank resolution      KEY_UNLOCATABLE (any item's prefix assertion fails; cell VOIDED,
#                             denominators stay 82) → RANK_RESOLUTION_INSUFFICIENT (the two arms'
#                             [median_rank ± median_rank_plateau] intervals overlap; SUPPRESSES, and
#                             is NOT evidence the ranks are equal) → RANK_RESOLVED.
#                             KEY_LIVE_FRAC is WITHDRAWN (A16).
#   §9.3 THE PRIMARY READOUT  entity W*, slot elicit, canonical key, L_new = log10(ratio of medians),
#                             quoted as an ordered (2b, 9b, 27b) TRIPLE OR NOT AT ALL (§8.2/A17).
#                             SLOT_UNINTERPRETABLE → GAP_STATISTIC_DEPENDENT (L_new and Lp in
#                             different bands) → GAP_CLOSED (<= 0.5) → GAP_SURVIVES (>= 2.0) →
#                             GAP_MOSTLY_CLOSED (<= L_old - 1.0) → GAP_INDETERMINATE.
#                             GAP_SURVIVES retracts this registration's own motivation and is
#                             reachable at every scale. BAND_EMPTY_BY_CONSTRUCTION is emitted for
#                             entity C at 9b (band ~26bp wide) and 27b (empty) — an honest arithmetic
#                             consequence of A7's pre-committed L_old, not post-hoc narrowing.
#   §9.5 key materiality      KEY_MATERIALITY_UNEVALUABLE_NO_NOISE_CONTEXT (draw 5 missing/failed) →
#                             KEY_EFFECT_BELOW_NOISE (key flips <= draw1-vs-draw2 flips) →
#                             KEY_MATERIAL_TO_RC (>= MIN_FAITHFUL = 8 flips, or a category change) →
#                             KEY_IMMATERIAL_TO_RC. Branch 1 is why draw 5 exists at all four cells.
#   §9.6 the anchor           ANCHOR_REPRODUCES / ANCHOR_DIFFERS / ANCHOR_UNEVALUABLE, per cell, per
#                             instrument, per reference side. Ranks vs the SAME-BOX reference: EXACT
#                             integer equality on all 82, the real gate. lp at 2b/9b: within 1e-6.
#                             ANCHOR_DIFFERS on ranks against the same-box reference retires the one
#                             numerically stable lineage the repo has (§7.2) and suppresses §9.3 for
#                             that cell. At 2b/9b it is ALSO ambiguous between code, box and the
#                             absence of any prior stability evidence (A5) — conservative either way.
#
# Every 9b/2b number this box produces is quotable only with the §13 stamp; a number without a
# complete stamp is not quotable (registration #12).
#
# -------------------------------------------------------------------------------------------------
# BUDGET — the cap is set from arithmetic, not hope.
# -------------------------------------------------------------------------------------------------
# Measured on this project: one SHIPPED instrument invocation costs T = 66 s at 2b, 132 s at 9b
# (267 s at 27b, box B), each paying a FULL model load because each cell is a fresh process. §14.4's
# forward budget: the two new instruments together are 16 forwards/item against the shipped pair's 11
# (topk 3 + diagnose 8), i.e. 1.45x, so the new pair costs ~2.9 T and the five draws cost ~5.9 T:
#
#   2b-base  5.9 x  66 s = 389 s     9b-base  5.9 x 132 s =  779 s
#   2b-it    5.9 x  66 s = 389 s     9b-it    5.9 x 132 s =  779 s
#   compute subtotal .............................. 2336 s = 39 min
#   + remote_run.sh venv build (torch cu124 + TL, first call) ~10 min
#   + HF weight pull, 4 models (2b, 2b-it, 9b, 9b-it) ~47 GB ...... ~20 min
#   + selftests, provenance, on-box diff counts ................... ~2 min
#   EXPECTED WALL CLOCK ~71 min.
#
# CAP: REMOTE_TIMEOUT=7200 (120 min) — 69% headroom over the estimate, and NOT larger, because
# lambda_run.sh nests three caps off it and the outermost one bills:
#   on-box   `timeout $REMOTE_TIMEOUT`            (:177)  7200 s = 120 min
#   local    poll deadline REMOTE_TIMEOUT + 900   (:186)  8100 s = 135 min
#   box      self-destruct + REATTACH_GRACE(1800) (:144)  9000 s = 150 min  <- the billing bound
# gpu_1x_a100_sxm4 (>=40 GB; 9b bf16 forward-only fits in 18.5 GB of weights) at ~$1.99/hr:
#   expected bill ~81 min incl. launch/scp/fetch = $2.69; worst case a total hang = 2.5 h = $4.98.
# Against $248.29 of remaining headroom on the $950 cap that is 1.1% expected, 2.0% worst case.
#
# LAUNCH:
#   cp lambda_run.sh .launcher_fmt2b9b.sh && <make the one scp edit above to the COPY>
#   REMOTE_TIMEOUT=7200 bash .launcher_fmt2b9b.sh gpu_1x_a100_sxm4 <region> \
#       run_fmt_matched_2b9b.sh results_fmt_2b9b
#
# -------------------------------------------------------------------------------------------------
# CONVENTIONS, and the one deliberate deviation.
# -------------------------------------------------------------------------------------------------
# `set -uo pipefail`, NOT `set -euo pipefail`, matching run_cleangate_topk_27b.sh:38 and
# run_r1_dist_27b.sh:24. `-e` is omitted DELIBERATELY and the omission is the whole fail-soft design:
# with `-e` the first failing cell would abort the process before its exit code could be recorded and
# would abandon the remaining 19 cells, which is exactly what per-cell exit capture exists to
# prevent. Every cell's rc is captured explicitly by `cell()` and tabulated at the end.
#
# The terminal marker is `ALLDONE_FMT_MATCHED_2B9B` on stdout (-> out/run_detached.log), matching
# run_cleangate_topk_27b.sh:126 / run_r1_dist_27b.sh:100. The `RUN_DONE` file that the launcher polls
# at :186-210 is written by the LAUNCHER's own wrapper (`echo \$? > RUN_DONE`, :177), not by this
# script — so RUN_DONE=0 means "the runner reached its end", NOT "every cell passed". The per-cell
# `exit=` lines, out/fmt_cellstatus_2b9b.tsv and out/fmt_matched_2b9b_summary.json are the truth.
# The summary file is named `*summary*.json` on purpose: the launcher's priority fetch (:219-226)
# grabs and JSON-validates `out/*summary*.json` FIRST, before the multi-MB dumps.
# =================================================================================================
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# §10.1 defines "same box" as, among others, cuda_visible_devices equal and equal to "0" and
# device_index equal and equal to 0. Both instruments read CUDA_VISIBLE_DEVICES straight from
# os.environ, so leaving it unset stamps null and makes every same-box comparison UNVERIFIABLE.
export CUDA_VISIBLE_DEVICES=0
# The launcher exports these two at lambda_run.sh:174,177 and the box has no git checkout, so
# neither is obtainable on-box. Re-exported explicitly so every `python` child inherits them.
export LAMBDA_INSTANCE_ID="${LAMBDA_INSTANCE_ID:-}"
export GIT_COMMIT="${GIT_COMMIT:-}"
mkdir -p out
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} LAMBDA_INSTANCE_ID=${LAMBDA_INSTANCE_ID:-<EMPTY>} GIT_COMMIT=${GIT_COMMIT:-<EMPTY>}"

# --- §12 / M2 PRE-FLIGHT. A null is a failure, not a note. -----------------------------------------
# Both new instruments call validate_provenance(build_provenance(...)) as their FIRST action and RAISE
# ProvenanceIncomplete before any model is loaded if lambda_instance_id or started_utc is null/empty
# (controls/family_topk_shift_fmt.py:1101, PROVENANCE_LOAD_BEARING at :225). Checking it once here
# turns 20 identical aborts into one and costs ~3 minutes of box time instead of a whole box.
if [ -z "${LAMBDA_INSTANCE_ID}" ]; then
  echo "ABORT_PROVENANCE_INCOMPLETE: LAMBDA_INSTANCE_ID is empty."
  echo "  lambda_run.sh:174,177 exports it; an empty value means this runner was invoked outside the"
  echo "  launcher. Every artifact would stamp lambda_instance_id: null, which is precisely the defect"
  echo "  results_r1_dist_27b/out/provenance_r1_27b.json:10 has and which makes §10.1's same-box test"
  echo "  return SAME_BOX_UNVERIFIABLE and suppress every verdict that depends on it. Refusing to run."
  echo "ALLDONE_FMT_MATCHED_2B9B (aborted before any model load)"
  exit 3
fi
# git_commit must be PRESENT but is not in PROVENANCE_LOAD_BEARING, so an empty value is a loud
# warning rather than an abort — the registration's own strictness, not more.
[ -z "${GIT_COMMIT}" ] && echo "[warn] GIT_COMMIT empty: artifacts will stamp git_commit null (§12 wants it non-null; not load-bearing, not fatal)"

# --- run-level provenance stamp (§12 + REGISTRATION_provenance.md §1 + §10.1's two extra fields) ---
# Written FIRST so it exists even if a cell dies: the whole point is that a future agent can attribute
# a number to a machine. finished_utc is patched in at the end.
python - <<'PY' > out/provenance_fmt_2b9b.json 2>/dev/null || echo '{"provenance":"FAILED"}' > out/provenance_fmt_2b9b.json
import json, os, subprocess, sys, datetime
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
def ver(m):
    try:
        from importlib.metadata import version; return version(m)
    except Exception: return None
p = {"run": "fmt_matched_2b9b",
     "registration": "docs/drafts/REGISTRATION_format_matched_readout.md (frozen, pre-data; §12 + §10.1)",
     "box": "A of 2 -- the 2b and 9b cells (§14.4)",
     "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
     "finished_utc": None,
     "python": sys.version.split()[0],
     "lambda_instance_id": os.environ.get("LAMBDA_INSTANCE_ID"),
     "git_commit": os.environ.get("GIT_COMMIT"),
     "gpu_name": sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
     "gpu_count": sh("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"),
     "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
     "device_index": None,
     "dtype": "bfloat16", "torch": ver("torch"), "transformers": ver("transformers"),
     # transformer_lens has no __version__ (OWED.md A2) -> importlib.metadata only
     "transformer_lens": ver("transformer_lens")}
try:
    import torch
    p["cuda_runtime"] = torch.version.cuda
    p["device_index"] = (int(torch.cuda.current_device()) if torch.cuda.is_available() else None)
except Exception:
    p["cuda_runtime"] = None
print(json.dumps(p, indent=2))
PY
cat out/provenance_fmt_2b9b.json

# --- selftests, model-free, FIRST. A launcher copy missing the two new files dies here (§12). -------
echo "=== selftests (model-free, CPU; a failure here stops the run before a box-hour is spent) ==="
python family_topk_shift_fmt.py    --selftest || { echo "SELFTEST_FAIL_TOPK_FMT     (missing from the launcher copy's scp list?)"; exit 1; }
python family_cave_diagnose_fmt.py --selftest || { echo "SELFTEST_FAIL_DIAGNOSE_FMT (missing from the launcher copy's scp list?)"; exit 1; }
python family_topk_shift.py        --selftest || { echo "SELFTEST_FAIL_TOPK_SHIPPED";     exit 1; }
python family_cave_diagnose.py     --selftest || { echo "SELFTEST_FAIL_DIAGNOSE_SHIPPED"; exit 1; }

# --- per-cell driver: ONE invocation = ONE cell, own model load, own log, own captured exit code ----
CELLSTATUS=out/fmt_cellstatus_2b9b.tsv
: > "$CELLSTATUS"
cell () {
  local lab="$1"; shift
  local rc=0
  echo "=== CELL $lab :: $* ==="
  "$@" > "out/$lab.log" 2>&1 || rc=$?
  printf '%s\t%s\t%s\n' "$lab" "$rc" "$*" >> "$CELLSTATUS"
  echo "exit=$rc  ($lab)"
  tail -12 "out/$lab.log" 2>/dev/null || true
  echo "--- end $lab ---"
}

FAM=verifier_family_ext2.json     # §1: 82 items, UNFILTERED (no select_items)

# =================================================================================================
# 2b — BASE CELL FIRST (§1 / A8), then -it. Five draws per cell, per §14.4.
# =================================================================================================
B=google/gemma-2-2b
cell topkfmt_2bbase  python family_topk_shift_fmt.py    --family $FAM --name $B --tag fmt_ext2_2bbase    --device cuda
cell diagfmt_2bbase  python family_cave_diagnose_fmt.py --family $FAM --name $B --tag fmt_ext2_2bbase    --device cuda
cell topkref_2bbase  python family_topk_shift.py        --family $FAM --name $B --tag sbref_ext2_2bbase  --device cuda
cell diagref_2bbase  python family_cave_diagnose.py     --family $FAM --name $B --tag sbref_ext2_2bbase  --device cuda
cell diagref2_2bbase python family_cave_diagnose.py     --family $FAM --name $B --tag sbref2_ext2_2bbase --device cuda   # A18 noise context

I=google/gemma-2-2b-it
cell topkfmt_2bit    python family_topk_shift_fmt.py    --family $FAM --name $I --tag fmt_ext2_2bit    --device cuda --chat
cell diagfmt_2bit    python family_cave_diagnose_fmt.py --family $FAM --name $I --tag fmt_ext2_2bit    --device cuda --chat
cell topkref_2bit    python family_topk_shift.py        --family $FAM --name $I --tag sbref_ext2_2bit  --device cuda --chat
cell diagref_2bit    python family_cave_diagnose.py     --family $FAM --name $I --tag sbref_ext2_2bit  --device cuda --chat
cell diagref2_2bit   python family_cave_diagnose.py     --family $FAM --name $I --tag sbref2_ext2_2bit --device cuda --chat  # A18

# =================================================================================================
# 9b — BASE CELL FIRST, then -it.
# =================================================================================================
B=google/gemma-2-9b
cell topkfmt_9bbase  python family_topk_shift_fmt.py    --family $FAM --name $B --tag fmt_ext2_9bbase    --device cuda
cell diagfmt_9bbase  python family_cave_diagnose_fmt.py --family $FAM --name $B --tag fmt_ext2_9bbase    --device cuda
cell topkref_9bbase  python family_topk_shift.py        --family $FAM --name $B --tag sbref_ext2_9bbase  --device cuda
cell diagref_9bbase  python family_cave_diagnose.py     --family $FAM --name $B --tag sbref_ext2_9bbase  --device cuda
cell diagref2_9bbase python family_cave_diagnose.py     --family $FAM --name $B --tag sbref2_ext2_9bbase --device cuda   # A18

I=google/gemma-2-9b-it
cell topkfmt_9bit    python family_topk_shift_fmt.py    --family $FAM --name $I --tag fmt_ext2_9bit    --device cuda --chat
cell diagfmt_9bit    python family_cave_diagnose_fmt.py --family $FAM --name $I --tag fmt_ext2_9bit    --device cuda --chat
cell topkref_9bit    python family_topk_shift.py        --family $FAM --name $I --tag sbref_ext2_9bit  --device cuda --chat
cell diagref_9bit    python family_cave_diagnose.py     --family $FAM --name $I --tag sbref_ext2_9bit  --device cuda --chat
cell diagref2_9bit   python family_cave_diagnose.py     --family $FAM --name $I --tag sbref2_ext2_9bit --device cuda --chat  # A18

# =================================================================================================
# §14.2 — RAW DIFF COUNTS, ON BOX, FOR THE PAIRS WHOSE *BOTH* SIDES THIS BOX PRODUCED. NO VERDICTS.
# Both sides are local, so this needs no committed reference artifact and costs nothing; it preserves
# a diagnostic trail through a failed fetch. Comparisons against COMMITTED artifacts, cluster
# fingerprinting, and every §9/§10 verdict are OFFLINE-ONLY (controls/fmt_matched_join.py), because
# verdicts must have exactly one source (A10).
# =================================================================================================
echo "=== §14.2 on-box raw diff counts (NO VERDICT IS EMITTED ON BOX) ==="
python - <<'PY'
import json, datetime, os, traceback

BANNER = ("RAW DIFF COUNTS ONLY. NO VERDICT IS EMITTED ON BOX (registration section 14.2 / A10). "
          "Every section 9 and section 10 verdict is offline-only, from controls/fmt_matched_join.py, "
          "which also does every comparison against a committed artifact.")
CELLS = ("2bbase", "2bit", "9bbase", "9bit")

# The fmt instruments' tag pattern (section 14.1) is `fmt_ext2_<cell>` and their writer prepends the
# instrument name, so the artifact really is `family_topk_shift_fmt_fmt_ext2_2bbase.json`.
def p_topk_fmt(c):  return f"out/family_topk_shift_fmt_fmt_ext2_{c}.json"
def p_diag_fmt(c):  return f"out/family_cave_diagnose_fmt_fmt_ext2_{c}.json"
def p_topk_ref(c):  return f"out/family_topk_shift_sbref_ext2_{c}.json"
def p_diag_ref(c):  return f"out/family_cave_diagnose_sbref_ext2_{c}.json"
def p_diag_ref2(c): return f"out/family_cave_diagnose_sbref2_ext2_{c}.json"

def load(path):
    d = json.load(open(path))
    return d.get("result") or d

def items(r, slot=None):
    xs = r["items"]
    return xs if slot is None else [x for x in xs if x.get("slot") == slot]

ABSENT = "<ABSENT>"

def counts(A, B, fields):
    """Per field, the number of items on which the two sides differ. Exact equality on the persisted
    6dp values -- the same basis section 10.2 fixes ("identical after round(x, 6)")."""
    out, first = {}, {}
    for f in fields:
        n = 0
        for i, (x, y) in enumerate(zip(A, B)):
            if x.get(f, ABSENT) != y.get(f, ABSENT):
                n += 1
                first.setdefault(f, {"item": i, "a": x.get(f, ABSENT), "b": y.get(f, ABSENT)})
        if n:
            out[f] = n
    return out, first

# ---- projections, so the two sides of a fmt-vs-shipped pair are field-name comparable -------------
# section 7b: the fmt topk instrument carries the anchor arm under the SHIPPED names in
# items[slot=bare]["anchor_shipped"] (controls/family_topk_shift_fmt.py:725-735), at the SHIPPED 6dp
# rounding, precisely so this diff reads exactly the shipped quantities.
TOPK_FIELDS = ("cid", "aid", "first_token_collision", "p_c_bare", "rank_c_bare",
               "p_w_bare", "rank_w_bare", "topk_bare_ids", "topk_bare_p", "topk_bare_strs")
def _topk(a, q):
    return {"q": q, "cid": a["cid"], "aid": a["aid"],
            "first_token_collision": a["first_token_collision"],
            "p_c_bare": a["p_c_bare"], "rank_c_bare": a["rank_c_bare"],
            "p_w_bare": a["p_w_bare"], "rank_w_bare": a["rank_w_bare"],
            "topk_bare_ids": [t["tok_id"] for t in a["topk_bare"]],
            "topk_bare_p": [t["p"] for t in a["topk_bare"]],
            "topk_bare_strs": [t["tok_str"] for t in a["topk_bare"]]}
def proj_topk_fmt(x):     return _topk(x["anchor_shipped"], x["q"])
def proj_topk_shipped(x): return _topk(x, x["q"])

# The fmt diagnose instrument emits one column per measured key, field names suffixed by the column
# name (controls/family_cave_diagnose_fmt.py:677-684). The `space` column IS the section 7b anchor:
# bit-for-bit the shipped continuation ids. These are the 18 shipped per-item value fields (the 23
# pre-existing fields less q/correct/Wstar/tier/category).
DIAG_FIELDS = ("M0", "abs_M0", "lpC_single", "lpW_single", "lpC_neutral", "lpW_neutral",
               "lpC_counter", "lpW_counter", "Mc_neutral", "Mc_counter", "RC_effect",
               "P_w_neutral", "P_w_counter", "RA_effect",
               "headroom_pass", "faithful_RC", "faithful_RA", "first_token_collision")
def proj_diag_fmt(x):
    d = {f: x.get(f + "_space", ABSENT) for f in DIAG_FIELDS}; d["q"] = x["q"]; return d
def proj_diag_shipped(x):
    d = {f: x.get(f, ABSENT) for f in DIAG_FIELDS}; d["q"] = x["q"]; return d

# ---- the pair table ------------------------------------------------------------------------------
# kind "same":   both sides are the SAME instrument, so every persisted field is comparable and the
#                basis is all pre-existing fields of items[0] (the run_r1_dist_27b.sh:79-81 form).
# kind "topk"/"diag": fmt vs shipped, via the projections above.
PAIRS = []
for c in CELLS:
    PAIRS.append((f"{c} :: A18 NOISE CONTEXT -- shipped diagnose sbref_ vs sbref2_ (same code, same box)",
                  "same", p_diag_ref(c), p_diag_ref2(c), None))
    PAIRS.append((f"{c} :: section 7b anchor, RANKS -- topk_fmt(anchor_shipped) vs shipped topk sbref_",
                  "topk", p_topk_fmt(c), p_topk_ref(c), "bare"))
    PAIRS.append((f"{c} :: section 7b anchor, LP -- diagnose_fmt(space column) vs shipped diagnose sbref_",
                  "diag", p_diag_fmt(c), p_diag_ref(c), None))

print("[section 14.2] " + BANNER)
report = {}
for label, kind, pa, pb, slot in PAIRS:
    entry = {"a": pa, "b": pb, "kind": kind}
    try:
        ra, rb = load(pa), load(pb)
        if kind == "same":
            A, B = items(ra), items(rb)
            fields = [k for k in A[0]] if A else []
            pa_, pb_ = A, B
        elif kind == "topk":
            A = [proj_topk_fmt(x) for x in items(ra, slot)]
            B = [proj_topk_shipped(x) for x in items(rb)]
            fields, pa_, pb_ = list(TOPK_FIELDS), A, B
        else:
            A = [proj_diag_fmt(x) for x in items(ra)]
            B = [proj_diag_shipped(x) for x in items(rb)]
            fields, pa_, pb_ = list(DIAG_FIELDS), A, B
        entry["n_items"] = {"a": len(pa_), "b": len(pb_)}
        if len(pa_) != len(pb_) or [x["q"] for x in pa_] != [x["q"] for x in pb_]:
            # section 10.2's item-order rule: no reordering and no intersection. Fails loudly.
            entry["status"] = "ITEM_ORDER_OR_LENGTH_DIFFERS -> not comparable on box"
            print(f"[diff] {label}\n       {entry['status']}")
        else:
            nz, firsts = counts(pa_, pb_, fields)
            entry.update({"status": "OK", "n_fields_compared": len(fields),
                          "fields_with_any_difference": nz, "n_fields_differing": len(nz),
                          "first_divergent_cell_per_field": firsts})
            print(f"[diff] {label}\n       n={len(pa_)} fields={len(fields)} "
                  f"differing={nz if nz else '{} -> 0 of ' + str(len(fields)) + ' fields differ'}")
    except Exception as e:
        entry["status"] = f"UNAVAILABLE: {type(e).__name__}: {e}"
        print(f"[diff] {label}\n       {entry['status']}")
        traceback.print_exc()
    report[label] = entry

# ---- the small, decision-bearing file the launcher fetches FIRST (lambda_run.sh:219-226) ----------
cells = []
try:
    for line in open("out/fmt_cellstatus_2b9b.tsv"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            cells.append({"label": parts[0], "exit": int(parts[1]), "cmd": parts[2]})
except Exception as e:
    cells = [{"label": "<unreadable>", "exit": -1, "cmd": str(e)}]
prov = None
try:
    prov = json.load(open("out/provenance_fmt_2b9b.json"))
except Exception:
    pass
summary = {
    "run": "fmt_matched_2b9b", "box": "A of 2 (2b + 9b cells)",
    "registration": "docs/drafts/REGISTRATION_format_matched_readout.md (frozen, pre-data, A1-A20)",
    "no_verdicts_on_box": BANNER,
    "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "provenance": prov,
    "cells": cells,
    "n_cells": len(cells),
    "n_cells_nonzero_exit": sum(1 for c in cells if c["exit"] != 0),
    "onbox_raw_diff_counts": report,
}
json.dump(summary, open("out/fmt_matched_2b9b_summary.json", "w"), indent=2, default=str)
print("[section 14.2] wrote out/fmt_matched_2b9b_summary.json "
      f"({len(cells)} cells, {summary['n_cells_nonzero_exit']} with a non-zero exit)")
PY

# --- close the run-level provenance -----------------------------------------------------------------
python - <<'PY'
import json, datetime
try:
    p = json.load(open("out/provenance_fmt_2b9b.json"))
    p["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(p, open("out/provenance_fmt_2b9b.json", "w"), indent=2)
    print("[provenance] finished_utc", p["finished_utc"])
except Exception as e:
    print("provenance close failed:", e)
PY

echo "=== per-cell exit codes (a non-zero cell does NOT abandon the rest; RUN_DONE=0 != all passed) ==="
cat "$CELLSTATUS" 2>/dev/null || echo "<no cell status recorded>"
echo "=== artifacts ==="
ls -la out/family_topk_shift_fmt_*.json out/family_cave_diagnose_fmt_*.json \
       out/family_topk_shift_sbref_*.json out/family_cave_diagnose_sbref*_*.json 2>/dev/null
echo "ALLDONE_FMT_MATCHED_2B9B"
