#!/usr/bin/env bash
# =================================================================================================
# BOX B of TWO — the FORMAT-MATCHED base-vs-`-it` readout at the 27b cells, PLUS §10's 27b
# STABILITY CONTROL. `gpu_1x_h100_sxm5` (§14.4; 27b bf16 needs an 80 GB card).
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
#     cp lambda_run.sh .launcher_fmt27b.sh           # .launcher_*.sh is gitignored
#
# Then, in `.launcher_fmt27b.sh`, the scp list currently reads (lambda_run.sh:119-123, verbatim):
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
# ALREADY IN THE LIST, and required by this box: `controls/family_cave_diagnose_arms.py` (`:97`) for
# §10's B1 draw, and `verifier_family_ext2.json` (`:123`).
# NOT added: `controls/fmt_matched_join.py` — offline-only, never ships to a box (§14.2, §12).
# NOTE: inserting a line SHIFTS every later line number in the copy by 1, so in
# `.launcher_fmt27b.sh` the citations below become :136 (remote_run.sh + RUNNER), :145 (backstop),
# :175/:178 (the LAMBDA_INSTANCE_ID + GIT_COMMIT export), :187 (poll deadline).
#
# A launcher copy missing the two files fails at THIS runner's first action — the selftests —
# before a single model load. That is intended (§12).
#
# -------------------------------------------------------------------------------------------------
# THE CELL LIST, DERIVED. 11 invocations. §14.4's ordering, followed literally.
# -------------------------------------------------------------------------------------------------
#  #  label            instrument                       tag                  owed by
#  1  topkfmt_27bbase  family_topk_shift_fmt.py         fmt_ext2_27bbase     §14 row 1 (R-RANK, NEW)
#  2  diagfmt_27bbase  family_cave_diagnose_fmt.py      fmt_ext2_27bbase     §14 row 2 (R-PROB, NEW)
#  3  topkfmt_27bit    family_topk_shift_fmt.py --chat  fmt_ext2_27bit       §14 row 1
#  4  diagfmt_27bit    family_cave_diagnose_fmt.py      fmt_ext2_27bit       §14 row 2
#  5  topkref_27bbase  family_topk_shift.py             sbref_ext2_27bbase   §7 req. 2 + A11
#  6  topkref_27bit    family_topk_shift.py --chat      sbref_ext2_27bit     §7 req. 2 + A11
#  7  diagref_27bit    family_cave_diagnose.py --chat   sbref_ext2_27bit     §7 req. 2 + A11
#  8  diagref2_27bit   family_cave_diagnose.py --chat   sbref2_ext2_27bit    A18 (noise context)
#  9  stab_shipA       family_cave_diagnose.py          stab27b_shipA        §10 A1 == 27b-base sbref_
# 10  stab_shipB       family_cave_diagnose.py          stab27b_shipB        §10 A2 == 27b-base sbref2_
# 11  stab_arms        family_cave_diagnose_arms.py     stab27b_arms         §10 B1 (--arm fold)
#
# WHY EACH GROUP EXISTS:
#
# * 1–4, the two NEW instruments at both cells (§14 rows 1–2). BASE CELL FIRST, then -it: §1 as
#   amended by A8. A8 relaxed "same process" to "same box, same session" because same-process pairing
#   is structurally impossible — one `--name` and one `is_chat` per invocation, and the model is freed
#   inside the measurement call (controls/family_cave_diagnose.py:260-262). Every invocation below is
#   ONE cell that loads and frees its own model; cells are NEVER batched into one process. §1: "a run
#   producing `-it` cells without their same-box base twins is not a run under this registration and
#   yields no §9 verdict."
#
# * 5–7, the THREE EXTRA 27b SHIPPED DRAWS AUTHORISED BY A11. §7 requirement 2 is a same-box,
#   same-session shipped reference — the control `clean_test_owed`
#   (out/b1_fold_identity_gate_27b.json:145) says should have been designed and was not. Before A11
#   only ONE of the four 27b instrument×cell combinations had one; A11 authorises the other three so
#   the requirement EXISTS at all four: topk at 27b-base, topk at 27b-it, diagnose at 27b-it. The
#   fourth, diagnose at 27b-base, is supplied by §10's A1/A2 (§14.1 states that identity explicitly:
#   "shipA/shipB are 27b-base's sbref_/sbref2_"), which is why no separate sbref_ext2_27bbase
#   diagnose draw appears — running one would pay 267 s twice for the same measurement.
#   This same-box reference is the ONLY exact-gated rank comparison that exists at 27b (A4): every
#   committed family_topk_shift* artifact is H100 PCIe / driver 570.148.08, no such artifact exists on
#   any other card, and this box must be h100_sxm5 because PCIe has zero capacity — so an exact gate
#   against the COMMITTED column would test hardware, not code, and §7.2 refuses it
#   (DISCLOSED_NOT_GATED, no 27b-vs-committed reproduction verdict at all).
#
# * 8, A18's SECOND SHIPPED DRAW at 27b-it. LOAD-BEARING, not optional. It is the only source of the
#   per-cell within-box run-to-run flip count, and without it §9.5 falls to branch 1,
#   KEY_MATERIALITY_UNEVALUABLE_NO_NOISE_CONTEXT — NEITHER a material nor an immaterial call, because
#   "material" licenses superseding committed numbers and "immaterial" licenses retracting §4.2, and
#   neither may be made by default. 8 forwards/item buys the rule. At 27b-base the same context comes
#   free from §10's A1/A2 pair.
#
# * 9–11, §10, THE 27b STABILITY CONTROL — the missing within-box repeat of the SHIPPED instrument on
#   the same box as the twin (§2.4). Compared: A1 vs A2 (within-box, same code — the rider's design,
#   and A18's 27b-base noise context), A1 vs B1 and A2 vs B1 (the cleangate comparison, now with TWO
#   reference draws). Basis: all 23 pre-existing fields × 82 items, identical after round(x, 6).
#   §10 states SHIPPED_SELF_DIFFERS is the branch this is most likely to land in, not a remote
#   possibility: an independent reader found results_r1_dist_27b's SHIPPED draw identical to
#   results_cleangate_27b's ARMS draw at 0 of 1148 cells while the cleangate SHIPPED draw differs from
#   both at 1079 of 1148, so the anomalous draw is the clean test's own reference side.
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
#   §9.2 rank resolution      KEY_UNLOCATABLE → RANK_RESOLUTION_INSUFFICIENT (the two arms'
#                             [median_rank ± median_rank_plateau] intervals overlap; SUPPRESSES, and
#                             is NOT evidence the ranks are equal) → RANK_RESOLVED. This is the gate
#                             the 27b bf16 tie structure bears on directly: §7.2 measured 498 of 2214
#                             adjacent top-10 gaps EXACTLY TIED at 27b-base, none in (0, 0.05).
#   §9.3 THE PRIMARY READOUT  entity W*, slot elicit, canonical key, L_new = log10(ratio of medians),
#                             quoted as an ordered (2b, 9b, 27b) TRIPLE OR NOT AT ALL (§8.2/A17) —
#                             so this box's 27b cell is one third of the only permitted quotation.
#                             SLOT_UNINTERPRETABLE → GAP_STATISTIC_DEPENDENT → GAP_CLOSED (<= 0.5) →
#                             GAP_SURVIVES (>= 2.0) → GAP_MOSTLY_CLOSED (<= L_old - 1.0) →
#                             GAP_INDETERMINATE. GAP_SURVIVES retracts this registration's own
#                             motivation and is reachable at every scale.
#                             BAND_EMPTY_BY_CONSTRUCTION is emitted for entity C at 27b (A7's
#                             pre-committed L_old = 1.398 makes step 5's band arithmetically empty).
#   §9.5 key materiality      KEY_MATERIALITY_UNEVALUABLE_NO_NOISE_CONTEXT (draw 8 / the A1-A2 pair
#                             missing or failed, or §10 returned STAB27B_UNEVALUABLE /
#                             SAME_BOX_UNVERIFIABLE) → KEY_EFFECT_BELOW_NOISE → KEY_MATERIAL_TO_RC
#                             (>= MIN_FAITHFUL = 8 flips, or a category change) → KEY_IMMATERIAL_TO_RC.
#   §9.6 the anchor           At 27b, ranks vs the SAME-BOX reference: EXACT integer equality on all
#                             82, the real and only gate (A4). lp vs the same-box reference at 27b:
#                             DISCLOSED_NOT_GATED. Everything vs the COMMITTED column at 27b:
#                             DISCLOSED_NOT_GATED, NO verdict emitted at all (§7.2/§7.3).
#                             ANCHOR_DIFFERS on ranks against the same-box reference retires the one
#                             numerically stable lineage the repo has and suppresses §9.3 at 27b.
#   §10.3 stability           STAB27B_UNEVALUABLE (any draw missing/OOM/capped, item order failing
#                             §10.2, or SAME_BOX_UNVERIFIABLE — not a pass, and it ALSO triggers §9.5
#                             branch 1 at 27b-base) → SHIPPED_SELF_DIFFERS (A1 != A2 on any field) →
#                             SHIPPED_SELF_IDENTICAL + ARMS_MATCHES_SHIPPED → SHIPPED_SELF_IDENTICAL
#                             + ARMS_DIFFERS. Branches 2 and 3 REOPEN the cleangate verdict and
#                             B1's listen withdrawal; NEITHER REVERSES IT. Nothing in this run
#                             restores a withdrawn number — restoring six cells of listen numbers
#                             needs its own registration stating the restoration rule first (§10.3's
#                             boundary clause, OWED.md §G).
#
# §11's 27b DISCLOSURE is MANDATORY on every printed 27b number from this run — all four parts, or
# the digit is not quotable. Printed at the end of this run and carried in the summary JSON.
#
# -------------------------------------------------------------------------------------------------
# BUDGET — the cap is set from arithmetic, not hope.
# -------------------------------------------------------------------------------------------------
# Measured on this project: one SHIPPED instrument invocation costs T = 267 s at 27b, paying a FULL
# model load because each cell is a fresh process. §14.4's forward budget: the two new instruments
# together are 16 forwards/item against the shipped pair's 11 (topk 3 + diagnose 8), i.e. 1.45x, so
# the new pair costs ~2.9 T per cell; B1 (--arm fold) is 8 forwards/item, i.e. ~1 T like the shipped
# diagnose it re-parameterises.
#
#   cells 1-2   new pair at 27b-base   2.9 x 267 s =  774 s
#   cells 3-4   new pair at 27b-it     2.9 x 267 s =  774 s
#   cells 5-8   four shipped draws     4.0 x 267 s = 1068 s
#   cells 9-11  §10's A1, A2, B1       3.0 x 267 s =  801 s
#   compute subtotal .............................. 3418 s = 57 min
#   + remote_run.sh venv build (torch cu124 + TL, first call) ~10 min
#   + HF weight pull, 2 models (27b, 27b-it) ~109 GB ............. ~35 min
#   + selftests, provenance, on-box diff counts .................. ~3 min
#   EXPECTED WALL CLOCK ~105 min.
#
# CAP: REMOTE_TIMEOUT=10800 (180 min) — 71% headroom over the estimate, and NOT larger, because
# lambda_run.sh nests three caps off it and the outermost one bills:
#   on-box   `timeout $REMOTE_TIMEOUT`            (:177) 10800 s = 180 min
#   local    poll deadline REMOTE_TIMEOUT + 900   (:186) 11700 s = 195 min
#   box      self-destruct + REATTACH_GRACE(1800) (:144) 12600 s = 210 min  <- the billing bound
# gpu_1x_h100_sxm5 at ~$4.29/hr:
#   expected bill ~115 min incl. launch/scp/fetch = $8.23; worst case a total hang = 3.5 h = $15.02.
# Against $248.29 of remaining headroom on the $950 cap that is 3.3% expected, 6.0% worst case; with
# box A the pair is $10.92 expected / $20.00 worst case. A cap materially larger than this would let
# a single 27b hang burn a double-digit percentage of the remaining budget for nothing, and a cap
# materially smaller would cut the run mid-list and forfeit the whole box: cells 9-11 are last, so a
# premature cap costs §10 — which is the control this box exists for and which cannot be retaken on
# the cleangate box, because that box is gone forever (§10.3's closing paragraph).
#
# LAUNCH:
#   cp lambda_run.sh .launcher_fmt27b.sh && <make the one scp edit above to the COPY>
#   REMOTE_TIMEOUT=10800 bash .launcher_fmt27b.sh gpu_1x_h100_sxm5 <region> \
#       run_fmt_matched_27b.sh results_fmt_27b
#
# -------------------------------------------------------------------------------------------------
# CONVENTIONS, and the one deliberate deviation.
# -------------------------------------------------------------------------------------------------
# `set -uo pipefail`, NOT `set -euo pipefail`, matching run_cleangate_topk_27b.sh:38 and
# run_r1_dist_27b.sh:24. `-e` is omitted DELIBERATELY and the omission is the whole fail-soft design:
# with `-e` the first failing cell would abort the process before its exit code could be recorded and
# would abandon the remaining cells — at 27b that means losing §10 to a failure in cell 3.
# Every cell's rc is captured explicitly by `cell()` and tabulated at the end.
#
# The terminal marker is `ALLDONE_FMT_MATCHED_27B` on stdout (-> out/run_detached.log), matching
# run_cleangate_topk_27b.sh:126 / run_r1_dist_27b.sh:100. The `RUN_DONE` file that the launcher polls
# at :186-210 is written by the LAUNCHER's own wrapper (`echo \$? > RUN_DONE`, :177), not by this
# script — so RUN_DONE=0 means "the runner reached its end", NOT "every cell passed". The per-cell
# `exit=` lines, out/fmt_cellstatus_27b.tsv and out/fmt_matched_27b_summary.json are the truth.
# The summary file is named `*summary*.json` on purpose: the launcher's priority fetch (:219-226)
# grabs and JSON-validates `out/*summary*.json` FIRST, before the multi-MB dumps.
# =================================================================================================
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# §10.1 defines "same box" as, among others, cuda_visible_devices equal and equal to "0" and
# device_index equal and equal to 0 (precedent run_cleangate_topk_27b.sh:43). Both instruments read
# CUDA_VISIBLE_DEVICES straight from os.environ, so leaving it unset stamps null and makes every
# same-box comparison SAME_BOX_UNVERIFIABLE — which on this box would take §10 with it.
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
# turns 11 identical aborts into one and costs ~3 minutes of box time instead of a whole box.
if [ -z "${LAMBDA_INSTANCE_ID}" ]; then
  echo "ABORT_PROVENANCE_INCOMPLETE: LAMBDA_INSTANCE_ID is empty."
  echo "  lambda_run.sh:174,177 exports it; an empty value means this runner was invoked outside the"
  echo "  launcher. Every artifact would stamp lambda_instance_id: null, which is precisely the defect"
  echo "  results_r1_dist_27b/out/provenance_r1_27b.json:10 has -- the reason the rider that established"
  echo "  WITHIN_BOX_DETERMINISTIC cannot even be joined to a box (§2.4) and the reason §10 exists."
  echo "  With it null, §10.1 returns SAME_BOX_UNVERIFIABLE and §10.3 branch 1 fires. Refusing to run."
  echo "ALLDONE_FMT_MATCHED_27B (aborted before any model load)"
  exit 3
fi
# git_commit must be PRESENT but is not in PROVENANCE_LOAD_BEARING, so an empty value is a loud
# warning rather than an abort — the registration's own strictness, not more.
[ -z "${GIT_COMMIT}" ] && echo "[warn] GIT_COMMIT empty: artifacts will stamp git_commit null (§12 wants it non-null; not load-bearing, not fatal)"

# --- run-level provenance stamp (§12 + REGISTRATION_provenance.md §1 + §10.1's two extra fields) ---
# Written FIRST so it exists even if a cell dies: the whole point is that a future agent can attribute
# a number to a machine. finished_utc is patched in at the end.
python - <<'PY' > out/provenance_fmt_27b.json 2>/dev/null || echo '{"provenance":"FAILED"}' > out/provenance_fmt_27b.json
import json, os, subprocess, sys, datetime
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
def ver(m):
    try:
        from importlib.metadata import version; return version(m)
    except Exception: return None
p = {"run": "fmt_matched_27b",
     "registration": "docs/drafts/REGISTRATION_format_matched_readout.md (frozen, pre-data; §12 + §10.1)",
     "box": "B of 2 -- the 27b cells + §10's stability control (§14.4)",
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
cat out/provenance_fmt_27b.json

# --- selftests, model-free, FIRST. A launcher copy missing the two new files dies here (§12). -------
echo "=== selftests (model-free, CPU; a failure here stops the run before a box-hour is spent) ==="
python family_topk_shift_fmt.py     --selftest || { echo "SELFTEST_FAIL_TOPK_FMT     (missing from the launcher copy's scp list?)"; exit 1; }
python family_cave_diagnose_fmt.py  --selftest || { echo "SELFTEST_FAIL_DIAGNOSE_FMT (missing from the launcher copy's scp list?)"; exit 1; }
python family_topk_shift.py         --selftest || { echo "SELFTEST_FAIL_TOPK_SHIPPED";     exit 1; }
python family_cave_diagnose.py      --selftest || { echo "SELFTEST_FAIL_DIAGNOSE_SHIPPED"; exit 1; }
python family_cave_diagnose_arms.py --selftest || { echo "SELFTEST_FAIL_DIAGNOSE_ARMS";    exit 1; }   # §10's B1

# --- per-cell driver: ONE invocation = ONE cell, own model load, own log, own captured exit code ----
CELLSTATUS=out/fmt_cellstatus_27b.tsv
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
B=google/gemma-2-27b
I=google/gemma-2-27b-it

# =================================================================================================
# 1-4. THE TWO NEW INSTRUMENTS. BASE CELL FIRST (§1 / A8), then -it.
# =================================================================================================
cell topkfmt_27bbase python family_topk_shift_fmt.py    --family $FAM --name $B --tag fmt_ext2_27bbase --device cuda
cell diagfmt_27bbase python family_cave_diagnose_fmt.py --family $FAM --name $B --tag fmt_ext2_27bbase --device cuda
cell topkfmt_27bit   python family_topk_shift_fmt.py    --family $FAM --name $I --tag fmt_ext2_27bit   --device cuda --chat
cell diagfmt_27bit   python family_cave_diagnose_fmt.py --family $FAM --name $I --tag fmt_ext2_27bit   --device cuda --chat

# =================================================================================================
# 5-8. THE SAME-BOX SHIPPED REFERENCES (§7 req. 2). Cells 5-7 are A11's three extra 27b draws; cell 8
#      is A18's second draw at 27b-it. Run UNCHANGED, in the same session (§1, §7, §14 row "shipped").
# =================================================================================================
cell topkref_27bbase python family_topk_shift.py    --family $FAM --name $B --tag sbref_ext2_27bbase  --device cuda            # A11
cell topkref_27bit   python family_topk_shift.py    --family $FAM --name $I --tag sbref_ext2_27bit    --device cuda --chat     # A11
cell diagref_27bit   python family_cave_diagnose.py --family $FAM --name $I --tag sbref_ext2_27bit    --device cuda --chat     # A11
cell diagref2_27bit  python family_cave_diagnose.py --family $FAM --name $I --tag sbref2_ext2_27bit   --device cuda --chat     # A18

# =================================================================================================
# 9-11. §10, THE 27b STABILITY CONTROL. §10.2's three draws, in §10.2's order, at 27b-base.
#       A1/A2 ARE 27b-base's sbref_/sbref2_ pair (§14.1), so they double as §7 requirement 2's
#       diagnose reference at 27b-base AND as A18's noise context there.
# =================================================================================================
cell stab_shipA python family_cave_diagnose.py      --family $FAM --name $B --tag stab27b_shipA --device cuda              # §10 A1
cell stab_shipB python family_cave_diagnose.py      --family $FAM --name $B --tag stab27b_shipB --device cuda              # §10 A2
cell stab_arms  python family_cave_diagnose_arms.py --family $FAM --name $B --tag stab27b_arms  --device cuda --arm fold   # §10 B1

# =================================================================================================
# §14.2 — RAW DIFF COUNTS, ON BOX, FOR THE PAIRS WHOSE *BOTH* SIDES THIS BOX PRODUCED. NO VERDICTS.
# Both sides are local, so this needs no committed reference artifact and costs nothing; it preserves
# a diagnostic trail through a failed fetch. Comparisons against COMMITTED artifacts, §10.2's cluster
# fingerprinting, and every §9/§10 verdict are OFFLINE-ONLY (controls/fmt_matched_join.py), because
# verdicts must have exactly one source (A10).
# =================================================================================================
echo "=== §14.2 on-box raw diff counts (NO VERDICT IS EMITTED ON BOX) ==="
python - <<'PY'
import json, datetime, traceback

BANNER = ("RAW DIFF COUNTS ONLY. NO VERDICT IS EMITTED ON BOX (registration section 14.2 / A10). "
          "Every section 9 and section 10 verdict is offline-only, from controls/fmt_matched_join.py, "
          "which also does every comparison against a committed artifact and section 10.2's cluster "
          "fingerprinting. In particular NO section 10.3 stability verdict is emitted here.")

# Section 14.1's tag patterns. The fmt writers prepend the instrument name to the tag, so the artifact
# really is `family_topk_shift_fmt_fmt_ext2_27bbase.json`.
P = {
    "topk_fmt_base": "out/family_topk_shift_fmt_fmt_ext2_27bbase.json",
    "topk_fmt_it":   "out/family_topk_shift_fmt_fmt_ext2_27bit.json",
    "diag_fmt_base": "out/family_cave_diagnose_fmt_fmt_ext2_27bbase.json",
    "diag_fmt_it":   "out/family_cave_diagnose_fmt_fmt_ext2_27bit.json",
    "topk_ref_base": "out/family_topk_shift_sbref_ext2_27bbase.json",
    "topk_ref_it":   "out/family_topk_shift_sbref_ext2_27bit.json",
    "diag_ref_it":   "out/family_cave_diagnose_sbref_ext2_27bit.json",
    "diag_ref2_it":  "out/family_cave_diagnose_sbref2_ext2_27bit.json",
    # section 10 / section 14.1: shipA and shipB ARE 27b-base's sbref_ and sbref2_.
    "A1":            "out/family_cave_diagnose_stab27b_shipA.json",
    "A2":            "out/family_cave_diagnose_stab27b_shipB.json",
    "B1":            "out/family_cave_diagnose_arms_stab27b_arms.json",
}

def load(path):
    d = json.load(open(path))
    return d.get("result") or d

def items(r, slot=None, arm=None):
    xs = r["items"]
    if slot is not None:
        xs = [x for x in xs if x.get("slot") == slot]
    if arm is not None:
        xs = [x for x in xs if x.get("arm", "fold") == arm]     # the arms sibling tags its rows
    return xs

ABSENT = "<ABSENT>"

def counts(A, B, fields):
    """Per field, the number of items on which the two sides differ, plus the first divergent cell.
    Exact equality on the persisted 6dp values -- the basis section 10.2 fixes ("identical after
    round(x, 6)"; no comparison here is tensor bit-identity, and the diagnose deltas at issue are
    ~5 orders above that floor)."""
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
# Section 7b: the fmt topk instrument carries the anchor arm under the SHIPPED names in
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
# kind "same":      both sides the SAME shipped instrument -> the basis is every pre-existing field of
#                   items[0], the run_r1_dist_27b.sh:79-81 / run_cleangate_topk_27b.sh:112-115 form.
# kind "same_arms": shipped vs the arms re-parameterisation, arm=fold rows only, fields from the
#                   SHIPPED side (the "pre-existing fields" basis of section 10.2).
# kind "topk"/"diag": fmt vs shipped, via the projections above.
PAIRS = [
    ("27b-base :: section 10 A1 vs A2 -- SHIPPED vs ITSELF, within box, same code "
     "(the rider's design; ALSO A18's 27b-base noise context)", "same", P["A1"], P["A2"], {}),
    ("27b-base :: section 10 A1 vs B1 -- shipped vs arms fold (the cleangate comparison, ref draw 1)",
     "same_arms", P["A1"], P["B1"], {}),
    ("27b-base :: section 10 A2 vs B1 -- shipped vs arms fold (the cleangate comparison, ref draw 2)",
     "same_arms", P["A2"], P["B1"], {}),
    ("27b-it   :: A18 NOISE CONTEXT -- shipped diagnose sbref_ vs sbref2_ (same code, same box)",
     "same", P["diag_ref_it"], P["diag_ref2_it"], {}),
    ("27b-base :: section 7b anchor, RANKS -- topk_fmt(anchor_shipped) vs shipped topk sbref_ "
     "[the ONLY exact-gated 27b rank comparison, A4 -- gated OFFLINE]",
     "topk", P["topk_fmt_base"], P["topk_ref_base"], {"slot": "bare"}),
    ("27b-it   :: section 7b anchor, RANKS -- topk_fmt(anchor_shipped) vs shipped topk sbref_ "
     "[the ONLY exact-gated 27b rank comparison, A4 -- gated OFFLINE]",
     "topk", P["topk_fmt_it"], P["topk_ref_it"], {"slot": "bare"}),
    ("27b-base :: section 7b anchor, LP -- diagnose_fmt(space column) vs shipped diagnose A1 "
     "[27b lp is DISCLOSED_NOT_GATED, section 7.2 -- counts only, no gate anywhere]",
     "diag", P["diag_fmt_base"], P["A1"], {}),
    ("27b-base :: section 7b anchor, LP -- diagnose_fmt(space column) vs shipped diagnose A2 "
     "[DISCLOSED_NOT_GATED; section 10.3 branch 2 evaluates the anchor against the PAIR]",
     "diag", P["diag_fmt_base"], P["A2"], {}),
    ("27b-it   :: section 7b anchor, LP -- diagnose_fmt(space column) vs shipped diagnose sbref_ "
     "[DISCLOSED_NOT_GATED, section 7.2]",
     "diag", P["diag_fmt_it"], P["diag_ref_it"], {}),
]

print("[section 14.2] " + BANNER)
report = {}
for label, kind, pa, pb, opts in PAIRS:
    entry = {"a": pa, "b": pb, "kind": kind}
    try:
        ra, rb = load(pa), load(pb)
        if kind == "same":
            A, B = items(ra), items(rb)
            fields = list(A[0]) if A else []
        elif kind == "same_arms":
            A, B = items(ra), items(rb, arm="fold")
            fields = list(A[0]) if A else []          # the SHIPPED side's pre-existing fields
        elif kind == "topk":
            A = [proj_topk_fmt(x) for x in items(ra, slot=opts.get("slot"))]
            B = [proj_topk_shipped(x) for x in items(rb)]
            fields = list(TOPK_FIELDS)
        else:
            A = [proj_diag_fmt(x) for x in items(ra)]
            B = [proj_diag_shipped(x) for x in items(rb)]
            fields = list(DIAG_FIELDS)
        entry["n_items"] = {"a": len(A), "b": len(B)}
        if len(A) != len(B) or [x["q"] for x in A] != [x["q"] for x in B]:
            # Section 10.2's item-order rule: no reordering and no intersection. Fails loudly.
            entry["status"] = "ITEM_ORDER_OR_LENGTH_DIFFERS -> not comparable on box"
            print(f"[diff] {label}\n       {entry['status']} (a={len(A)} b={len(B)})")
        else:
            nz, firsts = counts(A, B, fields)
            entry.update({"status": "OK", "n_fields_compared": len(fields),
                          "fields_with_any_difference": nz, "n_fields_differing": len(nz),
                          "first_divergent_cell_per_field": firsts})
            print(f"[diff] {label}\n       n={len(A)} fields={len(fields)} "
                  f"differing={nz if nz else '{} -> 0 of ' + str(len(fields)) + ' fields differ'}")
    except Exception as e:
        entry["status"] = f"UNAVAILABLE: {type(e).__name__}: {e}"
        print(f"[diff] {label}\n       {entry['status']}")
        traceback.print_exc()
    report[label] = entry

# ---- section 11's MANDATORY 27b disclosure. All four parts, or a 27b digit is not quotable. -------
prov = None
try:
    prov = json.load(open("out/provenance_fmt_27b.json"))
except Exception:
    pass
DISCLOSURE = {
    "i_provenance_pair": {"lambda_instance_id": (prov or {}).get("lambda_instance_id"),
                          "started_utc": (prov or {}).get("started_utc"),
                          "gpu_name": (prov or {}).get("gpu_name"),
                          "driver": (prov or {}).get("driver"),
                          "cuda_visible_devices": (prov or {}).get("cuda_visible_devices"),
                          "device_index": (prov or {}).get("device_index")},
    "ii_section10_verdict": ("NOT EMITTED ON BOX (section 14.2). Section 10 ran on THIS box, in this "
                             "session, at 27b-base: draws stab27b_shipA / stab27b_shipB / stab27b_arms. "
                             "The section 10.3 verdict is computed offline by controls/fmt_matched_join.py "
                             "stab27b, which also does section 10.2's cluster fingerprinting."),
    "iii_card_class": ("this run's 27b box is gpu_1x_h100_sxm5, while EVERY committed 27b artifact is "
                       "H100 PCIe / driver 570.148.08 -- so NO 27b comparison against a committed "
                       "artifact separates code from hardware (section 7.2; A4). Every 27b-vs-committed "
                       "row is DISCLOSED_NOT_GATED and emits no reproduction verdict at all."),
    "iv_known_spread": ("27b teacher-forced lp digits have a measured ACROSS-BOX spread of median "
                        "0.009-0.13 and max 0.44-0.59 nats, and THREE value-clusters exist at 27b-base "
                        "(section 10.2's table). A 27b digit printed without parts (i)-(iv) is not quotable "
                        "(section 11)."),
}
print("[section 11] MANDATORY 27b DISCLOSURE:")
for k, v in DISCLOSURE.items():
    print(f"  {k}: {v}")

# ---- the small, decision-bearing file the launcher fetches FIRST (lambda_run.sh:219-226) ----------
cells = []
try:
    for line in open("out/fmt_cellstatus_27b.tsv"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            cells.append({"label": parts[0], "exit": int(parts[1]), "cmd": parts[2]})
except Exception as e:
    cells = [{"label": "<unreadable>", "exit": -1, "cmd": str(e)}]
summary = {
    "run": "fmt_matched_27b", "box": "B of 2 (27b cells + section 10)",
    "registration": "docs/drafts/REGISTRATION_format_matched_readout.md (frozen, pre-data, A1-A20)",
    "no_verdicts_on_box": BANNER,
    "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "provenance": prov,
    "disclosure_27b_section11": DISCLOSURE,
    "cells": cells,
    "n_cells": len(cells),
    "n_cells_nonzero_exit": sum(1 for c in cells if c["exit"] != 0),
    "onbox_raw_diff_counts": report,
}
json.dump(summary, open("out/fmt_matched_27b_summary.json", "w"), indent=2, default=str)
print("[section 14.2] wrote out/fmt_matched_27b_summary.json "
      f"({len(cells)} cells, {summary['n_cells_nonzero_exit']} with a non-zero exit)")
PY

# --- close the run-level provenance -----------------------------------------------------------------
python - <<'PY'
import json, datetime
try:
    p = json.load(open("out/provenance_fmt_27b.json"))
    p["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(p, open("out/provenance_fmt_27b.json", "w"), indent=2)
    print("[provenance] finished_utc", p["finished_utc"])
except Exception as e:
    print("provenance close failed:", e)
PY

echo "=== per-cell exit codes (a non-zero cell does NOT abandon the rest; RUN_DONE=0 != all passed) ==="
cat "$CELLSTATUS" 2>/dev/null || echo "<no cell status recorded>"
echo "=== artifacts ==="
ls -la out/family_topk_shift_fmt_*.json out/family_cave_diagnose_fmt_*.json \
       out/family_topk_shift_sbref_*.json out/family_cave_diagnose_sbref*_*.json \
       out/family_cave_diagnose_stab27b_*.json out/family_cave_diagnose_arms_stab27b_*.json 2>/dev/null
echo "ALLDONE_FMT_MATCHED_27B"
