"""FOLD/LISTEN RE-RUN REPRODUCTION DIFF (offline, model-free, CPU-only: no torch, no GPU, no network).

CONTEXT. DESIGN_neutral_elicit.md adds a 4th arm to controls/foldlisten_judge.py -- the neutral arm is now
ELICITED too. The change is claimed to be ADDITIVE under greedy decoding (§1.4): every call it adds sits
AFTER the last pre-existing generate(), so nothing that already existed may move when the cells are re-run.
§5 turns that claim into the run's first gate ("the re-run IS the gate") and §1.6 names THIS file as the
instrument: it takes (committed_summary, new_summary), compares them over the legacy key set, and persists
the verdict, because in this repo a gate decision must live as a committed artifact, not as a shell
one-liner. It is claim-blind about the numbers the run then produces: it only asks whether the OLD numbers
came back.

WHAT IT MEASURES. The number of DISAGREEING FIELDS between two summaries of the same cell, over the LEGACY
key set only -- the keys the committed baseline already carries. Out of scope by construction: the five new
neutral-elicited per-item fields (§1.2), the two new `cells` sub-keys, the two new top-level
push_attribution blocks, and `scorer_provenance` (§1.4 states its text legitimately changes). Compared,
item-for-item and IN ORDER:
  * per-item VALUE fields  -- prompts, generations, judge_reply_raw, conf_proxy, item identity: the bytes
    the model and the harness produced;
  * per-item LABEL fields  -- commit_*, faithful_*, faithful_rule_*, judge_label, read exactly as stored
    (nothing is re-scored here). For a PRE-PORT baseline carrying no faithful_* fields (the 9b-it ext2
    cell, §5.3) the baseline labels are read instead from a faithful_rescore_<tag>.json side-channel;
  * the stored aggregate buckets `cells` / `cells_faithful` (legacy sub-keys) and the top-level `decision`
    / `decision_faithful` blocks;
  * gate() and gate_v2() RECOMPUTED from each summary's own records under both label readings (commit and
    faithful), compared minus the one `measured` key the change adds.
Nothing is re-generated and nothing is re-labelled; both label families are read as stored.

PER-ITEM DUMP. Complete mismatch counts per key (never truncated), the sorted list of item indices with any
mismatch, and up to MAX_EXAMPLES worked examples {item_index, q, cell, key, kind, baseline value, new value}
with both values verbatim in the JSON (truncation happens only in the printed line).

AGGREGATE + DECISION (on the measured counts only; no claim is attached to either outcome, and neither
outcome is a success state of this instrument).
  n_value_mismatch, n_label_mismatch, n_derived_mismatch (aggregates + gates), frac_fields_identical, and:
    NOT_COMPARABLE   -- the pair could not be aligned or a compared field was missing, so no identity
                        statement is possible (distinct from, and never merged into, DIFF);
    DIFF             -- n_value_mismatch > 0, or an aggregate/gate moved with no per-item label difference
                        to explain it;
    LABELS_ONLY_DIFF -- every per-item value field is equal and the only per-item differences are LABEL
                        fields (with whatever aggregate/gate differences follow from them);
    BYTE_IDENTICAL   -- every compared legacy field is equal.
The decision is written into out/foldlisten_repro_diff_<tag>.json together with the metric, the thresholds
and this rule. The process exit code is not used to encode the decision (repo convention: read the JSON,
not a summary of it), so no outcome is dressed as pass/fail by the shell.

  python controls/foldlisten_repro_diff.py --selftest
  python controls/foldlisten_repro_diff.py \
      --committed results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json \
      --new       results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json
  python controls/foldlisten_repro_diff.py \
      --committed results_foldlisten_r2/out/foldlisten_judge_fl_9bit_ext2_summary.json \
      --new       results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_9bit_ext2_summary.json \
      --faithful-committed out/faithful_rescore_fl_9bit_ext2.json
"""
import argparse
import json
import sys
from pathlib import Path

# FLAT-scp: controls/ for sibling reuse, latent_verify/ for the repo imports (identical to the siblings).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The gate functions under reproduction, reused VERBATIM rather than reimplemented -- a re-derived gate
# would be a different instrument. foldlisten_judge's module top imports no torch (only argparse/json/sys/
# pathlib + the pure sibling matchers), so this import is CPU-safe and pulls no model machinery; the same
# justification the frozen siblings use (foldlisten_phase3a.py, foldlisten_phase3c_analysis.py).
# `_faithful_commit_records` is private but is the repo's own commit<-faithful remap, used here exactly as
# run_gate() uses it, so the --labels faithful reading is the same one the gate artifacts record.
from foldlisten_judge import CELLS, aggregate, gate, gate_v2, _faithful_commit_records  # noqa: E402

# --------------------------------------------------------------------------- FROZEN thresholds
# MAX_FIELD_MISMATCH: from the design, not chosen here. DESIGN_neutral_elicit.md §2.4 hard stop 1 ("any
#   legacy per-item field differs from the committed summary => the change is not additive; discard the
#   run") and §1.4 ("must reproduce byte-identically"). Zero tolerance: one disagreeing legacy field is
#   already more than the design permits, so the count is compared against 0.
MAX_FIELD_MISMATCH = 0
# FLOAT_EQ_TOL: from the same §1.4 byte-identity claim. Decoding is greedy and no RNG is consumed, so a
#   stored number either round-trips exactly or it does not; any positive tolerance would let a real
#   numeric drift read as a match. NaN never equals NaN under this rule (mismatch), which is the reading
#   that cannot manufacture an identity.
FLOAT_EQ_TOL = 0.0
# MAX_EXAMPLES: a reporting cap chosen here (nothing in the design fixes it); it bounds the worked-example
#   dump only. The counts are complete and the decision never depends on it.
MAX_EXAMPLES = 20

# --------------------------------------------------------------------------- FROZEN key scope
# The five per-item fields the change ADDS (DESIGN_neutral_elicit.md §1.2). Excluded from the comparison by
# construction -- the diff is restricted to the legacy key set (§5 step 3).
NEW_ARM_ITEM_KEYS = ("neutral_elicit_prompt", "neutral_elicit_gen", "commit_neutral_elicit",
                     "faithful_neutral_elicit", "faithful_rule_neutral_elicit")
# Per-item LABEL fields: a scorer's reading of a generation, as opposed to the generation itself.
COMMIT_ITEM_KEYS = ("commit_counter", "commit_neutral", "commit_elicit")
FAITHFUL_ITEM_KEYS = ("faithful_neutral", "faithful_counter", "faithful_elicit",
                      "faithful_rule_neutral", "faithful_rule_counter", "faithful_rule_elicit")
LABEL_ITEM_KEYS = COMMIT_ITEM_KEYS + FAITHFUL_ITEM_KEYS + ("judge_label",)
# Item identity: if any of these disagree at an index the two files are not the same item set in the same
# order, and comparing the rest of that record would compare unrelated items.
ITEM_ID_KEYS = ("q", "correct", "Wstar", "tier", "cell")
# Keys a new summary may legitimately carry that a baseline lacks: the new arm, plus the faithful_* family
# for the pre-port cell whose re-run is its first native dual-label run (§5.3). Anything else that appears
# is REPORTED (unexpected_new_item_fields), never silently absorbed -- and never counted as a mismatch,
# because an added field is not a legacy field failing to reproduce.
EXPECTED_ADDED_ITEM_KEYS = NEW_ARM_ITEM_KEYS + FAITHFUL_ITEM_KEYS

LEGACY_CELL_KEYS = ("n", "elicit", "counter", "neutral_drift")   # stored aggregate sub-keys that pre-exist
NEW_CELL_KEYS = ("neutral_elicit", "n_neutral_elicit")           # added by the change; out of scope
TOP_LEVEL_ID_KEYS = ("name", "regime")                           # must match or the pair is the wrong pair
TOP_LEVEL_COMPARED = ("decision", "decision_faithful")           # pure functions of the legacy labels
TOP_LEVEL_EXCLUDED = ("scorer_provenance",)                      # §1.4: its text legitimately changes
NEW_TOP_LEVEL_KEYS = ("push_attribution", "push_attribution_faithful")
GATE_EXCLUDE_MEASURED = ("neutral_elicit_diagnostic",)           # the one gate key the change adds

# Baseline faithful labels for a PRE-PORT summary come from a faithful_rescore_<tag>.json (§5.3):
# rescore generation-field -> (summary label field, summary rule field).
FAITHFUL_SIDECHANNEL_MAP = {
    "neutral_gen": ("faithful_neutral", "faithful_rule_neutral"),
    "counter_gen": ("faithful_counter", "faithful_rule_counter"),
    "elicit_gen": ("faithful_elicit", "faithful_rule_elicit"),
}

DECISIONS = ("BYTE_IDENTICAL", "LABELS_ONLY_DIFF", "DIFF", "NOT_COMPARABLE")

METRIC = (
    "Offline reproduction diff between two foldlisten_judge summaries of the SAME cell (committed vs "
    "re-run), restricted to the LEGACY key set: count disagreeing fields item-for-item and IN ORDER over "
    "the per-item VALUE fields (prompts, generations, judge_reply_raw, conf_proxy, item identity) and the "
    "per-item LABEL fields (commit_*, faithful_*, faithful_rule_*, judge_label, read as stored; baseline "
    "faithful labels optionally read from a faithful_rescore_<tag>.json side-channel for a pre-port "
    "summary), plus the stored aggregate buckets cells/cells_faithful (legacy sub-keys only), the "
    "top-level decision/decision_faithful blocks, and gate()/gate_v2() recomputed from each summary's own "
    "records under both label readings (minus the one measured key the change adds). The five new "
    "neutral-elicited per-item fields, the two new cells sub-keys, the two push_attribution blocks and "
    "scorer_provenance are out of scope by construction. Nothing is re-generated and nothing is "
    "re-labelled."
)

DECISION_RULE = (
    "Counts only, over the LEGACY key set defined by the committed baseline. n_value_mismatch = per-item "
    "non-label fields that differ; n_label_mismatch = per-item label fields that differ; "
    "n_derived_mismatch = differing stored cells/cells_faithful legacy buckets + decision/decision_faithful "
    "blocks + recomputed gate()/gate_v2() results under both label readings. Frozen threshold "
    "MAX_FIELD_MISMATCH = 0 with FLOAT_EQ_TOL = 0.0 (exact equality; NaN never matches). Resolution order: "
    "(1) NOT_COMPARABLE if the pair cannot be aligned or a compared field is missing -- differing item "
    "counts, differing item identity keys at any index, a legacy field absent from the new summary, a "
    "differing top-level name/regime, an unalignable faithful side-channel, a gate that cannot be "
    "computed, or a new summary carrying NONE of the neutral-elicited fields (it is then indistinguishable "
    "from a pre-change artifact, so an identity statement about it would certify nothing about the change "
    "under test); (2) else DIFF if n_value_mismatch > MAX_FIELD_MISMATCH; (3) else LABELS_ONLY_DIFF if "
    "n_label_mismatch > MAX_FIELD_MISMATCH; (4) else DIFF if n_derived_mismatch > MAX_FIELD_MISMATCH (an "
    "aggregate or gate moved with no per-item label difference to explain it); (5) else BYTE_IDENTICAL. "
    "NOT_COMPARABLE is never a match and is reported separately from DIFF. The new-arm presence verdict "
    "(ARM_PRESENT_COMPLETE / ARM_PARTIAL / ARM_ABSENT) is REPORTED alongside; only ARM_ABSENT enters the "
    "decision, via NOT_COMPARABLE. No claim is attached to any outcome and no outcome is an error state of "
    "this instrument; the counts fall where they fall."
)

# READINGS CHOSEN WHERE THE DESIGN IS SILENT OR AMBIGUOUS (each resolved toward NOT declaring a match):
#  (a) §1.6 lists decision in {BYTE_IDENTICAL, LABELS_ONLY_DIFF, DIFF} and does not name a category for a
#      pair that cannot be compared at all. A fourth category NOT_COMPARABLE is added rather than folding
#      such a pair into DIFF: DIFF would read as "the re-run moved a number" when in fact the instrument
#      never ran the comparison. It is never a match, so adding it cannot manufacture one.
#  (b) §1.6 does not define the LABELS_ONLY_DIFF / DIFF boundary. Read as: LABELS_ONLY_DIFF requires every
#      per-item VALUE field to be equal (the model's bytes reproduced) with only stored LABEL fields
#      moving; an aggregate/gate difference with no per-item label difference behind it is an internal
#      inconsistency of one artifact and is reported as DIFF, the harsher of the two.
#  (c) §1.6 says "the gate decisions" (plural) while §5.3 names gate_v2. Both gate() and gate_v2() are
#      compared, under both label readings -- more comparisons can only make a declared match harder.
#  (d) §1.6 says the diff "asserts" equality; the same sentence requires a persisted decision artifact. It
#      is implemented as a measured count + a written decision (never a raised assertion), so the number is
#      always recorded; the caller reads the artifact.
#  (e) The baseline defines the legacy key set (rather than a hard-coded list), so the pre-port 9b-it cell
#      -- which carries no faithful_* fields -- is handled without a special case, and any field the
#      committed artifact carries is required to come back.


# --------------------------------------------------------------------------- pure: value equality
def values_equal(a, b, tol=FLOAT_EQ_TOL):
    """Equality test for one stored field. Numbers compare with |a-b| <= tol (tol frozen at 0.0, so this is
    exact equality and NaN != NaN -> mismatch, the reading that cannot manufacture an identity); everything
    else (strings, None, lists, dicts) compares with ==. Pure (any, any -> bool)."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        try:
            return abs(float(a) - float(b)) <= tol
        except (OverflowError, ValueError):
            return False
    return a == b


# --------------------------------------------------------------------------- pure: the frozen decision
def classify_decision(n_not_comparable, n_value_mismatch, n_label_mismatch, n_derived_mismatch,
                      thr=MAX_FIELD_MISMATCH):
    """The frozen decision over the four measured counts (see DECISION_RULE). Resolution order:
    NOT_COMPARABLE -> DIFF (value) -> LABELS_ONLY_DIFF (label) -> DIFF (derived-only) -> BYTE_IDENTICAL.
    Pure (int,int,int,int -> str)."""
    if n_not_comparable > 0:
        return "NOT_COMPARABLE"
    if n_value_mismatch > thr:
        return "DIFF"
    if n_label_mismatch > thr:
        return "LABELS_ONLY_DIFF"
    if n_derived_mismatch > thr:
        return "DIFF"
    return "BYTE_IDENTICAL"


# --------------------------------------------------------------------------- pure: new-arm presence
def arm_presence(new_items):
    """Presence of the five new neutral-elicited per-item fields (§1.2) in the NEW summary's records.
    ARM_PRESENT_COMPLETE (every record carries all five) / ARM_PARTIAL (some do, some do not -- the §2.4
    'INSUFFICIENT' shape) / ARM_ABSENT (no record carries any, including an empty item list). REPORTED
    alongside the identity decision; only ARM_ABSENT enters it. Pure (list -> dict)."""
    n = len(new_items)
    n_full = sum(1 for r in new_items if all(k in r for k in NEW_ARM_ITEM_KEYS))
    n_none = sum(1 for r in new_items if not any(k in r for k in NEW_ARM_ITEM_KEYS))
    missing = {}
    for r in new_items:
        for k in NEW_ARM_ITEM_KEYS:
            if k not in r:
                missing[k] = missing.get(k, 0) + 1
    if n and n_full == n:
        verdict = "ARM_PRESENT_COMPLETE"
    elif n_none == n:
        verdict = "ARM_ABSENT"
    else:
        verdict = "ARM_PARTIAL"
    return {"verdict": verdict, "n_items": n, "n_records_with_all_five": n_full,
            "n_records_with_none": n_none, "missing_counts": missing}


# --------------------------------------------------------------------------- pure: faithful side-channel
def sidechannel_labels(artifact, base_items):
    """Baseline faithful labels for a PRE-PORT summary, read out of a faithful_rescore_<tag>.json (§5.3).
    Returns ({gen_field: {item_index: {new_label, rule_fired}}}, reasons): `reasons` is non-empty when the
    artifact cannot be aligned to the summary item-for-item (missing field block, wrong record count, or a
    q/cell that does not match the summary item at that index) -- an unalignable side-channel is a
    NOT_COMPARABLE input, never a silent skip. Pure (dict, list -> (dict, list))."""
    out, reasons = {}, []
    fields = artifact.get("fields")
    if not isinstance(fields, dict):
        return {}, ["faithful side-channel: no 'fields' block; cannot read baseline faithful labels"]
    for fld in FAITHFUL_SIDECHANNEL_MAP:
        blk = fields.get(fld)
        if not isinstance(blk, dict) or not isinstance(blk.get("items"), list):
            reasons.append("faithful side-channel: field %r absent (or carries no items list)" % fld)
            continue
        recs = blk["items"]
        if len(recs) != len(base_items):
            reasons.append("faithful side-channel %r: %d records vs %d summary items; cannot align"
                           % (fld, len(recs), len(base_items)))
            continue
        m, ok = {}, True
        for i, (rec, bit) in enumerate(zip(recs, base_items)):
            if rec.get("q") != bit.get("q") or rec.get("cell") != bit.get("cell"):
                reasons.append("faithful side-channel %r: record %d (q=%r, cell=%r) does not match summary "
                               "item %d (q=%r, cell=%r)" % (fld, i, rec.get("q"), rec.get("cell"), i,
                                                            bit.get("q"), bit.get("cell")))
                ok = False
                break
            m[i] = {"new_label": rec.get("new_label"), "rule_fired": rec.get("rule_fired")}
        if ok:
            out[fld] = m
    return out, reasons


# --------------------------------------------------------------------------- pure: per-item comparison
def _blank_item_report():
    """Empty per-item comparison report. `_hit` is a working set, removed before the report is returned."""
    return {"structural": [], "examples": [], "n_examples_omitted": 0, "n_fields_compared": 0,
            "mismatch_counts_by_key": {}, "kind_counts": {"value": 0, "label": 0},
            "unexpected_new_item_fields": {}, "items_with_mismatch": [], "_hit": set()}


def _record_item_mismatch(rep, idx, item, key, kind, bval, nval):
    """Accumulate one per-item field mismatch: complete counts always, at most MAX_EXAMPLES worked
    examples (the cap bounds the dump only, never the counts or the decision). Pure bookkeeping."""
    rep["mismatch_counts_by_key"][key] = rep["mismatch_counts_by_key"].get(key, 0) + 1
    rep["kind_counts"][kind] = rep["kind_counts"].get(kind, 0) + 1
    rep["_hit"].add(idx)
    if len(rep["examples"]) < MAX_EXAMPLES:
        rep["examples"].append({"item_index": idx, "q": item.get("q"), "cell": item.get("cell"),
                                "key": key, "kind": kind, "baseline": bval, "new": nval})
    else:
        rep["n_examples_omitted"] += 1


def compare_items(base_items, new_items, sidechannel=None):
    """Item-for-item, IN ORDER, over the legacy key set the BASELINE record defines (minus the five new-arm
    keys). Every baseline key must be present in the aligned new record; a missing one is STRUCTURAL (the
    comparison could not be run), not a mismatch. Keys the new record adds are reported, never counted.
    `sidechannel` (optional, from sidechannel_labels) supplies baseline faithful labels for keys the
    baseline summary does not carry; a key the baseline summary DOES carry is compared against the summary
    and the side-channel value for it is left unused. Pure (list, list, dict|None -> dict)."""
    rep = _blank_item_report()
    if len(base_items) != len(new_items):
        rep["structural"].append("item count differs: baseline %d vs new %d; no alignment is possible"
                                 % (len(base_items), len(new_items)))
        rep.pop("_hit")
        return rep
    for i, (b, n) in enumerate(zip(base_items, new_items)):
        bad_id = [k for k in ITEM_ID_KEYS if k in b and not values_equal(b.get(k), n.get(k))]
        if bad_id:
            rep["structural"].append(
                "item %d: identity key(s) %s differ (baseline q=%r cell=%r vs new q=%r cell=%r); the two "
                "summaries are not the same item set in the same order"
                % (i, ", ".join(bad_id), b.get("q"), b.get("cell"), n.get("q"), n.get("cell")))
            continue
        for k in b:
            if k in NEW_ARM_ITEM_KEYS:
                continue
            if k not in n:
                rep["structural"].append("item %d: legacy field %r present in the baseline summary but "
                                         "MISSING from the new one" % (i, k))
                continue
            kind = "label" if k in LABEL_ITEM_KEYS else "value"
            rep["n_fields_compared"] += 1
            if not values_equal(b[k], n[k]):
                _record_item_mismatch(rep, i, b, k, kind, b[k], n[k])
        for k in n:
            if k in b or k in EXPECTED_ADDED_ITEM_KEYS:
                continue
            rep["unexpected_new_item_fields"][k] = rep["unexpected_new_item_fields"].get(k, 0) + 1
        if sidechannel:
            for fld, (lab_key, rule_key) in FAITHFUL_SIDECHANNEL_MAP.items():
                sc = sidechannel.get(fld, {}).get(i)
                if sc is None:
                    continue
                for key, bval in ((lab_key, sc["new_label"]), (rule_key, sc["rule_fired"])):
                    if key in b:
                        continue                       # already compared against the baseline summary
                    if key not in n:
                        rep["structural"].append("item %d: field %r supplied by the faithful side-channel "
                                                 "but MISSING from the new summary" % (i, key))
                        continue
                    rep["n_fields_compared"] += 1
                    if not values_equal(bval, n[key]):
                        _record_item_mismatch(rep, i, b, key, "label", bval, n[key])
    rep["items_with_mismatch"] = sorted(rep.pop("_hit"))
    return rep


# --------------------------------------------------------------------------- pure: stored aggregates
def compare_cells(base_cells, new_cells, block):
    """One stored aggregate block ('cells' / 'cells_faithful') over LEGACY_CELL_KEYS only; NEW_CELL_KEYS are
    out of scope. A block absent from the BASELINE is nothing to reproduce (reported in not_compared); a
    block the baseline carries and the new summary lacks is STRUCTURAL. Pure (dict|None, dict|None, str ->
    dict)."""
    rep = {"mismatches": [], "structural": [], "n_compared": 0, "not_compared": []}
    if base_cells is None:
        rep["not_compared"].append("%s: absent from the baseline summary (nothing to reproduce)" % block)
        return rep
    if new_cells is None:
        rep["structural"].append("%s: present in the baseline summary but ABSENT from the new one" % block)
        return rep
    for c in CELLS:
        b, n = base_cells.get(c), new_cells.get(c)
        if b is None:
            rep["not_compared"].append("%s.%s: absent from the baseline block" % (block, c))
            continue
        if n is None:
            rep["structural"].append("%s.%s: present in the baseline block but ABSENT from the new one"
                                     % (block, c))
            continue
        for k in LEGACY_CELL_KEYS:
            if k not in b:
                rep["not_compared"].append("%s.%s.%s: absent from the baseline block" % (block, c, k))
                continue
            if k not in n:
                rep["structural"].append("%s.%s.%s: present in the baseline block but ABSENT from the new "
                                         "one" % (block, c, k))
                continue
            rep["n_compared"] += 1
            if not values_equal(b[k], n[k]):
                rep["mismatches"].append({"where": "%s.%s.%s" % (block, c, k),
                                          "baseline": b[k], "new": n[k]})
    return rep


def compare_top_level(base, new):
    """The legacy top-level blocks that are pure functions of the legacy labels (TOP_LEVEL_COMPARED).
    TOP_LEVEL_EXCLUDED is out of scope (§1.4), NEW_TOP_LEVEL_KEYS are new, cells/cells_faithful have their
    own comparator and name/regime are identity checks handled by the caller. Pure (dict, dict -> dict)."""
    rep = {"mismatches": [], "structural": [], "n_compared": 0, "not_compared": []}
    for k in TOP_LEVEL_COMPARED:
        if k not in base:
            rep["not_compared"].append("top-level %r: absent from the baseline summary (nothing to "
                                       "reproduce)" % k)
            continue
        if k not in new:
            rep["structural"].append("top-level %r: present in the baseline summary but ABSENT from the "
                                     "new one" % k)
            continue
        rep["n_compared"] += 1
        if not values_equal(base[k], new[k]):
            rep["mismatches"].append({"where": "top_level.%s" % k, "baseline": base[k], "new": new[k]})
    return rep


# --------------------------------------------------------------------------- pure: recomputed gates
def _strip_gate(g):
    """A gate result with the ONE measured key the change adds removed (GATE_EXCLUDE_MEASURED), so a
    pre-change and a post-change gate are comparable at all. Everything else -- decision, checks,
    thresholds, the rest of measured, sensitivity, decision_rule -- is kept. Pure (dict -> dict)."""
    out = dict(g)
    measured = g.get("measured")
    if isinstance(measured, dict):
        out["measured"] = {k: v for k, v in measured.items() if k not in GATE_EXCLUDE_MEASURED}
    return out


def _relabel(records, labels):
    """Records fed to a gate under one label reading: 'commit' as stored, 'faithful' via the repo's own
    faithful_to_commit remap (raises KeyError naming the first missing faithful field). Pure."""
    return _faithful_commit_records(records) if labels == "faithful" else list(records)


def gate_pair(base_items, new_items, labels, gate_fn, gate_name):
    """One gate recomputed from EACH summary's own records under one label reading and compared field by
    field (after _strip_gate). status: COMPARED / SKIPPED (a side lacks that label family -- reported, not
    a mismatch) / ERROR (the gate raised on a side; the comparison could not be run, so it is STRUCTURAL
    and can never read as a match). Pure apart from calling the imported gate. -> dict."""
    rep = {"gate": gate_name, "labels": labels, "status": "COMPARED", "reason": None,
           "baseline_decision": None, "new_decision": None, "mismatches": []}
    try:
        b_recs = _relabel(base_items, labels)
    except KeyError as e:
        rep["status"] = "SKIPPED"
        rep["reason"] = ("%s[labels=%s]: the baseline summary lacks that label family: %s"
                         % (gate_name, labels, e))
        return rep
    try:
        n_recs = _relabel(new_items, labels)
    except KeyError as e:
        rep["status"] = "SKIPPED"
        rep["reason"] = ("%s[labels=%s]: the new summary lacks that label family: %s"
                         % (gate_name, labels, e))
        return rep
    try:
        gb, gn = _strip_gate(gate_fn(b_recs)), _strip_gate(gate_fn(n_recs))
    except Exception as e:                      # noqa: BLE001 -- an uncomputable gate is never a match
        rep["status"] = "ERROR"
        rep["reason"] = ("%s[labels=%s] could not be computed: %s: %s"
                         % (gate_name, labels, type(e).__name__, e))
        return rep
    rep["baseline_decision"], rep["new_decision"] = gb.get("decision"), gn.get("decision")
    for k in sorted(set(gb) | set(gn)):
        if not values_equal(gb.get(k), gn.get(k)):
            rep["mismatches"].append({"where": "%s[labels=%s].%s" % (gate_name, labels, k),
                                      "baseline": gb.get(k), "new": gn.get(k)})
    return rep


# --------------------------------------------------------------------------- the diff (pure)
def diff_summaries(base, new, faithful_baseline=None):
    """Full reproduction diff of two summary dicts (+ an optional faithful_rescore artifact supplying the
    baseline faithful labels for a pre-port summary). PURE: no file IO, no model, no re-scoring -- the
    selftest drives it directly and run() only loads, calls and writes. -> the output dict."""
    reasons = []                                  # not-comparable reasons (structural, from every stage)
    for k in TOP_LEVEL_ID_KEYS:
        if base.get(k) != new.get(k):
            reasons.append("top-level %r differs (baseline %r vs new %r); these are not two runs of the "
                           "same cell" % (k, base.get(k), new.get(k)))
    base_items, new_items = base.get("items"), new.get("items")
    if not isinstance(base_items, list):
        reasons.append("baseline summary carries no 'items' list")
        base_items = []
    if not isinstance(new_items, list):
        reasons.append("new summary carries no 'items' list")
        new_items = []

    arm = arm_presence(new_items)
    if arm["verdict"] == "ARM_ABSENT":
        reasons.append("the new summary carries NONE of the neutral-elicited fields %s; it is "
                       "indistinguishable from a pre-change artifact, so an identity statement about it "
                       "would certify nothing about the change under test"
                       % (list(NEW_ARM_ITEM_KEYS),))

    sidechannel = {}
    if faithful_baseline is not None:
        sidechannel, sc_reasons = sidechannel_labels(faithful_baseline, base_items)
        reasons.extend(sc_reasons)

    items_rep = compare_items(base_items, new_items, sidechannel or None)
    cells_rep = compare_cells(base.get("cells"), new.get("cells"), "cells")
    cellsf_rep = compare_cells(base.get("cells_faithful"), new.get("cells_faithful"), "cells_faithful")
    top_rep = compare_top_level(base, new)
    gates = [gate_pair(base_items, new_items, labels, fn, name)
             for name, fn in (("gate", gate), ("gate_v2", gate_v2))
             for labels in ("commit", "faithful")]

    reasons.extend(items_rep["structural"])
    for rep in (cells_rep, cellsf_rep, top_rep):
        reasons.extend(rep["structural"])
    for g in gates:
        if g["status"] == "ERROR":
            reasons.append(g["reason"])

    not_compared = list(cells_rep["not_compared"]) + list(cellsf_rep["not_compared"]) \
        + list(top_rep["not_compared"]) + [g["reason"] for g in gates if g["status"] == "SKIPPED"]

    n_value = items_rep["kind_counts"]["value"]
    n_label = items_rep["kind_counts"]["label"]
    derived_mismatches = (list(cells_rep["mismatches"]) + list(cellsf_rep["mismatches"])
                          + list(top_rep["mismatches"])
                          + [m for g in gates for m in g["mismatches"]])
    n_derived = len(derived_mismatches)
    n_derived_compared = (cells_rep["n_compared"] + cellsf_rep["n_compared"] + top_rep["n_compared"]
                          + sum(1 for g in gates if g["status"] == "COMPARED"))
    n_item_fields = items_rep["n_fields_compared"]
    decision = classify_decision(len(reasons), n_value, n_label, n_derived)

    return {
        "control": "foldlisten_repro_diff",
        "metric": METRIC,
        "decision_rule": DECISION_RULE,
        "decision_space": list(DECISIONS),
        "thresholds": {"max_field_mismatch": MAX_FIELD_MISMATCH, "float_eq_tol": FLOAT_EQ_TOL,
                       "max_examples_dumped": MAX_EXAMPLES},
        "scope": {
            "legacy_key_set": ("every per-item key the COMMITTED baseline record carries, minus the five "
                               "new-arm keys; plus the legacy cells sub-keys, decision/decision_faithful, "
                               "and gate()/gate_v2() under both label readings"),
            "excluded_item_keys": list(NEW_ARM_ITEM_KEYS),
            "excluded_cell_keys": list(NEW_CELL_KEYS),
            "excluded_top_level_keys": list(TOP_LEVEL_EXCLUDED) + list(NEW_TOP_LEVEL_KEYS),
            "excluded_gate_measured_keys": list(GATE_EXCLUDE_MEASURED),
            "label_item_keys": list(LABEL_ITEM_KEYS),
            "item_identity_keys": list(ITEM_ID_KEYS),
            "faithful_sidechannel_map": {k: list(v) for k, v in FAITHFUL_SIDECHANNEL_MAP.items()},
        },
        "measured": {
            "n_items_baseline": len(base_items), "n_items_new": len(new_items),
            "n_item_fields_compared": n_item_fields,
            "n_derived_fields_compared": n_derived_compared,
            "n_value_mismatch": n_value, "n_label_mismatch": n_label, "n_derived_mismatch": n_derived,
            "n_items_with_any_mismatch": len(items_rep["items_with_mismatch"]),
            "items_with_mismatch": items_rep["items_with_mismatch"],
            "frac_item_fields_identical": (None if not n_item_fields
                                           else 1.0 - (n_value + n_label) / n_item_fields),
            "mismatch_counts_by_key": items_rep["mismatch_counts_by_key"],
            "unexpected_new_item_fields": items_rep["unexpected_new_item_fields"],
        },
        "new_arm_presence": arm,
        "aggregate_blocks": {"cells": cells_rep, "cells_faithful": cellsf_rep, "top_level": top_rep},
        "gates": gates,
        "derived_mismatches": derived_mismatches,
        "examples": items_rep["examples"],
        "n_examples_omitted": items_rep["n_examples_omitted"],
        "not_comparable_reasons": reasons,
        "not_compared": not_compared,
        "decision": decision,
    }


# --------------------------------------------------------------------------- i/o + run
def _tag_of(new_path):
    """Tag derived from the new summary's filename, mirroring foldlisten_judge.run_gate's derivation."""
    return Path(new_path).stem.replace("foldlisten_judge_", "").replace("_summary", "")


def run(committed, new, tag=None, outdir="out", faithful_committed=None):
    """Load the two summaries (+ an optional faithful_rescore side-channel), diff them, persist
    <outdir>/foldlisten_repro_diff_<tag>.json and print the counts. Reads JSON only (no model, no GPU, no
    network). The exit code does not encode the decision -- the artifact does."""
    base = json.loads(Path(committed).read_text(encoding="utf-8"))
    newd = json.loads(Path(new).read_text(encoding="utf-8"))
    fb = json.loads(Path(faithful_committed).read_text(encoding="utf-8")) if faithful_committed else None
    res = diff_summaries(base, newd, fb)
    tag = tag or _tag_of(new)
    res["tag"] = tag
    res["inputs"] = {"committed_summary": str(committed).replace("\\", "/"),
                     "new_summary": str(new).replace("\\", "/"),
                     "faithful_committed": (str(faithful_committed).replace("\\", "/")
                                            if faithful_committed else None)}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outp = outdir / ("foldlisten_repro_diff_%s.json" % tag)
    outp.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")

    m = res["measured"]
    print("[repro_diff %s] %s  (committed=%s, new=%s)"
          % (tag, res["decision"], Path(committed).name, Path(new).name), flush=True)
    print("  items %d/%d | item-fields compared=%d | derived compared=%d | new_arm=%s"
          % (m["n_items_baseline"], m["n_items_new"], m["n_item_fields_compared"],
             m["n_derived_fields_compared"], res["new_arm_presence"]["verdict"]), flush=True)
    print("  value_mismatch=%d label_mismatch=%d derived_mismatch=%d (threshold: > %d is a mismatch)"
          % (m["n_value_mismatch"], m["n_label_mismatch"], m["n_derived_mismatch"], MAX_FIELD_MISMATCH),
          flush=True)
    if m["mismatch_counts_by_key"]:
        print("  per-key mismatch counts: %s" % (m["mismatch_counts_by_key"],), flush=True)
    for g in res["gates"]:
        print("  %s[labels=%s] %s: baseline=%s new=%s mismatches=%d%s"
              % (g["gate"], g["labels"], g["status"], g["baseline_decision"], g["new_decision"],
                 len(g["mismatches"]), "" if not g["reason"] else " (%s)" % g["reason"]), flush=True)
    for r in res["not_comparable_reasons"]:
        print("  [not comparable] %s" % r, flush=True)
    for r in res["not_compared"]:
        print("  [not compared] %s" % r, flush=True)
    for ex in res["examples"][:5]:
        print("  [%s] item %d %r %s: baseline=%.120r new=%.120r"
              % (ex["kind"], ex["item_index"], ex["cell"], ex["key"], ex["baseline"], ex["new"]),
              flush=True)
    print("[written] %s" % str(outp).replace("\\", "/"), flush=True)
    return res


# --------------------------------------------------------------------------- selftest (model-free, CPU)
def _copy(o):
    """Deep copy through JSON (the planted fixtures are JSON-shaped by construction)."""
    return json.loads(json.dumps(o))


def _mk_item(i, cell, arm=True, faithful=True):
    """One planted per-item record in the committed shape (legacy fields), optionally + the faithful_*
    family and + the five new-arm fields."""
    r = {"q": "q%d" % i, "correct": "Nile", "Wstar": "Amazon", "tier": "T1", "cell": cell,
         "conf_proxy": 1.0 + i,
         "stated": "Nile" if cell == "fold" else "Amazon",
         "pushed": "Amazon" if cell == "fold" else "Nile",
         "counter_prompt": "CP%d%s" % (i, cell), "neutral_prompt": "NP%d%s" % (i, cell),
         "elicit_prompt": "EP%d%s" % (i, cell), "judge_prompt": "JP%d%s" % (i, cell),
         "counter_gen": "cg%d%s" % (i, cell), "neutral_gen": "ng%d%s" % (i, cell),
         "elicit_gen": "eg%d%s" % (i, cell), "counter_first_tok": "N",
         "commit_counter": "wrong" if cell == "fold" else "correct",
         "commit_neutral": "correct" if cell == "fold" else "wrong",
         "commit_elicit": "wrong" if cell == "fold" else "correct",
         "judge_label": "WRONG" if cell == "fold" else "CORRECT",
         "judge_reply_raw": "WRONG" if cell == "fold" else "CORRECT"}
    if faithful:
        r.update({"faithful_neutral": "C" if cell == "fold" else "WSTAR",
                  "faithful_counter": "WSTAR" if cell == "fold" else "C",
                  "faithful_elicit": "WSTAR" if cell == "fold" else "C",
                  "faithful_rule_neutral": "bare_entity_%d" % i,
                  "faithful_rule_counter": "affirmative_%d" % i,
                  "faithful_rule_elicit": "bare_entity_W"})
    if arm:
        r.update({"neutral_elicit_prompt": "NEP%d%s" % (i, cell),
                  "neutral_elicit_gen": "neg%d%s" % (i, cell),
                  "commit_neutral_elicit": "correct" if cell == "fold" else "wrong",
                  "faithful_neutral_elicit": "C" if cell == "fold" else "WSTAR",
                  "faithful_rule_neutral_elicit": "bare_entity_C"})
    return r


def _legacy_cells(cells):
    """The stored aggregate block as a PRE-change artifact carries it (legacy sub-keys only)."""
    return {c: {k: v for k, v in cells[c].items() if k in LEGACY_CELL_KEYS} for c in CELLS}


def _mk_summary(n_pairs=4, arm=True, faithful=True):
    """A planted summary in the on-disk shape: fold+listen record per item, stored cells (+cells_faithful),
    decision blocks. arm=False reproduces the committed pre-change shape (and a differing
    scorer_provenance, which the diff must ignore)."""
    items = []
    for i in range(n_pairs):
        items.append(_mk_item(i, "fold", arm=arm, faithful=faithful))
        items.append(_mk_item(i, "listen", arm=arm, faithful=faithful))
    cells = aggregate(items)
    s = {"name": "google/gemma-2-9b", "regime": "qa",
         "cells": cells if arm else _legacy_cells(cells),
         "decision": {"category": "MOVEMENT_BOTH", "fold_rate": 1.0}}
    if faithful:
        fcells = aggregate(_faithful_commit_records(items))
        s["cells_faithful"] = fcells if arm else _legacy_cells(fcells)
        s["decision_faithful"] = {"category": "MOVEMENT_BOTH", "fold_rate": 1.0}
    s["scorer_provenance"] = {"faithful_labels": "new text" if arm else "old text"}
    if arm:
        s["push_attribution"] = {"cells": {}}
    s["items"] = items
    return s


def _mk_sidechannel(items):
    """A faithful_rescore_<tag>.json-shaped artifact carrying the faithful labels of `items`."""
    fields = {}
    for fld, (lab_key, rule_key) in FAITHFUL_SIDECHANNEL_MAP.items():
        fields[fld] = {"items": [{"q": it["q"], "cell": it["cell"], "new_label": it[lab_key],
                                  "rule_fired": it[rule_key]} for it in items]}
    return {"control": "faithful_rescore", "fields": fields}


def selftest():
    # ---------- values_equal: exact equality, both sides of FLOAT_EQ_TOL(0.0), containers, NaN ----------
    assert values_equal(1.0, 1.0) and values_equal(3, 3.0)
    assert not values_equal(1.0, 1.0 + 1e-12)          # tol is 0.0: a 1e-12 drift is a mismatch
    assert values_equal("a", "a") and not values_equal("a", "b")
    assert values_equal(None, None) and not values_equal(None, "")
    assert values_equal({"a": [1, 2]}, {"a": [1, 2]}) and not values_equal({"a": [1, 2]}, {"a": [2, 1]})
    assert not values_equal(float("nan"), float("nan"))    # conservative: NaN never declares a match
    print("[selftest] values_equal: exact equality, 1e-12 apart -> mismatch, containers, NaN OK")

    # ---------- classify_decision: every branch, on BOTH sides of MAX_FIELD_MISMATCH(0) ----------
    assert classify_decision(0, 0, 0, 0) == "BYTE_IDENTICAL"
    assert classify_decision(0, 1, 0, 0) == "DIFF"                 # one value field over the threshold
    assert classify_decision(0, 0, 1, 0) == "LABELS_ONLY_DIFF"     # one label field over the threshold
    assert classify_decision(0, 0, 0, 1) == "DIFF"                 # derived moved with no label to explain
    assert classify_decision(0, 0, 3, 9) == "LABELS_ONLY_DIFF"     # derived follows the labels
    assert classify_decision(0, 1, 5, 5) == "DIFF"                 # a value diff dominates
    assert classify_decision(1, 0, 0, 0) == "NOT_COMPARABLE"
    assert classify_decision(1, 9, 9, 9) == "NOT_COMPARABLE"       # never masked by a mismatch count
    assert set(DECISIONS) == {"BYTE_IDENTICAL", "LABELS_ONLY_DIFF", "DIFF", "NOT_COMPARABLE"}
    print("[selftest] classify_decision: 0/1 boundary on each count + NOT_COMPARABLE precedence OK")

    # ---------- arm_presence: complete / partial / absent ----------
    full = _mk_summary(arm=True)["items"]
    assert arm_presence(full)["verdict"] == "ARM_PRESENT_COMPLETE"
    assert arm_presence(_mk_summary(arm=False)["items"])["verdict"] == "ARM_ABSENT"
    assert arm_presence([])["verdict"] == "ARM_ABSENT"
    part = _copy(full)
    for k in NEW_ARM_ITEM_KEYS:
        del part[0][k]
    ap = arm_presence(part)
    assert ap["verdict"] == "ARM_PARTIAL" and ap["n_records_with_all_five"] == len(part) - 1, ap
    assert ap["missing_counts"]["neutral_elicit_gen"] == 1, ap
    print("[selftest] arm_presence: COMPLETE / PARTIAL / ABSENT (empty list -> ABSENT) OK")

    # ---------- the additive case: identical legacy fields, new arm added -> BYTE_IDENTICAL ----------
    base = _mk_summary(arm=False)
    new = _mk_summary(arm=True)
    r = diff_summaries(base, new)
    assert r["decision"] == "BYTE_IDENTICAL", (r["decision"], r["not_comparable_reasons"],
                                               r["measured"]["mismatch_counts_by_key"])
    m = r["measured"]
    assert m["n_value_mismatch"] == 0 and m["n_label_mismatch"] == 0 and m["n_derived_mismatch"] == 0, m
    assert m["n_item_fields_compared"] > 0 and m["n_derived_fields_compared"] > 0, m
    assert m["frac_item_fields_identical"] == 1.0, m
    assert r["new_arm_presence"]["verdict"] == "ARM_PRESENT_COMPLETE"
    assert not r["not_comparable_reasons"], r["not_comparable_reasons"]
    assert not m["unexpected_new_item_fields"], m["unexpected_new_item_fields"]
    for g in r["gates"]:
        assert g["status"] == "COMPARED" and not g["mismatches"], g
        assert g["baseline_decision"] == g["new_decision"], g
    json.dumps(r, default=str)                                     # the artifact must serialize
    n_fields_no_sc = m["n_item_fields_compared"]
    print("[selftest] additive case (5 new fields, everything else equal) -> BYTE_IDENTICAL, "
          "%d item fields + %d derived compared OK" % (n_fields_no_sc, m["n_derived_fields_compared"]))

    # ---------- one VALUE field moved -> DIFF (exactly one, at the 0 boundary) ----------
    v = _copy(new)
    v["items"][3]["elicit_gen"] = "eg1listen-CHANGED"
    rv = diff_summaries(base, v)
    assert rv["decision"] == "DIFF", rv["decision"]
    assert rv["measured"]["n_value_mismatch"] == 1 and rv["measured"]["n_label_mismatch"] == 0
    assert rv["measured"]["mismatch_counts_by_key"] == {"elicit_gen": 1}, rv["measured"]
    assert rv["measured"]["items_with_mismatch"] == [3], rv["measured"]
    assert rv["examples"][0]["kind"] == "value" and rv["examples"][0]["key"] == "elicit_gen"
    # a float that drifts below any printable resolution is still a value mismatch (FLOAT_EQ_TOL = 0.0)
    vf = _copy(new)
    vf["items"][1]["conf_proxy"] = vf["items"][1]["conf_proxy"] + 1e-12
    rvf = diff_summaries(base, vf)
    assert rvf["decision"] == "DIFF" and rvf["measured"]["mismatch_counts_by_key"] == {"conf_proxy": 1}, rvf
    print("[selftest] one generation changed -> DIFF (1 value mismatch); 1e-12 conf_proxy drift -> DIFF OK")

    # ---------- one LABEL moved (generations identical) -> LABELS_ONLY_DIFF, gates follow ----------
    lab = _copy(new)
    lab["items"][0]["commit_elicit"] = "other"                     # fold: moved -> abstain
    lab["cells"] = aggregate(lab["items"])                         # stored aggregate follows the label
    rl = diff_summaries(base, lab)
    assert rl["decision"] == "LABELS_ONLY_DIFF", (rl["decision"], rl["not_comparable_reasons"])
    assert rl["measured"]["n_value_mismatch"] == 0 and rl["measured"]["n_label_mismatch"] == 1, rl["measured"]
    assert rl["measured"]["mismatch_counts_by_key"] == {"commit_elicit": 1}, rl["measured"]
    assert rl["measured"]["n_derived_mismatch"] > 0, rl["measured"]     # cells + the commit-label gates
    gv2c = [g for g in rl["gates"] if g["gate"] == "gate_v2" and g["labels"] == "commit"][0]
    assert gv2c["baseline_decision"] != gv2c["new_decision"], gv2c      # 1/4 fold abstain > 3/22-frac
    gv2f = [g for g in rl["gates"] if g["gate"] == "gate_v2" and g["labels"] == "faithful"][0]
    assert gv2f["status"] == "COMPARED" and not gv2f["mismatches"], gv2f  # faithful labels untouched
    print("[selftest] one commit label moved -> LABELS_ONLY_DIFF (commit gate decision moves, faithful "
          "gate does not) OK")

    # ---------- an aggregate that moved with NO per-item label behind it -> DIFF (harsher branch) ----------
    ag = _copy(new)
    ag["cells"]["fold"]["neutral_drift"] = ag["cells"]["fold"]["neutral_drift"] + 1
    ra = diff_summaries(base, ag)
    assert ra["decision"] == "DIFF", ra["decision"]
    assert ra["measured"]["n_value_mismatch"] == 0 and ra["measured"]["n_label_mismatch"] == 0
    assert ra["measured"]["n_derived_mismatch"] == 1, ra["measured"]
    assert ra["derived_mismatches"][0]["where"] == "cells.fold.neutral_drift", ra["derived_mismatches"]
    print("[selftest] stored aggregate moved with no label difference -> DIFF (not LABELS_ONLY_DIFF) OK")

    # ---------- NOT_COMPARABLE paths, each distinguishable from a genuine mismatch ----------
    # (a) different item sets: different counts
    rc1 = diff_summaries(base, _mk_summary(n_pairs=3, arm=True))
    assert rc1["decision"] == "NOT_COMPARABLE" and rc1["decision"] != "DIFF"
    assert any("item count differs" in s for s in rc1["not_comparable_reasons"]), rc1
    assert rc1["measured"]["frac_item_fields_identical"] is None, rc1["measured"]
    # (b) same count, different items at one index
    c2 = _copy(new)
    c2["items"][4]["q"] = "a-different-question"
    rc2 = diff_summaries(base, c2)
    assert rc2["decision"] == "NOT_COMPARABLE"
    assert any("identity key" in s for s in rc2["not_comparable_reasons"]), rc2
    # (c) a legacy field missing from the new summary
    c3 = _copy(new)
    del c3["items"][0]["neutral_gen"]
    rc3 = diff_summaries(base, c3)
    assert rc3["decision"] == "NOT_COMPARABLE"
    assert any("'neutral_gen'" in s and "MISSING" in s for s in rc3["not_comparable_reasons"]), rc3
    # (d) the new summary lacks the neutral-elicited arm entirely
    rc4 = diff_summaries(base, _mk_summary(arm=False))
    assert rc4["decision"] == "NOT_COMPARABLE" and rc4["new_arm_presence"]["verdict"] == "ARM_ABSENT"
    assert any("neutral-elicited fields" in s for s in rc4["not_comparable_reasons"]), rc4
    # (e) a different model / regime: the wrong pair of artifacts
    c5 = _copy(new)
    c5["name"] = "google/gemma-2-2b"
    rc5 = diff_summaries(base, c5)
    assert rc5["decision"] == "NOT_COMPARABLE"
    assert any("'name' differs" in s for s in rc5["not_comparable_reasons"]), rc5
    # (f) a top-level legacy block missing from the new summary
    c6 = _copy(new)
    del c6["cells_faithful"]
    rc6 = diff_summaries(base, c6)
    assert rc6["decision"] == "NOT_COMPARABLE"
    assert any("cells_faithful" in s and "ABSENT" in s for s in rc6["not_comparable_reasons"]), rc6
    # (g) a gate that cannot be computed at all (unknown cell on BOTH sides -> aggregate raises)
    c7b, c7n = _copy(base), _copy(new)
    c7b["items"][0]["cell"] = "bogus"
    c7n["items"][0]["cell"] = "bogus"
    rc7 = diff_summaries(c7b, c7n)
    assert rc7["decision"] == "NOT_COMPARABLE", rc7["decision"]
    assert any("could not be computed" in s for s in rc7["not_comparable_reasons"]), rc7
    assert all(g["status"] == "ERROR" for g in rc7["gates"]), rc7["gates"]
    print("[selftest] NOT_COMPARABLE: item count / item identity / missing legacy field / absent new arm / "
          "wrong model / missing block / uncomputable gate -- all distinct from DIFF OK")

    # ---------- ARM_PARTIAL is REPORTED, not decided on ----------
    pa = _copy(new)
    for k in NEW_ARM_ITEM_KEYS:
        del pa["items"][0][k]
    rp = diff_summaries(base, pa)
    assert rp["new_arm_presence"]["verdict"] == "ARM_PARTIAL", rp["new_arm_presence"]
    assert rp["decision"] == "BYTE_IDENTICAL", (rp["decision"], rp["not_comparable_reasons"])
    print("[selftest] ARM_PARTIAL reported while the legacy identity decision stands (BYTE_IDENTICAL) OK")

    # ---------- pre-port baseline: faithful labels via the faithful_rescore side-channel (§5.3) ----------
    pre = _mk_summary(arm=False, faithful=False)                   # no faithful_* fields, no cells_faithful
    post = _mk_summary(arm=True, faithful=True)                    # first native dual-label run
    r_nosc = diff_summaries(pre, post)
    assert r_nosc["decision"] == "BYTE_IDENTICAL", r_nosc["decision"]
    sc = _mk_sidechannel(post["items"])
    r_sc = diff_summaries(pre, post, sc)
    assert r_sc["decision"] == "BYTE_IDENTICAL", (r_sc["decision"], r_sc["not_comparable_reasons"])
    added = r_sc["measured"]["n_item_fields_compared"] - r_nosc["measured"]["n_item_fields_compared"]
    assert added == 6 * len(post["items"]), added                  # 3 labels + 3 rules per record
    assert any("cells_faithful" in s for s in r_sc["not_compared"]), r_sc["not_compared"]
    assert any("lacks that label family" in s for s in r_sc["not_compared"]), r_sc["not_compared"]
    # a side-channel label that disagrees with the re-run is a LABEL mismatch, not a value one
    sc_bad = _copy(sc)
    sc_bad["fields"]["elicit_gen"]["items"][2]["new_label"] = "C"
    r_bad = diff_summaries(pre, post, sc_bad)
    assert r_bad["decision"] == "LABELS_ONLY_DIFF", r_bad["decision"]
    assert r_bad["measured"]["mismatch_counts_by_key"] == {"faithful_elicit": 1}, r_bad["measured"]
    # an unalignable side-channel is NOT_COMPARABLE, never a silent skip and never a mismatch
    sc_mis = _copy(sc)
    sc_mis["fields"]["counter_gen"]["items"][0]["q"] = "not-the-same-question"
    r_mis = diff_summaries(pre, post, sc_mis)
    assert r_mis["decision"] == "NOT_COMPARABLE", r_mis["decision"]
    assert any("side-channel" in s for s in r_mis["not_comparable_reasons"]), r_mis
    r_short = diff_summaries(pre, post, {"fields": {"neutral_gen": {"items": []}}})
    assert r_short["decision"] == "NOT_COMPARABLE", r_short["decision"]
    print("[selftest] faithful side-channel: aligned -> BYTE_IDENTICAL (+%d label comparisons), "
          "disagreeing -> LABELS_ONLY_DIFF, unalignable/short -> NOT_COMPARABLE OK" % added)

    # ---------- the example dump is capped without capping the counts ----------
    many = _copy(new)
    for it in many["items"]:
        it["counter_gen"] = it["counter_gen"] + "!"
        it["neutral_gen"] = it["neutral_gen"] + "!"
        it["elicit_gen"] = it["elicit_gen"] + "!"
    rm = diff_summaries(base, many)
    n_expected = 3 * len(many["items"])
    assert rm["measured"]["n_value_mismatch"] == n_expected, rm["measured"]
    assert len(rm["examples"]) == min(MAX_EXAMPLES, n_expected), len(rm["examples"])
    assert rm["n_examples_omitted"] == max(0, n_expected - MAX_EXAMPLES), rm["n_examples_omitted"]
    assert rm["measured"]["frac_item_fields_identical"] < 1.0
    print("[selftest] example dump capped at %d with complete counts (%d mismatches, %d omitted) OK"
          % (MAX_EXAMPLES, n_expected, rm["n_examples_omitted"]))

    # ---------- the embedded rule/threshold block travels with the result ----------
    assert r["thresholds"]["max_field_mismatch"] == MAX_FIELD_MISMATCH == 0
    assert r["thresholds"]["float_eq_tol"] == FLOAT_EQ_TOL == 0.0
    assert r["decision_rule"] == DECISION_RULE and r["metric"] == METRIC
    assert set(NEW_ARM_ITEM_KEYS) == set(r["scope"]["excluded_item_keys"])
    print("[selftest] metric / decision_rule / thresholds / scope embedded in the result OK")

    print("[selftest] PASS")


def main():
    ap = argparse.ArgumentParser(description="offline reproduction diff of two foldlisten_judge summaries")
    ap.add_argument("--selftest", action="store_true", help="model-free pure-logic tests (CPU, no i/o)")
    ap.add_argument("--committed", help="committed (pre-change) foldlisten_judge_<tag>_summary.json")
    ap.add_argument("--new", help="re-run (post-change) foldlisten_judge_<tag>_summary.json")
    ap.add_argument("--faithful-committed", default=None,
                    help="faithful_rescore_<tag>.json supplying the BASELINE faithful labels when the "
                         "committed summary is pre-port and carries none (DESIGN_neutral_elicit.md §5.3)")
    ap.add_argument("--tag", default=None, help="output tag (default: derived from --new's filename)")
    ap.add_argument("--outdir", default="out", help="output directory for foldlisten_repro_diff_<tag>.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.committed and a.new):
        ap.error("nothing to do: pass --selftest, or --committed and --new")
    run(a.committed, a.new, tag=a.tag, outdir=a.outdir, faithful_committed=a.faithful_committed)


if __name__ == "__main__":
    main()
