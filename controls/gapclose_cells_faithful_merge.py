"""OFFLINE MERGE OF THE MISSING `cells_faithful` BLOCKS (model-free, CPU-only: no torch, no GPU, no
network; stdlib + two pure sibling imports).

WHAT IS MEASURED (neutral, claim-blind). Six committed foldlisten judge summaries carry per-item `commit_*`
labels but NO aggregate `cells_faithful` block, while the strict per-item FAITHFUL labels for exactly those
records already exist offline in out/faithful_rescore_fl_*.json. This control recomputes the aggregate block
from those persisted labels and writes it to a NEW file per cell. It re-generates nothing, re-labels nothing
(every label is read off disk), and MODIFIES NOTHING IN PLACE.

THE ARITHMETIC IS THE JUDGE'S OWN, NOT A NEW DEFINITION. `moved` / `held` / `abstain` / `neutral_drift` /
`neutral_elicit` / `n_neutral_elicit` are whatever these functions in controls/foldlisten_judge.py say they
are; they are imported and CALLED, never paraphrased:

    foldlisten_judge.interpret                 lines  72- 81   (cell -> moved / held / abstain)
    foldlisten_judge.aggregate                 lines  84-109   (the per-cell block itself)
    foldlisten_judge.FAITHFUL_TO_COMMIT        line  185       (WSTAR/C/NEITHER/UNRESOLVED_ALIAS map)
    foldlisten_judge._faithful_commit_records  lines 196-216   (faithful_* -> commit_* remap)

composed exactly as the judge's own call site, foldlisten_judge.py:517
    cells_faithful = aggregate(_faithful_commit_records(records))
so a merged block is defined identically to the blocks already committed in the ext2 summaries.

INPUTS. out/faithful_rescore_fl_*.json, each carrying `fields: {<gen_field>: {old_label_field,
confidence_mapping, aggregate, items: [{q, cell, correct, Wstar, field, answer_span, old_label, new_label,
rule_fired}]}}`. The field blocks are written by one pass each over the SAME item list
(faithful_rescore.py:642-661), so they are joined BY INDEX and the join is VERIFIED by (q, cell, correct,
Wstar) identity at every index; a disagreement is SHAPE_MISMATCH, never a silent re-order. Each artifact's
target summary is resolved from its own `input_path` field (re-rooted onto this checkout when the recorded
absolute path does not exist), NEVER from a filename, and supplies only `model` (its stamped `name`, else
null) and the record-count cross-check.

VALIDATION IN THE INSTRUMENT (not in prose), INDEPENDENT OF THE MERGE. Whenever an artifact's records match
an already-committed `cells_faithful` block item-for-item (same count, same join keys, same isolated answer
span on every scored arm), the recomputed block is compared to the committed one field by field, and the
matched summary's OWN per-item faithful labels are compared to the artifact's, per arm, so a difference is
localized to the labels or to the arithmetic rather than guessed at. Keys the artifact cannot determine are
reported with BOTH values under `keys_not_determined_by_rescore`. A disagreement is EMITTED, never fixed.
Merging is decided separately, by whether the artifact's OWN target summary lacks the block, so a matching
re-run can neither suppress a merge nor create one.

OUTPUT. One out/gapclose_cells_faithful_<tag>.json per merged cell (<tag> = the artifact's own `tag`) plus
the roll-up out/gapclose_cells_faithful_merge.json. Writes outside out/ are refused. Writes are ALL-OR-
NOTHING and, by default, REFUSE to supersede an existing output path in place (--if-exists refuse, exit 3
with the computed roll-up printed to stdout); pass --if-exists overwrite to regenerate deliberately.

  python controls/gapclose_cells_faithful_merge.py --selftest
  python controls/gapclose_cells_faithful_merge.py
  python controls/gapclose_cells_faithful_merge.py --if-exists overwrite
"""
import argparse
import json
import os
import sys
from pathlib import Path

_CONTROLS = Path(__file__).resolve().parent
_REPO_ROOT = _CONTROLS.parent
sys.path.insert(0, str(_CONTROLS))
sys.path.insert(0, str(_REPO_ROOT))

# Pure sibling imports: neither module imports torch at module top (foldlisten_judge keeps torch inside its
# _measure path), so this stays CPU-safe.
from faithful_rescore import LABELS, OLD_TO_NEW, isolate_span                            # noqa: E402
from foldlisten_judge import (CELLS, FAITHFUL_TO_COMMIT, aggregate as judge_aggregate,   # noqa: E402
                              _faithful_commit_records)

OUT_ROOT = _REPO_ROOT / "out"
RESCORE_GLOB = "faithful_rescore_fl_*.json"
SUMMARY_GLOB = "results_foldlisten*/out/foldlisten_judge_*summary.json"
ROLLUP_NAME = "gapclose_cells_faithful_merge.json"
MAX_EXAMPLES = 5

# rescore generation field -> the judge's record arm; and which aggregate() block keys that arm determines.
FIELD_TO_ARM = {"neutral_gen": "neutral", "counter_gen": "counter", "elicit_gen": "elicit",
                "neutral_elicit_gen": "neutral_elicit"}
ARM_TO_FIELD = {a: f for f, a in FIELD_TO_ARM.items()}
ARM_BLOCK_KEYS = {"neutral": ("neutral_drift",), "counter": ("counter",), "elicit": ("elicit",),
                  "neutral_elicit": ("neutral_elicit", "n_neutral_elicit")}
REQUIRED_ARMS = ("neutral", "counter", "elicit")   # _faithful_commit_records demands all three per record
JOIN_KEYS = ("q", "cell", "correct", "Wstar")

THRESHOLDS = {
    "LEGACY_N_RECORDS": 44,      # per-field record count identifying a legacy (22 items x 2 cells) artifact
    "MAX_DIFFERING_FIELDS": 0,   # against-committed check: any differing comparable block field -> DIFFERS
}

METRIC = (
    "Offline merge (no model, no re-labelling): per out/faithful_rescore_fl_*.json artifact, join its "
    "per-field item records BY INDEX (join verified by (q, cell, correct, Wstar) identity at every index) "
    "into the judge's per-item record shape carrying faithful_<arm> = that field's new_label, then "
    "recompute the aggregate block by CALLING the judge's own functions: "
    "foldlisten_judge.aggregate(foldlisten_judge._faithful_commit_records(records)) -- the composition at "
    "foldlisten_judge.py:517, whose moved/held/abstain come from foldlisten_judge.interpret (72-81), whose "
    "neutral_drift is the neutral arm's 'moved' count, and whose neutral_elicit/n_neutral_elicit stay 0 "
    "when the artifact supplies no 4th-arm labels (the judge's own ABSENT-arm convention, never a faked "
    "all-held arm). Labels are faithful_rescore.LABELS {C, WSTAR, NEITHER, UNRESOLVED_ALIAS} mapped by "
    "foldlisten_judge.FAITHFUL_TO_COMMIT; each field's own confidence_mapping value (the elicited slot is "
    "scored strict) is carried into that slot's stamp. Against-committed checks additionally report "
    "label_disagreements: per arm, the number of records whose matched summary's OWN per-item faithful "
    "label differs from the rescore artifact's label for the same record (null = that summary carries no "
    "such field). 0 everywhere means both blocks were computed from identical labels; non-zero means the "
    "label inputs themselves differ. Nothing is modified in place: one NEW file per cell plus a roll-up, "
    "both under out/."
)

DECISION_RULE = (
    "Per cell (one rescore artifact whose own target summary lacks a cells_faithful block and whose "
    "per-field record count == LEGACY_N_RECORDS(44)): MERGED iff all three required arms (neutral_gen, "
    "counter_gen, elicit_gen) carry item records, the per-field record counts are equal to each other and "
    "to the target summary's item count, the index join agrees on (q, cell, correct, Wstar) at every index, "
    "every cell is in {fold, listen} and every new_label is in LABELS; LABELS_ABSENT if a required arm or "
    "its items are missing/empty; SHAPE_MISMATCH otherwise. An artifact whose target already carries the "
    "block, or whose record count is not 44, is not a merge cell and is listed under `skipped` with the "
    "reason. Per against-committed check: DIFFERS iff more than MAX_DIFFERING_FIELDS(0) comparable block "
    "fields differ, else MATCHES. Top level, in precedence order: MERGE_RULE_DISAGREES_WITH_COMMITTED if "
    "any check DIFFERS; else COMMITTED_CHECK_UNAVAILABLE if no committed block could be matched "
    "item-for-item (an unvalidated merge must not read as ALL_MERGED); else ALL_MERGED iff at least one "
    "cell was merged and every cell's decision is MERGED; else INCOMPLETE. Counts only; no claim is "
    "attached to any cell, label or outcome, and no outcome is a success state of this instrument."
)


# --------------------------------------------------------------------------- write guard
def _safe_out(path, is_dir=False):
    """Resolve `path` (cwd-relative allowed) and REFUSE anything that is not out/ or inside it. normpath,
    not resolve(), so a '..' escape is caught rather than followed. (str|Path -> Path; raises ValueError)."""
    p = Path(path)
    p = Path(os.path.normpath(str(p if p.is_absolute() else Path.cwd() / p)))
    root = Path(os.path.normpath(str(OUT_ROOT)))
    if not ((is_dir and p == root) or root in p.parents):
        raise ValueError("refusing to write outside %s: %r" % (root.as_posix(), str(path)))
    return p


# --------------------------------------------------------------------------- pure readers
def items_of(data):
    """Per-item list from either on-disk shape (mirrors faithful_rescore._load_items). Pure."""
    if isinstance(data.get("items"), list):
        return data["items"]
    res = data.get("result")
    if isinstance(res, dict) and isinstance(res.get("items"), list):
        return res["items"]
    return []


def model_of(data):
    """The summary's STAMPED model name (top level, or under 'result'), else None. NEVER inferred from a
    filename. Pure (dict|None -> str|None)."""
    if not isinstance(data, dict):
        return None
    if data.get("name") is not None:
        return data["name"]
    res = data.get("result")
    return res.get("name") if isinstance(res, dict) else None


def resolve_summary(input_path):
    """The target summary from the artifact's OWN `input_path`. The artifacts record an absolute path from
    the box they were written on, so when that path does not exist it is re-rooted onto this checkout at its
    'results_*' component. Pure (str|None -> (Path|None, str))."""
    if not input_path:
        return None, "no_input_path"
    p = Path(str(input_path))
    if p.exists():
        return p, "as_recorded"
    parts = p.parts
    for i, part in enumerate(parts):
        if part.startswith("results_"):
            q = _REPO_ROOT.joinpath(*parts[i:])
            if q.exists():
                return q, "rerooted_at_%s" % part
    return None, "unresolved"


# --------------------------------------------------------------------------- join + merge (pure)
def join_records(fields):
    """The artifact's per-field item lists -> the judge's per-item record shape. Returns
    (records, arms, problems, n_seen); each record carries the join keys plus faithful_<arm> = new_label,
    rule_<arm> = rule_fired and span_<arm> = answer_span. Joined BY INDEX (faithful_rescore writes one pass
    per field over the same item list) with the join VERIFIED at every index. Pure."""
    fields = fields or {}
    problems, present = [], [f for f in fields if f in FIELD_TO_ARM]
    for arm in REQUIRED_ARMS:
        f = ARM_TO_FIELD[arm]
        if f not in fields or not (fields[f] or {}).get("items"):
            problems.append("LABELS_ABSENT: field %r missing or carries no items" % f)
    lengths = {f: len((fields[f] or {}).get("items") or []) for f in present}
    n_seen = max(lengths.values()) if lengths else 0
    if problems:
        return [], [], problems, n_seen
    if len(set(lengths.values())) != 1:
        return [], [], ["SHAPE_MISMATCH: per-field record counts differ: %s" % lengths], n_seen
    arms = [FIELD_TO_ARM[f] for f in sorted(present)]
    records = []
    for i in range(n_seen):
        rows = [(a, fields[ARM_TO_FIELD[a]]["items"][i]) for a in arms]
        rec = {k: rows[0][1].get(k) for k in JOIN_KEYS}
        for a, row in rows:
            if any(row.get(k) != rec[k] for k in JOIN_KEYS):
                problems.append("SHAPE_MISMATCH: join key disagreement at index %d between %r and %r"
                                % (i, ARM_TO_FIELD[arms[0]], ARM_TO_FIELD[a]))
            elif row.get("new_label") not in LABELS:
                problems.append("SHAPE_MISMATCH: index %d field %r new_label %r not in LABELS"
                                % (i, ARM_TO_FIELD[a], row.get("new_label")))
            if problems:
                return [], arms, problems, n_seen
            rec["faithful_%s" % a] = row["new_label"]
            rec["rule_%s" % a] = row.get("rule_fired")
            rec["span_%s" % a] = row.get("answer_span")
        if rec["cell"] not in CELLS:
            return [], arms, ["SHAPE_MISMATCH: index %d cell %r not in %s"
                              % (i, rec["cell"], list(CELLS))], n_seen
        if "neutral_elicit" in arms:
            # presence gate only: _faithful_commit_records (foldlisten_judge.py:210-214) overwrites this
            # from faithful_neutral_elicit. Set so a 4th-arm artifact is NOT silently dropped.
            rec["commit_neutral_elicit"] = None
        records.append(rec)
    return records, arms, problems, n_seen


def merged_block(records):
    """The judge's own composition, foldlisten_judge.py:517. Pure (list -> dict)."""
    return judge_aggregate(_faithful_commit_records(records))


def stamps_for(records, fields, arms):
    """One stamp per (cell, slot), all five keys. `map_confidence` is that field's OWN confidence_mapping
    value read off the artifact (it differs by field: the elicited slot is scored strict). `tiebreak` counts
    the rule_fired values beginning with 'tiebreak' in that (cell, slot) group ({} = none fired). Pure."""
    out = []
    for cell in CELLS:
        for arm in arms:
            f, tb = ARM_TO_FIELD[arm], {}
            for r in records:
                rule = r.get("rule_%s" % arm) or ""
                if r["cell"] == cell and rule.startswith("tiebreak"):
                    tb[rule] = tb.get(rule, 0) + 1
            out.append({"arm": cell, "slot": f, "labels": list(LABELS),
                        "map_confidence": (fields.get(f) or {}).get("confidence_mapping"), "tiebreak": tb})
    return out


# --------------------------------------------------------------------------- against-committed check (pure)
def slim(data):
    """A summary reduced to what the check needs (the committed block, the join keys, the scored generations
    and the summary's OWN per-item faithful labels), so the committed index never holds the prompt fields.
    Pure (dict -> dict)."""
    keep = JOIN_KEYS + tuple(FIELD_TO_ARM) + tuple("faithful_%s" % a for a in FIELD_TO_ARM.values())
    return {"cells_faithful": data.get("cells_faithful"),
            "items": [{k: it.get(k) for k in keep if k in it} for it in items_of(data)]}


def committed_matches(records, arms, data):
    """True iff `data` carries a cells_faithful block AND its items match `records` item-for-item: same
    count, same (q, cell, correct, Wstar) at every index, and the same ISOLATED answer span on every arm the
    artifact scored (isolate_span is deterministic, so equal spans mean equal scored generations, which is
    what a span-based label set is a function of). Pure (list, list, dict -> bool)."""
    if not isinstance(data.get("cells_faithful"), dict) or not records:
        return False
    items = items_of(data)
    if len(items) != len(records):
        return False
    for rec, it in zip(records, items):
        if any(it.get(k) != rec[k] for k in JOIN_KEYS):
            return False
        for a in arms:
            if isolate_span(it.get(ARM_TO_FIELD[a], "")) != rec["span_%s" % a]:
                return False
    return True


def label_diffs(records, arms, items):
    """Per arm, the number of records whose matched summary's OWN per-item faithful label differs from the
    rescore artifact's label for the same record, plus up to MAX_EXAMPLES examples. An arm the summary does
    not carry gets None (ABSENT), never 0. Pure (list, list, list -> (dict, list))."""
    counts, ex = {}, []
    for a in arms:
        key = "faithful_%s" % a
        if not items or not all(key in it for it in items):
            counts[a] = None
            continue
        n = 0
        for rec, it in zip(records, items):
            if it[key] != rec[key]:
                n += 1
                if len(ex) < MAX_EXAMPLES:
                    ex.append({"q": rec["q"], "cell": rec["cell"], "arm": a,
                               "rescore_label": rec[key], "committed_label": it[key]})
        counts[a] = n
    return counts, ex


def check_against_committed(tag, source, recomputed, data, records, arms):
    """Compare the recomputed block to a matched committed one field by field, over the block keys the
    rescore artifact supplies labels for ('n' always, plus ARM_BLOCK_KEYS per scored arm). Keys it does not
    determine are reported with BOTH values instead of being dropped, and label_diffs localizes any
    difference to the labels or the arithmetic. NO hard assert: a disagreement must survive as an artifact,
    not as a traceback, and it is never repaired here. Pure -> dict."""
    committed = data.get("cells_faithful") or {}
    comparable = {"n"}
    for a in arms:
        comparable.update(ARM_BLOCK_KEYS[a])
    diff, undet = [], []
    for cell in CELLS:
        rc, cc = recomputed.get(cell) or {}, committed.get(cell) or {}
        for key in sorted(set(rc) | set(cc)):
            if rc.get(key) == cc.get(key):
                continue
            entry = {"cell": cell, "key": key, "recomputed": rc.get(key), "committed": cc.get(key)}
            (diff if key in comparable else undet).append(entry)
    counts, ex = label_diffs(records, arms, items_of(data))
    return {"tag": tag, "committed_summary": source, "arms_compared": list(arms),
            "comparable_keys": sorted(comparable), "n_differing_fields": len(diff),
            "differing_fields": diff, "keys_not_determined_by_rescore": undet,
            "label_disagreements": counts, "label_disagreement_examples": ex,
            "decision": "MATCHES" if len(diff) <= THRESHOLDS["MAX_DIFFERING_FIELDS"] else "DIFFERS"}


def decide_top(cells, checks):
    """The frozen top-level decision (see DECISION_RULE). Pure (list, list -> str)."""
    if any(c["decision"] == "DIFFERS" for c in checks):
        return "MERGE_RULE_DISAGREES_WITH_COMMITTED"
    if not checks:
        return "COMMITTED_CHECK_UNAVAILABLE"
    if cells and all(c["decision"] == "MERGED" for c in cells):
        return "ALL_MERGED"
    return "INCOMPLETE"


# --------------------------------------------------------------------------- run (reads JSON only)
def cell_output(tag, art_path, art, sum_path, how, model, block, records, arms, decision, n_summary,
                problems):
    """The per-cell artifact. Pure (-> dict)."""
    return {"control": "gapclose_cells_faithful_merge", "metric": METRIC,
            "thresholds": dict(THRESHOLDS), "decision_rule": DECISION_RULE, "decision": decision,
            "source_rescore": art_path, "source_summary": art.get("input_path"),
            "source_summary_resolved": (sum_path.as_posix() if sum_path else None),
            "source_summary_resolution": how, "model": model,
            "label_maps": {"labels": list(LABELS), "old_to_new": dict(OLD_TO_NEW),
                           "faithful_to_commit": dict(FAITHFUL_TO_COMMIT)},
            "arms_supplied": list(arms), "cells_faithful": block, "n_records_used": len(records),
            "n_records_summary": n_summary, "problems": problems,
            "stamps": stamps_for(records, art.get("fields") or {}, arms) if block else []}


def committed_index():
    """Every committed summary that carries a cells_faithful block, slimmed. Reads JSON only."""
    idx = []
    for p in sorted(_REPO_ROOT.glob(SUMMARY_GLOB)):
        d = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(d.get("cells_faithful"), dict):
            idx.append((p.relative_to(_REPO_ROOT).as_posix(), slim(d)))
    return idx


def run(outdir, if_exists="refuse"):
    outdir = _safe_out(outdir, is_dir=True)
    idx, pending, cells, checks, skipped = None, [], [], [], []
    for ap in sorted(OUT_ROOT.glob(RESCORE_GLOB)):
        art = json.loads(ap.read_text(encoding="utf-8"))
        tag, apx = art.get("tag") or ap.stem, ap.relative_to(_REPO_ROOT).as_posix()
        records, arms, problems, n_seen = join_records(art.get("fields"))
        sum_path, how = resolve_summary(art.get("input_path"))
        sdata = json.loads(sum_path.read_text(encoding="utf-8")) if sum_path else None
        n_summary = len(items_of(sdata)) if sdata is not None else None
        block = merged_block(records) if records else None
        # (1) validation, run whenever a committed block covers these exact records -- independent of
        #     whether this artifact is a merge gap.
        if block is not None:
            idx = committed_index() if idx is None else idx
            hit = next(((n, d) for n, d in idx if committed_matches(records, arms, d)), None)
            if hit:
                checks.append(check_against_committed(tag, hit[0], block, hit[1], records, arms))
        # (2) merge, decided ONLY by whether this artifact's own target lacks the block.
        if sdata is not None and isinstance(sdata.get("cells_faithful"), dict):
            skipped.append("%s: target %s already carries cells_faithful -> not a merge gap"
                           % (tag, sum_path.as_posix()))
            continue
        if n_seen != THRESHOLDS["LEGACY_N_RECORDS"]:
            skipped.append("%s: n_records=%s != LEGACY_N_RECORDS(%s) -> not a legacy merge cell"
                           % (tag, n_seen, THRESHOLDS["LEGACY_N_RECORDS"]))
            continue
        if problems:
            decision = problems[0].split(":")[0]
        elif n_summary is not None and n_summary != len(records):
            decision, block = "SHAPE_MISMATCH", None
            problems = ["SHAPE_MISMATCH: summary carries %s items, artifact %d" % (n_summary, len(records))]
        else:
            decision = "MERGED"
        out = cell_output(tag, apx, art, sum_path, how, model_of(sdata), block, records, arms, decision,
                          n_summary, problems)
        p = _safe_out(outdir / ("gapclose_cells_faithful_%s.json" % tag))
        pending.append((p, out))
        cells.append({"tag": tag, "decision": decision, "source_rescore": apx,
                      "source_summary": art.get("input_path"), "model": out["model"],
                      "n_records_used": out["n_records_used"], "problems": problems,
                      "out_path": p.as_posix()})
        print("[%s] %s n_records=%d model=%s target=%s"
              % (tag, decision, len(records), out["model"], p.as_posix()), flush=True)
    roll = {"control": "gapclose_cells_faithful_merge", "metric": METRIC, "thresholds": dict(THRESHOLDS),
            "decision_rule": DECISION_RULE, "decision": decide_top(cells, checks), "n_cells": len(cells),
            "cells": cells, "committed_checks": checks, "skipped": skipped}
    pending.append((_safe_out(outdir / ROLLUP_NAME), roll))
    for c in checks:
        print("[against-committed %s vs %s] %s (%d differing field(s), %d key(s) not determined, "
              "label_disagreements=%s)" % (c["tag"], c["committed_summary"], c["decision"],
                                           c["n_differing_fields"],
                                           len(c["keys_not_determined_by_rescore"]),
                                           c["label_disagreements"]), flush=True)
        for d in c["differing_fields"]:
            print("    DIFF %s.%s recomputed=%s committed=%s"
                  % (d["cell"], d["key"], d["recomputed"], d["committed"]), flush=True)
    for s in skipped:
        print("[skip] %s" % s, flush=True)
    print("[decision] %s over %d merged cell(s), %d check(s)"
          % (roll["decision"], len(cells), len(checks)), flush=True)
    # ALL-OR-NOTHING write, and by default no existing artifact is superseded in place.
    existing = [p.as_posix() for p, _ in pending if p.exists()]
    roll["refused_paths"] = existing if if_exists == "refuse" else []
    roll["written"] = not (existing and if_exists == "refuse")
    if not roll["written"]:
        print(json.dumps(roll, indent=2, default=str), flush=True)
        print("[refused] %d output path(s) already exist; NOTHING was written (the roll-up above is the "
              "computed result). Move them aside or pass --if-exists overwrite: %s"
              % (len(existing), ", ".join(existing)), flush=True)
        return roll
    outdir.mkdir(parents=True, exist_ok=True)
    for p, obj in pending:
        p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        print("[written] %s" % p.as_posix(), flush=True)
    return roll


# --------------------------------------------------------------------------- selftest (model-free, no i/o)
def _row(cell, label, rule="r", span="s"):
    return {"q": "q_%s" % cell, "cell": cell, "correct": "Nile", "Wstar": "Amazon", "answer_span": span,
            "field": "f", "old_label": None, "new_label": label, "rule_fired": rule}


def _fields(neutral, counter, elicit, cells, extra=None):
    """A synthetic rescore-artifact `fields` block: three label lists + the per-item cell. The first
    counter_gen record carries a tiebreak rule, so the stamp's tiebreak census has something to count."""
    f = {}
    for name, labs, mc in (("neutral_gen", neutral, True), ("counter_gen", counter, True),
                           ("elicit_gen", elicit, False)):
        f[name] = {"old_label_field": "commit_x", "confidence_mapping": mc,
                   "items": [_row(c, l, "tiebreak_but_C" if (name == "counter_gen" and i == 0) else "r")
                             for i, (c, l) in enumerate(zip(cells, labs))]}
    if extra:
        f.update(extra)
    return f


def selftest():
    import copy
    # ---- synthetic cells: fold moved + fold UNRESOLVED_ALIAS (abstain); listen moved + listen held ----
    cells = ["fold", "fold", "listen", "listen"]
    fl = _fields(neutral=["C", "WSTAR", "WSTAR", "C"],                    # fold drift 1, listen drift 1
                 counter=["WSTAR", "NEITHER", "C", "NEITHER"],
                 elicit=["WSTAR", "UNRESOLVED_ALIAS", "C", "WSTAR"], cells=cells)
    recs, arms, probs, n_seen = join_records(fl)
    assert probs == [] and n_seen == 4 and len(recs) == 4, (probs, n_seen)
    assert arms == ["counter", "elicit", "neutral"], arms
    b = merged_block(recs)
    assert b["fold"]["n"] == 2 and b["listen"]["n"] == 2, b
    assert b["fold"]["elicit"] == {"moved": 1, "held": 0, "abstain": 1}, b["fold"]        # WSTAR / UA
    assert b["listen"]["elicit"] == {"moved": 1, "held": 1, "abstain": 0}, b["listen"]    # C / WSTAR
    assert b["fold"]["counter"] == {"moved": 1, "held": 0, "abstain": 1}, b["fold"]
    assert b["listen"]["counter"] == {"moved": 1, "held": 0, "abstain": 1}, b["listen"]
    assert b["fold"]["neutral_drift"] == 1 and b["listen"]["neutral_drift"] == 1, b
    for c in CELLS:                        # the judge's ABSENT-arm convention, not a faked all-held arm
        assert b[c]["n_neutral_elicit"] == 0, b[c]
        assert b[c]["neutral_elicit"] == {"moved": 0, "held": 0, "abstain": 0}, b[c]
        for slot in ("elicit", "counter"):
            s = b[c][slot]
            assert s["moved"] + s["held"] + s["abstain"] == b[c]["n"], (c, slot, b[c])
    print("[selftest] merged counts: fold moved/held/abstain 1/0/1 (UA -> abstain), listen 1/1/0, "
          "drift 1/1, 4th arm absent (n_neutral_elicit=0) OK")

    # ---- stamps: one per (cell, slot), all five keys, per-field map_confidence, tiebreak counted ----
    st = stamps_for(recs, fl, arms)
    assert len(st) == len(CELLS) * 3, st
    for s in st:
        assert set(s) == {"arm", "slot", "labels", "map_confidence", "tiebreak"}, s
        assert s["arm"] in CELLS and s["slot"] in FIELD_TO_ARM and s["labels"] == list(LABELS), s
    mc = {(s["arm"], s["slot"]): s["map_confidence"] for s in st}
    assert mc[("fold", "elicit_gen")] is False, mc                     # STRICT_FIELDS: elicited slot
    assert mc[("fold", "neutral_gen")] is True and mc[("listen", "counter_gen")] is True, mc
    tb = {(s["arm"], s["slot"]): s["tiebreak"] for s in st}
    assert tb[("fold", "counter_gen")] == {"tiebreak_but_C": 1}, tb
    assert tb[("listen", "counter_gen")] == {} and tb[("fold", "elicit_gen")] == {}, tb
    print("[selftest] stamps: %d entries, five keys each, map_confidence per field (elicit strict), "
          "tiebreak census OK" % len(st))

    # ---- LABELS_ABSENT / SHAPE_MISMATCH ----
    miss = dict(fl)
    miss.pop("elicit_gen")
    p1 = join_records(miss)[2]
    assert p1 and p1[0].startswith("LABELS_ABSENT") and "elicit_gen" in p1[0], p1
    assert join_records({})[2][0].startswith("LABELS_ABSENT"), join_records({})[2]
    short = _fields(["C"] * 3, ["C"] * 3, ["C"] * 3, cells[:3])
    short["elicit_gen"]["items"] = short["elicit_gen"]["items"][:2]
    _, _, p2, n2 = join_records(short)
    assert p2 and p2[0].startswith("SHAPE_MISMATCH") and n2 == 3, (p2, n2)
    swap = _fields(["C", "C"], ["C", "C"], ["C", "C"], ["fold", "listen"])
    swap["elicit_gen"]["items"] = list(reversed(swap["elicit_gen"]["items"]))
    p3 = join_records(swap)[2]
    assert p3 and "join key" in p3[0], p3
    p4 = join_records(_fields(["C", "C"], ["C", "C"], ["C", "MOVED"], ["fold", "listen"]))[2]
    assert p4 and "not in LABELS" in p4[0], p4
    p5 = join_records(_fields(["C", "C"], ["C", "C"], ["C", "C"], ["fold", "sideways"]))[2]
    assert p5 and "'sideways' not in" in p5[0], p5
    print("[selftest] LABELS_ABSENT (missing arm) / SHAPE_MISMATCH (count, join key, label, cell) OK")

    # ---- the 4th arm, when an artifact supplies it: counted, never silently dropped ----
    ne = _fields(["C", "C"], ["C", "C"], ["C", "C"], ["fold", "listen"],
                 extra={"neutral_elicit_gen": {"confidence_mapping": False,
                                               "items": [_row("fold", "C"), _row("listen", "C")]}})
    r4, a4, p6, _ = join_records(ne)
    assert p6 == [] and "neutral_elicit" in a4, (p6, a4)
    b4 = merged_block(r4)
    assert b4["fold"]["n_neutral_elicit"] == 1 and b4["listen"]["n_neutral_elicit"] == 1, b4
    assert b4["fold"]["neutral_elicit"] == {"moved": 0, "held": 1, "abstain": 0}, b4["fold"]
    print("[selftest] 4th arm supplied -> counted (n_neutral_elicit=1 per cell), never dropped OK")

    # ---- against-committed: equal -> MATCHES; one count off -> DIFFERS naming the field ----
    cit = [{k: r[k] for k in ("q", "cell", "faithful_neutral", "faithful_counter", "faithful_elicit")}
           for r in recs]
    chk = check_against_committed("t", "s.json", b, {"cells_faithful": copy.deepcopy(b), "items": cit},
                                  recs, arms)
    assert chk["decision"] == "MATCHES" and chk["differing_fields"] == [], chk
    assert chk["comparable_keys"] == ["counter", "elicit", "n", "neutral_drift"], chk["comparable_keys"]
    assert chk["label_disagreements"] == {"counter": 0, "elicit": 0, "neutral": 0}, chk
    off = copy.deepcopy(b)
    off["fold"]["elicit"]["moved"] += 1
    d = check_against_committed("t", "s.json", b, {"cells_faithful": off, "items": cit}, recs, arms)
    assert d["decision"] == "DIFFERS" and d["n_differing_fields"] == 1, d
    assert d["differing_fields"][0]["cell"] == "fold" and d["differing_fields"][0]["key"] == "elicit", d
    assert d["label_disagreements"] == {"counter": 0, "elicit": 0, "neutral": 0}, d   # arithmetic, not labels
    # label drift: identical block, one committed per-item label different -> localized to the labels
    cit2 = copy.deepcopy(cit)
    cit2[0]["faithful_counter"] = "NEITHER"
    ld = check_against_committed("t", "s.json", b, {"cells_faithful": copy.deepcopy(b), "items": cit2},
                                 recs, arms)
    assert ld["label_disagreements"]["counter"] == 1 and ld["label_disagreements"]["elicit"] == 0, ld
    assert ld["label_disagreement_examples"][0]["arm"] == "counter", ld["label_disagreement_examples"]
    assert ld["label_disagreement_examples"][0]["committed_label"] == "NEITHER", ld
    cit3 = [{k: v for k, v in it.items() if k != "faithful_neutral"} for it in cit]
    na = check_against_committed("t", "s.json", b, {"cells_faithful": copy.deepcopy(b), "items": cit3},
                                 recs, arms)
    assert na["label_disagreements"]["neutral"] is None, na["label_disagreements"]   # ABSENT, never 0
    rich = copy.deepcopy(b)                 # a committed 4th arm this artifact cannot determine
    for c in CELLS:
        rich[c].update(neutral_elicit={"moved": 0, "held": 2, "abstain": 0}, n_neutral_elicit=2)
    u = check_against_committed("t", "s.json", b, {"cells_faithful": rich, "items": cit}, recs, arms)
    assert u["decision"] == "MATCHES" and len(u["keys_not_determined_by_rescore"]) == 4, u
    assert {e["key"] for e in u["keys_not_determined_by_rescore"]} == {"neutral_elicit",
                                                                      "n_neutral_elicit"}, u
    print("[selftest] against-committed: equal -> MATCHES; +1 -> DIFFERS (fold.elicit); label drift "
          "localized (counter=1); absent arm -> None; undetermined 4th arm reported not compared OK")

    # ---- committed_matches: needs the block, the count, the join keys AND the isolated spans ----
    it = [dict(_row(c, "C"), neutral_gen="x", counter_gen="y", elicit_gen="z", elicit_prompt="P",
               faithful_counter="C") for c in ("fold", "listen")]
    recs2, arms2 = join_records(_fields(["C"] * 2, ["C"] * 2, ["C"] * 2, ["fold", "listen"]))[:2]
    for r in recs2:
        r["span_neutral"], r["span_counter"], r["span_elicit"] = "x", "y", "z"
    assert committed_matches(recs2, arms2, {"cells_faithful": {}, "items": it}) is True
    assert committed_matches(recs2, arms2, {"items": it}) is False                     # no committed block
    assert committed_matches(recs2, arms2, {"cells_faithful": {}, "items": it[:1]}) is False
    for mutate in ({"elicit_gen": "different text"}, {"q": "other question"}):
        bad = copy.deepcopy(it)
        bad[1].update(mutate)
        assert committed_matches(recs2, arms2, {"cells_faithful": {}, "items": bad}) is False, mutate
    sl = slim({"cells_faithful": {"fold": {}}, "items": it})
    assert sl["cells_faithful"] == {"fold": {}} and "elicit_prompt" not in sl["items"][0], sl
    assert sl["items"][0]["elicit_gen"] == "z" and sl["items"][0]["q"] == "q_fold", sl
    assert sl["items"][0]["faithful_counter"] == "C", sl                     # labels kept for label_diffs
    assert committed_matches(recs2, arms2, sl) is True                       # slimming is lossless here
    print("[selftest] committed_matches needs block + count + join keys + identical isolated spans; "
          "slim() drops prompts, keeps labels, still matches OK")

    # ---- model + input_path resolution: stamped or null, never from a filename ----
    assert model_of({"name": "google/gemma-2-2b-it"}) == "google/gemma-2-2b-it"
    assert model_of({"result": {"name": "google/gemma-2-9b"}}) == "google/gemma-2-9b"
    assert model_of({"items": []}) is None and model_of(None) is None
    assert resolve_summary(None)[1] == "no_input_path"
    assert resolve_summary("/nowhere/at/all/x_gemma-2-27b-it_summary.json") == (None, "unresolved")
    print("[selftest] model from the stamp or null; unresolvable input_path -> (None, 'unresolved') OK")

    # ---- the write guard: nothing outside out/ ----
    assert _safe_out(OUT_ROOT / "x.json").name == "x.json"
    assert _safe_out(OUT_ROOT / "sub" / "x.json").name == "x.json"
    assert _safe_out(OUT_ROOT, is_dir=True) == Path(os.path.normpath(str(OUT_ROOT)))
    for bad_path in ("/tmp/x.json", str(OUT_ROOT / ".." / "x.json"), str(OUT_ROOT),
                     str(_REPO_ROOT / "results_foldlisten" / "out" / "y_summary.json"),
                     str(_REPO_ROOT / "outside" / "x.json")):
        try:
            _safe_out(bad_path)
            assert False, "must refuse %r" % bad_path
        except ValueError as e:
            assert "refusing to write outside" in str(e), e
    print("[selftest] write guard refuses /tmp, '..' escapes, results_* and out/ itself OK")

    # ---- the frozen top-level decision ----
    m = [{"decision": "MERGED"}]
    assert decide_top(m, [{"decision": "MATCHES"}]) == "ALL_MERGED"
    assert decide_top(m, [{"decision": "DIFFERS"}]) == "MERGE_RULE_DISAGREES_WITH_COMMITTED"
    assert decide_top(m, [{"decision": "MATCHES"}, {"decision": "DIFFERS"}]) == \
        "MERGE_RULE_DISAGREES_WITH_COMMITTED"
    assert decide_top(m, []) == "COMMITTED_CHECK_UNAVAILABLE"
    assert decide_top([], [{"decision": "MATCHES"}]) == "INCOMPLETE"
    for bad in ("SHAPE_MISMATCH", "LABELS_ABSENT"):
        assert decide_top(m + [{"decision": bad}], [{"decision": "MATCHES"}]) == "INCOMPLETE", bad
    print("[selftest] decision: ALL_MERGED / DISAGREES / CHECK_UNAVAILABLE / INCOMPLETE OK")

    # ---- the per-cell artifact: required keys, the rule travels with it, it serializes ----
    out = cell_output("t", "a.json", {"input_path": "/x/results_z/out/s.json", "fields": fl}, None,
                      "unresolved", None, b, recs, arms, "MERGED", 4, [])
    for k in ("control", "metric", "thresholds", "decision_rule", "source_rescore", "source_summary",
              "model", "cells_faithful", "n_records_used", "stamps"):
        assert k in out, k
    assert out["source_summary"] == "/x/results_z/out/s.json" and out["model"] is None
    assert out["n_records_used"] == 4 and out["thresholds"] == THRESHOLDS and out["metric"] == METRIC
    assert out["label_maps"]["faithful_to_commit"] == dict(FAITHFUL_TO_COMMIT)
    assert out["label_maps"]["old_to_new"] == dict(OLD_TO_NEW)
    assert len(out["stamps"]) == 6 and out["decision_rule"] == DECISION_RULE
    json.dumps(out, default=str)
    print("[selftest] per-cell artifact shape + embedded metric/thresholds/decision_rule/label maps OK")

    print("SELFTEST PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="offline merge of the missing cells_faithful blocks")
    ap.add_argument("--selftest", action="store_true", help="model-free tests (no result file is read)")
    ap.add_argument("--outdir", default=str(OUT_ROOT), help="output directory (must be out/ or inside it)")
    ap.add_argument("--if-exists", default="refuse", choices=["refuse", "overwrite"],
                    help="refuse (default): if ANY output path already exists, write NOTHING and exit 3 "
                         "after printing the computed roll-up, so no committed artifact is superseded in "
                         "place; overwrite: regenerate the paths deliberately")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        r = run(a.outdir, a.if_exists)
        if not r["written"]:
            raise SystemExit(3)
