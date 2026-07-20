"""OFFLINE CLASSIFIER-vs-HUMAN agreement check (no model, no GPU, no network, CPU-only).

WHAT THIS MEASURES (neutral, claim-blind). It re-reads PERSISTED elicited-final answers from the committed
fold/listen summaries and, for each item that carries a HUMAN hand-label, re-labels the elicited final with
the rule-based classifier `classify(gen, correct, wstar, stated, pushed)` (IMPORTED verbatim from
controls/faithful_rescore.py; CPU-only, no model), maps the classifier label into the human label vocabulary,
and counts how often the two agree. It does NOT re-run any model, load weights, or touch the network; it reads
text already on disk and lets the agreement fraction fall out.

THE HAND-LABELS live in results_foldlisten_ext/handlabel_fold_finals.json as
  labels[run][idx] = one of {"correct", "wrong", "other"}
where `run` in {"repro", "ext"} and `idx` (a string) is the POSITIONAL index of the item in that run's
summary items[] list. Each run's finals were produced by the run summarized in the matching summary JSON
(RUN_SUMMARIES below; cross-checked against the artifact's own `source_summaries`). The join key is the item's
positional index into items[]; the classifier is run on that item's `elicit_gen` with its `correct` / `Wstar`
(and `stated` / `pushed` when present, else None). Any label that cannot be joined to a stored item (index out
of range, or the item missing a required field) is a HARD ERROR -- it is never skipped silently.

LABEL MAPPING (explicit; unknown values are a hard error). Classifier labels are mapped into the human
vocabulary {correct, wrong, other} for the comparison:
    C  -> correct          WSTAR -> wrong          NEITHER -> other
A classifier label of UNRESOLVED_ALIAS maps to NO human label, so it never agrees with any hand-label
(it counts as a disagreement) AND is listed separately in `alias_items`.

OUTPUT (one JSON: <outdir>/classify_vs_handlabel_9bit.json). control name, input paths, metric string,
thresholds, decision_rule string, n, n_agree, agreement fraction (total + per source run), the per-item
disagreement dump ({q, human, classify_label, rule_fired, answer_span}), the alias_items list, and a decision.

NEUTRAL DECISION (on the measured number only; pre-registered, no claim attached). agreement = n_agree / n.
Following the repo H4 precedent recorded in results_foldlisten_ext/handlabel_validation.json
("scorer-vs-human >= 0.9 on a hand-labelled subset >= 20 items"): PASS iff n >= 20 AND agreement >= 0.9, else
FAIL. No other interpretation.

  python controls/classify_vs_handlabel.py --selftest
  python controls/classify_vs_handlabel.py --run
  python controls/classify_vs_handlabel.py --run --outdir out
"""
import argparse
import json
import sys
from pathlib import Path

# FLAT-scp: controls/ for the sibling-control import, latent_verify/ for the repo imports (mirrors the sibling
# controls). The repo root (which holds the result_* dirs) is the parent of controls/.
_CONTROLS = Path(__file__).resolve().parent
_REPO_ROOT = _CONTROLS.parent
sys.path.insert(0, str(_CONTROLS))
sys.path.insert(0, str(_REPO_ROOT))

# Reuse the sibling control's rule-based classifier verbatim (its module top imports no torch, so this import
# is CPU-safe and pulls no model machinery, no artifacts).
from faithful_rescore import classify  # noqa: E402

# --------------------------------------------------------------------------- pre-registered constants
AGREE_THR = 0.9        # agreement (n_agree/n) at or above this is required to PASS
MIN_N = 20             # hand-labelled subset must have at least this many joined items to PASS

HANDLABEL_PATH = "results_foldlisten_ext/handlabel_fold_finals.json"   # persisted human hand-labels
# run key (in the handlabel artifact) -> the summary JSON whose items[] the labels index into (repo-root rel).
RUN_SUMMARIES = {
    "repro": "results_foldlisten_ext/out/foldlisten_judge_fl_9bit_repro_summary.json",
    "ext":   "results_foldlisten_ext/out/foldlisten_judge_fl_9bit_ext_summary.json",
}

# the classifier's label space and the human label vocabulary, with the explicit cross-map.
CLASSIFY_LABELS = ("C", "WSTAR", "NEITHER", "UNRESOLVED_ALIAS")
HUMAN_VOCAB = ("correct", "wrong", "other")
ALIAS_LABEL = "UNRESOLVED_ALIAS"
# UNRESOLVED_ALIAS is intentionally absent: it maps to no human label and never agrees.
CLASSIFY_TO_HUMAN = {"C": "correct", "WSTAR": "wrong", "NEITHER": "other"}

# fields the joined item must carry for the classifier to score it (used by the fail-loud join guard).
REQUIRED_ITEM_FIELDS = ("elicit_gen", "correct", "Wstar")

METRIC = (
    "OFFLINE classifier-vs-human agreement (no model): for each hand-labelled elicited final in "
    "results_foldlisten_ext/handlabel_fold_finals.json (runs 'repro'/'ext'), join the label to its stored item "
    "by POSITIONAL index into that run's summary items[] list, run the rule-based classify(gen, correct, wstar, "
    "stated, pushed) (imported from controls/faithful_rescore.py) on the item's elicit_gen with its "
    "correct/Wstar/stated/pushed, map the classifier label into the human vocabulary (C->correct, WSTAR->wrong, "
    "NEITHER->other; UNRESOLVED_ALIAS maps to no human label and never agrees), and count agreements. Report n, "
    "n_agree, agreement fraction (total and per source run), the per-item disagreement dump {q, human, "
    "classify_label, rule_fired, answer_span}, and the alias_items list. Any unjoinable label is a hard error."
)

DECISION_RULE = (
    "Pre-registered (repo H4 precedent, results_foldlisten_ext/handlabel_validation.json: 'scorer-vs-human "
    ">= 0.9 on a hand-labelled subset >= 20 items'): agreement = n_agree / n; PASS iff n >= 20 AND agreement "
    ">= 0.9, else FAIL. No other interpretation; no claim attached to any item, run, or the number."
)

WHAT = (
    "Measure agreement between the rule-based classify() (controls/faithful_rescore.py, CPU-only) and the "
    "persisted human hand-labels on the hand-labelled elicited-finals set (fold/listen 9b-it repro + ext runs)."
)


# --------------------------------------------------------------------------- pure helpers (selftest-able)
def map_classify_label(label):
    """Map a classifier label into the human vocabulary, or None for UNRESOLVED_ALIAS (which matches no human
    label). Raises ValueError on any label outside the classifier's label space. Pure (str -> str|None)."""
    if label not in CLASSIFY_LABELS:
        raise ValueError("unknown classify label %r (expected one of %s)" % (label, list(CLASSIFY_LABELS)))
    return CLASSIFY_TO_HUMAN.get(label)   # None for UNRESOLVED_ALIAS


def agree(classify_label, human):
    """True iff the classifier label maps to a human label equal to `human`. Validates both vocabularies,
    raising ValueError on an unknown value. UNRESOLVED_ALIAS maps to None and so never agrees. Pure -> bool."""
    if human not in HUMAN_VOCAB:
        raise ValueError("unknown human label %r (expected one of %s)" % (human, list(HUMAN_VOCAB)))
    mapped = map_classify_label(classify_label)
    return mapped is not None and mapped == human


def decide(n, agreement):
    """Pre-registered neutral decision on the measured number only: 'PASS' iff n >= MIN_N and agreement >=
    AGREE_THR, else 'FAIL'. Pure (int, float -> str)."""
    return "PASS" if (n >= MIN_N and agreement >= AGREE_THR) else "FAIL"


def join_item(items, idx, run):
    """Join a hand-label to its stored item by POSITIONAL index. Returns items[idx] iff idx is in range AND the
    item carries every REQUIRED_ITEM_FIELDS field; otherwise raises ValueError (fail loudly -- never skip).
    Pure (list, int, str -> dict)."""
    if not isinstance(items, list):
        raise ValueError("unjoinable handlabel %s:%s: items is not a list" % (run, idx))
    if not (0 <= idx < len(items)):
        raise ValueError("unjoinable handlabel %s:%s: index out of range (items has %d)"
                         % (run, idx, len(items)))
    item = items[idx]
    missing = [f for f in REQUIRED_ITEM_FIELDS if f not in item]
    if missing:
        raise ValueError("unjoinable handlabel %s:%s: item missing required fields %s" % (run, idx, missing))
    return item


def make_record(run, idx, q, human, classify_label, rule_fired, answer_span):
    """Build one scored record; validates both label vocabularies (raises on unknown) and computes the mapped
    human label + the agree flag. Pure (uses only agree / map_classify_label)."""
    return {
        "run": run,
        "idx": idx,
        "q": q,
        "human": human,
        "classify_label": classify_label,
        "mapped": map_classify_label(classify_label),
        "agree": agree(classify_label, human),
        "rule_fired": rule_fired,
        "answer_span": answer_span,
    }


def _dump_fields(r):
    """The per-item dump record (locator + the spec's verbatim {q, human, classify_label, rule_fired,
    answer_span}). Pure."""
    return {
        "run": r["run"], "idx": r["idx"], "q": r["q"], "human": r["human"],
        "classify_label": r["classify_label"], "rule_fired": r["rule_fired"],
        "answer_span": r["answer_span"],
    }


def aggregate(records):
    """Aggregate over scored records: n, n_agree, agreement (total + per run), the disagreement dump, the
    alias_items list, and the pre-registered decision. Pure (list[dict] -> dict)."""
    n = len(records)
    n_agree = sum(1 for r in records if r["agree"])
    agreement = (n_agree / n) if n else 0.0
    per_run = {}
    for r in records:
        pr = per_run.setdefault(r["run"], {"n": 0, "n_agree": 0})
        pr["n"] += 1
        if r["agree"]:
            pr["n_agree"] += 1
    for pr in per_run.values():
        pr["agreement"] = (pr["n_agree"] / pr["n"]) if pr["n"] else 0.0
    disagreements = [_dump_fields(r) for r in records if not r["agree"]]
    alias_items = [_dump_fields(r) for r in records if r["classify_label"] == ALIAS_LABEL]
    return {
        "n": n,
        "n_agree": n_agree,
        "agreement": agreement,
        "per_run": per_run,
        "disagreements": disagreements,
        "alias_items": alias_items,
        "decision": decide(n, agreement),
    }


# --------------------------------------------------------------------------- i/o + run
def _load_items(data):
    """Read the per-item list from either on-disk shape: {..., 'items':[...]} or
    {..., 'result':{'items':[...]}}. Returns the list (possibly empty)."""
    if isinstance(data.get("items"), list):
        return data["items"]
    res = data.get("result")
    if isinstance(res, dict) and isinstance(res.get("items"), list):
        return res["items"]
    return []


def _score_records():
    """Read the hand-labels + the run summaries, join every label to its stored item (failing loudly on any
    unjoinable label), score each with classify(), and return (records, summaries_used). Reads text only."""
    hl_path = _REPO_ROOT / HANDLABEL_PATH
    hl = json.loads(hl_path.read_text(encoding="utf-8"))
    labels = hl.get("labels")
    if not isinstance(labels, dict):
        raise SystemExit("[classify_vs_handlabel] no 'labels' dict in %s" % hl_path)
    source_summaries = hl.get("source_summaries", [])

    records = []
    summaries_used = {}
    for run in sorted(labels):
        if run not in RUN_SUMMARIES:
            raise SystemExit("[classify_vs_handlabel] unknown run key %r in handlabels; known: %s"
                             % (run, sorted(RUN_SUMMARIES)))
        rel = RUN_SUMMARIES[run]
        summary_path = _REPO_ROOT / rel
        # provenance cross-check: the artifact's declared source summaries must reference this file by name.
        if source_summaries and not any(Path(s).name == summary_path.name for s in source_summaries):
            raise SystemExit("[classify_vs_handlabel] run %r summary %s not referenced in handlabel "
                             "source_summaries %s" % (run, summary_path.name, source_summaries))
        summaries_used[run] = rel.replace("\\", "/")
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        items = _load_items(data)
        if not items:
            raise SystemExit("[classify_vs_handlabel] no items[] in %s" % summary_path)
        for idx_str in sorted(labels[run], key=lambda s: int(s)):
            human = labels[run][idx_str]
            if human not in HUMAN_VOCAB:      # unknown human label -> hard error (never silently map)
                raise SystemExit("[classify_vs_handlabel] unknown human label %r at %s:%s (expected %s)"
                                 % (human, run, idx_str, list(HUMAN_VOCAB)))
            idx = int(idx_str)
            item = join_item(items, idx, run)
            label, rule, span = classify(item.get("elicit_gen", ""), item["correct"], item["Wstar"],
                                         item.get("stated"), item.get("pushed"))
            records.append(make_record(run, idx, item.get("q"), human, label, rule, span))
    return records, summaries_used


def run(outdir):
    """Score every hand-labelled elicited final against classify(), write the output JSON, print a summary."""
    records, summaries_used = _score_records()
    agg = aggregate(records)
    out = {
        "control": "classify_vs_handlabel",
        "what": WHAT,
        "metric": METRIC,
        "thresholds": {"AGREE_THR": AGREE_THR, "MIN_N": MIN_N},
        "decision_rule": DECISION_RULE,
        "label_mapping": {"C": "correct", "WSTAR": "wrong", "NEITHER": "other",
                          "UNRESOLVED_ALIAS": None},
        "label_mapping_note": "UNRESOLVED_ALIAS maps to no human label; it never agrees (counts as a "
                              "disagreement) and is also listed in alias_items.",
        "classifier": "controls/faithful_rescore.py::classify (matcher_spec sections 1-6; CPU-only, no model)",
        "input_paths": {"handlabel": HANDLABEL_PATH, "summaries": summaries_used},
        "n": agg["n"],
        "n_agree": agg["n_agree"],
        "agreement": agg["agreement"],
        "per_run": agg["per_run"],
        "disagreements": agg["disagreements"],
        "alias_items": agg["alias_items"],
        "decision": agg["decision"],
    }
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "classify_vs_handlabel_9bit.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("[classify_vs_handlabel] n=%d n_agree=%d agreement=%.4f -> %s"
          % (agg["n"], agg["n_agree"], agg["agreement"], agg["decision"]), flush=True)
    for run_key, pr in sorted(agg["per_run"].items()):
        print("  [%s] n=%d n_agree=%d agreement=%.4f"
              % (run_key, pr["n"], pr["n_agree"], pr["agreement"]), flush=True)
    print("  disagreements=%d alias_items=%d" % (len(agg["disagreements"]), len(agg["alias_items"])), flush=True)
    print("[done] wrote %s" % str(out_path).replace("\\", "/"), flush=True)


# --------------------------------------------------------------------------- selftest (model-free, no artifacts)
def selftest():
    """Synthetic tests: the label mapping (all 3 mapped pairs + UNRESOLVED_ALIAS), the join-failure path (must
    raise), and the decision boundary (just above / below 0.9, plus n<20). Prints PASS/FAIL per case; exits
    nonzero on any FAIL. No model load, no artifact reads."""
    passed = []

    def check(name, fn):
        try:
            fn()
            print("PASS: %s" % name)
            passed.append(True)
        except AssertionError as e:
            print("FAIL: %s :: %s" % (name, e))
            passed.append(False)
        except Exception as e:
            print("FAIL: %s :: unexpected %s: %s" % (name, type(e).__name__, e))
            passed.append(False)

    # ---- label mapping: the 3 mapped pairs agree; cross pairs disagree ----
    def case_map_pairs():
        assert agree("C", "correct") is True
        assert agree("WSTAR", "wrong") is True
        assert agree("NEITHER", "other") is True
        assert agree("C", "wrong") is False
        assert agree("WSTAR", "correct") is False
        assert agree("NEITHER", "correct") is False
        assert map_classify_label("C") == "correct"
        assert map_classify_label("WSTAR") == "wrong"
        assert map_classify_label("NEITHER") == "other"
    check("mapping: C/WSTAR/NEITHER <-> correct/wrong/other (matched + crossed)", case_map_pairs)

    # ---- UNRESOLVED_ALIAS never agrees with any human label, and maps to None ----
    def case_alias():
        assert map_classify_label("UNRESOLVED_ALIAS") is None
        for h in HUMAN_VOCAB:
            assert agree("UNRESOLVED_ALIAS", h) is False, h
    check("mapping: UNRESOLVED_ALIAS maps to None and never agrees", case_alias)

    # ---- unknown vocabulary values are a hard error ----
    def case_unknown():
        raised = False
        try:
            map_classify_label("BOGUS")
        except ValueError:
            raised = True
        assert raised, "unknown classify label must raise"
        raised = False
        try:
            agree("C", "banana")
        except ValueError:
            raised = True
        assert raised, "unknown human label must raise"
    check("mapping: unknown classify/human labels raise ValueError", case_unknown)

    # ---- join-failure path: out-of-range index and a required-field-missing item must raise ----
    def case_join_fail():
        items = [{"elicit_gen": "Nile", "correct": "Nile", "Wstar": "Amazon"}]
        # valid join returns the item.
        assert join_item(items, 0, "repro") is items[0]
        # index out of range -> raise.
        raised = False
        try:
            join_item(items, 5, "repro")
        except ValueError:
            raised = True
        assert raised, "out-of-range index must raise"
        # item missing a required field -> raise.
        raised = False
        try:
            join_item([{"correct": "Nile", "Wstar": "Amazon"}], 0, "ext")  # no elicit_gen
        except ValueError:
            raised = True
        assert raised, "missing required field must raise"
    check("join: valid join returns item; out-of-range + missing-field raise", case_join_fail)

    # ---- decision boundary: at/above 0.9 with n>=20 PASSes; just-below and n<20 FAIL ----
    def case_decision_boundary():
        assert decide(20, 0.9) == "PASS"          # exactly at both thresholds
        assert decide(20, 0.90001) == "PASS"      # just above
        assert decide(20, 0.89999) == "FAIL"      # just below 0.9
        assert decide(19, 1.0) == "FAIL"          # n below MIN_N
        assert decide(56, 0.982) == "PASS"
        assert decide(56, 0.5) == "FAIL"
    check("decision: n>=20 & agreement>=0.9 -> PASS; below either -> FAIL", case_decision_boundary)

    # ---- aggregate: end-to-end over synthetic records (counts, per-run, dumps, decision) ----
    def case_aggregate():
        recs = [make_record("repro", 0, "q0", "correct", "C", "affirmative_C(W_absent)", "Nile"),
                make_record("repro", 2, "q2", "wrong", "WSTAR", "bare_entity_W", "China"),
                make_record("ext", 0, "q0e", "other", "NEITHER", "hedge_no_entity", "hmm"),
                make_record("ext", 2, "q2e", "correct", "UNRESOLVED_ALIAS", "bare_alias_miss", "Yaounde")]
        agg = aggregate(recs)
        assert agg["n"] == 4, agg["n"]
        assert agg["n_agree"] == 3, agg["n_agree"]                 # only the alias record disagrees
        assert abs(agg["agreement"] - 0.75) < 1e-9, agg["agreement"]
        assert agg["per_run"]["repro"] == {"n": 2, "n_agree": 2, "agreement": 1.0}, agg["per_run"]["repro"]
        assert agg["per_run"]["ext"]["n"] == 2 and agg["per_run"]["ext"]["n_agree"] == 1
        assert len(agg["disagreements"]) == 1 and agg["disagreements"][0]["idx"] == 2
        assert agg["disagreements"][0]["classify_label"] == "UNRESOLVED_ALIAS"
        assert len(agg["alias_items"]) == 1 and agg["alias_items"][0]["run"] == "ext"
        assert agg["decision"] == "FAIL"                          # n=4 < MIN_N
        # dump carries exactly the locator + verbatim spec fields.
        assert set(agg["disagreements"][0]) == {"run", "idx", "q", "human", "classify_label",
                                                 "rule_fired", "answer_span"}
    check("aggregate: counts, per-run fractions, disagreement/alias dumps, decision", case_aggregate)

    # ---- aggregate at the PASS boundary: 18/20 agree = 0.9 with n=20 -> PASS ----
    def case_aggregate_pass():
        recs = [make_record("repro", i, "q%d" % i, "correct", "C", "affirmative_C(W_absent)", "C")
                for i in range(18)]
        recs += [make_record("repro", 100 + i, "q%d" % i, "correct", "NEITHER", "default_neither", "x")
                 for i in range(2)]                                # 2 disagreements -> 18/20 = 0.9
        agg = aggregate(recs)
        assert agg["n"] == 20 and agg["n_agree"] == 18, (agg["n"], agg["n_agree"])
        assert abs(agg["agreement"] - 0.9) < 1e-9, agg["agreement"]
        assert agg["decision"] == "PASS", agg["decision"]
    check("aggregate: 18/20 = 0.9 at n=20 -> PASS", case_aggregate_pass)

    ok = all(passed)
    print("[selftest] %s (%d/%d cases)" % ("PASS" if ok else "FAIL", sum(passed), len(passed)))
    if not ok:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true", help="model-free logic tests; no model, no artifacts")
    p.add_argument("--run", action="store_true", help="score the hand-labelled elicited finals")
    p.add_argument("--outdir", default="out", help="output dir for classify_vs_handlabel_9bit.json")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    if args.run:
        run(args.outdir)
        return
    p.error("nothing to do: pass --selftest or --run")


if __name__ == "__main__":
    main()
