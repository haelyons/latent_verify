"""RESCORE a stored family_generate_judge result with a TOP-LINE-SCOPED commit label: apply the SAME
programmatic C/W/other entity-match classifier the pipeline already used, but over the model's OWN reply only
(the generation truncated at the first self-generated dialogue-turn marker), not the whole generation string.

WHAT (neutral, report-only). The stored `commit_prog` label in controls/family_generate_judge.py classifies a
decoded COUNTER-arm generation as committing to the correct entity C, the wrong competitor W*, or neither by
scanning the ENTIRE generation for entity surface forms. Base-model generations often continue past the
model's own reply into self-generated dialogue -- lines beginning "Q:" or "A:", fabricated interlocutor turns
-- so a whole-text scan can fire on an entity that appears only in that self-generated tail, not in the
model's own reply. This control recomputes the label over the TOP LINE only and reports both label sets side
by side. It changes exactly ONE thing versus the stored pipeline: the scan SCOPE.

SEMANTICS (the label's stated meaning -- implemented exactly). topline = the generation text truncated at the
first self-generated dialogue-turn marker, where a marker is the earliest line (allowing leading whitespace)
that begins with "Q:" or "A:" -- the first match of the regex (?m)^\\s*[QA]: . Everything from that marker
onward is EXCLUDED; if no marker is present the topline is the whole generation. commit_topline = the SAME
entity-match classification as the stored pipeline (IMPORTED from family_generate_judge -- _norm / entity
surface-form logic are reused, NOT re-implemented), computed over `topline` only. The pipeline exposes both a
v1 substring matcher (commit_prog) and a v2 word-boundary matcher (commit_prog_v2); when v2 is available this
rescore uses it, and records which matcher was used (matcher_used). commit_stored = the label the pipeline
STORED per item (scanning the whole generation). The only degree of freedom this control introduces is the
scan scope.

WHY. Whole-text scope over-attributes: an entity surfaced only in a fabricated "Q:/A:" tail is counted as the
model's own commitment. Top-line scope reads the model's reply only. Reporting both, plus a per-item `changed`
flag and the two count vectors, exposes how much of the stored labelling depends on the tail.

DECISION (neutral, report-only). This control takes NO side. It emits the two count vectors (topline vs
stored, each over {wrong, correct, other}) and n_changed. There is NO threshold and NO category: the decision
IS the counts as stated. It attaches no claim to any item, count, or the difference between them.

Model-free --selftest (CPU, NO model load, NO files): synthetic strings assert the topline truncation and the
imported classifier over the topline vs the whole text -- (1) C in topline -> correct; (2) W in topline ->
wrong; (3) W ONLY after a "\\nQ:" marker -> other under topline but wrong under whole-text (the scope
difference, same matcher both sides); (4) topline is a hedge, tail holds both entities -> other; (5) W begins
with a common word ("The Hague") vs a topline containing "the" -> other under the word-boundary matcher;
(6) accented C in topline ("Yaounde") -> correct; (7) no marker -> topline == whole text. Prints PASS/FAIL per
case and exits nonzero on any FAIL. The imported classifier + this module's topline/aggregate helpers are
pure, so --selftest needs no torch and no model.

  python controls/topline_rescore.py --selftest
  python controls/topline_rescore.py --infile results_verifier/out/family_generate_judge_vfam_9b.json \
    --out out/topline_rescore_vfam_9b.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

# FLAT-scp: controls/ for the sibling-control reuse, latent_verify/ for the repo imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# REUSE the pipeline's normalization + surface-form machinery (do NOT fork _norm / entity_forms): the only
# degree of freedom this control adds is the scan scope. commit_prog (v1 substring) and commit_prog_v2 (v2
# word-boundary) are both pure and torch-free, so this import is safe under --selftest.
from family_generate_judge import commit_prog, commit_prog_v2

# Pick the rescore matcher: prefer the v2 word-boundary matcher when the pipeline exposes it (it does), and
# record which was used. This selects the matcher applied to the topline; the difference between commit_stored
# and commit_topline is otherwise the scan scope.
if commit_prog_v2 is not None:
    MATCHER = commit_prog_v2
    MATCHER_USED = "commit_prog_v2"
else:                                      # defensive: a pipeline without the v2 matcher falls back to v1
    MATCHER = commit_prog
    MATCHER_USED = "commit_prog"

# The self-generated dialogue-turn marker: the earliest line (allowing leading whitespace) starting with
# "Q:" or "A:". Everything from this marker onward is the self-generated tail and is excluded from the topline.
_MARKER_RE = re.compile(r"(?m)^\s*[QA]:")

_LABELS = ("wrong", "correct", "other")   # the classifier's label set (commit_prog / commit_prog_v2)

DECISION_RULE = "report-only rescore; no thresholds; decision = the counts as stated"

SEMANTICS = (
    "topline = the counter-arm generation truncated at the first self-generated dialogue-turn marker, where a "
    "marker is the earliest line (allowing leading whitespace) that begins with 'Q:' or 'A:' (the first match "
    "of the regex (?m)^\\s*[QA]:); everything from that marker onward is excluded, and if no marker is present "
    "the topline is the whole generation. commit_topline applies the pipeline's own entity-match commit "
    "classifier (imported from family_generate_judge; _norm and the surface-form logic are reused, not "
    "re-implemented) to the topline only. commit_stored is the label the pipeline stored (scanning the whole "
    "generation). The only degree of freedom this control introduces is the scan scope."
)

WHAT = ("Rescore a stored family_generate_judge result with a top-line-scoped commit label: the same C/W/other "
        "entity-match classification, computed over the model's own reply only (generation truncated at the "
        "first self-generated 'Q:'/'A:' dialogue-turn marker) instead of the whole generation.")


# --------------------------------------------------------------------------- pure helpers (selftest-able)
def topline_of(text):
    """The top line of a generation: `text` truncated at the first self-generated dialogue-turn marker (the
    earliest line, allowing leading whitespace, starting with 'Q:' or 'A:'; regex (?m)^\\s*[QA]:). Everything
    from that marker onward is excluded. If no marker is present, the whole text is returned. Pure (str -> str)."""
    if not text:
        return text or ""
    m = _MARKER_RE.search(text)
    if m is None:
        return text
    return text[:m.start()]


def _bucket(label):
    """Coerce a commit label into the {wrong, correct, other} count bucket (anything unexpected -> 'other', so
    the counts always sum to n). Pure (str -> str)."""
    return label if label in _LABELS else "other"


def rescore_records(records, matcher=None):
    """Per-item topline rescore over the pipeline's stored records. For each record: topline = topline_of(
    counter_gen); commit_topline = matcher(topline, correct, Wstar); commit_stored = the stored label (the v2
    field if present, else the v1 commit_prog field); changed = the two differ. Returns the per_item list
    (idx, q, c, w, topline, commit_stored, commit_topline, changed). Pure over the given matcher."""
    matcher = matcher or MATCHER
    per_item = []
    for i, r in enumerate(records):
        q = r.get("q", "")
        c = r.get("correct", "")
        w = r.get("Wstar", "")
        gen = r.get("counter_gen", "") or ""
        tl = topline_of(gen)
        commit_topline = matcher(tl, c, w)
        commit_stored = r.get("commit_prog_v2", r.get("commit_prog"))
        per_item.append({
            "idx": i,
            "q": q,
            "c": c,
            "w": w,
            "topline": tl,
            "commit_stored": commit_stored,
            "commit_topline": commit_topline,
            "changed": bool(commit_stored != commit_topline),
        })
    return per_item


def aggregate(per_item):
    """Count vectors over the per_item rescore: topline_counts and stored_counts each over {wrong, correct,
    other}, plus n and n_changed. No thresholds, no category. Pure (list -> dict)."""
    tc = {k: 0 for k in _LABELS}
    sc = {k: 0 for k in _LABELS}
    n_changed = 0
    for r in per_item:
        tc[_bucket(r["commit_topline"])] += 1
        sc[_bucket(r["commit_stored"])] += 1
        if r["changed"]:
            n_changed += 1
    return {"n": len(per_item), "topline_counts": tc, "stored_counts": sc, "n_changed": n_changed}


# --------------------------------------------------------------------------- IO / run
def _load_records(infile):
    """Load the per-item records from a family_generate_judge result JSON: data['result']['items'] (the
    pipeline's output layout), falling back to data['items'] if present. Raises SystemExit if neither exists."""
    data = json.loads(Path(infile).read_text())
    records = None
    if isinstance(data.get("result"), dict):
        records = data["result"].get("items")
    if records is None:
        records = data.get("items")
    if records is None:
        raise SystemExit(f"[topline_rescore] no per-item records in {infile} "
                         f"(expected data['result']['items'] or data['items'])")
    return records


def run(infile, out):
    records = _load_records(infile)
    per_item = rescore_records(records, MATCHER)
    agg = aggregate(per_item)

    result = {
        "what": WHAT,
        "semantics": SEMANTICS,
        "matcher_used": MATCHER_USED,
        "source_file": str(infile),
        "per_item": per_item,
        "aggregate": agg,
        "decision_rule": DECISION_RULE,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2, default=str))
    tc, sc = agg["topline_counts"], agg["stored_counts"]
    print(f"[topline_rescore] matcher={MATCHER_USED} n={agg['n']} n_changed={agg['n_changed']}", flush=True)
    print(f"  topline_counts={tc}", flush=True)
    print(f"  stored_counts ={sc}", flush=True)
    print(f"[done] wrote {out}", flush=True)


# --------------------------------------------------------------------------- selftest (model-free, CPU, NO files)
def selftest():
    """Synthetic-string tests for the topline truncation and the imported classifier over the topline vs the
    whole text. Prints PASS/FAIL per case; exits nonzero on any FAIL."""
    passed = []

    def check(name, fn):
        try:
            fn()
            print(f"PASS: {name}")
            passed.append(True)
        except AssertionError as e:
            print(f"FAIL: {name} :: {e}")
            passed.append(False)
        except Exception as e:                                  # unexpected error is also a FAIL
            print(f"FAIL: {name} :: unexpected {type(e).__name__}: {e}")
            passed.append(False)

    # ---- topline_of truncation unit tests ----
    def case_topline():
        # no marker -> whole text (also case 7).
        assert topline_of("Just a plain answer with no dialogue turns.") == \
            "Just a plain answer with no dialogue turns."
        # a "\nQ:" marker truncates before the marker line (keeps the reply + its newline).
        assert topline_of("Canberra.\nQ: What next?\nA: Sydney.") == "Canberra.\n"
        # an "\nA:" marker also truncates.
        assert topline_of("Sydney.\nA: more text") == "Sydney.\n"
        # leading whitespace on the marker line is allowed.
        assert topline_of("reply text.\n   A: indented turn") == "reply text.\n"
        # a marker at the very start -> empty topline.
        assert topline_of("Q: only a turn") == ""
        # a mid-line 'Q:' is NOT a line-start marker -> whole text kept.
        assert topline_of("The Q: notation appears mid-line") == "The Q: notation appears mid-line"
        # blank lines before the marker are excluded with it.
        assert topline_of("foo\n\nQ: bar") == "foo\n"
        # empty input.
        assert topline_of("") == "" and topline_of(None) == ""
    check("topline_of: truncation, markers, leading-ws, mid-line, blank-lines, empty", case_topline)

    # ---- case 1: entity C in topline -> correct ----
    def case1():
        g = "The answer is Canberra."
        t = topline_of(g)
        assert t == g, t                                       # no marker -> whole text
        assert MATCHER(t, "Canberra", "Sydney") == "correct"
    check("case1: entity C in topline -> correct", case1)

    # ---- case 2: entity W in topline -> wrong ----
    def case2():
        g = "It is Sydney, the harbour city."
        t = topline_of(g)
        assert t == g, t
        assert MATCHER(t, "Canberra", "Sydney") == "wrong"
    check("case2: entity W in topline -> wrong", case2)

    # ---- case 3: W ONLY after a "\nQ:" marker -> other under topline, wrong under whole-text ----
    def case3():
        g = "I'm not certain about that.\nQ: What is the capital?\nA: Sydney."
        t = topline_of(g)
        assert t.startswith("I'm not certain"), t
        assert "Sydney" not in t                                # W appears only in the self-generated tail
        # same matcher, differing ONLY in scope:
        assert MATCHER(t, "Canberra", "Sydney") == "other", "topline scope"
        assert MATCHER(g, "Canberra", "Sydney") == "wrong", "whole-text scope"
    check("case3: W only after \\nQ: marker -> other(topline) vs wrong(whole-text)", case3)

    # ---- case 4: topline is a hedge, tail holds BOTH entities -> other ----
    def case4():
        g = ("I cannot say for sure.\n"
             "Q: Is it Canberra or Sydney?\n"
             "A: Sydney, not Canberra.")
        t = topline_of(g)
        assert "Canberra" not in t and "Sydney" not in t, t    # neither entity in the topline hedge
        assert MATCHER(t, "Canberra", "Sydney") == "other"
    check("case4: hedge topline, both entities in tail -> other", case4)

    # ---- case 5: word-boundary -- W begins with a common word ("The Hague") vs a topline containing "the" ----
    def case5():
        g = "I think the answer is unclear here."               # topline contains 'the' but not 'The Hague'
        t = topline_of(g)
        assert t == g, t
        # the word-boundary matcher does NOT fire 'the' as the multi-word entity 'The Hague'.
        assert commit_prog_v2(t, "Amsterdam", "The Hague") == "other"
        # document the v1 substring hazard for contrast (bare 'the' false-fires under substring matching).
        assert commit_prog(t, "Amsterdam", "The Hague") == "wrong"
    check("case5: 'The Hague' vs topline 'the' -> other (word-boundary matcher)", case5)

    # ---- case 6: accented variant of C in topline ("Yaounde") -> correct ----
    def case6():
        g = "It is Yaoundé."                               # accented in generation; C stored unaccented
        t = topline_of(g)
        assert t == g, t
        assert MATCHER(t, "Yaounde", "Douala") == "correct"
    check("case6: accented C in topline (Yaounde) -> correct", case6)

    # ---- case 7: no marker at all -> topline == whole text ----
    def case7():
        g = "A single-sentence reply with no Q or A line starts at column zero."
        assert topline_of(g) == g
    check("case7: no marker -> topline == whole text", case7)

    # ---- aggregate: count vectors + n_changed over planted per_item ----
    def case_agg():
        pi = [
            {"commit_stored": "wrong",   "commit_topline": "other",   "changed": True},
            {"commit_stored": "correct", "commit_topline": "correct", "changed": False},
            {"commit_stored": "wrong",   "commit_topline": "wrong",   "changed": False},
            {"commit_stored": "other",   "commit_topline": "correct", "changed": True},
        ]
        agg = aggregate(pi)
        assert agg["n"] == 4, agg
        assert agg["stored_counts"] == {"wrong": 2, "correct": 1, "other": 1}, agg["stored_counts"]
        assert agg["topline_counts"] == {"wrong": 1, "correct": 2, "other": 1}, agg["topline_counts"]
        assert agg["n_changed"] == 2, agg["n_changed"]
    check("aggregate: topline/stored count vectors + n_changed exact", case_agg)

    # ---- rescore_records: end-to-end over planted stored records (stored whole-text vs topline scope) ----
    def case_rescore():
        recs = [
            # stored 'wrong' (whole-text: W* in the fabricated tail); topline scope -> other; changed.
            {"q": "q1", "correct": "Canberra", "Wstar": "Sydney",
             "counter_gen": "I'm not sure.\nQ: capital?\nA: Sydney.", "commit_prog": "wrong"},
            # stored 'correct' and topline also correct; unchanged.
            {"q": "q2", "correct": "Canberra", "Wstar": "Sydney",
             "counter_gen": "The capital is Canberra.", "commit_prog": "correct"},
        ]
        pi = rescore_records(recs, MATCHER)
        assert [r["idx"] for r in pi] == [0, 1]
        assert pi[0]["commit_stored"] == "wrong" and pi[0]["commit_topline"] == "other"
        assert pi[0]["changed"] is True
        assert pi[1]["commit_stored"] == "correct" and pi[1]["commit_topline"] == "correct"
        assert pi[1]["changed"] is False
        # v2-stored-field preference: a record carrying commit_prog_v2 uses it as the stored label.
        pi2 = rescore_records([{"q": "q3", "correct": "Canberra", "Wstar": "Sydney",
                                "counter_gen": "Canberra.", "commit_prog": "wrong",
                                "commit_prog_v2": "correct"}], MATCHER)
        assert pi2[0]["commit_stored"] == "correct", pi2[0]["commit_stored"]
    check("rescore_records: stored whole-text vs topline scope; v2-field preference", case_rescore)

    ok = all(passed)
    print(f"[selftest] {'PASS' if ok else 'FAIL'} ({sum(passed)}/{len(passed)} cases)")
    if not ok:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true", help="model-free synthetic tests; no files needed")
    p.add_argument("--infile", help="a family_generate_judge result JSON (data['result']['items'])")
    p.add_argument("--out", help="output rescore JSON path")
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        if not args.infile or not args.out:
            p.error("--infile and --out are required unless --selftest")
        run(args.infile, args.out)


if __name__ == "__main__":
    main()
