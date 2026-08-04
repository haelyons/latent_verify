"""forcedfinal_source_census.py -- the offline, $0, runs-FIRST census of REGISTRATION_forcedfinal_distributional.md.

WHY. Registration section 13.1: before a single GPU-second is spent, the six committed foldlisten
summaries are censused offline: section 3.5 source integrity, the SHA-256 of each source file (the
join later asserts it equals the SHA the GPU instrument computed on the file it was shipped -- a
truncated scp is caught there), every section 6.3 per-record contamination field, and the
generation-label vectors. This file is OFFLINE, CPU, imports NO torch at any level, and is NEVER
shipped to a box. The contamination functions are imported from controls/forcedfinal_dist.py (whose
module level is torch-free) so the census and the on-box recompute are the SAME code -- the join's
equality assert then checks the ship, not two implementations.

  python controls/forcedfinal_source_census.py --selftest
  python controls/forcedfinal_source_census.py --run          # all six cells, writes out/forcedfinal_census_<cell>.json
"""
import argparse
import json
import sys
from pathlib import Path

# FLAT-scp convention kept for symmetry; this file itself is never shipped.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forcedfinal_dist import (  # torch-free at module level (torch lives inside its run path only)
    N_ITEMS, N_RECORDS, DIRECTIONS, PUSH_COUNTER_TEMPLATE, NEUTRAL_TURN,
    contamination_fields, join_key, sha256_file,
)

SOURCES = {
    "2bbase": "results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_2bbase_ext2_summary.json",
    "2bit": "results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_2bit_ext2_summary.json",
    "9bbase": "results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json",
    "9bit": "results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_9bit_ext2_summary.json",
    "27bbase": "results_foldlisten_nelicit_27b/out/foldlisten_judge_fl_27bbase_ext2_summary.json",
    "27bit": "results_foldlisten_nelicit_27b/out/foldlisten_judge_fl_27bit_ext2_summary.json",
}
LABEL_FIELDS = ("faithful_elicit", "faithful_neutral_elicit", "commit_elicit", "commit_neutral_elicit")

DECISION_RULE = (
    "Per cell, offline, model-free, before any GPU second (registration sections 3.5, 6.3, 13.1): the "
    "summary parses; len(items) == 164; the ordered [join_key(q), direction] list has 82 distinct q, both "
    "directions per q, and the committed per-item (fold, listen) pairing order; the q-set is identical "
    "across all six cells (section 1.1, re-asserted not inherited). Any failure -> decision SOURCE_MISSING "
    "for that cell (the cell is VOIDED downstream; the path and the failure are printed). Otherwise PASS. "
    "SHA-256 of the source file is recorded for the join's census-vs-on-box cross-check. Section 6.3 "
    "contamination fields are computed per record per arm at slot forced_final from the persisted prompt "
    "strings, via the same imported functions the GPU instrument recomputes with; contamination is a stamp "
    "and a stratifier, never a band -- no threshold here reads a contamination count."
)


def integrity(items):
    """Section 3.5 checks on a summary's items list. Returns (checks_dict, ordered_keys). Pure."""
    checks = {"n_records_164": len(items) == N_RECORDS}
    keys, qset = [], {}
    for r in items:
        k = join_key(r.get("q"))
        keys.append([k, r.get("cell")])
        qset.setdefault(k, set()).add(r.get("cell"))
    checks["n_distinct_q_82"] = len(qset) == N_ITEMS
    checks["both_directions_per_q"] = all(v == set(DIRECTIONS) for v in qset.values())
    # committed item order: for each item, its fold record then its listen record, consecutively
    order_ok = len(items) == N_RECORDS
    for i in range(0, min(len(items), N_RECORDS) - 1, 2):
        a, b = items[i], items[i + 1]
        if not (join_key(a.get("q")) == join_key(b.get("q")) and a.get("cell") == "fold" and b.get("cell") == "listen"):
            order_ok = False
            break
    checks["item_order_fold_listen_pairs"] = order_ok
    return checks, keys


def census_cell(cell, path):
    p = Path(path)
    out = {"instrument": "forcedfinal_source_census", "cell": cell, "source_path": str(path),
           "registration": "REGISTRATION_forcedfinal_distributional.md",
           "decision_rule": DECISION_RULE, "thresholds": {"n_items": N_ITEMS, "n_records": N_RECORDS}}
    if not p.exists():
        out.update(decision="SOURCE_MISSING", failure=f"file absent: {path}")
        return out
    try:
        src = json.loads(p.read_text())
        items = src["items"]
    except Exception as e:
        out.update(decision="SOURCE_MISSING", failure=f"unparseable: {e!r}")
        return out
    checks, keys = integrity(items)
    out["checks"] = checks
    out["source_sha256"] = sha256_file(path)
    out["ordered_keys"] = keys
    if not all(checks.values()):
        out.update(decision="SOURCE_MISSING", failure=f"integrity: {checks}")
        return out
    is_chat = cell.endswith("it")
    recs = []
    for r in items:
        q, C, W = r["q"], r["correct"], r["Wstar"]
        stated, pushed = r["stated"], r["pushed"]
        challenge = PUSH_COUNTER_TEMPLATE.format(W=pushed)
        rec = {"join_key": join_key(q), "direction": r["cell"],
               "ctx_counter": contamination_fields(r["elicit_prompt"], r["counter_gen"],
                                                   q, stated, challenge, C, W, is_chat),
               "ctx_neutral": contamination_fields(r["neutral_elicit_prompt"], r["neutral_gen"],
                                                   q, stated, NEUTRAL_TURN, C, W, is_chat)}
        for f in LABEL_FIELDS:
            rec[f] = r[f]
        recs.append(rec)
    out["items"] = recs
    for arm in ("counter", "neutral"):
        out[f"n_ctx_clean_{arm}"] = sum(1 for r in recs if r[f"ctx_{arm}"]["ctx_clean"])
    out["decision"] = "PASS"
    return out


def run():
    outdir = Path("out")
    outdir.mkdir(parents=True, exist_ok=True)
    qsets, decisions = {}, {}
    for cell, path in SOURCES.items():
        c = census_cell(cell, path)
        decisions[cell] = c["decision"]
        if c["decision"] == "PASS":
            qsets[cell] = frozenset(k for k, _ in c["ordered_keys"])
        p = outdir / f"forcedfinal_census_{cell}.json"
        p.write_text(json.dumps(c, indent=2))
        print(f"[{cell}] {c['decision']}  clean(counter)={c.get('n_ctx_clean_counter')} "
              f"clean(neutral)={c.get('n_ctx_clean_neutral')} sha={c.get('source_sha256', '')[:12]} -> {p}",
              flush=True)
    passed = [c for c, d in decisions.items() if d == "PASS"]
    if len(passed) > 1:
        ref = qsets[passed[0]]
        same = all(qsets[c] == ref for c in passed)
        print(f"[cross-cell] q-set identical across {len(passed)} passing cells: {same}", flush=True)
        if not same:
            raise SystemExit("SOURCE_MISSING: q-set differs across cells (section 1.1 re-assert failed)")
    assert "torch" not in sys.modules, "census imported torch -- forbidden at any level (section 13)"
    print(f"[census] decisions: {decisions}", flush=True)


def selftest():
    ok = 0

    def ck(cond, msg):
        nonlocal ok
        assert cond, msg
        ok += 1

    def planted(n_items=N_ITEMS, drop_last=False, swap=False):
        items = []
        for i in range(n_items):
            for cell in ("fold", "listen"):
                items.append({"q": f"question {i}?", "cell": cell})
        if drop_last:
            items = items[:-1]
        if swap:
            items[0], items[1] = items[1], items[0]   # listen before fold -> order failure
        return items

    c, _ = integrity(planted())
    ck(all(c.values()), "planted 164 well-formed passes")
    c, _ = integrity(planted(drop_last=True))
    ck(not c["n_records_164"], "163 records -> SOURCE_MISSING branch")
    c, _ = integrity(planted(swap=True))
    ck(not c["item_order_fold_listen_pairs"], "reordered items -> SOURCE_MISSING branch")
    bad = planted()
    bad[3]["q"] = bad[1]["q"]   # duplicate q on a listen slot -> distinct-q and pairing break
    c, _ = integrity(bad)
    ck(not (c["n_distinct_q_82"] and c["both_directions_per_q"] and c["item_order_fold_listen_pairs"]),
       "key-set failure detected")

    # SOURCE_MISSING on absent / unparseable files
    ck(census_cell("2bbase", "does/not/exist.json")["decision"] == "SOURCE_MISSING", "absent file")

    # contamination via the SAME imported functions (one implementation; the S4 worked example)
    inv = "Ankara.\nQ: What is the capital of Turkey?\nA: Ankara"
    pr = f"Q: Which city is the most populous in Turkey?\nA: Istanbul.\nQ: {PUSH_COUNTER_TEMPLATE.format(W='Ankara')}\nA: {inv}\nQ: x\nA:"
    f = contamination_fields(pr, inv, "Which city is the most populous in Turkey?", "Istanbul",
                             PUSH_COUNTER_TEMPLATE.format(W="Ankara"), "Istanbul", "Ankara", False)
    ck(f["ctx_invented_question"] and f["ctx_contains_own_Wstar_outside_plant"] and not f["ctx_clean"],
       "S4 worked example: invented turn carrying the item's own W* detected")

    ck("torch" not in sys.modules, "selftest imported torch -- forbidden")
    print(f"[selftest] forcedfinal_source_census: {ok} asserts passed", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.run:
        run()
    else:
        ap.error("one of --selftest / --run is required")


if __name__ == "__main__":
    main()
