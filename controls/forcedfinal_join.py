"""forcedfinal_join.py -- the OFFLINE, ONLY verdict source of REGISTRATION_forcedfinal_distributional.md.

WHY. Registration section 13 / A10: verdict emission is offline-only and single-sourced. The GPU
instrument (controls/forcedfinal_dist.py) emits measurements; THIS file joins them to the offline census
(controls/forcedfinal_source_census.py) and emits every section 9 verdict: replay fidelity (9.1), context
cleanliness stamps (9.2), the descriptive state vectors (9.3), THE PRIMARY layer-agreement bands (9.4),
the within-chain transition matrices (9.5, cross-arm/cross-direction RAISES), and the per-half round
verdicts (9.6). The headline is the -it triple over (2b-it, 9b-it, 27b-it) at (direction=fold,
arm=counter, slot=forced_final), quoted as a triple or not at all (section 8.2). The base half is
SECONDARY, stamped CONTEXT_CONTAMINATED_MEASURED. No base-vs-it contrast is computed at any slot
(section 6.5) and no cross-arm or cross-direction transition exists (section 2).

Thresholds are IMPORTED, not chosen: CONCORDANT_MAX = foldlisten_judge.ARTIFACT_MAX_DELTA (0.10,
inclusive); DISCORDANT_MIN = faithful_rescore.CHANGE_THR (0.30, strict >). At n=82 the integer cuts are
<= 8 / 9-24 / >= 25. The reference line GATE_AGREE_MIN_FRAC = 18/22 is printed beside every 9.4 verdict
and is NOT a band edge.

  python controls/forcedfinal_join.py --selftest
  python controls/forcedfinal_join.py --run          # reads out/forcedfinal_census_*.json + out/forcedfinal_dist_*.json
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foldlisten_judge import ARTIFACT_MAX_DELTA, FAITHFUL_TO_COMMIT, GATE_AGREE_MIN_FRAC
from faithful_rescore import CHANGE_THR
from forcedfinal_dist import (
    N_ITEMS, DIRECTIONS, ARMS, SLOTS, STATES, KEYS,
    collapse_state, make_stamp, validate_provenance, ProvenanceIncomplete,
    PROVENANCE_LOAD_BEARING,
)

CELLS = ("2bbase", "2bit", "9bbase", "9bit", "27bbase", "27bit")
IT_HALF = ("2bit", "9bit", "27bit")
BASE_HALF = ("2bbase", "9bbase", "27bbase")
CONCORDANT_MAX = ARTIFACT_MAX_DELTA   # 0.10, inclusive (foldlisten_judge.py:129)
DISCORDANT_MIN = CHANGE_THR           # 0.30, strict >  (faithful_rescore.py:77)
assert CONCORDANT_MAX == 0.10 and DISCORDANT_MIN == 0.30, "borrowed constants moved -- see registration section 8"
GEN_LABELS = ("C", "WSTAR", "NEITHER", "UNRESOLVED_ALIAS")

DECISION_RULE = (
    "Per cell, resolution order total, earlier branch wins (section 9). 9.1: SOURCE_MISSING (census "
    "decision not PASS, dist artifact absent, census-vs-on-box SHA-256 mismatch, or ctx recompute "
    "inequality) -> cell VOIDED; PROMPT_REPLAY_MISMATCH (any record fails roundtrip/bos/rebuild) -> cell "
    "VOIDED, denominators stay 164, failing records printed; CONF_PROXY_SIGN_UNSTABLE "
    "(n_conf_proxy_sign_flips > 0) -> DOWNGRADED, stamped; else REPLAY_FAITHFUL. "
    "PROVENANCE_PER_ARTIFACT_ABSENT -> no 9.4 verdict from that cell. 9.4 per (cell, direction, arm) at "
    "slot forced_final: collapse Rule S {GREY_*->GREY, FAVOURS_C->C, FAVOURS_WSTAR->WSTAR}; collapse the "
    "faithful-strict label by the shipped FAITHFUL_TO_COMMIT (C->C, WSTAR->WSTAR, NEITHER->GREY, "
    "UNRESOLVED_ALIAS->GREY); disagree_frac over 82 grey-INCLUDED; LAYERS_CONCORDANT iff <= 0.10 (<=8/82), "
    "LAYERS_DISCORDANT iff > 0.30 (>=25/82), else LAYERS_PARTIAL; LAYERS_UNEVALUABLE on a voided cell or "
    "an absent label. The commit family is banded identically; different bands -> "
    "LAYER_AGREEMENT_CONTESTED, no single number from that cell. S-vs-S-set disagreement banded on the "
    "same edges; > 0.30 -> STATE_VARIANT_DEPENDENT, both state vectors published, no single vector. "
    "9.6 per half at (fold, counter, forced_final): ROUND_CONCORDANT iff LAYERS_CONCORDANT at >= 2 of 3 "
    "scales and no scale DISCORDANT; ROUND_DISCORDANT mirror; ROUND_UNEVALUABLE if < 2 scales produced a "
    "9.4 verdict; else ROUND_MIXED. Headline: the -it triple, quoted as a triple or not at all. The base "
    "half is SECONDARY, stamped CONTEXT_CONTAMINATED_MEASURED. No base-vs-it contrast at any slot; no "
    "neutral->counter or fold->listen transition (the builder raises)."
)


# ------------------------------------------------------------------ pure band + verdict functions
def band_layers(n_disagree, n=N_ITEMS):
    """Section 9.4 bands on disagree_frac = n_disagree/n, grey included. Pure."""
    f = n_disagree / n
    if f <= CONCORDANT_MAX:
        return "LAYERS_CONCORDANT"
    if f > DISCORDANT_MIN:
        return "LAYERS_DISCORDANT"
    return "LAYERS_PARTIAL"


def band_state_variant(n_disagree, n=N_ITEMS):
    """Section 4.4 bands, the SAME borrowed edges. Pure."""
    f = n_disagree / n
    if f <= CONCORDANT_MAX:
        return "STATE_VARIANT_STABLE"
    if f > DISCORDANT_MIN:
        return "STATE_VARIANT_DEPENDENT"
    return "STATE_VARIANT_PARTIAL"


def gen_collapse(label):
    """Faithful/commit 4-value label -> {C, WSTAR, GREY} via the SHIPPED FAITHFUL_TO_COMMIT map
    (C->correct->C, WSTAR->wrong->WSTAR, NEITHER/UNRESOLVED_ALIAS->other->GREY). Pure."""
    return {"correct": "C", "wrong": "WSTAR", "other": "GREY"}[FAITHFUL_TO_COMMIT[label]]


def replay_fidelity(source_ok, any_record_mismatch, n_sign_flips):
    """Section 9.1, earlier branch wins. Pure."""
    if not source_ok:
        return "SOURCE_MISSING"
    if any_record_mismatch:
        return "PROMPT_REPLAY_MISMATCH"
    if n_sign_flips > 0:
        return "CONF_PROXY_SIGN_UNSTABLE"
    return "REPLAY_FAITHFUL"


def ctx_verdict(clean_flags):
    """Section 9.2. Pure."""
    n = sum(1 for c in clean_flags if c)
    if n == len(clean_flags):
        return "CTX_CLEAN_ALL", n
    if n == 0:
        return "CTX_CONTAMINATED_ALL", n
    return "CTX_MIXED", n


def round_verdict(bands):
    """Section 9.6 over the three scales' 9.4 bands at the primary axis. Pure."""
    evaluable = [b for b in bands if b in ("LAYERS_CONCORDANT", "LAYERS_PARTIAL", "LAYERS_DISCORDANT")]
    if len(evaluable) < 2:
        return "ROUND_UNEVALUABLE"
    nc = evaluable.count("LAYERS_CONCORDANT")
    nd = evaluable.count("LAYERS_DISCORDANT")
    if nc >= 2 and nd == 0:
        return "ROUND_CONCORDANT"
    if nd >= 2 and nc == 0:
        return "ROUND_DISCORDANT"
    return "ROUND_MIXED"


def transition_matrix(chain_records):
    """5x5 Rule-S transition counts for ONE (cell, direction, arm) chain across two slots. RAISES on a
    cross-arm or cross-direction pair (section 9.5's hard structural constraint). Input: list of
    (rec_slot_a, rec_slot_b) pairs. Pure."""
    m = {a: {b: 0 for b in STATES} for a in STATES}
    for a, b in chain_records:
        if a["direction"] != b["direction"]:
            raise ValueError(f"cross-direction transition forbidden: {a['direction']} -> {b['direction']}")
        if a["turn2"] != b["turn2"]:
            raise ValueError(f"cross-arm transition forbidden: {a['turn2']} -> {b['turn2']}")
        m[a["state"]][b["state"]] += 1
    return m


def contingency(pairs, rows, cols):
    """Counts table {row: {col: n}} + marginals; asserts the total equals len(pairs). Pure."""
    t = {r: {c: 0 for c in cols} for r in rows}
    for r, c in pairs:
        t[r][c] += 1
    total = sum(sum(v.values()) for v in t.values())
    assert total == len(pairs), "table does not sum to n"
    return t


def state_vector(records):
    """Section 9.3 descriptive block for one (cell, direction, arm, slot) record set. Pure."""
    states = Counter(r["state"] for r in records)
    collapsed = Counter(collapse_state(r["state"]) for r in records)
    s_vs_set = sum(1 for r in records if r["state"] != r["state_set"])
    onset = [r for r in records if not r["measure"]["argmax_in_union"]]
    four = Counter()
    for r in records:
        m = r["measure"]
        four["both" if (m["argmax_in_V_C"] and m["argmax_in_V_W"]) else
             "C_only" if m["argmax_in_V_C"] else
             "W_only" if m["argmax_in_V_W"] else "neither"] += 1
    non_onset_toks = Counter(r["measure"]["argmax_tok_str"] for r in onset)
    canon = records[0]["key"] if records else "space"
    plateaus = [r["measure"]["entities"]["C"][canon]["tie_plateau"] for r in records]
    return {
        "n": len(records),
        "states": {s: states.get(s, 0) for s in STATES},
        "collapsed": {c: collapsed.get(c, 0) for c in ("C", "WSTAR", "GREY")},
        "n_state_disagree_S_vs_Sset": s_vs_set,
        "state_variant_band": band_state_variant(s_vs_set, max(len(records), 1)),
        "n_state_agrees_with_argmax": sum(1 for r in records if r["state_agrees_with_argmax"]),
        "frac_slot_answer_onset": (len(records) - len(onset)) / max(len(records), 1),
        "onset_decomposition": dict(four),
        "top5_non_onset_argmax": non_onset_toks.most_common(5),
        "modal_non_onset": (non_onset_toks.most_common(1)[0] if non_onset_toks else None),
        "n_rank_resolved": sum(1 for r in records
                               if r["measure"]["entities"]["C"][canon]["rank_resolved"]),
        "median_tie_plateau": (statistics.median(plateaus) if plateaus else None),
        "n_first_token_collision": {k: sum(1 for r in records if r["measure"][f"first_token_collision_{k}"])
                                    for k in KEYS},
    }


def layers_block(records, label_field):
    """Section 9.4 for one (cell, direction, arm) at slot forced_final: disagree_frac, band, the 3x3 and
    5x4 tables, and the same for the commit family + the contested check. `records` are the forced_final
    records of that (direction, arm). Pure."""
    if any(r.get(label_field) is None for r in records) or len(records) != N_ITEMS:
        return {"verdict": "LAYERS_UNEVALUABLE", "reason": "generation label absent or wrong record count"}
    out = {}
    for fam, field in (("faithful", label_field), ("commit", label_field.replace("faithful", "commit"))):
        if fam == "commit":
            def cc(lbl):   # commit vocabulary is already correct/wrong/other
                return {"correct": "C", "wrong": "WSTAR", "other": "GREY"}[lbl]
            pairs3 = [(collapse_state(r["state"]), cc(r[field])) for r in records]
            raw_labels = sorted({r[field] for r in records})
            pairs54 = [(r["state"], r[field]) for r in records]
            cols = raw_labels
        else:
            pairs3 = [(collapse_state(r["state"]), gen_collapse(r[field])) for r in records]
            pairs54 = [(r["state"], r[field]) for r in records]
            cols = list(GEN_LABELS)
        n_dis = sum(1 for a, b in pairs3 if a != b)
        out[fam] = {
            "n_disagree": n_dis, "disagree_frac": n_dis / N_ITEMS, "band": band_layers(n_dis),
            "table_3x3": contingency(pairs3, ("C", "WSTAR", "GREY"), ("C", "WSTAR", "GREY")),
            "table_5x4_unrolled": contingency(pairs54, STATES, cols),
        }
    contested = out["faithful"]["band"] != out["commit"]["band"]
    out["verdict"] = "LAYER_AGREEMENT_CONTESTED" if contested else out["faithful"]["band"]
    out["reference_line_not_a_band"] = {"GATE_AGREE_MIN_FRAC": GATE_AGREE_MIN_FRAC,
                                        "as_disagree_frac": round(1 - GATE_AGREE_MIN_FRAC, 6)}
    return out


# ------------------------------------------------------------------ per-cell processing
def process_cell(cell, census, dist):
    out = {"cell": cell}
    source_ok = bool(census) and census.get("decision") == "PASS" and bool(dist)
    sha_ok = source_ok and census["source_sha256"] == dist["source_provenance"]["source_sha256"]
    recs = dist.get("items", []) if dist else []
    ff = [r for r in recs if r["slot_id"] == "forced_final"]
    ctx_equal = True
    if source_ok and sha_ok:
        cmap = {(c["join_key"], c["direction"]): c for c in census["items"]}
        for r in ff:
            c = cmap.get((r["join_key"], r["direction"]))
            key = "ctx_counter" if r["turn2"] == "counter" else "ctx_neutral"
            if c is None or c[key] != r["ctx"]:
                ctx_equal = False
                print(f"[{cell}] CTX RECOMPUTE MISMATCH at {r['join_key']!r} {r['direction']}/{r['turn2']}",
                      flush=True)
                break
    integrity_ok = source_ok and sha_ok and ctx_equal
    mism = [r for r in recs if not (r["prompt_roundtrip_ok"] and r["bos_singleton_ok"]
                                    and r["prompt_rebuild_identical"])]
    n_flips = dist.get("n_conf_proxy_sign_flips", 0) if dist else 0
    fid = replay_fidelity(integrity_ok, bool(mism), n_flips)
    out["replay_fidelity"] = {"verdict": fid, "n_conf_proxy_sign_flips": n_flips,
                              "n_record_mismatches": len(mism),
                              "sha_cross_check_ok": sha_ok, "ctx_recompute_equal": ctx_equal,
                              "failing_records": [{k: r[k] for k in ("q", "direction", "turn2", "slot_id",
                                                                     "prompt_roundtrip_ok", "bos_singleton_ok",
                                                                     "prompt_rebuild_identical")}
                                                  for r in mism[:20]]}
    prov_ok = True
    if dist:
        try:
            validate_provenance(dist.get("provenance") or {})
        except ProvenanceIncomplete as e:
            prov_ok = False
            out["provenance_failure"] = f"PROVENANCE_PER_ARTIFACT_ABSENT: {e}"
    voided = fid in ("SOURCE_MISSING", "PROMPT_REPLAY_MISMATCH")

    # exactly-one-primary assertion (section 12) on the dist records
    if recs:
        combos = {(r["direction"], r["turn2"], r["slot_id"], r["variant_set"], r["state_rule"], r["register"])
                  for r in recs if r["readout_role"] == "primary"}
        assert combos == {("fold", "counter", "forced_final", "canonical", "S", "state_first_tok")}, combos

    base_half = cell.endswith("base")
    out["half"] = "base" if base_half else "it"
    out["half_stamp"] = "CONTEXT_CONTAMINATED_MEASURED (SECONDARY half)" if base_half else "primary half"
    out["h1_stamp"] = "every listen record and verdict below is LISTEN_CONTINGENT_ON_H1"

    per_axis = {}
    for d in DIRECTIONS:
        for a in ARMS:
            key = f"{d}/{a}"
            da = [r for r in recs if r["direction"] == d and r["turn2"] == a]
            block = {}
            ff_da = [r for r in da if r["slot_id"] == "forced_final"]
            if ff_da:
                v, n_clean = ctx_verdict([bool(r["ctx"]["ctx_clean"]) for r in ff_da])
                block["ctx"] = {"verdict": v, "n_ctx_clean": n_clean}
            for s in SLOTS:
                rs = [r for r in da if r["slot_id"] == s]
                if rs:
                    sv = state_vector(rs)
                    clean = [r for r in rs if (r.get("ctx") or {}).get("ctx_clean")]
                    sv["ctx_clean_subset"] = ({"n": len(clean), **{k: v for k, v in state_vector(clean).items()
                                                                   if k in ("states", "collapsed")}}
                                              if clean else {"n": 0})
                    block[f"state_vector_{s}"] = sv
            if voided or not ff_da:
                block["layers"] = {"verdict": "LAYERS_UNEVALUABLE",
                                   "reason": fid if voided else "no records"}
            elif not prov_ok:
                block["layers"] = {"verdict": "LAYERS_UNEVALUABLE",
                                   "reason": "PROVENANCE_PER_ARTIFACT_ABSENT (section 11.1)"}
            else:
                lf = "faithful_elicit" if a == "counter" else "faithful_neutral_elicit"
                block["layers"] = layers_block(ff_da, lf)
                if fid == "CONF_PROXY_SIGN_UNSTABLE":
                    block["layers"]["downgrade_stamp"] = (f"CONF_PROXY_SIGN_UNSTABLE: {n_flips} flips -- "
                                                          "quote no verdict without this count")
            # section 9.5 chains
            if da and not voided:
                by_item = {}
                for r in da:
                    by_item.setdefault(r["join_key"], {})[r["slot_id"]] = r
                p01 = [(v["single"], v["second_turn"]) for v in by_item.values()
                       if "single" in v and "second_turn" in v]
                p12 = [(v["second_turn"], v["forced_final"]) for v in by_item.values()
                       if "second_turn" in v and "forced_final" in v]
                block["chain"] = {
                    "T01": transition_matrix(p01), "T12": transition_matrix(p12),
                    "n_state_constant_along_chain": sum(
                        1 for v in by_item.values()
                        if len({v[s]["state"] for s in SLOTS if s in v}) == 1 and len(v) == 3),
                }
            block["stamp"] = make_stamp(
                d, f"slot forced_final verdict block, arm(turn2)={a}, sources per record",
                "faithful-strict primary; commit family banded beside it (section 9.4)",
                "False (STRICT_FIELDS register: the constrained forced-final slot)")
            per_axis[key] = block
    out["per_axis"] = per_axis
    return out


def run(outdir="out"):
    od = Path(outdir)
    cells = {}
    for cell in CELLS:
        cen_p = od / f"forcedfinal_census_{cell}.json"
        dist_p = od / f"forcedfinal_dist_ff_ext2_{cell}.json"
        census = json.loads(cen_p.read_text()) if cen_p.exists() else None
        dist = json.loads(dist_p.read_text()) if dist_p.exists() else None
        if census is None:
            print(f"[{cell}] census absent -> SOURCE_MISSING", flush=True)
        if dist is None:
            print(f"[{cell}] dist artifact absent -> LAYERS_UNEVALUABLE", flush=True)
        cells[cell] = process_cell(cell, census, dist)

    def triple(half_cells):
        return [cells[c]["per_axis"]["fold/counter"].get("layers", {}).get("verdict", "LAYERS_UNEVALUABLE")
                if cells[c].get("per_axis") else "LAYERS_UNEVALUABLE" for c in half_cells]

    it_triple, base_triple = triple(IT_HALF), triple(BASE_HALF)
    out = {
        "instrument": "forcedfinal_join", "registration": "REGISTRATION_forcedfinal_distributional.md",
        "thresholds": {"CONCORDANT_MAX": CONCORDANT_MAX, "DISCORDANT_MIN": DISCORDANT_MIN,
                       "integer_cuts_at_82": "<=8 / 9-24 / >=25", "quorum": ">=2 of 3 scales, no contradiction"},
        "decision_rule": DECISION_RULE,
        "cells": cells,
        "primary": {
            "axis": {"slot": "forced_final", "direction": "fold", "arm": "counter", "key": "canonical",
                     "rule": "S", "statistic": "the section 9.4 LAYERS_* verdict", "half": "-it"},
            "it_triple_2b_9b_27b": it_triple,
            "round_verdict_it": round_verdict(it_triple),
            "headline": ("the -it triple, quoted as a triple or not at all: "
                         f"({', '.join(it_triple)}) -> {round_verdict(it_triple)}"),
        },
        "secondary": {
            "base_triple_2b_9b_27b": base_triple,
            "round_verdict_base": round_verdict(base_triple),
            "stamp": "CONTEXT_CONTAMINATED_MEASURED on every base slot-2 number (section 6.4)",
        },
        "prohibitions": ("no base-vs-it contrast at any slot (section 6.5); no neutral->counter or "
                         "fold->listen transition (section 2); listen verdicts LISTEN_CONTINGENT_ON_H1 "
                         "(section 1.2); nothing here restores a withdrawn number"),
    }
    od.mkdir(parents=True, exist_ok=True)
    p = od / "forcedfinal_join.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n[join] PRIMARY -it triple: {it_triple} -> {round_verdict(it_triple)}", flush=True)
    print(f"[join] secondary base triple: {base_triple} -> {round_verdict(base_triple)} "
          f"(CONTEXT_CONTAMINATED_MEASURED)", flush=True)
    print(f"[join] -> {p}", flush=True)


# ------------------------------------------------------------------ selftest
def selftest():
    ok = 0

    def ck(cond, msg):
        nonlocal ok
        assert cond, msg
        ok += 1

    # section 9.4 bands at the exact integer edges (8/9/24/25 of 82)
    ck(band_layers(8) == "LAYERS_CONCORDANT", "8/82 concordant (inclusive)")
    ck(band_layers(9) == "LAYERS_PARTIAL", "9/82 partial")
    ck(band_layers(24) == "LAYERS_PARTIAL", "24/82 partial")
    ck(band_layers(25) == "LAYERS_DISCORDANT", "25/82 discordant (strict >)")
    ck(band_layers(0) == "LAYERS_CONCORDANT" and band_layers(82) == "LAYERS_DISCORDANT", "extremes")
    # section 4.4 same edges
    ck(band_state_variant(8) == "STATE_VARIANT_STABLE" and band_state_variant(25) == "STATE_VARIANT_DEPENDENT"
       and band_state_variant(9) == "STATE_VARIANT_PARTIAL", "state-variant bands")

    # planted 9.4: all-identical -> CONCORDANT; all-flipped -> DISCORDANT; offsetting case
    def rec(state, lbl, i, d="fold", a="counter"):
        return {"state": state, "state_set": state, "state_agrees_with_argmax": True,
                "direction": d, "turn2": a, "slot_id": "forced_final", "join_key": f"q{i}",
                "faithful_elicit": lbl, "commit_elicit": FAITHFUL_TO_COMMIT[lbl],
                "faithful_neutral_elicit": lbl, "commit_neutral_elicit": FAITHFUL_TO_COMMIT[lbl],
                "key": "space",
                "measure": {"argmax_in_union": True, "argmax_in_V_C": state == "FAVOURS_C",
                            "argmax_in_V_W": state == "FAVOURS_WSTAR", "argmax_tok_str": "x",
                            "entities": {"C": {"space": {"tie_plateau": 1, "rank_resolved": True}}},
                            "first_token_collision_space": False, "first_token_collision_bare": False},
                "ctx": {"ctx_clean": False}}
    same = [rec("FAVOURS_C", "C", i) for i in range(N_ITEMS)]
    lb = layers_block(same, "faithful_elicit")
    ck(lb["verdict"] == "LAYERS_CONCORDANT", "all-identical -> CONCORDANT (agreement is reportable)")
    flipped = [rec("FAVOURS_C", "WSTAR", i) for i in range(N_ITEMS)]
    lb = layers_block(flipped, "faithful_elicit")
    ck(lb["verdict"] == "LAYERS_DISCORDANT", "all-flipped -> DISCORDANT")
    # offsetting: marginals identical (41 C / 41 WSTAR each layer) but every item disagrees
    offset = [rec("FAVOURS_C", "WSTAR", i) for i in range(41)] + \
             [rec("FAVOURS_WSTAR", "C", 41 + i) for i in range(41)]
    lb = layers_block(offset, "faithful_elicit")
    ck(lb["faithful"]["n_disagree"] == 82 and lb["faithful"]["band"] == "LAYERS_DISCORDANT",
       "offsetting case: per-item statistic independent of the marginals")
    t = lb["faithful"]["table_3x3"]
    ck(sum(sum(v.values()) for v in t.values()) == 82, "3x3 sums to n")
    ck(sum(sum(v.values()) for v in lb["faithful"]["table_5x4_unrolled"].values()) == 82, "5x4 sums to n")

    # LAYER_AGREEMENT_CONTESTED: faithful and commit families in different bands
    contested = [dict(rec("FAVOURS_C", "C", i), commit_elicit="wrong") for i in range(N_ITEMS)]
    lb = layers_block(contested, "faithful_elicit")
    ck(lb["verdict"] == "LAYER_AGREEMENT_CONTESTED", "families in different bands -> CONTESTED")

    # UNEVALUABLE on absent label
    holed = [dict(rec("FAVOURS_C", "C", i)) for i in range(N_ITEMS)]
    holed[0]["faithful_elicit"] = None
    ck(layers_block(holed, "faithful_elicit")["verdict"] == "LAYERS_UNEVALUABLE", "absent label -> UNEVALUABLE")

    # section 9.1 categories + precedence (both mismatch and flips -> the EARLIER branch)
    ck(replay_fidelity(False, True, 5) == "SOURCE_MISSING", "source beats all")
    ck(replay_fidelity(True, True, 5) == "PROMPT_REPLAY_MISMATCH", "mismatch beats flips (earlier wins)")
    ck(replay_fidelity(True, False, 1) == "CONF_PROXY_SIGN_UNSTABLE", "flips downgrade")
    ck(replay_fidelity(True, False, 0) == "REPLAY_FAITHFUL", "clean")

    # section 9.2 categories
    ck(ctx_verdict([True] * 82)[0] == "CTX_CLEAN_ALL", "clean all")
    ck(ctx_verdict([False] * 82)[0] == "CTX_CONTAMINATED_ALL", "contaminated all")
    ck(ctx_verdict([True] + [False] * 81)[0] == "CTX_MIXED", "one record on the other side flips the branch")

    # section 9.6 categories + quorum edges
    ck(round_verdict(["LAYERS_CONCORDANT"] * 3) == "ROUND_CONCORDANT", "3/3 concordant")
    ck(round_verdict(["LAYERS_CONCORDANT", "LAYERS_CONCORDANT", "LAYERS_PARTIAL"]) == "ROUND_CONCORDANT",
       "2/3 + no contradiction")
    ck(round_verdict(["LAYERS_CONCORDANT", "LAYERS_CONCORDANT", "LAYERS_DISCORDANT"]) == "ROUND_MIXED",
       "quorum met but contradicted -> MIXED")
    ck(round_verdict(["LAYERS_DISCORDANT", "LAYERS_DISCORDANT", "LAYERS_PARTIAL"]) == "ROUND_DISCORDANT",
       "2/3 discordant")
    ck(round_verdict(["LAYERS_UNEVALUABLE", "LAYERS_UNEVALUABLE", "LAYERS_CONCORDANT"]) == "ROUND_UNEVALUABLE",
       "<2 evaluable -> UNEVALUABLE")
    ck(round_verdict(["LAYERS_PARTIAL", "LAYERS_PARTIAL", "LAYERS_PARTIAL"]) == "ROUND_MIXED", "no quorum -> MIXED")

    # section 9.5: the transition builder RAISES on cross-arm and cross-direction pairs
    a = rec("FAVOURS_C", "C", 0)
    b = rec("FAVOURS_WSTAR", "C", 0)
    m = transition_matrix([(a, b)])
    ck(m["FAVOURS_C"]["FAVOURS_WSTAR"] == 1, "within-chain transition counted")
    for mut in ({"turn2": "neutral"}, {"direction": "listen"}):
        try:
            transition_matrix([(a, dict(b, **mut))])
            raise AssertionError("must raise")
        except ValueError:
            ok += 1

    # provenance rejection reused from dist (null load-bearing, absent object)
    try:
        validate_provenance({})
        raise AssertionError("must raise")
    except ProvenanceIncomplete:
        ok += 1
    ck(tuple(PROVENANCE_LOAD_BEARING) == ("lambda_instance_id", "started_utc"), "load-bearing pair fixed")

    # gen_collapse via the SHIPPED map
    ck(gen_collapse("C") == "C" and gen_collapse("WSTAR") == "WSTAR", "identity legs")
    ck(gen_collapse("NEITHER") == "GREY" and gen_collapse("UNRESOLVED_ALIAS") == "GREY", "grey legs")

    print(f"[selftest] forcedfinal_join: {ok} asserts passed", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--outdir", default="out")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.run:
        run(a.outdir)
    else:
        ap.error("one of --selftest / --run is required")


if __name__ == "__main__":
    main()
