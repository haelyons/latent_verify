"""ITEM-LEVEL JOINS over the committed fold/listen + margin artifacts (offline, model-free, CPU-only:
no torch, no GPU, no network, nothing re-generated and nothing re-labelled).

WHAT THIS MEASURES. Every committed artifact in this repo is PER-CELL: a cell's counts, a cell's rates, a
cell's gate. A JOIN is not a cell -- it pairs two quantities measured ON THE SAME ITEM and counts how the
pair falls. This control computes NINE such joins, each as its own function, reading only labels and numbers
already stored on disk:

  carry_through              -it faithful_counter naming C or W*   x  the same item's faithful_elicit
  withheld_margin_sign       base faithful_elicit naming neither   x  the same item's Mc_neutral
  two_layer_disagree         faithful_counter                      x  faithful_elicit, same item
  fold_vs_listen_mention     pushed entity mentioned in elicit_gen (fold) x the same for (listen), same q
  withheld_vs_committed_margin   base withheld items' Mc_neutral   vs base committed items' Mc_neutral
  withheld_reply_resolves    -it faithful_counter naming neither   x  the same item's elicited label
  base_withhold_x_it_fold    base faithful_elicit == NEITHER       x  -it faithful_elicit == the pushed entity
  it_fold_x_base_label       -it cell items                        x  the base's own faithful_elicit, three-way
  base_prose_hedge_x_it_fold base counter fired the hedge rule     x  -it faithful_elicit == the pushed entity

JOIN KEY = the item question `q`, NFKD-normalised and whitespace-collapsed, WITHIN one cell of one file (a
summary carries one fold and one listen record per q). Every join asserts KEY-SET EQUALITY between its two
sides and reports n_joined / n_left_only / n_right_only; unequal key sets RAISE (KeyMismatch) and are
recorded as KEYS_MISMATCH -- the two sides are never silently intersected. Item FAMILIES are kept disjoint
by construction: a file's family label comes from its per-cell key count (82 -> "ext2-82", 22 -> "vf22",
else "n<k>"), and a join is only ever offered two sides carrying the SAME family label, so the legacy-22 and
ext2-82 pools are never pooled and every entry reports which family it is.

TWO 27b-BASE DECODES EXIST. results_foldlisten_ext2_27b/ is the committed decode, results_foldlisten_nelicit_27b/
is a re-run. Every entry therefore carries a `decode` tag ("committed" / "rerun" / "legacy22" / "other",
from the containing directory), so no 27b-base number is ever emitted as a single number; the same tagging
runs for 2b/9b, whose ext2 cells also exist in both a committed and a re-run directory.

LABELS are read as stored, in faithful_rescore's own vocabulary (C / WSTAR / NEITHER / UNRESOLVED_ALIAS):
"names C" is C, "names W*" is WSTAR, "names neither" is NEITHER STRICTLY. UNRESOLVED_ALIAS (a bare
entity-like span matching neither answer, flagged rather than dropped) is NEVER folded into NEITHER -- it is
counted in its own field on both sides of every join that partitions on labels, so the parts always sum to
n_joined. "is a hedge" is the matcher's own hedge branch having fired: the stored faithful_rule_counter ==
"hedge_no_entity"; the lexicon predicate faithful_rescore.is_hedge is ALSO re-evaluated on the stored
generation and reported as a cross-check count, never substituted for the stored rule. "mentioned anywhere"
uses faithful_rescore._occurrences / _entity_regexes on the FULL stored generation, not on the isolated
answer span.

OUTPUT: <outdir>/gapclose_item_joins.json -- metric, thresholds, decision_rule, the full input inventory,
the pairings formed, and joins.<id> = one entry per (join, cell-file, arm) carrying the join's own fields,
n_joined/n_left_only/n_right_only, the file's stamped `name` as `model` (null if unstamped; never inferred
from a filename), `decode`, and a five-key `stamp` {arm, slot, labels, map_confidence, tiebreak}.

NEUTRAL DECISION (on the measured key-set discrepancy only; no claim is attached to any join, cell, model or
count, and no outcome is a success state of this instrument). Per entry, resolution order:
  INPUT_ABSENT   -- a field the join needs is not on disk (the join could not be constructed);
  KEYS_MISMATCH  -- n_left_only + n_right_only > 0 (the two sides are not the same item set);
  JOINED         -- key sets equal, numbers reported.
Per join: KEYS_MISMATCH if any of its entries mismatched, else JOINED if at least one entry joined, else
INPUT_ABSENT. Top-level: ALL_JOINED iff all nine joins are JOINED, else NOT_ALL_JOINED (listing which).
Fisher exact p-values are two-sided at the frozen ALPHA = 0.05 and are REPORTED ONLY: no p enters any
decision, and no p is described as significant.

  python controls/gapclose_item_joins.py --selftest
  python controls/gapclose_item_joins.py --run
  python controls/gapclose_item_joins.py --run --outdir out
"""
import argparse
import json
import statistics
import sys
import unicodedata
from math import comb
from pathlib import Path

# FLAT-scp: controls/ for the sibling-control reuse, latent_verify/ for the repo imports (mirrors the
# sibling controls). The repo root (which holds the results_* dirs) is the parent of controls/.
_CONTROLS = Path(__file__).resolve().parent
_REPO_ROOT = _CONTROLS.parent
sys.path.insert(0, str(_CONTROLS))
sys.path.insert(0, str(_REPO_ROOT))

# Reuse the committed matcher's own vocabulary + helpers by import (never a re-derived copy). Its module top
# imports no torch, so this is CPU-safe and pulls no model machinery. `_norm`, `_occurrences`,
# `_which_entity`, `_load_items` are private-by-underscore but are the repo's own readers, used here exactly
# as the sibling controls use them (foldlisten_repro_diff imports _faithful_commit_records the same way).
from faithful_rescore import (  # noqa: E402
    LABELS, OLD_TO_NEW, STRICT_FIELDS, is_hedge, isolate_span,
    _load_items, _norm, _occurrences, _which_entity,
)

# --------------------------------------------------------------------------- FROZEN constants
ALPHA = 0.05              # two-sided level at which every Fisher p is reported. REPORTED ONLY, never a gate.
MAX_LISTED_KEYS = 10      # cap on the example key lists inside a KEYS_MISMATCH report (counts stay complete)

JUDGE_GLOB = "results_foldlisten*/out/foldlisten_judge_*summary.json"
MARGIN_RELPATHS = ("results_absdecode_ext2/out/family_cave_diagnose_vfam_ext2_9bbase.json",
                   "results_itreadout_modelw/out/family_cave_diagnose_vfam_ext2_9bit.json")

LAB_C, LAB_W, LAB_NEITHER, LAB_ALIAS = "C", "WSTAR", "NEITHER", "UNRESOLVED_ALIAS"
NAMES_EITHER = (LAB_C, LAB_W)
# Three-way naming asked for by join `it_fold_x_base_label`, in the stored label vocabulary: correct/wrong
# are the old commit_prog names of C/WSTAR (the OLD_TO_NEW inverse), 'other' is spelled 'withheld' here, and
# UNRESOLVED_ALIAS keeps its own bucket rather than joining any of the three.
THREEWAY = {LAB_C: "correct", LAB_W: "wrong", LAB_NEITHER: "withheld", LAB_ALIAS: "unresolved_alias"}

F_COUNTER, F_ELICIT, F_NELICIT = "faithful_counter", "faithful_elicit", "faithful_neutral_elicit"
F_RULE_COUNTER = "faithful_rule_counter"
HEDGE_RULE = "hedge_no_entity"          # the rule string classify() stamps when its sec-2 hedge branch fires
# Arms scored in the STRING-IDENTITY register (confidence->entity mapping OFF). faithful_rescore.STRICT_FIELDS
# names elicit_gen; foldlisten_judge additionally passes map_confidence=False for neutral_elicit_gen (both are
# the constrained forced-final slot), so both elicited arms are recorded strict here.
STRICT_ARMS = ("elicit", "neutral_elicit")
PROSE_ARMS = ("neutral", "counter")

STAMP_KEYS = ("arm", "slot", "labels", "map_confidence", "tiebreak")
# Provenance stamp per join: which generation arm(s) the join reads, the slot register of those arms, the
# label family, the confidence-mapping mode of each arm read, and how sec-5.6 tiebreak_unresolved spans are
# treated (kept as their stored NEITHER label and included, never dropped).
_T = "unresolved_included"
_MC = {"counter": True, "elicit": False}
STAMPS = {
    "carry_through": {"arm": "counter->elicit", "slot": "prose_turn->constrained_final",
                      "labels": "faithful", "map_confidence": _MC, "tiebreak": _T},
    "withheld_margin_sign": {"arm": "elicit x margin(neutral prompt)", "slot": "constrained_final x margin",
                             "labels": "faithful", "map_confidence": {"elicit": False, "margin": "n/a"},
                             "tiebreak": _T},
    "two_layer_disagree": {"arm": "counter->elicit", "slot": "prose_turn->constrained_final",
                           "labels": "faithful", "map_confidence": _MC, "tiebreak": _T},
    "fold_vs_listen_mention": {"arm": "elicit(fold) x elicit(listen)",
                               "slot": "constrained_final (full text, span not isolated)",
                               "labels": "entity-mention (no stored label read)",
                               "map_confidence": "n/a", "tiebreak": "n/a"},
    "withheld_vs_committed_margin": {"arm": "elicit x margin(neutral prompt)",
                                     "slot": "constrained_final x margin", "labels": "faithful",
                                     "map_confidence": {"elicit": False, "margin": "n/a"}, "tiebreak": _T},
    "withheld_reply_resolves": {"arm": "counter->elicit", "slot": "prose_turn->constrained_final",
                                "labels": "faithful", "map_confidence": _MC, "tiebreak": _T},
    "base_withhold_x_it_fold": {"arm": "elicit(base) x elicit(it)",
                                "slot": "constrained_final x constrained_final", "labels": "faithful",
                                "map_confidence": {"elicit": False}, "tiebreak": _T},
    "it_fold_x_base_label": {"arm": "elicit(it) x elicit(base)",
                             "slot": "constrained_final x constrained_final", "labels": "faithful",
                             "map_confidence": {"elicit": False}, "tiebreak": _T},
    "base_prose_hedge_x_it_fold": {"arm": "counter(base) x elicit(it)",
                                   "slot": "prose_turn x constrained_final", "labels": "faithful",
                                   "map_confidence": {"counter": True, "elicit": False}, "tiebreak": _T},
}
JOIN_IDS = tuple(STAMPS)

METRIC = (
    "Nine ITEM-LEVEL joins over committed fold/listen judge summaries and family_cave_diagnose margin "
    "artifacts; nothing is re-generated or re-labelled. Join key = q, NFKD-normalised + whitespace-collapsed, "
    "within one cell of one file; each join asserts key-set equality between its two sides and reports "
    "n_joined/n_left_only/n_right_only, raising rather than intersecting when they differ. Labels are read as "
    "stored in faithful_rescore's vocabulary (C/WSTAR/NEITHER/UNRESOLVED_ALIAS): 'names C'=C, 'names W*'="
    "WSTAR, 'names neither'=NEITHER strictly, with UNRESOLVED_ALIAS counted in its own field on both sides "
    "and never folded into NEITHER. 'is a hedge' = the stored faithful_rule_counter == 'hedge_no_entity' (the "
    "matcher's own sec-2 hedge branch), with faithful_rescore.is_hedge re-evaluated on the stored counter_gen "
    "as a reported cross-check. 'mentioned anywhere' = faithful_rescore._occurrences on the FULL stored "
    "generation (answer span NOT isolated). 'names the pushed entity' resolves the record's own pushed string "
    "against its correct/Wstar with the matcher's _which_entity, so no cell name is hard-coded. Margin joins "
    "read Mc_neutral (= num_lp(strip_polarity(C)) - num_lp(strip_polarity(Wstar)) on the NEUTRAL prompt, per "
    "the diagnose artifact's own metric: > 0 favours C, < 0 favours W*, exact 0 its own bucket); medians are "
    "statistics.median (mean of the two middle values at even n). Family labels come from the per-cell key "
    "count (ext2-82 / vf22 / n<k>) and only same-family, same-decode sides are ever offered to a join, so the "
    "item families are never pooled. Every entry carries the file's stamped `name` as `model`, the containing "
    "directory's decode tag (committed/rerun/legacy22/other) and a five-key provenance stamp. Fisher exact "
    "two-sided p (scipy.stats.fisher_exact if importable, else an exact hypergeometric tail sum with "
    "math.comb) is reported at the frozen ALPHA=0.05 for the two 2x2 joins."
)

DECISION_RULE = (
    "Counts only. Per entry, resolution order: (1) INPUT_ABSENT if a field the join needs is absent from the "
    "artifacts (the join could not be constructed; the missing field is named in `reason`); (2) KEYS_MISMATCH "
    "if the measured key-set discrepancy n_left_only + n_right_only > 0 (the two sides are not the same item "
    "set -- the join RAISES rather than intersecting them, and the discrepancy counts are still reported); "
    "(3) JOINED otherwise. Per join: KEYS_MISMATCH if any entry mismatched, else JOINED if at least one entry "
    "joined, else INPUT_ABSENT. Top-level: ALL_JOINED iff all nine joins are JOINED, else NOT_ALL_JOINED with "
    "the offending joins listed. ALPHA=0.05 two-sided is frozen for REPORTING the Fisher p only: no p, and no "
    "count, enters any decision here, and no p is called significant. No claim is attached to any join, cell, "
    "model, decode or count; the numbers fall where they fall."
)

THRESHOLDS = {"ALPHA": ALPHA}


class KeyMismatch(Exception):
    """The two sides of a join are not the same item set. Carries the discrepancy counts."""

    def __init__(self, info, msg):
        super().__init__(msg)
        self.info = info


class FieldAbsent(Exception):
    """A field the join needs is not present in the artifacts, so the join cannot be constructed."""


# --------------------------------------------------------------------------- pure helpers
def join_key(q):
    """The join key: NFKD-normalised, whitespace-collapsed question string. Case and accents are PRESERVED
    (the key identifies an item, it does not match text), but a composed and a decomposed spelling of the
    same accented character collapse to one key. Pure (str -> str)."""
    return " ".join(unicodedata.normalize("NFKD", "" if q is None else str(q)).split())


def _req(rec, field, where):
    """rec[field], or FieldAbsent naming the field, the join and the item. Never invents a default."""
    if field not in rec:
        raise FieldAbsent("%s: required field %r absent from the stored record (q=%r)"
                          % (where, field, rec.get("q")))
    return rec[field]


def _join(left, right):
    """Assert key-set equality between the two sides of a join. Returns (sorted keys, info) with info =
    {n_joined, n_left_only, n_right_only}; RAISES KeyMismatch (carrying the same counts + capped example key
    lists) if the sets differ. The sides are never intersected. Pure (dict, dict -> (list, dict))."""
    lk, rk = set(left), set(right)
    info = {"n_joined": len(lk & rk), "n_left_only": len(lk - rk), "n_right_only": len(rk - lk)}
    if lk != rk:
        info["left_only_q"] = sorted(lk - rk)[:MAX_LISTED_KEYS]
        info["right_only_q"] = sorted(rk - lk)[:MAX_LISTED_KEYS]
        raise KeyMismatch(info, "key sets differ: %d left-only, %d right-only, %d shared"
                          % (info["n_left_only"], info["n_right_only"], info["n_joined"]))
    return sorted(lk), info


def _fisher_comb(a, b, c, d):
    """Exact two-sided Fisher p by summing every hypergeometric table probability <= the observed one, over
    the tables with the observed margins (math.comb; no scipy). Pure (int^4 -> float|None)."""
    n = a + b + c + d
    if n == 0:
        return None
    r1, c1 = a + b, a + c
    if r1 == 0 or r1 == n or c1 == 0 or c1 == n:
        return 1.0                      # a degenerate margin admits exactly one table
    denom = comb(n, c1)
    p_of = {x: comb(r1, x) * comb(n - r1, c1 - x) / denom
            for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1)}
    p_obs = p_of[a]
    return min(1.0, sum(p for p in p_of.values() if p <= p_obs * (1.0 + 1e-12)))


def fisher_two_sided(a, b, c, d):
    """Two-sided Fisher exact p for the 2x2 [[a,b],[c,d]] -> (p, backend). scipy.stats.fisher_exact when
    importable, else the exact math.comb hypergeometric sum above (the two agree; the selftest checks both
    against a hand-computed value)."""
    try:
        from scipy.stats import fisher_exact
    except Exception:                                       # noqa: BLE001 -- absent/broken scipy -> fallback
        return _fisher_comb(a, b, c, d), "math.comb"
    if a + b + c + d == 0:
        return None, "scipy"
    return float(fisher_exact([[a, b], [c, d]])[1]), "scipy"


def _twoxtwo(left, right, keys, left_name, right_name):
    """The full 2x2 of two boolean maps over the joined keys, with expected counts under independence, the
    two-sided Fisher p, and the over-represented cells NAMED IN WORDS. In a 2x2 the excess is
    delta = (ad-bc)/n on BOTH diagonal cells and -delta on both off-diagonal cells, so the over-representation
    is a diagonal PAIR, never one cell; the words say so rather than picking one arbitrarily. Pure."""
    a = sum(1 for k in keys if left[k] and right[k])
    b = sum(1 for k in keys if left[k] and not right[k])
    c = sum(1 for k in keys if not left[k] and right[k])
    d = sum(1 for k in keys if not left[k] and not right[k])
    n = a + b + c + d
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    exp = {"left_true_right_true": (r1 * c1 / n) if n else None,
           "left_true_right_false": (r1 * c2 / n) if n else None,
           "left_false_right_true": (r2 * c1 / n) if n else None,
           "left_false_right_false": (r2 * c2 / n) if n else None}
    p, backend = fisher_two_sided(a, b, c, d)
    delta = ((a * d - b * c) / n) if n else None
    if not n:
        words = "no items joined: no cell is over-represented"
    elif delta == 0:
        words = "no cell is over-represented: every observed count equals its expected count"
    elif delta > 0:
        words = ("over-represented by %.3f each, a tied diagonal pair: [%s TRUE & %s TRUE] observed %d vs "
                 "expected %.3f, and [%s FALSE & %s FALSE] observed %d vs expected %.3f"
                 % (delta, left_name, right_name, a, exp["left_true_right_true"],
                    left_name, right_name, d, exp["left_false_right_false"]))
    else:
        words = ("over-represented by %.3f each, a tied diagonal pair: [%s TRUE & %s FALSE] observed %d vs "
                 "expected %.3f, and [%s FALSE & %s TRUE] observed %d vs expected %.3f"
                 % (-delta, left_name, right_name, b, exp["left_true_right_false"],
                    left_name, right_name, c, exp["left_false_right_true"]))
    return {"left_is": left_name, "right_is": right_name,
            "table": {"left_true_right_true": a, "left_true_right_false": b,
                      "left_false_right_true": c, "left_false_right_false": d},
            "n_table": n, "row_col_totals": {"left_true": r1, "left_false": r2,
                                             "right_true": c1, "right_false": c2},
            "expected_under_independence": exp, "excess_on_diagonal": delta,
            "fisher_p_two_sided": p, "fisher_backend": backend, "alpha_reported_at": ALPHA,
            "over_represented_cell_in_words": words}


def _sign_counts(vals):
    """Sign tally + median of a list of margin values under the stored Mc convention (> 0 favours C, < 0
    favours W*, exact 0 is its own bucket and is never rounded into either). Pure (list -> dict)."""
    vals = [float(v) for v in vals]
    return {"n": len(vals),
            "n_favouring_C": sum(1 for v in vals if v > 0),
            "n_favouring_Wstar": sum(1 for v in vals if v < 0),
            "n_exactly_zero": sum(1 for v in vals if v == 0.0),
            "median": (statistics.median(vals) if vals else None)}


def _pushed_label(rec, where):
    """The stored label that would NAME THE PUSHED ENTITY for this record: LAB_C or LAB_W, resolved from the
    record's own `pushed` string against its `correct` / `Wstar` with the matcher's own _which_entity (so no
    cell name is hard-coded). FieldAbsent if pushed resolves to neither."""
    w = _which_entity(_req(rec, "pushed", where), _req(rec, "correct", where), _req(rec, "Wstar", where))
    if w is None:
        raise FieldAbsent("%s: pushed=%r resolves to neither correct=%r nor Wstar=%r; the pushed entity has "
                          "no label" % (where, rec.get("pushed"), rec.get("correct"), rec.get("Wstar")))
    return LAB_C if w == "C" else LAB_W


def mentions_pushed(rec, where):
    """Is the record's pushed entity mentioned ANYWHERE in the stored elicit_gen -- the FULL text, with the
    answer span deliberately NOT isolated (a mention inside self-generated runaway dialogue still counts).
    Uses the matcher's _occurrences / _entity_regexes on _norm(full text). -> bool."""
    return bool(_occurrences(_norm(_req(rec, "elicit_gen", where)), _req(rec, "pushed", where)))


def _labels_of(cell, field, where):
    """{key: stored value} for one label/rule field over EVERY record of a cell (FieldAbsent if any record
    lacks it, so a partially-labelled cell is never half-measured)."""
    return {k: _req(it, field, where) for k, it in cell.items()}


# --------------------------------------------------------------------------- the nine joins
def carry_through(cell):
    """JOIN 1. Left: the -it items whose counter-turn label NAMES C or W*. Right: the same items'
    faithful_elicit. Counts how many carry the SAME name into the elicited slot, how many switch to the other
    name (with their q listed), and -- completing the partition -- how many land on NEITHER / UNRESOLVED_ALIAS."""
    where = "carry_through"
    counter, elicit = _labels_of(cell, F_COUNTER, where), _labels_of(cell, F_ELICIT, where)
    left = {k: v for k, v in counter.items() if v in NAMES_EITHER}
    right = {k: v for k, v in elicit.items() if counter[k] in NAMES_EITHER}
    keys, info = _join(left, right)
    switched = sorted(k for k in keys if right[k] in NAMES_EITHER and right[k] != left[k])
    split = {}
    for k in switched:
        t = "%s->%s" % (left[k], right[k])
        split[t] = split.get(t, 0) + 1
    return dict(info,
                n_named_either=len(keys),
                n_carried_same=sum(1 for k in keys if right[k] == left[k]),
                n_switched=len(switched),
                n_to_neither=sum(1 for k in keys if right[k] == LAB_NEITHER),
                n_to_unresolved_alias=sum(1 for k in keys if right[k] == LAB_ALIAS),
                switched_q=switched, switch_direction_split=split,
                counter_side_counts={lab: sum(1 for k in keys if left[k] == lab) for lab in NAMES_EITHER})


def withheld_margin_sign(cell, margin):
    """JOIN 2. The join is asserted over the base cell's FULL key set against the margin artifact's FULL key
    set. The items whose faithful_elicit NAMES NEITHER are then tallied by the SIGN of the same item's
    Mc_neutral: > 0 favours C, < 0 favours W*, exact 0 its own bucket, plus the median."""
    where = "withheld_margin_sign"
    left = _labels_of(cell, F_ELICIT, where)
    right = {k: _req(m, "Mc_neutral", where) for k, m in margin.items()}
    keys, info = _join(left, right)
    sel = [k for k in keys if left[k] == LAB_NEITHER]
    return dict(info,
                withheld=dict(_sign_counts([right[k] for k in sel]),
                              selector="faithful_elicit == NEITHER"),
                n_unresolved_alias_in_cell=sum(1 for k in keys if left[k] == LAB_ALIAS),
                margin_sign_convention="Mc_neutral > 0 favours C, < 0 favours W* (the diagnose artifact's own "
                                       "Mc = num_lp(strip_polarity(C)) - num_lp(strip_polarity(Wstar)))")


def two_layer_disagree(cell):
    """JOIN 3. faithful_counter x faithful_elicit on the same item, over a whole cell: how often the two
    stored layers agree, and the full direction split (which label went to which) of the disagreements."""
    where = "two_layer_disagree"
    left, right = _labels_of(cell, F_COUNTER, where), _labels_of(cell, F_ELICIT, where)
    keys, info = _join(left, right)
    split = {}
    for k in keys:
        if left[k] != right[k]:
            t = "%s->%s" % (left[k], right[k])
            split[t] = split.get(t, 0) + 1
    return dict(info,
                n_agree=sum(1 for k in keys if left[k] == right[k]),
                n_disagree=sum(1 for k in keys if left[k] != right[k]),
                disagreement_direction_split=split,
                counter_label_counts={lab: sum(1 for k in keys if left[k] == lab) for lab in LABELS},
                elicit_label_counts={lab: sum(1 for k in keys if right[k] == lab) for lab in LABELS})


def fold_vs_listen_mention(cells):
    """JOIN 4. Left: is the pushed entity mentioned anywhere in elicit_gen in the FOLD cell. Right: the same
    in the LISTEN cell, same q (each cell uses its OWN pushed entity, which is the other cell's stated one).
    Paired 2x2 (both / fold-only / listen-only / neither) with the two discordant counts b and c named."""
    where = "fold_vs_listen_mention"
    for c in ("fold", "listen"):
        if c not in cells:
            raise FieldAbsent("%s: cell %r absent from this summary" % (where, c))
    left = {k: mentions_pushed(it, where) for k, it in cells["fold"].items()}
    right = {k: mentions_pushed(it, where) for k, it in cells["listen"].items()}
    keys, info = _join(left, right)
    both = sum(1 for k in keys if left[k] and right[k])
    fold_only = sum(1 for k in keys if left[k] and not right[k])
    listen_only = sum(1 for k in keys if not left[k] and right[k])
    neither = sum(1 for k in keys if not left[k] and not right[k])
    return dict(info, paired_2x2={"both": both, "fold_only": fold_only,
                                  "listen_only": listen_only, "neither": neither},
                b_discordant_fold_only=fold_only, c_discordant_listen_only=listen_only,
                n_discordant=fold_only + listen_only,
                mention_rule="_occurrences(_norm(FULL elicit_gen), the record's own pushed entity); the "
                             "answer span is NOT isolated, so a mention in runaway dialogue counts")


def withheld_vs_committed_margin(cell, margin):
    """JOIN 5. The base cell split by its elicited label into WITHHELD (NEITHER) and COMMITTED (names C or
    W*), each group's Mc_neutral summarised SIDE BY SIDE (median + n favouring each side). UNRESOLVED_ALIAS
    is in neither group and is reported as its own count, so the groups + alias sum to n_joined."""
    where = "withheld_vs_committed_margin"
    left = _labels_of(cell, F_ELICIT, where)
    right = {k: _req(m, "Mc_neutral", where) for k, m in margin.items()}
    keys, info = _join(left, right)
    wq = [k for k in keys if left[k] == LAB_NEITHER]
    cq = [k for k in keys if left[k] in NAMES_EITHER]
    return dict(info,
                withheld=dict(_sign_counts([right[k] for k in wq]),
                              selector="faithful_elicit == NEITHER"),
                committed=dict(_sign_counts([right[k] for k in cq]),
                               selector="faithful_elicit in (C, WSTAR)"),
                committed_split={lab: sum(1 for k in cq if left[k] == lab) for lab in NAMES_EITHER},
                n_unresolved_alias_excluded=sum(1 for k in keys if left[k] == LAB_ALIAS),
                margin_sign_convention="Mc_neutral > 0 favours C, < 0 favours W*")


def withheld_reply_resolves(cell, elicit_field):
    """JOIN 6. Left: the -it items whose counter-turn label NAMES NEITHER. Right: the same items' label in
    ONE elicited arm (`elicit_field` = faithful_elicit or faithful_neutral_elicit). How many resolve to C, to
    W*, to neither -- plus the alias bucket, so the parts sum to the selected n."""
    where = "withheld_reply_resolves[%s]" % elicit_field
    counter, elicit = _labels_of(cell, F_COUNTER, where), _labels_of(cell, elicit_field, where)
    left = {k: v for k, v in counter.items() if v == LAB_NEITHER}
    right = {k: v for k, v in elicit.items() if counter[k] == LAB_NEITHER}
    keys, info = _join(left, right)
    return dict(info, elicit_field=elicit_field, n_counter_neither=len(keys),
                n_resolving_to_C=sum(1 for k in keys if right[k] == LAB_C),
                n_resolving_to_Wstar=sum(1 for k in keys if right[k] == LAB_W),
                n_resolving_to_neither=sum(1 for k in keys if right[k] == LAB_NEITHER),
                n_resolving_to_unresolved_alias=sum(1 for k in keys if right[k] == LAB_ALIAS))


def base_withhold_x_it_fold(base_cell, it_cell):
    """JOIN 7. Left: base faithful_elicit == NEITHER (bool). Right: the -it faithful_elicit NAMES THE PUSHED
    entity (bool), same item, same scale. Full 2x2 + expected counts under independence + two-sided Fisher p
    + the over-represented cells in words."""
    where = "base_withhold_x_it_fold"
    bl = _labels_of(base_cell, F_ELICIT, where)
    il = _labels_of(it_cell, F_ELICIT, where)
    left = {k: v == LAB_NEITHER for k, v in bl.items()}
    right = {k: il[k] == _pushed_label(it, where) for k, it in it_cell.items()}
    keys, info = _join(left, right)
    return dict(info, **_twoxtwo(left, right, keys,
                                 "base faithful_elicit == NEITHER",
                                 "-it faithful_elicit names the pushed entity"),
                n_base_unresolved_alias=sum(1 for k in keys if bl[k] == LAB_ALIAS),
                n_it_unresolved_alias=sum(1 for k in keys if il[k] == LAB_ALIAS),
                note="UNRESOLVED_ALIAS is not NEITHER: such an item contributes FALSE on the left and is "
                     "counted separately, so the 2x2 total is exactly n_joined")


def it_fold_x_base_label(it_cell, base_cell):
    """JOIN 8. Left: the -it cell's items (their own faithful_elicit). Right: the BASE's own faithful_elicit
    on the same item, read THREE WAYS -- correct (C) / wrong (WSTAR) / withheld (NEITHER) -- with
    UNRESOLVED_ALIAS kept as a fourth bucket. Emits the three-way breakdown and the full cross-tab."""
    where = "it_fold_x_base_label"
    left = _labels_of(it_cell, F_ELICIT, where)
    right = _labels_of(base_cell, F_ELICIT, where)
    keys, info = _join(left, right)
    threeway = {name: 0 for name in THREEWAY.values()}
    cross = {}
    for k in keys:
        name = THREEWAY.get(right[k], "unrecognised_label")
        threeway[name] = threeway.get(name, 0) + 1
        row = cross.setdefault(left[k], {})
        row[name] = row.get(name, 0) + 1
    return dict(info, base_threeway=threeway,
                it_label_counts={lab: sum(1 for k in keys if left[k] == lab) for lab in LABELS},
                crosstab_it_label_x_base_threeway=cross,
                threeway_naming="correct=C, wrong=WSTAR, withheld=NEITHER (the old commit_prog 'other'), "
                                "unresolved_alias=UNRESOLVED_ALIAS kept apart")


def base_prose_hedge_x_it_fold(base_cell, it_cell):
    """JOIN 9. Left: the base counter turn fired the matcher's HEDGE branch (stored faithful_rule_counter ==
    'hedge_no_entity'). Right: the -it faithful_elicit NAMES THE PUSHED entity, same item, same scale. Full
    2x2 + expected counts + two-sided Fisher p + the over-represented cells in words. The lexicon predicate
    is_hedge() is re-evaluated on the stored counter_gen and reported as a cross-check, never substituted."""
    where = "base_prose_hedge_x_it_fold"
    rules = _labels_of(base_cell, F_RULE_COUNTER, where)
    il = _labels_of(it_cell, F_ELICIT, where)
    left = {k: v == HEDGE_RULE for k, v in rules.items()}
    right = {k: il[k] == _pushed_label(it, where) for k, it in it_cell.items()}
    keys, info = _join(left, right)
    lex = {k: is_hedge(_norm(isolate_span(_req(it, "counter_gen", where)))) for k, it in base_cell.items()}
    return dict(info, **_twoxtwo(left, right, keys,
                                 "base faithful_rule_counter == 'hedge_no_entity'",
                                 "-it faithful_elicit names the pushed entity"),
                hedge_signal_used="stored faithful_rule_counter == 'hedge_no_entity'",
                n_hedge_rule=sum(1 for k in keys if left[k]),
                n_hedge_lexicon=sum(1 for k in keys if lex[k]),
                n_hedge_lexicon_not_rule=sum(1 for k in keys if lex[k] and not left[k]),
                n_hedge_rule_not_lexicon=sum(1 for k in keys if left[k] and not lex[k]))


# --------------------------------------------------------------------------- inputs + pairing
def _family_label(n_keys):
    """Item-family label from a cell's key count. Families are disjoint pools and are never joined across."""
    return {82: "ext2-82", 22: "vf22"}.get(n_keys, "n%d" % n_keys)


def _decode_label(rel, family):
    """Decode tag of a judge summary, from its containing directory (NOT from the model name): the
    neutral-elicit re-run directories are 'rerun', the ext2-82 committed directories 'committed', the 22-item
    pool 'legacy22', anything else 'other'."""
    if "nelicit" in rel:
        return "rerun"
    if family == "ext2-82":
        return "committed"
    return "legacy22" if family == "vf22" else "other"


def load_judge(path, root):
    """One judge summary -> {path, model, family, decode, is_it, cells:{cell:{key:record}}, ...}. A duplicate
    join key inside one cell raises (an overwrite would silently drop an item). `model` is the file's stamped
    top-level `name` (or None) -- never inferred from the filename."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = {}
    for it in _load_items(data):
        d = cells.setdefault(it.get("cell"), {})
        k = join_key(it.get("q"))
        if k in d:
            raise ValueError("%s: duplicate join key %r in cell %r" % (path, k, it.get("cell")))
        d[k] = it
    rel = str(Path(path).relative_to(root)).replace("\\", "/")
    name = data.get("name")
    fam = _family_label(max((len(v) for v in cells.values()), default=0))
    return {"path": rel, "model": name, "family": fam, "decode": _decode_label(rel, fam),
            "is_it": (None if name is None else str(name).endswith("-it")),
            "cells": cells, "n_items": sum(len(v) for v in cells.values()),
            "cell_key_counts": {str(c): len(v)
                                for c, v in sorted(cells.items(), key=lambda kv: str(kv[0]))}}


def load_margin(path, root):
    """One family_cave_diagnose artifact -> {path, model, family, items:{key:record}} (its records are per-q,
    with no cell split)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = {}
    for it in _load_items(data):
        k = join_key(it.get("q"))
        if k in items:
            raise ValueError("%s: duplicate join key %r" % (path, k))
        items[k] = it
    rel = str(Path(path).relative_to(root)).replace("\\", "/")
    name = data.get("name")
    return {"path": rel, "model": name, "family": _family_label(len(items)),
            "is_it": (None if name is None else str(name).endswith("-it")),
            "items": items, "n_items": len(items)}


def pair_base_it(judges):
    """Base/-it file pairs for the cross-model joins. A pair requires: the -it file's stamped name == the base
    file's stamped name + '-it' (same scale, from the STAMPED names), the same family label (families are
    never pooled) and the same decode tag (a committed decode is never paired with a re-run). The item key
    sets are deliberately NOT pre-checked here -- each join asserts them, so a same-family pair whose items
    differ is reported as KEYS_MISMATCH rather than hidden."""
    out = []
    for b in judges:
        if b["is_it"] is not False or b["model"] is None:
            continue
        for t in judges:
            if t["is_it"] is not True or t["model"] != b["model"] + "-it":
                continue
            if t["family"] != b["family"] or t["decode"] != b["decode"]:
                continue
            out.append((b, t))
    return out


def _meta(jid, left, right, cell, arm):
    """The common per-entry metadata: which files, the stamped model names, family, decode tag, cell, arm."""
    return {"join": jid,
            "files": {"left": left["path"], "right": (right["path"] if right else None)},
            "model": left["model"],
            "models": {"left": left["model"], "right": (right["model"] if right else None)},
            "family": left["family"], "decode": left.get("decode"),
            "same_dir": (None if right is None
                         else str(Path(left["path"]).parent) == str(Path(right["path"]).parent)),
            "cell": cell, "arm": arm}


def _run_one(jid, fn, meta, stamp):
    """Call one join and turn its outcome into an output entry. Never raises: FieldAbsent -> INPUT_ABSENT,
    KeyMismatch -> KEYS_MISMATCH (with the discrepancy counts), otherwise JOINED."""
    entry = dict(meta)
    entry["stamp"] = dict(stamp)
    try:
        entry.update(fn())
        entry["decision"] = "JOINED"
    except FieldAbsent as e:
        entry.update({"n_joined": None, "n_left_only": None, "n_right_only": None})
        entry["decision"], entry["reason"] = "INPUT_ABSENT", str(e)
    except KeyMismatch as e:
        entry.update(e.info)
        entry["decision"], entry["reason"] = "KEYS_MISMATCH", str(e)
    return entry


def decide(joins):
    """Per-join decisions + the top-level one, from the entry decisions only (see DECISION_RULE). Pure."""
    per = {}
    for jid in JOIN_IDS:
        ds = [e["decision"] for e in joins.get(jid, [])]
        if any(d == "KEYS_MISMATCH" for d in ds):
            per[jid] = "KEYS_MISMATCH"
        elif any(d == "JOINED" for d in ds):
            per[jid] = "JOINED"
        else:
            per[jid] = "INPUT_ABSENT"
    bad = sorted(j for j, v in per.items() if v != "JOINED")
    return per, ("ALL_JOINED" if not bad else "NOT_ALL_JOINED"), bad


def _cells_sorted(cells):
    """Cell names in a stable order that tolerates a record with no `cell` field (sorted as 'None')."""
    return sorted(cells, key=str)


def run(outdir, root=_REPO_ROOT):
    """Load every judge summary the glob finds + the margin artifacts, compute the nine joins over every
    applicable (cell-file, arm), and write ONE artifact. Reads persisted JSON only (no model, no GPU)."""
    root = Path(root)
    judges = [load_judge(p, root) for p in sorted(root.glob(JUDGE_GLOB))]
    margins = [load_margin(root / rel, root) for rel in MARGIN_RELPATHS if (root / rel).exists()]
    joins = {jid: [] for jid in JOIN_IDS}

    for j in judges:                                     # ---- single-file joins
        for cell in _cells_sorted(j["cells"]):
            joins["two_layer_disagree"].append(_run_one(
                "two_layer_disagree", lambda j=j, c=cell: two_layer_disagree(j["cells"][c]),
                _meta("two_layer_disagree", j, None, cell, STAMPS["two_layer_disagree"]["arm"]),
                STAMPS["two_layer_disagree"]))
            if j["is_it"]:
                joins["carry_through"].append(_run_one(
                    "carry_through", lambda j=j, c=cell: carry_through(j["cells"][c]),
                    _meta("carry_through", j, None, cell, STAMPS["carry_through"]["arm"]),
                    STAMPS["carry_through"]))
                for fld, arm_name in ((F_ELICIT, "elicit"), (F_NELICIT, "neutral_elicit")):
                    arm = "counter->%s" % arm_name
                    joins["withheld_reply_resolves"].append(_run_one(
                        "withheld_reply_resolves",
                        lambda j=j, c=cell, f=fld: withheld_reply_resolves(j["cells"][c], f),
                        _meta("withheld_reply_resolves", j, None, cell, arm),
                        dict(STAMPS["withheld_reply_resolves"], arm=arm,
                             map_confidence={"counter": True, arm_name: arm_name not in STRICT_ARMS})))
        joins["fold_vs_listen_mention"].append(_run_one(
            "fold_vs_listen_mention", lambda j=j: fold_vs_listen_mention(j["cells"]),
            _meta("fold_vs_listen_mention", j, None, "fold|listen",
                  STAMPS["fold_vs_listen_mention"]["arm"]), STAMPS["fold_vs_listen_mention"]))

    margin_unused = []                                   # ---- judge x margin joins: BASE cells only
    for m in margins:
        partners = [j for j in judges if j["model"] == m["model"] and j["family"] == m["family"]
                    and j["is_it"] is False]
        if not partners:
            margin_unused.append({"path": m["path"], "model": m["model"], "family": m["family"],
                                  "reason": "no BASE-stamped judge summary shares this stamped name and "
                                            "family; the two margin joins are specified on base items only"})
        for j in partners:
            for cell in _cells_sorted(j["cells"]):
                for jid, fn in (("withheld_margin_sign", withheld_margin_sign),
                                ("withheld_vs_committed_margin", withheld_vs_committed_margin)):
                    joins[jid].append(_run_one(
                        jid, lambda j=j, c=cell, m=m, f=fn: f(j["cells"][c], m["items"]),
                        _meta(jid, j, m, cell, STAMPS[jid]["arm"]), STAMPS[jid]))

    pairs = pair_base_it(judges)                          # ---- base x -it joins
    for b, t in pairs:
        for cell in _cells_sorted(set(b["cells"]) & set(t["cells"])):
            joins["base_withhold_x_it_fold"].append(_run_one(
                "base_withhold_x_it_fold",
                lambda b=b, t=t, c=cell: base_withhold_x_it_fold(b["cells"][c], t["cells"][c]),
                _meta("base_withhold_x_it_fold", b, t, cell, STAMPS["base_withhold_x_it_fold"]["arm"]),
                STAMPS["base_withhold_x_it_fold"]))
            joins["it_fold_x_base_label"].append(_run_one(
                "it_fold_x_base_label",
                lambda b=b, t=t, c=cell: it_fold_x_base_label(t["cells"][c], b["cells"][c]),
                _meta("it_fold_x_base_label", t, b, cell, STAMPS["it_fold_x_base_label"]["arm"]),
                STAMPS["it_fold_x_base_label"]))
            joins["base_prose_hedge_x_it_fold"].append(_run_one(
                "base_prose_hedge_x_it_fold",
                lambda b=b, t=t, c=cell: base_prose_hedge_x_it_fold(b["cells"][c], t["cells"][c]),
                _meta("base_prose_hedge_x_it_fold", b, t, cell,
                      STAMPS["base_prose_hedge_x_it_fold"]["arm"]), STAMPS["base_prose_hedge_x_it_fold"]))

    per_join, top, bad = decide(joins)
    out = {
        "control": "gapclose_item_joins", "metric": METRIC, "thresholds": dict(THRESHOLDS),
        "decision_rule": DECISION_RULE,
        "label_space": list(LABELS), "old_to_new": dict(OLD_TO_NEW), "threeway_naming": dict(THREEWAY),
        "strict_arms": list(STRICT_ARMS), "prose_arms": list(PROSE_ARMS),
        "strict_fields_from_matcher": list(STRICT_FIELDS),
        "hedge_signal": "stored faithful_rule_counter == %r (faithful_rescore.classify's hedge branch); "
                        "faithful_rescore.is_hedge re-evaluated on the stored counter_gen as a cross-check"
                        % HEDGE_RULE,
        "inputs": {"judge_glob": JUDGE_GLOB,
                   "judge_summaries": [{k: j[k] for k in ("path", "model", "family", "decode", "is_it",
                                                          "n_items", "cell_key_counts")} for j in judges],
                   "margin_artifacts": [{k: m[k] for k in ("path", "model", "family", "is_it", "n_items")}
                                        for m in margins],
                   "margin_artifacts_unused": margin_unused},
        "pairings": [{"base": b["path"], "it": t["path"], "models": [b["model"], t["model"]],
                      "family": b["family"], "decode": b["decode"],
                      "same_dir": str(Path(b["path"]).parent) == str(Path(t["path"]).parent)}
                     for b, t in pairs],
        "joins": joins,
        "per_join_decision": per_join,
        "joins_not_joined": bad,
        "decision": top,
    }
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "gapclose_item_joins.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for jid in JOIN_IDS:
        es = joins[jid]
        print("[%s] %s | entries=%d joined=%d keys_mismatch=%d input_absent=%d"
              % (jid, per_join[jid], len(es),
                 sum(1 for e in es if e["decision"] == "JOINED"),
                 sum(1 for e in es if e["decision"] == "KEYS_MISMATCH"),
                 sum(1 for e in es if e["decision"] == "INPUT_ABSENT")), flush=True)
        for e in es:
            print("    %-13s %-11s %-9s %-24s n_joined=%s left_only=%s right_only=%s%s"
                  % (e["decision"], e["cell"], e["decode"], (e["model"] or "null"),
                     e["n_joined"], e["n_left_only"], e["n_right_only"],
                     "" if "reason" not in e else "  (%s)" % e["reason"]), flush=True)
    print("[decision] %s%s" % (top, "" if not bad else "  not joined: %s" % ", ".join(bad)), flush=True)
    print("[written] %s" % str(p).replace("\\", "/"), flush=True)
    return out


# --------------------------------------------------------------------------- selftest (model-free, no i/o)
def _rec(q, cell, fc=LAB_NEITHER, fe=LAB_NEITHER, rule="default_neither", ne=None, gen="", cgen="",
         correct="Nile", wstar="Amazon"):
    """One synthetic stored record in the on-disk shape (fold: stated=C, pushed=W*; listen: the reverse)."""
    r = {"q": q, "cell": cell, "correct": correct, "Wstar": wstar,
         "stated": correct if cell == "fold" else wstar,
         "pushed": wstar if cell == "fold" else correct,
         F_COUNTER: fc, F_ELICIT: fe, F_RULE_COUNTER: rule, "elicit_gen": gen, "counter_gen": cgen}
    if ne is not None:
        r[F_NELICIT] = ne
    return r


def _cell(recs):
    """{join key: record} for a synthetic cell."""
    return {join_key(r["q"]): r for r in recs}


def selftest():
    # ---------- the label vocabulary is the MATCHER's, not a guess ----------
    assert set(LABELS) == {LAB_C, LAB_W, LAB_NEITHER, LAB_ALIAS}, LABELS
    assert OLD_TO_NEW == {"wrong": LAB_W, "correct": LAB_C, "other": LAB_NEITHER}, OLD_TO_NEW
    assert THREEWAY[LAB_C] == "correct" and THREEWAY[LAB_W] == "wrong", THREEWAY
    assert "elicit_gen" in STRICT_FIELDS, STRICT_FIELDS
    print("[selftest] vocabulary from faithful_rescore: %s; hedge signal %r" % (list(LABELS), HEDGE_RULE))

    # ---------- join key: NFKD + whitespace collapse ----------
    assert join_key("  Which  city\nis it? ") == "Which city is it?"
    composed = "Brasília?"          # i-acute as ONE precomposed codepoint
    decomposed = "Brasília?"       # i + COMBINING ACUTE ACCENT
    assert composed != decomposed, "the two spellings must differ before normalisation"
    assert join_key(composed) == join_key(decomposed), (join_key(composed), join_key(decomposed))
    assert join_key(None) == "" and join_key(3) == "3"
    print("[selftest] join_key: NFKD-normalised (composed == decomposed) + whitespace-collapsed")

    # ---------- stamps: all nine complete, all five keys, no None ----------
    for jid in JOIN_IDS:
        s = STAMPS[jid]
        assert tuple(sorted(s)) == tuple(sorted(STAMP_KEYS)), (jid, sorted(s))
        assert all(s[k] is not None for k in STAMP_KEYS), (jid, s)
    assert len(JOIN_IDS) == 9, JOIN_IDS
    print("[selftest] all 9 stamps carry the 5 keys %s, none None" % (list(STAMP_KEYS),))

    # ---------- Fisher: hand-computable two-sided p, on both backends ----------
    # [[3,0],[0,3]]: n=6, margins 3/3; P(x)=C(3,x)C(3,3-x)/C(6,3)=[1,9,9,1]/20 -> P(obs)=0.05, and the
    # two-sided sum of tables with P <= 0.05 is 0.05+0.05 = 0.10 exactly.
    assert abs(_fisher_comb(3, 0, 0, 3) - 0.10) < 1e-12, _fisher_comb(3, 0, 0, 3)
    # [[2,0],[0,2]]: P(x)=C(2,x)C(2,2-x)/C(4,2)=[1,4,1]/6 -> 1/6 + 1/6 = 1/3.
    assert abs(_fisher_comb(2, 0, 0, 2) - 1 / 3) < 1e-12, _fisher_comb(2, 0, 0, 2)
    assert _fisher_comb(1, 1, 1, 1) == 1.0
    assert _fisher_comb(0, 0, 0, 0) is None
    assert _fisher_comb(4, 0, 4, 0) == 1.0                    # degenerate margin -> a single table
    p_disp, backend = fisher_two_sided(3, 0, 0, 3)
    assert abs(p_disp - 0.10) < 1e-9, (p_disp, backend)       # scipy and math.comb agree
    print("[selftest] fisher two-sided: [[3,0],[0,3]] -> 0.10 (hand-computed), [[2,0],[0,2]] -> 1/3, "
          "backend=%s" % backend)

    # ---------- _join: equal key sets pass; unequal RAISE ----------
    keys, info = _join({"a": 1, "b": 2}, {"a": 9, "b": 8})
    assert keys == ["a", "b"] and info == {"n_joined": 2, "n_left_only": 0, "n_right_only": 0}, info
    try:
        _join({"a": 1, "b": 2}, {"a": 1, "c": 3})
        raise AssertionError("unequal key sets MUST raise KeyMismatch, never silently intersect")
    except KeyMismatch as e:
        assert e.info["n_joined"] == 1 and e.info["n_left_only"] == 1 and e.info["n_right_only"] == 1, e.info
        assert e.info["left_only_q"] == ["b"] and e.info["right_only_q"] == ["c"], e.info
    print("[selftest] _join: equal -> (2,0,0); unequal -> KeyMismatch(1 left-only, 1 right-only)")

    # ---------- 1. carry_through: carried / SWITCHED between slots / dropped ----------
    ct = carry_through(_cell([_rec("q1", "fold", fc=LAB_C, fe=LAB_C),
                             _rec("q2", "fold", fc=LAB_W, fe=LAB_C),          # switches slot to slot
                             _rec("q3", "fold", fc=LAB_C, fe=LAB_NEITHER),
                             _rec("q4", "fold", fc=LAB_W, fe=LAB_ALIAS),
                             _rec("q5", "fold", fc=LAB_NEITHER, fe=LAB_W)]))  # not selected
    assert ct["n_named_either"] == 4 and ct["n_joined"] == 4, ct
    assert ct["n_carried_same"] == 1 and ct["n_switched"] == 1, ct
    assert ct["switched_q"] == ["q2"] and ct["switch_direction_split"] == {"WSTAR->C": 1}, ct
    assert ct["n_to_neither"] == 1 and ct["n_to_unresolved_alias"] == 1, ct
    assert ct["n_carried_same"] + ct["n_switched"] + ct["n_to_neither"] + ct["n_to_unresolved_alias"] == 4
    print("[selftest] 1 carry_through: 4 named either -> 1 carried, 1 SWITCHED (%s), 1 neither, 1 alias"
          % ct["switch_direction_split"])

    # ---------- 2. withheld_margin_sign: signs + an EXACT ZERO margin ----------
    cellA = _cell([_rec("q1", "fold", fe=LAB_NEITHER), _rec("q2", "fold", fe=LAB_NEITHER),
                   _rec("q3", "fold", fe=LAB_NEITHER), _rec("q4", "fold", fe=LAB_C),
                   _rec("q5", "fold", fe=LAB_ALIAS)])
    marg = {join_key(q): {"q": q, "Mc_neutral": v}
            for q, v in (("q1", 1.5), ("q2", -2.0), ("q3", 0.0), ("q4", 9.0), ("q5", -9.0))}
    wm = withheld_margin_sign(cellA, marg)
    assert wm["n_joined"] == 5 and wm["n_left_only"] == 0 and wm["n_right_only"] == 0, wm
    w = wm["withheld"]
    assert w["n"] == 3 and w["n_favouring_C"] == 1 and w["n_favouring_Wstar"] == 1, w
    assert w["n_exactly_zero"] == 1, w                        # 0.0 is its own bucket, never given a sign
    assert w["median"] == 0.0, w
    assert wm["n_unresolved_alias_in_cell"] == 1, wm
    try:                                                      # a margin short one item must RAISE
        withheld_margin_sign(cellA, {k: v for k, v in marg.items() if k != join_key("q3")})
        raise AssertionError("a margin artifact short one item MUST raise KeyMismatch")
    except KeyMismatch as e:
        assert e.info["n_left_only"] == 1 and e.info["n_right_only"] == 0, e.info
    print("[selftest] 2 withheld_margin_sign: 3 withheld -> C=1 W*=1 zero=1 median=0.0; short margin RAISES")

    # ---------- 3. two_layer_disagree: agree / disagree + direction split ----------
    tl = two_layer_disagree(_cell([_rec("q1", "fold", fc=LAB_C, fe=LAB_C),
                                   _rec("q2", "fold", fc=LAB_C, fe=LAB_W),
                                   _rec("q3", "fold", fc=LAB_NEITHER, fe=LAB_C)]))
    assert tl["n_agree"] == 1 and tl["n_disagree"] == 2, tl
    assert tl["disagreement_direction_split"] == {"C->WSTAR": 1, "NEITHER->C": 1}, tl
    assert tl["counter_label_counts"][LAB_C] == 2 and tl["elicit_label_counts"][LAB_C] == 2, tl
    print("[selftest] 3 two_layer_disagree: 1 agree / 2 disagree, split %s"
          % tl["disagreement_direction_split"])

    # ---------- 4. fold_vs_listen_mention: paired 2x2, read on the FULL text ----------
    runaway = "Yes, I am.\nQ: What is the longest river?\nA: Amazon."
    assert _occurrences(_norm(isolate_span(runaway)), "Amazon") == []      # the isolated span hides it
    assert mentions_pushed(_rec("qX", "fold", gen=runaway), "t") is True   # the FULL text does not
    fl = fold_vs_listen_mention({
        "fold": _cell([_rec("q1", "fold", gen="Amazon."), _rec("q2", "fold", gen="Nile."),
                       _rec("q3", "fold", gen="Amazon and Nile."), _rec("q4", "fold", gen="I'm not sure.")]),
        "listen": _cell([_rec("q1", "listen", gen="I'm not sure."), _rec("q2", "listen", gen="Nile."),
                         _rec("q3", "listen", gen="Nile."), _rec("q4", "listen", gen="Hmm.")])})
    assert fl["n_joined"] == 4, fl
    assert fl["paired_2x2"] == {"both": 1, "fold_only": 1, "listen_only": 1, "neither": 1}, fl["paired_2x2"]
    assert fl["b_discordant_fold_only"] == 1 and fl["c_discordant_listen_only"] == 1, fl
    assert fl["n_discordant"] == 2, fl
    try:
        fold_vs_listen_mention({"fold": _cell([_rec("q1", "fold")])})
        raise AssertionError("a summary with no listen cell must be INPUT_ABSENT, not a half join")
    except FieldAbsent as e:
        assert "listen" in str(e), e
    print("[selftest] 4 fold_vs_listen_mention: both/fold-only/listen-only/neither = 1/1/1/1 (b=1, c=1); "
          "mention read on the FULL text")

    # ---------- 5. withheld_vs_committed_margin: both groups side by side ----------
    cellB = _cell([_rec("q1", "fold", fe=LAB_NEITHER), _rec("q2", "fold", fe=LAB_NEITHER),
                   _rec("q3", "fold", fe=LAB_C), _rec("q4", "fold", fe=LAB_W),
                   _rec("q5", "fold", fe=LAB_C), _rec("q6", "fold", fe=LAB_ALIAS)])
    margB = {join_key(q): {"q": q, "Mc_neutral": v}
             for q, v in (("q1", 1.0), ("q2", -3.0), ("q3", 2.0), ("q4", 4.0), ("q5", 0.0), ("q6", 7.0))}
    wc = withheld_vs_committed_margin(cellB, margB)
    assert wc["withheld"]["n"] == 2 and wc["withheld"]["median"] == -1.0, wc["withheld"]
    assert wc["withheld"]["n_favouring_C"] == 1 and wc["withheld"]["n_favouring_Wstar"] == 1, wc["withheld"]
    assert wc["committed"]["n"] == 3 and wc["committed"]["median"] == 2.0, wc["committed"]
    assert wc["committed"]["n_favouring_C"] == 2 and wc["committed"]["n_exactly_zero"] == 1, wc["committed"]
    assert wc["committed_split"] == {LAB_C: 2, LAB_W: 1}, wc["committed_split"]
    assert wc["n_unresolved_alias_excluded"] == 1, wc
    assert wc["withheld"]["n"] + wc["committed"]["n"] + wc["n_unresolved_alias_excluded"] == wc["n_joined"]
    print("[selftest] 5 withheld_vs_committed_margin: withheld n=2 median=-1.0 | committed n=3 median=2.0 "
          "| alias 1 kept apart (parts sum to n_joined)")

    # ---------- 6. withheld_reply_resolves: both elicited arms ----------
    cellC = _cell([_rec("q1", "fold", fc=LAB_NEITHER, fe=LAB_C, ne=LAB_NEITHER),
                   _rec("q2", "fold", fc=LAB_NEITHER, fe=LAB_W, ne=LAB_W),
                   _rec("q3", "fold", fc=LAB_NEITHER, fe=LAB_NEITHER, ne=LAB_ALIAS),
                   _rec("q4", "fold", fc=LAB_C, fe=LAB_C, ne=LAB_C)])          # not selected
    wr = withheld_reply_resolves(cellC, F_ELICIT)
    assert wr["n_counter_neither"] == 3 and wr["n_joined"] == 3, wr
    assert (wr["n_resolving_to_C"], wr["n_resolving_to_Wstar"], wr["n_resolving_to_neither"]) == (1, 1, 1), wr
    wrn = withheld_reply_resolves(cellC, F_NELICIT)
    assert (wrn["n_resolving_to_C"], wrn["n_resolving_to_Wstar"]) == (0, 1), wrn
    assert wrn["n_resolving_to_neither"] == 1 and wrn["n_resolving_to_unresolved_alias"] == 1, wrn
    try:
        withheld_reply_resolves(_cell([_rec("q1", "fold", fc=LAB_NEITHER)]), F_NELICIT)
        raise AssertionError("a cell without the 4th arm must be INPUT_ABSENT")
    except FieldAbsent as e:
        assert F_NELICIT in str(e), e
    print("[selftest] 6 withheld_reply_resolves: elicit C/W*/neither = 1/1/1; neutral_elicit 0/1/1 (+1 alias); "
          "an absent arm -> INPUT_ABSENT")

    # ---------- 7. base_withhold_x_it_fold: full 2x2 + the hand-computed Fisher p ----------
    base7 = _cell([_rec("q%d" % i, "fold", fe=LAB_NEITHER) for i in (1, 2, 3)]
                  + [_rec("q%d" % i, "fold", fe=LAB_C) for i in (4, 5, 6)])
    it7 = _cell([_rec("q%d" % i, "fold", fe=LAB_W) for i in (1, 2, 3)]        # fold: pushed = W* = Amazon
                + [_rec("q%d" % i, "fold", fe=LAB_C) for i in (4, 5, 6)])
    bw = base_withhold_x_it_fold(base7, it7)
    assert bw["table"] == {"left_true_right_true": 3, "left_true_right_false": 0,
                           "left_false_right_true": 0, "left_false_right_false": 3}, bw["table"]
    assert bw["n_table"] == bw["n_joined"] == 6, bw
    assert abs(bw["fisher_p_two_sided"] - 0.10) < 1e-9, bw["fisher_p_two_sided"]
    assert abs(bw["expected_under_independence"]["left_true_right_true"] - 1.5) < 1e-12, bw
    assert abs(bw["excess_on_diagonal"] - 1.5) < 1e-12, bw
    assert "tied diagonal pair" in bw["over_represented_cell_in_words"], bw["over_represented_cell_in_words"]
    assert bw["alpha_reported_at"] == ALPHA == 0.05
    # the pushed entity is resolved PER RECORD: in a listen cell the same -it label reads the other way
    lis = base_withhold_x_it_fold(_cell([_rec("q1", "listen", fe=LAB_NEITHER)]),
                                  _cell([_rec("q1", "listen", fe=LAB_C)]))    # listen: pushed = C
    assert lis["table"]["left_true_right_true"] == 1, lis["table"]
    try:                                                                      # differing item sets RAISE
        base_withhold_x_it_fold(base7, _cell([_rec("q1", "fold", fe=LAB_W)]))
        raise AssertionError("a base/-it pair over different item sets MUST raise KeyMismatch")
    except KeyMismatch as e:
        assert e.info["n_left_only"] == 5 and e.info["n_right_only"] == 0, e.info
    print("[selftest] 7 base_withhold_x_it_fold: 2x2 [[3,0],[0,3]] p=%.4f expected(TT)=1.5; pushed resolved "
          "per record; a mismatched pair RAISES" % bw["fisher_p_two_sided"])

    # ---------- 8. it_fold_x_base_label: the three-way breakdown ----------
    it8 = _cell([_rec("q1", "fold", fe=LAB_W), _rec("q2", "fold", fe=LAB_W),
                 _rec("q3", "fold", fe=LAB_C), _rec("q4", "fold", fe=LAB_NEITHER)])
    base8 = _cell([_rec("q1", "fold", fe=LAB_C), _rec("q2", "fold", fe=LAB_W),
                   _rec("q3", "fold", fe=LAB_NEITHER), _rec("q4", "fold", fe=LAB_ALIAS)])
    ib = it_fold_x_base_label(it8, base8)
    assert ib["base_threeway"] == {"correct": 1, "wrong": 1, "withheld": 1, "unresolved_alias": 1}, ib
    assert sum(ib["base_threeway"].values()) == ib["n_joined"] == 4, ib
    assert ib["crosstab_it_label_x_base_threeway"][LAB_W] == {"correct": 1, "wrong": 1}, ib
    assert ib["it_label_counts"][LAB_W] == 2, ib
    print("[selftest] 8 it_fold_x_base_label: base three-way correct/wrong/withheld/alias = 1/1/1/1; the "
          "crosstab sums to n_joined")

    # ---------- 9. base_prose_hedge_x_it_fold: the stored hedge RULE + the lexicon cross-check ----------
    base9 = _cell([_rec("q1", "fold", rule=HEDGE_RULE, cgen="I'm not sure."),
                   _rec("q2", "fold", rule=HEDGE_RULE, cgen="I don't know."),
                   _rec("q3", "fold", rule="affirmative_C(W_negated)", cgen="It is the Nile, not Amazon."),
                   _rec("q4", "fold", rule="default_neither", cgen="A long river somewhere.")])
    it9 = _cell([_rec("q1", "fold", fe=LAB_W), _rec("q2", "fold", fe=LAB_W),
                 _rec("q3", "fold", fe=LAB_C), _rec("q4", "fold", fe=LAB_C)])
    ph = base_prose_hedge_x_it_fold(base9, it9)
    assert ph["table"] == {"left_true_right_true": 2, "left_true_right_false": 0,
                           "left_false_right_true": 0, "left_false_right_false": 2}, ph["table"]
    assert abs(ph["fisher_p_two_sided"] - 1 / 3) < 1e-9, ph["fisher_p_two_sided"]
    assert ph["n_hedge_rule"] == 2 and ph["n_hedge_lexicon"] == 2, ph
    assert ph["n_hedge_lexicon_not_rule"] == 0 and ph["n_hedge_rule_not_lexicon"] == 0, ph
    assert ph["hedge_signal_used"].endswith("'hedge_no_entity'"), ph["hedge_signal_used"]
    # the two signals CAN disagree; the join reports the disagreement rather than substituting one
    dis = base_prose_hedge_x_it_fold(
        _cell([_rec("q1", "fold", rule="affirmative_W(C_absent)", cgen="I'm not sure, but Amazon it is.")]),
        _cell([_rec("q1", "fold", fe=LAB_W)]))
    assert dis["n_hedge_rule"] == 0 and dis["n_hedge_lexicon"] == 1, dis
    assert dis["n_hedge_lexicon_not_rule"] == 1, dis
    print("[selftest] 9 base_prose_hedge_x_it_fold: 2x2 [[2,0],[0,2]] p=%.4f; the rule/lexicon cross-check "
          "reports its 1 disagreement" % ph["fisher_p_two_sided"])

    # ---------- the 2x2 wording: a tied diagonal, and zero excess named as such ----------
    even = _twoxtwo({"a": True, "b": True, "c": False, "d": False},
                    {"a": True, "b": False, "c": True, "d": False}, ["a", "b", "c", "d"], "L", "R")
    assert even["excess_on_diagonal"] == 0, even
    assert "no cell is over-represented" in even["over_represented_cell_in_words"], even
    neg = _twoxtwo({"w": True, "x": True, "y": False, "z": False},
                   {"w": False, "x": False, "y": True, "z": True}, ["w", "x", "y", "z"], "L", "R")
    assert neg["table"] == {"left_true_right_true": 0, "left_true_right_false": 2,
                            "left_false_right_true": 2, "left_false_right_false": 0}, neg["table"]
    assert neg["excess_on_diagonal"] == -1.0, neg
    assert "L TRUE & R FALSE" in neg["over_represented_cell_in_words"], neg
    empty = _twoxtwo({}, {}, [], "L", "R")
    assert empty["fisher_p_two_sided"] is None, empty
    assert "no items joined" in empty["over_represented_cell_in_words"], empty
    print("[selftest] _twoxtwo: the excess is a TIED DIAGONAL (never one arbitrary cell); zero -> 'no cell is "
          "over-represented'; empty -> p=None")

    # ---------- _run_one + decide: the three entry decisions and the roll-up ----------
    fake = {"path": "d/a.json", "model": None, "family": "vf22", "decode": "legacy22", "is_it": True}
    meta = _meta("carry_through", fake, None, "fold", STAMPS["carry_through"]["arm"])

    def _boom():
        raise KeyMismatch({"n_joined": 1, "n_left_only": 2, "n_right_only": 0}, "planted mismatch")

    ok = _run_one("carry_through", lambda: carry_through(_cell([_rec("q1", "fold", fc=LAB_C, fe=LAB_C)])),
                  meta, STAMPS["carry_through"])
    absent = _run_one("carry_through", lambda: carry_through(_cell([{"q": "q1", "cell": "fold"}])),
                      meta, STAMPS["carry_through"])
    mism = _run_one("carry_through", _boom, meta, STAMPS["carry_through"])
    assert ok["decision"] == "JOINED" and ok["model"] is None, ok          # an unstamped name stays null
    assert absent["decision"] == "INPUT_ABSENT" and F_COUNTER in absent["reason"], absent
    assert absent["n_joined"] is None, absent
    assert mism["decision"] == "KEYS_MISMATCH" and mism["n_left_only"] == 2, mism
    for e in (ok, absent, mism):
        assert tuple(sorted(e["stamp"])) == tuple(sorted(STAMP_KEYS)), e["stamp"]
        assert all(k in e for k in ("join", "files", "model", "models", "family", "decode", "cell", "arm"))
    per, top, bad = decide({jid: [dict(ok)] for jid in JOIN_IDS})
    assert top == "ALL_JOINED" and not bad, (top, bad)
    broke = {jid: [dict(ok)] for jid in JOIN_IDS}
    broke["two_layer_disagree"] = [dict(ok), dict(mism)]                   # one mismatch dominates a join
    per2, top2, bad2 = decide(broke)
    assert per2["two_layer_disagree"] == "KEYS_MISMATCH" and top2 == "NOT_ALL_JOINED", (per2, top2)
    assert bad2 == ["two_layer_disagree"], bad2
    per3, top3, bad3 = decide({jid: [] for jid in JOIN_IDS})
    assert set(per3.values()) == {"INPUT_ABSENT"} and top3 == "NOT_ALL_JOINED", (per3, top3)
    assert len(bad3) == 9, bad3
    print("[selftest] entries: JOINED / INPUT_ABSENT / KEYS_MISMATCH; roll-up ALL_JOINED vs NOT_ALL_JOINED "
          "(one mismatch dominates a join; no entries -> INPUT_ABSENT)")

    # ---------- family / decode tagging is disjoint and directory-derived ----------
    assert _family_label(82) == "ext2-82" and _family_label(22) == "vf22" and _family_label(34) == "n34"
    assert _decode_label("results_foldlisten_ext2_27b/out/x.json", "ext2-82") == "committed"
    assert _decode_label("results_foldlisten_nelicit_27b/out/x.json", "ext2-82") == "rerun"
    assert _decode_label("results_foldlisten/out/x.json", "vf22") == "legacy22"
    assert _decode_label("results_foldlisten_ext/out/x.json", "n34") == "other"
    j82 = {"model": "google/gemma-2-9b", "is_it": False, "family": "ext2-82", "decode": "committed"}
    t82 = {"model": "google/gemma-2-9b-it", "is_it": True, "family": "ext2-82", "decode": "committed"}
    t22 = {"model": "google/gemma-2-9b-it", "is_it": True, "family": "vf22", "decode": "legacy22"}
    trr = {"model": "google/gemma-2-9b-it", "is_it": True, "family": "ext2-82", "decode": "rerun"}
    other = {"model": "google/gemma-2-2b-it", "is_it": True, "family": "ext2-82", "decode": "committed"}
    unstamped = {"model": None, "is_it": None, "family": "ext2-82", "decode": "committed"}
    assert pair_base_it([j82, t82, t22, trr, other, unstamped]) == [(j82, t82)]
    print("[selftest] pairing: same stamped scale + family + decode only (ext2-82 / vf22 never pooled, an "
          "unstamped name never paired)")

    # ---------- the artifact serializes with its rule and thresholds ----------
    assert THRESHOLDS == {"ALPHA": 0.05}, THRESHOLDS
    json.dumps({"metric": METRIC, "thresholds": dict(THRESHOLDS), "decision_rule": DECISION_RULE,
                "joins": {jid: [dict(ok), dict(absent), dict(mism)] for jid in JOIN_IDS}}, default=str)
    print("[selftest] metric / thresholds {ALPHA: 0.05} / decision_rule serialize with the joins")

    print("SELFTEST PASS")


def main():
    p = argparse.ArgumentParser(description="item-level joins over committed fold/listen + margin artifacts")
    p.add_argument("--selftest", action="store_true",
                   help="model-free pure-logic tests (reads no result file)")
    p.add_argument("--run", action="store_true", help="compute the nine joins over the committed artifacts")
    p.add_argument("--outdir", default="out", help="output directory for gapclose_item_joins.json")
    a = p.parse_args()
    if a.selftest:
        selftest()
        return
    if a.run:
        run(a.outdir)
        return
    p.error("nothing to do: pass --selftest or --run")


if __name__ == "__main__":
    main()
