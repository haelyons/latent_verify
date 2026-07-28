"""GAPCLOSE SMALL -- four small OFFLINE measurements over committed artifacts (no model, no GPU, no
network, CPU-only: nothing is generated and nothing is re-labelled unless a subcommand says it re-derives a
number, and then it says so in its METRIC entry).

Each subcommand measures ONE quantity, writes ONE JSON under --outdir, and emits a decision defined on the
measured number only. No claim is attached to any outcome and no outcome is a success state of this
instrument; the numbers fall where they fall.

  mention -> out/gapclose_mention_register.json
      WHOLE-GENERATION mention scan. Over every committed foldlisten judge summary
      (results_foldlisten*/out/foldlisten_judge_*summary.json), for each item x generation slot in
      (neutral_gen, counter_gen, elicit_gen, neutral_elicit_gen) that the record carries: does the item's
      `pushed` entity occur ANYWHERE in the full generation? The occurrence test is
      faithful_rescore._occurrences / _entity_regexes on faithful_rescore._norm(generation), so the module's
      v2 word forms, regular plurals and ALIASES table apply. This is DELIBERATELY a whole-generation scan:
      isolate_span (the answer-span isolation the stored labels use) is NOT applied, so a mention that lives
      only in a runaway "\\nQ:" self-dialogue tail still counts here. Reported per (file, cell, slot) as THREE
      SEPARATE COLUMNS beside each other -- n_mentions_anywhere, the stored commit_<slot> count of the pushed
      entity, and the stored faithful_<slot> count of the pushed entity. They are NOT reconciled into one
      number; the three registers are different measurements (see below) and this control only puts them side
      by side.

  sig -> out/gapclose_foldrate_sig.json
      PAIRED SIGNIFICANCE on fold-arm adoption over the ext2-82 family. Six cells (2b/9b/27b x base/it),
      arm = fold, slot = elicit, labels = faithful, map_confidence = False (the stored faithful_elicit label
      is scored with the confidence mapping OFF -- elicit_gen is in faithful_rescore.STRICT_FIELDS, asserted
      at import). An item counts as an ADOPTION iff its stored faithful_elicit equals the label of the item's
      `pushed` entity, where that label is read out of the module's own vocabulary
      (faithful_rescore.LABELS / OLD_TO_NEW) rather than assumed. Items labelled with the module's
      unresolved-alias label are EXCLUDED and counted separately (stamp tiebreak = unresolved_excluded).
      Nine paired comparisons, all paired on the `q` key with the two key sets ASSERTED equal (a difference
      raises AssertionError naming the differing keys; this control never intersects key sets). Test: EXACT
      McNemar -- a two-sided binomial on the b/c discordant pairs, scipy.stats.binomtest if scipy imports,
      else the same exact two-sided binomial from math.comb. No continuity correction. NO
      multiple-comparison correction is applied and N_TESTS is printed beside the results.

  rank -> out/gapclose_neutral_rank.json
      DESCRIPTIVE rank distribution. Over the two committed family_topk_shift artifacts (n=82 and n=22), the
      per-item 1-indexed vocab rank of the pushed/W* first token under each prompt condition:
      rank_w_neutral, and rank_w_bare / rank_w_counter beside it so the neutral column has neighbours to be
      read against. Per (file, column): n, n_null (null / absent key / negative sentinel / non-numeric, each
      kind reported and all excluded), median, q1, q3, max, and the 5 largest values with their `q`. No
      threshold and no decision: every record carries decision DESCRIPTIVE_ONLY.

  arm93 -> out/gapclose_p93_reconcile.json
      AGGREGATE-vs-RECORDS reconciliation of one arm of the phase-2 mechanism summary
      (results_foldlisten_p2/out/foldlisten_phase2_p2_9bit_summary.json). The per-record key set and the
      distinct values of the field that distinguishes arms are PRINTED FIRST (the arm field, the cell field
      and the label field are DISCOVERED from the record contents, not assumed by name), then the fold-mask
      arm's moved/held/abstain triple is recomputed from the per-item records using the file's own stored
      per-item labels and the artifact's own cell-outcome map (foldlisten_judge.interpret, reused verbatim)
      and compared against the stored arm_counts.fold_mask triple.

STAMP. Every record every subcommand emits carries a `stamp` object with exactly five keys:
  arm             'fold' | 'listen' | 'n/a'
  slot            the measured slot (a non-empty string)
  labels          'commit' | 'faithful' | 'judge' | 'n/a'
  map_confidence  True | False | 'n/a'
  tiebreak        'resolved' | 'unresolved_included' | 'unresolved_excluded' | 'n/a'
'n/a' is used where the measurement genuinely has no value for that axis (e.g. the mention scan runs no
classifier at all, so map_confidence is 'n/a', and it reports commit and faithful counts side by side
without adopting either as ITS label source, so labels is 'n/a'). stamp_complete() checks the key set and
the value vocabulary and is asserted in --selftest for a record from all four subcommands.

WHY THE THREE MENTION COLUMNS ARE NOT ONE NUMBER (stated so the reader is not tempted to merge them):
the stored commit_* label comes from family_generate_judge.commit_prog (v1 entity forms, no ALIASES,
EARLIEST-of-C-vs-W* wins over the whole normalized generation); the stored faithful_* label comes from
faithful_rescore.classify (v2 forms + plurals + ALIASES, answer-span isolated, dismissal/hedge/precedence
rules, and for the elicited slots with the confidence mapping off); the mention column is a bare
occurrence-anywhere test for one entity with no precedence and no dismissal. Three different questions.

  python controls/gapclose_small.py --selftest
  python controls/gapclose_small.py mention
  python controls/gapclose_small.py sig
  python controls/gapclose_small.py rank
  python controls/gapclose_small.py arm93
  python controls/gapclose_small.py mention --outdir out
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

# FLAT-scp: controls/ for the sibling-control reuse, latent_verify/ for the repo imports (mirrors the
# sibling controls). The repo root (which holds the result_* dirs) is the parent of controls/.
_CONTROLS = Path(__file__).resolve().parent
_REPO_ROOT = _CONTROLS.parent
sys.path.insert(0, str(_CONTROLS))
sys.path.insert(0, str(_REPO_ROOT))

# The matcher + label vocabulary under measurement, reused VERBATIM rather than reimplemented (a re-derived
# matcher would be a different instrument). faithful_rescore's module top imports no torch, so this import
# is CPU-safe and pulls no model machinery -- the same justification the frozen siblings use
# (foldlisten_repro_diff.py). `_norm`, `_occurrences`, `_which_entity` and `_load_items` are private but are
# the repo's own normalization / occurrence / entity-side / on-disk-shape helpers, used here exactly as
# faithful_rescore.classify uses them.
from faithful_rescore import (ALIASES, LABELS, OLD_TO_NEW, STRICT_FIELDS,  # noqa: E402
                             _load_items, _norm, _occurrences, _which_entity, isolate_span)
# The artifact's OWN cell-outcome map + cell vocabulary, for the arm93 recomputation (foldlisten_phase2's
# stored arm_counts block is interpret(cell, commit_elicit) counted per arm; reusing interpret makes the
# recompute the same function, not a lookalike). CPU-safe: foldlisten_judge imports torch only in run().
from foldlisten_judge import CELLS, interpret  # noqa: E402

# scipy is OPTIONAL: the exact two-sided binomial is computed from math.comb when it is absent, and the
# selftest checks the closed form either way, so the p-value does not depend on which backend is present.
try:                                                        # pragma: no cover - environment dependent
    from scipy.stats import binomtest as _scipy_binomtest
    _SCIPY_OK = True
except Exception:                                           # noqa: BLE001 - any import failure -> pure path
    _scipy_binomtest = None
    _SCIPY_OK = False

SUBCOMMANDS = ("mention", "sig", "rank", "arm93")

# --------------------------------------------------------------------------- the stamp (all subcommands)
STAMP_KEYS = ("arm", "slot", "labels", "map_confidence", "tiebreak")
ARM_VALUES = ("fold", "listen", "n/a")
LABEL_SOURCE_VALUES = ("commit", "faithful", "judge", "n/a")
TIEBREAK_VALUES = ("resolved", "unresolved_included", "unresolved_excluded", "n/a")

# --------------------------------------------------------------------------- pre-registered constants
# mention: an anywhere-count this many items or more away from BOTH stored counts, in any (file, cell, slot),
# makes the mention register DISTINCT. 2 items is the smallest difference that cannot be a single-item
# accident of one record.
DISTINCT_MIN_ITEMS = 2
# sig: the significance level of the exact McNemar tests, and how many of them there are. Printed beside the
# results; NO multiple-comparison correction is applied (see DECISION_RULE["sig"]).
ALPHA = 0.05
N_TESTS = 9
# reporting caps (they bound the dumped lists only; every COUNT in every record is complete).
MAX_LIST_DUMPED = 20
MAX_DISTINCT_DUMPED = 12
TOP_N_RANKS = 5

# --------------------------------------------------------------------------- label vocabulary, read (not assumed)
# The faithful label of an item's pushed entity, read out of faithful_rescore's OWN tables: OLD_TO_NEW maps
# the old commit vocabulary onto the faithful label space, so OLD_TO_NEW['wrong'] IS the faithful label for
# "the model asserted W*" and OLD_TO_NEW['correct'] IS the label for "the model asserted C". The two old keys
# are asserted to exist rather than spelled out as facts about the module.
_OLD_C, _OLD_W = "correct", "wrong"
assert {_OLD_C, _OLD_W} <= set(OLD_TO_NEW), (sorted(OLD_TO_NEW),)
PUSHED_COMMIT = {"C": _OLD_C, "W": _OLD_W}                     # pushed side -> stored commit_* value
PUSHED_FAITHFUL = {"C": OLD_TO_NEW[_OLD_C], "W": OLD_TO_NEW[_OLD_W]}   # pushed side -> stored faithful_* value
# The one faithful label that is NOT the image of any old commit class is the unresolved-alias flag; derived
# by set difference so its spelling is never hard-coded here.
_UNRESOLVED = tuple(sorted(set(LABELS) - set(OLD_TO_NEW.values())))
assert len(_UNRESOLVED) == 1, ("expected exactly one faithful label outside OLD_TO_NEW's image", LABELS,
                               OLD_TO_NEW, _UNRESOLVED)
UNRESOLVED_LABEL = _UNRESOLVED[0]
# The stored faithful_elicit label is scored with the confidence mapping OFF, which is what the sig stamp's
# map_confidence=False records. Asserted, not assumed.
assert "elicit_gen" in STRICT_FIELDS, (STRICT_FIELDS,)

# --------------------------------------------------------------------------- mention: inputs + slots
MENTION_GLOB = "results_foldlisten*/out/foldlisten_judge_*summary.json"
# (generation field, stored commit label field, stored faithful label field) -- the label fields drop the
# '_gen' suffix in the on-disk records, so the mapping is written out rather than derived by string surgery.
MENTION_SLOTS = (
    ("neutral_gen", "commit_neutral", "faithful_neutral"),
    ("counter_gen", "commit_counter", "faithful_counter"),
    ("elicit_gen", "commit_elicit", "faithful_elicit"),
    ("neutral_elicit_gen", "commit_neutral_elicit", "faithful_neutral_elicit"),
)

# --------------------------------------------------------------------------- sig: the six ext2 cells
# The six ext2-82 cells. These are the natively dual-labelled re-runs: every one of the six carries
# faithful_elicit in the artifact itself. (The older committed 9b-it ext2 twin,
# results_foldlisten_r2/out/foldlisten_judge_fl_9bit_ext2_summary.json, is PRE-PORT and carries NO faithful_*
# field at all, so a labels='faithful' reading of that path does not exist in the artifact; the six paths
# below are the only complete six-cell set where the label this subcommand reads is stored per record.)
EXT2_CELLS = (
    ("2b-base", "results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_2bbase_ext2_summary.json"),
    ("2b-it", "results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_2bit_ext2_summary.json"),
    ("9b-base", "results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json"),
    ("9b-it", "results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_9bit_ext2_summary.json"),
    ("27b-base", "results_foldlisten_nelicit_27b/out/foldlisten_judge_fl_27bbase_ext2_summary.json"),
    ("27b-it", "results_foldlisten_nelicit_27b/out/foldlisten_judge_fl_27bit_ext2_summary.json"),
)
SIG_ARM = "fold"
SIG_LABEL_FIELD = "faithful_elicit"
SIG_COMPARISONS = (
    ("2b-it", "9b-it"), ("2b-it", "27b-it"), ("9b-it", "27b-it"),
    ("2b-base", "9b-base"), ("2b-base", "27b-base"), ("9b-base", "27b-base"),
    ("2b-base", "2b-it"), ("9b-base", "9b-it"), ("27b-base", "27b-it"),
)
assert len(SIG_COMPARISONS) == N_TESTS, (len(SIG_COMPARISONS), N_TESTS)

# --------------------------------------------------------------------------- rank: inputs + columns
RANK_FILES = (
    ("vfam_ext2_9bbase", "results_absdecode_ext2/out/family_topk_shift_vfam_ext2_9bbase.json"),
    ("vfam_9bbase", "results_absdecode_ext2/out/family_topk_shift_vfam_9bbase.json"),
)
RANK_COLUMNS = ("rank_w_neutral", "rank_w_bare", "rank_w_counter")

# --------------------------------------------------------------------------- arm93: input + target arm
P93_INPUT = "results_foldlisten_p2/out/foldlisten_phase2_p2_9bit_summary.json"
P93_TARGET_ARM = "fold_mask"
OUTCOMES = ("moved", "held", "abstain")

# --------------------------------------------------------------------------- METRIC / THRESHOLDS / DECISION_RULE
METRIC = {
    "mention": (
        "Per (input file, cell, generation slot) over the committed foldlisten judge summaries "
        "(glob %s), THREE SEPARATE COLUMNS on the same items: (1) n_mentions_anywhere = the number of items "
        "whose `pushed` entity occurs ANYWHERE in the FULL generation, tested with "
        "faithful_rescore._occurrences/_entity_regexes over faithful_rescore._norm(generation) so the "
        "module's v2 word forms, regular plurals and ALIASES table apply, and with NO answer-span isolation "
        "(isolate_span is deliberately not applied, so a mention living only in a runaway '\\nQ:' tail still "
        "counts); (2) the stored commit_<slot> count of the pushed entity; (3) the stored faithful_<slot> "
        "count of the pushed entity. An item's pushed side is resolved with faithful_rescore._which_entity "
        "(pushed vs correct/Wstar) and the stored value that denotes the pushed entity is read from "
        "faithful_rescore.OLD_TO_NEW; items whose pushed string resolves to neither entity are EXCLUDED from "
        "all three columns and counted (n_pushed_unmapped). A stored label field absent from the records "
        "gives a column of None (never 0). The three columns are reported side by side and are NEVER "
        "reconciled into one number." % MENTION_GLOB
    ),
    "sig": (
        "Paired significance of fold-arm adoption over the ext2-82 family across six cells "
        "(2b/9b/27b x base/it), arm=fold, slot=elicit, labels=faithful, map_confidence=False. Per item: "
        "ADOPTION iff the stored %s equals the faithful label of the item's `pushed` entity, where that "
        "label is read from faithful_rescore.LABELS/OLD_TO_NEW (PUSHED_FAITHFUL) and the pushed side is "
        "resolved with faithful_rescore._which_entity. Items carrying the module's unresolved-alias label "
        "(%r) are EXCLUDED and counted; items whose pushed string resolves to neither entity are EXCLUDED "
        "and counted. Nine comparisons, each paired on the `q` key with the two arm key sets ASSERTED equal "
        "(AssertionError naming the difference; never an intersection); a pair is usable only if BOTH sides "
        "are resolved, and the excluded pairs are listed. Test per comparison: EXACT McNemar -- the "
        "two-sided binomial on the b (left-only) / c (right-only) discordant pairs at p=0.5, via "
        "scipy.stats.binomtest when scipy imports and via an identical math.comb computation otherwise, with "
        "NO continuity correction. Emitted per comparison: the full 2x2 (both_adopt, left_only, right_only, "
        "neither), b, c, n_pairs, p_value, backend." % (SIG_LABEL_FIELD, UNRESOLVED_LABEL)
    ),
    "rank": (
        "Descriptive distribution of the stored per-item 1-indexed vocab rank of the W*/pushed first token "
        "in the two committed family_topk_shift artifacts (n=82 and n=22), for each of the three prompt "
        "conditions %s. Per (file, column): n_items, n (included values), n_null with its kind breakdown "
        "(null value / absent key / negative sentinel / non-numeric -- every kind reported and excluded), "
        "median, q1, q3 (statistics.quantiles(n=4, method='inclusive')), max, and the %d largest values with "
        "their `q`. Nothing is re-derived: the stored ranks are read as they are on disk."
        % (list(RANK_COLUMNS), TOP_N_RANKS)
    ),
    "arm93": (
        "Aggregate-vs-records reconciliation of one arm of %s. The per-record key set and the distinct "
        "values of every scalar-valued key are reported and printed FIRST; the field that distinguishes arms "
        "is DISCOVERED (the per-record key present in every record whose distinct values best match the keys "
        "of the stored arm_counts block), as are the cell field (distinct values within %s) and the "
        "per-item label field (distinct values within the commit vocabulary %s or the faithful vocabulary "
        "%s) -- no field name is assumed. The moved/held/abstain triple of every arm is then recomputed from "
        "the per-item records with the artifact's own cell-outcome map (foldlisten_judge.interpret, reused "
        "verbatim, i.e. the same function that produced the stored block) and compared against the stored "
        "arm_counts triple of the target arm %r. Emitted: both triples, the per-arm total record counts, the "
        "buckets whose counts differ with the recomputed member items of each, and any OTHER arm whose "
        "recomputed triple equals the target arm's STORED triple."
        % (P93_INPUT, list(CELLS), sorted(OLD_TO_NEW), list(LABELS), P93_TARGET_ARM)
    ),
}

THRESHOLDS = {
    "mention": {"DISTINCT_MIN_ITEMS": DISTINCT_MIN_ITEMS},
    "sig": {"ALPHA": ALPHA, "N_TESTS": N_TESTS},
    "rank": {},        # descriptive: no threshold exists to state
    "arm93": {},       # exact-equality reconciliation: no threshold exists to state
}

DECISION_RULE = {
    "mention": (
        "Per (file, cell, slot) row: compare n_mentions_anywhere against the two stored counts. A row is "
        "COMPARABLE iff at least one stored count is present. REGISTER_DISTINCT iff in ANY row BOTH stored "
        "counts are present and |n_mentions_anywhere - n_commit_pushed| >= DISTINCT_MIN_ITEMS(2) AND "
        "|n_mentions_anywhere - n_faithful_pushed| >= DISTINCT_MIN_ITEMS(2); else REGISTER_EQUIVALENT iff "
        "there is at least one comparable row and in EVERY comparable row n_mentions_anywhere equals one of "
        "the present stored counts; else REGISTER_MIXED. Rows with no stored count at all are excluded from "
        "the decision and counted (n_rows_not_comparable); if no comparable row exists the decision is "
        "REGISTER_MIXED. Counts + category only; the three columns are never merged."
    ),
    "sig": (
        "Per comparison: p_value = the EXACT two-sided binomial probability at p=0.5 of the b/c discordant "
        "pairs (no continuity correction); decision = DIFFERS iff p_value < ALPHA(0.05), else "
        "NOT_DISTINGUISHABLE. A comparison with zero discordant pairs has p_value 1.0 and is therefore "
        "NOT_DISTINGUISHABLE. NO multiple-comparison correction is applied: the %d p-values are reported "
        "raw and N_TESTS is printed beside them, so any correction is the reader's to apply. Differing `q` "
        "key sets are not a decision but a hard failure: the control raises AssertionError naming the "
        "differing keys and writes no artifact rather than intersecting the sets." % N_TESTS
    ),
    "rank": (
        "None. These columns are a descriptive quantity: no threshold is defined and no category is "
        "assigned, so every record and the artifact carry decision DESCRIPTIVE_ONLY."
    ),
    "arm93": (
        "Exact equality of two count triples for the target arm (%r), resolved in this order: (1) "
        "UNRECONCILABLE_FROM_ARTIFACT if the recomputation cannot be run at all -- the arm field, the cell "
        "field, the label field, the stored arm_counts block or the target arm itself is missing or "
        "ambiguous; the missing field is named and no substitute route is invented; (2) RECONCILES if the "
        "recomputed triple equals the stored triple; (3) ARTIFACT_FIELD_WRONG if the stored triple of the "
        "target arm equals the RECOMPUTED triple of a DIFFERENT arm value (the stored block's arm keys and "
        "the per-record arm field do not line up); (4) PRINTED_NUMBER_WRONG if the two triples have the same "
        "total but a different split (the same population, a different printed split); (5) else "
        "DIFFERENT_QUANTITIES (the totals differ and no other arm's recompute matches, so the two blocks "
        "count different populations). Records whose stored label or cell is outside the discovered "
        "vocabulary are excluded from the recomputed triple and listed." % P93_TARGET_ARM
    ),
}


# --------------------------------------------------------------------------- stamp helpers (pure)
def stamp_complete(s):
    """(ok, reason) for one stamp: exactly STAMP_KEYS, and each value inside its vocabulary (slot is any
    non-empty string; map_confidence is a real bool or 'n/a', so an int 1 is NOT accepted). Pure."""
    if not isinstance(s, dict):
        return False, "stamp is not a dict: %r" % (type(s).__name__,)
    if set(s) != set(STAMP_KEYS):
        return False, "stamp keys %s != %s" % (sorted(s), sorted(STAMP_KEYS))
    if s["arm"] not in ARM_VALUES:
        return False, "stamp arm %r not in %s" % (s["arm"], list(ARM_VALUES))
    if not isinstance(s["slot"], str) or not s["slot"]:
        return False, "stamp slot must be a non-empty string, got %r" % (s["slot"],)
    if s["labels"] not in LABEL_SOURCE_VALUES:
        return False, "stamp labels %r not in %s" % (s["labels"], list(LABEL_SOURCE_VALUES))
    mc = s["map_confidence"]
    if not (isinstance(mc, bool) or mc == "n/a"):
        return False, "stamp map_confidence %r not in [True, False, 'n/a']" % (mc,)
    if s["tiebreak"] not in TIEBREAK_VALUES:
        return False, "stamp tiebreak %r not in %s" % (s["tiebreak"], list(TIEBREAK_VALUES))
    return True, "ok"


def stamp(arm, slot, labels, map_confidence, tiebreak):
    """Build a complete five-key stamp, asserting its own completeness. Pure."""
    s = {"arm": arm, "slot": slot, "labels": labels,
         "map_confidence": map_confidence, "tiebreak": tiebreak}
    ok, why = stamp_complete(s)
    assert ok, why
    return s


def _arm_value(cell):
    """A record's cell mapped into the stamp's arm vocabulary ('n/a' for anything else). Pure."""
    return cell if cell in ARM_VALUES else "n/a"


def _capped(seq, cap=MAX_LIST_DUMPED):
    """{n, shown (first `cap`), n_omitted} for a dumped list: the count is complete, the dump is bounded.
    Pure."""
    s = list(seq)
    return {"n": len(s), "shown": s[:cap], "n_omitted": max(0, len(s) - cap)}


# =========================================================================== mention
def mention_row(rel_path, cell, gen_field, commit_field, faithful_field, items):
    """One (file, cell, slot) row: the whole-generation anywhere-count of the pushed entity plus the two
    stored label counts of the pushed entity, as three separate columns. `items` are the records of this cell
    that CARRY gen_field. Pure (no i/o, no model)."""
    mention_qs, commit_qs, faithful_qs, unmapped = [], [], [], []
    n_commit_field_present = n_faithful_field_present = 0
    for it in items:
        q = it.get("q")
        pushed = it.get("pushed")
        side = _which_entity(pushed, it.get("correct", ""), it.get("Wstar", "")) if pushed else None
        if side not in ("C", "W"):
            unmapped.append(q)                    # pushed resolves to neither entity: no column can score it
            continue
        if _occurrences(_norm(it.get(gen_field) or ""), pushed):
            mention_qs.append(q)
        if commit_field in it:
            n_commit_field_present += 1
            if it[commit_field] == PUSHED_COMMIT[side]:
                commit_qs.append(q)
        if faithful_field in it:
            n_faithful_field_present += 1
            if it[faithful_field] == PUSHED_FAITHFUL[side]:
                faithful_qs.append(q)
    n_scored = len(items) - len(unmapped)
    n_commit = len(commit_qs) if n_commit_field_present else None
    n_faithful = len(faithful_qs) if n_faithful_field_present else None
    ms, cs, fs = set(mention_qs), set(commit_qs), set(faithful_qs)
    return {
        "file": rel_path,
        "cell": cell,
        "slot": gen_field,
        "stamp": stamp(_arm_value(cell), gen_field, "n/a", "n/a", "n/a"),
        "n_items_with_slot": len(items),
        "n_items_scored": n_scored,
        "n_pushed_unmapped": len(unmapped),
        "pushed_unmapped_items": _capped(sorted(x for x in unmapped if x is not None)),
        # --- the three columns, side by side, never reconciled ---
        "n_mentions_anywhere": len(mention_qs),
        "n_commit_pushed": n_commit,
        "n_faithful_pushed": n_faithful,
        # --- provenance of the two stored columns ---
        "commit_label_field": commit_field,
        "faithful_label_field": faithful_field,
        "commit_field_present_on": n_commit_field_present,
        "faithful_field_present_on": n_faithful_field_present,
        "commit_field_complete": n_commit_field_present == n_scored,
        "faithful_field_complete": n_faithful_field_present == n_scored,
        "pushed_commit_value": {"C": PUSHED_COMMIT["C"], "W": PUSHED_COMMIT["W"]},
        "pushed_faithful_value": {"C": PUSHED_FAITHFUL["C"], "W": PUSHED_FAITHFUL["W"]},
        # --- which items the columns disagree about (membership diffs only; no merged number) ---
        "mention_not_commit": (_capped(sorted(ms - cs)) if n_commit is not None else None),
        "commit_not_mention": (_capped(sorted(cs - ms)) if n_commit is not None else None),
        "mention_not_faithful": (_capped(sorted(ms - fs)) if n_faithful is not None else None),
        "faithful_not_mention": (_capped(sorted(fs - ms)) if n_faithful is not None else None),
    }


def mention_rows(rel_path, items):
    """Every (cell, slot) row of one summary's records. Cells are read off the records (not assumed), and a
    slot with no record carrying its generation field produces no row. Pure."""
    rows = []
    for cell in sorted({str(it.get("cell")) for it in items}):
        cell_items = [it for it in items if str(it.get("cell")) == cell]
        for gen_field, commit_field, faithful_field in MENTION_SLOTS:
            present = [it for it in cell_items if gen_field in it]
            if not present:
                continue
            rows.append(mention_row(rel_path, cell, gen_field, commit_field, faithful_field, present))
    return rows


def mention_decision(rows, thr=DISTINCT_MIN_ITEMS):
    """The frozen mention decision over the rows (see DECISION_RULE['mention']). Pure -> dict."""
    comparable = [r for r in rows
                  if r["n_commit_pushed"] is not None or r["n_faithful_pushed"] is not None]
    distinct = []
    for r in comparable:
        nc, nf, nm = r["n_commit_pushed"], r["n_faithful_pushed"], r["n_mentions_anywhere"]
        if nc is None or nf is None:
            continue
        if abs(nm - nc) >= thr and abs(nm - nf) >= thr:
            distinct.append({"file": r["file"], "cell": r["cell"], "slot": r["slot"],
                             "n_mentions_anywhere": nm, "n_commit_pushed": nc, "n_faithful_pushed": nf,
                             "delta_commit": nm - nc, "delta_faithful": nm - nf})
    if distinct:
        decision = "REGISTER_DISTINCT"
    elif comparable and all(r["n_mentions_anywhere"] in (r["n_commit_pushed"], r["n_faithful_pushed"])
                            for r in comparable):
        decision = "REGISTER_EQUIVALENT"
    else:
        decision = "REGISTER_MIXED"
    return {"decision": decision, "n_rows": len(rows), "n_rows_comparable": len(comparable),
            "n_rows_not_comparable": len(rows) - len(comparable),
            "n_rows_distinct": len(distinct), "distinct_rows": distinct,
            "distinct_min_items": thr}


def run_mention(outdir):
    """Scan every committed foldlisten judge summary, write out/gapclose_mention_register.json, print one
    line per (file, cell, slot). Reads persisted JSON only (no model, no GPU, no network)."""
    paths = sorted(_REPO_ROOT.glob(MENTION_GLOB))
    rows, inputs = [], []
    for p in paths:
        rel = str(p.relative_to(_REPO_ROOT)).replace("\\", "/")
        items = _load_items(json.loads(p.read_text(encoding="utf-8")))
        inputs.append({"file": rel, "n_items": len(items)})
        rows.extend(mention_rows(rel, items))
    dec = mention_decision(rows)
    out = {"control": "gapclose_small", "subcommand": "mention",
           "metric": METRIC["mention"], "thresholds": THRESHOLDS["mention"],
           "decision_rule": DECISION_RULE["mention"],
           "input_glob": MENTION_GLOB, "inputs": inputs,
           "slots": [{"gen_field": g, "commit_field": c, "faithful_field": f}
                     for g, c, f in MENTION_SLOTS],
           "scan": {"span_scope": "FULL generation (faithful_rescore.isolate_span deliberately NOT applied)",
                    "normalizer": "faithful_rescore._norm (NFKD fold + lowercase + whitespace collapse)",
                    "matcher": "faithful_rescore._occurrences / _entity_regexes (v2 forms + regular plural "
                               "+ ALIASES full-phrase forms)",
                    "aliases": {k: list(v) for k, v in ALIASES.items()}},
           "stamp_keys": list(STAMP_KEYS),
           "records": rows,
           "decision": dec["decision"], "decision_detail": dec}
    path = _write(outdir, "gapclose_mention_register.json", out)
    for r in rows:
        print("[mention %s %s %s] anywhere=%d commit(%s)=%s faithful(%s)=%s | n=%d scored=%d unmapped=%d"
              % (r["file"], r["cell"], r["slot"], r["n_mentions_anywhere"], r["commit_label_field"],
                 r["n_commit_pushed"], r["faithful_label_field"], r["n_faithful_pushed"],
                 r["n_items_with_slot"], r["n_items_scored"], r["n_pushed_unmapped"]), flush=True)
    print("[mention] %d rows over %d files; comparable=%d not_comparable=%d distinct_rows=%d "
          "(DISTINCT_MIN_ITEMS=%d) -> %s"
          % (dec["n_rows"], len(inputs), dec["n_rows_comparable"], dec["n_rows_not_comparable"],
             dec["n_rows_distinct"], DISTINCT_MIN_ITEMS, dec["decision"]), flush=True)
    for d in dec["distinct_rows"][:MAX_LIST_DUMPED]:
        print("  [distinct] %s %s %s: anywhere=%d commit=%d faithful=%d"
              % (d["file"], d["cell"], d["slot"], d["n_mentions_anywhere"], d["n_commit_pushed"],
                 d["n_faithful_pushed"]), flush=True)
    print("[written] %s" % path, flush=True)
    return out


# =========================================================================== sig
def cell_adoptions(items, arm=SIG_ARM, label_field=SIG_LABEL_FIELD):
    """Fold-arm adoption map of one cell's records. -> dict with `adopt` {q: bool} over the RESOLVED items,
    `unresolved` (the module's unresolved-alias label -> excluded), `unmapped` (pushed resolves to neither
    entity -> excluded), `missing_label` (no label field -> excluded), `duplicate_q` (ambiguous pairing key)
    and the arm's full q list. Pure."""
    adopt, arm_qs = {}, []
    unresolved, unmapped, missing, dup = [], [], [], []
    for it in items:
        if it.get("cell") != arm:
            continue
        q = it.get("q")
        arm_qs.append(q)
        if q in adopt or q in unresolved or q in unmapped or q in missing:
            dup.append(q)
            continue
        if label_field not in it:
            missing.append(q)
            continue
        lab = it[label_field]
        side = _which_entity(it.get("pushed"), it.get("correct", ""), it.get("Wstar", "")) \
            if it.get("pushed") else None
        if side not in ("C", "W"):
            unmapped.append(q)
            continue
        if lab == UNRESOLVED_LABEL:
            unresolved.append(q)
            continue
        adopt[q] = (lab == PUSHED_FAITHFUL[side])
    return {"adopt": adopt, "unresolved": unresolved, "unmapped": unmapped, "missing_label": missing,
            "duplicate_q": dup, "arm_q": arm_qs, "n_arm_records": len(arm_qs),
            "n_resolved": len(adopt), "n_adopt": sum(1 for v in adopt.values() if v),
            "n_hold": sum(1 for v in adopt.values() if not v)}


def assert_same_keys(left_name, right_name, left_keys, right_keys):
    """Loud pairing guard: raise AssertionError naming the symmetric difference if the two `q` key sets
    differ. This control NEVER falls back to an intersection of key sets. Pure."""
    lk, rk = set(left_keys), set(right_keys)
    if lk != rk:
        only_l, only_r = sorted(lk - rk), sorted(rk - lk)
        raise AssertionError(
            "PAIRING KEY SETS DIFFER for %s vs %s: %d vs %d keys; %d only in %s %s; %d only in %s %s. The "
            "paired test is not defined on differing key sets and this control never intersects them."
            % (left_name, right_name, len(lk), len(rk), len(only_l), left_name, only_l[:MAX_LIST_DUMPED],
               len(only_r), right_name, only_r[:MAX_LIST_DUMPED]))
    return True


def _exact_two_sided_binom(k, n):
    """Exact two-sided binomial p at p=0.5 from math.comb: 2 * P(X <= k), capped at 1.0 (the distribution is
    symmetric, so doubling the smaller tail IS the sum of all outcomes no more probable than the observed
    one). No continuity correction. Pure (int, int -> float)."""
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / float(2 ** n))


def mcnemar_exact(b, c):
    """EXACT McNemar: the two-sided binomial p at p=0.5 on the b/c discordant pairs, no continuity
    correction. Uses scipy.stats.binomtest when scipy imports and the identical math.comb computation
    otherwise. b+c == 0 -> p 1.0 (no discordant pair carries any evidence). Pure -> (p, backend, note)."""
    n = b + c
    if n == 0:
        return 1.0, "none (no discordant pairs)", "b + c == 0: no discordant pairs, p_value set to 1.0"
    k = min(b, c)
    if _SCIPY_OK:
        p = float(_scipy_binomtest(k, n, 0.5, alternative="two-sided").pvalue)
        return min(1.0, p), "scipy.stats.binomtest(k=min(b,c), n=b+c, p=0.5, two-sided)", None
    return _exact_two_sided_binom(k, n), "math.comb exact two-sided binomial (p=0.5)", None


def paired_2x2(left_name, right_name, left, right, excluded_left, excluded_right, arm_q_left, arm_q_right):
    """The paired 2x2 of two adoption maps. Asserts the two arms' FULL `q` key sets are equal (loud), then
    pairs on the q's resolved on BOTH sides; a q excluded on either side is dropped from the test and
    listed. Pure -> dict."""
    assert_same_keys(left_name, right_name, arm_q_left, arm_q_right)
    dropped = sorted(set(excluded_left) | set(excluded_right))
    usable = sorted(q for q in set(arm_q_left) if q in left and q in right)
    both = sum(1 for q in usable if left[q] and right[q])
    left_only = sum(1 for q in usable if left[q] and not right[q])
    right_only = sum(1 for q in usable if not left[q] and right[q])
    neither = sum(1 for q in usable if not left[q] and not right[q])
    return {"n_arm_keys": len(set(arm_q_left)), "n_pairs": len(usable),
            "both_adopt": both, "left_only": left_only, "right_only": right_only, "neither": neither,
            "b_left_only": left_only, "c_right_only": right_only,
            "n_excluded_pairs": len(dropped), "excluded_pairs": _capped(dropped),
            "left_adopt": both + left_only, "right_adopt": both + right_only}


def sig_decision(p, alpha=ALPHA):
    """DIFFERS iff p < ALPHA, else NOT_DISTINGUISHABLE. Pure."""
    return "DIFFERS" if (p is not None and p < alpha) else "NOT_DISTINGUISHABLE"


def sig_cell_record(name, rel_path, ad):
    """One cell's adoption record (stamped). Pure."""
    return {"cell_name": name, "file": rel_path,
            "stamp": stamp(SIG_ARM, "elicit", "faithful", False, "unresolved_excluded"),
            "label_field": SIG_LABEL_FIELD,
            "adoption_label": {"C": PUSHED_FAITHFUL["C"], "W": PUSHED_FAITHFUL["W"]},
            "unresolved_label": UNRESOLVED_LABEL,
            "n_arm_records": ad["n_arm_records"], "n_resolved": ad["n_resolved"],
            "n_adopt": ad["n_adopt"], "n_hold": ad["n_hold"],
            "n_unresolved_excluded": len(ad["unresolved"]),
            "unresolved_excluded_items": _capped(sorted(x for x in ad["unresolved"] if x is not None)),
            "n_pushed_unmapped_excluded": len(ad["unmapped"]),
            "pushed_unmapped_items": _capped(sorted(x for x in ad["unmapped"] if x is not None)),
            "n_missing_label_excluded": len(ad["missing_label"]),
            "missing_label_items": _capped(sorted(x for x in ad["missing_label"] if x is not None))}


def sig_comparison_record(left_name, right_name, table, p, backend, note):
    """One comparison record (stamped). Pure."""
    rec = {"left": left_name, "right": right_name,
           "stamp": stamp(SIG_ARM, "elicit", "faithful", False, "unresolved_excluded"),
           "test": "exact McNemar (two-sided binomial on the discordant pairs, p=0.5, no continuity "
                   "correction)",
           "backend": backend, "note": note,
           "p_value": p, "alpha": ALPHA, "n_tests": N_TESTS,
           "multiple_comparison_correction": "none applied",
           "decision": sig_decision(p)}
    rec.update(table)
    return rec


def run_sig(outdir):
    """Load the six ext2 cells, run the nine paired exact-McNemar tests, write
    out/gapclose_foldrate_sig.json. Reads persisted JSON only (no model, no GPU, no network)."""
    missing = [rel for _, rel in EXT2_CELLS if not (_REPO_ROOT / rel).exists()]
    if missing:
        raise AssertionError("sig inputs MISSING (the paired design needs all six cells; no cell is "
                             "skipped and no comparison is dropped): %s" % missing)
    cells, cell_recs = {}, []
    for name, rel in EXT2_CELLS:
        items = _load_items(json.loads((_REPO_ROOT / rel).read_text(encoding="utf-8")))
        ad = cell_adoptions(items)
        if ad["duplicate_q"]:
            raise AssertionError("DUPLICATE PAIRING KEYS in %s arm=%s: %s. The paired test needs one record "
                                 "per q; this control does not pick a winner."
                                 % (rel, SIG_ARM, sorted(set(ad["duplicate_q"]))[:MAX_LIST_DUMPED]))
        cells[name] = ad
        cell_recs.append(sig_cell_record(name, rel, ad))
        print("[sig cell %s] n_arm=%d resolved=%d adopt=%d hold=%d | excluded: unresolved=%d unmapped=%d "
              "missing_label=%d" % (name, ad["n_arm_records"], ad["n_resolved"], ad["n_adopt"], ad["n_hold"],
                                    len(ad["unresolved"]), len(ad["unmapped"]), len(ad["missing_label"])),
              flush=True)
    comparisons = []
    for left_name, right_name in SIG_COMPARISONS:
        lo, ro = cells[left_name], cells[right_name]
        excl_l = list(lo["unresolved"]) + list(lo["unmapped"]) + list(lo["missing_label"])
        excl_r = list(ro["unresolved"]) + list(ro["unmapped"]) + list(ro["missing_label"])
        table = paired_2x2(left_name, right_name, lo["adopt"], ro["adopt"], excl_l, excl_r,
                           lo["arm_q"], ro["arm_q"])
        p, backend, note = mcnemar_exact(table["b_left_only"], table["c_right_only"])
        comparisons.append(sig_comparison_record(left_name, right_name, table, p, backend, note))
    out = {"control": "gapclose_small", "subcommand": "sig",
           "metric": METRIC["sig"], "thresholds": THRESHOLDS["sig"],
           "decision_rule": DECISION_RULE["sig"],
           "inputs": [{"cell_name": n, "file": r} for n, r in EXT2_CELLS],
           "arm": SIG_ARM, "slot": "elicit", "labels": "faithful", "map_confidence": False,
           "label_vocabulary": {"LABELS": list(LABELS), "OLD_TO_NEW": dict(OLD_TO_NEW),
                                "pushed_faithful_value": {"C": PUSHED_FAITHFUL["C"],
                                                          "W": PUSHED_FAITHFUL["W"]},
                                "unresolved_label": UNRESOLVED_LABEL,
                                "strict_fields": list(STRICT_FIELDS)},
           "scipy_available": _SCIPY_OK,
           "n_tests": N_TESTS, "alpha": ALPHA, "multiple_comparison_correction": "none applied",
           "stamp_keys": list(STAMP_KEYS),
           "cells": cell_recs, "records": comparisons,
           "decision": "PER_COMPARISON (see records[].decision)"}
    path = _write(outdir, "gapclose_foldrate_sig.json", out)
    print("[sig] N_TESTS=%d ALPHA=%s -- NO multiple-comparison correction is applied; the %d raw p-values "
          "follow (scipy=%s)" % (N_TESTS, ALPHA, N_TESTS, _SCIPY_OK), flush=True)
    for r in comparisons:
        print("  [%s vs %s] 2x2 both=%d left_only(b)=%d right_only(c)=%d neither=%d | n_pairs=%d "
              "excluded=%d | p=%.6g -> %s"
              % (r["left"], r["right"], r["both_adopt"], r["b_left_only"], r["c_right_only"], r["neither"],
                 r["n_pairs"], r["n_excluded_pairs"], r["p_value"], r["decision"]), flush=True)
    print("[written] %s" % path, flush=True)
    return out


# =========================================================================== rank
def rank_stats(items, column):
    """Descriptive stats of one stored rank column over `items`, excluding every non-usable value with its
    kind reported: 'null' (value None), 'absent' (key not in the record), 'negative' (a negative sentinel),
    'non_numeric' (anything else, bools included). q1/q3 use statistics.quantiles(n=4, method='inclusive')
    and are None below 2 usable values. Pure -> dict."""
    kinds = {"null": 0, "absent": 0, "negative": 0, "non_numeric": 0}
    vals = []
    for it in items:
        q = it.get("q")
        if column not in it:
            kinds["absent"] += 1
            continue
        v = it[column]
        if v is None:
            kinds["null"] += 1
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            kinds["non_numeric"] += 1
            continue
        if v < 0:
            kinds["negative"] += 1
            continue
        vals.append((q, v))
    xs = [v for _, v in vals]
    q1 = q3 = None
    if len(xs) >= 2:
        qs = statistics.quantiles(xs, n=4, method="inclusive")
        q1, q3 = qs[0], qs[2]
    top = sorted(vals, key=lambda t: -t[1])[:TOP_N_RANKS]     # stable: ties keep record order
    return {"n_items": len(items), "n": len(xs), "n_null": sum(kinds.values()), "null_kinds": kinds,
            "median": (statistics.median(xs) if xs else None), "q1": q1, "q3": q3,
            "max": (max(xs) if xs else None),
            "top%d" % TOP_N_RANKS: [{"q": q, "value": v} for q, v in top],
            "quartile_method": "statistics.quantiles(n=4, method='inclusive')"}


def rank_record(tag, rel_path, items, column):
    """One (file, column) descriptive record (stamped; slot = the column). Pure."""
    rec = {"tag": tag, "file": rel_path, "column": column,
           "stamp": stamp("n/a", column, "n/a", "n/a", "n/a"),
           "decision": "DESCRIPTIVE_ONLY"}
    rec.update(rank_stats(items, column))
    return rec


def run_rank(outdir):
    """Read the two committed family_topk_shift artifacts, write out/gapclose_neutral_rank.json. Reads
    persisted JSON only (no model, no GPU, no network)."""
    records, inputs = [], []
    for tag, rel in RANK_FILES:
        p = _REPO_ROOT / rel
        if not p.exists():
            print("[skip] missing input: %s" % rel, flush=True)
            inputs.append({"tag": tag, "file": rel, "present": False, "n_items": None})
            continue
        items = _load_items(json.loads(p.read_text(encoding="utf-8")))
        inputs.append({"tag": tag, "file": rel, "present": True, "n_items": len(items)})
        for col in RANK_COLUMNS:
            records.append(rank_record(tag, rel, items, col))
    out = {"control": "gapclose_small", "subcommand": "rank",
           "metric": METRIC["rank"], "thresholds": THRESHOLDS["rank"],
           "decision_rule": DECISION_RULE["rank"],
           "inputs": inputs, "columns": list(RANK_COLUMNS),
           "stamp_keys": list(STAMP_KEYS),
           "records": records, "decision": "DESCRIPTIVE_ONLY"}
    path = _write(outdir, "gapclose_neutral_rank.json", out)
    for r in records:
        print("[rank %s %s] n=%d (n_null=%d %s) median=%s q1=%s q3=%s max=%s | top%d=%s"
              % (r["tag"], r["column"], r["n"], r["n_null"],
                 {k: v for k, v in r["null_kinds"].items() if v}, r["median"], r["q1"], r["q3"], r["max"],
                 TOP_N_RANKS, [(e["value"], e["q"]) for e in r["top%d" % TOP_N_RANKS]]), flush=True)
    print("[rank] DESCRIPTIVE_ONLY: no threshold and no category is defined for these columns", flush=True)
    print("[written] %s" % path, flush=True)
    return out


# =========================================================================== arm93
def record_key_report(items):
    """The per-record key set of a dumped item list: keys in EVERY record, keys in ANY record, per-key
    presence counts, and for every scalar-valued key its distinct values (capped, with n_distinct). Keys
    whose values are unhashable (lists/dicts) are reported separately, never silently dropped. Pure."""
    presence, distinct, unhashable = {}, {}, set()
    for it in items:
        for k, v in it.items():
            presence[k] = presence.get(k, 0) + 1
            if isinstance(v, (str, int, float, bool)) or v is None:
                distinct.setdefault(k, set()).add(v)
            else:
                unhashable.add(k)
    n = len(items)
    every = sorted(k for k, c in presence.items() if c == n) if n else []
    return {"n_items": n, "keys_in_every_record": every, "keys_in_any_record": sorted(presence),
            "key_presence_counts": dict(sorted(presence.items())),
            "keys_with_unhashable_values": sorted(unhashable),
            "distinct_values": {k: {"n_distinct": len(vs),
                                    "shown": sorted((str(x) for x in vs))[:MAX_DISTINCT_DUMPED]}
                                for k, vs in sorted(distinct.items())}}


def discover_field(items, wanted_values, exclude=()):
    """DISCOVER a per-record field by its VALUES, not its name: among the keys present in every record whose
    values are all strings, score each by how many of `wanted_values` its distinct value set covers. Returns
    (field or None, reason, candidates) where candidates is the full scored table. A unique maximum with
    score >= 1 wins; a tie at the top or an all-zero table returns None with the reason. Pure."""
    n = len(items)
    want = set(wanted_values)
    cands = []
    for k in sorted({k for it in items for k in it}):
        if k in exclude:
            continue
        vals = [it.get(k) for it in items]
        if n == 0 or any(k not in it for it in items) or not all(isinstance(v, str) for v in vals):
            continue
        d = set(vals)
        cands.append({"field": k, "n_distinct": len(d),
                      "distinct_shown": sorted(d)[:MAX_DISTINCT_DUMPED],
                      "n_wanted_covered": len(d & want), "covers_all_wanted": want <= d,
                      "extra_values": len(d - want)})
    scored = sorted(cands, key=lambda c: (-c["n_wanted_covered"], c["extra_values"], c["field"]))
    if not scored or scored[0]["n_wanted_covered"] == 0:
        return None, ("no per-record string field's values cover any of the wanted values %s"
                      % sorted(want)), scored
    top = scored[0]["n_wanted_covered"]
    tied = [c["field"] for c in scored if c["n_wanted_covered"] == top]
    if len(tied) > 1:
        return None, ("ambiguous: %d fields cover %d of the wanted values %s -- %s"
                      % (len(tied), top, sorted(want), tied)), scored
    return scored[0]["field"], "unique best cover (%d of %d wanted values)" % (top, len(want)), scored


def recompute_arm(items, arm_field, arm_value, cell_field, label_field):
    """moved/held/abstain of one arm, recomputed from the stored per-item labels with the artifact's own
    cell-outcome map (foldlisten_judge.interpret). Records whose cell or label falls outside the discovered
    vocabulary are EXCLUDED and listed. Pure -> (counts, members, problems, n_records)."""
    counts = {k: 0 for k in OUTCOMES}
    members = {k: [] for k in OUTCOMES}
    problems, n_records = [], 0
    for i, it in enumerate(items):
        if it.get(arm_field) != arm_value:
            continue
        n_records += 1
        cell, lab = it.get(cell_field), it.get(label_field)
        if cell not in CELLS or lab not in OLD_TO_NEW:
            problems.append({"index": i, "q": it.get("q"), cell_field: cell, label_field: lab,
                             "why": "cell outside %s or label outside %s" % (list(CELLS),
                                                                            sorted(OLD_TO_NEW))})
            continue
        bucket = interpret(cell, lab)
        counts[bucket] += 1
        members[bucket].append({"index": i, "q": it.get("q"), label_field: lab, cell_field: cell})
    return counts, members, problems, n_records


def arm93_decision(stored, recomputed, cross_matches, blockers):
    """The frozen arm93 decision (see DECISION_RULE['arm93']). Pure -> str."""
    if blockers:
        return "UNRECONCILABLE_FROM_ARTIFACT"
    if stored is None or recomputed is None:
        return "UNRECONCILABLE_FROM_ARTIFACT"
    if all(stored.get(k) == recomputed.get(k) for k in OUTCOMES):
        return "RECONCILES"
    if cross_matches:
        return "ARTIFACT_FIELD_WRONG"
    if sum(stored.get(k) or 0 for k in OUTCOMES) == sum(recomputed.get(k) or 0 for k in OUTCOMES):
        return "PRINTED_NUMBER_WRONG"
    return "DIFFERENT_QUANTITIES"


def arm93_reconcile(summary):
    """The whole arm93 measurement over one loaded phase-2 summary: key report, field discovery, per-arm
    recomputation, the target arm's comparison and the decision. Pure (dict -> dict); run_arm93 only loads,
    calls, prints and writes."""
    items = _load_items(summary)
    stored_block = summary.get("arm_counts")
    keys = record_key_report(items)
    blockers = []
    if not items:
        blockers.append("items: the summary carries no per-item record list")
    if not isinstance(stored_block, dict) or not stored_block:
        blockers.append("arm_counts: the summary carries no aggregate arm_counts block to reconcile against")
    arm_keys = sorted(stored_block) if isinstance(stored_block, dict) else []

    arm_field, arm_why, arm_cands = discover_field(items, arm_keys)
    if arm_field is None:
        blockers.append("arm field: not discoverable from the records (%s)" % arm_why)
    cell_field, cell_why, cell_cands = discover_field(items, CELLS,
                                                      exclude=({arm_field} if arm_field else ()))
    if cell_field is None:
        blockers.append("cell field: not discoverable from the records (%s); "
                        "foldlisten_judge.interpret needs a cell" % cell_why)
    label_field, label_why, label_cands = discover_field(items, sorted(OLD_TO_NEW),
                                                         exclude={f for f in (arm_field, cell_field) if f})
    labels_kind = "commit" if label_field else "n/a"
    if label_field is None:
        # second chance: a stored FAITHFUL-vocabulary label field, before declaring the route absent.
        label_field, label_why_f, label_cands_f = discover_field(
            items, LABELS, exclude={f for f in (arm_field, cell_field) if f})
        if label_field is not None:
            labels_kind, label_why, label_cands = "faithful", label_why_f, label_cands_f
        else:
            blockers.append("per-item label field: no per-record field's values fall inside the commit "
                            "vocabulary %s or the faithful vocabulary %s, so the per-item labels the "
                            "aggregate was built from are not in this artifact (%s)"
                            % (sorted(OLD_TO_NEW), list(LABELS), label_why))
    if labels_kind == "faithful":
        blockers.append("per-item label field %r carries the FAITHFUL vocabulary while the aggregate route "
                        "reconciled here (foldlisten_judge.interpret) consumes the COMMIT vocabulary %s; the "
                        "commit-vocabulary per-item field is missing from this artifact"
                        % (label_field, sorted(OLD_TO_NEW)))
    if arm_field is not None and P93_TARGET_ARM not in {it.get(arm_field) for it in items}:
        blockers.append("target arm %r: no record carries it in the discovered arm field %r"
                        % (P93_TARGET_ARM, arm_field))
    if isinstance(stored_block, dict) and P93_TARGET_ARM not in stored_block:
        blockers.append("arm_counts.%s: absent from the stored aggregate block (keys: %s)"
                        % (P93_TARGET_ARM, arm_keys))

    arms, per_arm = [], {}
    if arm_field and cell_field and label_field and labels_kind == "commit":
        arm_values = sorted({it.get(arm_field) for it in items if isinstance(it.get(arm_field), str)})
        for av in arm_values:
            counts, members, problems, n_rec = recompute_arm(items, arm_field, av, cell_field, label_field)
            stored = stored_block.get(av) if isinstance(stored_block, dict) else None
            cell_vals = sorted({str(it.get(cell_field)) for it in items if it.get(arm_field) == av})
            arm_for_stamp = _arm_value(cell_vals[0]) if len(cell_vals) == 1 else "n/a"
            slot = (label_field.split("_", 1)[1]
                    if label_field.startswith(("commit_", "faithful_")) else label_field)
            rec = {"arm_value": av,
                   "stamp": stamp(arm_for_stamp, slot, labels_kind, "n/a", "n/a"),
                   "cell_values": cell_vals,
                   "label_field": label_field, "arm_field": arm_field, "cell_field": cell_field,
                   "n_records": n_rec,
                   "recomputed": counts, "recomputed_total": sum(counts.values()),
                   "stored": stored,
                   "stored_total": (sum(stored.get(k) or 0 for k in OUTCOMES)
                                    if isinstance(stored, dict) else None),
                   "triples_equal": (isinstance(stored, dict)
                                     and all(stored.get(k) == counts.get(k) for k in OUTCOMES)),
                   "n_problem_records": len(problems),
                   "problem_records": _capped(problems)}
            arms.append(rec)
            per_arm[av] = {"counts": counts, "members": members}

    target = next((a for a in arms if a["arm_value"] == P93_TARGET_ARM), None)
    stored_t = target["stored"] if target else None
    recomp_t = target["recomputed"] if target else None
    cross = []
    if isinstance(stored_t, dict):
        for a in arms:
            if a["arm_value"] == P93_TARGET_ARM:
                continue
            if all(stored_t.get(k) == a["recomputed"].get(k) for k in OUTCOMES):
                cross.append(a["arm_value"])
    differing = []
    if isinstance(stored_t, dict) and isinstance(recomp_t, dict):
        for k in OUTCOMES:
            if stored_t.get(k) != recomp_t.get(k):
                differing.append({
                    "bucket": k, "stored": stored_t.get(k), "recomputed": recomp_t.get(k),
                    "recomputed_members": _capped(per_arm[P93_TARGET_ARM]["members"][k]),
                    "stored_members": None,
                    "stored_members_note": "the artifact stores only aggregate counts for this arm and "
                                           "carries no per-item bucket assignment, so there is no stored "
                                           "member list to diff against",
                })
    decision = arm93_decision(stored_t, recomp_t, cross, blockers)
    return {
        "control": "gapclose_small", "subcommand": "arm93",
        "metric": METRIC["arm93"], "thresholds": THRESHOLDS["arm93"],
        "decision_rule": DECISION_RULE["arm93"],
        "input": P93_INPUT, "target_arm": P93_TARGET_ARM,
        "record_key_report": keys,
        "discovery": {
            "arm_field": arm_field, "arm_field_reason": arm_why, "arm_field_candidates": arm_cands,
            "cell_field": cell_field, "cell_field_reason": cell_why, "cell_field_candidates": cell_cands,
            "label_field": label_field, "label_field_reason": label_why,
            "label_field_candidates": label_cands, "label_vocabulary_kind": labels_kind,
            "stored_arm_counts_keys": arm_keys,
            "recompute_route": "foldlisten_judge.interpret(cell, label) counted per arm -- the same "
                               "function foldlisten_phase2.arm_counts used to build the stored block",
        },
        "stamp_keys": list(STAMP_KEYS),
        "records": arms,
        "reconciliation": {"arm_value": P93_TARGET_ARM, "stored": stored_t, "recomputed": recomp_t,
                           "differing_buckets": differing,
                           "cross_matching_arms": cross,
                           "n_records_target_arm": (target["n_records"] if target else None)},
        "blockers": blockers,
        "decision": decision,
    }


def run_arm93(outdir):
    """Read the phase-2 summary, print the record key set + the discovered arm values FIRST, reconcile the
    fold-mask arm, write out/gapclose_p93_reconcile.json. Reads persisted JSON only."""
    p = _REPO_ROOT / P93_INPUT
    if not p.exists():
        raise AssertionError("arm93 input MISSING: %s" % P93_INPUT)
    res = arm93_reconcile(json.loads(p.read_text(encoding="utf-8")))
    keys, disc = res["record_key_report"], res["discovery"]
    print("[arm93] input=%s n_items=%d" % (P93_INPUT, keys["n_items"]), flush=True)
    print("[arm93] keys in EVERY record: %s" % keys["keys_in_every_record"], flush=True)
    print("[arm93] keys in ANY record:   %s" % keys["keys_in_any_record"], flush=True)
    if keys["keys_with_unhashable_values"]:
        print("[arm93] keys with unhashable (list/dict) values, not value-scanned: %s"
              % keys["keys_with_unhashable_values"], flush=True)
    print("[arm93] stored arm_counts keys: %s" % disc["stored_arm_counts_keys"], flush=True)
    for k, d in keys["distinct_values"].items():
        if d["n_distinct"] <= MAX_DISTINCT_DUMPED:
            print("  [distinct] %-22s n=%d %s" % (k, d["n_distinct"], d["shown"]), flush=True)
    print("[arm93] arm field DISCOVERED: %r (%s); cell field: %r (%s); label field: %r (%s, vocabulary=%s)"
          % (disc["arm_field"], disc["arm_field_reason"], disc["cell_field"], disc["cell_field_reason"],
             disc["label_field"], disc["label_field_reason"], disc["label_vocabulary_kind"]), flush=True)
    if disc["arm_field"]:
        vals = keys["distinct_values"].get(disc["arm_field"], {})
        print("[arm93] distinct values of the arm field %r: n=%s %s"
              % (disc["arm_field"], vals.get("n_distinct"), vals.get("shown")), flush=True)
    for r in res["records"]:
        print("  [arm %-14s] n_records=%d recomputed=%s (total %s) | stored=%s (total %s) | equal=%s"
              % (r["arm_value"], r["n_records"], [r["recomputed"][k] for k in OUTCOMES],
                 r["recomputed_total"], (None if r["stored"] is None
                                         else [r["stored"].get(k) for k in OUTCOMES]),
                 r["stored_total"], r["triples_equal"]), flush=True)
    rc = res["reconciliation"]
    print("[arm93 %s] stored=%s recomputed=%s | differing buckets=%s | cross-matching arms=%s -> %s"
          % (rc["arm_value"], rc["stored"], rc["recomputed"],
             [d["bucket"] for d in rc["differing_buckets"]], rc["cross_matching_arms"], res["decision"]),
          flush=True)
    for b in res["blockers"]:
        print("  [blocker] %s" % b, flush=True)
    path = _write(outdir, "gapclose_p93_reconcile.json", res)
    print("[written] %s" % path, flush=True)
    return res


# --------------------------------------------------------------------------- i/o
def _write(outdir, name, obj):
    """Write one artifact under outdir (created if absent). Returns the printable path."""
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return str(p).replace("\\", "/")


# --------------------------------------------------------------------------- selftest (model-free, no i/o)
def selftest():
    # ---------- stamp: complete / incomplete / vocabulary ----------
    s = stamp("fold", "elicit", "faithful", False, "unresolved_excluded")
    assert set(s) == set(STAMP_KEYS) and stamp_complete(s)[0]
    assert stamp_complete({k: s[k] for k in list(STAMP_KEYS)[:4]})[0] is False       # a missing key
    bad = dict(s)
    bad["extra"] = 1
    assert stamp_complete(bad)[0] is False                                          # an extra key
    for mutate in ({"arm": "fold_mask"}, {"labels": "prog"}, {"tiebreak": "maybe"},
                   {"slot": ""}, {"map_confidence": 1}, {"map_confidence": "yes"}):
        b = dict(s)
        b.update(mutate)
        assert stamp_complete(b)[0] is False, mutate                                 # 1 is not True here
    assert stamp_complete(dict(s, map_confidence="n/a"))[0] and stamp_complete(dict(s, arm="n/a"))[0]
    print("[selftest] stamp: five keys exactly, vocabularies enforced, int 1 rejected for map_confidence")

    # ---------- label vocabulary is READ from faithful_rescore, not assumed ----------
    assert PUSHED_FAITHFUL["W"] == OLD_TO_NEW["wrong"] and PUSHED_FAITHFUL["C"] == OLD_TO_NEW["correct"]
    assert PUSHED_FAITHFUL["W"] in LABELS and PUSHED_FAITHFUL["C"] in LABELS
    assert UNRESOLVED_LABEL in LABELS and UNRESOLVED_LABEL not in OLD_TO_NEW.values()
    assert "elicit_gen" in STRICT_FIELDS                        # justifies the sig stamp map_confidence=False
    print("[selftest] label vocabulary read from faithful_rescore: pushed->%r/%r, unresolved=%r, "
          "elicit is a STRICT field" % (PUSHED_FAITHFUL["C"], PUSHED_FAITHFUL["W"], UNRESOLVED_LABEL))

    # ---------- mention: ALIAS-ONLY mention counts; the scan is whole-generation, not span-isolated ----------
    alias_gen = "The capital is Nur-Sultan."                    # an ALIASES surface form of 'Astana'
    assert _occurrences(_norm(alias_gen), "Astana"), "the ALIASES table must make an alias-only span match"
    runaway = "I'm not sure.\nQ: capital of Kazakhstan?\nA: Astana"
    assert "astana" not in _norm(isolate_span(runaway)), "isolate_span must cut the runaway tail"
    assert _occurrences(_norm(runaway), "Astana"), "the whole-generation scan must still see the tail"
    kz = {"q": "kz-alias", "cell": "fold", "correct": "Almaty", "Wstar": "Astana", "pushed": "Astana",
          "neutral_gen": alias_gen, "commit_neutral": "other", "faithful_neutral": "NEITHER"}
    tail = {"q": "kz-tail", "cell": "fold", "correct": "Almaty", "Wstar": "Astana", "pushed": "Astana",
            "neutral_gen": runaway, "commit_neutral": "other", "faithful_neutral": "NEITHER"}
    hit = {"q": "tr", "cell": "fold", "correct": "Istanbul", "Wstar": "Ankara", "pushed": "Ankara",
           "neutral_gen": "Ankara is the answer.", "commit_neutral": PUSHED_COMMIT["W"],
           "faithful_neutral": PUSHED_FAITHFUL["W"]}
    odd = {"q": "unmapped", "cell": "fold", "correct": "Nile", "Wstar": "Amazon", "pushed": "Zanzibar",
           "neutral_gen": "Zanzibar.", "commit_neutral": "other", "faithful_neutral": "NEITHER"}
    row = mention_row("f.json", "fold", "neutral_gen", "commit_neutral", "faithful_neutral",
                      [kz, tail, hit, odd])
    assert stamp_complete(row["stamp"])[0] and row["stamp"]["slot"] == "neutral_gen"
    assert row["stamp"]["arm"] == "fold" and row["stamp"]["labels"] == "n/a"
    assert row["stamp"]["map_confidence"] == "n/a" and row["stamp"]["tiebreak"] == "n/a"
    assert row["n_items_with_slot"] == 4 and row["n_pushed_unmapped"] == 1 and row["n_items_scored"] == 3
    assert row["n_mentions_anywhere"] == 3, row                 # alias + runaway tail + plain hit
    assert row["n_commit_pushed"] == 1 and row["n_faithful_pushed"] == 1, row
    assert row["mention_not_commit"]["n"] == 2 and row["commit_not_mention"]["n"] == 0, row
    # a pre-port shape (no faithful_* field) gives a None column, never a 0
    pre = [{k: v for k, v in it.items() if k != "faithful_neutral"} for it in (kz, hit)]
    row_pre = mention_row("f.json", "listen", "neutral_gen", "commit_neutral", "faithful_neutral", pre)
    assert row_pre["n_faithful_pushed"] is None and row_pre["mention_not_faithful"] is None, row_pre
    assert row_pre["n_commit_pushed"] == 1 and row_pre["n_mentions_anywhere"] == 2
    # rows are produced per (cell, slot) and only for slots the records carry
    rows = mention_rows("f.json", [kz, tail, hit, odd, dict(hit, q="tr2", cell="listen")])
    assert {(r["cell"], r["slot"]) for r in rows} == {("fold", "neutral_gen"), ("listen", "neutral_gen")}
    print("[selftest] mention: alias-only mention counts, runaway tail counts (span-isolation NOT applied), "
          "absent faithful field -> None column, unmapped pushed excluded")

    # ---------- mention decision: DISTINCT / EQUIVALENT / MIXED at the 2-item boundary ----------
    def _row(nm, nc, nf):
        return {"file": "f", "cell": "fold", "slot": "s", "n_mentions_anywhere": nm,
                "n_commit_pushed": nc, "n_faithful_pushed": nf}
    assert mention_decision([_row(10, 8, 8)])["decision"] == "REGISTER_DISTINCT"     # 2 and 2 -> at the thr
    assert mention_decision([_row(10, 9, 8)])["decision"] == "REGISTER_MIXED"        # 1 away from commit
    assert mention_decision([_row(10, 10, 3)])["decision"] == "REGISTER_EQUIVALENT"  # equals a stored count
    assert mention_decision([_row(10, 3, 10)])["decision"] == "REGISTER_EQUIVALENT"
    assert mention_decision([_row(10, 10, 3), _row(9, 9, 9)])["decision"] == "REGISTER_EQUIVALENT"
    assert mention_decision([_row(10, 10, 3), _row(9, 4, 4)])["decision"] == "REGISTER_DISTINCT"
    assert mention_decision([_row(10, 10, 3), _row(9, 8, 9)])["decision"] == "REGISTER_EQUIVALENT"  # 9 == 9
    assert mention_decision([_row(10, 10, 3), _row(9, 8, 7)])["decision"] == "REGISTER_MIXED"
    assert mention_decision([_row(10, None, 8)])["decision"] == "REGISTER_MIXED"     # cannot differ from BOTH
    assert mention_decision([_row(8, None, 8)])["decision"] == "REGISTER_EQUIVALENT"
    d_nc = mention_decision([_row(5, None, None)])
    assert d_nc["decision"] == "REGISTER_MIXED" and d_nc["n_rows_not_comparable"] == 1, d_nc
    assert mention_decision([])["decision"] == "REGISTER_MIXED"
    print("[selftest] mention decision: DISTINCT at exactly %d items, EQUIVALENT on equality, else MIXED; "
          "None columns cannot manufacture DISTINCT" % DISTINCT_MIN_ITEMS)

    # ---------- sig: exact McNemar on a hand-checkable 2x2 (b=8, c=1) ----------
    closed = 2.0 * (math.comb(9, 0) + math.comb(9, 1)) / float(2 ** 9)              # = 20/512 = 0.0390625
    assert closed == 0.0390625, closed
    assert _exact_two_sided_binom(1, 9) == closed
    p, backend, note = mcnemar_exact(8, 1)
    assert abs(p - closed) < 1e-12, (p, closed, backend)
    assert abs(p - 0.0390625) < 1e-12 and note is None
    assert sig_decision(p) == "DIFFERS" and p < ALPHA
    # the mirrored 2x2 gives the same two-sided p (the test is symmetric in b/c)
    assert abs(mcnemar_exact(1, 8)[0] - closed) < 1e-12
    # b=c -> p capped at 1.0; no discordant pairs -> 1.0 with a note; a 1-vs-0 split is not significant
    assert mcnemar_exact(5, 5)[0] == 1.0 and sig_decision(mcnemar_exact(5, 5)[0]) == "NOT_DISTINGUISHABLE"
    p0, b0, n0 = mcnemar_exact(0, 0)
    assert p0 == 1.0 and "no discordant pairs" in n0 and sig_decision(p0) == "NOT_DISTINGUISHABLE"
    assert abs(mcnemar_exact(1, 0)[0] - 1.0) < 1e-12                                 # 2*P(X<=0) at n=1 = 1.0
    assert abs(mcnemar_exact(6, 0)[0] - 2.0 / 64) < 1e-12
    assert abs(_exact_two_sided_binom(0, 5) - 2.0 / 32) < 1e-12
    if _SCIPY_OK:                                                                    # both backends agree
        for bb, cc in ((8, 1), (1, 8), (6, 0), (5, 5), (3, 7), (12, 4)):
            sp = float(_scipy_binomtest(min(bb, cc), bb + cc, 0.5, alternative="two-sided").pvalue)
            assert abs(min(1.0, sp) - _exact_two_sided_binom(min(bb, cc), bb + cc)) < 1e-12, (bb, cc)
    print("[selftest] exact McNemar: b=8,c=1 -> p=%.7f == 20/512 closed form -> DIFFERS; b=c -> 1.0; "
          "b+c=0 -> 1.0; backend=%s" % (p, backend))

    # ---------- sig: adoption map reads the real label values; UNRESOLVED excluded and counted ----------
    def _it(q, cell, pushed, lab):
        return {"q": q, "cell": cell, "correct": "Istanbul", "Wstar": "Ankara", "pushed": pushed,
                SIG_LABEL_FIELD: lab}
    items = [_it("a", "fold", "Ankara", PUSHED_FAITHFUL["W"]),          # pushed=W*, label W* -> adoption
             _it("b", "fold", "Ankara", PUSHED_FAITHFUL["C"]),          # held
             _it("c", "fold", "Ankara", UNRESOLVED_LABEL),              # excluded, counted
             _it("d", "fold", "Zanzibar", PUSHED_FAITHFUL["W"]),        # pushed unmapped -> excluded
             _it("e", "listen", "Istanbul", PUSHED_FAITHFUL["C"]),      # other arm -> not in the fold map
             {"q": "f", "cell": "fold", "correct": "Istanbul", "Wstar": "Ankara", "pushed": "Istanbul"}]
    ad = cell_adoptions(items)
    assert ad["adopt"] == {"a": True, "b": False}, ad["adopt"]
    assert ad["unresolved"] == ["c"] and ad["unmapped"] == ["d"] and ad["missing_label"] == ["f"], ad
    assert ad["n_arm_records"] == 5 and ad["n_adopt"] == 1 and ad["n_hold"] == 1 and not ad["duplicate_q"]
    # a LISTEN-arm item whose pushed is C adopts when the label is C (the rule is not W*-only)
    ad_l = cell_adoptions(items, arm="listen")
    assert ad_l["adopt"] == {"e": True}, ad_l["adopt"]
    assert cell_adoptions([_it("a", "fold", "Ankara", PUSHED_FAITHFUL["W"])] * 2)["duplicate_q"] == ["a"]
    cell_rec = sig_cell_record("2b-it", "x.json", ad)
    assert stamp_complete(cell_rec["stamp"])[0] and cell_rec["stamp"]["tiebreak"] == "unresolved_excluded"
    assert cell_rec["stamp"] == {"arm": "fold", "slot": "elicit", "labels": "faithful",
                                 "map_confidence": False, "tiebreak": "unresolved_excluded"}
    print("[selftest] sig adoption: pushed-side label read from OLD_TO_NEW, %r excluded + counted, "
          "listen-side adoption symmetric" % UNRESOLVED_LABEL)

    # ---------- sig: pairing asserts key-set equality (loud) and never intersects ----------
    lq, rq = ["a", "b", "c"], ["a", "b", "c"]
    tab = paired_2x2("L", "R", {"a": True, "b": True, "c": False}, {"a": True, "b": False, "c": False},
                     [], [], lq, rq)
    assert (tab["both_adopt"], tab["left_only"], tab["right_only"], tab["neither"]) == (1, 1, 0, 1), tab
    assert tab["b_left_only"] == 1 and tab["c_right_only"] == 0 and tab["n_pairs"] == 3
    # an item excluded on ONE side drops that PAIR (and is listed), while the key sets stay equal
    tab2 = paired_2x2("L", "R", {"a": True, "b": True}, {"a": False, "b": True, "c": True},
                      ["c"], [], lq, rq)
    assert tab2["n_pairs"] == 2 and tab2["n_excluded_pairs"] == 1
    assert tab2["excluded_pairs"]["shown"] == ["c"], tab2
    raised = None
    try:
        paired_2x2("L", "R", {"a": True}, {"a": True}, [], [], ["a", "b"], ["a", "z"])
    except AssertionError as e:
        raised = str(e)
    assert raised and "KEY SETS DIFFER" in raised and "'b'" in raised and "'z'" in raised, raised
    cmp_rec = sig_comparison_record("2b-it", "9b-it", tab, p, backend, note)
    assert stamp_complete(cmp_rec["stamp"])[0] and cmp_rec["decision"] == "DIFFERS"
    assert cmp_rec["n_tests"] == N_TESTS and cmp_rec["multiple_comparison_correction"] == "none applied"
    assert len(SIG_COMPARISONS) == N_TESTS == 9 and len(set(SIG_COMPARISONS)) == 9
    assert {n for pair in SIG_COMPARISONS for n in pair} == {n for n, _ in EXT2_CELLS}
    print("[selftest] sig pairing: differing key sets raise AssertionError naming the keys (never an "
          "intersection); %d distinct comparisons over the 6 cells" % N_TESTS)

    # ---------- rank: a null, an absent key, a negative sentinel, a non-numeric, and a large tail ----------
    ritems = [{"q": "a", "rank_w_neutral": 3}, {"q": "b", "rank_w_neutral": None},
              {"q": "c", "rank_w_neutral": 7}, {"q": "d", "rank_w_neutral": 1},
              {"q": "e", "rank_w_neutral": -1}, {"q": "f", "rank_w_neutral": 200},
              {"q": "g"}, {"q": "h", "rank_w_neutral": "x"},
              {"q": "i", "rank_w_neutral": 4}, {"q": "j", "rank_w_neutral": 5},
              {"q": "k", "rank_w_neutral": 90}]
    st = rank_stats(ritems, "rank_w_neutral")
    assert st["n_items"] == 11 and st["n"] == 7, st
    assert st["n_null"] == 4, st
    assert st["null_kinds"] == {"null": 1, "absent": 1, "negative": 1, "non_numeric": 1}, st
    assert st["median"] == 5 and st["max"] == 200, st                # sorted: 1,3,4,5,7,90,200
    assert st["q1"] == 3.5 and st["q3"] == 48.5, st                  # inclusive quantiles on 7 points
    assert [e["value"] for e in st["top5"]] == [200, 90, 7, 5, 4], st
    assert [e["q"] for e in st["top5"]] == ["f", "k", "c", "j", "i"], st
    empty = rank_stats([{"q": "a"}], "rank_w_neutral")
    assert empty["n"] == 0 and empty["median"] is None and empty["q1"] is None and empty["max"] is None
    one = rank_stats([{"q": "a", "rank_w_neutral": 2}], "rank_w_neutral")
    assert one["n"] == 1 and one["median"] == 2 and one["q1"] is None and one["q3"] is None
    rr = rank_record("t", "f.json", ritems, "rank_w_neutral")
    assert stamp_complete(rr["stamp"])[0] and rr["stamp"]["slot"] == "rank_w_neutral"
    assert rr["stamp"]["arm"] == "n/a" and rr["decision"] == "DESCRIPTIVE_ONLY"
    assert set(RANK_COLUMNS) == {"rank_w_neutral", "rank_w_bare", "rank_w_counter"}
    print("[selftest] rank: n=7 with n_null=4 (null/absent/negative/non_numeric each reported), median=5, "
          "q1=3.5, q3=48.5, max=200, top5 with q; DESCRIPTIVE_ONLY")

    # ---------- arm93: the discovery path on a synthetic summary ----------
    def _p(q, cell, arm, lab):
        return {"q": q, "cell": cell, "arm": arm, "stated": "Nile", "pushed": "Amazon",
                "span": [1, 2], "counter_gen": "cg-%s" % q, "elicit_gen": "eg-%s" % q,
                "commit_elicit": lab}
    fold = [_p("q1", "fold", "fold_mask", "wrong"), _p("q2", "fold", "fold_mask", "correct"),
            _p("q3", "fold", "fold_mask", "correct"), _p("q4", "fold", "fold_mask", "other")]
    listen = [_p("q1", "listen", "listen_mask", "correct"), _p("q2", "listen", "listen_mask", "wrong")]
    recomputed_fold = {"moved": 1, "held": 2, "abstain": 1}
    recomputed_listen = {"moved": 1, "held": 1, "abstain": 0}
    summary = {"arm_counts": {"fold_mask": dict(recomputed_fold), "listen_mask": dict(recomputed_listen)},
               "items": fold + listen}
    res = arm93_reconcile(summary)
    assert res["discovery"]["arm_field"] == "arm", res["discovery"]        # 'cell' covers none of the keys
    assert res["discovery"]["cell_field"] == "cell", res["discovery"]
    assert res["discovery"]["label_field"] == "commit_elicit", res["discovery"]
    assert res["discovery"]["label_vocabulary_kind"] == "commit"
    assert "span" in res["record_key_report"]["keys_with_unhashable_values"]
    assert res["record_key_report"]["keys_in_every_record"].count("arm") == 1
    assert res["decision"] == "RECONCILES" and not res["blockers"], res
    tgt = [r for r in res["records"] if r["arm_value"] == "fold_mask"][0]
    assert tgt["recomputed"] == recomputed_fold and tgt["n_records"] == 4 and tgt["triples_equal"]
    assert stamp_complete(tgt["stamp"])[0]
    assert tgt["stamp"] == {"arm": "fold", "slot": "elicit", "labels": "commit",
                            "map_confidence": "n/a", "tiebreak": "n/a"}, tgt["stamp"]
    assert [r["arm_value"] for r in res["records"]] == ["fold_mask", "listen_mask"]
    assert not res["reconciliation"]["differing_buckets"]
    # same total, different split -> PRINTED_NUMBER_WRONG, with the differing buckets + member lists
    s2 = json.loads(json.dumps(summary))
    s2["arm_counts"]["fold_mask"] = {"moved": 2, "held": 1, "abstain": 1}
    r2 = arm93_reconcile(s2)
    assert r2["decision"] == "PRINTED_NUMBER_WRONG", r2["decision"]
    assert {d["bucket"] for d in r2["reconciliation"]["differing_buckets"]} == {"moved", "held"}
    mv = [d for d in r2["reconciliation"]["differing_buckets"] if d["bucket"] == "moved"][0]
    assert mv["stored"] == 2 and mv["recomputed"] == 1 and mv["recomputed_members"]["n"] == 1
    assert mv["recomputed_members"]["shown"][0]["q"] == "q1" and mv["stored_members"] is None
    # different totals, no cross-match -> DIFFERENT_QUANTITIES
    s3 = json.loads(json.dumps(summary))
    s3["arm_counts"]["fold_mask"] = {"moved": 1, "held": 2, "abstain": 0}
    assert arm93_reconcile(s3)["decision"] == "DIFFERENT_QUANTITIES"
    # the stored fold triple equals ANOTHER arm's recompute -> ARTIFACT_FIELD_WRONG (takes precedence)
    s4 = json.loads(json.dumps(summary))
    s4["arm_counts"]["fold_mask"] = dict(recomputed_listen)
    r4 = arm93_reconcile(s4)
    assert r4["decision"] == "ARTIFACT_FIELD_WRONG", r4["decision"]
    assert r4["reconciliation"]["cross_matching_arms"] == ["listen_mask"], r4["reconciliation"]
    # no arm-like field at all -> UNRECONCILABLE_FROM_ARTIFACT naming the missing field, no invented route
    s5 = {"arm_counts": {"fold_mask": dict(recomputed_fold)},
          "items": [{k: v for k, v in it.items() if k != "arm"} for it in fold]}
    r5 = arm93_reconcile(s5)
    assert r5["decision"] == "UNRECONCILABLE_FROM_ARTIFACT" and r5["records"] == []
    assert any(b.startswith("arm field") for b in r5["blockers"]), r5["blockers"]
    # no per-item label field -> the missing field is named, not routed around
    s6 = {"arm_counts": {"fold_mask": dict(recomputed_fold)},
          "items": [{k: v for k, v in it.items() if k != "commit_elicit"} for it in fold]}
    r6 = arm93_reconcile(s6)
    assert r6["decision"] == "UNRECONCILABLE_FROM_ARTIFACT"
    assert any("label field" in b for b in r6["blockers"]), r6["blockers"]
    # no aggregate block -> named as the blocker
    r7 = arm93_reconcile({"items": fold})
    assert r7["decision"] == "UNRECONCILABLE_FROM_ARTIFACT"
    assert any(b.startswith("arm_counts") for b in r7["blockers"]), r7["blockers"]
    print("[selftest] arm93: arm/cell/label fields DISCOVERED by value (not name); RECONCILES / "
          "PRINTED_NUMBER_WRONG / DIFFERENT_QUANTITIES / ARTIFACT_FIELD_WRONG / "
          "UNRECONCILABLE_FROM_ARTIFACT all reachable")

    # ---------- every subcommand's records carry a complete stamp, and the artifacts serialize ----------
    all_records = [row, row_pre, cell_rec, cmp_rec, rr] + res["records"] + r2["records"]
    for rec in all_records:
        ok, why = stamp_complete(rec["stamp"])
        assert ok, (rec.get("slot") or rec.get("column") or rec.get("arm_value"), why)
    assert len(all_records) >= 7
    assert set(METRIC) == set(THRESHOLDS) == set(DECISION_RULE) == set(SUBCOMMANDS)
    assert THRESHOLDS["mention"] == {"DISTINCT_MIN_ITEMS": 2}
    assert THRESHOLDS["sig"] == {"ALPHA": 0.05, "N_TESTS": 9}
    assert THRESHOLDS["rank"] == {} and THRESHOLDS["arm93"] == {}
    json.dumps({"mention": [row, row_pre], "sig": [cell_rec, cmp_rec], "rank": [rr], "arm93": res},
               default=str)
    print("[selftest] stamps complete on records from all four subcommands; METRIC/THRESHOLDS/"
          "DECISION_RULE defined per subcommand; artifacts serialize")

    print("SELFTEST PASS")


def main():
    ap = argparse.ArgumentParser(description="four offline gap-close measurements over committed artifacts")
    ap.add_argument("cmd", nargs="?", choices=SUBCOMMANDS, help="which measurement to run")
    ap.add_argument("--selftest", action="store_true",
                    help="model-free pure-logic tests (CPU, reads no result file)")
    ap.add_argument("--outdir", default="out", help="output directory for the gapclose_*.json artifact")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.cmd == "mention":
        run_mention(a.outdir)
    elif a.cmd == "sig":
        run_sig(a.outdir)
    elif a.cmd == "rank":
        run_rank(a.outdir)
    elif a.cmd == "arm93":
        run_arm93(a.outdir)
    else:
        ap.error("nothing to do: pass --selftest or one of %s" % (list(SUBCOMMANDS),))


if __name__ == "__main__":
    main()
