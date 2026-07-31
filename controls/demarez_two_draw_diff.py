"""TWO-DRAW FIELD DIFF for the De Marez span artifacts (offline, model-free, CPU-only: no torch, no GPU,
no network, nothing re-generated and nothing re-scored).

WHAT IT IS. Two independent draws of the SAME registered run exist on disk (the same instrument, the same
tag, the same frozen family, run twice on two boxes). This control takes the two summaries and counts, per
field, how many stored fields disagree. It is claim-blind: it asks only whether the same numbers and the
same bytes came back, and it attaches no meaning to either outcome. Neither outcome is a success state of
this instrument.

INPUTS. Two summaries written by controls/foldlisten_demarez_subst.py or controls/foldlisten_demarez_mask.py
(both draws must come from the SAME one of the two -- a subst/mask pair is NOT_COMPARABLE by construction).
Both record shapes the two writers persist are read, using the reader convention the offline join already
accepts (controls/foldlisten_demarez_join.py:499-518):
  * FLAT per-arm records carrying `turn_id`, in the top-level `items` list  (foldlisten_demarez_subst.py
    :1240-1261);
  * per-ITEM records carrying an `arms` object of per-arm records          (foldlisten_demarez_mask.py
    :1309, :1423). For that shape the PARENT record's own fields (q / join_key / correct / Wstar /
    span_record / b7_padding) are kept as ONE further record under the reserved turn_id "__item__", so
    nothing a writer persists per item is silently dropped from the diff.
Distribution records live under `distributions` (subst) or `dist` (mask); both names are read.

THE JOIN. Records are joined on the identity pair (item, turn_id) -- the pair each writer makes unique per
record. Duplicates on either side, a record carrying no item/turn_id, or an unreadable element make the
draw unreadable and are reported as NOT_COMPARABLE, never absorbed into a difference count. Inside a joined
pair the remaining identity fields (q, join_key, correct, Wstar, cell, arm, stated, pushed, push_target)
are CHECKED: a disagreement there means the two draws do not carry the same item at that key, so the pair
would compare unrelated records -- also NOT_COMPARABLE, never a mismatch count.

WHAT IT COUNTS, in five classes (complete counts, never truncated; only the worked-example dump is capped):
  1 gen_bytes     `counter_gen`, `elicit_gen`, `elicit_prior_gen`, `counter_prompt`, `elicit_prompt` -- the
                  bytes the model and the harness produced. Reported SEPARATELY from everything else and
                  resolved FIRST after NOT_COMPARABLE, because it is the strongest single signal available
                  to this instrument.
  2 labels        `commit_v2`, `commit_v1`, `commit_elicit`, `faithful_strict`, `faithful_strict_rule`,
                  `faithful_strict_span`, `faithful_strict_commit`, `faithful_strict_as_commit`, `outcome`,
                  `cell_outcome` -- read EXACTLY as stored. No scorer is imported; nothing is re-scored.
  3 distributions inside `distributions`/`dist`, per position: `argmax_tok_id`, `argmax_tok_str`; the
                  `topk_10` list element-wise on `tok_id`, `p` and `p_full`; and for each entity-key read
                  (`reads_c_space`, `reads_c_bare`, `reads_w_space`, `reads_w_bare`) the seven ENTKEY
                  fields `tok_id`, `p_full`, `lp_first`, `p_underflow`, `rank_first_tok`, `tie_plateau`,
                  `first_token_collision`.
  4 aggregates    the arm-keyed tables the writers persist: `arm_counts`, `arm_rates`, `arm_r_off`,
                  `arm_stats`, `arm_turn_content_tokens`, `arm_counts_located_subset`,
                  `arm_rates_located_subset`, `r_off`, `insufficient_eval_located`,
                  `dissociation_columns`.
  5 record_other  every OTHER field a joined record carries (turn text, token counts, spans, mask ranges,
                  stamps, axes, the margin fields, `tok_str`, the dist records' own axes). Compared so that
                  no persisted field is silently dropped; a difference here can only make an identity
                  statement harder, never easier.
STRUCTURAL, counted separately and never merged into a value count: records present in one draw and not the
other, fields present in one aligned record (or dist record, or topk row, or entity-key sub-record) and not
the other, top-level summary keys present in one file and not the other, dist positions present on one side
only, and list-length / container-shape differences.

FLOATS. Every float-bearing comparison is counted TWICE: an EXACT-equality count and a count OUTSIDE
--float-tol. `p_full` is repr(float(p)) -- an exactly round-tripping decimal STRING (the writers'
FULL_FIELD_CONVENTION) -- so it is compared both as a string (exact) and, for the tolerance count, parsed
back to a float. NaN never equals NaN and is never within tolerance; None equals only None; a bool equals
only a bool; a value that is a string on one side and a number on the other is never within tolerance.

OUT OF SCOPE BY CONSTRUCTION, and named in the artifact rather than dropped in silence: `provenance` (it
stamps the box, the clock and the instance id, and two draws on two boxes differ there by construction);
the writers' prose/threshold/registration blocks (constants of the instrument, not measurements); and the
writers' own derived verdict blocks (`decisions_recomputable_offline`, `provisional_verdicts`,
`concordance_columns`, `mask_totality_audit`, `span_locatability`, `span_stability`, `dist_contract`,
`cost`, `primary_readout`) -- each is a pure function of numbers this control already compares field by
field, so comparing them again would double-count, not add coverage. Their KEY PRESENCE is still checked
(top-level schema asymmetry is structural).

  python3 controls/demarez_two_draw_diff.py --selftest
  python3 controls/demarez_two_draw_diff.py \
      --draw-a out/foldlisten_demarez_subst_dmz_9bit_a_summary.json \
      --draw-b results_dmz_9bit_r2/out/foldlisten_demarez_subst_dmz_9bit_a_summary.json
  python3 controls/demarez_two_draw_diff.py \
      --draw-a out/foldlisten_demarez_mask_dmz_9bit_b_summary.json \
      --draw-b results_dmz_9bit_r2/out/foldlisten_demarez_mask_dmz_9bit_b_summary.json
"""
import argparse
import json
import sys
from pathlib import Path

# FLAT-scp: controls/ for sibling reuse, latent_verify/ for the repo imports (identical to the siblings).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The shipped five-key stamp, IMPORTED rather than transcribed. controls/foldlisten_demarez_join.py:88 --
# the closest sibling in purpose, and like this file an OFFLINE-only instrument that is never scp'd to a
# box -- imports the same constant from the same module, after recording (:86) that the chain is
# numpy-free and torch-free at module level. So this import pulls no model machinery and no heavy
# dependency. The GPU writers TRANSCRIBE the same tuple instead, because they ARE shipped and
# gapclose_item_joins.py is not in the scp list; that constraint does not apply here.
from gapclose_item_joins import STAMP_KEYS  # noqa: E402

# --------------------------------------------------------------------------- FROZEN thresholds
# MAX_FIELD_MISMATCH: a COUNT threshold, fixed at 0 by the definition of the measured quantity and not
#   calibrated from anything. The measurement is "how many stored fields disagree"; 0 is the only count
#   that means "none disagree", so every class is compared against 0. It is not a tuning knob and is not
#   exposed on the command line.
MAX_FIELD_MISMATCH = 0
# FLOAT_TOL_DEFAULT: see FLOAT_TOL_PROVENANCE below -- the repo's existing float-noise literal, reused,
#   with its ORIGINAL role disclosed. Overridable with --float-tol; --float-tol 0.0 collapses the
#   tolerance count onto the exact-equality count.
FLOAT_TOL_DEFAULT = 1e-9
FLOAT_TOL_PROVENANCE = (
    "1e-9. SOURCE: the repo's existing float-noise epsilon literal -- controls/foldlisten_demarez_subst.py"
    ":108 (BOUNDARY_EPS) and controls/foldlisten_demarez_mask.py:100 (EPS_F), both citing "
    "controls/foldlisten_phase3c_riders.py:128, and controls/foldlisten_demarez_join.py:97 (EPS) citing "
    "the same line. DISCLOSURE, so the provenance is not dressed up as more than it is: in every one of "
    "those places the constant is an INCLUSIVE-BOUNDARY epsilon for threshold comparisons (a >= b - eps), "
    "NOT a declared tolerance for numeric reproduction between two draws. No constant in this repo is "
    "declared as a cross-draw numeric-reproduction tolerance, so this default is the repo's float-noise "
    "literal REUSED here and nothing stronger. It is a REPORTING FLAG only: the exact-equality count is "
    "reported beside every tolerance count for every class, so a reader who rejects this default can read "
    "the exact counts instead, or re-run with --float-tol 0.0 and get the same numbers."
)
# MAX_EXAMPLES / MAX_STRUCTURAL: reporting caps chosen here (nothing in the repo fixes them). They bound
#   the worked-example and structural-entry DUMPS only. Every count is complete and no decision depends on
#   them; the number omitted is itself recorded.
MAX_EXAMPLES = 20            # per class
MAX_STRUCTURAL = 50          # total structural entries dumped

# --------------------------------------------------------------------------- FROZEN field scope
# Class 1: the bytes the model and the harness produced. subst persists counter_gen/elicit_gen +
# counter_prompt/elicit_prompt; mask persists those plus elicit_prior_gen (the text spliced into the
# elicit prompt in place of the model's own counter reply on the B5 arm).
GEN_BYTE_FIELDS = ("counter_gen", "elicit_gen", "elicit_prior_gen", "counter_prompt", "elicit_prompt")
# Class 2: a scorer's stored reading of a generation, as opposed to the generation itself. Read as stored.
LABEL_FIELDS = ("commit_v2", "commit_v1", "commit_elicit", "faithful_strict", "faithful_strict_rule",
                "faithful_strict_span", "faithful_strict_commit", "faithful_strict_as_commit",
                "outcome", "cell_outcome")
# Identity: if any of these disagree inside a joined pair the two draws are not carrying the same item at
# that key. Reported as NOT_COMPARABLE, never counted as a difference.
IDENTITY_FIELDS = ("item", "q", "join_key", "correct", "Wstar", "cell", "arm", "turn_id",
                   "stated", "pushed", "push_target")
DIST_CONTAINER_KEYS = ("distributions", "dist")     # subst / mask names for the same object
# Class 3, exactly the fields named in the measurement scope.
DIST_SCOPE_SCALARS = ("argmax_tok_id", "argmax_tok_str")
DIST_TOPK_KEY = "topk_10"
DIST_TOPK_SCOPE = ("tok_id", "p", "p_full")         # tok_id AND the probability, in both stored forms
DIST_READ_NAMES = ("reads_c_space", "reads_c_bare", "reads_w_space", "reads_w_bare")
ENTKEY_SCOPE = ("tok_id", "p_full", "lp_first", "p_underflow",
                "rank_first_tok", "tie_plateau", "first_token_collision")
# Class 4: the arm-keyed tables the two writers persist (foldlisten_demarez_subst.py:1375-1376, :1403;
# foldlisten_demarez_mask.py:1550-1552). Any of these a draw does not carry is simply not compared; a
# presence ASYMMETRY between the two draws is structural.
AGGREGATE_KEYS = ("arm_counts", "arm_rates", "arm_r_off", "arm_stats", "arm_turn_content_tokens",
                  "arm_counts_located_subset", "arm_rates_located_subset", "r_off",
                  "insufficient_eval_located", "dissociation_columns")
# Top-level identity: a disagreement (or a presence asymmetry) here means the pair is the wrong pair.
TOP_LEVEL_ID_KEYS = ("instrument", "run", "name", "registered_name", "family", "tag", "regime", "cell")
# The OPERATIVE rule at the top level is: only AGGREGATE_KEYS are value-compared; `items`/`records` are the
# record lists (joined, then compared record by record); every OTHER top-level key is checked for KEY
# PRESENCE only. The tuple below is the ENUMERATION of the keys that currently fall outside the aggregate
# scope in the two writers' schemas -- carried in the artifact so a reader can see exactly what was left
# out without diffing the schemas by hand. It is documentation of the rule, not the rule itself: a key
# neither writer has today is handled correctly without editing it.
TOP_LEVEL_NOT_VALUE_COMPARED = (
    "provenance", "metric", "decision_rule", "registration", "thresholds", "arms", "arm_strings_source",
    "registers", "rate_conventions", "floors_cited", "cited_never_recomputed", "full_field_convention",
    "hook_free", "hook_free_note", "margin_framing", "dissociation_note", "rule_k", "dist_fields",
    "entkey_fields", "dist_contract", "span_locatability", "span_stability", "mask_totality_audit",
    "decisions_recomputable_offline", "provisional_verdicts", "concordance_columns", "primary_readout",
    "verdict_authority", "not_emitted_here", "spec_ambiguities", "stamp_keys", "axis_keys", "cost",
    "n_items_measured", "n_items", "N_ITEMS_registered", "device", "run_label", "run_wide_stamp",
    "n_primary_role_fields")

PARENT_TURN_ID = "__item__"          # reserved turn_id for the per-ITEM parent record of the nested shape
RECORD_LIST_KEYS = ("items", "records")

CLASSES = ("gen_bytes", "labels", "distributions", "aggregates", "record_other", "identity")
TOL_CLASSES = ("distributions", "aggregates", "record_other")   # the classes a float tolerance can apply to

# The decision space, IN RESOLUTION ORDER (earlier wins).
DECISIONS = ("NOT_COMPARABLE",
             "GENERATED_BYTES_DIFFER",
             "RECORD_SET_OR_SCHEMA_DIFFERS",
             "LABELS_DIFFER",
             "DISTRIBUTIONS_DIFFER",
             "AGGREGATES_DIFFER",
             "OTHER_RECORD_FIELDS_DIFFER",
             "DIFFERS_WITHIN_FLOAT_TOL_ONLY",
             "BYTE_IDENTICAL")

METRIC = (
    "Offline field-level diff of TWO draws of the same registered De Marez run (same instrument, same tag, "
    "same frozen family, two independent artifacts). Per-record join on the identity pair (item, turn_id) "
    "across both persisted record shapes (a flat per-arm list, or per-item records carrying an `arms` "
    "object whose parent fields are kept under the reserved turn_id '__item__'); the remaining identity "
    "fields (q, join_key, correct, Wstar, cell, arm, stated, pushed, push_target) are checked inside each "
    "joined pair. Counted, per field, never truncated: (1) gen_bytes -- counter_gen, elicit_gen, "
    "elicit_prior_gen, counter_prompt, elicit_prompt; (2) labels -- commit_v2, commit_v1, commit_elicit, "
    "faithful_strict, faithful_strict_rule, faithful_strict_span, faithful_strict_commit, "
    "faithful_strict_as_commit, outcome, cell_outcome, read exactly as stored; (3) distributions -- inside "
    "`distributions`/`dist` at each position: argmax_tok_id, argmax_tok_str, topk_10 element-wise on "
    "tok_id/p/p_full, and for each of reads_c_space, reads_c_bare, reads_w_space, reads_w_bare the seven "
    "fields tok_id, p_full, lp_first, p_underflow, rank_first_tok, tie_plateau, first_token_collision; "
    "(4) aggregates -- the arm-keyed tables arm_counts, arm_rates, arm_r_off, arm_stats, "
    "arm_turn_content_tokens, arm_counts_located_subset, arm_rates_located_subset, r_off, "
    "insufficient_eval_located, dissociation_columns; (5) record_other -- every other field a joined record "
    "carries, so nothing persisted is silently dropped. Structural differences (records present in one "
    "draw only, fields present in one schema only, dist positions on one side only, list-length and "
    "container-shape differences) are counted separately and never merged into a value count. Every "
    "float-bearing comparison is counted twice: exact equality, and outside --float-tol. Nothing is "
    "re-generated, nothing is re-scored, no scorer or gate is imported, and `provenance` is out of scope "
    "because two draws on two boxes differ there by construction."
)

DECISION_RULE = (
    "Counts only. Thresholds: MAX_FIELD_MISMATCH = 0 (a count threshold fixed by the definition of the "
    "measurement -- 0 is the only count that means 'none disagree' -- not calibrated from data and not "
    "exposed on the command line) and float_tol (default 1e-9; see thresholds.float_tol_provenance). The "
    "gen_bytes and labels classes are byte/string fields and are decided on their EXACT-mismatch counts, "
    "to which no tolerance is applied; the distributions, aggregates and record_other classes are decided "
    "on their OUTSIDE-TOLERANCE counts, with their exact counts reported beside. Resolution order, TOTAL, "
    "the EARLIER branch winning wherever two are co-satisfiable: "
    "(1) NOT_COMPARABLE -- a top-level identity key differs or is present on one side only (including a "
    "subst/mask pair or two different tags), a draw carries neither `items` nor `records` as a list, an "
    "element is neither a flat record with `turn_id` nor an object with an `arms` object, a record carries "
    "no `item` or no `turn_id`, an (item, turn_id) pair repeats within a draw, an identity field disagrees "
    "inside a joined pair, or nothing joined at all. NOT_COMPARABLE is never a match and is never merged "
    "into a difference category. "
    "(2) GENERATED_BYTES_DIFFER -- n_gen_bytes_mismatch > 0 over the joined records. Placed first among "
    "the difference branches because a byte difference in a generation or a prompt is the strongest single "
    "signal this instrument can read; it is counted over the joined intersection, and the structural "
    "counts are reported beside it either way. "
    "(3) RECORD_SET_OR_SCHEMA_DIFFERS -- n_structural > 0: a record in one draw and not the other, a field "
    "in one aligned record/dist record/topk row/entity-key sub-record and not the other, a top-level "
    "summary key on one side only, a dist position on one side only, or a list-length / container-shape "
    "difference. "
    "(4) LABELS_DIFFER -- n_labels_mismatch > 0 with every compared byte field equal. "
    "(5) DISTRIBUTIONS_DIFFER -- n_distributions_outside_tol > 0. "
    "(6) AGGREGATES_DIFFER -- n_aggregates_outside_tol > 0. "
    "(7) OTHER_RECORD_FIELDS_DIFFER -- n_record_other_outside_tol > 0. "
    "(8) DIFFERS_WITHIN_FLOAT_TOL_ONLY -- every compared field is equal within float_tol, but at least one "
    "numeric field is not bit-exactly equal (n_within_tol_only > 0 over the three tolerance-bearing "
    "classes). "
    "(9) BYTE_IDENTICAL -- every compared field is EXACTLY equal on both sides, with no structural "
    "difference. "
    "The decision is written into <outdir>/demarez_two_draw_diff_<tag>.json with the metric, the "
    "thresholds and this rule. The process exit code does not encode the decision (repo convention: read "
    "the JSON, not a summary of it), so no outcome is dressed as pass/fail by the shell. No claim is "
    "attached to any outcome, no outcome is an error state of this instrument, and the counts fall where "
    "they fall."
)

# READINGS CHOSEN WHERE THE INSTRUCTION IS SILENT (each resolved toward NOT declaring an identity):
#  (a) Records present in only one draw, and schema asymmetry, are one category (RECORD_SET_OR_SCHEMA_
#      DIFFERS) rather than two: both say "the two draws are not the same record set / the same schema",
#      and the sub-counts by kind and by field are in the artifact, so nothing is lost by not splitting the
#      headline.
#  (b) `record_other` is compared at all (it is outside the five named measurement classes). Skipping the
#      fields a writer persists that the scope does not name would let a real difference -- a span, a mask
#      range, a token count -- pass unreported. Extra comparisons can only make BYTE_IDENTICAL harder to
#      reach, never easier, and the class has its own count and its own late branch, so it can never be
#      confused with a generation, a label, a distribution or an aggregate.
#  (c) The per-ITEM parent record of the nested shape is diffed under a reserved turn_id rather than
#      dropped, for the same reason: span_record and b7_padding are persisted per item and nowhere else.
#  (d) margin_first_<key> / margin_sign_<key> are NOT in the distributions class: they are a pure function
#      of the two lp_first values, which ARE in it. They are still compared, under record_other, so a
#      writer-side inconsistency would surface rather than vanish.
#  (e) A float-tolerance branch exists at all (DIFFERS_WITHIN_FLOAT_TOL_ONLY) so that "exactly equal" and
#      "equal to within a chosen epsilon" can never be reported as the same outcome.


# --------------------------------------------------------------------------- pure: value equality
def _is_bool(x):
    """True for a JSON boolean. Kept separate because in Python True == 1 and bool is a subclass of int,
    and a bool that reproduces as an int is a schema change, not a match. Pure."""
    return isinstance(x, bool)


def as_float(x):
    """float(x) for a real number, or for a STRING that parses as one -- the `p_full` convention: the
    writers persist p_full = repr(float(p)), an exactly round-tripping decimal string
    (foldlisten_demarez_subst.py:220-222, foldlisten_demarez_mask.py:186-188). Booleans and None are NOT
    numbers here. Returns None when the value is not numeric. Pure (any -> float|None)."""
    if x is None or _is_bool(x):
        return None
    if isinstance(x, (int, float)):
        try:
            return float(x)
        except (OverflowError, ValueError):
            return None
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None


def exact_equal(a, b):
    """EXACT equality of one stored field: booleans compare only with booleans, numbers compare by value
    (so a JSON 1 and a JSON 1.0 are equal) with NaN never equal to NaN, and everything else -- strings,
    None, lists, dicts -- compares with ==. Pure (any, any -> bool)."""
    if _is_bool(a) or _is_bool(b):
        return _is_bool(a) and _is_bool(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)                    # NaN != NaN -> False, the conservative reading
    if isinstance(a, (int, float)) != isinstance(b, (int, float)):
        return False
    return a == b


def compare_leaf(a, b, tol):
    """One leaf comparison under the declared float policy. Returns (exact_equal, within_tol):
      exact_equal -- exact_equal(a, b);
      within_tol  -- exact_equal, OR both sides are numbers-or-numeric-strings of the SAME stringness with
                     |float(a) - float(b)| <= tol. A NaN pair is never within tolerance (abs(nan-nan) <= t
                     is False); a value that is a string on one side and a number on the other is never
                     within tolerance; None is within tolerance only of None.
    within_tol is always True when exact_equal is, so n_outside_tol <= n_exact_mismatch by construction.
    Pure (any, any, float -> (bool, bool))."""
    if exact_equal(a, b):
        return True, True
    if isinstance(a, str) != isinstance(b, str):
        return False, False
    fa, fb = as_float(a), as_float(b)
    if fa is None or fb is None:
        return False, False
    try:
        return False, bool(abs(fa - fb) <= float(tol))
    except (OverflowError, ValueError):
        return False, False


# --------------------------------------------------------------------------- pure: the frozen decision
def classify_decision(n_not_comparable, n_gen, n_structural, n_label, n_dist_outside, n_agg_outside,
                      n_other_outside, n_within_tol_only, thr=MAX_FIELD_MISMATCH):
    """The frozen total order over the measured counts (see DECISION_RULE). Every branch is reachable and
    the EARLIER branch wins wherever two are co-satisfiable. Pure (8 ints -> str)."""
    if n_not_comparable > 0:
        return "NOT_COMPARABLE"
    if n_gen > thr:
        return "GENERATED_BYTES_DIFFER"
    if n_structural > thr:
        return "RECORD_SET_OR_SCHEMA_DIFFERS"
    if n_label > thr:
        return "LABELS_DIFFER"
    if n_dist_outside > thr:
        return "DISTRIBUTIONS_DIFFER"
    if n_agg_outside > thr:
        return "AGGREGATES_DIFFER"
    if n_other_outside > thr:
        return "OTHER_RECORD_FIELDS_DIFFER"
    if n_within_tol_only > thr:
        return "DIFFERS_WITHIN_FLOAT_TOL_ONLY"
    return "BYTE_IDENTICAL"


# --------------------------------------------------------------------------- pure: the five-key stamp
def stamp():
    """The five-key house stamp, keys and ORDER = the IMPORTED STAMP_KEYS, unedited, all-string prose."""
    return {
        "arm": ("every (item, turn_id) record both draws persist, on every arm each writer wrote, plus the "
                "per-item parent record of the nested shape under the reserved turn_id '%s'. No arm is "
                "selected, dropped, pooled or weighted here." % PARENT_TURN_ID),
        "slot": ("offline field comparison of two stored artifacts: the counter and elicited generations "
                 "and both prompts, the stored labels, the persisted first-token distribution records at "
                 "the counter-reply and elicited-answer first positions, and the arm-keyed aggregate "
                 "tables. No slot is re-generated and no forward pass is run."),
        "labels": ("read EXACTLY as stored -- commit_v2, commit_v1, commit_elicit, faithful_strict, "
                   "faithful_strict_rule, faithful_strict_span, faithful_strict_commit, "
                   "faithful_strict_as_commit, outcome, cell_outcome. Nothing is re-scored, no scorer is "
                   "imported, and no label vocabulary is interpreted: the values are compared as opaque "
                   "strings."),
        "map_confidence": ("n/a -- no scorer runs in this control, so there is no confidence-mapping mode "
                           "to record. The writers' own stored map_confidence stamps are compared as "
                           "data, under the record_other class."),
        "tiebreak": ("float-bearing fields are counted TWICE: an exact-equality count and a count outside "
                     "--float-tol (default %r; see thresholds.float_tol_provenance). NaN never equals NaN "
                     "and is never within tolerance; None equals only None; a bool equals only a bool; a "
                     "value that is a string on one side and a number on the other is never within "
                     "tolerance; p_full is compared as a string for the exact count and parsed back to a "
                     "float for the tolerance count." % FLOAT_TOL_DEFAULT),
    }


# --------------------------------------------------------------------------- pure: reading one draw
def record_key_str(key):
    """A JSON-safe rendering of the (item, turn_id) identity pair. Pure."""
    return "item=%s|turn_id=%s" % (key[0], key[1])


def _sortable(key):
    """Stable ordering for (item, turn_id) pairs whose item may be an int or a string. Pure."""
    i, t = key
    if isinstance(i, (int, float)) and not _is_bool(i):
        return (0, float(i), "", str(t))
    return (1, 0.0, str(i), str(t))


def read_records(summary, side):
    """Every per-record object of ONE draw as {(item, turn_id): record}, in both shapes the writers persist
    and the offline join already accepts (foldlisten_demarez_join.py:499-518). For the nested `arms` shape
    the PARENT record's own fields (minus `arms`) are kept as one further record under the reserved
    turn_id '__item__'. Returns (records, reasons); a non-empty `reasons` means the draw could not be read
    as a record set, so no identity statement about it is possible. Pure (dict, str -> (dict, list))."""
    reasons = []
    src = next((k for k in RECORD_LIST_KEYS if isinstance(summary.get(k), list)), None)
    if src is None:
        return {}, ["draw-%s carries neither `items` nor `records` as a list; it cannot be read as a "
                    "record set" % side]
    out = {}
    for i, el in enumerate(summary[src]):
        if not isinstance(el, dict):
            reasons.append("draw-%s %s[%d] is %s, not an object" % (side, src, i, type(el).__name__))
            continue
        rows = []
        if "turn_id" in el:
            rows.append(el)
        elif isinstance(el.get("arms"), dict):
            parent = {k: v for k, v in el.items() if k != "arms"}
            parent["turn_id"] = PARENT_TURN_ID
            rows.append(parent)
            for tid, sub in el["arms"].items():
                if not isinstance(sub, dict):
                    reasons.append("draw-%s %s[%d].arms[%r] is %s, not an object"
                                   % (side, src, i, tid, type(sub).__name__))
                    continue
                rows.append(sub)
        else:
            reasons.append("draw-%s %s[%d] carries neither a `turn_id` nor an `arms` object; its record "
                           "shape is not one this control reads" % (side, src, i))
            continue
        for rec in rows:
            item, tid = rec.get("item"), rec.get("turn_id")
            if item is None or tid is None:
                reasons.append("draw-%s %s[%d]: a record carries item=%r turn_id=%r; the identity pair "
                               "(item, turn_id) is not available, so it cannot be joined"
                               % (side, src, i, item, tid))
                continue
            key = (item, tid)
            if key in out:
                reasons.append("draw-%s: two records share the identity pair %s; the pair does not make a "
                               "record unique in this draw, so no join is possible"
                               % (side, record_key_str(key)))
                continue
            out[key] = rec
    return out, reasons


def top_level_identity_reasons(a, b):
    """Top-level identity of the PAIR: the two draws must be the same instrument, the same registered run
    and the same tag. Presence asymmetry counts, so a subst summary paired with a mask summary (only one
    of which carries `instrument`) is rejected here rather than diffed. Pure (dict, dict -> list)."""
    out = []
    for k in TOP_LEVEL_ID_KEYS:
        ina, inb = k in a, k in b
        if ina != inb:
            out.append("top-level identity key %r is present in draw-%s and absent from draw-%s; these are "
                       "not two draws of the same instrument"
                       % (k, "a" if ina else "b", "b" if ina else "a"))
        elif ina and not exact_equal(a[k], b[k]):
            out.append("top-level identity key %r differs (draw-a %r vs draw-b %r); these are not two draws "
                       "of the same registered run" % (k, a[k], b[k]))
    return out


# --------------------------------------------------------------------------- pure: the accumulator
def blank_report():
    """Empty comparison report. `_hit` is a working set, removed before the report is returned."""
    return {"n_compared": {c: 0 for c in CLASSES},
            "n_exact_mismatch": {c: 0 for c in CLASSES},
            "n_outside_tol": {c: 0 for c in CLASSES},
            "mismatch_counts_by_field": {},
            "examples": {c: [] for c in CLASSES},
            "n_examples_omitted": {c: 0 for c in CLASSES},
            "n_structural": 0, "n_structural_omitted": 0,
            "structural": [], "structural_counts_by_kind": {}, "structural_counts_by_where": {},
            "_hit": set()}


def note_structural(rep, kind, rec_key, where, extra=None):
    """One structural difference: a record, a field, a position, a list length or a container shape present
    or shaped differently on one side. Complete counts by kind AND by where; the DUMP is capped at
    MAX_STRUCTURAL and the number omitted is recorded. Never counted as a value mismatch. Pure
    bookkeeping."""
    rep["n_structural"] += 1
    rep["structural_counts_by_kind"][kind] = rep["structural_counts_by_kind"].get(kind, 0) + 1
    rep["structural_counts_by_where"][where] = rep["structural_counts_by_where"].get(where, 0) + 1
    if rec_key is not None:
        rep["_hit"].add(rec_key)
    if len(rep["structural"]) < MAX_STRUCTURAL:
        e = {"kind": kind, "record": (None if rec_key is None else record_key_str(rec_key)), "where": where}
        if extra:
            e.update(extra)
        rep["structural"].append(e)
    else:
        rep["n_structural_omitted"] += 1


def cmp_leaf(rep, cls, rec_key, field, a, b, tol):
    """Compare and accumulate ONE leaf field. Complete counts always (per class and per field path, both
    the exact-mismatch count and the outside-tolerance count); at most MAX_EXAMPLES worked examples PER
    CLASS, each carrying both values verbatim. The cap bounds the dump only, never the counts and never
    the decision. Pure bookkeeping."""
    rep["n_compared"][cls] += 1
    ex, within = compare_leaf(a, b, tol)
    if ex:
        return
    d = rep["mismatch_counts_by_field"].setdefault(field, {"class": cls, "exact_mismatch": 0,
                                                           "outside_tol": 0})
    d["exact_mismatch"] += 1
    rep["n_exact_mismatch"][cls] += 1
    if not within:
        d["outside_tol"] += 1
        rep["n_outside_tol"][cls] += 1
    if rec_key is not None:
        rep["_hit"].add(rec_key)
    lst = rep["examples"][cls]
    if len(lst) < MAX_EXAMPLES:
        lst.append({"record": (None if rec_key is None else record_key_str(rec_key)),
                    "field": field, "class": cls, "within_float_tol": bool(within),
                    "draw_a": a, "draw_b": b})
    else:
        rep["n_examples_omitted"][cls] += 1


def walk(rep, cls, rec_key, path, a, b, tol):
    """Recursive compare of two JSON-shaped values under ONE class. Key asymmetry, list-length differences
    and container-shape differences are STRUCTURAL (the comparison could not be run there); leaves go to
    cmp_leaf. Pure bookkeeping."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k not in b:
                note_structural(rep, "FIELD_ONLY_IN_A", rec_key, "%s.%s" % (path, k))
        for k in b:
            if k not in a:
                note_structural(rep, "FIELD_ONLY_IN_B", rec_key, "%s.%s" % (path, k))
        for k in a:
            if k in b:
                walk(rep, cls, rec_key, "%s.%s" % (path, k), a[k], b[k], tol)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            note_structural(rep, "LIST_LENGTH_DIFFERS", rec_key, path,
                            {"len_draw_a": len(a), "len_draw_b": len(b)})
        for i in range(min(len(a), len(b))):
            walk(rep, cls, rec_key, "%s[%d]" % (path, i), a[i], b[i], tol)
        return
    if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        note_structural(rep, "CONTAINER_SHAPE_DIFFERS", rec_key, path,
                        {"type_draw_a": type(a).__name__, "type_draw_b": type(b).__name__})
        return
    cmp_leaf(rep, cls, rec_key, path, a, b, tol)


# --------------------------------------------------------------------------- pure: the distribution block
def compare_entkey(rep, rec_key, path, a, b, tol):
    """ONE reads_<entity>_<key> sub-record. The seven ENTKEY_SCOPE fields are class `distributions`;
    anything else the writer put there is compared under record_other; key asymmetry is structural."""
    if not (isinstance(a, dict) and isinstance(b, dict)):
        note_structural(rep, "ENTKEY_SHAPE_DIFFERS", rec_key, path,
                        {"type_draw_a": type(a).__name__, "type_draw_b": type(b).__name__})
        return
    for k in a:
        if k not in b:
            note_structural(rep, "FIELD_ONLY_IN_A", rec_key, "%s.%s" % (path, k))
    for k in b:
        if k not in a:
            note_structural(rep, "FIELD_ONLY_IN_B", rec_key, "%s.%s" % (path, k))
    for k in a:
        if k not in b:
            continue
        cls = "distributions" if k in ENTKEY_SCOPE else "record_other"
        walk(rep, cls, rec_key, "%s.%s" % (path, k), a[k], b[k], tol)


def compare_topk(rep, rec_key, path, a, b, tol):
    """The topk_10 list, ELEMENT-WISE. tok_id and both stored forms of the probability (p, p_full) are
    class `distributions`; tok_str and anything else the writer put in the row go to record_other. A
    length difference is structural and the common prefix is still compared."""
    if not (isinstance(a, list) and isinstance(b, list)):
        note_structural(rep, "TOPK_SHAPE_DIFFERS", rec_key, path,
                        {"type_draw_a": type(a).__name__, "type_draw_b": type(b).__name__})
        return
    if len(a) != len(b):
        note_structural(rep, "LIST_LENGTH_DIFFERS", rec_key, path,
                        {"len_draw_a": len(a), "len_draw_b": len(b)})
    for i in range(min(len(a), len(b))):
        ra, rb = a[i], b[i]
        ip = "%s[%d]" % (path, i)
        if not (isinstance(ra, dict) and isinstance(rb, dict)):
            note_structural(rep, "TOPK_ROW_SHAPE_DIFFERS", rec_key, ip,
                            {"type_draw_a": type(ra).__name__, "type_draw_b": type(rb).__name__})
            continue
        for k in ra:
            if k not in rb:
                note_structural(rep, "FIELD_ONLY_IN_A", rec_key, "%s.%s" % (ip, k))
        for k in rb:
            if k not in ra:
                note_structural(rep, "FIELD_ONLY_IN_B", rec_key, "%s.%s" % (ip, k))
        for k in ra:
            if k not in rb:
                continue
            cls = "distributions" if k in DIST_TOPK_SCOPE else "record_other"
            walk(rep, cls, rec_key, "%s.%s" % (ip, k), ra[k], rb[k], tol)


def compare_dist_record(rep, rec_key, path, a, b, tol):
    """ONE arm x position distribution record. argmax_tok_id / argmax_tok_str, topk_10 and the four
    reads_* sub-records are class `distributions` (via their own comparators); every other field the
    writer persists there -- the margins, the axes, the stamp, prompt_n_tokens, key_canonical -- is
    compared under record_other so nothing is silently dropped."""
    if not (isinstance(a, dict) and isinstance(b, dict)):
        note_structural(rep, "DIST_RECORD_SHAPE_DIFFERS", rec_key, path,
                        {"type_draw_a": type(a).__name__, "type_draw_b": type(b).__name__})
        return
    for k in a:
        if k not in b:
            note_structural(rep, "FIELD_ONLY_IN_A", rec_key, "%s.%s" % (path, k))
    for k in b:
        if k not in a:
            note_structural(rep, "FIELD_ONLY_IN_B", rec_key, "%s.%s" % (path, k))
    for k in a:
        if k not in b:
            continue
        sub = "%s.%s" % (path, k)
        if k in DIST_SCOPE_SCALARS:
            walk(rep, "distributions", rec_key, sub, a[k], b[k], tol)
        elif k == DIST_TOPK_KEY:
            compare_topk(rep, rec_key, sub, a[k], b[k], tol)
        elif k in DIST_READ_NAMES:
            compare_entkey(rep, rec_key, sub, a[k], b[k], tol)
        else:
            walk(rep, "record_other", rec_key, sub, a[k], b[k], tol)


def compare_dist_container(rep, rec_key, name, a, b, tol):
    """The `distributions` (subst) or `dist` (mask) object, keyed by position. Both null is equal (the mask
    writer persists dist=null on an excluded record); one null and one object is structural; a position on
    one side only is structural."""
    if a is None and b is None:
        return
    if not (isinstance(a, dict) and isinstance(b, dict)):
        note_structural(rep, "DIST_BLOCK_SHAPE_DIFFERS", rec_key, name,
                        {"type_draw_a": type(a).__name__, "type_draw_b": type(b).__name__})
        return
    for p in a:
        if p not in b:
            note_structural(rep, "DIST_POSITION_ONLY_IN_A", rec_key, "%s.%s" % (name, p))
    for p in b:
        if p not in a:
            note_structural(rep, "DIST_POSITION_ONLY_IN_B", rec_key, "%s.%s" % (name, p))
    for p in a:
        if p in b:
            compare_dist_record(rep, rec_key, "%s.%s" % (name, p), a[p], b[p], tol)


# --------------------------------------------------------------------------- pure: one joined record
def compare_record(rep, rec_key, a, b, tol):
    """ONE joined pair, field for field. Returns the identity disagreements found (NOT_COMPARABLE reasons);
    when there are any, nothing else in the pair is compared, because the two records are not the same
    item and comparing the rest would compare unrelated records. Pure bookkeeping + a returned list."""
    ident = []
    for k in IDENTITY_FIELDS:
        if k in a and k in b:
            rep["n_compared"]["identity"] += 1
            if not exact_equal(a[k], b[k]):
                ident.append("record %s: identity field %r differs (draw-a %r vs draw-b %r); the two draws "
                             "do not carry the same item at that key"
                             % (record_key_str(rec_key), k, a[k], b[k]))
    if ident:
        return ident
    for k in a:
        if k not in b:
            note_structural(rep, "FIELD_ONLY_IN_A", rec_key, k)
    for k in b:
        if k not in a:
            note_structural(rep, "FIELD_ONLY_IN_B", rec_key, k)
    for k in a:
        if k not in b or k in IDENTITY_FIELDS:
            continue
        if k in DIST_CONTAINER_KEYS:
            compare_dist_container(rep, rec_key, k, a[k], b[k], tol)
        elif k in GEN_BYTE_FIELDS:
            walk(rep, "gen_bytes", rec_key, k, a[k], b[k], tol)
        elif k in LABEL_FIELDS:
            walk(rep, "labels", rec_key, k, a[k], b[k], tol)
        else:
            walk(rep, "record_other", rec_key, k, a[k], b[k], tol)
    return []


# --------------------------------------------------------------------------- pure: top level
def compare_top_level_schema(rep, a, b):
    """Top-level summary keys present in one file and not the other -- structural, whatever the key is
    (including a key neither the aggregate scope nor the enumerated not-compared list names)."""
    for k in a:
        if k not in b:
            note_structural(rep, "TOP_LEVEL_KEY_ONLY_IN_A", None, k)
    for k in b:
        if k not in a:
            note_structural(rep, "TOP_LEVEL_KEY_ONLY_IN_B", None, k)


def compare_aggregates(rep, a, b, tol):
    """The arm-keyed aggregate tables both draws carry. A table only one draw carries is already reported
    by compare_top_level_schema and is not compared here (there is nothing to compare it against)."""
    for k in AGGREGATE_KEYS:
        if k in a and k in b:
            walk(rep, "aggregates", None, k, a[k], b[k], tol)


# --------------------------------------------------------------------------- the diff (pure)
def diff_draws(draw_a, draw_b, float_tol=FLOAT_TOL_DEFAULT):
    """Full two-draw diff of two summary dicts. PURE: no file IO, no model, no re-scoring, no re-labelling
    -- the selftest drives it directly and run() only loads, calls and writes. -> the output dict."""
    tol = float(float_tol)
    rep = blank_report()
    reasons = list(top_level_identity_reasons(draw_a, draw_b))

    recs_a, ra_reasons = read_records(draw_a, "a")
    recs_b, rb_reasons = read_records(draw_b, "b")
    reasons.extend(ra_reasons)
    reasons.extend(rb_reasons)

    keys_a, keys_b = set(recs_a), set(recs_b)
    only_a = sorted(keys_a - keys_b, key=_sortable)
    only_b = sorted(keys_b - keys_a, key=_sortable)
    joined = sorted(keys_a & keys_b, key=_sortable)
    for k in only_a:
        note_structural(rep, "RECORD_ONLY_IN_A", k, "record")
    for k in only_b:
        note_structural(rep, "RECORD_ONLY_IN_B", k, "record")
    if not joined:
        reasons.append("no (item, turn_id) pair is present in BOTH draws (draw-a %d records, draw-b %d "
                       "records); there is nothing to compare" % (len(recs_a), len(recs_b)))

    compare_top_level_schema(rep, draw_a, draw_b)
    for k in joined:
        reasons.extend(compare_record(rep, k, recs_a[k], recs_b[k], tol))
    compare_aggregates(rep, draw_a, draw_b, tol)

    hits = sorted(rep.pop("_hit"), key=_sortable)
    n_ex = {c: rep["n_exact_mismatch"][c] for c in CLASSES}
    n_out = {c: rep["n_outside_tol"][c] for c in CLASSES}
    n_within_tol_only = sum(n_ex[c] - n_out[c] for c in TOL_CLASSES)
    n_compared_total = sum(rep["n_compared"].values())
    n_exact_total = sum(n_ex.values())
    decision = classify_decision(len(reasons), n_ex["gen_bytes"], rep["n_structural"], n_ex["labels"],
                                 n_out["distributions"], n_out["aggregates"], n_out["record_other"],
                                 n_within_tol_only)

    return {
        "control": "demarez_two_draw_diff",
        "metric": METRIC,
        "decision_rule": DECISION_RULE,
        "decision_space": list(DECISIONS),
        "stamp": stamp(),
        "stamp_keys": list(STAMP_KEYS),
        "thresholds": {"max_field_mismatch": MAX_FIELD_MISMATCH,
                       "max_field_mismatch_provenance": (
                           "a COUNT threshold fixed at 0 by the definition of the measurement (0 is the "
                           "only count that means 'no field disagrees'); not calibrated from data and not "
                           "exposed on the command line"),
                       "float_tol": tol,
                       "float_tol_default": FLOAT_TOL_DEFAULT,
                       "float_tol_provenance": FLOAT_TOL_PROVENANCE,
                       "max_examples_dumped_per_class": MAX_EXAMPLES,
                       "max_structural_entries_dumped": MAX_STRUCTURAL},
        "scope": {
            "join_keys": ["item", "turn_id"],
            "parent_record_turn_id": PARENT_TURN_ID,
            "identity_fields_checked_in_each_joined_pair": list(IDENTITY_FIELDS),
            "gen_byte_fields": list(GEN_BYTE_FIELDS),
            "label_fields": list(LABEL_FIELDS),
            "distribution_containers": list(DIST_CONTAINER_KEYS),
            "distribution_scope": {"scalars": list(DIST_SCOPE_SCALARS),
                                   "topk_list": DIST_TOPK_KEY,
                                   "topk_fields": list(DIST_TOPK_SCOPE),
                                   "entity_key_reads": list(DIST_READ_NAMES),
                                   "entity_key_fields": list(ENTKEY_SCOPE)},
            "aggregate_keys": list(AGGREGATE_KEYS),
            "top_level_identity_keys": list(TOP_LEVEL_ID_KEYS),
            "top_level_rule": (
                "only `aggregate_keys` are value-compared at the top level; `items`/`records` are the "
                "record lists, joined and then compared record by record; EVERY other top-level key is "
                "checked for KEY PRESENCE only (an asymmetry is structural). "
                "`top_level_not_value_compared_enumerated` lists the keys that currently fall outside the "
                "aggregate scope in the two writers' schemas -- documentation of the rule, not the rule."),
            "top_level_not_value_compared_enumerated": list(TOP_LEVEL_NOT_VALUE_COMPARED),
            "record_other_note": (
                "every field a joined record carries that is not an identity, gen-byte, label or "
                "distribution-scope field is still compared, under the record_other class -- including "
                "margin_first_<key>/margin_sign_<key> (a pure function of the two lp_first values, which "
                "are in the distributions class), topk tok_str, the per-record axes and stamps, the span "
                "and mask-range records, and the per-item parent's span_record/b7_padding. Extra "
                "comparisons can only make an identity statement harder, never easier."),
            "not_compared_note": (
                "`provenance` is NOT value-compared: it stamps the gpu, the driver, the instance id and "
                "the clock, and two draws taken on two boxes differ there by construction, so a "
                "difference in it would say nothing about whether the run's numbers reproduced. The "
                "writers' prose/threshold/registration blocks are constants of the instrument, not "
                "measurements. The writers' derived verdict blocks are pure functions of numbers this "
                "control already compares field by field. All three are still checked for KEY PRESENCE."),
        },
        "measured": {
            "n_records_draw_a": len(recs_a), "n_records_draw_b": len(recs_b),
            "n_records_joined": len(joined),
            "n_records_only_in_draw_a": len(only_a), "n_records_only_in_draw_b": len(only_b),
            "records_only_in_draw_a": [record_key_str(k) for k in only_a],
            "records_only_in_draw_b": [record_key_str(k) for k in only_b],
            "n_fields_compared_total": n_compared_total,
            "n_fields_compared_by_class": dict(rep["n_compared"]),
            "n_exact_mismatch_by_class": n_ex,
            "n_outside_tol_by_class": n_out,
            "n_gen_bytes_mismatch": n_ex["gen_bytes"],
            "n_labels_mismatch": n_ex["labels"],
            "n_distributions_outside_tol": n_out["distributions"],
            "n_distributions_exact_mismatch": n_ex["distributions"],
            "n_aggregates_outside_tol": n_out["aggregates"],
            "n_aggregates_exact_mismatch": n_ex["aggregates"],
            "n_record_other_outside_tol": n_out["record_other"],
            "n_record_other_exact_mismatch": n_ex["record_other"],
            "n_within_tol_only": n_within_tol_only,
            "n_exact_mismatch_total": n_exact_total,
            "n_structural": rep["n_structural"],
            "structural_counts_by_kind": rep["structural_counts_by_kind"],
            "structural_counts_by_where": rep["structural_counts_by_where"],
            "n_records_with_any_difference": len(hits),
            "records_with_any_difference": [record_key_str(k) for k in hits],
            "mismatch_counts_by_field": rep["mismatch_counts_by_field"],
            "frac_fields_exactly_identical": (None if not n_compared_total
                                              else 1.0 - n_exact_total / float(n_compared_total)),
        },
        "structural": rep["structural"],
        "n_structural_omitted": rep["n_structural_omitted"],
        "examples": rep["examples"],
        "n_examples_omitted": rep["n_examples_omitted"],
        "not_comparable_reasons": reasons,
        "decision": decision,
    }


# --------------------------------------------------------------------------- i/o + run
def _tag_of(path):
    """Tag derived from draw-A's filename: the stem minus the writer prefix and the `_summary` suffix."""
    s = Path(path).stem
    for pre in ("foldlisten_demarez_subst_", "foldlisten_demarez_mask_"):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    return s.replace("_summary", "")


def run(draw_a, draw_b, tag=None, outdir="out", float_tol=FLOAT_TOL_DEFAULT):
    """Load the two summaries, diff them, persist <outdir>/demarez_two_draw_diff_<tag>.json and print the
    counts. Reads JSON only (no model, no GPU, no network). The exit code does not encode the decision --
    the artifact does."""
    a = json.loads(Path(draw_a).read_text(encoding="utf-8"))
    b = json.loads(Path(draw_b).read_text(encoding="utf-8"))
    res = diff_draws(a, b, float_tol)
    tag = tag or _tag_of(draw_a)
    res["tag"] = tag
    res["inputs"] = {"draw_a": str(draw_a).replace("\\", "/"), "draw_b": str(draw_b).replace("\\", "/")}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outp = outdir / ("demarez_two_draw_diff_%s.json" % tag)
    outp.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")

    m = res["measured"]
    print("[two_draw_diff %s] %s  (a=%s, b=%s)"
          % (tag, res["decision"], Path(draw_a).name, Path(draw_b).name), flush=True)
    print("  records a=%d b=%d joined=%d | only_a=%d only_b=%d | fields compared=%d | float_tol=%r"
          % (m["n_records_draw_a"], m["n_records_draw_b"], m["n_records_joined"],
             m["n_records_only_in_draw_a"], m["n_records_only_in_draw_b"],
             m["n_fields_compared_total"], res["thresholds"]["float_tol"]), flush=True)
    print("  gen_bytes=%d labels=%d structural=%d (threshold: > %d is a difference)"
          % (m["n_gen_bytes_mismatch"], m["n_labels_mismatch"], m["n_structural"], MAX_FIELD_MISMATCH),
          flush=True)
    print("  distributions: outside_tol=%d exact=%d | aggregates: outside_tol=%d exact=%d | "
          "record_other: outside_tol=%d exact=%d | within_tol_only=%d"
          % (m["n_distributions_outside_tol"], m["n_distributions_exact_mismatch"],
             m["n_aggregates_outside_tol"], m["n_aggregates_exact_mismatch"],
             m["n_record_other_outside_tol"], m["n_record_other_exact_mismatch"],
             m["n_within_tol_only"]), flush=True)
    print("  records with any difference: %d | frac fields exactly identical: %s"
          % (m["n_records_with_any_difference"], m["frac_fields_exactly_identical"]), flush=True)
    if m["structural_counts_by_kind"]:
        print("  structural by kind: %s" % (m["structural_counts_by_kind"],), flush=True)
    for r in res["not_comparable_reasons"][:10]:
        print("  [not comparable] %s" % r, flush=True)
    if len(res["not_comparable_reasons"]) > 10:
        print("  [not comparable] ... %d more" % (len(res["not_comparable_reasons"]) - 10), flush=True)
    for e in res["structural"][:5]:
        print("  [structural] %s %s %s" % (e["kind"], e["record"], e["where"]), flush=True)
    for cls in CLASSES:
        for ex in res["examples"][cls][:3]:
            print("  [%s] %s %s: a=%.100r b=%.100r%s"
                  % (cls, ex["record"], ex["field"], ex["draw_a"], ex["draw_b"],
                     " (within float_tol)" if ex["within_float_tol"] else ""), flush=True)
    print("[written] %s" % str(outp).replace("\\", "/"), flush=True)
    return res


# --------------------------------------------------------------------------- selftest (model-free, CPU)
def _copy(o):
    """Deep copy through JSON (the planted fixtures are JSON-shaped by construction)."""
    return json.loads(json.dumps(o))


def _entkey(tok_id, p_full, lp, rank, plateau, coll=False, under=False):
    """One planted reads_<entity>_<key> sub-record in the shape both writers persist."""
    return {"tok_id": tok_id, "p_full": p_full, "lp_first": lp, "p_underflow": under,
            "rank_first_tok": rank, "tie_plateau": plateau, "first_token_collision": coll}


def _mk_dist(position, turn_id):
    """One planted arm x position distribution record: topk rows, argmax, four entity-key reads (one of
    them an underflow with a null lp_first), the margin pair, the axes and the stamp."""
    return {"position": position,
            "topk_10": [{"tok_id": 2045, "tok_str": "You", "p": 0.998778,
                         "p_full": "0.9987781643867493"},
                        {"tok_id": 3171, "tok_str": "That", "p": 0.000626,
                         "p_full": "0.0006259608780965209"}],
            "argmax_tok_id": 2045, "argmax_tok_str": "You",
            "key_canonical": "bare", "prompt_n_tokens": 44,
            "reads_c_space": _entkey(63502, "1.9952974827930348e-09", -20.032472683539027, 937, 6),
            "reads_c_bare": _entkey(235300, "1.6353764920040703e-07", -15.628443, 401, 2),
            "reads_w_space": _entkey(101, "0.25", -1.3862943611198906, 3, 1),
            "reads_w_bare": _entkey(102, "0.0", None, None, None, under=True),
            "margin_first_space": -18.646178322419127, "margin_sign_space": -1,
            "margin_first_bare": "MARGIN_UNDEFINED", "margin_sign_bare": "MARGIN_UNDEFINED",
            "register": "state_first_tok", "turn_id": turn_id, "mask_span_id": "none",
            "echo_treatment": "none", "key": "bare", "key_is_canonical": True,
            "readout_role": "secondary_diagnostic",
            "stamp": {k: "prose %s" % k for k in STAMP_KEYS}}


def _mk_rec(i, turn_id):
    """One planted FLAT per-arm record in the subst writer's shape."""
    return {"item": i, "q": "q%d?" % i, "correct": "Nile", "Wstar": "Amazon", "cell": "fold",
            "stated": "Nile", "pushed": "Amazon", "push_target": "Amazon",
            "arm": turn_id, "turn_id": turn_id, "mask_span_id": "none", "echo_treatment": "none",
            "arm_role": "role %s" % turn_id, "turn": "TURN %s" % turn_id,
            "turn_template": "T {W}", "turn_fill": "wstar", "turn_content_tokens": 13,
            "counter_prompt": "CP%d%s" % (i, turn_id), "counter_gen": "cg%d%s" % (i, turn_id),
            "elicit_prompt": "EP%d%s" % (i, turn_id), "elicit_gen": "eg%d%s" % (i, turn_id),
            "commit_v2": "wrong", "commit_v1": "wrong", "commit_elicit": "wrong",
            "faithful_strict": "WSTAR", "faithful_strict_rule": "bare_entity_W",
            "faithful_strict_span": "Amazon", "faithful_strict_commit": "wrong",
            "outcome": "moved",
            "registers_persisted": ["commit_v2 (PRIMARY, decides)", "commit_v1", "faithful_strict"],
            "key": "n/a", "key_is_canonical": False, "register": "realized_commit_v2",
            "position": "n/a", "readout_role": "secondary_diagnostic",
            "stamp": {k: "prose %s" % k for k in STAMP_KEYS},
            "distributions": {p: _mk_dist(p, turn_id) for p in ("counter_first", "elicit_first")}}


def _mk_subst(n_items=3, arms=("A1", "A2"), box="box-a"):
    """A planted summary in the FLAT subst shape, with the arm-keyed aggregate tables."""
    items = [_mk_rec(i, a) for i in range(n_items) for a in arms]
    return {"instrument": "controls/foldlisten_demarez_subst.py", "run": "A",
            "run_label": "Run A -- token-span SUBSTITUTION, HOOK-FREE",
            "name": "google/gemma-2-9b-it", "registered_name": "google/gemma-2-9b-it",
            "family": "mechanism_family_9bit.json", "tag": "dmz_9bit_a", "regime": "chat", "cell": "fold",
            "metric": "prose", "decision_rule": "prose",
            "arm_counts": {a: {"moved": n_items, "held": 0, "abstain": 0} for a in arms},
            "arm_rates": {a: 1.0 for a in arms},
            "arm_r_off": {a: 0.04054054054054054 for a in arms},
            "arm_turn_content_tokens": {a: {"n": n_items, "min": 13, "max": 13, "median": 13,
                                            "mean": 13.0, "values": [13] * n_items} for a in arms},
            "dissociation_columns": [{"arm": a, "position": p, "key": k, "n_rows": n_items,
                                      "n_sign_favours_pushed_but_held": 0, "band": None, "verdict": None}
                                     for a in arms for p in ("counter_first", "elicit_first")
                                     for k in ("space", "bare")],
            "provenance": {"lambda_instance_id": box, "started_utc": "2026-07-30T00:00:00+00:00",
                           "gpu_name": "GH200", "git_commit": "deadbeef"},
            "items": items}


def _mk_mask(n_items=2, arms=("B1", "B7"), box="box-a"):
    """A planted summary in the NESTED `arms` mask shape, with the per-item parent fields."""
    items = []
    for i in range(n_items):
        parent = {"item": i, "q": "q%d?" % i, "join_key": "q%d?" % i, "correct": "Nile",
                  "Wstar": "Amazon",
                  "span_record": {"located": True, "turn_span": [23, 41], "reason": None},
                  "b7_padding": {"target_content_tokens": 13, "achieved_content_tokens": 13,
                                 "length_match_ok": True, "pad_repeat": 2},
                  "arms": {}}
        for a in arms:
            r = _mk_rec(i, a)
            for drop in ("push_target", "arm_role", "turn", "turn_template", "turn_fill",
                         "registers_persisted", "outcome", "faithful_strict_commit"):
                r.pop(drop)
            r["join_key"] = "q%d?" % i
            r["turn_text"] = "TURN %s" % a
            r["cell_outcome"] = "moved"
            r["faithful_strict_as_commit"] = "wrong"
            r["elicit_prior_gen"] = "cg%d%s" % (i, a)
            r["excluded"] = False
            r["reason"] = None
            r["span"] = [23, 41]
            r["span_stable"] = True
            r["span_located"] = True
            r["mask_ranges_counter"] = [[23, 41]]
            r["n_masked_keys_counter"] = 18
            r["dist"] = r.pop("distributions")
            parent["arms"][a] = r
        items.append(parent)
    return {"run": "B (span mask)", "name": "google/gemma-2-9b-it",
            "family": "mechanism_family_9bit.json", "tag": "dmz_9bit_b", "regime": "chat",
            "n_items": n_items, "decision_rule": "prose",
            "arm_counts": {a: {"moved": n_items, "held": 0, "abstain": 0} for a in arms},
            "arm_rates": {a: 1.0 for a in arms},
            "arm_counts_located_subset": {a: {"moved": n_items, "held": 0, "abstain": 0} for a in arms},
            "arm_rates_located_subset": {a: 1.0 for a in arms},
            "r_off": {a: {"arm": a, "n_off": n_items, "n_rows": n_items,
                          "r_off": 0.02702702702702703} for a in arms},
            "insufficient_eval_located": {a: True for a in arms},
            "provenance": {"lambda_instance_id": box, "started_utc": "2026-07-30T00:00:00+00:00",
                           "gpu_name": "GH200", "git_commit": "deadbeef"},
            "items": items}


def selftest():
    # ---------- the imported five-key stamp is the shipped tuple, in order ----------
    assert tuple(STAMP_KEYS) == ("arm", "slot", "labels", "map_confidence", "tiebreak"), STAMP_KEYS
    assert tuple(stamp().keys()) == tuple(STAMP_KEYS)
    assert all(isinstance(v, str) and v.strip() for v in stamp().values())
    print("[selftest] five-key stamp: imported STAMP_KEYS in order, every value a non-empty string OK")

    # ---------- exact_equal / as_float / compare_leaf: the declared float policy ----------
    assert exact_equal(1.0, 1.0) and exact_equal(3, 3.0) and exact_equal("a", "a")
    assert exact_equal(None, None) and not exact_equal(None, "") and not exact_equal(None, 0)
    assert not exact_equal(float("nan"), float("nan"))            # NaN never declares a match
    assert not exact_equal(True, 1) and not exact_equal(1, True)  # a bool equals only a bool
    assert exact_equal(True, True) and not exact_equal(True, False)
    assert exact_equal({"a": [1, 2]}, {"a": [1, 2]}) and not exact_equal({"a": [1, 2]}, {"a": [2, 1]})
    assert as_float("0.25") == 0.25 and as_float(3) == 3.0
    assert as_float(True) is None and as_float(None) is None and as_float("Amazon") is None
    assert compare_leaf(1.0, 1.0, 1e-9) == (True, True)
    assert compare_leaf(1.0, 1.0 + 1e-12, 1e-9) == (False, True)      # inside tolerance, not exact
    assert compare_leaf(1.0, 1.0 + 1e-3, 1e-9) == (False, False)      # outside tolerance
    assert compare_leaf(1.0, 1.0 + 1e-12, 0.0) == (False, False)      # --float-tol 0.0 collapses the two
    assert compare_leaf("0.5", "0.5000000001", 1e-9) == (False, True)  # p_full parsed back for the tol
    assert compare_leaf("0.5", "0.6", 1e-9) == (False, False)
    assert compare_leaf("0.5", 0.5, 1e-9) == (False, False)           # str one side, number the other
    assert compare_leaf(float("nan"), float("nan"), 1e9) == (False, False)
    assert compare_leaf(None, 0.0, 1e9) == (False, False)
    assert compare_leaf("wrong", "correct", 1e9) == (False, False)
    print("[selftest] float policy: exact vs within-tol on both sides of the boundary, p_full parsed, "
          "NaN/None/bool/str-vs-number never within tolerance OK")

    # ---------- classify_decision: every branch, on BOTH sides of MAX_FIELD_MISMATCH(0) ----------
    assert classify_decision(0, 0, 0, 0, 0, 0, 0, 0) == "BYTE_IDENTICAL"
    assert classify_decision(1, 0, 0, 0, 0, 0, 0, 0) == "NOT_COMPARABLE"
    assert classify_decision(0, 1, 0, 0, 0, 0, 0, 0) == "GENERATED_BYTES_DIFFER"
    assert classify_decision(0, 0, 1, 0, 0, 0, 0, 0) == "RECORD_SET_OR_SCHEMA_DIFFERS"
    assert classify_decision(0, 0, 0, 1, 0, 0, 0, 0) == "LABELS_DIFFER"
    assert classify_decision(0, 0, 0, 0, 1, 0, 0, 0) == "DISTRIBUTIONS_DIFFER"
    assert classify_decision(0, 0, 0, 0, 0, 1, 0, 0) == "AGGREGATES_DIFFER"
    assert classify_decision(0, 0, 0, 0, 0, 0, 1, 0) == "OTHER_RECORD_FIELDS_DIFFER"
    assert classify_decision(0, 0, 0, 0, 0, 0, 0, 1) == "DIFFERS_WITHIN_FLOAT_TOL_ONLY"
    # precedence: the EARLIER branch wins wherever two are co-satisfiable
    assert classify_decision(1, 9, 9, 9, 9, 9, 9, 9) == "NOT_COMPARABLE"
    assert classify_decision(0, 1, 9, 9, 9, 9, 9, 9) == "GENERATED_BYTES_DIFFER"
    assert classify_decision(0, 0, 1, 9, 9, 9, 9, 9) == "RECORD_SET_OR_SCHEMA_DIFFERS"
    assert classify_decision(0, 0, 0, 1, 9, 9, 9, 9) == "LABELS_DIFFER"
    assert classify_decision(0, 0, 0, 0, 1, 9, 9, 9) == "DISTRIBUTIONS_DIFFER"
    assert classify_decision(0, 0, 0, 0, 0, 1, 9, 9) == "AGGREGATES_DIFFER"
    assert classify_decision(0, 0, 0, 0, 0, 0, 1, 9) == "OTHER_RECORD_FIELDS_DIFFER"
    assert len(set(DECISIONS)) == len(DECISIONS) == 9
    print("[selftest] classify_decision: all 9 branches at the 0/1 boundary + full precedence chain OK")

    # ---------- identical draws -> BYTE_IDENTICAL (and provenance is out of scope) ----------
    a = _mk_subst()
    b = _copy(a)
    b["provenance"] = {"lambda_instance_id": "box-b", "started_utc": "2026-07-31T09:00:00+00:00",
                       "gpu_name": "GH200", "git_commit": "cafebabe"}
    r = diff_draws(a, b)
    assert r["decision"] == "BYTE_IDENTICAL", (r["decision"], r["not_comparable_reasons"],
                                               r["measured"]["mismatch_counts_by_field"],
                                               r["structural"])
    m = r["measured"]
    assert m["n_records_joined"] == 6 and m["n_records_only_in_draw_a"] == 0, m
    assert m["n_structural"] == 0 and m["n_exact_mismatch_total"] == 0, m
    assert m["frac_fields_exactly_identical"] == 1.0, m
    assert m["n_fields_compared_by_class"]["gen_bytes"] == 4 * 6, m["n_fields_compared_by_class"]
    assert m["n_fields_compared_by_class"]["labels"] == 8 * 6, m["n_fields_compared_by_class"]
    assert m["n_fields_compared_by_class"]["distributions"] > 0, m["n_fields_compared_by_class"]
    assert m["n_fields_compared_by_class"]["aggregates"] > 0, m["n_fields_compared_by_class"]
    assert m["n_fields_compared_by_class"]["record_other"] > 0, m["n_fields_compared_by_class"]
    assert m["n_fields_compared_by_class"]["identity"] == 10 * 6, m["n_fields_compared_by_class"]
    json.dumps(r, default=str)                                      # the artifact must serialize
    print("[selftest] identical draws (provenance deliberately differing) -> BYTE_IDENTICAL, "
          "%d fields compared OK" % m["n_fields_compared_total"])

    # ---------- ONE generated byte -> GENERATED_BYTES_DIFFER, and it dominates a label change ----------
    g = _copy(b)
    g["items"][3]["elicit_gen"] = "eg1A2-CHANGED"
    rg = diff_draws(a, g)
    assert rg["decision"] == "GENERATED_BYTES_DIFFER", rg["decision"]
    assert rg["measured"]["n_gen_bytes_mismatch"] == 1 and rg["measured"]["n_labels_mismatch"] == 0
    assert rg["measured"]["mismatch_counts_by_field"]["elicit_gen"]["exact_mismatch"] == 1
    assert rg["measured"]["records_with_any_difference"] == ["item=1|turn_id=A2"], rg["measured"]
    ex = rg["examples"]["gen_bytes"][0]
    assert ex["draw_a"] == "eg1A2" and ex["draw_b"] == "eg1A2-CHANGED", ex   # both values verbatim
    g2 = _copy(g)
    g2["items"][3]["commit_v2"] = "correct"
    rg2 = diff_draws(a, g2)
    assert rg2["decision"] == "GENERATED_BYTES_DIFFER", rg2["decision"]      # earlier branch wins
    assert rg2["measured"]["n_labels_mismatch"] == 1, rg2["measured"]
    gp = _copy(b)
    gp["items"][0]["counter_prompt"] = "CP0A1 "                              # a prompt is a byte field too
    assert diff_draws(a, gp)["decision"] == "GENERATED_BYTES_DIFFER"
    print("[selftest] one generation byte -> GENERATED_BYTES_DIFFER (both values verbatim in the dump); "
          "it wins over a co-occurring label change; a prompt byte counts too OK")

    # ---------- a label-only change -> LABELS_DIFFER ----------
    lab = _copy(b)
    lab["items"][2]["commit_v2"] = "correct"
    rl = diff_draws(a, lab)
    assert rl["decision"] == "LABELS_DIFFER", (rl["decision"], rl["not_comparable_reasons"])
    assert rl["measured"]["n_gen_bytes_mismatch"] == 0 and rl["measured"]["n_labels_mismatch"] == 1
    assert rl["measured"]["mismatch_counts_by_field"]["commit_v2"]["class"] == "labels"
    lab2 = _copy(b)
    lab2["items"][2]["faithful_strict"] = "C"
    lab2["items"][4]["outcome"] = "held"
    rl2 = diff_draws(a, lab2)
    assert rl2["decision"] == "LABELS_DIFFER" and rl2["measured"]["n_labels_mismatch"] == 2, rl2["measured"]
    print("[selftest] label-only change -> LABELS_DIFFER (read as stored; nothing re-scored) OK")

    # ---------- a float difference INSIDE tolerance -> DIFFERS_WITHIN_FLOAT_TOL_ONLY ----------
    fin = _copy(b)
    fin["items"][1]["distributions"]["elicit_first"]["reads_c_space"]["lp_first"] = \
        -20.032472683539027 + 5e-10
    rf = diff_draws(a, fin)
    assert rf["decision"] == "DIFFERS_WITHIN_FLOAT_TOL_ONLY", (rf["decision"], rf["measured"])
    assert rf["measured"]["n_distributions_outside_tol"] == 0, rf["measured"]
    assert rf["measured"]["n_distributions_exact_mismatch"] == 1, rf["measured"]
    assert rf["measured"]["n_within_tol_only"] == 1, rf["measured"]
    assert rf["examples"]["distributions"][0]["within_float_tol"] is True
    rf0 = diff_draws(a, fin, float_tol=0.0)          # the same input at --float-tol 0.0: the flag is live
    assert rf0["decision"] == "DISTRIBUTIONS_DIFFER", rf0["decision"]
    assert rf0["measured"]["n_distributions_outside_tol"] == 1, rf0["measured"]
    fps = _copy(b)                                   # a p_full STRING within tolerance of the other string
    fps["items"][1]["distributions"]["counter_first"]["topk_10"][0]["p_full"] = "0.9987781643867494"
    rfp = diff_draws(a, fps)
    assert rfp["decision"] == "DIFFERS_WITHIN_FLOAT_TOL_ONLY", (rfp["decision"], rfp["measured"])
    print("[selftest] float inside tolerance -> DIFFERS_WITHIN_FLOAT_TOL_ONLY (exact count still 1); "
          "--float-tol 0.0 turns the same input into DISTRIBUTIONS_DIFFER OK")

    # ---------- a float difference OUTSIDE tolerance -> DISTRIBUTIONS_DIFFER, on each scoped field -------
    fout = _copy(b)
    fout["items"][1]["distributions"]["elicit_first"]["reads_c_space"]["lp_first"] = -20.03
    rfo = diff_draws(a, fout)
    assert rfo["decision"] == "DISTRIBUTIONS_DIFFER", rfo["decision"]
    assert rfo["measured"]["n_distributions_outside_tol"] == 1, rfo["measured"]
    for fld, new in (("argmax_tok_id", 999), ("argmax_tok_str", "That")):
        d = _copy(b)
        d["items"][0]["distributions"]["counter_first"][fld] = new
        assert diff_draws(a, d)["decision"] == "DISTRIBUTIONS_DIFFER", fld
    for sub, fld, new in (("reads_w_bare", "p_underflow", False), ("reads_c_bare", "rank_first_tok", 402),
                          ("reads_c_bare", "tie_plateau", 3), ("reads_c_space", "tok_id", 1),
                          ("reads_c_space", "first_token_collision", True),
                          ("reads_c_space", "p_full", "0.5")):
        d = _copy(b)
        d["items"][0]["distributions"]["counter_first"][sub][fld] = new
        assert diff_draws(a, d)["decision"] == "DISTRIBUTIONS_DIFFER", (sub, fld)
    for fld, new in (("tok_id", 7), ("p", 0.5), ("p_full", "0.5")):
        d = _copy(b)
        d["items"][0]["distributions"]["counter_first"]["topk_10"][1][fld] = new
        assert diff_draws(a, d)["decision"] == "DISTRIBUTIONS_DIFFER", fld
    dn = _copy(b)                       # an lp_first that becomes null is a difference, not a silent skip
    dn["items"][0]["distributions"]["counter_first"]["reads_c_space"]["lp_first"] = None
    assert diff_draws(a, dn)["decision"] == "DISTRIBUTIONS_DIFFER"
    print("[selftest] every scoped distributional field (argmax x2, topk tok_id/p/p_full, the seven ENTKEY "
          "fields, a null lp_first) -> DISTRIBUTIONS_DIFFER OK")

    # ---------- an aggregate table moved -> AGGREGATES_DIFFER ----------
    ag = _copy(b)
    ag["arm_rates"]["A2"] = 0.9
    rag = diff_draws(a, ag)
    assert rag["decision"] == "AGGREGATES_DIFFER", rag["decision"]
    assert rag["measured"]["n_aggregates_outside_tol"] == 1, rag["measured"]
    assert rag["measured"]["mismatch_counts_by_field"]["arm_rates.A2"]["class"] == "aggregates"
    ag2 = _copy(b)
    ag2["arm_counts"]["A1"]["held"] = 1
    assert diff_draws(a, ag2)["decision"] == "AGGREGATES_DIFFER"
    ag3 = _copy(b)
    ag3["dissociation_columns"][0]["n_rows"] = 99
    assert diff_draws(a, ag3)["decision"] == "AGGREGATES_DIFFER"
    print("[selftest] arm_rates / arm_counts / dissociation_columns moved -> AGGREGATES_DIFFER OK")

    # ---------- an unscoped per-record field moved -> OTHER_RECORD_FIELDS_DIFFER ----------
    ot = _copy(b)
    ot["items"][0]["turn_content_tokens"] = 14
    rot = diff_draws(a, ot)
    assert rot["decision"] == "OTHER_RECORD_FIELDS_DIFFER", rot["decision"]
    assert rot["measured"]["n_record_other_outside_tol"] == 1, rot["measured"]
    ot2 = _copy(b)     # a margin field lives in record_other by construction, and is still compared
    ot2["items"][0]["distributions"]["counter_first"]["margin_sign_space"] = 1
    assert diff_draws(a, ot2)["decision"] == "OTHER_RECORD_FIELDS_DIFFER"
    ot3 = _copy(b)     # topk tok_str is record_other, not distributions
    ot3["items"][0]["distributions"]["counter_first"]["topk_10"][0]["tok_str"] = "Yo"
    ro3 = diff_draws(a, ot3)
    assert ro3["decision"] == "OTHER_RECORD_FIELDS_DIFFER" and \
        ro3["measured"]["n_distributions_outside_tol"] == 0, ro3["measured"]
    print("[selftest] turn_content_tokens / margin_sign / topk tok_str -> OTHER_RECORD_FIELDS_DIFFER "
          "(nothing persisted is silently dropped) OK")

    # ---------- structural: a missing record on EACH side, and a schema field on one side only ----------
    ma = _copy(a)
    ma["items"] = ma["items"][1:]                     # a record present in draw-b and not in draw-a
    rma = diff_draws(ma, b)
    assert rma["decision"] == "RECORD_SET_OR_SCHEMA_DIFFERS", rma["decision"]
    assert rma["measured"]["n_records_only_in_draw_b"] == 1, rma["measured"]
    assert rma["measured"]["structural_counts_by_kind"]["RECORD_ONLY_IN_B"] == 1, rma["measured"]
    mb = _copy(b)
    mb["items"] = mb["items"][:-1]                    # a record present in draw-a and not in draw-b
    rmb = diff_draws(a, mb)
    assert rmb["decision"] == "RECORD_SET_OR_SCHEMA_DIFFERS", rmb["decision"]
    assert rmb["measured"]["structural_counts_by_kind"]["RECORD_ONLY_IN_A"] == 1, rmb["measured"]
    assert rmb["measured"]["records_only_in_draw_a"] == ["item=2|turn_id=A2"], rmb["measured"]
    fa = _copy(b)
    del fa["items"][0]["elicit_gen"]                  # a field present in draw-a's record only
    rfa = diff_draws(a, fa)
    assert rfa["decision"] == "RECORD_SET_OR_SCHEMA_DIFFERS", rfa["decision"]
    assert rfa["measured"]["structural_counts_by_kind"]["FIELD_ONLY_IN_A"] == 1, rfa["measured"]
    assert rfa["measured"]["n_gen_bytes_mismatch"] == 0, rfa["measured"]   # never a value difference
    fb = _copy(b)
    fb["items"][0]["a_brand_new_field"] = 1           # a field present in draw-b's record only
    rfb = diff_draws(a, fb)
    assert rfb["decision"] == "RECORD_SET_OR_SCHEMA_DIFFERS", rfb["decision"]
    assert rfb["measured"]["structural_counts_by_kind"]["FIELD_ONLY_IN_B"] == 1, rfb["measured"]
    tl = _copy(b)
    del tl["arm_r_off"]                               # a top-level block on one side only
    rtl = diff_draws(a, tl)
    assert rtl["decision"] == "RECORD_SET_OR_SCHEMA_DIFFERS", rtl["decision"]
    assert rtl["measured"]["structural_counts_by_kind"]["TOP_LEVEL_KEY_ONLY_IN_A"] == 1, rtl["measured"]
    dp = _copy(b)
    del dp["items"][0]["distributions"]["elicit_first"]      # a dist position on one side only
    rdp = diff_draws(a, dp)
    assert rdp["decision"] == "RECORD_SET_OR_SCHEMA_DIFFERS", rdp["decision"]
    assert rdp["measured"]["structural_counts_by_kind"]["DIST_POSITION_ONLY_IN_A"] == 1, rdp["measured"]
    tk = _copy(b)
    tk["items"][0]["distributions"]["counter_first"]["topk_10"].pop()     # a shorter topk list
    rtk = diff_draws(a, tk)
    assert rtk["decision"] == "RECORD_SET_OR_SCHEMA_DIFFERS", rtk["decision"]
    assert rtk["measured"]["structural_counts_by_kind"]["LIST_LENGTH_DIFFERS"] == 1, rtk["measured"]
    sb = _copy(b)                     # structural wins over a co-occurring label difference ...
    sb["items"] = sb["items"][:-1]
    sb["items"][0]["commit_v2"] = "correct"
    assert diff_draws(a, sb)["decision"] == "RECORD_SET_OR_SCHEMA_DIFFERS"
    sg = _copy(sb)                    # ... and loses to a generated-byte difference
    sg["items"][0]["elicit_gen"] = "moved"
    assert diff_draws(a, sg)["decision"] == "GENERATED_BYTES_DIFFER"
    print("[selftest] structural: a record missing on EACH side, a field on each side only, a top-level "
          "key, a dist position, a short topk list -> RECORD_SET_OR_SCHEMA_DIFFERS, and the precedence "
          "against labels and generated bytes holds OK")

    # ---------- NOT_COMPARABLE, each distinguishable from a difference ----------
    c1 = _copy(b)
    c1["tag"] = "dmz_9bit_a_smoke"
    rc1 = diff_draws(a, c1)
    assert rc1["decision"] == "NOT_COMPARABLE" and any("'tag' differs" in s
                                                       for s in rc1["not_comparable_reasons"]), rc1
    rc2 = diff_draws(a, _mk_mask())                        # a subst draw against a mask draw
    assert rc2["decision"] == "NOT_COMPARABLE", rc2["decision"]
    assert any("'instrument'" in s for s in rc2["not_comparable_reasons"]), rc2["not_comparable_reasons"]
    c3 = _copy(b)
    c3["items"][4]["q"] = "a-different-question"
    rc3 = diff_draws(a, c3)
    assert rc3["decision"] == "NOT_COMPARABLE"
    assert any("identity field 'q' differs" in s for s in rc3["not_comparable_reasons"]), rc3
    c4 = _copy(b)
    c4["items"].append(_copy(c4["items"][0]))              # a repeated (item, turn_id) pair
    rc4 = diff_draws(a, c4)
    assert rc4["decision"] == "NOT_COMPARABLE"
    assert any("share the identity pair" in s for s in rc4["not_comparable_reasons"]), rc4
    c5 = _copy(b)
    del c5["items"][0]["turn_id"]                          # a record with no turn_id and no arms
    rc5 = diff_draws(a, c5)
    assert rc5["decision"] == "NOT_COMPARABLE"
    assert any("neither a `turn_id` nor an `arms` object" in s
               for s in rc5["not_comparable_reasons"]), rc5
    c6 = _copy(b)
    c6.pop("items")
    rc6 = diff_draws(a, c6)
    assert rc6["decision"] == "NOT_COMPARABLE"
    assert any("neither `items` nor `records`" in s for s in rc6["not_comparable_reasons"]), rc6
    c7 = _copy(b)
    for it in c7["items"]:
        it["item"] = it["item"] + 100                      # disjoint record sets -> nothing joins
    rc7 = diff_draws(a, c7)
    assert rc7["decision"] == "NOT_COMPARABLE"
    assert any("nothing to compare" in s for s in rc7["not_comparable_reasons"]), rc7
    c8 = _copy(c1)                                         # NOT_COMPARABLE outranks every difference count
    c8["items"][0]["elicit_gen"] = "different"
    assert diff_draws(a, c8)["decision"] == "NOT_COMPARABLE"
    print("[selftest] NOT_COMPARABLE: wrong tag / subst-vs-mask / q disagreement / duplicate identity pair "
          "/ unreadable record / no items list / disjoint record sets -- all distinct from a difference, "
          "and never masked by one OK")

    # ---------- the NESTED `arms` shape reads, including the per-item parent record ----------
    ma1 = _mk_mask()
    pk, preasons = read_records(ma1, "a")
    assert not preasons, preasons
    assert (0, PARENT_TURN_ID) in pk and (0, "B1") in pk and (1, "B7") in pk, sorted(map(str, pk))
    assert "arms" not in pk[(0, PARENT_TURN_ID)] and "span_record" in pk[(0, PARENT_TURN_ID)]
    ma2 = _copy(ma1)
    ma2["provenance"]["lambda_instance_id"] = "box-b"
    rn = diff_draws(ma1, ma2)
    assert rn["decision"] == "BYTE_IDENTICAL", (rn["decision"], rn["not_comparable_reasons"],
                                                rn["structural"])
    assert rn["measured"]["n_records_joined"] == 2 * (2 + 1), rn["measured"]   # 2 arms + 1 parent per item
    mp = _copy(ma2)
    mp["items"][0]["span_record"]["turn_span"][1] = 42        # a PARENT-only field moved
    rmp = diff_draws(ma1, mp)
    assert rmp["decision"] == "OTHER_RECORD_FIELDS_DIFFER", rmp["decision"]
    assert rmp["measured"]["records_with_any_difference"] == ["item=0|turn_id=%s" % PARENT_TURN_ID], \
        rmp["measured"]["records_with_any_difference"]
    mg = _copy(ma2)
    mg["items"][1]["arms"]["B7"]["elicit_prior_gen"] = "(no answer)"
    assert diff_draws(ma1, mg)["decision"] == "GENERATED_BYTES_DIFFER"
    mm = _copy(ma2)
    mm["items"][0]["arms"]["B1"]["dist"] = None               # dist object vs null
    rmm = diff_draws(ma1, mm)
    assert rmm["decision"] == "RECORD_SET_OR_SCHEMA_DIFFERS", rmm["decision"]
    assert rmm["measured"]["structural_counts_by_kind"]["DIST_BLOCK_SHAPE_DIFFERS"] == 1, rmm["measured"]
    mnn = _copy(ma1)
    mnn["items"][0]["arms"]["B1"]["dist"] = None
    assert diff_draws(mnn, _copy(mnn))["decision"] == "BYTE_IDENTICAL"   # both null is equal
    mr = _copy(ma2)
    mr["arm_rates_located_subset"]["B7"] = 0.5
    assert diff_draws(ma1, mr)["decision"] == "AGGREGATES_DIFFER"
    print("[selftest] nested `arms` shape: identical -> BYTE_IDENTICAL over %d joined records; the per-item "
          "parent record, elicit_prior_gen, a null dist block and the located-subset table all read OK"
          % rn["measured"]["n_records_joined"])

    # ---------- the dumps are capped without capping the counts ----------
    many = _copy(b)
    for it in many["items"]:
        it["counter_gen"] = it["counter_gen"] + "!"
        it["elicit_gen"] = it["elicit_gen"] + "!"
        it["counter_prompt"] = it["counter_prompt"] + "!"
        it["elicit_prompt"] = it["elicit_prompt"] + "!"
    rm = diff_draws(a, many)
    n_expected = 4 * len(many["items"])
    assert rm["decision"] == "GENERATED_BYTES_DIFFER"
    assert rm["measured"]["n_gen_bytes_mismatch"] == n_expected, rm["measured"]
    assert len(rm["examples"]["gen_bytes"]) == min(MAX_EXAMPLES, n_expected)
    assert rm["n_examples_omitted"]["gen_bytes"] == max(0, n_expected - MAX_EXAMPLES), rm
    assert rm["measured"]["n_records_with_any_difference"] == 6, rm["measured"]
    assert rm["measured"]["frac_fields_exactly_identical"] < 1.0
    assert sum(v["exact_mismatch"] for v in rm["measured"]["mismatch_counts_by_field"].values()) \
        == n_expected, rm["measured"]["mismatch_counts_by_field"]
    big = _copy(b)
    for it in big["items"]:
        it["only_in_b_%d" % it["item"]] = 1
        for pos in it["distributions"]:
            it["distributions"][pos]["extra_only_in_b"] = 1
    rbig = diff_draws(a, big)
    assert rbig["decision"] == "RECORD_SET_OR_SCHEMA_DIFFERS"
    assert rbig["measured"]["n_structural"] == 6 + 12, rbig["measured"]["structural_counts_by_kind"]
    assert len(rbig["structural"]) <= MAX_STRUCTURAL
    print("[selftest] example dump capped at %d per class with COMPLETE counts (%d gen-byte mismatches, "
          "%d omitted); structural counts complete too OK"
          % (MAX_EXAMPLES, n_expected, rm["n_examples_omitted"]["gen_bytes"]))

    # ---------- the embedded rule/threshold/scope block travels with the result ----------
    assert r["metric"] == METRIC and r["decision_rule"] == DECISION_RULE
    assert r["decision_space"] == list(DECISIONS)
    assert r["thresholds"]["max_field_mismatch"] == MAX_FIELD_MISMATCH == 0
    assert r["thresholds"]["float_tol"] == FLOAT_TOL_DEFAULT == 1e-9
    assert "INCLUSIVE-BOUNDARY" in r["thresholds"]["float_tol_provenance"]
    assert r["scope"]["join_keys"] == ["item", "turn_id"]
    assert r["scope"]["gen_byte_fields"] == list(GEN_BYTE_FIELDS)
    assert r["scope"]["distribution_scope"]["entity_key_fields"] == list(ENTKEY_SCOPE)
    assert r["scope"]["aggregate_keys"] == list(AGGREGATE_KEYS)
    assert r["stamp_keys"] == list(STAMP_KEYS)
    assert _tag_of("out/foldlisten_demarez_subst_dmz_9bit_a_summary.json") == "dmz_9bit_a"
    assert _tag_of("x/foldlisten_demarez_mask_dmz_9bit_b_summary.json") == "dmz_9bit_b"
    print("[selftest] metric / decision_rule / thresholds (with the float-tol provenance disclosure) / "
          "scope / five-key stamp embedded in the result; tag derivation OK")

    print("[selftest] PASS")


def main():
    ap = argparse.ArgumentParser(
        description="offline field-level diff of two draws of the same De Marez span run")
    ap.add_argument("--selftest", action="store_true", help="model-free pure-logic tests (CPU, no i/o)")
    ap.add_argument("--draw-a", dest="draw_a", help="first foldlisten_demarez_{subst,mask}_*_summary.json")
    ap.add_argument("--draw-b", dest="draw_b", help="second summary of the SAME registered run")
    ap.add_argument("--tag", default=None, help="output tag (default: derived from --draw-a's filename)")
    ap.add_argument("--outdir", default="out",
                    help="output directory for demarez_two_draw_diff_<tag>.json (default: out)")
    ap.add_argument("--float-tol", dest="float_tol", type=float, default=FLOAT_TOL_DEFAULT,
                    help="tolerance for the OUTSIDE-TOLERANCE float counts; the exact-equality counts are "
                         "reported beside them either way (default: %(default)r -- see "
                         "thresholds.float_tol_provenance in the artifact)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.draw_a and a.draw_b):
        ap.error("nothing to do: pass --selftest, or --draw-a and --draw-b")
    if a.float_tol < 0:
        ap.error("--float-tol must be >= 0 (got %r)" % a.float_tol)
    run(a.draw_a, a.draw_b, tag=a.tag, outdir=a.outdir, float_tol=a.float_tol)


if __name__ == "__main__":
    main()
