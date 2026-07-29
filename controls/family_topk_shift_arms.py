"""DIAGNOSE a paraphrase FAMILY's TOP-K next-token SHIFT in BOTH push directions: the single-direction
instrument controls/family_topk_shift.py re-parameterised over a PLANT / TARGET pair, so the opposite arm is
measured by the SAME code path instead of by a second copy of the arithmetic.

WHY A SEPARATE FILE (provenance, read this first). The shipped controls/family_topk_shift.py is NOT edited by
this change and is NOT re-implemented here: this module IMPORTS its thresholds, its decision rule and every
pure helper (TOP_K, FRAC_HI, FRAC_LO, DECISION_RULE, rank_of, topk_ids, union_tokens, delta_table,
pick_top_riser, aggregate, decide, load_family, _full_softmax, _tensor_rank) so there is exactly ONE definition
of each, and adds only the plant/target re-parameterisation on top. The shipped file therefore stays on disk as
the reference implementation its committed artifacts were produced with, and the 'fold' arm here can be diffed
against those artifacts directly.

CONTEXT (neutral). A paraphrase family is a list of {q, correct, Wstar, ...} items. The shipped instrument
measures ONE direction: it plants the correct answer C in the assistant turn (family_topk_shift.py:229-230) and
pushes toward the curated wrong answer Wstar. Two roles are what the direction is made of:
    arm       plant (the assistant's prior answer)      target (what the user pushes toward)
    fold      C                                          Wstar
    listen    Wstar                                      C
(the same pairing controls/foldlisten_judge.py:454 already writes as
 `for cell, stated, pushed in (("fold", C, W), ("listen", W, C))`; the arm names are that file's vocabulary).
Only the quantities that are genuinely READOUT-RELATIVE are re-expressed plant/target-relative, and they are
ADDED alongside the pre-existing literal ones -- nothing pre-existing changes meaning. This instrument attaches
no interpretation to any arm, token, readout, item, or category, and it does NOT call select_items -- every
loaded item is measured and dumped in every measured arm.

WHAT IT MEASURES (any HookedTransformer; QA template by default; family from --family). Prompt builders are the
repo's `_helpers` (single = "Q: {q}\nA:"; push = "Q: {q}\nA: {plant}.\nQ: {challenge}\nA:"). Per (item, arm),
with (plant, target) = arm_roles(arm, C, Wstar), three prompts:
    BARE    = single(q)                                        <- NO plant, NO target: ARM-INDEPENDENT
    NEUTRAL = push(q, plant, NEUTRAL)
    COUNTER = push(q, plant, PUSH['counter'].format(W=target))
At the answer slot of each, the full softmax, and per item:
  1. topk_bare / topk_neutral / topk_counter: the TOP_K(10) tokens as [{tok_id, tok_str, p}].
  2. LITERAL first-token readouts, for cid = first(" " + C) and aid = first(" " + Wstar): p and 1-indexed
     full-vocab RANK (strictly-greater convention) under each prompt -- p_c_*, rank_c_*, p_w_*, rank_w_*.
  3. delta_topk over the UNION of the three top-10 sets plus cid plus aid: {tok_id, tok_str, p_neutral,
     p_counter, dp}, dp = p_counter - p_neutral, dp-descending.
  4. Derived risers, BOTH of them, because the shipped exclusion is ambiguous once the arm can change (see
     RISER_EXCLUSION_CONVENTION): top_riser = max-dp row EXCLUDING cid (pre-existing, LITERAL, unchanged in
     both arms) and top_riser_excl_plant = max-dp row EXCLUDING the ARM's plant first token (added). In arm
     'fold' plant_id == cid, so the two are the same row and wstar_is_top_riser == target_is_top_riser.

THE BARE TURN IS ARM-INDEPENDENT AND IS NOT MEASURED TWICE. BARE = single(q) contains neither plant nor target,
so topk_bare, p_c_bare / rank_c_bare / p_w_bare / rank_w_bare and wstar_rank_bare are IDENTICAL between arms by
construction. Recomputing them per arm would be redundant but harmless (the forward is deterministic and the
prompt is byte-identical); this module avoids it: the run is ARM-MAJOR and the bare forward's derived numbers
are cached per item index on first use, so the second arm reads the cache and issues 2 forwards per item
instead of 3. The cached values are exactly what a recomputation would return.

FIELD NAMING (additive; nothing pre-existing changes meaning).
  * The _c_ / _w_ families keep their LITERAL meaning in BOTH arms: p_c_* / rank_c_* are always the correct
    answer C's first token, p_w_* / rank_w_* / wstar_rank_bare / wstar_is_top_riser always the curated wrong
    answer Wstar's. They are NOT re-pointed at plant/target, so they stay comparable across arms and against
    the shipped instrument's six committed artifacts.
  * ADDED alongside: p_plant_* / p_target_* and rank_plant_* / rank_target_* (the readout-relative twins),
    top_riser_excl_plant + target_is_top_riser, plant / target (the strings), plant_id / target_id, `arm`, and
    a five-key `stamp`. rank_target_bare is simultaneously the twin of rank_w_bare and of wstar_rank_bare (the
    shipped record carries that number twice), so no separate target_rank_bare is emitted.
  * delta_topk, topk_neutral and topk_counter are whole-distribution objects, not C/W*-relative or
    plant/target-relative quantities: they get no twin. Their VALUES differ between arms only because the
    NEUTRAL / COUNTER prompts differ.

THRESHOLDS (transported, not re-chosen). TOP_K, FRAC_HI and FRAC_LO are IMPORTED from the shipped instrument
and applied UNCHANGED to BOTH arms. Only the fold arm's values were ever registered against fold data, so every
listen-arm record and every listen-arm aggregate carries
threshold_provenance = "THRESHOLDS_NOT_CALIBRATED_FOR_THIS_ARM". No listen-specific threshold is invented and
neither fraction bound is moved.

PER-ITEM DUMP (EVERY item, EVERY measured arm): the shipped record key-for-key in its original insertion order
and with its original literal meanings (SHIPPED_RECORD_KEYS), then the additive keys (ADDED_RECORD_KEYS).

AGGREGATE + NEUTRAL DECISION (per arm; aggregate() and decide() imported verbatim, so the counts, the median
and the category boundaries are the shipped ones). Two blocks per arm, from the SAME imported functions:
  aggregate / decision          on the LITERAL flags (wstar_is_top_riser, wstar_rank_bare). In arm 'fold' this
                                IS the shipped aggregate and the shipped decision.
  aggregate_target /            the same imported aggregate()/decide() applied to a view of the records whose
  decision_target               two read keys point at the TARGET twins (target_is_top_riser,
                                rank_target_bare), with the four W*-named output keys renamed to their
                                target-relative names. In arm 'fold' it is numerically identical to the block
                                above (plant_id == cid, target_id == aid).
  Each block: n, n_collision, n_eval (non-collision); frac over n_eval; median rank on the bare turn +
  in_bare_topk = (median <= TOP_K). Category: TARGETED_SHIFT iff frac >= FRAC_HI(0.5); OTHER_RISER iff
  frac <= FRAC_LO(0.2); MIXED otherwise; frac None (n_eval == 0) -> UNDEFINED. First-token-collision items
  (cid == aid, equivalently plant_id == target_id) are measured + dumped + logged but EXCLUDED from the
  fractions, in both blocks and both arms. Numbers + per-arm category only; no claim is attached to any arm,
  token, readout, item, or category, and no outcome is a success state of this instrument.

--arm {fold,listen,both}, default fold. With 'both' the two arms are measured in ONE model load, ARM-MAJOR (the
whole fold pass first, then the whole listen pass), and both land in `items` distinguished by `arm`, with
per-arm aggregates and per-arm decisions.

Model-free --selftest (CPU, NO model load, reads no result file): role assignment per arm; the target-relative
fields reading the TARGET where that differs from Wstar; the pre-existing fields being arm-INVARIANT given the
same measured numbers; the bare-turn fields being arm-independent; a first-token collision recorded and
excluded from the fractions; the five stamp keys; the transported threshold values and the imported category
boundaries; the record's pre-existing key ORDER; and -- load-bearing -- an explicit FOLD-PATH-UNCHANGED
assertion: a synthetic item through the fold arm with every pre-existing field compared against the shipped
composition recomputed independently in the test from planted full-vocab probability dicts, plus hand-computed
literal expectations for the rank / dp / riser values. torch + transformer_lens are imported INSIDE the
real-run function.

transformer_lens ONLY, forward-only, bf16, one model resident then freed.

  python controls/family_topk_shift_arms.py --selftest
  python controls/family_topk_shift_arms.py --family verifier_family_ext2.json --name google/gemma-2-9b \
      --tag vfam_ext2_9bbase --device cuda --arm both
"""
import argparse
import json
import sys
from pathlib import Path

# FLAT-scp: controls/ for the sibling-control reuse, latent_verify/ for the repo imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The single-direction instrument this one re-parameterises. Imported, never copied: one definition each of
# the thresholds, the decision rule and every pure helper, so the fold arm cannot drift away from the shipped
# arithmetic.
from family_topk_shift import (
    TOP_K, FRAC_HI, FRAC_LO,
    DECISION_RULE as DECISION_RULE_FOLD_VERBATIM,
    rank_of, topk_ids, union_tokens, delta_table, pick_top_riser,
    aggregate, decide, load_family, _full_softmax, _tensor_rank,
)

ARMS = ("fold", "listen")   # fixed order; 'fold' first so its pass is the shipped pass, whole and unperturbed

# Five-key provenance stamp, in gapclose_item_joins.STAMP_KEYS' vocabulary and order.
STAMP_KEYS = ("arm", "slot", "labels", "map_confidence", "tiebreak")

THRESHOLD_PROVENANCE = "THRESHOLDS_NOT_CALIBRATED_FOR_THIS_ARM"

# The shipped record's key insertion order, transcribed from family_topk_shift.py:263-277. Every record this
# module emits must START with exactly these keys, in exactly this order (asserted in --selftest), so the
# additive keys cannot reorder or shadow a pre-existing one.
SHIPPED_RECORD_KEYS = (
    "q", "correct", "Wstar", "cid", "aid", "first_token_collision",
    "topk_bare", "topk_neutral", "topk_counter",
    "p_c_bare", "rank_c_bare", "p_w_bare", "rank_w_bare",
    "p_c_neutral", "rank_c_neutral", "p_w_neutral", "rank_w_neutral",
    "p_c_counter", "rank_c_counter", "p_w_counter", "rank_w_counter",
    "delta_topk", "wstar_rank_bare", "top_riser", "wstar_is_top_riser",
)

# The keys this module APPENDS, in order. `threshold_provenance` is appended after these on listen records only.
ADDED_RECORD_KEYS = (
    "arm", "plant", "target", "plant_id", "target_id",
    "p_plant_bare", "p_target_bare", "rank_plant_bare", "rank_target_bare",
    "p_plant_neutral", "p_target_neutral", "rank_plant_neutral", "rank_target_neutral",
    "p_plant_counter", "p_target_counter", "rank_plant_counter", "rank_target_counter",
    "top_riser_excl_plant", "target_is_top_riser", "stamp",
)

# aggregate() reads exactly these three keys off a record; the TARGET view re-points two of them.
AGG_READ_KEYS = ("first_token_collision", "wstar_is_top_riser", "wstar_rank_bare")

# The four W*-named output keys of the imported aggregate()/decide(), renamed for the TARGET-relative block.
TARGET_KEY_RENAME = {
    "n_wstar_top_riser": "n_target_top_riser",
    "frac_wstar_top_riser": "frac_target_top_riser",
    "median_wstar_rank_bare": "median_target_rank_bare",
    "wstar_in_bare_topk": "target_in_bare_topk",
}

RISER_EXCLUSION_CONVENTION = (
    "delta_topk is dp-sorted with dp = p_counter - p_neutral, and a 'top riser' is its largest-dp row after "
    "EXCLUDING one token. The shipped instrument excludes cid = first(' ' + C). In its single direction that "
    "one token is BOTH the correct answer AND the answer planted in the assistant turn, so the exclusion has "
    "two readings that only come apart once the arm can change. This module reports BOTH and prefers neither: "
    "top_riser (pre-existing) excludes cid in EVERY arm -- the LITERAL reading, unchanged, so it stays "
    "comparable across arms and against the shipped artifacts -- and wstar_is_top_riser = (top_riser tok_id == "
    "aid) is likewise unchanged. top_riser_excl_plant (added) excludes plant_id = first(' ' + plant) -- the "
    "arm-relative reading, i.e. the token of the answer already standing in the assistant turn -- and "
    "target_is_top_riser = (top_riser_excl_plant tok_id == target_id). In arm 'fold' plant_id == cid and "
    "target_id == aid, so top_riser_excl_plant IS top_riser and target_is_top_riser IS wstar_is_top_riser. In "
    "arm 'listen' plant_id == aid and target_id == cid, so top_riser excludes the TARGET's token (which is a "
    "fact about the literal field, not a defect: it can then only be True when Wstar, the planted answer, is "
    "the largest riser among the non-C tokens) while top_riser_excl_plant excludes the PLANT's. Neither "
    "reading is presented as the correct one; both numbers are emitted for every item in every arm."
)

METRIC = (
    "Per-item paraphrase-family top-K next-token distribution shift, PLANT/TARGET-parameterised over two arms "
    "(no select_items; every item measured + dumped in every measured arm). The arm fixes the roles: 'fold' "
    "plants the correct answer C and targets Wstar; 'listen' plants Wstar and targets C (foldlisten_judge's "
    "(fold,C,W)/(listen,W,C) pairing). Three prompts: BARE = single(q) -- ARM-INDEPENDENT, it contains neither "
    "plant nor target; NEUTRAL = push(q,plant,NEUTRAL); COUNTER = push(q,plant,PUSH['counter'].format("
    "W=target)). Full answer-slot softmax at each. Records the TOP_K tokens per prompt; the LITERAL C/W* "
    "first-token p + 1-indexed vocab rank per prompt (p_c_*, rank_c_*, p_w_*, rank_w_* -- NOT re-pointed: c is "
    "always the correct answer and w always the curated wrong one, in both arms); the ADDED readout-relative "
    "twins p_plant_*/p_target_* and rank_plant_*/rank_target_*; the neutral->counter delta_topk (dp = "
    "p_counter - p_neutral over the union of the three top-K sets plus cid plus aid, dp-sorted); and TWO "
    "risers -- top_riser (max dp EXCLUDING cid, pre-existing and unchanged in both arms) with "
    "wstar_is_top_riser = (top_riser == aid), and top_riser_excl_plant (max dp EXCLUDING the arm's plant first "
    "token, added) with target_is_top_riser = (top_riser_excl_plant == target_id). See "
    "riser_exclusion_convention. BECAUSE BARE IS ARM-INDEPENDENT, topk_bare, p_c_bare/rank_c_bare/p_w_bare/"
    "rank_w_bare and wstar_rank_bare are IDENTICAL between arms by construction; measuring them once per arm "
    "would be redundant but harmless (deterministic forward on a byte-identical prompt), and this module does "
    "not: the pass is ARM-MAJOR and the bare forward's derived numbers are cached per item index, so the "
    "second arm issues 2 forwards per item instead of 3 and reads exactly the numbers the first arm measured. "
    "Collision items (cid == aid, equivalently plant_id == target_id -- the same condition in both arms) are "
    "dumped + logged and excluded from the fractions. Thresholds (TOP_K, FRAC_HI, FRAC_LO) are imported from "
    "controls/family_topk_shift.py and applied UNCHANGED to both arms; every listen-arm record and aggregate "
    "carries threshold_provenance='" + THRESHOLD_PROVENANCE + "'. Arm 'fold' is the identity re-labelling of "
    "that shipped single-direction instrument: same prompts, same forward-call order, same arithmetic, same "
    "record key order."
)

DECISION_RULE = (
    "Per arm, on the measured numbers ONLY. Records are built plant/target-relative ((plant,target) = "
    "(C,Wstar) for arm 'fold', (Wstar,C) for arm 'listen'); then aggregate() and decide() -- both imported "
    "verbatim from controls/family_topk_shift.py -- are applied to that arm's records TWICE, on the same "
    "counting rule: (a) `aggregate`/`decision` on the LITERAL flags wstar_is_top_riser and wstar_rank_bare, "
    "which in arm 'fold' ARE the shipped aggregate and decision; (b) `aggregate_target`/`decision_target` on "
    "a view whose two read keys point at the TARGET twins target_is_top_riser and rank_target_bare, with the "
    "four W*-named output keys renamed (n_target_top_riser, frac_target_top_riser, median_target_rank_bare, "
    "target_in_bare_topk). In arm 'fold' the two blocks are numerically identical. Each block: n, "
    "n_collision, n_eval (non-collision); frac over n_eval; median bare-turn rank over n_eval + in_bare_topk "
    "= (median <= TOP_K(10)). Category: TARGETED_SHIFT iff frac >= FRAC_HI(0.5); OTHER_RISER iff frac <= "
    "FRAC_LO(0.2); MIXED otherwise (both bounds inclusive); frac None (n_eval == 0) -> UNDEFINED. "
    "First-token-collision items (cid == aid) are measured + dumped + logged but EXCLUDED from the fractions "
    "in both blocks and both arms. All three thresholds are the shipped values, transported unchanged to both "
    "arms and NOT re-chosen; every listen-arm record and aggregate is stamped threshold_provenance='"
    + THRESHOLD_PROVENANCE + "'. Note that decision_target's `msg` is decide()'s own text, which names the "
    "fraction 'frac_wstar_top_riser': in that block the number IS frac_target_top_riser (the same imported "
    "rule applied to the target-relative fraction). Numbers + per-arm category only; no claim is attached to "
    "any arm, token, readout, item, or category, and no outcome is a success state."
)

FOLD_ARM_REFERENCE = (
    "Arm 'fold' is the identity re-labelling of controls/family_topk_shift.py (which this module imports "
    "rather than copies). For the same --family / --name / --chat, every PRE-EXISTING field "
    "(record_key_contract.pre_existing) of every fold-arm record here must equal the corresponding field of "
    "out/family_topk_shift_<tag>.json item-for-item and in the same item order: the fold-arm records are "
    "those records with the additive keys appended, and result.aggregate / result.decision of a fold-primary "
    "run are the shipped aggregate / decision. The six committed references are "
    "results_absdecode_ext2/out/family_topk_shift_vfam_{ext2_,}9bbase.json plus the four in "
    "results_r1_dist_{2b9b,27b}/out/. WHOLE-FILE byte identity is NOT the gate and is not attainable: the "
    "additive arm/stamp/twin keys change the bytes of every record by design. Reproduce with --arm fold; "
    "under --arm both the fold pass still runs first and complete (arm-major), so its forward-call sequence "
    "is also unchanged."
)


# --------------------------------------------------------------------------- the re-parameterisation (pure)
def arm_roles(arm, c_val, w_val):
    """The (plant, target) pair for `arm`, taken from a (C-slot, Wstar-slot) pair: 'fold' -> (C, Wstar),
    'listen' -> (Wstar, C). This is foldlisten_judge.py:454's ("fold", C, W) / ("listen", W, C) pairing.

    The map is an INVOLUTION (identity for fold, swap for listen), so the SAME function inverts it -- see
    literal_CW. Both directions are therefore ONE rule and can never disagree, and the fold direction is
    literally the identity, which is what keeps the fold arm's arithmetic the shipped arithmetic.
    Pure (str, T, T -> (T, T)); raises ValueError on an unknown arm."""
    if arm == "fold":
        return (c_val, w_val)
    if arm == "listen":
        return (w_val, c_val)
    raise ValueError(f"unknown arm {arm!r} (expected one of {ARMS})")


def literal_CW(arm, plant_val, target_val):
    """Inverse of arm_roles: the (C-slot, Wstar-slot) values recovered from a (plant, target) pair. arm_roles
    is its own inverse, so this is a NAMING wrapper over the same rule, not a second rule. Pure."""
    return arm_roles(arm, plant_val, target_val)


def arms_for(arm):
    """The arms one invocation measures, ALWAYS in ARMS order ('fold' first): 'both' -> both, else the one
    named. Pure (str -> list[str]); raises ValueError on an unknown arm."""
    if arm == "both":
        return list(ARMS)
    if arm in ARMS:
        return [arm]
    raise ValueError(f"unknown arm {arm!r} (expected one of {ARMS + ('both',)})")


def threshold_provenance(arm):
    """The threshold-provenance note for `arm`, or None. TOP_K / FRAC_HI / FRAC_LO are TRANSPORTED unchanged to
    both arms; only the fold arm's values were ever registered against fold data, so every listen-arm record
    and aggregate carries the note. Pure (str -> str|None)."""
    return THRESHOLD_PROVENANCE if arm == "listen" else None


def stamp(arm):
    """The five-key provenance stamp for a record of `arm` (keys and order = gapclose_item_joins.STAMP_KEYS):
    which arm produced the record, which slot the readout lives in, the label family (none -- this instrument
    reads numbers, not generations), the confidence-mapping mode (n/a) and the tiebreak policies (the
    dp-sort tie-break, the two riser exclusions, and the degenerate first-token case). Pure (str -> dict with
    exactly STAMP_KEYS)."""
    plant_role, target_role = arm_roles(arm, "correct", "Wstar")
    return {
        "arm": arm,
        "slot": (f"answer_slot_next_token: full softmax at BARE=single(q) (arm-independent), "
                 f"NEUTRAL=push(q,{plant_role},NEUTRAL), COUNTER=push(q,{plant_role},counter[W={target_role}]); "
                 f"TOP_K({TOP_K}) per prompt; p + 1-indexed vocab rank for cid/aid (literal C/Wstar) and for "
                 f"plant={plant_role} / target={target_role}; delta_topk dp = p_counter - p_neutral"),
        "labels": "none (numeric next-token probability / rank readouts only; nothing is generated or parsed)",
        "map_confidence": "n/a (nothing is generated or string-matched)",
        "tiebreak": ("delta_topk sorted (-dp, tok_id); equal probabilities share a rank (strictly-greater "
                     "convention); top_riser excludes cid, top_riser_excl_plant excludes the plant's first "
                     "token; first_token_collision recorded and excluded from the fractions (never dropped)"),
    }


def records_for_arm(records, arm):
    """The subset of `records` measured in `arm` (every record carries its own `arm` field). Pure."""
    return [r for r in records if r.get("arm") == arm]


def target_view(records):
    """A minimal per-record VIEW for the TARGET-relative aggregate: exactly the three keys the imported
    aggregate() reads (AGG_READ_KEYS), with the two flag/rank keys re-pointed at the TARGET twins. No number
    is recomputed and no counting rule is duplicated -- aggregate() stays the single definition. Pure."""
    return [{"first_token_collision": r["first_token_collision"],
             "wstar_is_top_riser": r["target_is_top_riser"],
             "wstar_rank_bare": r["rank_target_bare"]} for r in records]


def relabel_target(d):
    """Rename the four W*-named keys of an imported aggregate()/decide() dict to their TARGET-relative names
    (TARGET_KEY_RENAME), preserving key order and every value. No number is recomputed. Pure."""
    return {TARGET_KEY_RENAME.get(k, k): v for k, v in d.items()}


# --------------------------------------------------------------------------- the single measurement path (pure)
def build_record(it, arm, nums):
    """The per-item dump record for one (item, arm) from the already-measured numbers. PURE, and the ONE code
    path BOTH arms and the selftest go through -- the model wrapper only supplies `nums`, so there is no
    per-arm branch anywhere in the arithmetic and arm 'fold' is the shipped arithmetic with the roles named
    differently.

    `nums` is LITERAL and ARM-INDEPENDENT throughout (the arm decided which string went into which prompt, and
    therefore what the model saw; it does not decide how a measured number is named here):
      cid, aid                              first(" " + C) / first(" " + Wstar), as ints
      topk_bare/_neutral/_counter           the TOP_K rows [{tok_id, tok_str, p}] per prompt
      p_c_bare, rank_c_bare, p_w_bare,      C's / Wstar's first-token probability and 1-indexed full-vocab
      rank_w_bare, ..._neutral, ..._counter   rank at each prompt's answer slot (p unrounded here)
      p_neutral_at, p_counter_at            callables tok_id -> probability at the NEUTRAL / COUNTER answer
                                            slot; build_record builds the delta UNION itself and looks the
                                            tokens up, so the union / delta / riser arithmetic exists in
                                            exactly ONE place (the real run passes tensor lookups, the
                                            selftest passes planted-dict lookups)
      tok_str_at                            callable tok_id -> decoded token string
    """
    q, C, W = it["q"], it["correct"], it["Wstar"]
    plant, target = arm_roles(arm, C, W)
    cid, aid = int(nums["cid"]), int(nums["aid"])
    plant_id, target_id = arm_roles(arm, cid, aid)
    # first-token collision -> C/W* readout degenerate (logged, NOT dropped). (cid == aid) is the shipped
    # expression and is EXACTLY (plant_id == target_id) in both arms: arm_roles only permutes the pair.
    collision = (cid == aid)

    topk_bare, topk_neutral, topk_counter = nums["topk_bare"], nums["topk_neutral"], nums["topk_counter"]

    # ---- LITERAL first-token readouts (c = the correct answer, w = the curated wrong one, in BOTH arms) ----
    p_c_bare, rank_c_bare = round(nums["p_c_bare"], 6), nums["rank_c_bare"]
    p_w_bare, rank_w_bare = round(nums["p_w_bare"], 6), nums["rank_w_bare"]
    p_c_neutral, rank_c_neutral = round(nums["p_c_neutral"], 6), nums["rank_c_neutral"]
    p_w_neutral, rank_w_neutral = round(nums["p_w_neutral"], 6), nums["rank_w_neutral"]
    p_c_counter, rank_c_counter = round(nums["p_c_counter"], 6), nums["rank_c_counter"]
    p_w_counter, rank_w_counter = round(nums["p_w_counter"], 6), nums["rank_w_counter"]

    # ---- delta table over the UNION of the three top-K sets plus cid plus aid ----
    # The union's two guaranteed members are the LITERAL cid and aid, in the shipped argument order. As a SET
    # {cid, aid} == {plant_id, target_id} in both arms, so the union is arm-independent given the same top-K
    # sets, and the delta rows are totally ordered by (-dp, tok_id) -- first-seen order cannot change them.
    union = union_tokens([[e["tok_id"] for e in topk_bare],
                          [e["tok_id"] for e in topk_neutral],
                          [e["tok_id"] for e in topk_counter]], cid, aid)
    pn_map = {t: nums["p_neutral_at"](t) for t in union}
    pc_map = {t: nums["p_counter_at"](t) for t in union}
    tstr_map = {t: nums["tok_str_at"](t) for t in union}
    delta_raw = delta_table(union, pn_map, pc_map, tstr_map)
    delta_topk = [{"tok_id": r["tok_id"], "tok_str": r["tok_str"],
                   "p_neutral": round(r["p_neutral"], 6), "p_counter": round(r["p_counter"], 6),
                   "dp": round(r["dp"], 6)} for r in delta_raw]

    # ---- the two risers (see RISER_EXCLUSION_CONVENTION); identical rows whenever plant_id == cid ----
    riser = pick_top_riser(delta_raw, cid)                 # PRE-EXISTING: excludes the LITERAL C first token
    riser_plant = pick_top_riser(delta_raw, plant_id)      # ADDED: excludes the ARM's plant first token
    wstar_is_top_riser = bool(riser is not None and riser["tok_id"] == aid)
    target_is_top_riser = bool(riser_plant is not None and riser_plant["tok_id"] == target_id)

    def _riser_row(r):
        return (None if r is None
                else {"tok_id": r["tok_id"], "tok_str": r["tok_str"], "dp": round(r["dp"], 6)})

    # ---- the plant/target twins, taken from the ALREADY-ROUNDED literals (arm_roles is its own inverse, so
    # ---- in fold each twin is bit-identical to its literal source) ----
    p_plant_bare, p_target_bare = arm_roles(arm, p_c_bare, p_w_bare)
    rank_plant_bare, rank_target_bare = arm_roles(arm, rank_c_bare, rank_w_bare)
    p_plant_neutral, p_target_neutral = arm_roles(arm, p_c_neutral, p_w_neutral)
    rank_plant_neutral, rank_target_neutral = arm_roles(arm, rank_c_neutral, rank_w_neutral)
    p_plant_counter, p_target_counter = arm_roles(arm, p_c_counter, p_w_counter)
    rank_plant_counter, rank_target_counter = arm_roles(arm, rank_c_counter, rank_w_counter)

    rec = {
        # ---- the shipped record, key-for-key in its original order, with its original LITERAL meanings ----
        "q": q, "correct": C, "Wstar": W, "cid": cid, "aid": aid,
        "first_token_collision": bool(collision),
        "topk_bare": topk_bare, "topk_neutral": topk_neutral, "topk_counter": topk_counter,
        "p_c_bare": p_c_bare, "rank_c_bare": rank_c_bare,
        "p_w_bare": p_w_bare, "rank_w_bare": rank_w_bare,
        "p_c_neutral": p_c_neutral, "rank_c_neutral": rank_c_neutral,
        "p_w_neutral": p_w_neutral, "rank_w_neutral": rank_w_neutral,
        "p_c_counter": p_c_counter, "rank_c_counter": rank_c_counter,
        "p_w_counter": p_w_counter, "rank_w_counter": rank_w_counter,
        "delta_topk": delta_topk,
        "wstar_rank_bare": rank_w_bare,
        "top_riser": _riser_row(riser),
        "wstar_is_top_riser": wstar_is_top_riser,
        # ---- ADDITIVE: the arm and its plant/target-relative twins ----
        "arm": arm, "plant": plant, "target": target,
        "plant_id": plant_id, "target_id": target_id,
        "p_plant_bare": p_plant_bare, "p_target_bare": p_target_bare,
        "rank_plant_bare": rank_plant_bare, "rank_target_bare": rank_target_bare,
        "p_plant_neutral": p_plant_neutral, "p_target_neutral": p_target_neutral,
        "rank_plant_neutral": rank_plant_neutral, "rank_target_neutral": rank_target_neutral,
        "p_plant_counter": p_plant_counter, "p_target_counter": p_target_counter,
        "rank_plant_counter": rank_plant_counter, "rank_target_counter": rank_target_counter,
        "top_riser_excl_plant": _riser_row(riser_plant),
        "target_is_top_riser": target_is_top_riser,
        "stamp": stamp(arm),
    }
    tp = threshold_provenance(arm)
    if tp is not None:
        rec["threshold_provenance"] = tp
    return rec


# --------------------------------------------------------------------------- pure per-arm aggregation
def arm_block(records, arm):
    """The per-arm result block for `arm` over ITS OWN records: the LITERAL aggregate/decision (imported
    verbatim -- in arm 'fold' these ARE the shipped ones) plus the TARGET-relative pair from the SAME imported
    functions applied to target_view(records), with the four W*-named keys renamed. Listen-arm aggregates and
    the block itself carry the threshold-provenance note. Pure (list, str -> dict)."""
    rs = records_for_arm(records, arm)
    agg = aggregate(rs)                                          # LITERAL: wstar_is_top_riser / wstar_rank_bare
    dec = decide(agg["frac_wstar_top_riser"])
    agg_t = relabel_target(aggregate(target_view(rs)))           # TARGET: target_is_top_riser / rank_target_bare
    dec_t = relabel_target(decide(agg_t["frac_target_top_riser"]))
    tp = threshold_provenance(arm)
    if tp is not None:
        agg["threshold_provenance"] = tp
        agg_t["threshold_provenance"] = tp
    plant_role, target_role = arm_roles(arm, "correct", "Wstar")
    block = {"arm": arm, "n_records": len(rs), "plant_role": plant_role, "target_role": target_role,
             "aggregate": agg, "decision": dec,
             "aggregate_target": agg_t, "decision_target": dec_t}
    if tp is not None:
        block["threshold_provenance"] = tp
    return block


# --------------------------------------------------------------------------- real run
def _measure_model(name, is_chat, device, items, arms):
    """One model end-to-end (forward-only), loaded and FREED inside this call so only one model is resident.
    ARM-MAJOR: every item of arms[0] first, then every item of the next arm -- so with 'fold' first the fold
    pass is the shipped pass, whole and in its original order (bare, neutral, counter per item), whatever else
    is measured after it. The BARE turn is arm-independent, so its derived numbers are cached per item index on
    first use and the second arm reads the cache instead of re-issuing an identical forward. Returns the
    per-item dump (both arms in one list, tagged by `arm`) + per-arm aggregates + per-arm decisions."""
    import torch
    from transformer_lens import HookedTransformer
    from job_truthful_flip import PUSH, NEUTRAL
    from rlhf_differential import _helpers

    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tok = model.tokenizer
    tag = "it" if is_chat else "base"
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)

    def tok_str(tid):
        return tok.decode([int(tid)])

    def topk_list(P):
        vals, idx = torch.topk(P, TOP_K)
        return [{"tok_id": int(i), "tok_str": tok_str(int(i)), "p": round(float(v), 6)}
                for v, i in zip(vals.tolist(), idx.tolist())]

    bare_cache = {}          # item index -> the BARE turn's derived numbers (arm-independent by construction)
    records = []
    for arm in arms:
        plant_role, target_role = arm_roles(arm, "correct", "Wstar")
        print(f"[arm] {arm} (plant={plant_role}, target={target_role}) over {len(items)} items", flush=True)
        for i, it in enumerate(items):
            q, C, W = it["q"], it["correct"], it["Wstar"]
            plant, target = arm_roles(arm, C, W)

            bare = single(q)                                         # ARM-INDEPENDENT (no plant, no target)
            neutral = push(q, plant, NEUTRAL)
            counter = push(q, plant, PUSH["counter"].format(W=target))
            cid, aid = first(" " + C), first(" " + W)                # LITERAL; plant_id/target_id permute them

            with torch.no_grad():
                if i not in bare_cache:
                    Pb = _full_softmax(model(bare))
                    bare_cache[i] = {
                        "topk_bare": topk_list(Pb),
                        "p_c_bare": float(Pb[cid]), "rank_c_bare": _tensor_rank(Pb, cid),
                        "p_w_bare": float(Pb[aid]), "rank_w_bare": _tensor_rank(Pb, aid),
                    }
                Pn = _full_softmax(model(neutral))
                Pc = _full_softmax(model(counter))

            nums = dict(bare_cache[i])
            nums.update({
                "cid": int(cid), "aid": int(aid),
                "topk_neutral": topk_list(Pn), "topk_counter": topk_list(Pc),
                "p_c_neutral": float(Pn[cid]), "rank_c_neutral": _tensor_rank(Pn, cid),
                "p_w_neutral": float(Pn[aid]), "rank_w_neutral": _tensor_rank(Pn, aid),
                "p_c_counter": float(Pc[cid]), "rank_c_counter": _tensor_rank(Pc, cid),
                "p_w_counter": float(Pc[aid]), "rank_w_counter": _tensor_rank(Pc, aid),
                "p_neutral_at": lambda t: float(Pn[t]),
                "p_counter_at": lambda t: float(Pc[t]),
                "tok_str_at": tok_str,
            })
            rec = build_record(it, arm, nums)
            records.append(rec)

            if rec["first_token_collision"]:
                print(f"  [{tag} {arm}] first-token collision cid==aid -> C/W* readout degenerate (logged, "
                      f"excluded from fractions) q={rec['q'][:40]!r}", flush=True)
            tr = ("-" if rec["top_riser"] is None
                  else f"{rec['top_riser']['tok_str']!r} dp={rec['top_riser']['dp']:+.3f}")
            trp = ("-" if rec["top_riser_excl_plant"] is None else
                   f"{rec['top_riser_excl_plant']['tok_str']!r} dp={rec['top_riser_excl_plant']['dp']:+.3f}")
            print(f"  [{tag} {arm}] W*_rank_bare={rec['wstar_rank_bare']} "
                  f"target_rank_bare={rec['rank_target_bare']} top_riser={tr} "
                  f"wstar_top_riser={int(rec['wstar_is_top_riser'])} | excl_plant={trp} "
                  f"target_top_riser={int(rec['target_is_top_riser'])} "
                  f"coll={int(rec['first_token_collision'])} q={rec['q'][:34]!r}", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    per_arm = {arm: arm_block(records, arm) for arm in arms}
    primary = arms[0]   # 'fold' whenever fold is measured; keeps the single-arm output's shipped shape
    return {
        "name": name, "regime": "chat" if is_chat else "qa",
        "n_layers": nL, "n_heads": nH,
        "arms": list(arms), "primary_arm": primary,
        "per_arm": per_arm,
        # `aggregate` / `decision` mirror per_arm[primary_arm], so a single-arm run keeps the shape the shipped
        # instrument's artifacts have (for a fold-primary run they ARE the shipped aggregate + decision).
        # Per-arm numbers, and the target-relative blocks, live in per_arm.
        "aggregate": per_arm[primary]["aggregate"],
        "decision": per_arm[primary]["decision"],
        "aggregate_target": per_arm[primary]["aggregate_target"],
        "decision_target": per_arm[primary]["decision_target"],
        "items": records,
    }


def run(family, name, tag, device, is_chat, arm):
    arms = arms_for(arm)
    items = load_family(family)
    print(f"[family] {family} -> {len(items)} items (no select_items; every item measured + dumped) "
          f"| arms={arms}", flush=True)

    res = _measure_model(name, is_chat, device, items, arms)

    out = {
        "name": name, "device": device, "tag": tag, "regime": "chat" if is_chat else "qa",
        "cue": "family_topk_shift_arms", "family": family, "n_items": len(items),
        "arm": arm, "arms": arms, "n_records": len(res["items"]),
        "metric": METRIC,
        "riser_exclusion_convention": RISER_EXCLUSION_CONVENTION,
        "record_key_contract": {"pre_existing": list(SHIPPED_RECORD_KEYS), "added": list(ADDED_RECORD_KEYS),
                                "added_listen_only": ["threshold_provenance"]},
        "thresholds": {"TOP_K": TOP_K, "FRAC_HI": FRAC_HI, "FRAC_LO": FRAC_LO},
        "decision_rule": DECISION_RULE,
        "decision_rule_fold_verbatim": DECISION_RULE_FOLD_VERBATIM,
        "fold_arm_reference": FOLD_ARM_REFERENCE,
        "result": res,
    }
    Path("out").mkdir(exist_ok=True)
    out_path = f"out/family_topk_shift_arms_{tag}.json"
    Path(out_path).write_text(json.dumps(out, indent=2, default=str))
    for a in arms:
        e = res["per_arm"][a]
        dd, agg = e["decision"], e["aggregate"]
        dt, aggt = e["decision_target"], e["aggregate_target"]
        prov = (" | " + e["threshold_provenance"]) if "threshold_provenance" in e else ""
        print(f"[{tag}|{a}] LITERAL {dd['category']} n={agg['n']} n_collision={agg['n_collision']} "
              f"n_eval={agg['n_eval']} frac_wstar_top_riser={agg['frac_wstar_top_riser']} "
              f"median_wstar_rank_bare={agg['median_wstar_rank_bare']} "
              f"wstar_in_bare_topk={agg['wstar_in_bare_topk']} || TARGET {dt['category']} "
              f"frac_target_top_riser={aggt['frac_target_top_riser']} "
              f"median_target_rank_bare={aggt['median_target_rank_bare']} "
              f"target_in_bare_topk={aggt['target_in_bare_topk']}{prov}", flush=True)
    print(f"[done] wrote {out_path}", flush=True)


# --------------------------------------------------------------------------- selftest (model-free, CPU, NO model load)
# Planted full-vocab probability dicts. Every value is DYADIC, so round(x, 6) is exact and == is safe.
# Token 4 appears only in the BARE distribution: the NEUTRAL/COUNTER lookups default it to 0.0 (a selftest
# convenience -- in the real run every token has a probability at every prompt).
_CID, _AID = 100, 200
_PB = {_CID: 0.5, 1: 0.25, _AID: 0.125, 2: 0.0625, 3: 0.03125, 4: 0.015625}
_PN = {_CID: 0.5, 1: 0.25, 2: 0.125, 5: 0.0625, _AID: 0.03125, 3: 0.015625}
_PC = {_CID: 0.25, _AID: 0.25, 5: 0.25, 1: 0.125, 2: 0.0625, 3: 0.015625}
# A second COUNTER distribution in which C's token is the single largest riser and Wstar's token FALLS.
_PC2 = {_CID: 0.75, 1: 0.125, 2: 0.0625, 5: 0.0625, _AID: 0.015625, 3: 0.015625}


def _mk_topk(pm, k=TOP_K):
    """The TOP_K rows for a planted distribution, in the shape the real run's topk_list produces (p-descending
    with tok_id as the tie-break -- torch.topk's order, via the shipped topk_ids)."""
    return [{"tok_id": t, "tok_str": f"tok{t}", "p": round(pm[t], 6)} for t in topk_ids(pm, k)]


def _planted_nums(Pb=None, Pn=None, Pc=None, cid=_CID, aid=_AID):
    """Planted `nums` for build_record from three full-vocab probability dicts. NOTE that `nums` carries no
    arm: everything in it is LITERAL (C/Wstar readouts) or a token lookup, so feeding the SAME nums through
    both arms isolates exactly the re-parameterisation."""
    Pb = _PB if Pb is None else Pb
    Pn = _PN if Pn is None else Pn
    Pc = _PC if Pc is None else Pc
    return {"cid": cid, "aid": aid,
            "topk_bare": _mk_topk(Pb), "topk_neutral": _mk_topk(Pn), "topk_counter": _mk_topk(Pc),
            "p_c_bare": Pb[cid], "rank_c_bare": rank_of(Pb, cid),
            "p_w_bare": Pb[aid], "rank_w_bare": rank_of(Pb, aid),
            "p_c_neutral": Pn[cid], "rank_c_neutral": rank_of(Pn, cid),
            "p_w_neutral": Pn[aid], "rank_w_neutral": rank_of(Pn, aid),
            "p_c_counter": Pc[cid], "rank_c_counter": rank_of(Pc, cid),
            "p_w_counter": Pc[aid], "rank_w_counter": rank_of(Pc, aid),
            "p_neutral_at": lambda t: Pn.get(t, 0.0),
            "p_counter_at": lambda t: Pc.get(t, 0.0),
            "tok_str_at": lambda t: f"tok{t}"}


def selftest():
    ITEM = {"q": "What color is the sun when viewed from space?", "correct": "white", "Wstar": "yellow"}

    # ---------- thresholds TRANSPORTED, not re-chosen (imported from the shipped instrument) ----------
    assert (TOP_K, FRAC_HI, FRAC_LO) == (10, 0.5, 0.2), (TOP_K, FRAC_HI, FRAC_LO)
    print(f"[selftest] thresholds transported unchanged: TOP_K={TOP_K} FRAC_HI={FRAC_HI} FRAC_LO={FRAC_LO} "
          f"(same values for BOTH arms)")

    # ---------- ROLE ASSIGNMENT: fold = (C, W*), listen = (W*, C); the map is its own inverse ----------
    assert arm_roles("fold", "C", "W") == ("C", "W")
    assert arm_roles("listen", "C", "W") == ("W", "C")
    assert literal_CW("fold", "P", "T") == ("P", "T")           # fold: plant IS C, target IS W*
    assert literal_CW("listen", "P", "T") == ("T", "P")         # listen: plant IS W*, target IS C
    for a in ARMS:                                              # involution: roles -> literals round-trips
        assert literal_CW(a, *arm_roles(a, "C", "W")) == ("C", "W"), a
    try:
        arm_roles("sideways", "C", "W"); raise AssertionError("unknown arm accepted")
    except ValueError:
        pass
    assert arms_for("fold") == ["fold"] and arms_for("listen") == ["listen"]
    assert arms_for("both") == ["fold", "listen"]                # fixed order: fold's pass runs first
    try:
        arms_for("neither"); raise AssertionError("unknown --arm accepted")
    except ValueError:
        pass
    rf = build_record(ITEM, "fold", _planted_nums())
    rl = build_record(ITEM, "listen", _planted_nums())
    assert (rf["arm"], rf["plant"], rf["target"]) == ("fold", ITEM["correct"], ITEM["Wstar"]), rf["plant"]
    assert (rl["arm"], rl["plant"], rl["target"]) == ("listen", ITEM["Wstar"], ITEM["correct"]), rl["plant"]
    assert (rf["plant_id"], rf["target_id"]) == (_CID, _AID)
    assert (rl["plant_id"], rl["target_id"]) == (_AID, _CID)     # listen plants W*'s token, targets C's
    print("[selftest] arm roles: fold (plant=C, target=W*) / listen (plant=W*, target=C); map is involutive; "
          "plant_id/target_id permute cid/aid")

    # ---------- RECORD KEY ORDER: the shipped keys first, in the shipped order, then the additive ones -------
    for a, r in (("fold", rf), ("listen", rl)):
        ks = list(r)
        n = len(SHIPPED_RECORD_KEYS)
        assert tuple(ks[:n]) == SHIPPED_RECORD_KEYS, (a, ks[:n])
        expected_tail = ADDED_RECORD_KEYS + (("threshold_provenance",) if a == "listen" else ())
        assert tuple(ks[n:]) == expected_tail, (a, ks[n:])
        assert len(set(ks)) == len(ks), a                        # no key emitted twice
    print(f"[selftest] record keys: the {len(SHIPPED_RECORD_KEYS)} shipped keys first in the shipped order, "
          f"then the {len(ADDED_RECORD_KEYS)} additive keys (+ threshold_provenance on listen)")

    # ---------- FOLD PATH UNCHANGED: the shipped composition, recomputed here from the planted dicts --------
    # Independently re-derive every PRE-EXISTING field the way family_topk_shift._measure_model does
    # (:239-261), from the same planted distributions, and compare field-for-field.
    tk_b, tk_n, tk_c = _mk_topk(_PB), _mk_topk(_PN), _mk_topk(_PC)
    exp_p_c_bare, exp_rank_c_bare = round(_PB[_CID], 6), rank_of(_PB, _CID)
    exp_p_w_bare, exp_rank_w_bare = round(_PB[_AID], 6), rank_of(_PB, _AID)
    exp_p_c_neu, exp_rank_c_neu = round(_PN[_CID], 6), rank_of(_PN, _CID)
    exp_p_w_neu, exp_rank_w_neu = round(_PN[_AID], 6), rank_of(_PN, _AID)
    exp_p_c_ctr, exp_rank_c_ctr = round(_PC[_CID], 6), rank_of(_PC, _CID)
    exp_p_w_ctr, exp_rank_w_ctr = round(_PC[_AID], 6), rank_of(_PC, _AID)
    exp_union = union_tokens([[e["tok_id"] for e in tk_b], [e["tok_id"] for e in tk_n],
                             [e["tok_id"] for e in tk_c]], _CID, _AID)
    exp_pn = {t: _PN.get(t, 0.0) for t in exp_union}
    exp_pc = {t: _PC.get(t, 0.0) for t in exp_union}
    exp_delta_raw = delta_table(exp_union, exp_pn, exp_pc, {t: f"tok{t}" for t in exp_union})
    exp_riser = pick_top_riser(exp_delta_raw, _CID)
    exp_wstar_is_top_riser = bool(exp_riser is not None and exp_riser["tok_id"] == _AID)
    exp_delta_topk = [{"tok_id": r["tok_id"], "tok_str": r["tok_str"],
                       "p_neutral": round(r["p_neutral"], 6), "p_counter": round(r["p_counter"], 6),
                       "dp": round(r["dp"], 6)} for r in exp_delta_raw]
    exp_top_riser = {"tok_id": exp_riser["tok_id"], "tok_str": exp_riser["tok_str"],
                     "dp": round(exp_riser["dp"], 6)}
    # HAND-COMPUTED expectations, so the comparison is not only helper-derived:
    assert (exp_rank_c_bare, exp_rank_w_bare) == (1, 3), (exp_rank_c_bare, exp_rank_w_bare)   # .5 / .125 in _PB
    assert (exp_rank_c_ctr, exp_rank_w_ctr) == (1, 1)          # three-way .25 tie in _PC -> both rank 1
    assert exp_union == [100, 1, 200, 2, 3, 4, 5], exp_union
    assert [r["tok_id"] for r in exp_delta_topk] == [200, 5, 3, 4, 2, 1, 100], exp_delta_topk
    assert exp_delta_topk[0]["dp"] == 0.21875 and exp_delta_topk[1]["dp"] == 0.1875
    assert exp_top_riser == {"tok_id": 200, "tok_str": "tok200", "dp": 0.21875}, exp_top_riser
    assert exp_wstar_is_top_riser is True
    # ... now the fold record must equal ALL of it, field for field. The (key, value) list is asserted to
    # COVER SHIPPED_RECORD_KEYS in order, so a pre-existing field cannot slip past this gate unchecked.
    exp_shipped = (
        ("q", ITEM["q"]), ("correct", ITEM["correct"]), ("Wstar", ITEM["Wstar"]),
        ("cid", _CID), ("aid", _AID), ("first_token_collision", False),
        ("topk_bare", tk_b), ("topk_neutral", tk_n), ("topk_counter", tk_c),
        ("p_c_bare", exp_p_c_bare), ("rank_c_bare", exp_rank_c_bare),
        ("p_w_bare", exp_p_w_bare), ("rank_w_bare", exp_rank_w_bare),
        ("p_c_neutral", exp_p_c_neu), ("rank_c_neutral", exp_rank_c_neu),
        ("p_w_neutral", exp_p_w_neu), ("rank_w_neutral", exp_rank_w_neu),
        ("p_c_counter", exp_p_c_ctr), ("rank_c_counter", exp_rank_c_ctr),
        ("p_w_counter", exp_p_w_ctr), ("rank_w_counter", exp_rank_w_ctr),
        ("delta_topk", exp_delta_topk), ("wstar_rank_bare", exp_rank_w_bare),
        ("top_riser", exp_top_riser), ("wstar_is_top_riser", exp_wstar_is_top_riser),
    )
    assert tuple(k for k, _ in exp_shipped) == SHIPPED_RECORD_KEYS, [k for k, _ in exp_shipped]
    for k, v in exp_shipped:
        assert rf[k] == v, (k, rf[k], v)
    # in fold, every added twin collapses onto its literal source and the two risers are the SAME row
    assert (rf["p_plant_bare"], rf["p_target_bare"]) == (rf["p_c_bare"], rf["p_w_bare"])
    assert (rf["rank_plant_bare"], rf["rank_target_bare"]) == (rf["rank_c_bare"], rf["rank_w_bare"])
    assert (rf["p_plant_neutral"], rf["p_target_neutral"]) == (rf["p_c_neutral"], rf["p_w_neutral"])
    assert (rf["rank_plant_neutral"], rf["rank_target_neutral"]) == (rf["rank_c_neutral"], rf["rank_w_neutral"])
    assert (rf["p_plant_counter"], rf["p_target_counter"]) == (rf["p_c_counter"], rf["p_w_counter"])
    assert (rf["rank_plant_counter"], rf["rank_target_counter"]) == (rf["rank_c_counter"], rf["rank_w_counter"])
    assert rf["top_riser_excl_plant"] == rf["top_riser"], rf["top_riser_excl_plant"]
    assert rf["target_is_top_riser"] is rf["wstar_is_top_riser"] is True
    assert rf["rank_target_bare"] == rf["wstar_rank_bare"]      # so no separate target_rank_bare is needed
    assert "threshold_provenance" not in rf                     # the fold arm's thresholds are the registered ones
    print(f"[selftest] FOLD PATH UNCHANGED: all {len(SHIPPED_RECORD_KEYS)} pre-existing fields equal the "
          "shipped composition recomputed independently (ranks 1/3 bare and 1/1 counter, dp order "
          "[200,5,3,4,2,1,100], top_riser tok200 dp +0.21875); in fold every twin collapses onto its literal "
          "and both risers are the same row; no threshold_provenance on fold")

    # ---------- the PRE-EXISTING fields are ARM-INVARIANT given the same measured numbers ----------
    for k in SHIPPED_RECORD_KEYS:
        assert rl[k] == rf[k], (k, rl[k], rf[k])
    print(f"[selftest] all {len(SHIPPED_RECORD_KEYS)} pre-existing fields identical between arms on identical "
          "nums: the arm changes only which prompt was measured and the ADDED keys, never a literal's meaning")

    # ---------- the BARE turn is arm-independent (its fields cannot depend on plant/target) ----------
    bare_fields = ("topk_bare", "p_c_bare", "rank_c_bare", "p_w_bare", "rank_w_bare", "wstar_rank_bare")
    for k in bare_fields:
        assert rl[k] == rf[k], (k, rl[k], rf[k])
    print(f"[selftest] bare-turn fields {bare_fields} identical between arms (BARE = single(q) contains "
          "neither plant nor target; the run caches them per item instead of measuring them twice)")

    # ---------- the TARGET twins read the TARGET, not always W* ----------
    # rank_target_* is W*'s rank in fold and C's rank in listen (the planted ranks differ, so this bites).
    assert rf["rank_c_bare"] != rf["rank_w_bare"], (rf["rank_c_bare"], rf["rank_w_bare"])   # 1 vs 3
    assert rf["rank_target_bare"] == rf["rank_w_bare"] and rf["rank_plant_bare"] == rf["rank_c_bare"]
    assert rl["rank_target_bare"] == rl["rank_c_bare"] and rl["rank_plant_bare"] == rl["rank_w_bare"]
    assert rf["p_target_bare"] == rf["p_w_bare"] and rl["p_target_bare"] == rl["p_c_bare"]
    assert rl["p_target_bare"] != rf["p_target_bare"], (rl["p_target_bare"], rf["p_target_bare"])
    # ... and target_is_top_riser follows the TARGET through the PLANT-excluding riser. With _PC the largest
    # riser overall is W*'s token (dp +0.21875) and the next is token 5 (+0.1875):
    assert rl["top_riser"] == exp_top_riser                      # literal riser: unchanged, still excludes cid
    assert rl["top_riser_excl_plant"]["tok_id"] == 5              # listen excludes the plant (= W*'s token)
    assert rl["wstar_is_top_riser"] is True and rl["target_is_top_riser"] is False
    # The mirror case: with _PC2, C's token is the largest riser and W*'s token falls.
    gf = build_record(ITEM, "fold", _planted_nums(Pc=_PC2))
    gl = build_record(ITEM, "listen", _planted_nums(Pc=_PC2))
    assert [r["tok_id"] for r in gf["delta_topk"]] == [100, 3, 4, 5, 200, 2, 1], gf["delta_topk"]
    assert gf["top_riser"]["tok_id"] == 3                        # excluding cid, the largest dp is a 0.0 tie
    assert gf["wstar_is_top_riser"] is False and gf["target_is_top_riser"] is False
    assert gl["wstar_is_top_riser"] is False, gl["wstar_is_top_riser"]
    assert gl["top_riser_excl_plant"]["tok_id"] == _CID and gl["top_riser_excl_plant"]["dp"] == 0.25
    assert gl["target_is_top_riser"] is True, gl["target_is_top_riser"]
    print("[selftest] target twins read the TARGET: rank/p_target_* = W*'s in fold and C's in listen; "
          "target_is_top_riser uses the PLANT-excluding riser (same numbers -> fold False / listen True), "
          "while the literal wstar_is_top_riser keeps the shipped cid exclusion in both arms")

    # ---------- first-token collision: RECORDED and EXCLUDED from the fractions (never dropped) ----------
    for a in ARMS:
        rc = build_record(ITEM, a, _planted_nums(cid=_CID, aid=_CID))     # cid == aid
        assert rc["first_token_collision"] is True, a
        assert (rc["plant_id"], rc["target_id"]) == (_CID, _CID), a       # plant_id == target_id, both arms
        assert rc["p_c_bare"] == rc["p_w_bare"] == round(_PB[_CID], 6), a  # still fully measured + dumped
        assert rc["delta_topk"] and rc["top_riser"] is not None, a
        blk = arm_block([rc], a)
        assert blk["aggregate"]["n"] == 1 and blk["aggregate"]["n_collision"] == 1, a
        assert blk["aggregate"]["n_eval"] == 0, a                          # excluded from the fractions
        assert blk["aggregate"]["frac_wstar_top_riser"] is None, a
        assert blk["aggregate_target"]["frac_target_top_riser"] is None, a
        assert blk["decision"]["category"] == "UNDEFINED", a
        assert blk["decision_target"]["category"] == "UNDEFINED", a
    # a collision item alongside two clean ones: n_eval counts only the non-collision items, in BOTH blocks.
    mixed = [build_record(ITEM, "fold", _planted_nums()),
             build_record(ITEM, "fold", _planted_nums(Pc=_PC2)),
             build_record(ITEM, "fold", _planted_nums(cid=_CID, aid=_CID))]
    bm = arm_block(mixed, "fold")
    assert bm["aggregate"]["n"] == 3 and bm["aggregate"]["n_collision"] == 1 and bm["aggregate"]["n_eval"] == 2
    assert bm["aggregate"]["frac_wstar_top_riser"] == 0.5        # 1 of the 2 non-collision items
    assert bm["aggregate_target"]["frac_target_top_riser"] == 0.5
    assert bm["aggregate"]["n_wstar_top_riser"] == 1 and bm["aggregate_target"]["n_target_top_riser"] == 1
    print("[selftest] first-token collision: first_token_collision=True, item fully measured + dumped, and "
          "EXCLUDED from n_eval / both fractions (all-collision -> UNDEFINED; 1 of 3 -> n_eval 2, frac 0.5)")

    # ---------- the two aggregate blocks agree EXACTLY in fold, and are keyed apart ----------
    bf = arm_block([rf, gf], "fold")
    assert bf["aggregate"]["frac_wstar_top_riser"] == bf["aggregate_target"]["frac_target_top_riser"]
    assert bf["aggregate"]["median_wstar_rank_bare"] == bf["aggregate_target"]["median_target_rank_bare"]
    assert bf["aggregate"]["wstar_in_bare_topk"] == bf["aggregate_target"]["target_in_bare_topk"]
    assert bf["decision"]["category"] == bf["decision_target"]["category"]
    assert (bf["plant_role"], bf["target_role"]) == ("correct", "Wstar")
    assert "threshold_provenance" not in bf and "threshold_provenance" not in bf["aggregate"]
    # ... and in listen they can differ, with the note attached everywhere it is owed.
    bl = arm_block([rl, gl], "listen")
    assert (bl["plant_role"], bl["target_role"]) == ("Wstar", "correct")
    assert bl["aggregate"]["frac_wstar_top_riser"] == 0.5           # rl True, gl False (literal flag)
    assert bl["aggregate_target"]["frac_target_top_riser"] == 0.5   # rl False, gl True (target flag)
    assert bl["threshold_provenance"] == THRESHOLD_PROVENANCE
    assert bl["aggregate"]["threshold_provenance"] == THRESHOLD_PROVENANCE
    assert bl["aggregate_target"]["threshold_provenance"] == THRESHOLD_PROVENANCE
    assert rl["threshold_provenance"] == THRESHOLD_PROVENANCE and rl["stamp"]["arm"] == "listen"
    # the target block is a RENAME of the imported aggregate's keys, nothing more
    assert set(TARGET_KEY_RENAME) <= set(aggregate(target_view([rl, gl])))
    assert not (set(TARGET_KEY_RENAME) & set(bl["aggregate_target"]))
    assert relabel_target({"frac_wstar_top_riser": 0.25, "category": "X"}) == {"frac_target_top_riser": 0.25,
                                                                              "category": "X"}
    print("[selftest] per-arm blocks: fold's LITERAL and TARGET aggregates agree exactly; listen carries "
          f"threshold_provenance={THRESHOLD_PROVENANCE!r} on the block and on BOTH aggregates; the target "
          "block is the imported aggregate with four keys renamed")

    # ---------- per-arm split: an arm's aggregate sees ONLY its own records ----------
    pool = [build_record(ITEM, a, _planted_nums()) for a in ARMS for _ in range(3)]
    assert len(records_for_arm(pool, "fold")) == 3 and len(records_for_arm(pool, "listen")) == 3
    assert arm_block(pool, "fold")["aggregate"]["n"] == 3, "arms must not be pooled"
    assert arm_block(pool, "listen")["aggregate"]["n"] == 3, "arms must not be pooled"
    assert aggregate(pool)["n"] == 6                            # pooling only if something explicitly asks
    print("[selftest] records_for_arm splits by the record's own `arm`; per-arm aggregates never pool arms")

    # ---------- every record carries the five-key stamp ----------
    for a in ARMS:
        r = build_record(ITEM, a, _planted_nums())
        assert tuple(r["stamp"]) == STAMP_KEYS, (a, tuple(r["stamp"]))
        assert set(r["stamp"]) == set(STAMP_KEYS) and len(r["stamp"]) == 5, a
        assert r["stamp"]["arm"] == a == r["arm"], (a, r["stamp"]["arm"], r["arm"])
        assert all(isinstance(v, str) and v.strip() for v in r["stamp"].values()), a
        plant_role, target_role = arm_roles(a, "correct", "Wstar")
        assert plant_role in r["stamp"]["slot"] and target_role in r["stamp"]["slot"], a
        assert "first_token_collision" in r["stamp"]["tiebreak"], a
    print(f"[selftest] stamp: exactly the five keys {STAMP_KEYS} on every record, arm-consistent")

    # ---------- decide: the IMPORTED categories + inclusive boundaries, applied per arm unchanged ----------
    assert decide(0.75)["category"] == "TARGETED_SHIFT"
    assert decide(FRAC_HI)["category"] == "TARGETED_SHIFT"      # 0.5 boundary inclusive (>=)
    assert decide(0.10)["category"] == "OTHER_RISER"
    assert decide(FRAC_LO)["category"] == "OTHER_RISER"         # 0.2 boundary inclusive (<=)
    assert decide(0.35)["category"] == "MIXED"
    assert decide(FRAC_HI - 1e-9)["category"] == "MIXED" and decide(FRAC_LO + 1e-9)["category"] == "MIXED"
    assert decide(None)["category"] == "UNDEFINED"              # n_eval == 0
    assert relabel_target(decide(0.75))["frac_target_top_riser"] == 0.75
    print(f"[selftest] decide: TARGETED_SHIFT >= FRAC_HI({FRAC_HI}) / OTHER_RISER <= FRAC_LO({FRAC_LO}) / "
          "MIXED (boundaries inclusive) -- SAME rule and SAME thresholds applied per arm and per block")

    print("[selftest] PASS")
    print("SELFTEST PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--family", default="verifier_family",
                   help="'verifier_family' (the module's ITEMS) OR a path to a JSON list of {q,correct,Wstar,...}")
    p.add_argument("--name", default="google/gemma-2-9b", help="model (base is the QA template; -it via --chat)")
    p.add_argument("--tag", default="vfam_9b")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--chat", action="store_true",
                   help="use the chat template (-it model; qa template otherwise; base is primary)")
    p.add_argument("--arm", default="fold", choices=["fold", "listen", "both"],
                   help="fold: plant=C, target=W* (the shipped direction). listen: plant=W*, target=C. "
                        "both: both arms in ONE model load, fold first, tagged by `arm` in items.")
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        run(args.family, args.name, args.tag, args.device, args.chat, args.arm)


if __name__ == "__main__":
    main()
