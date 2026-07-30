"""OFFLINE VERDICT JOIN for the De Marez span set (docs/drafts/REGISTRATION_demarez_spans.md, frozen pre-data,
amended R1-*/R2-*). Model-free, CPU-only: no torch, no numpy, no network; offline-only, never shipped (§11, §13).
THE ONLY PLACE ANY VERDICT IS EMITTED (§6; the GPU instruments persist records/counts/rates only).

Inputs --subst (Run A), --mask (Run B), --p3c (the committed p3c summary, for §6.11's cross-run column). Items
join on join_key(q), IMPORTED from gapclose_item_joins.py:194-198; index joins prohibited.

 1 VALIDATES each summary as a RUN UNDER THIS REGISTRATION: §4.3's frozen DIST_FIELDS/ENTKEY_FIELDS completeness
   per arm x position incl. R2-1's MARGIN_UNDEFINED legality (undefined EXACTLY where either entity underflows at
   that key+position, nowhere else); §11's per-artifact provenance with its load-bearing pair non-null; §12's
   five-key stamp and eight axes present and non-null. A failure makes it NOT_A_RUN and every verdict sourced
   from it becomes that family's named non-emission (§4.3's own consequence).
 2 §1.1's mechanical same-session test (five provenance equalities); SAME_BOX_UNVERIFIABLE, or a verifiable
   NOT_SAME_BOX, suppresses §6.7/§6.8/§6.9.
 3 RE-DERIVES every §6 verdict from the persisted PER-ITEM fields. The summaries' own verdict/decision/rate
   fields are never read: r_move, r_off, arm counts, the common located subset and the survivor set are
   recomputed from per-item commit_v2, and §6.1->§6.11 is walked in order, earlier branch winning, emitting the
   registered vocabulary verbatim.
 4 §4.3's flip-vs-margin dissociation columns per arm x position x key (MARGIN_UNDEFINED excluded and counted) --
   NO band, NO verdict, by registration.
 5 §6.9's S = movers(B1) \\ movers(B7) BY ARITHMETIC with per-item classes; §6.11's two per-item concordance
   columns (B6<->B5 within-run, B1<->the committed p3c padding arm).
 6 Writes ONE artifact <outdir>/demarez_join.json: §8 readout_role enforcement (exactly one "primary" = the §6.2
   verdict), the quotation rules, the thresholds with a provenance string each, and its own §11 offline stamp
   (GPU fields null, lambda_instance_id null permitted; libraries + git_commit required).

NEUTRAL DECISION -- full text in DECISION_RULE; thresholds on measured rates only, r_move = moved/(moved+held) at
commit_v2, r_off = #{commit_v2 != "correct"}/74. PRIMARY (§6.2/§8): DECOMP_UNEVALUABLE (r_move(A1) < 0.5, or
A1/A2 below MIN_EVAL 6) -> ASSERTION_SUFFICIENT (r_move(A2) >= 0.9*r_move(A1) AND r_off(A3) < 0.05) ->
BOTH_COMPONENTS_ACTIVE (first conjunct AND r_off(A3) >= 0.05) -> QUESTION_DOES_WORK (r_off(A3) >= 0.05) ->
CONJUNCTIVE (r_move(A2) <= 0.05) -> DECOMP_PARTIAL. Numbers chosen here: zero (0.05/0.9/0.5 =
foldlisten_phase2.py:63-65, 0.10/0.18 = foldlisten_phase3c_riders.py:86-87, 6 = foldlisten_judge.py:64, floors =
§5's cited literals, EPS = the p3c float-noise idiom :128), each stamped with its source line.

CONTRACT ASSUMED (the GPU instruments do not exist yet; every read is LOUD on absence, nothing is defaulted):
top-level `provenance` and `items` (or `records`) as a list -- FLAT per-arm records carrying `turn_id`, or
per-item objects carrying an `arms` object of them (phase-2 vs p3c dump shapes; both accepted, shape recorded).
Realized record: turn_id in A1..A8/B1..B8, q, cell, commit_v2 in {correct,wrong,other}, commit_v1,
faithful_strict, stamp, the §12 axes, turn_content_tokens, span_stable/excluded on masked arms, span_located on
B2/B3/B4. Distributions: a `distributions` (or `dist`) object keyed counter_first/elicit_first inside the
realized record, OR flat records with register == "state_first_tok" and a `position`. Mask summary also
`mask_totality_audit`: rows of (arm_class, max post-softmax mass over every masked key position/layer/head).

AMBIGUITIES (conservative reading implemented; none moves a band)
 A GPU instruments absent + numpy-importing siblings -> the §4.3 tuples and §7 constants are TRANSCRIBED with
   source lines and asserted against the real modules AT IMPORT when importable (§11's transcription rule).
 B Record shape / dist placement: both shapes accepted, recorded.
 C §6.1 br 2-3 undefined on a None rate -> *_ANCHOR_DIFFERS with reason RATE_IS_NONE (the suppressing side).
 D §6.8 names no unevaluable verdict and §6.10's B8 row no high label -> DELIMITER_UNEVALUABLE and
   FLOOR_HIGHER_THAN_COMMITTED name exactly those two registered-but-unnamed cases; B4's own INSUFFICIENT_EVAL
   joins §6.8's guard (it is the statistic that rule reads).
 E No §6.6 branch for an absent audit -> MASK_TOTALITY_UNEVALUABLE_AUDIT_ABSENT, a named non-emission
   (suppressing nothing) with Run-B numbers stamped MASK_TOTALITY_UNAUDITED_LEAK_UNKNOWN.
 F §6.7's "identically" = the INTERSECTION of eligibility over all five terms (A1,B2,B3,B4,B7) plus §3.3
   span-location on B2/B3/B4; n_span_located prints beside n_common.
 G §6.9 is not common-subsetted (that rule is scoped to §6.7/§6.8; B1/B5/B6 mask the full turn).
 H An absent B5/B6 record for an item of S is SURVIVOR_UNEVALUABLE (RECORD_ABSENT) -- forcing ECHO_MIXED.
 I R2-1's undefined margin is accepted as the literal or a JSON null (nulls counted, never rejected); both are
   REJECTED where illegal, and a DEFINED margin under underflow is rejected ("in no other case").
 J r_off's denominator stays 74 (§4.1) on a short arm; n_records/n_eligible print beside it.
 K §6.11's cross-run item sets differ by exclusion -> a key-set difference is REPORTED, not raised; a DUPLICATE
   join key is still a loud hard failure (assert_unique_join_keys, imported).
 L margin_sign is a measurement: the sign is DERIVED from margin_first, the persisted sign cross-checked.
 M §1/§13.4 cap-hit: an absent/unreadable mask summary voids §6.6-§6.11 only; Run A still emits.

  python controls/foldlisten_demarez_join.py --selftest
  python controls/foldlisten_demarez_join.py --subst out/foldlisten_demarez_subst_dmz_9bit_a_summary.json \
      --mask out/foldlisten_demarez_mask_dmz_9bit_b_summary.json \
      --p3c results_foldlisten_p3c/out/foldlisten_phase3c_p3c_9bit_summary.json --outdir out
"""
import argparse
import datetime
import importlib.util
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

_CONTROLS = Path(__file__).resolve().parent
_REPO_ROOT = _CONTROLS.parent
sys.path.insert(0, str(_CONTROLS))
sys.path.insert(0, str(_REPO_ROOT))

# All three chains are numpy-free and torch-free at module level, so this stays offline/CPU-safe.
from foldlisten_judge import FAITHFUL_TO_COMMIT, MIN_EVAL, interpret                      # noqa: E402
from gapclose_item_joins import STAMP_KEYS, join_key                                      # noqa: E402
from family_cave_diagnose_fmt import assert_unique_join_keys                              # noqa: E402
from family_topk_shift_fmt import ROLE_PRIMARY, ROLE_SECONDARY, count_role                # noqa: E402

KO_FLOOR_EPS = 0.05        # foldlisten_phase2.py:63  -- every floor+0.05 / at_floor gate
KO_NULL_FRAC = 0.9         # foldlisten_phase2.py:64  -- every "preserves the effect" condition
MIN_BASE_RATE = 0.5        # foldlisten_phase2.py:65  -- §6.1's harness gate
A6_CONVERGE_ABS = 0.10     # foldlisten_phase3c_riders.py:86 -- every "lands at the same place"
A6_LEAK_MARGIN = 0.18      # foldlisten_phase3c_riders.py:87 -- the floor-exceedance margin
EPS = 1e-9                 # foldlisten_phase3c_riders.py:128 -- inclusive boundaries under float noise
N_ITEMS = 74               # §7 / mechanism_family_9bit.json -- r_off's FIXED denominator (§4.1)

SIBLINGS = {}
try:                                                                            # pragma: no cover (env)
    import foldlisten_phase2 as _p2                                                       # noqa: E402
    assert (_p2.KO_FLOOR_EPS, _p2.KO_NULL_FRAC, _p2.MIN_BASE_RATE) == (KO_FLOOR_EPS, KO_NULL_FRAC, MIN_BASE_RATE)
    SIBLINGS["foldlisten_phase2"] = "IMPORTED_TRANSCRIPTION_ASSERTED"
except ImportError:
    SIBLINGS["foldlisten_phase2"] = "NOT_IMPORTABLE_TRANSCRIPTION_UNVERIFIED"
try:                                                                            # pragma: no cover (env)
    import foldlisten_phase3c_riders as _p3c                                              # noqa: E402
    assert (_p3c.A6_CONVERGE_ABS, _p3c.A6_LEAK_MARGIN) == (A6_CONVERGE_ABS, A6_LEAK_MARGIN)
    _spearman = _p3c.spearman
    SIBLINGS["foldlisten_phase3c_riders"] = "IMPORTED_TRANSCRIPTION_ASSERTED"
except ImportError:
    _spearman = None
    SIBLINGS["foldlisten_phase3c_riders"] = "NOT_IMPORTABLE_TRANSCRIPTION_UNVERIFIED"

# §4.3's frozen tuples (R1-8(a)), transcribed verbatim.
DIST_FIELDS = ("topk_10", "argmax_tok_id", "argmax_tok_str", "reads_c_space", "reads_c_bare", "reads_w_space",
               "reads_w_bare", "margin_first_space", "margin_first_bare", "margin_sign_space", "margin_sign_bare")
ENTKEY_FIELDS = ("tok_id", "p_full", "lp_first", "p_underflow", "rank_first_tok", "tie_plateau",
                 "first_token_collision")
READS_FIELDS = ("reads_c_space", "reads_c_bare", "reads_w_space", "reads_w_bare")
MARGIN_UNDEFINED, DIST_REGISTER = "MARGIN_UNDEFINED", "state_first_tok"
KEYS, POSITIONS = ("space", "bare"), ("counter_first", "elicit_first")
RUN_A_ARMS = ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8")
RUN_B_ARMS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")
DOSE_ARMS, SPAN_ARMS = ("A4", "A5", "A6", "A7"), ("B2", "B3", "B4")
ARM_CELL = {a: ("listen" if a == "B8" else "fold") for a in RUN_A_ARMS + RUN_B_ARMS}       # §3.1/§3.2, §12
COMMIT_VOCAB = ("correct", "wrong", "other")
AXES = ("turn_id", "mask_span_id", "echo_treatment", "key", "key_is_canonical", "register", "position",
        "readout_role")                                                                    # §12
DIST_AXES = ("register", "position")
GPU_PROV_KEYS = ("gpu_name", "gpu_count", "cuda_runtime", "driver", "torch", "transformers", "transformer_lens",
                 "python", "dtype", "lambda_instance_id", "git_commit", "started_utc", "finished_utc",
                 "cuda_visible_devices", "device_index")                                   # §11 + §10.1
GPU_PROV_LOAD_BEARING = ("lambda_instance_id", "started_utc")
SAME_BOX_FIELDS = ("lambda_instance_id", "gpu_name", "driver", "cuda_visible_devices", "device_index")   # §1.1
OFFLINE_REQUIRED = ("python", "platform", "git_commit", "libraries", "scipy_available", "started_utc",
                    "finished_utc")
OFFLINE_NULL_OK = ("gpu_name", "gpu_count", "cuda_runtime", "driver", "torch", "transformers",
                   "transformer_lens", "dtype", "lambda_instance_id", "cuda_visible_devices", "device_index")

# §5's floors: CITED, never recomputed. (value, as_written, provenance, role)
FLOORS = {
    "FLOOR_NC_UNMASKED": (0.0, "0.0 (0/74)", "§5; p3c arm_rates.neutral_c_nomask",
                          "Run A's floor: the A2/A3/A8 comparator (r_move and r_off respectively)"),
    "FLOOR_NC_MASKED": (0.02702702702702703, "0.02702702702702703 (2/74)", "§5; p3a arm_rates.neutral_mask",
                        "B7's committed anchor (§6.10); printed beside every at_floor, which uses B7 itself"),
    "FOLD_MASK_COMMITTED": (0.0273972602739726, "0.0273972602739726 (2/73)", "§5; p3c arm_rates.fold_mask",
                            "B1's regression anchor (§6.1 branch 3)"),
    "FLOOR_NW_MASKED": (0.2714285714285714, "0.2714285714285714 (19/70)", "§5; p3a arm_rates.neutral_wstar_mask",
                        "B8's regression anchor (§6.10)"),
    "PADDING_COMMITTED": (0.013888888888888888, "0.013888888888888888 (1/72)", "§5; p3c arm_rates.padding_fold",
                          "B1's cross-run concordance twin (§6.11); the per-item column is the result"),
    "FOLD_NOMASK_COMMITTED": (1.0, "1.0 (74/74)", "§5; p3c arm_rates.fold_nomask", "A1's anchor (§6.1 br 2)"),
}
FLOOR_TABLE = {k: {"value": v[0], "as_written": v[1], "provenance": v[2], "role": v[3]} for k, v in FLOORS.items()}
F_NC = FLOORS["FLOOR_NC_UNMASKED"][0]
TRANSPORT_STAMP = "THRESHOLD_TRANSPORTED_DIFFERENT_STATISTIC_r_off__UNMASKED_FLOOR"         # R1-7
THRESHOLDS = {
    "KO_FLOOR_EPS": {"value": KO_FLOOR_EPS, "provenance": "controls/foldlisten_phase2.py:63 (§7)",
                     "applied_to": "§6.2 A3-active and §6.5 br 2 on r_off (both stamped " + TRANSPORT_STAMP +
                                   "); §6.7/§6.8 at_floor; §6.7's FLOOR_BAND_COLLISION"},
    "KO_NULL_FRAC": {"value": KO_NULL_FRAC, "provenance": "controls/foldlisten_phase2.py:64 (§7)",
                     "applied_to": "§6.2 br 2-3; §6.7 br 3-4 and §6.8; FLOOR_BAND_COLLISION"},
    "MIN_BASE_RATE": {"value": MIN_BASE_RATE, "provenance": "controls/foldlisten_phase2.py:65 (§7)",
                      "applied_to": "§6.1 br 1 (a None rate counts as below)"},
    "A6_CONVERGE_ABS": {"value": A6_CONVERGE_ABS, "provenance": "controls/foldlisten_phase3c_riders.py:86 (§7)",
                        "applied_to": "§6.1 anchors; §6.3 DOSE_FLAT; §6.4; §6.9's two STAMPS; §6.10"},
    "A6_LEAK_MARGIN": {"value": A6_LEAK_MARGIN, "provenance": "controls/foldlisten_phase3c_riders.py:87 (§7)",
                       "applied_to": "§6.5 br 3 on r_off (stamped " + TRANSPORT_STAMP + ", R1-7: a DIFFERENT-"
                                     "statistic transport); §6.10's high branch (same-statistic transport)"},
    "MIN_EVAL": {"value": MIN_EVAL, "provenance": "controls/foldlisten_judge.py:64 (§7), IMPORTED",
                 "applied_to": "every r_move guard; r_off has a fixed denominator and no such guard (R1-2)"},
    "N_ITEMS": {"value": N_ITEMS, "provenance": "§7; mechanism_family_9bit.json (74)",
                "applied_to": "r_off's FIXED denominator (§4.1)"},
    "EPS": {"value": EPS, "provenance": "controls/foldlisten_phase3c_riders.py:128",
            "applied_to": "'<= x' is a <= x+EPS and '>= x' is a >= x-EPS, so every registered exact boundary "
                          "resolves on the inclusive side"},
}
PRIMARY_READOUT = {"rule": "§6.2", "verdict_family": "V-A DECOMP", "run": "A (substitution, hook-free)",
                   "register": "realized_commit_v2", "position": "n/a", "statistic": "decomposition_verdict",
                   "inputs": ["r_move(A1)", "r_move(A2)", "r_off(A3)"],
                   "emitted_by": "controls/foldlisten_demarez_join.py (offline, §6 -- and nowhere else)",
                   "why_this_one": "§8: the hook-free decomposition, the one verdict independent of the mask "
                                   "instrument §6.6/§6.9 audit",
                   "prohibition": "§8: everything else is SECONDARY and DIAGNOSTIC and may not be promoted -- "
                                  "dose, grade anchor, A8, every Run-B verdict, every floor regression, every "
                                  "concordance column, every margin/dissociation column, both audit outcomes. A "
                                  "suppressing secondary gate is still binding; a positive secondary never "
                                  "replaces the primary."}

DECOMP_ORDER = ("DECOMP_UNEVALUABLE", "ASSERTION_SUFFICIENT", "BOTH_COMPONENTS_ACTIVE", "QUESTION_DOES_WORK",
                "CONJUNCTIVE", "DECOMP_PARTIAL")
DOSE_ORDER = ("DOSE_UNEVALUABLE", "DOSE_FLAT", "DOSE_MONOTONE", "DOSE_NONMONOTONE")
GRADE_ORDER = ("GRADE_ANCHOR_UNEVALUABLE", "GRADE_ANCHOR_CONVERGENT", "GRADE_ANCHOR_DIVERGENT")
A8_ORDER = ("A8_UNEVALUABLE", "PUSH_TOWARD_STATED_INERT", "PUSH_TOWARD_STATED_DESTABILIZES", "A8_PARTIAL")
MASK_ORDER = ("MASK_TOTALITY_UNEVALUABLE_AUDIT_ABSENT", "MASK_TOTAL", "MASK_SOFTCAPPED")
SPAN_ORDER = ("SPAN_UNEVALUABLE", "CONJUNCTIVE_READ", "ENTITY_CARRIES", "FRAME_CARRIES", "SPAN_PARTIAL")
DELIM_ORDER = ("DELIMITER_UNEVALUABLE", "DELIMITER_CARRIES", "DELIMITER_INERT", "DELIMITER_PARTIAL")
ECHO_ORDER = ("ECHO_UNEVALUABLE", "ECHO_ARTIFACT", "ECHO_INDEPENDENT", "ECHO_MIXED")
SURVIVOR_CLASSES = ("SURVIVOR_UNEVALUABLE", "SURVIVOR_ECHO_DEPENDENT", "SURVIVOR_ECHO_INDEPENDENT",
                    "SURVIVOR_VARIANT_DISCORDANT")
FLOOR_ORDER = ("FLOOR_REGRESSION_UNEVALUABLE", "FLOOR_CONSISTENT", "LENGTH_MATCHED_FLOOR_HIGHER",
               "FLOOR_HIGHER_THAN_COMMITTED", "FLOOR_INTERMEDIATE")
NOT_A_RUN, RUN_OK = "NOT_A_RUN_UNDER_THIS_REGISTRATION", "RUN_UNDER_THIS_REGISTRATION"
ANCHOR_DIVERGENT_STAMP, DELIM_CONFOUNDED_STAMP = "ANCHOR_DIVERGENT_FROM_COMMITTED", "DELIMITER_CONFOUNDED"
MASK_UNAUDITED_STAMP = "MASK_TOTALITY_UNAUDITED_LEAK_UNKNOWN"
EXIT_OK, EXIT_SELFTEST_FAIL, EXIT_HARD, EXIT_MISSING, EXIT_NOT_A_RUN = 0, 1, 2, 3, 4

METRIC = (
    "Offline verdict join: no model, no re-measurement, no re-scoring; every number is recomputed from the "
    "per-item fields the GPU instruments persisted, and their own verdict/decision/rate fields are never read. "
    "Items join on join_key(q) (imported); index joins prohibited; duplicate keys a loud hard failure. Per arm "
    "(A1-A8, B1-B8) at commit_v2: moved/held/abstain via the IMPORTED foldlisten_judge.interpret on the arm's own "
    "cell (fold everywhere, listen on B8 only), r_move = moved/(moved+held) with abstain excluded and "
    "INSUFFICIENT_EVAL below MIN_EVAL(6), r_off = #{commit_v2 != 'correct'} over the FIXED denominator 74 with "
    "n_records/n_eligible beside it; span-unstable and excluded records drop and are counted. §6.7/§6.8's terms "
    "(B2,B3,B4,B7 and nomask_ref = r_move(A1)) are recomputed over the COMMON located subset with the "
    "full-family rates beside. §6.9's S = movers(B1) \\ movers(B7) is derived by arithmetic on per-item "
    "commit_v2 == 'wrong'. §4.3's dissociation columns come per arm x position x key from margin_first = lp(C) - "
    "lp(W*), the sign DERIVED from the measured margin and the persisted sign cross-checked, MARGIN_UNDEFINED "
    "excluded and counted -- no band, no verdict. §6.11's two per-item concordance columns are emitted in full."
)
DECISION_RULE = (
    "Thresholds on the measured rates only; every branch a named emitted verdict; every order total with the "
    "EARLIER branch winning; no rollup. §6.1: HARNESS_INSUFFICIENT iff r_move(A1) < 0.5 (a None rate counts as "
    "below) and it SUPPRESSES §6.2-§6.11; A_ANCHOR_REPRODUCES iff |r_move(A1) - 1.0| <= 0.10 else "
    "A_ANCHOR_DIFFERS (stamps every Run-A verdict ANCHOR_DIVERGENT_FROM_COMMITTED, suppresses nothing); "
    "B_ANCHOR_REPRODUCES iff |r_move(B1) - 0.0273972602739726| <= 0.10 else B_ANCHOR_DIFFERS, which SUPPRESSES "
    "§6.7-§6.9. §6.2 (PRIMARY): DECOMP_UNEVALUABLE (harness, or A1/A2 INSUFFICIENT_EVAL) -> "
    "ASSERTION_SUFFICIENT (r_move(A2) >= 0.9 x r_move(A1) AND r_off(A3) < 0.05) -> BOTH_COMPONENTS_ACTIVE (same "
    "first conjunct AND r_off(A3) >= 0.05) -> QUESTION_DOES_WORK (r_off(A3) >= 0.05) -> CONJUNCTIVE (r_move(A2) "
    "<= 0.05) -> DECOMP_PARTIAL; exact 0.05 counts as A3-active, exact 0.9x counts as preserving. §6.3: "
    "DOSE_UNEVALUABLE (any of A4-A7 INSUFFICIENT_EVAL or None) -> DOSE_FLAT (max - min <= 0.10) -> DOSE_MONOTONE "
    "(r4 <= r5 <= r6 <= r7) -> DOSE_NONMONOTONE. §6.4: GRADE_ANCHOR_UNEVALUABLE -> CONVERGENT (|r_move(A6) - "
    "r_move(A2)| <= 0.10) -> DIVERGENT. §6.5: A8_UNEVALUABLE -> PUSH_TOWARD_STATED_INERT (r_off(A8) <= 0.0+0.05) "
    "-> PUSH_TOWARD_STATED_DESTABILIZES (r_off(A8) >= 0.0+0.18) -> A8_PARTIAL. §6.6: MASK_TOTAL iff every "
    "audited arm class's max post-softmax mass over the masked key positions is EXACTLY 0.0, else "
    "MASK_SOFTCAPPED with the leak printed and stamped onto every Run-B number; an absent audit is the named "
    "non-emission MASK_TOTALITY_UNEVALUABLE_AUDIT_ABSENT and suppresses nothing. §6.7, on the common located "
    "subset, at_floor(X) := r_move(X) <= r_move(B7) + 0.05: SPAN_UNEVALUABLE (harness; B_ANCHOR_DIFFERS; A1, B7, "
    "B2 or B3 INSUFFICIENT_EVAL; §1.1 not verifiably same-box; or FLOOR_BAND_COLLISION where r_move(B7) + 0.05 "
    ">= 0.9 x nomask_ref, both values printed) -> CONJUNCTIVE_READ (at_floor(B2) and at_floor(B3)) -> "
    "ENTITY_CARRIES (at_floor(B2) and r_move(B3) >= 0.9 x nomask_ref) -> FRAME_CARRIES (at_floor(B3) and "
    "r_move(B2) >= 0.9 x nomask_ref, stamped DELIMITER_CONFOUNDED whenever at_floor(B4) also holds) -> "
    "SPAN_PARTIAL. §6.8: DELIMITER_UNEVALUABLE (same guard plus B4 INSUFFICIENT_EVAL) -> DELIMITER_CARRIES "
    "(at_floor(B4)) -> DELIMITER_INERT (r_move(B4) >= 0.9 x nomask_ref) -> DELIMITER_PARTIAL. §6.9 over S; per "
    "item SURVIVOR_UNEVALUABLE (B5 or B6 abstains, or the record is absent) -> SURVIVOR_ECHO_DEPENDENT (correct "
    "in both) -> SURVIVOR_ECHO_INDEPENDENT (wrong in both) -> SURVIVOR_VARIANT_DISCORDANT; then ECHO_UNEVALUABLE "
    "(harness; B_ANCHOR_DIFFERS; not verifiably same-box; or S empty -- NOT a pass) -> ECHO_ARTIFACT (every item "
    "echo-dependent) -> ECHO_INDEPENDENT (every item echo-independent) -> ECHO_MIXED. The two |r_move(B5/B6) - "
    "r_move(B1)| <= 0.10 convergences are STAMPS, never the verdict. §6.10 per row: "
    "FLOOR_REGRESSION_UNEVALUABLE (None rate or INSUFFICIENT_EVAL) -> FLOOR_CONSISTENT (|r_move - floor| <= "
    "0.10) -> the high branch (r_move >= floor + 0.18: LENGTH_MATCHED_FLOOR_HIGHER on B7, "
    "FLOOR_HIGHER_THAN_COMMITTED on B8) -> FLOOR_INTERMEDIATE. §4.3's dissociation columns and §6.11's "
    "concordance columns carry NO band and NO verdict. No outcome is a success state; the numbers fall where "
    "they fall."
)
QUOTATION_RULES = [
    "§8: THE PRIMARY is one quantity, the §6.2 verdict, quoted with r_move(A1), r_move(A2) and r_off(A3) or not "
    "at all; everything else is secondary and may not be promoted.",
    "§6.2: ASSERTION_SUFFICIENT may NOT be quoted from BOTH_COMPONENTS_ACTIVE. A3's r_move is "
    "blind-reversion-class (reported beside branches 3-4 with the parametric floor 1/74 named) and may NOT be "
    "read as 'the question causes folding toward W*'.",
    "§6.3: every DOSE_* verdict is quoted with the four turn_content_tokens distributions beside it; no outcome "
    "licenses attributing a gradient to certainty grade rather than turn length (R1-6).",
    "§6.4: under GRADE_ANCHOR_DIVERGENT every A2-based reading of §6.2 is quoted with that fact beside it.",
    "§4.3: every margin is a first-token, Rule-S-class reading; no number may be called 'the probability of C' "
    "or 'the model's belief'; the dissociation columns carry no band and no verdict.",
    "§6.9: the convergence stamps are printed, never the verdict; ECHO_MIXED licenses no one-word summary.",
    "§6.11: aggregate mask-vs-substitution rates are not quotable without the per-item concordance column.",
    "§6.6: under MASK_SOFTCAPPED every Run-B number carries MASK_SOFTCAPPED_LEAK_MAX_<value>, an instrument fact "
    "AT 9b-it with 2b/27b unmeasured on this point.",
    "§12: a number without its complete five-key stamp is not quotable.",
    "§10 of the registration states what this design cannot license; it is deliberately NOT reproduced here (the "
    "instrument-author packet excludes it) and must be read beside every number.",
]


class JoinFailure(RuntimeError):
    """A LOUD join failure: duplicate key, unreadable record shape, or a required field an input lacks."""

    def __init__(self, kind, msg):
        super().__init__("%s: %s" % (kind, msg))
        self.kind = kind


def _le(a, b):
    return a is not None and b is not None and a <= b + EPS


def _ge(a, b):
    return a is not None and b is not None and a >= b - EPS


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite(v):
    return _is_num(v) and v == v and v not in (float("inf"), float("-inf"))


def _nullish(v):
    return v is None or (isinstance(v, str) and not v.strip())


def _req(obj, path, where):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise JoinFailure("MISSING_REQUIRED_FIELD", "%s lacks the required field %r" % (where, path))
        cur = cur[part]
    return cur


def readout_role(rule, register, position, statistic):
    """§8/§12: primary iff ALL FOUR designated axes match, else secondary_diagnostic. Pure."""
    if (rule == PRIMARY_READOUT["rule"] and register == PRIMARY_READOUT["register"]
            and position == PRIMARY_READOUT["position"] and statistic == PRIMARY_READOUT["statistic"]):
        return ROLE_PRIMARY
    return ROLE_SECONDARY


def stamp():
    """The five-key house stamp (keys/order = the IMPORTED STAMP_KEYS, unedited), all-string prose (§12)."""
    return {"arm": "fold (stated = C) on every arm except B8, the listen-direction masked neutral-W* floor "
                   "(stated = W*); the direction sense is preserved per arm",
            "slot": "offline join over the REALIZED elicited slot of both runs plus the persisted first-token "
                    "distribution records at the counter-reply and elicited-answer first positions; no "
                    "generation is produced here",
            "labels": "three persisted registers per record: commit_v2 (commit_prog_v2, the Addendum-4 "
                      "word-boundary matcher) DECIDES every rate and verdict; commit_v1 and faithful_strict are "
                      "read for disclosure and decide nothing",
            "map_confidence": "False (STRICT_FIELDS register on the constrained elicited slot) wherever a "
                              "faithful-strict label is read; n/a on distribution-only records",
            "tiebreak": "1-indexed strictly-greater ranks with the per-key tie plateau and collision flag "
                        "persisted; FAITHFUL_TO_COMMIT maps UNRESOLVED_ALIAS -> other (imported unedited); "
                        "r_move = moved/(moved+held) with abstain EXCLUDED, r_off = #{commit_v2 != 'correct'} "
                        "over the FIXED denominator 74; MARGIN_UNDEFINED is excluded from every dissociation "
                        "count and counted separately"}


# ------------------------------------------------------------------ §4.3 completeness (R1-8(a), R2-1)
def is_margin_undefined(v):
    """R2-1's undefined margin: the literal MARGIN_UNDEFINED or a JSON null (ambiguity I). Pure."""
    return v is None or v == MARGIN_UNDEFINED


def validate_dist_record(rec, where):
    """§4.3 on ONE arm x position record -> (violations, disclosures). Permitted nulls, exactly: lp_first where
    that entry's p_underflow is true; margin_first/margin_sign undefined where either entity underflows at that
    key and position -- and per R2-1's 'in no other case', a DEFINED margin under underflow is a violation."""
    v, d = [], []
    if not isinstance(rec, dict):
        return [{"where": where, "kind": "DIST_RECORD_NOT_AN_OBJECT"}], d
    for f in DIST_FIELDS:
        if f not in rec:
            v.append({"where": where, "kind": "DIST_FIELD_ABSENT", "field": f})
    und = {}
    for f in READS_FIELDS:
        if f not in rec:
            continue
        sub = rec.get(f)
        if not isinstance(sub, dict):
            v.append({"where": where, "kind": "ENTKEY_SUBRECORD_NOT_AN_OBJECT", "field": f})
            continue
        miss, extra = sorted(set(ENTKEY_FIELDS) - set(sub)), sorted(set(sub) - set(ENTKEY_FIELDS))
        if miss:
            v.append({"where": where, "kind": "ENTKEY_FIELD_ABSENT", "field": f, "absent": miss})
        if extra:                                                 # §4.3: EXACTLY the ENTKEY_FIELDS keys
            v.append({"where": where, "kind": "ENTKEY_FIELD_UNEXPECTED", "field": f, "unexpected": extra})
        u, lp = sub.get("p_underflow"), sub.get("lp_first")
        if not isinstance(u, bool):
            v.append({"where": where, "kind": "P_UNDERFLOW_NOT_BOOL", "field": f, "value": u})
            continue
        und[f] = u
        if lp is None and u is False:
            v.append({"where": where, "kind": "LP_FIRST_NULL_WITHOUT_UNDERFLOW", "field": f})
        elif lp is not None and not _finite(lp):
            v.append({"where": where, "kind": "LP_FIRST_NOT_FINITE", "field": f, "value": lp})
        elif lp is not None and u is True:
            d.append({"where": where, "kind": "LP_FIRST_PRESENT_UNDER_UNDERFLOW", "field": f})
    for k in KEYS:
        cf, wf, mk, sk = "reads_c_%s" % k, "reads_w_%s" % k, "margin_first_%s" % k, "margin_sign_%s" % k
        if cf not in und or wf not in und or mk not in rec or sk not in rec:
            continue
        u, m, s = bool(und[cf] or und[wf]), rec[mk], rec[sk]
        undef = (is_margin_undefined(m), is_margin_undefined(s))
        if u and not all(undef):
            v.append({"where": where, "kind": "MARGIN_DEFINED_UNDER_UNDERFLOW", "key": k, "margin": m})
        elif (not u) and any(undef):
            v.append({"where": where, "kind": "MARGIN_UNDEFINED_WITHOUT_UNDERFLOW", "key": k, "margin": m})
        elif u:
            if m is None or s is None:
                d.append({"where": where, "kind": "MARGIN_NULL_NOT_LITERAL", "key": k})
        elif not _finite(m):
            v.append({"where": where, "kind": "MARGIN_FIRST_NOT_FINITE", "key": k, "value": m})
    return v, d


def check_stamp(rec, where):
    """§12: the shipped five-key stamp, complete, in order, every value a non-empty string. Pure -> list."""
    st = rec.get("stamp")
    if not isinstance(st, dict):
        return [{"where": where, "kind": "STAMP_ABSENT"}]
    out = []
    if tuple(st.keys()) != tuple(STAMP_KEYS):
        out.append({"where": where, "kind": "STAMP_KEYS_NOT_THE_SHIPPED_TUPLE_IN_ORDER", "found": list(st)})
    for k in STAMP_KEYS:
        if not isinstance(st.get(k), str) or not st[k].strip():
            out.append({"where": where, "kind": "STAMP_VALUE_NOT_A_NON_EMPTY_STRING", "key": k})
    return out


def check_axes(rec, parent, where, axes=AXES):
    """§12: every axis present and non-null on the record or inherited from its parent. Pure -> list."""
    out = []
    for a in axes:
        if a in rec and rec[a] is not None and not (isinstance(rec[a], str) and not rec[a].strip()):
            continue
        if isinstance(parent, dict) and a in parent and parent[a] is not None:
            continue
        out.append({"where": where, "kind": "AXIS_ABSENT_OR_NULL", "axis": a})
    return out


# ------------------------------------------------------------------ §11 provenance, §1.1 same session
def validate_gpu_provenance(prov, where):
    """§11: no per-artifact provenance -> no verdict from that run; every key PRESENT; a null/empty
    lambda_instance_id or started_utc is a FAILURE, not a note. Pure -> dict."""
    out = {"where": where, "present": bool(isinstance(prov, dict) and prov), "absent_keys": [],
           "null_load_bearing": []}
    if not out["present"]:
        out["status"] = "PROVENANCE_ABSENT"
        return out
    out["absent_keys"] = sorted(k for k in GPU_PROV_KEYS if k not in prov)
    out["null_load_bearing"] = sorted(k for k in GPU_PROV_LOAD_BEARING if _nullish(prov.get(k)))
    out["status"] = ("PROVENANCE_COMPLETE" if not out["absent_keys"] and not out["null_load_bearing"]
                     else "PROVENANCE_INCOMPLETE")
    out["lambda_instance_id"], out["started_utc"] = prov.get("lambda_instance_id"), prov.get("started_utc")
    return out


def validate_offline_provenance(prov):
    """§11's offline carve-out for THIS file: GPU fields and lambda_instance_id may be null; libraries, flags and
    git_commit may not. Pure -> dict."""
    absent = sorted(k for k in OFFLINE_REQUIRED if k not in (prov or {}))
    nulls = sorted(k for k in OFFLINE_REQUIRED if k not in absent and _nullish((prov or {}).get(k)))
    return {"required": list(OFFLINE_REQUIRED), "null_permitted": list(OFFLINE_NULL_OK), "absent_keys": absent,
            "null_required_keys": nulls,
            "status": ("OFFLINE_PROVENANCE_COMPLETE" if not absent and not nulls
                       else "OFFLINE_PROVENANCE_INCOMPLETE"),
            "carve_out": "§11 (R1-8(c)): GPU fields null and lambda_instance_id null permitted; library and "
                         "git_commit fields required; no abort on missing GPU env."}


def same_session(pa, pb, na="subst", nb="mask"):
    """§1.1: SAME_BOX iff lambda_instance_id non-null and equal, gpu_name equal, driver equal,
    cuda_visible_devices equal and == "0", device_index equal and == 0. A null id or an ABSENT field ->
    SAME_BOX_UNVERIFIABLE, and §6.7/§6.8/§6.9 are then NOT emitted. Pure -> dict."""
    out = {"rule": "§1.1", "pair": [na, nb], "required_fields": list(SAME_BOX_FIELDS),
           "suppresses_when_not_same_box": ["§6.7", "§6.8", "§6.9"],
           "readout_role": readout_role("§1.1", "n/a", "n/a", "same_session_gate")}
    if not pa or not pb:
        out.update({"status": "SAME_BOX_UNVERIFIABLE", "reason": "PROVENANCE_ABSENT",
                    "msg": "one or both sides carry no provenance object"})
        return out
    for nm, p in ((na, pa), (nb, pb)):
        if _nullish(p.get("lambda_instance_id")):
            out.update({"status": "SAME_BOX_UNVERIFIABLE", "reason": "LAMBDA_INSTANCE_ID_NULL",
                        "msg": "%s stamps lambda_instance_id=%r, so §6.7-§6.9 are NOT emitted"
                               % (nm, p.get("lambda_instance_id"))})
            return out
    absent = sorted({f for f in SAME_BOX_FIELDS for p in (pa, pb) if f not in p})
    if absent:
        out.update({"status": "SAME_BOX_UNVERIFIABLE", "reason": "PROVENANCE_FIELD_ABSENT",
                    "absent_fields": absent, "msg": "%s absent: equality cannot be asserted" % absent})
        return out
    ch = {"lambda_instance_id_equal": pa["lambda_instance_id"] == pb["lambda_instance_id"],
          "gpu_name_equal": pa["gpu_name"] == pb["gpu_name"], "driver_equal": pa["driver"] == pb["driver"],
          "cuda_visible_devices_equal": pa["cuda_visible_devices"] == pb["cuda_visible_devices"],
          "cuda_visible_devices_is_0": str(pa["cuda_visible_devices"]) == "0",
          "device_index_equal": pa["device_index"] == pb["device_index"],
          "device_index_is_0": str(pa["device_index"]) == "0"}
    ok = all(ch.values())
    ta, tb = pa.get("started_utc"), pb.get("started_utc")
    out.update({"status": "SAME_BOX" if ok else "NOT_SAME_BOX", "reason": None, "checks": ch,
                "lambda_instance_id": pa["lambda_instance_id"], "started_utc": {na: ta, nb: tb},
                "run_a_started_first": (None if _nullish(ta) or _nullish(tb) else bool(str(ta) <= str(tb))),
                "msg": ("every §1.1 condition holds" if ok else
                        "failing: %s -- §1: Run B without its same-session Run A is not a run under this "
                        "registration and yields no §6.7-§6.9 verdict"
                        % sorted(k for k, x in ch.items() if not x))})
    return out


# ------------------------------------------------------------------ readers (ambiguity B)
def iter_raw_records(summary, where):
    """Every raw per-arm record as (rec, parent, shape) from either dump shape. LOUD on an unreadable shape."""
    src = next((k for k in ("items", "records") if isinstance(summary.get(k), list)), None)
    if src is None:
        raise JoinFailure("MISSING_REQUIRED_FIELD", "%s carries neither `items` nor `records` as a list" % where)
    rows = []
    for i, el in enumerate(summary[src]):
        if not isinstance(el, dict):
            raise JoinFailure("RECORD_NOT_AN_OBJECT", "%s %s[%d] is not an object" % (where, src, i))
        if "turn_id" in el:
            rows.append((el, el, "flat"))
        elif isinstance(el.get("arms"), dict):
            for sub in el["arms"].values():
                if not isinstance(sub, dict):
                    raise JoinFailure("RECORD_NOT_AN_OBJECT", "%s %s[%d].arms non-object" % (where, src, i))
                rows.append((sub, el, "nested_arms"))
        else:
            raise JoinFailure("MISSING_REQUIRED_FIELD",
                              "%s %s[%d] carries neither `turn_id` nor an `arms` object" % (where, src, i))
    return rows, src


def _is_dist(rec):
    return rec.get("register") == DIST_REGISTER or ("topk_10" in rec and "commit_v2" not in rec)


def read_run(summary, path, run, arms):
    """Normalise ONE GPU summary and validate it as a run under this registration. NEVER reads the summary's own
    verdict/decision/rate fields."""
    where = "%s(%s)" % (run, path)
    out = {"run": run, "path": path, "arms": {a: {} for a in arms}, "n_records": 0, "n_dist_records": 0,
           "tag": summary.get("tag"), "name": summary.get("name"), "family": summary.get("family")}
    prov = summary.get("provenance")
    out["provenance"] = prov if isinstance(prov, dict) else None
    out["provenance_check"] = validate_gpu_provenance(prov, where)
    viol, disc, pending, shapes = [], [], [], set()
    rows, src = iter_raw_records(summary, where)
    for rec, parent, shape in rows:
        shapes.add(shape)
        tid = rec.get("turn_id")
        if tid not in arms:
            viol.append({"where": where, "kind": "TURN_ID_NOT_IN_THIS_RUN", "turn_id": tid})
            continue
        q = rec.get("q", parent.get("q") if isinstance(parent, dict) else None)
        if _nullish(q):
            raise JoinFailure("MISSING_REQUIRED_FIELD", "%s: a %s record carries no `q`" % (where, tid))
        jk = join_key(q)
        w2 = "%s %s/%s" % (where, tid, jk[:40])
        viol.extend(check_axes(rec, parent, w2))
        if _is_dist(rec):
            pending.append((jk, tid, rec.get("position"), rec, parent, w2))
            continue
        viol.extend(check_stamp(rec, w2))
        cv2 = rec.get("commit_v2")
        if cv2 not in COMMIT_VOCAB:
            raise JoinFailure("MISSING_REQUIRED_FIELD", "%s: commit_v2=%r not in %s" % (w2, cv2, COMMIT_VOCAB))
        cell = rec.get("cell", ARM_CELL[tid])
        if cell != ARM_CELL[tid]:
            raise JoinFailure("CELL_DISAGREES_WITH_TURN_ID",
                              "%s: cell=%r but §3 fixes %s as %r" % (w2, cell, tid, ARM_CELL[tid]))
        fs = rec.get("faithful_strict")
        if fs is not None and fs not in FAITHFUL_TO_COMMIT:
            viol.append({"where": w2, "kind": "FAITHFUL_STRICT_NOT_IN_VOCABULARY", "value": fs})
        loc, loc_field = None, True
        if tid in SPAN_ARMS:
            if "span_located" in rec:
                loc = bool(rec["span_located"])
            elif "span_unlocatable_reason" in rec:
                loc = _nullish(rec["span_unlocatable_reason"])
            else:
                loc, loc_field = None, False
        stable = rec.get("span_stable")
        excl = bool(rec.get("excluded")) or stable is False
        if rec.get("readout_role") == ROLE_PRIMARY:
            viol.append({"where": w2, "kind": "INPUT_RECORD_CLAIMS_PRIMARY_ROLE"})
        nrec = {"q": q, "join_key": jk, "turn_id": tid, "cell": cell, "commit_v2": cv2,
                "commit_v1": rec.get("commit_v1"), "faithful_strict": fs,
                "faithful_as_commit": FAITHFUL_TO_COMMIT.get(fs) if fs is not None else None,
                "excluded": excl, "eligible": not excl,
                "excluded_reason": rec.get("reason") if excl else None, "span_stable": stable,
                "span_located": loc, "span_location_field_present": loc_field,
                "turn_content_tokens": rec.get("turn_content_tokens"),
                "length_match_ok": rec.get("length_match_ok"), "pad_repeat": rec.get("pad_repeat"),
                "counter_gen": rec.get("counter_gen"), "elicit_gen": rec.get("elicit_gen"), "dist": {}}
        for key in ("distributions", "dist"):
            if isinstance(rec.get(key), dict):
                for pos, sub in rec[key].items():
                    pending.append((jk, tid, pos, sub, rec, "%s dist[%s]" % (w2, pos)))
                break
        if jk in out["arms"][tid]:
            raise JoinFailure("DUPLICATE_JOIN_KEY", "%s: two %s records share %r" % (where, tid, jk))
        out["arms"][tid][jk] = nrec
        out["n_records"] += 1
    for jk, tid, pos, sub, parent, w2 in pending:
        if pos not in POSITIONS:
            viol.append({"where": w2, "kind": "DIST_POSITION_NOT_REGISTERED", "position": pos})
            continue
        host = out["arms"].get(tid, {}).get(jk)
        if host is None:
            viol.append({"where": w2, "kind": "DIST_RECORD_WITHOUT_A_REALIZED_RECORD"})
            continue
        if pos in host["dist"]:
            raise JoinFailure("DUPLICATE_JOIN_KEY", "%s: two %s/%s dist records" % (where, tid, pos))
        viol.extend(check_axes(sub, parent, w2, DIST_AXES))
        v, d = validate_dist_record(sub, w2)
        viol.extend(v)
        disc.extend(d)
        host["dist"][pos] = sub
        out["n_dist_records"] += 1
    for a in arms:                                 # §4.3: a run omitting any field on ANY arm is not a run
        if not out["arms"][a]:
            viol.append({"where": where, "kind": "ARM_ABSENT", "turn_id": a})
            continue
        for jk, r in out["arms"][a].items():
            for pos in POSITIONS:
                if pos not in r["dist"]:
                    viol.append({"where": where, "kind": "DIST_RECORD_ABSENT", "turn_id": a, "position": pos})
    try:
        for a in arms:
            assert_unique_join_keys(list(out["arms"][a]))
    except ValueError as e:
        raise JoinFailure("DUPLICATE_JOIN_KEY", "%s: %s" % (where, e))
    kinds, dkinds = {}, {}
    for x in viol:
        kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
    for x in disc:
        dkinds[x["kind"]] = dkinds.get(x["kind"], 0) + 1
    ok = not viol and out["provenance_check"]["status"] == "PROVENANCE_COMPLETE"
    out["validity"] = {
        "status": RUN_OK if ok else NOT_A_RUN,
        "basis": "§4.3 (DIST_FIELDS/ENTKEY_FIELDS completeness + R2-1's MARGIN_UNDEFINED legality), §11 "
                 "(per-artifact provenance, load-bearing pair non-null), §12 (five-key stamp + eight axes)",
        "provenance_status": out["provenance_check"]["status"], "n_violations": len(viol),
        "violation_kinds": kinds, "violations_first_20": viol[:20], "n_disclosures": len(disc),
        "disclosure_kinds": dkinds, "n_records": out["n_records"], "n_dist_records": out["n_dist_records"],
        "record_shapes": sorted(shapes), "records_key": src,
        "arms_present": sorted(a for a in arms if out["arms"][a]),
        "arms_absent": sorted(a for a in arms if not out["arms"][a]),
        "consequence": ("every verdict this run sources is emitted" if ok else
                        "§4.3: a run that omits any registered field on any arm is NOT a run under this "
                        "registration -- every verdict sourced from it is a named non-emission"),
        "readout_role": readout_role("§4.3", "n/a", "n/a", "run_validity")}
    return out


def audit_rows(summary):
    """§6.6's persisted on-box audit rows, normalised over the candidate field names. Pure -> list."""
    raw = (summary or {}).get("mask_totality_audit")
    rows = raw.get("arms") if isinstance(raw, dict) else raw
    out = []
    for r in (rows or []):
        if not isinstance(r, dict):
            continue
        val = next((r[k] for k in ("max_masked_pattern", "max_masked_pattern_post_softmax",
                                   "max_post_softmax_mass", "leak_max") if _is_num(r.get(k))), None)
        out.append({"arm_class": r.get("arm_class", r.get("arm")), "max_masked_pattern": val,
                    "n_layers": r.get("n_layers"), "item": r.get("item")})
    return out


def p3c_padding_labels(p3c, where):
    """§6.11's committed side: the p3c padding_fold arm's per-item label (its own field name is `commit_elicit`,
    scored there by commit_prog_v2), keyed on join_key(q). Pure -> dict."""
    out = {}
    for i, it in enumerate(_req(p3c, "items", where)):
        rec = (it.get("arms") or {}).get("padding_fold") if isinstance(it, dict) else None
        if not isinstance(rec, dict) or rec.get("excluded") or rec.get("span_stable") is False:
            continue
        q = rec.get("q", it.get("q"))
        if _nullish(q):
            raise JoinFailure("MISSING_REQUIRED_FIELD", "%s items[%d] padding_fold has no `q`" % (where, i))
        jk = join_key(q)
        if jk in out:
            raise JoinFailure("DUPLICATE_JOIN_KEY", "%s: two padding_fold records share %r" % (where, jk))
        out[jk] = {"q": q, "label": rec.get("commit_elicit")}
    assert_unique_join_keys(list(out))
    return out


# ------------------------------------------------------------------ recomputed statistics (§4.1)
def arm_stat(recs, keys=None):
    """One arm's §4.1 statistics from per-item labels: r_move = moved/(moved+held) on the IMPORTED interpret;
    r_off = #{commit_v2 != 'correct'} over the FIXED denominator 74. Pure -> dict."""
    c = {"moved": 0, "held": 0, "abstain": 0}
    n_rec = n_el = n_off = 0
    for jk, r in sorted(recs.items()):
        if keys is not None and jk not in keys:
            continue
        n_rec += 1
        if not r["eligible"]:
            continue
        n_el += 1
        c[interpret(r["cell"], r["commit_v2"])] += 1
        n_off += int(r["commit_v2"] != "correct")
    den = c["moved"] + c["held"]
    return {"counts": c, "n_records": n_rec, "n_eligible": n_el, "n_excluded": n_rec - n_el,
            "r_move": (c["moved"] / den) if den else None, "r_move_denominator": den,
            "insufficient_eval": den < MIN_EVAL, "n_off_stated": n_off, "r_off": n_off / float(N_ITEMS),
            "r_off_denominator": N_ITEMS, "subset": "common_located" if keys is not None else "full_family"}


def turn_tokens(recs):
    """§6.3/§3.1's mandatory turn-length column for one arm. Pure -> dict."""
    vals = sorted(r["turn_content_tokens"] for r in recs.values() if _is_num(r["turn_content_tokens"]))
    return {"n": len(vals), "n_missing": len(recs) - len(vals), "min": (vals[0] if vals else None),
            "median": (statistics.median(vals) if vals else None), "max": (vals[-1] if vals else None),
            "value_counts": {str(v): vals.count(v) for v in sorted(set(vals))}}


def common_located(a_arms, b_arms):
    """§6.7's R1-4 common subset: the INTERSECTION of eligibility over all five terms plus §3.3 span-location on
    B2/B3/B4, so no term carries an item another term dropped (ambiguity F). Pure -> dict."""
    terms = {"A1": a_arms.get("A1", {}), "B2": b_arms.get("B2", {}), "B3": b_arms.get("B3", {}),
             "B4": b_arms.get("B4", {}), "B7": b_arms.get("B7", {})}
    el = {t: {k for k, r in m.items() if r["eligible"]} for t, m in terms.items()}
    span_el = el["B2"] & el["B3"] & el["B4"]
    absent = sorted({t for t in SPAN_ARMS for r in terms[t].values() if not r["span_location_field_present"]})
    located = {k for k in span_el if all(terms[t][k]["span_located"] is True for t in SPAN_ARMS)}
    common = located & el["A1"] & el["B7"]
    return {"rule": "§6.7's common-subset rule (R1-4)", "n_common": len(common), "n_span_located": len(located),
            "n_span_unlocatable": len(span_el - located),
            "span_unlocatable_join_keys": sorted(span_el - located),
            "n_eligible_per_term": {t: len(v) for t, v in el.items()},
            "span_location_fields_absent_on": absent, "keys": common,
            "note": "SPAN_UNLOCATABLE items are removed from EVERY term identically; the full-family rates "
                    "print beside every subset rate",
            "readout_role": readout_role("§6.7", "n/a", "n/a", "common_subset")}


def dissociation(recs, cell, turn_id):
    """§4.3's dissociation columns for one arm, per position x key, from margin_first = lp(C) - lp(W*). The
    target (pushed) entity is W* in the fold cell and C in the listen cell, so the sign favours the target iff
    margin < 0 (fold) / > 0 (listen). MARGIN_UNDEFINED excluded and counted; abstains counted. NO band, NO
    verdict (§4.3). Pure -> list."""
    rows = []
    for pos in POSITIONS:
        for k in KEYS:
            n = dict(defined=0, undef=0, absent=0, zero=0, t=0, s=0, th=0, sm=0, ab=0, dis=0, unp=0)
            for r in sorted(recs.values(), key=lambda x: x["join_key"]):
                d = r["dist"].get(pos)
                if not isinstance(d, dict) or ("margin_first_%s" % k) not in d:
                    n["absent"] += 1
                    continue
                m, s = d["margin_first_%s" % k], d.get("margin_sign_%s" % k)
                if is_margin_undefined(m):
                    n["undef"] += 1
                    continue
                if not _finite(m):
                    n["absent"] += 1
                    continue
                n["defined"] += 1
                der = 0 if m == 0 else (1 if m > 0 else -1)
                if _is_num(s) and int(s) in (-1, 0, 1):
                    n["dis"] += int(int(s) != der)
                elif not is_margin_undefined(s):
                    n["unp"] += 1
                f_t = (der < 0) if cell == "fold" else (der > 0)
                f_s = (der > 0) if cell == "fold" else (der < 0)
                outcome = interpret(r["cell"], r["commit_v2"]) if r["eligible"] else "excluded"
                n["zero"] += int(der == 0)
                n["t"] += int(f_t)
                n["s"] += int(f_s)
                if outcome == "abstain":
                    n["ab"] += 1
                elif outcome == "held" and f_t:
                    n["th"] += 1
                elif outcome == "moved" and f_s:
                    n["sm"] += 1
            rows.append({"turn_id": turn_id, "cell": cell, "position": pos, "key": k,
                         "n_margin_defined": n["defined"], "n_margin_undefined_excluded": n["undef"],
                         "n_record_absent_or_not_finite": n["absent"], "n_sign_zero_neither_side": n["zero"],
                         "n_sign_favours_pushed": n["t"], "n_sign_favours_stated": n["s"],
                         "n_sign_favours_pushed_but_held": n["th"],
                         "n_sign_favours_stated_but_moved": n["sm"],
                         "n_abstain_excluded_from_the_two_columns": n["ab"],
                         "n_persisted_sign_disagrees_with_derived": n["dis"],
                         "n_persisted_sign_unparsed": n["unp"],
                         "framing": "first-token, Rule-S-class; NO band and NO verdict (§4.3); never 'the "
                                    "probability of C' and never 'the model's belief'",
                         "readout_role": readout_role("§4.3", "state_first_tok", pos, "dissociation_columns")})
    return rows


def labels_map(recs):
    """{join_key: {q, label}} over an arm's ELIGIBLE records, for §6.11. Pure."""
    return {jk: {"q": r["q"], "label": r["commit_v2"]} for jk, r in recs.items() if r["eligible"]}


def movers(recs):
    """§6.9's movers(X): items whose commit_v2 elicited label under X is `wrong` (adopted W*). Pure -> set."""
    return {jk for jk, r in recs.items() if r["eligible"] and r["commit_v2"] == "wrong"}


def concordance(left, right, lcol, rcol, pair, note):
    """§6.11's per-item column. Item sets differing by exclusion are REPORTED, never raised (ambiguity K)."""
    lk, rk = set(left), set(right)
    rows = [{"join_key": jk, "q": left[jk].get("q") or right[jk].get("q"), lcol: left[jk]["label"],
             rcol: right[jk]["label"], "concordant": bool(left[jk]["label"] == right[jk]["label"])}
            for jk in sorted(lk & rk)]
    n_c = sum(1 for r in rows if r["concordant"])
    return {"pair": pair, "twin_relation": note, "n_joined": len(rows), "n_left_only": len(lk - rk),
            "n_right_only": len(rk - lk), "left_only_join_keys": sorted(lk - rk)[:10],
            "right_only_join_keys": sorted(rk - lk)[:10], "n_concordant": n_c,
            "n_discordant": len(rows) - n_c, "rows": rows,
            "quotation_rule": "§6.11: the item-level column IS the result; aggregate rates may not be quoted "
                              "without it, and any committed-vs-new disagreement is reported per item",
            "readout_role": readout_role("§6.11", "realized_commit_v2", "n/a", "concordance_column")}


# ------------------------------------------------------------------ §6 resolvers (pure, total, earlier wins)
def resolve_harness(r_a1, blocked=None):
    """§6.1 br 1: HARNESS_INSUFFICIENT iff r_move(A1) < MIN_BASE_RATE (None counts as below); suppresses
    §6.2-§6.11."""
    bad = blocked is not None or r_a1 is None or r_a1 < MIN_BASE_RATE
    return {"rule": "§6.1 branch 1", "verdict": "HARNESS_INSUFFICIENT" if bad else "HARNESS_SUFFICIENT",
            "r_move_A1": r_a1, "MIN_BASE_RATE": MIN_BASE_RATE, "suppressing_cause": blocked,
            "insufficient": bool(bad), "consequence": "suppresses §6.2-§6.11" if bad else "no suppression",
            "readout_role": readout_role("§6.1", "realized_commit_v2", "n/a", "harness_gate")}


def resolve_anchor(rate, committed, label, blocked=None):
    """§6.1 br 2-3: REPRODUCES iff |rate - committed| <= 0.10, else DIFFERS (a None rate DIFFERS, ambiguity C)."""
    ok = rate is not None and abs(rate - committed) <= A6_CONVERGE_ABS + EPS
    return {"rule": "§6.1 branch %s" % ("2" if label == "A" else "3"),
            "verdict": "%s_ANCHOR_%s" % (label, "REPRODUCES" if ok else "DIFFERS"), "rate": rate,
            "committed": committed, "abs_delta": (None if rate is None else abs(rate - committed)),
            "A6_CONVERGE_ABS": A6_CONVERGE_ABS, "blocked": blocked,
            "reason": (None if ok else ("RATE_IS_NONE" if rate is None else "OUTSIDE_THE_BAND")),
            "consequence": ("no suppression" if ok else
                            ("stamps every Run-A verdict " + ANCHOR_DIVERGENT_STAMP if label == "A"
                             else "SUPPRESSES §6.7, §6.8 and §6.9")),
            "readout_role": readout_role("§6.1", "realized_commit_v2", "n/a", "anchor_gate")}


def resolve_decomp(harness_bad, a1_insuff, a2_insuff, r_a1, r_a2, r_off_a3, blocked=None, floor=F_NC):
    """§6.2, THE PRIMARY (R1-1, R1-2). A3-active := r_off(A3) >= floor + 0.05, exact boundary inclusive."""
    active = _ge(r_off_a3, floor + KO_FLOOR_EPS)
    high = r_a1 is not None and r_a2 is not None and _ge(r_a2, KO_NULL_FRAC * r_a1)
    if blocked or harness_bad or a1_insuff or a2_insuff:
        v = DECOMP_ORDER[0]
        cause = blocked or ("HARNESS_INSUFFICIENT" if harness_bad else
                            ("A1_INSUFFICIENT_EVAL" if a1_insuff else "A2_INSUFFICIENT_EVAL"))
    elif high and not active:
        v, cause = DECOMP_ORDER[1], None
    elif high and active:
        v, cause = DECOMP_ORDER[2], None
    elif active:
        v, cause = DECOMP_ORDER[3], None
    elif _le(r_a2, floor + KO_FLOOR_EPS):
        v, cause = DECOMP_ORDER[4], None
    else:
        v, cause = DECOMP_ORDER[5], None
    return {"rule": "§6.2", "verdict": v, "suppressing_cause": cause, "resolution_order": list(DECOMP_ORDER),
            "r_move_A1": r_a1, "r_move_A2": r_a2, "r_off_A3": r_off_a3, "floor_FLOOR_NC_UNMASKED": floor,
            "A3_active": bool(active), "A2_preserves_0_9x": bool(high), "floor_band_edge": floor + KO_FLOOR_EPS,
            "null_band_edge_0_9x_r_move_A1": (None if r_a1 is None else KO_NULL_FRAC * r_a1),
            "stamps": [TRANSPORT_STAMP],
            "msg": "r_move(A2)=%r vs 0.9x=%r ; r_off(A3)=%r vs %r -> %s" % (
                r_a2, None if r_a1 is None else KO_NULL_FRAC * r_a1, r_off_a3, floor + KO_FLOOR_EPS, v),
            "note": "every condition reading r_off carries " + TRANSPORT_STAMP + " (R1-7): KO_FLOOR_EPS was "
                    "calibrated on an r_move-class rate against a MASKED-neutral floor, while r_off differs in "
                    "numerator and denominator and is read against the UNMASKED floor",
            "readout_role": readout_role("§6.2", "realized_commit_v2", "n/a", "decomposition_verdict_resolution")}


def resolve_dose(rates, insuff, blocked=None):
    """§6.3 (R1-3): UNEVALUABLE -> FLAT (max-min <= 0.10) -> MONOTONE (non-strict) -> NONMONOTONE."""
    vals = [rates.get(a) for a in DOSE_ARMS]
    if blocked or any(insuff.get(a) for a in DOSE_ARMS) or any(v is None for v in vals):
        v, cause, spread, mono = DOSE_ORDER[0], (blocked or "ARM_INSUFFICIENT_EVAL_OR_NONE_RATE"), None, None
    else:
        spread = max(vals) - min(vals)
        mono, cause = all(_le(vals[i], vals[i + 1]) for i in range(3)), None
        v = DOSE_ORDER[1] if _le(spread, A6_CONVERGE_ABS) else (DOSE_ORDER[2] if mono else DOSE_ORDER[3])
    sp = _spearman(list(range(1, 5)), vals) if (_spearman and all(x is not None for x in vals)) else None
    return {"rule": "§6.3", "verdict": v, "suppressing_cause": cause, "resolution_order": list(DOSE_ORDER),
            "rates": {a: rates.get(a) for a in DOSE_ARMS}, "spread": spread, "A6_CONVERGE_ABS": A6_CONVERGE_ABS,
            "non_strict_monotone": mono, "spearman_grade_index_vs_rate": sp,
            "spearman_backend": ("foldlisten_phase3c_riders.spearman (imported; report-only)" if _spearman
                                 else "UNAVAILABLE_P3C_NOT_IMPORTABLE (report-only, decides nothing)"),
            "mandatory_caveat": "R1-6: A4-A7 are NOT token-length-matched. Every DOSE_* verdict must be quoted "
                                "with the four per-arm turn_content_tokens distributions beside it, and no "
                                "outcome licenses attributing a gradient to grade rather than turn length.",
            "readout_role": readout_role("§6.3", "realized_commit_v2", "n/a", "dose_verdict")}


def resolve_grade_anchor(r_a6, r_a2, insuff_a6, insuff_a2, blocked=None):
    """§6.4 (R1-3): UNEVALUABLE -> CONVERGENT (|r_move(A6) - r_move(A2)| <= 0.10) -> DIVERGENT."""
    if blocked or insuff_a6 or insuff_a2 or r_a6 is None or r_a2 is None:
        v, cause, d = GRADE_ORDER[0], (blocked or "ARM_INSUFFICIENT_EVAL_OR_NONE_RATE"), None
    else:
        d = abs(r_a6 - r_a2)
        v, cause = (GRADE_ORDER[1] if d <= A6_CONVERGE_ABS + EPS else GRADE_ORDER[2]), None
    return {"rule": "§6.4", "verdict": v, "suppressing_cause": cause, "resolution_order": list(GRADE_ORDER),
            "r_move_A6": r_a6, "r_move_A2": r_a2, "abs_delta": d, "A6_CONVERGE_ABS": A6_CONVERGE_ABS,
            "consequence": "under GRADE_ANCHOR_DIVERGENT the `Actually, ` marker is doing measurable work and "
                           "every A2-based reading of §6.2 must be quoted with that fact beside it",
            "readout_role": readout_role("§6.4", "realized_commit_v2", "n/a", "grade_anchor_verdict")}


def resolve_a8(r_off_a8, blocked=None, floor=F_NC):
    """§6.5: UNEVALUABLE -> INERT (<= floor+0.05) -> DESTABILIZES (>= floor+0.18) -> A8_PARTIAL; both banded
    branches stamped as different-statistic transports (R1-7)."""
    if blocked or r_off_a8 is None:
        v, cause = A8_ORDER[0], (blocked or "R_OFF_ABSENT")
    elif _le(r_off_a8, floor + KO_FLOOR_EPS):
        v, cause = A8_ORDER[1], None
    elif _ge(r_off_a8, floor + A6_LEAK_MARGIN):
        v, cause = A8_ORDER[2], None
    else:
        v, cause = A8_ORDER[3], None
    return {"rule": "§6.5", "verdict": v, "suppressing_cause": cause, "resolution_order": list(A8_ORDER),
            "r_off_A8": r_off_a8, "floor_FLOOR_NC_UNMASKED": floor, "inert_edge": floor + KO_FLOOR_EPS,
            "destabilizes_edge": floor + A6_LEAK_MARGIN,
            "stamps": ([TRANSPORT_STAMP] if v in (A8_ORDER[1], A8_ORDER[2]) else []),
            "comparator": "§4.2: the same statistic on the committed unmasked neutral-C records, r_off = 0/74",
            "readout_role": readout_role("§6.5", "realized_commit_v2", "n/a", "a8_symmetry_verdict")}


def resolve_mask_totality(rows):
    """§6.6: MASK_TOTAL iff every audited arm class's max post-softmax masked mass is EXACTLY 0.0 (exp(-1e9)
    underflows in every float width, exp(-cap) does not -- no tolerance to tune), else MASK_SOFTCAPPED with the
    leak printed; an absent audit is a named non-emission (ambiguity E)."""
    usable = [r for r in (rows or []) if _is_num(r.get("max_masked_pattern"))]
    base = {"rule": "§6.6", "rows": list(rows or []),
            "comparator": "== 0.0 exactly, post-softmax, over every masked key position, layer and head",
            "readout_role": readout_role("§6.6", "n/a", "n/a", "mask_totality_verdict")}
    if not usable:
        base.update({"verdict": MASK_ORDER[0], "leak_max": None, "stamp_for_run_b": MASK_UNAUDITED_STAMP,
                     "consequence": "§6.6 suppresses nothing; Run-B numbers carry " + MASK_UNAUDITED_STAMP})
        return base
    leak = max(r["max_masked_pattern"] for r in usable)
    total = all(r["max_masked_pattern"] == 0.0 for r in usable)
    base.update({"verdict": MASK_ORDER[1] if total else MASK_ORDER[2], "leak_max": leak, "rows": usable,
                 "leak_max_per_row": {str(r.get("arm_class")): r["max_masked_pattern"] for r in usable},
                 "stamp_for_run_b": (None if total else "MASK_SOFTCAPPED_LEAK_MAX_%r" % leak),
                 "consequence": ("the mask arms measure information removal; no stamp" if total else
                                 "every Run-B number is stamped MASK_SOFTCAPPED_LEAK_MAX_<value>; §6.7/§6.9 are "
                                 "still emitted (the empirical guard is §6.1 branch 3) and the finding is an "
                                 "instrument fact about this machinery AT 9b-it (R1-8(d))")})
    return base


def span_guard(blocked, harness_bad, b_anchor_differs, same_box_status, insuff, r_b7, ref, extra=()):
    """§6.7 branch 1's guard, shared by §6.8 (R1-2, R1-4) -> (reason|None, collision arithmetic)."""
    coll = {"floor_band_edge": (None if r_b7 is None else r_b7 + KO_FLOOR_EPS),
            "null_band_edge": (None if ref is None else KO_NULL_FRAC * ref),
            "collision": bool(r_b7 is not None and ref is not None
                              and _ge(r_b7 + KO_FLOOR_EPS, KO_NULL_FRAC * ref))}
    if blocked:
        return blocked, coll
    if harness_bad:
        return "HARNESS_INSUFFICIENT", coll
    if b_anchor_differs:
        return "B_ANCHOR_DIFFERS", coll
    if same_box_status != "SAME_BOX":
        return ("SAME_BOX_UNVERIFIABLE" if same_box_status == "SAME_BOX_UNVERIFIABLE"
                else "PAIR_NOT_SAME_BOX"), coll
    for a in ("A1", "B7") + tuple(extra):
        if insuff.get(a):
            return "%s_INSUFFICIENT_EVAL" % a, coll
    if r_b7 is None or ref is None:
        return "RATE_ABSENT", coll
    if coll["collision"]:
        return "FLOOR_BAND_COLLISION", coll
    return None, coll


def at_floor(r, r_b7):
    """§6.7: at_floor(X) := r_move(X) <= r_move(B7) + KO_FLOOR_EPS on the same-run length-matched floor."""
    return _le(r, None if r_b7 is None else r_b7 + KO_FLOOR_EPS)


def resolve_span(reason, coll, r_b2, r_b3, r_b4, r_b7, ref):
    """§6.7, total, earlier branch wins."""
    af = {"B2": at_floor(r_b2, r_b7), "B3": at_floor(r_b3, r_b7), "B4": at_floor(r_b4, r_b7)}
    edge = None if ref is None else KO_NULL_FRAC * ref
    pres = {"B2": _ge(r_b2, edge), "B3": _ge(r_b3, edge)}
    stamps = []
    if reason:
        v, cause = SPAN_ORDER[0], reason
    elif af["B2"] and af["B3"]:
        v, cause = SPAN_ORDER[1], None
    elif af["B2"] and pres["B3"]:
        v, cause = SPAN_ORDER[2], None
    elif af["B3"] and pres["B2"]:
        v, cause = SPAN_ORDER[3], None
        stamps.append(DELIM_CONFOUNDED_STAMP if af["B4"] else
                      ("DELIMITER_CONFOUND_UNCHECKED_B4_RATE_ABSENT" if r_b4 is None
                       else "DELIMITER_CONFOUND_CHECKED_B4_ABOVE_FLOOR"))
    else:
        v, cause = SPAN_ORDER[4], None
    return {"rule": "§6.7", "verdict": v, "suppressing_cause": cause, "resolution_order": list(SPAN_ORDER),
            "stamps": stamps, "at_floor": af, "preserves_0_9x": pres, "r_move_B2": r_b2, "r_move_B3": r_b3,
            "r_move_B4": r_b4, "r_move_B7_same_run_floor": r_b7, "nomask_ref_r_move_A1": ref,
            "floor_band_arithmetic": coll, "committed_floor_beside": FLOOR_TABLE["FLOOR_NC_MASKED"],
            "stamp_note": DELIM_CONFOUNDED_STAMP + " (R1-4): the delimiter span is a subset of the frame span, "
                          "so a frame-kill co-occurring with a delimiter-kill cannot attribute the necessity to "
                          "the frame's content",
            "readout_role": readout_role("§6.7", "realized_commit_v2", "n/a", "span_verdict")}


def resolve_delimiter(reason, coll, r_b4, r_b7, ref):
    """§6.8: same guard/common subset/FLOOR_BAND_COLLISION as §6.7 br 1, plus B4's own guard (ambiguity D)."""
    if reason:
        v, cause = DELIM_ORDER[0], reason
    elif at_floor(r_b4, r_b7):
        v, cause = DELIM_ORDER[1], None
    elif _ge(r_b4, None if ref is None else KO_NULL_FRAC * ref):
        v, cause = DELIM_ORDER[2], None
    else:
        v, cause = DELIM_ORDER[3], None
    return {"rule": "§6.8", "verdict": v, "suppressing_cause": cause, "resolution_order": list(DELIM_ORDER),
            "r_move_B4": r_b4, "r_move_B7_same_run_floor": r_b7, "nomask_ref_r_move_A1": ref,
            "at_floor_B4": at_floor(r_b4, r_b7), "floor_band_arithmetic": coll,
            "readout_role": readout_role("§6.8", "realized_commit_v2", "n/a", "delimiter_verdict")}


def classify_survivor(b5, b6):
    """§6.9's per-item class, in order: UNEVALUABLE (either label is the abstain class `other`, incl.
    UNRESOLVED_ALIAS via the imported FAITHFUL_TO_COMMIT, or the record is absent -- R1-5, ambiguity H) ->
    DEPENDENT (correct in both) -> INDEPENDENT (wrong in both) -> VARIANT_DISCORDANT."""
    if b5 is None or b6 is None:
        return SURVIVOR_CLASSES[0], "RECORD_ABSENT"
    if b5 == "other" or b6 == "other":
        return SURVIVOR_CLASSES[0], "ABSTAIN_CLASS_BLOCKS_BOTH_CLEAN_CLASSES"
    if b5 == "correct" and b6 == "correct":
        return SURVIVOR_CLASSES[1], "HOLDS_UNDER_BOTH_NEUTRALIZATIONS"
    if b5 == "wrong" and b6 == "wrong":
        return SURVIVOR_CLASSES[2], "MOVES_UNDER_BOTH_NEUTRALIZATIONS"
    return SURVIVOR_CLASSES[3], "B5_AND_B6_DISAGREE_BOTH_NON_ABSTAIN"


def resolve_echo(reason, classes, s_empty):
    """§6.9: ECHO_UNEVALUABLE (guard, or S empty -- NOT a pass) -> ARTIFACT (all DEPENDENT) -> INDEPENDENT (all
    INDEPENDENT) -> MIXED (otherwise, including any SURVIVOR_UNEVALUABLE)."""
    if reason or s_empty:
        v, cause = ECHO_ORDER[0], (reason or "S_EMPTY")
        msg = ("no echo verdict: %s" % reason if reason else
               "nothing to adjudicate: the replication produced no above-floor survivor -- NOT a pass")
    elif classes and all(c == SURVIVOR_CLASSES[1] for c in classes):
        v, cause, msg = ECHO_ORDER[1], None, "every item of S holds under BOTH neutralizations"
    elif classes and all(c == SURVIVOR_CLASSES[2] for c in classes):
        v, cause, msg = ECHO_ORDER[2], None, "every item of S moves under BOTH neutralizations"
    else:
        v, cause, msg = ECHO_ORDER[3], None, "the per-item table is the result; no one-word summary is licensed"
    return {"rule": "§6.9", "verdict": v, "suppressing_cause": cause, "resolution_order": list(ECHO_ORDER),
            "n_S": len(classes), "class_counts": {c: classes.count(c) for c in SURVIVOR_CLASSES}, "msg": msg,
            "note": "S = movers(B1) \\ movers(B7) is DERIVED by arithmetic, so a parametric floor-mover falls "
                    "out of S because it moves in B7 too, not by a hand exclusion",
            "readout_role": readout_role("§6.9", "realized_commit_v2", "n/a", "echo_verdict")}


def resolve_floor_regression(arm, r, insuff, floor_name, high_label, blocked=None):
    """§6.10 (R1-3), report-with-stamp, no suppression: UNEVALUABLE -> FLOOR_CONSISTENT (within 0.10) -> the high
    branch (>= floor+0.18) -> FLOOR_INTERMEDIATE."""
    floor = FLOORS[floor_name][0]
    if blocked or r is None or insuff:
        v, cause = FLOOR_ORDER[0], (blocked or "NONE_RATE_OR_INSUFFICIENT_EVAL")
    elif abs(r - floor) <= A6_CONVERGE_ABS + EPS:
        v, cause = FLOOR_ORDER[1], None
    elif _ge(r, floor + A6_LEAK_MARGIN):
        v, cause = high_label, None
    else:
        v, cause = FLOOR_ORDER[4], None
    return {"rule": "§6.10", "arm": arm, "verdict": v, "suppressing_cause": cause,
            "resolution_order": [FLOOR_ORDER[0], FLOOR_ORDER[1], high_label, FLOOR_ORDER[4]], "r_move": r,
            "committed_floor": FLOOR_TABLE[floor_name], "signed_delta": (None if r is None else r - floor),
            "consistent_edge": A6_CONVERGE_ABS, "high_edge": floor + A6_LEAK_MARGIN,
            "transport": "same-statistic (an r_move-class rate against the masked-neutral floor class "
                         "A6_LEAK_MARGIN was calibrated on)",
            "readout_role": readout_role("§6.10", "realized_commit_v2", "n/a", "floor_regression")}


def offline_provenance():
    """§11's offline carve-out stamp: GPU fields null; libraries recorded by AVAILABILITY without importing torch
    (this control never calls it, and saying which path was taken is the point of the carve-out)."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    git, src = os.environ.get("GIT_COMMIT"), "env:GIT_COMMIT"
    if _nullish(git):
        try:
            r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), capture_output=True,
                               text=True, timeout=15)
            git, src = (r.stdout.strip() or None), "git rev-parse HEAD"
        except (OSError, subprocess.SubprocessError):
            git, src = None, "UNAVAILABLE"
    libs = {m: bool(importlib.util.find_spec(m)) for m in ("numpy", "scipy", "torch", "transformers",
                                                          "transformer_lens")}
    prov = {k: None for k in OFFLINE_NULL_OK}
    prov.update({"python": sys.version.split()[0], "platform": sys.platform, "git_commit": git,
                 "git_commit_source": src, "libraries": libs, "scipy_available": False,
                 "scipy_note": "scipy is never called here; no number depends on its presence",
                 "sibling_imports": dict(SIBLINGS), "started_utc": now, "finished_utc": now,
                 "gpu_note": "offline join: no GPU is opened"})
    return prov


# ------------------------------------------------------------------ assembly
def assemble(subst, mask, p3c, paths, prov):
    """Build the whole artifact. Pure given the loaded JSONs and the provenance stamp."""
    hard, missing = [], sorted(k for k, v in (("subst", subst), ("mask", mask), ("p3c", p3c)) if v is None)

    def guard(unit, fn):
        try:
            return fn(), None
        except JoinFailure as e:
            hard.append({"unit": unit, "kind": e.kind, "failure": str(e)})
            return None, str(e)

    runs = {}
    for nm, data, arms in (("subst", subst, RUN_A_ARMS), ("mask", mask, RUN_B_ARMS)):
        if data is None:
            runs[nm] = None
            continue
        runs[nm], _ = guard(nm, lambda d=data, n=nm, a=arms: read_run(d, paths.get(n), n, a))
    ra, rb = runs["subst"], runs["mask"]
    a_arms = ra["arms"] if ra else {a: {} for a in RUN_A_ARMS}
    b_arms = rb["arms"] if rb else {a: {} for a in RUN_B_ARMS}
    block_a = (None if (ra and ra["validity"]["status"] == RUN_OK) else
               ("SUBST_ARTIFACT_ABSENT" if subst is None else ("SUBST_JOIN_FAILURE" if ra is None else NOT_A_RUN)))
    block_b = (None if (rb and rb["validity"]["status"] == RUN_OK) else
               ("MASK_ARTIFACT_ABSENT" if mask is None else ("MASK_JOIN_FAILURE" if rb is None else NOT_A_RUN)))
    block_ab = block_a or block_b

    full = {a: arm_stat(a_arms.get(a, {})) for a in RUN_A_ARMS}
    full.update({a: arm_stat(b_arms.get(a, {})) for a in RUN_B_ARMS})
    r_move = {a: s["r_move"] for a, s in full.items()}
    r_off = {a: (s["r_off"] if s["n_records"] else None) for a, s in full.items()}
    insuff = {a: s["insufficient_eval"] for a, s in full.items()}
    subset = common_located(a_arms, b_arms)
    sub_stats = {a: arm_stat((a_arms if a == "A1" else b_arms).get(a, {}), subset["keys"])
                 for a in ("A1", "B2", "B3", "B4", "B7")}
    sub_move = {a: s["r_move"] for a, s in sub_stats.items()}
    sub_insuff = {a: s["insufficient_eval"] for a, s in sub_stats.items()}

    harness = resolve_harness(r_move.get("A1"), block_a)
    h_bad = harness["insufficient"]
    anch_a = resolve_anchor(r_move.get("A1"), FLOORS["FOLD_NOMASK_COMMITTED"][0], "A", block_a)
    anch_b = resolve_anchor(r_move.get("B1"), FLOORS["FOLD_MASK_COMMITTED"][0], "B", block_b)
    sess = same_session((ra or {}).get("provenance"), (rb or {}).get("provenance"))
    run_a_stamps = ([] if anch_a["verdict"].endswith("REPRODUCES") else [ANCHOR_DIVERGENT_STAMP])
    mask_tot = resolve_mask_totality(audit_rows(mask) if mask is not None else [])
    run_b_stamps = [s for s in [mask_tot.get("stamp_for_run_b")] if s]
    h_block = block_a or ("HARNESS_INSUFFICIENT" if h_bad else None)

    decomp = resolve_decomp(h_bad, insuff.get("A1", True), insuff.get("A2", True), r_move.get("A1"),
                            r_move.get("A2"), r_off.get("A3"), block_a)
    dose = resolve_dose(r_move, insuff, h_block)
    grade = resolve_grade_anchor(r_move.get("A6"), r_move.get("A2"), insuff.get("A6", True),
                                insuff.get("A2", True), h_block)
    a8 = resolve_a8(r_off.get("A8"), h_block)
    b_diff = anch_b["verdict"] == "B_ANCHOR_DIFFERS"
    g_span, coll = span_guard(block_ab, h_bad, b_diff, sess.get("status"), sub_insuff, sub_move.get("B7"),
                              sub_move.get("A1"), ("B2", "B3"))
    span = resolve_span(g_span, coll, sub_move.get("B2"), sub_move.get("B3"), sub_move.get("B4"),
                        sub_move.get("B7"), sub_move.get("A1"))
    g_del, _ = span_guard(block_ab, h_bad, b_diff, sess.get("status"), sub_insuff, sub_move.get("B7"),
                          sub_move.get("A1"), ("B2", "B3", "B4"))
    delim = resolve_delimiter(g_del, coll, sub_move.get("B4"), sub_move.get("B7"), sub_move.get("A1"))

    m_b1, m_b7 = movers(b_arms.get("B1", {})), movers(b_arms.get("B7", {}))
    S = sorted(m_b1 - m_b7)
    echo_rows, classes = [], []
    for jk in S:
        r5, r6 = b_arms.get("B5", {}).get(jk), b_arms.get("B6", {}).get(jk)
        l5 = r5["commit_v2"] if (r5 and r5["eligible"]) else None
        l6 = r6["commit_v2"] if (r6 and r6["eligible"]) else None
        cls, why = classify_survivor(l5, l6)
        classes.append(cls)
        echo_rows.append({"join_key": jk, "q": b_arms["B1"][jk]["q"], "label_B1": "wrong", "label_B5": l5,
                          "label_B6": l6, "survivor_class": cls, "why": why,
                          "B5_elicit_gen": (r5 or {}).get("elicit_gen"),
                          "B6_elicit_gen": (r6 or {}).get("elicit_gen"),
                          "B5_counter_gen": (r5 or {}).get("counter_gen")})
    g_echo = (block_ab or ("HARNESS_INSUFFICIENT" if h_bad else
                           ("B_ANCHOR_DIFFERS" if b_diff else
                            (sess.get("status") if sess.get("status") != "SAME_BOX" else None))))
    echo = resolve_echo(g_echo, classes, not S)
    echo.update({"S_join_keys": S, "n_movers_B1": len(m_b1), "n_movers_B7": len(m_b7), "per_item": echo_rows,
                 "rate_level_stamps": {
                     "B5_within_0_10_of_B1": (None if r_move.get("B5") is None or r_move.get("B1") is None
                                              else abs(r_move["B5"] - r_move["B1"]) <= A6_CONVERGE_ABS + EPS),
                     "B6_within_0_10_of_B1": (None if r_move.get("B6") is None or r_move.get("B1") is None
                                              else abs(r_move["B6"] - r_move["B1"]) <= A6_CONVERGE_ABS + EPS),
                     "note": "printed as STAMPS, never as the verdict (§6.9)"},
                 "n_new_movers_under_B5": len(movers(b_arms.get("B5", {})) - m_b1),
                 "n_new_movers_under_B6": len(movers(b_arms.get("B6", {})) - m_b1)})

    floors = {"B7_vs_FLOOR_NC_MASKED": resolve_floor_regression("B7", r_move.get("B7"), insuff.get("B7", True),
                                                               "FLOOR_NC_MASKED", FLOOR_ORDER[2], block_b),
              "B8_vs_FLOOR_NW_MASKED": resolve_floor_regression("B8", r_move.get("B8"), insuff.get("B8", True),
                                                                "FLOOR_NW_MASKED", FLOOR_ORDER[3], block_b),
              "B1_vs_FOLD_MASK_COMMITTED": {"rule": "§6.10", "arm": "B1", "verdict": anch_b["verdict"],
                                            "note": "this row IS §6.1 branch 3 (the suppressing anchor); no "
                                                    "separate stamp is defined",
                                            "r_move": r_move.get("B1"),
                                            "committed_floor": FLOOR_TABLE["FOLD_MASK_COMMITTED"],
                                            "readout_role": readout_role("§6.10", "realized_commit_v2", "n/a",
                                                                         "floor_regression")}}
    diss = []
    for a in RUN_A_ARMS:
        diss.extend(dissociation(a_arms.get(a, {}), ARM_CELL[a], a))
    for a in RUN_B_ARMS:
        diss.extend(dissociation(b_arms.get(a, {}), ARM_CELL[a], a))

    conc = {"B6_vs_B5": concordance(labels_map(b_arms.get("B6", {})), labels_map(b_arms.get("B5", {})),
                                   "label_mask_B6", "label_subst_B5", "B6 <-> B5",
                                   "within-run: mask-the-echo vs substitute-the-echo")}
    if p3c is None:
        conc["B1_vs_PADDING_COMMITTED"] = {"pair": "B1 <-> PADDING_COMMITTED",
                                           "verdict": "CONCORDANCE_UNEVALUABLE_P3C_ARTIFACT_ABSENT",
                                           "committed_floor": FLOOR_TABLE["PADDING_COMMITTED"],
                                           "readout_role": readout_role("§6.11", "realized_commit_v2", "n/a",
                                                                        "concordance_column")}
    else:
        pad, _ = guard("p3c", lambda: p3c_padding_labels(p3c, "p3c(%s)" % paths.get("p3c")))
        conc["B1_vs_PADDING_COMMITTED"] = (
            concordance(labels_map(b_arms.get("B1", {})), pad or {}, "label_mask_B1", "label_subst_padding",
                        "B1 <-> PADDING_COMMITTED (cross-run, join on q)",
                        "cross-run: score-mask vs the committed p3c pad substitution")
            if pad is not None else {"pair": "B1 <-> PADDING_COMMITTED",
                                     "verdict": "CONCORDANCE_UNEVALUABLE_P3C_JOIN_FAILURE",
                                     "readout_role": readout_role("§6.11", "realized_commit_v2", "n/a",
                                                                  "concordance_column")})
        conc["B1_vs_PADDING_COMMITTED"]["committed_floor"] = FLOOR_TABLE["PADDING_COMMITTED"]

    primary = {"readout_role": readout_role(PRIMARY_READOUT["rule"], PRIMARY_READOUT["register"],
                                            PRIMARY_READOUT["position"], PRIMARY_READOUT["statistic"]),
               "designation": dict(PRIMARY_READOUT), "verdict": decomp["verdict"],
               "input_rates": {"r_move_A1": r_move.get("A1"), "r_move_A2": r_move.get("A2"),
                               "r_off_A3": r_off.get("A3")},
               "input_counts": {a: full[a]["counts"] for a in ("A1", "A2", "A3")},
               "suppressing_cause": decomp["suppressing_cause"], "stamps": run_a_stamps + decomp["stamps"],
               "quotation_requirements": [QUOTATION_RULES[0], QUOTATION_RULES[1]]
               + ([QUOTATION_RULES[3]] if grade["verdict"] == GRADE_ORDER[2] else []),
               "A3_r_move_blind_reversion_class": {"r_move_A3": r_move.get("A3"),
                                                   "parametric_floor_named": "1/74 (§6.2's requirement)"}}
    validity = {nm: (runs[nm]["validity"] if runs[nm] else
                     {"status": NOT_A_RUN,
                      "reason": ("ARTIFACT_ABSENT" if (subst if nm == "subst" else mask) is None
                                 else "JOIN_FAILURE"),
                      "readout_role": readout_role("§4.3", "n/a", "n/a", "run_validity")})
                for nm in ("subst", "mask")}
    not_a_run = [nm for nm in ("subst", "mask") if validity[nm]["status"] != RUN_OK]
    out = {"control": "foldlisten_demarez_join",
           "registration": "docs/drafts/REGISTRATION_demarez_spans.md (frozen, pre-data, R1-*/R2-*)",
           "metric": METRIC, "decision_rule": DECISION_RULE, "quotation_rules": QUOTATION_RULES,
           "stamp": stamp(), "stamp_keys": list(STAMP_KEYS), "thresholds": THRESHOLDS, "floors": FLOOR_TABLE,
           "inputs": dict(paths), "missing_inputs": missing, "run_validity": validity,
           "runs_not_under_this_registration": not_a_run, "same_session": sess, "primary_readout": primary,
           "verdicts": {"harness": harness, "anchor_A": anch_a, "anchor_B": anch_b, "decomp": decomp,
                        "dose": dose, "grade_anchor": grade, "a8": a8, "mask_totality": mask_tot, "span": span,
                        "delimiter": delim, "echo": echo, "floor_regressions": floors},
           "arm_measurements": {"full_family": {a: full[a] for a in sorted(full)},
                                "common_located_subset": {a: sub_stats[a] for a in sorted(sub_stats)}},
           "common_subset": {k: v for k, v in subset.items() if k != "keys"},
           "turn_content_tokens": {a: turn_tokens((a_arms if a in RUN_A_ARMS else b_arms).get(a, {}))
                                   for a in RUN_A_ARMS + RUN_B_ARMS},
           "dissociation_columns": diss, "concordance": conc,
           "run_stamps": {"run_A": run_a_stamps, "run_B": run_b_stamps},
           "provenance": prov, "provenance_validation": validate_offline_provenance(prov),
           "sibling_imports": dict(SIBLINGS), "hard_failures": hard}
    if out["provenance_validation"]["status"] != "OFFLINE_PROVENANCE_COMPLETE":
        hard.append({"unit": "provenance", "kind": "OFFLINE_PROVENANCE_INCOMPLETE",
                     "failure": str(out["provenance_validation"])})
    out["exit_code"] = (EXIT_HARD if hard else
                        (EXIT_NOT_A_RUN if not_a_run else (EXIT_MISSING if missing else EXIT_OK)))
    out["n_primary_role_fields"] = count_role(out, ROLE_PRIMARY)
    if out["n_primary_role_fields"] != 1:
        raise RuntimeError("§8/§12 violated: %d objects carry readout_role=%r, expected exactly 1"
                           % (out["n_primary_role_fields"], ROLE_PRIMARY))
    return out


def _load(path, unit, missing):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        missing.append({"unit": unit, "path": str(p), "kind": "ARTIFACT_ABSENT"})
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        missing.append({"unit": unit, "path": str(p), "kind": "ARTIFACT_UNREADABLE", "error": str(e)})
        return None


def run(subst_path, mask_path, p3c_path, outdir):
    miss = []
    subst = _load(subst_path, "subst", miss)
    mask = _load(mask_path, "mask", miss)
    p3c = _load(p3c_path, "p3c", miss)
    art = assemble(subst, mask, p3c, {"subst": subst_path, "mask": mask_path, "p3c": p3c_path},
                   offline_provenance())
    art["load_failures"] = miss
    p = Path(outdir) / "demarez_join.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(art, indent=2, default=str), encoding="utf-8")
    pr = art["primary_readout"]
    print("[PRIMARY §6.2/§8] %s | r_move(A1)=%s r_move(A2)=%s r_off(A3)=%s (quotable with all three or not at "
          "all)" % (pr["verdict"], pr["input_rates"]["r_move_A1"], pr["input_rates"]["r_move_A2"],
                    pr["input_rates"]["r_off_A3"]), flush=True)
    for nm in ("subst", "mask"):
        v = art["run_validity"][nm]
        print("[validity %s] %s (prov=%s, violations=%s)" % (nm, v["status"], v.get("provenance_status"),
                                                             v.get("n_violations")), flush=True)
    print("[§1.1] %s -- %s" % (art["same_session"]["status"], art["same_session"].get("msg")), flush=True)
    for k in ("harness", "anchor_A", "anchor_B", "decomp", "dose", "grade_anchor", "a8", "mask_totality",
              "span", "delimiter", "echo"):
        v = art["verdicts"][k]
        print("[%s] %s%s" % (v.get("rule", k), v["verdict"],
                             "" if not v.get("suppressing_cause") else " (cause=%s)" % v["suppressing_cause"]),
              flush=True)
    for k, v in art["verdicts"]["floor_regressions"].items():
        print("[§6.10 %s] %s (r_move=%s vs %s)" % (k, v["verdict"], v.get("r_move"),
                                                   v["committed_floor"]["as_written"]), flush=True)
    cs = art["common_subset"]
    print("[§6.7 subset] n_common=%s n_span_located=%s n_span_unlocatable=%s"
          % (cs["n_common"], cs["n_span_located"], cs["n_span_unlocatable"]), flush=True)
    for key, c in art["concordance"].items():
        print("[§6.11 %s] joined=%s concordant=%s discordant=%s%s"
              % (key, c.get("n_joined"), c.get("n_concordant"), c.get("n_discordant"),
                 "" if "verdict" not in c else " verdict=%s" % c["verdict"]), flush=True)
    print("[§4.3] %d dissociation rows (arm x position x key), NO band and NO verdict"
          % len(art["dissociation_columns"]), flush=True)
    for m in miss:
        print("[MISSING] %s: %s (%s) -> named non-emission, never a default" % (m["unit"], m["path"], m["kind"]),
              flush=True)
    for h in art["hard_failures"]:
        print("[FAIL] %s: %s" % (h["unit"], h["failure"]), flush=True)
    print("[primary] n_primary_role_fields=%d (§8: exactly 1)" % art["n_primary_role_fields"], flush=True)
    print("[done] wrote %s (exit %d)" % (p.as_posix(), art["exit_code"]), flush=True)
    return art["exit_code"]


# ------------------------------------------------------------------ selftest (model-free, artifact-free)
def _prov(iid="dmzbox", t="2026-07-30T00:00:00+00:00"):
    p = {k: "x" for k in GPU_PROV_KEYS}
    p.update({"lambda_instance_id": iid, "gpu_name": "A100", "driver": "570", "cuda_visible_devices": "0",
              "device_index": 0, "started_utc": t, "finished_utc": t, "gpu_count": 1})
    return p


def _ent(under=False):
    return {"tok_id": 7, "p_full": 0.0 if under else 0.5, "lp_first": None if under else -0.6931,
            "p_underflow": bool(under), "rank_first_tok": 1, "tie_plateau": 1, "first_token_collision": False}


def _dist(pos, margin=0.25, under=False, drop=None):
    rec = {"topk_10": [[1, "a", 0.5, 0.5]], "argmax_tok_id": 1, "argmax_tok_str": "a",
           "register": DIST_REGISTER, "position": pos}
    for e in ("c", "w"):
        for k in KEYS:
            rec["reads_%s_%s" % (e, k)] = _ent(under and e == "w")
    for k in KEYS:
        rec["margin_first_%s" % k] = MARGIN_UNDEFINED if under else margin
        rec["margin_sign_%s" % k] = (MARGIN_UNDEFINED if under
                                     else (0 if margin == 0 else (1 if margin > 0 else -1)))
    if drop:
        rec.pop(drop, None)
    return rec


def _rec(tid, i, label, margin=0.25, under=False, located=True, drop=None):
    return {"turn_id": tid, "q": "q%d?" % i, "cell": ARM_CELL[tid], "commit_v2": label, "commit_v1": label,
            "faithful_strict": {"correct": "C", "wrong": "WSTAR", "other": "NEITHER"}[label],
            "mask_span_id": "full_turn", "echo_treatment": "none", "key": "space+bare",
            "key_is_canonical": True, "register": "realized_commit_v2", "position": "n/a",
            "readout_role": ROLE_SECONDARY, "turn_content_tokens": 10 + (3 if tid == "A5" else 0),
            "span_stable": True, "span_located": located, "stamp": {k: "prose %s" % k for k in STAMP_KEYS},
            "elicit_gen": "gen", "counter_gen": "cg",
            "distributions": {p: _dist(p, margin, under, drop) for p in POSITIONS}}


def _labels(n, n_wrong, n_other=0):
    return ["wrong"] * n_wrong + ["other"] * n_other + ["correct"] * (n - n_wrong - n_other)


def _summary(arms, spec, n=8, prov=None, under_item=None, drop=None):
    items = []
    for a in arms:
        labs = spec.get(a, _labels(n, 0))
        for i in range(n):
            items.append(_rec(a, i, labs[i], under=(under_item == (a, i)),
                              drop=(drop if (a, i) == (arms[0], 0) else None)))
    return {"tag": "syn", "name": "google/gemma-2-9b-it", "family": "mechanism_family_9bit.json",
            "provenance": prov or _prov(), "items": items,
            "mask_totality_audit": [{"arm_class": "full_turn", "max_masked_pattern": 0.0, "n_layers": 42,
                                     "item": 0}]}


def selftest():
    n = 8
    assert FAITHFUL_TO_COMMIT["UNRESOLVED_ALIAS"] == "other" and FAITHFUL_TO_COMMIT["WSTAR"] == "wrong"
    assert (KO_FLOOR_EPS, KO_NULL_FRAC, MIN_BASE_RATE, A6_CONVERGE_ABS, A6_LEAK_MARGIN, MIN_EVAL, N_ITEMS) \
        == (0.05, 0.9, 0.5, 0.10, 0.18, 6, 74)
    assert tuple(STAMP_KEYS) == ("arm", "slot", "labels", "map_confidence", "tiebreak")
    assert len(DIST_FIELDS) == 11 and len(ENTKEY_FIELDS) == 7
    for m in ("foldlisten_demarez_subst", "foldlisten_demarez_mask"):
        if importlib.util.find_spec(m) is not None:                        # pragma: no cover (post-write)
            mod = __import__(m)
            assert tuple(mod.DIST_FIELDS) == DIST_FIELDS and tuple(mod.ENTKEY_FIELDS) == ENTKEY_FIELDS
    print("[selftest] thresholds/tuples/stamp keys asserted against their sources; siblings=%s" % SIBLINGS)

    # ---- §6.1 ----
    assert resolve_harness(0.49)["verdict"] == "HARNESS_INSUFFICIENT"
    assert resolve_harness(0.5)["verdict"] == "HARNESS_SUFFICIENT" and resolve_harness(None)["insufficient"]
    assert resolve_anchor(0.9, 1.0, "A")["verdict"] == "A_ANCHOR_REPRODUCES"
    assert resolve_anchor(0.89, 1.0, "A")["verdict"] == "A_ANCHOR_DIFFERS"
    assert resolve_anchor(None, 1.0, "A")["reason"] == "RATE_IS_NONE"
    fmc = FLOORS["FOLD_MASK_COMMITTED"][0]
    assert resolve_anchor(0.125, fmc, "B")["verdict"] == "B_ANCHOR_REPRODUCES"      # |0.125-0.0274| = 0.0976
    assert resolve_anchor(0.25, fmc, "B")["verdict"] == "B_ANCHOR_DIFFERS"

    # ---- §6.2: all six branches, the 2x2 walk, both boundary directions, guard precedence ----
    def D(a2, off3, r1=1.0, **kw):
        return resolve_decomp(False, False, False, r1, a2, off3, **kw)["verdict"]

    assert D(1.0, 0.0) == "ASSERTION_SUFFICIENT" and D(0.9, 0.049) == "ASSERTION_SUFFICIENT"
    assert D(1.0, 0.05) == "BOTH_COMPONENTS_ACTIVE"                        # exact 0.05 counts as ACTIVE
    assert D(0.89, 0.05) == "QUESTION_DOES_WORK" and D(0.0, 0.06) == "QUESTION_DOES_WORK"
    assert D(0.05, 0.0) == "CONJUNCTIVE" and D(0.0, 0.0) == "CONJUNCTIVE"  # exact 0.05 counts as at floor
    assert D(0.5, 0.0) == "DECOMP_PARTIAL" and D(0.051, 0.049) == "DECOMP_PARTIAL"
    assert D(0.9, 0.0499999) == "ASSERTION_SUFFICIENT" and D(0.899, 0.0) == "DECOMP_PARTIAL"
    assert resolve_decomp(True, False, False, 1.0, 1.0, 0.0)["verdict"] == "DECOMP_UNEVALUABLE"
    assert resolve_decomp(False, True, False, 1.0, 1.0, 0.9)["suppressing_cause"] == "A1_INSUFFICIENT_EVAL"
    assert resolve_decomp(False, False, True, 1.0, 1.0, 0.9)["suppressing_cause"] == "A2_INSUFFICIENT_EVAL"
    assert resolve_decomp(False, False, False, 1.0, 1.0, 0.9, blocked=NOT_A_RUN)["verdict"] \
        == "DECOMP_UNEVALUABLE"                                            # not-a-run precedes every branch
    assert {D(1.0, 0.0), D(1.0, 0.05), D(0.89, 0.05), D(0.05, 0.0), D(0.5, 0.0), "DECOMP_UNEVALUABLE"} \
        == set(DECOMP_ORDER)
    print("[selftest] §6.2: all six branches reachable, every 2x2 cell, exact 0.05 ACTIVE and exact 0.9x "
          "preserving, guards precede every band")

    # ---- §6.3-§6.5 ----
    fl = {a: False for a in DOSE_ARMS}
    assert resolve_dose({"A4": .1, "A5": .15, "A6": .2, "A7": .2}, fl)["verdict"] == "DOSE_FLAT"   # spread 0.10
    assert resolve_dose({"A4": .1, "A5": .2, "A6": .3, "A7": .4}, fl)["verdict"] == "DOSE_MONOTONE"
    assert resolve_dose({"A4": .4, "A5": .2, "A6": .3, "A7": .9}, fl)["verdict"] == "DOSE_NONMONOTONE"
    assert resolve_dose({"A4": .1, "A5": .2, "A6": .3, "A7": .4}, dict(fl, A5=True))["verdict"] \
        == "DOSE_UNEVALUABLE"
    assert resolve_grade_anchor(0.5, 0.6, False, False)["verdict"] == "GRADE_ANCHOR_CONVERGENT"
    assert resolve_grade_anchor(0.5, 0.61, False, False)["verdict"] == "GRADE_ANCHOR_DIVERGENT"
    assert resolve_grade_anchor(None, 0.6, False, False)["verdict"] == "GRADE_ANCHOR_UNEVALUABLE"
    assert resolve_a8(0.05)["verdict"] == "PUSH_TOWARD_STATED_INERT"
    assert resolve_a8(0.18)["verdict"] == "PUSH_TOWARD_STATED_DESTABILIZES"
    assert resolve_a8(0.1)["verdict"] == "A8_PARTIAL" and resolve_a8(None)["verdict"] == "A8_UNEVALUABLE"
    assert TRANSPORT_STAMP in resolve_a8(0.18)["stamps"] and TRANSPORT_STAMP in resolve_a8(0.05)["stamps"]

    # ---- §6.6: the exact-0.0 vs 1e-22 discrimination ----
    assert resolve_mask_totality([{"arm_class": "x", "max_masked_pattern": 0.0}])["verdict"] == "MASK_TOTAL"
    sc = resolve_mask_totality([{"arm_class": "x", "max_masked_pattern": 0.0},
                                {"arm_class": "y", "max_masked_pattern": 1e-22}])
    assert sc["verdict"] == "MASK_SOFTCAPPED" and sc["stamp_for_run_b"].startswith("MASK_SOFTCAPPED_LEAK_MAX_")
    assert resolve_mask_totality([])["verdict"] == MASK_ORDER[0]

    # ---- §6.7/§6.8 incl. FLOOR_BAND_COLLISION and every guard ----
    ins = {a: False for a in ("A1", "B2", "B3", "B4", "B7")}
    g, c = span_guard(None, False, False, "SAME_BOX", ins, 0.03, 1.0)
    assert g is None and c["collision"] is False
    assert resolve_span(g, c, 0.05, 1.0, 0.5, 0.03, 1.0)["verdict"] == "ENTITY_CARRIES"
    assert resolve_span(g, c, 0.05, 0.06, 0.5, 0.03, 1.0)["verdict"] == "CONJUNCTIVE_READ"
    fr = resolve_span(g, c, 1.0, 0.05, 0.02, 0.03, 1.0)
    assert fr["verdict"] == "FRAME_CARRIES" and DELIM_CONFOUNDED_STAMP in fr["stamps"]
    assert resolve_span(g, c, 0.5, 0.5, 0.5, 0.03, 1.0)["verdict"] == "SPAN_PARTIAL"
    gc, cc = span_guard(None, False, False, "SAME_BOX", ins, 0.9, 1.0)          # 0.9+0.05 >= 0.9*1.0
    assert gc == "FLOOR_BAND_COLLISION" and cc["collision"] is True
    assert resolve_span(gc, cc, 0.05, 0.05, 0.05, 0.9, 1.0)["verdict"] == "SPAN_UNEVALUABLE"
    assert span_guard(None, False, False, "SAME_BOX_UNVERIFIABLE", ins, 0.9, 1.0)[0] == "SAME_BOX_UNVERIFIABLE"
    assert span_guard(None, False, False, "NOT_SAME_BOX", ins, 0.03, 1.0)[0] == "PAIR_NOT_SAME_BOX"
    assert span_guard(None, False, True, "SAME_BOX", ins, 0.03, 1.0)[0] == "B_ANCHOR_DIFFERS"
    assert span_guard(None, True, True, "SAME_BOX", ins, 0.03, 1.0)[0] == "HARNESS_INSUFFICIENT"
    assert span_guard(NOT_A_RUN, True, True, "SAME_BOX", ins, 0.03, 1.0)[0] == NOT_A_RUN
    assert span_guard(None, False, False, "SAME_BOX", dict(ins, B2=True), 0.03, 1.0,
                      ("B2", "B3"))[0] == "B2_INSUFFICIENT_EVAL"
    assert resolve_delimiter(None, c, 0.02, 0.03, 1.0)["verdict"] == "DELIMITER_CARRIES"
    assert resolve_delimiter(None, c, 1.0, 0.03, 1.0)["verdict"] == "DELIMITER_INERT"
    assert resolve_delimiter(None, c, 0.5, 0.03, 1.0)["verdict"] == "DELIMITER_PARTIAL"
    assert resolve_delimiter("B4_INSUFFICIENT_EVAL", c, 0.0, 0.03, 1.0)["verdict"] == "DELIMITER_UNEVALUABLE"
    print("[selftest] §6.7/§6.8: every branch, FLOOR_BAND_COLLISION -> SPAN_UNEVALUABLE, DELIMITER_CONFOUNDED "
          "stamped, same-box/anchor/harness/not-a-run guards ordered")

    # ---- §6.9 ----
    assert classify_survivor("correct", "correct")[0] == "SURVIVOR_ECHO_DEPENDENT"
    assert classify_survivor("wrong", "wrong")[0] == "SURVIVOR_ECHO_INDEPENDENT"
    assert classify_survivor("wrong", "correct")[0] == "SURVIVOR_VARIANT_DISCORDANT"
    assert classify_survivor("other", "correct")[0] == "SURVIVOR_UNEVALUABLE"   # abstain precedes discordant
    assert classify_survivor(None, "wrong")[1] == "RECORD_ABSENT"
    assert resolve_echo(None, ["SURVIVOR_ECHO_DEPENDENT"], False)["verdict"] == "ECHO_ARTIFACT"
    assert resolve_echo(None, ["SURVIVOR_ECHO_INDEPENDENT"] * 2, False)["verdict"] == "ECHO_INDEPENDENT"
    assert resolve_echo(None, ["SURVIVOR_ECHO_DEPENDENT", "SURVIVOR_UNEVALUABLE"], False)["verdict"] \
        == "ECHO_MIXED"
    assert resolve_echo(None, [], True)["suppressing_cause"] == "S_EMPTY"
    assert resolve_echo("B_ANCHOR_DIFFERS", ["SURVIVOR_ECHO_DEPENDENT"], False)["verdict"] == "ECHO_UNEVALUABLE"

    # ---- §6.10 ----
    f7 = FLOORS["FLOOR_NC_MASKED"][0]
    R = resolve_floor_regression
    assert R("B7", f7, False, "FLOOR_NC_MASKED", FLOOR_ORDER[2])["verdict"] == "FLOOR_CONSISTENT"
    assert R("B7", f7 + 0.18, False, "FLOOR_NC_MASKED", FLOOR_ORDER[2])["verdict"] \
        == "LENGTH_MATCHED_FLOOR_HIGHER"
    assert R("B7", f7 + 0.14, False, "FLOOR_NC_MASKED", FLOOR_ORDER[2])["verdict"] == "FLOOR_INTERMEDIATE"
    assert R("B8", None, False, "FLOOR_NW_MASKED", FLOOR_ORDER[3])["verdict"] == "FLOOR_REGRESSION_UNEVALUABLE"
    assert R("B8", FLOORS["FLOOR_NW_MASKED"][0] + 0.2, False, "FLOOR_NW_MASKED", FLOOR_ORDER[3])["verdict"] \
        == "FLOOR_HIGHER_THAN_COMMITTED"

    # ---- §4.3 completeness, R2-1's MARGIN_UNDEFINED, and the null policing ----
    assert validate_dist_record(_dist("elicit_first"), "w") == ([], [])
    assert any(x["kind"] == "DIST_FIELD_ABSENT"
               for x in validate_dist_record(_dist("elicit_first", drop="topk_10"), "w")[0])
    assert validate_dist_record(_dist("elicit_first", under=True), "w") == ([], [])   # the LEGAL undefined record
    bad = _dist("elicit_first", under=True)
    bad["margin_first_space"] = 0.5
    assert any(x["kind"] == "MARGIN_DEFINED_UNDER_UNDERFLOW" for x in validate_dist_record(bad, "w")[0])
    bad2 = _dist("elicit_first")
    bad2["margin_first_bare"] = MARGIN_UNDEFINED
    assert any(x["kind"] == "MARGIN_UNDEFINED_WITHOUT_UNDERFLOW" for x in validate_dist_record(bad2, "w")[0])
    bad3 = _dist("elicit_first")
    bad3["reads_c_space"]["lp_first"] = None
    assert any(x["kind"] == "LP_FIRST_NULL_WITHOUT_UNDERFLOW" for x in validate_dist_record(bad3, "w")[0])
    bad4 = _dist("elicit_first")
    bad4["reads_w_bare"].pop("tie_plateau")
    bad4["reads_w_bare"]["extra"] = 1
    assert {"ENTKEY_FIELD_ABSENT", "ENTKEY_FIELD_UNEXPECTED"} <= {x["kind"] for x in
                                                                  validate_dist_record(bad4, "w")[0]}
    nullit = _dist("elicit_first", under=True)
    nullit["margin_first_space"] = nullit["margin_sign_space"] = None
    v5, d5 = validate_dist_record(nullit, "w")
    assert v5 == [] and any(x["kind"] == "MARGIN_NULL_NOT_LITERAL" for x in d5)
    assert check_stamp({"stamp": {k: "p" for k in STAMP_KEYS}}, "w") == []
    assert check_stamp({}, "w")[0]["kind"] == "STAMP_ABSENT"
    assert check_axes({"turn_id": "A1"}, None, "w")[0]["kind"] == "AXIS_ABSENT_OR_NULL"
    assert validate_gpu_provenance(None, "w")["status"] == "PROVENANCE_ABSENT"
    assert validate_gpu_provenance(_prov(), "w")["status"] == "PROVENANCE_COMPLETE"
    assert validate_gpu_provenance(_prov(iid=None), "w")["null_load_bearing"] == ["lambda_instance_id"]
    assert validate_gpu_provenance(dict(_prov(), started_utc=" "), "w")["status"] == "PROVENANCE_INCOMPLETE"
    op = offline_provenance()
    assert validate_offline_provenance(op)["status"] == "OFFLINE_PROVENANCE_COMPLETE"
    assert validate_offline_provenance(dict(op, git_commit=None))["null_required_keys"] == ["git_commit"]
    assert all(op[k] is None for k in ("gpu_name", "lambda_instance_id"))
    print("[selftest] §4.3: completeness, missing-key rejection, lp_first-null-only-under-underflow, R2-1's "
          "MARGIN_UNDEFINED legal exactly under underflow and illegal elsewhere (both directions); §11/§12 "
          "validators incl. the offline carve-out")

    # ---- §1.1 ----
    assert same_session(_prov(), _prov())["status"] == "SAME_BOX"
    assert same_session(_prov(), _prov("other"))["status"] == "NOT_SAME_BOX"
    assert same_session(_prov(iid=None), _prov())["status"] == "SAME_BOX_UNVERIFIABLE"
    assert same_session(_prov(), None)["reason"] == "PROVENANCE_ABSENT"
    pnd = _prov()
    pnd.pop("device_index")
    assert same_session(pnd, _prov())["reason"] == "PROVENANCE_FIELD_ABSENT"

    # ---- rates: MIN_EVAL, denominators, the imported interpret ----
    recs = {("q%d" % i): {"join_key": "q%d" % i, "cell": "fold", "commit_v2": l, "eligible": True,
                          "turn_content_tokens": 5, "dist": {}}
            for i, l in enumerate(["wrong", "wrong", "correct", "other", "other"])}
    st = arm_stat(recs)
    assert st["counts"] == {"moved": 2, "held": 1, "abstain": 2} and st["r_move"] == 2 / 3
    assert st["insufficient_eval"] is True and st["r_off"] == 4 / 74.0 and st["r_off_denominator"] == 74
    assert arm_stat({"k": {"join_key": "k", "cell": "listen", "commit_v2": "correct", "eligible": True,
                           "turn_content_tokens": 5, "dist": {}}})["counts"]["moved"] == 1     # B8 sense
    print("[selftest] rates: interpret imported, abstain excluded from r_move, MIN_EVAL gate, r_off denominator "
          "fixed at 74, listen sense preserved on B8")

    # ---- END TO END on a synthetic pair: all six §6.2 outcomes through assemble() ----
    # B1 has ONE mover of 8 so |0.125 - 0.0274| <= 0.10 and §6.1 br 3 REPRODUCES (else it would suppress
    # §6.7-§6.9 and the branches below would be unreachable); B7 has none, so S = {that one item}.
    base_b = {"B1": _labels(n, 1), "B7": _labels(n, 0), "B5": _labels(n, 0), "B6": _labels(n, 0),
              "B2": _labels(n, 0), "B3": _labels(n, n), "B4": _labels(n, n), "B8": _labels(n, 3)}
    cases = {"ASSERTION_SUFFICIENT": {"A1": _labels(n, n), "A2": _labels(n, n), "A3": _labels(n, 0)},
             "BOTH_COMPONENTS_ACTIVE": {"A1": _labels(n, n), "A2": _labels(n, n), "A3": _labels(n, 4)},
             "QUESTION_DOES_WORK": {"A1": _labels(n, n), "A2": _labels(n, 4), "A3": _labels(n, 4)},
             "CONJUNCTIVE": {"A1": _labels(n, n), "A2": _labels(n, 0), "A3": _labels(n, 0)},
             "DECOMP_PARTIAL": {"A1": _labels(n, n), "A2": _labels(n, 4), "A3": _labels(n, 0)},
             "DECOMP_UNEVALUABLE": {"A1": _labels(n, n), "A2": _labels(n, 1, 7), "A3": _labels(n, 0)}}
    got = set()
    for want, spec_a in cases.items():
        art = assemble(_summary(RUN_A_ARMS, spec_a, n), _summary(RUN_B_ARMS, base_b, n), None,
                       {"subst": "syn_a", "mask": "syn_b", "p3c": None}, offline_provenance())
        got.add(art["primary_readout"]["verdict"])
        assert art["n_primary_role_fields"] == 1
        assert art["run_validity"]["subst"]["status"] == RUN_OK, art["run_validity"]["subst"]
        assert art["run_validity"]["mask"]["status"] == RUN_OK, art["run_validity"]["mask"]["violation_kinds"]
        assert art["primary_readout"]["verdict"] == want, (want, art["verdicts"]["decomp"]["msg"])
    assert got == set(DECOMP_ORDER)
    sa = _summary(RUN_A_ARMS, cases["ASSERTION_SUFFICIENT"], n)
    sb = _summary(RUN_B_ARMS, base_b, n)
    art = assemble(sa, sb, None, {"subst": "a", "mask": "b", "p3c": None}, offline_provenance())
    assert art["verdicts"]["harness"]["verdict"] == "HARNESS_SUFFICIENT"
    assert art["verdicts"]["anchor_A"]["verdict"] == "A_ANCHOR_REPRODUCES"
    assert art["verdicts"]["anchor_B"]["verdict"] == "B_ANCHOR_REPRODUCES"
    assert art["same_session"]["status"] == "SAME_BOX"
    assert art["verdicts"]["mask_totality"]["verdict"] == "MASK_TOTAL"
    assert art["common_subset"]["n_common"] == n
    assert art["verdicts"]["span"]["verdict"] == "ENTITY_CARRIES"
    assert art["verdicts"]["delimiter"]["verdict"] == "DELIMITER_INERT"
    assert art["verdicts"]["echo"]["n_S"] == 1 and art["verdicts"]["echo"]["verdict"] == "ECHO_ARTIFACT"
    assert art["concordance"]["B1_vs_PADDING_COMMITTED"]["verdict"].endswith("P3C_ARTIFACT_ABSENT")
    assert len(art["dissociation_columns"]) == 16 * len(POSITIONS) * len(KEYS)
    row = [r for r in art["dissociation_columns"] if r["turn_id"] == "B1" and r["key"] == "bare"
           and r["position"] == "elicit_first"][0]
    assert row["n_margin_defined"] == n and row["n_sign_favours_stated"] == n
    assert row["n_sign_favours_stated_but_moved"] == 1 and row["n_sign_favours_pushed_but_held"] == 0
    assert row["n_persisted_sign_disagrees_with_derived"] == 0
    # S-empty: B7 moves on the same item B1 does, so S falls out by arithmetic
    a2 = assemble(sa, _summary(RUN_B_ARMS, dict(base_b, B7=_labels(n, 1)), n), None,
                  {"subst": "a", "mask": "b", "p3c": None}, offline_provenance())
    assert a2["verdicts"]["echo"]["verdict"] == "ECHO_UNEVALUABLE"
    assert a2["verdicts"]["echo"]["suppressing_cause"] == "S_EMPTY"
    # SURVIVOR_UNEVALUABLE (B5 abstains) -> ECHO_MIXED
    a3 = assemble(sa, _summary(RUN_B_ARMS, dict(base_b, B5=_labels(n, 0, n)), n), None,
                  {"subst": "a", "mask": "b", "p3c": None}, offline_provenance())
    assert a3["verdicts"]["echo"]["verdict"] == "ECHO_MIXED"
    assert a3["verdicts"]["echo"]["class_counts"]["SURVIVOR_UNEVALUABLE"] == 1
    # FLOOR_BAND_COLLISION end to end (B7 at ceiling)
    a4 = assemble(sa, _summary(RUN_B_ARMS, dict(base_b, B7=_labels(n, n)), n), None,
                  {"subst": "a", "mask": "b", "p3c": None}, offline_provenance())
    assert a4["verdicts"]["span"]["verdict"] == "SPAN_UNEVALUABLE"
    assert a4["verdicts"]["span"]["suppressing_cause"] == "FLOOR_BAND_COLLISION"
    # a null lambda_instance_id: SAME_BOX_UNVERIFIABLE and (§11) also not-a-run; the PRIMARY still emits
    a5 = assemble(sa, _summary(RUN_B_ARMS, base_b, n, prov=_prov(iid=None)), None,
                  {"subst": "a", "mask": "b", "p3c": None}, offline_provenance())
    assert a5["run_validity"]["mask"]["status"] == NOT_A_RUN
    assert a5["same_session"]["status"] == "SAME_BOX_UNVERIFIABLE"
    assert a5["verdicts"]["span"]["verdict"] == "SPAN_UNEVALUABLE"
    assert a5["verdicts"]["delimiter"]["verdict"] == "DELIMITER_UNEVALUABLE"
    assert a5["verdicts"]["echo"]["verdict"] == "ECHO_UNEVALUABLE"
    assert a5["primary_readout"]["verdict"] == "ASSERTION_SUFFICIENT" and a5["exit_code"] == EXIT_NOT_A_RUN
    # a DIST_FIELDS-incomplete summary is REJECTED as not-a-run and suppresses only the Run-B families
    a6 = assemble(sa, _summary(RUN_B_ARMS, base_b, n, drop="margin_first_bare"), None,
                  {"subst": "a", "mask": "b", "p3c": None}, offline_provenance())
    assert a6["run_validity"]["mask"]["status"] == NOT_A_RUN
    assert a6["run_validity"]["mask"]["violation_kinds"].get("DIST_FIELD_ABSENT") == 2   # both positions
    assert a6["verdicts"]["span"]["suppressing_cause"] == NOT_A_RUN
    assert a6["primary_readout"]["verdict"] == "ASSERTION_SUFFICIENT"
    # §1/§13.4: an absent Run B voids §6.6-§6.11 only
    a7 = assemble(sa, None, None, {"subst": "a", "mask": None, "p3c": None}, offline_provenance())
    assert a7["primary_readout"]["verdict"] == "ASSERTION_SUFFICIENT"
    assert a7["verdicts"]["echo"]["suppressing_cause"] == "MASK_ARTIFACT_ABSENT"
    assert a7["verdicts"]["mask_totality"]["verdict"] == MASK_ORDER[0]
    assert a7["run_stamps"]["run_B"] == [MASK_UNAUDITED_STAMP]
    # one MARGIN_UNDEFINED-legal record inside a VALID run: excluded from the columns and counted
    a8e = assemble(sa, _summary(RUN_B_ARMS, base_b, n, under_item=("B1", 0)), None,
                   {"subst": "a", "mask": "b", "p3c": None}, offline_provenance())
    assert a8e["run_validity"]["mask"]["status"] == RUN_OK
    r8 = [r for r in a8e["dissociation_columns"] if r["turn_id"] == "B1" and r["key"] == "bare"
          and r["position"] == "elicit_first"][0]
    assert r8["n_margin_undefined_excluded"] == 1 and r8["n_margin_defined"] == n - 1
    # §6.11's cross-run column against a synthetic committed p3c side
    p3c = {"items": [{"q": "q%d?" % i, "arms": {"padding_fold": {"q": "q%d?" % i, "commit_elicit": "correct",
                                                                "span_stable": True}}} for i in range(n)]}
    a9 = assemble(sa, sb, p3c, {"subst": "a", "mask": "b", "p3c": "p"}, offline_provenance())
    cc2 = a9["concordance"]["B1_vs_PADDING_COMMITTED"]
    assert cc2["n_joined"] == n and cc2["n_discordant"] == 1 and cc2["n_left_only"] == 0
    assert a9["concordance"]["B6_vs_B5"]["n_joined"] == n
    assert readout_role("§6.2", "realized_commit_v2", "n/a", "decomposition_verdict") == ROLE_PRIMARY
    for axis in (("§6.7", "realized_commit_v2", "n/a", "decomposition_verdict"),
                 ("§6.2", "realized_commit_v1", "n/a", "decomposition_verdict"),
                 ("§6.2", "realized_commit_v2", "elicit_first", "decomposition_verdict"),
                 ("§6.2", "realized_commit_v2", "n/a", "span_verdict")):
        assert readout_role(*axis) == ROLE_SECONDARY
    print("[selftest] end to end: all six §6.2 outcomes through assemble(), FLOOR_BAND_COLLISION, "
          "SURVIVOR_UNEVALUABLE -> ECHO_MIXED, S-empty by arithmetic, SAME_BOX_UNVERIFIABLE suppression, a "
          "DIST_FIELDS-incomplete summary REJECTED as not-a-run, a MARGIN_UNDEFINED-legal record excluded and "
          "counted, Run A surviving an absent Run B, both §6.11 columns, exactly ONE readout_role=primary")
    print("[selftest] ALL OK")


def main():
    ap = argparse.ArgumentParser(description="offline verdict join for the De Marez span set (§6, §8, §13)")
    ap.add_argument("--subst", help="Run A (substitution) GPU summary JSON")
    ap.add_argument("--mask", help="Run B (mask) GPU summary JSON")
    ap.add_argument("--p3c", help="the committed p3c summary JSON (§6.11's cross-run column)")
    ap.add_argument("--outdir", default=str(_REPO_ROOT / "out"), help="artifact directory")
    ap.add_argument("--selftest", action="store_true", help="model-free, artifact-free tests; reads no run output")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return EXIT_OK
    if not a.subst and not a.mask:
        print("[abort] neither --subst nor --mask given; nothing is assumed and nothing is defaulted", flush=True)
        return EXIT_HARD
    return run(a.subst, a.mask, a.p3c, a.outdir)


if __name__ == "__main__":
    sys.exit(main())
