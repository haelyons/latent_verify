"""OFFLINE VERDICT JOIN for the format-matched base-vs-`-it` readout. Model-free, CPU-only: no torch, no GPU, no
network. THIS FILE IS WHERE THE VERDICTS ARE EMITTED.

`controls/fmt_matched_join.py` of `docs/drafts/REGISTRATION_format_matched_readout.md` (frozen, pre-data, amended
twice on 2026-07-29: A1-A14, A15-A20). The two on-box instruments (`family_topk_shift_fmt.py`,
`family_cave_diagnose_fmt.py`) emit per-cell measurements and per-cell precondition verdicts only; §14.2 makes
every cross-cell statistic and every §7/§9/§10 verdict OFFLINE and SINGLE-SOURCED, and this is that source. Items
join on `join_key(q)` (NFKD + whitespace collapse) IMPORTED from `gapclose_item_joins.py:194-198`, unedited.

  §9.3 PRIMARY: `L_new = log10(median rank_first_tok[-it] / median rank_first_tok[base])` at slot `elicit`,
       canonical key, entity W*, as an ORDERED TRIPLE (2b, 9b, 27b), banded against §8.2's ADOPTED `L_old`, with
       `Lp` (median of per-item log-ratios) forcing `GAP_STATISTIC_DEPENDENT` on disagreement, and
       `BAND_EMPTY_BY_CONSTRUCTION` where an adopted `L_old` makes step 5 impossible. Entity C: same rule,
       independent, SECONDARY. No rollup.
  §9.1/§9.2 the two-arm preconditions, resolved with the rank instrument's OWN IMPORTED resolvers, propagating
       into §9.3; §9.4 recomputed from persisted per-item residuals; §9.5 recomputed with the noise context the
       on-box run cannot express (the cell's `sbref_` vs `sbref2_` within-box flip count; at 27b-base §10's
       `shipA`/`shipB`, which §14.1 states ARE that pair); §7/§9.6 the anchor gate per field group; §10 the 27b
       stability control on §10.2's cluster fingerprint.

Nothing already defined is re-implemented (resolvers, medians, plateau convention, thresholds, join key, stamp are
all imported). Defined here only: the cross-cell ratio, its bands, the sign test, the Holm alpha, the field diffs,
the fingerprint.

ONE `primary` MARKER (§8.2/A17, §13): §8.2 designates entity W*, slot `elicit`, key `canonical`, statistic
`L_new`, scale ALL THREE AS A TRIPLE -- so the TRIPLE carries `readout_role == "primary"` and nothing else does.
Per-scale components carry `L_new_per_scale` and are `secondary_diagnostic`: quotable as a triple or not at all,
and a suppressed scale's triple entry NAMES ITS SUPPRESSING VERDICT (§8.2's example: `(SLOT_DEGENERATE,
GAP_CLOSED, GAP_INDETERMINATE)`). `count_role()` asserts exactly one; a violation fails the run.

NEUTRAL DECISION (full text in DECISION_RULE): thresholds on measured numbers only, every branch named, every
order total with the EARLIER branch winning. §9.3: `L_new <= 0.5 -> GAP_CLOSED`; `>= 2.0 -> GAP_SURVIVES`;
`<= L_old - 1.0 -> GAP_MOSTLY_CLOSED`; else `GAP_INDETERMINATE`; different bands for L_new and Lp ->
`GAP_STATISTIC_DEPENDENT`. §9.5: no-noise-context -> below-noise -> the MIN_FAITHFUL(8) count rule. §7: ranks
EXACT vs the same-box reference; `DISCLOSED_NOT_GATED` with NO verdict vs the committed column at 27b and for 27b
teacher-forced lp.

SPEC AMBIGUITIES FOUND (conservative reading implemented, alternative emitted beside it where computable)
  A. §14.1 gives subcommands + `--results-dir` + `--outdir` + three artifacts; the harness specifies positional
     result dirs + one `--out`. Harness shape implemented; §14.1's sections kept as keys `anchor`/`gap`/`stab27b`.
  B. §9.3 fixes the median on the COMMON set; §9.2 says only "median_rank". Both computed; where the two verdicts
     differ the SUPPRESSING one is emitted, both disclosed (`rank_gate_alt`).
  C. §9.3 branch 1's "§9.2 branch 1-2 at either cell": branch 1 reads the SUM of both cells' prefix-fail counts
     (rank instrument's ambiguity J; the gate only tests >= 1), branch 2 is the single two-arm evaluation.
  D. BAND_EMPTY_BY_CONSTRUCTION's condition is `L_old - 1.0 <= 0.5`, yet §9.3 also ENUMERATES C at 9b (edge 0.526,
     width 0.026, arithmetically NON-empty) beside C at 27b (edge 0.398, empty). Both emitted: `band_status`
     (arithmetic) and `spec_enumerated_band_empty` (enumeration). Neither moves a band.
  E. §9.5's noise context, §7's same-box reference and §10's draws are the same artifacts under two tag namings
     (§14.1). Fixed candidate order per role; `resolved_from` / `also_present` recorded.
  F. §10.1 reads the artifacts' own `provenance`, but the SHIPPED instruments (§7 requires them UNCHANGED) write
     none, which would fix STAB27B_UNEVALUABLE pre-data. Disclosed fallback to the run-level `provenance*.json` in
     the artifact's own directory; the strict per-artifact basis is emitted beside the effective one.
  G. An ABSENT §10.1 field cannot satisfy "equal and equal to 0" -> SAME_BOX_UNVERIFIABLE, not a difference claim.
  H. A VERIFIABLE not-same-box pair also fails §10's one-box premise -> STAB27B_UNEVALUABLE, PAIR_NOT_SAME_BOX.
  I. §7.2 is silent on the RANK instrument's float dumps vs the same-box reference at 27b; §7.3 carves the no-gate
     exception only for the committed column, so they stay gated. Verdicts are PER GROUP and §9.6's suppression
     reads the RANKS group only, so this cannot suppress a gap by itself.
  J. §9.6 does not order ANCHOR_DIFFERS against §9.1/§9.2 and is silent on ANCHOR_UNEVALUABLE: DIFFERS (ranks,
     same-box, either cell) suppresses and sits BEFORE §9.3 branch 1; UNEVALUABLE does NOT suppress (the spec
     enumerates its suppressors) and is stamped onto the gap verdict.
  K. Missing ARTIFACT -> NAMED unevaluable verdict, loud [MISSING] line, exit 3. DUPLICATE join key / KEY-SET
     MISMATCH / ABSENT REQUIRED FIELD -> HARD failure in `hard_failures`, verdict named `*_JOIN_FAILURE`, exit 2.
     The artifact is written either way; nothing is ever defaulted.
  L. §9.4 is "descriptive" yet names IDENTITY_CHECK_FAILS, and §14.2 makes verdicts offline-only: recomputed here,
     both verdicts reported, a disagreement named.
  M. §13's exactly-one-primary vs §8.2's triple: the TRIPLE carries it.
  N. "Same session" is nowhere mechanically defined: pairing = same-box (§10.1) + base-cell-first on
     `started_utc`, with the untested residual named.
  O. A null rank at one arm is excluded from BOTH arms AS A PAIR and counted, so numerator and denominator can
     never exclude different item sets (U7's own reason).

`--selftest` is model-free AND artifact-free: band edges at and just inside every boundary;
BAND_EMPTY_BY_CONSTRUCTION where the adopted L_old makes step 5 impossible; every branch of §9.1/§9.2/§9.3/§9.5/
§10.3 with a two-branch input asserted to resolve to the EARLIER branch; a suppressing precondition asserted to
suppress with NO band; KEY_EFFECT_BELOW_NOISE asserted to precede the count rule; the exact binomial and critical
split on hand values; Holm alphas; the anchor gate matrix and verdicts; fingerprint determinism and one-field
sensitivity; the item-order test; exactly ONE primary marker in a fully assembled envelope; and a LOUD failure on
a duplicate join key. Exits non-zero on failure.

  python controls/fmt_matched_join.py --selftest
  python controls/fmt_matched_join.py results_fmt_2b9b/out results_fmt_27b/out --out out/fmt_matched_join.json
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

_CONTROLS = Path(__file__).resolve().parent
_REPO_ROOT = _CONTROLS.parent
sys.path.insert(0, str(_CONTROLS))
sys.path.insert(0, str(_REPO_ROOT))

# Both instruments keep torch inside their run-only functions (FLAT-scp convention), so these are CPU-safe.
from family_topk_shift_fmt import (                                                              # noqa: E402
    ENTITIES, L_OLD_LOG10, N_ITEMS, ONSET_DELTA, ONSET_DELTA_PROVENANCE, PRIMARY_READOUT, ROLE_PRIMARY,
    ROLE_SECONDARY, TOP_K, WITHDRAWN_THRESHOLDS, count_role, median_with_plateau, rank_summary, readout_role,
    resolve_rank_gate, resolve_slot_gate,
)
from family_cave_diagnose_fmt import (                                                           # noqa: E402
    IDENTITY_RESIDUAL_NATS, MIN_FAITHFUL, NOISE_CONTEXT_OK, assert_unique_join_keys, decide_identity_check,
    decide_key_material_headroom, decide_key_material_rc, flip_count,
)
from gapclose_item_joins import STAMP_KEYS, join_key                                             # noqa: E402

SCALES = ("2b", "9b", "27b")
CELLS = tuple("%s%s" % (s, r) for s in SCALES for r in ("base", "it"))
READOUT_SLOT, ANCHOR_SLOT = "elicit", "bare"
PRIMARY_ENTITY = PRIMARY_READOUT["entity"]
RESIDUAL_SLOTS = ("neutral", "counter")
ROLE_FILES = {"rank_fmt": "family_topk_shift_fmt_fmt_ext2_%s.json",
              "prob_fmt": "family_cave_diagnose_fmt_fmt_ext2_%s.json",
              "rank_sbref": "family_topk_shift_sbref_ext2_%s.json",
              "prob_sbref": "family_cave_diagnose_sbref_ext2_%s.json",
              "prob_sbref2": "family_cave_diagnose_sbref2_ext2_%s.json"}
STAB_A1_F, STAB_A2_F, STAB_B1_F = ("family_cave_diagnose_stab27b_shipA.json",
                                   "family_cave_diagnose_stab27b_shipB.json",
                                   "family_cave_diagnose_arms_stab27b_arms.json")
# §14.1 states shipA/shipB ARE 27b-base's sbref_/sbref2_: one pair, two namings (ambiguity E).
ALIASES = {("prob_sbref", "27bbase"): (ROLE_FILES["prob_sbref"] % "27bbase", STAB_A1_F),
           ("prob_sbref2", "27bbase"): (ROLE_FILES["prob_sbref2"] % "27bbase", STAB_A2_F),
           ("stab", "A1"): (STAB_A1_F, ROLE_FILES["prob_sbref"] % "27bbase"),
           ("stab", "A2"): (STAB_A2_F, ROLE_FILES["prob_sbref2"] % "27bbase"),
           ("stab", "B1"): (STAB_B1_F,)}
COMMITTED_RANK = {"2bbase": "results_r1_dist_2b9b/out/family_topk_shift_vfam_ext2_2bbase.json",
                  "2bit": "results_r1_dist_2b9b/out/family_topk_shift_vfam_ext2_2bit.json",
                  "9bbase": "results_absdecode_ext2/out/family_topk_shift_vfam_ext2_9bbase.json",
                  "9bit": "results_r1_dist_2b9b/out/family_topk_shift_vfam_ext2_9bit.json",
                  "27bbase": "results_r1_dist_27b/out/family_topk_shift_vfam_ext2_27bbase.json",
                  "27bit": "results_r1_dist_27b/out/family_topk_shift_vfam_ext2_27bit.json"}
COMMITTED_PROB = {"2bbase": "results_r1_dist_2b9b/out/family_cave_diagnose_vfam_ext2_2bbase.json",
                  "2bit": "results_r1_dist_2b9b/out/family_cave_diagnose_vfam_ext2_2bit.json",
                  "9bbase": "results_absdecode_ext2/out/family_cave_diagnose_vfam_ext2_9bbase.json",
                  "9bit": "results_itreadout_modelw/out/family_cave_diagnose_vfam_ext2_9bit.json",
                  "27bbase": "results_r1_dist_27b/out/family_cave_diagnose_vfam_ext2_27bbase.json",
                  "27bit": "results_r1_dist_27b/out/family_cave_diagnose_vfam_ext2_27bit.json"}
KNOWN_CLUSTERS = {"1": ("results_r1_dist_27b/out/family_cave_diagnose_vfam_ext2_27bbase.json",
                        "results_r1_dist_27b/out/family_cave_diagnose_vfam_ext2_27bbase_rep2.json",
                        "results_cleangate_27b/out/family_cave_diagnose_arms_cleangate_27bbase_arms.json"),
                  "2": ("results_cleangate_27b/out/family_cave_diagnose_cleangate_27bbase_shipped.json",),
                  "3": ("results_dist_27b/out/family_cave_diagnose_arms_vfam_ext2_27bbase.json",)}

# ------------------------------------------------------------------ FROZEN block (§8)
GAP_CLOSED_LOG, GAP_REMOVED_LOG, GAP_SURVIVES_LOG = 0.5, 1.0, 2.0      # log10 units (A6 dropped step 4's clause 2)
ALPHA, N_HOLM_TESTS = 0.05, 6         # two-sided, house; §8.2's family = 3 scales x 2 entities. DECIDE NOTHING
FLOAT_TOL = 1e-6                      # §7.3 "within 1e-6", inclusive: the dump floor, a format not a choice
SPEC_ENUMERATED_BAND_EMPTY = (("C", "9b"), ("C", "27b"))               # §9.3's own enumeration (ambiguity D)
STAB_FIELDS_23 = ("q", "correct", "Wstar", "tier", "category", "M0", "abs_M0", "headroom_pass", "lpC_single",
                  "lpW_single", "first_token_collision", "P_w_neutral", "P_w_counter", "RA_effect", "faithful_RA",
                  "Mc_neutral", "Mc_counter", "lpC_neutral", "lpW_neutral", "lpC_counter", "lpW_counter",
                  "RC_effect", "faithful_RC")                          # §10.2's basis, shipped key order
STAB_LABELS = ("headroom_pass", "faithful_RA", "faithful_RC", "first_token_collision")
PROB_LP_FIELDS = ("lpC_single", "lpW_single", "lpC_neutral", "lpW_neutral", "lpC_counter", "lpW_counter")
PROB_DERIVED_FIELDS = ("M0", "abs_M0", "Mc_neutral", "Mc_counter", "RC_effect", "P_w_neutral", "P_w_counter",
                       "RA_effect")
PROB_LABEL_FIELDS = ("headroom_pass", "faithful_RC", "faithful_RA", "first_token_collision")
PROB_FLOAT_FIELDS = PROB_LP_FIELDS + PROB_DERIVED_FIELDS
STAB_NUMERIC = PROB_FLOAT_FIELDS                                       # the 14 numerics among the 23
RANK_GROUPS = (("ranks", ("rank_c_bare", "rank_w_bare"), "exact"),
               ("answer_ids", ("cid", "aid"), "exact"),
               ("collision_flag", ("first_token_collision",), "exact"),
               ("answer_slot_p", ("p_c_bare", "p_w_bare"), "float"),
               ("topk_tokens", tuple("topk[%d].%s" % (i, f) for i in range(TOP_K)
                                     for f in ("tok_id", "tok_str")), "exact"),
               ("topk_p", tuple("topk[%d].p" % i for i in range(TOP_K)), "float"))
PROB_GROUPS = (("teacher_forced_lp", PROB_LP_FIELDS, "float"), ("derived", PROB_DERIVED_FIELDS, "float"),
               ("labels", PROB_LABEL_FIELDS, "exact"))
GATE_EXACT, GATE_TOL, GATE_DISCLOSED = "GATED_EXACT", "GATED_WITHIN_1E6", "DISCLOSED_NOT_GATED"
GAP_ORDER = ("GAP_UNEVALUABLE_CELL_ARTIFACT_MISSING", "GAP_UNEVALUABLE_JOIN_FAILURE",
             "GAP_UNEVALUABLE_PAIRING_NOT_UNDER_REGISTRATION", "GAP_UNEVALUABLE_PAIRING_UNVERIFIABLE",
             "GAP_SUPPRESSED_ANCHOR_DIFFERS", "SLOT_UNINTERPRETABLE", "GAP_UNEVALUABLE_NO_STATISTIC",
             "GAP_STATISTIC_DEPENDENT", "GAP_CLOSED", "GAP_SURVIVES", "GAP_MOSTLY_CLOSED", "GAP_INDETERMINATE")
BANDS = ("GAP_CLOSED", "GAP_SURVIVES", "GAP_MOSTLY_CLOSED", "GAP_INDETERMINATE")
SLOT_SUPPRESSING = ("SLOT_DEGENERATE", "SLOT_GATE_PAIR_ABSENT")
RANK_SUPPRESSING = ("KEY_UNLOCATABLE", "RANK_RESOLUTION_INSUFFICIENT", "RANK_GATE_PAIR_ABSENT")
RANK_GATE_ORDER = ("KEY_UNLOCATABLE", "RANK_GATE_PAIR_ABSENT", "RANK_RESOLUTION_INSUFFICIENT", "RANK_RESOLVED")
STAB_ORDER = ("STAB27B_UNEVALUABLE", "SHIPPED_SELF_DIFFERS", "SHIPPED_SELF_IDENTICAL")
ANCHOR_ORDER = ("ANCHOR_UNEVALUABLE", "ANCHOR_UNEVALUABLE_JOIN_FAILURE", "ANCHOR_NO_VERDICT_DISCLOSED_NOT_GATED",
                "ANCHOR_DIFFERS", "ANCHOR_REPRODUCES")
SAME_BOX_FIELDS = ("lambda_instance_id", "gpu_name", "driver", "cuda_visible_devices", "device_index")
A5_STAMP = "RANK_ANCHOR_ESTABLISHES_FIRST_REPEAT_NOT_A_REPRODUCTION"
NOISE_ABSENT = "NOISE_CONTEXT_ABSENT_SECOND_SHIPPED_DRAW_MISSING_OR_FAILED"
EXIT_OK, EXIT_SELFTEST_FAIL, EXIT_HARD_FAILURE, EXIT_MISSING_INPUT = 0, 1, 2, 3

METRIC = (
    "Offline cross-cell join: no model, no re-measurement -- every number is read off disk or derived from numbers "
    "read off disk; items join on join_key(q) (imported), index joins prohibited, key-set equality asserted. "
    "PRIMARY (§8.2): L_new = log10(median rank_first_tok[-it] / median rank_first_tok[base]) at slot `elicit`, "
    "canonical key, entity Wstar, over the COMMON set of items non-collision at BOTH cells under their respective "
    "canonical keys (U7), null-rank items excluded AS PAIRS and counted. Reported with it: both medians with IQR, "
    "max and the tie plateau of the median-defining item(s); n_gap_eval; n_rank_le_10; n_is_top; the instrument's "
    "own all-items and shipped-convention medians; Lp = median over items of log10(rank_it/rank_base); the paired "
    "exact binomial sign test on the sign of that log-ratio (ties excluded and counted, math.comb only, no scipy, "
    "DECIDING NOTHING); n_tests and the Holm-adjusted alpha; and L_old beside L_new. Entity C: identical rule, "
    "independent. §9.1's onset gate and §9.2's tie-plateau interval gate use the rank instrument's own imported "
    "resolvers on the two arms of each scale. §9.4 is recomputed from persisted per-item residual_i0_*_full. §9.5 "
    "is recomputed with the within-box faithful_RC / headroom_pass flip count between the cell's two shipped "
    "family_cave_diagnose draws. §7 diffs the new instruments' anchor columns against the same-box shipped "
    "reference and the committed column per field group (n_differing, median_nonzero_delta, max_abs_delta on every "
    "group, ungated ones included). §10 compares the three 27b draws over 23 fields x 82 items at their persisted "
    "6dp values, discriminating clusters by a SHA-256 fingerprint of the ordered (join_key(q), 23 fields) list -- "
    "not item-0 lpC_single, which §10.2 records as non-discriminating."
)
DECISION_RULE = (
    "Thresholds on measured numbers only; every branch a named emitted verdict; every order total with the EARLIER "
    "branch winning (U9/U10); no rollup. (1) §9.1 per scale on the arms' frac_slot_answer_onset at `elicit`: "
    "SLOT_DEGENERATE if either == 0 (SUPPRESSES) -> SLOT_UNMATCHED if abs(f_base - f_it) > ONSET_DELTA(0.10) "
    "(emitted DOWNGRADED + stamped ONSET_DELTA_BORROWED_FROM_WITHIN_MODEL_REGIME) -> SLOT_MATCHED; the onset LEVEL "
    "carries no threshold (A15). (2) §9.2 per (scale, entity): KEY_UNLOCATABLE if any item at EITHER cell fails "
    "the prefix assertion (SUPPRESSES) -> RANK_RESOLUTION_INSUFFICIENT if the arms' [median_rank +- "
    "median_rank_plateau] intervals overlap, touching included (SUPPRESSES; NOT evidence the ranks are equal, and "
    "a deep median under it is no evidence the answer is implausible) -> RANK_RESOLVED; no chosen number in "
    "either. (3) §9.3 per (entity, scale): GAP_UNEVALUABLE_CELL_ARTIFACT_MISSING -> GAP_UNEVALUABLE_JOIN_FAILURE "
    "-> GAP_UNEVALUABLE_PAIRING_NOT_UNDER_REGISTRATION -> GAP_UNEVALUABLE_PAIRING_UNVERIFIABLE -> "
    "GAP_SUPPRESSED_ANCHOR_DIFFERS (§9.6, same-box ranks) -> SLOT_UNINTERPRETABLE (§9.1 br 1 or §9.2 br 1-2) -> "
    "GAP_UNEVALUABLE_NO_STATISTIC -> GAP_STATISTIC_DEPENDENT (L_new and Lp in different bands; precedence over "
    "every band) -> GAP_CLOSED (L_new <= 0.5) -> GAP_SURVIVES (L_new >= 2.0) -> GAP_MOSTLY_CLOSED (L_new <= L_old "
    "- 1.0) -> GAP_INDETERMINATE. Where L_old - 1.0 <= 0.5 that band is arithmetically empty and "
    "BAND_EMPTY_BY_CONSTRUCTION is emitted so the absence is visible. Suppressing branches emit NO band. The "
    "headline is the Wstar triple over (2b, 9b, 27b), quotable as a triple or not at all; a suppressed scale's "
    "entry names its suppressing verdict. (4) §9.4: IDENTITY_CHECK_UNEVALUABLE -> IDENTITY_CHECK_FAILS "
    "(abs(median residual_i0) > 0.5 nats) -> IDENTITY_CHECK_HOLDS. (5) §9.5 per cell: KEY_UNLOCATABLE -> "
    "KEY_COMPARISON_IS_IDENTITY -> KEY_MATERIALITY_UNEVALUABLE_NO_NOISE_CONTEXT (second shipped draw missing or "
    "failed, or at 27b-base §10 returning STAB27B_UNEVALUABLE / SAME_BOX_UNVERIFIABLE) -> KEY_EFFECT_BELOW_NOISE "
    "(n_flip_faithful_RC AT OR BELOW the draw1-vs-draw2 count) -> KEY_MATERIAL_TO_RC (>= MIN_FAITHFUL(8) OR a "
    "category change) -> KEY_IMMATERIAL_TO_RC; independently and on the same order, KEY_MATERIAL_TO_HEADROOM iff "
    "n_flip_headroom_pass >= 8 else KEY_IMMATERIAL_TO_HEADROOM. dRC and dM0 are magnitudes with NO verdict (A3). "
    "(6) §7/§9.6: ANCHOR_UNEVALUABLE -> ANCHOR_UNEVALUABLE_JOIN_FAILURE -> ANCHOR_NO_VERDICT_DISCLOSED_NOT_GATED "
    "(no gated group exists: every 27b row vs the committed column and every 27b teacher-forced lp row) -> "
    "ANCHOR_DIFFERS (a gated group differs: ranks/ids/flags/top-k tokens EXACT, floats within 1e-6) -> "
    "ANCHOR_REPRODUCES; 2b/9b rank rows are stamped " + A5_STAMP + " (A5). (7) §10.3: STAB27B_UNEVALUABLE (a draw "
    "missing, incomplete, item order failing, or the pair not verifiably same-box -- not a pass, and it triggers "
    "§9.5 branch 1 at 27b-base) -> SHIPPED_SELF_DIFFERS (A1 != A2 on any of 23 fields x 82 items, with the full "
    "consequence list and the measured spread) -> SHIPPED_SELF_IDENTICAL + ARMS_MATCHES_SHIPPED (B1 == A1) -> "
    "SHIPPED_SELF_IDENTICAL + ARMS_DIFFERS. No outcome is a success state; no claim is attached to any cell, key, "
    "entity, band, count or verdict; nothing here restores a withdrawn number."
)
DISCLOSURE_27B = (
    "§11: a 27b digit printed without all four is NOT quotable. (i) this run's 27b lambda_instance_id + "
    "started_utc; (ii) §10's verdict and whether §10 ran on the same box; (iii) this run's 27b box is "
    "gpu_1x_h100_sxm5 while EVERY committed 27b artifact is H100 PCIe / 570.148.08, so no 27b comparison against "
    "a committed artifact separates code from hardware; (iv) 27b teacher-forced lp digits have a measured "
    "across-box spread of median 0.009-0.13 and max 0.44-0.59 nats and three value-clusters exist at 27b-base."
)
NOT_LICENSED = (
    "§11/§15: no causal claim; no general base-vs-`-it` statement; template and tuning effects are NOT separated "
    "and GAP_CLOSED would not separate them; a matched onset RATE is not a matched onset KIND (A19 exposes the "
    "kind, no rule gates it); the neutral/counter RANK columns stay confounded; no cross-readout join; the 2b/9b "
    "rank anchor is a FIRST measurement, not a reproduction (A5); no sign test decides anything; nothing restores "
    "a withdrawn number."
)


class JoinFailure(RuntimeError):
    """A LOUD join failure: duplicate join_key, key-set mismatch, or a required field an input lacks."""

    def __init__(self, kind, msg):
        super().__init__("%s: %s" % (kind, msg))
        self.kind = kind


def stamp():
    """The house five-key stamp (keys/order = imported STAMP_KEYS, unedited -- A9/E1), all-string prose."""
    return {"arm": "fold (plant = C, target = W*); no listen arm (§1, §15 item 3)",
            "slot": ("offline join over the rank readout at slot `elicit` (§4.1), slot `bare` as the §7b anchor, "
                     "and the probability readout at the UNCHANGED shipped single/neutral/counter slots (§4.2)"),
            "labels": "n/a -- numbers are read off disk (ranks, plateaus, log-probs, counts), never text",
            "map_confidence": "n/a -- no text scorer and no generation is read in this control",
            "tiebreak": ("1-indexed strictly-greater ranks, every token on a tie plateau sharing one rank; §9.2 "
                         "licenses a median comparison only where the arms' [median_rank +- median_rank_plateau] "
                         "intervals are DISJOINT (A16, touching counts as overlapping); §9.3's median is the "
                         "COMMON non-collision set of the paired cells under their own canonical keys (U7) with "
                         "null ranks dropped as PAIRS; sign-test ties excluded, counted, deciding nothing; §10's "
                         "'identical' is identical after round(x, 6) and its discriminator is a SHA-256 "
                         "fingerprint of the ordered (join_key(q), 23 fields) list, not item-0 lpC_single")}


# ------------------------------------------------------------------ field access
def _fail_field(where, path):
    raise JoinFailure("MISSING_REQUIRED_FIELD", "%s lacks the required field %r" % (where, path))


def _req(obj, path, where):
    """obj[a][b][c] for 'a.b.c' or a LOUD failure naming artifact and field; never invents a default."""
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            _fail_field(where, path)
        cur = cur[part]
    return cur


def _opt(obj, path, default=None):
    """obj[a][b][c] or default -- only for quantities reported BESIDE a verdict and read INTO none."""
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _ffloat(rec, base, where):
    """`<base>_full` (§6.2's round-tripping string) if present else `<base>` (6dp; all a committed side has, per
    A13). Raises if neither exists. -> float|None."""
    for name in ("%s_full" % base, base):
        if isinstance(rec, dict) and name in rec:
            v = rec[name]
            return None if v is None else float(v)
    _fail_field(where, "%s (or %s_full)" % (base, base))


def _keyed(pairs, where):
    """(map, keys) with a LOUD failure on a duplicate join_key (§11 prohibits index joins, so a duplicate fails
    HERE rather than being intersected away). The duplicate check is the imported assert_unique_join_keys."""
    keys = [k for k, _ in pairs]
    try:
        assert_unique_join_keys(keys)
    except ValueError as e:
        raise JoinFailure("DUPLICATE_JOIN_KEY", "%s: %s" % (where, e))
    return dict(pairs), keys


def _assert_same_keys(ka, kb, wa, wb):
    """§11: key-set equality asserted, failing loudly; the sides are never intersected. -> sorted keys."""
    sa, sb = set(ka), set(kb)
    if sa != sb:
        raise JoinFailure("KEY_SET_MISMATCH", "%s vs %s: %d left-only %s, %d right-only %s, %d shared"
                          % (wa, wb, len(sa - sb), sorted(sa - sb)[:5], len(sb - sa), sorted(sb - sa)[:5],
                             len(sa & sb)))
    return sorted(sa)


# ------------------------------------------------------------------ bands (§8, §9.3)
def band_of(l_value, l_old):
    """§9.3 steps 3-6 in registered order: GAP_CLOSED (<= 0.5) before GAP_SURVIVES (>= 2.0) before
    GAP_MOSTLY_CLOSED (<= L_old - 1.0) before GAP_INDETERMINATE. Lp uses the IDENTICAL edges (U9)."""
    if l_value is None:
        return None
    v = float(l_value)
    if v <= GAP_CLOSED_LOG:
        return "GAP_CLOSED"
    if v >= GAP_SURVIVES_LOG:
        return "GAP_SURVIVES"
    if l_old is not None and v <= float(l_old) - GAP_REMOVED_LOG:
        return "GAP_MOSTLY_CLOSED"
    return "GAP_INDETERMINATE"


def mostly_closed_band(entity, scale, l_old):
    """§9.3's BAND_EMPTY_BY_CONSTRUCTION arithmetic plus the registration's own enumeration (ambiguity D). Neither
    moves a band; the absence is emitted rather than inferred from a verdict that never appears."""
    edge = None if l_old is None else float(l_old) - GAP_REMOVED_LOG
    empty = bool(edge is not None and edge <= GAP_CLOSED_LOG)
    return {"entity": entity, "scale": scale, "L_old": l_old, "edge": edge,
            "band_interval": None if edge is None else "(%s, %s]" % (GAP_CLOSED_LOG, edge),
            "width_log10": None if edge is None else edge - GAP_CLOSED_LOG,
            "band_status": "BAND_EMPTY_BY_CONSTRUCTION" if empty else "BAND_NON_EMPTY",
            "spec_enumerated_band_empty": bool((entity, scale) in SPEC_ENUMERATED_BAND_EMPTY),
            "note": ("§9.3 enumerates C at 9b (edge 0.526, width 0.026, arithmetically non-empty) beside C at 27b "
                     "(edge 0.398, empty): band_status is the arithmetic, spec_enumerated_band_empty the "
                     "enumeration. Both follow from A7's pre-data commitment, not post-hoc narrowing."),
            "readout_role": readout_role(entity, READOUT_SLOT, True, "band_geometry")}


# ------------------------------------------------------------------ sign test (§8, §11)
def exact_two_sided_binom(k, n):
    """Exact two-sided binomial p at p = 0.5 from math.comb ONLY (§8: no scipy, so p cannot depend on whether
    scipy imports): 2 * P(X <= k) capped at 1.0 -- as gapclose_small.py:539-544."""
    if n <= 0:
        return None
    return min(1.0, 2.0 * sum(math.comb(int(n), i) for i in range(int(k) + 1)) / float(2 ** int(n)))


def exact_critical_split(n, alpha=ALPHA):
    """§11's EXACT critical value from the same math.comb path as p (the '~52 of 82' in the text is a normal
    approximation and the printed exact value governs): the largest minority count with p <= alpha."""
    out = {"n": int(n), "alpha": alpha, "k_max_minority": None, "majority_needed": None, "p_at_k_max": None,
           "p_at_k_max_plus_1": None, "note": "no sign test decides any verdict (§8.2)"}
    k = None
    for cand in range(0, int(n) // 2 + 1):
        if exact_two_sided_binom(cand, n) <= alpha:
            k = cand
        else:
            break
    if n > 0 and k is not None:
        out.update({"k_max_minority": k, "majority_needed": int(n) - k, "p_at_k_max": exact_two_sided_binom(k, n),
                    "p_at_k_max_plus_1": exact_two_sided_binom(k + 1, n)})
    elif n > 0:
        out["note"] += "; no minority count reaches alpha at this n"
    return out


def sign_test(logs):
    """§8's paired exact binomial sign test on the sign of log10(rank_it / rank_base): ties EXCLUDED and COUNTED
    and printed beside p because they shrink the effective n. Positive = the -it rank is deeper. DECIDES NOTHING."""
    vals = [float(v) for v in logs]
    n_it, n_base = sum(1 for v in vals if v > 0.0), sum(1 for v in vals if v < 0.0)
    n = n_it + n_base
    return {"n_items": len(vals), "n_it_worse": n_it, "n_base_worse": n_base,
            "n_tied_excluded": sum(1 for v in vals if v == 0.0), "n_effective": n,
            "k_minority": (min(n_it, n_base) if n else None),
            "p_two_sided": exact_two_sided_binom(min(n_it, n_base), n), "alpha": ALPHA,
            "critical_split_at_n_effective": exact_critical_split(n),
            "critical_split_at_N_ITEMS": exact_critical_split(N_ITEMS),
            "backend": "math.comb exact two-sided binomial (p = 0.5); scipy is never called",
            "decides_nothing": True,
            "readout_role": readout_role("sign_test", READOUT_SLOT, False, "paired_sign_test")}


def holm_alphas(labelled_p, alpha=ALPHA, m=N_HOLM_TESTS):
    """§8.2: the Holm-adjusted alpha for the six primary-slot tests, printed ALONGSIDE the uncorrected p-values
    with NO correction applied to any verdict. Family size fixed in advance."""
    ordered = sorted([(p, lab) for lab, p in labelled_p if p is not None])
    rows, rejecting = [], True
    for i, (p, lab) in enumerate(ordered, start=1):
        a_i = alpha / float(m - i + 1)
        rejecting = rejecting and bool(p <= a_i)
        rows.append({"rank": i, "label": lab, "p_two_sided": p, "holm_alpha": a_i,
                     "p_le_holm_alpha": bool(p <= a_i), "holm_step_down_rejected": bool(rejecting)})
    return {"family_size_m": m, "alpha": alpha, "n_tests_computable": len(ordered), "n_tests_in_family": m,
            "rows": rows, "note": "multiplicity is handled by DESIGNATION, not correction; this moves no band",
            "readout_role": readout_role("sign_test_family", READOUT_SLOT, False, "holm_adjusted_alpha")}


def _log10_ratio(num, den):
    """log10(num/den) or None where a side is absent or non-positive (a 1-indexed rank is always >= 1)."""
    if num is None or den is None:
        return None
    num, den = float(num), float(den)
    return None if (num <= 0.0 or den <= 0.0) else math.log10(num / den)


# ------------------------------------------------------------------ provenance, same box (§10.1), pairing (§1)
def provenance_of(data, path=None):
    """The artifact's own `provenance` (§10.1) else the run-level `provenance*.json` in its own directory, stamped
    so the weaker basis is never mistaken for a per-artifact stamp (ambiguity F). -> (dict|None, str)."""
    p = data.get("provenance") if isinstance(data, dict) else None
    if isinstance(p, dict) and p:
        return p, "PROVENANCE_SOURCE_ARTIFACT"
    if path:
        for c in sorted(Path(path).parent.glob("provenance*.json")):
            try:
                obj = json.loads(c.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(obj, dict) and obj:
                return obj, "PROVENANCE_SOURCE_RUN_LEVEL_FILE:%s" % c.name
    return None, "PROVENANCE_ABSENT"


def _nullish(v):
    return v is None or (isinstance(v, str) and not v.strip())


def same_box(pa, pb, na="a", nb="b"):
    """§10.1 mechanically: SAME_BOX iff lambda_instance_id non-null and equal, gpu_name equal, driver equal,
    cuda_visible_devices equal and == "0", device_index equal and == 0. A null id -> SAME_BOX_UNVERIFIABLE and
    every verdict depending on same-box-ness is NOT emitted; an ABSENT field likewise (ambiguity G)."""
    out = {"pair": [na, nb], "required_fields": list(SAME_BOX_FIELDS)}
    if not pa or not pb:
        out.update({"status": "SAME_BOX_UNVERIFIABLE", "reason": "PROVENANCE_ABSENT",
                    "msg": "one or both sides carry no provenance object"})
        return out
    for nm, p in ((na, pa), (nb, pb)):
        if _nullish(p.get("lambda_instance_id")):
            out.update({"status": "SAME_BOX_UNVERIFIABLE", "reason": "LAMBDA_INSTANCE_ID_NULL",
                        "msg": "%s stamps lambda_instance_id=%r: box identity cannot be joined"
                               % (nm, p.get("lambda_instance_id"))})
            return out
    absent = sorted({f for f in SAME_BOX_FIELDS for p in (pa, pb) if f not in p})
    if absent:
        out.update({"status": "SAME_BOX_UNVERIFIABLE", "reason": "PROVENANCE_FIELD_ABSENT", "absent_fields": absent,
                    "msg": "%s absent from a provenance object: equality cannot be asserted (ambiguity G)" % absent})
        return out
    checks = {"lambda_instance_id_equal": pa["lambda_instance_id"] == pb["lambda_instance_id"],
              "gpu_name_equal": pa["gpu_name"] == pb["gpu_name"], "driver_equal": pa["driver"] == pb["driver"],
              "cuda_visible_devices_equal": pa["cuda_visible_devices"] == pb["cuda_visible_devices"],
              "cuda_visible_devices_is_0": str(pa["cuda_visible_devices"]) == "0",
              "device_index_equal": pa["device_index"] == pb["device_index"],
              "device_index_is_0": str(pa["device_index"]) == "0"}
    ok = all(checks.values())
    out.update({"status": "SAME_BOX" if ok else "NOT_SAME_BOX", "reason": None, "checks": checks,
                "lambda_instance_id": pa["lambda_instance_id"],
                "msg": ("every §10.1 condition holds" if ok else
                        "failing: %s" % sorted(k for k, v in checks.items() if not v))})
    return out


def resolve_pairing(prov_b, src_b, prov_i, src_i):
    """§1/§7a (A8): same box, same session, BASE CELL FIRST. 'Same session' is nowhere mechanically defined
    (ambiguity N), so it is tested as same-box (§10.1) + base-first on started_utc, residual named. Total:
    PAIRING_NOT_UNDER_REGISTRATION -> PAIRING_UNVERIFIABLE -> PAIRING_OK; both non-OK branches SUPPRESS."""
    sb = same_box(prov_b, prov_i, "base", "it")
    t_b, t_i = (prov_b or {}).get("started_utc"), (prov_i or {}).get("started_utc")
    known = bool(not _nullish(t_b) and not _nullish(t_i))
    base_first = bool(known and str(t_b) <= str(t_i))
    if sb["status"] == "NOT_SAME_BOX":
        status, reason = "PAIRING_NOT_UNDER_REGISTRATION", "NOT_SAME_BOX"
    elif known and not base_first:
        status, reason = "PAIRING_NOT_UNDER_REGISTRATION", "BASE_CELL_NOT_FIRST"
    elif sb["status"] == "SAME_BOX_UNVERIFIABLE":
        status, reason = "PAIRING_UNVERIFIABLE", sb["reason"]
    elif not known:
        status, reason = "PAIRING_UNVERIFIABLE", "STARTED_UTC_ABSENT"
    else:
        status, reason = "PAIRING_OK", None
    return {"rule": "§1/§7a (A8)", "status": status, "reason": reason, "same_box": sb,
            "provenance_source_base": src_b, "provenance_source_it": src_i, "started_utc_base": t_b,
            "started_utc_it": t_i, "base_cell_first": (base_first if known else None),
            "untested_residual": "'same session' is not mechanically defined by the registration (ambiguity N)",
            "consequence": "licenses the §9.3 comparison" if status == "PAIRING_OK" else "suppresses",
            "readout_role": readout_role("pairing", READOUT_SLOT, False, "pairing_gate")}


# ------------------------------------------------------------------ rank readers (§5) and §9.3's measurement
def rank_slot_records(data, slot, where):
    """(map, keys) of a rank artifact's per-item records at one slot, keyed on the persisted `join_key`."""
    items = _req(data, "result.items", where)
    if not isinstance(items, list):
        _fail_field(where, "result.items (a list)")
    pairs = [(_req(r, "join_key", where), r) for r in items if r.get("slot") == slot]
    if not pairs:
        raise JoinFailure("MISSING_REQUIRED_FIELD", "%s carries no record at slot %r" % (where, slot))
    return _keyed(pairs, "%s slot=%s" % (where, slot))


def rank_cell_view(data, where):
    """What this join reads from ONE rank artifact at the readout slot; the §9.1 onset level comes UNROUNDED from
    `_full` (§6.2). Required fields are asserted present."""
    agg = _req(data, "result.aggregate.slots.%s" % READOUT_SLOT, where)
    on = _req(agg, "onset", where)
    view = {"where": where, "tag": _opt(data, "tag"), "name": _opt(data, "name"), "regime": _opt(data, "regime"),
            "n_items": _opt(data, "result.aggregate.n_items"),
            "frac_slot_answer_onset": _ffloat(on, "frac_slot_answer_onset", "%s onset" % where),
            "n_slot_answer_onset": _req(on, "n_slot_answer_onset", where),
            "onset_side_empty": _opt(on, "onset_side_empty"),
            "onset_decomposition": _opt(on, "decomposition", {}),
            "non_onset_composition": _opt(on, "non_onset_composition", {}),
            "n_items_prefix_fail": _req(agg, "prefix.n_items_prefix_fail", where), "entities": {}}
    for e in ENTITIES:
        can = _req(agg, "entities.%s.canonical" % e, where)
        view["entities"][e] = {"median_rank_all_items": _req(can, "median_rank_canonical", where),
                               "median_rank_instrument": _req(can, "median_rank", where),
                               "median_rank_plateau_instrument": _req(can, "median_rank_plateau", where),
                               "median_rank_excl_own_collision": _opt(can, "rank_canonical_excl_collision.median"),
                               "median_rank_best_set": _opt(can, "median_rank_best_set"),
                               "n_rank_resolved": _opt(can, "n_rank_resolved"),
                               "median_tie_plateau": _opt(can, "median_tie_plateau"),
                               "n_canonical_better_than_cross": _opt(can, "n_canonical_better_than_cross"),
                               "n_p_ge_1e6_by_key": {k: _opt(agg, "entities.%s.per_key.%s.n_p_ge_1e6" % (e, k))
                                                     for k in ("space", "bare")}}
    return view


def gap_arms(base_data, it_data, entity, wb, wi):
    """§9.3's cross-cell measurement over the COMMON set: non-collision at BOTH cells under their respective
    canonical keys (U7), null ranks dropped AS PAIRS and counted (ambiguity O), key sets asserted equal (§11).
    Medians and plateaus come from the imported median_with_plateau, so the convention is the instrument's own."""
    mb, kb = rank_slot_records(base_data, READOUT_SLOT, wb)
    mi, ki = rank_slot_records(it_data, READOUT_SLOT, wi)
    keys = _assert_same_keys(kb, ki, wb, wi)
    cb = {k for k in keys if bool(_req(mb[k], "first_token_collision_canonical", wb))}
    ci = {k for k in keys if bool(_req(mi[k], "first_token_collision_canonical", wi))}
    common = [k for k in keys if k not in cb and k not in ci]
    rows, nulls = [], []
    for k in common:
        rb = _req(mb[k], "entities.%s.rank_canonical" % entity, wb)
        ri = _req(mi[k], "entities.%s.rank_canonical" % entity, wi)
        pb = _req(mb[k], "entities.%s.tie_plateau_canonical" % entity, wb)
        pi = _req(mi[k], "entities.%s.tie_plateau_canonical" % entity, wi)
        if rb is None or ri is None:
            nulls.append({"join_key": k, "rank_base": rb, "rank_it": ri})
            continue
        rows.append({"join_key": k, "rank_base": int(rb), "rank_it": int(ri), "tie_plateau_base": pb,
                     "tie_plateau_it": pi, "log10_ratio": _log10_ratio(ri, rb)})
    med_b, plat_b, def_b = median_with_plateau([(r["rank_base"], r["tie_plateau_base"]) for r in rows])
    med_i, plat_i, def_i = median_with_plateau([(r["rank_it"], r["tie_plateau_it"]) for r in rows])
    logs = [r["log10_ratio"] for r in rows if r["log10_ratio"] is not None]
    arms = {}
    for nm, med, plat, dfn, fld in (("base", med_b, plat_b, def_b, "rank_base"),
                                    ("it", med_i, plat_i, def_i, "rank_it")):
        vals = [r[fld] for r in rows]
        arms[nm] = {"arm": nm, "median_rank": med, "median_rank_plateau": plat, "median_defining_items": dfn,
                    "summary": rank_summary([(v, None) for v in vals]),
                    "n_rank_le_10": sum(1 for v in vals if v <= TOP_K),
                    "n_is_top": sum(1 for v in vals if v == 1)}
    return {"entity": entity, "slot": READOUT_SLOT, "key": "canonical", "n_items_joined": len(keys),
            "n_collision_base": len(cb), "n_collision_it": len(ci), "n_common_non_collision": len(common),
            "n_rank_null_excluded": len(nulls), "rank_null_excluded_examples": nulls[:5], "n_gap_eval": len(rows),
            "arm_base": arms["base"], "arm_it": arms["it"], "L_new": _log10_ratio(med_i, med_b),
            "Lp": (statistics.median(logs) if logs else None), "n_logs": len(logs),
            "per_item_log10_ratios": logs,
            "denominator_rule": ("U7: the COMMON non-collision set of both cells, because collision is "
                                 "key-dependent and per-cell exclusion would let the two arms of the ratio "
                                 "exclude DIFFERENT item sets; null ranks dropped as PAIRS for the same reason"),
            "readout_role": readout_role(entity, READOUT_SLOT, True, "gap_arms_measurement")}


def _earlier_rank_gate(a, b):
    """The EARLIER §9.2 verdict of two: where the common-set and all-items medians disagree the SUPPRESSING
    verdict is emitted (ambiguity B)."""
    def idx(v):
        return RANK_GATE_ORDER.index(v) if v in RANK_GATE_ORDER else len(RANK_GATE_ORDER)
    return a if idx(a) <= idx(b) else b


def resolve_gap(missing_cells, join_failure, pairing_status, anchor_differs_cells, slot_verdict, rank_verdict,
                l_new, lp, l_old):
    """§9.3, TOTAL, registered order, EARLIER branch winning. Branches 1-7 are named non-emissions (missing input,
    failed join, pairing violation, §9.6's anchor suppression, §9.3's own branch 1, absent statistic); 8-12 are
    §9.3's steps 2-6. NO band past a suppressing branch. `triple_entry` is the band where one is emitted, else the
    SUPPRESSING CAUSE, because §8.2's example headline is `(SLOT_DEGENERATE, GAP_CLOSED, GAP_INDETERMINATE)`."""
    b_new, b_lp = band_of(l_new, l_old), band_of(lp, l_old)
    stamps, downgraded = [], False
    if missing_cells:
        v, cause, cons = GAP_ORDER[0], "CELL_ARTIFACT_MISSING", "suppresses"
        msg = "the rank artifact is absent for %s: a MISSING INPUT, named, never defaulted" % sorted(missing_cells)
    elif join_failure:
        v, cause, cons = GAP_ORDER[1], "JOIN_FAILURE", "suppresses"
        msg = "the item join failed loudly; no verdict may be computed past it: %s" % join_failure
    elif pairing_status == "PAIRING_NOT_UNDER_REGISTRATION":
        v, cause, cons = GAP_ORDER[2], "PAIRING_NOT_UNDER_REGISTRATION", "suppresses"
        msg = ("§1 (A8): base and -it must be same box, same session, base first. This pair verifiably is not, so "
               "it is not a run under this registration and yields no §9 verdict.")
    elif pairing_status == "PAIRING_UNVERIFIABLE":
        v, cause, cons = GAP_ORDER[3], "PAIRING_UNVERIFIABLE", "suppresses"
        msg = "§1 + §10.1: same-box-ness cannot be established, so the within-box premise is unverified"
    elif anchor_differs_cells:
        v, cause, cons = GAP_ORDER[4], "ANCHOR_DIFFERS", "suppresses"
        msg = ("§9.6: the anchor RANK column differs from the SAME-BOX shipped reference at %s, so the run is not "
               "comparable and no §9.3 verdict is emitted for this scale" % sorted(anchor_differs_cells))
    elif slot_verdict in SLOT_SUPPRESSING or rank_verdict in RANK_SUPPRESSING:
        v, cons = GAP_ORDER[5], "suppresses"
        cause = slot_verdict if slot_verdict in SLOT_SUPPRESSING else rank_verdict
        msg = ("§9.3 branch 1: %s holds, so NO gap verdict exists. This is NOT a confirmation of anything: "
               "RANK_RESOLUTION_INSUFFICIENT is not evidence the ranks are equal, and a deep median under it is "
               "no evidence the answer is implausible." % cause)
    elif l_new is None or lp is None:
        v, cause, cons = GAP_ORDER[6], "NO_STATISTIC", "suppresses"
        msg = "the statistic does not exist on the common set (L_new=%r, Lp=%r): no ratio, no band" % (l_new, lp)
    else:
        if slot_verdict == "SLOT_UNMATCHED":
            downgraded, stamps = True, ["SLOT_UNMATCHED", ONSET_DELTA_PROVENANCE]
        if b_new != b_lp:
            v, cause, cons = GAP_ORDER[7], None, "emitted_instead_of_a_band"
            msg = ("L_new=%r lands in %s while Lp=%r lands in %s: a verdict depending on which of two defensible "
                   "aggregations was chosen is not a verdict, so both numbers are reported and no band is quoted "
                   "(precedence over every band below)." % (l_new, b_new, lp, b_lp))
        else:
            v, cause, cons = b_new, None, ("emitted_downgraded" if downgraded else "emitted")
            msg = ("L_new=%r against 0.5 / 2.0 / L_old-1.0(%s) -> %s; Lp=%r agrees (%s)%s"
                   % (l_new, None if l_old is None else float(l_old) - GAP_REMOVED_LOG, v, lp, b_lp,
                      " -- DOWNGRADED by SLOT_UNMATCHED: emitted, but NOT a like-for-like comparison"
                      if downgraded else ""))
    banded = v in BANDS or v == GAP_ORDER[7]
    return {"rule": "§9.3", "verdict": v, "consequence": cons, "suppressing_cause": cause,
            "triple_entry": (v if banded else (cause or v)), "band_L_new": (b_new if banded else None),
            "band_Lp": (b_lp if banded else None), "L_new": l_new, "Lp": lp, "L_old": l_old,
            "mostly_closed_edge": (None if l_old is None else float(l_old) - GAP_REMOVED_LOG),
            "downgraded": downgraded, "stamps": stamps, "slot_gate_verdict": slot_verdict,
            "rank_gate_verdict": rank_verdict, "pairing_status": pairing_status,
            "anchor_differs_cells": sorted(anchor_differs_cells or []),
            "missing_cells": sorted(missing_cells or []), "resolution_order": list(GAP_ORDER),
            "thresholds": {"GAP_CLOSED_LOG": GAP_CLOSED_LOG, "GAP_SURVIVES_LOG": GAP_SURVIVES_LOG,
                           "GAP_REMOVED_LOG": GAP_REMOVED_LOG}, "msg": msg}


# ------------------------------------------------------------------ the anchor (§7, §9.6)
def gate_status(instrument, scale, side, kind):
    """§7.2/§7.3 as amended by A4: exact for ranks/ids/flags/top-k tokens, within 1e-6 for floats, EXCEPT every
    rank row vs the COMMITTED column at 27b and every probability row at 27b on either side (teacher-forced lp and
    what derives from it), which are DISCLOSED_NOT_GATED and emit NO verdict. §7.2 is silent on the rank
    instrument's floats vs the same-box reference at 27b and §7.3 carves the exception only for the committed
    column, so those stay gated (ambiguity I)."""
    if instrument == "prob" and scale == "27b":
        return GATE_DISCLOSED
    if instrument == "rank" and side == "committed" and scale == "27b":
        return GATE_DISCLOSED
    return GATE_EXACT if kind == "exact" else GATE_TOL


def diff_group(new_map, ref_map, keys, fields, kind, where, tol=FLOAT_TOL):
    """One group's diff: n_compared, n_differing, median_nonzero_delta, max_abs_delta, capped examples -- the form
    out/b1_fold_identity_gate_27b.json uses. A field absent on either side is a LOUD failure, never a skip."""
    n_cmp, diffs, deltas = 0, [], []
    for k in keys:
        for f in fields:
            if f not in new_map[k]:
                _fail_field("%s new side (item %s)" % (where, k), f)
            if f not in ref_map[k]:
                _fail_field("%s reference side (item %s)" % (where, k), f)
            nv, rv = new_map[k][f], ref_map[k][f]
            n_cmp += 1
            if kind == "float":
                d = abs(float(nv) - float(rv))
                deltas.append(d)
                bad = d > tol
            else:
                d, bad = None, (nv != rv)
            if bad:
                diffs.append({"join_key": k, "field": f, "new": nv, "reference": rv, "abs_delta": d})
    nz = [d for d in deltas if d > 0.0]
    return {"n_items": len(keys), "n_compared": n_cmp, "n_differing": len(diffs), "n_nonzero_delta": len(nz),
            "median_nonzero_delta": (statistics.median(nz) if nz else None),
            "max_abs_delta": (max(deltas) if deltas else None), "kind": kind,
            "tolerance": (tol if kind == "float" else "exact"), "examples": diffs[:3]}


def _topk_fields(rows, where):
    """The TOP_K dump flattened to comparable field names; a short dump is a LOUD failure, not a truncation."""
    if not isinstance(rows, list) or len(rows) != TOP_K:
        raise JoinFailure("MISSING_REQUIRED_FIELD", "%s: top-k dump has %s rows, expected %d"
                          % (where, (len(rows) if isinstance(rows, list) else None), TOP_K))
    out = {}
    for i, r in enumerate(rows):
        out["topk[%d].tok_id" % i], out["topk[%d].tok_str" % i] = _req(r, "tok_id", where), _req(r, "tok_str", where)
        out["topk[%d].p" % i] = _ffloat(r, "p", where)
    return out


def _rank_row(src, where):
    row = {f: _req(src, f, where) for f in ("rank_c_bare", "rank_w_bare", "cid", "aid", "first_token_collision")}
    row.update({"p_c_bare": _ffloat(src, "p_c_bare", where), "p_w_bare": _ffloat(src, "p_w_bare", where)})
    row.update(_topk_fields(_req(src, "topk_bare", where), where))
    return row


def rank_anchor_new(data, where):
    """The new instrument's §7b anchor column (slot `bare` x key `space` = the shipped construction verbatim), read
    under the SHIPPED field names from `anchor_shipped`."""
    pairs = [(_req(r, "join_key", where), _rank_row(_req(r, "anchor_shipped", where), where))
             for r in _req(data, "result.items", where) if r.get("slot") == ANCHOR_SLOT]
    if not pairs:
        raise JoinFailure("MISSING_REQUIRED_FIELD", "%s carries no slot=%r record" % (where, ANCHOR_SLOT))
    return _keyed(pairs, where)


def rank_anchor_ref(data, where):
    """The shipped rank artifact's own fields, keyed on join_key(q) (the shipped dump has no join_key field, so it
    is computed with the same imported join_key the new instrument persisted)."""
    return _keyed([(join_key(_req(r, "q", where)), _rank_row(r, where))
                   for r in _req(data, "result.items", where)], where)


def _prob_row(rec, where, suffix=""):
    row = {f: _ffloat(rec, "%s%s" % (f, suffix), where) for f in PROB_FLOAT_FIELDS}
    row.update({f: _req(rec, "%s%s" % (f, suffix), where) for f in PROB_LABEL_FIELDS})
    return row


def prob_anchor_new(data, where):
    """The new probability instrument's `space` column -- raw(" " + X.strip(), bos=False) verbatim, the §7b anchor
    -- under the shipped names, floats from `_full` (gates read unrounded)."""
    return _keyed([(_req(r, "join_key", where), _prob_row(r, where, "_space"))
                   for r in _req(data, "result.items", where)], where)


def prob_anchor_ref(data, where):
    """The shipped diagnose artifact's own fields, keyed on join_key(q)."""
    return _keyed([(join_key(_req(r, "q", where)), _prob_row(r, where))
                   for r in _req(data, "result.items", where)], where)


def resolve_anchor(groups, present, join_failure, meta):
    """§7/§9.6 for one (cell, instrument, reference side), TOTAL: ANCHOR_UNEVALUABLE -> *_JOIN_FAILURE ->
    ANCHOR_NO_VERDICT_DISCLOSED_NOT_GATED (no gated row exists, which is §7.2's own instruction: no reproduction
    verdict at all) -> ANCHOR_DIFFERS (a GATED group differs) -> ANCHOR_REPRODUCES. Ungated groups are excluded
    from the verdict and reported separately."""
    out = dict(meta)
    gated = {g: b for g, b in groups.items() if b.get("gate_status") != GATE_DISCLOSED}
    out.update({"rule": "§7/§9.6", "groups": groups, "resolution_order": list(ANCHOR_ORDER),
                "gated_groups": sorted(gated),
                "disclosed_not_gated_groups": sorted(g for g, b in groups.items()
                                                     if b.get("gate_status") == GATE_DISCLOSED),
                "readout_role": readout_role("anchor", ANCHOR_SLOT, True, "anchor_reproduction")})
    if not present:
        out.update({"verdict": ANCHOR_ORDER[0], "consequence": "no verdict",
                    "msg": "a side of this comparison is absent: %s" % meta.get("absent_reason")})
    elif join_failure:
        out.update({"verdict": ANCHOR_ORDER[1], "consequence": "no verdict",
                    "msg": "the item join failed loudly: %s" % join_failure})
    elif not gated:
        out.update({"verdict": ANCHOR_ORDER[2], "consequence": "no verdict, disclosure only",
                    "msg": ("every row here is DISCLOSED_NOT_GATED (A4: an exact gate would test hardware, not "
                            "code), so NO reproduction verdict is emitted; the per-field diffs stand")})
    else:
        differing = sorted(g for g, b in gated.items() if b["n_differing"] > 0)
        if differing:
            out.update({"verdict": ANCHOR_ORDER[3], "differing_groups": differing,
                        "consequence": ("suppresses the §9.3 verdict for this cell" if "ranks" in differing else
                                        "disclosed; §9.6 attaches the §9.3 suppression to the RANKS group only"),
                        "msg": "gated group(s) %s differ from the reference side" % differing})
            if "ranks" in differing and meta.get("side") == "same_box":
                out["independent_finding"] = ("§9.6: two SAME-BOX draws of the rank lineage disagree, which retires "
                                              "the one numerically stable lineage the repo has and forces a "
                                              "rank-spread disclosure onto every rank number in it")
            if meta.get("instrument") == "prob" and meta.get("scale") in ("2b", "9b"):
                out["independent_finding_lp"] = ("§9.6: ANCHOR_DIFFERS on 2b/9b lp contradicts "
                                                 "out/b1_fold_identity_gate.json's PASS (0 of 23 fields differing "
                                                 "at all four 2b/9b cells)")
        else:
            out.update({"verdict": ANCHOR_ORDER[4], "differing_groups": [],
                        "consequence": ("the new instrument is the shipped instrument plus the declared changes, so "
                                        "every §9.3 / §9.5 number here is a like-for-like successor"),
                        "msg": "every gated group %s is identical within its stated tolerance" % sorted(gated)})
    return out


# ------------------------------------------------------------------ §10's stability control
def _stab_items(data, where):
    """(items, ordered join keys) of a shipped/arms diagnose artifact; duplicates fail loudly (§10.2 M4)."""
    items = _req(data, "result.items", where)
    if not isinstance(items, list):
        _fail_field(where, "result.items (a list)")
    # An `--arm both` artifact carries every q TWICE (once per arm), so keying it whole is a
    # DUPLICATE_JOIN_KEY by construction. §10 compares the FOLD arm only, so filter to it when the
    # records carry `arm` at all; shipped diagnose artifacts have no such field and pass through.
    if any(isinstance(r, dict) and "arm" in r for r in items):
        items = [r for r in items if r.get("arm") == "fold"]
    pairs = [(join_key(_req(r, "q", where)), r) for r in items]
    _keyed(pairs, where)          # duplicates WITHIN the fold arm still fail loudly (§10.2 M4)
    return items, [k for k, _ in pairs]


def _stab_value(rec, field, where):
    """One of the 23 fields at its persisted 6dp value (§10.2's 'identical after round(x, 6)')."""
    v = _req(rec, field, where)
    if field in STAB_NUMERIC:
        return None if v is None else round(float(v), 6)
    return v


def cluster_fingerprint(items, where):
    """§10.2's DECIDED discriminator (U11), total by construction: SHA-256 of the canonical JSON
    (sort_keys, compact separators, ensure_ascii, allow_nan=False) of the ordered list of (join_key(q), the 23
    pre-existing fields at 6dp). Item-0 lpC_single does NOT discriminate: clusters 1 and 3 are both -0.187646."""
    payload = [[join_key(_req(r, "q", where)), [_stab_value(r, f, where) for f in STAB_FIELDS_23]] for r in items]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def item_order_test(keys_by_draw):
    """§10.2's item-order test (M4): identical iff the ordered [join_key(q)] lists are elementwise equal across
    draws AND have length N_ITEMS(82) (duplicates already failed loudly at load). No reordering, no intersection."""
    names = sorted(keys_by_draw)
    ref = keys_by_draw[names[0]]
    out = {"draws": names, "n_by_draw": {n: len(keys_by_draw[n]) for n in names}, "N_ITEMS": N_ITEMS,
           "length_ok": all(len(keys_by_draw[n]) == N_ITEMS for n in names), "elementwise_equal": True,
           "first_mismatch": None}
    for n in names[1:]:
        other = keys_by_draw[n]
        if other != ref:
            out["elementwise_equal"] = False
            idx = next((i for i in range(min(len(ref), len(other))) if ref[i] != other[i]), None)
            out["first_mismatch"] = ({"draw": n, "index": idx, "reference": ref[idx], "other": other[idx]}
                                     if idx is not None else {"draw": n, "reason": "LENGTH_DIFFERS"})
            break
    out["identical"] = bool(out["elementwise_equal"] and out["length_ok"])
    out["verdict"] = "ITEM_ORDER_IDENTICAL" if out["identical"] else "ITEM_ORDER_FAILED"
    return out


def stab_pair_diff(items_a, items_b, keys, na, nb, wa, wb):
    """§10.2's basis (23 fields x 82 items at 6dp) plus what §10.3 branch 2 consequence 3 requires printed beside
    every 27b lp digit: per-field n_differing, median_nonzero_delta, max_abs_delta, the threshold flips, and the
    first (item, field) cell that diverges."""
    ma = {join_key(_req(r, "q", wa)): r for r in items_a}
    mb = {join_key(_req(r, "q", wb)): r for r in items_b}
    per_field, first_div, n_cells, n_diff = {}, None, 0, 0
    for f in STAB_FIELDS_23:
        deltas, n_d = [], 0
        for k in keys:
            va, vb = _stab_value(ma[k], f, wa), _stab_value(mb[k], f, wb)
            n_cells += 1
            if f in STAB_NUMERIC and va is not None and vb is not None:
                deltas.append(abs(float(va) - float(vb)))
            if va != vb:
                n_d, n_diff = n_d + 1, n_diff + 1
                if first_div is None:
                    first_div = {"join_key": k, "field": f, na: va, nb: vb}
        nz = [d for d in deltas if d > 0.0]
        per_field[f] = {"n_differing": n_d, "median_nonzero_delta": (statistics.median(nz) if nz else None),
                        "max_abs_delta": (max(deltas) if deltas else None)}
    return {"pair": [na, nb], "n_items": len(keys), "n_fields": len(STAB_FIELDS_23), "n_cells": n_cells,
            "n_differing_cells": n_diff, "identical": bool(n_diff == 0), "per_field": per_field,
            "threshold_flips": {f: sum(1 for k in keys if bool(_stab_value(ma[k], f, wa))
                                       != bool(_stab_value(mb[k], f, wb))) for f in STAB_LABELS},
            "first_divergent_cell": first_div,
            "basis": "23 pre-existing fields x %d items, identical after round(x, 6) (§10.2)" % len(keys)}


STAB_CONSEQUENCES = {
    "SHIPPED_SELF_DIFFERS": [
        "1. A same-box difference between two DIFFERENT scripts carries no information about code, because two runs "
        "of the SAME script also differ: the cleangate comparison is UNINFORMATIVE ABOUT CODE and its verdict "
        "TOPK_NEUTRAL__DIAGNOSE_NOT_NEUTRAL__B1_LISTEN_WITHDRAWN is REOPENED (not reversed).",
        "2. WITHIN_BOX_DETERMINISTIC (r1_27b_determinism_rider.json) becomes a statement about that box, not a "
        "property of the instrument.",
        "3. EVERY 27b teacher-forced lp digit in the repo acquires a run-to-run spread that must be printed beside "
        "it; the spread measured here (per field n_differing, median_nonzero_delta, max_abs_delta, threshold flips, "
        "category on both draws) IS the number to print.",
        "4. This registration's own 27b R-PROB numbers inherit that spread and are quotable only with it, and §9.5 "
        "branch 2 (KEY_EFFECT_BELOW_NOISE) fires wherever the key's flip count does not exceed the noise count.",
        "5. G1's op-order hunt (OWED.md:58) would be chasing a difference that need not exist and should not be "
        "started on the strength of the cleangate result alone.",
        "6. The §7 same-box anchor gate at 27b is evaluated against the PAIR (A1, A2) with flip counts disclosed; "
        "lp stays DISCLOSED_NOT_GATED and RANKS REMAIN EXACT-GATED, because nothing here bears on the rank "
        "lineage."],
    "ARMS_MATCHES_SHIPPED": [
        "Two draws of the shipped code agree and the re-parameterisation agrees with both: the cleangate difference "
        "was a property of THAT DRAW, not of the code, and the anomalous side was the clean test's own reference. "
        "The cleangate verdict is REOPENED (not reversed)."],
    "ARMS_DIFFERS": [
        "The cleangate result reproduces against a TWO-DRAW reference on a second box: the code difference stands, "
        "G1 is the right route, and B1's listen withdrawal stands as registered."],
    "STAB27B_UNEVALUABLE": [
        "No verdict either way, and NOT A PASS. It also triggers §9.5 branch 1 at 27b-base."],
}
STAB_BOUNDARY = ("§10.3's boundary on 'reopened': B1's listen numbers were withdrawn as a registered consequence. "
                 "Branches 2 and 3 REOPEN that withdrawal -- neither reverses it. Restoring six cells of listen "
                 "numbers needs its own registration stating the restoration rule before the comparison is "
                 "re-read.")
STAB_CANNOT = ("§10.3: the cleangate box is gone. A new box may define a FOURTH cluster, and cluster membership "
               "measured on a new box is not evidence about the cleangate box, so SHIPPED_SELF_IDENTICAL bounds "
               "within-box repeatability ON THIS BOX and is only circumstantial about the cleangate draw.")


def resolve_stab27b(missing, incomplete, same_box_status, order_ok, a1_eq_a2, b1_eq_a1):
    """§10.3, TOTAL: STAB27B_UNEVALUABLE (draw missing, incomplete/OOM/capped, item order failing, or the pair not
    verifiably same-box -- ambiguity H folds a verifiable not-same-box pair in here) -> SHIPPED_SELF_DIFFERS (A1 !=
    A2 on any field) -> SHIPPED_SELF_IDENTICAL + ARMS_MATCHES_SHIPPED (B1 == A1) -> + ARMS_DIFFERS."""
    if missing:
        v, arms, reason = STAB_ORDER[0], None, "DRAW_MISSING:%s" % sorted(missing)
    elif incomplete:
        v, arms, reason = STAB_ORDER[0], None, "DRAW_INCOMPLETE_OOM_OR_CAPPED:%s" % sorted(incomplete)
    elif not order_ok:
        v, arms, reason = STAB_ORDER[0], None, "ITEM_ORDER_FAILED"
    elif same_box_status == "SAME_BOX_UNVERIFIABLE":
        v, arms, reason = STAB_ORDER[0], None, "SAME_BOX_UNVERIFIABLE"
    elif same_box_status == "NOT_SAME_BOX":
        v, arms, reason = STAB_ORDER[0], None, "PAIR_NOT_SAME_BOX"
    elif not a1_eq_a2:
        v, arms, reason = STAB_ORDER[1], None, "A1 != A2 on at least one of the 23 fields x 82 items"
    else:
        v, reason = STAB_ORDER[2], "A1 == A2 on all 23 fields x 82 items"
        arms = "ARMS_MATCHES_SHIPPED" if b1_eq_a1 else "ARMS_DIFFERS"
    return {"rule": "§10.3", "verdict": v, "arms_verdict": arms, "reason": reason,
            "resolution_order": list(STAB_ORDER), "consequences": list(STAB_CONSEQUENCES.get(arms or v, [])),
            "boundary_on_reopened": STAB_BOUNDARY, "what_this_cannot_do": STAB_CANNOT,
            "triggers_key_materiality_branch_1_at_27bbase": bool(v == STAB_ORDER[0]),
            "discriminator": "§10.2's cluster fingerprint, NOT item-0 lpC_single",
            "readout_role": readout_role("stab27b", "shipped_diagnose", False, "stability_verdict")}


# ------------------------------------------------------------------ §9.4 / §9.5 recompute
def prob_cell_view(data, where):
    """Every §9.5 input the probability instrument persists, plus §9.4 recomputed from the per-item residuals
    (ambiguity L); the on-box verdicts are carried beside the recomputed ones and a disagreement is named."""
    ke = _req(data, "result.key_effect", where)
    view = {"where": where, "canonical_key": _req(ke, "canonical_key", where),
            "canonical_is_anchor_key": bool(_req(ke, "canonical_is_anchor_key", where)),
            "canonical_prefix_ok": bool(_req(ke, "canonical_prefix_ok", where)),
            "n_flip_faithful_RC": _req(ke, "n_flip_faithful_RC", where),
            "n_flip_headroom_pass": _req(ke, "n_flip_headroom_pass", where),
            "n_flip_faithful_RA": _opt(ke, "n_flip_faithful_RA"),
            "category_canonical": _req(ke, "category_canonical", where),
            "category_space": _req(ke, "category_space", where), "dRC": _opt(ke, "dRC"), "dM0": _opt(ke, "dM0"),
            "dRC_dM0_note": "A3: reported MAGNITUDES with NO verdict attached",
            "on_box_verdict_RC": _opt(ke, "verdict_RC.verdict"),
            "on_box_verdict_headroom": _opt(ke, "verdict_headroom.verdict"), "identity_check": {}}
    items = _req(data, "result.items", where)
    for s in RESIDUAL_SLOTS:
        vals, n_und = [], 0
        for r in items:
            st = _req(r, "residual_i0_%s_status" % s, where)
            if st == "OK":
                vals.append(_ffloat(r, "residual_i0_%s" % s, where))
            elif st == "P_UNDERFLOW":
                n_und += 1
        med = statistics.median(vals) if vals else None
        off, on_box = decide_identity_check(med, len(vals)), _opt(data, "result.residual_i0.%s.verdict.verdict" % s)
        view["identity_check"][s] = {
            "n_computable": len(vals), "n_P_underflow": n_und, "median_residual_i0": med,
            "max_abs_residual_i0": (max(abs(v) for v in vals) if vals else None),
            "threshold_nats": IDENTITY_RESIDUAL_NATS, "offline_verdict": off["verdict"], "on_box_verdict": on_box,
            "agreement": ("ON_BOX_OFFLINE_DISAGREE" if (on_box and on_box != off["verdict"])
                          else "AGREE_OR_ON_BOX_ABSENT"),
            "note": ("P_UNDERFLOW items excluded and counted, ln(0) never taken, gate reads the unrounded median; "
                     "§14.2 makes the offline verdict the emitted one (ambiguity L)"),
            "readout_role": readout_role("identity_check", s, True, "residual_i0")}
    return view


def noise_flips(d1, d2, w1, w2):
    """A18's per-cell NOISE CONTEXT: the within-box flip count between the cell's two shipped
    family_cave_diagnose draws (sbref_ vs sbref2_; at 27b-base §10's shipA/shipB, per §14.1's identity), joined on
    join_key(q) with key-set equality asserted. The count itself is the imported flip_count."""
    i1, k1 = _stab_items(d1, w1)
    i2, k2 = _stab_items(d2, w2)
    keys = _assert_same_keys(k1, k2, w1, w2)
    m1 = {join_key(_req(r, "q", w1)): r for r in i1}
    m2 = {join_key(_req(r, "q", w2)): r for r in i2}
    out = {"n_items": len(keys), "draw1": w1, "draw2": w2,
           "category_draw1": _opt(d1, "result.decision.category"),
           "category_draw2": _opt(d2, "result.decision.category")}
    for f, nm in (("faithful_RC", "n_flip_faithful_RC"), ("headroom_pass", "n_flip_headroom_pass"),
                  ("faithful_RA", "n_flip_faithful_RA")):
        out[nm] = flip_count([bool(_req(m1[k], f, w1)) for k in keys], [bool(_req(m2[k], f, w2)) for k in keys])
    return out


def key_materiality(cell, pv, noise, noise_status):
    """§9.5 RECOMPUTED with the noise context, using the probability instrument's own imported resolvers so the
    precedence cannot drift: KEY_UNLOCATABLE -> KEY_COMPARISON_IS_IDENTITY -> the A18 no-noise-context branch ->
    KEY_EFFECT_BELOW_NOISE -> the MIN_FAITHFUL(8) count rule (or a category change) -> immaterial."""
    rc = decide_key_material_rc(pv["n_flip_faithful_RC"], pv["category_canonical"], pv["category_space"],
                                pv["canonical_prefix_ok"], pv["canonical_is_anchor_key"],
                                (noise or {}).get("n_flip_faithful_RC"), noise_status)
    hp = decide_key_material_headroom(pv["n_flip_headroom_pass"], pv["canonical_prefix_ok"],
                                      pv["canonical_is_anchor_key"], (noise or {}).get("n_flip_headroom_pass"),
                                      noise_status)
    return {"rule": "§9.5 (A3, A12, A18)", "cell": cell, "verdict_RC": rc, "verdict_headroom": hp,
            "noise_context_status": noise_status, "noise": noise, "MIN_FAITHFUL": MIN_FAITHFUL,
            "n_flip_faithful_RC": pv["n_flip_faithful_RC"], "n_flip_headroom_pass": pv["n_flip_headroom_pass"],
            "n_flip_faithful_RA": pv["n_flip_faithful_RA"], "category_canonical": pv["category_canonical"],
            "category_space": pv["category_space"], "dRC": pv["dRC"], "dM0": pv["dM0"],
            "dRC_dM0_note": pv["dRC_dM0_note"], "canonical_key": pv["canonical_key"],
            "on_box_verdict_RC": pv["on_box_verdict_RC"], "on_box_verdict_headroom": pv["on_box_verdict_headroom"],
            "supersedes_on_box": ("the on-box verdict is ALWAYS branch 1 because the frozen CLI (§14.1) cannot "
                                  "express a cross-invocation comparison; this recompute is the registered one"),
            "readout_role": readout_role("key_materiality", cell, True, "faithful_RC_flips")}


# ------------------------------------------------------------------ discovery
def _candidates(role, cell):
    return list(ALIASES[(role, cell)]) if (role, cell) in ALIASES else [ROLE_FILES[role] % cell]


def _entry(candidates, roots):
    """One input's ledger entry: the FIRST candidate found in the FIRST root having it (a fixed order, so no
    resolution can be chosen after the numbers are seen); other hits go to `also_present`, and two copies of the
    SAME filename are recorded as `duplicate_copies`. Absent -> present: False, never a default."""
    found = [Path(root) / name for name in candidates for root in roots if (Path(root) / name).exists()]
    e = {"candidates": list(candidates), "present": bool(found), "path": (str(found[0]) if found else None),
         "resolved_from": (found[0].name if found else None), "also_present": [str(p) for p in found[1:]],
         "data": None}
    if found:
        e["data"] = json.loads(found[0].read_text(encoding="utf-8"))
        dupes = [str(p) for p in found[1:] if p.name == found[0].name]
        if dupes:
            e["duplicate_copies"] = dupes
    return e


def discover(results_dirs):
    """Every expected input, resolved to a present/absent ledger. The committed column and §10.2's cluster paths
    resolve against the repo root; everything else against the given result directories, in order."""
    dirs = [str(d) for d in results_dirs]
    inputs = {}
    for cell in CELLS:
        for role in ROLE_FILES:
            inputs["%s/%s" % (role, cell)] = _entry(_candidates(role, cell), dirs)
        inputs["rank_committed/%s" % cell] = _entry([COMMITTED_RANK[cell]], [str(_REPO_ROOT)])
        inputs["prob_committed/%s" % cell] = _entry([COMMITTED_PROB[cell]], [str(_REPO_ROOT)])
    for nm in ("A1", "A2", "B1"):
        inputs["stab/%s" % nm] = _entry(_candidates("stab", nm), dirs)
    for cid, paths in KNOWN_CLUSTERS.items():
        for rel in paths:
            inputs["cluster/%s/%s" % (cid, rel)] = _entry([rel], [str(_REPO_ROOT)])
    return inputs


def _scipy_available():
    """REGISTRATION_provenance.md §1 requires this stamped. scipy is never CALLED (§8 fixes math.comb only)."""
    try:
        import importlib.util
        return bool(importlib.util.find_spec("scipy"))
    except Exception:                                                       # noqa: BLE001
        return False


# ------------------------------------------------------------------ assembly
def assemble(inputs, results_dirs=(), scipy_available=False):
    """Build the whole artifact from the input ledger. Pure given `inputs` (no filesystem access except the
    provenance fallback, which reads only paths already recorded in the ledger)."""
    hard, missing = [], sorted(k for k, e in inputs.items() if not e["present"])
    for k, e in inputs.items():
        if e.get("duplicate_copies"):
            hard.append({"unit": k, "kind": "DUPLICATE_ARTIFACT_ACROSS_RESULT_DIRS",
                         "failure": "the same filename exists in more than one result dir: %s"
                                    % e["duplicate_copies"]})

    def dat(key):
        e = inputs.get(key) or {}
        return e.get("data"), e.get("path"), key

    def guard(unit, fn):
        try:
            return fn(), None
        except JoinFailure as e:
            hard.append({"unit": unit, "kind": e.kind, "failure": str(e)})
            return None, str(e)

    # ---- per-cell probability views (§9.4 + §9.5 inputs) ----
    prob_views, prob_fail = {}, {}
    for cell in CELLS:
        d, _, w = dat("prob_fmt/%s" % cell)
        if d is None:
            prob_views[cell] = None
            continue
        prob_views[cell], prob_fail[cell] = guard("prob_fmt/%s" % cell, lambda d=d, w=w: prob_cell_view(d, w))

    # ---- §10, before §9.5, because it gates the 27b-base noise context ----
    draws = {nm: dat("stab/%s" % nm) for nm in ("A1", "A2", "B1")}
    stab = {"draws": {nm: {"path": p, "present": d is not None, "role": nm} for nm, (d, p, _) in draws.items()},
            "definition_same_box": "§10.1, mechanically; the fallback basis is disclosed (ambiguity F)"}
    stab_missing = sorted(nm for nm, (d, _, _) in draws.items() if d is None)
    items, keys, prov, srcs, incomplete = {}, {}, {}, {}, []
    jf = None
    for nm, (d, p, key) in draws.items():
        if d is None:
            continue
        res, err = guard("stab/%s" % nm, lambda d=d, key=key: _stab_items(d, key))
        if err:
            jf = jf or err
            continue
        items[nm], keys[nm] = res
        prov[nm], srcs[nm] = provenance_of(d, p)
        if len(keys[nm]) != N_ITEMS:
            incomplete.append(nm)
    order = item_order_test(keys) if len(keys) == 3 else {"identical": False, "verdict": "ITEM_ORDER_UNEVALUABLE",
                                                          "draws": sorted(keys)}
    sb_pairs = {}
    for a, b in (("A1", "A2"), ("A1", "B1"), ("A2", "B1")):
        if a in prov and b in prov:
            sb_pairs["%s_vs_%s" % (a, b)] = same_box(prov[a], prov[b], a, b)
    sb_status = "SAME_BOX_UNVERIFIABLE" if not sb_pairs else (
        "NOT_SAME_BOX" if any(v["status"] == "NOT_SAME_BOX" for v in sb_pairs.values()) else
        ("SAME_BOX_UNVERIFIABLE" if any(v["status"] == "SAME_BOX_UNVERIFIABLE" for v in sb_pairs.values())
         else "SAME_BOX"))
    fps, diffs = {}, {}
    if len(items) == 3 and order.get("identical") and not jf:
        for nm in items:
            fps[nm], _ = guard("stab/%s fingerprint" % nm,
                               lambda nm=nm: cluster_fingerprint(items[nm], "stab/%s" % nm))
        for a, b in (("A1", "A2"), ("A1", "B1"), ("A2", "B1")):
            diffs["%s_vs_%s" % (a, b)], _ = guard("stab diff %s/%s" % (a, b),
                                                  lambda a=a, b=b: stab_pair_diff(items[a], items[b], keys[a], a, b,
                                                                                  "stab/%s" % a, "stab/%s" % b))
    a1a2, a1b1 = diffs.get("A1_vs_A2"), diffs.get("A1_vs_B1")
    stab.update({"same_box_pairs": sb_pairs, "same_box_effective_basis": sb_status,
                 "same_box_artifact_basis": {nm: srcs.get(nm) for nm in draws},
                 "item_order": order, "fingerprints": fps, "pair_diffs": diffs,
                 "categories": {nm: _opt(draws[nm][0] or {}, "result.decision.category") for nm in draws},
                 "join_failure": jf})
    stab["verdict"] = resolve_stab27b(stab_missing, incomplete, sb_status, bool(order.get("identical")),
                                      bool(a1a2 and a1a2["identical"]), bool(a1b1 and a1b1["identical"]))
    if jf:
        stab["verdict"]["reason"] = "JOIN_FAILURE: %s" % jf
    known_fp = {}
    for cid, paths in KNOWN_CLUSTERS.items():
        for rel in paths:
            d, _, key = dat("cluster/%s/%s" % (cid, rel))
            if d is None:
                known_fp[rel] = {"cluster": cid, "present": False, "fingerprint": None}
                continue
            res, _ = guard("cluster/%s" % rel, lambda d=d, key=key: cluster_fingerprint(_stab_items(d, key)[0], key))
            known_fp[rel] = {"cluster": cid, "present": True, "fingerprint": res}
    stab["known_cluster_fingerprints"] = known_fp
    stab["cluster_membership"] = {
        nm: next((v["cluster"] for v in known_fp.values() if v["fingerprint"] and v["fingerprint"] == fp),
                 "CLUSTER_UNMATCHED_POSSIBLE_FOURTH") for nm, fp in fps.items()}

    # ---- §9.5 per cell, with the recomputed noise context ----
    materiality = {}
    for cell in CELLS:
        pv = prob_views.get(cell)
        d1, p1, w1 = dat("prob_sbref/%s" % cell)
        d2, p2, w2 = dat("prob_sbref2/%s" % cell)
        noise, status = None, NOISE_ABSENT
        if d1 is not None and d2 is not None:
            noise, err = guard("noise/%s" % cell, lambda: noise_flips(d1, d2, w1, w2))
            status = NOISE_CONTEXT_OK if noise is not None else "NOISE_CONTEXT_INVALID_JOIN_FAILED: %s" % err
        if cell == "27bbase" and stab["verdict"]["verdict"] == STAB_ORDER[0]:
            noise, status = noise, "NOISE_CONTEXT_INVALID_STAB27B_%s" % stab["verdict"]["reason"]
        if pv is None:
            materiality[cell] = {"rule": "§9.5", "cell": cell,
                                 "verdict_RC": {"verdict": "KEY_MATERIALITY_UNEVALUABLE_ARTIFACT_MISSING"},
                                 "verdict_headroom": {"verdict": "KEY_MATERIALITY_UNEVALUABLE_ARTIFACT_MISSING"},
                                 "reason": prob_fail.get(cell) or "prob_fmt artifact absent",
                                 "noise_context_status": status, "noise": noise,
                                 "readout_role": readout_role("key_materiality", cell, True, "unevaluable")}
        else:
            materiality[cell] = key_materiality(cell, pv, noise, status)

    # ---- §7 anchors ----
    anchor = {}
    for cell in CELLS:
        scale = cell[:-4] if cell.endswith("base") else cell[:-2]
        for instrument, new_key, reader_new, reader_ref, groups in (
                ("rank", "rank_fmt/%s" % cell, rank_anchor_new, rank_anchor_ref, RANK_GROUPS),
                ("prob", "prob_fmt/%s" % cell, prob_anchor_new, prob_anchor_ref, PROB_GROUPS)):
            nd, _, nw = dat(new_key)
            for side, ref_key in (("same_box", "%s_sbref/%s" % (instrument, cell)),
                                  ("committed", "%s_committed/%s" % (instrument, cell))):
                rd, _, rw = dat(ref_key)
                meta = {"cell": cell, "scale": scale, "instrument": instrument, "side": side,
                        "new_artifact": nw, "reference_artifact": rw,
                        "absent_reason": ("new artifact absent" if nd is None else
                                          ("reference artifact absent" if rd is None else None))}
                if instrument == "rank" and scale in ("2b", "9b"):
                    meta["status_note"] = A5_STAMP
                if scale == "27b":
                    meta["disclosure_27b"] = DISCLOSURE_27B
                gs = {}
                fail = None
                if nd is not None and rd is not None:
                    res, fail = guard("anchor/%s/%s/%s" % (cell, instrument, side),
                                      lambda: (reader_new(nd, nw), reader_ref(rd, rw)))
                    if res is not None:
                        (nm_, nk), (rm_, rk) = res
                        shared, fail2 = guard("anchor/%s/%s/%s keys" % (cell, instrument, side),
                                              lambda: _assert_same_keys(nk, rk, nw, rw))
                        fail = fail or fail2
                        if shared is not None:
                            for gname, fields, kind in groups:
                                blk, f3 = guard("anchor/%s/%s/%s/%s" % (cell, instrument, side, gname),
                                                lambda: diff_group(nm_, rm_, shared, fields, kind,
                                                                   "%s/%s" % (cell, gname)))
                                fail = fail or f3
                                if blk is not None:
                                    blk["gate_status"] = gate_status(instrument, scale, side, kind)
                                    blk["fields"] = list(fields)
                                    gs[gname] = blk
                            if instrument == "prob":
                                cn = _opt(nd, "result.per_column.space.decision.category")
                                cr = _opt(rd, "result.decision.category")
                                gs["category"] = {"gate_status": gate_status(instrument, scale, side, "exact"),
                                                  "fields": ["category"], "kind": "exact", "n_items": 1,
                                                  "n_compared": 1, "n_differing": int(cn != cr),
                                                  "new": cn, "reference": cr}
                anchor["%s/%s/%s" % (cell, instrument, side)] = resolve_anchor(
                    gs, bool(nd is not None and rd is not None), fail, meta)

    # ---- §9.1, §9.2, §9.3 per scale ----
    gap, precond, sign_labels = {}, {}, []
    for scale in SCALES:
        bc, ic = "%sbase" % scale, "%sit" % scale
        bd, bp, bw = dat("rank_fmt/%s" % bc)
        idd, ip, iw = dat("rank_fmt/%s" % ic)
        miss = [c for c, d in ((bc, bd), (ic, idd)) if d is None]
        views, vfail = {}, None
        for nm, d, w in (("base", bd, bw), ("it", idd, iw)):
            if d is not None:
                views[nm], e = guard("rank_fmt/%s" % (bc if nm == "base" else ic), lambda d=d, w=w:
                                     rank_cell_view(d, w))
                vfail = vfail or e
        pair = resolve_pairing(*(provenance_of(bd, bp) + provenance_of(idd, ip))) if not miss else {
            "rule": "§1/§7a (A8)", "status": "PAIRING_UNVERIFIABLE", "reason": "CELL_ARTIFACT_MISSING",
            "consequence": "suppresses", "readout_role": readout_role("pairing", READOUT_SLOT, False,
                                                                      "pairing_gate")}
        f_b = (views.get("base") or {}).get("frac_slot_answer_onset")
        f_i = (views.get("it") or {}).get("frac_slot_answer_onset")
        slot_gate = resolve_slot_gate(f_b, f_i)
        slot_gate["onset_diagnostics"] = {nm: {"frac_slot_answer_onset": (views.get(nm) or {})
                                               .get("frac_slot_answer_onset"),
                                               "n_slot_answer_onset": (views.get(nm) or {})
                                               .get("n_slot_answer_onset"),
                                               "decomposition": (views.get(nm) or {}).get("onset_decomposition"),
                                               "non_onset_composition": (views.get(nm) or {})
                                               .get("non_onset_composition")} for nm in ("base", "it")}
        slot_gate["A19_note"] = ("matched RATE is what the gate tests; matched KIND is what the per-arm non-onset "
                                 "composition shows, and both must be read together (§9.1, A19)")
        n_fail = sum((views.get(nm) or {}).get("n_items_prefix_fail") or 0 for nm in ("base", "it"))
        arms_anchor_differ = sorted(c for c in (bc, ic)
                                    if _opt(anchor, "%s/rank/same_box.verdict" % c) == "ANCHOR_DIFFERS"
                                    and "ranks" in (_opt(anchor, "%s/rank/same_box.differing_groups" % c) or []))
        precond[scale] = {"pairing": pair, "slot_gate": slot_gate, "n_items_prefix_fail_union": n_fail,
                          "rank_view_join_failure": vfail}
        gap[scale] = {}
        for entity in ENTITIES:
            l_old = _opt(L_OLD_LOG10, "%s.%s" % (entity, scale))
            arms, afail = (None, None)
            if not miss:
                arms, afail = guard("gap/%s/%s" % (scale, entity),
                                    lambda e=entity: gap_arms(bd, idd, e, bw, iw))
            jfail = vfail or afail
            rg_common = resolve_rank_gate(n_fail,
                                          None if not arms else {"median_rank": arms["arm_base"]["median_rank"],
                                                                 "median_rank_plateau":
                                                                     arms["arm_base"]["median_rank_plateau"],
                                                                 "arm": "base_common_set"},
                                          None if not arms else {"median_rank": arms["arm_it"]["median_rank"],
                                                                 "median_rank_plateau":
                                                                     arms["arm_it"]["median_rank_plateau"],
                                                                 "arm": "it_common_set"})
            rg_alt = resolve_rank_gate(n_fail, *[None if nm not in views else
                                                 {"median_rank": views[nm]["entities"][entity]
                                                  ["median_rank_instrument"],
                                                  "median_rank_plateau": views[nm]["entities"][entity]
                                                  ["median_rank_plateau_instrument"], "arm": "%s_all_items" % nm}
                                                 for nm in ("base", "it")])
            rank_verdict = _earlier_rank_gate(rg_common["verdict"], rg_alt["verdict"])
            st = sign_test(arms.get("per_item_log10_ratios") or []) if arms else None
            res = resolve_gap(miss, jfail, pair["status"], arms_anchor_differ, slot_gate["verdict"], rank_verdict,
                              (arms or {}).get("L_new"), (arms or {}).get("Lp"), l_old)
            res.update({"entity": entity, "scale": scale, "cells": [bc, ic],
                        "band_geometry": mostly_closed_band(entity, scale, l_old),
                        "measurement": arms, "rank_gate": rg_common, "rank_gate_alt": rg_alt,
                        "rank_gate_note": ("the gate is resolved on the COMMON-set median and on the instrument's "
                                           "all-items median; the SUPPRESSING verdict is the emitted one "
                                           "(ambiguity B)"),
                        "sign_test": st, "per_cell_medians": {nm: (views.get(nm) or {}).get("entities", {})
                                                              .get(entity) for nm in ("base", "it")},
                        "anchor_same_box_rank": {c: _opt(anchor, "%s/rank/same_box.verdict" % c) for c in (bc, ic)},
                        "anchor_note": ("ANCHOR_UNEVALUABLE is disclosed, not suppressing (ambiguity J); only "
                                        "ANCHOR_DIFFERS on the RANKS group of the same-box side suppresses"),
                        "readout_role": readout_role(entity, READOUT_SLOT, True, "L_new_per_scale")})
            if scale == "27b":
                res["disclosure_27b"] = DISCLOSURE_27B
            gap[scale][entity] = res
            if st and st.get("p_two_sided") is not None:
                sign_labels.append(("%s@%s" % (entity, scale), st["p_two_sided"]))

    triple = [gap[s][PRIMARY_ENTITY]["triple_entry"] for s in SCALES]
    headline = {"readout_role": readout_role(PRIMARY_ENTITY, READOUT_SLOT, True, "L_new"),
                "designation": dict(PRIMARY_READOUT), "scales_in_order": list(SCALES),
                "triple": triple, "triple_str": "(%s)" % ", ".join(triple),
                "L_new": [gap[s][PRIMARY_ENTITY]["L_new"] for s in SCALES],
                "Lp": [gap[s][PRIMARY_ENTITY]["Lp"] for s in SCALES],
                "L_old": [_opt(L_OLD_LOG10, "%s.%s" % (PRIMARY_ENTITY, s)) for s in SCALES],
                "verdicts": [gap[s][PRIMARY_ENTITY]["verdict"] for s in SCALES],
                "quotation_rule": ("§8.2: quotable as a TRIPLE or not at all, including when a scale is suppressed "
                                   "-- a suppressed scale's entry is its suppressing verdict and is part of the "
                                   "headline. Everything else in this artifact is SECONDARY and DIAGNOSTIC and may "
                                   "not be promoted; a suppressing secondary gate is still binding.")}
    out = {"control": "fmt_matched_join",
           "registration": "docs/drafts/REGISTRATION_format_matched_readout.md (frozen, pre-data, A1-A20)",
           "metric": METRIC, "decision_rule": DECISION_RULE, "stamp": stamp(), "stamp_keys": list(STAMP_KEYS),
           "thresholds": {"GAP_CLOSED_LOG": GAP_CLOSED_LOG, "GAP_REMOVED_LOG": GAP_REMOVED_LOG,
                          "GAP_SURVIVES_LOG": GAP_SURVIVES_LOG, "ALPHA": ALPHA, "N_ITEMS": N_ITEMS,
                          "TOP_K": TOP_K, "MIN_FAITHFUL": MIN_FAITHFUL, "ONSET_DELTA": ONSET_DELTA,
                          "ONSET_DELTA_provenance": ONSET_DELTA_PROVENANCE, "FLOAT_TOL": FLOAT_TOL,
                          "IDENTITY_RESIDUAL_NATS": IDENTITY_RESIDUAL_NATS, "N_HOLM_TESTS": N_HOLM_TESTS,
                          "WITHDRAWN": WITHDRAWN_THRESHOLDS, "L_OLD_LOG10_adopted": L_OLD_LOG10},
           "scipy_available": bool(scipy_available), "results_dirs": [str(d) for d in results_dirs],
           "headline": headline, "gap": gap, "preconditions": precond, "anchor": anchor,
           "key_materiality": materiality, "stab27b": stab,
           "identity_check": {c: (prob_views[c]["identity_check"] if prob_views.get(c) else
                                  {"verdict": "IDENTITY_CHECK_UNEVALUABLE_ARTIFACT_MISSING"}) for c in CELLS},
           "sign_test_family": holm_alphas(sign_labels), "n_sign_tests_computed": len(sign_labels),
           "inputs": {k: {kk: vv for kk, vv in e.items() if kk != "data"} for k, e in inputs.items()},
           "missing_inputs": missing, "hard_failures": hard, "disclosure_27b": DISCLOSURE_27B,
           "not_licensed": NOT_LICENSED,
           "exit_code": (EXIT_HARD_FAILURE if hard else (EXIT_MISSING_INPUT if missing else EXIT_OK))}
    out["n_primary_role_fields"] = count_role(out, ROLE_PRIMARY)
    if out["n_primary_role_fields"] != 1:
        raise RuntimeError("§13 violated: %d objects carry readout_role=%r, expected exactly 1"
                           % (out["n_primary_role_fields"], ROLE_PRIMARY))
    return out


def run(results_dirs, out_path):
    art = assemble(discover(results_dirs), results_dirs, _scipy_available())
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(art, indent=2, default=str), encoding="utf-8")
    print("[HEADLINE §8.2 primary readout] W*/elicit/canonical L_new triple (2b, 9b, 27b) = %s"
          % art["headline"]["triple_str"], flush=True)
    print("[HEADLINE] L_new=%s Lp=%s L_old=%s -- quotable as a triple or not at all"
          % (art["headline"]["L_new"], art["headline"]["Lp"], art["headline"]["L_old"]), flush=True)
    for s in SCALES:
        pc = art["preconditions"][s]
        print("[%s] pairing=%s slot_gate=%s (f_base=%s f_it=%s d=%s) prefix_fail_union=%s"
              % (s, pc["pairing"]["status"], pc["slot_gate"]["verdict"], pc["slot_gate"]["f_base"],
                 pc["slot_gate"]["f_it"], pc["slot_gate"]["abs_delta"], pc["n_items_prefix_fail_union"]),
              flush=True)
        for e in ENTITIES:
            g = art["gap"][s][e]
            print("[%s|%s] %s (%s) L_new=%s [%s] Lp=%s [%s] L_old=%s edge=%s band=%s n_gap_eval=%s rank_gate=%s "
                  "sign p=%s role=%s"
                  % (s, e, g["verdict"], g["consequence"], g["L_new"], g["band_L_new"], g["Lp"], g["band_Lp"],
                     g["L_old"], g["mostly_closed_edge"], g["band_geometry"]["band_status"],
                     (g["measurement"] or {}).get("n_gap_eval"), g["rank_gate_verdict"],
                     (g["sign_test"] or {}).get("p_two_sided"), g["readout_role"]), flush=True)
    for cell in CELLS:
        m = art["key_materiality"][cell]
        print("[%s|§9.5] %s / %s (noise=%s, flips RC/hp=%s/%s, MIN_FAITHFUL=%s, cat can/sp=%s/%s)"
              % (cell, m["verdict_RC"]["verdict"], m["verdict_headroom"]["verdict"], m.get("noise_context_status"),
                 m.get("n_flip_faithful_RC"), m.get("n_flip_headroom_pass"), MIN_FAITHFUL,
                 m.get("category_canonical"), m.get("category_space")), flush=True)
    for k in sorted(art["anchor"]):
        a = art["anchor"][k]
        print("[anchor %s] %s (%s) gated=%s disclosed=%s" % (k, a["verdict"], a["consequence"],
                                                             a["gated_groups"], a["disclosed_not_gated_groups"]),
              flush=True)
    sv = art["stab27b"]["verdict"]
    print("[§10] %s%s reason=%s same_box=%s clusters=%s"
          % (sv["verdict"], (" + " + sv["arms_verdict"]) if sv["arms_verdict"] else "", sv["reason"],
             art["stab27b"]["same_box_effective_basis"], art["stab27b"]["cluster_membership"]), flush=True)
    for k in art["missing_inputs"]:
        print("[MISSING] %s -> named unevaluable, never a default (%s)" % (k, art["inputs"][k]["candidates"]),
              flush=True)
    for h in art["hard_failures"]:
        print("[FAIL] %s: %s" % (h["unit"], h["failure"]), flush=True)
    print("[primary] n_primary_role_fields=%d (§13: exactly 1, the triple) | scipy_available=%s"
          % (art["n_primary_role_fields"], art["scipy_available"]), flush=True)
    print("[27b] every 27b number above is quotable only with disclosure_27b (§11)", flush=True)
    print("[done] wrote %s (exit %d)" % (p.as_posix(), art["exit_code"]), flush=True)
    return art["exit_code"]


# ------------------------------------------------------------------ selftest (model-free, artifact-free)
def _sp(iid="boxA", t="2026-07-29T00:00:00+00:00"):
    return {"lambda_instance_id": iid, "gpu_name": "H100", "driver": "580", "cuda_visible_devices": "0",
            "device_index": 0, "started_utc": t, "finished_utc": t}


def _topk():
    return [{"tok_id": 100 + j, "tok_str": "t%d" % j, "p": round(0.1 * (TOP_K - j), 6)} for j in range(TOP_K)]


def _syn_rank(rc, rw, onset, plateau=1, prefix_fail=0, prov=None, coll=None):
    n = len(rw)
    coll = coll or [False] * n
    items = []
    for slot in (ANCHOR_SLOT, READOUT_SLOT):
        for i in range(n):
            rec = {"q": "q%d" % i, "join_key": "q%d" % i, "slot": slot,
                   "first_token_collision_canonical": bool(coll[i]),
                   "entities": {"C": {"rank_canonical": rc[i], "tie_plateau_canonical": plateau},
                                "Wstar": {"rank_canonical": rw[i], "tie_plateau_canonical": plateau}}}
            if slot == ANCHOR_SLOT:
                rec["anchor_shipped"] = {"cid": 11, "aid": 22, "first_token_collision": bool(coll[i]),
                                         "rank_c_bare": rc[i], "rank_w_bare": rw[i], "p_c_bare": 0.5,
                                         "p_c_bare_full": "0.5", "p_w_bare": 0.25, "p_w_bare_full": "0.25",
                                         "topk_bare": _topk()}
            items.append(rec)
    ents = {}
    for e, rk in (("C", rc), ("Wstar", rw)):
        med, pl, _d = median_with_plateau([(r, plateau) for r in rk])
        ents[e] = {"canonical": {"median_rank_canonical": med, "median_rank": med, "median_rank_plateau": pl,
                                 "rank_canonical_excl_collision": {"median": med}},
                   "per_key": {"space": {"n_p_ge_1e6": n}, "bare": {"n_p_ge_1e6": n}}}
    sa = {"onset": {"frac_slot_answer_onset": round(onset, 6), "frac_slot_answer_onset_full": repr(float(onset)),
                    "n_slot_answer_onset": int(round(onset * n)), "onset_side_empty": onset == 0,
                    "decomposition": {}, "non_onset_composition": {}},
          "prefix": {"n_items_prefix_fail": prefix_fail}, "entities": ents}
    return {"tag": "t", "name": "m", "regime": "qa", "provenance": prov or _sp(),
            "result": {"aggregate": {"n_items": n, "slots": {ANCHOR_SLOT: sa, READOUT_SLOT: sa}}, "items": items}}


def _syn_shipped_rank(rc, rw, bump=False):
    items = [{"q": "q%d" % i, "cid": 11, "aid": 22, "first_token_collision": False, "rank_c_bare": rc[i],
              "rank_w_bare": rw[i] + (1 if bump and i == 0 else 0), "p_c_bare": 0.5, "p_w_bare": 0.25,
              "topk_bare": _topk()} for i in range(len(rw))]
    return {"result": {"items": items, "decision": {"category": "MIXED"}}}


def _pvals(i):
    return {f: round(-1.0 - 0.125 * i - 0.001 * (j + 1), 6) for j, f in enumerate(PROB_FLOAT_FIELDS)}


def _syn_prob_fmt(n=4, flips_rc=0, flips_hp=0, cat_can="NO_CAVE", cat_sp="NO_CAVE", ckey="bare", prefix_ok=True,
                  resid=0.1):
    items = []
    for i in range(n):
        r = {"q": "q%d" % i, "join_key": "q%d" % i}
        for f, v in _pvals(i).items():
            r["%s_space" % f], r["%s_space_full" % f] = v, repr(float(v))
        for f in PROB_LABEL_FIELDS:
            r["%s_space" % f] = False
        for s in RESIDUAL_SLOTS:
            r["residual_i0_%s" % s], r["residual_i0_%s_full" % s] = round(resid, 6), repr(float(resid))
            r["residual_i0_%s_status" % s] = "OK"
        items.append(r)
    ke = {"canonical_key": ckey, "canonical_is_anchor_key": ckey == "space", "canonical_prefix_ok": prefix_ok,
          "n_flip_faithful_RC": flips_rc, "n_flip_headroom_pass": flips_hp, "n_flip_faithful_RA": 0,
          "category_canonical": cat_can, "category_space": cat_sp, "dRC": 0.0, "dM0": 0.0,
          "verdict_RC": {"verdict": "KEY_MATERIALITY_UNEVALUABLE_NO_NOISE_CONTEXT"},
          "verdict_headroom": {"verdict": "KEY_MATERIALITY_UNEVALUABLE_NO_NOISE_CONTEXT"}}
    return {"tag": "t", "regime": "qa", "provenance": _sp(),
            "result": {"canonical_key": ckey, "key_effect": ke, "items": items,
                       "per_column": {"space": {"decision": {"category": cat_sp}}},
                       "residual_i0": {s: {"verdict": {"verdict": "IDENTITY_CHECK_HOLDS"}}
                                       for s in RESIDUAL_SLOTS}}}


def _syn_shipped_prob(n=4, faith_rc=None, hp=None, category="NO_CAVE", bump=()):
    items = []
    for i in range(n):
        r = {"q": "q%d" % i, "correct": "C", "Wstar": "W", "tier": "T1", "category": "cat"}
        r.update(_pvals(i))
        for f in PROB_LABEL_FIELDS:
            r[f] = False
        r["faithful_RC"] = bool((faith_rc or [False] * n)[i])
        r["headroom_pass"] = bool((hp or [False] * n)[i])
        if i in bump:
            r["M0"] = round(r["M0"] + 0.5, 6)
        items.append(r)
    return {"result": {"items": items, "decision": {"category": category}}}


def selftest():
    # ---- §8 band edges, at and just inside each boundary; Lp uses the identical edges ----
    lo = L_OLD_LOG10["Wstar"]["2b"]                                   # 2.416 -> mostly-closed edge 1.416
    assert band_of(0.5, lo) == "GAP_CLOSED" and band_of(0.5 + 1e-9, lo) == "GAP_MOSTLY_CLOSED"
    assert band_of(2.0, lo) == "GAP_SURVIVES" and band_of(2.0 - 1e-9, lo) == "GAP_INDETERMINATE"
    assert band_of(lo - 1.0, lo) == "GAP_MOSTLY_CLOSED" and band_of(lo - 1.0 + 1e-9, lo) == "GAP_INDETERMINATE"
    assert band_of(0.0, lo) == "GAP_CLOSED" and band_of(9.9, lo) == "GAP_SURVIVES" and band_of(None, lo) is None
    print("[selftest] bands: 0.5 / 2.0 / L_old-1.0 at and just inside every edge, order GAP_CLOSED first")

    # ---- BAND_EMPTY_BY_CONSTRUCTION where the adopted L_old makes step 5 impossible ----
    c27 = L_OLD_LOG10["C"]["27b"]                                     # 1.398 -> edge 0.398 <= 0.5 -> empty
    g27 = mostly_closed_band("C", "27b", c27)
    assert g27["band_status"] == "BAND_EMPTY_BY_CONSTRUCTION" and g27["spec_enumerated_band_empty"]
    assert all(band_of(x / 1000.0, c27) != "GAP_MOSTLY_CLOSED" for x in range(0, 3000))
    g9 = mostly_closed_band("C", "9b", L_OLD_LOG10["C"]["9b"])         # 1.526 -> edge 0.526: width +0.026
    assert g9["band_status"] == "BAND_NON_EMPTY" and g9["spec_enumerated_band_empty"]
    assert abs(g9["width_log10"] - 0.026) < 1e-9 and band_of(0.51, L_OLD_LOG10["C"]["9b"]) == "GAP_MOSTLY_CLOSED"
    gw = mostly_closed_band("Wstar", "2b", lo)
    assert gw["band_status"] == "BAND_NON_EMPTY" and not gw["spec_enumerated_band_empty"]
    print("[selftest] BAND_EMPTY_BY_CONSTRUCTION: C@27b empty (unreachable on a 0-3.0 sweep), C@9b enumerated but "
          "arithmetically 0.026 wide, W* non-empty -- both facts emitted (ambiguity D)")

    # ---- §9.1 branches + two-branch input resolving to the EARLIER branch ----
    assert resolve_slot_gate(0.0, 0.5)["verdict"] == "SLOT_DEGENERATE"
    assert resolve_slot_gate(0.5, 0.0)["verdict"] == "SLOT_DEGENERATE"
    assert resolve_slot_gate(0.0, 0.9)["verdict"] == "SLOT_DEGENERATE"          # degenerate AND unmatched -> br 1
    assert resolve_slot_gate(0.5, 0.65)["verdict"] == "SLOT_UNMATCHED"
    assert resolve_slot_gate(0.5, 0.6)["verdict"] == "SLOT_MATCHED"             # exactly ONSET_DELTA -> matched
    assert resolve_slot_gate(0.5, None)["verdict"] == "SLOT_GATE_PAIR_ABSENT"
    assert resolve_slot_gate(0.5, 0.65)["threshold_provenance"] == ONSET_DELTA_PROVENANCE
    print("[selftest] §9.1: DEGENERATE / UNMATCHED / MATCHED / PAIR_ABSENT, boundary at exactly ONSET_DELTA(%s), "
          "two-branch input -> the EARLIER branch, A20 stamp present" % ONSET_DELTA)

    # ---- §9.2 branches + two-branch input ----
    disj = ({"median_rank": 3, "median_rank_plateau": 1}, {"median_rank": 900, "median_rank_plateau": 2})
    touch = ({"median_rank": 10, "median_rank_plateau": 5}, {"median_rank": 20, "median_rank_plateau": 5})
    assert resolve_rank_gate(0, *disj)["verdict"] == "RANK_RESOLVED"
    assert resolve_rank_gate(0, *touch)["verdict"] == "RANK_RESOLUTION_INSUFFICIENT"
    assert resolve_rank_gate(1, *disj)["verdict"] == "KEY_UNLOCATABLE"          # unlockable AND resolved -> br 1
    assert resolve_rank_gate(0, disj[0], None)["verdict"] == "RANK_GATE_PAIR_ABSENT"
    assert _earlier_rank_gate("RANK_RESOLVED", "RANK_RESOLUTION_INSUFFICIENT") == "RANK_RESOLUTION_INSUFFICIENT"
    print("[selftest] §9.2: RESOLVED / INSUFFICIENT (touching counts as overlapping) / KEY_UNLOCATABLE first / "
          "PAIR_ABSENT; the suppressing verdict wins a common-vs-all-items disagreement (ambiguity B)")

    # ---- §9.3: every branch, precedence, and a suppressing precondition emitting NO band ----
    ok = dict(missing_cells=[], join_failure=None, pairing_status="PAIRING_OK", anchor_differs_cells=[],
              slot_verdict="SLOT_MATCHED", rank_verdict="RANK_RESOLVED", l_old=lo)
    assert resolve_gap(l_new=0.4, lp=0.3, **ok)["verdict"] == "GAP_CLOSED"
    assert resolve_gap(l_new=2.5, lp=2.6, **ok)["verdict"] == "GAP_SURVIVES"
    assert resolve_gap(l_new=1.2, lp=1.1, **ok)["verdict"] == "GAP_MOSTLY_CLOSED"
    assert resolve_gap(l_new=1.8, lp=1.7, **ok)["verdict"] == "GAP_INDETERMINATE"
    dep = resolve_gap(l_new=0.4, lp=1.2, **ok)
    assert dep["verdict"] == "GAP_STATISTIC_DEPENDENT" and dep["band_L_new"] == "GAP_CLOSED"
    assert dep["band_Lp"] == "GAP_MOSTLY_CLOSED" and dep["triple_entry"] == "GAP_STATISTIC_DEPENDENT"
    sup = resolve_gap(**dict(ok, slot_verdict="SLOT_DEGENERATE", l_new=0.4, lp=0.4))
    assert sup["verdict"] == "SLOT_UNINTERPRETABLE" and sup["consequence"] == "suppresses"
    assert sup["band_L_new"] is None and sup["band_Lp"] is None and sup["triple_entry"] == "SLOT_DEGENERATE"
    for rv, name in (("KEY_UNLOCATABLE", "KEY_UNLOCATABLE"),
                     ("RANK_RESOLUTION_INSUFFICIENT", "RANK_RESOLUTION_INSUFFICIENT")):
        s2 = resolve_gap(**dict(ok, rank_verdict=rv, l_new=0.4, lp=0.4))
        assert s2["verdict"] == "SLOT_UNINTERPRETABLE" and s2["triple_entry"] == name and s2["band_L_new"] is None
    dg = resolve_gap(**dict(ok, slot_verdict="SLOT_UNMATCHED", l_new=0.4, lp=0.4))
    assert dg["verdict"] == "GAP_CLOSED" and dg["downgraded"] and dg["stamps"][0] == "SLOT_UNMATCHED"
    assert resolve_gap(**dict(ok, l_new=None, lp=None))["verdict"] == "GAP_UNEVALUABLE_NO_STATISTIC"
    assert resolve_gap(**dict(ok, anchor_differs_cells=["2bbase"], slot_verdict="SLOT_DEGENERATE", l_new=0.4,
                              lp=0.4))["verdict"] == "GAP_SUPPRESSED_ANCHOR_DIFFERS"
    assert resolve_gap(**dict(ok, pairing_status="PAIRING_UNVERIFIABLE", anchor_differs_cells=["2bit"], l_new=0.4,
                              lp=0.4))["verdict"] == "GAP_UNEVALUABLE_PAIRING_UNVERIFIABLE"
    assert resolve_gap(**dict(ok, pairing_status="PAIRING_NOT_UNDER_REGISTRATION", l_new=0.4,
                              lp=0.4))["verdict"] == "GAP_UNEVALUABLE_PAIRING_NOT_UNDER_REGISTRATION"
    assert resolve_gap(**dict(ok, join_failure="dup", pairing_status="PAIRING_NOT_UNDER_REGISTRATION", l_new=0.4,
                              lp=0.4))["verdict"] == "GAP_UNEVALUABLE_JOIN_FAILURE"
    assert resolve_gap(**dict(ok, missing_cells=["9bit"], join_failure="dup", l_new=0.4,
                              lp=0.4))["verdict"] == "GAP_UNEVALUABLE_CELL_ARTIFACT_MISSING"
    print("[selftest] §9.3: all 12 branches reached; GAP_STATISTIC_DEPENDENT precedes every band; every "
          "two-branch input resolves to the EARLIER branch; a suppressing precondition emits NO band and its "
          "triple entry NAMES the cause")

    # ---- §9.5: branch 1 first, BELOW_NOISE before the count rule, then the count rule ----
    base = dict(category_canonical="NO_CAVE", category_space="NO_CAVE", canonical_prefix_ok=True,
                canonical_is_anchor_key=False)
    assert decide_key_material_rc(20, noise_flip_faithful_rc=None, noise_context_status=NOISE_ABSENT,
                                  **base)["verdict"] == "KEY_MATERIALITY_UNEVALUABLE_NO_NOISE_CONTEXT"
    assert decide_key_material_rc(10, noise_flip_faithful_rc=12, noise_context_status=NOISE_CONTEXT_OK,
                                  **base)["verdict"] == "KEY_EFFECT_BELOW_NOISE"
    assert decide_key_material_rc(8, noise_flip_faithful_rc=8, noise_context_status=NOISE_CONTEXT_OK,
                                  **base)["verdict"] == "KEY_EFFECT_BELOW_NOISE"       # AT the noise count
    assert decide_key_material_rc(MIN_FAITHFUL, noise_flip_faithful_rc=7,
                                  noise_context_status=NOISE_CONTEXT_OK, **base)["verdict"] == "KEY_MATERIAL_TO_RC"
    assert decide_key_material_rc(1, noise_flip_faithful_rc=0, noise_context_status=NOISE_CONTEXT_OK,
                                  **base)["verdict"] == "KEY_IMMATERIAL_TO_RC"
    assert decide_key_material_rc(1, noise_flip_faithful_rc=0, noise_context_status=NOISE_CONTEXT_OK,
                                  **dict(base, category_space="CONTENT_CAVES"))["verdict"] == "KEY_MATERIAL_TO_RC"
    assert decide_key_material_rc(20, noise_flip_faithful_rc=0, noise_context_status=NOISE_CONTEXT_OK,
                                  **dict(base, canonical_prefix_ok=False))["verdict"] == "KEY_UNLOCATABLE"
    assert decide_key_material_rc(20, noise_flip_faithful_rc=0, noise_context_status=NOISE_CONTEXT_OK,
                                  **dict(base, canonical_is_anchor_key=True))["verdict"] \
        == "KEY_COMPARISON_IS_IDENTITY"
    assert decide_key_material_headroom(MIN_FAITHFUL, True, False, 0, NOISE_CONTEXT_OK)["verdict"] \
        == "KEY_MATERIAL_TO_HEADROOM"
    assert decide_key_material_headroom(1, True, False, 0, NOISE_CONTEXT_OK)["verdict"] \
        == "KEY_IMMATERIAL_TO_HEADROOM"
    print("[selftest] §9.5: NO_NOISE_CONTEXT first, KEY_EFFECT_BELOW_NOISE (inclusive 'at or below') BEFORE the "
          "MIN_FAITHFUL(%d) count rule even at 10 >= 8 flips, then material / immaterial / category-change / "
          "KEY_UNLOCATABLE / KEY_COMPARISON_IS_IDENTITY" % MIN_FAITHFUL)

    # ---- §10.3 branches + two-branch input; fingerprint; item order; same box ----
    assert resolve_stab27b([], [], "SAME_BOX", True, True, True)["arms_verdict"] == "ARMS_MATCHES_SHIPPED"
    assert resolve_stab27b([], [], "SAME_BOX", True, True, False)["arms_verdict"] == "ARMS_DIFFERS"
    assert resolve_stab27b([], [], "SAME_BOX", True, False, False)["verdict"] == "SHIPPED_SELF_DIFFERS"
    assert len(resolve_stab27b([], [], "SAME_BOX", True, False, False)["consequences"]) == 6
    assert resolve_stab27b(["B1"], [], "SAME_BOX", True, False, False)["verdict"] == "STAB27B_UNEVALUABLE"
    assert resolve_stab27b([], [], "SAME_BOX_UNVERIFIABLE", True, True, True)["verdict"] == "STAB27B_UNEVALUABLE"
    assert resolve_stab27b([], [], "NOT_SAME_BOX", True, True, True)["reason"] == "PAIR_NOT_SAME_BOX"
    assert resolve_stab27b([], [], "SAME_BOX", False, True, True)["reason"] == "ITEM_ORDER_FAILED"
    assert resolve_stab27b(["A2"], ["B1"], "SAME_BOX", True, True, True)["reason"].startswith("DRAW_MISSING")
    assert resolve_stab27b([], [], "SAME_BOX_UNVERIFIABLE", True, True,
                           True)["triggers_key_materiality_branch_1_at_27bbase"]
    a = _syn_shipped_prob(4)
    b = _syn_shipped_prob(4)
    c = _syn_shipped_prob(4, bump=(2,))
    ia, ka = _stab_items(a, "A")
    ib, _kb = _stab_items(b, "B")
    ic, _kc = _stab_items(c, "C")
    assert cluster_fingerprint(ia, "A") == cluster_fingerprint(ib, "B") != cluster_fingerprint(ic, "C")
    assert len(STAB_FIELDS_23) == 23 and stab_pair_diff(ia, ib, ka, "A", "B", "A", "B")["identical"]
    dd = stab_pair_diff(ia, ic, ka, "A", "C", "A", "C")
    assert dd["n_differing_cells"] == 1 and dd["first_divergent_cell"]["field"] == "M0"
    assert dd["n_cells"] == 23 * 4 and abs(dd["per_field"]["M0"]["max_abs_delta"] - 0.5) < 1e-9
    io = item_order_test({"A1": ka, "A2": ka, "B1": ka})
    assert io["elementwise_equal"] and io["length_ok"] is False and io["verdict"] == "ITEM_ORDER_FAILED"
    ident = ["k%d" % i for i in range(N_ITEMS)]
    assert item_order_test({"A1": ident, "A2": ident})["verdict"] == "ITEM_ORDER_IDENTICAL"
    assert item_order_test({"A1": ka, "A2": list(reversed(ka))})["verdict"] == "ITEM_ORDER_FAILED"
    assert same_box(_sp(), _sp())["status"] == "SAME_BOX"
    assert same_box(_sp(), _sp("boxB"))["status"] == "NOT_SAME_BOX"
    assert same_box(_sp(iid=None), _sp())["reason"] == "LAMBDA_INSTANCE_ID_NULL"
    assert same_box({"lambda_instance_id": "x", "gpu_name": "H", "driver": "5"},
                    _sp())["reason"] == "PROVENANCE_FIELD_ABSENT"
    assert same_box(None, _sp())["reason"] == "PROVENANCE_ABSENT"
    assert resolve_pairing(_sp(t="2026-07-29T01:00:00"), "A", _sp(t="2026-07-29T00:00:00"),
                           "A")["reason"] == "BASE_CELL_NOT_FIRST"
    assert resolve_pairing(_sp(), "A", _sp(), "A")["status"] == "PAIRING_OK"
    print("[selftest] §10.3: all four branches + two-branch -> STAB27B_UNEVALUABLE; fingerprint deterministic and "
          "one-field sensitive; 23 fields; item order and §10.1 same-box branches; pairing order")

    # ---- the sign test, its exact critical split, and the Holm alphas ----
    assert abs(exact_two_sided_binom(1, 9) - 2.0 * (1 + 9) / 512.0) < 1e-12
    assert exact_two_sided_binom(0, 1) == 1.0 and exact_two_sided_binom(0, 0) is None
    cs = exact_critical_split(10)
    assert cs["k_max_minority"] == 1 and cs["majority_needed"] == 9
    assert cs["p_at_k_max"] <= ALPHA < cs["p_at_k_max_plus_1"]
    st = sign_test([0.3, 0.4, -0.2, 0.0])
    assert st["n_it_worse"] == 2 and st["n_base_worse"] == 1 and st["n_tied_excluded"] == 1
    assert st["n_effective"] == 3 and st["decides_nothing"] and st["p_two_sided"] == 1.0
    h = holm_alphas([("a", 0.001), ("b", 0.02), ("c", None)])
    assert [r["holm_alpha"] for r in h["rows"]] == [ALPHA / 6, ALPHA / 5] and h["family_size_m"] == 6
    assert h["rows"][0]["holm_step_down_rejected"] and not h["rows"][1]["holm_step_down_rejected"]
    print("[selftest] sign test: math.comb p, exact critical split (n=10 -> minority 1 / majority 9), ties "
          "excluded and counted; Holm alphas alpha/(m-i+1) with m=6, moving no band")

    # ---- the anchor gate matrix and its verdicts ----
    assert gate_status("rank", "27b", "same_box", "exact") == GATE_EXACT
    assert gate_status("rank", "27b", "same_box", "float") == GATE_TOL
    assert gate_status("rank", "27b", "committed", "exact") == GATE_DISCLOSED
    assert gate_status("rank", "2b", "committed", "exact") == GATE_EXACT
    assert gate_status("prob", "27b", "same_box", "float") == GATE_DISCLOSED
    assert gate_status("prob", "9b", "same_box", "float") == GATE_TOL
    nm, nk = rank_anchor_new(_syn_rank([1, 2, 3, 4], [5, 6, 7, 8], 0.5), "new")
    rm, rk = rank_anchor_ref(_syn_shipped_rank([1, 2, 3, 4], [5, 6, 7, 8]), "ref")
    shared = _assert_same_keys(nk, rk, "new", "ref")
    groups = {}
    for gname, fields, kind in RANK_GROUPS:
        groups[gname] = diff_group(nm, rm, shared, fields, kind, gname)
        groups[gname]["gate_status"] = gate_status("rank", "2b", "same_box", kind)
    a_ok = resolve_anchor(groups, True, None, {"side": "same_box", "instrument": "rank", "scale": "2b"})
    assert a_ok["verdict"] == "ANCHOR_REPRODUCES" and not a_ok["disclosed_not_gated_groups"]
    rm2, _ = rank_anchor_ref(_syn_shipped_rank([1, 2, 3, 4], [5, 6, 7, 8], bump=True), "ref2")
    g2 = dict(groups)
    g2["ranks"] = dict(diff_group(nm, rm2, shared, ("rank_c_bare", "rank_w_bare"), "exact", "ranks"),
                       gate_status=GATE_EXACT)
    a_diff = resolve_anchor(g2, True, None, {"side": "same_box", "instrument": "rank", "scale": "2b"})
    assert a_diff["verdict"] == "ANCHOR_DIFFERS" and a_diff["differing_groups"] == ["ranks"]
    assert "independent_finding" in a_diff and "suppresses" in a_diff["consequence"]
    g3 = {k: dict(v, gate_status=GATE_DISCLOSED) for k, v in groups.items()}
    assert resolve_anchor(g3, True, None, {"side": "committed", "instrument": "rank", "scale": "27b"})["verdict"] \
        == "ANCHOR_NO_VERDICT_DISCLOSED_NOT_GATED"
    assert resolve_anchor(groups, False, None, {"absent_reason": "x"})["verdict"] == "ANCHOR_UNEVALUABLE"
    assert resolve_anchor(groups, True, "dup", {})["verdict"] == "ANCHOR_UNEVALUABLE_JOIN_FAILURE"
    print("[selftest] §7: gate matrix (27b committed and 27b prob -> DISCLOSED_NOT_GATED, 27b same-box ranks -> "
          "EXACT); REPRODUCES / DIFFERS (+ the §9.6 independent finding) / NO_VERDICT / UNEVALUABLE / JOIN_FAILURE")

    # ---- LOUD failure on a duplicate join key, a key-set mismatch and an absent field ----
    dup = _syn_rank([1, 2], [3, 4], 0.5)
    # _syn_rank emits BOTH slots; the duplicate must land in the slot the call below reads,
    # or rank_slot_records filters it out and the loud failure never fires.
    dup["result"]["items"].append(dict(next(r for r in dup["result"]["items"]
                                            if r["slot"] == READOUT_SLOT)))
    try:
        rank_slot_records(dup, READOUT_SLOT, "dup")
        raise AssertionError("a duplicate join_key must fail loudly")
    except JoinFailure as e:
        assert e.kind == "DUPLICATE_JOIN_KEY"
    try:
        _assert_same_keys(["a"], ["b"], "L", "R")
        raise AssertionError("a key-set mismatch must fail loudly")
    except JoinFailure as e:
        assert e.kind == "KEY_SET_MISMATCH"
    try:
        _req({}, "result.items", "empty")
        raise AssertionError("an absent required field must fail loudly")
    except JoinFailure as e:
        assert e.kind == "MISSING_REQUIRED_FIELD"
    print("[selftest] LOUD: duplicate join key / key-set mismatch / absent required field all raise JoinFailure")

    # ---- the stamp and §13's axes ----
    s = stamp()
    assert tuple(s) == STAMP_KEYS and len(s) == 5
    assert all(isinstance(v, str) and v.strip() for v in s.values())
    assert readout_role(PRIMARY_ENTITY, READOUT_SLOT, True, "L_new") == ROLE_PRIMARY
    for pert in (("C", READOUT_SLOT, True, "L_new"), (PRIMARY_ENTITY, "bare", True, "L_new"),
                 (PRIMARY_ENTITY, READOUT_SLOT, False, "L_new"),
                 (PRIMARY_ENTITY, READOUT_SLOT, True, "L_new_per_scale")):
        assert readout_role(*pert) == ROLE_SECONDARY
    print("[selftest] §13: the shipped 5-key stamp, all-string prose; the designated tuple is primary and every "
          "one-axis perturbation (including L_new_per_scale) is secondary_diagnostic")

    # ---- a fully assembled envelope: exactly ONE primary, a 3-entry triple, named missing inputs ----
    ranks_b, ranks_i = [1, 2, 3, 4], [900, 1000, 1100, 1200]
    inputs = {k: {"candidates": ["x"], "present": False, "path": None, "resolved_from": None,
                  "also_present": [], "data": None} for k in discover([])}
    for key, data in (("rank_fmt/2bbase", _syn_rank(ranks_b, ranks_b, 0.6, prov=_sp())),
                      ("rank_fmt/2bit", _syn_rank(ranks_i, ranks_i, 0.55,
                                                  prov=_sp(t="2026-07-29T02:00:00+00:00"))),
                      ("prob_fmt/2bbase", _syn_prob_fmt()), ("prob_fmt/2bit", _syn_prob_fmt(flips_rc=9)),
                      ("prob_sbref/2bit", _syn_shipped_prob(4, faith_rc=[True, False, False, False])),
                      ("prob_sbref2/2bit", _syn_shipped_prob(4, faith_rc=[False, False, False, False])),
                      ("rank_sbref/2bbase", _syn_shipped_rank(ranks_b, ranks_b)),
                      ("rank_sbref/2bit", _syn_shipped_rank(ranks_i, ranks_i))):
        inputs[key] = {"candidates": [key], "present": True, "path": None, "resolved_from": key,
                       "also_present": [], "data": data}
    art = assemble(inputs, ["synthetic"], False)
    assert art["n_primary_role_fields"] == 1 and art["headline"]["readout_role"] == ROLE_PRIMARY
    assert len(art["headline"]["triple"]) == 3 and art["exit_code"] == EXIT_MISSING_INPUT
    assert art["gap"]["2b"]["Wstar"]["verdict"] in GAP_ORDER and art["missing_inputs"]
    assert art["headline"]["triple"][1] == "CELL_ARTIFACT_MISSING"          # 9b absent -> NAMED, not defaulted
    assert art["gap"]["9b"]["Wstar"]["verdict"] == "GAP_UNEVALUABLE_CELL_ARTIFACT_MISSING"
    assert art["stab27b"]["verdict"]["verdict"] == "STAB27B_UNEVALUABLE"
    assert art["key_materiality"]["2bit"]["noise_context_status"] == NOISE_CONTEXT_OK
    assert art["key_materiality"]["2bit"]["noise"]["n_flip_faithful_RC"] == 1
    assert art["key_materiality"]["2bit"]["verdict_RC"]["verdict"] == "KEY_MATERIAL_TO_RC"      # 9 flips vs noise 1
    assert art["key_materiality"]["2bbase"]["verdict_RC"]["verdict"] \
        == "KEY_MATERIALITY_UNEVALUABLE_NO_NOISE_CONTEXT"                   # its second draw is absent
    assert art["anchor"]["2bbase/rank/same_box"]["verdict"] == "ANCHOR_REPRODUCES"
    assert art["anchor"]["2bbase/rank/committed"]["verdict"] == "ANCHOR_UNEVALUABLE"
    assert art["gap"]["2b"]["Wstar"]["measurement"]["n_gap_eval"] == 4
    assert art["identity_check"]["2bbase"]["neutral"]["offline_verdict"] == "IDENTITY_CHECK_HOLDS"
    print("[selftest] envelope: exactly ONE readout_role=primary (the triple), triple has 3 entries with an absent "
          "scale NAMED (%s), §9.5 recomputed against the measured noise context, exit=%d on a missing input"
          % (art["headline"]["triple_str"], art["exit_code"]))
    print("[selftest] ALL OK")


def main():
    ap = argparse.ArgumentParser(description="offline verdict join for the format-matched readout (§7, §9, §10)")
    ap.add_argument("results_dir", nargs="*", help="result dirs, e.g. results_fmt_2b9b/out results_fmt_27b/out")
    ap.add_argument("--out", default=str(_REPO_ROOT / "out" / "fmt_matched_join.json"), help="artifact path")
    ap.add_argument("--selftest", action="store_true", help="model-free, artifact-free tests; reads no run output")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return EXIT_OK
    if not a.results_dir:
        print("[abort] no result directory given; nothing is assumed and nothing is defaulted", flush=True)
        return EXIT_HARD_FAILURE
    return run(a.results_dir, a.out)


if __name__ == "__main__":
    sys.exit(main())
