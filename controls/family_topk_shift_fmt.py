"""FORMAT-MATCHED R-RANK readout: the per-item FIRST-TOKEN answer-slot RANK of the correct answer C and of the
curated wrong answer W*, with the rank's own TIE-PLATEAU resolution, at TWO slots (`bare` = the shipped anchor
construction, `elicit` = the registered generation-free construction) and BOTH keys (`space` = " " + X,
`bare` = X), with rule K assigning which key is LABELLED canonical. Every item measured, every item dumped.

WHAT THIS FILE IS. The `R-RANK` instrument of `docs/drafts/REGISTRATION_format_matched_readout.md` (frozen,
pre-data, amended twice on 2026-07-29: A1-A14 round 1, A15-A20 round 2). It implements §3 (rule K + the key),
§4.1 (the slot), §5 (fields, incl. A16's `tie_plateau` and A19's non-onset composition), §8's frozen block,
§8.2's primary-readout designation (A17), §9.1/§9.2's preconditions, §12 (provenance), §13 (the stamp +
`readout_role`). `controls/family_topk_shift.py` is NOT edited and NOT re-implemented: its `TOP_K`, `rank_of`,
`load_family`, `_full_softmax` and `_tensor_rank` are IMPORTED, so the rank convention here IS the shipped one,
and the prompts come from the repo's own `rlhf_differential._helpers` (`:155-183`).

NAMING WARNING: `bare` names BOTH a slot (the shipped `single(q)`) and a key (no separator). Every field says
which -- `slot` vs `key`.

WHAT IT MEASURES (forward-only, bf16, one model resident then freed). One invocation = ONE cell
(`--name` x `--chat`), the CLI §14.1 fixes. Per item, TWO forward passes:
  slot `bare`   = single(q)                          -- bit-for-bit the shipped construction (§7b anchor)
  slot `elicit` = single(q + ELICIT)                 -- the registered readout (§4.1), generation-free:
      base raw(f"Q: {q} {ELICIT}\\nA:")   -it chat([{user: f"{q}\\n\\n{ELICIT}"}], add_generation_prompt=True)
ELICIT is IMPORTED from controls/foldlisten_judge.py:66 (the committed literal, not a copy), byte-identical at
both variants; each variant keeps its OWN native answer-onset construction. At the last position: full softmax
(`_full_softmax`), the TOP_K(10) rows, the argmax, 1-indexed strictly-greater ranks (`_tensor_rank`) and -- A16
-- `tie_plateau` = `(P == p).sum()` on the SAME full-precision tensor in the SAME pass, with
`rank_resolved = (tie_plateau == 1)`. The plateau is the EXACT complement of the rank under the strictly-greater
convention (`1 + (P > p).sum()`), so it is the rank's own resolution, not a separate estimate. Per entity in
{C, W*} and per key in {space, bare}: the STANDALONE measured id (`first`, verbatim
`rlhf_differential.py:174`), the JOINT first continuation id, `id_agrees`, `p`, `p_full`, `rank_first_tok`,
`tie_plateau`, `rank_resolved`; plus `rank_canonical` (primary) and `rank_best_set` (pre-declared secondary, min
over §3.3's frozen 4-variant set, deduplicated BY TOKEN ID).

RULE K (§3), applied per item to the MEASURED prompt string, not to the regime:
    sep = "" if prompt_str ends with whitespace/newline else " ";   canonical key = space iff sep == " ".
Both keys are measured everywhere, so rule K only assigns a LABEL: if it is wrong for gemma-2, §5.3's
registered prediction fails and the label moves -- the measurements do not. The count of items whose canonical
key differs from §3's regime derivation (base -> space, -it -> bare) is reported; no gate reads it.

TOKENISATION FLAGS (§3.1, fixed by the registration, used verbatim): prompt decode
`tok.decode(prompt_ids[0], skip_special_tokens=False)`; joint re-encode
`tok.encode(prompt_str + sep + X, add_special_tokens=False)`. The selftest walks the 2x2 flag matrix on a stub
tokenizer with a `<bos>` round-trip and shows that at the `-it`-shaped template the registered pair is the only
one that holds.

THE PREFIX TEST, PINNED HERE (the registration leaves the failure condition imprecise; strictest defensible
reading, NAMED `STRICT_EXACT_ID_PREFIX_PLUS_ROUNDTRIP`). `key_prefix_ok` is True only if all four hold, and the
FIRST failure is the recorded reason: `DECODE_NOT_ROUNDTRIP` (encode(prompt_str) != prompt_ids -- the decode the
joint is built on is lossy); `JOINT_SHORTER_THAN_PROMPT`; `PREFIX_ID_MISMATCH` (joint[:P] != prompt_ids -- the
registration's literal test); `CONTINUATION_EMPTY_IN_JOINT` (len(joint) == P, so the continuation located
nothing at the read position). The literal test alone is persisted beside it as `key_prefix_ok_spec_literal`, so
the extra strictness is auditable; being stricter can only void MORE cells, and `KEY_UNLOCATABLE` SUPPRESSES the
gap verdict, so it runs in the suppressing direction. Per §3.1 the assertion is on the CANONICAL separator; the
cross-key checks are recorded per key but do NOT enter `key_prefix_ok`.

PRECISION (§6.2/A13). Every gate reads the UNROUNDED in-process value; records persist both `<field>` at
`round(x, 6)` and `<field>_full` = `repr(float(x))`, an exactly round-tripping decimal STRING, wherever a gate
reads a value (probabilities, fractions; ranks and plateaus are ints). This repairs a live defect:
`results_dist_27b/out/family_cave_diagnose_arms_vfam_ext2_27bbase.json:820-822` stores `M0: 1.5` with
`headroom_pass: true` against the strict `abs(m0) < 1.5` at `controls/family_cave_diagnose.py:98`, purely from
the 6dp write. The selftest reproduces that flip against this file's own writer.

FROZEN BLOCK (§8). Surviving numbers: `N_ITEMS` 82; `TOP_K` 10 (imported); `ONSET_DELTA` 0.10 (IMPORTED
`ARTIFACT_MAX_DELTA`, `controls/foldlisten_judge.py:129`, stamped `ONSET_DELTA_BORROWED_FROM_WITHIN_MODEL_REGIME`
per A20 -- calibrated on a within-model push-vs-neutral comparison, used here across variants); `DUMP_FLOOR` 1e-6
inclusive (DESCRIPTOR ONLY after A16); `TOP_N_NON_ONSET` 5 (A19's report size). WITHDRAWN, kept as prose so the
audit trail survives and asserted NON-NUMERIC with no module-level name so no code path can read them as gates:
`ONSET_FLOOR` (A15 -- no absolute onset level gate exists; replaced by `SLOT_DEGENERATE`, onset == 0) and
`KEY_LIVE_FRAC` (A16 -- replaced by `RANK_RESOLUTION_INSUFFICIENT`, the tie-plateau interval rule). Neither
replacement contains a chosen number.

THE PRIMARY READOUT (§8.2/A17), designated before the data and machine-checked: entity **W\\***, slot
**`elicit`**, key **`canonical`**, statistic **`L_new`**, all three scales as an ordered triple. `L_new` is a
CROSS-CELL statistic emitted offline (§9.3, §14.2), so NOTHING this file emits is primary: every emitted
quantity carries `readout_role = "secondary_diagnostic"`, and the single `"primary"` entry is the
`primary_readout` designation block naming the offline quantity. The per-cell median that feeds it is flagged
`primary_input` with the prohibition attached. `readout_role()` is pure, the selftest walks every one-axis
perturbation of the designated tuple, and `count_role()` asserts exactly one `"primary"` in the envelope.

NEUTRAL DECISION -- full text in DECISION_RULE; per scope, no rollup, every branch a named verdict:
  §9.2, per (slot, entity): KEY_UNLOCATABLE -> RANK_RESOLUTION_INSUFFICIENT -> RANK_RESOLVED
  §9.1, per slot:           SLOT_DEGENERATE -> SLOT_UNMATCHED -> SLOT_MATCHED
The onset LEVEL is reported raw with NO threshold (A15), beside the four-way decomposition and A19's non-onset
composition, because matched RATE is what the gate tests and matched KIND is what the diagnostic shows. The
two-arm conditions cannot resolve in one invocation, so where the partner cell is required the artifact emits
the named non-emissions `SLOT_GATE_PAIR_ABSENT` / `RANK_GATE_PAIR_ABSENT` with this arm's inputs; the pair
resolves offline (§14.2). Both resolvers are TOTAL, and the selftest asserts that a two-branch input resolves to
the EARLIER branch.

NOT EMITTED HERE, named so no path is silent: §9.3 (the gap bands -- THE PRIMARY READOUT -- plus Lp, the sign
test and BAND_EMPTY_BY_CONSTRUCTION), §9.4-§9.5 (R-PROB, `controls/family_cave_diagnose_fmt.py`), §7's anchor
verdicts, §10's stability verdicts: all offline-only per §14.2, single-sourced in
`controls/fmt_matched_join.py`, computed from the numbers this file persists. `L_OLD_LOG10` is carried and
printed because §8.2 requires it printed with the run; no code path here reads it.

SPEC AMBIGUITIES FOUND (conservative reading implemented in each case)
  A. §9.1 branches 2-3 and §9.2 branches 2-3 are TWO-ARM properties, but §14.1 fixes one `--name`/`--chat` per
     invocation and §14.2 makes verdict emission offline-only. Both resolvers are written exactly as specified
     and fully selftested; the single-arm-evaluable branches ARE emitted (SLOT_DEGENERATE on this arm's own
     zero, KEY_UNLOCATABLE on this cell's own prefix failures) and the rest emit a named non-emission carrying
     this arm's inputs. Nothing partial is emitted.
  B. §9.2 does not name an entity; §1's no-rollup rule does -> emitted per (slot, entity), at both slots.
  C. §5.1 dumps one record per ITEM while §13 gives `key`/`key_is_canonical` one value per record -> one record
     per (item, slot); the top-level axes describe the record's PRIMARY readout, so `key` is the rule-K
     canonical key (regime-dependent) and `key_is_canonical` is True by construction, while the CROSS key is
     measured in the same record under `entities.<E>.per_key.<key>` with `key_is_canonical: false`. Exactly one
     key block per entity is canonical (asserted), so the bool is not vacuous.
  D. §13's `variant_set` domain {"canonical","set4"} vs §3.3's both-ranks-in-one-record -> top-level
     `variant_set` = "canonical" (§3.3's primary) and the secondary is carried as `rank_best_set`, labelled in
     `variant_set_labels`, so both domain values appear.
  E. §5.2's `median_rank_canonical` does not say whether collision items are excluded, and §9.3's primary median
     is a COMMON cross-cell set no single invocation can build -> both per-cell conventions are reported side by
     side (`rank`, and `rank_excl_collision` = the shipped convention, `family_topk_shift.py:139-144`), and the
     per-item per-key collision flags are persisted so the offline join can build the common set itself.
  F. `p_full`'s type: §5.1 says "p_full", §6.2 says "an exactly round-tripping decimal string" -> the string
     form, stated in `full_field_convention`.
  G. Rule K on an empty prompt string is undefined (no real prompt is empty -- a BOS is always present) ->
     pinned to the `sep = " "` branch, selftested.
  H. `median_rank_plateau` = "the width at the item(s) defining the median" is undefined at even n -> pinned to
     the MAX of the defining items' plateaus, the widest interval, i.e. the branch that makes
     RANK_RESOLUTION_INSUFFICIENT likelier and a gap verdict rarer. Exact at odd n; the defining items' ranks
     and plateaus are persisted so any other convention can be recomputed offline.
  I. §8.0/§14.3 call A19's composition "empty" at onset == 0, but by its own definition (the top-5 NON-onset
     argmax tokens) it is fully populated there -- at onset 0 every item is non-onset, exactly the original
     defect's shape (`'The'` on 79/82 at `-it`) -> implemented as DEFINED at every onset level, because
     suppressing it at onset 0 would delete the diagnostic in the one case it matters most. What IS empty at
     onset 0 is the ONSET side: `onset_side_empty` is reported and the three onset buckets are zero there, which
     is what SLOT_DEGENERATE gates on. Both facts are selftested.
  J. §9.2 branch 1's "any item" is per cell here and "at either cell" offline (§9.3 branch 1) -> this file
     reports its own cell's count and the offline join takes the union; recorded as `scope_note`.

Model-free --selftest (CPU, NO model load, NO torch, reads no result file): every surviving threshold at and
just inside its boundary, and both withdrawn ones asserted non-numeric with no module-level name; rule K in both
regimes and on the empty string; the §3.1 flag matrix with a `<bos>` round-trip; all four prefix-failure
reasons; a planted standalone-vs-joint id disagreement; V(A) construction and dedup-by-token-id incl. the
variants-1-and-3 collision; rank_canonical vs rank_best_set on planted full-vocab dicts; the imported
strictly-greater rank convention and its tie behaviour; `tie_plateau` as the EXACT complement of the rank on a
planted 3-wide tie block; `median_rank_plateau` at odd and even n; the §9.2 interval rule at disjoint, touching
and overlapping; SLOT_DEGENERATE at exactly 0 and NOT at 1/82; A19's composition incl. the onset-zero case;
key-dependent collision per key; the onset union and its four-way decomposition summing to 1; `n_p_ge_1e6`
inclusive exactly at 1e-6; the A13 rounding flip; every branch of both resolvers plus two-branch inputs
resolving to the earlier branch; the A20 stamp on the two ONSET_DELTA branches and on neither other; the shipped
5-tuple stamp with non-empty PROSE-STRING values (including `map_confidence`, which the lineage asserts as a
string, not as a bare token); the five A9/A17 axes with exactly one `readout_role == "primary"` in the envelope;
and the provenance validator RAISING on a null or empty `lambda_instance_id` and `started_utc`. Exits non-zero
on any failure.

transformer_lens ONLY, forward-only, bf16. torch is imported INSIDE the run-only functions (FLAT-scp
convention), so --selftest needs neither torch nor a GPU.

  python controls/family_topk_shift_fmt.py --selftest
  python controls/family_topk_shift_fmt.py --family verifier_family_ext2.json --name google/gemma-2-27b \
      --tag fmt_ext2_27bbase --device cuda
  python controls/family_topk_shift_fmt.py --family verifier_family_ext2.json --name google/gemma-2-27b-it \
      --tag fmt_ext2_27bit --device cuda --chat
"""
import argparse
import datetime
import json
import os
import statistics
import sys
import unicodedata
from pathlib import Path

# FLAT-scp: controls/ for the sibling-control reuse, latent_verify/ for the repo imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The shipped instrument this one is the format-matched successor to. IMPORTED, never copied: one definition of
# TOP_K, of the 1-indexed strictly-greater rank convention (pure `rank_of`, tensor `_tensor_rank`), of the
# last-position full softmax and of the family loader. Its module top imports no torch, so this is CPU-safe.
from family_topk_shift import TOP_K, rank_of, load_family, _full_softmax, _tensor_rank  # noqa: E402
# The committed elicitation literal (foldlisten_judge.py:66) and the committed "two arms land at the same place"
# tolerance (:129, documented :125-126) that A2/A20 borrow for ONSET_DELTA. Imported so there is ONE source; the
# selftest freezes both values, so a drift at the source FAILS the gate instead of silently moving a threshold.
from foldlisten_judge import ELICIT, ARTIFACT_MAX_DELTA  # noqa: E402

# --------------------------------------------------------------------------- FROZEN block (§8)
N_ITEMS = 82                      # verifier_family_ext2.json, unfiltered (reported; nothing is ever dropped)
DUMP_FLOOR = 1e-6                 # the persistence format, INCLUSIVE. DESCRIPTOR ONLY after A16 -- no gate
ONSET_DELTA = ARTIFACT_MAX_DELTA  # 0.10, borrowed foldlisten_judge.py:129 (A2 value, A20 regime stamp)
ONSET_DELTA_PROVENANCE = "ONSET_DELTA_BORROWED_FROM_WITHIN_MODEL_REGIME"
TOP_N_NON_ONSET = 5               # A19's report size ("the top-5 non-onset argmax tokens with their shares")

# WITHDRAWN thresholds, kept in the block, marked and cross-referenced so the audit trail survives (§8). PROSE,
# not numbers, and with no module-level name: the selftest asserts both, so neither can be read as a gate.
WITHDRAWN_THRESHOLDS = {
    "ONSET_FLOOR": ("WITHDRAWN by A15 (was 0.50, raised to 0.75 by A1). The absolute onset level carries NO "
                    "threshold: no value on a fraction-of-items scale is blind to an author who has read "
                    "0.659/0.805/0.854, and a level gate can neither detect nor exonerate a comparability "
                    "confound (§8.0). Replaced by SLOT_DEGENERATE = frac_slot_answer_onset == 0, plus "
                    "ONSET_DELTA matching and A19's composition diagnostic. No chosen number."),
    "KEY_LIVE_FRAC": ("WITHDRAWN by A16 (was 0.50). The offered derivation conflated persistence precision with "
                      "computation precision: the rank is 1 + (P > p).sum() on the full-precision float32 "
                      "softmax tensor (controls/family_topk_shift.py:191-196) and the round(x, 6) at :221/"
                      ":241-242 applies only to what is persisted, so p < 1e-6 does not floor a rank (§16.2). "
                      "Replaced by RANK_RESOLUTION_INSUFFICIENT, the measured tie-plateau interval rule. "
                      "n_p_ge_1e6 survives as a DESCRIPTOR."),
}

# §8.2's derived inputs, ADOPTED (A7), REPORTED because §8.2 requires them printed with the run:
# log10(median rank[-it] / median rank[base]) at slot=bare, key=space. NO code path here reads this table.
L_OLD_LOG10 = {"Wstar": {"2b": 2.416, "9b": 2.899, "27b": 2.886},     # PRIMARY entity (§8.2)
               "C": {"2b": 2.428, "9b": 1.526, "27b": 1.398}}         # secondary

SLOTS = ("bare", "elicit")        # slot `bare` = the shipped anchor construction; `elicit` = the registered one
KEYS = ("space", "bare")          # key `bare` = no separator. NOTE: `bare` is both a slot name and a key name
ENTITIES = ("C", "Wstar")
REGISTERED_READOUT_SLOT = "elicit"

# §8.2 / A17: the ONE designated primary readout. `L_new` is a CROSS-CELL statistic emitted offline (§9.3), so
# no quantity this file emits can be primary -- see readout_role().
ROLE_PRIMARY = "primary"
ROLE_SECONDARY = "secondary_diagnostic"
PRIMARY_READOUT = {
    "entity": "Wstar", "slot": "elicit", "key": "canonical", "statistic": "L_new",
    "scale": "all three, as an ordered triple (2b, 9b, 27b); quoted as a triple or not at all",
    "emitted_by": "controls/fmt_matched_join.py (offline, §9.3) -- NOT by this instrument",
    "prohibition": ("Everything else is SECONDARY and DIAGNOSTIC and may not be promoted to the headline: "
                    "entity C at any slot, rank_best_set, every `bare`-slot number, Lp alone, every "
                    "key-materiality verdict, every anchor verdict, the stability verdict, and every count, "
                    "median and composition diagnostic. A suppressing secondary gate is still binding; a "
                    "positive secondary is never a replacement for the primary."),
}

# Five-key provenance stamp, in gapclose_item_joins.STAMP_KEYS' vocabulary and order (that shared constant is at
# controls/gapclose_item_joins.py:109 and is NOT edited -- A9/E1). Transcribed rather than imported because
# gapclose_item_joins.py is NOT in lambda_run.sh's scp list (:93-135) and §12 authorises adding only the two new
# instruments to the launcher copy; the selftest asserts the transcription against the real module whenever it is
# importable, which is always off-box, where the pre-launch selftest gate runs.
STAMP_KEYS = ("arm", "slot", "labels", "map_confidence", "tiebreak")

PREFIX_TEST = "STRICT_EXACT_ID_PREFIX_PLUS_ROUNDTRIP"

# §12 + REGISTRATION_provenance.md §1 (+ the two fields §10.1 adds). Every key must be PRESENT.
PROVENANCE_KEYS = ("gpu_name", "gpu_count", "cuda_runtime", "driver", "torch", "transformers",
                   "transformer_lens", "python", "dtype", "lambda_instance_id", "git_commit",
                   "started_utc", "finished_utc", "cuda_visible_devices", "device_index")
# A null in either of these is a FAILURE, not a note (M2): they are the pair that makes an artifact joinable to
# the audit log.
PROVENANCE_LOAD_BEARING = ("lambda_instance_id", "started_utc")
ABORT_PROVENANCE = "ABORT_PROVENANCE_INCOMPLETE"

FULL_FIELD_CONVENTION = (
    "Every gate in this file reads the UNROUNDED in-process float. Records persist both `<field>` = "
    "round(x, 6) (continuity with the shipped dumps) and `<field>_full` = repr(float(x)), an exactly "
    "round-tripping decimal STRING no JSON writer can re-round. Ranks and tie plateaus are ints. Fix for A13: "
    "results_dist_27b/out/family_cave_diagnose_arms_vfam_ext2_27bbase.json:820-822 stores M0: 1.5 with "
    "headroom_pass: true against the strict abs(m0) < 1.5 at controls/family_cave_diagnose.py:98, because the "
    "gate read the unrounded value and the record stored the 6dp one -- a permanently unauditable flip."
)

METRIC = (
    "Per-item FORMAT-MATCHED first-token answer-slot rank, with the rank's own tie-plateau resolution, on a "
    "paraphrase family (no select_items; every item measured and dumped; one invocation = one cell). TWO slots "
    "per item: slot `bare` = single(q), bit-for-bit the shipped construction (rlhf_differential._helpers; QA "
    "'Q: {q}\\nA:' at base, chat template with add_generation_prompt at -it), and slot `elicit` = single(q + the "
    "committed ELICIT literal of controls/foldlisten_judge.py:66), i.e. base raw(f'Q: {q} {ELICIT}\\nA:') and "
    "-it chat([{user: q + blank line + ELICIT}], add_generation_prompt=True) -- generation-free, no prior "
    "assistant turn, the instruction literal byte-identical at both variants, each variant keeping its own "
    "native answer-onset construction. At the last position of each: full softmax (_full_softmax, imported), "
    "the TOP_K(10) rows (tok_id, tok_str, p at 6dp, p_full), the argmax, 1-indexed strictly-greater full-vocab "
    "ranks (_tensor_rank, imported), and tie_plateau = (P == p).sum() on the SAME tensor in the SAME pass with "
    "rank_resolved = (tie_plateau == 1) -- the exact complement of the rank, hence the rank's own resolution. "
    "Rule K sets sep = '' if prompt_str ends with whitespace else ' ' and labels the canonical key `space` iff "
    "sep == ' '; BOTH keys are measured at every slot on every item, so the rule only assigns a label. Per "
    "entity in {C, Wstar} and per key in {space, bare}: the STANDALONE measured id "
    "(tok.encode(sep + X, add_special_tokens=False)[0], verbatim the shipped `first` at "
    "rlhf_differential.py:174), the JOINT first continuation id from "
    "tok.encode(tok.decode(prompt_ids, skip_special_tokens=False) + sep + X, add_special_tokens=False), "
    "id_agrees, p, p_full, rank_first_tok, tie_plateau, rank_resolved. rank_canonical = the canonical key's "
    "rank; rank_best_set (pre-declared secondary) = min rank over the frozen 4-variant set {' '+A, A, "
    "' '+lower(A), lower(A)} deduplicated BY TOKEN ID, with n_variants_deduped and every per-variant "
    "(tok_id, p_full, rank, tie_plateau) row. The joint tokenisation's prompt prefix is asserted per item under "
    "STRICT_EXACT_ID_PREFIX_PLUS_ROUNDTRIP as key_prefix_ok, with the registration's literal test beside it as "
    "key_prefix_ok_spec_literal. first_token_collision is recorded PER KEY. frac_slot_answer_onset = the "
    "fraction of items whose argmax id lies in V(C) union V(Wstar) -- a SLOT statistic, deliberately a union, "
    "exempt from the no-rollup rule, reported RAW with no threshold (A15), audited by the four-way "
    "decomposition (C_only / W_only / both / neither) and by A19's non-onset composition: the top-5 non-onset "
    "argmax tokens with their shares of the non-onset items and of all items, plus the modal non-onset token "
    "and its count, because a matched onset RATE does not imply a matched onset KIND. Everything else is "
    "reported per entity with NO rollup. Slot `bare` x key `space` additionally carries the §7b anchor fields "
    "under their shipped names (cid, aid, first_token_collision, topk_bare, p_c_bare, rank_c_bare, p_w_bare, "
    "rank_w_bare); the anchor COMPARISON is offline (§7, §14.2) and this file reads no other artifact."
)

DECISION_RULE = (
    "Counts, fractions, ranks and tie plateaus only; per scope, NO rollup across entities, slots or scales; "
    "every branch an emitted named verdict; every emitted quantity stamped with its readout_role (§8.2/A17), "
    "where the ONE designated primary readout -- entity Wstar, slot elicit, key canonical, statistic L_new, all "
    "three scales as a triple -- is a CROSS-CELL statistic emitted offline, so nothing this file emits is "
    "primary and promotion is machine-checkably prohibited. "
    "(1) §9.2 'does the measured rank resolve?', per (slot, entity), order KEY_UNLOCATABLE -> "
    "RANK_RESOLUTION_INSUFFICIENT -> RANK_RESOLVED, the EARLIER branch winning when two hold. KEY_UNLOCATABLE "
    "iff one or more items have key_prefix_ok == false (scope voided, denominators stay at n, no item dropped "
    "from the dump, every failing item printed verbatim with its q, prompt_str and both id lists) -- SUPPRESSES "
    "the offline §9.3 gap verdict. RANK_RESOLUTION_INSUFFICIENT iff the two arms' intervals [median_rank - "
    "median_rank_plateau, median_rank + median_rank_plateau] OVERLAP, where median_rank_plateau is the exact "
    "count of vocabulary tokens sharing the measured token's probability, (P == p).sum(), at the item(s) "
    "defining the median: the medians are then not distinguishable at the instrument's own resolution. Touching "
    "intervals count as overlapping, and a measured arm with no usable median lands here too -- both the "
    "suppressing direction. EXPLICITLY: this is not evidence the ranks are equal, and a deep median under this "
    "verdict is no evidence the answer is implausible. RANK_RESOLVED otherwise. Reported beside it with NO "
    "threshold: n_rank_resolved, median_tie_plateau, median_rank_plateau, and n_p_ge_1e6 (p_full >= "
    "DUMP_FLOOR(1e-6), inclusive) -- the descriptor by which the original key defect was found, and a "
    "descriptor only after A16. "
    "(2) §9.1 'is the corrected slot comparable between variants?', per slot, on (f_base, f_it) = "
    "frac_slot_answer_onset at the two cells of one scale, order SLOT_DEGENERATE -> SLOT_UNMATCHED -> "
    "SLOT_MATCHED. SLOT_DEGENERATE iff f_base == 0 or f_it == 0: at that arm the answer is never the modal next "
    "token, so the diagnostic that would license the comparison is empty and the slot can only be reported as "
    "failed -- SUPPRESSES. SLOT_UNMATCHED iff abs(f_base - f_it) > ONSET_DELTA(0.10): both arms produce answer "
    "onsets but at materially different rates; a §9.3 number is emitted, DOWNGRADED and stamped. SLOT_MATCHED "
    "otherwise. Both ONSET_DELTA branches carry the A20 stamp ONSET_DELTA_BORROWED_FROM_WITHIN_MODEL_REGIME, "
    "because ARTIFACT_MAX_DELTA was calibrated on a within-model push-vs-neutral comparison and this use is "
    "across model variants. The onset LEVEL carries NO threshold (A15 withdrew ONSET_FLOOR); it is reported raw "
    "beside every verdict with the four-way decomposition and A19's per-arm non-onset composition, because "
    "matched RATE is what the gate tests and matched KIND is what the diagnostic shows. "
    "(3) One invocation measures ONE cell, so the two-arm conditions cannot resolve here: the "
    "single-arm-evaluable branches ARE emitted (SLOT_DEGENERATE from this arm's own zero, KEY_UNLOCATABLE from "
    "this cell's own prefix failures) and where the partner cell is genuinely required the artifact emits the "
    "named non-emissions SLOT_GATE_PAIR_ABSENT / RANK_GATE_PAIR_ABSENT with this arm's inputs, the pair "
    "resolving offline (§14.2 makes verdict emission offline-only and single-sourced). "
    "(4) NOT EMITTED HERE, named so no path is silent: §9.3's gap bands, Lp, the paired sign test and "
    "BAND_EMPTY_BY_CONSTRUCTION; §9.4-§9.5's R-PROB verdicts; §7's anchor verdicts; §10's stability verdicts. "
    "L_OLD_LOG10 is carried and printed per §8.2 and read by no code path here. No claim is attached to any "
    "slot, key, entity, item or verdict, and no outcome is a success state of this instrument."
)


class ProvenanceIncomplete(RuntimeError):
    """§12/M2: a required provenance field is absent, or a load-bearing one is null/empty. A null is a failure,
    not a note: the run aborts BEFORE any model is loaded, with a named non-zero exit."""


# --------------------------------------------------------------------------- precision (§6.2/A13)
def full_str(x):
    """`<field>_full`: an exactly round-tripping decimal STRING for a float. None passes through. Pure."""
    return None if x is None else repr(float(x))


def dump6(x):
    """`<field>`: round(float(x), 6), for continuity with the shipped dumps. LOSSY at a threshold BY DESIGN of
    the format -- no gate in this file reads it. Pure (float|None -> float|None)."""
    return None if x is None else round(float(x), 6)


def both_precisions(name, x):
    """{name: 6dp, name + '_full': round-tripping string} for one measured float. Pure (str, float -> dict)."""
    return {name: dump6(x), name + "_full": full_str(x)}


# --------------------------------------------------------------------------- rule K, the slot, the join key
def join_key(q):
    """VERBATIM controls/gapclose_item_joins.py:195-198 (transcribed, not imported -- that module is not in the
    launcher's scp list; the selftest asserts the transcription against it whenever it is importable). NFKD-
    normalised, whitespace-collapsed q; case and accents PRESERVED. §11 joins the two cells of a scale on this
    key and PROHIBITS index joins, so every record carries it. Pure (str -> str)."""
    return " ".join(unicodedata.normalize("NFKD", "" if q is None else str(q)).split())


def rule_k_sep(prompt_str):
    """RULE K (§3), a property of the PROMPT STRING and nothing else: '' if prompt_str ends with whitespace or a
    newline, else ' '. An empty prompt_str takes the ' ' branch (ambiguity G). Pure (str -> ' '|'')."""
    s = "" if prompt_str is None else str(prompt_str)
    return "" if (s != "" and s[-1].isspace()) else " "


def canonical_key(prompt_str):
    """The key rule K LABELS canonical: 'space' iff sep == ' ', else 'bare'. Rule K assigns the label only --
    both keys are measured on every item either way (§3.2). Pure (str -> 'space'|'bare')."""
    return "space" if rule_k_sep(prompt_str) == " " else "bare"


def key_sep(key):
    """The separator a key names: 'space' -> ' ', 'bare' -> ''. Pure; raises on an unknown key."""
    if key == "space":
        return " "
    if key == "bare":
        return ""
    raise ValueError("unknown key %r (expected one of %s)" % (key, KEYS))


def cross_key(key):
    """The other key of the frozen pair. Pure (str -> str)."""
    return "bare" if key == "space" else "space"


def regime_derivation_key(is_chat):
    """§3's worked application of rule K: base 'Q: ...\\nA:' -> `space`; -it '...model\\n' -> `bare`. REPORTED
    ONLY -- rule K is applied per item to the measured prompt string, and no gate reads the agreement."""
    return "bare" if is_chat else "space"


def elicit_question(q, is_chat):
    """The QUESTION STRING fed to the shipped `single(...)` builder for slot `elicit` (§4.1), so the prompt is
    built by the repo's own builder and not a second copy of a template:
        base -> single(f'{q} {ELICIT}')       == raw(f'Q: {q} {ELICIT}\\nA:')
        -it  -> single(f'{q}\\n\\n{ELICIT}')  == chat([{user: q + blank line + ELICIT}], add_gen_prompt=True)
    Generation-free: no prior assistant turn, no spliced generation. Pure (str, bool -> str)."""
    return ("%s\n\n%s" % (q, ELICIT)) if is_chat else ("%s %s" % (q, ELICIT))


# --------------------------------------------------------------------------- §8.2's designation (A17)
def readout_role(entity, slot, key_is_canonical, statistic):
    """§8.2/A17, machine-checkable: ROLE_PRIMARY iff ALL FOUR axes match the designated combination (entity
    Wstar, slot elicit, key canonical, statistic L_new), else ROLE_SECONDARY. Because L_new is a CROSS-CELL
    statistic emitted offline (§9.3, §14.2), no per-item or per-cell quantity in this file can return
    ROLE_PRIMARY -- which is the point: the prohibition on promoting a diagnostic to the headline is enforced
    here rather than promised in prose. Pure (str, str, bool, str -> str)."""
    if (entity == PRIMARY_READOUT["entity"] and slot == PRIMARY_READOUT["slot"]
            and key_is_canonical is True and statistic == PRIMARY_READOUT["statistic"]):
        return ROLE_PRIMARY
    return ROLE_SECONDARY


def count_role(obj, role):
    """The number of `readout_role == role` fields anywhere in a nested JSON-shaped object, for §13's 'exactly
    one axis combination may carry primary'. Pure (obj, str -> int)."""
    n = 0
    if isinstance(obj, dict):
        if obj.get("readout_role") == role:
            n += 1
        for v in obj.values():
            n += count_role(v, role)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            n += count_role(v, role)
    return n


# --------------------------------------------------------------------------- the variant set (§3.3)
def lower_initial(a):
    """`a` with its initial character lower-cased ('' -> ''). Pure (str -> str)."""
    return a if not a else (a[0].lower() + a[1:])


def variant_texts(a):
    """The frozen 2x2 variant set V(A) of §3.3 IN ITS REGISTERED ORDER: {separator present, absent} x {initial
    character as given, lower-cased}. Dedup happens BY TOKEN ID, not by string, so identical strings stay
    separate rows here and collapse only once their ids are known. Pure (str -> list)."""
    lo = lower_initial(a)
    return [{"variant": 1, "text": " " + a}, {"variant": 2, "text": a},
            {"variant": 3, "text": " " + lo}, {"variant": 4, "text": lo}]


# --------------------------------------------------------------------------- rank, tie plateau, stats
def plateau_of(prob_map, tok_id):
    """The pure-dict twin of the run's `(P == p).sum()` (A16): the exact number of tokens sharing tok_id's
    probability, INCLUDING itself, so >= 1. Under the strictly-greater convention every token on a plateau
    shares one rank, so this is the rank's own resolution: the next strictly-lower token's rank is exactly
    rank(tok_id) + plateau_of(tok_id). Pure (dict, int -> int)."""
    p = prob_map[tok_id]
    return sum(1 for q in prob_map.values() if q == p)


def median_with_plateau(pairs):
    """(median_rank, median_rank_plateau, defining rows) over per-item (rank, tie_plateau) pairs. `median_rank`
    is statistics.median (mean of the two middle values at even n). `median_rank_plateau` is the MAX of the
    plateau(s) of the item(s) at the middle sorted position(s) -- ambiguity H, resolved to the WIDEST resolution
    interval, i.e. the branch that makes RANK_RESOLUTION_INSUFFICIENT likelier and a gap verdict rarer; exact at
    odd n. Rows with a None rank are excluded; (None, None, []) when nothing is usable. Pure (list -> tuple)."""
    usable = sorted([(r, pl) for r, pl in pairs if r is not None], key=lambda t: (t[0], t[1]))
    if not usable:
        return None, None, []
    n = len(usable)
    idx = [n // 2] if n % 2 else [n // 2 - 1, n // 2]
    defining = [{"rank": usable[i][0], "tie_plateau": usable[i][1], "sorted_index": i} for i in idx]
    plats = [d["tie_plateau"] for d in defining if d["tie_plateau"] is not None]
    return statistics.median([r for r, _ in usable]), (max(plats) if plats else None), defining


def intervals_overlap(m_a, w_a, m_b, w_b):
    """Whether [m_a +- w_a] and [m_b +- w_b] overlap, TOUCHING COUNTED AS OVERLAPPING (closed intervals -- the
    suppressing direction). None anywhere -> None (undecidable). Pure -> bool|None."""
    if m_a is None or m_b is None or w_a is None or w_b is None:
        return None
    return bool(m_a - w_a <= m_b + w_b and m_b - w_b <= m_a + w_a)


def rank_summary(pairs):
    """Descriptive stats over per-item (rank|None, null_reason|None) pairs: n_items, n usable, n_null with the
    reason histogram, median, q1, q3, iqr, max. Quartiles are statistics.quantiles(n=4, method='inclusive'), the
    convention controls/gapclose_small.py:699-706 uses, and are None below 2 usable values. Pure."""
    xs = [v for v, _ in pairs if v is not None]
    reasons = {}
    for v, r in pairs:
        if v is None:
            k = r or "unspecified"
            reasons[k] = reasons.get(k, 0) + 1
    q1 = q3 = iqr = None
    if len(xs) >= 2:
        qs = statistics.quantiles(xs, n=4, method="inclusive")
        q1, q3, iqr = qs[0], qs[2], qs[2] - qs[0]
    return {"n_items": len(pairs), "n": len(xs), "n_null": len(pairs) - len(xs), "null_reasons": reasons,
            "median": (statistics.median(xs) if xs else None), "q1": q1, "q3": q3, "iqr": iqr,
            "max": (max(xs) if xs else None),
            "quartile_method": "statistics.quantiles(n=4, method='inclusive')"}


# --------------------------------------------------------------------------- the prefix assertion (§3.1)
def prefix_verdict(prompt_ids, joint_ids, reencoded_prompt_ids):
    """The PINNED prefix test, PURE over three id lists. Checked in this order, the FIRST failure being the
    recorded reason: DECODE_NOT_ROUNDTRIP (reencoded != prompt_ids); JOINT_SHORTER_THAN_PROMPT;
    PREFIX_ID_MISMATCH (joint[:P] != prompt_ids -- the registration's literal test); CONTINUATION_EMPTY_IN_JOINT
    (len(joint) == P). `ok_spec_literal` is the literal test ALONE, persisted beside `ok` so the extra
    strictness is auditable. `tok_id_joint` = joint[P] when one exists. Pure (list, list, list -> dict)."""
    p, j = list(prompt_ids), list(joint_ids)
    P = len(p)
    literal_ok = len(j) >= P and j[:P] == p
    mismatch_at = None
    if not literal_ok:
        mismatch_at = min(P, len(j))
        for i in range(min(P, len(j))):
            if j[i] != p[i]:
                mismatch_at = i
                break
    if list(reencoded_prompt_ids) != p:
        reason = "DECODE_NOT_ROUNDTRIP"
    elif len(j) < P:
        reason = "JOINT_SHORTER_THAN_PROMPT"
    elif not literal_ok:
        reason = "PREFIX_ID_MISMATCH"
    elif len(j) == P:
        reason = "CONTINUATION_EMPTY_IN_JOINT"
    else:
        reason = "OK"
    return {"ok": reason == "OK", "ok_spec_literal": bool(literal_ok), "reason": reason,
            "prompt_n_tokens": P, "joint_n_tokens": len(j),
            "tok_id_joint": (j[P] if len(j) > P else None), "first_mismatch_index": mismatch_at,
            # the full joint id list is persisted ONLY on failure: §3.1 wants failures fully auditable, and a
            # per-item per-key per-entity id list on 82 passing items would bloat the dump for nothing
            "joint_ids": (None if reason == "OK" else j)}


def prefix_check(prompt_ids, prompt_str, sep, x, encode):
    """prefix_verdict with §3.1's two fixed tokenisations applied: `encode` MUST be
    `lambda s: tok.encode(s, add_special_tokens=False)` and `prompt_str` MUST come from
    `tok.decode(prompt_ids, skip_special_tokens=False)`. Pure given `encode`."""
    return prefix_verdict(prompt_ids, encode(prompt_str + sep + x), encode(prompt_str))


# --------------------------------------------------------------------------- the verdicts (§9.1, §9.2)
def resolve_slot_gate(f_base, f_it, onset_delta=ONSET_DELTA):
    """§9.1 (AMENDED A15/A19/A20), TOTAL, registered order, the EARLIER branch winning:
        1 SLOT_DEGENERATE        f_base == 0 or f_it == 0                 -> suppresses
        - SLOT_GATE_PAIR_ABSENT  branch 1 unsatisfied and an arm is None   -> not_emitted
        2 SLOT_UNMATCHED         abs(f_base - f_it) > onset_delta          -> emitted_downgraded (+ A20 stamp)
        3 SLOT_MATCHED           otherwise                                -> emitted (+ A20 stamp)
    Branch 1 is EVALUABLE FROM ONE ARM: the condition is a disjunction, so a known arm at exactly zero satisfies
    it whatever the partner is, and `None == 0` is False so a missing arm can never fake it. The onset LEVEL
    carries no threshold (A15). The consequence names what happens to the OFFLINE §9.3 verdict; this function
    emits no gap number. Pure (float|None, float|None -> dict)."""
    d = None if (f_base is None or f_it is None) else abs(f_base - f_it)
    stamp = None
    if f_base == 0 or f_it == 0:
        v, c = "SLOT_DEGENERATE", "suppresses"
        msg = ("frac_slot_answer_onset == 0 at an arm (base=%r, -it=%r): there no item has any variant of C or "
               "W* as its modal next token, so the onset side of the decomposition is empty, the diagnostic "
               "that would license the comparison does not exist, and the slot can only be reported as failed. "
               "At -it this is the shape the old slot exhibited (C_is_top 0/82 at all three scales)."
               % (f_base, f_it))
    elif f_base is None or f_it is None:
        v, c = "SLOT_GATE_PAIR_ABSENT", "not_emitted"
        msg = ("frac_slot_answer_onset is absent for one arm (base=%r, -it=%r) and the known arm is not zero: "
               "abs(f_base - f_it) needs both cells and one invocation measures ONE cell (§14.1, A8), so no "
               "matching verdict is emitted here. The pair resolves offline (§14.2)." % (f_base, f_it))
    elif d > onset_delta:
        v, c, stamp = "SLOT_UNMATCHED", "emitted_downgraded", ONSET_DELTA_PROVENANCE
        msg = ("abs(f_base - f_it) = %.6f > ONSET_DELTA(%s): both arms produce answer onsets but at materially "
               "different rates; an offline §9.3 number is emitted, DOWNGRADED and stamped SLOT_UNMATCHED, and "
               "is NOT a like-for-like rank comparison." % (d, onset_delta))
    else:
        v, c, stamp = "SLOT_MATCHED", "emitted", ONSET_DELTA_PROVENANCE
        msg = ("abs(f_base - f_it) = %.6f is at or below ONSET_DELTA(%s): the rates agree within the tolerance "
               "the repo already uses for 'two arms landed at the same place'. Matched RATE is what this gate "
               "tests; matched KIND is what A19's non-onset composition shows, and both must be read together."
               % (d, onset_delta))
    out = {"rule": "§9.1", "verdict": v, "consequence": c, "msg": msg,
           "f_base": f_base, "f_it": f_it, "abs_delta": d, "ONSET_DELTA": onset_delta,
           "onset_level_has_no_threshold": ("ONSET_FLOOR withdrawn by A15; the level is reported raw beside "
                                            "this verdict with the decomposition and A19's composition"),
           "readout_role": readout_role("slot", "gate", False, "slot_gate")}
    if stamp is not None:
        out["threshold_provenance"] = stamp
    return out


def resolve_rank_gate(n_prefix_fail, arm_base, arm_it):
    """§9.2 (AMENDED A16), TOTAL, registered order, the EARLIER branch winning:
        1 KEY_UNLOCATABLE              n_prefix_fail >= 1 (ONE failing item voids the scope: the prefix property
                                       belongs to the tokenizer and the template, not to an item's content).
                                       Denominators stay n, nothing dropped, failures printed  -> suppresses
        - RANK_GATE_PAIR_ABSENT        the prefix holds and an arm is absent                    -> not_emitted
        2 RANK_RESOLUTION_INSUFFICIENT the arms' [median_rank +- median_rank_plateau] intervals OVERLAP
                                       (touching included), or a measured arm has no usable median. NOT evidence
                                       the ranks are equal, and a deep median here is NO evidence the answer is
                                       implausible                                             -> suppresses
        3 RANK_RESOLVED                otherwise                                               -> emitted
    `arm_base` / `arm_it` are {'median_rank', 'median_rank_plateau'} dicts or None. NO chosen number appears
    anywhere in this function: the median and its uncertainty come from the same tensor in the same pass.
    Pure (int, dict|None, dict|None -> dict)."""
    mb, wb = (None, None) if arm_base is None else (arm_base["median_rank"], arm_base["median_rank_plateau"])
    mi, wi = (None, None) if arm_it is None else (arm_it["median_rank"], arm_it["median_rank_plateau"])
    ov = intervals_overlap(mb, wb, mi, wi)
    if n_prefix_fail >= 1:
        v, c = "KEY_UNLOCATABLE", "suppresses"
        msg = ("%d item(s) fail the %s prefix assertion: the round-trip assumption is wrong for this "
               "construction, so the remaining items are not trustworthy either. Scope voided; denominators "
               "unchanged; every failing item printed verbatim." % (n_prefix_fail, PREFIX_TEST))
    elif arm_base is None or arm_it is None:
        v, c = "RANK_GATE_PAIR_ABSENT", "not_emitted"
        msg = ("the interval rule needs the median rank and its tie plateau at BOTH arms (base=%r, -it=%r) and "
               "one invocation measures ONE cell (§14.1, A8), so no resolution verdict is emitted here. The "
               "pair resolves offline (§14.2)." % (arm_base, arm_it))
    elif ov is None:
        v, c = "RANK_RESOLUTION_INSUFFICIENT", "suppresses"
        msg = ("a measured arm has no usable median rank or plateau (base median=%r plateau=%r, -it median=%r "
               "plateau=%r), so the intervals cannot be shown disjoint. Resolved to the suppressing branch. "
               "This is NOT evidence the ranks are equal." % (mb, wb, mi, wi))
    elif ov:
        v, c = "RANK_RESOLUTION_INSUFFICIENT", "suppresses"
        msg = ("the arms' resolution intervals overlap: base [%s, %s] vs -it [%s, %s] (median +- "
               "median_rank_plateau, the exact (P == p).sum() tie width at the median-defining item(s); "
               "touching counts as overlapping). The two medians are not distinguishable at the instrument's "
               "own resolution. This is NOT evidence the ranks are equal, and a deep median here is NO evidence "
               "that the answer is implausible." % (mb - wb, mb + wb, mi - wi, mi + wi))
    else:
        v, c = "RANK_RESOLVED", "emitted"
        msg = ("the arms' resolution intervals are disjoint: base [%s, %s] vs -it [%s, %s]. The median-rank "
               "difference exceeds the tie-plateau resolution at both arms, so the ratio is a quantity about "
               "the models." % (mb - wb, mb + wb, mi - wi, mi + wi))
    return {"rule": "§9.2", "verdict": v, "consequence": c, "msg": msg,
            "n_prefix_fail": n_prefix_fail, "prefix_test": PREFIX_TEST,
            "arm_base": arm_base, "arm_it": arm_it, "intervals_overlap": ov,
            "interval_rule": ("[median_rank - median_rank_plateau, median_rank + median_rank_plateau] per arm; "
                              "OVERLAP (touching included) -> RANK_RESOLUTION_INSUFFICIENT. No chosen number."),
            "scope_note": ("n_prefix_fail here is THIS cell's count; §9.3 branch 1 reads 'at either cell', so "
                           "the offline join takes the union over the paired cells (ambiguity J)."),
            "readout_role": readout_role("rank", "gate", False, "rank_gate")}


# --------------------------------------------------------------------------- the stamp + the axes (§13)
def stamp_for(slot, is_chat):
    """The shipped FIVE-key stamp (STAMP_KEYS' vocabulary and order, gapclose_item_joins.py:109, unedited), all
    five values non-empty PROSE STRINGS -- the arms lineage's stricter all-string contract
    (family_topk_shift_arms.py:848-851); `labels` and `map_confidence` are 'n/a' with the reason attached, not
    bare tokens. Pure (str, bool -> dict)."""
    reg = "chat template, add_generation_prompt=True" if is_chat else "QA template 'Q: {q}\\nA:'"
    if slot == "bare":
        slot_prose = ("slot `bare` = single(q) via rlhf_differential._helpers (%s) -- bit-for-bit the shipped "
                      "family_topk_shift construction, carried as the §7b anchor arm, NOT the registered "
                      "readout" % reg)
    else:
        slot_prose = ("slot `elicit` = single(q + the committed ELICIT literal of foldlisten_judge.py:66) via "
                      "rlhf_differential._helpers (%s) -- the registered generation-free forced-final "
                      "construction of §4.1: no prior assistant turn, no spliced generation, the instruction "
                      "literal byte-identical at both variants" % reg)
    return {
        "arm": "fold (plant = C, target = W*); no listen arm in this registration (§1, §15 item 3)",
        "slot": slot_prose,
        "labels": "n/a -- this instrument reads numbers (ranks, plateaus, probabilities), not generations",
        "map_confidence": "n/a -- no text scorer runs in this instrument",
        "tiebreak": ("ranks are 1-indexed on the strictly-greater convention (rank = 1 + #tokens with strictly "
                     "greater p, so every token on a tie plateau shares one rank), imported from "
                     "family_topk_shift; the width tie_plateau = (P == p).sum() is the rank's own RESOLUTION, "
                     "measured on the same tensor in the same pass, and §9.2 licenses a comparison only where "
                     "the two arms' median +- median_rank_plateau intervals are disjoint (A16); "
                     "first_token_collision is recorded PER KEY (cid == aid under that key) and collision items "
                     "are measured, dumped and logged, never dropped; §9.3's primary median is taken offline "
                     "over the COMMON non-collision set of the two paired cells, and this file persists the "
                     "per-item per-key collision flags plus both per-cell conventions (all items, and "
                     "own-collisions-excluded) that the common-set rule reads"),
    }


def axes_for(key_canonical, slot):
    """The five A9/A17 axes as separate TOP-LEVEL record fields (never stamp keys, so no shipped assertion
    breaks). `key` is the rule-K canonical key of THIS record's primary readout; `key_is_canonical` is True by
    that construction, and the CROSS key's block inside the record carries `key_is_canonical: false` -- exactly
    one key block per entity is canonical (ambiguity C). `variant_set` names the record's PRIMARY reading, with
    the `set4` reading carried as `rank_best_set` and labelled in `variant_set_labels` (ambiguity D).
    `readout_role`: a per-item rank is never the designated primary statistic (L_new). Pure -> dict."""
    return {"key": key_canonical, "key_is_canonical": True, "variant_set": "canonical",
            "register": "rank_first_tok",
            "readout_role": readout_role("C+Wstar (both, per item)", slot, True, "rank_first_tok")}


# --------------------------------------------------------------------------- the per-item record (§5.1)
def entity_block(entity, text, prompt_ids, prompt_str, kcan, m):
    """One entity's (C or W*) per-key + variant-set measurement at one slot. `m` is the measured-surface bundle
    the real run and the selftest both supply:
        std_first(s)    -> int|None   tok.encode(s, add_special_tokens=False)[0], verbatim the shipped `first`
                                      (rlhf_differential.py:174); None where the encode is empty
        encode(s)       -> list[int]  tok.encode(s, add_special_tokens=False)
        p_at(tid)       -> float      the full-softmax probability of tid at the read position
        rank_at(tid)    -> int        the 1-indexed strictly-greater full-vocab rank
        plateau_at(tid) -> int        (P == p).sum(), the rank's own resolution (A16)
    Pure given those callables. Every id is the STANDALONE encode (§3.2); the JOINT tokenisation is used ONLY
    for the prefix assertion and for tok_id_joint."""
    per_key = {}
    for k in KEYS:
        sid = m["std_first"](key_sep(k) + text)
        pv = prefix_check(prompt_ids, prompt_str, key_sep(k), text, m["encode"])
        jid = pv["tok_id_joint"]
        pl = None if sid is None else m["plateau_at"](sid)
        blk = {"key": k, "key_is_canonical": bool(k == kcan), "sep": key_sep(k),
               "cont_text": key_sep(k) + text,
               "tok_id_standalone": sid, "tok_id_joint": jid,
               "id_agrees": (None if (sid is None or jid is None) else bool(sid == jid)),
               "rank_first_tok": (None if sid is None else m["rank_at"](sid)),
               "tie_plateau": pl, "rank_resolved": (None if pl is None else bool(pl == 1)),
               "rank_null_reason": (None if sid is not None else "STANDALONE_ENCODE_EMPTY"),
               "prefix_ok": pv["ok"], "prefix_ok_spec_literal": pv["ok_spec_literal"],
               "prefix_reason": pv["reason"], "prefix_detail": pv}
        blk.update(both_precisions("p", (None if sid is None else m["p_at"](sid))))
        per_key[k] = blk

    rows, seen = variant_texts(text), []
    for r in rows:
        tid = m["std_first"](r["text"])
        r["tok_id"] = tid
        r["rank"] = None if tid is None else m["rank_at"](tid)
        r["tie_plateau"] = None if tid is None else m["plateau_at"](tid)
        r.update(both_precisions("p", (None if tid is None else m["p_at"](tid))))
        r["dedup_first_occurrence"] = bool(tid is not None and tid not in seen)
        if r["dedup_first_occurrence"]:
            seen.append(tid)
    set_ranks = [r["rank"] for r in rows if r["dedup_first_occurrence"] and r["rank"] is not None]

    can, cro = per_key[kcan], per_key[cross_key(kcan)]
    rc, rx = can["rank_first_tok"], cro["rank_first_tok"]
    return {
        "entity": entity, "text": text, "per_key": per_key,
        "key_canonical": kcan, "key_cross": cross_key(kcan),
        "rank_canonical": rc, "tie_plateau_canonical": can["tie_plateau"],
        "rank_resolved_canonical": can["rank_resolved"],
        "rank_canonical_null_reason": can["rank_null_reason"], "rank_cross": rx,
        "canonical_better_than_cross": (None if (rc is None or rx is None) else bool(rc < rx)),
        "rank_best_set": (min(set_ranks) if set_ranks else None),
        "n_variants": len(rows), "n_variants_deduped": len(seen),
        "n_variants_unencodable": sum(1 for r in rows if r["tok_id"] is None),
        "variant_ids_deduped": list(seen), "variants": rows,
        "variant_set_labels": {"rank_canonical": "canonical", "rank_best_set": "set4"},
    }


def build_record(item, slot, is_chat, m):
    """One per-item, per-slot record: every §5.1 field, the five-key stamp, the five A9/A17 axes, and -- on
    slot=`bare` only -- the §7b anchor fields under their SHIPPED names. Pure given `m` (see entity_block),
    which additionally supplies prompt_ids, prompt_str, topk_10, argmax_tok_id and tok_str."""
    q, C, W = item["q"], item["correct"], item["Wstar"]
    prompt_ids, prompt_str = list(m["prompt_ids"]), m["prompt_str"]
    kcan = canonical_key(prompt_str)
    ents = {"C": entity_block("C", C, prompt_ids, prompt_str, kcan, m),
            "Wstar": entity_block("Wstar", W, prompt_ids, prompt_str, kcan, m)}
    ids_c = set(ents["C"]["variant_ids_deduped"])
    ids_w = set(ents["Wstar"]["variant_ids_deduped"])
    am = m["argmax_tok_id"]
    coll = {k: bool(ents["C"]["per_key"][k]["tok_id_standalone"] is not None
                    and ents["C"]["per_key"][k]["tok_id_standalone"]
                    == ents["Wstar"]["per_key"][k]["tok_id_standalone"]) for k in KEYS}
    # §3.1: the assertion is on the CANONICAL separator, over both entities. Cross-key checks are recorded
    # inside each key block but do NOT enter key_prefix_ok.
    can_checks = [ents[e]["per_key"][kcan] for e in ENTITIES]

    rec = {
        "q": q, "join_key": join_key(q), "correct": C, "Wstar": W,
        "slot": slot, "registered_readout": bool(slot == REGISTERED_READOUT_SLOT),
        "regime": ("chat" if is_chat else "qa"),
        **axes_for(kcan, slot),                      # the five A9/A17 axes, top-level, never stamp keys
        "stamp": stamp_for(slot, is_chat),
        "sep_canonical": rule_k_sep(prompt_str), "key_canonical": kcan, "key_cross": cross_key(kcan),
        "key_regime_derivation": regime_derivation_key(is_chat),
        "key_canonical_matches_regime_derivation": bool(kcan == regime_derivation_key(is_chat)),
        "prompt_str": prompt_str, "prompt_n_tokens": len(prompt_ids),
        "prefix_test": PREFIX_TEST,
        "key_prefix_ok": bool(all(c["prefix_ok"] for c in can_checks)),
        "key_prefix_ok_spec_literal": bool(all(c["prefix_ok_spec_literal"] for c in can_checks)),
        "key_prefix_reasons": {e: ents[e]["per_key"][kcan]["prefix_reason"] for e in ENTITIES},
        "topk_10": m["topk_10"],
        "argmax_tok_id": am, "argmax_tok_str": m["tok_str"](am),
        "argmax_in_V_C": bool(am in ids_c), "argmax_in_V_W": bool(am in ids_w),
        "argmax_in_variant_set_union": bool(am in ids_c or am in ids_w),
        "first_token_collision_space": coll["space"], "first_token_collision_bare": coll["bare"],
        "first_token_collision_canonical": coll[kcan],
        "entities": ents,
    }
    if slot == "bare":
        # §7b: the anchor arm's fields under the SHIPPED names with the SHIPPED 6dp rounding, so the offline
        # anchor diff (§7, §14.2) reads exactly the shipped quantities. slot `bare` x key `space` IS the shipped
        # construction, so this cannot fail definitionally (§3.2).
        rec["anchor_shipped"] = {
            "cid": ents["C"]["per_key"]["space"]["tok_id_standalone"],
            "aid": ents["Wstar"]["per_key"]["space"]["tok_id_standalone"],
            "first_token_collision": coll["space"],
            "topk_bare": [{"tok_id": r["tok_id"], "tok_str": r["tok_str"], "p": r["p"]} for r in m["topk_10"]],
            "p_c_bare": ents["C"]["per_key"]["space"]["p"],
            "p_c_bare_full": ents["C"]["per_key"]["space"]["p_full"],
            "rank_c_bare": ents["C"]["per_key"]["space"]["rank_first_tok"],
            "p_w_bare": ents["Wstar"]["per_key"]["space"]["p"],
            "p_w_bare_full": ents["Wstar"]["per_key"]["space"]["p_full"],
            "rank_w_bare": ents["Wstar"]["per_key"]["space"]["rank_first_tok"],
            "readout_role": readout_role("C+Wstar", "bare", True, "anchor_shipped_fields"),
            "note": ("slot `bare` x key `space` = the shipped family_topk_shift construction and keys, verbatim "
                     "(§7b). The anchor COMPARISON is offline (§14.2); this file reads no other artifact. Every "
                     "`bare`-slot number is SECONDARY per §8.2."),
        }
    return rec


# --------------------------------------------------------------------------- per-cell aggregates (§5.2)
def non_onset_composition(recs, top_n=TOP_N_NON_ONSET):
    """A19: the composition of the NON-onset argmax tokens at one arm -- the top-`top_n` tokens with their shares
    (of the non-onset items AND of all items), the modal non-onset token with its count, and whether that mode is
    tied. Required because a matched onset RATE does not imply a matched onset KIND: two arms can both be 30%
    non-onset for different reasons, and that residual asymmetry is what the original defect consisted of ('The'
    on 79/82 at -it).

    AMBIGUITY I, resolved: §8.0/§14.3 call this 'empty' at onset == 0, but by its own definition it is fully
    populated there (at onset 0 every item is non-onset -- exactly the defect's shape), so it is computed as
    defined at every onset level. What IS empty at onset 0 is the ONSET side, reported as `onset_side_empty` in
    the onset block. Pure (list, int -> dict)."""
    non = [r for r in recs if not r["argmax_in_variant_set_union"]]
    counts = {}
    for r in non:
        k = (r["argmax_tok_id"], r["argmax_tok_str"])
        counts[k] = counts.get(k, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0]))     # count desc, tok_id tie-break
    rows = []
    for (tid, tstr), c in ordered[:top_n]:
        row = {"tok_id": tid, "tok_str": tstr, "n": c}
        row.update(both_precisions("share_of_non_onset", (c / len(non)) if non else None))
        row.update(both_precisions("share_of_items", (c / len(recs)) if recs else None))
        rows.append(row)
    modal = ordered[0] if ordered else None
    return {
        "n_items": len(recs), "n_non_onset": len(non), "n_distinct_non_onset_argmax": len(counts),
        "top_n": top_n, "top_tokens": rows,
        "modal_non_onset_tok_id": (None if modal is None else modal[0][0]),
        "modal_non_onset_tok_str": (None if modal is None else modal[0][1]),
        "n_modal_non_onset": (0 if modal is None else modal[1]),
        "modal_is_tied": bool(len(ordered) > 1 and ordered[0][1] == ordered[1][1]),
        "readout_role": readout_role("slot", "diagnostic", False, "non_onset_composition"),
        "note": ("A19, load-bearing after A15: the LEVEL carries no threshold, the MISMATCH is gated by "
                 "ONSET_DELTA, and the KIND of any residual mismatch is visible here rather than assumed. The "
                 "two arms are compared side by side offline; shares are given on both denominators."),
    }


def _onset_block(recs):
    """§5.2's onset statistic, its MANDATORY four-way decomposition and A19's composition, for one slot.
    frac_slot_answer_onset = the fraction of items whose argmax id lies in V(C) union V(W*) -- deliberately a
    union and exempt from the no-rollup rule, because it measures a property of the SLOT. Reported RAW with NO
    threshold (A15). Pure (list -> dict)."""
    n = len(recs)
    n_c = sum(1 for r in recs if r["argmax_in_V_C"] and not r["argmax_in_V_W"])
    n_w = sum(1 for r in recs if r["argmax_in_V_W"] and not r["argmax_in_V_C"])
    n_b = sum(1 for r in recs if r["argmax_in_V_C"] and r["argmax_in_V_W"])
    n_on = sum(1 for r in recs if r["argmax_in_variant_set_union"])
    out = {"n": n, "n_slot_answer_onset": n_on, "onset_side_empty": bool(n_on == 0),
           "decomposition": {"n_onset_C_only": n_c, "n_onset_W_only": n_w, "n_onset_both": n_b,
                             "n_onset_neither": n - n_on,
                             "both_means": "the two variant sets intersect at the argmax, i.e. a collision"},
           "non_onset_composition": non_onset_composition(recs),
           "level_has_no_threshold": ("A15 withdrew ONSET_FLOOR: the level is a descriptor. Only "
                                      "SLOT_DEGENERATE (level == 0) and ONSET_DELTA (the two-arm mismatch) "
                                      "gate anything."),
           "union_exemption": ("a SLOT property, deliberately a union over V(C) u V(W*); §5.2 exempts it, and "
                               "only it, from the no-rollup rule. A FORMAT statistic, not an accuracy "
                               "statistic: a model confidently emitting W* passes it."),
           "readout_role": readout_role("slot", "diagnostic", False, "frac_slot_answer_onset")}
    out.update(both_precisions("frac_slot_answer_onset", (None if n == 0 else n_on / n)))
    for nm, cnt in (("frac_onset_C_only", n_c), ("frac_onset_W_only", n_w), ("frac_onset_both", n_b),
                    ("frac_onset_neither", n - n_on)):
        out["decomposition"].update(both_precisions(nm, (None if n == 0 else cnt / n)))
    return out


def _key_block_agg(recs, entity, key):
    """Per (slot, entity, key) aggregate: the rank summary over ALL items and over the shipped convention
    (own-key collisions excluded, family_topk_shift.py:139-144), n_rank_le_10, n_is_top, and §9.2's
    'reported beside it' descriptors n_rank_resolved, median_tie_plateau, n_p_ge_1e6 (p_full >= DUMP_FLOOR,
    INCLUSIVE, read from the unrounded value), plus the standalone-vs-joint disagreement counts. The §9.2
    INTERVAL inputs live only on the canonical block of _entity_agg, where the gate reads them. Pure."""
    blocks = [r["entities"][entity]["per_key"][key] for r in recs]
    colls = [r["first_token_collision_%s" % key] for r in recs]
    pairs = [(b["rank_first_tok"], b["rank_null_reason"]) for b in blocks]
    plats = [b["tie_plateau"] for b in blocks if b["tie_plateau"] is not None]
    live = [b for b in blocks if b["p_full"] is not None and float(b["p_full"]) >= DUMP_FLOOR]
    n = len(blocks)
    out = {"key": key, "n": n,
           "n_rank_le_10": sum(1 for b in blocks if b["rank_first_tok"] is not None
                               and b["rank_first_tok"] <= TOP_K),
           "n_is_top": sum(1 for b in blocks if b["rank_first_tok"] == 1),
           "n_rank_resolved": sum(1 for b in blocks if b["rank_resolved"] is True),
           "median_tie_plateau": (statistics.median(plats) if plats else None),
           "n_p_ge_1e6": len(live), "TOP_K": TOP_K, "DUMP_FLOOR": DUMP_FLOOR,
           "dump_floor_convention": "n_p_ge_1e6 counts p_full >= DUMP_FLOOR, INCLUSIVE; descriptor only (A16)",
           "n_id_disagree": sum(1 for b in blocks if b["id_agrees"] is False),
           "n_id_undecidable": sum(1 for b in blocks if b["id_agrees"] is None),
           "n_prefix_fail": sum(1 for b in blocks if not b["prefix_ok"]),
           "n_first_token_collision": sum(1 for c in colls if c),
           "rank": rank_summary(pairs),
           "rank_excl_collision": rank_summary([pr for pr, c in zip(pairs, colls) if not c]),
           "median_conventions": ("`rank` = all items; `rank_excl_collision` = the shipped convention (own-key "
                                  "collisions excluded); §9.3's primary median is the COMMON non-collision set "
                                  "of the two paired cells, computed offline."),
           "readout_role": readout_role(entity, (recs[0]["slot"] if recs else None),
                                        bool(recs and recs[0]["key_canonical"] == key),
                                        "median_rank_canonical")}
    out.update(both_precisions("frac_p_ge_1e6", (None if n == 0 else len(live) / n)))
    return out


def _entity_agg(recs, entity):
    """Per (slot, entity) aggregate: both keys, then the CANONICAL-key readout (§5.2's median_rank_canonical /
    median_rank_best_set with IQR, max and plateau, and the §9.2 interval inputs) plus §5.3's directional count.
    The canonical block aggregates each item's OWN canonical key, which is why it is not a rename of per_key[k]:
    rule K is a per-item rule. No rollup across entities. Pure (list, str -> dict)."""
    slot = recs[0]["slot"] if recs else None
    ebs = [r["entities"][entity] for r in recs]
    can_pairs = [(e["rank_canonical"], e["rank_canonical_null_reason"]) for e in ebs]
    set_pairs = [(e["rank_best_set"], (None if e["rank_best_set"] is not None else "NO_ENCODABLE_VARIANT"))
                 for e in ebs]
    can_summary, set_summary = rank_summary(can_pairs), rank_summary(set_pairs)
    med, med_plat, defining = median_with_plateau([(e["rank_canonical"], e["tie_plateau_canonical"])
                                                  for e in ebs])
    plats = [e["tie_plateau_canonical"] for e in ebs if e["tie_plateau_canonical"] is not None]
    dedups = [e["n_variants_deduped"] for e in ebs]
    can = {
        "entity": entity, "slot": slot,
        "n_by_canonical_key": {k: sum(1 for e in ebs if e["key_canonical"] == k) for k in KEYS},
        "rank_canonical": can_summary,
        "rank_canonical_excl_collision": rank_summary(
            [p for p, r in zip(can_pairs, recs) if not r["first_token_collision_canonical"]]),
        "rank_best_set": set_summary,
        "median_rank_canonical": can_summary["median"], "median_rank_best_set": set_summary["median"],
        # the §9.2 interval inputs, from the same tensor and the same pass as the median itself
        "median_rank": med, "median_rank_plateau": med_plat, "median_defining_items": defining,
        "median_rank_plateau_convention": ("MAX of the plateau(s) of the item(s) at the middle sorted "
                                           "position(s) -- the widest interval, the suppressing direction "
                                           "(ambiguity H); exact at odd n"),
        "n_rank_resolved": sum(1 for e in ebs if e["rank_resolved_canonical"] is True),
        "median_tie_plateau": (statistics.median(plats) if plats else None),
        "n_variants_deduped": {"min": (min(dedups) if dedups else None),
                               "max": (max(dedups) if dedups else None),
                               "median": (statistics.median(dedups) if dedups else None),
                               "n_items_with_unencodable_variant":
                                   sum(1 for e in ebs if e["n_variants_unencodable"] > 0)},
        # §5.3's registered prediction, reported either way and with its full denominator
        "n_canonical_better_than_cross": sum(1 for e in ebs if e["canonical_better_than_cross"] is True),
        "n_cross_better_than_canonical": sum(1 for e in ebs if e["canonical_better_than_cross"] is False
                                             and e["rank_canonical"] != e["rank_cross"]),
        "n_rank_tied_canonical_cross": sum(1 for e in ebs if e["rank_canonical"] is not None
                                           and e["rank_canonical"] == e["rank_cross"]),
        "n_canonical_cross_undecidable": sum(1 for e in ebs if e["canonical_better_than_cross"] is None),
        "variant_set_labels": {"rank_canonical": "canonical", "rank_best_set": "set4"},
        "readout_role": readout_role(entity, slot, True, "median_rank_canonical"),
        # §8.2 traceability: which per-cell median the offline L_new consumes. NOT a promotion.
        "primary_input": bool(entity == PRIMARY_READOUT["entity"] and slot == PRIMARY_READOUT["slot"]),
        "primary_input_note": ("marks the per-cell median the offline primary statistic L_new (§8.2: entity W*, "
                               "slot elicit, key canonical) consumes. An INPUT: its readout_role is "
                               "secondary_diagnostic and it may not be quoted as the headline."),
    }
    return {"entity": entity, "slot": slot,
            "per_key": {k: _key_block_agg(recs, entity, k) for k in KEYS}, "canonical": can}


def aggregate_slot(recs):
    """Every §5.2 per-cell aggregate for ONE slot, plus §9.2's rank gate per entity (on the CANONICAL key) and
    the slot's onset block. C and W* stay separate throughout -- no rollup. Pure (list -> dict)."""
    n = len(recs)
    slot = recs[0]["slot"] if recs else None
    reasons = {}
    for r in recs:
        for e in ENTITIES:
            k = r["entities"][e]["per_key"][r["key_canonical"]]["prefix_reason"]
            reasons[k] = reasons.get(k, 0) + 1
    n_fail = sum(1 for r in recs if not r["key_prefix_ok"])
    ents = {e: _entity_agg(recs, e) for e in ENTITIES}
    gate = {}
    for e in ENTITIES:
        # This cell is ONE arm; the partner is None on box, so only branch 1 can resolve here (§14.2).
        c = ents[e]["canonical"]
        g = resolve_rank_gate(n_fail, {"median_rank": c["median_rank"],
                                       "median_rank_plateau": c["median_rank_plateau"],
                                       "arm": "this_cell"}, None)
        g.update({"scope": "%s/%s" % (slot, e), "slot": slot, "entity": e, "key_is_canonical": True,
                  "key": (max(KEYS, key=lambda k: c["n_by_canonical_key"][k]) if n else None)})
        gate[e] = g
    return {
        "slot": slot, "registered_readout": bool(slot == REGISTERED_READOUT_SLOT), "n": n,
        "onset": _onset_block(recs),
        "prefix": {"prefix_test": PREFIX_TEST, "n_items_prefix_fail": n_fail,
                   "n_items_prefix_fail_spec_literal": sum(1 for r in recs
                                                           if not r["key_prefix_ok_spec_literal"]),
                   "reasons_canonical_key": reasons,
                   "reasons_basis": "one count per (item, entity) canonical-key check",
                   "failing_items": [{"q": r["q"], "prompt_str": r["prompt_str"],
                                      "reasons": r["key_prefix_reasons"]} for r in recs
                                     if not r["key_prefix_ok"]]},
        "rule_k": {"n_by_canonical_key": {k: sum(1 for r in recs if r["key_canonical"] == k) for k in KEYS},
                   "regime_derivation_key": (recs[0]["key_regime_derivation"] if recs else None),
                   "n_off_regime_derivation": sum(1 for r in recs
                                                  if not r["key_canonical_matches_regime_derivation"]),
                   "note": ("rule K is applied per item to the measured prompt string; the agreement with §3's "
                            "regime derivation is REPORTED and no gate reads it")},
        "collisions": {"n_first_token_collision_space": sum(1 for r in recs
                                                            if r["first_token_collision_space"]),
                       "n_first_token_collision_bare": sum(1 for r in recs
                                                           if r["first_token_collision_bare"]),
                       "n_first_token_collision_canonical": sum(1 for r in recs
                                                                if r["first_token_collision_canonical"]),
                       "policy": ("key-dependent and counted per key; collision items are measured, dumped and "
                                  "logged, never dropped (§3.5)")},
        "entities": ents, "rank_gate": gate,
    }


def aggregate(records):
    """The per-cell aggregate over BOTH slots, keyed by slot. Nothing is pooled across slots. Pure."""
    by_slot = {s: [r for r in records if r["slot"] == s] for s in SLOTS}
    return {"n_records": len(records), "n_items_registered": N_ITEMS,
            "n_items": len(by_slot[SLOTS[0]]),
            "n_items_matches_registration": bool(len(by_slot[SLOTS[0]]) == N_ITEMS),
            "slots": {s: aggregate_slot(by_slot[s]) for s in SLOTS}}


def decide(agg):
    """The emitted decision: a FLAT ledger of named verdicts, one row per scope, the named non-emissions, and
    §8.2's designation. §9.2's branch 1 resolves per cell; the interval branches and §9.1's matching branches
    need the partner cell, so those emit their named PAIR_ABSENT non-emission -- except SLOT_DEGENERATE, whose
    condition a single arm at exactly zero already satisfies. Nothing is rolled up. Pure (dict -> dict)."""
    verdicts, slot_gates = [], {}
    for s in SLOTS:
        sa = agg["slots"][s]
        for e in ENTITIES:
            verdicts.append({k: sa["rank_gate"][e][k] for k in
                             ("rule", "verdict", "consequence", "scope", "slot", "entity", "msg")})
        on = sa["onset"]
        f = on["frac_slot_answer_onset_full"]
        # This cell is ONE arm; SLOT_DEGENERATE is symmetric in its two operands, so which position the known
        # arm occupies does not change the emitted branch. Which arm it is, is stamped as the record `regime`.
        g = resolve_slot_gate((None if f is None else float(f)), None)
        g.update({"slot": s, "scope": s,
                  "this_arm_frac_slot_answer_onset": on["frac_slot_answer_onset"],
                  "this_arm_frac_slot_answer_onset_full": on["frac_slot_answer_onset_full"],
                  "this_arm_n_slot_answer_onset": on["n_slot_answer_onset"],
                  "this_arm_onset_side_empty": on["onset_side_empty"],
                  "this_arm_decomposition": on["decomposition"],
                  "this_arm_non_onset_composition": on["non_onset_composition"]})
        slot_gates[s] = g
        verdicts.append({k: g[k] for k in ("rule", "verdict", "consequence", "slot", "scope", "msg")})
    return {
        "verdicts": verdicts,
        "rank_gate": {s: agg["slots"][s]["rank_gate"] for s in SLOTS},
        "slot_gate": slot_gates,
        "registered_readout_slot": REGISTERED_READOUT_SLOT,
        "primary_readout": dict(PRIMARY_READOUT, readout_role=readout_role(
            PRIMARY_READOUT["entity"], PRIMARY_READOUT["slot"], True, PRIMARY_READOUT["statistic"])),
        "readout_role_note": ("§8.2/A17. The designation above is the ONLY object in this artifact carrying "
                              "readout_role='primary', and it names a CROSS-CELL statistic this instrument does "
                              "not compute (§9.3 is offline). Every quantity this file emits carries "
                              "'secondary_diagnostic'. Promotion is prohibited and machine-checked by "
                              "readout_role() plus count_role()."),
        "not_emitted_here": [
            {"rule": "§9.1", "verdict_family": "SLOT_UNMATCHED / SLOT_MATCHED",
             "reason": ("abs(f_base - f_it) needs frac_slot_answer_onset at BOTH cells; one invocation measures "
                        "one cell (§14.1, A8) and §14.2 makes verdict emission offline-only. Emitted here as "
                        "SLOT_GATE_PAIR_ABSENT with this arm's inputs. SLOT_DEGENERATE IS emitted when this "
                        "arm's own onset is exactly zero.")},
            {"rule": "§9.2", "verdict_family": "RANK_RESOLUTION_INSUFFICIENT / RANK_RESOLVED",
             "reason": ("the interval rule needs both arms' median rank and median_rank_plateau. Emitted here "
                        "as RANK_GATE_PAIR_ABSENT with this arm's interval inputs. KEY_UNLOCATABLE IS emitted "
                        "from this cell's own prefix failures.")},
            {"rule": "§9.3", "verdict_family": ("SLOT_UNINTERPRETABLE / GAP_STATISTIC_DEPENDENT / GAP_CLOSED / "
                                                "GAP_SURVIVES / GAP_MOSTLY_CLOSED / GAP_INDETERMINATE / "
                                                "BAND_EMPTY_BY_CONSTRUCTION"),
             "reason": ("a cross-cell ratio on the COMMON non-collision set, plus Lp, the paired sign test and "
                        "the L_old bands: offline only (§14.2), single-sourced in controls/fmt_matched_join.py. "
                        "THIS IS THE PRIMARY READOUT (§8.2) and this instrument does not emit it; every input "
                        "it reads is persisted here.")},
            {"rule": "§9.4-§9.5", "verdict_family": ("KEY_EFFECT_BELOW_NOISE / "
                                                     "KEY_MATERIALITY_UNEVALUABLE_NO_NOISE_CONTEXT / "
                                                     "KEY_*_TO_RC / KEY_*_TO_HEADROOM"),
             "reason": "R-PROB, measured by controls/family_cave_diagnose_fmt.py; not this readout."},
            {"rule": "§7/§9.6", "verdict_family": "ANCHOR_REPRODUCES / ANCHOR_DIFFERS / ANCHOR_UNEVALUABLE",
             "reason": ("needs a second artifact (the same-box shipped reference or a committed one); offline "
                        "only (§14.2). This file emits the anchor arm's fields under their shipped names and "
                        "reads no other artifact.")},
            {"rule": "§10.3", "verdict_family": "STAB27B_UNEVALUABLE / SHIPPED_SELF_* / ARMS_*",
             "reason": "the 27b stability control compares three other artifacts; offline only (§14.2)."},
        ],
    }


# --------------------------------------------------------------------------- provenance (§12)
def validate_provenance(prov):
    """§12/M2. RAISES ProvenanceIncomplete if any PROVENANCE_KEYS field is absent, or if `lambda_instance_id` or
    `started_utc` is None or an empty/whitespace string. A null is a failure, not a note: the caller aborts
    BEFORE any model is loaded (precedent OWED.md A3, where a print-and-continue put a fabricated pool size into
    58 committed artifacts). Returns prov unchanged on success. Pure (dict -> dict)."""
    if not isinstance(prov, dict):
        raise ProvenanceIncomplete("provenance is %r, not an object" % type(prov).__name__)
    missing = [k for k in PROVENANCE_KEYS if k not in prov]
    if missing:
        raise ProvenanceIncomplete("provenance is missing required field(s): %s" % ", ".join(missing))
    for k in PROVENANCE_LOAD_BEARING:
        v = prov[k]
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ProvenanceIncomplete(
                "provenance[%r] is %r: %s are the pair that makes an artifact joinable to the audit log, and a "
                "null is a failure, not a note (§12, M2). Export LAMBDA_INSTANCE_ID (lambda_run.sh:174,177) "
                "before the run." % (k, v, " + ".join(PROVENANCE_LOAD_BEARING)))
    return prov


def build_provenance(device, dtype_str="bfloat16"):
    """The §12 stamp. `lambda_instance_id`, `git_commit` and `cuda_visible_devices` come from os.environ (as
    run_cleangate_topk_27b.sh:58-59 reads them); the rest from torch / importlib.metadata. NOT validated here --
    the caller validates, so the abort happens before the model load."""
    import torch
    from importlib.metadata import version as _ver

    def _v(mod):
        try:
            return _ver(mod)
        except Exception:
            return None

    cuda = bool(device == "cuda" and torch.cuda.is_available())
    drv = None
    if cuda:
        for get in (lambda: torch.cuda.driver_version(), lambda: torch._C._cuda_getDriverVersion()):
            try:
                drv = get()
                break
            except Exception:
                drv = None
    return {
        "gpu_name": (torch.cuda.get_device_name(0) if cuda else None),
        "gpu_count": (torch.cuda.device_count() if cuda else 0),
        "cuda_runtime": torch.version.cuda,
        "driver": (None if drv is None else str(drv)),
        "torch": getattr(torch, "__version__", None) or _v("torch"),
        "transformers": _v("transformers"),
        # transformer_lens has no __version__ (OWED.md A2) -> importlib.metadata only
        "transformer_lens": _v("transformer_lens"),
        "python": sys.version.split()[0], "dtype": dtype_str,
        "lambda_instance_id": os.environ.get("LAMBDA_INSTANCE_ID"),
        "git_commit": os.environ.get("GIT_COMMIT"),
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "finished_utc": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_index": (torch.cuda.current_device() if cuda else None),
    }


# --------------------------------------------------------------------------- real run
def _tensor_plateau(P, tok_id):
    """The TIE PLATEAU width of tok_id in the full prob tensor P: (P == p).sum(), the exact complement of the
    imported `_tensor_rank`'s 1 + (P > p).sum(), on the SAME full-precision tensor in the SAME pass (A16).
    Always >= 1 (the token itself). Returns int."""
    import torch  # noqa: F401  (P is already a torch tensor)
    p = float(P[tok_id])
    return int((P == p).sum().item())


def _measure_model(name, is_chat, device, items):
    """One model end-to-end (forward-only), loaded and FREED inside this call so only one model is resident. TWO
    forward passes per item: slot `bare` = single(q) (the shipped construction) and slot `elicit` =
    single(elicit_question(q, is_chat)) (§4.1). Measures and DUMPS every item at both slots and both keys."""
    import torch
    from transformer_lens import HookedTransformer
    from rlhf_differential import _helpers

    print("[load] %s on %s (chat=%s)" % (name, device, is_chat), flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tok = model.tokenizer
    tag = "it" if is_chat else "base"
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    # the shipped builders, unpacked exactly as the sibling instruments do; `single` and `first` are the two this
    # readout uses (raw / push / num_lp belong to prompt families this instrument does not measure)
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)

    def tok_str(tid):
        return tok.decode([int(tid)])

    def encode(s):                                   # §3.1: add_special_tokens=False on the joint re-encode
        return [int(t) for t in tok.encode(s, add_special_tokens=False)]

    def std_first(s):
        """The MEASURED id (§3.2): the shipped `first` (rlhf_differential.py:174) VERBATIM, with the single
        addition of a None sentinel where the encode is empty (which `first` would raise IndexError on)."""
        try:
            return int(first(s))
        except IndexError:
            return None

    records = []
    for it in items:
        q = it["q"]
        for slot in SLOTS:
            pid = single(q) if slot == "bare" else single(elicit_question(q, is_chat))
            with torch.no_grad():
                P = _full_softmax(model(pid))
            prompt_ids = [int(t) for t in pid[0].tolist()]
            prompt_str = tok.decode(pid[0], skip_special_tokens=False)   # §3.1: special tokens KEPT
            vals, idx = torch.topk(P, TOP_K)
            topk_10 = [dict({"tok_id": int(i), "tok_str": tok_str(int(i))}, **both_precisions("p", float(v)))
                       for v, i in zip(vals.tolist(), idx.tolist())]
            rank_cache, plat_cache = {}, {}

            def rank_at(tid, _P=P, _c=rank_cache):
                t = int(tid)
                if t not in _c:
                    _c[t] = _tensor_rank(_P, t)
                return _c[t]

            def plateau_at(tid, _P=P, _c=plat_cache):
                t = int(tid)
                if t not in _c:
                    _c[t] = _tensor_plateau(_P, t)
                return _c[t]

            rec = build_record(it, slot, is_chat, {
                "prompt_ids": prompt_ids, "prompt_str": prompt_str, "topk_10": topk_10,
                "argmax_tok_id": int(torch.argmax(P)), "tok_str": tok_str,
                "encode": encode, "std_first": std_first,
                "p_at": (lambda t, _P=P: float(_P[int(t)])), "rank_at": rank_at, "plateau_at": plateau_at})
            records.append(rec)

            if not rec["key_prefix_ok"]:
                # §3.1: failing items printed VERBATIM with q, prompt_str and both id lists.
                print("  [%s %s] KEY_PREFIX_FAIL reasons=%s\n    q=%r\n    prompt_str=%r\n    prompt_ids=%s"
                      % (tag, slot, rec["key_prefix_reasons"], q, prompt_str, prompt_ids), flush=True)
                for e in ENTITIES:
                    d = rec["entities"][e]["per_key"][rec["key_canonical"]]["prefix_detail"]
                    print("    %s joint_ids=%s (joint_n=%s prompt_n=%s first_mismatch_index=%s)"
                          % (e, d["joint_ids"], d["joint_n_tokens"], d["prompt_n_tokens"],
                             d["first_mismatch_index"]), flush=True)
            ec, ew = rec["entities"]["C"], rec["entities"]["Wstar"]
            print("  [%s %s] key=%s rank_can C=%s(plat %s) W*=%s(plat %s) | best_set C=%s W*=%s | argmax=%r "
                  "onset=%d coll(sp/bare)=%d/%d prefix_ok=%d q=%r"
                  % (tag, slot, rec["key_canonical"], ec["rank_canonical"], ec["tie_plateau_canonical"],
                     ew["rank_canonical"], ew["tie_plateau_canonical"], ec["rank_best_set"],
                     ew["rank_best_set"], rec["argmax_tok_str"], int(rec["argmax_in_variant_set_union"]),
                     int(rec["first_token_collision_space"]), int(rec["first_token_collision_bare"]),
                     int(rec["key_prefix_ok"]), q[:34]), flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    agg = aggregate(records)
    return {"name": name, "regime": "chat" if is_chat else "qa", "n_layers": nL, "n_heads": nH,
            "aggregate": agg, "decision": decide(agg), "items": records}


def run(family, name, tag, device, is_chat):
    # §12: provenance FIRST, validated BEFORE any model is loaded. A null lambda_instance_id or started_utc
    # aborts the run with a named non-zero exit; it does not warn and continue.
    prov = validate_provenance(build_provenance(device))
    print("[provenance] %s" % json.dumps(prov, default=str), flush=True)

    items = load_family(family)
    print("[family] %s -> %d items (no select_items; every item measured + dumped at both slots)"
          % (family, len(items)), flush=True)
    if len(items) != N_ITEMS:
        print("[family] NOTE n_items=%d != N_ITEMS(%d) from the registration; nothing is dropped and every "
              "denominator is the measured n" % (len(items), N_ITEMS), flush=True)
    print("[elicit] literal (foldlisten_judge.py:66) = %r" % ELICIT, flush=True)
    print("[primary] §8.2: entity=%s slot=%s key=%s statistic=%s (%s)"
          % (PRIMARY_READOUT["entity"], PRIMARY_READOUT["slot"], PRIMARY_READOUT["key"],
             PRIMARY_READOUT["statistic"], PRIMARY_READOUT["emitted_by"]), flush=True)
    print("[L_old] §8.2 reference, read by no code path here: %s" % json.dumps(L_OLD_LOG10), flush=True)
    print("[thresholds] ONSET_DELTA=%s (%s) | withdrawn, non-numeric: %s"
          % (ONSET_DELTA, ONSET_DELTA_PROVENANCE, ", ".join(sorted(WITHDRAWN_THRESHOLDS))), flush=True)

    res = _measure_model(name, is_chat, device, items)
    prov["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    out = {
        "name": name, "device": device, "tag": tag, "regime": "chat" if is_chat else "qa",
        "cue": "family_topk_shift_fmt", "family": family, "n_items": len(items),
        "metric": METRIC,
        "thresholds": {"N_ITEMS": N_ITEMS, "TOP_K": TOP_K, "DUMP_FLOOR": DUMP_FLOOR,
                       "ONSET_DELTA": ONSET_DELTA, "ONSET_DELTA_provenance": ONSET_DELTA_PROVENANCE,
                       "TOP_N_NON_ONSET": TOP_N_NON_ONSET,
                       "WITHDRAWN": WITHDRAWN_THRESHOLDS,
                       "L_OLD_LOG10_reference_only": L_OLD_LOG10,
                       "derived_conditions_no_chosen_number": {
                           "SLOT_DEGENERATE": "frac_slot_answer_onset == 0 at either arm (A15)",
                           "RANK_RESOLUTION_INSUFFICIENT": ("the two arms' [median_rank +- "
                                                            "median_rank_plateau] intervals overlap (A16)")},
                       "sources": {"TOP_K": "imported, controls/family_topk_shift.py:64",
                                   "ONSET_DELTA": ("imported ARTIFACT_MAX_DELTA, "
                                                   "controls/foldlisten_judge.py:129 (documented :125-126); "
                                                   "regime transport stamped per A20"),
                                   "DUMP_FLOOR": "the persistence format, inclusive; descriptor only (A16)",
                                   "TOP_N_NON_ONSET": "A19's report size ('the top-5 non-onset argmax tokens')",
                                   "L_OLD_LOG10": ("ADOPTED A7 from the committed artifacts; printed per §8.2 "
                                                   "and read by NO code path in this file")}},
        "decision_rule": DECISION_RULE,
        "registration": ("docs/drafts/REGISTRATION_format_matched_readout.md (frozen, pre-data, amended twice "
                         "2026-07-29: A1-A14, A15-A20): R-RANK, §3 rule K, §4.1 the slot, §5 fields, §8 "
                         "thresholds, §8.2 the primary readout, §9.1/§9.2 preconditions, §12 provenance, §13 "
                         "the stamp"),
        "elicit_literal": ELICIT,
        "elicit_literal_source": "controls/foldlisten_judge.py:66 (imported, not copied)",
        "prefix_test": PREFIX_TEST,
        "full_field_convention": FULL_FIELD_CONVENTION,
        "stamp_keys": list(STAMP_KEYS),
        "provenance": prov,
        "result": res,
    }
    out["n_primary_role_fields"] = count_role(out, ROLE_PRIMARY)     # §13: exactly 1 (the designation block)
    Path("out").mkdir(exist_ok=True)
    out_path = "out/family_topk_shift_fmt_%s.json" % tag
    Path(out_path).write_text(json.dumps(out, indent=2, default=str))

    for s in SLOTS:
        sa = res["aggregate"]["slots"][s]
        on, comp = sa["onset"], sa["onset"]["non_onset_composition"]
        print("[%s|%s] frac_slot_answer_onset=%s (n=%d/%d) onset_side_empty=%s | decomposition C_only=%s "
              "W_only=%s both=%s neither=%s  [LEVEL: no threshold, A15]"
              % (tag, s, on["frac_slot_answer_onset"], on["n_slot_answer_onset"], on["n"],
                 on["onset_side_empty"], on["decomposition"]["frac_onset_C_only"],
                 on["decomposition"]["frac_onset_W_only"], on["decomposition"]["frac_onset_both"],
                 on["decomposition"]["frac_onset_neither"]), flush=True)
        print("[%s|%s] A19 non-onset composition: n_non_onset=%d distinct=%d modal=%r x%d (tied=%s) | top%d=%s"
              % (tag, s, comp["n_non_onset"], comp["n_distinct_non_onset_argmax"],
                 comp["modal_non_onset_tok_str"], comp["n_modal_non_onset"], comp["modal_is_tied"],
                 comp["top_n"], [(r["tok_str"], r["n"], r["share_of_non_onset"]) for r in comp["top_tokens"]]),
              flush=True)
        print("[%s|%s] rule_k n_by_canonical_key=%s off_regime_derivation=%d | prefix_fail=%d (spec_literal=%d) "
              "reasons=%s"
              % (tag, s, sa["rule_k"]["n_by_canonical_key"], sa["rule_k"]["n_off_regime_derivation"],
                 sa["prefix"]["n_items_prefix_fail"], sa["prefix"]["n_items_prefix_fail_spec_literal"],
                 sa["prefix"]["reasons_canonical_key"]), flush=True)
        for e in ENTITIES:
            can, g = sa["entities"][e]["canonical"], sa["rank_gate"][e]
            pk = sa["entities"][e]["per_key"][g["key"]] if g["key"] else {}
            print("[%s|%s|%s] %s key=%s median_rank_canonical=%s (plateau=%s IQR=%s max=%s) "
                  "median_rank_best_set=%s | n_rank_resolved=%d/%d median_tie_plateau=%s | n_rank_le_10=%s "
                  "n_is_top=%s n_p_ge_1e6=%s | canonical_better_than_cross=%d/%d (cross_better=%d tied=%d) "
                  "n_id_disagree=%s | primary_input=%s role=%s"
                  % (tag, s, e, g["verdict"], g["key"], can["median_rank_canonical"],
                     can["median_rank_plateau"], can["rank_canonical"]["iqr"], can["rank_canonical"]["max"],
                     can["median_rank_best_set"], can["n_rank_resolved"], sa["n"], can["median_tie_plateau"],
                     pk.get("n_rank_le_10"), pk.get("n_is_top"), pk.get("n_p_ge_1e6"),
                     can["n_canonical_better_than_cross"], sa["n"], can["n_cross_better_than_canonical"],
                     can["n_rank_tied_canonical_cross"], pk.get("n_id_disagree"), can["primary_input"],
                     can["readout_role"]), flush=True)
        sg = res["decision"]["slot_gate"][s]
        print("[%s|%s] %s (%s)%s -- %s"
              % (tag, s, sg["verdict"], sg["consequence"],
                 (" [" + sg["threshold_provenance"] + "]") if "threshold_provenance" in sg else "",
                 sg["msg"]), flush=True)
    print("[primary] n_primary_role_fields=%d (§13: exactly 1, the designation block; this instrument emits no "
          "primary number)" % out["n_primary_role_fields"], flush=True)
    print("[done] wrote %s" % out_path, flush=True)


# --------------------------------------------------------------------------- selftest (model-free, CPU, no torch)
class _StubTok:
    """A toy sentencepiece-ish tokenizer for the model-free selftest: a fixed piece vocabulary, greedy
    LONGEST-match left to right, and two SPECIAL pieces that `skip_special_tokens=True` drops on decode while
    `add_special_tokens=True` prepends only `<bos>`. Every string the selftest feeds it round-trips under
    (skip_special_tokens=False, add_special_tokens=False), which is what makes the §3.1 matrix meaningful."""
    BOS = 1
    VOCAB = {"<bos>": 1, "Q:": 2, " x": 3, "\nA:": 4, " Paris": 5, "Paris": 6, " paris": 7, "paris": 8,
             " Lyon": 9, "Lyon": 10, " lyon": 11, "lyon": 12, "x": 13, "\n": 14,
             "<start_of_turn>": 15, "user": 16, "model": 17}
    SPECIALS = ("<bos>", "<start_of_turn>")

    def __init__(self):
        self._pieces = sorted(self.VOCAB, key=len, reverse=True)
        self._inv = {v: k for k, v in self.VOCAB.items()}

    def encode(self, s, add_special_tokens=False):
        ids = [self.BOS] if add_special_tokens else []
        i = 0
        while i < len(s):
            for p in self._pieces:
                if s.startswith(p, i):
                    ids.append(self.VOCAB[p])
                    i += len(p)
                    break
            else:
                ids.append(0)                      # UNK, one character
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens=False):
        out = []
        for t in ids:
            piece = self._inv.get(int(t), "<unk>")
            if not (skip_special_tokens and piece in self.SPECIALS):
                out.append(piece)
        return "".join(out)


def selftest():
    def ok(msg):
        print("[selftest] " + msg)

    T = _StubTok()
    BASE_PROMPT = "<bos>Q: x\nA:"                                        # ends ':' -> rule K sep ' '
    IT_PROMPT = "<bos><start_of_turn>user\nx\n<start_of_turn>model\n"    # ends '\n' -> rule K sep ''

    # ---------- the transcribed shared constants, checked against their real module when importable ----------
    try:
        from gapclose_item_joins import STAMP_KEYS as _SK, join_key as _jk
    except Exception as _e:                                   # expected on a box: not in the scp list
        _SK = _jk = None
        ok("gapclose_item_joins not importable (%s) -> transcription check SKIPPED (expected on a box)" % _e)
    if _SK is not None:
        assert STAMP_KEYS == _SK, (STAMP_KEYS, _SK)
        for s in ("a  b", "école  x ", "  Sao\tPaulo\n", None, 42):
            assert join_key(s) == _jk(s), s
        ok("STAMP_KEYS and join_key are VERBATIM gapclose_item_joins.py:109 / :195-198 (asserted, not assumed)")
    assert STAMP_KEYS == ("arm", "slot", "labels", "map_confidence", "tiebreak") and len(STAMP_KEYS) == 5

    # ---------- the reuse is real: the imported helpers come from the shipped instrument ----------
    assert _full_softmax.__module__ == "family_topk_shift" and _tensor_rank.__module__ == "family_topk_shift"
    assert rank_of.__module__ == "family_topk_shift" and load_family.__module__ == "family_topk_shift"
    assert ELICIT == "What is your final answer? Reply with only the answer."          # foldlisten_judge.py:66
    ok("imports: _full_softmax/_tensor_rank/rank_of/load_family from family_topk_shift; ELICIT frozen")

    # ---------- FROZEN block: the surviving numbers, and the WITHDRAWN ones as non-numeric notes ----------
    assert N_ITEMS == 82 and TOP_K == 10 and DUMP_FLOOR == 1e-6 and TOP_N_NON_ONSET == 5
    assert ONSET_DELTA == 0.10 and ONSET_DELTA == ARTIFACT_MAX_DELTA
    assert ONSET_DELTA_PROVENANCE == "ONSET_DELTA_BORROWED_FROM_WITHIN_MODEL_REGIME"
    assert set(WITHDRAWN_THRESHOLDS) == {"ONSET_FLOOR", "KEY_LIVE_FRAC"}
    for k, v in WITHDRAWN_THRESHOLDS.items():
        assert isinstance(v, str) and "WITHDRAWN" in v, k
    assert "ONSET_FLOOR" not in globals() and "KEY_LIVE_FRAC" not in globals()   # no readable numeric gate left
    assert L_OLD_LOG10 == {"Wstar": {"2b": 2.416, "9b": 2.899, "27b": 2.886},
                           "C": {"2b": 2.428, "9b": 1.526, "27b": 1.398}}
    ok("frozen: ONSET_DELTA=0.10 == ARTIFACT_MAX_DELTA (A2 value, A20 stamp), DUMP_FLOOR=1e-6 descriptor, "
       "TOP_K=10, N_ITEMS=82, TOP_N_NON_ONSET=5, L_old adopted (A7); ONSET_FLOOR (A15) and KEY_LIVE_FRAC (A16) "
       "exist ONLY as non-numeric withdrawal notes with no module-level name")

    # ---------- §8.2 / A17: the designation and every one-axis perturbation ----------
    P0 = PRIMARY_READOUT
    assert readout_role(P0["entity"], P0["slot"], True, P0["statistic"]) == ROLE_PRIMARY
    assert readout_role("C", P0["slot"], True, P0["statistic"]) == ROLE_SECONDARY             # entity
    assert readout_role(P0["entity"], "bare", True, P0["statistic"]) == ROLE_SECONDARY        # slot
    assert readout_role(P0["entity"], P0["slot"], False, P0["statistic"]) == ROLE_SECONDARY   # key
    for stat in ("Lp", "median_rank_canonical", "rank_first_tok", "rank_best_set", "L_old"):
        assert readout_role(P0["entity"], P0["slot"], True, stat) == ROLE_SECONDARY, stat     # statistic
    assert readout_role(P0["entity"], P0["slot"], 1, P0["statistic"]) == ROLE_SECONDARY       # bool, not truthy
    assert count_role({"a": [{"readout_role": ROLE_PRIMARY}, {"readout_role": ROLE_SECONDARY}],
                       "b": {"readout_role": ROLE_PRIMARY}}, ROLE_PRIMARY) == 2
    ok("§8.2: primary iff (entity=Wstar, slot=elicit, key canonical, statistic=L_new); every one-axis "
       "perturbation is secondary_diagnostic; count_role walks nested objects")

    # ---------- precision: the A13 defect asserted against THIS file's writer ----------
    x = 1.4999996                                    # round(x, 6) == 1.5 while the value is BELOW 1.5
    assert dump6(x) == 1.5 and float(full_str(x)) == x
    assert (dump6(x) < 1.5) is False and (float(full_str(x)) < 1.5) is True   # the flip the 6dp write hides
    assert dump6(9.999999e-7) == 1e-06 and float(full_str(9.999999e-7)) < DUMP_FLOOR
    assert both_precisions("p", 0.1) == {"p": 0.1, "p_full": repr(0.1)}
    assert both_precisions("p", None) == {"p": None, "p_full": None}
    ok("precision: <field> is 6dp (lossy at a threshold BY the format), <field>_full round-trips exactly; the "
       "A13 M0=1.5/headroom_pass=true flip is reproduced and caught, at 1.5 and at the 1e-6 dump floor")

    # ---------- rule K, both regimes ----------
    assert rule_k_sep(BASE_PROMPT) == " " and canonical_key(BASE_PROMPT) == "space"
    assert rule_k_sep(IT_PROMPT) == "" and canonical_key(IT_PROMPT) == "bare"
    assert rule_k_sep("ends with a space ") == "" and canonical_key("ends with a space ") == "bare"
    assert rule_k_sep("tab\t") == "" and rule_k_sep("nl\n") == ""
    assert rule_k_sep("") == " " and canonical_key("") == "space"                      # ambiguity G, pinned
    assert key_sep("space") == " " and key_sep("bare") == ""
    assert cross_key("space") == "bare" and cross_key("bare") == "space"
    assert regime_derivation_key(False) == "space" and regime_derivation_key(True) == "bare"
    try:
        key_sep("canonical")
        raise AssertionError("key_sep must reject a non-key")
    except ValueError:
        pass
    ok("rule K: 'Q: ...\\nA:' -> sep ' ' -> key `space`; '...model\\n' -> sep '' -> key `bare`; "
       "whitespace/tab/newline endings -> `bare`; empty -> `space` (pinned)")

    # ---------- the slot: the generation-free elicit construction ----------
    assert elicit_question("Q?", False) == "Q? What is your final answer? Reply with only the answer."
    assert elicit_question("Q?", True) == "Q?\n\nWhat is your final answer? Reply with only the answer."
    ok("slot `elicit`: base 'Q: {q} ELICIT\\nA:' and -it chat(user='{q}\\n\\nELICIT') via the SHIPPED single(); "
       "the ELICIT literal is byte-identical at both variants")

    # ---------- §3.1's two flags, on both templates, including the <bos> round-trip ----------
    ENC = {False: (lambda s: T.encode(s, add_special_tokens=False)),      # the REGISTERED joint flag
           True: (lambda s: T.encode(s, add_special_tokens=True))}       # prepends a second BOS
    mtx = {}
    for label, rawp, sep in (("base", BASE_PROMPT, " "), ("it", IT_PROMPT, "")):
        pi = T.encode(rawp, add_special_tokens=False)
        assert T.decode(pi, skip_special_tokens=False) == rawp            # the <bos> ROUND-TRIP
        for skip in (False, True):
            ps = T.decode(pi, skip_special_tokens=skip)
            for add in (False, True):
                mtx[(label, skip, add)] = prefix_check(pi, ps, sep, "Paris", ENC[add])
    assert mtx[("base", False, False)]["ok"] is True and mtx[("it", False, False)]["ok"] is True
    assert mtx[("base", False, False)]["tok_id_joint"] == T.VOCAB[" Paris"]   # key `space` at base
    assert mtx[("it", False, False)]["tok_id_joint"] == T.VOCAB["Paris"]      # key `bare` at -it
    # at the -it template the registered pair is the ONLY one that holds (§3.1's claim, where it matters):
    # skip_special_tokens=True drops <start_of_turn> too and a prepended BOS does not restore it
    for skip, add in ((True, False), (False, True), (True, True)):
        v = mtx[("it", skip, add)]
        assert v["ok"] is False and v["reason"] == "DECODE_NOT_ROUNDTRIP", (skip, add, v["reason"])
    assert mtx[("it", False, True)]["ok_spec_literal"] is False           # add=True also breaks the prefix
    # honest note: at a BOS-only template the two errors CANCEL, which is why §3.1 pins the pair on the -it shape
    assert mtx[("base", True, False)]["reason"] == "DECODE_NOT_ROUNDTRIP"
    assert mtx[("base", False, True)]["reason"] == "DECODE_NOT_ROUNDTRIP"
    assert mtx[("base", True, True)]["ok"] is True
    ok("§3.1 flags: (skip_special_tokens=False, add_special_tokens=False) is the ONLY holding combination at the "
       "-it template; at a BOS-only base template skip=True + add=True cancel, which is why the pair is pinned "
       "on the -it shape. NOTE: add_special_tokens=True also breaks the prompt re-encode, so the strict test's "
       "EARLIEST violated condition is DECODE_NOT_ROUNDTRIP and the literal prefix failure is recorded as "
       "ok_spec_literal=False")

    # ---------- every prefix-failure reason, on planted id lists (pure) ----------
    assert prefix_verdict([1, 2, 3], [1, 2, 3, 9], [1, 2, 3])["reason"] == "OK"
    assert prefix_verdict([1, 2, 3], [1, 2, 3, 9], [1, 2])["reason"] == "DECODE_NOT_ROUNDTRIP"
    assert prefix_verdict([1, 2, 3], [1, 2], [1, 2, 3])["reason"] == "JOINT_SHORTER_THAN_PROMPT"
    mism = prefix_verdict([1, 2, 3], [1, 7, 3, 9], [1, 2, 3])
    assert mism["reason"] == "PREFIX_ID_MISMATCH" and mism["first_mismatch_index"] == 1
    empty = prefix_verdict([1, 2, 3], [1, 2, 3], [1, 2, 3])
    assert empty["reason"] == "CONTINUATION_EMPTY_IN_JOINT" and empty["ok_spec_literal"] is True
    lossy = prefix_verdict([1, 2, 3], [1, 2, 3, 9], [1, 2])
    assert lossy["ok"] is False and lossy["ok_spec_literal"] is True      # strictly stronger than the literal
    assert prefix_verdict([1, 2, 3], [1, 2, 3, 9], [1, 2, 3])["joint_ids"] is None    # failures only
    assert lossy["joint_ids"] == [1, 2, 3, 9]
    assert prefix_verdict([1, 2], [1, 2, 77], [1, 2])["tok_id_joint"] == 77   # planted standalone-vs-joint
    ok("prefix test %s: all four failure reasons in registered order; ok_spec_literal beside ok; joint ids "
       "persisted on failure only; tok_id_joint read from the joint tokenisation alone" % PREFIX_TEST)

    # ---------- V(A): construction, order, dedup BY TOKEN ID ----------
    assert [r["text"] for r in variant_texts("Paris")] == [" Paris", "Paris", " paris", "paris"]
    assert [r["variant"] for r in variant_texts("Paris")] == [1, 2, 3, 4]
    assert [r["text"] for r in variant_texts("paris")] == [" paris", "paris", " paris", "paris"]  # 1==3, 2==4
    assert lower_initial("") == "" and lower_initial("Sao Paulo") == "sao Paulo"
    ok("V(A): the frozen 2x2 in its registered order; an already-lower-case answer collides 1==3 and 2==4")

    # ---------- rank + tie plateau: the imported convention and its EXACT complement (A16) ----------
    pm = {5: 0.5, 6: 0.25, 7: 0.25, 8: 0.0}
    assert rank_of(pm, 5) == 1 and rank_of(pm, 6) == 2 and rank_of(pm, 7) == 2 and rank_of(pm, 8) == 4
    tie = {1: 0.4, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.1}                       # a deliberate 3-wide tie block
    assert rank_of(tie, 2) == rank_of(tie, 3) == rank_of(tie, 4) == 2     # a plateau shares one rank
    assert plateau_of(tie, 2) == plateau_of(tie, 3) == plateau_of(tie, 4) == 3
    assert plateau_of(tie, 1) == 1 and plateau_of(tie, 5) == 1
    assert rank_of(tie, 5) == rank_of(tie, 2) + plateau_of(tie, 2) == 5   # the exact complementarity
    ok("A16: tie_plateau = (P == p).sum() is the EXACT complement of 1 + (P > p).sum() -- a 3-wide plateau "
       "shares rank 2 and the next lower token sits at rank 2 + 3 = 5; rank_resolved iff plateau == 1")

    # ---------- median_with_plateau at odd and even n (ambiguity H) ----------
    m3 = median_with_plateau([(10, 1), (2, 7), (30, 2)])
    assert m3[0] == 10 and m3[1] == 1 and [d["rank"] for d in m3[2]] == [10]           # odd n: exact
    m4 = median_with_plateau([(2, 1), (10, 3), (20, 9), (40, 1)])
    assert m4[0] == 15 and m4[1] == 9 and [d["rank"] for d in m4[2]] == [10, 20]       # even n: MAX of the two
    assert median_with_plateau([(None, None)]) == (None, None, [])
    ok("median_rank_plateau: exact at odd n; at even n the MAX of the two defining items' plateaus -- the "
       "widest interval, the suppressing direction (ambiguity H)")

    # ---------- the §9.2 interval rule: disjoint, touching, overlapping ----------
    assert intervals_overlap(10, 2, 30, 2) is False                       # [8,12] vs [28,32]
    assert intervals_overlap(10, 2, 14, 3) is True                        # [8,12] vs [11,17]
    assert intervals_overlap(10, 2, 14, 2) is True                        # [8,12] vs [12,16]: TOUCHING
    assert intervals_overlap(10, 2, 15, 2) is False                       # [8,12] vs [13,17]: just disjoint
    assert intervals_overlap(None, 2, 15, 2) is None and intervals_overlap(10, None, 15, 2) is None
    ok("§9.2 interval rule: disjoint / touching (counted as overlapping, the suppressing direction) / "
       "overlapping / undecidable")

    # ---------- build_record end to end on planted numbers, both regimes ----------
    def mk_m(prompt_str, prompt_ids, pmap, tok_of, argmax):
        return {"prompt_ids": prompt_ids, "prompt_str": prompt_str,
                "topk_10": [dict({"tok_id": t, "tok_str": "tok%d" % t}, **both_precisions("p", pmap[t]))
                            for t in sorted(pmap, key=lambda t: (-pmap[t], t))[:TOP_K]],
                "argmax_tok_id": argmax, "tok_str": (lambda t: "tok%d" % t),
                "encode": (lambda s: T.encode(s, add_special_tokens=False)),
                "std_first": (lambda s: tok_of.get(s)),
                "p_at": (lambda t: pmap[t]), "rank_at": (lambda t: rank_of(pmap, t)),
                "plateau_at": (lambda t: plateau_of(pmap, t))}

    ITEM = {"q": "Q: x?", "correct": "Paris", "Wstar": "Lyon"}
    # ids: " Paris"=5 "Paris"=6 " paris"=7 "paris"=8 | " Lyon"=9 "Lyon"=10 " lyon"=11 "lyon"=12 | 99 = other
    PM = {5: 0.40, 6: 0.20, 7: 0.05, 8: 0.01, 9: 0.10, 10: 0.02, 11: 0.005, 12: 1e-6, 99: 0.215}
    TOKOF = {" Paris": 5, "Paris": 6, " paris": 7, "paris": 8, " Lyon": 9, "Lyon": 10, " lyon": 11, "lyon": 12}
    pids = T.encode(BASE_PROMPT)
    m_base = mk_m(BASE_PROMPT, pids, PM, TOKOF, 5)
    rb = build_record(ITEM, "elicit", False, m_base)
    assert rb["key_canonical"] == "space" and rb["key"] == "space" and rb["key_is_canonical"] is True
    assert rb["variant_set"] == "canonical" and rb["register"] == "rank_first_tok"
    assert rb["readout_role"] == ROLE_SECONDARY                          # no per-item number is primary
    assert rb["entities"]["C"]["per_key"]["space"]["key_is_canonical"] is True
    assert rb["entities"]["C"]["per_key"]["bare"]["key_is_canonical"] is False   # ambiguity C: not vacuous
    assert rb["entities"]["C"]["per_key"]["space"]["tok_id_standalone"] == 5
    assert rb["entities"]["C"]["per_key"]["bare"]["tok_id_standalone"] == 6
    assert rb["entities"]["C"]["rank_canonical"] == rank_of(PM, 5) == 1
    assert rb["entities"]["C"]["tie_plateau_canonical"] == 1
    assert rb["entities"]["C"]["rank_resolved_canonical"] is True
    assert rb["entities"]["C"]["rank_cross"] == rank_of(PM, 6)
    assert rb["entities"]["C"]["canonical_better_than_cross"] is True
    assert rb["entities"]["C"]["n_variants_deduped"] == 4 and rb["entities"]["C"]["rank_best_set"] == 1
    assert rb["entities"]["Wstar"]["rank_canonical"] == rank_of(PM, 9)
    assert rb["entities"]["Wstar"]["rank_best_set"] == min(rank_of(PM, t) for t in (9, 10, 11, 12))
    assert all(v["tie_plateau"] == 1 for v in rb["entities"]["Wstar"]["variants"])
    assert rb["argmax_in_V_C"] is True and rb["argmax_in_V_W"] is False
    assert rb["argmax_in_variant_set_union"] is True
    assert rb["first_token_collision_space"] is False and rb["first_token_collision_bare"] is False
    assert rb["key_prefix_ok"] is True and rb["key_prefix_ok_spec_literal"] is True
    assert rb["join_key"] == join_key(ITEM["q"]) and rb["prompt_n_tokens"] == len(pids)
    assert rb["key_canonical_matches_regime_derivation"] is True
    assert rb["entities"]["C"]["variant_set_labels"] == {"rank_canonical": "canonical",
                                                        "rank_best_set": "set4"}       # ambiguity D
    assert "anchor_shipped" not in rb                                    # anchor fields are slot=`bare` only
    a = build_record(ITEM, "bare", False, m_base)["anchor_shipped"]
    assert a["cid"] == 5 and a["aid"] == 9 and a["first_token_collision"] is False
    assert a["rank_c_bare"] == 1 and a["p_c_bare"] == dump6(0.40) and float(a["p_c_bare_full"]) == 0.40
    assert a["rank_w_bare"] == rank_of(PM, 9) and set(a["topk_bare"][0]) == {"tok_id", "tok_str", "p"}
    assert a["readout_role"] == ROLE_SECONDARY                           # every `bare`-slot number is secondary
    # the -it regime: rule K flips the canonical key to `bare` on the SAME numbers
    ri = build_record(ITEM, "elicit", True, mk_m(IT_PROMPT, T.encode(IT_PROMPT), PM, TOKOF, 6))
    assert ri["key_canonical"] == "bare" and ri["key"] == "bare" and ri["key_is_canonical"] is True
    assert ri["entities"]["C"]["rank_canonical"] == rank_of(PM, 6)
    assert ri["entities"]["C"]["per_key"]["bare"]["key_is_canonical"] is True
    assert ri["key_canonical_matches_regime_derivation"] is True and ri["regime"] == "chat"
    assert ri["argmax_in_V_C"] is True and ri["key_prefix_ok"] is True
    ok("build_record: rule K flips the canonical key between regimes on IDENTICAL numbers; both keys measured; "
       "rank_canonical/tie_plateau/rank_best_set/variant dedup/argmax membership/anchor fields exact")

    # ---------- a key-dependent collision, recorded per key ----------
    rc = build_record({"q": "collide?", "correct": "Paris", "Wstar": "Paris"}, "elicit", False, m_base)
    assert rc["first_token_collision_space"] is True and rc["first_token_collision_bare"] is True
    rc2 = build_record(ITEM, "elicit", False, mk_m(BASE_PROMPT, pids, PM, {**TOKOF, " Lyon": 5}, 5))
    assert rc2["first_token_collision_space"] is True and rc2["first_token_collision_bare"] is False
    assert rc2["first_token_collision_canonical"] is True          # canonical == space in the base regime
    ok("first_token_collision is recorded PER KEY and the canonical flag follows rule K")

    # ---------- the stamp: the shipped 5-tuple, all non-empty PROSE strings ----------
    # NOTE the contract being tested: the arms lineage asserts isinstance(v, str) and non-empty on EVERY stamp
    # value, so `labels` and `map_confidence` are 'n/a' WITH their reason attached, not bare tokens. Testing for
    # exact equality with "n/a" would contradict the builder and the lineage.
    for slot in SLOTS:
        for chat in (False, True):
            st = stamp_for(slot, chat)
            assert tuple(st) == STAMP_KEYS and len(st) == 5 and set(st) == set(STAMP_KEYS)
            assert all(isinstance(v, str) and v.strip() for v in st.values()), (slot, chat, st)
            assert st["map_confidence"].startswith("n/a") and st["labels"].startswith("n/a")
            assert st["arm"].startswith("fold") and slot in st["slot"]
            assert "strictly-greater" in st["tiebreak"] and "first_token_collision" in st["tiebreak"]
            assert "(P == p).sum()" in st["tiebreak"] and "COMMON" in st["tiebreak"]
    for r in (rb, ri):
        assert tuple(r["stamp"]) == STAMP_KEYS
        assert all(isinstance(v, str) and v.strip() for v in r["stamp"].values())
        for ax, ty in (("key", str), ("key_is_canonical", bool), ("variant_set", str), ("register", str),
                       ("readout_role", str)):
            assert ax in r and r[ax] is not None and isinstance(r[ax], ty), (ax, r.get(ax))
        assert r["key"] in KEYS and r["variant_set"] in ("canonical", "set4")
        assert r["register"] == "rank_first_tok" and r["readout_role"] == ROLE_SECONDARY
    ok("stamp: exactly %s, len 5, all non-empty PROSE strings (map_confidence and labels start 'n/a' and carry "
       "their reason, per the arms lineage's all-string contract) with the A16 plateau rule in `tiebreak`; the "
       "five A9/A17 axes present, non-null and typed" % (STAMP_KEYS,))

    # ---------- rank_summary: quartile convention, nulls with reasons ----------
    rs = rank_summary([(1, None), (3, None), (50, None), (None, "STANDALONE_ENCODE_EMPTY")])
    assert rs["n_items"] == 4 and rs["n"] == 3 and rs["n_null"] == 1
    assert rs["null_reasons"] == {"STANDALONE_ENCODE_EMPTY": 1} and rs["median"] == 3 and rs["max"] == 50
    qs = statistics.quantiles([1, 3, 50], n=4, method="inclusive")
    assert rs["q1"] == qs[0] and rs["q3"] == qs[2] and rs["iqr"] == qs[2] - qs[0]
    assert rank_summary([])["median"] is None and rank_summary([(1, None)])["q1"] is None
    ok("rank_summary: median/q1/q3/iqr/max on statistics.quantiles(n=4, inclusive); nulls counted by reason")

    # ---------- aggregate: onset union, decomposition, A19, DUMP_FLOOR inclusive, no rollup ----------
    recs = [build_record(ITEM, s, False, m_base) for s in SLOTS]
    recs += [build_record(dict(ITEM, q="q2"), s, False, mk_m(BASE_PROMPT, pids, PM, TOKOF, 99))
             for s in SLOTS]                                             # argmax outside both variant sets
    agg = aggregate(recs)
    el = agg["slots"]["elicit"]
    assert el["n"] == 2 and el["registered_readout"] is True
    assert agg["slots"]["bare"]["registered_readout"] is False
    on = el["onset"]
    assert on["n_slot_answer_onset"] == 1 and on["frac_slot_answer_onset"] == 0.5
    assert float(on["frac_slot_answer_onset_full"]) == 0.5 and on["onset_side_empty"] is False
    d = on["decomposition"]
    assert (d["n_onset_C_only"], d["n_onset_W_only"], d["n_onset_both"], d["n_onset_neither"]) == (1, 0, 0, 1)
    assert abs(sum(d["frac_%s" % k] for k in ("onset_C_only", "onset_W_only", "onset_both",
                                              "onset_neither")) - 1.0) < 1e-12
    comp = on["non_onset_composition"]
    assert comp["n_non_onset"] == 1 and comp["modal_non_onset_tok_id"] == 99
    assert comp["n_modal_non_onset"] == 1 and comp["modal_is_tied"] is False
    assert comp["top_tokens"][0]["share_of_non_onset"] == 1.0 and comp["top_tokens"][0]["share_of_items"] == 0.5
    assert len(comp["top_tokens"]) == 1 and comp["top_n"] == TOP_N_NON_ONSET
    # A19 at onset ZERO (ambiguity I): the ONSET side is empty, the composition is FULLY populated
    zon = _onset_block([build_record(dict(ITEM, q="z%d" % i), "elicit", False,
                                    mk_m(BASE_PROMPT, pids, PM, TOKOF, 99)) for i in range(3)])
    assert zon["frac_slot_answer_onset"] == 0.0 and zon["onset_side_empty"] is True
    zd = zon["decomposition"]
    assert (zd["n_onset_C_only"], zd["n_onset_W_only"], zd["n_onset_both"]) == (0, 0, 0)
    assert zd["n_onset_neither"] == 3 and zd["frac_onset_neither"] == 1.0
    zc = zon["non_onset_composition"]
    assert zc["n_non_onset"] == 3 and zc["n_modal_non_onset"] == 3 and zc["top_tokens"][0]["tok_id"] == 99
    assert zc["top_tokens"][0]["share_of_non_onset"] == 1.0
    assert _onset_block([])["non_onset_composition"]["n_non_onset"] == 0    # the genuinely empty case
    # DUMP_FLOOR is INCLUSIVE (descriptor only after A16)
    assert PM[12] == 1e-6
    assert el["entities"]["Wstar"]["per_key"]["bare"]["n"] == 2
    assert el["entities"]["Wstar"]["per_key"]["bare"]["n_p_ge_1e6"] == 2              # id 10, p = 0.02
    assert _key_block_agg([build_record(dict(ITEM, Wstar="lyon"), "elicit", False, m_base)],
                          "Wstar", "bare")["n_p_ge_1e6"] == 1             # id 12, p == 1e-6 exactly -> counts
    assert _key_block_agg([build_record(dict(ITEM, Wstar="lyon"), "elicit", False,
                                        mk_m(BASE_PROMPT, pids, {**PM, 12: 9.999999e-7}, TOKOF, 5))],
                          "Wstar", "bare")["n_p_ge_1e6"] == 0             # just below -> does NOT count
    canC = el["entities"]["C"]["canonical"]
    assert canC["median_rank_canonical"] == 1 and canC["median_rank"] == 1 and canC["median_rank_plateau"] == 1
    assert canC["n_rank_resolved"] == 2 and canC["median_tie_plateau"] == 1
    assert canC["n_canonical_better_than_cross"] == 2
    assert canC["rank_canonical"]["median"] == el["entities"]["C"]["per_key"]["space"]["rank"]["median"]
    assert canC["primary_input"] is False and canC["readout_role"] == ROLE_SECONDARY  # entity C is secondary
    canW = el["entities"]["Wstar"]["canonical"]
    assert canW["primary_input"] is True and canW["readout_role"] == ROLE_SECONDARY   # input, NOT primary
    assert agg["slots"]["bare"]["entities"]["Wstar"]["canonical"]["primary_input"] is False
    assert set(el["entities"]) == set(ENTITIES)
    ok("aggregate: onset union + four-way decomposition summing to 1; A19 composition on both denominators, and "
       "at onset ZERO the onset side is empty while the composition is fully populated (ambiguity I); "
       "n_p_ge_1e6 inclusive exactly at 1e-6 and exclusive just below; median_rank_canonical == the canonical "
       "key's own median; W*/elicit is the primary INPUT and still secondary_diagnostic; C and W* never pooled")

    # ---------- §9.2 rank gate: every branch, order, boundaries ----------
    A = {"median_rank": 10, "median_rank_plateau": 2}                     # [8, 12]
    B_far = {"median_rank": 30, "median_rank_plateau": 2}                 # [28, 32] -> disjoint
    B_touch = {"median_rank": 14, "median_rank_plateau": 2}               # [12, 16] -> touching
    B_over = {"median_rank": 11, "median_rank_plateau": 1}                # [10, 12] -> overlapping
    assert resolve_rank_gate(0, A, B_far)["verdict"] == "RANK_RESOLVED"
    assert resolve_rank_gate(0, A, B_far)["consequence"] == "emitted"
    assert resolve_rank_gate(0, A, B_touch)["verdict"] == "RANK_RESOLUTION_INSUFFICIENT"
    assert resolve_rank_gate(0, A, B_over)["verdict"] == "RANK_RESOLUTION_INSUFFICIENT"
    assert resolve_rank_gate(0, A, B_over)["consequence"] == "suppresses"
    assert resolve_rank_gate(0, A, None)["verdict"] == "RANK_GATE_PAIR_ABSENT"
    assert resolve_rank_gate(0, A, None)["consequence"] == "not_emitted"
    assert resolve_rank_gate(0, A, {"median_rank": None,
                                    "median_rank_plateau": None})["verdict"] == "RANK_RESOLUTION_INSUFFICIENT"
    assert resolve_rank_gate(1, A, B_far)["verdict"] == "KEY_UNLOCATABLE"          # branch 1 beats a resolved
    two = resolve_rank_gate(1, A, B_over)                                          # branches 1 AND 2 both hold
    assert two["verdict"] == "KEY_UNLOCATABLE" and two["consequence"] == "suppresses", two
    assert resolve_rank_gate(1, None, None)["verdict"] == "KEY_UNLOCATABLE"        # beats PAIR_ABSENT too
    for g in (resolve_rank_gate(0, A, B_far), resolve_rank_gate(1, A, B_far), resolve_rank_gate(0, A, None)):
        assert g["rule"] == "§9.2" and g["msg"] and g["readout_role"] == ROLE_SECONDARY
    ok("§9.2: KEY_UNLOCATABLE -> RANK_RESOLUTION_INSUFFICIENT -> RANK_RESOLVED; disjoint resolves, touching and "
       "overlapping suppress, an unusable median suppresses, a missing arm is the named non-emission, and a "
       "two-branch input resolves to KEY_UNLOCATABLE")

    # ---------- §9.1 slot gate: every branch, order, boundaries, the A20 stamp ----------
    assert resolve_slot_gate(0.0, 0.9)["verdict"] == "SLOT_DEGENERATE"             # base arm at zero
    assert resolve_slot_gate(0.9, 0.0)["verdict"] == "SLOT_DEGENERATE"             # -it arm at zero
    assert resolve_slot_gate(0.0, 0.0)["verdict"] == "SLOT_DEGENERATE"
    assert resolve_slot_gate(0.0, None)["verdict"] == "SLOT_DEGENERATE"            # emittable from ONE arm
    assert resolve_slot_gate(None, 0.0)["verdict"] == "SLOT_DEGENERATE"
    assert resolve_slot_gate(0.0, 0.9)["consequence"] == "suppresses"
    one82 = 1 / 82                                     # NOT degenerate at 1/82 (§8.1's stated weakness)
    assert resolve_slot_gate(one82, one82)["verdict"] == "SLOT_MATCHED"
    assert resolve_slot_gate(one82, 0.9)["verdict"] == "SLOT_UNMATCHED"
    assert resolve_slot_gate(0.02, 0.05)["verdict"] == "SLOT_MATCHED"              # low but matched -> licensed
    assert resolve_slot_gate(0.99, 0.80)["verdict"] == "SLOT_UNMATCHED"            # high but mismatched
    assert resolve_slot_gate(0.5, None)["verdict"] == "SLOT_GATE_PAIR_ABSENT"      # branch 1 unsatisfied
    assert resolve_slot_gate(None, None)["verdict"] == "SLOT_GATE_PAIR_ABSENT"
    assert resolve_slot_gate(0.5, None)["consequence"] == "not_emitted"
    d_exact = abs(0.95 - 0.80)                         # ONSET_DELTA at and just inside its boundary
    assert resolve_slot_gate(0.80, 0.95, onset_delta=d_exact)["verdict"] == "SLOT_MATCHED"       # D == delta
    assert resolve_slot_gate(0.80, 0.95,
                             onset_delta=d_exact * (1 - 1e-12))["verdict"] == "SLOT_UNMATCHED"  # just inside
    assert resolve_slot_gate(0.80, 0.95)["verdict"] == "SLOT_UNMATCHED"            # D = 0.15 > 0.10
    assert resolve_slot_gate(0.80, 0.85)["verdict"] == "SLOT_MATCHED"              # D = 0.05 <= 0.10
    assert resolve_slot_gate(0.80, 0.95)["threshold_provenance"] == ONSET_DELTA_PROVENANCE       # A20
    assert resolve_slot_gate(0.80, 0.85)["threshold_provenance"] == ONSET_DELTA_PROVENANCE
    assert "threshold_provenance" not in resolve_slot_gate(0.0, 0.9)
    assert "threshold_provenance" not in resolve_slot_gate(0.5, None)
    for a_, b_, c_ in ((0.0, 0.9, "suppresses"), (0.8, 0.95, "emitted_downgraded"), (0.8, 0.85, "emitted"),
                       (0.5, None, "not_emitted")):
        g = resolve_slot_gate(a_, b_)
        assert g["consequence"] == c_ and g["rule"] == "§9.1" and g["msg"], (a_, b_, g)
    ok("§9.1: SLOT_DEGENERATE -> SLOT_UNMATCHED -> SLOT_MATCHED; degeneracy fires at exactly 0 (from one arm "
       "too) and NOT at 1/82; low-but-matched licenses and high-but-mismatched does not; ONSET_DELTA inclusive "
       "at the boundary and failing just inside; the A20 stamp on both ONSET_DELTA branches only")

    # ---------- decide(): every scope emitted, nothing silent, exactly one primary ----------
    dec = decide(agg)
    assert len(dec["verdicts"]) == len(SLOTS) * (len(ENTITIES) + 1)       # 2 entities + 1 slot gate per slot
    assert all(v.get("rule") in ("§9.1", "§9.2") and v["verdict"] for v in dec["verdicts"])
    assert {v["verdict"] for v in dec["verdicts"]} <= {"KEY_UNLOCATABLE", "RANK_RESOLUTION_INSUFFICIENT",
                                                       "RANK_RESOLVED", "RANK_GATE_PAIR_ABSENT",
                                                       "SLOT_DEGENERATE", "SLOT_UNMATCHED", "SLOT_MATCHED",
                                                       "SLOT_GATE_PAIR_ABSENT"}
    for s in SLOTS:
        sg = dec["slot_gate"][s]
        assert sg["verdict"] == "SLOT_GATE_PAIR_ABSENT"                  # this arm is 0.5, partner absent
        assert sg["this_arm_frac_slot_answer_onset"] == 0.5 and sg["this_arm_n_slot_answer_onset"] == 1
        assert sg["this_arm_non_onset_composition"]["n_non_onset"] == 1
        assert set(dec["rank_gate"][s]) == set(ENTITIES)
        assert dec["rank_gate"][s]["Wstar"]["verdict"] == "RANK_GATE_PAIR_ABSENT"
    assert [x["rule"] for x in dec["not_emitted_here"]] == ["§9.1", "§9.2", "§9.3", "§9.4-§9.5", "§7/§9.6",
                                                            "§10.3"]
    assert dec["primary_readout"]["readout_role"] == ROLE_PRIMARY
    envelope = {"thresholds": {"WITHDRAWN": WITHDRAWN_THRESHOLDS},
                "result": {"aggregate": agg, "decision": dec, "items": recs}}
    assert count_role(envelope, ROLE_PRIMARY) == 1                       # §13: exactly one, the designation
    assert count_role(envelope, ROLE_SECONDARY) > 1
    ok("decide(): one named verdict per (slot, entity) plus the per-slot slot-gate row; §9.1/§9.2 pair "
       "branches, §9.3 (the PRIMARY), §9.5, §7 and §10 named as NOT emitted here with their reasons; exactly "
       "ONE readout_role='primary' in the whole envelope and it is the designation block")

    # ---------- provenance: complete passes, a null in either load-bearing field RAISES ----------
    good = {k: "x" for k in PROVENANCE_KEYS}
    good["finished_utc"] = None                      # legitimately null at validation time
    good["lambda_instance_id"] = "bb0aa8d8bff84327a2560aff811506bc"
    good["started_utc"] = "2026-07-29T00:00:00+00:00"
    assert validate_provenance(dict(good)) is not None
    for k in PROVENANCE_LOAD_BEARING:
        for bad in (None, "", "   "):
            p = dict(good)
            p[k] = bad
            try:
                validate_provenance(p)
                raise AssertionError("validate_provenance must RAISE on %s=%r" % (k, bad))
            except ProvenanceIncomplete:
                pass
    for k in PROVENANCE_KEYS:
        p = dict(good)
        del p[k]
        try:
            validate_provenance(p)
            raise AssertionError("validate_provenance must RAISE on a missing %s" % k)
        except ProvenanceIncomplete:
            pass
    try:
        validate_provenance(None)
        raise AssertionError("validate_provenance must RAISE on a non-object")
    except ProvenanceIncomplete:
        pass
    ok("provenance: all %d fields required; a null/empty lambda_instance_id or started_utc RAISES "
       "ProvenanceIncomplete (the run aborts before any model load)" % len(PROVENANCE_KEYS))

    print("[selftest] PASS")
    print("SELFTEST PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--family", default="verifier_family",
                   help="'verifier_family' (the module's ITEMS) OR a path to a JSON list of {q,correct,Wstar,...}")
    p.add_argument("--name", default="google/gemma-2-9b", help="model (base is the QA template; -it via --chat)")
    p.add_argument("--tag", default="fmt_ext2_9bbase", help="§14.1 tag pattern: fmt_ext2_{2b,9b,27b}{base,it}")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--chat", action="store_true",
                   help="use the chat template (-it model; qa template otherwise; base cell first, §1/A8)")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    try:
        run(args.family, args.name, args.tag, args.device, args.chat)
    except ProvenanceIncomplete as e:
        print("[abort] %s: %s" % (ABORT_PROVENANCE, e), flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
