# REGISTRATION — the format-matched base-vs-`-it` readout (rank + probability)

Closes `OWED.md` **C1** and `DIST_COVERAGE.md` gap **8** ("base-vs-`-it` rank comparison is
uninterpretable"), and adds the missing 27b same-box control named at
`docs/drafts/GROUNDING_crossvariant_scale.md:543`. Written before any line of the instruments exists
and before any format-matched number has been computed at any cell.

Three things are fixed here and nothing else: **what token is measured**, **at what slot**, and
**what every possible outcome will mean**.

**AMENDED 2026-07-29 (twice), before any data, after two independent review rounds. The amendment log
is §0.2 and every changed threshold is marked `AMENDED` at the point of use. Nothing below was
rewritten silently. The primary readout is designated in §8.2; every other verdict is secondary.**

---

## 0. Honesty gate — what had been seen when this was written

**Seen, and it is a lot.** Every number quoted in §2 was computed BEFORE this registration was
written and is on disk in `docs/drafts/GROUNDING_crossvariant_scale.md` §4.1, §4.2, §5, §6, §12, §14,
in `out/cleangate_same_box_result.json`, `out/b1_fold_identity_gate{,_27b}.json` and
`results_r1_dist_27b/out/r1_27b_determinism_rider.json`. This document is therefore **not blind**,
and the only defensible test of it is whether its thresholds could have been fitted to those numbers.
§8.1 addresses that threshold by threshold.

**Not seen, because it does not exist anywhere:** any rank or probability measured at a
format-matched slot, at any cell, at any key other than `" " + X`; any per-token log-prob vector; any
tie-plateau width at any measured key; any second draw of the shipped `family_cave_diagnose` on the
box where its re-parameterised twin ran.

**The specific fitting hazard this design has to survive.** The prior numbers are extreme (a factor
260–790 on median rank; a 2.6–5.1 nat residual against a 0.5 nat margin). Extreme priors make it easy
to write a rule that only one outcome can satisfy. Three structural defences are used instead of
promises: (a) every surviving threshold is either a constant **already committed** elsewhere in the
repo with a documented meaning, or a condition **derived from the statistic** with no chosen number in
it; (b) the outcome "the confound was not the explanation" (§9.3 `GAP_SURVIVES`) and the outcome "the
motivating estimate in `GROUNDING_crossvariant_scale.md` §4.2 was wrong" (§9.5
`KEY_IMMATERIAL_TO_RC`) are both written out, both reachable, and both retract this registration's own
motivation; (c) **one** primary readout is designated in advance (§8.2), so a positive found among the
diagnostics cannot become the result.

**Claim-blind authorship.** The author does not know which outcome any draft wants. No sentence below
may be read as a prediction of preference. §5.3 and §6.3 register two directional *predictions*
explicitly so they cannot later be produced as post-hoc excuses, in the form
`REGISTRATION_listen_distributional.md` §4 established.

### 0.1 Second honesty disclosure — the two amendment rounds

**Round 1** was written after three further measurements were reported to the author and still before
any data from this registration exists: the top-k artifact/hardware census (§7.2), the 27b bf16 gap
structure (§7.2), and the 1148-cell cluster comparison (§10). Every round-1 move made a verdict
**harder** to emit or replaced an invented threshold with a borrowed one. No round-1 threshold was
loosened.

**Round 2** (this one) was written after an independent skeptic with no exposure to the earlier
version read the amended document cold. It is **not** uniformly conservative, and that must be stated
plainly:

- **A15 is the first LOOSENING in this document.** It removes an absolute floor. It is justified as
  the correction of an instrument-choice error — the floor measured a *level* when the confound is a
  *mismatch* — and not as a convenience. Its neutrality argument is that §9.1 gates all four §9.3
  outcomes identically, so relaxing it changes **which cells produce a verdict**, never **which
  verdict they produce**.
- **A16 is neither a loosening nor a tightening** but a replacement of a chosen fraction by a
  measured resolution condition. It is strictly stricter in the case the withdrawn gate existed for (a
  dead key produces an enormous tie plateau and an automatic suppression) and permissive where ranks
  genuinely resolve.
- **A17 restricts what may be quoted** and is a tightening of the reporting discipline, not of a
  threshold.

### 0.2 AMENDMENT LOG — 2026-07-29, pre-data

Round 1: A1–A14. Round 2: A15–A20.

| # | what changed | from → to | why |
|---|---|---|---|
| A1 | `ONSET_FLOOR` | 0.50 → 0.75 | The original justification cited old-slot `n_is_top` = 0.659/0.805/0.854, i.e. it selected from {0.25, 0.5, 0.75} knowing which values the data clears. Re-justified without citing a measured onset value, and moved in the harder direction. **SUPERSEDED BY A15 — the threshold is withdrawn entirely** |
| A2 | `ONSET_DELTA` | 0.25 → 0.10 | Round-1 rationale: with `ONSET_FLOOR = 0.75` both arms lie in [0.75, 1.0] so a 0.25 gate would be vacuous. **That rationale is superseded by A15** (no floor, so a 0.25 gate would not be vacuous); the **value is unchanged** and now stands solely on the borrowing — `ARTIFACT_MAX_DELTA` at `controls/foldlisten_judge.py:129`, documented at `:125-126` as "the repo's existing 'two arms land at the same place' tolerance". Recorded this way because a value whose justification changed while its number did not is exactly what an audit needs to see |
| A3 | `dRC >= 0.5` / `dM0 >= 1.5` as the §9.5 decision inputs | **removed as decision inputs**; replaced by a flip-count rule against `MIN_FAITHFUL = 8` | A median-of-per-item-differences is not the statistic `MARGIN_FAITHFUL` was calibrated on (M1). The replacement decides on the quantity that moves a verdict — the count of `faithful_*` flips — against the committed count the shipped categories turn on. `dRC` / `dM0` survive as reported magnitudes with no verdict. Direction: harder to declare the key material, i.e. harder to support this registration's own motivation |
| A4 | 27b rank tolerance | exact everywhere → exact **only** vs the same-box session reference; `DISCLOSED_NOT_GATED` vs the committed PCIe column | Every committed `family_topk_shift*` artifact is H100 PCIe / 570.148.08; no such artifact exists on any other card; we must launch on `gpu_1x_h100_sxm5` because PCIe has zero capacity. An exact gate there would test hardware (§7.2) |
| A5 | 2b/9b anchor status | "reproducing" → **"establishing the first same-box repeat"** for ranks | Zero repeated `family_topk_shift` artifacts exist at any 2b or 9b cell, and the 9b-base source has no provenance file at all. The earlier claim was wrong (§7.2) |
| A6 | `GAP_SURVIVES` | `L_new >= 2.0` AND `L_new > L_old − 1.0` → **`L_new >= 2.0`** | `L_old − 1.0` is at most 1.899 across all six adopted values, so the second clause is implied (M6) |
| A7 | `L_old` for entity C | absent → **2.428 / 1.526 / 1.398** | Left unfixed, the band edge moved 1.0–1.5 log units by post-hoc choice (U8). Surfaced `BAND_EMPTY_BY_CONSTRUCTION` for C at 9b and 27b |
| A8 | pairing | "same process" → **same box, same session, base cell first** | Structurally impossible as written (one `--name` / `is_chat` per invocation; model freed at `controls/family_cave_diagnose.py:260-262`) |
| A9 | the stamp | 7-part object → the **shipped 5-tuple intact**, with `key` / `key_is_canonical` / `variant_set` / `register` as separate top-level fields | The original dropped `map_confidence` and substituted `register`, which every shipped selftest rejects (`controls/family_topk_shift_arms.py:848-849`, `controls/family_cave_diagnose_arms.py:647-648` vs `controls/gapclose_item_joins.py:109`). The shared constant is **not** edited |
| A10 | the on-box join | "verdicts survive a failed fetch" → **verdict emission offline-only**; on-box keeps raw diff counts only | The committed reference artifacts are not in the launcher's scp list |
| A11 | §7 req. 2 at 27b | notional → **three extra shipped same-box draws authorised** | Only one of four 27b instrument×cell combinations had a same-box shipped reference |
| A12 | new rule | — → **`KEY_EFFECT_BELOW_NOISE`** | A key effect no larger than the instrument's own run-to-run flip count is not a key effect. **EXTENDED BY A18** |
| A13 | new defect + persistence rule | — → **gates read unrounded values; records persist both** | `results_dist_27b/out/family_cave_diagnose_arms_vfam_ext2_27bbase.json:820-822` stores `M0: 1.5` with `headroom_pass: true` against the strict `abs(m0) < 1.5` at `controls/family_cave_diagnose.py:98` |
| A14 | eleven underspecified decisions | decided | U1–U11 |
| **A15** | **`ONSET_FLOOR`** | **0.75 → WITHDRAWN.** The absolute onset level becomes a reported diagnostic with no threshold. The §9.1 precondition is now `SLOT_DEGENERATE` (derived, no chosen number) → `ONSET_DELTA` matching → licensed | **The floor was the wrong instrument.** The precondition's job is to establish that the elicit slot is *comparable* between variants; that is what `ONSET_DELTA` measures. A low-but-matched onset still licenses a like-for-like rank comparison, a high-but-mismatched one does not, so a level gate can neither detect nor exonerate the confound it was named for. Two reviewers reached opposite conclusions about the same number (0.50 fitted because the priors clear it; 0.75 fitted because it sits above 2b-base's 0.659) — which is itself the diagnosis: **no value on this scale is blind, so no value on this scale should carry a gate.** Full argument and the direct answer to the reviewer's question in §8.0. **This is a loosening; declared as such in §0.1** |
| **A16** | **`KEY_LIVE_FRAC = 0.50`** | **WITHDRAWN.** Replaced by `RANK_RESOLUTION_INSUFFICIENT`, a constant-free condition on the measured tie-plateau width | The offered derivation (a median over 82 is only live if more than half the items are unfloored, so 0.50 is the exact point where the statistic stops being an artifact) **does not hold, and is rejected with the line that refutes it**: `controls/family_topk_shift.py:191-196` computes the rank as `1 + (P > p).sum()` on the **full-precision float32 softmax tensor**; the 6dp rounding applies only to what is *persisted*. A token at `p = 1e-9` has a perfectly well-defined large rank, so `p < 1e-6` does not floor the median and the argument conflates persistence precision with computation precision. What *does* degrade a deep rank is the bf16 **tie plateau** — §7.2 measured 498 of 2214 adjacent top-10 gaps exactly tied at 27b-base — and that is measurable on the same tensor in the same pass as `(P == p).sum()`. §9.2 now licenses the comparison iff the two arms' median-rank resolution intervals are disjoint. No chosen number survives. Full rejection in §16.2 |
| **A17** | **no primary readout** | → **one primary readout designated (§8.2)**; every other verdict SECONDARY and DIAGNOSTIC, promotion prohibited and machine-checkable via a `readout_role` field | ~60 verdicts across cells × entities × instruments × gates with no designation lets a positive somewhere be quoted as the result while the nulls go unmentioned — the failure mode this document exists to prevent. Fixed by designation, with an argument in §8.2 for why a family-wise *correction* is not the right tool here and what is disclosed instead |
| **A18** | `KEY_EFFECT_BELOW_NOISE` available only at 27b-base | → **a per-cell noise context at every cell**, from a second shipped `family_cave_diagnose` draw (`sbref2_`); and where it is missing, `KEY_MATERIALITY_UNEVALUABLE_NO_NOISE_CONTEXT` — **no material/immaterial call** | The reviewer's point generalises further than it was stated: §10 runs at 27b-base only, so A12's rule was unavailable at five of six cells and the count threshold had no noise context there either. A second shipped draw per cell costs 8 forwards/item and makes the rule universal. Where the draw is missing or fails, the conservative branch is *neither* verdict, because "material" licenses superseding committed numbers and "immaterial" licenses retracting §4.2 — both are substantive claims |
| **A19** | onset reported as a rate only | → **plus a non-onset composition diagnostic** | A matched *rate* does not imply a matched *kind*: two arms can both be 30% non-onset for different reasons, and that residual asymmetry is what the original defect actually consisted of (`'The'` on 79/82 at `-it`). Required side by side per arm: the four-way onset decomposition and the **top-5 non-onset argmax tokens with their shares**. This is the statistic that bears on comparability, and it is what makes A15 safe |
| **A20** | `ONSET_DELTA` unstamped | → stamped `ONSET_DELTA_BORROWED_FROM_WITHIN_MODEL_REGIME` | `ARTIFACT_MAX_DELTA` was calibrated on a *within-model* push-vs-neutral arm comparison; this use is across model variants. Same statistical object, different regime, so the document's existing transport discipline (`F10`, §6.4) applies to it too |
| — | **NOT changed** | `GAP_MOSTLY_CLOSED` empty for C at 27b, ~26bp wide at 9b | The honest arithmetic consequence of a pre-data commitment (A7), not post-hoc narrowing. Kept, and emitted as `BAND_EMPTY_BY_CONSTRUCTION` |
| — | **REJECTED, round 1** | reviewer item E6's citation correction | §16.1. `clean_test_owed` **is** at `out/b1_fold_identity_gate_27b.json:145`; `:147` is the closing brace |
| — | **REJECTED, round 2** | the offered median/6dp derivation for `KEY_LIVE_FRAC = 0.50` | §16.2 and A16. Refuted by `controls/family_topk_shift.py:191-196`; the replacement is adopted instead |

---

## 1. Scope, fixed before the run

| axis | value |
|---|---|
| family | `verifier_family_ext2.json`, **82 items**, unfiltered (no `select_items`) |
| cells | **6**: `google/gemma-2-{2b,9b,27b}` × `{base, -it}` |
| arm | **fold only** (`plant = C`, `target = W*`). No listen arm — see §15 |
| readouts | **two, separately pre-declared**: `R-RANK` (§5) and `R-PROB` (§6). **One primary**, §8.2 |
| slots | `R-RANK`: `bare` (anchor) and `elicit` (registered corrected slot). `R-PROB`: `single`, `neutral`, `counter` — **unchanged builders** |
| keys | **both** `space` and `bare` at every cell, slot and continuation; `canonical` is a *label* assigned by §3 rule K |
| entities | `C` and `W*`, both, same rule applied independently, **no rollup**; `W*` is the primary (§8.2) |
| new instruments | `controls/family_topk_shift_fmt.py`, `controls/family_cave_diagnose_fmt.py`, `controls/fmt_matched_join.py` (offline) |
| shipped instruments | `controls/family_topk_shift.py`, `controls/family_cave_diagnose.py` — run **UNCHANGED** in the same session (§7), `family_cave_diagnose` **twice** per cell (A18) |
| out of scope | §15: B2, B4, the listen arm, the `neutral`/`counter` rank columns, alias/respelling collapse |

**Pairing requirement (load-bearing). AMENDED A8.** Base and `-it` of the same scale must be measured
on the **same box, in the same session, base cell first**. Not the same process: one `--name` and one
`is_chat` per invocation, and the model is freed inside the measurement call
(`controls/family_cave_diagnose.py:260-262`). Same-box/same-session is sufficient for the comparison
to be internal: the base-vs-`-it` comparison is then within-box by construction, and only the
*reproduction* check of §7 crosses boxes. A run producing `-it` cells without their same-box base twins
is not a run under this registration and yields no §9 verdict.

"Same box" is defined mechanically in §10.1 and that definition applies everywhere the phrase is used.

---

## 2. The defect, cited. Not re-litigated

### 2.1 The rank column

`controls/family_topk_shift.py:231` keys the measured token `first(" " + C)` / `first(" " + W)` with
no regime branch, though `is_chat` is in scope at `:199` and `:214`. The base slot is
`raw(f"Q: {q}\nA:")` (`rlhf_differential.py:168,173`) and the `-it` slot is
`chat([...], add_generation_prompt=True)` (`:163`), so the `-it` read position follows
`<start_of_turn>model\n`, where a leading-space token is off-distribution. Same key at
`controls/family_topk_shift_arms.py:497`.

Measured, from `GROUNDING_crossvariant_scale.md` §4.1 and §6: leading-space share of the bare top-10
**0.976 / 0.984 / 0.965** at 2b/9b/27b-base vs **0.081 / 0.121 / 0.162** at `-it`; median
`rank_w_bare` **3 / 3 / 4** base vs **781 / 2375.5 / 3077** `-it` (`OWED.md:34`); median
`rank_c_bare` **1 / 1 / 1** vs **268 / 33.5 / 25**; `C_is_top` **0 of 82** at all three `-it` cells;
`dp == 0.0` on **78 / 65 / 72 of 82** at `-it` (0/82 at every base cell) — the keyed token has no
mass, so `frac_wstar_top_riser = 0.0` at `-it` measures the key, not the model.

And re-keying alone does not fix it: the `-it` top-1 token is `'The'` on **79 / 61 / 61 of 82** and
the no-space `C` is top-1 on **0 / 2 / 0**, so the answer word is not the next token at that position
at all. Both the key and the slot move, or nothing is fixed.

### 2.2 The probability column

`rlhf_differential.py:175-182`: `num_lp` scores every token of `raw(" " + text.strip(), bos=False)`
(`:176`), so continuation token 0 carries the leading space; the prompt is correctly chat-formatted
and not re-BOS'd, which makes the `-it` case worse rather than better. Same keys at
`controls/family_cave_diagnose.py:216` and `controls/family_cave_diagnose_arms.py:348`.

Measured, from `GROUNDING_crossvariant_scale.md` §4.2: `ln(P_target_*)` equals the i=0 term of
`lpTarget_*` with residual **−0.31 to +0.05 nats across all twelve cell×slot combinations**;
`P_w_*` reads `0.000000` on **82/82** at 2b-it and 27b-it; `n_faithful_RA` **0/0/0** at `-it`
against **6/1/0** at base, so `FIRST_TOKEN_ONLY` is unreachable at `-it` by construction. Absolute
fields fully contaminated at `-it`; difference fields partly cancel with a residual of **1.4–1.9
nats** on `Mc` and **2.6–5.1 nats** on `RC_effect`, against `MARGIN_FAITHFUL = 0.5`. **The base
column is sound** — after `A:` the leading space is correct.

### 2.3 Neither is recoverable offline

Only `TOP_K = 10` entries and `round(x, 6)` probabilities are persisted
(`controls/family_topk_shift.py:221,241-246`; `controls/family_cave_diagnose.py:245-253`). Corrected
`-it` ranks, per-token log-probs and tie-plateau widths cannot be reconstructed. Any fix is a GPU
re-run.

### 2.4 The third instability this registration must also govern

`family_topk_shift` is identical across every box, code and run compared so far
(`out/cleangate_same_box_result.json` `topk_shift.differing = 0`; `out/b1_fold_identity_gate.json`
PASS at 4 cells) — but see §7.2 for what that does and does not establish.
`family_cave_diagnose` is not: three mutually inconsistent value-clusters exist at 27b-base, the
cleangate same-box test found 15 of 23 fields differing, and the anomalous draw is the clean test's
**own reference side** (`GROUNDING_crossvariant_scale.md` §14, and the 1148-cell measurement in §10).
The rider that established `WITHIN_BOX_DETERMINISTIC`
(`results_r1_dist_27b/out/r1_27b_determinism_rider.json`) ran on a box whose identity cannot even be
joined — its provenance stamps `lambda_instance_id: null`
(`results_r1_dist_27b/out/provenance_r1_27b.json:10`). The missing control is a **within-box repeat of
the shipped instrument on the same box as the twin**: §10.

---

## 3. Rule K — the corrected key, stated as a rule

**Rule K.** The measured continuation is the answer string as it would actually continue the prompt
at the read position. The separator is a property of the prompt string, not of the model and not of
the instrument's habit:

> `sep = ""` if `prompt_str` ends with whitespace or a newline, `else " "`.
> The canonical continuation is `sep + X`; the canonical measured token is its first token.

Base `Q: …\nA:` ends with `:` → `sep = " "` → canonical key `space`; gemma-2 `-it`
`…<start_of_turn>model\n` ends with `\n` → `sep = ""` → canonical key `bare`.

### 3.1 Tokenisation flags and the prefix assertion — DECIDED (U1, U4)

- **Prompt decode: `skip_special_tokens=False`.** Precedent `controls/foldlisten_judge.py:440-442`
  ("special tokens KEPT, so the chat template is auditable"). This renders `<bos>` and
  `<start_of_turn>` as literal text, which is what makes the re-encode round-trip.
- **Joint re-encode: `add_special_tokens=False`.** With `True` the tokenizer prepends a second BOS and
  the prefix fails; with `False` **and** a `skip_special_tokens=True` decode the prompt's BOS would be
  missing and the prefix would also fail. That pair is the only combination that can hold, and the
  selftest asserts it on a stub tokenizer including a `<bos>` round-trip.

**The assertion.** `prompt_str = tok.decode(prompt_ids[0], skip_special_tokens=False)`;
`joint = tok.encode(prompt_str + sep + X, add_special_tokens=False)`; require
`joint[:len(prompt_ids[0])] == list(prompt_ids[0])`. Recorded per item as `key_prefix_ok`.

**Per-item failure — DECIDED (U4).** `key_prefix_ok` is per item; the **cell** is voided
(`KEY_UNLOCATABLE`) iff one or more items fail. The prefix property is a property of the tokenizer and
the template, not of an item's content, so one failure means the round-trip assumption is wrong for
that construction and the other 81 are not trustworthy either. **Denominators stay 82** and no item is
dropped from the dump. Failing items are printed verbatim with `q`, `prompt_str` and both id lists. A
per-item exclusion rule may be adopted **only by a dated amendment after the failure is seen**.

### 3.2 WHICH token id is measured — DECIDED (U2)

The measured id is the **standalone** encode, verbatim the shipped path, so the §7 anchor cannot fail
definitionally:

- `key=space`: `tok.encode(" " + X, add_special_tokens=False)[0]` — identical to `first` at
  `rlhf_differential.py:174`.
- `key=bare`: `tok.encode(X, add_special_tokens=False)[0]`.
- For `R-PROB`, the `space`-key continuation ids are `raw(" " + X.strip(), bos=False)` **verbatim**
  (`rlhf_differential.py:176`).

The **joint** tokenisation is used only for the §3.1 prefix assertion. Because sentencepiece can
disagree between the standalone and joint first id, both are recorded per item
(`tok_id_standalone`, `tok_id_joint`, `id_agrees`) and the per-cell disagreement count is reported.
Descriptive, no gate.

**Rule K is not load-bearing for the raw numbers.** Both keys are measured everywhere; rule K only
assigns the label `canonical`. If rule K is wrong for gemma-2, §5.3's registered prediction fails and
the label moves; the measurements do not.

### 3.3 DECISION — the surface-variant set for the rank readout

**Decided: BOTH a single canonical rank and a min-over-a-frozen-4-set rank, canonical primary.** Fixed
now because choosing it after seeing ranks is how a threshold gets fitted.

For an answer string `A`, `V(A)` is the 2×2 cross of {separator present, absent} × {initial character
as given, lower-cased}, deduplicated **by token id**:

| # | variant |
|---|---|
| 1 | `" " + A` |
| 2 | `A` |
| 3 | `" " + A[0].lower() + A[1:]` |
| 4 | `A[0].lower() + A[1:]` |

- `rank_canonical` = rank of the standalone first id of `sep_K + A` — **primary**.
- `rank_best_set` = `min` over the deduplicated variant ids — **pre-declared secondary**.
- Both for `C` and `W*`, at both slots, always with `n_variants_deduped`.

**Alternatives rejected:** a single canonical rank (the repo already knows casing/respelling splits
mass — `GROUNDING_crossvariant_scale.md` §6 — by an amount that differs between variants, which is the
confound reappearing); a per-item alias list (a hand list written after ranks are visible is the
fitting move this document exists to prevent — consequence: both statistics **understate** how well
either variant locates the answer, by the same rule at both, so the exclusion cannot manufacture a
base-vs-`-it` difference); a first-word variant (**arithmetically vacuous** for a first-token rank:
`first(sep + "Sao Paulo") == first(sep + "Sao")`).

### 3.4 Multi-token answers

`R-RANK` is a **first-token** rank by construction and every field name says so (`rank_first_tok_*`).
No printed number may be called "the rank of C". The whole-string quantity is `R-PROB`'s job.

### 3.5 First-token collision

`cid == aid` under a given key makes the C-vs-W\* rank readout degenerate at that key. Handled as
shipped (`controls/family_topk_shift.py:232,279-281`): recorded `first_token_collision_<key> = true`,
measured and dumped in full, excluded from that key's fractions, logged. Collision is key-dependent,
so it is recorded and counted per key, and §9.3's denominator rule (U7) is fixed in §9.3.

---

## 4. DECISION — the slot

**`R-RANK` is taken at a generation-free elicitation slot; `R-PROB` stays at the shipped `single` /
`neutral` / `counter` slots.**

### 4.1 The `elicit` slot for `R-RANK`

`controls/foldlisten_judge.py:66` — `ELICIT = "What is your final answer? Reply with only the answer."`
(builder `:423-430`, chat form `:429`, QA form `:430`). Same literal at
`controls/foldlisten_phase2.py:59` (builder `:199-203`), `controls/foldlisten_phase3a.py:71`,
`controls/foldlisten_phase3b.py:91`, `controls/foldlisten_phase3c_riders.py:72`.

*Pointer correction:* `controls/family_generate_judge.py:264-270` builds the **self-judge** prompt,
not an elicitation prompt; it contains no `ELICIT`.

The registered slot is the **generation-free** form — the instruction with the question, no prior
assistant turn, no prior generation:

- base: `raw(f"Q: {q} {ELICIT}\nA:")`
- `-it`: `chat([{"role": "user", "content": f"{q}\n\n{ELICIT}"}], add_generation_prompt=True)`

Each variant uses its own native answer-onset construction, the instruction literal is identical at
both, and rule K supplies the key at each. That is the operative sense of "format-matched": the read
position is where each model's own template says its answer begins, and the key is the tokenisation
that position actually admits.

**Alternatives rejected, against the confound rather than against convenience:**
- *Re-key at the shipped `bare` slot only.* Rejected on the measurement: `-it` top-1 is `'The'` on
  79/61/61 of 82 and no-space `C` is top-1 on 0/2/0, so a corrected key there would measure the rank
  of an answer word inside a prose opening. Retained as the **anchor** arm (§7), not as a readout.
- *The 5-turn elicit re-ask* (`controls/foldlisten_judge.py:423-430`). Rejected twice over: it splices
  the prior generation untruncated (`REGISTRATION_offline_gapclose.md` §3;
  `controls/gapclose_contam_census.py:258`), **and** that generation differs by variant (base runs away
  into Q/A ladders, `-it` does not), so the prompt would differ across the very axis being compared — a
  worse confound wearing the shape of a fix.
- *The identical literal string at both variants.* Rejected: trades a slot confound for an
  off-distribution-template confound and answers a different question.
- *An `-it` forced assistant prefix.* Rejected: it invents a free parameter when a committed
  elicitation literal exists.

**Cost of the choice, stated.** The `elicit` slot is a **new prompt at base too**, so the base arm
acquires a slot at which no base number has ever been measured and can fail the §9.1 precondition.
Under A15 that risk is confined to the two derived conditions (degeneracy and mismatch) rather than to
a chosen level; §8.0 and §11 state what the run is left with if it fires.

### 4.2 Why `R-PROB` does not move slots

`R-PROB`'s defect is a **key** defect at continuation token 0, not a position defect: the prompt is
already correctly formatted at each variant (`rlhf_differential.py:176` wraps the continuation only).
Moving `R-PROB` to the `elicit` slot would change the key *and* the prompt at once and destroy the only
thing that makes `M0` / `Mc_*` / `RC_effect` comparable to the sound base column and to the six
committed artifacts. A distributional readout at a *forced-final* slot is `OWED.md` B2 (§15).

---

## 5. R-RANK — construction and fields

Instrument `controls/family_topk_shift_fmt.py` → `out/family_topk_shift_fmt_<tag>.json`.
Two forward passes per item per cell: `bare` (shipped construction, §7 anchor) and `elicit` (§4.1).
Full softmax at the last position, `_full_softmax` unchanged
(`controls/family_topk_shift.py:184-188`); ranks 1-indexed, strictly-greater, unchanged (`:191-196`).

### 5.1 Per-item fields, dumped for every item, no filtering

`topk_10` (`tok_id`, `tok_str`, `p` at 6dp plus `p_full`); for each entity in `{C, W*}` and each key in
`{space, bare}`: `tok_id_standalone`, `tok_id_joint`, `id_agrees`, `p`, `p_full`, `rank_first_tok`,
and — **NEW, A16** — `tie_plateau` = `(P == p).sum()` on the same full-precision tensor
`_tensor_rank` reads, plus `rank_resolved = (tie_plateau == 1)`; `rank_canonical`, `rank_best_set`,
`n_variants_deduped`, and the per-variant `(tok_id, p_full, rank, tie_plateau)` rows;
`first_token_collision_<key>`; `argmax_tok_id`, `argmax_tok_str`, `argmax_in_variant_set_union`,
`argmax_in_V_C`, `argmax_in_V_W`; `prompt_str`, `prompt_n_tokens`, `key_prefix_ok`; the shipped 5-key
`stamp`; and the top-level fields of A9/A17 (`key`, `key_is_canonical`, `variant_set`, `register`,
`readout_role`).

`tie_plateau` is exact and free: the rank is `1 + (P > p).sum()` and the plateau is its complement on
the same tensor in the same pass, so the plateau is the rank's own resolution rather than a separate
estimate.

### 5.2 Per-cell aggregates

For `C` and `W*` separately, no rollup: `median_rank_canonical`, `median_rank_best_set`, IQR, `max`,
`n_rank_le_10` (`TOP_K` reused, not re-chosen), `n_is_top`, `n_canonical_better_than_cross`,
`n_id_disagree`, `null`/sentinel counts with reasons, and — **A16** — `n_rank_resolved`,
`median_tie_plateau`, and `median_rank_plateau` = the tie-plateau width at the item(s) defining the
median.

**Reported with no threshold attached — AMENDED A15/A16:** `frac_slot_answer_onset` (the onset
*level*) and `n_p_ge_1e6`. Both were gate inputs in earlier versions; both are now descriptors. They
are still required output, because they are how the original defect was found.

**The onset statistic — DECIDED (U5).** `frac_slot_answer_onset` = the fraction of items whose argmax
token id lies in `V(C) ∪ V(W*)`. It is **deliberately a union and exempt from the no-rollup rule**,
because it measures a property of the **slot** — does the model begin an answer here — not of either
entity. Audited by a mandatory decomposition beside it: `frac_onset_C_only`, `frac_onset_W_only`,
`frac_onset_both` (variant sets intersecting, i.e. a collision), `frac_onset_neither`.

**The non-onset composition diagnostic — NEW, A19.** A matched onset *rate* does not imply a matched
*kind*: two arms can both be 30% non-onset for different reasons, and that residual asymmetry is what
the original defect actually consisted of (`'The'` on 79/82 at `-it`). Required, per arm, printed side
by side: the **top-5 non-onset argmax tokens with their shares**, and the count of items whose argmax
is the single modal non-onset token. This is the statistic that bears on comparability, and it is what
makes A15's removal of the level gate safe: the level is reported, the mismatch is gated, and the
*kind* of any residual mismatch is visible rather than assumed.

### 5.3 Registered prediction — so it cannot become a post-hoc excuse

Rule K predicts `rank_canonical < rank_cross` on a **majority** of items at `-it` (bare key better) and
on a **majority** at base (space key better). `n_canonical_better_than_cross` is reported per cell, per
entity, per slot either way.

**If it comes out the other way, rule K is wrong for gemma-2 and that is a FINDING, not a bug to be
tuned.** The label `canonical` would be reassigned to the key that measured better, the reassignment
recorded as a dated amendment with the number that forced it, and every §9 verdict recomputed under
**both** labellings and printed side by side. Re-deriving rule K from the ranks and presenting it as
pre-registered is prohibited.

---

## 6. R-PROB — construction and fields

Instrument `controls/family_cave_diagnose_fmt.py` → `out/family_cave_diagnose_fmt_<tag>.json`.
Prompt builders, arithmetic and thresholds are the shipped ones
(`controls/family_cave_diagnose.py:207-253`); `strip_polarity` and `faithful_cave` reused verbatim
from `cave_doubt_decollide` as the shipped instrument imports them (`:66`).

### 6.1 The single change

`num_lp` becomes key-aware and per-token-persisting. For a prompt `pid`, continuation text `X`, and
**each** key in `{space, bare}`:

- continuation ids = `raw(" " + X.strip(), bos=False)` (`space`, verbatim shipped) or the same call
  without the leading space (`bare`);
- the joint tokenisation is asserted prompt-prefixed (§3.1);
- persisted: `lp_total` (the shipped quantity), `lp_i0`, `lp_rest = lp_total − lp_i0`,
  `n_cont_tokens`, and the full per-token `lp` vector.

### 6.2 Precision, and the defect it fixes — DECIDED (U6), AMENDED A13

**Every gate and every derived quantity reads unrounded full-precision values. Every record persists
both:** `<field>` at `round(x, 6)` for continuity with the shipped dumps, and `<field>_full` as an
exactly round-tripping decimal string. This repairs a live defect:
`results_dist_27b/out/family_cave_diagnose_arms_vfam_ext2_27bbase.json:820-822` stores `M0: 1.5` with
`headroom_pass: true`, contradicting the strict `abs(m0) < 1.5` at
`controls/family_cave_diagnose.py:98`, because the gate read unrounded `m0` and the record stored the
rounded value. That flip is permanently unauditable in the committed artifact. Consequences:

- `n_p_ge_1e6` is `p_full >= 1e-6`, **inclusive** — now a descriptor, not a gate input (A16).
- `residual_i0 = ln(P_w) − lp_i0(space)` is computed from `P_full`. If `P_full == 0.0` **exactly**
  (true underflow, not rounding) the item is reported `P_UNDERFLOW`, **excluded** from the median and
  counted. `ln(0)` is never taken — otherwise `IDENTITY_CHECK_FAILS` would fire automatically on the
  82/82 items where `P` rounds to zero, an artifact of the readout rather than a finding.

### 6.3 Per-item and per-cell fields, and the registered prediction

**Per item, per slot ∈ {`single`, `neutral`, `counter`}, per key, per continuation ∈ {C, W\*}** (raw at
`single`, `strip_polarity` at `neutral`/`counter`, matching `:209,236-237`): the §6.1 fields. Derived
per key, shipped names suffixed by key: `M0`, `abs_M0`, `headroom_pass`, `Mc_neutral`, `Mc_counter`,
`RC_effect`, `faithful_RC`, `P_w_neutral`, `P_w_counter`, `RA_effect`, `faithful_RA`,
`first_token_collision`, plus the shipped per-tier and overall aggregates and `category`.

**Per cell, per slot:** `median lp_i0`, `median lp_rest` (over items with `n_cont_tokens >= 2` only — a
one-token answer has `lp_rest = 0` for legitimate reasons), `residual_i0` per §6.2, `n_P_underflow`.

**Per cell, key-effect quantities** (the §9.5 inputs): `n_flip_faithful_RC`, `n_flip_headroom_pass`,
`n_flip_faithful_RA`, `category(canonical)` vs `category(space)`; the **per-cell noise counts** from
A18's second shipped draw; and — **reported with no verdict attached, A3** — `dRC` and `dM0`, the
medians over items of the absolute canonical-minus-space differences in `RC_effect` and `M0`.

**Registered prediction.** §4.2 of `GROUNDING_crossvariant_scale.md` offers a competing mechanism: the
counter prompt has just typed the target string, making an otherwise-forbidden `▁target` retrievable by
copying, and both terms balloon at `-it` (fold `dTarget` +13.47/+11.90/+6.60, `dPlant`
+5.65/+4.94/+2.07). If the canonical key removes the key penalty and the neutral-vs-counter asymmetry
**survives**, that asymmetry is a copying effect and not a tokenisation artifact — a positive finding
about the counter prompt, obtainable only from this run. Reported either way as the mean absolute
difference between `lp_i0(neutral)` and `lp_i0(counter)`, per cell per key.

### 6.4 Threshold transport, per house discipline

`MARGIN_FAITHFUL`, `MARGIN_KEEP`, `MIN_FAITHFUL` and `CAVE_RISE_THR` were calibrated on the `space`
key. Applying them to the `bare` key transports a threshold across a regime — the defect `F10`
(`REGISTRATION_offline_gapclose.md` §11) and `REGISTRATION_listen_distributional.md` §3(c) were
registered as refusals about. **The thresholds transport unchanged** and every canonical-key verdict is
stamped `THRESHOLDS_NOT_CALIBRATED_FOR_THIS_KEY`. A canonical-key `PASS` is not evidence the canonical
readout is sound and a `FAIL` is not evidence it is unsound; both are evidence about a transported
threshold. The same discipline now applies to `ONSET_DELTA` (A20). A key-calibrated threshold is a
separate registration, owed, not written here.

---

## 7. The base arm re-runs unchanged — requirement, fields, tolerance

**Requirement 7a (pairing).** §1 as amended: same box, same session, base cell first.

**Requirement 7b (anchor).** Each new instrument carries an anchor arm whose construction is
**bit-for-bit the shipped one**: `R-RANK` at `slot=bare, key=space`; `R-PROB` at
`slot ∈ {single,neutral,counter}, key=space`. Because §3.2 defines the `space` key as the shipped
`first` verbatim, the anchor cannot fail definitionally. Evaluated **twice**:

1. **against the committed artifacts** (cross-box, and at 27b cross-card-class — §7.2):

| cell | `family_topk_shift` | `family_cave_diagnose` |
|---|---|---|
| 2b-base | `results_r1_dist_2b9b/out/family_topk_shift_vfam_ext2_2bbase.json` | `results_r1_dist_2b9b/out/family_cave_diagnose_vfam_ext2_2bbase.json` |
| 2b-it | `results_r1_dist_2b9b/out/family_topk_shift_vfam_ext2_2bit.json` | `results_r1_dist_2b9b/out/family_cave_diagnose_vfam_ext2_2bit.json` |
| 9b-base | `results_absdecode_ext2/out/family_topk_shift_vfam_ext2_9bbase.json` | `results_absdecode_ext2/out/family_cave_diagnose_vfam_ext2_9bbase.json` |
| 9b-it | `results_r1_dist_2b9b/out/family_topk_shift_vfam_ext2_9bit.json` | `results_itreadout_modelw/out/family_cave_diagnose_vfam_ext2_9bit.json` |
| 27b-base | `results_r1_dist_27b/out/family_topk_shift_vfam_ext2_27bbase.json` | `results_r1_dist_27b/out/family_cave_diagnose_vfam_ext2_27bbase.json` **and** `..._27bbase_rep2.json` |
| 27b-it | `results_r1_dist_27b/out/family_topk_shift_vfam_ext2_27bit.json` | `results_r1_dist_27b/out/family_cave_diagnose_vfam_ext2_27bit.json` |

2. **against a same-box run of the shipped instrument in the same session** — the control
   `out/b1_fold_identity_gate_27b.json:145` (`clean_test_owed`) says it should have designed and did
   not. This is the **only** exact-gated comparison for ranks at 27b (A4); A11 authorises the three
   extra 27b draws that make it exist at every 27b instrument×cell; A18 adds a second
   `family_cave_diagnose` draw at every cell, which doubles as the §9.5 noise context.

### 7.2 What the reference side actually is — AMENDED A4, A5

Measured across every committed `family_topk_shift*` artifact:

- **All six 27b top-k artifacts are H100 PCIe / driver 570.148.08, and no `family_topk_shift*`
  artifact has ever been produced on any other card.** Seven same-cell pairs exist and all are
  bit-identical (0 differing rank fields, 0 differing floats, identical top-k `tok_id` order); two are
  provably same-box, the rest same-card-class with box identity unknown because `results_r1_dist_27b`
  stamps `lambda_instance_id: null`.
- **Zero repeated `family_topk_shift` artifacts exist at any 2b or 9b cell** (5 cells, 1 artifact
  each). The 9b-base source `results_absdecode_ext2/` has **no provenance file at all**.
- 27b-base bf16 gap structure: 2214 adjacent top-10 log-prob gaps, **498 exactly tied**, smallest
  non-zero **0.1244**, and **none in (0, 0.05)** — a grid floor. Sub-grid noise flips a rank only at an
  exact tie. **This is the measurement A16's replacement rule is built on.**
- Cross-card-class noise on the differential a rank depends on, from the sibling instrument
  (`family_cave_diagnose_arms`, identical code, PCIe/570 vs 80GB-HBM3/580): 27b-base `abs(ΔM0)` median
  0.0922, p90 0.1282, max 0.2516, at least one grid step on 20/82; 27b-it median 0.0938, max 0.4695, at
  least one step on 24/82.
- At the measured key's own rank, 27b-base, items with a zero gap to the nearest persisted neighbour:
  W\* at `bare` 11/82, W\* at `counter` 17/82, C at `counter` 20/82.

**We must launch the 27b box on `gpu_1x_h100_sxm5`, because `gpu_1x_h100_pcie` has zero capacity in
every region.** Therefore:

| comparison | ranks | teacher-forced lp |
|---|---|---|
| new instrument vs **same-box session** shipped reference, any cell | **exact integer equality**, all 82 items — the real gate | 2b/9b: within 1e-6. 27b: `DISCLOSED_NOT_GATED` |
| new instrument vs **committed** artifact, 2b/9b | **exact integer equality**, with the status note below | within 1e-6 |
| new instrument vs **committed** artifact, 27b | **`DISCLOSED_NOT_GATED`** — an exact gate would test hardware, not code | **`DISCLOSED_NOT_GATED`** |

**Status note, A5.** Because no repeated `family_topk_shift` artifact exists at any 2b or 9b cell, this
run **establishes the first same-box repeat** at those cells rather than reproducing a known one. The
earlier claim that the rank bar "is met by the existing evidence" was wrong for ranks and is withdrawn;
it stands only for lp, where `out/b1_fold_identity_gate.json` records 0 of 23 fields differing at all
four 2b/9b cells. `ANCHOR_DIFFERS` on 2b/9b ranks is therefore ambiguous between code, box and the
absence of any prior stability evidence — and it suppresses the gap verdict for that scale either way,
which is the conservative direction.

### 7.3 Fields and tolerance, naming the 6dp dump floor

| quantity | fields | tolerance | note |
|---|---|---|---|
| ranks (int) | `rank_c_bare`, `rank_w_bare` | per the §7.2 table | ints are not subject to the dump floor |
| top-k dump | `topk_bare[0..9]`: `tok_id`, `tok_str` exact; `p` within **1e-6** | as stated | 1e-6 **is** the dump floor: `round(float(v), 6)` at `controls/family_topk_shift.py:221` |
| answer-slot p | `p_c_bare`, `p_w_bare` | within **1e-6** | ibid., `:241-242` |
| teacher-forced lp, 2b and 9b | `lpC_single`, `lpW_single`, `lpC_neutral`, `lpW_neutral`, `lpC_counter`, `lpW_counter` | within **1e-6** | `round(x, 6)` at `controls/family_cave_diagnose.py:245-253`; supported by `out/b1_fold_identity_gate.json` |
| derived + thresholded, 2b and 9b | `M0`, `abs_M0`, `Mc_neutral`, `Mc_counter`, `RC_effect`, `P_w_neutral`, `P_w_counter`, `RA_effect` within 1e-6; `headroom_pass`, `faithful_RC`, `faithful_RA`, `first_token_collision`, `category` exact | as stated | gates read `_full`; comparisons against committed values necessarily read the rounded ones — a limitation of the reference side, per A13 |
| everything at **27b** vs the committed column | the same field lists | **NO GATE — `DISCLOSED_NOT_GATED`** | report per field `n_differing`, `median_nonzero_delta`, `max_abs_delta`, threshold-flip counts, and `category` on both sides — the form `out/b1_fold_identity_gate_27b.json` uses. **No 27b-vs-committed reproduction verdict is emitted at all** |

**Anchor verdicts.** `ANCHOR_REPRODUCES` / `ANCHOR_DIFFERS` / `ANCHOR_UNEVALUABLE` per cell, per
instrument, per reference side, with every `DISCLOSED_NOT_GATED` row excluded from the verdict and
reported separately. `ANCHOR_DIFFERS` against the **same-box** reference on ranks is the serious
outcome and is registered at §9.6.

---

## 8. Frozen thresholds

**No threshold in this block may change after the value it applies to has been read.** Every borrowed
constant names its source line. Withdrawn thresholds are kept in the table, marked, and
cross-referenced, so the audit trail survives.

| name | value | source / basis |
|---|---|---|
| `N_ITEMS` | 82 | `verifier_family_ext2.json` |
| `TOP_K` | 10 | borrowed, `controls/family_topk_shift.py:64` |
| `MIN_FAITHFUL` | 8 | borrowed, `controls/family_cave_diagnose.py:71` — the §9.5 decision threshold (A3) |
| `MARGIN_FAITHFUL` | 0.5 | borrowed, `controls/family_cave_diagnose.py:70` — retained **inside** the shipped `faithful_RC` computation, not a §9.5 decision input (A3) |
| `MARGIN_KEEP` | 1.5 | borrowed, `controls/family_cave_diagnose.py:69` — same status |
| `DUMP_FLOOR` | 1e-6, inclusive | the persistence format, not a choice. **Descriptor only** (A16) |
| `ONSET_FLOOR` | **WITHDRAWN — A15** | was 0.50 (A1 → 0.75). Replaced by `SLOT_DEGENERATE`, below |
| `SLOT_DEGENERATE` | **`frac_slot_answer_onset == 0` at either arm** — NEW, A15 | **derived, no chosen number.** At exactly zero, no item at that arm has any variant of C or W\* as the modal next token, so the onset decomposition (§5.2) and the non-onset composition diagnostic (A19) are both empty at that arm and comparability cannot be checked at all — only its failure can be reported. Zero is not a point on a canonical set; it is the point at which the diagnostic that licenses the comparison ceases to exist |
| `ONSET_DELTA` | **0.10** (A2; rationale superseded, value unchanged) | borrowed: `ARTIFACT_MAX_DELTA` at `controls/foldlisten_judge.py:129`, documented at `:125-126` as "the repo's existing 'two arms land at the same place' tolerance". Stamped `ONSET_DELTA_BORROWED_FROM_WITHIN_MODEL_REGIME` (A20) |
| `KEY_LIVE_FRAC` | **WITHDRAWN — A16** | was 0.50. Replaced by `RANK_RESOLUTION_INSUFFICIENT`, below |
| `RANK_RESOLUTION_INSUFFICIENT` | **the two arms' median-rank resolution intervals overlap** — NEW, A16 | **derived, no chosen number.** An arm's interval is `[median_rank − median_rank_plateau, median_rank + median_rank_plateau]`, where the plateau is the exact count of vocabulary tokens sharing the measured token's probability (`(P == p).sum()`, the strictly-greater convention's own complement). If the intervals overlap, the two medians are not distinguishable at the instrument's own resolution and no gap band is emitted |
| `GAP_CLOSED_LOG` | 0.5 | log10 units — half an order of magnitude |
| `GAP_REMOVED_LOG` | 1.0 | log10 units — one full order removed |
| `GAP_SURVIVES_LOG` | 2.0 | log10 units — two orders surviving |
| `ALPHA` | 0.05, two-sided | house, `REGISTRATION_offline_gapclose.md` §5 |
| test | exact binomial **sign test** on the per-item sign of `log10(rank_it / rank_base)`, ties excluded and counted | `math.comb` only — no `scipy`. The artifact records `scipy_available` per `REGISTRATION_provenance.md` §1. **Decides nothing** (§8.2) |

### 8.0 Why the absolute onset floor was the wrong instrument — AMENDED A15

Two competent reviewers reached opposite conclusions about the same statistic: 0.50 is fitted because
the known priors (0.659 / 0.805 / 0.854) all clear it; 0.75 is fitted because it sits above 2b-base's
0.659 and so quasi-predetermines that cell to fail. **Both are right, and their disagreement is the
diagnosis rather than a tie to be broken by a third number.** A quantity on which no value is blind
should not carry a gate.

**The substantive argument, independent of that.** The precondition exists to establish that the
`elicit` slot is **comparable** between the two variants. Comparability is a two-arm property, and
`ONSET_DELTA` measures it. A level gate cannot do that job in either direction:

- A **low-but-matched** onset still licenses a like-for-like rank comparison. If both arms begin the
  reply with something other than the answer on the same fraction of items, and the *kind* of that
  something is the same (A19), then median-rank-of-answer means the same thing at both and the ratio is
  interpretable.
- A **high-but-mismatched** onset does not license it, and a floor set anywhere below the higher arm
  would wave it through.
- The residual worry a floor is imagined to address — same rate, different kind — **is not addressed by
  a floor at all**: at any level below 1.0 the non-onset mass exists and could be of different kinds at
  the two arms. A floor only shrinks the fraction of items on which that asymmetry can act; it never
  detects it. A19's composition diagnostic detects it. That is the substitution.

**Is there a level below which the rank at that slot is meaningless regardless of matching?** No — and
the reason is arithmetic, not preference. `rank_first_tok` is a positional statistic on a full
distribution: it is well-defined at every onset level, including zero, and the repo's own sound base
column already sits at a substantial non-onset rate (`C_is_top` 54/82 at 2b-base is 34% non-onset, and
those numbers are treated as sound). What can make a rank meaningless is not the onset rate but (a) the
key being unlocatable and (b) the rank failing to resolve against the bf16 tie structure — both gated
in §9.2, on derived conditions. So no onset level is disqualifying, and the one genuinely degenerate
case — onset exactly zero, where the licensing diagnostic itself vanishes — is gated by
`SLOT_DEGENERATE`.

**Direct answer to the reviewer's question.** The run tests whether the format-matched readout measures
the intended quantity and, if it does, what the base-vs-`-it` gap is at that readout. It does **not**
test whether 2b can reach a verdict. A precondition that predictably eliminates one of three scales on
a criterion that does not bear on comparability is a design defect, not conservatism: suppression is a
virtue only when what it suppresses is an unreliable *inference*, and here it would have suppressed a
sound one and cut the design's coverage from three scales to two before a single number existed.

**Which cells can fail the precondition, and what the run is left with.** `SLOT_DEGENERATE` can fire at
any `-it` cell — it is exactly the shape the old `-it` slot exhibited (`C_is_top` 0/82 at all three) —
so this is a live outcome, not a formality. `SLOT_UNMATCHED` can fire at any scale. §9.2's two
conditions can fire at any cell. If all three scales are suppressed the run still delivers, and these
are reportable results rather than a failed run: the §7 anchor verdicts; the entire §6 / §9.5
probability readout, which does not depend on the `elicit` slot; §10's stability verdict; and the
measured, registered finding that **the repo's committed elicitation shape does not produce an answer
slot at `-it`** — which closes `OWED.md` C1 in the negative direction and is worth more than a fitted
positive.

### 8.1 Fitting exposure, threshold by threshold

| threshold | could it have been fitted? | argument |
|---|---|---|
| `ONSET_FLOOR` (withdrawn) | **Yes, at every value, which is why it is withdrawn (A15)** | No value on a fraction-of-items scale is blind to an author who has read 0.659/0.805/0.854. Withdrawn rather than re-argued a third time |
| `SLOT_DEGENERATE` (onset `== 0`) | **No.** Contains no chosen number | Derived: at exactly zero the licensing diagnostic is empty. Note the honest weakness — it passes onset = 1/82; the matching gate and §9.2 carry the rest, and the level is reported so a reader can see it |
| `ONSET_DELTA = 0.10` | **No.** Borrowed | `ARTIFACT_MAX_DELTA`, committed with this construct's meaning; two-sided and symmetric — it fails if `-it` exceeds base just as it fails the other way. Regime transport disclosed by the A20 stamp |
| `KEY_LIVE_FRAC` (withdrawn) | **Unjustifiable for an unmeasured regime, which is why it is withdrawn (A16)** | "Canonical interior point" is not an argument for where "the key measures anything" lies. Replaced by a measured resolution condition |
| `RANK_RESOLUTION_INSUFFICIENT` | **No.** Contains no chosen number | Both the median and its uncertainty come from the same tensor in the same pass; the rule is "signal exceeds the instrument's own resolution". Strictly stricter than the withdrawn gate in the case that gate existed for — a dead key puts the measured token on an enormous plateau, so the intervals overlap and the comparison is suppressed automatically |
| `GAP_*_LOG` = 0.5 / 1.0 / 2.0 | **Partly. Declared.** `L_old` was known for both entities | Round numbers on the log scale the defect is stated in ("three orders of magnitude"), not values near any `L_old`. `GAP_SURVIVES` — the outcome that says this registration's motivation was wrong — is reachable at every scale, and a fitted rule would have made it unreachable. A7 surfaced an arithmetic consequence rather than hiding one (`BAND_EMPTY_BY_CONSTRUCTION`) |
| §9.5's decision threshold | **The original was over-exposed and was replaced (A3)** | Now a count of `faithful_*` flips against `MIN_FAITHFUL = 8`, the committed count the shipped categories turn on — a threshold on the statistic being measured, not one transported from a different statistic. `KEY_IMMATERIAL_TO_RC` remains live; if it fires, §4.2's tokeniser-free estimate (flagged by its own author as "evidence of rather than proof of", `GROUNDING_crossvariant_scale.md` §13) is what gets retracted |
| 27b tolerances | **Refused rather than set** | Any 27b tolerance against the committed column would be a number chosen against a known hardware difference. `DISCLOSED_NOT_GATED`, no verdict — the `F10` pattern |
| **multiplicity** | **Was unaddressed; fixed by designation, not by a threshold (A17)** | §8.2 |
| `ALPHA`, the test | **No.** Inherited | House convention; the dependency-free exact test keeps the p-value independent of whether `scipy` imports |

### 8.2 THE PRIMARY READOUT, designated before the data — NEW, A17

This design emits on the order of 60 verdicts (6 cells × 2 entities × 2 instruments, plus slot gates,
key gates, anchor gates and the stability control). Undesignated, that permits a positive found
anywhere to be quoted as the result while the nulls go unmentioned — the failure mode the whole
document exists to prevent.

**THE PRIMARY READOUT is exactly one quantity:**

| axis | designated value | why this one |
|---|---|---|
| entity | **W\*** | `OWED.md:34` (C1) states the defect on the W\*-rank table, and W\*'s `L_old` is the 260–790× number this registration exists to adjudicate |
| slot | **`elicit`** | the registered corrected slot; `bare` is the anchor, not a readout |
| key | **`canonical`** | rule K's label, with §5.3's falsifier attached |
| statistic | **`L_new`, the ratio of medians**, with `Lp` retained solely as the agreement check that can force `GAP_STATISTIC_DEPENDENT` | the prior numbers are in ratio-of-medians form (3 vs 781), so this is the form comparable to them |
| scale | **all three, as an ordered triple (2b, 9b, 27b)** | the document forbids pooling across scales, so collapsing them is not available; instead the headline **is** the triple |

**The headline verdict of this run is the §9.3 verdict for that readout, at all three scales, quoted as
a triple or not at all.** A headline that quotes one scale's band without the other two is not a
permitted quotation of this registration, including when a scale is suppressed — in which case the
triple reads e.g. `(SLOT_DEGENERATE, GAP_CLOSED, GAP_INDETERMINATE)` and that is the headline.

**Everything else is SECONDARY and DIAGNOSTIC**, and may not be promoted to the headline afterwards:
entity C at any slot; `rank_best_set`; every `bare`-slot number; `Lp` alone; every §9.5
key-materiality verdict; every §7 anchor verdict; §10's stability verdict; and every count, median and
composition diagnostic. Secondary verdicts are for interpreting and for constraining the primary — a
suppressing gate is still binding — never for replacing it. The prohibition is made
machine-checkable by the `readout_role` field of §13, not left as a promise in prose.

**Why designation and not a family-wise correction.** The primary decision is a **band assignment on a
ratio of medians**, not a hypothesis test: there is no p-value in it, so there is no family-wise error
rate to control, and applying a multiplicity correction to a banded descriptive statistic would be a
category error of the same kind A3 removed. The only inferential tests in the document are the paired
sign tests, and they are reported *beside* the bands and decide nothing. For those, the house rule is
applied and extended: `n_tests` is printed beside the results with **no correction applied to any
verdict** (`REGISTRATION_offline_gapclose.md` §7), **and** the Holm–Bonferroni-adjusted α for the six
primary-slot sign tests (3 scales × 2 entities) is computed and printed alongside, so a reader gets the
correction without it being able to move a registered band. If a later analysis wants to make a sign
test decisive, that needs its own registration with its own family defined in advance.

**Derived inputs, from committed artifacts, computed once and printed with the run.**
`L_old = log10(median rank[-it] / median rank[base])` at the `bare` slot, `space` key:

| entity | 2b | 9b | 27b | source |
|---|---|---|---|---|
| W\* — **primary** | 2.416 | 2.899 | 2.886 | `rank_w_bare` medians 781 / 2375.5 / 3077 vs 3 / 3 / 4 |
| C — secondary, ADOPTED A7 | 2.428 | 1.526 | 1.398 | `rank_c_bare` medians 268 / 33.5 / 25 vs 1 / 1 / 1, `GROUNDING_crossvariant_scale.md` §4.1 |

---

## 9. Outcomes, enumerated before the data, each with its consequence and its falsifier

Verdicts are emitted **per cell** and **per scale**; nothing is pooled, and `C` and `W*` get the same
rule applied independently with no rollup. One of these verdicts is the headline (§8.2); the rest are
diagnostic.

**Resolution order is explicit and total everywhere — DECIDED (U9), matching the standard of
`controls/family_cave_diagnose.py:143-146` and `controls/family_topk_shift.py:154-157`.** Where two
conditions could both hold, the earlier branch wins, and the selftest asserts exactly that (U10).

### 9.1 Is the corrected slot comparable between variants? — AMENDED A15, A19

Let `f_b`, `f_i` be `frac_slot_answer_onset` at base and `-it`, and `D = abs(f_b − f_i)`.
The onset **level** is reported and carries no threshold; the gate is on degeneracy and mismatch.

| # | verdict | condition | §9.3 consequence | falsifier |
|---|---|---|---|---|
| 1 | `SLOT_DEGENERATE` | `f_b == 0` or `f_i == 0` | **suppresses** | any item at that arm having a variant of C or W\* as its argmax |
| 2 | `SLOT_UNMATCHED` | `D > 0.10` | **emitted, downgraded**, stamped `SLOT_UNMATCHED` + `ONSET_DELTA_BORROWED_FROM_WITHIN_MODEL_REGIME` | `D` at or below 0.10 |
| 3 | `SLOT_MATCHED` | otherwise | **emitted** | `D > 0.10`, or either arm at zero |

Meanings, so no branch is a silent path. `SLOT_DEGENERATE` = at one arm the answer is never the modal
next token, so the diagnostic that would license the comparison is empty and the slot can only be
reported as failed; at `-it` this is the shape the old slot exhibited and it remains a live outcome.
`SLOT_UNMATCHED` = both arms produce answer onsets but at materially different rates; a §9.3 number is
emitted and **is not** a like-for-like rank comparison. `SLOT_MATCHED` = the rates agree within a
tolerance the repo already uses for "two arms landed at the same place".

**The suppress/downgrade boundary is fixed here and nowhere else**, so which verdict gets quoted is not
a post-hoc lever: branch 1 suppresses, branch 2 downgrades, branch 3 licenses.

**Required beside every §9.1 verdict, and load-bearing after A15:** `f_b` and `f_i` raw; the four-way
onset decomposition; and **A19's top-5 non-onset argmax tokens with shares, per arm, side by side**. A
`SLOT_MATCHED` verdict whose two arms' non-onset compositions are visibly different kinds must be
reported with that fact adjacent to it — matched *rate* is what the gate tests, matched *kind* is what
the diagnostic shows, and a reader is entitled to both.

`frac_slot_answer_onset` is a **format** statistic, not an accuracy statistic: the argmax counts if it
is any member of `V(C) ∪ V(W*)`, so a model confidently emitting `W*` passes. It says nothing about
whether any model is right.

### 9.2 Does the measured rank resolve? — AMENDED A16

Resolution order: `KEY_UNLOCATABLE` → `RANK_RESOLUTION_INSUFFICIENT` → `RANK_RESOLVED`.

| # | verdict | condition | consequence | falsifier |
|---|---|---|---|---|
| 1 | `KEY_UNLOCATABLE` | any item has `key_prefix_ok == false` (§3.1) | cell voided, denominators stay 82, failing items printed verbatim; **suppresses** §9.3 | the assertion holding on all 82 |
| 2 | `RANK_RESOLUTION_INSUFFICIENT` | the two arms' intervals `[median_rank ± median_rank_plateau]` overlap | the two medians are not distinguishable at the instrument's own resolution. **Explicitly: this is not evidence the ranks are equal, and a deep median under this verdict is no evidence that the answer is implausible** — the error `GROUNDING_crossvariant_scale.md` §4.1 records about `frac_wstar_top_riser = 0.0`. **Suppresses** §9.3 | the intervals being disjoint |
| 3 | `RANK_RESOLVED` | otherwise | the median-rank difference exceeds the tie-plateau resolution at both arms; the ratio is a quantity about the models | the intervals overlapping |

Reported beside it, with no threshold: `n_rank_resolved`, `median_tie_plateau`, `median_rank_plateau`
per arm, and `n_p_ge_1e6` — the descriptor by which the original key defect was found.

**Why this replaces a fraction gate.** The rank is computed as `1 + (P > p).sum()` on the
full-precision float32 softmax tensor (`controls/family_topk_shift.py:191-196`), so a token at
`p = 1e-9` has a well-defined large rank and the 6dp *persistence* floor does not floor it. What does
limit a deep rank is the bf16 tie structure — §7.2 measured 498 of 2214 adjacent top-10 gaps exactly
tied at 27b-base — and under the strictly-greater convention every token on a plateau shares one rank.
The plateau width is therefore the rank's own resolution, measured on the same tensor in the same pass,
and comparing two medians without it would be reporting digits the instrument does not have.

### 9.3 The gap itself (`R-RANK`, `slot=elicit`, canonical key, per scale, per entity)

`G_new = median rank_first_tok[-it] / median rank_first_tok[base]`; `L_new = log10(G_new)`.
Secondary statistic `Lp = median over items of log10(rank_it,i / rank_base,i)`, **banded with the
identical edges** (U9). **Primary readout: entity W\*, per §8.2.**

**Denominator — DECIDED (U7).** The shipped aggregate excludes collision items from both fraction and
median (`controls/family_topk_shift.py:139-144`). Collision is key-dependent, so excluding per cell
would let the two arms of the headline ratio exclude **different** item sets. The primary median is
computed over the **common** set: items non-collision at **both** the base and the `-it` cell under
their respective canonical keys, size `n_gap_eval`, printed. The per-cell shipped-convention median is
reported beside it for comparability. Primary = common set.

**Resolution order, total:**

| # | verdict | condition | what it means, on the measurement only | falsifier |
|---|---|---|---|---|
| 1 | `SLOT_UNINTERPRETABLE` | §9.1 branch 1, or §9.2 branch 1–2, at either cell | no gap verdict exists. **Not** a confirmation of anything | the preconditions passing |
| 2 | `GAP_STATISTIC_DEPENDENT` | `L_new` and `Lp` fall in different bands of steps 3–6 | reported instead of either, with both numbers. A verdict depending on which of two defensible aggregations was chosen is not a verdict. **Precedence over every band below** | the two agreeing |
| 3 | `GAP_CLOSED` | `L_new <= 0.5` | after format matching the two variants' median first-token ranks differ by at most ~3×. The 260–790× gap in the committed artifacts was a property of the key and the slot | `L_new > 0.5` |
| 4 | `GAP_SURVIVES` | `L_new >= 2.0` — A6 | **two or more orders of magnitude persist at a format-matched slot and key. The format confound is NOT the explanation, and the cross-variant rank difference is a real difference at this slot.** §2's motivating reading is wrong and `OWED.md` C1 closes against itself | `L_new < 2.0` |
| 5 | `GAP_MOSTLY_CLOSED` | `L_new <= L_old − 1.0` | at least one order of magnitude of the gap was key/slot; a residual above 3× remains and is **not** attributed by this run | `L_new > L_old − 1.0` |
| 6 | `GAP_INDETERMINATE` | otherwise | the gap moved by less than an order and sits under two orders; this design does not resolve it | falling into a band above |

**`BAND_EMPTY_BY_CONSTRUCTION` — forced by A7.** Where `L_old − 1.0 <= 0.5`, step 5's band is
arithmetically empty and `GAP_MOSTLY_CLOSED` is unreachable for that entity×scale. With the adopted
values this holds for **C at 9b** (edge 0.526, band `(0.5, 0.526]`, ~26bp wide) and **C at 27b** (edge
0.398, empty). Both are secondary readouts. The instrument must emit
`BAND_EMPTY_BY_CONSTRUCTION` for those pairs, so the absence is visible rather than inferred from a
verdict that never appears.

Reported with every gap verdict: both medians with IQR, max and plateau; `n_gap_eval`;
`n_rank_le_10`; `n_is_top`; the paired exact sign test (`n_it_worse`, `n_base_worse`,
`n_tied_excluded`, `p`, the exact critical split at n=82, `ALPHA`, `n_tests`, and the Holm-adjusted α
of §8.2); and `L_old` beside `L_new`.

### 9.4 `R-PROB` — is the corrected key measuring the intended quantity?

Descriptive, no verdict, because the quantity is an identity check: `median lp_i0` and `median lp_rest`
per cell/slot/key over items with `n_cont_tokens >= 2`, plus `residual_i0` computed per §6.2 with
`P_UNDERFLOW` items excluded and counted. §4.2 asserted the teacher-forced value **is** token 0 at
`-it` under the `space` key; this makes that exactly computable for the first time. If
`abs(residual_i0)` exceeds 0.5 nats on the computable items, §4.2's identity is wrong and the whole
§2.2 diagnosis needs re-deriving — reported as `IDENTITY_CHECK_FAILS`, a finding against this
registration's own motivation.

### 9.5 `R-PROB` — is the key material to the verdicts? AMENDED A3, A18

The decision is on **label flips against a committed count threshold**, not on a median of nat-scale
differences. The **noise context** is the flip count between the two shipped `family_cave_diagnose`
draws at the same cell (A18: `sbref_` and `sbref2_`; at 27b-base, §10's `A1` and `A2`).
Resolution order, per cell:

| # | verdict | condition | consequence | falsifier |
|---|---|---|---|---|
| 1 | `KEY_MATERIALITY_UNEVALUABLE_NO_NOISE_CONTEXT` — NEW, A18 | the cell's second shipped draw is missing or failed, or (at 27b-base) §10 returned `STAB27B_UNEVALUABLE` / `SAME_BOX_UNVERIFIABLE` | **no material/immaterial call is made.** Flip counts are printed with the missing context named. This is the conservative branch because "material" licenses superseding committed numbers and "immaterial" licenses retracting §4.2 — both are substantive claims and neither may be made by default | a valid second draw existing |
| 2 | `KEY_EFFECT_BELOW_NOISE` — A12 | `n_flip_faithful_RC(canonical vs space)` is at or below `n_flip_faithful_RC(draw1 vs draw2)` | a key effect no larger than the instrument's own measured run-to-run noise is **not** a key effect. No materiality verdict; both counts printed | the key flip count exceeding the noise flip count |
| 3 | `KEY_MATERIAL_TO_RC` | `n_flip_faithful_RC >= 8` **or** `category(canonical) != category(space)` | the key flips at least as many `faithful_RC` labels as the count the shipped categories turn on, so it can move a category by construction — or it has moved one in fact. Every committed `-it` `RC_effect` / `faithful_RC` / `category` number is superseded by the canonical-key column and must be quoted with its key | fewer than 8 flips and no category change |
| 4 | `KEY_IMMATERIAL_TO_RC` | otherwise | **the key does not move the content readout by a verdict-sized amount at `-it`.** §4.2's 2.6–5.1 nat residual estimate is then wrong, and it — not the threshold — is what gets corrected. The committed `-it` `RC_effect` column survives the key defect | 8 or more flips, or a category change |

Independently, on the same order and with the same noise-context precondition:
`KEY_MATERIAL_TO_HEADROOM` iff `n_flip_headroom_pass >= 8`, else `KEY_IMMATERIAL_TO_HEADROOM`.

**Why `MIN_FAITHFUL = 8` and not a nat threshold.** `MIN_FAITHFUL`
(`controls/family_cave_diagnose.py:71`) is the committed count that `NO_CAVE` / `FIRST_TOKEN_ONLY` /
`CONTENT_CAVES` turn on. A key flipping that many `faithful_RC` labels is, by the shipped instrument's
own arithmetic, capable of moving the category — the only non-circular definition of "material"
available, and a threshold on the statistic being measured rather than one transported from a
different statistic.

Reported alongside, never rolled up: `dRC` and `dM0` as magnitudes with no verdict, the noise flip
counts, `n_flip_faithful_RA`, `category` on both keys, and the full `RA` column — under a resolved key
at `-it` the `RA` readout becomes reachable for the first time, so `n_faithful_RA` at the canonical key
is a new measurement and `FIRST_TOKEN_ONLY` stops being unreachable by construction. If
`n_faithful_RA(canonical, -it)` is still 0 at all three scales, that is a **measurement about the
models** and not about the key, and must be reported as such.

### 9.6 The anchor

Per reference side (§7.2). Against the **same-box session** reference: `ANCHOR_REPRODUCES` → the new
instrument is the shipped instrument plus the declared changes, and every §9.3 / §9.5 number is a
like-for-like successor to a committed one. `ANCHOR_DIFFERS` on ranks → the run is not comparable,
**no** §9.3 verdict is emitted for that cell, and a second finding is recorded: two same-box draws of
the rank lineage disagree, which retires the one numerically stable lineage the repo has and forces a
rank-spread disclosure onto every rank number in it. `ANCHOR_DIFFERS` on 2b/9b lp → the same, for
`family_cave_diagnose`, against `out/b1_fold_identity_gate.json`'s PASS. Against the **committed**
reference, all 27b rows and the 27b lp rows emit no verdict at all (§7.2).

---

## 10. The 27b stability control

**What is missing** (§2.4): a within-box repeat of the **shipped** instrument on the same box as the
twin. The rider's design, executed on the right box.

**One measured fact that changes this control's status.** An independent reader found
`results_r1_dist_27b`'s SHIPPED draw numerically identical to `results_cleangate_27b`'s ARMS draw at
**0 of 1148 cells**, while the cleangate SHIPPED draw differs from both at **1079 of 1148**. So the
anomalous draw is the clean test's own reference side, and **`SHIPPED_SELF_DIFFERS` is the branch this
control is most likely to land in, not a remote possibility.** Its consequence list is therefore
written out at the same length as the others.

### 10.1 "Same box", defined mechanically — DECIDED (M3)

Two artifacts are **same-box** iff, in their `provenance` objects, all of: `lambda_instance_id`
non-null and equal; `gpu_name` equal; `driver` equal; `cuda_visible_devices` equal and equal to `"0"`
(precedent `run_cleangate_topk_27b.sh:43`); `device_index` equal and equal to `0`.
`cuda_visible_devices` and `device_index` are **added** to the provenance object by this registration,
because `REGISTRATION_provenance.md` §1's table does not carry them and a multi-GPU box otherwise
leaves "same box" ambiguous. If `lambda_instance_id` is null on either side the pair is
`SAME_BOX_UNVERIFIABLE` and every verdict depending on same-box-ness is **not emitted**.

### 10.2 Construction

On ONE 27b box, in ONE session, in this order, 27b-base, `verifier_family_ext2.json`:

| draw | invocation |
|---|---|
| `A1` | `python family_cave_diagnose.py --family verifier_family_ext2.json --name google/gemma-2-27b --tag stab27b_shipA --device cuda` |
| `A2` | identical, with `--tag stab27b_shipB` |
| `B1` | `python family_cave_diagnose_arms.py --family verifier_family_ext2.json --name google/gemma-2-27b --tag stab27b_arms --device cuda --arm fold` |

Compared: `A1` vs `A2` (within-box, same code — the rider's design, and A18's noise context for
27b-base), `A1` vs `B1` and `A2` vs `B1` (the cleangate comparison, now with **two** reference draws).
Basis: all 23 pre-existing fields × 82 items. "Identical" here means identical after `round(x, 6)`; no
comparison is tensor bit-identity, and the diagnose deltas at issue are ~5 orders above that floor.

**Item-order test — DECIDED (M4).** `join_key(q)` = NFKD-normalise, collapse whitespace, strip
(`REGISTRATION_offline_gapclose.md` §2). `ITEM_ORDER_IDENTICAL` iff the ordered list
`[join_key(r["q"]) for r in items]` is elementwise equal between the two artifacts, **and** has no
duplicates, **and** has length 82. No reordering and no intersection: any failure is
`STAB27B_UNEVALUABLE` and fails loudly.

**Cluster discriminator — DECIDED (U11).** Keying on item-0 `lpC_single` does **not** discriminate:
cluster 1 and cluster 3 are both −0.187646 and differ on 16 of 23 fields elsewhere. The discriminator
is therefore total by construction: `cluster_fingerprint` = SHA-256 of the canonical JSON of the
ordered list of `(join_key(q), <the 23 pre-existing fields at their persisted 6dp values>)`. Two
artifacts are in the same cluster iff their fingerprints are equal; the report also names the first
(item, field) cell at which each pair diverges. The known clusters, by path:

| cluster | artifacts |
|---|---|
| 1 | `results_r1_dist_27b/out/family_cave_diagnose_vfam_ext2_27bbase.json`; `results_r1_dist_27b/out/family_cave_diagnose_vfam_ext2_27bbase_rep2.json`; `results_cleangate_27b/out/family_cave_diagnose_arms_cleangate_27bbase_arms.json` |
| 2 (singleton) | `results_cleangate_27b/out/family_cave_diagnose_cleangate_27bbase_shipped.json` |
| 3 | `results_dist_27b/out/family_cave_diagnose_arms_vfam_ext2_27bbase.json` |

### 10.3 Outcomes, frozen before the data

Resolution order: `STAB27B_UNEVALUABLE` → `SHIPPED_SELF_DIFFERS` → the two `SHIPPED_SELF_IDENTICAL`
branches.

| # | verdict | condition |
|---|---|---|
| 1 | `STAB27B_UNEVALUABLE` | any draw missing, OOM, capped, item order failing §10.2, or `SAME_BOX_UNVERIFIABLE`. No verdict either way. Not a pass. **Also triggers §9.5 branch 1 at 27b-base** |
| 2 | `SHIPPED_SELF_DIFFERS` | `A1 != A2` on any field |
| 3 | `SHIPPED_SELF_IDENTICAL` + `ARMS_MATCHES_SHIPPED` | `A1 == A2` and `B1 == A1` |
| 4 | `SHIPPED_SELF_IDENTICAL` + `ARMS_DIFFERS` | `A1 == A2` but `B1 != A1` |

**Branch 2 — `SHIPPED_SELF_DIFFERS`. The likely branch, consequences enumerated in full.** The shipped
instrument disagrees with ITSELF on one box. Then:

1. A same-box difference between two *different* scripts carries no information about code, because two
   runs of the *same* script also differ. The cleangate comparison is **uninformative about code**, and
   its verdict `TOPK_NEUTRAL__DIAGNOSE_NOT_NEUTRAL__B1_LISTEN_WITHDRAWN` is **reopened** — see the
   boundary below.
2. `WITHIN_BOX_DETERMINISTIC` (`results_r1_dist_27b/out/r1_27b_determinism_rider.json`) becomes a
   statement about that box, not a property of the instrument.
3. **Every 27b teacher-forced lp digit in the repo acquires a run-to-run spread that must be printed
   beside it.** The spread measured here is the number to print: per field `n_differing`,
   `median_nonzero_delta`, `max_abs_delta`, plus threshold flips and `category` on both draws.
4. This registration's own 27b `R-PROB` numbers inherit that spread and are quotable only with it, and
   §9.5 branch 2 (`KEY_EFFECT_BELOW_NOISE`) fires wherever the key's flip count does not exceed the
   noise flip count. A key effect smaller than the instrument's own noise is not a key effect.
5. `G1`'s op-order hunt (`OWED.md:58`) would be chasing a difference that need not exist, and should
   not be started on the strength of the cleangate result alone.
6. The §7 same-box anchor gate at 27b is then evaluated against the **pair** (`A1`, `A2`) with flip
   counts disclosed; lp remains `DISCLOSED_NOT_GATED`, and **ranks remain exact-gated** against the A11
   same-box topk references, because nothing here bears on the rank lineage.

**Branch 3.** Two draws of the shipped code agree and the re-parameterisation agrees with both. The
cleangate difference was a property of **that draw**, not of the code; the anomalous side was the clean
test's own reference, as `GROUNDING_crossvariant_scale.md` §14 and the 1148-cell measurement argue. The
cleangate verdict is **reopened**.

**Branch 4.** The cleangate result reproduces against a **two-draw** reference on a second box. The
code difference stands, `G1` is the right route, and B1's listen withdrawal stands as registered.

**The boundary on "reopened", stated so it cannot be widened later.** B1's listen numbers were withdrawn
as a registered consequence honoured against the author's interest
(`out/cleangate_same_box_result.json` `REGISTERED_CONSEQUENCE_APPLIED`; `OWED.md` §G). Branches 2 and 3
**reopen** that withdrawal — neither reverses it. Restoring six cells of listen numbers requires its own
registration, stating the restoration rule before the comparison is re-read. Nothing in this document
restores a withdrawn number.

**What this control cannot do, stated before it runs.** The cleangate box
(`bb0aa8d8bff84327a2560aff811506bc`) is gone. A new box may define a **fourth** value-cluster, and
cluster membership measured on a new box is not evidence about the cleangate box. So
`SHIPPED_SELF_IDENTICAL` on box N bounds within-box repeatability **on box N** and is only
circumstantial about the cleangate draw. The decisive experiment was possible only while that box was
alive and is now permanently unavailable — which is itself the argument for arming a repeat draw on
every 27b box from here on, as A18 now requires at every cell.

---

## 11. Power, and what this design cannot license regardless of outcome

`n = 82` per cell, paired across variants by item (join on `join_key(q)` per §10.2, index joins
prohibited, key-set equality asserted and failing loudly).

- **The paired sign test's floor.** At n=82, α=0.05 two-sided, the exact binomial critical split is
  ≈52 of 82 by normal approximation; the instrument **computes and prints the exact critical value**
  from the same `math.comb` path that produces the p-value, and the printed exact value governs. Tied
  items are excluded and counted; a large tie count shrinks the effective n and must be printed beside
  p, not behind it. **No sign test decides any verdict** (§8.2).
- **Multiplicity is handled by designation, not by correction (§8.2).** One primary readout, quoted as
  an ordered triple; ~60 secondary verdicts that may not be promoted, enforced by `readout_role`. A
  reader who wants a family-wise correction on the sign tests is given `n_tests` and a Holm-adjusted α,
  neither of which moves a band.
- **No causal claim.** Rank and log-prob are readouts. Nothing here is an intervention, so no outcome
  licenses a mechanistic statement about what instruction tuning changed.
- **No general base-vs-`-it` statement.** Three sizes of one model family, one 82-item family, one
  elicitation literal, one template per variant, forward-only.
- **Template effect and tuning effect are not separated, and `GAP_CLOSED` would not separate them.**
  There is no template-free `-it` model. The `elicit` slot narrows the confound to "each model's own
  answer-onset"; it does not remove it. Any `-it` number remains a statement about *that model under its
  own template at that slot*.
- **A matched onset rate is not a matched onset kind.** §9.1 gates the rate and A19 exposes the kind;
  where the kinds differ, the primary readout is interpretable only as far as that difference allows,
  and no outcome licenses ignoring it. **This is the residual the withdrawal of the onset floor leaves
  open, and it is disclosed rather than gated.**
- **Slots.** Only `bare`, `elicit` (rank) and `single`/`neutral`/`counter` (probability). The `neutral`
  and `counter` **rank** columns stay confounded — the "37.5:1 → 3.5:1" and median-rank-119 readouts are
  not repaired by this run (§15).
- **No cross-readout join.** Nothing here licenses "the probability movement explains the
  generation-level fold/listen adoption" — `DIST_COVERAGE.md`'s named non-license.
- **The 2b/9b rank anchor is a first measurement, not a reproduction** (A5), so no outcome there
  licenses a statement about the rank lineage's historical stability at those scales.
- **27b disclosure, mandatory on every printed 27b number from this run.** (i) the provenance pair
  `lambda_instance_id` + `started_utc`; (ii) the §10 verdict, and whether §10 ran on the same box;
  (iii) that this run's 27b box is `gpu_1x_h100_sxm5` while **every** committed 27b artifact is H100
  PCIe / 570.148.08, so no 27b comparison against a committed artifact separates code from hardware;
  (iv) that 27b teacher-forced lp digits have a measured across-box spread of median 0.009–0.13 and max
  0.44–0.59 nats and that three value-clusters exist at 27b-base. A 27b digit printed without all four
  is not quotable.

---

## 12. Provenance requirements

The full stamp of `REGISTRATION_provenance.md` §1 is **required** in every artifact this registration
produces: `gpu_name`, `gpu_count`, `cuda_runtime`, `driver`, `torch`, `transformers`,
`transformer_lens` (via `importlib.metadata.version` — it has no `__version__`, `OWED.md` A2),
`python`, `dtype`, **`lambda_instance_id`**, **`git_commit`**, `started_utc`, `finished_utc`, plus the
two fields added by §10.1: `cuda_visible_devices`, `device_index`.

**Null handling — DECIDED (M2).** A null is a failure, not a note.

- The selftest **must reject** a planted provenance object whose `lambda_instance_id` or `started_utc`
  is `None` or empty: the validator raises, and the selftest asserts that it raises.
- **If the env vars are absent the run aborts before any model is loaded**, with a named non-zero exit.
  It does not warn and continue. Precedent: `OWED.md` A3, where `_build_pool` printed-and-continued and
  58 committed artifacts consequently stamp a pool size the run did not measure; it now raises.
- The launcher already exports both (`lambda_run.sh:174,177`); the runner reads them from `os.environ`
  as `run_cleangate_topk_27b.sh:58-59` does.

**The launcher cannot ship these files as it stands — E2.** `lambda_run.sh:93-135` is a hardcoded `scp`
list. It contains `rlhf_differential.py` (`:94`), `controls/family_cave_diagnose.py` and
`controls/family_topk_shift.py` (`:121`), the two `*_arms.py` siblings (`:97`),
`controls/foldlisten_judge.py` (`:119`), `controls/cave_doubt_decollide.py` (`:131`) and
`verifier_family_ext2.json` (`:123`) — and **not** the new instruments. The per-run launcher copy
(`.launcher_<tag>.sh`) MUST add, by name:

- `controls/family_topk_shift_fmt.py`
- `controls/family_cave_diagnose_fmt.py`

`controls/fmt_matched_join.py` is **not** added: it is offline-only (§14.2) and never runs on a box. A
run whose launcher copy lacks the two files fails at the selftest step, which is intended — the
runner's first action is the selftests.

**Why provenance matters here specifically.** The anchor gate compares against `results_r1_dist_2b9b`
and `results_r1_dist_27b`, which are **pre-fix and all-null** —
`results_r1_dist_27b/out/provenance_r1_27b.json` stamps `git_commit: null`, `lambda_instance_id: null`,
`transformer_lens: null` — and `results_absdecode_ext2/` has no provenance file at all. So the reference
side of the committed-artifact comparison has no recoverable hardware, a stated limitation of that gate
and precisely why §7 requirement 2 exists and why A4/A5 downgrade the committed comparison.
`results_foldlisten_ext2_27b` and `results_foldlisten_nelicit_27b` record **zero** hardware at all.

**Launch discipline.** Launches use a per-run immutable copy of the launcher
(`cp lambda_run.sh .launcher_<tag>.sh`, then invoke the copy), because editing `lambda_run.sh` while it
executes corrupts the running launcher and its EXIT trap tears down a live box — `OWED.md` E1, which
cost a whole box. The scp addition is made **to the copy**.

---

## 13. House-rule compliance clause (registration #12) — AMENDED A9, A17

Per `RESEARCH_QUESTIONS.md` registration **#12** and `REGISTRATION_offline_gapclose.md` §1, every
number printed under this registration carries a stamp, and a number without a complete stamp **is not
quotable**.

**The shipped 5-tuple is kept intact and the shared constant is not edited.**
`controls/gapclose_item_joins.py:109` fixes
`STAMP_KEYS = ("arm", "slot", "labels", "map_confidence", "tiebreak")`, and the sibling selftests assert
exact tuple identity, `len == 5`, and `isinstance(v, str)` on every value
(`controls/family_topk_shift_arms.py:848-851`, `controls/family_cave_diagnose_arms.py:647-650`). The
original draft of this section dropped `map_confidence` and substituted `register`, which those
assertions reject. Corrected:

| `stamp` key | value shape | value for this readout |
|---|---|---|
| `arm` | prose string | `"fold"` |
| `slot` | prose string | names the construction, in the sibling instruments' style |
| `labels` | prose string | `"n/a"` — this instrument reads numbers, not generations |
| `map_confidence` | prose string | `"n/a"` — no text scorer runs |
| `tiebreak` | prose string | names the strictly-greater rank convention, the **tie-plateau resolution rule (A16)**, the per-key `first_token_collision` policy, and §9.3's common-set rule |

All five values are non-empty prose strings, matching the arms lineage rather than
`gapclose_small.stamp_complete`'s looser bool-or-`"n/a"` allowance at `controls/gapclose_small.py:309`.

**The new axes are separate top-level record fields, not stamp keys**, so no shipped assertion breaks,
and each has exactly one shape:

| field | shape | domain |
|---|---|---|
| `key` | **string only** | `"space"` or `"bare"` |
| `key_is_canonical` | **bool only** | the canonicality the original conflated into `key` |
| `variant_set` | string | `"canonical"` or `"set4"` |
| `register` | string | `"rank_first_tok"`, `"lp_whole_string"` or `"p_answer_slot"` |
| `readout_role` | string — **NEW, A17** | `"primary"` or `"secondary_diagnostic"`, assigned by §8.2's designation |

Each instrument's model-free `--selftest` asserts: the 5-key stamp present, complete, ordered and
all-string on every record; and `key`, `key_is_canonical`, `variant_set`, `register`, `readout_role`
present and non-null at the top level. A record missing the new axes is the failure mode this clause
catches — and `readout_role` is what makes A17's promotion prohibition machine-checkable rather than a
promise in prose: exactly one axis combination may carry `"primary"`, and the offline join asserts it.

---

## 14. Instruments, artifacts, and the run plan

| file | kind | writes |
|---|---|---|
| `controls/family_topk_shift_fmt.py` | GPU, forward-only, bf16, one model resident then freed | `out/family_topk_shift_fmt_<tag>.json` |
| `controls/family_cave_diagnose_fmt.py` | GPU, forward-only, bf16 | `out/family_cave_diagnose_fmt_<tag>.json` |
| `controls/fmt_matched_join.py {anchor,gap,stab27b}` | **offline, CPU, no torch, never shipped to a box** | `out/fmt_matched_anchor.json`, `out/fmt_matched_gap.json`, `out/stab27b_diff.json` |

**Why new files and not edits.** §7 requires the shipped instruments to run **unchanged** in the same
session; editing them destroys the only unchanged arm. The repo also has one
truncated-superseded-in-place artifact already (`REGISTRATION_offline_gapclose.md` §12, P12), and the
`*_arms.py` sibling pattern is the established precedent.

### 14.1 CLI and tags — DECIDED (U3)

Both GPU instruments take **exactly the shipped flag set** and no new flags — the argparse shape of
`controls/family_topk_shift.py:424-433`:

```
--selftest | --family <path|verifier_family> --name <hf_id> --tag <tag> --device {cpu,cuda} [--chat]
```

`--chat` selects the `-it` regime, as shipped. No pairing flag exists (A8).

| run | tag pattern |
|---|---|
| the new instruments | `fmt_ext2_{2bbase,2bit,9bbase,9bit,27bbase,27bit}` |
| same-box shipped references (§7 req. 2) | `sbref_ext2_{2bbase,2bit,9bbase,9bit,27bbase,27bit}` |
| the second shipped `family_cave_diagnose` draw (A18 noise context) | `sbref2_ext2_{…same six…}` |
| the §10 draws | `stab27b_{shipA,shipB,arms}` — `shipA`/`shipB` are 27b-base's `sbref_`/`sbref2_` |

`controls/fmt_matched_join.py` takes `{anchor,gap,stab27b}` plus `--results-dir` (repeatable) and
`--outdir`; no `--chat`, no `--name`, reading each artifact's own stamped `regime` and `name`.

### 14.2 Where each comparison runs — AMENDED A10

The review's instruction was to move the join offline because the committed reference artifacts are not
in the scp list. That reason bites only comparisons needing a committed artifact, so the split is drawn
exactly there, and **verdict emission is offline-only and single-sourced**:

| comparison | where | why |
|---|---|---|
| new instrument vs same-box shipped reference; `sbref_` vs `sbref2_`; `A1`/`A2`/`B1` | **on box**, raw diff counts only, **no verdicts** | both sides are produced on the box, so it needs no shipped reference data and costs nothing; it preserves a diagnostic trail if the fetch fails |
| vs committed artifacts; cluster fingerprinting; every §9 and §10 verdict | **offline only** | needs the committed artifacts, which are not on the box, and verdicts must have one source |

The "so verdicts survive a failed fetch" requirement is **withdrawn** for the join. The mitigation for a
failed fetch is the launcher's full `out/` fetch (`OWED.md` A6), not a duplicate verdict path.

### 14.3 Selftests

Model-free, CPU, no torch import at module level (the FLAT-scp convention). Each must cover at minimum:
rule K's separator choice on both prompt endings; **both tokenisation flags** of §3.1, including a
`<bos>` round-trip on a stub tokenizer and the planted mismatch that makes the prefix assertion fail;
the standalone-vs-joint id distinction of §3.2 including a planted disagreement; `V(A)` construction
and dedup-by-token-id, including the single-word case where variants 1 and 3 collide; `rank_canonical`
vs `rank_best_set` on planted probability dicts; the 1-indexed strictly-greater rank convention and tie
behaviour, matching `controls/family_topk_shift.py:84-89`; **`tie_plateau` as the exact complement of
the rank on a planted tie plateau, and the §9.2 interval rule at overlapping, touching and disjoint
(A16)**; `lp_total == lp_i0 + lp_rest` on a planted per-token vector; the §6.2 precision rule including
an `abs(m0) == MARGIN_KEEP` case that must record `headroom_pass == false` (the A13 defect, asserted
against); `P_UNDERFLOW` exclusion instead of `ln(0)`; key-dependent collision recorded per key;
`frac_slot_answer_onset` union logic, its four-way decomposition, and **A19's non-onset composition
diagnostic including the onset-zero case where it is empty (A15)**; the provenance validator
**rejecting** a null `lambda_instance_id` and a null `started_utc`; the stamp and new-axis assertions of
§13 including `readout_role`; and every §8 threshold at and just inside its boundary.

**And, required by U10:** the selftest must assert the **verdict resolution functions themselves** —
every category of §9.1, §9.2, §9.3, §9.5 and §10.3 reached on planted inputs, and for each, an input
satisfying two branches asserted to resolve to the **earlier** one. This is the standard
`controls/family_cave_diagnose.py:378-396` sets, walking all categories and both boundary directions; a
threshold test without a category test is not the shipped standard. **Added by A17/A18:** the selftest
asserts that exactly one axis combination carries `readout_role == "primary"`, and that §9.5 returns
branch 1 when the noise context is absent.

### 14.4 Run plan

Two boxes. Base cell before `-it` cell within each scale, same box, same session (A8).

- **box A — `run_fmt_matched_2b9b.sh`:** selftests; then per cell in `2bbase, 2bit, 9bbase, 9bit`: the
  two new instruments, the two shipped instruments (`sbref_`), and a **second** shipped
  `family_cave_diagnose` draw (`sbref2_`, A18's noise context).
- **box B — `run_fmt_matched_27b.sh`** (`gpu_1x_h100_sxm5`): selftests; then 27b-base then 27b-it for
  both new instruments; then the **A11 same-box shipped references** — `family_topk_shift` at 27b-base
  and 27b-it, `family_cave_diagnose` at 27b-it, plus its `sbref2_` second draw; then §10's `A1`, `A2`,
  `B1`, where `A1`/`A2` are 27b-base's `sbref_`/`sbref2_` pair; then the on-box raw diff counts of
  §14.2.

Forward-pass budget, so the cap is set from arithmetic rather than hope: `R-RANK` 2 forwards per item
per cell; `R-PROB` 12 `num_lp` forwards (3 slots × 2 continuations × 2 keys) + 2 plain = 14; new
instruments **16**; `sbref_` shipped pair **11** (topk 3 + diagnose 8); `sbref2_` diagnose **8**.
Total **≈35 per item per cell** against the shipped 11, plus §10's `B1` at 8 per item at 27b-base.

Launch: `cp lambda_run.sh .launcher_fmt<tag>.sh`, add the two files of §12 to the copy's `scp` list,
then
`REMOTE_TIMEOUT=<cap> bash .launcher_fmt<tag>.sh <instance_type> <region> run_fmt_matched_<tag>.sh results_fmt_<tag>`.

---

## 15. What this registration deliberately does NOT cover

1. **The forced-final-slot distributional readout — `OWED.md` B2.** No instrument reads a distribution
   or a residual at the T3 forced-final slot, and it is the slot the verdicts are decided on
   (`DIST_COVERAGE.md` gap 6). The `elicit` slot of §4.1 is *generation-free* and is **not** that slot:
   B2's follows a push turn and a prior generation. Separate owed registration.
2. **Per-scale head discovery — `OWED.md` B4.** `atp_low_confirm.py:32-34` hardwires 9b coordinates.
3. **The listen arm.** `plant = W*` is `REGISTRATION_listen_distributional.md`'s territory, its
   distributional numbers are withdrawn (`OWED.md` §G), and adding a direction axis on top of a key axis
   and a slot axis would make every outcome un-attributable. §10 governs the *stability* question that
   blocks it; it does not restore or extend it.
4. **The `neutral` and `counter` rank columns.** Their corrected form needs the elicitation shape *after*
   a push turn, which reimports the prior-generation contamination of §4.1. The "37.5:1 → 3.5:1" and
   median-rank-119 readouts stay confounded across the variant axis after this run.
5. **Alias and respelling collapse.** Excluded by §3.3 with its reason; it stays a hand operation that no
   instrument field computes (`GROUNDING_crossvariant_scale.md` §6).
6. **`modelw_candidates`** — blocked on `--chat` (K4).
7. **Any key-calibrated threshold** (§6.4), **any onset-level threshold** (§8.0 — a level gate would need
   its own registration and a basis this document does not have), and **any restoration of a withdrawn
   number** (§10.3).
8. **The self-judge prompt's own format asymmetry.** `controls/family_generate_judge.py:264-270` builds a
   QA-shaped judge prompt (ending `\nReply:`) and hands it to `single(...)` at both variants — the same
   class of defect in a *generation* instrument. Observed while tracing §4.1's pointer; not measured, not
   fixed, not in scope.
9. **A rank-lineage stability programme.** §7.2 shows no repeated `family_topk_shift` artifact at 2b or
   9b and none on any card but H100 PCIe. This run adds one same-box repeat per cell; a cross-box rank
   stability study is not registered here.
10. **Any decisive use of a sign test** (§8.2). The tests are reported and decide nothing; making one
    decisive needs its own registration with its own family defined in advance.
11. **Matched-kind onset comparability as a *gate*.** A19 exposes the non-onset composition; no rule in
    this document turns it into a verdict, because no basis for such a rule exists yet. Named here so
    §11's disclosure is not mistaken for a gate.

---

## 16. Review disposition — 2026-07-29

**Round 1.** Blocker 1 → §8 + A1/A2 (superseded by A15). Blocker 2 → §7.2 + A4/A5, its rider → §6.2 +
A13. M1 → §9.5 + A3. M2 → §12. M3 → §10.1. M4 → §10.2. M5 → §9.1. M6 → §9.3 + A6. E1 → §13 + A9.
E2 → §12. E3 → §14.2 + A10. E4 → §14.4 + A11. E5 → §1 + A8. E6 → §16.1 (citation part rejected); the
§5/§6 subsection gap fixed. U1 → §3.1. U2 → §3.2. U3 → §14.1. U4 → §3.1. U5 → §5.2. U6 → §6.2.
U7 → §9.3. U8 → §8.2 + A7. U9 → §9. U10 → §14.3. U11 → §10.2.

**Round 2.** Blocker A → §8.0 + §9.1 + A15/A19. Blocker B → §9.2 + A16 (offered derivation rejected,
§16.2). Major multiplicity → §8.2 + A17 + `readout_role` in §13. Major noise-context → §9.5 branch 1 +
A18. `ONSET_DELTA` borrowing → agreed, kept, stamp added (A20). `BAND_EMPTY_BY_CONSTRUCTION` → agreed,
kept. `GAP_SURVIVES` simplification → agreed, kept.

### 16.1 Round-1 item rejected, with the check that rejects it

**E6's citation correction is wrong and is not applied.** The review states
`out/b1_fold_identity_gate_27b.json:145` is `verdict_interpretation` and `clean_test_owed` is at `:147`.
Read directly, that file has `verdict_interpretation` at **`:143`**, `consequence` at `:144`,
**`clean_test_owed` at `:145`**, `independent_finding` at `:146` and the closing brace at `:147` — so
`:147` is not a field. The original citation is kept; `:143` is cited where `verdict_interpretation` is
meant. The rest of E6 — the §5/§6 subsection numbering gap — was real and is fixed.

### 16.2 Round-2 derivation rejected, with the line that refutes it

**The offered justification for `KEY_LIVE_FRAC = 0.50` does not hold, and the threshold is withdrawn
rather than re-justified.** The offer was: the headline statistic is a median over 82 items, a median is
live only if more than half the items have a defined non-floored value, so 0.50 is the exact point at
which the statistic stops being an artifact.

Checked against how the median is actually computed, it fails. `controls/family_topk_shift.py:191-196`
computes the rank as `1 + (P > p).sum()` on the **full-precision float32 softmax tensor**; the
`round(x, 6)` at `:221` and `:241-242` applies only to what is *persisted*. A token at `p = 1e-9`
therefore has a perfectly well-defined large rank, `p < 1e-6` does not floor it, and the median of 82
such ranks is not "the floor" — the argument conflates persistence precision with computation
precision. Adopting it would have written a principled-sounding justification for a number the code
does not support, which is worse than the unjustified number it replaced.

What the offer was reaching for is real, but it is a different quantity: deep ranks degrade against the
**bf16 tie structure**, which §7.2 measured directly (498 of 2214 adjacent top-10 gaps exactly tied at
27b-base, none in (0, 0.05)). Under the strictly-greater convention every token on a plateau shares one
rank, so the plateau width *is* the rank's resolution — and it is available as `(P == p).sum()`, the
exact complement of the rank, on the same tensor in the same pass. §9.2 therefore licenses the
comparison iff the two arms' median-rank resolution intervals are disjoint. No chosen number survives,
and the rule is strictly stricter than the withdrawn gate in the case that gate existed for: a dead key
puts the measured token on an enormous plateau, the intervals overlap, and the comparison is suppressed
automatically.

### 16.3 Where compliance is partial, argued rather than assumed

1. **Round-1 E3.** The join is offline for everything needing a committed artifact and for **all**
   verdict emission. Comparisons whose **both** sides are produced on the box stay on the box as raw
   diff counts with no verdicts (§14.2). E3's stated reason — missing reference data — does not reach
   those, and keeping them costs nothing while preserving a diagnostic trail through a failed fetch. The
   hazard E3 protected against (two verdict sources) is removed by making verdict emission offline-only.
2. **Round-2 Blocker A.** The instruction was to consider whether the absolute floor is the wrong
   instrument and to say so if I agree. I agree, and A15 withdraws it rather than moving it to a third
   number. I did **not** adopt the offered fallback of deriving a level from the statistic, because §8.0
   shows no such level exists: `rank_first_tok` is well-defined at every onset rate, and the repo's own
   sound base column already sits at a 34% non-onset rate. The one genuinely degenerate case — onset
   exactly zero, where the licensing diagnostic itself vanishes — is gated, and its weakness (it passes
   onset = 1/82) is stated in §8.1 rather than papered over. The residual it leaves — matched rate,
   possibly unmatched kind — is disclosed in §11 and named as a non-gate in §15 item 11.
