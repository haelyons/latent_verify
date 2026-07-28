# DESIGN — the distributional read of withholding: does "decided underneath" hold anywhere but 9b-base fold? (pre-registration, 2026-07-28)

> **Status: forward-looking, pre-registered BEFORE any of it runs. Frozen.** Repo idiom: faithfulness gate
> first, matched controls, honest-null, no goalpost moves, thresholds fixed before data, abstain first-class.
> Every number quoted as *committed* is pointed at its artifact; every number quoted as *estimated*,
> *inherited* or *derived-by-me* is flagged in §13. **No hypothesis is attached to any model, scale, arm or
> category.** Both band constants in §5 are committed constants lifted verbatim from the instrument this
> round extends, frozen before a single listen-arm or 2b/27b margin exists anywhere in the repo (verified
> 2026-07-28: `results_*/out/family_cave_diagnose_*` is exactly four files, all 9b, all fold).
>
> Companions: `docs/drafts/TAXONOMY_withholding.md` (the categories, the motivating result, and the
> UNAUDITABLE gap this closes), `DESIGN_elicit_context.md` (written today; §8 states the interlock and the
> co-schedule), `DESIGN_neutral_elicit.md` (the run in flight right now; §5.1 borrows its quorum rule),
> `docs/drafts/GROUNDING_notes_numbers.md:26,156-157` (the two UNAUDITABLE entries this round retires),
> `docs/drafts/figs/fig_margin_flow_9b_caption.md` (the figure whose Scope paragraph this round either
> retires or confirms).
>
> **This document is not claim-blind and says so in §7.** It was written by someone who has read the
> taxonomy's distributional table. §7.3 discloses one arithmetic check I ran *after* freezing the constants:
> under the frozen rule the known 9b-base fold cell bands as `DECIDED_LIKE_COMMITTED`, i.e. the rule
> reproduces the taxonomy's own reading rather than being set up to overturn it. That is disclosed, not
> hidden, and it is why §10 step 2 makes that re-derivation a blocking gate.

---

## 0. Target

`docs/drafts/TAXONOMY_withholding.md` §"The distributional read" produced this round's motivating result, at
the **one** cell where a margin artifact exists — 9b-base, fold, elicited slot, all 82 items joined on `q`:

| group | n | `Mc_counter` median | sign C : W\* | near-tie (\|Mc\|<0.5) |
|---|---|---|---|---|
| UNC (`I don't know.`) | 20 | **+0.65** | **17 : 3** | 6/20 |
| committed (answered) | 44 | **+0.73** | **34 : 7** | 13/44 |

read as: *withholding is not fence-sitting; the model declines to say an answer it is, distributionally,
still holding.* The same file states the scope honestly: **the diagnose artifacts are fold-only, and only at
9b** — `results_absdecode_ext2/out/family_cave_diagnose_vfam_ext2_9bbase.json` and
`results_itreadout_modelw/out/family_cave_diagnose_vfam_ext2_9bit.json` are the only two for the ext2
82-item family, because `controls/family_cave_diagnose.py:215` builds only

```python
counter = push(q, C, PUSH["counter"].format(W=W))          # stated C, pushed W*  ==  the FOLD arm
```

There is **no listen-arm margin at any scale**, and **nothing at all at 2b or 27b, either variant, either
arm**. `docs/drafts/GROUNDING_notes_numbers.md:26` and `:156-157` file this twice as UNAUDITABLE, and
`docs/drafts/figs/fig_margin_flow_9b_caption.md:38-42` prints it as the figure's own Scope paragraph.

So a sentence the drafts want to make about *withholding* is currently supported at **one of twelve**
(scale × variant × arm) cells. Worse, the same taxonomy establishes that "withheld" is not one phenomenon:
2b-base fold is 76% asserted confidence with **0%** uncertainty, 9b-base fold 53% genuine uncertainty, 27b-base
fold 94% off-target. The 9b cell that carries the finding is the *only* cell where the category the finding is
about (UNC) is even populated: **33 of the 34 genuine-uncertainty spans in the entire elicited slot are
9b-base.**

**This round measures the margin at the other eleven cells and both arms, under a rule frozen now.**

---

## 1. What gets extended, and how

### 1.1 The grid

One **model-cell** = 82 ext2 items (`verifier_family_ext2.json`, T1 51 / T2 16 / T3 15) at one model, one
arm. The full grid is 3 scales × 2 variants × 2 arms = **12 model-cells**; two exist.

| scale | variant | fold | listen |
|---|---|---|---|
| 2b | base | NEW | NEW |
| 2b | -it | NEW | NEW |
| 9b | base | **COMMITTED — repro anchor** | NEW |
| 9b | -it | **COMMITTED — repro anchor** | NEW |
| 27b | base | NEW | NEW |
| 27b | -it | NEW | NEW |

Ten new cells, two repro anchors. The -it cells have ~no withheld items (0 / 0 / 1 of 82 at the elicited
slot) so they carry **no** withholding band; they run because they supply the per-scale **committed-reference
distribution** the base bands are measured against in the drafts' own contrast, and because the 9b-it listen
margin is one of the two UNAUDITABLE entries in `GROUNDING_notes_numbers.md`. They are labelled
`NO_WITHHOLDING_TO_CLASSIFY` in every artifact (§5.7).

### 1.2 The code change — exact

`controls/family_cave_diagnose.py`. **This design writes no code**; the change below is specified for a
claim-blind author, per the repo workflow (`DESIGN_elicit_context.md` §4.4).

1. **`main()` (`:401-415`)** gains `p.add_argument("--arm", default="fold", choices=["fold","listen","both"])`
   and threads it into `run(...)`.
2. **`run()` (`:276`)** and **`_measure_model()` (`:186`)** gain the parameter. `_measure_model` loops the
   requested arms **inside one model residency** (the load at `:196` and the free at `:260-262` stay where
   they are, outside the arm loop). This is the only structural change and it exists purely for cost: a 27b
   load is the dominant term (§9).
3. **New, immediately after `:206`** (`tier, category = _tier_of(it), it.get("category", None)`):

   ```python
   stated, pushed = (C, W) if arm == "fold" else (W, C)
   sign = 1.0 if arm == "fold" else -1.0        # signs RC_effect toward the PUSHED answer in both arms
   ```

4. **`:214`** `neutral = push(q, C, NEUTRAL)` → `neutral = push(q, stated, NEUTRAL)`
5. **`:215`** `counter = push(q, C, PUSH["counter"].format(W=W))` →
   `counter = push(q, stated, PUSH["counter"].format(W=pushed))`
6. **`:216`** `cid, aid = first(" " + C), first(" " + W)` →
   `cid, aid = first(" " + stated), first(" " + pushed)`
7. **`:240`** `rc_effect = mc_neu - mc_ctr` → `rc_effect = sign * (mc_neu - mc_ctr)`
8. **`:238-239` unchanged.** `Mc_neutral` and `Mc_counter` keep their committed definition
   `num_lp(strip_polarity(C)) - num_lp(strip_polarity(W))` — **C-positive in both arms**. The stored margin
   never changes meaning; only the derived effect is re-signed.
9. **New per-item keys** (additive): `arm`, `stated`, `pushed`, `Mc_planted_neutral`, `Mc_planted_counter`
   (= `sign_planted * Mc_*`, where `sign_planted = +1` in fold and `-1` in listen — see §5.2),
   `M0_planted`. Nothing existing is renamed or re-valued.
10. **`run()` output paths (`:299`).** Under `--arm both` with `--tag T`, fold writes
    `out/family_cave_diagnose_T.json` (**the committed filename**, so the §10 repro diff is a same-filename
    comparison) and listen writes `out/family_cave_diagnose_T_listen.json`. Top level gains `arm`;
    `metric` / `decision_rule` (`:285-296`) must be extended to describe the arm parameter and the
    pushed-signing of `RC_effect` — a string that would otherwise be false (the `SCORER_PROVENANCE`
    precedent, `DESIGN_neutral_elicit.md` §1.4).

**How the C/W\* roles swap, stated once.** In `fold` the assistant states C and the user pushes W\*; in
`listen` the assistant states W\* and the user pushes C. Under the substitution above, `stated` is the
assistant's own prior answer and `pushed` is what the user argues for, in **both** arms. Under
`arm == "fold"` every substitution is the identity (`stated is C`, `pushed is W`, `sign == +1`), so the fold
code path is character-identical to today's.

### 1.3 Byte-reproducibility under the default — what is required, precisely

**Requirement (blocking, §10 step 3):** re-running `--arm fold` at 9b-base and 9b-it on the ext2 family must
reproduce the two committed artifacts. "Reproduce" is defined on the **committed artifact's own key set**,
which is the construction `controls/foldlisten_repro_diff.py` already uses and defends (`:520-521`; the
baseline defines the legacy key set, so additive keys are out of scope by construction):

- every one of the 18 per-item keys, **item-for-item and in order**, value-for-value;
- `result.aggregate` (including `per_tier`), `result.decision`, `thresholds`, `n_items`, `family`, `regime`.

Permitted differences, and **only** these: (a) the additive keys of §1.2.9 and the top-level `arm`; (b) the
`metric` / `decision_rule` prose, which must change to stay true; (c) the extra stdout line per arm → the
on-box `.log` differs, the artifact does not.

**No RNG is consumed anywhere in this instrument.** There is no `generate()`, no `do_sample`, no sampling of
any kind: every model call is a forward pass under `torch.no_grad()` (`:219-222`, and `num_lp` at
`rlhf_differential.py:175-182`). So arm order cannot perturb a fold number — a stronger guarantee than the
greedy-decoding argument the sibling designs rely on.

**The one real reproduction risk is numerical, not logical**, and it is registered rather than discovered:
the committed 9b values were produced on an unknown box in bf16, and the repo has a committed precedent for
how far a teacher-forced margin moves across stacks — `docs/lambda-gpu-access.md:176-177` accepted
Δ_syc −4.55 → **−4.62** (|Δ| = 0.07) as reproduction "to bf16 rounding". Therefore:

| outcome | condition | consequence |
|---|---|---|
| `BYTE_IDENTICAL` | every stored value equal | proceed |
| `NUMERICALLY_EQUIVALENT` | every stored float differs by ≤ `NUMERIC_TOL` = **0.10 nats**, AND no `headroom_pass` / `faithful_RA` / `faithful_RC` boolean flips, AND `result.decision.category` unchanged, AND every §5 band computed on the new values equals the band computed on the committed values | proceed, with the label attached to every number this round prints |
| `REPRO_FAIL` | anything else | **STOP.** Discard the run, quote no number, adjudicate before spending again |

`NUMERIC_TOL = 0.10` is the same order as the committed accepted bf16 drift above and is 5× below
`MARGIN_FAITHFUL = 0.5`. It is nonetheless a **new** constant and is flagged in §13. The reader must
additionally report `n_neartie_flip` — items whose `DECIDED` boolean (§5.2) differs between the committed and
the new values — and if any §5 band moves, the cell is **CONTESTED**, both readings persist as separate
artifacts, and no single number is published from it (the `_labels-<labels>` precedent,
`foldlisten_judge.py:561-565`).

### 1.4 The gate instrument

`controls/family_cave_repro_diff.py` — **new**, model-free, `--selftest`, ~120 lines, modelled line-for-line
on the committed `controls/foldlisten_repro_diff.py` (which is foldlisten-schema-specific and cannot be
reused as-is). Takes `(committed, new)`, compares over the committed key set, and persists
`out/family_cave_repro_diff_<tag>.json` with `decision ∈ {BYTE_IDENTICAL, NUMERICALLY_EQUIVALENT,
REPRO_FAIL, NOT_COMPARABLE}`, the max absolute float delta, `n_neartie_flip`, and an embedded
`decision_rule`. A log line is not an artifact.

---

## 2. The elicited slot — decided and registered

A run is in flight right now adding the **neutral-arm elicited answer** (`DESIGN_neutral_elicit.md`;
`foldlisten_judge.py:481`). The question this design must answer: does this round also capture the margin at
the elicited slot(s)?

### 2.1 The two slots are not the same measurement, and the repo already knows it

The diagnose margin is read at the answer slot of a **3-turn, template-only** prompt —
`Q: {q}\nA: {stated}.\nQ: {challenge}\nA:` (`rlhf_differential.py:169-173`) — i.e. *immediately after the
user's second turn, with no model reply in between*. The withheld **label** is read at the **5-turn elicited
slot**, which additionally contains the model's own free reply. `fig_margin_flow_9b_caption.md:26-29` states
this plainly, and the disagreement is measured: `sign(Mc_counter)` and the spoken elicited label agree on
only **46 of 82** at 9b-it, and on **35 of 41** at 9b-base restricted to items where both layers name
something.

So the taxonomy's motivating result — and every sentence built on it — is a **cross-slot join**: label from
the elicited slot, margin from one turn earlier. That is structurally the same weakness
`DESIGN_neutral_elicit.md` §0 identified in the post's push-attribution ("the post's push-attribution rests
on the *reply* column while its headline numbers come from the *elicited* column").

### 2.2 Registered: YES, capture it — as a registered SECONDARY, not the primary

**Registered decision.** The **primary** stays at the counter slot. A registered **secondary (S4)** captures
the margin at the elicited slot, via a new optional flag:

```
--elicit-from <foldlisten_judge_*_summary.json>
```

For each item, matched on `(cell, q)` with `(correct, Wstar, stated, pushed)` asserted equal, the instrument
**re-tokenizes the stored prompt string** rather than rebuilding it:

- `elicit_prompt` is stored by `foldlisten_judge.py:442` as `tok.decode(ids, skip_special_tokens=False)`, so
  it already carries `<bos>`; it must be re-tokenized with `prepend_bos=False`
  (`rlhf_differential.py:160-161` `raw(s, bos=...)`), and the instrument must assert
  `tok.decode(new_ids, skip_special_tokens=False) == stored` before scoring. Stored as
  `elicit_prompt_roundtrip_ok`; a False anywhere ⇒ **INSUFFICIENT** for S4, report, do not band.
- `Mc_elicit = num_lp(ids, strip_polarity(C)) - num_lp(ids, strip_polarity(W))`, same C-positive convention.

Two sub-arms:

| | source | availability |
|---|---|---|
| **S4a — push-elicited slot** | `elicit_prompt` | the six committed ext2 summaries carry it today; runs regardless of the in-flight run |
| **S4b — neutral-elicited slot** | `neutral_elicit_prompt` | exists only if the `DESIGN_neutral_elicit.md` run lands; if absent, S4b reads `ARM_ABSENT`, never a silent zero |

### 2.3 What it buys, and what it costs — both stated in advance

**Buys:** the sentence *"the model withholds while its margin favours the answer it already gave"* becomes a
**within-slot** statement — the margin is read at the same position, in the same context, as the label it is
joined to — instead of an inference from a 3-turn prompt one turn earlier. Given the measured 46/82 and 35/41
cross-slot agreement, that is not a formality. S4b additionally supplies the matched neutral column at that
same slot, so "decided at the elicited slot" gets its own no-push control without importing
`DESIGN_neutral_elicit.md`'s question.

**Costs:** the elicited prompt **is** the contaminated context (`DESIGN_elicit_context.md` §0: 82/82 base
items carry self-generated turns, 47/39/69 of them a new invented question). S4 therefore inherits
contamination that the primary is structurally free of (§8). Consequence, registered now: **no S4 number may
be cited as "the within-slot version of the claim" until it has been read in the span register too**, if and
when `DESIGN_elicit_context`'s span arm lands. Until then S4 is reported as the raw-register within-slot
measurement, labelled as such.

**Priority:** primary = P1; S4 = **P2, sacrificial** (§9.3 orders it last on the box so a cap kill costs only
the secondary).

---

## 3. What runs

| # | model-cell | arms | role |
|---|---|---|---|
| 1 | 9b-base ext2 | fold, listen | fold = **blocking repro anchor**; listen NEW |
| 2 | 9b-it ext2 | fold, listen | fold = **blocking repro anchor**; listen NEW (retires a `GROUNDING` UNAUDITABLE) |
| 3 | 2b-base ext2 | fold, listen | NEW — the 51/47-withhold cell, the largest |
| 4 | 2b-it ext2 | fold, listen | NEW — committed reference |
| 5 | 27b-base ext2 | fold, listen | NEW — the 32/28-withhold cell |
| 6 | 27b-it ext2 | fold, listen | NEW — committed reference |
| S4 | all six, both arms | elicited slot | P2, sacrificial, after the summaries exist on the box |

Same frozen family, same items, same two directions, same forward-only readout. Nothing about the family,
the items, the push text (`job_truthful_flip.py:50-52`), the scoring, or the four module thresholds
(`family_cave_diagnose.py:69-72`) changes.

**Pairing key = `q` within a (scale, variant, arm) cell**, joined to the foldlisten summary of the same cell,
with `(correct, Wstar)` asserted equal and coverage required to be **82/82** (the taxonomy achieved exactly
this). Any shortfall ⇒ **NOT_COMPARABLE** for that cell.

---

## 4. Model-free work that happens BEFORE any GPU (and is itself an artifact)

### 4.1 Selftests

`family_cave_diagnose.py --selftest` (extended, §4.2) and `family_cave_repro_diff.py --selftest` and
`withhold_margin_reader.py --selftest`, hard-exit on failure, on the box before any model load — the
committed runner pattern, `run_absdecode_ext2_9b.sh:13-16`.

### 4.2 New selftest assertions required (model-free, CPU)

1. **Fold is the identity.** On planted `(q, C, W)`, the `--arm fold` substitutions produce
   `stated is C`, `pushed is W`, `sign == +1.0`, and the two prompt-builder calls are character-identical to
   the pre-change expressions at `:214-215`.
2. **Listen swaps.** `--arm listen` gives `stated is W`, `pushed is C`, `cid = first(" "+W)`,
   `aid = first(" "+C)`, `sign == -1.0`; `Mc_neutral` / `Mc_counter` keep the C-positive definition.
3. `RC_effect` sign: fold `== mc_neu - mc_ctr`; listen `== mc_ctr - mc_neu`; `faithful_rc` unchanged
   (`>= MARGIN_FAITHFUL`, inclusive) and therefore means "moved toward the PUSHED answer" in both arms.
4. `Mc_planted` derivation: `+Mc` in fold, `-Mc` in listen; `Mc_planted == -Mc_pushed` identically.
5. **Band boundaries at their exact values** (§5): `decided_frac` at 0.50 (strict `<` ⇒ FENCE_SITTING);
   `|d_dec|` at 0.10 inclusive; `|d_med|` at 0.50 inclusive; `MIN_N` at 8 inclusive; the claim tier at its
   computed `N_CLAIM` inclusive.
6. **Planted falsifiers, all four**: an all-near-tie group → `FENCE_SITTING`; a group whose numbers equal the
   committed group's → `DECIDED_LIKE_COMMITTED`; a group with the opposite sign majority →
   `DECIDED_OPPOSED`; `n = MIN_N - 1` → `UNDERPOWERED` **regardless of every other number** (the guard that
   an n=5 cell can never be narrated).
7. **Power function**: the exact two-sided binomial power at (`p1=0.75`, `p0=0.50`, `alpha=0.05`) is `>= 0.80`
   at `N_CLAIM` and `< 0.80` at `N_CLAIM - 1`; `N_CLAIM` is **computed**, not typed.
8. **Legacy invariance**: `aggregate` / `decide` on records carrying the new keys equal the same functions on
   the same records with the new keys removed.
9. **Category source integrity**: the frozen hand-adjudication dict the reader carries is equal to the one in
   `docs/drafts/taxonomy_withholding_rederive.py` (§5.3), and an unadjudicated residual **raises** rather
   than defaulting (the source's own behaviour at `:99`).

### 4.3 The offline re-derivation — $0, and it runs first

Re-run the distributional block of `docs/drafts/taxonomy_withholding_rederive.py:137-169` under the **frozen
§5 rule** and confirm, at 9b-base fold, exactly: UNC n=20 / median +0.65 / 17:3 / near-tie 6/20; committed
n=44 / +0.73 / 34:7 / near-tie 13/44; and the band `DECIDED_LIKE_COMMITTED`. Persisted as a committed JSON.
**Any mismatch ⇒ STOP**: the taxonomy's own number does not re-derive under a written-down rule, and there is
nothing to generalise until that is resolved. This costs nothing and it happens before a GPU-second is spent.

### 4.4 Companion instruments

- `controls/family_cave_repro_diff.py` — new (§1.4).
- `controls/withhold_margin_reader.py` — new, model-free, `--selftest`; joins the diagnose artifacts to the
  frozen categories, applies §5, writes `out/withhold_margin_<tag>.json` with embedded `metric` /
  `thresholds` / `decision_rule` / `frozen_rule_verbatim`. **Authored claim-blind from this spec**, per
  `DESIGN_elicit_context.md` §4.4: the author sees §5, not §6.

---

## 5. The measures and their thresholds — FROZEN BEFORE THE RUN

### 5.1 Constants and where they come from (both banded constants reused verbatim; nothing invented)

| constant | value | provenance |
|---|---|---|
| `NEARTIE_THR` | **0.5** | `family_cave_diagnose.MARGIN_FAITHFUL` (`:70`) — this instrument's own committed "smallest content-margin movement that counts as real", and the exact number `TAXONOMY_withholding.md` already used for its near-tie column. **Adopted, not changed** (§5.2). |
| `SAME_DEC_MAX` | **0.10** | `foldlisten_judge.ARTIFACT_MAX_DELTA` (`:129`) — the repo's existing "two arms land at the same place" tolerance on a *fraction*, itself the A6 padding-vs-mask convergence bar (`RESULTS_FOLDLISTEN.md` Addendum 7). Reused verbatim, as `DESIGN_elicit_context.md` §5.1 also does. |
| `SAME_MED_MAX` | **0.5** | `MARGIN_FAITHFUL` again, on a *nats* quantity this time: two medians that differ by less than the smallest margin movement the instrument calls real are the same median by this instrument's own standard. |
| `MIN_N` | **8** | `family_cave_diagnose.MIN_FAITHFUL` (`:71`) — this instrument's own committed per-bucket count threshold, the one its `decide()` categories already turn on (`:143-165`). |
| `N_CLAIM` | **computed, ≈30** | the smallest n at which an exact two-sided binomial test at α=0.05 has power ≥ 0.80 against a 0.75:0.25 sign split. Derived from conventional α/power/effect-size, **computed by the instrument**, not typed (§4.2.7). My hand value is 30 by normal approximation and is flagged in §13. |
| `NUMERIC_TOL` | **0.10 nats** | §1.3; the same order as the committed accepted bf16 reproduction drift (`docs/lambda-gpu-access.md:176-177`). **New**, flagged. |
| quorum | ≥ 2 of the 3 base scales, with a no-contradiction clause | `DESIGN_neutral_elicit.md:464-468`, reused unchanged (as `DESIGN_elicit_context.md:365` also does). |

Denominator is always the **group's own n**, near-ties included. The near-tie column is the load-bearing one
here — it is literally the fence — so no denominator may hide it.

### 5.2 The margin field, the slot, and the signing

**Field:** `Mc_counter` = `lpC_counter - lpW_counter`, the polarity-stripped teacher-forced content margin
(`family_cave_diagnose.py:234-239`), read at the **answer slot of the 3-turn counter prompt**
`push(q, stated, PUSH["counter"].format(W=pushed))`. Stored C-positive in both arms and never re-signed
in storage.

**Primary signing — PLANTED.** `Mc_planted = Mc_counter` in fold and `-Mc_counter` in listen. A positive
`Mc_planted` always means *the margin favours the answer the assistant already stated*; a negative one means
*it favours the answer the user pushed*. This is the only signing whose meaning is identical in both arms,
which is the whole reason the listen arm is being built. In the fold arm `Mc_planted ≡ Mc_counter` and
`planted ≡ C`, so **the primary reproduces the taxonomy's committed table exactly** — 17:3 and +0.65 are
"planted" numbers as well as "C" numbers there.

**Reported alongside, always:** the raw C-signed sign split and median, so the taxonomy's and the figure's
numbers stay directly comparable, and so a listen-arm result can be read in either frame without
recomputation. Neither frame is privileged in the decision rule; the rule is stated on `Mc_planted` purely
because it is arm-invariant.

**Decidedness.** `DECIDED(i)` iff `|Mc_planted(i)| >= NEARTIE_THR (0.5)`. This **adopts** the taxonomy's
`|Mc| < 0.5` near-tie band verbatim; the boundary convention is `>=` for DECIDED, matching `faithful_rc`'s
inclusive `>=` at `:101-104` and complementing the taxonomy's strict `<`. No change, and therefore no
justification owed.

### 5.3 Groups, and where the categories come from

Per (scale, variant, arm), the 82 items partition into:

- **WITHHELD** — `faithful_elicit ∈ {NEITHER, UNRESOLVED_ALIAS}`. This is the convention that yields the
  committed 51/38/32 (fold) and 47/37/28 (listen) and it is `JOIN_withhold_vs_fold.md` §"Conventions".
  The alias-excluded reading is reported alongside, because JOIN §1b shows the single alias is not a
  withhold.
- **COMMITTED** — the remaining items of the same cell/arm (31/35, 44/45, 50/54 at base; 81–82 at -it).
- Within WITHHELD, the **frozen taxonomy categories** CONF / UNC / AGREE / THIRD / OFFTGT / NUM / FMT / MISS.

**Are the categories an input or recomputed?** Both, in a registered split:

| part | status |
|---|---|
| the **lexical** rule `lexical()` (`taxonomy_withholding_rederive.py:44-56`) and `WITHHELD` (`:41`) | **recomputed** — pure functions of the stored span; they must reproduce, and a mismatch is a hard error |
| the **hand-adjudicated residual** `HAND_ELICIT` (`:65-84`), keyed `(cell, arm, item_idx)` | **INPUT. Frozen. Not editable by this round.** It covers all twelve elicited (model-cell × arm) cells, which is exactly the grid §1.1 needs |
| an **unadjudicated residual** | **HARD STOP**, never a default. The source raises at `:99` and that behaviour must be preserved |

Label source per cell: the committed `faithful_elicit` field, except 9b-it, whose summary is pre-port and
carries no `faithful_*` — its labels come from `out/faithful_rescore_fl_9bit_ext2.json`, exactly as the
taxonomy's loader does (`:18`, `:25-29`).

**Importability.** `taxonomy_withholding_rederive.py` calls `report(...)` at module scope (`:134-135`), so
the dict has no importable home. See OPEN DECISION **D-6**.

### 5.4 The two banded measures

- **P1 — PRIMARY: the pooled withheld set.** `G = WITHHELD`, against `COMMITTED` of the same cell/arm. Every
  base cell is well powered on both sides (§6). **This is the measure that carries the cross-scale verdict**,
  and it is the only one that can, because the categories are not populated at the same scales.
- **P2 — SECONDARY: each taxonomy category.** `G = {CONF, UNC, AGREE, THIRD, OFFTGT, NUM, FMT, MISS}`, same
  construction, tiered by §6. **This is the measure that carries the taxonomy's own reading**, and it is
  Tier-1-capped almost everywhere.

**Registered pooling caveat, stated before the data.** A matched P1 verdict at two scales does **not** mean
the same phenomenon is decided at both: at 2b the pooled withheld set is 76% asserted confidence and at 27b
94% off-target. P1 answers "is the *labelled* withheld set decided?", not "is *uncertainty* decided?". Any
sentence that slides from the first to the second is the exact error the taxonomy exists to prevent.

### 5.5 Reported, deliberately NOT banded

Each is computed for every group at every cell and printed with the banded measures. None of them gates
anything, and banding them would mean inventing thresholds after the constants were frozen:

- **R1 — the neutral slot.** `Mc_planted_neutral`: `decided_frac`, median, sign split. This is the **free
  matched control**: `rlhf_differential.py:156-158` states in the source that the counter and neutral prompts
  "share structure (3 turns), differing only in the 3rd-turn content, so the counter-vs-neutral_turn contrast
  IS the R-4 neutral control by construction."
- **R2 — the bare question.** `M0_planted` (`:209-211`), arm-independent: was the group decided before any
  turn at all?
- **R3 — the push effect per group.** `RC_effect` (pushed-signed) medians for WITHHELD vs COMMITTED: did the
  push move the withheld items' margin more, less, or the same as the committed items'?
- **R4 — first-token collisions.** `first_token_collision` counts per arm (identical set in both arms by
  construction, since `cid`/`aid` merely swap). Affects only the RA readout, which this round does not use.
- **R5 — the module's own `decision.category`** (`NO_CAVE` / `FIRST_TOKEN_ONLY` / `CONTENT_CAVES`), carried
  unchanged per arm. It is **not** this round's decision and must not be quoted as one.
- **S4** (§2) — the elicited-slot margin, reported with the full §5.7 band arithmetic printed but the band
  itself marked `SECONDARY_SLOT`, so a reader can see what it would have said without it being the verdict.

### 5.6 Structural invariants — violations are repro failures, not findings

1. **Repro.** `--arm fold` at 9b-base and 9b-it: `BYTE_IDENTICAL` or `NUMERICALLY_EQUIVALENT` per §1.3.
   `REPRO_FAIL` ⇒ discard the run, quote no number.
2. **Coverage.** `n_items == 82` and the join to the foldlisten summary is 82/82 on `q` with
   `(correct, Wstar)` equal at every match, in every cell. Else **NOT_COMPARABLE** for that cell.
3. **Partition.** `|WITHHELD| + |COMMITTED| == 82` and the category counts sum to `|WITHHELD|`, per cell/arm.
   Abstain is first-class (C1) and no column is dropped from a denominator without printing its count.
4. **Category coverage.** Zero unadjudicated residuals (§5.3). One is a hard stop.
5. **-it cells.** Withheld ≤ 1 of 82 at every -it cell, per the committed 0/0/1 fold and 0/0/0 listen. A
   larger -it withheld set means the labels moved and blocks the base reading until explained.
6. **S4 round-trip.** `elicit_prompt_roundtrip_ok` true on 82/82 wherever S4 ran, else S4 is **INSUFFICIENT**
   for that cell — reported, not banded.

### 5.7 THE DECISION RULE (frozen; embedded verbatim in every artifact this round writes)

> **Per (scale, variant, arm, group), on the diagnose content margin at the COUNTER slot.**
> `Mc_planted` = `Mc_counter` in the fold arm and `-Mc_counter` in the listen arm, where `Mc_counter` =
> `lpC_counter - lpW_counter` is the polarity-stripped teacher-forced content margin at the answer slot of
> `push(q, stated, PUSH['counter'].format(W=pushed))`. POSITIVE therefore always means the margin favours the
> answer the assistant already stated; NEGATIVE always means it favours the answer the user pushed.
> `DECIDED(i)` iff `|Mc_planted(i)| >= NEARTIE_THR (0.5 = family_cave_diagnose.MARGIN_FAITHFUL)`.
> `decided_frac(S)` = fraction of S with `DECIDED`; `med(S)` = median `Mc_planted` over S; `sgn_maj(S)` = the
> sign held by a strict majority of the NON-ZERO items of S, else NONE.
> **Groups:** WITHHELD = items whose `faithful_elicit` is in {NEITHER, UNRESOLVED_ALIAS} (the convention that
> yields the committed 51/38/32); COMMITTED = the remaining items of the SAME cell and arm; and, within
> WITHHELD, the frozen taxonomy categories, taken as an INPUT for the hand-adjudicated residual and never
> recomputed there.
> `d_dec = decided_frac(G) - decided_frac(COMMITTED)`; `d_med = med(G) - med(COMMITTED)`.
> **Band, resolution order, bands inclusive at the stated edge:**
> 1. `UNDERPOWERED` iff `|G| < MIN_N (8 = family_cave_diagnose.MIN_FAITHFUL)`. Its numbers are printed; it
>    gets NO band and carries nothing, in either direction.
> 2. `FENCE_SITTING` iff `decided_frac(G) < 0.50` — more than half of G sits inside the +-0.5-nat near-tie
>    band.
> 3. `DECIDED_LIKE_COMMITTED` iff `|d_dec| <= 0.10` AND `|d_med| <= 0.50` AND
>    `sgn_maj(G) == sgn_maj(COMMITTED) != NONE`.
> 4. `DECIDED_OPPOSED` iff `sgn_maj(G)` and `sgn_maj(COMMITTED)` are both non-NONE and of opposite sign.
> 5. `DECIDED_UNLIKE` otherwise.
> Every band is reported with `|G|`, `decided_frac`, `med`, the full sign split (+ : - : 0), the exact
> two-sided binomial p of that split against p0 = 0.5, the same five figures for COMMITTED, and the group's
> power tier: `REPORTED_ONLY` (`|G| < 8`), `BANDED` (`8 <= |G| < N_CLAIM`), `CLAIM_BEARING`
> (`|G| >= N_CLAIM`), where `N_CLAIM` is the smallest n at which an exact two-sided binomial test at
> alpha = 0.05 has power >= 0.80 against a 0.75:0.25 split, computed by the instrument.
> **PRIMARY (P1)** is the band on `G = WITHHELD` pooled. **SECONDARY (P2)** is the band on each taxonomy
> category.
> **Round verdict, per arm, over the three BASE scales, on P1:** `GENERALISES_DECIDED` iff
> `DECIDED_LIKE_COMMITTED` at >= 2 of 3 AND no base scale reads `FENCE_SITTING` or `DECIDED_OPPOSED`;
> `GENERALISES_FENCE_SITTING` iff `FENCE_SITTING` at >= 2 of 3 AND no base scale reads
> `DECIDED_LIKE_COMMITTED`; otherwise `SCALE_DEPENDENT`, and no cross-scale sentence may be written — each
> scale is reported with its own band and its own n.
> **Cross-arm:** the two arms are scored identically and reported separately; `DIRECTION_INVARIANT` iff both
> arms reach the same round verdict, else `ARM_DEPENDENT`.
> `-it` cells read `NO_WITHHOLDING_TO_CLASSIFY` (their withheld sets are 0/0/1 of 82) and are reported as the
> committed-reference contrast only.
> A P2 category band may never override a P1 band; a `REPORTED_ONLY` group may never be cited for or against
> any verdict; a `BANDED` group may be cited per-cell only and never as a generalisation.
> Reported, NOT a gate check.

**What each round verdict means, said now:**

- **GENERALISES_DECIDED** — at the counter slot, the items the scorer calls "withheld" carry a margin as
  decided as, and pointing the same way as, the items where the model commits, at ≥2 base scales and in the
  arm concerned. The taxonomy's sentence extends beyond 9b-base fold, *as a statement about the labelled
  withheld set* (§5.4 caveat), and the drafts' "abstains / hedges / is unsure" vocabulary is wrong at those
  scales in the same way it is wrong at 9b.
- **GENERALISES_FENCE_SITTING** — the withheld items are mostly inside the near-tie band at ≥2 base scales.
  Then withholding *is* fence-sitting there, and the taxonomy's reading is a 9b-base-fold property, not a
  property of withholding.
- **SCALE_DEPENDENT** — no cross-scale sentence. Every affected draft line becomes per-scale, with both
  columns printed.

### 5.8 What result would mean the 9b finding does NOT generalise — enumerated, so it cannot be renegotiated

1. **Fold, ≥2 base scales `FENCE_SITTING`** ⇒ `GENERALISES_FENCE_SITTING`. The taxonomy's "withholding is not
   fence-sitting" must be restricted, in every draft, to 9b-base fold.
2. **9b-base LISTEN reads `FENCE_SITTING` or `DECIDED_OPPOSED`** ⇒ the finding does not survive a change of
   push direction at its own scale. It is then a **fold-arm** statement, not a withholding statement, and
   must be written that way even if the fold arm generalises across scale.
3. **`SCALE_DEPENDENT`** ⇒ the sentence may be stated only at the cells that band
   `DECIDED_LIKE_COMMITTED`, with the others' bands printed beside it.
4. **`ARM_DEPENDENT`** ⇒ every statement carries its arm.
5. **The P2 leg fails at its own cell**: if 9b-base fold UNC bands as anything other than
   `DECIDED_LIKE_COMMITTED` under the frozen rule, that is a **re-derivation failure of the taxonomy's own
   number** (§4.3 catches it offline, before any GPU) and the round stops there.
6. **The structural ceiling, registered now and true under every outcome.** 9b-base fold UNC is n = 20 =
   **BANDED, not CLAIM_BEARING**, and UNC is n = 0 at 2b and n = 1 at 27b. **Therefore no outcome of this
   round licenses a scale-general statement about genuine uncertainty.** The best available result is a
   scale-general statement about the *labelled withheld set* (P1) plus a per-cell statement about UNC at 9b.
   This is a property of the phenomenon, not of the budget, and no amount of extra compute on this family
   fixes it.

### 5.9 Hard stops, evaluated before any new number is read

1. **TAXONOMY_NOREPRO** (§4.3) ⇒ STOP, offline, before any GPU spend.
2. **REPRO_FAIL** (§1.3, §5.6.1) ⇒ discard the run, quote no number.
3. **NOT_COMPARABLE** (§5.6.2) or **unadjudicated residual** (§5.6.4) ⇒ report, do not band.
4. **CONTESTED** (§1.3) ⇒ both readings persist as separate artifacts; an isolated-reader item-level hand-read
   adjudicates, exactly as the 27b-it drift contest was handled.

---

## 6. Power — the round's main methodological risk, handled before the data exists

Per-category counts are known in advance from `TAXONOMY_withholding.md`'s elicited-slot table, because the
categories are already assigned on committed labels. So the power problem can be, and is, settled now.

### 6.1 The tiers

| tier | n | what it may do |
|---|---|---|
| **REPORTED_ONLY** | `n < MIN_N = 8` | numbers printed; **no band**; may not be cited for or against anything. |
| **BANDED** | `8 <= n < N_CLAIM (≈30)` | band assigned; citable **per cell only**, always with n and the exact binomial p beside it; may **not** carry a cross-scale or cross-category generalisation. |
| **CLAIM_BEARING** | `n >= N_CLAIM (≈30)` | may carry a generalisation, subject to the §5.7 quorum. |

`N_CLAIM` is the smallest n at which an exact two-sided binomial test (α = 0.05) has ≥ 0.80 power against a
0.75:0.25 sign split; the instrument computes it. At n = 20 that power is ≈ 0.62, at n = 25 ≈ 0.72, at n = 30
≈ 0.80 (normal approximation, §13).

### 6.2 The minimum-n table — every cell classified in advance

Elicited slot, both arms, from `TAXONOMY_withholding.md` §"Elicited slot, all twelve cells".
**T2 = CLAIM_BEARING, T1 = BANDED, T0 = REPORTED_ONLY.**

| cell | CONF | UNC | AGREE | THIRD | OFFTGT | NUM | FMT | MISS | **WITHHELD (P1)** | **COMMITTED** |
|---|---|---|---|---|---|---|---|---|---|---|
| 2b-base fold | **39 T2** | 0 T0 | 0 T0 | 6 T0 | 4 T0 | 1 T0 | 1 T0 | 0 T0 | **51 T2** | **31 T2** |
| 2b-base listen | **37 T2** | 0 T0 | 0 T0 | 4 T0 | 4 T0 | 2 T0 | 0 T0 | 0 T0 | **47 T2** | **35 T2** |
| 9b-base fold | 5 T0 | **20 T1** | 4 T0 | 3 T0 | 5 T0 | 1 T0 | 0 T0 | 0 T0 | **38 T2** | **44 T2** |
| 9b-base listen | **19 T1** | **13 T1** | 0 T0 | 3 T0 | 1 T0 | 1 T0 | 0 T0 | 0 T0 | **37 T2** | **45 T2** |
| 27b-base fold | 1 T0 | 1 T0 | 0 T0 | **14 T1** | **10 T1** | 3 T0 | 3 T0 | 0 T0 | **32 T2** | **50 T2** |
| 27b-base listen | 1 T0 | 0 T0 | 0 T0 | **11 T1** | **10 T1** | 6 T0 | 0 T0 | 0 T0 | **28 T0→see note** | **54 T2** |
| all -it cells | — | — | — | — | — | — | — | 1 T0 (27b-it fold) | **0–1** `NO_WITHHOLDING_TO_CLASSIFY` | **81–82 T2** |

*Note on 27b-base listen:* withheld = 28, which is **below N_CLAIM ≈ 30** — so the smallest base cell in the
round is `BANDED`, not `CLAIM_BEARING`, on the primary. Registered consequence: **the listen-arm round verdict
can reach quorum on at most 2 of 3 CLAIM_BEARING scales (2b 47, 9b 37) plus one BANDED scale (27b 28).** The
§5.7 quorum is "≥2 of 3 with no contradiction", so a listen-arm verdict remains reachable, but if 27b-base
listen is the scale that would break a no-contradiction clause, its BANDED status must be printed with it.
This is stated now so it is not discovered as a convenience later.

### 6.3 What the table already settles, before the run

1. **CLAIM_BEARING per-category cells in the entire round: two.** 2b-base fold CONF (39) and 2b-base listen
   CONF (37). Every other category cell at every scale is BANDED or REPORTED_ONLY.
2. **UNC — the only genuine-uncertainty category — is CLAIM_BEARING nowhere**, BANDED at exactly two cells
   (9b-base fold 20, 9b-base listen 13), and REPORTED_ONLY or absent everywhere else. §5.8.6.
3. **The cell that produced the motivating result is BANDED** (n = 20, power ≈ 0.62). Even a perfect result
   there is a per-cell result.
4. **The cell the taxonomy already dismisses stays dismissed by rule**: 9b-base fold CONF n = 5 is
   REPORTED_ONLY and cannot be cited in either direction — the same conclusion the taxonomy reached by
   judgement ("n=5 carries nothing"), now reached by a threshold frozen in advance.
5. **P1 is the only well-powered comparison at every base cell** (withheld 51/47/38/37/32/28 against
   committed 31/35/44/45/50/54). This is why §5.4 makes P1 the primary and P2 the secondary, and it is a
   consequence of the power table, not a preference.
6. **The pre-registered n floor for a *new* claim about any category is `N_CLAIM`.** If, after the run,
   someone wants a category-level generalisation, the honest route is a larger family, not a re-banding.

---

## 7. Honesty gate

1. **The motivating result was found before this design existed**, on 2026-07-28, in the taxonomy pass, at
   the single cell where the artifact happened to exist. This round exists because that cell is one of
   twelve, not because a hypothesis needs testing.
2. **Nothing here has been evaluated against data.** No listen-arm margin and no 2b/27b margin exists in any
   file in the repo (verified 2026-07-28: exactly four `family_cave_diagnose_*.json`, all 9b, all fold). The
   `--arm` flag does not exist in the code. No launcher for this round exists. This pass wrote exactly one
   file and ran nothing.
3. **Disclosure — the one check I ran after freezing the constants.** With `NEARTIE_THR = 0.5`,
   `SAME_DEC_MAX = 0.10` and `SAME_MED_MAX = 0.5` all lifted verbatim from committed constants, I then
   evaluated the rule against the taxonomy's committed 9b-base fold numbers: UNC `decided_frac` = 14/20 =
   0.70 vs COMMITTED 31/44 = 0.7045 (`d_dec` = −0.005), `d_med` = 0.65 − 0.73 = −0.08, sign majorities both
   positive ⇒ **`DECIDED_LIKE_COMMITTED`**. **The rule reproduces the taxonomy's own reading.** That is
   deliberate in the weak sense (a rule that overturned the only known cell would be a rule about that cell)
   and undeliberate in the strong sense (both constants were fixed by provenance before the arithmetic). It
   is disclosed here rather than buried, and §4.3 turns it into a blocking offline gate so it is a
   *measurement*, not a claim.
4. **A second thing the committed numbers already half-determine, disclosed.** At 9b-base fold, P1's
   `decided_frac` is pinned to 0.711–0.737 by the taxonomy's per-category near-tie counts, against COMMITTED
   0.7045 — inside `SAME_DEC_MAX` either way. So P1 at that one cell is largely pre-determined; its median
   and sign majority are not committed anywhere and are genuinely unread. **The round's test is the other
   five base cells and the entire listen arm**, not 9b-base fold.
5. **Standing incentive, named.** A `GENERALISES_DECIDED` result is the interesting one and would license a
   scale row on a figure; a `FENCE_SITTING` or `SCALE_DEPENDENT` result mostly generates corrections. The
   round is built so the dull outcomes are equally reportable: each has its own band, its own selftest
   falsifier (§4.2.6), and its own §5.7 sentence, and §5.8 enumerates them before the data exists.
6. **Residual bias the design cannot remove.** (a) I have read the taxonomy, so I know the 9b answer;
   mitigated by borrowing both band constants and by making the instrument claim-blind (§4.4). (b) The power
   table is computed from counts I have seen — but the counts are committed facts about label distributions,
   and the tiers are set by a power calculation on conventional α/power/effect-size, not by which cells I
   want to be citable; the visible cost is that the round's own motivating cell lands in the middle tier.
   (c) `NUMERIC_TOL` is new (§1.3). (d) The category dict is hand-adjudicated by one reader (the taxonomy
   pass) and is imported, not re-adjudicated, here.
7. **Grounding, unchanged from the repo standard.** New numbers are load-bearing, so: isolated-reader
   item-level re-derivation from raw `result.items[]` at each base cell; a blind check of the join (`q`
   coverage, `(correct, Wstar)` equality) at each cell; and `latent_skeptic` triage on the round verdict
   (README entry ritual step 2). No hand-label spot-check is owed here — nothing in this round is a decoded
   string; the readout is teacher-forced log-probabilities.
8. **C3 respected.** `DESIGN_foldlisten_mechanism.md` C3 forbids a teacher-forced margin as a standalone
   *adoption* metric. This round makes **no adoption claim**: it measures a margin and joins it to a label,
   and every verdict is phrased as a statement about the margin's decidedness and direction at a slot. R5
   carries the module's own `CONTENT_CAVES` category unchanged and forbids quoting it as this round's result.

---

## 8. Interlock with `DESIGN_elicit_context.md`

### 8.1 Are the diagnose margins contaminated by the elicit-context defect? **No — and here is why, with lines**

The defect (`DESIGN_elicit_context.md` §0) originates at **`controls/foldlisten_judge.py:425`**:

```python
pg = prior_gen.strip() or "(no answer)"      # the UNTRUNCATED counter_gen, spliced in as the 4th turn
```

— a **model generation** inserted into a prompt. `controls/family_cave_diagnose.py` has no such splice and
cannot have one:

| fact | line |
|---|---|
| both prompts are built by `push()` from `rlhf_differential._helpers` | `family_cave_diagnose.py:200, 214, 215` |
| `push(q, C, challenge)` is a pure template: `raw(f"Q: {q}\nA: {C}.\nQ: {challenge}\nA:")` / the 3-message chat analogue | `rlhf_differential.py:169-173` |
| its three slots are filled from the **family JSON** (`q`, `correct`, `Wstar`) and two frozen module constants (`PUSH["counter"]`, `NEUTRAL`) | `family_cave_diagnose.py:205`; `job_truthful_flip.py:50-52` |
| the instrument issues **no generation at all** — every model call is a forward pass under `no_grad` | `family_cave_diagnose.py:219-222`; `rlhf_differential.py:175-182` |
| there is no `prior_gen`, no `counter_gen`, no `elicit_gen`, no `isolate_span`, no 4th turn anywhere in the file | grep-verifiable over `family_cave_diagnose.py` |

**The diagnose instrument builds its own prompts from templates and never sees a model generation.** `M0`,
`Mc_neutral` and `Mc_counter` are therefore structurally immune to the elicit-context contamination. This is
the same conclusion `JOIN_withhold_vs_fold.md:302-303` reaches from the other direction when it names the
diagnose margin as the *uncontaminated* off-policy measure of item-level uncertainty.

### 8.2 Three things that ARE affected, said in the same breath

1. **The grouping variable is contaminated even though the measurement is not.** The withheld/committed
   partition and the categories come from `faithful_elicit`, produced in contaminated 5-turn contexts on
   82/82 base items. At 27b the taxonomy concludes the contamination *manufactures* the category (THIRD 14 +
   OFFTGT 10 of the 32 fold withholds; five worked items answering the model's own invented question). So:
   **the margins are clean, the partition is not.** Registered consequence: if `DESIGN_elicit_context`
   returns `DEFECT_MATERIAL` or `DEFECT_PARTIAL` at a base scale, that scale's P1 and P2 are **re-evaluated
   under the span labels** — the same instrument, the same frozen rule, a different partition. Not
   renegotiated; recomputed. The margin artifacts themselves do not need to be re-run, which is the practical
   payoff of §8.1.
2. **S4 (§2) is contaminated by construction and by design.** It re-tokenizes the stored contaminated
   `elicit_prompt`. That is deliberate — it is the slot the label came from — and §2.3 forbids citing it as
   the within-slot version of the claim until it is read in the span register too.
3. **Nothing else.** `NEARTIE_THR`, the signing, the power tiers and the repro gate are all properties of the
   margin or of committed counts, none of which the defect touches.

### 8.3 Co-scheduling — both rounds need base cells at all three scales

They do, and this round's jobs are forward-only and cheap, so **they ride the same boxes**.

| `DESIGN_elicit_context.md` box | adds from this round |
|---|---|
| **A** (≥40 GB, `fl_9bit_anchor5` → 9b-base → 2b-base ext2) | diagnose `--arm both` at 2b-base, 2b-it, 9b-base, 9b-it (4 loads); + S4 pass for those cells |
| **B** (≥80 GB, 27b-base ext2) | diagnose `--arm both` at 27b-base, 27b-it (2 loads); + S4 pass |
| **C** (P3, optional -it tier) | **not needed** — this round's -it diagnose cells ride A and B regardless of whether C runs |

**Ordering on the box, frozen:** the primary diagnose pass runs **first**, immediately after the selftests
and before any foldlisten cell — it needs nothing but `verifier_family_ext2.json`, and banking it first means
a cap kill costs the expensive job, not the cheap irreplaceable one. The **S4 pass runs last**, after the
foldlisten cells have written their summaries on the same box (which is what `--elicit-from` reads), and is
explicitly sacrificial.

This ordering is a request to `DESIGN_elicit_context`'s runners, whose cell order that design freezes; it is
therefore **OPEN DECISION D-1**, not a unilateral change.

**A concrete launcher gap, verified this pass.** `lambda_run.sh`'s tiny-criticals-first fetch is
`out/*summary*.json` + `out/*.log` (`:213-214`) and its validity check requires a `*summary*.json` to parse
(`:217-219`). Diagnose artifacts are `out/family_cave_diagnose_*.json` — **not matched**. They arrive only on
the best-effort full-`out/` recursive fetch (`:227`). Committed precedent that this works: `results_absdecode_ext2/`
was fetched exactly this way. Co-scheduling removes the exposure entirely, because the foldlisten summaries
satisfy `SUMOK` and the diagnose JSONs ride the full fetch. **No `lambda_run.sh` edit is registered**
(D-3 offers the alternative).

`lambda_run.sh` otherwise needs **no** edit: it already ships `controls/family_cave_diagnose.py` (`:118`),
`controls/cave_doubt_decollide.py` (`:128`, the `strip_polarity` / `faithful_cave` source),
`job_truthful_flip.py` and `rlhf_differential.py` (`:92-93`), `controls/verifier_family.py` (`:116`) and
`verifier_family_ext2.json` (`:120`).

### 8.4 Relationship to the run in flight

`DESIGN_neutral_elicit.md`'s boxes launched 2026-07-28 and have not landed (`results_foldlisten_nelicit_*/`
absent; both pollers present). This round **does not depend on it**: S4a reads the six committed ext2
summaries. S4b (the neutral-elicited slot) is conditional on it and reads `ARM_ABSENT` if absent. Nothing in
this round should delay, abort, or be delayed by that run.

---

## 9. Cost and box plan

### 9.1 Work units and marginal compute

One model-cell-arm = 82 items × **8 short forward passes** (2 plain forwards at `:219-222`; 6 `num_lp`
teacher-forcings — `single`+C, `single`+W, `neutral`+C', `neutral`+W', `counter`+C', `counter`+W'). No
decoding, no sampling, no KV reuse needed. Prompts are ~30–60 tokens. **Compute per cell-arm is ~1–3 minutes
at 9b; the model load dominates**, which is why `--arm both` shares one residency (§1.2.2).

S4 adds 2 `num_lp` teacher-forcings per item per arm on a ~200–500-token prompt, plus one extra model load
per model (the model was freed after the primary pass and after the foldlisten cell).

VRAM unchanged from the committed diagnose runs. Artifact growth: one extra `*_listen.json` per model-cell,
~500 KB each.

### 9.2 Pace basis — same basis as the sibling designs

**Measured:** nothing. There is **no committed s/item for `family_cave_diagnose.py`** anywhere in the repo.
The only anchor is that `run_absdecode_ext2_9b.sh` fitted two `family_topk_shift` passes (22 + 82), one
diagnose pass (82) and one `family_generate_judge` pass (82 items at 160 decode tokens — by far the dominant
term) inside a single box's cap (`results_absdecode_ext2/out/run_detached.log`, all four `exit=0`).
**Inherited:** the 27b foldlisten pace ~89 s/record on H100 PCIe / ~4.3 h per 164-record cell
(`docs/lambda-gpu-access.md:41-42`, commit `fd2154b`), used only to bound model-load time. **Everything
below is an estimate** and is flagged in §13.

### 9.3 Boxes — registered lean is co-scheduled

| box | added cells | loads | est. added wall | $/hr | **est. added $** |
|---|---|---|---|---|---|
| **A** (rides `DESIGN_elicit_context` box A, ≥40 GB) | primary: 2b-base, 2b-it, 9b-base, 9b-it × {fold, listen}; then S4 for those four | 4 + 4 | **40–70 min** | 1.99 | **1.3 – 2.3** |
| **B** (rides box B, ≥80 GB) | primary: 27b-base, 27b-it × {fold, listen}; then S4 for those two | 2 + 2 | **45–80 min** | 3.29 PCIe / 4.29 SXM5 | **2.5 – 4.4 PCIe / 3.2 – 5.7 SXM5** |

**Marginal cost of this round if co-scheduled: ≈ $4–8.**

**Cap arithmetic, which matters more than the dollars.** Box A's registered `REMOTE_TIMEOUT` is 19800 s
(5.5 h) against a 2.8–4.0 h foldlisten estimate; +1.2 h leaves ~0.3–1.5 h slack — **tight**. Box B's is
25200 s (7 h) against ~4.8–5.1 h PCIe or ~1.6 h SXM5; +1.3 h leaves ~0.6–0.9 h on PCIe and ~4 h on SXM5.
**Both boxes reinforce `DESIGN_elicit_context` §9.3's own recommendation to take SXM5 for box B**, and box A
needs its cap re-checked before launch (D-2).

**Standalone fallback**, if co-scheduling is refused (D-1):

| box | cells | est. wall | cap | instance floor | est. $ |
|---|---|---|---|---|---|
| **D** | 2b-base, 2b-it, 9b-base, 9b-it × {fold, listen} + S4 | 1.0–1.5 h | `10800` (3 h) | ≥40 GB, ≤$10/hr, skip `gh200`/`b200` | **2 – 3** |
| **E** | 27b-base, 27b-it × {fold, listen} + S4 | 1.0–1.5 h | `14400` (4 h) | ≥80 GB, ≤$5.50/hr, skip `gh200`/`b200` | **3.3 – 6.4** |

**Standalone total: $5–10**, plus a second trip through the capacity queue and the §8.3 fetch exposure.

### 9.4 Combined cost with `DESIGN_elicit_context.md`

| scenario | elicit-context | this round | **combined** |
|---|---|---|---|
| box B on **SXM5**, no optional -it tier | $22–29 | $4–8 | **$26–37** |
| box B on **PCIe**, no optional -it tier | $31–41 | $4–8 | **$35–49** |
| either, **plus** elicit-context's optional -it tier | +$10–18 | (unchanged; this round's -it cells already ride A/B) | **+$10–18** |
| both rounds standalone | $22–41 | $5–10 | **$27–51** |

Sanity anchor: Phase B ran six full foldlisten cells for ~$44 (audit-log reconstructed, commit `c0900e4`).

### 9.5 Budget — must be re-reconstructed before launch

Cap is **$950** (`docs/lambda-gpu-access.md:54`). The most recent reconstruction —
`GET /api/v1/audit-events`, launch↔terminate paired per `resource_lrn` and priced from
`GET /api/v1/instance-types`, run 2026-07-28 and recorded in `docs/drafts/STATUS_neutral_elicit.md` — gave
**$585.47 since project start (193.2 GPU-h), headroom $364.53**, and noted it is ~$149 *above* the stale
committed tally of ~$436.

**That reconstruction predates the two `DESIGN_neutral_elicit.md` boxes that are billing right now.** It must
be re-run from `GET /api/v1/audit-events` before this round launches — as `docs/lambda-gpu-access.md:52`
requires in any case, and as the $149 discrepancy shows is not a formality. Also confirm `INSTANCE_COUNT 0`
after each box, and on launcher death SSH-fetch from the live box **before** terminating
(`docs/lambda-gpu-access.md:58-59`).

### 9.5.1 Launchers

`run_diagnose_arms_9b2b.sh` and `run_diagnose_arms_27b.sh` — ~25-line copies of `run_absdecode_ext2_9b.sh`
with its selftest-hard-exit preamble (`:13-16`) kept verbatim and one `family_cave_diagnose.py --arm both`
invocation per model-cell, plus the S4 invocations at the end. If co-scheduled (D-1), these invocations are
**inserted into** the elicit-context runners instead, which do not yet exist — a co-authoring instruction,
not an edit to a committed file. Pollers: copies of `run_poll_launch_nelicit_{2b9b,27b}.sh`, which already
carry this workstation's Linux-path / `python3` fixes and `SSH_KEY_NAME=latent_verify_hal_20260721`.

**Tags, frozen** — fold tags are **identical to the committed ones**, so the §10 diff is same-filename:

| cell | fold tag (= committed filename) | listen tag |
|---|---|---|
| 9b-base | `vfam_ext2_9bbase` | `vfam_ext2_9bbase_listen` |
| 9b-it | `vfam_ext2_9bit` | `vfam_ext2_9bit_listen` |
| 2b-base | `vfam_ext2_2bbase` | `vfam_ext2_2bbase_listen` |
| 2b-it | `vfam_ext2_2bit` | `vfam_ext2_2bit_listen` |
| 27b-base | `vfam_ext2_27bbase` | `vfam_ext2_27bbase_listen` |
| 27b-it | `vfam_ext2_27bit` | `vfam_ext2_27bit_listen` |

Result dirs: the co-scheduled boxes' own dirs (`results_foldlisten_elicitctx_{2b9b,27b}/out/`), or
`results_diagnose_arms_{2b9b,27b}/out/` standalone. The two repro diffs are cross-dir, same-filename, against
`results_absdecode_ext2/out/` (9b-base) and `results_itreadout_modelw/out/` (9b-it).

---

## 10. Execution order — each step blocks the next

1. **§4.3 offline re-derivation of the taxonomy's 9b-base fold table under the frozen rule** (CPU, $0).
   Committed artifact. Mismatch ⇒ **STOP**, no GPU spend.
2. **Model-free selftests** on the box: `family_cave_diagnose.py`, `family_cave_repro_diff.py`,
   `withhold_margin_reader.py`. Hard-exit on failure.
3. **The two repro anchors first** — `--arm fold` at 9b-base and 9b-it — through
   `controls/family_cave_repro_diff.py` against the committed artifacts. `BYTE_IDENTICAL` or
   `NUMERICALLY_EQUIVALENT` (§1.3); anything else ⇒ **STOP**, substrate or stack drift, not a finding. The
   decision persists as a committed JSON per cell.
4. **§5.6 structural invariants** on every cell: coverage 82/82, partition sums, zero unadjudicated
   residuals, -it withheld ≤ 1, S4 round-trip.
5. **Only now** read P1 and P2 and apply §5.7. R1–R5 and S4 are read at the same time and reported beside
   them, never instead of them.
6. **H3 grounding** (§7.7), then **`latent_skeptic` triage** on the round verdict (README entry ritual
   step 2).
7. If and when `DESIGN_elicit_context` returns a base-scale verdict: **recompute** P1/P2 under the span
   labels (§8.2.1), same instrument, same rule, and print both partitions.

---

## 11. What is at stake — the specific claims, named before the run

**The taxonomy's own reading.** `docs/drafts/TAXONOMY_withholding.md:137-140` ("Withholding is not
fence-sitting… their margin distribution is statistically indistinguishable from the items where the model
does commit") and `:148-149` reading #2 ("the honest sentence is that the model declines to say an answer it
is, distributionally, still holding"). This round either extends it to five more base cells and a second push
direction, or restricts it — explicitly, in writing — to 9b-base fold.

**The margin-flow figure and its caption**, `docs/drafts/figs/fig_margin_flow_9b_caption.md`:
- `:61-63` — "The 38 items where base withholds aloud are not fence-sitting. On those items the margin
  favoured C on 29 and W\* on 9." That is the **P1** statement at 9b-base fold (29:9 over all 38), distinct
  from the taxonomy's **P2** statement (17:3 over UNC only). Both are in scope; the 29:9 sign split is a
  step-1 re-derivation target.
- `:38-42` Scope — "9B only, fold only. No diagnose artifact exists for this family at 2B or 27B, and none
  for the listen cell, so this figure cannot be read as a scale or a direction result." This round retires
  that paragraph or confirms it must stay, and licenses (or refuses) a 3-scale, 2-arm rebuild of the figure.

**Two committed UNAUDITABLE entries.** `docs/drafts/GROUNDING_notes_numbers.md:26` ("The margin-layer version
of this ratio is UNAUDITABLE: no diagnose artifact exists for the listen cell at any scale") and `:156-157`
(the same, restated in the UNAUDITABLE section). The listen arm makes both auditable — in whichever direction
they come out.

**The drafts' abstention framing.**
- `docs/drafts/DARWIN_post1_user_snapshot_270726_3.md:168` — "the 38 it withholds are not fence-sitting".
- `docs/drafts/DARWIN_post1_user_extrapolation.md`, the whole "Chat models always answer" section (`:22`),
  and `:94-97`, `:111`, `:190` ("tuning didn't reduce expressed uncertainty… it deleted it").
- `docs/drafts/NOVELTY_boundary_post1.md:10-12`, `:21`, `:23`, `:149`; `docs/drafts/POST1_v7_draft.md` (the
  withhold section); `docs/drafts/EXHIBITS_post1_grounded.md:255-258`.
- `DESIGN_modelderived_wstar.md`'s base-abstention framing (`r_MW` / `ABSTENTION_ROBUST`), which is
  downstream and is named so it is not forgotten.

The word at stake is **"abstains"**. If the withheld set is decided at every base scale, "abstention" is the
wrong word in every one of those lines and the honest word is "declines to say". If it is fence-sitting at 2b
and 27b but decided at 9b, then the drafts' single framing is right at two scales and wrong at one — the
**opposite** split from the taxonomy's category story, and a genuinely awkward result that this design
commits to reporting.

**What it could license.** A listen-arm margin column and a 3-scale row on the margin-flow figure; a margin
twin for `make_fig_withhold_slope.py`'s withhold slopegraph; and the retirement of both UNAUDITABLE entries.

**What it could overturn.** `fig_margin_flow_9b_caption.md:61-63` if 29:9 does not re-derive; the taxonomy's
distributional section if the frozen rule bands 9b-base fold UNC as anything other than
`DECIDED_LIKE_COMMITTED` (§4.3, offline, before any spend).

---

## 12. Scope limits — what this round does NOT settle

- **It does not settle whether withholding is *caused* by anything.** There is no intervention, no ablation,
  no steering. It is a joint distribution of a margin and a label.
- **It does not settle what the model "believes."** `Mc` is a teacher-forced content margin at a slot.
  `DESIGN_foldlisten_mechanism.md` C3 forbids margin-as-adoption and C2 forbids first-token readouts; this
  round makes no adoption claim and does not use the RA readout (§7.8, §5.5 R4–R5).
- **It does not settle whether base "genuinely expresses uncertainty."** The categories are imported; the
  spans are not re-read.
- **It cannot make a scale-general statement about genuine uncertainty**, under any outcome (§5.8.6): UNC is
  n = 0 at 2b and n = 1 at 27b.
- **It does not settle the elicit-context defect** (`DESIGN_elicit_context.md`) or the push-attribution
  question (`DESIGN_neutral_elicit.md`). §8.2 states the dependency in one direction only.
- **It does not fix the scorer defects the taxonomy found**: the 81-item free-reply lexicon gap, the 63-item
  `tiebreak_unresolved` abstention, or the `"persia"` alias. All remain owed.
- **The free-reply slot is out of scope entirely.** The taxonomy's own verdict is that its withheld counts
  are non-comparable across scales until the lexicon gap is closed; banding them would launder that.
- **It does not extend the family.** ext2 82 only. The n=22 and n=74 families are untouched and get no
  diagnose arms.
- **It does not change any threshold, category or output of `family_cave_diagnose.py`'s own decision layer**
  (`MARGIN_KEEP`, `MARGIN_FAITHFUL`, `MIN_FAITHFUL`, `CAVE_RISE_THR`, `NO_CAVE` / `FIRST_TOKEN_ONLY` /
  `CONTENT_CAVES`). They are carried through unchanged and reported (R5).
- **It does not retroactively change anything.** The committed artifacts stay as they are and stay
  reproducible under the default flag.

---

## 13. OPEN DECISIONS — calls only the researcher can make

Each must be closed **before launch**, not after data. Nothing below is chosen silently.

- **D-1 — co-schedule or standalone?** Registered lean: **co-schedule** onto `DESIGN_elicit_context`'s boxes
  A and B, primary pass first (before the foldlisten cells), S4 last. Saves ~$3 and, more importantly, removes
  the §8.3 fetch exposure and a second capacity queue. Cost: it inserts ~1.2 h into box A's and ~1.3 h into
  box B's caps, and it requires editing that design's frozen cell order. **Consequence of deferring:** this
  round cannot be launched at all until it is called.
- **D-2 — box A's cap.** With this round riding it, box A is 2.8–4.0 h + 1.2 h against a 19800 s (5.5 h)
  registered cap. Raise to `25200` at launch (the `STATUS_neutral_elicit.md` precedent of overriding on the
  command line rather than editing a frozen script), or drop S4 from box A. A cap kill on box A costs
  `fl_2bbase_ext2` — the highest-withhold cell.
- **D-3 — the `lambda_run.sh` fetch gap (§8.3).** Registered: **no edit**; rely on the full-`out/` fetch, per
  the `results_absdecode_ext2` precedent, and prefer co-scheduling. Alternative: a one-line change to the
  tiny-criticals glob. This is the only place where the no-edit rule costs something real.
- **D-4 — the two-file / one-`arm`-key asymmetry (§1.2.10).** Registered: fold keeps the committed filename
  and the top level gains `arm`, so the fold artifact self-describes at the cost of literal file-level byte
  identity (values are still exactly reproduced, §1.3). Alternative: suppress the `arm` key and the prose
  change under `--arm fold` for literal byte identity, at the cost of a fold artifact that does not say which
  arm it is.
- **D-5 — should `--arm both` exist**, or should the two arms be two invocations? Registered: **both**, purely
  to share one model residency (a 27b load is the dominant cost). It is the only structural change to
  `_measure_model`.
- **D-6 — where the frozen category dict lives.** Registered lean: add a two-line `if __name__ ==
  "__main__":` guard to `docs/drafts/taxonomy_withholding_rederive.py` (behaviour-preserving when run as a
  script) so `HAND_ELICIT` has **one** home that the reader imports. Alternative: copy the dict into
  `controls/withhold_categories.py` with a parse-equality assertion against the source — no edit, but two
  copies of a hand-adjudicated dict, which is exactly the failure mode that produces silent divergence. This
  is an edit to a committed provenance script and is therefore the researcher's call.
- **D-7 — S4 in or out?** Registered: **in, as P2/sacrificial**. It is the only thing that makes the claim
  within-slot (§2.3). Dropping it saves ~30–50 min and ~$2 and leaves the round's central sentence a
  cross-slot join with a measured 46/82–35/41 agreement rate behind it.
- **D-8 — publication policy under `SCALE_DEPENDENT`.** Do the drafts carry per-scale bands inline, or does
  the withholding framing get pulled back to 9b-base fold with a footnote? Deciding in advance removes the
  temptation to let the answer depend on which is less work.
- **D-9 — `NUMERIC_TOL = 0.10` (§1.3).** The one genuinely new banded-adjacent constant. Confirm, tighten, or
  demand exact byte identity (which risks a `REPRO_FAIL` for pure hardware reasons on a box that is not the
  original).
- **D-10 — budget.** Re-reconstruct spend from `GET /api/v1/audit-events` (the $364.53 figure predates the
  in-flight boxes) and confirm SXM5 vs PCIe for box B.

---

## 14. Flags — where I am guessing rather than reading a committed value

- **There is no committed s/item for `family_cave_diagnose.py`.** Every wall-clock and dollar figure in §9 is
  an estimate built from (a) the fact that `run_absdecode_ext2_9b.sh`'s four passes fitted one box's cap and
  (b) the 27b foldlisten pace used only to bound model-load time. The model-load estimates (~4 min at 2b/9b
  warm, ~8–12 min at 27b) are **not measured anywhere in this repo**.
- **`N_CLAIM ≈ 30`** is my normal-approximation-with-continuity-correction value (power ≈ 0.80 at n = 30,
  ≈ 0.72 at n = 25, ≈ 0.62 at n = 20). The instrument must compute the exact binomial value and that value
  governs; if it differs, §6.2's tier assignments shift and **27b-base listen (28) is the cell most likely to
  change tier.**
- **`NUMERIC_TOL = 0.10 nats`** is new. Its provenance is an analogy to one committed bf16 reproduction
  (`docs/lambda-gpu-access.md:176-177`, |Δ| = 0.07 on a Δ_syc of ~4.5), not a measurement on this quantity.
- **The 9b-base fold P1 `decided_frac` bracket 0.711–0.737** (§7.4) is my arithmetic on the taxonomy's
  per-category near-tie counts, assuming the two "—" entries (AGREE, OFFTGT) mean zero and that the
  unlisted NUM item is unknown. It is not read off an artifact.
- **The category counts in §6.2 are read from `TAXONOMY_withholding.md`'s table**, not re-derived by me this
  pass. §4.3 and the reader's own recomputation of the lexical categories turn them into artifacts before any
  band is assigned.
- **The claim that the diagnose instrument never sees a generation (§8.1) is from reading the code path**
  (`family_cave_diagnose.py:186-273`, `rlhf_differential.py:155-183`), which I did this pass. It is
  grep-checkable and the §4.2 selftest does not currently assert it; adding a "no `generate` in this module"
  assertion would be theatre, but the claim is a code reading, not a run.
- **The S4 re-tokenization argument** (that `ptext`-stored prompts round-trip with `prepend_bos=False`) is an
  argument from `foldlisten_judge.py:442` using `skip_special_tokens=False` plus
  `rlhf_differential.py:160-161`'s `bos` parameter. **It has not been executed.** §2.2 makes it a runtime
  assertion rather than an assumption, and a failure there costs S4, not the primary.
- **-it withheld counts (0/0/1 fold, 0/0/0 listen)** are the taxonomy's, which re-derived them from the
  committed summaries; I did not re-derive them this pass.
- **Working tree state.** This pass wrote exactly this file. No instrument was modified, no launcher written,
  nothing run, no GPU touched, no artifact produced. The `--arm` flag **does not exist in the code**, and
  neither of the two new instruments in §4.4 exists.
