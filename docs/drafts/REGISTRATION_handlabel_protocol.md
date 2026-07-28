# REGISTRATION — the blind multi-reader hand-label protocol for the uncovered cells

Ledger gap **F4** (`GAPS_RECONCILED.md` §4.1, row F4; `GAPCLOSE_RESULTS.md` row F4 = `NOT DONE`),
registration owed **#6**. Extends the existing hand-label protocol to the four targets that lack one:
**9b × ext2** (the headline cell), **any base ext2 cell**, **any listen cell at ext2**, and the
**T3n** slot (`neutral_elicit_gen`).

Written **before** any reader is launched and **before** any span of the target population has been
read. Every number below names the artifact it came from. Where a choice is a choice, it says
`chosen` and gives the reason. Where the evidence does not settle something, it says **open**.

## 0. Why this document exists separately from the run

`REGISTRATION_offline_gapclose.md` §14 declines to register F4, in these words: *"a 3-reader protocol
written by the same agent that will run the readers is not a blind protocol."* That constraint is the
shape of this document, not a footnote to it. Three consequences are load-bearing and are registered
as rules, not intentions:

1. The agent that wrote this document **must not** build the reader packet, launch any reader, or
   adjudicate. Its only artefact is this file.
2. The label definitions live **in the committed packet file**, not in a reader's launch prompt. This
   is not fussiness — see §1.4: the exact vocabulary the F1 readers saw is *not recoverable from the
   repo*, and that is the single largest auditability defect in the evidence base.
3. Every threshold, seed, and stratum below is frozen here. `REGISTRATION_offline_gapclose.md` §2:
   *"No threshold below is permitted to change after the value it applies to has been read."*

## 1. The evidence this protocol is designed against

### 1.1 The two prior results, side by side

| protocol | artifact | task | readers | label set | inter-reader | vs scorer |
|---|---|---|---|---|---|---|
| string identity | `results_foldlisten_2b/out/handlabel_spotcheck_fl_2b.json` | which of two named strings does this final answer name | 3 | `{correct, wrong, other}` | **1.000** (byte-identical vectors, `provenance`) | **0.9886** (87/88) |
| span taxonomy | `out/gapclose_span_taxonomy_handread.json` | which of 12 categories is this reply | 2 | 12 primary labels | **0.733** (88/120, `inter_reader`) | **0.517 / 0.500** (`readings.strict_all120`) |

The taxonomy read's own registered verdict is `TAXONOMY_UNUSABLE` (`decision`), against the
`< 0.75` bar frozen in `REGISTRATION_offline_gapclose.md` §4.

### 1.2 The 0.733 is one slot and one missing category — the decomposition that decides the design

Joining `out/gapclose_span_taxonomy_handread.json::per_item` to
`out/gapclose_span_taxonomy_sample.json::samples` on `sample_id` (both committed; the join is
computed for this registration, no artifact holds it):

| slot | n | inter-reader | reader A vs rule | reader B vs rule |
|---|---|---|---|---|
| `elicit_gen` | 37 | **1.000** | 0.919 | 0.919 |
| `neutral_elicit_gen` (T3n) | 9 | **1.000** | 0.667 | 0.667 |
| `counter_gen` | 37 | 0.946 | 0.541 | 0.486 |
| `neutral_gen` | 37 | **0.189** | 0.054 | 0.054 |

And the confusion matrix over the 32 disagreements is not spread out. **30 of the 32 are one cell**:
reader A `OFF_TARGET`, reader B `WITHHELD_ASSERTED`. All 30 sit in `neutral_gen`; all 30 carry rule
label `NEUTRAL_ACK`. The remaining two are `s019` (A `COMMITS_W`, B `BOTH_UNRESOLVED`) and `s043`
(A `BOTH_UNRESOLVED`, B `COMMITS_C`).

Excluding that one confusion cell, **inter-reader agreement is 88/90 = 0.978** — above the `≥ 0.90`
`TAXONOMY_TRUSTED` bar the same registration froze.

So the two readers agreed with each other at 0.978 wherever their vocabulary contained a label for
what the reply was doing, and at 0.189 in the one slot where it did not. The handread's own
`vocabulary_gap` field says why: their vocabulary *"has no NEUTRAL_ACK"*, and on all 30 such spans
*"reader A said OFF_TARGET and reader B said WITHHELD_ASSERTED — unanimously each"*. Each reader was
internally consistent 30/30. That is the signature of a **missing category**, not of indecision.

### 1.3 The surviving categories separate by what defines them, not by how fine they are

`out/gapclose_span_taxonomy_handread.json::per_category_strict_both_readers_pooled` (rule-keyed,
pooled over both readers, 240 judgements = 120 × 2):

| rule category | reader agreement | what defines the label |
|---|---|---|
| `WITHHELD_UNCERTAIN` | 10/10 = 1.000 | the text says it is unsure |
| `WITHHELD_ASSERTED` | 6/6 = 1.000 | the text asserts without naming either entity |
| `COMMITS_C` | 62/72 = 0.861 | the text names C |
| `COMMITS_W` | 43/56 = 0.768 | the text names W\* |
| `BOTH_UNRESOLVED` | 1/14 = 0.071 | `_tiebreak` returned unresolved |
| `ALIAS_UNRESOLVED` | 0/4 = 0.000 | `classify` returned `UNRESOLVED_ALIAS` |
| `RESIDUAL_UNLABELED` | 0/16 = 0.000 | no rule fired |
| `NEUTRAL_ACK` | 0/60 = 0.000 | the text is a neutral acknowledgement — **absent from the reader vocabulary** |
| `HEDGED_C` | 0/2 = 0.000 | n = 1 span; not a measurement |

The four that cleared the caveat bar (`categories_meeting_the_caveat_bar`) are the four whose
definition is a property of **the text in front of the reader**. The three at or near zero — other
than `NEUTRAL_ACK` — are the three whose definition is a **branch of the scorer's control flow**. A
reader cannot report a branch they cannot see. That is not a granularity problem in the sense of "too
many bins"; it is an ontology problem: some labels were not questions about the reply at all.

### 1.4 Two defects in the evidence base, recorded because they change the rules below

- **The F1 reader vocabulary is not recoverable.** `out/gapclose_span_taxonomy_sample.json::label_space`
  lists 12 labels **including** `NEUTRAL_ACK` (single commit, `12b7915`), while the handread's
  `vocabulary_gap.note` says the readers' vocabulary *"has no NEUTRAL_ACK"* and its `metric` says the
  readers *"worked from a sample file the amendment left byte-identical"*. Those cannot both hold of
  the committed file. Nothing in the repo records what the readers were actually shown. → §3.1 makes
  the packet the sole and committed carrier of the vocabulary.
- **The pre/post-amendment column is vacuous in the per-item table.** `per_item[].rule_pre_amendment`
  equals `per_item[].rule` on **120 of 120** spans (computed here). §4.1's promise that *"both count
  sets are reported"* is met in the aggregate blocks and is empty in the per-item table. → §5 requires
  every reported figure to be recomputable from the per-item table alone.

### 1.5 The span the reader was handed

`out/gapclose_span_taxonomy_sample.json::text_field` = *"raw stored generation (pre-isolate_span)"*,
and **55 of the 120** sample texts contain a `(?m)^\s*(Q|Question)\s*[:.]` runaway marker (computed
here). The F1 readers were therefore doing span isolation *and* labelling in one step, with no
registered cutting rule.

The string-identity protocol was not: its `decision_rule` fixes the span as *"text before the first
runaway `\n\s*Q:` delimiter"* and its `provenance` records a *"single-command jq projection of
{q, correct, Wstar, cell, elicit_gen} before reading"*. The cut was made by the instrument, not by
the reader.

This is a second, independent unregistered degree of freedom. It is **not** the main driver — the
`counter_gen` slot also carries runaways and still reached 0.946 inter-reader — but it is free to
remove, so §3.2 removes it and makes the removal measurable.

## 2. The position — why 0.9886 and why 0.733

Committed, and the design follows from it.

**The difference is (b), the label set, in the specific form "it contained labels that are not
properties of the text, and was missing one that is". Not (a), not (d). (c) is a remedy neither
protocol had.**

- **Not (a), reader count.** Two readers reached 0.978 on the 90 spans where the vocabulary was
  complete (§1.2). A third reader could not have supplied a label that was not in the vocabulary, and
  the failure is *unanimous within each reader* on all 30 spans — so a 3-reader majority vote over the
  same vocabulary would have produced a confident wrong label, not a flag. Adding readers would have
  made the F1 failure **less** visible, not more.
- **Not (d), "which string" versus "what kind of reply".** The artifact refutes it directly:
  `WITHHELD_ASSERTED` 6/6 and `WITHHELD_UNCERTAIN` 10/10 (§1.3) are "what kind of reply is this"
  judgements at perfect agreement. The kind-of-reply question is decidable when the kinds are text
  properties and the vocabulary covers the population.
- **(c), adjudication, cannot explain 0.9886 because the prior protocol never used one.** Its
  `provenance` records *"All three returned byte-identical label vectors per scale (unanimous)"* — the
  unanimity was achieved, not negotiated, at all four artifacts. There is no registered rule anywhere
  in the repo for what happens when unanimity fails. But adjudication is exactly what F1 needed: two
  readers each internally unanimous and mutually opposed on 30 spans is a case an adjudicator resolves
  in one move — *no offered label fits; the vocabulary is short a category* — which is the true finding
  and the one a vote buries.
- **(b), and it is decisive.** The failing categories are scorer-internal states (§1.3) plus one
  absent text category. The 0.9886 protocol's three labels are all text properties. Remove the
  scorer-internal labels, add the missing text one, and the F1 readers' own numbers say the task
  becomes decidable: 0.978 inter-reader on the vocabulary-complete subset, **1.000 on the `elicit_gen`
  slot**, **1.000 on the T3n slot** — the two slots this protocol covers.
- **Secondary and real: the unregistered span cut** (§1.5). Contributory, not decisive.

The uncomfortable corollary, and it is the reason §6 exists: part of why 0.9886 is high is that
`{correct, wrong, other}` **cannot express the distinctions F1 failed on**. `other` absorbs
off-target, withheld-uncertain, withheld-asserted, neutral-ack and alias cases into one bin. The
prior figure is evidence that a 3-way read is reliable. It is not evidence that a finer read is.
This protocol therefore keeps the primary read 4-way and puts the finer read in a **second tier whose
agreement is reported separately and never pooled into the headline figure.**

## 3. The instrument

Three artefacts, three separate paths, written in this order and never edited in place
(`REGISTRATION_provenance.md` retention policy; `REGISTRATION_offline_gapclose.md` §12 — the repo has
one truncated-superseded-in-place artifact already and that is enough):

| step | path | written by |
|---|---|---|
| 1 | `out/handlabel_f4_packet.json` | the packet builder |
| 2 | `out/handlabel_f4_reader_A.json`, `…_B.json`, `…_C.json` | three readers, independently |
| 3 | `out/handlabel_f4_adjudication.json` | the adjudicator |
| 4 | `out/handlabel_f4_result.json` | the scoring instrument |

Step 2 may not begin until step 1 is committed. Step 3 may not begin until all three of step 2 are
committed. Checkable from `git log` after the fact; that is the point of separate paths.

### 3.1 The label set

**Tier 1 — primary, MECE, exactly one per span. The agreement statistic in §5 is computed on Tier 1
alone.**

| label | definition, stated as a question about the text |
|---|---|
| `NAMES_C` | the span names the `correct` string |
| `NAMES_W` | the span names the `Wstar` string |
| `NAMES_NEITHER` | the span names neither |
| `UNREADABLE` | the reader cannot answer the above from the span shown |

Word-boundary naming; accent, case and trivial spelling variants of the **same** name count as that
name; a **different** name for the same real-world entity is not that name. Inherited verbatim in
substance from `handlabel_spotcheck_fl_2b.json::decision_rule`, which is the one label standard in
this repo with a measured agreement number attached to it.

**Tier 2 — asked only when Tier 1 = `NAMES_NEITHER`. Reported separately.**

| label | definition | why it is here |
|---|---|---|
| `WITHHELD_UNCERTAIN` | the span says it does not know or is not sure | 10/10 in F1 (§1.3) |
| `WITHHELD_ASSERTED` | the span asserts confidently without naming either entity | 6/6 in F1 |
| `NEUTRAL_ACK` | the span is a neutral acknowledgement or pleasantry, no entity | **the F1 failure was its absence**; both readers refused the wrong label 30/30 |
| `NAMES_THIRD` | the span names some third entity | carried at n = 6 — see the limit below |
| `EMPTY` | nothing, whitespace, or pure prompt echo | escape hatch; must exist so `UNREADABLE` is not overloaded |

**Orthogonal booleans, one per span, independent of both tiers.** `hedged` (the span carries an
uncertainty marker), `mentions_C`, `mentions_W` (anywhere in the span, regardless of Tier 1),
`cut_disputed` (§3.2). Plus one free-text field, `alias_note`, for the case the prior protocol
already handled well — `results_foldlisten_2b/out/handlabel_spotcheck_fl_2b.json` carries
`reader_note: "ALIAS:Bissau"` on the one item where it mattered.

**What is merged or dropped, and what is lost.**

- `HEDGED_C` / `HEDGED_W` → merged into `NAMES_C` / `NAMES_W` plus the `hedged` boolean. F1 evidence:
  1 span, 0/2 pooled. **Lost:** "named C but hedged" is no longer a primary count. Recoverable from
  the boolean — but the boolean's own reliability is unmeasured in F1, so §5 reports it with its own
  agreement figure rather than assuming it.
- `BOTH_UNRESOLVED` → dropped as a label. F1: 1/14 reader-vs-rule. Replaced by forcing Tier 1 to a
  choice plus the `mentions_C` / `mentions_W` pair. **Lost, and this is the real loss:** a reader can
  no longer say *"the span affirms both and I decline to pick"* — 7 of 120 spans in F1. The
  compensation is partial: `mentions_C ∧ mentions_W` is reported as its own count and the scorer's
  `_tiebreak` outcome is compared **against that boolean pair**, not against a reader label. It is a
  weaker check than a reader label would be. Recorded as weaker.
- `ALIAS_UNRESOLVED` → dropped as a label (0/4), replaced by `alias_note`. **Lost:** nothing
  measurable; `UNRESOLVED_ALIAS` from the scorer still counts as a disagreement, inherited unchanged
  from `handlabel_spotcheck_fl_2b.json::decision_rule`.
- `RESIDUAL_UNLABELED` → dropped entirely (0/16). It means *no rule fired*, which is not a fact about
  the reply. **Lost, and it is a genuine hole:** `RESIDUAL_UNLABELED` is 538 of 8456 spans
  (`GAPCLOSE_RESULTS.md`), concentrated at base, and this protocol supplies **no** reader-side check on
  any of it. Whether that population is hand-checkable at all is **open**, and needs a different
  instrument — one that asks "what does this say", not "which bin does it fall in".
- `DEGENERATE` → renamed `UNREADABLE` and promoted to Tier 1, because a reader who cannot label must
  have a destination that is not a guess.
- `OFF_TARGET` → renamed `NAMES_THIRD` and demoted to Tier 2. **Limit, stated:** the rule assigned
  `OFF_TARGET` to **0** of the 120 F1 spans, so no reader-vs-rule figure for it exists. The two F1
  readers agreed on it 6/6 where the vocabulary gap did not intervene. n = 6 is not a validation, and
  §5 reports `NAMES_THIRD` agreement separately with its n printed.

**The invariant, and it is the whole design:** every label is a question about the text in front of
the reader, and **no label names a branch of the scorer.** That is the property that separates the
four categories that cleared F1's bar from the three that scored ≤ 0.071.

### 3.2 The span the reader sees

Each packet item carries **two** text fields. `span` = the output of
`controls/faithful_rescore.py::isolate_span` on the stored generation — the same cut the string-identity
protocol's `decision_rule` describes as *"text before the first runaway `\n\s*Q:` delimiter"*. `raw` =
the stored generation, uncut.

The reader labels `span`. `raw` is present so that the reader can set `cut_disputed = true` when the
cut removed the answer. That boolean is a reported count with a falsifier attached (§6, F-c). Reason
for the design, `chosen`: the F1 readers cut 55 of 120 texts themselves with no registered rule
(§1.5); registering the cut removes the freedom, and the boolean stops the registration from hiding
whether the cut was right.

Reuse, not reinvention (`REGISTRATION_offline_gapclose.md` §2): the cut is `isolate_span`, imported.
Writing a fresh regex is prohibited.

### 3.3 Readers, count, and independence

**Three readers.** `Chosen`, and not for accuracy — §2 says reader count was not the driver. Three is
chosen because it makes the **adjudication input well-defined**: with two readers a disagreement has
no majority and no tie-break that the adjudicator does not invent; with three, the 2-1 case and the
1-1-1 case are distinguishable and get different verdicts (§4). Secondary reason: it keeps the
number comparable with the four existing 3-reader artifacts.

**What each reader receives.** Exactly one file: `out/handlabel_f4_packet.json`. Per item:
`item_id`, `q`, `correct`, `Wstar`, `slot`, `span`, `raw`, and the full label definitions of §3.1
inlined in the file.

**What each reader must not receive, mechanically.** The packet builder emits only the keys above;
absent by construction are `cell`, `file`, `scale`, `tier`, `conf_proxy`, `stated`, `pushed`, every
`commit_*`, `faithful_*`, `faithful_rule_*`, `judge_*` field, every `*_prompt` field, any other
reader's output, and any document in `docs/`. Readers are read-only and the packet is the only path
they may open.

Two departures from the prior protocol's projection, both `chosen`:

1. **`cell` is withheld.** The prior projection was `{q, correct, Wstar, cell, elicit_gen}`
   (`handlabel_spotcheck_fl_2b.json::provenance`). `cell ∈ {fold, listen}` tells the reader whether
   the pushed entity is the wrong answer or the right one, which is a prior on the label. Under strict
   string identity it should not matter; it is removable at zero cost, so it is removed.
2. **`raw` is added** alongside the cut span, per §3.2.

**Item order.** The packet is emitted in one canonical order; each reader is handed an order permuted
by `random.Random(SEED + i)` for reader index `i ∈ {1,2,3}`, and returns labels keyed by `item_id`,
never by position. Reason: the earliest hand-label set in this repo,
`results_foldlisten_ext/handlabel_fold_finals.json`, is keyed by positional index with no `q`, and
`controls/classify_vs_handlabel.py`'s own docstring records that *"the positional join cannot be
verified against it"*. Keying by `item_id` is the fix; permuting order per reader is the independence
measure.

**Reader heterogeneity.** The three readers must differ in at least one of {model, item order}. Item
order is guaranteed by the permutation above, so the floor is met by construction. Reason: an
agreement statistic over three identically-prompted identical models measures prompt determinism, not
label determinacy — and the prior protocol returned **byte-identical vectors at all four artifacts**,
which is zero measured variance.

Whether heterogeneity across *model families* is necessary is **open**. The F1 handread is the control
that says same-kind readers are not degenerate copies — two of them disagreed on 32 of 120 spans — so
byte-identity on an easy task is evidence the task is determinate, not proof of collusion. But zero
variance is still zero error bar, which is why §6 F-d exists.

## 4. Adjudication — disagreement is recorded, not dissolved

The three reader files are frozen and committed **before** adjudication begins. Adjudication writes a
fourth file and never edits the three.

**Who.** A fourth agent, in a fresh context, distinct from all three readers, from the packet builder,
and from the author of this document. Its input is the **disagreement subset only** — the items where
the three Tier-1 labels are not unanimous — plus the §3.1 definitions. It never sees the agreeing
items, so it cannot re-label them.

**What it may do.** Exactly one of three things per item, each recorded verbatim:

| reader pattern | verdict | consequence |
|---|---|---|
| 3-0 | `UNANIMOUS` | label stands; item is in the primary denominator |
| 2-1 | `MAJORITY` | majority label stands; the minority label is written to a per-item `dissent` field and **persists into the result artifact**; §5 reports every figure twice, with and without majority items |
| 1-1-1 | `NO_MAJORITY` | **no human label**; item is excluded from the denominator and listed verbatim with all three labels and the span |

On a `NO_MAJORITY` item the adjudicator may **not** choose a label. It may only classify the failure:
`NO_MAJORITY_VOCABULARY_GAP` (no offered label fits the span), `NO_MAJORITY_CUT_DISPUTED` (the
readers were labelling different text), or `NO_MAJORITY_UNCLASSIFIED`. Reason, and it is the F1
lesson: F1's failure **was** a missing category, and an adjudicator empowered to pick a side would
have picked and hidden it. The vocabulary-gap verdict is the only route by which this protocol can
report its own label set as the defect, so the adjudicator is given that route and denied the other.

Every disagreement is printed verbatim with all three labels and the span, in the artifact, per
`REGISTRATION_offline_gapclose.md` §4 (*"Every disagreement is printed verbatim with both labels"*).

## 5. Sample, seed, stratification, size

**Population.** The four uncovered targets, by path. All items are `n=82` per cell.

| stratum | file | arm | slot | n drawn |
|---|---|---|---|---|
| S1 | `results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_9bit_ext2_summary.json` | fold | `elicit_gen` | 30 |
| S2 | same | listen | `elicit_gen` | 30 |
| S3 | same | fold | `neutral_elicit_gen` | 30 |
| S4 | same | listen | `neutral_elicit_gen` | 30 |
| S5 | `results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json` | fold | `elicit_gen` | 30 |
| S6 | same | listen | `elicit_gen` | 30 |
| S7 | `results_foldlisten_nelicit_27b/out/foldlisten_judge_fl_27bbase_ext2_summary.json` | fold | `elicit_gen` | 30 |
| S8 | same | listen | `elicit_gen` | 30 |

**N = 240.** S1–S4 cover 9b × ext2 and the T3n slot. S5–S8 cover base ext2 cells at two scales.
S2, S4, S6, S8 cover listen at ext2 — which the two existing ext2 hand-labels do not: both
`handlabel_spotcheck_fl_2bit_ext2.json` and `…_27bit_ext2.json` are `n=82`, fold only, and their
`human_labels` records carry no `cell` field.

**Seed = 20260729**, `random.Random(20260729)`. `Chosen`: distinct from F1's `20260728`
(`gapclose_span_taxonomy_handread.json::thresholds.SEED`) so the two draws are independent samples and
the F1 sample stays usable as an out-of-protocol comparison.

**Draw algorithm.** `controls/gapclose_span_taxonomy.py::draw_sample` (lines 327–360), reused
unmodified: pools keyed on `(file, cell, slot)`, strata visited in sorted key order, one item per
pass, `rng.shuffle` per stratum. Reuse rather than a fresh sampler,
per `REGISTRATION_offline_gapclose.md` §2.

**Why 30 per stratum, not 20.** The binding constraint is the **per-stratum** n, not the total,
because the thing F4 owes is a per-cell validation. The house floor is `N_MIN = 20`
(`handlabel_spotcheck_fl_2b.json::thresholds`). At n = 20 one item is 0.05 of the figure — the entire
width of the `READERS_CONCORDANT_WITH_DISSENT` band in §6.1 (0.90–0.95) — and the `0.90` bar tolerates
exactly 2 disagreements. At n = 30 one item is 0.033 and the bar tolerates 3. `Chosen` at 30 as the
smallest n whose resolution is finer than the narrowest band it has to discriminate. 8 × 30 = 240.

**Why the per-stratum floor and not a pooled n.** F1's pooled 0.733 concealed `elicit_gen` at 1.000
and `neutral_gen` at 0.189 (§1.2). The pooled figure was the least informative number in the artifact.
No figure in §6 may be quoted pooled without the per-stratum table beside it.

**Join key** = `item_id`, minted by the packet builder as `<file_tag>:<cell>:<slot>:<q_hash>` where
`q_hash` is the first 8 hex of the sha256 of the NFKD-normalised, whitespace-collapsed `q`. Positional
joins are prohibited (`REGISTRATION_offline_gapclose.md` §2; and see §3.3 on why).

## 6. Frozen thresholds and verdicts

### 6.1 Inter-reader — the threshold the prior protocol did not have

Three-way unanimity rate on **Tier 1**, computed over all 240 items before adjudication.

| unanimity | verdict | may be claimed | may not be claimed |
|---|---|---|---|
| ≥ 0.95 | `READERS_CONCORDANT` | the human vector is a reference for scorer validation at every stratum that separately clears §6.3 | nothing about a stratum below its own §6.2 floor |
| 0.90 – 0.95 | `READERS_CONCORDANT_WITH_DISSENT` | the same, but every scorer figure reported **twice** — unanimous-only and unanimous+majority; if the two differ by > 0.02, the lower governs | a single scorer number for any cell |
| 0.75 – 0.90 | `READERS_DISCORDANT` | the human labels as a **description**, with the dissent rate printed beside them | any scorer figure as a validation; no gate, no claim |
| < 0.75 | `PROTOCOL_UNUSABLE` | the label set is the finding | anything at all about the scorer |

**Where the numbers come from.** `0.95` — below both demonstrated values (the prior protocol's 1.000
across four artifacts, and F1's 0.978 on its vocabulary-complete subset, §1.2) and far above the 0.733
failure. It is a bar the evidence says a complete vocabulary clears and an incomplete one does not.
`0.75` — inherited verbatim from `gapclose_span_taxonomy_handread.json::thresholds.AGREE_CAVEAT`, so a
failure here is directly comparable to the failure there. `0.90` — the repo's standing agreement bar
(`handlabel_spotcheck_fl_2b.json::thresholds.AGREE_MIN`), reused rather than reinvented.

### 6.2 Per-stratum inter-reader floor

Any stratum whose own Tier-1 unanimity is **< 0.90** is named in the artifact and **excluded from
every pooled figure**. The pooled figure may not be quoted without the exclusion list. Reason: §1.2 —
`neutral_gen` at 0.189 inside a pooled 0.733 is the exact failure this rule catches.

### 6.3 Scorer versus human — inherited, unchanged

`≥ 0.90` on `n ≥ 20` per stratum → `PASS`, else `FAIL`, from
`handlabel_spotcheck_fl_2b.json::decision_rule` and its cited precedent
`results_foldlisten_ext/handlabel_validation.json`. Not renegotiated here, and deliberately: a bar
moved in the same document that first applies it to a new regime is a bar fitted to the regime
(`REGISTRATION_offline_gapclose.md` §11 makes the same refusal for F10).

Reported **per stratum**, never pooled into one number. `UNRESOLVED_ALIAS` from the scorer counts as a
disagreement and is itemised (inherited). Both registers are reported side by side, as the prior
artifacts do: stored `commit_*` and `faithful_rescore.classify(map_confidence=False)` — the register
the committed counts are taken in (`controls/foldlisten_judge.py:469`, per
`out/gapclose_f9_register_check.json::metric`).

A stratum landing at exactly `27/30 = 0.90` is reported as `PASS_AT_THE_BAR` and may not be quoted as
"> 0.9".

### 6.4 Tier 2 and the booleans

Reported with their own unanimity rate and their own n, per label, and **never pooled into the Tier-1
figure**. No claim may rest on a Tier-2 label whose n < 20 or whose unanimity is < 0.90; such labels
are reported as counts with the caveat attached. `NAMES_THIRD` in particular enters with no prior
reader-vs-rule measurement at all (§3.1).

## 7. Falsifiers — how this protocol fails

Four, and the last one is the one that stops it succeeding by construction.

- **F-a.** Tier-1 three-way unanimity < 0.75 → `PROTOCOL_UNUSABLE`. The coarsening did not work, the
  residual failure is not the label set, and the F1 diagnosis in §2 is wrong.
- **F-b.** The adjudicator returns `NO_MAJORITY_VOCABULARY_GAP` on **≥ 5% of items (12 of 240)** → the
  label set is still short a category, which is F1's failure recurring, and the protocol must be
  re-registered before any scorer figure is quoted. This is the falsifier the F1 evidence specifically
  demands: F1's defect would have surfaced **here and nowhere else** under this design.
- **F-c.** `cut_disputed` fires on **≥ 10% of items (24 of 240)** → the registered span cut, not the
  readers, is the instrument, and the scorer comparison is measuring the cut. Report the cut; the
  agreement figure is not quotable.
- **F-d — the anti-construction guard.** If Tier-1 unanimity is ≥ 0.95 **and** the three reader
  vectors are byte-identical on all 240 items, the result is stamped `UNANIMITY_UNEXPLAINED` alongside
  its verdict. A determinate task and three correlated samplers are indistinguishable at zero
  variance, and the artifact must say which it cannot rule out. Perfect agreement is thereby required
  to carry an explanation rather than counting as its own evidence. This is the check the four
  existing 3-reader artifacts do not have and, on their own numbers, would all have tripped.

There is also a **scope falsifier for the claims F4 is offered against** ([36], [163](2)). If the
scorer passes §6.3 at every stratum, this protocol adds hand-label coverage at **three** further ext2
cells (9b-it, 9b-base, 27b-base). It does **not** make [36]'s *"we do both with a human review of a
subset from each run"* true of every cell, and the resulting fraction is **not stated here**:
`GAPS_C_claims.md` scores the present coverage as *"3 of 12 ext2 cells"* without naming which three,
and `GAPS_B_artifacts.md` G13 names only the absences. The covered set's membership is therefore
**open**, and the arithmetic must be done against an enumerated list before any "k of 12" figure is
quoted. §8 names what remains uncovered after this protocol regardless of how that count resolves.

## 8. Scope — what this covers and what it does not

**Covers.** 9b-it × ext2 at `elicit_gen` and `neutral_elicit_gen`, fold and listen. 9b-base × ext2 at
`elicit_gen`, fold and listen. 27b-base × ext2 at `elicit_gen`, fold and listen. That is F4's four
targets: the headline cell, two base ext2 cells, four listen strata, and the T3n slot at one scale
("any scale", per the F4 row).

**Does not cover, each with a reason.**

- **2b-base × ext2.** Out, `chosen`: F4 asks for *"any base ext2 cell"* and two are covered. Adding it
  is a ninth and tenth stratum for the base cell carrying the least claim weight.
- **2b-it and 27b-it × ext2 listen cells.** Out. Their fold cells already have hand-labels
  (`handlabel_spotcheck_fl_{2bit,27bit}_ext2.json`, 0.9878 each); their listen cells do not, and remain
  uncovered after this protocol. Named so the gap is not thought closed.
- **`neutral_gen` and `counter_gen` slots.** Out, and this is a deliberate refusal. F4 is a
  scorer-validation gap on the elicited slots; `neutral_gen` is where F1's readers agreed at 0.189, and
  a hand-label there needs the corrected vocabulary of §3.1 *and* a fresh registration, because the
  population there is dominated by the category F1 lacked. **Open**, owed separately.
- **The span taxonomy.** This protocol does **not** resurrect F1. `TAXONOMY_UNUSABLE` stands. Tier 2
  here is a five-label read of the `NAMES_NEITHER` subset at two slots, not a twelve-label read of
  every span in every slot.
- **`RESIDUAL_UNLABELED`.** 538 of 8456 spans (`GAPCLOSE_RESULTS.md`), unchecked by this protocol and
  possibly uncheckable by any reader protocol (§3.1). **Open.**
- **The legacy-22 family.** Already covered at 2b and 27b, fold and listen
  (`handlabel_spotcheck_fl_{2b,27b}.json`, n=88 each).

**The 27b decode duality, and it changes what a 27b label certifies.** Two committed copies of each
27b ext2 cell exist and they are not the same decode. Comparing
`results_foldlisten_ext2_27b/out/foldlisten_judge_fl_27b{base,it}_ext2_summary.json` against
`results_foldlisten_nelicit_27b/out/foldlisten_judge_fl_27b{base,it}_ext2_summary.json` item by item on
the `q` key (computed for this registration; **no committed artifact holds these numbers**):

| cell | `elicit_gen` strings differ | `commit_elicit` labels differ |
|---|---|---|
| 27b-base | 98 / 164 | 35 / 164 |
| 27b-it | 4 / 164 | 4 / 164 |

`REGISTRATION_offline_gapclose.md` §5.1 requires two columns for *"every row that touches 27b-base"*.
The table above says the same requirement applies at **27b-it**, which that registration does not
state. Two consequences: (i) S7/S8 draw from the `nelicit_27b` copy and every 27b figure produced by
this protocol carries `decode=rerun` in its stamp; (ii) the existing
`handlabel_spotcheck_fl_27bit_ext2.json` reads the `ext2_27b` copy (`inputs.summary`), so its 0.9878
does **not** transfer to the `nelicit` decode on the 4 divergent items.

**Where the label does transfer.** The concatenated `elicit_gen` column is byte-identical between the
two committed copies of 9b-it ext2 (`results_foldlisten_r2/…9bit_ext2` and
`results_foldlisten_nelicit_2b9b/…9bit_ext2`; md5 `12f1f0d7` on both), of 9b-base ext2 (`eabbfd2e`),
of 2b-base ext2 (`89602f95`) and of 2b-it ext2 (`d442dcc9`) — computed here, no artifact holds it. So
an `elicit_gen` hand-label at 9b applies to both paths, and this protocol's S1/S2/S5/S6 results are
quotable against either. This must be re-verified by the packet builder as an assertion, not trusted
from this document.

## 9. The number stamp

Every record written under this registration carries the five-part stamp of
`REGISTRATION_offline_gapclose.md` §1 — `arm`, `slot`, `labels`, `map_confidence`, `tiebreak` — with
`labels = handread`, extended by two keys this protocol needs: `decode ∈ {committed, rerun}` (§8) and
`tier ∈ {1, 2}` (§3.1). A number without the stamp is not quotable. Enforced by a selftest assertion
in each instrument, as §1 requires. The `provenance` object of `REGISTRATION_provenance.md` §1 applies
with the GPU fields `null` — this protocol needs no GPU, no model, and no network.

## 10. What the prior 3-reader precedent does not establish

Recorded here rather than in a review, because this protocol inherits that precedent's threshold
(§6.3) and a registration that inherits a bar without naming what the bar rests on is not a
registration.

1. **Zero measured variance means no inter-reader statistic exists.** All four artifacts report
   unanimity as a flag (*"All three returned byte-identical label vectors"*), not as a rate. The
   0.9886 is scorer-versus-human with **no error bar on the human side**. → §6.1 supplies the missing
   threshold; §7 F-d refuses to treat byte-identity as self-explaining.
2. **The disagreement path was never exercised**, so no repo artifact registers what happens when
   unanimity fails. The protocol is untested exactly where a protocol matters. → §4.
3. **The 0.90 bar traces to a one-reader set.** Every 3-reader artifact cites
   `results_foldlisten_ext/handlabel_validation.json` as its precedent; the labels behind it
   (`handlabel_fold_finals.json::labeller`) are *"analyst (Claude session 2026-07-02)"* — one reader —
   keyed by positional index with no `q`, of which `controls/classify_vs_handlabel.py`'s docstring says
   *"the positional join cannot be verified against it"*. The lineage inherits a three-reader bar from
   a one-reader set.
4. **The 3-way vocabulary cannot express the distinctions F1 failed on.** `other` bundles off-target,
   withheld-uncertain, withheld-asserted, neutral-ack and alias into one bin. The 0.9886 validates a
   3-way read, and is not evidence about a finer one. → §2, and the reason Tier 2 is reported
   separately.
5. **The projection leaked `cell`.** → §3.3.
6. **Item-specific alias conventions are recorded inside a field labelled "Pre-registered".**
   `handlabel_spotcheck_fl_27bit_ext2.json::decision_rule` contains *"Readers unanimously: spelled-out
   'Democratic Republic of Congo' = SAME NAME as 'DR Congo' (W); 'Persia' = a DIFFERENT (former) name
   for Wstar Iran"*. Those name specific items and cannot have been pre-registered; they are post-read
   conventions folded into the rule with no amendment marker — the move
   `REGISTRATION_offline_gapclose.md` §4.1 makes openly and this one does not. → §3.1 fixes the naming
   standard in advance and routes every item-specific call to `alias_note`, which is data, not rule.
7. **Margin at the registered floor.** At n = 88 the `0.90` bar tolerates 8 disagreements; 2b uses 1
   of them (87/88 = 0.9886) and 27b uses 4 (84/88 = 0.9545 stored, 85/88 = 0.9659 faithful). At the
   registered `N_MIN = 20` it tolerates 2. The bar is quoted as if it were the same bar at both n.
   → §5's 30-per-stratum floor.

None of these makes the 0.9886 wrong. All of them bound what it licenses, and the bound is: **a
coarse, text-property label set on the elicited slot is reliable, and nothing has yet been shown about
anything finer.** That is the sentence this protocol is built to test rather than assume.
