# RETRACTION REGISTER

Registration owed #10. A claim that no amount of work can make defensible is **withdrawn**, not
repaired. This register separates the two failure modes that are easy to confuse:

- **UNFIXABLE** — the evidence that would support the sentence does not exist and cannot be created,
  because the thing it describes is gone.
- **OVER-SCOPED** — the measurement is sound; the sentence claims more breadth than the measurement
  covers. The fix is to narrow the sentence, not to run anything.

Every entry below was verified in this session against the artifacts, not inherited from a prior
note. Nothing here writes to the gold vault documents — the register names what must change and
where, and the change is the researcher's.

---

## R-1 — [69], [70]: the hardware attribution. UNFIXABLE → WITHDRAW.

The sentences: that the 27b reproduction failure occurred "on two different GPU types", and that
"the split is by model size and not by machine".

**Verified.** A key-level walk over **all 323 committed JSON artifacts** finds **zero** fields named
`gpu`, `gpu_type`, `instance_type`, `driver`, `cuda_version`, `torch_version`,
`transformers_version`, `hardware`, `device_name`, `gpu_name`, `host` or `hostname`. The only
machine-adjacent key in the repo is `device` (83 artifacts, values `cuda` / `cpu`), which records a
torch device string and not a machine. The single `instance` key is an item index inside
`results_2b_attrgraph/out/cave_attribution_graph_2b.json`, not a box.

The boxes are terminated (`GET /api/v1/instances` returns 0). Lambda's audit log does record an
`instance_type` per launch, so a *future* run's hardware is recordable — but **no artifact records a
launch id or a timestamp**, so no audit-log entry can be tied to any committed artifact. The
association the two sentences assert cannot be reconstructed from either side.

**Withdraw both.** What survives, and is now measured rather than asserted, is a statement about a
cell instead of a machine: across the committed ext2 decode and the neutral-elicit re-run, 2b-base
and 9b-base give **identical** gate quantities at both label families, and 27b-base does not —
fold_rate 0.2115 → 0.1373 on commit labels, 0.2200 → 0.1458 on faithful, `n_fold_faithful` 11 → 7.
→ `out/gapclose_base_gate.json`. The non-reproducibility is localised to 27b. Why it is localised
there is unanswered, and the sentence that named a cause must not survive the sentence that named
the effect.

## R-2 — [1], [29]: "language models" in general. OVER-SCOPED → NARROW.

Every artifact in the repo is Gemma 2 — 2b, 9b, 27b, base and `-it`. Two sentences are written about
language models as a class. Narrow them to Gemma 2, or carry an external citation for the general
claim. A second model family is a different program and is out of scope by the standing directional
commitment (single family, deliberately), so this is a wording fix, not a work item.

## R-3 — the 27b column's decode. NOT a retraction: a mandatory disclosure.

Two 27b-base decodes exist and disagree, and the newer drafts silently switched between them.
`docs/drafts/TAXONOMY_withholding.md`, `JOIN_withhold_vs_fold.md` and `GROUNDING_notes_numbers.md`
read the **committed** ext2 decode; `GROUNDING_neutral_elicit.md` reads the **re-run**. No document
flags the switch. 27b-`it` is unaffected — it is identical between the two runs.

**SHARPENED 2026-07-29, and it is now stronger than a disclosure rule.** A third draw settles which is
which. `out/27b_decode_determinism_result.json`: an independent 27b-base decode is **BYTE_IDENTICAL** to
the neutral-elicit re-run — 164/164 items, 4428 item-fields, 22 derived quantities, zero mismatches — and
**DIFFs from the committed ext2 draw** on 654 values and 216 labels. So the committed 27b-base decode is
the **outlier**, and the re-run is reproducible.

Consequence: every 27b-base number taken from the committed decode must be **replaced** by the re-run's,
not merely labelled with its provenance. The publishable 27b-base column is the re-run's — fold 7/44/31,
listen 16/34/32, fold_rate 0.137, `NO_MOVEMENT`.

This does **not** rescue R-1. Why the committed draw differs is still unattributable, because that run
recorded no hardware. What *is* now attributable: the divergence tracks the **driver version**, not the
card — same H100 80GB HBM3 model at driver 570.148.08 reproduces byte-identically, at 580.105.08 diverges
by up to 0.5 nats on teacher-forced logprobs.

**ADDENDUM 2026-07-29 — three corrections to R-3's own text (the entry above is left as written; this
register does not silently rewrite itself).**
(a) "27b-`it` is unaffected — it is identical between the two runs" is **false at the item level**:
`out/foldlisten_repro_diff_fl_27bit.json` = `DIFF`, 373 value / 55 label / 10 derived mismatches,
164/164 items differing. Only the aggregate push column matches — compensating flips
(`GROUNDING_neutral_elicit.md`). The two-decode disclosure rule extends to 27b-`it` (`OWED.md` C7).
(b) The final paragraph's attribution ("tracks the **driver version**, not the card") is **refuted**
by the fuller cluster table from the format-matched run (`OWED.md` H2): this run's box was H100 80GB
HBM3 @ 570.148.08 and matched cluster 3 (H100 80GB HBM3 @ 580.105.08) — same card + different driver
= same cluster; different card (H100 PCIe) + same driver = different cluster. The divergence tracks
the **card**. Cluster 2 remains a singleton on cluster 1's own card AND driver, explained by neither
axis. The same wrong attribution is frozen in `out/27b_decode_determinism_result.json` under the key
`the_divergence_TRACKS_THE_DRIVER_not_the_card` — the key is data; the attribution is withdrawn.
(c) The published column "fold 7/44/31 … `NO_MOVEMENT`" is the **commit register**; the faithful
register of the same draw reads `decision_faithful.category = MOVEMENT_LISTEN_ONLY` (both draws). A
quoted 27b-base verdict must name its register as well as its draw.

## R-4 — F9 as a gap. RETRACT THE GAP (not a claim).

The ledger files F9 as "`classify_vs_handlabel` never run on the 2b/27b hand-label sets that exist",
implying a missing measurement. True of the named script; **false of the measurement.** All four
`handlabel_spotcheck_*` artifacts already carry a `faithful_strict_vs_human` block computed by
`faithful_rescore.classify(map_confidence=False)` on `elicit_gen` against a unanimous 3-reader blind
string-identity human vector, with its own pre-registered ≥0.9 threshold and a PASS decision. F9 is
script-coverage bookkeeping, not a coverage hole.

Two genuine debts surfaced in its place, and they are instrument defects rather than claim defects
→ `out/gapclose_f9_register_check.json`:

1. `controls/classify_vs_handlabel.py` calls `classify()` at its **default `map_confidence=True`** on
   `elicit_gen`, against `faithful_rescore.py:88 STRICT_FIELDS` and
   `foldlisten_judge.py:469`, which score that slot with `map_confidence=False`. The validator
   certifies a different register from the one the counts are taken in. **Latent, not live:** on the
   56 hand-labelled items 0 labels differ and the committed 1.000 reproduces exactly. It would
   silently certify the wrong register on a set where confidence mapping fires — Gate 3 recorded 15
   of 44 such relabels at 2b-base.
2. The same instrument joins hand-labels to items by **positional index** (`join_item`), validating
   required fields but never checking that the item it landed on is the item that was labelled.

## R-5 — TAXONOMY's "verified 63/63 contain both". THE COUNT STANDS; THE CITATION DOES NOT.

`docs/drafts/taxonomy_withholding_rederive.py:86` sets `HAND_COUNTER_DEFAULT = "BOTH"`, and
`classify_row` returns it for every free-reply (counter-slot) span with no lexical hit and no hand
entry — and `HAND_COUNTER` holds exactly **one** entry. So the free-reply `BOTH` label is a
**default, not an adjudication**: the script the document cites does not check that both entities
appear in any of those spans.

An isolated reader then checked it independently, with a word-boundary entity read on both C and W\*,
and it **holds 63/63**. So this is not a wrong number — it is a citation pointing at a script that
does not perform the check. Fix the citation, or say the check was done by hand. Not a retraction of
the count.

## R-6 — GROUNDING L194's "67 of 74". WITHDRAW: THE NUMBER HAS NO SOURCE.

The line claims ‑chat "still names an answer on 67 of 74 items" under the challenge mask, split C 66
/ W\* 1 / NEITHER 7. Re-derived from `results_foldlisten_p2/out/foldlisten_phase2_p2_9bit_summary.json`:
the fold-mask arm names an answer on **73 of 74**. Its `commit_elicit` split is 70 / 3 / 1, its own
`arm_counts.fold_mask` is moved 3 / held 70 / abstain 1, and an independent word-boundary read gives
C 71 / W\* 2 / NEITHER 1. **No artifact anywhere in the tree holds a 66 / 1 / 7 triple at n = 74**,
and the line cites no file.

The claim's *point* survives and gets stronger — masked ‑chat still answers, and answers with its own
previous answer — but the printed pair must be replaced by a sourced one. This resolves ledger gap F8
against the printed number rather than against the artifact field.

## R-7 — GROUNDING L196's two paired-register figures. WITHDRAW: NOT ON DISK IN ANY REVISION.

27b-it "49 vs 65, 23-to-7" re-derives as **51 vs 66, McNemar 22-to-7** at both `497b2c0^` and HEAD,
and the refresh commit the line credits changed **zero** counter and zero elicit labels in that file.
9b "50 vs 67, paired 21-to-4" re-derives as **52 / 68 (48-20-4-10)** pre-plural and **52 / 67
(47-20-5-10)** at HEAD; the plural fix left the fold count at 52 and moved *listen* down.

Neither printed pair is a committed register in any revision. Both must be replaced with the on-disk
figures, and the "pre-plural" attribution dropped.

## R-8 — GROUNDING's RECONCILIATION headline. RECLASSIFY: UNAUDITABLE, not REPRODUCES.

Three of its rows are strict-register counter labels. `controls/faithful_rescore.py:88` sets
`STRICT_FIELDS = ("elicit_gen",)`, so `counter_gen` is scored with confidence mapping everywhere, and
strict counter labels exist in **no artifact and no git revision** (checked `b92edbe^`, `7edbbff^`,
`2c5a8bf^`, `2c5a8bf`, HEAD). The document files them under REPRODUCES. `JOIN` states the identical
limitation correctly about its own 50-vs-52, so the repo already knows; one document did not apply it.

## R-9 — "statistically indistinguishable". REPLACE THE WORD.

No test statistic, p-value or CI for that sentence exists in the document, its script, or any JSON.
Computed now: Mann-Whitney *p* = 0.971, permutation on the median difference *p* = 0.839 — consistent
with the sentence, on **n = 20 against 44**. That is an accepted null at low power, and
"indistinguishable" is standing in for "underpowered". Either report the test with its n, or say the
comparison was not powered to separate them.

## R-10 — JOIN's stale correction. PROPAGATE OR WITHDRAW THE FILE'S VERDICT.

`JOIN_withhold_vs_fold.md` corrects the uncertainty series 0/14/1 → 0/20/1 at L513–523 and names
`S(1b)` as the only site. The old number is still printed at **L182 and L288** — the prose and the
Verdict — so the file contradicts itself, and the correction's consequences were never propagated:
the S3 uncertainty-at-elicit row moves on every field, including Fisher two-sided **0.533 → 0.273**.
Full before/after table in `REDERIVE_20260728.md` §4.

## R-11 — the FREE class's own arithmetic. CORRECT THE DENOMINATOR.

Not a claim about the model; a claim about the audit. `GAPS_RECONCILED.md` §4.1 states the class
closes 42 claims. Two defects: F1's row **cites no claim IDs at all** (its 22 is recoverable only by
subtracting F3's [65]–[68] from the 26-ID span list in `GAPS_C_claims.md` §G4, an inference the
ledger never states), and **[36] is counted twice**, under F4 and again under F9. Corrected class
total: **41 distinct claims**, of which F9's share is now withdrawn per R-4.

---

## R-12 — "REDISTRIBUTE" as the `-it` substrate verdict. WITHDRAW THE LABEL; THE NUMBER IS UNAUDITABLE.

Added 2026-07-30, from two isolated readers who agreed independently (the second was claim-blind and
never saw the first's report).

Claimed at `RESEARCH_QUESTIONS.md:193-198` ("gaps 1 + 3 CLOSED … ALL-attention restores **0.875**,
ALL-MLP **0.751** → … honest verdict **REDISTRIBUTE**") and `POSITION_KNOWING_BEFORE_SAYING.md:314`.

Five findings, each from the artifact or the source:

1. **No instrument can emit the string.** `grep -rn REDISTRIBUTE --include=*.json` over the repo
   returns nothing. The categories `controls/cave_residstate_decisive.py:104-129` can emit are
   `ATTENTION_CARRIES / MLP_CARRIES / BOTH_REDUNDANT / NEITHER_LOCALIZED / CHANNEL_INERT /
   INSUFFICIENT`. The artifact's actual decision is
   `results_residstate_decisive/out/cave_residstate_decisive.json#decision.category` =
   **`BOTH_REDUNDANT`**. "REDISTRIBUTE" is a prose synthesis, and it reads as a measured verdict.
2. **The headline sits outside its own artifact's CI.** `#it_self.all_attn` = 0.874962 against
   `#it_self.all_attn_ci` = [0.571004, 0.862805]. Cause, in source: `:258` sets
   `all_attn = max([ar_m, aw_m])` — the max of two arm means — while `:265` bootstraps the **pooled**
   read+write item list. Whenever write ≫ read the point estimate exceeds its own interval. Same
   pattern at `results_anyscale_mc_9b/out/cave_residstate_decisive.json` (0.865831 vs [0.552485, 0.83528]).
3. **The `-it` random floor is a constant, not a measurement.** `it_rand = 0.0` is hardcoded at
   `cave_residstate_decisive.py:303`, so the decision message's "vs rand 0.000" measures nothing. The
   only measured `-it` random floor in the family is
   `results_residstate_close/out/cave_residstate_close.json#batteries.it_rand_write` = 0.0006151.
4. **The producer is not in the repo.** The artifact carries `#reprocessed_offline` = true, that
   string occurs in no committed `.py`/`.sh`, and the file lacks `base_aurocs`/`it_aurocs` which
   `cave_residstate_decisive.py:277-278` always writes. No per-item restoration records exist
   (`:292` writes `out/cave_residstate_decisive_cache.json`; it is nowhere in the repo). The most-cited
   `-it` number therefore **cannot be re-derived by anyone** — R-1 class, and R-1 does not cover it.
5. **Two committed artifacts contradict each other on the same model, axis layer and pool.**
   `results_residstate_close/out/cave_residstate_close.json#decision.category` = `DISTRIBUTED_CONFIRMED`
   ("at ‑it NEITHER the span heads (0.008) NOR the axis-writer heads (0.018) carry it … RLHF moves
   caving off the attention doubt-circuit") against `BOTH_REDUNDANT` (ALL-attention 0.875 does carry).
   The docs adopted the later one; the earlier still stands committed and undisclosed.

**Disposition. UNFIXABLE as a number → WITHDRAW 0.875 / 0.751 from all prose.** The label
"REDISTRIBUTE" is withdrawn; where the artifact is cited at all it is cited as `BOTH_REDUNDANT`
**under the self-judge axis**, with (2), (3) and (4) attached. A separate point that must not be
mistaken for a competing estimate: `#label_match_changes_verdict` = true and
`#decision_labelmatch.category` = `INSUFFICIENT` with `it_all_attn` = 0.0 — under realized-argmax
labels `#it_real.ncav` = 0, i.e. **no `-it` item counts as caved at all**, so the axis cannot be fit.
That is a failed gate, not a measurement of zero, and writing it as "really 0.0" would be a second
error. Re-running the instrument with the CI computed over the same quantity as the point estimate,
and with the per-item cache committed, would make a claim possible again; nothing else will.

## R-13 — the doubt circuit's readout. NOT A RETRACTION: A MANDATORY SCOPE LINE.

Added 2026-07-30, same two readers; the second recomputed every mean and count from persisted
per-item records, so unlike R-12 this one is fully auditable.

`RESEARCH_QUESTIONS.md:68-71` (claim 2) and its 27b extension at `:232-237` state the head-set READS
and WRITES the cave with decision `BOTH` at all three scales. **That stands.** What no doc carries is
that `controls/cave_doubt_decollide.py` — the control written to test exactly this — returns
`#result.decision.category` = **`READOUT_SENSITIVE` at 2b, 9b and 27b**
(`results_decollide/out/cave_doubt_decollide_{2b,9b,27b}_base.json`), with `read_delta` 0.259 / 0.458
/ 0.429 against its own `DELTA` = 0.2, and `read_stable` / `write_stable` false at all three.

The restorations are a property of the **first-token P(W\*) readout**. Under the stripped content
margin, the same interventions on the same heads and items give: 2b write 0.019146 against a matched
-random floor of 0.017612 (**1.09×**), 27b write 0.037247 against 0.022089 (1.69×), 9b write 0.050988
against 0.021552; only 9b READ (0.130187) clears its floor appreciably. The sibling
`cave_headset_specificity_decollide_{2b,9b,27b}_base.json` decides `READOUT_SENSITIVE` at all three
as well, and at 27b the content-margin K-sweep falls **below** its own random floor at K = 1, 3, 20.

The instrument's own `#decision_rule` ends "Numbers + category only; **no claim attached to any
readout, question class, or category**" — so it does not adjudicate which readout is right, and this
entry does not either. **Disposition: FIXABLE BY QUALIFIER.** Every statement of the doubt-circuit
result carries the readout it holds on, and the sentence "replicates at 2b and 9b base" may not be
written without it. This is the qualifier `SNAPSHOT_circuit_groundtruth.md` §7.1 S1 already attaches;
the ledger must attach it too.

## R-14 — a restoration above 1.0, in an artifact that declines to decide. DO NOT CITE.

`results_fold_vs_listen/out/cave_fold_vs_listen.json#models.it.battery.AGAINST_GRAIN.all_attn_write_alllayer`
= **1.078249** — a restoration fraction greater than one, unremarked in the artifact. Every cell of
that file carries `#models.*.decision.category` = `MOVE_UNMATCHED` with the message "the SC-S4
headroom confound is NOT cleared -> **no verdict**". The ALL-X numbers in it are therefore
pre-verdict quantities from an instrument that refused to issue one. **Disposition: HELD.** The
file's head-overlap counts (4/5 base, 5/5 `-it`) are a separate, auditable statistic and are not
affected; the battery restorations from this file are not quotable until the >1 value is explained.

---

## Standing rule this register establishes

A claim whose supporting artifact was never written is not "unverified" — it is **unsupported**, and
the difference is whether a sentence may stand while the work is queued. It may not. Where the
artifact can still be written, the claim is HELD pending it. Where the artifact cannot exist
(R-1, R-6, R-7), the claim is withdrawn today.
