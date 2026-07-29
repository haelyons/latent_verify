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

## Standing rule this register establishes

A claim whose supporting artifact was never written is not "unverified" — it is **unsupported**, and
the difference is whether a sentence may stand while the work is queued. It may not. Where the
artifact can still be written, the claim is HELD pending it. Where the artifact cannot exist
(R-1, R-6, R-7), the claim is withdrawn today.
