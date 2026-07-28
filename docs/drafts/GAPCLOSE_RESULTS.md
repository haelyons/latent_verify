# GAP-CLOSE RESULTS — the FREE / OFFLINE class

Outcome of `GAPS_RECONCILED.md` §4.1, worked in the ledger's own cost order, all offline, $0 GPU.
Pre-registration: `REGISTRATION_offline_gapclose.md`, committed before any value was computed.
Every decision below is the instrument's own, embedded in its artifact beside its `metric`,
`thresholds` and `decision_rule` — read the JSON, not this table.

| gap | instrument | artifact | decision |
|---|---|---|---|
| **F1** span taxonomy | `controls/gapclose_span_taxonomy.py` | `out/gapclose_span_taxonomy.json`, `…_sample.json`, `…_handread.json` | **TAXONOMY_UNUSABLE** — fails its registered validation |
| **F2** nine item joins | `controls/gapclose_item_joins.py` | `out/gapclose_item_joins.json` | **ALL_JOINED** (8 of 9 pairings answered; [86] untested, see below) |
| **F3** contamination census | `controls/gapclose_contam_census.py` | `out/gapclose_contam_census.json` | **CONTAMINATED** |
| **F5** mention-anywhere register | `controls/gapclose_small.py mention` | `out/gapclose_mention_register.json` | **REGISTER_DISTINCT** |
| **F6** fold-rate significance | `controls/gapclose_small.py sig` | `out/gapclose_foldrate_sig.json` | 9 tests, **6 DIFFERS / 3 NOT_DISTINGUISHABLE** |
| **F7** neutral-slot W\* rank | `controls/gapclose_small.py rank` | `out/gapclose_neutral_rank.json` | DESCRIPTIVE_ONLY — median 119 reproduces |
| **F8** the 67-of-74 reconciliation | `controls/gapclose_small.py arm93` | `out/gapclose_p93_reconcile.json` | **RECONCILES** — so the printed number, not the artifact, is wrong (→ `RETRACTIONS.md` R-6) |
| **F9** classify-vs-handlabel at 2b/27b | — | `out/gapclose_f9_register_check.json` | **RETRACTED AS A GAP** — the measurement already existed (→ R-4) |
| **F10** the base-cell gate | existing `foldlisten_judge.py --gate` | `out/gapclose_base_gate.json` | **ALL_BASE_CELLS_FAIL**, 15 runs, thresholds transported and stamped as uncalibrated |
| **F11** merge the strict labels | `controls/gapclose_cells_faithful_merge.py` | `out/gapclose_cells_faithful_*.json` | **6 MERGED**, and top-level **MERGE_RULE_DISAGREES_WITH_COMMITTED** — it refused to tune itself to match |
| **F4** hand-labels for uncovered cells | — | — | **NOT DONE.** Protocol is registration owed #6 and deliberately not written by the agent that would run the readers |

**Claims closed: not 42.** The ledger's own arithmetic was wrong by at least one ([36] double-counted;
F1's row cites no IDs at all — `REGISTRATION_offline_gapclose.md` §13). Of the corrected 41, F9's share
is withdrawn and **F1's 22 do not close** — its taxonomy is not usable, so the span-level claims remain
unsupported except where they rest on one of the four categories that survived per-category. F2, F3,
F5–F8 and F10 close their share and are persisted.

## CORRECTION (2026-07-28, later the same session) — F1's diagnosis was wrong

The registered verdict `TAXONOMY_UNUSABLE` **stands**: it is the pooled strict number against a
threshold fixed before the data, and it is not revised. But the *diagnosis* I attached to it — "the
construct is not reliably human-decidable at this granularity" — is **refuted by this session's own
numbers**, found by an independent agent re-reading the artifact and verified in the main thread.

Stratifying by slot (**declared post-hoc; not registered, so it licenses no usability claim of its own**):

| slot | n | inter-reader | reader A vs rule | reader B vs rule |
|---|---|---|---|---|
| `elicit_gen` | 37 | **1.000** | **0.919** | **0.919** |
| `neutral_elicit_gen` | 9 | **1.000** | 0.667 | 0.667 |
| `counter_gen` | 37 | 0.946 | 0.541 | 0.486 |
| `neutral_gen` | 37 | **0.189** | 0.054 | 0.054 |

**30 of the 32 disagreements are one confusion cell** — A `OFF_TARGET` / B `WITHHELD_ASSERTED`, every
one in `neutral_gen`, every one on a span the rule calls `NEUTRAL_ACK`: the label the readers'
vocabulary did not contain, because the §4.1 amendment was made *after* they were launched. Excluding
those spans, inter-reader is **88/90 = 0.978**, above the TRUSTED bar.

So the failure is a **vocabulary gap I introduced by amending mid-flight**, concentrated in the one
slot where neutral acknowledgements live — not an undecidable construct. And the elicited slot, which
is where every headline count in this project is taken, reads 1.000 inter-reader and 0.919 against the
rule. What that earns is **a new registration to test the elicited slot specifically**, not a
retroactive pass.

Two defects in my own artifacts, found the same way and recorded in
`out/gapclose_span_taxonomy_handread.json`:

1. `label_pre_amendment` is `None` on all 8456 `per_item` records — the writer never copied it out of
   `label_span`. So §4.1's promise that "both count sets are reported" is **not delivered**, and the
   "vs PRE-amendment rule" reading was computed against a fallback: it is identical to the strict
   reading *by construction, not by measurement*. That reading is withdrawn as vacuous.
2. The committed sample file's `label_space` lists 12 labels **including** `NEUTRAL_ACK`, because the
   taxonomy was re-run after the amendment and regenerated the sample. The readers received the
   11-label pre-amendment vocabulary. The committed artifact therefore misrepresents what they saw;
   the vocabulary as actually given is now recorded in the handread artifact, because it was not
   otherwise recoverable from the repo.

## The three results that were not on the ledger

Each came out of an instrument refusing to do the convenient thing.

1. **Scorer drift, measured repo-wide for the first time** → `out/gapclose_scorer_drift.json`.
   Surfaced because F11's merge instrument was registered to *fail loudly* rather than tune its
   arithmetic to match a committed block. Recomputing every committed `faithful_*` label with the
   current scorer: **2 disagreements in 6704 (0.03%)**, both in one file, both the same tie-break rule,
   both in a diagnostic slot. `PORT_DRIFTED_DIAGNOSTIC_ONLY` — the port is 99.97% equivalent with a
   bounded, named exception, and no headline count moves. **Three independent routes reached it**: the
   merge's against-committed check, my own recomputation from the anchor summary, and the merge's
   `label_disagreements` diagnostic, which localises it to the label inputs (`counter: 2, elicit: 0,
   neutral: 0`) rather than to the merge arithmetic.

   A provenance note that is itself the policy in `REGISTRATION_provenance.md` working: the merge
   instrument was revised *after* its first run, so its first outputs no longer matched the committed
   code. Rather than leave code and artifacts disagreeing, the stale six were moved to
   `out/superseded_20260728/` and the current version re-run — the earlier version merged 5 cells
   because it let an incidental anchor match suppress a merge. Both decisions survive at distinct
   paths; neither was overwritten in place.

2. **The 891-item pool is not reproducible from the repo** → `CODEBLOCKS_verified.md` §2.
   817 of its 891 items are downloaded at run time, and `_build_pool` **prints and continues** when the
   download fails. A networkless re-run silently measures 74 items while 58 committed artifacts stamp
   891. One line — raise, or assert `pool_size == 891` — closes it.

3. **The same defect the prior session retracted, recurring inside the instrument built to check it**
   → `REGISTRATION_offline_gapclose.md` §4.1. `HEDGE_LEADING` bundles genuine uncertainty with neutral
   acknowledgements; using `is_hedge` as an uncertainty signal read "You're welcome!" as withholding on
   59–81 of 82 spans per ‑it neutral arm. Both blind readers independently rejected that label.

## Two measurements that are new, and are not claim-closures

- **`runaway` is a second perfect base/‑it dissociation.** The taxonomy's runaway flag fires on 74–82
  of 82 at every base cell and slot, and **0 of 82** at every ‑it cell — reached from a different
  measurement than F3's structural census, which found the same 492/492 vs 0/492 split.
- **The counter-slot and elicit-slot labels disagree on 31–64 of 82 at base and 4–16 at ‑it.** This is
  *not* claim [86]: that claim's "two layers" are the polarity layer (`sign(M0)` against
  `sign(Mc_counter)`), which this join does not compute. [86] stays untested, and is recorded as such
  rather than credited to a number that happens to sit nearby.

## What the class did not do, and should be picked up next

- **F4** — the blind 3-reader protocol for the uncovered cells (9b×ext2, any base ext2 cell, any listen
  cell, the T3n slot). Registration owed #6. The F1 result changes what this protocol must be: with
  inter-reader agreement at 0.733 on a single-MECE-label task, a 3-reader protocol needs either a
  coarser label set or an adjudication step, and the protocol must say which before readers run.
- **The four surviving categories are the honest granularity.** COMMITS_C / COMMITS_W /
  WITHHELD_UNCERTAIN / WITHHELD_ASSERTED cleared the caveat bar. A taxonomy that stops there is
  defensible; the fine-grained splits are not, at either this session's rule or the prior session's
  single-reader hand dict.
- **`RESIDUAL_UNLABELED` is 538 of 8456 spans**, concentrated at base (21–31 per base cell against
  0–1 at ‑it). That is a measured limit of the existing scorer helpers, and it is invisible to any
  withheld-only taxonomy because it has no denominator there.
