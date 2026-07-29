# HANDOFF — put the 2026-07-29 battery through `latent_skeptic`, then read it back onto the draft

Written 2026-07-29 by the session that produced commits `a34d6e6`, `a7189d0`, `0a6dc95`. Your job is
adversarial: this battery has NOT been through triage, it was produced by the same session that
designed it, and the register below records where I think it is weakest. Treat my confidence as data
about me, not about the results.

**State.** No GPU running (`GET /api/v1/instances` returned 0, verified at the end of the session).
Spend reconstructed from `GET /api/v1/audit-events`: **$701.71 of the $950 cap** before this session's
runs, so roughly **$240 headroom** now. Reconstruct it again yourself; every committed tally in this
repo has been wrong, including the one in the seed above this file's own commit.

---

## 0. Read these first, in this order

1. `docs/drafts/REGISTRATION_format_matched_readout.md` — the pre-registration the run was built from.
   Read §0.1 and the §0.2 amendment log (A1-A20) before the body: two thresholds were **withdrawn**
   under review rather than re-guessed, and that history is the point.
2. `out/fmt_matched_join.json` — the verdicts. The only verdict source. Prose summaries of it,
   including mine, are not evidence.
3. `docs/drafts/GROUNDING_crossvariant_scale.md` — the cross-variant/scale grounding pass over the
   vault drafts, with §14 on the withdrawal boundary. **Its §4.2 and §6 numbers predate `a34d6e6`**;
   where the run supersedes them the newer artifact governs (§13 says so).
4. `docs/drafts/OWED.md` §H — the five items this battery opened.
5. `docs/drafts/REGISTRATION_forcedfinal_distributional.md` — the B2 registration, written but with
   **no instrument yet**. This is the one that unblocks the plots.

---

## 1. Set the triage off. Exact invocation.

The harness is a git submodule at `.claude/agents/latent_skeptic` (currently `4251e76`). Rules in its
`HEURISTICS.md` (H1 isolation, H2 a crux is decided by a number that could have come out otherwise,
H3 grounding re-derives from primary artifacts, H4 the instrument is itself a claim). Roles in
`agents/triage-{reader,runner,author}.md`. Entry point:

```
Workflow({ scriptPath: ".claude/agents/latent_skeptic/triage_workflow.js",
           args: { claims: [ ...see §2... ], execute_runs: false } })
```

Notes from the script header, so you do not have to rediscover them: **pass 1 is GPU-free** by design
(skeptics acquit what committed numbers kill, the rest queue); keep `rounds:1`, because breadth is
one-confound-per-skeptic and not re-sampling; `execute_runs:true` only AFTER you have read the queues;
`author:true` fans claim-blind authors over the `author_queue`; there is **no rolled-up verdict by
design** so you read the cruxes yourself; and keep prompt strings free of em-dashes. Queue items needed
by two or more claims are hoisted to a top-level `shared_queue` and run once.

The user has asked for this review, so the orchestration opt-in exists. Budget it: eleven claims
against the ten default confounds is a large fan-out.

---

## 2. The claims to triage, pre-drafted

Evidence slices are bounded on purpose (H1). Do not widen them to include my reasoning; the point is
that a skeptic sees the numbers and not the story. Order is roughly descending load-bearing-ness.

1. **`fmt-gap-artifact`** — "The base-vs-`-it` W\* rank gap in the committed plausibility table is an
   artifact of the measured token key and the read slot, not a property of the models."
   Evidence: `out/fmt_matched_join.json` `headline` L_new 0.125/0.196/0.079 vs L_old 2.416/2.899/2.886;
   per-cell W\* `elicit`/canonical medians base->it 3.0->4.0, 3.5->5.5, 5.0->6.0;
   `results_fmt_2b9b/out/family_topk_shift_fmt_fmt_ext2_*.json`.
2. **`fmt-unresolvable`** — "After correction no residual base-vs-`-it` rank difference is resolvable at
   this instrument's resolution." Evidence: the primary triple
   `(RANK_RESOLUTION_INSUFFICIENT, RANK_RESOLUTION_INSUFFICIENT, ANCHOR_DIFFERS)`; the bf16 tie-plateau
   interval rule; `median_rank_plateau` per cell. **Attack this hardest** (§3.1).
3. **`fmt-decomposition`** — "Both the key correction and the slot correction were necessary; neither
   alone closes the gap." Evidence: key alone, slot held at `bare`, W\* 781->14.5 (2b) and
   2375.5->22.0 (9b); slot alone, key held correct, 14.5->4.0 and 22.0->5.5; `-it` onset 0.012->0.866
   and 0.037->0.963 between slots.
4. **`fmt-anchor`** — "The new instruments are the shipped instruments plus the declared changes."
   Evidence: 17 of 18 gated anchor checks `ANCHOR_REPRODUCES`; the exception is
   `27bbase/rank/same_box` `ANCHOR_DIFFERS` (3.5 vs 4.0, 17/164 rank fields, max dp 0.039).
5. **`stab-arms-neutral`** — "The re-parameterised `family_cave_diagnose_arms` is algebraically neutral,
   so the stated cause of `a4a2ae0`'s listen withdrawal does not reproduce." Evidence: §10 verdict
   `SHIPPED_SELF_IDENTICAL + ARMS_MATCHES_SHIPPED`, A1 == A2 on all 23 fields x 82 items, one box,
   `same_box=SAME_BOX`; against the cleangate result of 15 of 23 fields differing.
6. **`card-not-driver`** — "The 27b cross-box divergence tracks the card, not the driver." Evidence:
   cluster 1 = H100 PCIe @570.148.08; cluster 3 = H100 80GB HBM3 @580.105.08; this run = H100 80GB HBM3
   @**570**.148.08 and matched cluster 3; cluster 2 is a singleton on cluster 1's own card and driver.
   Fingerprint over ordered (join_key, 23 fields).
7. **`key-immaterial-rc`** — "The leading-space key does not move the content-cave labels at any `-it`
   cell but does move the headroom gate at all three." Evidence: `n_flip_faithful_RC` 3/0/7 against
   `MIN_FAITHFUL` 8, with 9b-it `KEY_EFFECT_BELOW_NOISE` at 0; `n_flip_headroom_pass` 26/11/8.
8. **`it-magnitude-survives`** — "The `-it` RC_effect magnitude is mostly not a tokenisation artifact."
   Evidence: it-base RC_effect 5.05->4.58 (2b), 3.84->2.93 (9b), 2.55->2.04 (27b) under the corrected
   key, against `MARGIN_FAITHFUL` 0.5. **This reverses a claim I made earlier in the session** (§4).
9. **`slot-matched`** — "The generation-free elicitation slot is format-matched between variants."
   Evidence: `SLOT_MATCHED` at all three scales, onset deltas 0.0244/0.0122/0.0366 against
   `ONSET_DELTA` 0.10, itself borrowed from `ARTIFACT_MAX_DELTA` (`foldlisten_judge.py:129`) and stamped
   `ONSET_DELTA_BORROWED_FROM_WITHIN_MODEL_REGIME`.
10. **`neutral-floored`** — "At `-it` the W\* probability at the neutral slot is genuinely below the
    persistence floor, not merely mis-keyed." Evidence: 9b-it `P_w_neutral` 7.69e-10 -> 4.54e-08 under
    the corrected key, a ~59x gain still under 1e-6; mass above floor at neutral stays 0/0/1 of 82
    while counter rises to 68/77/48.
11. **`push-not-attributable-base`** — "Push attribution fails at base." Evidence:
    `results_foldlisten_nelicit_*` withhold verdicts `INVERTED_NEUTRAL_HIGHER` at 9b-base and 27b-base
    both directions, `PARTIAL`/`FORMAT_ARTIFACT` at 2b-base, 0 of 3 `PUSH_ATTRIBUTABLE` against a frozen
    rule requiring >=2 of 3; neutral-arm names-one-of-pair 47/82 vs 31/82 pushed at 2b-base.

Claim 11 is NOT from this battery. Include it: it is load-bearing for the intro, it has never been
triaged, and it contradicts a sentence currently in the vault.

---

## 3. Where I think this is weakest. Attack here first.

**3.1 `RANK_RESOLUTION_INSUFFICIENT` may be doing too much work.** It suppressed the primary readout at
every scale, which is convenient: a suppressed primary cannot be wrong. The gate is constant-free
(plateau = `(P == p).sum()`, the exact complement of the strictly-greater rank) and was adopted to
replace a threshold two reviewers called fitted, so I believe it is honest. But nobody has asked
whether `median_rank +/- median_rank_plateau` is the right interval, or whether a plateau-width interval
is conservative to the point of being unfalsifiable. If the interval rule cannot ever be cleared at
n=82 with bf16 ties, the design bought its own null. **This is the single most important thing to test**,
and H2 is exactly the tool: could it have come out otherwise?

**3.2 Every threshold is transported, and stamped as such.** Each canonical-key verdict carries
`threshold_provenance = THRESHOLDS_NOT_CALIBRATED_FOR_THIS_KEY`. The stamping is honest; the transport
may not be sound. `MIN_FAITHFUL` 8 in particular now gates a **count of label flips between two keys**,
having been calibrated as a count of faithful items. A reviewer already flagged that as borrowing one
level over, and the fix I accepted was to change the statistic rather than defend the constant.

**3.3 The `elicit` slot I registered is generation-free.** `single(q + ELICIT)`: no plant, no push. It
answers the format question and it is **not** the slot the verdicts are decided on. I declared B2 out of
scope at `REGISTRATION_format_matched_readout.md:116`. That was my call and it is the reason the plots
are still blocked. Check whether any number in the battery is being read as if that slot were the
forced final.

**3.4 27b rests on one box and one draw per instrument.** 27b-base's anchor differs, 27b-it has no third
draw, every 27b-vs-committed comparison is `DISCLOSED_NOT_GATED`, and `family_topk_shift` has never had a
determinism rider at any scale. The `card-not-driver` claim (6) is a single-observation inference from
cluster membership.

**3.5 §10's same-box test leaned on a run-level provenance fallback.** `stab27b_shipA.json` carries no
per-artifact provenance, so `PROVENANCE_SOURCE_RUN_LEVEL_FILE` was used. Without it §10 would have been
`STAB27B_UNEVALUABLE` by construction, and §10 is the claim that refutes a live withdrawal. That is a
structural decision made after seeing the artifact layout.

**3.6 The instruments are large.** 1870 and 1719 lines against 441 and 419 for the shipped pair.
Selftests pass and both had an independent review that found nothing, which for files that size should
raise your eyebrow rather than settle it. H4 applies: the instrument is itself a claim.

---

## 4. My errors this session, so you can calibrate

Each was caught by an agent or by the artifacts, not by me noticing.

- I reported the "`-it` components ~3x larger" magnitude as inseparable from a tokenisation artifact.
  **Wrong** — it survives the correction. Claim 8 above.
- I said "16 of 18" anchors reproduce. It is **17 of 18**. The wrong figure is in `a34d6e6`'s message.
- I claimed a **fourth** 27b value cluster. Artifact of the join's pre-fix duplicate-key bug; the three
  draws match cluster 3. Corrected in `a7189d0`.
- I described the experiment as a **fan** (`bare->neutral` and `bare->counter` from one source) when it is
  **two parallel chains**, each running plant -> second turn -> elicited. I had the prompt builders in
  front of me earlier and still reported the structure back as a discovery. This is the error that
  matters, because it is why the elicit slot got scoped wrong (3.3).
- I wrote three scale triples in `GROUNDING_crossvariant_scale.md` §4.2 in **three different scale
  orders**. Fixed in `0a6dc95`.
- I told the user contamination was "82/82 at every base scale". Measured at record level it is
  164/164, 164/164, **162/164** counter and **163/164, 163/164**, 164/164 neutral.
- I overwrote `.last_lambda_instance`, which was the **last surviving pointer** to the nelicit runs'
  hardware. See §6.

---

## 5. Then: other issues in the battery. Where I would look.

Beyond §3, and none of these has been checked:

- **The join's own arithmetic.** `controls/fmt_matched_join.py` emitted every verdict. Its selftest
  passes and it needed three rounds of fixes to get there, two of which were tests contradicting their
  own implementations. Re-derive the headline triple and the §9.5 flip counts independently.
- **The four selftests that were wrong before they were right.** `family_topk_shift_fmt.py` shipped four
  failing or self-contradictory assertions in sequence, all of them the test disagreeing with the
  implementation. That pattern says the selftest was written against an earlier shape of its own
  helpers. Assume more of it is vacuous and look for assertions that restate rather than constrain.
- **`ANCHOR_REPRODUCES` at 17 of 18 may be too clean.** If the anchor compares fields that cannot
  differ, it is not a test. Check what is actually in each gated field group.
- **Two entries in `OWED.md` A1-A3 are stamped FIXED while their own text says verification is owed.**
  This run verifies A1/A2. A3's selftest is still unrun and is GPU-box-only.
- **`OWED.md` §G contradicts its own table** on the scope of the listen withdrawal. Named in `a7189d0`
  and deliberately not repaired, so the audit trail survives. It is load-bearing now that claim 5
  refutes the premise.

---

## 6. Provenance harm I caused, and the standing fix

`.last_lambda_instance` is a single-slot file that every launch overwrites. It now names my 27b box.
It previously named `73a2c838...`, cited at `DESIGN_elicit_context.md:606` as the nelicit 27b box, and
it was the **last** route to that run's hardware: those summaries carry no `provenance` object, neither
nelicit directory has a run-level provenance file, and neither `run_detached.log` records an
`nvidia-smi` line. So any replay of those transcripts is cross-box against its source **by
construction**, and a same-box test returns `SAME_BOX_UNVERIFIABLE` by construction.

`OWED.md` H4 is therefore urgent, not tidy: instruments must stamp provenance **per artifact**. The
B2 registration already requires it (§11.1) and suppresses a cell's primary verdict on
`PROVENANCE_PER_ARTIFACT_ABSENT`. Consider also making `.last_lambda_instance` append-only.

---

## 7. How this lands on the draft, which is the actual point

The researcher's framing, and I think it is right: **the draft at some level just needed the
distributional and neutral-elicitation changes.** Status of exactly that:

**The neutral elicitation is DONE at the generation level.** `neutral_elicit_gen` populated 164/164 at
all six ext2 cells (44/44 at `fl_9bit_anchor4`). So the plant -> reply -> elicited chain exists for both
arms at every scale, which is what the sankeys draw. `DESIGN_neutral_elicit.md` still says `ARM_ABSENT`
and is stale.

**The distributional change is NOT done, and this battery did not do it.** What exists: margin states at
`single` / `neutral` / `counter` (fold plant, shipped instrument, all six cells) and a corrected-key
distribution at `bare` / generation-free `elicit`. What is missing is the readout at the **forced-final
slot inside each arm** — `OWED.md` B2, which all three blind audits converged on independently. Until
that lands, a distributional sankey cannot reach the column the generation sankey is decided on.

**It is cheap.** The full transcripts are already persisted (`elicit_prompt`, `neutral_elicit_prompt`,
per item, per cell, with the model's reply spliced in), so B2 is a **forward-only replay**: load the
prompt, one forward pass, read the answer slot. `REGISTRATION_forcedfinal_distributional.md` registers
it at roughly $7.4 expected. Three instruments are specified and **none is written**. Blocking pre-flight
detail it found: `lambda_run.sh` ships code TO a box and fetches `out/` FROM it, and the six source
summaries are this run's **input**, so the launcher copy needs an edit that has no precedent in the repo.

**The base half is gated on a design decision, not on compute.** Base elicit contexts are contaminated
at record level (§4) by the untruncated splice at `foldlisten_judge.py:423`. Verified example: the
9b-base `neutral_elicit_prompt` at line 312 of the 9b-base summary contains the model's invented
`Q: What is the capital of Turkey? / A: Ankara`, and Ankara is that item's own W\*. A replay there would
partly measure the contamination. `DESIGN_elicit_context.md` §12 now carries **reasoned proposals** for
D-1 to D-10 (added 2026-07-29, explicitly PROPOSED not closed), with **D-7 deliberately left open** as a
presentation call. D-8 is argued up from a cost question to a correctness gate: the neutral arm is the
control for the counter arm, and a contaminated control can seed the answer it exists to baseline.

**Figures completable right now, independent of all the above:** fold-across-scales and
listen-across-scales (Figure 5) - every frozen `EXPECT` cell re-derives, zero absent cells. Two defects
to fix first that have nothing to do with this battery: `figB_listen_ext2.png` on disk is stale (dated
07-22, never rebuilt after the plural fix) though its generator's asserts pass, and
`figB_synthesis_strict_ext2.png` has **no caption file** while `make_figB_matrix.py:220` points at the
non-strict one.

**Figures that would mislead if built:** `fig_margin_flow_9b.png` is drawn at the space key and its
frozen `EXPECT` matches that key, so it silently assert-passes on defective `-it` numbers (9b-it counter
moves 27/55 -> 18/64 corrected). Any 27b distributional column needs a draw disclosure. And placing the
generation-free `elicit` column to the right of `counter` asserts a chronology that does not exist.

**One distinction to get right before drawing any base-vs-`-it` contrast.** The join suppressed the
**rank** statistic. It never adjudicated a **margin-sign** comparison, because no registered gate covers
that statistic. So that contrast is **unregistered, not suppressed** - it needs its own registration,
which is offline and cheap, not another box.

---

## 8. Decisions owed by the researcher, none of which an agent should take

1. **`OWED.md` H1** - restore the six withdrawn listen distributional cells? No run needed, data intact
   on disk, and claim 5 refutes the stated cause of the withdrawal. This is the cheapest unblock in the
   repo and it is a judgement call, not a measurement.
2. **`DESIGN_elicit_context.md` D-7** - one column or two under `DEFECT_MATERIAL`. Presentation.
3. **D-1 to D-10 otherwise** - proposals are in §12; D-1, D-5, D-9, D-10 I would treat as settled by
   evidence and proceed on.
4. **Whether a margin-sign base-vs-`-it` contrast gets registered** (§7 last paragraph).

---

## 9. What NOT to redo

The format-matched run is banked and torn down. Do not re-run it. The three commits are
`a34d6e6` (run), `a7189d0` (ledgers), `0a6dc95` (grounding corrections). `results_fmt_2b9b_failed_scp/`
is an archived failed attempt with its cause in `WHY_FAILED.md` - the launcher's hardcoded scp list does
not carry transitive imports, and the fault was **asymmetric** between the two instruments, so testing
one would not have found it. Keep that archived rather than tidying it away; it is the evidence for
`OWED.md` H5.
