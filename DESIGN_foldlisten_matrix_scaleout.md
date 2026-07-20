# DESIGN — fold/listen pushback matrix: iron out the 22-set, then scale to the family — seed (2026-07-20)

> **Status: forward-looking SEED, pre-registered BEFORE running.** This is a handoff, not a set of
> conclusions. It carries POINTERS (file:line / result JSONs) and OPEN QUESTIONS, plus the ordering
> the work should follow (iron the small set, then replay on the family). It deliberately does NOT
> pre-decide the contested calls below.
>
> **How to use this (entry ritual, README).** Run the faithfulness gate first: reproduce the
> committed numbers this seed points at, from the raw result JSONs, before building on them. Question
> every framing here; re-derive every number from its artifact (each JSON embeds its own
> metric/thresholds/decision_rule). Do not inherit the prior session's assumptions or context — where
> this seed states a "current reading," treat it as a claim to verify, not a fact to reuse. Freeze
> any new threshold yourself, from the data, before you run.

## Target

The fold/listen pushback matrix (says-C / says-W\* / withholds, across scale × training × push
direction) is complete and gated at n=22 but faithful-and-grounded only at 9b; the larger family
(ext2, n=82) exists only at 9b-it. Two questions, in order: **(A)** what must be true for the n=22
matrix to be faithful-at-every-scale and instrument-validated (H4) before it is trusted at scale;
**(B)** what a family-wide (n=82) replay of the same instruments looks like once (A) holds. No new
behaviour, no new metric.

## Current state — POINTERS to verify, not conclusions

Reproduce each from its artifact before relying on it.

- Six behavioural cells committed at n=22: `results_foldlisten{,_2b,_27b}/out/foldlisten_judge_fl_*_summary.json`
  (+ `foldlisten_gatev2_fl_*it.json`). Rate = moved/(moved+held), abstain excluded
  (`foldlisten_judge.py:97-101`). Re-derive the six rates and the gate decisions.
- Measurement layer: `foldlisten_judge.py:192-240` (`select_faithful_v2` / `gate_v2`) + RESULTS_FOLDLISTEN.md
  L78-96. Live scoring uses `commit_prog` (`family_generate_judge.py:201`), NOT the faithful
  `classify()` (`foldlisten_judge.py:54` imports `commit_prog`; grep confirms `classify(` is absent
  from live generators). Verify what each arm (neutral/counter/elicit) is scored with.
- `controls/faithful_rescore.py` is an OFFLINE re-labeller run this session on the 9b files only
  (CONFIG `:71-98`); outputs `out/faithful_rescore_*.json`. Its divergence from `commit_prog` per
  arm is recorded in those JSONs (aggregate `change_frac`) — re-read it; do not take the prior
  session's characterization of it.
- 9b hand-label validity: `results_foldlisten_ext/handlabel_validation.json` (n=56, commit-vs-human;
  judge diagnostic). This is the existing H4 validity precedent; read its threshold and result.
- Base free-reply decode (9b only): `results_verifier/out/family_{generate_judge,cave_diagnose}_vfam_9b.json`,
  `results_absdecode_ext2/out/family_topk_shift_vfam_9bbase.json`. Base under the foldlisten protocol
  at 9b is recorded INSUFFICIENT — reproduce and decide for yourself what the base panels can carry.
- Larger set (ext2, n=82): 9b-it fold+listen only (`results_foldlisten_r2/...9bit_ext2_summary.json`,
  byte-identical-reproducible via `fl_9bit_anchor2`) + 9b-base free-reply decode
  (`results_absdecode_ext2/`). 2b, 27b, and 9b-base-under-foldlisten are absent.
- Entity-form misses: grep `"UNRESOLVED_ALIAS"` in `out/faithful_rescore_*.json` → 3 hits
  (Astana/Nur-Sultan, DR Congo/full name, Antarctica/Antarctic Polar Desert; 1 on the 22-set, 2 on
  the 82-set). These are raw-data facts; see them yourself.

## Open questions the iron-out must resolve (NOT pre-answered here)

- **Is `commit_prog`-on-elicit the right production readout, or should the stricter `classify()` be
  ported into the live generators?** Decide from the measured per-arm divergence (re-derive it),
  from whether the counter-reasoning arm is load-bearing for any published claim, and from the H4
  standard. This seed takes no position.
- **What is the acceptable faithful↔production divergence?** No threshold is assumed here. If you
  need one, freeze it from the data before running, and record the rationale.
- **Does the 9b hand-label gate (its committed threshold) transfer to 2b/27b unchanged, and at what
  n?** Read the precedent; set the spot-check size and gate yourself.
- **How should the entity-form misses be fixed** (alias/rename table vs a different normalization),
  and does the fix change any committed count when re-run? Choose the mechanism; prove it moves the
  3 misses and nothing else via a model-free selftest.
- **What can the base panels legitimately show** given the base free-reply vs forced-elicit
  distinction and the INSUFFICIENT foldlisten result? Re-derive both arms and decide the honest
  framing; do not assume the prior session's.
- **Are there iron-out debts beyond the ones pointed at above?** Re-scan; this list is what one
  grounding pass surfaced, not a proof of completeness.

## Gates to clear before scaling (Phase A → Phase B)

State the conditions; the executing agent sets any threshold not already frozen in the repo.

- Every 22-set scale is instrument-validated to the repo's existing hand-label standard (the 9b
  precedent), by an independent hand-label spot-check — not by trusting stored labels.
- The entity-form misses are resolved (count = 0 on re-run) by a selftested fix, with no unintended
  movement in committed counts.
- The scorer question above is settled explicitly (ported or not) with the decision and its evidence
  recorded, so a later run cannot silently change the readout.
- The faithful↔production divergence is measured at every scale and judged against a threshold the
  agent has frozen.

## Phase B — replay the validated instruments on the family (ext2, n=82)

Once the gates hold, the larger set is a replay of the SAME instruments (`foldlisten_judge.py`
+gate_v2; `family_{generate_judge,cave_diagnose,topk_shift}.py`; `faithful_rescore.py` with the
82-item paths added) on the 82-item family at the model×cells currently absent. All are already
`--family`-parameterised with fraction thresholds, so they carry with no code change beyond the
entity-form fix. Decision rules are INHERITED from the 22-set only if the gates above confirm the
instrument is valid at every scale; otherwise re-derive. Faithfulness gate first (reproduce the
committed 9b-it ext2 numbers as the anchor); H3-ground each new scale by the same hand-label
spot-check. Each cell writes its self-describing summary JSON; update the RESEARCH_QUESTIONS.md
handoff seed with file:line pointers, not restated numbers.

## Explicitly NOT in scope

GPU cell enumeration, cost, run logistics (parked by request). New metrics/readouts. Mechanism
claims. Changing measurement-layer v2 without first settling the open question about it. The
model-derived-W\* arm (`DESIGN_modelderived_wstar.md`).

## Workflow (per repo idiom)

Any new instrument work authored claim-blind → model-free `--selftest` → dual review → offline
re-run → H3 grounding by an isolated reader before any count is trusted. Freeze thresholds before
running. This seed is frozen before Phase B; it fixes ordering and gates, not the contested calls.
