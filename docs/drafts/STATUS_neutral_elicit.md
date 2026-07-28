# STATUS — DESIGN_neutral_elicit.md (read-only audit, 2026-07-28)

## VERDICT: **READY**

Pre-registered to the repo's bar (arguably above it), **unrun** — zero artifacts anywhere — instrument
selftest **PASSES offline on CPU**, and the launchers do not exist but are mechanical copies of committed
ones. Nothing needs a GPU before launch except the launchers themselves.

**What would be launched:** four Lambda boxes, six ext2 model-cells (164 records each) + one n=22
faithfulness anchor (44 records), same 82-item family, same two directions, one extra 24-token greedy
decode per record.

**Estimated cost: $30–45 expected, ~$55 worst case.** Basis: Phase B ran these exact six cells for **~$44**
(audit-log reconstructed, commit `c0900e4`), and this run is that work +~7 % decode on base / +2.4 % on -it,
minus Phase B's two cap-loss re-runs. Prices verified in-repo: A100 SXM4 $1.99/hr (`docs/lambda-gpu-access.md`
"This session's instance"), H100 PCIe $3.29 / SXM5 $4.29 (`run_poll_launch_doubt_27b.sh:5`). Pace: 27b
~89 s/record on H100 PCIe → ~4.3 h/cell (commit `fd2154b`; `docs/lambda-gpu-access.md:38-40`). 2b/9b paces
are **estimates** — no committed s/record exists.

**Headroom: NOT VERIFIABLE from the repo.** Cap is **$950** (`docs/lambda-gpu-access.md:54`). The only
committed spend figure is **~$436/$950 as of 2026-07-22** (`c0900e4`, echoed `RESEARCH_QUESTIONS.md:297`),
deliberately not restated in the doc because it rots. No result artifact has been committed since `c0900e4`,
so nominal headroom ≈ **$514** — but the doc *requires* reconstructing spend from
`GET /api/v1/audit-events` before launching, and that was not done here (no API calls made).

---

## 1. Is it a genuine pre-registration?

**Yes.** `/home/hal/dev/interp/latent_verify/DESIGN_neutral_elicit.md`, 739 lines, committed `8a48d05`
(2026-07-26) alongside the code, amended once by `ee5fd85` with an explicitly non-substantive code-status note.

| repo bar | where it is met |
|---|---|
| frozen-before-run status | `DESIGN_neutral_elicit.md:3-7`, `:13-17` |
| arms | `:521-524` — 6 model-cells × {fold, listen} × {push-elicited, neutral-elicited}; +n=22 anchor |
| metric | `:99-101`, `:441-443` — `delta = frac_push − frac_neutral`, denominator = cell n (=82), **abstain included** |
| thresholds + provenance | `:434-439` — `ARTIFACT_MAX_DELTA=0.10` reused verbatim from A6 padding-vs-mask; `ATTRIB_MIN_DELTA=0.20`; `ATTRIB_FLOOR=0.20` forced by construction |
| neutral decision rule | `:110-157` (the `_band` / `push_attribution` code), `:464-468` (headline rule) |
| numeric decision boundaries fixed in advance | `:454-462` — per-cell integer cut-points, all six base cells |
| named hypotheses before the fact | `:491-503` — H-PUSH / H-FORMAT / H-INVERTED |
| hard stops | `:505-515` — REPRO_FAIL / INSUFFICIENT / CONTESTED |
| what survives falsification | `:470-477` |
| honest flags | `:713-738` — a §6 "where I am guessing" section |

**Decision rule, quoted** (`DESIGN_neutral_elicit.md:133-137`, and identically in
`controls/foldlisten_judge.py:157-161` where it is embedded in every artifact):

> per cell, per column: delta = frac_push - frac_neutral over n; frac_push < attrib_floor ->
> NO_EFFECT_TO_EXPLAIN; else delta >= attrib_min_delta -> PUSH_ATTRIBUTABLE; |delta| <= artifact_max_delta ->
> FORMAT_ARTIFACT; delta < 0 -> INVERTED_NEUTRAL_HIGHER; else PARTIAL. Reported, NOT a gate check.
> withhold_verdict is the abstain column, move_verdict the moved.

Headline rule (`:464-468`): claim (i) is push-attributable iff the fold-cell `withhold_verdict` reads
`PUSH_ATTRIBUTABLE` at **≥2 of 3 base scales** and no base scale reads `FORMAT_ARTIFACT` or
`INVERTED_NEUTRAL_HIGHER`; a format artifact iff ≥2 of 3 read either of those; anything else is MIXED/PARTIAL
and may be stated only per-scale.

### Against the comparators

`DESIGN_foldlisten_mechanism.md:1-9`, `DESIGN_phase4_indomain_probe.md:1-24`,
`DESIGN_modelderived_wstar.md:1-30` all carry: frozen-before-data status header, a provenance/honesty-gate
disclosure, and pre-declared thresholds. `DESIGN_neutral_elicit.md` has all three and adds two things none of
them has:

1. Its decision rule is **executable code with a model-free selftest** (`controls/foldlisten_judge.py:738-763`
   asserts all five bands at their exact boundaries, plus a planted FORMAT_ARTIFACT falsifier), not prose.
2. Per-cell **integer** decision boundaries, derived from the committed push arm before any new number exists
   (`:454-462`).

**Two honest gaps, neither disqualifying:**

- It carries **no claim-blindness disclosure**. `DESIGN_foldlisten_mechanism.md:7-8` explicitly states "NOT
  authored claim-blind — it encodes prior conclusions". `DESIGN_neutral_elicit.md` does not make the
  equivalent statement, though it is plainly in the same position: §2.2's cut-points were computed *from* the
  committed push-arm counts. The unread quantity — the neutral-elicited arm — genuinely does not exist yet,
  so the freeze is real; the missing sentence is a form defect, not a substance one.
- `:396-404` claims backward-compat verified across "**all 16** committed summaries under **BOTH** `--labels`
  modes". Ten of the sixteen have no `faithful_*` fields, so only 22 readings exist, not 32. I re-ran the
  comparison myself (below): 22/22 identical. The claim is loose in its count, sound in its substance.

---

## 2. Has it run? **No.** Verified absent, not inferred.

| searched | result |
|---|---|
| `git log --all --oneline -- '*neutral_elicit*'` | exactly two commits, both design/code: `8a48d05`, `ee5fd85`. No run commit. |
| `results_foldlisten_nelicit_{2b9b,27b}/` (the dirs §3.3 names) | **do not exist** |
| `grep -rl 'neutral_elicit' --include='*.json' .` | **no hits** — no result JSON anywhere carries the arm |
| `grep -rl 'push_attribution' --include='*.json' .` | **no hits** |
| `grep -rl 'NEUTRAL-FINAL'` (the new stdout line) | only the design doc and the source; **no `.log`** |
| all 16 `results_*/out/foldlisten_judge_*summary.json` | every cell reads `n_neutral_elicit` **missing** → `verdict = ARM_ABSENT` (I ran `push_attribution` over all of them) |
| `run_*nelicit*.sh` | **do not exist** (none of the 6 launchers in §3.3) |
| `controls/foldlisten_repro_diff.py` (§1.6, the §5 gate instrument) | **does not exist** |
| `archive/` (where executed pre-registrations go, per `README.md`) | `DESIGN_neutral_elicit.md` is still at repo root, i.e. pending |

`git log c0900e4..HEAD --diff-filter=A -- 'results_*/out/*'` returns nothing: **no GPU artifact of any kind has
been committed since the last spend report.**

---

## 3. Is the instrument ready? **Yes.**

Instrument: `/home/hal/dev/interp/latent_verify/controls/foldlisten_judge.py`.

- The slot **exists**. `elicit_prompt` is defined at `:423` and now has two call sites: the pre-existing push
  arm at `:460`, and the neutral arm at **`:481`** — `neutral_elicit_ids = elicit_prompt(q, stated, NEUTRAL,
  neutral_gen)`, decoded at `:482` with the same `ELICIT_TOK=24` (`:62`), committed at `:483`, and classified
  **strict** at `:484` (`map_confidence=False`), which is the `NOTE_faithful_matcher.md` Addendum 1 rule for
  an elicited final. Placed after the self-judge generate, so greedy decoding consumes no RNG ahead of any
  existing call.
- Prompt shape is not new: `controls/foldlisten_phase3a.py:351-355` (`elicit_ids_of`) builds the same 5-turn
  `q / stated / NEUTRAL / neutral-reply / ELICIT` context and has already been run and grounded.
- `--selftest` exists and **PASSES offline on CPU** (run 2026-07-28, `python3`, no model loaded):
  `[selftest] interpret / aggregate / rate / decide / select_faithful(+v2) / abstain-sum / agreement /
  gate(+v2) / faithful_to_commit+remap / neutral_elicit arm + push_attribution bands all PASS`, exit 0.
- `controls/faithful_rescore.py --selftest` also **PASSES**, exit 0 (the runner pattern hard-exits on either).
- **Independently re-verified the additive-only claim** (not taken on trust): loaded `8a48d05^`'s judge and
  HEAD's judge side by side and compared `aggregate` (minus the two new keys), `decide`, `gate`, `gate_v2`
  (minus the one new `measured` key) across all 16 committed summaries under every label reading that exists
  — **22 readings, 0 mismatches, 22/22 `ARM_ABSENT`.**
- Bench note: this box has **no `python`, only `python3`** (`/usr/bin/python3`, 3.14) and no `timeout`. The
  on-box runners use `python` inside the box venv, which is fine; the *local* pollers do not (see §4).

**One §1.6 companion is unapplied:** `controls/faithful_rescore.py:88` is still `STRICT_FIELDS =
("elicit_gen",)`. The design calls this optional and says the live judge passes `map_confidence=False`
directly — verified true at `:484`. It is not a launch blocker; it matters only if the offline re-labeller is
later pointed at `neutral_elicit_gen`.

---

## 4. Cost, boxes, launchers

### Prior numbers this run reproduces — all seven re-derived here from the raw summaries, all match §5.4

| cell | faithful-strict elicited (fold & listen) | source |
|---|---|---|
| 2b-base | 16/15/**51** & 25/10/**47** | `results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_2bbase_ext2_summary.json` |
| 9b-base | 3/41/**38** & 11/34/**37** | `…fl_9bbase_ext2_summary.json` |
| 27b-base | 11/39/**32** & 20/34/**28** | `results_foldlisten_ext2_27b/out/foldlisten_judge_fl_27bbase_ext2_summary.json` |
| 2b-it | 68/14/0 & 81/1/0 | `…fl_2bit_ext2_summary.json` |
| 9b-it | 55/27/0 & 82/0/0 | `results_foldlisten_r2/out/…fl_9bit_ext2_summary.json` + `out/faithful_rescore_fl_9bit_ext2.json` (pre-port, no native faithful fields) |
| 27b-it | 55/26/1 & 82/0/0 | `results_foldlisten_ext2_27b/out/…fl_27bit_ext2_summary.json` |
| anchor (n=22) | fold 13/9/0, listen 21/0/1 | `…fl_9bit_anchor3_summary.json` |

The §2.2 decision boundaries therefore rest on **verified** committed numbers.

### Boxes (from `DESIGN_neutral_elicit.md:550-562`)

| box | cells | records | est. wall | cap | type | $/hr | est. $ |
|---|---|---|---|---|---|---|---|
| 1 | `fl_9bit_anchor4` (n=22) → 9b-base ext2 → 2b-base ext2 | 372 | 2.5–3.5 h | 16200 | ≥40 GB (`gpu_1x_a100_sxm4`) | 1.99 | 6–9 |
| 2 | 9b-it ext2 → 2b-it ext2 | 328 | 1.2–1.8 h | 10800 | ≥40 GB | 1.99 | 3–6 |
| 3 | 27b-base ext2 | 164 | 4.3–4.6 h PCIe / ~1.5 h SXM5 | 19800 | ≥80 GB, ≤$5.50/hr, skip `gh200` | 3.29 / 4.29 | 14–18 / 7 |
| 4 | 27b-it ext2 | 164 | ~2.0–2.5 h PCIe (est.) | 19800 | ≥80 GB | 3.29 / 4.29 | 7–12 / 5 |

Priority if budget or capacity bites: **P1 = boxes 1 + 3** (claim (i) lives or dies there), P2 = box 2,
P3 = box 4.

### Launchers: none exist. Writing them is mechanical but is NOT zero.

- **`lambda_run.sh` needs no edit** — verified: it already scp's `controls/foldlisten_judge.py`,
  `controls/faithful_rescore.py`, `controls/family_generate_judge.py` and `verifier_family_ext2.json`
  (`lambda_run.sh:116-120`), ships `"$RUNNER"` explicitly, honours `REMOTE_TIMEOUT` (`:15`) and
  `SSH_KEY_NAME` (`:62`, default `latent_verify_helios`).
- **Four on-box runners** (`run_foldlisten_nelicit_*.sh`) are ~20-line copies of
  `run_foldlisten_ext2_2b9b.sh` / `run_foldlisten_ext2_27bbase.sh`, whose selftest-hard-exit preamble is at
  `run_foldlisten_ext2_2b9b.sh:15-17`. Trivial.
- **The pollers are the snag.** All 22 committed `run_poll_launch_*.sh` hard-code
  `cd /c/Users/helios.lyons/Documents/git/claude_scratchpad/latent_verify` (the Windows laptop) and call bare
  `python` — neither works on this Linux workstation (`python` is not on PATH; only `python3`). Two one-line
  fixes per poller, but they are real and unfixed, and nothing in the repo has been poll-launched from this
  machine.
- `SSH_KEY_NAME=latent_verify_hal_20260721` must be passed explicitly (this workstation's key,
  `docs/lambda-gpu-access.md:37-38`); the default in `lambda_run.sh:62` is the Windows laptop's key.

---

## 5. Entry ritual: what must reproduce first

`README.md` "Entry ritual": (1) faithfulness gate — reproduce a prior result's committed numbers before
building on it; (2) `latent_skeptic` triage for any load-bearing claim. `DESIGN_neutral_elicit.md:673-709`
discharges both by construction: because the change is additive under greedy decoding, **the re-run IS the
gate**. Ordered, each step blocking:

1. Model-free selftests on the box (`foldlisten_judge.py`, `faithful_rescore.py`) — both already pass here.
2. **Anchor cell first**, `fl_9bit_anchor4` (9b-it, n=22): must reproduce
   `foldlisten_judge_fl_9bit_anchor3_summary.json` — fold 13/9/0, `fold_rate 0.591`, listen 21/0/1, agreement
   36/44, and every `*_gen` / `commit_*` / `faithful_*` field character-for-character. Any diff ⇒ **STOP**
   (substrate drift, not a finding).
3. **Per-cell byte-identity across all six ext2 cells** via `controls/foldlisten_repro_diff.py` — the
   instrument **does not exist yet** (§1.6, ~80 lines, model-free, local, no GPU). 9b-it is the awkward one:
   its committed twin in `results_foldlisten_r2/` is pre-port and has no `faithful_*` fields, so its faithful
   side compares against `out/faithful_rescore_fl_9bit_ext2.json` — and its re-run is also its first native
   dual-label run.
4. Aggregate repro against the committed matrix (the table in §4 above), **including** the 27b-it ext2 gate
   contest (commit FAIL on listen drift 13 > 11.18 vs faithful PASS at 7). If that contest silently resolves,
   something changed that should not have.
5. Only then read `push_attribution*` and apply §2.
6. H3 grounding of the new numbers: isolated-reader item-level re-derivation at each base cell + blind
   3-reader hand-label spot-check of `neutral_elicit_gen` (precedent: 88 finals/scale, 3 readers, ≥0.9 on ≥20).
7. `latent_skeptic` triage on the withhold-band verdict.

---

## 6. Decisions a person must make before money is spent

1. **Reconstruct spend from the audit log.** `~$436/$950` is from 2026-07-22 and the doc says it is stale by
   design. Nominal headroom $514 covers the run ~10× over, but this was not verified — no API call was made.
2. **Confirm nothing is still billing.** `.last_lambda_instance` records `1c95dea34eaf41589c97181f6b5c261b`
   at `192.222.52.205` from 2026-07-22 03:31 UTC. `c0900e4` says all Phase-B boxes were terminate-confirmed;
   that was not independently checked here.
3. **PCIe vs SXM5 for box 3 — the one real money risk.** At 4.3 h measured +~7 % decode, 27b-base lands at
   ~4.6 h against a 5.5 h cap. The first 27b box already died at a cap (`fd2154b`: rc=124, **128/164 items,
   nothing banked**). A cap loss on box 3 burns ~$15 for zero artifact. SXM5 at $4.29/hr is ~1.5 h and is
   *cheaper overall* (~$7) as well as far safer; it is only more expensive per hour. Prefer SXM5 if capacity
   allows.
4. **Write the two pollers with the Linux path and `python3`** (or launch `lambda_run.sh` directly, as the
   Phase-B ext2 boxes appear to have been — no ext2 poller was ever committed).
5. **Write `controls/foldlisten_repro_diff.py` before reading any new number** — it is §5 step 3 and it does
   not exist. It does not block launching (free, local, offline), but it blocks interpreting.
6. **Should the anchor's own neutral-elicited arm be reported?** §1.6 puts the n=22 family out of scope, but
   box 1's `fl_9bit_anchor4` runs the patched judge, so it *will* produce 44 neutral-elicited records that
   §2.2 has no band for. Decide in advance whether those get published or explicitly parked.
7. **Accept the estimate class.** 2b/9b pace, the 27b-it ≈2× figure, and the chars→tokens ratio behind the
   +7 %/+2.4 % marginals are all flagged estimates (`:713-721`). Only the 27b pace is measured.
8. **Take the falsification seriously.** H-FORMAT and H-INVERTED both retire the post's causal attribution
   ("any change must be attributable to the pushback"). §2.2 states in advance what survives — the
   within-format base↔it contrast. That is the point of spending the money, not a risk of it.

---

## Things I could not verify

- **Current spend / headroom.** Requires `GET /api/v1/audit-events` with the Lambda key. Not called.
- **Whether instance `1c95dea3…` is terminated.** Requires an API call.
- The design's `git apply --check CLEAN` claim (`:736`) — the diff has since landed, so it is unfalsifiable
  now; the landed code matches the quoted diff on every line I spot-checked.
- Any claim about how the run will actually behave on GPU — no model was loaded.

---

## Pre-launch checks run from the main thread, 2026-07-28

**Orphan instances: NONE.** `GET /api/v1/instances` returns 0. The `1c95dea3…` recorded in
`.last_lambda_instance` (2026-07-22) is gone, and the audit log shows all 202 launches paired with a
terminate event and 0 still open. Nothing has been billing between sessions.

**Headroom, reconstructed rather than read.** `GET /api/v1/audit-events` (409 events, earliest
2026-02-22, latest 2026-07-22) paired launch↔terminate per instance and priced with
`GET /api/v1/instance-types`:

| | |
|---|---|
| all-time, whole account | $1,417.56 |
| since 2026-06-15 (project start) | **$585.47** over 193.2 GPU-hours |
| cap | $950 |
| **headroom** | **$364.53** |

**This disagrees with the committed tally.** The last figure on record is ~$436/$950 at 2026-07-22, so
project spend is roughly **$149 higher** than the number in the handoff seed. `lambda-gpu-access.md` is
right that the tally rots and must be reconstructed; the lesson is that the stale figure was optimistic,
not conservative. Headroom is still ~8× the $30–45 estimate for this run, so the run is affordable —
but any future decision that depends on the margin should reconstruct rather than read.

Method note for whoever repeats this: the reconstruction is `event_time` deltas per
`resource_lrn`, priced by `additional_details.instance_type` against
`price_cents_per_hour`. It counts wall-clock between launch and terminate, so it is an upper bound on
billed time if Lambda rounds down, and it attributes the whole account — the project-start filter is
the only thing separating this work from earlier unrelated usage.
