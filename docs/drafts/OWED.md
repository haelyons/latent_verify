# OWED — the running list, so nothing lives only in a commit message

Written because four items flagged on 2026-07-28 existed **only** in commit bodies (`e17db28`,
`d892c3e`), which a stateless session will not find. A finding that is not in a ledger is not
recorded. Newest first within each class. Each row names what would close it.

## A. Cheap and load-bearing

| # | item | closes with | status |
|---|---|---|---|
| A1 | **`lambda_instance_id` was `null` in both provenance stamps.** It is half of the audit-log join that `RETRACTIONS.md` R-1 was withdrawn for wanting, so a stamp without it does not do the job the registration asked of it | `lambda_run.sh` now exports `LAMBDA_INSTANCE_ID` and `GIT_COMMIT` into the runner env (`:10`, `:171`); a runner must read them from `os.environ` | **FIXED 2026-07-28**, first exercised by `run_27b_decode_determinism.sh` — verify the stamp is non-null when it lands |
| A2 | `git_commit` `null` (the box has no git checkout) and `transformer_lens` `null` (no `__version__` attribute — needs `importlib.metadata.version`) | same as A1, plus the `ver()` helper in the new runner | **FIXED 2026-07-28**, same verification |
| A3 | **`_build_pool` prints and continues when the TruthfulQA download fails**, so a networkless re-run silently measures 74 items while 58 committed artifacts stamp `pool_size: 891` | one line in `controls/cave_copy_confidence_conditional.py` — raise, or assert `pool_size == 891` before measuring | **FIXED 2026-07-28** (`:328-333`, now raises). Selftest NOT run — that file imports torch at module level, so it is GPU-box-only; owed on the next box |
| A4 | `controls/classify_vs_handlabel.py` calls `classify()` at the **default `map_confidence=True`** on `elicit_gen`, against `STRICT_FIELDS` and `foldlisten_judge.py:469`. Latent (0 of 56 labels move) but it would certify the wrong register on a set where confidence mapping fires | pass `map_confidence=False`; re-run; the committed 1.000 should not move | **FIXED 2026-07-28** — re-run gives 56/56/1.0/PASS, byte-unchanged as predicted; register now stamped in the output |
| A5 | The same instrument joins hand-labels to items by **positional index**, never checking that the item it landed on is the item that was labelled | join on `q` | **CLOSED as far as the artifact allows.** The handlabel file keys labels ONLY by index string (its `notes` too), so a q-join is impossible from it. The positional join stays with the limitation machine-visible in a new `join_method` field. Inventing a key would be worse than the defect |
| A6 | The launcher's fetch globs `*summary*.json`, which does not match `family_*` outputs — both R1 boxes printed `no VALID summary fetched` and were rescued only by the full `out/` fetch | widen the glob, or accept the warning as expected for non-judge runs and say so in the launcher | OPEN — **no longer benign**, see H5: the same hand-maintained-list habit on the *ship* side killed an instrument on the box |

## B. Registrations still owed (of the original 12; **7 written, 5 remain**: B2, B3, B4, B6, B7)

| # | registration | why it blocks work |
|---|---|---|
| ~~B1~~ | **WRITTEN 2026-07-28**: `REGISTRATION_listen_distributional.md` | `family_cave_diagnose.py:214-215` plants C in both arms; a plant-W\* arm inverts the metric's sign convention and the headroom gate. Minimal-set step 5, 12 claims |
| B2 | a distribution or residual read at the **forced-final (T3) slot** (**#2**) | no instrument reads either there, and it is the slot the verdicts are decided on. Step 7, 6 claims |
| B3 | the **base arm** of the fold/listen mechanism (**#3**) | `assert is_chat` ×4; the whole 5-turn construction is chat-shaped, so retiring the assert means re-registering the arm |
| B4 | **per-scale head discovery** (**#4**) | `atp_low_confirm.py:32-34` hardwires 9b coordinates; blocks the two bolded standalone conclusions and inherits into K10 |
| ~~B5~~ | **DRAFTED 2026-07-28 by an independent agent**: `REGISTRATION_handlabel_protocol.md` — needs a read before use; see C6, which may supersede part of it | **the F1 result changes what this must be**: inter-reader agreement was 0.733 on a single-MECE-label task, so the protocol needs a coarser label set or an adjudication step, and must say which *before* readers run |
| B6 | `DESIGN_elicit_context.md` D-1..D-10 (**#7**) | the researcher's open decisions |
| B7 | `DESIGN_distributional_withholding.md` open decisions incl. its frozen power tiers (**#8**) | the researcher's; its own power table says UNC is n=0 at 2b and n=1 at 27b, so no outcome licenses a scale-general uncertainty statement |

## C. New from the R1 run (2026-07-28)

| # | item | why it is not just a note |
|---|---|---|
| ~~C1~~ | **CLOSED 2026-07-29 by `a34d6e6`**, registered before it ran: `docs/drafts/REGISTRATION_format_matched_readout.md` (written pre-data, amended twice under two independent reviewers); instruments `controls/family_topk_shift_fmt.py` / `controls/family_cave_diagnose_fmt.py` / `controls/fmt_matched_join.py`; verdict artifact `out/fmt_matched_join.json` | **The base-vs-`-it` rank gap is a FORMAT ARTIFACT** — L_new 0.125 / 0.196 / 0.079 against L_old 2.416 / 2.899 / 2.886 — and **no residual is resolvable**: the primary triple is `(RANK_RESOLUTION_INSUFFICIENT, RANK_RESOLUTION_INSUFFICIENT, ANCHOR_DIFFERS)`, quotable as a triple or not at all. The original row: median W\* rank 3 / 3 / 4 at base against 781 / 2375.5 / 3077 at ‑it was built with the QA template at base and the chat template at ‑it, so it was never evidence about plausibility |
| C2 | **A decode-path determinism rider at 27b.** The forward-layer rider returned `WITHIN_BOX_DETERMINISTIC` (0 of 14 numeric fields over 82 items), but that is `family_cave_diagnose`; the divergence lives in `model.generate` with its KV cache and different kernels | **LAUNCHED 2026-07-28** as `run_27b_decode_determinism.sh`, registration frozen in its header with all three outcomes written before the data |
| C3 | The `"withheld"` → **"no answer mentioned"** prose sweep across the drafts. Field names (`abstain`, `NEITHER`, commit `other`) are frozen artifacts and must NOT be renamed retroactively | the mapping is recorded in `RESEARCH_QUESTIONS.md` Terminology; the sweep of prose is not done |

## C-bis. From the F1 re-read (2026-07-28, later)

| # | item | closes with |
|---|---|---|
| C4 | **`label_pre_amendment` is `None` on all 8456 `per_item` records** in `out/gapclose_span_taxonomy.json` — the writer never copied it out of `label_span`, so §4.1's "both count sets are reported" is undelivered and the "vs PRE-amendment" reading is vacuous | fix the per_item writer, re-run, recompute that reading. Until then it is withdrawn, not merely caveated |
| C5 | **The committed sample file's `label_space` misrepresents what the readers saw** (12 labels incl. `NEUTRAL_ACK`; they were given 11) because the taxonomy was re-run after the amendment | the vocabulary as given is now recorded in the handread artifact. Going forward, a blind packet must be **frozen and committed before readers launch**, and never regenerated by a later run of the same instrument |
| C6 | **A registration to test the elicited slot specifically.** Post-hoc, `elicit_gen` reads 1.000 inter-reader and 0.919 vs rule — but post-hoc stratification cannot license a usability claim | a short registration: elicited slot only, its own sample, its own pre-fixed bar. This is the cheapest route to F1's 22 claims and it needs no GPU |
| C7 | **27b-`it` needs the two-decode disclosure too** — it is NOT identical between draws (`elicit_gen` 4/164, `counter_gen` 82/164, `faithful_counter` 11/164) despite matching aggregates | done in `REGISTRATION_offline_gapclose.md` §5.1 and `REDERIVE_20260728.md` §1; the drafts' 27b-`it` numbers still need the sweep |

## G. B1's listen numbers are WITHDRAWN (2026-07-29)

The clean same-box test fired and went against the change. `out/cleangate_same_box_result.json`:
`family_topk_shift_arms` is **ALGEBRAICALLY_NEUTRAL** (25/25 pre-existing fields identical, same box) but
`family_cave_diagnose_arms` is **NOT** — every logprob field differs on all 82 items, median non-zero
0.009–0.13, max 0.44. The runner header had already stated the consequence before the data: any field
differing means the 2b/9b passes were luck and **B1's listen numbers are withdrawn at every scale.**
Applied. Six cells' listen numbers are withdrawn, including the four whose gate passed.

| # | item | closes with |
|---|---|---|
| G1 | Find the **op-order** difference between `family_cave_diagnose` and `..._arms` — a diff of the forward-call sequence, not of the arithmetic. The topk twin is the existence proof that a neutral re-parameterisation of this shape is achievable | fix, then re-run the same-box test; on neutral, the listen numbers are restored |
| G2 | `family_topk_shift_arms`' listen numbers at 27b **are** usable and unreported: 27b-base `median_target_rank_bare` **4.0 fold / 1.0 listen**; 27b-it **3077 fold / 25 listen**. `OTHER_RISER` in all four arm-blocks | write them up against C1's format caveat, which still applies |

## E. OPERATIONAL HAZARD found the hard way (2026-07-29)

| # | item | status |
|---|---|---|
| E1 | **Editing `lambda_run.sh` while a launcher is executing it CORRUPTS that launcher and its EXIT trap then TERMINATES a live box.** Bash reads a script incrementally; an edit shifts byte offsets, so the launcher resumes mid-line and dies on a syntax error. Observed: `lambda_run.sh: line 183: syntax error` where the quoted text was a fragment of line 182 starting mid-word (`rt not confirmed`). Cost: the R6+R12 riders box was torn down ~1 minute into a 4-hour run and produced nothing. Two later edits put a live 27b run at the same risk | **MITIGATED 2026-07-29**: launches now use a per-run immutable copy — `cp lambda_run.sh .launcher_<tag>.sh` and invoke the copy, so edits to the original cannot reach a running launcher. `.launcher_*.sh` gitignored. The proper fix is for the launcher to copy itself, which would make this structural rather than a habit |
| E3 | **CORRECTED — I mis-diagnosed this once and the corrected version is weaker.** I first recorded the R3 box as a provider fault plus "the launcher hangs after first SSH contact with no timeout". Reading its log properly: the launcher did **not** hang, it **died of the E1 file corruption** at `line 146` (the backstop-arming line), exactly as the riders box did. Two faults were present at once — the box also genuinely became unreachable mid-`scp` and Lambda marked it `unhealthy` — and **this log cannot separate which of the two cost the billing time.** What survives as a real gap: the on-box self-destruct backstop is armed only **after** `scp` succeeds, so any failure before that point leaves billing bounded by nothing but the launcher's own liveness — and E1 shows the launcher's liveness is not something to rely on | OPEN. Fix: arm the backstop BEFORE shipping code, so the box is self-protected from the moment it exists |
| ~~E4~~ | **`instr_triangulation` at 9b needs >40 GB** — it OOM'd holding 38.60 of 39.49 GiB alone (sequential cells, no sharing) because it calls `.backward()`. No flag shrinks it without changing scope: `--no-knockout-sweep` removes a whole leg | **CLOSED — and it was already stale before this run.** It had ALREADY run at 9b on the 80 GB card in the cleangate run: `results_cleangate_27b/out/instr_triangulation_2b_curated.json`, model stamped `google/gemma-2-9b`, 42 layers / 16 heads, full scope, no OOM, own verdict `INCONCLUSIVE`. The filename AND its internal `"case"` field both say 2b while the model is 9b — read the model stamp, not either name |
| E2 | A box that Lambda marks `unhealthy` is a provider fault, not a measurement failure. The launcher's own ~4-minute sshd guard aborts and tears it down, so it is self-limiting — but the work must be relaunched, and the ledger row must not be recorded as attempted-and-failed | R3 (27b mechanism) hit this; **deferred, not abandoned** — 80 GB capacity is scarce and the 27b distributional fill is the better use of the slot |

## H. New from the format-matched run (2026-07-29, `a34d6e6`)

| # | item | closes with |
|---|---|---|
| H1 | **The `a4a2ae0` listen-arm withdrawal rests on a premise that does NOT reproduce.** §10 of this run returns `SHIPPED_SELF_IDENTICAL + ARMS_MATCHES_SHIPPED` — A1 == A2 on all 23 fields × 82 items, one box. The withdrawal honoured a registered consequence correctly; its stated CAUSE ("the code is at fault") is refuted | a decision on whether the six withdrawn cells are restored. **That is the researcher's call, not an automatic reversal** — the consequence fired on a real divergence, and what moved is the attribution of it (H2), not the fact of it |
| H2 | **`2dd19b8`'s attribution is WRONG: the 27b divergence tracks the CARD, not the driver.** Cluster 1 = H100 PCIe @ `570.148.08`; cluster 3 = H100 80GB HBM3 @ `580.105.08`; this run = H100 80GB HBM3 @ **`570`**`.148.08` and matched cluster 3. Same card + different driver = same cluster; different card + same driver = different cluster | correcting that commit's claim wherever it was carried forward. Cluster 2 remains a **singleton on cluster 1's own card AND driver**, so that cleangate draw is anomalous and is explained by neither axis |
| H3 | **The known-cluster table is a per-BOX-CLASS object, not a fixed list.** This run's three draws all matched cluster 3, so `controls/fmt_matched_join.py:139` is complete as of now | extending it per box class as new card/driver combinations run. A draw from an unlisted class is unclassifiable, not identical |
| H4 | **Provenance is stamped per RUN, not per ARTIFACT.** `results_fmt_27b/out/family_cave_diagnose_stab27b_shipA.json` carries no `provenance` object, so §10.1's same-box test had to fall back to the run-level file — stamped `PROVENANCE_SOURCE_RUN_LEVEL_FILE` in the output, with the strict per-artifact basis emitted beside it. Without that fallback §10 would have been `STAB27B_UNEVALUABLE` by construction | a forward fix in `docs/drafts/REGISTRATION_provenance.md`: **instruments** stamp provenance into each artifact, not only the **runner** into one file per run |
| H5 | **A6 is now more than benign.** The launcher's hardcoded scp list (`lambda_run.sh:93-135`) does not carry transitive dependencies, and the fault is **ASYMMETRIC between instruments** — `controls/family_cave_diagnose_fmt.py` imports `gapclose_item_joins` at module level and DIES, while `controls/family_topk_shift_fmt.py` degrades gracefully. Cost was ~$0.07 this time only because the runners run both model-free selftests before any model load | have the launcher copy itself and DERIVE its scp list from a recursive import walk, instead of a hand-maintained list — the same structural-vs-habit point as E1 |

## F. The distributional grid

Coverage is measured, not asserted, in **`docs/drafts/DIST_COVERAGE.md`** — 31 of 72
(instrument × cell × family), 4 with a listen arm. That file carries the remaining gaps in dependency
order; the two that need a registration before anything can run are **B2** (the T3 forced-final
readout — no instrument reads a distribution or residual there, and it is the slot the verdicts are
decided on) and **C1** (the format control) — **C1 is now CLOSED** (`a34d6e6`: registered, run, and the
base-vs-`-it` rank gap reads as a format artifact), so B2 is the only one of the two left.

One honest note on provenance of effort: `copyscore --sweep` at 2b was run because ledger row R6
listed it as an absent cell, **not** because a question demanded it. Its value emerged after the fact
and is real but narrow — the 2b reader `L18.H5` is a hardwired default that had never been checked
against all 208 heads, so the sweep is both the per-scale discovery procedure K7/K10 require and
evidence that the canonical coordinate is one of ~35 heads scoring median rank < 5 rather than a
unique object. It adds nothing to the *causal* copy claims.

## D. Code-first items, at the costs `CODEBLOCKS_verified.md` corrected them to

Cheapest real win first. **K3** (one line + argparse, 7 claims) needs no registration. **K11** is 16
one-line clamps blocking zero claims. **K4 / K12 / K13 / K14** are one repeated multi-line
prompt-builder refactor ×17, not the four argparse edits the ledger implied. **K1 / K2 / K5 / K6 /
K7 / K8** are all blocked on B1–B4 above, and **K8 needs a GPU re-run** — it changes the emitted
prompt, so it cannot be done offline.
