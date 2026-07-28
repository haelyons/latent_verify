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
| A6 | The launcher's fetch globs `*summary*.json`, which does not match `family_*` outputs — both R1 boxes printed `no VALID summary fetched` and were rescued only by the full `out/` fetch | widen the glob, or accept the warning as expected for non-judge runs and say so in the launcher | OPEN, benign |

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
| C1 | **A format control for the W\* bare-rank table.** Median rank of W\* is 3 / 3 / 4 at base and 781 / 2375.5 / 3077 at ‑it, and `wstar_in_bare_topk` flips true→false between 27b‑base and 27b‑it. **The ‑it cells build that slot with the chat template and the base cells with the QA template, so the gap is format-confounded and is NOT yet evidence about plausibility.** | three orders of magnitude is too large to leave as a caveated sentence. Needs the same prompt shape at both variants, registered before it is run, or the table stays uninterpretable across the variant axis |
| C2 | **A decode-path determinism rider at 27b.** The forward-layer rider returned `WITHIN_BOX_DETERMINISTIC` (0 of 14 numeric fields over 82 items), but that is `family_cave_diagnose`; the divergence lives in `model.generate` with its KV cache and different kernels | **LAUNCHED 2026-07-28** as `run_27b_decode_determinism.sh`, registration frozen in its header with all three outcomes written before the data |
| C3 | The `"withheld"` → **"no answer mentioned"** prose sweep across the drafts. Field names (`abstain`, `NEITHER`, commit `other`) are frozen artifacts and must NOT be renamed retroactively | the mapping is recorded in `RESEARCH_QUESTIONS.md` Terminology; the sweep of prose is not done |

## C-bis. From the F1 re-read (2026-07-28, later)

| # | item | closes with |
|---|---|---|
| C4 | **`label_pre_amendment` is `None` on all 8456 `per_item` records** in `out/gapclose_span_taxonomy.json` — the writer never copied it out of `label_span`, so §4.1's "both count sets are reported" is undelivered and the "vs PRE-amendment" reading is vacuous | fix the per_item writer, re-run, recompute that reading. Until then it is withdrawn, not merely caveated |
| C5 | **The committed sample file's `label_space` misrepresents what the readers saw** (12 labels incl. `NEUTRAL_ACK`; they were given 11) because the taxonomy was re-run after the amendment | the vocabulary as given is now recorded in the handread artifact. Going forward, a blind packet must be **frozen and committed before readers launch**, and never regenerated by a later run of the same instrument |
| C6 | **A registration to test the elicited slot specifically.** Post-hoc, `elicit_gen` reads 1.000 inter-reader and 0.919 vs rule — but post-hoc stratification cannot license a usability claim | a short registration: elicited slot only, its own sample, its own pre-fixed bar. This is the cheapest route to F1's 22 claims and it needs no GPU |
| C7 | **27b-`it` needs the two-decode disclosure too** — it is NOT identical between draws (`elicit_gen` 4/164, `counter_gen` 82/164, `faithful_counter` 11/164) despite matching aggregates | done in `REGISTRATION_offline_gapclose.md` §5.1 and `REDERIVE_20260728.md` §1; the drafts' 27b-`it` numbers still need the sweep |

## E. OPERATIONAL HAZARD found the hard way (2026-07-29)

| # | item | status |
|---|---|---|
| E1 | **Editing `lambda_run.sh` while a launcher is executing it CORRUPTS that launcher and its EXIT trap then TERMINATES a live box.** Bash reads a script incrementally; an edit shifts byte offsets, so the launcher resumes mid-line and dies on a syntax error. Observed: `lambda_run.sh: line 183: syntax error` where the quoted text was a fragment of line 182 starting mid-word (`rt not confirmed`). Cost: the R6+R12 riders box was torn down ~1 minute into a 4-hour run and produced nothing. Two later edits put a live 27b run at the same risk | **MITIGATED 2026-07-29**: launches now use a per-run immutable copy — `cp lambda_run.sh .launcher_<tag>.sh` and invoke the copy, so edits to the original cannot reach a running launcher. `.launcher_*.sh` gitignored. The proper fix is for the launcher to copy itself, which would make this structural rather than a habit |
| E2 | A box that Lambda marks `unhealthy` (sshd never comes up) is a provider fault, not a measurement failure. The launcher's own ~4-minute sshd guard aborts and tears it down, so it is self-limiting — but the work must be relaunched, and the ledger row must not be recorded as attempted-and-failed | R3 (27b mechanism) hit this; **deferred, not abandoned** — 80 GB capacity is scarce and the 27b distributional fill is the better use of the slot |

## D. Code-first items, at the costs `CODEBLOCKS_verified.md` corrected them to

Cheapest real win first. **K3** (one line + argparse, 7 claims) needs no registration. **K11** is 16
one-line clamps blocking zero claims. **K4 / K12 / K13 / K14** are one repeated multi-line
prompt-builder refactor ×17, not the four argparse edits the ledger implied. **K1 / K2 / K5 / K6 /
K7 / K8** are all blocked on B1–B4 above, and **K8 needs a GPU re-run** — it changes the emitted
prompt, so it cannot be done offline.
