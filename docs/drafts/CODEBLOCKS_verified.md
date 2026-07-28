# CODE-BLOCK LEDGER, VERIFIED BY READING THE LINE

`GAPS_RECONCILED.md` §4.2 classes 15 code-blocked gaps as "one-line clamp", "argparse" or "design
change". Every cited line was read this session by two independent readers, and the classification
re-derived rather than inherited. **Six of the fifteen were understated**, one class was resolved
outright, and one item's cost class is wrong in a way that matters for scheduling.

Verdict vocabulary: `ARGPARSE_ONLY` (add a flag, nothing else) · `ONE_LINE_CLAMP` · `MULTI_LINE_MECHANICAL`
(a repeated pattern, ~3–20 lines per file) · `DESIGN_CHANGE` (a threshold, metric or registration
moves) · `NOT_BLOCKED`.

| id | ledger said | verified | why the ledger's reading fails, or holds |
|---|---|---|---|
| K1 | design change | **DESIGN_CHANGE** ✔ | Holds. `family_cave_diagnose.py:214-215` plants the literal `C` in both arms; the same construction is in `family_topk_shift.py:229-230`, `modelw_candidates.py` and `gen_outputs_table.py:54`. Planting W\* inverts `RC_effect = Mc_neutral − Mc_counter`, moves `faithful_cave`'s target token from `aid` to `cid`, and re-points the `headroom_pass` reading. Four things re-register, not one flag |
| K2 | design change | **DESIGN_CHANGE** ✔ | Holds. `ELICIT` exists in the phase files only as a *generation* target (`ELICIT_TOK = 24`); no `cave_*` control reads past T2. A T3 readout is new instrumentation, though it disturbs no existing threshold |
| K3 | one line + argparse | **ONE_LINE_CLAMP + ARGPARSE** ✔ | Holds, and cheaply: `gen_outputs_table.py:21 ITEMS` is 4 demo items against a complete `:42 CELLS`, and the item field names already match the family schema (`q`, `correct`, `Wstar`) |
| K4 | "argparse + template branch ×14 — mechanical" | **MULTI_LINE_MECHANICAL ×14** ⚠ | Understated. None of the 14 has a chat path to call; each is QA-shaped throughout, so each needs the conditional prompt builder copied from `job_truthful_flip.py:147-155` — ~15 lines per file, not a flag. The ledger's "mechanical" is right; its "argparse" is not |
| K5 | one line + argparse, invalidates thresholds | **DESIGN_CHANGE** ⚠ | Understated as a cost. The flag is one line; the substrate swap invalidates every threshold calibrated on the 891 pool, so the work is a recalibration with the edit attached. See §2 for what the 891 actually is |
| K6 | design decision | **DESIGN_CHANGE** ✔ | Holds, and more strongly than stated. All four asserts read "*registered on the ‑it substrate (C5); run with `--chat`*". Removing them is insufficient: turn construction is a role/content message list, and challenge-span indices are recomputed from chat tokenisation. The 5-turn construction is chat-shaped end to end |
| K7 | design change, per-scale discovery | **DESIGN_CHANGE** ✔ | Holds. `atp_low_confirm.py:32-34` is 18 hardwired 9b `(L,H)` pairs plus `NH_9B = 16`, flattened as `f = L·NH_9B + H`. 2b and 27b have different head counts, so a discovery step must run *before* the patch and emit per-scale coordinates |
| K8 | design change | **DESIGN_CHANGE + REQUIRES A RE-RUN** ⚠ | The correction is not offline. `foldlisten_judge.py:423-430` echoes `prior_gen` into the emitted prompt; `faithful_rescore.isolate_span` truncates *after* generation. Truncating before the echo changes the string the model is shown, so the elicit generation must be regenerated. Consistent with minimal-set step 9 being CODE + GPU — but any reading of K8 as an offline fix is wrong |
| K9 | design change | **DESIGN_CHANGE** ✔ | Holds. `worker.py` is a load-once persistent worker that `exec`s job scripts against a model fixed at `poc_minimal.py:51`, with transcoders pinned alongside. Unpinning changes the worker protocol, not a constant |
| K10 | one line + a per-scale copy head | **ONE_LINE + SIGNATURE** ⚠ | `cave_copy_confidence_conditional.py:93 COPY_HEAD = (18,5)` is not read by `run()`; exposing it needs the argparse line *and* a signature change. The real cost is the dependency: it inherits K7's discovery step |
| K11 | one-line clamp ×17 | **ONE_LINE_CLAMP ×16, NOT_BLOCKED ×1** ⚠ | `cave_reader_pathpatch.py` already exposes `--layer`, so it is not blocked. The scale-general pattern to copy exists and is documented in `cave_residstate_anyscale.py:11` — `AXIS_LAYER := round(0.667·n_layers)`, clamped, with `READ_LAYERS` derived at ±4 |
| K12 | argparse | **MULTI_LINE_MECHANICAL** ⚠ | `realized_attention.py:37 HEADS` is not threaded into `run()`; ~3 lines (flag, signature, parse) |
| K13 | one-line branch | **MULTI_LINE_MECHANICAL** ⚠ | `scale9b_doubt_direction.py:58-59` calls `chat()` unconditionally at three sites and defaults `--name` to `-it`. Base support is the same conditional-prompt-builder refactor as K4, ~15–20 lines |
| K14 | argparse | **MULTI_LINE_MECHANICAL** ⚠ | `ov_behavioral_scale.py:153-156` exposes `--name-it` only and `run()` takes one model; ~3–4 lines |
| K15 | "cannot be costed until resolved" | **ALL FOUR RESOLVED FROM THE CODE** ✔✔ | See §1. No investigation round is needed |

## 1. K15 — the four "capability ambiguities", resolved

| ambiguity | resolution, from the code |
|---|---|
| which transcoders `cave_attribution_graph.py:99` supports | **2b only.** The default carries its own note that circuit-tracer supports 2b with GemmaScope-2b transcoders. Not an ambiguity — a documented single-model dependency |
| 27b SAE availability at `cave_direction_sae_decomp.py:55` | **Blocked, and by absence.** `SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"` is hardwired 9b; no 27b release is named anywhere in the file. Whether a 27b SAE exists upstream is an external fact the code cannot settle, but the code is 9b-only regardless |
| the `--judges` choice set on `cave_judge_panel` | **There is no choice set.** `--judges` takes a free-form comma-separated list of HF model IDs, defaulting to Qwen2.5-7B-Instruct + Mistral-7B-Instruct-v0.3 + Llama-3.1-8B-Instruct. Any judge is expressible; the "unenumerated axis" is a feature, not a block |
| whether `job_distractor_task` `-it` cells are validly expressible | **Yes, and badly.** `:111` selects the `-it` weights, but the prompts stay QA-shaped (`STEM = "The capital of {r} is the city of"`). The cell runs; it asks an instruction-tuned model a base-format question. That is a validity question about the measurement, not a capability block — and it should be recorded as such rather than left open |

## 2. D14 — the pool sizes, settled

The reconciled ledger leaves the 66-item pool "unaccounted for by any pass" and the 891 with "no
committed producer". Both close by reading `cave_copy_confidence_conditional._build_pool` and counting
the input files.

| pool | composition | evidence |
|---|---|---|
| 16 | `rlhf_differential.ITEMS`, imported as `_BASE16` | `misconception_pool.py:19` |
| 61 | `ITEMS_WIDE = list(_BASE16) + EXTRA` | `misconception_pool.py:70`; stamped `pool_size: 61` by the instruments that use it directly |
| **66** | `_build_pool(big_pool=False)` = 61 + the **5** `factual` entries of `sycophancy_items.json` | counted: 5. 66 − 61 = 5 ✓. Stamped `pool_size: 66` |
| 817 | the TruthfulQA `generation` validation split | the residual, 891 − 74 |
| **891** | `_build_pool(big_pool=True)` = 66 + the **8** `factual` entries of `sycophancy_items_lowconf.json` + 817 | counted: 8. 61 + 5 + 8 = 74; 74 + 817 = 891 ✓ |

So the nesting is 16 ⊂ 61 ⊂ 66 ⊂ 891, and the ledger's "fifth pool" is the small pool. The
`{SYC5, LOW8}` axis named in ledger row R8 is exactly the two sycophancy files.

**The finding this produces is worse than the accounting gap it closes.** The 891 pool is **not
reproducible from the repo**: 817 of its 891 items are downloaded from HuggingFace at run time, and
when the download fails `_build_pool` prints `[pool] TruthfulQA unavailable; proceeding without it`
and **returns the 74-item pool without raising**. A re-run on a box without network access, or after
an upstream repo-id change, silently measures a 74-item substrate while every summary field and every
threshold assumes 891 — and the only trace is the `pool_size` stamp, which nothing checks. Every
`pool_size: 891` artifact in the repo (58 of them) is a claim about an external download that no
committed file pins.

**FIXED 2026-07-28** (`controls/cave_copy_confidence_conditional.py:328-333`): the silent-continue branch
now raises, naming the item count it would otherwise have measured and the 58 artifacts that stamp 891.
The `big_pool=False` path is untouched — 66 items is a legitimate, self-consistent pool. 29 call sites
were checked; none wraps `_build_pool` in a `try`, so the raise propagates rather than being swallowed.
**Not verified by execution:** this file imports torch at module level, so its `--selftest` cannot run on
a CPU-only workstation — syntax is verified by `py_compile` and the selftest is owed on the next box that
loads this lineage.

## 3. What this changes about the ledger's cost ordering

The ledger's headline for this class — "**K11 is 17 one-line clamps and blocks zero claims**" — holds,
and is now 16. Its cheap-mechanical bucket is smaller than advertised: **K4, K12, K13 and K14 are all
multi-line refactors of the same shape** (thread an `is_chat` through, add a conditional prompt
builder), so they are one job repeated 17 times rather than four flags. Batching them is the right
call; costing them as argparse is not.

Two items move *down* in cost, not up: K15 needed an investigation round and no longer does, and one
of K11's seventeen was never blocked.
