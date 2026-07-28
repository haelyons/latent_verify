# GAPS_RECONCILED — three blind passes, one ledger

Inputs: **A** = `GAPS_A_instruments.md` (130 instruments, 311 absent combinations, 68 code-blocked) ·
**B** = `GAPS_B_artifacts.md` (300 JSONs, 15 structural absence groups, 11 inconsistent artifacts) ·
**C** = `GAPS_C_claims.md` (163 claims: 68 PRESENT, 43 PARTIAL, 41 ABSENT).

The three were firewalled from each other and from all analysis prose. Agreement is evidence;
disagreement is a finding. Nothing else was read except six files cited by the passes, read only to
settle §3: `controls/family_cave_diagnose.py`, `controls/cave_fold_vs_listen.py`,
`results_fold_vs_listen{,_2b}/out/cave_fold_vs_listen.json`, the two
`results_9b_entropyneuron_powered*/out/entropy_neuron_9b_powered.json`, `misconception_pool.py`.

**54 distinct gaps.** Counts per class:

| class | gaps | claims blocked (of 163) |
|---|---|---|
| FREE / OFFLINE | 11 | 42 |
| CODE FIRST | 15 | 52 |
| GPU RUN | 14 | 30 |
| PROVENANCE / INSTRUMENT DEBT | 14 | 13 |

Claim counts are per-gap **blocking** counts, not a partition: a claim written at a breadth that is
wrong on two axes is blocked twice and appears in two rows. The four class totals therefore exceed 163
by design. Tie-break for MECE: a gap needing both an edit and a model pass is filed **once** under
CODE FIRST (a code change is a GPU run plus an edit plus a re-registration), with its box named in the
row. PROVENANCE holds only gaps that are not measurement gaps at all.

---

## 1. CONVERGENT — found independently by two or more passes

### 1.1 All three passes agree, plainly

Fourteen absences were reached by all three readings from three different directions. Where A supplies
a line, B supplies a cell count and C supplies a claim count, the gap is as well-established as
anything in this repo.

| # | absence | A framed it as | B framed it as | C framed it as |
|---|---|---|---|---|
| 1 | **The fold/listen mechanism phases have no base arm at any scale** | 4 code-blocks: `foldlisten_phase2.py:155`, `phase3a.py:317`, `phase3b.py:734`, `phase3c_riders.py:325` — `assert is_chat` | G2: "7 instruments × 3 scales = 21 absent cells. Every existing one is `-it`" | G5: "no challenge-masking run at 2b, 27b, or on any base model" ([93], [94]) |
| 2 | **Phase 2 (the attention-KO read gate) exists at 9b-it only** | run-only absences 181–182 (2bI, 27bI) | G3: 2 absent cells, **4 voided downstream verdicts** | [93]/[94] PARTIAL — "9b-it only", the sole masking artifact in the repo |
| 3 | **27b is the missing column of the whole mechanism + margin program** | 27b in most of the 247 run-only cells; 27b explicitly on `cave_fold_vs_listen`, all `cave_direction_*`, `truthful_flip`, `sycophancy`, `substrate_margin_grid`, `cave_residstate_*` | G9 (~36 instruments absent at 27b) + G11 (no 27b logprob-margin column) | G5: [116] "9b+" asserts 27b with no artifact; [25] no 27b; [97] the trend's unmeasured endpoint |
| 4 | **The doubt read/write circuit is characterised on base weights only** | L8: `-it` absent on doubt_route (5), contentgate (3), decollide (3), polarity_× (5 each), confidence_recruitment (5), circuit_patch (5) | G10: 27 M-pool artifacts base-only, one thin `-it` exception | [122]/[124] PARTIAL — a universal negative about `-chat` resting on one scale |
| 5 | **Distributional (teacher-forced / top-k / rank) readout on the verifier families exists at 9b only** | absences 155–180: `family_cave_diagnose`, `family_generate_judge`, `family_topk_shift`, `verify_graph_poc`, `modelw_candidates` at 2bB/2bI/27bB/27bI | G8: 4 instruments × {2b,27b} × {base,it} × {F22,F82} = **32 absent cells** | G1: 11 claims written at "Gemma 2" / "the model" and measured at 9b |
| 6 | **The bare turn (T0) top-k exists at 9b-base only** | absence 179–180 (2bB, 27bB) **plus** code-block 53 — `modelw_candidates.py:420-425` has no `--chat`, so all 3 `-it` cells are unreachable | G14: present only in `family_topk_shift` + `modelw_candidates` (9b-base) and `truthful_flip`; absent at 27b entirely, absent for `-it` | [129], [135], [152], [153] — W\* plausibility and rank, 9b-base only |
| 7 | **The neutral forced-final (T3n) control is missing outside ext2-82** | absences 292–297: `NEUTRAL_ELICIT` on VF22 at 5 cells, on EXT34 at both | G7: present F82×6 and F22×9b-it; **6 absent cells** | [163](5): "nothing about the n=22 base cells, which carry no neutral arm" |
| 8 | **The committed 27b decode is not reproducible, and the draft's 27b column mixes two runs** | lists all 6 `foldlisten_repro_diff` files: 2b/9b BYTE_IDENTICAL, 27b DIFF | I2: committed fold_rate 0.2115 → MOVEMENT_LISTEN_ONLY vs re-run 0.1373 → NO_MOVEMENT; both internally consistent; `frac_item_fields_identical` 0.804 | [35], [69], [73], [91], [157]: "the pushed column those numbers are compared against is this run's, which at 27b is not the column the committed figures print" |
| 9 | **The salience/copy lineage is 2b-base only — and its artifacts do not record which model they are** | 19 instruments code-blocked at `poc_minimal.py:51` + `worker.py:49` | I4: 15 `framing_*` artifacts undeterminable; ten are bare top-level JSON lists with no metadata wrapper | [116], [121]: the copy-circuit claims, written "base, scale unspecified" |
| 10 | **Free-reply labels are register-dependent and not scorer-stable** | `faithful_rescore` exists and ran (9 files, 0 absences) | I11: `change_frac` up to **0.841** on `neutral_gen`, 0.318–0.750 on `counter_gen`, all MATERIALLY_RELABELED; every `elicit_gen` STABLE (≤0.114) — plus I9/I10, gates that flip with the register | [89] PRESENT: "any count taken off a free reply has to say which one it came from"; [80] recorded both ways (0/0/1 lenient vs 62 strict exceptions) |
| 11 | **The external judge panel is 9b-only** | `cave_judge_panel` absent for 2b and 27b generations; the `--judges` axis has no enumerated choice set | G15: 9b only (n=40, n=47); `gold_agreements` is `{}` in the n=47 artifact | [36](a) ABSENT — no persisted judge output of the named failure mode exists in either panel |
| 12 | **Substrate identity is unstamped and silently swapped under the claims** | family table: MISC61 = 16 committed + 45 EXTRA; TQA separate; **891 appears in no producer** | I6: five sizes (16/61/66/891 + n_pool 817), "no artifact states the nesting" | G6: "the mechanism substrate is the 891-item pool or a 5-pair anchor probe, while the sentences say *in our experiments*" |
| 13 | **Human-label validation covers 3 of 12 ext2 cells** | `classify_vs_handlabel` never run on the 2b/27b hand-label sets that exist on disk | G13: no human label for 9b×F82 (the headline cell), none for any base F82 cell, none for T3n, none for any listen cell | [36](b) PARTIAL, [163](2): "a subset from each run" is not met |
| 14 | **No distribution is read at the forced-final slot, by any instrument, anywhere** | §2.5: "no instrument outside the fold/listen lineage reads a T3 slot at all… every `cave_*` control reads T2 only" — stated as a repo-wide capability gap, deliberately not counted | dimension table builds the grid with "slot and quantity fixed by turn", i.e. the same fact as an axiom | G3: [82], [100], [101], [109] — "No artifact anywhere holds a C-vs-W\* margin at the forced-final slot" |

Row 14 is the sharpest case of the three passes meeting: A found it in the code and refused to count it,
B encoded it as a property of its own grid, and C found it as a caption on a published figure.

### 1.2 Two passes, with the third not positioned

| absence | passes | why the third could not see it |
|---|---|---|
| **The listen direction has no distributional readout anywhere.** B: G1, "every teacher-forced, top-k and rank instrument builds only `push(q, C, …)`". C: [137] "ABSENT entirely… the single widest readout gap"; plus the 8 direction-unstated rows of G2 | B + C | A projects each instrument onto only the axes it varies. The distributional controls have **no arm axis at all**, so a missing arm generated no row. A's silence is structural corroboration, not disagreement — confirmed at `controls/family_cave_diagnose.py:214-215` in §3.1 |
| **Activations and captures are discarded.** A: provenance note, `.gitignore:21`. B: G5 — four `.npz` named by 5 result JSONs are absent; every phase-3c / phase-4 / think-probe number is unreproducible | A + B | No claim in C's 163 cites a probe AUROC or a crossing class, so the write-up never depends on those captures. That is itself the finding: the largest instrument debt in the repo blocks **zero** claims |
| **No hardware, driver or library version is recorded anywhere.** C: [72] PRESENT/confirmed by grep over 306 artifacts. B: I4's sibling finding — 60 of 300 carry no model string, 17 undeterminable | B + C | A reads code, which cannot show what a run failed to stamp |
| **EXT34 (F34) is measured at 9b-it only — 5 absent cells.** A: absences 237–241. B: G6 | A + B | Not positioned: no claim in the write-up is written at F34 breadth. Priority follows from that, not from the count |
| **No gate has ever been computed on a base-model judge summary.** B: G12, all 27 gate artifacts are `-it`, 9 base cells reported but never gated | B (+A's adjacent absence 298–305) | C's claims quote the base cells' counts directly and never cite a gate verdict on them, so the ungated status is invisible from the prose |

---

## 2. SINGLE-SOURCE — one pass only, with a positioning judgement

**Absence of corroboration is not weak evidence for any row in this section.** In every case the other
two passes were structurally incapable of seeing the item. An instrument-level code block leaves no
artifact and no claim; an instrument that never ran leaves no artifact at all; a measurement nobody ever
implemented leaves neither code nor artifact.

### 2.1 A-only

| gap | why B and C were blind to it |
|---|---|
| **68 code-blocked axes** with their blocking lines (A §3) | A line of code is invisible to an artifact scan and to a claim audit. Both other passes see the *absence*; only A can say whether it is one flag or a design change — which is the whole cost question |
| **5 instruments with zero artifacts of any kind**: `cave_doubt_writes_cavedir`, `numeric_repair_controlled_nec`, `ov_qk_generality_probe`, `perhead_nec_null`, `salience_generality_arm` | An instrument that never ran leaves no trace for B by definition. Worth flagging beyond A's own note: **`perhead_nec_null` is the NULL baseline for the per-head necessity metric that claim [121] states at "proves" strength**, and `numeric_repair_controlled_nec` is its repair-controlled twin. C could not connect them because C never read code |
| **43 item-family absences** (A §2.2) — instruments that accept a family and were never run on it | B's matrix has no row for a family×instrument pair nobody produced; C has no claim at those families |
| **The T3 capability gap as a *cause*** (A §2.5) | C found the symptom (§1.1 row 14); only A can say the readout does not exist in any instrument |
| **4 capability ambiguities A refused to resolve**: the transcoder-supported set for `cave_attribution_graph.py:99`; 27b SAE availability at `cave_direction_sae_decomp.py:55`; the `--judges` axis; whether `job_distractor_task` `-it` cells are validly expressible | Unknowable from artifacts (absence looks the same either way) and from claims |
| **`logit_lens_attribution --crosslens` is executed-but-SKIPPED**, not absent; **`cave_reader_pathpatch --layer 32`** never run | Distinguishing "ran and declined" from "never ran" needs the code's own status contract |

### 2.2 B-only

| gap | why A and C were blind to it |
|---|---|
| **I1** `results_ablate_mlp/out/cave_ablate_late_mlp.json` is a truncated, unparseable write superseded in place | A counted 4 files in that directory and could not know one does not parse; no claim cites it |
| **I3 / I7** duplicate filenames with identical `tag` across run directories (entropy_neuron ×2, matched_item_deconfound ×2) | Requires opening both copies and comparing values |
| **I5** the `out/verify_graph_poc_*.json` copies record a Windows scratch path as `family_arg` and carry `pre_only=true` / `t3=null` | Invisible to a code read; the complete twins exist so nothing looks missing |
| **I8** phase-4 and phase-3c return **opposite validity verdicts on the same 9b-it captures** (PROBE_VALID_FOR_PUSHBACK vs PROBE_INVALID) | Two offline instruments over one input; only a reader of both outputs sees the split |
| **G15 tail** `gold_agreements` is `{}` in the n=47 judge-panel artifact | An empty dict is a run outcome, not a code property |
| **G12** the base cells are reported but never gated | See §1.2 |

### 2.3 C-only

| gap | claims | why A and B were blind to it |
|---|---|---|
| **Span-level content of any generation** — what a withheld, hedged or entrenched reply actually says | 22 | No instrument among A's 130 produces a span taxonomy, so A's "capable of" column has no row for it; a measurement never made leaves no artifact for B. B's nearest approach is I11 (the free-reply label is not scorer-stable), which is the *reason* the span read matters |
| **Item-level joins** — any claim pairing two cells, two slots or two readouts on the same items | 9 | A's grids have no join axis; B's coverage matrix is per-cell and a join is not a cell |
| **The "mentions the pushed entity anywhere" register** [146] | 1 | Described in prose, computed nowhere. The stored lenient register gives gaps of 6/15/25 items where the sentence claims at most one — a distinct register, not a re-reading |
| **A significance statement with no test** [13] | 1 | The rates are stored; nothing computes a statistic on a fold-rate difference |
| **Pre-registered thresholds live only in design documents** — the 5-to-25 listen band [160], the frozen attribution rule [54] | 2 | Both other passes read only code and artifacts, which is exactly where the registration is not |
| **The 24-token elicitation budget is not stamped in any summary** [154]; **no item-selection provenance for the 82 pairs** [151] | 2 | Requires knowing what the register *claims* to be |

---

## 3. DIVERGENT — where two passes disagree, and how each resolves

Fourteen divergences. Twelve are settled here; two remain open. In every settled case the pattern the
task anticipated held: **the artifact exists and does not measure what someone assumed.**

### 3.1 SETTLED BY A FILE READ

**D1 — `cave_fold_vs_listen`: what does it actually deliver?** *(the most consequential divergence in the set)*

- **A:** a full READ / WRITE / RANDOM battery over FOLD / LISTEN / AGAINST_GRAIN cells, `-it` positive
  control, all-attention and all-MLP brackets, matched-move gate, cross-cell axis transfer; 3 pairs
  capable, 2 run (9b, 2b), 27b absent.
- **B:** G1 — "Listen exists solely as a text label plus one AUROC (`cave_fold_vs_listen`)."
- **C:** [25] — base top-5 overlap 4, `-it` overlap 5, i.e. the artifact reads **opposite** to the claim;
  [120] — `move_gate.passed` false at base in both runs.

**Settled** by reading `results_fold_vs_listen/out/cave_fold_vs_listen.json` and
`results_fold_vs_listen_2b/out/cave_fold_vs_listen.json`. A describes the code correctly; B describes
the usable content correctly; C is right and **understates it**:

| cell | n_fold / n_listen | ncav_fold / ncav_listen | FOLD battery | LISTEN battery | axis_auroc_listen | move_gate | decision |
|---|---|---|---|---|---|---|---|
| 9b base | 30 / 16 | 8 / 8 | present | present | 0.82 | **false** | MOVE_UNMATCHED |
| 9b it | 30 / 16 | 13 / 14 | present | **null** | **null** | **false** | MOVE_UNMATCHED |
| 2b base | 22 / 24 | **2** / 8 | **null** | present | null | **false** | MOVE_UNMATCHED |
| 2b it | 22 / 24 | 9 / 21 | present | **null** | **null** | **false** | MOVE_UNMATCHED |

`pool_size` 891 and `big_pool` true in both files; `pos_control_restore` null in all four cells;
`bracketed` true only at 2b-base. The 2b-base FOLD null is explained by the artifact's own threshold
(`MIN_FAITHFUL: 8` against `ncav_fold: 2`).

**Resolution: 9b-base is the only cell in the repo where the FOLD and LISTEN batteries both exist, and
its own move gate fails.** The base-vs-`-it` fold/listen contrast that claim [25] asserts, and the
single-mechanism reading of [120], have never been computed at any scale. Running 27b (ledger R2) will
not fix this — the void is faithful-item count per cell, which needs the substrate change of K15.

**D2 — is the missing listen distributional readout code-blocked or merely unrun?** B (G1) calls it
structural; A counts no absence at all. **Settled** at `controls/family_cave_diagnose.py:214-215`:

```
214:        neutral = push(q, C, NEUTRAL)
215:        counter = push(q, C, PUSH["counter"].format(W=W))
```

The plant is the literal `C` in **both** arms, and the same construction is echoed in the instrument's
own `decision_rule` string at `:79` and `:287`. B is exactly right; A's silence is a consequence of its
counting rule (the arm axis does not exist in the instrument, so a missing arm produces no row).
**Class: CODE FIRST (K1), not GPU RUN** — and it is a design change, not a clamp, because the metric
`M = lp(C) − lp(W*)` and the headroom gate both assume C is the plant.

**D3 — `entropy_neuron_9b_powered.json` exists twice (B's I3): "nothing inside either file identifies
which run it is."** **Settled against B.** Both files carry `tag: "9b_powered"` and
`reference_mode: "long"`, but `reference_used` differs:
`results_9b_entropyneuron_powered/` → `{"base": "short_fallback", "it": "short_fallback"}`
(baseline_entropy 3.0115 / 2.2379); `results_9b_entropyneuron_powered2/` →
`{"base": "long_wikitext_20x256", "it": "long_wikitext_20x256"}` (2.182 / 1.9118). **A's reading is
correct.** Residual debt, worse than B's version of it: the first run *requested* `long` and silently
fell back to `short`, and `reference_mode` still says `long`. Both report
`entropy_neuron_count: 0`, so no verdict moves either way. → P10.

### 3.2 SETTLED FROM THE PASSES THEMSELVES, NO READ NEEDED

**D4 — B's G2 is three cost classes, not one.** B counts "7 instruments × 3 scales = 21 absent base
cells" as one group. Using A §3: `phase2` / `phase3a` / `phase3b` / `phase3c_riders` are
`assert is_chat`-blocked (**12 cells → CODE FIRST**, K5); `think_probe_identity` appears nowhere in A's
68, so it is **run-only (3 cells → GPU RUN**, R6/R14); `phase3c_analysis` and `phase4` are offline
capture-consumers (**6 cells → blocked by the discarded `.npz`**, P2). One of B's groups, three
different pieces of work.

**D5 — B's G3 void vs A's absent cell.** B reports 2b-it and 27b-it artifacts that "self-report the
hole" (`p2_committed` null, `a6_decision` INSUFFICIENT); A reports the phase-2 cells as simply absent.
Both are right: no phase-2 artifact exists at those cells (A), and the void fields sit in the
**downstream** phase-3c artifacts at those scales (B — `a6_decision` is a phase-3c-riders field).
Consequence for the ledger: running phase 2 at 2b-it and 27b-it does not add 2 cells, it **revalidates
4 already-committed downstream verdicts**. That is an amortisation argument for the 2b and 27b boxes.

**D6 — A's 6 "faithful-strict block absent" cells (absences 298–303) vs A's own 7 `faithful_rescore`
files.** Settled against A: the strict labels for VF22 at all six cells exist in
`out/faithful_rescore_fl_{2bbase,2bit,9bbase,9bit,27bbase,27bit}.json`; what is missing is the
in-summary `cells_faithful` block. C independently uses those legacy-22 strict counts at [74]. **Not a
measurement gap — an offline merge** (F9).

**D7 — is `cave_fold_vs_listen` at 27b code-blocked?** B (G9) and C ([25]) both want it and neither
could tell. A settles it: `controls/cave_fold_vs_listen.py:80 READ_LAYER = 28` is superseded at run
time by `pick_read_layer(n_layers)` at `:91`. The artifacts confirm (`read_layer` 28 at 9b, 17 at 2b).
**GPU RUN, no code change** (R2).

**D8 — `modelw_candidates` at `-it`: is it 2 of B's 32 cells, or a code block?** A settles it:
`controls/modelw_candidates.py:420-425` has no `--chat`. B's G8/G14 cell counts include cells that need
an edit first. → folded into K4.

**D9 — `matched_item_deconfound_9b.json` "exists twice" (B's I7).** Settled by A: the grid is
"3 pairs × {narrow 16, wide 61}", so the duplication is the two values of the pool axis, produced by
design in `results_9b_matched/` and `results_9b_matched_wide/`. Debt reduces to the narrow file
omitting `pool_size` → P10.

**D10 — `clean_entity.json` "is not in the tree" (B's I5).** Settled by A: CLEAN38 is generated by
`controls/clean_entity_pool.ENTITY_ITEMS`, not committed as JSON. B's finding is about a dumped input
path, not a missing family → P11.

**D11 — `cave_ablate_late_mlp` coverage.** A counts the cell covered (4 files); B says the primary write
is truncated and unauditable. Both hold: coverage survives via `_repaired.json` / `_mean.json`; the
original write is debt (P9). Separately, A's item 33 marks 2b blocked by `DEFAULT_LAYERS = [23,27,29,30]`
at `:52` — overridable by `--layers`, so a default, not a wall.

**D12 — `verify_graph_poc` coverage.** A counts T_PRE and T3 covered; B finds the `out/` copies
`pre_only=true`, `t3=null`. The `results_verifier/` twins are complete and the `t_pre` blocks are
byte-identical, so 9b-base coverage stands; the debt is the foreign scratch path (P11).

**D13 — Figure 2's slot ([82]).** A and C converge without contact: A's readout vocabulary separates
`RA`/`RC` (answer slot, immediately after the user's turn) from `ELI` (the elicited slot) and lists
`family_cave_diagnose` as RA/RC only; C reports the figure builder's own header saying the same. The
published margin figure is captioned one slot later than the quantity it draws.

### 3.3 OPEN

**D14 — the pool sizes 66 and 891.** Partly settled: `misconception_pool.py:70`
`ITEMS_WIDE = list(_BASE16) + EXTRA` gives 16 ⊂ 61 (A and B agree), and 817 is `truthful_flip`'s
TruthfulQA slice (A and B agree). But the **891 is a fifth pool that appears in none of A's four item
family producers**: `controls/cave_fold_vs_listen.py:611` calls
`cave_copy_confidence_conditional._build_pool(big_pool=True)`, and the artifacts stamp
`pool_size: 891`. **66 is unaccounted for by any pass.** Settling file: the `_build_pool` body in
`controls/cave_copy_confidence_conditional.py`. This matters because C's G6 makes the substrate
load-bearing: every mechanism number in the write-up is quoted beside figures built on ext2-82.

**D15 — claim [93]'s printed "67 of 74" vs the artifact's `arm_counts.fold_mask` (moved 3 / held 70 /
abstain 1 = 73 of 74).** C flags the mismatch and states it needs a per-item read of the 370-record
`items[]`. One artifact, one offline read → F11.

---

## 4. THE LEDGER

Ordered within each class by claims blocked. † marks a claim count shared with another row (a cheaper
alternative route to the same claims, or the same claim blocked on a second axis).

### 4.1 FREE / OFFLINE — derivable from committed artifacts, no model run

| id | gap | claims | inputs already on disk | found by |
|---|---|---|---|---|
| **F1** | **Span-level taxonomy of generated replies** — what a withheld, hedged, entrenched or off-target reply actually says, in all three slots | **22** | `neutral_gen`, `counter_gen`, `elicit_gen`, `neutral_elicit_gen` in the 6 ext2 + 6 nelicit summaries; 295 neutral withholds, 234 pushed-elicited, 231 free-reply spans | C only (§2.3) |
| **F2** | **Item-level joins** — any claim pairing two cells, two slots or two readouts on the same items ([78] carry-through, [83] withheld × margin sign, [86] the 46/36 disagreement, [96] the 21-to-4 paired selectivity, [102], [123], [142]–[144]) | **9** | `items[]` in each cell's summary; both sides of every join are committed | C only |
| **F3** | **Elicitation-prompt contamination census** ([65]–[68]) | 4 | `items[].elicit_prompt` / `neutral_elicit_prompt` — the runaway Q/A ladder is visible in every base context | C only |
| **F4** | **Hand-labels for the uncovered cells**: 9b×ext2, any base ext2 cell, any listen cell, the T3n slot | 2 | committed generations; human reading, no model | B G13 + C [36]/[163](2) |
| **F5** | **The "mentions the pushed entity anywhere" register** [146] | 1 | committed generations; the stored lenient register is not this register (gaps 6/15/25 vs the claimed ≤1) | C only |
| **F6** | **A significance test on any fold-rate difference** [13] | 1 | the six ext2 summaries' `cells_faithful` | C only |
| **F7** | **The neutral-slot W\* median rank (119)** — derivable from `rank_w_neutral`, recorded in no aggregate [152] | 1 | `family_topk_shift_vfam_ext2_9bbase.json` | C only |
| **F8** | **Reconcile [93]'s 67-of-74 against the arm block's 73-of-74** | 1 | `foldlisten_phase2_p2_9bit_summary.json` `items[]` (370 records) | C only (D15) |
| **F9** | **`classify_vs_handlabel` never run on the 2b/27b hand-label sets that exist** | 1 | `results_foldlisten_{2b,27b}/handlabel_spotcheck_fl_{2b,27b}.json` | A (+B G13, C [36]) |
| **F10** | **No gate ever computed on a base judge summary** — 9 base cells reported, never gated | 0 | 3 scales × {F22,F82} base summaries; all 27 gate artifacts are `-it` | B G12 |
| **F11** | **Merge the existing strict labels into the VF22/EXT34 summaries** (the `cells_faithful` block is missing; the labels exist) | 0 | `out/faithful_rescore_fl_*.json` (7 files) | A, resolved in D6 |

**Class total: 11 gaps, 42 claims. None needs a GPU.** This is the largest defensibility win per unit
of work in the whole ledger.

### 4.2 CODE FIRST — blocked by a line of code

| id | gap | blocking line(s) | one-line clamp or design change | claims |
|---|---|---|---|---|
| **K1** | **Every C-vs-W\* distributional prompt plants C → no distributional readout in the listen direction, at any scale or variant** | `controls/family_cave_diagnose.py:214-215` (verified, D2); same construction in `family_topk_shift`, `modelw_candidates`, `gen_outputs_table` | **Design change** — `M = lp(C) − lp(W*)` and the headroom gate both assume C is the plant; adding a plant-W\* arm inverts the metric's sign convention | **12** |
| **K2** | **No instrument reads a distribution or a residual state at the forced-final (T3) slot** | A §2.5: T3 exists only in `foldlisten_judge:66`, `phase2:59`, `phase3a:71`, `phase3c_riders:72`; every `cave_*` reads T2 | **Design change** — a new readout on the existing controls | 6 |
| **K3** | **`gen_outputs_table` cannot take a family** — the one instrument whose model axis is *already* all six cells is hardwired to 4 items | `gen_outputs_table.py:21 ITEMS` (against `:42 CELLS`, complete) | **One line + argparse.** The cheapest cross-scale probability artifact in the repo | 7† |
| **K4** | **No `--chat` on 14 instruments** → every `-it` cell of the L2/L3/numeric/candidate lineages, including the bare-turn top-k | A items 53–66: `modelw_candidates.py:420-425`, `job_copyscore.py:212-215`, `job_localize208.py:232-234`, `job_recurrence.py:232-234`, `job_refine_heads.py:205-212`, `job_forcedchoice.py:43-44`, `job_numeric_mechanism.py:44`, `instr_triangulation.py:473-481`, `scale9b_dose_response.py:180-182`, `scale9b_numeric_copy.py:195-198`, `scale9b_numeric_generality.py:240-244`, `numeric_repair_controlled_nec.py:331-335`, `perhead_nec_null.py:246-251`, `salience_generality_arm.py:352-356` | Argparse + template branch, ×14 — mechanical but repeated | 6 |
| **K5** | **The mechanism substrate is the 891 big-pool or a 5-pair anchor probe; it has never been the ext2-82 family the figures are built on** | `controls/cave_fold_vs_listen.py:611` + argparse `:842` (`--big-pool` store_true only); same `_build_pool` import across the `cave_*` lineage | **One line + argparse per control**, but it invalidates every stored threshold calibrated on 891 | 7 |
| **K6** | **`assert is_chat` blocks all base cells of phases 2 / 3a / 3b / 3c** — 12 cells | `foldlisten_phase2.py:155`, `phase3a.py:317`, `phase3b.py:734`, `phase3c_riders.py:325` | **Design decision**, not a bug: the phases are registered on the `-it` substrate (C5). Retiring the assert means re-registering the arm | 2 |
| **K7** | **9b head coordinates hardwired → the entire headset lineage cannot leave 9b** | `atp_low_confirm.py:32-34` (`HEADS`, `NH_9B = 16`), imported at `headset_direction.py:49` and `headset_joint_patch.py:60`, consumed by `matched_item_deconfound.py` | **Design change** — needs a per-scale head-discovery step ahead of the patch. Blocks the two **bolded standalone conclusions** [27] and [122] | 4 |
| **K8** | **A decontaminated elicitation arm cannot be built** — the truncation rule lives in the scorer, not the prompt builder | `scorer_provenance` (post-hoc) vs the elicit-prompt construction | **Design change**; C [148] specifies it, C [163](1) records it as never run | 4 |
| **K9** | **The L1 salience/transcoder lineage is pinned to 2b-base** — 19 instruments | `poc_minimal.py:51 MODEL_NAME` + `worker.py:49 exec`; `base_attn_qa.py:21-22`; `job_numeric_localize.py:26` | **Design change** — worker architecture plus a GemmaScope transcoder dependency | 3 |
| **K10** | **`COPY_HEAD = (18,5)`** (a 2b coordinate) blocks 9b and 27b on the copy×confidence conditional | `controls/cave_copy_confidence_conditional.py:93` | One line **plus** a per-scale copy head (so it inherits K7's discovery step) | 1 |
| **K11** | **9b layer indices hardwired → 2b out of range on 17 cave-direction / residstate controls** | `cave_direction_heldout.py:55 FIT_LAYERS`, `cave_residstate_diff.py:43 READ_LAYER=28`, `cave_residstate_close.py:34-35`, `cave_causal_localize.py:36`, `cave_direction_dla.py:71`, `cave_direction_sae_decomp.py:55`, + 11 more (A items 30–46) | **One-line clamp each.** The pattern already exists: `cave_residstate_anyscale.py:11`, `AXIS_LAYER := round(0.667·n_layers)` | **0** |
| **K12** | **`realized_attention.py:37 HEADS`** holds 27b coordinates and argparse `:144-148` has no `--heads` | that line | Argparse | 0 |
| **K13** | **`scale9b_doubt_direction` applies the chat template unconditionally** → 3 base cells | `scale9b_doubt_direction.py:58-59` | One-line branch | 0 |
| **K14** | **`ov_behavioral_scale` loads no base model** → 3 base cells | argparse `ov_behavioral_scale.py:153-156` (`--name-it` only) | Argparse | 0 |
| **K15** | **4 capability ambiguities** — supported-model set undeterminable from the code | `cave_attribution_graph.py:99`; `cave_direction_sae_decomp.py:55` (27b SAE); `cave_judge_panel --judges`; `job_distractor_task.py:111` | Investigate, then classify. Cannot be costed until then | 0 |

**Class total: 15 gaps, 52 claims.** Note K11 pointedly: **A's single largest code-block cluster — 17
instruments, 17 one-line clamps, the cheapest mechanical fix in the repo — blocks zero of the 163
claims.** The cave-direction lineage barely appears in the write-up. Conversely K1, K2 and K5, which
together block 25 claims, are three axes that pass A **explicitly declined to count** as combinations.
A's 311 is not the denominator for defensibility.

### 4.3 GPU RUN — needs a model pass, grouped by box

Box grouping: a box holds one model's weights for base and `-it`, so every row sharing a box amortises.

| id | gap | box | claims |
|---|---|---|---|
| **R1** | **ext2-82 distributional readout outside 9b** — `family_cave_diagnose`, `family_topk_shift`, `family_generate_judge`, `verify_graph_poc` at 2bB / 2bI / 27bB / 27bI (B's G8, 32 cells; A's 155–178) | **2b + 27b** | **11** |
| **R2** | **9b-it distributional holes** — `family_generate_judge`, `family_topk_shift`, `verify_graph_poc`, `think_probe` at 9bI (`modelw_candidates` needs K4 first) | 9b | 4 |
| **R3** | **27b mechanism column** — `cave_fold_vs_listen` 27b (run-only, D7), `cave_residstate_anyscale` / `_decisive` 27b, `cave_faithful_it_{diff,mc}` 27b, `faithful_copy_wstar` 27b, `cave_prompt_feature_mechanism` 27b, `cave_headset_specificity` 27b×copy | 27b | 3 |
| **R4** | **Copy-circuit localisation at 27b** — `copyscore`, `localize208`, `recurrence`, `refine_heads`, `scale_mechanism` at 27bB | 27b | 2 |
| **R5** | **Phase 2 at 2b-it and 27b-it** — adds 2 cells and **revalidates 4 voided downstream verdicts** (D5) | 2b + 27b | 2† |
| **R6** | **2b riders** — `copyscore --sweep` at 2b, `gate_dont_delete --select induction`, `think_probe` 2b, `cave_multisample_caverate` 2b | 2b | 2 |
| **R7** | **The 5 zero-artifact instruments** — `cave_doubt_writes_cavedir` (3 pairs), `numeric_repair_controlled_nec` (3 base), `ov_qk_generality_probe` (3 pairs), `perhead_nec_null` (3 base), `salience_generality_arm` (3 base) | all three | 1 (load-bearing: `perhead_nec_null` is the NULL baseline for the necessity metric [121] states at "proves" strength) |
| **R8** | **27b behaviour / margin column** — `truthful_flip` 27bB+27bI, `sycophancy` 27b × {SYC5,LOW8}, `substrate_margin_grid` 27b, `numeric_boundary` 27b | 27b | 1 |
| **R9** | **T3n / NEUTRAL_ELICIT on VF22 (5 cells) + EXT34** (B's G7, A's 292–297) | all three | 1 |
| **R10** | **Judge panel on 2b and 27b generations** (loads the external Qwen judge) | 2b + 27b + judge | 1 |
| **R11** | **Bit-width / seed sweep at 27b** — the only measurement that would support [71]'s "numerical perturbation rather than a change of logic" | 27b | 1 |
| **R12** | **9b mechanism holes** — `cave_attribution_graph` 9bB, `gate_dont_delete` 9b, `forcedchoice` 9bB, `numeric_mechanism` 9bB, `distractor_task` 9bB+9bI, `instr_triangulation` 9bB × 2 pair sets | 9b | 1 |
| **R13** | **EXT34 at the other 5 cells; MECH74 and COMB138 in `foldlisten_judge`** (B's G6; A's 237–247, 288) | all three | 0 |
| **R14** | **Single-flag riders on an already-loaded box** — `cave_ablate_late_mlp --mode resample`, `cave_copy_confidence_conditional --conf-var {entropy,margin}`, `cave_reader_pathpatch --layer 32`, `logit_lens_attribution --crosslens` (executed-but-SKIPPED, not absent), the three never-supplied `--items` overrides | any | 0 |

**Class total: 14 gaps, 30 claims. Three boxes carry all of it:** the **27b box** (R1, R3, R4, R5, R8,
R10, R11 — 20 claims), the **2b box** (R1, R5, R6, R10 — 13 claims, mostly shared with 27b), the **9b
box** (R2, R12 — 5 claims). R7, R9, R13, R14 ride whichever box is already loaded.

### 4.4 PROVENANCE / INSTRUMENT DEBT — recording gaps, not measurement gaps

| id | gap | claims | found by |
|---|---|---|---|
| **P1** | **The committed 27b decode is not reproducible, and the draft's 27b column mixes the two runs** — committed 0.2115 → MOVEMENT_LISTEN_ONLY vs re-run 0.1373 → NO_MOVEMENT, both internally consistent; withheld 32/28 published against 34/35 re-run | **5** | A + B (I2) + C ([35][69][73][91][157]) |
| **P2** | **No hardware, driver or library field in any of 306 artifacts** — the committed 27b box is unrecoverable, so [69] and [70] can never be made defensible, only retracted | 3 | C [72] + B I4 sibling |
| **P3** | **Pre-registered thresholds live only in design documents** — the 5-to-25 listen band [160], the frozen attribution rule [54]. The thresholds are stamped; the registration is not | 2 | C only |
| **P4** | **The register/gate contest is not flagged by any field** — gate v1 FAIL / v2 PASS on 2b-it F22; commit FAIL / faithful PASS inside one 27b-it file | 1 | B I9/I10 + C [158] |
| **P5** | **The 24-token elicitation budget is stamped in no summary** — the one element of the stated register that is not | 1 | C [154] |
| **P6** | **No item-selection provenance for the 82 pairs** — the family's construction is unauditable except by reading the JSON | 1 | C [151] |
| **P7** | **Activations and captures discarded** — `.gitignore:21-22`; four `.npz` named by 5 result JSONs absent; every phase-3c / phase-4 / think-probe number unreproducible; **this, not code and not GPU, is what blocks `phase3c_analysis` at 2b-it and 27b-it** (D4) | **0** | A + B (G5) |
| **P8** | **No artifact states the pool nesting** (16 / 61 / 66 / 817 / 891) and the 891 belongs to no committed producer (D14) | 0 | B I6 + C G6 + A's family table |
| **P9** | **60 of 300 artifacts carry no model string; 17 undeterminable; 10 `framing_*` are bare top-level lists with no metadata wrapper** — the L1 lineage's coverage is verifiable only from code | 0 | B I4 (+A supplies the answer) |
| **P10** | **Duplicate filenames with identical `tag` across run directories** — entropy_neuron ×2 (distinguished only by `reference_used`, and one silently fell back from its requested `long` ref, D3); matched_item_deconfound ×2 (the narrow file omits `pool_size`, D9) | 0 | B I3/I7, both resolved here |
| **P11** | **`verify_graph_poc` `out/` copies carry a foreign Windows scratch path as `family_arg` and name an input absent from the tree** | 0 | B I5 |
| **P12** | **`cave_ablate_late_mlp.json` is a truncated unparseable write, superseded in place** rather than replaced | 0 | B I1 |
| **P13** | **Phase-4 and phase-3c return opposite validity verdicts on the same 9b-it captures**, and the surviving verdict's captures are gone (chains to P7) | 0 | B I8 |
| **P14** | **`gold_agreements` is `{}` in the n=47 judge-panel artifact** | 0 | B G15 |

**Class total: 14 gaps, 13 claims.** P7 deserves its own sentence: it is the single largest instrument
debt in the repo and it blocks **zero** claims, because no claim in the write-up cites a probe AUROC or
a crossing class. P1 and P2 are the reverse — small recording gaps that carry 8 claims between them,
two of which ([69], [70]) are permanently unfixable.

---

## 5. THE MINIMAL SET

### 5.1 To make the write-up's existing claims defensible at the breadth they are written

Ordered by claims closed per unit of work. **No narrowing of any sentence is assumed** — this is what
the prose as written requires.

| step | work | class | claims closed | cumulative |
|---|---|---|---|---|
| 1 | **One offline span-classification pass** over the committed generations, persisting a taxonomy per slot per cell (F1) plus the contamination census (F3) | FREE | 26 | 26 |
| 2 | **One offline join file** over `items[]` — carry-through, withheld × margin sign, the two-layer disagreement, the paired selectivity, the base-withhold × `-it`-fold association (F2) | FREE | 9 | 35 |
| 3 | **Four small offline computations**: the mention-anywhere register (F5), a significance test on the fold-rate differences (F6), the neutral-slot W\* median rank (F7), the [93] arm-block reconciliation (F8) | FREE | 4 | 39 |
| 4 | **`family_cave_diagnose` + `family_topk_shift` on ext2-82 at 2bB / 2bI / 27bB / 27bI** (R1) — two boxes, four cells; closes the scale axis of the entire probability story | GPU (2b, 27b) | 11 | 50 |
| 5 | **Add a plant-W\* arm to the distributional controls** (K1, `family_cave_diagnose.py:214-215`), then re-run it in the same two boxes as step 4 | CODE + GPU | 12 | 62 |
| 6 | **Run the fold/listen mechanism on ext2-82** (K5) — the prose says "in our experiments"; the artifacts say `pool_size: 891` | CODE + GPU | 7 | 69 |
| 7 | **A forced-final (T3) distributional readout** (K2) — or accept that Figure 2, [100], [101] and [109] are captioned one slot from what they measure | CODE + GPU | 6 | 75 |
| 8 | **Per-scale head discovery for the headset lineage** (K7, `atp_low_confirm.py:32-34`) and the 27b mechanism + copy-localisation runs (R3, R4) | CODE + GPU (27b) | 9 | 84 |
| 9 | **The decontaminated elicitation arm** (K8) — without it the neutral-arm withholding result stands on a confound the notes themselves identify | CODE + GPU | 4 | 88 |
| 10 | **Hand-labels for the uncovered cells** (F4) and `classify_vs_handlabel` on the two existing sets (F9) | FREE (human) | 3 | 91 |
| 11 | **Decide which 27b run is canonical, then stamp hardware, thresholds, the token budget and item provenance going forward** (P1, P3, P5, P6) | PROVENANCE | 9 | 100 |

**Stop there.** 11 steps, of which **5 are offline and close 42 of the 163 claims with no GPU at all**.
Steps 4–9 need exactly **two boxes** (2b and 27b) plus one 9b re-run, and every GPU row in the minimal
set shares one of those boxes.

**Three claims cannot be made defensible by any work:**

- **[69], [70]** — "on two different GPU types", "the split is by model size and not by machine". The
  hardware record does not exist and cannot be reconstructed (P2). These must be retracted, not fixed.
- **[1], [29]** — stated about language models in general. Every artifact in the repo is Gemma 2. Either
  cite external work or narrow the sentence; a second model family is a different program.

### 5.2 What collective exhaustiveness would additionally cost — a larger and different list

The minimal set touches roughly **20 of A's 311 absent combinations** and leaves the following
untouched. None of it is required by any current claim.

| work | scope | claims |
|---|---|---|
| **K11** — 17 one-line layer clamps so the cave-direction lineage runs at 2b | 17 instruments; the cheapest mechanical fix in the repo | 0 |
| **K4** — `--chat` on 14 instruments, then the runs | every `-it` cell of the L2/L3/numeric lineages | 6 (already in the minimal set via step 4's dependencies) |
| **K6** — retire `assert is_chat` and re-register the phases on base | 12 cells, plus a design re-registration | 2 |
| **K9** — lift the L1 salience lineage off 2b-base | 19 instruments; worker architecture + transcoder dependency | 3 |
| **K12, K13, K14, K10** — four small argparse/coordinate fixes | 4 instruments | 1 |
| **K15** — resolve the 4 capability ambiguities before they can even be costed | 4 instruments | 0 |
| **R2, R6, R7, R8, R9, R10, R11, R12, R13, R14** — the remaining run-only cells, the 5 never-run instruments, EXT34 / MECH74 / COMB138, T3n on the legacy family, the judge panel at 2b/27b, the flag riders | ~200 of A's 311 | 9 |
| **P7** — retain activations, or accept that no phase-3c / phase-4 / think-probe number is reproducible | `.gitignore:21-22`; all three `-it` scales | 0 |
| **P4, P8–P14** — flag the register contest, stamp the pool nesting, stop tag collisions, replace the truncated write, adjudicate the phase-4/phase-3c verdict split | 8 recording gaps | 1 |
| **D14** — account for the 66-item pool, which no pass could place | 1 open question | 0 |

The asymmetry is the headline: **defensibility is 42 claims of offline work plus two boxes.
Exhaustiveness is ~290 further combinations, four design changes and a data-retention policy, and it
would close 22 more claims.** The two programs overlap in step 4 and almost nowhere else.
