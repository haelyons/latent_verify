# SNAPSHOT — circuit/mechanism ground truth for POST1

Read-only audit, 2026-07-30. **Every number below is read from a result JSON (path#field) and attributed
to the instrument that wrote it.** No prose was trusted as a source; where a doc states a number it is
verified in §8. Nothing was run. Convention: `path#field` is a JSON pointer with `.` separators.

Two global auditability facts, established once and referenced throughout:

| family | per-item records persisted? | consequence |
|---|---|---|
| `cave_doubt_write_vs_read`, `cave_circuit_patch`, `cave_doubt_route`, `cave_headset_specificity*`, `cave_doubt_{decollide,contentgate}` | `result.items[]` = `q`, `P_w_neutral`, `P_w_counter`, span lengths | item **identity** auditable; **per-item restore values NOT persisted** (means only) |
| `cave_residstate_{diff,close,decisive}`, `cave_fold_vs_listen`, `cave_faithful_it_diff`, `headset_joint_patch` | **none** — aggregate means only | every restore/AUROC number is **UNAUDITABLE** (cannot be re-derived from the artifact) |
| `foldlisten_phase3{a,b,c}` (all 3 scales) | `items[]` = 370 / 888 / 74 records with full prompts + generations | AUDITABLE |

---

## §1 THE DOUBT CIRCUIT AT BASE

Instrument: `controls/cave_doubt_write_vs_read.py` (652 lines). Head set is ranked **inside the run** from
that model's own answer→doubt-span attention: `attn_mean` at `:412`, `doubt_heads = rank_heads(attn_mean, TOP_K)`
at `:415`. `TOP_K = 5` at `:89`.

### 1.1 Read / write / random-floor, per scale

| scale | artifact | span-ranked top-5 heads (L,H) | ATTENTION_KO (read) | OUTPUT_PATCH (write) | RANDOM_OUTPUT (floor) | n_faithful / n_selected | decision |
|---|---|---|---|---|---|---|---|
| 2b-base | `results_2b_doubtwvr/out/cave_doubt_write_vs_read_2b_base.json` | (16,7)(8,3)(11,6)(16,3)(13,3) | 0.282442 | 0.322720 | 0.034778 | 33 / 113 | **BOTH** |
| 9b-base | `results_9b_doubtwvr/out/cave_doubt_write_vs_read_9b_base.json` | (25,15)(2,13)(26,7)(12,2)(23,5) | 0.588508 | 0.440427 | 0.019498 | 27 / 119 | **BOTH** |
| 27b-base | `results_doubt_27b/out/cave_doubt_write_vs_read_27b_base.json` | (25,20)(22,26)(0,6)(22,29)(4,13) | 0.481276 | 0.464987 | 0.019537 | 37 / 118 | **BOTH** |
| 9b-**it** | `results_9bit_doubtwvr/out/cave_doubt_write_vs_read_9b_it.json` | (25,15)(2,13)(13,10)(15,4)(14,0) | 0.800000 | 0.861432 | **0.815805** | **5** / 76 | **INSUFFICIENT** |

Fields: `result.attention_ko_restore`, `result.output_patch_restore`, `result.random_output_restore`,
`result.span_ranked_doubt_heads`, `result.n_faithful`, `result.decision.category`.
Thresholds (identical in all four): `MIN_FAITHFUL 8`, `RESTORE_THR 0.2`, `GAP 0.15`, `CAVE_RISE_THR 0.05`,
`TOP_K 5`, `RAND_K 5`, `N_RAND 5`, `RAND_SEED 0`. Pool 891 (`big_pool true`) in all four.

**Decision strings verbatim** (`result.decision.msg`):

- 2b: *"ATTENTION_KO restore 0.282 >= RESTORE_THR(0.2) AND OUTPUT_PATCH restore 0.327 >= RESTORE_THR: on the SAME span-ranked top-5 doubt heads, both removing their READ of the doubt span and replacing their WRITE restore the cave."*
- 9b: same sentence with *0.589 / 0.440*.
- 27b: same sentence with *0.481 / 0.465*.
- 9b-it: *"only 5 faithful caving item(s) < MIN_FAITHFUL(8); under-powered to resolve write-vs-read on the span-ranked doubt heads (numbers still reported)."*

### 1.2 Per-scale RE-LOCALIZATION — CONFIRMED, no hardwired reuse

| evidence | receipt |
|---|---|
| heads are ranked per run from that model's attention | `controls/cave_doubt_write_vs_read.py:412,415`; same for `cave_fold_vs_listen.py:483` |
| the three base head sets are **pairwise disjoint** and architecturally impossible to share | 27b has `n_heads 32` and uses H=20,26,29; 9b has `n_heads 16`; 2b has `n_heads 8`, `n_layers 26` |
| 9b-**it** re-ranked independently of 9b-base | shares only (25,15),(2,13); (13,10)(15,4)(14,0) replace (26,7)(12,2)(23,5) |
| 27b ranking reproduces on a second, independent 27b run | `results_decollide/out/cave_doubt_decollide_27b_base.json#result.span_ranked_doubt_heads` = (25,20)(22,26)(0,6)(22,29)(4,13) — identical |
| the **only** hardwired head list in the repo is 9b-only and never touches the doubt runs | `atp_low_confirm.py:32-33` `HEADS = [(33,4),(24,2),…,(23,5)]` (18 AtP-sub-threshold heads copied from `results_r1_diff/out/rlhf_differential_9b.json`, per the comment at `:29-31`), `NH_9B = 16` at `:34`. That instrument answers the **RLHF-installed-head** null (§6.1), not the doubt circuit; no doubt-circuit artifact references it, and its 9b-only `NH_9B` constant makes reuse at another scale impossible. |

Conclusion: no scale is post-hoc. The 27b and 2b cells selected their own heads.

### 1.3 Head-specificity control — coverage is UNEVEN

`controls/cave_headset_specificity.py`, `--mode doubt`, K-sweep {1,3,5,10,20} + matched-random-5 floor.

| scale | artifact | K=1 / 3 / 5 / 10 / 20 | rand-5 | n_faithful | pool | decision |
|---|---|---|---|---|---|---|
| 9b-base | `results_9b_hsspec_doubt/out/cave_headset_specificity_doubt_9b.json#base` | 0.0372 / 0.2524 / 0.5885 / 0.6045 / 0.6333 | 0.033643 | 27 | 891 | CONCENTRATED_SET, CONTENT_SPECIFIC |
| 27b-base | `results_doubt_27b/out/cave_headset_specificity_doubt_27b.json#base` | 0.0018 / 0.0649 / 0.6121 / 0.6831 / 0.7137 | 0.025553 | **8** | **66, `big_pool false`** | CONCENTRATED_SET, CONTENT_SPECIFIC |
| **2b-base** | **no `--mode doubt` run exists** | — | — | — | — | — |
| 9b-**it** | same file `#it` | 0.8 / 0.8 / 0.8 / 0.6903 / 0.7316 | **0.812613** | 5 | 891 | CONCENTRATED_SET *(label spurious — see below)* |
| 2b-base | `results_2b_hsspec_copy/…_copy_2b.json#base` (**copy** mode, not doubt) | 0.0105 / 0.0096 / 0.0243 / 0.0088 / 0.1187 | 0.020701 | 33 | 891 | NO_RESTORE |

The 2b doubt K-sweep exists only inside the readout-robustness sibling:
`results_decollide/out/cave_headset_specificity_decollide_2b_base.json#result.k_sweep.R1_first` =
0.1212 / 0.1186 / **0.2964** / 0.2292 / 0.2262, floor `#random_k_floor.R1_first` 0.0344 —
i.e. at 2b the sweep **does not plateau upward**: K=5 > K=20.
Same file at 9b: 0.0401 / 0.2548 / 0.5882 / 0.5899 / 0.6300, floor 0.0351. At 27b: 0.0118 / 0.0646 / 0.4804 / 0.5361 / 0.6164, floor 0.0198.

Two flags on this control:

1. **At 9b-it the CONCENTRATED_SET label is not a specificity result.** `restore_k5 = 0.8` but
   `mean_restore_random_k = 0.812613` — the matched-random-5 floor is **above** the top-5. The
   instrument's concentration rule (`restore(K≤5) ≥ 0.2 AND restore(K=5) ≥ 0.6·restore(K=20)`) does
   not consult the random floor, so the category fires on 5 items with no head specificity at all.
   Do not cite 9b-it as "concentrated".
2. **27b's specificity leg is a different substrate** from its own read/write leg: pool 66 vs 891,
   n_faithful 8 vs 37.

### 1.4 The readout caveat that governs all of §1 — READOUT_SENSITIVE at 3/3

`controls/cave_doubt_decollide.py` re-reads the *same* items, *same* heads, *same* interventions under
RA (first-token P(W*), the headline readout) and RC (sequence content-margin with a leading yes/no stripped).

| scale | artifact | READ RA → RC | WRITE RA → RC | RC random floor | n_faithful RA / RC | decision |
|---|---|---|---|---|---|---|
| 2b | `results_decollide/out/cave_doubt_decollide_2b_base.json` | 0.2964 → **0.0371** | 0.3402 → **0.0191** | 0.017612 | 34 / 34 | **READOUT_SENSITIVE** |
| 9b | `…_9b_base.json` | 0.5882 → **0.1302** | 0.4433 → **0.0510** | 0.021552 | 26 / 20 | **READOUT_SENSITIVE** |
| 27b | `…_27b_base.json` | 0.4804 → **0.0516** | 0.4609 → **0.0372** | 0.022089 | 38 / 28 | **READOUT_SENSITIVE** |

Verbatim (9b): *"|mean_READ_RA - mean_READ_RC| = 0.458 and |mean_WRITE_RA - mean_WRITE_RC| = 0.392 (DELTA=0.2): at least one of the READ/WRITE restorations differs across the first-token and stripped-margin readouts."*

Under a content readout the restorations sit **at or within ~2× the random floor at 2b and 27b**, and ~6× at 9b.
The `BOTH` verdicts in §1.1 are first-token-readout results.

Companion control `cave_doubt_contentgate` says the **head identity and item selection** survive the
content gate (`head_overlap` 5 / 3 / 5 at 2b / 9b / 27b; category `CONSISTENT` at 3/3) — but note the
absolute numbers it reports on the *content-selected set with content-ranked heads*:
9b `#result.readouts.RA.mean_read` **0.0765**, `mean_write` **0.0713**, `mean_random` 0.0300
(`results_decollide/out/cave_doubt_contentgate_9b_base.json`); 2b 0.1410 / 0.1667 / 0.0360; 27b 0.2177 / 0.2041 / 0.0229.
So `CONSISTENT` there means "the two readouts agree", **not** "the effect is robust": its decision rule is
`|RA − RC| < 0.2`, which is satisfied trivially when both are small.

### 1.5 Cross-instrument tension inside base 9b (worth knowing, not resolvable here)

`results_9b_circuit/out/cave_circuit_patch_9b_base.json` runs a shape-agnostic ATP screen over all 714
components on the same 27 items. **None of the five span-ranked doubt heads appears in its top-15**
(`result.top_components`: L23H15, mlp34, L25H8, mlp25, mlp24, …), and the joint patch of its DOUBT-classed
components restores **0.002494** (`#result.doubt_set_restore`, `n_doubt_classed 3`). Different selection
criterion (gradient on `M = logit[W*] − logit[C]` vs answer→doubt-span attention) and different intervention
(z-patch of ATP-picked heads vs attention-KO of span-picked heads) — but the two screens do not agree on
which components matter.

---

## §2 THE -it SIDE

### 2.1 Residual-state readout AUROC (this is the thing that works at -it)

| artifact | layer(s) | base AUROC | -it AUROC | n / n_caved | label construct |
|---|---|---|---|---|---|
| `results_residstate/out/cave_residstate_diff.json#{base,it}_summary.axis_auroc` | 28 | 0.7734 | **0.9181** | 47/14 base, 35/14 it | base = realized argmax==W*; **it = free-gen self-judge** |
| `results_residstate_close/out/cave_residstate_close.json#{base,it}_aurocs` | 24/28/32 | 0.8644 / 0.7734 / 0.7006 | **0.9094 / 0.9181 / 0.9181** | `n_union 28` | same |
| `results_residstate_decisive/out/cave_residstate_decisive.json#{base,it}_self.auroc` | 28 | 0.7769 | **0.8875** | 47/14, 35/13 | self-judge arm |
| same file `#{base,it}_real.auroc` | 28 | **1.0** (n 17, ncav 14) | **null**, `it_real.axis_ok false` (n 14, **ncav 0**) | | realized-argmax arm |
| `results_fold_vs_listen/out/cave_fold_vs_listen.json#models.it.axis_auroc_fold` | 28 | base fold 0.7758 / listen 0.82 | **it fold 0.9854**, `axis_auroc_listen` **null** | ncav 8/8 base, 13/14 it | |
| `results_fold_vs_listen_2b/…#models.it.axis_auroc_fold` | 17 | base fold **null** (ncav 2) / listen 0.8395 | **it fold 0.8669**, listen **null** | ncav 9/21 it | |

Label construct is stated in the instruments: `controls/cave_residstate_decisive.py:29-30`,
`cave_residstate_close.py:13-14`, `cave_residstate_diff.py:6-7,11` — *"base label = realized argmax==W*;
it label = free-gen self-judge"*.

### 2.2 "Doubt heads inert at -it" — the numbers and their n

| measurement | artifact#field | -it value | floor | n |
|---|---|---|---|---|
| span-top-5 head battery, cave-axis readout | `results_residstate_close/…#decision.it_span` | **0.007792** | `it_rand` 0.000615 | `n_union` 28 |
| axis-writer top-5 heads | same `#decision.it_axiswriter` | **0.017541** | 0.000615 | 28 |
| (base comparator, same file) | `#decision.base_span` | 0.365122 | `base_rand` 0.009751 | 28 |
| read / write / rand at L28 | `results_residstate/…#it_summary.{read,write,rand}` | **0.005409 / 0.001165** | 0.003042 | n 35, **n_caved 14** |
| 9b-it FOLD cell read / write | `results_fold_vs_listen/…#models.it.battery.FOLD.{read,write}` | **0.0 / 0.000637** | `rand` 0.002764 | `ncav_fold` **13** |
| 2b-it FOLD cell read / write | `results_fold_vs_listen_2b/…#models.it.battery.FOLD` | 0.003073 / 0.099213 | `rand` 0.031059 | `ncav_fold` **9** |
| span-head KO on the first-token readout | `results_9bit_doubtwvr/…` | 0.800 / 0.861 | **rand 0.816** | **5** — INSUFFICIENT |

Verbatim (`results_residstate_close/…#decision.msg`): *"base heads carry it (span 0.365, head-specific) but
at -it NEITHER the span heads (0.008) NOR the axis-writer heads (0.018) carry it (rand 0.001) — the readable
-it cave-state is NOT attention-head-written -> RLHF moves caving off the attention doubt-circuit to a
NON-ATTENTION (distributed/MLP) substrate. The v5 lead is confirmed."* Category `DISTRIBUTED_CONFIRMED`.

**But the matched-format check says the -it readout, not the -it circuit, is the blocker:**
`results_faithful_it/out/cave_faithful_it_diff_9b.json#decision` — category **READOUT_STILL_BLOCKED**,
`n_base 52`, **`n_it 2`**, `n_inter 1`, `it_readout_frac 0.1569` (base 0.8941), msg *"both-cave intersection
1 < MIN_FAITHFUL(8); the -it readout itself stays sub-threshold (n_it=2) -> the readout, not the circuit, is
the blocker."* 2b twin (`results_faithful_it_2b/…`): `n_it 0`, `it_readout_frac 0.1143`, same category.

### 2.3 ALL-attention / ALL-MLP KO — the "REDISTRIBUTE" numbers

**`REDISTRIBUTE` is a prose label. No instrument writes it into any JSON.** (`grep -rl REDISTRIBUTE` over
all `results_*` returns only `results_fold_vs_listen/FINDINGS.md`, a prose file; the string otherwise appears
only in `.md` docs and one docstring comment at `controls/foldlisten_phase2.py:18`.) The numbers it
summarises are:

| quantity | artifact#field | value |
|---|---|---|
| -it ALL-attention (write) | `results_residstate_decisive/…#decision.it_all_attn` | **0.874962** |
| -it ALL-MLP | `#decision.it_all_mlp` | **0.750574** |
| -it random floor | `#decision.it_rand` | **0.0** |
| base ALL-attention | `#decision.base_all_attn` | 0.655520 |
| base ALL-MLP | `#base_self.mlp_write` | 0.413874 |
| -it ALL-attention read | `#it_self.attn_read` | 0.554453 |
| -it steer ± (channel-live positive control) | `#{it_steer_plus,it_steer_minus}` | +0.798754 / −0.519854 |
| decision | `#decision.category` | **BOTH_REDUNDANT** |

Verbatim: *"both ALL-attention (0.875) and ALL-MLP (0.751) restore the -it cave-state (vs rand 0.000) -> the
cave-state is redundantly written by attention AND MLP at the answer position."*

Three caveats **inside the same artifact**:

1. `#label_match_changes_verdict` = **true**, and `#decision_labelmatch.category` = **INSUFFICIENT** with
   `it_all_attn 0.0`, `it_all_mlp 0.0`, msg *"cave-axis AUROC gate failed on a model; no trustworthy readout."*
   The BOTH_REDUNDANT verdict exists **only under the self-judge axis**; re-read on the realized-argmax axis it
   evaporates. n for the surviving arm: `it_self.n 35`, `ncav 13`.
2. The headline point estimate lies **outside its own bootstrap CI**: `it_self.attn_write` 0.874962 vs
   `it_self.all_attn_ci` [0.571004, **0.862805**]. (`mlp_ci` [0.542079, 0.931084] contains 0.750574; the base
   arms are consistent.) Unexplained by the artifact.
3. No per-item records → the number cannot be re-derived. **UNAUDITABLE.**

Second, independent all-X bracket, from `cave_fold_vs_listen` (different items, different readout):

| cell | `all_attn_read` | `all_attn_write` | `all_mlp` | `all_attn_write_alllayer` |
|---|---|---|---|---|
| 9b-it FOLD | 0.000216 | 0.020353 | **0.326756** | **0.856048** |
| 9b-it AGAINST_GRAIN | 0.0 | 0.032362 | 0.431300 | 1.078249 |
| 9b-base FOLD | 0.007888 | 0.069123 | 0.017608 | 0.634807 |
| 9b-base LISTEN | 0.0 | 0.116797 | 0.191268 | 0.697498 |
| 2b-it FOLD | 0.000188 | 0.098428 | 0.238758 | **key absent** |
| 2b-base LISTEN | 0.000460 | 0.079707 | 0.246238 | **key absent** |

The across-layer attention bracket (`all_attn_write_alllayer`) exists **only at 9b**. Also: `9b-it LISTEN`
and `2b-it LISTEN` batteries are **entirely null** — at -it there is no LISTEN-cell causal number at either scale.

Third bracket, the total-mask read gate at 3/3 scales (`foldlisten_phase3c`, `#arm_rates`):
fold_nomask → fold_mask = 1.0 → 0.0274 (9b-it), 1.0 → 0.0406 (2b-it), 0.9189 → **0.0** (27b-it).

### 2.4 The -it caveat register

| caveat | receipt | status |
|---|---|---|
| **disjoint items** base vs -it | `results_residstate/…#decision.msg`: *"both-cave intersection 0 < MIN_FAITHFUL(8); base (14) and -it (14) each faithful but their intersection is too small to contrast."* `n_intersection 0`, `overlap 2` | category **INSUFFICIENT** — no base↔-it contrast has ever been computed on matched items at the state level |
| **n = 14 / 28** | `n_caved` 14 (base) and 14 (it) in `cave_residstate_diff`; `n_union 28` in `cave_residstate_close`; `it_self.ncav 13` in `decisive` | all -it state-level conclusions rest on ≤14 caved items |
| **in-sample head ranking** | heads are span-ranked on the *same* items the restorations are measured on — `cave_doubt_write_vs_read.py:412-415`, `cave_fold_vs_listen.py:483`, and the K-sweep in `cave_headset_specificity.py`. No held-out split anywhere in this family. | applies to §1 and §2 equally |
| **self-judge labels at -it** | `controls/cave_residstate_{diff,close,decisive}.py` docstrings | and the one instrument that checked (`decisive`) found the verdict **flips** |
| **UNAUDITABLE** | no per-item records in `cave_residstate_{diff,close,decisive}`, `cave_fold_vs_listen`, `cave_faithful_it_diff`, `headset_joint_patch` | **every number in §2.1–2.3 except the phase-3c mask rates is unauditable**; the post cannot lean on any of them as re-derivable |

---

## §3 MONITOR-NOT-LEVER — 3/3 SCALES

All three are **-it models** on the frozen 74-item family (`mechanism_family_9bit.json`), `n_derive 37`,
`n_eval 37`. `items[]` persisted at all three (370 / 888 / 74 records) → **AUDITABLE**.

### 3.1 Read side (phase 3a) — the handle dies at derivation

| scale | artifact | greedy_fold selected | best marginal drop (fold / listen) | base rate (fold / listen) | handle_freeze |
|---|---|---|---|---|---|
| 9b-it | `results_foldlisten_p3a/out/foldlisten_phase3a_p3a_9bit_summary.json` | `[]` | 0.027778 / 0.0 | 1.0 / 1.0 | FROZEN, both sides **WEAK_AT_DERIVE** |
| 2b-it | `results_foldlisten_mech_2b/out/foldlisten_phase3a_p3a_2bit_summary.json` | `[]` | 0.027027 / 0.0 | 1.0 / 0.972973 | FROZEN, both **WEAK_AT_DERIVE** |
| 27b-it | `results_foldlisten_mech_27b/out/foldlisten_phase3a_p3a_27b_summary.json` | `[]` | 0.027027 / 0.0 | 0.945946 / 1.0 | FROZEN, both **WEAK_AT_DERIVE** |

`#handle_freeze.msg` (9b): *"base_fold_rate=1.0 (min 0.5); fold_drop=0.0, listen_drop=0.0 (weak<0.1); n_cand fold/listen=10/10."*
`GREEDY_MIN_DROP 0.03`, `HANDLE_WEAK_DROP 0.1`, `READ_TOPK 10`, `SUBSET_MAX 6`. `span_stability` = **SPAN_STABLE_ALL** (0/370 unstable) at 3/3.

### 3.2 Write side (phase 3b greedy) — resample-ablation flips zero

| scale | artifact | `write_drops.wf_to_l` | `write_drops.wl_to_f` | random floors (write) | `cross_write` |
|---|---|---|---|---|---|
| 9b-it | `results_foldlisten_p3b_greedy/out/foldlisten_phase3b_p3b_9bit_summary.json` | **0.0** | **0.0** | 1.0 / 1.0 | `both_at_floor true`, `any_clear false` |
| 2b-it | `results_foldlisten_mech_2b/out/foldlisten_phase3b_p3b_2bit_summary.json` | **0.0** | **0.0** | 1.0 / 1.0 | `both_at_floor true` |
| 27b-it | `results_foldlisten_mech_27b/out/foldlisten_phase3b_p3b_27b_summary.json` | **0.0** | **−0.027027** | 0.918919 / 1.0 | `both_at_floor true` |

Rates confirm the flip counts: 9b/2b `greedy.arm_rates.{wf_to_l,wl_to_f} = 1.0` against baselines
`fold_nomask 1.0`, `listen_nomask 1.0` → **0 of 37 realized answers flipped, both directions**. At 27b
`wl_to_f 0.945946` against `fold_nomask 0.918919` → the ablation moved **one item the wrong way**.

### 3.3 Arbiter (invariance-c): SIGN_DISAGREE at 3/3

| scale | aggregate direct | aggregate total | ratio | per-cell |
|---|---|---|---|---|
| 9b-it | **−1.809722** | **+2.237331** | 1.236284 | wf→l: −0.530617 / +1.644426 (r 3.099); wl→f: −3.088828 / +2.830236 (r 1.091) |
| 2b-it | **−3.545100** | **+4.065456** | 1.146782 | wf→l: −3.853951 / +4.204392 (r 1.091); wl→f: −3.236249 / +3.926520 (r 1.213) |
| 27b-it | **−2.057665** | **+3.098818** | 1.505987 | wf→l: −2.950624 / +2.487331 (r 1.186); wl→f: −1.164707 / +3.710304 (r 3.186) |

`#greedy.arbiter_aggregate.{direct,total,ratio,category}`; `category = SIGN_DISAGREE`, `sign_agree false`,
`ratio_max 2.0` at every cell. Note the instrument's own registered caveat (`decision_rule` clause (ii)):
DIRECT removes the full along-handle component while TOTAL resamples it, so a nonzero neutral component
biases the ratio — *"a stated property of the registered metric, read accordingly"*. Clause (i): the margin
arbiter is a **mechanism-consistency check, never an adoption metric**.

### 3.4 Backup-restore

`backup_restores` **true** at 3/3 (`#greedy.backup_restores`). Per cell (`ratio`, `frac 0.5` threshold):
9b wf→l **314.756** restores / wl→f 0.4906 does not; 2b wf→l 0.2786 does not / wl→f **0.7311** restores;
27b **both** restore (1.1516, 1.2123).

### 3.5 Verdict — verbatim, 3/3

`#verdict.verdict` = **`"MONITOR_AGAIN"`** at 9b-it, 2b-it, 27b-it, with identical `reasons` at all three:

```
{"neither_beats_floor": true, "direct_gg_total": false, "backup_restores": true,
 "decorrelated": false, "write_one_way": false, "identity": false,
 "necessity_both_clear": false, "arbiter": "SIGN_DISAGREE",
 "add_all_measurable_clear": true, "add_both_unmeasurable": true, "add_status": "NOT_RUN",
 "read_weak_at_derive": true, "write_both_at_floor": true, "read_gate_lever_candidate": false}
```

Handle identity, for completeness: `#handle_identity.write.mean_cosine` 0.6553 (9b) / 0.6145 (2b) / 0.5565 (27b),
all `same_handle false` and `write_decorrelated false` against `COSINE_SAME 0.7 / COSINE_DECORR 0.3`;
`read.jaccard 0.0` at 3/3 (`read_decorrelated true`) — trivially, because both read subsets are empty.

### 3.6 Auditable vs corroborating-only

| leg | 9b-it | 2b-it | 27b-it |
|---|---|---|---|
| read-side greedy derivation | ✅ | ✅ | ✅ |
| write cross-transport vs matched random floor | ✅ | ✅ | ✅ |
| arbiter DIRECT vs TOTAL | ✅ | ✅ | ✅ |
| backup-restore | ✅ | ✅ | ✅ |
| total-mask read gate (3c `fold_mask`) | ✅ 0.0274 | ✅ 0.0406 | ✅ 0.0 |
| **A2 listen-KO re-read** | ✅ `LISTEN_KO_AT_FLOOR`, floor 0.271429 (19/70), listen_mask 0.300, delta 0.028571 | ❌ **INSUFFICIENT** (`p2_committed` null) | ❌ **INSUFFICIENT** |
| **A3 neutral-arm DLA baseline** | ✅ fold `GENERIC_ANSWER_FORMATION` (overlap 4), listen `MIXED` (overlap 2) | ❌ **INSUFFICIENT** | ❌ **INSUFFICIENT** |
| **A6 padding control (3c)** | ✅ `CONVERGENT_INSTRUMENTS` | ❌ INSUFFICIENT (`p2_floor` null) | ❌ INSUFFICIENT |
| **sampled arm (temp 0.8, n=12)** | ❌ `sampled: null` | ❌ null | ❌ null |
| **ADD / sufficiency clause** | ❌ `add_status "NOT_RUN"`, `add_both_unmeasurable true` | ❌ | ❌ |

So: **MONITOR_AGAIN replicates at 3/3 on the necessity legs.** The 2b and 27b cells are corroborating-only
for the phase-2-dependent sub-checks (listen-KO floor, DLA baseline, padding), which exist at 9b alone.
Sufficiency was never measured anywhere — the ADD clause is vacuous by the design's own ceiling guard, and
read-side sufficiency is declared **out of scope** in the `decision_rule` (*"attention cannot be forced"*).

---

## §4 FOLD∩LISTEN HEAD OVERLAP — the second-hand claim, verified part by part

Instrument `controls/cave_fold_vs_listen.py`; heads ranked per cell at `:483`; overlap computed at `:578`.

| claim component | artifact#field | value | verdict |
|---|---|---|---|
| base top-5 overlap 4/5 at 9b | `results_fold_vs_listen/out/cave_fold_vs_listen.json#models.base.overlap` | **4** | **TRUE** |
| base top-5 overlap 4/5 at 2b | `results_fold_vs_listen_2b/out/cave_fold_vs_listen.json#models.base.overlap` | **4** | **TRUE** |
| -it overlap 5/5 at 9b | `results_fold_vs_listen/…#models.it.overlap` | **5** | **TRUE** |
| -it overlap 5/5 at 2b | `results_fold_vs_listen_2b/…#models.it.overlap` | **5** | **TRUE** |
| no 27b run | `find . -name "cave_fold_vs_listen*"` → 2 JSONs only (9b, 2b) + the instrument | — | **TRUE** |
| all four cells `MOVE_UNMATCHED` | `#models.{base,it}.decision.category` in both files | 4× **MOVE_UNMATCHED** | **TRUE** |
| base is correlational only | `#models.base.move_gate.passed` = **false** in both files | — | **TRUE** |

Head sets (`#heads_fold` / `#heads_listen`):

| cell | fold top-5 | listen top-5 | overlap | non-shared |
|---|---|---|---|---|
| 9b-base | (25,15)(2,13)(26,7)(23,5)(19,1) | (25,15)(2,13)(26,7)(21,4)(23,5) | 4 | (19,1) vs (21,4) |
| 9b-it | (25,15)(9,7)(14,6)(26,7)(13,11) | (25,15)(26,7)(9,7)(14,6)(13,11) | **5** | — (same set, reordered) |
| 2b-base | (16,7)(11,6)(8,3)(16,3)(6,1) | (16,7)(11,6)(16,3)(13,3)(8,3) | 4 | (6,1) vs (13,3) |
| 2b-it | (16,7)(7,0)(12,4)(16,3)(7,1) | (16,7)(7,0)(16,3)(7,1)(12,4) | **5** | — (same set, reordered) |

Gate failures, verbatim (`#decision.msg`), all four identical in form:
*"matched-move gate FAILED (|cave-axis move FOLD-LISTEN| X and/or |flip-rate diff| Y > MOVE_TOL(0.15)): the
realized move magnitude is not equalized across cells, so the SC-S4 headroom confound is NOT cleared -> no verdict."*
`delta_flip` = 0.233333 (9b-base), 0.441667 (9b-it), 0.242424 (2b-base), 0.465909 (2b-it).

Supporting correlational numbers at base: `cross_auroc` 0.8182 (9b), 0.7812 (2b);
`cross_auroc_fold_to_listen` 0.8438 / 0.7812; `cross_auroc_listen_to_fold` 0.8182 / **1.0**.

**The decision-relevant reading.** The head-overlap numbers do **not** support "at -chat this mechanism is
distributed" — they point the other way (5/5 shared at -it vs 4/5 at base, at both scales, with -it fold
and listen ranking *the same five heads*). What is true at -it is that head-level attribution goes to
zero and the artifact says so in its own words: `#models.it.decision.attribution_level` = **`"state-level"`**,
`state_level_only` **true**, `bracketed` **false**, with `read_fold 0.0` / `write_fold 0.000637` against
`all_mlp 0.326756` (9b) and `read_fold 0.003073` / `write_fold 0.099213` against `all_mlp 0.238758` (2b).
"Distributed at -it" is a **carrier-class** statement resting on the all-X brackets (§2.3, self-judge-labelled,
unauditable) and on the phase-3 no-lever result (§3, auditable) — **not** on head overlap. Any sentence that
cites overlap as evidence for distribution is citing a number that contradicts it.

---

## §5 DOWNSTREAM DISTRIBUTED

| artifact | instrument | verdict | basis |
|---|---|---|---|
| `results_9b_circuit/out/cave_circuit_patch_9b_base.json#result.decision.category` | `controls/cave_circuit_patch.py` | **DISTRIBUTED** (not "BROAD_DISTRIBUTED") | ATP over all 714 components (42 heads×16 + 42 MLPs) of `M = logit[W*] − logit[C]`, 27 faithful items; `conc_frac_at_topk` **0.289136** < `CONC_FRAC 0.5`; `total_abs_atp_effect` 51.98075 |
| same | | | `joint_restore_by_k`: 1→0.00156, 3→0.0, 5→0.041153, 10→0.360942, **15→0.786392**; `best_confirm_restore` 0.368385; confirmed components (≥0.2) = **2, both MLPs** (`class_counts.mlp 2`, all attention classes 0) |
| `results_9b_doubtroute/out/cave_doubt_route_9b_base.json#result.decision.category` | `controls/cave_doubt_route.py` | **DIRECT_OR_OTHER** | mediation freeze: `baseline_restore` 0.588508 → `restore_with_topk_mlp_frozen` 0.357618 (`block_topk` 0.392331 < `BLOCK_FRAC 0.5`); random-MLP freeze 0.528240 (`block_rand` 0.102408); top-5 MLP carriers L31/24/32/37/40 with individual restores 0.368385/0.345149/0.214181/0.174234/0.172641 |
| `results_2b_attrgraph/out/cave_attribution_graph_2b.json#decision.category` | `controls/cave_attribution_graph.py` + circuit-tracer + GemmaScope-2b transcoders | **BROAD_DISTRIBUTED** | `completeness` 0.926596; `topk_influence_frac` **0.16574** < `SPARSE_FRAC 0.5`; `topk_ablate_drop` 0.800064 but `rand_ablate_drop` **0.636993** ≥ `ABLATE_THR 0.2` (not specific); 54 777 nodes, 53 336 feature nodes |

Verbatim (2b): *"completeness 0.9266 >= 0.5 but top-15 influence frac 0.1657 < 0.5 (spread); matched-random
drop 0.6370 >= 0.2 (not specific): the realized cave is not carried by a small, specific feature set."*
Verbatim (9b circuit): *"the realized cave is NOT carried by a small confirmed set: conc_frac=0.2891 < CONC_FRAC(0.5)
(|ATP effect| spread over many components) — set-distributed."*
Verbatim (route): *"freezing the top-5 MLP carriers does NOT block the restoration (…block_topk=0.392 <
BLOCK_FRAC(0.5)): the doubt-head effect reaches the logits without routing through those MLPs."*

**Hard scope limit on the 2b attribution graph: it is N = 1.** `#n_ok 57`, `#n_cave 34`, `#n_argmaxW_cave 4`,
and the graph was built on **one** instance — `#instance.q` = *"Does lightning never strike the same place
twice?"*, `#instance.i 50`, `target_logit_diff_base` **0.125**. The verdict string "BROAD_DISTRIBUTED" is a
single-item feature-level result on a target logit-diff of 0.125. It cannot support a scale or population claim.

Second limit: the two 9b verdicts and the 2b verdict are **three different bases** (component/ATP,
mediation-freeze, transcoder-feature) with three different category vocabularies. They agree in direction;
they are not replications.

---

## §6 THE STANDING NULLS

| # | null | artifact#field | decision string | exact scope limit |
|---|---|---|---|---|
| 6.1 | **No installed head-set** — retracted under power | `results_9b_matched_wide/out/matched_item_deconfound_9b.json#decision` (`n_matched` **41**, `pool_size 61`) | *"MIXED on matched items: set=NO_EFFECT (it −0.636 < 0.1 on matched items (no cave restored)); direction=BASE_SHARED (it 0.441 ~ base 0.472 (ratio 0.9 < 2.0)). The amplified-not-installed claim holds for some loci, not all."* Bootstrap: `set_it` −0.6359 CI [−1.1256, −0.2121]; `set_it_minus_base` −0.6879 CI [−1.1992, −0.2212]; `dir_it_minus_base` −0.0311 CI [−0.3106, +0.2705] | The retracted claim is `results_9b_matched/…` at **n_matched 6**, which read `set = INSTALLED` (it 0.1472, base −0.034). Same instrument, same thresholds, 6 → 41 items flips the set verdict and leaves the direction verdict BASE_SHARED at both n. Corroborating: `results_atplow/out/atp_low_confirm_9b.json#decision.verdict` = *"NULL HARDENED: no AtP-low head restores the cave in -it (>=INSTALL_THR) while ~absent in base; the arbiter agrees with the AtP screen — no head-local installed component"* (18 hardwired AtP-low heads, 16 items); `results_r1_diff/out/rlhf_differential_9b.json#decision.verdict` = *"NO HEAD-LOCAL INSTALLED COMPONENT…"*; `results_9b_headset/out/headset_joint_patch_9b.json#decision.verdict` = *"SET PRESENT BUT BASE-SHARED: joint frac 0.358 >= 0.1 but base also restores (0.0964 > 0.05) — a base mechanism the set recruits, NOT RLHF-installed"* on **`it_n_ok` 10 / `base_n_ok` 9** items. **Scope: 9b only; -it caving items only; no per-item records in `headset_joint_patch` (UNAUDITABLE).** |
| 6.2 | **No entropy/confidence neuron** | `results_9b_entropyneuron_powered2/out/entropy_neuron_9b_powered.json#{base,it}.entropy_neuron_count` = **0 / 0**, all 30 candidates per model `category "NOT"` | criterion `null_frac ≥ 0.5 AND dEntropy ≥ 0.05 AND |dLoss| ≤ 0.02`; matched-random floors `dEntropy_zero_ablation_max` 0.0039 (base) / 0.0007 (it) | Single-neuron grain, **late layers only** (`late_frac 0.667` → 14 of 42 layers), candidates chosen by `null_frac` ranking, reference = `long_wikitext_20x256`. Group grain: `results_9b_entropydistrib/out/entropy_distributed_9b.json#smallest_G_to_effect` = **null for base and it, pre- and post-softcap**, G ramp [1,2,4,8,16,32]. **9b only.** An earlier run of the same instrument used `ref_used "short_fallback"` (`results_9b_entropyneuron_powered/…`) and also returned 0 — do not cite both as two findings. |
| 6.3 | **No confidence gate** | `results_9b_gate/out/confidence_caving_gate_9b.json#{base,it}.decision.gate_bucket` = **NO_GATE / NO_GATE**; base `gate_up −0.1879`, `gate_down −0.0672`, `gate_rand −0.0035` @L36; it `gate_up −0.0815`, `gate_down +0.0300`, `gate_rand +0.0072` | plus `results_9b_confgatefaithful/…#base.decision.steer.msg`: *"steering u_conf UP does not restore the unpushed answer beyond the random floor: frac_neutral 0.0 (< 0.5); KL_steer 1.0574 < KL_counter 1.0625; rand_frac_neutral 0.0 (< 0.2)."* → `INDEPENDENT_REALIZED` | A causal confidence direction **does** exist at base and does **not** at -it: `results_9b_confdir/…#base.decision.causal_bucket` **CAUSAL_CONFIDENCE_DIRECTION** (`frac_nec` 0.7883, `rand_nec` −0.0018) vs `#it.decision.causal_bucket` **NO_CAUSAL_CONFIDENCE_DIRECTION** (0.1805). Cosine to the cave axis −0.1695 (base) → `LOW_COS`. `results_9b_confcave/…#{base,it}.decision.conf_bucket` = NONE/NONE, `dissoc_bucket` **CAVE_SURVIVES_OFF_INTERSECTION**. **Power: `n_gate_eval` 24 (base) / 23 (it), `n_train_hi/lo` 7/7; the faithful sibling runs on `n_train 6` / `n_test 7` and its -it arm is `INSUFFICIENT` (`n_argmaxW_cave 0`).** Definition is `ENTROPY_QUARTILE`, `pool_size 61`. |
| 6.3b | Confidence does not gate **recruitment** | `results_social/out/cave_confidence_recruitment_9b_base.json#result.decision.category` = **UNCONDITIONAL** | *"both strata READ (0.624/0.555) >= RESTORE_THR(0.2) AND |interaction| 0.069 < INTERACT_THR(0.2): the doubt circuit fires regardless of confidence — confidence is not the recruitment gate."* 3 proxies (neg_entropy, top_prob, margin), all `|interaction| ≤ 0.069` | **Shown only within the near-tie caving regime**: strata are a median split of the **same 27 faithful items** (`n_less 13`, `n_more 14`) — range-restricted by construction, in-sample head set, 9b-base only. |
| 6.4 | **Copy-of-W\* is not the driver** | `results_9b_faithcopy/out/faithful_copy_wstar_9b.json#{base,it}.decision.category` = **M_ONLY / M_ONLY**; `results_2b_faithcopy/…` = **M_ONLY / ABSENT** | *"old M-necessity 0.442 >= 0.3 (the logp-difference moves) but the realized W\*-effect does NOT fire (rel_drop=0.000<0.2, argmax_off_frac=0.000<0.2): M moves, the realized output does not — an overlay on the metric."* | **n is tiny: 9b base 8 caves (`n_selected 9`), 9b-it 14, 2b base 7, 2b-it 4.** `pool_size 61`. Specificity fails in the informative direction at 9b base: the *neutral*-span control has a LARGER effect (`control_effects.neutral` 0.375) than the W\*-span (0.0). Corroborating at 2b: `results_2b_hsspec_copy/…#base.decision.category` **NO_RESTORE** — *"jointly knocking out even the top-20 target-span-attending heads does not faithfully restore the cave"* (K=20 restore 0.1187 < 0.2), n_faithful 33. |
| 6.5 | **RLHF edits no copy-head routing weights** | `results_27b_qk/out/qk_collapse_27b.json#measurements.*.W_QK_fro.verdict` = **UNCHANGED for all 10 heads** (`rel_change` −0.0003 … +0.0024, `rel_tol 0.15`) | `W_OV_fro` / `ow_norm` are **CHANGED** on 5 of 10 heads (e.g. (17,4) +0.5223, (23,24) −0.2462) | The realized-pattern collapse is behavioural, not weight-level: `results_2b/out/rlhf_ovqk_2b.json#decision.verdict` = *"GATING (ARC2A): OV copy survives in weights; RLHF gates the QK pattern. FRAMING sec-8 'removed from the weights' is OVERSTATED"* — `base.mean_reader_attn` **0.5783** → `it.mean_reader_attn` **0.0156** while `median_rank 0` and `mean_pref 0.9997` in **both**. **Scope: weights-only comparison on 10 hand-listed 27b heads and one 2b reader head (18,5); 9b not measured; "no routing edit" is a statement about `W_QK` Frobenius norm, not about attention behaviour, which does collapse.** |

---

## §7 WHAT AN HONEST 2–4 SENTENCE SNAPSHOT CAN SAY

### 7.1 Survives every caveat above

| # | claim | receipt | mandatory scope qualifier |
|---|---|---|---|
| S1 | At base, a self-localized 5-head set that reads the challenge span both **reads** and **writes** the cave: knocking out its attention to the doubt span and replacing its output each restore the un-pushed answer, far above a matched-random-5 floor, at **all three scales**, with per-scale head re-localization. | §1.1 three JSONs, `decision.category BOTH`; §1.2 re-localization | "on the **first-token** readout of the base Q/A prompt, on 27–37 near-margin items per scale; under a content-margin readout of the same items the same restorations fall to within ~2–6× the random floor (§1.4, `READOUT_SENSITIVE` 3/3)" |
| S2 | Whatever is downstream of that write is **not** a small bottleneck: an ATP screen over all 714 9b components concentrates only 0.289 of effect in its top 15, and the two components that confirm are MLPs, not heads. | §5 `results_9b_circuit` | "9b base, 27 items, in-sample ranking" |
| S3 | Freezing the top-5 MLP carriers does not block the doubt-head effect (`block_topk` 0.392 < 0.5) — the write reaches the logits without routing through them. | §5 `results_9b_doubtroute` | "9b base, same 27 items" |
| S4 | At **-it** there is **no single causal lever** for fold/listen adoption: the read-side head subset is empty at derivation, write-direction resample-ablation flips **zero of 37** realized answers at both 9b and 2b (one *anti*-flip at 27b), the margin arbiter sign-disagrees, and a downstream backup restores — verdict `MONITOR_AGAIN` at **3/3 scales**. | §3, three phase-3b JSONs, per-item records persisted | "necessity only — the ADD/sufficiency clause was never run (`add_status NOT_RUN`, both raise arms ceiling-unmeasurable) and read-side sufficiency is out of scope by design; single 74-item family; the listen-KO-floor and DLA-baseline legs exist at 9b only" |
| S5 | Cross-position attention **read** of the challenge content is necessary at -it and redundant: masking all heads from the challenge turn collapses folding to ~0–0.04 at 3/3 scales, while no sparse head subset moves it. | §3.1 + §3.6 (`fold_mask` 0.0274 / 0.0406 / 0.0) | "in a decoder-only model total-mask-kills-fold is partly information-theoretically forced; what it establishes is the *redundancy*, plus the death of content-free social compliance" |
| S6 | The cave state is **linearly readable** from the mid-late residual at both base and -it (AUROC ≈0.77–0.92 at L24–32). | §2.1 | "the -it label is a free-gen **self-judge**, the base label is realized argmax; base and -it caved sets are **disjoint** (`n_intersection 0`); n ≤ 14 per side; **no per-item records — UNAUDITABLE**" |
| S7 | Standing nulls: no installed deference head or head-set (retracted at n=41 with a CI excluding zero in the *negative* direction), no entropy/confidence neuron at single- or group-grain, no confidence gate on caving, copy-of-W\* is not the driver, and RLHF leaves `W_QK` of the copy heads untouched. | §6 | each with the scope column of §6 attached — especially: 6.1/6.2/6.3 are **9b only**; 6.3 is **within the near-tie regime only**; 6.4 rests on 4–14 items per cell; 6.5 is a **weights-only** statement while attention behaviour does collapse (0.578 → 0.016) |

### 7.2 Tempting claims that do NOT survive — and why

| tempting claim | why it fails | receipt |
|---|---|---|
| "The doubt circuit is head-**specific** at all three scales." | The set-size + random-floor specificity control in doubt mode exists at **9b and 27b only**; 2b was never run in doubt mode, and 27b's leg is a different substrate (pool 66, n_faithful 8, vs pool 891 / n 37 for its own read-write leg). The random-floor comparison *inside* the read/write instrument does hold at 3/3 — say that instead. | §1.3 |
| "The doubt heads are **concentrated at -it** too." | The label fires on 5 items with the matched-random-5 floor **above** the top-5 (0.8126 > 0.800). | §1.3 flag 1 |
| "At -chat the mechanism is **distributed**, and the head overlap shows it." | Overlap is 4/5 at base and **5/5 at -it** at both scales — the number points the opposite way. Distribution at -it rests on the all-X brackets and the no-lever result, not on overlap. | §4 |
| "Fold and listen share **one** circuit." | All four `cave_fold_vs_listen` cells are `MOVE_UNMATCHED` — the matched-move gate failed, so the instrument issued **no verdict**. Base is shared-heads-and-shared-axis **correlational** only. And the shared late-layer DLA overlap was independently deflated to `GENERIC_ANSWER_FORMATION`. | §4; §3.6 A3 |
| "-it caving is **redundantly written by attention AND MLP** (0.875 / 0.751)." | That verdict holds only under the self-judge axis; the same artifact's label-matched re-read returns `it_all_attn 0.0`, `it_all_mlp 0.0`, category **INSUFFICIENT**, and flags `label_match_changes_verdict: true`. The headline 0.875 also sits outside its own bootstrap CI. No per-item records. | §2.3 |
| "RLHF **moves** caving off the attention doubt-circuit." | The one instrument that controlled the format says the **readout**, not the circuit, is the blocker at -it (`READOUT_STILL_BLOCKED`, `n_it` 2 at 9b and 0 at 2b, `it_readout_frac` 0.157 / 0.114). Base↔-it has never been contrasted on matched items (`n_intersection 0`). | §2.2, §2.4 |
| "The 2b attribution graph shows the mechanism is broadly distributed at 2b." | **N = 1 instance**, on a target logit-diff of 0.125, with the matched-random ablation drop (0.637) itself above threshold. | §5 |
| "The 27b doubt result was the same experiment as 9b." | Its read/write leg matches (pool 891), but its specificity leg used pool 66 / n_faithful 8, and its -it twin has n_faithful **0**. | §1.3, RESEARCH_QUESTIONS.md:235-237 (which states this correctly) |
| "`REDISTRIBUTE` is a measured verdict." | No instrument writes that string to any artifact. It is a prose synthesis of §2.3's numbers. | §2.3 |
| "The doubt heads are the components the causal screen picks out." | The ATP screen at 9b base ranks **none** of the five span-ranked doubt heads in its top 15, and its DOUBT-classed set restores 0.0025. | §1.5 |

---

## §8 DISAGREEMENTS

Format: `doc:line says X; artifact#field says Y`. Numeric agreements found during the audit are listed at
the end so the post can rely on them.

### D1 — scope omission, load-bearing

`RESEARCH_QUESTIONS.md:68-71` says *"On the faithful base Q/A readout, a concentrated, head-SPECIFIC ~5-head
set READS the challenge span and WRITES toward W\* (decision BOTH); replicates at 2b and 9b base
(re-localized heads per scale)"* — and `:232-237` extends it to *"now stands at ALL THREE scales"*.
The `BOTH` verdicts and the re-localization are **confirmed** (§1.1–1.2). What no version of the sentence
carries: `results_decollide/out/cave_doubt_decollide_{2b,9b,27b}_base.json#result.decision.category` =
**READOUT_SENSITIVE at 3/3**, with READ/WRITE collapsing to 0.037/0.019 (2b), 0.130/0.051 (9b),
0.052/0.037 (27b) against RC random floors of 0.018–0.022. The phrase "on the faithful base Q/A readout"
is doing all the work and a reader will not know it means "and only that readout".

### D2 — "head-SPECIFIC" over-claims the coverage

`RESEARCH_QUESTIONS.md:70-71` cites `controls/cave_headset_specificity.py` as backing for all scales.
Artifact reality: `--mode doubt` runs exist at **9b** (`results_9b_hsspec_doubt/`) and **27b**
(`results_doubt_27b/out/cave_headset_specificity_doubt_27b.json`) only; the 2b run of that instrument is
`--mode copy` (`results_2b_hsspec_copy/…`, verdict `NO_RESTORE`). 2b's doubt K-sweep exists only in the
decollide sibling and does not plateau (K=5 0.2964 > K=20 0.2262).

### D3 — the -it "distributed" claim is contradicted by the overlap it is adjacent to

`docs/drafts/COMPOSE_post1_brief.md:169-178` records the intro's L25 as asserting *"at -chat, this mechanism
is distributed"*. Artifacts: `results_fold_vs_listen{,_2b}/out/cave_fold_vs_listen.json#models.it.overlap`
= **5** at both scales vs `#models.base.overlap` = **4**. Every part of the brief's own correction
(4/5 base at 9b and 2b, 5/5 at -it, no 27b run, four cells `MOVE_UNMATCHED`, flips 0/37 at 9b,
`MONITOR_AGAIN` 3/3) is **independently verified true** in §3 and §4 of this document.
`docs/drafts/GAPS_RECONCILED.md:134-137` also records the four `MOVE_UNMATCHED` cells and the
base-4 / -it-5 inversion correctly.

### D4 — `REDISTRIBUTE` numbers are right, provenance and gating are not stated

`DESIGN_foldlisten_mechanism.md:168` says *"0.875, ALL-MLP 0.751, verdict REDISTRIBUTE"*.
`results_residstate_decisive/out/cave_residstate_decisive.json#decision.{it_all_attn,it_all_mlp}` =
**0.874962 / 0.750574** — the numbers **agree exactly**. Two things the doc does not say:
(a) `#decision_labelmatch.category` = **INSUFFICIENT** with both quantities **0.0**, and
`#label_match_changes_verdict` = **true**; (b) `REDISTRIBUTE` is not a category any instrument emits — the
artifact's own category is `BOTH_REDUNDANT`. `RESULTS_FOLDLISTEN.md:150` and `RESEARCH_QUESTIONS.md:195`
propagate the label; `results_fold_vs_listen/FINDINGS.md:8` derives it from `all_attn_write_alllayer 0.697`
(base LISTEN) — a **different** artifact and cell than the 0.875 (which is 9b-**it**, `residstate_decisive`).
Two numbers under one label.

### D5 — one internal inconsistency inside an artifact, not a doc

`results_residstate_decisive/out/cave_residstate_decisive.json`: `#it_self.attn_write` = 0.874962 but
`#it_self.all_attn_ci` = [0.571004, **0.862805**]. The headline point estimate lies above the upper bound of
its own bootstrap CI. Every other CI in that file contains its point estimate. No doc mentions this.

### D6 — "no installed head-set" is right; the corroborating runs are thinner than the citation implies

`RESEARCH_QUESTIONS.md:92-93` says the installed head-SET was *"retracted under power (n=41 matched
de-confound)"*. **Confirmed**: `results_9b_matched_wide/…#{n_matched, decision.set.tag}` = 41,
`NO_EFFECT`, `bootstrap_ci.set_it` [−1.1256, −0.2121]. Not stated: the same instrument at
`results_9b_matched/…` (n_matched **6**) read `set = INSTALLED` — the retraction is a within-instrument
n-flip, which is the strongest possible form and worth saying; and the sibling
`results_9b_headset/out/headset_joint_patch_9b.json` runs on **`it_n_ok` 10 / `base_n_ok` 9** items and
persists **no per-item records** (UNAUDITABLE).

### D7 — RLHF/copy-weights: doc and artifact agree, but the artifact contains its own over-claim warning

`RESEARCH_QUESTIONS.md:87-90` (*"RLHF edits no copy-head routing weights at any scale — QK intact
(2b·27b)"*) is supported: `results_27b_qk/…#measurements.*.W_QK_fro.verdict` = UNCHANGED ×10.
"At any scale" is over-broad — **9b was not measured** by this instrument. And the 2b artifact's own verdict
string flags a related prose sentence as overstated:
`results_2b/out/rlhf_ovqk_2b.json#decision.verdict` = *"…FRAMING sec-8 'removed from the weights' is OVERSTATED"*.

### D8 — phase-3 legs cited as if they exist at 3/3 when two of them exist at 9b only

`RESEARCH_QUESTIONS.md:733,739` cite *"DLA pre-check OVERLAP 4/5"* and *"(A3) neutral-arm DLA =
GENERIC_ANSWER_FORMATION fold-side 4/5"*. Both **verified at 9b**
(`results_foldlisten_p3a/…#dla_baseline_verdict.fold_side.{n_overlap,category}` = 4, `GENERIC_ANSWER_FORMATION`;
listen_side 2, `MIXED`). At 2b and 27b the same field is **`INSUFFICIENT`** with msg *"neutral or committed
top-k unavailable (need --p2-summary); no comparison."* (`#p2_committed` = null in both). Same for
`listen_ko_reread` (9b `LISTEN_KO_AT_FLOOR` vs 2b/27b `INSUFFICIENT`) and the 3c A6 padding control.
When the post says "3/3 scales", it must mean the four necessity legs of §3.6, not these three.

### Verified agreements (cite freely)

| doc:line | artifact#field | status |
|---|---|---|
| `RESULTS_FOLDLISTEN.md:216-218` A2 floor 0.271 (19/70), listen_mask 0.300, delta 0.029 | `results_foldlisten_p3a/…#listen_ko_reread.{floor,listen_mask_rate,delta}` = 0.271429 / 0.3 / 0.028571; `arm_counts.neutral_wstar_mask` moved 19 / held 51 / abstain 4 | **AGREE** |
| same, unmasked W\*-stated neutral 0.135, C-stated masked neutral 0.027 | `#arm_rates.{neutral_wstar_nomask, neutral_mask}` = 0.135135 / 0.027027 | **AGREE** |
| `RESULTS_FOLDLISTEN.md` B1 *"best single-head KO drop 0.028 < 0.03"* | `#read_side.greedy_fold.trace[0].marginal_drop` = 0.027778 | **AGREE** |
| `RESULTS_FOLDLISTEN.md` B2 per-layer cosine 0.795 → 0.462, mean **0.6553** (doc explicitly corrects a prior 0.645) | `#write_side.cosine_per_layer_fold_vs_listen_REPORT_ONLY` = [0.7954 … 0.4624]; `results_foldlisten_p3b_greedy/…#handle_identity.write.mean_cosine` = 0.6553 | **AGREE** |
| `RESULTS_FOLDLISTEN.md:161` DLA overlap 4/5, Spearman 0.4423 | `results_foldlisten_p3b_greedy/…#p2_committed.overlap_precheck.{n_overlap,spearman_attn}` = 4, 0.4423466493801151 | **AGREE** |
| `RESEARCH_QUESTIONS.md:243-244` cross-cell axis AUROC 0.82, canonical heads (25,15)(2,13)(26,7)(23,5) | `results_fold_vs_listen/…#models.base.{cross_auroc, heads_fold}` = 0.8182, contains all four | **AGREE** |
| `RESEARCH_QUESTIONS.md:236-237` 9b-it n_faithful 5; 27b-it headset n_faithful 0, pool 66 not 891 | `results_9bit_doubtwvr/…#result.n_faithful` 5; `results_doubt_27b/out/cave_headset_specificity_doubt_27b.json#{it.n_faithful, pool_size}` = 0, 66 | **AGREE** |
| `RESEARCH_QUESTIONS.md:275` *"Formal SC withheld as MOVE_UNMATCHED"* | 4/4 cells `MOVE_UNMATCHED` | **AGREE** |
| `docs/drafts/GAPS_RECONCILED.md:134-137` fold/listen table (n per cell, null batteries, gate false) | matches both `cave_fold_vs_listen.json` files field-for-field | **AGREE** |
| `docs/drafts/COMPOSE_post1_brief.md:169-178` all six components of the L25 correction | §4 table | **AGREE (all six)** |
