# fig_dist_sankey — mandated §9.4 contingency tables

Generated from `out/forcedfinal_join.json`, the only verdict source (§13). §9.4 requires the full 3x3
collapsed table AND the 5x4 unrolled table beside every `LAYERS_*` verdict: "no §9.4 verdict may be
stated without it". The figures print their verdict in each panel title, so these tables are the
companion that makes those panels quotable. Rows are the DISTRIBUTIONAL class (Rule S, first token),
columns the GENERATION class (faithful-strict). Counter arm only -- the arm the figures draw.

`disagree_frac` denominator is 82 with grey INCLUDED (§9.4). Bands: <=0.10 CONCORDANT / 0.10-0.30
PARTIAL / >0.30 DISCORDANT; integer cuts at n=82 are <=8 / 9-24 / >=25.

## FOLD direction, counter arm, -it half

### 2bit

- §9.4 verdict: **LAYERS_CONCORDANT** (faithful band LAYERS_CONCORDANT, commit band LAYERS_CONCORDANT)
- faithful: n_disagree 1 / 82, disagree_frac 0.0122
- commit: n_disagree 0 / 82, disagree_frac 0.0000
- §9.1 replay fidelity: REPLAY_FAITHFUL, sign flips 0, record mismatches 0
- §9.2 context: CTX_CONTAMINATED_ALL, n_ctx_clean 0 / 82

3x3 collapsed (faithful); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 14 | 0 | 0 |
| WSTAR | 0 | 67 | 0 |
| GREY | 0 | 1 | 0 |

5x4 unrolled (faithful); rows Rule-S state, cols gen label:

| state \ gen | C | WSTAR | NEITHER | UNRESOLVED_ALIAS |
|---|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 | 0 |
| GREY_NO_ONSET | 0 | 1 | 0 | 0 |
| GREY_TIED | 0 | 0 | 0 | 0 |
| FAVOURS_C | 14 | 0 | 0 | 0 |
| FAVOURS_WSTAR | 0 | 67 | 0 | 0 |

3x3 collapsed (commit); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 14 | 0 | 0 |
| WSTAR | 0 | 67 | 0 |
| GREY | 0 | 0 | 1 |

5x4 unrolled (commit); rows Rule-S state, cols gen label:

| state \ gen | correct | other | wrong |
|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 |
| GREY_NO_ONSET | 0 | 1 | 0 |
| GREY_TIED | 0 | 0 | 0 |
| FAVOURS_C | 14 | 0 | 0 |
| FAVOURS_WSTAR | 0 | 0 | 67 |

- §9.3 state vector, slot `single`: GREY_COLLISION 0, GREY_NO_ONSET 81, GREY_TIED 0, FAVOURS_C 0, FAVOURS_WSTAR 1 | collapsed C 0 / W* 1 / GREY 81 | band STATE_VARIANT_STABLE | onset frac 0.012 | ctx_clean subset n 0
- §9.3 state vector, slot `second_turn`: GREY_COLLISION 0, GREY_NO_ONSET 82, GREY_TIED 0, FAVOURS_C 0, FAVOURS_WSTAR 0 | collapsed C 0 / W* 0 / GREY 82 | band STATE_VARIANT_STABLE | onset frac 0.000 | ctx_clean subset n 0
- §9.3 state vector, slot `forced_final`: GREY_COLLISION 0, GREY_NO_ONSET 1, GREY_TIED 0, FAVOURS_C 14, FAVOURS_WSTAR 67 | collapsed C 14 / W* 67 / GREY 1 | band STATE_VARIANT_STABLE | onset frac 0.988 | ctx_clean subset n 0

### 9bit

- §9.4 verdict: **LAYERS_CONCORDANT** (faithful band LAYERS_CONCORDANT, commit band LAYERS_CONCORDANT)
- faithful: n_disagree 2 / 82, disagree_frac 0.0244
- commit: n_disagree 0 / 82, disagree_frac 0.0000
- §9.1 replay fidelity: REPLAY_FAITHFUL, sign flips 0, record mismatches 0
- §9.2 context: CTX_CONTAMINATED_ALL, n_ctx_clean 0 / 82

3x3 collapsed (faithful); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 27 | 0 | 0 |
| WSTAR | 0 | 53 | 0 |
| GREY | 0 | 2 | 0 |

5x4 unrolled (faithful); rows Rule-S state, cols gen label:

| state \ gen | C | WSTAR | NEITHER | UNRESOLVED_ALIAS |
|---|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 | 0 |
| GREY_NO_ONSET | 0 | 2 | 0 | 0 |
| GREY_TIED | 0 | 0 | 0 | 0 |
| FAVOURS_C | 27 | 0 | 0 | 0 |
| FAVOURS_WSTAR | 0 | 53 | 0 | 0 |

3x3 collapsed (commit); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 27 | 0 | 0 |
| WSTAR | 0 | 53 | 0 |
| GREY | 0 | 0 | 2 |

5x4 unrolled (commit); rows Rule-S state, cols gen label:

| state \ gen | correct | other | wrong |
|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 |
| GREY_NO_ONSET | 0 | 2 | 0 |
| GREY_TIED | 0 | 0 | 0 |
| FAVOURS_C | 27 | 0 | 0 |
| FAVOURS_WSTAR | 0 | 0 | 53 |

- §9.3 state vector, slot `single`: GREY_COLLISION 0, GREY_NO_ONSET 79, GREY_TIED 0, FAVOURS_C 2, FAVOURS_WSTAR 1 | collapsed C 2 / W* 1 / GREY 79 | band STATE_VARIANT_STABLE | onset frac 0.037 | ctx_clean subset n 0
- §9.3 state vector, slot `second_turn`: GREY_COLLISION 0, GREY_NO_ONSET 82, GREY_TIED 0, FAVOURS_C 0, FAVOURS_WSTAR 0 | collapsed C 0 / W* 0 / GREY 82 | band STATE_VARIANT_STABLE | onset frac 0.000 | ctx_clean subset n 0
- §9.3 state vector, slot `forced_final`: GREY_COLLISION 0, GREY_NO_ONSET 2, GREY_TIED 0, FAVOURS_C 27, FAVOURS_WSTAR 53 | collapsed C 27 / W* 53 / GREY 2 | band STATE_VARIANT_STABLE | onset frac 0.976 | ctx_clean subset n 0

### 27bit

- §9.4 verdict: **LAYERS_CONCORDANT** (faithful band LAYERS_CONCORDANT, commit band LAYERS_CONCORDANT)
- faithful: n_disagree 2 / 82, disagree_frac 0.0244
- commit: n_disagree 1 / 82, disagree_frac 0.0122
- §9.1 replay fidelity: REPLAY_FAITHFUL, sign flips 0, record mismatches 0
- §9.2 context: CTX_MIXED, n_ctx_clean 1 / 82

3x3 collapsed (faithful); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 25 | 0 | 0 |
| WSTAR | 0 | 54 | 0 |
| GREY | 1 | 1 | 1 |

5x4 unrolled (faithful); rows Rule-S state, cols gen label:

| state \ gen | C | WSTAR | NEITHER | UNRESOLVED_ALIAS |
|---|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 | 0 |
| GREY_NO_ONSET | 1 | 1 | 0 | 1 |
| GREY_TIED | 0 | 0 | 0 | 0 |
| FAVOURS_C | 25 | 0 | 0 | 0 |
| FAVOURS_WSTAR | 0 | 54 | 0 | 0 |

3x3 collapsed (commit); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 25 | 0 | 0 |
| WSTAR | 0 | 54 | 0 |
| GREY | 1 | 0 | 2 |

5x4 unrolled (commit); rows Rule-S state, cols gen label:

| state \ gen | correct | other | wrong |
|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 |
| GREY_NO_ONSET | 1 | 2 | 0 |
| GREY_TIED | 0 | 0 | 0 |
| FAVOURS_C | 25 | 0 | 0 |
| FAVOURS_WSTAR | 0 | 0 | 54 |

- §9.3 state vector, slot `single`: GREY_COLLISION 0, GREY_NO_ONSET 81, GREY_TIED 0, FAVOURS_C 0, FAVOURS_WSTAR 1 | collapsed C 0 / W* 1 / GREY 81 | band STATE_VARIANT_STABLE | onset frac 0.012 | ctx_clean subset n 0
- §9.3 state vector, slot `second_turn`: GREY_COLLISION 0, GREY_NO_ONSET 82, GREY_TIED 0, FAVOURS_C 0, FAVOURS_WSTAR 0 | collapsed C 0 / W* 0 / GREY 82 | band STATE_VARIANT_STABLE | onset frac 0.000 | ctx_clean subset n 0
- §9.3 state vector, slot `forced_final`: GREY_COLLISION 0, GREY_NO_ONSET 3, GREY_TIED 0, FAVOURS_C 25, FAVOURS_WSTAR 54 | collapsed C 25 / W* 54 / GREY 3 | band STATE_VARIANT_STABLE | onset frac 0.963 | ctx_clean subset n 1

## LISTEN direction, counter arm, -it half

Every number in this block is `LISTEN_CONTINGENT_ON_H1` (§1.2).

### 2bit

- §9.4 verdict: **LAYERS_CONCORDANT** (faithful band LAYERS_CONCORDANT, commit band LAYERS_CONCORDANT)
- faithful: n_disagree 2 / 82, disagree_frac 0.0244
- commit: n_disagree 1 / 82, disagree_frac 0.0122
- §9.1 replay fidelity: REPLAY_FAITHFUL, sign flips 0, record mismatches 0
- §9.2 context: CTX_CONTAMINATED_ALL, n_ctx_clean 0 / 82

3x3 collapsed (faithful); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 79 | 0 | 0 |
| WSTAR | 1 | 1 | 0 |
| GREY | 1 | 0 | 0 |

5x4 unrolled (faithful); rows Rule-S state, cols gen label:

| state \ gen | C | WSTAR | NEITHER | UNRESOLVED_ALIAS |
|---|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 | 0 |
| GREY_NO_ONSET | 1 | 0 | 0 | 0 |
| GREY_TIED | 0 | 0 | 0 | 0 |
| FAVOURS_C | 79 | 0 | 0 | 0 |
| FAVOURS_WSTAR | 1 | 1 | 0 | 0 |

3x3 collapsed (commit); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 79 | 0 | 0 |
| WSTAR | 0 | 2 | 0 |
| GREY | 1 | 0 | 0 |

5x4 unrolled (commit); rows Rule-S state, cols gen label:

| state \ gen | correct | wrong |
|---|---|---|
| GREY_COLLISION | 0 | 0 |
| GREY_NO_ONSET | 1 | 0 |
| GREY_TIED | 0 | 0 |
| FAVOURS_C | 79 | 0 |
| FAVOURS_WSTAR | 0 | 2 |

- §9.3 state vector, slot `single`: GREY_COLLISION 0, GREY_NO_ONSET 81, GREY_TIED 0, FAVOURS_C 0, FAVOURS_WSTAR 1 | collapsed C 0 / W* 1 / GREY 81 | band STATE_VARIANT_STABLE | onset frac 0.012 | ctx_clean subset n 0
- §9.3 state vector, slot `second_turn`: GREY_COLLISION 0, GREY_NO_ONSET 82, GREY_TIED 0, FAVOURS_C 0, FAVOURS_WSTAR 0 | collapsed C 0 / W* 0 / GREY 82 | band STATE_VARIANT_STABLE | onset frac 0.000 | ctx_clean subset n 0
- §9.3 state vector, slot `forced_final`: GREY_COLLISION 0, GREY_NO_ONSET 1, GREY_TIED 0, FAVOURS_C 79, FAVOURS_WSTAR 2 | collapsed C 79 / W* 2 / GREY 1 | band STATE_VARIANT_STABLE | onset frac 0.988 | ctx_clean subset n 0

### 9bit

- §9.4 verdict: **LAYERS_CONCORDANT** (faithful band LAYERS_CONCORDANT, commit band LAYERS_CONCORDANT)
- faithful: n_disagree 2 / 82, disagree_frac 0.0244
- commit: n_disagree 1 / 82, disagree_frac 0.0122
- §9.1 replay fidelity: REPLAY_FAITHFUL, sign flips 0, record mismatches 0
- §9.2 context: CTX_CONTAMINATED_ALL, n_ctx_clean 0 / 82

3x3 collapsed (faithful); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 80 | 0 | 0 |
| WSTAR | 1 | 0 | 0 |
| GREY | 1 | 0 | 0 |

5x4 unrolled (faithful); rows Rule-S state, cols gen label:

| state \ gen | C | WSTAR | NEITHER | UNRESOLVED_ALIAS |
|---|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 | 0 |
| GREY_NO_ONSET | 1 | 0 | 0 | 0 |
| GREY_TIED | 0 | 0 | 0 | 0 |
| FAVOURS_C | 80 | 0 | 0 | 0 |
| FAVOURS_WSTAR | 1 | 0 | 0 | 0 |

3x3 collapsed (commit); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 80 | 0 | 0 |
| WSTAR | 0 | 1 | 0 |
| GREY | 1 | 0 | 0 |

5x4 unrolled (commit); rows Rule-S state, cols gen label:

| state \ gen | correct | wrong |
|---|---|---|
| GREY_COLLISION | 0 | 0 |
| GREY_NO_ONSET | 1 | 0 |
| GREY_TIED | 0 | 0 |
| FAVOURS_C | 80 | 0 |
| FAVOURS_WSTAR | 0 | 1 |

- §9.3 state vector, slot `single`: GREY_COLLISION 0, GREY_NO_ONSET 79, GREY_TIED 0, FAVOURS_C 2, FAVOURS_WSTAR 1 | collapsed C 2 / W* 1 / GREY 79 | band STATE_VARIANT_STABLE | onset frac 0.037 | ctx_clean subset n 0
- §9.3 state vector, slot `second_turn`: GREY_COLLISION 0, GREY_NO_ONSET 82, GREY_TIED 0, FAVOURS_C 0, FAVOURS_WSTAR 0 | collapsed C 0 / W* 0 / GREY 82 | band STATE_VARIANT_STABLE | onset frac 0.000 | ctx_clean subset n 0
- §9.3 state vector, slot `forced_final`: GREY_COLLISION 0, GREY_NO_ONSET 1, GREY_TIED 0, FAVOURS_C 80, FAVOURS_WSTAR 1 | collapsed C 80 / W* 1 / GREY 1 | band STATE_VARIANT_STABLE | onset frac 0.988 | ctx_clean subset n 0

### 27bit

- §9.4 verdict: **LAYERS_CONCORDANT** (faithful band LAYERS_CONCORDANT, commit band LAYERS_CONCORDANT)
- faithful: n_disagree 2 / 82, disagree_frac 0.0244
- commit: n_disagree 1 / 82, disagree_frac 0.0122
- §9.1 replay fidelity: REPLAY_FAITHFUL, sign flips 0, record mismatches 0
- §9.2 context: CTX_CONTAMINATED_ALL, n_ctx_clean 0 / 82

3x3 collapsed (faithful); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 80 | 0 | 0 |
| WSTAR | 1 | 0 | 0 |
| GREY | 1 | 0 | 0 |

5x4 unrolled (faithful); rows Rule-S state, cols gen label:

| state \ gen | C | WSTAR | NEITHER | UNRESOLVED_ALIAS |
|---|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 | 0 |
| GREY_NO_ONSET | 1 | 0 | 0 | 0 |
| GREY_TIED | 0 | 0 | 0 | 0 |
| FAVOURS_C | 80 | 0 | 0 | 0 |
| FAVOURS_WSTAR | 1 | 0 | 0 | 0 |

3x3 collapsed (commit); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 80 | 0 | 0 |
| WSTAR | 0 | 1 | 0 |
| GREY | 1 | 0 | 0 |

5x4 unrolled (commit); rows Rule-S state, cols gen label:

| state \ gen | correct | wrong |
|---|---|---|
| GREY_COLLISION | 0 | 0 |
| GREY_NO_ONSET | 1 | 0 |
| GREY_TIED | 0 | 0 |
| FAVOURS_C | 80 | 0 |
| FAVOURS_WSTAR | 0 | 1 |

- §9.3 state vector, slot `single`: GREY_COLLISION 0, GREY_NO_ONSET 81, GREY_TIED 0, FAVOURS_C 0, FAVOURS_WSTAR 1 | collapsed C 0 / W* 1 / GREY 81 | band STATE_VARIANT_STABLE | onset frac 0.012 | ctx_clean subset n 0
- §9.3 state vector, slot `second_turn`: GREY_COLLISION 0, GREY_NO_ONSET 82, GREY_TIED 0, FAVOURS_C 0, FAVOURS_WSTAR 0 | collapsed C 0 / W* 0 / GREY 82 | band STATE_VARIANT_STABLE | onset frac 0.000 | ctx_clean subset n 0
- §9.3 state vector, slot `forced_final`: GREY_COLLISION 0, GREY_NO_ONSET 1, GREY_TIED 0, FAVOURS_C 80, FAVOURS_WSTAR 1 | collapsed C 80 / W* 1 / GREY 1 | band STATE_VARIANT_STABLE | onset frac 0.988 | ctx_clean subset n 0

## FOLD direction, counter arm, base half

Every number in this block is SECONDARY and stamped `CONTEXT_CONTAMINATED_MEASURED` (§6.4).

### 2bbase

- §9.4 verdict: **LAYERS_CONCORDANT** (faithful band LAYERS_CONCORDANT, commit band LAYERS_CONCORDANT)
- faithful: n_disagree 3 / 82, disagree_frac 0.0366
- commit: n_disagree 4 / 82, disagree_frac 0.0488
- §9.1 replay fidelity: REPLAY_FAITHFUL, sign flips 0, record mismatches 0
- §9.2 context: CTX_CONTAMINATED_ALL, n_ctx_clean 0 / 82

3x3 collapsed (faithful); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 15 | 0 | 0 |
| WSTAR | 0 | 14 | 1 |
| GREY | 0 | 2 | 50 |

5x4 unrolled (faithful); rows Rule-S state, cols gen label:

| state \ gen | C | WSTAR | NEITHER | UNRESOLVED_ALIAS |
|---|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 | 0 |
| GREY_NO_ONSET | 0 | 0 | 46 | 4 |
| GREY_TIED | 0 | 2 | 0 | 0 |
| FAVOURS_C | 15 | 0 | 0 | 0 |
| FAVOURS_WSTAR | 0 | 14 | 0 | 1 |

3x3 collapsed (commit); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 15 | 0 | 0 |
| WSTAR | 0 | 15 | 0 |
| GREY | 1 | 3 | 48 |

5x4 unrolled (commit); rows Rule-S state, cols gen label:

| state \ gen | correct | other | wrong |
|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 |
| GREY_NO_ONSET | 1 | 48 | 1 |
| GREY_TIED | 0 | 0 | 2 |
| FAVOURS_C | 15 | 0 | 0 |
| FAVOURS_WSTAR | 0 | 0 | 15 |

- §9.3 state vector, slot `single`: GREY_COLLISION 0, GREY_NO_ONSET 8, GREY_TIED 0, FAVOURS_C 53, FAVOURS_WSTAR 21 | collapsed C 53 / W* 21 / GREY 8 | band STATE_VARIANT_STABLE | onset frac 0.902 | ctx_clean subset n 0
- §9.3 state vector, slot `second_turn`: GREY_COLLISION 0, GREY_NO_ONSET 82, GREY_TIED 0, FAVOURS_C 0, FAVOURS_WSTAR 0 | collapsed C 0 / W* 0 / GREY 82 | band STATE_VARIANT_STABLE | onset frac 0.000 | ctx_clean subset n 0
- §9.3 state vector, slot `forced_final`: GREY_COLLISION 0, GREY_NO_ONSET 50, GREY_TIED 2, FAVOURS_C 15, FAVOURS_WSTAR 15 | collapsed C 15 / W* 15 / GREY 52 | band STATE_VARIANT_STABLE | onset frac 0.390 | ctx_clean subset n 0

### 9bbase

- §9.4 verdict: **LAYERS_CONCORDANT** (faithful band LAYERS_CONCORDANT, commit band LAYERS_CONCORDANT)
- faithful: n_disagree 1 / 82, disagree_frac 0.0122
- commit: n_disagree 3 / 82, disagree_frac 0.0366
- §9.1 replay fidelity: REPLAY_FAITHFUL, sign flips 0, record mismatches 0
- §9.2 context: CTX_CONTAMINATED_ALL, n_ctx_clean 0 / 82

3x3 collapsed (faithful); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 40 | 0 | 0 |
| WSTAR | 0 | 3 | 0 |
| GREY | 1 | 0 | 38 |

5x4 unrolled (faithful); rows Rule-S state, cols gen label:

| state \ gen | C | WSTAR | NEITHER | UNRESOLVED_ALIAS |
|---|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 | 0 |
| GREY_NO_ONSET | 1 | 0 | 37 | 1 |
| GREY_TIED | 0 | 0 | 0 | 0 |
| FAVOURS_C | 40 | 0 | 0 | 0 |
| FAVOURS_WSTAR | 0 | 3 | 0 | 0 |

3x3 collapsed (commit); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 40 | 0 | 0 |
| WSTAR | 0 | 3 | 0 |
| GREY | 2 | 1 | 36 |

5x4 unrolled (commit); rows Rule-S state, cols gen label:

| state \ gen | correct | other | wrong |
|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 |
| GREY_NO_ONSET | 2 | 36 | 1 |
| GREY_TIED | 0 | 0 | 0 |
| FAVOURS_C | 40 | 0 | 0 |
| FAVOURS_WSTAR | 0 | 0 | 3 |

- §9.3 state vector, slot `single`: GREY_COLLISION 0, GREY_NO_ONSET 6, GREY_TIED 1, FAVOURS_C 65, FAVOURS_WSTAR 10 | collapsed C 65 / W* 10 / GREY 7 | band STATE_VARIANT_STABLE | onset frac 0.927 | ctx_clean subset n 0
- §9.3 state vector, slot `second_turn`: GREY_COLLISION 0, GREY_NO_ONSET 82, GREY_TIED 0, FAVOURS_C 0, FAVOURS_WSTAR 0 | collapsed C 0 / W* 0 / GREY 82 | band STATE_VARIANT_STABLE | onset frac 0.000 | ctx_clean subset n 0
- §9.3 state vector, slot `forced_final`: GREY_COLLISION 0, GREY_NO_ONSET 39, GREY_TIED 0, FAVOURS_C 40, FAVOURS_WSTAR 3 | collapsed C 40 / W* 3 / GREY 39 | band STATE_VARIANT_STABLE | onset frac 0.524 | ctx_clean subset n 0

### 27bbase

- §9.4 verdict: **LAYERS_CONCORDANT** (faithful band LAYERS_CONCORDANT, commit band LAYERS_CONCORDANT)
- faithful: n_disagree 3 / 82, disagree_frac 0.0366
- commit: n_disagree 6 / 82, disagree_frac 0.0732
- §9.1 replay fidelity: REPLAY_FAITHFUL, sign flips 0, record mismatches 0
- §9.2 context: CTX_CONTAMINATED_ALL, n_ctx_clean 0 / 82

3x3 collapsed (faithful); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 40 | 0 | 0 |
| WSTAR | 0 | 5 | 0 |
| GREY | 1 | 2 | 34 |

5x4 unrolled (faithful); rows Rule-S state, cols gen label:

| state \ gen | C | WSTAR | NEITHER | UNRESOLVED_ALIAS |
|---|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 | 0 |
| GREY_NO_ONSET | 0 | 1 | 22 | 12 |
| GREY_TIED | 1 | 1 | 0 | 0 |
| FAVOURS_C | 40 | 0 | 0 | 0 |
| FAVOURS_WSTAR | 0 | 5 | 0 | 0 |

3x3 collapsed (commit); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 40 | 0 | 0 |
| WSTAR | 0 | 5 | 0 |
| GREY | 4 | 2 | 31 |

5x4 unrolled (commit); rows Rule-S state, cols gen label:

| state \ gen | correct | other | wrong |
|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 |
| GREY_NO_ONSET | 3 | 31 | 1 |
| GREY_TIED | 1 | 0 | 1 |
| FAVOURS_C | 40 | 0 | 0 |
| FAVOURS_WSTAR | 0 | 0 | 5 |

- §9.3 state vector, slot `single`: GREY_COLLISION 0, GREY_NO_ONSET 3, GREY_TIED 0, FAVOURS_C 70, FAVOURS_WSTAR 9 | collapsed C 70 / W* 9 / GREY 3 | band STATE_VARIANT_STABLE | onset frac 0.963 | ctx_clean subset n 0
- §9.3 state vector, slot `second_turn`: GREY_COLLISION 0, GREY_NO_ONSET 82, GREY_TIED 0, FAVOURS_C 0, FAVOURS_WSTAR 0 | collapsed C 0 / W* 0 / GREY 82 | band STATE_VARIANT_STABLE | onset frac 0.000 | ctx_clean subset n 0
- §9.3 state vector, slot `forced_final`: GREY_COLLISION 0, GREY_NO_ONSET 35, GREY_TIED 2, FAVOURS_C 40, FAVOURS_WSTAR 5 | collapsed C 40 / W* 5 / GREY 37 | band STATE_VARIANT_STABLE | onset frac 0.573 | ctx_clean subset n 0

## LISTEN direction, counter arm, base half

Every number in this block is SECONDARY and stamped `CONTEXT_CONTAMINATED_MEASURED` (§6.4).

Every number in this block is `LISTEN_CONTINGENT_ON_H1` (§1.2).

### 2bbase

- §9.4 verdict: **LAYERS_CONCORDANT** (faithful band LAYERS_CONCORDANT, commit band LAYERS_CONCORDANT)
- faithful: n_disagree 6 / 82, disagree_frac 0.0732
- commit: n_disagree 6 / 82, disagree_frac 0.0732
- §9.1 replay fidelity: REPLAY_FAITHFUL, sign flips 0, record mismatches 0
- §9.2 context: CTX_CONTAMINATED_ALL, n_ctx_clean 0 / 82

3x3 collapsed (faithful); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 22 | 0 | 0 |
| WSTAR | 1 | 7 | 0 |
| GREY | 2 | 3 | 47 |

5x4 unrolled (faithful); rows Rule-S state, cols gen label:

| state \ gen | C | WSTAR | NEITHER | UNRESOLVED_ALIAS |
|---|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 | 0 |
| GREY_NO_ONSET | 1 | 3 | 44 | 3 |
| GREY_TIED | 1 | 0 | 0 | 0 |
| FAVOURS_C | 22 | 0 | 0 | 0 |
| FAVOURS_WSTAR | 1 | 7 | 0 | 0 |

3x3 collapsed (commit); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 22 | 0 | 0 |
| WSTAR | 0 | 8 | 0 |
| GREY | 2 | 4 | 46 |

5x4 unrolled (commit); rows Rule-S state, cols gen label:

| state \ gen | correct | other | wrong |
|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 |
| GREY_NO_ONSET | 1 | 46 | 4 |
| GREY_TIED | 1 | 0 | 0 |
| FAVOURS_C | 22 | 0 | 0 |
| FAVOURS_WSTAR | 0 | 0 | 8 |

- §9.3 state vector, slot `single`: GREY_COLLISION 0, GREY_NO_ONSET 8, GREY_TIED 0, FAVOURS_C 53, FAVOURS_WSTAR 21 | collapsed C 53 / W* 21 / GREY 8 | band STATE_VARIANT_STABLE | onset frac 0.902 | ctx_clean subset n 0
- §9.3 state vector, slot `second_turn`: GREY_COLLISION 0, GREY_NO_ONSET 82, GREY_TIED 0, FAVOURS_C 0, FAVOURS_WSTAR 0 | collapsed C 0 / W* 0 / GREY 82 | band STATE_VARIANT_STABLE | onset frac 0.000 | ctx_clean subset n 0
- §9.3 state vector, slot `forced_final`: GREY_COLLISION 0, GREY_NO_ONSET 51, GREY_TIED 1, FAVOURS_C 22, FAVOURS_WSTAR 8 | collapsed C 22 / W* 8 / GREY 52 | band STATE_VARIANT_STABLE | onset frac 0.378 | ctx_clean subset n 0

### 9bbase

- §9.4 verdict: **LAYERS_CONCORDANT** (faithful band LAYERS_CONCORDANT, commit band LAYERS_CONCORDANT)
- faithful: n_disagree 2 / 82, disagree_frac 0.0244
- commit: n_disagree 3 / 82, disagree_frac 0.0366
- §9.1 replay fidelity: REPLAY_FAITHFUL, sign flips 0, record mismatches 0
- §9.2 context: CTX_CONTAMINATED_ALL, n_ctx_clean 0 / 82

3x3 collapsed (faithful); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 9 | 0 | 0 |
| WSTAR | 0 | 34 | 0 |
| GREY | 2 | 0 | 37 |

5x4 unrolled (faithful); rows Rule-S state, cols gen label:

| state \ gen | C | WSTAR | NEITHER | UNRESOLVED_ALIAS |
|---|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 | 0 |
| GREY_NO_ONSET | 2 | 0 | 34 | 3 |
| GREY_TIED | 0 | 0 | 0 | 0 |
| FAVOURS_C | 9 | 0 | 0 | 0 |
| FAVOURS_WSTAR | 0 | 34 | 0 | 0 |

3x3 collapsed (commit); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 9 | 0 | 0 |
| WSTAR | 0 | 34 | 0 |
| GREY | 2 | 1 | 36 |

5x4 unrolled (commit); rows Rule-S state, cols gen label:

| state \ gen | correct | other | wrong |
|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 |
| GREY_NO_ONSET | 2 | 36 | 1 |
| GREY_TIED | 0 | 0 | 0 |
| FAVOURS_C | 9 | 0 | 0 |
| FAVOURS_WSTAR | 0 | 0 | 34 |

- §9.3 state vector, slot `single`: GREY_COLLISION 0, GREY_NO_ONSET 6, GREY_TIED 1, FAVOURS_C 65, FAVOURS_WSTAR 10 | collapsed C 65 / W* 10 / GREY 7 | band STATE_VARIANT_STABLE | onset frac 0.927 | ctx_clean subset n 0
- §9.3 state vector, slot `second_turn`: GREY_COLLISION 0, GREY_NO_ONSET 82, GREY_TIED 0, FAVOURS_C 0, FAVOURS_WSTAR 0 | collapsed C 0 / W* 0 / GREY 82 | band STATE_VARIANT_STABLE | onset frac 0.000 | ctx_clean subset n 0
- §9.3 state vector, slot `forced_final`: GREY_COLLISION 0, GREY_NO_ONSET 39, GREY_TIED 0, FAVOURS_C 9, FAVOURS_WSTAR 34 | collapsed C 9 / W* 34 / GREY 39 | band STATE_VARIANT_STABLE | onset frac 0.524 | ctx_clean subset n 0

### 27bbase

- §9.4 verdict: **LAYERS_CONCORDANT** (faithful band LAYERS_CONCORDANT, commit band LAYERS_CONCORDANT)
- faithful: n_disagree 1 / 82, disagree_frac 0.0122
- commit: n_disagree 4 / 82, disagree_frac 0.0488
- §9.1 replay fidelity: REPLAY_FAITHFUL, sign flips 0, record mismatches 0
- §9.2 context: CTX_CONTAMINATED_ALL, n_ctx_clean 0 / 82

3x3 collapsed (faithful); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 15 | 0 | 0 |
| WSTAR | 0 | 31 | 0 |
| GREY | 1 | 0 | 35 |

5x4 unrolled (faithful); rows Rule-S state, cols gen label:

| state \ gen | C | WSTAR | NEITHER | UNRESOLVED_ALIAS |
|---|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 | 0 |
| GREY_NO_ONSET | 1 | 0 | 20 | 15 |
| GREY_TIED | 0 | 0 | 0 | 0 |
| FAVOURS_C | 15 | 0 | 0 | 0 |
| FAVOURS_WSTAR | 0 | 31 | 0 | 0 |

3x3 collapsed (commit); rows dist, cols gen:

| dist \ gen | C | WSTAR | GREY |
|---|---|---|---|
| C | 15 | 0 | 0 |
| WSTAR | 0 | 31 | 0 |
| GREY | 1 | 3 | 32 |

5x4 unrolled (commit); rows Rule-S state, cols gen label:

| state \ gen | correct | other | wrong |
|---|---|---|---|
| GREY_COLLISION | 0 | 0 | 0 |
| GREY_NO_ONSET | 1 | 32 | 3 |
| GREY_TIED | 0 | 0 | 0 |
| FAVOURS_C | 15 | 0 | 0 |
| FAVOURS_WSTAR | 0 | 0 | 31 |

- §9.3 state vector, slot `single`: GREY_COLLISION 0, GREY_NO_ONSET 3, GREY_TIED 0, FAVOURS_C 70, FAVOURS_WSTAR 9 | collapsed C 70 / W* 9 / GREY 3 | band STATE_VARIANT_STABLE | onset frac 0.963 | ctx_clean subset n 0
- §9.3 state vector, slot `second_turn`: GREY_COLLISION 0, GREY_NO_ONSET 82, GREY_TIED 0, FAVOURS_C 0, FAVOURS_WSTAR 0 | collapsed C 0 / W* 0 / GREY 82 | band STATE_VARIANT_STABLE | onset frac 0.000 | ctx_clean subset n 0
- §9.3 state vector, slot `forced_final`: GREY_COLLISION 0, GREY_NO_ONSET 36, GREY_TIED 0, FAVOURS_C 15, FAVOURS_WSTAR 31 | collapsed C 15 / W* 31 / GREY 36 | band STATE_VARIANT_STABLE | onset frac 0.561 | ctx_clean subset n 0

