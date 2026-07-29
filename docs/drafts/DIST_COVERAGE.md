# THE DISTRIBUTIONAL GRID — coverage, measured

Written because the distributional program was being filled opportunistically with no statement of the
grid, so "how much is left" had no answer. Counts here are **computed from the artifacts on disk**, not
maintained by hand — the scan is ~25 lines of `json.load` over `results_*/out/*.json` + `out/*.json`,
keyed on each artifact's own stamped `name`. Re-run it rather than trusting this table's age.

**31 of 72** (6 instruments × 6 cells × 2 families) present. **4** carry a listen arm.

## EXT82 (`verifier_family_ext2.json`, 82 items) — the family the figures are built on

| instrument | 2b | 2b‑it | 9b | 9b‑it | 27b | 27b‑it |
|---|---|---|---|---|---|---|
| `family_cave_diagnose` | f | f | f | f | f | f |
| `family_cave_diagnose_arms` | **F+L** | **F+L** | **F+L** | **F+L** | running | running |
| `family_topk_shift` | f | f | f | f | f | f |
| `family_generate_judge` | f | f | f | f | running | running |
| `verify_graph_poc` (T3) | f | f | f | f | running | running |
| `modelw_candidates` | . | K4 | f | K4 | . | K4 |

`f` = fold only · `F+L` = both arms · `.` = absent · `K4` = code-blocked (no `--chat`)

## VF22 (the legacy 22-item family) — 5 of 12, all 9b

| instrument | 2b | 2b‑it | 9b | 9b‑it | 27b | 27b‑it |
|---|---|---|---|---|---|---|
| `family_cave_diagnose` | . | . | f | f | . | . |
| `family_cave_diagnose_arms` | . | . | . | . | . | . |
| `family_topk_shift` | . | . | f | . | . | . |
| `family_generate_judge` | . | . | f | . | . | . |
| `verify_graph_poc` | . | . | f | . | . | . |
| `modelw_candidates` | . | . | f | . | . | . |

## What remains, in dependency order

| # | gap | cells | blocked by |
|---|---|---|---|
| 1 | **`family_topk_shift` has no listen arm anywhere** | 6 | K1 again. The identical re-parameterisation on `family_cave_diagnose` is **already gate-validated** (all 23 pre-existing fields identical, 4/4 cells → `out/b1_fold_identity_gate.json`), so the pattern is proven and this is the cheapest large win left |
| 2 | `family_cave_diagnose_arms` at 27b | 2 | in flight |
| 3 | `family_generate_judge`, `verify_graph_poc` at 27b | 4 | in flight |
| 4 | `modelw_candidates` at 2b‑base and 27b‑base | 2 | nothing — run-only, never launched |
| 5 | `modelw_candidates` at all three `-it` cells | 3 | **K4** — `:420-425` has no `--chat`; a multi-line prompt-builder refactor |
| 6 | **No T3 forced-final distributional readout exists at all** | every cell | **B2, unregistered.** All three blind audits converged on this independently, and it is the slot the verdicts are decided on |
| 7 | VF22 outside 9b | 7 | nothing — run-only. Lowest value: no current claim is written at VF22 breadth |
| 8 | base‑vs‑`-it` rank comparison is **uninterpretable** | — | **C1, unregistered.** The `-it` cells build that slot with the chat template and base with QA, so the 3/3/4 vs 781/2375/3077 rank gap is format-confounded |

## Two things this grid corrected about itself

1. **`verify_graph_poc`'s T_PRE gate is a family property, not a cell property.** Its own docstring says
   it "runs with NO torch". So it needs **one run per family**, not six per family, and it ran offline
   for **$0** → `out/verify_graph_poc_vfam_ext2_TPRE.json` (VALID, n=82, `collision_frac` 0.000, 82/82
   entity answers). An earlier version of this table over-counted that row by five cells. Only the T3
   half needs a model.
2. **T_PRE and T3 disagree, and the disagreement is the finding.** T_PRE VALID at the family level while
   **T3 = INSUFFICIENT at 2b‑base, 2b‑it, 9b‑base and 9b‑it**. A family can be perfectly able to express
   a readout swap while no model produces enough of the behaviour to test one. The instrument reports
   the two gates separately with no rollup, which is why the distinction survived; a pooled verdict
   would have hidden it.

## What a completed grid would and would not license

It would give every probability-level statement a scale axis and a direction axis, which is what
`GAPS_B` meant by "the single widest readout gap".

It would **not** license the join everyone will want — that the probability movement *explains* the
generation-level fold/listen adoption. That is a cross-readout join on the same items, it is not
registered, and it is named here so it is not smuggled in as an interpretation of whichever cell lands
last.
