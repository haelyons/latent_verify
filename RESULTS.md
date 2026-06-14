# RESULTS — first real T0/T1 run (CPU, 2026-06-12)

Full run of `poc_minimal.py --stage all` on `google/gemma-2-2b` (bf16) +
GemmaScope per-layer transcoders, circuit-tracer @041a9b2, defaults
(multipliers {0, −2}, transport multiplier −2, recruit-frac 0.25,
min-recruited 3, transport-frac 0.5, n-controls 5, seed 0). Host: the
CPU-only container (4 cores, 15.6 GB RAM + 10 GB swap; see CPU_VALIDATION.md
and the `load_model` wrapper comments for the memory measures — all
bit-identical to the default load). Artifacts: `out/t0.json`, `out/t1.json`,
`out/run.log`.

## Sanity: harness faithfulness

All six pinned Texas features fire on the seed at position 9 (the " Dallas"
token) with activations matching the canonical Neuronpedia graph within bf16
tolerance (`reference/canonical_graph_texas.json`):
L19/7477 55.25 vs 55.78 · L16/25 28.12 vs 28.16 · L14/2268 25.62 vs 25.70 ·
L7/6861 15.44 vs 15.52 · L4/13154 15.12 vs 15.17 · L20/15589 fires at 9 and
peaks at 10 (49.25; canonical@9 45.66). Two features fire at one extra
position each (L20 at 10, L7 at 7), so the joint clamp places 9
interventions. Austin (id 22605) is base top-1 at logit 26.25.

## T0 — pre-registered criteria

| | m=0 (zero-ablation) | m=−2 (inhibition) |
|---|---|---|
| joint drop (rank after) | 1.50 (stays top-1) | **24.49 (rank 0 → 87 887)** |
| max single / joint (S2) | 0.75 / 1.50 = **0.50 ambiguous** | 8.50 / 24.49 = **0.35 ambiguous** |
| control mean (S3) | 0.05 → PASS (~30×) | 0.95 → PASS (~26×) |
| S1 | PASS (drop > 1.0) | PASS (rank flip, decisive) |

- **S1 PASS.** The pinned feature set is causally load-bearing; at −2 the
  joint clamp obliterates Austin. The feature list and transcoder set are
  right — the §6 "stop" row does not fire.
- **S2 ambiguous at both multipliers** — partial redundancy, formally
  "report as such". The structure under the ratio is informative:
  L20/15589 alone carries 8.5 of the 24.5 joint drop at m=−2; the other five
  singles are ≤ 1.75, and L19/7477's single inhibition reads *negative*
  (−0.375: removing it slightly helps Austin — compensation in the
  Hydra-effect direction). At m=0 the [Redesign] Step 6(ii) signature is
  visible in miniature: five of six single ablations read ≈ 0
  (|drop| ≤ 0.25) while the joint reads clearly positive (1.5).
  Single-feature ablation is not a safe adoption rule here, but one feature
  (the late L20 say-feature side) is individually potent — PIE-style "highly
  robust to single ablations" does NOT transfer wholesale to this PLT stack.
- **S3 PASS** with wide margin at both multipliers: the effect is specific
  to the Texas features, not to clamping any six magnitude-matched features
  in the same layers.

## T1 — paraphrase transport

**S4 PASS: 15/16 paraphrases preserve behaviour** (sole failure:
"Consider the state containing Dallas. Its capital is", margin −1.0).
The regime table is A-heavy to the maximum degree:

| structure | A | B | C |
|---|---|---|---|
| minimal | 5 | 0 | 0 |
| syntactic | 5 | 0 | 0 |
| reordered | 5 | 0 | 0 |

Every survivor recruits **all 6/6** seed features (recruit-frac 0.25), and
the joint-clamp effect transports at full strength: normalized drops
0.70–1.39 (8 of 15 above 1.0 — the clamp removes Austin from top-1 on every
survivor, ranks after 12 894–238 556). Regimes B and C are empty.

## Reading against §6

- **S1 fail?** No.
- **S2**: ambiguous band — neither the marginal-given-F justification nor
  its refutation is automatic; but the m=0 single-reads-≈0 signature plus
  the one-dominant-feature structure says the full loop needs
  marginal-given-F *and* per-feature response curves, not a single scalar
  rule.
- **S3 fail?** No.
- **T1 A-heavy** → "Claim transports; one-shot graph generalizes.
  Informative null; per [Handoff]: 'Both outcomes are informative'."
- The §4.4 probe-prompting prediction (some paraphrases simply won't
  recruit the seed features; expect regime C) did **not** materialize on
  this prompt family: recruitment was 6/6 on all 15 survivors, including
  every `reordered` paraphrase. Cross-prompt feature identity is not the
  binding constraint here.
- The shortcut-dominates row (seed joint drop ≈ 0) did not fire either.

**Net**: the gate opens toward the mediation-share estimator (the
B/C-pivot rows are moot on this evidence), with the S2 ambiguity logged as
the thing the full loop's adoption rule must respect.

## Caveats

- Single run, single seed, default thresholds; bf16 on CPU (canonical graph
  activations reproduced to ~1%, so numerics are sane).
- `drop_normalized` > 1 on 8/15 survivors just means several paraphrases
  are more clamp-sensitive than the seed; the A/B boundary (0.5) is nowhere
  near binding — the result is robust to the `transport-frac` choice.
- The one filtered paraphrase fails top-1 by a 1.0-logit margin; nothing
  was measured on it by design.
