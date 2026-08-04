# fig_dist_sankey — caption (full form; the figure carries only a one-line pointer here)

**What it shows.** The forced-final DISTRIBUTIONAL readout of
`REGISTRATION_forcedfinal_distributional.md`, one figure per (direction, arm, variant half), scales as
panels. States are FIRST-TOKEN Rule-S reads at the canonical Rule-K key (§4.2) — never "the probability
of C". Stages are the three slots of §5 within one chain: `single` → `second turn` → `forced final`.
Transitions are within-layer, within-direction, within-arm only (§2, §9.5): no neutral→counter,
fold→listen, base-vs-it, or cross-layer ribbon exists in any mode.

**Two registered forms, gated by `out/forcedfinal_join.json` (the only verdict source):**
- `LAYERS_CONCORDANT` at all three scales → the two-layer alluvial: the committed GENERATION layer
  (faithful-strict labels, planted → elicited final; the Figure-1 colours `#009E73`/`#CC3311`/`#b0b0ab`)
  stacked above the DISTRIBUTIONAL layer (`#0072B2` favours C / `#E69F00` favours W* / `#8a8a92` grey),
  each layer labelled, transition counts per-layer.
- any `LAYERS_PARTIAL`/`LAYERS_DISCORDANT` (all scales evaluable) → the distributional-only fallback
  (`_distonly`), generation layer not drawn.
- any `LAYERS_UNEVALUABLE` or missing verdict → no figure is drawn.

**Colour note.** The distributional layer is deliberately off-palette from the generation sankey
(blue/orange vs green/red) so the two measurements cannot be read as one. **Validated WITHIN-LAYER
only** — every within-layer adjacent pair clears ΔE·100 ≥ 15 normal / ≥ 8 protan+deutan, which is what
`_check_palette` asserts and what its docstring says ("Layers are separate, labelled bands, so
cross-layer pairs are not gated"). **An earlier form of this note claimed the cross-layer pairs were
validated too; they are not, and two of the nine fail.** Recomputed 2026-08-05 with the script's own
Vienot+OKLab helpers: gen `C` `#009E73` vs dist `GREY` `#8a8a92` is **13.7 normal / 6.7 protan / 4.2
deutan**, i.e. under deuteranopia the correct-answer green and the distributional grey are nearly the
same colour; gen `NEITHER` `#b0b0ab` vs dist `GREY` `#8a8a92` is **12.1** at all three. Both pairs sit
in the same six-entry legend in two-layer mode. The mitigation is real but is labelling, not hue: the
two layers are drawn on separate labelled axes, never joined by a ribbon (§2, §9.5), and `#E69F00`
fails 3:1 contrast on white so **every** nonzero node carries a direct count label unconditionally.
Read the counts, not the colours, when the two greys are adjacent.

**Counts.** Full mandated tables: **`docs/drafts/figs/fig_dist_sankey_tables.md`** — the §9.4 3×3
collapsed AND 5×4 unrolled tables for all twelve drawn axes (6 cells × 2 directions, counter arm), plus
each axis's §9.1 fidelity verdict, §9.2 context verdict and §9.3 state vectors at all three slots. §9.4
forbids stating a `LAYERS_*` verdict without those tables and the figures print the verdict in every
panel title, so that file is what makes these panels quotable; `make_fig_dist_sankey.py` writes no
sidecar of its own. Headline, from `out/forcedfinal_join.json` (2026-08-05 run, `git_commit` `9fc06d1`):
faithful-layer `n_disagree` out of 82 is **1 / 2 / 2** at 2b-it / 9b-it / 27b-it fold and **2 / 2 / 2**
listen; base is **3 / 1 / 3** fold and **6 / 2 / 1** listen. Every cell lands `LAYERS_CONCORDANT` on both
label families, so no cell is `LAYER_AGREEMENT_CONTESTED` and no panel title states a band the join did
not. All six cells are `REPLAY_FAITHFUL` with 0 sign flips and 0 record mismatches. The generation-layer
elicited counts were asserted against the frozen `make_figB_sankey.EXPECT` before a pixel was drawn,
12/12 across both halves and both directions.

**What concordance does and does not mean here.** At `-it` the agreement is informative: the two layers
land on the same answer on 80/82 of the fold items, with the distributional layer resolving to a named
answer on 97.6% of items at the forced-final slot (9b-it). At **base** the same verdict is largely
agreement about *silence* — the distributional layer reads `GREY_NO_ONSET` on 50 / 39 / 35 of 82 at
2b / 9b / 27b fold while the generation layer reads `NEITHER` on 51 / 38 / 34, so the layers agree
mostly by both declining to name an answer. `LAYERS_CONCORDANT` is a statement that two readouts agree,
never a statement that either one is informative, and §4.5 bars widening the frozen `V(A)` set now that
the onset rate has been seen — so that grey stays as measured.

**Quotation rules riding on this figure.** The headline is the -it triple at (fold, counter,
forced-final), quoted as a triple or not at all (§8.2) — for this run that triple is
**`(LAYERS_CONCORDANT, LAYERS_CONCORDANT, LAYERS_CONCORDANT)` → `ROUND_CONCORDANT`**. Base panels are
SECONDARY, stamped `CONTEXT_CONTAMINATED_MEASURED` (§6.4); their round verdict is also
`ROUND_CONCORDANT` and the two halves are never pooled (§9.6). Every listen-direction figure is
`LISTEN_CONTINGENT_ON_H1` (§1.2) — measured and emitted, quotation gated until `OWED.md` H1 is decided,
and nothing here restores a withdrawn number.

**§10 four-part disclosure, discharged for the 27b panels of this run (2026-08-05):**
(i) **provenance pair** — `lambda_instance_id` `76e8c8c4609b4ad79d1f7ee44e61374a`, `started_utc`
`2026-08-04T23:27:57.621650+00:00` (27b-base) and `2026-08-04T23:37:21.177562+00:00` (27b-it); both
cells ran on that one box, `git_commit` `9fc06d1`.
(ii) **box class and cluster** — card `NVIDIA H100 80GB HBM3`, driver `570.148.08`. That is the same
card-and-driver class as the format-matched run, which §10 records as matching cluster 3; it is neither
cluster 1 (H100 PCIe @ 570.148.08) nor cluster 3 as §10 lists it (H100 80GB HBM3 @ 580.105.08). Per
`OWED.md` H3 the known-cluster table is a per-box-class object rather than a fixed list, so this is
stated as a class match to the format-matched box, **not** as cluster identity.
**The driver is quoted from the box's own `nvidia-smi` line in `results_ff_27b/out/run_detached.log`,
because the artifacts' `provenance.driver` field is `null`** — see §0.6 Round 2. The card comes from the
artifact.
(iii) **the source run's hardware is unrecoverable** (§3.4), so no 27b comparison here separates code
from hardware, and §1.1's same-box test is `SAME_BOX_UNVERIFIABLE` by construction against the source.
(iv) **measured cross-box lp spread** — median 0.009–0.13, max 0.44–0.59 nats, the figure §10 carries.
This run adds no new cross-box measurement: it is one draw on one box per half, so the spread is
inherited as a disclosure, not re-measured here.
A 27b number printed without all four is not quotable.
