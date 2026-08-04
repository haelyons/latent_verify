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
(blue/orange vs green/red) so the two measurements cannot be read as one; the pairing was validated
(all within-layer and cross-layer adjacent pairs ≥ ΔE·100 15 normal / 8 protan+deutan). `#E69F00`
fails 3:1 contrast on white, so every nonzero node carries a direct count label.

**Counts.** [placeholder — filled at first licensed draw from `out/forcedfinal_join.json`'s §9.3 state
vectors and §9.4 tables; the generation-layer elicited counts are asserted against the frozen
`make_figB_sankey.EXPECT` before a pixel is drawn]

**Quotation rules riding on this figure.** The headline is the -it triple at (fold, counter,
forced-final), quoted as a triple or not at all (§8.2). Base panels are SECONDARY, stamped
`CONTEXT_CONTAMINATED_MEASURED` (§6.4). Every listen-direction figure is `LISTEN_CONTINGENT_ON_H1`
(§1.2). Any 27b panel is quotable only with §10's four-part disclosure (instance id + start time; card
and driver class; source hardware unrecoverable; the measured cross-box lp spread).
