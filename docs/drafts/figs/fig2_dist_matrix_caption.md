# fig2_dist_matrix_counter.png — full caption

**Figure 2: What the distribution favours.** The same 82 correct/plausibly-incorrect fact pairs and
the same 4x3 grid as Figure 1 — rows are (fold | listen) x (-base | -chat), columns are 2b / 9b / 27b
— but each ribbon is coloured by what the model's **next-token distribution** favours at that point,
not by what it says out loud. Blue favours the correct fact $C$, orange favours the plausibly
incorrect $W^*$, grey means the next token is not an answer token at all. Opacity encodes training,
as in Figure 1: muted is -base, bold is -chat.

## What a "state" is here

A first-token Rule-S read at the canonical key: the model's next-token distribution is compared over
the frozen variant sets for $C$ and $W^*$, and the item scores `FAVOURS_C`, `FAVOURS_WSTAR`, or one
of three grey states. It is **never "the probability of $C$"** — it is which of the two answers the
next token leans toward, if either. The per-panel line above each panel unrolls the grey band into
its three components (`no-onset` / `tied` / `collision`) at each of the three stages, because merging
them hides a distinction.

## The two places this figure does not line up with Figure 1

**The first column is measured before the plant.** Figure 1's first node is the planted answer — a
given, 82/82 one colour by construction. This figure's first node is slot `single`, the plain
question with no plant and no second turn, which is why it is 53/21/8 at 2b-base rather than a solid
block, and why it is *identical* between the fold and listen rows of the same cell: it is one shared
measurement, not two. Calling it "planted" would assert something the measurement does not contain.

The -base/-chat difference in that column is a **format effect, not a knowledge difference**: -base
reads a raw `Q: … A:` template, so its next token *is* the answer, while -chat opens in prose (modal
first token "The") and therefore scores grey on 79–81 of 82 items. Nothing here licenses "-base has a
prediction and -chat does not."

**The middle column is grey in every panel** — `GREY_NO_ONSET` 82/82 at every cell, both directions.
This is a measurement, not a rendering artifact: a reply to a challenge never *opens* with an answer
token. The modal first token is "You" at every -chat cell and a polarity word at -base (" Yes" 62 at
2b-fold, " No" 56 at 9b-fold, " Yes" 73 at 27b-fold). The collapse-to-grey and fan-out is a property
of a first-token readout, not of the model. Stages 2 and 3 do correspond to Figure 1's "counter
reply" and "elicited", and carry Figure 1's own labels.

## "vs Fig 1"

Top-right of each panel, in the slot Figure 1 uses for `drift n/82`: the number of items where this
readout's class at `elicited` differs from the same item's faithful-strict generation label. It is
the §9.4 `n_disagree`, read from `out/forcedfinal_join.json`, and it is why a panel here reads 67
where Figure 1 reads 68.

| | 2b | 9b | 27b |
|---|---|---|---|
| fold -base | 3 | 1 | 3 |
| fold -chat | 1 | 2 | 2 |
| listen -base | 6 | 2 | 1 |
| listen -chat | 2 | 2 | 2 |

All twelve fall in the `LAYERS_CONCORDANT` band (≤ 8 of 82). **The band itself is deliberately not
printed on the figure** (§9.3: "No band and no verdict attaches to a state count"); the bands and
their mandated 3x3 and 5x4 contingency tables live in `out/forcedfinal_join.json` and
`fig_dist_sankey_tables.md`.

That concordance is also this figure's main limitation, and it should be stated rather than hidden:
because the two readouts agree on 76–81 of 82 items, **the first-token distributional matrix largely
reproduces Figure 1**. `LAYERS_CONCORDANT` is a statement that two readouts agree, never a statement
that either one is informative. At -base the agreement is substantially agreement about *silence* —
dist `GREY_NO_ONSET` 50/39/35 of 82 against gen `NEITHER` 51/38/34 at 2b/9b/27b fold.

## Scope and stamps

- **Counter arm only.** The neutral arm is not drawn, so the figure is not exhaustive over arms;
  stage 1 is nonetheless common to both.
- **-base rows are stamped `CONTEXT_CONTAMINATED_MEASURED`** (§6.4) and are SECONDARY. The registered
  primary axis is the -chat half at (fold, counter, `forced_final`).
- **Listen rows are `LISTEN_CONTINGENT_ON_H1`** (§1.2) — measured and emitted, quotation gated until
  `OWED.md` H1 is decided. Nothing here restores a withdrawn number.
- **No -base-vs-chat contrast is computed** (§6.5 forbids one at this slot and requires the instrument
  to refuse). Adjacency in a grid is not a contrast.
- Ribbons exist only within one (cell, direction, arm) chain (§2, §5.2, §9.5).
- 27b panels are quotable only with §10's four-part disclosure.

## Provenance

- Drawn by `docs/drafts/figs/make_fig_dist_grid.py` (`--selftest` passes, 28 asserts).
- Data: `out/forcedfinal_dist_ff_ext2_{2b,9b,27b}{base,it}.json`; verdicts and gate:
  `out/forcedfinal_join.json` (the only verdict source, §13).
- Registration: `docs/drafts/REGISTRATION_forcedfinal_distributional.md`.
- Palette is the registered `bcc7aa0` blue/orange/grey, shared with the four two-layer PNGs. An
  earlier teal/magenta revision is withdrawn: its grey sat dE 5.5 from Figure 1's withhold grey, and
  the two greys mean different things. All three classes are now held dE >= 8 from their Figure 1
  counterpart under normal, protan and deutan vision.
- Supersedes `fig_dist_grid_counter.png`, which is stale and should be deleted.
