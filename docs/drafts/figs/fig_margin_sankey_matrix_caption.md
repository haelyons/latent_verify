# fig_margin_sankey_matrix.png — caption

**What the whole-string margin favours.** The same 82 pairs and the same 4×3 grid as Figure 1, but
each stack is the sign of the teacher-forced whole-answer-string margin
$M = \log P(C) - \log P(W^*)$ at slot `forced_final`. Blue favours $C$, orange favours $W^*$, grey is
the repo's existing near-margin zone $|M| < 1.5$ nats (`MARGIN_KEEP` /
`make_fig_margin_estimation.TORN`). Opacity encodes training, as in Figure 1.

## Why two stacks and no ribbon

Each panel is the neutral control turn beside the counter challenge. Those are alternative second user
turns from the same planted turn, not successive states, so **no ribbon connects them** (registration
section 2). The pairing is real and is what the estimation plot bootstraps; it is never drawn as a
flow. This is the discrete sibling of `fig_margin_estimation.png`, not a replacement for it — banding
throws away magnitude, which is why the estimation plot exists.

## How it differs from Figure 2 (`fig2_dist_matrix`)

Figure 2 bands the **first-token** Rule-S read across three stages of one chain. This figure bands the
**whole-string** margin at one slot, across two arms. Stage-2 greyness in Figure 2 (reply never opens
with an answer token) has no counterpart here: a whole-string margin is defined at `forced_final`
regardless of the first token.

## Stamps on the figure

- **counter torn n/82** — how many counter-arm items sit in $|M| < 1.5$.
- **vs Fig 1 flips** — items where the counter-arm margin names $C$ and the spoken elicited label names
  $W^*$, or the reverse. Torn-vs-named disagreements are a different kind of split and are not in this
  count.
- **-base** rows: `CONTEXT_CONTAMINATED_MEASURED`. **Listen** rows: `LISTEN_CONTINGENT_ON_H1`.
- No -base-vs-chat contrast is computed.

## Provenance

- Drawn by `docs/drafts/figs/make_fig_margin_sankey_grid.py --source forced_final`.
- Data: `out/forcedfinal_dist_ff_ext2_{2b,9b,27b}{base,it}.json` items[].`r_lp` at
  `slot_id == forced_final`, `register == lp_whole_string`.
- Threshold: `controls/family_cave_diagnose.MARGIN_KEEP = 1.5`, shared with the estimation plot.
- Sibling diagnose-slot figure (fold only, bare→after ribbons): `fig_margin_sankey_diagnose_fold.png`.
