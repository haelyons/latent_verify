# Figure status and what each needs — 2026-07-28

Provenance and register evidence is in `GROUNDING_neutral_elicit.md`, `TAXONOMY_withholding.md` and
`NOTE_27b_repro_fail.md`. This file is only the action list; it does not restate their findings.

Currency was checked two ways: byte-identity vault↔repo, and whether the render postdates both matcher
revisions (sec-5.6b tie-break, then the plural fix `2c5a8bf`, 2026-07-26). The fig rebuild `a61f247` is
a descendant of `2c5a8bf`, so anything from it is post-fix.

## No change needed

| embed | referenced | repo render | why current |
|---|---|---|---|
| `figB_synthesis_strict_ext2.png` | intro (Fig 1), notes §Sycophancy Scaling Laws | same name | byte-identical, `a61f247` post-fix |
| `IMG_3917.png` | notes Fig 1 | `figB_neutral_counterfactual_ext2.png` | byte-identical, post-fix |
| `IMG_3919.png` | notes Fig 3 | `figB_fold_strict_allscales.png` | byte-identical, post-fix |
| `IMG_3918.png` | notes Fig 2 | `fig_margin_flow_9b.png` | byte-identical; log-prob margins, matcher-independent |

## Needs rebuilding

**Figure 4, the listen sankey** — `Pasted image 20260724190541.png`, referenced at notes Figure 4 (and
carried into `NOTE_B_post1_notes.md`). Two independent problems: it matches **no** repo render, so it is
not reproducible from the tree; and the nearest equivalent, `figB_listen_ext2.png`, was built `21ac405`
(2026-07-22), which **predates both matcher revisions**. Build script exists
(`figs/make_figB_sankey.py`), so the fix is a rebuild plus a decision about whether Figure 4 should be
that render or the crop currently in the vault. Note this does not contradict the earlier finding that
its bands reproduce — a figure can show correct bands and still not be rebuildable, which is the worse
of the two faults.

## Needs creating — the session's main result has no figure

**Nothing draws the neutral-elicited column.** `grep` for `neutral_elicit|nelicit` across all ten
`figs/make_*.py` returns nothing. So the result that inverts the withholding story is carried only by the
appendix table in `NOTE_B_post1_notes.md`. In a document that leads with sankeys, that is the figure most
obviously owed: the same fold/listen flow, with a no-push arm beside the pushed arm, at the elicited slot.
Data is committed and complete (`results_foldlisten_nelicit_{2b9b,27b}/out/`, all five new fields on
1,012/1,012 records), so this is a plotting job with no GPU dependency.

## Instrument fix, not a figure

`figs/make_figB_matrix.py` asserts on the **elicited** column only, so its counter column can move
silently on the next matcher change. That is the mechanism by which a stale number reaches a figure
unnoticed; the assert should cover every column it draws.

---

# The two runs requested before logging off, and where they actually are

**Both were registered. Neither was run.** The elicit-context run was requested explicitly ("design,
registration, and run") and is not done; that deviation is called out here rather than buried.

**1. Elicit-context fix — `DESIGN_elicit_context.md`. REGISTERED, NOT RUN, and blocked on a decision.**
Its §12 D-1 requires choosing the truncation rule before launch (registered cut-only vs verbatim
`isolate_span`, which also strips markdown, perturbs -it contexts and would forfeit the free -it null).
Launching before D-1 is answered runs the wrong arm. D-9 also asks whether to pay for the base cells
twice given the neutral-elicit round already landed. Cost $22–29; primary measure and bands frozen.

**2. Distributional analysis across scales and variants — `DESIGN_distributional_withholding.md`.
REGISTERED, NOT RUN, and not yet buildable.** The instrument cannot express the listen arm:
`controls/family_cave_diagnose.py:215` builds only the counter push. So the code change is registered but
unwritten, and the diagnose artifacts remain **9b only, fold only** (both base and -it). Everything
distributional therefore still rests on one cell of one scale in one direction — the position
`TAXONOMY_withholding.md` reports and this round was designed to fix.

## What including distributional results in the post would take

In order, because each step gates the next:

1. **Write the listen-arm extension** to `family_cave_diagnose.py` per the registered spec, claim-blind,
   with a model-free selftest and a review. Offline, free.
2. **Run it** — 2b, 9b, 27b × base, -it × fold, listen. ~$4–8 marginal if co-scheduled onto the
   elicit-context boxes, $5–10 standalone. Rebase the 27b pace first: 27b-base took **4.9 h** against the
   design's ~1.5 h SXM5 estimate, so both designs' cap arithmetic is optimistic.
3. **Plot the pooled `WITHHELD` group, not the per-category split.** This is the constraint that decides
   what the figure can say: the registered power table makes UNC n=0 at 2b and n=1 at 27b, and the
   motivating 9b cell n=20 is BANDED — per-cell only. A per-category sankey across scales would be
   drawing categories that cannot carry a claim. The pooled group is the only comparison powered at every
   base cell.
4. **Expect a scale-dependent verdict and design the figure for it.** `SCALE_DEPENDENT` is a live
   registered outcome, in which case the design forbids any cross-scale sentence — so the figure must
   read per-scale rather than implying a single trend.

Sequencing note: run the elicit-context round **first**. If it returns MATERIAL, the distributional
groups are recomputed under the span labels and the margin artifacts do **not** need re-running — which
saves most of a GPU round. Running them the other way round wastes that.
