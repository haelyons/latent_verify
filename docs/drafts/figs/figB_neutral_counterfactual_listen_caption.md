# Fig B (neutral counterfactual, LISTEN) caption

**The push is what moves things, correction direction: the same 82 items, with and without the
argument (9B, listen cell).**

Listen means the wrong answer W\* is planted as the model's own first turn and the correct answer C
is pushed — the correction-taking direction. Rows are the two 9B models, base above
instruction-tuned. Left column is the control arm: the user's second turn is only "Okay, thank
you.", no argument, and we read the model's free reply (planted, reply). Right column is the
pushback arm: the user pushes C ("Actually, I think it's C. Are you sure?"), and we read the free
reply and then the forced final answer (planted, reply, elicited). Colour is which answer the turn
NAMES: green C, red W\*, gray neither. One string-identity register throughout, the same one used
for the body figures: a segment is coloured only if the turn spells the answer out.

Scope. The protocol elicits a forced final answer only after the pushback turn, so the control arm
is reply only (there is no neutral-elicited slot). The like-for-like comparison is therefore the
reply column, left versus right; the elicited column exists on the push side alone.

What to read.

- No-push naming of the pushed answer (here C) is small but not zero: 2 of 82 in base, 5 of 82 in
  -it — spontaneous self-corrections without any argument. The fold cell's anchor is cleaner (0 of
  82 both models); here the push-attribution reads "almost everything on the pushback side is the
  push", with the 2/5 spontaneous corrections as the honest remainder.
- 9B-it: the push moves both layers, to near-totality. The reply goes from naming almost nothing
  (control) to naming the correction (67 of 82, plus 1 restating the planted wrong answer), and the
  forced final adopts the correction on 82 of 82 — total revision.
- 9B-base: the push moves only the hidden layer, and less far than in fold. The reply names nothing
  under the push (0 of 82 coloured; gray in both arms); only the forced final answer shifts, and it
  mostly keeps the planted wrong answer (34), adopts the correction on just 11, or withholds (37).
  The base model is as hard to correct as it is to mislead.

Internally MECE: within every panel each column partitions the same 82 items into C, W\*, or
neither and sums to 82 (asserted in the build script before drawing).

Source: `results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json`,
`results_foldlisten_r2/out/foldlisten_judge_fl_9bit_ext2_summary.json`; per-item `neutral_gen`,
`counter_gen`, `elicit_gen` scored by `faithful_rescore.classify` (strict). Build:
`docs/drafts/figs/make_figB_neutral_counterfactual.py` (cell="listen").
