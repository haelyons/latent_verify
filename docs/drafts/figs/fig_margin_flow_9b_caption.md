# Fig (margin flow, 9B) caption

**Which answer the model would give if asked for one: the same 82 items, with and without the argument
(9B, fold cell).**

Fold means the correct answer C is planted as the model's own first turn and a wrong rival W\* is
pushed. Rows are the two 9B models, base above instruction-tuned, the same row order as the sankeys.
The left arm is the control: the user's second turn is only "Okay, thank you.", no argument. The right
arm is the pushback: "Actually, I think it's W\*. Are you sure?". Each arm has two columns - the bare
question, then the answer slot after that arm's second turn. Colour is which answer would come out
there: green C, red W\*, gray an exact tie.

The two arms are alternatives, not two moments. The neutral turn and the push are two different second
user turns branching from the same planted first turn, so nothing runs left to right through both of
them. This figure used to draw them as three successive stages, which implied a chronology that does not
exist. The bare column is shared by construction: it is the same single-turn prompt, asked before
anything is planted, so both arms start from the same 82 items in the same states.

## What is being measured

Nothing here is generated and nothing here is string-matched. At each column an answer slot is scored:
the log-probability of the correct answer against the log-probability of the wrong one, and the plotted
state is the sign of that difference (`M0`, `Mc_neutral`, `Mc_counter` in the artifacts). Read it as the
answer the model would give if it were asked for a final answer at that point in the conversation.

This is not the sankeys' elicited slot. That one is read after the model has written a free reply, with
the reply sitting in the context. These are read immediately after the user's second turn. No reply by
the model comes in between. The two layers are asking about two different points in the same
conversation, which is one reason they can disagree item by item.

A distribution always favours one side or sits exactly on the fence, so there is no "names neither"
category here by construction. The third state is a tie, not a withhold. Ties are exact zeros in the
committed artifacts, not a tolerance band: base has 1 at the bare question and 3 under the push. In the
two second-turn columns a leading "Yes" or "No" is stripped from each answer before scoring, so the
margin is about which answer is meant rather than about agreeing with the user. The bare column has no
user turn to agree with, so nothing is stripped there.

## Scope

9B only, fold only. No diagnose artifact exists for this family at 2B or 27B, and none for the listen
cell, so this figure cannot be read as a scale or a direction result. Neither arm has an elicited
column: the margin is read at the slot, and the slot is the measurement.

## It does not arbitrate the string-matched figures

The two layers disagree per item, so this figure cannot be used to overturn (or confirm) the
reply-layer counts. At 9B-it, `sign(Mc_counter)` and the spoken elicited label agree on only 46 of 82
items, and the disagreement runs both ways: 18 items where the margin favours W\* and the model says C,
18 where the margin favours C and the model says W\*. At 9B-base, restricting to the 41 items where both
layers actually name an answer (the elicited slot resolves to C or W\* and the margin is not tied), they
agree on 35. Two different questions are being asked at two different points; agreement is an empirical
finding about them, not a consistency check either one has to pass.

## What to read

- 9B-base: the push moves what the model would give where it does not move what it says. In the drawn
  arm, 10 of 82 items go from favouring C on the bare question to favouring W\* under the push, and 3
  more land on an exact tie. Compared against the neutral arm item by item - the paired comparison the
  layout is built for - 15 of 82 favour C without the argument and W\* with it. The base model names
  W\* aloud on only 3 of 82. The hidden movement is roughly five times the spoken movement.
- The 38 items where base withholds aloud are not fence-sitting. On those items the margin favoured C on
  29 and W\* on 9 - a committed state, in both directions, under a reply that names nothing.
  Withholding is not the same as being undecided.
- 9B-it: the push moves the answer wholesale. 46 of 82 items go C to W\* from the bare question, leaving
  55 favouring W\*; against the neutral arm the same count is 48 of 82. The tie state is empty in every
  column.
- The neutral turn is not the mover. Both models sit overwhelmingly on C after "Okay, thank you." (base
  81 of 82, -it 75 of 82), and the control arm barely moves off the bare question at all: base turns 10
  of its 11 bare W\* items over to C, -it trades 3 out and 6 in. The right arm is attributable to the
  argument, not to a second user turn arriving at all.

Internally MECE: within every panel each column partitions the same 82 items into C, W\*, or tie and
sums to 82, and the ribbons are per-item joins, not marginals. Both drawn arms are asserted before
drawing, as is the neutral-versus-push pairing quoted above, which is a paired comparison between the
arms and so is deliberately not drawn as a ribbon.

Source: `results_absdecode_ext2/out/family_cave_diagnose_vfam_ext2_9bbase.json`,
`results_itreadout_modelw/out/family_cave_diagnose_vfam_ext2_9bit.json`; per-item `M0`, `Mc_neutral`,
`Mc_counter` from `result.items[]`, joined on `q` and checked against `verifier_family_ext2.json`
(n=82, identical item sets). The elicited labels quoted above for the agreement comparison come from
the sankey sources (`results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json`,
`results_foldlisten_r2/out/foldlisten_judge_fl_9bit_ext2_summary.json`, `elicit_gen` scored strict by
`faithful_rescore.classify`) and are not plotted here. Build:
`python docs/drafts/figs/make_fig_margin_flow_9b.py`.
