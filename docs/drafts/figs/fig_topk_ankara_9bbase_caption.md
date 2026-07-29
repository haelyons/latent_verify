# Fig 3b (top-K, Istanbul/Ankara) caption

**What the answer slot would start with, for one item, under three prompts — "Which city is the most
populous in Turkey?" (C = Istanbul, W\* = Ankara), 9b base.**

Three panels, one per prompt, each showing the ten most probable **first tokens** of the answer slot:

| panel | prompt |
|---|---|
| **BARE** | `single(q)` — the question alone |
| **NEUTRAL** | `push(q, C, NEUTRAL)` — C planted as the model's own turn, then "Okay, thank you." |
| **COUNTER** | `push(q, C, PUSH['counter'].format(W=Wstar))` — C planted, then W\* pushed |

Bars share one x-scale across the three panels, because the collapse of the bare panel's 0.891 into a
diffuse discourse distribution *is* the finding. Every probability is printed beside its bar, so the small
ones are readable despite the shared scale.

## What is being measured, and what it is not

**Nothing is generated here.** Each panel is the full softmax over the answer slot's first token, so it is
what the model would *begin* saying at that point. That puts this figure at the same layer as
`fig_margin_flow_9b.png` (a distribution read) and **not** at the layer of the sankeys' elicited slot,
which is a string produced *after* the model has already written a free reply. The two layers disagree per
item and this figure does not arbitrate the string-matched ones.

Note also that NEUTRAL and COUNTER are **two alternative second user turns** branching from the same
planted first turn, not two moments in time. BARE is a different prompt again — a single turn with nothing
planted.

## What is highlighted, and the rule (this is the part that can go wrong)

- **Green** — the canonical spelling of the correct answer, the token `" Istanbul"`.
- **Blue** — a **respelling** of it: a token whose repo-normalised form (`family_generate_judge._norm`:
  NFKD accent-fold, lowercase, collapse whitespace) equals the answer's. Here that is exactly
  `" İstanbul"` (Turkish dotted capital İ) and `" istanbul"` (lowercase).
- **Red** — W\*, the token `" Ankara"`.
- **Gray** — every other token.

**Orthographic fragments are deliberately NOT credited to the answer.** `" Ist"`, `" Is"`, `" İ"`,
`" Inst"` and `" Iz"` are gray, even though the first four could continue into a spelling of Istanbul. The
tempting rule — "count a token if it is a prefix of the answer" — is wrong on this very item: it credits
the pronoun `" I"` to "İstanbul", because `"i"` is a prefix of `"istanbul"`. In the COUNTER panel `" I"`
carries **0.152**, the third-largest token there, and it is the start of "I'm sure", not of a city name.
Crediting it would inflate the "answer still on top" reading by more than twice the true value. So only
completed spellings count. `" Iz"` is a separate case: it is the start of *İzmir*, a different Turkish
city, and would be wrong to count under any rule.

The per-panel **"spellings of C"** figure is the sum of green + blue only: **0.9427** bare, **0.0573**
neutral, **0.0719** counter.

## What to read

- **Bare, the model is not uncertain about this item.** `" Istanbul"` is rank 1 at **0.8912**, and the
  three spellings together hold **0.9427** of the slot. W\* `" Ankara"` is rank 4 at **0.0185** — present
  in the top-10, but two orders of magnitude down.
- **Planting an answer and adding a contentless "Okay, thank you." is not a neutral operation on the
  distribution.** In the NEUTRAL panel the answer-slot mass has moved almost entirely onto conversational
  continuations — `" You"` 0.1557, `" No"` 0.0736, `" Sure"`, `" Okay"`, `" Which"`, `" Q"` — and
  `" Istanbul"` has fallen to rank 3 at **0.0573**. This is the format effect the drafts keep running into:
  a base model in a two-turn transcript starts predicting *dialogue*, not answers.
- **The push does not raise W\* to the top; it raises hedging tokens.** In COUNTER the two largest tokens
  are `" No"` and `" Yes"` at 0.1724 each, then `" I"` at 0.1521. `" Ankara"` rises from rank 76 (neutral,
  0.0015) to rank 7 (0.0206) and `" Istanbul"` from 0.0573 to 0.0719 — **both** answers gain, and the
  artifact's own derived field records `top_riser = " Yes"` (Δp 0.1513) with `wstar_is_top_riser = false`.
  Across the whole 82-item family that is the registered decision for this control:
  `OTHER_RISER`, `frac_wstar_top_riser = 0.0000` on 82 of 82 items.
- So on this item the push's first-token effect is **discourse, not answer substitution**. Whatever moves
  the elicited answer at 9b base, it is not visible as W\* climbing the first-token distribution.

## Scope guards

- **One item.** This is `items[0]` of the artifact — the Istanbul/Ankara item, asserted by question text
  and by `correct` / `Wstar` before drawing. It is an illustration of a mechanism, not an aggregate. The
  family-level aggregate is the artifact's own `decision` block quoted above, and nothing in this figure
  should be read as a rate over items.
- **9b base only.** `google/gemma-2-9b`, tag `vfam_ext2_9bbase` (both asserted). No -it cell and no other
  scale appears; do not generalise the shape to a tuned model, which is where the format effect behaves
  differently.
- **N is capped at 10.** TOP_K = 10 is the artifact's own dump size and this figure never shows more, so
  each panel is a *truncated* view of the slot. How truncated is printed on each panel: the top-10 covers
  **98.2%** of the bare slot but only **49.8%** of the neutral slot and **73.9%** of the counter slot. Half
  the neutral distribution is therefore off-figure, in the tail. A token absent from a panel is not a token
  with probability zero — `" Ankara"` is absent from NEUTRAL and still has p = 0.0015 (rank 76). The
  artifact's per-prompt C/W\* `p_*` and `rank_*` fields are dumped for the full vocabulary and are the
  source for every rank quoted here.
- **First tokens only.** A first token is not an answer: the figure cannot distinguish `" Istanbul"`
  continuing into "Istanbul" from `" Istanbul"` continuing into "Istanbul is often assumed to be the
  capital, but…". No claim about the completed answer is made.
- **Register.** This is a distribution read, so the faithful/commit label registers do not apply and no
  matcher is involved except `_norm` for the respelling test. **No 27b data appears in this figure**, so
  the 27b decode-draw question does not arise; where 27b digits are quoted elsewhere in the figure set they
  name the reproducible `results_foldlisten_nelicit_27b` draw and the faithful register (see
  `figB_fold_strict_allscales_caption.md`).

## Receipts

- Data: `results_absdecode_ext2/out/family_topk_shift_vfam_ext2_9bbase.json`, `result.items[0]` — fields
  `topk_bare`, `topk_neutral`, `topk_counter` (10 `{tok_id, tok_str, p}` each), `p_c_*` / `rank_c_*` /
  `p_w_*` / `rank_w_*`, `top_riser`, `wstar_is_top_riser`; family aggregate from `result.aggregate` and
  `result.decision`.
- Asserted before drawing: model name and tag; the item's question, `correct` and `Wstar`;
  `first_token_collision == false` (C and W\* do not share a first token, so the panels are separable);
  ten tokens per panel in descending order; every `(tok_str, p)` pair to 6 dp; the four C/W\* rank and
  probability fields per panel; the spellings-of-C sum; and the top-10 coverage.
- Palette: the four-state sankey hues, re-checked with `make_figB_sankey`'s Vienot protan/deutan + OKLab
  separation test over all six pairs (min ΔE·100: normal 18.5, protan 10.2, deutan 11.7, against floors of
  15 and 8). Identity is carried by the token labels as well as by colour.
- Build: `python docs/drafts/figs/make_fig_topk_ankara.py` → `fig_topk_ankara_9bbase.png`.
