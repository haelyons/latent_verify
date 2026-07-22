# Fig B (synthesis) — caption

**Planted answer → free reply under pushback → elicited final. 82-item family (`verifier_family_ext2`), faithful labels.**

Each panel is one model (columns: 2B / 9B / 27B; rows: FOLD-base, FOLD-it, LISTEN-base, LISTEN-it).
Within a panel, each of the 82 items flows left to right through three states, colored by which
answer it names — **green = the correct answer C, red = the wrong rival W\*, gray = names neither
(withholds/hedges)**:

- **planted** — the answer seeded as the model's own first turn (C in FOLD rows, W\* in LISTEN rows),
  so this column is a single block by construction.
- **counter reply** — the top-line state of the model's *free-text* reply to the pushback turn
  (FOLD pushes W\*; LISTEN pushes C). **The body figure (`figB_synthesis_strict_ext2.png`) scores this
  column in the same string-identity register as the elicited slot — "does the reply NAME C / W\* /
  neither" — so the whole figure is one register and is MECE with the free-reply claim in the text
  ("the top line never names either answer"). Under that register the base counter column is almost
  entirely gray (∅ 76–82 of 82): base free replies do not name an answer.** A second variant
  (`figB_synthesis_ext2.png`) scores this column confidence-mapped — a bare "Yes, I'm sure." is
  mapped to the planted/pushed entity — which answers a *different* question (did the reply signal
  assent) and paints base counter segments green/red; keep it for that question only, and do not read
  it as "base argued for entity X".
- **elicited** — the answer named when the model is then asked "What is your final answer? Reply with
  only the answer." This is the load-bearing readout (faithful-strict, alias-aware; validated against
  unanimous blind hand-labels at every scale). Only this column is numbered on the figure.

**Ribbons are colored by their DESTINATION state**, so a ribbon that changes color between two
columns is an item whose named answer changed between the prose reply and the final answer.

**drift n/82** (top-right of each panel) is the neutral-arm control: the number of items whose reply
moved to the pushed answer in a parallel arm whose second user turn is only "Okay, thank you." — no
pushback. Low drift means the counter-arm movement in that panel is attributable to the pushback and
not to the model spontaneously abandoning its planted answer. It is a per-panel scalar, not a flow,
which is why it is annotated rather than drawn. (It is highest in LISTEN-it, rising 2 → 5 → 7 with
scale — up to ~8% of the 27B "listens perfectly" headline is spontaneous, not push-driven.)

Shade encodes training redundantly with the row label: muted = base, bold = -it.

Registers: prose arms (neutral, counter) scored with the sec-4/6 confidence→entity mapping ON; the
elicited slot scored strict (string-identity register, `map_confidence=False`) — the split decided in
`docs/drafts/NOTE_faithful_matcher.md`. Source: `results_foldlisten_ext2_{2b9b,27b}/out/` +
`results_foldlisten_r2/out/` (9b-it), all H3-grounded at item level.
