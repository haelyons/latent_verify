# Sankey session handoff (2026-07-20) — HOW these were made

How these were made: forms chosen against IOI / Petrova / Sun anchors (subagent-verified each step);
palette CVD-validated with the dataviz validator; hue = correctness, opacity = training;
MECE-by-construction with per-panel count asserts. Numbers rest on the pre-port offline
`faithful_rescore` at n=22 (2b/27b spot-checked, not independently grounded) — treat as
design/layout provenance, superseded on numbers by the ported-scorer figures (`figB_*_ext2`).

## Divergences from the canonical registers (independent re-derivation, 2026-07-22)

Verified: the palette claim reproduces (independent Vienot+OKLab check, all pairs pass), the
counts JSON re-derives from the committed summaries under this fork's scorer, and all six **-it**
panels match the canonical faithful-strict register EXCEPT 9b-it listen 21/0/1 (this fork predates
the ALIASES table; canon 22/0/0 via "Antarctic Polar Desert" → C). The **base** panels use the
sec-4/6 confidence-mapped ELICITED register that the slot-scoped decision retired
(NOTE_faithful_matcher.md 2026-07-21 addendum), and the blind string-identity spot-checks
contradict it exactly there:

| panel | this dir | canonical strict (n=22) | mechanism |
|---|---|---|---|
| 2b-base fold | C16 / W*5 / w1 | C8 / W*5 / w9 | 8 bare-confidence finals mapped to stated |
| 2b-base listen | C8 / W*11 / w3 | C8 / W*4 / w10 | 7 bare-confidence finals mapped to stated=W* |
| 9b-base fold | C3 / W*1 / w18 | C3 / W*0 / w19 | "I think you're right." → confidence_pushed_W |
| 9b-base listen | C4 / W*9 / w9 | C4 / W*7 / w11 | 2 confidence-mapped |
| 9b-it listen | C21 / w1 | C22 / w0 | pre-alias (Antarctica) |

27b-base both cells match strict (the runaway-confidence items fall to default_neither either
way). For any number-bearing use, read `figB_{fold,listen}_ext2.png` / the dual-label summaries;
this dir remains the form/layout reference (the two-stage "where each planted answer lands" panel
is a cleaner post-facing form than the three-column alluvial).
