# Handoff prompt — stateless vetting pass

Paste as the opening prompt of a fresh session. It deliberately states no conclusions: the next agent
finds the thread by the entry ritual and verifies it themselves.

---

Orient yourself in this repo. The live thread is post drafting. For wider context read
`RESEARCH_QUESTIONS.md` and the entry ritual in `README.md`. Delegate extensively to subagents to
preserve usage, context, and independence. Conform to `/karpathy-guidelines`, and work MECE — prose
never restates what a plot, table, or attached result already carries.

Your goal is to vet the current write-up end to end. Read the researcher's drafts in Obsidian
(`interp/DARWIN.md_post1_user_intro.md`, `interp/DARWIN.md_post1_user_notes.md`, via MCP — the vault is
gold, never write there), then `docs/drafts/NOTE_B_post1_notes.md` and every patch, review, grounding
and taxonomy document beside it. Then check all of it yourself against ground truth: model inputs and
outputs, actual probability distributions, the committed result JSONs, and the pre-registrations in
`DESIGN_*.md`.

Trust nothing you read. Prior sessions' notes, the grounding documents, and the drafts have each
contained errors — a regex that silently misclassified, numbers true only in an unnamed register, a
claim contradicted by the very figure cited for it. Re-derive rather than confirm.

Report what fails to reproduce, what is over-scoped, and what a reviewer would attack first. Flag
missing facts honestly rather than filling them.
