# ORIGINS — where this project's method came from

Recorded because the seeding currently survives only as dangling internal references; this
anchors them. (Stub — if the upstream artifacts are recoverable, commit them here.)

## The seed
The project was seeded by a conversation with **Nora Petrova** about an attribution-graph /
circuit **"verifier"** given prompts — i.e. take a circuit hypothesis about a model on a prompt
and causally check it, rather than read it off a graph. The adaptation that became the core of
the best results: **paraphrases** — require a proposed mechanism to survive across a frozen
paraphrase family, not a single prompt (the "T1 transport" idiom; see `POSITIONING.md` §S4,
`FRAMING_NOTES` §3.8). This is the move that turns a one-prompt finding into a mechanism claim.

## Dangling provenance (artifacts NOT in-repo)
There was upstream material — referred to as **"the briefing"** and via the project-internal
handles **`[PIE]` / `[Handoff]` / `[Redesign]`** — that, among other things, **froze the original
`paraphrases.json`** (`CPU_VALIDATION.md`: "byte-identical to the briefing's frozen copy"). These
handles are deliberately not attributed to any external publication (`POSITIONING.md` line 8), and
the deep-research artifacts that kicked off the methodology are **not committed to this repo**.

So a future agent who hits `[PIE]` / `[Handoff]` / `[Redesign]` or "the briefing" should know:
those point at this seeding material, which lives outside the repo. If recovered, drop it (or a
pointer) here so the references resolve.
