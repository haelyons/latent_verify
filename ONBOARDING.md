# Onboarding — the init prompt for a new session on this repo

Paste the block below to seed a fresh session (local or cloud). It is deliberately minimal:
the repo self-describes, so the prompt only points at the discovery path and asserts the
verification standards. The one prerequisite is the submodule (the `latent_skeptic` triage
harness); the self-test catches a half-equipped session before any work is done.

> **Prerequisite (shell, before the session relies on subagents):**
> `git submodule update --init --recursive`
> This populates `.claude/agents/latent_skeptic/` (the triage harness). If your session
> started before this ran, the `triage-*` subagents won't be loaded — run it, then restart.

---

## Init prompt (copy verbatim)

```
Orient yourself in this repo and continue the live research thread.

0. Self-test (fail loud, don't proceed half-equipped). Confirm the substrate is wired:
   your available SUBAGENTS include triage-author / triage-reader / triage-runner (from the
   latent_skeptic submodule) and cavecrew-investigator / builder / reviewer; your available
   SKILLS include caveman and karpathy-guidelines. If any are missing, STOP and report —
   run `git submodule update --init --recursive` and restart; the discovery is broken.

1. Orient via the entry ritual (README.md): the filesystem IS the index — read the source
   JSONs to learn a result, not prose summaries. Faithfulness-gate any committed number
   before building on it. Pick up current state from RESEARCH_QUESTIONS.md (handoff seed +
   current claims/frontier); read a claim's cited source before extending it.

2. Standards (load-bearing):
   - Route every load-bearing claim or instrument through latent_skeptic. Its rules live in
     .claude/agents/latent_skeptic/HEURISTICS.md (H1: skeptics share no state; H2: a crux is
     settled by running, not reading; H3: grounding reproduces the numbers from the raw
     artifact). Read it when triaging.
   - Delegate to subagents (triage-*, cavecrew-*) to preserve context, usage, and independence.
   - Commit findings as ground truth, cite-don't-restate: reference the result JSON / source,
     never carry numbers across the codebase (they rot). Data-retention contract: see .gitignore
     (result JSONs + logs commit by default; raw *captures*.npz are regenerable, kept local).
   - Conform to /caveman and /karpathy-guidelines.
```

---

## What the self-test verifies (and why it's first)

A clone is only useful if it's fully equipped. The blanket-ignore era silently dropped result
data, the triage subagents, and the skills — invisible until something failed downstream. So the
first thing a session does is assert its tools exist:

- **Subagents** load from `.claude/agents/**/*.md` (recursive; identity = `name:` frontmatter).
  `triage-*` come from the `latent_skeptic` **submodule** at `.claude/agents/latent_skeptic/agents/`;
  `cavecrew-*` are vendored flat. (Verified: a fresh `claude` session discovers agents at the
  2-level submodule path; a submodule that wasn't `--init`'d leaves them missing.)
- **Skills** load from `.claude/skills/<name>/SKILL.md`. `caveman` and `karpathy-guidelines` are
  vendored (they originate from global plugins a cloud clone won't have). Available, not auto-on —
  invoked by this prompt.

If the self-test passes, the session has data + subagents + skills + the triage harness, and can
discover state → explore → formulate → draft controls → commit to standard, entirely from the clone.
