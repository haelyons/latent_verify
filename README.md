# latent_verify

Causal verification of mechanistic-interpretability claims in the Gemma-2 family.
The one method: take a *correlational* description of something happening inside the
model and convert it into a *causal* claim by intervening on the proposed mechanism
and watching behaviour change — "cut the wire, see if the light goes out." Reading a
graph is never accepted as evidence.

## Start here
- **`ONBOARDING.md`** — a fresh session (local or cloud) starts here: the minimal init
  prompt + a discovery self-test. Prerequisite on clone: `git submodule update --init
  --recursive` (pulls the `latent_skeptic` triage harness).
- **`RESEARCH_QUESTIONS.md`** — the living steering doc: origin, current claims (as
  pointers into source, *not* restatements — read the source), open questions,
  terminology, and the handoff seed for the next agent.
- **`docs/lambda-gpu-access.md`** — the run path: GPU runs are driven over SSH from
  the workstation (not the web sandbox), via the Lambda Cloud API; budget cap + spend
  tally reconstructed from the audit log there; terminate when done.
- **`.claude/agents/latent_skeptic/`** (git submodule) — the adversarial triage harness that
  gates every load-bearing claim (H1: skeptics share no state; H2: a crux is verified by
  running, not reading; H3: grounding reproduces from the raw artifact). Its `triage-*`
  subagents auto-load from `.claude/agents/`; rules in its `HEURISTICS.md`.

## Where things live (the filesystem IS the index — there is no hand-kept list to rot)
- `controls/*.py`, `job_*.py` — the instruments. Each is self-describing (a `--selftest`
  and a neutral decision rule). To know what one does, read it.
- `results_*/`, `out/*.json` — committed ground truth, never summarized away. Each result
  JSON embeds its own `metric`, `thresholds`, and `decision_rule`; read the JSON to learn
  a result rather than trusting any prose summary of it.
- `FRAMING_NOTES.md` — the §-numbered spine of the framing/copy arc.
- `POSITION_SYCOPHANCY.md`, `POSITIONING.md` — where this sits in the literature.
- `DARWIN.md` — the "warm little pond" framing.
- `archive/` — historical, kept for provenance, off the hot path: the full chronological
  research log (`research_log.md`), dated snapshots, the Arc-1 PoC run path, executed
  pre-registrations, per-run launchers.

## Entry ritual (do not skip)
1. **Faithfulness gate.** Before building on any prior result, reproduce its committed
   numbers to bf16 rounding. A stale or drifted claim is caught here, not downstream.
2. **Triage.** Any new load-bearing claim goes through `latent_skeptic`; a crux is settled
   by running a control, not by reading a doc (including this one).
