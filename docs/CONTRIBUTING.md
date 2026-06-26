# Contributing / sharing this work — a snapshot

**Status: not authoritative. A working snapshot, to build out later.**

This file is not about *what we found* (see the experiment branches and their
notes for that). It is about the open question of **how and where work like
this could be contributed and put up for verification** — what venues and
tooling exist today, what each can and cannot host, and which paths fit the
kind of artifacts this repo produces. Treat every external claim below as
"believed true as of 2026-06, spot-check before relying on it."

## What this work is, structurally

So the mapping below makes sense — described by artifact *type*, not by result:

- Target model is **gemma-2-2b** (base), with **GemmaScope per-layer
  transcoders**, via **circuit-tracer** (pinned commit in `run_poc.sh`).
- The artifacts are of three shapes:
  1. causal tests on **transcoder (MLP-stream) features**;
  2. verification of a published **attribution graph**;
  3. **attention-head interventions** (QK-space) — knockout / necessity tests.

Shape (3) is the awkward one to share: attribution graphs freeze attention, so
the most frictionless sharing paths (below) do not natively capture a raw
attention-head causal claim. That tension is the main thing this doc records.

## Where work like this is shared

| Venue | Hosts | Notes / what to check |
|---|---|---|
| **Neuronpedia** (Decode Research) | transcoder/SAE feature dashboards; attribution graphs (generate + share by URL); attention via **HeadVis / LORSA** | gemma-2-2b features and graphs: confirmed. Raw attention heads: **unconfirmed** — HeadVis demos are Gemma 3 + Haiku 3.5, and the broad "many heads / many models" support is **LORSA units** (a sparse *decomposition*), not raw heads. |
| **Transformer Circuits Thread** | Anthropic circuits write-ups (incl. the methods + "Biology" papers, HeadVis) | Non-peer-reviewed flagship; origin of the attribution-graph method and of the example one line of this work verifies. |
| **Circuits Research Landscape** (collaboration) | replications & extensions of attribution-graph work | Explicitly invites community circuits; has a Slack + GitHub. The closest thing to a standing home for corroboration/replication. |
| **circuit-tracer** / EleutherAI **Attribute** (libraries) | the runnable substrate (Colab demos on gemma-2-2b) | The natural base for a self-contained, "run-it-yourself" verification artifact. |
| **LessWrong / AI Alignment Forum** | exploratory single-model write-ups | The standard "here is a claim + a notebook, please replicate or break it" genre. |
| **arXiv + BlackboxNLP / ML conferences** | peer-reviewed mechanistic claims | BlackboxNLP is the workshop home for this granularity (the circuit-tracer library paper landed there). |

## Candidate contribution paths

Sketches, not commitments. An "up-for-verification" contribution likely needs
**both** an observational layer (look without running) and a causal layer
(actually re-run the intervention):

- **Observational:** a Neuronpedia list deep-linking the relevant transcoder
  features and the attribution graph (embeddable via IFrame).
- **Causal:** a single self-contained, dependency-pinned **Colab notebook**
  (on circuit-tracer / TransformerLens) reproducing the interventions. The
  existing job scripts on the experiment branches are most of this already.
- **Distribution:** the Circuits Research Slack / Landscape, and/or a
  LessWrong–Alignment Forum post; BlackboxNLP if peer review is wanted.

## Open questions (the actual question)

- Can gemma-2-2b **raw attention heads** (or a LORSA unit standing in for one)
  be hosted/observed on Neuronpedia today, or must the head-level claim live
  entirely in a runnable notebook?
- What is the minimum reproducible artifact that lets a third party *verify* a
  causal head-knockout claim, not just *view* the supporting features?
- Which single venue to lead with — and whether to frame the contribution as
  corroboration/replication rather than a primary finding.

## Addendum — the impact path (how work like this gets traction), 2026-06

Believed true as of 2026-06; spot-check. The independent route that repeatedly turns a single-model
interp result into reach and a role — Nina Rimsky/Panickssery (bioengineering → SERI MATS under Hubinger
→ a trail of small activation-steering posts → the CAA flagship → ACL Outstanding Paper → Anthropic), and
Arditi et al.'s "refusal is a single direction" (blog-first April 2024 → NeurIPS, which the open-weights
world adopted as "abliteration") — shares a fixed shape:

1. **One crisp, nameable claim** — the title is the result; never "a battery of N controls."
2. **Forum first** (LessWrong / Alignment Forum), paper later. The post earns the traction; arXiv /
   conference backfills the credential.
3. **Lead with intuition + a demo a non-expert can feel** (TL;DR, analogy, before/after); rigor below the fold.
4. **Runnable on an open model, open code** — reproducibility is what turns a post into a movement
   (refusal-direction → abliteration). Our open Gemma-2 substrate already fits.
5. **Sit inside a live lineage** so the work is legible and citable.
6. **Build a sequence** of small posts; competence and attention compound.
7. **Honest framing** — state the one caveat plainly; do not over-claim.

For this repo: to share a *result* (not just this venue map), extract one behaviour, one figure, one
intuition, open code, one caveat — and strip the battery, the redundancy, and the triage scaffolding
from the public artifact.

## References (spot-check before external use)

- Circuit Tracing — methods: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- On the Biology of a Large Language Model: https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- HeadVis: https://transformer-circuits.pub/2026/headvis/index.html
- LORSA (low-rank sparse attention): https://arxiv.org/abs/2504.20938
- Neuronpedia — gemma-2-2b: https://www.neuronpedia.org/gemma-2-2b
- Circuits Research Landscape (Aug 2025): https://www.neuronpedia.org/graph/info
- circuit-tracer: https://github.com/safety-research/circuit-tracer
- circuit-tracer library paper (BlackboxNLP 2025): https://aclanthology.org/2025.blackboxnlp-1.14.pdf
- Open Problems in Mechanistic Interpretability (2025): https://arxiv.org/abs/2501.16496
