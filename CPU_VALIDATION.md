# CPU validation report (no-GPU pivot)

GPU access fell through, so this session validated everything that does not
require a Gemma forward pass, from a CPU-only container (4 cores, 15 GB RAM,
no HF token, HTTPS-allowlisted egress that turned out to permit PyPI, GitHub,
HuggingFace, and Neuronpedia). Date: 2026-06-12.

## Validated here

1. **API contract re-verified from source at the pinned commit**
   (`circuit-tracer @ 041a9b2cbd7f3fe7e0a625a6794e66fc4aa5f883`). All three
   `# API` wrapper assumptions hold: `from_pretrained(model_name,
   transcoder_set, ..., dtype=)` with default `backend="transformerlens"`;
   `get_activations` returns 3-D logits `[batch, seq, vocab]` plus a stacked
   activation cache `[n_layers, seq, d_transcoder]` (per-layer `squeeze(0)`
   then `torch.stack`); `feature_intervention(inputs, [(layer, pos, feat,
   value), ...])` returns 3-D logits. The repo's wrappers (with the
   3-D fix from the previous session) are correct as committed.

2. **Stack identity (the §4.1 PLT-vs-CLT concern).** The `"gemma"` preset
   resolves to `mwhanna/gemma-scope-transcoders`, whose `config.yaml`
   declares `model_kind: "transcoder_set"` (per-layer transcoders, hooks
   `ln2.hook_normalized` → `hook_mlp_out`) for `google/gemma-2-2b`. The
   canonical graph's source set is `gemmascope-transcoder-16k`. So the
   experiment stack is confirmed to be GemmaScope *per-layer* transcoders,
   matching the graph the TEXAS features were pinned from. The transcoder
   repo is **ungated** (anonymous download works).

3. **Frozen TEXAS constants verified against the canonical artifact.**
   Fetched the `gemma-fact-dallas-austin` graph via the public Neuronpedia
   API. Its `Texas` supernode is exactly the six pinned (layer, feature)
   pairs, all at ctx_idx 9 (the " Dallas" token, with `<bos>` at 0 — the
   same indexing `get_activations` will produce since TransformerLens
   prepends BOS). The §4.3 answer-key nodes ("say a capital city"
   21_5943_10 / 17_7178_10 / 7_691_10 / 16_4298_10 and "say Austin"
   23_12237_10) are present as documented. Canonical seed activations for
   all of these are checked in at `reference/canonical_graph_texas.json` —
   diff t0's `firing_positions` audit record against it on the real run
   (expect position 9 and activations within float tolerance, e.g.
   L19/7477 ≈ 55.78, L20/15589 ≈ 45.66).

4. **Single-token target assumption.** `" Austin"` encodes to the single
   token id 22605 (`▁Austin`); `" Texas"` and `" Dallas"` are also single
   tokens. Caveat: checked with the tokenizer from the ungated
   `unsloth/gemma-2-2b` mirror because the official repo is gated; the
   script still asserts this at runtime against the real tokenizer.

5. **Full measurement-logic verification** — `test_poc_cpu.py`, 56/56
   checks. A deterministic mock implementing exactly the three API surfaces
   (3-D logits included) drives `stage_t0` and `stage_t1` end-to-end with
   causal ground truth fixed by construction:
   - intervention construction: every active position covered, inactive
     features skipped, vacuous no-op path; the mock *raises* on any
     intervention at a position where the feature is not active, so the
     per-prompt position re-derivation is checked on every call;
   - S1 evaluated via both rank flip and drop > 1; S2 classifies a
     fully-compensated world as `redundant` (ratio 0) and a
     single-dominant-feature world as `NOT redundant` (ratio 1); S3's 3×
     specificity branch exercised with non-zero control drops;
   - matched-null sampler: layer-matched, magnitude-band-matched, excludes
     TEXAS, band-widening fallback reaches the only available candidate;
   - t1: behaviour filter (rank + signed margin), recruitment at the 0.25×
     boundary, and all four regimes (`filtered` / `A_transported` /
     `B_backup_candidate` / `C_non_recruited`) assigned correctly;
     per-structure regime table prints; JSON artifacts have the expected
     shape.
   Run with `.venv/bin/python test_poc_cpu.py` (needs only CPU torch).

6. **Environment setup path.** `pip install` of the pinned circuit-tracer
   tree succeeds and `from circuit_tracer import ReplacementModel` imports
   on CPU-only torch 2.12 — the `run_poc.sh` setup steps are sound on a
   box without CUDA.

7. **Frozen inputs intact.** Repo `paraphrases.json` is byte-identical to
   the briefing's frozen copy; 16 paraphrases, no duplicates, none equal to
   the seed, valid structure tags, all mention Dallas, none leak "Austin"
   or "Texas".

## Still blocked (and why)

The only thing not validated is the thing that needs the model:
`google/gemma-2-2b` is gated (`gated: manual`; anonymous file fetch → 401)
and this environment has no HF token. Running T0/T1 for real therefore still
needs either the GPU path in `docs/lambda-gpu-access.md` (Arc-1 PoC path archived at `archive/README_RUN.md`) or, in principle, this very
container *if* a licence-accepted HF token is provided: the weights
(~5 GB bf16) plus transcoders (~4 GB) fit in 15 GB RAM, and the experiment
is only ~60 short-prompt forward passes, so a CPU run is slow but plausible.
An ungated weight mirror exists (`unsloth/gemma-2-2b`), but `MODEL_NAME` is
a frozen experimental constant and licence acceptance is the operator's
decision, so no substitution was made.

## What this buys the GPU run

Every pre-registered failure mode that is *not* about Gemma's actual
behaviour has now been ruled out in advance: wrong API shapes, wrong
intervention sites, mis-pinned features, wrong transcoder stack, broken
S1–S4 / regime logic, malformed paraphrase set. If S1 fails on the real
run, it is now evidence about the model/features, not about the harness.
