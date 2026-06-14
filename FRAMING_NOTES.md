# FRAMING — exploratory observations (CPU, 2026-06-13)

First exploratory pass at applying the attribution stack (gemma-2-2b +
GemmaScope per-layer transcoders, circuit-tracer @041a9b2) to **framing**
rather than to the Dallas->Austin graph. Run on the web CPU sandbox via the
memory-guarded `load_model` ported from the gqfo2t branch. Artifacts:
`out/framing_b0.json`, `out/framing_b1.json`. Tool: `framing_probe.py`,
inputs frozen in `framing_situations.json`.

This is a **measurement** pass, not a hypothesis test: no pre-registered
pass/fail, no intervention yet. The point is to see what framing does and to
shake out the instrument before committing to a causal design.

## 0. Infrastructure note (the thing that was actually stalled)

The earlier "harder sandbox" account does not reproduce. This container is the
same spec the prior sessions reported (4 vCPU, 15 GiB RAM, no GPU, ~no swap by
default). With the gqfo2t memory-guarded load path (bf16 `hf_model=` +
`lazy_encoder=True` + the `ProcessWeights` identity-guard that skips the
fp32 state-dict round-trip) and an 8 GB swapfile, `ReplacementModel` loads in
**78 s at 14.34 GB peak RSS** — comfortably under the ceiling. The previous
session's OOM was the unguarded fp32 construction transient (~25 GB committed),
not the environment. Sanity check on load: ` Austin` top-1 on the seed and all
six Texas features fire at magnitudes bit-matching the gqfo2t run.

So: the loading path was the blocker, the token was not, and the sandbox is
not special.

## 1. What was measured

Four "situations", each the same question under several framings, with a
neutral `baseline` every other framing is compared against:

| id | kind | question |
|---|---|---|
| `capital_australia` | factual_anchor | capital of Australia (answer Canberra) |
| `tallest_mountain` | factual_anchor | tallest mountain (answer Everest) |
| `arithmetic_7x8` | sycophantic_assertion | 7 x 8 (answer 56) |
| `sentiment_movie` | valence_frame | "The movie was ___" continuation |

- **b0 (behaviour):** for each framing, the top-k next token and — if an
  answer is pinned — its probability, rank, and the **change in
  log-probability** vs baseline (`dlogp`). Log-prob, not raw logit: raw logits
  are not comparable across different prompts (per-context softmax shift); an
  early version reported raw `dlogit` and produced a sign that contradicted
  the probability, which is the bug the `dlogp` switch fixes.
- **b1 (attribution, descriptive):** at the prediction position, the
  transcoder features whose activation moves most between baseline and each
  framing — the same feature units the Dallas->Austin work clamps, here read
  out rather than intervened on.

## 2. Behavioural findings (b0)

**Framing-susceptibility tracks baseline confidence.** The two factual anchors
behave oppositely under the *same* manipulation (a false "everyone knows X"
preamble):

| framing | Canberra (baseline p=0.39) | Everest (baseline p=0.98) |
|---|---|---|
| popular-wrong preamble | flips to **Sydney**; Canberra dlogp **-9.32**, rank 0->442 | **Everest** holds; dlogp **-0.01** |
| assertive-false preamble | **Sydney** p=0.92; Canberra rank 0->10 | **Everest** p=0.91; dlogp **-0.07** |
| hedge-true preamble | reinforces **Canberra** p=0.39->0.80 (dlogp +0.71) | — |

A ~100x difference in framing-induced log-prob shift between the low-confidence
fact the model flips on and the high-confidence fact it defends. The hedge
("although many get this wrong, in fact...") *raises* confidence in the correct
low-confidence answer — framing cuts both ways.

**Valence priming bends an open-ended continuation.** "The movie was ___":
baseline is neutral/structural (` a`, ` released`, ` filmed`); a positive
preamble shifts top-k to ` great` / ` amazing` / ` fantastic`; a negative
preamble to ` so` / ` terrible` (with ` good` still present — negative priming
is weaker/messier than positive here).

**No visible sycophancy on arithmetic — but the measurement is inconclusive,
not the model.** Asserting a wrong product ("I'm pretty sure 7x8 is 54")
*sharpened* the model toward a `5_` answer (p 0.52->0.95) rather than
capitulating. BUT the tracked answer 56 and the distractor 54 **share their
first digit token `5`** (gemma tokenizes digits individually), so first-token
tracking cannot tell 54 from 56. This situation needs a distractor with a
*distinct* first token (e.g. assert "63") before any sycophancy claim is made.

## 3. Attribution findings (b1, descriptive only)

- On `capital_australia`, one late feature **L25/4717** dominates the movement
  and moves *with the behaviour*: strongly down when the framing pushes Sydney
  (-184, -97.5) and up when the framing reinforces Canberra (+27). A clean
  candidate mediator.
- But L25/4717 also dominates the `arithmetic` movers (+155 on the wrong
  assertion) and appears on `tallest_mountain`'s assertive-false (-70). It is
  therefore more plausibly a **general late-layer answer-commitment / output
  feature** than a Canberra-specific one — a caveat against reading single
  movers as topic semantics.
- Valence has distinct signatures: L25/11867 (down) for positive priming,
  L22/2432 (down) for negative — different features, not one signed axis.

These are activation deltas: **correlational**. Whether any of these features
*causes* the framing effect needs the intervention step (clamp the mover, à la
T0, and watch the answer) — done in §3.5.

## 3.5 Intervention: do the movers actually mediate the effect? (No.)

`framing_intervention.py` clamps the top-K movers at the prediction position
and measures the fraction of the framing's log-prob shift they account for:
necessity (restore movers to baseline on the framed prompt), a matched-random
control, and sufficiency (inject framed values onto the baseline prompt).
Swept K = 1/3/8. Artifact: `out/framing_intervention.json`.

**On the one large, clean effect — `capital_australia / popular_wrong_city`,
the +9.30-nat Canberra->Sydney flip — restoring the top movers reverts almost
none of it:**

| K | necessity | control | sufficiency |
|---|---|---|---|
| 1 | +0.02 | +0.00 | +0.01 |
| 3 | -0.02 | -0.00 | -0.01 |
| 8 | -0.08 | +0.04 | +0.00 |

Necessity is ~0 and indistinguishable from the matched-random control;
clamping the biggest activation-movers (including L25/4717, which moved -184)
back to baseline does **not** restore Canberra (rank stays ~435; at K=8 it gets
slightly *worse*). `assertive_false` is the same story (necessity <= 0 at all
K). So the b1 movers **co-vary with the framing but do not cause the next-token
flip** — L25/4717's decoder direction is evidently near-orthogonal to the
Canberra<->Sydney logit difference. **Biggest activation mover != causal
driver**, the exact reason this project intervenes rather than reads graphs.

**Most likely locus of the real effect:** attention copying "Sydney" from the
framing text into the prediction position — a QK-space mechanism that, by
construction, the transcoder-MLP attribution cannot see or clamp (POSITIONING
§3, filter 2). Restoring last-position MLP features can't undo an
attention-mediated copy.

**Degenerate rows.** For `tallest_mountain` the framing effect is ~0.001-0.06
nats (the model didn't move), so the necessity/sufficiency *fractions* in the
JSON (+3.38, -8.84, ...) are divide-by-~zero artifacts, not signal. The code
now reports these as n/a below MIN_EFFECT = 0.5 nats; the committed JSON
predates that guard, so read every `tallest_mountain` fraction as n/a.
`arithmetic` effects are small (~0.6 nats) and confounded (54/56), so its
fractions are also uninformative.

**Net:** the first causal test returns a clean negative — the readout-position
feature-movers are not the mediators of the framing flip. This is an
informative null, and it points the next experiment at attention / earlier
positions rather than at more last-position features.

## 4. Caveats

- Single run, single seed. CPU bf16 is not bitwise deterministic: tail
  rank/probability vary ~2% run-to-run (e.g. Sydney p 0.51 vs 0.53 across two
  runs); the baseline, the qualitative flips, and the feature movers are
  stable.
- Single-token answer tracking is treacherous: leading-space splitting (the
  ` 56` -> ` ` bug, fixed with a trailing-space prompt) and shared first
  tokens (54/56, still open).
- b1 compares last-position activations across prompts of different lengths —
  valid as "what differs right before prediction", not a token-aligned diff.
- gemma-2-2b is a **base** model; "sycophancy" here is next-token priming, not
  RLHF assistant-style agreement. Different phenomenon from chat sycophancy.

## 5. Natural next steps

The §3.5 null reshapes the priority list: last-position activation-movers are
not the mediators, so the next moves are about *where* and *how* to attribute,
not about clamping more of the same.

1. **Select features by logit attribution, not activation delta.** Rank
   candidates by `delta_activation x (decoder . unembed_direction)` for the
   Canberra<->Sydney token difference (direct logit attribution), then clamp
   those. This targets features that actually move the *decision*, not just
   features that move. The current selection rule is what failed.
2. **Intervene at the framing-token positions, not just the readout.** Clamp /
   ablate the residual contribution at the "Sydney" token region, and test the
   attention-copy hypothesis directly (it's the most likely locus). This may
   fall partly outside the transcoder-MLP method (QK-space) — a scope finding
   in itself.
3. **Fix the arithmetic distractor** (distinct first token, e.g. assert "63")
   to actually test numeric sycophancy.
4. **Transport** — re-run each framing across paraphrases (the T1 idea) to see
   whether a framing effect is prompt-specific or a mechanism.
5. Broaden the situation set across the confidence range to test the
   confidence-vs-susceptibility relationship quantitatively.
