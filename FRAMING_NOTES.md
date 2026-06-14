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

## 3.6 Re-selecting by direct logit attribution: half the flip *is* MLP-mediated

§3.5 selected features by activation-delta and found ~0 mediation. The fix
(`framing_dla.py`): rank readout-position features by their *direct
contribution to the decision*, `delta_act . (W_dec . (W_U[:,Sydney] -
W_U[:,Canberra]))` — change in activation times decoder alignment with the
Sydney-minus-Canberra unembedding direction (the LN mean cancels in the token
difference). Then run the identical necessity/control test on that selection.
Artifact: `out/framing_dla.json`.

The DLA ranking picks a substantially different set (overlap 7-8 of 24 with the
activation-delta movers) — and those features **do** mediate. Necessity =
fraction of the framing's log-prob shift reverted by restoring the top-K DLA
features to baseline:

| framing (effect) | k=1 | k=3 | k=8 | k=24 | control@24 | Canberra rank |
|---|---|---|---|---|---|---|
| popular_wrong_city (+9.42) | +0.20 | +0.37 | +0.44 | **+0.49** | -0.13 | 435 -> 30 |
| assertive_false (+5.53) | +0.22 | +0.25 | +0.42 | **+0.48** | -0.22 | 10 -> 2 |

Restoring ~24 DLA-selected MLP features reverts **about half** the flip and
pulls Canberra from rank 435/10 back to 30/2 — versus **~0** for the same-count
activation-delta selection in §3.5. Same single mid-layer feature **L19/14947**
is the top mediator for both false-anchor framings (it fires lower under the
framing; restoring it alone reverts ~20%). Selecting by decoder-vs-decision
alignment, not by activation magnitude, is what surfaced the actual mediators.

**Two readings:**
- **The MLP path carries ~half.** A real, specific, reproducible chunk of the
  framing flip is mediated by readout-position transcoder features — the method
  *can* see it once you select correctly.
- **The other ~half is not here.** Necessity plateaus near 0.5; the remainder is
  most plausibly the attention copy of "Sydney" (QK-space, out of scope) — now a
  quantified gap (~50%), not a hand-wave.

**Asymmetry — flip vs reinforce.** The truthful-hedge framing (`hedge_true`,
which *raises* Canberra by 0.70 nats) is **not** MLP-mediated: its top DLA
feature is L25/4717 (the general answer-commitment feature) and restoring it
reverts ~0 (-0.02/-0.08/-0.02/-0.06 across K). So pushing the answer to a false
anchor and reinforcing the true one are not mirror-image mechanisms — the flip
has a partial MLP locus, the reinforcement (on this prompt) does not.

**Caveats specific to this test.** DLA is a first-order selection heuristic
(linear, ignores the LN nonlinearity beyond mean-cancellation and downstream
recomputation) — but the necessity numbers are the *actual* causal measurement
(full intervened forward pass), so the heuristic is validated by the test, not
trusted on its own. The matched-random control drifts to -0.13/-0.22 at k=24
(clamping 24 random features perturbs more), still well clear of the +0.48/+0.49
signal. Causal role only; no per-feature semantics asserted (L19/14947 is "a
feature whose restoration reverts the anchor flip", not "the Sydney feature").

## 3.7 The missing half is attention: severing attention to "Sydney" fully reverts the flip

§3.6 left ~half the flip unexplained and named attention-copy as the suspect.
Direct test (`job_attn.py` / `job_attn_sweep.py`, run against the warm worker):
at every layer, zero the attention *to* a chosen key token and renormalize, then
read Canberra. Artifacts: `out/framing_attn.json`, `out/framing_attn_sweep.json`.
(Cross-check: a plain `model()` forward reproduces the transcoder path exactly —
framed Canberra lp -10.357 both ways — so this is the same model the rest of the
experiment measures.)

**Severing attention to "Sydney" reverts the flip completely: necessity +1.04,
Canberra back to rank 0** (from rank 446). The per-token knockout sweep shows
this is specific — Sydney is the unique full-revert token:

| token (pos) | necessity | | token (pos) | necessity |
|---|---|---|---|---|
| **Sydney (1)** | **+1.04** | | famous (5) | +0.45 |
| is (2) | +0.26 | | Australia, question (13) | +0.42 |
| the/most/city/The/of (3,4,6,10,17) | ~0 (-0.01..+0.10) | | Australia, preamble (8) | **-0.49** |

The only other sizeable positives are *part of the Sydney-promoting clause*
("most **famous** city") or the question's own anchor ("capital of **Australia**");
genuine function words sit at ~0. Severing the preamble's "Australia" makes the
flip *worse* (-0.49) — it removes context competing with Sydney. So the earlier
"famous" control (0.45) wasn't a specificity failure of the method, it was a
bad control: famous is causally part of the framing.

**The mechanism, end to end.** The three interventions cut one pathway at
different depths and the numbers line up:

    "Sydney" token --[attention copy]--> prediction-position residual
                   --[late MLP "say-Sydney" features]--> Sydney >> Canberra logit

- cut at the **source** (attention to Sydney): full revert, ~1.0 (§3.7)
- cut **downstream**, at the DLA-selected MLP features: ~0.5 (§3.6)
- cut at the **wrong** downstream features (activation-delta movers): ~0 (§3.5)

The ~0.5 and ~1.0 are not additive — they are the same causal chain measured at
two points; the MLP features are a *partial readout* of the information attention
brings in. This is exactly the picture POSITIONING §3 predicts: the load-bearing
step is in QK-space (attention copy), which the transcoder-MLP attribution can
only see the downstream shadow of.

**Caveats.** The knockout zeros a key at all layers and all query positions then
renormalizes — a heavy, somewhat unphysical intervention (a query that attended
almost only to Sydney gets its row near-zeroed). necessity slightly over 1.0
(1.04) means removing Sydney leaves Canberra marginally *more* confident than the
no-preamble baseline. One prompt, one model, greedy single-token readout.

## 3.8 Transport: susceptibility is wording-dependent, the mechanism is invariant

Two transport questions (`job_transport.py`, `job_transport2.py`;
`out/framing_transport*.json`): does the *flip* generalize, and does the
*mechanism* generalize?

**Susceptibility depends on wording.** A neutral statement of the largest city
("Sydney is the largest city in Australia. The capital of Australia is...") does
**not** flip the answer for any of Australia / Texas / Canada / Switzerland /
Morocco (the model knows largest != capital; effect ~-0.2). Salience framing
("X is the most **famous** city in...") does flip. So the framing that bites is
about salience/relevance, not about asserting a fact — a behavioural finding in
its own right.

**The mechanism is pair-invariant.** Under the salience framing, restricting to
pairs whose capital is top-1 at baseline, 5/5 genuine flips are **fully reverted
by knocking out attention to the anchor city** — different countries, different
tokens, including multi-token anchors:

| pair | anchor -> flip | target rank | knockout necessity |
|---|---|---|---|
| Australia | Sydney | 0 -> 446 | +1.04 |
| Texas | Houston | 0 -> 8 | +1.04 |
| Canada | Toronto | 0 -> 2 | +1.04 |
| Switzerland | Zurich (Zur+ich) | 0 -> 31 | +1.02 |
| Morocco | Casablanca (multi-tok) | 0 -> 1 | +1.00 |

Necessity ~1.0 every time, capital restored to rank 0. (Brazil excluded — the
model's baseline top-1 for "capital of Brazil" is not Brasilia; New York/Buffalo
did not flip.) The generalization: **the false-anchor framing works by the model
attending to the anchored-city token and copying it to the prediction position.**
This is one mechanism, not a per-prompt coincidence — the same QK-space copy the
Dallas->Austin attribution method is blind to, now shown across five fact pairs.

## 3.9 Localizing the copy circuit: two principal heads, early-weighted

The §3.7 knockout was an all-layers/all-heads sledgehammer (necessity ~1.0).
Decomposing it (`job_localize_layers.py`, `job_localize_heads.py`,
`job_localize_joint.py`; `out/framing_localize_*.json`), Australia/Sydney:

**Per layer** (knock out attention to Sydney at one layer): distributed and
**early-weighted** — L0 +0.37, L7 +0.22, L3 +0.19, L18 +0.19, L1/L4 ~+0.13,
tail across L8/L11/L12/L22. No single layer dominates; the bias is laid down
mostly in the first few layers.

**Per head** (top-6 layers, 8 heads each): two heads stand out atop a diffuse
tail — **L0.H2 (+0.205)** and **L18.H5 (+0.197)** — then L0.H3, L7.H1, L1.H0
(<=0.1). An early head plus a mid head do the plurality of the copy.

**Joint cumulative** (knock out the top-k heads together):

| heads | necessity | Canberra rank |
|---|---|---|
| top-2 (L0.H2, L18.H5) | +0.398 | -> 54 |
| top-3 | +0.510 | -> 17 |
| top-5 | +0.702 | -> 3 |
| top-12 | +0.942 | -> 1 |

So the ~1.0 all-heads effect resolves into a **small circuit of ~12 heads across
6 layers**, two of them (one in L0, one in L18) carrying ~0.4 between them. Not a
single induction-style head, not fully diffuse — a compact early-to-mid circuit.
The early dominance (L0 the biggest single layer) is the notable shape: unlike
the late name-mover heads of IOI (Wang et al. 2023), the anchored-city bias here
is established in the *first* layer and read out mid-stack.

**Caveats.** Single-head necessities are not additive (the top-2 sum ~0.40 only
because L0.H2 and L18.H5 happen to interact little); heads were swept only in the
6 top layers, so the residual ~0.06 lives in untested layers; per-head
renormalization is the same heavy intervention as §3.7. One prompt/pair (the
transport of §3.8 was at the all-heads level, not re-localized per pair).

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

§3.5-3.8 now give a causal account that holds across fact pairs: salience
framing flips low-confidence facts (§2), the model does it by attending to and
copying the anchored city (§3.7, necessity ~1.0), late MLP features are the
partial downstream readout (§3.6, ~0.5), and this transports 5/5 across pairs
(§3.8). Remaining work sharpens and widens, it no longer chases the mechanism.

1. **Characterize the principal heads (L0.H2, L18.H5).** What do they attend to
   in general (copy/induction/previous-token)? Do the *same* heads carry the
   copy for Texas/Canada, or is the circuit pair-specific? Re-localize §3.8's
   pairs per head.
2. **Characterize L19/14947** — is it the "say the anchored city" feature across
   pairs, or Canberra/Sydney-specific? Run the DLA mediation on Texas/Canada.
3. **Map the susceptibility boundary.** Sweep framing wordings (largest /
   famous / asserted-false / hedged) x facts across the confidence range to
   chart when a prompt flips — §2's confidence-vs-susceptibility relationship,
   made quantitative.
4. **Fix the arithmetic distractor** (distinct first token, e.g. assert "63")
   to test numeric sycophancy now that the harness is warm and cheap.

(Done: §3.6 DLA selection, §3.7 attention-copy test, §3.8 transport, §3.9
head-level localization — the mechanism is identified, generalized, and mapped
to a compact early-to-mid head circuit.)

(Done: §3.6's logit-attribution selection, which was step 1 of the previous
list — it worked, hence the new priorities.)
