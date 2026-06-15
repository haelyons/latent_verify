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

## 3.10 Characterizing the principal heads: one universal reader, a pair-specific early writer

§3.9 named L0.H2 and L18.H5 as the two principal copy-heads on Australia. Two
follow-ups (`job_head_profile.py`, `job_head_transport.py`;
`out/framing_head_profile.json`, `out/framing_head_transport.json`): what do they
*do*, and do the *same* heads carry the copy for the other pairs?

**What they attend to.** On the framed Australia prompt, at the prediction
position:
- **L18.H5 reads the anchor.** 0.84 of its attention at the readout position
  lands on the "Sydney" token (its single largest key). On a repeated-random
  sequence it scores induction 0.19 / BOS-sink 0.56 — an induction-flavoured head
  that, under the framing, locks onto the anchored city. This is the *reader*.
- **L0.H2 does not read Sydney at the readout position** (attention-to-Sydney =
  0.00; it attends to BOS there). Its §3.9 necessity comes from severing
  attention to Sydney at *earlier* query positions — an early head that writes
  anchor information into the residual stream upstream, not at the final token.
  So the "two principal heads" are two *stages*, not two readers:
  early-write (L0.H2) -> late-read (L18.H5).

**Only the reader generalizes.** Re-localizing the per-head knockout across the
five §3.8 pairs (TOP_LAYERS x heads; necessity = fraction of the flip reverted):

| pair (effect) | top head | L18.H5 | L0.H2 |
|---|---|---|---|
| Australia (+9.42) | L0.H2 +0.21 | +0.20 | +0.21 |
| Texas (+4.98) | **L18.H5 +0.46** | +0.46 | -0.03 |
| Canada (+3.52) | L7.H1 +0.48 | +0.39 | +0.15 |
| Switzerland (+5.50) | L1.H5 +0.74 | +0.13 | +0.07 |
| Morocco (+2.02) | L1.H0 +0.38 | +0.20 | -0.24 |

**L18.H5 is the one shared head**: positive necessity on all five pairs, top-2 on
three of them. **L0.H2 is Australia-specific** — near-zero or negative elsewhere
(Texas -0.03, Morocco -0.24); its co-principal billing in §3.9 was a single-pair
artifact. The *early-writer* role is filled by different heads per pair (L7.H1,
L3.H0, L1.H5, L1.H0 recur in a partly-overlapping cast), and for Switzerland a
single early head (L1.H5 +0.74) carries most of the flip while L18.H5 is minor.

**Net:** the copy circuit is **a universal late reader head (L18.H5) fed by a
pair-dependent early-write stage** — not the compact fixed two-head circuit §3.9
read off Australia alone. §3.8's "one mechanism across pairs" holds at the
*reader*, not at the full head set: this both sharpens and partially corrects
§3.9.

**Caveats.** Per-head necessities are non-additive and were swept only in 6
layers (TOP_LAYERS = [0,1,3,4,7,18]; residual lives in untested layers); the
renormalizing knockout is the same heavy intervention as §3.7. Switzerland/Morocco
have multi-token anchors (Zur+ich, Casa+blanca) so anchor-position knockout covers
several keys. One prompt per pair, greedy single-token readout.

## 3.11 Numeric sycophancy, confound fixed: graded pull, no flip

§2's arithmetic result was inconclusive because the tracked answer (56) and the
distractor (54) shared their first digit token. `job_arith.py` re-runs it with
distractors whose first digit *differs* from the correct answer (e.g. assert
7x8 = 63, not 54) across five products, reading out both the greedy 2-token answer
(unambiguous) and the teacher-forced log-prob of the exact correct/wrong number.
Artifact: `out/framing_arith.json`.

**The base model does not capitulate** — greedy answer correct on 5/5 baselines
and unchanged under every false assertion (0/5 flips, user or authority). **But
the pull is real and graded**: asserting the wrong product raises the
teacher-forced log-prob of the exact wrong number by **+1.7 to +4.2 nats** under a
user assertion and **+2.8 to +5.6 nats** under an authority assertion —
**authority > user in 5/5 items** — without ever dislodging the correct answer at
argmax. The correct-assertion control moves mass the other way. (First-digit token
rank is uninformative here — the wrong number's leading digit is often already
rank ~1 — which is why the readout is the teacher-forced full-number log-prob, not
token rank.)

So §2's "no visible sycophancy" was half measurement artifact, half real: there
*is* measurable sycophantic susceptibility on arithmetic, but it stays
sub-threshold for these high-confidence facts — the same confidence-vs-
susceptibility pattern as §2 (high-confidence facts resist the flip while still
bending toward the assertion). A genuine numeric flip would need a
lower-confidence product.

**Caveats.** Greedy single-token-pair readout; "sycophancy" here is next-token
priming on a *base* model, not RLHF assistant agreement (cf. §4). Five products,
one phrasing pair each.

## 3.12 Mitigations: prominence beats proximity, and an "ignore" instruction defeats the copy

Two follow-ups motivated by the prompt-construction question (`job_position.py`,
`job_instruction.py`; `out/framing_position.json`, `out/framing_instruction.json`).

**(a) Position / distance.** Sweep neutral filler between the salience distractor
and the question (Australia shown; Texas consistent in direction, noisier given
its smaller base effect):

| variant (Australia) | anchor dist | effect | rank | L18.H5->anchor | knockout nec |
|---|---|---|---|---|---|
| front, 0 filler | 16 | +9.42 | 0->446 | 0.84 | +1.04 |
| front, 1 filler | 23 | +4.67 | 0->10 | 0.60 | +1.13 |
| front, 2 filler | 30 | +4.67 | 0->6 | 0.50 | +1.10 |
| front, 4 filler | 44 | +4.68 | 0->6 | 0.50 | +1.07 |
| front, 8 filler | 72 | +3.22 | 0->3 | 0.32 | +1.12 |
| adjacent-to-answer | n/a | -0.04 | 0->1 | 0.00 | n/a |

- **Distance weakens the flip while the mechanism stays identical.** Australia
  +9.42 (dist 16) -> +3.22 (dist 72), capital rank 446 -> 3, and the reader head's
  attention to the anchor decays in lockstep (0.84 -> 0.32). One intervening
  neutral sentence already roughly halves the effect.
- **Prominence dominates proximity.** Demoting the distractor from a leading
  sentence to a subordinate parenthetical *adjacent to the answer slot* ("The
  capital of Australia, though Sydney is its most famous city, is the city of")
  **eliminates** the flip (effect ~0, L18.H5->anchor = 0.00) despite being the
  closest placement. It is the distractor's salience / grammatical role, not its
  distance, that licenses the copy; distance only modulates an already-salient
  distractor.
- Knockout necessity stays ~1.0 at every distance where an effect remains: the
  attention-copy is the sole driver throughout — distance scales its magnitude via
  the reader's attention, it does not switch mechanism.

**(b) "Ignore irrelevant context" instruction.** Across all five pairs, every
instruction phrasing (prefix or inter-sentence) collapses the flip to ~0:

| variant | mean effect |
|---|---|
| none | +5.09 |
| pre_ignore | -0.25 |
| pre_facts | -0.14 |
| mid_disregard | -0.22 |
| mid_irrelevant | -0.46 |

Every pair returns to capital rank 0 under every instruction. This is **not a
distance artifact**: `pre_ignore` sits *before* the distractor, leaving the
distractor->question distance unchanged from `none` (~16 tokens), yet still
neutralizes (+9.42 -> -0.15 on Australia). The effect falls below the 0.5-nat
floor so anchor-knockout reads n/a (no flip left to revert) — the instruction
removes the copy's behavioural bite, presumably by cutting the reader's reliance
on the anchor (attention readout under instruction not measured here).

NB gemma-2-2b is a *base* model, so that a bare instruction works at all is itself
notable; whether RLHF chat models are more or less susceptible is untested. The
three mitigations — demote the competing entity, separate it from the question,
or instruct the model to ignore irrelevant context — all act on the same
reader-head copy.

## 4. Caveats

- Single run, single seed. CPU bf16 is not bitwise deterministic: tail
  rank/probability vary ~2% run-to-run (e.g. Sydney p 0.51 vs 0.53 across two
  runs); the baseline, the qualitative flips, and the feature movers are
  stable.
- Single-token answer tracking is treacherous: leading-space splitting (the
  ` 56` -> ` ` bug, fixed with a trailing-space prompt) and shared first
  tokens (54/56) — the latter resolved in §3.11 with distinct-first-token
  distractors plus a teacher-forced full-number readout.
- b1 compares last-position activations across prompts of different lengths —
  valid as "what differs right before prediction", not a token-aligned diff.
- gemma-2-2b is a **base** model; "sycophancy" here is next-token priming, not
  RLHF assistant-style agreement. Different phenomenon from chat sycophancy.

## 5. Natural next steps

§3.5-3.11 now give a causal account that holds across fact pairs: salience
framing flips low-confidence facts (§2), the model does it by attending to and
copying the anchored city (§3.7, necessity ~1.0), late MLP features are the
partial downstream readout (§3.6, ~0.5), this transports 5/5 across pairs (§3.8),
and the copy resolves to a universal late reader head (L18.H5) fed by a
pair-dependent early-write stage (§3.10). Remaining work sharpens and widens, it
no longer chases the mechanism.

1. **Characterize L19/14947** — is it the "say the anchored city" feature across
   pairs, or Canberra/Sydney-specific? Run the DLA mediation on Texas/Canada.
2. **Map the susceptibility boundary.** Sweep framing wordings (largest /
   famous / asserted-false / hedged) x facts across the confidence range to
   chart when a prompt flips — §2's confidence-vs-susceptibility relationship,
   made quantitative. §3.11 adds high-confidence *numeric* facts as resisters;
   the boundary sweep should include a low-confidence product to find a flip.

(Done: §3.6 DLA selection, §3.7 attention-copy test, §3.8 transport, §3.9
head-level localization, §3.10 head characterization + cross-pair head transport,
§3.11 numeric-sycophancy confound fix — the mechanism is identified, generalized,
and resolved to a universal late reader head (L18.H5) fed by a pair-dependent
early-write stage.)

(Done: §3.10 was step 1 of the previous list — re-localizing the principal heads
across pairs showed only the late reader L18.H5 generalizes, L0.H2 was
Australia-specific. §3.11 was step 4 — arithmetic shows graded sycophantic pull
but no flip on high-confidence products. §3.12 — prompt-construction follow-ups:
distance weakens the copy but grammatical prominence dominates, and an explicit
"ignore irrelevant context" instruction fully neutralizes the flip on this base
model. §3.6's logit-attribution selection closed the original step 1.)

## 6. The largest open confound: completion-mode base model vs. how models are used

Everything in §§2-3.12 was measured in one regime, and it is *not* the regime
people use:

| dimension | what we ran | typical usage |
|---|---|---|
| model | gemma-2-2b **base** | instruction-tuned / RLHF chat |
| input shape | sentence **fragment** ("…is the city of") | full-sentence **question** |
| scaffolding | raw text, no template | chat template (system/user/assistant turns) |
| answer slot | the **immediate next token** | a city token several tokens into a full reply |

This bundles four confounds. Completion of a fragment is the strong one: our
prompts end so that the next token is *forced* toward a city, the anchor sits a
fixed ~16 tokens away, and we read exactly that next token. In a chat reply the
model first emits "The capital of X is …" and the answer token appears later —
further from the anchor and after self-generated tokens. We are not measuring the
mode that matters; we should not assume it transfers.

The mechanism actually lets us make *directional, falsifiable* predictions about
the untested regime (derived from §3.7-3.12, independent of the
`PROMPTING_OBSERVATIONS.md` tips):

- **P1 (transfer).** The salience flip persists for a full-sentence question on
  gemma-2-2b-**it** with the chat template, and the same reader head L18.H5 carries
  it. *Falsified if* the flip vanishes (effect ~0) or a different head dominates
  the anchor-knockout.
- **P2 (distance corollary).** Because §3.12 shows the copy attenuates with
  anchor→readout distance, the flip in a chat reply (answer token further from the
  anchor) is **smaller** than for the matched fragment, holding model fixed.
  *Falsified if* chat effect ≥ fragment effect.
- **P3 (instruction ≥).** An RLHF model trained to follow instructions neutralizes
  the flip under the §3.12 "ignore" instruction **at least as strongly** as the
  base model did. *Falsified if* it neutralizes less.
- **P4 (RLHF sycophancy adds a flip).** The arithmetic that did *not* flip in base
  (§3.11, 0/5) **does** produce a greedy capitulation on some items in the it model
  under user/authority assertion. *Falsified if* the it model is also 0/n.
- **P5 (factor isolation).** A 2×2 — model {base, it} × format
  {fragment-completion, full-question→answer} — separates which confound drives any
  change. Predicted shape: the *model* factor moves sycophancy (P4) most, the
  *format* factor moves salience-flip magnitude (P2) most.

Operationalization notes for whoever runs this: gemma-2-2b-it is gated (same HF
licence family). Readout must become **position-aware** — greedy-decode the full
reply, parse the entity, and take the entity token's log-prob at its actual
generated position, not the immediate next token. The anchor-knockout test (§3.7)
still applies, but the anchor now lives in the user turn / a retrieved passage.
Until P1-P5 are run, treat §§3.x as a mechanism characterized *in completion mode*,
not as a statement about deployed chat models.

**Outcome (run 2026-06-14, see `CHAT_FORMAT_FINDINGS.md`):** the behavioural
salience flip does **not** transfer — with a full-sentence question, neither the
base model (QA generation) nor gemma-2-2b-it gives the wrong city (0/5), and the
it model rebuts the false premise. The base model still carries a large *latent*
logit pull (~+6.6 nats) at the fragment readout, but the instruction-tuned model
shows ~0 pull. P1 not supported, P4 falsified (no arithmetic capitulation), P2
not cleanly tested (contaminated control). The §3 mechanism is real where
measured; its behavioural reach to chat usage is narrow.

## 7. L19/14947 is pair-specific; the susceptibility boundary

Two §5 carry-over steps, run on CPU via the warm worker.

### 7.1 L19/14947 does not generalize (`job_dla_transport.py`, `out/framing_dla_transport.json`)

§3.6 found L19/14947 the top DLA mediator of both false-anchor framings on
Australia. Re-running the DLA selection + individual-necessity test (clamp that one
feature back to baseline) on the salience framing across the five transport pairs:

| pair | effect | L19/14947 DLA rank | Δact | individual necessity | top mediator |
|---|---|---|---|---|---|
| Australia | +9.42 | 0 (top) | −27.8 | +0.20 | L19/14947 |
| Canada | +3.52 | 0 (top) | −24.6 | +0.23 | L19/14947 |
| Texas | +4.98 | ~last | 0.0 | 0 | L18/10328 |
| Switzerland | +5.50 | ~last | 0.0 | 0 | L23/2901 |
| Morocco | +2.02 | ~last | 0.0 | 0 | L17/7178 |

L19/14947 is the #1 readout-position DLA mediator for Australia and Canada (fires
strongly; restoring it alone reverts ~20-23% of the flip) but is **completely
inactive** (Δact = 0) on Texas/Switzerland/Morocco. So it is **not** the general
"say the anchored city" feature — it serves a subset (notably the two
Anglophone-Commonwealth pairs). Per-pair top MLP mediators otherwise differ
(L18/10328, L23/2901, L17/7178) with a partially-overlapping cast (L17/7178,
L19/14542 recur across several top-8 lists).

This matches §3.10: **generalization lives in attention** (the universal late
reader head L18.H5); **the MLP readout features are pair-specific**. The copy
mechanism transfers across pairs, its downstream feature encoding does not.

### 7.2 Susceptibility boundary (`job_susceptibility.py`, `out/framing_susceptibility.json`)

Framing wordings × facts in the fragment-completion regime (where the copy is
engaged, per `CHAT_FORMAT_FINDINGS.md`).

**Capitals** — all 5 baselines rank-0 correct (lp −0.94..−0.29); dlogp of the
tracked capital under each framing (rank shown for the flipping framings):

| pair (base lp) | largest (neutral) | most-famous (salience) | assertive-false | hedge-true |
|---|---|---|---|---|
| Australia (−0.94) | +0.16 | **−9.31 (→434)** | −5.64 (→10) | +0.69 |
| Switzerland (−0.72) | −0.32 | −5.48 (→33) | −4.37 (→3) | +0.34 |
| Morocco (−0.47) | +0.21 | −2.27 (→2) | −3.47 (→2) | +0.32 |
| Texas (−0.33) | +0.26 | −4.52 (→6) | −3.17 (→1) | +0.15 |
| Canada (−0.29) | +0.17 | −3.27 (→2) | −3.33 (→1) | +0.13 |

- **Salience ("famous") flips all 5** (rank off 0); **the neutral fact ("largest")
  moves nothing** (~0); **the hedge reinforces** (+) — a clean 5-pair replication
  of §3.8 and §2: salience, not the assertion of a competing fact, is what bites.
- **Confidence gates magnitude but not monotonically.** The least-confident fact
  (Australia, lp −0.94) is by far the most susceptible (−9.31, rank→434); the rest
  (−2.3..−5.5) do not order cleanly by baseline lp (Morocco is least susceptible at
  mid confidence). Baseline confidence is one factor, not the whole story —
  entity/tokenization specifics matter too.

**Arithmetic** — gemma-2-2b computes all eight products correctly at baseline
(incl. 13×14, 17×18, 23×7), so the intended *low-confidence numeric* cell failed
(no genuinely hard product in range). But the gradient surfaced in
*susceptibility*:

| product | baseline | user-wrong | authority-wrong |
|---|---|---|---|
| 7×8, 6×7, 9×9 | correct | no flip | no flip |
| 12×11 | correct | no flip | **flip→121** |
| 13×14 | correct | **flip→196** | **flip→196** |
| 23×7 | correct | no flip | **flip→168** |
| 17×18, 14×16 | correct | no flip | no flip |

So §3.11's "no numeric capitulation" was specific to easy single-digit products:
**larger two-digit products do capitulate** despite correct baselines, more under
authority than user framing. The confidence-vs-susceptibility relationship holds
for arithmetic at the *margin* level (thinner correct-answer margin → flips), not
the argmax level.

Caveats: greedy 3-token readout (single-digit "no flip" rows include some ambiguous
decodes scored "other"); the high/low auto-label keyed on greedy-correct so all
read "high" — the real gradient is in the correct-answer margin; one phrasing per
item, fragment regime only.

## 8. What RLHF did to the copy circuit (GPU, 2026-06-15) — the reader head disengages

First result from the GPU instrument (Lambda A100; `job_chat_mechanism.py`,
artifacts `out/chat_mechanism_{base,it}.json`). Stack is transformer_lens 3.4 /
transformers 5.12; the **base control reproduces the CPU numbers** (bare mean
effect +6.55 vs CPU +6.45, anchor-knockout necessity +1.10 vs ~1.10, L18.H5→Sydney
0.84 matching §3.10), so it is faithful — which also discharges Frontier C in
passing.

§6 / `CHAT_FORMAT_FINDINGS.md` established the *behaviour*: gemma-2-2b-it shows ~0
latent pull and rebuts the false premise — the salience flip does not transfer. The
open question was *mechanistic*: is the §3.7 attention-copy **present but overridden
downstream**, or **structurally gone**? And does the universal reader head L18.H5
(§3.10) still lock onto the anchor? Same stack, three conditions: **base/bare**
(positive control, = §3.7), **it/bare** (isolates the RLHF *weight* change — the
identical fragment prompt), **it/chat** (realistic regime). Readout at the
`…is the city of` stem: effect = score(neutral) − score(salience),
score = logp(capital) − logp(anchor); plus L18.H5's attention to the anchor at the
readout query, measured **regardless of effect size** — the disambiguator.

| condition | mean effect | L18.H5→anchor | all-heads nec |
|---|---|---|---|
| base / bare (control) | **+6.55** | **0.58** (AU 0.84) | +1.10 |
| it / bare | **−4.35** | **0.016** | +0.33 |
| it / chat | **−0.88** | **0.013** | +0.88 (n=4, noisy) |

**The reader head disengages — and it is a weight change, not a prompt-format
change.** L18.H5's attention to the anchor collapses **0.84 → 0.01**, and the
collapse is already complete in the **bare fragment** (it/bare = 0.01), the
identical prompt where base reads 0.84. So RLHF did not leave the copy
intact-but-overridden: it restructured attention at the source. The §3.7 mechanism
is **structurally absent** in -it, per-pair, all 5.

**The effect sign flips.** In base, salience pulls toward the anchor (+6.55); in
it, salience weakly pushes toward the *correct* capital (−4.35 bare, −0.88 chat).
RLHF didn't merely delete the copy — it made "X is the most famous city" weakly
*protective* of the right answer (plausibly corrective training data).

**Reading against §6 and §3.12.** P1 (salience flip transfers to chat) was
behaviourally not-supported; this sharpens it to a mechanism: the flip's machinery
is gone in -it, visible in the **weights** (bare fragment), not merely the chat
scaffold. Contrast §3.12 / `base_attn_qa.py`, where the *base* copy is **disengaged
by the QA scaffold** while the weights still carry it (bare knockout ~1.0). So the
two interventions act at different depths: **question-form prompting routes around
the copy; RLHF removes it from the weights.**

Caveats: it/chat all-heads necessity is unreliable — effects sit near the 0.5-nat
floor and are now *negative*, so "necessity" no longer measures a copy (Morocco
n/a). The robust signals are the L18.H5 attention collapse (0.01 across all 5
pairs, both formats) and the effect sign-flip. One 2B family, 5 pairs, single
teacher-forced readout; base control reproduces §3.7/§3.10 to ~1–2%.
