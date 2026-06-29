# Reflection: the attribution-graph verifier, reframed

*What the de-collide / mechanism arc (see `archive/SEQUENCE_290626_decollide-confound.md`) says about the program's founding idea — an attribution-graph "verifier" that uses paraphrase-stability to tell a real mechanism from a one-prompt artifact.*

## 1. How the results reframe the attribution-graph verifier

The original idea: a graph is a *hypothesis*; verify it's a real mechanism (not a one-prompt overfit) by checking it **survives a frozen family of paraphrases**. Implicit assumption — the failure mode worth guarding against is *prompt-specificity*, and paraphrase-invariance catches it.

The results show that's the wrong threat model. The doubt circuit **passed every paraphrase-stability test** — the READ/WRITE restoration reproduced across the whole misconception family and across 2b/9b/27b, the head set was stable (`overlap 5/3/5`) — and it was still an **artifact**. Because the confound (first-token "Yes" colliding with the affirmation to "Are you sure?") is itself **paraphrase-invariant**: every polar rephrasing carries it. So:

> **Paraphrase-invariance is necessary but not sufficient.** A confound that lives in the *readout* (answer format) rather than the *input* survives the paraphrase verifier and looks exactly like essential circuitry.

The reframe: a verifier needs **three** independent invariances, not one —
- **(a) paraphrase-stability** — robust to rephrasing the *input* (the original axis).
- **(b) readout-robustness** — robust to how you score the *output* (first-token vs content-margin vs judge). The axis that actually broke the circuit (de-collide: RA reproduces, RC → floor).
- **(c) intervention-consistency** — direct-effect (graph edges / DLA) must agree with total-effect (ablation). Redundancy makes marginal attributions lie; the polarity write, the defer direction, and the WRITE table all failed this (ablation ≈ floor while DLA/patch looked large).

The doubt circuit passed (a), failed (b) and (c). The verifier as conceived only tested (a).

And the sharp irony: the **attribution graph was the one instrument that got it right from the start** — `cave_attribution_graph` returned BROAD_DISTRIBUTED (completeness 0.93, top-15 features 17%, top-15 ablation ≈ random). The graph didn't need a paraphrase-verifier to catch an overfit; it was already honest. What needed verifying was the *ablation table we replaced it with*.

## 2. How we deviated from the original idea

- **We abandoned the graph instead of trusting it.** The graph said "distributed, no sparse circuit." We didn't believe it — rationalised it away ("graphs freeze attention; caving is a QK copy") — and pivoted to IOI-style one-at-a-time ablation. That pivot is exactly where the artifact entered.
- **We swapped the readout without re-verifying the new one.** The graph scores a logit-*difference* over the answer (a content readout). The ablation table switched to first-token `P(W*)` — a cheaper, collision-prone readout — and nobody asked whether the new instrument was as honest as the graph. The paraphrase family stayed constant; the *measurement* changed silently.
- **We let the selection gate be the confound.** Faithful items were chosen by first-token rise, which over-sampled the affirmation-collision items (the faithful sets were ~90–100% polar; the honest content-gated population is ~half wh). So the "paraphrase family" was self-selected toward the artifact.
- **We treated the verifier's own pass criterion (survives paraphrase + reproduces across scale + concentrated heads) as proof** — the precise mistake the verifier was meant to prevent, applied to the replacement method instead of the graph.

So the de-collide / content-gate / mechanism work is really us *returning* to the verifier idea — but supplying the two invariances (b, c) the original was missing, the hard way: by building a false circuit and dismantling it.

## 3. What the results say about valid paraphrases for the verifier

The central practical finding — **a valid paraphrase family must decorrelate surface form from mechanism, especially in the *output*:**

- **Paraphrase the answer space, not just the question.** Stability across question wordings is worthless if every wording elicits the same output token. Our family held the readout token ("Yes") constant — so it tested input-invariance while *freezing the confound*. Valid families must vary how the answer is expressed: **include wh/entity items where the committed token is content, not a polarity marker.** The content-gate showed the honest caving population is ~half wh — *those* are the verifier-valid paraphrases; the polar ones are confound-carriers and should be balanced, not allowed to dominate.

- **Span the polarity axis.** Include items where the correct answer is "Yes" *and* where it is "No," so a content mechanism survives both while an affirmation-artifact sign-flips. (Caveat from our reversal arm: it was underpowered — denialist No-myths are high-confidence and don't cave — so the family must *also* control **confidence**: low-confidence items in **both** polarities.)

- **Select the family by a content criterion, not a surface-token one.** Our faithful gate (first-token rise) biased selection toward the artifact, and leaked implausible items ("pigs fly"). A valid verifier selects paraphrases by whether the *content commitment* actually moves (a content-margin gate) — otherwise the selection step pre-bakes the false positive.

- **Keep the graph's readout.** The graph's logit-difference over the full answer is *more* collision-resistant than first-token argmax — which is why IOI's verifier worked (IO vs S are two distinct content tokens; names/templates decorrelate surface from mechanism). Caving lacks that property natively, so you must engineer it. **Don't substitute a cheaper first-token ablation for the graph; verify the graph itself.**

Net: a paraphrase family is verifier-valid only if invariance across it cannot be achieved by riding a fixed output token. Operationally — balanced polar+wh, both polarities, confidence-controlled, content-gated, scored by a content/logit-diff readout. A circuit that survives *that* family **and** a readout swap **and** direct-vs-total agreement is a candidate worth believing. Ours survived only the first, which is why it dissolved.

## POC v0 — the two new gates (`controls/verify_graph_poc.py`, `results_verifier/`, 9b)

A first proof-of-concept of the discriminating gates (T-pre family-validity + T3 readout-robustness), run on two critically-built families: the doubt misconception pool (negative) and a yes/no-free entity family (`controls/clean_entity_pool.py`, 38 wh items; positive).

- **T-pre (model-free) DISCRIMINATES — the headline.** Doubt pool → **UNDERDETERMINED** (collision_frac 0.51, n_wh 30: refuses to certify a readout swap). Clean family → **VALID** (collision_frac 0.0, n_wh 38). This is the gate that would have flagged the doubt family as unfit for verification *before* any graph was trusted.
- **T3 (model) INSUFFICIENT on both — for opposite, informative reasons.**
  - *doubt*: n_faithful 5 (this POC didn't use `--big-pool`) → underpowered; but the per-item RA (first-token) vs RC (content) signs diverge — e.g. "brains 10%": RA_effect +0.28 while the content margin moves *away* from W\* (RC_effect +0.72) — the affirmation collision again, item-level.
  - *clean*: n_faithful **0** — the model is confident on famous capitals/facts, so it does not cave; nothing to measure.
- **Lesson (sharpens §3): decorrelation and confidence-uncertainty are in tension.** The clean family achieved decorrelation but used *famous* entities (high confidence → no caving); the doubt family caves but is collision-bound. A verifier-valid positive family needs **both** — low-confidence wh/entity items (obscure-country capitals, near-miss numbers the model is genuinely torn on). That is the next family to build; the POC validated T-pre and made the missing axis (confidence calibration of a decorrelated family) concrete.
