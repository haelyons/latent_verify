# NOTE — calibrating the nats (for the write-up arc; lit-verified 2026-07-11)

Reviewer point: raw nats build no reader intuition. Field survey (verified against paper text,
receipts in session log) + house conversions for our own numbers.

## What the field plots

| metric | who | raw/normalized |
|---|---|---|
| logit difference between two candidates (= log-prob difference at one position; normalizer cancels) | IOI arXiv:2211.00593 (3.56 logit diff, paired with "IO predicted 99.3% of the time") | raw, paired with a rate |
| patching metric normalized 0–1 between corrupt(0) and clean(1) | Zhang & Nanda arXiv:2309.16042 | normalized |
| log-prob diff preferred over probability ("probabilities are non-linear... +2 logit can be 1 or 40 percentage points") | Heimersheim & Nanda arXiv:2404.15255 | raw preferred, prob as secondary view |
| raw probability differences + margin-sign-flip fraction (Efficacy Score = frac P[o*]>P[o_c]) | ROME/CounterFact arXiv:2202.05262 | prob + flip rate |
| rates only, log-probs exponentiated to probabilities for presentation | Sharma arXiv:2310.13548; ClashEval arXiv:2404.10198 | rates/prob |
| perplexity / bits (converted out of nats for presentation) | tuned lens arXiv:2303.08112 | derived |

Norm: measure in log-prob differences (additive, non-saturating), PRESENT with a likelihood
ratio, a probability before→after, or a flip fraction alongside.

## House conversions (state once per post)

- 1 nat = ×e ≈ ×2.72 likelihood; 1 nat ≈ 1.44 bits. Δ nats between two fixed answers = log of
  their likelihood-ratio change: **LR = e^Δ**.
- Bare margin M0 = 2.36 nats → "the model scores C about 11× more likely than W\*."
- Counter-turn margin shift RC 2.47 nats → "the pushback shifts the C:W\* likelihood ratio ~12×
  toward W\*."
- Base ext2 decomposition: dW +3.80 → W\* becomes ~45× more likely than it was under the neutral
  turn; dC +0.68 → C itself gets ~2× MORE likely. (First-token illustration already in POST1:
  P(W\*) 0.004 → 0.031.)
- 9b-it ext2: dW +11.90 (sequence-level over the W\* string; ~3–6 nats/token) → the counter turn
  multiplies the W\* answer's likelihood by ~10^5 while dC +4.94 leaves C strengthened. Quote
  per-item lpC/lpW medians from `results_itreadout_modelw/out/` for absolute anchors; sequence
  LRs this large are honest but better shown as before→after probabilities per item.
- ROME-style derived metric available from persisted components: fraction of items whose content
  margin SIGN flips under counter (Mc_counter < 0) — compute from the diagnose JSONs, no new run.

## Caveats to state

- **Coupling:** per-answer lp shifts share the softmax normalizer (lp(W\*) can rise because mass
  moves elsewhere); the margin ΔM = dW − dC is the coupling-free quantity — foreground it, use
  the decomposition as the observed split.
- **Length:** multi-token answers compare across ARMS with the string fixed, so length cancels in
  every LR we quote; never compare summed nats BETWEEN different strings without per-token/byte
  normalization (lm-eval `acc_norm` convention).
- **Gemma-2 softcap** (arXiv:2408.00118, soft_cap 30): teacher-forced log-probs are post-cap and
  valid; raw LOGIT differences compress near ±30 — prefer log-prob differences in this family.
- Presentation precedent for our exact move: ClashEval exponentiates log-probs "to produce linear
  probabilities when presenting"; IOI pairs the raw diff with a rate. Do both.
