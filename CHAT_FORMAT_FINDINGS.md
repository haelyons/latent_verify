# Chat-format / model confound — results (FRAMING_NOTES §6, predictions P1–P5)

> **Ground-truth results doc.** Raw artifacts: `out/chat_base.json`,
> `out/chat_it.json`, `out/chat_base.log`, `out/chat_it.log`. Tool: `chat_exp.py`.
> Run 2026-06-14 on CPU via plain `transformers` (no transcoders). Scope: 5
> capital-city pairs + 5 multiplication facts, one phrasing each, gemma-2-2b
> {base, it}, greedy decode / teacher-forced log-prob. Small N — observations on
> these examples.

## Headline

The dramatic salience flip of §§3.2–3.12 is **largely specific to base-model
*fragment completion*.** In the regime people actually use — a full-sentence
question, generated answer — **neither model gives the wrong city**, and the
instruction-tuned model additionally (a) shows no logit-level pull and (b)
actively rebuts the false premise. The mechanism (§3.7–3.10) is real where we
measured it; its *behavioural* reach to deployed usage is narrow.

## The 2×2 — does the salience framing flip the answer?

Behavioural readout = does the model's answer give the distractor city as the
capital? (presence-based; see "classifier caveat").

| | fragment completion (`…is the city of`) | full-question, generated answer |
|---|---|---|
| **base** (gemma-2-2b) | **flip: +9.42 nats, Canberra→rank 446** (§3.x) | **0/5 flips** — answers the capital |
| **it** (gemma-2-2b-it) | n/a (chat model not run in fragment mode) | **0/5 flips** — answers the capital and rebuts |

Example `-it` reply under "Sydney is the most famous city in Australia. What is
the capital of Australia?": *"You're right that Sydney is a very famous city in
Australia, but it's not the capital! The ca[pital…]"* — i.e. correction, not
capitulation, on all 5 pairs.

## Latent logit pull (teacher-forced at the fragment stem `…is the city of`)

The same readout position §3.x used, scored as logp(capital) − logp(distractor),
neutral→salience shift (higher = more pull toward the distractor):

| model | short lead-in | long lead-in |
|---|---|---|
| base | **+6.57** | +7.23 |
| it | **−0.83** | −0.14 |

So at the fragment readout the **base model still carries a large salience pull
(~+6.6 nats)** even though it answers correctly in QA generation — the pull is
present in the logits but doesn't dominate a full-sentence answer. The
**instruction-tuned model shows ~0 pull**: RLHF appears to have removed the
susceptibility at the readout, not merely added a refusal layer.
(Caveat: the "long" lead-in wording contains "famous", so the short↔long contrast
is contaminated and the distance prediction P2 is **not** cleanly tested here.)

## Arithmetic (P4)

| | base fragment (§3.11) | base QA generated | it chat |
|---|---|---|---|
| baseline correct | 5/5 | 5/5 | 5/5 |
| capitulations (user / authority) | 0 / 0 (graded logit pull only) | 0 / 0 | 0 / 0 |

The `-it` model corrects the false assertion explicitly: *"Your math teacher made
a mistake! 7 times 8 is actually 56."* No capitulation in any cell.

## Prediction outcomes

- **P1 (salience flip transfers to chat): NOT SUPPORTED.** The behavioural flip
  does not appear for a full-sentence question, in either model (0/5).
- **P2 (distance attenuation): SUPPORTED** (after fixing the contaminated
  control) — see follow-up below: neutral filler attenuates the latent pull in
  5/5 pairs (mean +6.45 → +3.42).
- **P3 (instruction ≥ base): UNINFORMATIVE here** — the it model already shows 0
  flips under plain salience, so the "ignore" instruction had no flip to prevent
  (it just made replies terse). Can't compare strengths when the base rate is 0.
- **P4 (RLHF adds an arithmetic flip): FALSIFIED.** The it model is 5/5 correct
  and corrects the asserter; 0 capitulations.
- **P5 (factor isolation): partially answered.** For the *behavioural* outcome the
  **format factor dominates** — moving base from fragment to full-question removes
  the flip on its own; the **model factor (RLHF) additionally removes the latent
  logit pull** (+6.6 → ~0) and adds active rebuttal.

## Follow-ups (base, HookedTransformer; `base_attn_qa.py`, `out/base_attn_qa.json`)

Two questions left open above, run on base gemma-2-2b via `HookedTransformer`
(`from_pretrained_no_processing`, bf16). Readout = logp(capital) − logp(distractor)
at the `…is the city of` position; effect = neutral − salience (higher = more pull
toward the distractor).

**(a) Distance, with a salience-word-free filler.** Re-running the §3.12 distance
sweep with neutral filler (`" This is a frequently discussed matter."`, no "famous"):

| pair | short (0 filler) | long (+8 filler) |
|---|---|---|
| Australia | +10.00 | +5.50 |
| Texas | +6.62 | +2.12 |
| Canada | +5.88 | +2.62 |
| Switzerland | +6.88 | +5.62 |
| Morocco | +2.88 | +1.25 |
| **mean** | **+6.45** | **+3.42** |

Attenuates in **5/5** pairs (~47% at +8 filler). So the earlier non-attenuation
was the contaminated control; distance genuinely reduces the latent pull,
consistent with §3.12.

**(b) Is the QA-format pull still attention-copy-mediated — or disengaged?**
Salience effect and all-heads anchor-knockout necessity (§3.7) in the bare
fragment vs the QA scaffold (`Question: …? Answer: The capital of … is the city of`):

| pair | bare effect | bare knockout nec | QA effect | QA knockout nec |
|---|---|---|---|---|
| Australia | +10.00 | +1.08 | +0.50 | n/a (<0.5) |
| Texas | +6.62 | +1.17 | +1.00 | +0.38 |
| Canada | +5.88 | +1.11 | +0.06 | n/a |
| Switzerland | +6.88 | +1.04 | +1.12 | +1.11 |
| Morocco | +2.88 | +1.13 | +0.12 | n/a |
| **mean** | **+6.45** | ~1.10 | **+0.56** | — |

**The QA scaffold disengages the copy.** The latent pull collapses from mean
**+6.45 to +0.56** (~91%) at the *same* readout position — 3/5 pairs fall below
the 0.5-nat floor entirely. So the base model answers QA questions correctly not
because the copy is engaged-but-overridden downstream, but because the
question-form scaffold **redirects attention away from the anchor at the readout
itself**. Where a small residual pull survives (Texas, Switzerland) it is still
attention-mediated (necessity +0.38 / +1.11). In the bare fragment the knockout
reproduces §3.7 (necessity ~1.04–1.17 across all 5) — an independent replication.

**End-to-end picture of the salience flip:**

    fragment completion  -> copy fully engaged (~+6.5..+10 nats, knockout ~1.0) -> WRONG city
    QA scaffold (base)   -> copy disengaged at readout (~+0.6 nats)             -> correct
    chat model (it)      -> no latent pull (~0) + active rebuttal               -> correct + correction

One mechanism, three regimes: the dramatic flip needs the fragment that forces a
city next-token; question-form prompting alone mostly switches the copy off, and
RLHF removes the residue and adds correction.

## Caveats

- **Classifier artifact (found and fixed).** The first version classified by
  first-entity-mention / leading integer; this mislabeled verbose `-it` replies —
  a rebuttal that names the distractor first ("…While Houston is a major city…")
  was scored a flip, and "7 times 8 is 56" parsed as "7". Switched to
  presence-based (flip = correct answer absent AND wrong present) with 60-token
  generations; numbers above are post-fix. The first-pass summary said `-it`
  flipped 5/5 — that was the artifact, not the model.
- Base in QA mode uses a "Question:/Answer:" scaffold, itself a mild framing.
- Small N, one phrasing per item, one 2B model family, greedy decode. The
  mechanism work (§3.7–3.10) is unaffected by this; what is bounded here is the
  *behavioural transfer* of the flip to question-answering / chat usage.
