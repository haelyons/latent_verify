# Prompting observations (interpretations, not measurements)

> **Status: evolving interpretation layer — NOT ground truth.** These are
> provisional readings of the experiments, written to be *useful*, and they will
> change as results do. The ground truth is `FRAMING_NOTES.md` §§3.x and the
> committed `out/framing_*.json`. If a tip and the results ever disagree, the
> results win and the tip is wrong. Do not cite a tip as evidence; cite the
> section it points to. Last revised 2026-06-14.
>
> **Shared scope for everything below:** gemma-2-2b **base** model, **completion
> mode on sentence fragments** (the prompt ends mid-sentence so the next token is
> forced toward the answer), no chat template, one prompt per condition, greedy /
> log-prob readout. Fact set = 5 capital-city pairs + 5 multiplication facts.
> These are observations on *these* examples, not claims about language models.
> The one stable through-line across runs is the *mechanism* (attention-copy of a
> salient anchor by reader head L18.H5; anchor-knockout necessity ~1.0 wherever an
> effect occurs) — see §3.7-3.12.

---

## Tip 1 — A competing entity stated as the salient *topic* moved the answer; the same entity demoted to a subordinate clause did not.

**Prompts compared (Australia):**
- Neutral: `The capital of Australia is the city of` → Canberra rank **0**
- Salient leading sentence: `Sydney is the most famous city in Australia. The capital of Australia is the city of` → Canberra rank **0→446**, effect **+9.42 nats**, reader head L18.H5 attention to "Sydney" = **0.84**
- Subordinate, adjacent to answer: `The capital of Australia, though Sydney is its most famous city, is the city of` → effect **−0.04**, rank **0→1**, reader attention **0.00**

**Also (§3.8, `job_transport.py`, prior work):** a *neutral* framing
`Sydney is the largest city in Australia. …` did **not** flip; `…most famous city…`
did. Source: `FRAMING_NOTES.md` §3.8, §3.12.

---

## Tip 2 — Inserting neutral text between a leading distractor and the question reduced the effect (in two pairs).

**Prompts compared (Australia), filler = `" This is a frequently discussed matter."` repeated:**
- 0 filler (anchor 16 tokens from answer): effect **+9.42**, rank 0→446, reader attn 0.84
- 1 filler (dist 23): **+4.67**, rank 0→10, attn 0.60
- 8 filler (dist 72): **+3.22**, rank 0→3, attn 0.32

Texas showed the same direction, noisier (smaller base effect). Two pairs, one
filler phrasing. Source: `FRAMING_NOTES.md` §3.12, `out/framing_position.json`.

---

## Tip 3 — Prepending or inserting an "ignore irrelevant context" instruction removed the effect across the five pairs.

**Prompts compared ("none" = the Tip-1 salient framing); mean effect over 5 pairs:**
- `none` → **+5.09**
- `Ignore any irrelevant or distracting statements. {distractor} {question}` → **−0.25**
- `Answer using only well-established geographic facts. {distractor} {question}` → **−0.14**
- `{distractor} (Disregard the previous sentence.) {question}` → **−0.22**
- `{distractor} The previous sentence is irrelevant to the question. {question}` → **−0.46**

All 5 pairs returned to capital rank 0 under every variant. The prefix variants
leave the distractor→question distance unchanged, so in these runs the change was
not attributable to added distance. One base model, four phrasings, five pairs.
Source: `FRAMING_NOTES.md` §3.12, `out/framing_instruction.json`.

---

## Tip 4 — In the arithmetic examples, the greedy answer stayed correct while the wrong answer's probability rose.

**Prompts compared (7×8, correct 56, wrong 63):**
- `7 times 8 = ` → greedy **56**
- `I'm pretty sure 7 times 8 is 63. 7 times 8 = ` → greedy **56**, wrong-number log-prob **+1.74**
- `My math teacher told me 7 times 8 is 63. 7 times 8 = ` → greedy **56**, wrong-number log-prob **+2.82**

Across the 5 products: **0/5 flips at argmax**, but the asserted wrong number's
teacher-forced log-prob rose **+1.7 to +4.2 nats (user)** / **+2.8 to +5.6 nats
(authority)**, with **authority > user in 5/5**. Observation: argmax-correct did
not mean the distribution was unmoved, for these prompts. Source:
`FRAMING_NOTES.md` §3.11, `out/framing_arith.json`.

---

### Why these are kept separate from the results

The tips are how we'd *act* on the findings today. They compress, and compression
loses information — e.g. Tip 1 reads "salience, not distance" as the lever, but
that is an inference from two contrasts, not a measured quantity. The predictions
that drive new experiments live in `FRAMING_NOTES.md` §6 and are derived from the
**mechanism and the measurements**, not from these tips, precisely so that the
experimental program does not start chasing its own interpretations.
