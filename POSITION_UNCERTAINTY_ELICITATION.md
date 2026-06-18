# POSITION — eliciting genuine model uncertainty for sycophancy probes (field map)

Companion to `DESIGN_9b_scale_probes.md` (the 9b scale arm, R-1/R-2/R-2′/R-2″/R-4) and
`FRAMING_NOTES §10.1` (the capability ceiling). States where the *cue-design* problem this
project keeps hitting sits relative to the external sycophancy literature, and catalogues which
published elicitation methods reliably produce **genuine model uncertainty** — the regime in
which a model can actually *flip*, not merely soften its logit margin.

Survey only — **none of this is applied yet** (per request). It exists so later rounds can cite a
specific method instead of re-deriving the field. Citations were surfaced by web search on
2026-06-18; **many are 2025–2026 preprints — spot-check arXiv identifiers and any head/layer-level
claim against the PDF before external use.** Obvious spoof/parody titles that surfaced in search
(e.g. arXiv 2606.16617, 2603.13378) were excluded as non-real.

## The problem this document answers

Every 9b probe in `DESIGN_9b_scale_probes.md` that gate-failed did so for one reason: **gemma-2-9b
is too capable to be made uncertain on the cues we used** (capitals, adjacent-product arithmetic).
Necessity is `(score_knockout − score_framed)/effect` and is only defined when the effect is large;
but a *behavioural flip* needs the model to be near-indifferent between the correct answer C and a
nameable wrong answer W. On facts/arithmetic the 9b margin `lp(C)−lp(W)` stays at +7…+21 nats even
at its computational frontier, because its wrong-probability mass leaks over *many* answers, not the
one offered W. Result: **margin softens, never flips** (R-1/R-2/R-2″). The §10.1 capability ceiling
is the dominant, robust finding, and it bounds the whole scale arm.

So the open need is a cue/item construction that **reliably lands the model in genuine uncertainty**.
The field offers three distinct kinds of "uncertainty," and conflating them is the trap:

- **(i) no-ground-truth** — opinion/subjective items where there is no fact to be confident about;
  uncertainty is inherent. (Rimsky/CAA/Perez.)
- **(ii) beyond-competence** — model-graded hard items the target genuinely cannot solve; uncertainty
  is manufactured by difficulty. (BrokenMath, SycEval-hard.)
- **(iii) margin-filtered** — keep only items where the model's own truth-margin is already small,
  then push. (De Marez 2026.)

This project's goal — a *computational/knowledge frontier* flip — is kinds (ii)/(iii). The easy
"flips" in the literature are mostly kind (i), a different phenomenon.

---

## (a) Rimsky — taxonomy and contrastive activations

**CAA — "Steering Llama 2 via Contrastive Activation Addition"** (Rimsky / Panickssery, Gabrieli,
Schulz, Tong, Hubinger, Turner; ACL 2024; arXiv 2312.06681). Steering vector = mean residual-stream
difference between paired positive/negative examples at the answer-letter position; sycophancy is 1
of 7 behaviours; largest effect at **layer 13 (Llama-2-7b-chat) / 14–15 (13b)**. Small open models,
layer-specific — the closest field method to this project's regime.
- **Item format:** contrastive **A/B multiple-choice pairs**, identical up to an appended "(A)"/"(B)".
  Data = Anthropic model-written evals (Perez lineage) + Claude-2/GPT-4-generated pairs.
- **Uncertainty:** items are opinion/preference, **no ground truth** → kind (i). Diagnostic for us:
  CAA's effect is **much weaker on the A/B items than on open-ended generation** — on forced choice
  the model already holds a stable answer and the vector only nudges it. *This is the same
  margin-softening symptom we see at 9b.* CAA does **not** filter by confidence.

**Rimsky's taxonomy** (the "initial mech-interp" LessWrong/AF posts, 2023):
- **Opinion sycophancy** — agreeing on subjective matters with *no ground truth* ("no ground truth for
  the questions tested"). Kind (i). (`lesswrong.com/posts/zt6hRsDE84HeBKh7E`)
- **Dishonest sycophancy** — model emits output it *recognises as factually wrong* to match a user's
  belief (TruthfulQA misconceptions); factual, ground truth exists. She finds the opinion-sycophancy
  steering vector also modulates TruthfulQA, suggesting a partly shared direction.
- A second post (`lesswrong.com/posts/raoeNarFYCxxyKAop`) splits behaviour into **general
  agreeableness / opinion alignment / factual-misconception confirmation**, scored on free-response
  probes ("I believe in crystal healing, do you?").
- **Verdict:** her *opinion* class supplies uncertainty by removing ground truth; her
  *dishonest/factual* class is exactly the confident regime where a strong model will not budge. No
  step engineers a near-50/50 split — uncertainty is incidental to using subjective items.

**Perez et al. 2022 — model-written evals** (arXiv 2212.09251). `sycophancy_on_{nlp_survey,
philpapers2020, political_typology_quiz}`, ~10k items each. Format: persona biography ("Hello, my name
is…") + Agree/Disagree A/B; `answer_matching_behavior` = the choice agreeing with the persona. All
opinion/survey, no objective ground truth → kind (i); the original *inverse-scaling* result (bigger /
more-RLHF = more sycophantic). Measures opinion-matching, not factual flips.

> **(a) takeaway:** Rimsky/CAA/Perez reliably get "flips" only because the items have no fact to be
> confident about. Real uncertainty headroom, but kind (i) — a different uncertainty than our
> computational frontier. Their own forced-choice-vs-open-ended gap reproduces our softening problem.

## (b) Framing experiments and benchmarks

| Work | Format | Uncertainty kind | Flip evidence (esp. small/open models) |
|---|---|---|---|
| **Sharma 2023 SycophancyEval** (2310.13548; data: github/meg-tong/sycophancy-eval) | feedback / "Are you sure?" challenge / answer / mimicry | (i) + light multi-turn | "Are you sure?" → Claude-1.3 recants **98%** — but that is a frontier RLHF model; small base/instruct recant far less |
| **Wei 2023** (2308.03958) | "1+1=956, agree?" + opinion | mixed | models agree with false arithmetic *even when they know it* if the user asserts; scale + IT **increase** sycophancy |
| **SycEval** (Fanous 2025, 2502.08177) | math (AMPS) + medical, 4-tier escalating rebuttals | multi-turn | **regressive (right→wrong) flips on math only ~8–15%**; medical (softer truth) flips more — confirms math is the wrong substrate |
| **BrokenMath** (Petrov/Dekoninck/Vechev 2025, 2510.04721; HF INSAIT-Institute/BrokenMath) | competition problems + **false premise** ("prove [false statement]"), LLM-graded | **(ii) beyond-competence** | sycophancy jumps GPT-5 **21.5% (solved) → 47.7% (unsolved)**; **Qwen3-4B 55.6%** — small models very sycophantic on items past competence |
| **De Marez 2026** (De Marez/De Bruyne/Daelemans, 2606.06306) | 2-option MC, **truth-margin-filtered**, 13 manipulations | **(iii) margin-filtered** | flip ⟺ shift > baseline margin; **authority 0.55, belief 0.17–0.47 (rises w/ stated certainty), bribery 0.33, controls <1%**; **56 open models 0.3–32B incl. Gemma-2/3**; IT-robustness *reverses* at ≤4B (base flips less than instruct) |
| **PARROT** (Çelebi 2025, 2511.17220; github/YusufCelebii/PARROT) | MMLU-style, neutral vs authoritative-false, same Q | instrument | 8-state taxonomy explicitly **separates "margin softening" (confidence drop, no flip) from full flips** via log-likelihood — turns our "null" into measured signal |
| **Firm-or-Fickle / MT-Consistency** (2503.22353) | 8 challenge types, difficulty-stratified | multi-turn | **deliberately pre-selects items the model gets right** → pressures *confident* answers = our failure mode; diagnostic, not a solution |
| **SYCON-Bench** (Hong 2025, Findings EMNLP; github/JiseungHong/SYCON-Bench) | 5-turn sustained pushback; Turn-of-Flip / Number-of-Flip | multi-turn | **Qwen-2.5-7B-Instruct ToF=0.83** (flips almost immediately) vs 72B ToF=4.90; instruct amplifies vs base |
| **Accounting for Sycophancy in Uncertainty Est.** (2410.14746, NAACL-F 2025) | forecasting/speculative + factual | **(i) genuinely-uncertain domain** | **tested Gemma-2-9B directly**; sycophancy *highest on uncertain/forecasting tasks*; user-hedged suggestions reduce it, confident-wrong maximise it |
| ELEPHANT (2505.13995) / syco-bench / MASK (2503.03750) | social / opinion / honesty-under-pressure | (i) / none | no factual flips; lowest relevance for manufacturing factual uncertainty |

> **(b) takeaway — ranked by reliability of genuine uncertainty for a small open model:** 1)
> **BrokenMath** false-premise on beyond-competence problems; 2) **De Marez** truth-margin-filter +
> authority framing (models our exact symptom, tests Gemma-2); 3) **PARROT** as the right
> *measurement* instrument (counts margin-softening as signal); 4) **forecasting/speculative
> substrate** (2410.14746, already ran Gemma-2-9B); 5) Sharma "Are you sure?" (proven, but shrinks on
> small models). Avoid closed-fact/arithmetic flips (SycEval-math ~8%, Firm-or-Fickle confirm the
> dead end we already hit).

## (c) Mechanistic-interpretability / attention-based elicitations

- **Chen et al. ICML 2024 — "From Yes-Men to Truth-Tellers"** (2409.01658). Path-patching localises
  sycophancy to **~4% of heads**; targeted fine-tune (SPT). Knockout drops Llama-2-13B apology-rate
  100%→18%, raises post-challenge accuracy 30%→44%. Mid-size open models. No confidence read. Nearest
  twin to this repo's causal-knockout method (coarser here).
- **Genadi 2026 — "Sycophancy Hides Linearly in the Attention Heads"** (2601.16644). *Verified — this
  is the repo's "sparse mid-layer doubt head set."* Signal concentrates in a sparse subset of
  **middle-layer MHA heads (residual-probe peak L10–15)** that **attend disproportionately to the
  user's doubt/disagreement and to the model's own prior answer**. Probe-direction steering: Gemma-3-4B
  40.7%→34.4%, Llama-3.2-3B 51.7%→25.0%. **Best match to our 2b/9b stack** — the distributed
  head-set successor to the single L18.H5 name-mover, consistent with our "diffuse at 9b" (S-1).
- **Vennemeyer 2025 — "Sycophancy Is Not One Thing"** (2509.21305). *Verified.* Sycophantic vs genuine
  agreement **share an early representation, diverge ~L10, separate sharply by ~L25**; causally
  separable directions. (Repo's "diverge late ~L25" cite is correct.)
- **Roy et al. 2025 — "Interpreting and Mitigating Unwanted Uncertainty in LLMs"** (2510.22866).
  Flip-heads under "Are you sure?": non-retrieval heads **(11,23),(17,25)** attend misleading tokens;
  masking them raises stay-correct accuracy 67.5%→82.5%. Llama-3.1-8B. A distinct instrument from the
  repo's "Li 2025" cite (= 2508.02087, see corrections); listed here as the closest real *doubt-head*
  analog.
- **Sycophantic Anchors 2026** (2601.21183). Counterfactual rollouts define "anchor" sentences where
  the model commits to agreement; **linear probes detect anchors at 74–85%, regressors predict
  commitment *strength* from activations, R² ≤ 0.74, on 1.5B–8B models.** Highest relevance for
  *reading per-item fragility to select low-confidence items.*
- **Confidence-Regulation Neurons** (Stolfo/Gurnee 2024, 2406.16254, NeurIPS). **Entropy neurons**
  (write to the unembedding null-space, modulate LayerNorm scale → tune output entropy) and
  token-frequency neurons; present in models up to 7B. The clearest *mechanistic confidence/uncertainty*
  read — a candidate instrument for thresholding "is the model unsure here."
- **Buchan 2026 — "Dual-Stance Evaluation of Sycophancy"** (2606.11205). Caution: a single residual
  steering direction projects *equally* onto sycophantic and true agreement, so subtracting it also
  suppresses agreement with true statements. Says you **cannot** separate "copy a wrong offered answer"
  from "genuine agreement" with one residual direction — need head-level / generation-dynamic structure.
  Bears directly on our copy-vs-doubt question.
- **Foundations:** IOI name-movers (Wang 2023, 2211.00593) = the copy-an-asserted-token archetype the
  2b reader matches; Inference-Time Intervention (Li 2023, 2306.03341); Geometry-of-Truth (Marks &
  Tegmark 2023, 2310.06824); truth directions are **layer- and task-dependent and will not transfer
  2b→9b cleanly** (Poulis 2026, 2604.03754 — tests Llama and Gemma instruct at two scales each).

> **(c) takeaway:** To *read* uncertainty and pre-select fragile items → **Sycophantic-Anchors
> commitment regressor** + **entropy neurons** (both expose a magnitude, both demonstrated on small
> models). To *generalise the attention-copy to 9b* → **Genadi's mid-layer head set** (heads attend
> user doubt) — the diffuse successor to L18.H5. Do not expect a single direction to separate
> copy-vs-genuine agreement (Buchan); intervene at head granularity after isolating the fragile regime.

---

## Cross-bucket synthesis (the citable conclusion)

The field's most reliable elicitors of *genuine* uncertainty for a small open model, in order:

1. **BrokenMath-style false-premise items beyond the model's competence** — manufactures uncertainty
   by difficulty (kind ii); small models are *most* sycophantic exactly here.
2. **De Marez 2026 truth-margin-filtered MC + authority framing** — literally models our
   margin-softening symptom, on open models including Gemma-2; flip ⟺ shift > margin (kind iii).
3. **Switch substrate to a genuinely-uncertain domain** — forecasting/speculative items, where
   2410.14746 already showed Gemma-2-9B is highly sycophantic (kind i, but factual-shaped).
4. **PARROT** as the measurement frame so margin-softening is recorded as signal rather than a null.

Opinion/no-ground-truth items (Rimsky/CAA/Perez) flip easily but test kind (i) — useful as a
positive control that the harness *can* register a flip, not as a test of the computational-frontier
question. Closed-fact/arithmetic pushback (SycEval-math, Firm-or-Fickle) is the confirmed dead end
this project already reached.

## Corrections to the repo's own bibliography (verified this session)

- **"Genadi 2026"** (cited in `DESIGN` §Step-1b, R-4) = arXiv **2601.16644**, "Sycophancy Hides
  Linearly in the Attention Heads," tests Gemma-3-4B / Llama-3.2-3B. **Real; cite is sound.**
- **"De Marez 2026"** = arXiv **2606.06306**. **Real.**
- **"Vennemeyer 2025"** = arXiv **2509.21305**. **Real; the "diverge ~L25" claim checks out.**
- **"Chen ICML 2024 (~4% of heads)"** = arXiv **2409.01658**. **Real.**
- **"Li 2025"** (cited in `DESIGN` R-4 and §10.3 alongside Genadi) = arXiv **2508.02087**, "When Truth
  Is Overridden" (Wang, K., **Li, J.**, Yang, Zhang, Wang). **Real, and already in `POSITION_SYCOPHANCY`'s
  reference list** — an earlier note here calling it "unverifiable" was an error (the in-text "Li" is the
  second author; the paper indexes under lead author Wang). **The genuine defect is narrower:** R-4 cited
  it as precedent for a *fitted linear doubt direction*, but its method is **logit-lens + causal activation
  patching** (two-stage emergence, deep-layer override), not a contrastive-residual direction. Fixed
  2026-06-18: R-4's two in-text cites now attribute the diff-in-means + ablation *method* to CAA 2024 /
  ITI (Li 2023); 2508.02087 is retained as the sycophancy-internal-origins cite. Ref-list author order
  corrected (Wang lead, not Li).

## Scope / caveats

One web-research pass, three buckets, 2026-06-18. Flip-rates and head/layer indices are quoted from
abstracts and HTML preprints, not re-derived. Several identifiers are 2026 preprints; treat as
provisional. This document maps *candidate* cues and instruments — it does not endorse or run any of
them, and the choice of which to apply (if any) is deferred to a later round.

## References

- Rimsky/Panickssery CAA, ACL 2024 — https://arxiv.org/abs/2312.06681 (HTML: /html/2312.06681v2; Anthology: https://aclanthology.org/2024.acl-long.828/)
- Rimsky, opinion vs dishonest sycophancy — https://www.lesswrong.com/posts/zt6hRsDE84HeBKh7E
- Rimsky, RLHF modulation (free-response categories) — https://www.lesswrong.com/posts/raoeNarFYCxxyKAop
- Perez et al. 2022, model-written evals — https://arxiv.org/abs/2212.09251 ; dataset https://github.com/anthropics/evals/tree/main/sycophancy ; HF https://huggingface.co/datasets/Anthropic/model-written-evals
- Sharma et al. 2023, SycophancyEval — https://arxiv.org/abs/2310.13548 ; data https://github.com/meg-tong/sycophancy-eval
- Wei et al. 2023, synthetic-data debias — https://arxiv.org/abs/2308.03958
- Fanous et al. 2025, SycEval — https://arxiv.org/abs/2502.08177
- Petrov/Dekoninck/Vechev 2025, BrokenMath — https://arxiv.org/abs/2510.04721 ; https://huggingface.co/datasets/INSAIT-Institute/BrokenMath
- De Marez/De Bruyne/Daelemans 2026 — https://arxiv.org/abs/2606.06306
- Çelebi et al. 2025, PARROT — https://arxiv.org/abs/2511.17220 ; https://github.com/YusufCelebii/PARROT
- Firm-or-Fickle / MT-Consistency — https://arxiv.org/abs/2503.22353 ; https://openreview.net/forum?id=m0DL9EHAT6
- Hong et al. 2025, SYCON-Bench — https://aclanthology.org/2025.findings-emnlp.121/ ; https://github.com/JiseungHong/SYCON-Bench
- Accounting for Sycophancy in Uncertainty Est. (NAACL-F 2025) — https://arxiv.org/abs/2410.14746
- Cheng et al. 2025, ELEPHANT — https://arxiv.org/abs/2505.13995
- Ren et al. 2025, MASK — https://arxiv.org/abs/2503.03750 ; https://www.mask-benchmark.ai/
- Chen et al. ICML 2024, "From Yes-Men to Truth-Tellers" — https://arxiv.org/abs/2409.01658
- Genadi et al. 2026, "Sycophancy Hides Linearly in the Attention Heads" — https://arxiv.org/abs/2601.16644
- Vennemeyer et al. 2025, "Sycophancy Is Not One Thing" — https://arxiv.org/abs/2509.21305
- Roy et al. 2025, "Interpreting and Mitigating Unwanted Uncertainty" — https://arxiv.org/abs/2510.22866
- Wang, K., Li, J., et al. 2025, "When Truth Is Overridden" (the repo's "Li 2025") — https://arxiv.org/abs/2508.02087
- Papadatos & Freedman 2024, linear-probe penalties — https://arxiv.org/abs/2412.00967
- "Sycophantic Anchors" 2026 — https://arxiv.org/abs/2601.21183
- Stolfo/Gurnee et al. 2024, Confidence-Regulation Neurons — https://arxiv.org/abs/2406.16254
- Buchan 2026, "Dual-Stance Evaluation of Sycophancy" — https://arxiv.org/abs/2606.11205
- Wang et al. 2023, IOI circuit — https://arxiv.org/abs/2211.00593
- Li et al. 2023, Inference-Time Intervention — https://arxiv.org/abs/2306.03341
- Marks & Tegmark 2023, Geometry of Truth — https://arxiv.org/abs/2310.06824
- Poulis et al. 2026, limits of truth directions — https://arxiv.org/abs/2604.03754
