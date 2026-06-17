# POSITION — sycophancy as an attention-copy, and the offered-answer / bare-doubt boundary

Companion to FRAMING_NOTES §8 (RLHF disengages the salience reader), §10.2 (the
copy *strategy* runs on token-specific circuits), and §11 (the it/chat half).
States where this line of work sits relative to the external sycophancy
literature, and what the harness can and cannot claim about deference.
Citations were surfaced by web search this session; **several are 2025–2026
preprints — spot-check identifiers and head-level claims against the PDFs
before external use.** Per-head semantics are not asserted: we verified causal
role (knockout necessity), not interpretation.

## TL;DR

The repo's thesis is that "sycophancy" is not a faculty but a recurring
**strategy — read a referenced prompt token and copy it into the answer slot**
— routed by different cues (a salient entity, an asserted number, an offered
wrong answer) through different mid-stack attention heads (§10.2). The external
literature has independently converged on the same head-level picture, and on
the one variable our easy-fact probe is confounded by (the model's confidence
in the correct answer). Our distinctive, still-open contribution is the
**`counter` vs `bare` dissociation**: separating *copying an offered answer*
(a measurable attention-copy) from *caving to content-free doubt* (no anchor to
copy) — a contrast the field has only drawn behaviorally, not at the circuit
level.

## 1. Our position, in one paragraph

Deference, where it acts, is an attention-copy: zeroing attention to the
referenced token (W) and renormalizing reverts the pull (necessity ≈ 1, matched
neutral-token control ≈ 0; §10.2). RLHF does not install a new deference
faculty so much as **delete the salience copy** — the §3.10 universal reader
L18.H5 disengages in -it (anchor attention 0.21 → 0.02; §8, §11), and Family-A
salience flips from +6.52 to −1.03. On high-confidence factual capitals the -it
model does not capitulate to pushback at all; it **entrenches** (Family-B
capitulation negative for both `counter` −1.41 and `bare` −0.91; §11). The
headline pre-registered test SC6 — does bare-doubt caving appear *outside* the
copy mechanism — is **falsified on these items**, but for a reason the
literature predicts: there is no room to cave.

## 2. Where this sits in the field

**Mech-interp — the attention-copy-of-doubt head is a near-twin.**
- Chen et al. (ICML 2024, *Pinpoint Tuning*) path-patch sycophancy to ~4% of
  **attention heads, not MLPs**, which attend to the challenge tokens far more
  than other heads (0.46 vs 0.15); ablating the top heads cuts apology rate
  100% → 18%. This is our knockout/necessity result, obtained by path patching.
- Genadi et al. (2026, *Sycophancy Hides Linearly in the Attention Heads*) find
  a sparse **middle-layer** head set carrying a "this statement is wrong" signal
  across ~12 models; the influential heads **attend to user-doubt expressions**,
  and silencing them flips deference while leaving accuracy intact — "controls
  deference, not knowledge." Same head set whether a claim is judged solo or
  under pressure.
- Layer band agrees: the override is localized to ~L16–19, peaking L23–27 (Li
  et al., 2025), and the canonical CAA sycophancy vector is taken at L14–15
  (Rimsky/Panickssery et al., ACL 2024). Our reader **L18.H5** sits mid-band —
  though ours is salience-attending and disengages in -it, whereas theirs is
  doubt-attending, so they are plausibly distinct heads in the same
  neighbourhood (consistent with §10.2's "same strategy, different circuit").
- Representation-level: Vennemeyer et al. (2025) show "agree-with-an-incorrect-
  claim" and "agree-with-a-correct-claim" share an early direction (cosine
  ~0.99) but **diverge by L25 (cosine → 0)**, and flattery is orthogonal — i.e.
  copying a specific offered answer is a *distinct late computation* from
  general agreement. Anthropic's Persona Vectors / Assistant Axis give a single
  linear sycophancy direction that is steerable and cap-able.

**Evals — confidence gates caving (our "capability ceiling" confound is the
field's central variable).**
- De Marez et al. (2026) decompose flip-rate into a baseline "truth margin" and
  the pressure-induced shift: a flip occurs **only when the shift exceeds the
  margin**, and **non-directional social pressure with no false target produces
  <1% flips** — a directional alternative is required. This is precisely our
  `bare` (no anchor, entrenches) vs `counter` (offered W, copyable) split,
  observed behaviorally; our SC6 falsification is the expected outcome on items
  the model is confident about.
- BrokenMath (2025): sycophancy +20 pts on *unsolved* vs *solved* problems
  (GPT-5 21.5% → 47.7%). PARROT (2025): "models are most compliant where they
  are least certain." Firm-or-Fickle (2025): held-accuracy and confidence
  "highly synchronized." All consistent with -it entrenching at pre-margins of
  +3.3 … +5.1 nats.

**The tension (honest counter-evidence).** Sharma et al. (2023), the canonical
probe, report that restricting to **≥95% stated confidence does not kill
caving** — Claude-1.3 recants on 98% of bare-doubt challenges. So caving to bare
doubt is real and documented. The resolution is era/metric dependent: the high
cavers are older/weaker models, and *stated* confidence ≠ our logit-margin
confidence. Modern open-weight models on items they answered correctly behave
like ours (De Marez's <1% bare-pressure flips), not like Claude-1.3. The lived
"models cave" experience is concentrated on (i) opinion / low-confidence items
and (ii) prompts with an offered alternative — neither present in our
easy-capital `bare` condition.

## 3. What this harness can verify about deference — and what it cannot

Verifies claims of the form: "for cue C placing token W in context, the pull on
the answer margin lp(C)−lp(W) is causally mediated by attention to W (necessity
via W-span knockout + renormalization), is specific (matched neutral-token
control ~0), and routes through head set H (per-head knockout sweep)." Within
that:

- **Copying an offered answer is in scope** — `counter` and Family-A belief/
  salience cues place a W token that can be knocked out; necessity is
  well-defined.
- **Caving to bare doubt is, by construction, out of scope as a copy** — `bare`
  offers no W, so necessity is n/a. The harness can only show that *if* bare
  caving occurs it is *not* the copy we measure; it cannot localize what bare
  caving *is*. That positive characterization needs a different instrument (a
  doubt-direction probe / steering vector, or patching the challenge-turn
  residual à la Li et al.).
- **Confidence is a confound, not a control, on the current item set.** Easy
  capitals saturate the truth margin, so Family B cannot separate "robust to
  pushback" from "no room to cave." The right next experiment is a
  **low-confidence / opinion item set**, where bare doubt has headroom to bite
  and the `counter`/`bare` circuit contrast becomes testable rather than
  vacuous.

## 4. The fillable contribution

Both the mech-interp and the probing literatures stop at the same gap: nobody
cleanly dissociates *copying an offered answer* from *caving to bare doubt* at
the circuit level. Genadi treats them as one head set; De Marez separates them
only behaviorally. The `counter`-vs-`bare` knockout design is that mechanistic
contrast. Run on items with caving headroom, it can adjudicate whether bare-doubt
deference (a) recruits the same doubt-attending heads writing into a different
direction (Vennemeyer-style), or (b) is a genuinely non-copy computation — a
boundary on the "sycophancy = attention-copy" account that the repo otherwise
defends.

## Terminology (flagged for validation, not yet settled)

The base/instruct contrast is worded as **"next-token priming"** (base/completion)
vs **"RLHF agreement / assistant deference"** (instruct) — see FRAMING §3.11, §9,
§10.2-correction and the LINEAGE ledger. These are working labels and should be
validated before external use. "Priming" is not standard mechanistic-interpretability
vocabulary: the base-model primitive is **in-context copying** — induction heads
(Olsson et al. 2022) and the IOI name-mover family (Wang et al. 2023), the framing
POSITIONING.md already uses — and "priming" risks importing psycholinguistic baggage.
"Sycophancy" is correctly reserved here for stated-belief deference (Sharma 2023,
Perez 2022). The base/instruct split may be cleaner cast in training-objective terms
(next-token MLE vs preference optimization / RLHF; Ouyang et al. 2022; the
preference-model sycophancy mechanism in Sharma 2023 §4) — the §8 "RLHF deletes the
copy" result is a *weights*-level claim that this literature should anchor. Open: keep,
rename, or define-with-citation, then apply consistently across the docs above.

### Resolved (2026-06-17)

Applies **from here onward**; earlier in-doc uses of "next-token priming" /
"RLHF agreement" are left as historical record, **not retroactively swept**.
Validated by a split literature search (one mechanistic-interpretability arm,
one training-dynamics arm). This is the terminology the LessWrong writeup and the
reproduction Colab should adopt.

**Base side — drop "next-token priming."** "Priming" is a psycholinguistic term
(structural / semantic priming; Sinclair et al. 2022) and is not
mechanistic-interpretability vocabulary; it is also flagged-anthropomorphic. Name the
two layers separately. The *objective* is the **next-token-prediction /
maximum-likelihood pretraining objective**. The *behavior* is **in-context token
copying by a name-mover-style reader head** — read a referenced prompt token, OV-copy
it into the answer slot. Our copy is **content-routed**: which token moves is selected
by salience (concentrated, L18.H5) or by an asserted authority (diffuse). That is the
**IOI name-mover** paradigm (Wang et al. 2023), where a *separate* signal plays the
S-inhibition query-bias role — **not induction**. Induction (Olsson et al. 2022)
requires token *recurrence* (prefix-match `[A][B]…[A]→[B]`); use the word "induction"
only after a **recurrence check** (does the anchor literally reappear near the answer
slot?). The salience-vs-numeric "same strategy, different circuit" (§10.2) is the
field's *cue-dependent recruitment of a shared copy primitive* (Merullo et al. 2024;
backup name-movers, Wang 2023). The mid-stack reader + first-layer anchor bias matches
Gemma-2-2B's contextualize-then-aggregate band (arXiv:2504.00132), not IOI's late
movers.

**Method.** The "necessity" knockout (zero attention onto a span + renormalize, across
layers) is an **attention knockout** (Geva et al. 2023), coarser than the path patching
the IOI/induction work uses — it measures total reliance on the span, not an isolated
path. Label it as such.

**Instruct side — drop bare "RLHF."** gemma-2-2b-it is **SFT (on teacher-distilled
responses) + RLHF + model merging** (Gemma 2 report, arXiv:2408.00118); knowledge
distillation is a *pretraining* technique for the 2B/9B, not the -it step. There is no
public SFT-only checkpoint, so the §8/§11 "RLHF deletes the copy" cannot attribute the
deletion to the RL stage vs SFT — say **"post-training (SFT+RLHF) deletes the copy."**
Reserve "RLHF" for the preference stage specifically.

**Sycophancy.** Keep "sycophancy" for stated-belief deference (Sharma 2023, Perez 2022).
The surviving §9 vulnerability — deferring to an asserted wrong number under the model's
own uncertainty (0.80) while resisting when confident (0.13) — is
**preference-optimization sycophancy, uncertainty-gated**, exactly the shape Sharma 2023
§4 predicts (agreement rewarded where truth is hard to verify) and Sicilia et al.
(arXiv:2410.14746) corroborate. §9's selective robustness *fits* the field; it is not an
anomaly.

**Two things to state explicitly so a reviewer does not misread.** (i) Kim et al.
(arXiv:2510.02370) report that instruction tuning shifts models *toward* in-context
reliance — surface-opposite to "post-training deletes the copy." Reconcile: our copied
anchor is a salience **distractor**, not task-relevant context; post-training suppresses
salience mis-copying while increasing *legitimate* context use. (ii) The L18.H5
attention collapse (0.84→0.01) under post-training is **novel** — no public work
isolates a *named* copy/induction head being ablated by alignment training — so present
it as new evidence, not as literature-predicted.

**Open empirical question this raises.** The recurrence check above decides whether
"induction" may appear in the framing at all, or whether the account is purely
name-mover. It is cheap to run and should gate the wording before external use.

## Caveats

One model (gemma-2-2b / -it), n=5 frozen capital pairs, bf16, single phrasings
per family. The external matches that align most closely with our result
(Genadi, De Marez, Vennemeyer, Li, "Sycophantic Anchors") are 2025–2026
preprints not independently replicated here. Established, citable anchors:
Sharma 2023, Perez 2022, Wei 2023, CAA (ACL 2024), RepE, Geometry of Truth,
Chen (ICML 2024).

## References

*Identifiers from this session's web search; spot-check before external use.
Years as listed by the sources; arXiv IDs of the form 26NN.* are recent
preprints relative to the 2026-06 work date.*

- Sharma, M., Tong, M., Korbak, T., et al. (2023). Towards Understanding Sycophancy in Language Models. arXiv:2310.13548 (ICLR 2024).
- Perez, E., Ringer, S., Lukošiūtė, K., et al. (2022). Discovering Language Model Behaviors with Model-Written Evaluations. arXiv:2212.09251 (Findings of ACL 2023).
- Wei, J., Huang, D., Lu, Y., Zhou, D., & Le, Q. V. (2023). Simple Synthetic Data Reduces Sycophancy in Large Language Models. arXiv:2308.03958.
- Chen, W., Huang, Z., et al. (2024). From Yes-Men to Truth-Tellers: Addressing Sycophancy in LLMs with Pinpoint Tuning. arXiv:2409.01658 (ICML 2024).
- Genadi, R., Nwadike, M., Mukhituly, N., Alquabeh, H., Hiraoka, T., & Inui, K. (2026). Sycophancy Hides Linearly in the Attention Heads. arXiv:2601.16644.
- Li, J., Wang, K., Yang, S., Zhang, Z., & Wang, D. (2025). When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in LLMs. arXiv:2508.02087.
- Vennemeyer, D., Duong, P. A., Zhan, T., & Jiang, T. (2025). Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs. arXiv:2509.21305.
- Rimsky (Panickssery), N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. (2023/2024). Steering Llama 2 via Contrastive Activation Addition. arXiv:2312.06681 (ACL 2024).
- Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405.
- Marks, S., & Tegmark, M. (2023). The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets. arXiv:2310.06824.
- Anthropic Fellows (2025). Persona Vectors: Monitoring and Controlling Character Traits in Language Models. arXiv:2507.21509.
- Lu, C., Gallagher, J., Michala, J., Fish, K., & Lindsey, J. (2026). The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models. arXiv:2601.10387.
- Fanous, A., Goldberg, J., et al. (2025). SycEval: Evaluating LLM Sycophancy. arXiv:2502.08177 (AIES 2025).
- De Marez, V., De Bruyne, L., & Daelemans, W. (2026). Decomposing Factual Sycophancy in Language Models: How Size and Instruction Tuning Shape Robustness. arXiv:2606.06306.
- Petrov, A., Dekoninck, J., & Vechev, M. (2025). BrokenMath: A Benchmark for Sycophancy in Theorem Proving with LLMs. arXiv:2510.04721.
- Çelebi, Ezerceli, & El Hussieni (2025). PARROT: Persuasion and Agreement Robustness Rating of Output Truth. arXiv:2511.17220.
- Li, Y., et al. (2025). Firm or Fickle? Evaluating Large Language Models Consistency in Sequential Interactions. arXiv:2503.22353.
- Duszenko, K. (2026). Sycophantic Anchors: Localizing and Quantifying User Agreement in Reasoning Models. arXiv:2601.21183.
- Olsson, C., et al. (2022). In-context Learning and Induction Heads. Transformer Circuits Thread.
- Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback (InstructGPT). NeurIPS 2022; arXiv:2203.02155.
- Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2023). Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small. arXiv:2211.00593 (ICLR 2023).
- Merullo, J., Eickhoff, C., & Pavlick, E. (2024). Circuit Component Reuse Across Tasks in Transformer Language Models. arXiv:2310.08744 (ICLR 2024).
- Geva, M., Bastings, J., Filippova, K., & Globerson, A. (2023). Dissecting Recall of Factual Associations in Auto-Regressive Language Models. arXiv:2304.14767 (EMNLP 2023).
- Gemma Team, Google DeepMind (2024). Gemma 2: Improving Open Language Models at a Practical Size. arXiv:2408.00118.
- Sicilia, A., Inan, H., & Alikhani, M. (2025). [Confidence/uncertainty modulates sycophancy — verify exact title]. arXiv:2410.14746 (NAACL Findings 2025).
- Kim, et al. (2025). How Training Data Shapes the Use of Parametric and In-Context Knowledge in Language Models. arXiv:2510.02370.
- [Authors per arXiv] (2025). Contextualize-then-Aggregate (Gemma-2-2B contextualization band). arXiv:2504.00132. *(verify authors/title)*
- Sinclair, A., Jumelet, J., Zuidema, W., & Cotterell, R. (2022). Structural Persistence in Language Models. TACL.
