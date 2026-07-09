# POSITION — sycophancy as an attention-copy, and the offered-answer / bare-doubt boundary

Companion to FRAMING_NOTES §8 (RLHF disengages the salience reader), §10.2 (the
copy *strategy* runs on token-specific circuits), and §11 (the it/chat half).
States where this line of work sits relative to the external sycophancy
literature, and what the harness can and cannot claim about deference.
Citations were surfaced by web search this session; **several are 2025–2026
preprints — spot-check identifiers and head-level claims against the PDFs
before external use.** Per-head semantics are not asserted: we verified causal
role (knockout necessity), not interpretation.

## Headline — SEQUENCE_170626 (2026-06-17)

**TLDR (the corrected headline).** In gemma-2-2b, "framing-induced wrong answers"
are mechanically a **name-mover copy**: one mid-stack head (L18.H5) reads a prompt
token and writes it to the answer — copy-score rank 0/5 (`out/copyscore_2b.json`),
attention-knockout necessity ~1.0 across 5 pairs. **Not induction** (token-recurrence
trap rejected ≤0.015, `out/recurrence_2b_repro.json`); the trigger is token
**position/prominence, not "salience"** (move the anchor off sentence-initial and the
reader jumps to the region token). Post-training deletes the copy at the weights
(head attention 0.84→0.01, §8/§11) — a *named copy head ablated by alignment*, not
previously shown. On confident facts the -it model never caves to content-free doubt;
it entrenches (`out/sycophancy_lowconf_it.json`). Scope: one 2B family, n=5, bf16.

**Positioning (grounds §2 below).** Method niche is **causal attention-knockout**,
not the field's behavioural evals (Sharma 2023, De Marez 2026, BrokenMath, PARROT) or
linear probes / steering vectors (Genadi 2026, CAA 2024, Vennemeyer 2025) and
logit-lens / activation patching (Li 2025).
Nearest twin is Chen (ICML 2024), who path-patches sycophancy to ~4% of attention
heads — same "heads not MLPs" conclusion, coarser knockout here. The field's
doubt-attending mid-layer heads (Genadi 2026) sit in our reader's band but do a
*different* job: ours attends a positionally-planted entity in a **base** model and
**disengages under post-training**. The one genuinely novel claim is that alignment
training ablates a *named* copy head; the rest corroborates and sharpens existing
work. The §11 capability-ceiling confound *is* the field's central variable (De
Marez's truth-margin; PARROT's "compliant where least certain").

**SEQUENCE corrections to the prior framing (do not skip).** (1) "salience reader" is
**position-confounded** — the cleaner salience×position 2×2 is unrun, so call L18.H5 a
*position/prominence-gated OV-copy head*, not a salience reader. (2) "concentrated
salience vs diffuse numeric" is fully retracted — the 208-head sweep
(`out/localize_salience_208_2b.json`) shows salience is *also* distributed; only
L18.H5's cross-pair consistency survives. (3) The `counter`/`bare` circuit
dissociation (§4) was **attempted, not delivered**: the two caving items are the two
the model already got wrong single-turn, so copyable-anchor and prior-lean are not
separated. The robust result is *no caving from a held correct belief, even at a
+0.06-nat margin*. (4) [2026-07-09] Every "deletes the copy **at the weights**" /
"structurally absent" phrasing in this doc (L20, §3, §8-refs, the SFT-vs-RLHF note) is
**superseded by the weight-level battery** (RESEARCH_QUESTIONS claim 5): QK routing is
intact at every scale (2b `qk_dir_cos 0.998`, 27b `W_QK` unchanged), the 27b OV write
direction is preserved with rescaled gain (`dir_cos 0.999`, α 1.33 — latent), and the 2b
realized-attention collapse (0.578→0.016) is **residual-INPUT-mediated** (swapping the
base residual input recovers 0.973). Read "deletes the copy" throughout as *functionally
silences via the head's input; the weights persist* → `results_2b_qkweight*/`,
`results_27b_qk/`, `results_27b_ovmag/`, `results_27b_realattn/`.

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

**Scale scope — the named single reader is 2b-specific (2026-06-17, FRAMING §10.3).**
The whole line was re-run on **gemma-2-9b** base + -9b-it, re-localizing every head from
scratch (672 heads). The concentrated attention-copy *reader* does **not** transfer: at 9b the
salience effect collapses behaviorally (mean ≈0, per-pair ±0.5–1.25 cancelling) and dissolves
mechanically — attention-to-anchor, OV-copy capacity, and causal necessity (all unified in the
one 2b head) are carried by **different** 9b heads, none causal (top necessity 0.14; 79/672
heads OV-copy the anchor but the attending head is not among them; max-attn head at necessity
rank 112). What *does* transfer is the qualitative character: still name-mover-not-induction
(induction 0.02, decoy ≤0.0085), still position-fragile, and -it still does not cave (counter
−4.06, bare −1.19) — though the 9b-it caving test is **vacuous under a capability ceiling**
(the 2b-calibrated lowconf set saturates: 7/8 pre-margins +3.2…+7.4). So the "deference is an
attention-copy on a *named* reader head" claim should be scoped to small models; the
*strategy*-level claim (where a copy occurs it is name-mover, content-routed, not recurrence)
is what generalizes. The `counter`/`bare` circuit dissociation (§4) needs a
9b-uncertainty-calibrated item set to be testable at scale.

**Adversarial verification (2026-06-18, `latent_skeptic`@`a8fc434`).** The two load-bearing recent claims
were triaged (fresh independent skeptics, one confound each; verify-by-running, not -reading). **C1**
(post-training is QK-gated / OV-preserved, FRAMING §8): **robust** — a magnitude control (`ov_norm_probe.py`)
shows the L18.H5 OV write is preserved in *magnitude* (write-norm −0.5%, pre-LN anchor logit −1.1%), not only
direction, closing the saturated-metric crux. **C3** (the concentrated single-head copy does not transfer to
9b, FRAMING §10.3): **core robust, sharpened** — the numeric copy's diffuse+decoupled signature is
item/phrasing/scale-robust (held-out products × 3 phrasings × bootstrap; 9b top1 0.072, 2b 0.107), but the
concentrated reader that fails to transfer is **salience-specific** (L18.H5); the 9b-*salience* per-head
necessity is noise-dominated (matched control ≈ signal) and cannot itself carry "dissolves mechanically."
Controls + result JSONs are committed; the skeptic harness is **pinned, not vendored** — its SHA above is the
record, the tool itself stays out of this repo.

## References

*Identifiers from this session's web search; spot-check before external use.
Years as listed by the sources; arXiv IDs of the form 26NN.* are recent
preprints relative to the 2026-06 work date.*

- Sharma, M., Tong, M., Korbak, T., et al. (2023). Towards Understanding Sycophancy in Language Models. arXiv:2310.13548 (ICLR 2024).
- Perez, E., Ringer, S., Lukošiūtė, K., et al. (2022). Discovering Language Model Behaviors with Model-Written Evaluations. arXiv:2212.09251 (Findings of ACL 2023).
- Wei, J., Huang, D., Lu, Y., Zhou, D., & Le, Q. V. (2023). Simple Synthetic Data Reduces Sycophancy in Large Language Models. arXiv:2308.03958.
- Chen, W., Huang, Z., et al. (2024). From Yes-Men to Truth-Tellers: Addressing Sycophancy in LLMs with Pinpoint Tuning. arXiv:2409.01658 (ICML 2024).
- Genadi, R., Nwadike, M., Mukhituly, N., Alquabeh, H., Hiraoka, T., & Inui, K. (2026). Sycophancy Hides Linearly in the Attention Heads. arXiv:2601.16644.
- Wang, K., Li, J., Yang, S., Zhang, Z., & Wang, D. (2025). When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models. arXiv:2508.02087. *(method: logit-lens + causal activation patching, not a fitted doubt direction; cite for sycophancy's internal origins, not for the R-4 instrument.)*
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
