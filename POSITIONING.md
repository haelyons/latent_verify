# POSITIONING — the results in plain terms, from first principles

Companion to RESULTS.md: what the four criteria mean, why the experiment is
shaped the way it is, where the outcomes sit in the literature, and what
class of concepts this harness can and cannot verify. Citations from
training knowledge; spot-check identifiers before external use. "[PIE]"
remains the project-internal handoff reference and is deliberately not
attributed to an external publication. Per-feature semantics (e.g. what
L20/15589 "means") are deliberately not asserted: we verified causal role,
not interpretation.

## TL;DR

We took a published, correlational description of how gemma-2-2b answers a
two-hop factual question (the attribution graph for "the capital of the
state containing Dallas is" -> "Austin") and subjected its central claim —
that six specific transcoder features carry the latent "Texas" step — to
four pre-registered causal tests. The claim survived three decisively
(causal necessity, specificity, transport across paraphrases) and returned
a structured "it's complicated" on the fourth (redundancy): the six
features act as an ensemble with one dominant member and at least one
compensating member, so single-feature tests systematically under-measure
importance.

## 1. Why the experiment is shaped this way

**Neurons are the wrong unit.** Occasionally single neurons encode single
concepts — the sentiment neuron (Radford et al., 2017), knowledge neurons
(Dai et al., 2022) — but systematically they are polysemantic (Olah et
al., 2020), explained by superposition: models store more concepts than
they have neurons, as non-axis-aligned directions (Elhage et al., 2022).
Apparent monosemanticity can itself be an analysis artifact (Bolukbasi et
al., 2021).

**Features via dictionary learning.** Sparse autoencoders/transcoders
re-express activations as sparse combinations of learned directions that
are empirically far more monosemantic than neurons (Bricken et al., 2023;
Templeton et al., 2024). Transcoders replace an MLP's input->output map
with a sparse interpretable bottleneck (Dunefsky et al., 2024); this
project uses the GemmaScope per-layer transcoders (Lieberum et al., 2024).
The unit of analysis is a feature, not a neuron.

**Attribution graphs are hypotheses, not findings.** Circuit tracing
substitutes transcoders for MLPs, freezes attention patterns and
normalization, and traces linear attributions between active features for
one prompt (Ameisen et al., 2025), applied at scale in Lindsey et al.
(2025), with graphs on Neuronpedia and reproducible via the open-source
circuit-tracer library (Hanna & Piotrowski et al., 2025). The authors are
explicit that a graph is a descriptive, linearized account of a single
forward pass — frozen attention, reconstruction error swept into error
nodes — to be validated by interventions.

**Verification means causal intervention.** The standard for converting
descriptions into causal claims: intervene on the proposed mediator and
measure behavior — causal mediation analysis (Vig et al., 2020; Pearl,
2001), causal tracing in ROME (Meng et al., 2022), interchange
interventions / causal abstraction (Geiger et al., 2021, 2022), causal
scrubbing (Chan et al., 2022), activation-patching practice (Heimersheim
& Nanda, 2024), ablation-validated sparse feature circuits (Marks et al.,
2024). T0/T1 is this step, applied to a published graph, with criteria
fixed before the run.

## 2. What each criterion asks, and what we found

**S1 — is the story causally real at all?** Jointly clamping all six
features (at every firing position, to -2x observed activations) collapses
Austin's logit by 24.5 and demotes it from rank 1 to ~rank 88,000 of a
256k vocabulary. The graph's central nodes are causally load-bearing —
pass, decisively.

**S2 — one feature or an ensemble?** Best single feature: 35% of the joint
effect at m=-2 (50% at m=0); five of six singles read ~0 under
zero-ablation; one (L19/7477) reads slightly negative — suppressing it
alone mildly helps Austin. Pre-registered band: ambiguous. Structure: a
redundant ensemble with one dominant member (L20/15589) and active
compensation. Practical lesson — single-feature ablations under-measure
importance; adoption rules must intervene on sets, the same lesson causal
scrubbing draws about joint hypotheses (Chan et al., 2022).

**S3 — specific mechanism, or just brain damage?** Large interventions can
move behavior nonspecifically — the interpretability-illusion failure mode
for subspace patching (Makelov, Lange & Nanda, 2023; cf. Bolukbasi et al.,
2021). Control: six random activation-magnitude-matched features, same
layers, same positions. Mean control drop 0.95 vs 24.5 — ~26x separation.
Pass: the effect belongs to these features, not to perturbation size.

**S4 / T1 — one prompt, or a mechanism?** Graphs are per-prompt by
construction (a stated limitation in Lindsey et al., 2025); circuits found
on one template can fail to generalize (Lieberum et al., 2023; Hanna et
al., 2024). Sixteen pre-frozen paraphrases (minimal / syntactic /
reordered); fifteen preserve behavior; all fifteen re-recruit all six seed
features and the joint clamp transports at 0.70-1.39x the seed effect
(regime table A=15, B=0, C=0). The Texas ensemble is a prompt-family-level
mechanism. The pre-registered worry that paraphrases would activate
different features for the same concept (regime C, the cross-prompt
feature-identity problem) did not materialize on this family.

**Methodological footnote.** Zero-ablation produced a 1.5-logit joint drop
vs 24.5 at m=-2: removal is much weaker than active inhibition here,
consistent with Lindsey et al.'s practice of clamping to negative
multiples, and with self-repair: downstream compensation for ablations
(Hydra effect, McGrath et al., 2023; backup name-mover heads, Wang et al.,
2023; Rushing & Nanda, 2024). The negative single (L19/7477) is a concrete
instance. This is also why the single-ablation robustness property the
handoff cites as [PIE] did not transfer wholesale to this PLT stack — one
feature is individually potent — though the m=0 signature (singles ~0,
joint clearly positive) matches the redundancy picture in miniature.

**The task itself.** The result causally confirms latent two-hop
composition (Dallas -> Texas -> Austin, "Texas" never emitted) on this
prompt family — aligned with behavioral/representational evidence that
latent multi-hop reasoning exists but is uneven (Yang et al., 2024),
factual-recall mechanics (Geva et al., 2023), and late second-hop
resolution (Biran et al., 2024).

## 3. What this tool can verify — and what it cannot

The harness verifies claims of the form: "feature set F, at token
positions P, causally and specifically mediates output Y, and this
transports across a defined prompt family." Three filters bound the
verifiable class:

1. **The concept must be in the dictionary.** Transcoders expose only
   concepts they learned; coverage is incomplete and features split with
   dictionary size (Bricken et al., 2023; Templeton et al., 2024).
2. **The concept must be MLP-stream-expressed.** Attention patterns are
   frozen (Ameisen et al., 2025), so QK-space concepts — binding, routing,
   positional selection of the kind IOI's name-movers perform (Wang et
   al., 2023) — are out of scope.
3. **The concept must not live in the error nodes.** Reconstruction error
   is lumped into per-layer terms the graph cannot decompose — flagged by
   Lindsey et al. (2025) as a primary limitation.

Within those filters the verifiable class includes exactly the node types
these graphs feature: entity features (the tested "Texas" set),
relation/extraction features, and output-promoting "say X" features
(Lindsey et al., 2025). Granularity: finer and more concept-shaped than
head-level circuits (Wang et al., 2023), better-grounded than
single-neuron claims (Elhage et al., 2022), more constrained than
arbitrary-subspace patching, where illusions live (Makelov et al., 2023).
Closest methodological relatives: feature steering (Templeton et al.,
2024) and ablation-validated sparse feature circuits (Marks et al., 2024);
the addition here is the pre-registered four-way structure — necessity,
redundancy decomposition, matched-control specificity, paraphrase
transport — applied to a published graph as the hypothesis under test.

**Caveats.** One model, one fact family, one seed, default thresholds,
bf16 on CPU (numerics validated: the six features' activations reproduce
the canonical graph to ~1%). T1 paraphrases keep the same entities and
relation; transport across relations/entity pairs is untested. S2's
ambiguity is a finding about the scalar criterion as much as the model —
the planned mediation-share estimator (Pearl, 2001; Vig et al., 2020) plus
per-feature response curves is the right instrument.

## References

- Ameisen, E., et al. (2025). Circuit Tracing: Revealing Computational Graphs in Language Models. Transformer Circuits Thread.
- Biran, E., et al. (2024). Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries. EMNLP 2024.
- Bolukbasi, T., et al. (2021). An Interpretability Illusion for BERT. arXiv:2104.07143.
- Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits Thread.
- Chan, L., et al. (2022). Causal Scrubbing: A Method for Rigorously Testing Interpretability Hypotheses. AI Alignment Forum (Redwood Research).
- Dai, D., et al. (2022). Knowledge Neurons in Pretrained Transformers. ACL 2022.
- Dunefsky, J., Chlenski, P., & Nanda, N. (2024). Transcoders Find Interpretable LLM Feature Circuits. NeurIPS 2024.
- Elhage, N., et al. (2022). Toy Models of Superposition. Transformer Circuits Thread.
- Geiger, A., et al. (2021). Causal Abstractions of Neural Networks. NeurIPS 2021; Geiger, A., et al. (2022). Inducing Causal Structure for Interpretable Neural Networks. ICML 2022.
- Geva, M., et al. (2023). Dissecting Recall of Factual Associations in Auto-Regressive Language Models. EMNLP 2023.
- Hanna, M., Pezzelle, S., & Belinkov, Y. (2024). Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms. COLM 2024.
- Hanna, M., Piotrowski, M., et al. (2025). circuit-tracer (open-source library, Anthropic Fellows program / Decode Research, with Neuronpedia).
- Heimersheim, S., & Nanda, N. (2024). How to Use and Interpret Activation Patching. arXiv:2404.15255.
- Lieberum, T., et al. (2023). Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla. arXiv:2307.09458.
- Lieberum, T., et al. (2024). Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. arXiv:2408.05147.
- Lindsey, J., et al. (2025). On the Biology of a Large Language Model. Transformer Circuits Thread.
- Makelov, A., Lange, G., & Nanda, N. (2023). Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching. arXiv:2311.17030.
- Marks, S., et al. (2024). Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models. arXiv:2403.19647.
- McGrath, T., et al. (2023). The Hydra Effect: Emergent Self-Repair in Language Model Computations. arXiv:2307.15771.
- Meng, K., et al. (2022). Locating and Editing Factual Associations in GPT. NeurIPS 2022.
- Olah, C., et al. (2020). Zoom In: An Introduction to Circuits. Distill.
- Pearl, J. (2001). Direct and Indirect Effects. UAI 2001.
- Radford, A., et al. (2017). Learning to Generate Reviews and Discovering Sentiment. arXiv:1704.01444.
- Rushing, C., & Nanda, N. (2024). Explorations of Self-Repair in Language Models. arXiv:2402.15390.
- Templeton, A., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. Transformer Circuits Thread.
- Vig, J., et al. (2020). Investigating Gender Bias in Language Models Using Causal Mediation Analysis. NeurIPS 2020.
- Wang, K., et al. (2023). Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small. ICLR 2023.
- Yang, S., et al. (2024). Do Large Language Models Latently Perform Multi-Hop Reasoning? ACL 2024.
