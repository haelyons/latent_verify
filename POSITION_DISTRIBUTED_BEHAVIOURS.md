# POSITION — caving as a distributed behaviour

**TL;DR.** Caving is not a sparse read+write circuit. It is distributed (the apparent
localisation was a first-token Yes/No readout confound; even the polarity write is
ablation-redundant), carried by a steerable-but-not-necessary monitor direction and gated
upstream by a weak doubt-reading head set. This is the *norm* in model biology, not an
anomaly — clean circuits (IOI, induction) are the exception. Position the result inside the
distributed-behaviour lineage; do not frame it as a failed circuit hunt.

## The five legs, and their prior art

1. **Ablation-robust / redundant write → self-repair + backup.**
   - Wang (2022), *Interpretability in the Wild* — knock out all Name Movers, only ~5% drop
     (Backup Name Movers take over); "faithfulness alone is not enough".
   - McGrath (2023), *The Hydra Effect* — ablate a layer, a downstream layer compensates;
     generic; raised explicitly as a problem for circuit-level attribution.
   - Mechanism: Rushing & Nanda (2024), *Explorations of Self-Repair in Language Models*
     (LayerNorm rescaling + anti-erasure); McDougall (2023), *Copy Suppression* (negative
     heads stop suppressing when the upstream copier is ablated).

2. **Localisation ≠ causation → DLA/attribution mislead.**
   - Hase (2023), *Does Localization Inform Editing?* — causal-tracing location explains
     ~0.1% of edit-success variance. The tightest precedent for "DLA ranks writers, ablation
     compensated".
   - Makelov (2023), *Is This the Subspace You Are Looking For?* — a subspace patch steers via
     a dormant parallel pathway "causally disconnected from outputs". The monitor leg.
   - Syed (2023), *Attribution Patching Outperforms ACDC* — attribution patching is a linear
     approx; misses redundant sets. Why marginal ATP buried the doubt heads.
   - Huang (2025), *Causality ≠ Decodability* — decodable reps can be inert. The cave-direction
     is a monitor, not a lever.

3. **Behaviour-as-direction (sufficient to steer, not the whole mechanism).**
   - Marks & Tegmark (2023), *The Geometry of Truth* — diff-in-means direction steers
     true↔false, one causal direction among many.
   - Todd (2023), *Function Vectors*; Ilharco (2022), *Editing Models with Task Arithmetic*;
     Hendel (2023), *In-Context Learning Creates Task Vectors*; Park (2023), *The Linear
     Representation Hypothesis*.
   - Contrast: Arditi (2024), *Refusal Is Mediated by a Single Direction* — necessary AND
     sufficient. Caving fails the necessity test refusal passes. (And even refusal erodes:
     Joad (2026), *There Is More to Refusal than a Single Direction*.)

4. **Distributed-by-construction → superposition.**
   - Elhage (2022), *Toy Models of Superposition*; Bricken (2023), *Towards Monosemanticity*;
     Templeton (2024), *Scaling Monosemanticity* — more features than neurons, polysemantic
     units, recoverable only by SAEs. Why no component-aligned circuit exists to find.

5. **Worked precedents (distributed recall + confidence).**
   - Meng (2022), *Locating and Editing Factual Associations* → Geva (2023), *Dissecting Recall
     of Factual Associations* — recall is a multi-step distributed process (same attention-
     knockout we use).
   - Stolfo & Gurnee (2024), *Confidence Regulation Neurons* — the closest structural twin:
     confidence regulated by distributed components acting indirectly (LayerNorm scale,
     unembedding null space, minimal direct logit effect). Mirrors "doubt heads are a read-gate,
     not the writer; no single mediator". **Investigated in-house and NULL on Gemma-2-9b**
     (`controls/entropy_neuron_gemma2.py`, `entropy_distributed_presoftcap.py`: count 0,
     single-neuron and group, pre- and post-softcap) — our own corroboration that the mechanism
     does not localise here.

## The meta-finding

Lindsey (2025), *On the Biology of a Large Language Model* — attribution graphs give satisfying
insight on "about a quarter of the prompts", capture "a small fraction of the mechanisms";
refusal is "more complicated than a single linear binary classifier"; medical reasoning is "many
heuristics in parallel"; attention-pattern formation is "invisible to our current approach" (the
upstream gate the method admits it misses = our read-gate). Distributed/diffuse is the default.

## Positioning guidance

- Frame caving as a **novel instance/transfer**, not a new phenomenon class: the shape
  (upstream attention gate + distributed downstream write + steerable monitor) is assembled
  from known motifs (IOI S-inhibition→mover-query gate; Confidence Regulation Neurons'
  distributed-indirect regulation; Geometry-of-Truth / Makelov steerable-not-necessary
  direction). What is new is applying it to a **social/contextual** behaviour.
- Lead the contrast with refusal (Arditi): caving is steerable like refusal but, unlike it,
  **not necessary** (projection-out ≈ floor, behavioural) — and refusal's own 1-D account is
  already eroding (Joad).
- Treat the in-house entropy-neuron null as supporting evidence, not a side-quest: even the
  confidence regulator that is distributed-indirect elsewhere does not localise on Gemma-2.
- Keep the lineage arc explicit: IOI looked clean (2022) → backups → self-repair generalised
  (Hydra) → attribution shown to mislead (Hase, Makelov) → graphs explain ~25% (Lindsey).
  Caving sits squarely on that trajectory.
