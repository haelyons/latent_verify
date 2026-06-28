# Diagram V3 — consistency audit and a math-overlay second layer

Scope: the `RESIDUAL_STREAM_BLOCK` figure (Figma `pypXIq84cLz3BfYgTwzr1b`, node `76:4`, page "V3"). It marks where five interventions act on one transformer block: **R** (read), **W** (write), **X** (all-X), **RO** (read-out), **S** (steer). This doc (1) audits the figure for mathematical and visual consistency, (2) distils diagram heuristics from our three inspirations, and (3) mocks up a math overlay — a second annotation layer that states the exact tensor operation at each marker.

The figure accompanies EXPLAINER §5 ("What we actually intervene on") and the same five operations in the LW draft. Ground truth for each operation is taken from there.

---

## 1. What V3 gets right (keep these)

- **Intervention loci are clean and on the correct path.** R sits on the attention pattern; W on the per-head output before `W_O`; the two `X` badges on the attention card and the MLP card; RO/S on the residual spine. The *placement-as-grammar* (where on the block each edit lands) is the figure's strongest asset and the reason it is worth fixing rather than redrawing.
- **The block topology is right.** Pre-norm → QKV → attention → concat → `W_O` → residual add; second norm → MLP → residual add. Both `+` adds are present and correctly placed.
- **The core attention label is correct:** `softmax(QKᵀ/√d)·V`.

## 2. Mathematical inconsistencies (what is actually wrong, not just ugly)

Ranked by how badly they misrepresent the operation.

**2.1 — The doubt-span matrix inverts R (HIGH).**
R zeroes attention **at the answer-slot query row only** — one row of the pattern, the columns over the doubt span — then renormalizes that row. The figure highlights a **full vertical column-block across every query row**. A reader infers R is a query-position-agnostic surgery on the whole pattern ("cut these keys for all positions"), which is the opposite of the single-row edit we run. This is the load-bearing mechanism in the figure and it currently reads backwards.

**2.2 — The renormalize step is invisible (MED).**
R is *zero-then-renormalize* so the answer-slot row still sums to 1; we cut *where* the head looks, not *how much* total attention it has. The matrix shows only the zeroing, so a reader infers attention mass is destroyed. The renorm is exactly what makes R a clean "sever the QK read" rather than a magnitude knock-down.

**2.3 — RO and S are drawn at the wrong depth (HIGH).**
Both markers sit on the residual spine **below the second add — i.e. at block output**. We read and steer the cave-direction at a **mid layer** (`ℓ* ≈ L28/42` on 9b, the ~two-thirds-depth tap), on the **answer-slot** residual. As drawn, the figure asserts "block-ℓ output residual," which misstates where the readout lives.

**2.4 — RO and S share one glyph, hiding the read/mutate distinction (HIGH).**
RO is a measurement (`no edit`); S writes into the stream. They are the only read-vs-write pair in the set, and the figure renders them as two identical dots stacked at the same site — so a reader takes them for a matched read/write at one point. The single most important visual contract in an interventions figure is *which markers mutate the stream and which only measure it*; V3 erases it.

**2.5 — W is ambiguous between one-head and all-head (HIGH).**
W replaces **one head's** output `z` at the answer slot with its **neutral-run** value. The marker sits on the line between `Concat` and `W_O`, i.e. on the *concatenated* z. From the page you cannot tell (a) single head vs whole concat, (b) answer-slot only, or (c) that the substituted value comes from a *different forward pass* (neutral), not the current one. The "from the neutral run" fact is the entire content of the operation and it is unshown.

**2.6 — `unembed / softcap / argmax` hang off a single block (MED).**
`OUTPUT ◄ argmax ◄ softcap ◄ unembed` reads correctly as `argmax(softcap(unembed(·)))`, Gemma-style. But these are **model-global**, applied once after the last block — drawing them off block ℓ is a category slip. Also missing: **final norm (`ln_f`)** before unembed, which an expert expects.

**2.7 — `all-X` collapses two different operations into one glyph (MED, defensible).**
`X` on attention means "patch every head's `z` counter→neutral"; `X` on MLP means "patch the MLP output counter→neutral." Same legend symbol, two different tensors. Acceptable as an upper-bound shorthand, but the overlay should make the two maps explicit.

## 3. Visual / layout shortcomings

- **Dead mid-band.** The residual spine is a lonely vertical line on the far right (`x≈906`) while all components live left of `x≈620`. The whole middle third is empty. Collapse it: either pull the spine inward or reserve that right gutter for the math callouts (§5).
- **The two load-bearing markers are the least visible.** RO and S are tiny, in the bottom-right corner. The most conceptually important "where" markers should be the most findable.
- **Cramped top-left.** The stacked green "head-stack" cards, the matrix, the `keys→ / q↓` labels and the `X` badge all collide in one corner.
- **Tall and sparse.** Large vertical whitespace between the `W_O` add and the second norm makes the figure scroll without adding information.
- **Empty MLP box.** A large rectangle with only "MLP" centered — either shrink it or show `W_in → GELU → W_out` so it earns its area and the MLP `X` has something to annotate.
- **`doubt span` term used without gloss** on a figure titled "generic."

---

## 4. Heuristics from the inspirations

Distilled from IOI (Wang et al. 2023), the Mathematical Framework for Transformer Circuits (Elhage et al. 2021), and the learnmechinterp.com induction-heads article. Each traces to a source.

1. **Name components by function, not index** — boxes read "copy head", "doubt read", not "head 9.6". Reader parses role at a glance. *(IOI)*
2. **Residual stream is the spine; everything reads from and writes to it.** Components don't connect to each other — they connect to the stream. Draw interventions as **taps on the spine**. *(Framework)*
3. **Abstract a head/MLP to one box that adds into the stream.** Don't draw internal QKV plumbing in the base layer; the math overlay carries the internals. *(Framework)*
4. **Color encodes role, not decoration; keep the palette small (≤~6).** QK vs OV get distinct, consistent hues; the color *is* the legend. *(IOI; Framework's QK/OV split)*
5. **Interventions are edits on existing edges, not new structure** — recolor or cut an arrow and tag it; don't add a subgraph. *(IOI path-patching)*
6. **One concept per panel; caption teaches the mechanism in place.** Don't strand a dense figure with its explanation elsewhere. *(learnmechinterp)*
7. **Caption register: capitalized term-of-art + plain-language verb.** "Read-out — *projects the answer-slot residual onto the fitted direction*." Formal noun, conversational gloss. *(learnmechinterp)*
8. **Omit to fit, and say so once.** Fold norm into adjacent weights, drop biases, freeze patterns — state the omission in the caption so the overlay stays short and linear. *(Framework)*

**The single worst mistake, and it is the one V3 makes:** giving read-only and mutating markers the same glyph. A reader must see at a glance which interventions edit the stream and which only measure it — that distinction is the whole point of the figure.

---

## 5. The math overlay — a second layer (mockup)

Design contract: **the base figure is untouched.** The overlay is a toggleable annotation tier that rides on top, keyed to each marker by color. One callout per marker, anchored at its tap point by a leader line, with the equation box living in the **right gutter** (the dead mid-band reclaimed). Shared symbols are factored into a one-line legend so each callout is ≤ 1 line.

### 5.1 Factored symbol legend (stated once, bottom of figure)

```
x_t^ℓ   residual vector, block ℓ, token position t        a     answer-slot position (we read/edit here)
D       doubt span = challenge tokens ("Actually… sure?")  ℓ*    readout layer ≈ L28/42 (9b), ~2/3 depth
A^h     attention pattern of head h = softmax(QKᵀ/√d)       z_a^h head h output at slot a;  m_a^ℓ MLP output at a
(neu)   value from the NEUTRAL-prompt forward pass          d̂     cave-direction = (μ_cave − μ_hold)/‖·‖
```

### 5.2 Per-marker callouts (the overlay content)

Notation: `↦` means "this slot's value is replaced/shifted to the RHS." Solid tap = **mutates** the stream; hollow/dashed tap = **reads only**.

| Mark | Tap style | Operation (overlay equation) | Plain gloss in caption |
|---|---|---|---|
| **R** | solid, on pattern | `A^h_{a,j} ↦ 0  ∀ j∈D`, then renormalize row: `A^h_{a,:} ↦ A^h_{a,:} / Σ_{j∉D} A^h_{a,j}` | sever the answer slot's QK read of the doubt span; row still sums to 1 |
| **W** | solid, on one z-lane | `z_a^h ↦ z_a^{h,(neu)}`  (one head, answer slot) | cancel this head's OV write by pasting its neutral-run output |
| **X** (attn) | solid, on attn card | `z_a^h ↦ z_a^{h,(neu)}  ∀ h`  → upper bound | do W for every head at once |
| **X** (MLP) | solid, on MLP card | `m_a^ℓ ↦ m_a^{ℓ,(neu)}`  → upper bound | paste the whole MLP's neutral-run output |
| **RO** | **hollow / dashed** | `s = ⟨x_a^{ℓ*}, d̂⟩`  — *no edit* | measure how far the slot leans along the cave-direction |
| **S** | solid, `+` glyph | `x_a^{ℓ*} ↦ x_a^{ℓ*} + α·d̂` | add the direction back in; test if it drives, not just reads |

Readout metric, in a footnote box next to RO (this is what every result number means):

```
restoration  ρ = (s_counter − s_after) / (s_counter − s_hold)
s_hold = mean projection on items the model did NOT cave (held-firm anchor)
ρ = 1  intervention fully undid the internal cave  ·  ρ = 0  changed nothing
```

### 5.3 ASCII mock of the overlaid figure (layout intent)

Right gutter (reclaimed dead band) holds the equations; color links callout↔marker. Spine pulled inward.

```
 Block ℓ                                         RESIDUAL STREAM        MATH OVERLAY (right gutter)
 ┌───────────────────────────────────────┐            │
 │  ┌──────────┐                          │            │
 │  │ LayerNorm│◄─────────────────────────┼────────────┤
 │  └────┬─────┘                          │            │
 │   W_K   W_Q   W_V                      │            │
 │     ▼     ▼     ▼                      │            │
 │  ┌───────────────────────────[X]──┐    │            │   X(attn): z_a^h ↦ z_a^{h,(neu)} ∀h
 │  │ Attention  softmax(QKᵀ/√d)·V    │    │            │
 │  │        keys →                   │    │            │
 │  │  q=a → ███░░░░  ◄═[R] (solid)   │────┼────────────┤   R: A_{a,j}↦0 ∀j∈D, renorm row→Σ=1
 │  │  (only the answer-slot ROW)     │    │            │       (highlight ONE row × D cols)
 │  │  doubt span = cols j∈D          │    │            │
 │  └─────────────┬───────────────────┘    │            │
 │          one head z-lane ═[W] (solid)   │            │   W: z_a^h ↦ z_a^{h,(neu)}  (1 head, slot a)
 │              ▼                          │            │
 │           Concat → W_O ──────────────►(+)───────────┤
 │  ┌──────────┐                          │            │
 │  │ LayerNorm│◄─────────────────────────┼────────────┤
 │  └────┬─────┘                          │            │
 │  ┌────▼───────────────────────[X]─┐    │            │   X(MLP): m_a^ℓ ↦ m_a^{ℓ,(neu)}
 │  │ MLP   W_in → GELU → W_out       │────┼──────────►(+)
 │  └─────────────────────────────────┘    │            │
 └─────────────────────────────────────────┘  ┄┄[RO]┄┄┄┤   RO (hollow): s = ⟨x_a^{ℓ*}, d̂⟩   no edit
        readout tap at mid layer ℓ* ───────────══[S]════┤   S (solid):  x_a^{ℓ*} ↦ x_a^{ℓ*}+α·d̂
        (answer slot a)                            │
                                                   ▼
        model-global (outside block):  ln_f → unembed → softcap → argmax → OUTPUT
```

### 5.4 The three changes that do the most work

1. **Redraw the doubt-span matrix** to highlight **one row** (the answer-slot query) × the doubt-span key columns, and draw the renormalize (row re-sums to 1). Fixes the inverted mechanism (2.1, 2.2).
2. **Split read from mutate visually:** RO becomes a hollow/dashed tap, every editing marker stays solid; move RO/S to a labelled **mid-layer tap on the answer slot** instead of block output (2.3, 2.4).
3. **Reclaim the dead mid-band as the equation gutter,** pull the spine inward, and add the `↦`-notation callouts above keyed by color. This is what makes it a *second layer* rather than a relabel (3, §5.1–5.2).

### 5.5 Smaller corrections
- Tag W "answer slot · value from neutral run"; anchor it to a single head's z-lane, not the concat.
- Show `ln_f` before `unembed`; move `unembed/softcap/argmax` into a clearly *model-global* strip outside block ℓ.
- Fill or shrink the MLP box (`W_in → GELU → W_out`) so its `X` annotation has a referent.
- Gloss "doubt span" in the caption, or rename it generically (e.g. "challenge span") if the figure is meant to stay model-agnostic.
