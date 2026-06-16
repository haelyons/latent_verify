"""Generator for the "from first principles" Colab notebook.

Mirrors the repo idiom in `visual/build.py`: the notebook is an *artifact*, this
script is its source of truth. Cells are authored here as plain strings, so there
is no hand-escaped .ipynb JSON to drift. Re-run to regenerate:

    python colab/build_notebook.py        # -> colab/latent_verify_first_principles.ipynb

The notebook follows the research LINEAGE's own progression: start with the
smallest fully-observable thing (induction heads on a sequence whose answer we
generate ourselves), then climb the same intervene-don't-read instrument up to
the headline attention-copy / sycophancy results. Section 1 is implemented in
full; sections 2-6 are markdown stubs carrying the outline so the whole arc is
visible from the first commit.
"""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append(("markdown", text.strip("\n") + "\n"))


def code(text):
    cells.append(("code", text.strip("\n") + "\n"))


# ----------------------------------------------------------------------------
# Section 0 -- Preface: the one method
# ----------------------------------------------------------------------------
md(r"""
# Latent verification from first principles
### Attention-copy circuits, and how to turn "I see it on a graph" into "I cut the wire and the light went out"

This notebook rebuilds the `latent_verify` project from the ground up. It has **one method**, applied at growing levels of difficulty:

> Take a *correlational* description of something happening inside a language model — "this feature lights up", "this head attends here", "this graph shows a path" — and convert it into a *causal* claim by **intervening on the proposed mechanism and watching the behaviour change.** Reading a graph is never the evidence. Cutting the wire and seeing the light go out is.

That is the whole game. Everything below is the same move on a harder target each time.

**How the notebook is laid out** — it follows the research lineage's own progression, smallest and most observable first:

| Section | What it does | Model | How solid is the ground truth? |
|---|---|---|---|
| **1. The primitive** | An *induction head* copying a token, on a sequence whose answer we generate ourselves | `gpt2-small` | **Exact** — we built the repeat, so the right answer is known by construction |
| 2. Verify a published graph | Clamp 6 features → "Austin" collapses (the Texas→Austin attribution graph) | `gemma-2-2b` | A published claim we test |
| 3. A real phenomenon, and a null | Framing flips an answer; the *obvious* culprit turns out not to cause it | `gemma-2-2b` | A measured behaviour |
| 4. **The headline** | The flip is carried by attention copying the salient word; localise it to one reader head | `gemma-2-2b` | Knockout + matched control |
| 5. Scale & root mechanism | RLHF deletes the head from the *weights*; "sycophancy" is one *strategy*, many circuits | `gemma-2-2b/9b` | — |
| 6. Honest ledger | Caveats, retractions, and how to actually verify/share a claim like this | — | — |

**Section 1 is fully runnable here.** Sections 2–6 carry the outline (and point at the scripts and `out/*.json` artifacts that already produced those numbers).

*Why start with an induction head?* Because Section 2 — verifying someone else's attribution graph — already asks you to *trust the instrument*. Before we trust it, we point it at a case where we know the answer in advance, with no model-behaviour ambiguity left to hide an instrument bug. If the knockout doesn't behave on ground truth, every number after it is suspect.

**Runtime.** Section 1 needs only `gpt2-small` and runs in a couple of minutes on a free Colab CPU (a T4 is faster but not required). The later sections load `gemma-2-2b`, which wants a GPU and a Hugging Face token with the Gemma-2 licence accepted.

**Literature.** The induction-head story below is the well-trodden path from [*A Mathematical Framework for Transformer Circuits* (Elhage et al., 2021)](https://transformer-circuits.pub/2021/framework/index.html) and [*In-context Learning and Induction Heads* (Olsson et al., 2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html), and is told pedagogically on [Learn Mechanistic Interpretability — *Induction Heads and In-Context Learning*](https://learnmechinterp.com/topics/induction-heads/). The tooling is [TransformerLens (Nanda)](https://github.com/TransformerLensOrg/TransformerLens).
""")

# ----------------------------------------------------------------------------
# Section 1 -- The primitive on ground truth
# ----------------------------------------------------------------------------
md(r"""
---
## 1. The primitive, on ground truth we control

### 1.0 What we are about to do

We will watch a transformer do **in-context learning** in its simplest possible form — completing a repeated sequence — find the single attention head responsible, and then *causally* confirm it by severing exactly the connection we think carries the work and watching the prediction collapse. A matched control (severing a different, equally-large connection) will show the effect is specific, not just "any big perturbation breaks things."

This is the entire `latent_verify` instrument, demonstrated on a case where **we know the correct answer by construction**.
""")

md(r"""
### 1.1 What one attention head actually does

A useful one-paragraph model, following [Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html). An attention head does two separable things:

- **Where to look** is decided by the **QK circuit**: each query position scores every earlier key position, and the scores become attention weights (a probability distribution over earlier positions). This is *pure routing* — it moves information but doesn't decide *what* the information says.
- **What to copy** is decided by the **OV circuit**: whatever the head reads from the position it attended to gets written into the current position's residual stream.

So a head is a little "go fetch *that* earlier position and paste it here" device. The **induction head** is the cleanest instance: it implements the rule

$$[A]\,[B]\;\dots\;[A]\;\rightarrow\;[B]$$

"if the current token already appeared earlier, predict whatever came *after* it last time." That requires two heads composed across layers — a *previous-token head* that tags each position with its predecessor, and the induction head that uses those tags to find the right place to copy from (this composition is the subject of the [Learn Mechanistic Interpretability page](https://learnmechinterp.com/topics/induction-heads/)). We won't need to separate the two heads to make the causal point; we only need to find *a* head that does the copy and cut its wire.
""")

code(r"""
# Section 1 needs only gpt2-small. No GPU required, no token required.
# (The later gemma sections were run on transformer_lens 3.4; gpt2 here is
#  insensitive to the exact version.)
%pip install -q transformer_lens
""")

code(r"""
import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

# gpt2-small: 12 layers x 12 heads. from_pretrained folds LayerNorm and centres
# the weights -- standard for interpretability, leaves attention patterns intact.
model = HookedTransformer.from_pretrained("gpt2", device=device)
model.eval()

print(f"device          : {device}")
print(f"layers x heads  : {model.cfg.n_layers} x {model.cfg.n_heads}")
print(f"vocab size      : {model.cfg.d_vocab}")
""")

md(r"""
### 1.2 A task with ground truth we generate ourselves

The trap with interpretability is confounds: maybe the model "knew" the answer for some unrelated reason, and the mechanism we cut was incidental. We remove that trap by making the *only* way to solve the task be the mechanism we are studying.

We feed the model a sequence of **random tokens, then the same random tokens again**:

```
[BOS]  r0 r1 r2 ... r49   r0 r1 r2 ... r49
       └─ first copy ─┘   └─ second copy ─┘
```

The tokens are random, so there is no English, no fact, no prior — *nothing* in the world tells you what comes after `r10`. The only signal is the first copy. So if, partway through the second copy, the model confidently predicts the next token, it can only be doing so by looking back at the first copy and reading off what followed last time. **That is induction, and the correct answer is whatever we put there.**
""")

code(r"""
torch.manual_seed(0)

seq_len = 50          # length of one copy
batch   = 16          # average over 16 independent random sequences
bos     = model.tokenizer.bos_token_id

rand   = torch.randint(0, model.cfg.d_vocab, (batch, seq_len))
tokens = torch.cat(
    [torch.full((batch, 1), bos), rand, rand], dim=1
).to(device)          # shape [batch, 1 + 2*seq_len]

print(f"tokens shape    : {tuple(tokens.shape)}   (BOS + two identical copies)")
print(f"first copy [:8] : {tokens[0, 1:9].tolist()}")
print(f"second copy[:8] : {tokens[0, 1+seq_len:1+seq_len+8].tolist()}   <- identical, by construction")
""")

md(r"""
### 1.3 Observe the behaviour first (the correlational step)

Before touching internals, measure the *behaviour*. For every position we compute the loss — the negative log-probability the model assigned to the token that actually came next. We do this in **one forward pass**; the model is given no gradient updates. Any drop in loss is therefore learning *from context alone*.

The signature of induction is unmistakable: loss is high across the first copy (the tokens are random and unpredictable), then **falls off a cliff at the start of the second copy** and stays low. That cliff is in-context learning happening in real time — the behavioural fingerprint [Olsson et al. (2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) tie to induction heads.
""")

code(r"""
import matplotlib.pyplot as plt

logits = model(tokens)                              # [batch, pos, vocab]
log_probs = logits.log_softmax(-1)
correct = tokens[:, 1:]                             # next-token targets
lp_correct = log_probs[:, :-1].gather(-1, correct.unsqueeze(-1)).squeeze(-1)
loss_per_pos = (-lp_correct).mean(0).float().cpu()  # mean over the 16 sequences

plt.figure(figsize=(9, 3.2))
plt.plot(range(1, loss_per_pos.shape[0] + 1), loss_per_pos, lw=1.5)
plt.axvline(seq_len, color="crimson", ls="--", lw=1, label="second copy begins")
plt.xlabel("position"); plt.ylabel("loss (nats)")
plt.title("In-context learning, in one forward pass: loss collapses on the repeat")
plt.legend(); plt.tight_layout(); plt.show()

first_half  = loss_per_pos[:seq_len].mean()
second_half = loss_per_pos[seq_len:].mean()
print(f"mean loss, first copy  : {first_half:.2f} nats   (random -> unpredictable)")
print(f"mean loss, second copy : {second_half:.2f} nats   (the model is now copying)")
""")

md(r"""
**Pause and read the plot.** The drop is the *what*: the model is solving the task. It says nothing about the *how* — which is exactly the gap between a correlational observation and a mechanism. Everything from here is closing that gap.
""")

md(r"""
### 1.4 Find the head (still correlational)

If a head is doing induction, then at a query position in the second copy it should be attending to a very specific earlier key: the position **one step after** where the current token first appeared. Work the indices out once and it is a fixed offset.

At query position $q$ in the second copy, the current token is the same one that sat at $q - L$ (one copy earlier, where $L$ is the copy length). To predict what comes *next*, the head should look at the position right after that first occurrence: key $= q - L + 1 = q - (L-1)$. So an induction head's attention concentrates on a single off-diagonal **stripe** at offset $-(L-1)$.

We score every head by how much attention mass it places on that stripe, averaged over our 16 sequences. This is the project's "read the graph" step — informative, but still only a correlation.
""")

code(r"""
_, cache = model.run_with_cache(tokens)

L = seq_len
n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
induction = torch.zeros(n_layers, n_heads)

for layer in range(n_layers):
    pattern = cache["pattern", layer]                       # [batch, head, q, k]
    stripe  = pattern.diagonal(dim1=-2, dim2=-1, offset=1 - L)  # key = q-(L-1)
    induction[layer] = stripe.float().mean(dim=(0, 2))      # mean over batch & stripe

plt.figure(figsize=(6, 5))
plt.imshow(induction, cmap="viridis", aspect="auto")
plt.colorbar(label="mean attention on the induction stripe")
plt.xlabel("head"); plt.ylabel("layer")
plt.title("Induction score by head")
plt.tight_layout(); plt.show()

top_vals, top_idx = induction.flatten().topk(5)
print("top induction heads (layer.head : score):")
for v, i in zip(top_vals, top_idx):
    print(f"  L{i.item() // n_heads}.H{i.item() % n_heads} : {v:.3f}")

best = int(top_idx[0])
best_layer, best_head = best // n_heads, best % n_heads
print(f"\nstrongest induction head -> L{best_layer}.H{best_head}")
""")

md(r"""
### 1.5 Read the attention pattern

Picture the strongest head's attention on a single sequence. An induction head shows the tell described above: a bright diagonal **stripe**, parallel to the main diagonal but shifted, appearing only in the second-copy region. That stripe *is* "go fetch the token after my earlier self."
""")

code(r"""
pat = cache["pattern", best_layer][0, best_head].float().cpu()   # [q, k] for sequence 0

plt.figure(figsize=(5.2, 5))
plt.imshow(pat, cmap="magma", aspect="auto")
plt.colorbar(label="attention weight")
plt.xlabel("key position (looked at)"); plt.ylabel("query position (doing the looking)")
plt.title(f"L{best_layer}.H{best_head}: the induction stripe (offset -{L-1})")
plt.tight_layout(); plt.show()
""")

md(r"""
### 1.6 Cut the wire (the causal step)

Here is the move the whole project is built on. We have a *correlational* story: "this head attends to the source token." We make it *causal* by removing exactly that connection and nothing else.

Concretely, for one chosen query position in the second copy:

1. record the model's log-probability for the **correct** next token (we know it — it is `tokens[q+1]`);
2. **knock out** the head's attention *to the source key*: set that one attention weight to zero and renormalise the rest so it remains a valid distribution (this is the identical operation the gemma experiments use — zero the key in `hook_pattern`, then divide by the new sum);
3. re-read the correct token's log-probability.

If the head's attention to the source is what carries the answer, step 3 collapses. To prove the collapse is *specific* and not just damage from perturbing the head at all, we run a **matched control**: the same head, the same operation, but pointed at a different (wrong) earlier key. A specific mechanism breaks under (2) and shrugs off the control.
""")

code(r"""
# the canonical knockout: zero one head's attention to chosen key positions, renormalise.
# identical in form to job_chat_mechanism.py's ko_head / job_attn_sweep.py's ko_hook.
def knockout(layer, head, key_positions):
    name = f"blocks.{layer}.attn.hook_pattern"
    keys = torch.as_tensor(key_positions, device=device)
    def hook(pattern, hook):
        pattern[:, head, :, keys] = 0.0
        denom = pattern[:, head].sum(-1, keepdim=True).clamp_min(1e-9)
        pattern[:, head] = pattern[:, head] / denom
        return pattern
    return name, hook

q          = 1 + L + 20                 # a query well inside the second copy
source     = q - (L - 1)                # the induction source the head should use
control    = source + 7                 # a different, equally-valid earlier key (wrong target)
target_tok = tokens[0, q + 1].item()    # the correct next token -- known by construction

def correct_logp(fwd_hooks):
    out = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)[0, q]
    return float(out.log_softmax(-1)[target_tok])

clean   = correct_logp([])
severed = correct_logp([knockout(best_layer, best_head, [source])])
ctrl    = correct_logp([knockout(best_layer, best_head, [control])])

dec = lambda t: repr(model.tokenizer.decode([t]))
print(f"query position q = {q},  correct next token = {dec(target_tok)}")
print(f"  source key  (pos {source:>3}) = {dec(tokens[0, source].item())}")
print(f"  control key (pos {control:>3}) = {dec(tokens[0, control].item())}  (a wrong target)\n")
print(f"clean                       logp(correct) = {clean:7.2f}")
print(f"sever head -> SOURCE        logp(correct) = {severed:7.2f}   drop = {clean - severed:5.2f}")
print(f"sever head -> CONTROL key   logp(correct) = {ctrl:7.2f}   drop = {clean - ctrl:5.2f}")
""")

code(r"""
plt.figure(figsize=(5.4, 3.2))
bars = ["clean", "sever\nSOURCE", "sever\nCONTROL"]
vals = [clean, severed, ctrl]
plt.bar(bars, vals, color=["#4c72b0", "#c44e52", "#999999"])
plt.ylabel("log-prob of the correct token")
plt.title("Cut the right wire and the light goes out")
plt.tight_layout(); plt.show()
""")

md(r"""
### 1.7 What just happened — this is the whole instrument

Read the three numbers. Severing the head's attention **to the source** crushes the correct token's probability; severing the *same head's* attention to an equally-valid but **wrong** key barely moves it. The head's attention to the induction source is **causally necessary** for the copy — and we proved it on a task whose answer we manufactured, so no quirk of the model's knowledge can be hiding the result.

That logic — *intervene on the proposed mechanism, compare against a matched control, read the behaviour* — is the entire `latent_verify` method. The repo wraps one number around it, **necessity**:

> **necessity** = (fraction of an effect that is reverted / destroyed when you knock out the proposed carrier), measured against a matched-random control and reported only when the effect clears a floor (the repo uses 0.5 nats, below which the fraction is a divide-by-almost-zero).

In Section 4 you will see the *exact same* knockout — zero the attention to a key, renormalise — sever a real model's attention to the word "Sydney" and revert a factual error, with necessity ≈ 1.0. The only thing that changes from here to there is that the target gets messier and the ground truth gets weaker. The instrument is the one you just ran.

Everything below applies it, rung by rung, up the lineage.
""")

# ----------------------------------------------------------------------------
# Sections 2-6 -- outline stubs (to be filled in)
# ----------------------------------------------------------------------------
md(r"""
---
## 2. Verify a *published* attribution graph — Texas → Austin  *(outline)*

> Mirrors LINEAGE Part 1 (Arc 1). Model: `gemma-2-2b` + GemmaScope per-layer transcoders, via `circuit-tracer`. Source artifacts: `out/t0.json`, `out/t1.json`; method in `poc_minimal.py`.

The first real target is someone else's claim. A published circuit-tracing graph asserts that gemma-2-2b answers *"the capital of the state containing Dallas is → Austin"* through a **latent two-hop step** (Dallas → *Texas* → Austin) carried by six specific transcoder features — even though "Texas" is never written. The graph is a *hypothesis*; we test it with four pre-registered knockouts:

- **S1 — is it causally real?** Clamp all six features → does "Austin" collapse? *(Result: yes — Austin's logit drops 24.5 and its rank falls from 1 to ~88,000 of 256k.)*
- **S2 — one feature or an ensemble?** *(The interesting wrinkle: the biggest single mover explains only ~35% of the joint effect, and removing one feature alone slightly **helps** Austin — the **Hydra effect** / self-repair. Lesson, carried forward: the biggest mover is not the cause; removal ≠ inhibition; **intervene on sets**.)*
- **S3 — specific or just damage?** Compare against six random magnitude-matched features. *(~26× margin: specific.)*
- **S4 — one prompt or a mechanism?** Re-run across 16 paraphrases. *(15/16 hold.)*

*To implement:* the feature-clamp version of the same knockout, on the `gemma-2-2b` transcoder stack. Attention is **frozen** in this method (Ameisen et al., 2025) — note that ceiling now, because Section 4 is the discovery that the load-bearing mechanism lives exactly in that blind spot.
""")

md(r"""
## 3. A real phenomenon, and an informative null — framing  *(outline)*

> Mirrors LINEAGE §2.0–3.5. Source: `FRAMING_NOTES.md`, `framing_situations.json`, `framing_intervention.py`; artifacts `out/framing_b0.json`, `out/framing_intervention.json`.

Point the instrument at something people worry about: **framing**. Prepend a misleading-but-true salience cue and watch a low-confidence fact flip:

> *"Sydney is the most famous city in Australia. The capital of Australia is the city of ___"* → the model says **Sydney**.

- **Behaviour first** *(use log-prob `dlogp`, not raw `dlogit` — a sign bug the repo hit and fixed)*: susceptibility tracks **baseline confidence** — Australia's capital (p≈0.39) flips; Everest (p≈0.98) doesn't budge under the same manipulation.
- **First causal test is a null.** Clamp the biggest *activation movers* back to baseline → **necessity ≈ 0**, indistinguishable from random. The thing that *moves most is not the thing that causes* — the Section 2 lesson, recurring at scale. The prime suspect is named immediately: attention copying "Sydney" — a QK-space mechanism the transcoder method is structurally blind to.

*To implement:* behavioural sweep + the activation-mover knockout that returns ~0, setting up the headline.
""")

md(r"""
## 4. The headline — the flip is an attention copy  *(outline)*

> Mirrors LINEAGE §3.6–3.10. Source: `framing_dla.py`, `job_attn_sweep.py`, `job_localize_joint.py`, `job_head_profile.py`; artifacts `out/framing_dla.json`, `out/framing_attn_sweep.json`, `out/framing_localize_joint.json`.

- **Re-select by decision-alignment (DLA), not magnitude** → restoring ~24 DLA-chosen MLP features reverts **~half** the flip (vs ~0 for activation-movers). The MLP path carries half.
- **The other half is attention.** Run the **identical knockout from Section 1** — zero attention to a key, renormalise — but on gemma, severing attention to the single token **"Sydney"**: the flip reverts **completely**, necessity **≈ 1.0**, "Sydney" back to a non-answer. Function words sit at ~0; severing "Australia" makes it *worse*. Specific.
- **Transport:** 5/5 fact pairs (Texas/Houston, Canada/Toronto, …) fully revert under the same anchor knockout — a *mechanism*, not one prompt.
- **Localise:** the all-heads sledgehammer resolves to a compact ~12-head circuit and one **universal late reader head, L18.H5**, fed by a *pair-specific early writer* — including the repo's own self-correction (an early "co-principal" head was a single-pair artifact). Unlike IOI's *late* name-movers (Wang et al., 2023), the anchor bias is set early and read mid-stack.

This is the section where Section 1's toy knockout and the real finding become visibly the *same operation*.
""")

md(r"""
## 5. Scale and the root mechanism  *(outline)*

> Mirrors LINEAGE §8–10.2. Source: `job_chat_mechanism.py`, `job_numeric_boundary.py`, `job_scale_mechanism.py`, `job_numeric_mechanism.py`, `job_numeric_localize.py`; artifacts in `out/`.

- **RLHF removes the copy from the *weights*, not the prompt.** On the *identical* fragment where base reads 0.84 attention-to-anchor, the instruction-tuned model reads 0.01 — the reader head simply stops looking. Contrast: a QA scaffold *routes around* the copy (base weights still carry it); RLHF *deletes* it.
- **Robustness is selective.** RLHF trained out *salience* distraction structurally, but **authority-asserted numeric sycophancy survives** — and on hard products the model can't self-verify, it still capitulates.
- **The deepest result:** "sycophancy" is **not one circuit**. Both the salience flip and numeric sycophancy are the same *strategy* — read a referenced prompt token, copy it to the answer — but different cues route through **different heads** (a concentrated reader for salience; a diffuse mid-stack set for the asserted number). The salience reader L18.H5 carries ≈0 of the numeric copy.

*To implement:* the within-stack base-vs-it reader probe and the two distinct localisations, shown side by side.
""")

md(r"""
## 6. The honest ledger — and how to verify or share a claim like this  *(outline)*

> Mirrors LINEAGE Part 5 / Synthesis. Source: `POSITIONING.md`, `CONTRIBUTING.md`, `visual/`.

- **Scope:** one model family (Gemma 2B/9B), small N (5 pairs, 5–60 products), single seed, greedy/teacher-forced readouts, one phrasing per cue. Mechanism claims are well-supported *within* that scope; generalisation claims are explicitly bounded.
- **"Necessity" has known artifacts:** it can exceed 1.0 (over-correction); the knockout (zero a key at all layers/queries + renormalise) is heavy and somewhat unphysical; below a 0.5-nat effect floor it is reported as n/a.
- **Two senses of "sycophancy"** (base next-token priming vs RLHF assistant-agreement) are kept distinct on purpose.
- **The credibility comes partly from the retractions** — §3.10 corrects §3.9; §10.1 retracts a monotonic-pull trend. A summary that omits them overclaims.
- **Sharing:** a head-level *causal* claim doesn't fit the frictionless venues (attribution-graph sharing freezes attention). The proposed artifact is two-layer — an observational layer (Neuronpedia feature/graph deep-links) plus a causal layer (this notebook). Every number in `visual/index.html` is loaded from a committed `out/*.json` and re-printed on build, so the page is checkable against raw artifacts.

**The through-line:** the same instrument you ran in Section 1 on a manufactured ground truth scales — unchanged in form — all the way to "sycophancy is a copy strategy routed through different circuits." That is what *latent verification* means.
""")

# ----------------------------------------------------------------------------
# Serialise to .ipynb
# ----------------------------------------------------------------------------
def to_cell(kind, source):
    base = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        base["outputs"] = []
        base["execution_count"] = None
    return base


notebook = {
    "cells": [to_cell(kind, src) for kind, src in cells],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent / "latent_verify_first_principles.ipynb"
out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"wrote {out}  ({len(cells)} cells)")
