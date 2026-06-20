"""Distributed (GROUP) entropy-neuron ablation in late-layer MLPs of gemma-2-9b, base vs -it, measured
PRE- and POST-softcap.

CONTEXT. The sibling control controls/entropy_neuron_gemma2.py screens late-layer MLP neurons by a
weights-only null-space signature (a neuron whose output direction w_out writes mostly into the
unembedding's LOW-sensitivity subspace) and validates SINGLE neurons causally by mean/zero ablation,
reporting (dEntropy, dLoss). This control asks two orthogonal questions that single-neuron ablation
cannot, and attaches NO hypothesis about which answer holds:

  (1) Is an entropy effect DISTRIBUTED across many candidate neurons rather than carried by any one?
      It ablates the top-G candidates TOGETHER in a single forward over a cumulative ramp
      GRAMP = [1, 2, 4, 8, 16, 32] (nested: the G=1 set is a subset of G=2, ...), in BOTH group
      mean-ablation (every chosen channel set to its cached clean mean at all positions) and group
      zero-ablation (every chosen channel set to 0). A fixed-seed matched-random set of the SAME sizes,
      drawn from late-layer neurons NOT among the candidates, gives a specificity baseline at each G.

  (2) Is a measured entropy move an artifact of gemma-2's final-logit softcap, or present in the raw
      logits? Entropy is computed TWO ways, separately and reported side by side:
        POST-softcap : the model's returned logits (logits = sc * tanh(raw / sc)), as the sibling does.
        PRE-softcap  : the raw logits before the softcap. Recovered FAITHFULLY by temporarily setting
                       model.cfg.final_logit_softcap = None (the TransformerLens config field; the
                       HuggingFace name final_logit_softcapping is also cleared defensively) and running
                       a SECOND forward, then restoring the original value in a finally block. This is
                       the model's own forward with the cap disabled -- not an external re-derivation --
                       so the only thing that changes is whether the final tanh squash is applied.

  WEIGHT SCREEN (weights only). In float32 build C = W_U @ W_U.T (d_model x d_model; W_U is model.W_U
  [d_model, d_vocab], MEAN-CENTERED over the vocab axis before forming C so the constant direction does
  not dominate). Eigendecompose C (symmetric); take the K=50 eigenvectors with the SMALLEST eigenvalues
  -> N [d_model, K] (the unembedding's least-sensitive directions). For each late-layer MLP neuron i
  (layer index >= floor(late_frac * n_layers), late_frac=0.667) with w_out = model.W_out[L][i],
  null_frac(L,i) = ||N.T @ w_out|| / (||w_out|| + 1e-9). Rank late-layer neurons by null_frac; keep the
  top K_CAND=32 candidates. GRAMP indexes into these ranked candidates (G=top-G by null_frac).

  CAUSAL (forward). Over a long-context WikiText-2 reference (the entropy-neuron literature's regime;
  reused loader, short-sentence fallback) one clean forward per sequence gives the baseline mean-over-
  positions entropy (POST and PRE) and teacher-forced next-token CE, and the cached
  blocks.{L}.mlp.hook_post mean per candidate. For each G and each ablation mode, the top-G candidates
  are ablated together in one forward; dEntropy = entropy_ablated - baseline_entropy and
  dLoss = loss_ablated - baseline_loss (means over predicting positions), per G, per ablation mode, per
  softcap mode. The matched-random set of size G gets the identical treatment.

NEUTRAL DECISION (module-constant thresholds; reports numbers + categories only, asserts nothing about
which model, which grain G, or which softcap mode should qualify):
  Per (G, ablation_mode, softcap_mode):
    GROUP_ENTROPY_EFFECT iff group |dEntropy| >= ENT_TOL AND |dLoss| <= LOSS_TOL
                          AND (group |dEntropy| - matched_random_G |dEntropy|) >= MARGIN; else NONE.
  Per (G, ablation_mode): report pre_minus_post = (PRE-softcap |dEntropy|) - (POST-softcap |dEntropy|).
  Per model: emit the smallest G reaching GROUP_ENTROPY_EFFECT (or null) SEPARATELY for PRE- and
  POST-softcap (across ablation modes -- the smallest G that qualifies under either mean or zero).
  No claim is attached to any number, category, model, or grain.

Run model-free selftest (no model load, CPU):
    python controls/entropy_distributed_presoftcap.py --selftest
Run the measurement (9b; needs the GPU box):
    python controls/entropy_distributed_presoftcap.py --device cuda --ref long --k 50 \
        --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b
"""

import argparse
import json
import math
from pathlib import Path

import torch

# The pure functions below are RE-IMPLEMENTED from controls/entropy_neuron_gemma2.py (not imported): on
# the box the controls are scp'd FLAT into latent_verify/, so `from controls.entropy_neuron_gemma2 import`
# would not resolve, and re-implementing keeps --selftest standalone on CPU with no model load and nothing
# else on sys.path. Conventions (null_basis / null_frac / entropy_of_logits / ce_next_token / late-layer
# floor selection / _long_reference + REFERENCE_TEXT) are matched verbatim to the sibling.

# Pre-registered thresholds + params (see module docstring). Neutral: stated on the measured numbers only.
K = 50                # number of smallest-eigenvalue (least-sensitive) unembedding directions in the null basis N
K_CAND = 32           # late-layer neurons kept as candidates after ranking by null_frac
SEED = 0              # fixed seed -> reproducible matched-random selection
GRAMP = (1, 2, 4, 8, 16, 32)   # cumulative (nested) group sizes: top-G candidates by null_frac
ENT_TOL = 0.05        # group |dEntropy| >= this many nats is required for GROUP_ENTROPY_EFFECT
LOSS_TOL = 0.02       # |dLoss| <= this many nats is required (group ablation barely moves loss)
MARGIN = 0.02         # (group |dEntropy| - matched_random_G |dEntropy|) >= this nats required (specificity)

ABLATION_MODES = ("mean", "zero")
SOFTCAP_MODES = ("post", "pre")

DECISION_RULE = (
    "weights: null_frac(L,i) = ||N.T @ W_out[L][i]|| / ||W_out[L][i]||, N = the K=50 smallest-eigenvalue "
    "eigenvectors of mean-centered W_U @ W_U.T; candidates = top K_CAND=32 late-layer neurons by null_frac "
    "(late = layer >= floor(late_frac * n_layers)). GROUP ablation over GRAMP=[1,2,4,8,16,32] (nested: top-G "
    "candidates): mean (each chosen channel -> its cached clean mean at all positions) AND zero (each -> 0), "
    "ablated together in one forward over a long-context WikiText reference. dEntropy = entropy_ablated - "
    "baseline, dLoss = loss_ablated - baseline (nats), entropy computed POST-softcap (returned logits) AND "
    "PRE-softcap (cfg.final_logit_softcap=None second forward, restored after). Per (G, ablation_mode, "
    "softcap_mode): GROUP_ENTROPY_EFFECT iff |dEntropy|>=0.05 AND |dLoss|<=0.02 AND (|dEntropy| - "
    "matched_random_G |dEntropy|)>=0.02 else NONE. Per (G, mode): pre_minus_post = (pre |dEntropy|) - "
    "(post |dEntropy|). Per model: smallest G reaching GROUP_ENTROPY_EFFECT (or null) separately for pre and "
    "post. Numbers + categories only; no claim attached to any model, grain, or softcap mode."
)

MODELS = ("base", "it")

# Fixed reference set (short-sentence fallback when the long WikiText reference is unavailable): ~40 short
# generic declarative English sentences. Plain text, no chat template; BOS prepended at tokenization.
REFERENCE_TEXT = [
    "The sun rose slowly over the quiet harbor.",
    "She poured a cup of coffee and read the morning paper.",
    "The train arrived at the station a few minutes late.",
    "A small dog ran across the wet green field.",
    "The library was silent except for the turning of pages.",
    "He fixed the broken fence before the rain started.",
    "The children laughed as the kite climbed higher.",
    "Fresh bread was cooling on the kitchen counter.",
    "The river wound gently through the narrow valley.",
    "Old photographs filled the box in the attic.",
    "The teacher wrote the date on the chalkboard.",
    "Two sailors checked the ropes before the voyage.",
    "The market was crowded with shoppers on Saturday.",
    "Snow covered the rooftops of the small town.",
    "A gardener pruned the roses along the stone wall.",
    "The orchestra tuned their instruments before the concert.",
    "He parked the car and walked into the office.",
    "The lake reflected the colors of the evening sky.",
    "A letter arrived from a friend who had moved away.",
    "The baker opened the shop just before dawn.",
    "Wind rattled the windows during the long night.",
    "The museum displayed paintings from many countries.",
    "She planted tomatoes and herbs in the back garden.",
    "The old clock on the wall had stopped at noon.",
    "Travelers waited patiently at the busy airport gate.",
    "The cat slept in a warm patch of sunlight.",
    "He counted the coins and put them in a jar.",
    "The bridge crossed the river near the old mill.",
    "Students gathered in the hall before the lecture.",
    "A candle flickered on the wooden dinner table.",
    "The farmer drove the tractor across the dusty road.",
    "Rain fell steadily on the quiet city streets.",
    "The bookstore smelled of paper and fresh ink.",
    "She tied her shoes and went for an early run.",
    "The mountains stood tall against the pale blue sky.",
    "A waiter brought soup and bread to the table.",
    "The factory whistle blew at the end of the shift.",
    "He repaired the radio with a small screwdriver.",
    "The boat drifted gently toward the sandy shore.",
    "Leaves fell from the trees in the cool autumn air.",
]


# --------------------------------------------------------------------------- weight side (null basis)
def null_basis(W_U, k=K):
    """The unembedding's K LEAST-sensitive directions. W_U is [d_model, d_vocab]. MEAN-CENTER over the
    vocab axis, form C = Wc @ Wc.T (d_model x d_model, float32), eigendecompose (symmetric), and return the
    eigenvectors for the K SMALLEST eigenvalues as N [d_model, k]. Pure. Upcasts to float32 first (bf16
    eigendecomposition is unsafe)."""
    Wc = W_U.float()
    Wc = Wc - Wc.mean(dim=1, keepdim=True)            # mean-center over vocab (drop the constant direction)
    C = Wc @ Wc.T                                     # [d_model, d_model], symmetric PSD
    evals, evecs = torch.linalg.eigh(C)               # ascending eigenvalues; columns are eigenvectors
    k = min(k, evecs.shape[1])
    return evecs[:, :k].contiguous()                  # the k smallest-eigenvalue directions


def null_frac(N, w_out):
    """Fraction of w_out's norm that lies in the null basis N [d_model, k]. w_out [d_model]. Pure.
    null_frac = ||N.T @ w_out|| / (||w_out|| + 1e-9)."""
    w = w_out.float()
    proj = N.T @ w                                    # [k] coordinates in the null basis (N orthonormal)
    return float(proj.norm() / (w.norm() + 1e-9))


# --------------------------------------------------------------------------- output-distribution math
def entropy_of_logits(logits):
    """Shannon entropy (nats) of softmax(logits) along the last dim. logits [..., d_vocab] -> [...]. Pure.
    Uses log_softmax for numerical stability: H = -sum p * logp. Upcast to float32 first."""
    logits = logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


def ce_next_token(logits, ids):
    """Teacher-forced next-token cross-entropy (nats) per predicting position. logits [seq, d_vocab],
    ids [seq] (the token sequence incl BOS). Position j predicts token ids[j+1], so there are seq-1
    predicting positions. Returns a [seq-1] tensor of per-position CE. Pure. Upcast to float32 first."""
    logits = logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    pred = logp[:-1]                                  # [seq-1, d_vocab] predictions for positions 0..seq-2
    tgt = ids[1:]                                     # [seq-1] the realized next tokens
    return -pred[torch.arange(pred.shape[0]), tgt]    # [seq-1]


def softcap_logits(raw, sc):
    """Apply gemma-2's final-logit softcap to raw logits: sc * tanh(raw / sc); identity if sc is falsy.
    Pure. Used ONLY in --selftest to exercise the pre/post entropy transform on synthetic logits; the
    real measurement gets pre/post from the model's own forward (cfg toggle), not this re-derivation."""
    if not sc:
        return raw
    raw = raw.float()
    return sc * torch.tanh(raw / sc)


# --------------------------------------------------------------------------- group ablation hook
def group_ablate_hook_factory(idx_to_val):
    """Build ONE forward hook for a single blocks.{L}.mlp.hook_post that sets EACH channel i in the dict
    idx_to_val {channel_index: value} to its target value at EVERY (batch,pos) position, in one forward.
    Group mean-ablation passes the cached clean means; group zero-ablation passes 0.0 for each. Other
    channels are left untouched; the cast to the activation dtype keeps a bf16 forward intact. Pure
    constructor; returns a hook fn (act, hook) -> act."""
    items = tuple(idx_to_val.items())
    def hook(act, hook=None, _items=items):
        for i, v in _items:
            act[..., i] = torch.as_tensor(v, dtype=act.dtype, device=act.device)
        return act
    return hook


def cumulative_groups(candidates, gramp=GRAMP):
    """Map each G in gramp to the top-G candidates (the first G entries of the null_frac-ranked list).
    Returns {G: [(L, i, null_frac), ...]} -- nested by construction (G=1 subset of G=2 ...). Pure."""
    out = {}
    for G in gramp:
        g = min(G, len(candidates))
        out[G] = list(candidates[:g])
    return out


def group_by_layer(group):
    """Group a list of (L, i, nf) candidate triples into {L: [i, ...]} for per-layer hooking. Pure."""
    by_L = {}
    for (L, i, _nf) in group:
        by_L.setdefault(L, []).append(i)
    return by_L


# --------------------------------------------------------------------------- neutral verdict
def decide(dent, dloss, rand_dent, ent_tol=ENT_TOL, loss_tol=LOSS_TOL, margin=MARGIN):
    """GROUP_ENTROPY_EFFECT iff |dEntropy| >= ENT_TOL AND |dLoss| <= LOSS_TOL AND
    (|dEntropy| - |matched_random dEntropy|) >= MARGIN; else NONE. Pure. Asserts nothing about which G,
    model, or softcap mode should qualify."""
    fires = (abs(dent) >= ent_tol) and (abs(dloss) <= loss_tol) and ((abs(dent) - abs(rand_dent)) >= margin)
    return "GROUP_ENTROPY_EFFECT" if fires else "NONE"


# --------------------------------------------------------------------------- real measurement
def _late_layers(n_layers, late_frac):
    """Layer indices counted as 'late': index >= floor(late_frac * n_layers). Pure."""
    start = int(math.floor(late_frac * n_layers))
    start = max(0, min(start, n_layers - 1))
    return list(range(start, n_layers))


def _screen_weights(model, late_layers, k, k_cand, n_rand, seed):
    """Weights-only screen. Returns (candidates, randoms, d_mlp): candidates = top k_cand late-layer
    neurons by null_frac (each (L, i, null_frac)); randoms = n_rand late-layer neurons NOT in the
    candidate set (fixed seed). Computes null_frac for every late-layer neuron from N = null_basis(W_U)."""
    import random as _r
    N = null_basis(model.W_U, k)
    W_out = model.W_out                               # [n_layers, d_mlp, d_model]
    d_mlp = W_out.shape[1]
    scored = []
    for L in late_layers:
        WL = W_out[L].float()                         # [d_mlp, d_model]
        for i in range(d_mlp):
            scored.append((L, i, null_frac(N, WL[i])))
    scored.sort(key=lambda t: t[2], reverse=True)     # highest null_frac first
    candidates = scored[:k_cand]
    cand_keys = {(L, i) for (L, i, _f) in candidates}
    pool = [(L, i, f) for (L, i, f) in scored if (L, i) not in cand_keys]
    rng = _r.Random(seed)
    randoms = rng.sample(pool, min(n_rand, len(pool)))
    return candidates, randoms, d_mlp


def _softcap_value(model):
    """The model's gemma-2 final-logit softcap value (float) or None. Reads the TransformerLens config
    field final_logit_softcap, with the HuggingFace name final_logit_softcapping as a fallback."""
    sc = getattr(model.cfg, "final_logit_softcap", None)
    if sc is None:
        sc = getattr(model.cfg, "final_logit_softcapping", None)
    return sc


class _no_softcap:
    """Context manager: temporarily DISABLE the model's final-logit softcap so a forward returns the raw
    PRE-softcap logits, then RESTORE the original value on exit (even on error). Sets both the
    TransformerLens name (final_logit_softcap) and the HuggingFace name (final_logit_softcapping) to None
    so whichever the build honours is cleared. Faithful: it is the model's own forward with the final
    tanh squash removed, nothing re-derived externally."""
    _FIELDS = ("final_logit_softcap", "final_logit_softcapping")

    def __init__(self, model):
        self.cfg = model.cfg
        self.saved = {}

    def __enter__(self):
        for f in self._FIELDS:
            if hasattr(self.cfg, f):
                self.saved[f] = getattr(self.cfg, f)
                setattr(self.cfg, f, None)
        return self

    def __exit__(self, *exc):
        for f, v in self.saved.items():
            setattr(self.cfg, f, v)
        return False


def _entropy_loss_from_logits(lg, ids):
    """Per-sequence (entropy_sum, entropy_cnt, loss_sum, loss_cnt) from one [seq, d_vocab] logits tensor
    and its [seq] id tensor. Entropy over all positions; CE over the seq-1 predicting positions. Pure."""
    ent = entropy_of_logits(lg)                       # [seq]
    ce = ce_next_token(lg, ids)                       # [seq-1]
    return float(ent.sum()), ent.shape[0], float(ce.sum()), ce.shape[0]


def _clean_pass(model, ref_ids, layers_needed, device):
    """One clean forward PER reference sequence (no padding). Returns:
      base_ent_post, base_ent_pre : mean-over-positions entropy (POST- and PRE-softcap)
      base_loss                   : mean-over-positions teacher-forced next-token CE (softcap-invariant
                                    for argmax-correct targets but recomputed POST-softcap, the model's
                                    default returned logits, to match the sibling baseline)
      mean_act                    : {L: [d_mlp]} mean of blocks.{L}.mlp.hook_post over all (seq,pos)
    For each sequence: one cached POST-softcap forward (baseline entropy/loss + activations), then one
    PRE-softcap forward (cfg softcap disabled) for the PRE-softcap baseline entropy."""
    post_names = {f"blocks.{L}.mlp.hook_post" for L in layers_needed}
    nf = lambda n: n in post_names                    # noqa: E731
    ep_sum = ec = 0          # entropy POST sum / count
    er_sum = 0              # entropy PRE sum (same count ec -- same positions)
    ls_sum = lc = 0          # loss sum / count
    act_sum = {L: None for L in layers_needed}
    pos_total = 0
    for ids in ref_ids:
        with torch.no_grad():
            logits, cache = model.run_with_cache(ids, names_filter=nf)
        lg = logits[0]                                # [seq, d_vocab], POST-softcap
        es, en, lss, lcn = _entropy_loss_from_logits(lg, ids[0])
        ep_sum += es; ec += en; ls_sum += lss; lc += lcn
        with torch.no_grad(), _no_softcap(model):
            lg_pre = model(ids)[0]                    # [seq, d_vocab], PRE-softcap
        er_sum += float(entropy_of_logits(lg_pre).sum())
        for L in layers_needed:
            post = cache[f"blocks.{L}.mlp.hook_post"][0].float()   # [seq, d_mlp]
            s = post.sum(dim=0)                                    # [d_mlp]
            act_sum[L] = s if act_sum[L] is None else act_sum[L] + s
        pos_total += lg.shape[0]
    mean_act = {L: (act_sum[L] / max(pos_total, 1)) for L in layers_needed}
    return (ep_sum / max(ec, 1), er_sum / max(ec, 1), ls_sum / max(lc, 1), mean_act)


def _group_ablated_pass(model, ref_ids, by_L, mode, mean_act):
    """Group-ablate the channels in by_L {L: [i,...]} over the reference set in ONE forward per sequence,
    measuring entropy POST- and PRE-softcap and loss. mode is 'mean' (each channel -> its cached clean
    mean from mean_act[L][i]) or 'zero' (each channel -> 0.0). Builds one hook per layer covering all that
    layer's chosen channels. Returns (ent_post_mean, ent_pre_mean, loss_mean) over predicting positions.
    The PRE-softcap entropy comes from a second forward with cfg softcap disabled, SAME hooks applied."""
    fwd_hooks = []
    for L, idxs in by_L.items():
        if mode == "mean":
            idx_to_val = {i: float(mean_act[L][i]) for i in idxs}
        else:
            idx_to_val = {i: 0.0 for i in idxs}
        fwd_hooks.append((f"blocks.{L}.mlp.hook_post", group_ablate_hook_factory(idx_to_val)))
    ep_sum = ec = 0
    er_sum = 0
    ls_sum = lc = 0
    for ids in ref_ids:
        with torch.no_grad():
            lg = model.run_with_hooks(ids, fwd_hooks=fwd_hooks)[0]      # POST-softcap
        es, en, lss, lcn = _entropy_loss_from_logits(lg, ids[0])
        ep_sum += es; ec += en; ls_sum += lss; lc += lcn
        with torch.no_grad(), _no_softcap(model):
            lg_pre = model.run_with_hooks(ids, fwd_hooks=fwd_hooks)[0]  # PRE-softcap
        er_sum += float(entropy_of_logits(lg_pre).sum())
    return (ep_sum / max(ec, 1), er_sum / max(ec, 1), ls_sum / max(lc, 1))


def _long_reference(model, device, n_seq=20, seqlen=256):
    """Long-context reference (the regime the entropy-neuron literature uses): WikiText-2 test split,
    tokenized and chunked into seqlen-token BOS-prepended windows. Returns a list of [1, seqlen] id
    tensors, or None on any failure (caller falls back to the short REFERENCE_TEXT)."""
    try:
        from datasets import load_dataset
        try:
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")   # namespaced (current)
        except Exception:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")              # legacy id fallback
        text = "\n".join(t for t in ds["text"] if t and not t.isspace())
        ids = model.tokenizer(text, add_special_tokens=False)["input_ids"]
        bos = model.tokenizer.bos_token_id
        out, step = [], seqlen - 1
        for s in range(0, len(ids) - step, step):
            out.append(torch.tensor([[bos] + ids[s:s + step]], device=device))
            if len(out) >= n_seq:
                break
        return out or None
    except Exception as e:
        print(f"[ref] long reference unavailable ({type(e).__name__}: {e}); falling back to short", flush=True)
        return None


def _per_model(name, device, late_frac, k, k_cand, ref_mode="long"):
    """Load one model, run the weight screen + clean baseline (POST + PRE) + GROUP ablation over GRAMP for
    candidates AND a matched-random set (both ablation modes, both softcap modes). Returns the per-model
    result dict including the per-(G,mode,softcap) categories and the smallest-G-to-effect per softcap."""
    from transformer_lens import HookedTransformer
    print(f"[load] {name} on {device}", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    n_layers = model.cfg.n_layers
    sc = _softcap_value(model)
    late_layers = _late_layers(n_layers, late_frac)
    print(f"[screen] n_layers={n_layers} softcap={sc} late_layers={late_layers[0]}..{late_layers[-1]} "
          f"(>= floor({late_frac}*{n_layers}))", flush=True)

    n_rand = max(GRAMP)                                # matched-random pool size = largest group size
    candidates, randoms, d_mlp = _screen_weights(model, late_layers, k, k_cand, n_rand, SEED)
    print(f"[screen] d_mlp={d_mlp} | top null_frac {candidates[0][:2]} nf={candidates[0][2]:.4f} "
          f"| lowest kept {candidates[-1][:2]} nf={candidates[-1][2]:.4f}", flush=True)

    ref_ids, ref_used = None, "short"
    if ref_mode == "long":
        ref_ids = _long_reference(model, device)
        ref_used = f"long_wikitext_{len(ref_ids)}x{ref_ids[0].shape[1]}" if ref_ids else "short_fallback"
    if ref_ids is None:
        ref_ids = [model.to_tokens(s, prepend_bos=True).to(device) for s in REFERENCE_TEXT]
        ref_used = "short" if ref_mode != "long" else "short_fallback"
    print(f"[ref] {ref_used}: {len(ref_ids)} sequences, "
          f"{sum(x.shape[1] for x in ref_ids)} positions", flush=True)

    # cumulative (nested) candidate groups + matched-random groups of the same sizes
    cand_groups = cumulative_groups(candidates, GRAMP)
    rand_groups = cumulative_groups(randoms, GRAMP)
    layers_needed = sorted({L for (L, _i, _f) in candidates} | {L for (L, _i, _f) in randoms})

    base_ent_post, base_ent_pre, base_loss, mean_act = _clean_pass(model, ref_ids, layers_needed, device)
    print(f"[baseline] entropy post={base_ent_post:.4f} pre={base_ent_pre:.4f} loss={base_loss:.4f} (nats)",
          flush=True)

    base_ent = {"post": base_ent_post, "pre": base_ent_pre}

    # ---- group ablation per G, per ablation mode; candidate group AND matched-random group ----
    per_G = []
    smallest_G = {"post": None, "pre": None}          # smallest G to GROUP_ENTROPY_EFFECT (either abl mode)
    for G in GRAMP:
        cand_by_L = group_by_layer(cand_groups[G])
        rand_by_L = group_by_layer(rand_groups[G])
        row = {"G": G,
               "candidate_neurons": [{"L": L, "i": i, "null_frac": round(nf, 4)} for (L, i, nf) in cand_groups[G]],
               "modes": {}}
        for mode in ABLATION_MODES:
            cep, cpr, closs = _group_ablated_pass(model, ref_ids, cand_by_L, mode, mean_act)
            rep, rpr, _rl = _group_ablated_pass(model, ref_ids, rand_by_L, mode, mean_act)
            cand_d = {"post": cep - base_ent["post"], "pre": cpr - base_ent["pre"]}
            rand_d = {"post": rep - base_ent["post"], "pre": rpr - base_ent["pre"]}
            dloss = closs - base_loss
            mode_out = {"dLoss": round(dloss, 4),
                        "pre_minus_post": round(abs(cand_d["pre"]) - abs(cand_d["post"]), 4),
                        "softcap": {}}
            for sm in SOFTCAP_MODES:
                cat = decide(cand_d[sm], dloss, rand_d[sm])
                if cat == "GROUP_ENTROPY_EFFECT" and (smallest_G[sm] is None or G < smallest_G[sm]):
                    smallest_G[sm] = G
                mode_out["softcap"][sm] = {
                    "group_dEntropy": round(cand_d[sm], 4),
                    "matched_random_dEntropy": round(rand_d[sm], 4),
                    "specificity_gap": round(abs(cand_d[sm]) - abs(rand_d[sm]), 4),
                    "category": cat,
                }
            row["modes"][mode] = mode_out
            print(f"  G={G:>2} {mode:>4}: dLoss={dloss:+.4f} | "
                  f"post dEnt grp/rnd={cand_d['post']:+.4f}/{rand_d['post']:+.4f}[{mode_out['softcap']['post']['category']}] | "
                  f"pre dEnt grp/rnd={cand_d['pre']:+.4f}/{rand_d['pre']:+.4f}[{mode_out['softcap']['pre']['category']}] | "
                  f"pre-post={mode_out['pre_minus_post']:+.4f}", flush=True)
        per_G.append(row)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"[{name}] smallest G to GROUP_ENTROPY_EFFECT: post={smallest_G['post']} pre={smallest_G['pre']}",
          flush=True)

    return {"n_layers": n_layers, "late_layers": late_layers, "d_mlp": d_mlp, "softcap": sc,
            "ref_used": ref_used,
            "baseline_entropy_post": round(base_ent_post, 4),
            "baseline_entropy_pre": round(base_ent_pre, 4),
            "baseline_loss": round(base_loss, 4),
            "candidates": [{"L": L, "i": i, "null_frac": round(nf, 4)} for (L, i, nf) in candidates],
            "matched_random_pool": [{"L": L, "i": i, "null_frac": round(nf, 4)} for (L, i, nf) in randoms],
            "per_G": per_G,
            "smallest_G_to_effect": smallest_G}


def run(name_base, name_it, tag, device, late_frac, k, k_cand, ref_mode="long"):
    res = {"base": _per_model(name_base, device, late_frac, k, k_cand, ref_mode),
           "it": _per_model(name_it, device, late_frac, k, k_cand, ref_mode)}

    out = {
        "name_base": name_base, "name_it": name_it, "device": device, "tag": tag,
        "cue": "entropy_distributed_presoftcap",
        "reference_mode": ref_mode, "reference_used": {m: res[m]["ref_used"] for m in MODELS},
        "params": {"K": k, "K_CAND": k_cand, "GRAMP": list(GRAMP), "late_frac": late_frac, "SEED": SEED},
        "thresholds": {"ENT_TOL": ENT_TOL, "LOSS_TOL": LOSS_TOL, "MARGIN": MARGIN},
        "decision_rule": DECISION_RULE,
        "base": res["base"], "it": res["it"],
        "smallest_G_to_effect": {m: res[m]["smallest_G_to_effect"] for m in MODELS},
    }
    Path("out").mkdir(exist_ok=True)
    Path(f"out/entropy_distributed_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[done] base smallest-G post/pre={res['base']['smallest_G_to_effect']['post']}/"
          f"{res['base']['smallest_G_to_effect']['pre']} | "
          f"it smallest-G post/pre={res['it']['smallest_G_to_effect']['post']}/"
          f"{res['it']['smallest_G_to_effect']['pre']}", flush=True)
    print(f"[done] wrote out/entropy_distributed_{tag}.json", flush=True)


# --------------------------------------------------------------------------- selftest (model-free)
def selftest():
    torch.manual_seed(0)

    # ---- (i) pre/post softcap entropy transform on synthetic logits ----
    # A known PRE-softcap vector with some LARGE-magnitude logits. The cap post = sc*tanh(pre/sc) SQUASHES
    # the large magnitudes toward +/- sc, COMPRESSING the gaps between logits, which RAISES the softmax
    # entropy. So entropy(post) > entropy(pre): the cap can only flatten, never sharpen, a distribution.
    sc = 30.0
    pre = torch.tensor([[80.0, 40.0, 5.0, -10.0, -60.0]])     # spread well beyond +/- sc -> heavy squash
    post = softcap_logits(pre, sc)
    assert torch.allclose(post, sc * torch.tanh(pre / sc)), post     # the documented transform
    H_pre = float(entropy_of_logits(pre)[0])
    H_post = float(entropy_of_logits(post)[0])
    assert H_post > H_pre + 1e-3, (H_pre, H_post)                    # squash flattens -> entropy rises
    # and a vector already inside +/- sc is barely moved -> entropies nearly equal
    small = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
    assert abs(float(entropy_of_logits(softcap_logits(small, sc))[0]) - float(entropy_of_logits(small)[0])) < 0.05
    # sc falsy -> identity -> entropies exactly equal
    assert torch.allclose(softcap_logits(pre, None), pre)
    assert abs(float(entropy_of_logits(softcap_logits(pre, 0))[0]) - H_pre) < 1e-6
    print(f"[selftest] (i) softcap entropy: pre={H_pre:.4f} < post={H_post:.4f} (cap squashes -> flatter); "
          f"in-range nearly unchanged; sc=None is identity")

    # entropy sanity (matches the sibling): uniform -> ln n, near one-hot -> ~0
    assert abs(float(entropy_of_logits(torch.zeros(1, 8))[0]) - math.log(8)) < 1e-5
    assert float(entropy_of_logits(torch.tensor([[100.0, 0.0, 0.0, 0.0]]))[0]) < 1e-3

    # ---- (ii) group-ablation hook sets exactly the G chosen channels; others intact; dtype preserved ----
    act = torch.arange(2 * 3 * 6, dtype=torch.float32).reshape(2, 3, 6).clone()    # [batch,pos,d_mlp=6]
    before = act.clone()
    chosen = {1: 0.5, 4: -2.0, 5: 0.0}                       # G=3 channels with distinct target values
    out_act = group_ablate_hook_factory(chosen)(act)
    for i, v in chosen.items():
        assert torch.all(out_act[..., i] == v), (i, out_act[..., i])     # each chosen channel -> its value, everywhere
    for ch in (0, 2, 3):                                     # the non-chosen channels untouched
        assert torch.all(out_act[..., ch] == before[..., ch]), ch
    # zero-mode: every chosen channel -> 0
    actz = torch.ones(1, 2, 6, dtype=torch.float32)
    outz = group_ablate_hook_factory({i: 0.0 for i in (0, 3)})(actz)
    assert torch.all(outz[..., 0] == 0.0) and torch.all(outz[..., 3] == 0.0)
    assert torch.all(outz[..., 1] == 1.0) and torch.all(outz[..., 2] == 1.0)      # others kept
    # dtype preserved (bf16 forward stays bf16)
    actb = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    outb = group_ablate_hook_factory({1: 1.25, 2: -3.0})(actb)
    assert outb.dtype == torch.bfloat16, outb.dtype
    assert float(outb[0, 0, 1]) != 0.0 and float(outb[0, 0, 2]) != 0.0
    print("[selftest] (ii) group hook: exactly the G chosen channels set (mean or 0) at all positions; others intact; dtype kept")

    # ---- (iii) decide() fires only when all three gates pass, not otherwise ----
    # gate 1: |dEntropy| >= ENT_TOL ; gate 2: |dLoss| <= LOSS_TOL ; gate 3: |dEnt| - |rand| >= MARGIN
    assert decide(0.10, 0.005, 0.00) == "GROUP_ENTROPY_EFFECT"           # all three pass
    assert decide(-0.10, 0.005, 0.00) == "GROUP_ENTROPY_EFFECT"          # sign-agnostic (|dEntropy|)
    assert decide(0.01, 0.005, 0.00) == "NONE"                          # gate 1 fails (entropy too small)
    assert decide(0.10, 0.10, 0.00) == "NONE"                           # gate 2 fails (loss too big)
    assert decide(0.10, -0.10, 0.00) == "NONE"                          # gate 2 fails (negative side)
    assert decide(0.10, 0.005, 0.09) == "NONE"                          # gate 3 fails (random ~ as large)
    assert decide(0.10, 0.005, -0.10) == "NONE"                        # random equal |magnitude| (opposite sign) -> |dEnt|-|rand| = 0 -> fails (specificity gap is sign-agnostic)
    # exactly at the thresholds -> passes (>= / <= boundaries): dEnt=ENT_TOL, dLoss=LOSS_TOL, gap=MARGIN
    assert decide(ENT_TOL, LOSS_TOL, ENT_TOL - MARGIN) == "GROUP_ENTROPY_EFFECT"
    assert decide(ENT_TOL, -LOSS_TOL, ENT_TOL - MARGIN) == "GROUP_ENTROPY_EFFECT"
    # just past each boundary -> NONE
    assert decide(ENT_TOL - 1e-6, LOSS_TOL, 0.0) == "NONE"              # entropy a hair under tol
    assert decide(ENT_TOL, LOSS_TOL + 1e-6, 0.0) == "NONE"             # loss a hair over tol
    assert decide(ENT_TOL, LOSS_TOL, ENT_TOL - MARGIN + 1e-6) == "NONE"   # gap a hair under margin
    print("[selftest] (iii) decide(): fires iff |dEntropy|>=ENT_TOL AND |dLoss|<=LOSS_TOL AND (|dEntropy|-|rand|)>=MARGIN")

    # ---- (iv) cumulative ramp membership is nested (G=1 subset of G=2 ...) ----
    cands = [(40, i, 1.0 - 0.01 * i) for i in range(K_CAND)]            # 32 fake ranked candidates
    groups = cumulative_groups(cands, GRAMP)
    keys = lambda g: [(L, i) for (L, i, _f) in g]
    prev_set = set()
    for G in GRAMP:
        kg = keys(groups[G])
        assert len(kg) == min(G, len(cands)), (G, len(kg))             # top-G entries
        assert kg == keys(cands[:min(G, len(cands))]), G               # exactly the first G ranked candidates
        cur = set(kg)
        assert prev_set <= cur, (G, prev_set - cur)                    # nested: previous group is a subset
        prev_set = cur
    # group_by_layer collects channels per layer; total channel count preserved
    g8 = groups[8]
    by_L = group_by_layer(g8)
    assert sum(len(v) for v in by_L.values()) == len(g8)
    print(f"[selftest] (iv) cumulative ramp nested: sizes={[len(groups[G]) for G in GRAMP]} (each G subset of the next)")

    # ---- late-layer selection: index >= floor(late_frac * n_layers) (matches sibling) ----
    assert _late_layers(42, 0.667) == list(range(28, 42)), _late_layers(42, 0.667)   # gemma-2-9b: floor(28.01)=28
    assert _late_layers(46, 0.667) == list(range(30, 46)), _late_layers(46, 0.667)   # gemma-2-27b: floor(30.68)=30
    assert _late_layers(1, 0.667) == [0]                                             # degenerate-short clamp
    print("[selftest] late-layer cutoff = floor(late_frac * n_layers)")

    # ---- bf16 upcast guard: null_basis upcasts a bf16 W_U before eigendecomposition ----
    d_model, d_vocab = 6, 40
    Q, _ = torch.linalg.qr(torch.randn(d_model, d_model))
    s = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0, 1e-4])
    R = torch.randn(d_vocab, d_model)
    ones = torch.ones(d_vocab, 1)
    R = R - ones @ (ones.T @ R) / d_vocab
    Vt = torch.linalg.qr(R)[0].T
    W_U = Q @ torch.diag(s) @ Vt
    N1 = null_basis(W_U, k=1)
    assert null_frac(N1, Q[:, -1]) > 0.999 and null_frac(N1, Q[:, 0]) < 0.001       # null vs sensitive direction
    N_bf = null_basis(W_U.to(torch.bfloat16), k=1)
    assert N_bf.dtype == torch.float32, N_bf.dtype
    assert math.isfinite(null_frac(N_bf, Q[:, -1].to(torch.bfloat16)))
    print("[selftest] null_frac: bottom~1 top~0; bf16 guard -> N float32")

    print("[selftest] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--name-base", default="google/gemma-2-9b")
    p.add_argument("--name-it", default="google/gemma-2-9b-it")
    p.add_argument("--tag", default="9b")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                   help="cpu allowed for tiny smoke only; the 9b measurement needs cuda")
    p.add_argument("--late-frac", type=float, default=0.667,
                   help="late layers = index >= floor(late_frac * n_layers)")
    p.add_argument("--k", type=int, default=K, help="number of smallest-eigenvalue unembedding directions")
    p.add_argument("--k-cand", type=int, default=K_CAND, help="top null_frac late neurons kept as candidates")
    p.add_argument("--ref", default="long", choices=["short", "long"],
                   help="reference distribution: short (sentences) or long (WikiText-2 256-token windows)")
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        run(args.name_base, args.name_it, args.tag, args.device, args.late_frac, args.k, args.k_cand, args.ref)


if __name__ == "__main__":
    main()
