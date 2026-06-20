"""Entropy-neuron screen + causal mean-ablation in late-layer MLPs of gemma-2-9b, base vs -it.

CONTEXT. An "entropy neuron" is an MLP neuron whose output direction w_out writes mainly into the
unembedding's LOW-sensitivity subspace: the directions the unembedding W_U maps to (near-)zero logit
movement. Such a neuron can change the OUTPUT ENTROPY a lot (it shifts the residual norm / the part of
the distribution the unembedding barely resolves) while changing the next-token LOSS little. This control
finds candidates by a weights-only null-space signature and then validates them causally by mean-ablation,
separately for base and -it, with a matched-random specificity baseline. It measures two numbers per
candidate (dEntropy, dLoss in nats) plus the weight null_frac, and emits a neutral category. No hypothesis
is attached: which neurons qualify, and whether base and -it differ, are reported, not predicted.

  WEIGHT SCREEN (weights only). In float32 build C = W_U @ W_U.T  (d_model x d_model; W_U is
  model.W_U shape [d_model, d_vocab], MEAN-CENTERED over the vocab axis before forming C so the constant
  direction does not dominate). Eigendecompose C (symmetric); take the K eigenvectors with the SMALLEST
  eigenvalues -> N [d_model, K] (the unembedding's least-sensitive directions). For each MLP neuron i in
  the late layers (layer index >= floor(late_frac * n_layers)), w_out = model.W_out[L][i]
  (model.W_out shape [n_layers, d_mlp, d_model]); null_frac(L,i) = ||N.T @ w_out|| / (||w_out|| + 1e-9).
  Rank late-layer neurons by null_frac; the top N_CAND are the candidates.

  CAUSAL (forward). Over a fixed reference set of ~40 generic English sentences (no chat template, BOS
  prepended), one clean forward gives the baseline mean-over-positions entropy and teacher-forced
  next-token cross-entropy loss, and the cached blocks.{L}.mlp.hook_post mean activation per candidate
  neuron. For each candidate, mean-ablate (set blocks.{L}.mlp.hook_post[...,i] to its cached mean at every
  position) and recompute: dEntropy = entropy_ablated - baseline_entropy; dLoss = loss_ablated -
  baseline_loss; ratio = dEntropy / max(|dLoss|, 1e-6). A matched-random set of N_RAND late-layer neurons
  NOT among the candidates gets the same treatment (specificity baseline).

NEUTRAL DECISION (module-constant thresholds; reports numbers + categories only, asserts nothing about
which neurons or which model should qualify):
  Per candidate: ENTROPY_NEURON if null_frac >= NULL_TOL AND dEntropy >= ENT_TOL AND |dLoss| <= LOSS_TOL;
  else NOT. Per model, count the ENTROPY_NEURONs. If both models run, emit a base-vs-it differential keyed
  by (L,i): null_frac and dEntropy in each.

Run model-free selftest (no model load, CPU):
    python controls/entropy_neuron_gemma2.py --selftest
Run the measurement (9b; needs the GPU box):
    python controls/entropy_neuron_gemma2.py --device cuda --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b
"""

import argparse
import json
import math
from pathlib import Path

import torch

# Repo-internal / heavy imports (transformer_lens) are deferred into the functions that use them so
# --selftest runs standalone from controls/ on CPU with no model load and nothing else on sys.path.

# Pre-registered thresholds (see module docstring). Neutral: stated on the measured numbers only.
K = 10                # number of smallest-eigenvalue (least-sensitive) unembedding directions in the null basis N
N_CAND = 30           # late-layer neurons kept as candidates after ranking by null_frac
N_RAND = 8            # matched-random late-layer neurons (not candidates) for the specificity baseline
SEED = 0              # fixed seed -> reproducible random-neuron selection
NULL_TOL = 0.5        # null_frac >= this is required for ENTROPY_NEURON (weight is mostly in the null basis)
ENT_TOL = 0.05        # dEntropy >= this many nats is required for ENTROPY_NEURON (ablation moves entropy)
LOSS_TOL = 0.02       # |dLoss| <= this many nats is required for ENTROPY_NEURON (ablation barely moves loss)

DECISION_RULE = (
    "weights: null_frac(L,i) = ||N.T @ W_out[L][i]|| / ||W_out[L][i]||, N = the K=10 smallest-eigenvalue "
    "eigenvectors of mean-centered W_U @ W_U.T; candidates = top N_CAND=30 late-layer neurons by null_frac "
    "(late = layer >= floor(late_frac * n_layers)). Causal: mean-ablate hook_post[...,i] over the reference "
    "set; dEntropy = entropy_ablated - baseline, dLoss = loss_ablated - baseline (nats); ratio = "
    "dEntropy/max(|dLoss|,1e-6). Per candidate ENTROPY_NEURON if null_frac>=0.5 AND dEntropy>=0.05 AND "
    "|dLoss|<=0.02 else NOT. Matched-random N_RAND=8 non-candidate late neurons give a specificity baseline. "
    "Numbers + categories only; no claim attached to any neuron or model."
)

MODELS = ("base", "it")

# Fixed reference set: ~40 short generic declarative English sentences. Plain text, no chat template; BOS is
# prepended at tokenization. Used both for the mean activation and the entropy/loss baseline.
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


def mean_ablate_hook_factory(neuron_idx, mean_val):
    """Build a forward hook for blocks.{L}.mlp.hook_post that sets channel `neuron_idx` to `mean_val` at
    EVERY (batch,pos) position. Pure constructor; returns a hook fn (act, hook) -> act. The cast to the
    activation dtype keeps a bf16 forward intact."""
    def hook(act, hook=None, _i=neuron_idx, _m=mean_val):
        act[..., _i] = torch.as_tensor(_m, dtype=act.dtype, device=act.device)
        return act
    return hook


# --------------------------------------------------------------------------- neutral verdict
def decide(nf, dent, dloss, null_tol=NULL_TOL, ent_tol=ENT_TOL, loss_tol=LOSS_TOL):
    """ENTROPY_NEURON iff null_frac >= NULL_TOL AND dEntropy >= ENT_TOL AND |dLoss| <= LOSS_TOL; else NOT.
    Pure. Asserts nothing about which neuron should qualify."""
    is_en = (nf >= null_tol) and (dent >= ent_tol) and (abs(dloss) <= loss_tol)
    return "ENTROPY_NEURON" if is_en else "NOT"


def ratio_of(dent, dloss):
    """dEntropy / max(|dLoss|, 1e-6): large -> entropy moves a lot per unit loss movement. Pure."""
    return dent / max(abs(dloss), 1e-6)


# --------------------------------------------------------------------------- real measurement
def _late_layers(n_layers, late_frac):
    """Layer indices counted as 'late': index >= floor(late_frac * n_layers). Pure."""
    start = int(math.floor(late_frac * n_layers))
    start = max(0, min(start, n_layers - 1))
    return list(range(start, n_layers))


def _screen_weights(model, late_layers, k, n_cand, n_rand, seed):
    """Weights-only screen. Returns (candidates, randoms): each a list of (L, i, null_frac). Candidates =
    top n_cand late-layer neurons by null_frac; randoms = n_rand late-layer neurons NOT in the candidate
    set (fixed seed). Computes null_frac for every late-layer neuron from N = null_basis(W_U)."""
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
    candidates = scored[:n_cand]
    cand_keys = {(L, i) for (L, i, _f) in candidates}
    pool = [(L, i, f) for (L, i, f) in scored if (L, i) not in cand_keys]
    rng = _r.Random(seed)
    randoms = rng.sample(pool, min(n_rand, len(pool)))
    return candidates, randoms, d_mlp


def _clean_pass(model, ref_ids, layers_needed, device):
    """One clean forward PER reference sentence (no padding -> nothing to mask). Returns:
      baseline_entropy : mean over all predicting positions of the per-position softmax entropy
      baseline_loss    : mean over all predicting positions of the teacher-forced next-token CE
      mean_act         : {L: [d_mlp] tensor} mean of blocks.{L}.mlp.hook_post over all (sentence,pos)
    Per-sentence forwards keep positions exactly aligned (no cross-sentence padding entanglement)."""
    post_names = {f"blocks.{L}.mlp.hook_post" for L in layers_needed}
    nf = lambda n: n in post_names                    # noqa: E731
    ent_sum, ent_cnt = 0.0, 0
    loss_sum, loss_cnt = 0.0, 0
    act_sum = {L: None for L in layers_needed}
    pos_total = 0
    for ids in ref_ids:
        with torch.no_grad():
            logits, cache = model.run_with_cache(ids, names_filter=nf)
        lg = logits[0]                                # [seq, d_vocab]
        ent = entropy_of_logits(lg)                   # [seq]
        ent_sum += float(ent.sum()); ent_cnt += ent.shape[0]
        ce = ce_next_token(lg, ids[0])                # [seq-1]
        loss_sum += float(ce.sum()); loss_cnt += ce.shape[0]
        for L in layers_needed:
            post = cache[f"blocks.{L}.mlp.hook_post"][0].float()   # [seq, d_mlp]
            s = post.sum(dim=0)                                    # [d_mlp]
            act_sum[L] = s if act_sum[L] is None else act_sum[L] + s
        pos_total += lg.shape[0]
    mean_act = {L: (act_sum[L] / max(pos_total, 1)) for L in layers_needed}
    baseline_entropy = ent_sum / max(ent_cnt, 1)
    baseline_loss = loss_sum / max(loss_cnt, 1)
    return baseline_entropy, baseline_loss, mean_act


def _ablated_pass(model, ref_ids, L, i, mean_val):
    """Mean-ablate channel i of blocks.{L}.mlp.hook_post (set to mean_val at all positions) over the
    reference set; return (mean entropy, mean loss) over all predicting positions. Per-sentence forwards."""
    name = f"blocks.{L}.mlp.hook_post"
    hook = mean_ablate_hook_factory(i, mean_val)
    ent_sum, ent_cnt = 0.0, 0
    loss_sum, loss_cnt = 0.0, 0
    for ids in ref_ids:
        with torch.no_grad():
            logits = model.run_with_hooks(ids, fwd_hooks=[(name, hook)])
        lg = logits[0]
        ent = entropy_of_logits(lg)
        ent_sum += float(ent.sum()); ent_cnt += ent.shape[0]
        ce = ce_next_token(lg, ids[0])
        loss_sum += float(ce.sum()); loss_cnt += ce.shape[0]
    return ent_sum / max(ent_cnt, 1), loss_sum / max(loss_cnt, 1)


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


def _per_model(name, device, late_frac, k, n_cand, n_rand, ref_mode="short"):
    """Load one model, run the weight screen + clean baseline + per-neuron ablation (BOTH mean- and
    zero-ablation) for candidates and the matched-random set. Returns the per-model result dict."""
    from transformer_lens import HookedTransformer
    print(f"[load] {name} on {device}", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    n_layers = model.cfg.n_layers
    late_layers = _late_layers(n_layers, late_frac)
    print(f"[screen] n_layers={n_layers} late_layers={late_layers[0]}..{late_layers[-1]} "
          f"(>= floor({late_frac}*{n_layers}))", flush=True)
    candidates, randoms, d_mlp = _screen_weights(model, late_layers, k, n_cand, n_rand, SEED)
    print(f"[screen] d_mlp={d_mlp} | top null_frac candidate {candidates[0][:2]} nf={candidates[0][2]:.4f} "
          f"| lowest kept {candidates[-1][:2]} nf={candidates[-1][2]:.4f}", flush=True)

    ref_ids, ref_used = None, "short"
    if ref_mode == "long":
        ref_ids = _long_reference(model, device)
        ref_used = f"long_wikitext_{len(ref_ids)}x{ref_ids[0].shape[1]}" if ref_ids else "short_fallback"
    if ref_ids is None:
        ref_ids = [model.to_tokens(s, prepend_bos=True).to(device) for s in REFERENCE_TEXT]
    print(f"[ref] {ref_used}: {len(ref_ids)} sequences, "
          f"{sum(x.shape[1] for x in ref_ids)} positions", flush=True)
    layers_needed = sorted({L for (L, _i, _f) in candidates} | {L for (L, _i, _f) in randoms})
    base_ent, base_loss, mean_act = _clean_pass(model, ref_ids, layers_needed, device)
    print(f"[baseline] entropy={base_ent:.4f} nats loss={base_loss:.4f} nats", flush=True)

    # BOTH ablation modes per neuron: mean-ablation (preserves the neuron's mean null-space write) and
    # zero-ablation (removes it). decide ENTROPY_NEURON if EITHER mode crosses threshold.
    def both_modes(L, i, m):
        ent_m, loss_m = _ablated_pass(model, ref_ids, L, i, m)       # mean
        ent_z, loss_z = _ablated_pass(model, ref_ids, L, i, 0.0)     # zero
        return (ent_m - base_ent, loss_m - base_loss, ent_z - base_ent, loss_z - base_loss)

    cand_out, en_count = [], 0
    for (L, i, nf) in candidates:
        m = float(mean_act[L][i])
        dent_m, dloss_m, dent_z, dloss_z = both_modes(L, i, m)
        cat_m, cat_z = decide(nf, dent_m, dloss_m), decide(nf, dent_z, dloss_z)
        cat = "ENTROPY_NEURON" if "ENTROPY_NEURON" in (cat_m, cat_z) else "NOT"
        en_count += int(cat == "ENTROPY_NEURON")
        cand_out.append({"L": L, "i": i, "null_frac": round(nf, 4), "mean_act": round(m, 4),
                         "dEntropy_mean": round(dent_m, 4), "dLoss_mean": round(dloss_m, 4),
                         "dEntropy_zero": round(dent_z, 4), "dLoss_zero": round(dloss_z, 4),
                         "category": cat})

    rand_dent_m, rand_dent_z = [], []
    for (L, i, nf) in randoms:
        m = float(mean_act[L][i])
        dent_m, _dl, dent_z, _dlz = both_modes(L, i, m)
        rand_dent_m.append(dent_m); rand_dent_z.append(dent_z)
    rand_stats = {
        "n": len(randoms),
        "dEntropy_mean_ablation_avg": round(sum(rand_dent_m) / len(rand_dent_m), 4) if rand_dent_m else None,
        "dEntropy_zero_ablation_avg": round(sum(rand_dent_z) / len(rand_dent_z), 4) if rand_dent_z else None,
        "dEntropy_zero_ablation_max": round(max(rand_dent_z), 4) if rand_dent_z else None,
        "neurons": [{"L": L, "i": i, "null_frac": round(nf, 4)} for (L, i, nf) in randoms],
    }

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    cand_out.sort(key=lambda r: max(abs(r["dEntropy_zero"]), abs(r["dEntropy_mean"])), reverse=True)
    print(f"[{name}] ENTROPY_NEURONs={en_count}/{len(cand_out)} | random zero-abl dEntropy avg="
          f"{rand_stats['dEntropy_zero_ablation_avg']} max={rand_stats['dEntropy_zero_ablation_max']}", flush=True)
    for r in cand_out[:6]:
        print(f"  L{r['L']}.n{r['i']} null_frac={r['null_frac']} dEnt[mean/zero]="
              f"{r['dEntropy_mean']:+.4f}/{r['dEntropy_zero']:+.4f} -> {r['category']}", flush=True)

    return {"n_layers": n_layers, "late_layers": late_layers, "d_mlp": d_mlp, "ref_used": ref_used,
            "baseline_entropy": round(base_ent, 4), "baseline_loss": round(base_loss, 4),
            "candidates": cand_out, "matched_random": rand_stats,
            "entropy_neuron_count": en_count}


def run(name_base, name_it, tag, device, late_frac, k, n_cand, ref_mode="short"):
    res = {"base": _per_model(name_base, device, late_frac, k, n_cand, N_RAND, ref_mode),
           "it": _per_model(name_it, device, late_frac, k, n_cand, N_RAND, ref_mode)}

    # base-vs-it differential keyed by (L,i): null_frac and dEntropy in each model where the key appears.
    def index_by_key(model_res):
        return {(c["L"], c["i"]): c for c in model_res["candidates"]}
    bk, ik = index_by_key(res["base"]), index_by_key(res["it"])
    diff = []
    for key in sorted(set(bk) | set(ik)):
        b, it = bk.get(key), ik.get(key)
        diff.append({
            "L": key[0], "i": key[1],
            "base_null_frac": b["null_frac"] if b else None,
            "it_null_frac": it["null_frac"] if it else None,
            "base_dEntropy_zero": b["dEntropy_zero"] if b else None,
            "it_dEntropy_zero": it["dEntropy_zero"] if it else None,
            "base_category": b["category"] if b else None,
            "it_category": it["category"] if it else None,
            "in_both": bool(b and it),
        })

    out = {
        "name_base": name_base, "name_it": name_it, "device": device, "tag": tag,
        "cue": "entropy_neuron_gemma2",
        "reference_mode": ref_mode, "reference_used": {m: res[m]["ref_used"] for m in ("base", "it")},
        "params": {"K": k, "N_CAND": n_cand, "N_RAND": N_RAND, "late_frac": late_frac, "SEED": SEED},
        "thresholds": {"NULL_TOL": NULL_TOL, "ENT_TOL": ENT_TOL, "LOSS_TOL": LOSS_TOL},
        "decision_rule": DECISION_RULE,
        "base": res["base"], "it": res["it"],
        "base_vs_it_differential": diff,
    }
    Path("out").mkdir(exist_ok=True)
    Path(f"out/entropy_neuron_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[done] base ENTROPY_NEURONs={res['base']['entropy_neuron_count']} | "
          f"it ENTROPY_NEURONs={res['it']['entropy_neuron_count']} | "
          f"shared candidate keys={sum(1 for d in diff if d['in_both'])}", flush=True)
    print(f"[done] wrote out/entropy_neuron_{tag}.json", flush=True)


# --------------------------------------------------------------------------- selftest (model-free)
def selftest():
    torch.manual_seed(0)

    # ---- null_frac: a synthetic W_U-like matrix with a deliberately tiny-singular-value direction ----
    # W_U [d_model, d_vocab] = Q @ diag(s) @ Vt with one tiny singular value. Vt's rows are orthonormal AND
    # each orthogonal to the all-ones vocab vector, so mean-centering over the vocab axis is an exact no-op
    # (W_U @ ones = Q diag(s) (Vt @ ones) = 0). The LEAST-sensitive direction is then exactly the tiny-s
    # left singular vector Q[:,-1]: a w_out along it -> null_frac ~ 1; along a large-s direction -> ~0.
    d_model, d_vocab = 6, 40
    Q, _ = torch.linalg.qr(torch.randn(d_model, d_model))      # orthonormal left singular vectors
    s = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0, 1e-4])         # last direction is near-null
    R = torch.randn(d_vocab, d_model)
    ones = torch.ones(d_vocab, 1)
    R = R - ones @ (ones.T @ R) / d_vocab                      # project the all-ones vocab direction out of R
    Vt = torch.linalg.qr(R)[0].T                               # [d_model, d_vocab] rows orthonormal, sum ~0
    assert float(Vt.sum(dim=1).abs().max()) < 1e-4, Vt.sum(dim=1)   # each row orthogonal to ones -> sum 0
    W_U = Q @ torch.diag(s) @ Vt                               # [d_model, d_vocab]; row means ~ 0
    assert float(W_U.mean(dim=1).abs().max()) < 1e-4           # mean-centering is a no-op for this W_U
    N = null_basis(W_U, k=1)
    bottom_dir = Q[:, -1]                                      # the tiny-singular-value direction
    top_dir = Q[:, 0]                                          # the largest-singular-value direction
    nf_bottom = null_frac(N, bottom_dir)
    nf_top = null_frac(N, top_dir)
    assert nf_bottom > 0.999, nf_bottom                        # aligned with the null direction -> ~1
    assert nf_top < 0.001, nf_top                              # aligned with a sensitive direction -> ~0
    mix = (bottom_dir + top_dir)                               # 50/50 -> 1/sqrt(2)
    nf_mix = null_frac(N, mix)
    assert abs(nf_mix - 1 / math.sqrt(2)) < 1e-3, nf_mix
    print(f"[selftest] null_frac: bottom={nf_bottom:.4f} (~1) top={nf_top:.4f} (~0) mix={nf_mix:.4f} (~0.707)")

    # null_basis returns the SMALLEST eigenvalue directions: eigh on C=Wc Wc^T returns ascending eigenvalues
    Wc = W_U.float() - W_U.float().mean(dim=1, keepdim=True)
    evals = torch.linalg.eigvalsh(Wc @ Wc.T)
    assert torch.all(evals[:-1] <= evals[1:] + 1e-6), "eigh not ascending"   # ascending order assumption
    print(f"[selftest] eigh ascending; smallest eval={float(evals[0]):.3e} largest={float(evals[-1]):.3e}")

    # ---- entropy on synthetic logits matches hand computation ----
    n = 8
    uni = torch.zeros(1, n)                                    # uniform over n classes -> entropy = ln(n)
    assert abs(float(entropy_of_logits(uni)[0]) - math.log(n)) < 1e-5, entropy_of_logits(uni)
    sharp = torch.tensor([[100.0, 0.0, 0.0, 0.0]])            # near one-hot -> entropy ~ 0
    assert float(entropy_of_logits(sharp)[0]) < 1e-3, entropy_of_logits(sharp)
    assert abs(float(entropy_of_logits(torch.tensor([[1.0, 1.0]]))[0]) - math.log(2)) < 1e-5   # equal -> ln2
    print(f"[selftest] entropy: uniform(8)=ln8={math.log(8):.4f} matched; sharp ~0; equal2=ln2 matched")

    # ---- next-token CE matches hand computation; only seq-1 predicting positions; padding excluded ----
    seq, V = 3, 5
    logits = torch.zeros(seq, V)
    logits[0, 1] = 100.0      # pos 0 predicts token 1 with ~certainty
    logits[1, 2] = 100.0      # pos 1 predicts token 2 with ~certainty
    ids = torch.tensor([4, 1, 2])     # next tokens for pos0,pos1 are ids[1]=1, ids[2]=2 -> both confident
    ce = ce_next_token(logits, ids)
    assert ce.shape[0] == seq - 1, ce.shape         # exactly seq-1 predicting positions (no off-by-one)
    assert float(ce[0]) < 1e-3 and float(ce[1]) < 1e-3, ce   # confident correct -> ~0 loss
    logits2 = torch.zeros(2, V)
    ids2 = torch.tensor([0, 3])
    ce2 = ce_next_token(logits2, ids2)
    assert abs(float(ce2[0]) - math.log(V)) < 1e-5, ce2     # uniform -> ln V per position
    # the per-sentence design has no padding; here we show the means are taken only over real positions:
    # appending a phantom position would add to the count, so excluding it is just not summing it.
    ce_real_mean = float(ce.sum()) / ce.shape[0]
    assert math.isfinite(ce_real_mean) and ce_real_mean < 1e-3
    print(f"[selftest] CE: {seq} tokens -> {seq-1} predicting positions; confident ~0; uniform=lnV={math.log(V):.4f}; mean over real positions only")

    # ---- mean-ablation hook sets exactly channel i to the mean at every position ----
    act = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4).clone()   # [batch,pos,d_mlp]
    before = act.clone()
    hook = mean_ablate_hook_factory(2, 0.5)
    out_act = hook(act)
    assert torch.all(out_act[..., 2] == 0.5), out_act[..., 2]        # channel 2 -> the mean everywhere
    for ch in (0, 1, 3):
        assert torch.all(out_act[..., ch] == before[..., ch]), ch    # other channels untouched
    actb = torch.zeros(1, 2, 4, dtype=torch.bfloat16)                # dtype cast: bf16 stays bf16
    outb = mean_ablate_hook_factory(1, 1.25)(actb)
    assert outb.dtype == torch.bfloat16 and float(outb[0, 0, 1]) != 0.0, outb.dtype
    print("[selftest] mean-ablation hook: only channel i set to mean (all positions); others intact; dtype kept")

    # ---- decide(): ENTROPY_NEURON only when ALL three gates pass ----
    assert decide(0.9, 0.10, 0.005) == "ENTROPY_NEURON"          # all pass
    assert decide(0.3, 0.10, 0.005) == "NOT"                     # low null_frac
    assert decide(0.9, 0.01, 0.005) == "NOT"                     # low dEntropy
    assert decide(0.9, 0.10, 0.10) == "NOT"                      # high |dLoss| (real computation)
    assert decide(0.9, 0.10, -0.10) == "NOT"                     # high |dLoss| negative side
    assert decide(NULL_TOL, ENT_TOL, LOSS_TOL) == "ENTROPY_NEURON"   # exactly at the thresholds -> passes
    assert decide(NULL_TOL, ENT_TOL, -LOSS_TOL) == "ENTROPY_NEURON"  # |dLoss| == LOSS_TOL passes
    print("[selftest] decide(): fires only when null_frac>=tol AND dEntropy>=tol AND |dLoss|<=tol")

    # ratio
    assert abs(ratio_of(0.10, 0.0) - 0.10 / 1e-6) < 1e-3        # tiny |dLoss| -> ratio dominated by the floor
    assert abs(ratio_of(0.10, 0.05) - 2.0) < 1e-6
    print("[selftest] ratio = dEntropy / max(|dLoss|, 1e-6)")

    # ---- late-layer selection: index >= floor(late_frac * n_layers) ----
    assert _late_layers(42, 0.667) == list(range(28, 42)), _late_layers(42, 0.667)   # floor(28.01)=28 (gemma-2-9b)
    assert _late_layers(46, 0.667) == list(range(30, 46)), _late_layers(46, 0.667)   # floor(30.68)=30 (gemma-2-27b)
    assert _late_layers(3, 0.667) == [2], _late_layers(3, 0.667)
    assert _late_layers(1, 0.667) == [0]                         # degenerate-short clamp
    print("[selftest] late-layer cutoff = floor(late_frac * n_layers)")

    # ---- bf16 upcast guard: null_basis upcasts a bf16 W_U before eigendecomposition (no bf16 eigh) ----
    W_U_bf = W_U.to(torch.bfloat16)
    N_bf = null_basis(W_U_bf, k=1)
    assert N_bf.dtype == torch.float32, N_bf.dtype
    assert math.isfinite(null_frac(N_bf, bottom_dir.to(torch.bfloat16)))
    print("[selftest] bf16 guard: W_U.float() before eigendecomposition; N is float32")

    print("[selftest] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--name-base", default="google/gemma-2-9b")
    p.add_argument("--name-it", default="google/gemma-2-9b-it")
    p.add_argument("--tag", default="9b")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--late-frac", type=float, default=0.667,
                   help="late layers = index >= floor(late_frac * n_layers)")
    p.add_argument("--k", type=int, default=K, help="number of smallest-eigenvalue unembedding directions")
    p.add_argument("--n-cand", type=int, default=N_CAND, help="top null_frac late neurons kept as candidates")
    p.add_argument("--ref", default="short", choices=["short", "long"],
                   help="reference distribution: short (sentences) or long (WikiText-2 256-token windows)")
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        run(args.name_base, args.name_it, args.tag, args.device, args.late_frac, args.k, args.n_cand, args.ref)


if __name__ == "__main__":
    main()
