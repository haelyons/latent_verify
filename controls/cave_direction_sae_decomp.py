"""SAE-FEATURE DECOMPOSITION of the fitted residual cave/defer DIRECTION (neutral measurement).

WHY (neutral). headset_direction.py / cave_direction_heldout.py fit a rank-1 diff-of-means "cave/defer"
direction u_cave(L) in the residual stream (u = normalize(mean_items( resid_post[L][-1](counter) -
resid_post[L][-1](neutral) ))) and measure its necessity. This control asks a DIFFERENT, purely descriptive
question about that SAME fitted vector: when u_cave is projected onto a trained sparse-autoencoder
dictionary, is it reconstructed by a SMALL interpretable feature set, or is it spread across MANY features?
It introduces no new mechanism and makes no causal claim -- it only decomposes the geometry of u_cave in the
GemmaScope SAE basis and reports a reconstruction fraction.

WHAT IT MEASURES, per model (base, -it) and per mid-layer L (default [28, 32], chosen to avoid the
late-layer readout-coupling concern):
  1. Fit u_cave at layer L: diff-of-means counter-vs-neutral over the caving items, reusing the EXACT
     construction in headset_direction._dir_pass (u = normalize(mean_i(rc_i - rn_i)), the |gap| >=
     MIN_EFFECT_NET gate, the metric M = logp(C) - logp(W*), the FIT_LAYERS sweep restricted to SAE_LAYERS,
     and the _helpers / _logp_diff machinery). All imported; nothing reimplemented.
  2. Load the GemmaScope canonical residual-stream SAE for gemma-2-9b at layer L via sae_lens:
         SAE.from_pretrained(release="gemma-scope-9b-pt-res-canonical",
                             sae_id=f"layer_{L}/width_16k/canonical", device=device)
     (the canonical JumpReLU residual SAE; release/sae_id recorded in the output). If the load fails, print a
     clear diagnostic and record sae_loaded=false for that layer -- the run does NOT crash.
  3. Project u_cave onto the SAE DECODER dictionary W_dec [n_features, d_model]: per-feature cosine
     c_f = cos(u_cave, W_dec[f]); rank features by |c_f|; report the top-TOPK (=20) indices and cosines, and
     the least-squares reconstruction fraction ||proj_span(top-k)||^2 / ||u_cave||^2 for k in [1,5,10,20,50].
  4. base vs -it: overlap count of the top-20 feature index sets.

NEUTRAL DECISION (module constants FRAC_TOL=0.5, TOPK=20; numbers + categories only, no hypothesis, no
statement about which model/layer should win). Per model per layer:
  SPARSE_RECONSTRUCTION iff recon_frac@TOPK >= FRAC_TOL (the top-TOPK decoder directions capture at least
      FRAC_TOL of u_cave);
  DISTRIBUTED           otherwise.
Plus the base-vs-it top-feature overlap count, reported per layer.

The SAE load + the SAE-feature decomposition are the only new logic; both are inert in --selftest, which is
model-free, needs NO sae_lens, and exercises the cosine ranking and the least-squares reconstruction-fraction
math (monotone nondecreasing in k, <= 1.0) on synthetic dictionaries plus the SPARSE/DISTRIBUTED decision.

The box needs `pip install sae_lens` for the real run (NOT for --selftest).

  python controls/cave_direction_sae_decomp.py --selftest
  python controls/cave_direction_sae_decomp.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

# Repo-internal imports (item pool, turn builders, metric, fit machinery) and sae_lens are deferred into
# run()/load helpers so --selftest runs standalone from controls/ on CPU with NO model load and NO sae_lens
# on the path. On the box the reference files are scp'd flat into latent_verify/ alongside controls/.

# --------------------------------------------------------------------------- pre-registered constants
SAE_LAYERS = [28, 32]            # mid-layers; avoid the late-layer readout-coupling concern
SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"          # GemmaScope canonical residual SAE for gemma-2-9b
SAE_WIDTH = "width_16k"          # canonical 16k JumpReLU width
KS = [1, 5, 10, 20, 50]          # reconstruction-fraction probe depths
TOPK = 20                        # #top features for the decision + the base-vs-it overlap
FRAC_TOL = 0.5                   # recon_frac@TOPK >= this -> SPARSE_RECONSTRUCTION, else DISTRIBUTED


def _sae_id(L):
    return f"layer_{L}/{SAE_WIDTH}/canonical"


# --------------------------------------------------------------------------- pure decomposition math
def feature_cosines(u, W_dec):
    """Per-feature cosine of the unit direction u [d] against each decoder row W_dec[f] [d].
    Returns a 1-D tensor c [n_features], c_f = cos(u, W_dec[f]). Pure (tensors in, tensor out).
    u is assumed (and re-normalized) to unit length; W_dec rows are normalized here."""
    u = u.float()
    u = u / (u.norm() + 1e-8)
    W = W_dec.float()
    Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    return Wn @ u                                            # [n_features]


def rank_by_abs(cos):
    """Feature indices sorted by descending |cosine|. Pure (tensor in, python list out)."""
    order = torch.argsort(cos.abs(), descending=True)
    return [int(i) for i in order.tolist()]


def recon_fraction(u, W_dec, idxs):
    """Fraction of u captured by the least-squares projection of u onto the span of the W_dec rows in idxs.
        proj = B (B^T B)^+ B^T u,   frac = ||proj||^2 / ||u||^2,   B = W_dec[idxs]^T  [d, k].
    Pure (tensors + index list in, float out). Uses lstsq (least norm / pseudo-inverse) so collinear /
    rank-deficient decoder rows are handled without blowing up; clamps the result to [0, 1] against fp
    round-off. An empty idxs returns 0.0 (nothing in the span). frac is the squared-norm ratio of the
    orthogonal projection, hence in [0, 1] and monotone nondecreasing as idxs grows (a larger span cannot
    capture less)."""
    u = u.float()
    denom = float(u @ u)
    if denom <= 0 or not idxs:
        return 0.0
    B = W_dec.float()[idxs].t()                              # [d, k]
    # least-squares coefficients a = argmin ||B a - u||  ->  proj = B a
    a = torch.linalg.lstsq(B, u.unsqueeze(1)).solution      # [k, 1]
    proj = (B @ a).squeeze(1)                                # [d]
    frac = float(proj @ proj) / denom
    if frac < 0.0:
        frac = 0.0
    if frac > 1.0:
        frac = 1.0
    return frac


def recon_curve(u, W_dec, ranked, ks=KS):
    """recon_fraction at each depth k in ks, taking the top-k of the ranked feature list. Returns an ordered
    dict {k: frac}. Pure. k values beyond len(ranked) are capped at the full ranked list."""
    out = {}
    for k in ks:
        kk = min(k, len(ranked))
        out[k] = round(recon_fraction(u, W_dec, ranked[:kk]), 6)
    return out


def overlap_count(idxs_a, idxs_b):
    """Size of the intersection of two top-feature index lists. Pure."""
    return len(set(idxs_a) & set(idxs_b))


# --------------------------------------------------------------------------- pure decision
def decide_sparsity(recon_frac_topk, frac_tol=FRAC_TOL, topk=TOPK):
    """SPARSE_RECONSTRUCTION iff the top-TOPK decoder directions reconstruct >= frac_tol of u_cave; else
    DISTRIBUTED. Pure over the single measured number recon_frac@TOPK (a missing/None value -> UNAVAILABLE,
    which carries no SPARSE/DISTRIBUTED verdict)."""
    if recon_frac_topk is None:
        return {"category": "UNAVAILABLE", "sparse_reconstruction": False,
                "recon_frac_at_topk": None, "topk": topk, "frac_tol": frac_tol,
                "msg": "SAE not loaded / no direction fit -- reconstruction fraction unavailable."}
    sparse = recon_frac_topk >= frac_tol
    cat = "SPARSE_RECONSTRUCTION" if sparse else "DISTRIBUTED"
    msg = (f"top-{topk} decoder directions reconstruct recon_frac@{topk}={recon_frac_topk:.4f} "
           f"{'>=' if sparse else '<'} {frac_tol} -> {cat}.")
    return {"category": cat, "sparse_reconstruction": bool(sparse),
            "recon_frac_at_topk": round(recon_frac_topk, 6), "topk": topk, "frac_tol": frac_tol, "msg": msg}


# --------------------------------------------------------------------------- real run helpers
def _rname(L):
    return f"blocks.{L}.hook_resid_post"


def _fit_u_cave(model, device, is_chat, fit_layers, refs):
    """Diff-of-means cave direction at each fit layer, EXACTLY as headset_direction._dir_pass: for each item
    build counter/neutral, cache resid_post[L][-1] for both, gate on |gap| = |M_neutral - M_counter| >=
    MIN_EFFECT_NET, then u(L) = normalize(mean_items(rc(L) - rn(L))). Returns {L: {"u": tensor[d], "n_ok"}}."""
    (ITEMS, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET) = refs
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    ctxs = []
    for it in ITEMS:
        q, C, W = it["q"], it["correct"], it["Wstar"]
        cid, aid = first(" " + C), first(" " + W)
        counter = push(q, C, PUSH["counter"].format(W=W))
        neutral = push(q, C, NEUTRAL)
        rc, rn = {}, {}
        def grab_c(r, hook):
            rc[hook.layer()] = r[0, -1].detach().float(); return r
        def grab_n(r, hook):
            rn[hook.layer()] = r[0, -1].detach().float(); return r
        with torch.no_grad():
            M_ctr = float(_logp_diff(model.run_with_hooks(counter, fwd_hooks=[(_rname(L), grab_c) for L in fit_layers]), cid, aid))
            M_neu = float(_logp_diff(model.run_with_hooks(neutral, fwd_hooks=[(_rname(L), grab_n) for L in fit_layers]), cid, aid))
        gap = M_neu - M_ctr
        if abs(gap) < MIN_EFFECT_NET:
            continue
        ctxs.append({"rc": rc, "rn": rn, "gap": gap})
        print(f"  [{'it' if is_chat else 'base'}] gap={gap:+.2f} q={q[:40]!r}", flush=True)
    per_layer = {}
    for L in fit_layers:
        if len(ctxs) < 3:
            per_layer[L] = {"u": None, "n_ok": len(ctxs)}
            continue
        D = torch.stack([c["rc"][L] - c["rn"][L] for c in ctxs])          # [n, d]
        cave = D.mean(0)
        u = cave / (cave.norm() + 1e-8)
        per_layer[L] = {"u": u.cpu(), "n_ok": len(ctxs)}
    return per_layer


def _load_sae_wdec(L, device):
    """Load the GemmaScope canonical residual SAE for gemma-2-9b at layer L and return its decoder dictionary
    W_dec [n_features, d_model] on CPU (float). On ANY failure (sae_lens missing, network/cache miss, id
    mismatch) print a diagnostic and return None -- the caller records sae_loaded=false for this layer and
    keeps going. The release/sae_id used is the canonical JumpReLU 16k residual SAE."""
    sae_id = _sae_id(L)
    try:
        from sae_lens import SAE
    except Exception as e:
        print(f"  [sae L{L}] sae_lens import FAILED ({type(e).__name__}: {e}); "
              f"`pip install sae_lens` on the box. recording sae_loaded=false.", flush=True)
        return None, sae_id
    try:
        loaded = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device=device)
        # sae_lens has returned either a bare SAE or an (SAE, cfg, sparsity) tuple across versions; unwrap.
        sae = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        W_dec = sae.W_dec.detach().float().cpu()             # [n_features, d_model]
        print(f"  [sae L{L}] loaded {SAE_RELEASE} / {sae_id}: W_dec {tuple(W_dec.shape)}", flush=True)
        return W_dec, sae_id
    except Exception as e:
        print(f"  [sae L{L}] SAE.from_pretrained FAILED for {SAE_RELEASE}/{sae_id} "
              f"({type(e).__name__}: {e}); recording sae_loaded=false.", flush=True)
        return None, sae_id


def _decompose_layer(u, W_dec):
    """Full per-layer decomposition for a fitted u and a loaded decoder dictionary. Returns the top-TOPK
    feature indices + cosines, the recon-fraction curve over KS, and the SPARSE/DISTRIBUTED decision."""
    cos = feature_cosines(u, W_dec)
    ranked = rank_by_abs(cos)
    top = ranked[:TOPK]
    curve = recon_curve(u, W_dec, ranked, KS)
    dec = decide_sparsity(curve.get(TOPK))
    return {"n_features": int(W_dec.shape[0]), "d_model": int(W_dec.shape[1]),
            "top_features": [{"feature": f, "cosine": round(float(cos[f]), 6)} for f in top],
            "top_feature_indices": top,
            "recon_frac_by_k": {str(k): v for k, v in curve.items()},
            "decision": dec}


def _model_pass(name, is_chat, device, refs):
    """One model: fit u_cave at every SAE_LAYER, load the SAE for that layer, decompose. Returns
    {L: {sae_loaded, sae_id, n_ok, ...decomposition...}} plus the fitted u per layer (for cross overlap)."""
    from transformer_lens import HookedTransformer
    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    fits = _fit_u_cave(model, device, is_chat, SAE_LAYERS, refs)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    layers = {}
    top_idx_by_layer = {}
    for L in SAE_LAYERS:
        u = fits[L]["u"]
        n_ok = fits[L]["n_ok"]
        if u is None:
            layers[L] = {"sae_loaded": False, "sae_id": _sae_id(L), "n_ok": n_ok,
                         "note": f"too few qualifying items ({n_ok}) -- no direction fit",
                         "decision": decide_sparsity(None)}
            top_idx_by_layer[L] = None
            continue
        W_dec, sae_id = _load_sae_wdec(L, device)
        if W_dec is None:
            layers[L] = {"sae_loaded": False, "sae_id": sae_id, "n_ok": n_ok,
                         "note": "SAE load failed (see stdout diagnostic)",
                         "decision": decide_sparsity(None)}
            top_idx_by_layer[L] = None
            continue
        dl = _decompose_layer(u, W_dec)
        dl.update({"sae_loaded": True, "sae_id": sae_id, "n_ok": n_ok,
                   "sae_release": SAE_RELEASE})
        layers[L] = dl
        top_idx_by_layer[L] = dl["top_feature_indices"]
        print(f"  [L{L}] {dl['decision']['category']} recon@{TOPK}={dl['decision']['recon_frac_at_topk']} "
              f"(n_ok={n_ok})", flush=True)
    return {"layers": layers, "top_idx_by_layer": top_idx_by_layer}


def run(name_base, name_it, tag):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # latent_verify/ for the repo imports
    from rlhf_differential import ITEMS, _helpers, _logp_diff, MIN_EFFECT_NET
    from job_truthful_flip import PUSH, NEUTRAL
    refs = (ITEMS, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    it = _model_pass(name_it, True, device, refs)
    base = _model_pass(name_base, False, device, refs)

    # base-vs-it top-feature overlap per layer (only where BOTH models loaded an SAE + fit a direction)
    overlap = {}
    for L in SAE_LAYERS:
        ti, tb = it["top_idx_by_layer"].get(L), base["top_idx_by_layer"].get(L)
        if ti is not None and tb is not None:
            overlap[L] = {"overlap_count": overlap_count(ti, tb), "topk": TOPK,
                          "it_top": ti, "base_top": tb}
        else:
            overlap[L] = {"overlap_count": None, "topk": TOPK,
                          "note": "one or both models lack a loaded SAE / fitted direction at this layer"}

    out = {"model_base": name_base, "model_it": name_it, "cue": "cave_direction_sae_decomp",
           "substrate": "misconception caving items (TruthfulQA-style); rlhf_differential.ITEMS",
           "measures": ("decompose the fitted diff-of-means residual cave/defer direction u_cave(L) into "
                        "GemmaScope SAE decoder features; report per-feature cosines, top-K features, and the "
                        "least-squares reconstruction fraction of u_cave over the top-k decoder directions"),
           "metric_for_fit": "M = logp(C) - logp(W*) first-token margin; |gap| = |M_neutral - M_counter| gate",
           "sae_release": SAE_RELEASE, "sae_width": SAE_WIDTH, "sae_layers": SAE_LAYERS,
           "ks": KS, "thresholds": {"topk": TOPK, "frac_tol": FRAC_TOL},
           "it": it["layers"], "base": base["layers"],
           "base_vs_it_top_feature_overlap": overlap}
    Path("out").mkdir(exist_ok=True)
    fn = f"out/cave_direction_sae_decomp_{tag}.json"
    Path(fn).write_text(json.dumps(out, indent=2, default=str))
    for regime, res in (("it", it), ("base", base)):
        for L in SAE_LAYERS:
            lr = res["layers"][L]
            print(f"[{regime} L{L}] {lr['decision']['category']} "
                  f"recon@{TOPK}={lr['decision']['recon_frac_at_topk']} sae_loaded={lr.get('sae_loaded')}")
    for L in SAE_LAYERS:
        print(f"[overlap L{L}] base-vs-it top-{TOPK} overlap = {overlap[L]['overlap_count']}")
    print(f"[done] wrote {fn}")


# --------------------------------------------------------------------------- selftest (model-free, CPU, NO sae_lens)
def _orthonormal_rows(n, d, seed):
    """n orthonormal rows in R^d (n <= d), as a [n, d] tensor. Pure given the seed (QR of a Gaussian)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, n, generator=g)
    Q, _ = torch.linalg.qr(A)                                # [d, n], orthonormal columns
    return Q.t().contiguous()                               # [n, d], orthonormal rows


def selftest():
    torch.manual_seed(0)
    d, nfeat = 128, 600

    # ---------- feature_cosines: a u aligned exactly with one decoder row reads cos~1 on that row ----------
    g = torch.Generator().manual_seed(7)
    W = torch.randn(nfeat, d, generator=g)
    u_align = W[42].clone()
    u_align = u_align / u_align.norm()
    cos = feature_cosines(u_align, W)
    assert abs(float(cos[42]) - 1.0) < 1e-4, float(cos[42])
    ranked = rank_by_abs(cos)
    assert ranked[0] == 42, ranked[:3]                       # the aligned row ranks first by |cosine|
    # ranking is by |cosine|: a u anti-aligned with a row (cos ~ -1) must still rank that row first
    u_anti = -W[17].clone(); u_anti = u_anti / u_anti.norm()
    assert rank_by_abs(feature_cosines(u_anti, W))[0] == 17
    print(f"[selftest] feature_cosines + |cos| ranking: aligned row cos={float(cos[42]):.4f}, ranks #1; "
          f"anti-aligned row ranks #1 too")

    # ---------- SPARSE: u_cave is EXACTLY a 3-row combination of ORTHONORMAL decoder rows ----------
    # rows 0,1,2 are orthonormal; the rest of the dictionary is unrelated noise. The top-3 (hence top-5)
    # span recovers u fully -> recon_frac@5 ~ 1.0 -> SPARSE_RECONSTRUCTION.
    Wsp = torch.zeros(nfeat, d)
    Wsp[:3] = _orthonormal_rows(3, d, seed=11)               # 3 orthonormal "interpretable" rows
    Wsp[3:] = torch.randn(nfeat - 3, d, generator=torch.Generator().manual_seed(12)) * 0.5
    u_sparse = 0.7 * Wsp[0] + 0.5 * Wsp[1] + 0.6 * Wsp[2]    # exact 3-row combination
    u_sparse = u_sparse / u_sparse.norm()
    ranked_sp = rank_by_abs(feature_cosines(u_sparse, Wsp))
    assert set(ranked_sp[:3]) == {0, 1, 2}, ranked_sp[:6]    # the 3 true rows rank top-3 by |cosine|
    curve_sp = recon_curve(u_sparse, Wsp, ranked_sp, KS)
    fr = [curve_sp[k] for k in KS]
    assert all(b >= a - 1e-9 for a, b in zip(fr, fr[1:])), curve_sp     # monotone nondecreasing in k
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in fr), curve_sp           # bounded to [0,1]
    assert curve_sp[5] > 0.999, curve_sp                               # top-5 (>=3) span recovers u fully
    assert curve_sp[1] < curve_sp[5], curve_sp                         # 1 row alone cannot capture all 3
    d_sp = decide_sparsity(curve_sp[TOPK])
    assert d_sp["category"] == "SPARSE_RECONSTRUCTION" and d_sp["sparse_reconstruction"], d_sp
    print(f"[selftest] SPARSE: 3-row combination -> recon@1={curve_sp[1]:.3f} recon@5={curve_sp[5]:.3f} "
          f"recon@{TOPK}={curve_sp[TOPK]:.3f} monotone+bounded -> {d_sp['category']}")

    # ---------- DISTRIBUTED: u_cave spread with EQUAL weight over a full orthonormal basis of R^d ----------
    # Construction so the math is exact: the dictionary's first d rows are an orthonormal basis of R^d, with
    # u uniform over them, so each basis row has cosine exactly 1/sqrt(d) with u; the padding rows are made
    # ORTHOGONAL to u (cosine 0), so every basis row strictly outranks every pad row. Then the top-k ranked
    # features are k of the basis rows and, because they are orthonormal, recon_frac@k = k/d EXACTLY ->
    # recon_frac@20 = 20/128 ~ 0.156 << FRAC_TOL -> DISTRIBUTED.
    basis = _orthonormal_rows(d, d, seed=21)                 # full orthonormal basis of R^d  [d, d]
    u_dist = basis.sum(0)                                    # equal weight on every basis row
    u_dist = u_dist / u_dist.norm()
    Wdist = torch.zeros(nfeat, d)
    Wdist[:d] = basis
    pad = torch.randn(nfeat - d, d, generator=torch.Generator().manual_seed(22))
    pad = pad - (pad @ u_dist).unsqueeze(1) * u_dist         # remove the u_dist component -> cosine 0 with u
    Wdist[d:] = pad
    cos_dist = feature_cosines(u_dist, Wdist)
    assert float(cos_dist[d:].abs().max()) < 1e-5, float(cos_dist[d:].abs().max())   # pad rows orthogonal
    ranked_dist = rank_by_abs(cos_dist)
    assert all(i < d for i in ranked_dist[:TOPK]), ranked_dist[:TOPK]   # top-TOPK are all basis rows
    curve_dist = recon_curve(u_dist, Wdist, ranked_dist, KS)
    frd = [curve_dist[k] for k in KS]
    assert all(b >= a - 1e-9 for a, b in zip(frd, frd[1:])), curve_dist    # monotone nondecreasing
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in frd), curve_dist           # bounded to [0,1]
    for k in KS:                                                          # orthonormal -> recon_frac@k = k/d
        assert abs(curve_dist[k] - min(k, d) / d) < 1e-4, (k, curve_dist[k], min(k, d) / d)
    assert curve_dist[TOPK] < FRAC_TOL, curve_dist
    d_dist = decide_sparsity(curve_dist[TOPK])
    assert d_dist["category"] == "DISTRIBUTED" and not d_dist["sparse_reconstruction"], d_dist
    print(f"[selftest] DISTRIBUTED: uniform over {d}-dim basis -> recon@{TOPK}={curve_dist[TOPK]:.3f} "
          f"(={TOPK}/{d}={TOPK/d:.3f}) < {FRAC_TOL} -> {d_dist['category']}")

    # ---------- recon_fraction edge cases ----------
    assert recon_fraction(u_sparse, Wsp, []) == 0.0          # empty span captures nothing
    full = recon_fraction(u_dist, Wdist, list(range(d)))     # the basis alone spans R^d -> captures u fully
    assert 0.0 <= full <= 1.0 + 1e-9 and full > 0.999, full  # bounded to <=1, no overflow on a spanning set
    # a single row exactly equal to u -> frac 1.0; a row orthogonal to u -> frac ~0
    one = torch.zeros(2, d); one[0] = u_sparse
    orth = _orthonormal_rows(d, d, 31)[0]
    one[1] = orth - (orth @ u_sparse) * u_sparse; one[1] = one[1] / one[1].norm()
    assert recon_fraction(u_sparse, one, [0]) > 0.999, recon_fraction(u_sparse, one, [0])
    assert recon_fraction(u_sparse, one, [1]) < 1e-3, recon_fraction(u_sparse, one, [1])
    print("[selftest] recon_fraction edge cases: empty=0, spanning-set<=1, exact-row=1, orthogonal-row~0 OK")

    # ---------- decision threshold + overlap + UNAVAILABLE ----------
    assert decide_sparsity(FRAC_TOL)["category"] == "SPARSE_RECONSTRUCTION"          # boundary is inclusive
    assert decide_sparsity(FRAC_TOL - 1e-6)["category"] == "DISTRIBUTED"
    assert decide_sparsity(None)["category"] == "UNAVAILABLE" and not decide_sparsity(None)["sparse_reconstruction"]
    assert overlap_count([1, 2, 3, 4], [3, 4, 5, 6]) == 2
    assert overlap_count([1, 2], [3, 4]) == 0
    assert overlap_count(ranked_sp[:TOPK], ranked_sp[:TOPK]) == TOPK                 # identical lists -> full overlap
    print("[selftest] decision boundary (inclusive @ FRAC_TOL) / UNAVAILABLE / overlap_count OK")

    # ---------- _sae_id formatting matches the documented GemmaScope canonical id ----------
    assert _sae_id(28) == "layer_28/width_16k/canonical", _sae_id(28)
    assert _sae_id(32) == "layer_32/width_16k/canonical", _sae_id(32)
    print(f"[selftest] sae_id format: {SAE_RELEASE} / {_sae_id(28)}")
    print("[selftest] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--name-base", default="google/gemma-2-9b")
    ap.add_argument("--name-it", default="google/gemma-2-9b-it")
    ap.add_argument("--tag", default="9b")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a.name_base, a.name_it, a.tag)
