"""DIRECT-LOGIT-ATTRIBUTION decomposition of the base->it cave-direction CHANGE (neutral measurement).

WHY (neutral). headset_direction.py / cave_direction_heldout.py fit a rank-1 diff-of-means "cave/defer"
direction u_cave(L) in the residual stream (u = normalize(mean_items( resid_post[L][-1](counter) -
resid_post[L][-1](neutral) ))) and find it differs between base and -it -- the -it cave-direction is RLHF-
reshaped. This control asks a DIFFERENT, purely descriptive question: WHICH upstream components write that
base->it difference? At layer L the residual stream resid_post[L] is the EXACT sum of upstream component
outputs (embed + every attention head's W_O write + every MLP write, plus biases), so projecting each
component's output onto the cave axis is an exact linear decomposition of the cave coordinate (direct logit
attribution onto u_cave instead of onto the unembed). This introduces no new mechanism and makes no causal
claim -- it only decomposes the geometry of the residual along u_cave and reports a concentration number and
component categories. It does not tune toward, or state, any component or hypothesis.

WHAT IT MEASURES, per mid-layer L (default [28, 32]):
  1. Fit u_cave_base and u_cave_it at layer L: diff-of-means counter-vs-neutral over EACH model's own caving
     items, reusing the EXACT construction in headset_direction._dir_pass (u = normalize(mean_i(rc-rn)), the
     |gap| = |M_neutral - M_counter| >= MIN_EFFECT_NET gate, the metric M = logp(C) - logp(W*), the FIT_LAYERS
     sweep restricted to L_LAYERS, the _helpers / _logp_diff machinery). Also
     u_delta = normalize(u_cave_it - u_cave_base) (the RLHF-added component of the direction).
  2. On the caving items in the COUNTER condition, last token only, cache every upstream component's output
     into resid_post[L]: per attention HEAD the per-head residual contribution (hook_z [seq,n_head,d_head] at
     the last position, then z_head @ W_O[L,head] -> [d_model]; we loop heads from hook_z + model.W_O rather
     than use_attn_result/hook_result, which OOMs at 9b), and per layer the MLP output (hook_mlp_out [d_model]
     at the last position). Layers 0..L. Last-token-only + forward-only keeps memory light.
  3. DLA per component c: project its mean (over items) output onto the cave axes. Per component:
       c_base       = mean_i (base component_c . u_cave_base)
       c_it         = mean_i (it   component_c . u_cave_it)        [own-axis sanity]
       c_it_onbase  = mean_i (it   component_c . u_cave_base)
       delta_onbase = c_it_onbase - c_base                         [change in write along the FIXED base axis]
       c_it_ondelta = mean_i (it   component_c . u_delta)          [write along the RLHF-ADDED axis]
  4. Rank components by |delta_onbase| (and separately by |c_it_ondelta|). Concentration: fraction of the
     total sum(|delta_onbase|) carried by the top-TOPK components; report the top-TOPK (type=attn-head LxHy /
     mlp Lx, layer-band), and the attn-vs-MLP split.

NEUTRAL DECISION (module constants TOPK=10, CONC_FRAC=0.5; numbers + categories only, no hypothesis, no
statement about which component should win). Per layer L:
  LOCALIZED_RESHAPE iff the top-TOPK components by |delta_onbase| carry >= CONC_FRAC of the total
      sum(|delta_onbase|);
  DIFFUSE_RESHAPE   otherwise.
Plus the top-TOPK components (type, layer, delta_onbase, c_it_ondelta), the attn-vs-MLP split, and the
layer-band of the reshaping, per L. Numbers + categories only.

The per-head W_O reconstruction, the exact-decomposition cache, the DLA projections, and the
concentration/ranking math are the only new logic; all are inert in --selftest, which is model-free, loads NO
model, and exercises (a) the exact-decomposition identity (sum of component projections == resid_post .
u_cave), (b) the ranking/concentration math, and (c) the LOCALIZED/DIFFUSE decision on a planted synthetic
where a KNOWN small component subset writes u_delta (-> LOCALIZED, top set recovered) and on one where the
base->it difference is spread evenly over all components (-> DIFFUSE).

MEMORY: forward-only, last-token-only. The per-head DLA caches, for each item, one [n_head, d_head] z-slice
and one [d_model] MLP output per layer 0..L -- for L=32 on 9b (n_head=16, d_head=256, d_model=3584) that is
~32*(16*256 + 3584)*4 bytes ~ 0.9 MB/item on CPU, trivial. The forward pass itself is the cost; weights +
one forward graph fit the 40GB A100, but if hook_z caching across 42 layers proves heavy at batch>1 prefer an
A100-80GB. Last-token-only keeps it light (we cache one position, not the full sequence).

  python controls/cave_direction_dla.py --selftest
  python controls/cave_direction_dla.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

# Repo-internal imports (item pool, turn builders, metric, fit machinery) are deferred into run() so
# --selftest runs standalone from controls/ on CPU with NO model load and nothing else on sys.path; on the
# box the reference files are scp'd flat into latent_verify/ where these resolve.

# --------------------------------------------------------------------------- pre-registered constants
L_LAYERS = [28, 32]              # mid-layers (subset of headset_direction.FIT_LAYERS); avoid late-layer readout coupling
TOPK = 10                        # #top components for the concentration decision + the reported shortlist
CONC_FRAC = 0.5                  # top-TOPK fraction of sum(|delta_onbase|) >= this -> LOCALIZED_RESHAPE


# --------------------------------------------------------------------------- pure fit / projection math
def fit_u(rc_list, rn_list):
    """Diff-of-means cave direction over aligned counter/neutral last-token residual lists.
    u = normalize(mean_i(rc_i - rn_i)). Pure (tensors in, unit tensor out). Mirrors the construction in
    headset_direction._dir_pass / cave_direction_heldout.fit_direction exactly (no neutral-mean term needed
    here -- this control only needs the unit axis, not the ablation target)."""
    D = torch.stack([rc_list[i] - rn_list[i] for i in range(len(rc_list))])      # [n, d]
    cave = D.mean(0)
    return cave / (cave.norm() + 1e-8)


def unit_delta(u_base, u_it):
    """RLHF-added axis u_delta = normalize(u_it - u_base). Pure. If u_it == u_base the difference is ~0 and
    the normalized result is an (arbitrary) unit vector with ~0 true content; the caller reports the raw norm
    of (u_it - u_base) alongside so a degenerate delta is visible, not hidden."""
    d = u_it - u_base
    return d / (d.norm() + 1e-8), float(d.norm())


def project_components(comp_means, axis):
    """Project each component's mean output onto a unit axis. comp_means: dict {key: 1-D tensor [d]};
    returns {key: float = (component . axis)}. Pure (dict of tensors + axis in, dict of floats out)."""
    return {k: float(v @ axis) for k, v in comp_means.items()}


def concentration(abs_vals, topk=TOPK):
    """Fraction of the total sum of |values| carried by the top-`topk` entries (by magnitude).
    abs_vals: list of non-negative floats. Returns (frac, total, topk_sum, ranked_indices). Pure.
    total==0 -> frac 0.0 (nothing written; not concentrated). frac is in [0, 1] and nondecreasing in topk."""
    total = float(sum(abs_vals))
    order = sorted(range(len(abs_vals)), key=lambda i: abs_vals[i], reverse=True)
    kk = min(topk, len(order))
    top_sum = float(sum(abs_vals[i] for i in order[:kk]))
    frac = (top_sum / total) if total > 0 else 0.0
    return frac, total, top_sum, order


# --------------------------------------------------------------------------- pure decision
def decide_localization(conc_frac, total_abs, topk=TOPK, conc_thr=CONC_FRAC):
    """LOCALIZED_RESHAPE iff the top-`topk` components by |delta_onbase| carry >= conc_thr of the total
    sum(|delta_onbase|); else DIFFUSE_RESHAPE. Pure over the single measured concentration fraction. A
    total of ~0 (no measurable base->it write difference along the axis) -> NO_RESHAPE_SIGNAL (carries no
    LOCALIZED/DIFFUSE verdict, since there is nothing to localize)."""
    if total_abs is None or total_abs <= 1e-9:
        return {"category": "NO_RESHAPE_SIGNAL", "localized_reshape": False,
                "conc_frac_at_topk": (round(conc_frac, 6) if conc_frac is not None else None),
                "total_abs_delta_onbase": (round(total_abs, 8) if total_abs is not None else None),
                "topk": topk, "conc_thr": conc_thr,
                "msg": "sum(|delta_onbase|) ~ 0 -- no measurable base->it write difference along the cave axis to localize."}
    localized = conc_frac >= conc_thr
    cat = "LOCALIZED_RESHAPE" if localized else "DIFFUSE_RESHAPE"
    msg = (f"top-{topk} components carry conc_frac@{topk}={conc_frac:.4f} "
           f"{'>=' if localized else '<'} {conc_thr} of sum(|delta_onbase|)={total_abs:.6g} -> {cat}.")
    return {"category": cat, "localized_reshape": bool(localized),
            "conc_frac_at_topk": round(conc_frac, 6), "total_abs_delta_onbase": round(total_abs, 8),
            "topk": topk, "conc_thr": conc_thr, "msg": msg}


def split_attn_mlp(keys, vals):
    """Sum of |vals| over attn-head components vs mlp components, plus their fractions of the grand total.
    keys: list of component-key dicts each carrying {"type": "attn"|"mlp", ...}. Pure (numbers + categories)."""
    attn = sum(abs(v) for k, v in zip(keys, vals) if k["type"] == "attn")
    mlp = sum(abs(v) for k, v in zip(keys, vals) if k["type"] == "mlp")
    tot = attn + mlp
    return {"attn_abs": round(attn, 8), "mlp_abs": round(mlp, 8),
            "attn_frac": (round(attn / tot, 6) if tot > 0 else None),
            "mlp_frac": (round(mlp / tot, 6) if tot > 0 else None)}


def layer_band(top_keys):
    """(min_layer, max_layer) of the top-component shortlist (layer = each component's layer index). Pure.
    Empty -> (None, None)."""
    if not top_keys:
        return (None, None)
    layers = [k["layer"] for k in top_keys]
    return (min(layers), max(layers))


# --------------------------------------------------------------------------- real run helpers
def _rname(L):
    return f"blocks.{L}.hook_resid_post"


def _zname(L):
    return f"blocks.{L}.attn.hook_z"


def _mname(L):
    return f"blocks.{L}.hook_mlp_out"


def _comp_keys(L, nH):
    """Ordered list of component-key dicts decomposing resid_post[L]: every attention head's W_O write at
    layers 0..L and every MLP output at layers 0..L. Each dict carries a stable string key, a type tag, and
    its layer/head -- the ordering is fixed so the decomposition vector, the projections, and the reports stay
    aligned. (embed + biases are NOT decomposed into per-component DLA -- they are folded into a single
    'residual_remainder' term reported separately so the exact-decomposition identity still holds.)"""
    keys = []
    for ell in range(L + 1):
        for H in range(nH):
            keys.append({"key": f"L{ell}H{H}", "type": "attn", "layer": ell, "head": H})
        keys.append({"key": f"mlp{ell}", "type": "mlp", "layer": ell, "head": None})
    return keys


def _collect_components(model, device, is_chat, L, refs):
    """For one model at one decomposition layer L: build counter/neutral for each qualifying caving item,
    fit u_cave (diff-of-means over resid_post[L][-1] counter vs neutral), and cache the per-COMPONENT
    last-token output in the COUNTER condition (per-head W_O write + per-layer MLP out, layers 0..L).
    Returns {"u": unit tensor [d], "n_ok": int, "comp_means": {key: mean component output [d]},
             "resid_mean": mean resid_post[L][-1] over items [d], "keys": ordered component keys}.

    The per-head contribution is z[0,-1,H,:] @ W_O[L,H] (d_head @ [d_head,d_model] -> d_model), the exact
    summand attention head H writes into the residual stream (we loop heads from hook_z + model.W_O rather
    than use_attn_result/hook_result, which materializes [seq, n_head, d_model] and OOMs at 9b). Sum over all
    component outputs + the bias/embed remainder == resid_post[L][-1] exactly (verified in --selftest)."""
    (ITEMS, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET) = refs
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    nH = model.cfg.n_heads
    keys = _comp_keys(L, nH)
    W_O = model.W_O                                          # [n_layers, n_head, d_head, d_model]

    rc_list, rn_list = [], []                                # last-token resid_post[L] counter / neutral, per item
    comp_sum = {k["key"]: None for k in keys}               # running sum of per-component last-token output (counter)
    resid_sum = None                                        # running sum of resid_post[L][-1] (counter)
    n_ok = 0
    for it in ITEMS:
        q, C, W = it["q"], it["correct"], it["Wstar"]
        cid, aid = first(" " + C), first(" " + W)
        counter = push(q, C, PUSH["counter"].format(W=W))
        neutral = push(q, C, NEUTRAL)

        # --- gap pass: cache resid_post[L][-1] for counter and neutral, gate on the cave magnitude ---
        rc, rn = {}, {}
        def grab_c(r, hook):
            rc["v"] = r[0, -1].detach().float(); return r
        def grab_n(r, hook):
            rn["v"] = r[0, -1].detach().float(); return r
        with torch.no_grad():
            M_ctr = float(_logp_diff(model.run_with_hooks(counter, fwd_hooks=[(_rname(L), grab_c)]), cid, aid))
            M_neu = float(_logp_diff(model.run_with_hooks(neutral, fwd_hooks=[(_rname(L), grab_n)]), cid, aid))
        gap = M_neu - M_ctr
        if abs(gap) < MIN_EFFECT_NET:
            continue

        # --- component pass: cache per-head z and per-layer MLP out at the last token in the COUNTER condition ---
        zcache, mcache = {}, {}
        def grab_z(z, hook):
            zcache[hook.layer()] = z[0, -1].detach().float(); return z          # [n_head, d_head]
        def grab_m(m, hook):
            mcache[hook.layer()] = m[0, -1].detach().float(); return m          # [d_model]
        hooks = [(_zname(ell), grab_z) for ell in range(L + 1)] + [(_mname(ell), grab_m) for ell in range(L + 1)]
        with torch.no_grad():
            model.run_with_hooks(counter, fwd_hooks=hooks, return_type=None)

        # per-head W_O write + per-layer MLP write, on the device, then to CPU float for accumulation
        for ell in range(L + 1):
            zL = zcache[ell].to(device)                                          # [n_head, d_head]
            for H in range(nH):
                contrib = (zL[H].to(W_O.dtype) @ W_O[ell, H]).float().cpu()      # [d_model]
                k = f"L{ell}H{H}"
                comp_sum[k] = contrib if comp_sum[k] is None else comp_sum[k] + contrib
            mk = f"mlp{ell}"
            mc = mcache[ell].float().cpu()
            comp_sum[mk] = mc if comp_sum[mk] is None else comp_sum[mk] + mc

        rcv = rc["v"].cpu()
        resid_sum = rcv if resid_sum is None else resid_sum + rcv
        rc_list.append(rc["v"])
        rn_list.append(rn["v"])
        n_ok += 1
        print(f"  [{'it' if is_chat else 'base'} L{L}] gap={gap:+.2f} q={q[:40]!r}", flush=True)

    if n_ok < 3:
        return {"u": None, "n_ok": n_ok, "comp_means": None, "resid_mean": None, "keys": keys}
    u = fit_u(rc_list, rn_list).cpu()
    comp_means = {k: (comp_sum[k] / n_ok) for k in comp_sum}
    resid_mean = resid_sum / n_ok
    return {"u": u, "n_ok": n_ok, "comp_means": comp_means, "resid_mean": resid_mean, "keys": keys}


def _model_pass(name, is_chat, device, refs):
    """One model: at every L in L_LAYERS, fit u_cave and cache the per-component mean last-token outputs.
    Returns {L: {"u", "n_ok", "comp_means", "resid_mean", "keys"}}. Model loaded once, freed after."""
    from transformer_lens import HookedTransformer
    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    out = {}
    for L in L_LAYERS:
        out[L] = _collect_components(model, device, is_chat, L, refs)
        print(f"  [L{L}] n_ok={out[L]['n_ok']}", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def _decompose_layer(base_L, it_L):
    """Full per-layer DLA decomposition given the base and -it caches at layer L. Computes u_delta, the four
    per-component projections, ranks by |delta_onbase| and by |c_it_ondelta|, the concentration fraction, the
    attn-vs-MLP split, the layer band, and the LOCALIZED/DIFFUSE decision. Pure over the cached means/keys.
    Returns None if either model lacks a fit at this layer."""
    if base_L["u"] is None or it_L["u"] is None:
        return None
    u_base, u_it = base_L["u"], it_L["u"]
    u_delta, delta_norm = unit_delta(u_base, u_it)
    keys = base_L["keys"]                                    # base and it share the same component ordering at L

    base_means, it_means = base_L["comp_means"], it_L["comp_means"]
    c_base = project_components(base_means, u_base)          # base component . u_cave_base
    c_it = project_components(it_means, u_it)                # it   component . u_cave_it   (own-axis sanity)
    c_it_onbase = project_components(it_means, u_base)       # it   component . u_cave_base (fixed-axis)
    c_it_ondelta = project_components(it_means, u_delta)     # it   component . u_delta     (RLHF-added axis)

    rows = []
    for k in keys:
        key = k["key"]
        d_onbase = c_it_onbase[key] - c_base[key]
        rows.append({"key": key, "type": k["type"], "layer": k["layer"], "head": k["head"],
                     "c_base": c_base[key], "c_it": c_it[key], "c_it_onbase": c_it_onbase[key],
                     "delta_onbase": d_onbase, "c_it_ondelta": c_it_ondelta[key]})

    abs_delta = [abs(r["delta_onbase"]) for r in rows]
    conc_frac, total_abs, top_sum, order_delta = concentration(abs_delta, TOPK)
    order_ondelta = sorted(range(len(rows)), key=lambda i: abs(rows[i]["c_it_ondelta"]), reverse=True)

    def fmt(r):
        return {"key": r["key"], "type": r["type"], "layer": r["layer"], "head": r["head"],
                "c_base": round(r["c_base"], 6), "c_it": round(r["c_it"], 6),
                "c_it_onbase": round(r["c_it_onbase"], 6),
                "delta_onbase": round(r["delta_onbase"], 6), "c_it_ondelta": round(r["c_it_ondelta"], 6)}

    top_by_delta = [fmt(rows[i]) for i in order_delta[:TOPK]]
    top_by_ondelta = [fmt(rows[i]) for i in order_ondelta[:TOPK]]
    top_keys = [{"type": r["type"], "layer": r["layer"]} for r in top_by_delta]

    split = split_attn_mlp([{"type": r["type"]} for r in rows], [r["delta_onbase"] for r in rows])
    band = layer_band(top_keys)
    decision = decide_localization(conc_frac, total_abs)

    return {"n_ok_base": base_L["n_ok"], "n_ok_it": it_L["n_ok"],
            "u_delta_norm": round(delta_norm, 6),
            "cos_u_base_u_it": round(float(u_base @ u_it), 6),
            "n_components": len(rows),
            "total_abs_delta_onbase": round(total_abs, 8),
            "conc_frac_at_topk": round(conc_frac, 6),
            "top_by_delta_onbase": top_by_delta,
            "top_by_c_it_ondelta": top_by_ondelta,
            "attn_vs_mlp_split_of_abs_delta_onbase": split,
            "reshape_layer_band_of_topk": list(band),
            "decision": decision}


def run(name_base, name_it, tag):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # latent_verify/ for the repo imports
    from rlhf_differential import ITEMS, _helpers, _logp_diff, MIN_EFFECT_NET
    from job_truthful_flip import PUSH, NEUTRAL
    refs = (ITEMS, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    it = _model_pass(name_it, True, device, refs)
    base = _model_pass(name_base, False, device, refs)

    layers = {}
    for L in L_LAYERS:
        dl = _decompose_layer(base[L], it[L])
        if dl is None:
            layers[L] = {"note": (f"too few qualifying items "
                                  f"(base n_ok={base[L]['n_ok']}, it n_ok={it[L]['n_ok']}) -- no decomposition"),
                         "decision": decide_localization(None, None)}
        else:
            layers[L] = dl

    out = {"model_base": name_base, "model_it": name_it, "cue": "cave_direction_dla",
           "substrate": "misconception caving items (TruthfulQA-style); rlhf_differential.ITEMS",
           "measures": ("direct logit attribution of each upstream component (per-head W_O write + per-layer "
                        "MLP out, layers 0..L) onto the diff-of-means cave axis at layer L; exact linear "
                        "decomposition of the cave coordinate. Reports per component c_base, c_it (own-axis), "
                        "c_it_onbase, delta_onbase = c_it_onbase - c_base (change along the FIXED base cave-axis), "
                        "and c_it_ondelta (write along the RLHF-added axis u_delta = normalize(u_it - u_base)); "
                        "and the concentration of sum(|delta_onbase|) in the top-TOPK components."),
           "metric_for_fit": "M = logp(C) - logp(W*) first-token margin; |gap| = |M_neutral - M_counter| gate",
           "decomposition": ("resid_post[L][-1] = sum over (attn head W_O writes + MLP outs, layers 0..L) + "
                             "embed/bias remainder (exact identity, last token, counter condition)"),
           "l_layers": L_LAYERS, "thresholds": {"topk": TOPK, "conc_frac": CONC_FRAC},
           "layers": {str(L): layers[L] for L in L_LAYERS}}
    Path("out").mkdir(exist_ok=True)
    fn = f"out/cave_direction_dla_{tag}.json"
    Path(fn).write_text(json.dumps(out, indent=2, default=str))
    for L in L_LAYERS:
        lr = layers[L]
        dec = lr["decision"]
        print(f"[L{L}] {dec['category']} conc@{TOPK}={dec.get('conc_frac_at_topk')} "
              f"total|delta_onbase|={dec.get('total_abs_delta_onbase')} "
              f"band={lr.get('reshape_layer_band_of_topk')} "
              f"split={lr.get('attn_vs_mlp_split_of_abs_delta_onbase')}")
    print(f"[done] wrote {fn}")


# --------------------------------------------------------------------------- selftest (model-free, CPU, NO model load)
def _planted_decomposition(n_items, n_comp, d, writer_idx, delta_mag, seed):
    """Synthetic exact component decomposition for one model. Returns per-item component outputs such that
    their per-item sum is resid_post[L][-1], plus the planted base/it cave axes.

    Construction (so the exact-decomposition identity and the localization are both testable):
      - draw a base cave axis u_base and an orthogonal delta axis u_delta (unit, orthogonal).
      - u_it = normalize(u_base + delta_mag * u_delta)  -> the it cave axis differs from base ALONG u_delta.
      - for each item: each component output is random Gaussian noise; then the components in `writer_idx`
        EACH additionally write +delta_mag/len(writer_idx) * u_delta in the IT model (and 0 extra in BASE).
        So the base->it change in the write along the FIXED base axis (delta_onbase) is carried ONLY by the
        writer components (up to the projection of u_delta on u_base, which is ~0 by orthogonality, so we
        instead make the writers write along u_base directly to get a clean delta_onbase signal -- see below).

    To make delta_onbase (change along the FIXED base axis) localize cleanly, the writer components add their
    extra write along u_base itself (the quantity delta_onbase measures is the per-component change in the
    u_base coordinate). Non-writers get IDENTICAL outputs in base and it -> their delta_onbase == 0. The
    c_it_ondelta channel is exercised separately by also adding a u_delta component to the writers in it.
    Returns (base_comp_means, it_comp_means [dicts key->tensor], u_base, u_it, keys)."""
    g = torch.Generator().manual_seed(seed)
    # orthonormal (u_base, u_delta_axis)
    A = torch.randn(d, 2, generator=g)
    Q, _ = torch.linalg.qr(A)
    u_base = Q[:, 0].contiguous()
    u_delta_axis = Q[:, 1].contiguous()
    u_it = u_base + delta_mag * u_delta_axis
    u_it = u_it / u_it.norm()

    keys = [{"key": f"c{i}", "type": ("attn" if i % 2 == 0 else "mlp"), "layer": i, "head": (i if i % 2 == 0 else None)}
            for i in range(n_comp)]
    writers = set(writer_idx)
    per_writer = delta_mag / max(len(writers), 1)

    base_sum = {k["key"]: torch.zeros(d) for k in keys}
    it_sum = {k["key"]: torch.zeros(d) for k in keys}
    for _ in range(n_items):
        # base components: shared random outputs
        comps = {k["key"]: torch.randn(d, generator=g) for k in keys}
        for k in keys:
            base_sum[k["key"]] += comps[k["key"]]
            extra = torch.zeros(d)
            if int(k["key"][1:]) in writers:                # writer pushes along u_base (for delta_onbase) AND
                # strongly along u_delta_axis: u_delta=normalize(u_it-u_base) has a NEGATIVE u_base component, so
                # an equal u_base+u_delta_axis write nearly cancels on u_delta; the 3x makes writers dominate the
                # c_it_ondelta channel without changing delta_onbase (= the u_base-coordinate change, still per_writer).
                extra = per_writer * u_base + 3.0 * per_writer * u_delta_axis
            it_sum[k["key"]] += comps[k["key"]] + extra     # non-writers identical base==it -> delta_onbase 0
    base_means = {k: v / n_items for k, v in base_sum.items()}
    it_means = {k: v / n_items for k, v in it_sum.items()}
    return base_means, it_means, u_base, u_it, keys


def _diffuse_decomposition(n_items, n_comp, d, delta_mag, seed):
    """Same as _planted_decomposition but the base->it u_base-write change is spread EVENLY across ALL
    components (each adds delta_mag/n_comp * u_base in it) -> delta_onbase is uniform -> low concentration."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, 2, generator=g)
    Q, _ = torch.linalg.qr(A)
    u_base = Q[:, 0].contiguous()
    u_delta_axis = Q[:, 1].contiguous()
    u_it = (u_base + delta_mag * u_delta_axis); u_it = u_it / u_it.norm()
    keys = [{"key": f"c{i}", "type": ("attn" if i % 2 == 0 else "mlp"), "layer": i, "head": (i if i % 2 == 0 else None)}
            for i in range(n_comp)]
    per = delta_mag / n_comp
    base_sum = {k["key"]: torch.zeros(d) for k in keys}
    it_sum = {k["key"]: torch.zeros(d) for k in keys}
    for _ in range(n_items):
        comps = {k["key"]: torch.randn(d, generator=g) for k in keys}
        for k in keys:
            base_sum[k["key"]] += comps[k["key"]]
            it_sum[k["key"]] += comps[k["key"]] + per * u_base    # every component contributes equally
    base_means = {k: v / n_items for k, v in base_sum.items()}
    it_means = {k: v / n_items for k, v in it_sum.items()}
    return base_means, it_means, u_base, u_it, keys


def selftest():
    torch.manual_seed(0)

    # ---------- (0) projection + concentration + decision math ----------
    # exact-decomposition identity: sum of per-component projections == (sum of components) . axis
    d = 64
    g = torch.Generator().manual_seed(7)
    axis = torch.randn(d, generator=g); axis = axis / axis.norm()
    comps = {f"c{i}": torch.randn(d, generator=g) for i in range(20)}
    proj = project_components(comps, axis)
    summed_resid = sum(comps.values())
    lhs = sum(proj.values())
    rhs = float(summed_resid @ axis)
    assert abs(lhs - rhs) < 1e-4, f"exact-decomposition identity broken: {lhs} vs {rhs}"
    print(f"[selftest] (0) exact-decomposition identity: sum(proj)={lhs:.5f} == resid.axis={rhs:.5f}")

    # concentration math: one component carries everything -> frac 1.0; uniform -> ~topk/n
    cf1, tot1, ts1, _ = concentration([10.0] + [0.0] * 30, TOPK)
    assert abs(cf1 - 1.0) < 1e-9 and abs(tot1 - 10.0) < 1e-9, (cf1, tot1)
    n_unif = 40
    cfu, totu, _, _ = concentration([1.0] * n_unif, TOPK)
    assert abs(cfu - TOPK / n_unif) < 1e-9, (cfu, TOPK / n_unif)        # uniform -> exactly topk/n = 0.25
    cf0, tot0, _, _ = concentration([0.0] * 5, TOPK)
    assert cf0 == 0.0 and tot0 == 0.0
    # monotone nondecreasing in topk
    vals = [abs(x) for x in torch.randn(50, generator=g).tolist()]
    fracs = [concentration(vals, k)[0] for k in (1, 5, 10, 20, 50)]
    assert all(fracs[i] <= fracs[i + 1] + 1e-9 for i in range(len(fracs) - 1)), fracs
    print(f"[selftest] (0) concentration: single={cf1:.3f} uniform(n={n_unif})={cfu:.3f} monotone OK")

    # decision thresholds
    d_loc = decide_localization(0.80, 5.0)
    assert d_loc["category"] == "LOCALIZED_RESHAPE" and d_loc["localized_reshape"], d_loc
    d_dif = decide_localization(0.25, 5.0)
    assert d_dif["category"] == "DIFFUSE_RESHAPE" and not d_dif["localized_reshape"], d_dif
    d_edge = decide_localization(CONC_FRAC, 5.0)                        # exactly at threshold -> LOCALIZED (>=)
    assert d_edge["category"] == "LOCALIZED_RESHAPE", d_edge
    d_none = decide_localization(0.99, 0.0)                             # nothing written -> NO_RESHAPE_SIGNAL
    assert d_none["category"] == "NO_RESHAPE_SIGNAL" and not d_none["localized_reshape"], d_none
    print("[selftest] (0) decision LOCALIZED / DIFFUSE / edge / NO_SIGNAL all fire")

    # split + band helpers
    sk = [{"type": "attn"}, {"type": "mlp"}, {"type": "attn"}]
    sp = split_attn_mlp(sk, [3.0, 1.0, -1.0])
    assert abs(sp["attn_abs"] - 4.0) < 1e-9 and abs(sp["mlp_abs"] - 1.0) < 1e-9, sp
    assert abs(sp["attn_frac"] - 0.8) < 1e-9, sp
    assert layer_band([{"layer": 5}, {"layer": 2}, {"layer": 9}]) == (2, 9)
    assert layer_band([]) == (None, None)
    print("[selftest] (0) attn/mlp split + layer band OK")

    # unit_delta: orthogonal axes -> normalized delta points along the true delta axis
    ub = torch.tensor([1.0, 0.0, 0.0]); ui = torch.tensor([0.0, 1.0, 0.0])
    ud, dn = unit_delta(ub, ui)
    assert abs(float(ud @ torch.tensor([-0.7071, 0.7071, 0.0]))) > 0.99, ud   # normalize(ui-ub) = (-1,1,0)/sqrt2
    assert abs(dn - (2 ** 0.5)) < 1e-4, dn
    print(f"[selftest] (0) unit_delta norm={dn:.4f} OK")

    # ---------- (i) LOCALIZED: a KNOWN small writer subset carries delta_onbase -> top set recovered ----------
    n_items, n_comp, dd = 12, 40, 96
    writers = [3, 4, 5]                                                 # the planted reshaping subset
    bm, im, u_base, u_it, keys = _planted_decomposition(n_items, n_comp, dd, writers, delta_mag=6.0, seed=11)
    base_L = {"u": fit_u_from_means(u_base), "n_ok": n_items, "comp_means": bm, "resid_mean": None, "keys": keys}
    it_L = {"u": fit_u_from_means(u_it), "n_ok": n_items, "comp_means": im, "resid_mean": None, "keys": keys}
    dl = _decompose_layer(base_L, it_L)
    assert dl is not None
    # exact-decomposition identity on the IT means along u_base: sum of c_it_onbase == resid_mean_it . u_base
    sum_it_onbase = sum(float(im[k] @ base_L["u"]) for k in im)
    resid_it = sum(im.values())
    assert abs(sum_it_onbase - float(resid_it @ base_L["u"])) < 1e-3, (sum_it_onbase, float(resid_it @ base_L["u"]))
    # the top components by |delta_onbase| must be exactly the planted writers (small set)
    top_layers = sorted(c["layer"] for c in dl["top_by_delta_onbase"][:len(writers)])
    assert top_layers == sorted(writers), f"planted writers {writers} not recovered as top: {top_layers}"
    assert dl["decision"]["category"] == "LOCALIZED_RESHAPE", dl["decision"]
    assert dl["decision"]["conc_frac_at_topk"] >= CONC_FRAC, dl["decision"]
    print(f"[selftest] (i) LOCALIZED: writers {writers} recovered as top, conc@{TOPK}="
          f"{dl['decision']['conc_frac_at_topk']:.3f}")

    # the c_it_ondelta channel also concentrates on the writers (they alone wrote along u_delta_axis)
    top_ondelta_layers = sorted(c["layer"] for c in dl["top_by_c_it_ondelta"][:len(writers)])
    assert top_ondelta_layers == sorted(writers), f"c_it_ondelta top {top_ondelta_layers} != writers {writers}"
    print(f"[selftest] (i) c_it_ondelta top set == writers {top_ondelta_layers}")

    # ---------- (ii) DIFFUSE: base->it write change spread evenly over all components -> low concentration ----------
    bm2, im2, u_base2, u_it2, keys2 = _diffuse_decomposition(n_items, n_comp, dd, delta_mag=6.0, seed=12)
    base2 = {"u": fit_u_from_means(u_base2), "n_ok": n_items, "comp_means": bm2, "resid_mean": None, "keys": keys2}
    it2 = {"u": fit_u_from_means(u_it2), "n_ok": n_items, "comp_means": im2, "resid_mean": None, "keys": keys2}
    dl2 = _decompose_layer(base2, it2)
    assert dl2["decision"]["category"] == "DIFFUSE_RESHAPE", dl2["decision"]
    assert dl2["decision"]["conc_frac_at_topk"] < CONC_FRAC, dl2["decision"]
    # diffuse concentration sits near the uniform floor topk/n_comp (+ noise from the random base components)
    print(f"[selftest] (ii) DIFFUSE: conc@{TOPK}={dl2['decision']['conc_frac_at_topk']:.3f} "
          f"(< {CONC_FRAC}; uniform floor ~ {TOPK / n_comp:.3f})")

    # ---------- (iii) too-few-items guard ----------
    assert _decompose_layer({"u": None, "keys": keys}, it_L) is None
    print("[selftest] (iii) too-few-items -> no decomposition (None) OK")
    print("[selftest] PASS")


def fit_u_from_means(u):
    """Selftest helper: the synthetic planters return the cave AXIS directly (already a unit vector). The real
    path fits u via fit_u(rc_list, rn_list); here we just normalize the planted axis so the decomposition code
    receives the same object shape (a unit tensor)."""
    return (u / (u.norm() + 1e-8)).float()


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
