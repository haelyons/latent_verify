"""ROBUSTNESS HARDENING of the base->it cave-direction DLA (neutral measurement).

WHY (neutral). cave_direction_dla.py decomposes the base->it CHANGE in the cave-direction's residual
coordinate into per-component direct logit attribution (per attn-head z@W_O + per-layer hook_mlp_out, last
token, projected onto the diff-of-means cave axis) and reports a top-10 concentration at two readout layers
(L in {28, 32}). That readout has three un-controlled features that this control de-confounds, WITHOUT adding
any mechanism or claim:

  (1) POSITIONAL READOUT ARTIFACT. The DLA reads resid_post[L_READ][-1], which is the exact sum of all
      upstream component writes up to L_READ. A contiguous "band" of top components at one L_READ may simply
      be the band of components nearest L_READ -- i.e. the band SLIDES with L_READ rather than naming a stable
      writer set. We add a fixed late readout AND a readout SWEEP over several L_READ and ask whether the SAME
      component set dominates ACROSS readouts (position-stable) or whether the set turns over with L_READ
      (positional artifact).

  (2) UN-NORMALIZED MAGNITUDE TREND. delta_onbase = (it_comp . u_base) - (base_comp . u_base) tends to be
      larger for components whose base write |c_base| is larger (a bigger writer has a bigger absolute change).
      Ranking by |delta_onbase| then partly ranks by base-write size, not by genuine base->it reshaping. We
      regress |delta_onbase| on |c_base| across ALL components (simple linear fit) and rank by the RESIDUAL
      r_c = |delta_onbase| - predicted, so a component ranks high only if its change EXCEEDS what its base-write
      magnitude predicts. (We use the regression residual, NOT the ratio |delta|/|c_base|, which blows up for
      tiny c_base.)

  (3) NO CONFIDENCE INTERVAL. The item set is small (the qualifying caving items). We bootstrap the items with
      replacement (refitting u_base/u_it and recomputing per resample) to put percentile CIs on the
      residualized concentration and the attn/mlp split.

This control introduces no new mechanism. It reuses cave_direction_dla's exact diff-of-means cave-direction
fit, its per-component DLA cache (per-head W_O write + per-layer MLP out, last token, counter condition, with
the |gap| = |M_neutral - M_counter| >= MIN_EFFECT_NET gate and the metric M = logp(C) - logp(W*)), its
projection / concentration / split / band helpers, and the counter/neutral construction (job_truthful_flip
PUSH/NEUTRAL via rlhf_differential), and cave_direction_heldout's percentile-bootstrap mechanics. The only new
logic is the magnitude-controlled residual ranking, the cross-readout intersection, and the item bootstrap --
all pure and exercised by the model-free --selftest.

WHAT IT MEASURES, per readout layer L_READ in L_READS (default [28, 32, 36, 40]):
  Fit u_base / u_it at L_READ (each model's own diff-of-means cave axis at that L_READ; same construction as
  cave_direction_dla.fit_u), and cache per-component last-token outputs in the COUNTER condition (per-head W_O
  write + per-layer MLP out, layers 0..L_READ). Then per component c:
    c_base       = mean_i (base component_c . u_base)
    delta_onbase = mean_i (it component_c . u_base) - c_base       (change along the FIXED base cave-axis)
  1. RAW: conc@10 of sum(|delta_onbase|), attn/mlp aggregate split of |delta_onbase|, top-10 component keys.
  2. RESIDUALIZED: r_c = |delta_onbase| - (a + b|c_base|) from the OLS fit of |delta_onbase| on |c_base| over
     ALL components; normalized conc@10 = fraction of sum(|r_c|) in the top-10 by |r_c|, the attn/mlp split of
     |r_c|, and the top-10 keys by |r_c|. (Because OLS centers residuals, a sparse localized signal puts the
     top-K residual mass near 0.6 and a fully diffuse signal near the uniform floor TOPK/n_components; the
     CONC_FRAC=0.5 threshold sits cleanly between the two regimes.)
  3. CROSS-READOUT STABILITY: the intersection of the top-10-by-|r_c| component KEY SETS across every L_READ,
     and the per-L_READ "only here" keys. A large intersection = position-stable; full turnover = positional.
  4. BOOTSTRAP CI (N_BOOT item resamples, fixed seed; refit u_base/u_it and recompute per resample): percentile
     CIs for the RAW conc@10, the RESIDUALIZED conc@10, and the attn_frac on |r_c|, per L_READ.

NEUTRAL DECISION (module constants CONC_FRAC=0.5, STABLE_MIN=3; numbers + categories only, no hypothesis, no
statement about which components should win):
  granularity-invariant split:
    MLP_DOMINATED iff the aggregate mlp_frac (of |delta_onbase|) >= 0.5 at a MAJORITY of L_READ; else
      ATTN_DOMINATED / MIXED.
  position / localization:
    POSITION_STABLE_LOCALIZATION iff the cross-readout top-10-by-|r_c| intersection has >= STABLE_MIN
      components AND the RESIDUALIZED conc@10 bootstrap-CI LOWER BOUND >= CONC_FRAC at a MAJORITY of L_READ;
    else POSITION_DEPENDENT (intersection < STABLE_MIN: the dominant set turns over with the readout layer),
    or DIFFUSE (intersection >= STABLE_MIN but the residualized conc@10 CI lower bound stays < CONC_FRAC at a
      majority of L_READ: no small stable set concentrates the magnitude-controlled change).
Reported per L_READ and as a cross-readout summary. Numbers + categories only.

MEMORY: forward-only, last-token-only, same caches as cave_direction_dla (reused verbatim). The deepest readout
L_READ=40 caches one [n_head,d_head] z-slice and one [d_model] MLP out per layer 0..40 per item (~1 MB/item on
CPU at 9b), trivial; the forward pass + weights are the cost. Bootstrap is pure arithmetic over the cached
per-item component projections (no extra forwards).

  python controls/cave_direction_dla_robust.py --selftest
  python controls/cave_direction_dla_robust.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it
"""
import argparse
import json
from pathlib import Path

import torch

# Repo-internal imports (item pool, turn builders, metric, fit machinery, AND the reused cave_direction_dla
# primitives) are deferred into run()/selftest() so --selftest runs standalone from controls/ on CPU with NO
# model load; on the box the reference files are scp'd flat into latent_verify/ where these resolve. The pure
# helpers we import for --selftest (concentration, split_attn_mlp, fit_u, ...) are imported at selftest time.

# --------------------------------------------------------------------------- pre-registered constants
L_READS = [28, 32, 36, 40]       # fixed-late readout + readout sweep (do top components SLIDE with L_READ?)
TOPK = 10                        # #top components for the concentration / intersection
CONC_FRAC = 0.5                  # residualized conc@TOPK CI lower bound >= this -> concentrated
STABLE_MIN = 3                   # #components in the cross-readout top-TOPK-by-residual intersection -> stable
N_BOOT = 200                     # item bootstrap resamples (fixed seed)
SEED = 0
CI_LO, CI_HI = 2.5, 97.5         # 95 percent percentile interval


# --------------------------------------------------------------------------- pure: magnitude-controlled residual
def ols_residual(y, x):
    """Simple-linear-regression residuals r = y - (a + b x) from the OLS fit of y on x. Pure (lists in, list
    out). Returns (residuals, a, b). Degenerate x (zero variance) -> b=0, a=mean(y) -> r = y - mean(y) (just
    de-meaned, the best constant fit). By construction sum(r) ~ 0 and r is orthogonal to x (the two normal
    equations of OLS), which is what makes r_c a magnitude-CONTROLLED quantity: it carries only the part of y
    NOT explained by x."""
    n = len(y)
    if n == 0:
        return [], 0.0, 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    b = (sxy / sxx) if sxx > 1e-18 else 0.0
    a = my - b * mx
    r = [y[i] - (a + b * x[i]) for i in range(n)]
    return r, a, b


def residual_rank(keys, abs_delta, abs_cbase, topk=TOPK):
    """Magnitude-controlled ranking. Regress |delta_onbase| on |c_base| over ALL components; rank by |residual|.
    Returns a dict with the normalized conc@topk (fraction of sum(|r|) in the top-topk by |r|), the OLS (a,b),
    the residual orthogonality/zero-sum diagnostics, and the ordered top-topk component KEYS. Pure.

    keys: list of component-key dicts (each carrying 'key' + 'type'); abs_delta/abs_cbase: aligned lists."""
    from cave_direction_dla import concentration, split_attn_mlp
    r, a, b = ols_residual(abs_delta, abs_cbase)
    abs_r = [abs(v) for v in r]
    conc_frac, total_abs_r, top_sum, order = concentration(abs_r, topk)   # reuse the SAME concentration math
    kk = min(topk, len(order))
    top_idx = order[:kk]
    top_keys = [keys[i]["key"] for i in top_idx]
    split = split_attn_mlp([{"type": keys[i]["type"]} for i in range(len(keys))], r)   # attn/mlp split on |r|
    sum_r = float(sum(r))
    dot_rx = float(sum(r[i] * abs_cbase[i] for i in range(len(r))))       # ~0 by OLS orthogonality
    return {"conc_frac_at_topk_residual": round(conc_frac, 6),
            "total_abs_residual": round(total_abs_r, 8),
            "ols_a": round(a, 8), "ols_b": round(b, 8),
            "residual_sum": round(sum_r, 8), "residual_dot_abs_cbase": round(dot_rx, 8),
            "attn_vs_mlp_split_of_abs_residual": split,
            "top_keys_by_residual": top_keys,
            "top_full_by_residual": [{"key": keys[i]["key"], "type": keys[i]["type"], "layer": keys[i]["layer"],
                                      "head": keys[i]["head"],
                                      "abs_delta_onbase": round(abs_delta[i], 6),
                                      "abs_c_base": round(abs_cbase[i], 6),
                                      "residual": round(r[i], 6)} for i in top_idx]}


# --------------------------------------------------------------------------- pure: cross-readout intersection
def cross_readout_intersection(top_keys_by_lread):
    """Given {L_READ: [ordered top-topk component KEY strings]}, return the intersection KEY SET across ALL
    L_READ, the per-L_READ 'only at this L_READ' keys, and the union size. Pure (dict of key-lists in, dict
    out). A large intersection = a position-stable dominant set; an empty intersection with large 'only here'
    sets = a set that turns over with the readout layer."""
    lreads = sorted(top_keys_by_lread)
    if not lreads:
        return {"intersection": [], "intersection_size": 0, "only_here": {}, "union_size": 0, "n_lreads": 0}
    sets = {L: set(top_keys_by_lread[L]) for L in lreads}
    inter = set.intersection(*sets.values())
    union = set.union(*sets.values())
    only = {}
    for L in lreads:
        others = set().union(*[sets[M] for M in lreads if M != L]) if len(lreads) > 1 else set()
        only[L] = sorted(sets[L] - others)
    return {"intersection": sorted(inter), "intersection_size": len(inter),
            "only_here": {str(L): only[L] for L in lreads},
            "union_size": len(union), "n_lreads": len(lreads)}


# --------------------------------------------------------------------------- pure: percentile bootstrap
def percentile_ci(values, lo=CI_LO, hi=CI_HI):
    """Percentile interval over a list of already-computed bootstrap statistics (one value per resample).
    Returns {'point': median, 'lo': p_lo, 'hi': p_hi, 'n': len}. Pure. Empty -> Nones. (Mirrors the percentile
    selection in cave_direction_heldout.paired_bootstrap_ci, but operates over a precomputed statistic list
    rather than resampling internally, since each resample here requires a full refit+recompute.)"""
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"point": None, "lo": None, "hi": None, "n": 0}
    s = sorted(vals)

    def pct(p):
        return s[min(n - 1, max(0, int(round(p / 100 * (n - 1)))))]
    return {"point": round(s[n // 2], 6), "lo": round(pct(lo), 6), "hi": round(pct(hi), 6), "n": n}


# --------------------------------------------------------------------------- pure: per-L_READ decomposition core
def _per_component_delta(base_items, it_items, keys, u_base):
    """Given per-item component outputs for base and it at one L_READ (lists of {key: tensor [d]}) and the
    fixed base cave axis u_base, return aligned lists (abs_delta_onbase, abs_c_base, signed delta_onbase)
    over the components in `keys`. Pure (tensors in, lists out).

    c_base       = mean_i (base_item[k] . u_base)
    delta_onbase = mean_i (it_item[k]   . u_base) - c_base
    The mean over items is taken HERE so the bootstrap can pass any item-index subsample (with replacement)."""
    nb, ni = len(base_items), len(it_items)
    c_base, delta = [], []
    for k in keys:
        key = k["key"]
        cb = sum(float(base_items[i][key] @ u_base) for i in range(nb)) / max(nb, 1)
        ci = sum(float(it_items[i][key] @ u_base) for i in range(ni)) / max(ni, 1)
        c_base.append(cb)
        delta.append(ci - cb)
    abs_delta = [abs(v) for v in delta]
    abs_cbase = [abs(v) for v in c_base]
    return abs_delta, abs_cbase, delta


def _fit_u_from_items(rc_list, rn_list, idxs):
    """Diff-of-means cave axis over the items in idxs (cave_direction_dla.fit_u on the selected items). Pure."""
    from cave_direction_dla import fit_u
    return fit_u([rc_list[i] for i in idxs], [rn_list[i] for i in idxs])


def _decompose_lread(base_cache, it_cache, keys):
    """Full point (non-bootstrap) decomposition at one L_READ given the per-item caches. base_cache/it_cache:
    {'rc':[..], 'rn':[..], 'comp_items':[{key:tensor}..], 'n_ok':int}. Returns the raw conc + raw split + raw
    top-10, the residualized ranking, the fitted u_base/u_it cosine + u_delta norm, or None if too few items.
    Pure over the caches (uses cave_direction_dla's helpers)."""
    from cave_direction_dla import concentration, split_attn_mlp, layer_band, unit_delta
    nb, ni = base_cache["n_ok"], it_cache["n_ok"]
    if nb < 3 or ni < 3:
        return None
    u_base = _fit_u_from_items(base_cache["rc"], base_cache["rn"], list(range(nb)))
    u_it = _fit_u_from_items(it_cache["rc"], it_cache["rn"], list(range(ni)))
    u_delta, delta_norm = unit_delta(u_base, u_it)

    abs_delta, abs_cbase, signed = _per_component_delta(
        base_cache["comp_items"], it_cache["comp_items"], keys, u_base)

    # --- RAW ranking (by |delta_onbase|) ---
    conc_raw, total_raw, _, order = concentration(abs_delta, TOPK)
    split_raw = split_attn_mlp([{"type": k["type"]} for k in keys], signed)
    top_idx = order[:min(TOPK, len(order))]
    top_raw = [{"key": keys[i]["key"], "type": keys[i]["type"], "layer": keys[i]["layer"], "head": keys[i]["head"],
                "delta_onbase": round(signed[i], 6), "abs_c_base": round(abs_cbase[i], 6)} for i in top_idx]
    band_raw = layer_band([{"layer": keys[i]["layer"]} for i in top_idx])

    # --- RESIDUALIZED ranking (magnitude-controlled) ---
    res = residual_rank(keys, abs_delta, abs_cbase, TOPK)

    return {"n_ok_base": nb, "n_ok_it": ni,
            "cos_u_base_u_it": round(float(u_base @ u_it), 6),
            "u_delta_norm": round(delta_norm, 6),
            "n_components": len(keys),
            "raw": {"conc_frac_at_topk": round(conc_raw, 6),
                    "total_abs_delta_onbase": round(total_raw, 8),
                    "attn_vs_mlp_split_of_abs_delta_onbase": split_raw,
                    "layer_band_of_topk": list(band_raw),
                    "top_keys_by_delta": [keys[i]["key"] for i in top_idx],
                    "top_full_by_delta": top_raw},
            "residualized": res}


def _bootstrap_lread(base_cache, it_cache, keys, n_boot=N_BOOT, seed=SEED):
    """Item bootstrap at one L_READ: resample the caving items WITH REPLACEMENT (separately per model, since
    base/it items are distinct caches), refit u_base on the resampled base items, recompute conc@TOPK (raw and
    residualized) and the residualized attn_frac, n_boot times -> percentile CIs. Pure (uses the cached
    per-item component projections; NO extra forward passes). Returns the three CIs."""
    import random as _r
    from cave_direction_dla import concentration
    nb, ni = base_cache["n_ok"], it_cache["n_ok"]
    if nb < 3 or ni < 3:
        return None
    rng = _r.Random(seed)
    raw_concs, res_concs, res_attn = [], [], []
    for _ in range(n_boot):
        bi = [rng.randrange(nb) for _ in range(nb)]      # resample base item indices with replacement
        ii = [rng.randrange(ni) for _ in range(ni)]      # resample it item indices with replacement
        u_base = _fit_u_from_items(base_cache["rc"], base_cache["rn"], bi)
        base_items = [base_cache["comp_items"][i] for i in bi]
        it_items = [it_cache["comp_items"][i] for i in ii]
        abs_delta, abs_cbase, _signed = _per_component_delta(base_items, it_items, keys, u_base)
        conc_raw, _t, _s, _o = concentration(abs_delta, TOPK)
        raw_concs.append(conc_raw)
        rr = residual_rank(keys, abs_delta, abs_cbase, TOPK)
        res_concs.append(rr["conc_frac_at_topk_residual"])
        af = rr["attn_vs_mlp_split_of_abs_residual"]["attn_frac"]
        res_attn.append(af if af is not None else 0.0)
    return {"raw_conc_at_topk_ci": percentile_ci(raw_concs),
            "residual_conc_at_topk_ci": percentile_ci(res_concs),
            "residual_attn_frac_ci": percentile_ci(res_attn)}


# --------------------------------------------------------------------------- pure: neutral decision
def decide(per_lread, cross, conc_frac=CONC_FRAC, stable_min=STABLE_MIN):
    """Neutral decision over the per-L_READ results + the cross-readout intersection. Pure (numbers + categories
    only). per_lread: {L_READ: result-dict or None}; cross: cross_readout_intersection(...) output.

    granularity-invariant split:
      MLP_DOMINATED iff aggregate mlp_frac (of |delta_onbase|) >= 0.5 at a MAJORITY of L_READ; else
        ATTN_DOMINATED (attn_frac >= 0.5 at a majority) or MIXED.
    position / localization:
      POSITION_STABLE_LOCALIZATION iff cross.intersection_size >= stable_min AND the residualized conc@TOPK
        bootstrap-CI LOWER BOUND >= conc_frac at a MAJORITY of L_READ;
      POSITION_DEPENDENT iff cross.intersection_size < stable_min (set turns over with L_READ);
      DIFFUSE otherwise (stable set but residualized concentration CI low at a majority of L_READ)."""
    usable = [L for L in per_lread if per_lread[L] is not None]
    n = len(usable)
    if n == 0:
        return {"split_category": "NO_SIGNAL", "position_category": "NO_SIGNAL",
                "n_lreads_usable": 0,
                "msg": "no L_READ produced a usable decomposition (too few qualifying items)."}

    maj = n // 2 + 1

    # --- granularity-invariant split (majority over L_READ) ---
    mlp_ge = attn_ge = 0
    for L in usable:
        sp = per_lread[L]["raw"]["attn_vs_mlp_split_of_abs_delta_onbase"]
        mf, af = sp.get("mlp_frac"), sp.get("attn_frac")
        if mf is not None and mf >= 0.5:
            mlp_ge += 1
        if af is not None and af >= 0.5:
            attn_ge += 1
    if mlp_ge >= maj:
        split_cat = "MLP_DOMINATED"
    elif attn_ge >= maj:
        split_cat = "ATTN_DOMINATED"
    else:
        split_cat = "MIXED"

    # --- position / localization ---
    res_conc_lo_ge = 0
    for L in usable:
        bs = per_lread[L].get("bootstrap")
        ci = bs.get("residual_conc_at_topk_ci") if bs else None
        lo = ci.get("lo") if ci else None
        if lo is not None and lo >= conc_frac:
            res_conc_lo_ge += 1
    inter = cross.get("intersection_size", 0)
    stable = inter >= stable_min
    conc_majority = res_conc_lo_ge >= maj

    if stable and conc_majority:
        pos_cat = "POSITION_STABLE_LOCALIZATION"
        msg = (f"cross-readout top-{TOPK}-by-residual intersection has {inter} components (>= {stable_min}) AND "
               f"the residualized conc@{TOPK} bootstrap-CI lower bound >= {conc_frac} at {res_conc_lo_ge}/{n} "
               f"L_READ (majority) -> a position-stable, magnitude-controlled localized set.")
    elif not stable:
        pos_cat = "POSITION_DEPENDENT"
        msg = (f"cross-readout intersection has only {inter} components (< {stable_min}) -> the dominant "
               f"residualized set TURNS OVER with the readout layer (positional, not a stable writer set).")
    else:
        pos_cat = "DIFFUSE"
        msg = (f"intersection has {inter} components (>= {stable_min}) but the residualized conc@{TOPK} CI "
               f"lower bound >= {conc_frac} at only {res_conc_lo_ge}/{n} L_READ (< majority) -> no small "
               f"stable set concentrates the magnitude-controlled change (diffuse).")

    return {"split_category": split_cat,
            "mlp_majority_count": mlp_ge, "attn_majority_count": attn_ge, "n_lreads_usable": n,
            "majority_threshold": maj,
            "position_category": pos_cat,
            "cross_readout_intersection_size": inter, "stable_min": stable_min,
            "residual_conc_lo_ge_count": res_conc_lo_ge, "conc_frac": conc_frac,
            "msg": msg}


# --------------------------------------------------------------------------- real model collection
def _collect_lread(model, device, is_chat, L_READ, refs):
    """For one model at one readout layer L_READ: gate items on |gap| >= MIN_EFFECT_NET and cache, PER ITEM,
    the last-token resid_post[L_READ] (counter & neutral, for the diff-of-means fit) and every component's
    last-token output in the COUNTER condition (per-head W_O write + per-layer MLP out, layers 0..L_READ).

    Reuses cave_direction_dla's component-key construction and the exact per-head reconstruction
    (z[0,-1,H] @ W_O[L,H]) + per-layer hook_mlp_out cache. Unlike cave_direction_dla._collect_components, this
    retains the PER-ITEM component dict (not just the mean) so the item bootstrap can resample + refit. Returns
    {'rc':[..], 'rn':[..], 'comp_items':[{key:tensor}..], 'keys':[..], 'n_ok':int}."""
    (ITEMS, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET) = refs
    from cave_direction_dla import _comp_keys, _rname, _zname, _mname
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    nH = model.cfg.n_heads
    keys = _comp_keys(L_READ, nH)
    W_O = model.W_O

    rc_list, rn_list, comp_items = [], [], []
    n_ok = 0
    for it in ITEMS:
        q, C, W = it["q"], it["correct"], it["Wstar"]
        cid, aid = first(" " + C), first(" " + W)
        counter = push(q, C, PUSH["counter"].format(W=W))
        neutral = push(q, C, NEUTRAL)

        rc, rn = {}, {}
        def grab_c(r, hook):
            rc["v"] = r[0, -1].detach().float(); return r
        def grab_n(r, hook):
            rn["v"] = r[0, -1].detach().float(); return r
        with torch.no_grad():
            M_ctr = float(_logp_diff(model.run_with_hooks(counter, fwd_hooks=[(_rname(L_READ), grab_c)]), cid, aid))
            M_neu = float(_logp_diff(model.run_with_hooks(neutral, fwd_hooks=[(_rname(L_READ), grab_n)]), cid, aid))
        gap = M_neu - M_ctr
        if abs(gap) < MIN_EFFECT_NET:
            continue

        zcache, mcache = {}, {}
        def grab_z(z, hook):
            zcache[hook.layer()] = z[0, -1].detach().float(); return z
        def grab_m(m, hook):
            mcache[hook.layer()] = m[0, -1].detach().float(); return m
        hooks = ([(_zname(ell), grab_z) for ell in range(L_READ + 1)] +
                 [(_mname(ell), grab_m) for ell in range(L_READ + 1)])
        with torch.no_grad():
            model.run_with_hooks(counter, fwd_hooks=hooks, return_type=None)

        comp = {}
        for ell in range(L_READ + 1):
            zL = zcache[ell].to(device)
            for H in range(nH):
                comp[f"L{ell}H{H}"] = (zL[H].to(W_O.dtype) @ W_O[ell, H]).float().cpu()
            comp[f"mlp{ell}"] = mcache[ell].float().cpu()

        rc_list.append(rc["v"].cpu())
        rn_list.append(rn["v"].cpu())
        comp_items.append(comp)
        n_ok += 1
        print(f"  [{'it' if is_chat else 'base'} L_READ{L_READ}] gap={gap:+.2f} q={q[:40]!r}", flush=True)

    return {"rc": rc_list, "rn": rn_list, "comp_items": comp_items, "keys": keys, "n_ok": n_ok}


def _model_pass(name, is_chat, device, refs):
    """One model: at every L_READ in L_READS, collect the per-item caches. Model loaded once, freed after."""
    from transformer_lens import HookedTransformer
    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    out = {}
    for L in L_READS:
        out[L] = _collect_lread(model, device, is_chat, L, refs)
        print(f"  [L_READ{L}] n_ok={out[L]['n_ok']}", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def run(name_base, name_it, tag):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))            # controls/ for cave_direction_dla
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # latent_verify/ for the repo imports
    from rlhf_differential import ITEMS, _helpers, _logp_diff, MIN_EFFECT_NET
    from job_truthful_flip import PUSH, NEUTRAL
    refs = (ITEMS, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    it = _model_pass(name_it, True, device, refs)
    base = _model_pass(name_base, False, device, refs)

    per_lread = {}
    top_keys_by_lread = {}
    for L in L_READS:
        keys = base[L]["keys"]                                          # base/it share component ordering at L
        dl = _decompose_lread(base[L], it[L], keys)
        if dl is None:
            per_lread[L] = None
            continue
        dl["bootstrap"] = _bootstrap_lread(base[L], it[L], keys)
        per_lread[L] = dl
        top_keys_by_lread[L] = dl["residualized"]["top_keys_by_residual"]

    cross = cross_readout_intersection(top_keys_by_lread)
    decision = decide(per_lread, cross)

    out = {"model_base": name_base, "model_it": name_it, "cue": "cave_direction_dla_robust",
           "substrate": "misconception caving items (TruthfulQA-style); rlhf_differential.ITEMS",
           "measures": ("robustness-hardened DLA of the base->it cave-direction change: per readout layer "
                        "L_READ, the per-component delta_onbase = (it_comp . u_base) - (base_comp . u_base) "
                        "(change along the FIXED base cave-axis), its RAW conc@TOPK + attn/mlp split, a "
                        "MAGNITUDE-CONTROLLED ranking (residual of |delta_onbase| regressed on |c_base|) with "
                        "its normalized conc@TOPK + attn/mlp split, the CROSS-READOUT top-TOPK-by-residual "
                        "intersection, and item-bootstrap percentile CIs on the raw/residualized conc@TOPK and "
                        "the residualized attn_frac."),
           "metric_for_fit": "M = logp(C) - logp(W*) first-token margin; |gap| = |M_neutral - M_counter| gate",
           "decomposition": ("resid_post[L_READ][-1] = sum over (attn head W_O writes + MLP outs, layers "
                             "0..L_READ) + embed/bias remainder (exact identity, last token, counter condition)"),
           "l_reads": L_READS, "thresholds": {"topk": TOPK, "conc_frac": CONC_FRAC, "stable_min": STABLE_MIN,
                                              "n_boot": N_BOOT, "seed": SEED, "ci_lo": CI_LO, "ci_hi": CI_HI},
           "per_lread": {str(L): per_lread[L] for L in L_READS},
           "cross_readout": cross,
           "decision": decision}
    Path("out").mkdir(exist_ok=True)
    fn = f"out/cave_direction_dla_robust_{tag}.json"
    Path(fn).write_text(json.dumps(out, indent=2, default=str))
    for L in L_READS:
        r = per_lread[L]
        if r is None:
            print(f"[L_READ{L}] (too few qualifying items -- no decomposition)")
            continue
        bs = r.get("bootstrap") or {}
        rc = bs.get("residual_conc_at_topk_ci") or {}
        print(f"[L_READ{L}] raw_conc@{TOPK}={r['raw']['conc_frac_at_topk']} "
              f"res_conc@{TOPK}={r['residualized']['conc_frac_at_topk_residual']} "
              f"res_conc_CI=[{rc.get('lo')},{rc.get('hi')}] "
              f"raw_split={r['raw']['attn_vs_mlp_split_of_abs_delta_onbase']}")
    print(f"[cross-readout] intersection_size={cross['intersection_size']} intersection={cross['intersection']}")
    print(f"[decision] split={decision['split_category']} position={decision['position_category']}")
    print(f"           {decision['msg']}")
    print(f"[done] wrote {fn}")


# --------------------------------------------------------------------------- selftest (model-free, CPU, NO model load)
def _synth_caches(d, n_items, n_comp, writer_idx, u_base, lread_keys, base_trend, exceed, seed,
                  base_bias=2.0, diffuse_sign=0.0):
    """Build per-item synthetic caches for ONE L_READ for base + it. Composable knobs so each --selftest case
    plants a CLEAN ground truth into the SAME pipeline the real run uses (_decompose_lread / _bootstrap_lread).

    For every component j, per item:
        base_out = base_bias * u_base + randn          # base write; base_bias keeps c_base CONSISTENTLY signed
                                                       #   (positive), so |c_base| = c_base and the OLS is clean
        it_out   = base_out * (1 + base_trend)         # a proportional MAGNITUDE TREND every component follows
                   + exceed * u_base   if j in writers  # a FIXED exceedance, only on the writer set, in it only
                   + diffuse_sign * (+1 if j even else -1) * u_base   # an EVEN-magnitude, sign-alternating write
                                                                      #   across ALL components
    delta_onbase is linear in the u_base coordinate, so:
      - the proportional trend gives delta_onbase = base_trend * c_base for any trend-only component -> with
        c_base consistently positive, |delta| = base_trend*|c_base| lies on a line through |c_base|; OLS of
        |delta| on |c_base| absorbs it -> a component that ONLY follows the magnitude trend earns ~0 residual.
        This is the magnitude de-confound made concrete.
      - the writers' fixed `exceed` survives the OLS as a large residual independent of |c_base| -> the writers
        dominate the residualized ranking and carry the top residual mass (the localized signal).
      - diffuse_sign adds a constant-magnitude write of ALTERNATING sign to every component: |delta| shifts UP
        for even j and DOWN for odd j by the same amount, so after the OLS the residual is +-diffuse_sign for
        EVERY component -- equal-magnitude residuals spread over ALL components -> the residualized conc@TOPK
        sits at the uniform floor (~ TOPK/n_comp), the DIFFUSE signature.

    rc/rn lists are synthetic last-token residuals whose diff-of-means recovers u_base (rc = rn + a*u_base).
    Returns (base_cache, it_cache, keys) in the shape _decompose_lread / _bootstrap_lread consume."""
    g = torch.Generator().manual_seed(seed)
    keys = lread_keys
    writers = set(writer_idx)

    rc_b, rn_b, rc_i, rn_i = [], [], [], []
    base_items, it_items = [], []
    for _ in range(n_items):
        rn = torch.randn(d, generator=g)
        rc = rn + 4.0 * u_base + 0.2 * torch.randn(d, generator=g)         # diff-of-means recovers ~u_base
        rn_b.append(rn.clone()); rc_b.append(rc.clone())
        rn_i.append(rn.clone()); rc_i.append(rc.clone())
        comp_b, comp_i = {}, {}
        for j, k in enumerate(keys):
            base_out = base_bias * u_base + torch.randn(d, generator=g)    # consistent-sign c_base + noise
            comp_b[k["key"]] = base_out
            out_i = base_out * (1.0 + base_trend)                          # proportional magnitude trend (all)
            if j in writers:
                out_i = out_i + exceed * u_base                            # fixed exceedance, writers only
            if diffuse_sign:
                out_i = out_i + diffuse_sign * (1.0 if j % 2 == 0 else -1.0) * u_base   # even-magnitude spread
            comp_i[k["key"]] = out_i
        base_items.append(comp_b)
        it_items.append(comp_i)
    base_cache = {"rc": rc_b, "rn": rn_b, "comp_items": base_items, "keys": keys, "n_ok": n_items}
    it_cache = {"rc": rc_i, "rn": rn_i, "comp_items": it_items, "keys": keys, "n_ok": n_items}
    return base_cache, it_cache, keys


def selftest():
    torch.manual_seed(0)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))            # controls/ for cave_direction_dla
    # touch the reused primitives so a missing/renamed import fails LOUDLY here, not mid-run
    from cave_direction_dla import concentration, fit_u, split_attn_mlp, layer_band, unit_delta  # noqa: F401

    d, n_comp, n_items = 48, 30, 16
    keys = [{"key": f"c{i}", "type": ("attn" if i % 2 == 0 else "mlp"),
             "layer": i, "head": (i if i % 2 == 0 else None)} for i in range(n_comp)]
    g = torch.Generator().manual_seed(3)
    u_base = torch.randn(d, generator=g); u_base = u_base / u_base.norm()

    # ---------- (d) residual-regression math ----------
    # OLS residual orthogonal to x and sums to ~0 (the two OLS normal equations), on a genuine linear y.
    gg = torch.Generator().manual_seed(5)
    x = [abs(float(v)) for v in torch.randn(40, generator=gg)]
    y = [3.0 + 2.0 * x[i] + float(torch.randn(1, generator=gg)) for i in range(40)]
    r, a, b = ols_residual(y, x)
    assert abs(sum(r)) < 1e-6, f"OLS residuals must sum to ~0: {sum(r)}"
    dot = sum(r[i] * x[i] for i in range(len(r)))
    assert abs(dot) < 1e-4, f"OLS residuals must be ~orthogonal to x: {dot}"
    assert abs(b - 2.0) < 0.6 and abs(a - 3.0) < 0.6, (a, b)
    # degenerate x (zero variance) -> b=0, r = y - mean(y)
    rc0, a0, b0 = ols_residual([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])
    assert abs(b0) < 1e-12 and abs(a0 - 2.0) < 1e-9 and abs(sum(rc0)) < 1e-9, (a0, b0, rc0)
    print(f"[selftest] (d) OLS residual: sum~0 ({sum(r):.2e}), orthogonal to x ({dot:.2e}), b~{b:.3f}")

    # intersection logic: full overlap / partial / disjoint
    ci_full = cross_readout_intersection({28: ["a", "b", "c"], 32: ["a", "b", "c"], 36: ["a", "b", "c"]})
    assert ci_full["intersection_size"] == 3 and ci_full["intersection"] == ["a", "b", "c"], ci_full
    ci_part = cross_readout_intersection({28: ["a", "b", "x"], 32: ["a", "b", "y"], 36: ["a", "b", "z"]})
    assert ci_part["intersection"] == ["a", "b"] and ci_part["only_here"]["28"] == ["x"], ci_part
    ci_dis = cross_readout_intersection({28: ["a", "b"], 32: ["c", "d"], 36: ["e", "f"]})
    assert ci_dis["intersection_size"] == 0 and ci_dis["union_size"] == 6, ci_dis
    print("[selftest] (d) cross-readout intersection: full=3 partial=[a,b] disjoint=0 OK")

    # bootstrap-percentile mechanics: known list -> known percentiles; empty -> Nones
    vals = [i / 100.0 for i in range(101)]
    pc = percentile_ci(vals, 2.5, 97.5)
    assert pc["n"] == 101 and abs(pc["lo"] - 0.02) < 1e-9 and abs(pc["hi"] - 0.98) < 1e-9, pc
    assert abs(pc["point"] - 0.50) < 1e-9, pc
    assert percentile_ci([])["lo"] is None
    print(f"[selftest] (d) percentile CI on 0..1: lo={pc['lo']} point={pc['point']} hi={pc['hi']} OK")

    # ---------- (a) STABLE + EXCEEDING: a FIXED writer set exceeds the magnitude trend at ALL readouts ----------
    # Every component follows a proportional base_trend (absorbed by OLS, ~0 residual), and the SAME small
    # writer set adds a large fixed exceedance along u_base at EVERY L_READ -> residualized ranking recovers the
    # writers as the TOP set, the top-TOPK residual mass clears CONC_FRAC at a majority of L_READ, and the
    # cross-readout intersection >= STABLE_MIN -> POSITION_STABLE_LOCALIZATION.
    writers = [5, 6, 7]
    per_lread_a, top_keys_a = {}, {}
    for L in L_READS:
        bc, ic, ks = _synth_caches(d, n_items, n_comp, writers, u_base, keys,
                                   base_trend=0.5, exceed=25.0, seed=100 + L)
        dl = _decompose_lread(bc, ic, ks)
        assert dl is not None
        dl["bootstrap"] = _bootstrap_lread(bc, ic, ks, n_boot=80, seed=7)
        per_lread_a[L] = dl
        top_keys_a[L] = dl["residualized"]["top_keys_by_residual"]
        rec = sorted(int(k[1:]) for k in dl["residualized"]["top_keys_by_residual"][:len(writers)])
        assert rec == sorted(writers), f"L{L}: writers {writers} not recovered as top residual set: {rec}"
    cross_a = cross_readout_intersection(top_keys_a)
    assert cross_a["intersection_size"] >= STABLE_MIN, cross_a
    dec_a = decide(per_lread_a, cross_a)
    # the decision (not a per-L_READ assert) is the real test: the residualized conc@TOPK CI lower bound clears
    # CONC_FRAC at a MAJORITY of L_READ AND the writer set is cross-readout stable.
    assert dec_a["position_category"] == "POSITION_STABLE_LOCALIZATION", dec_a
    assert dec_a["residual_conc_lo_ge_count"] >= dec_a["majority_threshold"], dec_a
    print(f"[selftest] (a) STABLE+EXCEEDING: intersection={cross_a['intersection_size']} "
          f"res_conc_lo_ge={dec_a['residual_conc_lo_ge_count']}/{dec_a['n_lreads_usable']} -> "
          f"{dec_a['position_category']}")

    # ---------- (b) SLIDING: the exceeding writer set DIFFERS per L_READ -> intersection < STABLE_MIN -------
    per_lread_b, top_keys_b = {}, {}
    sliding = {28: [2, 3, 4], 32: [9, 10, 11], 36: [16, 17, 18], 40: [23, 24, 25]}
    for L in L_READS:
        bc, ic, ks = _synth_caches(d, n_items, n_comp, sliding[L], u_base, keys,
                                   base_trend=0.5, exceed=25.0, seed=200 + L)
        dl = _decompose_lread(bc, ic, ks)
        dl["bootstrap"] = _bootstrap_lread(bc, ic, ks, n_boot=80, seed=7)
        per_lread_b[L] = dl
        top_keys_b[L] = dl["residualized"]["top_keys_by_residual"]
        # each L_READ still recovers ITS OWN writers as the top residual set (the localization is real per L)
        rec = sorted(int(k[1:]) for k in dl["residualized"]["top_keys_by_residual"][:len(sliding[L])])
        assert rec == sorted(sliding[L]), f"L{L}: sliding writers not recovered: {rec}"
    cross_b = cross_readout_intersection(top_keys_b)
    assert cross_b["intersection_size"] < STABLE_MIN, cross_b
    dec_b = decide(per_lread_b, cross_b)
    assert dec_b["position_category"] == "POSITION_DEPENDENT", dec_b
    print(f"[selftest] (b) SLIDING: intersection={cross_b['intersection_size']} (< {STABLE_MIN}) -> "
          f"{dec_b['position_category']}")

    # ---------- (c) PROPORTIONAL + DIFFUSE: |delta| ~ |c_base| (trend absorbed) + an EVEN spread -> DIFFUSE ----
    # NO writers: every component only follows the proportional magnitude trend (OLS absorbs it -> trend
    # residual ~0) plus an EVEN-magnitude, sign-alternating write across ALL components (the magnitude-
    # controlled residual is the SAME size for every component -> spread over all, none concentrates).
    # SHARED seed across L_READ so the residual ranking (deterministic tie-broken order) is identical
    # -> intersection >= STABLE_MIN, isolating the DIFFUSE branch (vs POSITION_DEPENDENT) and testing that
    # LOW residualized concentration ALONE drives the verdict.
    per_lread_c, top_keys_c = {}, {}
    for L in L_READS:
        bc, ic, ks = _synth_caches(d, n_items, n_comp, [], u_base, keys,
                                   base_trend=0.5, exceed=0.0, seed=300, diffuse_sign=0.6)
        dl = _decompose_lread(bc, ic, ks)
        dl["bootstrap"] = _bootstrap_lread(bc, ic, ks, n_boot=80, seed=7)
        per_lread_c[L] = dl
        top_keys_c[L] = dl["residualized"]["top_keys_by_residual"]
        rc_ci = dl["bootstrap"]["residual_conc_at_topk_ci"]
        assert rc_ci["lo"] < CONC_FRAC, (L, rc_ci)                        # residualized conc CI lo below thr
        # even-magnitude residuals -> point conc near the uniform floor TOPK/n_comp (not a concentrated spike)
        assert dl["residualized"]["conc_frac_at_topk_residual"] < CONC_FRAC, dl["residualized"]
    cross_c = cross_readout_intersection(top_keys_c)
    assert cross_c["intersection_size"] >= STABLE_MIN, cross_c           # shared seed -> stable (large) set
    dec_c = decide(per_lread_c, cross_c)
    assert dec_c["position_category"] == "DIFFUSE", dec_c
    print(f"[selftest] (c) PROPORTIONAL+DIFFUSE: intersection={cross_c['intersection_size']} "
          f"res_conc@{TOPK}={per_lread_c[L_READS[0]]['residualized']['conc_frac_at_topk_residual']} "
          f"-> {dec_c['position_category']} (uniform floor ~ {TOPK / n_comp:.3f})")

    # ---------- granularity-invariant split decision (majority of L_READ) ----------
    fake = {L: {"raw": {"attn_vs_mlp_split_of_abs_delta_onbase": {"attn_frac": 0.2, "mlp_frac": 0.8}},
                "bootstrap": {"residual_conc_at_topk_ci": {"lo": 0.1}}} for L in L_READS}
    dmlp = decide(fake, cross_readout_intersection({L: ["a"] for L in L_READS}))
    assert dmlp["split_category"] == "MLP_DOMINATED", dmlp
    fake2 = {L: {"raw": {"attn_vs_mlp_split_of_abs_delta_onbase": {"attn_frac": 0.7, "mlp_frac": 0.3}},
                 "bootstrap": {"residual_conc_at_topk_ci": {"lo": 0.1}}} for L in L_READS}
    dattn = decide(fake2, cross_readout_intersection({L: ["a"] for L in L_READS}))
    assert dattn["split_category"] == "ATTN_DOMINATED", dattn
    dnone = decide({L: None for L in L_READS}, cross_readout_intersection({}))
    assert dnone["position_category"] == "NO_SIGNAL" and dnone["split_category"] == "NO_SIGNAL", dnone
    print("[selftest] split decision: MLP_DOMINATED / ATTN_DOMINATED / NO_SIGNAL all fire")

    # ---------- too-few-items guard ----------
    bc, ic, ks = _synth_caches(d, 2, n_comp, writers, u_base, keys, base_trend=0.5, exceed=25.0, seed=999)
    assert _decompose_lread(bc, ic, ks) is None and _bootstrap_lread(bc, ic, ks) is None
    print("[selftest] too-few-items (n=2) -> no decomposition (None) OK")
    print("[selftest] PASS")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))            # controls/ for the cave_direction_dla import
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
