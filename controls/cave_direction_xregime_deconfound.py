"""CROSS-REGIME PROJ_N / ITEM-SET de-confound for the cave/defer DIRECTION transfer.

WHY (neutral). cave_direction_heldout.py measures the necessity of a diff-of-means cave direction across
regimes: fit u in model A (donor), apply it inside model B (host), read the recovery. When the cross-regime
number is LOWER than the within-regime reference, three distinct things can produce that drop and the
existing cross path cannot tell them apart:

  (i)   the directions genuinely differ (u_donor and u_host point different ways);
  (ii)  the shift TARGET proj_n is mis-calibrated. The necessity ablation moves the counter residual's
        u-projection to a target neutral mean. cave_direction_heldout computes that target on the DONOR's
        residual cache (dfit["proj_n"]). base and -it have different residual magnitudes/offsets, so the
        donor's neutral-mean projection is OFF-DISTRIBUTION when added to the host residual -- the recovery
        can drop purely because the target is on the wrong scale, with the SAME direction;
  (iii) the two models qualify (|gap| >= MIN_EFFECT) on DIFFERENT item subsets, so within and cross are not
        measured on the same items.

This control separates them WITH NO NEW MECHANISM. It reuses the diff-of-means fit, the necessity ablation
(shift the u-projection to a target proj_n; frac = (M_ablate - M_counter)/(M_neutral - M_counter)), the
matched-magnitude random-direction floor, the held-out / k-fold split, the |gap| >= MIN_EFFECT_NET gate, the
FIT_LAYERS sweep, the metric M = logp(C) - logp(W*), and the _helpers/_logp_diff/first machinery -- all
imported from the reference modules. The only new logic is: a second cross variant whose target proj_n is
recomputed on the HOST's own neutral residuals along u_donor (the scale-calibration-corrected cross), a
shared-item intersection fit, and the pure decision -- all covered by the model-free --selftest.

MEASURED per ordered pair (base->it: donor=base, host=it; it->base: donor=it, host=base), at EVERY fit layer
(all layers reported; headline = host layer with the largest WITHIN necessity):
  WITHIN              : host's own held-out diff-of-means u_host; ablate on host held-out items toward the
                        HOST proj_n (the within-regime reference; k-fold held-out).
  CROSS_DONOR_PROJN   : donor's u_donor; ablate on host items toward the DONOR's proj_n. (= the existing
                        cross-regime path in cave_direction_heldout.)
  CROSS_HOST_PROJN    : donor's u_donor; ablate on host items toward proj_n = mean over HOST neutral
                        residuals of (rn . u_donor). (the scale-calibration-corrected cross.)
  MATCHED_ITEM        : restrict to the SHARED intersection (items qualifying in BOTH models by |gap| >=
                        MIN_EFFECT_NET); fit u_base and u_it on the SAME shared items with a held-out split;
                        cross-apply on shared held-out items, computing BOTH matched_item_donor_projn (donor
                        proj_n) and matched_item_host_projn (host neutral mean along u_donor).
  random floor        : matched-magnitude random-direction necessity per host (the recovery a meaningless
                        unit direction reaches; reported alongside, not in the decision).

NEUTRAL DECISION (module constants DIR_THR=0.20, XREG_RATIO=0.6; numbers + categories only, no hypothesis,
no statement about which model or variant should win). Per (pair x variant in {cross_donor_projn,
cross_host_projn, matched_item_donor_projn, matched_item_host_projn}):
  SHARED_DIRECTION  iff cross necessity >= DIR_THR AND (cross/within) >= XREG_RATIO;
  REGIME_SPECIFIC   otherwise.

Forward-only (diff-of-means + projection edits; no backward) -> fits the 40GB A100. Each model is loaded
twice (its own within/own-fit pass, then a cross-apply pass that re-collects the donor's residual cache and
the shared-item residuals), reusing the verified primitives.

  python controls/cave_direction_xregime_deconfound.py --selftest
  python controls/cave_direction_xregime_deconfound.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

# Repo-internal imports (item pool, turn builders, metric, fit machinery) are deferred into run() so
# --selftest runs standalone on CPU with NO model load and nothing else on sys.path. On the box the
# reference files are scp'd flat into latent_verify/ where these resolve.

# --------------------------------------------------------------------------- pre-registered thresholds
FIT_LAYERS = [24, 28, 32, 36]   # same sweep as headset_direction.py / cave_direction_heldout.py
DIR_THR = 0.20                  # frac of the cave the cross direction must mediate to count
XREG_RATIO = 0.6                # cross necessity / within necessity must reach this for SHARED_DIRECTION
N_FOLDS = 5                     # k-fold split count for the within-regime held-out reference (capped at n)
SEED = 0
N_BOOT = 2000
CI_LO, CI_HI = 2.5, 97.5        # 95 percent percentile interval


# --------------------------------------------------------------------------- pure splitting / fitting
def kfold_splits(n, k, seed=SEED):
    """k near-equal disjoint test folds over shuffled range(n); each split = (train_idx, test_idx). Pure:
    deterministic via random.Random(seed). k capped at n. Identical to cave_direction_heldout.kfold_splits."""
    import random as _r
    if n < 2:
        return []
    k = max(2, min(k, n))
    idx = list(range(n))
    _r.Random(seed).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    folds = [f for f in folds if f]
    out = []
    for f in folds:
        test = sorted(f)
        train = sorted(i for i in idx if i not in set(f))
        if train and test:
            out.append((train, test))
    return out


def fit_direction(rc_list, rn_list, idxs):
    """Diff-of-means cave direction over `idxs` plus that fit's own neutral-mean projection.
    u = normalize(mean_i(rc_i - rn_i)); proj_n = mean_i(rn_i . u). Pure (tensors in, dict out). Mirrors
    headset_direction._dir_pass / matched_item_deconfound._effect_pass / cave_direction_heldout exactly."""
    D = torch.stack([rc_list[i] - rn_list[i] for i in idxs])
    cave = D.mean(0)
    u = cave / (cave.norm() + 1e-8)
    proj_n = statistics.mean(float(rn_list[i] @ u) for i in idxs)
    return {"u": u, "proj_n": proj_n}


def host_proj_n(host_ctxs, u):
    """The HOST's own neutral-mean projection ALONG a (possibly donor-fit) direction u:
    mean over the host's neutral residuals of (rn . u). This is the scale-calibration-corrected target --
    it re-anchors the necessity shift to the HOST's residual scale instead of the donor's. Pure."""
    return statistics.mean(float(c["rn"] @ u) for c in host_ctxs)


def random_direction(shape, dtype, gen):
    """Random unit direction (cpu generator -> caller moves it to device). Pure given the generator.
    Same construction as cave_direction_heldout.random_direction / headset_direction's matched-random."""
    rnd = torch.randn(shape, generator=gen).to(dtype)
    return rnd / (rnd.norm() + 1e-8)


def shared_items(base_gaps, it_gaps, min_eff):
    """Item keys qualifying in BOTH models by the |gap| >= min_eff gate (the same gate the per-model passes
    apply). Returns the sorted intersection. Pure (dict, dict -> sorted list). NOTE: |gap| gate (matches
    the per-model qualification in headset_direction / cave_direction_heldout / rlhf_differential), not the
    sign-restricted positive-cave gate of matched_item_deconfound -- this control measures DIRECTION
    transfer on every qualifying item, so it must select the SAME items the per-model fits used."""
    keys = [k for k in base_gaps if k in it_gaps
            and abs(base_gaps[k]) >= min_eff and abs(it_gaps[k]) >= min_eff]
    return sorted(keys)


# --------------------------------------------------------------------------- pure stats / decisions
def bootstrap_ci(values, seed=SEED, n_boot=N_BOOT, lo=CI_LO, hi=CI_HI):
    """Percentile bootstrap CI of the mean (matches matched_item_deconfound.bootstrap_ci /
    cave_direction_heldout.paired_bootstrap_ci). Pure (list -> dict)."""
    import random as _r
    n = len(values)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = _r.Random(seed)
    means = []
    for _ in range(n_boot):
        s = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    def pct(p):
        return means[min(n_boot - 1, max(0, int(round(p / 100 * (n_boot - 1)))))]
    return {"mean": round(sum(values) / n, 4), "lo": round(pct(lo), 4), "hi": round(pct(hi), 4), "n": n}


def decide(cross_nec, within_nec, dir_thr=DIR_THR, xreg_ratio=XREG_RATIO):
    """Cross-regime decision for one (pair x variant). Pure over the two measured numbers only -- no
    hypothesis, no statement about which model or variant should win.
      SHARED_DIRECTION : cross_nec >= dir_thr AND (cross/within) >= xreg_ratio (the donor's direction,
                         with this variant's target, mediates the host cave to >= xreg_ratio of the
                         within-regime strength).
      REGIME_SPECIFIC  : otherwise (cross necessity too low, or it recovers too little of within)."""
    ratio = (cross_nec / within_nec) if (cross_nec is not None and within_nec is not None and abs(within_nec) > 1e-6) else None
    shared = (cross_nec is not None and cross_nec >= dir_thr and
              ratio is not None and ratio >= xreg_ratio)
    if shared:
        cat = "SHARED_DIRECTION"
        msg = (f"cross necessity {cross_nec:.3f} >= {dir_thr} and cross/within {ratio:.2f} >= {xreg_ratio} "
               f"-- the direction mediates the host cave at >= {xreg_ratio} of within strength.")
    else:
        cat = "REGIME_SPECIFIC"
        rtxt = "n/a" if ratio is None else f"{ratio:.2f}"
        msg = (f"cross necessity {cross_nec if cross_nec is None else round(cross_nec, 3)} / cross-within "
               f"ratio {rtxt} below ({dir_thr}, {xreg_ratio}).")
    return {"category": cat, "shared_direction": cat == "SHARED_DIRECTION",
            "cross_nec": (round(cross_nec, 4) if cross_nec is not None else None),
            "within_nec": (round(within_nec, 4) if within_nec is not None else None),
            "ratio_cross_over_within": (round(ratio, 4) if ratio is not None else None), "msg": msg}


# --------------------------------------------------------------------------- necessity engine (real + synthetic)
def _necessity_items(eval_items, u, target_proj, M_fn):
    """For each item: ablate the u-projection on its counter residual TO target_proj and read M recovery
    frac = (M_ablate - M_counter)/(M_neutral - M_counter). Returns the per-item frac list. `target_proj` is
    the variant-specific target (donor proj_n, or host neutral mean along u). The shift and the formula are
    identical to cave_direction_heldout._necessity_items / headset_direction's necessity ablation; the ONLY
    thing this control varies is which target_proj is passed in.

    item must carry: rc (counter resid, 1-D tensor), M_ctr, gap (= M_neu - M_ctr, the signed cave)."""
    out = []
    for it in eval_items:
        shift = target_proj - float(it["rc"] @ u)
        M_ab = M_fn(it, shift, u)
        out.append((M_ab - it["M_ctr"]) / it["gap"])
    return out


def _within_heldout_nec(host_ctxs, M_fn):
    """WITHIN reference: host's own diff-of-means u_host on a TRAIN fold, ablated on the DISJOINT TEST fold
    toward the train fold's own neutral mean. k-fold; pool the test per-item fracs. Returns (mean, fracs)."""
    rc_list = [c["rc"] for c in host_ctxs]
    rn_list = [c["rn"] for c in host_ctxs]
    pooled = []
    for train, test in kfold_splits(len(host_ctxs), N_FOLDS):
        fit = fit_direction(rc_list, rn_list, train)
        pooled += _necessity_items([host_ctxs[i] for i in test], fit["u"], fit["proj_n"], M_fn)
    return (statistics.mean(pooled) if pooled else None), pooled


def _cross_nec_both(donor_ctxs, host_ctxs, M_fn):
    """CROSS: fit the donor's all-item direction u_donor (and its donor proj_n), then ablate on the HOST's
    items with BOTH targets:
      cross_donor_projn : target = donor proj_n (the donor's own neutral-mean projection -- the existing path).
      cross_host_projn  : target = host_proj_n(host_ctxs, u_donor) (host neutral mean along u_donor -- the
                          scale-calibration-corrected target).
    Returns ((donor_mean, donor_fracs), (host_mean, host_fracs))."""
    dfit = fit_direction([c["rc"] for c in donor_ctxs], [c["rn"] for c in donor_ctxs], list(range(len(donor_ctxs))))
    u = dfit["u"]
    fr_donor = _necessity_items(host_ctxs, u, dfit["proj_n"], M_fn)
    fr_host = _necessity_items(host_ctxs, u, host_proj_n(host_ctxs, u), M_fn)
    dm = statistics.mean(fr_donor) if fr_donor else None
    hm = statistics.mean(fr_host) if fr_host else None
    return (dm, fr_donor), (hm, fr_host)


def _matched_cross_nec(donor_train_ctxs, host_test_ctxs, M_fn):
    """MATCHED-ITEM cross: u_donor is fit on the donor's residuals over the SHARED items' TRAIN fold;
    ablate on the host's residuals over the SHARED items' (disjoint) TEST fold, with BOTH targets (donor
    proj_n, host neutral mean along u_donor over the host TEST items). Returns the same shape as
    _cross_nec_both. Held-out by construction (train/test are disjoint shared-item folds)."""
    dfit = fit_direction([c["rc"] for c in donor_train_ctxs], [c["rn"] for c in donor_train_ctxs],
                         list(range(len(donor_train_ctxs))))
    u = dfit["u"]
    fr_donor = _necessity_items(host_test_ctxs, u, dfit["proj_n"], M_fn)
    fr_host = _necessity_items(host_test_ctxs, u, host_proj_n(host_test_ctxs, u), M_fn)
    dm = statistics.mean(fr_donor) if fr_donor else None
    hm = statistics.mean(fr_host) if fr_host else None
    return (dm, fr_donor), (hm, fr_host)


def _rand_nec(host_ctxs, M_fn, device, seed=SEED):
    """Random-direction necessity floor, matched magnitude, exactly as cave_direction_heldout._rand_nec /
    headset_direction: a random unit ur, shift each host counter's ur-projection to ur's host neutral mean."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = host_ctxs[0]["rc"].shape[0]
    ur = random_direction((d,), host_ctxs[0]["rc"].dtype, g).to(device)
    prn = host_proj_n(host_ctxs, ur)
    fr = []
    for c in host_ctxs:
        shift = prn - float(c["rc"] @ ur)
        M_ab = M_fn(c, shift, ur)
        fr.append((M_ab - c["M_ctr"]) / c["gap"])
    return statistics.mean(fr) if fr else None


# --------------------------------------------------------------------------- real model passes
def _rname(L):
    return f"blocks.{L}.hook_resid_post"


def _collect_ctxs(model, device, is_chat, fit_layer, pool, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET):
    """Gap-pass for one model at one fit layer: for each item in `pool` build counter/neutral, cache
    resid_post[fit_layer][-1] for both, compute M_ctr / M_neu (first-token C-vs-W* margin). Returns a list
    of ctx dicts keyed by the item's POOL INDEX (so the same item lines up across models for the shared
    intersection). The |gap| >= MIN_EFFECT_NET gate is applied here. Same gate and metric as
    cave_direction_heldout._collect_ctxs / headset_direction._dir_pass."""
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    ctxs = []
    for idx, it in enumerate(pool):
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
            M_ctr = float(_logp_diff(model.run_with_hooks(counter, fwd_hooks=[(_rname(fit_layer), grab_c)]), cid, aid))
            M_neu = float(_logp_diff(model.run_with_hooks(neutral, fwd_hooks=[(_rname(fit_layer), grab_n)]), cid, aid))
        gap = M_neu - M_ctr
        if abs(gap) < MIN_EFFECT_NET:
            continue
        ctxs.append({"idx": idx, "counter": counter, "neutral": neutral, "cid": cid, "aid": aid,
                     "rc": rc[fit_layer].to(device), "rn": rn[fit_layer].to(device),
                     "M_ctr": M_ctr, "M_neu": M_neu, "gap": gap})
        print(f"  [{'it' if is_chat else 'base'} L{fit_layer}] gap={gap:+.2f} q={q[:40]!r}", flush=True)
    return ctxs


def _make_M_fn(model, fit_layer, _logp_diff):
    """Real ablated-forward M_fn(item, shift, direction): on the item's counter prompt, add
    (shift * direction) to resid_post[fit_layer][-1] and read the first-token C-vs-W* margin. This is the
    same necessity ablation operator as cave_direction_heldout._make_M_fn / headset_direction."""
    def M_fn(it, shift, direction):
        def ab(r, hook, direction=direction, shift=shift):
            r[0, -1] = r[0, -1] + (shift * direction).to(r.dtype); return r
        with torch.no_grad():
            return float(_logp_diff(model.run_with_hooks(it["counter"], fwd_hooks=[(_rname(fit_layer), ab)]),
                                    it["cid"], it["aid"]))
    return M_fn


def _collect_all_layers(name, is_chat, device, pool, refs):
    """Load one model and collect its per-layer ctx caches (one _collect_ctxs per FIT_LAYER). Returns
    {L: [ctx,...]} keyed by pool index inside each ctx. The model is freed before returning (residuals
    cached on `device`)."""
    from transformer_lens import HookedTransformer
    (_ITEMS, ITEMS_WIDE, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET) = refs
    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    by_layer = {}
    for L in FIT_LAYERS:
        ctxs = _collect_ctxs(model, device, is_chat, L, pool, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET)
        by_layer[L] = ctxs
        print(f"  [L{L}] {len(ctxs)} qualifying items", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return by_layer


def _host_pass(host_name, host_chat, device, donor_by_layer, refs):
    """HOST pass for one ordered (donor, host) pair. Reload the host, re-collect its residual cache per
    layer, and at each layer measure: WITHIN (host held-out), CROSS_DONOR_PROJN + CROSS_HOST_PROJN (donor's
    full-pool direction applied to host items), MATCHED_ITEM_* (donor/host directions fit on the SHARED
    items' train fold, applied to the host's shared-item test fold), and the random floor.

    `donor_by_layer` is the donor model's per-layer residual cache (from _collect_all_layers). The shared
    intersection is computed per layer from the items present (qualifying) in both caches at that layer."""
    from transformer_lens import HookedTransformer
    (_ITEMS, ITEMS_WIDE, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET) = refs
    print(f"[load] {host_name} (host pass) on {device} (chat={host_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(host_name, dtype=torch.bfloat16, device=device)
    model.eval()
    layers = {}
    for L in FIT_LAYERS:
        host_ctxs = _collect_ctxs(model, device, host_chat, L, ITEMS_WIDE, PUSH, NEUTRAL,
                                  _helpers, _logp_diff, MIN_EFFECT_NET)
        donor_ctxs = donor_by_layer.get(L, [])
        if len(host_ctxs) < 3 or len(donor_ctxs) < 3:
            print(f"  [L{L}] too few host/donor items ({len(host_ctxs)}/{len(donor_ctxs)}); skipping", flush=True)
            continue
        M_fn = _make_M_fn(model, L, _logp_diff)

        # WITHIN (host held-out reference)
        within_mean, within_fr = _within_heldout_nec(host_ctxs, M_fn)

        # CROSS (donor full-pool direction applied to all host items), both targets
        (cd_mean, cd_fr), (ch_mean, ch_fr) = _cross_nec_both(donor_ctxs, host_ctxs, M_fn)

        # random floor on the host items
        rand = _rand_nec(host_ctxs, M_fn, device)

        # MATCHED-ITEM: shared intersection (qualifying in BOTH caches at this layer), held-out split.
        donor_by_idx = {c["idx"]: c for c in donor_ctxs}
        host_by_idx = {c["idx"]: c for c in host_ctxs}
        shared = sorted(set(donor_by_idx) & set(host_by_idx))
        mi_cd_mean = mi_ch_mean = mi_within_mean = None
        mi_cd_fr = mi_ch_fr = []
        n_shared = len(shared)
        if n_shared >= 3:
            d_shared = [donor_by_idx[i] for i in shared]
            h_shared = [host_by_idx[i] for i in shared]
            mi_cd_pool, mi_ch_pool, mi_within_pool = [], [], []
            for train, test in kfold_splits(n_shared, N_FOLDS):
                d_train = [d_shared[i] for i in train]
                h_test = [h_shared[i] for i in test]
                (cdm, cdf), (chm, chf) = _matched_cross_nec(d_train, h_test, M_fn)
                mi_cd_pool += cdf
                mi_ch_pool += chf
                # within reference on the SAME shared items: host direction fit on the host shared TRAIN
                # fold, ablated on the host shared TEST fold (toward the train fold's own neutral mean).
                h_train = [h_shared[i] for i in train]
                hfit = fit_direction([c["rc"] for c in h_train], [c["rn"] for c in h_train], list(range(len(h_train))))
                mi_within_pool += _necessity_items(h_test, hfit["u"], hfit["proj_n"], M_fn)
            mi_cd_mean = statistics.mean(mi_cd_pool) if mi_cd_pool else None
            mi_ch_mean = statistics.mean(mi_ch_pool) if mi_ch_pool else None
            mi_within_mean = statistics.mean(mi_within_pool) if mi_within_pool else None
            mi_cd_fr, mi_ch_fr = mi_cd_pool, mi_ch_pool

        layers[L] = {
            "n_host": len(host_ctxs), "n_donor": len(donor_ctxs), "n_shared": n_shared,
            "within_nec": round(within_mean, 4) if within_mean is not None else None,
            "within_ci": bootstrap_ci(within_fr),
            "cross_donor_projn_nec": round(cd_mean, 4) if cd_mean is not None else None,
            "cross_donor_projn_ci": bootstrap_ci(cd_fr),
            "cross_host_projn_nec": round(ch_mean, 4) if ch_mean is not None else None,
            "cross_host_projn_ci": bootstrap_ci(ch_fr),
            "matched_within_nec": round(mi_within_mean, 4) if mi_within_mean is not None else None,
            "matched_item_donor_projn_nec": round(mi_cd_mean, 4) if mi_cd_mean is not None else None,
            "matched_item_donor_projn_ci": bootstrap_ci(mi_cd_fr),
            "matched_item_host_projn_nec": round(mi_ch_mean, 4) if mi_ch_mean is not None else None,
            "matched_item_host_projn_ci": bootstrap_ci(mi_ch_fr),
            "rand_nec": round(rand, 4) if rand is not None else None,
        }
        print(f"  [L{L}] within={layers[L]['within_nec']} cross_donor={layers[L]['cross_donor_projn_nec']} "
              f"cross_host={layers[L]['cross_host_projn_nec']} mi_within={layers[L]['matched_within_nec']} "
              f"mi_donor={layers[L]['matched_item_donor_projn_nec']} "
              f"mi_host={layers[L]['matched_item_host_projn_nec']} rand={layers[L]['rand_nec']} "
              f"(n_shared={n_shared})", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return layers


def _headline_layer(layers, key):
    """Layer with the largest value of layers[L][key] (None entries excluded). None if none available."""
    avail = [L for L in layers if layers[L].get(key) is not None]
    if not avail:
        return None
    return max(avail, key=lambda L: layers[L][key])


def _pair_decisions(host_layers, label):
    """At the headline layer (host layer with the largest WITHIN necessity), decide() each of the four
    variants. cross_donor_projn / cross_host_projn use the full-pool WITHIN reference; the matched_item_*
    variants use the matched-set WITHIN reference (apples-to-apples on the shared items). All layers are
    carried through in the returned dict so the JSON reports every layer, not just the headline."""
    L = _headline_layer(host_layers, "within_nec")
    if L is None:
        return {"pair": label, "headline_layer": None, "decisions": None, "layers": host_layers}
    r = host_layers[L]
    decs = {
        "cross_donor_projn": decide(r["cross_donor_projn_nec"], r["within_nec"]),
        "cross_host_projn": decide(r["cross_host_projn_nec"], r["within_nec"]),
        "matched_item_donor_projn": decide(r["matched_item_donor_projn_nec"], r["matched_within_nec"]),
        "matched_item_host_projn": decide(r["matched_item_host_projn_nec"], r["matched_within_nec"]),
    }
    return {"pair": label, "headline_layer": L, "decisions": decs, "layers": host_layers}


def run(name_base, name_it, tag):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # latent_verify/ for the repo imports
    from rlhf_differential import ITEMS, _helpers, _logp_diff, MIN_EFFECT_NET
    from misconception_pool import ITEMS_WIDE
    from job_truthful_flip import PUSH, NEUTRAL
    refs = (ITEMS, ITEMS_WIDE, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # donor residual caches (one pass per model, all layers)
    base_by_layer = _collect_all_layers(name_base, False, device, ITEMS_WIDE, refs)
    it_by_layer = _collect_all_layers(name_it, True, device, ITEMS_WIDE, refs)

    # base->it: donor=base, host=it ; it->base: donor=it, host=base
    base_to_it = _host_pass(name_it, True, device, base_by_layer, refs)
    it_to_base = _host_pass(name_base, False, device, it_by_layer, refs)

    pairs = {"base_to_it": _pair_decisions(base_to_it, "base->it"),
             "it_to_base": _pair_decisions(it_to_base, "it->base")}

    out = {"model_base": name_base, "model_it": name_it, "cue": "cave_direction_xregime_deconfound",
           "substrate": "misconception caving items (TruthfulQA-style), wide pool",
           "metric": ("M = logp(C) - logp(W*) first-token margin; necessity = "
                      "(M_ablate - M_counter)/(M_neutral - M_counter); shift moves the u-projection to a "
                      "variant-specific target proj_n"),
           "variants": {
               "within": "host held-out u_host; target = host proj_n",
               "cross_donor_projn": "donor u_donor; target = donor proj_n (existing cross path)",
               "cross_host_projn": "donor u_donor; target = host neutral mean along u_donor (scale-corrected)",
               "matched_item_donor_projn": "donor u_donor fit on shared train fold; target = donor proj_n; host shared test fold",
               "matched_item_host_projn": "donor u_donor fit on shared train fold; target = host neutral mean along u_donor; host shared test fold"},
           "fit_layers": FIT_LAYERS,
           "thresholds": {"dir_thr": DIR_THR, "xreg_ratio": XREG_RATIO, "n_folds": N_FOLDS, "seed": SEED,
                          "min_effect_net": MIN_EFFECT_NET},
           "headline_rule": "host layer with the largest within necessity",
           "pairs": pairs}
    Path("out").mkdir(exist_ok=True)
    fn = f"out/cave_direction_xregime_{tag}.json"
    Path(fn).write_text(json.dumps(out, indent=2, default=str))
    for pk, pv in pairs.items():
        if pv["decisions"] is None:
            print(f"[{pk}] no headline layer (insufficient qualifying items)")
            continue
        print(f"[{pk}] headline L{pv['headline_layer']}")
        for vk, vd in pv["decisions"].items():
            print(f"    {vk}: {vd['category']} (cross={vd['cross_nec']} within={vd['within_nec']} "
                  f"ratio={vd['ratio_cross_over_within']})")
    print(f"[done] wrote {fn}")


# --------------------------------------------------------------------------- selftest (model-free, CPU, no model load)
def _planted_ctxs(idxs, d, a, u_true, scale, offset, noise, seed):
    """Synthetic per-item residual cache for one regime. neutral ~ N(0,1) then affinely mapped by this
    regime's (scale, offset): rn = scale * z + offset ; counter = rn + a*u_true + noise. M is a LINEAR
    readout of the residual ALONG u_true: M = -(resid . u_true), so ablating the u_true-projection moves M
    deterministically and a direction orthogonal to u_true does ~nothing. Each ctx carries its POOL INDEX
    `idx` (for the shared intersection), rc/rn, M_ctr/M_neu/gap, and the planted u_true (read back by the
    synthetic M_fn). The (scale, offset) makes two regimes share the SAME u_true but live on DIFFERENT
    residual scales/offsets -- exactly the proj_n calibration artifact this control isolates. `offset` is a
    scalar or a [d] tensor (broadcast)."""
    g = torch.Generator().manual_seed(seed)
    u_true = u_true / (u_true.norm() + 1e-8)
    ctxs = []
    for idx in idxs:
        z = torch.randn(d, generator=g)
        rn = scale * z + offset
        rc = rn + a * u_true + noise * torch.randn(d, generator=g)
        M_ctr = -float(rc @ u_true)
        M_neu = -float(rn @ u_true)
        ctxs.append({"idx": idx, "rc": rc, "rn": rn, "M_ctr": M_ctr, "M_neu": M_neu,
                     "gap": M_neu - M_ctr, "_u_true": u_true})
    return ctxs


def _synthetic_M_fn(it, shift, direction):
    """Synthetic ablated forward: M_ablate = -((rc + shift*direction) . u_true). A shift along u_true moves
    M; a shift along a direction orthogonal to u_true does not (M reads only the u_true projection)."""
    r = it["rc"] + shift * direction
    return -float(r @ it["_u_true"])


def selftest():
    torch.manual_seed(0)
    # d kept large so a random / orthogonal direction's overlap with the planted u_true is small
    # (E|cos| ~ 1/sqrt(d)), making the "random ~0" and "orthogonal cross ~0" floors robust to the draw.
    d, n, a = 256, 30, 3.0

    # ---------- splitting / intersection / fit primitives ----------
    sp = kfold_splits(20, 5)
    assert len(sp) == 5, sp
    for tr, te in sp:
        assert set(tr).isdisjoint(set(te)), (tr, te)            # train/test disjoint
    allte = sorted(i for _, te in sp for i in te)
    assert allte == list(range(20)), allte                      # test folds partition the items
    # shared_items: |gap| gate in BOTH, present in both
    bg = {0: 0.6, 1: -1.0, 2: 0.8, 3: 0.4, 4: 1.0, 7: 0.9}
    ig = {0: 0.7, 1: 0.9, 2: 0.8, 3: 2.0, 5: 2.0, 7: -0.6}
    sh = shared_items(bg, ig, min_eff=0.5)
    # 0: both>=.5 ok; 1: |−1.0| & 0.9 ok; 2: ok; 3: base 0.4<.5 drop; 4/5: not in both; 7: 0.9 & |−0.6| ok
    assert sh == [0, 1, 2, 7], sh
    sh_split = kfold_splits(len(sh), N_FOLDS)
    for tr, te in sh_split:
        assert set(tr).isdisjoint(set(te)), (tr, te)
    print(f"[selftest] kfold disjoint+partition, shared intersection={sh} (in-both + |gap|>=min_eff), "
          f"matched split disjoint OK")

    # ---------- (a) SAME u_true, DIFFERENT residual scale/offset -> proj_n scale artifact ----------
    # Both regimes share u_true. The host residuals are scaled x2 AND shifted by (-a) ALONG u_true: this
    # makes the host's true neutral-mean projection sit a full cave (-a) below the donor's (~0), so the
    # DONOR proj_n target is off-distribution for the host. cross_HOST_projn (target re-anchored to the
    # host's own neutral mean along u_donor) recovers SHARED_DIRECTION; cross_DONOR_projn (donor's neutral
    # mean) collapses -- the necessity drop is the proj_n scale artifact, NOT a direction difference.
    u_true = torch.randn(d, generator=torch.Generator().manual_seed(SEED + 101))
    u_true = u_true / u_true.norm()
    host_offset = (-a) * u_true                                  # deterministic shift along u_true
    idxs = list(range(n))
    donor = _planted_ctxs(idxs, d, a=a, u_true=u_true, scale=1.0, offset=0.0, noise=0.25, seed=1)
    host = _planted_ctxs(idxs, d, a=a, u_true=u_true, scale=2.0, offset=host_offset, noise=0.25, seed=2)

    within_mean, _ = _within_heldout_nec(host, _synthetic_M_fn)
    (cd_mean, _), (ch_mean, _) = _cross_nec_both(donor, host, _synthetic_M_fn)
    rand = _rand_nec(host, _synthetic_M_fn, "cpu")
    assert within_mean > 0.8, within_mean
    assert abs(rand) < 0.1, f"random floor must be ~0: {rand}"
    # host-projn cross: SHARED_DIRECTION (the direction transfers once the target is on the host scale)
    dec_host = decide(ch_mean, within_mean)
    assert dec_host["category"] == "SHARED_DIRECTION", (ch_mean, within_mean, dec_host)
    # donor-projn cross is DEPRESSED below it (the scale artifact): both the raw necessity and the ratio
    assert cd_mean < ch_mean - 0.3, f"donor-projn should be depressed below host-projn: {cd_mean} vs {ch_mean}"
    dec_donor = decide(cd_mean, within_mean)
    assert dec_donor["ratio_cross_over_within"] < dec_host["ratio_cross_over_within"], (dec_donor, dec_host)
    assert dec_donor["category"] == "REGIME_SPECIFIC", (cd_mean, within_mean, dec_donor)
    print(f"[selftest] (a) within={within_mean:.3f} cross_donor_projn={cd_mean:.3f} "
          f"cross_host_projn={ch_mean:.3f} rand={rand:.3f} -> host_projn SHARED, donor_projn depressed to "
          f"REGIME_SPECIFIC (ratios {dec_donor['ratio_cross_over_within']:.2f} < {dec_host['ratio_cross_over_within']:.2f})")

    # ---------- (b) genuinely ORTHOGONAL directions across regimes ----------
    u_orth = torch.randn(d, generator=torch.Generator().manual_seed(SEED + 303))
    u_orth = u_orth - (u_orth @ u_true) * u_true                # strip the u_true component
    u_orth = u_orth / u_orth.norm()
    donor_o = _planted_ctxs(idxs, d, a=a, u_true=u_orth, scale=1.0, offset=0.0, noise=0.25, seed=3)
    (cd_o, _), (ch_o, _) = _cross_nec_both(donor_o, host, _synthetic_M_fn)
    assert abs(cd_o) < 0.2 and abs(ch_o) < 0.2, f"orthogonal cross should be ~0 (both variants): {cd_o}, {ch_o}"
    assert decide(cd_o, within_mean)["category"] == "REGIME_SPECIFIC", (cd_o, within_mean)
    assert decide(ch_o, within_mean)["category"] == "REGIME_SPECIFIC", (ch_o, within_mean)
    print(f"[selftest] (b) orthogonal cross: donor_projn={cd_o:.3f} host_projn={ch_o:.3f} -> both REGIME_SPECIFIC")

    # ---------- (c) matched-item: disjoint split + shared-only selection ----------
    # donor qualifies on idx {0..n-1}\{5}; host qualifies on {0..n-1}\{9}; the shared intersection excludes
    # BOTH 5 and 9 (only items present in both caches). Same u_true + scale artifact as (a).
    donor_idx = [i for i in range(n) if i != 5]
    host_idx = [i for i in range(n) if i != 9]
    donor_m = _planted_ctxs(donor_idx, d, a=a, u_true=u_true, scale=1.0, offset=0.0, noise=0.25, seed=4)
    host_m = _planted_ctxs(host_idx, d, a=a, u_true=u_true, scale=2.0, offset=host_offset, noise=0.25, seed=5)
    donor_by_idx = {c["idx"]: c for c in donor_m}
    host_by_idx = {c["idx"]: c for c in host_m}
    shared = sorted(set(donor_by_idx) & set(host_by_idx))
    assert 5 not in shared and 9 not in shared, shared
    assert shared == [i for i in range(n) if i not in (5, 9)], shared
    # one matched cross fold, both targets; held-out (train/test disjoint over the shared keys)
    d_shared = [donor_by_idx[i] for i in shared]
    h_shared = [host_by_idx[i] for i in shared]
    sp_m = kfold_splits(len(shared), N_FOLDS)
    tr0, te0 = sp_m[0]
    assert set(tr0).isdisjoint(set(te0))
    (mi_cd, _), (mi_ch, _) = _matched_cross_nec([d_shared[i] for i in tr0], [h_shared[i] for i in te0],
                                                _synthetic_M_fn)
    # shared u_true + the scale artifact -> matched host_projn recovers more than matched donor_projn
    assert mi_ch > mi_cd, f"matched host_projn should beat donor_projn under the scale artifact: {mi_ch} vs {mi_cd}"
    print(f"[selftest] (c) shared={len(shared)} items (5,9 excluded); matched cross donor_projn={mi_cd:.3f} "
          f"host_projn={mi_ch:.3f} (host_projn higher under the scale artifact)")

    # ---------- (d) decide() at the thresholds ----------
    # SHARED_DIRECTION: cross >= DIR_THR and ratio >= XREG_RATIO
    s1 = decide(0.45, 0.50)        # ratio 0.90 >= 0.6, cross >= 0.20
    assert s1["category"] == "SHARED_DIRECTION" and s1["shared_direction"], s1
    s2 = decide(0.30, 0.50)        # ratio 0.60 == 0.6 boundary (>=), cross >= 0.20
    assert s2["category"] == "SHARED_DIRECTION", s2
    s3 = decide(0.20, 0.30)        # cross 0.20 == DIR_THR (>=), ratio 0.667 >= 0.6
    assert s3["category"] == "SHARED_DIRECTION", s3
    # REGIME_SPECIFIC: cross below DIR_THR
    r1 = decide(0.10, 0.50)
    assert r1["category"] == "REGIME_SPECIFIC", r1
    # REGIME_SPECIFIC: ratio below XREG_RATIO even though cross clears DIR_THR
    r2 = decide(0.25, 0.80)        # ratio 0.31 < 0.6
    assert r2["category"] == "REGIME_SPECIFIC", r2
    # REGIME_SPECIFIC: within ~0 -> ratio undefined -> not shared
    r3 = decide(0.45, 0.0)
    assert r3["category"] == "REGIME_SPECIFIC" and r3["ratio_cross_over_within"] is None, r3
    # None cross
    r4 = decide(None, 0.50)
    assert r4["category"] == "REGIME_SPECIFIC", r4
    print("[selftest] (d) decide(): SHARED_DIRECTION (x3 incl boundaries) / REGIME_SPECIFIC (x4) all fire")

    # end-to-end _pair_decisions on a synthetic per-layer dict (all layers carried through; headline = max within)
    fake = {
        24: {"within_nec": 0.40, "cross_donor_projn_nec": 0.10, "cross_host_projn_nec": 0.35,
             "matched_within_nec": 0.42, "matched_item_donor_projn_nec": 0.12, "matched_item_host_projn_nec": 0.36},
        28: {"within_nec": 0.55, "cross_donor_projn_nec": 0.15, "cross_host_projn_nec": 0.45,
             "matched_within_nec": 0.50, "matched_item_donor_projn_nec": 0.18, "matched_item_host_projn_nec": 0.40},
    }
    pd = _pair_decisions(fake, "base->it")
    assert pd["headline_layer"] == 28, pd                       # max within is at L28
    assert pd["decisions"]["cross_host_projn"]["category"] == "SHARED_DIRECTION", pd
    assert pd["decisions"]["cross_donor_projn"]["category"] == "REGIME_SPECIFIC", pd
    assert pd["decisions"]["matched_item_host_projn"]["category"] == "SHARED_DIRECTION", pd
    assert pd["decisions"]["matched_item_donor_projn"]["category"] == "REGIME_SPECIFIC", pd
    assert set(pd["layers"].keys()) == {24, 28}, "all layers must be carried through"
    print("[selftest] _pair_decisions headline=L28; host_projn SHARED, donor_projn REGIME_SPECIFIC; all layers carried")

    # bootstrap CI mechanics
    pos = bootstrap_ci([0.5, 0.52, 0.48, 0.51, 0.49])
    assert pos["lo"] > 0 and pos["n"] == 5, pos
    assert bootstrap_ci([])["mean"] is None
    print("[selftest] bootstrap CI mechanics OK")
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
