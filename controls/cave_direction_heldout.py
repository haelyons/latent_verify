"""HELD-OUT + CROSS-REGIME generalization control for the rank-1 "cave/defer" DIRECTION.

WHY. headset_direction.py fits the diff-of-means cave direction u and measures its NECESSITY (ablate the
u-projection on counter, move it to the neutral mean, read M recovery) IN-SAMPLE: u is fit on exactly the
items it is then ablated on. A diff-of-means vector fit on n items has n degrees of freedom and will
recover some of the cave on those same items even with no shared low-rank structure -- the in-sample
necessity number alone cannot separate "a real generalizing direction" from "overfit to the fit items".
This control measures, with NO new mechanism, whether the necessity SURVIVES the standard generalization
splits, on TWO models independently:

  1. HELD-OUT necessity. Fit u on a TRAIN fold of qualifying caving items; measure necessity on the
     DISJOINT TEST fold. k-fold and leave-one-out passes. In-sample vs held-out reported side by side,
     with a paired bootstrap CI over the held-out per-item recoveries.
  2. CROSS-REGIME transfer. Fit u_base on base items and u_it on -it items; apply u_base inside -it (on
     held-out -it items) and u_it inside base (on held-out base items). within-regime vs cross-regime
     necessity + the ratio cross/within, per cross pair (base->it, it->base).
  3. LABEL-PERMUTED null. Refit diff-of-means after shuffling which residual of each item is "counter"
     vs "neutral", measure its IN-SAMPLE necessity, averaged over N_PERM permutations (fixed seed). This
     bounds the necessity reachable by the fitting degrees of freedom alone (no real condition contrast).
  4. RANDOM-DIRECTION necessity (matched magnitude), exactly as headset_direction.py.

Metric, items, push/NEUTRAL turns, the diff-of-means construction, the projection-edit ablation, the
random matched-magnitude control, and the FIT_LAYERS sweep are all reused from the reference modules
(rlhf_differential / job_truthful_flip / headset_direction); the only new logic is the train/test
splitting, the cross-regime application, the label permutation, and the two generalization decisions --
all pure and covered by the model-free --selftest.

NEUTRAL DECISION (module-constant thresholds; numbers + categories only, no hypothesis, no statement
about which model should win):
  HELD_OUT_DIRECTION iff held-out necessity >= DIR_THR AND held-out necessity lies within the in-sample
      bootstrap CI AND (held-out necessity - label_permuted necessity) >= MARGIN; else IN_SAMPLE_ONLY
      (held-out collapses while in-sample clears DIR_THR) or NO_DIRECTION (in-sample also < DIR_THR).
  CROSS_REGIME iff cross-regime necessity >= DIR_THR AND (cross/within) >= XREG_RATIO; else
      REGIME_SPECIFIC. Reported per cross pair (base->it, it->base).

Forward-only (diff-of-means + projection edits; no backward) -> fits the 40GB A100. Reuses the verified
primitives; runs each model once for its fits/held-out passes, with the cross-regime application using the
directions fit on each model's own residual cache.

  python controls/cave_direction_heldout.py --selftest
  python controls/cave_direction_heldout.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

# Repo-internal imports (item pool, turn builders, metric, fit machinery) are deferred into run() so
# --selftest runs standalone from controls/ on CPU with NO model load and nothing else on sys.path; on
# the box the reference files are scp'd flat into latent_verify/ where these resolve.

# --------------------------------------------------------------------------- pre-registered thresholds
FIT_LAYERS = [24, 28, 32, 36]   # same sweep as headset_direction.py (set heads span L21-L34)
DIR_THR = 0.20                  # frac of the cave the direction must mediate to count
BASE_FLOOR = 0.05               # random-direction necessity must sit below this to be "clean"
MARGIN = 0.05                   # held-out necessity must beat the label-permuted null by this much
XREG_RATIO = 0.6                # cross-regime necessity / within-regime necessity must reach this
N_PERM = 20                     # label-permutation repeats for the null (fixed seed)
N_FOLDS = 5                     # k-fold split count (capped at n_items)
SEED = 0
N_BOOT = 2000
CI_LO, CI_HI = 2.5, 97.5        # 95 percent percentile interval


# --------------------------------------------------------------------------- pure splitting / fitting
def kfold_splits(n, k, seed=SEED):
    """k near-equal disjoint test folds over shuffled range(n); each split = (train_idx, test_idx).
    Pure: deterministic via random.Random(seed). k is capped at n (>=2 folds need n>=2)."""
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


def loo_splits(n):
    """Leave-one-out: n splits, each holding out exactly one item. Pure."""
    if n < 2:
        return []
    return [([j for j in range(n) if j != i], [i]) for i in range(n)]


def fit_direction(rc_list, rn_list, idxs):
    """Diff-of-means cave direction over the items in `idxs`, with the neutral mean projection.
    u = normalize(mean_i(rc_i - rn_i)); proj_n = mean_i(rn_i . u). Inputs are aligned lists of 1-D
    residual tensors (counter, neutral) per item. Pure (tensors in, dict out). Mirrors the construction
    in headset_direction._dir_pass / matched_item_deconfound._effect_pass exactly."""
    D = torch.stack([rc_list[i] - rn_list[i] for i in idxs])
    cave = D.mean(0)
    u = cave / (cave.norm() + 1e-8)
    proj_n = statistics.mean(float(rn_list[i] @ u) for i in idxs)
    return {"u": u, "proj_n": proj_n}


def random_direction(shape, dtype, gen):
    """Random unit direction (cpu generator -> caller moves it to device). Pure given the generator."""
    rnd = torch.randn(shape, generator=gen).to(dtype)
    return rnd / (rnd.norm() + 1e-8)


def permute_labels(n_items, seed):
    """Per-item independent bernoulli swap of (counter, neutral) labels: returns a list of bools, True =>
    swap this item's two residuals before re-fitting. Pure: deterministic via random.Random(seed). The
    diff-of-means of a swapped item flips sign, so the fitted vector loses its shared condition contrast
    and keeps only fit-noise structure -- the necessity it then recovers is the degrees-of-freedom floor."""
    import random as _r
    rng = _r.Random(seed)
    return [rng.random() < 0.5 for _ in range(n_items)]


# --------------------------------------------------------------------------- pure stats / decisions
def paired_bootstrap_ci(values, seed=SEED, n_boot=N_BOOT, lo=CI_LO, hi=CI_HI):
    """Percentile bootstrap CI of the mean (matches matched_item_deconfound.bootstrap_ci). Pure."""
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


def in_ci(value, ci):
    """True iff value falls inside [ci.lo, ci.hi]. Pure. None ci/value -> False."""
    if value is None or ci is None or ci.get("lo") is None or ci.get("hi") is None:
        return False
    return ci["lo"] <= value <= ci["hi"]


def decide_heldout(in_sample, held_out, in_sample_ci, perm_nec, rand_nec,
                   dir_thr=DIR_THR, margin=MARGIN, base_floor=BASE_FLOOR):
    """Generalization decision for one model's direction. Pure over the measured numbers only.
      HELD_OUT_DIRECTION : held_out >= dir_thr AND held_out in the in-sample bootstrap CI AND
                           (held_out - perm_nec) >= margin (and the random control, if given, is clean).
      IN_SAMPLE_ONLY     : in_sample >= dir_thr but held_out fails the above (the direction does not
                           transfer off the items it was fit on).
      NO_DIRECTION       : in_sample < dir_thr (no in-sample direction to generalize)."""
    rand_clean = rand_nec is None or rand_nec < base_floor
    within_ci = in_ci(held_out, in_sample_ci)
    beats_perm = (perm_nec is None) or (held_out is not None and (held_out - perm_nec) >= margin)
    held_ok = (held_out is not None and held_out >= dir_thr and within_ci and beats_perm and rand_clean)
    in_ok = in_sample is not None and in_sample >= dir_thr
    if held_ok:
        cat = "HELD_OUT_DIRECTION"
        msg = (f"held-out necessity {held_out:.3f} >= {dir_thr}, within the in-sample CI "
               f"[{in_sample_ci['lo']},{in_sample_ci['hi']}], and beats the label-permuted null "
               f"({perm_nec}) by >= {margin} -- the cave direction generalizes off its fit items.")
    elif in_ok:
        cat = "IN_SAMPLE_ONLY"
        why = []
        if held_out is None or held_out < dir_thr:
            why.append(f"held-out {held_out if held_out is None else round(held_out, 3)} < {dir_thr}")
        if not within_ci:
            why.append("held-out outside the in-sample CI")
        if not beats_perm:
            why.append(f"held-out does not beat the permuted null by {margin}")
        if not rand_clean:
            why.append(f"random control not clean (rand_nec {rand_nec})")
        msg = (f"in-sample necessity {in_sample:.3f} >= {dir_thr} but held-out collapses ("
               + "; ".join(why) + ") -- the in-sample number reflects fit degrees of freedom, not a "
               "generalizing direction.")
    else:
        cat = "NO_DIRECTION"
        msg = (f"in-sample necessity {in_sample if in_sample is None else round(in_sample, 3)} < {dir_thr} "
               f"-- no in-sample direction to generalize.")
    return {"category": cat,
            "held_out_direction": cat == "HELD_OUT_DIRECTION",
            "in_sample_nec": (round(in_sample, 4) if in_sample is not None else None),
            "held_out_nec": (round(held_out, 4) if held_out is not None else None),
            "in_sample_ci": in_sample_ci,
            "held_out_within_ci": within_ci,
            "label_permuted_nec": (round(perm_nec, 4) if perm_nec is not None else None),
            "held_minus_perm": (round(held_out - perm_nec, 4) if (held_out is not None and perm_nec is not None) else None),
            "rand_nec": (round(rand_nec, 4) if rand_nec is not None else None),
            "rand_clean": rand_clean, "msg": msg}


def decide_xreg(cross_nec, within_nec, dir_thr=DIR_THR, xreg_ratio=XREG_RATIO):
    """Cross-regime decision for one cross pair (e.g. base->it). Pure over the measured numbers.
      CROSS_REGIME    : cross_nec >= dir_thr AND (cross/within) >= xreg_ratio (the direction fit in one
                        regime mediates the cave in the other to >= xreg_ratio of within-regime strength).
      REGIME_SPECIFIC : otherwise (cross direction does not carry across regimes)."""
    ratio = (cross_nec / within_nec) if (within_nec and abs(within_nec) > 1e-6) else None
    transfers = (cross_nec is not None and cross_nec >= dir_thr and
                 ratio is not None and ratio >= xreg_ratio)
    if transfers:
        cat = "CROSS_REGIME"
        msg = (f"cross-regime necessity {cross_nec:.3f} >= {dir_thr} and cross/within "
               f"{ratio:.2f} >= {xreg_ratio} -- the direction carries across regimes.")
    else:
        cat = "REGIME_SPECIFIC"
        rtxt = "n/a" if ratio is None else f"{ratio:.2f}"
        msg = (f"cross-regime necessity {cross_nec if cross_nec is None else round(cross_nec, 3)} / "
               f"cross-within ratio {rtxt} below ({dir_thr}, {xreg_ratio}) -- the direction is "
               "regime-specific.")
    return {"category": cat, "cross_regime": cat == "CROSS_REGIME",
            "cross_nec": (round(cross_nec, 4) if cross_nec is not None else None),
            "within_nec": (round(within_nec, 4) if within_nec is not None else None),
            "ratio_cross_over_within": (round(ratio, 4) if ratio is not None else None), "msg": msg}


# --------------------------------------------------------------------------- necessity engine (real + synthetic)
def _necessity_items(eval_items, u, proj_n, M_fn):
    """For each item in eval_items, ablate the u-projection on its counter residual TO proj_n and read M
    recovery frac = (M_ablate - M_counter) / (M_neutral - M_counter). Returns the per-item frac list.
    `M_fn(item, shift, direction) -> M_ablate` runs the actual ablated forward pass (real model) OR the
    planted synthetic margin (selftest); the rest of the formula is identical to headset_direction.

    item must carry: rc (counter resid, 1-D tensor), M_ctr, gap (= M_neu - M_ctr, the signed cave)."""
    out = []
    for it in eval_items:
        shift = proj_n - float(it["rc"] @ u)
        M_ab = M_fn(it, shift, u)
        out.append((M_ab - it["M_ctr"]) / it["gap"])
    return out


def _split_nec(ctxs, splits, M_fn):
    """Held-out necessity over a list of (train_idx, test_idx) splits: fit u on train residuals, evaluate
    necessity on the disjoint test items, pool all test per-item fracs. Returns (mean, per_item_list).
    Pure given M_fn + the residual cache; shared by the real run and the selftest."""
    rc_list = [c["rc"] for c in ctxs]
    rn_list = [c["rn"] for c in ctxs]
    pooled = []
    for train, test in splits:
        fit = fit_direction(rc_list, rn_list, train)
        pooled += _necessity_items([ctxs[i] for i in test], fit["u"], fit["proj_n"], M_fn)
    mean = statistics.mean(pooled) if pooled else None
    return mean, pooled


def _insample_nec(ctxs, M_fn):
    """In-sample necessity: fit on ALL items, ablate on the SAME items. Returns (mean, per_item_list)."""
    rc_list = [c["rc"] for c in ctxs]
    rn_list = [c["rn"] for c in ctxs]
    fit = fit_direction(rc_list, rn_list, list(range(len(ctxs))))
    fr = _necessity_items(ctxs, fit["u"], fit["proj_n"], M_fn)
    return (statistics.mean(fr) if fr else None), fr


def _perm_nec(ctxs, M_fn, n_perm=N_PERM, seed=SEED):
    """Label-permuted in-sample necessity averaged over n_perm independent permutations. Each permutation
    swaps a random subset of items' (counter, neutral) residuals before the diff-of-means fit, then
    measures in-sample necessity against the (unswapped) real counter/neutral evaluation."""
    rc_list = [c["rc"] for c in ctxs]
    rn_list = [c["rn"] for c in ctxs]
    n = len(ctxs)
    means = []
    for p in range(n_perm):
        swap = permute_labels(n, seed + p)
        prc = [(rn_list[i] if swap[i] else rc_list[i]) for i in range(n)]
        prn = [(rc_list[i] if swap[i] else rn_list[i]) for i in range(n)]
        D = torch.stack([prc[i] - prn[i] for i in range(n)])
        cave = D.mean(0)
        u = cave / (cave.norm() + 1e-8)
        proj_n = statistics.mean(float(prn[i] @ u) for i in range(n))   # the permuted fit's own neutral mean
        fr = _necessity_items(ctxs, u, proj_n, M_fn)                    # eval on the REAL counter/neutral
        if fr:
            means.append(statistics.mean(fr))
    return statistics.mean(means) if means else None


def _rand_nec(ctxs, M_fn, device, seed=SEED):
    """Random-direction necessity, matched-magnitude, exactly as headset_direction: fit a random unit
    direction ur, shift each counter's ur-projection to the random direction's neutral mean."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = ctxs[0]["rc"].shape[0]
    ur = random_direction((d,), ctxs[0]["rc"].dtype, g).to(device)
    prn = statistics.mean(float(c["rn"] @ ur) for c in ctxs)
    fr = []
    for c in ctxs:
        shift = prn - float(c["rc"] @ ur)
        M_ab = M_fn(c, shift, ur)
        fr.append((M_ab - c["M_ctr"]) / c["gap"])
    return statistics.mean(fr) if fr else None


# --------------------------------------------------------------------------- real model passes
def _rname(L):
    return f"blocks.{L}.hook_resid_post"


def _collect_ctxs(model, device, is_chat, fit_layer, pool, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET):
    """Gap-pass: for each item build counter/neutral, cache resid_post[fit_layer][-1] for both, compute
    M_ctr / M_neu (first-token C-vs-W* margin), keep items with |gap| >= MIN_EFFECT_NET. Same gate and
    metric as headset_direction._dir_pass; only one fit layer at a time to keep the cache small."""
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    ctxs = []
    for it in pool:
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
        ctxs.append({"counter": counter, "neutral": neutral, "cid": cid, "aid": aid,
                     "rc": rc[fit_layer].to(device), "rn": rn[fit_layer].to(device),
                     "M_ctr": M_ctr, "M_neu": M_neu, "gap": gap})
        print(f"  [{'it' if is_chat else 'base'} L{fit_layer}] gap={gap:+.2f} q={q[:40]!r}", flush=True)
    return ctxs


def _make_M_fn(model, fit_layer, _logp_diff):
    """Real ablated-forward M_fn(item, shift, direction): on the item's counter prompt, add
    (shift * direction) to resid_post[fit_layer][-1] and read the first-token C-vs-W* margin. This is
    headset_direction's necessity ablation operator (move the projection to the target mean)."""
    def M_fn(it, shift, direction):
        def ab(r, hook, direction=direction, shift=shift):
            r[0, -1] = r[0, -1] + (shift * direction).to(r.dtype); return r
        with torch.no_grad():
            return float(_logp_diff(model.run_with_hooks(it["counter"], fwd_hooks=[(_rname(fit_layer), ab)]),
                                    it["cid"], it["aid"]))
    return M_fn


def _model_pass(name, is_chat, device, refs):
    """One model: for each fit layer, collect ctxs and measure in-sample / k-fold / LOO held-out /
    label-permuted / random necessity. Returns per-layer results + the cached ctxs (for cross-regime)."""
    from transformer_lens import HookedTransformer
    (_ITEMS, ITEMS_WIDE, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET) = refs
    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    layers = {}
    ctxs_by_layer = {}
    for L in FIT_LAYERS:
        ctxs = _collect_ctxs(model, device, is_chat, L, ITEMS_WIDE, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET)
        if len(ctxs) < 3:
            print(f"  [L{L}] too few qualifying items ({len(ctxs)}); skipping", flush=True)
            continue
        ctxs_by_layer[L] = ctxs
        M_fn = _make_M_fn(model, L, _logp_diff)
        ins_mean, ins_per_item = _insample_nec(ctxs, M_fn)              # per-item fracs feed the bootstrap CI
        kf_mean, kf_fr = _split_nec(ctxs, kfold_splits(len(ctxs), N_FOLDS), M_fn)
        loo_mean, loo_fr = _split_nec(ctxs, loo_splits(len(ctxs)), M_fn)
        perm = _perm_nec(ctxs, M_fn)
        rand = _rand_nec(ctxs, M_fn, device)
        layers[L] = {
            "n_ok": len(ctxs),
            "in_sample_nec": round(ins_mean, 4) if ins_mean is not None else None,
            "in_sample_ci": paired_bootstrap_ci(ins_per_item),
            "kfold_heldout_nec": round(kf_mean, 4) if kf_mean is not None else None,
            "kfold_heldout_ci": paired_bootstrap_ci(kf_fr),
            "loo_heldout_nec": round(loo_mean, 4) if loo_mean is not None else None,
            "loo_heldout_ci": paired_bootstrap_ci(loo_fr),
            "label_permuted_nec": round(perm, 4) if perm is not None else None,
            "rand_nec": round(rand, 4) if rand is not None else None,
        }
        print(f"  [L{L}] in={layers[L]['in_sample_nec']} kf={layers[L]['kfold_heldout_nec']} "
              f"loo={layers[L]['loo_heldout_nec']} perm={layers[L]['label_permuted_nec']} "
              f"rand={layers[L]['rand_nec']}", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"layers": layers, "ctxs_by_layer": ctxs_by_layer}


def _cross_pass(name, is_chat, device, donor_ctxs_by_layer, refs):
    """CROSS-REGIME: reload `name` and, on its own qualifying items, measure necessity of a direction fit
    on the DONOR regime's residuals (held-out by construction -- the donor items are the OTHER model's
    items). within-regime = held-out necessity of `name`'s own direction (k-fold on its ctxs); cross-regime
    = necessity of the donor's all-item direction applied to `name`'s items. Per fit layer."""
    from transformer_lens import HookedTransformer
    (_ITEMS, ITEMS_WIDE, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET) = refs
    print(f"[load] {name} (cross-regime apply) on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    layers = {}
    for L in FIT_LAYERS:
        if L not in donor_ctxs_by_layer:
            continue
        ctxs = _collect_ctxs(model, device, is_chat, L, ITEMS_WIDE, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET)
        if len(ctxs) < 3:
            continue
        M_fn = _make_M_fn(model, L, _logp_diff)
        # within-regime (held-out, k-fold on this model's own items)
        within_mean, within_fr = _split_nec(ctxs, kfold_splits(len(ctxs), N_FOLDS), M_fn)
        # cross-regime: the donor model's all-item direction, evaluated on this model's items
        donor = donor_ctxs_by_layer[L]
        dfit = fit_direction([c["rc"] for c in donor], [c["rn"] for c in donor], list(range(len(donor))))
        cross_fr = _necessity_items(ctxs, dfit["u"], dfit["proj_n"], M_fn)
        cross_mean = statistics.mean(cross_fr) if cross_fr else None
        layers[L] = {"n_ok": len(ctxs),
                     "within_nec": round(within_mean, 4) if within_mean is not None else None,
                     "within_ci": paired_bootstrap_ci(within_fr),
                     "cross_nec": round(cross_mean, 4) if cross_mean is not None else None,
                     "cross_ci": paired_bootstrap_ci(cross_fr)}
        print(f"  [L{L}] within={layers[L]['within_nec']} cross={layers[L]['cross_nec']}", flush=True)
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


def run(name_base, name_it, tag):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # latent_verify/ for the repo imports
    from rlhf_differential import ITEMS, _helpers, _logp_diff, MIN_EFFECT_NET
    from misconception_pool import ITEMS_WIDE
    from job_truthful_flip import PUSH, NEUTRAL
    refs = (ITEMS, ITEMS_WIDE, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    it = _model_pass(name_it, True, device, refs)
    base = _model_pass(name_base, False, device, refs)

    # per-model held-out decision at each model's headline (best held-out) layer
    def model_decision(res):
        L = _headline_layer(res["layers"], "loo_heldout_nec") or _headline_layer(res["layers"], "in_sample_nec")
        if L is None:
            return None, None
        r = res["layers"][L]
        dec = decide_heldout(r["in_sample_nec"], r["loo_heldout_nec"], r["in_sample_ci"],
                             r["label_permuted_nec"], r["rand_nec"])
        return L, dec
    it_L, it_dec = model_decision(it)
    base_L, base_dec = model_decision(base)

    # cross-regime: u_base applied in -it (donor=base, host=it) and u_it applied in base (donor=it, host=base)
    base2it = _cross_pass(name_it, True, device, base["ctxs_by_layer"], refs)     # u_base in -it
    it2base = _cross_pass(name_base, False, device, it["ctxs_by_layer"], refs)    # u_it in base

    def xreg_decision(host_layers, label):
        L = _headline_layer(host_layers, "within_nec")
        if L is None:
            return {"pair": label, "headline_layer": None, "decision": None, "layers": host_layers}
        r = host_layers[L]
        return {"pair": label, "headline_layer": L,
                "decision": decide_xreg(r["cross_nec"], r["within_nec"]), "layers": host_layers}
    xreg = {"base_to_it": xreg_decision(base2it, "base->it"),
            "it_to_base": xreg_decision(it2base, "it->base")}

    out = {"model_base": name_base, "model_it": name_it, "cue": "cave_direction_heldout",
           "substrate": "misconception caving items (TruthfulQA-style), wide pool",
           "metric": "M = logp(C) - logp(W*) first-token margin; necessity = (M_ablate - M_counter)/(M_neutral - M_counter)",
           "fit_layers": FIT_LAYERS,
           "thresholds": {"dir_thr": DIR_THR, "base_floor": BASE_FLOOR, "margin": MARGIN,
                          "xreg_ratio": XREG_RATIO, "n_perm": N_PERM, "n_folds": N_FOLDS, "seed": SEED},
           "it": {"headline_layer": it_L, "layers": it["layers"], "heldout_decision": it_dec},
           "base": {"headline_layer": base_L, "layers": base["layers"], "heldout_decision": base_dec},
           "cross_regime": xreg}
    Path("out").mkdir(exist_ok=True)
    fn = f"out/cave_direction_heldout_{tag}.json"
    Path(fn).write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[it heldout]   {it_dec['category'] if it_dec else None} (L{it_L})")
    print(f"[base heldout] {base_dec['category'] if base_dec else None} (L{base_L})")
    print(f"[xreg base->it] {xreg['base_to_it']['decision']['category'] if xreg['base_to_it']['decision'] else None}")
    print(f"[xreg it->base] {xreg['it_to_base']['decision']['category'] if xreg['it_to_base']['decision'] else None}")
    print(f"[done] wrote {fn}")


# --------------------------------------------------------------------------- selftest (model-free, CPU, no model load)
def _planted_ctxs(n, d, a, u_true, noise, seed):
    """Synthetic per-item residual cache: neutral ~ N(0,1); counter = neutral + a*u_true + noise. M is a
    LINEAR readout of the residual ALONG u_true (M = -(resid . u_true)), so ablating the u_true-projection
    moves M deterministically and a direction orthogonal to u_true does ~nothing -- exactly the
    necessity / random-control contrast the real engine tests. Each item carries rc, rn, M_ctr, M_neu, gap,
    plus the planted u_true so the synthetic M_fn can read it back. Returns (ctxs, normalized u_true)."""
    g = torch.Generator().manual_seed(seed)
    u_true = u_true / (u_true.norm() + 1e-8)
    ctxs = []
    for _ in range(n):
        rn = torch.randn(d, generator=g)
        rc = rn + a * u_true + noise * torch.randn(d, generator=g)
        M_ctr = -float(rc @ u_true)
        M_neu = -float(rn @ u_true)
        ctxs.append({"rc": rc, "rn": rn, "M_ctr": M_ctr, "M_neu": M_neu, "gap": M_neu - M_ctr,
                     "_u_true": u_true})
    return ctxs, u_true


def _synthetic_M_fn(it, shift, direction):
    """Synthetic ablated forward: M_ablate = -((rc + shift*direction) . u_true). A shift along u_true moves
    M; a shift along a direction orthogonal to u_true does not (M reads only the u_true projection)."""
    r = it["rc"] + shift * direction
    return -float(r @ it["_u_true"])


def selftest():
    torch.manual_seed(0)
    # d kept large so a random / orthogonal direction's residual overlap with the planted u_true is small
    # (E|cos| ~ 1/sqrt(d)); this makes the "random ~0" and "orthogonal cross ~0" controls robust to the
    # particular draw, not the headline of the test (which is the high real necessity vs those floors).
    d, n = 256, 24

    # ---------- splitting primitives ----------
    sp = kfold_splits(20, 5)
    assert len(sp) == 5, sp
    for tr, te in sp:
        assert set(tr).isdisjoint(set(te)), (tr, te)            # train/test disjoint
    allte = sorted(i for _, te in sp for i in te)
    assert allte == list(range(20)), allte                      # test folds partition the items
    loo = loo_splits(6)
    assert len(loo) == 6 and all(len(te) == 1 for _, te in loo)
    assert loo[2][1] == [2] and 2 not in loo[2][0]
    print("[selftest] kfold disjoint+partition, LOO holds one out OK")

    # ---------- (i) diff-of-means recovers u_true; held-out necessity high; random ~0 ----------
    # draw u_true from a generator whose seed is DISTINCT from SEED (the random-direction seed), else the
    # matched-random control _rand_nec would draw the same vector as u_true and spuriously "recover" ~1.0
    u_true = torch.randn(d, generator=torch.Generator().manual_seed(SEED + 101))
    ctxs, ut = _planted_ctxs(n, d, a=3.0, u_true=u_true, noise=0.25, seed=1)
    fit = fit_direction([c["rc"] for c in ctxs], [c["rn"] for c in ctxs], list(range(n)))
    cos = float(torch.nn.functional.cosine_similarity(fit["u"], ut, dim=0))
    assert cos > 0.95, f"diff-of-means should recover u_true: cos={cos}"   # a=3,noise=.25,n=24,d=256 -> ~0.966; >>random ~1/sqrt(256)=0.06
    ins_mean, ins_fr = _insample_nec(ctxs, _synthetic_M_fn)
    kf_mean, kf_fr = _split_nec(ctxs, kfold_splits(n, N_FOLDS), _synthetic_M_fn)
    loo_mean, _ = _split_nec(ctxs, loo_splits(n), _synthetic_M_fn)
    assert ins_mean > 0.8 and kf_mean > 0.8 and loo_mean > 0.8, (ins_mean, kf_mean, loo_mean)
    rand = _rand_nec(ctxs, _synthetic_M_fn, "cpu")
    assert abs(rand) < BASE_FLOOR, f"random matched-magnitude direction must not recover: {rand}"
    print(f"[selftest] (i) cos(u,u_true)={cos:.3f} in={ins_mean:.3f} kf={kf_mean:.3f} loo={loo_mean:.3f} rand={rand:.3f}")

    # ---------- (ii) label-permuted necessity ~0 ----------
    perm = _perm_nec(ctxs, _synthetic_M_fn)
    assert abs(perm) < 0.2, f"label-permuted necessity should collapse toward 0: {perm}"
    assert (loo_mean - perm) >= MARGIN, (loo_mean, perm)
    print(f"[selftest] (ii) label-permuted necessity={perm:.3f} (real held-out {loo_mean:.3f} beats it by >= {MARGIN})")

    # ---------- (iii) cross-regime: shared u_true high; orthogonal directions low ----------
    host, _ = _planted_ctxs(n, d, a=3.0, u_true=u_true, noise=0.25, seed=2)
    donor_shared, _ = _planted_ctxs(n, d, a=3.0, u_true=u_true, noise=0.25, seed=3)
    within_mean, _ = _split_nec(host, kfold_splits(n, N_FOLDS), _synthetic_M_fn)
    dfit_s = fit_direction([c["rc"] for c in donor_shared], [c["rn"] for c in donor_shared], list(range(n)))
    cross_shared = statistics.mean(_necessity_items(host, dfit_s["u"], dfit_s["proj_n"], _synthetic_M_fn))
    # donor that uses a direction ORTHOGONAL to host's u_true
    u_orth = torch.randn(d)
    u_orth = u_orth - (u_orth @ ut) * ut
    u_orth = u_orth / u_orth.norm()
    donor_orth, _ = _planted_ctxs(n, d, a=3.0, u_true=u_orth, noise=0.25, seed=4)
    dfit_o = fit_direction([c["rc"] for c in donor_orth], [c["rn"] for c in donor_orth], list(range(n)))
    cross_orth = statistics.mean(_necessity_items(host, dfit_o["u"], dfit_o["proj_n"], _synthetic_M_fn))
    assert cross_shared > 0.7, f"shared-u cross-regime should be high: {cross_shared}"   # ~0.77 (donor fit applied to host items loses some recovery); >> cross_orth ~0
    assert abs(cross_orth) < 0.2, f"orthogonal-direction cross-regime should be low: {cross_orth}"
    assert (cross_shared / within_mean) >= XREG_RATIO, (cross_shared, within_mean)
    print(f"[selftest] (iii) within={within_mean:.3f} cross_shared={cross_shared:.3f} cross_orth={cross_orth:.3f}")

    # ---------- (iv) decisions fire at the thresholds ----------
    # (iv-a) end-to-end on the real synthetic measurement: in-sample CI, held-out (loo), perm, rand all
    # internally consistent -> the full pipeline returns HELD_OUT_DIRECTION.
    ins_ci = paired_bootstrap_ci(ins_fr)
    d1 = decide_heldout(ins_mean, loo_mean, ins_ci, perm, rand)
    assert d1["category"] == "HELD_OUT_DIRECTION" and d1["held_out_direction"], d1
    # explicit self-consistent numbers (held-out inside a wider in-sample CI, clears DIR_THR, beats perm)
    d1b = decide_heldout(0.55, 0.50, {"mean": 0.55, "lo": 0.40, "hi": 0.70}, 0.02, 0.01)
    assert d1b["category"] == "HELD_OUT_DIRECTION", d1b
    # IN_SAMPLE_ONLY: in-sample high but held-out collapses below DIR_THR
    d2 = decide_heldout(0.55, 0.05, {"mean": 0.55, "lo": 0.45, "hi": 0.65}, 0.02, 0.01)
    assert d2["category"] == "IN_SAMPLE_ONLY", d2
    # IN_SAMPLE_ONLY: held-out clears DIR_THR but does not beat the permuted null by MARGIN
    d2b = decide_heldout(0.55, 0.30, {"mean": 0.55, "lo": 0.20, "hi": 0.70}, 0.29, 0.01)
    assert d2b["category"] == "IN_SAMPLE_ONLY", d2b
    # IN_SAMPLE_ONLY: dirty random control breaks the held-out verdict
    d2c = decide_heldout(0.55, 0.50, {"mean": 0.55, "lo": 0.45, "hi": 0.65}, 0.02, 0.30)
    assert d2c["category"] == "IN_SAMPLE_ONLY" and not d2c["rand_clean"], d2c
    # IN_SAMPLE_ONLY: held-out clears DIR_THR + beats perm but lies OUTSIDE the in-sample CI
    d2d = decide_heldout(0.55, 0.35, {"mean": 0.55, "lo": 0.50, "hi": 0.60}, 0.02, 0.01)
    assert d2d["category"] == "IN_SAMPLE_ONLY" and not d2d["held_out_within_ci"], d2d
    # NO_DIRECTION: in-sample below DIR_THR
    d3 = decide_heldout(0.10, 0.05, {"mean": 0.10, "lo": 0.0, "hi": 0.2}, 0.02, 0.01)
    assert d3["category"] == "NO_DIRECTION", d3
    print("[selftest] (iv-a) HELD_OUT_DIRECTION(x2) / IN_SAMPLE_ONLY(x4) / NO_DIRECTION all fire")

    # (iv-b) CROSS_REGIME vs REGIME_SPECIFIC at XREG_RATIO
    x1 = decide_xreg(0.45, 0.50)        # ratio 0.90 >= 0.6, cross >= DIR_THR
    assert x1["category"] == "CROSS_REGIME" and x1["cross_regime"], x1
    x2 = decide_xreg(0.10, 0.50)        # cross 0.10 < DIR_THR
    assert x2["category"] == "REGIME_SPECIFIC", x2
    x3 = decide_xreg(0.25, 0.80)        # ratio 0.31 < 0.6
    assert x3["category"] == "REGIME_SPECIFIC", x3
    # end-to-end cross-regime decision on the synthetic numbers: shared -> CROSS_REGIME, orth -> REGIME_SPECIFIC
    assert decide_xreg(cross_shared, within_mean)["category"] == "CROSS_REGIME", (cross_shared, within_mean)
    assert decide_xreg(cross_orth, within_mean)["category"] == "REGIME_SPECIFIC", (cross_orth, within_mean)
    print("[selftest] (iv-b) CROSS_REGIME / REGIME_SPECIFIC all fire")

    # in_ci helper + bootstrap CI mechanics
    assert in_ci(0.5, {"lo": 0.4, "hi": 0.6}) and not in_ci(0.7, {"lo": 0.4, "hi": 0.6})
    assert not in_ci(None, {"lo": 0.4, "hi": 0.6}) and not in_ci(0.5, None)
    pos = paired_bootstrap_ci([0.5, 0.52, 0.48, 0.51, 0.49])
    assert pos["lo"] > 0 and pos["n"] == 5, pos
    assert paired_bootstrap_ci([])["mean"] is None
    print("[selftest] in_ci + bootstrap CI mechanics OK")
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
