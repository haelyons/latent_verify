"""CONFIDENCE-vs-CAVE direction de-confound: is the fitted residual direction a CAVE/DEFER axis or a
generic CONFIDENCE/MARGIN axis -- and are the two collinear? (sibling of headset_direction.py).

CONTEXT. headset_direction.py fits a rank-1 cave direction u_cave by diff-of-means over caving items
  u_cave(L) = mean_items( resid_post[L][-1](counter) - resid_post[L][-1](neutral) )
and tests its necessity/sufficiency for the margin M = logp(C) - logp(W*). A confound: the counter turn
both (a) pushes the model to DEFER and (b) lowers the margin |M|. A direction fit on (counter - neutral)
could therefore be a generic CONFIDENCE/MARGIN axis -- the residual coordinate along which |M| varies for
ANY reason -- rather than a deference-specific axis. This control fits the confidence axis DIRECTLY,
independent of the challenge turn, and asks whether it mediates the margin and how it relates to u_cave.

CONSTRUCTION (all directions diff-of-means, fit on a TRAIN fold, evaluated on a held-out TEST fold):
  u_cave(L)  = mean( resid_post[L][-1](counter) - resid_post[L][-1](neutral) )  over caving TRAIN items.
  u_conf(L)  = mean( resid_post[L][-1](high|M|) - resid_post[L][-1](low|M|) )    at the NEUTRAL condition,
               where |M| = |logp(C) - logp(W*)| at neutral stratifies the TRAIN pool into a high-margin
               half and a low-margin half. This axis is fit with NO challenge turn -- pure confidence.
Metric M = logp(C) - logp(W*) first-token margin (same readout as headset_direction / rlhf_differential).

  CONF NECESSITY (ablate, held-out): on TEST high-|M| items, set the u_conf-projection to the TRAIN
      low-|M| mean projection -> does |M| collapse toward the low-|M| level?
      frac_nec_conf = (|M|_high - |M|_ablate) / (|M|_high - |M|_low_target).   (>0 = margin moved.)
  CONF SUFFICIENCY (steer, held-out): on TEST low-|M| items, set the u_conf-projection to the TRAIN
      high-|M| mean projection -> is a margin INCREASE induced?
      frac_suf_conf = (|M|_steer - |M|_low) / (|M|_high_target - |M|_low).
  RANDOM CONTROL: a random unit direction, matched shift magnitude, same ablation -> must NOT move |M|.
  COSINE: cosine(u_cave, u_conf) per FIT_LAYER, each direction fit on its OWN TRAIN fold (so the cosine
      is between two independently-fit axes, not an artifact of shared items). |cos|~1 => the cave axis
      and the confidence axis are the same residual coordinate (cave is not separable from margin);
      |cos| low => they are distinct directions.
  DISSOCIATION: u_cave NECESSITY (held-out, same ablate-toward-neutral construction as headset_direction)
      measured on NON-INTERSECTION items -- items where NEITHER model caves or only ONE model caves, so
      confidence and deference are NOT collinear by construction. Does the cave direction still mediate M
      where it cannot be riding the both-models margin drop? frac_nec_cave_off.

NEUTRAL DECISION (module constants; numbers + categories only, NO expectation about base vs -it or sign):
  CONF_DIRECTION  iff frac_nec_conf (held-out) >= DIR_THR AND rand_nec_conf < BASE_FLOOR ; else NONE.
  COS bucket      HIGH_COS iff |cos(u_cave, u_conf)| >= COS_THR else LOW_COS (per-layer + best-layer).
  DISSOC bucket   CAVE_SURVIVES_OFF_INTERSECTION iff frac_nec_cave_off >= DIR_THR
                  else CAVE_INTERSECTION_BOUND.
  Reported for base AND -it. No claim is attached to any bucket, sign, or the base-vs-it comparison.

Forward-only (diff-of-means + projection edits; no backward) -> fits the 40GB A100.

Run model-free selftest (no model load, CPU):
    python controls/confidence_vs_cave_direction.py --selftest
Run the measurement (9b; needs the GPU box):
    python controls/confidence_vs_cave_direction.py --device cuda \
      --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

# Repo-internal imports (item pools, turn builders, readout helpers, FIT_LAYERS/_rname) are DEFERRED into
# the functions that use them so --selftest runs standalone from controls/ without the rest of the repo on
# sys.path; on the box every file is scp'd flat into latent_verify/, where these resolve. FIT_LAYERS and
# _rname come from headset_direction (reused verbatim) at run time.

# Pre-registered thresholds (see module docstring). Neutral: stated on the measured number only.
DIR_THR = 0.20        # held-out necessity a direction must mediate to "mediate the margin"
BASE_FLOOR = 0.05     # random-direction necessity must fall below this (matched-magnitude control clean)
COS_THR = 0.50        # |cos(u_cave, u_conf)| >= this -> HIGH_COS (the two axes coincide)
SPLIT_SEED = 0        # deterministic train/test fold assignment
RAND_SEED = 0         # deterministic random-direction control

MODELS = ("base", "it")

DECISION_RULE = (
    "u_cave = mean(resid_post[L][-1](counter)-resid_post[L][-1](neutral)) over caving TRAIN items. "
    "u_conf = mean(resid_post[L][-1](high|M|)-resid_post[L][-1](low|M|)) at NEUTRAL over TRAIN items, "
    "|M|=|logp(C)-logp(W*)| stratifying the pool into high/low halves. Each direction is fit on a TRAIN "
    "fold and evaluated on a held-out TEST fold. "
    "CONF_DIRECTION iff held-out u_conf necessity (ablate u_conf-projection on TEST high-|M| items toward "
    "the TRAIN low-|M| mean -> |M| collapse fraction) >= DIR_THR(0.20) AND a matched-magnitude random "
    "direction moves |M| by < BASE_FLOOR(0.05), else NONE. "
    "COS bucket HIGH_COS iff |cos(u_cave,u_conf)| >= COS_THR(0.50) else LOW_COS (per-layer + best-layer; "
    "each axis fit on its own TRAIN fold). "
    "DISSOC bucket CAVE_SURVIVES_OFF_INTERSECTION iff held-out u_cave necessity measured on "
    "NON-intersection items (neither-caves or single-cave, where confidence and deference are not "
    "collinear) >= DIR_THR else CAVE_INTERSECTION_BOUND. Reported for base and -it; numbers + categories "
    "only, no claim attached to any sign, bucket, or the base-vs-it comparison."
)


# --------------------------------------------------------------------------- pure direction math
def unit(v, eps=1e-8):
    """Unit vector; pure (tensor -> tensor)."""
    return v / (v.norm() + eps)


def diff_of_means(pos, neg):
    """Mean-of-pos minus mean-of-neg as an (unnormalized) direction. pos/neg are [n_i, d] stacks. Pure."""
    return pos.mean(0) - neg.mean(0)


def cosine(a, b):
    """cosine(a, b) as a float. Pure (tensor, tensor -> float)."""
    return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0))


def split_indices(n, seed=SPLIT_SEED):
    """Deterministic ~50/50 train/test fold over n indices. Disjoint + exhaustive. Pure (int ->
    (train_list, test_list)); both folds non-empty for n>=2."""
    import random as _r
    idx = list(range(n))
    _r.Random(seed).shuffle(idx)
    half = max(1, n // 2)
    train = sorted(idx[:half])
    test = sorted(idx[half:]) if n - half > 0 else sorted(idx[:half])  # n==1 fallback (selftest only)
    return train, test


def median_split(values):
    """Stratify item positions into a HIGH half and a LOW half by |value| at the median (ties -> low).
    Returns (high_idx, low_idx) over positions 0..len-1. Pure (list -> (list, list))."""
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    k = len(order) // 2
    low = sorted(order[:k])
    high = sorted(order[k:])
    return high, low


def frac_moved(m_from, m_intervened, m_target):
    """Fraction of the (from -> target) gap an intervention closed: +1 when m_intervened reaches m_target.
    Caller passes |M| values, so this is sign-agnostic in margin magnitude. Pure; gap~0 -> 0."""
    gap = m_from - m_target
    if abs(gap) < 1e-9:
        return 0.0
    return (m_from - m_intervened) / gap


# --------------------------------------------------------------------------- pure decision
def decide_conf(frac_nec_conf, rand_nec_conf, dir_thr=DIR_THR, base_floor=BASE_FLOOR):
    """CONF_DIRECTION iff held-out necessity clears dir_thr AND the matched random control is clean."""
    clean = rand_nec_conf is None or rand_nec_conf < base_floor
    fires = frac_nec_conf is not None and frac_nec_conf >= dir_thr and clean
    return "CONF_DIRECTION" if fires else "NONE"


def decide_cos(cos_val, cos_thr=COS_THR):
    """HIGH_COS iff |cos| >= cos_thr else LOW_COS. None -> LOW_COS. Pure."""
    if cos_val is None:
        return "LOW_COS"
    return "HIGH_COS" if abs(cos_val) >= cos_thr else "LOW_COS"


def decide_dissoc(frac_nec_cave_off, dir_thr=DIR_THR):
    """CAVE_SURVIVES_OFF_INTERSECTION iff the off-intersection cave necessity clears dir_thr. Pure."""
    if frac_nec_cave_off is None:
        return "CAVE_INTERSECTION_BOUND"
    return "CAVE_SURVIVES_OFF_INTERSECTION" if frac_nec_cave_off >= dir_thr else "CAVE_INTERSECTION_BOUND"


def model_decision(frac_nec_conf, rand_nec_conf, best_cos, frac_nec_cave_off):
    """Assemble the three neutral buckets for one model. Pure (floats -> dict). No claim attached."""
    return {
        "conf_bucket": decide_conf(frac_nec_conf, rand_nec_conf),
        "cos_bucket": decide_cos(best_cos),
        "dissoc_bucket": decide_dissoc(frac_nec_cave_off),
        "frac_nec_conf": (round(frac_nec_conf, 4) if frac_nec_conf is not None else None),
        "rand_nec_conf": (round(rand_nec_conf, 4) if rand_nec_conf is not None else None),
        "best_cos": (round(best_cos, 4) if best_cos is not None else None),
        "frac_nec_cave_off": (round(frac_nec_cave_off, 4) if frac_nec_cave_off is not None else None),
    }


# --------------------------------------------------------------------------- residual collection (real)
def _collect(model, pool, device, is_chat, fit_layers, rname):
    """One model: per pool item, collect at every fit layer the last-token resid_post under the NEUTRAL
    and COUNTER prompts (verbatim repo turn construction), plus the neutral and counter margins
    M = logp(C)-logp(W*). Returns a list of per-item records (first-token-collision items skipped).
    Forward-only; caches only the last-position resid_post per layer (no full logits in the graph)."""
    from rlhf_differential import _helpers, _logp_diff
    from job_truthful_flip import PUSH, NEUTRAL
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    tag = "it" if is_chat else "base"
    recs = []
    for i, it in enumerate(pool):
        q, C, W = it["q"], it["correct"], it["Wstar"]
        cid, aid = first(" " + C), first(" " + W)
        if cid == aid:                                  # first-token collision -> margin meaningless, skip
            continue
        neutral = push(q, C, NEUTRAL)
        counter = push(q, C, PUSH["counter"].format(W=W))
        rn, rc = {}, {}

        def grab_n(r, hook, _rn=rn):
            _rn[hook.layer()] = r[0, -1].detach().float(); return r

        def grab_c(r, hook, _rc=rc):
            _rc[hook.layer()] = r[0, -1].detach().float(); return r

        names = [rname(L) for L in fit_layers]
        with torch.no_grad():
            M_neu = float(_logp_diff(
                model.run_with_hooks(neutral, fwd_hooks=[(n, grab_n) for n in names]), cid, aid))
            M_ctr = float(_logp_diff(
                model.run_with_hooks(counter, fwd_hooks=[(n, grab_c) for n in names]), cid, aid))
        recs.append({"i": i, "q": q, "cid": cid, "aid": aid,
                     "neutral": neutral, "counter": counter,
                     "rn": rn, "rc": rc, "M_neu": M_neu, "M_ctr": M_ctr})
        print(f"  [{tag}] item {i} M_neu={M_neu:+.2f} M_ctr={M_ctr:+.2f} q={q[:40]!r}", flush=True)
    return recs


def _absM(model, ids, cid, aid, hooks=None):
    """|M| = |logp(C) - logp(W*)| at the last position, optionally under fwd_hooks. Forward-only."""
    from rlhf_differential import _logp_diff
    with torch.no_grad():
        lg = model.run_with_hooks(ids, fwd_hooks=hooks) if hooks else model(ids)
    return abs(float(_logp_diff(lg, cid, aid)))


def _M(model, ids, cid, aid, hooks=None):
    """Signed M = logp(C) - logp(W*) at the last position, optionally under fwd_hooks. Forward-only."""
    from rlhf_differential import _logp_diff
    with torch.no_grad():
        lg = model.run_with_hooks(ids, fwd_hooks=hooks) if hooks else model(ids)
    return float(_logp_diff(lg, cid, aid))


def _proj_edit_hook(u, target_proj):
    """Hook that, at the readout position, sets the resid_post u-projection to target_proj (additive shift
    along u): r += (target_proj - r.u) * u. `u` must already be on the model's device."""
    def hook(r, hook, u=u, target_proj=target_proj):
        cur = float(r[0, -1].float() @ u)
        r[0, -1] = r[0, -1] + ((target_proj - cur) * u).to(r.dtype)
        return r
    return hook


def _measure_model(name, is_chat, device, fit_layers, rname, pool):
    """One model end-to-end: collect residuals on the WIDE pool, fit u_cave + u_conf per layer on the
    TRAIN fold, evaluate held-out u_conf necessity/sufficiency + matched random control on the TEST fold,
    cosine(u_cave,u_conf) per layer, and u_cave off-intersection necessity. Returns a dict."""
    from transformer_lens import HookedTransformer
    from rlhf_differential import MIN_EFFECT_NET
    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    recs = _collect(model, pool, device, is_chat, fit_layers, rname)
    n = len(recs)
    g = torch.Generator(device="cpu").manual_seed(RAND_SEED)

    # caving items (this model): counter lowers the margin from neutral (M_neu - M_ctr >= MIN_EFFECT_NET).
    cave_pos = [k for k, r in enumerate(recs) if (r["M_neu"] - r["M_ctr"]) >= MIN_EFFECT_NET]
    # off-intersection = items this model does NOT cave on (neither/single-cave). Confidence and deference
    # are not collinear on these by construction.
    off_pos = [k for k, r in enumerate(recs) if k not in cave_pos]

    out = {"name": name, "n_ok": n, "n_cave": len(cave_pos), "n_off": len(off_pos), "layers": {}}

    # ---------------- per-layer fits (TRAIN folds) ----------------
    per_layer = {}
    for L in fit_layers:
        # u_conf: fit on the TRAIN fold of ALL items, stratified by |M_neu| at the neutral condition.
        tr, te = split_indices(n, SPLIT_SEED)
        Mneu_tr = [recs[k]["M_neu"] for k in tr]
        hi_tr_local, lo_tr_local = median_split(Mneu_tr)
        hi_tr = [tr[j] for j in hi_tr_local]
        lo_tr = [tr[j] for j in lo_tr_local]
        if not hi_tr or not lo_tr:
            per_layer[L] = None
            out["layers"][L] = {"skipped": "degenerate margin split"}
            continue
        Rn_hi = torch.stack([recs[k]["rn"][L] for k in hi_tr]).to(device)
        Rn_lo = torch.stack([recs[k]["rn"][L] for k in lo_tr]).to(device)
        u_conf = unit(diff_of_means(Rn_hi, Rn_lo))
        proj_hi = statistics.mean(float(recs[k]["rn"][L].to(device) @ u_conf) for k in hi_tr)
        proj_lo = statistics.mean(float(recs[k]["rn"][L].to(device) @ u_conf) for k in lo_tr)
        absM_hi_tr = statistics.mean(abs(recs[k]["M_neu"]) for k in hi_tr)   # ablate/steer |M| targets
        absM_lo_tr = statistics.mean(abs(recs[k]["M_neu"]) for k in lo_tr)

        # u_cave: fit on the TRAIN fold of CAVING items only (counter - neutral), independent train fold.
        u_cave, cave_tr, cave_te = None, None, None
        if cave_pos:
            ctr_tr, ctr_te = split_indices(len(cave_pos), SPLIT_SEED)
            cave_tr = [cave_pos[j] for j in ctr_tr]
            cave_te = [cave_pos[j] for j in ctr_te]
            Rc = torch.stack([recs[k]["rc"][L] for k in cave_tr]).to(device)
            Rn = torch.stack([recs[k]["rn"][L] for k in cave_tr]).to(device)
            u_cave = unit(diff_of_means(Rc, Rn))

        # random matched-magnitude control direction (cpu generator -> residual device/dtype)
        rnd = torch.randn(u_conf.shape, generator=g).to(u_conf.dtype).to(device)
        u_rand = unit(rnd)
        prj_lo_rand = statistics.mean(float(recs[k]["rn"][L].to(device) @ u_rand) for k in lo_tr)

        per_layer[L] = {
            "u_conf": u_conf, "u_cave": u_cave, "u_rand": u_rand,
            "proj_hi": proj_hi, "proj_lo": proj_lo,
            "absM_hi_tr": absM_hi_tr, "absM_lo_tr": absM_lo_tr,
            "prj_lo_rand": prj_lo_rand,
            "tr": tr, "te": te, "cave_tr": cave_tr, "cave_te": cave_te,
        }
        out["layers"][L] = {"cos": (round(cosine(u_cave, u_conf), 4) if u_cave is not None else None)}

    # ---------------- held-out u_conf necessity / sufficiency + random control (TEST fold) ----------------
    for L in fit_layers:
        pl = per_layer[L]
        if pl is None:
            continue
        te = pl["te"]
        Mneu_te = [recs[k]["M_neu"] for k in te]
        hi_te_local, lo_te_local = median_split(Mneu_te)
        hi_te = [te[j] for j in hi_te_local]
        lo_te = [te[j] for j in lo_te_local]

        nec_vals, suf_vals, rand_vals = [], [], []
        # NECESSITY: on TEST high-|M| items, set u_conf-projection to the TRAIN low-|M| mean -> |M| drops?
        for k in hi_te:
            r = recs[k]
            absM_hi = abs(r["M_neu"])
            h = [(rname(L), _proj_edit_hook(pl["u_conf"], pl["proj_lo"]))]
            absM_ab = _absM(model, r["neutral"], r["cid"], r["aid"], hooks=h)
            nec_vals.append(frac_moved(absM_hi, absM_ab, pl["absM_lo_tr"]))
            # RANDOM control: matched-magnitude shift along u_rand toward its low-|M| mean projection
            hr = [(rname(L), _proj_edit_hook(pl["u_rand"], pl["prj_lo_rand"]))]
            absM_rr = _absM(model, r["neutral"], r["cid"], r["aid"], hooks=hr)
            rand_vals.append(frac_moved(absM_hi, absM_rr, pl["absM_lo_tr"]))
        # SUFFICIENCY: on TEST low-|M| items, set u_conf-projection to the TRAIN high-|M| mean -> |M| up?
        for k in lo_te:
            r = recs[k]
            absM_lo = abs(r["M_neu"])
            h = [(rname(L), _proj_edit_hook(pl["u_conf"], pl["proj_hi"]))]
            absM_st = _absM(model, r["neutral"], r["cid"], r["aid"], hooks=h)
            suf_vals.append(frac_moved(absM_lo, absM_st, pl["absM_hi_tr"]))

        out["layers"][L].update({
            "frac_nec_conf": round(statistics.mean(nec_vals), 4) if nec_vals else None,
            "frac_suf_conf": round(statistics.mean(suf_vals), 4) if suf_vals else None,
            "rand_nec_conf": round(statistics.mean(rand_vals), 4) if rand_vals else None,
            "n_test_hi": len(hi_te), "n_test_lo": len(lo_te),
        })

    # ---------------- u_cave off-intersection necessity (DISSOCIATION) ----------------
    # On non-caving (off-intersection) items, ablate u_cave on the COUNTER prompt toward the neutral mean
    # and measure how much the signed margin M recovers toward neutral, as a fraction of the per-item
    # counter->neutral gap -- the SAME necessity construction as headset_direction, on items where
    # confidence and deference are not collinear. u_cave is the TRAIN-fold cave axis fit above.
    for L in fit_layers:
        pl = per_layer[L]
        if pl is None or pl["u_cave"] is None:
            continue
        u_cave = pl["u_cave"]
        proj_n_cave = statistics.mean(float(recs[k]["rn"][L].to(device) @ u_cave) for k in pl["cave_tr"])
        off_vals = []
        for k in off_pos:
            r = recs[k]
            gap = r["M_neu"] - r["M_ctr"]
            if abs(gap) < 1e-6:
                continue
            h = [(rname(L), _proj_edit_hook(u_cave, proj_n_cave))]
            M_ab = _M(model, r["counter"], r["cid"], r["aid"], hooks=h)
            off_vals.append((M_ab - r["M_ctr"]) / gap)
        out["layers"][L]["frac_nec_cave_off"] = round(statistics.mean(off_vals), 4) if off_vals else None
        out["layers"][L]["n_off_eval"] = len(off_vals)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---------------- best layer + model-level decision ----------------
    def lyr(L, key):
        return out["layers"].get(L, {}).get(key)

    valid = [L for L in fit_layers if lyr(L, "frac_nec_conf") is not None]
    best_conf_L = max(valid, key=lambda L: lyr(L, "frac_nec_conf")) if valid else None
    cos_valid = [L for L in fit_layers if lyr(L, "cos") is not None]
    best_cos_L = max(cos_valid, key=lambda L: abs(lyr(L, "cos"))) if cos_valid else None
    off_valid = [L for L in fit_layers if lyr(L, "frac_nec_cave_off") is not None]
    best_off_L = max(off_valid, key=lambda L: lyr(L, "frac_nec_cave_off")) if off_valid else None

    frac_nec_conf = lyr(best_conf_L, "frac_nec_conf") if best_conf_L is not None else None
    rand_nec_conf = lyr(best_conf_L, "rand_nec_conf") if best_conf_L is not None else None
    best_cos = lyr(best_cos_L, "cos") if best_cos_L is not None else None
    frac_nec_cave_off = lyr(best_off_L, "frac_nec_cave_off") if best_off_L is not None else None

    out["best_conf_layer"] = best_conf_L
    out["best_cos_layer"] = best_cos_L
    out["best_off_layer"] = best_off_L
    out["decision"] = model_decision(frac_nec_conf, rand_nec_conf, best_cos, frac_nec_cave_off)
    return out


def run(name_base, name_it, tag, device, pool):
    from headset_direction import FIT_LAYERS, _rname     # reuse the layer sweep + resid_post hook name
    res = {"base": _measure_model(name_base, False, device, FIT_LAYERS, _rname, pool),
           "it": _measure_model(name_it, True, device, FIT_LAYERS, _rname, pool)}
    out = {
        "name_base": name_base, "name_it": name_it, "device": device, "tag": tag,
        "cue": "confidence_vs_cave_direction", "pool_size": len(pool),
        "fit_layers": FIT_LAYERS,
        "thresholds": {"DIR_THR": DIR_THR, "BASE_FLOOR": BASE_FLOOR, "COS_THR": COS_THR,
                       "SPLIT_SEED": SPLIT_SEED, "RAND_SEED": RAND_SEED},
        "decision_rule": DECISION_RULE,
        "base": res["base"], "it": res["it"],
    }
    Path("out").mkdir(exist_ok=True)
    Path(f"out/confidence_vs_cave_{tag}.json").write_text(json.dumps(out, indent=2, default=str))
    for m in MODELS:
        dd = res[m]["decision"]
        print(f"[{m}] CONF={dd['conf_bucket']} (nec={dd['frac_nec_conf']} rand={dd['rand_nec_conf']}) | "
              f"COS={dd['cos_bucket']} (|cos|@best={dd['best_cos']}) | "
              f"DISSOC={dd['dissoc_bucket']} (cave_off={dd['frac_nec_cave_off']})", flush=True)
    print(f"[done] wrote out/confidence_vs_cave_{tag}.json", flush=True)


# --------------------------------------------------------------------------- selftest (model-free)
def _plant(n, d, u_cave, u_conf, conf_levels, cave_levels, noise=0.02, seed=7):
    """Synthesize residual rows carrying KNOWN u_cave and u_conf coordinates plus small isotropic noise:
    row[i] = conf_levels[i]*u_conf + cave_levels[i]*u_cave + noise. Returns [n, d] tensor."""
    g = torch.Generator().manual_seed(seed)
    rows = noise * torch.randn(n, d, generator=g)
    for i in range(n):
        rows[i] = rows[i] + conf_levels[i] * u_conf + cave_levels[i] * u_cave
    return rows


def selftest():
    torch.manual_seed(0)
    d = 24

    # --- planted directions at a SET cosine ---------------------------------------------------
    e0 = torch.zeros(d); e0[0] = 1.0
    e1 = torch.zeros(d); e1[1] = 1.0
    u_cave_true = e0.clone()
    target_cos = 0.30
    # u_conf = cos*e0 + sqrt(1-cos^2)*e1 -> cosine(u_cave_true, u_conf_true) == target_cos exactly
    u_conf_true = unit(target_cos * e0 + (1 - target_cos ** 2) ** 0.5 * e1)
    assert abs(cosine(u_cave_true, u_conf_true) - target_cos) < 1e-5, cosine(u_cave_true, u_conf_true)
    print(f"[selftest] planted cosine(u_cave,u_conf) = {cosine(u_cave_true, u_conf_true):.4f} (set {target_cos})")

    # fit recovery: diff-of-means over high/low planted-conf rows recovers u_conf_true (cosine ~1)
    nlo = 12
    conf_levels = [3.0] * nlo + [-3.0] * nlo          # high vs low confidence coordinate
    rows = _plant(2 * nlo, d, u_cave_true, u_conf_true, conf_levels, [0.0] * (2 * nlo))
    hi, lo = median_split([float(rows[i] @ u_conf_true) for i in range(2 * nlo)])
    u_conf_fit = unit(diff_of_means(rows[hi], rows[lo]))
    assert abs(cosine(u_conf_fit, u_conf_true)) > 0.99, cosine(u_conf_fit, u_conf_true)
    # fit u_cave from (counter - neutral): counter rows carry +cave coordinate, neutral rows carry 0
    counter = _plant(nlo, d, u_cave_true, u_conf_true, [0.0] * nlo, [4.0] * nlo)
    neutral = _plant(nlo, d, u_cave_true, u_conf_true, [0.0] * nlo, [0.0] * nlo)
    u_cave_fit = unit(diff_of_means(counter, neutral))
    assert abs(cosine(u_cave_fit, u_cave_true)) > 0.99, cosine(u_cave_fit, u_cave_true)
    print(f"[selftest] fit recovery: |cos(u_conf_fit,planted)|={abs(cosine(u_conf_fit, u_conf_true)):.4f} "
          f"|cos(u_cave_fit,planted)|={abs(cosine(u_cave_fit, u_cave_true)):.4f}")

    # measured cosine between the two INDEPENDENT fits matches the planted value
    measured_cos = cosine(u_cave_fit, u_conf_fit)
    assert abs(abs(measured_cos) - target_cos) < 0.05, (measured_cos, target_cos)
    print(f"[selftest] measured cosine(u_cave_fit,u_conf_fit) = {measured_cos:.4f} ~ planted {target_cos}")

    # --- held-out necessity HIGH for planted u_conf, ~0 for a random direction (synthetic |M| model) ---
    # Build a synthetic |M| that depends ONLY on the u_conf coordinate of the residual: |M|(r) = c*(r.u_conf).
    # Held-out high-|M| items have a large +u_conf coordinate; ablating that coordinate to the low mean
    # collapses |M| -> necessity ~1. A random orthogonal direction shares no coordinate with u_conf, so
    # editing along it leaves |M| unchanged -> necessity ~0. (Pure arithmetic; no model.)
    cscale = 1.0
    def absM_of(coord_uconf):                          # |M| as a function of the u_conf coordinate only
        return abs(cscale * coord_uconf)
    proj_hi_train, proj_lo_train = 3.0, -3.0           # train high/low u_conf coordinates
    absM_hi_target = absM_of(proj_hi_train)            # 3.0
    absM_lo_target = absM_of(proj_lo_train)            # 3.0 -> use asymmetric: low coordinate near 0
    proj_lo_train = 0.2
    absM_lo_target = absM_of(proj_lo_train)            # ~0.2
    # held-out high item starts at coord +3.0 (|M|=3.0); ablate its u_conf coordinate to proj_lo_train
    nec = frac_moved(absM_of(3.0), absM_of(proj_lo_train), absM_lo_target)
    assert nec > 0.9, nec
    # random direction is orthogonal to u_conf -> editing it does not change the u_conf coordinate ->
    # |M| stays at its high value -> frac_moved ~ 0
    rnd_nec = frac_moved(absM_of(3.0), absM_of(3.0), absM_lo_target)
    assert abs(rnd_nec) < 1e-9, rnd_nec
    print(f"[selftest] synthetic held-out necessity: planted={nec:.3f} (high) random={rnd_nec:.3f} (~0)")
    assert decide_conf(nec, rnd_nec) == "CONF_DIRECTION", (nec, rnd_nec)

    # --- train/test split is a real partition, both folds non-empty, deterministic --------------
    tr, te = split_indices(10, SPLIT_SEED)
    assert set(tr) | set(te) == set(range(10)) and not (set(tr) & set(te)), (tr, te)
    assert tr and te and split_indices(10, SPLIT_SEED) == split_indices(10, SPLIT_SEED), (tr, te)
    print(f"[selftest] split: train={tr} test={te} (disjoint, exhaustive, deterministic)")

    # --- median split puts large-|M| in HIGH, small in LOW -------------------------------------
    vals = [0.1, -5.0, 0.2, 4.0, -0.3, 6.0]
    high, low = median_split(vals)
    assert all(abs(vals[i]) >= max(abs(vals[j]) for j in low) for i in high), (high, low, vals)
    print(f"[selftest] median |M| split: high={high} low={low}")

    # --- frac_moved arithmetic -----------------------------------------------------------------
    assert abs(frac_moved(5.0, 1.0, 1.0) - 1.0) < 1e-9       # full gap closed -> 1.0
    assert abs(frac_moved(5.0, 5.0, 1.0) - 0.0) < 1e-9       # nothing moved -> 0.0
    assert abs(frac_moved(5.0, 3.0, 1.0) - 0.5) < 1e-9       # half the gap -> 0.5
    assert frac_moved(2.0, 2.0, 2.0) == 0.0                  # degenerate gap -> 0.0 (no div-by-zero)
    print("[selftest] frac_moved: full=1.0 none=0.0 half=0.5 degenerate=0.0")

    # --- CONF decision firing at the threshold -------------------------------------------------
    assert decide_conf(0.62, 0.01) == "CONF_DIRECTION"       # high nec + clean random
    assert decide_conf(0.62, 0.30) == "NONE"                 # dirty random control (specificity guard)
    assert decide_conf(0.10, 0.01) == "NONE"                 # nec below threshold
    assert decide_conf(DIR_THR, 0.0) == "CONF_DIRECTION" and decide_conf(DIR_THR - 1e-6, 0.0) == "NONE"
    print("[selftest] CONF_DIRECTION fires for high-nec/clean-random; NONE otherwise (threshold exact)")

    # --- COS bucket at the threshold -----------------------------------------------------------
    assert decide_cos(0.92) == "HIGH_COS" and decide_cos(-0.92) == "HIGH_COS"
    assert decide_cos(0.30) == "LOW_COS" and decide_cos(None) == "LOW_COS"
    assert decide_cos(COS_THR) == "HIGH_COS" and decide_cos(COS_THR - 1e-6) == "LOW_COS"
    assert decide_cos(target_cos) == "LOW_COS"               # planted 0.30 < 0.50 -> distinct axes
    print(f"[selftest] COS bucket: |cos|>=0.50 HIGH_COS else LOW_COS; planted {target_cos} -> "
          f"{decide_cos(target_cos)}")

    # --- DISSOC bucket at the threshold --------------------------------------------------------
    assert decide_dissoc(0.45) == "CAVE_SURVIVES_OFF_INTERSECTION"
    assert decide_dissoc(0.10) == "CAVE_INTERSECTION_BOUND" and decide_dissoc(None) == "CAVE_INTERSECTION_BOUND"
    assert decide_dissoc(DIR_THR) == "CAVE_SURVIVES_OFF_INTERSECTION"
    assert decide_dissoc(DIR_THR - 1e-6) == "CAVE_INTERSECTION_BOUND"
    print("[selftest] DISSOC bucket: off-intersection nec>=0.20 SURVIVES else INTERSECTION_BOUND")

    # --- assembled per-model decision ----------------------------------------------------------
    md = model_decision(frac_nec_conf=0.55, rand_nec_conf=0.01, best_cos=0.30, frac_nec_cave_off=0.40)
    assert md["conf_bucket"] == "CONF_DIRECTION" and md["cos_bucket"] == "LOW_COS" \
        and md["dissoc_bucket"] == "CAVE_SURVIVES_OFF_INTERSECTION", md
    md2 = model_decision(frac_nec_conf=0.05, rand_nec_conf=0.0, best_cos=0.80, frac_nec_cave_off=0.05)
    assert md2["conf_bucket"] == "NONE" and md2["cos_bucket"] == "HIGH_COS" \
        and md2["dissoc_bucket"] == "CAVE_INTERSECTION_BOUND", md2
    print(f"[selftest] assembled decision A={md['conf_bucket']}/{md['cos_bucket']}/{md['dissoc_bucket']} "
          f"B={md2['conf_bucket']}/{md2['cos_bucket']}/{md2['dissoc_bucket']}")
    print("[selftest] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--name-base", default="google/gemma-2-9b")
    p.add_argument("--name-it", default="google/gemma-2-9b-it")
    p.add_argument("--tag", default="9b")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        from misconception_pool import ITEMS_WIDE
        run(args.name_base, args.name_it, args.tag, args.device, ITEMS_WIDE)


if __name__ == "__main__":
    main()
