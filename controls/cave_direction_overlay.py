"""OVERLAY vs MECHANISM de-confound for the rank-1 cave/defer DIRECTION (sibling of headset_direction.py
and confidence_vs_cave_direction.py).

CONTEXT. headset_direction.py fits a rank-1 cave direction u_cave by diff-of-means over caving items
  u_cave(L) = mean_items( resid_post[L][-1](counter) - resid_post[L][-1](neutral) )
and shows it is CAUSAL for the first-token margin M = logp(C) - logp(W*): ablating the u_cave-projection
on a caving COUNTER residual (move the projection to the neutral mean) recovers the caved margin. But a
causal direction need not be the model's own computation. Makelov et al. (2023, "Is This the Subspace
You Are Looking For?") show that an activation-patching direction with output leverage can be an OVERLAY:
a control axis the editor uses to drive the readout, not a coordinate the network reads. An overlay edit
moves the readout BROADLY -- it pushes the whole next-token distribution around -- whereas a mechanism
edit moves it in a TARGETED, regime-SPECIFIC way (mass flows between the two competitors, only when the
model is actually caving). This control measures that targetedness/specificity with NO new mechanism.

WHAT IT MEASURES (base vs it, at FIT_LAYERS; headline = the layer with the largest cave-NECESSITY):
  Fit u_cave (diff-of-means counter-neutral over the caving items, HELD-OUT: fit on a TRAIN fold,
  evaluate on the disjoint TEST fold, exactly the held-out construction of cave_direction_heldout.py).
  Ablation = the projection edit: set the residual's u_cave-projection to the TRAIN neutral mean.

  (1) TARGETEDNESS. On TEST caving items (COUNTER prompt), apply the u_cave ablation and read the change
      in the FULL next-token softmax P. Report
        delta_P_C      = P_ablate(C)  - P_counter(C)         (rise in the correct competitor)
        delta_P_Wstar  = P_ablate(W*) - P_counter(W*)        (fall in the misconception competitor)
        L1             = sum_v |P_ablate(v) - P_counter(v)|   (total probability-mass movement)
        target_frac    = (|delta_P_C| + |delta_P_Wstar|) / L1 -- fraction of the mass movement on {C,W*}.
      Also the KL of the change restricted to {C,W*} vs total KL(P_ablate || P_counter):
        kl_total, kl_pair, kl_pair_frac = kl_pair / kl_total.
  (2) OFF-REGIME SPECIFICITY. Apply the SAME u_cave ablation (same direction, same projection target) to
        (a) the NEUTRAL condition of the caving items (no pushback present), and
        (b) NON-caving control items (items this model does not cave on),
      and report the L1 distribution change there. off_regime_L1 = mean over (a)+(b). A mechanism perturbs
      little off the caving regime; an overlay perturbs broadly regardless of regime. Reported as the
      ratio offreg_ratio = off_regime_L1 / on_regime_L1.
  (3) RANDOM-direction control (matched magnitude): a random unit direction, shifted by the matched
      projection magnitude, on the same TEST caving COUNTER items -> rand_target_frac (and rand on/off L1).
      A mechanism's u_cave moves a high target_frac while a matched random direction moves a low one.

NEUTRAL DECISION (module constants TARGET_FRAC=0.5, OFFREG_FLOOR=0.10, RAND_GAP=0.15; numbers +
categories only, no hypothesis, no statement about which model should win):
  MECHANISM_LIKE iff target_frac >= TARGET_FRAC AND off_regime_L1 < OFFREG_FLOOR * on_regime_L1 (i.e.
      offreg_ratio < OFFREG_FLOOR) AND (target_frac - rand_target_frac) >= RAND_GAP (the random direction's
      target_frac is much lower); else OVERLAY_LIKE (the ablation is broad, or non-specific off-regime, or
      no better than a random direction). Reported per model with target_frac, delta_P_C, delta_P_Wstar,
      offreg_ratio, rand_target_frac.

Forward-only (diff-of-means + projection edits; no backward) -> fits the 40GB A100. Reuses the verified
primitives (FIT_LAYERS, the resid_post hook name, the held-out fold split, the diff-of-means fit, the
projection-edit hook, the matched random control, M=logp(C)-logp(W*), entropy_of_logits) from the
reference modules.

  python controls/cave_direction_overlay.py --selftest
  python controls/cave_direction_overlay.py --device cuda \
    --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

# Repo-internal imports (item pools, turn builders, readout helpers, FIT_LAYERS, MIN_EFFECT_NET) are
# DEFERRED into the functions that use them so --selftest runs standalone from controls/ on CPU with NO
# model load and nothing else on sys.path; on the box every file is scp'd flat into latent_verify/ where
# these resolve. entropy_of_logits is RE-IMPLEMENTED below (not imported) for the same reason -- the same
# convention controls/entropy_distributed_presoftcap.py uses for its sibling-control pure helpers.

# Pre-registered thresholds (see module docstring). Neutral: stated on the measured number only.
TARGET_FRAC = 0.5     # fraction of the L1 mass-movement that must land on {C, W*} for the edit to be targeted
OFFREG_FLOOR = 0.10   # off-regime L1 must be < this x the on-regime L1 for the edit to be regime-specific
RAND_GAP = 0.15       # u_cave target_frac must exceed the matched-random direction's target_frac by this
SPLIT_SEED = 0        # deterministic train/test fold assignment (same as confidence_vs_cave_direction)
RAND_SEED = 0         # deterministic random-direction control

MODELS = ("base", "it")

DECISION_RULE = (
    "u_cave = mean(resid_post[L][-1](counter)-resid_post[L][-1](neutral)) over caving TRAIN items; ablation "
    "= set the residual u_cave-projection to the TRAIN neutral mean. Evaluated on a held-out TEST fold. "
    "target_frac = (|dP(C)|+|dP(W*)|)/L1, where dP is the change in the FULL next-token softmax under the "
    "ablation on TEST caving COUNTER items and L1 is the total |.| mass movement. off_regime_L1 = mean L1 "
    "of the SAME ablation applied to (a) the NEUTRAL condition of the caving items and (b) non-caving "
    "control items; offreg_ratio = off_regime_L1 / on_regime_L1. rand_target_frac = target_frac of a "
    "matched-magnitude random unit direction on the same TEST caving COUNTER items. "
    "MECHANISM_LIKE iff target_frac >= TARGET_FRAC(0.5) AND offreg_ratio < OFFREG_FLOOR(0.10) AND "
    "(target_frac - rand_target_frac) >= RAND_GAP(0.15); else OVERLAY_LIKE. Reported for base and -it; "
    "numbers + categories only, no claim attached to any sign, bucket, or the base-vs-it comparison."
)


# --------------------------------------------------------------------------- pure direction math
def unit(v, eps=1e-8):
    """Unit vector; pure (tensor -> tensor)."""
    return v / (v.norm() + eps)


def diff_of_means(pos, neg):
    """Mean-of-pos minus mean-of-neg as an (unnormalized) direction. pos/neg are [n_i, d] stacks. Pure."""
    return pos.mean(0) - neg.mean(0)


def split_indices(n, seed=SPLIT_SEED):
    """Deterministic ~50/50 train/test fold over n indices (matches confidence_vs_cave_direction). Disjoint
    + exhaustive; both folds non-empty for n>=2. Pure (int -> (train_list, test_list))."""
    import random as _r
    idx = list(range(n))
    _r.Random(seed).shuffle(idx)
    half = max(1, n // 2)
    train = sorted(idx[:half])
    test = sorted(idx[half:]) if n - half > 0 else sorted(idx[:half])  # n==1 fallback (selftest only)
    return train, test


# --------------------------------------------------------------------------- pure distribution-change math
def entropy_of_logits(logits):
    """Shannon entropy (nats) of softmax(logits) along the last dim. logits [..., d_vocab] -> [...]. Pure.
    Uses log_softmax for numerical stability: H = -sum p * logp. Upcast to float32 first. RE-IMPLEMENTED
    (not imported) from controls/entropy_neuron_gemma2.py verbatim: on the box the controls are scp'd FLAT
    into latent_verify/, so `from controls.entropy_neuron_gemma2 import` would not resolve, and
    re-implementing keeps --selftest standalone on CPU with nothing else on sys.path (same convention as
    controls/entropy_distributed_presoftcap.py)."""
    logits = logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


def l1_change(p_from, p_to):
    """Total variation x2 = sum_v |p_to(v) - p_from(v)|, the L1 mass movement of the distribution. Pure
    (1-D probability tensors in, float out). This is the denominator of target_frac and the off-regime
    perturbation magnitude."""
    return float((p_to.float() - p_from.float()).abs().sum())


def target_fraction(p_from, p_to, cid, aid):
    """Fraction of the L1 mass movement that lands on the {C, W*} pair:
        (|p_to(C)-p_from(C)| + |p_to(W*)-p_from(W*)|) / sum_v |p_to(v)-p_from(v)|.
    1.0 => the edit ONLY redistributes mass between C and W* (a targeted, competitor-level edit);
    ~2/V => the edit spreads mass across the whole vocabulary (a broad overlay edit). Pure; L1~0 -> 0.0."""
    pf, pt = p_from.float(), p_to.float()
    l1 = float((pt - pf).abs().sum())
    if l1 < 1e-12:
        return 0.0
    pair = float((pt[cid] - pf[cid]).abs() + (pt[aid] - pf[aid]).abs())
    return pair / l1


def delta_pair(p_from, p_to, cid, aid):
    """(delta_P_C, delta_P_Wstar) = signed change in P(C) and P(W*) under the edit. Pure."""
    pf, pt = p_from.float(), p_to.float()
    return float(pt[cid] - pf[cid]), float(pt[aid] - pf[aid])


def kl_total_and_pair(p_from, p_to, cid, aid, eps=1e-12):
    """KL(p_to || p_from) total, and the contribution of the {C, W*} terms only (kl_pair). Returns
    (kl_total, kl_pair, kl_pair_frac=kl_pair/kl_total). KL = sum_v p_to(v) * log(p_to(v)/p_from(v)); the
    pair restriction sums only the v in {C, W*}. Pure; kl_total~0 -> frac 0.0. (Per-coordinate KL terms
    can be negative; kl_pair_frac uses the signed pair sum over the signed total, matching the L1 framing
    of 'how much of the movement is on the competitor pair'.)"""
    pf = p_from.float().clamp_min(eps)
    pt = p_to.float().clamp_min(eps)
    terms = pt * (pt.log() - pf.log())
    kl_total = float(terms.sum())
    kl_pair = float(terms[cid] + terms[aid])
    frac = (kl_pair / kl_total) if abs(kl_total) > 1e-12 else 0.0
    return kl_total, kl_pair, frac


# --------------------------------------------------------------------------- pure decision
def decide_overlay(target_frac, offreg_ratio, rand_target_frac,
                   target_thr=TARGET_FRAC, offreg_floor=OFFREG_FLOOR, rand_gap=RAND_GAP):
    """MECHANISM_LIKE iff the ablation is TARGETED (target_frac >= target_thr) AND regime-SPECIFIC
    (offreg_ratio < offreg_floor) AND beats a matched random direction (target_frac - rand_target_frac
    >= rand_gap); else OVERLAY_LIKE. Pure over the measured numbers only. None inputs -> OVERLAY_LIKE
    (cannot establish a mechanism). No claim attached to either category."""
    if target_frac is None or offreg_ratio is None:
        return {"category": "OVERLAY_LIKE", "mechanism_like": False,
                "targeted": False, "specific": False, "beats_random": False,
                "target_frac": None, "offreg_ratio": None,
                "rand_target_frac": (round(rand_target_frac, 4) if rand_target_frac is not None else None),
                "msg": "insufficient measurement (target_frac or offreg_ratio is None) -> OVERLAY_LIKE."}
    targeted = target_frac >= target_thr
    specific = offreg_ratio < offreg_floor
    beats_random = (rand_target_frac is None) or ((target_frac - rand_target_frac) >= rand_gap)
    mech = targeted and specific and beats_random
    if mech:
        msg = (f"target_frac {target_frac:.3f} >= {target_thr} (mass-movement concentrated on the "
               f"{{C,W*}} pair) AND off-regime L1 ratio {offreg_ratio:.3f} < {offreg_floor} (the edit "
               f"barely perturbs off the caving regime) AND it beats a matched random direction "
               f"({'' if rand_target_frac is None else f'rand {rand_target_frac:.3f}, '}gap >= {rand_gap}) "
               f"-- the ablation is targeted and regime-specific.")
    else:
        why = []
        if not targeted:
            why.append(f"target_frac {target_frac:.3f} < {target_thr} (broad mass movement)")
        if not specific:
            why.append(f"off-regime L1 ratio {offreg_ratio:.3f} >= {offreg_floor} (perturbs off-regime)")
        if not beats_random:
            why.append(f"does not beat the random direction by {rand_gap} "
                       f"(rand {None if rand_target_frac is None else round(rand_target_frac, 3)})")
        msg = "OVERLAY_LIKE: " + "; ".join(why) + "."
    return {"category": "MECHANISM_LIKE" if mech else "OVERLAY_LIKE", "mechanism_like": bool(mech),
            "targeted": bool(targeted), "specific": bool(specific), "beats_random": bool(beats_random),
            "target_frac": round(target_frac, 4), "offreg_ratio": round(offreg_ratio, 4),
            "rand_target_frac": (round(rand_target_frac, 4) if rand_target_frac is not None else None),
            "msg": msg}


# --------------------------------------------------------------------------- residual collection (real)
def _rname(L):
    """resid_post hook name at layer L (same convention as headset_direction._rname)."""
    return f"blocks.{L}.hook_resid_post"


def _full_softmax(logits):
    """Full next-token probability vector P at the last position from a model's output logits (gemma-2's
    final-logit softcap is already applied inside the forward, so softmax(logits[0,-1]) is the realized
    next-token distribution). Returns a 1-D float tensor. Pure given the logits."""
    return torch.softmax(logits[0, -1].float(), dim=-1)


def _proj_edit_hook(u, target_proj):
    """Hook that, at the readout position, sets the resid_post u-projection to target_proj (additive shift
    along u): r += (target_proj - r.u) * u. `u` must already be on the model's device. (Same projection
    edit as confidence_vs_cave_direction._proj_edit_hook / headset_direction's necessity ablation.)"""
    def hook(r, hook, u=u, target_proj=target_proj):
        cur = float(r[0, -1].float() @ u)
        r[0, -1] = r[0, -1] + ((target_proj - cur) * u).to(r.dtype)
        return r
    return hook


def _collect(model, pool, device, is_chat, fit_layers):
    """One model: per pool item, collect at every fit layer the last-token resid_post under the NEUTRAL
    and COUNTER prompts (verbatim repo turn construction), plus the neutral and counter margins
    M = logp(C)-logp(W*) (so caving items can be selected). Forward-only; caches only the last-position
    resid_post per layer. First-token-collision items (cid==aid) skipped (margin meaningless)."""
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

        names = [_rname(L) for L in fit_layers]
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


def _logits(model, ids, hooks=None):
    """Full last-position logits (optionally under fwd_hooks). Forward-only. Returns the model logits."""
    with torch.no_grad():
        return model.run_with_hooks(ids, fwd_hooks=hooks) if hooks else model(ids)


def _necessity_layer(model, recs, cave_pos, L, u_cave, proj_n):
    """In-sample cave-NECESSITY at layer L over the caving items (used ONLY to pick the headline layer):
    ablate the u_cave-projection on each caving COUNTER residual to proj_n and read the margin recovery
    frac = (M_ablate - M_ctr) / (M_neu - M_ctr). Mean over caving items. Same necessity formula as
    headset_direction / cave_direction_heldout. Forward-only."""
    from rlhf_differential import _logp_diff
    fr = []
    for k in cave_pos:
        r = recs[k]
        gap = r["M_neu"] - r["M_ctr"]
        if abs(gap) < 1e-6:
            continue
        h = [(_rname(L), _proj_edit_hook(u_cave, proj_n))]
        M_ab = float(_logp_diff(_logits(model, r["counter"], hooks=h), r["cid"], r["aid"]))
        fr.append((M_ab - r["M_ctr"]) / gap)
    return statistics.mean(fr) if fr else None


def _measure_model(name, is_chat, device, fit_layers, pool):
    """One model end-to-end. Collect residuals on the WIDE pool; identify caving items (counter lowers the
    margin from neutral by >= MIN_EFFECT_NET) and off-regime (non-caving) items. Per layer, fit u_cave on
    the TRAIN fold of caving items and measure in-sample necessity; pick the HEADLINE layer = max necessity.
    At the headline layer, on the held-out TEST caving items, measure (1) targetedness of the u_cave
    ablation on COUNTER (full-softmax change, target_frac, delta_P_C/W*, KL), (2) off-regime L1 (same
    ablation on the caving NEUTRAL prompt + on non-caving control items), and (3) a matched random-direction
    control. Returns a dict."""
    from transformer_lens import HookedTransformer
    from rlhf_differential import MIN_EFFECT_NET
    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    recs = _collect(model, pool, device, is_chat, fit_layers)
    n = len(recs)
    g = torch.Generator(device="cpu").manual_seed(RAND_SEED)

    # caving items (this model): counter lowers the margin from neutral by >= MIN_EFFECT_NET.
    cave_pos = [k for k, r in enumerate(recs) if (r["M_neu"] - r["M_ctr"]) >= MIN_EFFECT_NET]
    # off-regime = items this model does NOT cave on (the ablation should perturb little here).
    off_pos = [k for k, r in enumerate(recs) if k not in cave_pos]

    out = {"name": name, "n_ok": n, "n_cave": len(cave_pos), "n_off": len(off_pos), "layers": {}}

    if len(cave_pos) < 2:
        print(f"  [{name}] fewer than 2 caving items ({len(cave_pos)}); cannot fit/hold-out a direction.",
              flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        out["headline_layer"] = None
        out["decision"] = decide_overlay(None, None, None)
        return out

    # held-out fold over the CAVING items: fit u_cave on TRAIN, evaluate targetedness on TEST.
    ctr_tr, ctr_te = split_indices(len(cave_pos), SPLIT_SEED)
    cave_tr = [cave_pos[j] for j in ctr_tr]
    cave_te = [cave_pos[j] for j in ctr_te]

    # ---------------- per-layer u_cave fit (TRAIN) + in-sample necessity (headline selection) ----------
    per_layer = {}
    for L in fit_layers:
        Rc = torch.stack([recs[k]["rc"][L] for k in cave_tr]).to(device)
        Rn = torch.stack([recs[k]["rn"][L] for k in cave_tr]).to(device)
        u_cave = unit(diff_of_means(Rc, Rn))
        proj_n = statistics.mean(float(recs[k]["rn"][L].to(device) @ u_cave) for k in cave_tr)
        # matched random unit direction; target it to its OWN train neutral mean projection (matched magnitude)
        rnd = torch.randn(u_cave.shape, generator=g).to(u_cave.dtype).to(device)
        u_rand = unit(rnd)
        proj_n_rand = statistics.mean(float(recs[k]["rn"][L].to(device) @ u_rand) for k in cave_tr)
        nec = _necessity_layer(model, recs, cave_pos, L, u_cave, proj_n)
        per_layer[L] = {"u_cave": u_cave, "proj_n": proj_n,
                        "u_rand": u_rand, "proj_n_rand": proj_n_rand, "necessity": nec}
        out["layers"][L] = {"in_sample_necessity": (round(nec, 4) if nec is not None else None)}
        print(f"  [{'it' if is_chat else 'base'} L{L}] in-sample cave-necessity="
              f"{out['layers'][L]['in_sample_necessity']}", flush=True)

    valid = [L for L in fit_layers if per_layer[L]["necessity"] is not None]
    headline = max(valid, key=lambda L: per_layer[L]["necessity"]) if valid else None
    out["headline_layer"] = headline
    if headline is None:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        out["decision"] = decide_overlay(None, None, None)
        return out

    pl = per_layer[headline]
    u_cave, proj_n = pl["u_cave"], pl["proj_n"]
    u_rand, proj_n_rand = pl["u_rand"], pl["proj_n_rand"]

    # ---------------- (1) TARGETEDNESS on TEST caving COUNTER items ----------------
    tfracs, dPC, dPW, l1_on, kl_tot, kl_pair_frac = [], [], [], [], [], []
    rt_fracs, rl1_on = [], []
    for k in cave_te:
        r = recs[k]
        cid, aid = r["cid"], r["aid"]
        P0 = _full_softmax(_logits(model, r["counter"]))
        # u_cave ablation: set the counter u_cave-projection to the TRAIN neutral mean
        h = [(_rname(headline), _proj_edit_hook(u_cave, proj_n))]
        P1 = _full_softmax(_logits(model, r["counter"], hooks=h))
        tfracs.append(target_fraction(P0, P1, cid, aid))
        dc, dw = delta_pair(P0, P1, cid, aid)
        dPC.append(dc); dPW.append(dw)
        l1_on.append(l1_change(P0, P1))
        kt, _kp, kpf = kl_total_and_pair(P0, P1, cid, aid)
        kl_tot.append(kt); kl_pair_frac.append(kpf)
        # matched random direction on the same counter item
        hr = [(_rname(headline), _proj_edit_hook(u_rand, proj_n_rand))]
        Pr = _full_softmax(_logits(model, r["counter"], hooks=hr))
        rt_fracs.append(target_fraction(P0, Pr, cid, aid))
        rl1_on.append(l1_change(P0, Pr))

    # ---------------- (2) OFF-REGIME SPECIFICITY ----------------
    # (a) caving items, NEUTRAL prompt (no pushback present); (b) non-caving control items, COUNTER prompt.
    off_l1, off_l1_neutral, off_l1_noncave = [], [], []
    rand_off_l1 = []
    for k in cave_te:                                   # (a) same caving items, neutral condition
        r = recs[k]
        P0 = _full_softmax(_logits(model, r["neutral"]))
        h = [(_rname(headline), _proj_edit_hook(u_cave, proj_n))]
        P1 = _full_softmax(_logits(model, r["neutral"], hooks=h))
        off_l1_neutral.append(l1_change(P0, P1)); off_l1.append(l1_change(P0, P1))
        hr = [(_rname(headline), _proj_edit_hook(u_rand, proj_n_rand))]
        Pr = _full_softmax(_logits(model, r["neutral"], hooks=hr))
        rand_off_l1.append(l1_change(P0, Pr))
    for k in off_pos:                                   # (b) non-caving control items, counter condition
        r = recs[k]
        P0 = _full_softmax(_logits(model, r["counter"]))
        h = [(_rname(headline), _proj_edit_hook(u_cave, proj_n))]
        P1 = _full_softmax(_logits(model, r["counter"], hooks=h))
        off_l1_noncave.append(l1_change(P0, P1)); off_l1.append(l1_change(P0, P1))
        hr = [(_rname(headline), _proj_edit_hook(u_rand, proj_n_rand))]
        Pr = _full_softmax(_logits(model, r["counter"], hooks=hr))
        rand_off_l1.append(l1_change(P0, Pr))

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    def _mean(xs):
        return statistics.mean(xs) if xs else None

    on_l1_mean = _mean(l1_on)
    off_l1_mean = _mean(off_l1)
    offreg_ratio = (off_l1_mean / on_l1_mean) if (on_l1_mean and on_l1_mean > 1e-9) else None
    target_frac = _mean(tfracs)
    rand_target_frac = _mean(rt_fracs)

    headline_stats = {
        "n_test_cave": len(cave_te), "n_train_cave": len(cave_tr), "n_off": len(off_pos),
        "target_frac": (round(target_frac, 4) if target_frac is not None else None),
        "delta_P_C": (round(_mean(dPC), 6) if dPC else None),
        "delta_P_Wstar": (round(_mean(dPW), 6) if dPW else None),
        "on_regime_L1": (round(on_l1_mean, 6) if on_l1_mean is not None else None),
        "off_regime_L1": (round(off_l1_mean, 6) if off_l1_mean is not None else None),
        "off_regime_L1_neutral": (round(_mean(off_l1_neutral), 6) if off_l1_neutral else None),
        "off_regime_L1_noncave": (round(_mean(off_l1_noncave), 6) if off_l1_noncave else None),
        "offreg_ratio": (round(offreg_ratio, 4) if offreg_ratio is not None else None),
        "kl_total": (round(_mean(kl_tot), 6) if kl_tot else None),
        "kl_pair_frac": (round(_mean(kl_pair_frac), 4) if kl_pair_frac else None),
        "rand_target_frac": (round(rand_target_frac, 4) if rand_target_frac is not None else None),
        "rand_on_regime_L1": (round(_mean(rl1_on), 6) if rl1_on else None),
        "rand_off_regime_L1": (round(_mean(rand_off_l1), 6) if rand_off_l1 else None),
    }
    out["layers"][headline].update(headline_stats)
    out["headline_stats"] = headline_stats
    out["decision"] = decide_overlay(target_frac, offreg_ratio, rand_target_frac)
    return out


def run(name_base, name_it, tag, device, pool):
    from headset_direction import FIT_LAYERS         # reuse the layer sweep
    res = {"base": _measure_model(name_base, False, device, FIT_LAYERS, pool),
           "it": _measure_model(name_it, True, device, FIT_LAYERS, pool)}
    out = {
        "name_base": name_base, "name_it": name_it, "device": device, "tag": tag,
        "cue": "cave_direction_overlay", "pool_size": len(pool),
        "fit_layers": FIT_LAYERS,
        "metric": "M = logp(C) - logp(W*) first-token margin (caving selection); full next-token softmax change (overlay test)",
        "thresholds": {"TARGET_FRAC": TARGET_FRAC, "OFFREG_FLOOR": OFFREG_FLOOR, "RAND_GAP": RAND_GAP,
                       "SPLIT_SEED": SPLIT_SEED, "RAND_SEED": RAND_SEED},
        "decision_rule": DECISION_RULE,
        "base": res["base"], "it": res["it"],
    }
    Path("out").mkdir(exist_ok=True)
    Path(f"out/cave_direction_overlay_{tag}.json").write_text(json.dumps(out, indent=2, default=str))
    for m in MODELS:
        dd = res[m]["decision"]
        hs = res[m].get("headline_stats") or {}
        print(f"[{m}] {dd['category']} (L{res[m].get('headline_layer')}) "
              f"target_frac={dd.get('target_frac')} dP(C)={hs.get('delta_P_C')} "
              f"dP(W*)={hs.get('delta_P_Wstar')} offreg_ratio={dd.get('offreg_ratio')} "
              f"rand={dd.get('rand_target_frac')}", flush=True)
    print(f"[done] wrote out/cave_direction_overlay_{tag}.json", flush=True)


# --------------------------------------------------------------------------- selftest (model-free, CPU, no model load)
def _onehot(V, idx, p):
    """Probability vector of length V with mass `p` on idx and (1-p) spread uniformly over the rest.
    Pure (selftest only)."""
    q = torch.full((V,), (1.0 - p) / (V - 1))
    q[idx] = p
    return q


def selftest():
    torch.manual_seed(0)
    V = 1000
    cid, aid = 3, 7                                     # C index, W* index

    # ---------- split primitive: real partition, both folds non-empty, deterministic ----------
    tr, te = split_indices(10, SPLIT_SEED)
    assert set(tr) | set(te) == set(range(10)) and not (set(tr) & set(te)), (tr, te)
    assert tr and te and split_indices(10, SPLIT_SEED) == split_indices(10, SPLIT_SEED)
    print(f"[selftest] split: train={tr} test={te} (disjoint, exhaustive, deterministic)")

    # ---------- l1_change / delta_pair / target_fraction / KL arithmetic ----------
    p_from = _onehot(V, aid, 0.6)                       # mass mostly on W* (the caved counter state)
    # (A) TARGETED edit: ablation moves mass ONLY between C and W* (a competitor-level edit)
    p_to_targeted = p_from.clone()
    moved = 0.4
    p_to_targeted[aid] -= moved                          # W* loses
    p_to_targeted[cid] += moved                          # C gains the same mass
    l1 = l1_change(p_from, p_to_targeted)
    assert abs(l1 - 2 * moved) < 1e-6, l1                # L1 = |dC| + |dW*| = 2*moved
    dc, dw = delta_pair(p_from, p_to_targeted, cid, aid)
    assert abs(dc - moved) < 1e-6 and abs(dw + moved) < 1e-6, (dc, dw)
    tf_targeted = target_fraction(p_from, p_to_targeted, cid, aid)
    assert abs(tf_targeted - 1.0) < 1e-6, tf_targeted    # ALL the movement is on {C, W*} -> target_frac ~1
    kt, kp, kpf = kl_total_and_pair(p_from, p_to_targeted, cid, aid)
    assert abs(kpf - 1.0) < 1e-6, (kt, kp, kpf)          # all KL movement on the pair too
    assert kt > 0, kt                                    # KL is non-negative
    print(f"[selftest] TARGETED edit: L1={l1:.3f} dP(C)={dc:+.3f} dP(W*)={dw:+.3f} "
          f"target_frac={tf_targeted:.3f} kl_pair_frac={kpf:.3f}")

    # (B) BROAD overlay edit: ablation spreads mass across MANY tokens (not the competitor pair)
    p_to_broad = p_from.clone()
    spread = 0.4
    p_to_broad[aid] -= spread                            # W* loses 'spread'
    others = [j for j in range(V) if j not in (cid, aid)]
    p_to_broad[others] += spread / len(others)           # ...redistributed across the rest of the vocab
    tf_broad = target_fraction(p_from, p_to_broad, cid, aid)
    # only W*'s drop touches the pair; C is unchanged; the rest of L1 is the broad spread -> low target_frac
    assert tf_broad < 0.6, tf_broad
    # L1~0 -> target_frac 0.0 (no div-by-zero); identical distributions -> KL 0
    assert target_fraction(p_from, p_from.clone(), cid, aid) == 0.0
    assert kl_total_and_pair(p_from, p_from.clone(), cid, aid)[2] == 0.0
    print(f"[selftest] BROAD edit: target_frac={tf_broad:.3f} (< {TARGET_FRAC}) -> spread across vocab")

    # ---------- DECISION boundaries ----------
    # MECHANISM_LIKE: targeted (>=0.5) AND specific (offreg_ratio < 0.10) AND beats random by >= RAND_GAP
    dm = decide_overlay(target_frac=0.92, offreg_ratio=0.03, rand_target_frac=0.10)
    assert dm["category"] == "MECHANISM_LIKE" and dm["mechanism_like"], dm
    assert dm["targeted"] and dm["specific"] and dm["beats_random"], dm
    # OVERLAY (broad): low target_frac even if specific + beats random
    do_broad = decide_overlay(target_frac=0.20, offreg_ratio=0.02, rand_target_frac=0.02)
    assert do_broad["category"] == "OVERLAY_LIKE" and not do_broad["targeted"], do_broad
    # OVERLAY (non-specific): targeted on-regime but off-regime perturbed AS MUCH as on-regime
    do_offreg = decide_overlay(target_frac=0.90, offreg_ratio=1.00, rand_target_frac=0.05)
    assert do_offreg["category"] == "OVERLAY_LIKE" and not do_offreg["specific"], do_offreg
    # OVERLAY (no better than random): targeted + specific but the random direction is just as targeted
    do_rand = decide_overlay(target_frac=0.90, offreg_ratio=0.02, rand_target_frac=0.85)
    assert do_rand["category"] == "OVERLAY_LIKE" and not do_rand["beats_random"], do_rand
    # exact threshold behaviour
    assert decide_overlay(TARGET_FRAC, OFFREG_FLOOR - 1e-6, 0.0)["category"] == "MECHANISM_LIKE"
    assert decide_overlay(TARGET_FRAC - 1e-6, OFFREG_FLOOR - 1e-6, 0.0)["category"] == "OVERLAY_LIKE"
    assert decide_overlay(TARGET_FRAC, OFFREG_FLOOR, 0.0)["category"] == "OVERLAY_LIKE"   # ratio==floor fails (strict <)
    # None inputs -> OVERLAY_LIKE (cannot establish a mechanism)
    assert decide_overlay(None, 0.02, 0.0)["category"] == "OVERLAY_LIKE"
    assert decide_overlay(0.9, None, 0.0)["category"] == "OVERLAY_LIKE"
    print("[selftest] decision: MECHANISM_LIKE / OVERLAY(broad) / OVERLAY(non-specific) / "
          "OVERLAY(=random) / thresholds / None all fire")

    # ---------- END-TO-END synthetic on/off-regime contrast (no model) ----------
    # ON-regime: targeted edit (target_frac ~1, on-regime L1 = 0.8).
    on_p0 = _onehot(V, aid, 0.6)
    on_p1 = on_p0.clone(); on_p1[aid] -= 0.4; on_p1[cid] += 0.4
    on_tf = target_fraction(on_p0, on_p1, cid, aid)
    on_l1 = l1_change(on_p0, on_p1)
    # MECHANISM off-regime: SAME edit barely moves the off-regime distribution (off L1 = 0.02 << on)
    off_p0 = _onehot(V, aid, 0.6)
    off_p1_mech = off_p0.clone(); off_p1_mech[aid] -= 0.01; off_p1_mech[cid] += 0.01
    off_l1_mech = l1_change(off_p0, off_p1_mech)
    ratio_mech = off_l1_mech / on_l1
    rand_tf_mech = 0.05
    d_mech = decide_overlay(on_tf, ratio_mech, rand_tf_mech)
    assert d_mech["category"] == "MECHANISM_LIKE", (on_tf, ratio_mech, d_mech)
    # OVERLAY off-regime: SAME edit perturbs the off-regime distribution AS MUCH as on-regime (off L1 ~ on)
    off_p1_ov = off_p0.clone(); off_p1_ov[aid] -= 0.4; off_p1_ov[cid] += 0.4
    off_l1_ov = l1_change(off_p0, off_p1_ov)
    ratio_ov = off_l1_ov / on_l1
    d_ov = decide_overlay(on_tf, ratio_ov, rand_tf_mech)
    assert d_ov["category"] == "OVERLAY_LIKE" and not d_ov["specific"], (ratio_ov, d_ov)
    print(f"[selftest] end-to-end: on_tf={on_tf:.3f} on_L1={on_l1:.3f} | "
          f"MECH off_L1={off_l1_mech:.3f} ratio={ratio_mech:.3f}->{d_mech['category']} | "
          f"OVERLAY off_L1={off_l1_ov:.3f} ratio={ratio_ov:.3f}->{d_ov['category']}")

    # ---------- entropy_of_logits sanity (reused readout): uniform = log V, peaked ~ 0 ----------
    import math as _m
    H_uniform = float(entropy_of_logits(torch.zeros(V)))
    assert abs(H_uniform - _m.log(V)) < 1e-4, H_uniform
    H_peaked = float(entropy_of_logits(torch.tensor([0.0, 50.0] + [0.0] * (V - 2))))
    assert H_peaked < 1e-3, H_peaked
    print(f"[selftest] entropy_of_logits: uniform={H_uniform:.3f} (=log {V}) peaked={H_peaked:.5f} (~0)")
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
