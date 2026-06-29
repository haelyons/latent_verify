"""DOSE-RESPONSE of ADDING a fitted residual direction, split by whether the item has a plausible
alternative, separating a CONFIDENCE/ENTROPY change from an ANSWER-IDENTITY (argmax) change (neutral
measurement).

CONTEXT (neutral). The repo fits a rank-1 diff-of-means 'defer' direction u in the residual stream
(cave_defer_direction.fit_defer / headset_direction._dir_pass: u = unit(mean_items(resid_post[L][-1](COUNTER)
- resid_post[L][-1](NEUTRAL))), the answer-slot resid_post[L][-1] convention) at L = n_layers//2, and ADDS a
scaled multiple of u at the residual stream (the steering/ADD convention from cave_defer_direction
_all_layer_add_hooks + scale9b_dose_response): r := r + amount * u. A prior control flagged that ADD in
resid-norm/nat units of order alpha 4-8 over-steers and COLLAPSES the realized answer (P(yes) collapse), so
the dose is capped. This control runs ONE such ADD dose-response and reads TWO orthogonal quantities at each
dose -- the ENTROPY of the answer-slot next-token distribution (the confidence readout) and the ARGMAX
identity (the answer-identity readout) -- SEPARATELY on items that DO vs DO NOT have a plausible alternative.
It attaches no hypothesis to any direction, dose, item-group, or readout; it measures the curves and lets the
numbers fall where they do.

WHAT IT MEASURES (gemma-2 BASE by default; --name/--tag; --big-pool for n; QA template; --chat for -it):
  (a) POOL + PLAUSIBILITY GATE SPLIT. _build_pool (incl. --big-pool). For each pool item read the single-turn
      answer-slot first-token log-probs logP(C) and logP(W*) (and logP(W2*), the second-ranked rival).
      The plausibility gate is the job_truthful_flip select_items condition (re-implemented inline):
        |logP(C) - logP(W*)| < MARGIN_KEEP(1.5) nats  AND  rho = exp(logP(W*) - logP(W2*)) > RHO_MIN(2.0).
      HAS_ALT = the gate is satisfied (near-margin, one dominant rival -> a plausible alternative exists).
      NO_ALT  = high-confidence with no plausible rival: |logP(C) - logP(W*)| >= NOALT_MARGIN(3.0) nats AND
        (rho <= RHO_MIN OR no second rival), i.e. the model is far from its margin and not torn between two
        answers. (Items in neither band -- the middle -- are dropped; they are neither a clean plausible-alt
        item nor a clean high-confidence item.) First-token-collision items (C-first-tok == W*-first-tok) are
        skipped (the argmax-identity readout would be degenerate).
  (b) FIT u. At L_FIT = n_layers//2 fit u = unit(mean(resid_post[L_FIT][-1](COUNTER) - resid_post[L_FIT][-1]
      (NEUTRAL))) over the HAS_ALT caving items (COUNTER = push(q,C,PUSH['counter'].format(W=W)); NEUTRAL =
      push(q,C,NEUTRAL)) -- the cave_defer_direction.fit_defer construction. proj_unit = the mean caved-held
      u-projection magnitude at L_FIT (the natural caved-held separation; the resid-norm/nat unit the dose
      multiplies, exactly as cave_defer_direction's ADD).
  (c) DOSE-RESPONSE. For each item (in BOTH groups), at each dose alpha in ALPHAS (capped to avoid the
      over-steer collapse the prior control flagged), ADD alpha * proj_unit * u at L_FIT across ALL positions
      on the item's NEUTRAL run (the steering/ADD convention), and read the answer-slot next-token softmax:
        delta_entropy   = entropy(steered) - entropy(unsteered)        (the CONFIDENCE readout, nats)
        argmax_flip     = (steered argmax != unsteered argmax) AND (steered argmax == W*-first-tok OR != the
                          item's unsteered argmax)                      (the IDENTITY readout; W* OR any
                          non-original answer -- a flip away from the original answer)
      Aggregated PER (group x alpha): mean delta_entropy and argmax_flip_rate, with n per group. Both full
      dose-response curves (HAS_ALT, NO_ALT) reported.

NEUTRAL DECISION (module constants; numbers + categories only; no hypothesis named, nothing said about which
direction/dose/group/readout supports any claim). alpha* = the SMALLEST alpha at which HAS_ALT
argmax_flip_rate >= FLIP_THR (None if it never reaches FLIP_THR).
  NO_EFFECT            iff NO_ALT delta_entropy < ENT_THR at ALL alpha                    (checked FIRST).
  FLIP_ANYWHERE        iff alpha* exists AND NO_ALT argmax_flip_rate >= FLIP_THR at alpha*.
  DOUBT_WITHOUT_FLIP   iff alpha* exists AND at alpha* NO_ALT delta_entropy >= ENT_THR AND NO_ALT
                          argmax_flip_rate < FLIP_THR.
  UNRESOLVED           iff none of the above (e.g. NO_ALT moves entropy somewhere but alpha* never reached, or
                          at alpha* NO_ALT delta_entropy < ENT_THR with flip < FLIP_THR).
  Resolution order: NO_EFFECT -> FLIP_ANYWHERE -> DOUBT_WITHOUT_FLIP -> UNRESOLVED. Thresholds inclusive (>=).
  Reported: alpha*, the full per-(group,alpha) curves (delta_entropy + argmax_flip_rate + n), L_FIT,
  proj_unit, the category.

Model-free --selftest (CPU, NO model load): synthetic dose-response arrays exercising DOUBT_WITHOUT_FLIP /
FLIP_ANYWHERE / NO_EFFECT / UNRESOLVED + the alpha* selection + the inclusive >= boundaries + the entropy /
argmax-flip helpers + the plausibility-gate split. torch is imported INSIDE the real-run fns; the pure
helpers + decision run standalone on CPU (the same FLAT-scp convention the sibling controls use -- on the box
every file is scp'd flat into latent_verify/).

transformer_lens ONLY, forward-only (resid_post ADD hooks + full-softmax readouts; no backward), bf16, one
model resident then freed; --big-pool needs `datasets`. Writes results_calib/out/cave_dir_doubt_injection_
{tag}.json.

  python controls/cave_dir_doubt_injection.py --selftest
  python controls/cave_dir_doubt_injection.py --name google/gemma-2-9b --tag 9b_base --device cuda --big-pool
"""
import argparse
import json
import math
import statistics
from pathlib import Path

# Pre-registered constants (neutral: stated on the measured numbers only).
MARGIN_KEEP = 1.5      # |logP(C)-logP(W*)| below this nats == near-margin (HAS_ALT band); job_truthful_flip
RHO_MIN = 2.0          # rho = P(W*)/P(W2*) above this == ONE dominant rival (HAS_ALT band); job_truthful_flip
NOALT_MARGIN = 3.0     # |logP(C)-logP(W*)| at/above this nats == high-confidence (NO_ALT band: far from margin)
MIN_PER_GROUP = 4      # below this many items in a group the curve is under-powered (still reported)

ALPHAS = [2, 4, 8, 16]  # ADD doses in resid-norm/nat units (alpha * proj_unit * u at L_FIT); capped at 16 to
                        # bracket the over-steer collapse the prior control flagged at resid-norm alpha 4-8.

FLIP_THR = 0.3         # argmax_flip_rate at/above this counts as an IDENTITY change in a group
ENT_THR = 0.3          # delta_entropy (nats) at/above this counts as a CONFIDENCE change in a group

DECISION_RULE = (
    "Plausibility gate split (job_truthful_flip select_items condition): HAS_ALT iff |logP(C)-logP(W*)| < "
    "MARGIN_KEEP(1.5) nats AND rho=P(W*)/P(W2*) > RHO_MIN(2.0) (near-margin, one dominant rival -> a plausible "
    "alternative exists); NO_ALT iff |logP(C)-logP(W*)| >= NOALT_MARGIN(3.0) nats AND (rho <= RHO_MIN OR no "
    "second rival) (high-confidence, no plausible rival); middle band dropped; first-token-collision items "
    "skipped. Fit u = unit(mean(resid_post[L_FIT][-1](COUNTER) - resid_post[L_FIT][-1](NEUTRAL))) over the "
    "HAS_ALT caving items at L_FIT=n_layers//2 (cave_defer_direction.fit_defer); proj_unit = mean caved-held "
    "u-projection magnitude at L_FIT. DOSE: ADD alpha*proj_unit*u at L_FIT across all positions on the NEUTRAL "
    "run (alpha in {2,4,8,16}); per item read the answer-slot next-token softmax: delta_entropy = "
    "entropy(steered)-entropy(unsteered) (CONFIDENCE), argmax_flip = (steered argmax != unsteered argmax) AND "
    "(steered argmax == W*-first-tok OR != the unsteered argmax) (IDENTITY). Aggregate per (group x alpha): "
    "mean delta_entropy + argmax_flip_rate (n per group). alpha* = smallest alpha with HAS_ALT "
    "argmax_flip_rate >= FLIP_THR(0.3). NO_EFFECT iff NO_ALT delta_entropy < ENT_THR(0.3) at ALL alpha; else "
    "FLIP_ANYWHERE iff alpha* exists AND NO_ALT argmax_flip_rate >= FLIP_THR at alpha*; else "
    "DOUBT_WITHOUT_FLIP iff alpha* exists AND at alpha* NO_ALT delta_entropy >= ENT_THR AND NO_ALT "
    "argmax_flip_rate < FLIP_THR; else UNRESOLVED. Thresholds inclusive (>=); numbers + categories only, no "
    "claim attached to any direction, dose, group, or readout."
)


# --------------------------------------------------------------------------- pure helpers (selftest-able)
def plausibility_group(lp_c, lp_w, lp_w2, margin_keep=MARGIN_KEEP, rho_min=RHO_MIN, noalt_margin=NOALT_MARGIN):
    """Assign ONE item to a plausibility group from its single-turn first-token log-probs (nats):
      lp_c  = logP(C-first-tok)
      lp_w  = logP(W*-first-tok)  (the top-ranked WRONG rival)
      lp_w2 = logP(W2*-first-tok) (the second-ranked rival; None if no second rival)
    margin = lp_c - lp_w; rho = exp(lp_w - lp_w2) (inf if lp_w2 is None == no second rival).
      'HAS_ALT' iff |margin| < margin_keep AND rho > rho_min  (near-margin, one dominant rival; the
                  job_truthful_flip select_items condition -- a plausible alternative exists).
      'NO_ALT'  iff |margin| >= noalt_margin AND rho <= rho_min  (high-confidence, far from margin, NOT torn
                  between two answers; no plausible rival). rho<=rho_min is ALSO satisfied when there is no
                  second rival? No -- with no second rival rho=inf > rho_min, so a no-second-rival item lands in
                  NO_ALT only via the explicit lp_w2-None branch below (high-confidence single-answer item).
      'MIDDLE'  otherwise (dropped: neither a clean plausible-alt nor a clean high-confidence item).
    Pure (floats|None -> str). NOTE the NO_ALT 'no plausible rival' is read two ways: rho<=rho_min (a flat,
    non-dominant rival field) OR lp_w2 is None (only one rival modeled); both, combined with the far-margin
    requirement, mean the model is confident and not torn -- exactly the spec's NO_ALT."""
    margin = lp_c - lp_w
    am = abs(margin)
    if lp_w2 is None:
        rho = float("inf")
        no_rival = True            # only one rival modeled -> no SECOND plausible rival
    else:
        rho = math.exp(lp_w - lp_w2)
        no_rival = rho <= rho_min  # rival field is flat / W* not dominant -> not torn between two answers
    if am < margin_keep and rho > rho_min:
        return "HAS_ALT"
    if am >= noalt_margin and no_rival:
        return "NO_ALT"
    return "MIDDLE"


def entropy_from_probs(p):
    """Shannon entropy (nats) of a probability vector `p` (1-D tensor) = -sum p*log p, computed in float with
    a log-clamp for stability. Mirrors entropy_neuron_gemma2.entropy_of_logits applied to a softmax. Pure
    (tensor -> float). (The real run reads entropy from logits via entropy_of_logits; this CPU form is used in
    --selftest on planted distributions.)"""
    import torch
    pp = p.float().clamp_min(1e-12)
    return float(-(pp * pp.log()).sum())


def argmax_flip(amx_steered, amx_unsteered, aid):
    """IDENTITY readout for ONE item: did the ADD change the argmax AWAY from the unsteered answer?
      True iff amx_steered != amx_unsteered AND (amx_steered == aid (== W*-first-tok) OR amx_steered !=
      amx_unsteered). The second clause is always true once the argmax changed, so this is: the argmax
      changed to W* OR to ANY non-original token. False if the argmax is unchanged. Pure (ints -> bool).
    (Written as the spec states it -- 'changes to W* or to any non-original answer' -- so the W* case is named
    explicitly even though a change to W* is itself a change to a non-original token.)"""
    if amx_steered == amx_unsteered:
        return False
    return (amx_steered == aid) or (amx_steered != amx_unsteered)


def _mean(xs):
    """Mean of `xs`, or None if empty. Pure."""
    return statistics.mean(xs) if xs else None


def aggregate_group(per_item_delta_ent, per_item_flip):
    """Per (group x alpha) aggregate: mean delta_entropy + argmax_flip_rate over a group's items at one alpha.
      per_item_delta_ent : list of delta_entropy floats (one per item).
      per_item_flip      : list of bools (one per item; the argmax_flip readout).
    Returns {'delta_entropy', 'argmax_flip_rate', 'n'} (None means / 0.0 rate on empty). Pure."""
    n = len(per_item_flip)
    return {
        "delta_entropy": (round(_mean(per_item_delta_ent), 6) if per_item_delta_ent else None),
        "argmax_flip_rate": (round(sum(1 for b in per_item_flip if b) / n, 6) if n else 0.0),
        "n": n,
    }


def alpha_star(has_alt_curve, alphas=ALPHAS, flip_thr=FLIP_THR):
    """alpha* = the SMALLEST alpha (in `alphas` order) at which the HAS_ALT argmax_flip_rate >= flip_thr.
    `has_alt_curve` = {alpha: {'argmax_flip_rate': float, ...}}. Returns the alpha (int) or None if no alpha
    reaches flip_thr. Inclusive (>=). Pure."""
    for a in alphas:
        cell = has_alt_curve.get(a) or {}
        r = cell.get("argmax_flip_rate")
        if r is not None and r >= flip_thr:
            return a
    return None


# --------------------------------------------------------------------------- pure decision
def decide(has_alt_curve, no_alt_curve, alphas=ALPHAS, flip_thr=FLIP_THR, ent_thr=ENT_THR):
    """Neutral 4-way decision over the measured dose-response curves only (no claim attached). Curves are
    {alpha: {'delta_entropy': float|None, 'argmax_flip_rate': float, 'n': int}}.
      alpha* = smallest alpha with HAS_ALT argmax_flip_rate >= flip_thr.
      NO_EFFECT          iff NO_ALT delta_entropy < ent_thr at ALL alpha            (checked FIRST).
      FLIP_ANYWHERE      iff alpha* exists AND NO_ALT argmax_flip_rate >= flip_thr at alpha*.
      DOUBT_WITHOUT_FLIP iff alpha* exists AND at alpha* NO_ALT delta_entropy >= ent_thr AND NO_ALT
                            argmax_flip_rate < flip_thr.
      UNRESOLVED         iff none of the above.
    Resolution order: NO_EFFECT -> FLIP_ANYWHERE -> DOUBT_WITHOUT_FLIP -> UNRESOLVED. Thresholds inclusive
    (>=). Pure (dicts -> dict)."""
    def _r(x):
        return round(float(x), 6) if x is not None else None

    astar = alpha_star(has_alt_curve, alphas, flip_thr)

    # NO_EFFECT: NO_ALT delta_entropy below ent_thr at EVERY alpha (a None entropy is treated as below thr).
    noalt_ents = []
    any_ent_at_or_above = False
    for a in alphas:
        de = (no_alt_curve.get(a) or {}).get("delta_entropy")
        noalt_ents.append(de)
        if de is not None and de >= ent_thr:
            any_ent_at_or_above = True

    # the NO_ALT cell at alpha* (if alpha* exists).
    noalt_at_star = (no_alt_curve.get(astar) or {}) if astar is not None else {}
    noalt_ent_star = noalt_at_star.get("delta_entropy")
    noalt_flip_star = noalt_at_star.get("argmax_flip_rate")

    if not any_ent_at_or_above:
        cat = "NO_EFFECT"
        msg = (f"NO_ALT delta_entropy < ENT_THR({ent_thr}) at ALL alpha {alphas}: adding u does not move the "
               f"confidence/entropy of the high-confidence (no-plausible-alt) items at any dose.")
    elif astar is not None and noalt_flip_star is not None and noalt_flip_star >= flip_thr:
        cat = "FLIP_ANYWHERE"
        msg = (f"at alpha* = {astar} (smallest alpha with HAS_ALT argmax_flip_rate >= FLIP_THR({flip_thr})), "
               f"NO_ALT argmax_flip_rate {noalt_flip_star:.3f} >= FLIP_THR: adding u flips the answer identity "
               f"of the high-confidence (no-plausible-alt) items too.")
    elif (astar is not None and noalt_ent_star is not None and noalt_ent_star >= ent_thr
          and (noalt_flip_star is None or noalt_flip_star < flip_thr)):
        cat = "DOUBT_WITHOUT_FLIP"
        msg = (f"at alpha* = {astar}, NO_ALT delta_entropy {noalt_ent_star:.3f} >= ENT_THR({ent_thr}) AND "
               f"NO_ALT argmax_flip_rate "
               f"{None if noalt_flip_star is None else round(noalt_flip_star, 3)} < FLIP_THR({flip_thr}): "
               f"adding u raises the entropy of the high-confidence (no-plausible-alt) items without changing "
               f"their argmax answer.")
    else:
        cat = "UNRESOLVED"
        if astar is None:
            why = (f"HAS_ALT argmax_flip_rate never reaches FLIP_THR({flip_thr}) at any alpha {alphas} (no "
                   f"alpha* to read NO_ALT at), though NO_ALT delta_entropy reaches ENT_THR somewhere")
        else:
            why = (f"at alpha* = {astar}, NO_ALT delta_entropy "
                   f"{None if noalt_ent_star is None else round(noalt_ent_star, 3)} < ENT_THR({ent_thr}) with "
                   f"NO_ALT argmax_flip_rate "
                   f"{None if noalt_flip_star is None else round(noalt_flip_star, 3)} < FLIP_THR({flip_thr})")
        msg = f"none of NO_EFFECT / FLIP_ANYWHERE / DOUBT_WITHOUT_FLIP holds: {why}."

    return {
        "category": cat,
        "alpha_star": astar,
        "noalt_delta_entropy_by_alpha": {str(a): _r((no_alt_curve.get(a) or {}).get("delta_entropy"))
                                         for a in alphas},
        "noalt_flip_rate_by_alpha": {str(a): _r((no_alt_curve.get(a) or {}).get("argmax_flip_rate"))
                                     for a in alphas},
        "hasalt_flip_rate_by_alpha": {str(a): _r((has_alt_curve.get(a) or {}).get("argmax_flip_rate"))
                                      for a in alphas},
        "noalt_delta_entropy_at_alpha_star": _r(noalt_ent_star),
        "noalt_flip_rate_at_alpha_star": _r(noalt_flip_star),
        "any_noalt_entropy_at_or_above_thr": bool(any_ent_at_or_above),
        "flip_thr": flip_thr, "ent_thr": ent_thr, "alphas": list(alphas),
        "msg": msg,
    }


# --------------------------------------------------------------------------- real-run helpers
def _rname(L):
    """resid_post hook name at layer L (headset_direction / cave_defer_direction convention)."""
    return f"blocks.{L}.hook_resid_post"


def fit_defer(rc_list, rn_list):
    """Diff-of-means 'defer' direction over aligned COUNTER/NEUTRAL answer-slot residual lists:
    u = unit(mean_i(rc_i - rn_i)) (cave_defer_direction.fit_defer / headset_direction._dir_pass). Pure (tensor
    lists in, unit tensor out)."""
    import torch
    D = torch.stack([rc_list[i] - rn_list[i] for i in range(len(rc_list))])     # [n, d]
    d = D.mean(0)
    return d / (d.norm() + 1e-8)


def _all_layer_add_at(L, u, amount):
    """ADD hook at blocks.{L}.hook_resid_post, ALL positions: r := r + amount * u (a unit direction scaled by
    the scalar `amount`). Returns [(hook_name, hook)] (single layer L_FIT, the spec's 'add alpha*u at L across
    positions'; mirrors cave_defer_direction._all_layer_add_hooks restricted to one layer). Forward-only."""
    def f(r, hook, uu=u, amt=amount):
        r[:] = r + amt * uu.to(r.dtype)
        return r
    return [(_rname(L), f)]


def _measure_model(name, is_chat, device, pool):
    """One model end-to-end (forward-only), loaded and FREED inside this call so only one model is resident.
    (a) Split the pool into HAS_ALT / NO_ALT by the plausibility gate on single-turn first-token log-probs;
    cache per item the NEUTRAL prompt + the W*-first-tok id + (for HAS_ALT) the COUNTER prompt and the answer-
    slot resid_post[L_FIT][-1] under both conditions. (b) Fit u over the HAS_ALT caving items; proj_unit = the
    mean caved-held u-projection magnitude at L_FIT. (c) Dose-response: for each item in BOTH groups at each
    alpha, ADD alpha*proj_unit*u at L_FIT on the NEUTRAL run and read delta_entropy + argmax_flip. Returns the
    per-model record + decision."""
    import torch
    from transformer_lens import HookedTransformer
    from job_truthful_flip import PUSH, NEUTRAL
    from rlhf_differential import _helpers, MIN_EFFECT_NET
    from entropy_neuron_gemma2 import entropy_of_logits

    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tag = "it" if is_chat else "base"
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    L_FIT = nL // 2
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    print(f"[{tag}] n_layers={nL} n_heads={nH} L_FIT={L_FIT}", flush=True)

    def _ent(logits):
        return float(entropy_of_logits(logits[0, -1]))

    # ---- (a) PLAUSIBILITY GATE SPLIT over the pool ----
    has_alt, no_alt = [], []
    for r in pool:
        q, C = r["q"], r["correct"]
        wrongs = r["wrong"] if "wrong" in r else [r["Wstar"]]
        sid = single(q)
        cid = first(" " + C)
        # rank rivals by first-token log-prob (the select_items model-grading, but on first-token ids so the
        # argmax-identity readout register matches).
        lp_c = num_lp(sid, C)
        scored = sorted(((num_lp(sid, w), w) for w in wrongs), reverse=True)
        if not scored:
            continue
        lp_w, Wstar = scored[0]
        aid = first(" " + Wstar)
        if cid == aid:                                   # first-token collision -> identity readout degenerate
            continue
        lp_w2 = scored[1][0] if len(scored) > 1 else None
        grp = plausibility_group(lp_c, lp_w, lp_w2)
        if grp == "MIDDLE":
            continue
        rec = {"q": q, "correct": C, "Wstar": Wstar, "cid": cid, "aid": aid,
               "lp_c": round(lp_c, 4), "lp_w": round(lp_w, 4),
               "lp_w2": (round(lp_w2, 4) if lp_w2 is not None else None),
               "margin": round(lp_c - lp_w, 4), "_neutral": push(q, C, NEUTRAL)}
        if grp == "HAS_ALT":
            rec["_counter"] = push(q, C, PUSH["counter"].format(W=Wstar))
            has_alt.append(rec)
        else:
            no_alt.append(rec)
    print(f"[{tag}] HAS_ALT={len(has_alt)} NO_ALT={len(no_alt)} (from pool {len(pool)})", flush=True)

    # ---- (b) FIT u over the HAS_ALT CAVING items (cave: counter lowers M from neutral by >= MIN_EFFECT_NET) ----
    rc_fit, rn_fit = [], []
    n_cave = 0
    for rec in has_alt:
        cid, aid = rec["cid"], rec["aid"]
        neutral, counter = rec["_neutral"], rec["_counter"]
        rc, rn = {}, {}

        def grab_c(t, hook, _rc=rc):
            _rc[hook.layer()] = t[0, -1].detach().float().cpu(); return t

        def grab_n(t, hook, _rn=rn):
            _rn[hook.layer()] = t[0, -1].detach().float().cpu(); return t
        with torch.no_grad():
            lg_c = model.run_with_hooks(counter, fwd_hooks=[(_rname(L_FIT), grab_c)])
            lg_n = model.run_with_hooks(neutral, fwd_hooks=[(_rname(L_FIT), grab_n)])
        lp = lambda lg: torch.log_softmax(lg[0, -1].float(), -1)
        M_neu = float(lp(lg_n)[cid] - lp(lg_n)[aid])
        M_ctr = float(lp(lg_c)[cid] - lp(lg_c)[aid])
        rec["_rc"] = rc[L_FIT]
        rec["_rn"] = rn[L_FIT]
        if (M_neu - M_ctr) >= MIN_EFFECT_NET:
            rc_fit.append(rc[L_FIT]); rn_fit.append(rn[L_FIT]); n_cave += 1
    print(f"[{tag}] HAS_ALT caving items for the fit: {n_cave}", flush=True)

    fit_ok = n_cave >= 1
    proj_unit = None
    u_dev = None
    if fit_ok:
        u_cpu = fit_defer(rc_fit, rn_fit)                                  # unit, CPU float
        proj_caved = statistics.mean(float(rc @ u_cpu) for rc in rc_fit)
        proj_held = statistics.mean(float(rn @ u_cpu) for rn in rn_fit)
        proj_unit = proj_caved - proj_held
        u_dev = u_cpu.to(device)
        print(f"[{tag}] proj_caved={proj_caved:.4f} proj_held={proj_held:.4f} proj_unit={proj_unit:.4f}",
              flush=True)

    # ---- (c) DOSE-RESPONSE on BOTH groups (ADD alpha*proj_unit*u at L_FIT on the NEUTRAL run) ----
    def _dose_group(recs):
        # per-item baseline (unsteered) entropy + argmax on the NEUTRAL run, then each alpha.
        per_alpha = {a: {"de": [], "flip": []} for a in ALPHAS}
        rows = []
        for rec in recs:
            neutral, aid = rec["_neutral"], rec["aid"]
            with torch.no_grad():
                lg0 = model(neutral)
            ent0 = _ent(lg0)
            amx0 = int(lg0[0, -1].argmax())
            row = {"q": rec["q"], "Wstar": rec["Wstar"], "margin": rec["margin"],
                   "lp_c": rec["lp_c"], "lp_w": rec["lp_w"], "lp_w2": rec["lp_w2"],
                   "ent_unsteered": round(ent0, 6), "amx_unsteered": amx0,
                   "doses": {}}
            for a in ALPHAS:
                if not fit_ok:
                    continue
                hooks = _all_layer_add_at(L_FIT, u_dev, a * proj_unit)
                with torch.no_grad():
                    lg = model.run_with_hooks(neutral, fwd_hooks=hooks)
                ent = _ent(lg)
                amx = int(lg[0, -1].argmax())
                de = ent - ent0
                fl = argmax_flip(amx, amx0, aid)
                per_alpha[a]["de"].append(de)
                per_alpha[a]["flip"].append(fl)
                row["doses"][str(a)] = {"delta_entropy": round(de, 6), "argmax_flip": bool(fl),
                                        "amx_steered": amx}
            rows.append(row)
        curve = {a: aggregate_group(per_alpha[a]["de"], per_alpha[a]["flip"]) for a in ALPHAS}
        return curve, rows

    has_alt_curve, has_alt_rows = _dose_group(has_alt)
    no_alt_curve, no_alt_rows = _dose_group(no_alt)
    for a in ALPHAS:
        ha, na = has_alt_curve[a], no_alt_curve[a]
        print(f"  [{tag} alpha={a}] HAS_ALT dEnt={ha['delta_entropy']} flip={ha['argmax_flip_rate']} "
              f"(n={ha['n']}) | NO_ALT dEnt={na['delta_entropy']} flip={na['argmax_flip_rate']} "
              f"(n={na['n']})", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    decision = decide(has_alt_curve, no_alt_curve)

    def _r(x):
        return round(float(x), 6) if x is not None else None

    return {
        "name": name, "regime": "chat" if is_chat else "qa",
        "n_has_alt": len(has_alt), "n_no_alt": len(no_alt), "n_has_alt_caving_fit": n_cave,
        "n_layers": nL, "n_heads": nH, "l_fit": L_FIT, "proj_unit": _r(proj_unit), "fit_ok": bool(fit_ok),
        "min_per_group_ok": bool(len(has_alt) >= MIN_PER_GROUP and len(no_alt) >= MIN_PER_GROUP),
        "has_alt_curve": {str(a): has_alt_curve[a] for a in ALPHAS},
        "no_alt_curve": {str(a): no_alt_curve[a] for a in ALPHAS},
        "decision": decision,
        "has_alt_items": has_alt_rows,
        "no_alt_items": no_alt_rows,
    }


def run(name, tag, device, is_chat, big_pool):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))            # controls/ for sibling-control imports
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # latent_verify/ for the repo imports
    from cave_copy_confidence_conditional import _build_pool
    pool = _build_pool(big_pool=big_pool)

    res = _measure_model(name, is_chat, device, pool)

    out = {
        "name": name, "device": device, "tag": tag, "regime": "chat" if is_chat else "qa",
        "cue": "cave_dir_doubt_injection", "pool_size": len(pool), "big_pool": bool(big_pool),
        "metric": ("Plausibility-gate split (HAS_ALT = near-margin + one dominant rival; NO_ALT = "
                   "high-confidence + no plausible rival; middle dropped; first-token-collision skipped). Fit "
                   "u = unit(mean(resid_post[L_FIT][-1](COUNTER) - resid_post[L_FIT][-1](NEUTRAL))) over the "
                   "HAS_ALT caving items at L_FIT=n_layers//2 (cave_defer_direction.fit_defer); proj_unit = "
                   "mean caved-held u-projection magnitude. DOSE: ADD alpha*proj_unit*u at L_FIT across all "
                   "positions on the NEUTRAL run (alpha in {2,4,8,16}); per (group x alpha) report "
                   "delta_entropy = entropy(steered)-entropy(unsteered) (CONFIDENCE) and argmax_flip_rate "
                   "(IDENTITY: argmax changed to W* or any non-original token). Full HAS_ALT + NO_ALT "
                   "dose-response curves with n per group."),
        "thresholds": {"MARGIN_KEEP": MARGIN_KEEP, "RHO_MIN": RHO_MIN, "NOALT_MARGIN": NOALT_MARGIN,
                       "MIN_PER_GROUP": MIN_PER_GROUP, "ALPHAS": ALPHAS, "FLIP_THR": FLIP_THR,
                       "ENT_THR": ENT_THR},
        "decision_rule": DECISION_RULE,
        "result": res,
    }
    out_dir = Path("results_calib") / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cave_dir_doubt_injection_{tag}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    dd = res["decision"]
    print(f"[{tag}] {dd['category']} alpha*={dd['alpha_star']} "
          f"NO_ALT dEnt@a*={dd['noalt_delta_entropy_at_alpha_star']} "
          f"NO_ALT flip@a*={dd['noalt_flip_rate_at_alpha_star']} "
          f"n_has_alt={res['n_has_alt']} n_no_alt={res['n_no_alt']} L_FIT={res['l_fit']}", flush=True)
    print(f"[done] wrote {out_path}", flush=True)


# --------------------------------------------------------------------------- selftest (model-free, CPU, NO model load)
def selftest():
    import torch
    torch.manual_seed(0)

    # ---------- plausibility_group (the job_truthful_flip select_items split) ----------
    # HAS_ALT: |margin| < 1.5 AND rho > 2 (near-margin, one dominant rival). lp_w - lp_w2 = log(3) ~ 1.099 ->
    # rho 3 > 2; margin = -0.5 -> |margin| 0.5 < 1.5.
    assert plausibility_group(-0.5, 0.0, -math.log(3.0)) == "HAS_ALT"
    # NOT HAS_ALT (rho too low): lp_w - lp_w2 = log(1.5) -> rho 1.5 <= 2; |margin| 0.5 < noalt_margin -> MIDDLE.
    assert plausibility_group(-0.5, 0.0, -math.log(1.5)) == "MIDDLE"
    # NO_ALT via flat rival field: |margin| 4.0 >= 3.0 AND rho 1.5 <= 2 (not torn) -> NO_ALT.
    assert plausibility_group(4.0, 0.0, -math.log(1.5)) == "NO_ALT"
    # NO_ALT via single rival (no second): |margin| 5.0 >= 3.0 AND lp_w2 None -> NO_ALT.
    assert plausibility_group(5.0, 0.0, None) == "NO_ALT"
    # far margin BUT a dominant second rival (rho 5 > 2) -> NOT NO_ALT (still torn) -> MIDDLE.
    assert plausibility_group(4.0, 0.0, -math.log(5.0)) == "MIDDLE"
    # near margin BUT no second rival (rho inf > 2) -> HAS_ALT (one dominant rival, near margin).
    assert plausibility_group(-0.4, 0.0, None) == "HAS_ALT"
    # MARGIN_KEEP boundary: |margin| exactly 1.5 is NOT < 1.5 -> not HAS_ALT; and 1.5 < 3.0 -> MIDDLE.
    assert plausibility_group(-MARGIN_KEEP, 0.0, -math.log(3.0)) == "MIDDLE"
    # NOALT_MARGIN boundary: |margin| exactly 3.0 (>=) with flat rival -> NO_ALT.
    assert plausibility_group(NOALT_MARGIN, 0.0, -math.log(1.5)) == "NO_ALT"
    print("[selftest] plausibility_group: HAS_ALT / NO_ALT / MIDDLE bands + rho/margin boundaries OK")

    # ---------- entropy_from_probs: uniform max, point-mass ~0, monotone ----------
    V = 8
    uni = torch.full((V,), 1.0 / V)
    assert abs(entropy_from_probs(uni) - math.log(V)) < 1e-5
    pm = torch.zeros(V); pm[0] = 1.0
    assert entropy_from_probs(pm) < 1e-5
    spread = torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
    sharp = torch.tensor([0.9, 0.05, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0])
    assert entropy_from_probs(spread) > entropy_from_probs(sharp)         # flatter -> higher entropy
    print(f"[selftest] entropy_from_probs: uniform={entropy_from_probs(uni):.3f}=log(V), point-mass~0, "
          f"flat>sharp")

    # ---------- argmax_flip: identity readout ----------
    aid = 7
    assert argmax_flip(7, 3, aid) is True                # changed to W* (aid)
    assert argmax_flip(9, 3, aid) is True                # changed to a non-original third token
    assert argmax_flip(3, 3, aid) is False               # unchanged -> no flip
    assert argmax_flip(3, 7, aid) is True                # changed from W* to C (a change away from original)
    print("[selftest] argmax_flip: change->W*/third = flip, unchanged = no flip")

    # ---------- aggregate_group + alpha_star ----------
    agg = aggregate_group([0.1, 0.3, 0.5], [True, False, True])
    assert abs(agg["delta_entropy"] - 0.3) < 1e-5 and abs(agg["argmax_flip_rate"] - (2 / 3)) < 1e-5 \
        and agg["n"] == 3, agg
    assert aggregate_group([], []) == {"delta_entropy": None, "argmax_flip_rate": 0.0, "n": 0}
    # alpha_star: smallest alpha with HAS_ALT flip rate >= FLIP_THR.
    ha = {2: {"argmax_flip_rate": 0.10}, 4: {"argmax_flip_rate": 0.30}, 8: {"argmax_flip_rate": 0.70},
          16: {"argmax_flip_rate": 0.90}}
    assert alpha_star(ha) == 4, alpha_star(ha)            # 0.30 >= FLIP_THR(0.3) inclusive
    ha_none = {a: {"argmax_flip_rate": 0.1} for a in ALPHAS}
    assert alpha_star(ha_none) is None                   # never reaches FLIP_THR
    ha_first = {2: {"argmax_flip_rate": 0.30}, 4: {"argmax_flip_rate": 0.5}, 8: {"argmax_flip_rate": 0.6},
                16: {"argmax_flip_rate": 0.7}}
    assert alpha_star(ha_first) == 2                      # smallest is the first
    print(f"[selftest] aggregate_group + alpha_star (inclusive >=) -> a*={alpha_star(ha)}")

    # ============================================================ DECISION scenarios ===================
    # Shared HAS_ALT curve: flips kick in at alpha=4 (alpha* = 4).
    HA = {2: {"delta_entropy": 0.20, "argmax_flip_rate": 0.10, "n": 6},
          4: {"delta_entropy": 0.45, "argmax_flip_rate": 0.50, "n": 6},
          8: {"delta_entropy": 0.80, "argmax_flip_rate": 0.80, "n": 6},
          16: {"delta_entropy": 1.20, "argmax_flip_rate": 0.95, "n": 6}}

    # (i) DOUBT_WITHOUT_FLIP: at alpha*=4 NO_ALT entropy rises (>=0.3) but NO_ALT argmax does NOT flip (<0.3).
    NA_doubt = {2: {"delta_entropy": 0.15, "argmax_flip_rate": 0.0, "n": 6},
                4: {"delta_entropy": 0.40, "argmax_flip_rate": 0.05, "n": 6},
                8: {"delta_entropy": 0.70, "argmax_flip_rate": 0.10, "n": 6},
                16: {"delta_entropy": 1.10, "argmax_flip_rate": 0.20, "n": 6}}
    d1 = decide(HA, NA_doubt)
    assert d1["category"] == "DOUBT_WITHOUT_FLIP", d1
    assert d1["alpha_star"] == 4 and d1["noalt_flip_rate_at_alpha_star"] == 0.05, d1

    # (ii) FLIP_ANYWHERE: at alpha*=4 NO_ALT argmax flips too (>= 0.3).
    NA_flip = {2: {"delta_entropy": 0.15, "argmax_flip_rate": 0.10, "n": 6},
               4: {"delta_entropy": 0.40, "argmax_flip_rate": 0.50, "n": 6},
               8: {"delta_entropy": 0.70, "argmax_flip_rate": 0.80, "n": 6},
               16: {"delta_entropy": 1.10, "argmax_flip_rate": 0.95, "n": 6}}
    d2 = decide(HA, NA_flip)
    assert d2["category"] == "FLIP_ANYWHERE", d2
    assert d2["alpha_star"] == 4 and d2["noalt_flip_rate_at_alpha_star"] == 0.5, d2

    # (iii) NO_EFFECT: NO_ALT entropy never reaches ENT_THR at any alpha (checked FIRST, even if it would flip).
    NA_noeff = {a: {"delta_entropy": 0.10, "argmax_flip_rate": 0.40, "n": 6} for a in ALPHAS}
    d3 = decide(HA, NA_noeff)
    assert d3["category"] == "NO_EFFECT", d3
    assert d3["any_noalt_entropy_at_or_above_thr"] is False, d3

    # (iv) UNRESOLVED via no alpha*: HAS_ALT never flips (no alpha*), but NO_ALT entropy reaches ENT_THR.
    HA_noflip = {a: {"delta_entropy": 0.40, "argmax_flip_rate": 0.10, "n": 6} for a in ALPHAS}
    NA_someent = {a: {"delta_entropy": 0.40, "argmax_flip_rate": 0.05, "n": 6} for a in ALPHAS}
    d4 = decide(HA_noflip, NA_someent)
    assert d4["category"] == "UNRESOLVED" and d4["alpha_star"] is None, d4

    # (v) UNRESOLVED via low entropy at alpha*: NO_ALT entropy reaches ENT_THR at SOME alpha (so not NO_EFFECT)
    #     but NOT at alpha*, and NO_ALT does not flip at alpha*.
    NA_lateent = {2: {"delta_entropy": 0.10, "argmax_flip_rate": 0.05, "n": 6},
                  4: {"delta_entropy": 0.10, "argmax_flip_rate": 0.05, "n": 6},   # alpha*=4: ent 0.10 < 0.3
                  8: {"delta_entropy": 0.50, "argmax_flip_rate": 0.10, "n": 6},   # entropy reaches 0.3 later
                  16: {"delta_entropy": 0.90, "argmax_flip_rate": 0.20, "n": 6}}
    d5 = decide(HA, NA_lateent)
    assert d5["category"] == "UNRESOLVED" and d5["alpha_star"] == 4, d5
    assert d5["any_noalt_entropy_at_or_above_thr"] is True, d5
    print("[selftest] decide: DOUBT_WITHOUT_FLIP / FLIP_ANYWHERE / NO_EFFECT / UNRESOLVED(x2) all fire")

    # ---------- decide boundaries (inclusive >=) ----------
    # ENT_THR boundary at alpha*: NO_ALT entropy exactly ENT_THR (with flip < FLIP_THR) -> DOUBT_WITHOUT_FLIP.
    NA_entb = {2: {"delta_entropy": 0.0, "argmax_flip_rate": 0.0, "n": 6},
               4: {"delta_entropy": ENT_THR, "argmax_flip_rate": 0.0, "n": 6},
               8: {"delta_entropy": ENT_THR, "argmax_flip_rate": 0.0, "n": 6},
               16: {"delta_entropy": ENT_THR, "argmax_flip_rate": 0.0, "n": 6}}
    assert decide(HA, NA_entb)["category"] == "DOUBT_WITHOUT_FLIP", decide(HA, NA_entb)
    # just below ENT_THR at every alpha -> NO_EFFECT.
    NA_entlo = {a: {"delta_entropy": ENT_THR - 1e-6, "argmax_flip_rate": 0.0, "n": 6} for a in ALPHAS}
    assert decide(HA, NA_entlo)["category"] == "NO_EFFECT", decide(HA, NA_entlo)
    # FLIP_THR boundary at alpha*: NO_ALT flip exactly FLIP_THR (with entropy high) -> FLIP_ANYWHERE.
    NA_flipb = {2: {"delta_entropy": 0.4, "argmax_flip_rate": 0.0, "n": 6},
                4: {"delta_entropy": 0.4, "argmax_flip_rate": FLIP_THR, "n": 6},
                8: {"delta_entropy": 0.4, "argmax_flip_rate": FLIP_THR, "n": 6},
                16: {"delta_entropy": 0.4, "argmax_flip_rate": FLIP_THR, "n": 6}}
    assert decide(HA, NA_flipb)["category"] == "FLIP_ANYWHERE", decide(HA, NA_flipb)
    # NO_ALT flip just below FLIP_THR at alpha* (entropy high) -> DOUBT_WITHOUT_FLIP.
    NA_fliplo = {2: {"delta_entropy": 0.4, "argmax_flip_rate": 0.0, "n": 6},
                 4: {"delta_entropy": 0.4, "argmax_flip_rate": FLIP_THR - 1e-6, "n": 6},
                 8: {"delta_entropy": 0.4, "argmax_flip_rate": FLIP_THR - 1e-6, "n": 6},
                 16: {"delta_entropy": 0.4, "argmax_flip_rate": FLIP_THR - 1e-6, "n": 6}}
    assert decide(HA, NA_fliplo)["category"] == "DOUBT_WITHOUT_FLIP", decide(HA, NA_fliplo)
    # alpha* inclusive at HAS_ALT flip == FLIP_THR exactly.
    HA_exact = {2: {"delta_entropy": 0.2, "argmax_flip_rate": FLIP_THR, "n": 6},
                4: {"delta_entropy": 0.4, "argmax_flip_rate": 0.6, "n": 6},
                8: {"delta_entropy": 0.6, "argmax_flip_rate": 0.8, "n": 6},
                16: {"delta_entropy": 0.8, "argmax_flip_rate": 0.9, "n": 6}}
    assert decide(HA_exact, NA_doubt)["alpha_star"] == 2, decide(HA_exact, NA_doubt)
    print("[selftest] decide boundaries: ENT_THR / FLIP_THR / alpha* inclusive (>=) OK")

    # ---------- end-to-end on the ADD math (planted): adding u raises entropy; argmax-flip depends on geometry
    # A tiny synthetic 'model': a unit unembed over a few token slots; the answer-slot logit = (resid . W[slot]).
    # adding alpha*proj_unit*u flattens the logits (raises entropy) when u points away from the dominant slot;
    # the argmax flips only when alpha is large enough to overtake. Verifies the real-run readout direction. ----
    d = 64
    g = torch.Generator().manual_seed(3)
    u = fit_defer([torch.randn(d, generator=g) + 2.0 for _ in range(5)],
                  [torch.randn(d, generator=g) for _ in range(5)])
    assert abs(float(u.norm()) - 1.0) < 1e-5                         # fit_defer returns a unit vector
    # planted resid + 3-slot unembed; slot 0 dominates (the unsteered argmax), slot 1 == W*.
    resid = torch.randn(d, generator=g)
    W = torch.stack([3.0 * resid / resid.norm(), 2.0 * u, torch.randn(d, generator=g)])   # [3, d]
    cid_slot, aid_slot = 0, 1

    def logits_of(r):
        return torch.tensor([float(r @ W[s]) for s in range(3)])
    base_logits = logits_of(resid)
    amx0 = int(base_logits.argmax())
    ent0 = entropy_from_probs(torch.softmax(base_logits, -1))
    # add a small dose along u: entropy moves; with a big enough dose the argmax flips toward slot 1 (u-aligned).
    small = resid + 1.0 * 1.0 * u
    big = resid + 20.0 * 1.0 * u
    ent_small = entropy_from_probs(torch.softmax(logits_of(small), -1))
    amx_big = int(logits_of(big).argmax())
    assert ent_small != ent0, "adding u must move the answer-slot entropy"
    assert argmax_flip(amx_big, amx0, aid_slot) is True             # large dose flips the argmax
    assert argmax_flip(amx0, amx0, aid_slot) is False               # no change -> no flip
    print(f"[selftest] ADD math: dEntropy(small)={ent_small - ent0:+.3f}, big-dose argmax flip OK")

    print("[selftest] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--name", default="google/gemma-2-9b",
                   help="model (gemma-2 base is the primary site; -it via --chat)")
    p.add_argument("--tag", default="9b_base")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--chat", action="store_true",
                   help="use the chat template (-it model; qa template otherwise; base is primary)")
    p.add_argument("--big-pool", action="store_true",
                   help="merge sycophancy_items_lowconf.json + TruthfulQA generation for n (needs datasets)")
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        run(args.name, args.tag, args.device, args.chat, args.big_pool)


if __name__ == "__main__":
    main()
