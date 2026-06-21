"""FAITHFUL cross-intervention: does steering a causal CONFIDENCE direction change the model's REALIZED
caved answer (the actual next-token argmax), and what is cos(u_cave, u_conf)? (sibling of
confidence_caving_gate.py / cave_suppress_vs_install.py / confidence_vs_cave_direction.py.)

CONTEXT (neutral). Two prior controls measure adjacent quantities, SEPARATELY:
  - confidence_caving_gate.py steers the entropy-quartile CONFIDENCE direction u_conf and reads the
    cave on the LOGP-DIFFERENCE metric M = logp(C) - logp(W*) (gate_fraction on M).
  - cave_suppress_vs_install.py reads the cave on the REALIZED next-token argmax: it restricts to CAVING
    items whose COUNTER argmax IS the W*-first-token, ablates u_cave, and classifies whether the emitted
    argmax returns to the item's NEUTRAL-condition argmax.
This control runs the confidence STEER of the first against the realized-argmax readout of the second. It
does NOT refit or re-establish either prior result; it measures one cross-intervention -- steer u_conf UP on
the COUNTER condition of the argmax-W* caving items and read the FULL realized next-token softmax -- and
lets the number fall where it lands. It also reports cos(u_cave, u_conf) with an item-bootstrap CI.

It REUSES, verbatim, the verified primitives (read those files for the exact signatures):
  - misconception_pool.ITEMS_WIDE (the item pool).
  - job_truthful_flip: PUSH / NEUTRAL turns (counter = PUSH["counter"].format(W=W); neutral = NEUTRAL).
  - rlhf_differential: _helpers, _logp_diff, MIN_EFFECT_NET (the caving gate, |M_neu - M_ctr| >= 0.5).
  - confidence_direction_causal: quantile_split, _signal (ENTROPY_QUARTILE), QUANTILE=0.25.
  - entropy_neuron_gemma2: entropy_of_logits.
  - confidence_vs_cave_direction: unit, diff_of_means, cosine, split_indices, _proj_edit_hook.
  - cave_suppress_vs_install: the realized full-softmax readout + argmax-W* selection + headline-layer
    (max in-sample cave-necessity) construction, mirrored here.
  - headset_direction: FIT_LAYERS (the layer sweep) + _rname (resid_post hook name), deferred at run time.

MEASURE per model (base, it; defaults google/gemma-2-9b / google/gemma-2-9b-it), headline = base. On the
wide pool, build NEUTRAL and COUNTER (W* asserted) prompts. Restrict to CAVING items (counter lowers the
first-token margin M=logp(C)-logp(W*) from neutral by >= MIN_EFFECT_NET) whose COUNTER argmax IS the
W*-first-token (the model would ACTUALLY emit W*). Split those into a TRAIN/TEST fold. At the headline
layer L (max in-sample cave-necessity over the argmax-W* caving TRAIN fold):
  1. Fit u_conf = entropy-quartile diff-of-means(resid_post[L][-1] over TRAIN high-confidence items -
     over TRAIN low-confidence items) at the NEUTRAL condition (ENTROPY_QUARTILE; entropy negated so the
     high-quantile end is the high-confidence end -- exactly confidence_direction_causal / the sibling
     confidence_caving_gate). proj_hi / proj_lo = the TRAIN high-/low-confidence mean u_conf-projections.
  2. Fit u_cave = unit(mean(resid_post[L][-1](counter) - resid_post[L][-1](neutral))) over the argmax-W*
     caving TRAIN fold.
  3. On the held-out argmax-W* caving items in the COUNTER condition: STEER u_conf UP -- set its projection
     on the counter residual to proj_hi -- then read the FULL realized next-token softmax (P_steer). Bucket
     the new argmax: == the item's NEUTRAL-condition argmax (restore-to-unpushed) / == W*-first-tok (cave
     persists) / == C-first-tok / third. Also report dP(W*) = P_steer(W*) - P_counter(W*) and
     KL(P_steer || P_neutral) vs KL(P_counter || P_neutral).
  4. Matched-magnitude RANDOM unit direction, identical steer-up edit (set its counter projection to its
     own TRAIN high mean) -> the control floor, same buckets.
  5. cos(u_cave, u_conf) at L, with an item-bootstrap CI (resample the fit items, refit both directions,
     recompute cos).
  6. -it: if there are 0 argmax-W* caving items, report INSUFFICIENT for the steer arm; still report
     cos(u_cave, u_conf) if both directions are fittable.

NEUTRAL DECISION (module constants THR=0.5, COS_THR=0.3, EFFECT_EPS=0.01, RAND_FLOOR=0.2; numbers +
categories only, NO hypothesis, NO statement about base vs -it):
  Steer arm:
    GATES_REALIZED_CAVE   iff frac(steered argmax == neutral-argmax) >= THR AND
                              KL(P_steer||P_neutral) < KL(P_counter||P_neutral) AND
                              the matched-random direction's restore-frac < RAND_FLOOR;
    else INDEPENDENT_REALIZED (steering confidence does not restore the unpushed answer beyond the random
                              floor).
    INSUFFICIENT          iff no argmax-W* items.
  Axis arm (separate tag): AXIS_ALIGNED iff |cos(u_cave,u_conf)| >= COS_THR else AXIS_ORTHOGONAL. Report
    cos + bootstrap CI.
  No claim is attached to any bucket, sign, the steer/axis combination, or the base-vs-it comparison.

Forward-only (diff-of-means + projection edits + full-softmax readouts; no backward) -> fits the 40GB A100.
The only NEW logic is the steer-up readout of the REALIZED argmax/softmax (a reuse of the existing proj-edit
hook + full-softmax readout), the matched-random steer-up, the cosine item-bootstrap, and the two pure
decisions -- all covered by the model-free --selftest, which loads NO model.

  python controls/confidence_caving_gate_faithful.py --selftest
  python controls/confidence_caving_gate_faithful.py --device cuda \
    --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

# Repo-internal and sibling-control imports (item pool, turn builders, readout helpers, direction math,
# split machinery, FIT_LAYERS/_rname, entropy) are DEFERRED into the functions that use them so --selftest
# runs standalone on CPU with NO model load and nothing else on sys.path. On the box every file (repo
# modules AND controls) is scp'd flat into latent_verify/, where these resolve; locally we put both
# controls/ and the repo root on sys.path. The pure helpers used by --selftest are imported through
# _pure() below; the entropy-quartile fit machinery is reused from confidence_direction_causal.

# ----------------------------------------------------------------- pre-registered thresholds (neutral)
FIT_LAYERS = [24, 28, 32, 36]   # same sweep as headset_direction / confidence_caving_gate (module constant
                                # so --selftest needs nothing on sys.path; the real run defers headset's).
THR = 0.5                       # restore-to-neutral argmax fraction the steer must reach to "gate"
COS_THR = 0.3                   # |cos(u_cave, u_conf)| >= this -> AXIS_ALIGNED else AXIS_ORTHOGONAL
EFFECT_EPS = 0.01               # |dP(W*)| below this means the steer did not move the realized W* mass
RAND_FLOOR = 0.2                # the matched-random restore-frac must fall below this (clean control)
QUANTILE = 0.25                 # entropy quantile: top-QUANTILE vs bottom-QUANTILE confidence contrast
SPLIT_SEED = 0                  # deterministic train/test fold assignment (shared with the sibling controls)
RAND_SEED = 0                   # deterministic matched random-direction control
BOOT_SEED = 0                   # deterministic cosine item-bootstrap
N_BOOT = 2000                   # cosine bootstrap resamples (matches cave_direction_heldout.N_BOOT)
CI_LO, CI_HI = 2.5, 97.5        # 95 percent percentile interval (matches cave_direction_heldout)
MIN_FIT = 3                     # below this many argmax-W* caving items, no held-out direction can be fit
DEFINITION = "ENTROPY_QUARTILE"  # the confidence-direction fit reused from confidence_direction_causal
MODELS = ("base", "it")

DECISION_RULE = (
    "On the wide misconception pool, build NEUTRAL and COUNTER (W* asserted) prompts (job_truthful_flip "
    "turns; qa template for base, chat template for -it). Restrict to CAVING items (counter lowers the "
    "first-token margin M=logp(C)-logp(W*) from neutral by >= MIN_EFFECT_NET) whose COUNTER argmax IS the "
    "W*-first-token (the model would actually emit W*). Split those argmax-W* caving items into a TRAIN/TEST "
    "fold. Headline layer L = the fit layer with the largest IN-SAMPLE u_cave cave-necessity over the "
    "argmax-W* caving TRAIN fold. At L: (1) fit u_conf = diff-of-means(resid_post[L][-1] over TRAIN "
    "high-confidence items - over TRAIN low-confidence items) at the NEUTRAL condition, where the quantile "
    "is the top/bottom QUANTILE(0.25) of next-token output entropy at the answer position (ENTROPY_QUARTILE; "
    "entropy negated so the high-quantile end is the high-confidence end); proj_hi / proj_lo = the TRAIN "
    "high-/low-confidence mean u_conf-projections. (2) fit u_cave = unit(mean(resid_post[L][-1](counter) - "
    "resid_post[L][-1](neutral))) over the argmax-W* caving TRAIN fold. (3) On the held-out argmax-W* caving "
    "items in the COUNTER condition, STEER u_conf UP (set its counter-residual projection to proj_hi) and "
    "read the FULL realized next-token softmax P_steer; bucket the new argmax == the item's NEUTRAL-condition "
    "argmax (restore-to-unpushed) / == W*-first-tok (cave persists) / == C-first-tok / third; report "
    "dP(W*)=P_steer(W*)-P_counter(W*) and KL(P_steer||P_neutral) vs KL(P_counter||P_neutral). (4) matched-"
    "magnitude RANDOM unit direction, identical steer-up edit (set its counter projection to its own TRAIN "
    "high mean) -> the floor, same buckets. (5) cos(u_cave, u_conf) at L with an item-bootstrap CI (resample "
    "the fit items, refit both directions, recompute cos). "
    "STEER ARM (THR=0.5, EFFECT_EPS=0.01, RAND_FLOOR=0.2): GATES_REALIZED_CAVE iff frac(steered "
    "argmax==neutral-argmax) >= THR AND KL(P_steer||P_neutral) < KL(P_counter||P_neutral) AND the matched-"
    "random restore-frac < RAND_FLOOR; else INDEPENDENT_REALIZED; INSUFFICIENT iff no argmax-W* items. "
    "AXIS ARM (separate tag, COS_THR=0.3): AXIS_ALIGNED iff |cos(u_cave,u_conf)| >= COS_THR else "
    "AXIS_ORTHOGONAL (cos + bootstrap CI reported). Reported for base and -it; numbers + categories only, no "
    "claim attached to any bucket, sign, the steer/axis combination, or the base-vs-it comparison."
)


# ----------------------------------------------------------------- pure helper resolution (shared)
def _pure():
    """Import the reused PURE helpers from the sibling controls + repo modules. Deferred + sys.path-guarded
    so it resolves both locally (controls/ + repo root on path) and on the box (everything flat in
    latent_verify/). No model load, no torch device work -- safe inside --selftest. Returns a namespace."""
    here = Path(__file__).resolve().parent          # .../controls
    for p in (str(here), str(here.parent)):         # controls/ for siblings; repo root for repo modules
        if p not in sys.path:
            sys.path.insert(0, p)
    from confidence_vs_cave_direction import (
        unit, diff_of_means, cosine, split_indices, _proj_edit_hook)
    from confidence_direction_causal import quantile_split, _signal
    from entropy_neuron_gemma2 import entropy_of_logits
    return {"unit": unit, "diff_of_means": diff_of_means, "cosine": cosine,
            "split_indices": split_indices, "_proj_edit_hook": _proj_edit_hook,
            "quantile_split": quantile_split, "_signal": _signal,
            "entropy_of_logits": entropy_of_logits}


# ----------------------------------------------------------------- pure distribution math
def kl_div(p, q, eps=1e-12):
    """KL(p || q) = sum_v p(v) * log(p(v)/q(v)) over 1-D probability tensors. Non-negative for proper
    distributions. Pure (clamps both to eps for numerical stability; same per-coordinate form as
    cave_suppress_vs_install.kl_div). Used for KL(P_steer||P_neutral) and KL(P_counter||P_neutral): the
    distance of the steered / counter distribution to the NEUTRAL distribution."""
    pp = p.float().clamp_min(eps)
    qq = q.float().clamp_min(eps)
    return float((pp * (pp.log() - qq.log())).sum())


# ----------------------------------------------------------------- pure argmax-identity buckets
def argmax_bucket(amx, cid, aid, neu_argmax):
    """Classify the post-steer argmax token id `amx` into ONE of four mutually-exclusive buckets:
      'C'       : amx == C-first-tok                  (the correct competitor)
      'Wstar'   : amx == W*-first-tok                 (the cave persists -- still emits W*)
      'neutral' : amx == the item's NEUTRAL-condition argmax (restore-to-unpushed), and it is NOT C/W*
      'third'   : none of the above                   (a third token; mass diffused elsewhere)
    Priority C > Wstar > neutral > third so the four buckets PARTITION the items even when the neutral
    argmax coincides with C or W* (those collapse into C / Wstar). Pure (ints in, str out). Mirrors
    cave_suppress_vs_install.argmax_bucket exactly."""
    if amx == cid:
        return "C"
    if amx == aid:
        return "Wstar"
    if neu_argmax is not None and amx == neu_argmax:
        return "neutral"
    return "third"


def bucket_fracs(buckets):
    """Fractions of each bucket over a list of bucket labels. Returns C/Wstar/neutral/third fractions (0.0
    when empty) plus the raw count n. Pure. Mirrors cave_suppress_vs_install.bucket_fracs."""
    n = len(buckets)
    keys = ("C", "Wstar", "neutral", "third")
    if n == 0:
        return {k: 0.0 for k in keys} | {"n": 0}
    out = {k: round(sum(1 for b in buckets if b == k) / n, 4) for k in keys}
    out["n"] = n
    return out


# ----------------------------------------------------------------- pure cosine item-bootstrap
def cosine_bootstrap_ci(cave_rows_c, cave_rows_n, conf_rows_hi, conf_rows_lo, P,
                        seed=BOOT_SEED, n_boot=N_BOOT, lo=CI_LO, hi=CI_HI):
    """Percentile bootstrap CI of cos(u_cave, u_conf) by resampling the FIT ITEMS and refitting both
    directions each resample. Inputs are aligned lists/stacks of last-token residual rows (already on a
    single device, float):
      cave_rows_c / cave_rows_n : the COUNTER / NEUTRAL residuals of the argmax-W* caving TRAIN items
                                  (u_cave = unit(mean(c - n))).
      conf_rows_hi / conf_rows_lo : the NEUTRAL residuals of the TRAIN high-/low-confidence items
                                    (u_conf = unit(mean(hi) - mean(lo))).
    Each bootstrap resamples WITH REPLACEMENT, independently within the cave and confidence item sets, then
    recomputes the cosine of the two refit directions. Returns {point, lo, hi, n_boot} where `point` is the
    cosine on the full (un-resampled) sets. Pure (tensors + generator -> dict); n_boot=0 or any empty set
    returns the point estimate with None bounds."""
    import random as _r
    unit, diff_of_means, cosine = P["unit"], P["diff_of_means"], P["cosine"]

    def _fit(c_idx, n_idx, hi_idx, lo_idx):
        u_cave = unit((cave_rows_c[c_idx] - cave_rows_n[n_idx]).mean(0))
        u_conf = unit(diff_of_means(conf_rows_hi[hi_idx], conf_rows_lo[lo_idx]))
        return cosine(u_cave, u_conf)

    nc = cave_rows_c.shape[0]
    nhi, nlo = conf_rows_hi.shape[0], conf_rows_lo.shape[0]
    if nc == 0 or nhi == 0 or nlo == 0:
        return {"point": None, "lo": None, "hi": None, "n_boot": 0}
    # point estimate: u_cave over ALL cave train items (counter - neutral aligned), u_conf over ALL hi/lo.
    full = list(range(nc))
    point = _fit(full, full, list(range(nhi)), list(range(nlo)))
    if n_boot <= 0:
        return {"point": round(point, 4), "lo": None, "hi": None, "n_boot": 0}
    rng = _r.Random(seed)
    cs = []
    for _ in range(n_boot):
        ci = [rng.randrange(nc) for _ in range(nc)]          # cave items resampled (counter/neutral paired)
        hii = [rng.randrange(nhi) for _ in range(nhi)]       # confidence-high items resampled
        loi = [rng.randrange(nlo) for _ in range(nlo)]       # confidence-low items resampled
        cs.append(_fit(ci, ci, hii, loi))
    cs.sort()

    def pct(p):
        return cs[min(n_boot - 1, max(0, int(round(p / 100 * (n_boot - 1)))))]
    return {"point": round(point, 4), "lo": round(pct(lo), 4), "hi": round(pct(hi), 4), "n_boot": n_boot}


# ----------------------------------------------------------------- pure decisions
def decide_steer(frac_neutral, dP_wstar, kl_steer, kl_counter, rand_frac_neutral,
                 n_fit=None, thr=THR, effect_eps=EFFECT_EPS, rand_floor=RAND_FLOOR, min_fit=MIN_FIT):
    """STEER-arm decision over the measured numbers only (no hypothesis attached). Pure.
      INSUFFICIENT        iff n_fit is not None and n_fit < min_fit (no held-out argmax-W* substrate).
      GATES_REALIZED_CAVE iff frac(steered argmax == neutral-argmax) >= thr AND
                              KL(P_steer||P_neutral) < KL(P_counter||P_neutral) AND
                              the matched-random restore-frac < rand_floor.
      else INDEPENDENT_REALIZED (steering confidence does not restore the unpushed answer beyond the random
                                floor). dP(W*) and EFFECT_EPS are reported alongside (a steered argmax that
                                returns to neutral implies the realized W* mass moved); the verdict is on the
                                restore-frac + KL + clean random, exactly as stated above."""
    if n_fit is not None and n_fit < min_fit:
        return {"category": "INSUFFICIENT", "n_fit": n_fit,
                "frac_argmax_neutral": (round(frac_neutral, 4) if frac_neutral is not None else None),
                "dP_Wstar": (round(dP_wstar, 6) if dP_wstar is not None else None),
                "kl_steer_to_neutral": (round(kl_steer, 6) if kl_steer is not None else None),
                "kl_counter_to_neutral": (round(kl_counter, 6) if kl_counter is not None else None),
                "kl_moves_toward_neutral": None,
                "rand_frac_argmax_neutral": (round(rand_frac_neutral, 4) if rand_frac_neutral is not None
                                             else None),
                "msg": f"only {n_fit} held-out argmax-W* item(s) < MIN_FIT({min_fit}); no substrate to test."}

    restores = frac_neutral is not None and frac_neutral >= thr
    kl_back = kl_steer is not None and kl_counter is not None and kl_steer < kl_counter
    rand_clean = rand_frac_neutral is None or rand_frac_neutral < rand_floor
    moved_w = dP_wstar is not None and abs(dP_wstar) >= effect_eps

    fires = restores and kl_back and rand_clean
    if fires:
        cat = "GATES_REALIZED_CAVE"
        msg = (f"frac(argmax==neutral-argmax) {frac_neutral:.3f} >= {thr} AND KL(P_steer||P_neutral) "
               f"{kl_steer:.4f} < KL(P_counter||P_neutral) {kl_counter:.4f} AND matched-random restore-frac "
               f"{None if rand_frac_neutral is None else round(rand_frac_neutral, 3)} < {rand_floor}: "
               f"steering u_conf UP returns the EMITTED answer to the model's unpushed token and moves the "
               f"realized distribution back toward NEUTRAL, beyond the random floor.")
    else:
        cat = "INDEPENDENT_REALIZED"
        msg = (f"steering u_conf UP does not restore the unpushed answer beyond the random floor: "
               f"frac_neutral {None if frac_neutral is None else round(frac_neutral, 3)} "
               f"({'>=' if restores else '<'} {thr}); KL_steer "
               f"{None if kl_steer is None else round(kl_steer, 4)} "
               f"{'<' if kl_back else '>='} KL_counter {None if kl_counter is None else round(kl_counter, 4)}; "
               f"rand_frac_neutral {None if rand_frac_neutral is None else round(rand_frac_neutral, 3)} "
               f"({'<' if rand_clean else '>='} {rand_floor}).")
    return {"category": cat,
            "n_fit": n_fit,
            "gates_realized_cave": cat == "GATES_REALIZED_CAVE",
            "frac_argmax_neutral": (round(frac_neutral, 4) if frac_neutral is not None else None),
            "dP_Wstar": (round(dP_wstar, 6) if dP_wstar is not None else None),
            "wstar_mass_moved": bool(moved_w),
            "kl_steer_to_neutral": (round(kl_steer, 6) if kl_steer is not None else None),
            "kl_counter_to_neutral": (round(kl_counter, 6) if kl_counter is not None else None),
            "kl_moves_toward_neutral": bool(kl_back),
            "rand_frac_argmax_neutral": (round(rand_frac_neutral, 4) if rand_frac_neutral is not None
                                         else None),
            "rand_clean": bool(rand_clean),
            "msg": msg}


def decide_axis(cos_val, cos_thr=COS_THR):
    """AXIS_ALIGNED iff |cos(u_cave, u_conf)| >= cos_thr else AXIS_ORTHOGONAL. None -> AXIS_ORTHOGONAL
    (no fittable cosine). Pure."""
    if cos_val is None:
        return "AXIS_ORTHOGONAL"
    return "AXIS_ALIGNED" if abs(cos_val) >= cos_thr else "AXIS_ORTHOGONAL"


def model_decision(steer, cos_ci, headline_layer):
    """Assemble the neutral STEER + AXIS decision for one model. Pure (dicts/floats -> dict). No claim
    attached. `steer` is the decide_steer dict; `cos_ci` is the cosine_bootstrap_ci dict (or None)."""
    cos_point = cos_ci.get("point") if cos_ci else None
    return {
        "steer_bucket": steer["category"],
        "axis_bucket": decide_axis(cos_point),
        "cos_cave_conf": (round(cos_point, 4) if cos_point is not None else None),
        "cos_ci": cos_ci,
        "headline_layer": headline_layer,
        "steer": steer,
    }


# ----------------------------------------------------------------- real-run helpers
def _full_softmax(logits):
    """Full next-token probability vector at the LAST position. gemma-2's final softcap is applied inside
    the forward, so softmax(logits[0,-1]) is the realized next-token distribution
    (cave_suppress_vs_install._full_softmax). Returns a 1-D float tensor."""
    return torch.softmax(logits[0, -1].float(), dim=-1)


def _logits(model, ids, hooks=None):
    """Full last-position logits (optionally under fwd_hooks). Forward-only."""
    with torch.no_grad():
        return model.run_with_hooks(ids, fwd_hooks=hooks) if hooks else model(ids)


def _collect(model, pool, device, is_chat, fit_layers, rname, P):
    """One model: per pool item, under NEUTRAL and COUNTER (W* asserted), in ONE forward each, cache the
    last-token resid_post at every fit layer AND read the full next-token softmax (P1=neutral, P2=counter),
    the signed margins M = logp(C)-logp(W*), the realized argmaxes, and the next-token ENTROPY at the answer
    position under the neutral prompt (for the ENTROPY_QUARTILE confidence fit). First-token-collision items
    (cid==aid) skipped. Forward-only. Mirrors cave_suppress_vs_install._collect +
    confidence_caving_gate._collect (entropy)."""
    from rlhf_differential import _helpers, _logp_diff
    from job_truthful_flip import PUSH, NEUTRAL
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    tag = "it" if is_chat else "base"
    recs = []
    for i, it in enumerate(pool):
        q, C, W = it["q"], it["correct"], it["Wstar"]
        cid, aid = first(" " + C), first(" " + W)
        if cid == aid:                                   # first-token collision -> readout degenerate, skip
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
            lg_n = model.run_with_hooks(neutral, fwd_hooks=[(n, grab_n) for n in names])
            lg_c = model.run_with_hooks(counter, fwd_hooks=[(n, grab_c) for n in names])
        Pn, Pc = _full_softmax(lg_n), _full_softmax(lg_c)
        M_neu = float(_logp_diff(lg_n, cid, aid))
        M_ctr = float(_logp_diff(lg_c, cid, aid))
        ent_neu = float(P["entropy_of_logits"](lg_n[0, -1]))      # answer-position next-token entropy
        neu_argmax = int(Pn.argmax())
        ctr_argmax = int(Pc.argmax())
        recs.append({"i": i, "q": q, "cid": cid, "aid": aid, "neutral": neutral, "counter": counter,
                     "rn": rn, "rc": rc, "M_neu": M_neu, "M_ctr": M_ctr, "ent_neu": ent_neu,
                     "neu_argmax": neu_argmax, "ctr_argmax": ctr_argmax,
                     "P1": Pn.detach().cpu(), "P2": Pc.detach().cpu()})
        print(f"  [{tag}] item {i} M_neu={M_neu:+.2f} ent_neu={ent_neu:.3f} M_ctr={M_ctr:+.2f} "
              f"amx_neu={neu_argmax} amx_ctr={ctr_argmax} P(W*)ctr={float(Pc[aid]):.2e} "
              f"q={q[:36]!r}", flush=True)
    return recs


def _u_cave(recs, idxs, L, device, P):
    """Diff-of-means cave direction over the items in `idxs` at layer L, plus the train-neutral-mean
    projection. u = unit(mean(rc - rn)); proj_n = mean(rn . u). Forward-free (operates on the cached
    residuals). Mirrors cave_suppress_vs_install._u_cave."""
    unit = P["unit"]
    Rc = torch.stack([recs[k]["rc"][L] for k in idxs]).to(device)
    Rn = torch.stack([recs[k]["rn"][L] for k in idxs]).to(device)
    u = unit((Rc - Rn).mean(0))
    proj_n = statistics.mean(float(recs[k]["rn"][L].to(device) @ u) for k in idxs)
    return u, proj_n


def _headline_layer(model, recs, fit_idx, fit_layers, device, rname, P):
    """Headline layer = the fit layer with the largest IN-SAMPLE u_cave cave-necessity over the fit-set
    (the SAME headline-selection rule as cave_suppress_vs_install._headline_layer). For each layer, ablate
    the u_cave-projection on each fit COUNTER residual to the fit-set neutral mean and read the margin
    recovery frac=(M_ablate-M_ctr)/(M_neu-M_ctr); pick the max-mean layer. Forward-only."""
    from rlhf_differential import _logp_diff
    proj_edit = P["_proj_edit_hook"]

    def _Mlast(model, ids, cid, aid, hooks):
        return float(_logp_diff(_logits(model, ids, hooks=hooks), cid, aid))

    def _nec(L):
        u, proj_n = _u_cave(recs, fit_idx, L, device, P)
        fr = []
        for k in fit_idx:
            r = recs[k]
            gap = r["M_neu"] - r["M_ctr"]
            if abs(gap) < 1e-6:
                continue
            h = [(rname(L), proj_edit(u, proj_n))]
            M_ab = _Mlast(model, r["counter"], r["cid"], r["aid"], h)
            fr.append((M_ab - r["M_ctr"]) / gap)
        return statistics.mean(fr) if fr else None
    per_layer = {L: _nec(L) for L in fit_layers}
    valid = [L for L in fit_layers if per_layer[L] is not None]
    headline = max(valid, key=lambda L: per_layer[L]) if valid else None
    return headline, {L: (round(v, 4) if v is not None else None) for L, v in per_layer.items()}


def _measure_model(name, is_chat, device, pool, fit_layers, rname, P):
    """One model end-to-end. Collect realized + M + entropy readouts under neutral/counter on the wide pool;
    restrict to CAVING items (counter lowers M from neutral by >= MIN_EFFECT_NET) whose COUNTER argmax is the
    W*-first-tok; split into a TRAIN/TEST fold; pick the headline layer (max in-sample cave-necessity over
    the TRAIN fold); at L fit u_conf (entropy-quartile diff-of-means at NEUTRAL, TRAIN fold) and u_cave
    (counter-neutral, TRAIN fold); on the TEST fold steer u_conf UP on the COUNTER residual, read the
    realized softmax, bucket the argmax, and compute dP(W*) + KL; matched-random steer-up floor; cosine
    item-bootstrap. Forward-only. Returns a dict with the per-model decision."""
    from transformer_lens import HookedTransformer
    from rlhf_differential import MIN_EFFECT_NET
    unit = P["unit"]
    proj_edit = P["_proj_edit_hook"]
    quantile_split, _signal = P["quantile_split"], P["_signal"]

    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    recs = _collect(model, pool, device, is_chat, fit_layers, rname, P)
    n = len(recs)
    g = torch.Generator(device="cpu").manual_seed(RAND_SEED)

    # CAVING items: counter lowers M from neutral by >= MIN_EFFECT_NET (same gate as the siblings).
    cave_pos = [k for k, r in enumerate(recs) if (r["M_neu"] - r["M_ctr"]) >= MIN_EFFECT_NET]
    # restrict to items whose COUNTER argmax IS the W*-first-tok (the model would actually emit W*).
    argmaxW = [k for k in cave_pos if recs[k]["ctr_argmax"] == recs[k]["aid"]]

    out = {"name": name, "regime": "chat" if is_chat else "qa", "n_ok": n,
           "n_cave": len(cave_pos), "n_argmaxW_cave": len(argmaxW)}

    if len(argmaxW) < MIN_FIT:
        print(f"  [{name}] only {len(argmaxW)} argmax-W* caving items < MIN_FIT({MIN_FIT}); "
              f"no held-out direction.", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        out["headline_layer"] = None
        out["numbers"] = None
        out["random_numbers"] = None
        steer = decide_steer(None, None, None, None, None, n_fit=len(argmaxW))
        out["decision"] = model_decision(steer, None, None)
        return out

    # held-out fold over the argmax-W* caving items.
    tr_pos, te_pos = P["split_indices"](len(argmaxW), SPLIT_SEED)
    train = [argmaxW[j] for j in tr_pos]
    test = [argmaxW[j] for j in te_pos]

    headline, per_layer_nec = _headline_layer(model, recs, train, fit_layers, device, rname, P)
    out["in_sample_necessity_by_layer"] = per_layer_nec
    out["headline_layer"] = headline
    if headline is None:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        out["numbers"] = None
        out["random_numbers"] = None
        steer = decide_steer(None, None, None, None, None, n_fit=len(argmaxW))
        out["decision"] = model_decision(steer, None, None)
        return out

    L = headline
    # ---- u_cave: diff-of-means(counter - neutral) over the argmax-W* caving TRAIN fold ----
    u_cave, _proj_n_cave = _u_cave(recs, train, L, device, P)

    # ---- u_conf: entropy-quartile diff-of-means at NEUTRAL over the argmax-W* caving TRAIN fold ----
    # the confidence contrast is taken WITHIN the same TRAIN fold the cave direction is fit on, so both
    # directions are fit on the same held-out-disjoint substrate (and the cosine bootstrap resamples it).
    sig_tr = [_signal(recs[k], DEFINITION) for k in train]
    hi_tr_local, lo_tr_local = quantile_split(sig_tr, QUANTILE)
    hi_tr = [train[j] for j in hi_tr_local]
    lo_tr = [train[j] for j in lo_tr_local]
    out["n_train"] = len(train)
    out["n_test"] = len(test)
    out["n_train_hi"] = len(hi_tr)
    out["n_train_lo"] = len(lo_tr)

    if not (hi_tr and lo_tr):
        # cannot fit u_conf (degenerate entropy-quartile split on the small argmax-W* TRAIN fold).
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        out["numbers"] = None
        out["random_numbers"] = None
        out["skipped_conf"] = "degenerate entropy-quartile train split on argmax-W* caving items"
        steer = decide_steer(None, None, None, None, None, n_fit=len(argmaxW))
        out["decision"] = model_decision(steer, None, None)
        return out

    Rn_hi = torch.stack([recs[k]["rn"][L] for k in hi_tr]).to(device)
    Rn_lo = torch.stack([recs[k]["rn"][L] for k in lo_tr]).to(device)
    u_conf = unit(P["diff_of_means"](Rn_hi, Rn_lo))
    proj_hi = statistics.mean(float(recs[k]["rn"][L].to(device) @ u_conf) for k in hi_tr)
    proj_lo = statistics.mean(float(recs[k]["rn"][L].to(device) @ u_conf) for k in lo_tr)

    # matched-magnitude random unit direction; its OWN TRAIN high-confidence mean projection is the
    # steer-up target for the matched random control.
    rnd = torch.randn(u_conf.shape, generator=g).to(u_conf.dtype).to(device)
    u_rand = unit(rnd)
    prj_hi_rand = statistics.mean(float(recs[k]["rn"][L].to(device) @ u_rand) for k in hi_tr)

    # ---- cosine(u_cave, u_conf) at L + item-bootstrap CI (resample fit items, refit both, recompute) ----
    cave_c = torch.stack([recs[k]["rc"][L] for k in train]).to(device)
    cave_n = torch.stack([recs[k]["rn"][L] for k in train]).to(device)
    cos_ci = cosine_bootstrap_ci(cave_c, cave_n, Rn_hi, Rn_lo, P)
    out["cos_cave_conf"] = cos_ci["point"]
    out["cos_ci"] = cos_ci

    # ---- STEER u_conf UP on the held-out COUNTER residual; read the realized softmax ----
    buckets, r_buckets = [], []
    dPW, r_dPW = [], []
    klS, klC, r_klS = [], [], []
    rows = []
    for k in test:
        r = recs[k]
        cid, aid = r["cid"], r["aid"]
        P1 = r["P1"].to(device)                                   # NEUTRAL (cached)
        P2 = r["P2"].to(device)                                   # COUNTER (cached)
        neu_argmax = r["neu_argmax"]
        # STEER UP: set the u_conf projection on the counter residual to the TRAIN high-confidence mean.
        hu = [(rname(L), proj_edit(u_conf, proj_hi))]
        Ps = _full_softmax(_logits(model, r["counter"], hooks=hu))
        # RANDOM matched steer-up: random direction's projection on counter -> its own TRAIN high mean.
        hr = [(rname(L), proj_edit(u_rand, prj_hi_rand))]
        Pr = _full_softmax(_logits(model, r["counter"], hooks=hr))

        amx_s, amx_r = int(Ps.argmax()), int(Pr.argmax())
        b_s = argmax_bucket(amx_s, cid, aid, neu_argmax)
        b_r = argmax_bucket(amx_r, cid, aid, neu_argmax)
        buckets.append(b_s); r_buckets.append(b_r)
        dPW.append(float(Ps[aid]) - float(P2[aid])); r_dPW.append(float(Pr[aid]) - float(P2[aid]))
        ks, kc = kl_div(Ps, P1), kl_div(P2, P1)
        klS.append(ks); klC.append(kc); r_klS.append(kl_div(Pr, P1))
        rows.append({"i": r["i"], "q": r["q"], "cid": cid, "aid": aid, "neu_argmax": neu_argmax,
                     "ctr_argmax": r["ctr_argmax"], "amx_steer": amx_s, "amx_rand": amx_r,
                     "bucket_steer": b_s, "bucket_rand": b_r,
                     "P2_W": round(float(P2[aid]), 6), "Ps_W": round(float(Ps[aid]), 6),
                     "kl_steer_to_neutral": round(ks, 6), "kl_counter_to_neutral": round(kc, 6)})
        print(f"  [{'it' if is_chat else 'base'} L{L}] item {r['i']} amx_steer={amx_s} bucket={b_s} "
              f"dP(W*)={dPW[-1]:+.4f} KLs={ks:.4f} KLc={kc:.4f}", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    def _mean(xs):
        return statistics.mean(xs) if xs else None

    bf = bucket_fracs(buckets)
    rbf = bucket_fracs(r_buckets)
    numbers = {
        "headline_layer": L, "n_train": len(train), "n_test": len(test),
        "argmax_fracs": bf,
        "dP_Wstar": (round(_mean(dPW), 6) if dPW else None),
        "kl_steer_to_neutral": (round(_mean(klS), 6) if klS else None),
        "kl_counter_to_neutral": (round(_mean(klC), 6) if klC else None),
        "proj_hi": round(proj_hi, 4), "proj_lo": round(proj_lo, 4),
        "rows": rows,
    }
    random_numbers = {
        "argmax_fracs": rbf,
        "dP_Wstar": (round(_mean(r_dPW), 6) if r_dPW else None),
        "kl_steer_to_neutral": (round(_mean(r_klS), 6) if r_klS else None),
        "kl_counter_to_neutral": (round(_mean(klC), 6) if klC else None),   # same COUNTER baseline KL
    }
    out["numbers"] = numbers
    out["random_numbers"] = random_numbers
    steer = decide_steer(bf["neutral"], numbers["dP_Wstar"], numbers["kl_steer_to_neutral"],
                         numbers["kl_counter_to_neutral"], rbf["neutral"], n_fit=len(argmaxW))
    out["decision"] = model_decision(steer, cos_ci, L)
    return out


def run(name_base, name_it, tag, device, pool):
    from headset_direction import FIT_LAYERS as _FL, _rname   # reuse the layer sweep + resid_post hook name
    P = _pure()
    layers = list(_FL)
    res = {"base": _measure_model(name_base, False, device, pool, layers, _rname, P),
           "it": _measure_model(name_it, True, device, pool, layers, _rname, P)}
    out = {
        "name_base": name_base, "name_it": name_it, "device": device, "tag": tag,
        "cue": "confidence_caving_gate_faithful", "pool_size": len(pool),
        "definition": DEFINITION, "fit_layers": layers,
        "thresholds": {"THR": THR, "COS_THR": COS_THR, "EFFECT_EPS": EFFECT_EPS, "RAND_FLOOR": RAND_FLOOR,
                       "QUANTILE": QUANTILE, "SPLIT_SEED": SPLIT_SEED, "RAND_SEED": RAND_SEED,
                       "BOOT_SEED": BOOT_SEED, "N_BOOT": N_BOOT, "MIN_FIT": MIN_FIT},
        "decision_rule": DECISION_RULE,
        "base": res["base"], "it": res["it"],
    }
    Path("out").mkdir(exist_ok=True)
    fn = f"out/confidence_caving_gate_faithful_{tag}.json"
    Path(fn).write_text(json.dumps(out, indent=2, default=str))
    for m in MODELS:
        dd = res[m]["decision"]
        s = dd["steer"]
        nb = res[m].get("numbers") or {}
        af = (nb.get("argmax_fracs") or {})
        print(f"[{m}] STEER={dd['steer_bucket']} (L{dd['headline_layer']}) "
              f"n_argmaxW={res[m]['n_argmaxW_cave']} frac_neutral={af.get('neutral')} "
              f"frac_Wstar={af.get('Wstar')} frac_C={af.get('C')} frac_third={af.get('third')} "
              f"dP(W*)={s.get('dP_Wstar')} KLs={s.get('kl_steer_to_neutral')} "
              f"KLc={s.get('kl_counter_to_neutral')} rand_neutral={s.get('rand_frac_argmax_neutral')} "
              f"| AXIS={dd['axis_bucket']} cos={dd['cos_cave_conf']} ci={dd.get('cos_ci')}", flush=True)
    print(f"[done] wrote {fn}", flush=True)


# ----------------------------------------------------------------- selftest (model-free, CPU, no model load)
def _dist(V, masses):
    """Probability vector of length V from a {idx: mass} dict; remaining mass spread uniformly over the
    rest. Pure (selftest only)."""
    rest = [j for j in range(V) if j not in masses]
    q = torch.zeros(V)
    for idx, m in masses.items():
        q[idx] = m
    if rest:
        q[rest] = (1.0 - sum(masses.values())) / len(rest)
    return q


def _plant(n, d, axis, levels, noise, gen):
    """Synthesize n residual rows carrying a planted AXIS at given per-item levels plus small isotropic
    noise: row[i] = levels[i]*axis + noise*randn. Returns [n, d]. Pure given the generator."""
    rows = noise * torch.randn(n, d, generator=gen)
    for i in range(n):
        rows[i] = rows[i] + levels[i] * axis
    return rows


def selftest():
    P = _pure()
    unit, diff_of_means, cosine = P["unit"], P["diff_of_means"], P["cosine"]
    quantile_split = P["quantile_split"]
    torch.manual_seed(0)
    V = 200
    cid, aid, third = 3, 7, 11        # C-first-tok, W*-first-tok, an arbitrary third token

    # ---------- argmax_bucket partition (priority C > Wstar > neutral > third) ----------
    assert argmax_bucket(cid, cid, aid, cid) == "C"            # argmax==C wins even if neutral argmax==C
    assert argmax_bucket(aid, cid, aid, aid) == "Wstar"        # argmax==W* (cave persists)
    assert argmax_bucket(13, cid, aid, 13) == "neutral"        # argmax==the item's neutral argmax (not C/W*)
    assert argmax_bucket(99, cid, aid, 13) == "third"          # none of the above -> third token
    assert argmax_bucket(cid, cid, aid, 13) == "C"             # C beats neutral
    bf = bucket_fracs(["neutral", "neutral", "Wstar", "third", "C"])
    assert abs(bf["C"] + bf["Wstar"] + bf["neutral"] + bf["third"] - 1.0) < 1e-9 and bf["n"] == 5, bf
    assert bf["neutral"] == 0.4 and bf["C"] == 0.2, bf
    assert bucket_fracs([]) == {"C": 0.0, "Wstar": 0.0, "neutral": 0.0, "third": 0.0, "n": 0}
    print("[selftest] argmax_bucket priority + partition + empty OK")

    # ---------- KL math: identical -> 0; closer distribution -> smaller KL ----------
    P1 = _dist(V, {cid: 0.5, aid: 0.1})           # NEUTRAL: argmax C
    assert abs(kl_div(P1, P1)) < 1e-6
    P2 = _dist(V, {aid: 0.6, cid: 0.05})          # COUNTER: argmax W*
    Ps_close = _dist(V, {cid: 0.5, aid: 0.1})     # steered back to NEUTRAL -> KL(Ps||P1) ~ 0 < KL(P2||P1)
    assert kl_div(Ps_close, P1) < kl_div(P2, P1), (kl_div(Ps_close, P1), kl_div(P2, P1))
    assert kl_div(P2, P1) > 0 and kl_div(Ps_close, P1) >= 0
    print(f"[selftest] kl_div: identical=0, steer KLs={kl_div(Ps_close, P1):.4f} < counter "
          f"KLc={kl_div(P2, P1):.4f}")

    # ---------- entropy-quartile contrast (reused) is sharp + disjoint ----------
    vals = [float(x) for x in range(12)]
    hi, lo = quantile_split(vals, QUANTILE)
    assert hi == [9, 10, 11] and lo == [0, 1, 2], (hi, lo)
    assert set(hi).isdisjoint(set(lo))
    print(f"[selftest] entropy-quartile split high={hi} low={lo} (sharp, disjoint)")

    # ---------- split primitive ----------
    tr, te = P["split_indices"](10, SPLIT_SEED)
    assert set(tr) | set(te) == set(range(10)) and not (set(tr) & set(te)) and tr and te
    assert P["split_indices"](10, SPLIT_SEED) == P["split_indices"](10, SPLIT_SEED)
    print(f"[selftest] split_indices train={tr} test={te} (disjoint, exhaustive, deterministic)")

    d = 256

    # ============================================================================================
    # (a) THE REALIZED CAVE IS A FUNCTION OF u_conf.
    #     Synthesize residual rows with a planted u_conf axis and a planted u_cave axis at a KNOWN cosine.
    #     A small fixed unembed maps a row to logits over a few token slots; the realized argmax is taken
    #     over those slots. The COUNTER row sits at the LOW u_conf level (caved: argmax = W* slot); steering
    #     the u_conf projection UP to proj_hi moves the row to the NEUTRAL/high-confidence region where the
    #     argmax is the NEUTRAL slot -> restore-to-unpushed. A matched RANDOM direction (~orthogonal to
    #     u_conf) does not move the u_conf coordinate -> argmax stays on W*. Here u_cave is COLLINEAR with
    #     u_conf (cos~1) so the cave is carried by the confidence axis -> GATES_REALIZED_CAVE + AXIS_ALIGNED.
    # ============================================================================================
    g = torch.Generator().manual_seed(11)
    u_conf_true = unit(torch.randn(d, generator=g))
    # plant high/low CONFIDENCE residuals at NEUTRAL to fit u_conf from (the entropy-quartile contrast).
    nfit = 40
    levels = [(-1.0 + 2.0 * i / (nfit - 1)) * 3.0 for i in range(nfit)]
    conf_rows = _plant(nfit, d, u_conf_true, levels, noise=0.05, gen=g)
    conf_coord = [float(conf_rows[i] @ u_conf_true) for i in range(nfit)]
    hi_idx, lo_idx = quantile_split(conf_coord, QUANTILE)
    Rn_hi = conf_rows[hi_idx]
    Rn_lo = conf_rows[lo_idx]
    u_conf_fit = unit(diff_of_means(Rn_hi, Rn_lo))
    assert abs(cosine(u_conf_fit, u_conf_true)) > 0.95, cosine(u_conf_fit, u_conf_true)
    proj_hi = statistics.mean(float(conf_rows[k] @ u_conf_fit) for k in hi_idx)
    proj_lo = statistics.mean(float(conf_rows[k] @ u_conf_fit) for k in lo_idx)

    # tiny fixed unembed: a logit per token slot = (row . slot_axis). Plant axes so that the NEUTRAL slot
    # dominates at HIGH u_conf coordinate and the W* slot dominates at LOW u_conf coordinate.
    slot_neu, slot_w, slot_c = 0, 1, 2
    W_slots = torch.zeros(3, d)
    W_slots[slot_neu] = (proj_hi * 0.6) * u_conf_true        # neutral slot grows with the u_conf coordinate
    W_slots[slot_w] = (-proj_hi * 0.6) * u_conf_true + 4.0 * unit(torch.randn(d, generator=g))  # W* slot
    W_slots[slot_c] = 2.0 * unit(torch.randn(d, generator=g))                                   # C slot

    def realized_argmax(row):
        logit = torch.tensor([float(row @ W_slots[j]) for j in range(3)])
        slot = int(logit.argmax())
        return {0: NEU_ID, 1: aid, 2: cid}[slot]
    NEU_ID = 17                                              # the NEUTRAL-condition argmax token id (not C/W*)
    # COUNTER row: low u_conf coordinate (caved). build it to actually emit W* at the counter.
    counter_row = proj_lo * u_conf_true + 0.5 * unit(torch.randn(d, generator=g))
    # bias the W* slot so the counter argmax is W* (the model would emit W*).
    W_slots[slot_w] = W_slots[slot_w] + 6.0 * unit(counter_row)
    assert realized_argmax(counter_row) == aid, "counter row must emit W* (argmax-W* substrate)"
    # STEER UP: set the u_conf coordinate to proj_hi -> the neutral slot dominates -> argmax = NEU_ID.
    cur = float(counter_row @ u_conf_fit)
    steered_row = counter_row + (proj_hi - cur) * u_conf_fit
    amx_s = realized_argmax(steered_row)
    b_s = argmax_bucket(amx_s, cid, aid, NEU_ID)
    # RANDOM matched steer-up: ~orthogonal to u_conf -> u_conf coordinate ~unchanged -> argmax stays W*.
    grand = torch.Generator().manual_seed(RAND_SEED)
    u_rand = unit(torch.randn(d, generator=grand))
    prj_hi_rand = statistics.mean(float(conf_rows[k] @ u_rand) for k in hi_idx)
    cur_r = float(counter_row @ u_rand)
    rand_row = counter_row + (prj_hi_rand - cur_r) * u_rand
    amx_r = realized_argmax(rand_row)
    b_r = argmax_bucket(amx_r, cid, aid, NEU_ID)
    assert b_s == "neutral", f"(a) steering u_conf up should restore the neutral argmax, got {b_s}"
    assert b_r == "Wstar", f"(a) random matched steer should leave the argmax on W*, got {b_r}"
    # KL: steered distribution close to neutral; counter far. (use the realized softmaxes over the 3 slots.)
    P1a = _dist(V, {NEU_ID: 0.6, cid: 0.1, aid: 0.05})
    P2a = _dist(V, {aid: 0.6, cid: 0.05})
    Psa = _dist(V, {NEU_ID: 0.58, cid: 0.1, aid: 0.06})
    kl_s, kl_c = kl_div(Psa, P1a), kl_div(P2a, P1a)
    dPW_a = float(Psa[aid]) - float(P2a[aid])
    steer_a = decide_steer(bucket_fracs([b_s])["neutral"], dPW_a, kl_s, kl_c,
                           bucket_fracs([b_r])["neutral"], n_fit=8)
    assert steer_a["category"] == "GATES_REALIZED_CAVE", steer_a
    print(f"[selftest] (a) realized cave==f(u_conf): bucket_steer={b_s} bucket_rand={b_r} "
          f"dP(W*)={dPW_a:+.3f} KLs<KLc={kl_s < kl_c} -> {steer_a['category']}")

    # AXIS arm: here u_cave is collinear with u_conf (cave carried by the confidence axis) -> AXIS_ALIGNED.
    cave_c = _plant(12, d, u_conf_true, [4.0] * 12, noise=0.05, gen=g)       # counter rows: +u_conf coord
    cave_n = _plant(12, d, u_conf_true, [0.0] * 12, noise=0.05, gen=g)       # neutral rows: 0 coord
    cos_ci_a = cosine_bootstrap_ci(cave_c, cave_n, Rn_hi, Rn_lo, P, n_boot=200)
    assert abs(cos_ci_a["point"]) >= COS_THR, cos_ci_a
    assert decide_axis(cos_ci_a["point"]) == "AXIS_ALIGNED", cos_ci_a
    assert cos_ci_a["lo"] is not None and cos_ci_a["hi"] is not None and cos_ci_a["lo"] <= cos_ci_a["hi"]
    print(f"[selftest] (a) cos(u_cave,u_conf)={cos_ci_a['point']:.3f} CI=[{cos_ci_a['lo']},{cos_ci_a['hi']}]"
          f" -> {decide_axis(cos_ci_a['point'])}")

    # ============================================================================================
    # (b) THE REALIZED CAVE IS CARRIED BY AN AXIS ORTHOGONAL TO u_conf.
    #     The W*/neutral slot logits depend on a cave axis e_cave ORTHOGONAL to u_conf. Steering the u_conf
    #     projection up moves the u_conf coordinate but NOT the cave coordinate -> the argmax stays on W* ->
    #     INDEPENDENT_REALIZED, and cos(u_cave, u_conf) ~ 0 -> AXIS_ORTHOGONAL.
    # ============================================================================================
    e_conf = torch.zeros(d); e_conf[0] = 1.0
    e_cave = torch.zeros(d); e_cave[1] = 1.0                  # orthogonal to e_conf
    # fit u_conf from a high/low confidence contrast along e_conf (recovers e_conf)
    gb = torch.Generator().manual_seed(5)
    lv = [3.0] * 10 + [-3.0] * 10
    rconf = _plant(20, d, e_conf, lv, noise=0.02, gen=gb)
    cc = [float(rconf[i] @ e_conf) for i in range(20)]
    hq, lq = quantile_split(cc, 0.5)
    u_conf_b = unit(diff_of_means(rconf[hq], rconf[lq]))
    assert abs(cosine(u_conf_b, e_conf)) > 0.99
    proj_hi_b = statistics.mean(float(rconf[k] @ u_conf_b) for k in hq)
    # realized argmax driven ONLY by the cave coordinate (along e_cave), independent of u_conf.
    Wb = torch.zeros(3, d)
    Wb[0] = -3.0 * e_cave        # neutral slot dominates when cave coord is LOW (negative)
    Wb[1] = 3.0 * e_cave         # W* slot dominates when cave coord is HIGH (positive == caved)
    Wb[2] = 0.5 * unit(torch.randn(d, generator=gb))

    def realized_b(row):
        logit = torch.tensor([float(row @ Wb[j]) for j in range(3)])
        slot = int(logit.argmax())
        return {0: NEU_ID, 1: aid, 2: cid}[slot]
    counter_b = (2.0) * e_cave + (-2.0) * u_conf_b + 0.02 * torch.randn(d, generator=gb)  # caved, low u_conf
    assert realized_b(counter_b) == aid, "(b) counter row must emit W*"
    cur_b = float(counter_b @ u_conf_b)
    steered_b = counter_b + (proj_hi_b - cur_b) * u_conf_b   # u_conf coord up; cave coord untouched
    amx_sb = realized_b(steered_b)
    b_sb = argmax_bucket(amx_sb, cid, aid, NEU_ID)
    assert b_sb == "Wstar", f"(b) orthogonal-axis cave should persist on W*, got {b_sb}"
    # KL: steering did not move the distribution back toward neutral (argmax still W*).
    P1b = _dist(V, {NEU_ID: 0.6, cid: 0.1, aid: 0.05})
    P2b = _dist(V, {aid: 0.6, cid: 0.05})
    Psb = _dist(V, {aid: 0.58, cid: 0.06})                   # steered ~ still caved
    steer_b = decide_steer(bucket_fracs([b_sb])["neutral"], float(Psb[aid]) - float(P2b[aid]),
                           kl_div(Psb, P1b), kl_div(P2b, P1b), 0.0, n_fit=8)
    assert steer_b["category"] == "INDEPENDENT_REALIZED", steer_b
    # cosine: u_cave (along e_cave) orthogonal to u_conf (along e_conf) -> ~0 -> AXIS_ORTHOGONAL.
    cave_c_b = _plant(12, d, e_cave, [4.0] * 12, noise=0.02, gen=gb)
    cave_n_b = _plant(12, d, e_cave, [0.0] * 12, noise=0.02, gen=gb)
    cos_ci_b = cosine_bootstrap_ci(cave_c_b, cave_n_b, rconf[hq], rconf[lq], P, n_boot=200)
    assert abs(cos_ci_b["point"]) < COS_THR, cos_ci_b
    assert decide_axis(cos_ci_b["point"]) == "AXIS_ORTHOGONAL", cos_ci_b
    print(f"[selftest] (b) realized cave _|_ u_conf: bucket_steer={b_sb} -> {steer_b['category']}; "
          f"cos={cos_ci_b['point']:.3f} -> {decide_axis(cos_ci_b['point'])}")

    # ============================================================================================
    # (c) EXACT BOUNDARY CHECKS on THR / COS_THR / RAND_FLOOR / EFFECT_EPS + assembled per-model dict +
    #     the INSUFFICIENT (no argmax-W* items) path.
    # ============================================================================================
    # decide_steer THR boundary (frac_neutral): exactly at THR fires (KL back + clean random).
    assert decide_steer(THR, -0.3, 0.1, 0.9, 0.0, n_fit=8)["category"] == "GATES_REALIZED_CAVE"
    assert decide_steer(THR - 1e-6, -0.3, 0.1, 0.9, 0.0, n_fit=8)["category"] == "INDEPENDENT_REALIZED"
    # KL must move TOWARD neutral (KL_steer < KL_counter); equal/greater -> INDEPENDENT.
    assert decide_steer(0.9, -0.3, 0.9, 0.9, 0.0, n_fit=8)["category"] == "INDEPENDENT_REALIZED"   # KLs==KLc
    assert decide_steer(0.9, -0.3, 1.0, 0.9, 0.0, n_fit=8)["category"] == "INDEPENDENT_REALIZED"   # KLs>KLc
    # RAND_FLOOR: random restore-frac must be STRICTLY below the floor.
    assert decide_steer(0.9, -0.3, 0.1, 0.9, RAND_FLOOR, n_fit=8)["category"] == "INDEPENDENT_REALIZED"
    assert decide_steer(0.9, -0.3, 0.1, 0.9, RAND_FLOOR - 1e-6, n_fit=8)["category"] == "GATES_REALIZED_CAVE"
    assert decide_steer(0.9, -0.3, 0.1, 0.9, None, n_fit=8)["category"] == "GATES_REALIZED_CAVE"   # None rand=clean
    # EFFECT_EPS reporting: a steered argmax that returns to neutral with a tiny W* move still classified by
    # the restore-frac/KL/random rule; wstar_mass_moved flags the |dP(W*)| >= EFFECT_EPS condition.
    d_small = decide_steer(0.9, -(EFFECT_EPS - 1e-4), 0.1, 0.9, 0.0, n_fit=8)
    d_big = decide_steer(0.9, -(EFFECT_EPS + 1e-4), 0.1, 0.9, 0.0, n_fit=8)
    assert d_small["wstar_mass_moved"] is False and d_big["wstar_mass_moved"] is True, (d_small, d_big)
    assert d_small["category"] == "GATES_REALIZED_CAVE" and d_big["category"] == "GATES_REALIZED_CAVE"
    print(f"[selftest] (c) decide_steer thresholds exact (frac>=THR AND KLs<KLc AND rand<RAND_FLOOR); "
          f"EFFECT_EPS flags wstar_mass_moved={d_small['wstar_mass_moved']}/{d_big['wstar_mass_moved']}")

    # decide_axis COS_THR boundary.
    assert decide_axis(COS_THR) == "AXIS_ALIGNED" and decide_axis(COS_THR - 1e-6) == "AXIS_ORTHOGONAL"
    assert decide_axis(-COS_THR) == "AXIS_ALIGNED"            # sign-agnostic (|cos|)
    assert decide_axis(0.0) == "AXIS_ORTHOGONAL" and decide_axis(None) == "AXIS_ORTHOGONAL"
    print("[selftest] (c) decide_axis: |cos|>=COS_THR AXIS_ALIGNED else AXIS_ORTHOGONAL (sign-agnostic)")

    # INSUFFICIENT path: no argmax-W* items (n_fit < MIN_FIT).
    ins = decide_steer(None, None, None, None, None, n_fit=2)
    assert ins["category"] == "INSUFFICIENT" and ins["n_fit"] == 2, ins
    assert decide_steer(None, None, None, None, None, n_fit=0)["category"] == "INSUFFICIENT"
    print(f"[selftest] (c) INSUFFICIENT when n_fit < MIN_FIT({MIN_FIT}) -> {ins['category']}")

    # assembled per-model decision dicts.
    md_a = model_decision(steer_a, cos_ci_a, headline_layer=28)
    assert md_a["steer_bucket"] == "GATES_REALIZED_CAVE" and md_a["axis_bucket"] == "AXIS_ALIGNED" \
        and md_a["headline_layer"] == 28 and md_a["cos_cave_conf"] is not None, md_a
    md_b = model_decision(steer_b, cos_ci_b, headline_layer=32)
    assert md_b["steer_bucket"] == "INDEPENDENT_REALIZED" and md_b["axis_bucket"] == "AXIS_ORTHOGONAL", md_b
    md_ins = model_decision(ins, None, headline_layer=None)
    assert md_ins["steer_bucket"] == "INSUFFICIENT" and md_ins["axis_bucket"] == "AXIS_ORTHOGONAL" \
        and md_ins["cos_cave_conf"] is None, md_ins
    print(f"[selftest] (c) assembled decisions: A={md_a['steer_bucket']}/{md_a['axis_bucket']} "
          f"B={md_b['steer_bucket']}/{md_b['axis_bucket']} INS={md_ins['steer_bucket']}/"
          f"{md_ins['axis_bucket']}")

    # cosine_bootstrap_ci edge: empty set -> point None.
    empty = cosine_bootstrap_ci(torch.zeros(0, d), torch.zeros(0, d), Rn_hi, Rn_lo, P)
    assert empty["point"] is None and empty["n_boot"] == 0, empty
    print("[selftest] cosine_bootstrap_ci empty-set -> point None")
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
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root for the item pool
        from misconception_pool import ITEMS_WIDE
        run(args.name_base, args.name_it, args.tag, args.device, ITEMS_WIDE)


if __name__ == "__main__":
    main()
