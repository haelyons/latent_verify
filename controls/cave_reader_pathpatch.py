"""CAVE-READER PATH-PATCH: when the rank-1 cave-direction u_cave is removed at the headline layer L, WHICH
downstream components carry the resulting answer-restoration to the logits -- a few (a localized reader
circuit edge) or many (distributed)? And how much acts via the DIRECT residual->unembed path vs through
downstream components? (sibling of cave_suppress_vs_install.py / cave_carrier_deconfound.py /
faithful_caving.py / cave_direction_dla.py / headset_direction.py.)

CONTEXT (neutral). A prior control (cave_suppress_vs_install.py) established that at base (Q/A), on the
held-out argmax-W* caving items (the counter argmax IS the W*-aligned first-token), ablating the rank-1
diff-of-means cave direction
  u_cave(L) = normalize(mean_items( resid_post[L][-1](counter) - resid_post[L][-1](neutral) ))
by SETTING its projection on the counter residual to the TRAIN-neutral mean returns the emitted argmax to
the model's UNPUSHED (NEUTRAL) answer on the held-out items (full restoration ~1.0), while a matched random
direction does nothing. The headline layer L = the FIT_LAYERS layer with the max in-sample u_cave necessity
(the same headline-selection rule as cave_suppress_vs_install / headset_direction); on 9b it sits near the
top (e.g. L36 of 42), so there are only a FEW downstream layers.

This control path-patches the READER side of that established SENDER intervention, with NO new mechanism.
SENDER = delta, the REMOVAL of the u_cave component at resid_post[L][-1]: delta is the residual change
(ablate u_cave projection -> train-neutral mean) MINUS clean. We never apply a new direction; we re-use the
exact projection-edit ablation of cave_suppress_vs_install (the established full restoration). We then ask
how that delta reaches the logits, forward-only, last-token:

CONDITIONS (read the realized next-token softmax P; restoration is measured against the NEUTRAL distribution
P1 and the per-item NEUTRAL-condition argmax):
  1. CLEAN counter (cave present; argmax = W*-first-tok)                                          P2.
  2. FULL ablation: delta applied at resid_post[L][-1], normal forward to the logits.             P_full.
     (= the established full restoration; argmax -> the item's neutral argmax, ~1.0 at base.)
  3. DIRECT-PATH only: apply delta at resid_post[L][-1] but FREEZE every downstream component (every
     attention head's hook_z[L'][-1] and every hook_mlp_out[L'][-1], L'>L) to its CLEAN (cave-present)
     value, so delta reaches the logits ONLY through the direct residual->final-LN->unembed path (no
     downstream attn/MLP re-reads the delta-modified residual).                                   P_direct.
  4. PER-DOWNSTREAM-COMPONENT path patch: for each receiver R in layers L'>L (each attention head L'.H and
     each MLP L'), let ONLY R recompute from the delta-modified residual while EVERY OTHER downstream
     component is frozen to its CLEAN value. (Standard single-edge path patch: the receiver reads
     resid_pre[L'] = clean_resid_pre[L'] + delta because all intermediate downstream writes are frozen to
     clean; only R's changed output then flows on, with everything downstream of R also frozen to clean, so
     R's effect reaches the logits alone.) effect(R) = frac(argmax == neutral-argmax) under R-alone, and the
     logit-restoration fraction (how much of the full direct->indirect logit move R reproduces).
  5. matched-RANDOM direction delta (u_rand -> its own train-neutral mean) through the same DIRECT path and
     the same per-component path patches -- the specificity floor.

DECOMPOSITION read: the DIRECT-path restoration (3) vs the FULL restoration (2) gives the direct fraction;
the INDIRECT restoration is what the downstream components carry; the per-component effect(R) ranking, its
top-TOPK concentration, the attn-vs-MLP split, and the joint-top-k restoration (all top-TOPK receivers
recomputing together) say whether the indirect restoration is localized to a few reader components or
spread over many.

This is claim-blind: it measures where, downstream of the u_cave-removal at L, the answer-restoration is
carried. It attaches no hypothesis to the direct fraction, any component, the attn-vs-MLP split, or the
base-vs-it comparison. (-it is run for completeness; we expect 0 argmax-W* caving items there -> INSUFFICIENT,
reported as such, exactly as cave_suppress_vs_install / cave_carrier_deconfound find.)

NEUTRAL DECISION (module constants DIRECT_THR=0.5, CONC_FRAC=0.5, TOPK=5; numbers + categories only, no
hypothesis):
  - INSUFFICIENT iff n_argmaxW < MIN_FIT (no held-out argmax-W* substrate).
  - NO_RESTORATION iff the FULL ablation (2) does not restore the neutral argmax (restoration_full < a small
    floor): there is no answer-restoration to attribute.
  - DIRECT_WRITE iff the DIRECT-PATH restoration (3) >= DIRECT_THR of the FULL restoration (2): u_cave acts
    (mostly) directly on the logits -> minimal reader circuit (it is effectively a logit-write direction).
  - LOCALIZED_READERS iff NOT DIRECT_WRITE AND the top-TOPK downstream components by effect(R) carry
    >= CONC_FRAC of the total downstream restoration AND jointly reconstruct it (joint-top-k restoration
    >= CONC_FRAC of the indirect restoration) -> a reader-circuit edge u_cave -> {those components}.
  - DISTRIBUTED_READERS iff the indirect restoration is spread (top-TOPK carry < CONC_FRAC; many needed)
    -> no clean reader.
  The matched RANDOM direction must NOT reproduce the effect (its direct fraction / per-component effects are
  reported as the floor). Report the direct-path fraction, the per-component effect ranking (top-TOPK with
  layer/type), the attn-vs-MLP split of the indirect effect, and the joint-top-k restoration. Per model.

Forward-only (diff-of-means + projection-edit delta + clean-component freeze + per-receiver path patch +
full-softmax readouts; no backward) -> fits the 40GB A100. Reuses verified primitives: PUSH/NEUTRAL from
job_truthful_flip; _helpers/MIN_EFFECT_NET from rlhf_differential; ITEMS_WIDE from misconception_pool; the
held-out fold split / diff-of-means cave fit / projection-edit ablation / full-softmax + first-token readouts
/ KL helper / headline-layer selection / argmax-W* caving-item selection construction from
cave_suppress_vs_install (re-implemented here verbatim as small pure helpers so --selftest is standalone on
CPU with nothing else on sys.path -- the same FLAT-scp convention cave_suppress_vs_install /
cave_carrier_deconfound / faithful_caving use); the per-head hook_z@W_O + hook_mlp_out last-token component
access from cave_direction_dla; FIT_LAYERS / _rname from headset_direction (deferred at real-run time). The
new logic is the clean-downstream-component freeze hooks, the direct-path and per-receiver path-patch
forwards, and the direct-vs-indirect / concentration / joint-reconstruction decision -- all pure parts are
covered by the model-free --selftest.

  python controls/cave_reader_pathpatch.py --selftest
  python controls/cave_reader_pathpatch.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it \
    --tag 9b --device cuda --chat
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

# Pre-registered thresholds (see module docstring). Neutral: stated on the measured numbers only.
DIRECT_THR = 0.5     # DIRECT-path restoration >= this fraction of the FULL restoration -> DIRECT_WRITE
CONC_FRAC = 0.5      # top-TOPK downstream components carry >= this fraction of the indirect restoration -> localized
TOPK = 5             # #top downstream receivers for the concentration decision + the reported shortlist
RESTORE_FLOOR = 0.05  # FULL-ablation restoration below this -> NO_RESTORATION (nothing to attribute)
SPLIT_SEED = 0       # deterministic train/test fold (same convention as cave_suppress_vs_install)
RAND_SEED = 0        # deterministic matched-random-direction control
MIN_FIT = 3          # below this many argmax-W* caving items, no held-out direction can be fit/tested

# Diff-of-means cave-direction layer sweep. SAME value as headset_direction.FIT_LAYERS /
# cave_suppress_vs_install.FIT_LAYERS / cave_carrier_deconfound.FIT_LAYERS ([24,28,32,36], the set's output
# range L21-L34); defined here as a module constant so --selftest needs nothing on sys.path. The real run
# also defers a `from headset_direction import FIT_LAYERS` so the sweep stays pinned to the reference.
FIT_LAYERS = [24, 28, 32, 36]

MODELS = ("base", "it")

DECISION_RULE = (
    "On the wide misconception pool, build NEUTRAL and COUNTER (W* asserted) prompts (job_truthful_flip "
    "turns; qa template for base, chat template for -it). Restrict to the argmax-W* CAVING items (counter "
    "lowers M=logp(C)-logp(W*) from neutral by >= MIN_EFFECT_NET AND counter argmax IS the W*-first-token), "
    "the same selection as cave_suppress_vs_install. Fit u_cave = normalize(mean(resid_post[L][-1](counter)-"
    "resid_post[L][-1](neutral))) over a TRAIN fold; headline layer L = max in-sample u_cave necessity over "
    "the train fold. The SENDER = delta = (ablate u_cave projection at resid_post[L][-1] to the train-neutral "
    "mean) - clean. On the held-out TEST counter residual read the realized next-token softmax under: "
    "(1) CLEAN counter (P2); (2) FULL: delta at L, normal forward (P_full); (3) DIRECT: delta at L with every "
    "downstream component (hook_z[L'][-1] per head, hook_mlp_out[L'][-1], L'>L) frozen to its CLEAN value "
    "(P_direct, delta reaches the logits only via resid->final-LN->unembed); (4) per receiver R (each head "
    "L'.H, each MLP L', L'>L): only R recomputes from the delta-modified residual, every other downstream "
    "component frozen to clean (effect(R)=frac(argmax==neutral-argmax), logit-restoration fraction); (5) the "
    "same DIRECT + per-receiver patches through a matched RANDOM direction (floor). restoration := "
    "frac(argmax==neutral-argmax). direct_fraction = restoration_direct / restoration_full. INDIRECT = "
    "restoration_full - restoration_direct. "
    "Per model (DIRECT_THR=0.5, CONC_FRAC=0.5, TOPK=5): INSUFFICIENT iff n_argmaxW<MIN_FIT(3); else "
    "NO_RESTORATION iff restoration_full<RESTORE_FLOOR(0.05); else DIRECT_WRITE iff direct_fraction>=DIRECT_THR; "
    "else LOCALIZED_READERS iff the top-TOPK downstream receivers by effect(R) carry >=CONC_FRAC of the total "
    "downstream restoration AND jointly reconstruct it (joint-top-k restoration >= CONC_FRAC of the indirect "
    "restoration); else DISTRIBUTED_READERS. The matched RANDOM direction must not reproduce the effect "
    "(reported as the floor). Reported for base and -it; numbers + categories only, no claim attached to the "
    "direct fraction, any component, the attn-vs-MLP split, or the base-vs-it comparison."
)


# --------------------------------------------------------------------------- pure direction / fold helpers
def unit(v, eps=1e-8):
    """Unit vector; pure (tensor -> tensor). (cave_suppress_vs_install.unit / cave_direction_dla.fit_u.)"""
    return v / (v.norm() + eps)


def diff_of_means(pos, neg):
    """mean(pos) - mean(neg) as an (unnormalized) direction. pos/neg are [n_i, d] stacks. Pure.
    (cave_suppress_vs_install.diff_of_means; the diff-of-means cave fit of headset_direction.)"""
    return pos.mean(0) - neg.mean(0)


def split_indices(n, seed=SPLIT_SEED):
    """Deterministic ~50/50 train/test fold over n indices (cave_suppress_vs_install.split_indices /
    cave_carrier_deconfound.split_indices). Disjoint + exhaustive; both folds non-empty for n>=2 (n==1
    fallback maps train==test, selftest only). Pure."""
    import random as _r
    idx = list(range(n))
    _r.Random(seed).shuffle(idx)
    half = max(1, n // 2)
    train = sorted(idx[:half])
    test = sorted(idx[half:]) if n - half > 0 else sorted(idx[:half])
    return train, test


# --------------------------------------------------------------------------- pure distribution math
def kl_div(p, q, eps=1e-12):
    """KL(p || q) = sum_v p(v) * log(p(v)/q(v)) over 1-D probability tensors. Non-negative for proper
    distributions. Pure (clamps both to eps; same per-coordinate form as cave_suppress_vs_install.kl_div).
    Used for KL(P_cond||P1): distance of each condition's distribution to the NEUTRAL one."""
    pp = p.float().clamp_min(eps)
    qq = q.float().clamp_min(eps)
    return float((pp * (pp.log() - qq.log())).sum())


# --------------------------------------------------------------------------- pure direct-vs-indirect split
def restoration_frac(buckets):
    """Fraction of items whose post-intervention argmax == the item's NEUTRAL-condition argmax, over a list
    of bool flags (one per item). 0.0 for an empty list. Pure -- this is the single 'restoration' number used
    for the FULL, DIRECT, per-component, and joint-top-k conditions and for the random floor."""
    n = len(buckets)
    return (sum(1 for b in buckets if b) / n) if n else 0.0


def direct_indirect_split(restoration_full, restoration_direct):
    """Decompose the FULL restoration into the DIRECT residual->unembed share and the INDIRECT (carried by
    downstream components) share. Pure (floats in, dict out). direct_fraction is the share of the full
    restoration that the direct path alone reproduces; indirect = full - direct (clamped at 0 for reporting,
    since a direct path that over-restores is logged but never makes indirect negative for the concentration
    denominator). None-safe."""
    if restoration_full is None or restoration_direct is None:
        return {"restoration_full": None, "restoration_direct": None,
                "direct_fraction": None, "indirect_restoration": None}
    direct_fraction = (restoration_direct / restoration_full) if restoration_full > 0 else 0.0
    indirect = max(0.0, restoration_full - restoration_direct)
    return {"restoration_full": round(restoration_full, 6),
            "restoration_direct": round(restoration_direct, 6),
            "direct_fraction": round(direct_fraction, 6),
            "indirect_restoration": round(indirect, 6)}


def concentration(effects, topk=TOPK):
    """Fraction of the total sum of per-component effects carried by the top-`topk` receivers (by effect
    magnitude). effects: list of non-negative floats (effect(R) = the receiver's restoration). Returns
    (frac, total, top_sum, ranked_indices). Pure. total==0 -> frac 0.0 (nothing carries; not concentrated).
    frac is in [0,1] and nondecreasing in topk. (cave_direction_dla.concentration.)"""
    total = float(sum(effects))
    order = sorted(range(len(effects)), key=lambda i: effects[i], reverse=True)
    kk = min(topk, len(order))
    top_sum = float(sum(effects[i] for i in order[:kk]))
    frac = (top_sum / total) if total > 0 else 0.0
    return frac, total, top_sum, order


def split_attn_mlp(keys, vals):
    """Sum of vals over attn-head receivers vs mlp receivers, plus their fractions of the grand total.
    keys: list of receiver-key dicts each carrying {"type": "attn"|"mlp", ...}; vals the per-receiver effect.
    Pure (numbers + categories). (cave_direction_dla.split_attn_mlp, on non-negative effects.)"""
    attn = sum(v for k, v in zip(keys, vals) if k["type"] == "attn")
    mlp = sum(v for k, v in zip(keys, vals) if k["type"] == "mlp")
    tot = attn + mlp
    return {"attn_effect": round(attn, 6), "mlp_effect": round(mlp, 6),
            "attn_frac": (round(attn / tot, 6) if tot > 0 else None),
            "mlp_frac": (round(mlp / tot, 6) if tot > 0 else None)}


def layer_band(top_keys):
    """(min_layer, max_layer) of the top-receiver shortlist (layer = each receiver's layer index). Pure.
    Empty -> (None, None). (cave_direction_dla.layer_band.)"""
    if not top_keys:
        return (None, None)
    layers = [k["layer"] for k in top_keys]
    return (min(layers), max(layers))


# --------------------------------------------------------------------------- pure decision
def decide_reader(restoration_full, restoration_direct, comp_effects, comp_keys, joint_topk_restoration,
                  n_fit=None, min_fit=MIN_FIT, direct_thr=DIRECT_THR, conc_frac=CONC_FRAC, topk=TOPK,
                  restore_floor=RESTORE_FLOOR):
    """Reader-circuit decision over the measured numbers only (no hypothesis attached). Pure.
      INSUFFICIENT        iff n_fit is not None and n_fit < min_fit (no held-out argmax-W* substrate).
      NO_RESTORATION      iff restoration_full < restore_floor (the FULL ablation does not restore the neutral
                              argmax; there is no answer-restoration to attribute).
      DIRECT_WRITE        iff direct_fraction = restoration_direct/restoration_full >= direct_thr (u_cave acts
                              mostly directly on the logits -> minimal reader circuit).
      LOCALIZED_READERS   iff NOT DIRECT_WRITE AND the top-topk downstream receivers carry >= conc_frac of the
                              total downstream restoration (sum of effect(R)) AND jointly reconstruct it
                              (joint-top-k restoration >= conc_frac of the INDIRECT restoration).
      DISTRIBUTED_READERS otherwise (the indirect restoration is spread; top-topk carry < conc_frac).
    `comp_effects` is a list of non-negative per-receiver restorations effect(R); `comp_keys` the aligned
    receiver-key dicts (type/layer/head/key); `joint_topk_restoration` the restoration when all top-topk
    receivers recompute together. Returns the full numbers + the headline category."""
    split = direct_indirect_split(restoration_full, restoration_direct)
    if n_fit is not None and n_fit < min_fit:
        return {"category": "INSUFFICIENT", "n_fit": n_fit, **split,
                "conc_frac_at_topk": None, "downstream_total_effect": None,
                "joint_topk_restoration": None, "top_receivers": [],
                "attn_vs_mlp_split_of_indirect": None, "receiver_layer_band": [None, None],
                "msg": f"only {n_fit} held-out argmax-W* item(s) < MIN_FIT({min_fit}); no substrate to test."}

    if restoration_full is None or restoration_full < restore_floor:
        return {"category": "NO_RESTORATION", "n_fit": n_fit, **split,
                "conc_frac_at_topk": None, "downstream_total_effect": None,
                "joint_topk_restoration": (round(joint_topk_restoration, 6)
                                           if joint_topk_restoration is not None else None),
                "top_receivers": [], "attn_vs_mlp_split_of_indirect": None,
                "receiver_layer_band": [None, None],
                "msg": (f"FULL-ablation restoration "
                        f"{None if restoration_full is None else round(restoration_full, 4)} < "
                        f"RESTORE_FLOOR({restore_floor}): the u_cave removal does not restore the neutral "
                        f"argmax on the held-out items; no answer-restoration to attribute.")}

    effects = [max(0.0, e) for e in (comp_effects or [])]
    conc, total, top_sum, order = concentration(effects, topk)
    top_keys = [comp_keys[i] for i in order[:min(topk, len(order))]]
    top_receivers = [{"key": comp_keys[i]["key"], "type": comp_keys[i]["type"],
                      "layer": comp_keys[i]["layer"], "head": comp_keys[i]["head"],
                      "effect": round(effects[i], 6)} for i in order[:min(topk, len(order))]]
    split_am = split_attn_mlp(top_keys + [comp_keys[i] for i in order[min(topk, len(order)):]],
                              [effects[i] for i in order]) if effects else None
    band = layer_band(top_keys)
    direct_fraction = split["direct_fraction"]
    indirect = split["indirect_restoration"]

    direct_write = (direct_fraction is not None) and (direct_fraction >= direct_thr)
    # joint reconstruction: the top-topk receivers, recomputing together, must reproduce >= conc_frac of the
    # INDIRECT restoration (so a high per-component concentration that does NOT jointly reconstruct -- e.g.
    # redundant receivers each individually restoring but not summing -- does not count as LOCALIZED).
    joint_ok = (joint_topk_restoration is not None and indirect is not None and indirect > 0
                and joint_topk_restoration >= conc_frac * indirect)
    concentrated = (conc >= conc_frac)

    if direct_write:
        cat = "DIRECT_WRITE"
        msg = (f"DIRECT-path restoration {split['restoration_direct']} is {direct_fraction:.3f} of the FULL "
               f"restoration {split['restoration_full']} (>= {direct_thr}): u_cave acts mostly directly on "
               f"the logits via resid->final-LN->unembed -> minimal reader circuit (effectively a "
               f"logit-write direction).")
    elif concentrated and joint_ok:
        cat = "LOCALIZED_READERS"
        msg = (f"direct fraction {direct_fraction:.3f} < {direct_thr}; the top-{topk} downstream receivers "
               f"carry conc@{topk}={conc:.3f} (>= {conc_frac}) of the total downstream effect {total:.4f} AND "
               f"jointly reconstruct {joint_topk_restoration:.4f} >= {conc_frac} x indirect {indirect:.4f}: a "
               f"reader-circuit edge u_cave -> {{{', '.join(r['key'] for r in top_receivers)}}}.")
    else:
        cat = "DISTRIBUTED_READERS"
        msg = (f"direct fraction {direct_fraction:.3f} < {direct_thr}; the indirect restoration is spread "
               f"(top-{topk} conc@{topk}={conc:.3f} {'>=' if concentrated else '<'} {conc_frac}, joint-top-k "
               f"{None if joint_topk_restoration is None else round(joint_topk_restoration, 4)} "
               f"{'>=' if joint_ok else '<'} {conc_frac} x indirect "
               f"{None if indirect is None else round(indirect, 4)}): no clean reader -- many components "
               f"needed.")
    return {"category": cat, "n_fit": n_fit, **split,
            "direct_write": cat == "DIRECT_WRITE", "localized_readers": cat == "LOCALIZED_READERS",
            "distributed_readers": cat == "DISTRIBUTED_READERS",
            "conc_frac_at_topk": round(conc, 6), "downstream_total_effect": round(total, 6),
            "concentrated": bool(concentrated), "joint_reconstructs": bool(joint_ok),
            "joint_topk_restoration": (round(joint_topk_restoration, 6)
                                       if joint_topk_restoration is not None else None),
            "top_receivers": top_receivers,
            "attn_vs_mlp_split_of_indirect": split_am,
            "receiver_layer_band": list(band),
            "msg": msg}


# --------------------------------------------------------------------------- real-run hook names
def _rname(L):
    """resid_post hook name at layer L (cave_suppress_vs_install._rname / headset_direction._rname)."""
    return f"blocks.{L}.hook_resid_post"


def _zname(L):
    """attn hook_z name at layer L (cave_direction_dla._zname). z is [batch, seq, n_head, d_head]."""
    return f"blocks.{L}.attn.hook_z"


def _mname(L):
    """hook_mlp_out name at layer L (cave_direction_dla._mname). [batch, seq, d_model]."""
    return f"blocks.{L}.hook_mlp_out"


# --------------------------------------------------------------------------- real-run readout helpers
def _full_softmax(logits):
    """Full next-token probability vector at the LAST position (cave_suppress_vs_install._full_softmax).
    gemma-2's final softcap is applied inside the forward, so softmax(logits[0,-1]) is the realized
    next-token distribution. Returns a 1-D float tensor."""
    return torch.softmax(logits[0, -1].float(), dim=-1)


def _logp_diff_local(logits, cid, aid):
    """First-token margin M = logp(C) - logp(W*) at the last position (cave_suppress_vs_install /
    faithful_caving._logp_diff_local)."""
    lp = torch.log_softmax(logits[0, -1].float(), -1)
    return float(lp[cid] - lp[aid])


def _readout(P, cid, aid):
    """Realized readout from a full softmax P: argmax token id, P(C first-tok), P(W* first-tok). Pure.
    (cave_suppress_vs_install._readout.)"""
    return {"argmax": int(P.argmax()), "p_c": float(P[cid]), "p_w": float(P[aid])}


def _proj_edit_hook(u, target_proj):
    """Hook that, at the readout position, sets the resid_post u-projection to target_proj (additive shift
    along u): r += (target_proj - r.u) * u. `u` must be on the model device. The projection-edit ablation of
    cave_suppress_vs_install / headset_direction / cave_carrier_deconfound. This IS the SENDER: applying it at
    resid_post[L] is the u_cave removal whose downstream propagation this control path-patches."""
    def hook(r, hook, u=u, target_proj=target_proj):
        cur = float(r[0, -1].float() @ u)
        r[0, -1] = r[0, -1] + ((target_proj - cur) * u).to(r.dtype)
        return r
    return hook


def _logits(model, ids, hooks=None):
    """Full last-position logits (optionally under fwd_hooks). Forward-only."""
    with torch.no_grad():
        return model.run_with_hooks(ids, fwd_hooks=hooks) if hooks else model(ids)


# --------------------------------------------------------------------------- real-run path-patch hooks
def _freeze_z_hook(clean_z_last, exempt_heads=None):
    """Hook on hook_z[L']: overwrite the LAST-token per-head z with its CLEAN (cave-present) value for every
    head EXCEPT the `exempt_heads` set, which is left as the network recomputed it (so an exempt receiver head
    sees the delta-modified residual through its own q/k/v read while every other head writes clean).
    clean_z_last is [n_head, d_head] on CPU/float; exempt_heads is a set of head indices (None/empty -> freeze
    ALL heads, the DIRECT-path case). Forward-only, in-place on the last position only."""
    exempt = exempt_heads or set()

    def hook(z, hook, clean=clean_z_last, exempt=exempt):
        nH = z.shape[2]
        cz = clean.to(z.dtype).to(z.device)
        for H in range(nH):
            if H in exempt:
                continue
            z[0, -1, H, :] = cz[H]
        return z
    return hook


def _freeze_mlp_hook(clean_mlp_last, exempt=False):
    """Hook on hook_mlp_out[L']: overwrite the LAST-token MLP output with its CLEAN value unless `exempt`
    (the MLP is the receiver being studied -> leave it as recomputed from the delta-modified residual).
    clean_mlp_last is [d_model]. Forward-only, in-place on the last position only."""
    def hook(m, hook, clean=clean_mlp_last, exempt=exempt):
        if not exempt:
            m[0, -1, :] = clean.to(m.dtype).to(m.device)
        return m
    return hook


def _component_keys(L, nH, nL):
    """Ordered receiver-key dicts for every DOWNSTREAM component (layers L'>L): each attention head L'.H and
    each MLP L'. Each dict carries a stable string key, a type tag, and its layer/head. The ordering is fixed
    so the per-receiver effects and the reports stay aligned. (cave_direction_dla._comp_keys, restricted to
    the downstream layers and including the readout-layer-exclusive set L+1..nL-1.)"""
    keys = []
    for ell in range(L + 1, nL):
        for H in range(nH):
            keys.append({"key": f"L{ell}H{H}", "type": "attn", "layer": ell, "head": H})
        keys.append({"key": f"mlp{ell}", "type": "mlp", "layer": ell, "head": None})
    return keys


# --------------------------------------------------------------------------- real-run collect / fit
def _collect(model, pool, device, is_chat, fit_layers):
    """One model: per pool item, under NEUTRAL and COUNTER (W* asserted), in ONE forward each, cache the
    last-token resid_post at every fit layer AND read the full next-token softmax (P1=neutral, P2=counter)
    + first-token M. First-token-collision items (cid==aid) skipped. Forward-only. (Structure follows
    cave_suppress_vs_install._collect / cave_carrier_deconfound._collect exactly.)"""
    from rlhf_differential import _helpers
    from job_truthful_flip import PUSH, NEUTRAL
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    tag = "it" if is_chat else "base"
    recs = []
    for i, it in enumerate(pool):
        q, C, W = it["q"], it["correct"], it["Wstar"]
        cid, aid = first(" " + C), first(" " + W)
        if cid == aid:                                  # first-token collision -> readout degenerate, skip
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
            lg_n = model.run_with_hooks(neutral, fwd_hooks=[(n, grab_n) for n in names])
            lg_c = model.run_with_hooks(counter, fwd_hooks=[(n, grab_c) for n in names])
        Pn, Pc = _full_softmax(lg_n), _full_softmax(lg_c)
        neu = _readout(Pn, cid, aid)
        ctr = _readout(Pc, cid, aid)
        M_neu = _logp_diff_local(lg_n, cid, aid)
        M_ctr = _logp_diff_local(lg_c, cid, aid)
        recs.append({"i": i, "q": q, "cid": cid, "aid": aid, "neutral": neutral, "counter": counter,
                     "rn": rn, "rc": rc, "M_neu": M_neu, "M_ctr": M_ctr, "neu": neu, "ctr": ctr,
                     "P1": Pn.detach().cpu(), "P2": Pc.detach().cpu()})
        print(f"  [{tag}] item {i} M_neu={M_neu:+.2f} M_ctr={M_ctr:+.2f} "
              f"amx_neu={neu['argmax']} amx_ctr={ctr['argmax']} P(W*)ctr={ctr['p_w']:.2e} "
              f"q={q[:36]!r}", flush=True)
    return recs


def _u_cave(recs, idxs, L, device):
    """Diff-of-means cave direction over the items in `idxs` at layer L, plus the train-neutral-mean
    projection. u = normalize(mean(rc - rn)); proj_n = mean(rn . u). Forward-free (operates on the cached
    residuals). Mirrors cave_suppress_vs_install._u_cave / headset_direction."""
    Rc = torch.stack([recs[k]["rc"][L] for k in idxs]).to(device)
    Rn = torch.stack([recs[k]["rn"][L] for k in idxs]).to(device)
    u = unit(diff_of_means(Rc, Rn))
    proj_n = statistics.mean(float(recs[k]["rn"][L].to(device) @ u) for k in idxs)
    return u, proj_n


def _headline_layer(model, recs, fit_idx, fit_layers, device):
    """Headline layer = the fit layer with the largest IN-SAMPLE u_cave cave-necessity over the fit-set (the
    same headline-selection rule as cave_suppress_vs_install._headline_layer / cave_carrier_deconfound). For
    each layer, ablate the u_cave-projection on each fit COUNTER residual to the fit-set neutral mean and read
    the margin recovery frac=(M_ablate-M_ctr)/(M_neu-M_ctr); pick the max-mean layer. Forward-only."""
    def _nec(L):
        u, proj_n = _u_cave(recs, fit_idx, L, device)
        fr = []
        for k in fit_idx:
            r = recs[k]
            gap = r["M_neu"] - r["M_ctr"]
            if abs(gap) < 1e-6:
                continue
            h = [(_rname(L), _proj_edit_hook(u, proj_n))]
            M_ab = _logp_diff_local(_logits(model, r["counter"], hooks=h), r["cid"], r["aid"])
            fr.append((M_ab - r["M_ctr"]) / gap)
        return statistics.mean(fr) if fr else None
    per_layer = {L: _nec(L) for L in fit_layers}
    valid = [L for L in fit_layers if per_layer[L] is not None]
    headline = max(valid, key=lambda L: per_layer[L]) if valid else None
    return headline, {L: (round(v, 4) if v is not None else None) for L, v in per_layer.items()}


def _clean_downstream_cache(model, ids, L, nH, nL):
    """ONE clean (cave-present) counter forward: cache the LAST-token per-head z (hook_z[L'][-1] -> [n_head,
    d_head]) and per-layer MLP out (hook_mlp_out[L'][-1] -> [d_model]) for every downstream layer L'>L.
    Returns ({L': z_last}, {L': mlp_last}) on CPU/float. Forward-only, last-token-only (one position per
    layer; the per-head z slice + mlp vec is ~light at 9b, as in cave_direction_dla's memory note)."""
    zc, mc = {}, {}

    def grab_z(z, hook, _zc=zc):
        _zc[hook.layer()] = z[0, -1].detach().float().cpu(); return z       # [n_head, d_head]

    def grab_m(m, hook, _mc=mc):
        _mc[hook.layer()] = m[0, -1].detach().float().cpu(); return m       # [d_model]

    hooks = ([(_zname(ell), grab_z) for ell in range(L + 1, nL)]
             + [(_mname(ell), grab_m) for ell in range(L + 1, nL)])
    with torch.no_grad():
        model.run_with_hooks(ids, fwd_hooks=hooks, return_type=None)
    return zc, mc


def _freeze_hooks(zc, mc, L, nL, exempt_layer=None, exempt_head=None, exempt_is_mlp=False):
    """Build the downstream-freeze hook list for layers L'>L. Every head's hook_z[L'][-1] and every
    hook_mlp_out[L'][-1] is overwritten with its CLEAN value, EXCEPT the single exempt receiver (if any):
      - exempt a head: exempt_layer=L', exempt_head=H, exempt_is_mlp=False -> that head slot is left as the
        network recomputed it (it reads the delta-modified residual through its own q/k/v).
      - exempt an MLP: exempt_layer=L', exempt_is_mlp=True -> that hook_mlp_out is left as recomputed.
      - exempt nothing (DIRECT-path / joint base): freeze everything.
    Returns a list of (hook_name, hook_fn). Forward-only."""
    hooks = []
    for ell in range(L + 1, nL):
        ex_heads = ({exempt_head} if (exempt_layer == ell and exempt_head is not None
                                      and not exempt_is_mlp) else set())
        hooks.append((_zname(ell), _freeze_z_hook(zc[ell], exempt_heads=ex_heads)))
        mlp_exempt = (exempt_layer == ell and exempt_is_mlp)
        hooks.append((_mname(ell), _freeze_mlp_hook(mc[ell], exempt=mlp_exempt)))
    return hooks


def _freeze_hooks_multi(zc, mc, L, nL, exempt_keys):
    """Like _freeze_hooks but EXEMPTS a SET of receivers simultaneously (the joint-top-k path patch): every
    receiver in `exempt_keys` recomputes from the delta-modified residual; every other downstream component
    is frozen to clean. exempt_keys is a list of receiver-key dicts (type/layer/head). Forward-only."""
    exempt_heads_by_layer = {}
    exempt_mlp_layers = set()
    for k in exempt_keys:
        if k["type"] == "attn":
            exempt_heads_by_layer.setdefault(k["layer"], set()).add(k["head"])
        else:
            exempt_mlp_layers.add(k["layer"])
    hooks = []
    for ell in range(L + 1, nL):
        hooks.append((_zname(ell), _freeze_z_hook(zc[ell], exempt_heads=exempt_heads_by_layer.get(ell, set()))))
        hooks.append((_mname(ell), _freeze_mlp_hook(mc[ell], exempt=(ell in exempt_mlp_layers))))
    return hooks


# --------------------------------------------------------------------------- one model end-to-end
def _measure_one_direction(model, recs, test, L, u, proj_n, device, nH, nL, comp_keys, tag, dir_name):
    """Run the FULL / DIRECT / per-receiver / joint-top-k path patches for ONE sender direction `u` (either
    u_cave or the matched random direction) and aggregate the restoration numbers. The SENDER delta is the
    projection-edit ablation of u to proj_n at resid_post[L][-1]. Forward-only, last-token. Returns a dict
    with restoration_full, restoration_direct, per-receiver effect(R) (frac argmax==neutral-argmax) and the
    logit-restoration fraction, and the joint-top-k restoration."""
    sender = [(_rname(L), _proj_edit_hook(u, proj_n))]
    full_flags, direct_flags = [], []
    # per-receiver: accumulate restoration flags + logit-restoration fractions, one list per receiver key idx
    comp_flags = [[] for _ in comp_keys]
    comp_logit = [[] for _ in comp_keys]
    # for the joint-top-k path patch we first need a per-receiver ranking; we compute per-item then aggregate.
    # joint is evaluated AFTER ranking, in a second per-item pass over the SAME test items (cached resid is
    # cheap; the model forwards dominate). Collect per-item context for the second pass.
    per_item_ctx = []
    for k in test:
        r = recs[k]
        cid, aid = r["cid"], r["aid"]
        neu_argmax = r["neu"]["argmax"]
        # CLEAN downstream cache for this item (the cave-present counter component outputs).
        zc, mc = _clean_downstream_cache(model, r["counter"], L, nH, nL)
        # logit-restoration reference: full-ablation last-position logits and clean (counter) logits.
        lg_clean = _logits(model, r["counter"])
        lg_full = _logits(model, r["counter"], hooks=sender)
        amx_full = int(_full_softmax(lg_full).argmax())
        full_flags.append(amx_full == neu_argmax)
        # DIRECT path: sender + freeze ALL downstream components to clean.
        h_direct = sender + _freeze_hooks(zc, mc, L, nL)
        lg_direct = _logits(model, r["counter"], hooks=h_direct)
        amx_direct = int(_full_softmax(lg_direct).argmax())
        direct_flags.append(amx_direct == neu_argmax)
        # per-item logit-restoration baseline: the answer-logit move from clean -> full, used to normalize the
        # per-receiver logit move. dM_full = M(full) - M(clean) at the answer slot (W*-vs-C margin moves up
        # toward neutral under restoration). guard tiny moves.
        M_clean = _logp_diff_local(lg_clean, cid, aid)
        M_full = _logp_diff_local(lg_full, cid, aid)
        dM_full = M_full - M_clean
        # PER-RECEIVER path patch: only receiver R recomputes; all other downstream components frozen to clean.
        for ridx, ck in enumerate(comp_keys):
            h_r = sender + _freeze_hooks(zc, mc, L, nL,
                                         exempt_layer=ck["layer"],
                                         exempt_head=ck["head"],
                                         exempt_is_mlp=(ck["type"] == "mlp"))
            lg_r = _logits(model, r["counter"], hooks=h_r)
            amx_r = int(_full_softmax(lg_r).argmax())
            comp_flags[ridx].append(amx_r == neu_argmax)
            if abs(dM_full) > 1e-6:
                comp_logit[ridx].append((_logp_diff_local(lg_r, cid, aid) - M_clean) / dM_full)
        per_item_ctx.append({"k": k, "zc": zc, "mc": mc, "neu_argmax": neu_argmax, "cid": cid, "aid": aid})
        print(f"    [{tag} {dir_name} L{L}] item {r['i']} amx_full={amx_full} (neu={neu_argmax}) "
              f"amx_direct={amx_direct} dM_full={dM_full:+.3f}", flush=True)

    restoration_full = restoration_frac(full_flags)
    restoration_direct = restoration_frac(direct_flags)
    comp_effects = [restoration_frac(f) for f in comp_flags]
    comp_logit_frac = [(statistics.mean(x) if x else 0.0) for x in comp_logit]

    # rank receivers by effect(R) and evaluate the JOINT top-TOPK path patch (all top-TOPK recompute together)
    order = sorted(range(len(comp_keys)), key=lambda i: comp_effects[i], reverse=True)
    top_idx = order[:min(TOPK, len(order))]
    top_keys = [comp_keys[i] for i in top_idx]
    joint_flags = []
    if top_keys:
        for ctx in per_item_ctx:
            r = recs[ctx["k"]]
            h_joint = sender + _freeze_hooks_multi(ctx["zc"], ctx["mc"], L, nL, top_keys)
            amx_j = int(_full_softmax(_logits(model, r["counter"], hooks=h_joint)).argmax())
            joint_flags.append(amx_j == ctx["neu_argmax"])
    joint_topk_restoration = restoration_frac(joint_flags) if joint_flags else 0.0

    return {
        "restoration_full": restoration_full,
        "restoration_direct": restoration_direct,
        "comp_effects": comp_effects,
        "comp_logit_frac": comp_logit_frac,
        "joint_topk_restoration": joint_topk_restoration,
        "n_test": len(test),
    }


def _measure_model(name, is_chat, device, pool, fit_layers):
    """One model end-to-end. Collect realized + M readouts under neutral/counter on the wide pool; restrict to
    the argmax-W* caving items; fit u_cave HELD-OUT (TRAIN fold) at the headline layer; on the TEST fold run
    the direct-path + per-receiver + joint-top-k path patches for u_cave and a matched random direction; build
    the reader decision. Forward-only."""
    from transformer_lens import HookedTransformer
    from rlhf_differential import MIN_EFFECT_NET
    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    nH, nL = model.cfg.n_heads, model.cfg.n_layers
    recs = _collect(model, pool, device, is_chat, fit_layers)
    n = len(recs)

    # CAVING items: counter lowers M from neutral by >= MIN_EFFECT_NET (same gate as cave_suppress_vs_install).
    cave_pos = [k for k, r in enumerate(recs) if (r["M_neu"] - r["M_ctr"]) >= MIN_EFFECT_NET]
    # restrict to items whose COUNTER argmax IS the W*-first-tok (the model would actually emit W*).
    argmaxW = [k for k in cave_pos if recs[k]["ctr"]["argmax"] == recs[k]["aid"]]

    out = {"name": name, "regime": "chat" if is_chat else "qa", "n_ok": n,
           "n_cave": len(cave_pos), "n_argmaxW_cave": len(argmaxW), "n_layers": nL, "n_heads": nH}

    if len(argmaxW) < MIN_FIT:
        print(f"  [{name}] only {len(argmaxW)} argmax-W* caving items < MIN_FIT({MIN_FIT}); "
              f"no held-out direction.", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        out["headline_layer"] = None
        out["numbers"] = None
        out["random_numbers"] = None
        out["decision"] = decide_reader(None, None, [], [], None, n_fit=len(argmaxW))
        out["random_decision"] = decide_reader(None, None, [], [], None, n_fit=len(argmaxW))
        return out

    # held-out fold over the argmax-W* caving items.
    tr_pos, te_pos = split_indices(len(argmaxW), SPLIT_SEED)
    train = [argmaxW[j] for j in tr_pos]
    test = [argmaxW[j] for j in te_pos]

    headline, per_layer_nec = _headline_layer(model, recs, train, fit_layers, device)
    out["in_sample_necessity_by_layer"] = per_layer_nec
    out["headline_layer"] = headline
    if headline is None:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        out["numbers"] = None
        out["random_numbers"] = None
        out["decision"] = decide_reader(None, None, [], [], None, n_fit=len(argmaxW))
        out["random_decision"] = decide_reader(None, None, [], [], None, n_fit=len(argmaxW))
        return out

    L = headline
    u_cave, proj_n = _u_cave(recs, train, L, device)
    # matched-random unit direction; target its OWN train-neutral-mean projection (matched magnitude shift).
    g = torch.Generator(device="cpu").manual_seed(RAND_SEED)
    u_rand = unit(torch.randn(u_cave.shape, generator=g).to(u_cave.dtype).to(device))
    proj_n_rand = statistics.mean(float(recs[k]["rn"][L].to(device) @ u_rand) for k in train)

    comp_keys = _component_keys(L, nH, nL)
    out["n_downstream_components"] = len(comp_keys)

    # u_cave path patches.
    res_cave = _measure_one_direction(model, recs, test, L, u_cave, proj_n, device, nH, nL,
                                      comp_keys, ("it" if is_chat else "base"), "u_cave")
    # matched-random floor (same direct + per-receiver + joint-top-k patches).
    res_rand = _measure_one_direction(model, recs, test, L, u_rand, proj_n_rand, device, nH, nL,
                                      comp_keys, ("it" if is_chat else "base"), "u_rand")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    decision = decide_reader(res_cave["restoration_full"], res_cave["restoration_direct"],
                             res_cave["comp_effects"], comp_keys, res_cave["joint_topk_restoration"],
                             n_fit=len(argmaxW))
    random_decision = decide_reader(res_rand["restoration_full"], res_rand["restoration_direct"],
                                    res_rand["comp_effects"], comp_keys, res_rand["joint_topk_restoration"],
                                    n_fit=len(argmaxW))

    # attach the ranked per-receiver table (effect + logit-restoration fraction) for u_cave to the numbers.
    order = sorted(range(len(comp_keys)), key=lambda i: res_cave["comp_effects"][i], reverse=True)
    ranked = [{"key": comp_keys[i]["key"], "type": comp_keys[i]["type"], "layer": comp_keys[i]["layer"],
               "head": comp_keys[i]["head"], "effect": round(res_cave["comp_effects"][i], 6),
               "logit_restore_frac": round(res_cave["comp_logit_frac"][i], 6)}
              for i in order[:max(TOPK * 3, 15)]]

    out["headline_layer"] = L
    out["n_train"] = len(train)
    out["n_test"] = len(test)
    out["numbers"] = {
        "restoration_full": round(res_cave["restoration_full"], 6),
        "restoration_direct": round(res_cave["restoration_direct"], 6),
        "joint_topk_restoration": round(res_cave["joint_topk_restoration"], 6),
        "downstream_total_effect": round(float(sum(res_cave["comp_effects"])), 6),
        "ranked_receivers": ranked,
    }
    out["random_numbers"] = {
        "restoration_full": round(res_rand["restoration_full"], 6),
        "restoration_direct": round(res_rand["restoration_direct"], 6),
        "joint_topk_restoration": round(res_rand["joint_topk_restoration"], 6),
        "downstream_total_effect": round(float(sum(res_rand["comp_effects"])), 6),
    }
    out["decision"] = decision
    out["random_decision"] = random_decision
    return out


def run(name_base, name_it, tag, device, chat_it, pool):
    # Real run: pin the diff-of-means layer sweep to the reference if importable; fall back to the module
    # constant (same value). --selftest never reaches here, so it stays import-free on CPU.
    try:
        from headset_direction import FIT_LAYERS as _FL
        fit_layers = list(_FL)
    except Exception:
        fit_layers = list(FIT_LAYERS)
    res = {"base": _measure_model(name_base, False, device, pool, fit_layers),
           "it": _measure_model(name_it, bool(chat_it), device, pool, fit_layers)}
    out = {
        "name_base": name_base, "name_it": name_it, "device": device, "tag": tag,
        "cue": "cave_reader_pathpatch", "pool_size": len(pool), "fit_layers": fit_layers,
        "metric": ("on argmax-W* caving items, the realized next-token argmax-restoration (frac argmax=="
                   "neutral-argmax) carried by the u_cave-removal delta at the headline layer L through "
                   "(2) the FULL forward, (3) the DIRECT residual->unembed path with all downstream "
                   "components frozen to clean, (4) each downstream receiver alone (per-head / per-MLP path "
                   "patch), and the joint top-TOPK receivers; plus the matched-random floor"),
        "thresholds": {"DIRECT_THR": DIRECT_THR, "CONC_FRAC": CONC_FRAC, "TOPK": TOPK,
                       "RESTORE_FLOOR": RESTORE_FLOOR, "SPLIT_SEED": SPLIT_SEED, "RAND_SEED": RAND_SEED,
                       "MIN_FIT": MIN_FIT},
        "decision_rule": DECISION_RULE,
        "base": res["base"], "it": res["it"],
    }
    Path("out").mkdir(exist_ok=True)
    Path(f"out/cave_reader_pathpatch_{tag}.json").write_text(json.dumps(out, indent=2, default=str))
    for m in MODELS:
        r = res[m]
        dd = r["decision"]
        rd = r["random_decision"]
        print(f"[{m}] {dd['category']} (L{r.get('headline_layer')}) n_argmaxW={r['n_argmaxW_cave']} "
              f"direct_frac={dd.get('direct_fraction')} (full={dd.get('restoration_full')} "
              f"direct={dd.get('restoration_direct')}) conc@{TOPK}={dd.get('conc_frac_at_topk')} "
              f"joint={dd.get('joint_topk_restoration')} band={dd.get('receiver_layer_band')} "
              f"split={dd.get('attn_vs_mlp_split_of_indirect')} "
              f"top={[t['key'] for t in dd.get('top_receivers', [])]} "
              f"| RANDOM={rd['category']} (direct_frac={rd.get('direct_fraction')}, floor)", flush=True)
    print(f"[done] wrote out/cave_reader_pathpatch_{tag}.json", flush=True)


# --------------------------------------------------------------------------- selftest (model-free, CPU, NO model load)
def _planted_pathpatch(n_comp, regime, seed, n_items=8):
    """Build a synthetic per-receiver path-patch outcome for ONE direction under a planted regime, then run
    the SAME aggregation + decision pipeline as the real path (minus the model forwards). The model forward is
    replaced by a planted restoration rule. Returns (restoration_full, restoration_direct, comp_effects,
    comp_keys, joint_topk_restoration). Pure (selftest only).

    Each receiver gets a planted per-item 'restores the neutral argmax' bit. The construction encodes the
    three target regimes:
      regime='direct'      : the delta restores the neutral argmax DIRECTLY (restoration_direct == full) and no
                             single downstream receiver alone restores (comp_effects ~0). -> DIRECT_WRITE.
      regime='localized'   : the direct path restores nothing; a SMALL writer subset (the first `nw`
                             receivers) each individually restore on EVERY item AND jointly reconstruct the
                             full indirect restoration; all other receivers restore on no item. The random
                             direction (built separately by the caller) restores nothing. -> LOCALIZED_READERS
                             (top-TOPK carry >= CONC_FRAC and joint reconstructs).
      regime='distributed' : the direct path restores nothing; the indirect restoration is spread -- MANY
                             receivers each restore on a DIFFERENT small share of items so no top-TOPK subset
                             carries >= CONC_FRAC. -> DISTRIBUTED_READERS.
      regime='norestore'   : the FULL ablation itself restores nothing (restoration_full < RESTORE_FLOOR).
                             -> NO_RESTORATION."""
    import random as _r
    rng = _r.Random(seed)
    # downstream receiver keys: alternate attn/mlp across a few synthetic layers (layers 1..) so the
    # attn-vs-MLP split and layer band are exercised. (Layer 0 reserved as 'sender' analogue.)
    comp_keys = []
    for c in range(n_comp):
        ell = 1 + c // 3
        if c % 3 == 2:
            comp_keys.append({"key": f"mlp{ell}", "type": "mlp", "layer": ell, "head": None})
        else:
            comp_keys.append({"key": f"L{ell}H{c % 3}", "type": "attn", "layer": ell, "head": c % 3})

    full_flags = [True] * n_items
    if regime == "norestore":
        full_flags = [False] * n_items
        comp_flags = [[False] * n_items for _ in comp_keys]
        direct_flags = [False] * n_items
        joint_flags = [False] * n_items
    elif regime == "direct":
        direct_flags = [True] * n_items                         # direct path restores fully
        comp_flags = [[False] * n_items for _ in comp_keys]     # no single downstream receiver restores
        joint_flags = [True] * n_items                          # (top-k recompute over a frozen-clean residual
        #                                                          that already restores directly -> still True;
        #                                                          irrelevant since DIRECT_WRITE wins first.)
    elif regime == "localized":
        direct_flags = [False] * n_items                        # direct path restores nothing
        nw = 3                                                  # the planted writer subset
        comp_flags = []
        for ci in range(len(comp_keys)):
            comp_flags.append([ci < nw] * n_items)             # each of the first nw restores on every item
        joint_flags = [True] * n_items                          # the writers jointly reconstruct full
    else:  # distributed
        direct_flags = [False] * n_items
        # every receiver restores on a DIFFERENT single item (spread); each effect = 1/n_items, no concentration
        comp_flags = []
        for ci in range(len(comp_keys)):
            flags = [False] * n_items
            flags[ci % n_items] = True
            comp_flags.append(flags)
        # joint top-k reconstructs only its own few items (k items out of n) -> < CONC_FRAC*indirect
        joint_flags = [False] * n_items
        for ci in range(min(TOPK, len(comp_keys))):
            joint_flags[ci % n_items] = True

    restoration_full = restoration_frac(full_flags)
    restoration_direct = restoration_frac(direct_flags)
    comp_effects = [restoration_frac(f) for f in comp_flags]
    joint_topk_restoration = restoration_frac(joint_flags)
    return restoration_full, restoration_direct, comp_effects, comp_keys, joint_topk_restoration


def selftest():
    torch.manual_seed(0)

    # ---------- (0) direct/indirect split arithmetic ----------
    s = direct_indirect_split(1.0, 0.8)
    assert s["direct_fraction"] == 0.8 and s["indirect_restoration"] == round(0.2, 6), s
    s2 = direct_indirect_split(0.6, 0.0)
    assert s2["direct_fraction"] == 0.0 and s2["indirect_restoration"] == 0.6, s2
    s3 = direct_indirect_split(0.5, 0.7)        # direct over-restores: fraction>1, indirect clamps to 0
    assert s3["direct_fraction"] == round(0.7 / 0.5, 6) and s3["indirect_restoration"] == 0.0, s3
    assert direct_indirect_split(0.0, 0.0)["direct_fraction"] == 0.0
    assert direct_indirect_split(None, 0.5)["direct_fraction"] is None
    print(f"[selftest] (0) direct/indirect split: 0.8/1.0 -> dfrac={s['direct_fraction']} "
          f"indirect={s['indirect_restoration']}; over-restore clamps indirect to 0")

    # ---------- (0) concentration + split + band primitives ----------
    cf1, tot1, ts1, _ = concentration([1.0] + [0.0] * 9, TOPK)
    assert abs(cf1 - 1.0) < 1e-9 and abs(tot1 - 1.0) < 1e-9, (cf1, tot1)
    n_unif = 40
    cfu, totu, _, _ = concentration([1.0] * n_unif, TOPK)
    assert abs(cfu - TOPK / n_unif) < 1e-9, (cfu, TOPK / n_unif)     # uniform -> exactly topk/n
    cf0, tot0, _, _ = concentration([0.0] * 5, TOPK)
    assert cf0 == 0.0 and tot0 == 0.0
    vals = [abs(x) for x in torch.randn(50).tolist()]
    fracs = [concentration(vals, k)[0] for k in (1, 5, 10, 25, 50)]
    assert all(fracs[i] <= fracs[i + 1] + 1e-9 for i in range(len(fracs) - 1)), fracs
    sp = split_attn_mlp([{"type": "attn"}, {"type": "mlp"}, {"type": "attn"}], [0.6, 0.2, 0.2])
    assert abs(sp["attn_effect"] - 0.8) < 1e-9 and abs(sp["mlp_effect"] - 0.2) < 1e-9, sp
    assert abs(sp["attn_frac"] - 0.8) < 1e-9, sp
    assert layer_band([{"layer": 38}, {"layer": 36}, {"layer": 41}]) == (36, 41)
    assert layer_band([]) == (None, None)
    print(f"[selftest] (0) concentration single={cf1:.3f} uniform={cfu:.3f} monotone; split + band OK")

    # ---------- (0) decision boundaries on hand-built numbers ----------
    # DIRECT_WRITE: direct fraction at/over threshold.
    ck5 = [{"key": f"c{i}", "type": ("attn" if i % 2 == 0 else "mlp"), "layer": 36 + i // 4,
            "head": (i if i % 2 == 0 else None)} for i in range(20)]
    dw = decide_reader(1.0, 0.6, [0.1] * 20, ck5, 0.0, n_fit=8)
    assert dw["category"] == "DIRECT_WRITE" and dw["direct_write"], dw
    dw_edge = decide_reader(1.0, DIRECT_THR, [0.0] * 20, ck5, 0.0, n_fit=8)   # exactly at threshold -> DIRECT
    assert dw_edge["category"] == "DIRECT_WRITE", dw_edge
    # LOCALIZED_READERS: direct below threshold, top-TOPK concentrate AND joint reconstructs the indirect.
    eff_loc = [0.9, 0.9, 0.9, 0.9, 0.9] + [0.0] * 15        # top-5 carry everything
    loc = decide_reader(1.0, 0.1, eff_loc, ck5, 0.85, n_fit=8)               # indirect=0.9, joint 0.85>=0.45
    assert loc["category"] == "LOCALIZED_READERS" and loc["localized_readers"], loc
    assert loc["concentrated"] and loc["joint_reconstructs"], loc
    # LOCALIZED fails if joint does NOT reconstruct (redundant receivers individually restore but don't sum)
    loc_nojoint = decide_reader(1.0, 0.1, eff_loc, ck5, 0.2, n_fit=8)        # joint 0.2 < 0.45*indirect(0.9)
    assert loc_nojoint["category"] == "DISTRIBUTED_READERS" and not loc_nojoint["joint_reconstructs"], loc_nojoint
    # DISTRIBUTED_READERS: direct below threshold, effect spread (top-TOPK < CONC_FRAC).
    eff_dist = [0.1] * 20                                    # uniform: top-5 carry 5/20=0.25 < 0.5
    dist = decide_reader(1.0, 0.1, eff_dist, ck5, 0.1, n_fit=8)
    assert dist["category"] == "DISTRIBUTED_READERS" and not dist["concentrated"], dist
    # NO_RESTORATION: full ablation does not restore.
    nr = decide_reader(0.0, 0.0, [0.0] * 20, ck5, 0.0, n_fit=8)
    assert nr["category"] == "NO_RESTORATION", nr
    # INSUFFICIENT below MIN_FIT.
    ins = decide_reader(None, None, [], [], None, n_fit=1)
    assert ins["category"] == "INSUFFICIENT", ins
    print("[selftest] (0) decide_reader: DIRECT_WRITE / edge / LOCALIZED / LOCALIZED-fails-no-joint / "
          "DISTRIBUTED / NO_RESTORATION / INSUFFICIENT all fire")

    # ---------- (0) freeze-hook exemption math (model-free; verify the head/MLP exemption set logic) ----------
    # _freeze_z_hook: exempting head 2 must overwrite heads {0,1,3} and leave head 2 as recomputed.
    nH, dH = 4, 6
    recomputed = torch.arange(nH * dH, dtype=torch.float32).reshape(1, 1, nH, dH).clone()
    clean = -torch.ones(nH, dH)
    z = recomputed.clone()
    _freeze_z_hook(clean, exempt_heads={2})(z, hook=None)
    assert torch.allclose(z[0, -1, 2], recomputed[0, -1, 2]), "exempt head 2 must be left as recomputed"
    for H in (0, 1, 3):
        assert torch.allclose(z[0, -1, H], clean[H]), f"head {H} must be frozen to clean"
    # freeze ALL heads (DIRECT path: exempt_heads empty) -> every head == clean
    z2 = recomputed.clone()
    _freeze_z_hook(clean, exempt_heads=set())(z2, hook=None)
    assert torch.allclose(z2[0, -1], clean), "DIRECT path: all heads frozen to clean"
    # _freeze_mlp_hook: exempt=False overwrites; exempt=True leaves recomputed.
    m_re = torch.arange(dH, dtype=torch.float32).reshape(1, 1, dH).clone()
    m_cl = torch.full((dH,), 9.0)
    mm = m_re.clone(); _freeze_mlp_hook(m_cl, exempt=False)(mm, hook=None)
    assert torch.allclose(mm[0, -1], m_cl), "mlp frozen to clean when not exempt"
    mm2 = m_re.clone(); _freeze_mlp_hook(m_cl, exempt=True)(mm2, hook=None)
    assert torch.allclose(mm2[0, -1], m_re[0, -1]), "mlp left as recomputed when exempt"
    # _component_keys: only downstream layers L+1..nL-1; correct count = (nL-1-L)*(nH+1).
    L, nLc = 36, 42
    keys = _component_keys(L, nH=16, nL=nLc)
    assert all(k["layer"] > L for k in keys), "all receiver keys must be strictly downstream of L"
    assert len(keys) == (nLc - 1 - L) * (16 + 1), len(keys)        # heads + one MLP per layer
    assert keys[0]["layer"] == L + 1 and keys[-1]["layer"] == nLc - 1
    print(f"[selftest] (0) freeze-hook head/MLP exemption + downstream component keys "
          f"({len(keys)} receivers over L{L+1}..L{nLc-1}) OK")

    # ============================================================ END-TO-END synthetic regimes =============
    # Each regime drives the SAME aggregation + decision the real path uses (restoration_frac, ranking,
    # concentration, joint reconstruction, decide_reader). The random direction's planted outcome is a clean
    # floor (restores nothing) in every regime, so it never reproduces the effect.

    def run_regime(regime, seed):
        rf, rd, eff, ck, joint = _planted_pathpatch(20, regime, seed)
        dec = decide_reader(rf, rd, eff, ck, joint, n_fit=8)
        return dec, (rf, rd, eff, joint)

    # random floor: full ablation restores nothing -> NO_RESTORATION for the random direction in every regime.
    def random_floor():
        rf, rd, eff, ck, joint = _planted_pathpatch(20, "norestore", seed=999)
        return decide_reader(rf, rd, eff, ck, joint, n_fit=8)

    rand_dec = random_floor()
    assert rand_dec["category"] == "NO_RESTORATION", rand_dec

    # (i) DIRECT_WRITE: the restoration flows entirely through the direct residual->unembed path.
    di, (rf_i, rd_i, _, _) = run_regime("direct", 1)
    assert di["category"] == "DIRECT_WRITE", di
    assert di["direct_fraction"] >= DIRECT_THR and rf_i == 1.0 and rd_i == 1.0, di
    assert rand_dec["category"] != "DIRECT_WRITE", rand_dec      # random does not reproduce it
    print(f"[selftest] (i) DIRECT_WRITE: direct_frac={di['direct_fraction']} (full={rf_i} direct={rd_i}) | "
          f"RANDOM floor={rand_dec['category']}")

    # (ii) LOCALIZED_READERS: a small downstream subset carries it AND a random direction does nothing.
    dii, _ = run_regime("localized", 2)
    assert dii["category"] == "LOCALIZED_READERS", dii
    assert dii["conc_frac_at_topk"] >= CONC_FRAC and dii["joint_reconstructs"], dii
    # the top receivers must be the planted writer subset (the first nw keys, all distinct components)
    top_keys = [t["key"] for t in dii["top_receivers"]]
    assert len(top_keys) == TOPK and dii["downstream_total_effect"] > 0, dii
    assert rand_dec["category"] != "LOCALIZED_READERS", rand_dec
    print(f"[selftest] (ii) LOCALIZED_READERS: conc@{TOPK}={dii['conc_frac_at_topk']} "
          f"joint={dii['joint_topk_restoration']} top={top_keys} split={dii['attn_vs_mlp_split_of_indirect']} "
          f"| RANDOM floor={rand_dec['category']}")

    # (iii) DISTRIBUTED_READERS: the indirect restoration is spread evenly over many components.
    diii, _ = run_regime("distributed", 3)
    assert diii["category"] == "DISTRIBUTED_READERS", diii
    assert diii["conc_frac_at_topk"] < CONC_FRAC and not diii["direct_write"], diii
    print(f"[selftest] (iii) DISTRIBUTED_READERS: conc@{TOPK}={diii['conc_frac_at_topk']} "
          f"(< {CONC_FRAC}) joint={diii['joint_topk_restoration']} direct_frac={diii['direct_fraction']}")

    # (iv) NO_RESTORATION: the FULL ablation itself does not restore the neutral argmax.
    div, _ = run_regime("norestore", 4)
    assert div["category"] == "NO_RESTORATION", div
    print(f"[selftest] (iv) NO_RESTORATION: restoration_full={div['restoration_full']} "
          f"(< RESTORE_FLOOR={RESTORE_FLOOR})")

    # KL helper sanity (used in the real readouts; here just the pure math).
    V = 64
    p = torch.zeros(V); p[3] = 0.7; p[1:] += 0.3 / (V - 1)
    assert abs(kl_div(p, p)) < 1e-6
    q = torch.zeros(V); q[7] = 0.7; q[1:] += 0.3 / (V - 1)
    assert kl_div(p, q) > 0
    print("[selftest] kl_div identical=0, distinct>0 OK")

    print("[selftest] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--name-base", default="google/gemma-2-9b")
    p.add_argument("--name-it", default="google/gemma-2-9b-it")
    p.add_argument("--tag", default="9b")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--chat", action="store_true",
                   help="use chat template for the -it model (qa template otherwise)")
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        from misconception_pool import ITEMS_WIDE
        run(args.name_base, args.name_it, args.tag, args.device, args.chat, ITEMS_WIDE)


if __name__ == "__main__":
    main()
