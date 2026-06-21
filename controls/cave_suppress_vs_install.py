"""SUPPRESS vs INSTALL: when ablating the diff-of-means cave-DIRECTION u_cave deflates the asserted wrong
answer W*, WHERE does the realized (argmax) answer go? (sibling of faithful_caving.py / cave_direction_overlay.py
/ cave_direction_heldout.py / headset_direction.py.)

CONTEXT (neutral). A prior control (cave_direction_overlay.py + faithful_caving.py) showed that at base
(Q/A), ablating the diff-of-means cave-direction
  u_cave(L) = mean_items( resid_post[L][-1](counter) - resid_post[L][-1](neutral) )
on items the model caves on DEFLATES the realized first-token probability of the asserted wrong answer W*
(P(W*) drops) but does NOT raise the correct competitor C (dP(C) < 0). That leaves WHAT the suppression
does to the EMITTED answer unresolved: when the W* mass comes down, does the argmax become C (a literal
install of the correct answer), return to the model's UNPUSHED (NEUTRAL-condition) answer, or go to some
THIRD token (mass diffuses)? This control measures exactly that, with NO new mechanism.

SUBSTRATE. base (qa) and -it (chat), default gemma-2-9b / gemma-2-9b-it, on the wide misconception pool
(misconception_pool.ITEMS_WIDE). Restrict to CAVING items where the COUNTER argmax IS the W*-aligned
first-token (argmax == W*-first-tok), i.e. items where the model would ACTUALLY emit W* under pushback --
the only items where "what does suppression do to the emitted answer" is a well-posed question.

MEASUREMENT. Fit u_cave (diff-of-means counter-neutral, HELD-OUT: fit on a TRAIN fold of the argmax-W*
caving items, evaluate on the disjoint TEST fold -- the construction of cave_direction_heldout /
cave_direction_overlay / faithful_caving). On held-out argmax-W* items, read the realized next-token
distribution at the answer slot under FOUR conditions:
  (1) NEUTRAL prompt;   (2) COUNTER prompt;
  (3) COUNTER + ablate the u_cave-projection to the TRAIN-neutral mean;
  (4) COUNTER + ablate a matched-magnitude RANDOM direction (the specificity floor).
Over the held-out argmax-W* items, report:
  - new_argmax identity under (3): fraction whose ablated argmax ==
        C-first-tok (literal install-C), == W*-first-tok (no change), == the NEUTRAL-condition argmax of
        that item (restore-to-unpushed), == none-of-these (third token; only counted when it is not also
        C / W* / neutral-argmax so the four buckets partition the items).
  - dP(C)  = P3(C)  - P2(C),   dP(W*) = P3(W*) - P2(W*)   (realized first-token mass changes).
  - distribution-distance: KL(P3 || P1) vs KL(P2 || P1) -- does the ablation move the realized distribution
        BACK toward the NEUTRAL distribution (KL3 < KL2) or not.
  - the SAME four quantities for the RANDOM-direction condition (4) as the specificity floor.

This is claim-blind: it measures where the emitted answer goes when u_cave is ablated, on the items where
the model would emit W*. It attaches no hypothesis to any bucket, sign, or the base-vs-it comparison.

NEUTRAL DECISION (module constant THR=0.5; numbers + categories only, no hypothesis). Per model, classify
the DOMINANT outcome of ablating u_cave on the argmax-W* items:
  - INSTALLS_C       iff frac(argmax==C) >= THR.
  - RESTORES_NEUTRAL iff frac(argmax==neutral-argmax) >= THR AND KL3 < KL2 (moves back toward the unpushed
                         distribution).
  - SUPPRESSES_ONLY  iff W* mass drops (dP(W*) < 0) but neither INSTALLS_C nor RESTORES_NEUTRAL (the argmax
                         goes to a third token; mass diffuses).
  - NO_EFFECT        iff dP(W*) ~ 0 (|dP(W*)| < EFFECT_EPS).
  Resolution order when more than one fires: NO_EFFECT (nothing moved) is checked first; then INSTALLS_C,
  then RESTORES_NEUTRAL, then SUPPRESSES_ONLY. The matched RANDOM direction must NOT reproduce whatever
  u_cave does (its fractions / KL are reported as the floor). Reported for base and -it.

Forward-only (diff-of-means + projection edits + full-softmax readouts; no backward) -> fits the 40GB A100.
Reuses verified primitives: PUSH/NEUTRAL from job_truthful_flip; _helpers (qa/chat prompt builders,
qa-vs-chat handling) from rlhf_differential; ITEMS_WIDE from misconception_pool; the held-out fold split
(split_indices), diff-of-means fit (unit/diff_of_means), projection-edit ablation (_proj_edit_hook),
full-softmax readout (_full_softmax) and L1 helper from cave_direction_overlay; first-token margin
(_logp_diff_local), realized readout (_readout) and the per-item collect from faithful_caving; FIT_LAYERS
from headset_direction (deferred at real-run time). KL(P||Q) is re-implemented locally as a small pure
helper so --selftest is standalone on CPU with nothing else on sys.path -- the same FLAT-scp convention
cave_direction_overlay / faithful_caving use.

  python controls/cave_suppress_vs_install.py --selftest
  python controls/cave_suppress_vs_install.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it \
    --tag 9b --device cuda --chat
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

# Pre-registered thresholds (see module docstring). Neutral: stated on the measured numbers only.
THR = 0.5            # dominant-outcome fraction threshold for INSTALLS_C / RESTORES_NEUTRAL
EFFECT_EPS = 0.01    # |dP(W*)| below this -> NO_EFFECT (the ablation did not move the realized W* mass)
SPLIT_SEED = 0       # deterministic train/test fold (same convention as cave_direction_overlay)
RAND_SEED = 0        # deterministic matched-random-direction control
MIN_FIT = 3          # below this many argmax-W* caving items, no held-out direction can be fit/tested

# Diff-of-means cave-direction layer sweep. SAME value as headset_direction.FIT_LAYERS /
# cave_direction_overlay.FIT_LAYERS / faithful_caving.FIT_LAYERS ([24,28,32,36], the set's output range
# L21-L34); defined here as a module constant so --selftest needs nothing on sys.path. The real run also
# defers a `from headset_direction import FIT_LAYERS` so the sweep stays pinned to the reference.
FIT_LAYERS = [24, 28, 32, 36]

MODELS = ("base", "it")

DECISION_RULE = (
    "On the wide misconception pool, build NEUTRAL and COUNTER (W* asserted) prompts (job_truthful_flip "
    "turns; qa template for base, chat template for -it). Restrict to CAVING items (counter lowers the "
    "first-token margin M=logp(C)-logp(W*) from neutral by >= MIN_EFFECT_NET) whose COUNTER argmax IS the "
    "W*-first-token (the model would actually emit W*). Fit u_cave = mean(resid_post[L][-1](counter)-"
    "resid_post[L][-1](neutral)) over a TRAIN fold of those items; headline layer = max in-sample cave-"
    "necessity. On the held-out TEST fold, read the realized next-token softmax under (1) NEUTRAL, "
    "(2) COUNTER, (3) COUNTER+ablate u_cave-projection to the TRAIN-neutral mean, (4) COUNTER+ablate a "
    "matched-magnitude RANDOM direction. Report, over the held-out argmax-W* items: new_argmax under (3) "
    "== C-first-tok (install-C) / == W*-first-tok (no change) / == the item's NEUTRAL-condition argmax "
    "(restore-to-unpushed) / none-of-these (third token); dP(C)=P3(C)-P2(C); dP(W*)=P3(W*)-P2(W*); "
    "KL(P3||P1) vs KL(P2||P1); the same four for the random direction. "
    "Per model (THR=0.5): NO_EFFECT iff |dP(W*)| < EFFECT_EPS(0.01); else INSTALLS_C iff frac(argmax==C) "
    ">= THR; else RESTORES_NEUTRAL iff frac(argmax==neutral-argmax) >= THR AND KL3 < KL2; else "
    "SUPPRESSES_ONLY iff dP(W*) < 0 (W* mass drops, argmax goes to a third token); else NO_EFFECT. The "
    "matched random direction must not reproduce u_cave's effect (reported as the floor). Reported for "
    "base and -it; numbers + categories only, no claim attached to any bucket, sign, or the base-vs-it "
    "comparison."
)


# --------------------------------------------------------------------------- pure direction / fold helpers
def unit(v, eps=1e-8):
    """Unit vector; pure (tensor -> tensor). (cave_direction_overlay.unit / faithful_caving.unit.)"""
    return v / (v.norm() + eps)


def diff_of_means(pos, neg):
    """mean(pos) - mean(neg) as an (unnormalized) direction. pos/neg are [n_i, d] stacks. Pure.
    (cave_direction_overlay.diff_of_means; the diff-of-means cave fit of headset_direction.)"""
    return pos.mean(0) - neg.mean(0)


def split_indices(n, seed=SPLIT_SEED):
    """Deterministic ~50/50 train/test fold over n indices (cave_direction_overlay.split_indices /
    faithful_caving.split_indices). Disjoint + exhaustive; both folds non-empty for n>=2 (n==1 fallback
    maps train==test, selftest only). Pure."""
    import random as _r
    idx = list(range(n))
    _r.Random(seed).shuffle(idx)
    half = max(1, n // 2)
    train = sorted(idx[:half])
    test = sorted(idx[half:]) if n - half > 0 else sorted(idx[:half])
    return train, test


# --------------------------------------------------------------------------- pure distribution math
def l1_change(p_from, p_to):
    """Total |.| mass movement sum_v |p_to(v)-p_from(v)| (cave_direction_overlay.l1_change). Pure."""
    return float((p_to.float() - p_from.float()).abs().sum())


def kl_div(p, q, eps=1e-12):
    """KL(p || q) = sum_v p(v) * log(p(v)/q(v)) over 1-D probability tensors. Non-negative for proper
    distributions. Pure (clamps both to eps for numerical stability; the same per-coordinate form as
    cave_direction_overlay.kl_total_and_pair, summed over the whole vocab). Used for KL(P3||P1) and
    KL(P2||P1): the distance of the ablated / counter distribution to the NEUTRAL distribution."""
    pp = p.float().clamp_min(eps)
    qq = q.float().clamp_min(eps)
    return float((pp * (pp.log() - qq.log())).sum())


# --------------------------------------------------------------------------- pure argmax-identity buckets
def argmax_bucket(amx3, cid, aid, neu_argmax):
    """Classify the post-ablation argmax token id `amx3` into ONE of four mutually-exclusive buckets:
      'C'       : amx3 == C-first-tok                 (literal install of the correct competitor)
      'Wstar'   : amx3 == W*-first-tok                (no change -- still emits W*)
      'neutral' : amx3 == the item's NEUTRAL-condition argmax (restore-to-unpushed), and it is NOT C/W*
      'third'   : none of the above                   (a third token; mass diffused elsewhere)
    Priority C > Wstar > neutral > third so the four buckets PARTITION the items even when the neutral
    argmax happens to coincide with C or W* (those collapse into C / Wstar). Pure (ints in, str out)."""
    if amx3 == cid:
        return "C"
    if amx3 == aid:
        return "Wstar"
    if neu_argmax is not None and amx3 == neu_argmax:
        return "neutral"
    return "third"


def bucket_fracs(buckets):
    """Fractions of each bucket over a list of bucket labels. Returns a dict with C/Wstar/neutral/third
    fractions (0.0 when the list is empty) plus the raw count n. Pure."""
    n = len(buckets)
    keys = ("C", "Wstar", "neutral", "third")
    if n == 0:
        return {k: 0.0 for k in keys} | {"n": 0}
    out = {k: round(sum(1 for b in buckets if b == k) / n, 4) for k in keys}
    out["n"] = n
    return out


# --------------------------------------------------------------------------- pure decision
def decide_suppress(frac_c, frac_neutral, dP_wstar, kl3, kl2,
                    thr=THR, effect_eps=EFFECT_EPS, n_fit=None, min_fit=MIN_FIT):
    """Per-model decision over the measured numbers only (no hypothesis attached). Pure.
      INSUFFICIENT     iff n_fit is not None and n_fit < min_fit (no held-out argmax-W* substrate).
      NO_EFFECT        iff |dP_wstar| < effect_eps (the ablation did not move the realized W* mass).
      INSTALLS_C       iff frac_c >= thr (the ablated argmax becomes C on a majority of items).
      RESTORES_NEUTRAL iff frac_neutral >= thr AND kl3 < kl2 (the ablated argmax returns to the unpushed
                           token on a majority AND the distribution moves back toward NEUTRAL).
      SUPPRESSES_ONLY  iff dP_wstar < 0 (W* mass drops) but neither INSTALLS_C nor RESTORES_NEUTRAL holds
                           (the argmax goes to a third token; mass diffuses).
      else NO_EFFECT (nothing reached a category).
    Resolution order: INSUFFICIENT -> NO_EFFECT(|dP|) -> INSTALLS_C -> RESTORES_NEUTRAL -> SUPPRESSES_ONLY
    -> NO_EFFECT(fallthrough)."""
    if n_fit is not None and n_fit < min_fit:
        return {"category": "INSUFFICIENT", "n_fit": n_fit,
                "frac_argmax_C": (round(frac_c, 4) if frac_c is not None else None),
                "frac_argmax_neutral": (round(frac_neutral, 4) if frac_neutral is not None else None),
                "dP_Wstar": (round(dP_wstar, 6) if dP_wstar is not None else None),
                "kl3_to_neutral": (round(kl3, 6) if kl3 is not None else None),
                "kl2_to_neutral": (round(kl2, 6) if kl2 is not None else None),
                "kl_moves_toward_neutral": None,
                "msg": f"only {n_fit} held-out argmax-W* item(s) < MIN_FIT({min_fit}); no substrate to test."}

    def _f(x):
        return x if x is not None else 0.0
    moves_w = abs(_f(dP_wstar)) >= effect_eps
    installs = (frac_c is not None and frac_c >= thr)
    kl_back = (kl3 is not None and kl2 is not None and kl3 < kl2)
    restores = (frac_neutral is not None and frac_neutral >= thr and kl_back)
    suppresses = (dP_wstar is not None and dP_wstar < 0)

    if not moves_w:
        cat = "NO_EFFECT"
        msg = (f"|dP(W*)| {abs(_f(dP_wstar)):.4f} < {effect_eps}: ablating u_cave does not move the realized "
               f"W* mass at the answer slot.")
    elif installs:
        cat = "INSTALLS_C"
        msg = (f"frac(argmax==C) {frac_c:.3f} >= {thr}: ablating u_cave makes the emitted answer the correct "
               f"competitor C on a majority of the argmax-W* items (literal install-C).")
    elif restores:
        cat = "RESTORES_NEUTRAL"
        msg = (f"frac(argmax==neutral-argmax) {frac_neutral:.3f} >= {thr} AND KL(P3||P1) {kl3:.4f} < "
               f"KL(P2||P1) {kl2:.4f}: the emitted answer returns to the model's UNPUSHED token and the "
               f"realized distribution moves back toward NEUTRAL (restore-to-unpushed).")
    elif suppresses:
        cat = "SUPPRESSES_ONLY"
        msg = (f"dP(W*) {dP_wstar:+.4f} < 0 (W* mass drops) but neither INSTALLS_C (frac_C "
               f"{None if frac_c is None else round(frac_c, 3)} < {thr}) nor RESTORES_NEUTRAL (frac_neutral "
               f"{None if frac_neutral is None else round(frac_neutral, 3)} or KL3 {None if kl3 is None else round(kl3, 4)} "
               f">= KL2 {None if kl2 is None else round(kl2, 4)}): the argmax goes to a THIRD token, mass "
               f"diffuses.")
    else:
        cat = "NO_EFFECT"
        msg = (f"dP(W*) {_f(dP_wstar):+.4f} >= 0 and no category reached: the ablation does not suppress the "
               f"realized W* mass.")
    return {"category": cat,
            "n_fit": n_fit,
            "frac_argmax_C": (round(frac_c, 4) if frac_c is not None else None),
            "frac_argmax_neutral": (round(frac_neutral, 4) if frac_neutral is not None else None),
            "dP_Wstar": (round(dP_wstar, 6) if dP_wstar is not None else None),
            "kl3_to_neutral": (round(kl3, 6) if kl3 is not None else None),
            "kl2_to_neutral": (round(kl2, 6) if kl2 is not None else None),
            "kl_moves_toward_neutral": bool(kl_back),
            "installs_C": cat == "INSTALLS_C", "restores_neutral": cat == "RESTORES_NEUTRAL",
            "suppresses_only": cat == "SUPPRESSES_ONLY", "no_effect": cat == "NO_EFFECT",
            "msg": msg}


# --------------------------------------------------------------------------- real-run helpers
def _rname(L):
    """resid_post hook name at layer L (cave_direction_overlay._rname / faithful_caving._rname)."""
    return f"blocks.{L}.hook_resid_post"


def _full_softmax(logits):
    """Full next-token probability vector at the LAST position. gemma-2's final softcap is applied inside
    the forward, so softmax(logits[0,-1]) is the realized next-token distribution
    (cave_direction_overlay._full_softmax / faithful_caving._full_softmax). Returns a 1-D float tensor."""
    return torch.softmax(logits[0, -1].float(), dim=-1)


def _logp_diff_local(logits, cid, aid):
    """First-token margin M = logp(C) - logp(W*) at the last position (faithful_caving._logp_diff_local /
    rlhf_differential._logp_diff)."""
    lp = torch.log_softmax(logits[0, -1].float(), -1)
    return float(lp[cid] - lp[aid])


def _readout(P, cid, aid):
    """Realized readout from a full softmax P: argmax token id, P(C first-tok), P(W* first-tok). Pure.
    (faithful_caving._readout.)"""
    return {"argmax": int(P.argmax()), "p_c": float(P[cid]), "p_w": float(P[aid])}


def _proj_edit_hook(u, target_proj):
    """Hook that, at the readout position, sets the resid_post u-projection to target_proj (additive shift
    along u): r += (target_proj - r.u) * u. `u` must be on the model device. The necessity ablation of
    headset_direction / cave_direction_heldout / cave_direction_overlay / faithful_caving."""
    def hook(r, hook, u=u, target_proj=target_proj):
        cur = float(r[0, -1].float() @ u)
        r[0, -1] = r[0, -1] + ((target_proj - cur) * u).to(r.dtype)
        return r
    return hook


def _logits(model, ids, hooks=None):
    """Full last-position logits (optionally under fwd_hooks). Forward-only."""
    with torch.no_grad():
        return model.run_with_hooks(ids, fwd_hooks=hooks) if hooks else model(ids)


def _collect(model, pool, device, is_chat, fit_layers):
    """One model: per pool item, under NEUTRAL and COUNTER (W* asserted), in ONE forward each, cache the
    last-token resid_post at every fit layer AND read the full next-token softmax (P1=neutral, P2=counter)
    + first-token M. First-token-collision items (cid==aid) skipped. Forward-only.
    (Structure follows faithful_caving._collect / cave_direction_overlay._collect.)"""
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
    projection. u = normalize(mean(rc - rn)); proj_n = mean(rn . u). Returns (u, proj_n). Forward-free
    (operates on the cached residuals). Mirrors cave_direction_overlay / headset_direction."""
    Rc = torch.stack([recs[k]["rc"][L] for k in idxs]).to(device)
    Rn = torch.stack([recs[k]["rn"][L] for k in idxs]).to(device)
    u = unit(diff_of_means(Rc, Rn))
    proj_n = statistics.mean(float(recs[k]["rn"][L].to(device) @ u) for k in idxs)
    return u, proj_n


def _headline_layer(model, recs, fit_idx, fit_layers, device):
    """Headline layer = the fit layer with the largest IN-SAMPLE u_cave cave-necessity over the fit-set
    (the same headline-selection rule as cave_direction_overlay._necessity_layer / faithful_caving). For
    each layer, ablate the u_cave-projection on each fit COUNTER residual to the fit-set neutral mean and
    read the margin recovery frac=(M_ablate-M_ctr)/(M_neu-M_ctr); pick the max-mean layer. Forward-only."""
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


def _measure_model(name, is_chat, device, pool, fit_layers):
    """One model end-to-end. Collect realized + M readouts under neutral/counter on the wide pool; restrict
    to CAVING items (counter lowers M from neutral by >= MIN_EFFECT_NET) whose COUNTER argmax is the W*-
    first-tok; fit u_cave HELD-OUT (TRAIN fold) at the headline layer; on the TEST fold read P1/P2/P3/P4 and
    compute the argmax-identity buckets, dP, and KL distances for u_cave and a matched random direction.
    Forward-only. Returns a dict with the per-model decision."""
    from transformer_lens import HookedTransformer
    from rlhf_differential import MIN_EFFECT_NET
    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    recs = _collect(model, pool, device, is_chat, fit_layers)
    n = len(recs)
    g = torch.Generator(device="cpu").manual_seed(RAND_SEED)

    # CAVING items: counter lowers M from neutral by >= MIN_EFFECT_NET (same gate as cave_direction_overlay).
    cave_pos = [k for k, r in enumerate(recs) if (r["M_neu"] - r["M_ctr"]) >= MIN_EFFECT_NET]
    # restrict to items whose COUNTER argmax IS the W*-first-tok (the model would actually emit W*).
    argmaxW = [k for k in cave_pos if recs[k]["ctr"]["argmax"] == recs[k]["aid"]]

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
        out["decision"] = decide_suppress(None, None, None, None, None, n_fit=len(argmaxW))
        out["random_decision"] = decide_suppress(None, None, None, None, None, n_fit=len(argmaxW))
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
        out["decision"] = decide_suppress(None, None, None, None, None, n_fit=len(argmaxW))
        out["random_decision"] = decide_suppress(None, None, None, None, None, n_fit=len(argmaxW))
        return out

    L = headline
    u_cave, proj_n = _u_cave(recs, train, L, device)
    # matched-random unit direction; target its OWN train-neutral-mean projection (matched magnitude)
    rnd = torch.randn(u_cave.shape, generator=g).to(u_cave.dtype).to(device)
    u_rand = unit(rnd)
    proj_n_rand = statistics.mean(float(recs[k]["rn"][L].to(device) @ u_rand) for k in train)

    buckets3, r_buckets3 = [], []          # post-ablation argmax buckets (u_cave / random)
    dPC, dPW = [], []                      # u_cave realized first-token mass changes
    r_dPC, r_dPW = [], []
    kl3, kl2 = [], []                      # KL(P3||P1) and KL(P2||P1)
    r_kl3 = []                             # KL(P4_rand||P1)
    rows = []
    for k in test:
        r = recs[k]
        cid, aid = r["cid"], r["aid"]
        P1 = r["P1"].to(device)                                   # NEUTRAL (cached)
        P2 = r["P2"].to(device)                                   # COUNTER (cached)
        neu_argmax = r["neu"]["argmax"]
        # (3) COUNTER + ablate u_cave to the train-neutral mean
        h = [(_rname(L), _proj_edit_hook(u_cave, proj_n))]
        P3 = _full_softmax(_logits(model, r["counter"], hooks=h))
        # (4) COUNTER + ablate a matched-magnitude RANDOM direction
        hr = [(_rname(L), _proj_edit_hook(u_rand, proj_n_rand))]
        P4 = _full_softmax(_logits(model, r["counter"], hooks=hr))

        amx3, amx4 = int(P3.argmax()), int(P4.argmax())
        b3 = argmax_bucket(amx3, cid, aid, neu_argmax)
        b4 = argmax_bucket(amx4, cid, aid, neu_argmax)
        buckets3.append(b3); r_buckets3.append(b4)
        dPC.append(float(P3[cid]) - float(P2[cid])); dPW.append(float(P3[aid]) - float(P2[aid]))
        r_dPC.append(float(P4[cid]) - float(P2[cid])); r_dPW.append(float(P4[aid]) - float(P2[aid]))
        k3, k2 = kl_div(P3, P1), kl_div(P2, P1)
        kl3.append(k3); kl2.append(k2); r_kl3.append(kl_div(P4, P1))
        rows.append({"i": r["i"], "q": r["q"], "cid": cid, "aid": aid, "neu_argmax": neu_argmax,
                     "ctr_argmax": r["ctr"]["argmax"], "amx_ablate": amx3, "amx_rand": amx4,
                     "bucket_ablate": b3, "bucket_rand": b4,
                     "P2_C": round(float(P2[cid]), 6), "P2_W": round(float(P2[aid]), 6),
                     "P3_C": round(float(P3[cid]), 6), "P3_W": round(float(P3[aid]), 6),
                     "kl3_to_neutral": round(k3, 6), "kl2_to_neutral": round(k2, 6)})
        print(f"  [{'it' if is_chat else 'base'} L{L}] item {r['i']} amx3={amx3} bucket={b3} "
              f"dP(C)={dPC[-1]:+.4f} dP(W*)={dPW[-1]:+.4f} KL3={k3:.4f} KL2={k2:.4f}", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    def _mean(xs):
        return statistics.mean(xs) if xs else None

    bf = bucket_fracs(buckets3)
    rbf = bucket_fracs(r_buckets3)
    numbers = {
        "headline_layer": L, "n_train": len(train), "n_test": len(test),
        "argmax_fracs": bf,
        "dP_C": (round(_mean(dPC), 6) if dPC else None),
        "dP_Wstar": (round(_mean(dPW), 6) if dPW else None),
        "kl3_to_neutral": (round(_mean(kl3), 6) if kl3 else None),
        "kl2_to_neutral": (round(_mean(kl2), 6) if kl2 else None),
        "rows": rows,
    }
    random_numbers = {
        "argmax_fracs": rbf,
        "dP_C": (round(_mean(r_dPC), 6) if r_dPC else None),
        "dP_Wstar": (round(_mean(r_dPW), 6) if r_dPW else None),
        "kl3_to_neutral": (round(_mean(r_kl3), 6) if r_kl3 else None),
        "kl2_to_neutral": (round(_mean(kl2), 6) if kl2 else None),   # same COUNTER baseline KL2
    }
    out["numbers"] = numbers
    out["random_numbers"] = random_numbers
    out["decision"] = decide_suppress(bf["C"], bf["neutral"], numbers["dP_Wstar"],
                                      numbers["kl3_to_neutral"], numbers["kl2_to_neutral"],
                                      n_fit=len(argmaxW))
    out["random_decision"] = decide_suppress(rbf["C"], rbf["neutral"], random_numbers["dP_Wstar"],
                                             random_numbers["kl3_to_neutral"],
                                             random_numbers["kl2_to_neutral"], n_fit=len(argmaxW))
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
        "cue": "cave_suppress_vs_install", "pool_size": len(pool), "fit_layers": fit_layers,
        "metric": ("on argmax-W* caving items, the post-u_cave-ablation realized next-token distribution "
                   "under NEUTRAL(P1)/COUNTER(P2)/COUNTER+ablate-u_cave(P3)/COUNTER+ablate-random(P4): "
                   "argmax-identity buckets (C / W* / neutral-argmax / third), dP(C), dP(W*), and "
                   "KL(P3||P1) vs KL(P2||P1)"),
        "thresholds": {"THR": THR, "EFFECT_EPS": EFFECT_EPS, "SPLIT_SEED": SPLIT_SEED,
                       "RAND_SEED": RAND_SEED, "MIN_FIT": MIN_FIT},
        "decision_rule": DECISION_RULE,
        "base": res["base"], "it": res["it"],
    }
    Path("out").mkdir(exist_ok=True)
    Path(f"out/cave_suppress_vs_install_{tag}.json").write_text(json.dumps(out, indent=2, default=str))
    for m in MODELS:
        r = res[m]
        dd = r["decision"]
        rd = r["random_decision"]
        nb = r.get("numbers") or {}
        af = (nb.get("argmax_fracs") or {})
        print(f"[{m}] {dd['category']} (L{r.get('headline_layer')}) n_argmaxW={r['n_argmaxW_cave']} "
              f"frac_C={af.get('C')} frac_neutral={af.get('neutral')} frac_Wstar={af.get('Wstar')} "
              f"frac_third={af.get('third')} dP(W*)={dd.get('dP_Wstar')} "
              f"KL3={dd.get('kl3_to_neutral')} KL2={dd.get('kl2_to_neutral')} "
              f"| RANDOM={rd['category']} (floor)", flush=True)
    print(f"[done] wrote out/cave_suppress_vs_install_{tag}.json", flush=True)


# --------------------------------------------------------------------------- selftest (model-free, CPU, no model load)
def _dist(V, masses):
    """Probability vector of length V from a {idx: mass} dict; remaining mass spread uniformly over the
    rest. Pure (selftest only)."""
    spec = sum(masses.values())
    rest = [j for j in range(V) if j not in masses]
    q = torch.zeros(V)
    for idx, m in masses.items():
        q[idx] = m
    if rest:
        q[rest] = (1.0 - spec) / len(rest)
    return q


def selftest():
    torch.manual_seed(0)
    V = 200
    cid, aid, third = 3, 7, 11        # C-first-tok, W*-first-tok, an arbitrary third token

    # ---------- argmax_bucket partition (priority C > Wstar > neutral > third) ----------
    assert argmax_bucket(cid, cid, aid, cid) == "C"            # argmax==C wins even if neutral argmax==C
    assert argmax_bucket(aid, cid, aid, aid) == "Wstar"        # argmax==W* (no change)
    assert argmax_bucket(13, cid, aid, 13) == "neutral"        # argmax==the item's neutral argmax (not C/W*)
    assert argmax_bucket(99, cid, aid, 13) == "third"          # none of the above -> third token
    assert argmax_bucket(cid, cid, aid, 13) == "C"             # C beats neutral
    # buckets partition: fracs sum to 1.0
    bf = bucket_fracs(["C", "C", "neutral", "third", "Wstar"])
    assert abs(bf["C"] + bf["Wstar"] + bf["neutral"] + bf["third"] - 1.0) < 1e-9 and bf["n"] == 5, bf
    assert bf["C"] == 0.4 and bf["neutral"] == 0.2, bf
    assert bucket_fracs([]) == {"C": 0.0, "Wstar": 0.0, "neutral": 0.0, "third": 0.0, "n": 0}
    print("[selftest] argmax_bucket priority + partition + empty OK")

    # ---------- KL math: identical -> 0; closer distribution -> smaller KL ----------
    P1 = _dist(V, {cid: 0.5, aid: 0.1})           # NEUTRAL: argmax C
    assert abs(kl_div(P1, P1)) < 1e-6
    P2 = _dist(V, {aid: 0.6, cid: 0.05})          # COUNTER: argmax W*
    P3_close = _dist(V, {cid: 0.5, aid: 0.1})     # ablated back to NEUTRAL -> KL(P3||P1) ~ 0 < KL(P2||P1)
    assert kl_div(P3_close, P1) < kl_div(P2, P1), (kl_div(P3_close, P1), kl_div(P2, P1))
    assert kl_div(P2, P1) > 0 and kl_div(P3_close, P1) >= 0
    # l1_change sanity (matches cave_direction_overlay)
    a = _dist(V, {aid: 0.6}); b = a.clone(); b[aid] -= 0.4; b[cid] += 0.4
    assert abs(l1_change(a, b) - 0.8) < 1e-6, l1_change(a, b)
    print(f"[selftest] kl_div: identical=0, restore KL3={kl_div(P3_close, P1):.4f} < counter KL2={kl_div(P2, P1):.4f}; l1 OK")

    # ---------- split primitive ----------
    tr, te = split_indices(10, SPLIT_SEED)
    assert set(tr) | set(te) == set(range(10)) and not (set(tr) & set(te)) and tr and te
    assert split_indices(10, SPLIT_SEED) == split_indices(10, SPLIT_SEED)
    print(f"[selftest] split_indices train={tr} test={te} (disjoint, exhaustive, deterministic)")

    # ============================================================ DECISION-BOUNDARY scenarios =============
    # (i) INSTALLS_C (and the neutral argmax is C -> RESTORES_NEUTRAL would also be true, but INSTALLS_C
    #     wins by resolution order): ablation moves argmax to C, KL toward neutral.
    di = decide_suppress(frac_c=0.8, frac_neutral=0.8, dP_wstar=-0.4, kl3=0.05, kl2=0.9, n_fit=8)
    assert di["category"] == "INSTALLS_C" and di["installs_C"], di
    print(f"[selftest] (i) INSTALLS_C: frac_C=0.8 dP(W*)=-0.4 KL3<KL2 -> {di['category']}")

    # (ii) SUPPRESSES_ONLY: W* mass drops but argmax goes to a THIRD token, KL not reduced.
    dii = decide_suppress(frac_c=0.0, frac_neutral=0.0, dP_wstar=-0.3, kl3=1.2, kl2=0.9, n_fit=8)
    assert dii["category"] == "SUPPRESSES_ONLY" and dii["suppresses_only"], dii
    assert dii["kl_moves_toward_neutral"] is False, dii      # KL3 >= KL2 -> NOT toward neutral
    print(f"[selftest] (ii) SUPPRESSES_ONLY: dP(W*)=-0.3, argmax third, KL3>=KL2 -> {dii['category']}")

    # (iii) RESTORES_NEUTRAL: argmax returns to the neutral-condition token (NOT C), KL3 < KL2.
    diii = decide_suppress(frac_c=0.0, frac_neutral=0.7, dP_wstar=-0.2, kl3=0.1, kl2=0.8, n_fit=8)
    assert diii["category"] == "RESTORES_NEUTRAL" and diii["restores_neutral"], diii
    assert diii["kl_moves_toward_neutral"] is True, diii
    # RESTORES_NEUTRAL must FAIL if the argmax returns to neutral but KL does NOT move back (KL3 >= KL2)
    diii_b = decide_suppress(frac_c=0.0, frac_neutral=0.7, dP_wstar=-0.2, kl3=0.9, kl2=0.8, n_fit=8)
    assert diii_b["category"] == "SUPPRESSES_ONLY", diii_b   # frac_neutral high but KL not back -> suppress
    print(f"[selftest] (iii) RESTORES_NEUTRAL: frac_neutral=0.7 KL3<KL2 -> {diii['category']}; "
          f"KL3>=KL2 falls through to {diii_b['category']}")

    # (iv) NO_EFFECT: |dP(W*)| below EFFECT_EPS (checked FIRST, before any frac).
    div = decide_suppress(frac_c=0.9, frac_neutral=0.9, dP_wstar=0.0, kl3=0.0, kl2=0.0, n_fit=8)
    assert div["category"] == "NO_EFFECT" and div["no_effect"], div
    # tiny-but-nonzero W* move still NO_EFFECT below the floor; just above the floor reaches a category
    assert decide_suppress(0.9, 0.0, -(EFFECT_EPS - 1e-4), 0.0, 0.0, n_fit=8)["category"] == "NO_EFFECT"
    assert decide_suppress(0.9, 0.0, -(EFFECT_EPS + 1e-4), 0.05, 0.9, n_fit=8)["category"] == "INSTALLS_C"
    print(f"[selftest] (iv) NO_EFFECT: |dP(W*)|<{EFFECT_EPS} -> {div['category']} (checked before fracs)")

    # threshold boundaries on THR
    assert decide_suppress(THR, 0.0, -0.3, 0.1, 0.9, n_fit=8)["category"] == "INSTALLS_C"           # frac_C==THR
    assert decide_suppress(THR - 1e-6, 0.0, -0.3, 1.0, 0.9, n_fit=8)["category"] == "SUPPRESSES_ONLY"
    assert decide_suppress(0.0, THR, -0.3, 0.1, 0.9, n_fit=8)["category"] == "RESTORES_NEUTRAL"      # frac_neu==THR
    # INSUFFICIENT below MIN_FIT
    assert decide_suppress(None, None, None, None, None, n_fit=1)["category"] == "INSUFFICIENT"
    print("[selftest] THR boundaries + INSUFFICIENT OK")

    # ============================================================ END-TO-END synthetic distributions =====
    # Build full per-item distributions for the FOUR conditions and run the bucket/dP/KL pipeline exactly
    # as _measure_model does (minus the model forward), then feed decide_suppress. The neutral argmax for
    # every item is C (the unpushed answer); the counter argmax is W* (caved).
    def end_to_end(make_P3, make_P4, n=8):
        buckets3, r_buckets3, dPW, kl3, kl2, r_dPW, r_kl3 = [], [], [], [], [], [], []
        for j in range(n):
            P1 = _dist(V, {cid: 0.6, aid: 0.05})    # NEUTRAL: argmax C
            P2 = _dist(V, {aid: 0.6, cid: 0.05})    # COUNTER: argmax W* (caved)
            neu_argmax = int(P1.argmax())           # == cid
            P3 = make_P3(j); P4 = make_P4(j)
            buckets3.append(argmax_bucket(int(P3.argmax()), cid, aid, neu_argmax))
            r_buckets3.append(argmax_bucket(int(P4.argmax()), cid, aid, neu_argmax))
            dPW.append(float(P3[aid]) - float(P2[aid])); r_dPW.append(float(P4[aid]) - float(P2[aid]))
            kl3.append(kl_div(P3, P1)); kl2.append(kl_div(P2, P1)); r_kl3.append(kl_div(P4, P1))
        bf = bucket_fracs(buckets3); rbf = bucket_fracs(r_buckets3)
        dec = decide_suppress(bf["C"], bf["neutral"], statistics.mean(dPW),
                              statistics.mean(kl3), statistics.mean(kl2), n_fit=n)
        rdec = decide_suppress(rbf["C"], rbf["neutral"], statistics.mean(r_dPW),
                               statistics.mean(r_kl3), statistics.mean(kl2), n_fit=n)
        return dec, rdec, bf, rbf

    # SCENARIO A: u_cave INSTALLS_C (argmax->C, KL toward neutral); RANDOM does NOT (leaves argmax on W*).
    P3_install = _dist(V, {cid: 0.55, aid: 0.1})    # argmax C, close to NEUTRAL
    P4_floor = _dist(V, {aid: 0.58, cid: 0.06})     # random ~ leaves the counter argmax on W*
    decA, rdecA, bfA, rbfA = end_to_end(lambda j: P3_install, lambda j: P4_floor)
    assert decA["category"] == "INSTALLS_C", (decA, bfA)
    assert rdecA["category"] != "INSTALLS_C" and rbfA["Wstar"] == 1.0, (rdecA, rbfA)   # floor: random doesn't install
    print(f"[selftest] (A) end-to-end u_cave={decA['category']} (frac_C={bfA['C']}) | "
          f"RANDOM floor={rdecA['category']} (frac_W*={rbfA['Wstar']})")

    # SCENARIO B: u_cave SUPPRESSES_ONLY (W* down, argmax->third, KL up); RANDOM ~ does nothing.
    P3_third = _dist(V, {third: 0.4, aid: 0.2, cid: 0.05})   # argmax a THIRD token; W* still > before? no: 0.2<0.6 so dP(W*)<0
    decB, rdecB, bfB, rbfB = end_to_end(lambda j: P3_third, lambda j: P4_floor)
    assert decB["category"] == "SUPPRESSES_ONLY", (decB, bfB)
    assert bfB["third"] == 1.0 and decB["dP_Wstar"] < 0, (bfB, decB)
    print(f"[selftest] (B) end-to-end u_cave={decB['category']} (frac_third={bfB['third']} dP(W*)={decB['dP_Wstar']})")

    # SCENARIO C: u_cave RESTORES_NEUTRAL to a neutral-argmax token that is NOT C. Use a neutral whose
    # argmax is a token != C, and an ablation that returns to it with KL toward neutral.
    na = 17                                          # the neutral-condition argmax (a non-C, non-W* token)
    def restore_e2e(n=8):
        buckets3, dPW, kl3, kl2 = [], [], [], []
        for _ in range(n):
            P1 = _dist(V, {na: 0.6, cid: 0.1, aid: 0.05})   # NEUTRAL argmax == token na (not C)
            P2 = _dist(V, {aid: 0.6, cid: 0.05})            # COUNTER argmax == W*
            P3 = _dist(V, {na: 0.58, cid: 0.1, aid: 0.06})  # ablated back to ~NEUTRAL (argmax na), KL small
            buckets3.append(argmax_bucket(int(P3.argmax()), cid, aid, int(P1.argmax())))
            dPW.append(float(P3[aid]) - float(P2[aid])); kl3.append(kl_div(P3, P1)); kl2.append(kl_div(P2, P1))
        bf = bucket_fracs(buckets3)
        return decide_suppress(bf["C"], bf["neutral"], statistics.mean(dPW),
                               statistics.mean(kl3), statistics.mean(kl2), n_fit=n), bf
    decC, bfC = restore_e2e()
    assert decC["category"] == "RESTORES_NEUTRAL", (decC, bfC)
    assert bfC["neutral"] == 1.0 and decC["kl_moves_toward_neutral"], (bfC, decC)
    print(f"[selftest] (C) end-to-end u_cave={decC['category']} (frac_neutral={bfC['neutral']} "
          f"KL3<KL2={decC['kl_moves_toward_neutral']})")

    # SCENARIO D: NO_EFFECT -- ablation leaves the counter distribution essentially unchanged.
    P3_noop = _dist(V, {aid: 0.6, cid: 0.05})        # == P2 -> dP(W*)=0, argmax stays W*
    decD, rdecD, bfD, _ = end_to_end(lambda j: P3_noop, lambda j: P4_floor)
    assert decD["category"] == "NO_EFFECT", (decD, bfD)
    print(f"[selftest] (D) end-to-end u_cave={decD['category']} (dP(W*)={decD['dP_Wstar']})")

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
