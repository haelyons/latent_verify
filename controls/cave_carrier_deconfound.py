"""CAVE-CARRIER DE-CONFOUND: is the rank-1 cave-direction u_cave a SPECIFIC carrier of the cave, or is the
prior RESTORES_NEUTRAL result an artifact of (1) installing the neutral mean it then measures (CIRCULARITY)
and (2) comparing against a near-orthogonal random direction (IN-SHIFT SPECIFICITY)? (sibling of
cave_suppress_vs_install.py / faithful_caving.py / cave_direction_heldout.py / headset_direction.py.)

CONTEXT (neutral). A prior control (cave_suppress_vs_install.py) found that at base Q/A, on the held-out
argmax-W* caving items (counter argmax = W*-aligned first-token), ablating the rank-1 diff-of-means cave
direction
  u_cave(L) = normalize(mean_items( resid_post[L][-1](counter) - resid_post[L][-1](neutral) ))
by SETTING its projection on the counter residual to the TRAIN-neutral mean returns the emitted argmax to
the model's NEUTRAL answer on all held-out items (KL to the neutral distribution collapses), while a
matched ISOTROPIC-RANDOM rank-1 direction does nothing (RESTORES_NEUTRAL, random floor clean). Two
confounds remain in that result:
  (1) CIRCULARITY -- the ablation INSTALLS the train-neutral-mean projection value, which is exactly what
      the success metric (argmax == neutral-argmax, KL toward neutral) rewards. A "restore" could be the
      installed neutral value rather than the REMOVAL of the cave component.
  (2) IN-SHIFT SPECIFICITY -- an isotropic random direction is, in high d, near-orthogonal to the
      counter-neutral shift, so it trivially does nothing; it does NOT test whether u_cave is special
      AMONG directions that lie INSIDE that shift.

This control measures both, with NO new mechanism. SUBSTRATE: base (qa) and -it (chat), default
gemma-2-9b / gemma-2-9b-it, on the wide misconception pool (misconception_pool.ITEMS_WIDE), restricted to
the SAME argmax-W* caving items as cave_suppress_vs_install (CAVING gate: counter lowers the first-token
margin M=logp(C)-logp(W*) from neutral by >= MIN_EFFECT_NET; AND counter argmax IS the W*-first-token).
Held-out: fit u_cave on a TRAIN fold, evaluate on the disjoint TEST fold, at the HEADLINE layer (max
in-sample u_cave necessity over the train fold, the same headline-selection rule as cave_suppress_vs_install
/ headset_direction). At base we expect a non-trivial argmax-W* set; -it is run for completeness (expect 0
argmax-W* items -> INSUFFICIENT, reported as such).

IN-SHIFT CONTROL DIRECTION. From the per-item (rc - rn) residual-difference stack D=[n_train, d] at the
headline layer, SVD the CENTERED matrix (D - mean(D)) -> right singular vectors (principal components of the
shift). u_orth = a high-variance in-shift direction ORTHOGONALIZED against u_cave: take the top PC, remove
its u_cave component (Gram-Schmidt), unit-normalize; if its residual norm collapses (top PC ~ u_cave), fall
back to the next PC. u_orth is unit-norm, orthogonal to u_cave (|u_orth . u_cave| ~ 0), and lives inside the
shift (high variance fraction). The variance fractions of u_cave and u_orth in the (uncentered) shift are
reported so "in-shift" is interpretable.

CONDITIONS (per held-out counter residual, read the realized next-token distribution P; neutral = P1,
counter = P2; all ablations are projection edits at resid_post[headline][-1]):
  (A) u_cave -> TRAIN-neutral-mean projection         (the original RESTORES_NEUTRAL operation).
  (B) u_cave ZERO-ablation: set the u_cave projection to 0 (REMOVE the component; do NOT install the
      neutral mean).
  (C) u_cave RESAMPLE-ablation: set the u_cave projection to a SHUFFLED neutral value (another test item's
      neutral u_cave-projection) -- removes the 'install the specific per-item neutral value' content.
  (D) u_orth -> its OWN train-neutral-mean projection (in-shift specificity control: an orthogonalized
      in-shift direction, MATCHED operation to (A)).
  (E) matched ISOTROPIC-RANDOM rank-1 -> its own train-neutral mean (the existing floor from
      cave_suppress_vs_install).
Per condition, over the held-out items: frac(argmax == neutral-argmax), frac(argmax == W*-first-tok),
dP(W*) = mean(P_cond(W*) - P2(W*)), and KL(P_cond || P1) vs KL(P2 || P1) (the counter baseline). Also
reported: the train-neutral-mean u_cave projection value vs 0 (so the zero-vs-neutral-mean contrast is
interpretable), and the variance fractions of u_cave / u_orth in the shift.

This is claim-blind: it measures whether removing the cave component (without installing the neutral mean)
still restores neutral, and whether an in-shift orthogonal direction restores neutral. It attaches no
hypothesis to any condition, sign, or the base-vs-it comparison.

NEUTRAL DECISION (module constant THR=0.5; numbers + categories only, no hypothesis):
  - NOT_CIRCULAR iff the ZERO-ablation (B) OR the RESAMPLE-ablation (C) STILL restores neutral
        (frac(argmax == neutral-argmax) >= THR AND KL reduced vs counter) -- removing the counter component
        restores neutral even WITHOUT installing the neutral-mean value; else CIRCULAR (only the
        neutral-mean substitution (A) restores).
  - SPECIFIC_CARRIER iff the orthogonalized in-shift direction (D) does NOT restore
        (frac(argmax == neutral-argmax) < THR OR KL not reduced) WHILE u_cave (A) DOES restore
        (frac >= THR AND KL reduced) -- u_cave is special among in-shift directions; else SHARED_BY_SHIFT
        (D also restores -> u_cave is just one slice of a higher-rank shift), or NO_RESTORE if (A) itself
        does not restore (no carrier to harden).
  - HARDENED_CARRIER iff NOT_CIRCULAR AND SPECIFIC_CARRIER. All conditions' numbers reported per model.

Forward-only (diff-of-means + SVD + projection edits + full-softmax readouts; no backward) -> fits the 40GB
A100. Reuses verified primitives: PUSH/NEUTRAL from job_truthful_flip; _helpers/MIN_EFFECT_NET from
rlhf_differential; ITEMS_WIDE from misconception_pool; the held-out fold split / diff-of-means cave fit /
projection-edit ablation / full-softmax + first-token readouts / KL helper / argmax buckets / headline-layer
selection construction from cave_suppress_vs_install (re-implemented here verbatim as small pure helpers so
--selftest is standalone on CPU with nothing else on sys.path -- the same FLAT-scp convention
cave_suppress_vs_install / faithful_caving / cave_direction_overlay use); FIT_LAYERS from headset_direction
(deferred at real-run time). The new logic is the SVD in-shift direction, the zero / resample ablations, the
in-shift / orthogonality readouts and the three de-confound decisions -- all pure and covered by the
model-free --selftest.

  python controls/cave_carrier_deconfound.py --selftest
  python controls/cave_carrier_deconfound.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it \
    --tag 9b --device cuda --chat
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

# Pre-registered thresholds (see module docstring). Neutral: stated on the measured numbers only.
THR = 0.5            # frac(argmax == neutral-argmax) threshold for "restores neutral" in every condition
SPLIT_SEED = 0       # deterministic train/test fold (same convention as cave_suppress_vs_install)
RAND_SEED = 0        # deterministic matched-random-direction control
RESAMPLE_SEED = 0    # deterministic neutral-projection shuffle for the RESAMPLE ablation (C)
MIN_FIT = 3          # below this many argmax-W* caving items, no held-out direction can be fit/tested
ORTH_EPS = 1e-3      # residual-norm floor for the Gram-Schmidt in-shift direction (top PC ~ u_cave -> next)
ORTH_DOT_TOL = 1e-3  # |u_orth . u_cave| must sit below this for u_orth to count as orthogonal

# Diff-of-means cave-direction layer sweep. SAME value as headset_direction.FIT_LAYERS /
# cave_suppress_vs_install.FIT_LAYERS ([24,28,32,36], the set's output range L21-L34); defined here as a
# module constant so --selftest needs nothing on sys.path. The real run also defers a `from
# headset_direction import FIT_LAYERS` so the sweep stays pinned to the reference.
FIT_LAYERS = [24, 28, 32, 36]

MODELS = ("base", "it")

DECISION_RULE = (
    "On the wide misconception pool, build NEUTRAL and COUNTER (W* asserted) prompts (job_truthful_flip "
    "turns; qa template for base, chat template for -it). Restrict to the argmax-W* CAVING items (counter "
    "lowers M=logp(C)-logp(W*) from neutral by >= MIN_EFFECT_NET AND counter argmax IS the W*-first-token), "
    "the same selection as cave_suppress_vs_install. Fit u_cave = normalize(mean(resid_post[L][-1](counter)-"
    "resid_post[L][-1](neutral))) over a TRAIN fold; headline layer L = max in-sample u_cave necessity over "
    "the train fold. From the per-item (rc-rn) train stack at L, SVD the centered matrix and Gram-Schmidt a "
    "high-variance in-shift PC against u_cave -> u_orth (unit, orthogonal to u_cave, in-shift). On the "
    "held-out TEST counter residual read the realized next-token softmax under: (A) u_cave->train-neutral "
    "mean; (B) u_cave projection->0 (zero-ablation); (C) u_cave projection->a SHUFFLED test-item neutral "
    "value (resample-ablation); (D) u_orth->its own train-neutral mean; (E) matched isotropic-random rank-1 "
    "->its own neutral mean. NEUTRAL=P1, COUNTER=P2. Per condition: frac(argmax==neutral-argmax), "
    "frac(argmax==W*), dP(W*)=mean(P_cond(W*)-P2(W*)), KL(P_cond||P1) vs KL(P2||P1). 'restores neutral' := "
    "frac(argmax==neutral-argmax) >= THR(0.5) AND KL(P_cond||P1) < KL(P2||P1). "
    "NOT_CIRCULAR iff (B) OR (C) restores; else CIRCULAR. SPECIFIC_CARRIER iff (A) restores AND (D) does "
    "NOT restore; else SHARED_BY_SHIFT (D also restores) or NO_RESTORE (A does not restore). "
    "HARDENED_CARRIER iff NOT_CIRCULAR AND SPECIFIC_CARRIER. Reported for base and -it; numbers + categories "
    "only, no claim attached to any condition, sign, or the base-vs-it comparison."
)


# --------------------------------------------------------------------------- pure direction / fold helpers
def unit(v, eps=1e-8):
    """Unit vector; pure (tensor -> tensor). (cave_suppress_vs_install.unit / cave_direction_overlay.unit.)"""
    return v / (v.norm() + eps)


def diff_of_means(pos, neg):
    """mean(pos) - mean(neg) as an (unnormalized) direction. pos/neg are [n_i, d] stacks. Pure.
    (cave_suppress_vs_install.diff_of_means; the diff-of-means cave fit of headset_direction.)"""
    return pos.mean(0) - neg.mean(0)


def split_indices(n, seed=SPLIT_SEED):
    """Deterministic ~50/50 train/test fold over n indices (cave_suppress_vs_install.split_indices /
    faithful_caving.split_indices). Disjoint + exhaustive; both folds non-empty for n>=2 (n==1 fallback maps
    train==test, selftest only). Pure."""
    import random as _r
    idx = list(range(n))
    _r.Random(seed).shuffle(idx)
    half = max(1, n // 2)
    train = sorted(idx[:half])
    test = sorted(idx[half:]) if n - half > 0 else sorted(idx[:half])
    return train, test


def in_shift_direction(diffs, u_cave, eps=ORTH_EPS):
    """Orthogonalized high-variance IN-SHIFT direction from the per-item difference stack diffs=[n, d]
    (rc - rn at the fit layer). SVD the CENTERED matrix (D - mean(D)) -> right singular vectors V (the
    principal components of the per-item shift variation). Walk the PCs in descending variance order; for
    each, Gram-Schmidt-remove its u_cave component and, if the residual norm exceeds eps (the PC is not
    essentially u_cave itself), return its unit-normalized residual as u_orth. This yields a direction that
    is (i) unit-norm, (ii) orthogonal to u_cave, (iii) high-variance inside the shift. Returns
    (u_orth, info) where info carries the chosen PC index, the centered-SVD variance fractions, and the
    realized |u_orth . u_cave|. u_orth is None (info still returned) if no PC clears eps. Pure (torch in)."""
    u_cave = unit(u_cave)
    X = diffs.float()
    Xc = X - X.mean(0, keepdim=True)
    # right singular vectors are the PCs of the per-item shift variation; svdvals^2 give the variance.
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    sv2 = (S * S)
    tot = float(sv2.sum())
    var_frac = [(float(sv2[i]) / tot if tot > 0 else 0.0) for i in range(Vh.shape[0])]
    chosen, u_orth, resid_norm = None, None, None
    for i in range(Vh.shape[0]):
        pc = Vh[i]
        resid = pc - (pc @ u_cave) * u_cave           # Gram-Schmidt remove the u_cave component
        rn = float(resid.norm())
        if rn > eps:
            chosen = i
            u_orth = (resid / rn)
            resid_norm = rn
            break
    dot = (float(u_orth @ u_cave) if u_orth is not None else None)
    return u_orth, {"pc_index": chosen, "centered_var_fracs": [round(v, 6) for v in var_frac],
                    "gs_residual_norm": (round(resid_norm, 6) if resid_norm is not None else None),
                    "u_orth_dot_u_cave": (round(dot, 8) if dot is not None else None)}


def variance_fraction(diffs, u):
    """Fraction of the total per-item squared shift magnitude that lies ALONG unit direction u:
        sum_i (d_i . u)^2 / sum_i ||d_i||^2,  over the rows d_i of diffs=[n, d] (the uncentered shift stack).
    1.0 => the whole shift is along u; ~1/d => u is generic. Pure (torch in, float out)."""
    X = diffs.float()
    u = unit(u)
    proj = X @ u                                       # [n]
    num = float((proj * proj).sum())
    den = float((X * X).sum())
    return (num / den) if den > 0 else 0.0


# --------------------------------------------------------------------------- pure distribution math
def kl_div(p, q, eps=1e-12):
    """KL(p || q) = sum_v p(v) * log(p(v)/q(v)) over 1-D probability tensors. Non-negative for proper
    distributions. Pure (clamps both to eps; same per-coordinate form as cave_suppress_vs_install.kl_div).
    Used for KL(P_cond||P1) vs KL(P2||P1): distance of each condition's distribution to the NEUTRAL one."""
    pp = p.float().clamp_min(eps)
    qq = q.float().clamp_min(eps)
    return float((pp * (pp.log() - qq.log())).sum())


# --------------------------------------------------------------------------- pure restore classification
def restores_neutral(frac_neutral, kl_cond, kl_counter, thr=THR):
    """A condition 'restores neutral' iff the emitted argmax returns to the item's NEUTRAL-condition argmax
    on a majority (frac_neutral >= thr) AND the realized distribution moves BACK toward NEUTRAL
    (KL(P_cond||P1) < KL(P2||P1)). Pure (floats in, bool out). None inputs -> False."""
    if frac_neutral is None or kl_cond is None or kl_counter is None:
        return False
    return (frac_neutral >= thr) and (kl_cond < kl_counter)


def decide_carrier(cond, n_fit=None, min_fit=MIN_FIT, thr=THR):
    """De-confound decision over the measured per-condition numbers only (no hypothesis attached). Pure.
    `cond` is a dict keyed 'A'..'E', each value a dict with at least:
        frac_neutral (frac argmax == neutral-argmax), kl_cond (KL(P_cond||P1)), kl_counter (KL(P2||P1)).
      INSUFFICIENT       iff n_fit is not None and n_fit < min_fit (no held-out argmax-W* substrate).
      restore(X) := restores_neutral(cond[X].frac_neutral, cond[X].kl_cond, cond[X].kl_counter).
      NOT_CIRCULAR  iff restore(B) OR restore(C); else CIRCULAR.
      SPECIFIC_CARRIER iff restore(A) AND NOT restore(D); SHARED_BY_SHIFT iff restore(A) AND restore(D);
                       NO_RESTORE iff NOT restore(A).
      HARDENED_CARRIER iff NOT_CIRCULAR AND SPECIFIC_CARRIER.
    All four restore() booleans + the two sub-decisions + the headline category are returned."""
    if n_fit is not None and n_fit < min_fit:
        return {"category": "INSUFFICIENT", "n_fit": n_fit,
                "restore_A": None, "restore_B": None, "restore_C": None, "restore_D": None,
                "circularity": None, "specificity": None, "hardened_carrier": False,
                "msg": f"only {n_fit} held-out argmax-W* item(s) < MIN_FIT({min_fit}); no substrate to test."}

    def _r(x):
        c = cond.get(x) or {}
        return restores_neutral(c.get("frac_neutral"), c.get("kl_cond"), c.get("kl_counter"), thr=thr)

    rA, rB, rC, rD = _r("A"), _r("B"), _r("C"), _r("D")

    # circularity prong
    if rB or rC:
        circ = "NOT_CIRCULAR"
        circ_msg = (f"the ZERO-ablation (B restores={rB}) or RESAMPLE-ablation (C restores={rC}) STILL "
                    f"restores neutral -- removing the cave component restores even WITHOUT installing the "
                    f"neutral-mean value.")
    else:
        circ = "CIRCULAR"
        circ_msg = (f"only the neutral-mean substitution (A restores={rA}) restores; neither the "
                    f"zero-ablation (B) nor the resample-ablation (C) restores -- the 'restore' is the "
                    f"installed neutral-mean value, not the removal of the cave component.")

    # specificity prong
    if not rA:
        spec = "NO_RESTORE"
        spec_msg = (f"u_cave (A) itself does not restore neutral (restore_A={rA}); no carrier to harden "
                    f"against the in-shift control (D restores={rD}).")
    elif not rD:
        spec = "SPECIFIC_CARRIER"
        spec_msg = (f"u_cave (A restores={rA}) restores neutral while the orthogonalized in-shift direction "
                    f"(D restores={rD}) does NOT -- u_cave is special among in-shift directions.")
    else:
        spec = "SHARED_BY_SHIFT"
        spec_msg = (f"u_cave (A) AND the orthogonalized in-shift direction (D) both restore neutral -- the "
                    f"restore is shared by the shift, u_cave is one slice of a higher-rank shift.")

    hardened = (circ == "NOT_CIRCULAR") and (spec == "SPECIFIC_CARRIER")
    category = "HARDENED_CARRIER" if hardened else f"{circ}+{spec}"
    return {"category": category, "n_fit": n_fit,
            "restore_A": rA, "restore_B": rB, "restore_C": rC, "restore_D": rD,
            "circularity": circ, "specificity": spec, "hardened_carrier": hardened,
            "msg": f"{circ_msg} {spec_msg} -> "
                   + ("HARDENED_CARRIER (NOT_CIRCULAR and SPECIFIC_CARRIER)." if hardened
                      else f"NOT a hardened carrier ({circ}, {spec}).")}


# --------------------------------------------------------------------------- real-run helpers
def _rname(L):
    """resid_post hook name at layer L (cave_suppress_vs_install._rname / faithful_caving._rname)."""
    return f"blocks.{L}.hook_resid_post"


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
    (cave_suppress_vs_install._readout / faithful_caving._readout.)"""
    return {"argmax": int(P.argmax()), "p_c": float(P[cid]), "p_w": float(P[aid])}


def _proj_edit_hook(u, target_proj):
    """Hook that, at the readout position, sets the resid_post u-projection to target_proj (additive shift
    along u): r += (target_proj - r.u) * u. `u` must be on the model device. The projection-edit ablation
    of cave_suppress_vs_install / headset_direction / cave_direction_heldout / faithful_caving. Used for ALL
    five conditions (A target = neutral mean; B target = 0; C target = a shuffled neutral value;
    D/E target = the control direction's own neutral mean)."""
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
    + first-token M. First-token-collision items (cid==aid) skipped. Forward-only. (Structure follows
    cave_suppress_vs_install._collect / faithful_caving._collect exactly.)"""
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
    """Headline layer = the fit layer with the largest IN-SAMPLE u_cave cave-necessity over the fit-set,
    the same headline-selection rule as cave_suppress_vs_install._headline_layer / faithful_caving. For
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


def _cond_numbers(rows_buckets, dPW, kls, kl2_list):
    """Aggregate one condition's per-item readouts into the reported numbers. rows_buckets is a list of
    bool pairs (is_neutral_argmax, is_wstar_argmax); dPW the per-item P_cond(W*)-P2(W*); kls the per-item
    KL(P_cond||P1); kl2_list the per-item KL(P2||P1) (the shared counter baseline). Returns the dict the
    decision consumes (frac_neutral / frac_wstar / dP_Wstar / kl_cond / kl_counter). Pure."""
    n = len(rows_buckets)
    frac_neutral = (sum(1 for b in rows_buckets if b[0]) / n) if n else None
    frac_wstar = (sum(1 for b in rows_buckets if b[1]) / n) if n else None
    return {"n": n,
            "frac_neutral": (round(frac_neutral, 4) if frac_neutral is not None else None),
            "frac_wstar": (round(frac_wstar, 4) if frac_wstar is not None else None),
            "dP_Wstar": (round(statistics.mean(dPW), 6) if dPW else None),
            "kl_cond": (round(statistics.mean(kls), 6) if kls else None),
            "kl_counter": (round(statistics.mean(kl2_list), 6) if kl2_list else None)}


def _resample_perm(n, seed=RESAMPLE_SEED):
    """Deterministic derangement-ish permutation of range(n): a mapping where no index maps to itself
    (so the RESAMPLE ablation installs ANOTHER item's neutral projection, never its own). n==1 -> [0]
    (degenerate; the single-item case maps to itself, logged by the caller). Pure."""
    import random as _r
    if n <= 1:
        return [0] if n == 1 else []
    perm = list(range(n))
    rng = _r.Random(seed)
    for _try in range(64):
        rng.shuffle(perm)
        if all(perm[j] != j for j in range(n)):
            return perm
    return [(j + 1) % n for j in range(n)]   # last-resort rotate-by-one (n>=2 -> all differ)


def _measure_model(name, is_chat, device, pool, fit_layers):
    """One model end-to-end. Collect realized + M readouts under neutral/counter on the wide pool; restrict
    to the argmax-W* caving items; fit u_cave HELD-OUT (TRAIN fold) at the headline layer; build u_orth
    (in-shift, orthogonalized) and a matched random direction; on the TEST fold read the realized softmax
    under conditions A-E and aggregate the per-condition numbers + the de-confound decision. Forward-only."""
    from transformer_lens import HookedTransformer
    from rlhf_differential import MIN_EFFECT_NET
    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    recs = _collect(model, pool, device, is_chat, fit_layers)
    n = len(recs)

    # CAVING items: counter lowers M from neutral by >= MIN_EFFECT_NET (same gate as cave_suppress_vs_install).
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
        out["conditions"] = None
        out["in_shift_info"] = None
        out["decision"] = decide_carrier({}, n_fit=len(argmaxW))
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
        out["conditions"] = None
        out["in_shift_info"] = None
        out["decision"] = decide_carrier({}, n_fit=len(argmaxW))
        return out

    L = headline
    u_cave, proj_n = _u_cave(recs, train, L, device)

    # --- in-shift orthogonalized direction u_orth from the TRAIN (rc - rn) stack at the headline layer ---
    Dtr = torch.stack([(recs[k]["rc"][L] - recs[k]["rn"][L]) for k in train]).to(device).float()
    u_orth, in_shift_info = in_shift_direction(Dtr, u_cave)
    # variance fractions of u_cave and u_orth IN the (uncentered) shift, so "in-shift" is interpretable.
    in_shift_info["var_frac_u_cave"] = round(variance_fraction(Dtr, u_cave), 6)
    in_shift_info["var_frac_u_orth"] = (round(variance_fraction(Dtr, u_orth), 6) if u_orth is not None
                                        else None)
    proj_n_orth = (statistics.mean(float(recs[k]["rn"][L].to(device) @ u_orth) for k in train)
                   if u_orth is not None else None)

    # --- matched isotropic-random rank-1 direction (the existing floor) ---
    g = torch.Generator(device="cpu").manual_seed(RAND_SEED)
    rnd = torch.randn(u_cave.shape, generator=g).to(u_cave.dtype).to(device)
    u_rand = unit(rnd)
    proj_n_rand = statistics.mean(float(recs[k]["rn"][L].to(device) @ u_rand) for k in train)

    # --- per-test-item NEUTRAL u_cave projections, for the RESAMPLE ablation (C) ---
    test_neutral_projs = [float(recs[k]["rn"][L].to(device) @ u_cave) for k in test]
    perm = _resample_perm(len(test), RESAMPLE_SEED)

    # accumulators per condition
    keys = ("A", "B", "C", "D", "E")
    bk = {k: [] for k in keys}      # list of (is_neutral_argmax, is_wstar_argmax)
    dPW = {k: [] for k in keys}
    kls = {k: [] for k in keys}
    kl2_list = []                   # shared counter baseline KL(P2||P1)
    rows = []
    for jpos, k in enumerate(test):
        r = recs[k]
        cid, aid = r["cid"], r["aid"]
        P1 = r["P1"].to(device)                                   # NEUTRAL (cached)
        P2 = r["P2"].to(device)                                   # COUNTER (cached)
        neu_argmax = r["neu"]["argmax"]
        kl2 = kl_div(P2, P1)
        kl2_list.append(kl2)

        # condition target projections
        targets = {
            "A": (u_cave, proj_n),                               # u_cave -> train-neutral mean
            "B": (u_cave, 0.0),                                  # u_cave -> 0 (zero-ablation)
            "C": (u_cave, test_neutral_projs[perm[jpos]]),       # u_cave -> shuffled neutral value
        }
        if u_orth is not None:
            targets["D"] = (u_orth, proj_n_orth)                 # u_orth -> its own train-neutral mean
        targets["E"] = (u_rand, proj_n_rand)                     # random -> its own train-neutral mean

        row = {"i": r["i"], "q": r["q"], "cid": cid, "aid": aid, "neu_argmax": neu_argmax,
               "ctr_argmax": r["ctr"]["argmax"], "kl_counter": round(kl2, 6),
               "P2_W": round(float(P2[aid]), 6)}
        for cond in keys:
            if cond not in targets:                              # D missing when u_orth couldn't be built
                continue
            u, tgt = targets[cond]
            h = [(_rname(L), _proj_edit_hook(u, tgt))]
            Pc = _full_softmax(_logits(model, r["counter"], hooks=h))
            amx = int(Pc.argmax())
            is_neu = (amx == neu_argmax)
            is_w = (amx == aid)
            bk[cond].append((is_neu, is_w))
            dPW[cond].append(float(Pc[aid]) - float(P2[aid]))
            kls[cond].append(kl_div(Pc, P1))
            row[f"amx_{cond}"] = amx
            row[f"isneu_{cond}"] = is_neu
            row[f"kl_{cond}"] = round(kls[cond][-1], 6)
        rows.append(row)
        print(f"  [{'it' if is_chat else 'base'} L{L}] item {r['i']} amxA={row.get('amx_A')} "
              f"isneuA={row.get('isneu_A')} amxB={row.get('amx_B')} isneuB={row.get('isneu_B')} "
              f"amxD={row.get('amx_D')} isneuD={row.get('isneu_D')} KL2={kl2:.4f}", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    conditions = {}
    for cond in keys:
        conditions[cond] = (_cond_numbers(bk[cond], dPW[cond], kls[cond], kl2_list) if bk[cond] else None)

    out["headline_layer"] = L
    out["n_train"] = len(train)
    out["n_test"] = len(test)
    out["neutral_mean_u_cave_proj"] = round(proj_n, 6)        # vs 0 (zero-vs-neutral-mean interpretable)
    out["in_shift_info"] = in_shift_info
    out["conditions"] = conditions
    out["rows"] = rows
    # decision consumes only conditions A..D (E is the reported floor; not part of the de-confound logic)
    cond_for_decision = {c: (conditions.get(c) or {}) for c in ("A", "B", "C", "D")}
    out["decision"] = decide_carrier(cond_for_decision, n_fit=len(argmaxW))
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
        "cue": "cave_carrier_deconfound", "pool_size": len(pool), "fit_layers": fit_layers,
        "metric": ("on argmax-W* caving items, the realized next-token distribution under five projection-"
                   "edit ablations at the headline-layer resid_post: (A) u_cave->train-neutral mean; (B) "
                   "u_cave->0; (C) u_cave->shuffled-neutral; (D) u_orth (orthogonalized in-shift)->its "
                   "neutral mean; (E) isotropic-random->its neutral mean. Per condition: frac(argmax=="
                   "neutral-argmax), frac(argmax==W*), dP(W*), KL(P_cond||P1) vs KL(P2||P1)"),
        "thresholds": {"THR": THR, "SPLIT_SEED": SPLIT_SEED, "RAND_SEED": RAND_SEED,
                       "RESAMPLE_SEED": RESAMPLE_SEED, "MIN_FIT": MIN_FIT, "ORTH_EPS": ORTH_EPS,
                       "ORTH_DOT_TOL": ORTH_DOT_TOL},
        "decision_rule": DECISION_RULE,
        "base": res["base"], "it": res["it"],
    }
    Path("out").mkdir(exist_ok=True)
    Path(f"out/cave_carrier_deconfound_{tag}.json").write_text(json.dumps(out, indent=2, default=str))
    for m in MODELS:
        r = res[m]
        dd = r["decision"]
        cnd = r.get("conditions") or {}

        def _fn(c):
            cc = cnd.get(c) or {}
            return cc.get("frac_neutral"), cc.get("kl_cond")
        fa, ka = _fn("A"); fb, kb = _fn("B"); fc, kc = _fn("C"); fdd, kd = _fn("D"); fe, ke = _fn("E")
        print(f"[{m}] {dd['category']} (L{r.get('headline_layer')}) n_argmaxW={r['n_argmaxW_cave']} "
              f"circ={dd.get('circularity')} spec={dd.get('specificity')} "
              f"| A(frac_neu={fa},KL={ka}) B(={fb},{kb}) C(={fc},{kc}) D(={fdd},{kd}) E(={fe},{ke})",
              flush=True)
    print(f"[done] wrote out/cave_carrier_deconfound_{tag}.json", flush=True)


# --------------------------------------------------------------------------- selftest (model-free, CPU, no model load)
def _dist(V, masses):
    """Probability vector of length V from a {idx: mass} dict; remaining mass spread uniformly over the
    rest. Pure (selftest only). (cave_suppress_vs_install._dist.)"""
    rest = [j for j in range(V) if j not in masses]
    q = torch.zeros(V)
    spec = 0.0
    for idx, m in masses.items():
        q[idx] = m
        spec += m
    if rest:
        q[rest] = (1.0 - spec) / len(rest)
    return q


def _synth_readout(resid, u_cave, u_other, cid, aid, na, V, regime, mu_cave):
    """Synthetic realized next-token softmax for a residual under one of three planted regimes. The argmax
    is driven by where the residual sits relative to the NEUTRAL state. Pure (selftest only).

      pc = resid . u_cave  (the cave coordinate);  po = resid . u_other  (an orthogonal in-shift coord).
      `mu_cave` is the TRAIN-neutral mean of pc (the value condition A installs).

      regime='cavecoord' : the answer depends ONLY on the u_cave coordinate, via a THRESHOLD: high pc (the
          caved counter state) -> argmax W*; pc dropped below the threshold -> argmax the NEUTRAL token na.
          BOTH the zero-ablation (pc->0) AND the neutral-mean substitution (pc->mu_cave) drop pc below the
          threshold and restore na, AND any per-item resampled neutral value also sits below it (the
          neutral values cluster low), so C restores too; a u_other (orthogonal in-shift) ablation never
          touches pc -> does NOTHING. -> NOT_CIRCULAR + SPECIFIC_CARRIER.
      regime='exactmean' : the answer depends on pc being EXACTLY at the train-neutral mean mu_cave. Only
          |pc - mu_cave| < tol restores na; pc->0 (zero-ablation) and pc->a-DIFFERENT-item's-neutral-value
          (resample, off by > tol) stay on W*; only A (which installs exactly mu_cave) restores.
          -> CIRCULAR.
      regime='wholeshift': the answer depends on the WHOLE shift -- bringing EITHER the u_cave coord OR the
          u_other coord back near its neutral value (below a threshold) restores na. So u_orth (D) also
          restores. -> SHARED_BY_SHIFT."""
    pc = float(resid @ u_cave)
    po = float(resid @ u_other)
    logits = torch.full((V,), -30.0)
    if regime == "cavecoord":
        thresh = 1.5
        win = pc <= thresh
    elif regime == "exactmean":
        tol = 0.15
        win = abs(pc - mu_cave) < tol
    else:  # wholeshift
        thresh, thresh_o = 1.5, 1.5
        win = (pc <= thresh) or (po <= thresh_o)
    if win:
        logits[na] = 5.0; logits[aid] = 0.0
    else:
        logits[aid] = 5.0; logits[na] = 0.0
    logits[cid] = -8.0                              # C never the argmax here (the cave goes to na, not C)
    return torch.softmax(logits.float(), -1)


def _run_conditions_synth(test_resids, test_neutral_projs, u_cave, u_other, proj_n_cave, proj_n_other,
                          proj_n_rand, u_rand, cid, aid, na, V, regime, P1):
    """Run the FIVE conditions on synthetic residuals exactly as _measure_model does (minus the model
    forward; the ablated 'forward' is the planted _synth_readout). Returns the conditions dict the decision
    consumes. Pure (selftest only). test_resids is a list of 1-D counter residual tensors; test_neutral_projs
    the per-item NEUTRAL u_cave projection (distinct per item, for a genuine RESAMPLE); P1 the shared NEUTRAL
    distribution (its argmax == na). proj_n_cave is the TRAIN-neutral mean (the value condition A installs)."""
    keys = ("A", "B", "C", "D", "E")
    bk = {k: [] for k in keys}
    dPW = {k: [] for k in keys}
    kls = {k: [] for k in keys}
    kl2_list = []
    perm = _resample_perm(len(test_resids), RESAMPLE_SEED)

    def ablate(resid, u, target):
        return resid + (target - float(resid @ u)) * u

    neu_argmax = int(P1.argmax())                  # == na
    for jpos, rc in enumerate(test_resids):
        P2 = _synth_readout(rc, u_cave, u_other, cid, aid, na, V, regime, proj_n_cave)
        kl2 = kl_div(P2, P1); kl2_list.append(kl2)
        targets = {
            "A": (u_cave, proj_n_cave),                          # train-neutral mean (what A installs)
            "B": (u_cave, 0.0),                                  # zero
            "C": (u_cave, test_neutral_projs[perm[jpos]]),       # ANOTHER item's neutral value
            "D": (u_other, proj_n_other),
            "E": (u_rand, proj_n_rand),
        }
        for cond, (u, tgt) in targets.items():
            ra = ablate(rc, u, tgt)
            Pc = _synth_readout(ra, u_cave, u_other, cid, aid, na, V, regime, proj_n_cave)
            amx = int(Pc.argmax())
            bk[cond].append((amx == neu_argmax, amx == aid))
            dPW[cond].append(float(Pc[aid]) - float(P2[aid]))
            kls[cond].append(kl_div(Pc, P1))
    return {c: _cond_numbers(bk[c], dPW[c], kls[c], kl2_list) for c in keys}


def selftest():
    torch.manual_seed(0)
    d, V = 64, 200
    cid, aid, na = 3, 7, 17        # C-first-tok, W*-first-tok, the NEUTRAL-condition argmax token

    # ---------- restores_neutral predicate ----------
    assert restores_neutral(0.6, 0.1, 0.9) is True            # majority neutral argmax AND KL toward neutral
    assert restores_neutral(0.6, 0.9, 0.8) is False           # KL not reduced
    assert restores_neutral(0.4, 0.1, 0.9) is False           # below THR
    assert restores_neutral(THR, 0.1, 0.9) is True            # exactly THR -> restores (>=)
    assert restores_neutral(None, 0.1, 0.9) is False
    print("[selftest] restores_neutral predicate (THR boundary, KL gate, None) OK")

    # ---------- KL math ----------
    P1 = _dist(V, {na: 0.6, cid: 0.1})
    assert abs(kl_div(P1, P1)) < 1e-6
    P2 = _dist(V, {aid: 0.6, cid: 0.05})
    P3 = _dist(V, {na: 0.58, cid: 0.1})
    assert kl_div(P3, P1) < kl_div(P2, P1) and kl_div(P2, P1) > 0
    print(f"[selftest] kl_div identical=0, restore KL={kl_div(P3, P1):.4f} < counter KL={kl_div(P2, P1):.4f}")

    # ---------- split + resample-perm primitives ----------
    tr, te = split_indices(10, SPLIT_SEED)
    assert set(tr) | set(te) == set(range(10)) and not (set(tr) & set(te)) and tr and te
    assert split_indices(10, SPLIT_SEED) == split_indices(10, SPLIT_SEED)
    rp = _resample_perm(8, RESAMPLE_SEED)
    assert sorted(rp) == list(range(8)) and all(rp[j] != j for j in range(8)), rp   # derangement (no fixed pt)
    assert _resample_perm(8) == _resample_perm(8)                                   # deterministic
    print(f"[selftest] split_indices train={tr} test={te}; resample_perm derangement={rp}")

    # ---------- SVD in-shift direction: orthogonal to u_cave, in-shift, high variance ----------
    # Build a shift stack whose mean is along u_cave (so diff-of-means recovers u_cave) but whose per-item
    # VARIATION is dominated by a second orthogonal direction u_planted (the in-shift PC). The centered-SVD
    # top PC should then be ~u_planted; Gram-Schmidt against u_cave leaves it ~unchanged (already orthogonal).
    g = torch.Generator().manual_seed(7)
    u_cave_t = unit(torch.randn(d, generator=g))
    u_planted = torch.randn(d, generator=g)
    u_planted = unit(u_planted - (u_planted @ u_cave_t) * u_cave_t)     # exactly orthogonal to u_cave
    n = 20
    a_mean = 3.0
    stack = []
    for i in range(n):
        coef = float(torch.randn(1, generator=g)) * 4.0                # large per-item variation along u_planted
        small = 0.05 * torch.randn(d, generator=g)                     # tiny isotropic noise
        stack.append(a_mean * u_cave_t + coef * u_planted + small)
    D = torch.stack(stack)
    u_cave_fit = unit(D.mean(0))                                       # diff-of-means recovers u_cave
    assert float(torch.nn.functional.cosine_similarity(u_cave_fit, u_cave_t, dim=0)) > 0.9
    u_orth, info = in_shift_direction(D, u_cave_fit)
    assert u_orth is not None and abs(info["u_orth_dot_u_cave"]) < ORTH_DOT_TOL, info
    cos_op = abs(float(torch.nn.functional.cosine_similarity(u_orth, u_planted, dim=0)))
    assert cos_op > 0.9, f"u_orth should align with the planted in-shift PC: cos={cos_op}"
    vf_orth = variance_fraction(D, u_orth)
    vf_cave = variance_fraction(D, u_cave_fit)
    assert vf_orth > 0.3, (vf_orth, info)             # the planted in-shift PC dominates the variation
    assert vf_cave > 1.0 / d, vf_cave                 # u_cave carries more than a generic-direction share
    print(f"[selftest] in_shift_direction: |u_orth.u_cave|={abs(info['u_orth_dot_u_cave']):.2e} "
          f"cos(u_orth,planted)={cos_op:.3f} var_frac(u_orth)={vf_orth:.3f} var_frac(u_cave)={vf_cave:.3f}")
    # fallback: when ALL variation is along u_cave, every PC's GS residual collapses -> u_orth is None
    D_caveonly = torch.stack([a_mean * u_cave_t + 0.5 * float(torch.randn(1, generator=g)) * u_cave_t
                              for _ in range(n)])
    uo2, info2 = in_shift_direction(D_caveonly, u_cave_t)
    assert uo2 is None or abs(info2["u_orth_dot_u_cave"]) < ORTH_DOT_TOL, info2
    print(f"[selftest] in_shift_direction fallback (cave-only variation) -> u_orth={uo2 is not None}")

    # ---------- variance_fraction sanity ----------
    Xa = torch.stack([3.0 * u_cave_t, -2.0 * u_cave_t, 1.5 * u_cave_t])    # all along u_cave
    assert variance_fraction(Xa, u_cave_t) > 0.999
    assert variance_fraction(Xa, u_planted) < 1e-6                          # orthogonal -> 0
    print("[selftest] variance_fraction: along=1.0, orthogonal=0.0 OK")

    # ============================================================ DECISION-BOUNDARY scenarios =============
    # direct decide_carrier checks on hand-built condition numbers (KL2 = 0.9 counter baseline throughout)
    def C(frac_neutral, kl_cond, kl2=0.9):
        return {"frac_neutral": frac_neutral, "kl_cond": kl_cond, "kl_counter": kl2}

    # HARDENED: B restores (NOT_CIRCULAR) and A restores, D does not (SPECIFIC) -> HARDENED_CARRIER
    dh = decide_carrier({"A": C(0.8, 0.1), "B": C(0.7, 0.15), "C": C(0.0, 0.95), "D": C(0.0, 0.95)}, n_fit=8)
    assert dh["category"] == "HARDENED_CARRIER" and dh["hardened_carrier"], dh
    assert dh["circularity"] == "NOT_CIRCULAR" and dh["specificity"] == "SPECIFIC_CARRIER", dh
    print(f"[selftest] decide HARDENED_CARRIER (B restores, D does not) -> {dh['category']}")

    # CIRCULAR: only A restores (B and C do not) -> CIRCULAR; A restores, D does not -> SPECIFIC
    dc = decide_carrier({"A": C(0.8, 0.1), "B": C(0.0, 0.95), "C": C(0.0, 0.95), "D": C(0.0, 0.95)}, n_fit=8)
    assert dc["circularity"] == "CIRCULAR" and dc["specificity"] == "SPECIFIC_CARRIER", dc
    assert dc["category"] == "CIRCULAR+SPECIFIC_CARRIER" and not dc["hardened_carrier"], dc
    print(f"[selftest] decide CIRCULAR (only A restores) -> {dc['category']}")

    # SHARED_BY_SHIFT: A and D both restore; B restores too (NOT_CIRCULAR) -> NOT hardened (D restores)
    ds = decide_carrier({"A": C(0.8, 0.1), "B": C(0.7, 0.15), "C": C(0.0, 0.95), "D": C(0.8, 0.1)}, n_fit=8)
    assert ds["circularity"] == "NOT_CIRCULAR" and ds["specificity"] == "SHARED_BY_SHIFT", ds
    assert not ds["hardened_carrier"], ds
    print(f"[selftest] decide SHARED_BY_SHIFT (D also restores) -> {ds['category']}")

    # NO_RESTORE: A does not restore -> NO_RESTORE specificity regardless of D
    dn = decide_carrier({"A": C(0.1, 0.95), "B": C(0.0, 0.95), "C": C(0.0, 0.95), "D": C(0.0, 0.95)}, n_fit=8)
    assert dn["specificity"] == "NO_RESTORE" and not dn["hardened_carrier"], dn
    print(f"[selftest] decide NO_RESTORE (A does not restore) -> {dn['category']}")

    # INSUFFICIENT below MIN_FIT
    di = decide_carrier({}, n_fit=1)
    assert di["category"] == "INSUFFICIENT" and not di["hardened_carrier"], di
    print(f"[selftest] decide INSUFFICIENT (n_fit<MIN_FIT) -> {di['category']}")

    # ============================================================ END-TO-END synthetic regimes ===========
    # Build counter residuals: pc (u_cave coord) ~ a_cave (caved), po (in-shift orth coord) ~ a_o (caved).
    # The TRAIN-neutral mean of pc is mu_cave (the value A installs). Per-item NEUTRAL pc values are DISTINCT
    # and spread away from mu_cave by more than the exactmean tolerance, so the RESAMPLE (C) installs a value
    # that is at the neutral mean only in cavecoord (threshold) but NOT at mu_cave (so C does not restore in
    # exactmean). The matched-random direction is orthogonal to both u_c and u_o (E does nothing).
    ge = torch.Generator().manual_seed(11)
    u_c = unit(torch.randn(d, generator=ge))
    u_o = torch.randn(d, generator=ge)
    u_o = unit(u_o - (u_o @ u_c) * u_c)                  # orthogonal in-shift control direction
    u_rand = torch.randn(d, generator=torch.Generator().manual_seed(RAND_SEED + 99))
    u_rand = unit(u_rand - (u_rand @ u_c) * u_c)         # make E truly orthogonal to u_c (cannot move pc)
    u_rand = unit(u_rand - (u_rand @ u_o) * u_o)         # ... and to u_o (cannot move po)
    a_cave, mu_cave = 3.0, 0.3                            # counter pc ~ a_cave; train-neutral mean of pc = mu_cave
    a_o, mu_o = 3.0, 0.0                                  # counter po ~ a_o;   neutral mean of po = mu_o
    proj_n_cave, proj_n_other, proj_n_rand = mu_cave, mu_o, 0.0
    nT = 8
    test_resids, test_neutral_projs = [], []
    for i in range(nT):
        perp = 0.02 * torch.randn(d, generator=ge)
        perp = perp - (perp @ u_c) * u_c - (perp @ u_o) * u_o   # keep pc/po clean for the planted readouts
        test_resids.append(a_cave * u_c + a_o * u_o + perp)
        # distinct per-item neutral pc value: low (below the cavecoord threshold) but OFF mu_cave by > tol
        test_neutral_projs.append(0.8 + 0.1 * i)               # 0.8..1.5, all > mu_cave+tol and < thresh
    P1 = _dist(V, {na: 0.6, cid: 0.1, aid: 0.05})               # NEUTRAL argmax == na

    # (i) cavecoord -> pc-threshold readout: zero-ablation AND neutral-mean AND resample (all drop pc below
    #     the threshold) restore na; u_orth (D) only moves po -> does NOT restore. -> HARDENED_CARRIER.
    cond_i = _run_conditions_synth(test_resids, test_neutral_projs, u_c, u_o, proj_n_cave, proj_n_other,
                                   proj_n_rand, u_rand, cid, aid, na, V, "cavecoord", P1)
    dec_i = decide_carrier({c: cond_i[c] for c in ("A", "B", "C", "D")}, n_fit=nT)
    assert dec_i["category"] == "HARDENED_CARRIER", (dec_i, cond_i)
    assert cond_i["A"]["frac_neutral"] == 1.0 and cond_i["B"]["frac_neutral"] == 1.0, cond_i
    assert cond_i["C"]["frac_neutral"] == 1.0, cond_i          # resample also drops pc below the threshold
    assert cond_i["D"]["frac_neutral"] == 0.0, cond_i          # in-shift orth direction does NOT restore
    assert cond_i["E"]["frac_neutral"] == 0.0, cond_i          # random floor does nothing
    print(f"[selftest] (i) cavecoord -> {dec_i['category']} "
          f"[A={cond_i['A']['frac_neutral']} B={cond_i['B']['frac_neutral']} "
          f"C={cond_i['C']['frac_neutral']} D={cond_i['D']['frac_neutral']} E={cond_i['E']['frac_neutral']}]")

    # (ii) exactmean -> only pc EXACTLY at mu_cave restores. A installs mu_cave -> restores; zero (B) and the
    #      resampled OTHER-item neutral value (C, off by > tol) do NOT restore. -> CIRCULAR (+SPECIFIC).
    cond_ii = _run_conditions_synth(test_resids, test_neutral_projs, u_c, u_o, proj_n_cave, proj_n_other,
                                    proj_n_rand, u_rand, cid, aid, na, V, "exactmean", P1)
    dec_ii = decide_carrier({c: cond_ii[c] for c in ("A", "B", "C", "D")}, n_fit=nT)
    assert cond_ii["A"]["frac_neutral"] == 1.0, cond_ii        # A installs exactly mu_cave -> restores
    assert cond_ii["B"]["frac_neutral"] == 0.0, cond_ii        # zero != mu_cave -> does not restore
    assert cond_ii["C"]["frac_neutral"] == 0.0, cond_ii        # resampled OTHER value != mu_cave -> no restore
    assert dec_ii["circularity"] == "CIRCULAR" and not dec_ii["hardened_carrier"], (dec_ii, cond_ii)
    print(f"[selftest] (ii) exactmean -> {dec_ii['category']} "
          f"[A={cond_ii['A']['frac_neutral']} B={cond_ii['B']['frac_neutral']} "
          f"C={cond_ii['C']['frac_neutral']} D={cond_ii['D']['frac_neutral']}]")

    # (iii) wholeshift -> bringing EITHER pc OR po below its threshold restores na; A/B/C move pc, D moves po
    #       (a_o=3 > thresh on counter; D sets po to mu_o=0 < thresh) -> A AND D both restore.
    #       -> SHARED_BY_SHIFT (not hardened).
    cond_iii = _run_conditions_synth(test_resids, test_neutral_projs, u_c, u_o, proj_n_cave, proj_n_other,
                                     proj_n_rand, u_rand, cid, aid, na, V, "wholeshift", P1)
    dec_iii = decide_carrier({c: cond_iii[c] for c in ("A", "B", "C", "D")}, n_fit=nT)
    assert cond_iii["A"]["frac_neutral"] == 1.0 and cond_iii["D"]["frac_neutral"] == 1.0, cond_iii
    assert dec_iii["specificity"] == "SHARED_BY_SHIFT" and not dec_iii["hardened_carrier"], (dec_iii, cond_iii)
    print(f"[selftest] (iii) wholeshift -> {dec_iii['category']} "
          f"[A={cond_iii['A']['frac_neutral']} D={cond_iii['D']['frac_neutral']}]")

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
