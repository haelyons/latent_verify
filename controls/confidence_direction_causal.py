"""STRONGER-CONSTRUCTION search for a CAUSAL confidence/margin DIRECTION (sibling of
confidence_vs_cave_direction.py / cave_direction_heldout.py).

CONTEXT (neutral). An earlier control (confidence_vs_cave_direction.py) fit a confidence/margin direction
by a MEDIAN split of |M| at the neutral condition and measured its held-out necessity. That construction
returned near-zero held-out necessity (the median-split confidence axis did not causally mediate the
margin). The median split is the WEAKEST contrast (the two halves straddle the median, so the diff-of-means
vector mixes near-median items into both ends). This control asks whether ANY STRONGER construction of a
confidence axis is held-out CAUSAL, and -- only if one is -- how it relates to the cave/defer direction.

It changes the FIT CONTRAST, not the measurement: same diff-of-means construction, same projection-edit
ablation/steer, same matched-magnitude random control, same metric M = logp(C) - logp(W*), same FIT_LAYERS
sweep, same train/test split machinery -- all reused from confidence_vs_cave_direction / cave_direction_heldout.

CANDIDATE CONFIDENCE DIRECTIONS (each fit by diff-of-means on the NEUTRAL-condition last-token resid_post,
on a TRAIN fold; evaluated held-out on a disjoint TEST fold), over the misconception_pool WIDE items:
  (a) MARGIN_QUARTILE : top-quartile |M| vs bottom-quartile |M| at the neutral condition (extreme contrast,
      NOT the median split -- the near-median items are dropped, sharpening the axis).
  (b) ENTROPY_QUARTILE: top-quartile vs bottom-quartile output ENTROPY of the next-token distribution at the
      answer position under the neutral prompt (a model-internal confidence proxy independent of which two
      tokens C/W* are scored).
The signal each candidate's necessity/sufficiency is read on is its OWN definition's quantity: |M| for the
margin candidate, entropy for the entropy candidate (the "gap" is high-quartile minus low-quartile of that
quantity), so each axis is tested against the confidence variable it was fit to move.

PER MODEL (base, it), AT EVERY FIT_LAYER, held-out:
  CONF NECESSITY (ablate): on TEST high-confidence items set the u_conf-projection to the TRAIN
      low-confidence mean projection -> fraction of the (high - low) signal gap that collapses.
  CONF SUFFICIENCY (steer): on TEST low-confidence items set the u_conf-projection to the TRAIN
      high-confidence mean projection -> induced fraction of the gap.
  RANDOM control: a matched-magnitude random unit direction, same edit -> must NOT move the signal.
  BEST candidate = the (definition, layer) with the largest held-out necessity whose random control is clean.
  COSINE: u_cave = diff-of-means(counter - neutral) over caving items, fit held-out (cave_direction_heldout
      construction); cosine(u_cave, u_conf) per layer for the BEST confidence candidate. Each axis is fit on
      its own TRAIN fold so the cosine is between two independently-fit directions.

NEUTRAL DECISION (module constants DIR_THR=0.20, BASE_FLOOR=0.05, COS_THR=0.50; numbers + categories only,
no hypothesis, no statement about base vs -it or about the cosine):
  CAUSAL_CONFIDENCE_DIRECTION iff the best candidate's held-out NECESSITY >= DIR_THR AND its matched
      RANDOM-direction necessity < BASE_FLOOR; else NO_CAUSAL_CONFIDENCE_DIRECTION.
  COS bucket (reported ONLY when a causal confidence direction exists, else flagged N/A):
      HIGH_COS iff |cos(u_cave, u_conf)| >= COS_THR else LOW_COS.
  Reported per model, per definition, per layer. No claim attached to any sign, bucket, or comparison.

Forward-only (diff-of-means + projection edits; no backward) -> fits the 40GB A100. Reuses the verified
pure primitives from confidence_vs_cave_direction (direction math, split, proj-edit hook, |M|/entropy
readouts) and cave_direction_heldout (k-fold splits, diff-of-means fit, bootstrap CI). The only new logic is
the quartile-contrast candidate construction and the per-candidate held-out necessity/sufficiency loop --
pure, covered by the model-free --selftest.

  python controls/confidence_direction_causal.py --selftest
  python controls/confidence_direction_causal.py --device cuda \
    --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

# Repo-internal and sibling-control imports (item pools, turn builders, readout helpers, direction math,
# split/CI machinery, FIT_LAYERS/_rname) are DEFERRED into the functions that use them so --selftest runs
# standalone on CPU with NO model load and nothing else on sys.path. On the box every file (repo modules
# AND controls) is scp'd flat into latent_verify/, where these resolve; locally we put both controls/ and
# the repo root on sys.path. The pure helpers used by --selftest are imported through _pure() below.

# ----------------------------------------------------------------- pre-registered thresholds (neutral)
FIT_LAYERS = [24, 28, 32, 36]   # same sweep as headset_direction / confidence_vs_cave_direction
DIR_THR = 0.20                  # held-out necessity a candidate must mediate to be "causal"
BASE_FLOOR = 0.05               # matched random-direction necessity must fall below this (clean control)
COS_THR = 0.50                  # |cos(u_cave, u_conf)| >= this -> HIGH_COS (the two axes coincide)
SPLIT_SEED = 0                  # deterministic train/test fold assignment (shared with the sibling control)
RAND_SEED = 0                   # deterministic matched random-direction control
QUANTILE = 0.25                 # quartile fraction: top-QUANTILE vs bottom-QUANTILE contrast
DEFINITIONS = ("MARGIN_QUARTILE", "ENTROPY_QUARTILE")
MODELS = ("base", "it")

DECISION_RULE = (
    "For each candidate confidence direction definition d in {MARGIN_QUARTILE, ENTROPY_QUARTILE}: fit "
    "u_conf = diff-of-means(resid_post[L][-1] over TRAIN high-quantile items - over TRAIN low-quantile "
    "items) at the NEUTRAL condition, where the quantile is the top/bottom QUANTILE(0.25) of the signal "
    "(|M|=|logp(C)-logp(W*)| for MARGIN_QUARTILE; next-token entropy at the answer position for "
    "ENTROPY_QUARTILE). Held-out NECESSITY: on TEST high-quantile items set the u_conf-projection to the "
    "TRAIN low-quantile mean -> fraction of the (high-low) signal gap that collapses. SUFFICIENCY: on TEST "
    "low-quantile items set the projection to the TRAIN high-quantile mean -> induced fraction. RANDOM: a "
    "matched-magnitude random unit direction, same edit. BEST = the (definition,layer) with the largest "
    "held-out necessity whose random control < BASE_FLOOR(0.05). "
    "CAUSAL_CONFIDENCE_DIRECTION iff best held-out necessity >= DIR_THR(0.20) AND best random necessity < "
    "BASE_FLOOR(0.05), else NO_CAUSAL_CONFIDENCE_DIRECTION. "
    "u_cave = diff-of-means(counter - neutral) over caving items, fit held-out; cosine(u_cave, u_conf) per "
    "layer for the BEST candidate. COS bucket (reported only when a causal confidence direction exists, "
    "else N/A): HIGH_COS iff |cos(u_cave,u_conf)| >= COS_THR(0.50) else LOW_COS. Reported per model, per "
    "definition, per layer; numbers + categories only, no claim attached to any sign, bucket, or the "
    "base-vs-it comparison."
)


# ----------------------------------------------------------------- pure helper resolution (shared)
def _pure():
    """Import the reused PURE helpers from the sibling controls. Deferred + sys.path-guarded so it resolves
    both locally (controls/ + repo root on path) and on the box (everything flat in latent_verify/). No
    model load, no torch device work -- safe inside --selftest. Returns a small namespace dict."""
    here = Path(__file__).resolve().parent          # .../controls
    for p in (str(here), str(here.parent)):         # controls/ for siblings; repo root for repo modules
        if p not in sys.path:
            sys.path.insert(0, p)
    from confidence_vs_cave_direction import (
        unit, diff_of_means, cosine, split_indices, frac_moved, _proj_edit_hook)
    from cave_direction_heldout import kfold_splits, paired_bootstrap_ci
    from entropy_neuron_gemma2 import entropy_of_logits
    return {"unit": unit, "diff_of_means": diff_of_means, "cosine": cosine,
            "split_indices": split_indices, "frac_moved": frac_moved,
            "_proj_edit_hook": _proj_edit_hook, "kfold_splits": kfold_splits,
            "paired_bootstrap_ci": paired_bootstrap_ci, "entropy_of_logits": entropy_of_logits}


# ----------------------------------------------------------------- pure quantile contrast
def quantile_split(values, q=QUANTILE):
    """Stratify item positions into a TOP-q and a BOTTOM-q contrast by SIGNED value (callers pass already
    sign-resolved confidence signals: |M| for the margin axis, entropy for the entropy axis -- both
    larger == more of the signal). Returns (high_idx, low_idx) over positions 0..len-1, sorted; the
    near-median middle (1-2q fraction) is DROPPED so the diff-of-means contrast is sharp. Pure.
    k = min(max(1, floor(q*n)), n//2); for n>=2 both folds are non-empty and disjoint."""
    n = len(values)
    if n < 2:
        return [], []
    order = sorted(range(n), key=lambda i: values[i])       # ascending
    k = max(1, int(q * n))
    k = min(k, n // 2)                                       # never overlap the two ends
    k = max(1, k)
    low = sorted(order[:k])
    high = sorted(order[n - k:])
    return high, low


# ----------------------------------------------------------------- pure decisions
def decide_causal(best_nec, best_rand, dir_thr=DIR_THR, base_floor=BASE_FLOOR):
    """CAUSAL_CONFIDENCE_DIRECTION iff the best candidate's held-out necessity clears dir_thr AND its
    matched random control is clean (< base_floor); else NO_CAUSAL_CONFIDENCE_DIRECTION. Pure."""
    clean = best_rand is not None and best_rand < base_floor
    fires = best_nec is not None and best_nec >= dir_thr and clean
    return "CAUSAL_CONFIDENCE_DIRECTION" if fires else "NO_CAUSAL_CONFIDENCE_DIRECTION"


def decide_cos(cos_val, causal, cos_thr=COS_THR):
    """COS bucket, reported ONLY when a causal confidence direction exists. When no causal direction exists
    the cosine is meaningless (there is no validated u_conf to compare), so it is flagged N/A. Pure.
      causal True  + |cos| >= cos_thr -> HIGH_COS
      causal True  + |cos| <  cos_thr -> LOW_COS
      causal False                    -> N/A"""
    if not causal:
        return "N/A"
    if cos_val is None:
        return "N/A"
    return "HIGH_COS" if abs(cos_val) >= cos_thr else "LOW_COS"


def model_decision(best_nec, best_rand, best_cos):
    """Assemble the neutral buckets for one model from its best-candidate statistics. Pure (floats -> dict).
    No claim attached; the cosine bucket is only meaningful (else N/A) when a causal direction exists."""
    causal_cat = decide_causal(best_nec, best_rand)
    causal = causal_cat == "CAUSAL_CONFIDENCE_DIRECTION"
    return {
        "causal_bucket": causal_cat,
        "cos_bucket": decide_cos(best_cos, causal),
        "best_nec": (round(best_nec, 4) if best_nec is not None else None),
        "best_rand": (round(best_rand, 4) if best_rand is not None else None),
        "best_cos": (round(best_cos, 4) if (causal and best_cos is not None) else None),
    }


# ----------------------------------------------------------------- residual + signal collection (real)
def _collect(model, pool, device, is_chat, fit_layers, rname, P):
    """One model: per pool item, cache at every fit layer the last-token resid_post under the NEUTRAL and
    COUNTER prompts (verbatim repo turn construction), the neutral and counter margins M = logp(C)-logp(W*),
    and the next-token ENTROPY of the answer-position distribution under the neutral prompt. Forward-only;
    caches only the last-position resid_post per layer + two scalars. First-token-collision items skipped."""
    from rlhf_differential import _helpers, _logp_diff
    from job_truthful_flip import PUSH, NEUTRAL
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    tag = "it" if is_chat else "base"
    recs = []
    for i, it in enumerate(pool):
        q, C, W = it["q"], it["correct"], it["Wstar"]
        cid, aid = first(" " + C), first(" " + W)
        if cid == aid:                                   # first-token collision -> margin meaningless, skip
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
            lg_neu = model.run_with_hooks(neutral, fwd_hooks=[(n, grab_n) for n in names])
            M_neu = float(_logp_diff(lg_neu, cid, aid))
            ent_neu = float(P["entropy_of_logits"](lg_neu[0, -1]))     # answer-position next-token entropy
            M_ctr = float(_logp_diff(
                model.run_with_hooks(counter, fwd_hooks=[(n, grab_c) for n in names]), cid, aid))
        recs.append({"i": i, "q": q, "cid": cid, "aid": aid,
                     "neutral": neutral, "counter": counter,
                     "rn": rn, "rc": rc, "M_neu": M_neu, "M_ctr": M_ctr, "ent_neu": ent_neu})
        print(f"  [{tag}] item {i} M_neu={M_neu:+.2f} ent_neu={ent_neu:.3f} M_ctr={M_ctr:+.2f} "
              f"q={q[:40]!r}", flush=True)
    return recs


def _signal(rec, definition):
    """The confidence SIGNAL a candidate definition is fit/evaluated on, at the NEUTRAL condition. Larger ==
    more confident for both definitions (|M| for margin; entropy NEGATED so larger == lower entropy ==
    more confident -> the high-quantile contrast end is the confident end for both axes). Pure."""
    if definition == "MARGIN_QUARTILE":
        return abs(rec["M_neu"])
    if definition == "ENTROPY_QUARTILE":
        return -rec["ent_neu"]
    raise ValueError(definition)


def _readout(model, ids, cid, aid, definition, P, hooks=None):
    """Read the candidate's confidence signal under an (optional) intervention, matching _signal's sign.
    MARGIN_QUARTILE -> |M| = |logp(C)-logp(W*)|; ENTROPY_QUARTILE -> -entropy(next token). Forward-only."""
    from rlhf_differential import _logp_diff
    with torch.no_grad():
        lg = model.run_with_hooks(ids, fwd_hooks=hooks) if hooks else model(ids)
    if definition == "MARGIN_QUARTILE":
        return abs(float(_logp_diff(lg, cid, aid)))
    return -float(P["entropy_of_logits"](lg[0, -1]))


def _measure_model(name, is_chat, device, fit_layers, rname, pool, P):
    """One model end-to-end: collect residuals + signals on the WIDE pool; for each definition and each
    layer fit u_conf on the TRAIN fold (quartile contrast on that definition's signal), evaluate held-out
    necessity/sufficiency + matched random control on the TEST fold; fit u_cave held-out and compute
    cosine(u_cave, u_conf) per layer; pick the best (definition, layer) by held-out necessity (random clean);
    assemble the neutral decision. Returns a dict."""
    from transformer_lens import HookedTransformer
    from rlhf_differential import MIN_EFFECT_NET
    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    recs = _collect(model, pool, device, is_chat, fit_layers, rname, P)
    n = len(recs)
    g = torch.Generator(device="cpu").manual_seed(RAND_SEED)

    unit, diff_of_means, cosine = P["unit"], P["diff_of_means"], P["cosine"]
    split_indices, frac_moved, proj_edit = P["split_indices"], P["frac_moved"], P["_proj_edit_hook"]

    # caving items (this model): counter lowers the margin from neutral by >= MIN_EFFECT_NET.
    cave_pos = [k for k, r in enumerate(recs) if (r["M_neu"] - r["M_ctr"]) >= MIN_EFFECT_NET]

    out = {"name": name, "n_ok": n, "n_cave": len(cave_pos),
           "definitions": {d: {"layers": {}} for d in DEFINITIONS}}

    # one shared train/test split over ALL items (same SPLIT_SEED as the sibling control), reused across
    # definitions and layers so the held-out comparison is on a fixed fold.
    tr, te = split_indices(n, SPLIT_SEED)

    # ---- u_cave per layer (held-out fit on the TRAIN fold of caving items), for the cosine ----
    u_cave_by_L = {}
    if cave_pos:
        ctr_tr, _ = split_indices(len(cave_pos), SPLIT_SEED)
        cave_tr = [cave_pos[j] for j in ctr_tr]
        for L in fit_layers:
            Rc = torch.stack([recs[k]["rc"][L] for k in cave_tr]).to(device)
            Rn = torch.stack([recs[k]["rn"][L] for k in cave_tr]).to(device)
            u_cave_by_L[L] = unit(diff_of_means(Rc, Rn))

    # ---- per-definition, per-layer candidate fit + held-out necessity/sufficiency/random + cosine ----
    for definition in DEFINITIONS:
        sig = [_signal(r, definition) for r in recs]
        for L in fit_layers:
            sig_tr = [sig[k] for k in tr]
            hi_tr_local, lo_tr_local = quantile_split(sig_tr, QUANTILE)
            hi_tr = [tr[j] for j in hi_tr_local]
            lo_tr = [tr[j] for j in lo_tr_local]
            sig_te = [sig[k] for k in te]
            hi_te_local, lo_te_local = quantile_split(sig_te, QUANTILE)
            hi_te = [te[j] for j in hi_te_local]
            lo_te = [te[j] for j in lo_te_local]
            entry = {"n_train_hi": len(hi_tr), "n_train_lo": len(lo_tr),
                     "n_test_hi": len(hi_te), "n_test_lo": len(lo_te)}
            if not (hi_tr and lo_tr and hi_te and lo_te):
                entry["skipped"] = "degenerate quartile split"
                out["definitions"][definition]["layers"][L] = entry
                continue

            # u_conf: diff-of-means(high-quantile resid - low-quantile resid) at NEUTRAL, on the TRAIN fold.
            Rn_hi = torch.stack([recs[k]["rn"][L] for k in hi_tr]).to(device)
            Rn_lo = torch.stack([recs[k]["rn"][L] for k in lo_tr]).to(device)
            u_conf = unit(diff_of_means(Rn_hi, Rn_lo))
            proj_hi = statistics.mean(float(recs[k]["rn"][L].to(device) @ u_conf) for k in hi_tr)
            proj_lo = statistics.mean(float(recs[k]["rn"][L].to(device) @ u_conf) for k in lo_tr)
            # TRAIN-fold target signal levels (the gap each intervention is scored against).
            sig_hi_tr = statistics.mean(_signal(recs[k], definition) for k in hi_tr)
            sig_lo_tr = statistics.mean(_signal(recs[k], definition) for k in lo_tr)

            # matched-magnitude random unit direction (cpu generator -> residual device/dtype).
            rnd = torch.randn(u_conf.shape, generator=g).to(u_conf.dtype).to(device)
            u_rand = unit(rnd)
            prj_lo_rand = statistics.mean(float(recs[k]["rn"][L].to(device) @ u_rand) for k in lo_tr)

            nec_vals, suf_vals, rand_vals = [], [], []
            # NECESSITY: on TEST high items, set u_conf-projection to TRAIN low mean -> signal collapses?
            for k in hi_te:
                r = recs[k]
                sig_hi = _signal(r, definition)
                h = [(rname(L), proj_edit(u_conf, proj_lo))]
                sig_ab = _readout(model, r["neutral"], r["cid"], r["aid"], definition, P, hooks=h)
                nec_vals.append(frac_moved(sig_hi, sig_ab, sig_lo_tr))
                hr = [(rname(L), proj_edit(u_rand, prj_lo_rand))]
                sig_rr = _readout(model, r["neutral"], r["cid"], r["aid"], definition, P, hooks=hr)
                rand_vals.append(frac_moved(sig_hi, sig_rr, sig_lo_tr))
            # SUFFICIENCY: on TEST low items, set u_conf-projection to TRAIN high mean -> signal induced?
            for k in lo_te:
                r = recs[k]
                sig_lo = _signal(r, definition)
                h = [(rname(L), proj_edit(u_conf, proj_hi))]
                sig_st = _readout(model, r["neutral"], r["cid"], r["aid"], definition, P, hooks=h)
                suf_vals.append(frac_moved(sig_lo, sig_st, sig_hi_tr))

            cos = cosine(u_cave_by_L[L], u_conf) if L in u_cave_by_L else None
            entry.update({
                "frac_nec": round(statistics.mean(nec_vals), 4) if nec_vals else None,
                "frac_suf": round(statistics.mean(suf_vals), 4) if suf_vals else None,
                "rand_nec": round(statistics.mean(rand_vals), 4) if rand_vals else None,
                "cos_cave_conf": (round(cos, 4) if cos is not None else None),
            })
            out["definitions"][definition]["layers"][L] = entry
            print(f"  [{('it' if is_chat else 'base')} {definition} L{L}] "
                  f"nec={entry['frac_nec']} suf={entry['frac_suf']} rand={entry['rand_nec']} "
                  f"cos={entry['cos_cave_conf']}", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- best (definition, layer) by held-out necessity among clean-random candidates ----
    cands = []
    for definition in DEFINITIONS:
        for L, e in out["definitions"][definition]["layers"].items():
            nec = e.get("frac_nec")
            rnd = e.get("rand_nec")
            if nec is None:
                continue
            clean = rnd is not None and rnd < BASE_FLOOR
            cands.append({"definition": definition, "layer": L, "nec": nec, "rand": rnd,
                          "cos": e.get("cos_cave_conf"), "clean": clean})
    clean_cands = [c for c in cands if c["clean"]]
    pool_best = clean_cands if clean_cands else cands       # prefer clean; fall back so a best is reported
    best = max(pool_best, key=lambda c: (c["nec"] if c["nec"] is not None else -9)) if pool_best else None

    if best is not None:
        out["best"] = {"definition": best["definition"], "layer": best["layer"],
                       "frac_nec": best["nec"], "rand_nec": best["rand"], "cos_cave_conf": best["cos"]}
        out["decision"] = model_decision(best["nec"], best["rand"], best["cos"])
    else:
        out["best"] = None
        out["decision"] = model_decision(None, None, None)
    return out


def run(name_base, name_it, tag, device, pool):
    from headset_direction import FIT_LAYERS as _FL, _rname   # reuse the layer sweep + resid_post hook name
    P = _pure()
    layers = list(_FL)
    res = {"base": _measure_model(name_base, False, device, layers, _rname, pool, P),
           "it": _measure_model(name_it, True, device, layers, _rname, pool, P)}
    out = {
        "name_base": name_base, "name_it": name_it, "device": device, "tag": tag,
        "cue": "confidence_direction_causal", "pool_size": len(pool),
        "fit_layers": layers,
        "definitions": list(DEFINITIONS),
        "thresholds": {"DIR_THR": DIR_THR, "BASE_FLOOR": BASE_FLOOR, "COS_THR": COS_THR,
                       "QUANTILE": QUANTILE, "SPLIT_SEED": SPLIT_SEED, "RAND_SEED": RAND_SEED},
        "decision_rule": DECISION_RULE,
        "base": res["base"], "it": res["it"],
    }
    Path("out").mkdir(exist_ok=True)
    fn = f"out/confidence_direction_causal_{tag}.json"
    Path(fn).write_text(json.dumps(out, indent=2, default=str))
    for m in MODELS:
        dd = res[m]["decision"]
        b = res[m]["best"]
        bd = f"{b['definition']}@L{b['layer']}" if b else "none"
        print(f"[{m}] {dd['causal_bucket']} (best={bd} nec={dd['best_nec']} rand={dd['best_rand']}) | "
              f"COS={dd['cos_bucket']} (|cos|={dd['best_cos']})", flush=True)
    print(f"[done] wrote {fn}", flush=True)


# ----------------------------------------------------------------- selftest (model-free, CPU, no model load)
def _plant(n, d, axis, levels, noise, gen):
    """Synthesize n residual rows carrying a planted confidence AXIS at given per-item levels plus small
    isotropic noise: row[i] = levels[i]*axis + noise*randn. Returns [n, d]. Pure given the generator."""
    rows = noise * torch.randn(n, d, generator=gen)
    for i in range(n):
        rows[i] = rows[i] + levels[i] * axis
    return rows


def selftest():
    P = _pure()
    unit, diff_of_means, cosine = P["unit"], P["diff_of_means"], P["cosine"]
    frac_moved = P["frac_moved"]
    torch.manual_seed(0)

    # ---------- quartile contrast is sharp, disjoint, drops the middle ----------
    vals = [float(x) for x in range(12)]                 # 0..11
    hi, lo = quantile_split(vals, 0.25)
    assert hi == [9, 10, 11] and lo == [0, 1, 2], (hi, lo)         # top-3 vs bottom-3 (k=floor(.25*12)=3)
    assert set(hi).isdisjoint(set(lo)) and 5 not in hi + lo and 6 not in hi + lo  # near-median dropped
    h2, l2 = quantile_split([3.0, 1.0], 0.25)            # n=2 -> one each, still disjoint
    assert len(h2) == 1 and len(l2) == 1 and h2 != l2, (h2, l2)
    assert quantile_split([1.0], 0.25) == ([], [])       # n<2 degenerate -> empty
    print(f"[selftest] quartile split: high={hi} low={lo} (middle dropped, disjoint); n=2 -> 1/1")

    # ---------- (i) planted confidence axis: diff-of-means recovers it; necessity high; random ~0 ----------
    # d large so a random direction's overlap with the planted axis is ~1/sqrt(d) (clean random control).
    d, n = 256, 40
    g = torch.Generator().manual_seed(11)
    axis = unit(torch.randn(d, generator=g))
    # per-item confidence level spread across [-3,3]; the synthetic signal = the row's axis coordinate, so
    # ablating the axis-projection to the low mean collapses the signal (necessity high), a random
    # direction does not touch the axis coordinate (necessity ~0). Pure arithmetic, no model.
    levels = [(-1.0 + 2.0 * i / (n - 1)) * 3.0 for i in range(n)]
    rows = _plant(n, d, axis, levels, noise=0.05, gen=g)
    sig = [float(rows[i] @ axis) for i in range(n)]       # signal = axis coordinate
    hi_idx, lo_idx = quantile_split(sig, 0.25)
    u_conf = unit(diff_of_means(rows[hi_idx], rows[lo_idx]))
    cos_rec = abs(cosine(u_conf, axis))
    assert cos_rec > 0.95, f"diff-of-means should recover the planted axis: {cos_rec}"
    proj_lo = statistics.mean(float(rows[k] @ u_conf) for k in lo_idx)
    sig_lo_tr = statistics.mean(sig[k] for k in lo_idx)
    # NECESSITY on a held-out high item: set its u_conf-projection to proj_lo -> signal -> ~proj_lo
    nec_list = []
    for k in hi_idx:
        cur = float(rows[k] @ u_conf)
        ablated = rows[k] + (proj_lo - cur) * u_conf     # the proj-edit applied to the synthetic row
        sig_ab = float(ablated @ axis)
        nec_list.append(frac_moved(sig[k], sig_ab, sig_lo_tr))
    nec = statistics.mean(nec_list)
    assert nec > 0.8, f"planted-axis held-out necessity should be high: {nec}"
    # RANDOM matched-magnitude direction: orthogonal-ish to the axis -> editing it leaves the signal ~unchanged
    grand = torch.Generator().manual_seed(RAND_SEED)
    u_rand = unit(torch.randn(d, generator=grand))
    prj_lo_rand = statistics.mean(float(rows[k] @ u_rand) for k in lo_idx)
    rand_list = []
    for k in hi_idx:
        cur = float(rows[k] @ u_rand)
        edited = rows[k] + (prj_lo_rand - cur) * u_rand
        sig_rr = float(edited @ axis)
        rand_list.append(frac_moved(sig[k], sig_rr, sig_lo_tr))
    rand = statistics.mean(rand_list)
    assert abs(rand) < BASE_FLOOR, f"random matched-magnitude direction must not move the signal: {rand}"
    assert decide_causal(nec, rand) == "CAUSAL_CONFIDENCE_DIRECTION", (nec, rand)
    print(f"[selftest] (i) cos(u_conf,axis)={cos_rec:.3f} nec={nec:.3f} rand={rand:.3f} -> "
          f"{decide_causal(nec, rand)}")

    # ---------- (ii) flat / no-axis synthetic -> NO_CAUSAL_CONFIDENCE_DIRECTION ----------
    # The confidence SIGNAL exists and has a clean high/low spread (stable gap), but it is an EXTERNAL scalar
    # NOT carried by any residual coordinate: the residuals are isotropic noise, so the fitted diff-of-means
    # direction has no causal link to the signal, and (the signal being external) the residual proj-edit
    # provably leaves it unchanged -> necessity exactly 0. This is the spec's "flat / no-axis" case: there is
    # a confidence variable but no residual axis that mediates it. Pure arithmetic, no model.
    gflat = torch.Generator().manual_seed(99)
    flat = torch.randn(n, d, generator=gflat)             # isotropic residuals, NO planted axis
    ext_sig = sorted([(-1.0 + 2.0 * i / (n - 1)) * 3.0 for i in range(n)])   # external confidence scalar
    hf, lf = quantile_split(ext_sig, 0.25)                # clean high/low spread -> stable denominators
    u_flat = unit(diff_of_means(flat[hf], flat[lf]))      # fit on isotropic residuals -> fit-noise direction
    proj_lo_f = statistics.mean(float(flat[j] @ u_flat) for j in lf)
    sig_lo_f = statistics.mean(ext_sig[j] for j in lf)
    nec_flat_list = []
    for j in hf:
        cur = float(flat[j] @ u_flat)
        _edited = flat[j] + (proj_lo_f - cur) * u_flat    # the residual edit (does not feed ext_sig)
        sig_ab = ext_sig[j]                               # external signal: unchanged by any residual edit
        nec_flat_list.append(frac_moved(ext_sig[j], sig_ab, sig_lo_f))
    nec_flat = statistics.mean(nec_flat_list)
    assert abs(nec_flat) < 1e-9, f"flat synthetic necessity must be ~0 (no residual axis mediates): {nec_flat}"
    assert abs(nec_flat) < DIR_THR
    assert decide_causal(nec_flat, 0.0) == "NO_CAUSAL_CONFIDENCE_DIRECTION", nec_flat
    print(f"[selftest] (ii) flat-synthetic nec={nec_flat:.3f} -> {decide_causal(nec_flat, 0.0)}")

    # ---------- (iii) planted cave + confidence axes at a KNOWN cosine -> measured cosine + COS bucket ----------
    e0 = torch.zeros(d); e0[0] = 1.0
    e1 = torch.zeros(d); e1[1] = 1.0
    u_cave_true = e0.clone()
    for target_cos, want_bucket in ((0.80, "HIGH_COS"), (0.30, "LOW_COS")):
        u_conf_true = unit(target_cos * e0 + (1 - target_cos ** 2) ** 0.5 * e1)
        assert abs(cosine(u_cave_true, u_conf_true) - target_cos) < 1e-5
        gg = torch.Generator().manual_seed(7)
        # fit u_conf from a high/low confidence contrast along u_conf_true
        lv = [3.0] * 8 + [-3.0] * 8
        rconf = _plant(16, d, u_conf_true, lv, noise=0.02, gen=gg)
        sg = [float(rconf[i] @ u_conf_true) for i in range(16)]
        hq, lq = quantile_split(sg, 0.5)                 # halves here (small synthetic set)
        u_conf_fit = unit(diff_of_means(rconf[hq], rconf[lq]))
        # fit u_cave from a counter-vs-neutral contrast along u_cave_true
        counter = _plant(8, d, u_cave_true, [4.0] * 8, noise=0.02, gen=gg)
        neutral = _plant(8, d, u_cave_true, [0.0] * 8, noise=0.02, gen=gg)
        u_cave_fit = unit(diff_of_means(counter, neutral))
        measured = cosine(u_cave_fit, u_conf_fit)
        assert abs(abs(measured) - target_cos) < 0.05, (measured, target_cos)
        assert decide_cos(measured, causal=True) == want_bucket, (measured, want_bucket)
        print(f"[selftest] (iii) planted cos={target_cos} measured={measured:.3f} -> "
              f"{decide_cos(measured, causal=True)} (want {want_bucket})")

    # ---------- (iv) decisions fire exactly at the thresholds ----------
    assert decide_causal(0.55, 0.01) == "CAUSAL_CONFIDENCE_DIRECTION"          # high nec + clean random
    assert decide_causal(0.55, 0.30) == "NO_CAUSAL_CONFIDENCE_DIRECTION"       # dirty random (specificity)
    assert decide_causal(0.10, 0.01) == "NO_CAUSAL_CONFIDENCE_DIRECTION"       # nec below threshold
    assert decide_causal(DIR_THR, 0.0) == "CAUSAL_CONFIDENCE_DIRECTION"
    assert decide_causal(DIR_THR - 1e-6, 0.0) == "NO_CAUSAL_CONFIDENCE_DIRECTION"
    assert decide_causal(0.55, BASE_FLOOR) == "NO_CAUSAL_CONFIDENCE_DIRECTION"  # rand must be strictly below
    assert decide_causal(None, None) == "NO_CAUSAL_CONFIDENCE_DIRECTION"
    print("[selftest] (iv) decide_causal thresholds exact (nec>=DIR_THR AND rand<BASE_FLOOR)")

    # COS bucket only meaningful when causal; else N/A
    assert decide_cos(0.92, causal=True) == "HIGH_COS" and decide_cos(-0.92, causal=True) == "HIGH_COS"
    assert decide_cos(0.30, causal=True) == "LOW_COS"
    assert decide_cos(COS_THR, causal=True) == "HIGH_COS" and decide_cos(COS_THR - 1e-6, causal=True) == "LOW_COS"
    assert decide_cos(0.92, causal=False) == "N/A" and decide_cos(None, causal=True) == "N/A"
    print("[selftest] (iv) COS bucket: HIGH/LOW at COS_THR when causal; N/A when not causal")

    # assembled per-model decisions
    md = model_decision(best_nec=0.45, best_rand=0.01, best_cos=0.80)
    assert md["causal_bucket"] == "CAUSAL_CONFIDENCE_DIRECTION" and md["cos_bucket"] == "HIGH_COS" \
        and md["best_cos"] == 0.8, md
    md2 = model_decision(best_nec=0.45, best_rand=0.01, best_cos=0.30)
    assert md2["causal_bucket"] == "CAUSAL_CONFIDENCE_DIRECTION" and md2["cos_bucket"] == "LOW_COS", md2
    md3 = model_decision(best_nec=0.05, best_rand=0.0, best_cos=0.95)
    assert md3["causal_bucket"] == "NO_CAUSAL_CONFIDENCE_DIRECTION" and md3["cos_bucket"] == "N/A" \
        and md3["best_cos"] is None, md3                   # cosine suppressed (N/A) when not causal
    print(f"[selftest] (iv) assembled decisions: A={md['causal_bucket']}/{md['cos_bucket']} "
          f"B={md2['causal_bucket']}/{md2['cos_bucket']} C={md3['causal_bucket']}/{md3['cos_bucket']}")

    # ---------- frac_moved arithmetic sanity (reused helper) ----------
    assert abs(frac_moved(5.0, 1.0, 1.0) - 1.0) < 1e-9 and abs(frac_moved(5.0, 5.0, 1.0)) < 1e-9
    assert frac_moved(2.0, 2.0, 2.0) == 0.0
    print("[selftest] frac_moved: full=1.0 none=0.0 degenerate=0.0")
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
