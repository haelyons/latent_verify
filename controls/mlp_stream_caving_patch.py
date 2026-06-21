"""CAUSAL test of the MLP-stream cave-direction WRITE: does removing it reduce caving? (in-distribution)

WHY (neutral). cave_direction_dla.py decomposed the base->it change in the rank-1 cave/defer direction
u_cave and found the MLP stream carries the WRITE of that change (the attn-vs-MLP split of |delta_onbase|
leans MLP). That is a DESCRIPTIVE fact about the geometry -- the MLP writes along the direction -- not a
CAUSAL one: a component can write along an axis that the model never actually reads downstream. This
control tests the causal half on the -it caving items, IN-DISTRIBUTION (no cross-model substitution, no
off-distribution joint patch): on the caved (counter) state, REMOVE the cave-direction component that the
MLP stream carries -- project hook_mlp_out onto u_cave and subtract that projection, per layer L'<=L, in
ONE forward -- and measure whether caving REDUCES (M = logp(C) - logp(W*) recovers toward the neutral
state). Two negatives delimit it: the SAME projection-out applied to the ATTENTION output stream instead
(the DLA predicts attn does less), and projecting out a RANDOM matched-magnitude direction from the MLP
stream (specificity). Reported on -it and on base (the RLHF differential).

Projection-out (exact): for component output o and unit u, o' = o - (o . u) u. Then o' . u == 0 -- the
cave-component of that component's write is exactly zeroed, while the orthogonal complement of the write
is untouched. Applied at hook_mlp_out[L'][-1] (resp. hook_attn_out[L'][-1]) for every L' in 0..L in a
single forward pass, this removes the cave-direction part the MLP (resp. attention) stream writes into the
residual at every layer up to and including the readout layer L, then lets the rest of the network run.

WHAT IT MEASURES, per model (default google/gemma-2-9b-it; also google/gemma-2-9b base as a differential):
  0. Headline readout layer L: the FIT_LAYERS layer with the largest cave-NECESSITY on this model
     (necessity = projection-edit ablation of u_cave's coordinate to the neutral mean on counter, the
     headset_direction operator: frac = (M_ablate - M_counter)/(M_neutral - M_counter)). L is chosen per
     model from its own necessity sweep, exactly as headset_direction.run picks bestL.
  1. Fit u_cave at L: diff-of-means counter-vs-neutral over this model's qualifying caving items
     (u = normalize(mean_i(rc_i - rn_i)), the |gap| = |M_neutral - M_counter| >= MIN_EFFECT_NET gate,
     the metric M = logp(C) - logp(W*) first-token margin) -- the EXACT construction in
     headset_direction._dir_pass / cave_direction_heldout.fit_direction. Held-out report: also fit on a
     train fold and apply on a disjoint test fold (LOO), so the headline recovery is not purely in-sample.
  2. MLP-stream cave-component ablation. On each caving item (counter condition), in ONE forward, for every
     layer L'<=L subtract from hook_mlp_out[L'][-1] its projection onto u_cave. Recovery frac =
     (M_ablate - M_counter)/(M_neutral - M_counter). mlp_frac_recovered = mean over items.
  3. ATTENTION-stream cave-component ablation. Same projection-out, applied to hook_attn_out[L'][-1] (the
     summed-head attention contribution into the residual) per layer L'<=L. attn_frac_recovered = mean.
  4. RANDOM-direction control. Project out a random unit direction (matched magnitude, fixed seed) from the
     MLP stream per layer L'<=L. random_frac_recovered = mean (specificity floor).
  5. base vs -it: all four numbers per model -- does removing the MLP cave-component reduce caving MORE in
     -it than base (the RLHF differential)?

NEUTRAL DECISION (module constants DRIVE_THR=0.20, BASE_FLOOR=0.05; numbers + categories only, no
hypothesis, no statement about which stream should win). Per model:
  MLP_STREAM_DRIVES_CAVING iff mlp_frac_recovered >= DRIVE_THR  AND  random_frac_recovered < BASE_FLOOR
      AND  mlp_frac_recovered > attn_frac_recovered;
  else NOT_MLP_DRIVEN (sub-category: ATTN_MATCHES_OR_EXCEEDS_MLP if attn >= mlp; NON_SPECIFIC if the random
      control also recovers; WRITES_BUT_DOES_NOT_DRIVE if mlp < DRIVE_THR -- the MLP writes the direction
      but removing it does not causally reduce caving via it). Reports mlp_frac_recovered,
      attn_frac_recovered, random_frac_recovered, headline layer L, and the base-vs-it differential.

The projection-out edit (output minus its u-projection), the multi-layer single-forward stream ablation,
the per-model headline-by-necessity selection, the recovery-fraction aggregation, and the decision are the
only new logic. All pure parts are inert in --selftest, which is model-free, loads NO model, and exercises:
the projection-out math (o' . u == 0 to numerical tolerance; magnitude-matched random removal), a synthetic
where M is a function of the MLP-stream cave-component (removing it recovers M; removing the attention
cave-component or a random direction does not) -> MLP_STREAM_DRIVES_CAVING; a synthetic where M depends on
the ATTENTION cave-component instead -> NOT (attn matches/exceeds mlp); a synthetic where M is independent
of any cave-component -> NOT (recovery ~0); and the recovery-fraction + decision boundaries.

MEMORY (9b note). The intervention is a per-layer projection-OUT done in ONE forward: each hooked layer L'
edits only hook_mlp_out[0,-1,:] (resp. hook_attn_out[0,-1,:]) in place -- a single [d_model] vector per
layer, no extra activation cache, no backward. Peak memory = weights + one forward graph (same budget as
cave_direction_dla's forward passes; it fits the 40GB A100). u_cave is one [d_model] tensor. The headline
necessity sweep and the fits reuse headset_direction's forward-only resid_post caching (last token only).
hook_attn_out / hook_mlp_out are the standard transformer_lens summed-stream hooks -- no use_attn_result,
so nothing materializes [seq, n_head, d_model]; the per-layer edit is applied to the already-summed write.

  python controls/mlp_stream_caving_patch.py --selftest
  python controls/mlp_stream_caving_patch.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

# Repo-internal imports (item pool, turn builders, metric, fit machinery) are DEFERRED into run() so
# --selftest runs standalone from controls/ on CPU with NO model load and nothing else on sys.path; on the
# box the reference files are scp'd flat into latent_verify/ where these resolve.

# --------------------------------------------------------------------------- pre-registered constants
FIT_LAYERS = [24, 28, 32, 36]   # same sweep as headset_direction.py / cave_direction_heldout.py
DRIVE_THR = 0.20                # mlp cave-component ablation must recover >= this frac of the cave to "drive"
BASE_FLOOR = 0.05               # random-direction recovery must sit below this to be "clean" (specific)
SEED = 0                        # random matched-direction generator seed
N_FOLDS_LOO = True              # held-out report uses leave-one-out (fit train fold, eval disjoint item)


# --------------------------------------------------------------------------- pure fit / projection math
def fit_u(rc_list, rn_list, idxs=None):
    """Diff-of-means cave direction over aligned counter/neutral last-token residual lists.
    u = normalize(mean_i(rc_i - rn_i)) over items in idxs (default all). Pure (tensors in, unit tensor out).
    Mirrors headset_direction._dir_pass / cave_direction_heldout.fit_direction exactly."""
    if idxs is None:
        idxs = range(len(rc_list))
    D = torch.stack([rc_list[i] - rn_list[i] for i in idxs])      # [n, d]
    cave = D.mean(0)
    return cave / (cave.norm() + 1e-8)


def project_out(o, u):
    """Remove the u-component of o: o' = o - (o . u) u, where u is a unit vector. Pure (tensors in, tensor
    out). The result is orthogonal to u (o' . u == 0 to numerical tolerance), so the cave-direction part of
    o is exactly zeroed and the orthogonal complement is untouched. This is the in-distribution edit (we
    only remove a component the model already produced; we do not substitute any cross-model activation)."""
    coef = (o * u).sum(-1, keepdim=True)
    return o - coef * u


def random_unit(shape, dtype, gen):
    """Random unit direction (cpu generator -> caller moves it to device). Matched magnitude by construction:
    project_out removes whatever (o . ur) is, so the removed vector ||(o.ur)ur|| is the o-magnitude along a
    random axis -- the same projection-out operator as the cave axis, just a different unit direction. Pure."""
    r = torch.randn(shape, generator=gen).to(dtype)
    return r / (r.norm() + 1e-8)


def recovery_frac(M_ablate, M_counter, M_neutral):
    """Fraction of the cave recovered by the ablation: (M_ablate - M_counter)/(M_neutral - M_counter).
    Same sign convention as headset_direction / rlhf_differential._confirm: 0 = no change from the caved
    state, 1 = fully back to the neutral (uncaved) margin. Pure. gap ~ 0 -> None (div-by-~0, not scored)."""
    gap = M_neutral - M_counter
    if abs(gap) < 1e-9:
        return None
    return (M_ablate - M_counter) / gap


# --------------------------------------------------------------------------- pure decision
def decide_drives(mlp_frac, attn_frac, random_frac, drive_thr=DRIVE_THR, base_floor=BASE_FLOOR):
    """Neutral decision over one model's three recovery fractions. Numbers + categories only.
      MLP_STREAM_DRIVES_CAVING iff mlp_frac >= drive_thr AND random_frac < base_floor AND mlp_frac > attn_frac.
      else NOT_MLP_DRIVEN, with a sub-category naming WHICH condition failed:
        WRITES_BUT_DOES_NOT_DRIVE : mlp_frac < drive_thr (removing the MLP cave-component does not reduce
            caving enough -- the MLP writes the direction but caving does not causally flow through it).
        NON_SPECIFIC              : the random matched-magnitude control also recovers (>= base_floor).
        ATTN_MATCHES_OR_EXCEEDS_MLP : the attention-stream ablation recovers >= the MLP-stream ablation.
    Pure over the three measured floats."""
    def r(x):
        return round(x, 4) if x is not None else None
    have = mlp_frac is not None and random_frac is not None and attn_frac is not None
    mlp_ok = mlp_frac is not None and mlp_frac >= drive_thr
    rand_clean = random_frac is not None and random_frac < base_floor
    mlp_beats_attn = mlp_frac is not None and attn_frac is not None and mlp_frac > attn_frac
    drives = bool(have and mlp_ok and rand_clean and mlp_beats_attn)
    if drives:
        cat, sub = "MLP_STREAM_DRIVES_CAVING", None
        msg = (f"removing the MLP-stream cave-component recovers {mlp_frac:.3f} (>= {drive_thr}) of the cave, "
               f"the random matched-magnitude control recovers {random_frac:.3f} (< {base_floor}), and the MLP "
               f"recovery exceeds the attention recovery ({attn_frac:.3f}) -- the MLP cave-component CAUSALLY "
               f"drives caving in-distribution, not just correlates with the direction.")
    else:
        cat = "NOT_MLP_DRIVEN"
        reasons = []
        if not mlp_ok:
            sub = "WRITES_BUT_DOES_NOT_DRIVE"
            reasons.append(f"mlp_frac {r(mlp_frac)} < {drive_thr}")
        elif not rand_clean:
            sub = "NON_SPECIFIC"
            reasons.append(f"random control {r(random_frac)} >= {base_floor}")
        elif not mlp_beats_attn:
            sub = "ATTN_MATCHES_OR_EXCEEDS_MLP"
            reasons.append(f"attn_frac {r(attn_frac)} >= mlp_frac {r(mlp_frac)}")
        else:
            sub = "INSUFFICIENT_DATA"
            reasons.append("one or more recovery fractions unavailable (gap ~ 0)")
        msg = ("removing the MLP-stream cave-component does NOT establish a causal drive: " +
               "; ".join(reasons) + ".")
    return {"category": cat, "subcategory": sub, "mlp_stream_drives_caving": drives,
            "mlp_frac_recovered": r(mlp_frac), "attn_frac_recovered": r(attn_frac),
            "random_frac_recovered": r(random_frac),
            "drive_thr": drive_thr, "base_floor": base_floor, "msg": msg}


def differential_note(it_dec, base_dec):
    """Neutral base-vs-it differential over the two MLP recoveries. Numbers + a category only; no hypothesis.
    INSTALLED  : it drives AND base does not (it >= DRIVE_THR, base < DRIVE_THR).
    AMPLIFIED  : both >= DRIVE_THR but it > base by >= (DRIVE_THR/2).
    SHARED     : both >= DRIVE_THR and within DRIVE_THR/2 of each other.
    NEITHER    : both < DRIVE_THR."""
    mi = it_dec.get("mlp_frac_recovered")
    mb = base_dec.get("mlp_frac_recovered")
    if mi is None or mb is None:
        return {"category": "INSUFFICIENT_DATA", "it_mlp_frac": mi, "base_mlp_frac": mb,
                "msg": "a model's mlp recovery is unavailable."}
    it_ok = mi >= DRIVE_THR
    base_ok = mb >= DRIVE_THR
    half = DRIVE_THR / 2
    if it_ok and not base_ok:
        cat, msg = "INSTALLED", (f"-it MLP recovery {mi:.3f} >= {DRIVE_THR} while base {mb:.3f} < {DRIVE_THR} "
                                 f"-- the MLP cave-drive is RLHF-installed.")
    elif it_ok and base_ok and (mi - mb) >= half:
        cat, msg = "AMPLIFIED", (f"both recover (it {mi:.3f}, base {mb:.3f}) but -it exceeds base by "
                                 f">= {half:.2f} -- the MLP cave-drive is RLHF-amplified.")
    elif it_ok and base_ok:
        cat, msg = "SHARED", (f"both recover comparably (it {mi:.3f}, base {mb:.3f}; within {half:.2f}) "
                              f"-- the MLP cave-drive is base-present, not an RLHF differential.")
    else:
        cat, msg = "NEITHER", (f"neither model's MLP recovery reaches {DRIVE_THR} (it {mi:.3f}, base {mb:.3f}).")
    return {"category": cat, "it_mlp_frac": round(mi, 4), "base_mlp_frac": round(mb, 4),
            "it_minus_base": round(mi - mb, 4), "drive_thr": DRIVE_THR, "msg": msg}


# --------------------------------------------------------------------------- real run hook names
def _rname(L):
    return f"blocks.{L}.hook_resid_post"


def _mname(L):
    return f"blocks.{L}.hook_mlp_out"


def _aname(L):
    return f"blocks.{L}.hook_attn_out"


# --------------------------------------------------------------------------- real model passes
def _collect_ctxs(model, device, is_chat, fit_layer, pool, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET):
    """Gap-pass at one fit layer: for each item build counter/neutral, cache resid_post[fit_layer][-1] for
    both, compute M_ctr/M_neu (first-token C-vs-W* margin), keep items with |gap| >= MIN_EFFECT_NET. Same
    gate and metric as headset_direction._dir_pass / cave_direction_heldout._collect_ctxs."""
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


def _necessity_at_layer(model, ctxs, fit_layer, _logp_diff, device):
    """Cave-NECESSITY at one fit layer (headset_direction operator): fit u_cave on all ctxs, then on each
    counter move the u_cave coordinate to the neutral-mean projection (a 1-D projection edit on
    resid_post[fit_layer][-1]) and read the recovery frac. Returns mean recovery -- the quantity used to pick
    the headline layer L per model (largest cave-necessity), exactly as headset_direction.run picks bestL."""
    rc_list = [c["rc"] for c in ctxs]
    rn_list = [c["rn"] for c in ctxs]
    u = fit_u(rc_list, rn_list).to(device)
    proj_n = statistics.mean(float(c["rn"] @ u) for c in ctxs)
    fracs = []
    for c in ctxs:
        shift = proj_n - float(c["rc"] @ u)
        def ab(r, hook, u=u, shift=shift):
            r[0, -1] = r[0, -1] + (shift * u).to(r.dtype); return r
        with torch.no_grad():
            M_ab = float(_logp_diff(model.run_with_hooks(c["counter"], fwd_hooks=[(_rname(fit_layer), ab)]),
                                    c["cid"], c["aid"]))
        f = recovery_frac(M_ab, c["M_ctr"], c["M_neu"])
        if f is not None:
            fracs.append(f)
    return statistics.mean(fracs) if fracs else None, u, proj_n


def _stream_ablate_frac(model, ctxs, L, u, _logp_diff, hook_name_fn, eval_idxs=None):
    """Per-item recovery from projecting u OUT of one stream (hook_name_fn -> hook_mlp_out or hook_attn_out)
    at every layer L'<=L in ONE forward. On each item's counter prompt, install a project_out hook on
    stream[L'][-1] for L' in 0..L, run one forward, read M. recovery = (M_ablate - M_counter)/gap.
    eval_idxs restricts which items are evaluated (held-out support). Returns the per-item frac list."""
    if eval_idxs is None:
        eval_idxs = range(len(ctxs))
    u = u.to(device=u.device)
    fracs = []
    for i in eval_idxs:
        c = ctxs[i]
        def proj_hook(o, hook, u=u):
            o[0, -1] = project_out(o[0, -1], u.to(o.dtype)); return o
        hooks = [(hook_name_fn(Lp), proj_hook) for Lp in range(L + 1)]
        with torch.no_grad():
            M_ab = float(_logp_diff(model.run_with_hooks(c["counter"], fwd_hooks=hooks), c["cid"], c["aid"]))
        f = recovery_frac(M_ab, c["M_ctr"], c["M_neu"])
        if f is not None:
            fracs.append(f)
    return fracs


def _heldout_mlp_frac(model, ctxs, L, _logp_diff, device):
    """Leave-one-out held-out MLP-stream recovery: for each item i, fit u_cave on the OTHER items, project
    it out of the MLP stream on item i's counter (layers 0..L), read recovery. Mean over the held-out items.
    Guards against the in-sample diff-of-means fit recovering on its own fit items."""
    rc_list = [c["rc"] for c in ctxs]
    rn_list = [c["rn"] for c in ctxs]
    n = len(ctxs)
    if n < 3:
        return None
    pooled = []
    for i in range(n):
        train = [j for j in range(n) if j != i]
        u = fit_u(rc_list, rn_list, train).to(device)
        pooled += _stream_ablate_frac(model, ctxs, L, u, _logp_diff, _mname, eval_idxs=[i])
    return statistics.mean(pooled) if pooled else None


def _model_pass(name, is_chat, device, refs):
    """One model: collect ctxs at each FIT_LAYER, pick the headline layer L by largest cave-necessity, then
    at L measure MLP-stream / attention-stream / random-direction cave-component ablation recovery (and a
    leave-one-out held-out MLP recovery). Returns a dict of numbers + the per-model decision."""
    from transformer_lens import HookedTransformer
    (_ITEMS, ITEMS_WIDE, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET) = refs
    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()

    # --- necessity sweep over FIT_LAYERS: headline L = layer with the largest cave-necessity ---
    nec, ctxs_by_L, u_by_L, projn_by_L = {}, {}, {}, {}
    for L in FIT_LAYERS:
        ctxs = _collect_ctxs(model, device, is_chat, L, ITEMS_WIDE, PUSH, NEUTRAL, _helpers, _logp_diff, MIN_EFFECT_NET)
        if len(ctxs) < 3:
            print(f"  [L{L}] too few qualifying items ({len(ctxs)}); skipping", flush=True)
            continue
        ctxs_by_L[L] = ctxs
        n, u, proj_n = _necessity_at_layer(model, ctxs, L, _logp_diff, device)
        nec[L] = n; u_by_L[L] = u; projn_by_L[L] = proj_n
        print(f"  [L{L}] n_ok={len(ctxs)} cave_necessity={None if n is None else round(n, 4)}", flush=True)

    avail = [L for L in FIT_LAYERS if nec.get(L) is not None]
    if not avail:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return {"headline_layer": None, "n_ok": 0, "nec_by_layer": {},
                "decision": decide_drives(None, None, None)}
    L = max(avail, key=lambda Lx: nec[Lx])
    ctxs, u = ctxs_by_L[L], u_by_L[L]
    print(f"  [headline] L{L} (cave_necessity={round(nec[L], 4)})", flush=True)

    # --- the three in-distribution stream ablations at the headline layer L ---
    mlp_fr = _stream_ablate_frac(model, ctxs, L, u, _logp_diff, _mname)
    attn_fr = _stream_ablate_frac(model, ctxs, L, u, _logp_diff, _aname)
    g = torch.Generator(device="cpu").manual_seed(SEED)
    ur = random_unit((u.shape[0],), u.dtype, g).to(device)
    rand_fr = _stream_ablate_frac(model, ctxs, L, ur, _logp_diff, _mname)
    mlp_frac = statistics.mean(mlp_fr) if mlp_fr else None
    attn_frac = statistics.mean(attn_fr) if attn_fr else None
    rand_frac = statistics.mean(rand_fr) if rand_fr else None

    # --- held-out (LOO) MLP recovery, so the headline is not purely in-sample ---
    held = _heldout_mlp_frac(model, ctxs, L, _logp_diff, device) if N_FOLDS_LOO else None

    decision = decide_drives(mlp_frac, attn_frac, rand_frac)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"headline_layer": L, "n_ok": len(ctxs),
            "nec_by_layer": {str(Lx): (round(nec[Lx], 4) if nec.get(Lx) is not None else None) for Lx in FIT_LAYERS},
            "cave_necessity_at_headline": round(nec[L], 4),
            "mlp_frac_recovered": (round(mlp_frac, 4) if mlp_frac is not None else None),
            "attn_frac_recovered": (round(attn_frac, 4) if attn_frac is not None else None),
            "random_frac_recovered": (round(rand_frac, 4) if rand_frac is not None else None),
            "mlp_heldout_loo_frac": (round(held, 4) if held is not None else None),
            "decision": decision}


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
    diff = differential_note(it["decision"], base["decision"])

    out = {"model_base": name_base, "model_it": name_it, "cue": "mlp_stream_caving_patch",
           "substrate": "misconception caving items (TruthfulQA-style), wide pool",
           "measures": ("in-distribution causal test of the MLP-stream cave-direction WRITE: on the caved "
                        "(counter) state, project u_cave OUT of hook_mlp_out (resp. hook_attn_out, resp. a "
                        "random matched-magnitude direction out of hook_mlp_out) at every layer L'<=L in one "
                        "forward, and measure caving reduction recovery = (M_ablate - M_counter)/(M_neutral - "
                        "M_counter). NO cross-model substitution -- only a component the model produced is "
                        "removed (project_out: o' = o - (o.u)u, o'.u==0)."),
           "metric": "M = logp(C) - logp(W*) first-token margin; |gap| = |M_neutral - M_counter| >= MIN_EFFECT_NET gate",
           "headline_layer_rule": "FIT_LAYERS layer with the largest cave-necessity, chosen per model",
           "fit_layers": FIT_LAYERS,
           "thresholds": {"drive_thr": DRIVE_THR, "base_floor": BASE_FLOOR, "seed": SEED},
           "it": it, "base": base, "rlhf_differential": diff}
    Path("out").mkdir(exist_ok=True)
    fn = f"out/mlp_stream_caving_patch_{tag}.json"
    Path(fn).write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[it]   L{it['headline_layer']} mlp={it.get('mlp_frac_recovered')} "
          f"attn={it.get('attn_frac_recovered')} rand={it.get('random_frac_recovered')} "
          f"loo={it.get('mlp_heldout_loo_frac')} -> {it['decision']['category']}"
          f"{'/' + it['decision']['subcategory'] if it['decision'].get('subcategory') else ''}")
    print(f"[base] L{base['headline_layer']} mlp={base.get('mlp_frac_recovered')} "
          f"attn={base.get('attn_frac_recovered')} rand={base.get('random_frac_recovered')} "
          f"loo={base.get('mlp_heldout_loo_frac')} -> {base['decision']['category']}"
          f"{'/' + base['decision']['subcategory'] if base['decision'].get('subcategory') else ''}")
    print(f"[rlhf differential] {diff['category']}: {diff['msg']}")
    print(f"[done] wrote {fn}")


# --------------------------------------------------------------------------- selftest (model-free, CPU, NO model load)
def _planted_streams(n, d, driver, a, noise, seed):
    """Synthetic per-item streams + a margin readout, with NO model. For each item we draw two stream
    last-token vectors -- mlp and attn -- and a cave axis u. The caved (counter) state writes +a*u into the
    `driver` stream ("mlp" or "attn"; "none" => neither). M is a LINEAR readout of the residual along u
    (M = -(resid . u)), where resid = mlp + attn. So projecting u OUT of the driver stream removes the +a*u
    write and recovers M toward the neutral value, while projecting u out of the OTHER stream (or projecting
    a random direction out of either) removes only the small ambient u-overlap -> ~0 recovery. This mirrors
    the real engine: the recovery is high only when we remove the cave-component from the stream that
    actually carries the caving write. Returns the ctxs list + the cave axis u (unit) + a synthetic M_fn."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, 1, generator=g)
    u = (A[:, 0] / A[:, 0].norm()).contiguous()
    ctxs = []
    for _ in range(n):
        mlp_n = noise * torch.randn(d, generator=g)
        attn_n = noise * torch.randn(d, generator=g)
        # neutral state: ambient streams, no cave write. counter state: +a*u added to the driver stream.
        mlp_c = mlp_n.clone(); attn_c = attn_n.clone()
        if driver == "mlp":
            mlp_c = mlp_c + a * u
        elif driver == "attn":
            attn_c = attn_c + a * u
        resid_c = mlp_c + attn_c
        resid_n = mlp_n + attn_n
        M_ctr = -float(resid_c @ u)
        M_neu = -float(resid_n @ u)
        ctxs.append({"mlp_c": mlp_c, "attn_c": attn_c, "M_ctr": M_ctr, "M_neu": M_neu,
                     "gap": M_neu - M_ctr, "_u": u})
    return ctxs, u


def _synthetic_stream_frac(ctxs, axis, stream):
    """Synthetic analog of _stream_ablate_frac: project `axis` OUT of `stream` ('mlp' or 'attn') of each
    item's COUNTER state, recompute M = -((mlp' + attn') . u_true), and return the per-item recovery frac.
    Mirrors the real edit (project_out on the chosen stream) + the real recovery formula, model-free."""
    fracs = []
    for c in ctxs:
        mlp = c["mlp_c"].clone(); attn = c["attn_c"].clone()
        if stream == "mlp":
            mlp = project_out(mlp, axis)
        else:
            attn = project_out(attn, axis)
        M_ab = -float((mlp + attn) @ c["_u"])
        f = recovery_frac(M_ab, c["M_ctr"], c["M_neu"])
        if f is not None:
            fracs.append(f)
    return statistics.mean(fracs) if fracs else None


def _measure(ctxs, u, d, seed):
    """Run the three synthetic ablations (mlp cave-component, attn cave-component, random-out-of-mlp) and
    return (mlp_frac, attn_frac, random_frac), exactly the three numbers the real pass feeds decide_drives."""
    g = torch.Generator().manual_seed(seed + 777)
    ur = random_unit((d,), u.dtype, g)
    mlp_frac = _synthetic_stream_frac(ctxs, u, "mlp")
    attn_frac = _synthetic_stream_frac(ctxs, u, "attn")
    rand_frac = _synthetic_stream_frac(ctxs, ur, "mlp")
    return mlp_frac, attn_frac, rand_frac


def selftest():
    torch.manual_seed(0)
    d, n = 256, 16

    # ---------- (0) projection-out math: o' . u == 0 ; magnitude removed == |o . u| ----------
    g = torch.Generator().manual_seed(3)
    u = torch.randn(d, generator=g); u = u / u.norm()
    o = torch.randn(d, generator=g)
    o2 = project_out(o, u)
    assert abs(float(o2 @ u)) < 1e-4, f"project_out residual not orthogonal to u: {float(o2 @ u)}"
    removed = o - o2
    assert abs(float(removed.norm()) - abs(float(o @ u))) < 1e-4, (float(removed.norm()), abs(float(o @ u)))
    # idempotent: projecting out twice == once
    assert torch.allclose(project_out(o2, u), o2, atol=1e-5)
    # random_unit is unit-norm; project_out with it also lands orthogonal
    ur = random_unit((d,), torch.float32, torch.Generator().manual_seed(9))
    assert abs(float(ur.norm()) - 1.0) < 1e-5
    assert abs(float(project_out(o, ur) @ ur)) < 1e-4
    print(f"[selftest] (0) project_out: o'.u={float(o2 @ u):.2e}~0, removed_mag=|o.u|, idempotent OK")

    # ---------- recovery_frac sign convention ----------
    assert abs(recovery_frac(2.0, 0.0, 2.0) - 1.0) < 1e-9     # ablate back to neutral -> 1
    assert abs(recovery_frac(0.0, 0.0, 2.0) - 0.0) < 1e-9     # no change -> 0
    assert recovery_frac(1.0, 1.0, 1.0) is None               # gap ~ 0 -> None
    print("[selftest] (0) recovery_frac: full=1, none=0, zero-gap=None OK")

    # ---------- (i) M driven by the MLP-stream cave-component -> MLP_STREAM_DRIVES_CAVING ----------
    ctxs_mlp, u_mlp = _planted_streams(n, d, driver="mlp", a=4.0, noise=0.2, seed=11)
    mlp_f, attn_f, rand_f = _measure(ctxs_mlp, u_mlp, d, seed=11)
    assert mlp_f > 0.8, f"MLP-driven: removing the MLP cave-component should recover M: {mlp_f}"
    assert abs(attn_f) < BASE_FLOOR, f"MLP-driven: removing the ATTN cave-component should not recover: {attn_f}"
    assert abs(rand_f) < BASE_FLOOR, f"MLP-driven: removing a random dir from MLP should not recover: {rand_f}"
    d_mlp = decide_drives(mlp_f, attn_f, rand_f)
    assert d_mlp["category"] == "MLP_STREAM_DRIVES_CAVING" and d_mlp["mlp_stream_drives_caving"], d_mlp
    print(f"[selftest] (i) MLP-driven: mlp={mlp_f:.3f} attn={attn_f:.3f} rand={rand_f:.3f} -> {d_mlp['category']}")

    # ---------- (ii) M driven by the ATTENTION-stream cave-component -> NOT (attn matches/exceeds mlp) ----------
    ctxs_attn, u_attn = _planted_streams(n, d, driver="attn", a=4.0, noise=0.2, seed=12)
    mlp_f2, attn_f2, rand_f2 = _measure(ctxs_attn, u_attn, d, seed=12)
    assert attn_f2 > 0.8, f"ATTN-driven: removing the ATTN cave-component should recover: {attn_f2}"
    assert abs(mlp_f2) < BASE_FLOOR, f"ATTN-driven: removing the MLP cave-component should not recover: {mlp_f2}"
    d_attn = decide_drives(mlp_f2, attn_f2, rand_f2)
    assert d_attn["category"] == "NOT_MLP_DRIVEN", d_attn
    # mlp_f2 ~ 0 < DRIVE_THR, so the sub-category names WRITES_BUT_DOES_NOT_DRIVE (mlp fails first);
    # also exercise the explicit attn>=mlp branch with mlp above DRIVE_THR.
    d_attn_dom = decide_drives(0.30, 0.55, 0.01)
    assert d_attn_dom["category"] == "NOT_MLP_DRIVEN" and d_attn_dom["subcategory"] == "ATTN_MATCHES_OR_EXCEEDS_MLP", d_attn_dom
    print(f"[selftest] (ii) ATTN-driven: mlp={mlp_f2:.3f} attn={attn_f2:.3f} -> {d_attn['category']}; "
          f"attn-dominates branch -> {d_attn_dom['subcategory']}")

    # ---------- (iii) M independent of any cave-component -> NOT (recovery ~0) ----------
    ctxs_none, u_none = _planted_streams(n, d, driver="none", a=4.0, noise=0.2, seed=13)
    mlp_f3, attn_f3, rand_f3 = _measure(ctxs_none, u_none, d, seed=13)
    # no item writes along u, so gap ~ 0 for every item -> recovery undefined (None) OR tiny; either way
    # the decision is NOT_MLP_DRIVEN. Build an explicit small-but-defined case for the boundary.
    assert (mlp_f3 is None) or (abs(mlp_f3) < BASE_FLOOR), f"cave-independent MLP recovery should be ~0/None: {mlp_f3}"
    d_none = decide_drives(0.02, 0.01, 0.0)
    assert d_none["category"] == "NOT_MLP_DRIVEN" and d_none["subcategory"] == "WRITES_BUT_DOES_NOT_DRIVE", d_none
    print(f"[selftest] (iii) cave-independent: mlp={mlp_f3} -> NOT_MLP_DRIVEN/WRITES_BUT_DOES_NOT_DRIVE")

    # ---------- (iv) decision boundaries ----------
    # exactly at DRIVE_THR with clean random and mlp>attn -> drives (>=)
    assert decide_drives(DRIVE_THR, 0.10, 0.01)["category"] == "MLP_STREAM_DRIVES_CAVING"
    # just below DRIVE_THR -> WRITES_BUT_DOES_NOT_DRIVE
    de = decide_drives(DRIVE_THR - 1e-6, 0.10, 0.01)
    assert de["category"] == "NOT_MLP_DRIVEN" and de["subcategory"] == "WRITES_BUT_DOES_NOT_DRIVE", de
    # mlp clears thr, beats attn, but random control dirty -> NON_SPECIFIC
    dn = decide_drives(0.40, 0.10, BASE_FLOOR)
    assert dn["category"] == "NOT_MLP_DRIVEN" and dn["subcategory"] == "NON_SPECIFIC", dn
    # mlp clears thr, clean random, but ties attn (not strictly greater) -> ATTN_MATCHES_OR_EXCEEDS_MLP
    dt = decide_drives(0.40, 0.40, 0.01)
    assert dt["category"] == "NOT_MLP_DRIVEN" and dt["subcategory"] == "ATTN_MATCHES_OR_EXCEEDS_MLP", dt
    print("[selftest] (iv) decision boundaries: at-thr / below-thr / dirty-random / attn-tie all fire")

    # ---------- (v) base-vs-it differential ----------
    it_dec = decide_drives(0.45, 0.10, 0.01)        # it drives
    base_dec0 = decide_drives(0.03, 0.02, 0.00)     # base ~0 -> INSTALLED
    assert differential_note(it_dec, base_dec0)["category"] == "INSTALLED", differential_note(it_dec, base_dec0)
    base_dec1 = decide_drives(0.22, 0.05, 0.01)     # base also drives, it >> base -> AMPLIFIED
    assert differential_note(it_dec, base_dec1)["category"] == "AMPLIFIED", differential_note(it_dec, base_dec1)
    base_dec2 = decide_drives(0.42, 0.05, 0.01)     # both ~equal -> SHARED
    assert differential_note(it_dec, base_dec2)["category"] == "SHARED", differential_note(it_dec, base_dec2)
    both_low = differential_note(decide_drives(0.05, 0.01, 0.0), decide_drives(0.04, 0.01, 0.0))
    assert both_low["category"] == "NEITHER", both_low
    print("[selftest] (v) differential: INSTALLED / AMPLIFIED / SHARED / NEITHER all fire")

    # ---------- (vi) fit_u recovers the planted cave axis (the real fit path) ----------
    # build a residual-style cache (rc = rn + a*u + noise) and confirm diff-of-means lands on u
    gg = torch.Generator().manual_seed(SEED + 101)
    u_true = torch.randn(d, generator=gg); u_true = u_true / u_true.norm()
    rc_list, rn_list = [], []
    for _ in range(n):
        rn = torch.randn(d, generator=gg)
        rc_list.append(rn + 3.0 * u_true + 0.25 * torch.randn(d, generator=gg))
        rn_list.append(rn)
    u_fit = fit_u(rc_list, rn_list)
    cos = float(torch.nn.functional.cosine_similarity(u_fit, u_true, dim=0))
    assert cos > 0.93, f"fit_u should recover the planted cave axis: cos={cos}"   # ~0.949 w/ synthetic noise; >>random ~1/sqrt(d)
    print(f"[selftest] (vi) fit_u cos(u,u_true)={cos:.3f} OK")
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
