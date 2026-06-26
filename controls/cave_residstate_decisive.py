"""DECISIVE close-of-the-close (PART8 v7): the latent_skeptic correction (wf_f807a702) downgraded
"RLHF moves caving to a NON-ATTENTION/distributed substrate" to OPEN, because the -it inertness was shown
only for ~10 hand-picked heads (span-top5 + DLA-top5) and NO -it intervention had been shown to move the
cave-projection at all (the restoration CHANNEL was unverified at -it -- only the readout AUROC was). This
control runs the four owed measurements, on the SAME matched-union caved set + cave-axis readout as
cave_residstate_close, so the only new thing is the intervention coverage:

  (A) ALL-ATTENTION upper bound -- patch/KO EVERY head (not a top-k), per model. This bounds what the entire
      attention mechanism at the answer position can account for. (= reuse the proven READ attn-KO-to-span +
      WRITE z-patch counter->neutral, with head_set = ALL heads.)
  (B) ALL-MLP upper bound -- patch EVERY layer's mlp_out at the answer position counter->neutral, per model.
      Positive localization for the "distributed/MLP" hypothesis.
  (C) -it POSITIVE CONTROL (channel-live) -- steer the residual at AXIS_LAYER by +/- the axis gap along the
      cave-axis and read the realized OUTPUT first-token margin lp(W*)-lp(C). A signed-monotone response
      (+steer raises W*-preference, -steer lowers it) proves SOME -it residual intervention moves the cave
      behaviour, i.e. the cave-axis is not behaviourally inert at -it (rules out the SyA-overlay reading where
      a separable axis carries no behaviour). If this FAILS, every restoration null at -it is uninformative.
  (D) LABEL-MATCH -- all projections are computed PURELY post-hoc from cached residuals, so each restoration is
      re-read under BOTH the self-judge cave-axis and the realized-argmax cave-axis (the base construct). If the
      -it verdict flips between the two axes, the base<->it contrast was a label/construct artifact.

  Plus bootstrap CIs over the union items on the headline restorations (no in-sample point estimate alone).

ALL HOOKS ARE NAMED `def f(t, hook): ...` (the v5 inline-lambda crash lesson). One model reload per model:
all interventions run in that load and cache the post-intervention answer-slot residual (full vector), so every
projection -- both axes, all restorations -- is pure post-processing.

readout = resid_post[AXIS_LAYER].cave-axis (NOT the emitted token). base label = realized argmax==W*; it label =
free-gen self-judge (same as the close). cave-axis = unit diff-of-means(caved vs not), gated on held-out AUROC.

  python controls/cave_residstate_decisive.py --selftest
  python controls/cave_residstate_decisive.py --base google/gemma-2-9b --it google/gemma-2-9b-it --device cuda --big-pool
"""
import argparse, json, sys, statistics, random
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cave_doubt_write_vs_read import (  # noqa: E402
    MIN_FAITHFUL, RESTORE_THR, GAP, find_subseq, _zname, _ko_heads_to,
)
from cave_faithful_it_diff import _zpatch_hooks  # noqa: E402
from cave_residstate_diff import proj, unit, proj_restoration  # noqa: E402
from spike_eot_cavestate import diff_of_means, heldout_auroc  # noqa: E402
from cave_residstate_close import _measure, AXIS_LAYER, AUROC_THR  # noqa: E402  (reuse the matched-set machinery)

STEER_EPS = 0.10       # min signed-monotone output-margin response (nats) for the -it positive control to PASS
N_BOOT = 2000
BOOT_SEED = 0


# ----------------------------------------------------------------------------- pure
def bootstrap_ci(values, n_boot=N_BOOT, seed=BOOT_SEED, lo=2.5, hi=97.5):
    """Percentile bootstrap CI of the mean of `values`. Pure (lists + seeded RNG). None if empty."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    rng = random.Random(seed)
    means = []
    k = len(vals)
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(k)] for _ in range(k)) / k)
    means.sort()
    def pct(p):
        idx = min(len(means) - 1, max(0, int(round(p / 100.0 * (len(means) - 1)))))
        return means[idx]
    return [round(pct(lo), 6), round(pct(hi), 6)]


def axis_means(rc_at_layer_by_q, labels_by_q, axis):
    """(caved_mean, notcaved_mean) of proj(resid, axis) over items that HAVE a label (a missing label is unknown,
    NOT not-caved -- the realized-argmax arm only labels the union subset). Pure."""
    items = [q for q in labels_by_q if q in rc_at_layer_by_q]
    cav = [proj(rc_at_layer_by_q[q], axis) for q in items if labels_by_q[q]]
    notc = [proj(rc_at_layer_by_q[q], axis) for q in items if not labels_by_q[q]]
    return (statistics.mean(cav) if cav else 0.0), (statistics.mean(notc) if notc else 0.0)


def restoration_mean(union_q, base_resid_by_q, int_resid_by_q, axis, caved_mean, notcaved_mean):
    """Mean projection-restoration over union items for one intervention (cached residuals -> pure)."""
    out = []
    for q in union_q:
        if q not in base_resid_by_q or q not in int_resid_by_q or int_resid_by_q[q] is None:
            continue
        p_ctr = proj(base_resid_by_q[q], axis)
        p_int = proj(int_resid_by_q[q], axis)
        out.append(proj_restoration(p_ctr, p_int, caved_mean, notcaved_mean))
    return (statistics.mean(out) if out else None), out


def decisive_decision(it_all_attn, it_all_mlp, it_rand, base_all_attn,
                      steer_plus_delta, steer_minus_delta, axis_ok):
    """Neutral verdict from the measured numbers only (no hypothesis attached to any model/sign/substrate). Pure.
    Restorations are the max(READ,WRITE) means; deltas are mean output-margin shifts toward W* under +/- steer."""
    def f(x):
        return float(x) if x is not None else 0.0
    iaa, iam, ir, baa = f(it_all_attn), f(it_all_mlp), f(it_rand), f(base_all_attn)
    sp, sm = steer_plus_delta, steer_minus_delta
    chan_live = (sp is not None and sm is not None and (sp - sm) >= STEER_EPS)   # +steer > -steer toward W*
    attn_carry = (iaa - ir) >= GAP and iaa >= RESTORE_THR
    mlp_carry = (iam - ir) >= GAP and iam >= RESTORE_THR
    if not axis_ok:
        cat, msg = "INSUFFICIENT", "cave-axis AUROC gate failed on a model; no trustworthy readout."
    elif not chan_live:
        cat = "CHANNEL_INERT"
        msg = (f"the -it cave-axis does NOT move the output (steer +{f(sp):.3f} / -{f(sm):.3f}, need gap>={STEER_EPS}) "
               f"-> the monitor axis is behaviourally inert at -it (SyA-overlay realized); the restoration nulls "
               f"are UNINFORMATIVE, not evidence of relocation.")
    elif attn_carry and not mlp_carry:
        cat = "ATTENTION_CARRIES"
        msg = (f"with ALL heads patched the -it cave-state restores ({iaa:.3f} vs rand {ir:.3f}) while ALL-MLP does "
               f"not ({iam:.3f}) -> -it caving IS attention-borne; the ~10-head per-head null was a head-SELECTION "
               f"artifact (the carriers are attention heads outside span-top5/DLA-top5).")
    elif mlp_carry and not attn_carry:
        cat = "MLP_CARRIES"
        msg = (f"ALL-MLP restores the -it cave-state ({iam:.3f} vs rand {ir:.3f}) while ALL-attention does not "
               f"({iaa:.3f}) -> RLHF moves caving onto a NON-ATTENTION (MLP) substrate, now positively localized "
               f"(not by elimination).")
    elif attn_carry and mlp_carry:
        cat = "BOTH_REDUNDANT"
        msg = (f"both ALL-attention ({iaa:.3f}) and ALL-MLP ({iam:.3f}) restore the -it cave-state (vs rand {ir:.3f}) "
               f"-> the cave-state is redundantly written by attention AND MLP at the answer position.")
    else:
        cat = "NEITHER_LOCALIZED"
        msg = (f"the -it cave-axis DOES move the output (channel live: +{f(sp):.3f}/-{f(sm):.3f}) but NEITHER "
               f"ALL-attention ({iaa:.3f}) NOR ALL-MLP ({iam:.3f}) output-patch at L{AXIS_LAYER} restores it "
               f"(vs rand {ir:.3f}) -> the cave-state is carried by deeper composition / residual flow, not by "
               f"the answer-position attention-out or mlp-out alone.")
    return {"category": cat, "it_all_attn": round(iaa, 6), "it_all_mlp": round(iam, 6), "it_rand": round(ir, 6),
            "base_all_attn": round(baa, 6), "steer_plus_delta": (round(sp, 6) if sp is not None else None),
            "steer_minus_delta": (round(sm, 6) if sm is not None else None), "channel_live": chan_live,
            "attn_carry": attn_carry, "mlp_carry": mlp_carry, "msg": msg}


# ----------------------------------------------------------------------------- hooks (all named) + real run
def _mlp_patch_hooks(neu_mlp):
    """Patch each layer's mlp_out[0,-1] to the cached NEUTRAL mlp_out (the ALL-MLP write intervention)."""
    hooks = []
    for L, v in neu_mlp.items():
        def f(t, hook, v=v):
            t[0, -1, :] = v.to(t.dtype)
            return t
        hooks.append((f"blocks.{L}.hook_mlp_out", f))
    return hooks


def _steer_hook(u_t, coef):
    """Add coef * unit-axis to resid at the answer position (the sufficiency / channel-live intervention)."""
    def f(t, hook):
        t[0, -1] = t[0, -1] + coef * u_t.to(t.dtype)
        return t
    return f


def _grab_resid(store):
    def f(t, hook):
        store["r"] = t[0, -1].detach().float().cpu().tolist()
        return t
    return f


def _decisive(name, is_chat, device, rec, union_q):
    """ONE reload of `name`; run ALL interventions on the union items, caching the post-intervention answer-slot
    resid_post[AXIS_LAYER] (full vector) so every projection is pure post-processing. Also caches the realized
    argmax (label-match) and the +/- steer output margins (positive control). Returns a dict of cached arrays."""
    from transformer_lens import HookedTransformer
    from job_truthful_flip import PUSH, NEUTRAL
    from rlhf_differential import _helpers
    print(f"[decisive load] {name} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    all_heads = [(L, H) for L in range(nL) for H in range(nH)]
    rl = f"blocks.{AXIS_LAYER}.hook_resid_post"
    recs = rec["_recs"]

    base_resid, attn_read, attn_write, mlp_write = {}, {}, {}, {}
    realized_caved, steer_plus, steer_minus, base_margin = {}, {}, {}, {}
    # axis (this model's primary-label axis) as a tensor for steering; gap = caved-notcaved on cached residuals
    axis = rec["_axis"]
    u_t = torch.tensor(unit(axis), dtype=torch.float32, device=device)
    rc_axis = {q: recs[q]["rc"][AXIS_LAYER] for q in recs}
    prim_lbl = {q: recs[q]["caved"] for q in recs}
    cm, ncm = axis_means(rc_axis, prim_lbl, axis)
    gap_steer = cm - ncm
    print(f"[decisive {rec['tag']}] axis gap (caved-notcaved) = {gap_steer:.4f}", flush=True)

    for q in union_q:
        if q not in recs:
            continue
        r = recs[q]; counter, dpos, zneu = r["_counter"], r["_dpos"], r["_zneu"]
        C, W, aid = r["C"], r["W"], r["aid"]
        cid = first(" " + C)
        base_resid[q] = rc_axis[q]                       # cached counter resid_post[AXIS_LAYER] (no forward)
        # baseline output (argmax for label-match + baseline margin for steer reference)
        with torch.no_grad():
            lg = model(counter)
            lp0 = torch.log_softmax(lg[0, -1].float(), -1)
        realized_caved[q] = 1 if int(lg[0, -1].argmax()) == aid else 0
        base_margin[q] = float(lp0[aid] - lp0[cid])
        # (A) ALL-attention READ (attn-KO every head to the doubt span) + WRITE (z-patch every head ctr->neu)
        if dpos:
            s = {}
            with torch.no_grad():
                model.run_with_hooks(counter, fwd_hooks=_ko_heads_to(all_heads, dpos) + [(rl, _grab_resid(s))])
            attn_read[q] = s.get("r")
        else:
            attn_read[q] = None
        s = {}
        with torch.no_grad():
            model.run_with_hooks(counter, fwd_hooks=_zpatch_hooks(zneu, all_heads) + [(rl, _grab_resid(s))])
        attn_write[q] = s.get("r")
        # (B) ALL-MLP WRITE: cache neutral mlp_out, patch counter mlp_out[-1]<-neutral at every layer
        neutral = push(q, C, NEUTRAL)
        nm = {}
        def grab_mlp(t, hook, nm=nm):
            nm[hook.layer()] = t[0, -1].detach().clone(); return t
        with torch.no_grad():
            model.run_with_hooks(neutral, fwd_hooks=[(f"blocks.{L}.hook_mlp_out", grab_mlp) for L in range(nL)])
        s = {}
        with torch.no_grad():
            model.run_with_hooks(counter, fwd_hooks=_mlp_patch_hooks(nm) + [(rl, _grab_resid(s))])
        mlp_write[q] = s.get("r")
        # (C) positive control: +/- gap steer along the cave-axis at AXIS_LAYER, read output margin lp(W*)-lp(C)
        with torch.no_grad():
            lgp = model.run_with_hooks(counter, fwd_hooks=[(rl, _steer_hook(u_t, +gap_steer))])
            lgm = model.run_with_hooks(counter, fwd_hooks=[(rl, _steer_hook(u_t, -gap_steer))])
            lpp = torch.log_softmax(lgp[0, -1].float(), -1); lpm = torch.log_softmax(lgm[0, -1].float(), -1)
        steer_plus[q] = float(lpp[aid] - lpp[cid]) - base_margin[q]
        steer_minus[q] = float(lpm[aid] - lpm[cid]) - base_margin[q]
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"tag": rec["tag"], "base_resid": base_resid, "attn_read": attn_read, "attn_write": attn_write,
            "mlp_write": mlp_write, "realized_caved": realized_caved, "self_caved": prim_lbl,
            "steer_plus": steer_plus, "steer_minus": steer_minus, "rc_axis": rc_axis, "gap_steer": gap_steer}


def _arm(cache, union_q, label_key):
    """All restorations under the cave-axis fitted to `label_key` ('self_caved' or 'realized_caved'), over the
    items that carry that label. Pure over the dumped cache (no model / no recs) -> offline-reprocessable."""
    labels_by_q = cache[label_key]
    rc_axis = cache["rc_axis"]
    items = [q for q in rc_axis if q in labels_by_q]
    vecs = [rc_axis[q] for q in items]
    labs = [labels_by_q[q] for q in items]
    ncav = sum(labs)
    if ncav < 3 or len(labs) - ncav < 3:
        return {"label": label_key, "auroc": None, "axis_ok": False, "n": len(labs), "ncav": ncav}
    au, _ = heldout_auroc(vecs, labs)
    axis = unit(diff_of_means(vecs, labs))
    cm, ncm = axis_means(rc_axis, labels_by_q, axis)
    ar_m, ar_l = restoration_mean(union_q, cache["base_resid"], cache["attn_read"], axis, cm, ncm)
    aw_m, aw_l = restoration_mean(union_q, cache["base_resid"], cache["attn_write"], axis, cm, ncm)
    mw_m, mw_l = restoration_mean(union_q, cache["base_resid"], cache["mlp_write"], axis, cm, ncm)
    all_attn = max([x for x in [ar_m, aw_m] if x is not None], default=None)
    return {"label": label_key, "auroc": (round(au, 4) if au is not None else None),
            "axis_ok": (au is not None and au >= AUROC_THR),
            "attn_read": (round(ar_m, 6) if ar_m is not None else None),
            "attn_write": (round(aw_m, 6) if aw_m is not None else None),
            "mlp_write": (round(mw_m, 6) if mw_m is not None else None),
            "all_attn": (round(all_attn, 6) if all_attn is not None else None),
            "all_attn_ci": bootstrap_ci((ar_l or []) + (aw_l or [])), "mlp_ci": bootstrap_ci(mw_l or []),
            "n": len(labs), "ncav": ncav}


def run(base_name, it_name, device, big_pool):
    from cave_copy_confidence_conditional import _build_pool
    pool = _build_pool(big_pool=big_pool)
    # 1) matched-set + axis machinery (reuse the close's _measure: 1 load per model)
    base = _measure(base_name, False, device, pool)
    it = _measure(it_name, True, device, pool)
    union_q = sorted(set(base["_caved_q"]) | set(it["_caved_q"]))
    print(f"[decisive] union caved set n={len(union_q)} (base {len(base['_caved_q'])} | it {len(it['_caved_q'])})", flush=True)
    out = {"base": base_name, "it": it_name, "axis_layer": AXIS_LAYER, "n_union": len(union_q),
           "base_aurocs": base["aurocs"], "it_aurocs": it["aurocs"]}
    if not (base["axis_ok"] and it["axis_ok"] and len(union_q) >= MIN_FAITHFUL and base["_axis"] and it["_axis"]):
        out["decision"] = {"category": "INSUFFICIENT",
                           "msg": f"axis_ok base/it={base['axis_ok']}/{it['axis_ok']}, union {len(union_q)} (need>={MIN_FAITHFUL})"}
        Path("out").mkdir(exist_ok=True)
        Path("out/cave_residstate_decisive.json").write_text(json.dumps(out, indent=2, default=str))
        print(f"[DECISIVE] {out['decision']['category']}: {out['decision']['msg']}", flush=True)
        return
    # 2) decisive batteries (1 reload per model; all interventions + cached residuals; nL/nH read from model.cfg)
    base_c = _decisive(base_name, False, device, base, union_q)
    it_c = _decisive(it_name, True, device, it, union_q)
    # dump the raw GPU cache BEFORE any post-processing -> a pure-Python bug below can be reprocessed OFFLINE
    # (no GPU re-run). All values are JSON-safe lists/floats/ints.
    Path("out").mkdir(exist_ok=True)
    Path("out/cave_residstate_decisive_cache.json").write_text(
        json.dumps({"union_q": union_q, "base_c": base_c, "it_c": it_c}, default=str))
    # 3) arms: primary-label + label-match, per model (PURE over the dumped cache)
    base_self = _arm(base_c, union_q, "self_caved")
    base_real = _arm(base_c, union_q, "realized_caved")
    it_self = _arm(it_c, union_q, "self_caved")
    it_real = _arm(it_c, union_q, "realized_caved")
    # Floor for the carry thresholds: a no-op intervention restores 0 by construction, and the close already
    # established a matched-random-head WRITE floor ~0.01 (results_residstate_close). The binding gate is
    # RESTORE_THR (0.2) >> any plausible random-head restoration, so the verdict is insensitive to the exact floor;
    # use 0.0 (conservative for the carry branches: makes ATTENTION/MLP_CARRIES no easier than the 0.2 threshold).
    it_rand = 0.0
    # steer deltas (positive control), it primary
    sp = statistics.mean([v for v in it_c["steer_plus"].values()]) if it_c["steer_plus"] else None
    sm = statistics.mean([v for v in it_c["steer_minus"].values()]) if it_c["steer_minus"] else None
    sp_base = statistics.mean([v for v in base_c["steer_plus"].values()]) if base_c["steer_plus"] else None
    sm_base = statistics.mean([v for v in base_c["steer_minus"].values()]) if base_c["steer_minus"] else None
    out.update({"base_self": base_self, "base_real": base_real, "it_self": it_self, "it_real": it_real,
                "it_steer_plus": (round(sp, 6) if sp is not None else None),
                "it_steer_minus": (round(sm, 6) if sm is not None else None),
                "base_steer_plus": (round(sp_base, 6) if sp_base is not None else None),
                "base_steer_minus": (round(sm_base, 6) if sm_base is not None else None),
                "it_gap_steer": round(it_c["gap_steer"], 6), "base_gap_steer": round(base_c["gap_steer"], 6),
                "label_match_changes_verdict": None})
    # 4) headline decision (it primary-label axis) + label-match cross-check. Use .get(): an arm that early-exits
    #    (axis can't be fit -- e.g. -it realized-argmax has too few W*-emissions) lacks the restoration keys.
    dec_self = decisive_decision(it_self.get("all_attn"), it_self.get("mlp_write"), it_rand, base_self.get("all_attn"),
                                 sp, sm, it_self["axis_ok"] and base_self["axis_ok"])
    dec_real = decisive_decision(it_real.get("all_attn"), it_real.get("mlp_write"), it_rand, base_real.get("all_attn"),
                                 sp, sm, it_real["axis_ok"] and base_real["axis_ok"])
    out["label_match_changes_verdict"] = (dec_self["category"] != dec_real["category"])
    out["decision"] = dec_self
    out["decision_labelmatch"] = dec_real
    Path("out").mkdir(exist_ok=True)
    Path("out/cave_residstate_decisive.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[DECISIVE] {dec_self['category']}: {dec_self['msg']}", flush=True)
    print(f"[DECISIVE] label-match (argmax-axis) -> {dec_real['category']} "
          f"(verdict changes: {out['label_match_changes_verdict']})", flush=True)
    print(f"[DECISIVE] it_self={it_self} | it_real={it_real}", flush=True)
    print(f"[DECISIVE] steer it +{sp}/-{sm} | base +{sp_base}/-{sm_base}", flush=True)
    print("[done] wrote out/cave_residstate_decisive.json", flush=True)


def selftest():
    # bootstrap_ci
    assert bootstrap_ci([0.5, 0.5, 0.5]) == [0.5, 0.5]
    ci = bootstrap_ci([0.0, 1.0] * 50); assert ci[0] < 0.5 < ci[1], ci
    assert bootstrap_ci([]) is None
    # axis_means + restoration_mean
    rc = {"a": [2.0, 0.0], "b": [0.0, 0.0]}; lab = {"a": 1, "b": 0}; ax = [1.0, 0.0]
    cm, ncm = axis_means(rc, lab, ax); assert abs(cm - 2.0) < 1e-9 and abs(ncm) < 1e-9, (cm, ncm)
    base = {"a": [2.0, 0.0]}; inter = {"a": [0.0, 0.0]}
    m, lst = restoration_mean(["a"], base, inter, ax, 2.0, 0.0); assert abs(m - 1.0) < 1e-9, m
    # decisive_decision branches
    d_attn = decisive_decision(0.40, 0.02, 0.01, 0.38, 0.5, -0.4, True); assert d_attn["category"] == "ATTENTION_CARRIES", d_attn
    d_mlp = decisive_decision(0.02, 0.40, 0.01, 0.38, 0.5, -0.4, True); assert d_mlp["category"] == "MLP_CARRIES", d_mlp
    d_both = decisive_decision(0.40, 0.40, 0.01, 0.38, 0.5, -0.4, True); assert d_both["category"] == "BOTH_REDUNDANT", d_both
    d_none = decisive_decision(0.02, 0.02, 0.01, 0.38, 0.5, -0.4, True); assert d_none["category"] == "NEITHER_LOCALIZED", d_none
    d_inert = decisive_decision(0.02, 0.02, 0.01, 0.38, 0.05, 0.02, True); assert d_inert["category"] == "CHANNEL_INERT", d_inert
    d_ins = decisive_decision(0.40, 0.02, 0.01, 0.38, 0.5, -0.4, False); assert d_ins["category"] == "INSUFFICIENT", d_ins
    print("[selftest] bootstrap_ci + axis_means + restoration_mean + decisive_decision (6 branches) PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--base", default="google/gemma-2-9b")
    p.add_argument("--it", default="google/gemma-2-9b-it")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--big-pool", action="store_true")
    a = p.parse_args()
    selftest() if a.selftest else run(a.base, a.it, a.device, a.big_pool)


if __name__ == "__main__":
    main()
