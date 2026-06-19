"""NEXT-1 (direction prong) -- does the distributed caving head-set reduce to a single low-rank
"cave/defer" DIRECTION in the -it residual? (RESEARCH_QUESTIONS Part 4 NEXT-1; the function-vector view.)

CONTEXT. `headset_joint_patch.py` showed the 9b-it misconception cave is carried by a distributed,
SPECIFIC, additive head SET (18 sub-threshold heads jointly restore 0.36-0.45 of the cave; matched-random
~0), base-present and RLHF-amplified. The set-prong and the direction-prong of NEXT-1 "may be the same
object" -- a coordinated set that writes one low-rank direction IS a function/task vector (Todd 2023;
Hendel 2023). This prong tests that: fit the direction, check it is necessary + sufficient + low-rank,
and check the head-set WRITES it (unification).

METHOD (rank-1 diff-of-means steering/function vector, the standard construction):
  cave_dir(L) = mean_items( resid_post[L][-1](counter) - resid_post[L][-1](neutral_turn) )
  -- counter = the caved state (low M), neutral_turn = the R-4 control (high M); the difference points
  toward caving. u = cave_dir / ||cave_dir||. Fit over a small layer sweep in the set's output range.
Metric M = logp(C) - logp(W*) first-token margin (same as the set prong / _confirm).
  NECESSITY  (ablate): on counter, set the u-projection to its neutral mean -> M recovers?
      frac_nec = (M_ablate - M_counter) / (M_neutral - M_counter).
  SUFFICIENCY (steer): on neutral, set the u-projection to its counter mean -> M caves?
      frac_suf = (M_steer  - M_neutral) / (M_counter - M_neutral).
  LOW-RANK: SVD of the centered difference matrix; top-PC variance fraction (rank-1 justified if high).
  SPECIFICITY: a random unit direction (same magnitude shift) must NOT recover/induce.
  BASE DIFFERENTIAL: same fit+ablate on base -> install (base~0) vs amplify (base present), mirroring
      the set result.
  UNIFICATION: joint-patch the head SET (counter z->neutral z, reusing headset_joint_patch._patch_set's
      hook) and read the change it makes to resid_post[L][-1]; cosine(set_write, u). |cos|~1 => the set
      writes the direction => set and direction are the same distributed object.

PRE-REGISTERED DECISION (decide_direction, model-free --selftest):
  FUNCTION-VECTOR FOUND  iff necessary (frac_nec_it >= DIR_THR, random ~0) AND sufficient (frac_suf_it
      >= DIR_THR) AND low-rank (top-PC frac >= LOWRANK_MIN) AND the set writes it (|cos| >= COS_THR).
      -> + "installed" (base frac_nec ~0) or "amplified" (base present), mirroring the set prong.
  Guards: NECESSARY-NOT-SUFFICIENT (downstream, not causal); DIRECTION-SEPARATE-FROM-SET (mediates but
      set doesn't write it); NO-LOW-RANK-DIRECTION (rank-1 doesn't mediate -> higher-rank/non-linear).

Forward-only (diff-of-means + projection edits + the set hook; no backward) -> fits the 40GB A100.

  python headset_direction.py --selftest
  python headset_direction.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

from rlhf_differential import ITEMS, _helpers, _logp_diff, INSTALL_THR, BASE_FLOOR, MIN_EFFECT_NET
from job_truthful_flip import PUSH, NEUTRAL
from atp_low_confirm import HEADS, NH_9B
from headset_joint_patch import _zname

FIT_LAYERS = [24, 28, 32, 36]   # sweep in the set's output range (set heads span L21-L34)
DIR_THR = 0.20                  # frac of the cave a single direction must mediate to count (~half the set)
COS_THR = 0.50                  # |cos(set_write, u)| for set<->direction unification
LOWRANK_MIN = 0.50              # top-PC variance fraction for "low-rank"


# --------------------------------------------------------------------------- pure decision
def toppc_frac(diffs):
    """Fraction of variance in the top principal component of the centered difference matrix [n,d].
    High => the per-item cave shifts are ~rank-1 (a single shared direction). Pure (torch in, float out)."""
    X = diffs - diffs.mean(0, keepdim=True)
    s = torch.linalg.svdvals(X.float())
    tot = float((s * s).sum())
    return float(s[0] * s[0]) / tot if tot > 0 else 0.0


def decide_direction(nec_it, nec_base, suf_it, rand_nec_it, lowrank_frac, set_cos,
                     dir_thr=DIR_THR, base_floor=BASE_FLOOR, cos_thr=COS_THR, lowrank_min=LOWRANK_MIN):
    """Pure decision over the best-layer direction statistics."""
    mediates = nec_it >= dir_thr and (rand_nec_it is None or rand_nec_it < base_floor)
    sufficient = suf_it >= dir_thr
    lowrank = lowrank_frac >= lowrank_min
    unified = set_cos is not None and abs(set_cos) >= cos_thr
    base_clean = nec_base is not None and nec_base <= base_floor
    install_tag = "RLHF-INSTALLED (base ~0)" if base_clean else "base-present, RLHF-AMPLIFIED"
    unif_tag = (f"the head-set WRITES it (|cos| {abs(set_cos):.2f})" if unified else
                f"the head-set does NOT write it (|cos| "
                f"{abs(set_cos) if set_cos is not None else float('nan'):.2f} < {cos_thr}; set-orthogonal -> distinct loci)")
    if mediates and sufficient and lowrank and unified:
        v = (f"FUNCTION VECTOR FOUND: a low-rank cave direction is necessary (ablate recovers "
             f"{nec_it:.3f}) AND sufficient (steer induces {suf_it:.3f}) AND low-rank (top-PC "
             f"{lowrank_frac:.2f}) AND {unif_tag} -- the set and the direction are the same distributed "
             f"object [{install_tag}].")
    elif mediates and sufficient and lowrank and not unified:
        v = (f"LOW-RANK CAVE DIRECTION (necessary {nec_it:.3f} + sufficient {suf_it:.3f}, top-PC "
             f"{lowrank_frac:.2f}) but {unif_tag} [{install_tag}].")
    elif mediates and sufficient and not lowrank:
        v = (f"CAVE SUBSPACE (NOT a single function vector): a residual direction is necessary (ablate "
             f"recovers {nec_it:.3f}) AND sufficient (steer induces {suf_it:.3f}) AND specific (random "
             f"~0) -- but HIGHER-RANK (top-PC {lowrank_frac:.2f} < {lowrank_min}), so caving lives in a "
             f"multi-dim residual subspace, not a line; and {unif_tag} [{install_tag}].")
    elif mediates and not sufficient:
        v = (f"NECESSARY NOT SUFFICIENT: ablating the direction recovers ({nec_it:.3f}) but steering it "
             f"does not induce the cave ({suf_it:.3f} < {dir_thr}) -- the direction is downstream of the "
             f"cause, not itself the lever.")
    elif not mediates:
        v = (f"NO LOW-RANK DIRECTION: ablating the rank-1 cave direction does not recover the cave "
             f"(nec {nec_it:.3f} < {dir_thr}, or random control not clean) -- 9b caving is higher-rank "
             f"or non-linear in the residual; the distributed set is not a single function vector.")
    else:
        v = (f"PARTIAL: mediates={mediates} sufficient={sufficient} lowrank={lowrank} unified={unified}.")
    return {"function_vector": bool(mediates and sufficient and lowrank and unified),
            "mediates": mediates, "sufficient": sufficient, "low_rank": lowrank, "unified": unified,
            "base_clean_installed": base_clean,
            "frac_nec_it": round(nec_it, 4), "frac_nec_base": (round(nec_base, 4) if nec_base is not None else None),
            "frac_suf_it": round(suf_it, 4), "rand_nec_it": (round(rand_nec_it, 4) if rand_nec_it is not None else None),
            "top_pc_frac": round(lowrank_frac, 4), "set_cosine": (round(set_cos, 4) if set_cos is not None else None),
            "verdict": v}


# --------------------------------------------------------------------------- real run
def _rname(L):
    return f"blocks.{L}.hook_resid_post"


def _set_write(model, counter, zneu, heads, fit_layers, nH):
    """Joint-patch the head SET (counter z[-1]->neutral z[-1] for all heads) and return the change it
    makes to resid_post[L][-1] at each fit layer: delta[L] = resid_post(set-patched) - resid_post(counter)."""
    by_layer = {}
    for (L, H) in heads:
        by_layer.setdefault(L, []).append(H)
    def mkz(L, Hs):
        def patch(z, hook, L=L, Hs=Hs):
            for H in Hs:
                z[0, -1, H, :] = zneu[L][H].to(z.dtype)
            return z
        return patch
    store = {}
    def grab(r, hook):
        store[hook.layer()] = r[0, -1].detach().float(); return r
    hooks = [(_zname(L), mkz(L, Hs)) for L, Hs in by_layer.items()]
    hooks += [(_rname(L), grab) for L in fit_layers]
    with torch.no_grad():
        model.run_with_hooks(counter, fwd_hooks=hooks)
    return store    # resid_post[-1] WITH the set patched


def _dir_pass(name, is_chat, device, fit_layers, heads, nH):
    from transformer_lens import HookedTransformer
    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    g = torch.Generator(device="cpu").manual_seed(0)

    ctxs = []
    for it in ITEMS:
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
            M_ctr = float(_logp_diff(model.run_with_hooks(counter, fwd_hooks=[(_rname(L), grab_c) for L in fit_layers]), cid, aid))
            M_neu = float(_logp_diff(model.run_with_hooks(neutral, fwd_hooks=[(_rname(L), grab_n) for L in fit_layers]), cid, aid))
        gap = M_neu - M_ctr
        if abs(gap) < MIN_EFFECT_NET:
            continue
        ctxs.append({"counter": counter, "neutral": neutral, "cid": cid, "aid": aid,
                     "rc": rc, "rn": rn, "M_ctr": M_ctr, "M_neu": M_neu, "gap": gap})
        print(f"  [{'it' if is_chat else 'base'}] gap={gap:+.2f} q={q[:40]!r}", flush=True)

    n = len(ctxs)
    per_layer = {}
    for L in fit_layers:
        D = torch.stack([c["rc"][L] - c["rn"][L] for c in ctxs])          # [n, d]
        cave = D.mean(0)
        u = cave / (cave.norm() + 1e-8)
        proj_n = statistics.mean(float(c["rn"][L] @ u) for c in ctxs)      # neutral mean projection
        proj_c = statistics.mean(float(c["rc"][L] @ u) for c in ctxs)      # counter mean projection
        # random unit direction, matched magnitude (cpu generator -> move to the residual's device)
        rnd = torch.randn(u.shape, generator=g).to(u.dtype).to(u.device)
        ur = rnd / (rnd.norm() + 1e-8)
        prn = statistics.mean(float(c["rn"][L] @ ur) for c in ctxs)
        prc = statistics.mean(float(c["rc"][L] @ ur) for c in ctxs)
        per_layer[L] = {"u": u.to(device), "ur": ur.to(device), "proj_n": proj_n, "proj_c": proj_c,
                        "prn": prn, "prc": prc, "toppc": toppc_frac(D)}

    # interventions + set-write, per item
    nec = {L: [] for L in fit_layers}; suf = {L: [] for L in fit_layers}
    rnec = {L: [] for L in fit_layers}; cos = {L: [] for L in fit_layers}
    # neutral z for the set-write (grab once per item)
    nL = model.cfg.n_layers
    for c in ctxs:
        cid, aid, gap = c["cid"], c["aid"], c["gap"]
        # neutral attn-z readout for the set patch
        zneu = {}
        def grabz(z, hook):
            zneu[hook.layer()] = z[0, -1].detach().clone(); return z
        with torch.no_grad():
            model.run_with_hooks(c["neutral"], fwd_hooks=[(_zname(L), grabz) for L in range(nL)], return_type=None)
        sw = _set_write(model, c["counter"], zneu, heads, fit_layers, nH)
        for L in fit_layers:
            pl = per_layer[L]; u = pl["u"]; ur = pl["ur"]
            rc, rn = c["rc"][L].to(device), c["rn"][L].to(device)
            # NECESSITY: on counter, move u-projection to the neutral mean
            shift = (pl["proj_n"] - float(rc @ u))
            def ab(r, hook, u=u, shift=shift):
                r[0, -1] = r[0, -1] + (shift * u).to(r.dtype); return r
            # SUFFICIENCY: on neutral, move u-projection to the counter mean
            ssh = (pl["proj_c"] - float(rn @ u))
            def st(r, hook, u=u, ssh=ssh):
                r[0, -1] = r[0, -1] + (ssh * u).to(r.dtype); return r
            # RANDOM control on counter, matched shift magnitude along ur
            rsh = (pl["prn"] - float(rc @ ur))
            def ab_r(r, hook, ur=ur, rsh=rsh):
                r[0, -1] = r[0, -1] + (rsh * ur).to(r.dtype); return r
            with torch.no_grad():
                M_ab = float(_logp_diff(model.run_with_hooks(c["counter"], fwd_hooks=[(_rname(L), ab)]), cid, aid))
                M_st = float(_logp_diff(model.run_with_hooks(c["neutral"], fwd_hooks=[(_rname(L), st)]), cid, aid))
                M_rr = float(_logp_diff(model.run_with_hooks(c["counter"], fwd_hooks=[(_rname(L), ab_r)]), cid, aid))
            nec[L].append((M_ab - c["M_ctr"]) / gap)
            suf[L].append((M_st - c["M_neu"]) / (c["M_ctr"] - c["M_neu"]))
            rnec[L].append((M_rr - c["M_ctr"]) / gap)
            delta = sw[L].to(device) - rc                                  # set's write to resid_post[-1]
            cos[L].append(float(torch.nn.functional.cosine_similarity(delta, u, dim=0)))

    out = {"n_ok": n, "layers": {}}
    for L in fit_layers:
        out["layers"][L] = {
            "frac_nec": round(statistics.mean(nec[L]), 4) if nec[L] else None,
            "frac_suf": round(statistics.mean(suf[L]), 4) if suf[L] else None,
            "rand_nec": round(statistics.mean(rnec[L]), 4) if rnec[L] else None,
            "set_cos": round(statistics.mean(cos[L]), 4) if cos[L] else None,
            "top_pc_frac": round(per_layer[L]["toppc"], 4)}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def run(name_base, name_it):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    it = _dir_pass(name_it, True, device, FIT_LAYERS, HEADS, NH_9B)
    base = _dir_pass(name_base, False, device, FIT_LAYERS, HEADS, NH_9B)
    # headline layer = the -it layer with the largest necessity recovery
    bestL = max(FIT_LAYERS, key=lambda L: (it["layers"][L]["frac_nec"] or -9))
    li, lb = it["layers"][bestL], base["layers"][bestL]
    dec = decide_direction(li["frac_nec"], lb["frac_nec"], li["frac_suf"], li["rand_nec"],
                           li["top_pc_frac"], li["set_cos"])
    out = {"model_base": name_base, "model_it": name_it, "cue": "headset_cave_direction",
           "substrate": "I1 9b-it caving items (TruthfulQA misconceptions)",
           "fit_layers": FIT_LAYERS, "headline_layer": bestL,
           "thresholds": {"dir_thr": DIR_THR, "base_floor": BASE_FLOOR, "cos_thr": COS_THR, "lowrank_min": LOWRANK_MIN},
           "it_n_ok": it["n_ok"], "base_n_ok": base["n_ok"],
           "it_layers": it["layers"], "base_layers": base["layers"], "decision": dec}
    Path("out").mkdir(exist_ok=True)
    Path("out/headset_direction_9b.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[headline layer] L{bestL}")
    print("[it layers]", json.dumps(it["layers"], indent=2, default=str))
    print("[verdict]", dec["verdict"])
    print("[done] wrote out/headset_direction_9b.json")


# --------------------------------------------------------------------------- selftest (model-free)
def selftest():
    # toppc_frac: a rank-1 difference matrix (all rows ~ one direction) -> top-PC fraction ~1
    d = torch.tensor([3.0, 0.0, 0.0])
    M = torch.stack([d, 1.1 * d, 0.9 * d, 1.05 * d])
    assert toppc_frac(M) > 0.99, toppc_frac(M)
    # an isotropic matrix -> top-PC fraction small (well below LOWRANK_MIN for d=3 ~0.33-ish)
    g = torch.Generator().manual_seed(1)
    assert toppc_frac(torch.randn(40, 8, generator=g)) < 0.5
    print(f"[selftest] toppc_frac: rank-1={toppc_frac(M):.3f} isotropic<0.5 OK")

    # FUNCTION VECTOR: necessary + sufficient + low-rank + set writes it, base ~0 (installed)
    f = decide_direction(0.40, 0.03, 0.35, 0.01, 0.85, -0.92)
    assert f["function_vector"] and f["verdict"].startswith("FUNCTION VECTOR FOUND"), f
    assert "INSTALLED" in f["verdict"], f
    print("[selftest] FUNCTION VECTOR (installed) fires")

    # amplified variant: base present
    fa = decide_direction(0.40, 0.20, 0.35, 0.01, 0.85, -0.92)
    assert fa["function_vector"] and "AMPLIFIED" in fa["verdict"], fa
    print("[selftest] FUNCTION VECTOR (amplified) fires")

    # DIRECTION SEPARATE FROM SET: mediates+sufficient+lowrank but set doesn't write it
    s = decide_direction(0.40, 0.03, 0.35, 0.01, 0.85, 0.10)
    assert (not s["function_vector"]) and s["verdict"].startswith("LOW-RANK CAVE DIRECTION"), s
    print("[selftest] DIRECTION-SEPARATE-FROM-SET fires")

    # CAVE SUBSPACE: necessary+sufficient+specific but higher-rank and set-orthogonal (the real result)
    cs = decide_direction(0.50, 0.16, 0.26, 0.002, 0.33, -0.037)
    assert (not cs["function_vector"]) and cs["verdict"].startswith("CAVE SUBSPACE"), cs
    assert "AMPLIFIED" in cs["verdict"] and "distinct loci" in cs["verdict"], cs
    print("[selftest] CAVE-SUBSPACE (higher-rank, set-orthogonal, amplified) fires")

    # NECESSARY NOT SUFFICIENT
    ns = decide_direction(0.40, 0.03, 0.05, 0.01, 0.85, -0.92)
    assert ns["verdict"].startswith("NECESSARY NOT SUFFICIENT"), ns
    print("[selftest] NECESSARY-NOT-SUFFICIENT fires")

    # NO LOW-RANK DIRECTION: ablation doesn't recover
    nz = decide_direction(0.05, 0.0, 0.35, 0.01, 0.85, -0.92)
    assert nz["verdict"].startswith("NO LOW-RANK DIRECTION"), nz
    # also fires when random control not clean
    nz2 = decide_direction(0.40, 0.0, 0.35, 0.30, 0.85, -0.92)
    assert nz2["verdict"].startswith("NO LOW-RANK DIRECTION"), nz2
    print("[selftest] NO-LOW-RANK-DIRECTION fires (low nec; and dirty random control)")
    print("[selftest] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--name-base", default="google/gemma-2-9b")
    ap.add_argument("--name-it", default="google/gemma-2-9b-it")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a.name_base, a.name_it)
