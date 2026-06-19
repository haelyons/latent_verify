"""Matched-item de-confound for the NEXT-1 "amplified-not-installed" claim (the shared, load-bearing
crux of BOTH prongs). RESEARCH_QUESTIONS Part 4.

WHY. headset_joint_patch (set) and headset_direction (direction) each gate items per-model with
`abs(gap) < MIN_EFFECT_NET: continue` -- so -it kept 10 items and base kept 9, a DIFFERENT subset each,
and both included negative-gap "anti-cave" items (|gap| gate, not sign-restricted). The central claim
"base-present but RLHF-AMPLIFIED ~3-4x, not installed" compares joint_it/nec_it vs joint_base/nec_base
across those mismatched subsets -- not apples-to-apples. This control removes that confound by measuring
both loci on the SAME, sign-restricted caving intersection.

PROCEDURE.
  1. gap-pass each model: per item gap = M(neutral_turn) - M(counter), M = logp(C) - logp(W*) first-token.
  2. matched set = items with base_gap > +MIN_EFF AND it_gap > +MIN_EFF (a real cave on BOTH models,
     positive sign only -- drops the anti-cave items the |gap| gate let through).
  3. on the matched set, re-measure both loci, base vs it:
       SET   : joint activation-patch of the full top-K diffuse head set (reuse headset_joint_patch._patch_set)
               -> joint frac of the cave restored.
       DIR   : fit the rank-1 cave direction (diff-of-means) at the headline layer L28, ablate the
               u-projection to its neutral mean -> necessity frac.
  4. per locus, decide INSTALLED (base ~0) / AMPLIFIED (it >= AMP_RATIO x base) / BASE-SHARED / NO-EFFECT,
     now on matched items. Overall: does "amplified-not-installed" survive the de-confound?

Forward-only -> fits the 40GB A100. Reuses the verified primitives; the only new logic is the matched-set
intersection (pure, selftested) and the amplification decision (pure, selftested).

  python matched_item_deconfound.py --selftest
  python matched_item_deconfound.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

from rlhf_differential import ITEMS, _helpers, _logp_diff, INSTALL_THR, BASE_FLOOR
from misconception_pool import ITEMS_WIDE
from job_truthful_flip import PUSH, NEUTRAL
from atp_low_confirm import HEADS, NH_9B
from headset_joint_patch import _zname, _patch_set
from headset_direction import _rname

MIN_EFF = 0.5        # sign-restricted: a real positive cave (neutral margin exceeds counter by >0.5 nat)
DIR_LAYER = 28       # headline layer from the direction prong (largest necessity)
AMP_RATIO = 2.0      # it-effect must be >= this x base-effect to read "amplified" (vs "base-shared")


# --------------------------------------------------------------------------- pure logic
def matched(base_gaps, it_gaps, min_eff=MIN_EFF):
    """Item indices with a real positive cave on BOTH models. Pure (dict,dict -> sorted list)."""
    return sorted(i for i in base_gaps if i in it_gaps and base_gaps[i] > min_eff and it_gaps[i] > min_eff)


def bootstrap_ci(values, seed=0, n_boot=2000, lo=2.5, hi=97.5):
    """Percentile bootstrap CI of the mean. Pure (list -> dict). Deterministic via random.Random(seed)."""
    import random as _r
    n = len(values)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = _r.Random(seed)
    means = []
    for _ in range(n_boot):
        s = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    def pct(p):
        return means[min(n_boot - 1, max(0, int(round(p / 100 * (n_boot - 1)))))]
    return {"mean": round(sum(values) / n, 4), "lo": round(pct(lo), 4), "hi": round(pct(hi), 4), "n": n}


def decide_amp(it_eff, base_eff, install_thr=INSTALL_THR, base_floor=BASE_FLOOR, amp_ratio=AMP_RATIO):
    """Per-locus classification on the matched set."""
    ratio = (it_eff / base_eff) if base_eff and base_eff > 1e-6 else float("inf")
    if it_eff < install_thr:
        tag, msg = "NO_EFFECT", f"it {it_eff:.3f} < {install_thr} on matched items (no cave restored)"
    elif base_eff <= base_floor:
        tag, msg = "INSTALLED", f"it {it_eff:.3f} >= {install_thr}, base {base_eff:.3f} <= {base_floor} (~0)"
    elif ratio >= amp_ratio:
        tag, msg = "AMPLIFIED", f"it {it_eff:.3f} >= {amp_ratio}x base {base_eff:.3f} (ratio {ratio:.1f})"
    else:
        tag, msg = "BASE_SHARED", f"it {it_eff:.3f} ~ base {base_eff:.3f} (ratio {ratio:.1f} < {amp_ratio})"
    return {"tag": tag, "it_eff": round(it_eff, 4), "base_eff": round(base_eff, 4),
            "ratio": (round(ratio, 2) if ratio != float("inf") else None), "msg": msg}


def overall(set_dec, dir_dec):
    amp = {"INSTALLED", "AMPLIFIED"}
    if set_dec["tag"] in amp and dir_dec["tag"] in amp:
        v = (f"AMPLIFIED-NOT-INSTALLED SURVIVES the matched-item de-confound: set={set_dec['tag']} "
             f"({set_dec['msg']}); direction={dir_dec['tag']} ({dir_dec['msg']}). The multi-locus claim holds "
             f"apples-to-apples.")
    elif set_dec["tag"] == "BASE_SHARED" and dir_dec["tag"] == "BASE_SHARED":
        v = (f"CLAIM WEAKENED: on matched items both loci read BASE-SHARED (set {set_dec['msg']}; dir "
             f"{dir_dec['msg']}) -- the it>>base gap was an item-mismatch artifact; the loci are base "
             f"mechanisms, not RLHF-amplified.")
    else:
        v = (f"MIXED on matched items: set={set_dec['tag']} ({set_dec['msg']}); direction={dir_dec['tag']} "
             f"({dir_dec['msg']}). The amplified-not-installed claim holds for some loci, not all.")
    return {"set": set_dec, "direction": dir_dec, "verdict": v}


# --------------------------------------------------------------------------- real passes
def _gap_pass(name, is_chat, device, pool):
    """Forward-only: per-item gap = M(neutral) - M(counter). No interventions, no big-tensor caching."""
    from transformer_lens import HookedTransformer
    print(f"[gap] load {name} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    gaps = {}
    for i, it in enumerate(pool):
        q, C, W = it["q"], it["correct"], it["Wstar"]
        cid, aid = first(" " + C), first(" " + W)
        counter = push(q, C, PUSH["counter"].format(W=W))
        neutral = push(q, C, NEUTRAL)
        with torch.no_grad():
            M_ctr = float(_logp_diff(model(counter), cid, aid))
            M_neu = float(_logp_diff(model(neutral), cid, aid))
        gaps[i] = M_neu - M_ctr
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return gaps


def _effect_pass(name, is_chat, device, idxs, heads, nH, dir_layer, pool):
    """On the matched items only: SET joint frac + DIRECTION necessity frac at dir_layer. One model load.
    Also returns per-item set fracs + raw nat components (triage: per-item dispersion + readout-artifact)."""
    from transformer_lens import HookedTransformer
    print(f"[effect] load {name} on {len(idxs)} matched items", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    nL = model.cfg.n_layers
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    ctx = []
    set_fracs = []
    for i in idxs:
        it = pool[i]
        q, C, W = it["q"], it["correct"], it["Wstar"]
        cid, aid = first(" " + C), first(" " + W)
        counter = push(q, C, PUSH["counter"].format(W=W))
        neutral = push(q, C, NEUTRAL)
        zneu, rc, rn = {}, {}, {}
        def gz(z, hook):
            zneu[hook.layer()] = z[0, -1].detach().clone(); return z
        def gr_c(r, hook):
            rc[hook.layer()] = r[0, -1].detach().float(); return r
        def gr_n(r, hook):
            rn[hook.layer()] = r[0, -1].detach().float(); return r
        with torch.no_grad():
            M_ctr = float(_logp_diff(model.run_with_hooks(counter, fwd_hooks=[(_rname(dir_layer), gr_c)]), cid, aid))
            M_neu = float(_logp_diff(model.run_with_hooks(
                neutral, fwd_hooks=[(_zname(L), gz) for L in range(nL)] + [(_rname(dir_layer), gr_n)]), cid, aid))
        gap = M_neu - M_ctr
        M_set = _patch_set(model, counter, zneu, heads, cid, aid)             # SET joint patch
        set_fracs.append((M_set - M_ctr) / gap)
        ctx.append({"counter": counter, "cid": cid, "aid": aid, "M_ctr": M_ctr, "gap": gap,
                    "rc": rc[dir_layer], "rn": rn[dir_layer]})
    # DIRECTION: fit cave_dir on matched items, ablate u-projection to neutral mean
    D = torch.stack([c["rc"] - c["rn"] for c in ctx])
    u = (D.mean(0) / (D.mean(0).norm() + 1e-8)).to(device)
    proj_n = statistics.mean(float(c["rn"].to(device) @ u) for c in ctx)
    nec_fracs = []
    for c in ctx:
        shift = proj_n - float(c["rc"].to(device) @ u)
        def ab(r, hook, u=u, shift=shift):
            r[0, -1] = r[0, -1] + (shift * u).to(r.dtype); return r
        with torch.no_grad():
            M_ab = float(_logp_diff(model.run_with_hooks(c["counter"], fwd_hooks=[(_rname(dir_layer), ab)]), c["cid"], c["aid"]))
        nec_fracs.append((M_ab - c["M_ctr"]) / c["gap"])
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"set_joint": round(statistics.mean(set_fracs), 4), "dir_nec": round(statistics.mean(nec_fracs), 4),
            "set_per_item": [round(x, 4) for x in set_fracs],     # per-item dispersion (triage: noise floor)
            "dir_per_item": [round(x, 4) for x in nec_fracs]}


def run(name_base, name_it, pool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    it_gaps = _gap_pass(name_it, True, device, pool)
    base_gaps = _gap_pass(name_base, False, device, pool)
    idxs = matched(base_gaps, it_gaps)
    print(f"[matched] {len(idxs)}/{len(pool)} items cave on BOTH (gap>+{MIN_EFF}): {idxs}", flush=True)
    if len(idxs) < 3:
        print("[abort] too few matched items for a stable differential"); return
    eff_it = _effect_pass(name_it, True, device, idxs, HEADS, NH_9B, DIR_LAYER, pool)
    eff_base = _effect_pass(name_base, False, device, idxs, HEADS, NH_9B, DIR_LAYER, pool)
    set_dec = decide_amp(eff_it["set_joint"], eff_base["set_joint"])
    dir_dec = decide_amp(eff_it["dir_nec"], eff_base["dir_nec"])
    dec = overall(set_dec, dir_dec)
    # paired bootstrap CIs (items are aligned across base/it) -- does the it-vs-base gap exclude 0?
    set_diff = [i - b for i, b in zip(eff_it["set_per_item"], eff_base["set_per_item"])]
    dir_diff = [i - b for i, b in zip(eff_it["dir_per_item"], eff_base["dir_per_item"])]
    ci = {"set_it": bootstrap_ci(eff_it["set_per_item"]), "set_base": bootstrap_ci(eff_base["set_per_item"]),
          "set_it_minus_base": bootstrap_ci(set_diff),
          "dir_it": bootstrap_ci(eff_it["dir_per_item"]), "dir_base": bootstrap_ci(eff_base["dir_per_item"]),
          "dir_it_minus_base": bootstrap_ci(dir_diff)}
    out = {"model_base": name_base, "model_it": name_it, "cue": "matched_item_deconfound",
           "pool_size": len(pool), "min_eff": MIN_EFF, "dir_layer": DIR_LAYER, "amp_ratio": AMP_RATIO,
           "n_matched": len(idxs), "matched_idxs": idxs,
           "it_gaps": {i: round(g, 3) for i, g in it_gaps.items()},
           "base_gaps": {i: round(g, 3) for i, g in base_gaps.items()},
           "eff_it": eff_it, "eff_base": eff_base, "bootstrap_ci": ci, "decision": dec}
    Path("out").mkdir(exist_ok=True)
    Path("out/matched_item_deconfound_9b.json").write_text(json.dumps(out, indent=2))
    print(f"\n[eff_it] set={eff_it['set_joint']} dir={eff_it['dir_nec']}  "
          f"[eff_base] set={eff_base['set_joint']} dir={eff_base['dir_nec']}")
    print(f"[ci set it-base] {ci['set_it_minus_base']}   [ci dir it-base] {ci['dir_it_minus_base']}")
    print("[verdict]", dec["verdict"])
    print("[done] wrote out/matched_item_deconfound_9b.json")


# --------------------------------------------------------------------------- selftest
def selftest():
    # matched: both must positively cave; anti-cave (negative) and sub-threshold items dropped
    bg = {0: 0.6, 1: -0.2, 2: 0.8, 3: 0.4, 4: 1.0}
    ig = {0: 0.7, 1: 0.9, 2: 0.8, 3: 0.3, 5: 2.0}
    assert matched(bg, ig) == [0, 2], matched(bg, ig)   # 1: base anti-cave; 3: it sub-thr; 4/5: not in both
    print(f"[selftest] matched intersection = {matched(bg, ig)}")

    # decide_amp branches
    assert decide_amp(0.40, 0.02)["tag"] == "INSTALLED"
    assert decide_amp(0.40, 0.10)["tag"] == "AMPLIFIED"            # 0.40 >= 2x 0.10
    assert decide_amp(0.40, 0.30)["tag"] == "BASE_SHARED"         # ratio 1.33 < 2
    assert decide_amp(0.05, 0.0)["tag"] == "NO_EFFECT"
    print("[selftest] decide_amp: INSTALLED/AMPLIFIED/BASE_SHARED/NO_EFFECT all fire")

    # overall: both amplified -> survives; both base-shared -> weakened; mixed -> mixed
    s_amp = decide_amp(0.40, 0.10); d_amp = decide_amp(0.50, 0.16)
    assert overall(s_amp, d_amp)["verdict"].startswith("AMPLIFIED-NOT-INSTALLED SURVIVES"), overall(s_amp, d_amp)
    s_bs = decide_amp(0.40, 0.30); d_bs = decide_amp(0.50, 0.40)
    assert overall(s_bs, d_bs)["verdict"].startswith("CLAIM WEAKENED"), overall(s_bs, d_bs)
    assert overall(s_amp, d_bs)["verdict"].startswith("MIXED"), overall(s_amp, d_bs)
    print("[selftest] overall: SURVIVES / WEAKENED / MIXED all fire")

    # bootstrap_ci: a tight positive sample excludes 0; a zero-centered sample straddles 0
    pos = bootstrap_ci([0.3, 0.32, 0.28, 0.31, 0.29, 0.30])
    assert pos["lo"] > 0 and pos["n"] == 6, pos
    zero = bootstrap_ci([0.3, -0.3, 0.2, -0.2, 0.1, -0.1])
    assert zero["lo"] < 0 < zero["hi"], zero
    assert bootstrap_ci([])["mean"] is None
    print(f"[selftest] bootstrap_ci: positive CI {pos['lo']}..{pos['hi']} excludes 0; zero-centered straddles")
    print("[selftest] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--narrow", action="store_true", help="use the committed 16 items (default: wide pool)")
    ap.add_argument("--name-base", default="google/gemma-2-9b")
    ap.add_argument("--name-it", default="google/gemma-2-9b-it")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a.name_base, a.name_it, ITEMS if a.narrow else ITEMS_WIDE)
