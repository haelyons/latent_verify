"""NEXT-1 (head-SET prong) -- is the 9b-it misconception caving carried by a jointly-necessary head SET
that per-head intervention structurally cannot see? (RESEARCH_QUESTIONS Part 4 "NEXT-1 head-set test".)

WHY THIS EXISTS. Every instrument the per-head arc used (AtP differential `rlhf_differential.py`,
activation-patch arbiter `atp_low_confirm.py`, `gate_dont_delete`) scored INDIVIDUAL heads and ranked
per-head. The 9b result is precisely "no single installed head" (frac_it < 0.10 each; arbiter max
0.067). It does NOT exclude a jointly-necessary installed *set* whose members are each sub-threshold --
which is exactly what the diffuse 9b band (18 heads, net_it ~0.03-0.07) looks like. Arc-1 SS3.9/SS3.12:
the 2b salience effect is carried JOINTLY by a ~12-head set (all-heads necessity ~1.0) while no single
head exceeds ~0.2 -- non-additive. Field precedent that the head-SET is the standard unit, not a
workaround: IOI is a ~26-head set with backup name-movers that compensate under single-head ablation
(Wang 2023); self-repair/Hydra (Rushing & Nanda 2024; McGrath 2023) is why per-head under-measures;
sycophancy itself was path-patched to a ~4% head set (Chen et al. ICML 2024). (Spot-check IDs per
POSITIONING before external use.)

WHAT IT DOES. Reuses the validated `rlhf_differential._confirm` activation-patch arbiter, but patches a
LIST of heads in ONE forward (the only new mechanism): set each head's -it counter readout z[-1,H] to
its neutral-turn value for ALL heads in the set at once, across whatever layers they live in, and
measure the JOINT fraction of the cave restored. Run on -it AND base (set-level differential). Substrate
= the 16 I1 9b-it caving items (TruthfulQA misconceptions), already in `rlhf_differential.ITEMS`.
Candidate set = the 18 diffuse heads `atp_low_confirm.HEADS` (net_it ~0.03-0.07), i.e. exactly the band
a per-head sweep cannot resolve. Metric/sign convention identical to `_confirm`:
  frac = (M_patch - M_counter) / (M_neutral - M_counter),  M = logp(C) - logp(W*) first-token margin.
POSITIVE & large = the set restores the cave (patching toward neutral raises the C-vs-W* margin).

Shape + specificity controls (the SS3.9 idiom at SET level):
  (a) CUMULATIVE RAMP top-1,2,...,K in individual-arbiter-frac order -> concentrated if 2-3 heads carry
      it, distributed-set if it needs many.
  (b) MATCHED-RANDOM-K set (n_rand fixed random k-head sets, heads NOT in the candidate band) -> the
      SS3.5/S3 specificity control at set level: patching k heads must not restore the cave on its own.
  (c) JOINT frac vs SUM of individual fracs -> quantifies super-additivity (the non-additive signature).

PRE-REGISTERED SUCCESS CRITERIA (decide_installed_set, exercised model-free by --selftest):
  INSTALLED-SET  iff joint frac_it >= INSTALL_THR (0.10)  AND  every member < INSTALL_THR individually
                 AND  matched-random-K ~0 (< BASE_FLOOR)  AND  joint frac_base <= BASE_FLOOR.
                 -> the per-head NULL was SET-BLIND; the installed deference object is a jointly-necessary
                    head set. (Positive, distributed account the program's NULL streak needs.)
  NULL-HOLDS     iff even jointly the top-K set does not reach threshold over base.
                 -> the 9b caving locus is NOT attention-head-local even as a set: pivot to the NEXT-1
                    residual-direction / MLP probe. Either way the per-head NULL's SCOPE is settled.
  (Guards: SET-PRESENT-BUT-BASE-SHARED if the set also restores in base; NON-SPECIFIC if matched-random-K
   restores comparably; SINGLE-HEAD-DOMINATES if a member alone reaches threshold.)

Forward-only (no AtP backward) -> fits the 40GB A100. Imports the verified `_confirm` harness pieces and
ITEMS verbatim; the joint patch is a list-loop inside one hook, the trivial extension the protocol named.

  python headset_joint_patch.py --selftest
  python headset_joint_patch.py --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it
"""
import argparse
import json
import random
import statistics
from pathlib import Path

import torch

from rlhf_differential import ITEMS, _helpers, _logp_diff, INSTALL_THR, BASE_FLOOR, MIN_EFFECT_NET
from job_truthful_flip import PUSH, NEUTRAL
from atp_low_confirm import HEADS, NH_9B

SEED = 0          # matched-random sets are deterministic + identical across base/it (proper control)
N_RAND = 5        # #matched-random K-head sets to average over (variance reduction)


# --------------------------------------------------------------------------- pure decision
def decide_installed_set(ramp_it, ramp_base, indiv_it, rand_it, rand_base,
                         install_thr=INSTALL_THR, base_floor=BASE_FLOOR):
    """Pure decision over the joint-patch fractions. ramp_it/ramp_base: dict k(int)->joint frac for the
    cumulative top-k set. indiv_it: list of per-member individual fracs (-it). rand_it/rand_base: mean
    matched-random-K frac. The headline 'joint' is the FULL top-K set; the ramp gives the shape."""
    K = max(ramp_it)
    joint_it = ramp_it[K]
    joint_base = ramp_base.get(K)
    peak_k = max(ramp_it, key=lambda k: ramp_it[k])              # concentration shape: where the set peaks
    peak_frac = ramp_it[peak_k]
    members_subthr = all(v < install_thr for v in indiv_it)
    max_member = max(indiv_it) if indiv_it else 0.0
    sum_indiv = sum(indiv_it)
    reaches = joint_it >= install_thr
    base_clean = (joint_base is None) or (joint_base <= base_floor)
    rand_clean = (rand_it is None) or (rand_it < base_floor)
    installed_set = reaches and members_subthr and base_clean and rand_clean
    if installed_set:
        v = (f"INSTALLED SET FOUND: top-{K} diffuse head set JOINTLY restores the cave in -it "
             f"(joint frac {joint_it:.3f} >= {install_thr}) while every member < {install_thr} "
             f"individually (max {max_member:.3f}), matched-random-{K} ~0 ({rand_it}) and base ~0 "
             f"({joint_base}). The per-head NULL was SET-BLIND -- the installed deference object is a "
             f"jointly-necessary head set (peak at k={peak_k}).")
    elif reaches and not members_subthr:
        v = (f"SINGLE HEAD DOMINATES: a member alone reaches {max_member:.3f} >= {install_thr} -- not a "
             f"distributed-set effect; revisit the per-head result, not a head-set finding.")
    elif reaches and not base_clean:
        v = (f"SET PRESENT BUT BASE-SHARED: joint frac {joint_it:.3f} >= {install_thr} but base also "
             f"restores ({joint_base} > {base_floor}) -- a base mechanism the set recruits, NOT "
             f"RLHF-installed (the set-level analog of base-shared L38.H14).")
    elif reaches and not rand_clean:
        v = (f"NON-SPECIFIC: joint frac {joint_it:.3f} >= {install_thr} but a matched-random-{K} set "
             f"restores comparably ({rand_it} >= {base_floor}) -- patching k heads moves M regardless; "
             f"not specific to these heads.")
    else:
        v = (f"NULL HOLDS (set-level): even jointly the top-{K} diffuse head set does not restore the "
             f"cave over base (max joint frac {peak_frac:.3f} at k={peak_k} < {install_thr}). The 9b "
             f"caving locus is NOT attention-head-local even as a SET -- pivot to the NEXT-1 residual "
             f"direction / MLP probe. The per-head NULL's scope is now settled at the set level.")
    return {"installed_set": installed_set, "reaches_threshold": reaches,
            "joint_frac_it": round(joint_it, 4),
            "joint_frac_base": (round(joint_base, 4) if joint_base is not None else None),
            "peak_k": peak_k, "peak_frac_it": round(peak_frac, 4),
            "max_member_it": round(max_member, 4), "members_subthreshold": members_subthr,
            "sum_individual_it": round(sum_indiv, 4),
            "super_additivity_it": round(joint_it - sum_indiv, 4),   # joint - sum(individual)
            "rand_frac_it": rand_it, "rand_frac_base": rand_base,
            "base_clean": base_clean, "rand_clean": rand_clean, "verdict": v}


# --------------------------------------------------------------------------- joint patch (the new bit)
def _zname(L):
    return f"blocks.{L}.attn.hook_z"


def _patch_set(model, counter, zneu, head_set, cid, aid):
    """JOINT activation-patch: set z[-1,H] -> its neutral-turn value for EVERY (L,H) in head_set, across
    all involved layers, in ONE forward. (The single-head `_confirm` hook, lifted to a list of heads.)"""
    by_layer = {}
    for (L, H) in head_set:
        by_layer.setdefault(L, []).append(H)
    def mk(L, Hs):
        def patch(z, hook, L=L, Hs=Hs):
            for H in Hs:
                z[0, -1, H, :] = zneu[L][H].to(z.dtype)
            return z
        return patch
    hooks = [(_zname(L), mk(L, Hs)) for L, Hs in by_layer.items()]
    with torch.no_grad():
        lg = model.run_with_hooks(counter, fwd_hooks=hooks)
    return float(_logp_diff(lg, cid, aid))


def _confirm_set(name, is_chat, device, heads, nH, ramp_order=None):
    """One model load. Pass 1: per-item context (counter, neutral readouts, gap) + per-head INDIVIDUAL
    joint=single frac. Rank by individual frac (claim-blind, in-run) unless ramp_order is given (base
    reuses the -it order). Pass 2: cumulative ramp top-1..K + matched-random-K, reusing cached contexts."""
    from transformer_lens import HookedTransformer
    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    nL = model.cfg.n_layers
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    K = len(heads)
    cand = set(heads)
    rng = random.Random(SEED)
    allheads = [(L, H) for L in range(nL) for H in range(nH) if (L, H) not in cand]
    rand_sets = [rng.sample(allheads, K) for _ in range(N_RAND)]   # fixed across items + base/it (matched)

    indiv = {h: [] for h in heads}
    ctxs = []
    for it in ITEMS:
        q, C, W = it["q"], it["correct"], it["Wstar"]
        cid, aid = first(" " + C), first(" " + W)
        counter = push(q, C, PUSH["counter"].format(W=W))
        neutral = push(q, C, NEUTRAL)
        zneu = {}
        def grab(z, hook):
            zneu[hook.layer()] = z[0, -1].detach().clone(); return z
        with torch.no_grad():
            M_neu = float(_logp_diff(model.run_with_hooks(neutral, fwd_hooks=[(_zname(L), grab) for L in range(nL)]), cid, aid))
            M_ctr = float(_logp_diff(model(counter), cid, aid))
        gap = M_neu - M_ctr
        if abs(gap) < MIN_EFFECT_NET:                              # vacuous if no real cave to restore
            continue
        ctxs.append({"counter": counter, "cid": cid, "aid": aid, "zneu": zneu, "M_ctr": M_ctr, "gap": gap})
        for h in heads:
            M_p = _patch_set(model, counter, zneu, [h], cid, aid)
            indiv[h].append((M_p - M_ctr) / gap)
        print(f"  [{'it' if is_chat else 'base'}] gap={gap:+.2f} q={q[:44]!r}", flush=True)

    indiv_mean = {h: (statistics.mean(v) if v else 0.0) for h, v in indiv.items()}
    order = ramp_order or sorted(heads, key=lambda h: indiv_mean[h], reverse=True)

    ramp = {k: [] for k in range(1, K + 1)}
    rand_acc = []
    for ctx in ctxs:
        for k in range(1, K + 1):
            M_p = _patch_set(model, ctx["counter"], ctx["zneu"], order[:k], ctx["cid"], ctx["aid"])
            ramp[k].append((M_p - ctx["M_ctr"]) / ctx["gap"])
        rs = [(_patch_set(model, ctx["counter"], ctx["zneu"], rset, ctx["cid"], ctx["aid"]) - ctx["M_ctr"]) / ctx["gap"]
              for rset in rand_sets]
        rand_acc.append(statistics.mean(rs))

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"n_ok": len(ctxs),
            "indiv": {f"{L}.{H}": round(indiv_mean[(L, H)], 4) for (L, H) in heads},
            "order": [[L, H] for (L, H) in order],
            "ramp": {k: round(statistics.mean(v), 4) for k, v in ramp.items() if v},
            "rand": (round(statistics.mean(rand_acc), 4) if rand_acc else None)}


def run(name_base, name_it):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    heads, nH = HEADS, NH_9B
    it = _confirm_set(name_it, True, device, heads, nH)            # -it first; capture the ramp order
    order = [tuple(x) for x in it["order"]]
    base = _confirm_set(name_base, False, device, heads, nH, ramp_order=order)  # base patches the SAME set

    ramp_it = {int(k): v for k, v in it["ramp"].items()}
    ramp_base = {int(k): v for k, v in base["ramp"].items()}
    indiv_it = [it["indiv"][f"{L}.{H}"] for (L, H) in heads]
    dec = decide_installed_set(ramp_it, ramp_base, indiv_it, it["rand"], base["rand"])

    out = {"model_base": name_base, "model_it": name_it, "cue": "headset_joint_patch",
           "substrate": "I1 9b-it caving items (TruthfulQA misconceptions)",
           "n_heads_in_set": len(heads), "n_rand_sets": N_RAND, "seed": SEED,
           "thresholds": {"install_thr": INSTALL_THR, "base_floor": BASE_FLOOR, "min_effect_net": MIN_EFFECT_NET},
           "it_n_ok": it["n_ok"], "base_n_ok": base["n_ok"],
           "ramp_order": it["order"],
           "indiv_it": it["indiv"], "indiv_base": base["indiv"],
           "ramp_it": it["ramp"], "ramp_base": base["ramp"],
           "rand_it": it["rand"], "rand_base": base["rand"],
           "decision": dec}
    Path("out").mkdir(exist_ok=True)
    Path("out/headset_joint_patch_9b.json").write_text(json.dumps(out, indent=2))
    print("\n[ramp_it]", it["ramp"])
    print("[rand_it]", it["rand"], "[joint_base@K]", ramp_base.get(max(ramp_base)) if ramp_base else None)
    print("[verdict]", dec["verdict"])
    print("[done] wrote out/headset_joint_patch_9b.json")


# --------------------------------------------------------------------------- selftest (model-free)
def selftest():
    """Exercise decide_installed_set across every branch on synthetic joint-patch fractions. K=4."""
    base_clean = {1: 0.0, 2: 0.01, 3: 0.01, 4: 0.02}

    # INSTALLED SET: joint reaches 0.13 at K, all members < 0.10, random ~0, base ~0
    d = decide_installed_set({1: 0.04, 2: 0.07, 3: 0.10, 4: 0.13}, base_clean,
                             [0.06, 0.05, 0.04, 0.03], 0.01, 0.0)
    assert d["installed_set"] and d["verdict"].startswith("INSTALLED SET FOUND"), d
    assert d["peak_k"] == 4 and d["super_additivity_it"] == round(0.13 - 0.18, 4), d
    print(f"[selftest] INSTALLED SET fires: joint={d['joint_frac_it']} super_add={d['super_additivity_it']}")

    # NULL HOLDS: joint never reaches 0.10
    d2 = decide_installed_set({1: 0.02, 2: 0.04, 3: 0.05, 4: 0.06}, base_clean,
                              [0.03, 0.02, 0.02, 0.01], 0.01, 0.0)
    assert (not d2["installed_set"]) and d2["verdict"].startswith("NULL HOLDS"), d2
    print(f"[selftest] NULL HOLDS fires: max joint {d2['peak_frac_it']} < {INSTALL_THR}")

    # BASE-SHARED: joint >= 0.10 but base also restores
    d3 = decide_installed_set({1: 0.05, 2: 0.09, 3: 0.12, 4: 0.15}, {1: 0.0, 2: 0.05, 3: 0.10, 4: 0.12},
                              [0.06, 0.05, 0.04, 0.03], 0.01, 0.10)
    assert (not d3["installed_set"]) and d3["verdict"].startswith("SET PRESENT BUT BASE-SHARED"), d3
    print(f"[selftest] BASE-SHARED fires: joint_base={d3['joint_frac_base']}")

    # NON-SPECIFIC: joint >= 0.10, base clean, but matched-random-K also restores
    d4 = decide_installed_set({1: 0.05, 2: 0.09, 3: 0.12, 4: 0.15}, base_clean,
                              [0.06, 0.05, 0.04, 0.03], 0.13, 0.0)
    assert (not d4["installed_set"]) and d4["verdict"].startswith("NON-SPECIFIC"), d4
    print(f"[selftest] NON-SPECIFIC fires: rand_it={d4['rand_frac_it']}")

    # SINGLE HEAD DOMINATES: a member alone reaches threshold
    d5 = decide_installed_set({1: 0.12, 2: 0.13, 3: 0.14, 4: 0.15}, base_clean,
                              [0.12, 0.05, 0.04, 0.03], 0.01, 0.0)
    assert (not d5["installed_set"]) and d5["verdict"].startswith("SINGLE HEAD DOMINATES"), d5
    print(f"[selftest] SINGLE HEAD DOMINATES fires: max_member={d5['max_member_it']}")

    # candidate-set wiring sanity: the head set is the 18 diffuse AtP-low heads, no dupes
    assert len(HEADS) == 18 and len(set(HEADS)) == 18, HEADS
    print(f"[selftest] candidate set = {len(HEADS)} diffuse heads (atp_low_confirm.HEADS), NH={NH_9B}")
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
