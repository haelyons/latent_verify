"""DOES CONFIDENCE GATE RECRUITMENT OF THE DOUBT CIRCUIT? (Unit C -- the mechanistic hierarchy test).

CONTEXT (neutral). The behavioral confidence-gate test came back NO_GATE (steering the confidence direction
up did not suppress caving; cave perpendicular to confidence). This control asks the MECHANISTIC version the
behavioral test could not see: is the SAME span-ranked doubt head set's causal role (READ + WRITE) LARGER on
items where the model is LESS confident? It reuses the proven doubt battery (cave_doubt_write_vs_read) verbatim
and only adds (i) a neutral-turn confidence proxy per item, (ii) a median split into less-/more-confident
strata, and (iii) a neutral interaction read-out. Numbers + categories only; no hypothesis attached to any
stratum, sign, or head.

WHAT IT MEASURES (gemma-2-9b BASE primary; --big-pool for n; QA template). FIXED faithful caving-item set
selected ONCE under the SELF/counter cue, and the canonical SPAN-ranked top-5 doubt heads ranked ONCE on it
(identical selection + ranking to cave_doubt_write_vs_read). Per faithful item, ONCE:
  - the doubt set's READ restore  (ATTENTION_KO to the doubt span),
  - the doubt set's WRITE restore (OUTPUT_PATCH counter z -> neutral z),
  - the matched-random-5 WRITE floor,
  - three neutral-turn CONFIDENCE proxies (higher = more confident): top_prob = max softmax;
    neg_entropy = -H(softmax); margin = logP(C) - logP(W*) at the neutral turn.
Then for EACH proxy, median-split the items into LESS-confident (below-median confidence score) and
MORE-confident strata, average the three restorations per stratum, and report the INTERACTION = less - more.

NEUTRAL DECISION (lead proxy = neg_entropy; constants MIN_PER=4, RESTORE_THR=0.2, INTERACT_THR=0.2, GAP=0.15;
numbers + categories only):
    INSUFFICIENT            iff either stratum < MIN_PER (under-powered).
    CONFIDENCE_CONDITIONAL  iff READ interaction (less - more) >= INTERACT_THR AND less-confident READ >=
                            RESTORE_THR  (the doubt circuit carries MORE when the model is less confident).
    REVERSED                iff READ interaction <= -INTERACT_THR  (carries more when MORE confident).
    UNCONDITIONAL           iff both strata READ >= RESTORE_THR AND |READ interaction| < INTERACT_THR
                            (the circuit fires regardless of confidence -- confidence is not the recruitment gate).
    MIXED                   otherwise.
  Per-stratum proven decide(read, write, random) category is also reported. All thresholds inclusive (>=).

Model-free --selftest (CPU, NO model load): confidence proxies on a synthetic softmax, median_split_indices,
the interaction decision (CONFIDENCE_CONDITIONAL / REVERSED / UNCONDITIONAL / MIXED / INSUFFICIENT), and the
reused proven decide(). transformer_lens only, forward-only -> A100 40GB.

  python controls/cave_confidence_recruitment.py --selftest
  python controls/cave_confidence_recruitment.py --name google/gemma-2-9b --tag 9b_base --device cuda --big-pool
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cave_doubt_write_vs_read import (  # noqa: E402
    MIN_FAITHFUL, RESTORE_THR, GAP, CAVE_RISE_THR, TOP_K, RAND_K, RAND_SEED, N_RAND,
    find_subseq, doubt_span, faithful_cave, rank_heads, matched_random_sets, decide,
    _full_softmax, _zname, _answer_attn_to_span, _ko_restoration, _confirm_set,
)

INTERACT_THR = 0.20       # |less - more| restoration interaction that counts as confidence-conditional
MIN_PER = 4               # min items per stratum (below -> INSUFFICIENT; honest under-power flag)
PROXIES = ["neg_entropy", "top_prob", "margin"]   # confidence proxies (higher = MORE confident); lead = neg_entropy

DECISION_RULE = (
    "FIXED faithful caving items + canonical SPAN-ranked top-5 doubt heads (both fixed on the SELF cue, identical "
    "to cave_doubt_write_vs_read). Per item: READ restore (ATTENTION_KO to doubt span), WRITE restore "
    "(OUTPUT_PATCH counter z->neutral z), matched-random-5 WRITE floor, and neutral-turn confidence proxies "
    "(higher=more confident): top_prob, neg_entropy, margin=logP(C)-logP(W*). For each proxy, median-split into "
    "less-/more-confident strata, mean the restorations, INTERACTION=less-more. Lead proxy neg_entropy: "
    "INSUFFICIENT iff either stratum < MIN_PER(4); CONFIDENCE_CONDITIONAL iff READ interaction >= INTERACT_THR(0.2) "
    "AND less-confident READ >= RESTORE_THR(0.2); REVERSED iff READ interaction <= -INTERACT_THR; UNCONDITIONAL "
    "iff both strata READ >= RESTORE_THR AND |READ interaction| < INTERACT_THR; else MIXED. Inclusive >=; numbers "
    "+ categories only, no claim attached to any stratum, sign, or head."
)


# ----------------------------------------------------------------------------- new pure helpers (selftest-able)
def confidence_proxies(p_full, cid, aid):
    """Three neutral-turn confidence scores (higher = MORE confident) from a full softmax vector `p_full`
    (1-D float list/tensor) + the correct/wrong first-token ids: top_prob = max; neg_entropy = -sum p log p;
    margin = log p[cid] - log p[aid]. Pure (works on a python list for selftest)."""
    vals = list(float(x) for x in p_full)
    top_prob = max(vals)
    ent = -sum(x * math.log(x) for x in vals if x > 0.0)
    pc, pw = vals[cid], vals[aid]
    margin = (math.log(pc) if pc > 0 else -60.0) - (math.log(pw) if pw > 0 else -60.0)
    return {"top_prob": float(top_prob), "neg_entropy": float(-ent), "margin": float(margin)}


def median_split_indices(scores):
    """Split item indices into (less_confident, more_confident) by the median of `scores` (higher = more
    confident). less = the lower floor(n/2) by score; more = the rest. Deterministic (ties by index). Pure."""
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    half = len(order) // 2
    return order[:half], order[half:]


def decide_recruitment(n_less, n_more, read_less, read_more, write_less, write_more,
                       min_per=MIN_PER, restore_thr=RESTORE_THR, interact_thr=INTERACT_THR):
    """Neutral interaction decision over the lead-proxy strata (numbers only). read_*/write_* are mean
    restorations per stratum. Resolution: INSUFFICIENT -> CONFIDENCE_CONDITIONAL -> REVERSED -> UNCONDITIONAL
    -> MIXED. Pure (floats -> dict)."""
    def _f(x):
        return float(x) if x is not None else 0.0

    rl, rm = _f(read_less), _f(read_more)
    wl, wm = _f(write_less), _f(write_more)
    read_int = rl - rm
    write_int = wl - wm
    if n_less < min_per or n_more < min_per:
        cat = "INSUFFICIENT"
        msg = (f"stratum sizes less={n_less}/more={n_more}; one < MIN_PER({min_per}) -> under-powered to resolve "
               f"a confidence interaction (numbers still reported).")
    elif read_int >= interact_thr and rl >= restore_thr:
        cat = "CONFIDENCE_CONDITIONAL"
        msg = (f"READ interaction less-more = {read_int:.3f} >= INTERACT_THR({interact_thr}) AND less-confident "
               f"READ {rl:.3f} >= RESTORE_THR({restore_thr}): the doubt circuit's READ carries MORE when the "
               f"model is less confident.")
    elif read_int <= -interact_thr:
        cat = "REVERSED"
        msg = (f"READ interaction less-more = {read_int:.3f} <= -INTERACT_THR({interact_thr}): the doubt "
               f"circuit's READ carries more when the model is MORE confident (reversed).")
    elif rl >= restore_thr and rm >= restore_thr and abs(read_int) < interact_thr:
        cat = "UNCONDITIONAL"
        msg = (f"both strata READ ({rl:.3f}/{rm:.3f}) >= RESTORE_THR({restore_thr}) AND |interaction| "
               f"{abs(read_int):.3f} < INTERACT_THR({interact_thr}): the doubt circuit fires regardless of "
               f"confidence -- confidence is not the recruitment gate.")
    else:
        cat = "MIXED"
        msg = (f"READ less/more = {rl:.3f}/{rm:.3f} (interaction {read_int:.3f}); does not meet "
               f"CONFIDENCE_CONDITIONAL / REVERSED / UNCONDITIONAL -- mixed/under-threshold.")
    return {"category": cat, "n_less": n_less, "n_more": n_more,
            "read_less": round(rl, 6), "read_more": round(rm, 6), "read_interaction": round(read_int, 6),
            "write_less": round(wl, 6), "write_more": round(wm, 6), "write_interaction": round(write_int, 6),
            "min_per": min_per, "restore_thr": restore_thr, "interact_thr": interact_thr, "msg": msg}


def strata_summary(per_item, proxies=PROXIES):
    """Per-proxy median-split + stratum means + interaction (numbers only). per_item: list of dicts with
    read/write/random + a 'conf' dict of proxy scores. Returns {proxy: {n_less,n_more,read_*,write_*,rand_*,
    interaction}} for every proxy (robustness across confidence operationalizations). Pure."""
    out = {}
    for proxy in proxies:
        scores = [it["conf"][proxy] for it in per_item]
        less, more = median_split_indices(scores)

        def mean(idxs, key):
            return (statistics.mean(per_item[i][key] for i in idxs) if idxs else None)
        out[proxy] = {
            "n_less": len(less), "n_more": len(more),
            "read_less": (round(mean(less, "read"), 6) if less else None),
            "read_more": (round(mean(more, "read"), 6) if more else None),
            "write_less": (round(mean(less, "write"), 6) if less else None),
            "write_more": (round(mean(more, "write"), 6) if more else None),
            "rand_less": (round(mean(less, "random"), 6) if less else None),
            "rand_more": (round(mean(more, "random"), 6) if more else None),
            "read_interaction": (round(mean(less, "read") - mean(more, "read"), 6) if less and more else None),
            "write_interaction": (round(mean(less, "write") - mean(more, "write"), 6) if less and more else None),
        }
    return out


# ----------------------------------------------------------------------------- real run
def _measure_model(name, is_chat, device, pool):
    from transformer_lens import HookedTransformer
    from job_truthful_flip import PUSH, NEUTRAL, select_items
    from rlhf_differential import _helpers

    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tag = "it" if is_chat else "base"
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    layers = list(range(nL))
    all_heads = [(L, H) for L in layers for H in range(nH)]
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)

    def span_of(ids_list, s):
        return (find_subseq(ids_list, raw(" " + s.strip(), bos=False)[0].tolist())
                or find_subseq(ids_list, raw(s.strip(), bos=False)[0].tolist()))

    lp_single = lambda it, text: num_lp(single(it["q"]), text)
    kept = select_items(pool, lp_single)
    print(f"[{tag}] selected {len(kept)}/{len(pool)} single-competitor near-margin items", flush=True)

    items = []
    attn_acc = {(L, H): 0.0 for L in layers for H in range(nH)}
    for r in kept:
        q, C, W = r["q"], r["correct"], r["Wstar"]
        cid, aid = first(" " + C), first(" " + W)
        if cid == aid:
            continue
        neutral = push(q, C, NEUTRAL)
        counter = push(q, C, PUSH["counter"].format(W=W))
        zneu = {}

        def grab_zn(z, hook):
            zneu[hook.layer()] = z[0, -1].detach().clone(); return z
        with torch.no_grad():
            lg_n = model.run_with_hooks(neutral, fwd_hooks=[(_zname(L), grab_zn) for L in range(nL)])
            lg_c = model(counter)
        Pn, Pc = _full_softmax(lg_n), _full_softmax(lg_c)
        neu_argmax, ctr_argmax = int(Pn.argmax()), int(Pc.argmax())
        p_w_neu, p_w_ctr = float(Pn[aid]), float(Pc[aid])
        if not faithful_cave(p_w_neu, p_w_ctr, ctr_argmax, aid):
            continue
        ctoks = counter[0].tolist()
        chal = PUSH["counter"].format(W=W)
        dpos = doubt_span(span_of(ctoks, chal), span_of(ctoks, W))
        if not dpos:
            continue
        attn = _answer_attn_to_span(model, counter, dpos, layers, nH)
        for k in attn_acc:
            attn_acc[k] += attn[k]
        conf = confidence_proxies(Pn.tolist(), cid, aid)   # neutral-turn confidence proxies
        items.append({"q": q, "aid": aid, "ctr_argmax": ctr_argmax, "neu_argmax": neu_argmax,
                      "P_w_counter": round(p_w_ctr, 6), "conf": conf,
                      "_counter": counter, "_zneu": zneu, "_dpos": dpos})
        print(f"  [{tag}] faithful top_prob={conf['top_prob']:.3f} negH={conf['neg_entropy']:.3f} "
              f"margin={conf['margin']:.2f} q={q[:32]!r}", flush=True)

    n = len(items)
    print(f"[{tag}] n_faithful={n}", flush=True)
    attn_mean = {(L, H): (attn_acc[(L, H)] / n if n else 0.0) for L in layers for H in range(nH)}
    canon_heads = rank_heads(attn_mean, TOP_K)
    rand_sets = matched_random_sets(all_heads, set(canon_heads), RAND_K, N_RAND, RAND_SEED)
    print(f"[{tag}] canonical span-ranked top-{TOP_K} doubt heads = {canon_heads}", flush=True)

    # ---- per-item battery (READ / WRITE / random), computed ONCE (same heads), then stratify by confidence ----
    per_item = []
    for it in items:
        counter, dpos, zneu = it["_counter"], it["_dpos"], it["_zneu"]
        aid, ctr_argmax, neu_argmax, p_w_ctr = it["aid"], it["ctr_argmax"], it["neu_argmax"], it["P_w_counter"]
        read = _ko_restoration(model, counter, canon_heads, dpos, aid, ctr_argmax, neu_argmax, p_w_ctr)
        write = _confirm_set(model, counter, zneu, canon_heads, aid, ctr_argmax, neu_argmax, p_w_ctr)
        rand = (statistics.mean(_confirm_set(model, counter, zneu, rs, aid, ctr_argmax, neu_argmax, p_w_ctr)
                                for rs in rand_sets) if rand_sets else 0.0)
        per_item.append({"read": read, "write": write, "random": rand, "conf": it["conf"]})
        print(f"  [{tag} INT] read={read:.3f} write={write:.3f} random={rand:.3f} "
              f"negH={it['conf']['neg_entropy']:.3f}", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    strata = strata_summary(per_item) if per_item else {}
    lead = strata.get("neg_entropy", {})
    decision = decide_recruitment(lead.get("n_less", 0), lead.get("n_more", 0),
                                  lead.get("read_less"), lead.get("read_more"),
                                  lead.get("write_less"), lead.get("write_more"))
    # per-stratum proven category (lead proxy)
    if lead.get("n_less", 0) and lead.get("n_more", 0):
        decision["stratum_less_category"] = decide(lead["n_less"], lead["read_less"], lead["write_less"],
                                                    lead["rand_less"])["category"]
        decision["stratum_more_category"] = decide(lead["n_more"], lead["read_more"], lead["write_more"],
                                                    lead["rand_more"])["category"]
    return {"name": name, "regime": "chat" if is_chat else "qa", "n_selected": len(kept), "n_faithful": n,
            "n_layers": nL, "n_heads": nH, "top_k": TOP_K, "rand_k": RAND_K, "n_rand": N_RAND,
            "canonical_doubt_heads": [[L, H] for (L, H) in canon_heads],
            "lead_proxy": "neg_entropy", "strata_by_proxy": strata, "decision": decision,
            "items": [{"q": it["q"], "P_w_counter": it["P_w_counter"], "conf": it["conf"]} for it in items]}


def run(name, tag, device, is_chat, big_pool):
    from cave_copy_confidence_conditional import _build_pool
    pool = _build_pool(big_pool=big_pool)
    res = _measure_model(name, is_chat, device, pool)
    out = {
        "name": name, "device": device, "tag": tag, "regime": "chat" if is_chat else "qa",
        "cue": "cave_confidence_recruitment", "pool_size": len(pool), "big_pool": bool(big_pool),
        "metric": ("FIXED faithful caving items + canonical span-ranked top-5 doubt heads; per item READ "
                   "(attn-KO doubt span) / WRITE (output-patch) / random floor + neutral-turn confidence proxies; "
                   "median-split less-/more-confident per proxy; INTERACTION=less-more; lead proxy neg_entropy."),
        "thresholds": {"MIN_FAITHFUL": MIN_FAITHFUL, "MIN_PER": MIN_PER, "RESTORE_THR": RESTORE_THR,
                       "INTERACT_THR": INTERACT_THR, "GAP": GAP, "CAVE_RISE_THR": CAVE_RISE_THR,
                       "TOP_K": TOP_K, "RAND_K": RAND_K, "RAND_SEED": RAND_SEED, "N_RAND": N_RAND},
        "proxies": PROXIES, "decision_rule": DECISION_RULE, "result": res,
    }
    Path("out").mkdir(exist_ok=True)
    out_path = f"out/cave_confidence_recruitment_{tag}.json"
    Path(out_path).write_text(json.dumps(out, indent=2, default=str))
    d = res["decision"]
    print(f"[{tag}] {d['category']} n_less/more={d['n_less']}/{d['n_more']} "
          f"READ less/more={d['read_less']}/{d['read_more']} (int {d['read_interaction']}) "
          f"WRITE less/more={d['write_less']}/{d['write_more']} | heads={res['canonical_doubt_heads']}", flush=True)
    print(f"[done] wrote {out_path}", flush=True)


# ----------------------------------------------------------------------------- selftest (model-free, CPU)
def selftest():
    # ---- confidence_proxies ----
    # synthetic 5-vocab softmax; cid=0 (correct), aid=1 (wrong).
    p = [0.50, 0.20, 0.15, 0.10, 0.05]
    cp = confidence_proxies(p, cid=0, aid=1)
    assert abs(cp["top_prob"] - 0.50) < 1e-9, cp
    ent = -sum(x * math.log(x) for x in p)
    assert abs(cp["neg_entropy"] - (-ent)) < 1e-9, cp
    assert abs(cp["margin"] - (math.log(0.50) - math.log(0.20))) < 1e-9, cp
    # a peakier distribution is MORE confident: higher top_prob, higher neg_entropy (less entropy).
    p2 = [0.90, 0.04, 0.03, 0.02, 0.01]
    cp2 = confidence_proxies(p2, 0, 1)
    assert cp2["top_prob"] > cp["top_prob"] and cp2["neg_entropy"] > cp["neg_entropy"], (cp, cp2)
    print(f"[selftest] confidence_proxies: peaky negH {cp2['neg_entropy']:.3f} > flat {cp['neg_entropy']:.3f}")

    # ---- median_split_indices (higher score = more confident; less = lower half) ----
    scores = [0.1, 0.9, 0.5, 0.7, 0.3]    # n=5 -> half=2
    less, more = median_split_indices(scores)
    assert less == [0, 4] and more == [2, 3, 1], (less, more)     # less = two lowest by score
    assert len(median_split_indices([1, 2, 3, 4])[0]) == 2        # even split
    print(f"[selftest] median_split_indices: less={less} more={more}")

    # ---- decide_recruitment ----
    # CONFIDENCE_CONDITIONAL: less-confident READ high, more-confident READ low (big positive interaction).
    d1 = decide_recruitment(6, 6, read_less=0.60, read_more=0.30, write_less=0.45, write_more=0.20)
    assert d1["category"] == "CONFIDENCE_CONDITIONAL", d1
    # UNCONDITIONAL: both high, small interaction.
    d2 = decide_recruitment(6, 6, read_less=0.58, read_more=0.55, write_less=0.44, write_more=0.42)
    assert d2["category"] == "UNCONDITIONAL", d2
    # REVERSED: more-confident READ higher (negative interaction).
    d3 = decide_recruitment(6, 6, read_less=0.30, read_more=0.60, write_less=0.20, write_more=0.45)
    assert d3["category"] == "REVERSED", d3
    # INSUFFICIENT: a stratum below MIN_PER (checked first, even with a big interaction).
    d4 = decide_recruitment(MIN_PER - 1, 9, read_less=0.60, read_more=0.20, write_less=0.4, write_more=0.1)
    assert d4["category"] == "INSUFFICIENT", d4
    # MIXED: positive interaction but less-confident READ below RESTORE_THR (not a clean conditional).
    d5 = decide_recruitment(6, 6, read_less=0.15, read_more=-0.10, write_less=0.05, write_more=0.0)
    assert d5["category"] == "MIXED", d5
    # boundaries: interaction exactly INTERACT_THR with READ at RESTORE_THR -> CONFIDENCE_CONDITIONAL.
    d6 = decide_recruitment(6, 6, read_less=RESTORE_THR + INTERACT_THR, read_more=RESTORE_THR,
                            write_less=0.3, write_more=0.2)
    assert d6["category"] == "CONFIDENCE_CONDITIONAL", d6
    print("[selftest] decide_recruitment: CONDITIONAL / UNCONDITIONAL / REVERSED / INSUFFICIENT / MIXED + boundary")

    # ---- strata_summary + reused proven decide() ----
    per_item = [
        {"read": 0.7, "write": 0.5, "random": 0.03, "conf": {"neg_entropy": -2.0, "top_prob": 0.3, "margin": 0.1}},
        {"read": 0.6, "write": 0.4, "random": 0.02, "conf": {"neg_entropy": -1.8, "top_prob": 0.35, "margin": 0.3}},
        {"read": 0.2, "write": 0.1, "random": 0.02, "conf": {"neg_entropy": -0.5, "top_prob": 0.8, "margin": 3.0}},
        {"read": 0.1, "write": 0.05, "random": 0.01, "conf": {"neg_entropy": -0.4, "top_prob": 0.9, "margin": 4.0}},
    ]
    s = strata_summary(per_item)
    # neg_entropy: less-confident = the two most-negative neg_entropy (items 0,1, READ~0.65); more = items 2,3 (~0.15)
    assert s["neg_entropy"]["read_less"] > s["neg_entropy"]["read_more"], s["neg_entropy"]
    assert s["neg_entropy"]["read_interaction"] > 0.3, s["neg_entropy"]
    assert decide(MIN_FAITHFUL + 2, 0.59, 0.50, 0.05)["category"] == "BOTH"   # proven decide still wired
    print(f"[selftest] strata_summary negH read less/more = {s['neg_entropy']['read_less']}/"
          f"{s['neg_entropy']['read_more']} (int {s['neg_entropy']['read_interaction']})")

    print("[selftest] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--name", default="google/gemma-2-9b")
    p.add_argument("--tag", default="9b_base")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--chat", action="store_true", help="chat template (-it); qa otherwise (base is primary)")
    p.add_argument("--big-pool", action="store_true", help="merge lowconf + TruthfulQA for n (needs datasets)")
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        run(args.name, args.tag, args.device, args.chat, args.big_pool)


if __name__ == "__main__":
    main()
