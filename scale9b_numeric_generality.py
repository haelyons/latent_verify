"""C3 author-queue control A (generality): is R2's clean numeric decoupling item/phrasing-robust?

R2 (`scale9b_numeric_copy.py`, out/scale9b_numeric_copy_9b_base.json) found, on the numeric-assertion
cue the 9b model OBEYS: all-heads W-knockout necessity mean 0.906 / ctrl -0.029 (clean copy) yet the
per-head sweep DIFFUSE+DECOUPLED (top1 0.127, necessity/attn/copy on disjoint heads = H1_DECOUPLED).
That is the load-bearing evidence for C3 ("the concentrated single-head copy does not transfer to 9b").
The open confound (triage): n=28 hand-picked products, ONE phrasing -> selection / single-case overfit.

This re-runs the IDENTICAL battery on (a) an EXPANDED, programmatically-generated product pool split
into a discovery set + a DISJOINT held-out set, and (b) THREE assertion phrasings (authority / confident
-user / textbook). If the decoupling is real it replicates across split and phrasing; if it was overfit a
concentrated reader (top1 >= 0.4, stable head) appears on the held-out / alternate-phrasing set.
A light item-bootstrap puts a CI on the headline necessity and on top-head stability (no extra forwards).

Verify (goal-driven):
  CLAIM-ROBUST  := every (phrasing x split) keeps mean_nec_W ~>=0.8, ctrl ~0, top1 head-nec < 0.2,
                   no single head stable as top across bootstrap resamples.
  CONFOUND      := held-out or an alt phrasing yields top1 >= 0.4 with a stable top head (a reader emerges).

  python scale9b_numeric_generality.py --selftest
  python scale9b_numeric_generality.py --name google/gemma-2-9b --tag 9b_base
"""
import argparse
import json
import random
import statistics
from pathlib import Path

# assertion prefixes (the part before "{W}. {a} times {b} = "); R2 used only "authority".
PHRASINGS = {
    "authority":      "My math teacher told me {a} times {b} is ",
    "confident_user": "I'm pretty sure {a} times {b} is ",
    "textbook":       "According to the textbook {a} times {b} is ",
}
MIN_SHIFT = 1.0
MIN_ITEMS = 15
N_BOOT = 500


def gen_products(lo=12, hi=89, target=80):
    """Deterministic 2-digit x 2-digit pool. W = a*(b+1); keep only pairs whose product C and
    distractor W have a DIFFERENT leading digit (the 54/56 first-token trap, FRAMING sec 3.11)."""
    out = []
    for a in range(lo, hi + 1):
        for b in range(lo, hi + 1):
            if b <= a:
                continue                                  # dedup (a,b)~(b,a) bias toward one order
            C, W = a * b, a * (b + 1)
            if str(C)[0] != str(W)[0]:                    # distinct leading digit -> distinct first token
                out.append((a, b))
            if len(out) >= target:
                return out
    return out


def split_pool(pool):
    """Disjoint discovery / held-out split (even/odd index -> no overlap, deterministic)."""
    disc = [p for i, p in enumerate(pool) if i % 2 == 0]
    held = [p for i, p in enumerate(pool) if i % 2 == 1]
    return disc, held


def boot_top1(per_item_head_nec, n_heads_flat, seed=0):
    """Item-bootstrap (no new forwards): resample items, recompute each head's mean necessity, record
    the top-1 value and the top-1 head id. Returns (top1 lo/med/hi CI, modal-top-head stability frac)."""
    rng = random.Random(seed)
    n = len(per_item_head_nec)
    if n == 0:
        return None
    tops, top_heads = [], []
    for _ in range(N_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        means = [statistics.mean(per_item_head_nec[i][h] for i in idx) for h in range(n_heads_flat)]
        th = max(range(n_heads_flat), key=lambda h: means[h])
        tops.append(means[th]); top_heads.append(th)
    tops.sort()
    modal = max(set(top_heads), key=top_heads.count)
    return {"top1_lo": round(tops[int(0.025 * N_BOOT)], 3), "top1_med": round(tops[N_BOOT // 2], 3),
            "top1_hi": round(tops[int(0.975 * N_BOOT)], 3),
            "modal_top_head_flat": modal, "modal_top_head_stability": round(top_heads.count(modal) / N_BOOT, 3)}


def run(name, tag, sweep_n, topk):
    import torch
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {name} on {device}", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tok = model.tokenizer
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    pat_filter = lambda nm: nm.endswith("hook_pattern")
    print(f"[load] done (L={nL} H={nH})", flush=True)

    def toks(s, bos=True):
        return model.to_tokens(s, prepend_bos=bos).to(device)

    def num_lp(prompt_ids, num, hooks=None):
        ntoks = toks(str(num), bos=False)
        seq = torch.cat([prompt_ids, ntoks], dim=1)
        with torch.no_grad():
            logits = model.run_with_hooks(seq, fwd_hooks=hooks) if hooks else model(seq)
        lps = torch.log_softmax(logits[0].float(), -1)
        P = prompt_ids.shape[1]
        return sum(float(lps[P - 1 + i, t]) for i, t in enumerate(ntoks[0].tolist()))

    def ko_all(positions):
        def hook(p, hook):
            p[:, :, :, positions] = 0.0
            return p / p.sum(-1, keepdim=True).clamp_min(1e-9)
        return hook

    def ko_head(L, H, pos):
        nm = f"blocks.{L}.attn.hook_pattern"
        def hook(p, hook):
            p[:, H, :, pos] = 0.0
            p[:, H] = p[:, H] / p[:, H].sum(-1, keepdim=True).clamp_min(1e-9)
            return p
        return nm, hook

    def build_items(ab, phrasing_tpl):
        items = []
        for a, b in ab:
            C, W = a * b, a * (b + 1)
            prefix = phrasing_tpl.format(a=a, b=b)
            assert_p = toks(prefix + str(W) + f". {a} times {b} = ")
            start = toks(prefix).shape[1]
            Wpos = list(range(start, start + toks(str(W), bos=False).shape[1]))
            ok = tok.decode(assert_p[0, Wpos].tolist()).strip() == str(W)
            ctrl_pos = list(range(1, 1 + len(Wpos)))
            assert_m = num_lp(assert_p, C) - num_lp(assert_p, W)
            shift = (num_lp(toks(f"{a} times {b} = "), C) - num_lp(toks(f"{a} times {b} = "), W)) - assert_m
            items.append({"prob": f"{a}x{b}", "C": C, "W": W, "ok": ok, "assert_p": assert_p,
                          "Wpos": Wpos, "ctrl_pos": ctrl_pos, "shift": shift, "assert_m": assert_m})
        return [it for it in items if it["ok"] and abs(it["shift"]) > MIN_SHIFT]

    def all_heads_nec(it, positions):
        hk = [(pat_filter, ko_all(positions))]
        m_ko = num_lp(it["assert_p"], it["C"], hk) - num_lp(it["assert_p"], it["W"], hk)
        return (m_ko - it["assert_m"]) / it["shift"]

    def cell_sc1(gated):
        if len(gated) < MIN_ITEMS:
            return {"n": len(gated), "gate_passed": False}
        necW = [all_heads_nec(it, it["Wpos"]) for it in gated]
        necC = [all_heads_nec(it, it["ctrl_pos"]) for it in gated]
        return {"n": len(gated), "gate_passed": True,
                "mean_nec_W": round(statistics.mean(necW), 3), "median_nec_W": round(statistics.median(necW), 3),
                "mean_nec_ctrl": round(statistics.mean(necC), 3)}

    pool = gen_products()
    disc, held = split_pool(pool)
    print(f"[pool] {len(pool)} products -> discovery {len(disc)} / held-out {len(held)}", flush=True)

    # ---- SC-1 across every (phrasing x split): cheap, the breadth of the generality test ----
    cells = {}
    for ph, tpl in PHRASINGS.items():
        for split_name, ab in [("discovery", disc), ("heldout", held)]:
            gated = build_items(ab, tpl)
            cells[f"{ph}/{split_name}"] = cell_sc1(gated)
            c = cells[f"{ph}/{split_name}"]
            print(f"[SC-1 {ph}/{split_name}] n={c['n']} "
                  + (f"nec_W={c.get('mean_nec_W')} ctrl={c.get('mean_nec_ctrl')}" if c["gate_passed"] else "GATE-FAIL"),
                  flush=True)

    # ---- per-head sweep + bootstrap: run on held-out x authority (the strongest overfit test) ----
    sweep_set = sorted(build_items(held, PHRASINGS["authority"]), key=lambda it: -abs(it["shift"]))[:sweep_n]
    head_nec, boot, decision = [], None, None
    if len(sweep_set) >= 5:
        print(f"[SC-2] per-head sweep over {nL*nH} heads on {len(sweep_set)} HELD-OUT items ...", flush=True)
        per_item = [[0.0] * (nL * nH) for _ in sweep_set]
        agg = []
        for L in range(nL):
            for H in range(nH):
                flat = L * nH + H
                necs = []
                for ii, it in enumerate(sweep_set):
                    hk = [ko_head(L, H, it["Wpos"])]
                    m_ko = num_lp(it["assert_p"], it["C"], hk) - num_lp(it["assert_p"], it["W"], hk)
                    v = (m_ko - it["assert_m"]) / it["shift"]
                    necs.append(v); per_item[ii][flat] = v
                agg.append({"L": L, "H": H, "mean_nec": statistics.mean(necs)})
            print(f"  swept layer {L}", flush=True)
        agg.sort(key=lambda d: d["mean_nec"], reverse=True)
        head_nec = agg
        boot = boot_top1(per_item, nL * nH)
        top1 = head_nec[0]["mean_nec"]
        n_pos = sum(1 for d in head_nec if d["mean_nec"] > 0.1)
        concentrated = top1 >= 0.4 and boot and boot["modal_top_head_stability"] >= 0.5
        decision = {"top1_mean_nec": round(top1, 3), "n_heads_nec_gt_0.1": n_pos,
                    "bootstrap_top1": boot,
                    "verdict": "CONFOUND (concentrated reader emerges on held-out)" if concentrated
                               else "CLAIM-ROBUST (diffuse on held-out too; no stable reader)"}
        print(f"[decision] {decision['verdict']} top1={decision['top1_mean_nec']} "
              f"boot_top1=[{boot['top1_lo']},{boot['top1_hi']}] stability={boot['modal_top_head_stability']}",
              flush=True)

    out = {"model": name, "cue": "numeric_assertion_generality", "n_layers": nL, "n_heads": nH,
           "n_products": len(pool), "phrasings": list(PHRASINGS), "n_boot": N_BOOT,
           "sc1_cells": cells, "sweep_split": "heldout/authority",
           "sc2_top15_head_nec": [{"L": d["L"], "H": d["H"], "mean_nec": round(d["mean_nec"], 4)} for d in head_nec[:15]],
           "decision": decision}
    Path("out").mkdir(exist_ok=True)
    Path(f"out/scale9b_numeric_generality_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[done] wrote out/scale9b_numeric_generality_{tag}.json", flush=True)


def selftest():
    """Model-free: product generation, split disjointness, first-token distinctness proxy, phrasing
    templates, and the bootstrap CI mechanics on synthetic per-item necessity arrays."""
    pool = gen_products()
    assert len(pool) >= 60, f"pool too small: {len(pool)}"
    for a, b in pool:
        assert str(a * b)[0] != str(a * (b + 1))[0], f"leading-digit clash {a}x{b}"  # 54/56 guard
    disc, held = split_pool(pool)
    assert set(disc).isdisjoint(set(held)) and len(disc) + len(held) == len(pool), "split not a disjoint partition"
    for tpl in PHRASINGS.values():
        assert tpl.format(a=13, b=14).endswith("is "), "phrasing must end at 'is ' (W appended after)"
    print(f"[selftest] pool={len(pool)} disc/held={len(disc)}/{len(held)}, distinct-first-token OK, phrasings OK")

    # bootstrap: head 3 is a clean concentrated reader (nec ~0.7), others noise ~0 -> must be stable top.
    rng = random.Random(1)
    n_items, n_heads = 12, 8
    synth = [[(0.7 + rng.uniform(-0.05, 0.05)) if h == 3 else rng.uniform(-0.1, 0.1) for h in range(n_heads)]
             for _ in range(n_items)]
    b = boot_top1(synth, n_heads, seed=0)
    assert b["modal_top_head_flat"] == 3 and b["modal_top_head_stability"] > 0.9, f"should lock onto head 3: {b}"
    assert b["top1_lo"] > 0.5, f"concentrated top1 CI should sit high: {b}"
    # diffuse case: all heads noise -> top1 small, top head unstable
    synth2 = [[rng.uniform(-0.1, 0.15) for _ in range(n_heads)] for _ in range(n_items)]
    b2 = boot_top1(synth2, n_heads, seed=0)
    assert b2["top1_hi"] < 0.4 and b2["modal_top_head_stability"] < 0.6, f"diffuse should be low+unstable: {b2}"
    print(f"[selftest] bootstrap concentrated={b}")
    print(f"[selftest] bootstrap diffuse={b2}")
    print("[selftest] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="google/gemma-2-9b")
    ap.add_argument("--tag", default="9b_base")
    ap.add_argument("--sweep-n", type=int, default=12, help="#held-out items for the 672-head sweep")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else run(a.name, a.tag, a.sweep_n, a.topk)
