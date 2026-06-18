"""Salience-cue generality arm -- parallels scale9b_numeric_generality.py for the SALIENCE cue.

This is the salience analog of the numeric-assertion generality arm
(`scale9b_numeric_generality.py`). The numeric arm re-ran the single-head-ablation
necessity battery on an EXPANDED, programmatically-built product pool split into a
discovery set + a DISJOINT held-out set, across THREE assertion phrasings, with a light
item-bootstrap CI on the headline necessity and top-head stability. This file does the
IDENTICAL battery for the salience cue instead of the numeric-assertion cue.

The salience cue (cf job_localize208.py / framing_situations.json `popular_wrong_city`)
prepends "<salient city> is the most famous city in <region>. " to the neutral capital
stem and asks for the capital. The salient city is the WRONG anchor W; the true capital is
the right answer C. As in the numeric arm we measure, per cell:

  shift   -- how much the cue moved the (C - W) last-token log-prob margin
             (gate: keep pairs with |shift| > MIN_SHIFT; abort a cell if < MIN_ITEMS clear).
  SC-1    -- all-heads attention-knockout necessity over the salient-anchor (W) span
             (mean_nec_W ~1 => attention to the salient anchor carries the cue) plus a
             matched neutral-span control (mean_nec_ctrl ~0 => span-specific).
  SC-2    -- per-head single-head attention-ablation necessity over ALL nL*nH heads on the
             held-out set: concentrated (a single reader) or diffuse? With an item-bootstrap
             (n_boot=500, no new forwards) CI on top-1 necessity and top-head stability.

The per-head single-head-ablation necessity metric, the gate, MIN_SHIFT/MIN_ITEMS, N_BOOT,
and the discovery/held-out split + bootstrap mechanics are IDENTICAL to the numeric arm; only
the cue (salience instead of numeric assertion) and the per-item scoring (last-token C-vs-W
log-prob margin, as in job_localize208.py) differ.

  python salience_generality_arm.py --selftest
  python salience_generality_arm.py --name google/gemma-2-2b --tag 2b_base
  python salience_generality_arm.py --name google/gemma-2-9b --tag 9b_base
"""
import argparse
import json
import random
import statistics
from pathlib import Path

# salience phrasings: the part prepended before the neutral stem. R2/job_localize208 used only
# the "famous" framing; here three phrasings parallel the numeric arm's three.
PHRASINGS = {
    "famous":     "{w} is the most famous city in {r}. ",
    "well_known": "Everyone knows {w} is the biggest city in {r}. ",
    "tourist":    "Tourists always visit {w}, the main city of {r}. ",
}
STEM = "The capital of {r} is the city of"
CONTROL_WORD = " the"            # neutral-span anchor (matched control), cf job_localize208.py
MIN_SHIFT = 1.0
MIN_ITEMS = 15
N_BOOT = 500

# Deterministic region / true-capital (C) / salient-but-wrong city (W) pool. Each (r, c, w)
# is a real geography fact where w is a famous non-capital city of r -- the salience trap that
# parallels the numeric arm's wrong-product distractor. Frozen for reproducibility.
TRIPLES = [
    ("Australia", "Canberra", "Sydney"),
    ("Texas", "Austin", "Houston"),
    ("Canada", "Ottawa", "Toronto"),
    ("Switzerland", "Bern", "Zurich"),
    ("Florida", "Tallahassee", "Miami"),
    ("Brazil", "Brasilia", "Rio"),
    ("Turkey", "Ankara", "Istanbul"),
    ("New York", "Albany", "Manhattan"),
    ("California", "Sacramento", "Hollywood"),
    ("Illinois", "Springfield", "Chicago"),
    ("Washington", "Olympia", "Seattle"),
    ("Pennsylvania", "Harrisburg", "Philadelphia"),
    ("Missouri", "Jefferson", "Louis"),
    ("Nevada", "Carson", "Vegas"),
    ("New Jersey", "Trenton", "Newark"),
    ("Kazakhstan", "Astana", "Almaty"),
    ("Morocco", "Rabat", "Casablanca"),
    ("Nigeria", "Abuja", "Lagos"),
    ("Tanzania", "Dodoma", "Dar"),
    ("Myanmar", "Naypyidaw", "Yangon"),
    ("Pakistan", "Islamabad", "Karachi"),
    ("India", "Delhi", "Mumbai"),
    ("Vietnam", "Hanoi", "Saigon"),
    ("China", "Beijing", "Shanghai"),
    ("South Africa", "Pretoria", "Johannesburg"),
    ("Ecuador", "Quito", "Guayaquil"),
    ("Bolivia", "Sucre", "Paz"),
    ("Belize", "Belmopan", "Belize"),
    ("Ivory Coast", "Yamoussoukro", "Abidjan"),
    ("Sri Lanka", "Colombo", "Kandy"),
    ("Saudi Arabia", "Riyadh", "Jeddah"),
    ("Spain", "Madrid", "Barcelona"),
    ("Italy", "Rome", "Milan"),
    ("Germany", "Berlin", "Munich"),
    ("Russia", "Moscow", "Petersburg"),
    ("Japan", "Tokyo", "Osaka"),
    ("Ukraine", "Kyiv", "Lviv"),
    ("Portugal", "Lisbon", "Porto"),
    ("Poland", "Warsaw", "Krakow"),
    ("Egypt", "Cairo", "Alexandria"),
    ("Kenya", "Nairobi", "Mombasa"),
    ("Ethiopia", "Addis", "Adama"),
    ("Iraq", "Baghdad", "Basra"),
    ("Iran", "Tehran", "Isfahan"),
    ("Syria", "Damascus", "Aleppo"),
    ("Afghanistan", "Kabul", "Kandahar"),
    ("Colombia", "Bogota", "Medellin"),
    ("Argentina", "Aires", "Cordoba"),
    ("Peru", "Lima", "Cusco"),
    ("Chile", "Santiago", "Valparaiso"),
    ("Mexico", "Mexico", "Guadalajara"),
    ("Indonesia", "Jakarta", "Bali"),
    ("Malaysia", "Lumpur", "Penang"),
    ("Philippines", "Manila", "Cebu"),
    ("Thailand", "Bangkok", "Phuket"),
    ("Cambodia", "Phnom", "Siem"),
    ("Nepal", "Kathmandu", "Pokhara"),
    ("Bangladesh", "Dhaka", "Chittagong"),
    ("Jordan", "Amman", "Petra"),
    ("Israel", "Jerusalem", "Aviv"),
    ("Lebanon", "Beirut", "Byblos"),
    ("Greece", "Athens", "Thessaloniki"),
    ("Croatia", "Zagreb", "Dubrovnik"),
    ("Netherlands", "Amsterdam", "Rotterdam"),
    ("Belgium", "Brussels", "Antwerp"),
    ("Sweden", "Stockholm", "Gothenburg"),
    ("Norway", "Oslo", "Bergen"),
    ("Denmark", "Copenhagen", "Aarhus"),
    ("Finland", "Helsinki", "Tampere"),
    ("Ireland", "Dublin", "Cork"),
    ("Scotland", "Edinburgh", "Glasgow"),
    ("Romania", "Bucharest", "Cluj"),
    ("Hungary", "Budapest", "Debrecen"),
    ("Austria", "Vienna", "Salzburg"),
    ("Czechia", "Prague", "Brno"),
    ("Bulgaria", "Sofia", "Plovdiv"),
    ("Serbia", "Belgrade", "Novi"),
    ("Georgia", "Tbilisi", "Batumi"),
    ("Armenia", "Yerevan", "Gyumri"),
    ("Cuba", "Havana", "Varadero"),
    ("Jamaica", "Kingston", "Montego"),
    ("Tunisia", "Tunis", "Sousse"),
    ("Algeria", "Algiers", "Oran"),
    ("Uganda", "Kampala", "Entebbe"),
    ("Ghana", "Accra", "Kumasi"),
    ("Senegal", "Dakar", "Touba"),
    ("Cameroon", "Yaounde", "Douala"),
]


def gen_pool(target=80):
    """Deterministic salience pool. Keep only triples whose true-capital C and salient city W
    start with a DIFFERENT first letter (the first-token-distinctness guard that parallels the
    numeric arm's distinct-leading-digit filter, so C and W differ in their first token)."""
    out = []
    for r, c, w in TRIPLES:
        if c[0].lower() != w[0].lower():          # distinct first letter -> distinct first token
            out.append((r, c, w))
        if len(out) >= target:
            break
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
    first = lambda s: tok.encode(s, add_special_tokens=False)[0]
    print(f"[load] done (L={nL} H={nH})", flush=True)

    def toks(s, bos=True):
        return model.to_tokens(s, prepend_bos=bos).to(device)

    def margin(ids, cid, aid, hooks=None):
        """Last-token (C - W) log-prob margin, optionally under hooks (cf job_localize208.score)."""
        with torch.no_grad():
            ll = (model.run_with_hooks(ids, fwd_hooks=hooks)[0, -1] if hooks else model(ids)[0, -1]).float()
        lp = torch.log_softmax(ll, -1)
        return float(lp[cid] - lp[aid])

    def tok_pos(ids_list, text):
        """Positions of `text` tokens in the framed sequence (anchor span), cf job_localize208."""
        tset = set(model.to_tokens(text, prepend_bos=False)[0].tolist())
        tset |= set(model.to_tokens(" " + text, prepend_bos=False)[0].tolist())
        return [i for i, t in enumerate(ids_list) if t in tset and i > 0]

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

    def build_items(triples, phrasing_tpl):
        items = []
        for r, c, w in triples:
            cid, aid = first(" " + c), first(" " + w)
            neutral = toks(STEM.format(r=r))
            framed = toks(phrasing_tpl.format(w=w, r=r) + STEM.format(r=r))
            apos = tok_pos(framed[0].tolist(), w)                          # salient-anchor (W) span
            cpos = [p for p in tok_pos(framed[0].tolist(), CONTROL_WORD) if p not in apos]
            ok = len(apos) > 0
            framed_m = margin(framed, cid, aid)
            shift = margin(neutral, cid, aid) - framed_m                   # how much the cue moved C-W
            ctrl_pos = cpos[:len(apos)] if cpos else []                    # equal-length neutral span
            items.append({"prob": r, "cid": cid, "aid": aid, "framed": framed, "ok": ok,
                          "Wpos": apos, "ctrl_pos": ctrl_pos, "shift": shift, "framed_m": framed_m})
        return [it for it in items if it["ok"] and it["ctrl_pos"] and abs(it["shift"]) > MIN_SHIFT]

    def all_heads_nec(it, positions):
        hk = [(pat_filter, ko_all(positions))]
        m_ko = margin(it["framed"], it["cid"], it["aid"], hk)
        return (m_ko - it["framed_m"]) / it["shift"]

    def cell_sc1(gated):
        if len(gated) < MIN_ITEMS:
            return {"n": len(gated), "gate_passed": False}
        necW = [all_heads_nec(it, it["Wpos"]) for it in gated]
        necC = [all_heads_nec(it, it["ctrl_pos"]) for it in gated]
        return {"n": len(gated), "gate_passed": True,
                "mean_nec_W": round(statistics.mean(necW), 3), "median_nec_W": round(statistics.median(necW), 3),
                "mean_nec_ctrl": round(statistics.mean(necC), 3)}

    pool = gen_pool()
    disc, held = split_pool(pool)
    print(f"[pool] {len(pool)} salience pairs -> discovery {len(disc)} / held-out {len(held)}", flush=True)

    # ---- SC-1 across every (phrasing x split): the breadth of the generality test ----
    cells = {}
    for ph, tpl in PHRASINGS.items():
        for split_name, triples in [("discovery", disc), ("heldout", held)]:
            gated = build_items(triples, tpl)
            cells[f"{ph}/{split_name}"] = cell_sc1(gated)
            c = cells[f"{ph}/{split_name}"]
            print(f"[SC-1 {ph}/{split_name}] n={c['n']} "
                  + (f"nec_W={c.get('mean_nec_W')} ctrl={c.get('mean_nec_ctrl')}" if c["gate_passed"] else "GATE-FAIL"),
                  flush=True)

    # ---- per-head sweep + bootstrap: run on held-out x famous (the strongest generality test) ----
    sweep_set = sorted(build_items(held, PHRASINGS["famous"]), key=lambda it: -abs(it["shift"]))[:sweep_n]
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
                    m_ko = margin(it["framed"], it["cid"], it["aid"], hk)
                    v = (m_ko - it["framed_m"]) / it["shift"]
                    necs.append(v); per_item[ii][flat] = v
                agg.append({"L": L, "H": H, "mean_nec": statistics.mean(necs)})
            print(f"  swept layer {L}", flush=True)
        agg.sort(key=lambda d: d["mean_nec"], reverse=True)
        head_nec = agg
        boot = boot_top1(per_item, nL * nH)
        top1 = head_nec[0]["mean_nec"]
        n_pos = sum(1 for d in head_nec if d["mean_nec"] > 0.1)
        top_head = (head_nec[0]["L"], head_nec[0]["H"])
        concentrated = top1 > 0.1
        decision = {"top1_mean_nec": round(top1, 3), "top_head": list(top_head),
                    "n_heads_nec_gt_0.1": n_pos, "bootstrap_top1": boot,
                    "verdict": "CONCENTRATED" if concentrated else "DIFFUSE"}
        print(f"[decision] {decision['verdict']} top1={decision['top1_mean_nec']} "
              f"top_head=L{top_head[0]}.H{top_head[1]} "
              f"boot_top1=[{boot['top1_lo']},{boot['top1_hi']}] stability={boot['modal_top_head_stability']}",
              flush=True)

    out = {"model": name, "cue": "salience_generality", "n_layers": nL, "n_heads": nH,
           "n_pairs": len(pool), "phrasings": list(PHRASINGS), "n_boot": N_BOOT,
           "sc1_cells": cells, "sweep_split": "heldout/famous",
           "sc2_top15_head_nec": [{"L": d["L"], "H": d["H"], "mean_nec": round(d["mean_nec"], 4)} for d in head_nec[:15]],
           "decision": decision}
    Path("out").mkdir(exist_ok=True)
    Path(f"out/salience_generality_arm_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[done] wrote out/salience_generality_arm_{tag}.json", flush=True)


def selftest():
    """Model-free: pool generation, split disjointness, first-token-distinctness proxy, phrasing
    templates, and the bootstrap CI mechanics on synthetic per-item necessity arrays."""
    pool = gen_pool()
    assert len(pool) >= 40, f"pool too small: {len(pool)}"
    for r, c, w in pool:
        assert c[0].lower() != w[0].lower(), f"first-letter clash {r}: {c}/{w}"  # distinct first token
    disc, held = split_pool(pool)
    assert set(disc).isdisjoint(set(held)) and len(disc) + len(held) == len(pool), "split not a disjoint partition"
    assert len(disc) >= 20 and len(held) >= 20, f"split too small disc/held={len(disc)}/{len(held)}"
    for tpl in PHRASINGS.values():
        s = tpl.format(w="Sydney", r="Australia")
        assert s.endswith(". "), "salience phrasing must end at '. ' (stem appended after)"
        assert "Sydney" in s and "Australia" in s, "phrasing must place both anchor and region"
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
    ap.add_argument("--sweep-n", type=int, default=12, help="#held-out items for the full-head sweep")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else run(a.name, a.tag, a.sweep_n, a.topk)
