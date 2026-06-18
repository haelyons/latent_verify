"""P-C + N-2 (SEQUENCE_170626): matched-scope full-head re-localization of the SALIENCE
copy, plus a router hunt.

P-C: salience is swept over ALL heads (2b 26x8=208; 9b 42x16=672) and averaged across the
5 pairs -- the SAME basis as the numeric localize -- so "concentrated salience vs diffuse
numeric" can be compared like-for-like, and the word "diffuse" either survives or is retracted.

N-2: which UPSTREAM head selects the salient anchor that the reader copies (the IOI
S-inhibition / duplicate-token analog)? For each head in layers 0..reader-1, zero its output
(hook_z) and measure the drop in the reader's anchor attention at the readout. Large drop = router.

SCALE PORT: re-localizes from scratch (the 2b reader L18.H5 does NOT transfer to 9b).
  - reader head determined per model: --reader auto = modal max-attn-to-anchor head across pairs.
  - max-attn reader scan reported per pair regardless of effect size.
  - if NO pair clears MIN_EFFECT (the salience copy collapsed at scale, FRAMING sec 10.1
    flags effect +0.02 for 9b base), the per-head necessity sweep + router are SKIPPED as
    vacuous (necessity = (ko-framed)/effect is undefined at effect~0); effects + max-attn
    are still reported -- that is the honest finding.

  python job_localize208.py                                          # -> out/localize_salience_2b.json
  python job_localize208.py --name google/gemma-2-9b --tag 9b --reader auto
"""
import argparse
import collections
import json
from pathlib import Path

STEM = "The capital of {r} is the city of"
SALIENCE = "{w} is the most famous city in {r}. "
CONTROL_WORD = " the"
MIN_EFFECT = 0.5
PAIRS = [
    {"r": "Australia",   "c": "Canberra",    "w": "Sydney"},
    {"r": "Texas",       "c": "Austin",      "w": "Houston"},
    {"r": "Canada",      "c": "Ottawa",      "w": "Toronto"},
    {"r": "Switzerland", "c": "Bern",        "w": "Zurich"},
    {"r": "Florida",     "c": "Tallahassee", "w": "Miami"},
]


def run(name, tag, reader_arg):
    import torch
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {name} on {device}", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tok = model.tokenizer
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    first = lambda s: tok.encode(s, add_special_tokens=False)[0]
    pat_filter = lambda nm: nm.endswith("hook_pattern")

    def score(ids, cid, aid, hooks=None):
        with torch.no_grad():
            ll = (model.run_with_hooks(ids, fwd_hooks=hooks)[0, -1] if hooks else model(ids)[0, -1]).float()
        lp = torch.log_softmax(ll, -1)
        return float(lp[cid] - lp[aid])

    def tok_pos(ids_list, text):
        tset = set(model.to_tokens(text, prepend_bos=False)[0].tolist())
        tset |= set(model.to_tokens(" " + text, prepend_bos=False)[0].tolist())
        return [i for i, t in enumerate(ids_list) if t in tset and i > 0]

    def ko_all(positions):
        def hook(p, hook):
            p[:, :, :, positions] = 0.0
            return p / p.sum(-1, keepdim=True).clamp_min(1e-9)
        return hook

    def ko_head(L, H, positions):
        nm = f"blocks.{L}.attn.hook_pattern"
        def hook(p, hook):
            p[:, H, :, positions] = 0.0
            p[:, H] = p[:, H] / p[:, H].sum(-1, keepdim=True).clamp_min(1e-9)
            return p
        return nm, hook

    def necessity(ko_sc, framed_sc, eff):
        if abs(eff) <= MIN_EFFECT:
            return None
        return (ko_sc - framed_sc) / eff

    def max_attn_scan(ids, positions):
        """Max attn-to-anchor over ALL heads at readout; returns (best_attn, (L,H), full {head->attn per layer})."""
        if not positions:
            return 0.0, None
        cache = {}
        def grab(p, hook):
            cache[hook.name] = p[0, :, -1, :].detach().float()
            return p
        with torch.no_grad():
            model.run_with_hooks(ids, fwd_hooks=[(pat_filter, grab)])
        best, best_lh = -1.0, None
        for L in range(nL):
            attn = cache[f"blocks.{L}.attn.hook_pattern"][:, positions].sum(-1)
            h = int(attn.argmax())
            if float(attn[h]) > best:
                best, best_lh = float(attn[h]), (L, h)
        return best, best_lh

    def reader_anchor_attn(ids, positions, L, H, zhook=None):
        store = {}
        def grab(p, hook):
            store["p"] = p[0, H, -1, :].detach().float()
            return p
        hooks = ([zhook] if zhook else []) + [(f"blocks.{L}.attn.hook_pattern", grab)]
        with torch.no_grad():
            model.run_with_hooks(ids, fwd_hooks=hooks)
        return float(store["p"][positions].sum()) if positions else 0.0

    # ---- pass 1: per-pair effect + max-attn reader (always; cheap) ----
    prepared = []
    for pr in PAIRS:
        r, c, w = pr["r"], pr["c"], pr["w"]
        cid, aid = first(" " + c), first(" " + w)
        neutral = model.to_tokens(STEM.format(r=r)).to(device)
        framed = model.to_tokens(SALIENCE.format(w=w, r=r) + STEM.format(r=r)).to(device)
        s_fr = score(framed, cid, aid)
        eff = score(neutral, cid, aid) - s_fr
        apos = tok_pos(framed[0].tolist(), w)
        cpos = [p for p in tok_pos(framed[0].tolist(), CONTROL_WORD) if p not in apos]
        max_attn, max_lh = max_attn_scan(framed, apos)
        prepared.append({"r": r, "cid": cid, "aid": aid, "framed": framed, "s_fr": s_fr,
                         "eff": eff, "apos": apos, "cpos": cpos, "max_attn": max_attn, "max_lh": max_lh})
        print(f"[{r:<12}] eff={eff:+.2f}  max_attn->anchor={max_attn:.3f} @ "
              f"L{max_lh[0]}.H{max_lh[1]}" if max_lh else f"[{r}] eff={eff:+.2f} (no anchor)", flush=True)

    # reader for the router pass
    maxheads = [tuple(p["max_lh"]) for p in prepared if p["max_lh"]]
    modal_maxattn = max(set(maxheads), key=maxheads.count) if maxheads else None
    READER = modal_maxattn if reader_arg == "auto" else tuple(reader_arg)
    print(f"[reader] {'auto modal-max-attn' if reader_arg=='auto' else 'arg'} -> "
          f"L{READER[0]}.H{READER[1]}" if READER else "[reader] none", flush=True)

    active = [p for p in prepared if abs(p["eff"]) > MIN_EFFECT and p["apos"]]
    vacuous = len(active) == 0

    nec_sum = collections.defaultdict(float); nec_n = collections.defaultdict(int)
    router_sum = collections.defaultdict(float); router_n = collections.defaultdict(int)
    per_pair = []
    if vacuous:
        print(f"[skip] NO pair clears |effect|>{MIN_EFFECT} -> necessity-localization vacuous "
              f"(salience copy collapsed at scale; cf FRAMING sec 10.1). Reporting effects + max-attn only.",
              flush=True)
        for p in prepared:
            per_pair.append({"region": p["r"], "effect": round(p["eff"], 3),
                             "max_attn_to_anchor": round(p["max_attn"], 3),
                             "max_attn_head": list(p["max_lh"]) if p["max_lh"] else None,
                             "necessity_localized": False})
    else:
        for p in active:
            r, cid, aid, framed, s_fr, eff, apos, cpos = (p["r"], p["cid"], p["aid"], p["framed"],
                                                          p["s_fr"], p["eff"], p["apos"], p["cpos"])
            all_nec = necessity(score(framed, cid, aid, [(pat_filter, ko_all(apos))]), s_fr, eff)
            ctrl_nec = necessity(score(framed, cid, aid, [(pat_filter, ko_all(cpos))]), s_fr, eff) if cpos else None
            heads = []                                          # P-C: per-head over ALL heads
            for L in range(nL):
                for Hh in range(nH):
                    nm, hk = ko_head(L, Hh, apos)
                    n = necessity(score(framed, cid, aid, [(nm, hk)]), s_fr, eff)
                    heads.append((L, Hh, n))
                    if n is not None:
                        nec_sum[(L, Hh)] += n; nec_n[(L, Hh)] += 1
            base_attn = reader_anchor_attn(framed, apos, *READER)   # N-2: router z-ablation, layers 0..reader
            routers = []
            for L in range(READER[0]):
                for Hh in range(nH):
                    def zhook_fn(z, hook, Hh=Hh):
                        z[:, :, Hh, :] = 0.0
                        return z
                    a = reader_anchor_attn(framed, apos, *READER, zhook=(f"blocks.{L}.attn.hook_z", zhook_fn))
                    drop = base_attn - a
                    routers.append((L, Hh, drop))
                    router_sum[(L, Hh)] += drop; router_n[(L, Hh)] += 1
            hs = sorted([h for h in heads if h[2] is not None], key=lambda x: x[2], reverse=True)
            rs = sorted(routers, key=lambda x: x[2], reverse=True)
            per_pair.append({"region": r, "effect": round(eff, 3), "all_nec": all_nec, "ctrl_nec": ctrl_nec,
                             "max_attn_to_anchor": round(p["max_attn"], 3),
                             "max_attn_head": list(p["max_lh"]) if p["max_lh"] else None,
                             "base_reader_attn": round(base_attn, 3), "necessity_localized": True,
                             "top_heads_nec": [{"L": L, "H": H, "nec": round(n, 3)} for L, H, n in hs[:5]],
                             "top_routers": [{"L": L, "H": H, "drop": round(d, 3)} for L, H, d in rs[:5]]})
            print(f"[{r}] eff={eff:+.2f} all_nec={all_nec} top_head=L{hs[0][0]}.H{hs[0][1]}({hs[0][2]:.2f}) "
                  f"base_attn={base_attn:.2f} top_router=L{rs[0][0]}.H{rs[0][1]}(drop {rs[0][2]:.2f})", flush=True)
        # pairs that didn't clear MIN_EFFECT still get a row
        for p in prepared:
            if p not in active:
                per_pair.append({"region": p["r"], "effect": round(p["eff"], 3),
                                 "max_attn_to_anchor": round(p["max_attn"], 3),
                                 "max_attn_head": list(p["max_lh"]) if p["max_lh"] else None,
                                 "necessity_localized": False})

    mean_nec = sorted([{"L": L, "H": H, "mean_nec": round(nec_sum[(L, H)] / nec_n[(L, H)], 4)}
                       for (L, H) in nec_sum], key=lambda d: d["mean_nec"], reverse=True)
    mean_router = sorted([{"L": L, "H": H, "mean_drop": round(router_sum[(L, H)] / router_n[(L, H)], 4)}
                          for (L, H) in router_sum], key=lambda d: d["mean_drop"], reverse=True)
    concentration = None
    if mean_nec:
        top1 = mean_nec[0]["mean_nec"]; top5 = sum(d["mean_nec"] for d in mean_nec[:5])
        n_pos = sum(1 for d in mean_nec if d["mean_nec"] > 0.1)
        reader_rank = next((i for i, d in enumerate(mean_nec) if (d["L"], d["H"]) == tuple(READER)), None)
        concentration = {"top1_mean_nec": top1, "top5_sum": round(top5, 3),
                         "n_heads_nec_gt_0.1": n_pos, "reader_rank": reader_rank}
    out = {"model": name, "sweep": f"all_{nL*nH}_heads", "cue": "salience", "n_pairs": len(PAIRS),
           "reader": list(READER) if READER else None,
           "reader_source": reader_arg if reader_arg != "auto" else "auto-modal-max-attn",
           "necessity_localization_vacuous": vacuous,
           "n_pairs_clearing_min_effect": len(active),
           "concentration": concentration,
           "mean_nec_top15": mean_nec[:15], "mean_router_top10": mean_router[:10], "per_pair": per_pair}
    Path("out").mkdir(exist_ok=True)
    Path(f"out/localize_salience_{tag}.json").write_text(json.dumps(out, indent=2))
    if vacuous:
        print(f"[done] VACUOUS necessity-localization (no effect to localize); "
              f"max-attn modal head L{READER[0]}.H{READER[1]}" if READER else "[done] vacuous, no anchor")
    else:
        print(f"[done] concentration={concentration}; "
              f"top router L{mean_router[0]['L']}.H{mean_router[0]['H']} drop {mean_router[0]['mean_drop']}"
              if mean_router else f"[done] concentration={concentration}")


def _parse_reader(vals):
    if len(vals) == 1 and vals[0] == "auto":
        return "auto"
    if len(vals) == 2:
        return [int(vals[0]), int(vals[1])]
    raise SystemExit("--reader takes 'auto' or two ints (L H)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="google/gemma-2-2b")
    ap.add_argument("--tag", default="2b")
    ap.add_argument("--reader", nargs="+", default=["18", "5"], help="'auto' or 'L H' (default 18 5 = 2b)")
    a = ap.parse_args()
    run(a.name, a.tag, _parse_reader(a.reader))
