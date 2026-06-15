"""Forced-choice QUESTION flip: same circuit as the salience copy, or a different failure mode?

The salience copy was base-FRAGMENT-only (sec 6/8: the QA scaffold disengages it). But a
forced-choice QUESTION -- "Which city is the capital of X, A or B?" with A,B both non-capitals --
flips gemma-2-2b in QA form (probe: 3/3). This asks whether that flip rides the SAME
attention-copy machinery (the sec-3.10 reader head L18.H5 reading the offered option token)
or something else.

Mirrors the sec-3.7 / job_scale_mechanism knockout+score machinery, with:
  neutral  = plain question     "Question: What is the capital of X?\nAnswer:"
  salience = forced-choice      "Question: Which city is the capital of X, A or B?\nAnswer:"
  anchor   = the offered option tokens (A, B)
  score    = logp(capital) - logp(chosen distractor)   at the answer slot (last position)
  effect   = score(plain) - score(forced)              (>0 => the forced-choice pulls to a distractor)
  necessity= (score_knockout - score_forced) / effect  (1.0 => knockout fully reverts)

Reports all-heads necessity, the L18.H5-specific necessity, and the top head from a candidate
sweep -- so "same circuit" (necessity high AND reader is L18.H5) is distinguishable from
"different mechanism" (necessity ~0).

  python job_forcedchoice.py --name google/gemma-2-2b --tag fc_2b
"""
import argparse
import json
import statistics

import torch
from transformer_lens import HookedTransformer

PAIRS = [("Australia", "Canberra", "Sydney", "Melbourne"),
         ("Texas", "Austin", "Houston", "Dallas"),
         ("Canada", "Ottawa", "Toronto", "Montreal"),
         ("Switzerland", "Bern", "Zurich", "Geneva"),
         ("Florida", "Tallahassee", "Miami", "Orlando")]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SALIENCE_READER = (18, 5)          # the sec-3.10 reader head, on 2b
CAND_LAYERS = [0, 1, 3, 4, 7, 18]  # sec-3.9 search space
MIN_EFFECT = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    print(f"[load] {a.name} on {DEVICE}", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(a.name, dtype=torch.bfloat16, device=DEVICE)
    model.eval()
    tok = model.tokenizer
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    print(f"[load] done (n_layers={nL}, n_heads={nH})", flush=True)

    first = lambda s: tok.encode(s, add_special_tokens=False)[0]
    pat = lambda nm: nm.endswith("hook_pattern")
    lp = lambda logits: torch.log_softmax(logits.float(), -1)

    def last(ids, hooks=None):
        with torch.no_grad():
            return model.run_with_hooks(ids, fwd_hooks=hooks)[0, -1] if hooks else model(ids)[0, -1]

    def anchor_pos(ids_list, words):
        s = set()
        for w in words:
            for t in model.to_tokens(w, prepend_bos=False)[0].tolist():
                s.add(t)
        return [i for i, t in enumerate(ids_list) if t in s and i > 0]

    def ko_all(positions):
        def hook(p, hook):
            p[:, :, :, positions] = 0.0
            return p / p.sum(-1, keepdim=True).clamp_min(1e-9)
        return hook

    def ko_heads(positions, heads):
        by = {}
        for (l, h) in heads:
            by.setdefault(l, []).append(h)

        def mk(hs):
            def hook(p, hook):
                for h in hs:
                    p[:, h, :, positions] = 0.0
                    p[:, h] = p[:, h] / p[:, h].sum(-1, keepdim=True).clamp_min(1e-9)
                return p
            return hook
        return [(f"blocks.{l}.attn.hook_pattern", mk(hs)) for l, hs in by.items()]

    def reader_scan(ids, positions):
        if not positions:
            return 0.0, None
        cache = {}
        def grab(p, hook):
            cache[hook.name] = p[0, :, -1, :].detach().float()
            return p
        with torch.no_grad():
            model.run_with_hooks(ids, fwd_hooks=[(pat, grab)])
        best, blh = 0.0, None
        for L in range(nL):
            p = cache.get(f"blocks.{L}.attn.hook_pattern")
            if p is None:
                continue
            attn = p[:, positions].sum(-1)
            h = int(attn.argmax())
            if float(attn[h]) > best:
                best, blh = float(attn[h]), (L, h)
        return best, blh

    def sc(L, cid, dc):
        return float(L[cid] - L[dc])

    results = []
    for region, cap, d1, d2 in PAIRS:
        cid, d1id, d2id = first(" " + cap), first(" " + d1), first(" " + d2)
        plain = model.to_tokens(f"Question: What is the capital of {region}?\nAnswer:").to(DEVICE)
        forced = model.to_tokens(f"Question: Which city is the capital of {region}, {d1} or {d2}?\nAnswer:").to(DEVICE)
        Lp, Lf = lp(last(plain)), lp(last(forced))
        dc = d1id if float(Lf[d1id]) >= float(Lf[d2id]) else d2id
        dcname = d1 if dc == d1id else d2
        sp, sf = sc(Lp, cid, dc), sc(Lf, cid, dc)
        eff = sp - sf
        flipped = int(Lf.argmax()) in (d1id, d2id)
        apos = anchor_pos(forced[0].tolist(), [" " + d1, " " + d2, d1, d2])
        max_attn, reader = reader_scan(forced, apos)
        nec_all = nec_reader = reverted = None
        top = None
        if apos and abs(eff) > MIN_EFFECT:
            ko_log = last(forced, hooks=[(pat, ko_all(apos))])
            nec_all = (sc(lp(ko_log), cid, dc) - sf) / eff
            reverted = int(ko_log.argmax()) == cid
            nec_reader = (sc(lp(last(forced, hooks=ko_heads(apos, [SALIENCE_READER]))), cid, dc) - sf) / eff
            sweep = []
            for l in CAND_LAYERS:
                for h in range(nH):
                    nh = (sc(lp(last(forced, hooks=ko_heads(apos, [(l, h)]))), cid, dc) - sf) / eff
                    sweep.append(((l, h), nh))
            top = max(sweep, key=lambda kv: kv[1])
        rec = {"pair": f"{region}->{cap}", "chosen_distractor": dcname, "flipped": flipped,
               "score_plain": sp, "score_forced": sf, "effect": eff,
               "necessity_allheads": nec_all, "necessity_L18H5": nec_reader,
               "argmax_reverts_to_capital": reverted,
               "max_attn_to_options": max_attn, "reader_head": list(reader) if reader else None,
               "top_head": list(top[0]) if top else None, "top_head_necessity": top[1] if top else None}
        results.append(rec)
        ns = f"{nec_all:+.2f}" if nec_all is not None else "n/a"
        rs = f"{nec_reader:+.2f}" if nec_reader is not None else "n/a"
        th = f"L{top[0][0]}.H{top[0][1]} ({top[1]:+.2f})" if top else "n/a"
        rh = f"L{reader[0]}.H{reader[1]}" if reader else "n/a"
        print(f"[{region:12}] flip={flipped} chose {dcname:9} eff={eff:+.2f} | "
              f"allheads_nec={ns} L18.H5_nec={rs} revert={reverted} | "
              f"attn->opts={max_attn:.2f}@{rh} top={th}")

    necs = [r["necessity_allheads"] for r in results if r["necessity_allheads"] is not None]
    rnec = [r["necessity_L18H5"] for r in results if r["necessity_L18H5"] is not None]
    heads = [tuple(r["reader_head"]) for r in results if r["reader_head"]]
    tops = [tuple(r["top_head"]) for r in results if r["top_head"]]
    summary = {"model": a.name, "n_flipped": sum(r["flipped"] for r in results),
               "n_pairs": len(results),
               "mean_necessity_allheads": statistics.mean(necs) if necs else None,
               "mean_necessity_L18H5": statistics.mean(rnec) if rnec else None,
               "modal_reader_head": list(max(set(heads), key=heads.count)) if heads else None,
               "top_heads": [list(t) for t in tops]}
    print("\n[summary]", json.dumps(summary, indent=2))
    from pathlib import Path
    Path("out").mkdir(exist_ok=True)
    out = f"out/forcedchoice_{a.tag}.json"
    Path(out).write_text(json.dumps({"summary": summary, "pairs": results}, indent=2))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
