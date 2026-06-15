"""Scale test of the salience-copy mechanism (FRAMING_NOTES sec 8, generalized).

job_chat_mechanism.py hardcoded the gemma-2-2b reader head L18.H5. To validate the
sec-8 picture on a larger same-family model (gemma-2-9b / -it), the reader head must
be RE-LOCALIZED per model: this scans attention-to-anchor across ALL (layer, head)
at the readout and reports the single head that most attends the anchor, plus the
model-agnostic effect and all-heads anchor-knockout necessity.

Core questions, per model:
  base : is the copy present?  effect > 0, all-heads necessity ~1, some head's
         attention-to-anchor high (the reader).
  it   : did RLHF remove it?   effect ~0 / sign-flipped, all-heads necessity drops,
         AND max attention-to-anchor over ALL heads is low (copy gone, not merely
         a different head) -- the disambiguator from sec 8.

  python job_scale_mechanism.py --name google/gemma-2-9b     --tag 9b_base
  python job_scale_mechanism.py --name google/gemma-2-9b-it  --tag 9b_it  --chat
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

PAIRS = [("Australia", "Sydney", "Canberra"), ("Texas", "Houston", "Austin"),
         ("Canada", "Toronto", "Ottawa"), ("Switzerland", "Zurich", "Bern"),
         ("Morocco", "Casablanca", "Rabat")]
STEM = "The capital of {r} is the city of"
MIN_EFFECT = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="HF repo")
    ap.add_argument("--tag", required=True, help="output-file suffix")
    ap.add_argument("--chat", action="store_true", help="also test chat-template format")
    args = ap.parse_args()
    print(f"[load] HookedTransformer {args.name} on {DEVICE}", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(args.name, dtype=torch.bfloat16, device=DEVICE)
    model.eval()
    tok = model.tokenizer
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    print(f"[load] done (n_layers={nL}, n_heads={nH})", flush=True)

    first = lambda s: tok.encode(s, add_special_tokens=False)[0]
    pat_filter = lambda nm: nm.endswith("hook_pattern")

    def score(last_logits, cid, aid):
        lp = torch.log_softmax(last_logits.float(), -1)
        return float(lp[cid] - lp[aid])

    def last_logits(ids, hooks=None):
        with torch.no_grad():
            if hooks:
                return model.run_with_hooks(ids, fwd_hooks=hooks)[0, -1]
            return model(ids)[0, -1]

    def anchor_pos(ids_list, anchor):
        aset = set(model.to_tokens(anchor, prepend_bos=False)[0].tolist())
        return [i for i, t in enumerate(ids_list) if t in aset and i > 0]

    def ko_all(positions):
        def hook(pattern, hook):
            pattern[:, :, :, positions] = 0.0
            return pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-9)
        return hook

    def reader_scan(ids, positions):
        """Max attention-to-anchor over ALL (layer, head) at the readout query."""
        if not positions:
            return 0.0, None
        cache = {}
        def grab(pattern, hook):
            cache[hook.name] = pattern[0, :, -1, :].detach().float()  # [head, key]
            return pattern
        with torch.no_grad():
            model.run_with_hooks(ids, fwd_hooks=[(pat_filter, grab)])
        best, best_lh = 0.0, None
        for L in range(nL):
            p = cache.get(f"blocks.{L}.attn.hook_pattern")
            if p is None:
                continue
            attn = p[:, positions].sum(-1)         # [head] attn mass on anchor
            h = int(attn.argmax())
            if float(attn[h]) > best:
                best, best_lh = float(attn[h]), (L, h)
        return best, best_lh

    def build_bare(region, frame):
        return model.to_tokens(frame + STEM.format(r=region)).to(DEVICE)

    def build_chat(region, frame):
        user = frame + f"What is the capital of {region}?"
        ids = tok.apply_chat_template([{"role": "user", "content": user}],
                                      add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        pref = tok(STEM.format(r=region), add_special_tokens=False, return_tensors="pt").input_ids
        return torch.cat([ids, pref], dim=1).to(DEVICE)

    conditions = [("bare", build_bare)] + ([("chat", build_chat)] if args.chat else [])

    results = []
    for region, anchor, cap in PAIRS:
        cid, aid = first(" " + cap), first(" " + anchor)
        salience = f"{anchor} is the most famous city in {region}. "
        rec = {"pair": f"{region}->{cap}", "anchor": anchor, "conditions": {}}
        print(f"\n=== {region} (cap {cap}, anchor {anchor}) ===")
        for cname, build in conditions:
            n_sc = score(last_logits(build(region, "")), cid, aid)
            s_ids = build(region, salience)
            s_sc = score(last_logits(s_ids), cid, aid)
            eff = n_sc - s_sc
            apos = anchor_pos(s_ids[0].tolist(), anchor)
            nec = None
            if apos and abs(eff) > MIN_EFFECT:
                ko_sc = score(last_logits(s_ids, hooks=[(pat_filter, ko_all(apos))]), cid, aid)
                nec = (ko_sc - s_sc) / eff
            max_attn, reader_lh = reader_scan(s_ids, apos)
            rec["conditions"][cname] = {
                "neutral": n_sc, "salience": s_sc, "effect": eff,
                "anchor_pos": apos, "allheads_necessity": nec,
                "max_attn_to_anchor": max_attn,
                "reader_head": list(reader_lh) if reader_lh else None,
            }
            ns = f"{nec:+.2f}" if nec is not None else "n/a"
            rh = f"L{reader_lh[0]}.H{reader_lh[1]}" if reader_lh else "n/a"
            print(f"  [{cname:>4}] effect={eff:+.2f}  allheads_nec={ns}  "
                  f"max_attn->anchor={max_attn:.2f} @ {rh}")
        results.append(rec)

    summary = {"model": args.name, "n_layers": nL, "n_heads": nH}
    for cname, _ in conditions:
        effs = [r["conditions"][cname]["effect"] for r in results]
        attns = [r["conditions"][cname]["max_attn_to_anchor"] for r in results]
        necs = [r["conditions"][cname]["allheads_necessity"] for r in results
                if r["conditions"][cname]["allheads_necessity"] is not None]
        # modal reader head
        heads = [tuple(r["conditions"][cname]["reader_head"]) for r in results
                 if r["conditions"][cname]["reader_head"]]
        modal = max(set(heads), key=heads.count) if heads else None
        summary[cname] = {
            "mean_effect": statistics.mean(effs),
            "mean_max_attn_to_anchor": statistics.mean(attns),
            "mean_allheads_necessity": statistics.mean(necs) if necs else None,
            "modal_reader_head": list(modal) if modal else None,
            "reader_heads_per_pair": heads,
        }
    print("\n[summary]", json.dumps(summary, indent=2))
    Path("out").mkdir(exist_ok=True)
    out = f"out/scale_mechanism_{args.tag}.json"
    Path(out).write_text(json.dumps({"summary": summary, "pairs": results}, indent=2))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
