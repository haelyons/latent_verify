"""Frontier A -- what did RLHF do to the salience-copy circuit? (FRAMING_NOTES sec 6 P1, mechanistic)

CHAT_FORMAT_FINDINGS established BEHAVIOUR: gemma-2-2b-it shows ~0 latent pull and
rebuts the false premise (the salience flip does not transfer). What it did NOT do
is localize the CIRCUIT in -it: is the attention-copy mechanism (sec 3.7) still
present but downstream-overridden, or structurally gone? And does the universal
reader head L18.H5 (sec 3.10, attention-to-anchor ~0.84 in base) still lock onto
the anchor under RLHF?

This runs the sec-3.7 anchor-knockout AND the sec-3.10 reader-head probe on the
SAME HookedTransformer stack for base and it, so the comparison is within-stack
(immune to library-version drift). base/bare reproduces sec 3.7 as a positive
control; it/bare isolates the RLHF WEIGHT change; it/chat is the realistic regime.

Per (pair, condition) we measure at the readout stem:
  effect     score(neutral) - score(salience), score = logp(cap) - logp(anchor)
  allheads   anchor-knockout necessity (sec 3.7): zero attn to anchor (all L/H),
             renormalize; necessity = (ko_score - salience_score)/effect
  L18H5_attn reader-head signature (sec 3.10): head (18,5) attention mass on the
             anchor key(s) at the readout query position -- measured REGARDLESS of
             effect, because if effect~0 in -it this is what disambiguates
             "copy gone" (attn~0) from "copy present but overridden" (attn high)
  top_head   per-head knockout over TOP_LAYERS x heads -> principal reader in this
             model (is it still L18.H5?)

  python job_chat_mechanism.py --model base   # -> out/chat_mechanism_base.json
  python job_chat_mechanism.py --model it     # -> out/chat_mechanism_it.json
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
READER = (18, 5)                       # the sec-3.10 universal reader head
TOP_LAYERS = [0, 1, 3, 4, 7, 18]       # sec 3.9/3.10 candidate layers
MIN_EFFECT = 0.5                       # below this, necessity is a div-by-~0 -> n/a
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "it"], required=True)
    args = ap.parse_args()
    name = "google/gemma-2-2b" if args.model == "base" else "google/gemma-2-2b-it"
    is_chat = args.model == "it"
    print(f"[load] HookedTransformer {name} on {DEVICE}", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(
        name, dtype=torch.bfloat16, device=DEVICE)
    model.eval()
    tok = model.tokenizer
    n_heads = model.cfg.n_heads
    print(f"[load] done (n_heads={n_heads}, n_layers={model.cfg.n_layers})", flush=True)

    first = lambda s: tok.encode(s, add_special_tokens=False)[0]
    pat_filter = lambda nm: nm.endswith("hook_pattern")

    def score(last_logits, cid, aid):
        lp = torch.log_softmax(last_logits.float(), -1)
        return float(lp[cid] - lp[aid])          # capital - anchor margin

    def last_logits(ids, hooks=None):
        with torch.no_grad():
            if hooks:
                return model.run_with_hooks(ids, fwd_hooks=hooks)[0, -1]
            return model(ids)[0, -1]

    def anchor_pos(ids_list, anchor):
        aset = set(model.to_tokens(anchor, prepend_bos=False)[0].tolist())
        return [i for i, t in enumerate(ids_list) if t in aset and i > 0]

    def ko_all(positions):                       # zero ALL heads' attn to anchor
        def hook(pattern, hook):                 # [b, head, q, k]
            pattern[:, :, :, positions] = 0.0
            return pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-9)
        return hook

    def ko_head(layer, head, positions):         # zero ONE head's attn to anchor
        name_l = f"blocks.{layer}.attn.hook_pattern"
        def hook(pattern, hook):
            pattern[:, head, :, positions] = 0.0
            denom = pattern[:, head].sum(-1, keepdim=True).clamp_min(1e-9)
            pattern[:, head] = pattern[:, head] / denom
            return pattern
        return name_l, hook

    def reader_attn_to_anchor(ids, positions):   # head (18,5) attn mass on anchor @ readout
        layer, head = READER
        store = {}
        def grab(pattern, hook):
            store["p"] = pattern[0, head, -1, :].detach().float()
            return pattern
        with torch.no_grad():
            model.run_with_hooks(ids, fwd_hooks=[(f"blocks.{layer}.attn.hook_pattern", grab)])
        return float(store["p"][positions].sum()) if positions else 0.0

    # ---- prompt builders (each adds BOS exactly once) ----
    def build_bare(region, frame):
        return model.to_tokens(frame + STEM.format(r=region)).to(DEVICE)

    def build_chat(region, frame):
        user = frame + f"What is the capital of {region}?"
        ids = tok.apply_chat_template([{"role": "user", "content": user}],
                                      add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):            # transformers 5.x returns BatchEncoding
            ids = ids["input_ids"]
        pref = tok(STEM.format(r=region), add_special_tokens=False,
                   return_tensors="pt").input_ids
        return torch.cat([ids, pref], dim=1).to(DEVICE)

    conditions = [("bare", build_bare)]
    if is_chat:
        conditions.append(("chat", build_chat))

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

            # all-heads knockout necessity (only meaningful if there is an effect)
            nec = None
            if apos and abs(eff) > MIN_EFFECT:
                ko_sc = score(last_logits(s_ids, hooks=[(pat_filter, ko_all(apos))]), cid, aid)
                nec = (ko_sc - s_sc) / eff

            # reader-head signature -- measured REGARDLESS of effect
            r_attn = reader_attn_to_anchor(s_ids, apos)

            # per-head knockout sweep -> principal reader (necessity if effect exists)
            heads = []
            if apos and abs(eff) > MIN_EFFECT:
                for L in TOP_LAYERS:
                    for H in range(n_heads):
                        nm, hk = ko_head(L, H, apos)
                        h_sc = score(last_logits(s_ids, hooks=[(nm, hk)]), cid, aid)
                        heads.append({"layer": L, "head": H, "necessity": (h_sc - s_sc) / eff})
                heads.sort(key=lambda d: d["necessity"], reverse=True)

            rec["conditions"][cname] = {
                "neutral": n_sc, "salience": s_sc, "effect": eff,
                "anchor_pos": apos, "allheads_necessity": nec,
                "reader_L18H5_attn_to_anchor": r_attn,
                "top_heads": heads[:5],
            }
            ns = f"{nec:+.2f}" if nec is not None else "n/a"
            th = (f"{heads[0]['layer']}.{heads[0]['head']}={heads[0]['necessity']:+.2f}"
                  if heads else "n/a")
            print(f"  [{cname:>4}] effect={eff:+.2f}  allheads_nec={ns}  "
                  f"L18.H5->anchor={r_attn:.2f}  top_head={th}")
        results.append(rec)

    # ---- summary ----
    def col(cname, key):
        xs = [r["conditions"][cname][key] for r in results if cname in r["conditions"]
              and r["conditions"][cname][key] is not None]
        return xs
    summary = {"model": name, "chat_available": is_chat, "reader_head": list(READER)}
    for cname, _ in conditions:
        effs = col(cname, "effect")
        attns = col(cname, "reader_L18H5_attn_to_anchor")
        necs = col(cname, "allheads_necessity")
        summary[cname] = {
            "mean_effect": statistics.mean(effs) if effs else None,
            "mean_reader_L18H5_attn": statistics.mean(attns) if attns else None,
            "mean_allheads_necessity": statistics.mean(necs) if necs else None,
            "n_necessity_measured": len(necs),
        }
    print("\n[summary]", json.dumps(summary, indent=2))

    Path("out").mkdir(exist_ok=True)
    out = f"out/chat_mechanism_{args.model}.json"
    Path(out).write_text(json.dumps({"summary": summary, "pairs": results}, indent=2))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
