"""ARC2C -- format x weights 2x2: is the salience-copy disengagement caused by the
chat/QA *format* or by the RLHF *weights*?

Unifies three prior results onto ONE stack and ONE readout:
  - base/fragment   = sec 3.7  (copy fully engaged, knockout ~1.0)
  - base/QA         = base_attn_qa / CHAT_FORMAT_FINDINGS (QA scaffold disengages)
  - it/fragment     = sec 8 it/bare (RLHF weight change, identical prompt to base)
  - it/QA(chat)     = sec 8 it/chat (realistic regime)

Factor reading (the whole point of the 2x2):
  base/fragment -> base/QA      isolates FORMAT  (weights fixed)
  base/fragment -> it/fragment  isolates WEIGHTS (format fixed)
  the interaction tells us whether format and weights act on the same locus.

Per (model, format) cell, at a readout token matched across cells ("...is the city of"):
  effect      = score(neutral) - score(salience),  score = logp(cap) - logp(anchor)
  allheads    = anchor-knockout necessity (sec 3.7): zero attn to anchor (all L/H),
                renormalize; necessity = (ko_score - salience_score)/effect
  L18H5_attn  = reader-head (18,5) attention mass on the anchor key(s) at the readout
                query -- measured REGARDLESS of effect (disambiguates copy-gone from
                copy-present-but-overridden)

  python job_arc2c_format_weights.py            # both models, both formats -> out/arc2c_format_weights.json
"""
import json
import statistics
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

PAIRS = [("Australia", "Sydney", "Canberra"), ("Texas", "Houston", "Austin"),
         ("Canada", "Toronto", "Ottawa"), ("Switzerland", "Zurich", "Bern"),
         ("Morocco", "Casablanca", "Rabat")]
STEM = "The capital of {r} is the city of"          # matched readout across all cells
READER = (18, 5)
MIN_EFFECT = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_model(model_key, name):
    print(f"\n[load] HookedTransformer {name} on {DEVICE}", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=DEVICE)
    model.eval()
    tok = model.tokenizer
    n_heads = model.cfg.n_heads
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
        def hook(pattern, hook):                    # [b, head, q, k]
            pattern[:, :, :, positions] = 0.0
            return pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-9)
        return hook

    def reader_attn(ids, positions):
        layer, head = READER
        store = {}
        def grab(pattern, hook):
            store["p"] = pattern[0, head, -1, :].detach().float()
            return pattern
        with torch.no_grad():
            model.run_with_hooks(ids, fwd_hooks=[(f"blocks.{layer}.attn.hook_pattern", grab)])
        return float(store["p"][positions].sum()) if positions else 0.0

    # ---- format builders (each adds BOS exactly once); readout token is the STEM tail ----
    def build_fragment(region, frame):
        return model.to_tokens(frame + STEM.format(r=region)).to(DEVICE)

    def build_qa_base(region, frame):               # plain-text QA scaffold (base has no chat template)
        scaffold = (frame + f"Question: What is the capital of {region}?\n"
                    f"Answer: " + STEM.format(r=region))
        return model.to_tokens(scaffold).to(DEVICE)

    def build_qa_chat(region, frame):               # real chat template (it)
        user = frame + f"What is the capital of {region}?"
        ids = tok.apply_chat_template([{"role": "user", "content": user}],
                                      add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        pref = tok(STEM.format(r=region), add_special_tokens=False, return_tensors="pt").input_ids
        return torch.cat([ids, pref], dim=1).to(DEVICE)

    if model_key == "base":
        formats = [("fragment", build_fragment), ("qa", build_qa_base)]
    else:
        formats = [("fragment", build_fragment), ("qa", build_qa_chat)]

    cells = {}
    for fname, build in formats:
        rows = []
        for region, anchor, cap in PAIRS:
            cid, aid = first(" " + cap), first(" " + anchor)
            salience = f"{anchor} is the most famous city in {region}. "
            n_sc = score(last_logits(build(region, "")), cid, aid)
            s_ids = build(region, salience)
            s_sc = score(last_logits(s_ids), cid, aid)
            eff = n_sc - s_sc
            apos = anchor_pos(s_ids[0].tolist(), anchor)
            nec = None
            if apos and abs(eff) > MIN_EFFECT:
                ko_sc = score(last_logits(s_ids, hooks=[(pat_filter, ko_all(apos))]), cid, aid)
                nec = (ko_sc - s_sc) / eff
            r_attn = reader_attn(s_ids, apos)
            rows.append({"pair": f"{region}->{cap}", "neutral": n_sc, "salience": s_sc,
                         "effect": eff, "allheads_necessity": nec,
                         "reader_L18H5_attn_to_anchor": r_attn, "anchor_pos": apos})
            ns = f"{nec:+.2f}" if nec is not None else "n/a"
            print(f"  [{model_key:>4}/{fname:<8}] {region:<12} effect={eff:+.2f} "
                  f"nec={ns} L18.H5->anchor={r_attn:.2f}", flush=True)
        effs = [r["effect"] for r in rows]
        attns = [r["reader_L18H5_attn_to_anchor"] for r in rows]
        necs = [r["allheads_necessity"] for r in rows if r["allheads_necessity"] is not None]
        cells[fname] = {
            "rows": rows,
            "mean_effect": statistics.mean(effs),
            "mean_reader_L18H5_attn": statistics.mean(attns),
            "mean_allheads_necessity": (statistics.mean(necs) if necs else None),
            "n_necessity_measured": len(necs),
        }
        print(f"  >> {model_key}/{fname}: mean_effect={cells[fname]['mean_effect']:+.2f} "
              f"mean_L18H5_attn={cells[fname]['mean_reader_L18H5_attn']:.3f}", flush=True)

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return cells


def main():
    out = {"reader_head": list(READER), "stem": STEM, "models": {}}
    out["models"]["base"] = run_model("base", "google/gemma-2-2b")
    out["models"]["it"] = run_model("it", "google/gemma-2-2b-it")

    # 2x2 summary on mean_effect and mean reader attention
    def cell(m, f, key):
        return out["models"][m][f][key]
    out["summary_2x2"] = {
        "mean_effect": {f"{m}/{f}": cell(m, f, "mean_effect")
                        for m in ("base", "it") for f in ("fragment", "qa")},
        "mean_L18H5_attn": {f"{m}/{f}": cell(m, f, "mean_reader_L18H5_attn")
                            for m in ("base", "it") for f in ("fragment", "qa")},
        "format_effect_base": cell("base", "fragment", "mean_effect") - cell("base", "qa", "mean_effect"),
        "weight_effect_fragment": cell("base", "fragment", "mean_effect") - cell("it", "fragment", "mean_effect"),
    }
    print("\n[summary_2x2]", json.dumps(out["summary_2x2"], indent=2), flush=True)
    Path("out").mkdir(exist_ok=True)
    Path("out/arc2c_format_weights.json").write_text(json.dumps(out, indent=2))
    print("[done] wrote out/arc2c_format_weights.json", flush=True)


if __name__ == "__main__":
    main()
