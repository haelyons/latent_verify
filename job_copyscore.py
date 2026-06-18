"""N-3 + N-4 (SEQUENCE_170626): field-standard legibility for the sec-3.7 attention-knockout
necessity, for the Colab/LessWrong artifact.

N-3 COPY-SCORE (Olsson 2022 / Wang 2023): does a reader head's OV circuit actually COPY
the anchor token? Compose W_U @ (W_O[L,H] @ W_V[L,H]) @ W_E for the anchor token id and read
the rank of the anchor in the resulting logits. Low rank / top-k membership = a copy head.
Reported for the --reader head and control heads, over the 5 anchor tokens. With --sweep,
score ALL (layer, head) and report the heads that most copy the anchor ("find any OV-copy
reader") -- the scale port (2b reader L18.H5 does NOT transfer to 9b; re-localize from
scratch). (Approximate: the OV-unembed composition omits per-position LayerNorm; standard
for a copy-score.)

N-4 OUTPUT ABLATION: a lighter, more standard intervention than the heavy all-layers
zero-attention+renormalize knockout -- zero the --reader head's output (hook_z) on the
framed prompt and read the fraction of the salience effect reverted. Compared to the
all-heads attention-knockout necessity (~1 on 2b) and the per-head attention-knockout
necessity for the reader.

  python job_copyscore.py                                         # -> out/copyscore_2b.json (reader 18 5)
  python job_copyscore.py --name google/gemma-2-9b --tag 9b --reader auto --sweep
"""
import argparse
import json
import statistics
from pathlib import Path

STEM = "The capital of {r} is the city of"
SALIENCE = "{w} is the most famous city in {r}. "
CONTROL_HEADS = [(0, 2), (7, 1), (10, 4)]   # an early writer, a mid head, an arbitrary control
MIN_EFFECT = 0.5
# reference prompt for --reader auto (max attn-to-anchor at readout)
REF = {"r": "Australia", "c": "Canberra", "w": "Sydney"}
PAIRS = [
    {"r": "Australia",   "c": "Canberra",    "w": "Sydney"},
    {"r": "Texas",       "c": "Austin",      "w": "Houston"},
    {"r": "Canada",      "c": "Ottawa",      "w": "Toronto"},
    {"r": "Switzerland", "c": "Bern",        "w": "Zurich"},
    {"r": "Florida",     "c": "Tallahassee", "w": "Miami"},
]


def run(name, tag, reader_arg, do_sweep):
    import torch
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {name} on {device}", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tok = model.tokenizer
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    n_v = model.W_V.shape[1]                      # GQA: W_V may store n_key_value_heads
    grp = nH // n_v if n_v and n_v < nH else 1    # query-head -> value-head map
    first = lambda s: tok.encode(s, add_special_tokens=False)[0]
    pat_filter = lambda nm: nm.endswith("hook_pattern")

    def tok_pos(ids_list, text):
        tset = set(model.to_tokens(text, prepend_bos=False)[0].tolist())
        tset |= set(model.to_tokens(" " + text, prepend_bos=False)[0].tolist())
        return [i for i, t in enumerate(ids_list) if t in tset and i > 0]

    # ---- localize the reader (max attn-to-anchor at readout) if --reader auto ----
    def localize_reader():
        framed = model.to_tokens(SALIENCE.format(**REF) + STEM.format(r=REF["r"])).to(device)
        apos = tok_pos(framed[0].tolist(), REF["w"])
        cache = {}
        def grab(p, hook):
            cache[hook.name] = p[0, :, -1, :].detach().float()
            return p
        with torch.no_grad():
            model.run_with_hooks(framed, fwd_hooks=[(pat_filter, grab)])
        best, best_lh = -1.0, None
        for L in range(nL):
            attn = cache[f"blocks.{L}.attn.hook_pattern"][:, apos].sum(-1)  # [head]
            h = int(attn.argmax())
            if float(attn[h]) > best:
                best, best_lh = float(attn[h]), (L, h)
        print(f"[auto-reader] max attn->anchor {best:.3f} @ L{best_lh[0]}.H{best_lh[1]} "
              f"(on '{REF['w']}'/{REF['r']})", flush=True)
        return best_lh, best

    auto_attn = None
    if reader_arg == "auto":
        READER, auto_attn = localize_reader()
    else:
        READER = tuple(reader_arg)

    def score(ids, cid, aid, hooks=None):
        with torch.no_grad():
            ll = (model.run_with_hooks(ids, fwd_hooks=hooks)[0, -1] if hooks else model(ids)[0, -1]).float()
        lp = torch.log_softmax(ll, -1)
        return float(lp[cid] - lp[aid])

    def ko_all(positions):
        def hook(p, hook):
            p[:, :, :, positions] = 0.0
            return p / p.sum(-1, keepdim=True).clamp_min(1e-9)
        return hook

    def ko_head_attn(L, H, positions):
        nm = f"blocks.{L}.attn.hook_pattern"
        def hook(p, hook):
            p[:, H, :, positions] = 0.0
            p[:, H] = p[:, H] / p[:, H].sum(-1, keepdim=True).clamp_min(1e-9)
            return p
        return nm, hook

    def ablate_z(L, H):
        nm = f"blocks.{L}.attn.hook_z"
        def hook(z, hook):
            z[:, :, H, :] = 0.0
            return z
        return nm, hook

    def necessity(ko_sc, fr_sc, eff):
        if abs(eff) <= MIN_EFFECT:
            return None
        return (ko_sc - fr_sc) / eff

    # ---- N-3 copy-score: W_U @ (W_O @ W_V) @ W_E[anchor], rank of anchor ----
    def copy_rank(L, H, aid):
        vH = H // grp if grp > 1 else H
        W_OV = model.W_V[L, vH] @ model.W_O[L, H]            # [d_model, d_model]
        e = model.W_E[aid].to(W_OV.dtype)                    # [d_model]
        ov = e @ W_OV                                        # [d_model]
        normed = model.ln_final(ov.unsqueeze(0).unsqueeze(0))[0, 0]
        logits = (normed @ model.W_U).float()                # [d_vocab]
        rank = int((logits > logits[aid]).sum().item())
        return rank

    anchor_ids = [first(" " + pr["w"]) for pr in PAIRS]

    def head_copyscore(L, H):
        ranks = [copy_rank(L, H, aid) for aid in anchor_ids]
        return {"anchor_ranks": ranks, "median_rank": int(statistics.median(ranks)),
                "frac_top5": round(sum(r < 5 for r in ranks) / len(ranks), 2),
                "frac_top20": round(sum(r < 20 for r in ranks) / len(ranks), 2)}

    copyscore = {}
    for (L, H) in [READER] + CONTROL_HEADS:
        cs = head_copyscore(L, H)
        copyscore[f"L{L}.H{H}"] = cs
        print(f"[copyscore L{L}.H{H}] ranks={cs['anchor_ranks']} median={cs['median_rank']} "
              f"top5={cs['frac_top5']}", flush=True)

    # ---- N-3 SWEEP: copy-score over ALL heads, rank by median anchor-rank ----
    sweep = None
    if do_sweep:
        print(f"[sweep] copy-score over all {nL*nH} heads ...", flush=True)
        rows = []
        for L in range(nL):
            for H in range(nH):
                cs = head_copyscore(L, H)
                rows.append({"L": L, "H": H, "median_rank": cs["median_rank"],
                             "frac_top5": cs["frac_top5"], "anchor_ranks": cs["anchor_ranks"]})
        rows.sort(key=lambda d: (d["median_rank"], -d["frac_top5"]))
        reader_sweep_rank = next((i for i, d in enumerate(rows)
                                  if (d["L"], d["H"]) == tuple(READER)), None)
        sweep = {"top15_by_median_rank": rows[:15],
                 "n_heads_median_top5": sum(1 for d in rows if d["median_rank"] < 5),
                 "n_heads_median_top20": sum(1 for d in rows if d["median_rank"] < 20),
                 "reader_sweep_rank": reader_sweep_rank}
        top = rows[0]
        print(f"[sweep] best OV-copy head L{top['L']}.H{top['H']} median_rank={top['median_rank']} "
              f"top5={top['frac_top5']}; #heads median<5={sweep['n_heads_median_top5']}; "
              f"reader L{READER[0]}.H{READER[1]} sweep-rank={reader_sweep_rank}", flush=True)

    # ---- N-4 output-ablation necessity vs attention-knockout, per pair ----
    abl = []
    for pr in PAIRS:
        r, c, w = pr["r"], pr["c"], pr["w"]
        cid, aid = first(" " + c), first(" " + w)
        neutral = model.to_tokens(STEM.format(r=r)).to(device)
        framed = model.to_tokens(SALIENCE.format(w=w, r=r) + STEM.format(r=r)).to(device)
        s_fr = score(framed, cid, aid)
        eff = score(neutral, cid, aid) - s_fr
        apos = tok_pos(framed[0].tolist(), w)
        nec_allheads = necessity(score(framed, cid, aid, [(pat_filter, ko_all(apos))]), s_fr, eff)
        nm, hk = ko_head_attn(*READER, apos)
        nec_reader_attn = necessity(score(framed, cid, aid, [(nm, hk)]), s_fr, eff)
        nm2, hk2 = ablate_z(*READER)
        nec_reader_out = necessity(score(framed, cid, aid, [(nm2, hk2)]), s_fr, eff)
        abl.append({"region": r, "effect": round(eff, 3), "nec_allheads_attn_ko": nec_allheads,
                    "nec_reader_attn_ko": nec_reader_attn, "nec_reader_output_ablation": nec_reader_out})
        print(f"[abl {r:<12}] eff={eff:+.2f} allheads={nec_allheads} readerAttnKO={nec_reader_attn} "
              f"readerOutAbl={nec_reader_out}", flush=True)

    mean = lambda k: round(statistics.mean([a[k] for a in abl if a[k] is not None]), 3) \
        if any(a[k] is not None for a in abl) else None
    out = {"model": name, "reader": list(READER), "reader_source": reader_arg if reader_arg != "auto" else "auto-localized",
           "reader_auto_attn": auto_attn,
           "copy_score": copyscore, "copy_score_sweep": sweep,
           "ablation_summary": {"mean_nec_allheads_attn_ko": mean("nec_allheads_attn_ko"),
                                "mean_nec_reader_attn_ko": mean("nec_reader_attn_ko"),
                                "mean_nec_reader_output_ablation": mean("nec_reader_output_ablation")},
           "ablation_per_pair": abl}
    Path("out").mkdir(exist_ok=True)
    Path(f"out/copyscore_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[done] reader L{READER[0]}.H{READER[1]} copy-score median_rank="
          f"{copyscore[f'L{READER[0]}.H{READER[1]}']['median_rank']}; ablation {out['ablation_summary']}")


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
    ap.add_argument("--sweep", action="store_true", help="copy-score ALL heads, find any OV-copy reader")
    a = ap.parse_args()
    run(a.name, a.tag, _parse_reader(a.reader), a.sweep)
