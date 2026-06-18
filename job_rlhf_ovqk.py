"""I2 (drill-deeper) -- does RLHF DELETE the salience-copy, or only GATE it?

Resolves the standing contradiction between two branches:
  - consolidated FRAMING_NOTES sec 8: "RLHF removes the copy from the weights."
  - arc2 ARC2A: "RLHF did NOT delete the copy primitive; it gated engagement.
    L18.H5's OV write-direction toward the anchor is unchanged (ov_pref base ~= it);
    what changed is the QK pattern -- where the head looks."

The two are mechanistically different and only one can be right. We decompose the
2b reader head L18.H5 into its two halves and compare base vs -it on the SAME stack:

  OV half (what the head WRITES if it reads the anchor) -- WEIGHT-ONLY, no forward.
      Compose  W_U @ ln_final(W_E[anchor] @ W_O@W_V)  and read how strongly the head's
      output pushes the anchor token (rank + softmax pref). This is the Olsson-2022 /
      job_copyscore copy-score, here read for BOTH base and -it weights. If RLHF left
      the OV matrix alone, this is ~unchanged base->it.
  QK half (WHERE the head looks) -- needs one forward/pair. Reader attention mass on
      the anchor key at the readout query (the sec-3.10 signature; chat_mechanism got
      0.58 base -> 0.016 it). This is the input-dependent engagement.

Pre-registered verdict (thresholds fixed before the run):
  OV_PRESERVED := median anchor-rank_it <= 5  AND  |mean_ov_pref_it - base|/base <= 0.30
  QK_GATED     := mean_reader_attn_base > 0.40  AND  mean_reader_attn_it < 0.10
  -> OV_PRESERVED & QK_GATED  => "GATING (ARC2A): sec-8 'removed from the weights' overstated"
  -> not OV_PRESERVED         => "DELETION: OV copy weakened in weights too (supports sec-8)"
  -> not QK_GATED             => "UNEXPECTED: behavioural disengagement is not via QK"

Usage:
  python job_rlhf_ovqk.py --selftest          # model-free; validates the metric + verdict
  python job_rlhf_ovqk.py                      # loads gemma-2-2b base & -it -> out/rlhf_ovqk_2b.json
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

READER = (18, 5)                                   # the sec-3.10 universal reader head (2b)
STEM = "The capital of {r} is the city of"
SALIENCE = "{w} is the most famous city in {r}. "  # the salient distractor (anchor = {w})
PAIRS = [                                          # (region, capital=C, anchor=W)
    ("Australia",   "Canberra",    "Sydney"),
    ("Texas",       "Austin",      "Houston"),
    ("Canada",      "Ottawa",      "Toronto"),
    ("Switzerland", "Bern",        "Zurich"),
    ("Morocco",     "Rabat",       "Casablanca"),
]

# Pre-registered thresholds (do not tune after seeing data).
OV_RANK_TOP = 5
OV_PREF_REL_TOL = 0.30
QK_BASE_MIN = 0.40
QK_IT_MAX = 0.10


def ov_copy(W_E, W_V, W_O, W_U, ln_final, L, H, grp, aid):
    """WEIGHT-ONLY OV copy-score of token `aid` for head (L,H). Returns rank+pref+cos.

    rank = number of vocab tokens the head's OV-output ranks ABOVE the anchor (0 = top).
    pref = softmax probability mass the OV-output places on the anchor token.
    cos  = cosine between the OV-output direction and the anchor's unembedding column.
    """
    vH = H // grp if grp > 1 else H                # GQA: query-head -> value-head
    W_OV = W_V[L, vH] @ W_O[L, H]                  # [d_model, d_model]
    e = W_E[aid].to(W_OV.dtype)                    # [d_model] anchor token embedding
    ov = e @ W_OV                                  # [d_model] what the head writes
    normed = ln_final(ov.unsqueeze(0).unsqueeze(0))[0, 0]
    logits = (normed @ W_U).float()                # [d_vocab]
    rank = int((logits > logits[aid]).sum().item())
    pref = float(torch.softmax(logits, -1)[aid])
    u = W_U[:, aid].float()
    cos = float(torch.nn.functional.cosine_similarity(ov.float(), u, dim=0))
    return {"rank": rank, "pref": pref, "cos": round(cos, 4)}


def verdict(base, it):
    """Pure decision from the two summaries (also exercised by --selftest)."""
    ov_preserved = (it["median_rank"] <= OV_RANK_TOP and
                    abs(it["mean_pref"] - base["mean_pref"]) <= OV_PREF_REL_TOL * max(base["mean_pref"], 1e-9))
    qk_gated = (base["mean_reader_attn"] > QK_BASE_MIN and it["mean_reader_attn"] < QK_IT_MAX)
    if not ov_preserved:
        v = "DELETION: OV copy weakened in -it weights too (supports FRAMING sec-8)"
    elif qk_gated:
        v = "GATING (ARC2A): OV copy survives in weights; RLHF gates the QK pattern. FRAMING sec-8 'removed from the weights' is OVERSTATED"
    else:
        v = "UNEXPECTED: OV preserved but QK not clearly gated -- behavioural disengagement is elsewhere"
    return {"ov_preserved": ov_preserved, "qk_gated": qk_gated, "verdict": v}


# --------------------------------------------------------------------------- real run
def run():
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pat = lambda layer: f"blocks.{layer}.attn.hook_pattern"
    per_model = {}

    for label, name in [("base", "google/gemma-2-2b"), ("it", "google/gemma-2-2b-it")]:
        print(f"[load] {name} on {device}", flush=True)
        model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
        model.eval()
        tok = model.tokenizer
        nH, n_v = model.cfg.n_heads, model.W_V.shape[1]
        grp = nH // n_v if n_v and n_v < nH else 1
        first = lambda s: tok.encode(s, add_special_tokens=False)[0]
        L, H = READER

        def reader_attn(ids, positions):                     # QK half: forward + grab pattern
            store = {}
            def grab(p, hook):
                store["p"] = p[0, H, -1, :].detach().float()  # head H, last query, all keys
                return p
            with torch.no_grad():
                model.run_with_hooks(ids, fwd_hooks=[(pat(L), grab)])
            return float(store["p"][positions].sum()) if positions else 0.0

        rows = []
        for region, cap, anchor in PAIRS:
            aid = first(" " + anchor)
            ov = ov_copy(model.W_E, model.W_V, model.W_O, model.W_U, model.ln_final, L, H, grp, aid)
            s_ids = model.to_tokens(SALIENCE.format(w=anchor, r=region) + STEM.format(r=region)).to(device)
            aset = set(model.to_tokens(anchor, prepend_bos=False)[0].tolist())
            apos = [i for i, t in enumerate(s_ids[0].tolist()) if t in aset and i > 0]
            r_attn = reader_attn(s_ids, apos)
            rows.append({"region": region, "anchor": anchor, **ov, "reader_attn": round(r_attn, 4)})
            print(f"  [{label} {region:<12}] ov_rank={ov['rank']:<3} ov_pref={ov['pref']:.4f} "
                  f"cos={ov['cos']:+.3f} reader_attn->anchor={r_attn:.3f}", flush=True)

        per_model[label] = {
            "rows": rows,
            "median_rank": int(statistics.median([r["rank"] for r in rows])),
            "mean_pref": round(statistics.mean([r["pref"] for r in rows]), 4),
            "mean_reader_attn": round(statistics.mean([r["reader_attn"] for r in rows]), 4),
        }
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    out = {"reader": list(READER), "thresholds": {
        "ov_rank_top": OV_RANK_TOP, "ov_pref_rel_tol": OV_PREF_REL_TOL,
        "qk_base_min": QK_BASE_MIN, "qk_it_max": QK_IT_MAX},
        "base": per_model["base"], "it": per_model["it"],
        "decision": verdict(per_model["base"], per_model["it"])}
    Path("out").mkdir(exist_ok=True)
    Path("out/rlhf_ovqk_2b.json").write_text(json.dumps(out, indent=2))
    print("\n[verdict]", out["decision"]["verdict"])
    print(f"[done] wrote out/rlhf_ovqk_2b.json")


# --------------------------------------------------------------------------- selftest
def selftest():
    """Model-free: plant a known copy head and a random head in toy weights; assert the
    OV metric discriminates them, and that verdict() fires correctly on synthetic summaries."""
    torch.manual_seed(0)
    d_vocab, d_model, d_head, n_layers, n_heads = 50, 16, 16, 1, 2
    W_E = torch.nn.functional.normalize(torch.randn(d_vocab, d_model), dim=1)
    W_U = W_E.t().contiguous()                      # tied unembed -> token v scores by e . W_E[v]
    ln_final = lambda x: x                          # identity LN for the toy
    I = torch.eye(d_model)
    # head 0 = perfect copy: W_OV = I  =>  ov(e) = e  => peaks at the anchor token.
    # head 1 = random write: should NOT rank the anchor at the top.
    W_V = torch.stack([torch.stack([I, I])])                       # [1, 2, d_model, d_head]
    W_O = torch.stack([torch.stack([I, torch.randn(d_head, d_model)])])
    aid = 7

    copy = ov_copy(W_E, W_V, W_O, W_U, ln_final, 0, 0, 1, aid)
    rand = ov_copy(W_E, W_V, W_O, W_U, ln_final, 0, 1, 1, aid)
    assert copy["rank"] == 0, f"copy head should rank anchor top, got {copy['rank']}"
    assert copy["pref"] > 0.05, f"copy head pref too low: {copy['pref']}"
    assert copy["cos"] > 0.99, f"copy head should align with anchor unembed, got {copy['cos']}"
    assert rand["rank"] > copy["rank"], "random head should rank anchor worse than copy head"
    print(f"[selftest] OV metric OK  copy={copy}  rand={rand}")

    # verdict logic: OV unchanged + QK collapses -> GATING
    base = {"median_rank": 0, "mean_pref": 0.22, "mean_reader_attn": 0.58}
    it_gate = {"median_rank": 0, "mean_pref": 0.23, "mean_reader_attn": 0.016}
    it_del = {"median_rank": 40, "mean_pref": 0.02, "mean_reader_attn": 0.016}
    vg, vd = verdict(base, it_gate), verdict(base, it_del)
    assert vg["ov_preserved"] and vg["qk_gated"] and vg["verdict"].startswith("GATING"), vg
    assert (not vd["ov_preserved"]) and vd["verdict"].startswith("DELETION"), vd
    print(f"[selftest] verdict(gating) -> {vg['verdict']}")
    print(f"[selftest] verdict(deletion) -> {vd['verdict']}")
    print("[selftest] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="model-free logic check")
    a = ap.parse_args()
    selftest() if a.selftest else run()
