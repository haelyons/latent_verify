"""C1 triage control -- does RLHF preserve the OV copy in MAGNITUDE, or only in DIRECTION?

Standalone probe derived from job_rlhf_ovqk.py (which is left untouched / pinned). Authored to
settle the two surviving cruxes the latent_skeptic triage raised against the I2 "OV preserved"
claim (FRAMING_NOTES sec-8 refinement):

  crux 1 (construct validity): ov_pref / ov_rank are SCALE-INVARIANT. softmax-pref and rank measure
      the DIRECTION of the OV write, not its MAGNITUDE. "the copy survives / is preserved" is
      functionally a magnitude statement -- a head can keep writing toward the anchor (rank 0, pref
      ~1) while writing it far more weakly. rank/pref/cos cannot see that.
  crux 2 (ceiling/floor): pref ~0.9997 and rank 0 are pinned at ceiling/floor; identical base=-it
      values on saturated metrics are the EXPECTED signature of saturation, not proof the OV matrix
      is unchanged.

Fix: add metrics WITH dynamic range, weight-only (no forward pass), base vs -it on the same stack:
  ow_norm     = || e_anchor @ W_OV ||_2          the anchor write-vector magnitude
  preln_logit = (e_anchor @ W_OV) @ W_U[:,anchor] the anchor logit contribution BEFORE ln_final
                                                  (the functional push, in raw nats)
  W_OV_fro / W_OV_op = Frobenius / spectral norm of W_OV = W_V@W_O   (head-level, anchor-independent)
  cos         = direction (already in job_rlhf_ovqk; carried for completeness)

Pre-registered thresholds (fixed before any model run):
  MAG_REL_TOL = 0.15
  DIRECTION_PRESERVED := median it rank <= 5  AND  mean |cos_it - cos_base| <= 0.02
  MAGNITUDE_PRESERVED := |rel| <= MAG_REL_TOL for mean ow_norm AND mean preln_logit AND W_OV_fro
      where rel(x) = (x_it - x_base) / |x_base|
  -> DIRECTION & MAGNITUDE preserved => "OV PRESERVED (functional): I2 'OV preserved' licensed as a
                                          functional, not merely directional, claim."
  -> DIRECTION preserved, MAGNITUDE not => "OV ATTENUATED: direction intact but write-magnitude drops;
                                          I2 'preserved' OVERSTATED -- part of the -it disengagement is
                                          OV attenuation invisible to rank/pref/cos."
  -> DIRECTION not preserved            => "OV ALTERED: write direction changed in -it weights too."

Usage:
  python ov_norm_probe.py --selftest    # model-free; proves the magnitude metric catches an
                                         # attenuation that rank/pref/cos miss, and verdict() fires
  python ov_norm_probe.py               # gemma-2-2b base & -it (needs transformer_lens + gated -it);
                                         # weight-only, CPU-ok -> out/ov_norm_probe_2b.json
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

READER = (18, 5)                                   # sec-3.10 universal reader head (2b); same as I2
STEM = "The capital of {r} is the city of"
SALIENCE = "{w} is the most famous city in {r}. "
PAIRS = [                                           # (region, capital=C, anchor=W) -- same as I2
    ("Australia",   "Canberra",    "Sydney"),
    ("Texas",       "Austin",      "Houston"),
    ("Canada",      "Ottawa",      "Toronto"),
    ("Switzerland", "Bern",        "Zurich"),
    ("Morocco",     "Rabat",       "Casablanca"),
]

MAG_REL_TOL = 0.15
DIR_RANK_TOP = 5
DIR_COS_TOL = 0.02


def ov_metrics(W_E, W_V, W_O, W_U, ln_final, L, H, grp, aid):
    """WEIGHT-ONLY OV metrics for token `aid`, head (L,H). Direction (saturated) + magnitude (ranged).

    rank/pref/cos  : DIRECTION of the write (scale-invariant -- the saturated metrics).
    ow_norm        : || e @ W_OV ||_2 -- magnitude of the anchor write vector.
    preln_logit    : (e @ W_OV) @ W_U[:,aid] -- anchor logit push BEFORE ln_final (functional nats).
    W_OV_fro/op    : Frobenius / spectral norm of W_OV (head-level, independent of the anchor).
    """
    vH = H // grp if grp > 1 else H                 # GQA: query-head -> value-head
    W_OV = (W_V[L, vH] @ W_O[L, H]).float()         # [d_model, d_model]
    e = W_E[aid].float()                            # [d_model] anchor token embedding
    ov = e @ W_OV                                   # [d_model] what the head writes for this token
    normed = ln_final(ov.unsqueeze(0).unsqueeze(0).to(W_E.dtype))[0, 0].float()
    logits = normed @ W_U.float()                   # [d_vocab]
    rank = int((logits > logits[aid]).sum().item())
    pref = float(torch.softmax(logits, -1)[aid])
    u = W_U[:, aid].float()
    cos = float(torch.nn.functional.cosine_similarity(ov, u, dim=0))
    return {
        "rank": rank,
        "pref": round(pref, 6),
        "cos": round(cos, 4),
        "ow_norm": round(float(ov.norm()), 5),
        "preln_logit": round(float(ov @ u), 5),
        "W_OV_fro": round(float(W_OV.norm()), 5),
        "W_OV_op": round(float(torch.linalg.matrix_norm(W_OV, ord=2)), 5),
    }


def _rel(x_it, x_base):
    return (x_it - x_base) / max(abs(x_base), 1e-9)


def verdict(base, it):
    """Pure decision from the two per-model summaries (also exercised by --selftest)."""
    direction_preserved = (it["median_rank"] <= DIR_RANK_TOP and
                           abs(it["mean_cos"] - base["mean_cos"]) <= DIR_COS_TOL)
    rels = {
        "ow_norm": _rel(it["mean_ow_norm"], base["mean_ow_norm"]),
        "preln_logit": _rel(it["mean_preln_logit"], base["mean_preln_logit"]),
        "W_OV_fro": _rel(it["W_OV_fro"], base["W_OV_fro"]),
    }
    magnitude_preserved = all(abs(v) <= MAG_REL_TOL for v in rels.values())
    if not direction_preserved:
        v = "OV ALTERED: write direction changed in -it weights too"
    elif magnitude_preserved:
        v = ("OV PRESERVED (functional): direction AND write-magnitude unchanged base->it; "
             "I2 'OV preserved' is licensed as a functional, not merely directional, claim")
    else:
        v = ("OV ATTENUATED: direction intact but write-magnitude drops; I2 'OV preserved' is "
             "OVERSTATED -- part of the -it disengagement is OV attenuation invisible to rank/pref/cos")
    return {"direction_preserved": direction_preserved, "magnitude_preserved": magnitude_preserved,
            "rel_change": {k: round(v, 4) for k, v in rels.items()}, "verdict": v}


def _summarize(rows):
    return {
        "rows": rows,
        "median_rank": int(statistics.median([r["rank"] for r in rows])),
        "mean_pref": round(statistics.mean([r["pref"] for r in rows]), 6),
        "mean_cos": round(statistics.mean([r["cos"] for r in rows]), 4),
        "mean_ow_norm": round(statistics.mean([r["ow_norm"] for r in rows]), 5),
        "mean_preln_logit": round(statistics.mean([r["preln_logit"] for r in rows]), 5),
        "W_OV_fro": rows[0]["W_OV_fro"],   # head-level, identical across pairs
        "W_OV_op": rows[0]["W_OV_op"],
    }


# --------------------------------------------------------------------------- real run
def run():
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        rows = []
        for region, cap, anchor in PAIRS:
            aid = first(" " + anchor)
            m = ov_metrics(model.W_E, model.W_V, model.W_O, model.W_U, model.ln_final, L, H, grp, aid)
            rows.append({"region": region, "anchor": anchor, **m})
            print(f"  [{label} {region:<12}] rank={m['rank']:<3} pref={m['pref']:.4f} cos={m['cos']:+.3f} "
                  f"ow_norm={m['ow_norm']:.3f} preln_logit={m['preln_logit']:+.3f} "
                  f"W_OV_fro={m['W_OV_fro']:.3f}", flush=True)
        per_model[label] = _summarize(rows)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    out = {"reader": list(READER),
           "thresholds": {"mag_rel_tol": MAG_REL_TOL, "dir_rank_top": DIR_RANK_TOP, "dir_cos_tol": DIR_COS_TOL},
           "base": per_model["base"], "it": per_model["it"],
           "decision": verdict(per_model["base"], per_model["it"])}
    Path("out").mkdir(exist_ok=True)
    Path("out/ov_norm_probe_2b.json").write_text(json.dumps(out, indent=2))
    print("\n[verdict]", out["decision"]["verdict"])
    print("[rel_change]", out["decision"]["rel_change"])
    print("[done] wrote out/ov_norm_probe_2b.json")


# --------------------------------------------------------------------------- selftest
def selftest():
    """Model-free. The point of this probe: a magnitude attenuation that the saturated direction
    metrics (rank/pref/cos) CANNOT see. Plant (a) a perfect copy head, (b) the SAME copy head scaled
    down 3x, (c) a random head. Assert the scaled head matches the copy head on rank/pref/cos but
    its ow_norm / preln_logit drop ~3x -- i.e. the new metrics add the dynamic range the cruxes asked
    for. Then assert verdict() flags ATTENUATED on a saturated-but-shrunk -it summary."""
    torch.manual_seed(0)
    d_vocab, d_model, d_head = 64, 16, 16
    W_E = torch.nn.functional.normalize(torch.randn(d_vocab, d_model), dim=1)
    W_U = W_E.t().contiguous()                      # tied unembed
    # Real gemma ln_final is RMSNorm -> it STRIPS magnitude before the unembed, which is precisely
    # why rank/pref/cos saturate (the crux). The toy must mirror that, else pref leaks the scale.
    ln_final = lambda x: x / x.pow(2).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    I = torch.eye(d_model)
    aid = 7
    # 3 heads stacked: 0 = copy (W_OV=I), 1 = attenuated copy (W_OV=I/3), 2 = random write.
    W_V = torch.stack([torch.stack([I, I, I])])                                  # [1,3,d,d]
    W_O = torch.stack([torch.stack([I, I / 3.0, torch.randn(d_head, d_model)])])

    copy = ov_metrics(W_E, W_V, W_O, W_U, ln_final, 0, 0, 1, aid)
    att = ov_metrics(W_E, W_V, W_O, W_U, ln_final, 0, 1, 1, aid)
    rand = ov_metrics(W_E, W_V, W_O, W_U, ln_final, 0, 2, 1, aid)

    # direction metrics are BLIND to the 3x attenuation (the crux):
    assert copy["rank"] == 0 and att["rank"] == 0, f"both copy heads should rank anchor top: {copy['rank']}, {att['rank']}"
    assert abs(copy["pref"] - att["pref"]) < 1e-3, f"pref should be scale-invariant: {copy['pref']} vs {att['pref']}"
    assert abs(copy["cos"] - att["cos"]) < 1e-3, f"cos should be scale-invariant: {copy['cos']} vs {att['cos']}"
    # magnitude metrics DO see it (~3x), which is the whole point:
    assert abs(att["ow_norm"] - copy["ow_norm"] / 3.0) < 1e-3, f"ow_norm should drop 3x: {copy['ow_norm']} -> {att['ow_norm']}"
    assert abs(att["preln_logit"] - copy["preln_logit"] / 3.0) < 1e-3, f"preln_logit should drop 3x: {copy['preln_logit']} -> {att['preln_logit']}"
    assert rand["rank"] > 0, "random head should not rank anchor top"
    print(f"[selftest] copy={copy}")
    print(f"[selftest] attenuated={att}  (rank/pref/cos identical to copy; ow_norm/preln_logit ~1/3)")
    print(f"[selftest] random={rand}")

    # verdict logic: saturated direction unchanged + magnitude shrunk -> ATTENUATED (the masked case)
    base = {"median_rank": 0, "mean_cos": 0.99, "mean_ow_norm": 1.00, "mean_preln_logit": 0.90, "W_OV_fro": 4.0, "W_OV_op": 1.0}
    it_pres = {"median_rank": 0, "mean_cos": 0.99, "mean_ow_norm": 0.97, "mean_preln_logit": 0.88, "W_OV_fro": 3.9, "W_OV_op": 1.0}
    it_att = {"median_rank": 0, "mean_cos": 0.99, "mean_ow_norm": 0.50, "mean_preln_logit": 0.45, "W_OV_fro": 2.0, "W_OV_op": 0.5}
    vp, va = verdict(base, it_pres), verdict(base, it_att)
    assert vp["magnitude_preserved"] and vp["verdict"].startswith("OV PRESERVED"), vp
    assert (not va["magnitude_preserved"]) and va["verdict"].startswith("OV ATTENUATED"), va
    print(f"[selftest] verdict(preserved) -> {vp['verdict']}")
    print(f"[selftest] verdict(attenuated) -> {va['verdict']}  rel={va['rel_change']}")
    print("[selftest] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="model-free logic + metric check")
    a = ap.parse_args()
    selftest() if a.selftest else run()
