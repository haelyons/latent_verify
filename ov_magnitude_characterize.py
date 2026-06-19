"""NEXT-3: characterize the 27b RLHF OV-magnitude increase -- what do those copy heads write MORE of?
(RESEARCH_QUESTIONS Part 4 NEXT-3.) Weights-only, forward-free.

CONTEXT. The 27b QK-collapse control (results_27b_qk/, qk_collapse_metric.py) found W_QK_fro UNCHANGED
base->it for all 10 copy heads, but W_OV_fro + ow_norm CHANGED for some: L11.{2,4,7,21} (GQA-shared kv
group, +0.34), L17.H4 (+0.52); L23.H24 DECREASED (-0.25); L16.3/L19.{2,5,7} unchanged. ow_norm tracked
W_OV_fro almost exactly, hinting at pure magnitude scaling rather than a new write. This control settles
"more of WHAT": is the changed OV a SCALED copy of the base write (same direction, just bigger -> RLHF
amplifies these heads' existing function) or a REDIRECTED write (new content)? And does the head still
copy?

Per head, base and -it, all WEIGHT-ONLY (W_V, W_O, W_E, W_U; no forward pass):
  W_OV = W_V[L,vH] @ W_O[L,H]                          [d_model, d_model]  (GQA: vH = H // grp)
  per probe token t:  ov_t = e_t @ W_OV (write vector); logits_t = ov_t @ W_U (vocab push);
                      copy_hit = (argmax_v logits_t == t); copy_logit = logits_t[t]; top5_t = top5(logits_t)
Cross-model (base vs -it), the load-bearing decomposition:
  alpha       = <W_OV_it, W_OV_base>_F / ||W_OV_base||_F^2     (best scalar s.t. W_OV_it ~ alpha W_OV_base)
  resid_frac  = ||W_OV_it - alpha W_OV_base||_F / ||W_OV_it||_F   (0 -> pure scalar scaling; ~1 -> new matrix)
  dir_cos     = cos(flat W_OV_base, flat W_OV_it)              (matrix-direction alignment)
  write_cos   = mean_t cos(ov_base_t, ov_it_t)                 (per-token write-direction alignment)
  top5_overlap= mean_t Jaccard(top5_base_t, top5_it_t)         (does it promote the same vocab?)
  copy_hit_rate / mean copy_logit, base & it                    (is it a copy head, and does that change?)

NEUTRAL decision (claim-free, like qk_collapse_metric): per head,
  AMPLIFY_SAME  if dir_cos >= DIR_TOL and resid_frac <= RESID_TOL  (it OV ~ scaled base OV: same write, bigger)
  REDIRECT      otherwise                                          (the OV write direction moved)
plus a copy tag (COPY if copy_hit_rate >= COPY_TOL in base). Reports numbers + verdict only, no claim.

CAVEAT (stated, not resolved here): gemma-2-it is SFT+RLHF+merge, so a base->it weight diff conflates the
three (the qk_collapse_metric / B5 caveat). This says WHAT changed in the OV weights, not which stage.

  python ov_magnitude_characterize.py --selftest        # model-free, synthetic W_OV, analytic asserts
  python ov_magnitude_characterize.py --device cpu --name-base google/gemma-2-27b --name-it google/gemma-2-27b-it --tag 27b
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

DIR_TOL = 0.95        # dir_cos >= this AND resid_frac <= RESID_TOL -> the OV write is the same direction
RESID_TOL = 0.20      # scalar-fit residual fraction below this -> pure magnitude scaling
COPY_TOL = 0.5        # copy_hit_rate >= this (in base) -> a copy head
PROBE_WORDS = ["the", "city", "of", "is", "and", "to", "in", "a", "water", "king"]
# the 27b copy basket from results_27b_qk (CHANGED + UNCHANGED controls)
DEFAULT_HEADS = [(11, 2), (11, 4), (11, 7), (11, 21), (16, 3), (17, 4), (19, 2), (19, 5), (19, 7), (23, 24)]


# --------------------------------------------------------------------------- pure weight math
def w_ov(W_V, W_O, L, H, grp):
    vH = H // grp if grp > 1 else H
    return (W_V[L, vH] @ W_O[L, H]).float()           # [d_model, d_model]


def scalar_decomp(ov_base, ov_it):
    """alpha (best scalar), residual fraction, and flattened-matrix direction cosine. Pure."""
    b, t = ov_base.flatten(), ov_it.flatten()
    bb = float(b @ b)
    alpha = float(t @ b) / bb if bb > 0 else 0.0
    resid = (t - alpha * b).norm()
    resid_frac = float(resid / (t.norm() + 1e-9))
    dir_cos = float(torch.nn.functional.cosine_similarity(b, t, dim=0))
    return alpha, resid_frac, dir_cos


def token_writes(ov, W_E, W_U, probe_ids):
    """Per probe token: write-vector, copy-hit (argmax vocab == token), copy-logit, top5 vocab. Pure.
    Upcasts to float32 so a bf16 model's W_U/W_E do not dtype-clash with the float W_OV write vector."""
    ov = ov.float()
    W_U = W_U.float()
    out = []
    for t in probe_ids:
        e = W_E[t].float()
        wv = e @ ov                                   # [d_model] write vector
        logits = wv @ W_U                             # [d_vocab] vocab push
        top5 = set(torch.topk(logits, 5).indices.tolist())
        out.append({"wv": wv, "copy_hit": int(int(torch.argmax(logits)) == t),
                    "copy_logit": float(logits[t]), "top5": top5})
    return out


def decide(dir_cos, resid_frac, copy_hit_rate_base, dir_tol=DIR_TOL, resid_tol=RESID_TOL, copy_tol=COPY_TOL):
    same = dir_cos >= dir_tol and resid_frac <= resid_tol
    tag = "AMPLIFY_SAME" if same else "REDIRECT"
    copy = "COPY" if copy_hit_rate_base >= copy_tol else "NONCOPY"
    return {"ov_change": tag, "head_type_base": copy,
            "note": ("it OV is the base OV scaled (same write direction, magnitude only)" if same
                     else "it OV write direction moved (not a pure rescale)")}


# --------------------------------------------------------------------------- real run
def _load(name, device):
    from transformer_lens import HookedTransformer
    print(f"[load] {name} on {device}", flush=True)
    m = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    m.eval()
    return m


def _per_model(model, heads, probe_ids):
    nH, n_v = model.cfg.n_heads, model.W_V.shape[1]
    grp = nH // n_v if n_v and n_v < nH else 1
    rows = {}
    for (L, H) in heads:
        ov = w_ov(model.W_V, model.W_O, L, H, grp)
        tw = token_writes(ov, model.W_E, model.W_U, probe_ids)
        rows[(L, H)] = {"ov": ov, "fro": float(ov.norm()),
                        "ow_norm": statistics.mean(float(t["wv"].norm()) for t in tw),
                        "copy_hit_rate": statistics.mean(t["copy_hit"] for t in tw),
                        "copy_logit": statistics.mean(t["copy_logit"] for t in tw),
                        "writes": [t["wv"] for t in tw], "top5": [t["top5"] for t in tw]}
    return rows


def run(heads, name_base, name_it, tag, device):
    pb = _load(name_base, device)
    tok = pb.tokenizer
    first = lambda s: tok.encode(s, add_special_tokens=False)[0]
    probe_ids = [first(" " + w) for w in PROBE_WORDS]
    base = _per_model(pb, heads, probe_ids)
    del pb
    if device == "cuda":
        torch.cuda.empty_cache()
    pit = _load(name_it, device)
    it = _per_model(pit, heads, probe_ids)
    del pit
    if device == "cuda":
        torch.cuda.empty_cache()

    measurements = {}
    for (L, H) in heads:
        b, i = base[(L, H)], it[(L, H)]
        alpha, resid_frac, dir_cos = scalar_decomp(b["ov"], i["ov"])
        write_cos = statistics.mean(
            float(torch.nn.functional.cosine_similarity(wb, wi, dim=0)) for wb, wi in zip(b["writes"], i["writes"]))
        overlap = statistics.mean(len(sb & si) / 5.0 for sb, si in zip(b["top5"], i["top5"]))
        dec = decide(dir_cos, resid_frac, b["copy_hit_rate"])
        measurements[f"{L},{H}"] = {
            "fro_base": round(b["fro"], 4), "fro_it": round(i["fro"], 4),
            "fro_rel": round((i["fro"] - b["fro"]) / max(abs(b["fro"]), 1e-9), 4),
            "alpha": round(alpha, 4), "resid_frac": round(resid_frac, 4), "dir_cos": round(dir_cos, 4),
            "write_cos": round(write_cos, 4), "top5_overlap": round(overlap, 4),
            "copy_hit_rate_base": round(b["copy_hit_rate"], 3), "copy_hit_rate_it": round(i["copy_hit_rate"], 3),
            "copy_logit_base": round(b["copy_logit"], 4), "copy_logit_it": round(i["copy_logit"], 4),
            **dec}
    out = {"tag": tag, "name_base": name_base, "name_it": name_it, "heads": [list(h) for h in heads],
           "probe_words": PROBE_WORDS, "dir_tol": DIR_TOL, "resid_tol": RESID_TOL, "copy_tol": COPY_TOL,
           "decision_rule": "AMPLIFY_SAME if dir_cos>=DIR_TOL and resid_frac<=RESID_TOL else REDIRECT",
           "measurements": measurements}
    Path("out").mkdir(exist_ok=True)
    Path(f"out/ov_magnitude_{tag}.json").write_text(json.dumps(out, indent=2))
    print("\n[summary] per-head OV change base->it:", flush=True)
    for (L, H) in heads:
        m = measurements[f"{L},{H}"]
        print(f"  L{L}.H{H}: fro_rel={m['fro_rel']:+.3f} alpha={m['alpha']:.3f} resid_frac={m['resid_frac']:.3f} "
              f"dir_cos={m['dir_cos']:.3f} write_cos={m['write_cos']:.3f} top5_ov={m['top5_overlap']:.2f} "
              f"copy_hit b/it={m['copy_hit_rate_base']:.2f}/{m['copy_hit_rate_it']:.2f} "
              f"copy_logit {m['copy_logit_base']:.2f}->{m['copy_logit_it']:.2f} [{m['ov_change']},{m['head_type_base']}]", flush=True)
    print(f"[done] wrote out/ov_magnitude_{tag}.json", flush=True)


# --------------------------------------------------------------------------- selftest (model-free)
def selftest():
    torch.manual_seed(0)
    d_vocab, d_model = 50, 12
    W_E = torch.nn.functional.normalize(torch.randn(d_vocab, d_model), dim=1)
    W_U = W_E.t().contiguous()                          # copy head: e_t @ I @ W_U peaks at t
    I = torch.eye(d_model)

    # PURE SCALING: it OV = 1.5 * base OV -> alpha=1.5, resid_frac~0, dir_cos~1
    ov_b, ov_i = I.clone(), 1.5 * I
    a, rf, dc = scalar_decomp(ov_b, ov_i)
    assert abs(a - 1.5) < 1e-4 and rf < 1e-4 and dc > 0.999, (a, rf, dc)
    print(f"[selftest] pure-scaling: alpha={a:.3f} resid_frac={rf:.4f} dir_cos={dc:.4f}")

    # REDIRECT: it OV is a different (orthogonal-ish) matrix -> high resid_frac, lower dir_cos
    g = torch.Generator().manual_seed(3)
    ov_i2 = torch.randn(d_model, d_model, generator=g)
    a2, rf2, dc2 = scalar_decomp(ov_b, ov_i2)
    assert rf2 > 0.5, (a2, rf2, dc2)
    print(f"[selftest] redirect: resid_frac={rf2:.3f} dir_cos={dc2:.3f}")

    # token_writes on a copy head (W_OV = I): every probe token's argmax vocab is itself -> copy_hit=1
    tw = token_writes(I, W_E, W_U, [3, 11, 27, 42])
    assert all(t["copy_hit"] == 1 for t in tw), [t["copy_hit"] for t in tw]
    print(f"[selftest] copy head: copy_hit_rate={statistics.mean(t['copy_hit'] for t in tw):.2f}")

    # bf16 regression guard: a bf16 model's W_OV/W_E/W_U must not dtype-clash (the 27b run's first crash)
    twb = token_writes(I.bfloat16(), W_E.bfloat16(), W_U.bfloat16(), [3, 11, 27, 42])
    assert all(t["copy_hit"] == 1 for t in twb), "bf16 inputs must upcast and still copy"
    print("[selftest] bf16 inputs upcast cleanly (no float/bf16 matmul clash)")

    # decide: pure-scaling copy head -> AMPLIFY_SAME, COPY ; redirect -> REDIRECT
    assert decide(0.999, 0.0, 1.0)["ov_change"] == "AMPLIFY_SAME"
    assert decide(0.999, 0.0, 1.0)["head_type_base"] == "COPY"
    assert decide(0.3, 0.8, 0.0)["ov_change"] == "REDIRECT"
    assert decide(0.999, 0.0, 0.1)["head_type_base"] == "NONCOPY"
    print("[selftest] decide: AMPLIFY_SAME/COPY and REDIRECT/NONCOPY fire")
    print("[selftest] PASS")


def _parse_heads(s):
    out = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if chunk:
            p = chunk.replace(" ", "").split(",")
            if len(p) != 2:
                raise SystemExit("--heads takes 'L,H;L,H;...'")
            out.append((int(p[0]), int(p[1])))
    if not out:
        raise SystemExit("--heads empty")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--heads", default=";".join(f"{L},{H}" for L, H in DEFAULT_HEADS))
    ap.add_argument("--name-base", default="google/gemma-2-27b")
    ap.add_argument("--name-it", default="google/gemma-2-27b-it")
    ap.add_argument("--tag", default="27b")
    ap.add_argument("--device", default="cpu", help="cpu (weights-only, fits 200GB-RAM box) or cuda (needs 80GB)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(_parse_heads(a.heads), a.name_base, a.name_it, a.tag, a.device)
