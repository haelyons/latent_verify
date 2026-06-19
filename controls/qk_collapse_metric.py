"""Measurement control: weight-only per-head base->it magnitude read for a list of attention heads.

This control is a pure INSTRUMENT. It reports numbers and a neutral threshold verdict on each
number; it does not reference any hypothesis, claim, or confound. It reuses the conventions of
controls/ov_qk_generality_probe.py verbatim:
  - model loading: HookedTransformer.from_pretrained_no_processing(name, dtype=bf16, device)
                   for ("base", --name-base) and ("it", --name-it), one model at a time
  - GQA map:       n_v = W_V.shape[1]; grp = nH // n_v if n_v < nH else 1;
                   value-head vH = H // grp, key-head kH = H // grp  (query-head -> kv-head)
  - QK read:       qk_fro = || W_Q[L,H] @ W_K[L,kH].T ||_F            (reused verbatim)
  - OV read:       ov_read -> ow_norm / preln_logit / W_OV_fro        (reused verbatim)
  - rel + verdict: _rel(it, base) = (it - base)/max(|base|,1e-9); CHANGED if |rel|>REL_TOL

For each head in --heads and each of three WEIGHT-ONLY per-head magnitudes, report base, it,
rel_change, and a neutral verdict:

  W_QK_fro = || W_Q[L,H] @ W_K[L,kH].T ||_F   (the QK bilinear magnitude; reuse qk_fro)
  W_OV_fro = || W_V[L,vH] @ W_O[L,H] ||_F      (the OV magnitude; reuse ov_read's W_OV_fro)
  ow_norm  = || e_t @ W_OV ||_2 averaged over a small fixed probe-token sample (reuse ov_read)

NEUTRAL decision: for EACH measured rel_change, |rel_change| > REL_TOL -> "CHANGED" else
"UNCHANGED". Verdicts are reported independently per head per metric, in terms of the measured
numbers only. No rolled-up score.

Usage:
  python controls/qk_collapse_metric.py --selftest    # model-free; synthetic weights, analytic asserts
  python controls/qk_collapse_metric.py               # gemma-2-2b base & -it -> out/qk_collapse_2b.json
  python controls/qk_collapse_metric.py --heads "18,5;18,6" --tag 2b
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

REL_TOL = 0.15                                      # neutral threshold on |rel_change|
DEFAULT_HEADS = [(18, 5), (18, 6), (12, 0)]         # a few sensible heads (L18.H5 reader + others)

# Small fixed probe-token sample for the averaged ow_norm. Single common tokens (space-prefixed,
# first-token convention from ov_qk_generality_probe). Fixed and claim-free: just a few embeddings
# whose write-magnitude through W_OV we average. Kept free of em-dashes.
PROBE_WORDS = ["the", "city", "of", "is", "and", "to", "in", "a"]


def _rel(x_it, x_base):
    """Base->it relative change. Undefined-safe denominator (matches ov_qk_generality_probe._rel)."""
    return (x_it - x_base) / max(abs(x_base), 1e-9)


def _verdict(rel):
    """Neutral, claim-free verdict purely on the measured number."""
    return "CHANGED" if abs(rel) > REL_TOL else "UNCHANGED"


# --------------------------------------------------------------------------- weight-only reads
def ov_read(W_E, W_V, W_O, W_U, L, H, grp, aid):
    """WEIGHT-ONLY OV magnitudes for token `aid`, head (L,H). No forward pass.

    Reused verbatim from controls/ov_qk_generality_probe.ov_read.

    ow_norm     = || e_anchor @ W_OV ||_2
    preln_logit = (e_anchor @ W_OV) @ W_U[:, aid]   (anchor logit push BEFORE ln_final)
    W_OV_fro    = || W_OV ||_F                       (head-level; independent of the anchor)
    """
    vH = H // grp if grp > 1 else H                 # GQA: query-head -> value-head
    W_OV = (W_V[L, vH] @ W_O[L, H]).float()         # [d_model, d_model]
    e = W_E[aid].float()                            # [d_model] anchor token embedding
    ov = e @ W_OV                                   # [d_model] what the head writes for this token
    u = W_U[:, aid].float()
    return {
        "ow_norm": round(float(ov.norm()), 5),
        "preln_logit": round(float(ov @ u), 5),
        "W_OV_fro": round(float(W_OV.norm()), 5),
    }


def qk_fro(W_Q, W_K, L, H, grp):
    """WEIGHT-ONLY QK bilinear magnitude || W_Q[L,H] @ W_K[L,kH].T ||_F. No forward pass.

    Reused verbatim from controls/ov_qk_generality_probe.qk_fro.

    W_QK = W_Q[L,H] @ W_K[L,kH].T is the [d_model, d_model] bilinear form whose entry (q,k) is
    the contribution of residual dims (q at query side, k at key side) to the attention score.
    Under GQA the key head kH = H // grp, mirroring the value-head map used on the OV side.
    """
    kH = H // grp if grp > 1 else H                 # GQA: query-head -> key-head
    W_QK = (W_Q[L, H] @ W_K[L, kH].t()).float()     # [d_model, d_model]
    return round(float(W_QK.norm()), 5)


def head_metrics(model, L, H, grp, probe_ids):
    """Three WEIGHT-ONLY per-head magnitudes for head (L,H) of one model.

    W_QK_fro = qk_fro(...)                      (QK bilinear magnitude)
    W_OV_fro = ov_read(...)["W_OV_fro"]         (OV magnitude, anchor-independent)
    ow_norm  = mean over probe_ids of ov_read(...)["ow_norm"]   (averaged write-vector magnitude)
    """
    wqk = qk_fro(model.W_Q, model.W_K, L, H, grp)
    per_tok = [ov_read(model.W_E, model.W_V, model.W_O, model.W_U, L, H, grp, aid)
               for aid in probe_ids]
    w_ov_fro = per_tok[0]["W_OV_fro"]               # head-level; identical across tokens
    ow_norm = round(statistics.mean(t["ow_norm"] for t in per_tok), 5)
    return {"W_QK_fro": wqk, "W_OV_fro": w_ov_fro, "ow_norm": ow_norm}


# --------------------------------------------------------------------------- real run
def run(heads, name_base, name_it, tag):
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    per_model = {}
    for label, name in [("base", name_base), ("it", name_it)]:
        print(f"[load] {name} on {device}", flush=True)
        model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
        model.eval()
        tok = model.tokenizer
        nH, n_v = model.cfg.n_heads, model.W_V.shape[1]
        grp = nH // n_v if n_v and n_v < nH else 1
        first = lambda s: tok.encode(s, add_special_tokens=False)[0]
        probe_ids = [first(" " + w) for w in PROBE_WORDS]

        rows = {}
        for (L, H) in heads:
            m = head_metrics(model, L, H, grp, probe_ids)
            rows[f"{L},{H}"] = m
            print(f"  [{label}] L{L}.H{H}: W_QK_fro={m['W_QK_fro']:.4f} "
                  f"W_OV_fro={m['W_OV_fro']:.4f} ow_norm={m['ow_norm']:.4f}", flush=True)
        per_model[label] = rows
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    METRICS = ["W_QK_fro", "W_OV_fro", "ow_norm"]
    measurements = {}
    for (L, H) in heads:
        key = f"{L},{H}"
        b, it = per_model["base"][key], per_model["it"][key]
        measurements[key] = {}
        for mname in METRICS:
            rel = round(_rel(it[mname], b[mname]), 4)
            measurements[key][mname] = {
                "base": b[mname], "it": it[mname],
                "rel_change": rel, "verdict": _verdict(rel),
            }

    out = {
        "tag": tag,
        "name_base": name_base,
        "name_it": name_it,
        "heads": [list(h) for h in heads],
        "metrics": METRICS,
        "probe_words": PROBE_WORDS,
        "rel_tol": REL_TOL,
        "decision_rule": "for each head, each metric rel_change: |rel_change| > REL_TOL -> CHANGED else UNCHANGED",
        "base_rows": per_model["base"],
        "it_rows": per_model["it"],
        "measurements": measurements,
    }
    Path("out").mkdir(exist_ok=True)
    Path(f"out/qk_collapse_{tag}.json").write_text(json.dumps(out, indent=2))

    print("\n[summary] per-head per-metric base->it rel_change (verdict):", flush=True)
    for (L, H) in heads:
        key = f"{L},{H}"
        parts = []
        for mname in METRICS:
            v = measurements[key][mname]
            parts.append(f"{mname}: {v['base']:.4f}->{v['it']:.4f} rel={v['rel_change']:+.4f} {v['verdict']}")
        print(f"  L{L}.H{H}: " + " | ".join(parts), flush=True)
    print(f"[done] wrote out/qk_collapse_{tag}.json", flush=True)


# --------------------------------------------------------------------------- selftest
def selftest():
    """Model-free. Synthetic weight matrices with KNOWN norms; assert the computed magnitudes
    match analytic values, that the GQA head map (H -> H//grp) is used on BOTH QK and OV, and
    that the neutral CHANGED/UNCHANGED verdict thresholds at REL_TOL."""
    torch.manual_seed(0)
    d_vocab, d_model = 64, 16
    W_E = torch.nn.functional.normalize(torch.randn(d_vocab, d_model), dim=1)
    W_U = W_E.t().contiguous()
    I = torch.eye(d_model)
    aid = 7

    # OV: head 0 = copy (c=1), head 1 = attenuated copy (c=1/3). W_OV = c*I => analytic norms:
    #   ow_norm  = |c| * ||e|| = |c|     (e is unit-norm)
    #   W_OV_fro = ||c*I||_F = |c| * sqrt(d_model)
    for c, hidx in [(1.0, 0), (1.0 / 3.0, 1)]:
        W_V = torch.stack([torch.stack([I, I])])                  # [1, 2, d, d]
        W_O = torch.stack([torch.stack([I, I / 3.0])])
        m = ov_read(W_E, W_V, W_O, W_U, 0, hidx, 1, aid)
        assert abs(m["ow_norm"] - abs(c)) < 1e-3, f"ow_norm {m['ow_norm']} != {abs(c)}"
        assert abs(m["W_OV_fro"] - abs(c) * (d_model ** 0.5)) < 1e-3, f"W_OV_fro {m['W_OV_fro']}"
    print("[selftest] OV magnitudes match analytic values (copy c=1 and attenuated c=1/3)")

    # QK: W_Q = a*I, W_K = b*I => W_QK = a*b*I => ||W_QK||_F = |a*b| * sqrt(d_model).
    a_q, b_k = 2.0, 0.5
    W_Q = torch.stack([torch.stack([a_q * I])])                   # [1, 1, d, d]
    W_K = torch.stack([torch.stack([b_k * I])])
    fro = qk_fro(W_Q, W_K, 0, 0, 1)
    exp_qk = abs(a_q * b_k) * (d_model ** 0.5)
    assert abs(fro - exp_qk) < 1e-3, f"W_QK_fro {fro} != {exp_qk}"
    print(f"[selftest] QK magnitude matches analytic value: ||W_QK||_F={fro} (expected {exp_qk:.5f})")

    # GQA map on BOTH sides: grp=2, query head 3 maps to kv head 3//2 = 1. Build distinct kv heads
    # and assert each read uses head H//grp on the V and K stacks.
    kv0, kv1 = 1.0 * I, 4.0 * I
    W_Vg = torch.stack([torch.stack([kv0, kv1])])                # 2 kv value heads
    W_Og = torch.stack([torch.stack([I, I, I, I])])             # 4 query heads (O is per-query-head)
    mg = ov_read(W_E, W_Vg, W_Og, W_U, 0, 3, 2, aid)           # H=3, grp=2 -> vH=1 (kv1=4I)
    assert abs(mg["W_OV_fro"] - 4.0 * (d_model ** 0.5)) < 1e-3, f"GQA vH map wrong: {mg['W_OV_fro']}"
    W_Kg = torch.stack([torch.stack([1.0 * I, 3.0 * I])])      # 2 kv key heads
    W_Qg = torch.stack([torch.stack([I, I, I, I])])           # 4 query heads
    frg = qk_fro(W_Qg, W_Kg, 0, 3, 2)                          # H=3, grp=2 -> kH=1 (3I)
    assert abs(frg - 3.0 * (d_model ** 0.5)) < 1e-3, f"GQA kH map wrong: {frg}"
    print("[selftest] GQA head map (query->kv via H//grp) correct on both OV and QK reads")

    # head_metrics on a tiny stub model: averages ow_norm over the probe sample, returns all three.
    class _Stub:
        pass
    stub = _Stub()
    stub.W_E, stub.W_U = W_E, W_U
    stub.W_V = torch.stack([torch.stack([I, I])])               # 2 kv value heads
    stub.W_O = torch.stack([torch.stack([I, I / 3.0])])
    stub.W_Q = torch.stack([torch.stack([a_q * I, a_q * I])])
    stub.W_K = torch.stack([torch.stack([b_k * I, b_k * I])])
    probe_ids = [3, 11, 27]                                     # arbitrary fixed token ids
    hm = head_metrics(stub, 0, 0, 1, probe_ids)                # copy head: W_OV = I
    exp_ow = round(statistics.mean(round(float((W_E[i].float() @ I).norm()), 5) for i in probe_ids), 5)
    assert abs(hm["ow_norm"] - exp_ow) < 1e-3, f"head_metrics ow_norm {hm['ow_norm']} != {exp_ow}"
    assert abs(hm["W_OV_fro"] - (d_model ** 0.5)) < 1e-3, f"head_metrics W_OV_fro {hm['W_OV_fro']}"
    assert abs(hm["W_QK_fro"] - exp_qk) < 1e-3, f"head_metrics W_QK_fro {hm['W_QK_fro']}"
    print(f"[selftest] head_metrics returns all three weight-only magnitudes: {hm}")

    # Neutral verdict thresholding at REL_TOL (claim-free) on _rel + _verdict.
    assert _verdict(0.0) == "UNCHANGED" and _verdict(0.10) == "UNCHANGED" and _verdict(REL_TOL) == "UNCHANGED"
    assert _verdict(0.16) == "CHANGED" and _verdict(-0.50) == "CHANGED"
    assert abs(_rel(0.5, 1.0) - (-0.5)) < 1e-9 and abs(_rel(1.3, 1.0) - 0.3) < 1e-9
    assert abs(_rel(1.0, 0.0)) <= 1.0 / 1e-9 and _verdict(_rel(1.0, 0.0)) == "CHANGED"
    print(f"[selftest] neutral verdict thresholds at REL_TOL={REL_TOL} (CHANGED if |rel|>REL_TOL)")

    # head-parser round trip
    assert _parse_heads("18,5;18,6;12,0") == [(18, 5), (18, 6), (12, 0)]
    assert _parse_heads(" 0 , 1 ") == [(0, 1)]
    print("[selftest] PASS")


def _parse_heads(s):
    """Parse '--heads L,H;L,H;...' into a list of (L,H) int tuples."""
    out = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.replace(" ", "").split(",")
        if len(parts) != 2:
            raise SystemExit("--heads takes 'L,H;L,H;...' (each entry two ints)")
        out.append((int(parts[0]), int(parts[1])))
    if not out:
        raise SystemExit("--heads parsed to an empty list")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="model-free analytic-norm + verdict check")
    ap.add_argument("--heads", default=";".join(f"{L},{H}" for L, H in DEFAULT_HEADS),
                    help="heads 'L,H;L,H;...' (default a few heads incl. L18.H5)")
    ap.add_argument("--name-base", default="google/gemma-2-2b")
    ap.add_argument("--name-it", default="google/gemma-2-2b-it")
    ap.add_argument("--tag", default="2b")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(_parse_heads(a.heads), a.name_base, a.name_it, a.tag)
