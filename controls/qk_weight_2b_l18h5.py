"""QK weight-vs-realized control for a small basket of gemma-2-2b heads (v3).

CONTEXT. A realized attention change base->it can have distinct causes; this control separates
them, head by head, on gemma-2-2b BASE vs -IT, with no aggregation across heads:

  1. WEIGHT magnitude (weights only). fro_base, fro_it, fro_rel of
     W_QK = W_Q[L,H] @ W_K[L, H//grp].T  (GQA grp = n_heads // n_kv_heads; float32 upcast).
  2. WEIGHT direction (weights only). Scalar decomposition of W_QK_it vs W_QK_base:
       alpha (best scalar fit), resid_frac (||it-alpha*base||/||it||), dir_cos (flat-matrix cosine).
     dir_cos ~ 1 and resid_frac ~ 0 -> same QK up to scale (no rotation); else re-oriented.
  3. REALIZED attention (forward). Head attention from the readout query (last token of the
     "...is the city of" stem) onto the anchor token, mean over 5 capital pairs: attn_base, attn_it.
  4. RESIDUAL-INPUT SWAPS (forward, causal). Run -it but patch blocks.{L}.hook_resid_pre (the input
     to layer L's attention) with the BASE residual (same prompt + tokenizer, so positions align),
     keep -it weights, and re-measure attention-to-anchor. Three position sets isolate where the
     change lives:
       query : only the readout position  -> recovery_query  (the residual feeding the query)
       key   : all positions except readout -> recovery_key   (the anchor/context key residuals)
       full  : all positions               -> recovery_full   (the whole layer input)
     recovery_X = (attn_swapX - attn_it) / (attn_base - attn_it). Since L's own weights are held,
     recovery_full ~ 1 means the collapse is ENTIRELY explained by the residual input to layer L
     (weights exonerated); recovery_full < FULL_TOL flags an L-local non-QK difference (e.g. LN).
     The query/key split says which side of the input carries it.

NEUTRAL DECISION (module-constant thresholds; reports numbers + categories only):
  weight_mag : CHANGED if |fro_rel|>=WQK_THR; UNCHANGED if <WQK_SMALL; else BORDERLINE.
  weight_dir : QK_SAME_DIR if dir_cos>=QK_DIR_TOL and resid_frac<=QK_RESID_TOL; else QK_ROTATED.
  pattern    : COLLAPSED if (attn_base-attn_it)>=ATTN_DROP; else STABLE.
  dominant_side : QUERY if recovery_query>=recovery_key else KEY (reported whenever COLLAPSED).
  cause (when COLLAPSED):
    weights intact (UNCHANGED + QK_SAME_DIR): INPUT_MEDIATED if recovery_full>=FULL_TOL else PARTLY_LOCAL.
    weights changed (CHANGED or QK_ROTATED) : WEIGHT_LOCAL if recovery_full<FULL_TOL else MIXED.

Run model-free selftest (no model load, CPU):
    python controls/qk_weight_2b_l18h5.py --selftest
Run the measurement (2b fits CPU):
    python controls/qk_weight_2b_l18h5.py --device cpu --name-base google/gemma-2-2b --name-it google/gemma-2-2b-it --tag 2b
"""

import argparse
import json
import math
from pathlib import Path

import torch

# Pre-registered thresholds (see module docstring).
WQK_THR = 0.10        # |fro_rel| >= this -> QK weight magnitude changed
WQK_SMALL = 0.05      # |fro_rel| <  this -> QK weight magnitude effectively unchanged
QK_DIR_TOL = 0.95     # dir_cos >= this AND resid_frac <= QK_RESID_TOL -> same QK direction
QK_RESID_TOL = 0.20   # scalar-fit residual fraction at/below this -> pure rescale, no rotation
ATTN_DROP = 0.30      # (attn_base - attn_it) >= this -> realized attention collapsed
FULL_TOL = 0.85       # recovery_full >= this -> collapse is entirely residual-input-mediated at L

DECISION_RULE = (
    "weight_mag CHANGED if |fro_rel|>=0.10, UNCHANGED if <0.05; weight_dir QK_SAME_DIR if "
    "dir_cos>=0.95 and resid_frac<=0.20 else QK_ROTATED; pattern COLLAPSED if "
    "(attn_base-attn_it)>=0.30; dominant_side QUERY if recovery_query>=recovery_key else KEY; "
    "cause (when COLLAPSED): weights intact -> INPUT_MEDIATED if recovery_full>=0.85 else "
    "PARTLY_LOCAL; weights changed -> WEIGHT_LOCAL if recovery_full<0.85 else MIXED."
)

DEFAULT_HEADS = [(18, 5), (18, 6)]
SWAP_MODES = ("query", "key", "full")

# Capital-pair salience prompts, frozen to match base_attn_qa.py / job_head_profile.py.
PAIRS = [("Australia", "Sydney", "Canberra"), ("Texas", "Houston", "Austin"),
         ("Canada", "Toronto", "Ottawa"), ("Switzerland", "Zurich", "Bern"),
         ("Morocco", "Casablanca", "Rabat")]


def salience_prompt(region, anchor):
    return (f"{anchor} is the most famous city in {region}. "
            f"The capital of {region} is the city of")


# --------------------------------------------------------------------------- weight side
def gqa_group(cfg):
    """grp = n_heads // n_kv_heads, tolerant of the several attribute names; MHA falls back to 1."""
    n_heads = int(cfg.n_heads)
    n_kv = None
    for attr in ("n_key_value_heads", "n_kv_heads", "num_key_value_heads"):
        v = getattr(cfg, attr, None)
        if v:
            n_kv = int(v)
            break
    if not n_kv:
        n_kv = n_heads
    assert n_heads % n_kv == 0, f"n_heads={n_heads} not divisible by n_kv={n_kv}"
    return n_heads // n_kv


def w_qk_matrix(W_Q, W_K, layer, head, grp):
    """W_QK = W_Q[L,H] @ W_K[L, H//grp].T in float32, plus the GQA kv-head index."""
    kvH = head // grp
    wq = W_Q[layer, head].to(torch.float32)
    wk = W_K[layer, kvH].to(torch.float32)
    return wq @ wk.T, kvH


def fro(M):
    return float(torch.linalg.norm(M))


def fro_rel(fro_base, fro_it):
    return (fro_it - fro_base) / max(abs(fro_base), 1e-9)


def qk_scalar_decomp(M_base, M_it):
    b, t = M_base.flatten().to(torch.float32), M_it.flatten().to(torch.float32)
    bb = float(b @ b)
    alpha = float(t @ b) / bb if bb > 0 else 0.0
    resid_frac = float((t - alpha * b).norm() / (t.norm() + 1e-9))
    dir_cos = float(torch.nn.functional.cosine_similarity(b, t, dim=0))
    return alpha, resid_frac, dir_cos


# --------------------------------------------------------------------------- realized side
def anchor_positions(model, ids_list, anchor):
    aset = set(model.to_tokens(anchor, prepend_bos=False)[0].tolist())
    return [i for i, t in enumerate(ids_list) if t in aset and i > 0]


def swap_positions(mode, readout, seq_len):
    """Residual positions to overwrite with base, per swap mode."""
    if mode == "query":
        return [readout]
    if mode == "key":
        return [i for i in range(seq_len) if i != readout]
    return list(range(seq_len))   # full


def base_attn_and_resid(model, heads):
    """Base pass: mean attention-to-anchor per (L,H), plus cached resid_pre per pair/layer."""
    layers = sorted({l for (l, _h) in heads})
    pat_names = {f"blocks.{l}.attn.hook_pattern" for l in layers}
    resid_names = {f"blocks.{l}.hook_resid_pre" for l in layers}
    nf = lambda n: n in pat_names or n in resid_names  # noqa: E731
    sums, counts, store = {}, {}, []
    for region, anchor, _cap in PAIRS:
        ids = model.to_tokens(salience_prompt(region, anchor))
        apos = anchor_positions(model, ids[0].tolist(), anchor)
        with torch.no_grad():
            _, cache = model.run_with_cache(ids, names_filter=nf)
        for l in layers:
            pat = cache[f"blocks.{l}.attn.hook_pattern"][0]
            for h in range(pat.shape[0]):
                w = float(pat[h][-1][apos].sum()) if apos else 0.0
                sums[(l, h)] = sums.get((l, h), 0.0) + w
                counts[(l, h)] = counts.get((l, h), 0) + 1
        store.append({"resid": {l: cache[f"blocks.{l}.hook_resid_pre"][0].detach().clone()
                                for l in layers},
                      "apos": apos, "readout": ids.shape[1] - 1, "seq_len": ids.shape[1]})
    attn = {k: sums[k] / counts[k] for k in sums}
    return attn, store


def it_attn_and_swaps(model, heads, base_store):
    """IT pass: plain attention-to-anchor per (L,H), and the query/key/full residual-swap attention."""
    layers = sorted({l for (l, _h) in heads})
    heads_by_layer = {l: [h for (ll, h) in heads if ll == l] for l in layers}
    pat_names = {f"blocks.{l}.attn.hook_pattern" for l in layers}
    nf = lambda n: n in pat_names  # noqa: E731
    sums, counts = {}, {}
    ssum = {m: {} for m in SWAP_MODES}
    scnt = {m: {} for m in SWAP_MODES}
    for pi, (region, anchor, _cap) in enumerate(PAIRS):
        ids = model.to_tokens(salience_prompt(region, anchor))
        apos = anchor_positions(model, ids[0].tolist(), anchor)
        readout, seq_len = base_store[pi]["readout"], base_store[pi]["seq_len"]
        with torch.no_grad():
            _, cache = model.run_with_cache(ids, names_filter=nf)
        for l in layers:
            pat = cache[f"blocks.{l}.attn.hook_pattern"][0]
            for h in range(pat.shape[0]):
                w = float(pat[h][-1][apos].sum()) if apos else 0.0
                sums[(l, h)] = sums.get((l, h), 0.0) + w
                counts[(l, h)] = counts.get((l, h), 0) + 1
        for l in layers:
            base_resid = base_store[pi]["resid"][l]
            for m in SWAP_MODES:
                positions = swap_positions(m, readout, seq_len)

                def patch(resid, hook, _br=base_resid, _pos=positions):
                    for p in _pos:
                        resid[0, p, :] = _br[p].to(resid.dtype)
                    return resid

                with model.hooks(fwd_hooks=[(f"blocks.{l}.hook_resid_pre", patch)]):
                    with torch.no_grad():
                        _, c2 = model.run_with_cache(
                            ids, names_filter=lambda n, _t=f"blocks.{l}.attn.hook_pattern": n == _t)
                pat2 = c2[f"blocks.{l}.attn.hook_pattern"][0]
                for h in heads_by_layer[l]:
                    w2 = float(pat2[h][-1][apos].sum()) if apos else 0.0
                    ssum[m][(l, h)] = ssum[m].get((l, h), 0.0) + w2
                    scnt[m][(l, h)] = scnt[m].get((l, h), 0) + 1
    attn = {k: sums[k] / counts[k] for k in sums}
    swaps = {m: {k: ssum[m][k] / scnt[m][k] for k in ssum[m]} for m in SWAP_MODES}
    return attn, swaps


# --------------------------------------------------------------------------- neutral verdict
def recovery_frac(attn_base, attn_it, attn_swap):
    denom = attn_base - attn_it
    return (attn_swap - attn_it) / denom if abs(denom) > 1e-6 else 0.0


def classify(frorel, dir_cos, resid_fr, attn_base, attn_it, rec_q, rec_k, rec_full):
    if abs(frorel) >= WQK_THR:
        weight_mag = "CHANGED"
    elif abs(frorel) < WQK_SMALL:
        weight_mag = "UNCHANGED"
    else:
        weight_mag = "BORDERLINE"
    weight_dir = "QK_SAME_DIR" if (dir_cos >= QK_DIR_TOL and resid_fr <= QK_RESID_TOL) else "QK_ROTATED"
    pattern = "COLLAPSED" if (attn_base - attn_it) >= ATTN_DROP else "STABLE"
    dominant = "QUERY" if rec_q >= rec_k else "KEY"
    weights_intact = (weight_mag == "UNCHANGED" and weight_dir == "QK_SAME_DIR")
    if pattern != "COLLAPSED":
        cause = "NA"
    elif weights_intact:
        cause = "INPUT_MEDIATED" if rec_full >= FULL_TOL else "PARTLY_LOCAL"
    else:
        cause = "WEIGHT_LOCAL" if rec_full < FULL_TOL else "MIXED"
    return {"weight_mag": weight_mag, "weight_dir": weight_dir, "pattern": pattern,
            "dominant_side": dominant, "cause": cause}


# --------------------------------------------------------------------------- model run
def load_model(name, device):
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16)
    model.eval()
    return model.to(device)


def run_measure(args):
    heads = parse_heads(args.heads) if args.heads else list(DEFAULT_HEADS)

    print(f"[load] base {args.name_base}", flush=True)
    m_base = load_model(args.name_base, args.device)
    grp = gqa_group(m_base.cfg)
    Wq_b, Wk_b = m_base.W_Q.detach(), m_base.W_K.detach()
    qk_base = {(l, h): w_qk_matrix(Wq_b, Wk_b, l, h, grp) for (l, h) in heads}
    attn_b, base_store = base_attn_and_resid(m_base, heads)
    del m_base

    print(f"[load] it {args.name_it}", flush=True)
    m_it = load_model(args.name_it, args.device)
    assert gqa_group(m_it.cfg) == grp, "GQA group differs between models"
    Wq_i, Wk_i = m_it.W_Q.detach(), m_it.W_K.detach()
    qk_it = {(l, h): w_qk_matrix(Wq_i, Wk_i, l, h, grp) for (l, h) in heads}
    attn_i, swaps = it_attn_and_swaps(m_it, heads, base_store)
    del m_it

    out_heads = {}
    for (l, h) in heads:
        Mb, kvH = qk_base[(l, h)]
        Mi, _ = qk_it[(l, h)]
        fb, fi = fro(Mb), fro(Mi)
        rel = fro_rel(fb, fi)
        alpha, resid_fr, dir_cos = qk_scalar_decomp(Mb, Mi)
        ab, ai = float(attn_b.get((l, h), 0.0)), float(attn_i.get((l, h), 0.0))
        rq = recovery_frac(ab, ai, float(swaps["query"].get((l, h), 0.0)))
        rk = recovery_frac(ab, ai, float(swaps["key"].get((l, h), 0.0)))
        rfull = recovery_frac(ab, ai, float(swaps["full"].get((l, h), 0.0)))
        verdict = classify(rel, dir_cos, resid_fr, ab, ai, rq, rk, rfull)
        out_heads[f"L{l}.H{h}"] = {
            "layer": l, "head": h, "kv_head": kvH, "grp": grp,
            "fro_base": fb, "fro_it": fi, "fro_rel": rel,
            "qk_alpha": alpha, "qk_resid_frac": resid_fr, "qk_dir_cos": dir_cos,
            "attn_base": ab, "attn_it": ai, "attn_drop": ab - ai,
            "attn_swap_query": float(swaps["query"].get((l, h), 0.0)),
            "attn_swap_key": float(swaps["key"].get((l, h), 0.0)),
            "attn_swap_full": float(swaps["full"].get((l, h), 0.0)),
            "recovery_query": round(rq, 4), "recovery_key": round(rk, 4),
            "recovery_full": round(rfull, 4),
            **verdict,
        }
        print(f"L{l}.H{h}: fro_rel={rel:+.4f} dir_cos={dir_cos:.4f} | attn b/it={ab:.3f}/{ai:.3f} "
              f"rec q/k/full={rq:+.3f}/{rk:+.3f}/{rfull:+.3f} -> "
              f"[{verdict['weight_mag']},{verdict['weight_dir']},{verdict['pattern']},"
              f"{verdict['cause']},dom={verdict['dominant_side']}]", flush=True)

    result = {
        "name_base": args.name_base, "name_it": args.name_it, "device": args.device, "grp": grp,
        "pairs": [list(p) for p in PAIRS],
        "thresholds": {"WQK_THR": WQK_THR, "WQK_SMALL": WQK_SMALL, "QK_DIR_TOL": QK_DIR_TOL,
                       "QK_RESID_TOL": QK_RESID_TOL, "ATTN_DROP": ATTN_DROP, "FULL_TOL": FULL_TOL},
        "decision_rule": DECISION_RULE, "heads": out_heads,
    }
    Path("out").mkdir(exist_ok=True)
    Path(f"out/qk_weight_{args.tag}.json").write_text(json.dumps(result, indent=2))
    print(f"[done] wrote out/qk_weight_{args.tag}.json", flush=True)


def parse_heads(spec):
    heads = []
    for part in spec.split(";"):
        part = part.strip()
        if part:
            l, h = part.split(",")
            heads.append((int(l), int(h)))
    assert heads, f"no heads parsed from {spec!r}"
    return heads


# --------------------------------------------------------------------------- selftest (model-free)
def selftest():
    torch.manual_seed(0)
    d_model, d_head = 8, 4

    # pure scaling: W_QK_it = 1.5 * W_QK_base -> fro_rel 0.5, alpha 1.5, resid_frac ~0, dir_cos ~1
    base = torch.randn(d_model, d_head)
    Mb = base @ base.T
    Mi = 1.5 * Mb
    assert abs(fro_rel(fro(Mb), fro(Mi)) - 0.5) < 1e-4
    a, rf, dc = qk_scalar_decomp(Mb, Mi)
    assert abs(a - 1.5) < 1e-4 and rf < 1e-4 and dc > 0.999, (a, rf, dc)

    # rotation -> high resid_frac
    g = torch.Generator().manual_seed(3)
    _, rf2, _ = qk_scalar_decomp(Mb, torch.randn(d_model, d_model, generator=g))
    assert rf2 > 0.5

    # swap_positions
    assert swap_positions("query", 5, 8) == [5]
    assert swap_positions("key", 5, 8) == [0, 1, 2, 3, 4, 6, 7]
    assert swap_positions("full", 5, 8) == list(range(8))

    # recovery_frac + zero-denominator guard
    assert abs(recovery_frac(0.58, 0.016, 0.55) - (0.534 / 0.564)) < 1e-4
    assert recovery_frac(0.58, 0.58, 0.10) == 0.0

    # classify: weights intact + full recovery >= FULL_TOL -> INPUT_MEDIATED, dominant side reported
    v = classify(-0.005, 0.998, 0.07, 0.58, 0.016, 0.42, 0.30, 0.95)
    assert v["cause"] == "INPUT_MEDIATED" and v["dominant_side"] == "QUERY", v

    # classify: weights intact but full recovery < FULL_TOL -> PARTLY_LOCAL
    assert classify(-0.005, 0.998, 0.07, 0.58, 0.016, 0.42, 0.30, 0.40)["cause"] == "PARTLY_LOCAL"

    # classify: weights changed (rotated) + low full recovery -> WEIGHT_LOCAL
    assert classify(0.0, 0.30, 0.80, 0.58, 0.016, 0.10, 0.10, 0.20)["cause"] == "WEIGHT_LOCAL"

    # classify: weights changed (magnitude) but full recovery high -> MIXED
    assert classify(0.50, 0.999, 0.0, 0.58, 0.016, 0.40, 0.40, 0.95)["cause"] == "MIXED"

    # classify: no collapse -> NA
    assert classify(0.0, 0.999, 0.0, 0.58, 0.57, 0.0, 0.0, 0.0)["cause"] == "NA"

    # dominant side flips on key > query
    assert classify(-0.005, 0.998, 0.07, 0.58, 0.016, 0.20, 0.60, 0.95)["dominant_side"] == "KEY"

    # GQA mapping kvH = H // grp
    class _Cfg:
        n_heads = 8
        n_key_value_heads = 4
    grp = gqa_group(_Cfg())
    assert grp == 2
    W_Q = torch.randn(1, 8, d_model, d_head)
    W_K = torch.randn(1, 4, d_model, d_head)
    _, k4 = w_qk_matrix(W_Q, W_K, 0, 4, grp)
    _, k5 = w_qk_matrix(W_Q, W_K, 0, 5, grp)
    assert k4 == k5 == 2

    # bf16 guard
    Wb = torch.randn(1, 8, d_model, d_head).to(torch.bfloat16)
    Wk = torch.randn(1, 4, d_model, d_head).to(torch.bfloat16)
    M, _ = w_qk_matrix(Wb, Wk, 0, 5, grp)
    assert math.isfinite(fro(M)) and fro(M) > 0

    # MHA fallback
    class _CfgMHA:
        n_heads = 8
    assert gqa_group(_CfgMHA()) == 1

    print("[selftest] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--heads", default=None, help="basket as 'L,H;L,H' (default 18,5;18,6)")
    p.add_argument("--name-base", default="google/gemma-2-2b")
    p.add_argument("--name-it", default="google/gemma-2-2b-it")
    p.add_argument("--tag", default="2b")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        run_measure(args)


if __name__ == "__main__":
    main()
