"""Measurement control: extend the weight-only OV write-magnitude read at reader head L18.H5
(gemma-2-2b base vs -it) with three additional base->it relative-change measurements.

This control is a pure INSTRUMENT. It reports numbers and a neutral threshold verdict on each
number; it does not reference any hypothesis, claim, or confound. It reuses the conventions of
ov_norm_probe.py / job_rlhf_ovqk.py / job_localize208.py verbatim:
  - model loading: HookedTransformer.from_pretrained_no_processing(name, dtype=bf16, device)
                   for ("base","google/gemma-2-2b") and ("it","google/gemma-2-2b-it")
  - GQA map:       n_v = W_V.shape[1]; grp = nH // n_v if n_v < nH else 1; vH = H // grp
  - first-token:   first(s) = tok.encode(s, add_special_tokens=False)[0]; anchor = " " + anchor
  - OV write read: W_OV = W_V[L,vH] @ W_O[L,H]; ov = e_anchor @ W_OV;
                   preln anchor logit = ov @ W_U[:, aid]; W_OV_fro = ||W_OV||_F
  - reader attn:   forward, grab blocks.{L}.attn.hook_pattern[0, H, -1, :], sum over anchor positions
                   (the same SALIENCE + STEM framing prompt used by job_rlhf_ovqk.py)

Three measurements (each reports base, it, and rel_change = (it - base)/|base|):

  (a) GENERALITY -- held-out set of K (>=20) randomly drawn region/anchor pairs, disjoint from the
      small curated set (CURATED). Per-pair and mean of:
        ow_norm     = || e_anchor @ W_OV ||_2          (anchor write-vector magnitude)
        preln_logit = (e_anchor @ W_OV) @ W_U[:, aid]  (anchor logit push BEFORE ln_final)
        W_OV_fro    = || W_OV ||_F                      (head-level; identical across pairs)
      rel_change reported on the MEANS of ow_norm and preln_logit, and on W_OV_fro.

  (b) CONTROL HEAD -- the IDENTICAL OV read on a non-target head (--ctrl-head L,H; default L18.H6),
      over the same held-out pairs, base vs -it. rel_change on mean ow_norm / mean preln_logit /
      W_OV_fro of the control head.

  (c) QK SIDE -- for L18.H5:
        W_QK_fro       = || W_Q[L,H] @ W_K[L,kH].T ||_F   (weight-only QK bilinear magnitude)
        qk_anchor_attn = reader attention mass on the anchor token at the readout query
                         (forward pass, mean over the held-out pairs)
      rel_change reported on both.

NEUTRAL decision: for EACH measured rel_change, |rel_change| > REL_TOL -> "CHANGED" else "UNCHANGED".
Verdicts are reported independently per measurement, in terms of the measured numbers only.

Usage:
  python ov_qk_generality_probe.py --selftest          # model-free; synthetic weights, analytic asserts
  python ov_qk_generality_probe.py                     # gemma-2-2b base & -it -> out/ov_qk_generality_probe_2b.json
  python ov_qk_generality_probe.py --ctrl-head 18,6 --k 24 --seed 0
"""
import argparse
import json
import random
import statistics
from pathlib import Path

import torch

READER = (18, 5)                                   # sec-3.10 universal reader head (2b); same as I2
STEM = "The capital of {r} is the city of"
SALIENCE = "{w} is the most famous city in {r}. "  # salient distractor (anchor = {w})

# Small curated set (the pairs the prior probes used). The held-out generality set is drawn
# RANDOMLY from POOL and is forced disjoint from these, so (a) is not the curated cases.
CURATED = {
    ("Australia",   "Canberra",    "Sydney"),
    ("Texas",       "Austin",      "Houston"),
    ("Canada",      "Ottawa",      "Toronto"),
    ("Switzerland", "Bern",        "Zurich"),
    ("Morocco",     "Rabat",       "Casablanca"),
    ("Florida",     "Tallahassee", "Miami"),
}

# Held-out pool of (region, capital=C, anchor=W) triples, disjoint from CURATED. Drawn from this
# pool at random (seeded) to form the K held-out pairs for measurement (a)/(b). >=20 available.
POOL = [
    ("Brazil",        "Brasilia",     "Rio"),
    ("Turkey",        "Ankara",       "Istanbul"),
    ("China",         "Beijing",      "Shanghai"),
    ("India",         "Delhi",        "Mumbai"),
    ("Pakistan",      "Islamabad",    "Karachi"),
    ("Vietnam",       "Hanoi",        "Saigon"),
    ("Myanmar",       "Naypyidaw",    "Yangon"),
    ("Tanzania",      "Dodoma",       "Dar"),
    ("Nigeria",       "Abuja",        "Lagos"),
    ("Kazakhstan",    "Astana",       "Almaty"),
    ("NewYork",       "Albany",       "Manhattan"),
    ("California",    "Sacramento",   "LosAngeles"),
    ("Illinois",      "Springfield",  "Chicago"),
    ("Washington",    "Olympia",      "Seattle"),
    ("Pennsylvania",  "Harrisburg",   "Philadelphia"),
    ("Missouri",      "Jefferson",    "StLouis"),
    ("Ontario",       "Toronto",      "Hamilton"),
    ("SouthAfrica",   "Pretoria",     "Johannesburg"),
    ("Ecuador",       "Quito",        "Guayaquil"),
    ("Bolivia",       "Sucre",        "LaPaz"),
    ("Cameroon",      "Yaounde",      "Douala"),
    ("CotedIvoire",   "Yamoussoukro", "Abidjan"),
    ("Malaysia",      "Putrajaya",    "KualaLumpur"),
    ("SriLanka",      "Colombo",      "Kandy"),
    ("Liechtenstein", "Vaduz",        "Schaan"),
    ("Belize",        "Belmopan",     "BelizeCity"),
]

REL_TOL = 0.15                                      # neutral threshold on |rel_change|
DEFAULT_CTRL_HEAD = (18, 6)                         # a different head in the same layer 18


def _rel(x_it, x_base):
    """Base->it relative change. Undefined-safe denominator (matches ov_norm_probe._rel)."""
    return (x_it - x_base) / max(abs(x_base), 1e-9)


def _verdict(rel):
    """Neutral, claim-free verdict purely on the measured number."""
    return "CHANGED" if abs(rel) > REL_TOL else "UNCHANGED"


# --------------------------------------------------------------------------- weight-only reads
def ov_read(W_E, W_V, W_O, W_U, L, H, grp, aid):
    """WEIGHT-ONLY OV magnitudes for token `aid`, head (L,H). No forward pass.

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

    W_QK = W_Q[L,H] @ W_K[L,kH].T is the [d_model, d_model] bilinear form whose entry (q,k) is
    the contribution of residual dims (q at query side, k at key side) to the attention score.
    Under GQA the key head kH = H // grp, mirroring the value-head map used on the OV side.
    """
    kH = H // grp if grp > 1 else H                 # GQA: query-head -> key-head
    W_QK = (W_Q[L, H] @ W_K[L, kH].t()).float()     # [d_model, d_model]
    return round(float(W_QK.norm()), 5)


def _summ(rows, key):
    return round(statistics.mean([r[key] for r in rows]), 5)


# --------------------------------------------------------------------------- real run
def run(ctrl_head, k, seed, name_base, name_it):
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Held-out pairs: random, seeded, disjoint from CURATED (guaranteed by construction of POOL).
    pool = [p for p in POOL if p not in CURATED]
    if k > len(pool):
        raise SystemExit(f"--k {k} exceeds held-out pool size {len(pool)}")
    rng = random.Random(seed)
    heldout = rng.sample(pool, k)

    L, H = READER
    cL, cH = ctrl_head
    pat = lambda layer: f"blocks.{layer}.attn.hook_pattern"

    per_model = {}
    for label, name in [("base", name_base), ("it", name_it)]:
        print(f"[load] {name} on {device}", flush=True)
        model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
        model.eval()
        tok = model.tokenizer
        nH, n_v = model.cfg.n_heads, model.W_V.shape[1]
        grp = nH // n_v if n_v and n_v < nH else 1
        first = lambda s: tok.encode(s, add_special_tokens=False)[0]

        def reader_attn(ids, positions, qL, qH):              # QK forward half: grab pattern
            store = {}
            def grab(p, hook):
                store["p"] = p[0, qH, -1, :].detach().float() # head qH, last query, all keys
                return p
            with torch.no_grad():
                model.run_with_hooks(ids, fwd_hooks=[(pat(qL), grab)])
            return float(store["p"][positions].sum()) if positions else 0.0

        ov_rows, ctrl_rows, qk_attn_vals = [], [], []
        for region, cap, anchor in heldout:
            aid = first(" " + anchor)
            # (a) reader-head OV read
            ov = ov_read(model.W_E, model.W_V, model.W_O, model.W_U, L, H, grp, aid)
            ov_rows.append({"region": region, "anchor": anchor, **ov})
            # (b) control-head OV read, identical pipeline
            ovc = ov_read(model.W_E, model.W_V, model.W_O, model.W_U, cL, cH, grp, aid)
            ctrl_rows.append({"region": region, "anchor": anchor, **ovc})
            # (c) QK forward attention onto the anchor token, reader head
            s_ids = model.to_tokens(SALIENCE.format(w=anchor, r=region) + STEM.format(r=region)).to(device)
            aset = set(model.to_tokens(anchor, prepend_bos=False)[0].tolist())
            apos = [i for i, t in enumerate(s_ids[0].tolist()) if t in aset and i > 0]
            qk_attn_vals.append(reader_attn(s_ids, apos, L, H))

        # (c) QK weight-only magnitude (anchor-independent; one value per model)
        wqk = qk_fro(model.W_Q, model.W_K, L, H, grp)

        per_model[label] = {
            "ov_rows": ov_rows,
            "ctrl_rows": ctrl_rows,
            "mean_ow_norm": _summ(ov_rows, "ow_norm"),
            "mean_preln_logit": _summ(ov_rows, "preln_logit"),
            "W_OV_fro": ov_rows[0]["W_OV_fro"],
            "ctrl_mean_ow_norm": _summ(ctrl_rows, "ow_norm"),
            "ctrl_mean_preln_logit": _summ(ctrl_rows, "preln_logit"),
            "ctrl_W_OV_fro": ctrl_rows[0]["W_OV_fro"],
            "W_QK_fro": wqk,
            "mean_qk_anchor_attn": round(statistics.mean(qk_attn_vals), 5),
        }
        print(f"  [{label}] reader L{L}.H{H}: mean_ow_norm={per_model[label]['mean_ow_norm']:.4f} "
              f"mean_preln_logit={per_model[label]['mean_preln_logit']:.4f} W_OV_fro={per_model[label]['W_OV_fro']:.4f}",
              flush=True)
        print(f"  [{label}] ctrl   L{cL}.H{cH}: mean_ow_norm={per_model[label]['ctrl_mean_ow_norm']:.4f} "
              f"mean_preln_logit={per_model[label]['ctrl_mean_preln_logit']:.4f} ctrl_W_OV_fro={per_model[label]['ctrl_W_OV_fro']:.4f}",
              flush=True)
        print(f"  [{label}] qk     L{L}.H{H}: W_QK_fro={per_model[label]['W_QK_fro']:.4f} "
              f"mean_qk_anchor_attn={per_model[label]['mean_qk_anchor_attn']:.4f}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    b, it = per_model["base"], per_model["it"]
    measurements = {
        "a_generality": {
            "mean_ow_norm":     {"base": b["mean_ow_norm"],     "it": it["mean_ow_norm"],     "rel_change": round(_rel(it["mean_ow_norm"], b["mean_ow_norm"]), 4)},
            "mean_preln_logit": {"base": b["mean_preln_logit"], "it": it["mean_preln_logit"], "rel_change": round(_rel(it["mean_preln_logit"], b["mean_preln_logit"]), 4)},
            "W_OV_fro":         {"base": b["W_OV_fro"],         "it": it["W_OV_fro"],         "rel_change": round(_rel(it["W_OV_fro"], b["W_OV_fro"]), 4)},
        },
        "b_control_head": {
            "head": list(ctrl_head),
            "mean_ow_norm":     {"base": b["ctrl_mean_ow_norm"],     "it": it["ctrl_mean_ow_norm"],     "rel_change": round(_rel(it["ctrl_mean_ow_norm"], b["ctrl_mean_ow_norm"]), 4)},
            "mean_preln_logit": {"base": b["ctrl_mean_preln_logit"], "it": it["ctrl_mean_preln_logit"], "rel_change": round(_rel(it["ctrl_mean_preln_logit"], b["ctrl_mean_preln_logit"]), 4)},
            "W_OV_fro":         {"base": b["ctrl_W_OV_fro"],         "it": it["ctrl_W_OV_fro"],         "rel_change": round(_rel(it["ctrl_W_OV_fro"], b["ctrl_W_OV_fro"]), 4)},
        },
        "c_qk_side": {
            "W_QK_fro":            {"base": b["W_QK_fro"],            "it": it["W_QK_fro"],            "rel_change": round(_rel(it["W_QK_fro"], b["W_QK_fro"]), 4)},
            "mean_qk_anchor_attn": {"base": b["mean_qk_anchor_attn"], "it": it["mean_qk_anchor_attn"], "rel_change": round(_rel(it["mean_qk_anchor_attn"], b["mean_qk_anchor_attn"]), 4)},
        },
    }
    decision = _decide(measurements)

    out = {
        "reader": list(READER),
        "ctrl_head": list(ctrl_head),
        "k_heldout": k,
        "seed": seed,
        "heldout_pairs": [list(p) for p in heldout],
        "rel_tol": REL_TOL,
        "decision_rule": "for each rel_change: |rel_change| > REL_TOL -> CHANGED else UNCHANGED",
        "base_summary": {k2: v for k2, v in b.items() if k2 not in ("ov_rows", "ctrl_rows")},
        "it_summary": {k2: v for k2, v in it.items() if k2 not in ("ov_rows", "ctrl_rows")},
        "base_rows": {"ov": b["ov_rows"], "ctrl": b["ctrl_rows"]},
        "it_rows": {"ov": it["ov_rows"], "ctrl": it["ctrl_rows"]},
        "measurements": measurements,
        "decision": decision,
    }
    Path("out").mkdir(exist_ok=True)
    Path("out/ov_qk_generality_probe_2b.json").write_text(json.dumps(out, indent=2))
    print("\n[decision]", json.dumps(decision, indent=2))
    print("[done] wrote out/ov_qk_generality_probe_2b.json")


def _decide(measurements):
    """Neutral verdicts: threshold |rel_change| at REL_TOL per measured number, reported
    independently. No reference to any hypothesis, claim, or confound."""
    out = {}
    for group, metrics in measurements.items():
        out[group] = {}
        for mname, vals in metrics.items():
            if isinstance(vals, dict) and "rel_change" in vals:
                out[group][mname] = {"rel_change": vals["rel_change"], "verdict": _verdict(vals["rel_change"])}
    return out


# --------------------------------------------------------------------------- selftest
def selftest():
    """Model-free. Synthetic weight matrices with KNOWN norms; assert the computed magnitudes
    match analytic values, and that the neutral verdict thresholds at REL_TOL."""
    torch.manual_seed(0)
    d_vocab, d_model, d_head = 64, 16, 16
    # Tied-embedding toy. W_OV = c * I  =>  ov(e) = c*e, so:
    #   ow_norm     = |c| * ||e||                          (e is unit-norm here)
    #   preln_logit = (c*e) . W_U[:,aid] = c * (e . e) = c  (since W_U = W_E.T and e is unit)
    #   W_OV_fro    = ||c*I||_F = |c| * sqrt(d_model)
    W_E = torch.nn.functional.normalize(torch.randn(d_vocab, d_model), dim=1)
    W_U = W_E.t().contiguous()
    I = torch.eye(d_model)
    aid = 7

    # OV: head 0 = copy (c=1), head 1 = attenuated copy (c=1/3). Norms are analytic.
    for c, hidx in [(1.0, 0), (1.0 / 3.0, 1)]:
        W_V = torch.stack([torch.stack([I, I])])                 # [1, 2, d, d]
        W_O = torch.stack([torch.stack([I, I / 3.0])])
        m = ov_read(W_E, W_V, W_O, W_U, 0, hidx, 1, aid)
        exp_ow = abs(c) * 1.0
        exp_logit = c
        exp_fro = abs(c) * (d_model ** 0.5)
        assert abs(m["ow_norm"] - exp_ow) < 1e-3, f"ow_norm {m['ow_norm']} != {exp_ow}"
        assert abs(m["preln_logit"] - exp_logit) < 1e-3, f"preln_logit {m['preln_logit']} != {exp_logit}"
        assert abs(m["W_OV_fro"] - exp_fro) < 1e-3, f"W_OV_fro {m['W_OV_fro']} != {exp_fro}"
    print("[selftest] OV magnitudes match analytic values (copy c=1 and attenuated c=1/3)")

    # QK: W_Q = a*I, W_K = b*I  =>  W_QK = a*b * I  =>  ||W_QK||_F = |a*b| * sqrt(d_model).
    a_q, b_k = 2.0, 0.5
    W_Q = torch.stack([torch.stack([a_q * I])])                  # [1, 1, d, d]
    W_K = torch.stack([torch.stack([b_k * I])])
    fro = qk_fro(W_Q, W_K, 0, 0, 1)
    exp_qk = abs(a_q * b_k) * (d_model ** 0.5)
    assert abs(fro - exp_qk) < 1e-3, f"W_QK_fro {fro} != {exp_qk}"
    print(f"[selftest] QK magnitude matches analytic value: ||W_QK||_F={fro} (expected {exp_qk:.5f})")

    # GQA map: with grp=2, query head 3 maps to value/key head 1. Build distinct kv heads and
    # assert the read uses head H//grp on the V and K stacks.
    kv0, kv1 = 1.0 * I, 4.0 * I
    W_Vg = torch.stack([torch.stack([kv0, kv1])])               # 2 kv heads
    W_Og = torch.stack([torch.stack([I, I, I, I])])            # 4 query heads (O is per-query-head)
    mg = ov_read(W_E, W_Vg, W_Og, W_U, 0, 3, 2, aid)           # H=3, grp=2 -> vH=1 (kv1=4I)
    assert abs(mg["W_OV_fro"] - 4.0 * (d_model ** 0.5)) < 1e-3, f"GQA vH map wrong: {mg['W_OV_fro']}"
    W_Kg = torch.stack([torch.stack([1.0 * I, 3.0 * I])])      # 2 kv key heads
    W_Qg = torch.stack([torch.stack([I, I, I, I])])           # 4 query heads
    frg = qk_fro(W_Qg, W_Kg, 0, 3, 2)                          # H=3, grp=2 -> kH=1 (3I)
    assert abs(frg - 3.0 * (d_model ** 0.5)) < 1e-3, f"GQA kH map wrong: {frg}"
    print("[selftest] GQA head map (query->kv via H//grp) correct on both OV and QK reads")

    # Pool integrity: held-out pool disjoint from curated, K>=20 available.
    pool = [p for p in POOL if p not in CURATED]
    assert set(pool).isdisjoint(CURATED), "held-out pool overlaps curated set"
    assert len(pool) >= 20, f"held-out pool too small for K>=20: {len(pool)}"
    rng = random.Random(0)
    sample = rng.sample(pool, 20)
    assert len(set(sample)) == 20 and set(sample).isdisjoint(CURATED), "held-out sample not disjoint/unique"
    print(f"[selftest] held-out pool size={len(pool)} (>=20), disjoint from curated, sample OK")

    # Neutral verdict thresholding at REL_TOL (claim-free).
    assert _verdict(0.0) == "UNCHANGED" and _verdict(0.10) == "UNCHANGED" and _verdict(REL_TOL) == "UNCHANGED"
    assert _verdict(0.16) == "CHANGED" and _verdict(-0.50) == "CHANGED"
    assert abs(_rel(0.5, 1.0) - (-0.5)) < 1e-9 and abs(_rel(1.3, 1.0) - 0.3) < 1e-9
    # full _decide() shape on synthetic measurements
    synth = {
        "a_generality": {"mean_ow_norm": {"base": 1.0, "it": 0.5, "rel_change": _rel(0.5, 1.0)}},
        "c_qk_side": {"W_QK_fro": {"base": 1.0, "it": 1.05, "rel_change": _rel(1.05, 1.0)}},
    }
    d = _decide(synth)
    assert d["a_generality"]["mean_ow_norm"]["verdict"] == "CHANGED", d
    assert d["c_qk_side"]["W_QK_fro"]["verdict"] == "UNCHANGED", d
    print(f"[selftest] neutral verdict thresholds at REL_TOL={REL_TOL}: {d}")
    print("[selftest] PASS")


def _parse_head(s):
    parts = s.replace(" ", "").split(",")
    if len(parts) != 2:
        raise SystemExit("--ctrl-head takes 'L,H' (two ints)")
    return (int(parts[0]), int(parts[1]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="model-free analytic-norm + verdict check")
    ap.add_argument("--ctrl-head", default=f"{DEFAULT_CTRL_HEAD[0]},{DEFAULT_CTRL_HEAD[1]}",
                    help="non-target control head 'L,H' (default 18,6 = different head, same layer)")
    ap.add_argument("--k", type=int, default=20, help="#held-out random pairs (>=20)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for held-out sampling")
    ap.add_argument("--name-base", default="google/gemma-2-2b")
    ap.add_argument("--name-it", default="google/gemma-2-2b-it")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(_parse_head(a.ctrl_head), a.k, a.seed, a.name_base, a.name_it)
