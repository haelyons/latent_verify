"""R1-MOTIF -- is "gate-don't-delete" a general Gemma post-training strategy? (RESEARCH_QUESTIONS C2).

The C1/I2 result for the salience reader L18.H5: post-training collapses its QK attention onto the
anchor (0.84 -> 0.016) while the OV copy is preserved (pref 0.9997 both, write-norm -0.5%). This asks
whether that QK-gated / OV-preserved signature is a GENERAL strategy of gemma-2-2b post-training, or
specific to the one salience head. If general -> a reusable motif that mechanistically explains the
known fact that post-training reshapes behaviour without destroying capability.

Basket (claim-blind, selected by base mechanistic character, NOT by any RLHF prediction): the top
induction heads of gemma-2-2b base, elicited by a repeated-random-token probe, EXCLUDING the known
salience reader L18.H5 (so the test is on heads UNRELATED to the sycophancy circuit). Each basket head
is a copy/induction head in base; we then ask what post-training did to it.

Per basket head, base vs -it:
  QK (realized pattern):  induction_attn = the head's induction attention on the probe (forward).
  OV (weight-only):       copy_rank / copy_pref of probe tokens under W_U . ln_final(e . W_OV)
                          (DIRECTION of the OV copy), and W_OV_fro = ||W_V@W_O||_F (MAGNITUDE).
Report base, it, and rel_change = (it - base)/|base| for induction_attn and W_OV_fro.

Per-head label (measured numbers only):
  QK_GATED      := rel(induction_attn) <= -GATE_DROP            (realized copy attention collapses)
  OV_PRESERVED  := it copy_rank <= COPY_RANK_TOP AND |rel(W_OV_fro)| <= MAG_TOL  (direction+magnitude kept)
  -> "gate_dont_delete" (QK_GATED & OV_PRESERVED) | "deleted_or_ov_changed" (QK_GATED & not OV_PRESERVED)
     | "untouched" (not QK_GATED & OV_PRESERVED) | "other"

Decision: fraction of the basket labelled "gate_dont_delete".
  >= MAJORITY -> "MOTIF: gate-don't-delete is the general post-training strategy across the basket"
  <  MAJORITY -> "SPECIFIC/MIXED: not a majority pattern (the L18.H5 signature is head-specific or the
                  basket is heterogeneous)"

Caveat (baked in, reported in the artifact): gemma-2-2b-it is SFT + RLHF + model-merge, so a base->it
weight diff conflates three operations -- this is a SCREEN, not a stage attribution (cf. N-5, blocked).
The QK side is the REALIZED pattern (a matched-prompt activation diff, the stronger read); the OV side
is weight-only (direction saturates, so W_OV_fro magnitude carries the dynamic range, per C1).

Usage:
  python gate_dont_delete.py --selftest    # model-free: induction score, copy-score, classify, verdict
  python gate_dont_delete.py               # gemma-2-2b base & -it -> out/gate_dont_delete_2b.json
  python gate_dont_delete.py --n-basket 10 --seq-len 40 --seed 0
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

EXCLUDE = {(18, 5)}        # the salience reader -- basket must be heads OTHER than the sycophancy circuit
GATE_DROP = 0.5            # it induction attn <= 50% of base -> QK gated
MAG_TOL = 0.15             # |rel(W_OV_fro)| within this -> OV magnitude preserved (matches C1 MAG_REL_TOL)
COPY_RANK_TOP = 5          # it OV copy rank <= this -> OV direction preserved
MIN_BASE_INDUCTION = 0.20  # a basket head must be a real induction/copy head in base
N_BASKET = 10
MAJORITY = 0.5


def _rel(x_it, x_base):
    return (x_it - x_base) / max(abs(x_base), 1e-9)


# --------------------------------------------------------------------------- pure measurement math
def induction_score(pattern, S, bos=1):
    """Induction attention on a [t_0..t_{S-1}] x2 repeated sequence (BOS at index `bos-1`=0).
    Second occurrence of token j sits at pos bos+S+j; its induction target (token after the FIRST
    occurrence) sits at pos bos+1+j. Score = mean over j of pattern[bos+S+j, bos+1+j].
    pattern: [Q, K] attention for ONE head. Pure (selftest builds a synthetic pattern)."""
    vals = []
    for j in range(0, S - 1):
        q = bos + S + j
        k = bos + 1 + j
        if q < pattern.shape[0] and k < pattern.shape[1]:
            vals.append(float(pattern[q, k]))
    return statistics.mean(vals) if vals else 0.0


def copy_metrics(W_OV, W_E, W_U, ln_final, token_ids):
    """WEIGHT-ONLY OV copy read: for each probe token t, does the head's OV image of e_t point back at
    t under the unembed? Returns mean copy-rank, frac rank-0, mean copy-pref. (direction; saturates.)"""
    ranks, prefs, r0 = [], [], 0
    for t in token_ids:
        e = W_E[t].float()
        ov = e @ W_OV
        normed = ln_final(ov.unsqueeze(0).unsqueeze(0).to(W_E.dtype))[0, 0].float()
        logits = normed @ W_U.float()
        rank = int((logits > logits[t]).sum().item())
        ranks.append(rank); prefs.append(float(torch.softmax(logits, -1)[t]))
        r0 += int(rank == 0)
    n = len(token_ids)
    return {"mean_copy_rank": round(statistics.mean(ranks), 3), "frac_rank0": round(r0 / n, 3),
            "mean_copy_pref": round(statistics.mean(prefs), 6)}


def classify(rel_induction, it_copy_rank, rel_ovfro):
    """Per-head label from the measured rel-changes (pure; exercised by --selftest)."""
    qk_gated = rel_induction <= -GATE_DROP
    ov_preserved = (it_copy_rank <= COPY_RANK_TOP) and (abs(rel_ovfro) <= MAG_TOL)
    if qk_gated and ov_preserved:
        return "gate_dont_delete"
    if qk_gated and not ov_preserved:
        return "deleted_or_ov_changed"
    if (not qk_gated) and ov_preserved:
        return "untouched"
    return "other"


def motif_verdict(labels):
    n = len(labels)
    if n == 0:
        return {"n": 0, "frac_gate_dont_delete": None, "verdict": "NO BASKET (no copy heads selected)"}
    g = sum(1 for x in labels if x == "gate_dont_delete")
    frac = g / n
    v = ("MOTIF: gate-don't-delete is the general post-training strategy across the basket"
         if frac >= MAJORITY else
         "SPECIFIC/MIXED: not a majority pattern (the L18.H5 signature is head-specific or heterogeneous)")
    return {"n": n, "n_gate_dont_delete": g, "frac_gate_dont_delete": round(frac, 3), "verdict": v}


# --------------------------------------------------------------------------- real run
def _probe_ids(model, S, seed, device):
    """Repeated-random-token probe: BOS + [t_0..t_{S-1}] + [t_0..t_{S-1}]. Same ids for base & it."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = 100, model.cfg.d_vocab - 100
    base = torch.randint(lo, hi, (S,), generator=g)
    seq = torch.cat([base, base]).unsqueeze(0).to(device)
    bos = model.tokenizer.bos_token_id
    return torch.cat([torch.tensor([[bos]], device=device), seq], dim=1)


def _head_patterns(model, ids):
    """Forward once; return dict (L) -> pattern[0] = [head, Q, K] (detached)."""
    store = {}
    def grab(p, hook):
        store[hook.layer()] = p[0].detach().float()
        return p
    hooks = [(f"blocks.{L}.attn.hook_pattern", grab) for L in range(model.cfg.n_layers)]
    with torch.no_grad():
        model.run_with_hooks(ids, fwd_hooks=hooks)
    return store


def run(n_basket, seq_len, seed, name_base, name_it):
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- BASE: select the basket (top induction heads, copy-capable, excluding the salience reader) ----
    print(f"[load] {name_base} on {device}", flush=True)
    base = HookedTransformer.from_pretrained_no_processing(name_base, dtype=torch.bfloat16, device=device)
    base.eval()
    nL, nH = base.cfg.n_layers, base.cfg.n_heads
    n_v = base.W_V.shape[1]
    grp = nH // n_v if n_v and n_v < nH else 1
    ids = _probe_ids(base, seq_len, seed, device)
    sample_tokens = ids[0, 1:1 + seq_len].tolist()                  # the probe's own tokens
    pats = _head_patterns(base, ids)
    ind_all = []
    for L in range(nL):
        for H in range(nH):
            ind_all.append({"L": L, "H": H, "induction": induction_score(pats[L][H], seq_len)})
    ind_all.sort(key=lambda d: d["induction"], reverse=True)
    basket = [(d["L"], d["H"]) for d in ind_all
              if (d["L"], d["H"]) not in EXCLUDE and d["induction"] >= MIN_BASE_INDUCTION][:n_basket]
    print(f"[basket] {len(basket)} induction heads (excl {sorted(EXCLUDE)}): {basket}", flush=True)

    def ov_fro_and_copy(model, L, H):
        vH = H // grp if grp > 1 else H
        W_OV = (model.W_V[L, vH] @ model.W_O[L, H]).float()
        cm = copy_metrics(W_OV, model.W_E, model.W_U, model.ln_final, sample_tokens)
        return round(float(W_OV.norm()), 5), cm

    base_rows = {}
    for (L, H) in basket:
        fro, cm = ov_fro_and_copy(base, L, H)
        base_rows[(L, H)] = {"induction": round(induction_score(pats[L][H], seq_len), 5),
                             "W_OV_fro": fro, **cm}
    del base
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- IT: same heads, same probe ----
    print(f"[load] {name_it} on {device}", flush=True)
    it = HookedTransformer.from_pretrained_no_processing(name_it, dtype=torch.bfloat16, device=device)
    it.eval()
    it_pats = _head_patterns(it, ids)
    it_rows = {}
    for (L, H) in basket:
        fro, cm = ov_fro_and_copy(it, L, H)
        it_rows[(L, H)] = {"induction": round(induction_score(it_pats[L][H], seq_len), 5),
                           "W_OV_fro": fro, **cm}
    del it
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- per-head rel-changes + label ----
    rows, labels = [], []
    for (L, H) in basket:
        b, i = base_rows[(L, H)], it_rows[(L, H)]
        rel_ind = _rel(i["induction"], b["induction"])
        rel_fro = _rel(i["W_OV_fro"], b["W_OV_fro"])
        label = classify(rel_ind, i["mean_copy_rank"], rel_fro)
        labels.append(label)
        rows.append({"head": [L, H],
                     "base": b, "it": i,
                     "rel_induction": round(rel_ind, 4), "rel_W_OV_fro": round(rel_fro, 4),
                     "it_copy_rank": i["mean_copy_rank"], "label": label})
        print(f"  [{L:>2}.{H}] induction {b['induction']:.3f}->{i['induction']:.3f} (rel {rel_ind:+.2f}) "
              f"copy_rank {b['mean_copy_rank']:.1f}->{i['mean_copy_rank']:.1f} "
              f"W_OV_fro rel {rel_fro:+.2f} -> {label}", flush=True)

    verdict = motif_verdict(labels)
    out = {"model_base": name_base, "model_it": name_it, "cue": "gate_dont_delete_motif",
           "n_layers": nL, "n_heads": nH, "seq_len": seq_len, "seed": seed,
           "thresholds": {"gate_drop": GATE_DROP, "mag_tol": MAG_TOL, "copy_rank_top": COPY_RANK_TOP,
                          "min_base_induction": MIN_BASE_INDUCTION, "majority": MAJORITY},
           "excluded_heads": [list(h) for h in sorted(EXCLUDE)],
           "basket": [list(h) for h in basket],
           "rows": rows, "decision": verdict,
           "caveat": "gemma-2-2b-it is SFT+RLHF+merge; base->it weight diff conflates three ops. "
                     "QK side is the realized pattern (activation diff); OV side is weight-only."}
    Path("out").mkdir(exist_ok=True)
    Path("out/gate_dont_delete_2b.json").write_text(json.dumps(out, indent=2))
    print("\n[decision]", json.dumps(verdict, indent=2))
    print("[done] wrote out/gate_dont_delete_2b.json")


# --------------------------------------------------------------------------- selftest
def selftest():
    """Model-free. (1) induction_score reads ~1 on a planted induction pattern, ~0 on uniform.
    (2) copy_metrics: identity OV = perfect copy (rank 0), random OV = not. (3) classify + verdict."""
    S = 5
    Q = K = 1 + 2 * S
    # planted perfect induction: pattern[1+S+j, 1+1+j] = 1 (second-occ attends induction target)
    P = torch.zeros(Q, K)
    for j in range(S - 1):
        P[1 + S + j, 1 + 1 + j] = 1.0
    sc = induction_score(P, S)
    assert sc > 0.99, f"planted induction should score ~1: {sc}"
    U = torch.full((Q, K), 1.0 / K)                       # uniform attention
    su = induction_score(U, S)
    assert su < 0.2, f"uniform attention should score low: {su}"
    print(f"[selftest] induction_score: planted={sc:.3f} uniform={su:.3f}")

    # copy-score: tied-embedding toy with RMSNorm ln_final (strips magnitude, per C1).
    torch.manual_seed(0)
    d_vocab, d_model = 64, 16
    W_E = torch.nn.functional.normalize(torch.randn(d_vocab, d_model), dim=1)
    W_U = W_E.t().contiguous()
    ln = lambda x: x / x.pow(2).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    toks = [3, 7, 11, 20, 41]
    cm_copy = copy_metrics(torch.eye(d_model), W_E, W_U, ln, toks)       # W_OV = I -> perfect copy
    assert cm_copy["frac_rank0"] == 1.0, f"identity OV must copy every token (rank 0): {cm_copy}"
    cm_rand = copy_metrics(torch.randn(d_model, d_model), W_E, W_U, ln, toks)
    assert cm_rand["mean_copy_rank"] > cm_copy["mean_copy_rank"], "random OV should not copy"
    print(f"[selftest] copy_metrics: identity={cm_copy} random_rank={cm_rand['mean_copy_rank']}")

    # classify: QK gated + OV preserved -> gate_dont_delete (the L18.H5 signature)
    assert classify(-0.98, it_copy_rank=0, rel_ovfro=-0.005) == "gate_dont_delete"
    # QK gated + OV magnitude collapses -> deleted_or_ov_changed
    assert classify(-0.95, it_copy_rank=0, rel_ovfro=-0.80) == "deleted_or_ov_changed"
    # QK gated + OV direction lost -> deleted_or_ov_changed
    assert classify(-0.95, it_copy_rank=900, rel_ovfro=-0.01) == "deleted_or_ov_changed"
    # QK intact + OV preserved -> untouched
    assert classify(-0.10, it_copy_rank=1, rel_ovfro=0.02) == "untouched"
    print("[selftest] classify OK (gate_dont_delete / deleted_or_ov_changed / untouched)")

    # verdict majority
    maj = motif_verdict(["gate_dont_delete"] * 6 + ["untouched"] * 2 + ["deleted_or_ov_changed"] * 2)
    assert maj["verdict"].startswith("MOTIF"), maj
    mix = motif_verdict(["gate_dont_delete"] * 3 + ["untouched"] * 4 + ["other"] * 3)
    assert mix["verdict"].startswith("SPECIFIC"), mix
    print(f"[selftest] verdict majority={maj['frac_gate_dont_delete']} -> MOTIF; "
          f"mixed={mix['frac_gate_dont_delete']} -> SPECIFIC")
    print("[selftest] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="model-free measurement + decision check")
    ap.add_argument("--n-basket", type=int, default=N_BASKET)
    ap.add_argument("--seq-len", type=int, default=40, help="repeated-token probe half-length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name-base", default="google/gemma-2-2b")
    ap.add_argument("--name-it", default="google/gemma-2-2b-it")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a.n_basket, a.seq_len, a.seed, a.name_base, a.name_it)
