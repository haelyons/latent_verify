"""R1-INSTR -- instrument triangulation on the KNOWN 2b salience case (RESEARCH_QUESTIONS A1-A5).

Validate two new causal instruments against the incumbent on the one case where ground truth is
established end to end: the salience flip on gemma-2-2b base, reader head L18.H5, all-heads
anchor-knockout necessity ~1.0, per-head L18.H5 ~0.20-0.24 (FRAMING_NOTES sec-3.7/3.10). A new
instrument earns the right to operate in the blind regime (effects <0.5 nat, the R1-DIFF hunt)
ONLY if it reproduces this. If it disagrees here, that disagreement is the headline -- the knockout
necessity numbers (incl. C1/C3) may be off-distribution artifacts -- and R1-DIFF does not run.

Metric (the established readout): M = logp(C) - logp(W) at the "...is the city of" stem.
  effect = M(neutral) - M(salience)   (>0: salience pulls the answer toward the anchor W)
A head that is a mediator of the pull scores POSITIVE under all three instruments (sign-aligned):

  (1) KNOCKOUT-necessity (incumbent): per head, zero its attention onto the anchor key positions in
      the salience prompt + renormalize; necessity = (M_ko - M_sal) / effect. Off-distribution
      (zero+renorm); undefined when |effect| < MIN_EFFECT (the structural blindspot A2 names).
  (2) ACTIVATION-PATCH (on-distribution, A1): per head, replace its readout-position output z[-1,h]
      with the value from the NEUTRAL run; frac = (M_patch - M_sal) / effect. Substitutes a REAL
      activation (no zero/renorm), so it answers "are the knockout numbers an off-distribution
      artifact?". Run as CONFIRM on the AtP top-K (+ reader + control), per the screen/confirm design.
  (3) ATP attribution (gradient, A2): one backward pass of M on the salience run; per head
      attribution = ((z_neutral[-1,h] - z_salience[-1,h]) . dM/dz_salience[-1,h]). The first-order
      approximation of (2); defined at ANY effect size (no floor), full head coverage in one pass.
      This is the SCREEN. AtP==activation-patch exactly for a locally-linear M (asserted in selftest).

All three rank heads at the readout position; the cross-pair claim is about the READER L18.H5, which
acts at the readout, so readout-position attribution is well matched to it (the Australia-specific
early-writer L0.H2 acts at earlier query positions and is out of this readout-position scope by design;
the incumbent all-position knockout sweep is reported alongside as the reference ranking).

Pre-registered success criteria (fixed before any model run):
  SC-INSTR-1 (concordance): aggregate (mean over pairs) AtP AND activation-patch both rank the reader
      L18.H5 within the top SC1_TOPK heads, and the control head L18.H6 sits outside top SC1_CTRL_OUT
      with |frac(ctrl)| <= SC1_CTRL_FRAC.
  SC-INSTR-2 (small-effect range): on the QA-scaffold pairs, where >=1 pair has |effect| < MIN_EFFECT
      (knockout returns n/a there, reproducing CHAT_FORMAT mean QA effect +0.56, 3/5 below floor),
      AtP returns finite, non-degenerate (std>0), ranked scores. The floor is gone.
  SC-INSTR-3 (on-distribution specificity + linear validity): activation-patch reader mean frac
      >= MIN_FRAC and control mean |frac| <= SC1_CTRL_FRAC; AtP and activation-patch agree in SIGN on
      >= SIGN_AGREE of the confirm-set heads (the linear approximation is valid on this case).

Decision:
  all three pass -> "CONCORDANT": adopt AtP-screen + activation-patch-confirm as the R1-DIFF
      instrument; the necessity floor is retired (SC-INSTR-2).
  reader is top under the incumbent KNOCKOUT but NOT top under AtP/activation-patch -> "DISCORDANT":
      the new on-distribution instruments disagree with knockout on the known case. Headline result;
      do NOT proceed to the blind regime; escalate to re-examining the incumbent.
  otherwise -> "INCONCLUSIVE" (e.g. reader not top under any instrument -> faithfulness gate failed).

Usage:
  python instr_triangulation.py --selftest    # model-free: AtP==activation-patch for linear M,
                                               # reader ranks top, decision logic fires
  python instr_triangulation.py               # gemma-2-2b base -> out/instr_triangulation_2b.json
  python instr_triangulation.py --topk 10 --no-knockout-sweep   # skip the heavy incumbent full sweep
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

READER = (18, 5)                                   # sec-3.10 universal salience reader (2b)
CTRL = (18, 6)                                     # matched control head, same layer
STEM = "The capital of {r} is the city of"
SALIENCE = "{w} is the most famous city in {r}. "
QA_STEM = ("Question: What is the capital of {r}?\n"
           "Answer: The capital of {r} is the city of")
PAIRS = [                                           # (region, capital=C, anchor=W) -- the sec-3.x set
    ("Australia",   "Canberra", "Sydney"),
    ("Texas",       "Austin",   "Houston"),
    ("Canada",      "Ottawa",   "Toronto"),
    ("Switzerland", "Bern",     "Zurich"),
    ("Morocco",     "Rabat",    "Casablanca"),
]

MIN_EFFECT = 0.5          # nats; the incumbent floor below which knockout necessity is n/a
SC1_TOPK = 5              # reader must rank within top-K under AtP and activation-patch
SC1_CTRL_OUT = 20         # control head must rank outside top-this
SC1_CTRL_FRAC = 0.05      # and contribute <= this |frac| of the effect
MIN_FRAC = 0.10           # activation-patch reader must restore >= this fraction of the effect
SIGN_AGREE = 0.80         # AtP vs activation-patch sign agreement on the confirm set
TOPK_CONFIRM = 10         # #AtP-top heads to confirm with activation-patch


# --------------------------------------------------------------------------- pure instrument math
def atp_scores(z_clean, z_corrupt, grad):
    """First-order attribution of patching each head's readout output from corrupt->clean.
    z_clean, z_corrupt, grad: [n_heads, d_head] (grad = dM/dz at the corrupt/salience run).
    Returns [n_heads]: AtP estimate of (M_patch - M_corrupt) per head. Exact for locally-linear M."""
    return ((z_clean - z_corrupt) * grad).sum(-1)


def linear_patch_delta(z_clean, z_corrupt, w, h):
    """Reference (selftest only): exact Delta-M of replacing head h's z, for M(Z)=sum_h w_h . z_h + c.
    Equals atp_scores()[h] when grad = w. Used to prove AtP == activation-patch on a linear metric."""
    return float(torch.dot(w[h], (z_clean[h] - z_corrupt[h])))


def flat(L, H, nH):
    return L * nH + H


def rank_of(scores_flat, target_flat):
    """Descending-score rank (0 = top). scores_flat: list/1d over flat head ids."""
    v = scores_flat[target_flat]
    return int(sum(1 for s in scores_flat if s > v))


def sign_agreement(a, b):
    """Fraction of paired entries with the same sign (zeros count as agreeing with zero)."""
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if (x > 0) == (y > 0)) / len(a)


def decide(reader_atp_rank, reader_patch_rank, ctrl_atp_rank, ctrl_patch_frac,
           reader_patch_frac, sign_agree, reader_knockout_rank, atp_finite_below_floor):
    """Pure decision over the aggregate numbers (exercised by --selftest)."""
    sc1 = (reader_atp_rank < SC1_TOPK and reader_patch_rank < SC1_TOPK and
           ctrl_atp_rank >= SC1_CTRL_OUT and abs(ctrl_patch_frac) <= SC1_CTRL_FRAC)
    sc2 = bool(atp_finite_below_floor)
    sc3 = (reader_patch_frac >= MIN_FRAC and abs(ctrl_patch_frac) <= SC1_CTRL_FRAC and
           sign_agree >= SIGN_AGREE)
    reader_top_knockout = reader_knockout_rank is not None and reader_knockout_rank < SC1_TOPK
    reader_top_new = reader_atp_rank < SC1_TOPK and reader_patch_rank < SC1_TOPK
    if sc1 and sc2 and sc3:
        v = ("CONCORDANT: AtP and activation-patch reproduce the incumbent on the known case; "
             "adopt AtP-screen + activation-patch-confirm for R1-DIFF; necessity floor retired (SC-INSTR-2)")
    elif reader_top_knockout and not reader_top_new:
        v = ("DISCORDANT: the on-distribution instruments do NOT rank the reader top while knockout "
             "does -- knockout necessity may be an off-distribution artifact; HEADLINE, do not enter "
             "the blind regime, re-examine the incumbent")
    else:
        v = "INCONCLUSIVE: faithfulness/criteria not met as pre-registered; inspect per-pair rows"
    return {"SC_INSTR_1_concordance": sc1, "SC_INSTR_2_small_effect": sc2,
            "SC_INSTR_3_specificity_linear": sc3,
            "reader_atp_rank": reader_atp_rank, "reader_patch_rank": reader_patch_rank,
            "reader_knockout_rank": reader_knockout_rank, "ctrl_atp_rank": ctrl_atp_rank,
            "ctrl_patch_frac": round(ctrl_patch_frac, 4), "reader_patch_frac": round(reader_patch_frac, 4),
            "sign_agreement": round(sign_agree, 3), "verdict": v}


# --------------------------------------------------------------------------- real run
def _logp_diff(logits, cid, aid):
    lp = torch.log_softmax(logits[0, -1].float(), -1)
    return lp[cid] - lp[aid]                       # keeps grad for AtP


def run(topk, do_knockout_sweep, name):
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {name} on {device}", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tok = model.tokenizer
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    print(f"[load] done (L={nL} H={nH})", flush=True)
    first = lambda s: tok.encode(s, add_special_tokens=False)[0]
    zname = lambda L: f"blocks.{L}.attn.hook_z"
    pat_filter = lambda n: n.endswith("hook_pattern")
    rL, rH = READER
    cL, cH = CTRL

    def to_ids(s):
        return model.to_tokens(s).to(device)

    def anchor_pos(ids_list, anchor):
        aset = set(model.to_tokens(anchor, prepend_bos=False)[0].tolist())
        return [i for i, t in enumerate(ids_list) if t in aset and i > 0]

    def M_nohook(ids, cid, aid):
        with torch.no_grad():
            return float(_logp_diff(model(ids), cid, aid))

    def cache_z_last(ids):
        """Neutral run: store z[-1] per layer, detached. Returns dict L -> [nH, d_head]."""
        store = {}
        def grab(z, hook):
            store[hook.layer()] = z[0, -1].detach().clone()
            return z
        with torch.no_grad():
            model.run_with_hooks(ids, fwd_hooks=[(zname(L), grab) for L in range(nL)])
        return store

    def atp_and_zsal(ids, cid, aid):
        """Salience run with grad: returns (M_sal float, z_sal[L][-1], grad[L][-1]) per layer."""
        kept = {}
        def grab(z, hook):
            z.retain_grad()
            kept[hook.layer()] = z
            return z
        logits = model.run_with_hooks(ids, fwd_hooks=[(zname(L), grab) for L in range(nL)])
        M = _logp_diff(logits, cid, aid)
        model.zero_grad(set_to_none=True)
        M.backward()
        z_sal = {L: kept[L][0, -1].detach().clone() for L in kept}
        grad = {L: kept[L].grad[0, -1].detach().clone() for L in kept}
        return float(M.detach()), z_sal, grad

    def patch_frac(ids, cid, aid, L, H, z_neu_val, M_sal, effect):
        """Activation-patch head (L,H) readout output to its neutral value; fraction of effect restored."""
        def hook(z, hook):
            z[0, -1, H, :] = z_neu_val.to(z.dtype)
            return z
        with torch.no_grad():
            M_p = float(_logp_diff(model.run_with_hooks(ids, fwd_hooks=[(zname(L), hook)]), cid, aid))
        return (M_p - M_sal) / effect

    def knockout_nec(ids, cid, aid, L, H, apos, M_sal, effect):
        nm = f"blocks.{L}.attn.hook_pattern"
        def hook(p, hook):
            p[:, H, :, apos] = 0.0
            p[:, H] = p[:, H] / p[:, H].sum(-1, keepdim=True).clamp_min(1e-9)
            return p
        with torch.no_grad():
            M_ko = float(_logp_diff(model.run_with_hooks(ids, fwd_hooks=[(nm, hook)]), cid, aid))
        return (M_ko - M_sal) / effect

    # ---- main case: salience stem ----
    nhf = nL * nH
    atp_per_pair, patch_rows, knock_per_pair = [], [], []
    pair_meta = []
    for region, C, W in PAIRS:
        cid, aid = first(" " + C), first(" " + W)
        neu = to_ids(STEM.format(r=region))                          # neutral = bare stem
        sal = to_ids(SALIENCE.format(w=W, r=region) + STEM.format(r=region))
        M_neu = M_nohook(neu, cid, aid)
        z_neu = cache_z_last(neu)
        M_sal, z_sal, grad = atp_and_zsal(sal, cid, aid)
        effect = M_neu - M_sal
        pair_meta.append({"pair": f"{region}->{C}", "effect": round(effect, 4),
                          "M_neutral": round(M_neu, 4), "M_salience": round(M_sal, 4)})
        print(f"[pair] {region:<12} effect={effect:+.3f}", flush=True)
        if abs(effect) < 1e-6:
            continue
        # AtP over all heads (one backward already done)
        atp_flat = [0.0] * nhf
        for L in z_sal:
            s = atp_scores(z_neu[L], z_sal[L], grad[L]) / effect      # normalize to fraction-of-effect
            for H in range(nH):
                atp_flat[flat(L, H, nH)] = float(s[H])
        atp_per_pair.append(atp_flat)
        # incumbent knockout per-head full sweep (optional)
        if do_knockout_sweep:
            apos = anchor_pos(sal[0].tolist(), W)
            kflat = [0.0] * nhf
            for L in range(nL):
                for H in range(nH):
                    kflat[flat(L, H, nH)] = knockout_nec(sal, cid, aid, L, H, apos, M_sal, effect)
                print(f"  [knockout] {region} swept layer {L}", flush=True)
            knock_per_pair.append(kflat)
        # activation-patch confirm: AtP top-K (by value) UNION {reader, ctrl}
        order = sorted(range(nhf), key=lambda f: atp_flat[f], reverse=True)[:topk]
        confirm = sorted(set(order) | {flat(rL, rH, nH), flat(cL, cH, nH)})
        row = {"pair": f"{region}->{C}"}
        for f in confirm:
            L, H = divmod(f, nH)
            row[f] = patch_frac(sal, cid, aid, L, H, z_neu[L][H], M_sal, effect)
        patch_rows.append(row)

    def agg(per_pair):
        return [statistics.mean(p[f] for p in per_pair) for f in range(nhf)] if per_pair else []

    atp_agg = agg(atp_per_pair)
    knock_agg = agg(knock_per_pair) if knock_per_pair else None
    # activation-patch aggregate only over the confirm heads (others absent -> excluded from ranking)
    confirm_heads = sorted(set().union(*[set(k for k in r if isinstance(k, int)) for r in patch_rows])) if patch_rows else []
    patch_agg_map = {f: statistics.mean(r[f] for r in patch_rows if f in r) for f in confirm_heads}

    rf, cf = flat(rL, rH, nH), flat(cL, cH, nH)
    reader_atp_rank = rank_of(atp_agg, rf) if atp_agg else None
    ctrl_atp_rank = rank_of(atp_agg, cf) if atp_agg else None
    reader_knock_rank = rank_of(knock_agg, rf) if knock_agg else None
    # activation-patch rank within the confirm set (reader vs the AtP-top heads)
    patch_sorted = sorted(patch_agg_map.items(), key=lambda kv: kv[1], reverse=True)
    reader_patch_rank = next((i for i, (f, _) in enumerate(patch_sorted) if f == rf), 999)
    reader_patch_frac = patch_agg_map.get(rf, 0.0)
    ctrl_patch_frac = patch_agg_map.get(cf, 0.0)
    # AtP vs activation-patch sign agreement on the confirm heads
    sa = sign_agreement([atp_agg[f] for f in confirm_heads], [patch_agg_map[f] for f in confirm_heads]) if confirm_heads else 0.0

    # ---- SC-INSTR-2: QA scaffold, AtP finite below the knockout floor ----
    qa_rows, atp_finite_below_floor = [], False
    for region, C, W in PAIRS:
        cid, aid = first(" " + C), first(" " + W)
        neu = to_ids(QA_STEM.format(r=region))
        sal = to_ids(SALIENCE.format(w=W, r=region) + QA_STEM.format(r=region))
        M_neu = M_nohook(neu, cid, aid)
        z_neu = cache_z_last(neu)
        M_sal, z_sal, grad = atp_and_zsal(sal, cid, aid)
        effect = M_neu - M_sal
        below = abs(effect) < MIN_EFFECT
        finite, std = False, 0.0
        if abs(effect) > 1e-6:
            sc = [float(atp_scores(z_neu[L], z_sal[L], grad[L])[H]) for L in z_sal for H in range(nH)]
            finite = all(s == s for s in sc) and not any(s in (float("inf"), float("-inf")) for s in sc)
            std = statistics.pstdev(sc)
        if below and finite and std > 0:
            atp_finite_below_floor = True
        qa_rows.append({"pair": f"{region}->{C}", "effect": round(effect, 4),
                        "knockout_na_below_floor": below, "atp_finite": finite, "atp_std": round(std, 5)})
        print(f"[qa] {region:<12} effect={effect:+.3f} below_floor={below} atp_finite={finite}", flush=True)

    decision = decide(reader_atp_rank, reader_patch_rank, ctrl_atp_rank, ctrl_patch_frac,
                      reader_patch_frac, sa, reader_knock_rank, atp_finite_below_floor)

    def topn(scores, n=8):
        return [{"L": f // nH, "H": f % nH, "score": round(scores[f], 4)}
                for f in sorted(range(len(scores)), key=lambda f: scores[f], reverse=True)[:n]]

    out = {"model": name, "case": "salience_flip_2b", "reader": list(READER), "ctrl": list(CTRL),
           "n_layers": nL, "n_heads": nH, "pairs": pair_meta,
           "thresholds": {"min_effect": MIN_EFFECT, "sc1_topk": SC1_TOPK, "sc1_ctrl_out": SC1_CTRL_OUT,
                          "sc1_ctrl_frac": SC1_CTRL_FRAC, "min_frac": MIN_FRAC, "sign_agree": SIGN_AGREE},
           "atp_top8": topn(atp_agg) if atp_agg else [],
           "knockout_top8": topn(knock_agg) if knock_agg else None,
           "activation_patch_confirm": [{"L": f // nH, "H": f % nH, "frac": round(v, 4)} for f, v in patch_sorted],
           "qa_small_effect": qa_rows,
           "decision": decision}
    Path("out").mkdir(exist_ok=True)
    Path("out/instr_triangulation_2b.json").write_text(json.dumps(out, indent=2))
    print("\n[decision]", json.dumps(decision, indent=2))
    print("[done] wrote out/instr_triangulation_2b.json")


# --------------------------------------------------------------------------- selftest
def selftest():
    """Model-free. (1) AtP == exact activation-patch Delta-M for a locally-linear metric. (2) a planted
    reader head ranks top while controls sit at ~0. (3) the decision logic fires CONCORDANT and
    DISCORDANT on the matching synthetic aggregates."""
    torch.manual_seed(0)
    nH, d = 8, 16
    z_clean = torch.randn(nH, d)
    z_corrupt = torch.randn(nH, d)
    w = torch.zeros(nH, d)
    # planted "reader" = head 5: large alignment so its patch Delta-M dominates; head 6 (ctrl) ~ 0.
    w[5] = torch.randn(d) * 3.0
    w[1] = torch.randn(d) * 0.5
    # M(Z) = sum_h w_h . z_h  => dM/dz = w (constant) => AtP exact.
    atp = atp_scores(z_clean, z_corrupt, w)
    for h in range(nH):
        exact = linear_patch_delta(z_clean, z_corrupt, w, h)
        assert abs(float(atp[h]) - exact) < 1e-4, f"AtP != activation-patch for linear M at head {h}"
    print("[selftest] AtP == exact activation-patch Delta-M for a locally-linear metric (all heads)")

    # ranking: make head 5 the clear top by giving it the largest aligned delta
    w2 = torch.zeros(nH, d); w2[5] = torch.ones(d)
    zc = torch.zeros(nH, d); zk = torch.zeros(nH, d)
    zc[5] = torch.ones(d)                              # (z_clean-z_corrupt).w = d for head 5, 0 else
    s = atp_scores(zc, zk, w2).tolist()
    assert rank_of(s, 5) == 0, f"reader head should rank top: {rank_of(s,5)}"
    assert rank_of(s, 6) >= 1, "control head should not rank top"
    print(f"[selftest] planted reader ranks top (rank {rank_of(s,5)}); control rank {rank_of(s,6)}")

    # sign agreement
    assert abs(sign_agreement([1.0, -1.0, 0.5], [2.0, -3.0, 0.1]) - 1.0) < 1e-9
    assert abs(sign_agreement([1.0, -1.0], [1.0, 2.0]) - 0.5) < 1e-9
    print("[selftest] sign_agreement OK")

    # decision: CONCORDANT (reader top under both new instruments, ctrl out, floor retired)
    d_ok = decide(reader_atp_rank=0, reader_patch_rank=0, ctrl_atp_rank=42, ctrl_patch_frac=0.01,
                  reader_patch_frac=0.30, sign_agree=0.95, reader_knockout_rank=1,
                  atp_finite_below_floor=True)
    assert d_ok["verdict"].startswith("CONCORDANT"), d_ok
    # DISCORDANT (knockout ranks reader top, the new instruments do not)
    d_bad = decide(reader_atp_rank=37, reader_patch_rank=40, ctrl_atp_rank=5, ctrl_patch_frac=0.2,
                   reader_patch_frac=0.0, sign_agree=0.3, reader_knockout_rank=1,
                   atp_finite_below_floor=True)
    assert d_bad["verdict"].startswith("DISCORDANT"), d_bad
    print(f"[selftest] decision CONCORDANT: {d_ok['verdict'][:40]}...")
    print(f"[selftest] decision DISCORDANT: {d_bad['verdict'][:40]}...")
    print("[selftest] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="model-free: AtP==patch, ranking, decision")
    ap.add_argument("--topk", type=int, default=TOPK_CONFIRM, help="#AtP-top heads to confirm with activation-patch")
    ap.add_argument("--no-knockout-sweep", action="store_true", help="skip the heavy incumbent full per-head sweep")
    ap.add_argument("--name", default="google/gemma-2-2b")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a.topk, not a.no_knockout_sweep, a.name)
