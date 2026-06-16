"""SUPP-RED -- is L18.H5 *the* reader, or a primary among backups (self-repair / Hydra)?

The lit review (Wang et al. 2023 backup name-movers; McDougall et al. 2023) warns:
knocking out a copy head can trigger compensatory copying by backup heads. §3.10
already shows L18.H5 alone carries only ~0.20 of the copy (12 heads to reach ~0.94),
so "concentrated reader" is already shaky. This test asks the sharper question: is the
rest of the copy a PRE-EXISTING distributed pool, or backups RECRUITED when L18.H5 goes?

Base-2b, framed prompt (copy engaged). Two independent ablation modes:

(1) RECRUITMENT (attention shift). Ablate L18.H5's OUTPUT (zero hook_z head 5) so the
    downstream residual actually changes, then re-read every head's attention-to-anchor
    at the readout. Recruitment = a head whose anchor-attention RISES vs baseline.
    (Only layers >=18 can shift; earlier heads are causally upstream of L18.)

(2) SUPER-ADDITIVE NECESSITY (the Hydra signature). For the top-k baseline anchor-readers
    X, compare:
        nec({X})              = anchor-knockout necessity of X alone
        nec({L18.H5, X}) - nec({L18.H5})   = X's MARGINAL necessity once L18.H5 is off
    Backup/self-repair => marginal(X | L18 off) > nec(X alone)  (X does more work when
    L18 is gone). Equivalently super-additive: nec({L18,X}) > nec({L18}) + nec({X}).

    necessity(set) = (score_with_set's_anchor_attn_zeroed - salience_score) / effect
    effect = score(neutral) - score(salience),  score = logp(cap) - logp(anchor)

Instrument check: all-heads anchor-knockout necessity should reproduce §3.7 (~1.0).

VERDICTS (pre-registered, observation-first -- the per-head table is the finding):
  CONCENTRATED        : nec({L18.H5}) >= 0.30 AND no head recruits (Δattn <= 0.05)
                        AND no super-additive head.  (unlikely given §3.10)
  RECRUITED BACKUPS   : >=1 head with Δattn > 0.05 under L18 ablation, OR a super-additive
                        head.  => "concentrated reader" overstates; it's primary-among-backups.
  PRE-EXISTING POOL   : effect spread over many heads at baseline, but NO recruitment and
                        NO super-additivity.  => L18.H5 is one co-reader of a fixed pool,
                        not a hydra. (most defensible honest claim if it lands here.)

  python job_supp_redundancy.py    # base-2b only -> out/supp_redundancy.json
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

PAIRS = [("Australia", "Sydney", "Canberra"), ("Texas", "Houston", "Austin"),
         ("Canada", "Toronto", "Ottawa"), ("Switzerland", "Zurich", "Bern"),
         ("Morocco", "Casablanca", "Rabat"), ("Turkey", "Istanbul", "Ankara"),
         ("China", "Shanghai", "Beijing"), ("Spain", "Barcelona", "Madrid"),
         ("Italy", "Milan", "Rome"), ("Pakistan", "Karachi", "Islamabad"),
         ("Nigeria", "Lagos", "Abuja"), ("Florida", "Miami", "Tallahassee")]
STEM = "The capital of {r} is the city of"
READER = (18, 5)
MIN_EFFECT = 0.5
TOPK = 8                       # top baseline anchor-readers to run the double-ablation on
RECRUIT_THRESH = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    RL, RH = READER
    print(f"[load] google/gemma-2-2b on {DEVICE}", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(
        "google/gemma-2-2b", dtype=torch.bfloat16, device=DEVICE)
    model.eval()
    tok = model.tokenizer
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    first = lambda s: tok.encode(s, add_special_tokens=False)[0]

    def last_logits(ids, hooks=None):
        with torch.no_grad():
            return (model.run_with_hooks(ids, fwd_hooks=hooks) if hooks else model(ids))[0, -1]

    def score(ids, cid, aid, hooks=None):
        lp = torch.log_softmax(last_logits(ids, hooks).float(), -1)
        return float(lp[cid] - lp[aid])

    def anchor_pos(ids_list, anchor):
        aset = set(model.to_tokens(anchor, prepend_bos=False)[0].tolist())
        return [i for i, t in enumerate(ids_list) if t in aset and i > 0]

    def all_patterns(ids, hooks=None):
        """attention-to-... row at the readout query for every (layer, head)."""
        store = {}
        def mk(layer):
            def grab(pattern, hook):
                store[layer] = pattern[0, :, -1, :].detach().float()   # [head, key]
                return pattern
            return (f"blocks.{layer}.attn.hook_pattern", grab)
        fh = [mk(l) for l in range(n_layers)] + (hooks or [])
        with torch.no_grad():
            model.run_with_hooks(ids, fwd_hooks=fh)
        return store

    def z_ablate(layer, head):
        nm = f"blocks.{layer}.attn.hook_z"
        def hook(z, hook):            # [b, pos, head, d_head]
            z[:, :, head, :] = 0.0
            return z
        return (nm, hook)

    def ko_anchor(pairs, positions):
        """zero anchor attention for the given (layer,head) pairs; renorm those heads."""
        by_layer = defaultdict(list)
        for l, h in pairs:
            by_layer[l].append(h)
        hooks = []
        for l, heads in by_layer.items():
            hs = list(heads)
            def hook(pattern, hook, hs=hs):
                for h in hs:
                    pattern[:, h, :, positions] = 0.0
                    d = pattern[:, h].sum(-1, keepdim=True).clamp_min(1e-9)
                    pattern[:, h] = pattern[:, h] / d
                return pattern
            hooks.append((f"blocks.{l}.attn.hook_pattern", hook))
        return hooks

    recruit_acc = defaultdict(list)        # (l,h) -> list of Δattn(L18 off - base)
    base_attn_acc = defaultdict(list)      # (l,h) -> baseline attn-to-anchor
    double_acc = defaultdict(lambda: {"nec_alone": [], "marginal_given_L18": []})
    allheads_nec, l18_nec = [], []
    rows = []

    for region, anchor, cap in PAIRS:
        cid, aid = first(" " + cap), first(" " + anchor)
        framed = f"{anchor} is the most famous city in {region}. " + STEM.format(r=region)
        ids = model.to_tokens(framed).to(DEVICE)
        apos = anchor_pos(ids[0].tolist(), anchor)
        if not apos:
            continue
        neutral = model.to_tokens(STEM.format(r=region)).to(DEVICE)
        n_sc = score(neutral, cid, aid)
        s_sc = score(ids, cid, aid)
        eff = n_sc - s_sc
        if abs(eff) < MIN_EFFECT:
            continue

        # baseline attention-to-anchor, every head
        base_pat = all_patterns(ids)
        base_attn = {(l, h): float(base_pat[l][h][apos].sum())
                     for l in range(n_layers) for h in range(n_heads)}
        for k, v in base_attn.items():
            base_attn_acc[k].append(v)

        # (1) recruitment: ablate L18.H5 OUTPUT, re-read attention-to-anchor
        abl_pat = all_patterns(ids, hooks=[z_ablate(RL, RH)])
        for l in range(n_layers):
            for h in range(n_heads):
                d = float(abl_pat[l][h][apos].sum()) - base_attn[(l, h)]
                recruit_acc[(l, h)].append(d)

        # necessities
        nec_all = (score(ids, cid, aid, hooks=ko_anchor(
            [(l, h) for l in range(n_layers) for h in range(n_heads)], apos)) - s_sc) / eff
        allheads_nec.append(nec_all)
        nec_l18 = (score(ids, cid, aid, hooks=ko_anchor([(RL, RH)], apos)) - s_sc) / eff
        l18_nec.append(nec_l18)

        # (2) super-additivity for top-k baseline anchor-readers (excluding L18.H5)
        ranked = sorted((k for k in base_attn if k != (RL, RH)),
                        key=lambda k: base_attn[k], reverse=True)[:TOPK]
        for (l, h) in ranked:
            nec_x = (score(ids, cid, aid, hooks=ko_anchor([(l, h)], apos)) - s_sc) / eff
            nec_both = (score(ids, cid, aid, hooks=ko_anchor([(RL, RH), (l, h)], apos)) - s_sc) / eff
            double_acc[(l, h)]["nec_alone"].append(nec_x)
            double_acc[(l, h)]["marginal_given_L18"].append(nec_both - nec_l18)

        rows.append({"pair": f"{region}->{cap}", "effect": eff,
                     "nec_allheads": nec_all, "nec_L18H5_alone": nec_l18,
                     "L18H5_attn_to_anchor": base_attn[(RL, RH)]})
        print(f"  {region:<12} eff={eff:+.2f} nec_all={nec_all:+.2f} "
              f"nec_L18={nec_l18:+.2f} L18->anchor={base_attn[(RL, RH)]:.2f}", flush=True)

    def mean(xs):
        return statistics.mean(xs) if xs else None

    # top recruiters (largest mean rise in anchor-attention once L18.H5 is ablated)
    recruiters = sorted(
        ({"layer": l, "head": h, "mean_delta_attn": mean(v),
          "mean_base_attn": mean(base_attn_acc[(l, h)])}
         for (l, h), v in recruit_acc.items()),
        key=lambda d: d["mean_delta_attn"], reverse=True)[:12]

    # super-additive heads: marginal-given-L18 minus alone (>0 => backup)
    backups = sorted(
        ({"layer": l, "head": h,
          "nec_alone": mean(d["nec_alone"]),
          "marginal_given_L18off": mean(d["marginal_given_L18"]),
          "super_additivity": mean(d["marginal_given_L18"]) - mean(d["nec_alone"])}
         for (l, h), d in double_acc.items()),
        key=lambda r: r["super_additivity"], reverse=True)

    top_recruit = recruiters[0]["mean_delta_attn"] if recruiters else 0.0
    top_super = backups[0]["super_additivity"] if backups else 0.0
    med_l18 = statistics.median(l18_nec) if l18_nec else None
    if (med_l18 or 0) >= 0.30 and top_recruit <= RECRUIT_THRESH and top_super <= RECRUIT_THRESH:
        verdict = "CONCENTRATED: L18.H5 carries the copy, no recruitment, no super-additivity"
    elif top_recruit > RECRUIT_THRESH or top_super > RECRUIT_THRESH:
        verdict = (f"RECRUITED BACKUPS: top Δattn={top_recruit:+.3f} "
                   f"top super-additivity={top_super:+.3f} -> primary-among-backups, not 'the' reader")
    else:
        verdict = "PRE-EXISTING POOL: distributed at baseline, no recruitment/self-repair"

    summary = {
        "reader_head": list(READER), "n": len(rows),
        "median_nec_allheads": (statistics.median(allheads_nec) if allheads_nec else None),
        "median_nec_L18H5_alone": med_l18,
        "top_recruiters_delta_attn": recruiters,
        "top_backups_super_additive": backups[:12],
        "recruit_threshold": RECRUIT_THRESH,
        "verdict": verdict,
    }
    print("\n[summary]", json.dumps(summary, indent=2), flush=True)
    Path("out").mkdir(exist_ok=True)
    Path("out/supp_redundancy.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("[done] wrote out/supp_redundancy.json", flush=True)


if __name__ == "__main__":
    main()
