"""Framing intervention -- does clamping the b1 feature-movers causally mediate
the framing effect? Turns the descriptive b1 observation into a causal test,
using the same circuit-tracer feature_intervention the Dallas->Austin T0 uses.

For each situation with a pinned `target` answer, and each non-baseline
framing, at the PREDICTION position (last token):

  necessity   On the FRAMED prompt, restore the top-K mover features to their
              BASELINE activation values. If the framing effect is carried by
              those features, the target answer reverts toward baseline.
              mediation_fraction = (restored_lp - framed_lp)
                                 / (baseline_lp - framed_lp)
              ~1 = those features fully account for the framing effect at the
              readout; ~0 = they don't.

  control     Same, but K magnitude-matched random non-mover features in the
              same layers restored to baseline -- specificity check (expect ~0).

  sufficiency On the BASELINE prompt, set the same K features to their FRAMED
              values. If sufficient, the answer moves toward the framed
              outcome. sufficiency_fraction = (suff_lp - baseline_lp)
                                            / (framed_lp - baseline_lp)

All readouts are target log-probability (calibrated across prompts). K is swept
so we can see how mediation accumulates with the number of clamped features.

    python framing_intervention.py
    python framing_intervention.py --k-sweep 1 3 8 --situations framing_situations.json
"""

import argparse
import json
import random
from pathlib import Path

import torch

from poc_minimal import load_model, logits_and_acts, intervened_logits, rank_of
from framing_probe import first_token_id


def target_logprob(logits, tid):
    return float(torch.log_softmax(logits, dim=-1)[tid])


def top_movers(base_last, framed_last, k):
    """Top-k (layer, feat) by |framed - baseline| at the prediction position."""
    delta = (framed_last - base_last).abs().flatten()
    idx = delta.topk(min(k, delta.numel())).indices.tolist()
    return [divmod(i, base_last.shape[1]) for i in idx]


def matched_random(base_last, framed_last, movers, k, rng):
    """k random non-mover features, one per mover layer, |delta| within
    [0.5x, 2x] of the mover it replaces (widening if empty). Same idea as the
    T0 matched-null: control for layer and for how much the feature moved."""
    mset = set(movers)
    delta = (framed_last - base_last).abs()   # [n_layers, d]
    out = []
    for (layer, feat) in movers[:k]:
        tgt = float(delta[layer, feat])
        row = delta[layer]
        for lo, hi in [(0.5, 2.0), (0.25, 4.0), (0.0, float("inf"))]:
            cand = torch.nonzero((row > max(lo * tgt, 1e-6))
                                 & (row < hi * tgt)).flatten().tolist()
            cand = [c for c in cand if (layer, c) not in mset
                    and (layer, c) not in set(out)]
            if cand:
                out.append((layer, rng.choice(cand)))
                break
    return out


def clamp_to(features, pos, source_last):
    """Interventions setting each feature at `pos` to its value in source_last."""
    return [(l, pos, f, float(source_last[l, f])) for (l, f) in features]


def run_situation(model, sit, args, rng):
    target = sit.get("target")
    if not target:
        return None  # mediation needs a measurable target
    tid = first_token_id(model, target)
    framings = sit["framings"]

    base_logits, base_acts = logits_and_acts(model, framings["baseline"])
    base_last = base_acts[:, -1, :].float()
    base_pos = base_acts.shape[1] - 1
    base_lp = target_logprob(base_logits, tid)

    out = {"id": sit["id"], "target": target, "baseline_logprob": base_lp,
           "baseline_rank": rank_of(base_logits, tid), "framings": {}}
    print(f"\n=== {sit['id']} target={target!r} "
          f"baseline lp={base_lp:.3f} rank={out['baseline_rank']} ===")

    for name, prompt in framings.items():
        if name == "baseline":
            continue
        fr_logits, fr_acts = logits_and_acts(model, prompt)
        fr_last = fr_acts[:, -1, :].float()
        fr_pos = fr_acts.shape[1] - 1
        fr_lp = target_logprob(fr_logits, tid)
        denom_nec = base_lp - fr_lp        # framing effect size (necessity)
        denom_suf = fr_lp - base_lp

        movers_full = top_movers(base_last, fr_last, max(args.k_sweep))
        rec = {"framed_logprob": fr_lp, "framed_rank": rank_of(fr_logits, tid),
               "framing_effect_logprob": denom_nec, "by_k": {}}
        print(f"[{name}] framed lp={fr_lp:.3f} rank={rec['framed_rank']} "
              f"(effect {denom_nec:+.3f})")

        for k in args.k_sweep:
            movers = movers_full[:k]
            ctrl = matched_random(base_last, fr_last, movers_full, k, rng)

            # necessity: restore movers to baseline on the FRAMED prompt
            nec_logits = intervened_logits(
                model, prompt, clamp_to(movers, fr_pos, base_last))
            nec_lp = target_logprob(nec_logits, tid)
            nec_frac = (nec_lp - fr_lp) / denom_nec if abs(denom_nec) > 1e-6 else float("nan")

            # control: restore matched-random to baseline on the FRAMED prompt
            ctrl_logits = intervened_logits(
                model, prompt, clamp_to(ctrl, fr_pos, base_last))
            ctrl_lp = target_logprob(ctrl_logits, tid)
            ctrl_frac = (ctrl_lp - fr_lp) / denom_nec if abs(denom_nec) > 1e-6 else float("nan")

            # sufficiency: inject framed values onto the BASELINE prompt
            suf_logits = intervened_logits(
                model, framings["baseline"], clamp_to(movers, base_pos, fr_last))
            suf_lp = target_logprob(suf_logits, tid)
            suf_frac = (suf_lp - base_lp) / denom_suf if abs(denom_suf) > 1e-6 else float("nan")

            rec["by_k"][str(k)] = {
                "movers": [f"L{l}/{f}" for (l, f) in movers],
                "necessity_logprob": nec_lp, "necessity_fraction": nec_frac,
                "necessity_rank": rank_of(nec_logits, tid),
                "control_fraction": ctrl_frac,
                "sufficiency_fraction": suf_frac,
            }
            print(f"   k={k:>2}: necessity={nec_frac:+.2f} "
                  f"(rank {rec['framed_rank']}->{rank_of(nec_logits, tid)}) | "
                  f"control={ctrl_frac:+.2f} | sufficiency={suf_frac:+.2f}")
        out["framings"][name] = rec
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--situations", default="framing_situations.json")
    p.add_argument("--k-sweep", nargs="+", type=int, default=[1, 3, 8])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    situations = json.loads(Path(args.situations).read_text())["situations"]
    model = load_model()
    rng = random.Random(args.seed)

    results = [r for sit in situations
               if (r := run_situation(model, sit, args, rng)) is not None]

    out_dir = Path("./out")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "framing_intervention.json").write_text(json.dumps(results, indent=2))
    print(f"\n[intervention] written to {out_dir / 'framing_intervention.json'}")


if __name__ == "__main__":
    main()
