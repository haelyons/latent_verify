"""Direct-logit-attribution feature selection + mediation retest.

The §3.5 result: clamping the top *activation-delta* movers reverts ~none of
the Canberra->Sydney flip -- biggest mover != causal driver. This script picks
candidates a different way and reruns the same causal test.

For a situation with a `target` and a `distractor` (distinct first tokens), the
framing's *direct* contribution to the target-vs-distractor logit gap, from a
transcoder feature f at the prediction position, is approximately

    score_f = (act_framed_f - act_baseline_f) . (W_dec[f] . (W_U[:,distr] - W_U[:,targ]))

i.e. the framing-induced change in the feature's activation times how much its
decoder vector points along the distractor-minus-target unembedding direction.
The final-LayerNorm mean cancels in the *token difference*, and its positive
scale is common to all features, so this ranking is LN-robust. Features with
large positive score pushed the answer from target toward distractor through
the readout-position MLP -- the candidate mediators activation-delta ranking
misses.

Then: clamp the top-K DLA features back to baseline on the framed prompt
(necessity), with a matched-random control, exactly as framing_intervention.

    python framing_dla.py
"""

import argparse
import json
import random
from pathlib import Path

import torch

from poc_minimal import load_model, logits_and_acts, intervened_logits, rank_of
from framing_probe import first_token_id
from framing_intervention import (target_logprob, clamp_to, matched_random,
                                  top_movers, MIN_EFFECT)


def dla_direction(model, target_id, distractor_id):
    """W_U[:,distractor] - W_U[:,target], the 'suppress target' logit direction."""
    W_U = model.W_U  # [d_model, d_vocab]
    return (W_U[:, distractor_id].float() - W_U[:, target_id].float())


def dla_scores(model, base_last, framed_last, direction):
    """Per-feature score [n_layers, d_transcoder]: framing-induced activation
    change times decoder alignment with the suppress-target direction."""
    delta = (framed_last - base_last)                       # [n_layers, d]
    n_layers, d_trans = delta.shape
    scores = torch.zeros_like(delta)
    for layer in range(n_layers):
        W_dec = model.transcoders[layer].W_dec.float()      # [d_transcoder, d_model]
        proj = W_dec @ direction                            # [d_transcoder]
        scores[layer] = delta[layer] * proj
    return scores


def top_by_score(scores, k):
    """Top-k (layer, feat) by score descending (most push toward distractor)."""
    flat = scores.flatten()
    idx = flat.topk(min(k, flat.numel())).indices.tolist()
    return [divmod(i, scores.shape[1]) for i in idx]


def run_situation(model, sit, args, rng):
    target, distractor = sit.get("target"), sit.get("distractor")
    if not (target and distractor):
        return None
    tid, did = first_token_id(model, target), first_token_id(model, distractor)
    framings = sit["framings"]

    base_logits, base_acts = logits_and_acts(model, framings["baseline"])
    base_last = base_acts[:, -1, :].float()
    base_pos = base_acts.shape[1] - 1
    base_lp = target_logprob(base_logits, tid)
    direction = dla_direction(model, tid, did)

    out = {"id": sit["id"], "target": target, "distractor": distractor,
           "baseline_logprob": base_lp, "framings": {}}
    print(f"\n=== {sit['id']} target={target!r} distractor={distractor!r} "
          f"baseline lp={base_lp:.3f} ===")

    for name, prompt in framings.items():
        if name == "baseline":
            continue
        fr_logits, fr_acts = logits_and_acts(model, prompt)
        fr_last = fr_acts[:, -1, :].float()
        fr_pos = fr_acts.shape[1] - 1
        fr_lp = target_logprob(fr_logits, tid)
        denom = base_lp - fr_lp
        if abs(denom) < MIN_EFFECT:
            print(f"[{name}] effect {denom:+.3f} < MIN_EFFECT; skipping")
            continue

        scores = dla_scores(model, base_last, fr_last, direction)
        dla_top = top_by_score(scores, max(args.k_sweep))
        actd_top = top_movers(base_last, fr_last, max(args.k_sweep))

        # how different is DLA selection from activation-delta selection?
        overlap = len(set(dla_top) & set(actd_top))
        rec = {"framed_logprob": fr_lp, "framing_effect_logprob": denom,
               "dla_top": [f"L{l}/{f}" for (l, f) in dla_top],
               "actd_top": [f"L{l}/{f}" for (l, f) in actd_top],
               "overlap_with_actd": overlap, "by_k": {}}
        print(f"[{name}] framed lp={fr_lp:.3f} (effect {denom:+.3f}) | "
              f"DLA top != actd (overlap {overlap}/{len(dla_top)})")
        # show the top DLA feature's score components for intuition
        (l0, f0) = dla_top[0]
        print(f"   top DLA L{l0}/{f0}: dact={float(fr_last[l0,f0]-base_last[l0,f0]):+.2f}"
              f" score={float(scores[l0,f0]):+.3f}")

        for k in args.k_sweep:
            movers = dla_top[:k]
            ctrl = matched_random(base_last, fr_last, dla_top, k, rng)
            nec_logits = intervened_logits(model, prompt,
                                           clamp_to(movers, fr_pos, base_last))
            nec_lp = target_logprob(nec_logits, tid)
            nec_frac = (nec_lp - fr_lp) / denom
            ctrl_logits = intervened_logits(model, prompt,
                                            clamp_to(ctrl, fr_pos, base_last))
            ctrl_frac = (target_logprob(ctrl_logits, tid) - fr_lp) / denom
            rec["by_k"][str(k)] = {
                "movers": [f"L{l}/{f}" for (l, f) in movers],
                "necessity_fraction": nec_frac, "control_fraction": ctrl_frac,
                "necessity_rank": rank_of(nec_logits, tid),
            }
            print(f"   k={k:>2}: necessity={nec_frac:+.2f} "
                  f"(rank ->{rank_of(nec_logits, tid)}) | control={ctrl_frac:+.2f}")
        out["framings"][name] = rec
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--situations", default="framing_situations.json")
    p.add_argument("--k-sweep", nargs="+", type=int, default=[1, 3, 8, 24])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    situations = json.loads(Path(args.situations).read_text())["situations"]
    model = load_model()
    rng = random.Random(args.seed)
    results = [r for sit in situations
               if (r := run_situation(model, sit, args, rng)) is not None]

    out_dir = Path("./out")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "framing_dla.json").write_text(json.dumps(results, indent=2))
    print(f"\n[dla] written to {out_dir / 'framing_dla.json'}")


if __name__ == "__main__":
    main()
