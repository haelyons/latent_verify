# Transport v2: use the framing that actually flips ("most famous city", the
# salience prime that flipped Australia), restrict to pairs whose capital is
# top-1 at baseline, and for every genuine flip test whether knocking out
# attention to the anchor reverts it. Proper flip def: baseline rank 0 AND
# framed rank > 0.
import json
from pathlib import Path
import torch

tok = model.tokenizer
first_tok = lambda s: tok.encode(s, add_special_tokens=False)[0]
def readout(logits, tid):
    lp = float(torch.log_softmax(logits.float(), -1)[tid].detach())
    return lp, int((logits > logits[tid]).sum())

PAIRS = [
    ("Australia", "Sydney", "Canberra"),
    ("Texas", "Houston", "Austin"),
    ("Canada", "Toronto", "Ottawa"),
    ("Switzerland", "Zurich", "Bern"),
    ("Morocco", "Casablanca", "Rabat"),
    ("Brazil", "Rio", "Brasilia"),
    ("New York State", "Buffalo", "Albany"),
]

pat_filter = lambda name: name.endswith("hook_pattern")
def ko_hook(positions):
    def hook(pattern, hook):
        pattern[:, :, :, positions] = 0.0
        return pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-9)
    return hook

results = []
with torch.no_grad():
    for region, anchor, cap in PAIRS:
        framed = (f"{anchor} is the most famous city in {region}. "
                  f"The capital of {region} is the city of")
        baseline = f"The capital of {region} is the city of"
        tid = first_tok(" " + cap)
        ftoks = model.to_tokens(framed)
        decoded = [tok.decode([t]) for t in ftoks[0].tolist()]
        anchor_pos = [i for i, d in enumerate(decoded) if anchor.lower() in d.lower()]

        b_lp, b_rank = readout(model(model.to_tokens(baseline))[0, -1], tid)
        fr_logits = model(ftoks)[0, -1]
        f_lp, f_rank = readout(fr_logits, tid)
        top1 = tok.decode([int(fr_logits.argmax())])
        effect = b_lp - f_lp
        flip = (b_rank == 0 and f_rank > 0)

        ko_rank = frac = None
        if flip and anchor_pos and effect > 0.5:
            ko_logits = model.run_with_hooks(
                ftoks, fwd_hooks=[(pat_filter, ko_hook(anchor_pos))])[0, -1]
            ko_lp, ko_rank = readout(ko_logits, tid)
            frac = (ko_lp - f_lp) / effect

        results.append({"name": f"{region}:{anchor}->{cap}",
                        "baseline_rank": b_rank, "framed_rank": f_rank,
                        "framed_top1": top1, "effect": effect, "flip": flip,
                        "anchor_found": bool(anchor_pos),
                        "knockout_rank": ko_rank, "necessity": frac})
        fr = f"{frac:+.2f}" if frac is not None else " n/a"
        tag = "FLIP" if flip else ("base!=top1" if b_rank else "no-flip")
        print(f"[t2] {region:<15} {anchor:>11}->{cap:<9} {tag:>10} "
              f"top1={top1!r:>11} target {b_rank}->{f_rank} eff={effect:+5.2f} "
              f"| knockout necessity={fr} (rank ->{ko_rank})")

Path("out").mkdir(exist_ok=True)
Path("out/framing_transport2.json").write_text(json.dumps(results, indent=2))
flips = [r for r in results if r["flip"] and r["effect"] > 0.5]
rev = [r for r in flips if r["necessity"] and r["necessity"] > 0.7]
print(f"[t2] {len(flips)} genuine flips; {len(rev)}/{len(flips)} reverted >0.7 "
      f"by anchor-attention knockout "
      f"(necessities: {[round(r['necessity'],2) for r in flips]})")
print("[t2] written out/framing_transport2.json")
