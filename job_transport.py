# Transport test: does the attention-copy mechanism generalize?
# For each case: confirm the false anchor flips the capital, then sever attention
# to the anchor token and measure how much of the flip reverts. If "attention
# copies the anchored city" is the mechanism, knockout necessity ~1 everywhere.
import json
from pathlib import Path
import torch

tok = model.tokenizer

def first_tok(s):
    return tok.encode(s, add_special_tokens=False)[0]

def readout(logits, tid):
    lp = float(torch.log_softmax(logits.float(), -1)[tid].detach())
    return lp, int((logits > logits[tid]).sum())

# (region, anchor city = largest, target capital). All are largest-city != capital.
PAIRS = [
    ("Australia", "Sydney", "Canberra"),
    ("Texas", "Houston", "Austin"),
    ("Canada", "Toronto", "Ottawa"),
    ("Turkey", "Istanbul", "Ankara"),
    ("Switzerland", "Zurich", "Bern"),
    ("Morocco", "Casablanca", "Rabat"),
]

cases = []
for region, anchor, cap in PAIRS:
    cases.append({
        "name": f"{region}:{anchor}->{cap}", "anchor": anchor, "target": cap,
        "framed": f"{anchor} is the largest city in {region}. "
                  f"The capital of {region} is the city of",
        "baseline": f"The capital of {region} is the city of"})
# two paraphrase variants of the Australia case (wording / anchor placement)
cases += [
    {"name": "AU-para:everyone", "anchor": "Sydney", "target": "Canberra",
     "framed": "Everyone always talks about Sydney. "
               "The capital of Australia is the city of",
     "baseline": "The capital of Australia is the city of"},
    {"name": "AU-para:confuse", "anchor": "Sydney", "target": "Canberra",
     "framed": "The capital of Australia, often confused with Sydney, "
               "is the city of",
     "baseline": "The capital of Australia is the city of"},
]

pat_filter = lambda name: name.endswith("hook_pattern")
def ko_hook(positions):
    def hook(pattern, hook):
        pattern[:, :, :, positions] = 0.0
        return pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-9)
    return hook

results = []
with torch.no_grad():
    for c in cases:
        tid = first_tok(" " + c["target"])
        did = first_tok(" " + c["anchor"])
        ftoks = model.to_tokens(c["framed"])
        ids = ftoks[0].tolist()
        decoded = [tok.decode([t]) for t in ids]
        anchor_pos = [i for i, d in enumerate(decoded)
                      if c["anchor"].lower() in d.lower()]

        b_lp, b_rank = readout(model(model.to_tokens(c["baseline"]))[0, -1], tid)
        fr_logits = model(ftoks)[0, -1]
        f_lp, f_rank = readout(fr_logits, tid)
        top1 = tok.decode([int(fr_logits.argmax())])
        effect = b_lp - f_lp

        ko_lp = ko_rank = frac = None
        if anchor_pos and effect > 0.5:
            ko_logits = model.run_with_hooks(
                ftoks, fwd_hooks=[(pat_filter, ko_hook(anchor_pos))])[0, -1]
            ko_lp, ko_rank = readout(ko_logits, tid)
            frac = (ko_lp - f_lp) / effect

        flipped = f_rank > 0
        results.append({
            "name": c["name"], "anchor_pos": anchor_pos,
            "baseline_rank": b_rank, "framed_rank": f_rank, "framed_top1": top1,
            "effect": effect, "flipped": flipped,
            "knockout_rank": ko_rank, "necessity": frac})
        fr = f"{frac:+.2f}" if frac is not None else " n/a"
        print(f"[transport] {c['name']:<22} flip={'Y' if flipped else 'N'} "
              f"(top1={top1!r:>11}, target rank {b_rank}->{f_rank}) "
              f"effect={effect:+5.2f} | knockout necessity={fr} "
              f"(rank ->{ko_rank})")

Path("out").mkdir(exist_ok=True)
Path("out/framing_transport.json").write_text(json.dumps(results, indent=2))
flips = [r for r in results if r["flipped"]]
mediated = [r for r in flips if r["necessity"] and r["necessity"] > 0.7]
print(f"[transport] {len(flips)}/{len(results)} cases flipped; "
      f"{len(mediated)}/{len(flips)} of flips >0.7 reverted by anchor knockout")
print("[transport] written out/framing_transport.json")
