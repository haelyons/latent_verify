# Step 1b (FRAMING_NOTES sec 5.1): does the SAME head circuit carry the copy
# across fact pairs, or is it pair-specific? For every pair that genuinely flips
# under salience framing, sweep (layer in TOP_LAYERS) x (head) knockouts of
# attention to the anchor city and measure per-head necessity. Then compare the
# top heads across pairs and track the Australia principals L0.H2 / L18.H5
# specifically. Runs inside worker.py (model, torch in scope).
import json
import time
from pathlib import Path
import torch

tok = model.tokenizer
n_heads = model.cfg.n_heads
TOP_LAYERS = [0, 1, 3, 4, 7, 18]

PAIRS = [
    ("Australia", "Sydney", "Canberra"),
    ("Texas", "Houston", "Austin"),
    ("Canada", "Toronto", "Ottawa"),
    ("Switzerland", "Zurich", "Bern"),
    ("Morocco", "Casablanca", "Rabat"),
]

first_tok = lambda s: tok.encode(s, add_special_tokens=False)[0]


def readout(logits, tid):
    return (float(torch.log_softmax(logits.float(), -1)[tid]),
            int((logits > logits[tid]).sum()))


def head_hook(head, positions):
    def hook(pattern, hook):                # [batch, head, q, k]
        pattern[:, head, :, positions] = 0.0
        denom = pattern[:, head].sum(-1, keepdim=True).clamp_min(1e-9)
        pattern[:, head] = pattern[:, head] / denom
        return pattern
    return hook


out_pairs = []
t0 = time.time()
with torch.no_grad():
    for region, anchor, cap in PAIRS:
        framed = (f"{anchor} is the most famous city in {region}. "
                  f"The capital of {region} is the city of")
        baseline = f"The capital of {region} is the city of"
        tid = first_tok(" " + cap)
        ftoks = model.to_tokens(framed)
        seq = ftoks[0].tolist()
        anchor_ids = set(model.to_tokens(anchor, prepend_bos=False)[0].tolist())
        anchor_pos = [i for i, t in enumerate(seq) if t in anchor_ids and i > 0]

        b_lp, b_rank = readout(model(model.to_tokens(baseline))[0, -1], tid)
        f_lp, f_rank = readout(model(ftoks)[0, -1], tid)
        effect = b_lp - f_lp
        flip = (b_rank == 0 and f_rank > 0)
        rec = {"pair": f"{region}:{anchor}->{cap}", "effect": effect,
               "flip": flip, "anchor_pos": anchor_pos,
               "baseline_rank": b_rank, "framed_rank": f_rank, "heads": []}
        if flip and anchor_pos and effect > 0.5:
            for l in TOP_LAYERS:
                name = f"blocks.{l}.attn.hook_pattern"
                for h in range(n_heads):
                    ko = model.run_with_hooks(
                        ftoks, fwd_hooks=[(name, head_hook(h, anchor_pos))])[0, -1]
                    lp, rank = readout(ko, tid)
                    rec["heads"].append({"layer": l, "head": h,
                                         "necessity": (lp - f_lp) / effect,
                                         "rank": rank})
        out_pairs.append(rec)
        top = sorted(rec["heads"], key=lambda r: -r["necessity"])[:5]
        print(f"[ht] {region:<12} eff={effect:+.2f} flip={flip} top: " +
              ", ".join(f"L{r['layer']}.H{r['head']}({r['necessity']:+.2f})"
                        for r in top))
print(f"[ht] elapsed {time.time() - t0:.0f}s")


def nec(rec, l, h):
    for r in rec["heads"]:
        if r["layer"] == l and r["head"] == h:
            return r["necessity"]
    return None


print("\n[ht] principal-head necessity across pairs:")
for (l, h) in [(0, 2), (18, 5)]:
    line = ", ".join(f"{rec['pair'].split(':')[0]}={nec(rec, l, h):+.2f}"
                     for rec in out_pairs if rec["heads"])
    print(f"  L{l}.H{h}: {line}")

Path("out").mkdir(exist_ok=True)
Path("out/framing_head_transport.json").write_text(json.dumps(out_pairs, indent=2))
print("[ht] written out/framing_head_transport.json")
