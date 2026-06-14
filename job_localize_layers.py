# Localize the anchor->prediction copy in DEPTH: knock out attention to the
# "Sydney" token at one layer at a time (all heads, all queries) and measure how
# much of the flip that single layer's contribution accounts for. Finds which
# layers carry the copy, vs the all-layers sledgehammer (necessity ~1.0).
import json
from pathlib import Path
import torch

tok = model.tokenizer
def canberra(logits):
    tid = tok.encode(" Canberra", add_special_tokens=False)[0]
    return float(torch.log_softmax(logits.float(), -1)[tid].detach()), \
        int((logits > logits[tid]).sum())

framed = ("Sydney is the most famous city in Australia. "
          "The capital of Australia is the city of")
baseline = "The capital of Australia is the city of"
ftoks = model.to_tokens(framed)
syd = [i for i, t in enumerate(ftoks[0].tolist())
       if "sydney" in tok.decode([t]).lower()]
n_layers = model.cfg.n_layers
print(f"[loc] n_layers={n_layers} n_heads={model.cfg.n_heads} sydney_pos={syd}")

with torch.no_grad():
    b_lp, _ = canberra(model(model.to_tokens(baseline))[0, -1])
    f_lp, f_rank = canberra(model(ftoks)[0, -1])
effect = b_lp - f_lp
print(f"[loc] baseline lp={b_lp:.2f} framed lp={f_lp:.2f} rank={f_rank} "
      f"effect={effect:+.2f}")

def ko_hook(pattern, hook):
    pattern[:, :, :, syd] = 0.0
    return pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-9)

rows = []
with torch.no_grad():
    for l in range(n_layers):
        name = f"blocks.{l}.attn.hook_pattern"
        logits = model.run_with_hooks(ftoks, fwd_hooks=[(name, ko_hook)])[0, -1]
        lp, rank = canberra(logits)
        frac = (lp - f_lp) / effect
        rows.append({"layer": l, "necessity": frac, "rank": rank})

rows_sorted = sorted(rows, key=lambda r: -r["necessity"])
print("[loc] per-layer necessity (top 10 by |effect|):")
for r in sorted(rows, key=lambda r: -abs(r["necessity"]))[:10]:
    print(f"[loc]   layer {r['layer']:>2}: necessity={r['necessity']:+.3f} "
          f"(rank ->{r['rank']})")

Path("out").mkdir(exist_ok=True)
Path("out/framing_localize_layers.json").write_text(json.dumps(
    {"effect": effect, "rows": rows}, indent=2))
print("[loc] written out/framing_localize_layers.json")
