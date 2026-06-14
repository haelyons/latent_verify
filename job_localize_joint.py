# Joint knockout of the top copy-heads (cumulative) to see how concentrated the
# circuit is: knock out attention to "Sydney" for the top-k heads together and
# measure necessity. Heads ordered by the per-head sweep.
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

# (layer, head) ordered by per-head necessity (out/framing_localize_heads.json)
TOP = [(0, 2), (18, 5), (0, 3), (7, 1), (1, 0), (4, 5),
       (3, 0), (3, 4), (0, 0), (1, 6), (4, 6), (7, 5)]

with torch.no_grad():
    b_lp, _ = canberra(model(model.to_tokens(baseline))[0, -1])
    f_lp, f_rank = canberra(model(ftoks)[0, -1])
effect = b_lp - f_lp
print(f"[joint] effect={effect:+.2f}")

def make_layer_hook(heads):
    def hook(pattern, hook):           # [batch, head, q, k]
        for h in heads:
            pattern[:, h, :, syd] = 0.0
            denom = pattern[:, h].sum(-1, keepdim=True).clamp_min(1e-9)
            pattern[:, h] = pattern[:, h] / denom
        return pattern
    return hook

def knockout(head_set):
    by_layer = {}
    for (l, h) in head_set:
        by_layer.setdefault(l, []).append(h)
    hooks = [(f"blocks.{l}.attn.hook_pattern", make_layer_hook(hs))
             for l, hs in by_layer.items()]
    return model.run_with_hooks(ftoks, fwd_hooks=hooks)[0, -1]

rows = []
with torch.no_grad():
    for k in [1, 2, 3, 5, 8, 12]:
        lp, rank = canberra(knockout(TOP[:k]))
        frac = (lp - f_lp) / effect
        rows.append({"k": k, "heads": [f"L{l}.H{h}" for l, h in TOP[:k]],
                     "necessity": frac, "rank": rank})
        print(f"[joint] top-{k:>2} heads: necessity={frac:+.3f} (rank ->{rank})")

Path("out").mkdir(exist_ok=True)
Path("out/framing_localize_joint.json").write_text(json.dumps(
    {"effect": effect, "rows": rows}, indent=2))
print("[joint] written out/framing_localize_joint.json")
