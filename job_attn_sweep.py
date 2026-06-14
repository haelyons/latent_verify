# Per-token attention-knockout sweep: sever attention to each framing token in
# turn and measure how much of the Canberra->Sydney flip reverts. Clean
# specificity picture -- which tokens carry the framing via attention.
import json
from pathlib import Path
import torch

tok = model.tokenizer

def canberra(logits):
    tid = tok.encode(" Canberra", add_special_tokens=False)[0]
    return (float(torch.log_softmax(logits.float(), -1)[tid].detach()),
            int((logits > logits[tid]).sum()))

framed_prompt = ("Sydney is the most famous city in Australia. "
                 "The capital of Australia is the city of")
baseline_prompt = "The capital of Australia is the city of"
ftoks = model.to_tokens(framed_prompt)
ids = ftoks[0].tolist()
decoded = [tok.decode([t]) for t in ids]

with torch.no_grad():
    b_lp, _ = canberra(model(model.to_tokens(baseline_prompt))[0, -1])
    f_lp, f_rank = canberra(model(ftoks)[0, -1])
effect = b_lp - f_lp
print(f"[sweep] baseline lp={b_lp:.3f} | framed lp={f_lp:.3f} rank={f_rank} | "
      f"effect={effect:+.3f}")

pat_filter = lambda name: name.endswith("hook_pattern")
def ko_hook(positions):
    def hook(pattern, hook):
        pattern[:, :, :, positions] = 0.0
        return pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-9)
    return hook

rows = []
with torch.no_grad():
    for pos in range(1, len(ids)):  # skip BOS
        logits = model.run_with_hooks(
            ftoks, fwd_hooks=[(pat_filter, ko_hook([pos]))])[0, -1]
        lp, rank = canberra(logits)
        frac = (lp - f_lp) / effect
        rows.append({"pos": pos, "token": decoded[pos], "necessity": frac,
                     "rank": rank})
        print(f"[sweep] pos {pos:>2} {decoded[pos]!r:>12}: "
              f"necessity={frac:+.2f} (rank {rank})")

Path("out").mkdir(exist_ok=True)
Path("out/framing_attn_sweep.json").write_text(json.dumps(
    {"baseline_lp": b_lp, "framed_lp": f_lp, "effect": effect, "rows": rows},
    indent=2))
print("[sweep] written out/framing_attn_sweep.json")
