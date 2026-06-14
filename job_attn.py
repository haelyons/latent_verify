# Attention-copy test: is the unmediated ~half of the flip carried by attention
# TO the "Sydney" token? Sever attention to that key at every layer (renormalize)
# and measure how much of the Canberra->Sydney flip reverts. Neutral-token
# knockout = specificity control. Runs inside worker.py (model in scope).
import json
import torch

tok = model.tokenizer

def readout(logits, s):
    tid = tok.encode(s, add_special_tokens=False)[0]
    lp = float(torch.log_softmax(logits.float(), -1)[tid])
    rank = int((logits > logits[tid]).sum())
    return lp, rank, tid

baseline_prompt = "The capital of Australia is the city of"
framed_prompt = ("Sydney is the most famous city in Australia. "
                 "The capital of Australia is the city of")

ftoks = model.to_tokens(framed_prompt)
ids = ftoks[0].tolist()
decoded = [tok.decode([t]) for t in ids]
print("[attn] framed tokens:", list(enumerate(decoded)))

def positions_of(substr):
    return [i for i, d in enumerate(decoded) if substr.lower() in d.lower()]

syd_pos = positions_of("sydney")
ctrl_pos = positions_of("famous") or positions_of("most")
print(f"[attn] Sydney positions={syd_pos} ({[decoded[i] for i in syd_pos]}) | "
      f"control positions={ctrl_pos} ({[decoded[i] for i in ctrl_pos]})")

# plain-forward logits (check this matches the get_activations behaviour)
base_logits = model(model.to_tokens(baseline_prompt))[0, -1]
fr_logits = model(ftoks)[0, -1]
ga_logits, _ = logits_and_acts(model, framed_prompt)  # transcoder path, for cross-check
b_can = readout(base_logits, " Canberra")
f_can = readout(fr_logits, " Canberra")
print(f"[attn] baseline Canberra lp={b_can[0]:.3f} rank={b_can[1]} | "
      f"framed (plain fwd) lp={f_can[0]:.3f} rank={f_can[1]} | "
      f"framed (transcoder) lp={readout(ga_logits,' Canberra')[0]:.3f}")
effect = b_can[0] - f_can[0]
print(f"[attn] framing effect (plain fwd) = {effect:+.3f} nats")


def knockout_hook(positions):
    def hook(pattern, hook):       # pattern [batch, head, q_pos, k_pos]
        pattern[:, :, :, positions] = 0.0
        pattern = pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-9)
        return pattern
    return hook

pat_filter = lambda name: name.endswith("hook_pattern")

def run_knockout(positions):
    logits = model.run_with_hooks(
        ftoks, fwd_hooks=[(pat_filter, knockout_hook(positions))])[0, -1]
    return readout(logits, " Canberra")

ko_syd = run_knockout(syd_pos)
ko_ctrl = run_knockout(ctrl_pos) if ctrl_pos else (float("nan"),) * 3
syd_frac = (ko_syd[0] - f_can[0]) / effect if abs(effect) > 0.5 else float("nan")
ctrl_frac = (ko_ctrl[0] - f_can[0]) / effect if abs(effect) > 0.5 else float("nan")

print(f"[attn] knockout(Sydney):  Canberra lp={ko_syd[0]:.3f} rank={ko_syd[1]} "
      f"-> necessity={syd_frac:+.2f}")
print(f"[attn] knockout(control): Canberra lp={ko_ctrl[0]:.3f} rank={ko_ctrl[1]} "
      f"-> necessity={ctrl_frac:+.2f}")

out = {"baseline_canberra_lp": b_can[0], "framed_canberra_lp": f_can[0],
       "framing_effect": effect,
       "sydney_positions": syd_pos, "control_positions": ctrl_pos,
       "knockout_sydney": {"lp": ko_syd[0], "rank": ko_syd[1], "necessity": syd_frac},
       "knockout_control": {"lp": ko_ctrl[0], "rank": ko_ctrl[1], "necessity": ctrl_frac}}
from pathlib import Path
Path("out").mkdir(exist_ok=True)
Path("out/framing_attn.json").write_text(json.dumps(out, indent=2))
print("[attn] written out/framing_attn.json")
