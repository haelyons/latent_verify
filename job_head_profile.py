# Step 1a (FRAMING_NOTES sec 5.1): characterize the principal copy-heads
# L0.H2 and L18.H5 (and runners-up). Two readouts:
#  (i)  GENERAL attention profile on a repeated-random sequence -- prev-token,
#       BOS-sink, self, and an induction score (does the head attend to the
#       token *after* a key's previous occurrence?).
#  (ii) PREDICTION-position attention on the framed Australia prompt -- does the
#       head read the "Sydney" anchor at the moment of prediction?
# Runs inside worker.py (model, torch in scope).
import json
from pathlib import Path
import torch

tok = model.tokenizer
n_heads = model.cfg.n_heads
PROFILE_LAYERS = [0, 1, 3, 4, 7, 18]
HEADS_OF_INTEREST = [(0, 2), (18, 5), (0, 3), (7, 1), (1, 0)]

pat_names = {f"blocks.{l}.attn.hook_pattern" for l in PROFILE_LAYERS}
nf = lambda name: name in pat_names

# ---- (i) general profile via a repeated random sequence (induction test) ----
L = 25
torch.manual_seed(0)
vocab = model.cfg.d_vocab
rand = torch.randint(1000, min(vocab, 50000), (L,))
bos = torch.tensor([[tok.bos_token_id]])
seq = torch.cat([bos, rand.unsqueeze(0), rand.unsqueeze(0)], dim=1)  # [1, 2L+1]
with torch.no_grad():
    _, gcache = model.run_with_cache(seq, names_filter=nf)


def general_scores(l, h):
    p = gcache[f"blocks.{l}.attn.hook_pattern"][0, h]      # [q, k]
    Q = p.shape[0]
    qs = range(1, Q)                                       # skip BOS query
    prev = sum(float(p[q, q - 1]) for q in qs) / (Q - 1)
    sink = sum(float(p[q, 0]) for q in qs) / (Q - 1)
    diag = sum(float(p[q, q]) for q in qs) / (Q - 1)
    # seq = [BOS, rand(0..L-1), rand(0..L-1)]; second-occurrence query at index
    # 1+L+j attends, if inductive, to 1+j+1 (token after rand[j]'s 1st occ).
    ind = sum(float(p[1 + L + j, 1 + j + 1]) for j in range(L - 1)) / (L - 1)
    return {"prev_token": prev, "bos_sink": sink, "self": diag, "induction": ind}


# ---- (ii) prediction-position attention on the framed Australia prompt -------
framed = ("Sydney is the most famous city in Australia. "
          "The capital of Australia is the city of")
ftoks = model.to_tokens(framed)
fdec = [tok.decode([t]) for t in ftoks[0].tolist()]
syd_pos = [i for i, d in enumerate(fdec) if "sydney" in d.lower()]
with torch.no_grad():
    _, fcache = model.run_with_cache(ftoks, names_filter=nf)


def pred_attn(l, h):
    row = fcache[f"blocks.{l}.attn.hook_pattern"][0, h][-1]   # last (pred) query
    to_syd = float(row[syd_pos].sum()) if syd_pos else 0.0
    top = torch.topk(row, min(4, row.shape[0]))
    tops = [{"pos": int(i), "tok": fdec[int(i)], "w": float(w)}
            for w, i in zip(top.values, top.indices)]
    return {"attn_to_sydney": to_syd, "top_keys": tops}


profile = {}
print(f"[profile] Sydney at positions {syd_pos} "
      f"({[fdec[i] for i in syd_pos]}); prediction token {fdec[-1]!r}")
for (l, h) in HEADS_OF_INTEREST:
    g, fp = general_scores(l, h), pred_attn(l, h)
    profile[f"L{l}.H{h}"] = {"general": g, "framed_pred": fp}
    print(f"L{l}.H{h}: prev={g['prev_token']:.2f} sink={g['bos_sink']:.2f} "
          f"self={g['self']:.2f} induction={g['induction']:.2f} | "
          f"->Sydney(pred)={fp['attn_to_sydney']:.2f} "
          f"top={[(t['tok'], round(t['w'], 2)) for t in fp['top_keys']]}")

Path("out").mkdir(exist_ok=True)
Path("out/framing_head_profile.json").write_text(json.dumps(
    {"sydney_positions": syd_pos, "framed_tokens": fdec, "heads": profile}, indent=2))
print("[profile] written out/framing_head_profile.json")
