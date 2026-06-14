# Follow-ups to CHAT_FORMAT_FINDINGS (base gemma-2-2b, HookedTransformer):
#  (a) distance, fixed: re-test whether neutral filler between the salience
#      distractor and the readout attenuates the LATENT logit pull -- using a
#      filler with NO salience words (fixes the contaminated P2 control).
#  (b) is the QA-format latent pull still attention-copy-mediated? Measure the
#      salience effect and the all-heads anchor-knockout necessity (a la sec 3.7)
#      in the bare fragment vs the QA scaffold. necessity ~1 in QA => circuit
#      active at the readout but overridden downstream in free generation;
#      effect ~0 in QA => the QA scaffold disengages the copy.
import json
from pathlib import Path
import torch
from transformer_lens import HookedTransformer

PAIRS = [("Australia", "Sydney", "Canberra"), ("Texas", "Houston", "Austin"),
         ("Canada", "Toronto", "Ottawa"), ("Switzerland", "Zurich", "Bern"),
         ("Morocco", "Casablanca", "Rabat")]
FILLER = " This is a frequently discussed matter."

print("[load] HookedTransformer gemma-2-2b", flush=True)
model = HookedTransformer.from_pretrained_no_processing(
    "google/gemma-2-2b", dtype=torch.bfloat16)
model.eval()
tok = model.tokenizer
print("[load] done", flush=True)

first = lambda s: tok.encode(s, add_special_tokens=False)[0]
pat_filter = lambda n: n.endswith("hook_pattern")


def score(logits, cid, aid):
    lp = torch.log_softmax(logits.float(), -1)
    return float(lp[cid] - lp[aid])          # capital - distractor margin


def last_logits(ids, hooks=None):
    with torch.no_grad():
        if hooks:
            return model.run_with_hooks(ids, fwd_hooks=hooks)[0, -1]
        return model(ids)[0, -1]


def ko_hook(positions):
    def hook(pattern, hook):                  # [b, head, q, k]
        pattern[:, :, :, positions] = 0.0
        return pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-9)
    return hook


def anchor_pos(ids_list, anchor):
    aset = set(model.to_tokens(anchor, prepend_bos=False)[0].tolist())
    return [i for i, t in enumerate(ids_list) if t in aset and i > 0]


# ---------------- (a) distance with NEUTRAL filler ----------------
dist = []
for region, anchor, cap in PAIRS:
    cid, aid = first(" " + cap), first(" " + anchor)
    salience = f"{anchor} is the most famous city in {region}. "
    row = {"pair": f"{region}->{cap}", "lengths": {}}
    for lname, k in [("short", 0), ("long", 8)]:
        stem = FILLER * k + f"The capital of {region} is the city of"
        sc = {fr: score(last_logits(model.to_tokens(frame + stem)), cid, aid)
              for fr, frame in [("neutral", ""), ("salience", salience)]}
        row["lengths"][lname] = {"neutral": sc["neutral"], "salience": sc["salience"],
                                 "effect": sc["neutral"] - sc["salience"]}
    dist.append(row)
    print(f"[dist] {region:<12} short eff={row['lengths']['short']['effect']:+.2f} "
          f"long(+8 filler) eff={row['lengths']['long']['effect']:+.2f}")

# ---------------- (b) bare fragment vs QA scaffold + knockout ----------------
def bare(frame, r): return f"{frame}The capital of {r} is the city of"
def qa(frame, r):
    return (f"{frame}Question: What is the capital of {r}?\n"
            f"Answer: The capital of {r} is the city of")

mech = []
for region, anchor, cap in PAIRS:
    cid, aid = first(" " + cap), first(" " + anchor)
    salience = f"{anchor} is the most famous city in {region}. "
    row = {"pair": f"{region}->{cap}", "contexts": {}}
    for ctx, build in [("bare", bare), ("qa", qa)]:
        n_sc = score(last_logits(model.to_tokens(build("", region))), cid, aid)
        s_ids = model.to_tokens(build(salience, region))
        s_sc = score(last_logits(s_ids), cid, aid)
        eff = n_sc - s_sc
        apos = anchor_pos(s_ids[0].tolist(), anchor)
        nec = None
        if apos and abs(eff) > 0.5:
            ko_sc = score(last_logits(s_ids, hooks=[(pat_filter, ko_hook(apos))]), cid, aid)
            nec = (ko_sc - s_sc) / eff
        row["contexts"][ctx] = {"neutral": n_sc, "salience": s_sc, "effect": eff,
                                "anchor_pos": apos, "necessity": nec}
        ns = f"{nec:+.2f}" if nec is not None else "n/a"
        print(f"[mech] {region:<12} {ctx:>4}: effect={eff:+.2f} knockout_nec={ns}")
    mech.append(row)

import statistics
print("\n[dist] mean effect: short=%.2f  long=%.2f"
      % (statistics.mean(r["lengths"]["short"]["effect"] for r in dist),
         statistics.mean(r["lengths"]["long"]["effect"] for r in dist)))
print("[mech] mean effect: bare=%.2f  qa=%.2f"
      % (statistics.mean(r["contexts"]["bare"]["effect"] for r in mech),
         statistics.mean(r["contexts"]["qa"]["effect"] for r in mech)))
necs = [r["contexts"]["qa"]["necessity"] for r in mech
        if r["contexts"]["qa"]["necessity"] is not None]
if necs:
    print("[mech] mean QA knockout necessity=%.2f (n=%d)"
          % (statistics.mean(necs), len(necs)))

Path("out").mkdir(exist_ok=True)
Path("out/base_attn_qa.json").write_text(json.dumps(
    {"distance": dist, "mechanism": mech}, indent=2))
print("[done] wrote out/base_attn_qa.json")
