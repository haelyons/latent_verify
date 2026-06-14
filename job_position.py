# Experiment (a) (follow-up to sec 3.10): does the distractor's POSITION / its
# distance from the answer slot change the copy? Sweep neutral filler between the
# salience distractor and the question, plus an "adjacent to the answer" variant.
# For each: framing effect, capital rank, the reader head L18.H5's
# prediction-position attention to the anchor, the anchor's token distance from
# the prediction position, and the all-heads anchor-knockout necessity (sec 3.7).
# Runs inside worker.py (model, torch in scope).
import json
from pathlib import Path
import torch

tok = model.tokenizer
READER = (18, 5)
READER_NAME = f"blocks.{READER[0]}.attn.hook_pattern"
FILLER = " This is a frequently discussed matter."
PAIRS = [("Australia", "Sydney", "Canberra"), ("Texas", "Houston", "Austin")]

first_tok = lambda s: tok.encode(s, add_special_tokens=False)[0]


def readout(logits, tid):
    return (float(torch.log_softmax(logits.float(), -1)[tid]),
            int((logits > logits[tid]).sum()))


pat_filter = lambda name: name.endswith("hook_pattern")
reader_filter = lambda name: name == READER_NAME


def ko_hook(positions):
    def hook(pattern, hook):
        pattern[:, :, :, positions] = 0.0
        return pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-9)
    return hook


results = []
for region, anchor, cap in PAIRS:
    tid = first_tok(" " + cap)
    anchor_ids = set(model.to_tokens(anchor, prepend_bos=False)[0].tolist())
    baseline = f"The capital of {region} is the city of"
    with torch.no_grad():
        b_lp, b_rank = readout(model(model.to_tokens(baseline))[0, -1], tid)

    variants = {}
    for k in [0, 1, 2, 4, 8]:                          # distance sweep
        variants[f"front+{k}filler"] = (
            f"{anchor} is the most famous city in {region}.{FILLER * k} "
            f"The capital of {region} is the city of")
    variants["adjacent_answer"] = (                    # distractor next to slot
        f"The capital of {region}, though {anchor} is its most famous city, "
        f"is the city of")

    rec = {"pair": f"{region}->{cap}", "baseline_lp": b_lp,
           "baseline_rank": b_rank, "variants": {}}
    print(f"\n=== {region}->{cap} (baseline lp={b_lp:.2f} rank={b_rank}) ===")
    for vname, prompt in variants.items():
        toks = model.to_tokens(prompt)
        apos = [i for i, t in enumerate(toks[0].tolist())
                if t in anchor_ids and i > 0]
        with torch.no_grad():
            f_lp, f_rank = readout(model(toks)[0, -1], tid)
            _, cache = model.run_with_cache(toks, names_filter=reader_filter)
        effect = b_lp - f_lp
        row = cache[READER_NAME][0, READER[1]][-1]
        reader_attn = float(row[apos].sum()) if apos else 0.0
        dist = (toks.shape[1] - 1 - max(apos)) if apos else None
        nec = None
        if apos and effect > 0.5:
            with torch.no_grad():
                ko = model.run_with_hooks(
                    toks, fwd_hooks=[(pat_filter, ko_hook(apos))])[0, -1]
            nec = (readout(ko, tid)[0] - f_lp) / effect
        rec["variants"][vname] = {
            "effect": effect, "framed_rank": f_rank,
            "reader_attn_to_anchor": reader_attn, "anchor_dist": dist,
            "necessity": nec}
        ns = f"{nec:+.2f}" if nec is not None else " n/a"
        print(f"[pos] {vname:>16} dist={str(dist):>4} eff={effect:+5.2f} "
              f"rank {b_rank}->{f_rank:<4} L18.H5->anchor={reader_attn:.2f} "
              f"knockout_nec={ns}")
    results.append(rec)

Path("out").mkdir(exist_ok=True)
Path("out/framing_position.json").write_text(json.dumps(results, indent=2))
print("\n[pos] written out/framing_position.json")
