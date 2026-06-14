# Experiment (b) (follow-up to sec 3.10): does an explicit "ignore irrelevant
# context" instruction reduce the copy-driven flip? Test several instruction
# phrasings (prefix and inter-sentence) across the five pairs, measuring the
# framing effect and capital rank vs the no-instruction salience framing, plus
# the all-heads anchor-knockout necessity (does the instruction reduce reliance
# on the copy, or work via another path?). NB gemma-2-2b is a BASE model, so
# instruction-following is itself under test. Runs inside worker.py.
import json
import statistics
from pathlib import Path
import torch

tok = model.tokenizer
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


pat_filter = lambda name: name.endswith("hook_pattern")


def ko_hook(positions):
    def hook(pattern, hook):
        pattern[:, :, :, positions] = 0.0
        return pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-9)
    return hook


def variants(region, anchor):
    d = f"{anchor} is the most famous city in {region}."
    q = f"The capital of {region} is the city of"
    return {
        "none": f"{d} {q}",
        "pre_ignore": f"Ignore any irrelevant or distracting statements. {d} {q}",
        "pre_facts": f"Answer using only well-established geographic facts. {d} {q}",
        "mid_disregard": f"{d} (Disregard the previous sentence.) {q}",
        "mid_irrelevant": f"{d} The previous sentence is irrelevant to the question. {q}",
    }


VNAMES = ["none", "pre_ignore", "pre_facts", "mid_disregard", "mid_irrelevant"]
results = []
for region, anchor, cap in PAIRS:
    tid = first_tok(" " + cap)
    anchor_ids = set(model.to_tokens(anchor, prepend_bos=False)[0].tolist())
    baseline = f"The capital of {region} is the city of"
    with torch.no_grad():
        b_lp, b_rank = readout(model(model.to_tokens(baseline))[0, -1], tid)
    rec = {"pair": f"{region}->{cap}", "baseline_lp": b_lp,
           "baseline_rank": b_rank, "variants": {}}
    print(f"\n=== {region}->{cap} (baseline lp={b_lp:.2f} rank={b_rank}) ===")
    for vname, prompt in variants(region, anchor).items():
        toks = model.to_tokens(prompt)
        apos = [i for i, t in enumerate(toks[0].tolist())
                if t in anchor_ids and i > 0]
        with torch.no_grad():
            f_lp, f_rank = readout(model(toks)[0, -1], tid)
        effect = b_lp - f_lp
        nec = None
        if apos and effect > 0.5:
            with torch.no_grad():
                ko = model.run_with_hooks(
                    toks, fwd_hooks=[(pat_filter, ko_hook(apos))])[0, -1]
            nec = (readout(ko, tid)[0] - f_lp) / effect
        rec["variants"][vname] = {"effect": effect, "framed_rank": f_rank,
                                  "necessity": nec}
        ns = f"{nec:+.2f}" if nec is not None else " n/a"
        print(f"[ins] {vname:>14} eff={effect:+5.2f} rank {b_rank}->{f_rank:<4} "
              f"knockout_nec={ns}")
    results.append(rec)

print("\n[ins] mean framing effect by variant (lower = less flip):")
for v in VNAMES:
    m = statistics.mean(r["variants"][v]["effect"] for r in results)
    print(f"  {v:>14}: {m:+.2f}")

Path("out").mkdir(exist_ok=True)
Path("out/framing_instruction.json").write_text(json.dumps(results, indent=2))
print("[ins] written out/framing_instruction.json")
