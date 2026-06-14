# Step sec7.1: is L19/14947 a GENERAL "say the anchored city" mediator, or
# Canberra/Sydney-specific? sec3.6 found it the top DLA mediator for both
# false-anchor framings on Australia. Re-run the DLA selection + necessity test
# (framing_dla machinery) on the salience framing across the transport pairs, and
# track L19/14947 specifically: its DLA-score rank, and its INDIVIDUAL necessity
# (clamp just that one feature back to baseline) per pair. Runs in worker.py.
import json
import random
from pathlib import Path
import torch

from framing_dla import dla_direction, dla_scores, top_by_score
from framing_intervention import target_logprob, clamp_to, MIN_EFFECT

tok = model.tokenizer
first = lambda s: tok.encode(s, add_special_tokens=False)[0]
PROBE = (19, 14947)
PAIRS = [("Australia", "Sydney", "Canberra"), ("Texas", "Houston", "Austin"),
         ("Canada", "Toronto", "Ottawa"), ("Switzerland", "Zurich", "Bern"),
         ("Morocco", "Casablanca", "Rabat")]
rng = random.Random(0)

results = []
for region, anchor, cap in PAIRS:
    tid, did = first(" " + cap), first(" " + anchor)
    baseline = f"The capital of {region} is the city of"
    framed = (f"{anchor} is the most famous city in {region}. "
              f"The capital of {region} is the city of")

    b_logits, b_acts = logits_and_acts(model, baseline)
    f_logits, f_acts = logits_and_acts(model, framed)
    base_last, fr_last = b_acts[:, -1, :].float(), f_acts[:, -1, :].float()
    fr_pos = f_acts.shape[1] - 1
    base_lp, fr_lp = target_logprob(b_logits, tid), target_logprob(f_logits, tid)
    effect = base_lp - fr_lp

    direction = dla_direction(model, tid, did)
    scores = dla_scores(model, base_last, fr_last, direction)
    ranked = top_by_score(scores, scores.numel())          # full ranking
    probe_rank = ranked.index(PROBE) if PROBE in ranked else None
    probe_score = float(scores[PROBE[0], PROBE[1]])
    probe_dact = float(fr_last[PROBE[0], PROBE[1]] - base_last[PROBE[0], PROBE[1]])

    rec = {"pair": f"{region}->{cap}", "effect": effect,
           "top8_dla": [f"L{l}/{f}" for (l, f) in ranked[:8]],
           "probe_feature": f"L{PROBE[0]}/{PROBE[1]}",
           "probe_dla_rank": probe_rank, "probe_score": probe_score,
           "probe_delta_act": probe_dact}

    if abs(effect) >= MIN_EFFECT:
        # individual necessity of L19/14947 alone
        nec_lp = target_logprob(
            intervened_logits(model, framed, clamp_to([PROBE], fr_pos, base_last)), tid)
        rec["probe_individual_necessity"] = (nec_lp - fr_lp) / effect
        # top-1 DLA feature individual necessity, for comparison
        top1 = ranked[0]
        t1_lp = target_logprob(
            intervened_logits(model, framed, clamp_to([top1], fr_pos, base_last)), tid)
        rec["top1_feature"] = f"L{top1[0]}/{top1[1]}"
        rec["top1_individual_necessity"] = (t1_lp - fr_lp) / effect
    else:
        rec["probe_individual_necessity"] = None
        rec["top1_feature"] = None
        rec["top1_individual_necessity"] = None

    results.append(rec)
    pn = rec["probe_individual_necessity"]
    pns = f"{pn:+.2f}" if pn is not None else "n/a"
    print(f"[dlaT] {region:<12} eff={effect:+5.2f} | L19/14947 dla_rank="
          f"{str(probe_rank):>5} score={probe_score:+.3f} dact={probe_dact:+.1f} "
          f"indiv_nec={pns} | top1={rec['top1_feature']} "
          f"({rec['top1_individual_necessity'] if rec['top1_individual_necessity'] is None else round(rec['top1_individual_necessity'],2)})")

print("\n[dlaT] is L19/14947 a top-8 DLA mediator per pair?")
for r in results:
    inn = r["probe_individual_necessity"]
    print(f"  {r['pair']:<22} top8={'YES' if r['probe_feature'] in r['top8_dla'] else 'no '} "
          f"rank={r['probe_dla_rank']} indiv_nec={inn if inn is None else round(inn,2)}")

Path("out").mkdir(exist_ok=True)
Path("out/framing_dla_transport.json").write_text(json.dumps(results, indent=2))
print("[dlaT] written out/framing_dla_transport.json")
