"""N-6 (SEQUENCE_170626): does post-training suppress DISTRACTOR copying while preserving
TASK-RELEVANT context use? Reconciles FRAMING_NOTES sec 8 ("RLHF deletes the copy") with
Kim et al. 2510.02370 (instruction tuning -> MORE in-context reliance).

Two conditions, fragment regime for BOTH base and it (matches sec-8's weight-level claim,
the it/bare cell), same in-context-entity slot:
  distractor: "{w} is the most famous city in {r}. " + stem   -- entity w (famous non-capital)
              is IRRELEVANT to the capital answer.
  taskrel   : "The capital of {r} is {c}. " + stem            -- entity c (correct capital)
              IS the answer; using it is correct.
Per condition: boost = lp(entity | context) - lp(entity | neutral) at the readout; the
L18.H5 reader attention to the entity; and the max attention any head puts on the entity.
Prediction: base uses/attends the entity in BOTH; it uses/attends it in taskrel but NOT in
distractor. Falsified if it suppresses the taskrel entity too (then sec-8 deletion is a
blanket context down-weighting, not distractor-specific) -> the Resolved-note reconciliation
would be wrong.
  python job_distractor_task.py --model base|it     # -> out/distractor_vs_task_{base,it}.json
"""
import argparse
import json
import statistics
from pathlib import Path

STEM = "The capital of {r} is the city of"
PAIRS = [
    {"r": "Australia",   "c": "Canberra",    "w": "Sydney"},
    {"r": "Texas",       "c": "Austin",      "w": "Houston"},
    {"r": "Canada",      "c": "Ottawa",      "w": "Toronto"},
    {"r": "Switzerland", "c": "Bern",        "w": "Zurich"},
    {"r": "Florida",     "c": "Tallahassee", "w": "Miami"},
]
READER = (18, 5)


def run(name, tag):
    import torch
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {name} on {device}", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tok = model.tokenizer
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    first = lambda s: tok.encode(s, add_special_tokens=False)[0]
    pat_nf = lambda nm: nm.endswith("hook_pattern")

    def lp_entity(ids, eid):
        with torch.no_grad():
            ll = model(ids)[0, -1].float()
        return float(torch.log_softmax(ll, -1)[eid])

    def tok_pos(ids_list, text):
        tset = set(model.to_tokens(text, prepend_bos=False)[0].tolist())
        tset |= set(model.to_tokens(" " + text, prepend_bos=False)[0].tolist())
        return [i for i, t in enumerate(ids_list) if t in tset and i > 0]

    def attns_to(ids, positions):
        """L18.H5 attention and max-over-all-heads attention to `positions` at the readout."""
        with torch.no_grad():
            _, cache = model.run_with_cache(ids, names_filter=pat_nf)
        reader = float(cache[f"blocks.{READER[0]}.attn.hook_pattern"][0, READER[1], -1, positions].sum())
        best = (-1.0, None, None)
        for L in range(nL):
            p = cache[f"blocks.{L}.attn.hook_pattern"][0]      # [head, q, k]
            for H in range(nH):
                v = float(p[H, -1, positions].sum())
                if v > best[0]:
                    best = (v, L, H)
        return reader, best

    rows = []
    for pr in PAIRS:
        r, c, w = pr["r"], pr["c"], pr["w"]
        neutral = model.to_tokens(STEM.format(r=r)).to(device)
        rec = {"region": r}
        for cond, entity, ctx_text in [("distractor", w, f"{w} is the most famous city in {r}. "),
                                       ("taskrel",    c, f"The capital of {r} is {c}. ")]:
            eid = first(" " + entity)
            ids = model.to_tokens(ctx_text + STEM.format(r=r)).to(device)
            epos = tok_pos(ids[0].tolist(), entity)
            boost = lp_entity(ids, eid) - lp_entity(neutral, eid)
            reader_attn, (mx, mL, mH) = attns_to(ids, epos)
            rec[cond] = {"entity": entity, "boost": round(boost, 3),
                         "reader_L18H5_attn": round(reader_attn, 3),
                         "max_head_attn": round(mx, 3), "max_head": [mL, mH]}
            print(f"[{tag} {r:<12} {cond:<10}] entity={entity:<11} boost={boost:+.2f} "
                  f"reader->ent={reader_attn:.2f} maxhead L{mL}.H{mH}={mx:.2f}", flush=True)
        rows.append(rec)

    mean = lambda xs: round(statistics.mean(xs), 3)
    summary = {cond: {"mean_boost": mean([r[cond]["boost"] for r in rows]),
                      "mean_reader_attn": mean([r[cond]["reader_L18H5_attn"] for r in rows]),
                      "mean_max_head_attn": mean([r[cond]["max_head_attn"] for r in rows])}
               for cond in ("distractor", "taskrel")}
    out = {"model": name, "regime": "fragment", "n_pairs": len(PAIRS),
           "summary": summary, "per_pair": rows}
    Path("out").mkdir(exist_ok=True)
    Path(f"out/distractor_vs_task_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[done] {name}\n  distractor: {summary['distractor']}\n  taskrel:    {summary['taskrel']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "it"])
    ap.add_argument("--name")
    ap.add_argument("--tag")
    a = ap.parse_args()
    if a.name:
        name, tag = a.name, (a.tag or a.name.split("/")[-1].replace("-", "_"))
    else:
        name = "google/gemma-2-2b" if a.model == "base" else "google/gemma-2-2b-it"
        tag = a.tag or a.model
    run(name, tag)


if __name__ == "__main__":
    main()
