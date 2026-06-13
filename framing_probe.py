"""Exploratory framing probe -- reuses the verified attribution stack to make
basic observations about how prompt FRAMING moves gemma-2-2b's behaviour.

This is deliberately a measurement instrument, not a hypothesis test: no
pre-registered pass/fail. It reads framing_situations.json, and for each
situation renders the same question under several framings, then reports:

  b0  Behaviour: for each framing, the model's top-k next-token continuation
      and -- if the situation pins a `target` answer string -- that answer's
      first-token logit / probability / rank, plus the shift (delta) versus the
      `baseline` framing. Answers the question "does the framing move the
      output, and which way?"

  b1  Attribution of the shift: at the prediction position (last token), the
      transcoder features whose activation moves MOST between baseline and each
      framing. Answers "which interpretable features does the framing recruit
      or suppress?" -- the same feature units the Dallas->Austin work clamps,
      here used descriptively rather than via intervention.

Loading path is shared with poc_minimal.load_model (the bf16 / memory-guarded
ReplacementModel load verified against circuit-tracer@041a9b2). Run on CPU:

    python framing_probe.py --stage all
    python framing_probe.py --stage b0          # behaviour only (no acts scan)
    python framing_probe.py --top-features 8
"""

import argparse
import json
from pathlib import Path

import torch

from poc_minimal import load_model, logits_and_acts, OUT_DIR

SITUATIONS = "framing_situations.json"


# --------------------------------------------------------------------------
# small helpers (no library coupling beyond the shared wrappers)
# --------------------------------------------------------------------------

def first_token_id(model, text):
    """First token id of `text` (the answer we track). Returns None if empty."""
    ids = model.tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if ids else None


def topk_tokens(model, logits, k):
    probs = torch.softmax(logits, dim=-1)
    vals, idx = probs.topk(k)
    return [{"token": model.tokenizer.decode([int(i)]),
             "id": int(i), "prob": float(p), "logit": float(logits[int(i)])}
            for p, i in zip(vals, idx)]


def rank_of(logits, token_id):
    return int((logits > logits[token_id]).sum().item())  # 0 = top-1


def target_readout(model, logits, target_id):
    probs = torch.softmax(logits, dim=-1)
    logprobs = torch.log_softmax(logits, dim=-1)
    return {"logit": float(logits[target_id]),
            "logprob": float(logprobs[target_id]),
            "prob": float(probs[target_id]),
            "rank": rank_of(logits, target_id)}


# --------------------------------------------------------------------------
# b0: behaviour under framing
# --------------------------------------------------------------------------

def stage_b0(model, situations, args):
    out = []
    for sit in situations:
        target = sit.get("target")
        tid = first_token_id(model, target) if target else None
        framings = sit["framings"]
        assert "baseline" in framings, f"{sit['id']} has no 'baseline' framing"

        # baseline first so every framing can report a delta against it
        cache = {}
        for name, prompt in framings.items():
            logits, _ = logits_and_acts(model, prompt)
            cache[name] = logits

        base = cache["baseline"]
        base_tgt = target_readout(model, base, tid) if tid is not None else None

        rec = {"id": sit["id"], "kind": sit.get("kind"), "target": target,
               "target_first_token":
                   (model.tokenizer.decode([tid]) if tid is not None else None),
               "framings": {}}
        print(f"\n=== {sit['id']} ({sit.get('kind')}) "
              f"target={target!r} ===")
        for name, prompt in framings.items():
            logits = cache[name]
            top = topk_tokens(model, logits, args.topk)
            entry = {"prompt": prompt, "top1": top[0]["token"], "topk": top}
            line = (f"[{name:>18}] top1={top[0]['token']!r:>12} "
                    f"p={top[0]['prob']:.3f}")
            if tid is not None:
                tr = target_readout(model, logits, tid)
                entry["target"] = tr
                # raw logits are NOT comparable across different prompts
                # (per-context softmax shift); the calibrated cross-framing
                # signal is the change in log-probability of the answer.
                d_logp = tr["logprob"] - base_tgt["logprob"]
                entry["target_delta_logprob_vs_baseline"] = d_logp
                entry["target_delta_rank_vs_baseline"] = (
                    tr["rank"] - base_tgt["rank"])
                line += (f" | target p={tr['prob']:.3f} rank={tr['rank']:>4}"
                         f" dlogp={d_logp:+.3f}")
            rec["framings"][name] = entry
            print(line)
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# b1: which features move between baseline and each framing (descriptive)
# --------------------------------------------------------------------------

def stage_b1(model, situations, args):
    out = []
    for sit in situations:
        framings = sit["framings"]
        # activations at the LAST (prediction) position for every framing
        last_acts = {}
        for name, prompt in framings.items():
            _, acts = logits_and_acts(model, prompt)   # [n_layers, seq, d]
            last_acts[name] = acts[:, -1, :].float()   # [n_layers, d]

        base = last_acts["baseline"]
        rec = {"id": sit["id"], "movers": {}}
        print(f"\n=== {sit['id']} :: feature movers at prediction position ===")
        for name in framings:
            if name == "baseline":
                continue
            delta = last_acts[name] - base                 # [n_layers, d]
            flat = delta.abs().flatten()
            k = min(args.top_features, flat.numel())
            top_idx = flat.topk(k).indices
            movers = []
            for fi in top_idx.tolist():
                layer, feat = divmod(fi, delta.shape[1])
                movers.append({
                    "feature": f"L{layer}/{feat}",
                    "baseline": float(base[layer, feat]),
                    "framed": float(last_acts[name][layer, feat]),
                    "delta": float(delta[layer, feat]),
                })
            rec["movers"][name] = movers
            tops = ", ".join(f"{m['feature']}({m['delta']:+.1f})"
                             for m in movers[:5])
            print(f"[{name:>18}] {tops}")
        out.append(rec)
    return out


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["b0", "b1", "all"], default="all")
    p.add_argument("--situations", default=SITUATIONS)
    p.add_argument("--topk", type=int, default=5,
                   help="how many next-token candidates to log per framing")
    p.add_argument("--top-features", type=int, default=8,
                   help="how many feature movers to log per framing in b1")
    args = p.parse_args()

    situations = json.loads(Path(args.situations).read_text())["situations"]
    model = load_model()
    OUT_DIR.mkdir(exist_ok=True)

    if args.stage in ("b0", "all"):
        b0 = stage_b0(model, situations, args)
        (OUT_DIR / "framing_b0.json").write_text(json.dumps(b0, indent=2))
        print(f"\n[b0] written to {OUT_DIR / 'framing_b0.json'}")
    if args.stage in ("b1", "all"):
        b1 = stage_b1(model, situations, args)
        (OUT_DIR / "framing_b1.json").write_text(json.dumps(b1, indent=2))
        print(f"[b1] written to {OUT_DIR / 'framing_b1.json'}")


if __name__ == "__main__":
    main()
