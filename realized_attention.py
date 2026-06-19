"""NEXT-2: realized attention-to-source on a CONTENT-copy input, base vs -it (RESEARCH_QUESTIONS Part 4).

Two questions, one probe:
  (NEXT-2)  27b found W_QK weights UNCHANGED base->it, but the 2b L18.H5 result was a realized-PATTERN
            collapse the weight norm could not see, and the induction probe was insensitive. Does the
            REALIZED attention-to-source change base->it on a content-copy input despite W_QK unchanged?
  (NEXT-3b) NEXT-3b's behavioral scale-ablation was noisy because the OV-gain heads were selected by
            WEIGHTS-ONLY copy-pref and never shown to REALIZE copying. Do they actually attend the source
            token they would copy? If realized attention is ~0, their OV gain is behaviorally moot.

Probe (content-copy, not random-token induction): "The secret word is {w}. Remember it. The secret word
is" -> the readout (last position) should copy {w}; a copy head attends from the readout to the SOURCE
{w} token in the first clause. {w} is a single content token (multi-token words are skipped).

Per head, mean over prompts: realized_attn = pattern[head, readout, source], base and -it; plus the
model's copy accuracy (argmax at readout == w). NEUTRAL decision per head:
  REALIZES_COPY    if realized_attn_it >= ATTN_FLOOR        (the head does attend the source it would copy)
  NImportant: realized_attn near 0 -> the head does NOT realize copy on content (weights-only copy-pref
              does not imply realized copying; bears on NEXT-3b).
  QK_GATED_AT_SCALE if |realized_attn_it - realized_attn_base| / max(base, eps) > REL_TOL   (realized
              attention changed base->it even though W_QK weight norm did not -> the 2b-style realized
              collapse, invisible to the weight metric)

  python realized_attention.py --selftest
  python realized_attention.py --name-base google/gemma-2-27b --name-it google/gemma-2-27b-it --tag 27b   # 80GB GPU
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

ATTN_FLOOR = 0.10
REL_TOL = 0.15
# copy basket (same heads as qk_collapse / ov_magnitude); the OV-gain heads are L17.H4, L11 group, L23.H24
HEADS = [(11, 2), (11, 4), (11, 7), (11, 21), (16, 3), (17, 4), (19, 2), (19, 5), (19, 7), (23, 24)]
# single-token content words for the copy source (verified single-token at runtime; multi-token skipped)
WORDS = ["apple", "ocean", "silver", "tiger", "castle", "planet", "violin", "copper", "garden", "rocket",
         "monkey", "pencil", "window", "dragon", "harbor", "velvet", "cactus", "meadow", "lantern", "marble"]
TEMPLATE = "The secret word is {w}. Remember it. The secret word is"


def _rel(it, base):
    return (it - base) / max(abs(base), 1e-9)


def decide(attn_base, attn_it, attn_floor=ATTN_FLOOR, rel_tol=REL_TOL):
    realizes = attn_it >= attn_floor
    rel = _rel(attn_it, attn_base)
    gated = abs(rel) > rel_tol
    return {"realizes_copy": realizes, "realized_attn_rel": round(rel, 4),
            "qk_gated_at_scale": gated,
            "verdict": ("REALIZES_COPY" if realizes else "DOES_NOT_REALIZE_COPY")
                       + ("; QK_GATED_AT_SCALE" if gated else "")}


def _zname(L):
    return f"blocks.{L}.attn.hook_pattern"


def _per_model(model, heads, device):
    tok = model.tokenizer
    layers = sorted({L for (L, _) in heads})
    attn = {(L, H): [] for (L, H) in heads}
    n_copy, n_ok = 0, 0
    for w in WORDS:
        enc = tok.encode(" " + w, add_special_tokens=False)
        if len(enc) != 1:                           # require a single-token word for a clean source pos
            continue
        wid = enc[0]
        ids = model.to_tokens(TEMPLATE.format(w=w)).to(device)
        toklist = ids[0].tolist()
        if wid not in toklist:                      # word not a clean standalone token in this prompt
            continue
        source = toklist.index(wid)                 # first occurrence = the source clause
        readout = ids.shape[1] - 1
        store = {}
        def grab(p, hook):
            store[hook.layer()] = p[0].detach().float(); return p   # [head, query, key]
        with torch.no_grad():
            lg = model.run_with_hooks(ids, fwd_hooks=[(_zname(L), grab) for L in layers])
        n_ok += 1
        n_copy += int(int(lg[0, readout].argmax()) == wid)
        for (L, H) in heads:
            attn[(L, H)].append(float(store[L][H, readout, source]))
    return ({k: (statistics.mean(v) if v else 0.0) for k, v in attn.items()},
            (n_copy / n_ok if n_ok else 0.0), n_ok)


def run(name_base, name_it, tag, device):
    from transformer_lens import HookedTransformer
    res = {}
    for label, name in [("base", name_base), ("it", name_it)]:
        print(f"[load] {name} on {device}", flush=True)
        m = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
        m.eval()
        a, acc, n_ok = _per_model(m, HEADS, device)
        res[label] = {"attn": a, "copy_acc": acc, "n_ok": n_ok}
        print(f"  [{label}] copy_acc={acc:.2f} over {n_ok} prompts", flush=True)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()

    measurements = {}
    for (L, H) in HEADS:
        ab, ai = res["base"]["attn"][(L, H)], res["it"]["attn"][(L, H)]
        d = decide(ab, ai)
        measurements[f"{L},{H}"] = {"attn_base": round(ab, 4), "attn_it": round(ai, 4), **d}
    out = {"name_base": name_base, "name_it": name_it, "tag": tag, "template": TEMPLATE,
           "attn_floor": ATTN_FLOOR, "rel_tol": REL_TOL,
           "base_copy_acc": round(res["base"]["copy_acc"], 3), "it_copy_acc": round(res["it"]["copy_acc"], 3),
           "n_prompts_base": res["base"]["n_ok"], "n_prompts_it": res["it"]["n_ok"],
           "measurements": measurements}
    Path("out").mkdir(exist_ok=True)
    Path(f"out/realized_attention_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\n[copy_acc] base={out['base_copy_acc']} it={out['it_copy_acc']}", flush=True)
    for (L, H) in HEADS:
        m = measurements[f"{L},{H}"]
        print(f"  L{L}.H{H}: attn base->it {m['attn_base']:.3f}->{m['attn_it']:.3f} (rel {m['realized_attn_rel']:+.2f}) [{m['verdict']}]", flush=True)
    print(f"[done] wrote out/realized_attention_{tag}.json", flush=True)


def selftest():
    # decide: attends source -> REALIZES; below floor -> DOES_NOT; large base->it change -> gated tag
    assert decide(0.30, 0.32)["verdict"] == "REALIZES_COPY"
    assert decide(0.30, 0.02)["realizes_copy"] is False
    assert decide(0.30, 0.02)["verdict"].startswith("DOES_NOT_REALIZE_COPY")
    g = decide(0.30, 0.05)                         # realized attention collapsed base->it
    assert g["qk_gated_at_scale"] and "QK_GATED_AT_SCALE" in g["verdict"]
    ng = decide(0.30, 0.31)                        # ~unchanged -> not gated
    assert not ng["qk_gated_at_scale"]
    assert decide(0.005, 0.30)["qk_gated_at_scale"]   # rose from ~0 -> gated (rel huge)
    print("[selftest] decide: REALIZES / DOES_NOT / QK_GATED all fire")
    # source-finding logic mirror: first occurrence index of the source token
    toks = [2, 5, 9, 7, 9, 3]                       # source token 9 first appears at index 2
    assert toks.index(9) == 2
    print("[selftest] source = first occurrence index; readout = last position")
    print("[selftest] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--name-base", default="google/gemma-2-27b")
    ap.add_argument("--name-it", default="google/gemma-2-27b-it")
    ap.add_argument("--tag", default="27b")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a.name_base, a.name_it, a.tag, a.device)
