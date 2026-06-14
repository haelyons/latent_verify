"""Minimal load de-risk: does the gqfo2t memory-guarded load path actually
bring up ReplacementModel(gemma-2-2b + GemmaScope) on THIS CPU sandbox?

Loads, runs one forward pass on the canonical seed, and checks the known-good
result (Austin top-1, the six Texas features fire). Prints peak RSS so we can
compare against the cpu-load-investigation numbers. No experiment logic here.
"""
import resource
import time

import torch

from poc_minimal import SEED_PROMPT, TEXAS, load_model, logits_and_acts, target_token_id


def peak_rss_gb():
    # ru_maxrss is kB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


t = time.time()
print("[load] starting model load ...", flush=True)
model = load_model()
print(f"[load] model up in {time.time()-t:.0f}s | peak RSS {peak_rss_gb():.2f} GB", flush=True)

tid = target_token_id(model)
t = time.time()
base, acts = logits_and_acts(model, SEED_PROMPT)
print(f"[fwd] forward pass in {time.time()-t:.1f}s | acts {tuple(acts.shape)} "
      f"| peak RSS {peak_rss_gb():.2f} GB", flush=True)

top = int(base.argmax())
print(f"[check] top1={model.tokenizer.decode([top])!r} (id {top}) | "
      f"Austin id {tid} logit {float(base[tid]):.2f} rank {int((base>base[tid]).sum())}")
for (l, f) in TEXAS:
    print(f"[check] L{l}/{f} seed max act = {float(acts[l,:,f].max()):.2f}")
print("[check] PASS" if top == tid else "[check] FAIL (Austin not top-1)")
