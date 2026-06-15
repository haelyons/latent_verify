"""CPU-session smoke test: load ReplacementModel from local mirror weights and
validate against two external references before any experiment runs:
  1. behaviour: " Austin" is top-1 on the seed prompt (canonical graph claim);
  2. activations: the six pinned Texas features fire at the Dallas-token region
     with magnitudes matching Neuronpedia's anonymously-queried values
     (out/neuronpedia_seed_reference.txt) within a loose bf16 tolerance.
Passing both means mirror weights + transcoders + wrappers are trustworthy.
"""
import time

import torch

from poc_minimal import (SEED_PROMPT, TEXAS, load_model, logits_and_acts,
                         target_token_id)

# (layer, feat) -> Neuronpedia max activation on seed (fetched 2026-06-12)
NEURONPEDIA_MAX = {
    (20, 15589): 49.0,   # fires ' Dallas'=46, ' is'=49
    (19, 7477): 55.0,
    (16, 25): 27.875,
    (14, 2268): 25.625,
    (7, 6861): 15.5,
    (4, 13154): 15.0625,
}

t0 = time.time()
model = load_model()
print(f"[smoke] model loaded in {time.time()-t0:.0f}s")

tid = target_token_id(model)
t0 = time.time()
base, acts = logits_and_acts(model, SEED_PROMPT)
print(f"[smoke] forward pass in {time.time()-t0:.0f}s; acts shape {tuple(acts.shape)}")

top = int(base.argmax())
print(f"[smoke] top-1 token: {model.tokenizer.decode([top])!r} (id {top}); "
      f"Austin id {tid}, logit {float(base[tid]):.3f}, "
      f"rank {int((base > base[tid]).sum())}")

ok = True
for (l, f), ref in NEURONPEDIA_MAX.items():
    got = float(acts[l, :, f].max())
    rel = abs(got - ref) / ref
    status = "OK" if rel < 0.15 else "MISMATCH"
    ok &= status == "OK"
    print(f"[smoke] L{l}/{f}: local max={got:.2f} neuronpedia={ref} "
          f"rel_err={rel:.3f} {status}")
print(f"[smoke] {'PASS' if ok and top == tid else 'FAIL'}")
