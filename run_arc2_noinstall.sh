#!/bin/bash
# No-install runner: env already fixed (TL 3.4 / transformers 5.12.1 / numpy<2 /
# fresh Pillow). Just smoke-test gated load, then run the 3 jobs independently.
cd ~/arc2 || exit 1
rm -f DONE
echo "[run2] $(date -u) start on $(hostname)"
python3 -c "from transformer_lens import HookedTransformer; import torch; m=HookedTransformer.from_pretrained_no_processing('google/gemma-2-2b-it',dtype=torch.bfloat16,device='cuda'); print('[run2] SMOKE OK n_layers',m.cfg.n_layers)" || { echo '[run2] SMOKE FAILED'; touch DONE; exit 1; }
for j in job_arc2c_format_weights job_arc2a_transplant job_arc2b_numeric_it; do
  echo "[run2] ===== $j START $(date -u) ====="
  python3 $j.py
  echo "[run2] ===== $j END rc=$? $(date -u) ====="
done
echo "[run2] ALL DONE $(date -u)"
touch DONE
