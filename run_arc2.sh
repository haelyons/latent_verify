#!/bin/bash
# Remote runner for the ARC2 spine. scp'd to the Lambda box alongside the 3 job
# scripts; nohup'd. Independence preserved: each job runs even if a prior one fails.
cd ~/arc2 || exit 1
echo "[run] $(date -u) starting on $(hostname)"
python3 -c "import torch; print('[run] torch', torch.__version__, 'cuda', torch.cuda.is_available())"
pip install -q -U transformer_lens transformers 2>&1 | tail -3
echo "[run] deps installed"

# gated-access smoke test before the long jobs
python3 -c "
from transformer_lens import HookedTransformer
import torch
m = HookedTransformer.from_pretrained_no_processing('google/gemma-2-2b-it', dtype=torch.bfloat16, device='cuda')
print('[run] SMOKE OK: gemma-2-2b-it loaded, n_layers', m.cfg.n_layers)
" || { echo "[run] SMOKE FAILED (HF gated access?)"; touch ~/arc2/DONE; exit 1; }

for j in job_arc2c_format_weights job_arc2a_transplant job_arc2b_numeric_it; do
  echo "[run] ===== $j START $(date -u) ====="
  python3 $j.py
  echo "[run] ===== $j END rc=$? $(date -u) ====="
done
echo "[run] ALL DONE $(date -u)"
touch ~/arc2/DONE
