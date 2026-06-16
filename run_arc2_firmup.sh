#!/bin/bash
# Firm-up wave (run 2) runner. Env already fixed (TL 3.4 / transformers 5.12.1 /
# numpy<2 / fresh Pillow), per run_arc2_noinstall.sh. Does NOT re-run the paid run-1
# jobs. Base-only correctives first (cheaper, the headline-protecting tests), then the
# dual-model ladder. Independence preserved: each job runs even if a prior one fails.
cd ~/arc2 || exit 1
rm -f DONE_FIRMUP
echo "[firmup] $(date -u) start on $(hostname)"
python3 -c "from transformer_lens import HookedTransformer; import torch; m=HookedTransformer.from_pretrained_no_processing('google/gemma-2-2b',dtype=torch.bfloat16,device='cuda'); print('[firmup] SMOKE OK n_layers',m.cfg.n_layers)" || { echo '[firmup] SMOKE FAILED'; touch DONE_FIRMUP; exit 1; }
for j in job_supp_attr_extract job_supp_redundancy job_arc2a2_transplant_pool; do
  echo "[firmup] ===== $j START $(date -u) ====="
  python3 $j.py
  echo "[firmup] ===== $j END rc=$? $(date -u) ====="
done
echo "[firmup] ALL DONE $(date -u)"
touch DONE_FIRMUP
