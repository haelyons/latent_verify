#!/usr/bin/env bash
# Poll Lambda for ANY single-GPU box that fits 9b (>=40 GB VRAM, <= $10/hr) and launch the 9b fold-vs-listen
# the moment one is free. Dynamic pick (parses gpu_description for GB) so it catches a6000/gh200/a100/h100
# alike, cheapest first -- per the user's "anything that works under $10/hr; momentum is the point". 9b OOM'd
# on the 24 GB A10 (weights ~21.8 GiB), so >=40 GB is the hard floor. Single poller; NO concurrent manual launch.
set -uo pipefail
cd /c/Users/helios.lyons/Documents/git/claude_scratchpad/latent_verify
K=$(grep '^LAMBDA_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
API=https://cloud.lambda.ai/api/v1
for i in $(seq 1 90); do      # ~4.5h at 180s
  PICK=$(curl -sS -m 30 -H "Authorization: Bearer $K" $API/instance-types 2>/dev/null | python -c "
import sys,json,re
try: d=json.load(sys.stdin)['data']
except Exception: sys.exit(0)
cands=[]
for name,info in d.items():
    if not name.startswith('gpu_1x_'): continue            # single GPU only (TL loads to one device; no model-parallel)
    if 'gh200' in name: continue                            # ARM/Grace-Hopper: remote_run.sh's x86 cu124 torch wheel -> CUDA NOT available -> setup aborts
    it=info.get('instance_type',{})
    price=it.get('price_cents_per_hour',10**9)
    if price>1000: continue                                 # <= \$10/hr
    m=re.search(r'(\d+)\s*GB', it.get('gpu_description','') or '')
    gb=int(m.group(1)) if m else 0
    if gb<40: continue                                      # 9b bf16 ~21.8 GiB resident -> need >=40 GB headroom
    regs=[r['name'] for r in info.get('regions_with_capacity_available',[])]
    if regs: cands.append((price,name,gb,regs[0]))
cands.sort()                                                # cheapest that fits, first
if cands:
    p,name,gb,reg=cands[0]; print(name,reg,gb,p)
" 2>/dev/null)
  if [ -n "$PICK" ]; then
    TYPE=$(echo "$PICK" | awk '{print $1}'); REGION=$(echo "$PICK" | awk '{print $2}')
    GB=$(echo "$PICK" | awk '{print $3}'); PR=$(echo "$PICK" | awk '{print $4}')
    echo "[poll $i] capacity: $TYPE @ $REGION (${GB}GB, ${PR}c/hr) -> launching 9b fold-vs-listen"
    # lambda_run exits 1 ONLY on launch/poll/ssh failure (no box, or unreachable) -> a capacity race; keep polling.
    # It exits 0 once the box ran (even if the on-box job itself failed) -> we are done, the result is fetched.
    if REMOTE_TIMEOUT=9000 bash lambda_run.sh "$TYPE" "$REGION" run_fold_vs_listen.sh results_fold_vs_listen; then
      echo "[poll] run box completed (result fetched); done"; exit 0
    fi
    echo "[poll $i] launch/run failed (capacity race?); continue polling in 180s"
  else
    echo "[poll $i] no >=40GB <=\$10/hr capacity; retry 180s"
  fi
  sleep 180
done
echo "[poll] gave up: no capacity window in ~4.5h"
exit 1
