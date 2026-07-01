#!/usr/bin/env bash
# Poll Lambda for ANY single-GPU box that fits 9b (>=40 GB VRAM, <= $10/hr) and launch the FOLD-vs-LISTEN
# behavioural judge run (run_foldlisten.sh) the moment one is free. Mirrors run_poll_launch_fvl_9b.sh: dynamic
# cheapest-that-fits pick, skip gh200 (ARM/cu124 mismatch), >=40 GB floor (9b bf16 ~21.8 GiB resident, and we
# load TWO 9b models sequentially). Single poller; NO concurrent manual launch.
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
    if not name.startswith('gpu_1x_'): continue
    if 'gh200' in name: continue
    it=info.get('instance_type',{})
    price=it.get('price_cents_per_hour',10**9)
    if price>1000: continue
    m=re.search(r'(\d+)\s*GB', it.get('gpu_description','') or '')
    gb=int(m.group(1)) if m else 0
    if gb<40: continue
    regs=[r['name'] for r in info.get('regions_with_capacity_available',[])]
    if regs: cands.append((price,name,gb,regs[0]))
cands.sort()
if cands:
    p,name,gb,reg=cands[0]; print(name,reg,gb,p)
" 2>/dev/null)
  if [ -n "$PICK" ]; then
    TYPE=$(echo "$PICK" | awk '{print $1}'); REGION=$(echo "$PICK" | awk '{print $2}')
    GB=$(echo "$PICK" | awk '{print $3}'); PR=$(echo "$PICK" | awk '{print $4}')
    echo "[poll $i] capacity: $TYPE @ $REGION (${GB}GB, ${PR}c/hr) -> launching fold-vs-listen judge"
    if REMOTE_TIMEOUT=5400 bash lambda_run.sh "$TYPE" "$REGION" run_foldlisten.sh results_foldlisten; then
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
