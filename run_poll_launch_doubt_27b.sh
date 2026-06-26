#!/usr/bin/env bash
# Poll Lambda for an >=80GB single-GPU box (fits gemma-2-27b bf16 ~54GB resident) and launch the 27b doubt
# circuit (Job B) the moment one frees. H100 has NO capacity at launch time, so poll. Cheapest-fitting first.
# Skips gh200 (ARM/Grace-Hopper: remote_run.sh's x86 cu124 torch wheel -> CUDA unavailable -> CPU fallback).
# Caps price <= $5.50/hr so it takes an h100_pcie ($3.29) / h100_sxm5 ($4.29) but NOT a b200 (too new for cu124).
set -uo pipefail
cd /c/Users/helios.lyons/Documents/git/claude_scratchpad/latent_verify
K=$(grep '^LAMBDA_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
API=https://cloud.lambda.ai/api/v1
for i in $(seq 1 120); do      # ~6h at 180s
  PICK=$(curl -sS -m 30 -H "Authorization: Bearer $K" $API/instance-types 2>/dev/null | python -c "
import sys,json,re
try: d=json.load(sys.stdin)['data']
except Exception: sys.exit(0)
cands=[]
for name,info in d.items():
    if not name.startswith('gpu_1x_'): continue
    if 'gh200' in name: continue
    it=info.get('instance_type',{}); price=it.get('price_cents_per_hour',10**9)
    if price>550: continue
    m=re.search(r'(\d+)\s*GB', it.get('gpu_description','') or ''); gb=int(m.group(1)) if m else 0
    if gb<80: continue
    regs=[r['name'] for r in info.get('regions_with_capacity_available',[])]
    if regs: cands.append((price,name,gb,regs[0]))
cands.sort()
if cands:
    p,name,gb,reg=cands[0]; print(name,reg,gb,p)
" 2>/dev/null)
  if [ -n "$PICK" ]; then
    TYPE=$(echo "$PICK" | awk '{print $1}'); REGION=$(echo "$PICK" | awk '{print $2}')
    GB=$(echo "$PICK" | awk '{print $3}'); PR=$(echo "$PICK" | awk '{print $4}')
    echo "[poll $i] capacity: $TYPE @ $REGION (${GB}GB, ${PR}c/hr) -> launching 27b doubt circuit"
    if REMOTE_TIMEOUT=12000 bash lambda_run.sh "$TYPE" "$REGION" run_doubt_27b.sh results_doubt_27b; then
      echo "[poll] run box completed (result fetched); done"; exit 0
    fi
    echo "[poll $i] launch/run failed (capacity race?); continue polling in 180s"
  else
    echo "[poll $i] no >=80GB <=\$5.50/hr capacity; retry 180s"
  fi
  sleep 180
done
echo "[poll] gave up: no >=80GB capacity window in ~6h"
exit 1
