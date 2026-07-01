#!/usr/bin/env bash
# Sequential FOLD-vs-LISTEN across scales, ONE box at a time (torn down before the next launches): 9b -> 2b -> 27b,
# each base+it. Per scale: poll Lambda for the cheapest single-GPU box that fits, launch via lambda_run.sh, run,
# fetch, terminate. gh200 skipped (ARM/cu124). 27b needs >=80GB (~54GB bf16 resident).
set -uo pipefail
cd /c/Users/helios.lyons/Documents/git/claude_scratchpad/latent_verify
K=$(grep '^LAMBDA_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
API=https://cloud.lambda.ai/api/v1

poll_run(){  # $1=runner  $2=rdir  $3=minGB  $4=maxCents  $5=remoteTimeout
  local RUNNER=$1 RDIR=$2 MINGB=$3 MAXC=$4 TMO=$5
  echo "[scale] === $RUNNER -> $RDIR (>= ${MINGB}GB, <= ${MAXC}c/hr, cap ${TMO}s) ==="
  for i in $(seq 1 120); do      # ~6h at 180s
    PICK=$(curl -sS -m 30 -H "Authorization: Bearer $K" $API/instance-types 2>/dev/null | MINGB=$MINGB MAXC=$MAXC python -c "
import sys,json,re,os
mingb=int(os.environ['MINGB']); maxc=int(os.environ['MAXC'])
try: d=json.load(sys.stdin)['data']
except Exception: sys.exit(0)
c=[]
for name,info in d.items():
    if not name.startswith('gpu_1x_'): continue
    if 'gh200' in name: continue
    it=info.get('instance_type',{}); price=it.get('price_cents_per_hour',10**9)
    if price>maxc: continue
    m=re.search(r'(\d+)\s*GB', it.get('gpu_description','') or ''); gb=int(m.group(1)) if m else 0
    if gb<mingb: continue
    regs=[r['name'] for r in info.get('regions_with_capacity_available',[])]
    if regs: c.append((price,name,gb,regs[0]))
c.sort()
if c: p,n,gb,r=c[0]; print(n,r,gb,p)
" 2>/dev/null)
    if [ -n "$PICK" ]; then
      TYPE=$(echo "$PICK"|awk '{print $1}'); REGION=$(echo "$PICK"|awk '{print $2}')
      GB=$(echo "$PICK"|awk '{print $3}'); PR=$(echo "$PICK"|awk '{print $4}')
      echo "[scale $RUNNER poll $i] capacity: $TYPE @ $REGION (${GB}GB, ${PR}c/hr) -> launching"
      if REMOTE_TIMEOUT=$TMO bash lambda_run.sh "$TYPE" "$REGION" "$RUNNER" "$RDIR"; then
        echo "[scale $RUNNER] completed (result fetched)"; return 0
      fi
      echo "[scale $RUNNER poll $i] launch/run failed (capacity race?); retry 180s"
    else
      echo "[scale $RUNNER poll $i] no >=${MINGB}GB <=${MAXC}c/hr capacity; retry 180s"
    fi
    sleep 180
  done
  echo "[scale $RUNNER] GAVE UP (no capacity window)"; return 1
}

poll_run run_foldlisten_9b.sh  results_foldlisten      40 1000 5400  || echo "[allscales] 9b failed; continuing"
poll_run run_foldlisten_2b.sh  results_foldlisten_2b   20  600 5400  || echo "[allscales] 2b failed; continuing"
poll_run run_foldlisten_27b.sh results_foldlisten_27b  80  550 12000 || echo "[allscales] 27b failed"
echo "ALLSCALES_DONE"
