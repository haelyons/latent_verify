#!/usr/bin/env bash
# Poll Lambda for a cheap box (A10 24GB fits 2b; fall back A100) and launch the CALIB 2b run.
set -uo pipefail
cd /c/Users/helios.lyons/Documents/git/claude_scratchpad/latent_verify
K=$(grep '^LAMBDA_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
API=https://cloud.lambda.ai/api/v1
for i in $(seq 1 80); do
  PICK=$(curl -sS -m 30 -H "Authorization: Bearer $K" $API/instance-types 2>/dev/null | python -c "
import sys,json
try: d=json.load(sys.stdin)['data']
except Exception: sys.exit(0)
for name in ('gpu_1x_a10','gpu_1x_a100_sxm4','gpu_1x_a100'):
    info=d.get(name)
    if not info: continue
    regs=[r['name'] for r in info.get('regions_with_capacity_available',[])]
    if regs:
        print(name, regs[0]); break
" 2>/dev/null)
  if [ -n "$PICK" ]; then
    TYPE=$(echo "$PICK" | awk '{print $1}'); REGION=$(echo "$PICK" | awk '{print $2}')
    echo "[poll $i] capacity: $TYPE @ $REGION -> launching CALIB 2b"
    REMOTE_TIMEOUT=5400 bash lambda_run.sh "$TYPE" "$REGION" run_calib_2b.sh results_calib_2b
    echo "[poll] lambda_run returned exit=$?"
    exit 0
  fi
  echo "[poll $i] no A10/A100 capacity; retry 120s"
  sleep 120
done
echo "[poll] gave up: no capacity window"
exit 1
