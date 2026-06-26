#!/usr/bin/env bash
# Poll Lambda preferring H100 (80GB, headroom for the DLA caching + generation) then A100; launch CAUSAL+DLA.
set -uo pipefail
cd /c/Users/helios.lyons/Documents/git/claude_scratchpad/latent_verify
K=$(grep '^LAMBDA_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
API=https://cloud.lambda.ai/api/v1
for i in $(seq 1 80); do
  PICK=$(curl -sS -m 30 -H "Authorization: Bearer $K" $API/instance-types 2>/dev/null | python -c "
import sys,json
try: d=json.load(sys.stdin)['data']
except Exception: sys.exit(0)
for name in ('gpu_1x_h100_sxm5','gpu_1x_h100_pcie','gpu_1x_a100_sxm4','gpu_1x_a100'):
    info=d.get(name)
    if not info: continue
    regs=[r['name'] for r in info.get('regions_with_capacity_available',[])]
    if regs:
        print(name, regs[0]); break
" 2>/dev/null)
  if [ -n "$PICK" ]; then
    TYPE=$(echo "$PICK" | awk '{print $1}'); REGION=$(echo "$PICK" | awk '{print $2}')
    echo "[poll $i] capacity: $TYPE @ $REGION -> launching CAUSAL+DLA (OOM-refactored)"
    REMOTE_TIMEOUT=9000 bash lambda_run.sh "$TYPE" "$REGION" run_causal_localize.sh results_causal_localize
    echo "[poll] lambda_run returned exit=$?"
    exit 0
  fi
  echo "[poll $i] no A100/H100 capacity; retry 180s"
  sleep 180
done
echo "[poll] gave up: no capacity window in ~4h"
exit 1
