#!/usr/bin/env bash
# Poll Lambda for A100/H100 capacity; launch the residual-state base<->it run the moment a box is free.
# Prefers the cheaper A100_sxm4, falls back to H100_sxm5. lambda_run.sh's trap handles teardown.
set -uo pipefail
cd /c/Users/helios.lyons/Documents/git/claude_scratchpad/latent_verify
K=$(grep '^LAMBDA_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
API=https://cloud.lambda.ai/api/v1
for i in $(seq 1 80); do      # ~4h max at 180s
  PICK=$(curl -sS -m 30 -H "Authorization: Bearer $K" $API/instance-types 2>/dev/null | python -c "
import sys,json
try: d=json.load(sys.stdin)['data']
except Exception: sys.exit(0)
for name in ('gpu_1x_a100_sxm4','gpu_1x_h100_sxm5'):
    info=d.get(name)
    if not info: continue
    regs=[r['name'] for r in info.get('regions_with_capacity_available',[])]
    if regs:
        print(name, regs[0]); break
" 2>/dev/null)
  if [ -n "$PICK" ]; then
    TYPE=$(echo "$PICK" | awk '{print $1}'); REGION=$(echo "$PICK" | awk '{print $2}')
    echo "[poll $i] capacity: $TYPE @ $REGION -> launching residstate run"
    REMOTE_TIMEOUT=7200 bash lambda_run.sh "$TYPE" "$REGION" run_residstate_diff.sh results_residstate
    echo "[poll] lambda_run returned exit=$?"
    exit 0
  fi
  echo "[poll $i] no A100/H100 capacity; retry in 180s"
  sleep 180
done
echo "[poll] gave up: no capacity window in ~4h"
exit 1
