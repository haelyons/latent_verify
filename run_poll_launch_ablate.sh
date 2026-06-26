#!/usr/bin/env bash
# Poll Lambda for a HIGH-MEMORY box (>=80GB: GH200 96GB / H100 80GB) and launch the MLP-ablation run.
# NO A100 (40GB) fallback by design: this control does live per-condition -it free-generation + self-judge,
# the exact pattern that OOM'd a 40GB A100 (archive/research_log.md:2488-2489). User directive: prefer the
# higher memory ceiling, do not squeeze onto an undersized GPU.
set -uo pipefail
cd /c/Users/helios.lyons/Documents/git/claude_scratchpad/latent_verify
K=$(grep '^LAMBDA_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
API=https://cloud.lambda.ai/api/v1
for i in $(seq 1 120); do
  PICK=$(curl -sS -m 30 -H "Authorization: Bearer $K" $API/instance-types 2>/dev/null | python -c "
import sys,json
try: d=json.load(sys.stdin)['data']
except Exception: sys.exit(0)
for name in ('gpu_1x_gh200','gpu_1x_h100_sxm5','gpu_1x_h100_pcie'):
    info=d.get(name)
    if not info: continue
    regs=[r['name'] for r in info.get('regions_with_capacity_available',[])]
    if regs:
        print(name, regs[0]); break
" 2>/dev/null)
  if [ -n "$PICK" ]; then
    TYPE=$(echo "$PICK" | awk '{print $1}'); REGION=$(echo "$PICK" | awk '{print $2}')
    echo "[poll $i] capacity: $TYPE @ $REGION -> launching MLP-ABLATION (high-mem)"
    REMOTE_TIMEOUT=10800 bash lambda_run.sh "$TYPE" "$REGION" run_ablate_mlp.sh results_ablate_mlp
    echo "[poll] lambda_run returned exit=$?"
    exit 0
  fi
  echo "[poll $i] no GH200/H100 capacity; retry 180s"
  sleep 180
done
echo "[poll] gave up: no high-mem capacity window in ~6h"
exit 1
