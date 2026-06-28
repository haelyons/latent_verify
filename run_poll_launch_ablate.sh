#!/usr/bin/env bash
# Poll Lambda for an H100 box (>=80GB/GPU, < $20/hr) and launch the MLP-ablation run. H100-ONLY by design:
# the remote stack is x86 + torch cu124 (remote_run.sh:9), which is Hopper-proven. GH200 is aarch64 -> the
# x86 cu124 wheel gives NO CUDA (FRAMING sec 11 guard aborted a GH200 attempt 2026-06-26); B200 is Blackwell
# sm_100, newer than cu124. Both excluded. NO 40GB A100 (incl 8x_a100): live per-condition -it free-gen +
# self-judge OOM'd a 40GB A100 (archive/research_log.md:2488-2489). Model loads on cuda:0 only, so multi-GPU
# H100 boxes just idle the extra GPUs. Over-budget excluded: 8x_a100_80gb $22.32, 8x_h100 $31.92.
set -uo pipefail
cd /c/Users/helios.lyons/Documents/git/claude_scratchpad/latent_verify
K=$(grep '^LAMBDA_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
API=https://cloud.lambda.ai/api/v1
for i in $(seq 1 160); do
  [ -f STOP_ABLATE ] && { echo "[poll $i] STOP_ABLATE flag set -> exit (no launch)"; exit 0; }
  PICK=$(curl -sS -m 30 -H "Authorization: Bearer $K" $API/instance-types 2>/dev/null | python -c "
import sys,json
try: d=json.load(sys.stdin)['data']
except Exception: sys.exit(0)
# H100 only (x86 + cu124 proven), >=80GB/GPU and < \$20/hr, cheapest-first
for name in ('gpu_1x_h100_pcie','gpu_1x_h100_sxm5','gpu_2x_h100_sxm5','gpu_4x_h100_sxm5'):
    info=d.get(name)
    if not info: continue
    regs=[r['name'] for r in info.get('regions_with_capacity_available',[])]
    if regs:
        print(name, regs[0]); break
" 2>/dev/null)
  if [ -n "$PICK" ]; then
    TYPE=$(echo "$PICK" | awk '{print $1}'); REGION=$(echo "$PICK" | awk '{print $2}')
    echo "[poll $i] capacity: $TYPE @ $REGION -> launching MLP-ABLATION"
    REMOTE_TIMEOUT=10800 bash lambda_run.sh "$TYPE" "$REGION" run_ablate_mlp.sh results_ablate_mlp; RC=$?
    # lambda_run rc 0 == a box was launched + run + torn down (stop). rc 1 == never got a usable box
    # (capacity race / rejected launch) -> do NOT give up, keep polling.
    if [ "$RC" = 0 ]; then echo "[poll $i] a box ran (lambda_run rc 0) -> done"; exit 0; fi
    echo "[poll $i] launch failed (rc $RC; capacity race or rejected) -> keep polling"
  else
    echo "[poll $i] no H100 capacity"
  fi
  sleep 180
done
echo "[poll] gave up: no qualifying capacity window in ~8h"
exit 1
