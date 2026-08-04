#!/usr/bin/env bash
# Poll Lambda for a >=80GB single-GPU box and run BOX B of the forced-final distributional replay
# (REGISTRATION_forcedfinal_distributional.md section 13.5-13.6): 27b-base + 27b-it, forward-only,
# ~78 min expected, cap REMOTE_TIMEOUT=7200 (120 min). h100_sxm5 pinned per D-6 / section 13.5
# (capacity + the OWED.md H2 card-class note; every 27b number needs section 10's four-part
# disclosure regardless). Launches the LAUNCHER COPY .launcher_ff27b.sh -- never lambda_run.sh.
# If h100_sxm5 capacity never appears the deliberate fallback is a ONE-LINE edit widening the pin
# to h100_pcie; do NOT lower the cap when you do (the nelicit precedent, fd2154b rc=124).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SSH_KEY_NAME="${SSH_KEY_NAME:-latent_verify_hal_20260721}"
K=$(grep '^LAMBDA_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
API=https://cloud.lambda.ai/api/v1
RDIR=results_ff_27b

# Launcher COPY (.launcher_*.sh is gitignored by design): recreate deterministically if absent
# (section 11.4/13.6). lambda_run.sh itself is NEVER edited.
if [ ! -f .launcher_ff27b.sh ]; then
  cp lambda_run.sh .launcher_ff27b.sh
  python3 - <<'PY'
anchor = '  remote_run.sh "$RUNNER" ubuntu@$IP:latent_verify/\n'
ins = ("  controls/forcedfinal_dist.py \\\n"
       "  results_foldlisten_nelicit_27b/out/foldlisten_judge_fl_27bbase_ext2_summary.json \\\n"
       "  results_foldlisten_nelicit_27b/out/foldlisten_judge_fl_27bit_ext2_summary.json \\\n")
f = ".launcher_ff27b.sh"
s = open(f).read()
assert s.count(anchor) == 1
open(f, "w").write(s.replace(anchor, ins + anchor))
print("[launcher] .launcher_ff27b.sh created (scp list + instrument + 2 source summaries)")
PY
fi

DONE=0
for i in $(seq 1 160); do     # ~8h at 180s (the sxm5 pin narrows the candidate set)
  PICK=$(curl -sS -m 30 -H "Authorization: Bearer $K" $API/instance-types 2>/dev/null | python3 -c "
import sys,json,re
try: d=json.load(sys.stdin)['data']
except Exception: sys.exit(0)
cands=[]
for name,info in d.items():
    if not name.startswith('gpu_1x_'): continue
    if 'gh200' in name: continue
    if 'h100_sxm5' not in name: continue
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
    echo "[poll $i] capacity: $TYPE @ $REGION (${GB}GB, ${PR}c/hr) -> launching BOX B (ff 27b, cap 120min)"
    if REMOTE_TIMEOUT=7200 bash .launcher_ff27b.sh "$TYPE" "$REGION" run_forcedfinal_dist_27b.sh "$RDIR"; then
      echo "[poll] BOX B completed (results fetched)"; DONE=1; break
    fi
    echo "[poll $i] launch/run failed (capacity race?); continue polling in 180s"
  else
    echo "[poll $i] no h100_sxm5 (>=80GB <=\$5.50/hr) capacity; retry 180s"
  fi
  sleep 180
done
[ "$DONE" = 0 ] && { echo "[poll] gave up: no h100_sxm5 window in ~8h -- widen the pin deliberately, keep the cap"; exit 1; }
echo "[poll] BOX B done -> $RDIR/out ; copy out/forcedfinal_dist_ff_ext2_{27bbase,27bit}.json to ./out/, then: python3 controls/forcedfinal_join.py --run ; confirm INSTANCE_COUNT 0"
exit 0
