#!/usr/bin/env bash
# Poll Lambda for a >=40GB single-GPU box and run BOX A of the forced-final distributional replay
# (REGISTRATION_forcedfinal_distributional.md section 13.5-13.6): 2b-base, 2b-it, 9b-base, 9b-it,
# forward-only, ~55 min expected, cap REMOTE_TIMEOUT=5400 (90 min). Mirrors
# run_poll_launch_nelicit_27b.sh's portability fixes (own-dir resolve, python3, explicit
# SSH_KEY_NAME). Launches the LAUNCHER COPY .launcher_ff2b9b.sh -- never lambda_run.sh directly
# (section 13.6; editing lambda_run.sh while a launcher runs tears down a live box, OWED.md E1).
# Skip gh200 (ARM vs the x86 cu124 wheel); price cap $5.50/hr excludes b200.
#
# BEFORE LAUNCH (registration): run the offline census locally --
#   python3 controls/forcedfinal_source_census.py --run
# and re-reconstruct spend headroom from GET /api/v1/audit-events (never a committed tally).
# AFTER FETCH: python3 controls/forcedfinal_join.py --run   (offline; the only verdict source)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SSH_KEY_NAME="${SSH_KEY_NAME:-latent_verify_hal_20260721}"
K=$(grep '^LAMBDA_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
API=https://cloud.lambda.ai/api/v1
RDIR=results_ff_2b9b

# Launcher COPY (.launcher_*.sh is gitignored by design): recreate deterministically if absent --
# cp lambda_run.sh + insert the instrument and the four 2b/9b source summaries into the scp list
# (section 11.4/13.6). lambda_run.sh itself is NEVER edited.
if [ ! -f .launcher_ff2b9b.sh ]; then
  cp lambda_run.sh .launcher_ff2b9b.sh
  python3 - <<'PY'
anchor = '  remote_run.sh "$RUNNER" ubuntu@$IP:latent_verify/\n'
ins = ("  controls/forcedfinal_dist.py \\\n"
       "  results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_2bbase_ext2_summary.json \\\n"
       "  results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_2bit_ext2_summary.json \\\n"
       "  results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json \\\n"
       "  results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_9bit_ext2_summary.json \\\n")
f = ".launcher_ff2b9b.sh"
s = open(f).read()
assert s.count(anchor) == 1
open(f, "w").write(s.replace(anchor, ins + anchor))
print("[launcher] .launcher_ff2b9b.sh created (scp list + instrument + 4 source summaries)")
PY
fi

DONE=0
for i in $(seq 1 120); do     # ~6h at 180s
  PICK=$(curl -sS -m 30 -H "Authorization: Bearer $K" $API/instance-types 2>/dev/null | python3 -c "
import sys,json,re
try: d=json.load(sys.stdin)['data']
except Exception: sys.exit(0)
cands=[]
for name,info in d.items():
    if not name.startswith('gpu_1x_'): continue
    if 'gh200' in name: continue
    if 'b200' in name: continue
    it=info.get('instance_type',{}); price=it.get('price_cents_per_hour',10**9)
    if price>550: continue
    m=re.search(r'(\d+)\s*GB', it.get('gpu_description','') or ''); gb=int(m.group(1)) if m else 0
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
    echo "[poll $i] capacity: $TYPE @ $REGION (${GB}GB, ${PR}c/hr) -> launching BOX A (ff 2b9b, cap 90min)"
    if REMOTE_TIMEOUT=5400 bash .launcher_ff2b9b.sh "$TYPE" "$REGION" run_forcedfinal_dist_2b9b.sh "$RDIR"; then
      echo "[poll] BOX A completed (results fetched)"; DONE=1; break
    fi
    echo "[poll $i] launch/run failed (capacity race?); continue polling in 180s"
  else
    echo "[poll $i] no >=40GB <=\$5.50/hr capacity; retry 180s"
  fi
  sleep 180
done
[ "$DONE" = 0 ] && { echo "[poll] gave up: no >=40GB window in ~6h"; exit 1; }
echo "[poll] BOX A done -> $RDIR/out ; copy out/forcedfinal_dist_ff_ext2_{2bbase,2bit,9bbase,9bit}.json to ./out/, then run the 27b box and the offline join; confirm INSTANCE_COUNT 0"
exit 0
