#!/usr/bin/env bash
# Poll Lambda for ANY single-GPU box that fits 9b (>=40 GB VRAM, <= $10/hr) and run the NEUTRAL-ARM
# ELICITATION small-model boxes -- DESIGN_neutral_elicit.md sec 3.3 boxes 1 then 2, SEQUENTIALLY on two
# separate boxes -- the moment capacity frees. Mirrors run_poll_launch_foldlisten.sh / _fvl_9b.sh: dynamic
# cheapest-that-fits pick, skip gh200 (ARM/Grace-Hopper vs remote_run.sh's x86 cu124 wheel -> CUDA
# unavailable -> silent CPU fallback), >=40 GB hard floor (9b bf16 ~21.8 GiB resident; 9b OOM'd on a 24 GB
# A10). Single poller; NO concurrent manual launch, and do not run this alongside the 27b poller's boxes
# unless you have reconstructed spend from the audit log first.
#
#   box 1  run_foldlisten_nelicit_9b2b.sh     fl_9bit_anchor4 (n=22) -> 9b-base ext2 -> 2b-base ext2
#          372 records, est 2.5-3.5 h, cap 16200 (4.5 h)
#   box 2  run_foldlisten_nelicit_9b2bit.sh   9b-it ext2 -> 2b-it ext2
#          328 records, est 1.2-1.8 h, cap 10800 (3 h)
#
# PORTABILITY FIX (2026-07-28). All 22 committed run_poll_launch_*.sh hard-code the Windows laptop's
# working directory (`cd /c/Users/helios.lyons/...`) and call bare `python`. This Linux workstation has
# NEITHER: the repo is elsewhere and only python3 exists (/usr/bin/python3, no `python` on PATH). Those 22
# remain broken and are deliberately not touched here. This poller resolves its own directory and calls
# python3. SSH_KEY_NAME is likewise passed explicitly: lambda_run.sh:62 defaults to `latent_verify_helios`,
# the WINDOWS laptop's key, which this box's ~/.ssh/lambda_ed25519 does not match -- this workstation
# registered `latent_verify_hal_20260721` (docs/lambda-gpu-access.md:36-38).
#
# TEARDOWN is lambda_run.sh's and is not re-implemented here: it arms the on-box self-destruct backstop
# (REMOTE_TIMEOUT + REATTACH_GRACE) and CONFIRMS it with pgrep before starting the detached job, and it
# fetches out/*summary*.json + out/*.log + RUN_DONE BEFORE the EXIT trap terminates the box. If THIS poller
# dies mid-run the box keeps running: fetch FIRST, then terminate, via
#   bash lambda_reattach.sh results_foldlisten_nelicit_2b9b
# (or `bash lambda_reattach.sh <id> <ip> results_foldlisten_nelicit_2b9b` -- both boxes launch under the
# same drill_<rdir> name, so use the explicit form if you are unsure which one is live). Confirm the
# account shows INSTANCE_COUNT 0 after each box.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SSH_KEY_NAME="${SSH_KEY_NAME:-latent_verify_hal_20260721}"
K=$(grep '^LAMBDA_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
API=https://cloud.lambda.ai/api/v1
RDIR=results_foldlisten_nelicit_2b9b

BOX1=0
for i in $(seq 1 90); do      # ~4.5h at 180s
  PICK=$(curl -sS -m 30 -H "Authorization: Bearer $K" $API/instance-types 2>/dev/null | python3 -c "
import sys,json,re
try: d=json.load(sys.stdin)['data']
except Exception: sys.exit(0)
cands=[]
for name,info in d.items():
    if not name.startswith('gpu_1x_'): continue            # single GPU only (TL loads to one device)
    if 'gh200' in name: continue                            # ARM/Grace-Hopper -> x86 cu124 wheel -> CPU fallback
    if 'b200' in name: continue                             # Blackwell: no cu124 kernels -> remote_run.sh's assert kills the box
    it=info.get('instance_type',{})
    price=it.get('price_cents_per_hour',10**9)
    if price>1000: continue                                 # <= \$10/hr
    m=re.search(r'(\d+)\s*GB', it.get('gpu_description','') or '')
    gb=int(m.group(1)) if m else 0
    if gb<40: continue                                      # 9b bf16 ~21.8 GiB resident -> >=40 GB floor
    regs=[r['name'] for r in info.get('regions_with_capacity_available',[])]
    if regs: cands.append((price,name,gb,regs[0]))
cands.sort()                                                # cheapest that fits, first
if cands:
    p,name,gb,reg=cands[0]; print(name,reg,gb,p)
" 2>/dev/null)
  if [ -n "$PICK" ]; then
    TYPE=$(echo "$PICK" | awk '{print $1}'); REGION=$(echo "$PICK" | awk '{print $2}')
    GB=$(echo "$PICK" | awk '{print $3}'); PR=$(echo "$PICK" | awk '{print $4}')
    echo "[poll $i] capacity: $TYPE @ $REGION (${GB}GB, ${PR}c/hr) -> launching BOX 1 (anchor4 + 9b-base + 2b-base)"
    # lambda_run exits 1 ONLY on launch/poll/ssh failure (no box, or unreachable) -> a capacity race; keep
    # polling. It exits 0 once the box ran (even if the on-box job itself failed) -> results are fetched.
    # 16200 is the design's frozen value; parameterised so a launch can raise it without editing the
    # script. Box 1 carries 372 records against a Phase-B datapoint of 328 fitting the same 4.5h,
    # plus ~7% for the new decode - a cap kill here costs the cell in flight, which in the frozen
    # order is fl_2bbase_ext2, the highest-withhold base cell. Launched at 19800 on 2026-07-28.
    if REMOTE_TIMEOUT="${BOX1_TIMEOUT:-16200}" bash lambda_run.sh "$TYPE" "$REGION" run_foldlisten_nelicit_9b2b.sh "$RDIR"; then
      echo "[poll] BOX 1 completed (results fetched); moving to box 2"; BOX1=1; break
    fi
    echo "[poll $i] launch/run failed (capacity race?); continue polling in 180s"
  else
    echo "[poll $i] no >=40GB <=\$10/hr capacity; retry 180s"
  fi
  sleep 180
done
[ "$BOX1" = 0 ] && { echo "[poll] gave up: no >=40GB capacity window for BOX 1 in ~4.5h"; exit 1; }

BOX2=0
for i in $(seq 1 90); do      # ~4.5h at 180s
  PICK=$(curl -sS -m 30 -H "Authorization: Bearer $K" $API/instance-types 2>/dev/null | python3 -c "
import sys,json,re
try: d=json.load(sys.stdin)['data']
except Exception: sys.exit(0)
cands=[]
for name,info in d.items():
    if not name.startswith('gpu_1x_'): continue
    if 'gh200' in name: continue
    if 'b200' in name: continue
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
    echo "[poll $i] capacity: $TYPE @ $REGION (${GB}GB, ${PR}c/hr) -> launching BOX 2 (9b-it + 2b-it ext2)"
    if REMOTE_TIMEOUT=10800 bash lambda_run.sh "$TYPE" "$REGION" run_foldlisten_nelicit_9b2bit.sh "$RDIR"; then
      echo "[poll] BOX 2 completed (results fetched); done"; BOX2=1; break
    fi
    echo "[poll $i] launch/run failed (capacity race?); continue polling in 180s"
  else
    echo "[poll $i] no >=40GB <=\$10/hr capacity; retry 180s"
  fi
  sleep 180
done
[ "$BOX2" = 0 ] && { echo "[poll] gave up: no >=40GB capacity window for BOX 2 in ~4.5h (BOX 1 IS BANKED)"; exit 1; }
echo "[poll] both small boxes done -> $RDIR/out ; confirm INSTANCE_COUNT 0 before walking away"
exit 0
