#!/usr/bin/env bash
# Poll Lambda for a >=80GB single-GPU box (gemma-2-27b bf16 ~54GB resident) and run the NEUTRAL-ARM
# ELICITATION 27b boxes -- DESIGN_neutral_elicit.md sec 3.3 boxes 3 then 4, ONE CELL PER BOX, sequentially.
# Mirrors run_poll_launch_doubt_27b.sh / _calib_27b.sh: skip gh200 (ARM/Grace-Hopper vs remote_run.sh's x86
# cu124 wheel -> CUDA unavailable -> silent CPU fallback), price cap $5.50/hr so b200 (too new for cu124) is
# excluded. Single poller; NO concurrent manual launch.
#
#   box 3  run_foldlisten_nelicit_27bbase.sh  27b-base ext2   164 records   PINNED h100_sxm5   cap 25200 (7 h)
#   box 4  run_foldlisten_nelicit_27bit.sh    27b-it   ext2   164 records   cheapest >=80GB    cap 19800 (5.5 h)
#
# WHY BOX 3 IS PINNED TO H100 **SXM5** AND CARRIES A 7 h CAP (the one deviation from the sec 3.2 table).
# One 27b cell measures ~4.3 h on H100 PCIe (89 s/record; commit fd2154b, docs/lambda-gpu-access.md:41-42),
# and the neutral-elicited 4th arm adds ~+7 % base decode -> ~4.6 h against the design's 5.5 h cap: ~20 %
# headroom on an ESTIMATED marginal. A single-cell box that hits its cap banks NOTHING -- the precedent is
# this exact cell (fd2154b: rc=124, 128/164 items, zero artifact, ~$15 burnt for no data). SXM5 runs it in
# ~1.4-1.5 h and is therefore CHEAPER OVERALL (~$7 at $4.29/hr) than PCIe (~$15 at $3.29/hr) despite the
# higher hourly rate, as well as being nowhere near any cap. The 25200 s cap is then sized so the cell still
# FITS even if the SXM5 box turns out no faster than the measured PCIe pace (4.6 h x ~1.5); a cap that is
# never reached costs nothing. Reviewer bound: the on-box self-destruct fires at REMOTE_TIMEOUT +
# REATTACH_GRACE (default 1800) = 27000 s, so a dead launcher that nobody reattaches bills at most
# ~7.5 h x $4.29 ~= $32 on this box. If h100_sxm5 capacity never appears, the deliberate fallback is a
# ONE-LINE edit -- widen the `'h100_sxm5' not in name` filter below to also accept h100_pcie -- and the cap
# above was chosen so that fallback is still safe. Do NOT drop the cap back to 19800 when you do.
#
# PORTABILITY FIX (2026-07-28). All 22 committed run_poll_launch_*.sh hard-code the Windows laptop's
# working directory (`cd /c/Users/helios.lyons/...`) and call bare `python`. This Linux workstation has
# NEITHER (repo elsewhere; only /usr/bin/python3, no `python` on PATH). Those 22 remain broken and are
# deliberately not touched. This poller resolves its own directory and calls python3. SSH_KEY_NAME is
# passed explicitly because lambda_run.sh:62 defaults to the WINDOWS laptop's key name; this workstation
# registered `latent_verify_hal_20260721` (docs/lambda-gpu-access.md:36-38).
#
# TEARDOWN is lambda_run.sh's and is not re-implemented here: it arms the on-box self-destruct backstop and
# CONFIRMS it with pgrep before starting the detached job, and it fetches out/*summary*.json + out/*.log +
# RUN_DONE BEFORE the EXIT trap terminates. If THIS poller dies mid-run the box keeps running: fetch FIRST,
# then terminate, via
#   bash lambda_reattach.sh results_foldlisten_nelicit_27b
# (or `bash lambda_reattach.sh <id> <ip> results_foldlisten_nelicit_27b` -- both boxes launch under the same
# drill_<rdir> name, so use the explicit form if you are unsure which one is live). Confirm the account
# shows INSTANCE_COUNT 0 after each box.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SSH_KEY_NAME="${SSH_KEY_NAME:-latent_verify_hal_20260721}"
K=$(grep '^LAMBDA_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
API=https://cloud.lambda.ai/api/v1
RDIR=results_foldlisten_nelicit_27b

BOX3=0
for i in $(seq 1 160); do     # ~8h at 180s -- a longer window than the usual 6h because the SXM5 pin
                              # narrows the candidate set to one instance type (see the header)
  PICK=$(curl -sS -m 30 -H "Authorization: Bearer $K" $API/instance-types 2>/dev/null | python3 -c "
import sys,json,re
try: d=json.load(sys.stdin)['data']
except Exception: sys.exit(0)
cands=[]
for name,info in d.items():
    if not name.startswith('gpu_1x_'): continue
    if 'gh200' in name: continue
    if 'h100_sxm5' not in name: continue                    # PINNED for 27b-base: see the header (cap economics)
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
    echo "[poll $i] capacity: $TYPE @ $REGION (${GB}GB, ${PR}c/hr) -> launching BOX 3 (27b-base ext2, cap 7h)"
    if REMOTE_TIMEOUT=25200 bash lambda_run.sh "$TYPE" "$REGION" run_foldlisten_nelicit_27bbase.sh "$RDIR"; then
      echo "[poll] BOX 3 completed (results fetched); moving to box 4"; BOX3=1; break
    fi
    echo "[poll $i] launch/run failed (capacity race?); continue polling in 180s"
  else
    echo "[poll $i] no h100_sxm5 (>=80GB <=\$5.50/hr) capacity; retry 180s"
  fi
  sleep 180
done
[ "$BOX3" = 0 ] && { echo "[poll] gave up: no h100_sxm5 window for BOX 3 in ~8h -- see the header before widening the pin"; exit 1; }

BOX4=0
for i in $(seq 1 120); do     # ~6h at 180s
  PICK=$(curl -sS -m 30 -H "Authorization: Bearer $K" $API/instance-types 2>/dev/null | python3 -c "
import sys,json,re
try: d=json.load(sys.stdin)['data']
except Exception: sys.exit(0)
cands=[]
for name,info in d.items():
    if not name.startswith('gpu_1x_'): continue
    if 'gh200' in name: continue
    if 'b200' in name: continue                             # Blackwell: no cu124 kernels (the <=\$5.50 cap was meant to exclude it)
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
    echo "[poll $i] capacity: $TYPE @ $REGION (${GB}GB, ${PR}c/hr) -> launching BOX 4 (27b-it ext2)"
    if REMOTE_TIMEOUT=19800 bash lambda_run.sh "$TYPE" "$REGION" run_foldlisten_nelicit_27bit.sh "$RDIR"; then
      echo "[poll] BOX 4 completed (results fetched); done"; BOX4=1; break
    fi
    echo "[poll $i] launch/run failed (capacity race?); continue polling in 180s"
  else
    echo "[poll $i] no >=80GB <=\$5.50/hr capacity; retry 180s"
  fi
  sleep 180
done
[ "$BOX4" = 0 ] && { echo "[poll] gave up: no >=80GB capacity window for BOX 4 in ~6h (BOX 3 IS BANKED)"; exit 1; }
echo "[poll] both 27b boxes done -> $RDIR/out ; confirm INSTANCE_COUNT 0 before walking away"
exit 0
