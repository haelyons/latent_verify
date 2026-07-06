#!/usr/bin/env bash
# Reattach to a Lambda box whose driving lambda_run.sh died mid-run (e.g. the laptop holding the session
# was closed). The on-box job is setsid-detached (keeps running) and the on-box self-destruct backstop
# bounds billing, so a dead launcher only loses the FETCH -- this recovers it, then terminates. Reads the
# instance from .last_lambda_instance (written by lambda_run.sh at launch) unless given explicitly.
# Idempotent + safe: if the run is still going (no RUN_DONE) it fetches nothing and does NOT terminate --
# re-run it later. Run from any session, from the repo root (needs ./.keys + ~/.ssh/lambda_ed25519).
#   usage: bash lambda_reattach.sh [instance_id ip result_dir]
set -uo pipefail
KEY=$(grep '^LAMBDA_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
API=https://cloud.lambda.ai/api/v1
SSHOPT="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -i $HOME/.ssh/lambda_ed25519"
auth(){ curl -sS -H "Authorization: Bearer $KEY" "$@"; }

if [ $# -ge 3 ]; then
  ID=$1; IP=$2; RDIR=$3
elif [ $# -eq 1 ]; then
  # <rdir>: look the box up by its launch name (drill_<rdir>) via the API. Robust for CONCURRENT runs
  # (each box is uniquely named), and needs no local record file -- works from any fresh session.
  RDIR=$1
  read -r ID IP < <(auth $API/instances | python -c "
import sys,json
for i in json.load(sys.stdin).get('data',[]):
    if i.get('name')=='drill_$RDIR' and i.get('status')=='active':
        print(i.get('id',''), i.get('ip','') or ''); break
")
  { [ -n "${ID:-}" ] && [ -n "${IP:-}" ]; } || { echo "[reattach] no ACTIVE instance named drill_$RDIR (already gone, or never launched)"; exit 1; }
else
  [ -f .last_lambda_instance ] || { echo "[reattach] usage: lambda_reattach.sh <rdir> | <id ip rdir>  (or have .last_lambda_instance)"; exit 1; }
  read -r ID IP RDIR _ < .last_lambda_instance
fi
{ [ -n "${ID:-}" ] && [ -n "${IP:-}" ] && [ -n "${RDIR:-}" ]; } || { echo "[reattach] bad instance record"; exit 1; }
echo "[reattach] id=$ID ip=$IP -> $RDIR"

S=$(auth $API/instances/$ID | python -c 'import sys,json;print((json.load(sys.stdin).get("data") or {}).get("status",""))' 2>/dev/null)
echo "[reattach] instance status=${S:-unknown}"
[ "$S" = active ] || { echo "[reattach] not active -> box already gone; nothing to fetch (results lost with the box)"; exit 1; }

M=$(ssh $SSHOPT ubuntu@$IP "cat latent_verify/RUN_DONE 2>/dev/null" 2>/dev/null | tr -dc '0-9')
if [ -z "$M" ]; then
  echo "[reattach] RUN_DONE not present -> run still going (or box unreachable). Tail:"
  ssh $SSHOPT ubuntu@$IP "tail -3 latent_verify/out/run_detached.log 2>/dev/null" 2>/dev/null
  echo "[reattach] NOT terminating; re-run this script once RUN_DONE appears."
  exit 2
fi
echo "[reattach] RUN_DONE rc=$M -> fetching"

mkdir -p "$RDIR/out"
for f in 1 2 3 4 5; do
  scp $SSHOPT "ubuntu@$IP:latent_verify/out/*summary*.json" "$RDIR/out/" 2>/dev/null
  scp $SSHOPT "ubuntu@$IP:latent_verify/out/*.log" "$RDIR/out/" 2>/dev/null
  scp $SSHOPT "ubuntu@$IP:latent_verify/RUN_DONE" "$RDIR/" 2>/dev/null
  if ls "$RDIR"/out/*summary*.json >/dev/null 2>&1 && \
     python -c "import json,glob;[json.load(open(p)) for p in glob.glob('$RDIR/out/*summary*.json')]" 2>/dev/null; then
    echo "[reattach] summaries+logs verified (try $f)"; break
  fi
  echo "[reattach] summary try $f not yet valid; retry 10s"; sleep 10
done
for f in 1 2 3 4 5; do
  if scp -r $SSHOPT ubuntu@$IP:latent_verify/out "$RDIR/" 2>/dev/null; then echo "[reattach] full out/ ok (try $f)"; break; fi
  echo "[reattach] full out/ try $f failed; retry 15s"; sleep 15
done

echo "[reattach] fetch done -> $RDIR ; terminating $ID"
for t in $(seq 1 30); do
  if auth -m 20 -X POST $API/instance-operations/terminate -H 'Content-Type: application/json' \
       --data "{\"instance_ids\":[\"$ID\"]}" 2>/dev/null | grep -q 'terminat'; then
    echo "[reattach] terminate accepted"; rm -f .last_lambda_instance; exit 0
  fi
  echo "[reattach] terminate try $t failed; retry 20s"; sleep 20
done
echo "[reattach] WARNING: terminate NOT confirmed after 30 tries -- verify/kill $ID MANUALLY"
