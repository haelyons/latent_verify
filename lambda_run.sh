#!/usr/bin/env bash
# Local Lambda lifecycle for ONE box: launch -> poll -> scp code -> run on-box batch ->
# fetch results -> TERMINATE. Teardown is guaranteed via `trap ... EXIT` so a box can never
# orphan billing, and the remote batch self-kills after a hard timeout. Credentials read from
# ./.keys (LAMBDA_KEY_ONE, HF_KEY_ONE) and never echoed.
#   usage: bash lambda_run.sh <instance_type> <region> <onbox_runner.sh> <local_result_dir>
set -uo pipefail
TYPE=$1; REGION=$2; RUNNER=$3; RDIR=$4
API=https://cloud.lambda.ai/api/v1
KEY=$(grep '^LAMBDA_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
HF=$(grep '^HF_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
SSHOPT="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -i $HOME/.ssh/lambda_ed25519"
REMOTE_TIMEOUT=5400          # 90 min hard cap on the on-box batch (safety against a hung job)
auth(){ curl -sS -H "Authorization: Bearer $KEY" "$@"; }

ID=""
terminate(){
  if [ -n "$ID" ]; then
    echo "[teardown] terminating $ID"
    auth -X POST $API/instance-operations/terminate -H 'Content-Type: application/json' \
      --data "{\"instance_ids\":[\"$ID\"]}" >/dev/null && echo "[teardown] terminate sent for $ID"
  fi
}
trap terminate EXIT          # fires on success, error, or Ctrl-C -> box always torn down

echo "[launch] $TYPE @ $REGION"
ID=$(auth -X POST $API/instance-operations/launch -H 'Content-Type: application/json' \
  --data "{\"region_name\":\"$REGION\",\"instance_type_name\":\"$TYPE\",\"ssh_key_names\":[\"latent_verify_helios\"],\"name\":\"drill_$RDIR\"}" \
  | python -c 'import sys,json;d=json.load(sys.stdin);print((d.get("data") or {}).get("instance_ids",[""])[0])')
[ -z "$ID" ] && { echo "[launch] FAILED (no id; capacity?)"; exit 1; }
echo "[launch] id=$ID"

IP=""
for i in $(seq 1 80); do
  R=$(auth $API/instances/$ID)
  S=$(echo "$R" | python -c 'import sys,json;print(json.load(sys.stdin)["data"].get("status",""))' 2>/dev/null)
  IP=$(echo "$R" | python -c 'import sys,json;print(json.load(sys.stdin)["data"].get("ip") or "")' 2>/dev/null)
  echo "[poll $i] status=$S ip=$IP"
  [ "$S" = active ] && [ -n "$IP" ] && break
  sleep 15
done
[ -z "$IP" ] && { echo "[poll] never became active"; exit 1; }

echo "[ssh] waiting for sshd @ $IP"
for i in $(seq 1 24); do ssh $SSHOPT ubuntu@$IP true 2>/dev/null && { echo "[ssh] up"; break; }; sleep 10; done

echo "[scp] code -> box"
ssh $SSHOPT ubuntu@$IP "mkdir -p latent_verify/out"
scp $SSHOPT job_rlhf_ovqk.py job_truthful_flip.py ov_norm_probe.py scale9b_numeric_generality.py \
  instr_triangulation.py gate_dont_delete.py rlhf_differential.py controls/qk_collapse_metric.py \
  atp_low_confirm.py headset_joint_patch.py headset_direction.py matched_item_deconfound.py misconception_pool.py \
  ov_magnitude_characterize.py remote_run.sh "$RUNNER" ubuntu@$IP:latent_verify/

echo "[run] $RUNNER (hard cap ${REMOTE_TIMEOUT}s)"
ssh $SSHOPT ubuntu@$IP "cd latent_verify && timeout $REMOTE_TIMEOUT env HF_TOKEN='$HF' bash remote_run.sh bash $RUNNER"
echo "[run] remote exit=$?"

echo "[fetch] out/ -> $RDIR"
mkdir -p "$RDIR"
scp -r $SSHOPT ubuntu@$IP:latent_verify/out "$RDIR/" 2>/dev/null && echo "[fetch] ok" || echo "[fetch] FAILED"
echo "[done] $RDIR (teardown trap will terminate $ID)"
