#!/usr/bin/env bash
# Local Lambda lifecycle for ONE box: launch -> poll -> scp code -> run on-box batch ->
# fetch results -> TERMINATE. Teardown on NORMAL completion/error is guaranteed via `trap ... EXIT`;
# a LOCAL interrupt (Ctrl-C/SIGHUP) AFTER the on-box self-destruct backstop is armed deliberately does
# NOT tear down (the run is detached and the backstop bounds billing) so a stray Ctrl-C can't kill a
# live run. The remote batch also self-kills after a hard timeout. Credentials read from
# ./.keys (LAMBDA_KEY_ONE, HF_KEY_ONE) and never echoed.
#   usage: bash lambda_run.sh <instance_type> <region> <onbox_runner.sh> <local_result_dir>
set -uo pipefail
TYPE=$1; REGION=$2; RUNNER=$3; RDIR=$4
API=https://cloud.lambda.ai/api/v1
KEY=$(grep '^LAMBDA_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
HF=$(grep '^HF_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
SSHOPT="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=120 -i $HOME/.ssh/lambda_ed25519"   # ServerAlive keeps the SSH session up through long QUIET compute (model load + 891-item screen) -- a quiet idle is what dropped the run on 2026-06-22; ~1h unresponsiveness tolerated
REMOTE_TIMEOUT=${REMOTE_TIMEOUT:-5400}   # 90 min hard cap (env-overridable for heavier multi-load runs)
POLL_EVERY=${POLL_EVERY:-60}             # local marker-poll cadence for the DETACHED on-box run (see below)
REATTACH_GRACE=${REATTACH_GRACE:-1800}   # extra seconds the on-box backstop waits past REMOTE_TIMEOUT before
                                         # self-destruct -- also the window a LATER session has to reattach +
                                         # fetch if THIS launcher dies (e.g. the laptop is closed mid-run).
                                         # Bump it (e.g. 21600) for long runs you may need to recover offline.
auth(){ curl -sS -H "Authorization: Bearer $KEY" "$@"; }

ID=""
BACKSTOP_ARMED=0   # flipped to 1 once the on-box self-destruct timer is armed; gates local-kill behaviour below
terminate(){
  if [ -n "$ID" ]; then
    echo "[teardown] terminating $ID"
    # RETRY: a transient local DNS/network blip at teardown previously left the box billing (2026-06-22).
    # Retry until the API confirms 'terminating', so one failed curl can no longer orphan the instance.
    # ~10 min retry window: a DNS/network outage at teardown previously ran >60s and orphaned a box
    # (2026-06-26). 30 tries x 20s survives a longer local-network blackout before giving up.
    for t in $(seq 1 30); do
      if auth -m 20 -X POST $API/instance-operations/terminate -H 'Content-Type: application/json' \
           --data "{\"instance_ids\":[\"$ID\"]}" 2>/dev/null | grep -q 'terminat'; then
        echo "[teardown] terminate accepted for $ID (try $t)"; return
      fi
      echo "[teardown] terminate try $t failed (network/DNS?); retrying in 20s"; sleep 20
    done
    echo "[teardown] WARNING: terminate NOT confirmed for $ID after 30 tries -- VERIFY/KILL MANUALLY: $ID"
  fi
}
# EXIT trap tears the box down on NORMAL completion or error. A LOCAL interrupt (Ctrl-C / SIGHUP /
# terminal loss) is handled separately: AFTER the on-box self-destruct backstop is armed it must NOT
# kill the box -- the job is DETACHED (setsid) and the backstop bounds billing, so a stray local Ctrl-C
# should leave the run going (the earlier failure: watching the poll loop, Ctrl-C fired `trap terminate
# EXIT`, box torn down mid-run + cascading ssh-abort 255). BEFORE the backstop is armed a local interrupt
# still tears down (the box is not yet self-protected, so declining would orphan billing).
on_signal(){
  if [ "$BACKSTOP_ARMED" = 1 ]; then
    echo "[signal] local interrupt after backstop armed -> LEAVING box $ID RUNNING; detached job continues, self-destruct backstop terminates it within ${SELFDESTRUCT_AFTER:-?}s. Reconnect via ssh, or terminate $ID manually to stop billing sooner."
    trap - EXIT            # cancel the imminent EXIT teardown so the box survives
    exit 130
  fi
  echo "[signal] local interrupt BEFORE backstop armed -> box not yet self-protected; tearing down"
  exit 1                   # EXIT trap fires -> terminate()
}
trap terminate EXIT          # normal completion / error -> teardown (billing safety net)
trap on_signal INT TERM HUP  # local kill -> defer to on-box backstop if armed (see block above)

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
# Record the instance so a LATER session can reattach + fetch if THIS launcher dies (closed laptop / killed
# terminal). The on-box run is setsid-detached and the backstop bounds billing, so a dead launcher only
# loses the FETCH -- lambda_reattach.sh recovers it from this file within the REATTACH_GRACE window.
printf '%s %s %s %s\n' "$ID" "$IP" "$RDIR" "$(date -u +%s 2>/dev/null || echo 0)" > .last_lambda_instance
echo "[reattach] recorded $ID $IP -> .last_lambda_instance (recover via: bash lambda_reattach.sh)"

echo "[ssh] waiting for sshd @ $IP"
SSHUP=0
for i in $(seq 1 24); do ssh $SSHOPT ubuntu@$IP true 2>/dev/null && { echo "[ssh] up"; SSHUP=1; break; }; sleep 10; done
# UNHEALTHY box: sshd never came up in ~4min -> abort NOW (exit fires the teardown trap -> terminate). A box
# that never accepts SSH otherwise bleeds billing for the full marker deadline (2026-06-28 orphan, ~3h).
[ "$SSHUP" = 0 ] && { echo "[ssh] sshd NEVER came up -> unhealthy box; aborting + tearing down"; exit 1; }

echo "[scp] code -> box"
ssh $SSHOPT ubuntu@$IP "mkdir -p latent_verify/out"
scp $SSHOPT job_rlhf_ovqk.py job_truthful_flip.py ov_norm_probe.py scale9b_numeric_generality.py \
  instr_triangulation.py gate_dont_delete.py rlhf_differential.py controls/qk_collapse_metric.py \
  atp_low_confirm.py headset_joint_patch.py headset_direction.py matched_item_deconfound.py misconception_pool.py \
  ov_magnitude_characterize.py ov_behavioral_scale.py realized_attention.py controls/qk_weight_2b_l18h5.py \
  controls/logit_lens_margin_trajectory.py controls/logit_lens_margin_matched.py controls/logit_lens_attribution.py \
  controls/entropy_neuron_gemma2.py controls/cave_direction_heldout.py controls/confidence_vs_cave_direction.py \
  controls/entropy_distributed_presoftcap.py controls/cave_direction_xregime_deconfound.py \
  controls/substrate_margin_grid.py sycophancy_items.json sycophancy_items_lowconf.json \
  panel_gens.json panel_gold.json causal_it_labels.json \
  controls/confidence_direction_causal.py controls/cave_direction_sae_decomp.py \
  controls/confidence_caving_gate.py controls/cave_direction_dla.py controls/cave_direction_dla_robust.py \
  controls/cave_direction_overlay.py controls/mlp_stream_caving_patch.py controls/faithful_copy_wstar.py \
  controls/faithful_caving.py controls/cave_suppress_vs_install.py controls/cave_carrier_deconfound.py \
  controls/confidence_caving_gate_faithful.py controls/cave_reader_pathpatch.py \
  controls/cave_attribution_graph.py controls/cave_copy_confidence_conditional.py \
  controls/cave_doubt_cue_attention.py controls/cave_headset_specificity.py \
  controls/cave_doubt_writes_cavedir.py controls/cave_prompt_feature_mechanism.py \
  controls/cave_circuit_patch.py controls/cave_doubt_write_vs_read.py controls/cave_doubt_route.py \
  controls/cave_social_source.py controls/cave_confidence_recruitment.py controls/cave_faithful_it_diff.py \
  controls/spike_eot_cavestate.py controls/cave_residstate_diff.py controls/cave_residstate_close.py \
  controls/cave_residstate_decisive.py \
  controls/cave_multisample_caverate.py controls/cave_judge_panel.py controls/cave_causal_localize.py \
  controls/cave_ablate_late_mlp.py \
  controls/cave_fold_vs_listen.py \
  controls/foldlisten_judge.py controls/family_generate_judge.py controls/verifier_family.py \
  controls/family_cave_diagnose.py controls/family_topk_shift.py controls/modelw_candidates.py \
  controls/verifier_family_ext.py controls/think_probe_identity.py \
  verifier_family_ext2.json combined_family.json mechanism_family_9bit.json controls/foldlisten_phase2.py \
  controls/foldlisten_phase3a.py results_foldlisten_p2/out/foldlisten_phase2_p2_9bit_summary.json \
  controls/foldlisten_phase3b.py results_foldlisten_p3a/out/phase3_handles_p3a_9bit.json \
  results_foldlisten_p3a/out/phase3_handles_p3a_9bit.npz \
  results_foldlisten_mech_2b/out/phase3_handles_p3a_2bit.json \
  results_foldlisten_mech_2b/out/phase3_handles_p3a_2bit.npz \
  controls/foldlisten_phase3c_riders.py \
  controls/cave_dir_calibration_geometry.py controls/cave_dir_doubt_injection.py \
  controls/cave_dir_mechanism.py controls/cave_dir_dose_finegrained.py controls/cave_doubt_decollide.py \
  controls/cave_defer_direction.py controls/cave_polarity_isolation.py scale9b_dose_response.py \
  controls/cave_residstate_anyscale.py controls/cave_faithful_it_mc.py \
  gen_outputs_table.py \
  remote_run.sh "$RUNNER" ubuntu@$IP:latent_verify/
# CRLF guard: a Windows-checkout (autocrlf) CRLF script shipped raw kills the on-box run at
# `set -uo pipefail` (2026-07-11 rc=2). Normalize every shipped .sh on the box, unconditionally.
ssh $SSHOPT ubuntu@$IP "sed -i 's/\r\$//' latent_verify/*.sh"

# --- ORPHAN BACKSTOP: the box self-terminates after the run cap + grace, even if THIS machine dies mid-run
# (2026-06-26: a local process exit before teardown orphaned a box). A detached on-box timer terminates THIS
# instance via the API. Never fires in the happy path (local fetch+terminate is far sooner) -- pure billing
# safety net. Key is written to a root-only (umask 077) file on the ephemeral, single-tenant box.
SELFDESTRUCT_AFTER=$((REMOTE_TIMEOUT + REATTACH_GRACE))
echo "[backstop] arming box self-destruct in ${SELFDESTRUCT_AFTER}s ($ID)"
ssh $SSHOPT ubuntu@$IP "umask 077; cat > ~/selfdestruct.sh" <<EOF
#!/usr/bin/env bash
sleep $SELFDESTRUCT_AFTER
for t in 1 2 3 4 5 6 7 8 9 10 11 12; do
  curl -sS -m 30 -X POST $API/instance-operations/terminate -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" --data '{"instance_ids":["$ID"]}' | grep -q terminat && exit 0
  sleep 30
done
EOF
ssh $SSHOPT ubuntu@$IP "setsid bash ~/selfdestruct.sh < /dev/null > /tmp/selfdestruct.log 2>&1 &" 2>/dev/null
# ARM local-kill-survives ONLY if the backstop process is CONFIRMED running on the box. The start ssh
# above is `2>/dev/null` and `set -u` has no `-e`, so a silent ssh failure would otherwise still reach an
# unconditional arm -> a later local Ctrl-C would then leave the box with NEITHER the EXIT-trap teardown
# NOR a backstop = orphaned billing. Verify (pgrep), don't trust the exit path.
if ssh $SSHOPT ubuntu@$IP "pgrep -f selfdestruct.sh >/dev/null" 2>/dev/null; then
  BACKSTOP_ARMED=1   # box self-protected: a local Ctrl-C/SIGHUP now leaves the detached run going (on_signal)
  echo "[backstop] armed + confirmed running"
else
  echo "[backstop] WARNING: self-destruct NOT confirmed on box -> keeping EXIT-trap teardown ACTIVE (a local Ctrl-C will tear down the box; no orphan risk)"
fi

echo "[run] $RUNNER DETACHED on box (cap ${REMOTE_TIMEOUT}s); local marker-poll every ${POLL_EVERY}s"
# Start the job under setsid + </dev/null so a dropped local SSH cannot SIGHUP-kill it; on finish it writes
# latent_verify/RUN_DONE=<exitcode>. We then poll that marker over fresh short-lived SSH connections, so a
# transient local-network drop just retries next cycle instead of killing the run (the 2026-06-26 failure).
# $REMOTE_TIMEOUT/$HF/$RUNNER expand LOCALLY (double quotes); \$? is the REMOTE exit code; the inner
# single-quoted bash -c body is passed literally to the box.
STARTED=0
for s in 1 2 3; do
  ssh $SSHOPT ubuntu@$IP "cd latent_verify && rm -f RUN_DONE && setsid bash -c 'timeout $REMOTE_TIMEOUT env HF_TOKEN=\"$HF\" bash remote_run.sh bash $RUNNER > out/run_detached.log 2>&1; echo \$? > RUN_DONE' < /dev/null > /dev/null 2>&1 &" 2>/dev/null
  sleep 8
  if ssh $SSHOPT ubuntu@$IP "test -f latent_verify/out/run_detached.log" 2>/dev/null; then
    echo "[run] confirmed started (try $s)"; STARTED=1; break
  fi
  echo "[run] start not confirmed (try $s); retrying"
done
[ "$STARTED" = 0 ] && echo "[run] WARNING: start unconfirmed; polling for marker anyway"

RC="?"; DEADLINE=$((REMOTE_TIMEOUT + 900)); EL=0; NOCONN=0; EVERCONN=0
while [ $EL -lt $DEADLINE ]; do
  sleep $POLL_EVERY; EL=$((EL + POLL_EVERY))
  [ -f STOP_ABLATE ] && { echo "[run] STOP_ABLATE flag set -> abort + teardown"; RC="STOPPED"; break; }
  if ssh $SSHOPT ubuntu@$IP true 2>/dev/null; then              # box reachable this cycle
    EVERCONN=1; NOCONN=0
    M=$(ssh $SSHOPT ubuntu@$IP "cat latent_verify/RUN_DONE 2>/dev/null" 2>/dev/null | tr -dc '0-9')
    if [ -n "$M" ]; then RC="$M"; echo "[run] DONE marker rc=$RC after ${EL}s"; break; fi
    P=$(ssh $SSHOPT ubuntu@$IP "tail -1 latent_verify/out/run_detached.log 2>/dev/null" 2>/dev/null | tr -d '\r' | head -c 110)
    echo "[poll-run] +${EL}s | ${P:-<running>}"
  else                                                          # could not reach the box this cycle
    NOCONN=$((NOCONN + 1))
    echo "[poll-run] +${EL}s | NO SSH (consec=$NOCONN ever_conn=$EVERCONN)"
    # UNHEALTHY: never reachable after ~8min -> terminate early (the 2026-06-28 orphan bled the full deadline)
    if [ "$EVERCONN" = 0 ] && [ "$NOCONN" -ge 8 ]; then
      echo "[run] ABORT: box never reachable in $((NOCONN * POLL_EVERY))s -> UNHEALTHY; terminate early"; RC="UNHEALTHY"; break
    fi
    # was healthy then went dark ~40min -> give up + terminate (longer tolerance survives a transient
    # network blip / brief laptop sleep; a genuinely closed laptop kills THIS process so this never runs --
    # that case is handled by the backstop + lambda_reattach.sh, not here).
    if [ "$NOCONN" -ge 40 ]; then
      echo "[run] ABORT: box unreachable $((NOCONN * POLL_EVERY))s -> LOST; terminate"; RC="LOST"; break
    fi
  fi
done
[ "$RC" = "?" ] && echo "[run] WARNING: no DONE marker before deadline (${DEADLINE}s); fetch what exists, tear down"

echo "[fetch] -> $RDIR : tiny criticals FIRST (summary/log/marker), then the big out/ blobs"
mkdir -p "$RDIR/out"
# Small, decision-bearing files first -- they survive a flaky link where a multi-MB gens json would truncate
# (2026-06-26). *summary*.json carries the numbers+decision; logs carry the printed verdict.
SUMOK=0
for f in 1 2 3 4 5; do
  scp $SSHOPT "ubuntu@$IP:latent_verify/out/*summary*.json" "$RDIR/out/" 2>/dev/null
  scp $SSHOPT "ubuntu@$IP:latent_verify/out/*.log" "$RDIR/out/" 2>/dev/null
  scp $SSHOPT "ubuntu@$IP:latent_verify/RUN_DONE" "$RDIR/" 2>/dev/null
  # verify a summary actually parses (a truncated scp is caught here, not trusted)
  if ls "$RDIR"/out/*summary*.json >/dev/null 2>&1 && \
     python -c "import json,glob,sys; [json.load(open(p)) for p in glob.glob('$RDIR/out/*summary*.json')]" 2>/dev/null; then
    echo "[fetch] summary+logs verified (try $f)"; SUMOK=1; break
  fi
  echo "[fetch] summary try $f not yet valid; retry 10s"; sleep 10
done
[ "$SUMOK" = 0 ] && echo "[fetch] WARNING: no VALID summary fetched; rely on logs"
# then the full out/ (big generation blobs for H3) -- best-effort; result already safe in the summary
FETCHED=0
for f in 1 2 3 4 5; do
  if scp -r $SSHOPT ubuntu@$IP:latent_verify/out "$RDIR/" 2>/dev/null; then echo "[fetch] full out/ ok (try $f)"; FETCHED=1; break; fi
  echo "[fetch] full try $f failed; retry 15s"; sleep 15
done
[ "$FETCHED" = 0 ] && echo "[fetch] full out/ incomplete; summary+logs above are authoritative"
echo "[done] $RDIR (teardown trap will terminate $ID)"
