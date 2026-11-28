#!/usr/bin/env bash
# pod_noclose_queue.sh : runs ON THE POD (nohup setsid), started by the lead. Card PREREG_V8F_PROOF_ADD33 (NOCLOSE). Evaluation only, it trains nothing.
# The vision-fed distance rule with closing speed 0: RZ (threshold 0.5, gap as read; margin tool) and QZ (threshold P*, gap through the map; qcal tool).
# One rollout at a time, never as a fourth GPU job. It first waits for the queue of card ADD31 to end (QCAL_DONE.txt or QCAL_STOP.txt).
# Arms from noclose_arms.txt, lines: NEW QUEUE BASE KIND P X   (KIND m: X = margin in m, run script pod_margin_run.sh; KIND q: X = map file, run script pod_qcal_run.sh)
set -u
W=/workspace/v8proof; D=$W/distrule; R=$W/runs/distrule/t1; L=$D/noclose_queue.log
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$L"; }
[ -e "$D/NOCLOSE_DONE.txt" ] && { echo "REFUSED: $D/NOCLOSE_DONE.txt exists"; exit 2; }
[ -f "$D/noclose_arms.txt" ] || { echo "REFUSED: no noclose_arms.txt"; exit 2; }
log "queue start; arms $(sha256sum "$D/noclose_arms.txt" | cut -c1-16)"
stop() { log "ZZ NOCLOSE STOP: $1"; echo "$1" > "$D/NOCLOSE_STOP.txt"; exit 4; }
for i in $(seq 1 960); do
  [ -e "$D/QCAL_DONE.txt" ] || [ -e "$D/QCAL_STOP.txt" ] && break
  sleep 30
done
[ -e "$D/QCAL_DONE.txt" ] || [ -e "$D/QCAL_STOP.txt" ] || stop "WAIT-QCAL (8 h)"
[ -f "$D/pod_margin_run.sh" ] && [ -f "$D/pod_qcal_run.sh" ] || stop "RUN-SCRIPTS-MISSING"
log "the queue of card ADD31 has ended; starting"
run1() { # NEW Q BASE KIND P X
  local NEW=$1 Q=$2 BASE=$3 KIND=$4 P=$5 X=$6 rc n i RUN PYF
  [ -e "$R/$NEW/T1_DONE" ] && { log "$NEW already done"; return 0; }
  [ -e "$R/$NEW" ] && { log "SKIP $NEW: output folder exists without T1_DONE"; return 1; }
  [ -f "$W/runs/$Q/$BASE/ckpt_last.pt" ] || { log "SKIP $NEW: no checkpoint $Q/$BASE"; return 1; }
  if [ "$KIND" = m ]; then RUN=$D/pod_margin_run.sh; PYF=v8f_t1_eval_rule_margin.py; else RUN=$D/pod_qcal_run.sh; PYF=v8f_t1_eval_rule_qcal.py; X=$D/$X; [ -f "$X" ] || { log "SKIP $NEW: no map $X"; return 1; }; fi
  for i in $(seq 1 240); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ "${n:-0}" -lt 3 ] && break
    sleep 30
  done
  DR_FLOOR=cap-only DR_MIX="head head zero $P $X" DR_RULEPY=$PYF bash "$RUN" "$NEW" "$Q" "$BASE" mix > "$D/run_$NEW.out" 2>&1; rc=$?
  log "$NEW (kind $KIND, P $P, X $X, closing zero, on $Q/$BASE) rc $rc : $(tail -n 1 "$D/run_$NEW.out" | cut -c1-160)"
  return $rc
}
bad=0; tot=0
while read -r NEW Q BASE KIND P X; do
  case "$NEW" in ''|\#*) continue;; esac
  tot=$((tot+1))
  run1 "$NEW" "$Q" "$BASE" "$KIND" "$P" "$X" || bad=$((bad+1))
done < "$D/noclose_arms.txt"
nd=$(ls -d $R/*[RQ]Z_s[01]/T1_DONE 2>/dev/null | wc -l)
log "ZZ NOCLOSE QUEUE DONE: $nd of $tot rollouts with T1_DONE, $bad failed"
echo "$nd" > "$D/NOCLOSE_DONE.txt"
