#!/usr/bin/env bash
# pod_margin_queue.sh : runs ON THE POD (nohup setsid), started by the lead. Card PREREG_V8F_PROOF_ADD30 (MARGIN). Evaluation only, it trains nothing.
# The vision-fed distance rule of card ADD26 (note N1, cap-only) with a presence threshold P and a gap margin M on the head read, one rollout at a time and never
# as a fourth GPU job. The arms and their constants come from margin_arms.txt (written on the dev box from margin/CALIB.json, lines: NEW QUEUE BASE P M).
#   control g1 : LSM0_s0 (P 0.5, M 0) must give arrays equal to the banked head-fed dump LSRH_s0
# It first waits for the queue of card ADD29 to end (RULEATTR_DONE.txt or RULEATTR_STOP.txt), so that the two queues never run a rollout at the same time.
set -u
W=/workspace/v8proof; D=$W/distrule; R=$W/runs/distrule/t1; L=$D/margin_queue.log
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$L"; }
[ -e "$D/MARGIN_DONE.txt" ] && { echo "REFUSED: $D/MARGIN_DONE.txt exists"; exit 2; }
[ -f "$D/v8f_t1_eval_rule_margin.py" ] && [ -f "$D/v8f_t1_eval_rule_mix.py" ] || { echo "REFUSED: wrapper files missing"; exit 2; }
[ -f "$D/margin_arms.txt" ] || { echo "REFUSED: no margin_arms.txt"; exit 2; }
[ -f "$D/pod_ruleattr_run.sh" ] || { echo "REFUSED: no pod_ruleattr_run.sh to derive from"; exit 2; }
RUN=$D/pod_margin_run.sh
if [ ! -f "$RUN" ]; then
  sed -e 's|ARGV=(--tree "$TREE" --mix-presence "$1" --mix-gap "$2" --mix-closing "$3")|ARGV=(--head-p-min "$4" --gap-margin-m "$5" --tree "$TREE" --mix-presence "$1" --mix-gap "$2" --mix-closing "$3")|' "$D/pod_ruleattr_run.sh" > "$RUN"
fi
[ "$(grep -c -- '--head-p-min' "$RUN")" = 1 ] && bash -n "$RUN" || { echo "REFUSED: the derived run script is not as expected"; exit 2; }
log "queue start; run script sha256 $(sha256sum "$RUN" | cut -c1-16) wrapper $(sha256sum "$D/v8f_t1_eval_rule_margin.py" | cut -c1-16) arms $(sha256sum "$D/margin_arms.txt" | cut -c1-16)"
stop() { log "ZZ MARGIN STOP: $1"; echo "$1" > "$D/MARGIN_STOP.txt"; exit 4; }
for i in $(seq 1 480); do
  [ -e "$D/RULEATTR_DONE.txt" ] || [ -e "$D/RULEATTR_STOP.txt" ] && break
  sleep 30
done
[ -e "$D/RULEATTR_DONE.txt" ] || [ -e "$D/RULEATTR_STOP.txt" ] || stop "WAIT-RULEATTR (4 h)"
log "the queue of card ADD29 has ended; starting"
run1() { # NEW Q BASE P M
  local NEW=$1 Q=$2 BASE=$3 P=$4 M=$5 rc n i
  [ -e "$R/$NEW/T1_DONE" ] && { log "$NEW already done"; return 0; }
  [ -e "$R/$NEW" ] && { log "SKIP $NEW: output folder exists without T1_DONE"; return 1; }
  [ -f "$W/runs/$Q/$BASE/ckpt_last.pt" ] || { log "SKIP $NEW: no checkpoint $Q/$BASE"; return 1; }
  for i in $(seq 1 240); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ "${n:-0}" -lt 3 ] && break
    sleep 30
  done
  DR_FLOOR=cap-only DR_MIX="head head head $P $M" DR_RULEPY=v8f_t1_eval_rule_margin.py bash "$RUN" "$NEW" "$Q" "$BASE" mix > "$D/run_$NEW.out" 2>&1; rc=$?
  log "$NEW (P $P, M $M on $Q/$BASE) rc $rc : $(tail -n 1 "$D/run_$NEW.out" | cut -c1-160)"
  return $rc
}
cmp() { /workspace/venv/bin/python "$D/cmp_dumps.py" "$R/$1/dump_$1_m10_causal_heads" "$R/$2/dump_$2_m10_causal_heads" 2>&1 | tail -n 1; }
# control g1
run1 LSM0_s0 leadsup_a LS_s0 0.5 0 || stop "G1-RUN"
c1=$(cmp LSM0_s0 LSRH_s0); log "g1 margin tool at P 0.5, M 0 against the head-fed dump: $c1"
echo "$c1" | grep -q 'differing=0 ' || stop "G1"
log "control g1 passed"
bad=0; tot=0
while read -r NEW Q BASE P M; do
  case "$NEW" in ''|\#*) continue;; esac
  tot=$((tot+1))
  run1 "$NEW" "$Q" "$BASE" "$P" "$M" || bad=$((bad+1))
done < "$D/margin_arms.txt"
nd=$(ls -d $R/*M[CTGLHR]_s[01]/T1_DONE 2>/dev/null | wc -l)
log "ZZ MARGIN QUEUE DONE: $nd of $tot rollouts with T1_DONE, $bad failed"
echo "$nd" > "$D/MARGIN_DONE.txt"
