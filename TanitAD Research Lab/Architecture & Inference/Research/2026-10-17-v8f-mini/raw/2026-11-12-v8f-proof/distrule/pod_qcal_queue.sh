#!/usr/bin/env bash
# pod_qcal_queue.sh : runs ON THE POD (nohup setsid), started by the lead. Card PREREG_V8F_PROOF_ADD31 (QCAL). Evaluation only, it trains nothing.
# The vision-fed distance rule of card ADD26 (note N1, cap-only) with a presence threshold P and the calibrated gap map of the checkpoint, one rollout at a time and
# never as a fourth GPU job. Arms from qcal_arms.txt (lines: NEW QUEUE BASE P MAPFILE, the map relative to this folder).
#   control q1 : LSQ0_s0 (P 0.5, identity map) must give arrays equal to the banked head-fed dump LSRH_s0
# It first waits for the queue of card ADD30 to end (MARGIN_DONE.txt or MARGIN_STOP.txt).
set -u
W=/workspace/v8proof; D=$W/distrule; R=$W/runs/distrule/t1; L=$D/qcal_queue.log
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$L"; }
[ -e "$D/QCAL_DONE.txt" ] && { echo "REFUSED: $D/QCAL_DONE.txt exists"; exit 2; }
[ -f "$D/v8f_t1_eval_rule_qcal.py" ] && [ -f "$D/qcal_map.py" ] && [ -f "$D/v8f_t1_eval_rule_mix.py" ] || { echo "REFUSED: tool files missing"; exit 2; }
[ -f "$D/qcal_arms.txt" ] && [ -f "$D/qcal/identity.json" ] || { echo "REFUSED: no qcal_arms.txt or identity map"; exit 2; }
[ -f "$D/pod_ruleattr_run.sh" ] || { echo "REFUSED: no pod_ruleattr_run.sh to derive from"; exit 2; }
RUN=$D/pod_qcal_run.sh
if [ ! -f "$RUN" ]; then
  sed -e 's|ARGV=(--tree "$TREE" --mix-presence "$1" --mix-gap "$2" --mix-closing "$3")|ARGV=(--head-p-min "$4" --gap-map "$5" --tree "$TREE" --mix-presence "$1" --mix-gap "$2" --mix-closing "$3")|' "$D/pod_ruleattr_run.sh" > "$RUN"
fi
[ "$(grep -c -- '--gap-map' "$RUN")" = 1 ] && bash -n "$RUN" || { echo "REFUSED: the derived run script is not as expected"; exit 2; }
log "queue start; run script sha256 $(sha256sum "$RUN" | cut -c1-16) tool $(sha256sum "$D/v8f_t1_eval_rule_qcal.py" | cut -c1-16) map code $(sha256sum "$D/qcal_map.py" | cut -c1-16) arms $(sha256sum "$D/qcal_arms.txt" | cut -c1-16)"
stop() { log "ZZ QCAL STOP: $1"; echo "$1" > "$D/QCAL_STOP.txt"; exit 4; }
for i in $(seq 1 720); do
  [ -e "$D/MARGIN_DONE.txt" ] || [ -e "$D/MARGIN_STOP.txt" ] && break
  sleep 30
done
[ -e "$D/MARGIN_DONE.txt" ] || [ -e "$D/MARGIN_STOP.txt" ] || stop "WAIT-MARGIN (6 h)"
log "the queue of card ADD30 has ended; starting"
run1() { # NEW Q BASE P MAP
  local NEW=$1 Q=$2 BASE=$3 P=$4 MAP=$5 rc n i
  [ -e "$R/$NEW/T1_DONE" ] && { log "$NEW already done"; return 0; }
  [ -e "$R/$NEW" ] && { log "SKIP $NEW: output folder exists without T1_DONE"; return 1; }
  [ -f "$W/runs/$Q/$BASE/ckpt_last.pt" ] || { log "SKIP $NEW: no checkpoint $Q/$BASE"; return 1; }
  [ -f "$D/$MAP" ] || { log "SKIP $NEW: no map $MAP"; return 1; }
  for i in $(seq 1 240); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ "${n:-0}" -lt 3 ] && break
    sleep 30
  done
  DR_FLOOR=cap-only DR_MIX="head head head $P $D/$MAP" DR_RULEPY=v8f_t1_eval_rule_qcal.py bash "$RUN" "$NEW" "$Q" "$BASE" mix > "$D/run_$NEW.out" 2>&1; rc=$?
  log "$NEW (P $P, map $MAP sha256 $(sha256sum "$D/$MAP" | cut -c1-16) on $Q/$BASE) rc $rc : $(tail -n 1 "$D/run_$NEW.out" | cut -c1-160)"
  return $rc
}
cmp() { /workspace/venv/bin/python "$D/cmp_dumps.py" "$R/$1/dump_$1_m10_causal_heads" "$R/$2/dump_$2_m10_causal_heads" 2>&1 | tail -n 1; }
# control q1
run1 LSQ0_s0 leadsup_a LS_s0 0.5 qcal/identity.json || stop "Q1-RUN"
c1=$(cmp LSQ0_s0 LSRH_s0); log "q1 tool at P 0.5 with the identity map against the head-fed dump: $c1"
echo "$c1" | grep -q 'differing=0 ' || stop "Q1"
log "control q1 passed; q5 lines: $(grep -h 'q5 unit check' "$D/run_LSQ0_s0.out" | cut -c1-200)"
bad=0; tot=0
while read -r NEW Q BASE P MAP; do
  case "$NEW" in ''|\#*) continue;; esac
  tot=$((tot+1))
  run1 "$NEW" "$Q" "$BASE" "$P" "$MAP" || bad=$((bad+1))
done < "$D/qcal_arms.txt"
nd=$(ls -d $R/*QC_s[01]/T1_DONE 2>/dev/null | wc -l)
log "ZZ QCAL QUEUE DONE: $nd of $tot rollouts with T1_DONE, $bad failed"
echo "$nd" > "$D/QCAL_DONE.txt"
