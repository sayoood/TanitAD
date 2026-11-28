#!/bin/bash
# pod_qhead_qx_rollouts.sh -- pod_qhead_n1_rollouts.sh with the run folder and the heads folder by env (QN_RUNF, QN_HEADS) and an optional wait for the end marker
# of another rollout queue (QN_WAIT_A, QN_WAIT_B). Made by make_qf_tools.py. Used for the dose arms of note N2 section 3 (LSDQF). The text of the first form follows.
# pod_qhead_n1_rollouts.sh -- runs ON THE POD (nohup). Notes PREREG_V8F_PROOF_ADD42_N1 and _N2: the rollouts of the arm types whose two checkpoints passed control h5
# on the check clips. Arguments: one quoted group per arm "NEW Q BASE P" (for example "LSQE_s0 leadsup_a LS_s0 0.4"). An arm whose head record does not say
# h5 true is refused. One rollout at a time; before each a slot wait: fewer than 3 GPU compute processes AND more than 28 GB of the container memory limit free,
# on two probes 45 s apart (poll 60 s, at most 6 h, then a STOP marker). Never a fourth GPU job. No process-name test. Deletes nothing. A second start is refused
# by a marker. End markers in qhead_run_n1: ROLLOUTS_DONE.txt or ROLLOUTS_STOP.txt. Last line: ZZ QHEAD-N1 ROLLOUTS DONE <n> of <tot> | ZZ QHEAD-N1 ROLLOUTS STOP: <why>.
set -u
W=/workspace/v8proof; D=$W/distrule; R=$W/runs/distrule/t1; RUNF=${QN_RUNF:-$W/qhead_run_n1}; HEADS=${QN_HEADS:-$RUNF/heads}; RUN=$D/pod_qhead_run.sh
mkdir -p "$RUNF" || exit 9
PY=/workspace/venv/bin/python
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say() { echo "[$(ts)] $*"; }
stop() { say "ZZ QHEAD-N1 ROLLOUTS STOP: $*"; echo "$(ts) $*" > "$RUNF/ROLLOUTS_STOP.txt"; exit 8; }
[ -d "$HEADS" ] || { echo "no heads folder"; exit 9; }
( set -o noclobber; echo "started $(ts)" > "$RUNF/ROLLOUTS_STARTED.txt" ) 2>/dev/null || { say "REFUSED: $RUNF/ROLLOUTS_STARTED.txt exists"; exit 3; }
[ -f "$RUN" ] || stop "missing run script $RUN"
[ -f "$D/v8f_t1_eval_rule_qhead.py" ] || stop "missing rollout tool"
if [ -n "${QN_WAIT_A:-}" ]; then j=0; while ! { [ -e "$QN_WAIT_A" ] || [ -e "${QN_WAIT_B:-$QN_WAIT_A}" ]; }; do j=$((j+1)); [ $j -gt 480 ] && stop "the other rollout queue has not ended in 8 h"; sleep 60; done; say "the other rollout queue has ended"; fi
say "queue start; run script sha256 $(sha256sum "$RUN" | cut -c1-16) tool $(sha256sum "$D/v8f_t1_eval_rule_qhead.py" | cut -c1-16) arms $#"
gpu_jobs() { local o; o=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) || { echo 99; return; }; printf '%s\n' "$o" | grep -c '[0-9]'; }
freegb() { local a m; a=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat); m=$(cat /sys/fs/cgroup/memory.max); echo $(( (m - a) / 1073741824 )); }
probe() { local n g; n=$(gpu_jobs); g=$(freegb); [ "$n" -lt 3 ] && [ "$g" -gt 28 ]; }
slot_wait() {
  local t0=$SECONDS
  while true; do
    if probe; then sleep 45; if probe; then say "slot free on two probes for $1 (gpu jobs $(gpu_jobs), free $(freegb) GB)"; return 0; fi; fi
    [ $((SECONDS - t0)) -ge 21600 ] && stop "no free slot for $1 after 6 h: never a fourth GPU job"
    sleep 60
  done
}
h5_true() { "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('h5_check_coverage_in_0.15_0.25') is True else 1)" "$HEADS/$1.qhead.json"; }
tot=$#; nd=0
for grp in "$@"; do
  set -- $grp; NEW=$1; Q=$2; BASE=$3; P=$4
  [ -f "$HEADS/$BASE.qhead.npz" ] && [ -f "$HEADS/$BASE.qhead.json" ] || stop "no head for $BASE"
  h5_true "$BASE" || stop "$BASE: the head record does not say h5 true; its arm is not rolled out"
  if [ -e "$R/$NEW/T1_DONE" ]; then say "$NEW already done"; nd=$((nd+1)); continue; fi
  [ -e "$R/$NEW" ] && stop "$NEW: output folder exists without T1_DONE"
  slot_wait "rollout $NEW"
  say "rollout $NEW start (Q $Q, base $BASE, threshold $P, head $(sha256sum "$HEADS/$BASE.qhead.npz" | cut -c1-16))"
  DR_FLOOR=cap-only DISTRULE_DIR=$D DR_MIX="head head head $P --qhead $HEADS/$BASE.qhead.npz" DR_RULEPY=v8f_t1_eval_rule_qhead.py bash "$RUN" "$NEW" "$Q" "$BASE" mix > "$RUNF/run_$NEW.out" 2>&1
  rc=$?
  if [ -e "$R/$NEW/T1_DONE" ]; then nd=$((nd+1)); say "rollout $NEW rc $rc T1_DONE"; else stop "rollout $NEW rc $rc without T1_DONE: $(tail -n 2 "$RUNF/run_$NEW.out" | tr '\n' ' ' | cut -c1-240)"; fi
done
echo "$(ts) $nd of $tot" > "$RUNF/ROLLOUTS_DONE.txt"
say "ZZ QHEAD-N1 ROLLOUTS DONE $nd of $tot"
