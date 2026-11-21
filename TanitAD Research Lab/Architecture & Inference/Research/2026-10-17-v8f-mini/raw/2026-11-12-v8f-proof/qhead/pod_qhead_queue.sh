#!/usr/bin/env bash
# pod_qhead_queue.sh : runs ON THE POD (nohup setsid), started by the lead. Card PREREG_V8F_PROOF_ADD42 (QHEAD). Fails closed. DRY=1 prints the commands only (no check, no wait, no marker, no run).
# Order: four fits of the quantile head (qhead_fit.py, CPU), the control rollout LSQHI_s0 (q_head = g_head, P 0.5; must equal the banked head-fed dump LSRH_s0), then the four rollouts, one at a time.
# Before every fit and every rollout: a SLOT WAIT = fewer than 3 GPU compute processes (nvidia-smi --query-compute-apps=pid) AND more than 28 GB of the container memory limit free
# (memory.max minus the anon line of memory.stat), both on two probes 45 s apart; poll every 60 s for at most 10 h, then QHEAD_STOP.txt. Never a fourth GPU job.
# No process-name test anywhere (no pgrep, no pkill, no ps | grep): the guards read marker files and the GPU compute-apps list. No rm: a marker that must go is moved by the lead.
# A second start is refused: $RUNF/QHEAD_STARTED.txt is created once, atomically (noclobber). End markers in $RUNF: QHEAD_DONE.txt (count of rollouts with T1_DONE) or QHEAD_STOP.txt (reason).
# Files go next to the banked tools, in $D (default /workspace/v8proof/distrule): qhead_model.py qhead_fit.py v8f_t1_eval_rule_qhead.py pod_qhead_queue.sh qhead_arms.txt train_pack.npz.
# Env: QH_W (/workspace/v8proof) QH_D ($QH_W/distrule) QH_STATES ($QH_W/qhead_states: <ARM>_train.npz + .json [+ <ARM>_train_lead.npz + .head.npz]) QH_PACK ($QH_D/train_pack.npz)
#      QH_RUNF ($QH_W/qhead_run) QH_PY QH_DEVICE (fit device, cpu) QH_FIT_ARGS QH_MIN_FREE_GB (28) QH_MAX_GPU_JOBS (3) QH_PROBE_GAP_S (45) QH_POLL_S (60) QH_MAX_WAIT_S (36000) QH_CG_MAX QH_CG_STAT DRY
# Tier T0 for the fits, T1-kin (kinematic rollout on the model's own actions, logged frames) for the rollouts; none is driving performance.
set -u
W=${QH_W:-/workspace/v8proof}
D=${QH_D:-$W/distrule}
R=${QH_R:-$W/runs/distrule/t1}
STATES=${QH_STATES:-$W/qhead_states}
PACK=${QH_PACK:-$D/train_pack.npz}
RUNF=${QH_RUNF:-$W/qhead_run}
PY=${QH_PY:-/workspace/venv/bin/python}
FITDEV=${QH_DEVICE:-cpu}
FITARGS=${QH_FIT_ARGS:-}
MIN_FREE_GB=${QH_MIN_FREE_GB:-28}
MAX_GPU_JOBS=${QH_MAX_GPU_JOBS:-3}
PROBE_GAP_S=${QH_PROBE_GAP_S:-45}
POLL_S=${QH_POLL_S:-60}
MAX_WAIT_S=${QH_MAX_WAIT_S:-36000}
CG_MAX=${QH_CG_MAX:-/sys/fs/cgroup/memory.max}
CG_STAT=${QH_CG_STAT:-/sys/fs/cgroup/memory.stat}
DRY=${DRY:-0}
ARMS=${QH_ARMS:-$D/qhead_arms.txt}
LOG=$RUNF/qhead_queue.log
HEADS=$RUNF/heads
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} PYTHONIOENCODING=utf-8

say() {
  local line="[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] qhead: $*"
  echo "$line"
  if [ "$DRY" != 1 ] && [ -d "$RUNF" ]; then echo "$line" >> "$LOG"; fi
}
stop() {  # reason
  say "ZZ QHEAD STOP: $1"
  if [ "$DRY" != 1 ]; then mkdir -p "$RUNF"; printf 'QHEAD_STOP %s\n%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" > "$RUNF/QHEAD_STOP.txt"; fi
  exit 4
}
refuse() { echo "REFUSED: $*"; exit 2; }
run() { if [ "$DRY" = 1 ]; then echo "DRY: $*"; return 0; fi; say "run: $*"; "$@"; }

gpu_jobs() {  # number of GPU compute processes; an unreadable list counts as busy
  local o rc
  o=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1); rc=$?
  if [ $rc -ne 0 ]; then echo 99; return; fi
  printf '%s\n' "$o" | grep -c .
}
free_gb() {  # free memory of the container in whole GB: memory.max minus the anon line of memory.stat
  local maxb anon freeb
  maxb=$(cat "$CG_MAX" 2>/dev/null || echo max)
  anon=$(awk '$1=="anon"{print $2}' "$CG_STAT" 2>/dev/null)
  if [ "$maxb" = max ] || [ -z "$maxb" ]; then
    freeb=$(awk '/MemAvailable/{print $2*1024}' /proc/meminfo 2>/dev/null)
  elif [ -z "$anon" ]; then
    echo 0; return
  else
    freeb=$((maxb - anon))
  fi
  freeb=${freeb:-0}
  echo $((freeb / 1073741824))
}
slot_probe() {
  local n g
  n=$(gpu_jobs); g=$(free_gb)
  say "slot probe: gpu_jobs=$n (need < $MAX_GPU_JOBS), container free=${g} GB (need > $MIN_FREE_GB)"
  [ "$n" -lt "$MAX_GPU_JOBS" ] && [ "$g" -gt "$MIN_FREE_GB" ]
}
slot_wait() {  # label: two probes PROBE_GAP_S apart must both find a slot
  if [ "$DRY" = 1 ]; then echo "DRY: slot wait before $1 (fewer than $MAX_GPU_JOBS GPU compute processes and more than $MIN_FREE_GB GB free, two probes ${PROBE_GAP_S} s apart)"; return 0; fi
  SECONDS=0
  while true; do
    if slot_probe; then
      sleep "$PROBE_GAP_S"
      if slot_probe; then say "slot free on two probes for $1"; return 0; fi
    fi
    [ "$SECONDS" -ge "$MAX_WAIT_S" ] && stop "no free slot for $1 after ${SECONDS} s (limit ${MAX_WAIT_S} s): never a fourth GPU job"
    say "waiting for a slot for $1 (${SECONDS} s of ${MAX_WAIT_S} s)"
    sleep "$POLL_S"
  done
}

if [ "$DRY" = 1 ]; then
  echo "DRY=1: commands only (no check, no wait, no marker, no run)"
else
  for m in QHEAD_DONE.txt QHEAD_STOP.txt QHEAD_STARTED.txt; do
    [ -e "$RUNF/$m" ] && refuse "$RUNF/$m exists (a second start is refused; the lead moves the marker if a restart is meant)"
  done
  mkdir -p "$RUNF" "$HEADS" || refuse "cannot create $RUNF"
  ( set -o noclobber; printf 'QHEAD_STARTED %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUNF/QHEAD_STARTED.txt" ) 2>/dev/null || refuse "another start created $RUNF/QHEAD_STARTED.txt first"
  for f in qhead_model.py qhead_fit.py v8f_t1_eval_rule_qhead.py v8f_t1_eval_rule_mix.py v8f_t1_eval_rule.py dist_rule.py cmp_dumps.py pod_ruleattr_run.sh; do
    [ -f "$D/$f" ] || stop "missing $D/$f"
  done
  [ -f "$ARMS" ] || stop "missing $ARMS"
  [ -f "$PACK" ] || stop "missing train pack $PACK"
  [ -x "$PY" ] || stop "no python $PY"
fi
RUN=$D/pod_qhead_run.sh
if [ "$DRY" = 1 ]; then
  echo "DRY: derive $RUN from $D/pod_ruleattr_run.sh (sed: the argv line gets --head-p-min \$4 \$5 \$6 in front)"
else
  if [ ! -f "$RUN" ]; then
    sed -e 's|ARGV=(--tree "$TREE" --mix-presence "$1" --mix-gap "$2" --mix-closing "$3")|ARGV=(--head-p-min "$4" $5 ${6:-} --tree "$TREE" --mix-presence "$1" --mix-gap "$2" --mix-closing "$3")|' "$D/pod_ruleattr_run.sh" > "$RUN"
  fi
  [ "$(grep -c -- '--head-p-min' "$RUN")" = 1 ] && bash -n "$RUN" || stop "the derived run script is not as expected"
  say "queue start; run script sha256 $(sha256sum "$RUN" | cut -c1-16) tool $(sha256sum "$D/v8f_t1_eval_rule_qhead.py" | cut -c1-16) fit $(sha256sum "$D/qhead_fit.py" | cut -c1-16) model $(sha256sum "$D/qhead_model.py" | cut -c1-16) arms $(sha256sum "$ARMS" | cut -c1-16)"
fi

# arms file lines: NEW QUEUE BASE SEED P  (the control line is fixed here: LSQHI_s0 on leadsup_a LS_s0, P 0.5)
ARMLINES=""
if [ "$DRY" = 1 ] && [ ! -f "$ARMS" ]; then
  ARMLINES="LSQH_s0 leadsup_a LS_s0 0 0.4
LSQH_s1 leadsup_a LS_s1 1 0.5
LSDQH_s0 leadsup_b LSD_s0 0 0.4
LSDQH_s1 leadsup_b LSD_s1 1 0.3"
else
  ARMLINES=$(grep -v '^[[:space:]]*#' "$ARMS" | grep -v '^[[:space:]]*$')
fi
[ -n "$ARMLINES" ] || stop "no arms in $ARMS"

# ---- 1. the four fits (one at a time)
nfit=0
while read -r NEW Q BASE SEED P; do
  [ -n "${NEW:-}" ] || continue
  slot_wait "fit $BASE"
  if [ "$DRY" != 1 ]; then
    [ -f "$HEADS/$BASE.qhead.json" ] && { say "fit $BASE already written"; nfit=$((nfit+1)); continue; }
    [ -f "$STATES/${BASE}_train.npz" ] && [ -f "$STATES/${BASE}_train.npz.json" ] || stop "no train states for $BASE in $STATES"
  fi
  EXTRA=()
  [ -f "$STATES/${BASE}_train_lead.npz" ] && EXTRA+=(--lead "$STATES/${BASE}_train_lead.npz")
  [ -f "$STATES/${BASE}_train_lead.npz.head.npz" ] && EXTRA+=(--banked-head "$STATES/${BASE}_train_lead.npz.head.npz")
  if [ "$DRY" = 1 ]; then
    echo "DRY: $PY $D/qhead_fit.py --arm $BASE --seed $SEED --states $STATES/${BASE}_train.npz --pack $PACK --out-dir $HEADS --device $FITDEV ${EXTRA[*]:-} $FITARGS > $RUNF/fit_$BASE.out"
  else
    say "run: fit $BASE seed $SEED"
    "$PY" "$D/qhead_fit.py" --arm "$BASE" --seed "$SEED" --states "$STATES/${BASE}_train.npz" --pack "$PACK" --out-dir "$HEADS" --device "$FITDEV" ${EXTRA[@]+"${EXTRA[@]}"} $FITARGS > "$RUNF/fit_$BASE.out" 2>&1 \
      || stop "fit $BASE failed: $(tail -n 2 "$RUNF/fit_$BASE.out" | cut -c1-200 | tr '\n' ' ')"
    say "fit $BASE: $(tail -n 1 "$RUNF/fit_$BASE.out" | cut -c1-220)"
  fi
  nfit=$((nfit+1))
done <<EOF
$ARMLINES
EOF
[ "$DRY" = 1 ] || say "$nfit fits done"

# ---- 2. the rollouts: run1 NEW QUEUE BASE P FLAGS...   (flags: --qhead FILE | --q-equals-head)
run1() {
  local NEW=$1 Q=$2 BASE=$3 P=$4 F1=$5 F2=${6:-} rc
  if [ "$DRY" = 1 ]; then
    slot_wait "rollout $NEW"
    echo "DRY: DR_FLOOR=cap-only DR_MIX=\"head head head $P $F1 $F2\" DR_RULEPY=v8f_t1_eval_rule_qhead.py bash $RUN $NEW $Q $BASE mix > $RUNF/run_$NEW.out"
    return 0
  fi
  [ -e "$R/$NEW/T1_DONE" ] && { say "$NEW already done"; return 0; }
  [ -e "$R/$NEW" ] && { say "SKIP $NEW: output folder exists without T1_DONE"; return 1; }
  [ -f "$W/runs/$Q/$BASE/ckpt_last.pt" ] || { say "SKIP $NEW: no checkpoint $Q/$BASE"; return 1; }
  slot_wait "rollout $NEW"
  DR_FLOOR=cap-only DISTRULE_DIR=$D DR_MIX="head head head $P $F1 $F2" DR_RULEPY=v8f_t1_eval_rule_qhead.py bash "$RUN" "$NEW" "$Q" "$BASE" mix > "$RUNF/run_$NEW.out" 2>&1; rc=$?
  say "$NEW (P $P, $F1 $F2 on $Q/$BASE) rc $rc : $(tail -n 1 "$RUNF/run_$NEW.out" | cut -c1-160)"
  return $rc
}
cmpd() { "$PY" "$D/cmp_dumps.py" "$R/$1/dump_$1_m10_causal_heads" "$R/$2/dump_$2_m10_causal_heads" 2>&1 | tail -n 1; }

# control h1: the control rollout against the banked head-fed dump LSRH_s0
run1 LSQHI_s0 leadsup_a LS_s0 0.5 --q-equals-head || stop "H1-RUN"
if [ "$DRY" = 1 ]; then
  echo "DRY: $PY $D/cmp_dumps.py $R/LSQHI_s0/dump_LSQHI_s0_m10_causal_heads $R/LSRH_s0/dump_LSRH_s0_m10_causal_heads  (must read 'differing=0 ')"
else
  c1=$(cmpd LSQHI_s0 LSRH_s0)
  say "h1 the control rollout (q_head = g_head, P 0.5) against the banked head-fed dump: $c1"
  echo "$c1" | grep -q 'differing=0 ' || stop "H1 (0 files differing was required)"
  say "control h1 passed"
fi

bad=0; tot=0
while read -r NEW Q BASE SEED P; do
  [ -n "${NEW:-}" ] || continue
  tot=$((tot+1))
  run1 "$NEW" "$Q" "$BASE" "$P" --qhead "$HEADS/$BASE.qhead.npz" || bad=$((bad+1))
done <<EOF
$ARMLINES
EOF
if [ "$DRY" = 1 ]; then echo "DRY: at the end QHEAD_DONE.txt in $RUNF holds the count of rollouts with T1_DONE"; exit 0; fi
nd=0
while read -r NEW Q BASE SEED P; do
  [ -n "${NEW:-}" ] || continue
  [ -e "$R/$NEW/T1_DONE" ] && nd=$((nd+1))
done <<EOF
$ARMLINES
EOF
say "ZZ QHEAD QUEUE DONE: $nd of $tot rollouts with T1_DONE, $bad failed"
echo "$nd" > "$RUNF/QHEAD_DONE.txt"
