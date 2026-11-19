#!/usr/bin/env bash
# pod_qhead_dose_inputs.sh : runs ON THE POD under nohup, started by the lead. Note PREREG_V8F_PROOF_ADD42_N2 section 3 (dose step, LSD type). Fails closed. DRY=1 prints every command only
# (no check, no wait, no marker, no run).
# Per checkpoint ARM (LSD_s0, LSD_s1; queue leadsup_b): the level-1 states of manifest positions 0 .. 4571 (extract_states.py --lo 0 --hi 4572) into $OUT/states/<ARM>_dose_train.npz (new
# names; no banked file is touched), then the lead read of the same rows (pod_lead_apply.py) into <ARM>_dose_train_lead.npz (+ .head.npz + .json). Before them, once, the row layout and the
# ego kinematics of the same 4,572 clips (pod_dose_layout.py) into $OUT/layout/ (the label pack is built from it on the dev box). Tier T0 (input preparation; no model is trained).
# Before EVERY step a SLOT WAIT: fewer than 3 GPU compute processes (nvidia-smi --query-compute-apps=pid) AND more than 28 GB of the container limit free (memory.max minus the anon line of
# memory.stat), both on two probes 45 s apart; poll every 60 s for at most 6 h, then a STOP marker. No process-name test anywhere (no pgrep, pkill, ps | grep). No rm, no mv.
# A second start is refused: $RUNF/QHEAD_DOSE_INPUTS_STARTED.txt is created once, atomically (noclobber). End markers in $RUNF: QHEAD_DOSE_INPUTS_DONE.txt (count of arms complete) or
# QHEAD_DOSE_INPUTS_STOP.txt (reason). Last log line: ZZ QHEAD-DOSE INPUTS DONE <n>   or   ZZ QHEAD-DOSE INPUTS STOP: <reason>.
# Env: QD_W (/workspace/v8proof) QD_PROBE ($QD_W/probe: extract_states.py, pod_lead_apply.py and the banked states) QD_TOOLS ($QD_W/qhead_dose/tools: pod_dose_layout.py)
#      QD_TREE ($QD_W/tree_leadsup) QD_CACHE ($QD_W/data/cache_pod) QD_OUT ($QD_W/qhead_dose) QD_RUNQ ($QD_W/runs/leadsup_b) QD_ARMS ("LSD_s0 LSD_s1") QD_HI (4572) QD_PY QD_NICE
#      QD_EXTRACT_ARGS (extra arguments appended to the extract_states.py command, default none)
#      QD_MIN_FREE_GB (28) QD_MAX_GPU_JOBS (3) QD_PROBE_GAP_S (45) QD_POLL_S (60) QD_MAX_WAIT_S (21600) QD_CG_MAX QD_CG_STAT DRY
set -u
W=${QD_W:-/workspace/v8proof}
PROBE=${QD_PROBE:-$W/probe}
OUT=${QD_OUT:-$W/qhead_dose}
TOOLS=${QD_TOOLS:-$OUT/tools}
TREE=${QD_TREE:-$W/tree_leadsup}
CACHE=${QD_CACHE:-$W/data/cache_pod}
RUNQ=${QD_RUNQ:-$W/runs/leadsup_b}
ARMS=${QD_ARMS:-"LSD_s0 LSD_s1"}
HI=${QD_HI:-4572}
PY=${QD_PY:-/workspace/venv/bin/python}
NICE=${QD_NICE-nice -n 19 ionice -c3}
EXARGS=${QD_EXTRACT_ARGS:-}
MIN_FREE_GB=${QD_MIN_FREE_GB:-28}
MAX_GPU_JOBS=${QD_MAX_GPU_JOBS:-3}
PROBE_GAP_S=${QD_PROBE_GAP_S:-45}
POLL_S=${QD_POLL_S:-60}
MAX_WAIT_S=${QD_MAX_WAIT_S:-21600}
CG_MAX=${QD_CG_MAX:-/sys/fs/cgroup/memory.max}
CG_STAT=${QD_CG_STAT:-/sys/fs/cgroup/memory.stat}
DRY=${DRY:-0}
RUNF=$OUT/run
LOG=$RUNF/dose_inputs.log
ST=$OUT/states
LAY=$OUT/layout/train_layout_0_${HI}.npz
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} PYTHONIOENCODING=utf-8

say() {
  local line="[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] qhead-dose: $*"
  echo "$line"
  if [ "$DRY" != 1 ] && [ -d "$RUNF" ]; then echo "$line" >> "$LOG"; fi
}
stop() {  # reason
  say "ZZ QHEAD-DOSE INPUTS STOP: $1"
  if [ "$DRY" != 1 ]; then mkdir -p "$RUNF"; printf 'QHEAD_DOSE_INPUTS_STOP %s\n%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" > "$RUNF/QHEAD_DOSE_INPUTS_STOP.txt"; fi
  exit 4
}
refuse() { echo "REFUSED: $*"; exit 2; }

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
  if [ "$DRY" = 1 ]; then echo "DRY: slot wait before $1 (fewer than $MAX_GPU_JOBS GPU compute processes and more than $MIN_FREE_GB GB free, two probes ${PROBE_GAP_S} s apart, poll ${POLL_S} s, at most ${MAX_WAIT_S} s)"; return 0; fi
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
  for m in QHEAD_DOSE_INPUTS_DONE.txt QHEAD_DOSE_INPUTS_STOP.txt QHEAD_DOSE_INPUTS_STARTED.txt; do
    [ -e "$RUNF/$m" ] && refuse "$RUNF/$m exists (a second start is refused; the lead moves the marker if a restart is meant)"
  done
  mkdir -p "$RUNF" "$ST" "$OUT/layout" || refuse "cannot create folders under $OUT"
  ( set -o noclobber; printf 'QHEAD_DOSE_INPUTS_STARTED %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUNF/QHEAD_DOSE_INPUTS_STARTED.txt" ) 2>/dev/null || refuse "another start created $RUNF/QHEAD_DOSE_INPUTS_STARTED.txt first"
  [ -f "$PROBE/extract_states.py" ] || stop "missing $PROBE/extract_states.py"
  [ -f "$PROBE/pod_lead_apply.py" ] || stop "missing $PROBE/pod_lead_apply.py"
  [ -f "$TOOLS/pod_dose_layout.py" ] || stop "missing $TOOLS/pod_dose_layout.py"
  [ -d "$CACHE/train/clips" ] || stop "missing token cache $CACHE/train/clips"
  [ -x "$PY" ] || stop "no python $PY"
  for ARM in $ARMS; do
    [ -f "$RUNQ/$ARM/ckpt_last.pt" ] || stop "missing checkpoint $RUNQ/$ARM/ckpt_last.pt"
    [ -f "$PROBE/states/${ARM}_train.npz.json" ] || stop "missing banked record $PROBE/states/${ARM}_train.npz.json"
  done
  say "start; tools sha256 layout $(sha256sum "$TOOLS/pod_dose_layout.py" | cut -c1-16) extract_states $(sha256sum "$PROBE/extract_states.py" | cut -c1-16) lead_apply $(sha256sum "$PROBE/pod_lead_apply.py" | cut -c1-16); positions 0..$((HI - 1)); arms: $ARMS"
fi

# runs one command; a failure is a STOP (never a silent continue)
step() {  # label, logfile, command...
  local label=$1 lf=$2; shift 2
  if [ "$DRY" = 1 ]; then echo "DRY: $* > $lf 2>&1"; return 0; fi
  say "run: $label"
  "$@" > "$lf" 2>&1 || stop "$label failed: $(tail -n 2 "$lf" | cut -c1-200 | tr '\n' ' ')"
  say "$label: $(tail -n 1 "$lf" | cut -c1-220)"
}
sha_of() { grep -o '"ckpt_sha256": "[0-9a-f]*"' "$1" 2>/dev/null | head -n 1 | cut -d'"' -f4; }

# ---- 1. the layout (once)
slot_wait "layout"
if [ "$DRY" != 1 ] && [ -f "$LAY.json" ]; then
  say "layout already written: $LAY"
else
  [ "$DRY" != 1 ] && [ -e "$LAY" ] && stop "$LAY exists without its record (partial file; the lead moves it)"
  step "layout" "$OUT/layout/layout.log" "$PY" "$TOOLS/pod_dose_layout.py" --tree "$TREE" --cache "$CACHE" --split train --lo 0 --hi "$HI" --out "$LAY"
fi

# ---- 2. per checkpoint: states, then the lead read
ndone=0
for ARM in $ARMS; do
  S="$ST/${ARM}_dose_train.npz"
  L="$ST/${ARM}_dose_train_lead.npz"
  if [ "$DRY" != 1 ] && [ -f "$S.json" ] && [ -f "$L.json" ]; then say "$ARM: states and lead read already written"; ndone=$((ndone+1)); continue; fi
  if [ "$DRY" != 1 ] && [ -f "$S.json" ]; then
    say "$ARM: states already written"
  else
    [ "$DRY" != 1 ] && [ -e "$S" ] && stop "$S exists without its record (partial file; the lead moves it)"
    slot_wait "states $ARM"
    step "states $ARM" "$ST/${ARM}_dose_train.log" $NICE "$PY" "$PROBE/extract_states.py" --tree "$TREE" --ckpt "$RUNQ/$ARM/ckpt_last.pt" --cache "$CACHE" --split train --lo 0 --hi "$HI" --out "$S" $EXARGS
    if [ "$DRY" != 1 ]; then
      grep -q "\"n_clips\": $HI" "$S.json" && grep -q '"lo": 0,' "$S.json" && grep -q "\"hi\": $HI" "$S.json" || stop "$ARM: record $S.json does not say lo 0, hi $HI, n_clips $HI"
      [ -n "$(sha_of "$S.json")" ] && [ "$(sha_of "$S.json")" = "$(sha_of "$PROBE/states/${ARM}_train.npz.json")" ] || stop "$ARM: checkpoint sha256 of the new record differs from the banked record"
    fi
  fi
  [ "$DRY" != 1 ] && [ -e "$L" ] && stop "$L exists without its record (partial file; the lead moves it)"
  slot_wait "lead read $ARM"
  step "lead read $ARM" "$ST/${ARM}_dose_train_lead.log" $NICE "$PY" "$PROBE/pod_lead_apply.py" --tree "$TREE" --ckpt "$RUNQ/$ARM/ckpt_last.pt" --states "$S" --out "$L"
  if [ "$DRY" != 1 ]; then
    [ -f "$L.json" ] && [ -f "$L.head.npz" ] || stop "$ARM: lead read wrote no record or head file"
    ndone=$((ndone+1))
  fi
done

if [ "$DRY" = 1 ]; then echo "DRY: at the end QHEAD_DOSE_INPUTS_DONE.txt in $RUNF holds the count of arms complete; last line ZZ QHEAD-DOSE INPUTS DONE <n>"; exit 0; fi
say "ZZ QHEAD-DOSE INPUTS DONE $ndone"
echo "$ndone" > "$RUNF/QHEAD_DOSE_INPUTS_DONE.txt"
