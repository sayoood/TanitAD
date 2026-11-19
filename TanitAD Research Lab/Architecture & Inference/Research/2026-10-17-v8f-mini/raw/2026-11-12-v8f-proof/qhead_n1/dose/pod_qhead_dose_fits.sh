#!/usr/bin/env bash
# pod_qhead_dose_fits.sh : runs ON THE POD (CPU), started by the lead after pod_qhead_dose_inputs.sh is DONE, the label pack is in place and the controls d1 and d2 passed.
# Note PREREG_V8F_PROOF_ADD42_N2 section 3: the fit of note N1 (qhead_fit_n1.py, unchanged) on the states of all 4,572 training clips: fit clips 0..3047 (--fit-clips 3048), select clips
# 3048..3809 (--select-clips 762), check clips 3810..4571 (--check-clips 762). Two fits: LSD_s0 (seed 0) and LSD_s1 (seed 1). Heads and records in $HEADS (default /workspace/v8proof/qhead_dose/heads).
# Tier T0 (a fit on logged frames, no model is trained; never driving performance). Fails closed. DRY=1 prints every command only.
# Before the first fit it refuses unless: the tool files are present and their sha256 start with the recorded prefixes (fit tool 57e1242713e0b42e, model f6ff755e3dd4e212), the pack and the
# states exist, and the control files say pass: $OUT/controls/d1_pack.json (built on the dev box, copied here) and $OUT/controls/d2_<ARM>.json (written on the pod). Before every fit: the
# container has more than 20 GB free (memory.max minus the anon line of memory.stat) on one probe; poll every 60 s for at most 2 h, then a STOP marker. No process-name test, no rm, no mv.
# A second start is refused: $RUNF/QHEAD_DOSE_FITS_STARTED.txt is created once, atomically (noclobber). End markers: QHEAD_DOSE_FITS_DONE.txt (count of fits written) or
# QHEAD_DOSE_FITS_STOP.txt (reason). Last log line: ZZ QHEAD-DOSE FITS DONE <n> of 2   or   ZZ QHEAD-DOSE FITS STOP: <reason>.
# Env: QD_W QD_OUT ($QD_W/qhead_dose) QD_TOOLS ($QD_OUT/tools: qhead_fit_n1.py qhead_model.py) QD_PACK ($QD_OUT/pack/train_pack_all.npz) QD_PY QD_MIN_FREE_GB (20) QD_POLL_S (60)
#      QD_MAX_WAIT_S (7200) QD_FIT_SHA (57e1242713e0b42e) QD_MODEL_SHA (f6ff755e3dd4e212) QD_FIT_ARGS (extra arguments) QD_CG_MAX QD_CG_STAT DRY
set -u
W=${QD_W:-/workspace/v8proof}
OUT=${QD_OUT:-$W/qhead_dose}
TOOLS=${QD_TOOLS:-$OUT/tools}
PACK=${QD_PACK:-$OUT/pack/train_pack_all.npz}
PY=${QD_PY:-/workspace/venv/bin/python}
MIN_FREE_GB=${QD_MIN_FREE_GB:-20}
POLL_S=${QD_POLL_S:-60}
MAX_WAIT_S=${QD_MAX_WAIT_S:-7200}
FIT_SHA=${QD_FIT_SHA:-57e1242713e0b42e}
MODEL_SHA=${QD_MODEL_SHA:-f6ff755e3dd4e212}
FITARGS=${QD_FIT_ARGS:-}
CG_MAX=${QD_CG_MAX:-/sys/fs/cgroup/memory.max}
CG_STAT=${QD_CG_STAT:-/sys/fs/cgroup/memory.stat}
DRY=${DRY:-0}
RUNF=$OUT/run
LOG=$RUNF/dose_fits.log
ST=$OUT/states
HEADS=$OUT/heads
CTL=$OUT/controls
ARMLINES="LSD_s0 0
LSD_s1 1"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} PYTHONIOENCODING=utf-8

say() {
  local line="[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] qhead-dose-fit: $*"
  echo "$line"
  if [ "$DRY" != 1 ] && [ -d "$RUNF" ]; then echo "$line" >> "$LOG"; fi
}
stop() {  # reason
  say "ZZ QHEAD-DOSE FITS STOP: $1"
  if [ "$DRY" != 1 ]; then mkdir -p "$RUNF"; printf 'QHEAD_DOSE_FITS_STOP %s\n%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" > "$RUNF/QHEAD_DOSE_FITS_STOP.txt"; fi
  exit 4
}
refuse() { echo "REFUSED: $*"; exit 2; }

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
mem_wait() {  # label
  if [ "$DRY" = 1 ]; then echo "DRY: memory wait before $1 (more than $MIN_FREE_GB GB of the container free, poll ${POLL_S} s, at most ${MAX_WAIT_S} s)"; return 0; fi
  SECONDS=0
  while true; do
    local g; g=$(free_gb)
    say "memory probe: container free=${g} GB (need > $MIN_FREE_GB)"
    [ "$g" -gt "$MIN_FREE_GB" ] && { say "memory free for $1"; return 0; }
    [ "$SECONDS" -ge "$MAX_WAIT_S" ] && stop "less than $MIN_FREE_GB GB free for $1 after ${SECONDS} s (limit ${MAX_WAIT_S} s)"
    sleep "$POLL_S"
  done
}

if [ "$DRY" = 1 ]; then
  echo "DRY=1: commands only (no check, no wait, no marker, no run)"
else
  for m in QHEAD_DOSE_FITS_DONE.txt QHEAD_DOSE_FITS_STOP.txt QHEAD_DOSE_FITS_STARTED.txt; do
    [ -e "$RUNF/$m" ] && refuse "$RUNF/$m exists (a second start is refused; the lead moves the marker if a restart is meant)"
  done
  mkdir -p "$RUNF" "$HEADS" || refuse "cannot create folders under $OUT"
  ( set -o noclobber; printf 'QHEAD_DOSE_FITS_STARTED %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUNF/QHEAD_DOSE_FITS_STARTED.txt" ) 2>/dev/null || refuse "another start created $RUNF/QHEAD_DOSE_FITS_STARTED.txt first"
  [ -x "$PY" ] || stop "no python $PY"
  [ -f "$TOOLS/qhead_fit_n1.py" ] && [ -f "$TOOLS/qhead_model.py" ] || stop "missing qhead_fit_n1.py or qhead_model.py in $TOOLS"
  fs=$(sha256sum "$TOOLS/qhead_fit_n1.py" | cut -c1-64); ms=$(sha256sum "$TOOLS/qhead_model.py" | cut -c1-64)
  case "$fs" in "$FIT_SHA"*) ;; *) stop "qhead_fit_n1.py sha256 $fs does not start with $FIT_SHA";; esac
  case "$ms" in "$MODEL_SHA"*) ;; *) stop "qhead_model.py sha256 $ms does not start with $MODEL_SHA";; esac
  [ -f "$PACK" ] || stop "missing label pack $PACK"
  grep -q '"pass": true' "$CTL/d1_pack.json" 2>/dev/null || stop "control d1 is not passed: no pass true in $CTL/d1_pack.json"
  while read -r ARM SEED; do
    [ -n "${ARM:-}" ] || continue
    [ -f "$ST/${ARM}_dose_train.npz" ] && [ -f "$ST/${ARM}_dose_train.npz.json" ] || stop "no dose states for $ARM in $ST"
    grep -q '"pass": true' "$CTL/d2_${ARM}.json" 2>/dev/null || stop "control d2 is not passed: no pass true in $CTL/d2_${ARM}.json"
  done <<EOF
$ARMLINES
EOF
  say "start; fit tool $(echo "$fs" | cut -c1-16) model $(echo "$ms" | cut -c1-16) pack $(sha256sum "$PACK" | cut -c1-16)"
fi

nfit=0
while read -r ARM SEED; do
  [ -n "${ARM:-}" ] || continue
  S="$ST/${ARM}_dose_train.npz"
  L="$ST/${ARM}_dose_train_lead.npz"
  EXTRA=()
  [ -f "$L" ] && EXTRA+=(--lead "$L")
  [ -f "$L.head.npz" ] && EXTRA+=(--banked-head "$L.head.npz")
  if [ "$DRY" = 1 ]; then
    mem_wait "fit $ARM"
    echo "DRY: $PY $TOOLS/qhead_fit_n1.py --arm $ARM --seed $SEED --states $S --pack $PACK --out-dir $HEADS --device cpu --fit-clips 3048 --select-clips 762 --check-clips 762 --lead $L --banked-head $L.head.npz $FITARGS > $RUNF/fit_$ARM.out"
    nfit=$((nfit+1))
    continue
  fi
  if [ -f "$HEADS/$ARM.qhead.json" ]; then say "fit $ARM already written"; nfit=$((nfit+1)); continue; fi
  mem_wait "fit $ARM"
  say "run: fit $ARM seed $SEED"
  "$PY" "$TOOLS/qhead_fit_n1.py" --arm "$ARM" --seed "$SEED" --states "$S" --pack "$PACK" --out-dir "$HEADS" --device cpu --fit-clips 3048 --select-clips 762 --check-clips 762 ${EXTRA[@]+"${EXTRA[@]}"} $FITARGS > "$RUNF/fit_$ARM.out" 2>&1 \
    || stop "fit $ARM failed: $(tail -n 2 "$RUNF/fit_$ARM.out" | cut -c1-200 | tr '\n' ' ')"
  [ -f "$HEADS/$ARM.qhead.json" ] && [ -f "$HEADS/$ARM.qhead.npz" ] || stop "fit $ARM wrote no head and record"
  say "fit $ARM: $(tail -n 1 "$RUNF/fit_$ARM.out" | cut -c1-260)"
  nfit=$((nfit+1))
done <<EOF
$ARMLINES
EOF
if [ "$DRY" = 1 ]; then echo "DRY: at the end QHEAD_DOSE_FITS_DONE.txt in $RUNF holds the count of fits written; last line ZZ QHEAD-DOSE FITS DONE <n> of 2"; exit 0; fi
say "ZZ QHEAD-DOSE FITS DONE $nfit of 2"
echo "$nfit" > "$RUNF/QHEAD_DOSE_FITS_DONE.txt"
