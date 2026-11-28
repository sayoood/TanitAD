#!/bin/bash
# pod_boxread_queue.sh -- BOXREAD-T0 detector passes (card ADD127), pod, DETACHED, RESUMABLE. No rm, no mv, no kill anywhere.
#   stages (STAGES, default "smoke wide tele"):
#     smoke : 3 clips per camera into $OUT/smoke, then the projected GPU hours of both full passes from the measured seconds per frame;
#             above $MAX_H (default 1.0) it writes BOXREAD_STOP.txt and exits 4
#     wide  : the full wide pass into $OUT/wide        tele : the full tele pass into $OUT/tele
# Markers in $OUT: BOXREAD_START.txt (written with set -C; a start marker that exists refuses a new start unless RESUME=1),
#   BOXREAD_DONE.txt, BOXREAD_STOP.txt.  Log: $OUT/queue.log.  The last line printed is  ZZ BOXREAD-QUEUE <stage> DONE rc <rc> <time>
# DRY=1 prints every command as a "CMD:" line and starts nothing (no folder, no marker).
# Start:  nohup setsid env OUT=/workspace/v8proof/boxread/out bash /workspace/v8proof/boxread/kit/det/pod_boxread_queue.sh > /dev/null 2>&1 < /dev/null &
set -u

# ---- settings (every one from the environment; the defaults marked SET-ME are guesses of the pod layout: check them) ----
PY=${PY:-/workspace/venv/bin/python}                                   # SET-ME: python of the pod venv (torch + torchvision + av)
KIT=${KIT:-/workspace/v8proof/boxread/kit/det}                         # SET-ME: folder holding box_detect.py
DET=${DET:-$KIT/box_detect.py}                                         # the detector script (a test points this at a stand-in)
PROJ=${PROJ:-$KIT/box_detect.py}                                       # the script that prints PROJECT_H
OUT=${OUT:-/workspace/v8proof/boxread/out}                             # SET-ME: output folder of this card
WEIGHTS=${WEIGHTS:-/workspace/v8proof/boxread/fetch/fasterrcnn_resnet50_fpn_v2_coco.pth}   # SET-ME: the local weights file
SHA_PREFIX=${SHA_PREFIX:-dd69338a}                                     # the card's published file-name prefix
ROOT_WIDE=${ROOT_WIDE:-/workspace/v8proof/cropread/stage}              # SET-ME: stage root of the wide clips (ADD99)
ROOT_TELE=${ROOT_TELE:-/workspace/v8proof/telegap/stage}               # SET-ME: stage root of the tele clips (ADD122, tele_stage.py)
LIST_EVAL=${LIST_EVAL:-/workspace/v8proof/cropread/enc_out/lists/eval141.txt}            # SET-ME: the 141 evaluation clips
LIST_TRAIN=${LIST_TRAIN:-/workspace/v8proof/cropread/enc_out/lists/train_final.txt}      # SET-ME: the 500 train clips
WIDE_SIZE=${WIDE_SIZE:-32}; WIDE_PX=${WIDE_PX:-768}; WIDE_AZ=${WIDE_AZ:-0}; WIDE_EL=${WIDE_EL:--6}      # card section 2
TELE_SIZE=${TELE_SIZE:-16}; TELE_PX=${TELE_PX:-1024}; TELE_AZ=${TELE_AZ:-0}; TELE_EL=${TELE_EL:-0}     # card section 2 (centre of the tele camera)
SCORE_KEEP=${SCORE_KEEP:-0.3}
BATCH=${BATCH:-8}
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-2}                                                  # clip preparation threads, at most 4
SMOKE_CLIPS=${SMOKE_CLIPS:-3}
MAX_H=${MAX_H:-1.0}                                                    # stop rule, GPU hours of both full passes (card section 6)
MIN_FREE_GB=${MIN_FREE_GB:-28}                                         # container memory floor
CG_DIR=${CG_DIR:-/sys/fs/cgroup}
STAGES=${STAGES:-smoke wide tele}
DRY=${DRY:-0}
RESUME=${RESUME:-0}
FORK_STATE_FILE=${FORK_STATE_FILE-/workspace/FORK_STATE.txt}             # SET-ME: the shared pod state file (lines "BUSY <what> <UTC>" / "IDLE <UTC>"); set it EMPTY to switch off
BOX_DIR=${BOX_DIR:-/workspace/v8proof/cropread/enc_out/box}            # SET-ME: folder of <clip>.npz box files with has_lead (ADD99 lead_box.py output)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PAI_DECODE_THREADS=1
export TORCH_HOME=${TORCH_HOME:-$OUT/torch_home_unused}                # nothing is downloaded; this keeps the default cache path unused

START=$OUT/BOXREAD_START.txt
DONE=$OUT/BOXREAD_DONE.txt
STOPM=$OUT/BOXREAD_STOP.txt
LOG=$OUT/queue.log
T0=$(date -u +%s)

det_cmd() {   # camera out-dir [extra args]
  local cam=$1 od=$2; shift 2
  if [ "$cam" = wide ]; then
    echo "$PY $DET --camera wide --list $LIST_EVAL --list $LIST_TRAIN --root $ROOT_WIDE --weights $WEIGHTS --weights-sha-prefix $SHA_PREFIX --out-dir $od --win-size-deg $WIDE_SIZE --win-px $WIDE_PX --az-left-deg=$WIDE_AZ --el-up-deg=$WIDE_EL --score-keep $SCORE_KEEP --batch $BATCH --device $DEVICE --workers $WORKERS --box-dir $BOX_DIR --only-lead-frames $*"
  else
    echo "$PY $DET --camera tele --list $LIST_EVAL --list $LIST_TRAIN --root $ROOT_WIDE --tele-root $ROOT_TELE --weights $WEIGHTS --weights-sha-prefix $SHA_PREFIX --out-dir $od --win-size-deg $TELE_SIZE --win-px $TELE_PX --az-left-deg=$TELE_AZ --el-up-deg=$TELE_EL --score-keep $SCORE_KEEP --batch $BATCH --device $DEVICE --workers $WORKERS --box-dir $BOX_DIR --only-lead-frames $*"
  fi
}

n_clips() {   # distinct clips in both list files
  cat "$LIST_EVAL" "$LIST_TRAIN" 2>/dev/null | awk 'NF{c=(NF==2 && $1 ~ /^[0-9]+$/)?$2:$1; if(!s[c]++)n++} END{print n+0}'
}

memfree_gb() {   # container memory free: memory.max minus the anon line of memory.stat (never `free`, which shows the host)
  local mx an
  mx=$(cat "$CG_DIR/memory.max" 2>/dev/null)
  an=$(grep -E '^anon ' "$CG_DIR/memory.stat" 2>/dev/null | cut -d' ' -f2)
  if [ -z "$mx" ] || [ "$mx" = "max" ] || [ -z "$an" ]; then echo 9999; else echo $(( (mx - an) / 1000000000 )); fi
}

# ------------------------------------------------------------------ DRY: print, start nothing
if [ "$DRY" = "1" ]; then
  echo "DRY: OUT=$OUT STAGES=$STAGES MAX_H=$MAX_H floor=${MIN_FREE_GB}GB threads<=4 (nothing is created or started)"
  for st in $STAGES; do
    case "$st" in
      smoke)
        echo "CMD: [smoke] $(det_cmd wide "$OUT/smoke" --limit-clips "$SMOKE_CLIPS")"
        echo "CMD: [smoke] $(det_cmd tele "$OUT/smoke" --limit-clips "$SMOKE_CLIPS")"
        echo "CMD: [smoke] $PY $PROJ --count-lead-frames --box-dir $BOX_DIR --list $LIST_EVAL --list $LIST_TRAIN"
        echo "CMD: [smoke] $PY $PROJ --project $OUT/smoke/wide --project-frames <selected lead frames of the lists>"
        echo "CMD: [smoke] $PY $PROJ --project $OUT/smoke/tele --project-frames <selected lead frames of the lists>   (sum above MAX_H=$MAX_H: STOP, exit 4)" ;;
      wide) echo "CMD: [wide] $(det_cmd wide "$OUT")" ;;
      tele) echo "CMD: [tele] $(det_cmd tele "$OUT")" ;;
      *) echo "refused: unknown stage $st"; exit 2 ;;
    esac
  done
  echo "ZZ BOXREAD-QUEUE dry DONE rc 0 $(date -u +%FT%TZ)"
  exit 0
fi

# ------------------------------------------------------------------ refusals (before anything is created)
for st in $STAGES; do
  case "$st" in smoke|wide|tele) ;; *) echo "refused: unknown stage $st"; exit 2 ;; esac
done
mf=$(memfree_gb)
if [ "$mf" -lt "$MIN_FREE_GB" ]; then
  echo "refused: container memory free ${mf} GB < MIN_FREE_GB ${MIN_FREE_GB}"
  exit 2
fi
if [ -e "$START" ] && [ "$RESUME" != "1" ]; then
  echo "refused: $START exists (a run was started; set RESUME=1 to continue it)"
  exit 3
fi
mkdir -p "$OUT" || { echo "refused: cannot create $OUT"; exit 4; }
say() { echo "[queue $(date -u +%FT%TZ)] $*" >> "$LOG"; }
if [ -e "$START" ]; then
  echo "$T0 $(date -u +%FT%TZ) pid $$ resume" >> "$START"
else
  ( set -C; echo "$T0 $(date -u +%FT%TZ) pid $$" > "$START" ) || { echo "refused: cannot write the START marker"; exit 3; }
fi
state() {     # the pod's state file: one appended line, the last line counts; FORK_STATE_FILE empty = off
  [ -z "$FORK_STATE_FILE" ] && return 0
  if [ "$1" = BUSY ]; then echo "BUSY box-read detector passes of the v8proof fork $(date -u +%FT%TZ)" >> "$FORK_STATE_FILE" 2>/dev/null
  else echo "IDLE $(date -u +%FT%TZ)" >> "$FORK_STATE_FILE" 2>/dev/null; fi
  return 0
}
GPU_BUSY=0
finish() {    # stage rc [marker text]
  local st=$1 rc=$2 line
  if [ "$GPU_BUSY" = 1 ]; then state IDLE; GPU_BUSY=0; fi
  line="ZZ BOXREAD-QUEUE $st DONE rc $rc $(date -u +%FT%TZ)"
  say "$line"
  if [ "$rc" = 0 ]; then
    [ "${3:-}" = final ] && echo "$line" > "$DONE"
  else
    echo "${3:-stage $st rc $rc}" > "$STOPM"
  fi
  echo "$line"
  exit "$rc"
}
say "start: STAGES=$STAGES MAX_H=$MAX_H floor=${MIN_FREE_GB}GB free=${mf}GB WORKERS=$WORKERS"

last=""
for st in $STAGES; do last=$st; done
for st in $STAGES; do
  say "stage $st begin"
  state BUSY; GPU_BUSY=1                                               # every stage is a GPU stage; the last line of the state file counts
  case "$st" in
    smoke)
      for cam in wide tele; do
        $(det_cmd $cam "$OUT/smoke" --limit-clips "$SMOKE_CLIPS") >> "$LOG" 2>&1
        rc=$?
        [ $rc -ne 0 ] && finish smoke $rc "smoke $cam rc $rc (see queue.log)"
      done
      n=$(n_clips)
      cnt=$($PY $PROJ --count-lead-frames --box-dir "$BOX_DIR" --list "$LIST_EVAL" --list "$LIST_TRAIN" | grep -E '^LEAD_FRAMES ' | tail -1)
      nf=$(echo "$cnt" | awk '{print $2}'); nm=$(echo "$cnt" | awk '{print $4}')
      if [ -z "$nf" ] || [ "${nm:-1}" != "0" ]; then
        finish smoke 4 "stop rule: cannot count the selected frames (box files missing in $BOX_DIR: ${nm:-unknown})"
      fi
      hw=$($PY $PROJ --project "$OUT/smoke/wide" --project-frames "$nf" | grep -E '^PROJECT_H ' | tail -1 | cut -d' ' -f2)
      ht=$($PY $PROJ --project "$OUT/smoke/tele" --project-frames "$nf" | grep -E '^PROJECT_H ' | tail -1 | cut -d' ' -f2)
      say "projection for $n clips, $nf selected (lead) frames: wide ${hw:-nan} GPU-h, tele ${ht:-nan} GPU-h, MAX_H $MAX_H"
      echo "projected GPU hours of the full passes: wide ${hw:-nan}, tele ${ht:-nan}, both $(awk -v a="${hw:-nan}" -v b="${ht:-nan}" 'BEGIN{printf "%.3f", a+b}') (limit $MAX_H)"
      if [ -z "${hw:-}" ] || [ -z "${ht:-}" ] || [ "${hw:-}" = nan ] || [ "${ht:-}" = nan ]; then
        over=1
      else
        over=$(awk -v a="$hw" -v b="$ht" -v m="$MAX_H" 'BEGIN{ print ((a+b) > m) ? 1 : 0 }')
      fi
      if [ "$over" != "0" ]; then
        finish smoke 4 "stop rule: projected GPU hours wide ${hw:-nan} + tele ${ht:-nan} above MAX_H $MAX_H (or no measurement)"
      fi ;;
    wide|tele)
      $(det_cmd $st "$OUT") >> "$LOG" 2>&1
      rc=$?
      [ $rc -ne 0 ] && finish "$st" $rc "$st rc $rc (see queue.log)" ;;
  esac
  say "stage $st end"
  if [ "$st" = "$last" ]; then finish "$st" 0 final; fi
done
finish none 0 final
