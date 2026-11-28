#!/bin/bash
# pod_telegap_enc_queue.sh -- TELEGAP data-side queue (card ADD122), pod, DETACHED, RESUMABLE. No rm and no mv anywhere.
#   preflight -> lists -> stage tele clips (641) -> smoke encode (3 clips) -> control t2 -> full encode -> DONE.
# Markers in $ROOT: TELEGAP_ENC_START.txt (first line "<epoch seconds> <ISO time>Z pid <pid>", written with set -C),
#   TELEGAP_ENC_DONE.txt (last line "ZZ TELEGAP_ENC DONE <ISO time>Z all steps done"), TELEGAP_ENC_STOP.txt (one line why).
# Log: $ROOT/queue.log.  A START marker that exists refuses the start unless TG_RESUME=1 (then a resume line is appended).
# Environment (all optional):
#   TG_ROOT=/workspace/v8proof/telegap   TG_STOP_S=14400   TG_MAX_GB=40   TG_MIN_FREE_GB=60   TG_FREE_PATH=/workspace
#   DRY=1   print the commands as "CMD: ..." lines, create nothing
#   WORKERS=3 (encode threads)  STAGE_WORKERS=3  TG_RESUME=1
# Start (lead):  nohup setsid env WORKERS=3 bash /workspace/v8proof/telegap/kit/enc/pod_telegap_enc_queue.sh > /dev/null 2>&1 < /dev/null &
set -u
ROOT=${TG_ROOT:-/workspace/v8proof/telegap}
KIT=${TG_KIT:-$ROOT/kit/enc}
KIT99=${TG_KIT99:-/workspace/v8proof/cropread/kit/enc}
PY=${TG_PY:-/workspace/venv/bin/python}
WIDE=${TG_WIDE_STAGE:-/workspace/v8proof/cropread/stage}
SRC_LISTS=${TG_LISTS_SRC:-/workspace/v8proof/cropread/enc_out/lists}
WIDE_TOK=${TG_WIDE_TOK:-/workspace/v8proof/cropread/enc_out/tok/D}
HFDIR=${TG_HFDIR:-/workspace/v8proof/cropread/hfdl}
STOP_S=${TG_STOP_S:-14400}
MAX_GB=${TG_MAX_GB:-40}
MIN_FREE=${TG_MIN_FREE_GB:-60}
FREE_PATH=${TG_FREE_PATH:-/workspace}
WORKERS=${WORKERS:-3}
STAGE_WORKERS=${STAGE_WORKERS:-3}
DRY=${DRY:-0}
EXPECT_CLIPS=${TG_EXPECT_CLIPS:-641}
TSTAGE=$ROOT/stage
OUT=$ROOT/enc_out
SMOKE=$ROOT/smoke_enc
LISTS=$ROOT/lists
START=$ROOT/TELEGAP_ENC_START.txt
DONE=$ROOT/TELEGAP_ENC_DONE.txt
STOPM=$ROOT/TELEGAP_ENC_STOP.txt
BASEF=$ROOT/TELEGAP_ENC_BASE_KB.txt
LOG=$ROOT/queue.log
T0=$(date -u +%s)

# ------------------------------------------------------------------ refusals (before anything is created)
free_gb=$(df -Pk "$FREE_PATH" 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
free_gb=${free_gb:-0}
if [ "$free_gb" -lt "$MIN_FREE" ]; then
  echo "refused: $free_gb GB free on $FREE_PATH < $MIN_FREE GB"
  exit 2
fi
if [ "$DRY" != "1" ] && [ -e "$START" ] && [ "${TG_RESUME:-0}" != "1" ]; then
  echo "refused: $START exists (a run was started; set TG_RESUME=1 to continue it)"
  exit 3
fi

if [ "$DRY" = "1" ]; then
  run() { echo "CMD: $*"; return 0; }
  guarded() { local name=$1; shift; echo "CMD: [$name] $*"; return 0; }
  say() { :; }
  stop() { echo "STOP $*"; exit 1; }
else
  mkdir -p "$ROOT" "$LISTS" "$TSTAGE" "$OUT" "$SMOKE" "$HFDIR" "$OUT/ctl" || { echo "refused: cannot create $ROOT"; exit 4; }
  exec 9>"$ROOT/queue.lock"; flock -n 9 || { echo "queue already running in $ROOT"; exit 1; }
  say() { echo "[queue $(date -u +%FT%TZ)] $*" >> "$LOG"; }
  if [ -e "$START" ]; then
    echo "$T0 $(date -u +%FT%TZ) pid $$ resume" >> "$START"
  else
    ( set -C; echo "$T0 $(date -u +%FT%TZ) pid $$" > "$START" ) || { echo "refused: cannot write START marker"; exit 3; }
  fi
  ( set -C; du -sk "$ROOT" | cut -f1 > "$BASEF" ) 2>/dev/null
  say "start: TG_STOP_S=$STOP_S TG_MAX_GB=$MAX_GB WORKERS=$WORKERS free=${free_gb}GB"
  stop() { say "STOP $*"; echo "$*" > "$STOPM"; exit 1; }
  new_gb() { local b n; b=$(cat "$BASEF" 2>/dev/null || echo 0); n=$(du -sk "$ROOT" | cut -f1); echo $(( (n - b) / 1048576 )); }
  limits_hit() {   # prints the reason and returns 0 when a stop rule is hit
    local el; el=$(( $(date -u +%s) - T0 ))
    if [ "$el" -gt "$STOP_S" ]; then echo "stop rule: $el s since start > $STOP_S s"; return 0; fi
    local g; g=$(new_gb)
    if [ "$g" -gt "$MAX_GB" ]; then echo "stop rule: $g GB of new files under $ROOT > $MAX_GB GB"; return 0; fi
    return 1
  }
  memfree_gb() {   # container memory free: memory.max minus the anon line of memory.stat (never `free`)
    local mx an; mx=$(cat /sys/fs/cgroup/memory.max 2>/dev/null); an=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat 2>/dev/null | cut -d' ' -f2)
    if [ -z "$mx" ] || [ "$mx" = "max" ] || [ -z "$an" ]; then echo 9999; else echo $(( (mx - an) / 1000000000 )); fi
  }
  waitgate() {     # before a GPU step: fewer than 3 GPU processes and more than 28 GB of container memory free
    while true; do
      local n m r
      n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
      m=$(memfree_gb)
      if [ "${n:-0}" -lt 3 ] && [ "$m" -gt 28 ]; then return 0; fi
      r=$(limits_hit) && stop "$r (while waiting: $n GPU processes, $m GB memory free)"
      say "waiting: $n GPU processes, $m GB memory free"
      sleep 30
    done
  }
  guarded() {      # run a command in the background and enforce the stop rules while it runs; kills it when a rule is hit
    local name=$1; shift
    say "step $name: $*"
    "$@" >> "$LOG" 2>&1 &
    local pid=$! r rc
    while kill -0 "$pid" 2>/dev/null; do
      sleep 15
      if r=$(limits_hit); then kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; stop "$r (during $name)"; fi
    done
    wait "$pid"; rc=$?
    say "step $name rc=$rc"
    return $rc
  }
  run() { "$@" >> "$LOG" 2>&1; }
fi

export HF_HUB_DISABLE_XET=1 HF_HUB_CACHE=/workspace/v8proof/hf PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TANITAD_STACK=/workspace/v8proof/tree/stack TG_ADD99_KIT=$KIT99
export PYTHONPATH=/workspace/v8proof/tree/stack:/workspace/v8proof/tree/stack/scripts:/workspace/v8proof/tree/taniteval:$KIT99:$KIT
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 V2_TORCH_THREADS=2 PAI_DECODE_THREADS=1
export DINO_SAFETENSORS=$(cat /workspace/v8proof/hierrep/dino_safetensors.path 2>/dev/null)
CPU="env CUDA_VISIBLE_DEVICES=-1"
GPU="env -u CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"

# ------------------------------------------------------------------ 1 lists (the pilot's lists, first token of each line)
EV=$LISTS/eval141.txt
ALL=$LISTS/all_clips.txt
if [ "$DRY" = "1" ]; then
  echo "CMD: awk '{print \$1}' $SRC_LISTS/eval141.txt > $EV"
  echo "CMD: awk '{print \$1}' $SRC_LISTS/train_final_ids.txt > $LISTS/train_final_ids.txt"
  echo "CMD: cat $EV $LISTS/train_final_ids.txt > $ALL    (expect $EXPECT_CLIPS clips)"
else
  say "step 1: lists"
  [ -s "$SRC_LISTS/eval141.txt" ] && [ -s "$SRC_LISTS/train_final_ids.txt" ] || stop "pilot lists not found under $SRC_LISTS"
  awk '{print $1}' "$SRC_LISTS/eval141.txt" > "$EV"
  awk '{print $1}' "$SRC_LISTS/train_final_ids.txt" > "$LISTS/train_final_ids.txt"
  cat "$EV" "$LISTS/train_final_ids.txt" > "$ALL"
  n=$(grep -c . "$ALL")
  say "lists: $n clips"
  [ "$n" = "$EXPECT_CLIPS" ] || stop "the pilot lists hold $n clips, expected $EXPECT_CLIPS"
fi

# ------------------------------------------------------------------ 2 stage the tele clips (CPU + network)
guarded stage $CPU $PY $KIT/tele_stage.py --list $ALL --root $TSTAGE --hf-dir $HFDIR --workers $STAGE_WORKERS \
  || stop "tele_stage failed (see queue.log)"

# ------------------------------------------------------------------ 3 smoke encode of 3 clips (GPU)
[ "$DRY" = "1" ] || waitgate
guarded smoke $GPU $PY $KIT/tele_encode.py --list $ALL --wide-root $WIDE --tele-root $TSTAGE --out $SMOKE --limit 3 --keep-windows \
  --wide-tok-dir $WIDE_TOK --workers $WORKERS || stop "smoke encode failed (see queue.log)"

# ------------------------------------------------------------------ 4 control t2 (CPU; stages missing wide eval clips with the unchanged stage_clips.py)
guarded t2 $CPU $PY $KIT/ctl_t2.py --eval-list $EV --wide-root $WIDE --tele-root $TSTAGE --out $OUT/ctl/T2.json --n-clips 100 \
  --stage-wide --hf-dir $HFDIR --workers 4 || stop "control t2 FAILED or errored (see ctl/T2.json); the full encode is not started"

# ------------------------------------------------------------------ 5 full encode (GPU)
[ "$DRY" = "1" ] || waitgate
guarded encode $GPU $PY $KIT/tele_encode.py --list $ALL --wide-root $WIDE --tele-root $TSTAGE --out $OUT \
  --wide-tok-dir $WIDE_TOK --workers $WORKERS || stop "full encode failed (see queue.log)"

# ------------------------------------------------------------------ done
if [ "$DRY" = "1" ]; then
  echo "CMD: echo \"ZZ TELEGAP_ENC DONE <ISO time>Z all steps done\" >> $DONE"
else
  say "DONE"
  echo "ZZ TELEGAP_ENC DONE $(date -u +%FT%TZ) all steps done" >> "$DONE"
fi
exit 0
