#!/usr/bin/env bash
# pod_closeread_run.sh -- card PREREG_V8F_PROOF_ADD77 (CLOSEREAD): stages dump (reader seeds 0 and 1, both splits) -> fit/read. No training of any model but a ridge and tiny nets on CPU.
# Helper functions derived from pod_closrc10k_queue.sh (lines 7 to 106). No pgrep, no pkill, no rm, no mv, no kill.
# Markers START_CLOSEREAD.txt / CLOSEREAD_DONE.txt / CLOSEREAD_STOP.txt in $W; step markers in $W/markers. DRY=1 prints every command and runs nothing. Refuses to run twice.
# The tool files (closeread_dump.py, closeread_fit.py) are expected in $CR (default $W/tools): copy them there first (the stage checks they exist).
set -u
W=${CR_W:-/workspace/v8proof/closeread}
CR=${CR_TOOLS:-$W/tools}
I=${I10_W:-/workspace/v8proof/integ10k}
B=$I/build
PY=${I10_PY:-/workspace/venv/bin/python}
V=/workspace/v8proof
TRAINC=$V/data/cache_pod
EVALC=$I/eval_cache
JT=$V/data/inputs/join_train.npz
JE=$V/hierrep/pins/join_eval141.npz
META=$I/lr16/train_meta.json
RUN=I10RQ50KF5Hb20   # note N1 of the card: the run named in control c3 (I10RQ50KHb20) does not exist on the pod; the replan-tick rule log of this run is read
DUMP=$W/dump
READ=$W/read
EPMAP=${CR_EPISODE_MAP:-$W/episode_map.json}
DRY=${DRY:-0}
MINGB=${CR_MIN_FREE_GB:-28}
MAXGPU=3
TMO=${CR_TIMEOUT:-5400}
LOG=$W/queue_closeread.log
MM=${I10_MEMMAX_FILE:-/sys/fs/cgroup/memory.max}; MS=${I10_MEMSTAT_FILE:-/sys/fs/cgroup/memory.stat}
SEEDS=(0 1)

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { if [ "$DRY" = 1 ]; then echo "$*"; else echo "[$(ts)] $*" >> "$LOG"; fi; }
stop() { if [ "$DRY" = 1 ]; then echo "STOP $*"; exit 3; fi; echo "$(ts) $*" > "$W/CLOSEREAD_STOP.txt"; log "STOP $*"; exit 3; }

free_gb() { # container memory free in GB (memory.max minus the anon line); empty on failure
  local mmax anon
  [ -r "$MM" ] && [ -r "$MS" ] || return 1
  mmax=$(tr -d ' \r\n' < "$MM"); anon=$(awk '$1=="anon"{print $2}' "$MS" | head -n 1)
  [[ "$anon" =~ ^[0-9]+$ ]] || return 1
  if [ "$mmax" = max ]; then
    local av; av=$(awk '$1=="MemAvailable:"{print $2*1024}' /proc/meminfo | head -n 1); [[ "$av" =~ ^[0-9]+$ ]] || return 1
    awk -v b="$av" 'BEGIN{printf "%.1f", b/1073741824}'
  else
    [[ "$mmax" =~ ^[0-9]+$ ]] || return 1
    awk -v b="$((mmax - anon))" 'BEGIN{printf "%.1f", b/1073741824}'
  fi
}
n_gpu() { local n; n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ); [[ "$n" =~ ^[0-9]+$ ]] && echo "$n"; }
slot_ok() { # one probe: 0 when fewer than MAXGPU compute processes and more than MINGB free
  local g f; g=$(n_gpu) || return 2; f=$(free_gb) || return 2
  [ "$g" -lt "$MAXGPU" ] && awk -v f="$f" -v m="$MINGB" 'BEGIN{exit !(f > m)}'
}
wait_slot() { # step name; two successful probes 45 s apart; up to 6 h; an unreadable probe is a STOP (fail closed)
  local step=$1 t0 r
  if [ "$DRY" = 1 ]; then echo "SLOTWAIT before $step: (nvidia-smi compute apps < $MAXGPU) and (memory.max - anon > $MINGB GB), two probes 45 s apart"; return 0; fi
  t0=$(date +%s)
  while :; do
    slot_ok; r=$?
    [ $r = 2 ] && stop "SLOT-PROBE-UNREADABLE before $step"
    if [ $r = 0 ]; then
      sleep 45
      slot_ok; r=$?
      [ $r = 2 ] && stop "SLOT-PROBE-UNREADABLE before $step"
      [ $r = 0 ] && { log "slot free before $step (gpu $(n_gpu), free $(free_gb) GB)"; return 0; }
    fi
    [ $(( $(date +%s) - t0 )) -gt 21600 ] && stop "SLOT-TIMEOUT before $step"
    sleep 60
  done
}
done_step() { [ -f "$W/markers/$1.done" ]; }
mark() { [ "$DRY" = 1 ] || { mkdir -p "$W/markers"; echo "$(ts)" > "$W/markers/$1.done"; }; }

if [ "$DRY" != 1 ]; then
  [ -d "$W" ] || { echo "REFUSED: no $W"; exit 2; }
  ( set -C; echo "$(ts) pid $$" > "$W/START_CLOSEREAD.txt" ) 2>/dev/null || { echo "REFUSED: $W/START_CLOSEREAD.txt exists (a run was started)"; exit 2; }
  mkdir -p "$W/markers"
fi
log "run start (DRY=$DRY): card ADD77 (CLOSEREAD): dump for reader seeds 0 and 1 (fit and score splits), then the fit and read"

# ---- checks of the inputs (a missing input is a STOP, nothing runs)
if [ "$DRY" != 1 ]; then
  for f in closeread_dump.py closeread_fit.py; do [ -f "$CR/$f" ] || stop "TOOL-MISSING $CR/$f"; done
  [ -f "$B/tok_reader.py" ] || stop "BUILD-FILE-MISSING $B/tok_reader.py"
  for f in "$JT" "$JE" "$META"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
  [ -f "$TRAINC/train/manifest.jsonl" ] || stop "NO-TRAIN-CACHE $TRAINC"
  [ -f "$EVALC/eval/manifest.jsonl" ] || stop "NO-EVAL-CACHE $EVALC"
  [ -f "$EPMAP" ] || stop "NO-EPISODE-MAP $EPMAP (note N1 of the card: written by make_episode_map.py before the run)"
  for k in "${SEEDS[@]}"; do
    [ -f "$I/lr16/TK16_s$k/ckpt_last.pt" ] || stop "NO-READER TK16_s$k"
    [ -f "$I/t1/${RUN}_s$k/ticks/rule_log_${RUN}_s$k.npz" ] || stop "NO-RULE-LOG ${RUN}_s$k"
    [ -f "$I/qmap_sweep/QMAP_LR16_s${k}_q50.json" ] || stop "NO-MAP-Q50 s$k"
  done
  [ -e "$DUMP" ] && stop "OUTPUT-EXISTS $DUMP"
  [ -e "$READ" ] && stop "OUTPUT-EXISTS $READ"
fi

# ---- stage dump: one process per reader seed, sequential (the output folder of the tool must not exist: one folder holds both seeds, so each seed writes its own subfolder and the files are then read from them)
for k in "${SEEDS[@]}"; do
  step="dump_s$k"
  done_step "$step" && continue
  wait_slot "$step"
  out="$DUMP/s$k"
  if [ "$DRY" = 1 ]; then
    echo "CMD: PYTHONDONTWRITEBYTECODE=1 timeout $TMO $PY $CR/closeread_dump.py --reader $I/lr16/TK16_s$k/ckpt_last.pt --reader-seed $k --train-cache $TRAINC --eval-cache $EVALC --join-train $JT --join-eval $JE --train-meta $META --build-dir $B --out-dir $out --device cuda"
    continue
  fi
  mkdir -p "$DUMP"
  PYTHONDONTWRITEBYTECODE=1 timeout "$TMO" "$PY" "$CR/closeread_dump.py" --reader "$I/lr16/TK16_s$k/ckpt_last.pt" --reader-seed "$k" --train-cache "$TRAINC" --eval-cache "$EVALC" --join-train "$JT" --join-eval "$JE" --train-meta "$META" --build-dir "$B" --out-dir "$out" --device cuda >> "$W/dump_s$k.log" 2>&1
  rc=$?
  [ $rc = 0 ] || stop "DUMP-RC $rc seed $k"
  [ -f "$out/reader_s${k}_fit.npz" ] && [ -f "$out/reader_s${k}_score.npz" ] || stop "DUMP-FILES-MISSING seed $k"
  mark "$step"
  log "dump seed $k done"
done

# ---- stage fit/read: CPU only. The fit tool reads one folder with both seeds: the per-seed files are listed by the runner into READ_IN (hard links are not used: the tool is given --dump-dir of a folder made by copy)
step="fit_read"
if ! done_step "$step"; then
  wait_slot "$step"
  EP=(--episode-map "$EPMAP")
  if [ "$DRY" = 1 ]; then
    echo "COPY: cp $DUMP/s0/reader_s0_* $DUMP/s1/reader_s1_* -> $DUMP/both"
    echo "CMD: CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 timeout $TMO $PY $CR/closeread_fit.py --dump-dir $DUMP/both --out-dir $READ ${EP[*]} --rulelog0 $I/t1/${RUN}_s0/ticks/rule_log_${RUN}_s0.npz --qmap0 $I/qmap_sweep/QMAP_LR16_s0_q50.json --rulelog1 $I/t1/${RUN}_s1/ticks/rule_log_${RUN}_s1.npz --qmap1 $I/qmap_sweep/QMAP_LR16_s1_q50.json --seeds 0,1 --boot 2000"
  else
    mkdir -p "$DUMP/both"
    for k in "${SEEDS[@]}"; do cp -n "$DUMP/s$k"/reader_s${k}_*.npz "$DUMP/s$k"/reader_s${k}_lists.json "$DUMP/both/"; done
    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 timeout "$TMO" "$PY" "$CR/closeread_fit.py" --dump-dir "$DUMP/both" --out-dir "$READ" ${EP[@]+"${EP[@]}"} \
      --rulelog0 "$I/t1/${RUN}_s0/ticks/rule_log_${RUN}_s0.npz" --qmap0 "$I/qmap_sweep/QMAP_LR16_s0_q50.json" --rulelog1 "$I/t1/${RUN}_s1/ticks/rule_log_${RUN}_s1.npz" --qmap1 "$I/qmap_sweep/QMAP_LR16_s1_q50.json" \
      --seeds 0,1 --boot 2000 > "$W/fit_read.log" 2>&1
    rc=$?
    tail -n 1 "$W/fit_read.log" >> "$LOG"
    [ $rc = 0 ] || stop "FIT-RC $rc (rc 1 = a control did not pass: see $READ/CLOSEREAD_CONTROLS.json; no name was written)"
    mark "$step"
  fi
fi
if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi
echo "$(ts) all steps done" > "$W/CLOSEREAD_DONE.txt"
log "CLOSEREAD_DONE"
