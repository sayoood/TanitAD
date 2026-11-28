#!/usr/bin/env bash
# run_tokclose.sh -- pod chain of card PREREG_V8F_PROOF_ADD81 (TOKCLOSE), tier T0: eight fits (E10, E25, T10, T25 x seeds 0 and 1), one planted fit (T10 seed 0, shuffled closing labels), then the score.
# Before every fit: wait for a slot (container memory free > MINGB GB and fewer than 3 GPU compute processes, two probes 45 s apart). Writes new files only: no removal, no process listing, no stopping.
# Markers START_TOKCLOSE.txt / TOKCLOSE_DONE.txt / TOKCLOSE_STOP.txt in $W; step markers in $W/markers. A second start is refused. DRY=1 prints every command and runs nothing.
# Tool files expected in $W/tools: tokclose_fit.py; in $V/closeread/tools: closeread_dump.py closeread_fit.py closest.py; in $B: tok_reader.py.
set -u
V=/workspace/v8proof
W=${TC_W:-$V/tokclose}
PY=${TC_PY:-/workspace/venv/bin/python}
B=${TC_BUILD:-$V/integ10k/build}
CT=${TC_CRTOOLS:-$V/closeread/tools}
TOOLS=$W/tools
I=$V/integ10k
TRAINC=${TC_TRAIN_CACHE:-$V/data/cache_pod}
EVALC=${TC_EVAL_CACHE:-$I/eval_cache}
JT=${TC_JOIN_TRAIN:-$V/data/inputs/join_train.npz}
JE=${TC_JOIN_EVAL:-$V/hierrep/pins/join_eval141.npz}
META=${TC_TRAIN_META:-$I/lr16/train_meta.json}
DUMP=${TC_DUMP:-$V/closeread/dump/both}
EST=${TC_EST:-$V/closeread/est}
ADD78=${TC_ADD78:-$V/closeread/read2/CLOSEREAD_RESULT.json}
EPMAP=${TC_EPMAP:-$V/closeread/episode_map.json}
FITS=$W/fits
OUT=$W/score
LOGD=$W/logs
DRY=${DRY:-0}
MINGB=${TC_MIN_FREE_GB:-28}
MAXGPU=3
TMO_E=${TC_TIMEOUT_E:-3600}
TMO_T=${TC_TIMEOUT_T:-14400}
MM=${TC_MEMMAX_FILE:-/sys/fs/cgroup/memory.max}; MS=${TC_MEMSTAT_FILE:-/sys/fs/cgroup/memory.stat}
export PYTHONPATH="$TOOLS:$CT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export TOKCLOSE_BUILD_DIR="$B"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { if [ "$DRY" = 1 ]; then echo "$*"; else echo "[$(ts)] $*" >> "$W/queue_tokclose.log"; fi; }
stop() { if [ "$DRY" = 1 ]; then echo "STOP $*"; exit 3; fi; echo "$(ts) $*" > "$W/TOKCLOSE_STOP.txt"; log "STOP $*"; exit 3; }

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
  ( set -C; echo "$(ts) pid $$" > "$W/START_TOKCLOSE.txt" ) 2>/dev/null || { echo "REFUSED: $W/START_TOKCLOSE.txt exists (a chain was started)"; exit 2; }
  mkdir -p "$W/markers" "$FITS" "$LOGD"
fi
log "chain start (DRY=$DRY): card ADD81 (TOKCLOSE): 8 fits + 1 planted fit, then score"
for f in "$TOOLS/tokclose_fit.py" "$CT/closeread_dump.py" "$CT/closeread_fit.py" "$CT/closest.py" "$B/tok_reader.py"; do
  [ "$DRY" = 1 ] || [ -f "$f" ] || stop "TOOL-FILE-MISSING $f"
done
for f in "$JT" "$JE" "$META" "$ADD78" "$EPMAP" "$TRAINC/train/manifest.jsonl" "$EVALC/eval/manifest.jsonl"; do
  [ "$DRY" = 1 ] || [ -f "$f" ] || stop "MISSING-INPUT $f"
done
for k in 0 1; do
  for f in "$I/lr16/TK16_s$k/ckpt_last.pt" "$DUMP/reader_s${k}_score.npz" "$DUMP/reader_s${k}_lists.json" "$EST/closest_N10_s$k.npz" "$EST/closest_L25_s$k.npz"; do
    [ "$DRY" = 1 ] || [ -f "$f" ] || stop "MISSING-INPUT $f"
  done
done
if [ "$DRY" != 1 ]; then
  [ -e "$OUT" ] && stop "OUTPUT-EXISTS $OUT"
  "$PY" "$TOOLS/tokclose_fit.py" --selftest > "$LOGD/selftest.log" 2>&1 || stop "SELFTEST-FAIL"
  tail -n 1 "$LOGD/selftest.log" | grep -q "ZZ TOKCLOSE-SELFTEST rc 0" || stop "SELFTEST-NOT-GREEN"
else
  echo "CMD: $PY $TOOLS/tokclose_fit.py --selftest"
fi

# fit ARM SEED [shuf]
fit() {
  local arm=$1 k=$2 shuf=${3:-} tag tmo extra=()
  tag=${arm}${shuf:+shuf}_s$k
  done_step "fit_$tag" && return 0
  case "$arm" in E*) tmo=$TMO_E; extra+=(--reader "$I/lr16/TK16_s$k/ckpt_last.pt") ;; *) tmo=$TMO_T ;; esac
  [ -n "$shuf" ] && extra+=(--shuffle-closing-labels)
  wait_slot "fit $tag"
  local cmd=("$PY" "$TOOLS/tokclose_fit.py" fit --arm "$arm" --seed "$k" --train-cache "$TRAINC" --eval-cache "$EVALC" --join-train "$JT" --join-eval "$JE" --train-meta "$META"
             --lists-json "$DUMP/reader_s${k}_lists.json" --build-dir "$B" --out-dir "$FITS" --device cuda "${extra[@]}")
  if [ "$DRY" = 1 ]; then echo "CMD: timeout $tmo ${cmd[*]}"; return 0; fi
  [ -e "$FITS/tokclose_${tag}_log.json" ] && stop "OUTPUT-EXISTS fit $tag"
  log "fit $tag start"
  timeout "$tmo" "${cmd[@]}" > "$LOGD/fit_$tag.log" 2>&1
  local rc=$?
  [ $rc = 0 ] || stop "FIT-FAILED $tag rc $rc"
  [ -f "$FITS/tokclose_${tag}_log.json" ] && [ -f "$FITS/tokclose_${tag}_table.npz" ] && [ -f "$FITS/tokclose_${tag}.pt" ] || stop "FIT-OUTPUT-MISSING $tag"
  mark "fit_$tag"
  log "fit $tag done: $(tail -n 1 "$LOGD/fit_$tag.log")"
}

for k in 0 1; do for arm in E10 E25; do fit "$arm" "$k"; done; done
for k in 0 1; do for arm in T10 T25; do fit "$arm" "$k"; done; done
fit T10 0 shuf

SC=("$PY" "$TOOLS/tokclose_fit.py" score --fit-dir "$FITS" --dump-dir "$DUMP" --est-dir "$EST" --add78-result "$ADD78" --episode-map "$EPMAP" --eval-cache "$EVALC" --join-eval "$JE" --build-dir "$B" --out-dir "$OUT")
if [ "$DRY" = 1 ]; then echo "CMD: ${SC[*]}"; echo "DRY run complete"; exit 0; fi
"${SC[@]}" > "$LOGD/score.log" 2>&1
rc=$?
echo "$(ts) score rc $rc (0 = controls pass and names written; 1 = a control failed, no name)" > "$W/TOKCLOSE_DONE.txt"
log "score rc $rc"
exit 0
