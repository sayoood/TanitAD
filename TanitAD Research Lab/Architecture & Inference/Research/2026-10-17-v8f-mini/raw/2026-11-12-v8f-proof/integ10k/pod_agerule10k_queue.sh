#!/usr/bin/env bash
# pod_agerule10k_queue.sh -- card PREREG_V8F_PROOF_ADD71 (AGERULE10K), part A: six rollouts at a rollout age of 6 to 12 s on the P = 30 grid of card ADD70 (pre-roll 30 steps,
# 2.0 s cycle, wrapper of record of card ADD70, --kappa-max 0.2, model seeds 0 and 1): I10RLcP30Hb20, I10RQ50P30Hb20 (pinned readout, rule at every replan) and I10R0C10XP30Hb20
# (causal readout on the executed path with the own acceleration). Written by make_agerule_queue.py from the queue of card ADD70: helper functions, start checks and the rollout
# function are its lines. No training, no map fit, no wait for another queue. One GPU job at a time, a slot wait before every rollout. No pgrep, no pkill, no rm, no mv, no kill.
# Markers START_AGERULE10K.txt / AGERULE10K_DONE.txt / AGERULE10K_STOP.txt in $W; step markers ro_<arm> in $W/markers. DRY=1 prints every command and runs nothing.
set -u
W=${I10_W:-/workspace/v8proof/integ10k}
B=$W/build
PY=${I10_PY:-/workspace/venv/bin/python}
TREE=$B/tree_ov
V=/workspace/v8proof
TRACKS=$V/data/inputs/native_tracks.pt
GRID=$V/hierrep/pins/eval_grid_windows.json
JOINE=$V/hierrep/pins/join_eval141.npz
EVALC=$W/eval_cache
PINNED=61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397
WRAP=$B/v8f_t1_eval_reader_rule_rzgfhq.py
LG=$W/longroll
G30=$LG/grid_p30.json
G60=$LG/grid_p60.json
SM=$W/small_ar
DRY=${DRY:-0}
TMO=${I10_TIMEOUT:-5400}
MINGB=${I10_MIN_FREE_GB:-28}
MAXGPU=3
LOG=$W/queue_agerule.log
MM=${I10_MEMMAX_FILE:-/sys/fs/cgroup/memory.max}; MS=${I10_MEMSTAT_FILE:-/sys/fs/cgroup/memory.stat}
ARMS=(0 1)

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { if [ "$DRY" = 1 ]; then echo "$*"; else echo "[$(ts)] $*" >> "$LOG"; fi; }
stop() { if [ "$DRY" = 1 ]; then echo "STOP $*"; exit 3; fi; echo "$(ts) $*" > "$W/AGERULE10K_STOP.txt"; log "STOP $*"; exit 3; }

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
grid_counts() { # GRIDFILE -> "<clips> <windows>" (windows counted from the lists, not from the n_windows field)
  "$PY" - "$1" <<'PYEOF'
import json, sys
g = json.load(open(sys.argv[1], encoding="utf-8"))
print(len(g["order"]), sum(len(g["windows"][c]) for c in g["order"]))
PYEOF
}

if [ "$DRY" != 1 ]; then
  [ -d "$W" ] || { echo "REFUSED: no $W"; exit 2; }
  ( set -C; echo "$(ts) pid $$" > "$W/START_AGERULE10K.txt" ) 2>/dev/null || { echo "REFUSED: $W/START_AGERULE10K.txt exists (a queue was started)"; exit 2; }
  mkdir -p "$W/markers" "$W/t1" "$SM"
fi
log "queue start (DRY=$DRY): card ADD71 (AGERULE10K), part A: six rollouts with a pre-roll of 30 steps on the P = 30 grid"
if [ "$DRY" != 1 ]; then
  for k in "${ARMS[@]}"; do
    [ -f "$W/t1/I10R0Hb20_s$k/T1_DONE" ] || stop "NO-BANKED-I10R0Hb20 s$k"
  done
fi
for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzgfhq.py v8f_t1_eval_reader_rule_rzgfh.py v8f_t1_eval_reader_rule_rz.py check_preroll_b.py make_longroll_grid.py dist_rule.py v8f_t1_eval_rule.py v8f_t1_eval_rule_mix.py v8f_t1_eval_reader_rule.py tok_reader.py tok_probe_lr.py qmap_fit.py qcal_map.py reader_controls.py; do
  [ "$DRY" = 1 ] || [ -f "$B/$f" ] || stop "BUILD-FILE-MISSING $f"
done
if [ "$DRY" != 1 ]; then
  [ -d "$TREE/taniteval/taniteval" ] && [ -e "$TREE/stack/scripts/v8f_t1_eval.py" ] || stop "NO-OVERLAY-TREE $TREE"
  got=$(sha256sum "$TREE/stack/scripts/v8f_t1_eval.py" | cut -c1-64); [ "$got" = "$PINNED" ] || stop "PINNED-SHA $got"
  for f in "$TRACKS" "$GRID" "$JOINE"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
  [ -f "$EVALC/eval/manifest.jsonl" ] || stop "NO-EVAL-CACHE"
  for k in "${ARMS[@]}"; do
    [ -f "$W/ckpt/UE_s$k/A/ckpt_last.pt" ] || stop "NO-CKPT UE_s$k"
    [ -f "$W/lr16/TK16_s$k/ckpt_last.pt" ] || stop "NO-READER TK16_s$k"
    [ -f "$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" ] || stop "NO-MAP-Q50 s$k"
  done
fi

# rollout NAME FORM MODELSEED READERSEED     environment: RGRID (grid file, default pinned), PREROLL (default 0), PREEXEC (plan|logged, optional), NAV (default deadreckon),
#   READOUT, ACCGAIN, PATHEXEC, KAPPA, EVERY, REPLAN (default 10), QMAP_FILE
rollout() {
  local name=$1 form=$2 m=$3 r=$4
  local arm=${name}_s$m out=$W/t1/${name}_s$m tag=${name}_s${m}_m10_causal_heads
  local grid=${RGRID:-$GRID} pre=${PREROLL:-0} nav=${NAV:-deadreckon}
  done_step "ro_$arm" && return 0
  [ "$DRY" = 1 ] || { [ -e "$out" ] && stop "OUTPUT-EXISTS $out"; }
  wait_slot "rollout $arm"
  [ "$DRY" = 1 ] || mkdir -p "$out/ticks"
  local extra=()
  case "$form" in
    RR) extra+=(--reader "$W/lr16/TK16_s$r/ckpt_last.pt") ;;
  esac
  [ -n "${EVERY:-}" ] && extra+=(--rule-every "$EVERY")
  [ -n "${KAPPA:-}" ] && extra+=(--kappa-max "$KAPPA")
  [ -n "${ACCGAIN:-}" ] && extra+=(--acc-gain "$ACCGAIN")
  [ "${PATHEXEC:-0}" = 1 ] && extra+=(--path-exec)
  [ -n "${READOUT:-}" ] && extra+=(--readout "$READOUT" --readout-log "$out/ticks/readout_log_$arm.json")
  extra+=(--preroll-steps "$pre")
  [ -n "${PREEXEC:-}" ] && extra+=(--preroll-exec "$PREEXEC")
  case "$form" in RR) extra+=(--qmap "${QMAP_FILE:?}") ;; esac
  [ "$form" = R0 ] || extra+=(--rule-log "$out/ticks/rule_log_$arm.npz")
  if [ "$DRY" = 1 ]; then
    echo "CMD: DISTRULE_DIR=$B PYTHONPATH=$TREE/stack timeout $TMO $PY $WRAP --tree $TREE --form $form ${extra[*]} run --ckpt $W/ckpt/UE_s$m/A/ckpt_last.pt --tag $tag --tracks $TRACKS --eval-cache $EVALC --eval-grid $grid --eval-join $JOINE --out-dir $out --replan-steps ${REPLAN:-10} --nav $nav --scene logged --schedule periodic --limit causal --dump-ticks $out/ticks --dump-heads --no-analyze --device cuda"
    return 0
  fi
  echo "arm $arm form $form replan steps ${REPLAN:-10} rule every ${EVERY:-10} readout ${READOUT:-pinned} kappa ${KAPPA:-none} preroll $pre exec ${PREEXEC:-plan} nav $nav grid $(basename "$grid") model seed $m reader seed $r ckpt sha256 $(sha256sum "$W/ckpt/UE_s$m/A/ckpt_last.pt" | cut -c1-64) wrapper sha256 $(sha256sum "$WRAP" | cut -c1-64) started $(ts)" > "$out/ticks/run_$arm.txt"
  DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" timeout "$TMO" "$PY" "$WRAP" --tree "$TREE" --form "$form" "${extra[@]}" run --ckpt "$W/ckpt/UE_s$m/A/ckpt_last.pt" --tag "$tag" --tracks "$TRACKS" --eval-cache "$EVALC" --eval-grid "$grid" --eval-join "$JOINE" --out-dir "$out" --replan-steps ${REPLAN:-10} --nav "$nav" --scene logged --schedule periodic --limit causal --dump-ticks "$out/ticks" --dump-heads --no-analyze --device cuda >> "$out/t1.log" 2>&1
  local rc=$?
  echo "rc $rc finished $(ts)" >> "$out/ticks/run_$arm.txt"
  [ $rc = 0 ] || stop "ROLLOUT-RC $rc $arm"
  local want n cw
  cw=$(grid_counts "$grid") || stop "GRID-UNREADABLE $arm"
  want=${cw%% *}
  n=$(ls "$out/dump_$tag" 2>/dev/null | grep -c -- '^ep.*npz$')
  [ "$n" = "$want" ] || stop "ROLLOUT-DUMP-COUNT $n expected $want $arm"
  if [ "$pre" != 0 ]; then
    [ -f "$out/ticks/preroll_log_$arm.json" ] || stop "NO-PREROLL-LOG $arm"
  fi
  touch "$out/T1_DONE"
  mark "ro_$arm"
  log "done $arm (dump files $n)"
}

[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"
if [ "$DRY" != 1 ]; then
  for nm in I10RLcKF5P30Hb20 I10RQ50KF5P30Hb20 I10R0KP30Hb20 I10R0P30Hb20 I10RLcHb20 I10RQ50Hb20 I10R0C10XHb20 I10R0KHb20 I10RLcKF5Hb20 I10RQ50KF5Hb20; do
    for k in "${ARMS[@]}"; do [ -d "$W/t1/${nm}_s$k/dump_${nm}_s${k}_m10_causal_heads" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done
  done
fi
for k in "${ARMS[@]}"; do
  Q="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"
  RGRID=$G30 PREROLL=30 READOUT=pinned KAPPA=0.2 rollout I10RLcP30Hb20 RLc "$k" "$k"
  RGRID=$G30 PREROLL=30 READOUT=pinned KAPPA=0.2 QMAP_FILE="$Q" rollout I10RQ50P30Hb20 RR "$k" "$k"
  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=1 PATHEXEC=1 KAPPA=0.2 rollout I10R0C10XP30Hb20 R0 "$k" "$k"
done
if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi
cp -f "$LOG" "$SM/" 2>/dev/null
for nm in I10RLcP30Hb20 I10RQ50P30Hb20 I10R0C10XP30Hb20; do for k in "${ARMS[@]}"; do cp -f "$W/t1/${nm}_s$k/ticks/run_${nm}_s$k.txt" "$W/t1/${nm}_s$k/ticks/preroll_log_${nm}_s$k.json" "$SM/" 2>/dev/null; done; done
echo "$(ts) all steps done" > "$W/AGERULE10K_DONE.txt"
log "AGERULE10K_DONE"
