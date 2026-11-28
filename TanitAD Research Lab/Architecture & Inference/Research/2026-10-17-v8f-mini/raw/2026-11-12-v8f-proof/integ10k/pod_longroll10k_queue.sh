#!/usr/bin/env bash
# pod_longroll10k_queue.sh -- ON THE POD (nohup setsid, started by chain_longroll10k.sh). Card PREREG_V8F_PROOF_ADD70 (LONGROLL10K) + note N1. Built from the pattern pod_cyclebound10k_queue.sh.
# No training, no map fit. Wrapper of record $B/v8f_t1_eval_reader_rule_rzgfhq.py in EVERY run, --kappa-max 0.2 in every run, 2.0 s cycle (--replan-steps 10), --no-analyze (as in the smoke
# pod_lr_smoke_b.sh that passed; the reads of the card run on the dumps and logs). The queue waits for the end marker of the queue of card ADD69 (WAIT_DONE / WAIT_STOP, defaults
# CYCLEBOUND10K_DONE.txt / CYCLEBOUND10K_STOP.txt in $W; WAIT_MAX seconds, default 86400). Markers START_LONGROLL10K.txt / LONGROLL10K_DONE.txt / LONGROLL10K_STOP.txt; step markers
# lr6check, lr1cmp, lr2cmp in $W/markers (plus ro_<arm> per rollout). Steps, in this order:
#   grids   make_longroll_grid.py on the pinned grid -> $W/longroll/grid_p30.json, grid_p60.json; STOP unless the window counts are 1271 and 284
#   lr6check check_preroll_b.py: last line "ZZ CHECK-PREROLL-B PASS"
#   lr1cmp  I10R0LrID_s0 (P = 0, no rule, --readout pinned, full pinned grid) against the banked dump of I10R0Hb20_s0 with cmp_dumps.py: differing=0
#   lr2cmp  I10R0LrFresh_s0 (P = 0, --nav logged, pinned readout, P = 30 grid) and I10R0LrCarry_s0 (P = 30, --preroll-exec logged, --nav logged, same grid): max_abs <= 1e-6, and in the
#           pre-roll log of the carry run the largest |a_m0| and the largest position offset <= 1e-6
#   then 20 outcome rollouts, model seeds 0 and 1 (seed 0 first), per seed in this order: R0P30Hb20 R0KP30Hb20 RLcKF5P30Hb20 RQ50KF5P30Hb20 R0KNlHb20 R0KNlP30Hb20 R0P60Hb20 R0KP60Hb20
#           RLcKF5P60Hb20 RQ50KF5P60Hb20 (names get the prefix I10 and the suffix _s<seed>)
# One GPU job at a time, a slot wait before every rollout. Uses no pgrep, no pkill, no rm, no mv, no kill. Writes only inside $W. DRY=1 prints every command and runs nothing.
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
SM=$W/small_lr
DRY=${DRY:-0}
TMO=${I10_TIMEOUT:-5400}
MINGB=${I10_MIN_FREE_GB:-28}
MAXGPU=3
LOG=$W/queue_longroll.log
MM=${I10_MEMMAX_FILE:-/sys/fs/cgroup/memory.max}; MS=${I10_MEMSTAT_FILE:-/sys/fs/cgroup/memory.stat}
ARMS=(0 1)

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { if [ "$DRY" = 1 ]; then echo "$*"; else echo "[$(ts)] $*" >> "$LOG"; fi; }
stop() { if [ "$DRY" = 1 ]; then echo "STOP $*"; exit 3; fi; echo "$(ts) $*" > "$W/LONGROLL10K_STOP.txt"; log "STOP $*"; exit 3; }

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

# ---------------------------------------------------------------- start
if [ "$DRY" != 1 ]; then
  [ -d "$W" ] || { echo "REFUSED: no $W"; exit 2; }
  ( set -C; echo "$(ts) pid $$" > "$W/START_LONGROLL10K.txt" ) 2>/dev/null || { echo "REFUSED: $W/START_LONGROLL10K.txt exists (a queue was started)"; exit 2; }
  mkdir -p "$W/markers" "$W/t1" "$SM"
fi
log "queue start (DRY=$DRY): card ADD70 (LONGROLL10K): grids, three controls with the checks, then twenty outcome rollouts at a 2.0 s cycle on the bounded executor with a pre-roll"
if [ "$DRY" != 1 ]; then
  t0=$(date +%s)
  WAIT_DONE=${WAIT_DONE:-CYCLEBOUND10K_DONE.txt}; WAIT_STOP=${WAIT_STOP:-CYCLEBOUND10K_STOP.txt}; WAIT_MAX=${WAIT_MAX:-86400}
  until [ -f "$W/$WAIT_DONE" ] || [ -f "$W/$WAIT_STOP" ]; do
    [ $(( $(date +%s) - t0 )) -gt "$WAIT_MAX" ] && stop "WAIT-PREVIOUS-QUEUE-TIMEOUT"
    sleep 30
  done
  log "the previous queue has ended ($WAIT_DONE or $WAIT_STOP); the grids, controls and rollouts of card ADD70 start"
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

# ---------------------------------------------------------------- grids, made once from the pinned grid
if [ "$DRY" = 1 ]; then
  echo "CMD: $PY $B/make_longroll_grid.py --grid $GRID --preroll-steps 30 --out $G30   (skipped when the file exists)"
  echo "CMD: $PY $B/make_longroll_grid.py --grid $GRID --preroll-steps 60 --out $G60   (skipped when the file exists)"
  echo "CMD: grid_counts $G30 -> windows must be 1271; grid_counts $G60 -> windows must be 284 (else STOP GRID-COUNT); counts go to $SM/GRID_COUNTS.txt"
else
  mkdir -p "$LG"
  [ -f "$G30" ] || "$PY" "$B/make_longroll_grid.py" --grid "$GRID" --preroll-steps 30 --out "$G30" >> "$LG/make_grid.log" 2>&1 || stop "MAKE-GRID-P30-FAILED"
  [ -f "$G60" ] || "$PY" "$B/make_longroll_grid.py" --grid "$GRID" --preroll-steps 60 --out "$G60" >> "$LG/make_grid.log" 2>&1 || stop "MAKE-GRID-P60-FAILED"
  c0=$(grid_counts "$GRID") || stop "GRID-COUNT-UNREADABLE pinned"
  c30=$(grid_counts "$G30") || stop "GRID-COUNT-UNREADABLE p30"
  c60=$(grid_counts "$G60") || stop "GRID-COUNT-UNREADABLE p60"
  { echo "pinned grid (clips windows): $c0"; echo "grid_p30 (clips windows): $c30"; echo "grid_p60 (clips windows): $c60"; echo "grid_p30 sha256 $(sha256sum "$G30" | cut -c1-64)"; echo "grid_p60 sha256 $(sha256sum "$G60" | cut -c1-64)"; } > "$SM/GRID_COUNTS.txt"
  log "grid counts: pinned $c0, p30 $c30, p60 $c60"
  [ "${c30##* }" = 1271 ] || stop "GRID-COUNT-P30 ${c30##* } (expected 1271)"
  [ "${c60##* }" = 284 ] || stop "GRID-COUNT-P60 ${c60##* } (expected 284)"
fi

# ---------------------------------------------------------------- the rollout function
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

# ---------------------------------------------------------------- lr6check: the tool check on the pod (no model, no GPU, two threads)
if ! done_step lr6check; then
  if [ "$DRY" = 1 ]; then
    echo "CMD: CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 DISTRULE_DIR=$B PYTHONPATH=$TREE/stack $PY $B/check_preroll_b.py $WRAP > $W/lr_test/CHECK_PREROLL_B.txt   (last line must be ZZ CHECK-PREROLL-B PASS)"
  else
    mkdir -p "$W/lr_test"
    cp -f "$WRAP" "$W/lr_test/v8f_t1_eval_reader_rule_rzgfhq.py"
    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" timeout 1800 "$PY" "$B/check_preroll_b.py" "$WRAP" > "$W/lr_test/CHECK_PREROLL_B.txt" 2>&1
    tail -n 1 "$W/lr_test/CHECK_PREROLL_B.txt" | grep -q "ZZ CHECK-PREROLL-B PASS" || stop "TOOL-CHECK-PREROLL-B-FAILED"
    mark lr6check
  fi
fi

# ---------------------------------------------------------------- lr1cmp: P = 0 on the full pinned grid against the banked dump of I10R0Hb20_s0
READOUT=pinned KAPPA=0.2 PREROLL=0 rollout I10R0LrID R0 0 0
if [ "$DRY" = 1 ]; then
  echo "CMD: $PY $B/cmp_dumps.py $W/t1/I10R0Hb20_s0/dump_I10R0Hb20_s0_m10_causal_heads $W/t1/I10R0LrID_s0/dump_I10R0LrID_s0_m10_causal_heads > $SM/CMP_LrID.txt   (CMP line must read differing=0)"
elif ! done_step lr1cmp; then
  "$PY" "$B/cmp_dumps.py" "$W/t1/I10R0Hb20_s0/dump_I10R0Hb20_s0_m10_causal_heads" "$W/t1/I10R0LrID_s0/dump_I10R0LrID_s0_m10_causal_heads" > "$SM/CMP_LrID.txt" 2>&1
  l=$(grep "^CMP" "$SM/CMP_LrID.txt" | tail -n 1)
  log "lr1cmp $l"
  case "$l" in *"differing=0 "*|*"differing=0") ;; *) stop "LR1-RERUN-DIFFERS $l" ;; esac
  mark lr1cmp
fi

# ---------------------------------------------------------------- lr2cmp: the state carry (logged pre-roll reads as a fresh window), on the P = 30 grid
RGRID=$G30 NAV=logged READOUT=pinned KAPPA=0.2 PREROLL=0 rollout I10R0LrFresh R0 0 0
RGRID=$G30 NAV=logged READOUT=pinned KAPPA=0.2 PREROLL=30 PREEXEC=logged rollout I10R0LrCarry R0 0 0
if [ "$DRY" = 1 ]; then
  echo "CMD: $PY $B/cmp_dumps.py $W/t1/I10R0LrFresh_s0/dump_I10R0LrFresh_s0_m10_causal_heads $W/t1/I10R0LrCarry_s0/dump_I10R0LrCarry_s0_m10_causal_heads > $SM/CMP_LrCarry.txt   (CMP max_abs <= 1e-6)"
  echo "CMD: read $W/t1/I10R0LrCarry_s0/ticks/preroll_log_I10R0LrCarry_s0.json: a_m0.<kind>.main_max_abs, a_m0.<kind>.segments_max_abs, offsets_at_t0.<kind>.dpos.max, all <= 1e-6 -> $SM/CARRY_PREROLL_LINES.txt"
elif ! done_step lr2cmp; then
  "$PY" "$B/cmp_dumps.py" "$W/t1/I10R0LrFresh_s0/dump_I10R0LrFresh_s0_m10_causal_heads" "$W/t1/I10R0LrCarry_s0/dump_I10R0LrCarry_s0_m10_causal_heads" > "$SM/CMP_LrCarry.txt" 2>&1
  l=$(grep "^CMP" "$SM/CMP_LrCarry.txt" | tail -n 1)
  log "lr2cmp $l"
  mx=$(echo "$l" | sed -n 's/.*max_abs=\([^ ]*\).*/\1/p')
  [[ "$mx" =~ ^[0-9.]+([eE][-+]?[0-9]+)?$ ]] || stop "LR2-CMP-UNREADABLE $l"
  awk -v m="$mx" 'BEGIN{exit !(m <= 1e-6)}' || stop "LR2-CARRY-ABOVE-1E-6 max_abs=$mx"
  "$PY" - "$W/t1/I10R0LrCarry_s0/ticks/preroll_log_I10R0LrCarry_s0.json" > "$SM/CARRY_PREROLL_LINES.txt" <<'PYEOF'
import json, sys
j = json.load(open(sys.argv[1]))
a, o = j.get("a_m0", {}), j.get("offsets_at_t0", {})
vals = []
for kind in sorted(a):
    m, s = a[kind].get("main_max_abs"), a[kind].get("segments_max_abs")
    print("a_m0 kind=%s main_max_abs=%s segments_max_abs=%s" % (kind, m, s))
    vals.append(m)
    if s is not None:
        vals.append(s)
for kind in sorted(o):
    d = o[kind]["dpos"]["max"]
    print("offset kind=%s dpos_max_m=%s dpsi_max_deg=%s" % (kind, d, o[kind]["dpsi_deg"]["max"]))
    vals.append(d)
ok = bool(vals) and all(v is not None and abs(float(v)) <= 1e-6 for v in vals)
print("CARRY-PREROLL-OK" if ok else "CARRY-PREROLL-FAIL")
sys.exit(0 if ok else 1)
PYEOF
  [ $? = 0 ] || stop "LR2-PREROLL-LOG-ABOVE-1E-6 (see $SM/CARRY_PREROLL_LINES.txt)"
  mark lr2cmp
fi

# ---------------------------------------------------------------- the twenty outcome rollouts (model seeds 0 and 1; reader seed = model seed; map file of the model seed)
for k in "${ARMS[@]}"; do
  Q="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"
  # P = 30
  RGRID=$G30 PREROLL=30 READOUT=pinned KAPPA=0.2 rollout I10R0P30Hb20 R0 "$k" "$k"
  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 rollout I10R0KP30Hb20 R0 "$k" "$k"
  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 rollout I10RLcKF5P30Hb20 RLc "$k" "$k"
  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$Q" rollout I10RQ50KF5P30Hb20 RR "$k" "$k"
  # diagnostic pair, logged route, P = 30 grid
  RGRID=$G30 PREROLL=0 NAV=logged READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 rollout I10R0KNlHb20 R0 "$k" "$k"
  RGRID=$G30 PREROLL=30 NAV=logged READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 rollout I10R0KNlP30Hb20 R0 "$k" "$k"
  # P = 60
  RGRID=$G60 PREROLL=60 READOUT=pinned KAPPA=0.2 rollout I10R0P60Hb20 R0 "$k" "$k"
  RGRID=$G60 PREROLL=60 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 rollout I10R0KP60Hb20 R0 "$k" "$k"
  RGRID=$G60 PREROLL=60 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 rollout I10RLcKF5P60Hb20 RLc "$k" "$k"
  RGRID=$G60 PREROLL=60 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$Q" rollout I10RQ50KF5P60Hb20 RR "$k" "$k"
done

# ---------------------------------------------------------------- small result folder
if [ "$DRY" = 1 ]; then echo "CMD: cp the compare files, grid counts, check output, carry lines, the preroll logs of the controls and the queue log into $SM"; echo "DRY DONE"; exit 0; fi
cp -f "$W/lr_test/CHECK_PREROLL_B.txt" "$LOG" "$SM/" 2>/dev/null
cp -f "$G30" "$G60" "$LG/make_grid.log" "$SM/" 2>/dev/null
for nm in I10R0LrFresh_s0 I10R0LrCarry_s0; do cp -f "$W/t1/$nm/ticks/preroll_log_$nm.json" "$SM/" 2>/dev/null; done
for nm in I10R0LrID_s0 I10R0LrFresh_s0 I10R0LrCarry_s0; do cp -f "$W/t1/$nm/ticks/run_$nm.txt" "$SM/" 2>/dev/null; done
echo "$(ts) all steps done" > "$W/LONGROLL10K_DONE.txt"
log "LONGROLL10K_DONE"
