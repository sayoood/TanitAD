#!/usr/bin/env bash
# pod_price10k_queue.sh -- written by make_price10k.py from pod_integ10k_queue_n2.sh. Card PREREG_V8F_PROOF_ADD47 (PRICE10K): maps at the levels 0.35 and 0.50
# from the banked reader rows, their check, then six rollouts (RQ50, RQ35, RRaw per model seed) with the map file as a parameter. No training, no reader step.
# Markers START_PRICE10K.txt / PRICE10K_STOP.txt / PRICE10K_DONE.txt. The header lines below are those of the first queue.
# pod_integ10k_queue_n2.sh -- written by make_n2.py from pod_integ10k_queue.sh. Continuation of card PREREG_V8F_PROOF_ADD46 by its note N2:
# the reader arms are those the first queue trained (no training here; checkpoints checked by sha256), the train-log step is check_k8_n2.py, own markers
# START_PRICE10K.txt / PRICE10K_STOP.txt / PRICE10K_DONE.txt. The text below is the original apart from these edits.
# pod_integ10k_queue.sh -- ON THE POD (nohup setsid, started by chain_i10k.sh). Card PREREG_V8F_PROOF_ADD46 (INTEG10K-PRE) + note N1.
# Steps, one GPU job at a time, a slot wait before EVERY job (fewer than 3 GPU compute processes AND more than 28 GB of memory.max minus the anon line of memory.stat free, on two probes 45 s apart; fail closed):
#   1 train the reader arms LR16_s0, LR16_s1 (= arms TK16_s0, TK16_s1 of tok_probe_lr.py, held-out clips left out, 19,164 steps) into $W/lr16
#   2 the train-log check of card ADD43 (control k8)      3 the reader rows on the held-out clips (tok_reader.py extract)     4 the quantile maps (qmap_fit.py)
#   5 twelve rollouts of the card: per model seed k: R0 RLb RR RRid RR0 (reader seed k), plus RRx_s0 (model 0 with reader 1) and RRx_s1 (model 1 with reader 0)
# Folder: W=/workspace/v8proof/integ10k ; code of this queue in $W/build ; rollout outputs $W/t1/<NAME>_s<k>/ (dump_<NAME>_s<k>_m10_causal_heads, ticks/, t1v8f_*.json, T1_DONE)
# Markers: $W/markers/<step>.done (a relaunch skips finished steps), $W/START_PRICE10K.txt (noclobber), $W/PRICE10K_DONE.txt or $W/PRICE10K_STOP.txt (reason). DRY=1 prints every command and runs nothing.
# Uses no pgrep, no pkill and no rm. Never writes outside $W.
set -u
W=${I10_W:-/workspace/v8proof/integ10k}
B=$W/build
PY=${I10_PY:-/workspace/venv/bin/python}
TREE=$B/tree_ov     # overlay tree: stack = link to $W/tree/stack, taniteval = the taniteval of commit cdbff971 (the pinned tool imports taniteval from <tree>/taniteval); built by chain_i10k.sh
LEAD_TREE=${I10_LEAD_TREE:-/workspace/v8proof/tree_leadsup}
V=/workspace/v8proof
CACHE_TR=$V/data/cache_pod
JOIN_TR=$V/data/inputs/join_train.npz
TRACKS=$V/data/inputs/native_tracks.pt
GRID=$V/hierrep/pins/eval_grid_windows.json
JOINE=$V/hierrep/pins/join_eval141.npz
EVALC=$W/eval_cache
PINNED=61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397
DRY=${DRY:-0}
TMO=${I10_TIMEOUT:-2700}
MINGB=${I10_MIN_FREE_GB:-28}
MAXGPU=3
LOG=$W/queue_i10k.log
MM=${I10_MEMMAX_FILE:-/sys/fs/cgroup/memory.max}; MS=${I10_MEMSTAT_FILE:-/sys/fs/cgroup/memory.stat}
ARMS=(0 1)

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { if [ "$DRY" = 1 ]; then echo "$*"; else echo "[$(ts)] $*" >> "$LOG"; fi; }
stop() { if [ "$DRY" = 1 ]; then echo "STOP $*"; exit 3; fi; echo "$(ts) $*" > "$W/PRICE10K_STOP.txt"; log "STOP $*"; exit 3; }
runlog() { # LOGFILE then the command words: print it in DRY, else run it with stdout and stderr appended to LOGFILE
  local lf=$1; shift
  if [ "$DRY" = 1 ]; then echo "CMD: $* >> $lf"; return 0; fi
  "$@" >> "$lf" 2>&1
}

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

# ---------------------------------------------------------------- start
if [ "$DRY" != 1 ]; then
  [ -d "$W" ] || { echo "REFUSED: no $W"; exit 2; }
  ( set -C; echo "$(ts) pid $$" > "$W/START_PRICE10K.txt" ) 2>/dev/null || { echo "REFUSED: $W/START_PRICE10K.txt exists (a queue was started)"; exit 2; }
  mkdir -p "$W/markers" "$W/lr16" "$W/rows" "$W/qmap" "$W/qmap_sweep" "$W/t1"
fi
log "queue N2 start (DRY=$DRY): card ADD47 (PRICE10K): maps of the sweep and six rollouts; reader arms and checkpoints of card ADD46"
for f in qmap_fit_lvl.py price_maps_check.py QMAP_IDENTITY.json LS_REF40_WINDOWS.json reader_controls.py v8f_t1_eval_reader_rule.py tok_reader.py tok_probe_lr.py qmap_fit.py qcal_map.py dist_rule.py v8f_t1_eval_rule.py v8f_t1_eval_rule_mix.py; do
  [ "$DRY" = 1 ] || [ -f "$B/$f" ] || stop "BUILD-FILE-MISSING $f"
done
if [ "$DRY" != 1 ]; then
  [ -d "$TREE/taniteval/taniteval" ] && [ -e "$TREE/stack/scripts/v8f_t1_eval.py" ] || stop "NO-OVERLAY-TREE $TREE"
  got=$(sha256sum "$TREE/stack/scripts/v8f_t1_eval.py" | cut -c1-64); [ "$got" = "$PINNED" ] || stop "PINNED-SHA $got"
  for f in "$TRACKS" "$GRID" "$JOINE" "$JOIN_TR"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
  [ -f "$EVALC/eval/manifest.jsonl" ] || stop "NO-EVAL-CACHE"
  for k in "${ARMS[@]}"; do [ -f "$W/ckpt/UE_s$k/A/ckpt_last.pt" ] || stop "NO-CKPT UE_s$k"; done
fi

# ---------------------------------------------------------------- 1 maps of the sweep (CPU) and their check (control m1)
for k in "${ARMS[@]}"; do
  [ "$DRY" = 1 ] || [ -f "$W/rows/rows_LR16_s$k.npz" ] || stop "NO-READER-ROWS s$k"
  for L in 35 50; do
    if ! done_step "pmap${k}_$L"; then
      runlog "$W/qmap_sweep.log" env QMAP_LEVEL=$L timeout 600 "$PY" "$B/qmap_fit_lvl.py" --rows "$W/rows/rows_LR16_s$k.npz" --arm "LR16_s${k}_q$L" --out-dir "$W/qmap_sweep" || stop "QMAP-SWEEP-FAIL s$k q$L"
      mark "pmap${k}_$L"
    fi
  done
done
if ! done_step pmapcheck; then
  runlog "$W/qmap_sweep.log" timeout 600 "$PY" "$B/price_maps_check.py" --q20-dir "$W/qmap" --sweep-dir "$W/qmap_sweep" --identity "$B/QMAP_IDENTITY.json" --out "$W/qmap_sweep/PRICE_MAPS_CHECK.json" || stop "MAPS-CHECK-FAIL"
  mark pmapcheck
fi

# ---------------------------------------------------------------- 5 the twelve rollouts
rollout() { # NAME FORM MODELSEED READERSEED
  local name=$1 form=$2 m=$3 r=$4
  local arm=${name}_s$m out=$W/t1/${name}_s$m tag=${name}_s${m}_m10_causal_heads
  done_step "ro_$arm" && return 0
  [ "$DRY" = 1 ] || { [ -e "$out" ] && stop "OUTPUT-EXISTS $out"; }
  wait_slot "rollout $arm"
  [ "$DRY" = 1 ] || mkdir -p "$out/ticks"
  local extra=()
  case "$form" in
    RR|RRid|RR0) extra+=(--reader "$W/lr16/TK16_s$r/ckpt_last.pt") ;;
  esac
  [ "$form" = RR ] && extra+=(--qmap "${QMAP_FILE:?}")          # card ADD47: the map file is given by the caller
  [ "$form" = R0 ] || extra+=(--rule-log "$out/ticks/rule_log_$arm.npz")
  if [ "$DRY" = 1 ]; then
    echo "CMD: DISTRULE_DIR=$B PYTHONPATH=$TREE/stack timeout $TMO $PY $B/v8f_t1_eval_reader_rule.py --tree $TREE --form $form ${extra[*]} run --ckpt $W/ckpt/UE_s$m/A/ckpt_last.pt --tag $tag --tracks $TRACKS --eval-cache $EVALC --eval-grid $GRID --eval-join $JOINE --out-dir $out --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks $out/ticks --dump-heads --device cuda"
    return 0
  fi
  echo "arm $arm form $form model seed $m reader seed $r ckpt sha256 $(sha256sum "$W/ckpt/UE_s$m/A/ckpt_last.pt" | cut -c1-64) started $(ts)" > "$out/ticks/run_$arm.txt"
  DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" timeout "$TMO" "$PY" "$B/v8f_t1_eval_reader_rule.py" --tree "$TREE" --form "$form" "${extra[@]}" run --ckpt "$W/ckpt/UE_s$m/A/ckpt_last.pt" --tag "$tag" --tracks "$TRACKS" --eval-cache "$EVALC" --eval-grid "$GRID" --eval-join "$JOINE" --out-dir "$out" --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks "$out/ticks" --dump-heads --device cuda >> "$out/t1.log" 2>&1
  local rc=$?
  echo "rc $rc finished $(ts)" >> "$out/ticks/run_$arm.txt"
  [ $rc = 0 ] || stop "ROLLOUT-RC $rc $arm"
  local n; n=$(ls "$out/dump_$tag" 2>/dev/null | grep -c -- '^ep.*npz$')
  [ "$n" = 141 ] || stop "ROLLOUT-DUMP-COUNT $n $arm"
  touch "$out/T1_DONE"
  mark "ro_$arm"
  log "done $arm"
}
for k in "${ARMS[@]}"; do
  QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50 RR "$k" "$k"      # the deciding level first
done
for k in "${ARMS[@]}"; do
  QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q35.json" rollout I10RQ35 RR "$k" "$k"
  QMAP_FILE="$B/QMAP_IDENTITY.json" rollout I10RRaw RR "$k" "$k"
done
if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi
echo "$(ts) all steps done" > "$W/PRICE10K_DONE.txt"
log "INTEG10K_DONE"
