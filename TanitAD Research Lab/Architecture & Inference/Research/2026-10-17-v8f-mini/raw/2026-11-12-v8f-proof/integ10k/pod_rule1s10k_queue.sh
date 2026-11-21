#!/usr/bin/env bash
# pod_rule1s10k_queue.sh -- ON THE POD (nohup setsid, started by chain_rule1s10k.sh). Card PREREG_V8F_PROOF_ADD59 (RULE1S10K), written from pod_brake10k_queue.sh (rollout
# function of pod_accwin10k_queue.sh: --readout / --readout-log) with the wrapper v8f_t1_eval_reader_rule_rz.py of card ADD55 (sha256 prefix fa7207d72311b5cf, checked below).
# Fifteen rollouts, no training, no map fit, no limit, no tau:
#   control  I10R0BkID_s0 (no rule, backward readout, 5 steps, model seed 0; must equal the dump of I10R0B5_s0 of card ADD54, control r1, comparison marker r1scmp)
#   per model seed k (reader seed = model seed): backward readout: I10RLcBk5, I10RQ50Bk5 (5 steps), I10R0Bk10, I10RLcBk10, I10RQ50Bk10 (10 steps);
#                                                 causal_z readout: I10RLcZ5, I10RQ50Z5 (5 steps; second readout, beside; run last)
# It waits for the end marker of the previous queue (WAIT_DONE / WAIT_STOP, defaults BRAKE10K_DONE.txt / BRAKE10K_STOP.txt, in $W) before its first rollout.
# Markers START_RULE1S10K.txt / RULE1S10K_STOP.txt / RULE1S10K_DONE.txt. DRY=1 prints every command and runs nothing.
# Folder: W=/workspace/v8proof/integ10k ; code in $W/build ; rollout outputs $W/t1/<NAME>_s<k>/ (dump_<NAME>_s<k>_m10_causal_heads, ticks/, t1v8f_*.json, T1_DONE)
# Step markers: $W/markers/<step>.done (a relaunch skips finished steps). Uses no pgrep, no pkill and no rm. Never writes outside $W.
# Slot wait before EVERY job: fewer than 3 GPU compute processes AND more than 28 GB of memory.max minus the anon line of memory.stat free, on two probes 45 s apart; fail closed.
# The header lines of the earlier queues (cards ADD46 to ADD57) are those of pod_brake10k_queue.sh; the rollout function keeps the argv of the earlier queues, minus the
# parameters this card does not use (--tau, --b-max, --a-max, --b-lim, --slope), which are not read from the environment here.
set -u
W=${I10_W:-/workspace/v8proof/integ10k}
B=$W/build
PY=${I10_PY:-/workspace/venv/bin/python}
TREE=$B/tree_ov     # overlay tree: stack = link to $W/tree/stack, taniteval = the taniteval of commit cdbff971 (the pinned tool imports taniteval from <tree>/taniteval); built by chain_i10k.sh
WRAP=v8f_t1_eval_reader_rule_rz.py
WRAP_SHA16=fa7207d72311b5cf
V=/workspace/v8proof
JOIN_TR=$V/data/inputs/join_train.npz
TRACKS=$V/data/inputs/native_tracks.pt
GRID=$V/hierrep/pins/eval_grid_windows.json
JOINE=$V/hierrep/pins/join_eval141.npz
EVALC=$W/eval_cache
PINNED=61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397
DRY=${DRY:-0}
TMO=${I10_TIMEOUT:-5400}
MINGB=${I10_MIN_FREE_GB:-28}
MAXGPU=3
LOG=$W/queue_i10k.log
MM=${I10_MEMMAX_FILE:-/sys/fs/cgroup/memory.max}; MS=${I10_MEMSTAT_FILE:-/sys/fs/cgroup/memory.stat}
WAIT_DONE=${WAIT_DONE:-BRAKE10K_DONE.txt}
WAIT_STOP=${WAIT_STOP:-BRAKE10K_STOP.txt}
ARMS=(0 1)

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { if [ "$DRY" = 1 ]; then echo "$*"; else echo "[$(ts)] $*" >> "$LOG"; fi; }
stop() { if [ "$DRY" = 1 ]; then echo "STOP $*"; exit 3; fi; echo "$(ts) $*" > "$W/RULE1S10K_STOP.txt"; log "STOP $*"; exit 3; }

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
  ( set -C; echo "$(ts) pid $$" > "$W/START_RULE1S10K.txt" ) 2>/dev/null || { echo "REFUSED: $W/START_RULE1S10K.txt exists (a queue was started)"; exit 2; }
  mkdir -p "$W/markers" "$W/lr16" "$W/rows" "$W/qmap" "$W/qmap_sweep" "$W/t1"
fi
log "queue start (DRY=$DRY): card ADD59 (lead rule at the 1.0 s cycle, backward and ZERO readouts): fifteen rollouts; waits for $WAIT_DONE / $WAIT_STOP; checkpoints and maps of cards ADD46/ADD47"
if [ "$DRY" != 1 ]; then
  t0=$(date +%s)
  until [ -f "$W/$WAIT_DONE" ] || [ -f "$W/$WAIT_STOP" ]; do
    [ $(( $(date +%s) - t0 )) -gt 28800 ] && stop "WAIT-PREVIOUS-QUEUE-TIMEOUT"
    sleep 30
  done
  log "the previous queue has ended ($WAIT_DONE or $WAIT_STOP); the rollouts of card ADD59 start"
fi
for f in cmp_dumps.py $WRAP LS_REF40_WINDOWS.json reader_controls.py v8f_t1_eval_reader_rule.py tok_reader.py tok_probe_lr.py qmap_fit.py qcal_map.py dist_rule.py v8f_t1_eval_rule.py v8f_t1_eval_rule_mix.py; do
  [ "$DRY" = 1 ] || [ -f "$B/$f" ] || stop "BUILD-FILE-MISSING $f"
done
if [ "$DRY" != 1 ]; then
  got=$(sha256sum "$B/$WRAP" | cut -c1-16); [ "$got" = "$WRAP_SHA16" ] || stop "WRAPPER-SHA $got"     # control r9
  [ -d "$TREE/taniteval/taniteval" ] && [ -e "$TREE/stack/scripts/v8f_t1_eval.py" ] || stop "NO-OVERLAY-TREE $TREE"
  got=$(sha256sum "$TREE/stack/scripts/v8f_t1_eval.py" | cut -c1-64); [ "$got" = "$PINNED" ] || stop "PINNED-SHA $got"
  for f in "$TRACKS" "$GRID" "$JOINE" "$JOIN_TR"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
  [ -f "$EVALC/eval/manifest.jsonl" ] || stop "NO-EVAL-CACHE"
  for k in "${ARMS[@]}"; do [ -f "$W/ckpt/UE_s$k/A/ckpt_last.pt" ] || stop "NO-CKPT UE_s$k"; done
fi

# ---------------------------------------------------------------- 1 inputs (no map is fitted here) and the banked runs this card compares with
for k in "${ARMS[@]}"; do
  [ "$DRY" = 1 ] || [ -f "$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" ] || stop "NO-MAP-Q50 s$k"
  [ "$DRY" = 1 ] || [ -f "$W/lr16/TK16_s$k/ckpt_last.pt" ] || stop "NO-READER-CKPT s$k"
  for nm in I10R0B5 I10R0Z5 I10R0 I10RLc I10RQ50 I10R0M5 I10RLcM5 I10RQ50M5; do
    [ "$DRY" = 1 ] || [ -f "$W/t1/${nm}_s$k/T1_DONE" ] || stop "NO-BANKED-$nm s$k"
  done
done

# ---------------------------------------------------------------- 2 the fifteen rollouts
rollout() { # NAME FORM MODELSEED READERSEED ; READOUT and REPLAN come from the environment of the call (both required), QMAP_FILE for the reader-fed form
  local name=$1 form=$2 m=$3 r=$4
  local arm=${name}_s$m out=$W/t1/${name}_s$m tag=${name}_s${m}_m10_causal_heads
  done_step "ro_$arm" && return 0
  [ -n "${READOUT:-}" ] && [ -n "${REPLAN:-}" ] || stop "ROLLOUT-PARAM-MISSING $arm"
  [ "$DRY" = 1 ] || { [ -e "$out" ] && stop "OUTPUT-EXISTS $out"; }
  wait_slot "rollout $arm"
  [ "$DRY" = 1 ] || mkdir -p "$out/ticks"
  local extra=()
  case "$form" in
    RR) extra+=(--reader "$W/lr16/TK16_s$r/ckpt_last.pt" --qmap "${QMAP_FILE:?}") ;;      # the map file is given by the caller
  esac
  extra+=(--readout "$READOUT" --readout-log "$out/ticks/readout_log_$arm.json")
  [ "$form" = R0 ] || extra+=(--rule-log "$out/ticks/rule_log_$arm.npz")
  if [ "$DRY" = 1 ]; then
    echo "CMD: DISTRULE_DIR=$B PYTHONPATH=$TREE/stack timeout $TMO $PY $B/$WRAP --tree $TREE --form $form ${extra[*]} run --ckpt $W/ckpt/UE_s$m/A/ckpt_last.pt --tag $tag --tracks $TRACKS --eval-cache $EVALC --eval-grid $GRID --eval-join $JOINE --out-dir $out --replan-steps $REPLAN --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks $out/ticks --dump-heads --device cuda"
    return 0
  fi
  echo "arm $arm form $form readout $READOUT replan steps $REPLAN model seed $m reader seed $r ckpt sha256 $(sha256sum "$W/ckpt/UE_s$m/A/ckpt_last.pt" | cut -c1-64) started $(ts)" > "$out/ticks/run_$arm.txt"
  DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" timeout "$TMO" "$PY" "$B/$WRAP" --tree "$TREE" --form "$form" "${extra[@]}" run --ckpt "$W/ckpt/UE_s$m/A/ckpt_last.pt" --tag "$tag" --tracks "$TRACKS" --eval-cache "$EVALC" --eval-grid "$GRID" --eval-join "$JOINE" --out-dir "$out" --replan-steps "$REPLAN" --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks "$out/ticks" --dump-heads --device cuda >> "$out/t1.log" 2>&1
  local rc=$?
  echo "rc $rc finished $(ts)" >> "$out/ticks/run_$arm.txt"
  [ $rc = 0 ] || stop "ROLLOUT-RC $rc $arm"
  local n; n=$(ls "$out/dump_$tag" 2>/dev/null | grep -c -- '^ep.*npz$')
  [ "$n" = 141 ] || stop "ROLLOUT-DUMP-COUNT $n $arm"
  touch "$out/T1_DONE"
  mark "ro_$arm"
  log "done $arm"
}
# control r1 first: backward readout, no rule, 5 steps, model seed 0 must give the dump of the banked I10R0B5 (card ADD54)
READOUT=backward REPLAN=5 rollout I10R0BkID R0 0 0
if [ "$DRY" != 1 ] && ! done_step r1scmp; then
  mkdir -p "$W/small_rule1s"
  "$PY" "$B/cmp_dumps.py" "$W/t1/I10R0B5_s0/dump_I10R0B5_s0_m10_causal_heads" "$W/t1/I10R0BkID_s0/dump_I10R0BkID_s0_m10_causal_heads" > "$W/small_rule1s/CMP_R0BkID.txt" 2>&1 || stop "R1-RERUN-DIFFERS"
  mark r1scmp
fi
# primary readout (backward): both model seeds before the second readout, so a stopped queue leaves the primary reads
for k in "${ARMS[@]}"; do
  Q="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"
  READOUT=backward REPLAN=5 rollout I10RLcBk5 RLc "$k" "$k"
  READOUT=backward REPLAN=5 QMAP_FILE="$Q" rollout I10RQ50Bk5 RR "$k" "$k"
  READOUT=backward REPLAN=10 rollout I10R0Bk10 R0 "$k" "$k"
  READOUT=backward REPLAN=10 rollout I10RLcBk10 RLc "$k" "$k"
  READOUT=backward REPLAN=10 QMAP_FILE="$Q" rollout I10RQ50Bk10 RR "$k" "$k"
done
# second readout (ZERO = causal_z), beside
for k in "${ARMS[@]}"; do
  Q="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"
  READOUT=causal_z REPLAN=5 rollout I10RLcZ5 RLc "$k" "$k"
  READOUT=causal_z REPLAN=5 QMAP_FILE="$Q" rollout I10RQ50Z5 RR "$k" "$k"
done
if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi
echo "$(ts) all steps done" > "$W/RULE1S10K_DONE.txt"
log "RULE1S10K_DONE"
