#!/usr/bin/env bash
# pod_gapspread10k_queue.sh -- card ADD134 (GAPSPREAD10K): 4 rollouts, model seeds 0 to 3, through the wrapper variant rzgfhqetjv of card ADD102 (hash 28bc31f21ecb4ccb, unchanged):
# the candidate form (reader-fed form RR with --vis-gap-file and --vis-closing, no label at inference) with the gap file VGAPSP90.npz (the base gap times the margin placed by the spread
# of the three fit files) at readout K with no pre-roll (setting K0; run name I10RQ50KF5Hb20gSP90; --rule-every 5), pinned grid. Each run is called as the banked A90 run of the same model seed
# (I10RQ50KF5Hb20gA90) with the gap file exchanged; the reader seed and map are those of model seed m for m < 2 and of seed m - 2 for m >= 2. A changed gap input of the rule is a study arm.
# Step 0 gate: exit 4 (a line starting GATE) unless $GE/lead_tests/lead_pytest_queue.out exists and its last non-empty line is a pytest pass line (the test of this kit passed in the lead's run).
# Step 1 builds no gap file: the queue stops (no rollout) unless $GF holds GAPSPREAD10K_GAPFILE_RECORD.json and VGAPSP90.npz, the record has card PREREG_V8F_PROOF_ADD134, pass_p0 true
# and files.VGAPSP90.sha256 equal to the sha256 of the file beside it.
# Step 2: the four rollouts, one after the other, each started only with fewer than 3 GPU compute processes and more than 28 GB free (memory.max minus anon); a run folder that exists is refused.
# Time rule: before each start the seconds since the first line of $GE/GAPSPREAD10K_START.txt must be at most GS_STOP_S (default 3600, 1.0 pod-hours), else $GE/GAPSPREAD10K_STOP.txt. A failed rollout writes the stop marker.
# Shared pod state file FORK_STATE_FILE (default /workspace/FORK_STATE.txt; empty = no write): a BUSY line before the first rollout and an IDLE line on every exit path after it.
# Done marker $GE/GAPSPREAD10K_DONE.txt. Log $GE/gapspread10k_queue.log. DRY=1 prints every command and creates nothing. RESUME=1 continues after a stop marker. Nothing is removed or moved.
# Banked runs needed (all four model seeds, T1_DONE): I10RLcKF5Hb20 I10RQ50KF5Hb20 I10RQ50KF5Hb20gGEc I10RQ50KF5Hb20gA90. Written by make_gapspread_queue.py from the queue of card ADD131 (sha256 prefix 03be2485ff4d9c94). No training.
set -u
W=${I10_W:-/workspace/v8proof/integ10k}
GE=${GS_ROOT:-/workspace/v8proof/gapspread10k}
GP=$GE
B=$W/build
PY=${I10_PY:-/workspace/venv/bin/python}
TREE=$B/tree_ov
V=/workspace/v8proof
GF=${GF:-$GE/gapfiles}
GSREC=$GF/GAPSPREAD10K_GAPFILE_RECORD.json
KITG=/workspace/v8proof/gaperr/kit_gap
KQ=${GS_KIT:-$GE/kit_queue}
GREC=/workspace/v8proof/gaperr/gapfiles/GAPERR_GAPFILES_RECORD.json
BASEF=/workspace/v8proof/cropscale/gapfiles/VGAPLE.npz
BASER=/workspace/v8proof/cropscale/gapfiles/CROPSCALE_GAPFILES_RECORD.json
LAF=/workspace/v8proof/fallattr/gapfiles/VGAPA_LA.npz
CHK=${GR_CHECK:-/workspace/v8proof/visrule10k/check_v}
TRACKS=$V/data/inputs/native_tracks.pt
GRID=$V/hierrep/pins/eval_grid_windows.json
JOINE=$V/hierrep/pins/join_eval141.npz
EVALC=$W/eval_cache
PINNED=61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397
WRAP=$B/v8f_t1_eval_reader_rule_rzgfhqetjv.py
LG=$W/longroll
G30=$LG/grid_p30.json
G60=$LG/grid_p60.json
SM=$GP/small_gapspread10k
DRY=${DRY:-0}
TMO=${I10_TIMEOUT:-5400}
MINGB=${I10_MIN_FREE_GB:-28}
MAXGPU=3
STOPS=${GS_STOP_S:-3600}
RESUME=${RESUME:-0}
SEEDSW=ALL
ARMS=(0 1 2 3); PS=""
LOG=$GP/gapspread10k_queue.log
MM=${I10_MEMMAX_FILE:-/sys/fs/cgroup/memory.max}; MS=${I10_MEMSTAT_FILE:-/sys/fs/cgroup/memory.stat}

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { if [ "$DRY" = 1 ]; then echo "$*"; else echo "[$(ts)] $*" >> "$LOG"; fi; }
stop() { if [ "$DRY" = 1 ]; then echo "STOP $*"; exit 3; fi; echo "$(ts) $*" >> "$GP/GAPSPREAD10K_STOP.txt"; log "STOP $*"; exit 3; }

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
done_step() { [ -f "$GP/markers/$1.done" ]; }
mark() { [ "$DRY" = 1 ] || { mkdir -p "$W/markers"; echo "$(ts)" > "$GP/markers/$1.done"; }; }
grid_counts() { # GRIDFILE -> "<clips> <windows>" (windows counted from the lists, not from the n_windows field)
  "$PY" - "$1" <<'PYEOF'
import json, sys
g = json.load(open(sys.argv[1], encoding="utf-8"))
print(len(g["order"]), sum(len(g["windows"][c]) for c in g["order"]))
PYEOF
}

# step 0 gate: the test of this kit passed in the lead's run (control p4); nothing is created before this point
gate_ok() { # the lead's test output exists and its last non-empty line is a pytest pass line
  local f="$GE/lead_tests/lead_pytest_queue.out" last
  [ -f "$f" ] || return 1
  last=$(grep -v '^[[:space:]]*$' "$f" | tail -n 1 | tr -d '\r')
  [[ "$last" =~ [0-9]+\ passed ]] || return 1
  [[ "$last" =~ (failed|error) ]] && return 1
  return 0
}
if [ "$DRY" = 1 ]; then
  echo "GATE: needs $GE/lead_tests/lead_pytest_queue.out ending in a pytest pass line (exit 4 without it)"
else
  gate_ok || { echo "GATE: REFUSED, step 0 gate: $GE/lead_tests/lead_pytest_queue.out is missing or does not end in a pytest pass line"; exit 4; }
fi
check_time() { # step name; seconds since the first line of the start file at most STOPS
  if [ "$DRY" = 1 ]; then echo "STOPRULE before $1: seconds since the first line of $GP/GAPSPREAD10K_START.txt at most $STOPS"; return 0; fi
  local t0 now
  t0=$(head -n 1 "$GP/GAPSPREAD10K_START.txt" 2>/dev/null | awk '{print $1}')
  [[ "$t0" =~ ^[0-9]+$ ]] || stop "START-LINE-UNREADABLE before $1"
  now=$(date +%s)
  [ $(( now - t0 )) -le "$STOPS" ] || stop "STOP-RULE $(( now - t0 )) s since the first start line, limit $STOPS s, before $1"
}
if [ "$DRY" != 1 ]; then
  [ -d "$W" ] || { echo "REFUSED: no $W"; exit 2; }
  mkdir -p "$GP" || { echo "REFUSED: cannot make $GP"; exit 2; }
  if [ "$RESUME" = 1 ] && [ -f "$GP/GAPSPREAD10K_START.txt" ]; then
    echo "$(date +%s) resume $(ts) pid $$" >> "$GP/GAPSPREAD10K_START.txt"
  else
    ( set -C; echo "$(date +%s) $(ts) pid $$" > "$GP/GAPSPREAD10K_START.txt" ) 2>/dev/null || { echo "REFUSED: $GP/GAPSPREAD10K_START.txt exists (a queue was started; RESUME=1 continues)"; exit 2; }
  fi
  mkdir -p "$GP/markers" "$W/t1" "$SM"
fi
log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD134 (GAPSPREAD10K): the candidate form with the gap file VGAPSP90.npz, at readout K with no pre-roll, model seeds ${ARMS[*]}"
for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzgfhqetjv.py v8f_t1_eval_reader_rule_rzgfhqetj.py v8f_t1_eval_reader_rule_rzgfhqet.py v8f_t1_eval_reader_rule_rzgfhqe.py closest.py v8f_t1_eval_reader_rule_rzgfhqc.py v8f_t1_eval_reader_rule_rzgfhq.py v8f_t1_eval_reader_rule_rzgfh.py v8f_t1_eval_reader_rule_rz.py check_preroll_b.py make_longroll_grid.py dist_rule.py v8f_t1_eval_rule.py v8f_t1_eval_rule_mix.py v8f_t1_eval_reader_rule.py tok_reader.py tok_probe_lr.py qmap_fit.py qcal_map.py reader_controls.py; do
  [ "$DRY" = 1 ] || [ -f "$B/$f" ] || stop "BUILD-FILE-MISSING $f"
done
if [ "$DRY" != 1 ]; then
  [ -d "$TREE/taniteval/taniteval" ] && [ -e "$TREE/stack/scripts/v8f_t1_eval.py" ] || stop "NO-OVERLAY-TREE $TREE"
  got=$(sha256sum "$TREE/stack/scripts/v8f_t1_eval.py" | cut -c1-64); [ "$got" = "$PINNED" ] || stop "PINNED-SHA $got"
  for f in "$TRACKS" "$GRID" "$JOINE"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
  [ -f "$EVALC/eval/manifest.jsonl" ] || stop "NO-EVAL-CACHE"
  for k in "${ARMS[@]}"; do
    [ -f "$W/ckpt/UE_s$k/A/ckpt_last.pt" ] || stop "NO-CKPT UE_s$k"
  done
fi

# rollout NAME FORM MODELSEED READERSEED     environment: RGRID (grid file, default pinned), PREROLL (default 0), PREEXEC (plan|logged, optional), NAV (default deadreckon),
#   READOUT, ACCGAIN, PATHEXEC, KAPPA, EVERY, REPLAN (default 10), QMAP_FILE
rollout() {
  local name=$1 form=$2 m=$3 r=$4
  local arm=${name}_s$m out=$W/t1/${name}_s$m tag=${name}_s${m}_m10_causal_heads
  local grid=${RGRID:-$GRID} pre=${PREROLL:-0} nav=${NAV:-deadreckon}
  done_step "ro_$arm" && return 0
  [ -f "$out/T1_DONE" ] && return 0
  [ "$DRY" = 1 ] || { [ -e "$out" ] && stop "OUTPUT-EXISTS $out"; }
  check_time "rollout $arm"
  wait_slot "rollout $arm"
  [ "$DRY" = 1 ] || mkdir -p "$out/ticks"
  local extra=()
  case "$form" in
    RR|RRc|RRh|RRe) extra+=(--reader "$W/lr16/TK16_s$r/ckpt_last.pt") ;;
  esac
  [ -n "${EVERY:-}" ] && extra+=(--rule-every "$EVERY")
  [ -n "${KAPPA:-}" ] && extra+=(--kappa-max "$KAPPA")
  [ -n "${ACCGAIN:-}" ] && extra+=(--acc-gain "$ACCGAIN")
  [ "${PATHEXEC:-0}" = 1 ] && extra+=(--path-exec)
  [ -n "${READOUT:-}" ] && extra+=(--readout "$READOUT" --readout-log "$out/ticks/readout_log_$arm.json")
  extra+=(--preroll-steps "$pre")
  [ -n "${PREEXEC:-}" ] && extra+=(--preroll-exec "$PREEXEC")
  case "$form" in RR|RRc|RRh|RRe) extra+=(--qmap "${QMAP_FILE:?}") ;; esac
  if [ "$form" = RRe ]; then
    [ "$DRY" = 1 ] || [ -f "${TAB:-}" ] || stop "NO-TABLE-FILE ${TAB:-unset} $arm"
    extra+=(--closing-table "${TAB:?}")
  fi
  [ "$form" = R0 ] || extra+=(--rule-log "$out/ticks/rule_log_$arm.npz")
  if [ -n "${VGAP:-}" ]; then
    [ "$DRY" = 1 ] || [ -f "$VGAP" ] || stop "NO-GAP-FILE $VGAP $arm"
    extra+=(--vis-gap-file "$VGAP")
  fi
  [ "${VCLOSE:-0}" = 1 ] && extra+=(--vis-closing)
  if [ -n "${RJOIN:-}" ]; then
    [ "$DRY" = 1 ] || [ -f "$RJOIN" ] || stop "NO-RULE-JOIN $RJOIN $arm"
    extra+=(--rule-join "$RJOIN")
  fi
  if [ "$DRY" = 1 ]; then
    echo "CMD: DISTRULE_DIR=$B PYTHONPATH=$TREE/stack timeout $TMO $PY $WRAP --tree $TREE --form $form ${extra[*]} run --ckpt $W/ckpt/UE_s$m/A/ckpt_last.pt --tag $tag --tracks $TRACKS --eval-cache $EVALC --eval-grid $grid --eval-join $JOINE --out-dir $out --replan-steps ${REPLAN:-10} --nav $nav --scene logged --schedule periodic --limit causal --dump-ticks $out/ticks --dump-heads --no-analyze --device cuda"
    return 0
  fi
  echo "arm $arm form $form replan steps ${REPLAN:-10} rule every ${EVERY:-10} readout ${READOUT:-pinned} kappa ${KAPPA:-none} preroll $pre exec ${PREEXEC:-plan} nav $nav grid $(basename "$grid") model seed $m reader seed $r ckpt sha256 $(sha256sum "$W/ckpt/UE_s$m/A/ckpt_last.pt" | cut -c1-64) wrapper sha256 $(sha256sum "$WRAP" | cut -c1-64) started $(ts)${RJOIN:+ rule-join $RJOIN rule join sha256 $(sha256sum "$RJOIN" | cut -c1-64)}${VGAP:+ vis-gap-file $VGAP gap file sha256 $(sha256sum "$VGAP" | cut -c1-64) vis-closing ${VCLOSE:-0}}" > "$out/ticks/run_$arm.txt"
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

step_inputs() { # step 1: the gap file of card ADD134 and its record are checked; nothing is built
  if [ "$DRY" = 1 ]; then
    echo "CHECK: $GF holds GAPSPREAD10K_GAPFILE_RECORD.json and VGAPSP90.npz"
    echo "CHECK: $GSREC has card PREREG_V8F_PROOF_ADD134, pass_p0 true and files.VGAPSP90.sha256 equal to the sha256 of the gap file"
    return 0
  fi
  [ -f "$GSREC" ] || stop "NO-GAP-RECORD $GSREC"
  [ -f "$GF/VGAPSP90.npz" ] || stop "NO-GAP-FILE $GF/VGAPSP90.npz"
  local adm
  adm=$("$PY" - "$GSREC" "$GF" <<'PYEOF'
import hashlib, json, os, sys
rec_path, gf = sys.argv[1:3]
rec = json.load(open(rec_path, encoding="utf-8"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if rec.get("card") != "PREREG_V8F_PROOF_ADD134":
    print("RECORD-IS-NOT-OF-THIS-CARD"); sys.exit(1)
if rec.get("pass_p0") is not True:
    print("PASS_P0-NOT-PASS"); sys.exit(1)
p = os.path.join(gf, "VGAPSP90.npz")
if not os.path.isfile(p) or ((rec.get("files") or {}).get("VGAPSP90") or {}).get("sha256") != sha(p):
    print("GAP-FILE-IS-NOT-THE-RECORDED-ONE VGAPSP90"); sys.exit(1)
print("ADMIT-OK gap file")
PYEOF
) || stop "INPUT-ADMISSION $adm"
  log "$adm"
}
[ "$DRY" = 1 ] || [ -f "$GRID" ] || stop "NO-PINNED-GRID"
if [ "$DRY" != 1 ]; then
  CK="$CHK/CHECK_WRAPPER_V_DONE.txt"
  [ "$(tail -n 1 "$CK" 2>/dev/null)" = "ZZ CHECK-WRAPPER-V PASS" ] || stop "NO-PASS-OF-THE-WRAPPER-CHECK"
  got=$(sha256sum "$WRAP" | cut -c1-16); [ "$got" = "28bc31f21ecb4ccb" ] || stop "WRAPPER-SHA $got"
  grep -q "28bc31f21ecb4ccb" "$CHK/CHECK_WRAPPER_V.hashes" 2>/dev/null || stop "CHECK-RECORD-HAS-NOT-THIS-WRAPPER"
  for nm in I10RLcKF5Hb20 I10RQ50KF5Hb20 I10RQ50KF5Hb20gGEc I10RQ50KF5Hb20gA90; do
    for k in "${ARMS[@]}"; do [ -f "$W/t1/${nm}_s$k/T1_DONE" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done
  done
fi
step_inputs
# fork-state begin
FS=${FORK_STATE_FILE-/workspace/FORK_STATE.txt}
BUSYSET=0
fork_idle() { [ "$BUSYSET" = 1 ] || return 0; BUSYSET=0; [ -n "$FS" ] && echo "IDLE $(ts)" >> "$FS"; return 0; }
trap fork_idle EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [ "$DRY" = 1 ]; then
  echo "FORKSTATE: append 'BUSY card ADD134 rollouts of the v8proof fork <UTC>' to ${FS:-(no write)} before the first rollout and 'IDLE <UTC>' on every exit path after it"
elif [ -n "$FS" ]; then
  echo "BUSY card ADD134 rollouts of the v8proof fork $(ts)" >> "$FS" && BUSYSET=1
fi
# fork-state end
for k in "${ARMS[@]}"; do
  r=$(( k >= 2 ? k - 2 : k ))
  READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${r}_q50.json" VGAP="$GF/VGAPSP90.npz" VCLOSE=1 rollout I10RQ50KF5Hb20gSP90 RR "$k" "$r"
done
if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi
cp -f "$LOG" "$SM/" 2>/dev/null
NAMES="I10RQ50KF5Hb20gSP90"
for nm in $NAMES; do for k in "${ARMS[@]}"; do cp -f "$W/t1/${nm}_s$k/ticks/run_${nm}_s$k.txt" "$W/t1/${nm}_s$k/ticks/preroll_log_${nm}_s$k.json" "$SM/" 2>/dev/null; done; done
echo "ZZ GAPSPREAD10K DONE $(ts) all steps done" > "$GP/GAPSPREAD10K_DONE.txt"
log "GAPSPREAD10K_DONE"
