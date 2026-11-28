#!/usr/bin/env bash
# pod_gapmargin10k_queue.sh -- card ADD117 (GAPMARGIN10K): 8 rollouts, model seeds 0 and 1, through the wrapper variant rzgfhqetjv of card ADD102 (hash 28bc31f21ecb4ccb, unchanged):
# the candidate form (reader-fed form RR with --vis-gap-file and --vis-closing, no label at inference) with the camera gap file of card ADD106 scaled by 0.90 (A90) and by 0.80 (A80),
# at readout K with no pre-roll (I10RQ50KF5Hb20gA90, ...gA80; --rule-every 5) and at the convention (I10RQ50Hb20gA90, ...gA80; pinned readout, the rule every 10 steps), pinned grid.
# A margin on the gap is a parameter of the rule's input: a study arm, measured here; its adoption is a design decision.
# Step 0 gate: exit 4 unless $GE/lead_tests/lead_pytest_queue.out exists (the test of this kit passed in the lead's run before the queue start).
# Step 1: the base gap file (sha256 prefix 55de2f7c) is checked; when $GE/gapfiles is absent make_margin_gapfiles.py runs on the CPU;
# the queue stops unless GAPMARGIN_GAPFILES_RECORD.json has pass_k0 true and the hashes of the base file and of both scaled files.
# Step 2: the eight rollouts, one after the other, each started only with fewer than 3 GPU compute processes and more than 28 GB free (memory.max minus anon); a run with its T1_DONE is not run again.
# Time rule: before each start the seconds since the first line of $GE/GAPMARGIN10K_START.txt must be at most GM_STOP_S (default 10800), else $GE/GAPMARGIN10K_STOP.txt. A failed rollout writes the stop marker.
# Done marker $GE/GAPMARGIN10K_DONE.txt. Log $GE/gapmargin_queue.log. DRY=1 prints every command and creates nothing. RESUME=1 continues after a stop marker. Nothing is removed or moved.
# Banked runs needed (both seeds, T1_DONE): I10RLcKF5Hb20 I10RQ50KF5Hb20 I10RQ50KF5Hb20gGEc I10RLcHb20 I10RQ50Hb20 I10RQ50Hb20gGEc I10R0Hb20. Written by make_gapmargin_queue.py from the queue of card ADD115 (sha256 prefix 033b776d6338ee3e). No training.
set -u
W=${I10_W:-/workspace/v8proof/integ10k}
GE=${GM_ROOT:-/workspace/v8proof/gapmargin}
GP=$GE
B=$W/build
PY=${I10_PY:-/workspace/venv/bin/python}
TREE=$B/tree_ov
V=/workspace/v8proof
GF=$GE/gapfiles
GMREC=$GF/GAPMARGIN_GAPFILES_RECORD.json
KITG=/workspace/v8proof/gaperr/kit_gap
KQ=${GM_KIT:-$GE/kit_queue}
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
SM=$GP/small_gm
DRY=${DRY:-0}
TMO=${I10_TIMEOUT:-5400}
MINGB=${I10_MIN_FREE_GB:-28}
MAXGPU=3
STOPS=${GM_STOP_S:-10800}
RESUME=${RESUME:-0}
SEEDSW=ALL
ARMS=(0 1); PS=""
LOG=$GP/gapmargin_queue.log
MM=${I10_MEMMAX_FILE:-/sys/fs/cgroup/memory.max}; MS=${I10_MEMSTAT_FILE:-/sys/fs/cgroup/memory.stat}

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { if [ "$DRY" = 1 ]; then echo "$*"; else echo "[$(ts)] $*" >> "$LOG"; fi; }
stop() { if [ "$DRY" = 1 ]; then echo "STOP $*"; exit 3; fi; echo "$(ts) $*" >> "$GP/GAPMARGIN10K_STOP.txt"; log "STOP $*"; exit 3; }

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

# step 0 gate: the test of this kit passed in the lead's run (control d4); nothing is created before this point
gate_ok() { [ -f "$GE/lead_tests/lead_pytest_queue.out" ]; }
if [ "$DRY" = 1 ]; then
  echo "GATE: needs $GE/lead_tests/lead_pytest_queue.out (exit 4 without it)"
else
  gate_ok || { echo "REFUSED: step 0 gate: $GE/lead_tests/lead_pytest_queue.out is missing"; exit 4; }
fi
check_time() { # step name; seconds since the first line of the start file at most STOPS
  if [ "$DRY" = 1 ]; then echo "STOPRULE before $1: seconds since the first line of $GP/GAPMARGIN10K_START.txt at most $STOPS"; return 0; fi
  local t0 now
  t0=$(head -n 1 "$GP/GAPMARGIN10K_START.txt" 2>/dev/null | awk '{print $1}')
  [[ "$t0" =~ ^[0-9]+$ ]] || stop "START-LINE-UNREADABLE before $1"
  now=$(date +%s)
  [ $(( now - t0 )) -le "$STOPS" ] || stop "STOP-RULE $(( now - t0 )) s since the first start line, limit $STOPS s, before $1"
}
if [ "$DRY" != 1 ]; then
  [ -d "$W" ] || { echo "REFUSED: no $W"; exit 2; }
  mkdir -p "$GP" || { echo "REFUSED: cannot make $GP"; exit 2; }
  if [ "$RESUME" = 1 ] && [ -f "$GP/GAPMARGIN10K_START.txt" ]; then
    echo "$(date +%s) resume $(ts) pid $$" >> "$GP/GAPMARGIN10K_START.txt"
  else
    ( set -C; echo "$(date +%s) $(ts) pid $$" > "$GP/GAPMARGIN10K_START.txt" ) 2>/dev/null || { echo "REFUSED: $GP/GAPMARGIN10K_START.txt exists (a queue was started; RESUME=1 continues)"; exit 2; }
  fi
  mkdir -p "$GP/markers" "$W/t1" "$SM"
fi
log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD117 (GAPMARGIN10K): the candidate form with the camera gap scaled by 0.90 and by 0.80, at readout K with no pre-roll and at the convention, model seeds ${ARMS[*]}"
if [ "$DRY" != 1 ]; then
  for k in "${ARMS[@]}"; do
    [ -f "$W/t1/I10R0Hb20_s$k/T1_DONE" ] || stop "NO-BANKED-I10R0Hb20 s$k"
  done
fi
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

step_inputs() { # step 1: the base gap file as it is, and the two scaled gap files (made on the CPU when $GF is absent)
  if [ "$DRY" = 1 ]; then
    echo "CHECK: sha256 of $BASEF begins 55de2f7c"
    echo "CMD: [only when $GF is absent] CUDA_VISIBLE_DEVICES=-1 $PY $KQ/make_margin_gapfiles.py --base $BASEF --out-dir $GF --scales 0.90,0.80"
    echo "CHECK: $GMREC has pass_k0 true and the sha256 of the base file and of both scaled files"
    return 0
  fi
  [ -f "$BASEF" ] || stop "NO-BASE-GAP-FILE $BASEF"
  local got; got=$(sha256sum "$BASEF" | cut -c1-8); [ "$got" = "55de2f7c" ] || stop "BASE-GAP-FILE-SHA $got"
  if [ ! -d "$GF" ]; then
    check_time "scaled gap files"
    [ -f "$KQ/make_margin_gapfiles.py" ] || stop "NO-GAP-TOOL $KQ/make_margin_gapfiles.py"
    CUDA_VISIBLE_DEVICES=-1 "$PY" "$KQ/make_margin_gapfiles.py" --base "$BASEF" --out-dir "$GF" --scales 0.90,0.80 >> "$LOG" 2>&1 || log "gap file build rc $?"
  fi
  [ -f "$GMREC" ] || stop "NO-GAP-RECORD"
  local adm
  adm=$("$PY" - "$GMREC" "$BASEF" "$GF" <<'PYEOF'
import hashlib, json, os, sys
rec_path, basef, gf = sys.argv[1:4]
rec = json.load(open(rec_path, encoding="utf-8"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if rec.get("pass_k0") is not True:
    print("K0-NOT-PASS"); sys.exit(1)
if (rec.get("base") or {}).get("sha256") != sha(basef):
    print("BASE-IS-NOT-THE-RECORDED-ONE"); sys.exit(1)
for nm in ("VGAPLE_A90", "VGAPLE_A80"):
    p = os.path.join(gf, nm + ".npz")
    if not os.path.isfile(p) or ((rec.get("files") or {}).get(nm) or {}).get("sha256") != sha(p):
        print("SCALED-FILE-IS-NOT-THE-RECORDED-ONE " + nm); sys.exit(1)
print("ADMIT-OK gap files")
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
  for nm in I10RLcKF5Hb20 I10RQ50KF5Hb20 I10RQ50KF5Hb20gGEc I10RLcHb20 I10RQ50Hb20 I10RQ50Hb20gGEc I10R0Hb20; do
    for k in "${ARMS[@]}"; do [ -f "$W/t1/${nm}_s$k/T1_DONE" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done
  done
fi
step_inputs
for k in "${ARMS[@]}"; do
  READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" VGAP="$GF/VGAPLE_A90.npz" VCLOSE=1 rollout I10RQ50KF5Hb20gA90 RR "$k" "$k"
done
for k in "${ARMS[@]}"; do
  READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" VGAP="$GF/VGAPLE_A80.npz" VCLOSE=1 rollout I10RQ50KF5Hb20gA80 RR "$k" "$k"
done
for k in "${ARMS[@]}"; do
  READOUT=pinned KAPPA=0.2 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" VGAP="$GF/VGAPLE_A90.npz" VCLOSE=1 rollout I10RQ50Hb20gA90 RR "$k" "$k"
done
for k in "${ARMS[@]}"; do
  READOUT=pinned KAPPA=0.2 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" VGAP="$GF/VGAPLE_A80.npz" VCLOSE=1 rollout I10RQ50Hb20gA80 RR "$k" "$k"
done
if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi
cp -f "$LOG" "$SM/" 2>/dev/null
NAMES="I10RQ50KF5Hb20gA90 I10RQ50KF5Hb20gA80 I10RQ50Hb20gA90 I10RQ50Hb20gA80"
for nm in $NAMES; do for k in "${ARMS[@]}"; do cp -f "$W/t1/${nm}_s$k/ticks/run_${nm}_s$k.txt" "$W/t1/${nm}_s$k/ticks/preroll_log_${nm}_s$k.json" "$SM/" 2>/dev/null; done; done
echo "ZZ GAPMARGIN10K DONE $(ts) all steps done" > "$GP/GAPMARGIN10K_DONE.txt"
log "GAPMARGIN10K_DONE"
