#!/usr/bin/env bash
# pod_gaperr10k_queue.sh -- card ADD111 (GAPERR10K): 8 rollouts of the reader-fed form RR with the wrapper variant rzgfhqetjv of card ADD102 (hash 28bc31f21ecb4ccb, unchanged), the arguments of the VD5c call of the ADD102 queue
# (pre-roll 30 steps = 6 s, readout K, --rule-every 5, --kappa-max 0.2, heading bound 0.2 1/m, --vis-closing), model seeds 0 and 1, with one gap file per arm: NR, BR, SM, SL (label gap plus the slow part /
# label gap plus the fast part / median of the base gap over the trailing run / end value of the line over the trailing run). Run names I10RQ50KF5P30Hb20b<ARM>_s<k>. NR and BR carry label values and are attribution only.
# Step 0 gate: exit 4 unless $GE/kit_gap/lead_pytest.out and $GE/kit_read/lead_pytest.out exist (control b4: the tests of every kit passed before the first rollout).
# Step 1: when $GE/gapfiles is absent, make_gaperr_gapfiles.py runs on the CPU; the queue stops unless GAPERR_GAPFILES_RECORD.json has pass_b0 true and the hashes of the four files, of the LA file and of the evaluation join.
# Step 2: the eight rollouts, one after the other, each started only with fewer than 3 GPU compute processes and more than 28 GB free (memory.max minus anon); a run with its T1_DONE is not run again.
# Time rule: before each start the seconds since the first line of $GE/GAPERR10K_START.txt must be at most GE_STOP_S (default 10800), else $GE/GAPERR10K_STOP.txt. A failed rollout writes the stop marker.
# Done marker $GE/GAPERR10K_DONE.txt. Log $GE/gaperr_queue.log. DRY=1 prints every command and creates nothing. RESUME=1 continues after a stop marker. Nothing is removed or moved.
# Banked runs needed: RR (I10RQ50KF5P30Hb20), RLc (I10RLcKF5P30Hb20), V0 of card ADD102, GEc (I10RQ50KF5P30Hb20gGEc) of card ADD106 and LA (I10RQ50KF5P30Hb20aLA) of card ADD110, both seeds.
# Written by make_gaperr_queue.py from the queue of card ADD110 (sha256 prefix dc6f6c6405092fe3). No training.
set -u
W=${I10_W:-/workspace/v8proof/integ10k}
GE=${GE_ROOT:-/workspace/v8proof/gaperr}
GP=$GE
B=$W/build
PY=${I10_PY:-/workspace/venv/bin/python}
TREE=$B/tree_ov
V=/workspace/v8proof
GF=$GE/gapfiles
KG=$GE/kit_gap
KR=$GE/kit_read
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
SM=$GP/small_ge
DRY=${DRY:-0}
TMO=${I10_TIMEOUT:-5400}
MINGB=${I10_MIN_FREE_GB:-28}
MAXGPU=3
STOPS=${GE_STOP_S:-10800}
RESUME=${RESUME:-0}
SEEDSW=ALL
ARMS=(0 1); PS=""
LOG=$GP/gaperr_queue.log
MM=${I10_MEMMAX_FILE:-/sys/fs/cgroup/memory.max}; MS=${I10_MEMSTAT_FILE:-/sys/fs/cgroup/memory.stat}

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { if [ "$DRY" = 1 ]; then echo "$*"; else echo "[$(ts)] $*" >> "$LOG"; fi; }
stop() { if [ "$DRY" = 1 ]; then echo "STOP $*"; exit 3; fi; echo "$(ts) $*" >> "$GP/GAPERR10K_STOP.txt"; log "STOP $*"; exit 3; }

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

# step 0 gate: the tests of both kits passed in the lead's run (control a4); nothing is created before this point
gate_ok() { [ -f "$KG/lead_pytest.out" ] && [ -f "$KR/lead_pytest.out" ]; }
if [ "$DRY" = 1 ]; then
  echo "GATE: needs $KG/lead_pytest.out and $KR/lead_pytest.out (exit 4 without them)"
else
  gate_ok || { echo "REFUSED: step 0 gate: $KG/lead_pytest.out or $KR/lead_pytest.out is missing"; exit 4; }
fi
check_time() { # step name; seconds since the first line of the start file at most STOPS
  if [ "$DRY" = 1 ]; then echo "STOPRULE before $1: seconds since the first line of $GP/GAPERR10K_START.txt at most $STOPS"; return 0; fi
  local t0 now
  t0=$(head -n 1 "$GP/GAPERR10K_START.txt" 2>/dev/null | awk '{print $1}')
  [[ "$t0" =~ ^[0-9]+$ ]] || stop "START-LINE-UNREADABLE before $1"
  now=$(date +%s)
  [ $(( now - t0 )) -le "$STOPS" ] || stop "STOP-RULE $(( now - t0 )) s since the first start line, limit $STOPS s, before $1"
}
if [ "$DRY" != 1 ]; then
  [ -d "$W" ] || { echo "REFUSED: no $W"; exit 2; }
  mkdir -p "$GP" || { echo "REFUSED: cannot make $GP"; exit 2; }
  if [ "$RESUME" = 1 ] && [ -f "$GP/GAPERR10K_START.txt" ]; then
    echo "$(date +%s) resume $(ts) pid $$" >> "$GP/GAPERR10K_START.txt"
  else
    ( set -C; echo "$(date +%s) $(ts) pid $$" > "$GP/GAPERR10K_START.txt" ) 2>/dev/null || { echo "REFUSED: $GP/GAPERR10K_START.txt exists (a queue was started; RESUME=1 continues)"; exit 2; }
  fi
  mkdir -p "$GP/markers" "$W/t1" "$SM"
fi
log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD111 (GAPERR10K): reader-fed rule RR with the closing input of the run rule and one gap file per arm (NR, BR, SM, SL: label gap plus the slow part, label gap plus the fast part, two smoothing forms of the base gap), at age"
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

admit() { # RECORD GAPDIR EVALJOIN BASEFILE LAFILE -> prints why on failure
  "$PY" - "$@" <<'PYEOF'
import hashlib, json, os, sys
rec_path, gdir, ejoin, basef, laf = sys.argv[1:6]
rec = json.load(open(rec_path, encoding="utf-8"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if rec.get("pass_b0") is not True:
    print("B0-NOT-PASS"); sys.exit(1)
if rec.get("eval_join_sha256") != sha(ejoin):
    print("RECORD-IS-FOR-ANOTHER-EVAL-JOIN"); sys.exit(1)
b = rec.get("base") or {}
if b.get("sha256") != sha(basef) or b.get("named_in_record_add106") is not True:
    print("BASE-IS-NOT-THE-RECORDED-ONE"); sys.exit(1)
if not os.path.isfile(laf) or (rec.get("la_file") or {}).get("sha256") != sha(laf):
    print("LA-FILE-IS-NOT-THE-RECORDED-ONE"); sys.exit(1)
for a in ("NR", "BR", "SM", "SL"):
    fn = os.path.join(gdir, "VGAPA_%s.npz" % a)
    if not os.path.isfile(fn):
        print("NO-GAPFILE " + a); sys.exit(1)
    if sha(fn) != (rec.get("files", {}).get(a) or {}).get("sha256"):
        print("GAPFILE-HASH-DIFFERS " + a); sys.exit(1)
print("ADMIT-OK files 4")
PYEOF
}
step_gapfiles() { # step 1: make the gap files when $GF is absent (CPU), then admit them
  if [ "$DRY" = 1 ]; then
    echo "CMD: [only when $GF is absent] CUDA_VISIBLE_DEVICES=-1 $PY $KG/make_gaperr_gapfiles.py --join $JOINE --base $BASEF --base-record $BASER --la-file $LAF --out-dir $GF"
    echo "CHECK: $GF/GAPERR_GAPFILES_RECORD.json has pass_b0 true, the sha256 of the evaluation join, of the base file, of the LA file and of each of the four gap files"
    return 0
  fi
  if [ ! -d "$GF" ]; then
    check_time "gap files"
    [ -f "$KG/make_gaperr_gapfiles.py" ] || stop "NO-GAPFILE-TOOL $KG/make_gaperr_gapfiles.py"
    [ -f "$BASEF" ] && [ -f "$BASER" ] && [ -f "$LAF" ] || stop "NO-BASE-FILE-RECORD-OR-LA-FILE"
    CUDA_VISIBLE_DEVICES=-1 "$PY" "$KG/make_gaperr_gapfiles.py" --join "$JOINE" --base "$BASEF" --base-record "$BASER" --la-file "$LAF" --out-dir "$GF" >> "$LOG" 2>&1 || log "gap-file build rc $?"
  fi
  [ -f "$GF/GAPERR_GAPFILES_RECORD.json" ] || stop "NO-GAPFILES-RECORD"
  local adm
  adm=$(admit "$GF/GAPERR_GAPFILES_RECORD.json" "$GF" "$JOINE" "$BASEF" "$LAF") || stop "GAPFILE-ADMISSION $adm"
  log "$adm"
}
[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"
if [ "$DRY" != 1 ]; then
  CK="$CHK/CHECK_WRAPPER_V_DONE.txt"
  [ "$(tail -n 1 "$CK" 2>/dev/null)" = "ZZ CHECK-WRAPPER-V PASS" ] || stop "NO-PASS-OF-THE-WRAPPER-CHECK"
  got=$(sha256sum "$WRAP" | cut -c1-16); [ "$got" = "28bc31f21ecb4ccb" ] || stop "WRAPPER-SHA $got"
  grep -q "28bc31f21ecb4ccb" "$CHK/CHECK_WRAPPER_V.hashes" 2>/dev/null || stop "CHECK-RECORD-HAS-NOT-THIS-WRAPPER"
  for nm in I10RLcKF5P30Hb20 I10RQ50KF5P30Hb20; do
    for k in "${ARMS[@]}"; do [ -d "$W/t1/${nm}_s$k/dump_${nm}_s${k}_m10_causal_heads" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done
  done
  for nm in I10RQ50KF5P30Hb20vV0 I10RQ50KF5P30Hb20gGEc I10RQ50KF5P30Hb20aLA; do
    for k in "${ARMS[@]}"; do [ -f "$W/t1/${nm}_s$k/T1_DONE" ] || stop "NO-BANKED-RUN-OF-ADD102-ADD106-OR-ADD110 ${nm}_s$k"; done
  done
fi
step_gapfiles
for k in "${ARMS[@]}"; do
  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" VGAP="$GF/VGAPA_NR.npz" VCLOSE=1 rollout I10RQ50KF5P30Hb20bNR RR "$k" "$k"
done
for k in "${ARMS[@]}"; do
  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" VGAP="$GF/VGAPA_BR.npz" VCLOSE=1 rollout I10RQ50KF5P30Hb20bBR RR "$k" "$k"
done
for k in "${ARMS[@]}"; do
  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" VGAP="$GF/VGAPA_SM.npz" VCLOSE=1 rollout I10RQ50KF5P30Hb20bSM RR "$k" "$k"
done
for k in "${ARMS[@]}"; do
  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" VGAP="$GF/VGAPA_SL.npz" VCLOSE=1 rollout I10RQ50KF5P30Hb20bSL RR "$k" "$k"
done
if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi
cp -f "$LOG" "$SM/" 2>/dev/null
NAMES="I10RQ50KF5P30Hb20bNR I10RQ50KF5P30Hb20bBR I10RQ50KF5P30Hb20bSM I10RQ50KF5P30Hb20bSL"
for nm in $NAMES; do for k in "${ARMS[@]}"; do cp -f "$W/t1/${nm}_s$k/ticks/run_${nm}_s$k.txt" "$W/t1/${nm}_s$k/ticks/preroll_log_${nm}_s$k.json" "$SM/" 2>/dev/null; done; done
echo "ZZ GAPERR10K DONE $(ts) all steps done" > "$GP/GAPERR10K_DONE.txt"
log "GAPERR10K_DONE"
