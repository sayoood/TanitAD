#!/usr/bin/env bash
# pod_errcut10k_queue.sh -- card ADD96 (ERRCUT10K): 8 rollouts of the label-fed form with closing speed (form RLc, the arguments of the banked arm I10RLcKF5P30Hb20: pre-roll 30 steps = 6 s, readout K,
# --rule-every 5, --kappa-max 0.2, heading bound 0.2 1/m, model seeds 0 and 1) whose lead rule reads its lead from a MIXED JOIN given with --rule-join; the scoring keeps the evaluation join.
# Arms K100, K50, K25, K12 (joins $GP/joins/<ARM>_s<m>.npz from make_mix_joins.py, one per arm and model seed, model seed m uses reader seed m); run names I10RLcKF5P30Hb20m<ARM>_s<m>.
# Wrapper v8f_t1_eval_reader_rule_rzgfhqetj.py (hash cbf7063d867c9f4e), admitted only with the pod check of control j2 of card ADD95 (CHECK_WRAPPER_J_DONE.txt last line PASS, its .hashes holding the wrapper hash)
# and with MIX_JOINS_RECORD.json saying e0 PASS and e1 PASS with the hash of every join file and of the evaluation join. Banked runs needed: I10RLcKF5P30Hb20 and I10RQ50KF5P30Hb20 (RR), both seeds.
# GR_SEED=0|1|ALL (default ALL): one model seed per process with the markers START_ERRCUT10K_s<k>.txt, ERRCUT10K_DONE_s<k>.txt, ERRCUT10K_STOP_s<k>.txt in $GP and the log queue_errcut_s<k>.log.
# DRY=1 prints every command and runs nothing. A rollout starts only with more than 28 GB free (memory.max minus anon) and fewer than 3 GPU compute processes.
# Written by make_errcut_queue.py from the queue of card ADD95 (gaprule/pod_gaprule10k_queue.sh). No training. No process listing or stopping, no file removal or move.
set -u
W=${I10_W:-/workspace/v8proof/integ10k}
GP=${GR_ROOT:-/workspace/v8proof/errcut10k}
B=$W/build
PY=${I10_PY:-/workspace/venv/bin/python}
TREE=$B/tree_ov
V=/workspace/v8proof
PJ=${GR_JOINS:-$GP/joins}
CHK=${GR_CHECK:-/workspace/v8proof/gaprule10kb/check_j2}
TRACKS=$V/data/inputs/native_tracks.pt
GRID=$V/hierrep/pins/eval_grid_windows.json
JOINE=$V/hierrep/pins/join_eval141.npz
EVALC=$W/eval_cache
PINNED=61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397
WRAP=$B/v8f_t1_eval_reader_rule_rzgfhqetj.py
LG=$W/longroll
G30=$LG/grid_p30.json
G60=$LG/grid_p60.json
SM=$GP/small_ec
DRY=${DRY:-0}
TMO=${I10_TIMEOUT:-5400}
MINGB=${I10_MIN_FREE_GB:-28}
MAXGPU=3
SEEDSW=${GR_SEED:-ALL}
case "$SEEDSW" in 0|1|ALL) ;; *) echo "REFUSED: GR_SEED must be 0, 1 or ALL (got $SEEDSW)"; exit 2;; esac
ARMS=(0 1); PS=""
[ "$SEEDSW" = ALL ] || { ARMS=("$SEEDSW"); PS="_s$SEEDSW"; }
LOG=$GP/queue_errcut$PS.log
MM=${I10_MEMMAX_FILE:-/sys/fs/cgroup/memory.max}; MS=${I10_MEMSTAT_FILE:-/sys/fs/cgroup/memory.stat}

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { if [ "$DRY" = 1 ]; then echo "$*"; else echo "[$(ts)] $*" >> "$LOG"; fi; }
stop() { if [ "$DRY" = 1 ]; then echo "STOP $*"; exit 3; fi; echo "$(ts) $*" > "$GP/ERRCUT10K_STOP$PS.txt"; log "STOP $*"; exit 3; }

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

if [ "$DRY" != 1 ]; then
  [ -d "$W" ] || { echo "REFUSED: no $W"; exit 2; }
  mkdir -p "$GP" || { echo "REFUSED: cannot make $GP"; exit 2; }
  ( set -C; echo "$(ts) pid $$" > "$GP/START_ERRCUT10K$PS.txt" ) 2>/dev/null || { echo "REFUSED: $GP/START_ERRCUT10K$PS.txt exists (a queue was started)"; exit 2; }
  mkdir -p "$GP/markers" "$W/t1" "$SM"
fi
log "queue start (DRY=$DRY, seeds $SEEDSW): card ADD96 (ERRCUT10K): label-fed rule with closing speed, lead read from a mixed join (K100, K50, K25, K12), scoring on the evaluation join, at age"
if [ "$DRY" != 1 ]; then
  for k in "${ARMS[@]}"; do
    [ -f "$W/t1/I10R0Hb20_s$k/T1_DONE" ] || stop "NO-BANKED-I10R0Hb20 s$k"
  done
fi
for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzgfhqetj.py v8f_t1_eval_reader_rule_rzgfhqet.py v8f_t1_eval_reader_rule_rzgfhqe.py closest.py v8f_t1_eval_reader_rule_rzgfhqc.py v8f_t1_eval_reader_rule_rzgfhq.py v8f_t1_eval_reader_rule_rzgfh.py v8f_t1_eval_reader_rule_rz.py check_preroll_b.py make_longroll_grid.py dist_rule.py v8f_t1_eval_rule.py v8f_t1_eval_rule_mix.py v8f_t1_eval_reader_rule.py tok_reader.py tok_probe_lr.py qmap_fit.py qcal_map.py reader_controls.py; do
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
  [ "$DRY" = 1 ] || { [ -e "$out" ] && stop "OUTPUT-EXISTS $out"; }
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
  if [ -n "${RJOIN:-}" ]; then
    [ "$DRY" = 1 ] || [ -f "$RJOIN" ] || stop "NO-RULE-JOIN $RJOIN $arm"
    extra+=(--rule-join "$RJOIN")
  fi
  if [ "$DRY" = 1 ]; then
    echo "CMD: DISTRULE_DIR=$B PYTHONPATH=$TREE/stack timeout $TMO $PY $WRAP --tree $TREE --form $form ${extra[*]} run --ckpt $W/ckpt/UE_s$m/A/ckpt_last.pt --tag $tag --tracks $TRACKS --eval-cache $EVALC --eval-grid $grid --eval-join $JOINE --out-dir $out --replan-steps ${REPLAN:-10} --nav $nav --scene logged --schedule periodic --limit causal --dump-ticks $out/ticks --dump-heads --no-analyze --device cuda"
    return 0
  fi
  echo "arm $arm form $form replan steps ${REPLAN:-10} rule every ${EVERY:-10} readout ${READOUT:-pinned} kappa ${KAPPA:-none} preroll $pre exec ${PREEXEC:-plan} nav $nav grid $(basename "$grid") model seed $m reader seed $r ckpt sha256 $(sha256sum "$W/ckpt/UE_s$m/A/ckpt_last.pt" | cut -c1-64) wrapper sha256 $(sha256sum "$WRAP" | cut -c1-64) started $(ts)${RJOIN:+ rule-join $RJOIN rule join sha256 $(sha256sum "$RJOIN" | cut -c1-64)}" > "$out/ticks/run_$arm.txt"
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

admit() { # RECORD JOINDIR EVALJOIN NAMES -> prints why on failure
  "$PY" - "$@" <<'PYEOF'
import hashlib, json, os, sys
rec_path, jdir, ejoin, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
rec = json.load(open(rec_path, encoding="utf-8"))
for c in ("e0", "e1"):
    if not isinstance(rec.get(c), dict) or rec[c].get("PASS") is not True:
        print(c.upper() + "-NOT-PASS"); sys.exit(1)
h = hashlib.sha256(open(ejoin, "rb").read()).hexdigest()
if rec.get("eval_join_sha256") != h:
    print("RECORD-IS-FOR-ANOTHER-EVAL-JOIN"); sys.exit(1)
for a in names:
    fn = os.path.join(jdir, a + ".npz")
    if not os.path.isfile(fn):
        print("NO-JOIN " + a); sys.exit(1)
    if hashlib.sha256(open(fn, "rb").read()).hexdigest() != (rec.get("arms", {}).get(a) or {}).get("sha256"):
        print("JOIN-HASH-DIFFERS " + a); sys.exit(1)
print("ADMIT-OK joins %d" % len(names))
PYEOF
}
[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"
if [ "$DRY" != 1 ]; then
  CK="$CHK/CHECK_WRAPPER_J_DONE.txt"
  [ "$(tail -n 1 "$CK" 2>/dev/null)" = "ZZ CHECK-WRAPPER-J PASS" ] || stop "NO-PASS-OF-THE-WRAPPER-CHECK"
  got=$(sha256sum "$WRAP" | cut -c1-16); [ "$got" = "cbf7063d867c9f4e" ] || stop "WRAPPER-SHA $got"
  grep -q "cbf7063d867c9f4e" "$CHK/CHECK_WRAPPER_J.hashes" 2>/dev/null || stop "CHECK-RECORD-HAS-NOT-THIS-WRAPPER"
  [ -f "$PJ/MIX_JOINS_RECORD.json" ] || stop "NO-MIX-JOINS-RECORD"
  adm=$(admit "$PJ/MIX_JOINS_RECORD.json" "$PJ" "$JOINE" "K100_s0,K100_s1,K50_s0,K50_s1,K25_s0,K25_s1,K12_s0,K12_s1") || stop "JOIN-ADMISSION $adm"
  log "$adm"
  for nm in I10RLcKF5P30Hb20 I10RQ50KF5P30Hb20; do
    for k in "${ARMS[@]}"; do [ -d "$W/t1/${nm}_s$k/dump_${nm}_s${k}_m10_causal_heads" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done
  done
fi
for k in "${ARMS[@]}"; do
  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 RJOIN="$PJ/K100_s$k.npz" rollout I10RLcKF5P30Hb20mK100 RLc "$k" "$k"
done
for k in "${ARMS[@]}"; do
  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 RJOIN="$PJ/K50_s$k.npz" rollout I10RLcKF5P30Hb20mK50 RLc "$k" "$k"
done
for k in "${ARMS[@]}"; do
  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 RJOIN="$PJ/K25_s$k.npz" rollout I10RLcKF5P30Hb20mK25 RLc "$k" "$k"
done
for k in "${ARMS[@]}"; do
  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 RJOIN="$PJ/K12_s$k.npz" rollout I10RLcKF5P30Hb20mK12 RLc "$k" "$k"
done
if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi
cp -f "$LOG" "$SM/" 2>/dev/null
NAMES="I10RLcKF5P30Hb20mK100 I10RLcKF5P30Hb20mK50 I10RLcKF5P30Hb20mK25 I10RLcKF5P30Hb20mK12"
for nm in $NAMES; do for k in "${ARMS[@]}"; do cp -f "$W/t1/${nm}_s$k/ticks/run_${nm}_s$k.txt" "$W/t1/${nm}_s$k/ticks/preroll_log_${nm}_s$k.json" "$SM/" 2>/dev/null; done; done
echo "$(ts) all steps done (seeds $SEEDSW)" > "$GP/ERRCUT10K_DONE$PS.txt"
log "ERRCUT10K_DONE seeds $SEEDSW"
