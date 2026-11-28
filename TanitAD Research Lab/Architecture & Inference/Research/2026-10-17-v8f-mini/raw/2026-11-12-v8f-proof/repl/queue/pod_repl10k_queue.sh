#!/usr/bin/env bash
# pod_repl10k_queue.sh -- card ADD121 (REPL10K): the lead-rule candidate set on the model seeds 2 and 3 of the 10k line, through the wrapper variant rzgfhqetjv (hash 28bc31f21ecb4ccb,
# unchanged). Per new model seed m, with the reader seed and map of seed m - 2: at the convention the spread run R0, the label form RLc, the reader-fed form RR, the candidate GEc and the
# candidate with the gap file scaled by 0.90 (A90); at readout K with no pre-roll and with a pre-roll of 30 steps (grid_p30) RLc, RR, GEc and A90. 26 rollouts, names as banked.
# RP_STAGE=identity (run before the main stage, needs model seed 0 only): three banked reference runs are made again through this wrapper (I10R0Hb20IDR, I10RLcKF5Hb20IDR,
# I10RQ50Hb20IDR, model seed 0) and each is compared with its banked run (dump files with cmp_dumps.py, rule log arrays); a difference stops the queue. Markers REPL10K_ID_*.
# Step 0 gate: exit 4 unless $GE/lead_tests/lead_pytest_queue.out exists (the test of this kit passed in the lead's run before the queue start).
# Main stage: needs $GE/REPL10K_ID_DONE.txt and the lead's record of the two new models ($GE/models/REPL_MODELS_RECORD.json: for UE_s2 and UE_s3 the checkpoint hash, gate_pass true,
# recipe_same_but_seed true); the checkpoints on disk must have the recorded hashes. Gap files: the base file (sha256 prefix 55de2f7c) and the record of card ADD117.
# Each rollout starts only with fewer than 3 GPU compute processes and more than 28 GB free (memory.max minus anon); a run with its T1_DONE is not run again.
# Time rule: before each start the seconds since the first line of the stage's start file must be at most RP_STOP_S (default 25200), else the stage's stop file. A failed rollout writes it.
# Done marker $GE/REPL10K_DONE.txt (identity stage: REPL10K_ID_DONE.txt). Log $GE/repl_queue.log. DRY=1 prints every command and creates nothing. RESUME=1 continues after a stop marker.
# Nothing is removed or moved. Written by make_repl_queue.py from the queue of card ADD117 (sha256 prefix e5811bf79826a718). No training.
set -u
W=${I10_W:-/workspace/v8proof/integ10k}
GE=${RP_ROOT:-/workspace/v8proof/repl}
GP=$GE
B=$W/build
PY=${I10_PY:-/workspace/venv/bin/python}
TREE=$B/tree_ov
V=/workspace/v8proof
GF=${RP_GAPFILES:-/workspace/v8proof/gapmargin/gapfiles}
MREC=${RP_MODELS_RECORD:-$GE/models/REPL_MODELS_RECORD.json}
STAGE=${RP_STAGE:-main}
GMREC=$GF/GAPMARGIN_GAPFILES_RECORD.json
KITG=/workspace/v8proof/gaperr/kit_gap
KQ=${RP_KIT:-$GE/kit_queue}
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
SM=$GP/small_rp
DRY=${DRY:-0}
TMO=${I10_TIMEOUT:-5400}
MINGB=${I10_MIN_FREE_GB:-28}
MAXGPU=3
STOPS=${RP_STOP_S:-25200}
RESUME=${RESUME:-0}
SEEDSW=ALL
ARMS=(2 3); FIRST=(0 1); PS=""
MK=REPL10K; CKS="${ARMS[*]}"
case "$STAGE" in main) ;; identity) MK=REPL10K_ID; CKS=0 ;; *) echo "REFUSED: RP_STAGE must be main or identity"; exit 2 ;; esac
LOG=$GP/repl_queue.log
MM=${I10_MEMMAX_FILE:-/sys/fs/cgroup/memory.max}; MS=${I10_MEMSTAT_FILE:-/sys/fs/cgroup/memory.stat}

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { if [ "$DRY" = 1 ]; then echo "$*"; else echo "[$(ts)] $*" >> "$LOG"; fi; }
stop() { if [ "$DRY" = 1 ]; then echo "STOP $*"; exit 3; fi; echo "$(ts) $*" >> "$GP/${MK}_STOP.txt"; log "STOP $*"; exit 3; }

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
  if [ "$DRY" = 1 ]; then echo "STOPRULE before $1: seconds since the first line of $GP/${MK}_START.txt at most $STOPS"; return 0; fi
  local t0 now
  t0=$(head -n 1 "$GP/${MK}_START.txt" 2>/dev/null | awk '{print $1}')
  [[ "$t0" =~ ^[0-9]+$ ]] || stop "START-LINE-UNREADABLE before $1"
  now=$(date +%s)
  [ $(( now - t0 )) -le "$STOPS" ] || stop "STOP-RULE $(( now - t0 )) s since the first start line, limit $STOPS s, before $1"
}
if [ "$DRY" != 1 ]; then
  [ -d "$W" ] || { echo "REFUSED: no $W"; exit 2; }
  mkdir -p "$GP" || { echo "REFUSED: cannot make $GP"; exit 2; }
  if [ "$RESUME" = 1 ] && [ -f "$GP/${MK}_START.txt" ]; then
    echo "$(date +%s) resume $(ts) pid $$" >> "$GP/${MK}_START.txt"
  else
    ( set -C; echo "$(date +%s) $(ts) pid $$" > "$GP/${MK}_START.txt" ) 2>/dev/null || { echo "REFUSED: $GP/${MK}_START.txt exists (a queue was started; RESUME=1 continues)"; exit 2; }
  fi
  mkdir -p "$GP/markers" "$W/t1" "$SM"
fi
log "queue start (stage $STAGE, DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD121 (REPL10K): the lead-rule candidate set on the model seeds ${ARMS[*]} (reader seeds of the first pair), identity stage on model seed 0"
if [ "$DRY" != 1 ]; then
  for k in "${FIRST[@]}"; do
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
  for k in $CKS; do
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
  for nm in I10R0Hb20 I10RLcHb20 I10RQ50Hb20 I10RQ50Hb20gGEc I10RQ50Hb20gA90 I10RLcKF5Hb20 I10RQ50KF5Hb20 I10RQ50KF5Hb20gGEc I10RQ50KF5Hb20gA90 I10RLcKF5P30Hb20 I10RQ50KF5P30Hb20 I10RQ50KF5P30Hb20gGEc I10RQ50KF5P30Hb20gA90; do
    for k in "${FIRST[@]}"; do [ -f "$W/t1/${nm}_s$k/T1_DONE" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done
  done
fi
step_inputs
step_models() { # the lead's record of the two new models against the checkpoints on disk
  if [ "$DRY" = 1 ]; then
    echo "CHECK: $MREC names for UE_s2 and UE_s3 the sha256 of $W/ckpt/UE_s<m>/A/ckpt_last.pt, gate_pass true and recipe_same_but_seed true"
    return 0
  fi
  [ -f "$MREC" ] || stop "NO-MODELS-RECORD"
  local adm
  adm=$("$PY" - "$MREC" "$W/ckpt" "${ARMS[@]}" <<'PYEOF'
import hashlib, json, os, re, sys
rec_path, ck, seeds = sys.argv[1], sys.argv[2], sys.argv[3:]
rec = json.load(open(rec_path, encoding="utf-8"))
if not re.fullmatch(r"[0-9a-f]{64}", str(rec.get("note_sha256") or "")):
    print("NO-NOTE-HASH"); sys.exit(1)
for k in seeds:
    m = (rec.get("models") or {}).get("UE_s" + k) or {}
    p = os.path.join(ck, "UE_s" + k, "A", "ckpt_last.pt")
    if m.get("gate_pass") is not True or m.get("recipe_same_but_seed") is not True:
        print("MODEL-NOT-ADMITTED UE_s" + k); sys.exit(1)
    if not os.path.isfile(p) or hashlib.sha256(open(p, "rb").read()).hexdigest() != m.get("ckpt_sha256"):
        print("CKPT-IS-NOT-THE-RECORDED-ONE UE_s" + k); sys.exit(1)
print("ADMIT-OK models")
PYEOF
) || stop "MODELS-ADMISSION $adm"
  log "$adm"
}
step_identity() { # NEW BANKED (run folder names): dump files and rule log arrays; a difference stops the queue
  local a=$1 b=$2
  if [ "$DRY" = 1 ]; then
    echo "CMD: CUDA_VISIBLE_DEVICES=-1 $PY $B/cmp_dumps.py $W/t1/$a/dump_${a}_m10_causal_heads $W/t1/$b/dump_${b}_m10_causal_heads"
    echo "CHECK: the arrays of the rule logs of $a (where the banked run has them) equal those of $b; a difference stops the queue"
    return 0
  fi
  done_step "idr_$a" && return 0
  [ -f "$W/t1/$a/T1_DONE" ] || stop "NO-IDENTITY-RUN $a"
  mkdir -p "$GP/idr"
  CUDA_VISIBLE_DEVICES=-1 "$PY" "$B/cmp_dumps.py" "$W/t1/$a/dump_${a}_m10_causal_heads" "$W/t1/$b/dump_${b}_m10_causal_heads" > "$GP/idr/CMP_DUMPS_$a.out" 2>&1 || stop "IDENTITY-RUN-DUMPS-DIFFER $a"
  local adm
  adm=$(CUDA_VISIBLE_DEVICES=-1 "$PY" - "$W/t1/$a/ticks" "$a" "$W/t1/$b/ticks" "$b" <<'PYEOF'
import os, sys
import numpy as np
da, a, db, b = sys.argv[1:5]
bad, n, skipped, files, extra = [], 0, 0, 0, 0
for pre in ("rule_log_", "rule_log_mid_"):
    for suf in (".npz", "_rd.npz"):
        pa, pb = os.path.join(da, pre + a + suf), os.path.join(db, pre + b + suf)
        ea, eb = os.path.isfile(pa), os.path.isfile(pb)
        if not ea and not eb:
            continue
        if eb and not ea:
            bad.append("lost-file:" + pre + suf); continue
        if ea and not eb:
            extra += 1; continue                                                    # a file of a later wrapper variant, counted
        files += 1
        za, zb = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
        lost = [k for k in zb.files if k not in za.files and zb[k].ndim >= 1]      # a row array of the banked log that the new log lacks
        if lost:
            bad.append("lost:" + ",".join(lost[:4]) + ":" + pre + suf)
        extra += len(set(za.files) - set(zb.files))                                 # keys of a later wrapper variant, counted
        for k in sorted(set(za.files) & set(zb.files)):
            x, y = za[k], zb[k]
            if x.dtype.kind == "O" or y.dtype.kind == "O" or (x.dtype.kind in "US" and x.ndim == 0):
                skipped += 1; continue
            same = x.shape == y.shape and (np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f" else np.array_equal(x, y))
            n += 1
            if not same:
                bad.append(k + ":" + pre + suf)
if bad:
    print("RULE-LOG-DIFFERS files %d arrays %d: %s" % (files, n, " ".join(bad[:12]))); sys.exit(1)
print("ADMIT-OK identity of the rule logs: files %d, arrays compared %d, not compared (objects, text of one value) %d, keys only in the new run %d" % (files, n, skipped, extra))
PYEOF
) || { echo "$adm" > "$GP/idr/CMP_RULELOG_$a.out"; stop "IDENTITY-RUN-RULE-LOG-DIFFERS $a"; }
  echo "$adm" > "$GP/idr/CMP_RULELOG_$a.out"
  mark "idr_$a"
  log "identity run $a equals the banked run $b: $(tail -n 1 "$GP/idr/CMP_DUMPS_$a.out" | cut -c1-80); $adm"
}
if [ "$STAGE" = identity ]; then
  READOUT=pinned KAPPA=0.2 rollout I10R0Hb20IDR R0 0 0
  step_identity I10R0Hb20IDR_s0 I10R0Hb20_s0
  READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 rollout I10RLcKF5Hb20IDR RLc 0 0
  step_identity I10RLcKF5Hb20IDR_s0 I10RLcKF5Hb20_s0
  READOUT=pinned KAPPA=0.2 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s0_q50.json" rollout I10RQ50Hb20IDR RR 0 0
  step_identity I10RQ50Hb20IDR_s0 I10RQ50Hb20_s0
else
  [ "$DRY" = 1 ] || [ -f "$GP/REPL10K_ID_DONE.txt" ] || stop "NO-IDENTITY-STAGE"
  step_models
  for k in "${ARMS[@]}"; do
    r=$((k - 2))
    READOUT=pinned KAPPA=0.2 rollout I10R0Hb20 R0 "$k" "$r"
    READOUT=pinned KAPPA=0.2 rollout I10RLcHb20 RLc "$k" "$r"
    READOUT=pinned KAPPA=0.2 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${r}_q50.json" rollout I10RQ50Hb20 RR "$k" "$r"
    READOUT=pinned KAPPA=0.2 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${r}_q50.json" VGAP="$BASEF" VCLOSE=1 rollout I10RQ50Hb20gGEc RR "$k" "$r"
    READOUT=pinned KAPPA=0.2 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${r}_q50.json" VGAP="$GF/VGAPLE_A90.npz" VCLOSE=1 rollout I10RQ50Hb20gA90 RR "$k" "$r"
    READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 rollout I10RLcKF5Hb20 RLc "$k" "$r"
    READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${r}_q50.json" rollout I10RQ50KF5Hb20 RR "$k" "$r"
    READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${r}_q50.json" VGAP="$BASEF" VCLOSE=1 rollout I10RQ50KF5Hb20gGEc RR "$k" "$r"
    READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${r}_q50.json" VGAP="$GF/VGAPLE_A90.npz" VCLOSE=1 rollout I10RQ50KF5Hb20gA90 RR "$k" "$r"
    RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 rollout I10RLcKF5P30Hb20 RLc "$k" "$r"
    RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${r}_q50.json" rollout I10RQ50KF5P30Hb20 RR "$k" "$r"
    RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${r}_q50.json" VGAP="$BASEF" VCLOSE=1 rollout I10RQ50KF5P30Hb20gGEc RR "$k" "$r"
    RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${r}_q50.json" VGAP="$GF/VGAPLE_A90.npz" VCLOSE=1 rollout I10RQ50KF5P30Hb20gA90 RR "$k" "$r"
  done
fi
if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi
cp -f "$LOG" "$SM/" 2>/dev/null
NAMES="I10R0Hb20 I10RLcHb20 I10RQ50Hb20 I10RQ50Hb20gGEc I10RQ50Hb20gA90 I10RLcKF5Hb20 I10RQ50KF5Hb20 I10RQ50KF5Hb20gGEc I10RQ50KF5Hb20gA90 I10RLcKF5P30Hb20 I10RQ50KF5P30Hb20 I10RQ50KF5P30Hb20gGEc I10RQ50KF5P30Hb20gA90"
cp -f "$GP/idr/"*.out "$SM/" 2>/dev/null
for nm in $NAMES; do for k in "${ARMS[@]}"; do cp -f "$W/t1/${nm}_s$k/ticks/run_${nm}_s$k.txt" "$W/t1/${nm}_s$k/ticks/preroll_log_${nm}_s$k.json" "$SM/" 2>/dev/null; done; done
echo "ZZ ${MK} DONE $(ts) all steps done" > "$GP/${MK}_DONE.txt"
log "${MK}_DONE"
