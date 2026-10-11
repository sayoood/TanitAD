#!/bin/bash
# P2R: a FRESH A40 pod -> a finished P2 joint fine-tune, banked OFF the pod.  Runbook: eval/P2R_POD_RUNBOOK.md.
# Amends: eval/PREREG_P2_AMENDMENT_1.md.  Replaces refe/p2_chain.sh, which waited on an SFT-4 that no longer runs, on
# PDM labels that died with the old pod, and on a gate runner that was never started.
#
# Needs: HF_TOKEN in the environment (write-scoped, the PI provides it; read here, never echoed, never an argument, and
# stripped from every child except p2r_sync.py) and code/ refe/ splits/ shipped to /workspace/refe-plan.
# Start (detached):  nohup setsid bash /workspace/refe-plan/refe/p2r_pod.sh > /workspace/data/refe_p2/p2r_stdout.log 2>&1 < /dev/null &
# Re-running after ANY interruption is safe: finished stages leave /workspace/data/refe_p2/stage/<name>.ok.
# P2R_STOP_AFTER=<env|weights|restore|recover|dbs|relabel|pix|pdm_push|gate> ends the chain after that stage (a cheap dry run).
#
# Stages: env -> weights -> restore (HF) -> recover (HF, a lost pod's work) -> [dbs -> relabel (PDM, complete)] + pix (background) -> pdm_push (HF) -> gate ->
#         preflight -> train (supervised) -> final push (HF, remote-verified).  A watcher pushes logs/checkpoints all along.
# Every child is launched with fd 200 closed (the chain holds its flock on 200).  Markers are opaque ZZ tokens.
set -u
WORK=/workspace; PKG=$WORK/refe-plan; DATA=$WORK/data
P2=$DATA/refe_p2; S=$P2/stage; L=$P2/chain.log
RUN=$DATA/refe_runs/p2_joint_pdm
INIT=$DATA/refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt
SETS=$DATA/refe_navtrain/onpolicy/sets
PDM=$DATA/refe_sft2/train_pdm; PDMLOG=$DATA/refe_sft2/pdm_train.log
WSTOP=$P2/sync.stop
mkdir -p "$P2" "$S" "$DATA/refe_sft2"
log() { echo "$(date -u +%FT%T) $*" >> $L; }
refuse() { log "REFUSED: $*"; exit 1; }
stage_done() { [ -f "$S/$1.ok" ]; }
stage_mark() { date -u +%FT%T > "$S/$1.ok"; }
stop_after() { if [ "${P2R_STOP_AFTER:-}" = "$1" ]; then log "stopping after $1 (P2R_STOP_AFTER)"; exit 0; fi; }
# The gate JSON is PARSED, never grepped (sft4r_chain.sh Amendment 2): all four gates must be present AND ok, so a
# --skip-trainer run (no G4) cannot pass.   gate_pass <gate.json>   -> rc 0 iff PASS
gate_pass() {
  $PY_C - "$@" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
g = d.get("gates", {})
bad = [p for p in ("G1", "G2", "G3", "G4") if not any(k.startswith(p + "_") and v.get("ok") is True for k, v in g.items())]
if d.get("verdict") != "PASS" or bad:
    sys.exit("gate not PASS: verdict=%s missing_or_not_ok=%s" % (d.get("verdict"), bad))
PYEOF
}
# The number of sets the relabel must cover, recomputed from the restored files with the relabeller's OWN definition.
#   served_count <sets dir>   -> prints the count
served_count() {
  $PY_C - "$@" <<'PYEOF'
import glob, os, sys
sys.path.insert(0, os.path.join(os.environ["PKG"], "refe"))
from onpolicy_relabel_pdm import served_keys
print(len(served_keys(sorted(glob.glob(os.path.join(sys.argv[1], "onpolicy_*.jsonl"))))))
PYEOF
}
on_exit() {
  rc=$?
  touch "$WSTOP" 2>/dev/null
  if [ -n "${WPID:-}" ]; then
    for _w in $(seq 1 36); do kill -0 "$WPID" 2>/dev/null || break; sleep 5 200>&-; done
    kill -0 "$WPID" 2>/dev/null && kill "$WPID" 2>/dev/null
  fi
  if [ -n "${PY:-}" ] && [ -d "$PKG/refe" ]; then       # one last push, so a REFUSED line reaches HF too
    ( cd "$PKG/refe" && "$PY" p2r_sync.py tick --run "$RUN" --logs "$P2" --chain-log "$L" --outbox "$P2/outbox_exit" ) \
      >> "$P2/exit_push.log" 2>&1 200>&-
  fi
  exit $rc
}

exec 200>"$P2/chain.lock"
flock -n 200 || { echo "ZZP2R_REFUSED another p2r_pod.sh holds $P2/chain.lock"; exit 1; }
trap on_exit EXIT
log "chain start (p2r_pod.sh md5 $(md5sum "$0" | cut -c1-12))"

# ---- 0. the pod, before anything is downloaded
[ -n "${HF_TOKEN:-}" ] || refuse "HF_TOKEN is not set in the environment"
for f in refe/train.py refe/p2_gate.py refe/p2r_sync.py refe/p2r_gate_prep.py refe/pdm_label_complete.py \
         refe/onpolicy_relabel_pdm.py code/pod_teacher_env.sh code/pod_train.sh code/pod_dataprep.sh splits/navtrain.yaml; do
  [ -f "$PKG/$f" ] || refuse "$PKG/$f missing: ship code/ refe/ splits/ to $PKG first"
done
# ⛔ never judge the disk with df (a pod volume's quota is invisible to it): WRITE to it.
DDGB="${P2R_DD_GB:-8}"
dd if=/dev/zero of="$DATA/.dd_test" bs=1M count=$((DDGB * 1024)) conv=fsync 2>> $L 200>&-
[ "$(stat -c %s "$DATA/.dd_test" 2>/dev/null)" = "$((DDGB * 1024 * 1024 * 1024))" ] || { rm -f "$DATA/.dd_test"; refuse "dd write test of ${DDGB} GB failed"; }
rm -f "$DATA/.dd_test"
log "dd write test ${DDGB} GB ok"

# ---- 1. env
if ! [ -f "$WORK/teacher_env.sh" ]; then
  env -u HF_TOKEN bash "$PKG/code/pod_teacher_env.sh" > "$P2/teacher_env.log" 2>&1 200>&-
  grep -q "^ZZOK teacher env ready" "$P2/teacher_env.log" || refuse "pod_teacher_env.sh did not finish (teacher_env.log)"
fi
source "$WORK/teacher_env.sh"
PY="$DRIVERL_EVAL_PYTHON"; PY_C="$PY"; export PKG PY_C
export REFE_BACKBONE_ROOT="$DATA/backbones" PYTHONUNBUFFERED=1
if ! command -v zstd > /dev/null 2>&1 && ! "$PY" -c "import zstandard" 2> /dev/null; then
  env -u HF_TOKEN apt-get install -y -q zstd >> "$P2/zstd_install.log" 2>&1 200>&- \
    || env -u HF_TOKEN uv pip install -q --python "$PY" --no-deps zstandard >> "$P2/zstd_install.log" 2>&1 200>&-
fi
command -v zstd > /dev/null 2>&1 || "$PY" -c "import zstandard" 2> /dev/null || refuse "no zstd decoder (zstd_install.log)"
cd "$PKG/refe" || refuse "cannot cd $PKG/refe"
stage_mark env; log "env ok"; stop_after env

# ---- watcher: logs every P2R_SYNC_MIN (30), checkpoints every P2R_HEAVY_MIN (30); a failed push never stops the run
rm -f "$WSTOP"
"$PY" p2r_sync.py watch --run "$RUN" --logs "$P2" --chain-log "$L" --outbox "$P2/outbox" --stop-file "$WSTOP" \
  --interval-min "${P2R_SYNC_MIN:-30}" --heavy-min "${P2R_HEAVY_MIN:-30}" \
  --tail "$PDMLOG:pdm_train.tail.log:300" >> "$P2/sync.log" 2>&1 200>&- &
WPID=$!

# ---- 2. DINOv3 weights (pinned commit + sha256, pod_train.sh)
if ! stage_done weights; then
  env -u HF_TOKEN STAGES=weights bash "$PKG/code/pod_train.sh" > "$P2/weights.log" 2>&1 200>&-
  grep -q "^ZZOK weights ZZ" "$P2/weights.log" || refuse "pod_train.sh weights stage did not print ZZOK weights"
  stage_mark weights
fi
log "weights ok"; stop_after weights

# ---- 3. model_final + the training bank + the on-policy sets, from the private HF repos, md5-pinned
if ! stage_done restore; then
  "$PY" p2r_sync.py restore --work "$DATA" > "$P2/restore.log" 2>&1 200>&-
  grep -q "^ZZRESTORE_OK" "$P2/restore.log" || refuse "restore failed: $(tail -n 1 "$P2/restore.log" | cut -c1-300)"
  stage_mark restore
fi
log "restore ok"; stop_after restore

# ---- 3b. a NEW pod after a lost one: the PDM labels (+ the relabeller's log) and ckpt_last.pt come back from the OUTPUT
#          repo, each checked against its sha256 / size. On a first run that repo is empty and this is a no-op.
if ! stage_done recover; then
  "$PY" p2r_sync.py recover --work "$DATA" --run "$RUN" > "$P2/recover.log" 2>&1 200>&-
  grep -q "^ZZRECOVER_OK" "$P2/recover.log" || refuse "recover failed: $(tail -n 1 "$P2/recover.log" | cut -c1-300)"
  stage_mark recover
fi
log "recover: $(grep '^ZZRECOVER_OK' "$P2/recover.log" | tail -n 1 | cut -c1-80)"; stop_after recover

# ---- 4. is the PDM relabel already complete? Restored labels are AUDITED by the same check as a fresh relabel, not trusted.
want="${P2R_EXPECT_SERVED:-174912}"
NEED=1
if stage_done relabel; then
  NEED=0
elif [ -f "$PDMLOG" ] && "$PY" pdm_label_complete.py --log "$PDMLOG" --out "$PDM" --sets "$SETS" --expect "$want" \
       --json "$P2/pdm_complete.json" > "$P2/pdm_complete.out" 2>&1 200>&- && grep -q "^ZZPDM_COMPLETE" "$P2/pdm_complete.out"; then
  NEED=0; stage_mark relabel
fi

# ---- 5. navtrain DBs (only the relabel needs them), then the pixels in the background (the relabel does not read pixels)
if [ "$NEED" = 1 ] && ! stage_done dbs; then
  env -u HF_TOKEN STAGES=dbs bash "$PKG/code/pod_dataprep.sh" > "$P2/dbs.log" 2>&1 200>&-
  grep -q "^ZZOK dbs 1192 files ZZ" "$P2/dbs.log" || refuse "navtrain DBs: want 1192 files (dbs.log)"
  stage_mark dbs
fi
log "dbs ok (relabel needed: $NEED)"; stop_after dbs
PIXPID=""
if ! stage_done pix; then
  # flock: a pix job orphaned by an earlier refusal makes this one WAIT, then finish the resumable stage (no two writers)
  ( flock 201; env -u HF_TOKEN STAGES=pix bash "$PKG/code/pod_dataprep.sh" >> "$P2/pix.log" 2>&1 ) 201>"$P2/pix.lock" 200>&- &
  PIXPID=$!
fi

# ---- 6. the PDM training relabel, to COMPLETION (pdm_label_complete.py decides; a restart redoes nothing)
if [ "$NEED" = 1 ]; then
  export NUPLAN_DATA_ROOT="$DATA" REFE_NUPLAN_DB_ROOT="$DATA/navtrain_dbs" REFE_NAVTRAIN_YAML="$PKG/splits/navtrain.yaml"
  have="$(served_count "$SETS" 2>> $L)"
  [ "$have" = "$want" ] || refuse "restored sets serve $have keys, expected $want (P2R_EXPECT_SERVED)"
  CORES="$(nproc)"
  if { read -r q per < /sys/fs/cgroup/cpu.max; } 2> /dev/null && [ "$q" != "max" ] && [ "${per:-0}" -gt 0 ]; then
    CORES=$(( q / per > 0 ? q / per : 1 ))
  fi
  RW="${P2R_RELABEL_WORKERS:-$(( CORES < 16 ? CORES : 16 ))}"
  log "relabel: $have sets, $RW workers (cgroup cores $CORES)"
  for att in 1 2 3; do
    ( echo 1000 > /proc/self/oom_score_adj 2> /dev/null
      exec env -u HF_TOKEN OMP_NUM_THREADS=1 nice -n 19 "$PY" onpolicy_relabel_pdm.py --sets "$SETS" --out "$PDM" \
        --served-only --workers "$RW" ) >> "$PDMLOG" 2>&1 200>&-
    echo "ZZPDM_TRAIN_EXIT $?" >> "$PDMLOG"
    "$PY" pdm_label_complete.py --log "$PDMLOG" --out "$PDM" --sets "$SETS" --expect "$want" --json "$P2/pdm_complete.json" \
      > "$P2/pdm_complete.out" 2>&1 200>&-
    grep -q "^ZZPDM_COMPLETE" "$P2/pdm_complete.out" && break
    log "relabel attempt $att incomplete: $(tail -n 1 "$P2/pdm_complete.out" | cut -c1-200)"
  done
  grep -q "^ZZPDM_COMPLETE" "$P2/pdm_complete.out" || refuse "PDM labels incomplete after 3 attempts (pdm_complete.out)"
  stage_mark relabel
fi
log "PDM labels complete"; stop_after relabel
if [ -n "$PIXPID" ]; then
  wait "$PIXPID"
  [ "$(grep '^ZZOK pix [0-9]*/32 shards ZZ$' "$P2/pix.log" | tail -n 1)" = "ZZOK pix 32/32 shards ZZ" ] || refuse "pixels incomplete: want 32/32 shards (pix.log; rerun to resume)"
  stage_mark pix
fi
log "pixels ok"; stop_after pix

# ---- 7. bank the labels OFF the pod before training (~11 h of CPU; they were lost once). Labels that came FROM HF are banked.
if ! stage_done pdm_push; then
  if grep -q "labels=yes" "$P2/recover.log"; then
    log "PDM labels were restored from HF: already banked"
  else
    "$PY" p2r_sync.py pdm-push --labels "$PDM" --log "$PDMLOG" --chain-log "$L" --outbox "$P2/outbox_pdm" \
      > "$P2/pdm_push.log" 2>&1 200>&-
    grep -q " ZZEGRESS_OK p2r_pdm_labels " "$L" || refuse "PDM labels were not banked to HF (pdm_push.log)"
  fi
  stage_mark pdm_push
fi
log "PDM labels banked on HF"; stop_after pdm_push

# ---- 8. the launch gate: G1-G4, on a small bank where G3's mutation is NOT inert; the JSON is parsed
if ! stage_done gate; then
  rm -rf "$P2/gate_sets" "$P2/gate_pdm" "$P2/gate"
  "$PY" p2r_gate_prep.py --sets "$SETS" --pdm "$PDM" --out-sets "$P2/gate_sets" --out-pdm "$P2/gate_pdm" \
    > "$P2/gate_prep.log" 2>&1 200>&- || refuse "gate subset build failed: $(tail -n 1 "$P2/gate_prep.log" | cut -c1-200)"
  env -u HF_TOKEN OMP_NUM_THREADS=2 "$PY" p2_gate.py --init-from "$INIT" --onpolicy "$P2/gate_sets" \
    --pdm-labels "$P2/gate_pdm" --out "$P2/gate" > "$P2/gate.log" 2>&1 200>&-
  echo "ZZEXIT $?" >> "$P2/gate.log"
  gate_pass "$P2/gate/gate.json" 2>> $L || refuse "the P2 gate did not PASS (gate.log, gate/gate.json)"
  stage_mark gate
fi
log "gate PASS"; stop_after gate

# ---- 9. preflight on the REAL configuration, then the supervised run (the arguments are p2_chain.sh's, unchanged)
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
ARGS="--backbone vitl16 --targets /workspace/data/refe_navtrain/train_grow --scorer-targets /workspace/data/refe_navtrain/train_grow --images /workspace/data/navtrain_pixels --calib /workspace/data/refe_navtrain/train_grow/calib_table.json --init-from /workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt --scorer-mode onpolicy --onpolicy-targets /workspace/data/refe_navtrain/onpolicy/sets --pdm-labels /workspace/data/refe_sft2/train_pdm --pdm-only --scorer-sees-trunk --epochs 3 --epoch-unit scenes --batch 4 --accum 64 --lr 5e-5 --amp bf16 --tf32 --compile --workers 2 --log-every 1 --ckpt-every-min 20 --out $RUN --resume"
env -u HF_TOKEN "$PY" train.py $ARGS --preflight > "$P2/preflight.log" 2>&1 200>&-
grep -q "^PREFLIGHT_OK" "$P2/preflight.log" || refuse "train.py --preflight did not print PREFLIGHT_OK"
log "train.py preflight OK; launching"
for att in 1 2 3 4 5 6; do
  if [ -f $RUN/summary.json ] && grep -q '"done": true' $RUN/summary.json; then break; fi
  env -u HF_TOKEN "$PY" train.py $ARGS >> "$P2/train.log" 2>&1 200>&-
  log "attempt $att exited rc $?"
  sleep 60 200>&-
done
if [ -f $RUN/summary.json ] && grep -q '"done": true' $RUN/summary.json; then log "ZZP2_DONE"
else log "ZZP2_FAILED (no done=true after the attempts)"; exit 1; fi

# ---- 10. the final push is the hard gate: success is the REMOTE listing (and model_final's sha256), not the push call
touch "$WSTOP"
for _w in $(seq 1 120); do kill -0 "$WPID" 2> /dev/null || break; sleep 5 200>&-; done
"$PY" p2r_sync.py final --run "$RUN" --logs "$P2" --chain-log "$L" --outbox "$P2/outbox_final" > "$P2/final.log" 2>&1 200>&-
if [ $? -eq 0 ] && grep -q "^ZZP2R_FINAL_OK" "$P2/final.log"; then log "ZZP2R_DONE"; exit 0; fi
log "ZZP2R_FINAL_FAILED (final.log): the checkpoint is on the pod only until a retry succeeds"
exit 1
