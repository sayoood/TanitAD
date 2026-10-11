#!/bin/bash
# P6 (paper-faithful REFe, SPEC_P6_PAPER_FAITHFUL.md section 5 / D10): the SAME A40 pod, right AFTER P2 -> a finished 25-epoch P6 run, banked OFF the pod.
# Runbook: eval/P6_POD_RUNBOOK.md.  Pattern: refe/p2r_pod.sh.  Tests: refe/test_p6_chain.py.  Pre-registration: eval/PREREG_P6.md (not edited here).
#
# Needs: HF_TOKEN in the environment (write-scoped, the PI provides it; read here, never echoed, never an argument, stripped from every child except the HF tools) and
#        code/ refe/ splits/ SHIP_MANIFEST.json shipped to $P6_PKG (default /workspace/refe-plan-p6).
#        ⛔ NOT the directory P2 runs from: P2's token binds ITS code tree, and overwriting refe-plan/ would void it mid-run.  P2's DATA (DBs, pixels, bank, weights) is reused IN PLACE.
# PI ruling P6-Y (2026-10-11): unwrapped yaw L1, as the paper => --yaw-wrap off.  It is the default of the ONE variable P6_YAW_WRAP; a blank value, or any other, REFUSES (and the
# refe_p6 profile refuses `--yaw-wrap on` as a value mismatch).
# Start (detached):  P6_YAW_WRAP=<on|off> nohup setsid bash /workspace/refe-plan-p6/refe/p6_pod.sh > /workspace/data/refe_p6/p6_stdout.log 2>&1 < /dev/null &
# Re-running after ANY interruption is safe: finished stages leave /workspace/data/refe_p6/stage/<name>.ok (a stage whose result depends on the argv carries the argv hash).
# P6_STOP_AFTER=<env|weights|recover|inputs|commands|cache|lanes|lab|mem|egr|launch_gate> ends the chain after that stage (a cheap dry run).
#
# Stages: pod checks -> env -> [watcher: logs + checkpoints to a NEW private HF repo every P6_SYNC_MIN / P6_HEAVY_MIN = 30 min] -> weights -> recover (resume from HF) ->
#         inputs (P2's bank / DBs / pixels REUSED in place if present, fetched only if absent) -> commands (D6, P6-N-SRC: NAVSIM's driving_command on rank-0 rows) ->
#         cache (D9: the metric-cache parts, once per token, + the D9 exactness check) -> lanes (D7: lane-rank twins; REFUSES when its inputs are missing) ->
#         [ARGS, defined ONCE, the yaw value checked] -> lab (G-LIVE-LAB) -> mem (G-LIVE-MEM) -> egr (G-EGR incl. the first-checkpoint size) ->
#         launch_gate (the BINDING launch gate, CLAUDE.md PI 2026-09-26: a PASS token) -> preflight -> train (supervised; the token is VERIFIED against the exact ARGS before the
#         preflight and before EVERY launch) -> final push (HF, remote-verified).
# Every child is launched with fd 200 closed (the chain holds its flock on 200).  Markers are opaque ZZ tokens.
set -u
WORK="${WORK:-/workspace}"; DATA=$WORK/data
PKG="${P6_PKG:-$WORK/refe-plan-p6}"
P2DIR=$DATA/refe_p2
P6=$DATA/refe_p6; S=$P6/stage; L=$P6/chain.log
RUN=$DATA/refe_runs/p6_paper_faithful
BANK=$DATA/refe_navtrain/train_grow                  # P2's bank, read-only here
CBANK=$DATA/refe_navtrain/train_grow_cmd             # the same rows + NAVSIM's command on the rank-0 ones (annotate_commands.py writes a NEW directory)
MC=$DATA/refe_navtrain/metric_cache                  # the D9 parts
DBS=$DATA/navtrain_dbs; PIX=$DATA/navtrain_pixels
NAVLOGS="${P6_NAVSIM_LOGS:-$DATA/navsim_logs/trainval}"
# the NAVSIM metadata (a <log>.pkl per log). URL = NAVSIM's download script, read from its GitHub page (PUBLISHED-SECONDARY); size and md5 UNVERIFIED -> content is verified instead.
# `${VAR-default}`: P6_NAVSIM_LOGS_URL="" FORBIDS the download and the stage then refuses if the logs are absent.
NAVLOGS_URL="${P6_NAVSIM_LOGS_URL-https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz}"
REPO="${P6_REPO:-Sayood/tanitad-refe-p6-paper-faithful}"          # a NEW private model repo, created by the first push
EXP_DBS="${P6_EXPECT_DBS:-1192}"; EXP_PIX="${P6_EXPECT_PIX_SHARDS:-32}"
# ⛔ THE ONE PLACE PI RULING P6-Y ENTERS (2026-10-11: "unwrapped yaw, as the paper"): off = the paper's literal L1 on the raw yaw difference (on = the wrapped error of P2 / SFT-4).
# `${VAR-off}` (no colon): UNSET means off; a BLANKED variable stays blank and yaw_check refuses it.
YAW_WRAP="${P6_YAW_WRAP-off}"
WSTOP=$P6/sync.stop
# the launch-gate HMAC key lives on the persistent volume but OUTSIDE the shipped code tree and outside anything the watcher pushes (*.log / *.json only)
export TANITAD_GATE_KEY_FILE="$P6/launch_gate.key"
mkdir -p "$P6" "$S"
log() { echo "$(date -u +%FT%T) $*" >> $L; }
refuse() { log "REFUSED: $*"; exit 1; }
stage_done() { [ -f "$S/$1.ok" ]; }
stage_mark() { date -u +%FT%T > "$S/$1.ok"; }
key_done() { [ "$(cat "$S/$1.ok" 2> /dev/null)" = "$2" ]; }        # a stage whose result is only valid for one configuration
key_mark() { printf '%s' "$2" > "$S/$1.ok"; }
invalidate() { local s; for s in "$@"; do rm -f "$S/$s.ok"; done; }   # a re-measured instrument or a rebuilt input voids everything downstream of it
stop_after() { if [ "${P6_STOP_AFTER:-}" = "$1" ]; then log "stopping after $1 (P6_STOP_AFTER)"; exit 0; fi; }
yaw_check() {
  case "$YAW_WRAP" in
    off) return 0 ;;
  esac
  refuse "PI ruling P6-Y (2026-10-11) is yaw-wrap = off (unwrapped, as the paper), but P6_YAW_WRAP is '${YAW_WRAP}': the chain will not guess the value of --yaw-wrap (SPEC_P6 D5), and the refe_p6 profile refuses any other"
}
# cgroup CPU quota, NEVER nproc (96 cores shown on a 7.65-CPU quota, MEASURED, code/pod_dataprep.sh).  v2 cpu.max, then v1 cfs, then nproc (logged as UNVERIFIED).
cgroup_cores() {
  local c q per
  c="$(nproc)"; CORES_SRC="nproc (UNVERIFIED: no cgroup quota file)"
  if { read -r q per < /sys/fs/cgroup/cpu.max; } 2> /dev/null && [ "$q" != "max" ] && [ "${per:-0}" -gt 0 ]; then
    c=$(( q / per > 0 ? q / per : 1 )); CORES_SRC="cgroup v2 cpu.max"
  elif { read -r q < /sys/fs/cgroup/cpu/cpu.cfs_quota_us; read -r per < /sys/fs/cgroup/cpu/cpu.cfs_period_us; } 2> /dev/null && [ "${q:-0}" -gt 0 ] && [ "${per:-0}" -gt 0 ]; then
    c=$(( q / per > 0 ? q / per : 1 )); CORES_SRC="cgroup v1 cfs"
  fi
  echo "$c|$CORES_SRC"          # printed together: this runs in a command substitution, so a variable set here would not survive it
}
on_exit() {
  rc=$?
  touch "$WSTOP" 2>/dev/null
  if [ -n "${WPID:-}" ]; then
    for _w in $(seq 1 36); do kill -0 "$WPID" 2>/dev/null || break; sleep 5 200>&-; done
    kill -0 "$WPID" 2>/dev/null && kill "$WPID" 2>/dev/null
  fi
  if [ -n "${PY:-}" ] && [ -d "$PKG/refe" ]; then       # one last push, so a REFUSED line reaches HF too
    ( cd "$PKG/refe" && "$PY" p2r_sync.py tick --repo "$REPO" --run "$RUN" --logs "$P6" --chain-log "$L" --outbox "$P6/outbox_exit" ) \
      >> "$P6/exit_push.log" 2>&1 200>&-
  fi
  exit $rc
}

exec 200>"$P6/chain.lock"
flock -n 200 || { echo "ZZP6_REFUSED another p6_pod.sh holds $P6/chain.lock"; exit 1; }
trap on_exit EXIT
log "chain start (p6_pod.sh md5 $(md5sum "$0" | cut -c1-12))"

# ---- P2 must be OVER: never add GPU/RAM load to a pod that is training (CLAUDE.md).  P2's chain holds its flock for its whole life.
if [ -e "$P2DIR/chain.lock" ] && ! ( flock -n 9 ) 9> "$P2DIR/chain.lock"; then
  refuse "the P2 chain still holds $P2DIR/chain.lock: P6 starts only after ZZP2R_DONE"
fi
# ---- the PI's ruling P6-Y, checked NOW (cheap) and again right before ARGS: a blanked or different value stops the chain before hours of CPU are spent
yaw_check

# ---- 0. the pod, before anything is downloaded
[ -n "${HF_TOKEN:-}" ] || refuse "HF_TOKEN is not set in the environment"
for f in refe/train.py refe/p2r_sync.py refe/hf_egress.py refe/p6_prep.py refe/build_metric_cache.py refe/annotate_commands.py \
         refe/diag_p6_labeller_identity.py refe/launch_gate.py refe/gate_token.py refe/gate_checks.py refe/gate_fixture.py \
         refe/test_launch_gate.py refe/test_p2r_chain.py refe/test_p6_levers.py refe/test_p6_chain.py SHIP_MANIFEST.json \
         code/pod_teacher_env.sh code/pod_train.sh code/pod_dataprep.sh splits/navtrain.yaml; do
  [ -f "$PKG/$f" ] || refuse "$PKG/$f missing: ship code/ refe/ splits/ SHIP_MANIFEST.json to $PKG first (a directory of its own, NOT the one P2 runs from)"
done
# ⛔ never judge the disk with df (a pod volume's quota is invisible to it): WRITE to it.
DDGB="${P6_DD_GB:-8}"
dd if=/dev/zero of="$DATA/.dd_test" bs=1M count=$((DDGB * 1024)) conv=fsync 2>> $L 200>&-
[ "$(stat -c %s "$DATA/.dd_test" 2>/dev/null)" = "$((DDGB * 1024 * 1024 * 1024))" ] || { rm -f "$DATA/.dd_test"; refuse "dd write test of ${DDGB} GB failed"; }
rm -f "$DATA/.dd_test"
log "dd write test ${DDGB} GB ok"

# ---- 1. env (P2's teacher env is REUSED when present: it holds the NAVSIM stack, the maps and the DriveRL fork)
if ! [ -f "$WORK/teacher_env.sh" ]; then
  env -u HF_TOKEN bash "$PKG/code/pod_teacher_env.sh" > "$P6/teacher_env.log" 2>&1 200>&-
  grep -q "^ZZOK teacher env ready" "$P6/teacher_env.log" || refuse "pod_teacher_env.sh did not finish (teacher_env.log)"
fi
source "$WORK/teacher_env.sh"
PY="$DRIVERL_EVAL_PYTHON"; export PKG
export REFE_BACKBONE_ROOT="$DATA/backbones" PYTHONUNBUFFERED=1
CG="$(cgroup_cores)"; CORES="${CG%%|*}"; CORES_SRC="${CG#*|}"
# labeller pool: leave 4 vCPU for the trainer, its loaders and IPC (M-LAB sizing); at most 16
LW="${P6_LABEL_WORKERS:-$(( CORES - 4 < 1 ? 1 : (CORES - 4 > 16 ? 16 : CORES - 4) ))}"
CW="${P6_CACHE_WORKERS:-$(( CORES < 1 ? 1 : (CORES > 24 ? 24 : CORES) ))}"
log "cores $CORES ($CORES_SRC): label workers $LW, cache workers $CW"
[ "$LW" -ge 12 ] || log "WARNING: $LW label workers (< 12): G-LIVE-LAB's 6.735 sets/s bar is unlikely to hold; the gate decides"
cd "$PKG/refe" || refuse "cannot cd $PKG/refe"
stage_mark env; log "env ok"; stop_after env

# ---- watcher: logs every P6_SYNC_MIN (30), checkpoints every P6_HEAVY_MIN (30), to a NEW private repo; a failed push never stops the run
rm -f "$WSTOP"
"$PY" p2r_sync.py watch --repo "$REPO" --run "$RUN" --logs "$P6" --chain-log "$L" --outbox "$P6/outbox" --stop-file "$WSTOP" \
  --interval-min "${P6_SYNC_MIN:-30}" --heavy-min "${P6_HEAVY_MIN:-30}" --tail "$P6/train.log:train.tail.log:400" >> "$P6/sync.log" 2>&1 200>&- &
WPID=$!

# ---- 2. DINOv3 weights (pinned commit + sha256; an existing file is HASHED and kept, never re-downloaded)
if ! stage_done weights; then
  env -u HF_TOKEN STAGES=weights WORK="$WORK" DATA="$DATA" PKG="$PKG" bash "$PKG/code/pod_train.sh" > "$P6/weights.log" 2>&1 200>&-
  grep -q "^ZZOK weights ZZ" "$P6/weights.log" || refuse "pod_train.sh weights stage did not print ZZOK weights"
  stage_mark weights
fi
log "weights ok"; stop_after weights

# ---- 3. RESUME FROM HF: a NEW pod picks up what a lost one pushed to the P6 repo (ckpt_last.pt + the run files), each checked against its sha256 / size.  A first run: a no-op.
if ! stage_done recover; then
  "$PY" p2r_sync.py recover --work "$DATA" --run "$RUN" --repo "$REPO" > "$P6/recover.log" 2>&1 200>&-
  grep -q "^ZZRECOVER_OK" "$P6/recover.log" || refuse "recover failed: $(tail -n 1 "$P6/recover.log" | cut -c1-300)"
  stage_mark recover
fi
log "recover: $(grep '^ZZRECOVER_OK' "$P6/recover.log" | tail -n 1 | cut -c1-80)"; stop_after recover

# ---- 4. inputs: P2's files are REUSED IN PLACE.  Each fetch exists ONLY inside a fetch_* function and is reached ONLY as `have_X || fetch_X` (test_p6_chain pins both).
have_bank() { [ -s "$BANK/calib_table.json" ] && ls "$BANK"/targets_rank*.jsonl > /dev/null 2>&1; }
have_dbs() { [ "$(ls "$DBS"/*.db 2> /dev/null | wc -l)" -ge "$EXP_DBS" ]; }
have_pix() { [ "$(ls "$PIX"/.done_* 2> /dev/null | wc -l)" -ge "$EXP_PIX" ]; }
have_navlogs() { ls "$NAVLOGS"/*.pkl > /dev/null 2>&1; }
fetch_bank() {
  if ! command -v zstd > /dev/null 2>&1 && ! "$PY" -c "import zstandard" 2> /dev/null 200>&-; then
    env -u HF_TOKEN apt-get install -y -q zstd >> "$P6/zstd_install.log" 2>&1 200>&- \
      || env -u HF_TOKEN uv pip install -q --python "$PY" --no-deps zstandard >> "$P6/zstd_install.log" 2>&1 200>&-
  fi
  "$PY" p2r_sync.py restore --work "$DATA" > "$P6/restore.log" 2>&1 200>&-
  grep -q "^ZZRESTORE_OK" "$P6/restore.log"
}
fetch_dbs() {
  env -u HF_TOKEN STAGES=dbs WORK="$WORK" DATA="$DATA" PKG="$PKG" bash "$PKG/code/pod_dataprep.sh" > "$P6/dbs.log" 2>&1 200>&-
  grep -q "^ZZOK dbs $EXP_DBS files ZZ" "$P6/dbs.log"
}
fetch_pix() {
  env -u HF_TOKEN STAGES=pix WORK="$WORK" DATA="$DATA" PKG="$PKG" bash "$PKG/code/pod_dataprep.sh" > "$P6/pix.log" 2>&1 200>&-
  [ "$(grep '^ZZOK pix [0-9]*/[0-9]* shards ZZ$' "$P6/pix.log" | tail -n 1)" = "ZZOK pix $EXP_PIX/$EXP_PIX shards ZZ" ]
}
fetch_navlogs() {
  [ -n "$NAVLOGS_URL" ] || return 1
  mkdir -p "$DATA/navsim_logs_dl" "$NAVLOGS"
  curl -sSL --retry 10 --retry-all-errors -C - -o "$DATA/navsim_logs_dl/meta.tgz" "$NAVLOGS_URL" 200>&- || return 1
  tar -xzf "$DATA/navsim_logs_dl/meta.tgz" -C "$DATA/navsim_logs_dl" 200>&- || return 1
  find "$DATA/navsim_logs_dl" -name '*.pkl' -exec mv -f {} "$NAVLOGS"/ \; 200>&-
  rm -f "$DATA/navsim_logs_dl/meta.tgz"
  have_navlogs
}
# (no stage marker gates this stage: the probes are cheap, so a deleted file is noticed on every start instead of trusted from a marker)
have_bank || fetch_bank || refuse "the navtrain training bank is absent and could not be restored (restore.log)"
have_dbs || fetch_dbs || refuse "navtrain DBs: want $EXP_DBS files (dbs.log)"
have_pix || fetch_pix || refuse "pixels incomplete: want $EXP_PIX/$EXP_PIX shards (pix.log; rerun to resume)"
[ "$(ls -d "${NUPLAN_MAPS_ROOT:-/nonexistent}"/*/ 2> /dev/null | wc -l)" -ge 4 ] || refuse "nuPlan maps: fewer than 4 city dirs under NUPLAN_MAPS_ROOT"
stage_mark inputs
log "inputs ok (bank, $EXP_DBS DBs, $EXP_PIX pixel shards, maps: reused in place where present)"; stop_after inputs

# ---- 5. D6 (PI P6-N-SRC): NAVSIM's OWN driving_command on the rank-0 rows, so the student sees at TRAIN time what NAVSIM hands it at TEST time.  Needs the navtrain metadata: REFUSES without it.
if ! { stage_done commands && ls "$CBANK"/targets_rank*.jsonl > /dev/null 2>&1; }; then
  have_navlogs || fetch_navlogs || refuse "D6: no NAVSIM metadata (*.pkl) under $NAVLOGS and no download (P6_NAVSIM_LOGS_URL): the command source P6-N-SRC cannot be built"
  "$PY" p6_prep.py command-table --logs "$NAVLOGS" --bank "$BANK" --out "$P6/commands.json" > "$P6/commands_table.out" 2>&1 200>&-
  grep -q "^ZZCMD_TABLE_OK " "$P6/commands_table.out" || refuse "D6: the command table does not cover the bank: $(tail -n 1 "$P6/commands_table.out" | cut -c1-200)"
  rm -rf "$CBANK.tmp"
  "$PY" annotate_commands.py --bank "$BANK" --table "$P6/commands.json" --out-dir "$CBANK.tmp" > "$P6/annotate.out" 2>&1 200>&-
  grep -q "^ZZANNOTATE rows=[0-9]* annotated=[1-9][0-9]*" "$P6/annotate.out" || refuse "D6: annotate_commands.py annotated nothing (annotate.out)"
  rm -rf "$CBANK"; mv "$CBANK.tmp" "$CBANK"
  invalidate cache lanes lab mem egr launch_gate
  stage_mark commands
fi
log "commands ok: $(tail -n 1 "$P6/commands_table.out" | cut -c1-120)"; stop_after commands

# ---- 6. D9: the metric-cache PARTS, once per token (~24 CPU-h, ESTIMATED; resumable: existing files are skipped), then the EXACTNESS check on a sample
if ! { stage_done cache && [ -d "$MC" ]; }; then
  export NUPLAN_DATA_ROOT="$DATA" REFE_NUPLAN_DB_ROOT="$DBS" REFE_NAVTRAIN_YAML="$PKG/splits/navtrain.yaml"
  for att in 1 2 3; do
    ( echo 1000 > /proc/self/oom_score_adj 2> /dev/null
      exec env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK OMP_NUM_THREADS=1 nice -n 10 "$PY" build_metric_cache.py build --targets "$CBANK" --out "$MC" \
        --db-root "$DBS" --map-root "${NUPLAN_MAPS_ROOT:-}" --workers "$CW" ) > "$P6/cache_attempt$att.out" 2>&1 200>&-
    cat "$P6/cache_attempt$att.out" >> "$P6/cache.log"
    grep -q "^ZZMETRIC_CACHE_DONE .* failed=0 " "$P6/cache_attempt$att.out" && break
    log "cache attempt $att incomplete: $(tail -n 1 "$P6/cache_attempt$att.out" | cut -c1-200)"
  done
  grep -q "^ZZMETRIC_CACHE_DONE .* failed=0 " "$P6/cache_attempt$att.out" || refuse "metric cache incomplete after 3 attempts (cache.log)"
  TOKS="$("$PY" p6_prep.py sample-tokens --cache "$MC" --n "${P6_D9_VERIFY_TOKENS:-24}" 200>&- | sed -n 's/^ZZTOKENS //p')"
  [ -n "$TOKS" ] || refuse "the metric cache holds no parts after the build"
  env -u HF_TOKEN OMP_NUM_THREADS=1 "$PY" build_metric_cache.py verify --cache "$MC" --db-root "$DBS" --map-root "${NUPLAN_MAPS_ROOT:-}" --tokens "$TOKS" --n-sets 4 \
    > "$P6/d9_verify.out" 2>&1 200>&-
  grep -q "^ZZD9_PASS" "$P6/d9_verify.out" || refuse "D9: cached parts are NOT bit-identical to fresh parts (d9_verify.out)"
  invalidate lanes lab mem egr launch_gate
  stage_mark cache
fi
log "metric cache ok: $(grep -h '^ZZMETRIC_CACHE_DONE' "$P6"/cache_attempt*.out 2> /dev/null | tail -n 1 | cut -c1-120)"; stop_after cache

# ---- 7. D7: lane-rank TWINS get their OWN route's parts.  Explicit, never silently skipped: it REFUSES when its inputs are missing.  A bank with no twins is logged as such.
if ! stage_done lanes; then
  RANKS="$("$PY" p6_prep.py lane-ranks --bank "$CBANK" 200>&- | sed -n 's/^ZZLANE_RANKS//p')"
  if [ -z "${RANKS// /}" ]; then
    log "D7: the bank carries no lane-rank twins: nothing to build (goal-horizon twins share the logged route's parts)"
  else
    # inputs: the DriveRL fork's route patch (pod only) must import, else the twins would be cached on the LOGGED route and labelled as twins
    env -u HF_TOKEN PYTHONPATH="$PKG/code:${PYTHONPATH:-}" "$PY" -c "import route_lane_rank_patch; from nuplan.planning.script import driverl_runtime_map_features" \
      > "$P6/lanes_probe.out" 2>&1 200>&- \
      || refuse "D7: the DriveRL route patch is not importable (lanes_probe.out): lane-rank twins cannot be built; fix the env or decide the fallback (PI/Master Mind)"
    for k in $RANKS; do
      # DRIVERL_EVAL_ROUTE_LANE_RANK reaches the spawned pool workers through the venv's sitecustomize (a patch installed only in the parent never would)
      ( echo 1000 > /proc/self/oom_score_adj 2> /dev/null
        exec env -u HF_TOKEN DRIVERL_EVAL_ROUTE_LANE_RANK="$k" OMP_NUM_THREADS=1 nice -n 10 "$PY" build_metric_cache.py build --targets "$CBANK" --out "$MC" \
          --db-root "$DBS" --map-root "${NUPLAN_MAPS_ROOT:-}" --workers "$CW" --lane-rank "$k" ) > "$P6/lanes_$k.out" 2>&1 200>&-
      cat "$P6/lanes_$k.out" >> "$P6/cache.log"
      grep -q "^ZZMETRIC_CACHE_DONE .* failed=0 " "$P6/lanes_$k.out" || refuse "D7: the rank-$k twin build failed (lanes_$k.out)"
      "$PY" p6_prep.py lane-distinct --bank "$CBANK" --cache "$MC" --k "$k" --n 40 --min-frac "${P6_D7_MIN_DISTINCT:-0.9}" > "$P6/lanes_distinct_$k.out" 2>&1 200>&-
      if grep -q "^ZZD7_DISTINCT_OK " "$P6/lanes_distinct_$k.out"; then
        log "D7 rank $k: $(tail -n 1 "$P6/lanes_distinct_$k.out" | cut -c1-120)"
      elif [ "${P6_D7_FALLBACK:-0}" = 1 ]; then
        "$PY" p6_prep.py quarantine --cache "$MC" --k "$k" --dst "$DATA/refe_navtrain/metric_cache_twins_rejected" > "$P6/lanes_quarantine_$k.out" 2>&1 200>&-
        log "D7 FALLBACK (P6_D7_FALLBACK=1): PDM-Closed did not honour the route patch for rank $k; its twins are quarantined and train with the trajectory loss only"
      else
        refuse "D7: rank-$k twins are NOT on their own route ($(tail -n 1 "$P6/lanes_distinct_$k.out" | cut -c1-120)): PDM-Closed does not honour the patch. Set P6_D7_FALLBACK=1 for the declared fallback (trajectory loss only)"
      fi
    done
  fi
  invalidate lab mem egr launch_gate
  stage_mark lanes
fi
log "lanes ok"; stop_after lanes

# ---- 7b. THE TRAINING ARGUMENTS, defined ONCE and BEFORE every instrument and the launch gate: the token is minted over EXACTLY this string and verified against it
#          before the preflight and before every train.py launch.  --yaw-wrap is filled from YAW_WRAP (top of this file) and nothing else.
yaw_check
ARGS="--backbone vitl16 --targets $CBANK --images $PIX --calib $BANK/calib_table.json --metric-cache $MC --scorer-sees-trunk --yaw-wrap $YAW_WRAP --nav-mode command --labeller online --label-workers $LW --epochs 25 --epoch-unit scenes --batch 4 --accum 64 --lr 2e-4 --weight-decay 0.01 --amp bf16 --score-w 0.1 --tf32 --compile --workers 2 --log-every 1 --ckpt-every-min 20 --out $RUN --resume"
ARGS_SHA="$(printf '%s' "$ARGS" | sha256sum | cut -c1-16)"
log "ARGS sha $ARGS_SHA (--yaw-wrap $YAW_WRAP, --label-workers $LW)"
LG=$P6/lg
# ⛔ THE LAUNCH-GATE TOKEN (CLAUDE.md, PI 2026-09-26, BINDING): the trainer does not start unless this returns 0.  `launch_gate.py verify` RE-MEASURES the argv, the code tree
# and the data and checks the HMAC; it is never a grep of a log.   token_ok  -> rc 0 iff ZZGATE-MATCHZZ
token_ok() {
  tok="$(ls "$LG"/PASS_*.json 2> /dev/null | head -n 1)"
  [ -n "$tok" ] || return 1
  env -u HF_TOKEN "$PY" "$PKG/refe/launch_gate.py" verify --profile refe_p6 --token "$tok" --args "$ARGS" --pkg "$PKG" \
    --json "$P6/token_verify.json" >> "$P6/token_verify.log" 2>&1 200>&-
}
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

# ---- 8. G-LIVE-LAB's instrument: the ONLINE labeller == the OFFLINE one bit for bit (>= 64 sets), a stale proposal REFUSED, throughput at the LAUNCH worker count
if ! key_done lab "w$LW"; then
  TOKS="$("$PY" p6_prep.py sample-tokens --cache "$MC" --n 128 200>&- | sed -n 's/^ZZTOKENS //p')"
  [ -n "$TOKS" ] || refuse "no parts to sample for the labeller instrument"
  env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK "$PY" diag_p6_labeller_identity.py --cache "$MC" --db-root "$DBS" --map-root "${NUPLAN_MAPS_ROOT:-}" --tokens "$TOKS" \
    --workers "$LW" --n-sets "${P6_LAB_SETS:-256}" --out "$P6/lab.json" > "$P6/lab.log" 2>&1 200>&-
  grep -q "^ZZLAB_IDENTITY_PASS" "$P6/lab.log" || refuse "G-LIVE-LAB instrument: identity or the stale-proposal refusal failed (lab.log)"
  log "lab: $(grep '^ZZLAB_THROUGHPUT' "$P6/lab.log" | tail -n 1) sets/s at $LW workers (the gate judges it against 6.735)"
  invalidate launch_gate
  key_mark lab "w$LW"
fi
log "lab ok"; stop_after lab

# ---- 9. G-LIVE-MEM's instrument: the REAL launch configuration for 200 micro-batches on this card (peak <= 44 GB); its ckpt_last.pt is G-EGR's probe.  An instrument run, NOT a training run:
#          scratch --out, no --resume, never the real run directory.
if ! key_done mem "$ARGS_SHA"; then
  rm -rf "$P6/mem_run"
  MEM_ARGS="$("$PY" p6_prep.py mem-argv --args "$ARGS" --out "$P6/mem_run" 200>&- | sed -n 's/^ZZMEM_ARGV //p')"
  [ -n "$MEM_ARGS" ] || refuse "could not derive the memory-run argv"
  env -u HF_TOKEN "$PY" train.py $MEM_ARGS > "$P6/mem.log" 2>&1 200>&-
  grep -q "^TRAIN_DONE" "$P6/mem.log" || refuse "the G-LIVE-MEM run did not finish (mem.log)"
  { [ -s "$P6/mem_run/metrics.jsonl" ] && [ -s "$P6/mem_run/ckpt_last.pt" ]; } || refuse "the G-LIVE-MEM run left no metrics.jsonl / ckpt_last.pt"
  cp "$P6/mem_run/metrics.jsonl" "$P6/mem_metrics.jsonl"
  invalidate egr launch_gate
  key_mark mem "$ARGS_SHA"
fi
log "mem ok"; stop_after mem

# ---- 10. G-EGR's instrument: the gate JSON reaches the P6 repo, a push to a repo the token cannot write is refused BY THE SERVER, and the FIRST checkpoint's local size == its LISTING size
if ! key_done egr "$ARGS_SHA"; then
  rm -rf "$P6/egr"
  "$PY" hf_egress.py gate --repo "$REPO" --out "$P6/egr" --ckpt "$P6/mem_run/ckpt_last.pt" --chain-log "$L" > "$P6/egr.log" 2>&1 200>&-
  "$PY" p6_prep.py egr-ok --json "$P6/egr/g_egr_result.json" > "$P6/egr_ok.out" 2>&1 200>&-
  grep -q "^ZZEGR_OK" "$P6/egr_ok.out" || refuse "G-EGR instrument failed: $(tail -n 1 "$P6/egr_ok.out" | cut -c1-200) (egr.log)"
  invalidate launch_gate
  key_mark egr "$ARGS_SHA"
fi
log "egr ok"; stop_after egr

# ---- 11. THE LAUNCH GATE (G-HYG G-DVB G-LIVE G-CLOCK G-EVAL G-CKPT G-SUITE + G-LIVE-LAB G-LIVE-MEM G-EGR), on the REAL argv and the REAL data.  The smoke bank is carved from rows whose D9
#          parts exist.  A PASS token or a REFUSAL; the token is then VERIFIED (not grepped).  Route and contract: eval/REFE_LAUNCH_GATE.md.
if ! stage_done launch_gate; then
  rm -rf "$LG" "$P6/gate_bank"
  env -u HF_TOKEN OMP_NUM_THREADS=4 "$PY" launch_gate.py run --profile refe_p6 --pkg "$PKG" --args "$ARGS" --out-dir "$LG" \
    --gate-bank "$P6/gate_bank" --gate-bank-n 64 --gate-lab-json "$P6/lab.json" --gate-mem-metrics "$P6/mem_metrics.jsonl" \
    --gate-egr-json "$P6/egr/g_egr_result.json" > "$P6/launch_gate.log" 2>&1 200>&-
  echo "ZZEXIT $?" >> "$P6/launch_gate.log"
  token_ok || refuse "the launch gate did not mint a PASS token that verifies for this argv (launch_gate.log, lg/, token_verify.log)"
  mkdir -p "$P6/gate"; cp "$LG"/evidence_*.json "$LG"/PASS_*.json "$P6/gate/" 2> /dev/null      # visible on HF through the watcher (the HMAC key is not)
  stage_mark launch_gate
fi
token_ok || refuse "the launch-gate token no longer verifies (token_verify.log)"
log "launch gate PASS (token verified)"; stop_after launch_gate

# ---- 12. preflight on the REAL configuration, then the supervised run
token_ok || refuse "no launch-gate token verifies for this argv: the preflight and the trainer do not start"
env -u HF_TOKEN "$PY" train.py $ARGS --preflight > "$P6/preflight.log" 2>&1 200>&-
grep -q "^PREFLIGHT_OK" "$P6/preflight.log" || refuse "train.py --preflight did not print PREFLIGHT_OK"
log "train.py preflight OK; launching"
for att in $(seq 1 "${P6_MAX_ATTEMPTS:-12}"); do
  if [ -f $RUN/summary.json ] && grep -q '"done": true' $RUN/summary.json; then break; fi
  token_ok || refuse "the launch-gate token does not verify for this argv (attempt $att): train.py is NOT started"
  env -u HF_TOKEN "$PY" train.py $ARGS >> "$P6/train.log" 2>&1 200>&-
  log "attempt $att exited rc $?"
  sleep "${P6_ATTEMPT_SLEEP:-60}" 200>&-
done
if [ -f $RUN/summary.json ] && grep -q '"done": true' $RUN/summary.json; then log "ZZP6_TRAIN_DONE"
else log "ZZP6_FAILED (no done=true after the attempts)"; exit 1; fi

# ---- 13. the final push is the hard gate: success is the REMOTE listing (and model_final's sha256), not the push call
touch "$WSTOP"
for _w in $(seq 1 120); do kill -0 "$WPID" 2> /dev/null || break; sleep 5 200>&-; done
"$PY" p2r_sync.py final --repo "$REPO" --run "$RUN" --logs "$P6" --chain-log "$L" --outbox "$P6/outbox_final" > "$P6/final.log" 2>&1 200>&-
if [ $? -eq 0 ] && grep -q "^ZZP2R_FINAL_OK" "$P6/final.log"; then log "ZZP6_DONE"; exit 0; fi
log "ZZP6_FINAL_FAILED (final.log): the checkpoint is on the pod only until a retry succeeds"
exit 1
