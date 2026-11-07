#!/bin/bash
# P7 (SPEC_P7.md): P6's recipe + the P7.1 training-only drivable-area head + the session reserve out of every training input.  Derived from refe/p6_pod.sh; P6's own chain is NOT edited.
# Runbook: HANDOFF.md of the P7-LAUNCH-READY package.  Tests: refe/test_p7_chain.py.  Gate profile: refe_p7 (refe/launch_gate.py, pins P7-PINS 2026-10-23).
#
# Needs: HF_TOKEN in the environment (write-scoped, the PI provides it; read here, never echoed, never an argument, stripped from every child except the HF tools) and the package
#        (code/ refe/ splits/ eval/<what the suite reads> raw/<same> SHIP_MANIFEST.json) in $P7_PKG (default /workspace/refe-plan-p7).  ⛔ NOT P6's package: P6's token binds ITS tree.
# Start (detached, may be started BEFORE P6 ends with P7_WAIT_P6=1: it then waits, adding no load, for P6's ZZP6_DONE):
#   (P7_LABEL_WORKERS: the value the pod's G-LIVE-LAB passes at; 13 on the 15.3-CPU L40, where 6 REFUSED for P6 -- PREREG_P6 Amendment (e))
#   P7_LABEL_WORKERS=13 P7_WAIT_P6=1 nohup setsid bash /workspace/refe-plan-p7/refe/p7_pod.sh > /workspace/data/refe_p7/p7_stdout.log 2>&1 < /dev/null &
# Re-running after ANY interruption is safe: finished stages leave $P7/stage/<name>.ok (a stage whose result depends on the argv carries the argv hash).
# P7_STOP_AFTER=<env|weights|recover|inputs|reserve|yaw|cache|lanes|lab|wait|mem|egr|launch_gate> ends the chain after that stage (a cheap dry run).
#
# What P7 changes against p6_pod.sh, and why (every item is pinned by refe/test_p7_chain.py with a deliberate-regression arm):
#  * its OWN run dir ($P7_RUN), HF repo ($P7_REPO), chain dir ($DATA/refe_p7) and package; each REFUSES to equal P6's.
#  * the reserve stage (p6_pod.sh 5a) is MANDATORY here, not dormant; the bank is `<P7_CBANK>_p7` (train_grow_cmd_3c_p7, pins registered in launch_gate.py).
#  * the gate is `refe_p7` with --gate-reserve-json; the ARGS add the P7.1 head's levers (all named: --w-aux-drv defaults to 0.0).
#  * AMENDMENT YAW (SPEC_P7 section 7): stage 5b builds the MEASURED yaw-rate table from $DBS (`ego_yaw_table.py build`, no --allow-partial), checks it against the P7 bank at 100 % (`check-bank`), and the ARGS
#    carry `--ego-yaw-table`; the table's sha256 enters the launch token as a data flag.  A partial table or an uncovered bank token REFUSES.
#  * R67: the kept twins' parts are LINKED into the metric cache by the chain itself (p7_prep.py twins apply: 100 % coverage or STOP), THEN `p6_prep.py lane-distinct` must read OK at every rank
#    of P7_EXPECT_RANKS (1 2 3), BEFORE lab / mem / egr / the gate.  The chain NEVER builds a twin (no `build_metric_cache.py --lane-rank`): PDM-Closed does not consult the route patch, a build writes
#    twins on the LOGGED route.  P7_D7_FALLBACK is 0 and any other value REFUSES (a declared fallback is a silent deviation once its reason is gone).
#  * a `no_parts` guard on the first labeller events of EVERY attempt, and p7_pins.py --check-run after the first checkpoint: either one REFUSES, STOPS the trainer by its pid and ends the chain.
#  * --label-workers is PINNED by P7_LABEL_WORKERS (required; P6 ran 6 on the A40s, 13 on the L40), never derived from the CPU quota (a 15.3-CPU pod would derive 13 and change the argv).
#  * the run dir and the metric cache must not sit on a network / FUSE filesystem (the labeller stalled 141-430 s after every checkpoint save there).
set -u
WORK="${WORK:-/workspace}"; DATA=$WORK/data
PKG="${P7_PKG:-$WORK/refe-plan-p7}"
P2DIR=$DATA/refe_p2
P6DIR=$DATA/refe_p6; P6RUN=$DATA/refe_runs/p6_paper_faithful; P6REPO="Sayood/tanitad-refe-p6-paper-faithful"
P7=$DATA/refe_p7; S=$P7/stage; L=$P7/chain.log
RUN="${P7_RUN:-$DATA/refe_runs/p7_reserve_head}"
REPO="${P7_REPO:-Sayood/tanitad-refe-p7-reserve-head}"          # a NEW private model repo, created by the first push
BANK=$DATA/refe_navtrain/train_grow                  # the calibration table's source (the reserve stage filters a copy)
CBANK="${P7_CBANK:-$DATA/refe_navtrain/train_grow_cmd_3c}"   # the 3-C twin bank (the reserve stage writes <it>_p7); a path already ending _p7 is taken as filtered (a RESUME)
MC="${P7_MC:-$DATA/refe_navtrain/metric_cache}"      # the D9 parts
TWIN_SRC="${P7_TWIN_SRC:-$DATA/refe_navtrain/metric_cache_twins_3c}"   # the 44,409 correct twin parts (build_twin_parts_3c.sh); only the KEPT rows' of the P7 bank are linked
DBS=$DATA/navtrain_dbs; PIX=$DATA/navtrain_pixels
EXP_DBS="${P7_EXPECT_DBS:-1192}"; EXP_PIX="${P7_EXPECT_PIX_SHARDS:-32}"
P7_RESERVE="${P7_RESERVE:-$PKG/refe/fixtures/p7_reserve_sessions.json}"   # sha256-PINNED in session_reserve.py; any other byte REFUSES
YAWT="$DATA/refe_navtrain/ego_yaw_navtrain.json.gz"   # AMENDMENT YAW: the token -> ego_pose.angular_rate_z table, built by stage 5b from $DBS; never overridden (the token binds this file's bytes)
P7_LABEL_STORE="${P7_LABEL_STORE:-$DATA/refe_navtrain/aux_label_store_p7}"   # the FULL aux store (103,139 keys) on entry; split in place, re-finalised (92,993 keys) = the trainer's --aux-drv-labels
YAW_WRAP="${P7_YAW_WRAP-off}"                        # PI ruling P6-Y carried over (P7 = P6's recipe): `-` and not `:-`, a BLANKED variable stays blank and yaw_check refuses it
EXPECT_RANKS="${P7_EXPECT_RANKS:-1 2 3}"
WSTOP=$P7/sync.stop
export TANITAD_GATE_KEY_FILE="$P7/launch_gate.key"     # the HMAC key: persistent, OUTSIDE the shipped tree and anything the watcher pushes
mkdir -p "$P7" "$S"
log() { echo "$(date -u +%FT%T) $*" >> $L; }
refuse() { log "REFUSED: $*"; exit 1; }
stage_done() { [ -f "$S/$1.ok" ]; }
stage_mark() { date -u +%FT%T > "$S/$1.ok"; }
key_done() { [ "$(cat "$S/$1.ok" 2> /dev/null)" = "$2" ]; }
key_mark() { printf '%s' "$2" > "$S/$1.ok"; }
invalidate() { local s; for s in "$@"; do rm -f "$S/$s.ok"; done; }
stop_after() { if [ "${P7_STOP_AFTER:-}" = "$1" ]; then log "stopping after $1 (P7_STOP_AFTER)"; exit 0; fi; }
yaw_check() {
  case "$YAW_WRAP" in
    off) return 0 ;;
  esac
  refuse "PI ruling P6-Y (2026-10-11) is yaw-wrap = off, P7 keeps P6's recipe, but P7_YAW_WRAP is '${YAW_WRAP}': the chain will not guess the value of --yaw-wrap, and the refe_p7 profile refuses any other"
}
cgroup_cores() {
  local c q per
  c="$(nproc)"; CORES_SRC="nproc (UNVERIFIED: no cgroup quota file)"
  if { read -r q per < /sys/fs/cgroup/cpu.max; } 2> /dev/null && [ "$q" != "max" ] && [ "${per:-0}" -gt 0 ]; then
    c=$(( q / per > 0 ? q / per : 1 )); CORES_SRC="cgroup v2 cpu.max"
  elif { read -r q < /sys/fs/cgroup/cpu/cpu.cfs_quota_us; read -r per < /sys/fs/cgroup/cpu/cpu.cfs_period_us; } 2> /dev/null && [ "${q:-0}" -gt 0 ] && [ "${per:-0}" -gt 0 ]; then
    c=$(( q / per > 0 ? q / per : 1 )); CORES_SRC="cgroup v1 cfs"
  fi
  echo "$c|$CORES_SRC"
}
on_exit() {
  rc=$?
  touch "$WSTOP" 2>/dev/null
  if [ -n "${WPID:-}" ]; then
    for _w in $(seq 1 36); do kill -0 "$WPID" 2>/dev/null || break; sleep 5 200>&-; done
    kill -0 "$WPID" 2>/dev/null && kill "$WPID" 2>/dev/null
  fi
  if [ -n "${PY:-}" ] && [ -d "$PKG/refe" ]; then       # one last push, so a REFUSED line reaches HF too
    ( cd "$PKG/refe" && "$PY" p2r_sync.py tick --repo "$REPO" --run "$RUN" --logs "$P7" --chain-log "$L" --outbox "$P7/outbox_exit" ) >> "$P7/exit_push.log" 2>&1 200>&-
  fi
  exit $rc
}

exec 200>"$P7/chain.lock"
flock -n 200 || { echo "ZZP7_REFUSED another p7_pod.sh holds $P7/chain.lock"; exit 1; }
trap on_exit EXIT
log "chain start (p7_pod.sh md5 $(md5sum "$0" | cut -c1-12))"

# ---- who holds the pod?  P2, RL-S1, S1 and P6 must be OVER (a chain holds its flock for its whole life).  ⛔ a lock probe takes a READ-WRITE fd (`9<>`): an exclusive flock on a read-only fd is EBADF on nfs4.
lock_held() { [ -e "$1/chain.lock" ] && ! ( flock -n 9 ) 9<> "$1/chain.lock"; }
# ---- ISOLATION FROM P6: its own chain dir, run dir, HF repo and package.  P6's token binds P6's code tree; a shared run dir would let P7 resume (or overwrite) P6's checkpoints.
[ "$RUN" != "$P6RUN" ] && [ "${RUN#"$P6RUN"/}" = "$RUN" ] || refuse "P7_RUN ($RUN) is P6's run directory ($P6RUN): P7 has its own, a shared one would resume or overwrite P6's checkpoints"
[ "$REPO" != "$P6REPO" ] || refuse "P7_REPO ($REPO) is P6's HF repo: P7 pushes to a repo of its own"
[ "$P7" != "$P6DIR" ] || refuse "the chain directory is P6's"
[ "$PKG" != "$WORK/refe-plan-p6" ] || refuse "P7_PKG ($PKG) is P6's package directory: P6's token binds that tree"
case "${P7_D7_FALLBACK:-0}" in
  0) ;;
  *) refuse "P7_D7_FALLBACK='${P7_D7_FALLBACK}': the D7 fallback does not exist in the P7 chain (R67: a declared fallback set before the 3-C bank existed turned D7's correct refusal into a silent quarantine of 21,821 twins); a D7 failure is a defect to FIX" ;;
esac
# ---- --label-workers is PINNED by the environment, never derived from the CPU quota (PREREG_P6 Amendment (d) item 2: the chain would derive 13 from a 15.3-CPU quota and change the argv and its sha)
case "${P7_LABEL_WORKERS:-}" in
  ''|*[!0-9]*) refuse "P7_LABEL_WORKERS is not set to an integer (got '${P7_LABEL_WORKERS:-}'): the label-worker count is part of the argv sha and is never derived from the cgroup quota (P6 ran 6 on the A40s, 13 on the L40)" ;;
esac
LW="$P7_LABEL_WORKERS"; { [ "$LW" -ge 1 ] && [ "$LW" -le 16 ]; } || refuse "P7_LABEL_WORKERS=$LW is outside 1..16"
LC="${P7_LABEL_CHUNKS:-2}"
yaw_check
[ -n "${HF_TOKEN:-}" ] || refuse "HF_TOKEN is not set in the environment"
# ---- WAIT FOR P6 (optional, adds NO load): started before P6 ends, the chain waits for P6's lock to be free; it goes on ONLY if P6 logged ZZP6_DONE (a P6 that died is the Master Mind's, not a P7 launch)
if [ "${P7_WAIT_P6:-0}" = 1 ]; then
  _w=0
  while lock_held "$P6DIR"; do
    { [ "$_w" = 0 ] || [ $((_w % 600)) = 0 ]; } && log "ZZP7_WAITING_FOR_P6 the P6 chain still holds $P6DIR/chain.lock (waited ${_w}s)"
    [ "${P7_WAIT_P6_MAX_S:-0}" -gt 0 ] && [ "$_w" -ge "${P7_WAIT_P6_MAX_S:-0}" ] && refuse "waited ${_w}s for P6 (P7_WAIT_P6_MAX_S)"
    sleep "${P7_POLL_S:-30}" 200>&-; _w=$((_w + ${P7_POLL_S:-30}))
  done
  grep -q "ZZP6_DONE" "$P6DIR/chain.log" 2> /dev/null || refuse "the P6 chain released its lock but $P6DIR/chain.log has no ZZP6_DONE: P6 did not finish, P7 does not start on its pod (Master Mind)"
fi
for u in "$P2DIR" "$DATA/rls1" "$DATA/s1aux" "$P6DIR"; do
  if lock_held "$u"; then refuse "the chain at $u still holds its chain.lock: P7 starts only after it is over (P7_WAIT_P6=1 waits for P6)"; fi
done
# ---- 0. the package, BEFORE anything is downloaded: PKG-SHIP-COMPLETE.  P6 lost ~40 min of A40 because its package lacked four files its own G-SUITE reads.
PY0="${P7_BOOT_PYTHON:-python3}"
for f in refe/p7_ship_list.py refe/p7_ship_list.json SHIP_MANIFEST.json; do
  [ -f "$PKG/$f" ] || refuse "$PKG/$f missing: ship the package first (a directory of its own, NOT P6's)"
done
SHIPOUT="$("$PY0" "$PKG/refe/p7_ship_list.py" check --pkg "$PKG" --list "$PKG/refe/p7_ship_list.json" 200>&- 2>&1)" || refuse "the package is incomplete or stale: $(printf '%s' "$SHIPOUT" | tail -n 1 | cut -c1-400)"
log "ship list ok: $(printf '%s' "$SHIPOUT" | tail -n 1 | cut -c1-120)"
for f in refe/train.py refe/p2r_sync.py refe/hf_egress.py refe/p6_prep.py refe/p7_prep.py refe/p7_pins.py refe/session_reserve.py refe/aux_label_store.py refe/aux_drivable_head.py refe/online_labeller.py \
         refe/build_metric_cache.py refe/diag_p6_labeller_identity.py refe/launch_gate.py refe/gate_token.py refe/gate_checks.py refe/gate_fixture.py refe/test_launch_gate.py refe/test_p2r_chain.py \
         refe/test_p6_levers.py refe/test_p6_chain.py refe/test_p7_head_wiring.py refe/test_session_reserve.py refe/test_p7_integration.py refe/test_p7_chain.py \
         refe/ego_yaw_table.py refe/ego_yaw.py refe/test_ego_yaw.py code/pod_teacher_env.sh code/pod_train.sh code/pod_dataprep.sh splits/navtrain.yaml; do
  [ -f "$PKG/$f" ] || refuse "$PKG/$f missing: ship the package first"
done
[ -f "$P7_RESERVE" ] || refuse "P7_RESERVE=$P7_RESERVE is not a file"
[ -f "$P7_LABEL_STORE/manifest.json" ] || refuse "P7_LABEL_STORE=$P7_LABEL_STORE holds no manifest.json: the aux label store (103,139 keys, store_sha256 4a17db67) is NOT ON THE POD: fetch refe_aux_drivable_label_store_2026-10-23.tar (353,802,240 B) from the navtrain-banks repo (authenticated, failing on any HTTP error) and unpack it there first"
# ---- (e) the run dir and the metric cache are LOCAL disk (symlinks followed): MEASURED 2026-10-25, the labeller stalled 141-430 s after each checkpoint save while they lived on the network volume
for _d in "$RUN" "$MC"; do
  _o="$("$PY0" "$PKG/refe/p7_prep.py" local-check --path "$_d" ${P7_MOUNTS:+--mounts "$P7_MOUNTS"} 200>&- 2>&1)" || refuse "$_o"
  log "$_o"
done
# ⛔ never judge the disk with df: WRITE to it.
DDGB="${P7_DD_GB:-96}"
dd if=/dev/zero of="$DATA/.dd_test" bs=1M count=$((DDGB * 1024)) conv=fsync 2>> $L 200>&-
[ "$(stat -c %s "$DATA/.dd_test" 2>/dev/null)" = "$((DDGB * 1024 * 1024 * 1024))" ] || { rm -f "$DATA/.dd_test"; refuse "dd write test of ${DDGB} GB failed"; }
rm -f "$DATA/.dd_test"
log "dd write test ${DDGB} GB ok"

# ---- 1. env (the teacher env is REUSED when present)
if ! [ -f "$WORK/teacher_env.sh" ]; then
  env -u HF_TOKEN bash "$PKG/code/pod_teacher_env.sh" > "$P7/teacher_env.log" 2>&1 200>&-
  grep -q "^ZZOK teacher env ready" "$P7/teacher_env.log" || refuse "pod_teacher_env.sh did not finish (teacher_env.log)"
fi
source "$WORK/teacher_env.sh"
PY="$DRIVERL_EVAL_PYTHON"; export PKG
export REFE_BACKBONE_ROOT="$DATA/backbones" PYTHONUNBUFFERED=1
# R43: fragmentation alone can end a multi-day run; the gate, every instrument and the launch inherit it.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CG="$(cgroup_cores)"; CORES="${CG%%|*}"; CORES_SRC="${CG#*|}"
CW="${P7_CACHE_WORKERS:-$(( CORES < 1 ? 1 : (CORES > 24 ? 24 : CORES) ))}"
log "cores $CORES ($CORES_SRC): label workers $LW (PINNED by P7_LABEL_WORKERS, not derived), jobs per set $LC, cache workers $CW"
cd "$PKG/refe" || refuse "cannot cd $PKG/refe"
stage_mark env; log "env ok"; stop_after env

# ---- watcher: logs every P7_SYNC_MIN (30), checkpoints every P7_HEAVY_MIN (30), to P7's NEW private repo; a failed push never stops the run
rm -f "$WSTOP"
"$PY" p2r_sync.py watch --repo "$REPO" --run "$RUN" --logs "$P7" --chain-log "$L" --outbox "$P7/outbox" --stop-file "$WSTOP" \
  --interval-min "${P7_SYNC_MIN:-30}" --heavy-min "${P7_HEAVY_MIN:-30}" --tail "$P7/train.log:train.tail.log:400" >> "$P7/sync.log" 2>&1 200>&- &
WPID=$!

# ---- 2. DINOv3 weights (pinned commit + sha256; an existing file is HASHED and kept)
if ! stage_done weights; then
  env -u HF_TOKEN STAGES=weights WORK="$WORK" DATA="$DATA" PKG="$PKG" bash "$PKG/code/pod_train.sh" > "$P7/weights.log" 2>&1 200>&-
  grep -q "^ZZOK weights ZZ" "$P7/weights.log" || refuse "pod_train.sh weights stage did not print ZZOK weights"
  stage_mark weights
fi
log "weights ok"; stop_after weights

# ---- 3. RESUME FROM HF: a NEW pod picks up what a lost one pushed to P7's repo.  A first run: a no-op.
if ! stage_done recover; then
  "$PY" p2r_sync.py recover --work "$DATA" --run "$RUN" --repo "$REPO" > "$P7/recover.log" 2>&1 200>&-
  grep -q "^ZZRECOVER_OK" "$P7/recover.log" || refuse "recover failed: $(tail -n 1 "$P7/recover.log" | cut -c1-300)"
  stage_mark recover
fi
# ---- 3b. A RESUME MUST READ THE BANK THE RUN STARTED ON (MEASURED 2026-10-20): compares the PATH in <run>/config.json (args.targets) with this chain's CBANK; the trainer's own identity check stays the authority.
#          Called again where CBANK is rebound to the reserve-free bank and after ARGS.  No config.json = a fresh start.
resume_bank_check() {
  [ -e "$RUN/config.json" ] || [ -L "$RUN/config.json" ] || return 0
  local out rc=0 msg
  out="$("$PY" p6_prep.py resume-bank --config "$RUN/config.json" --bank "$CBANK" 200>&- 2>&1)" || rc=$?
  case "$rc" in
    0) log "resume bank ok: $out"; return 0 ;;
    3) msg="ZZP7_RESUME_BANK_MISMATCH ${out#ZZRESUME_BANK_MISMATCH } -- the run in $RUN was STARTED on another bank than this chain's CBANK; set P7_CBANK to the run's bank; no stage after this one starts" ;;
    4) msg="ZZP7_RESUME_BANK_UNREADABLE ${out#ZZRESUME_BANK_UNREADABLE } -- $RUN/config.json does not say which bank the run started on, so the resume is not checked; no stage after this one starts" ;;
    *) msg="ZZP7_RESUME_BANK_UNCHECKED rc $rc: $(printf '%s' "$out" | tail -n 1 | cut -c1-200) -- the bank check itself failed; no stage after this one starts" ;;
  esac
  echo "$msg"; refuse "$msg"
}
log "recover: $(grep '^ZZRECOVER_OK' "$P7/recover.log" | tail -n 1 | cut -c1-80)"; stop_after recover

# ---- 4. inputs: reused IN PLACE.  Each fetch exists ONLY inside a fetch_* function and is reached ONLY as `have_X || fetch_X` (test_p7_chain pins both).
have_bank() { [ -s "$BANK/calib_table.json" ] && ls "$BANK"/targets_rank*.jsonl > /dev/null 2>&1; }
have_dbs() { [ "$(ls "$DBS"/*.db 2> /dev/null | wc -l)" -ge "$EXP_DBS" ]; }
have_pix() { [ "$(ls "$PIX"/.done_* 2> /dev/null | wc -l)" -ge "$EXP_PIX" ]; }
fetch_bank() {
  if ! command -v zstd > /dev/null 2>&1 && ! "$PY" -c "import zstandard" 2> /dev/null 200>&-; then
    env -u HF_TOKEN apt-get install -y -q zstd >> "$P7/zstd_install.log" 2>&1 200>&- \
      || env -u HF_TOKEN uv pip install -q --python "$PY" --no-deps zstandard >> "$P7/zstd_install.log" 2>&1 200>&-
  fi
  "$PY" p2r_sync.py restore --work "$DATA" > "$P7/restore.log" 2>&1 200>&-
  grep -q "^ZZRESTORE_OK" "$P7/restore.log"
}
fetch_dbs() {
  env -u HF_TOKEN STAGES=dbs WORK="$WORK" DATA="$DATA" PKG="$PKG" bash "$PKG/code/pod_dataprep.sh" > "$P7/dbs.log" 2>&1 200>&-
  grep -q "^ZZOK dbs $EXP_DBS files ZZ" "$P7/dbs.log"
}
fetch_pix() {
  env -u HF_TOKEN STAGES=pix WORK="$WORK" DATA="$DATA" PKG="$PKG" bash "$PKG/code/pod_dataprep.sh" > "$P7/pix.log" 2>&1 200>&-
  [ "$(grep '^ZZOK pix [0-9]*/[0-9]* shards ZZ$' "$P7/pix.log" | tail -n 1)" = "ZZOK pix $EXP_PIX/$EXP_PIX shards ZZ" ]
}
have_bank || fetch_bank || refuse "the calibration source bank $BANK is absent and could not be restored (restore.log)"
have_dbs || fetch_dbs || refuse "navtrain DBs: want $EXP_DBS files (dbs.log)"
have_pix || fetch_pix || refuse "pixels incomplete: want $EXP_PIX/$EXP_PIX shards (pix.log; rerun to resume)"
[ "$(ls -d "${NUPLAN_MAPS_ROOT:-/nonexistent}"/*/ 2> /dev/null | wc -l)" -ge 4 ] || refuse "nuPlan maps: fewer than 4 city dirs under NUPLAN_MAPS_ROOT"
case "${CBANK##*/}" in *_p7) ;; *) ls "$CBANK"/targets_rank*.jsonl > /dev/null 2>&1 || refuse "the 3-C source bank $CBANK holds no targets_rank*.jsonl: P7 filters it, it does not build it" ;; esac
[ -d "$TWIN_SRC" ] || refuse "the twin-parts source $TWIN_SRC is absent (build_twin_parts_3c.sh): the kept twins' parts cannot be linked (R67), and the chain never builds a twin"
stage_mark inputs
log "inputs ok (calibration bank, $EXP_DBS DBs, $EXP_PIX pixel shards, maps, 3-C bank, twin source: reused in place)"; stop_after inputs

# ---- 5. THE SESSION RESERVE OUT OF EVERY LOG-KEYED TRAINING INPUT (SPEC_P7 2; refe/session_reserve.py; MANDATORY in P7).  (1) verify the reserve file (sha256-pinned); (2) rebind CBANK to `<bank>_p7` and BANK to a
#          calibration root of its own; (3) THE GUARD: the resume-bank check on the REBOUND CBANK before any file moves; (4) filter the bank (rank-0 AND twins, byte-for-byte rows) and the calibration table, MOVE the
#          reserve's dirs out of the D9 cache and its zips out of the image root (reversible: p6_prep.py reserve-move --restore), split + re-finalise the aux label store; (5) AUDIT all five inputs.
P7OUT="$("$PY" p6_prep.py reserve-verify --reserve "$P7_RESERVE" 200>&- 2>&1)" || refuse "P7 reserve file refused: $(printf '%s' "$P7OUT" | tail -n 1 | cut -c1-300)"
P7SHA="${P7OUT#*sha256=}"; P7SHA="${P7SHA%% *}"
P7D="$P7/p7_reserve"; mkdir -p "$P7D"
P7_SRC=""; P7_CALIB_SRC="$BANK/calib_table.json"
case "${CBANK##*/}" in
  *_p7) log "P7: $CBANK already names a filtered bank (_p7): verified by the audit, not re-filtered" ;;
  *) P7_SRC="$CBANK"; CBANK="${P7_SRC}_p7" ;;
esac
BANK="$P7D/calib"
resume_bank_check
p7_audit() {
  local co=""
  if [ -n "$(ls -A "$MC" 2> /dev/null | head -n 1)" ]; then co="--cache $MC"; fi
  "$PY" p6_prep.py reserve-audit --reserve "$P7_RESERVE" --bank "$CBANK" $co --images "$PIX" --calib "$BANK/calib_table.json" --label-store "$P7_LABEL_STORE" \
    --out "$P7D/audit.json" > "$P7D/audit.out" 2>&1 200>&-
  grep -q "^ZZRESERVE_AUDIT_OK " "$P7D/audit.out" || refuse "P7 reserve audit FAILED (a reserve key in an input, an unresolved log, or an empty input): $(tail -n 1 "$P7D/audit.out" | cut -c1-300)"
}
P7_ID="$P7SHA:${P7_SRC:-filtered}:${P7_LABEL_STORE}"
if [ -n "$P7_SRC" ] && ! { key_done reserve "$P7_ID" && ls "$CBANK"/targets_rank*.jsonl > /dev/null 2>&1 && [ -s "$BANK/calib_table.json" ]; }; then
  # the D9 cache and the image root are SHARED with P6: never move them from under a trainer that reads them (the top guard already refused a held P6 lock; kept as the move's own guard)
  if lock_held "$P6DIR"; then refuse "P7: the P6 chain at $P6DIR holds its chain.lock: its trainer reads the metric cache and pixels this stage moves"; fi
  rm -rf "$CBANK.tmp"
  nice -n 10 "$PY" p6_prep.py reserve-filter --reserve "$P7_RESERVE" --bank "$P7_SRC" --out "$CBANK.tmp" ${P7_TOKEN_TABLE:+--token-table "$P7_TOKEN_TABLE"} > "$P7D/filter.out" 2>&1 200>&-
  grep -q "^ZZRESERVE_FILTER_OK " "$P7D/filter.out" || refuse "P7 reserve: the bank filter refused: $(tail -n 1 "$P7D/filter.out" | cut -c1-300)"
  rm -rf "$CBANK"; mv "$CBANK.tmp" "$CBANK"
  mkdir -p "$BANK"
  "$PY" p6_prep.py reserve-calib --reserve "$P7_RESERVE" --calib "$P7_CALIB_SRC" --out "$BANK/calib_table.json" > "$P7D/calib.out" 2>&1 200>&-
  grep -q "^ZZRESERVE_CALIB_OK " "$P7D/calib.out" || refuse "P7 reserve: the calibration filter refused: $(tail -n 1 "$P7D/calib.out" | cut -c1-300)"
  if [ -d "$MC" ]; then
    "$PY" p6_prep.py reserve-move --reserve "$P7_RESERVE" --kind cache --root "$MC" --to "${MC}_reserve" > "$P7D/move_cache.out" 2>&1 200>&-
    grep -q "^ZZRESERVE_MOVE_OK " "$P7D/move_cache.out" || refuse "P7 reserve: moving the reserve out of the metric cache failed: $(tail -n 1 "$P7D/move_cache.out" | cut -c1-300)"
  fi
  "$PY" p6_prep.py reserve-move --reserve "$P7_RESERVE" --kind images --root "$PIX" --to "${PIX}_reserve" > "$P7D/move_images.out" 2>&1 200>&-
  grep -q "^ZZRESERVE_MOVE_OK " "$P7D/move_images.out" || refuse "P7 reserve: moving the reserve out of the image root failed: $(tail -n 1 "$P7D/move_images.out" | cut -c1-300)"
  "$PY" p6_prep.py reserve-labelstore --reserve "$P7_RESERVE" --store "$P7_LABEL_STORE" --reserve-store "${P7_LABEL_STORE}_reserve" > "$P7D/labelstore.out" 2>&1 200>&-
  grep -q "^ZZRESERVE_LABELSTORE_OK " "$P7D/labelstore.out" || refuse "P7 reserve: splitting the aux label store failed: $(tail -n 1 "$P7D/labelstore.out" | cut -c1-300)"
  "$PY" aux_label_store.py finalize --out "$P7_LABEL_STORE" > "$P7D/finalize_train.out" 2>&1 200>&- || refuse "P7 reserve: re-finalising the training label store failed"
  "$PY" aux_label_store.py finalize --out "${P7_LABEL_STORE}_reserve" > "$P7D/finalize_reserve.out" 2>&1 200>&- || refuse "P7 reserve: finalising the reserve label store failed"
  invalidate cache lanes lab mem egr launch_gate
  key_mark reserve "$P7_ID"
fi
p7_audit
log "P7 reserve ok: bank $CBANK ($(tail -n 1 "$P7D/filter.out" 2> /dev/null | cut -c1-160)); $(tail -n 1 "$P7D/audit.out" | cut -c1-160)"; stop_after reserve

# ---- 5b. AMENDMENT YAW (SPEC_P7 section 7; PI go 2026-11-06 "do 1 and 2"): the MEASURED yaw rate for ego slot 4.  The token -> ego_pose.angular_rate_z TABLE is built HERE, on the pod: the pod holds all $EXP_DBS navtrain DBs
#          ($DBS), the dev box holds 539 of the 1,192 (42.9 % of the tokens).  It runs AFTER the P6 wait (no CPU beside P6's trainer) and AFTER the reserve stage, so the bank it is checked against is the one the trainer READS
#          (CBANK, rebound to <bank>_p7 above).  (1) `ego_yaw_table.py build` over the navtrain split file: it REFUSES (rc 4, nothing written) when any asked token has no DB row, and --allow-partial is NEVER passed;
#          (2) `check-bank` against CBANK: every row's token in the table (100 %), or REFUSE; (3) the path enters ARGS as --ego-yaw-table, its sha256 enters the launch token's data manifest and the checkpoint meta.
#          Idempotent: a keyed marker (the builder's bytes + the split file + the DB count) skips the build; the coverage check runs on EVERY start (seconds), because the bank may have changed since.
YAW_ID="$(sha256sum "$PKG/refe/ego_yaw_table.py" "$PKG/splits/navtrain.yaml" | cut -c1-16 | tr -d '\n'):$EXP_DBS"
if ! { key_done yaw "$YAW_ID" && [ -s "$YAWT" ]; }; then
  rm -f "$YAWT" "$YAWT.report.json"
  env -u HF_TOKEN nice -n 10 "$PY" ego_yaw_table.py build --split navtrain --yaml "$PKG/splits/navtrain.yaml" --db-root "$DBS" --out "$YAWT" > "$P7/yaw_build.out" 2>&1 200>&-
  { grep -q "^ZZYAWTABLE_BUILT [1-9][0-9]* [0-9a-f]\{64\}$" "$P7/yaw_build.out" && [ -s "$YAWT" ]; } \
    || refuse "the yaw-rate table build refused or failed, no partial table is ever used (yaw_build.out: $(tail -n 1 "$P7/yaw_build.out" | cut -c1-250))"
  invalidate mem egr launch_gate
  key_mark yaw "$YAW_ID"
fi
env -u HF_TOKEN "$PY" ego_yaw_table.py check-bank --table "$YAWT" --bank-dir "$CBANK" > "$P7/yaw_check.out" 2>&1 200>&-; YRC=$?
{ [ "$YRC" = 0 ] && grep -q "^ZZYAWTABLE_BANK_PASS rows=[1-9][0-9]* distinct_tokens=[0-9]* missing_rows=0$" "$P7/yaw_check.out"; } \
  || refuse "the yaw-rate table does not cover 100 % of the P7 bank $CBANK (check-bank rc $YRC; yaw_check.out: $(tail -n 1 "$P7/yaw_check.out" | cut -c1-250))"
YAWSHA="$(sed -n 's/^ *"table_sha256": "\([0-9a-f]*\)".*/\1/p' "$P7/yaw_check.out" | head -n 1)"
[ "${#YAWSHA}" = 64 ] || refuse "check-bank did not report the table's sha256 (yaw_check.out)"
log "yaw table ok: $(tail -n 1 "$P7/yaw_check.out" | cut -c1-120), table sha256 ${YAWSHA:0:16}"; stop_after yaw

# ---- 6. D9: the metric-cache PARTS of the P7 bank's rows (the logged-route part of every row; existing files are SKIPPED, so after P6 this is a scan), then the EXACTNESS check on a sample
if ! { stage_done cache && [ -d "$MC" ]; }; then
  export NUPLAN_DATA_ROOT="$DATA" REFE_NUPLAN_DB_ROOT="$DBS" REFE_NAVTRAIN_YAML="$PKG/splits/navtrain.yaml"
  for att in 1 2 3; do
    ( echo 1000 > /proc/self/oom_score_adj 2> /dev/null
      exec env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK OMP_NUM_THREADS=1 nice -n 10 "$PY" build_metric_cache.py build --targets "$CBANK" --out "$MC" \
        --db-root "$DBS" --map-root "${NUPLAN_MAPS_ROOT:-}" --workers "$CW" ) > "$P7/cache_attempt$att.out" 2>&1 200>&-
    cat "$P7/cache_attempt$att.out" >> "$P7/cache.log"
    grep -q "^ZZMETRIC_CACHE_DONE .* failed=0 " "$P7/cache_attempt$att.out" && break
    log "cache attempt $att incomplete: $(tail -n 1 "$P7/cache_attempt$att.out" | cut -c1-200)"
  done
  grep -q "^ZZMETRIC_CACHE_DONE .* failed=0 " "$P7/cache_attempt$att.out" || refuse "metric cache incomplete after 3 attempts (cache.log)"
  TOKS="$("$PY" p6_prep.py sample-tokens --cache "$MC" --n "${P7_D9_VERIFY_TOKENS:-24}" 200>&- | sed -n 's/^ZZTOKENS //p')"
  [ -n "$TOKS" ] || refuse "the metric cache holds no parts after the build"
  env -u HF_TOKEN OMP_NUM_THREADS=1 "$PY" build_metric_cache.py verify --cache "$MC" --db-root "$DBS" --map-root "${NUPLAN_MAPS_ROOT:-}" --tokens "$TOKS" --n-sets 4 \
    > "$P7/d9_verify.out" 2>&1 200>&-
  grep -q "^ZZD9_PASS" "$P7/d9_verify.out" || refuse "D9: cached parts are NOT bit-identical to fresh parts (d9_verify.out)"
  invalidate lanes lab mem egr launch_gate
  stage_mark cache
fi
log "metric cache ok: $(grep -h '^ZZMETRIC_CACHE_DONE' "$P7"/cache_attempt*.out 2> /dev/null | tail -n 1 | cut -c1-120)"; stop_after cache

# ---- 7. D7 / POST_MERGE step (d) AS A CHAIN STAGE (R67).  (i) the twin ranks the P7 bank carries must be EXACTLY P7_EXPECT_RANKS (a bank whose twins vanished is not "nothing to build");
#          (ii) the kept twins' parts are LINKED into the metric cache, 100 % coverage or STOP: only the lane_rank rows of THIS (reserve-free) bank, never a reserve log's (p7_prep.py twins --reserve REFUSES one);
#          (iii) `p6_prep.py lane-distinct` must read OK at every rank.  NO twin is ever BUILT here, and there is NO fallback: either failure REFUSES.
if ! stage_done lanes; then
  RANKS="$("$PY" p6_prep.py lane-ranks --bank "$CBANK" 200>&- | sed -n 's/^ZZLANE_RANKS//p' | xargs)"
  [ "$RANKS" = "$EXPECT_RANKS" ] || refuse "D7: the P7 bank carries lane-rank twins '$RANKS', the registered set is '$EXPECT_RANKS' (P7_EXPECT_RANKS): a bank that lost its twins is a defect, not 'nothing to build'"
  "$PY" p7_prep.py twins plan --bank "$CBANK" --src "$TWIN_SRC" --dst "$MC" --reserve "$P7_RESERVE" > "$P7/twins_plan.out" 2>&1 200>&-
  grep -q "^ZZP7TWINS_OK " "$P7/twins_plan.out" || refuse "D7 step (d): the twin plan is not 100 % placeable: $(tail -n 1 "$P7/twins_plan.out" | cut -c1-300)"
  "$PY" p7_prep.py twins apply --bank "$CBANK" --src "$TWIN_SRC" --dst "$MC" --reserve "$P7_RESERVE" --how "${P7_TWIN_HOW:-auto}" > "$P7/twins_apply.out" 2>&1 200>&-
  grep -q "^ZZP7TWINS_OK " "$P7/twins_apply.out" || refuse "D7 step (d): placing the kept twins' parts failed (100 % coverage or STOP): $(tail -n 1 "$P7/twins_apply.out" | cut -c1-300)"
  log "D7 twins: $(tail -n 1 "$P7/twins_apply.out" | cut -c1-200)"
  for k in $RANKS; do
    "$PY" p6_prep.py lane-distinct --bank "$CBANK" --cache "$MC" --k "$k" --n 40 --min-frac "${P7_D7_MIN_DISTINCT:-0.9}" > "$P7/lanes_distinct_$k.out" 2>&1 200>&-
    grep -q "^ZZD7_DISTINCT_OK " "$P7/lanes_distinct_$k.out" || refuse "D7: rank-$k twins are NOT on their own route ($(tail -n 1 "$P7/lanes_distinct_$k.out" | cut -c1-120)): the twin parts in the metric cache are not the 3-C twins; there is no fallback in P7 (R67)"
    log "D7 rank $k: $(tail -n 1 "$P7/lanes_distinct_$k.out" | cut -c1-120)"
  done
  invalidate lab mem egr launch_gate
  stage_mark lanes
fi
log "lanes ok"; stop_after lanes

# ---- 7a. the audit again on the FINAL state of every input, right before ARGS: the cache and twin stages have written since 5, and the proof must be of what the trainer will read
p7_audit; log "P7 reserve audit (final): $(tail -n 1 "$P7D/audit.out" | cut -c1-160)"

# ---- 7b. THE TRAINING ARGUMENTS, defined ONCE and BEFORE every instrument and the launch gate: the token is minted over EXACTLY this string and verified against it before the preflight and before every launch.
#          P6's argv + the P7.1 head.  ⛔ --w-aux-drv DEFAULTS to 0.0 and --w-field to a non-zero-able default: both are NAMED (a default-disables lever is not a test of the lever).
yaw_check
ARGS="--backbone vitl16 --targets $CBANK --images $PIX --calib $BANK/calib_table.json --metric-cache $MC --scorer-sees-trunk --yaw-wrap $YAW_WRAP --nav-mode command --labeller online --label-workers $LW --label-chunks $LC --epochs 25 --epoch-unit scenes --batch 4 --accum 64 --lr 2e-4 --weight-decay 0.01 --amp bf16 --score-w 0.1 --aux-drv --w-aux-drv 0.1 --aux-drv-tap visual --aux-drv-depth 2 --aux-drv-feed off --aux-drv-labels $P7_LABEL_STORE --w-field 0.0 --ego-yaw-table $YAWT --tf32 --compile --workers 2 --log-every 1 --ckpt-every-min 20 --out $RUN --resume"
ARGS_SHA="$(printf '%s' "$ARGS" | sha256sum | cut -c1-16)"
log "ARGS sha $ARGS_SHA (--yaw-wrap $YAW_WRAP, --label-workers $LW, --label-chunks $LC)"
resume_bank_check
LG=$P7/lg
# ⛔ THE LAUNCH-GATE TOKEN (CLAUDE.md, PI 2026-09-26, BINDING): the trainer does not start unless this returns 0.  `launch_gate.py verify` RE-MEASURES the argv, the code tree and the data and checks the HMAC.
token_ok() {
  tok="$(ls "$LG"/PASS_*.json 2> /dev/null | head -n 1)"
  [ -n "$tok" ] || return 1
  env -u HF_TOKEN "$PY" "$PKG/refe/launch_gate.py" verify --profile refe_p7 --token "$tok" --args "$ARGS" --pkg "$PKG" --json "$P7/token_verify.json" >> "$P7/token_verify.log" 2>&1 200>&-
}
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

# ---- 8. G-LIVE-LAB's instrument: the ONLINE labeller == the OFFLINE one bit for bit (>= 64 sets), a stale proposal REFUSED, throughput at the LAUNCH worker count
if ! key_done lab "w$LW-c$LC"; then
  TOKS="$("$PY" p6_prep.py sample-tokens --cache "$MC" --n 128 200>&- | sed -n 's/^ZZTOKENS //p')"
  [ -n "$TOKS" ] || refuse "no parts to sample for the labeller instrument"
  ( echo 1000 > /proc/self/oom_score_adj 2> /dev/null
    exec env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK "$PY" diag_p6_labeller_identity.py --cache "$MC" --db-root "$DBS" --map-root "${NUPLAN_MAPS_ROOT:-}" --tokens "$TOKS" \
      --workers "$LW" --chunks "$LC" --n-sets "${P7_LAB_SETS:-256}" --out "$P7/lab.json" ) > "$P7/lab.log" 2>&1 200>&-
  grep -q "^ZZLAB_IDENTITY_PASS" "$P7/lab.log" || refuse "G-LIVE-LAB instrument: identity or the stale-proposal refusal failed (lab.log)"
  log "lab: $(grep '^ZZLAB_THROUGHPUT' "$P7/lab.log" | tail -n 1) sets/s at $LW workers (the gate judges it against 6.735)"
  invalidate launch_gate
  key_mark lab "w$LW-c$LC"
fi
log "lab ok"; stop_after lab

# ---- 8b. THE WAIT before the FIRST GPU STAGE (9, G-LIVE-MEM): no other chain holds the pod, and nvidia-smi shows no compute process on P7_IDLE_POLLS consecutive polls (an nvidia-smi that cannot answer counts as busy)
gpu_busy() {
  local nv="${P7_NVSMI:-nvidia-smi}" apps
  apps="$("$nv" --query-compute-apps=pid --format=csv,noheader 2> /dev/null 200>&-)" || return 0
  [ -n "$(echo "$apps" | tr -d '[:space:]')" ]
}
wait_clear() {
  local poll="${P7_POLL_S:-10}" need="${P7_IDLE_POLLS:-3}" max="${P7_WAIT_MAX_S:-0}" idle=0 waited=0 since=0 why rc u
  while :; do
    why=""; rc=0
    for u in "$P2DIR" "$DATA/rls1" "$DATA/s1aux" "$P6DIR"; do
      if lock_held "$u"; then why="the chain at $u still holds its chain.lock"; rc=2; break; fi
    done
    if [ "$rc" = 0 ] && gpu_busy; then why="a process is on the GPU (or nvidia-smi cannot say)"; rc=2; fi
    if [ "$rc" = 0 ]; then
      idle=$((idle + 1))
      [ "$idle" -ge "$need" ] && return 0
    else
      idle=0
      if [ "$since" = 0 ] || [ "$since" -ge 600 ]; then log "ZZP7_WAITING $why (waited ${waited}s)"; since=0; fi
    fi
    [ "$max" -gt 0 ] && [ "$waited" -ge "$max" ] && refuse "waited ${waited}s for the pod (P7_WAIT_MAX_S): $why"
    sleep "$poll" 200>&-
    waited=$((waited + poll)); since=$((since + poll))
  done
}
wait_clear
log "ZZP7_WAIT_OK (no other chain holds the pod, GPU idle): the GPU stages start"; stop_after wait

# ---- 9. G-LIVE-MEM's instrument: the REAL launch configuration (head included) for 200 micro-batches on this card; its ckpt_last.pt is G-EGR's probe.  An instrument run: scratch --out, no --resume.
if ! key_done mem "$ARGS_SHA"; then
  rm -rf "$P7/mem_run"
  MEM_ARGS="$("$PY" p6_prep.py mem-argv --args "$ARGS" --out "$P7/mem_run" 200>&- | sed -n 's/^ZZMEM_ARGV //p')"
  [ -n "$MEM_ARGS" ] || refuse "could not derive the memory-run argv"
  env -u HF_TOKEN "$PY" train.py $MEM_ARGS > "$P7/mem.log" 2>&1 200>&-
  grep -q "^TRAIN_DONE" "$P7/mem.log" || refuse "the G-LIVE-MEM run did not finish (mem.log)"
  { [ -s "$P7/mem_run/metrics.jsonl" ] && [ -s "$P7/mem_run/ckpt_last.pt" ]; } || refuse "the G-LIVE-MEM run left no metrics.jsonl / ckpt_last.pt"
  cp "$P7/mem_run/metrics.jsonl" "$P7/mem_metrics.jsonl"
  invalidate egr launch_gate
  key_mark mem "$ARGS_SHA"
fi
log "mem ok"; stop_after mem

# ---- 10. G-EGR's instrument: the gate JSON reaches P7's repo, a push to a repo the token cannot write is refused BY THE SERVER, and the FIRST checkpoint's local size == its LISTING size
if ! key_done egr "$ARGS_SHA"; then
  rm -rf "$P7/egr"
  "$PY" hf_egress.py gate --repo "$REPO" --out "$P7/egr" --ckpt "$P7/mem_run/ckpt_last.pt" --chain-log "$L" > "$P7/egr.log" 2>&1 200>&-
  "$PY" p6_prep.py egr-ok --json "$P7/egr/g_egr_result.json" > "$P7/egr_ok.out" 2>&1 200>&-
  grep -q "^ZZEGR_OK" "$P7/egr_ok.out" || refuse "G-EGR instrument failed: $(tail -n 1 "$P7/egr_ok.out" | cut -c1-200) (egr.log)"
  invalidate launch_gate
  key_mark egr "$ARGS_SHA"
fi
log "egr ok"; stop_after egr

# ---- 11. THE LAUNCH GATE (G-HYG G-DVB G-LIVE G-CLOCK G-EVAL G-CKPT G-SUITE G-LIVE-LAB G-LIVE-MEM G-EGR G-RESERVE), profile refe_p7, on the REAL argv and the REAL data.  A PASS token or a REFUSAL; the token is then VERIFIED.
if ! stage_done launch_gate; then
  rm -rf "$LG" "$P7/gate_bank"
  env -u HF_TOKEN OMP_NUM_THREADS=4 "$PY" launch_gate.py run --profile refe_p7 --pkg "$PKG" --args "$ARGS" --out-dir "$LG" \
    --gate-bank "$P7/gate_bank" --gate-bank-n 64 --gate-lab-json "$P7/lab.json" --gate-mem-metrics "$P7/mem_metrics.jsonl" \
    --gate-egr-json "$P7/egr/g_egr_result.json" --gate-reserve-json "$P7_RESERVE" > "$P7/launch_gate.log" 2>&1 200>&-
  echo "ZZEXIT $?" >> "$P7/launch_gate.log"
  token_ok || refuse "the launch gate did not mint a PASS token that verifies for this argv (launch_gate.log, lg/, token_verify.log)"
  mkdir -p "$P7/gate"; cp "$LG"/evidence_*.json "$LG"/PASS_*.json "$P7/gate/" 2> /dev/null
  stage_mark launch_gate
fi
token_ok || refuse "the launch-gate token no longer verifies (token_verify.log)"
log "launch gate PASS (token verified)"; stop_after launch_gate

# ---- 12. preflight on the REAL configuration, then the supervised run with its two GUARDS.
token_ok || refuse "no launch-gate token verifies for this argv: the preflight and the trainer do not start"
env -u HF_TOKEN "$PY" train.py $ARGS --preflight > "$P7/preflight.log" 2>&1 200>&-
grep -q "^PREFLIGHT_OK" "$P7/preflight.log" || refuse "train.py --preflight did not print PREFLIGHT_OK"
log "train.py preflight OK; launching"
# the GUARDS, one background process per attempt.  Both STOP the trainer by its pid (never `pkill -f`: it self-matches) and leave a refusal marker the loop below turns into a REFUSED chain end.
#   * p7_pins.py --check-run <run> after the first checkpoint: the trainer's OWN recorded identity (targets_sha256, n_tuples) must equal the registered pins of its bank.  Closes the window the token leaves
#     (it fingerprints a directory by its listing: a same-size edit of the bank between G-CLOCK and the trainer's start).
#   * p7_prep.py noparts on the first labeller events of THIS attempt (R67): no_parts / sets above P7_NOPARTS_MAX_FRAC refuses.
kill_pid() {
  local p="$1" i
  kill -0 "$p" 2> /dev/null || return 0
  kill "$p" 2> /dev/null
  for i in $(seq 1 "${P7_KILL_WAIT_S:-30}"); do kill -0 "$p" 2> /dev/null || return 0; sleep 1 200>&-; done
  kill -9 "$p" 2> /dev/null
  return 0
}
guards() {
  local pid="$1" skip="$2" pins_done=0 np_done=0 g
  while kill -0 "$pid" 2> /dev/null; do
    if [ "$pins_done" = 0 ] && [ -f "$RUN/ckpt_last.pt" ]; then
      if "$PY" p7_pins.py --check-run "$RUN" > "$P7/pins_check.out" 2>&1 200>&-; then
        pins_done=1; log "ZZP7_PINS_OK $(tail -n 1 "$P7/pins_check.out" | cut -c1-160)"
      else
        : > "$P7/pins.refused"; log "ZZP7_PINS_REFUSED $(tail -n 1 "$P7/pins_check.out" | cut -c1-300): the trainer is stopped (pid $pid)"; kill_pid "$pid"; return 0
      fi
    fi
    if [ "$np_done" = 0 ]; then
      "$PY" p7_prep.py noparts --run "$RUN" --skip "$skip" --events "${P7_NOPARTS_EVENTS:-20}" --max-frac "${P7_NOPARTS_MAX_FRAC:-0.001}" > "$P7/noparts.out" 2>&1 200>&-; g=$?
      case "$g" in
        0) np_done=1; log "ZZP7_NOPARTS_OK $(tail -n 1 "$P7/noparts.out" | cut -c1-160)" ;;
        2) ;;
        *) : > "$P7/noparts.refused"; log "ZZP7_NOPARTS_REFUSED $(tail -n 1 "$P7/noparts.out" | cut -c1-300): the trainer is stopped (pid $pid)"; kill_pid "$pid"; return 0 ;;
      esac
    fi
    [ "$pins_done$np_done" = 11 ] && return 0
    sleep "${P7_GUARD_POLL_S:-20}" 200>&-
  done
  return 0
}
rm -f "$P7/pins.refused" "$P7/noparts.refused"
for att in $(seq 1 "${P7_MAX_ATTEMPTS:-12}"); do
  if [ -f $RUN/summary.json ] && grep -q '"done": true' $RUN/summary.json; then break; fi
  token_ok || refuse "the launch-gate token does not verify for this argv (attempt $att): train.py is NOT started"
  NP_SKIP="$("$PY" p7_prep.py noparts --run "$RUN" --count-only 200>&- | sed -n 's/^ZZP7NOPARTS_COUNT //p')"
  env -u HF_TOKEN "$PY" train.py $ARGS >> "$P7/train.log" 2>&1 200>&- &
  TPID=$!
  guards "$TPID" "${NP_SKIP:-0}" 200>&- &
  GPID=$!
  wait "$TPID"; trc=$?
  log "attempt $att exited rc $trc"
  wait "$GPID" 2> /dev/null
  [ ! -f "$P7/pins.refused" ] || refuse "ZZP7_PINS_REFUSED: the trainer's recorded identity is not the registered pins of its bank (pins_check.out); the trainer was stopped, no further attempt"
  [ ! -f "$P7/noparts.refused" ] || refuse "ZZP7_NOPARTS_REFUSED: scorer targets are missing for the first labeller events (noparts.out); the trainer was stopped, no further attempt (R67)"
  sleep "${P7_ATTEMPT_SLEEP:-60}" 200>&-
done
if [ -f $RUN/summary.json ] && grep -q '"done": true' $RUN/summary.json; then log "ZZP7_TRAIN_DONE"
else log "ZZP7_FAILED (no done=true after the attempts)"; exit 1; fi

# ---- 13. the final push is the hard gate: success is the REMOTE listing (and model_final's sha256), not the push call
touch "$WSTOP"
for _w in $(seq 1 120); do kill -0 "$WPID" 2> /dev/null || break; sleep 5 200>&-; done
"$PY" p2r_sync.py final --repo "$REPO" --run "$RUN" --logs "$P7" --chain-log "$L" --outbox "$P7/outbox_final" > "$P7/final.log" 2>&1 200>&-
if [ $? -eq 0 ] && grep -q "^ZZP2R_FINAL_OK" "$P7/final.log"; then log "ZZP7_DONE"; exit 0; fi
log "ZZP7_FINAL_FAILED (final.log): the checkpoint is on the pod only until a retry succeeds"
exit 1
