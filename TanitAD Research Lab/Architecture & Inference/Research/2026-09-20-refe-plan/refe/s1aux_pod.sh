#!/bin/bash
# S1 of the auxiliary drivable-area head on the A40 pod, AFTER RL-S1 and BEFORE P6 (PI order: P2 -> RL-S1 -> S1 -> P6): precheck (waits for the GPU) -> u400 -> held-out data and labels ->
# relabel (held-out PDM, train teacher lane) -> nine scorer fine-tunes (arms N, H1, H2 x seeds 0, 1, 2), each behind its own launch-gate PASS token -> a PRIVATE HF repo.
# Runbook: eval/S1_POD_RUNBOOK.md.  Spec: eval/PREREG_AUX_DRIVABLE_S1.md (+ Amendment A2); the exact argv: raw/2026-10-13-aux-head/PREREG_BLANKS_PROPOSED.md section 3; gate profile refe_aux_s1.
#
# Needs: HF_TOKEN in the environment (write-scoped; read here, never echoed, never an argument, stripped from every child except the HF ones) and the package shipped to
# /workspace/refe-plan-s1aux (a directory of its OWN: RL-S1 / P6 re-measure their own trees and adding files to those would break them), and the aux label store at
# /workspace/data/aux_drivable/label_store (352 MB; the chain verifies it by the store's own sha check).
# Start (detached):  nohup setsid bash /workspace/refe-plan-s1aux/refe/s1aux_pod.sh > /workspace/data/s1aux/stdout.log 2>&1 < /dev/null &
# Re-running after ANY interruption is safe: finished stages leave /workspace/data/s1aux/stage/<name>.ok, inputs are re-verified by content on every start.  An arm interrupted in TRAINING
# restarts at update 0 under a fresh preflight and a fresh token (the trainer has no resume; PREREG_AUX_DRIVABLE_S1 section 4 rule 1): its directory is renamed <ARM>_killed_<k>, never stitched.
# S1AUX_STOP_AFTER=<precheck|env|fetch|data|relabel|bank_labels|gate_<ARM>|arm_<ARM>> ends the chain after that stage (a dry run; gate_N-s0 stops after the first token).
# S1AUX_ARMS="N-s0 H2-s0 ..." selects the arms (default: all nine, in the REGISTERED priority order; a list must be a subsequence of it).  The registered drop of H1-s1 / H1-s2 is the Master Mind's.
# S1AUX_RLS1_ABANDONED=1 lets the chain start when RL-S1's chain never logged ZZRLS1_DONE (a killed arm, a refusal): only on the Master Mind's word.
# S1AUX_TLANE_N (default 15000) = the size of the seeded train teacher-lane subset (SFT-4R's: seed 20261004).  The same seed gives a PREFIX of the same shuffled order, so 2500 is a subset of 15000.
# The chain WAITS (it never starts beside RL-S1, P2 or any process on the GPU) and it re-checks the GPU right before every arm.  Every child is launched with fd 200 closed (the chain holds its flock
# on 200).  Markers are opaque ZZS1AUX_* tokens; every stop is `ZZS1AUX_REFUSED stage=<stage>: <why>` in chain.log.
set -u
WORK=/workspace; PKG=$WORK/refe-plan-s1aux; DATA=$WORK/data
P2=$DATA/refe_p2; RLS1=$DATA/rls1                  # their chain.lock / chain.log decide "RL-S1 is over"
R=$DATA/s1aux; S=$R/stage; L=$R/chain.log; STOPF=$R/chain.stop
OUT=$R/runs; LGROOT=$R/lg; MANIFEST=$R/arms_manifest.jsonl
OUT_REPO=Sayood/tanitad-refe-s1aux                 # PRIVATE; created by the first push (hf_egress.py push: create_repo exist_ok, private, verified private)
BANKS_REPO=Sayood/tanitad-refe-navtrain-banks      # private dataset: the held-out bank and the queue
U400_REPO=Sayood/tanitad-refe-rls1; U400_PATH=base/model_sft4_B_u400.pt
U400=$R/base/model_sft4_B_u400.pt
U400_SIZE=1288489838; U400_MD5=8e382eebb35b3e155862453617e5786e; U400_SHA256=8b4dba4818b5cc96c640fe892efe114037ace7cec9dda060c5b4cdd5fb92410d
D=$DATA/refe_sft2                                  # the SFT-4 label root: train_teacher_lane heldout_teacher_lane heldout_lane train_pdm heldout_pdm
TARGETS=$DATA/refe_navtrain/train_grow; IMAGES=$DATA/navtrain_pixels; CALIB=$TARGETS/calib_table.json
ONP=$DATA/refe_navtrain/onpolicy/sets              # P2's restored bank AFTER its carve (the 24 held-out logs were moved out)
HO_ROOT=$DATA/refe_heldout; HO=$HO_ROOT/sets; HOLOGS=$HO_ROOT/heldout_logs.txt
HO_TAR=$DATA/hf_dl/heldout/refe_heldout_v5.tar.zst; HO_TAR_PATH=refe_heldout_v5.tar.zst; HO_TAR_MD5=1f43409ba8ea2a8909e5035236c73676
QUEUE_TAR=$DATA/hf_dl/queue/onpolicy_queue.tar.zst; QUEUE_TAR_PATH=onpolicy_queue.tar.zst; QUEUE_TAR_SIZE=1817869216; QDIR=$R/queue_src
LABELS=$DATA/aux_drivable/label_store              # store_sha256 4a17db67...f287a, 103,139 frames (pinned in aux_drivable_head.py, gate_checks.py and the trainer's data event)
LANE_XZ=$PKG/raw/2026-10-04-lane-discipline        # the held-out lane labels SFT-4 trained against (3,137 rows each), restored from the repo (sft4r_relabel.sh step 1)
HELDOUT_SETS_N=3137; TLANE_MIN_ROWS=2109; TLANE_SEED=20261004; EXP_DBS=1192; EXP_PIX=32
ARM_ORDER="N-s0 H2-s0 H1-s0 N-s1 H2-s1 N-s2 H2-s2 H1-s1 H1-s2"      # PREREG_BLANKS_PROPOSED section 2, the registered priority order
ARMS="${S1AUX_ARMS:-$ARM_ORDER}"
# the launch-gate HMAC key lives on the persistent volume but OUTSIDE the shipped code tree and outside anything the egress pushes
export TANITAD_GATE_KEY_FILE="$R/launch_gate.key"
mkdir -p "$R" "$S"
STAGE=start
log() { echo "$(date -u +%FT%T) $*" >> $L; }
refuse() { log "ZZS1AUX_REFUSED stage=$STAGE: $*"; exit 1; }
stage_done() { [ -f "$S/$1.ok" ]; }
stage_mark() { date -u +%FT%T > "$S/$1.ok"; }
stop_after() { if [ "${S1AUX_STOP_AFTER:-}" = "$1" ]; then log "stopping after $1 (S1AUX_STOP_AFTER)"; exit 0; fi; }

# cores the cgroup grants (MEASURED 2026-10-11 on the P2 pod: cgroup v1, cfs 1105000/100000 = 11 cores while nproc said 96)
cgroup_cores() {
  local root="${S1AUX_CG_ROOT:-/sys/fs/cgroup}" q per n
  n="$(nproc)"
  if { read -r q per < "$root/cpu.max"; } 2> /dev/null && [ "$q" != "max" ] && [ "${per:-0}" -gt 0 ]; then
    n=$(( q / per > 0 ? q / per : 1 ))
  elif { read -r q < "$root/cpu/cpu.cfs_quota_us"; read -r per < "$root/cpu/cpu.cfs_period_us"; } 2> /dev/null && [ "${q:-0}" -gt 0 ] && [ "${per:-0}" -gt 0 ]; then
    n=$(( q / per > 0 ? q / per : 1 ))
  fi
  echo "$n"
}
# rc 0 = a process is on the GPU, OR the GPU cannot be asked (what cannot be proven idle is busy).  Never pgrep -f: it matches its own command line.
gpu_busy() {
  local nv="${S1AUX_NVSMI:-nvidia-smi}" apps
  apps="$("$nv" --query-compute-apps=pid --format=csv,noheader 2> /dev/null)" || return 0
  [ -n "$(echo "$apps" | tr -d '[:space:]')" ]
}
# rc 0 = nothing upstream holds the pod; rc 2 = a chain still runs (WAIT); rc 1 = it never will be over (REFUSE).  PI order: P2 -> RL-S1 -> S1 -> P6.
upstream() {
  if [ -e "$P2/chain.lock" ] && ! ( flock -n 9 ) 9< "$P2/chain.lock"; then echo "the P2 chain still holds $P2/chain.lock"; return 2; fi
  if [ -e "$RLS1/chain.lock" ] && ! ( flock -n 9 ) 9< "$RLS1/chain.lock"; then echo "the RL-S1 chain still holds $RLS1/chain.lock"; return 2; fi
  if ! grep -q "ZZRLS1_DONE" "$RLS1/chain.log" 2> /dev/null && [ "${S1AUX_RLS1_ABANDONED:-}" != "1" ]; then
    echo "RL-S1 is neither running nor finished: ZZRLS1_DONE is not in $RLS1/chain.log (S1AUX_RLS1_ABANDONED=1 only on the Master Mind's word)"; return 1; fi
  return 0
}
# BLOCK until upstream is over AND the GPU was idle on S1AUX_IDLE_POLLS consecutive polls: S1 never adds load to a pod that is training (CLAUDE.md), and an RL-S1 gap between two arms is not "free".
wait_clear() {
  local poll="${S1AUX_POLL_S:-30}" need="${S1AUX_IDLE_POLLS:-3}" max="${S1AUX_WAIT_MAX_S:-0}" idle=0 waited=0 since=0 why rc
  while :; do
    why="$(upstream)"; rc=$?
    [ "$rc" = 1 ] && refuse "$why"
    if [ "$rc" = 0 ] && gpu_busy; then why="a process is on the GPU (or nvidia-smi cannot say)"; rc=2; fi
    if [ "$rc" = 0 ]; then
      idle=$((idle + 1))
      [ "$idle" -ge "$need" ] && return 0
    else
      idle=0
      if [ "$since" = 0 ] || [ "$since" -ge 600 ]; then log "ZZS1AUX_WAITING $why (waited ${waited}s)"; since=0; fi
    fi
    [ "$max" -gt 0 ] && [ "$waited" -ge "$max" ] && refuse "waited ${waited}s for the pod (S1AUX_WAIT_MAX_S): $why"
    sleep "$poll" 200>&-
    waited=$((waited + poll)); since=$((since + poll))
  done
}
# size + md5 + sha256 of a pinned file.  rc 0 iff the file has exactly these.
file_pin_ok() {
  [ -f "$1" ] || return 1
  [ "$(stat -c %s "$1" 2> /dev/null)" = "$2" ] || return 1
  [ "$(md5sum "$1" | cut -c1-32)" = "$3" ] || return 1
  [ "$(sha256sum "$1" | cut -c1-64)" = "$4" ]
}
# arms: every one registered, none twice, in the registered relative order
arms_ok() {
  local a o i pos=0 found seen=" "
  [ -n "$(echo "$ARMS" | tr -d '[:space:]')" ] || { echo "S1AUX_ARMS is empty"; return 1; }
  for a in $ARMS; do
    case "$seen" in *" $a "*) echo "arm $a twice"; return 1 ;; esac
    seen="$seen$a "; i=0; found=0
    for o in $ARM_ORDER; do i=$((i + 1)); [ "$o" = "$a" ] && found=$i; done
    [ "$found" -gt 0 ] || { echo "unknown arm '$a' (one of: $ARM_ORDER)"; return 1; }
    [ "$found" -gt "$pos" ] || { echo "arm $a is out of the registered priority order ($ARM_ORDER)"; return 1; }
    pos=$found
  done
}
# THE argv of one arm: PREREG_BLANKS_PROPOSED section 3, character for character.  N = the SFT-4 recipe on u400 for 400 updates; H1 = N + the four head flags; H2 = H1 + --aux-drv-feed on.
# --warmup 50 --batch 8 --accum 8 are the parser defaults and are NOT named (the gate checks the effective values).  --dump-scores is on EVERY arm, N included.
arm_args() {
  local arm="$1" cls seed ad="$OUT/$1" base
  case "$arm" in N-s0|N-s1|N-s2|H1-s0|H1-s1|H1-s2|H2-s0|H2-s1|H2-s2) ;; *) return 1 ;; esac
  cls="${arm%%-*}"; seed="${arm##*-s}"
  base="--ckpt $U400 --targets $TARGETS --images $IMAGES --calib $CALIB --onpolicy $ONP --heldout $HO --heldout-logs $HOLOGS --workers 3 --lr 1e-4 --b-mode expreward --lam 1.0 --tau-s 0.1 --lane-head --lane-weight 1.0 --teacher-lane-train $D/train_teacher_lane --teacher-lane-heldout $D/heldout_teacher_lane --lane-labels-heldout $D/heldout_lane --pdm-labels-train $D/train_pdm --pdm-labels-heldout $D/heldout_pdm --require-pdm --max-updates 400 --eval-every 400 --dump-scores $ad/dump --seed $seed --out $ad"
  case "$cls" in
    N) echo "$base" ;;
    H1) echo "$base --aux-drv --w-aux-drv 0.1 --aux-drv-lr 2e-4 --aux-drv-labels $LABELS" ;;
    H2) echo "$base --aux-drv --w-aux-drv 0.1 --aux-drv-lr 2e-4 --aux-drv-labels $LABELS --aux-drv-feed on" ;;
  esac
}
# ⛔ THE LAUNCH-GATE TOKEN (CLAUDE.md, PI 2026-09-26, BINDING): scorer_finetune.py does not start unless this returns 0.  `launch_gate.py verify` RE-MEASURES the argv,
# the code tree and the data and checks the HMAC; it is never a grep of a log.   token_ok  -> rc 0 iff ZZGATE-MATCHZZ
token_ok() {
  tok="$(ls "$LG"/PASS_*.json 2> /dev/null | head -n 1)"
  [ -n "$tok" ] || return 1
  env -u HF_TOKEN "$PY" "$PKG/refe/launch_gate.py" verify --profile refe_aux_s1 --token "$tok" --args "$ARGS" --pkg "$PKG" \
    --json "$ARMDIR/token_verify.json" >> "$ARMDIR/token_verify.log" 2>&1 200>&-
}
# push files to the PRIVATE repo, then prove them by the REMOTE listing: hf_egress.py's size check (ZZEGRESS_OK <tag>) AND the content (LFS sha256 / git blob id), never the push's return code.
# bank_push <tag> <prefix> <local>::<name-under-prefix> (a double colon: a Windows path has a single one)...
bank_push() {
  local tag="$1" prefix="$2" n0 spec files="" names=""
  shift 2
  for spec in "$@"; do files="$files ${spec%%::*}"; names="$names ${spec#*::}"; done
  n0=$(wc -l < $L)           # only THIS push's lines may vouch for it (chain.log is appended across attempts)
  "$PY" hf_egress.py push --repo "$OUT_REPO" --prefix "$prefix" --tag "$tag" --chain-log "$L" --egress-log "$R/egress.jsonl" --max-push-bytes 12e9 \
    --files $files --names $names >> "$R/push_$tag.log" 2>&1 200>&-
  tail -n +$((n0 + 1)) $L | grep -q " ZZEGRESS_OK $tag " || return 1
  "$PY" s1aux_checks.py remote-verify --repo "$OUT_REPO" --prefix "$prefix" --files $files --names $names > "$R/remote_$tag.log" 2>&1 200>&-
  grep -q "^ZZS1AUX_REMOTE_OK" "$R/remote_$tag.log"
}
on_exit() {
  rc=$?
  touch "$STOPF" 2> /dev/null
  [ -n "${TICKPID:-}" ] && kill "$TICKPID" 2> /dev/null
  if [ -n "${PY:-}" ] && [ -d "$PKG/refe" ] && [ -n "${HF_TOKEN:-}" ] && [ "${S1AUX_NO_EXIT_PUSH:-}" != "1" ]; then       # one last push, so a REFUSED line reaches HF too
    ( cd "$PKG/refe" && "$PY" hf_egress.py push --repo "$OUT_REPO" --prefix logs --tag chain_exit --files "$L" --chain-log "$R/egress_exit.log" ) \
      >> "$R/exit_push.log" 2>&1 200>&-
  fi
  exit $rc
}

exec 200>"$R/chain.lock"
flock -n 200 || { echo "ZZS1AUX_REFUSED another s1aux_pod.sh holds $R/chain.lock"; exit 1; }
trap on_exit EXIT
rm -f "$STOPF"
log "ZZS1AUX_START chain (s1aux_pod.sh md5 $(md5sum "$0" | cut -c1-12), arms: $ARMS)"

# ---- 0. precheck: cheap refusals FIRST, then the wait, then the disk
STAGE=precheck
[ -n "${HF_TOKEN:-}" ] || refuse "HF_TOKEN is not set in the environment"
for f in refe/scorer_finetune.py refe/aux_drivable_head.py refe/aux_label_store.py refe/field_loss.py refe/train.py refe/model.py refe/ckpt_io.py refe/planner.py refe/navsim_dac.py \
         refe/launch_gate.py refe/gate_token.py refe/gate_checks.py refe/gate_fixture.py refe/hf_egress.py refe/p2r_sync.py refe/s1aux_checks.py refe/s1aux_pod.sh \
         refe/onpolicy_relabel_pdm.py refe/pdm_label_complete.py refe/onpolicy_relabel_teacher_lane.py refe/select_bank_subset.py refe/onpolicy_label.py refe/navsim_pdm_targets.py \
         refe/navtrain_scenarios.py refe/scorer_gate.py refe/score_proposals.py refe/route_rank.py refe/map_feature_cache.py refe/query_cache.py refe/build_scorer_targets.py \
         refe/test_rl_reward.py refe/test_rl_epo.py refe/test_aux_drivable_head.py refe/test_aux_s1_wiring.py refe/test_launch_gate.py \
         code/pod_teacher_env.sh code/pod_dataprep.sh splits/navtrain.yaml SHIP_MANIFEST.json \
         raw/2026-10-04-lane-discipline/heldout_navsim_lane_labels.jsonl.xz raw/2026-10-04-lane-discipline/heldout_teacher_lane_labels.jsonl.xz \
         raw/2026-10-11-aux-drivable/head_proto_selftest.out; do
  [ -f "$PKG/$f" ] || refuse "$PKG/$f missing: ship code/ refe/ splits/ raw/ SHIP_MANIFEST.json to $PKG first (a directory of its own, NOT the one RL-S1 or P6 run from)"
done
[ -f "$WORK/teacher_env.sh" ] || refuse "$WORK/teacher_env.sh is missing: P2 / RL-S1 built the teacher env on this pod; run code/pod_teacher_env.sh first (S1 never builds an env beside a training job)"
source "$WORK/teacher_env.sh"
PY="${DRIVERL_EVAL_PYTHON:-}"; export PKG
[ -n "$PY" ] || refuse "teacher_env.sh did not set DRIVERL_EVAL_PYTHON"
cd "$PKG/refe" || refuse "cannot cd $PKG/refe"
# the shipped tree must be the manifest's, file for file (the launch gate re-measures the same tree before every arm: a drift is caught HERE, not after the wait)
env -u HF_TOKEN "$PY" s1aux_checks.py tree --pkg "$PKG" > "$R/tree_check.log" 2>&1 200>&-
grep -q "^ZZS1AUX_TREE_OK" "$R/tree_check.log" || refuse "the shipped code tree is not its SHIP_MANIFEST.json: $(tail -n 1 "$R/tree_check.log" | cut -c1-400)"
bad="$(arms_ok)" || refuse "$bad"
wait_clear
# ⛔ never judge the disk with df (a pod volume's quota is invisible to it): WRITE to it.  Nine runs leave about 24 GB (model_sft_A.pt + model_sft_B.pt, 1.29 GB each, per arm) plus the inputs.
DDMB="${S1AUX_DD_MB:-24576}"
dd if=/dev/zero of="$DATA/.dd_test_s1" bs=1M count="$DDMB" conv=fsync 2>> "$R/dd_test.log" 200>&-
[ "$(stat -c %s "$DATA/.dd_test_s1" 2>/dev/null)" = "$((DDMB * 1024 * 1024))" ] || { rm -f "$DATA/.dd_test_s1"; refuse "dd write test of ${DDMB} MB failed"; }
rm -f "$DATA/.dd_test_s1"
log "ZZS1AUX_PRECHECK_OK (tree $(tail -n 1 "$R/tree_check.log" | cut -c1-80); upstream over, GPU idle; dd ${DDMB} MB)"; stop_after precheck

# ---- 1. env (P2's teacher env is REUSED: it holds the NAVSIM stack, the maps and the DriveRL fork)
STAGE=env
export REFE_BACKBONE_ROOT="$DATA/backbones" PYTHONUNBUFFERED=1 AUX_LABEL_STORE="$LABELS"        # AUX_LABEL_STORE: the gate's G-SUITE runs test_aux_drivable_head.py, which reads the store there
[ -d "$REFE_BACKBONE_ROOT" ] || refuse "$REFE_BACKBONE_ROOT is missing: P2's weights stage did not run on this pod"
CORES="$(cgroup_cores)"
LW="${S1AUX_LABEL_WORKERS:-$(( CORES - 1 < 1 ? 1 : (CORES - 1 > 16 ? 16 : CORES - 1) ))}"
# a ticker pushes chain.log every 30 min, so the hours of labelling and training are not invisible
if [ "${S1AUX_TICK:-1}" = 1 ]; then
  ( while [ ! -f "$STOPF" ]; do sleep 1800 200>&-; [ -f "$STOPF" ] && break
      "$PY" hf_egress.py push --repo "$OUT_REPO" --prefix logs --tag chain_tick --files "$L" --chain-log "$R/egress_tick.log" 200>&-; done ) >> "$R/ticker.log" 2>&1 200>&- &
  TICKPID=$!
fi
log "ZZS1AUX_ENV_OK (cgroup cores $CORES, label workers $LW)"; stop_after env

# ---- 2. fetch: u400 (HF $U400_REPO $U400_PATH), verified by size + md5 + sha256 against the pin.  A wrong file is DELETED: the argv's --ckpt points at that path.
STAGE=fetch
if ! file_pin_ok "$U400" "$U400_SIZE" "$U400_MD5" "$U400_SHA256"; then
  log "u400 is not on the pod (or does not verify): fetching $U400_REPO $U400_PATH"
  rm -f "$U400"; mkdir -p "$R/base"
  "$PY" s1aux_checks.py fetch --repo "$U400_REPO" --path "$U400_PATH" --local-dir "$R" > "$R/fetch_u400.log" 2>&1 200>&-
  if ! file_pin_ok "$U400" "$U400_SIZE" "$U400_MD5" "$U400_SHA256"; then
    got="$(stat -c %s "$U400" 2> /dev/null) bytes md5 $(md5sum "$U400" 2> /dev/null | cut -c1-12)"
    rm -f "$U400"
    refuse "u400 (model_sft4_B_u400.pt) is not the pinned file (want $U400_SIZE bytes md5 ${U400_MD5:0:12}; got $got; fetch: $(tail -n 1 "$R/fetch_u400.log" | cut -c1-200))"
  fi
fi
log "ZZS1AUX_FETCH_OK u400 $U400_SIZE bytes md5 ${U400_MD5:0:12}"; stop_after fetch

# ---- 3. data: what the pod holds is CHECKED; what it lacks is restored (held-out bank from HF, held-out lane labels from the repo) or refused (the aux label store: shipped)
STAGE=data
need_zstd() {
  if ! command -v zstd > /dev/null 2>&1 && ! "$PY" -c "import zstandard" 2> /dev/null 200>&-; then
    env -u HF_TOKEN apt-get install -y -q zstd >> "$R/zstd_install.log" 2>&1 200>&- \
      || env -u HF_TOKEN uv pip install -q --python "$PY" --no-deps zstandard >> "$R/zstd_install.log" 2>&1 200>&-
  fi
}
heldout_present() { [ "$(ls "$HO"/onpolicy_*.jsonl 2> /dev/null | wc -l)" = 7 ] && [ "$(wc -w < "$HOLOGS" 2> /dev/null)" = 24 ]; }
for p in "$CALIB" "$ONP" "$IMAGES" "$TARGETS"; do [ -e "$p" ] || refuse "$p is missing: the pod's P2 inputs are gone (audit: it held them)"; done
ls "$TARGETS"/targets_rank*.jsonl > /dev/null 2>&1 || refuse "no $TARGETS/targets_rank*.jsonl"
ls "$ONP"/onpolicy_*.jsonl > /dev/null 2>&1 || refuse "no $ONP/onpolicy_*.jsonl"
[ "$(ls -d "$IMAGES"/navtrain_current_*/ 2> /dev/null | wc -l)" -ge "$EXP_PIX" ] || refuse "fewer than $EXP_PIX navtrain_current_* pixel shard dirs under $IMAGES"
ls "$D"/train_pdm/pdm_*.jsonl > /dev/null 2>&1 || refuse "$D/train_pdm has no pdm_*.jsonl: P2's PDM relabel output is gone"
if ! heldout_present; then
  if ! [ -f "$HO_TAR" ] || [ "$(md5sum "$HO_TAR" | cut -c1-32)" != "$HO_TAR_MD5" ]; then
    log "held-out bank tarball absent or not the pinned file: fetching $BANKS_REPO $HO_TAR_PATH"
    rm -f "$HO_TAR"; mkdir -p "$(dirname "$HO_TAR")"
    "$PY" s1aux_checks.py fetch --repo "$BANKS_REPO" --repo-type dataset --path "$HO_TAR_PATH" --local-dir "$(dirname "$HO_TAR")" > "$R/fetch_heldout.log" 2>&1 200>&-
    [ -f "$HO_TAR" ] && [ "$(md5sum "$HO_TAR" | cut -c1-32)" = "$HO_TAR_MD5" ] \
      || refuse "$HO_TAR is not the pinned refe_heldout_v5.tar.zst (md5 ${HO_TAR_MD5:0:12}); fetch: $(tail -n 1 "$R/fetch_heldout.log" | cut -c1-200)"
  fi
  need_zstd
  env -u HF_TOKEN "$PY" s1aux_checks.py extract --archive "$HO_TAR" --dest "$DATA" > "$R/extract_heldout.log" 2>&1 200>&-
  grep -q "^ZZS1AUX_EXTRACT_OK" "$R/extract_heldout.log" || refuse "the held-out bank did not extract: $(tail -n 1 "$R/extract_heldout.log" | cut -c1-250)"
  heldout_present || refuse "after extraction $HO has not 7 onpolicy_*.jsonl and $HOLOGS not 24 logs (the tarball's layout is not refe_heldout/{sets,heldout_logs.txt})"
fi
for pair in "heldout_navsim_lane_labels heldout_lane lane_heldout_restored" "heldout_teacher_lane_labels heldout_teacher_lane teacher_lane_r0_restored"; do
  set -- $pair
  mkdir -p "$D/$2"
  if ! ls "$D/$2"/*.jsonl > /dev/null 2>&1; then
    xz -dc "$LANE_XZ/$1.jsonl.xz" > "$D/$2/$3.jsonl.part" && mv "$D/$2/$3.jsonl.part" "$D/$2/$3.jsonl" || refuse "cannot restore $2 from $LANE_XZ/$1.jsonl.xz"
    log "$2 restored from the repo: $(wc -l < "$D/$2/$3.jsonl") rows"
  fi
  [ "$(cat "$D/$2"/*.jsonl | wc -l)" = "$HELDOUT_SETS_N" ] || refuse "$D/$2 holds $(cat "$D/$2"/*.jsonl | wc -l) rows, want $HELDOUT_SETS_N (the labels SFT-4 trained against)"
done
[ -e "$LABELS/manifest.json" ] || refuse "the aux label store is absent: $LABELS (ship the 352 MB store there; eval/S1_POD_RUNBOOK.md section 2)"
env -u HF_TOKEN "$PY" s1aux_checks.py store --store "$LABELS" > "$R/store_check.log" 2>&1 200>&-
grep -q "^ZZS1AUX_STORE_OK" "$R/store_check.log" || refuse "the aux label store fails its own sha check: $(tail -n 1 "$R/store_check.log" | cut -c1-300)"
log "ZZS1AUX_DATA_OK (held-out bank $(ls "$HO" | wc -l) files, 24 logs; held-out lane + teacher lane $HELDOUT_SETS_N rows each; $(tail -n 1 "$R/store_check.log" | cut -c1-120))"; stop_after data

# ---- 4. relabel.  The labellers must be ABLE to label before hours of them (2026-10-11: navsim was missing from the venv and the relabel failed EVERY set).
STAGE=relabel
export NUPLAN_DATA_ROOT="$DATA" REFE_NUPLAN_DB_ROOT="$DATA/navtrain_dbs" REFE_NAVTRAIN_YAML="$PKG/splits/navtrain.yaml"
if [ "$(ls "$DATA"/navtrain_dbs/*.db 2> /dev/null | wc -l)" -lt "$EXP_DBS" ]; then
  env -u HF_TOKEN STAGES=dbs WORK="$WORK" DATA="$DATA" PKG="$PKG" bash "$PKG/code/pod_dataprep.sh" > "$R/dbs.log" 2>&1 200>&-
  grep -q "^ZZOK dbs $EXP_DBS files ZZ" "$R/dbs.log" || refuse "navtrain DBs: want $EXP_DBS files (dbs.log)"
fi
env -u HF_TOKEN "$PY" -c "import navsim_pdm_targets as PT; PT._setup(); import onpolicy_relabel_teacher_lane; print('ZZPDM_SETUP_OK')" > "$R/pdm_setup.log" 2>&1 200>&-
grep -q "^ZZPDM_SETUP_OK" "$R/pdm_setup.log" || refuse "the labellers cannot start: $(tail -n 1 "$R/pdm_setup.log" | cut -c1-200)"
# ---- 4a. heldout_pdm: recomputed by the pod's own relabel path (p2r_pod.sh stage 6, --served-only: one label per served set), complete or refused (SFT-4R's guard: --expect 3137)
if ! stage_done pdm_heldout; then
  for att in 1 2 3; do
    ( echo 1000 > /proc/self/oom_score_adj 2> /dev/null
      exec env -u HF_TOKEN OMP_NUM_THREADS=1 nice -n 19 "$PY" onpolicy_relabel_pdm.py --sets "$HO" --out "$D/heldout_pdm" --served-only --workers "$LW" ) >> "$R/pdm_heldout.log" 2>&1 200>&-
    echo "ZZPDM_HELDOUT_EXIT $?" >> "$R/pdm_heldout.log"
    env -u HF_TOKEN "$PY" pdm_label_complete.py --log "$R/pdm_heldout.log" --out "$D/heldout_pdm" --sets "$HO" --expect "$HELDOUT_SETS_N" --json "$R/pdm_complete_heldout.json" \
      > "$R/pdm_complete_heldout.out" 2>&1 200>&-
    grep -q "^ZZPDM_COMPLETE" "$R/pdm_complete_heldout.out" && break
    log "held-out PDM relabel attempt $att incomplete: $(tail -n 1 "$R/pdm_complete_heldout.out" | cut -c1-200)"
  done
  grep -q "^ZZPDM_COMPLETE" "$R/pdm_complete_heldout.out" || refuse "held-out PDM labels incomplete after 3 attempts (pdm_complete_heldout.out)"
  stage_mark pdm_heldout
fi
NPH=$(cat "$D"/heldout_pdm/*.jsonl 2> /dev/null | wc -l)
[ "$NPH" -ge "$HELDOUT_SETS_N" ] || refuse "heldout_pdm holds $NPH rows, want >= $HELDOUT_SETS_N (with --require-pdm an empty dir leaves the held-out index EMPTY and every evaluation reads n = 0)"
log "held-out PDM complete: $NPH rows"
# ---- 4b. train_teacher_lane: the seeded subset (select_bank_subset.py, seed $TLANE_SEED) of the sets the trainer serves, relabelled on both ranks.  NAMED GAP (PREREG_SFT4R): SFT-4's original queue was never
#          banked; LANE_QUEUE_SRC is the HF banks' onpolicy_queue (the registered candidate).  A regenerated subset need not equal SFT-4's; the chain requires >= $TLANE_MIN_ROWS rows.
Q=$D/train_subset_queue; O=$D/train_teacher_lane; TN="${S1AUX_TLANE_N:-15000}"
if ! stage_done tlane_train; then
  if ! ls "$Q"/props_r*_subset.jsonl > /dev/null 2>&1; then
    if [ -n "${LANE_QUEUE_SRC:-}" ]; then
      [ -d "$LANE_QUEUE_SRC" ] || refuse "LANE_QUEUE_SRC=$LANE_QUEUE_SRC is not a directory"
    else
      "$PY" s1aux_checks.py fetch --repo "$BANKS_REPO" --repo-type dataset --path MD5SUMS.txt --local-dir "$DATA/hf_dl/queue" > "$R/fetch_md5sums.log" 2>&1 200>&-
      qmd5="$(awk -v n="$QUEUE_TAR_PATH" '($2 == n || $2 == "*" n) { print $1 }' "$DATA/hf_dl/queue/MD5SUMS.txt" 2> /dev/null | head -n 1)"
      [ ${#qmd5} = 32 ] || refuse "MD5SUMS.txt of $BANKS_REPO has no md5 for $QUEUE_TAR_PATH ($(tail -n 1 "$R/fetch_md5sums.log" | cut -c1-200)): the queue tarball cannot be verified"
      if ! [ -f "$QUEUE_TAR" ] || [ "$(stat -c %s "$QUEUE_TAR")" != "$QUEUE_TAR_SIZE" ] || [ "$(md5sum "$QUEUE_TAR" | cut -c1-32)" != "$qmd5" ]; then
        rm -f "$QUEUE_TAR"
        "$PY" s1aux_checks.py fetch --repo "$BANKS_REPO" --repo-type dataset --path "$QUEUE_TAR_PATH" --local-dir "$(dirname "$QUEUE_TAR")" > "$R/fetch_queue.log" 2>&1 200>&-
        [ -f "$QUEUE_TAR" ] && [ "$(stat -c %s "$QUEUE_TAR")" = "$QUEUE_TAR_SIZE" ] && [ "$(md5sum "$QUEUE_TAR" | cut -c1-32)" = "$qmd5" ] \
          || refuse "$QUEUE_TAR is not the banked onpolicy_queue.tar.zst ($QUEUE_TAR_SIZE bytes, md5 ${qmd5:0:12}): $(tail -n 1 "$R/fetch_queue.log" | cut -c1-200)"
      fi
      need_zstd
      env -u HF_TOKEN "$PY" s1aux_checks.py extract --archive "$QUEUE_TAR" --dest "$QDIR" > "$R/extract_queue.log" 2>&1 200>&-
      grep -q "^ZZS1AUX_EXTRACT_OK" "$R/extract_queue.log" || refuse "the queue did not extract: $(tail -n 1 "$R/extract_queue.log" | cut -c1-250)"
      qd="$(find "$QDIR" -name 'props_*.jsonl*' 2> /dev/null | sed 's|/[^/]*$||' | sort -u)"
      [ -n "$qd" ] && [ "$(echo "$qd" | wc -l)" = 1 ] || refuse "the extracted queue has props_*.jsonl* in $(echo "$qd" | wc -l) directories under $QDIR, want exactly 1 (set LANE_QUEUE_SRC=<dir> to name it)"
      LANE_QUEUE_SRC="$qd"
    fi
    env -u HF_TOKEN OMP_NUM_THREADS=1 "$PY" select_bank_subset.py --sets "$ONP" --queue "$LANE_QUEUE_SRC" --heldout-logs "$HOLOGS" --n "$TN" --seed "$TLANE_SEED" --out "$Q" \
      > "$R/subset.log" 2>&1 200>&-
    found="$(sed -n 's/^ZZSUBSET served [0-9]* picked [0-9]* queue_rows_found \([0-9]*\)$/\1/p' "$R/subset.log" | tail -n 1)"
    [ -n "$found" ] && [ "$found" -ge "$TLANE_MIN_ROWS" ] || { rm -rf "$Q"; refuse "the queue source covers $found of the picked subset ($TN), want >= $TLANE_MIN_ROWS rows ($(tail -n 1 "$R/subset.log" | cut -c1-200))"; }
    log "subset: $(grep ZZSUBSET "$R/subset.log" | tail -n 1) (queue $LANE_QUEUE_SRC)"
  fi
  W0=$(( (LW + 1) / 2 )); W1=$(( LW / 2 < 1 ? 1 : LW / 2 ))
  ( echo 1000 > /proc/self/oom_score_adj 2> /dev/null
    exec env -u HF_TOKEN OMP_NUM_THREADS=1 nice -n 19 "$PY" onpolicy_relabel_teacher_lane.py --queue "$Q" --files 'props_r0_*' --rank 0 --out "$O" --workers "$W0" --repair-last-heading ) >> "$R/tl_train_r0.log" 2>&1 200>&- &
  TP0=$!
  ( echo 1000 > /proc/self/oom_score_adj 2> /dev/null
    exec env -u HF_TOKEN OMP_NUM_THREADS=1 nice -n 19 "$PY" onpolicy_relabel_teacher_lane.py --queue "$Q" --files 'props_r1_*' --rank 1 --out "$O" --workers "$W1" --repair-last-heading ) >> "$R/tl_train_r1.log" 2>&1 200>&- &
  TP1=$!
  wait $TP0; TR0=$?; wait $TP1; TR1=$?
  echo "ZZLANE_TRAIN_EXIT $TR0 $TR1" >> "$R/tl_train.log"
  log "teacher-lane relabel: rc $TR0 $TR1, rows $(cat "$O"/*.jsonl 2> /dev/null | wc -l)"
  for rk in 0 1; do
    tail -n 1 "$R/tl_train_r$rk.log" | grep -q '^ZZTEACHER_LANE_DONE [0-9]* [0-9]*$' || refuse "the rank-$rk teacher-lane labeller did not run to its end (tl_train_r$rk.log; a re-run resumes)"
  done
  [ "$(cat "$O"/*.jsonl 2> /dev/null | wc -l)" -ge "$TLANE_MIN_ROWS" ] || refuse "train teacher-lane rows $(cat "$O"/*.jsonl 2> /dev/null | wc -l) < $TLANE_MIN_ROWS (SFT-4's count at launch): $O would train a lane head on almost no labels"
  stage_mark tlane_train
fi
NTL=$(cat "$O"/*.jsonl 2> /dev/null | wc -l)
[ "$NTL" -ge "$TLANE_MIN_ROWS" ] || refuse "train teacher-lane rows $NTL < $TLANE_MIN_ROWS (SFT-4's count at launch): $O would train a lane head on almost no labels"
for d in heldout_lane heldout_teacher_lane; do [ "$(cat "$D/$d"/*.jsonl 2> /dev/null | wc -l)" -gt 0 ] || refuse "$D/$d is empty"; done
log "ZZS1AUX_RELABEL_OK (held-out PDM $NPH rows; train teacher lane $NTL rows; held-out lane $(cat "$D"/heldout_lane/*.jsonl | wc -l), teacher $(cat "$D"/heldout_teacher_lane/*.jsonl | wc -l))"; stop_after relabel

# ---- 4c. bank the regenerated labels OFF the pod before any arm trains (they cost CPU-hours and were lost once)
STAGE=bank_labels
if ! stage_done bank_labels; then
  mkdir -p "$R/outbox"
  for d in heldout_pdm train_teacher_lane heldout_lane heldout_teacher_lane; do
    tar --force-local -C "$D" -cf "$R/outbox/s1_$d.tar" "$d" || refuse "tar of $d failed"
  done
  ( cd "$R/outbox" && sha256sum s1_*.tar > MANIFEST_s1_labels_sha256.txt )
  bank_push s1_labels labels "$R"/outbox/s1_heldout_pdm.tar::s1_heldout_pdm.tar "$R"/outbox/s1_train_teacher_lane.tar::s1_train_teacher_lane.tar \
    "$R"/outbox/s1_heldout_lane.tar::s1_heldout_lane.tar "$R"/outbox/s1_heldout_teacher_lane.tar::s1_heldout_teacher_lane.tar "$R"/outbox/MANIFEST_s1_labels_sha256.txt::MANIFEST_s1_labels_sha256.txt \
    "$R"/pdm_complete_heldout.json::pdm_complete_heldout.json "$Q"/subset_keys.json::subset_keys.json \
    || refuse "the regenerated labels were not banked to HF (push_s1_labels.log, remote_s1_labels.log): they are on the pod ONLY"
  stage_mark bank_labels
fi
log "ZZS1AUX_LABELS_BANKED"; stop_after bank_labels

# ---- 5. the arms, one at a time, in the registered order: preflight -> data check -> gate (PASS token) -> manifest row -> wait for the GPU -> train -> run check -> egress.
export OMP_NUM_THREADS=4
for ARM in $ARMS; do
  STAGE="arm_$ARM"
  stage_done "arm_$ARM" && { log "arm $ARM already done"; continue; }
  ARMDIR=$OUT/$ARM; LG=$LGROOT/$ARM; ARGS="$(arm_args "$ARM")" || refuse "no argv for arm $ARM"
  # a previous attempt that STARTED training and did not finish is archived whole: no resume, a fresh preflight, a FRESH token, the old directory kept and never stitched
  if stage_done "trainstart_$ARM" && ! stage_done "train_$ARM"; then
    k=1; while [ -e "$OUT/${ARM}_killed_$k" ]; do k=$((k + 1)); done
    mv "$ARMDIR" "$OUT/${ARM}_killed_$k" || refuse "cannot archive the killed attempt $ARMDIR"
    [ -d "$LG" ] && mv "$LG" "$LGROOT/${ARM}_killed_$k"
    rm -f "$S/pre_$ARM.ok" "$S/gate_$ARM.ok" "$S/trainstart_$ARM.ok" "$S/token_row_$ARM.ok"
    env -u HF_TOKEN "$PY" s1aux_checks.py manifest-row --arm "$ARM" --event restart --k "$k" --out "$MANIFEST" --note "attempt killed before update 400; directory kept as ${ARM}_killed_$k" > /dev/null 2>&1 200>&-
    log "ZZS1AUX_ARM_RESTART $ARM k=$k (the killed attempt is kept in ${ARM}_killed_$k, never stitched; a fresh preflight and a fresh token follow)"
    bank_push "arm_${ARM}_killed_$k" "arms/${ARM}_killed_$k" "$OUT/${ARM}_killed_$k/sft.log::sft.log" "$OUT/${ARM}_killed_$k/sft.jsonl::sft.jsonl" > /dev/null 2>&1 \
      || log "ZZS1AUX_WARN the logs of the killed attempt ${ARM}_killed_$k were not banked to HF"
  fi
  mkdir -p "$ARMDIR" "$LG"
  # ---- 5a. preflight on the REAL configuration (G1-G11, and G12 on a head arm); the gate CONSUMES its preflight.jsonl, so it comes first
  if ! stage_done "pre_$ARM"; then
    rm -f "$ARMDIR/preflight.jsonl" "$ARMDIR/PREFLIGHT_PASS.json"
    env -u HF_TOKEN "$PY" scorer_finetune.py $ARGS --preflight > "$ARMDIR/preflight.log" 2>&1 200>&-
    grep -q "^ZZSFT_PREFLIGHT_PASS" "$ARMDIR/preflight.log" || refuse "scorer_finetune.py --preflight did not pass for $ARM (preflight.log): $(tail -n 2 "$ARMDIR/preflight.log" | tr '\n' ' ' | cut -c1-250)"
    env -u HF_TOKEN "$PY" s1aux_checks.py data-check --jsonl "$ARMDIR/preflight.jsonl" > "$ARMDIR/data_check.log" 2>&1 200>&-
    grep -q "^ZZS1AUX_DATA_OK" "$ARMDIR/data_check.log" || refuse "the data of $ARM is not the audited data: $(tail -n 1 "$ARMDIR/data_check.log" | cut -c1-400)"
    stage_mark "pre_$ARM"
  fi
  # ---- 5b. the launch gate, for THIS arm: a PASS token or a refusal; the token is then VERIFIED (not grepped)
  if ! stage_done "gate_$ARM"; then
    rm -rf "$LG"; mkdir -p "$LG"
    env -u HF_TOKEN OMP_NUM_THREADS=4 "$PY" launch_gate.py run --profile refe_aux_s1 --arm-name "$ARM" --pkg "$PKG" --args "$ARGS" --out-dir "$LG" \
      --gate-preflight-json "$ARMDIR/preflight.jsonl" > "$ARMDIR/launch_gate.log" 2>&1 200>&-
    echo "ZZEXIT $?" >> "$ARMDIR/launch_gate.log"
    token_ok || refuse "the launch gate did not mint a PASS token that verifies for $ARM (launch_gate.log, lg/, token_verify.log)"
    stage_mark "gate_$ARM"
  fi
  token_ok || refuse "the launch-gate token of $ARM no longer verifies (token_verify.log)"
  # PREREG_AUX_DRIVABLE_S1 section 2: the token id is written to the arms manifest BEFORE the arm's first step
  if ! stage_done "token_row_$ARM"; then
    tok="$(ls "$LG"/PASS_*.json | head -n 1)"
    env -u HF_TOKEN "$PY" s1aux_checks.py manifest-row --arm "$ARM" --event token_verified --token "$tok" --out "$MANIFEST" > "$ARMDIR/manifest_row.log" 2>&1 200>&- \
      && grep -q "^ZZS1AUX_MANIFEST_OK" "$ARMDIR/manifest_row.log" || refuse "the token row of $ARM was not written to $MANIFEST"
    stage_mark "token_row_$ARM"
  fi
  log "ZZS1AUX_ARM_GATE_OK $ARM (preflight PASS, data audited, launch gate PASS, token verified and recorded)"; stop_after "gate_$ARM"
  # ---- 5c. train: S1 never trains beside anything (the GPU is re-checked right before every arm), no resume
  if ! stage_done "train_$ARM"; then
    wait_clear
    token_ok || refuse "the launch-gate token does not verify for $ARM: scorer_finetune.py is NOT started"
    stage_mark "trainstart_$ARM"
    log "ZZS1AUX_ARM_TRAIN_START $ARM"
    ( env -u HF_TOKEN OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 "$PY" scorer_finetune.py $ARGS >> "$ARMDIR/sft.log" 2>&1 ) 200>&-
    TRC=$?
    log "arm $ARM: trainer rc $TRC"
    if [ "$TRC" -ne 0 ] || ! grep -q "^ZZSFT_DONE" "$ARMDIR/sft.log"; then
      refuse "arm $ARM: the trainer did not finish (rc $TRC; tail of sft.log: $(tail -n 1 "$ARMDIR/sft.log" | cut -c1-200)); a re-run restarts it at update 0 under a fresh token"
    fi
    stage_mark "train_$ARM"
  fi
  # ---- 5d. the finished arm IS an S1 arm: start/done read updates == 400, evals and dumps exist, head files iff a head arm
  if ! stage_done "check_$ARM"; then
    env -u HF_TOKEN "$PY" s1aux_checks.py run-check --dir "$ARMDIR" --arm "$ARM" --json "$ARMDIR/run_check.json" > "$ARMDIR/run_check.log" 2>&1 200>&-
    grep -q "^ZZS1AUX_RUN_OK" "$ARMDIR/run_check.log" || refuse "arm $ARM is not a valid S1 arm: $(tail -n 1 "$ARMDIR/run_check.log" | cut -c1-500)"
    stage_mark "check_$ARM"
  fi
  log "ZZS1AUX_ARM_TRAINED $ARM ($(tail -n 1 "$ARMDIR/run_check.log" | cut -c1-120))"
  # ---- 5e. egress: the arm's checkpoint (B), head, states, logs, dumps (scores + head_pred), token, manifest.  Success is the REMOTE listing + content, not the push call.
  if ! stage_done "egress_$ARM"; then
    SPECS=""
    for f in model_sft_B.pt aux_drv.pt scorer_states_last.pt sft.jsonl sft.log preflight.jsonl preflight.log PREFLIGHT_PASS.json launch_gate.log token_verify.json run_check.json; do
      [ -f "$ARMDIR/$f" ] && SPECS="$SPECS $ARMDIR/$f::$f"
    done
    for f in "$ARMDIR"/dump/*.npz; do [ -f "$f" ] && SPECS="$SPECS $f::dump/$(basename "$f")"; done
    for f in "$LG"/PASS_*.json; do [ -f "$f" ] && SPECS="$SPECS $f::$(basename "$f")"; done
    [ -f "$MANIFEST" ] && SPECS="$SPECS $MANIFEST::arms_manifest.jsonl"
    bank_push "arm_$ARM" "arms/$ARM" $SPECS || refuse "the evidence of arm $ARM was not banked to HF (push_arm_$ARM.log, remote_arm_$ARM.log): it is on the pod ONLY; do not delete the pod"
    stage_mark "egress_$ARM"
  fi
  stage_mark "arm_$ARM"
  log "ZZS1AUX_ARM_DONE $ARM"; stop_after "arm_$ARM"
done
touch "$STOPF"
log "ZZS1AUX_DONE arms: $ARMS"
exit 0
