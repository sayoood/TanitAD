#!/bin/bash
# P5-C pod chain (PREREG_P5C + A1): from "P7's final checkpoint is banked" to "C1, C0 (and, only with --with-c0r, C0r) trained, G-ADAM-gated and banked off the pod".  Modelled on vdac_chain.sh / p7_pod.sh.
#   bash refe/p5c_pod.sh [--with-c0r] [--dry-run] [--clear-fail <stage>]     (run ON THE POD, after P7's last step; WRITTEN AND TESTED IN A STUB SANDBOX ONLY, NEVER RUN BY ITS AUTHOR)
# Stage order (each one a marker $ROOT/stage/<name>.ok = date + key; a stage that FAILS writes <name>.FAIL itself and the chain then REFUSES to run at all until `--clear-fail <name>` removes it, logged):
#   precheck -> inputs (+ pod idle) -> label (p5c_pod_label.sh, P5CL_NSHARD <= floor(the cgroup CPU quota), read at run time) -> table (verify, independent of the runner) -> pins (the MM's three pins: REFUSES until filled, NEVER fills them)
#   -> lab, mem, egr (gate evidence) -> per arm C1, C0 [, C0r]:  gate_<arm> (launch_gate mint + verify) -> preflight_<arm> -> train_<arm> -> postgate_<arm> (G-ADAM) -> egress_<arm> (HF, verified by the remote listing).
# No marker and no launch-gate token is ever hand-written: markers are written by this script after the stage's own check, the token only by `launch_gate.py run`.  A trainer is started only if `launch_gate.py verify` re-measures the token.
# A3.1 (PREREG_P5C_A3 section 8 + 9): the label shard count is capped by floor(the cgroup CPU QUOTA) -- cpu.max (v2) or cfs_quota_us/cfs_period_us (v1), as code/pod_dataprep.sh does -- NEVER by nproc (128 on the pod that grants 15.3 CPUs);
#   P5C_LABEL=p5r-rj5 (A3: C1 trains on the RJ5-based labels) REFUSES unless the A3.1 sha256 is supplied (P5C_A31_SHA256) AND equals the sha256 of the shipped $SHIP/PREREG_P5C_A3_1.md, and the RJ5 join sha256 is supplied (P5C_JOIN_SHA256) AND equals the shipped join's.
# Env (defaults in the code): P5C_WS (/workspace)  P5C_PKG  P5C_PY (python for the gate and tools)  P5C_LABEL (REQUIRED: tc20s | p5r | p5r-rj5 -- the MM's decision; never defaulted)  P5C_NSHARD (default = floor(cgroup CPU quota), capped by it; P5C_CGROUP_ROOT, default /sys/fs/cgroup, is for tests)  P5C_LABEL_WORKERS (REQUIRED, pinned)
#   p5r-rj5 only: P5C_A31_SHA256  P5C_A31_FILE (default $SHIP/PREREG_P5C_A3_1.md)  P5C_JOIN_SHA256  P5C_JOIN (default $SHIP/rj5_bank_join.jsonl.gz)  P5C_RJ5_PLATFORMS ("SRC=devbox|pod ...", declared per join source, never guessed)  P5C_RJ5_R0_ROWS ("SRC=DIR ..." R0 control row dirs for TB-8)
#   P5C_HF_TOKEN_FILE (REQUIRED)  P5C_REPO  P5C_P7_RUN  P5C_P7_DIR  P5C_REUSE_P7_LAB=1 (reuse P7's G-LIVE-LAB file; MEM is ALWAYS re-measured: it is head-dependent)  P5C_STOP_AFTER=<stage>  P5C_POLL_S  P5C_STEPS (2000).
# FAILURE LINES in chain.log start with `REFUSED: stage=` or `ZZP5C_CHAIN_FAIL`; a success path never prints the bare word REFUSED.
set -u
WITH_C0R=0; DRY=0; CLEAR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --with-c0r) WITH_C0R=1 ;;
    --dry-run) DRY=1 ;;
    --clear-fail) shift; CLEAR="${1:-}" ;;
    *) echo "REFUSED: stage=start: unknown argument '$1' (usage: p5c_pod.sh [--with-c0r] [--dry-run] [--clear-fail <stage>])"; exit 2 ;;
  esac
  shift
done
[ "${P5C_DRY_RUN:-0}" = 1 ] && DRY=1
WS=${P5C_WS:-/workspace}
DATA=$WS/data
PKG=${P5C_PKG:-$WS/refe-plan-p5c}
RUN=$PKG/refe
ROOT=$DATA/refe_p5c
if [ "$DRY" = 1 ]; then DRYROOT="$(mktemp -d)"; ROOT="$DRYROOT"; trap 'rm -rf "${DRYROOT:?}"' EXIT; fi          # a dry run writes its logs to a throwaway dir and touches nothing on the volume
S=$ROOT/stage
L=$ROOT/chain.log
SHIP=${P5C_SHIP:-$DATA/p5c_ship}
LABELW=${P5C_LABELWORK:-$DATA/p5c_label}
BANK=${P5C_BANK:-$DATA/refe_navtrain/train_grow_cmd_3c_p7}
P7RUN=${P5C_P7_RUN:-$DATA/refe_runs/p7_reserve_head}
P7DIR=${P5C_P7_DIR:-$DATA/refe_p7}
BASE=${P5C_BASE:-$P7RUN/model_final.pt}
PY=${P5C_PY:-$WS/venv/bin/python}
REPO=${P5C_REPO:-Sayood/tanitad-refe-p5c}
HFTOK=${P5C_HF_TOKEN_FILE:-}
STEPS=${P5C_STEPS:-2000}
EXPECT_ROWS=150603
STAGE=start
TREE=""
exec 3>&1          # fd 3: the dry run prints each command HERE, so a caller's `> file 2>&1` never swallows it
LG=$RUN/launch_gate.py
export TANITAD_GATE_KEY_FILE=${TANITAD_GATE_KEY_FILE:-$ROOT/launch_gate.key}      # the HMAC key lives on the volume, OUTSIDE the shipped tree and anything the egress pushes
ARMS="C1 C0"; [ "$WITH_C0R" = 1 ] && ARMS="C1 C0 C0r"
[ "$DRY" = 1 ] || mkdir -p "$S"
log() { echo "$(date -u +%FT%T) $*" >> "$L" 2> /dev/null || true; echo "$*"; }
refuse() {
  log "REFUSED: stage=$STAGE: $*"
  [ "$DRY" = 1 ] || [ "$STAGE" = precheck ] || { { date -u +%FT%T; echo "$*"; } > "$S/$STAGE.FAIL"; }
  exit 1
}
Q() { if [ "$DRY" = 1 ]; then echo "DRY: $*" >&3; echo "<dry>"; return 0; fi; "$@" 200>&-; }          # the same for a command whose stdout is CAPTURED
X() { if [ "$DRY" = 1 ]; then echo "DRY: $*" >&3; return 0; fi; "$@" 200>&-; }          # EVERY side-effecting command goes through X: a dry run prints it, never runs it
stop_after() { if [ "${P5C_STOP_AFTER:-}" = "$1" ]; then log "stopping after $1 (P5C_STOP_AFTER)"; exit 0; fi; }
sha_of() { sha256sum "$1" 2> /dev/null | cut -c1-64; }
tree_sha() { sha256sum "$PKG/SHIP_MANIFEST.json" 2> /dev/null | cut -c1-64; }
lcase() { echo "$1" | tr 'A-Z' 'a-z'; }
is_sha256() { case "$1" in *[!0-9a-f]*|'') return 1 ;; esac; [ "${#1}" = 64 ]; }
cpu_quota() {          # prints floor(the cgroup CPU quota) (>= 1); rc 1 when no quota is readable.  NEVER nproc: MEASURED 2026-11-15 on the P7 pod, nproc 128 against cpu.max 1530000/100000 = 15.3 CPUs (code/pod_dataprep.sh:30-41)
  local cg="${P5C_CGROUP_ROOT:-/sys/fs/cgroup}" q="" per=""
  { read -r q per < "$cg/cpu.max"; } 2> /dev/null
  case "$q" in ''|max|*[!0-9]*) ;; *) case "$per" in ''|*[!0-9]*|0) ;; *) echo $(( q / per > 0 ? q / per : 1 )); return 0 ;; esac ;; esac
  q=""; per=""
  { read -r q < "$cg/cpu/cpu.cfs_quota_us"; read -r per < "$cg/cpu/cpu.cfs_period_us"; } 2> /dev/null
  case "$q" in ''|*[!0-9]*|0) return 1 ;; esac
  case "$per" in ''|*[!0-9]*|0) return 1 ;; esac
  echo $(( q / per > 0 ? q / per : 1 ))
}
stage_done() { [ -f "$S/$1.ok" ] && [ "$(sed -n 2p "$S/$1.ok")" = "$2" ]; }
stage_mark() { { date -u +%FT%T; echo "$2"; } > "$S/$1.ok"; }
do_stage() {          # do_stage <name> <key> <function>
  local n="$1" k="$2" fn="$3"
  STAGE=$n
  [ ! -f "$S/$n.FAIL" ] || refuse "a FAIL marker exists ($S/$n.FAIL: $(tail -n 1 "$S/$n.FAIL" | cut -c1-200)): a previous attempt failed; look at the log, fix the cause, then --clear-fail $n"
  if [ "$DRY" = 1 ]; then echo "DRY STAGE $n"; $fn; stop_after "$n"; return 0; fi
  if stage_done "$n" "$k"; then log "stage $n already done (marker keyed $k)"; else $fn; stage_mark "$n" "$k"; log "ZZP5C_STAGE_OK $n"; fi
  stop_after "$n"
}
gpu_busy() {          # rc 0 = a process is on the GPU, OR the GPU cannot be asked (what cannot be proven idle is busy).  Never a process-name search.
  local nv="${P5C_NVSMI:-nvidia-smi}" apps
  apps="$("$nv" --query-compute-apps=pid --format=csv,noheader 2> /dev/null)" || return 0
  [ -n "$(echo "$apps" | tr -d '[:space:]')" ]
}
lock_held() { [ -e "$1/chain.lock" ] && ! ( flock -n 9 ) 9<> "$1/chain.lock"; }        # READ-WRITE fd: an exclusive flock on a read-only fd is EBADF on nfs4
others_hold() {
  local c
  for c in refe_p2 refe_p2rep rls1 s1aux refe_p6 refe_p7 refe_scorer1p6 refe_s1z refe_s1zsm refe_rep refe_dose refe_div refe_vdac; do
    if lock_held "$DATA/$c"; then echo "the $c chain holds $DATA/$c/chain.lock"; return 2; fi
  done
  return 0
}
wait_clear() {        # BLOCK until no chain holds the pod (P7 is OVER) and the GPU was idle on P5C_IDLE_POLLS consecutive polls
  local poll="${P5C_POLL_S:-30}" need="${P5C_IDLE_POLLS:-3}" max="${P5C_WAIT_MAX_S:-0}" idle=0 waited=0 why rc
  while :; do
    why="$(others_hold)"; rc=$?
    if [ "$rc" = 0 ] && gpu_busy; then why="a process is on the GPU (or nvidia-smi cannot say)"; rc=2; fi
    if [ "$rc" = 0 ]; then idle=$((idle + 1)); [ "$idle" -ge "$need" ] && return 0; else idle=0; log "ZZP5C_WAITING $why (waited ${waited}s)"; fi
    [ "$max" -gt 0 ] && [ "$waited" -ge "$max" ] && refuse "waited ${waited}s for the pod (P5C_WAIT_MAX_S): $why"
    sleep "$poll" 200>&-; waited=$((waited + poll))
  done
}
arm_out() { echo "$DATA/refe_runs/p5c_$(lcase "$1")"; }
arm_profile() { case "$1" in C1) echo refe_p5c ;; C0) echo refe_p5c_c0 ;; C0r) echo refe_p5c_c0r ;; *) refuse "unknown arm $1" ;; esac; }
arm_argv() {          # ARGS for arm $1: launch_gate.p5c_argv with THIS pod's real paths (the gate compares flags and levers, never the paths); the table path is the one the label stage printed
  [ "$DRY" = 1 ] && { ARGS="<launch_gate.p5c_argv $1 paths=pod>"; return 0; }
  ARGS="$(P5C_ARM=$1 P5C_TABLE_PATH=${TABLE:-} P5C_BANK_PATH=$BANK P5C_BASE_PATH=$BASE P5C_DATA=$DATA P5C_LW=$LW "$PY" - "$RUN" 200>&- <<'PYEOF'
import os, sys
sys.path.insert(0, sys.argv[1])
import launch_gate as L
d = os.environ["P5C_DATA"]
paths = {"targets": os.environ["P5C_BANK_PATH"], "images": d + "/navtrain_pixels", "calib": os.environ["P5C_BANK_PATH"] + "/calib_table.json", "metric_cache": d + "/refe_navtrain/metric_cache",
         "init_from": os.environ["P5C_BASE_PATH"], "ego_yaw_table": d + "/refe_navtrain/ego_yaw_navtrain.json.gz", "out_root": d + "/refe_runs", "label_workers": os.environ["P5C_LW"]}
if os.environ.get("P5C_TABLE_PATH"):
    paths["traj_label_table"] = os.environ["P5C_TABLE_PATH"]
print(" ".join(L.p5c_argv(os.environ["P5C_ARM"], paths=paths)))
PYEOF
)" || refuse "launch_gate.p5c_argv could not build the $1 argv"
  [ -n "$ARGS" ] || refuse "the $1 argv is empty"
}
token_ok() {          # $1 = arm; uses $ARGS.  `launch_gate.py verify` RE-MEASURES the argv, the code tree, the data and the checkpoint, and checks the HMAC.
  local tok
  tok="$(ls "$ROOT/lg_$1"/PASS_*.json 2> /dev/null | head -n 1)"
  [ -n "$tok" ] || return 1
  env -u HF_TOKEN "$PY" "$LG" verify --profile "$(arm_profile "$1")" --token "$tok" --args "$ARGS" --pkg "$PKG" --json "$ROOT/token_verify_$1.json" >> "$ROOT/token_verify_$1.log" 2>&1 200>&-
}

wait_clear_once() { [ "$DRY" = 1 ] || wait_clear; }
# ---- 0. the lock, the arguments, the shipped tree
if [ -n "$CLEAR" ]; then
  [ -f "$S/$CLEAR.FAIL" ] || { case "$CLEAR" in train_*) [ -f "$S/trainstart_${CLEAR#train_}.ok" ] ;; *) false ;; esac; } || { echo "no FAIL marker $S/$CLEAR.FAIL (and no trainstart marker to release)"; exit 1; }
  log "FAIL marker $CLEAR removed by --clear-fail (was: $(tail -n 1 "$S/$CLEAR.FAIL" 2> /dev/null | cut -c1-200))"; rm -f "$S/$CLEAR.FAIL"
  case "$CLEAR" in train_*) a="${CLEAR#train_}"; if [ ! -e "$(arm_out "$a")/ckpt_last.pt" ]; then rm -f "$S/trainstart_$a.ok"; log "trainstart_$a.ok released: $(arm_out "$a") holds no ckpt_last.pt (archived)"; fi ;; esac
  exit 0
fi
[ "$DRY" = 1 ] || { exec 200>"$ROOT/chain.lock"; flock -n 200 || { echo "REFUSED: stage=precheck: another p5c_pod.sh holds $ROOT/chain.lock"; exit 1; }; }
st_precheck() {
  local f c ncpu fm
  fm="$(ls "$S"/*.FAIL 2> /dev/null | head -n 1)"
  [ -z "$fm" ] || refuse "a FAIL marker exists: $fm ($(tail -n 1 "$fm" | cut -c1-200)): fix the cause, then --clear-fail <stage>"
  [ -n "$HFTOK" ] || refuse "P5C_HF_TOKEN_FILE is not set (hf_egress.py reads the token file in place; no default path in source)"
  [ -r "$HFTOK" ] && grep -qE 'hf_[A-Za-z0-9]{8,}' "$HFTOK" || refuse "$HFTOK is missing, unreadable or holds no hf_ token"
  case "${P5C_LABEL:-}" in tc20s|p5r|p5r-rj5) ;; *) refuse "P5C_LABEL is '${P5C_LABEL:-}': the label that trains (tc20s | p5r | p5r-rj5) is the MM's decision and is never defaulted" ;; esac
  if [ "$P5C_LABEL" = p5r-rj5 ]; then
    A31FILE="${P5C_A31_FILE:-$SHIP/PREREG_P5C_A3_1.md}"; JOINF="${P5C_JOIN:-$SHIP/rj5_bank_join.jsonl.gz}"
    is_sha256 "${P5C_A31_SHA256:-}" || refuse "P5C_LABEL=p5r-rj5 needs the A3.1 sha256 in P5C_A31_SHA256 (64 hex): the chain refuses to build a C1 label until Amendment A3.1 is frozen (PREREG_P5C_A3 section 9)"
    [ -f "$A31FILE" ] || refuse "P5C_LABEL=p5r-rj5: the frozen A3.1 file $A31FILE is not shipped"
    [ "$(sha_of "$A31FILE")" = "$P5C_A31_SHA256" ] || refuse "P5C_A31_SHA256 does not match the sha256 of $A31FILE ($(sha_of "$A31FILE" | cut -c1-16)...): the label chain runs only under the frozen A3.1"
    is_sha256 "${P5C_JOIN_SHA256:-}" || refuse "P5C_LABEL=p5r-rj5 needs the RJ5 join sha256 in P5C_JOIN_SHA256 (64 hex; A3 section 2 pins it before feed runs)"
    [ -f "$JOINF" ] || refuse "P5C_LABEL=p5r-rj5: the RJ5 join $JOINF is not shipped"
    [ "$(sha_of "$JOINF")" = "$P5C_JOIN_SHA256" ] || refuse "P5C_JOIN_SHA256 does not match the sha256 of $JOINF"
    [ -n "${P5C_RJ5_PLATFORMS:-}" ] || refuse "P5C_RJ5_PLATFORMS is not set: each join source's roll platform (devbox | pod) is declared, never guessed"
    [ -n "${P5C_RJ5_R0_ROWS:-}" ] || refuse "P5C_RJ5_R0_ROWS is not set: TB-8's per-log R0 control rows are required"
  fi
  case "${P5C_LABEL_WORKERS:-}" in ''|*[!0-9]*) refuse "P5C_LABEL_WORKERS is not set to an integer: the label-worker count is part of the argv sha and is never derived from the cgroup quota" ;; esac
  LW="$P5C_LABEL_WORKERS"; { [ "$LW" -ge 1 ] && [ "$LW" -le 16 ]; } || refuse "P5C_LABEL_WORKERS=$LW is outside 1..16"
  ncpu="$(cpu_quota)" || refuse "no readable cgroup CPU quota (cpu.max, or cpu/cpu.cfs_quota_us + cfs_period_us, under ${P5C_CGROUP_ROOT:-/sys/fs/cgroup}): the label shard count is capped by the quota, never by nproc"
  NSHARD="${P5C_NSHARD:-$ncpu}"
  case "$NSHARD" in ''|*[!0-9]*) refuse "P5C_NSHARD='$NSHARD' is not an integer" ;; esac
  { [ "$NSHARD" -ge 1 ] && [ "$NSHARD" -le "$ncpu" ]; } || refuse "P5C_NSHARD=$NSHARD exceeds this pod's $ncpu CPUs (floor of the cgroup CPU quota, read at run time; never nproc): the runner's own default 16 is for a 64-core box"
  TREE="$(tree_sha)"
  [ -n "$TREE" ] || refuse "no $PKG/SHIP_MANIFEST.json: ship the landed tree with its manifest (launch_gate.py ship-manifest --commit <40-hex>)"
  for f in refe/train.py refe/launch_gate.py refe/gate_token.py refe/gate_checks.py refe/gate_fixture.py refe/hf_egress.py refe/traj_label_table.py refe/p5c_gate.py refe/p5c_postgate.py refe/p6_prep.py refe/test_p5c_live.py refe/test_p5c_postgate.py refe/test_p5c_profile.py refe/test_traj_label_table.py SHIP_MANIFEST.json; do
    [ -f "$PKG/$f" ] || refuse "$PKG/$f is missing from the shipped package"
  done
  [ -f "$SHIP/p5c_pod_label.sh" ] || refuse "$SHIP/p5c_pod_label.sh is missing: ship the label kit (a1/POD_SHIP_LIST.txt + the launchkit's p5c_pod_label.sh and traj_label_table.py)"
  for c in C1 C0 $( [ "$WITH_C0R" = 1 ] && echo C0r ); do
    [ -f "$S/train_$c.ok" ] || [ ! -e "$(arm_out $c)/ckpt_last.pt" ] || refuse "$(arm_out $c)/ckpt_last.pt exists: a fresh fine-tune never starts over an old run (archive it: mv $(arm_out $c) $(arm_out $c).killed_N)"
  done
  X env -u HF_TOKEN "$PY" - "$PKG" "$RUN" > "$ROOT/tree_check.log" 2>&1 <<'PYEOF' || refuse "the shipped tree is not its SHIP_MANIFEST.json: $(tail -n 1 "$ROOT/tree_check.log" | cut -c1-300)"
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
import gate_token as GT
pkg = Path(sys.argv[1])
man = GT.read_json(pkg / GT.SHIP_MANIFEST)
tree = GT.tree_manifest(pkg)
if man.get("files") != tree:
    sys.exit("tree differs from the manifest: %s" % GT.tree_diff(man.get("files") or {}, tree))
print("ZZP5C_TREE_OK %d files commit %s" % (len(tree), str(man.get("commit"))[:12]))
PYEOF
  log "ZZP5C_START arms [$ARMS], label $P5C_LABEL, $NSHARD label shards of $ncpu CPUs, p5c_pod.sh md5 $(md5sum "$0" | cut -c1-12), tree ${TREE:0:12}"
}
if [ "$DRY" = 1 ]; then LW=${P5C_LABEL_WORKERS:-13}; NSHARD=${P5C_NSHARD:-$(cpu_quota 2> /dev/null || echo 15)}; TREE=dry; P5C_LABEL=${P5C_LABEL:-p5r}; fi
STAGE=precheck
if [ "$DRY" = 1 ]; then echo "DRY STAGE precheck (token file, P5C_LABEL, label workers, NSHARD <= cgroup CPU quota, p5r-rj5 => A3.1 sha + join sha match, tree == SHIP_MANIFEST, no ckpt_last.pt, no FAIL marker)"; else st_precheck; fi
stop_after precheck
TREE="${TREE:-$(tree_sha)}"; LKEY="$(sha_of "$SHIP/p5c_pod_label.sh")-$(sha_of "$SHIP/traj_label_table.py")-n$NSHARD"
if [ "${P5C_LABEL:-}" = p5r-rj5 ]; then J="${P5C_JOIN_SHA256:-}"; A="${P5C_A31_SHA256:-}"; LKEY="$LKEY-$(sha_of "$SHIP/p5c_rj5_path.py" | cut -c1-16)-${J:0:16}-${A:0:16}"; fi

# ---- 1. the inputs: the base (P7's final model_final.pt) hashed, the bank, the pod idle
st_inputs() {
  local p
  if [ "$DRY" = 1 ]; then echo "DRY: check the base / bank / pixels / metric cache / yaw table; sha256sum $BASE; wait for the pod to be idle"; BASE_SHA="<base sha256>"; return 0; fi
  for p in "$BASE" "$BANK" "$DATA/navtrain_pixels" "$DATA/refe_navtrain/metric_cache" "$DATA/refe_navtrain/ego_yaw_navtrain.json.gz" "$P7RUN/config.json"; do
    [ -e "$p" ] || refuse "$p is missing: P7's final outputs and inputs must be on the pod"
  done
  BASE_SHA="$(sha_of "$BASE")"; echo "$BASE_SHA  $BASE" > "$ROOT/BASE_SHA256.txt"
  log "ZZP5C_BASE sha256 $BASE_SHA ($BASE)"
  wait_clear
}
do_stage inputs "$TREE" st_inputs
BASE_SHA="$(cut -c1-64 "$ROOT/BASE_SHA256.txt" 2> /dev/null)"

# ---- 2. label the bank (the runner is resumable with the SAME shard count; it builds and verifies its own tables)
st_label() {
  [ "$DRY" = 1 ] || wait_clear
  X env P5CL_NSHARD="$NSHARD" P5CL_SHIP="$SHIP" P5CL_WORK="$LABELW" P5CL_LABEL="$P5C_LABEL" P5CL_JOIN="${JOINF:-}" P5CL_JOIN_SHA256="${P5C_JOIN_SHA256:-}" P5CL_A31_SHA256="${P5C_A31_SHA256:-}" P5CL_PLATFORMS="${P5C_RJ5_PLATFORMS:-}" P5CL_R0_ROWS="${P5C_RJ5_R0_ROWS:-}" P5CL_CGROUP_ROOT="${P5C_CGROUP_ROOT:-}" bash "$SHIP/p5c_pod_label.sh" > "$ROOT/label.log" 2>&1
  [ "$DRY" = 1 ] && return 0
  grep -q "^ZZP5CL_DONE" "$ROOT/label.log" || refuse "the label runner did not finish ($(tail -n 2 "$ROOT/label.log" | tr '\n' ' ' | cut -c1-300)); a STOP / FAIL there is a verdict about the labels, not the chain's"
  grep -q "^ZZP5CL_TABLE $P5C_LABEL " "$ROOT/label.log" || refuse "the runner built no table for the label '$P5C_LABEL' (P5C_LABEL): its lines are $(grep '^ZZP5CL_TABLE' "$ROOT/label.log" | cut -c1-80 | tr '\n' ' ')"
}
do_stage label "$LKEY" st_label

# ---- 3. the table, verified here independently of the runner: rows == covered == 150,603, no refusal, the policy the runner printed
st_table() {
  local line
  if [ "$DRY" = 1 ]; then echo "DRY: traj_label_table.py verify --table <table of label $P5C_LABEL> --bank-dir $BANK; rows == covered == $EXPECT_ROWS, rank_policy as printed"; return 0; fi
  line="$(grep "^ZZP5CL_TABLE $P5C_LABEL " "$ROOT/label.log" | tail -n 1)"
  TABLE="$(echo "$line" | sed -n 's/.* table=\([^ ]*\).*/\1/p')"; POLICY="$(echo "$line" | sed -n 's/.* rank_policy=\([a-z0-9]*\).*/\1/p')"; TSHA="$(echo "$line" | sed -n 's/.* sha256=\([0-9a-f]*\).*/\1/p')"
  [ -f "$TABLE" ] || refuse "the table $TABLE the runner printed does not exist"
  [ "$(sha_of "$TABLE")" = "$TSHA" ] || refuse "the table's sha256 is not the one the runner printed ($TSHA)"
  X env -u HF_TOKEN "$PY" "$RUN/traj_label_table.py" verify --table "$TABLE" --bank-dir "$BANK" > "$ROOT/table_verify.json" 2>&1 || refuse "traj_label_table.py verify refused the table ($(tail -n 1 "$ROOT/table_verify.json" | cut -c1-300))"
  X env -u HF_TOKEN "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['rows']==$EXPECT_ROWS and d['covered']==$EXPECT_ROWS and not d['refused'] and d['rank_policy']==sys.argv[2], d" "$ROOT/table_verify.json" "$POLICY" > "$ROOT/table_verify_counts.log" 2>&1 || refuse "the table does not cover all $EXPECT_ROWS rows or its rank_policy is not '$POLICY' ($(tail -n 1 "$ROOT/table_verify_counts.log" | cut -c1-300))"
  { echo "$TABLE"; echo "$POLICY"; echo "$TSHA"; } > "$ROOT/TABLE_SELECTED.txt"
  log "ZZP5C_TABLE $P5C_LABEL rank_policy=$POLICY sha256=$TSHA"
}
do_stage table "$LKEY" st_table
if [ "$DRY" = 1 ]; then TABLE='<table>'; POLICY='<policy>'; TSHA='<sha>'; else TABLE="$(sed -n 1p "$ROOT/TABLE_SELECTED.txt")"; POLICY="$(sed -n 2p "$ROOT/TABLE_SELECTED.txt")"; TSHA="$(sed -n 3p "$ROOT/TABLE_SELECTED.txt")"; fi

# ---- 4. THE MM PIN CHECK.  launch_gate.py's three pins are filled BY THE MM (in the landed source), never by this script: until they are, the profiles REFUSE and so does the chain.  Once filled they must be the MEASURED values.
st_pins() {
  local out
  if [ "$DRY" = 1 ]; then echo "DRY: REFUSE until launch_gate.P5C_BASE_SHA256 / P5C_LABEL_TABLE_SHA256 / P5C_RANK_POLICY are filled by the MM and equal the measured base sha256, table sha256 and rank policy"; return 0; fi
  out="$(env -u HF_TOKEN "$PY" - "$RUN" "$BASE_SHA" "$TSHA" "$POLICY" 200>&- <<'PYEOF'
import re, sys
sys.path.insert(0, sys.argv[1])
import launch_gate as L
base, tsha, pol = sys.argv[2:5]
pins = {"P5C_BASE_SHA256": (L.P5C_BASE_SHA256, base), "P5C_LABEL_TABLE_SHA256": (L.P5C_LABEL_TABLE_SHA256, tsha), "P5C_RANK_POLICY": (L.P5C_RANK_POLICY, pol)}
unfilled = [k for k, (v, _) in pins.items() if v is None]
if unfilled:
    print("UNFILLED " + " ".join(unfilled)); sys.exit(3)
bad = [f"{k}: pinned {str(v)[:16]}, measured {str(m)[:16]}" for k, (v, m) in pins.items() if v != m]
if bad:
    print("MISMATCH " + "; ".join(bad)); sys.exit(4)
print("ZZP5C_PINS_OK")
PYEOF
)" || refuse "launch_gate.py's P5-C pins are not usable ($out): the MM fills P5C_BASE_SHA256 (= $BASE_SHA), P5C_LABEL_TABLE_SHA256 (= $TSHA) and P5C_RANK_POLICY (= $POLICY) in the landed source, re-derives SHIP_MANIFEST.json and ships the tree again; this chain never fills a pin"
  log "$out"
}
do_stage pins "$TREE-$BASE_SHA-$TSHA" st_pins

# ---- 5. the gate's pod-run evidence: G-LIVE-LAB (reusable from P7 only on the MM's say), G-LIVE-MEM (ALWAYS re-measured: the memory of a head-less model), G-EGR
st_lab() {
  if [ "${P5C_REUSE_P7_LAB:-0}" = 1 ]; then
    [ -s "$P7DIR/lab.json" ] || refuse "P5C_REUSE_P7_LAB=1 but $P7DIR/lab.json is missing"
    cp "$P7DIR/lab.json" "$ROOT/lab.json"; log "ZZP5C_LAB reused P7's $P7DIR/lab.json (sha256 $(sha_of "$ROOT/lab.json")): its worker count and chunks must be this argv's ($LW, 2)"
  else
    local toks
    toks="$(Q env -u HF_TOKEN "$PY" "$RUN/p6_prep.py" sample-tokens --cache "$DATA/refe_navtrain/metric_cache" --n 128 | sed -n 's/^ZZTOKENS //p')"
    [ -n "$toks" ] || [ "$DRY" = 1 ] || refuse "no parts to sample for the labeller instrument"
    X env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK "$PY" "$RUN/diag_p6_labeller_identity.py" --cache "$DATA/refe_navtrain/metric_cache" --db-root "$DATA/navtrain_dbs" --tokens "$toks" --workers "$LW" --chunks 2 --n-sets "${P5C_LAB_SETS:-256}" --out "$ROOT/lab.json" > "$ROOT/lab.log" 2>&1
    grep -q "^ZZLAB_IDENTITY_PASS" "$ROOT/lab.log" || [ "$DRY" = 1 ] || refuse "G-LIVE-LAB instrument: identity or the stale-proposal refusal failed (lab.log)"
  fi
}
st_mem() {          # one measurement with C1's argv (C0 differs only by the table flag: same model, same shapes); its ckpt_last.pt is G-EGR's probe
  local m
  arm_argv C1
  X rm -rf "$ROOT/mem_run"
  wait_clear_once
  m="$(Q env -u HF_TOKEN "$PY" "$RUN/p6_prep.py" mem-argv --args "$ARGS" --out "$ROOT/mem_run" | sed -n 's/^ZZMEM_ARGV //p')"
  [ -n "$m" ] || [ "$DRY" = 1 ] || refuse "could not derive the memory-run argv"
  X env -u HF_TOKEN "$PY" "$RUN/train.py" $m > "$ROOT/mem.log" 2>&1
  grep -q "^TRAIN_DONE" "$ROOT/mem.log" || [ "$DRY" = 1 ] || refuse "the G-LIVE-MEM run did not finish (mem.log)"
  [ "$DRY" = 1 ] || { [ -s "$ROOT/mem_run/metrics.jsonl" ] && [ -s "$ROOT/mem_run/ckpt_last.pt" ]; } || refuse "the G-LIVE-MEM run left no metrics.jsonl / ckpt_last.pt"
  X cp "$ROOT/mem_run/metrics.jsonl" "$ROOT/mem_metrics.jsonl"
}
st_egr() {
  X env -u HF_TOKEN "$PY" "$RUN/hf_egress.py" gate --repo "$REPO" --out "$ROOT/egr" --ckpt "$ROOT/mem_run/ckpt_last.pt" --chain-log "$L" --token-file "$HFTOK" > "$ROOT/egr.log" 2>&1
  X env -u HF_TOKEN "$PY" "$RUN/p6_prep.py" egr-ok --json "$ROOT/egr/g_egr_result.json" > "$ROOT/egr_ok.out" 2>&1
  grep -q "^ZZEGR_OK" "$ROOT/egr_ok.out" || [ "$DRY" = 1 ] || refuse "G-EGR instrument failed ($(tail -n 1 "$ROOT/egr_ok.out" | cut -c1-200))"
}
do_stage lab "$TREE-w$LW-reuse${P5C_REUSE_P7_LAB:-0}" st_lab
do_stage mem "$TREE-$BASE_SHA-$TSHA" st_mem
do_stage egr "$TREE-$BASE_SHA" st_egr

# ---- 6. the arms, one after the other (C1 first, then C0; C0r only with --with-c0r)
arm_gate() {
  local a="$1"
  arm_argv "$a"
  X rm -rf "$ROOT/lg_$a" "$ROOT/gate_bank_$a"; X mkdir -p "$ROOT/lg_$a"
  X env -u HF_TOKEN OMP_NUM_THREADS=4 "$PY" "$LG" run --profile "$(arm_profile "$a")" --pkg "$PKG" --args "$ARGS" --out-dir "$ROOT/lg_$a" \
    --gate-bank "$ROOT/gate_bank_$a" --gate-bank-n 64 --gate-lab-json "$ROOT/lab.json" --gate-mem-metrics "$ROOT/mem_metrics.jsonl" --gate-egr-json "$ROOT/egr/g_egr_result.json" > "$ROOT/launch_gate_$a.log" 2>&1
  [ "$DRY" = 1 ] || echo "ZZEXIT $?" >> "$ROOT/launch_gate_$a.log"
  [ "$DRY" = 1 ] || token_ok "$a" || refuse "the launch gate did not mint a PASS token that verifies for $a: $(grep -E 'REASON|FAIL' "$ROOT/launch_gate_$a.log" | head -n 3 | cut -c1-300 | tr '\n' ' ')"
}
arm_preflight() {
  local a="$1"
  arm_argv "$a"
  [ "$DRY" = 1 ] || token_ok "$a" || refuse "no launch-gate token verifies for $a: its preflight and trainer do not start"
  X env -u HF_TOKEN "$PY" "$RUN/train.py" $ARGS --preflight > "$ROOT/preflight_$a.log" 2>&1
  grep -q "^PREFLIGHT_OK" "$ROOT/preflight_$a.log" || [ "$DRY" = 1 ] || refuse "train.py --preflight did not print PREFLIGHT_OK for $a"
}
arm_train() {
  local a="$1" out; out="$(arm_out "$a")"
  arm_argv "$a"
  if [ "$DRY" != 1 ] && [ -f "$S/trainstart_$a.ok" ]; then refuse "a previous attempt of $a STARTED training and did not finish (trainstart_$a.ok without train_$a.ok): archive it (mv $out $out.killed_N), remove nothing else; the arm restarts at step 0 under a fresh gate and token"; fi
  wait_clear_once
  [ "$DRY" = 1 ] || token_ok "$a" || refuse "the launch-gate token does not verify for $a: train.py is NOT started ($(tail -n 2 "$ROOT/token_verify_$a.log" | tr '\n' ' ' | cut -c1-300))"
  [ "$DRY" = 1 ] || { { date -u +%FT%T; echo "$TREE"; } > "$S/trainstart_$a.ok"; log "ZZP5C_TRAIN_START $a"; }
  X env -u HF_TOKEN PYTHONUNBUFFERED=1 "$PY" "$RUN/train.py" $ARGS > "$ROOT/train_$a.log" 2>&1
  local rc=$?
  log "TRAIN EXIT $a rc $rc ($(tail -n 1 "$ROOT/train_$a.log" 2> /dev/null | cut -c1-160))"
  [ "$DRY" = 1 ] || { [ "$rc" = 0 ] && [ -s "$out/model_final.pt" ] && [ -s "$out/ckpt_last.pt" ]; } || refuse "$a training did not finish (rc $rc; no model_final.pt / ckpt_last.pt): ZZP5C_CHAIN_FAIL"
  [ "$DRY" = 1 ] || "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['done'] is True and d['steps']==int(sys.argv[2]), d" "$out/summary.json" "$STEPS" 200>&- || refuse "$a's summary.json is not done=true at $STEPS steps"
}
arm_postgate() {
  local a="$1" out other=""; out="$(arm_out "$a")"
  [ "$a" = C0 ] && other="--other $(arm_out C1)"
  X env -u HF_TOKEN "$PY" "$RUN/p5c_postgate.py" "$out" --base "$BASE" --steps "$STEPS" --n-params-from "$P7RUN/config.json" $other > "$ROOT/postgate_$a.log" 2>&1
  [ "$DRY" = 1 ] || grep -q "^ZZP5C_POSTGATE_OK" "$ROOT/postgate_$a.log" || refuse "the post-training G-ADAM gate failed for $a ($(tail -n 1 "$ROOT/postgate_$a.log" | cut -c1-300)): the arm is not read, ZZP5C_CHAIN_FAIL"
  log "post-training gate $a: $(tail -n 1 "$ROOT/postgate_$a.log" 2> /dev/null | cut -c1-200)"
}
arm_egress() {
  local a="$1" out n0 la; out="$(arm_out "$a")"; la="$(lcase "$a")"
  n0=$(wc -l < "$L" 2> /dev/null || echo 0)
  X env -u HF_TOKEN "$PY" "$RUN/hf_egress.py" push --repo "$REPO" --prefix "p5c/$la" --max-push-bytes 6e9 --tag "arm_p5c_$la" --chain-log "$L" --egress-log "$ROOT/egress.jsonl" --token-file "$HFTOK" \
    --files "$out/model_final.pt" "$out/ckpt_last.pt" "$out/config.json" "$out/summary.json" "$out/metrics.jsonl" "$ROOT/postgate_$a.log" "$ROOT/train_$a.log" "$ROOT/launch_gate_$a.log" >> "$ROOT/egress_$a.log" 2>&1
  [ "$DRY" = 1 ] || tail -n +$((n0 + 1)) "$L" | grep -q " ZZEGRESS_OK arm_p5c_$la " || refuse "the $a egress push did not verify against the remote listing: the arm lives on the pod only"
  X sha256sum "$out/model_final.pt" > "$ROOT/model_final_sha256_$a.txt"
}
for A in $ARMS; do
  [ "$A" = C0r ] && [ ! -f "$S/postgate_C0.ok" ] && [ "$DRY" != 1 ] && { STAGE=gate_C0r; refuse "C0r (the conditional replicate, the PI's approval) runs only after C1 and C0 are trained and gated"; }
  do_stage "gate_$A" "$TREE-$BASE_SHA-$TSHA" "arm_gate $A"
  do_stage "preflight_$A" "$TREE-$BASE_SHA-$TSHA" "arm_preflight $A"
  do_stage "train_$A" "$TREE-$BASE_SHA-$TSHA" "arm_train $A"
  do_stage "postgate_$A" "$TREE-$BASE_SHA-$TSHA" "arm_postgate $A"
  do_stage "egress_$A" "$TREE-$BASE_SHA-$TSHA" "arm_egress $A"
  log "ZZP5C_ARM_DONE $A"
done
log "ZZP5C_CHAIN_DONE arms [$ARMS] (both arms' model_final.pt sha256 in $ROOT/model_final_sha256_<arm>.txt: seal them in ONE manifest before either is read, card 2.1 / 11)"
