#!/bin/bash
# P6 RESTAGE v2: a FRESH RunPod GPU pod -> "P6 training resumed" (2026-10-25, MM / engineering agent).  Successor of rebuild_driver.sh (2026-10-22, ~1.5 days),
# which REBUILT what is now BANKED.  Differences: (1) the D9 metric cache and the 3-C bank are FETCHED from HF (size + sha256 + member layout checked) instead of
# rebuilt (5.2 CPU-h + 23 h); (2) the twin parts are built, then LINKED into metric_cache (POST_MERGE step (d)) and the chain's own D7 check must read OK for K=1,2,3 BEFORE
# the chain starts -- the omission of that step is what quarantined all 21,821 twins on 2026-10-22; (3) P6_D7_FALLBACK=0 (a D7 BAD REFUSES, never quarantines);
# (4) DBs + pixels go to the LOCAL container disk with symlinks back, fetched in parallel with everything else; (5) a post-launch coverage guard;
# (6) optional rollback to an earlier HF checkpoint (P6_RESUME_REV).
# Resumable: re-run it; finished steps leave $O/<step>.done.  Any mismatch STOPS (ZZRS_FAIL + $O/FAILED), never relaxes.
# The token is read from <TOKEN_FILE>: into the environment of the HF-using children only (HF_TOKEN=... on one command) and into a mode-600 curl header file that is
# deleted before the final chain.  It is never an argument, never printed, never written anywhere else.
# RESTAGE_DRY=1: print every step, estimate and check; no network, no writes.
#
# Start (detached):  setsid -f nohup bash /workspace/ops/restage_v2/restage_v2.sh > /workspace/ops/restage_v2/driver.out 2>&1 < /dev/null
# Knobs (env): P6_RESUME_REV, P6_EXPECT_RESUME_STEP, P6_RESUME_PUSH_REPO, P6_RESUME_TAG, RS_TWIN_BANK (pre3c|3c), RS_LOCAL, RS_NP, RS_PIX_PAR, RS_DB_JOBS, RS_LABEL_WORKERS, WS_NEED_GB.
set -u
DRY=0; [ "${RESTAGE_DRY:-0}" = 1 ] && DRY=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
MAIN=$$
# the path is bound by the launch-gate token (--targets <CBANK> is in ARGS): it is NOT a knob
W=/workspace; O=$W/ops/restage_v2; DATA=$W/data; NB=$DATA/refe_navtrain; P6=$DATA/refe_p6; RUN=$DATA/refe_runs/p6_paper_faithful
MC=$NB/metric_cache; TW=$NB/metric_cache_twins_3c; CB=$NB/train_grow_cmd_3c; F3=$W/ops/bank3c/full
LOCAL="${RS_LOCAL:-/root/tad}"                       # the local container disk (the 2026-10-22 pod: / = 700G local; /workspace = ~270 GiB network quota that df cannot see)
R="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"; T="$R/raw/2026-10-17-twin-bank-3c"
BCC=bcc04b52233e28ef8eaafc346472c080e027e4a9; TIP=f2f604f5d8067352287e113d90bc925f76c9641e; D7BASE=c1dc5d33233259a8f5f876ecd89f2f4f2c3c9573
GIT_URL=https://github.com/sayoood/TanitAD.git     # public repo; no credentials
G="git -C $W/repo"
PKGT=$W/refe-plan-p6; PKGB=$W/refe-plan-p6-prebuild; D7PKG=$W/ops/d7fix/pkg
BANKS_URL=https://huggingface.co/datasets/Sayood/tanitad-refe-navtrain-banks/resolve/main
D9_NAME=refe_d9_metric_cache_newpod_2026-10-22.tar; D9_SIZE=36282746880
D9_SHA=2bc68e31b179e134a99e53d8a16d09f67dd48a28139edecd7eecebea2d13f624    # LFS oid, MEASURED 2026-10-25 from the HF tree API (prefix 2bc68e31b179e134 = the PI's note)
D9_FILES="${D9_FILES_EXPECT:-107860}"                                       # the packer's own `find metric_cache -type f | wc -l` (d9cache_hf.sh log, 2026-10-22)
C3_NAME=refe_train_grow_cmd_3c.tar.zst; C3_SIZE=217062290
C3_SHA=8b965193c23a5afe78b97a125a07e162838f5479ba4dccab227f4450ace7074d
S0=daa23b962e9513673347694e097fb714a53b646c4d402802d6d7b96354c0bc5e        # rank0, identical in the pre-3C and the 3-C bank
S1IN=2f3e461391d7bbc6c92a4153ef1837943c458d9635aad13fdb27b74a46035898      # pre-3C train_grow_cmd rank1 (what the commands stage must reproduce)
S1OUT=2a63acc2f63404818939b21d45cfa27efb5412cbefdf38e14aeb8673598cf409     # 3-C rank1
PODH_SHA=2d26ea957d9a15daf133d8e4bd0192e86dc26f337291b286b669f4999b7a6efa   # pod_helpers.py (repo blob at the tip, == the rehearsed twinfix copy)
TIP_TREE12=b0fad37d9704; TIP_FILES=272                                      # SHIP_MANIFEST of the tip package on the 2026-10-21 pod (the identity P6 trained under)
ARGS_SHA=570619873b1ebaa0                                                   # the chain's ARGS sha for the run (logged by the chain: "ARGS sha ...")
TWINS=21821; RANKS_WANT="1 2 3"
NP="${RS_NP:-8}"; PIX_PAR="${RS_PIX_PAR:-2}"; DB_JOBS="${RS_DB_JOBS:-8}"; LW="${RS_LABEL_WORKERS:-6}"   # LW=6 reproduces ARGS sha 570619873b1ebaa0
LOCAL_NEED_GB="${LOCAL_NEED_GB:-360}"; WS_NEED_GB="${WS_NEED_GB:-110}"
TWIN_BANK="${RS_TWIN_BANK:-pre3c}"
P6REPO="${P6_REPO:-Sayood/tanitad-refe-p6-paper-faithful}"
RESUME_REV="${P6_RESUME_REV:-}"
if [ -n "$RESUME_REV" ]; then
  case "$RESUME_REV" in fc2adb00b59e*) EXPECT_STEP="${P6_EXPECT_RESUME_STEP:-2603}" ;; *) EXPECT_STEP="${P6_EXPECT_RESUME_STEP:-}" ;; esac
  # rollback pushes go to a NEW repo by default so nothing in the original repo is overwritten (HF keeps history anyway; 'same' = push to the original + a protective tag)
  case "${P6_RESUME_PUSH_REPO:-}" in "") PUSH_REPO="${P6REPO}-rb${EXPECT_STEP}" ;; same) PUSH_REPO="$P6REPO" ;; *) PUSH_REPO="$P6_RESUME_PUSH_REPO" ;; esac
else
  EXPECT_STEP="${P6_EXPECT_RESUME_STEP:-4321}"; PUSH_REPO="$P6REPO"
fi
AUTH=/root/.hf_auth_hdr

log() { local m; m="$(date -u +%FT%TZ) $*"; if [ $DRY = 1 ]; then echo "[DRY] $m"; else echo "$m" >> $O/driver.log; echo "$m"; fi; }
kill_tracks() { [ -f $O/track_pids ] && for p in $(cat $O/track_pids); do pkill -P $p 2> /dev/null; kill $p 2> /dev/null; done; true; }
fail() { log "ZZRS_FAIL $*"; [ $DRY = 1 ] && exit 1; echo "$*" > $O/FAILED; [ "${BASHPID:-$$}" = "$MAIN" ] && kill_tracks; exit 1; }
done_() { [ -f $O/$1.done ]; }
mark() { date -u +%FT%T > $O/$1.done; log "ZZRS_OK $1 ${2:-}"; }
sha() { sha256sum < "$1" | cut -c1-64; }
# step NAME EST DESC "check|check|..."  -> returns 0 when the body must run.  In dry mode: prints and returns 1.
step() {
  if [ $DRY = 1 ]; then
    echo "[DRY] STEP $1 | est $2 | $3"; local c; IFS='|' read -ra _cs <<< "$4"; for c in "${_cs[@]}"; do echo "[DRY]     check: $c"; done; return 1
  fi
  if done_ $1; then log "SKIP $1 (already done)"; return 1; fi
  log "BEGIN $1 (est $2)"; return 0
}
wait_for() { # NAME...  block until every named step has its .done; abort when any step failed
  if [ $DRY = 1 ]; then echo "[DRY] WAIT for: $*"; return 0; fi
  local n; for n in "$@"; do until done_ $n; do [ -f $O/FAILED ] && fail "aborting while waiting for $n: $(cat $O/FAILED)"; sleep 20; done; done
}
track() { # NAME FUNCTION : run a function as a background track
  if [ $DRY = 1 ]; then echo "[DRY] TRACK $1 (background)"; "$2"; return 0; fi
  ( "$2" ) >> $O/track_$1.out 2>&1 < /dev/null &
  echo $! >> $O/track_pids
}
cgroup_cores() { local c q per; c="$(nproc)"; CORES_SRC="nproc (UNVERIFIED)"
  if { read -r q per < /sys/fs/cgroup/cpu.max; } 2> /dev/null && [ "$q" != "max" ] && [ "${per:-0}" -gt 0 ]; then c=$(( q / per > 0 ? q / per : 1 )); CORES_SRC="cgroup v2 cpu.max"
  elif { read -r q < /sys/fs/cgroup/cpu/cpu.cfs_quota_us; read -r per < /sys/fs/cgroup/cpu/cpu.cfs_period_us; } 2> /dev/null && [ "${q:-0}" -gt 0 ] && [ "${per:-0}" -gt 0 ]; then c=$(( q / per > 0 ? q / per : 1 )); CORES_SRC="cgroup v1 cfs"; fi
  echo "$c|$CORES_SRC"; }
# fetch URL OUT SIZE : authenticated, `curl -f` (an HTTP error body must never be saved as an archive), resumable, size-checked
fetch_file() {
  local url=$1 out=$2 want=$3 a sz
  for a in 1 2 3 4 5; do
    sz=$(stat -c %s "$out" 2> /dev/null || echo 0)
    [ "$sz" = "$want" ] && return 0
    [ "$sz" -gt "$want" ] && rm -f "$out"
    curl -fsSL --retry 5 --retry-all-errors -C - -H @$AUTH -o "$out" "$url" 2>> $O/fetch_err.log
    sz=$(stat -c %s "$out" 2> /dev/null || echo 0); [ "$sz" = "$want" ] && return 0
    log "ZZRS_RETRY fetch $(basename "$out") attempt $a size $sz/$want"; sleep $((30 * a))
  done; return 1
}
# zzv FILE KIND KEY : value of KEY on the last `ZZV <KIND> ...` line (keys matched independently: pod_helpers prints state= AFTER the base keys in apply mode)
zzv() { grep -a "^ZZV $2 " "$1" 2> /dev/null | tail -n 1 | tr ' ' '\n' | sed -n "s/^$3=//p" | head -n 1; }
blob_check() { # COMMIT REPOPATH LOCALFILE : the file on disk == the commit's blob
  [ "$($G hash-object --no-filters "$3")" = "$($G rev-parse "$1:$2")" ]; }
write_env() { # FILE with|without  -> the chain's start env (P6_D7_FALLBACK=0: a D7 BAD REFUSES, never quarantines)
  { echo P6_FREE_NAVLOGS=1; echo P6_YAW_WRAP=off; echo P6_DD_GB=16; echo P6_D7_FALLBACK=0; echo P6_LABEL_WORKERS=$LW
    [ "$PUSH_REPO" != "$P6REPO" ] && echo P6_REPO=$PUSH_REPO
    [ "$2" = with ] && echo P6_CBANK=$CB; true; } > "$1"; }

# ============================================================================================ steps run in the main flow
s_preflight() {
  step preflight "1 min" "tools, token file, cgroup cores, local-disk write test, GPU idle" \
    "token file present, non-empty, mode 600 (never printed)|curl tar zstd python3 git flock sha256sum nvidia-smi present (zstd installed via apt if absent)|cgroup CPU quota measured (never nproc): $NP twin workers|local disk $LOCAL: df >= ${LOCAL_NEED_GB} GB AND a real 20 GiB write (DBs 164 + pixels 133 + D9 tar 36 + temp)|no GPU compute process (never add load to a training pod)|rollback knobs consistent (P6_RESUME_REV needs an expected step)" || return 0
  [ -s <TOKEN_FILE> ] || fail "no <TOKEN_FILE> (the MM places it: pod-migration-checklist step 3)"
  chmod 600 <TOKEN_FILE>
  local t; for t in curl tar python3 git flock sha256sum nvidia-smi; do command -v $t > /dev/null || fail "missing tool $t"; done
  command -v zstd > /dev/null || { env -u HF_TOKEN apt-get install -y -q zstd >> $O/apt.log 2>&1; command -v zstd > /dev/null || fail "zstd missing and apt install failed (apt.log)"; }
  CG="$(cgroup_cores)"; log "cores ${CG%%|*} (${CG#*|}); twin workers $NP; label workers $LW (pinned: reproduces ARGS sha $ARGS_SHA)"
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null | wc -l)" = 0 ] || fail "a GPU compute process is running"
  if [ -n "$RESUME_REV" ] && [ -z "$EXPECT_STEP" ]; then fail "P6_RESUME_REV=$RESUME_REV needs P6_EXPECT_RESUME_STEP (the last ckpt step of that revision's metrics.jsonl): it is verified from the file, never trusted"; fi
  mkdir -p $LOCAL $DATA $P6/stage
  local fg; fg=$(df -Pk $LOCAL | awk 'NR==2{print int($4/1048576)}'); [ "$fg" -ge "$LOCAL_NEED_GB" ] || fail "local disk $LOCAL: df says ${fg} GB free < $LOCAL_NEED_GB"
  local n=0; while [ $n -lt 20 ]; do dd if=/dev/zero of=$LOCAL/zz_q_$n bs=1M count=1024 conv=fsync status=none 2> /dev/null || break; n=$((n+1)); done; rm -f $LOCAL/zz_q_*
  [ $n = 20 ] || fail "local write test: only $n GiB written to $LOCAL"
  mark preflight "local ${fg} GB free, 20 GiB write ok"
}
s_code() {
  step code "3-6 min" "git clone (treeless) + build the four code trees on the pod, every file checked against its commit blob" \
    "clone has $TIP (tip), $BCC (prebuild chain), $D7BASE (frozen D7 base)|tip package: SHIP_MANIFEST has $TIP_FILES files and tree sha256 starting $TIP_TREE12 (the identity P6 trained under)|G-SUITE extra files (eval/rls1{b,d,e}_arms.json, PREREG_R0E_PICK.md) == their commit blobs|d7fix tree: FIXED_FILES.sha256 6/6 OK|bank3c code/* == commit blobs; pod_helpers.py sha256 == $PODH_SHA" || return 0
  if [ ! -d $W/repo/.git ]; then rm -rf $W/repo; git clone -q --filter=tree:0 --no-checkout $GIT_URL $W/repo >> $O/git.log 2>&1 || fail "git clone (git.log)"; fi
  local c; for c in $TIP $BCC $D7BASE; do $G cat-file -e $c^{commit} 2>> $O/git.log || fail "commit $c not in the clone"; done
  build_pkg() { local c=$1 d=$2; [ -f $d/SHIP_MANIFEST.json ] && return 0; rm -rf $d; mkdir -p $d
    $G archive "$c:$R" code refe splits | tar -x -C $d || fail "git archive $c"
    ( cd $d/refe && python3 launch_gate.py ship-manifest --pkg $d --commit $c --git-dir $W/repo/.git --repo-prefix "$R/" --out $d/SHIP_MANIFEST.json ) > $O/ship_${c:0:7}.out 2>&1 || fail "ship-manifest $c (ship_${c:0:7}.out)"; }
  build_pkg $TIP $PKGT; build_pkg $BCC $PKGB
  python3 - "$PKGT/SHIP_MANIFEST.json" "$TIP_FILES" "$TIP_TREE12" <<'PY' || fail "tip SHIP_MANIFEST identity differs from the 2026-10-21 pod (files / tree sha256): not the code P6 trained under"
import json, sys
m = json.load(open(sys.argv[1])); n, t = len(m["files"]), m["tree_sha256"]
print("manifest files", n, "tree", t[:12], "commit", m["commit"][:12])
sys.exit(0 if (n == int(sys.argv[2]) and t.startswith(sys.argv[3])) else 1)
PY
  local f; for f in eval/rls1b_arms.json eval/rls1d_arms.json eval/rls1e_arms.json raw/2026-10-18-r0e-pick/PREREG_R0E_PICK.md; do      # what the gate's own G-SUITE reads outside refe/
    mkdir -p $PKGT/$(dirname $f); $G show "$TIP:$R/$f" > $PKGT/$f || fail "cannot read $f at the tip"; blob_check $TIP "$R/$f" $PKGT/$f || fail "blob mismatch: $f"
  done
  if [ ! -f $D7PKG/.done ]; then rm -rf $D7PKG; mkdir -p $D7PKG                                                                      # the frozen D7 tree the twin parts are built on
    $G archive "$D7BASE:$R" code refe splits | tar -x -C $D7PKG || fail "archive $D7BASE"
    $G archive "$TIP:$R/raw/2026-10-16-p6-d7-pdm-route-rank/code" refe | tar -x -C $D7PKG || fail "archive d7 fix files"
    $G show "$TIP:$R/raw/2026-10-16-p6-d7-pdm-route-rank/FIXED_FILES.sha256" > $D7PKG/FIXED_FILES.sha256 || fail "FIXED_FILES.sha256"
    ( cd $D7PKG && sha256sum -c FIXED_FILES.sha256 ) > $O/d7fix_sha.out 2>&1; [ "$(grep -c ': OK$' $O/d7fix_sha.out)" = 6 ] || fail "d7fix FIXED_FILES.sha256 is not 6/6 OK (d7fix_sha.out)"
    touch $D7PKG/.done; fi
  mkdir -p $W/ops/bank3c; $G archive "$TIP:$T" code run_full.sh MANIFEST.json | tar -x -C $W/ops/bank3c || fail "archive bank3c code"
  for f in $($G ls-tree --name-only "$TIP:$T/code"); do blob_check $TIP "$T/code/$f" $W/ops/bank3c/code/$f || fail "bank3c code blob mismatch: $f"; done
  $G show "$TIP:$T/post_merge/pod/pod_helpers.py" > $O/pod_helpers.py || fail "pod_helpers.py"
  [ "$(sha $O/pod_helpers.py)" = $PODH_SHA ] || fail "pod_helpers.py sha256 != $PODH_SHA"
  mark code "tip tree $TIP_TREE12 ($TIP_FILES files), bcc prebuild tree, d7fix 6/6, bank3c code, pod_helpers $PODH_SHA"
}
s_prebuild() {
  step prebuild "25-35 min (commands 19 MEASURED + D9 no-op build/verify ~5-10 ESTIMATED)" "bcc04b5 chain, P6_STOP_AFTER=cache, no P6_CBANK: mints commands.ok + cache.ok and rebuilds the pre-3C train_grow_cmd (not banked on HF) from the restored train_grow + NAVSIM metadata" \
    "chain.log ends 'stopping after cache (P6_STOP_AFTER)'|train_grow_cmd rank0 sha256 == daa23b96 and rank1 == 2f3e4613 (bit-identical to the old pod)|stage markers commands.ok and cache.ok exist|the D9 cache stage built NOTHING new (ZZMETRIC_CACHE_DONE built=0): the fetched cache is complete|resume point on disk (ckpt_last.pt + metrics.jsonl) verified after the chain's recover stage" || return 0
  write_env $W/ops/p6_prebuild.env without
  ( set -a; . $W/ops/p6_prebuild.env; set +a; export P6_PKG=$PKGB P6_STOP_AFTER=cache P6_EXPECT_PIX_SHARDS=0
    HF_TOKEN="$(cat <TOKEN_FILE>)" bash $PKGB/refe/p6_pod.sh ) >> $P6/p6_stdout.log 2>&1 < /dev/null
  tail -n 20 $P6/chain.log | grep -q "stopping after cache (P6_STOP_AFTER)" || fail "prebuild did not stop after cache: $(tail -n 1 $P6/chain.log | cut -c1-250)"
  [ "$(sha $NB/train_grow_cmd/targets_rank0.jsonl)" = $S0 ] || fail "train_grow_cmd rank0 sha != daa23b96"
  [ "$(sha $NB/train_grow_cmd/targets_rank1.jsonl)" = $S1IN ] || fail "train_grow_cmd rank1 sha != 2f3e4613"
  [ -f $P6/stage/commands.ok ] && [ -f $P6/stage/cache.ok ] || fail "commands.ok / cache.ok missing after the prebuild"
  local cl; cl="$(grep -h '^ZZMETRIC_CACHE_DONE' $P6/cache_attempt*.out 2> /dev/null | tail -n 1 | cut -c1-120)"
  case "$cl" in *"built=0 "*) ;; *) if [ "${RS_ALLOW_D9_BUILT:-0}" = 1 ]; then log "WARNING: the D9 cache stage built parts (RS_ALLOW_D9_BUILT=1): $cl"; else fail "the D9 cache stage built parts (the fetched cache is incomplete; RS_ALLOW_D9_BUILT=1 accepts it): $cl"; fi ;; esac
  verify_resume || fail "resume point wrong after the prebuild's recover stage"
  mark prebuild "train_grow_cmd rank0 daa23b96 rank1 2f3e4613; $cl"
}
verify_resume() { # the resume point, read from DISK (never trusted from a revision or a log)
  local sh=""; [ -f $O/rollback_report.json ] && sh="$(python3 -c "import json;print(json.load(open('$O/rollback_report.json'))['ckpt_sha256'])")"
  python3 $HERE/restage_helpers.py verify --run $RUN --expect-step "$EXPECT_STEP" ${sh:+--ckpt-sha256 $sh} >> $O/verify_resume.log 2>&1; local rc=$?; tail -n 1 $O/verify_resume.log; return $rc; }
twin_body() {
  local gen=$O/build_twin_parts_3c_pp.sh   # the 2026-10-22 fix: the route patch imports pdm_route_rank from refe/, so refe/ must be on PYTHONPATH too
  sed 's#^export PYTHONPATH=\$PKG/code:#export PYTHONPATH=$PKG/code:$PKG/refe:#' $W/ops/bank3c/code/build_twin_parts_3c.sh > $gen
  if [ "$TWIN_BANK" = 3c ]; then sed -i "s#^BANK=/workspace/data/refe_navtrain/train_grow_cmd\$#BANK=$CB#" $gen; fi   # opt-in speed-up (UNVERIFIED on a pod): only the 21,821 kept twins
  grep -q 'PKG/refe:' $gen || fail "twin script: the PYTHONPATH fix did not apply"
  [ "$TWIN_BANK" != 3c ] || grep -q "^BANK=$CB\$" $gen || fail "twin script: the 3c BANK patch did not apply"
  rm -f $F3/PARTS_DONE; WORKERS=$NP bash $gen > $O/parts.out 2>&1
  [ -f $F3/PARTS_DONE ] || fail "twin parts: $(tail -n 1 $O/parts.out | cut -c1-250)"
}
s_twins() {
  step twin_parts "31 min at 8 workers (the PI's figure; ~16 min with RS_TWIN_BANK=3c, UNVERIFIED)" "build the lane-rank twin parts into metric_cache_twins_3c on the frozen D7 tree (build_twin_parts_3c_pp.sh)" \
    "the script's own gate: every K=1,2,3 build ZZMETRIC_CACHE_DONE failed=0 and check_twin_parts_distinct ZZD7_DISTINCT_OK|PARTS_DONE written|no GPU process (script refuses otherwise)" || return 0
  twin_body; mark twin_parts "PARTS_DONE ($TWIN_BANK bank)"
}
s_link() {
  step twin_link "3-6 min" "POST_MERGE step (d): link the KEPT twins' parts into metric_cache, then the chain's OWN D7 check for K=1,2,3 -- BEFORE the chain starts" \
    "pod_helpers parts --mode plan: state=plan_ok, missing_src=0, missing_logged=0, conflict=0, distinct=$TWINS|metric_cache holds ZERO lane-rank files before the link|pod_helpers parts --mode apply --how link: state=ok, errors=0, copied=0, dst_missing_after=0 (keys matched independently)|p6_prep.py lane-ranks == '$RANKS_WANT'|p6_prep.py lane-distinct --bank CBANK --cache MC --k K --n 40 --min-frac 0.9 prints ZZD7_DISTINCT_OK for K=1,2,3" || return 0
  source $W/teacher_env.sh; local PY=$DRIVERL_EVAL_PYTHON k
  [ "$(find $MC -name '*__lane_rank_*.pkl.xz' | wc -l)" = 0 ] || [ -f $O/link_apply.out ] || fail "metric_cache already holds lane files that are not ours (wrong-route twins from an earlier attempt?)"
  ( cd $O && nice -n 10 $PY -I pod_helpers.py parts --bank $CB --src $TW --dst $MC --p6refe $PKGT/refe --mode plan ) > $O/link_plan.out 2>&1
  [ "$(zzv $O/link_plan.out parts state)" = plan_ok ] && [ "$(zzv $O/link_plan.out parts missing_src)" = 0 ] && [ "$(zzv $O/link_plan.out parts missing_logged)" = 0 ] \
    && [ "$(zzv $O/link_plan.out parts conflict)" = 0 ] && [ "$(zzv $O/link_plan.out parts distinct)" = $TWINS ] || fail "link plan: $(grep -a '^ZZV' $O/link_plan.out | cut -c1-300)"
  log "ZZRS_PLAN $(grep -a '^ZZV' $O/link_plan.out | cut -c1-250)"
  ( cd $O && nice -n 10 $PY -I pod_helpers.py parts --bank $CB --src $TW --dst $MC --p6refe $PKGT/refe --mode apply --how link --report $O/link_apply.json ) > $O/link_apply.out 2>&1
  [ "$(zzv $O/link_apply.out parts state)" = ok ] && [ "$(zzv $O/link_apply.out parts dst_missing_after)" = 0 ] && [ "$(zzv $O/link_apply.out parts errors)" = 0 ] \
    && [ "$(zzv $O/link_apply.out parts copied)" = 0 ] || fail "link apply: $(grep -a '^ZZV' $O/link_apply.out | cut -c1-300)"
  log "ZZRS_LINKED $(grep -a '^ZZV' $O/link_apply.out | cut -c1-250)"
  cd $PKGT/refe || fail "cd $PKGT/refe"
  local rk; rk="$(env -u HF_TOKEN $PY p6_prep.py lane-ranks --bank $CB | sed -n 's/^ZZLANE_RANKS//p' | xargs)"; [ "$rk" = "$RANKS_WANT" ] || fail "lane ranks of the 3-C bank are '$rk', expected '$RANKS_WANT'"
  for k in $RANKS_WANT; do
    env -u HF_TOKEN $PY p6_prep.py lane-distinct --bank $CB --cache $MC --k $k --n 40 --min-frac 0.9 > $O/distinct_$k.out 2>&1
    grep -q "^ZZD7_DISTINCT_OK " $O/distinct_$k.out || fail "D7 rank $k: $(tail -n 1 $O/distinct_$k.out | cut -c1-200)"
    log "ZZRS_D7 $(tail -n 1 $O/distinct_$k.out | cut -c1-160)"
  done
  mark twin_link "$TWINS twins linked, D7 distinct OK for K=1,2,3"
}
s_final() {
  if [ $DRY = 1 ]; then
    echo "[DRY] STEP final_chain | est ~45 min to training (MEASURED 2026-10-22: lab 2.7 + mem 8.0 + egr 0.5 + launch gate 23.8 + preflight 1.6 min) | the tip chain with P6_CBANK, P6_D7_FALLBACK=0"
    for c in "start env pins P6_CBANK=$CB and P6_D7_FALLBACK=0 (grep -x)|all 32 pixel shards present (.done_N)|auth header file removed before the chain|resume point re-verified from disk (ckpt_last.pt + last ckpt event step == $EXPECT_STEP${RESUME_REV:+, rollback rev $RESUME_REV})|no GPU process|coverage guard (background): first labeller events after launch have no_parts == 0, else ALERT_COVERAGE.txt (training is NOT killed)|ARGS sha $ARGS_SHA appears in chain.log (else ALERT_ARGS.txt)"; do IFS='|' read -ra _cs <<< "$c"; for c in "${_cs[@]}"; do echo "[DRY]     check: $c"; done; done; return 0
  fi
  wait_for pix
  rm -f $AUTH
  write_env $W/ops/p6_start.env with
  grep -qx "P6_CBANK=$CB" $W/ops/p6_start.env && grep -qx 'P6_D7_FALLBACK=0' $W/ops/p6_start.env || fail "p6_start.env does not pin P6_CBANK and P6_D7_FALLBACK=0"
  verify_resume || fail "resume point wrong right before the final chain"
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null | wc -l)" = 0 ] || fail "a GPU process is running before the final chain"
  local off coff; off=$(stat -c %s $RUN/metrics.jsonl); coff=$(stat -c %s $P6/chain.log 2> /dev/null || echo 0)
  log "ZZRS_FINAL_CHAIN launching (tip $TIP, p6_start.env: P6_D7_FALLBACK=0, P6_CBANK=3-C${RESUME_REV:+, ROLLBACK rev $RESUME_REV step $EXPECT_STEP, pushes to $PUSH_REPO})"
  echo "$(date -u +%FT%TZ) restage_v2: final chain, env $W/ops/p6_start.env, resume point step $EXPECT_STEP" >> $P6/p6_stdout.log
  python3 $HERE/restage_helpers.py guard --metrics $RUN/metrics.jsonl --offset "$off" --wait-s 14400 --lines 5 --alert $O/ALERT_COVERAGE.txt >> $O/guard.log 2>&1 &
  local GPID=$!
  ( for i in $(seq 1 720); do l="$(tail -c +$((coff + 1)) $P6/chain.log 2> /dev/null | grep -m1 'ARGS sha')"; [ -n "$l" ] && break; sleep 10; done
    case "$l" in *"ARGS sha $ARGS_SHA"*) log "ZZRS_ARGS_OK $l" ;; *) echo "ARGS sha line: '$l' (expected $ARGS_SHA)" > $O/ALERT_ARGS.txt; log "ZZRS_ALERT ARGS sha differs from $ARGS_SHA: the trainer's own preflight will refuse a changed identity" ;; esac ) &
  ( set -a; . $W/ops/p6_start.env; set +a
    HF_TOKEN="$(cat <TOKEN_FILE>)" bash $PKGT/refe/p6_pod.sh ) >> $P6/p6_stdout.log 2>&1 < /dev/null
  local rc=$?; kill $GPID 2> /dev/null
  [ -f $O/ALERT_COVERAGE.txt ] && cp $O/ALERT_COVERAGE.txt $P6/ALERT_COVERAGE.txt
  log "ZZRS_FINAL_CHAIN_EXIT rc $rc last: $(tail -n 1 $P6/chain.log | cut -c1-200)"
}

# ============================================================================================ background tracks
track_env() {
  step teacher_env "15-25 min (ESTIMATED; the old log has no timestamps)" "build venv-teacher (python 3.11, pinned deps, torch cu128) + nuPlan maps via the tip's pod_teacher_env.sh" \
    "log prints 'ZZOK teacher env ready'|/workspace/teacher_env.sh exists|real conv2d on CUDA and imports from the shipped checkout (the script's own checks)" || return 0
  wait_for code
  ( export PKG=$PKGT; env -u HF_TOKEN bash $PKGT/code/pod_teacher_env.sh ) > $P6/teacher_env.log 2>&1
  grep -q '^ZZOK teacher env ready' $P6/teacher_env.log && [ -f $W/teacher_env.sh ] || fail "teacher env: $(tail -n 1 $P6/teacher_env.log | cut -c1-200)"
  mark teacher_env
}
track_dbs() {
  step dbs "30-60 min (ESTIMATED; bandwidth-bound, the longest fetch)" "1,192 nuPlan DBs range-fetched from the public nuPlan S3 zips onto the LOCAL disk; $DATA/navtrain_dbs is a symlink" \
    "dbs.log: fetch tool exits 0 (it CRC-verifies every DB)|ls navtrain_dbs/*.db >= 1192 through the symlink" || return 0
  wait_for code
  local D=$DATA/navtrain_dbs
  if [ -L $D ]; then :; elif [ -d $D ] && [ "$(ls $D/*.db 2> /dev/null | wc -l)" -gt 0 ]; then log "DBs: a real directory with DBs exists: used in place"; else rmdir $D 2> /dev/null; mkdir -p $LOCAL/navtrain_dbs; ln -s $LOCAL/navtrain_dbs $D; fi
  export NUPLAN_DATA_ROOT=$DATA REFE_NUPLAN_DB_ROOT=$D REFE_NAVTRAIN_YAML=$PKGT/splits/navtrain.yaml REFE_BACKBONE_ROOT=$DATA/backbones REFE_CD_CACHE=$DATA/nuplan_cdcache PYTHONUNBUFFERED=1
  local PYX=python3
  ( cd $PKGT/code && timeout 300 python3 fetch_navtrain_dbs.py --plan ) > $O/dbs_plan_sys.out 2>&1 || { log "DBs: system python3 cannot run the fetch tool: waiting for the teacher env"; wait_for teacher_env; PYX="$(source $W/teacher_env.sh; echo $DRIVERL_EVAL_PYTHON)"; }
  local a; for a in 1 2 3; do
    ( cd $PKGT/code && $PYX fetch_navtrain_dbs.py --out $D --held-root /nonexistent --jobs $DB_JOBS ) >> $O/dbs.log 2>&1 && [ "$(ls $D/*.db | wc -l)" -ge 1192 ] && break
    log "ZZRS_RETRY dbs attempt $a: $(ls $D/*.db 2> /dev/null | wc -l) DBs"; sleep 60
  done
  [ "$(ls $D/*.db | wc -l)" -ge 1192 ] || fail "DBs: $(ls $D/*.db 2> /dev/null | wc -l)/1192 (dbs.log)"
  mark dbs "$(ls $D/*.db | wc -l) DBs"
}
pix_one() {
  local n=$1 PIX=$DATA/navtrain_pixels a tgz HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/navsim
  [ -f $PIX/.done_$n ] && return 0
  for a in 1 2 3 4 5; do
    mkdir -p $LOCAL/tgz; tgz=$LOCAL/tgz/navtrain_current_$n.tgz; rm -f "$tgz"
    if curl -fsSL --retry 5 --retry-all-errors -H @$AUTH -o "$tgz" "$HF/navtrain_current_$n.tgz" 2>> $O/pix_err.log \
       && tar -xzf "$tgz" --no-same-owner -C $PIX --wildcards '*/CAM_F0/*' '*/CAM_B0/*' '*/CAM_L0/*' '*/CAM_R0/*' 2>> $O/pix_err.log; then
      touch $PIX/.done_$n; rm -f "$tgz"; echo "ZZPIX_OK $n a$a $(date -u +%T)" >> $O/pix.log; return 0
    fi
    echo "ZZPIX_RETRY $n a$a $(date -u +%T) $(stat -c %s "$tgz" 2> /dev/null)" >> $O/pix.log; rm -f "$tgz"; sleep $((60 * a))
  done
  echo "ZZPIX_FAIL $n" >> $O/pix.log; return 1
}
track_pix() {
  step pix "30-60 min (ESTIMATED; overlaps everything, needed only at the end)" "32 pixel shards (4 cameras) from the public OpenScene HF repo, authenticated, curl -f, onto the LOCAL disk; $DATA/navtrain_pixels is a symlink" \
    "every shard: curl -f + tar extract of CAM_F0/B0/L0/R0 succeeded, then .done_N|32 .done_N markers through the symlink" || return 0
  local P=$DATA/navtrain_pixels
  if [ -L $P ] || [ -d $P ]; then :; else mkdir -p $LOCAL/navtrain_pixels; ln -s $LOCAL/navtrain_pixels $P; fi
  for n in $(seq 1 32); do [ -f $P/.done_$n ] || echo $n; done > $O/pix_todo.txt
  export -f pix_one log; export DATA LOCAL O AUTH DRY
  xargs -a $O/pix_todo.txt -P $PIX_PAR -I{} bash -c 'pix_one {}'
  local c; c=$(ls $P/.done_* 2> /dev/null | wc -l); [ "$c" -ge 32 ] || fail "pixels $c/32 (pix.log, pix_err.log)"
  mark pix "32/32 shards"
}
track_quota() {
  step quota "3-5 min" "real write probe of /workspace (never df): the volume quota is invisible to df" "a ${WS_NEED_GB} GiB write in 1 GiB pieces succeeds (then deleted)" || return 0
  local n=0; : > $DATA/.restage_quota
  while [ $n -lt $WS_NEED_GB ]; do dd if=/dev/zero of=$DATA/.restage_quota bs=1M count=1024 oflag=append conv=notrunc,fsync status=none 2>> $O/quota.err || break; n=$((n + 1)); done
  rm -f $DATA/.restage_quota; [ $n -ge $WS_NEED_GB ] || fail "/workspace quota: only $n GiB writable, want $WS_NEED_GB (the volume is too small: resize it, or keep more on the local disk)"
  mark quota "$n GiB written"
}
track_d9() {
  step d9 "10-18 min (ESTIMATED: 36 GB at >=100 MB/s + sha256 + extract of 107,860 files)" "FETCH the D9 metric-cache tar (instead of a 5.2 CPU-h rebuild), verify, extract into refe_navtrain/metric_cache" \
    "curl -f; size == $D9_SIZE and sha256 == $D9_SHA (HF LFS oid)|first tar member is 'metric_cache/' (layout: metric_cache/<log>/<token>.pkl.xz, measured by HTTP Range)|extracted file count == $D9_FILES|ZERO lane-rank files in the extracted cache (no wrong-route twins)" || return 0
  local f=$LOCAL/$D9_NAME; mkdir -p $NB
  fetch_file "$BANKS_URL/$D9_NAME" $f $D9_SIZE || fail "D9 tar fetch (fetch_err.log)"
  [ "$(sha $f)" = $D9_SHA ] || fail "D9 tar sha256 mismatch"
  [ "$(tar -tf $f | head -n 1)" = "metric_cache/" ] || fail "D9 tar layout: first member is not metric_cache/"
  wait_for quota
  nice -n 10 tar -xf $f -C $NB --no-same-owner || fail "D9 tar extract"
  local n; n=$(find $MC -type f | wc -l); [ "$n" = "$D9_FILES" ] || fail "D9 extracted $n files, expected $D9_FILES"
  [ "$(find $MC -name '*__lane_rank_*' | wc -l)" = 0 ] || fail "the D9 tar holds lane-rank files (wrong-route twins): not the clean cache"
  rm -f $f; mark d9 "$n files"
}
track_3c() {
  step bank3c "1-2 min" "FETCH the 3-C twin bank (instead of the 23 h re-roll), verify, extract into refe_navtrain/train_grow_cmd_3c" \
    "curl -f; size == $C3_SIZE and sha256 == $C3_SHA|layout: train_grow_cmd_3c/{merge_report.json,targets_rank0.jsonl,...} (first members read by HTTP Range)|targets_rank0.jsonl sha256 == daa23b96... and targets_rank1.jsonl == 2a63acc2... (the bank P6 trained on)" || return 0
  local f=$LOCAL/$C3_NAME; mkdir -p $NB
  fetch_file "$BANKS_URL/$C3_NAME" $f $C3_SIZE || fail "3-C bank fetch (fetch_err.log)"
  [ "$(sha $f)" = $C3_SHA ] || fail "3-C bank sha256 mismatch"
  [ "$(zstd -dc $f | tar -tf - | head -n 1)" = "train_grow_cmd_3c/" ] || fail "3-C bank layout: first member is not train_grow_cmd_3c/"
  zstd -dc $f | tar -x --no-same-owner -C $NB || fail "3-C bank extract"
  [ "$(sha $CB/targets_rank0.jsonl)" = $S0 ] || fail "3-C rank0 sha != daa23b96"
  [ "$(sha $CB/targets_rank1.jsonl)" = $S1OUT ] || fail "3-C rank1 sha != 2a63acc2 -- NOT the bank P6 trained on; no resume"
  [ -f $CB/merge_report.json ] || fail "3-C bank: merge_report.json missing"
  rm -f $f; mark bank3c "rank1 2a63acc2 rank0 daa23b96"
}
track_restore() {
  step restore "~4 min MEASURED (2026-10-22: 4.4 GB download + 17.6 GB extract)" "the chain's own bank restore (p2r_sync.py restore): train_grow (pre-3C bank, calib_table.json), on-policy sets, model_final; md5-pinned" \
    "ZZRESTORE_OK (md5 of each archive == the pins in p2r_sync.py)|train_grow/calib_table.json non-empty and targets_rank*.jsonl present (the chain's have_bank)" || return 0
  wait_for teacher_env
  ( source $W/teacher_env.sh; cd $PKGT/refe && HF_TOKEN="$(cat <TOKEN_FILE>)" $DRIVERL_EVAL_PYTHON p2r_sync.py restore --work $DATA ) > $P6/restore.log 2>&1
  grep -q '^ZZRESTORE_OK' $P6/restore.log || fail "restore: $(tail -n 1 $P6/restore.log | cut -c1-200)"
  [ -s $NB/train_grow/calib_table.json ] && ls $NB/train_grow/targets_rank*.jsonl > /dev/null 2>&1 || fail "restore: train_grow/calib_table.json or targets_rank*.jsonl missing"
  mark restore
}
track_rollback() {
  [ -n "$RESUME_REV" ] || { [ $DRY = 1 ] && echo "[DRY] STEP rollback | skipped: P6_RESUME_REV unset (resume from HF main = latest, ckpt step $EXPECT_STEP, via the chain's recover stage)"; return 0; }
  step rollback "2-4 min (700 MB)" "ROLLBACK: place run/{ckpt_last.pt,config.json,metrics.jsonl,snap_epoch001..006.pt} of HF revision $RESUME_REV into the run dir BEFORE any chain runs" \
    "every file exists at the revision; sizes; sha256 of the .pt files == HF LFS oid|the DOWNLOADED metrics.jsonl's last 'event: ckpt' has step == $EXPECT_STEP (read from the file)|config.json byte-equal to main's|an existing different ckpt_last.pt is refused|recover cannot overwrite: p2r_sync.recover restores ONLY when run/ckpt_last.pt is absent|nothing in the HF repo is modified (optional tag on main's head); pushes go to $PUSH_REPO" || return 0
  local tag=""; [ "$PUSH_REPO" = "$P6REPO" ] && [ "${P6_RESUME_TAG:-1}" != 0 ] && tag="--tag pre-rollback-$(date -u +%Y%m%dT%H%M)"
  python3 $HERE/restage_helpers.py place --repo $P6REPO --rev $RESUME_REV --run $RUN --expect-step $EXPECT_STEP --report $O/rollback_report.json $tag > $O/rollback.out 2>&1
  grep -q '^ZZREV_PLACED' $O/rollback.out || fail "rollback: $(tail -n 1 $O/rollback.out | cut -c1-250)"
  verify_resume || fail "rollback: the placed resume point does not verify from disk"
  mark rollback "$(grep '^ZZREV_PLACED' $O/rollback.out | cut -c1-200)"
}
twin_track() { # RS_TWIN_BANK=3c only: the twins need nothing from the prebuild
  step twin_parts "(see s_twins)" "twin parts on the kept bank, overlapped with the prebuild" "as s_twins" || return 0
  wait_for code teacher_env dbs d9 bank3c; [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null | wc -l)" = 0 ] || fail "GPU busy"
  twin_body; mark twin_parts "PARTS_DONE (3c bank, overlapped)"
}

# ============================================================================================ main
if [ $DRY = 0 ]; then
  mkdir -p $O; rm -f $O/FAILED $O/track_pids
  [ "$(id -u)" = 0 ] || fail "run as root (the token file and the pod layout assume it)"
  exec 9> $O/driver.lock; flock -n 9 || { echo "ZZRS_REFUSED another restage_v2.sh holds $O/driver.lock"; exit 1; }
  trap 'rm -f $AUTH' EXIT
  log "driver start pid $$ (resume: ${RESUME_REV:-main/latest} step $EXPECT_STEP; twin bank $TWIN_BANK; push repo $PUSH_REPO)"
fi
[ $DRY = 1 ] && log "DRY RUN: nothing below touches the network or the disk (resume ${RESUME_REV:-main/latest}, expected step $EXPECT_STEP, push repo $PUSH_REPO, twin bank $TWIN_BANK)"
s_preflight
if [ $DRY = 0 ]; then   # EVERY run (a re-run skips preflight, and the header file is deleted at exit): the curl header file, mode 600, never printed
  [ -s <TOKEN_FILE> ] || fail "no <TOKEN_FILE>"; mkdir -p $LOCAL $DATA $P6/stage
  ( umask 077; printf 'Authorization: Bearer %s\n' "$(cat <TOKEN_FILE>)" > $AUTH )
fi
s_code
track env track_env; track dbs track_dbs; track pix track_pix; track quota track_quota; track d9 track_d9; track bank3c track_3c; track restore track_restore; track rollback track_rollback
[ "$TWIN_BANK" = 3c ] && track twin twin_track
wait_for teacher_env restore dbs d9 bank3c ${RESUME_REV:+rollback}
s_prebuild
if [ "$TWIN_BANK" = 3c ]; then wait_for twin_parts; else s_twins; fi
s_link
s_final
