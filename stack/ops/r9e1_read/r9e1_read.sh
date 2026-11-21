#!/usr/bin/env bash
# r9e1_read.sh -- THOR-SIDE automatic read of the refcv9 E1 floor arms C0 and C1.  E1-READ, 2026-11-03.
#
#   WAIT   a FILE-ONLY poll (every R9E1_POLL_S = 60 s: no GPU, no torch, python stdlib only via r9e1_read_state.py) of the chain's
#          $R/gate3/CHAIN_STATUS.txt until C0 AND C1 are done (the chain's own rule: <run>/summary.json "done": true + "zerograd_gate",
#          no ZEROGRAD_FAIL.json) -- or a C0/C1 FAILED / ZEROGRAD-FAIL / REFUSED / NOT-LAUNCHED line, or ALLDONE without both:
#          then READ_ABORTED (the offending chain line verbatim) and exit.   Only lines after the LAST `START` are read.
#   LOCK   take /home/nvidia/refcv7_post/thor_gpu.lock (flock, the chain's discipline; the chain frees it right after ALLDONE),
#          confirm no compute app on the GPU.
#   DUMP   r9_e1_dump.py on each arm's FINAL EMA checkpoint (<run>/model_ema.pt) --require-ema, --split dev of E0i's DEV_HOLD_SPLIT.json
#          (sha256 checked against its .sha256 file), C0 then C1.
#   EVAL   r9_e1_eval.py per arm (+ the rasters), and once PAIRED (--packs C1 --packs-b C0): CPU, CUDA_VISIBLE_DEVICES empty.
#   FLOOR  r9_e1_floor.py -> FLOOR.json + FLOOR.md in the read dir; then ZZR9E1READ-DONEZZ and the lock is released.
# Every verdict is read from an ARTIFACT (dump.json / EVAL_*.json / FLOOR.json status), never from $?.  Resumable: a step whose
# artifact exists is skipped.  No model turns poll this: it is plain bash.
# Logs: $RD (= $R/read) READ_STATUS.txt (append-only events), ALIVE (last poll), logs/*.out|err; opaque markers ZZR9E1READ-<EVENT>ZZ
# also go to stdout.   Detach:  nohup setsid bash r9e1_read.sh > $RD/r9e1_read.out 2>&1 < /dev/null &
#
# DRY RUN (dev box, no GPU, no flock): R9E1_DRY=1 + R9E1_ROOT / R9E1_READ_DIR / R9E1_TREE pointing at SYNTHETIC dirs,
#   R9E1_DUMP_SCRIPT=<r9e1_read_stub_dump.py>, R9E1_PYTHONPATH, R9E1_PY, R9E1_GATE_ARGS="--gate 0.5", R9E1_SPLIT, R9E1_SPLIT_SHA_FILE,
#   R9E1_BOOT, R9E1_POLL_S.  The GPU step (dump) is the stub; the GPU lock is emulated with mkdir; everything else is the real code.
# TRAPS: no `pgrep -f` on a pattern this command line contains; detach with `nohup setsid ... < /dev/null &`; ssh with `-n`.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="${R9E1_ROOT:-/home/nvidia/r8ns/r9e1}"
RD="${R9E1_READ_DIR:-$R/read}"
STAT="${R9E1_STATUS:-$R/gate3/CHAIN_STATUS.txt}"
T="${R9E1_TREE:-$R/launch_tree}"
PY="${R9E1_PY:-/home/nvidia/venvs/tanitad-train/bin/python}"
LOCKF="${R9E1_LOCKF:-/home/nvidia/refcv7_post/thor_gpu.lock}"
POLL_S="${R9E1_POLL_S:-60}"
MAX_WAIT_S="${R9E1_MAX_WAIT_S:-1209600}"          # 14 days: a wedged chain must not leave a waiter forever
LOCK_MAX_S="${R9E1_LOCK_MAX_S:-7200}"
DRY="${R9E1_DRY:-0}"
BOOT="${R9E1_BOOT:-2000}"
SEED=0
SPLIT="${R9E1_SPLIT:-/home/nvidia/r8ns/r9dump/DEV_HOLD_SPLIT.json}"
SPLIT_SHA_FILE="${R9E1_SPLIT_SHA_FILE:-$HERE/DEV_HOLD_SPLIT.sha256}"
SPLIT_SHA_PIN="ca5af0203320aba4f89e09be81a6bf079d9d4d77708ccb8a3ca2a0ea57e8db0a"     # E0i's DEV_HOLD_SPLIT.sha256 (repo, tip 6af1611)
S2="${R9E1_S2:-/home/nvidia/r8ns/perc8/inputs/s2_windows.json}"
GATES="${R9E1_GATES:-/home/nvidia/r8ns/perc8/out/T_nosync/refcv8_det_presence_gates_train.json}"
GATES_SHA="f4cb9a94f1dc3208599438e6748d19dfd12a0261dd7e476a8934b7ae53b94611"         # the refcv8 TRAIN-fit P=R gates (box3d 0.2660), same for both arms
STATE_PY="${R9E1_STATE_PY:-$HERE/r9e1_read_state.py}"
DUMP_CWD="${R9E1_DUMP_CWD:-$T/stack}"
DUMP_SCRIPT="${R9E1_DUMP_SCRIPT:-scripts/r9_e1_dump.py}"
EVAL_SCRIPT="${R9E1_EVAL_SCRIPT:-$T/stack/scripts/r9_e1_eval.py}"
if [ -n "${R9E1_FLOOR_SCRIPT:-}" ]; then FLOOR_SCRIPT="$R9E1_FLOOR_SCRIPT"
elif [ -f "$HERE/r9_e1_floor.py" ]; then FLOOR_SCRIPT="$HERE/r9_e1_floor.py"
else FLOOR_SCRIPT="$HERE/../../scripts/r9_e1_floor.py"; fi                        # the repo layout (stack/ops/r9e1_read -> stack/scripts)
PYPATH="${R9E1_PYTHONPATH:-$T/stack:$T/stack/scripts:$T/taniteval}"
command -v flock >/dev/null 2>&1 && HAVE_FLOCK=1 || HAVE_FLOCK=0
mkdir -p "$RD/logs"
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$RD/READ_STATUS.txt"; echo "ZZR9E1READ-$1ZZ $(date -u +%H:%M:%SZ) ${*:2}"; }

LOCK_HELD=0
take_lock() {      # $1 = max seconds -> 0 when held
  if [ "$HAVE_FLOCK" = 1 ]; then
    exec 9>"$LOCKF" || return 1
    flock -w "$1" 9 || { exec 9>&-; return 1; }
  else             # dry run on a box with no flock (Git bash): an mkdir lock with the same take / release order
    local t0; t0=$(date +%s)
    while ! mkdir "$LOCKF.d" 2>/dev/null; do [ $(( $(date +%s) - t0 )) -gt "$1" ] && return 1; sleep 1; done
  fi
  LOCK_HELD=1; return 0
}
release_lock() {
  [ "$LOCK_HELD" = 1 ] || return 0
  if [ "$HAVE_FLOCK" = 1 ]; then exec 9>&-; else rmdir "$LOCKF.d" 2>/dev/null; fi
  LOCK_HELD=0
}
abort() {          # $1 = TAG, $2 = detail (for a chain event: the chain line, verbatim), $3 = extra
  printf '%s %s\n%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "${2:-}" "${3:-}" > "$RD/READ_ABORTED"
  ev ABORTED "$1 -- ${2:-} ${3:-}"
  release_lock
  exit 3
}

if [ "$HAVE_FLOCK" = 0 ] && [ "$DRY" != 1 ]; then echo "[r9e1_read] no flock: refusing a real run without the GPU-lock discipline" >&2; exit 4; fi
if [ "$HAVE_FLOCK" = 1 ]; then          # one reader only (a second start exits quietly)
  exec 8>"$RD/instance.lock"
  flock -n 8 || { echo "ZZR9E1READ-ALREADYRUNNINGZZ another r9e1_read holds $RD/instance.lock"; exit 0; }
fi
echo $$ > "$RD/r9e1_read.pid"
if [ -e "$RD/READ_DONE" ]; then echo "ZZR9E1READ-ALREADYDONEZZ $RD/READ_DONE exists (remove it to re-read)"; exit 0; fi
rm -f "$RD/READ_ABORTED"
ev START "pid $$ dry=$DRY status=$STAT root=$R poll=${POLL_S}s"

# ---- WAIT: file-only ------------------------------------------------------------------------------------------------------------
t0=$(date +%s); lastsig=""
while :; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RD/ALIVE"
  S_STATE=ERROR; S_SIG=ERROR; S_REASON=""; S_LINE=""; S_C0_RUN=""; S_C1_RUN=""
  if "$PY" -I "$STATE_PY" --status "$STAT" --root "$R" --out "$RD/state.env" > "$RD/state.out" 2> "$RD/state.err"; then
    . "$RD/state.env"
  else
    S_REASON="state helper failed: $(tail -c 300 "$RD/state.err" 2>/dev/null | tr '\n' ' ')"
  fi
  if [ "$S_SIG" != "$lastsig" ]; then ev "$S_STATE" "$S_REASON"; lastsig="$S_SIG"; fi
  case "$S_STATE" in
    READY) break;;
    ABORT) abort CHAIN "$S_LINE" "$S_REASON";;
  esac
  [ $(( $(date +%s) - t0 )) -gt "$MAX_WAIT_S" ] && abort WAIT-TIMEOUT "no C0+C1 completion in ${MAX_WAIT_S}s" "$S_REASON"
  sleep "$POLL_S"
done
C0_RUN="$S_C0_RUN"; C1_RUN="$S_C1_RUN"

# ---- inputs, checked BEFORE the GPU lock is taken --------------------------------------------------------------------------------
[ -s "$SPLIT" ] || abort NO-SPLIT "$SPLIT is missing"
if [ -s "$SPLIT_SHA_FILE" ]; then EXP="$(awk '{print $1; exit}' "$SPLIT_SHA_FILE")"; else EXP="$SPLIT_SHA_PIN"; fi
GOT="$(sha256sum "$SPLIT" | cut -d' ' -f1)"
[ "$GOT" = "$EXP" ] || abort SPLIT-SHA "$SPLIT sha256 $GOT != expected $EXP (source: ${SPLIT_SHA_FILE:-pin})"
if [ -n "${R9E1_GATE_ARGS:-}" ]; then GATEARGS="$R9E1_GATE_ARGS"; else
  [ -s "$GATES" ] || abort NO-GATES "$GATES is missing"
  GG="$(sha256sum "$GATES" | cut -d' ' -f1)"; [ "$GG" = "$GATES_SHA" ] || abort GATES-SHA "$GATES sha256 $GG != pinned $GATES_SHA"
  GATEARGS="--gates-json $GATES"
fi
[ -s "$S2" ] || abort NO-S2 "$S2 is missing"
for f in "$EVAL_SCRIPT" "$FLOOR_SCRIPT" "$STATE_PY"; do [ -s "$f" ] || abort NO-SCRIPT "$f is missing"; done
ev INPUTS-OK "split sha256 ${GOT:0:12} matches ${SPLIT_SHA_FILE##*/}; gate args: $GATEARGS"

penv() {           # $1 = PYTHONPATH, rest = the command
  local pp="$1"; shift
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u REFCV7_KIT -u REFCV7_REMAP_OVERRIDES -u REFCV7_REPO \
  PYTHONPATH="$pp" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  HF_TOKEN_PATH=/nonexistent/hf_token REFCV6_KIT=/home/nvidia REFCV6_REMAP_OVERRIDES="$T/stack/ops/refcv8_eval_remap_a2_thor.json" "$@"; }

# Thor's training venv has numpy 2.5 + torch but NO scipy (MEASURED 2026-11-03: no scipy under venvs/tanitad-train/lib; the system
# python3's scipy 1.11 is built for numpy 1.26), and r9_e1_eval's raster path needs scipy.ndimage.binary_dilation.  The CPU steps
# (eval, floor) therefore get shim_scipy/ on PYTHONPATH -- only when the real scipy cannot be imported (never ahead of a working one).
PYPATH_CPU="$PYPATH"
if [ "${R9E1_FORCE_SCIPY_SHIM:-0}" = 1 ] || ! penv "$PYPATH" "$PY" -c 'import scipy.ndimage' > /dev/null 2>&1; then
  [ -f "$HERE/shim_scipy/scipy/ndimage/__init__.py" ] || abort NO-SCIPY "the venv has no scipy and $HERE/shim_scipy is missing"
  SHIMD="$HERE/shim_scipy"; command -v cygpath >/dev/null 2>&1 && SHIMD="$(cygpath -m "$SHIMD")"          # a native-python path on Git bash (dry run)
  PYPATH_CPU="$PYPATH${R9E1_PATHSEP:-:}$SHIMD"
  ev SCIPY-SHIM "eval/floor use $HERE/shim_scipy (numpy binary_dilation, held equal to scipy by test)"
fi

# ---- LOCK ----------------------------------------------------------------------------------------------------------------------------
ev LOCK-WAIT "taking the Thor GPU lock $LOCKF (<= ${LOCK_MAX_S}s; the chain frees it right after ALLDONE)"
take_lock "$LOCK_MAX_S" || abort LOCK-TIMEOUT "the Thor GPU lock stayed held for ${LOCK_MAX_S}s; nothing was run"
ev LOCK-TAKEN "$LOCKF"
if [ "$DRY" = 1 ]; then ev GPU-CHECK "skipped (dry run)"; else
  NA=1; for _i in $(seq 1 20); do NA="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')"; [ "$NA" = 0 ] && break; sleep 15; done
  [ "$NA" = 0 ] || abort GPU-NOT-EMPTY "$NA compute app(s) under the lock after 5 min; never share the GPU"
  ev GPU-CHECK "no compute app"
fi

# ---- DUMP (the only GPU step) ------------------------------------------------------------------------------------------------------
dump_arm() {       # $1 = C0|C1, $2 = run dir
  local A="$1" RUN="$2" O="$RD/dump_$1" rc
  if [ -s "$O/dump.json" ]; then ev DUMP-SKIP "$A: $O/dump.json exists"; return 0; fi
  [ -s "$RUN/model_ema.pt" ] && [ -s "$RUN/config.json" ] || abort NO-CHECKPOINT "$A: $RUN/model_ema.pt or $RUN/config.json is missing"
  rm -rf "$O"; ev DUMP-START "$A ckpt $RUN/model_ema.pt"
  ( cd "$DUMP_CWD" && penv "$PYPATH" "$PY" -X utf8 "$DUMP_SCRIPT" --ckpt "$RUN/model_ema.pt" --config "$RUN/config.json" --windows "$SPLIT" \
      --split dev --s2-windows "$S2" --require-ema --out "$O" > "$RD/logs/dump_$A.out" 2> "$RD/logs/dump_$A.err" ); rc=$?
  [ -s "$O/dump.json" ] || abort DUMP-FAILED "$A rc=$rc (read $RD/logs/dump_$A.err)"
  ev DUMP-DONE "$A rc=$rc $O"
}
dump_arm C0 "$C0_RUN"
dump_arm C1 "$C1_RUN"
"$PY" -I - "$RD/dump_C0/dump.json" "$RD/dump_C1/dump.json" > "$RD/logs/dump_check.out" 2>&1 <<'PYEOF'
import json, sys
a, b = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:3])
bad = []
for k in ("order_digest", "n_windows", "n_clips", "step"):
    if a.get(k) != b.get(k):
        bad.append("%s differs: C0 %r vs C1 %r" % (k, a.get(k), b.get(k)))
for n, d in (("C0", a), ("C1", b)):
    if d.get("weights_are_ema") is not True or d.get("ckpt_kind") != "ema_model_only":
        bad.append("%s dump: weights_are_ema=%r ckpt_kind=%r (need an EMA file)" % (n, d.get("weights_are_ema"), d.get("ckpt_kind")))
    if d.get("SMOKE_RESTRICTED") not in (False, None):
        bad.append("%s dump is SMOKE_RESTRICTED" % n)
print("; ".join(bad) or "OK")
sys.exit(1 if bad else 0)
PYEOF
[ $? = 0 ] || abort DUMP-MISMATCH "the two dumps are not one read" "$(cat "$RD/logs/dump_check.out")"
ev DUMP-CHECK "C0 and C1 dumps: same windows (order_digest), both EMA"

# ---- EVAL (CPU) ----------------------------------------------------------------------------------------------------------------------
run_eval() {       # $1 = tag, $2 = out json, rest = r9_e1_eval.py args
  local tag="$1" outj="$2" rc; shift 2
  if [ -s "$outj" ]; then ev EVAL-SKIP "$tag: $outj exists"; return 0; fi
  ( cd "$RD" && penv "$PYPATH_CPU" CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$EVAL_SCRIPT" "$@" --out "$outj" > "$RD/logs/eval_$tag.out" 2> "$RD/logs/eval_$tag.err" ); rc=$?
  [ -s "$outj" ] || abort EVAL-FAILED "$tag rc=$rc (read $RD/logs/eval_$tag.err)"
  ev EVAL-DONE "$tag rc=$rc $outj"
}
EVC="--windows $SPLIT --split dev $GATEARGS --boot $BOOT --seed $SEED"
run_eval C0 "$RD/EVAL_C0.json" --packs "$RD/dump_C0/perc_det.pkl" --rast-dir "$RD/dump_C0/rast" $EVC
run_eval C1 "$RD/EVAL_C1.json" --packs "$RD/dump_C1/perc_det.pkl" --rast-dir "$RD/dump_C1/rast" $EVC
run_eval C1_vs_C0 "$RD/EVAL_C1_vs_C0.json" --packs "$RD/dump_C1/perc_det.pkl" --packs-b "$RD/dump_C0/perc_det.pkl" $EVC

# ---- FLOOR (CPU) ---------------------------------------------------------------------------------------------------------------------
( cd "$RD" && penv "$PYPATH_CPU" CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$FLOOR_SCRIPT" --eval-c0 "$RD/EVAL_C0.json" --eval-c1 "$RD/EVAL_C1.json" \
    --paired "$RD/EVAL_C1_vs_C0.json" --rast-c0 "$RD/dump_C0/rast" --rast-c1 "$RD/dump_C1/rast" --boot "$BOOT" --seed "$SEED" \
    --out-dir "$RD" > "$RD/logs/floor.out" 2> "$RD/logs/floor.err" ); rc=$?
FST="$("$PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8")).get("status"))' "$RD/FLOOR.json" 2>/dev/null)"
[ "$FST" = OK ] || abort FLOOR-NOT-OK "FLOOR.json status ${FST:-MISSING} rc=$rc" "read $RD/FLOOR.json and $RD/logs/floor.err"
ev FLOOR-DONE "$RD/FLOOR.json $RD/FLOOR.md"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RD/READ_DONE"
ev DONE "C0 $C0_RUN + C1 $C1_RUN read; floor at $RD/FLOOR.md"
release_lock
ev LOCK-RELEASED "$LOCKF"
exit 0
