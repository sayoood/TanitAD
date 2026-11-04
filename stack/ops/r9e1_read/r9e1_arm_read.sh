#!/usr/bin/env bash
# r9e1_arm_read.sh <ARM> [REF] -- THOR-SIDE automatic read of ONE refcv9 E1 lever arm against ITS reference.  E1-VERDICT, 2026-11-04.
# Modelled on r9e1_read.sh (the C0 / C1 floor read), which it does NOT edit and does NOT replace: that script reads the floor, this one reads a lever.
#
#   WAIT   a FILE-ONLY poll (every R9E1_POLL_S = 60 s: no GPU, no torch, python stdlib only: `r9_e1_verdict.py state`, which reuses the chain's own
#          done rule from r9e1_read_state.py -- that file is C0 / C1-only, so the arm-aware check lives in r9_e1_verdict.py) until
#            (a) $RD/FLOOR.json exists with status OK  (the floor read finished), AND
#            (b) the ARM is done by the chain's own rule (<run>/summary.json "done": true + "zerograd_gate", no ZEROGRAD_FAIL.json, final EMA +
#                config.json, step == the launch argv's steps), and likewise the REFERENCE when it is not C0 / C1.
#          An ARM-FAILED / ARM-ZEROGRAD-FAIL / ARM-REFUSED / ARM-NOT-LAUNCHED chain line, an ALLDONE without the arm, or a REFUSED floor: READ_ABORTED.
#   LOCK   (only if a dump is missing) flock -w >= 12 h on /home/nvidia/refcv7_post/thor_gpu.lock (the chain's discipline), GPU must be empty.
#   DUMP   r9_e1_dump.py on the ARM's final EMA (<run>/model_ema.pt) with the SAME split / S2 windows / flags as the floor's dumps.  The REFERENCE dump
#          is REUSED (C0: $RD/dump_C0, never re-dumped); a non-floor reference (M1, B2, T1, ...) is dumped once into $RD/arm_<REF>/dump_<REF> if no
#          earlier arm read has made it.  The lock is released right after the dump (the floor script holds it to the end; nothing after it needs the GPU).
#   CHECK  dump.json of arm, reference and the floor's C0 agree on order_digest / n_windows / n_clips / step and are EMA, not SMOKE_RESTRICTED.
#   EVAL   r9_e1_eval.py: the arm single (+ rasters), the reference single when missing, and once PAIRED (--packs ARM --packs-b REF): CPU only.
#   VERDICT r9_e1_verdict.py -> VERDICT_<ARM>.json + .md in $RD/arm_<ARM>/.   Then ZZR9E1ARMREAD-DONEZZ.
# Every verdict is read from an ARTIFACT (dump.json / EVAL_*.json / VERDICT_*.json status), never from $?.  Resumable: a step whose artifact exists is
# skipped.  No model turns poll this: it is plain bash.   Events: $RD/arm_<ARM>/READ_STATUS.txt (append-only), ALIVE, logs/*.out|err; opaque markers
# ZZR9E1ARMREAD-<EVENT>ZZ also go to stdout.   Detach:  nohup setsid bash r9e1_arm_read.sh B1 > $RD/arm_B1.out 2>&1 < /dev/null &
#
# DRY RUN (dev box, no GPU, no flock): R9E1_DRY=1 + R9E1_ROOT / R9E1_READ_DIR / R9E1_TREE pointing at SYNTHETIC dirs, R9E1_DUMP_SCRIPT=<r9e1_read_stub_dump.py>,
#   R9E1_PYTHONPATH, R9E1_PY, R9E1_GATE_ARGS="--gate 0.5", R9E1_SPLIT, R9E1_SPLIT_SHA_FILE, R9E1_BOOT, R9E1_POLL_S, R9E1_LOCKF -- exactly r9e1_read.sh's.
# R9E1_LIFT_VALID=<sidecar .npz | raster dir of an unobs-writing arm> (OPTIONAL): the ARM-INDEPENDENT lift-valid mask of the M1 MAP PRIMARY (nf_swedge_prec_lv_*).  When set it is
#   passed as `--lift-valid <path>` to BOTH eval calls (the single evals and the paired one) and to the verdict, which REFUSES unless the mask's sha256 equals the one the FLOOR
#   was read on (so the FLOOR must have been read with the same mask).  Unset => no `--lift-valid` anywhere: the argv is exactly the earlier one (test_r9_e1_arm_read.py pins it).
# TRAPS: no `pgrep -f` on a pattern this command line contains; detach with `nohup setsid ... < /dev/null &`; ssh with `-n`; the Thor python needs `-X utf8`.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM="${1:-}"
REFARG="${2:-}"
usage() { echo "usage: $0 <ARM> [REF]   (ARM = a lever arm of REFCV9_DESIGN 4.3, e.g. B1 M2a T1; REF defaults to the design's reference)" >&2; }
if [ -z "$ARM" ] || ! [[ "$ARM" =~ ^[A-Za-z][A-Za-z0-9-]{0,15}$ ]]; then usage; exit 4; fi
case "$ARM" in C0|C1) echo "[r9e1_arm_read] $ARM is a FLOOR arm: r9e1_read.sh reads it (this script reads lever arms)" >&2; exit 4;; esac
if [ -n "$REFARG" ] && ! [[ "$REFARG" =~ ^[A-Za-z][A-Za-z0-9-]{0,15}$ ]]; then usage; exit 4; fi
R="${R9E1_ROOT:-/home/nvidia/r8ns/r9e1}"
RD="${R9E1_READ_DIR:-$R/read}"
AD="$RD/arm_$ARM"
STAT="${R9E1_STATUS:-$R/gate3/CHAIN_STATUS.txt}"
T="${R9E1_TREE:-$R/launch_tree}"
PY="${R9E1_PY:-/home/nvidia/venvs/tanitad-train/bin/python}"
LOCKF="${R9E1_LOCKF:-/home/nvidia/refcv7_post/thor_gpu.lock}"
POLL_S="${R9E1_POLL_S:-60}"
MAX_WAIT_S="${R9E1_MAX_WAIT_S:-1209600}"          # 14 days: a wedged chain must not leave a waiter forever
LOCK_MAX_S="${R9E1_LOCK_MAX_S:-43200}"            # >= 12 h: the chain trains arms back to back under this lock and frees it only between arms
DRY="${R9E1_DRY:-0}"
BOOT="${R9E1_BOOT:-2000}"
SEED=0
# ---- the floor read's pins (r9e1_read.sh: SPLIT_SHA_PIN / GATES_SHA / S2); tests/test_r9_e1_arm_read.py asserts they equal r9e1_read.sh's ----
SPLIT="${R9E1_SPLIT:-/home/nvidia/r8ns/r9dump/DEV_HOLD_SPLIT.json}"
SPLIT_SHA_FILE="${R9E1_SPLIT_SHA_FILE:-$HERE/DEV_HOLD_SPLIT.sha256}"
SPLIT_SHA_PIN="ca5af0203320aba4f89e09be81a6bf079d9d4d77708ccb8a3ca2a0ea57e8db0a"
S2="${R9E1_S2:-/home/nvidia/r8ns/perc8/inputs/s2_windows.json}"
GATES="${R9E1_GATES:-/home/nvidia/r8ns/perc8/out/T_nosync/refcv8_det_presence_gates_train.json}"
GATES_SHA="f4cb9a94f1dc3208599438e6748d19dfd12a0261dd7e476a8934b7ae53b94611"
DUMP_CWD="${R9E1_DUMP_CWD:-$T/stack}"
DUMP_SCRIPT="${R9E1_DUMP_SCRIPT:-scripts/r9_e1_dump.py}"
EVAL_SCRIPT="${R9E1_EVAL_SCRIPT:-$T/stack/scripts/r9_e1_eval.py}"
if [ -n "${R9E1_VERDICT_SCRIPT:-}" ]; then VERDICT_SCRIPT="$R9E1_VERDICT_SCRIPT"
elif [ -f "$HERE/r9_e1_verdict.py" ]; then VERDICT_SCRIPT="$HERE/r9_e1_verdict.py"
else VERDICT_SCRIPT="$HERE/../../scripts/r9_e1_verdict.py"; fi                    # the repo layout (stack/ops/r9e1_read -> stack/scripts)
export R9E1_STATE_PY="${R9E1_STATE_PY:-$HERE/r9e1_read_state.py}"                # the chain's own marker rules, imported (read-only) by `r9_e1_verdict.py state`
PYPATH="${R9E1_PYTHONPATH:-$T/stack:$T/stack/scripts:$T/taniteval}"
command -v flock >/dev/null 2>&1 && HAVE_FLOCK=1 || HAVE_FLOCK=0
mkdir -p "$AD/logs"
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$AD/READ_STATUS.txt"; echo "ZZR9E1ARMREAD-$1ZZ $(date -u +%H:%M:%SZ) ${*:2}"; }

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
  printf '%s %s\n%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "${2:-}" "${3:-}" > "$AD/READ_ABORTED"
  ev ABORTED "$1 -- ${2:-} ${3:-}"
  release_lock
  exit 3
}

if [ "$HAVE_FLOCK" = 0 ] && [ "$DRY" != 1 ]; then echo "[r9e1_arm_read] no flock: refusing a real run without the GPU-lock discipline" >&2; exit 4; fi
if [ "$HAVE_FLOCK" = 1 ]; then          # one reader per arm only (a second start exits quietly)
  exec 8>"$AD/instance.lock"
  flock -n 8 || { echo "ZZR9E1ARMREAD-ALREADYRUNNINGZZ another r9e1_arm_read $ARM holds $AD/instance.lock"; exit 0; }
fi
echo $$ > "$AD/r9e1_arm_read.pid"
if [ -e "$AD/READ_DONE" ]; then echo "ZZR9E1ARMREAD-ALREADYDONEZZ $AD/READ_DONE exists (remove it to re-read)"; exit 0; fi
rm -f "$AD/READ_ABORTED"
[ -s "$VERDICT_SCRIPT" ] || abort NO-SCRIPT "$VERDICT_SCRIPT is missing"
# the reference: the argument, else REFCV9_DESIGN 4.3's (the mapping lives in r9_e1_verdict.py: ONE definition)
if [ -n "$REFARG" ]; then REF="$REFARG"; else
  REF="$("$PY" -I -X utf8 "$VERDICT_SCRIPT" ref --arm "$ARM" 2> "$AD/logs/ref.err")" || abort NO-REFERENCE "REFCV9_DESIGN 4.3 names no reference for $ARM: pass one" "$(cat "$AD/logs/ref.err" 2>/dev/null)"
fi
ev START "pid $$ dry=$DRY arm=$ARM ref=$REF status=$STAT root=$R poll=${POLL_S}s"

dump_dir() { case "$1" in C0|C1) echo "$RD/dump_$1";; *) echo "$RD/arm_$1/dump_$1";; esac; }     # the floor's layout for C0 / C1, one dir per other arm
eval_json() { case "$1" in C0|C1) echo "$RD/EVAL_$1.json";; *) echo "$RD/arm_$1/EVAL_$1.json";; esac; }

# ---- WAIT: file-only ------------------------------------------------------------------------------------------------------------
t0=$(date +%s); lastsig=""
while :; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$AD/ALIVE"
  S_STATE=ERROR; S_SIG=ERROR; S_REASON=""; S_LINE=""; S_RUN=""; S_REF_RUN=""
  if "$PY" -I -X utf8 "$VERDICT_SCRIPT" state --arm "$ARM" --ref "$REF" --status "$STAT" --root "$R" --read-dir "$RD" --out "$AD/state.env" > "$AD/state.out" 2> "$AD/state.err"; then
    . "$AD/state.env"
  else
    S_REASON="state helper failed: $(tail -c 300 "$AD/state.err" 2>/dev/null | tr '\n' ' ')"
  fi
  if [ "$S_SIG" != "$lastsig" ]; then ev "$S_STATE" "$S_REASON"; lastsig="$S_SIG"; fi
  case "$S_STATE" in
    READY) break;;
    ABORT) abort CHAIN "$S_LINE" "$S_REASON";;
  esac
  [ $(( $(date +%s) - t0 )) -gt "$MAX_WAIT_S" ] && abort WAIT-TIMEOUT "arm $ARM not done in ${MAX_WAIT_S}s" "$S_REASON"
  sleep "$POLL_S"
done
ARM_RUN="$S_RUN"; REF_RUN="$S_REF_RUN"

# ---- inputs, checked BEFORE the GPU lock is taken (the floor read's, identical) ---------------------------------------------------
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
for f in "$EVAL_SCRIPT"; do [ -s "$f" ] || abort NO-SCRIPT "$f is missing"; done
[ -s "$RD/FLOOR.json" ] && [ -s "$RD/EVAL_C0.json" ] || abort NO-FLOOR "$RD/FLOOR.json or $RD/EVAL_C0.json is missing (the floor read must have finished)"
[ -s "$RD/dump_C0/dump.json" ] || abort NO-FLOOR-DUMP "$RD/dump_C0/dump.json is missing: the floor's windows cannot be compared"
case "$REF" in C0|C1) [ -s "$(dump_dir "$REF")/dump.json" ] && [ -s "$(eval_json "$REF")" ] || abort NO-REF-DUMP "reference $REF: $(dump_dir "$REF")/dump.json or $(eval_json "$REF") is missing; a floor arm is never re-dumped here";; esac
ev INPUTS-OK "split sha256 ${GOT:0:12} matches ${SPLIT_SHA_FILE##*/}; gate args: $GATEARGS; arm run $ARM_RUN; reference $REF ${REF_RUN:+run $REF_RUN}"
LVA=()
if [ -n "${R9E1_LIFT_VALID:-}" ]; then          # the map primary's mask: it must exist; the verdict itself refuses a mask that is not the floor's (sha256)
  [ -e "$R9E1_LIFT_VALID" ] || abort NO-LIFT-VALID "R9E1_LIFT_VALID=$R9E1_LIFT_VALID is neither a sidecar .npz nor a raster directory"
  LVA=(--lift-valid "$R9E1_LIFT_VALID")
  ev LIFT-VALID "the map primary is read on the arm-independent mask $R9E1_LIFT_VALID (--lift-valid to the evals and the verdict; its sha256 must equal the floor's)"
fi

penv() {           # $1 = PYTHONPATH, rest = the command
  local pp="$1"; shift
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u REFCV7_KIT -u REFCV7_REMAP_OVERRIDES -u REFCV7_REPO \
  PYTHONPATH="$pp" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  HF_TOKEN_PATH=/nonexistent/hf_token REFCV6_KIT=/home/nvidia REFCV6_REMAP_OVERRIDES="$T/stack/ops/refcv8_eval_remap_a2_thor.json" "$@"; }

# Thor's training venv has numpy but NO scipy (r9e1_read.sh, MEASURED 2026-11-03): the CPU steps get shim_scipy/ on PYTHONPATH only when the real scipy
# cannot be imported (never ahead of a working one).  It sits next to this script, as for r9e1_read.sh.
PYPATH_CPU="$PYPATH"
if [ "${R9E1_FORCE_SCIPY_SHIM:-0}" = 1 ] || ! penv "$PYPATH" "$PY" -c 'import scipy.ndimage' > /dev/null 2>&1; then
  [ -f "$HERE/shim_scipy/scipy/ndimage/__init__.py" ] || abort NO-SCIPY "the venv has no scipy and $HERE/shim_scipy is missing"
  SHIMD="$HERE/shim_scipy"; command -v cygpath >/dev/null 2>&1 && SHIMD="$(cygpath -m "$SHIMD")"          # a native-python path on Git bash (dry run)
  PYPATH_CPU="$PYPATH${R9E1_PATHSEP:-:}$SHIMD"
  ev SCIPY-SHIM "eval/verdict use $HERE/shim_scipy (numpy binary_dilation, held equal to scipy by test)"
fi

# ---- DUMP (the only GPU step; skipped, lock untaken, when every dump exists) -------------------------------------------------------
dump_arm() {       # $1 = arm, $2 = run dir
  local A="$1" RUN="$2" O rc; O="$(dump_dir "$1")"
  if [ -s "$O/dump.json" ]; then ev DUMP-SKIP "$A: $O/dump.json exists"; return 0; fi
  [ -s "$RUN/model_ema.pt" ] && [ -s "$RUN/config.json" ] || abort NO-CHECKPOINT "$A: $RUN/model_ema.pt or $RUN/config.json is missing"
  mkdir -p "$(dirname "$O")"; rm -rf "$O"; ev DUMP-START "$A ckpt $RUN/model_ema.pt"
  ( cd "$DUMP_CWD" && penv "$PYPATH" "$PY" -X utf8 "$DUMP_SCRIPT" --ckpt "$RUN/model_ema.pt" --config "$RUN/config.json" --windows "$SPLIT" \
      --split dev --s2-windows "$S2" --require-ema --out "$O" > "$AD/logs/dump_$A.out" 2> "$AD/logs/dump_$A.err" ); rc=$?
  [ -s "$O/dump.json" ] || abort DUMP-FAILED "$A rc=$rc (read $AD/logs/dump_$A.err)"
  ev DUMP-DONE "$A rc=$rc $O"
}
NEED=0
[ -s "$(dump_dir "$ARM")/dump.json" ] || NEED=1
case "$REF" in C0|C1) ;; *) [ -s "$(dump_dir "$REF")/dump.json" ] || NEED=1;; esac
if [ "$NEED" = 1 ]; then
  ev LOCK-WAIT "taking the Thor GPU lock $LOCKF (<= ${LOCK_MAX_S}s; the chain frees it between arms)"
  take_lock "$LOCK_MAX_S" || abort LOCK-TIMEOUT "the Thor GPU lock stayed held for ${LOCK_MAX_S}s; nothing was run"
  ev LOCK-TAKEN "$LOCKF"
  if [ "$DRY" = 1 ]; then ev GPU-CHECK "skipped (dry run)"; else
    NA=1; for _i in $(seq 1 20); do NA="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')"; [ "$NA" = 0 ] && break; sleep 15; done
    [ "$NA" = 0 ] || abort GPU-NOT-EMPTY "$NA compute app(s) under the lock after 5 min; never share the GPU"
    ev GPU-CHECK "no compute app"
  fi
  dump_arm "$ARM" "$ARM_RUN"
  case "$REF" in C0|C1) ;; *) dump_arm "$REF" "$REF_RUN";; esac
  release_lock
  ev LOCK-RELEASED "$LOCKF (the dump was the only GPU step)"
fi
DA="$(dump_dir "$ARM")"; DR="$(dump_dir "$REF")"
CHK="$("$PY" -I -X utf8 "$VERDICT_SCRIPT" dumpcheck --dump "$DA/dump.json" --dump "$DR/dump.json" --dump "$RD/dump_C0/dump.json" 2>&1)"
[ $? = 0 ] || abort DUMP-MISMATCH "the dumps are not one read (arm / reference / the floor's C0)" "$CHK"
ev DUMP-CHECK "arm $ARM, reference $REF and the floor's C0: same windows (order_digest), same step, all EMA"

# ---- EVAL (CPU) ----------------------------------------------------------------------------------------------------------------------
run_eval() {       # $1 = tag, $2 = out json, rest = r9_e1_eval.py args
  local tag="$1" outj="$2" rc; shift 2
  if [ -s "$outj" ]; then ev EVAL-SKIP "$tag: $outj exists"; return 0; fi
  ( cd "$AD" && penv "$PYPATH_CPU" CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$EVAL_SCRIPT" "$@" --out "$outj" > "$AD/logs/eval_$tag.out" 2> "$AD/logs/eval_$tag.err" ); rc=$?
  [ -s "$outj" ] || abort EVAL-FAILED "$tag rc=$rc (read $AD/logs/eval_$tag.err)"
  ev EVAL-DONE "$tag rc=$rc $outj"
}
EVC="--windows $SPLIT --split dev $GATEARGS --boot $BOOT --seed $SEED"
run_eval "$ARM" "$(eval_json "$ARM")" --packs "$DA/perc_det.pkl" --rast-dir "$DA/rast" $EVC ${LVA[@]+"${LVA[@]}"}
run_eval "$REF" "$(eval_json "$REF")" --packs "$DR/perc_det.pkl" --rast-dir "$DR/rast" $EVC ${LVA[@]+"${LVA[@]}"}
PAIRED="$AD/EVAL_${ARM}_vs_${REF}.json"
run_eval "${ARM}_vs_${REF}" "$PAIRED" --packs "$DA/perc_det.pkl" --packs-b "$DR/perc_det.pkl" $EVC ${LVA[@]+"${LVA[@]}"}

# ---- VERDICT (CPU) -------------------------------------------------------------------------------------------------------------------
( cd "$AD" && penv "$PYPATH_CPU" CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$VERDICT_SCRIPT" --floor "$RD/FLOOR.json" --floor-eval-c0 "$RD/EVAL_C0.json" \
    --paired "$PAIRED" --eval-arm "$(eval_json "$ARM")" --eval-ref "$(eval_json "$REF")" --arm "$ARM" --ref "$REF" \
    --rast-arm "$DA/rast" --rast-ref "$DR/rast" ${LVA[@]+"${LVA[@]}"} --boot "$BOOT" --seed "$SEED" --out-dir "$AD" > "$AD/logs/verdict.out" 2> "$AD/logs/verdict.err" ); rc=$?
VST="$("$PY" -I -X utf8 -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); print(d.get("status"), d.get("verdict"))' "$AD/VERDICT_$ARM.json" 2>/dev/null)"
case "$VST" in "OK "*) ;; *) abort VERDICT-NOT-OK "VERDICT_$ARM.json status/verdict ${VST:-MISSING} rc=$rc" "read $AD/VERDICT_$ARM.md and $AD/logs/verdict.err";; esac
ev VERDICT-DONE "$AD/VERDICT_$ARM.json $AD/VERDICT_$ARM.md -> ${VST#OK }"
date -u +%Y-%m-%dT%H:%M:%SZ > "$AD/READ_DONE"
ev DONE "arm $ARM vs $REF read: ${VST#OK }"
exit 0
