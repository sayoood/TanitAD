#!/usr/bin/env bash
# r9e2_roll.sh -- THOR-SIDE DRAFT: the refcv9 E2 roll + read of ONE K2 against ONE K0 on DEV, at inference seeds 0 AND 1.  E2-ROLL-2, 2026-11-04.
# NEW file.  Nothing here has run on Thor (no K0 / K2 checkpoint existed when it was written); the dev-box proof is test_r9e2_roll.py (a DRY RUN + structure).
#
#   R9E2_DRY=1 prints EVERY command ("[dry] ...") and touches nothing: no mkdir, no lock, no log, no status file, no python.
#
#   ONE GPU-lock hold (copied from r9e1_arm_read.sh take_lock / GPU-empty loop / release_lock): flock -w 43200 on thor_gpu.lock, GPU must be empty, then
#     (1) the DEV-filtered, staged battery roll of K0 then K2:  run_battery_r9_e2.py --no-g0 --no-baselines --infer-seeds 0,1 --lock-held
#     (2) per seed the K2 CAMERA-BLIND roll (W4 preview):       camblind_roll_r9_e2.py -> camblind_roll_r9.py (UNCHANGED) --camera derange --require-seal --os-only
#     (3) the perception dumps of K2 then K0:                    r9_e1_dump.py --require-ema --split dev     (the only other GPU step)
#     (4) the MAPSEL roll (S6c input) of K2 then K0 at inference seeds 0 AND 1 (four rolls, DEV windows only): mapsel_roll_r9_e2.py, inside THIS hold -- no second lock.
#         --own-planes is derived from the run's config.json: `skip` iff its argv holds `--r9-map-banded` (a C1-banded K2: K2M1N20 / K2M3N20), else `require`.
#   lock released; then CPU only: per seed r9_e2_families.py (families, camblind) + r9_e2_distkeep.py --merge-into families (LONGITUDINAL
#   distance-keeping rows); r9_e1_eval.py (K2, K0); then the S6c GATE; r9e2_inputs.py -> INPUTS.json; r9_e2_verdict.py -> E2_VERDICT.json / .md.
#   Markers ZZR9E2ROLL-<EVENT>ZZ on stdout and in $OUT/ROLL_STATUS.txt.
#   S6c-DEV is SCORED OFF THOR (r9e2_s6c_resume.sh, dev box: s6c_dev.py needs the banked refcv8 eval artifacts + retime stage that exist only there).  The gate: when ALL FOUR
#   $R9E2_S6C_DIR (default $OUT/read/s6c) /S6C_{K2,K0}_s{0,1}.json are present the run goes on to INPUTS + the verdict; when ANY is missing it STOPS with the marker
#   ZZR9E2ROLL-AWAIT-S6CZZ, writes $OUT/AWAIT_S6C (what to pull, what to run) and exits 5 -- INPUTS.json is NOT written and the judge NOT run, so a verdict with a null
#   S6c can never be computed while the rolls exist.  Resume: pull $OUT to the dev box and run r9e2_s6c_resume.sh (scores the four traces, then INPUTS -> verdict).
#
# ENV PINS (all REQUIRED, exit 4 when unset -- no dev-box default is ever taken on Thor; R8_NO_POSE_SYNC must be UNSET, exit 4):
#   R9E2_K0_RUN R9E2_K2_RUN   run dirs (model_ema.pt + config.json)         R9E2_TREE   launch tree (= R8_LAUNCH_TREE; PYTHONPATH $T/stack:$T/stack/scripts:$T/taniteval)
#   R9E2_PF_CODE              2026-10-20-refcv8-perframe-final/code            PF_R8_CODE  2026-10-15-refcv8-milestone-battery/code (r8_repoint.py)
#   PF_NF_HOLD                2026-10-20-refcv8-nav-feed-fix                   R8_EVAL_KIT eval139 cache + s2_labels_v8_eval.jsonl.gz
#   R9E2_REMAP                an EMPTY {} json (Thor)                          R9E2_SPLIT  DEV_HOLD_SPLIT.json     R9E2_S2  s2_windows.json
#   R9E2_LIFT_VALID           the arm-independent lift-valid mask (map primary) R9E2_OUT   output root
#   R9E2_LEAD_BLOCK           the B1 EVAL lead block on the roll's 0.5 s grid (taniteval/tools/build_lead_block_b1.py --k 4 --dt 0.5)
#   L1_PF_CODE L1_R8_CODE     camblind_roll_r9.py's own pins (== R9E2_PF_CODE / PF_R8_CODE)
#   R9E2_MAP_GT_ROOT          the refcv6 eval GT-map root (sam3_gt_v3_eval; the MAPSEL roll's --map-gt-root)
# OPTIONAL: R9E2_MAPSEL_CODE (default $INC/2026-10-25-refcv9-mapsel/code), R9E2_S6C_DIR, R9E2_PY, R9E2_LOCKF, R9E2_LOCK_MAX_S, R9E2_PATHSEP, R9E2_PYTHONPATH, R9E2_OPS, R9E2_CAMBLIND_SCRIPT, R9E2_CB_CWD,
#   R9E2_SPLIT_SHA_FILE, R9E2_GATES / R9E2_GATE_ARGS, R9E2_OS_ONLY_ALL=1 (battery --os-only-all: only `os` + controls are rolled; the families / W4 read use `os` only),
#   R9E2_SHIM (shim_scipy dir), R9E2_DRY, R9E2_BASE_CWD (the battery parent's cwd; see HAZARD below).
# HAZARD (r8_battery_patch.apply -> fix_baselines, UNCONDITIONAL even with --no-baselines): the battery parent REFUSES unless ALL THREE refcv5cmp baseline dumps
#   (refcv4b_devbox_dump, refcv5-v2_dump, refcv5-v2-seed1_dump; each manifest.json + ep000.npz + decisions/ep000.npz) exist under one of D:/Projects/TanitAD-artifacts/refcv5cmp/out,
#   E:/..., C:/Users/Admin/c_cleanup_archive_2026-10-15_b/refcv5cmp/out, C:/Users/Admin/refcv5cmp/out -- all RELATIVE paths on Thor: set R9E2_BASE_CWD to a dir holding one such tree.
# R9E2_S6C_K0 (the old single-file S6c switch) is REFUSED (exit 4): one file would be stamped on both inference seeds.
# EXIT: 0 GO | 1 NO-GO | 2 verdict REFUSED | 3 harness abort (ROLL_ABORTED written) | 4 refused to start | 5 rolls done, AWAITING the off-Thor S6c scoring (no verdict computed).
# TRAPS (TRAPS.md): no `pgrep -f` on a pattern this command line contains; detach with `nohup setsid ... < /dev/null &`; the Thor python needs `-X utf8`.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY="${R9E2_DRY:-0}"
PINS="R9E2_K0_RUN R9E2_K2_RUN R9E2_TREE R9E2_PF_CODE PF_R8_CODE PF_NF_HOLD R8_EVAL_KIT R9E2_REMAP R9E2_SPLIT R9E2_S2 R9E2_LIFT_VALID R9E2_OUT L1_PF_CODE L1_R8_CODE R9E2_LEAD_BLOCK R9E2_MAP_GT_ROOT"
MISS=""
for _p in $PINS; do [ -n "${!_p:-}" ] || MISS="$MISS $_p"; done
if [ -n "$MISS" ]; then echo "[r9e2_roll] refusing: unset env pin(s):$MISS -- no dev-box default is taken on Thor" >&2; exit 4; fi
if [ -n "${R9E2_S6C_K0:-}" ]; then echo "[r9e2_roll] refusing: R9E2_S6C_K0 names ONE S6c file for both inference seeds; seeds are separate runs -- S6c is per arm per seed (see R9E2_S6C_DIR)" >&2; exit 4; fi
if [ -n "${R8_NO_POSE_SYNC:-}" ]; then echo "[r9e2_roll] refusing: R8_NO_POSE_SYNC is set -- E2 reads the CORRECTED pose clock (K0 trained on it); unset it" >&2; exit 4; fi

T="$R9E2_TREE"; OUT="$R9E2_OUT"; RD="$OUT/read"; LOGS="$OUT/logs"
RUN0="$R9E2_K0_RUN"; RUN2="$R9E2_K2_RUN"
SPLIT="$R9E2_SPLIT"; S2="$R9E2_S2"
export R8_LAUNCH_TREE="$T"                                   # r8_repoint.LAUNCH_TREE (run_battery_r9_e2.pin_repoint asserts it equals this pin)
PY="${R9E2_PY:-/home/nvidia/venvs/tanitad-train/bin/python}"
LOCKF="${R9E2_LOCKF:-/home/nvidia/refcv7_post/thor_gpu.lock}"
LOCK_MAX_S="${R9E2_LOCK_MAX_S:-43200}"                       # >= 12 h: the chain trains arms back to back under this lock
SEP="${R9E2_PATHSEP:-:}"
PYPATH="${R9E2_PYTHONPATH:-$T/stack$SEP$T/stack/scripts$SEP$T/taniteval}"
OPS="${R9E2_OPS:-$HERE}"
INC="$T/FlyWheels/TanitAD_EvalFlyWheel/incoming"
MS_CODE="${R9E2_MAPSEL_CODE:-$INC/2026-10-25-refcv9-mapsel/code}"
S6D="${R9E2_S6C_DIR:-$RD/s6c}"
CB_SCRIPT="${R9E2_CAMBLIND_SCRIPT:-$INC/2026-11-03-refcv9-camblind/code/camblind_roll_r9.py}"
CB_CWD="${R9E2_CB_CWD:-$OUT/camblind_cwd}"
CB_REMAP_REL="C:/Users/Admin/refcv8_pf/refcv9_l1/code/refcv8_eval_remap_devbox_l1.json"       # camblind_roll_r9.py hard-codes this RELATIVE-on-Thor path (EVAL_REMAP)
SHIM="${R9E2_SHIM:-$HERE/../r9e1_read/shim_scipy}"
# ---- the floor read's pins (r9e1_arm_read.sh: SPLIT_SHA_PIN / GATES_SHA; test_r9e2_roll.py asserts they equal arm_read's) ----
SPLIT_SHA_PIN="ca5af0203320aba4f89e09be81a6bf079d9d4d77708ccb8a3ca2a0ea57e8db0a"
GATES="${R9E2_GATES:-/home/nvidia/r8ns/perc8/out/T_nosync/refcv8_det_presence_gates_train.json}"
GATES_SHA="f4cb9a94f1dc3208599438e6748d19dfd12a0261dd7e476a8934b7ae53b94611"
BOOT=2000; SEED=0
command -v flock >/dev/null 2>&1 && HAVE_FLOCK=1 || HAVE_FLOCK=0

q() { printf '%q ' "$@"; }
ev() { echo "ZZR9E2ROLL-$1ZZ $(date -u +%H:%M:%SZ) ${*:2}"; [ "$DRY" = 1 ] || echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$OUT/ROLL_STATUS.txt"; }
X() { if [ "$DRY" = 1 ]; then echo "[dry] $(q "$@")"; return 0; fi; "$@"; }                                       # run, or only print
XL() { local lg="$1"; shift; if [ "$DRY" = 1 ]; then echo "[dry] $(q "$@")> $lg.out 2> $lg.err"; return 0; fi; "$@" > "$lg.out" 2> "$lg.err"; }
XC() { local d="$1"; shift; if [ "$DRY" = 1 ]; then echo "[dry] cd $d"; XL "$@"; else ( cd "$d" && XL "$@" ); fi; }
penv() {           # $1 = PYTHONPATH, rest = the command   (r9e1_arm_read.sh's penv: dump / eval / verdict)
  local pp="$1"; shift
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u REFCV7_KIT -u REFCV7_REMAP_OVERRIDES -u REFCV7_REPO \
  PYTHONPATH="$pp" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  HF_TOKEN_PATH=/nonexistent/hf_token REFCV6_KIT=/home/nvidia REFCV6_REMAP_OVERRIDES="$T/stack/ops/refcv8_eval_remap_a2_thor.json" "$@"; }
renv() {           # $1 = PYTHONPATH, rest = the command   (the ROLL: r8_repoint.configure_env() sets REFCV7_* itself and REFUSES a conflicting REFCV6_*, so those are unset)
  local pp="$1"; shift
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u REFCV7_KIT -u REFCV7_REMAP_OVERRIDES -u REFCV7_REPO \
  -u REFCV6_KIT -u REFCV6_REMAP_OVERRIDES -u REFCV6_REPO -u R8_NO_POSE_SYNC \
  PYTHONPATH="$pp" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  HF_TOKEN_PATH=/nonexistent/hf_token "$@"; }

LOCK_HELD=0
take_lock() {      # $1 = max seconds -> 0 when held   (r9e1_arm_read.sh, verbatim)
  if [ "$HAVE_FLOCK" = 1 ]; then
    exec 9>"$LOCKF" || return 1
    flock -w "$1" 9 || { exec 9>&-; return 1; }
  else
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
abort() {          # $1 = TAG, $2 = detail
  if [ "$DRY" = 1 ]; then echo "ZZR9E2ROLL-ABORTEDZZ (dry) $1 -- ${2:-}"; exit 3; fi
  printf '%s %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "${2:-}" > "$OUT/ROLL_ABORTED"
  ev ABORTED "$1 -- ${2:-}"
  release_lock
  exit 3
}
need() {           # $1 = file, $2 = what   (a real run: must exist and be non-empty; a dry run: print it)
  if [ "$DRY" = 1 ]; then echo "[dry] require nonempty $1  ($2)"; return 0; fi
  [ -s "$1" ] || abort NO-INPUT "$2: $1 is missing or empty"
}

if [ "$HAVE_FLOCK" = 0 ] && [ "$DRY" != 1 ]; then echo "[r9e2_roll] no flock: refusing a real run without the GPU-lock discipline" >&2; exit 4; fi
if [ "$DRY" = 1 ]; then echo "[dry] R9E2_DRY=1: every command below is only printed; nothing is created or run"; else mkdir -p "$OUT" "$RD" "$LOGS" || exit 4; fi
ev START "dry=$DRY k0=$RUN0 k2=$RUN2 out=$OUT tree=$T split=$SPLIT seeds=0,1"

# ---- inputs, checked BEFORE the GPU lock is taken ----------------------------------------------------------------------------------------
for R in "$RUN0" "$RUN2"; do need "$R/model_ema.pt" "run checkpoint (EMA)"; need "$R/config.json" "run config"; done
need "$SPLIT" "DEV_HOLD_SPLIT.json"
need "$S2" "s2_windows.json"
need "$R9E2_REMAP" "Thor remap json ({})"
need "$PF_R8_CODE/r8_repoint.py" "PF_R8_CODE"
need "$R9E2_PF_CODE/roll_seed_r8_pf.py" "R9E2_PF_CODE"
need "$CB_SCRIPT" "camblind_roll_r9.py"
need "$INC/2026-10-27-refcv9-anticheat/PREREG_ANTICHEAT_REFCV8.md" "camblind PKG (the prereg its --require-seal checks)"
need "$INC/2026-10-27-refcv9-anticheat/PREREG_ANTICHEAT_REFCV8.sha256" "camblind PKG seal"
need "$INC/2026-10-27-refcv9-anticheat/code/anticheat_common.py" "camblind PKG source"
need "$OPS/run_battery_r9_e2.py" "staged battery driver"; need "$OPS/camblind_roll_r9_e2.py" "staged camblind launcher"; need "$OPS/r9e2_inputs.py" "INPUTS writer"
need "$OPS/mapsel_roll_r9_e2.py" "staged MAPSEL roll driver (S6c)"; need "$OPS/r9e2_s6c_tools.py" "MAPSEL trace count check"
need "$MS_CODE/mapsel_roll.py" "MAPSEL roll (refcv9-mapsel code)"; need "$MS_CODE/../PREREG_MAPSEL.md" "MAPSEL prereg (its --require-seal checks)"
need "$MS_CODE/../PREREG_MAPSEL.sha256" "MAPSEL prereg seal"
need "$T/stack/scripts/r9_e1_dump.py" "dump"; need "$T/stack/scripts/r9_e1_eval.py" "eval"
need "$T/stack/scripts/r9_e2_families.py" "families"; need "$T/stack/scripts/r9_e2_verdict.py" "judge"
need "$T/stack/scripts/r9_e2_distkeep.py" "distance-keeping rows"
if [ ! -s "$R9E2_LEAD_BLOCK" ] && [ "$DRY" != 1 ]; then abort NO-LEAD-BLOCK "R9E2_LEAD_BLOCK=$R9E2_LEAD_BLOCK is not a file"; fi
[ "$DRY" = 1 ] && echo "[dry] require exists $R9E2_LEAD_BLOCK  (the lead block on the roll's 0.5 s grid)"
if [ ! -e "$R9E2_LIFT_VALID" ] && [ "$DRY" != 1 ]; then abort NO-LIFT-VALID "R9E2_LIFT_VALID=$R9E2_LIFT_VALID is neither a sidecar .npz nor a raster directory"; fi
[ "$DRY" = 1 ] && echo "[dry] require exists $R9E2_LIFT_VALID  (the arm-independent lift-valid mask)"
if [ "$DRY" = 1 ]; then echo "[dry] require sha256($SPLIT) == ${R9E2_SPLIT_SHA_FILE:-$(dirname "$SPLIT")/DEV_HOLD_SPLIT.sha256 if present, else the pin} ($SPLIT_SHA_PIN)"; else
  SHAF="${R9E2_SPLIT_SHA_FILE:-$(dirname "$SPLIT")/DEV_HOLD_SPLIT.sha256}"
  if [ -s "$SHAF" ]; then EXP="$(awk '{print $1; exit}' "$SHAF")"; else EXP="$SPLIT_SHA_PIN"; fi
  GOT="$(sha256sum "$SPLIT" | cut -d' ' -f1)"
  [ "$GOT" = "$EXP" ] || abort SPLIT-SHA "$SPLIT sha256 $GOT != expected $EXP"
fi
if [ -n "${R9E2_GATE_ARGS:-}" ]; then GATEARGS="$R9E2_GATE_ARGS"; else
  if [ "$DRY" = 1 ]; then echo "[dry] require sha256($GATES) == $GATES_SHA"; else
    need "$GATES" "presence gates"; GG="$(sha256sum "$GATES" | cut -d' ' -f1)"; [ "$GG" = "$GATES_SHA" ] || abort GATES-SHA "$GATES sha256 $GG != pinned $GATES_SHA"
  fi
  GATEARGS="--gates-json $GATES"
fi
# Thor's training venv has numpy but NO scipy (r9e1_read.sh, MEASURED 2026-11-03): the CPU steps get shim_scipy on PYTHONPATH only when the real scipy cannot be imported.
PYPATH_CPU="$PYPATH"
if [ "$DRY" = 1 ]; then echo "[dry] probe: import scipy.ndimage under the venv; when absent the CPU steps append $SHIM (assumed absent in a dry run)"; PYPATH_CPU="$PYPATH$SEP$SHIM"; else
  if ! penv "$PYPATH" "$PY" -c 'import scipy.ndimage' > /dev/null 2>&1; then
    [ -f "$SHIM/scipy/ndimage/__init__.py" ] || abort NO-SCIPY "the venv has no scipy and $SHIM is missing"
    PYPATH_CPU="$PYPATH$SEP$SHIM"; ev SCIPY-SHIM "eval / families / verdict use $SHIM"
  fi
fi
ev INPUTS-OK "pins, run dirs, split, S2, lift-valid, camblind PKG, scripts, gates checked before the lock"

# ---- the ONE lock hold: (1) rolls (2) camera-blind (3) perception dumps -------------------------------------------------------------------
ev LOCK-WAIT "taking the Thor GPU lock $LOCKF (<= ${LOCK_MAX_S}s)"
if [ "$DRY" = 1 ]; then echo "[dry] flock -w $LOCK_MAX_S $LOCKF (take_lock) ; GPU-empty loop over nvidia-smi --query-compute-apps (20 x 15 s)"; else
  take_lock "$LOCK_MAX_S" || abort LOCK-TIMEOUT "the Thor GPU lock stayed held for ${LOCK_MAX_S}s; nothing was run"
fi
ev LOCK-TAKEN "$LOCKF"
if [ "$DRY" != 1 ]; then
  NA=1; for _i in $(seq 1 20); do NA="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')"; [ "$NA" = 0 ] && break; sleep 15; done
  [ "$NA" = 0 ] || abort GPU-NOT-EMPTY "$NA compute app(s) under the lock after 5 min; never share the GPU"
fi
ev GPU-CHECK "no compute app (dry run: skipped)"

OSALL=(); [ "${R9E2_OS_ONLY_ALL:-0}" = 1 ] && OSALL=(--os-only-all)
roll_arm() {       # $1 = K0|K2, $2 = run dir
  local A="$1" RUN="$2" TAG="E2_$1_pf"
  if [ "$DRY" != 1 ] && [ -s "$OUT/$TAG/r9e2_dev_filter.json" ] && [ -s "$OUT/$TAG/dump_s0/manifest.json" ] && [ -s "$OUT/$TAG/dump_s1/manifest.json" ]; then ev ROLL-SKIP "$A: $OUT/$TAG is complete"; return 0; fi
  ev ROLL-START "$A tag $TAG ckpt $RUN/model_ema.pt"
  XC "${R9E2_BASE_CWD:-.}" "$LOGS/roll_$A" renv "$PYPATH" "$PY" -X utf8 "$OPS/run_battery_r9_e2.py" --ckpt "$RUN/model_ema.pt" --config "$RUN/config.json" --no-g0 --no-baselines \
      --tag "$TAG" --infer-seeds 0,1 --lock-held --out-root "$OUT" ${OSALL[@]+"${OSALL[@]}"}; local rc=$?
  if [ "$DRY" != 1 ]; then
    [ -s "$OUT/$TAG/r9e2_dev_filter.json" ] || abort ROLL-FAILED "$A rc=$rc: no $OUT/$TAG/r9e2_dev_filter.json (read $LOGS/roll_$A.err)"
    for _s in 0 1; do [ -s "$OUT/$TAG/dump_s$_s/manifest.json" ] && [ -s "$OUT/$TAG/windows_s$_s.json" ] || abort ROLL-FAILED "$A rc=$rc: seed $_s dump / windows missing (read $LOGS/roll_$A.err)"; done
  fi
  ev ROLL-DONE "$A rc=$rc $OUT/$TAG"
}
roll_arm K0 "$RUN0"
roll_arm K2 "$RUN2"

CBD="$OUT/E2_K2_camblind"
if [ "$DRY" = 1 ]; then
  echo "[dry] mkdir -p $CBD $CB_CWD/$(dirname "$CB_REMAP_REL")"; echo "[dry] cp $R9E2_REMAP $CB_CWD/$CB_REMAP_REL   (camblind_roll_r9.py's hard-coded EVAL_REMAP is a RELATIVE path on Thor: cwd = $CB_CWD)"
else
  mkdir -p "$CBD" "$CB_CWD/$(dirname "$CB_REMAP_REL")" && cp "$R9E2_REMAP" "$CB_CWD/$CB_REMAP_REL" || abort CB-CWD "cannot provision $CB_CWD/$CB_REMAP_REL"
fi
for S in 0 1; do
  if [ "$DRY" != 1 ] && [ -s "$CBD/blind_roll_s$S.json" ] && [ -s "$CBD/dump_s$S/manifest.json" ]; then ev CAMBLIND-SKIP "seed $S: $CBD/blind_roll_s$S.json exists"; continue; fi
  ev CAMBLIND-START "seed $S derange-seed 0"
  XC "$CB_CWD" "$LOGS/camblind_s$S" renv "$PYPATH" "$PY" -X utf8 "$OPS/camblind_roll_r9_e2.py" --camblind-script "$CB_SCRIPT" --camera derange --derange-seed 0 \
      --ztac-out "$CBD/ztac_blind_s$S.npz" --require-seal -- --ckpt "$RUN2/model_ema.pt" --config "$RUN2/config.json" --seed "$S" --dump-dir "$CBD/dump_s$S" \
      --windows-json "$OUT/E2_K2_pf/windows_s$S.json" --device cuda --out-json "$CBD/blind_roll_s$S.json" --os-only; rc=$?
  if [ "$DRY" != 1 ]; then [ -s "$CBD/blind_roll_s$S.json" ] && [ -s "$CBD/dump_s$S/manifest.json" ] || abort CAMBLIND-FAILED "seed $S rc=$rc (read $LOGS/camblind_s$S.err)"; fi
  ev CAMBLIND-DONE "seed $S rc=$rc $CBD/blind_roll_s$S.json"
done

dump_arm() {       # $1 = K2|K0, $2 = run dir
  local A="$1" RUN="$2" O="$RD/dump_$1"
  if [ "$DRY" != 1 ] && [ -s "$O/dump.json" ]; then ev DUMP-SKIP "$A: $O/dump.json exists"; return 0; fi
  ev DUMP-START "$A ckpt $RUN/model_ema.pt"
  [ "$DRY" = 1 ] || rm -rf "$O"
  XC "$T/stack" "$LOGS/dump_$A" penv "$PYPATH" "$PY" -X utf8 scripts/r9_e1_dump.py --ckpt "$RUN/model_ema.pt" --config "$RUN/config.json" --windows "$SPLIT" \
      --split dev --s2-windows "$S2" --require-ema --out "$O"; local rc=$?
  if [ "$DRY" != 1 ]; then [ -s "$O/dump.json" ] || abort DUMP-FAILED "$A rc=$rc (read $LOGS/dump_$A.err)"; fi
  ev DUMP-DONE "$A rc=$rc $O"
}
dump_arm K2 "$RUN2"
dump_arm K0 "$RUN0"

own_planes_mode() {   # $1 = a run's config.json -> skip iff its argv holds the exact token "--r9-map-banded" (not "--r9-map-banded-near-m"): a banded head has no 8-way raster by design
  if grep -q -- '"--r9-map-banded"' "$1" 2>/dev/null; then echo skip; else echo require; fi
}
mapsel_arm() {     # $1 = K2|K0, $2 = run dir: the MAPSEL roll at inference seeds 0 and 1, DEV windows only (the wrapper restricts to DEV and refuses HOLD), inside the SAME lock hold
  local A="$1" RUN="$2" MD="$OUT/E2_$1_mapsel" S OPM rc
  OPM="$(own_planes_mode "$RUN/config.json")"
  [ "$DRY" = 1 ] || mkdir -p "$MD" || abort MAPSEL-DIR "cannot create $MD"
  for S in 0 1; do
    if [ "$DRY" != 1 ] && [ -s "$MD/trace_s$S.npz" ] && "$PY" -X utf8 "$OPS/r9e2_s6c_tools.py" final "$MD/trace_s$S.npz" "$SPLIT" > /dev/null 2>&1; then ev MAPSEL-SKIP "$A seed $S: $MD/trace_s$S.npz is final"; continue; fi
    ev MAPSEL-START "$A seed $S own-planes $OPM ckpt $RUN/model_ema.pt"
    [ "$DRY" = 1 ] || rm -rf "$MD/rast_s$S" "$MD/dump_s$S"
    XC "${R9E2_BASE_CWD:-.}" "$LOGS/mapsel_${A}_s$S" renv "$PYPATH" "$PY" -X utf8 "$OPS/mapsel_roll_r9_e2.py" --mapsel-code "$MS_CODE" --infer-seed "$S" --own-planes "$OPM" \
        --trace-out "$MD/trace_s$S.npz" --rast-dir "$MD/rast_s$S" --map-gt-root "$R9E2_MAP_GT_ROOT" --require-seal -- \
        --ckpt "$RUN/model_ema.pt" --config "$RUN/config.json" --dump-dir "$MD/dump_s$S" --windows-json "$OUT/E2_${A}_pf/windows_s$S.json" --device cuda \
        --out-json "$MD/roll_s$S.json" --os-only; rc=$?
    if [ "$DRY" != 1 ]; then
      "$PY" -X utf8 "$OPS/r9e2_s6c_tools.py" final "$MD/trace_s$S.npz" "$SPLIT" > "$LOGS/mapsel_${A}_s$S.final" 2>&1 || abort MAPSEL-FAILED "$A seed $S rc=$rc: trace not final / not the DEV windows ($(cat "$LOGS/mapsel_${A}_s$S.final" 2>/dev/null); read $LOGS/mapsel_${A}_s$S.err)"
      rm -rf "$MD/dump_s$S"                                          # the battery dump of the roll is a temp copy once the trace is final
    fi
    ev MAPSEL-DONE "$A seed $S rc=$rc $MD/trace_s$S.npz own-planes $OPM"
  done
}
mapsel_arm K2 "$RUN2"
mapsel_arm K0 "$RUN0"
if [ "$DRY" = 1 ]; then echo "[dry] release the GPU lock $LOCKF"; else release_lock; fi
ev LOCK-RELEASED "$LOCKF (rolls, camera-blind, dumps and the MAPSEL rolls were the only GPU steps)"

# ---- CPU only -------------------------------------------------------------------------------------------------------------------------------
for S in 0 1; do
  XL "$LOGS/families_s$S" penv "$PYPATH_CPU" CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$T/stack/scripts/r9_e2_families.py" --mode families \
      --a-dump "$OUT/E2_K2_pf/dump_s$S" --b-dump "$OUT/E2_K0_pf/dump_s$S" --dev-split "$SPLIT" --out "$RD/families_s$S.json"; rc=$?
  if [ "$DRY" != 1 ]; then [ -s "$RD/families_s$S.json" ] || abort FAMILIES-FAILED "seed $S rc=$rc (read $LOGS/families_s$S.err)"; fi
  ev FAMILIES-DONE "seed $S rc=$rc $RD/families_s$S.json"
  XL "$LOGS/distkeep_s$S" penv "$PYPATH_CPU" CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$T/stack/scripts/r9_e2_distkeep.py" \
      --a-dump "$OUT/E2_K2_pf/dump_s$S" --b-dump "$OUT/E2_K0_pf/dump_s$S" --dev-split "$SPLIT" --lead-block "$R9E2_LEAD_BLOCK" \
      --out "$RD/distkeep_s$S.json" --merge-into "$RD/families_s$S.json"; rc=$?
  if [ "$DRY" != 1 ]; then { [ "$rc" = 0 ] && [ -s "$RD/distkeep_s$S.json" ]; } || abort DISTKEEP-FAILED "seed $S rc=$rc (read $LOGS/distkeep_s$S.err)"; fi
  ev DISTKEEP-DONE "seed $S rc=$rc $RD/distkeep_s$S.json (merged into families_s$S.json under LONGITUDINAL)"
  XL "$LOGS/camblind_read_s$S" penv "$PYPATH_CPU" CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$T/stack/scripts/r9_e2_families.py" --mode camblind \
      --a-dump "$OUT/E2_K2_pf/dump_s$S" --b-dump "$CBD/dump_s$S" --dev-split "$SPLIT" --out "$RD/camblind_s$S.json"; rc=$?
  if [ "$DRY" != 1 ]; then [ -s "$RD/camblind_s$S.json" ] || abort CAMBLIND-READ-FAILED "seed $S rc=$rc (read $LOGS/camblind_read_s$S.err)"; fi
  ev CAMBLIND-READ-DONE "seed $S rc=$rc $RD/camblind_s$S.json"
done
for A in K2 K0; do
  if [ "$DRY" != 1 ] && [ -s "$RD/EVAL_$A.json" ]; then ev EVAL-SKIP "$A: $RD/EVAL_$A.json exists"; continue; fi
  XC "$RD" "$LOGS/eval_$A" penv "$PYPATH_CPU" CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$T/stack/scripts/r9_e1_eval.py" --packs "$RD/dump_$A/perc_det.pkl" --rast-dir "$RD/dump_$A/rast" \
      --windows "$SPLIT" --split dev $GATEARGS --boot "$BOOT" --seed "$SEED" --lift-valid "$R9E2_LIFT_VALID" --out "$RD/EVAL_$A.json"; rc=$?
  if [ "$DRY" != 1 ]; then [ -s "$RD/EVAL_$A.json" ] || abort EVAL-FAILED "$A rc=$rc (read $LOGS/eval_$A.err)"; fi
  ev EVAL-DONE "$A rc=$rc $RD/EVAL_$A.json"
done
# ---- the S6c GATE: the four per-seed, per-arm scores must exist BEFORE INPUTS / the verdict -------------------------------------------------
S6CF="$S6D/S6C_K2_s0.json $S6D/S6C_K2_s1.json $S6D/S6C_K0_s0.json $S6D/S6C_K0_s1.json"
if [ "$DRY" = 1 ]; then for f in $S6CF; do echo "[dry] require nonempty $f  (S6c-DEV score, scored off Thor by r9e2_s6c_resume.sh; absent => exit 5, no INPUTS, no verdict)"; done; else
  S6MISS=""; for f in $S6CF; do [ -s "$f" ] || S6MISS="$S6MISS $f"; done
  if [ -n "$S6MISS" ]; then
    printf 'S6c scoring is pending (the verdict is NOT computed).\nmissing:%s\npull (dev box): %s/E2_K2_mapsel/trace_s{0,1}.npz + rast_s{0,1}/, %s/E2_K0_mapsel/..., and the whole %s\nrun: r9e2_s6c_resume.sh (R9E2_RD=<pulled read dir> R9E2_MS_ROOT=<pulled %s>)\n' "$S6MISS" "$OUT" "$OUT" "$RD" "$OUT" > "$OUT/AWAIT_S6C"
    ev AWAIT-S6C "rolls done; S6c files missing:$S6MISS -- see $OUT/AWAIT_S6C; exit 5, INPUTS.json NOT written, judge NOT run"
    exit 5
  fi
fi
ev S6C-READY "four S6c files present under $S6D"
XL "$LOGS/inputs" penv "$PYPATH_CPU" "$PY" -X utf8 "$OPS/r9e2_inputs.py" --rd "$RD" --out "$RD/INPUTS.json" --k2-label K2 --k0-label K0 --s6c-dir "$S6D"; rc=$?
if [ "$DRY" != 1 ]; then [ -s "$RD/INPUTS.json" ] || abort INPUTS-FAILED "rc=$rc (read $LOGS/inputs.err)"; fi
ev INPUTS-WRITTEN "$RD/INPUTS.json (per-seed S6c from $S6D)"
XL "$LOGS/verdict" penv "$PYPATH_CPU" CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$T/stack/scripts/r9_e2_verdict.py" --inputs "$RD/INPUTS.json" --out-dir "$RD"; VRC=$?
if [ "$DRY" = 1 ]; then ev DRY-DONE "nothing was run"; exit 0; fi
ev VERDICT-DONE "rc=$VRC $RD/E2_VERDICT.json $RD/E2_VERDICT.md (0 GO / 1 NO-GO / 2 REFUSED)"
case "$VRC" in 0|1) exit "$VRC";; *) echo "[r9e2_roll] the judge REFUSED (rc=$VRC): read $LOGS/verdict.err" >&2; exit 2;; esac
