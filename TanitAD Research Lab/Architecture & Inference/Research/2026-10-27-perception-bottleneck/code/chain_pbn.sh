#!/usr/bin/env bash
# chain_pbn.sh -- PERC-BOTTLENECK capture chain on the shared dev-box RTX 4060, under the dev-box gpu_lock
# (O_EXCL lock file; a foreign holder is NEVER broken; released in a trap). SMOKE=1 -> 3 windows per split.
# Markers (disjoint from any searched word): ZZPBN-<stage>-<rc>ZZ in $W/logs/chain.log
set -u
C="$(cd "$(dirname "$0")" && pwd)"
W=${PBN_WORK:-F:/perc_bottleneck}
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
T=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58
LOCKPY=$T/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/gpu_lock.py
JOB=perc-bottleneck
SMOKE=${SMOKE:-0}
export R8_LAUNCH_TREE=$T R8_NO_POSE_SYNC=1 OMP_NUM_THREADS=4 PYTHONUTF8=1 TMP=F:/c_overflow_2026-10-26/tmp TEMP=F:/c_overflow_2026-10-26/tmp
mkdir -p "$W/logs"
HELD=0
release() { [ $HELD -eq 1 ] || return; HELD=0; "$PY" "$LOCKPY" release --job "$JOB" --pid $$ >> "$W/logs/lock.log" 2>&1; echo "ZZPBN-released-0ZZ $(date +%T)" >> "$W/logs/chain.log"; }
trap 'release' EXIT INT TERM
"$PY" "$LOCKPY" acquire --job "$JOB" --pid $$ --max-wait-s 3600 --out "$W/logs/lock.json" >> "$W/logs/lock.log" 2>&1
if ! grep -q "\"ok\": true" "$W/logs/lock.json" 2>/dev/null; then echo "ZZPBN-lock-1ZZ $(date +%T)" >> "$W/logs/chain.log"; exit 3; fi
HELD=1
echo "ZZPBN-lock-0ZZ $(date +%T)" >> "$W/logs/chain.log"
MAXW=0; SUF=""; [ "$SMOKE" = "1" ] && { MAXW=3; SUF="_smoke"; }
# the TRAIN pass needs the run's config with --r8-v9-eval-md5 naming the TRAIN v9 file (perc8 mk_cfg_trdiag.py, verbatim)
"$PY" -X utf8 "$C/mk_cfg_trdiag.py" E:/refcv8_eval_kit/ckpt/config.json "$W/config_trdiag.json" >> "$W/logs/chain.log" 2>&1
for S in ${SPLITS:-eval train}; do
  CFG=E:/refcv8_eval_kit/ckpt/config.json; [ "$S" = train ] && CFG="$W/config_trdiag.json"
  "$PY" -X utf8 "$C/pbn_capture.py" --split $S --remap "$C/remap_${S}_devbox.json" --out-dir "$W/cap_${S}${SUF}" \
      --config "$CFG" --max-windows $MAXW > "$W/logs/cap_${S}${SUF}.log" 2>&1
  rc=$?
  echo "ZZPBN-cap_${S}${SUF}-${rc}ZZ $(date +%T)" >> "$W/logs/chain.log"
  [ $rc -eq 0 ] || exit $rc
done
echo "ZZPBN-alldone-0ZZ $(date +%T)" >> "$W/logs/chain.log"
