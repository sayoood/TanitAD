#!/bin/sh
# run_r8oracle.sh -- the R8-ORACLE GPU roll on the dev box (PREREG_R8_ORACLE.md + AMENDMENT-1, sealed), then the CPU analysis.
# DERIVED 2026-10-21 from run_bcand.sh (the refcv8 per-frame battery launcher): SAME gpu_lock.py acquire/release (the exclusive
# lock file C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock + the nvidia-smi gate), SAME env, SAME child argv as the battery's
# seed child with --os-only; the child is wrapped by r8oracle_roll.py.  The lock is taken ONCE and held across BOTH seeds, so the
# job never interleaves inside another job; it waits (polling, up to MAXWAIT s) while any battery holds the lock.
#   optional: WAIT_FOR_LOGS="<bwise.log> ..." -> before acquiring, wait until each named log carries ZZBWSENDZZ (to queue
#   strictly BEHIND named batteries: gpu_lock is a polling lock with no FIFO order).
# Markers ZZR8O<NAME>ZZ in $OUT/r8oracle.log.  Output: $OUT/{oracle_s0,oracle_s1}.npz, dump_s*/, roll_s*.json, R8ORACLE.json, RESULT.md
HERE=C:/Users/Admin/refcv8_pf/r8oracle/code
BAT=C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code
BANK=C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf
CK=E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt; CKMD5=b4364029b68f29a33eeb4aebbbbd002b
CFG=E:/refcv8_eval_kit/ckpt/config.json
OUT=${OUT:-C:/Users/Admin/refcv8_pf/r8oracle/run_48852ema_pf}
JOB=${JOB:-r8oracle-48852ema-pf}
MAXWAIT=${MAXWAIT:-86400}
ARMS=${ARMS:-GT_CONS,GT_HYP,GT_BOTH,GT_SPEED_ONLY,OWN_ARGMAX,GT_HYP_SWAP,GT_BOTH_SWAP,SELF_NAN}   # prereg section 2 + amendment 1
SEEDS=${SEEDS:-"1 0"}                 # seed 1 first: its C1 (OWN == banked dump_s1 <= 1e-5) is the binding reproduction control
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
mkdir -p "$OUT"; LOG="$OUT/r8oracle.log"
mark() { echo "ZZR8O$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
mark START "pid $$ seeds=$SEEDS"
[ "$(md5sum "$CK" | cut -c1-32)" = "$CKMD5" ] || { mark MD5BAD ""; exit 2; }
[ -z "$R8O_MUTATE_SELFNAN_MIN" ] || { mark MUTATIONSET "R8O_MUTATE_SELFNAN_MIN is a smoke-only mutation; refusing a real run"; exit 5; }
for f in $WAIT_FOR_LOGS; do
  mark WAITFOR "$f"
  until grep -q ZZBWSENDZZ "$f" 2>/dev/null; do sleep 60; done
  mark WAITED "$f"
done
rm -f "$OUT/lock.json"
"$PY" "$BAT/gpu_lock.py" acquire --job "$JOB" --pid $$ --max-wait-s "$MAXWAIT" --out "$OUT/lock.json" >> "$OUT/lockwait.log" 2>&1
"$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$OUT/lock.json" || { mark NOLOCK ""; exit 3; }
trap '"$PY" "$BAT/gpu_lock.py" release --job "$JOB" --pid $$ >> "$OUT/lockwait.log" 2>&1; mark RELEASED trap' EXIT
trap 'exit 143' TERM INT HUP
mark LOCKED "$JOB"
for s in $SEEDS; do
  T0=$(date +%s)
  ( export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 R8_NO_POSE_SYNC=1
    timeout 28800 "$PY" -X utf8 "$HERE/r8oracle_roll.py" --oracle-out "$OUT/oracle_s$s.npz" --arms "$ARMS" -- \
        --ckpt "$CK" --config "$CFG" --seed "$s" --dump-dir "$OUT/dump_s$s" --windows-json "$BANK/windows_s$s.json" \
        --device cuda --out-json "$OUT/roll_s$s.json" --os-only > "$OUT/roll_s$s.log" 2>&1 )
  RC=$?
  # assert on the ARTIFACT, never on the exit code: a final oracle record over all 4,754 windows
  OK=$("$PY" -I -c "import json,sys,numpy as np; z=np.load(sys.argv[1],allow_pickle=True); m=json.loads(str(z['meta_json'])); print('OK' if (m.get('final') is True and m.get('n_windows')==4754) else 'BAD n=%s final=%s' % (m.get('n_windows'), m.get('final')))" "$OUT/oracle_s$s.npz" 2>/dev/null || echo "BAD no-artifact")
  mark "ROLL_s$s" "rc=$RC $(( $(date +%s) - T0 )) s artifact=$OK"
  [ "$OK" = "OK" ] || { mark ABORT "seed $s"; exit 4; }
done
"$PY" "$BAT/gpu_lock.py" release --job "$JOB" --pid $$ >> "$OUT/lockwait.log" 2>&1; mark RELEASED explicit; trap - EXIT
# ---- CPU analysis (no GPU, after the lock is released): R8ORACLE.json + RESULT.md, mechanical from the sealed prereg
( export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=6 R8_NO_POSE_SYNC=1
  "$PY" -X utf8 "$HERE/r8oracle_analyze.py" --root "$OUT" --seeds 0,1 --arms "$ARMS" > "$OUT/analyze.log" 2>&1 )
[ -s "$OUT/R8ORACLE.json" ] && [ -s "$OUT/RESULT.md" ] && mark ANALYZED "$(grep -m1 -o 'Verdict[^*]*: [^*]*' "$OUT/RESULT.md")" || mark ANALYZEFAIL "see analyze.log"
mark END ""
