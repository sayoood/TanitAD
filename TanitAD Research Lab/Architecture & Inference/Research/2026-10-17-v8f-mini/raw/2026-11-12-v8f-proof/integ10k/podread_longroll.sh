#!/usr/bin/env bash
# podread_longroll.sh -- pod-side read of card PREREG_V8F_PROOF_ADD70 (LONGROLL10K) with note N1. Adapted from podread_cyclebound.sh (card ADD69).
# Order: control e1 first (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf, same reader command); only if e1 passes, the read of card ADD70.
# The read itself is ONE call of read_longroll10k.py, which drives the engines of the headway and family readers in memory on the windows of each grid (the stock command lines of those two
# readers are not used for it: their own controls hold the 2,399 windows of the full grid). CPU only (CUDA_VISIBLE_DEVICES=-1). Starts no rollout and no training.
# Never reads twice into one folder: stops if out/read_longroll/READ_LONGROLL10K.json already exists. No rm, no mv, no kill.
# Last line: ZZ PODREAD-LONGROLL DONE or ZZ PODREAD-LONGROLL STOP <why>.
# Files it needs on the pod (pushed before it runs): $P/root/integ10k/read_longroll10k.py, $P/root/integ10k/LONGROLL10K_REUSED.json, $P/root/integ10k/strat_roll_read.py (card ADD66, unchanged).
# Files the QUEUE of the card must have written (names are the ones this script and the reader expect; every one can be moved with the env variables below):
#   $W/longroll/grid_p30.json, grid_p60.json (make_longroll_grid.py), $W/lr_test/CHECK_PREROLL_B.txt (check_preroll_b.py output) and $W/lr_test/v8f_t1_eval_reader_rule_rzgfhq.py (the wrapper
#   copy the check ran on), $W/build/v8f_t1_eval_reader_rule_rzgfhq.py (wrapper of record) and v8f_t1_eval_reader_rule_rzgfh.py (parent), the folders $W/t1/<RUN>_s<k>/ of the 23 rollouts of the card and
#   of the banked runs, the marker $W/LONGROLL10K_DONE.txt. The pairs that need the runs of card ADD67 are read only if out/read_headbound/READ_HEADBOUND10K.json has controls_all_pass true.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
LR=${LR_DIR:-$W/longroll}; TST=${LR_TEST_DIR:-$W/lr_test}; BLD=${LR_BUILD_DIR:-$W/build}
GRIDFULL=${LR_GRID_FULL:-/workspace/v8proof/hierrep/pins/eval_grid_windows.json}
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-LONGROLL STOP $*"; exit 3; }
[ -f "$W/LONGROLL10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
ROOTS=$W/t1
OUT=$P/out; [ -e "$OUT/read_longroll/READ_LONGROLL10K.json" ] && stop "ALREADY-READ out/read_longroll/READ_LONGROLL10K.json exists"
mkdir -p "$OUT/e1lr" "$OUT/read_longroll"
PL=$P/root/integ10k/pull; mkdir -p "$PL"
for d in "$W"/t1/*_s[01]; do n=$(basename "$d"); [ -e "$PL/$n" ] || ln -s "$d" "$PL/$n"; done
ADD67JSON=$P/out/read_headbound/READ_HEADBOUND10K.json
[ -f "$P/root/integ10k/read_longroll10k.py" ] || stop "NO-READER-FILE"
[ -f "$P/root/integ10k/LONGROLL10K_REUSED.json" ] || stop "NO-REUSED-RECORD"
[ -f "$P/root/integ10k/strat_roll_read.py" ] || stop "NO-STRATEGIC-READER-FILE"
for f in "$LR/grid_p30.json" "$LR/grid_p60.json" "$GRIDFULL" "$TST/CHECK_PREROLL_B.txt" "$TST/v8f_t1_eval_reader_rule_rzgfhq.py" "$BLD/v8f_t1_eval_reader_rule_rzgfhq.py" "$BLD/v8f_t1_eval_reader_rule_rzgfh.py"; do
  [ -f "$f" ] || stop "MISSING-INPUT $f"
done
[ -f "$ADD67JSON" ] && grep -q '"controls_all_pass": true' "$ADD67JSON" || stop "ADD67-READ-ABSENT-OR-NOT-ADMITTED (control l6 and the fresh references need it)"
for r in R0P30Hb20 R0KP30Hb20 RLcKF5P30Hb20 RQ50KF5P30Hb20 R0P60Hb20 R0KP60Hb20 RLcKF5P60Hb20 RQ50KF5P60Hb20 R0KNlP30Hb20 R0KNlHb20; do
  for s in 0 1; do [ -d "$W/t1/I10${r}_s$s" ] || stop "MISSING-RUN I10${r}_s$s"; done
done
for r in R0LrID R0LrCarry R0LrFresh; do [ -d "$W/t1/I10${r}_s0" ] || stop "MISSING-RUN I10${r}_s0"; done
for r in R0 R0Hb20 R0KHb20 RLcKF5Hb20 RQ50KF5Hb20; do
  for s in 0 1; do [ -d "$W/t1/I10${r}_s$s" ] || stop "MISSING-BANKED-RUN I10${r}_s$s"; done
done

# ---- hashes of the files that exist only on the pod (recorded in the result; sha256  path)
HF=$OUT/read_longroll/POD_HASHES.txt
: > "$HF"
for f in "$LR/grid_p30.json" "$LR/grid_p60.json" "$GRIDFULL" "$TST/CHECK_PREROLL_B.txt" "$TST/v8f_t1_eval_reader_rule_rzgfhq.py" "$BLD/v8f_t1_eval_reader_rule_rzgfhq.py" "$BLD/v8f_t1_eval_reader_rule_rzgfh.py" ${LR_PINNED:-} ${QMAP_FILE:-}; do
  [ -f "$f" ] && sha256sum "$f" >> "$HF"
done
echo "[$(ts)] pod-side hashes: $(wc -l < "$HF") files"

# ---- control e1: the banked read of card ADD46, same command (pairs and order of read_i10k_n2.sh)
PA="I10RR:I10R0,I10RR:I10RLb,I10RLb:I10R0,I10RR0:I10R0,I10RRid:I10R0,I10RRx:I10R0"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/e1lr/hw" > "$OUT/e1lr/hw.log" 2>&1 ); r1=$?; echo "[$(ts)] e1 headway rc $r1"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/e1lr/q3" > "$OUT/e1lr/q3.log" 2>&1 ); r2=$?; echo "[$(ts)] e1 families rc $r2"
[ "$r1" = 0 ] && [ "$r2" = 0 ] || { tail -n 6 "$OUT/e1lr/hw.log" "$OUT/e1lr/q3.log" | cut -c1-300; stop "E1-READER-RC $r1 $r2"; }
"$PY" "$P/podread_envcheck.py" --bank "$P/bank" --got "$OUT/e1lr" --out "$OUT/e1lr/ENV_E1.json" || stop "E1-FAIL"

# ---- the read of card ADD70 (one call; the reader restricts every run to the windows of its grid in memory)
( cd "$P/root/integ10k" && "$PY" read_longroll10k.py --read-dir "$OUT/read_longroll" --pull "$PL" --reused "$P/root/integ10k/LONGROLL10K_REUSED.json" --code "$P/code" --grid-dir "$LR" --grid-full "$GRIDFULL" \
  --wrapper "$BLD/v8f_t1_eval_reader_rule_rzgfhq.py" --parent-wrapper "$BLD/v8f_t1_eval_reader_rule_rzgfh.py" --test-wrapper "$TST/v8f_t1_eval_reader_rule_rzgfhq.py" --check-preroll "$TST/CHECK_PREROLL_B.txt" \
  --e1 "$OUT/e1lr/ENV_E1.json" --add67-json "$ADD67JSON" --add67-hw "$P/out/read_headbound/hw/RESULT_HEADWAY.json" --add67-q3 "$P/out/read_headbound/q3/RESULT_Q3.json" --pod-hashes "$HF" \
  --w "$W" --tree "$W/tree" --out "$OUT/read_longroll/READ_LONGROLL10K.json" > "$OUT/read_longroll/read_longroll10k.log" 2>&1 )
echo "[$(ts)] read rc $?"
tail -n 230 "$OUT/read_longroll/read_longroll10k.log" | cut -c1-430
echo "ZZ PODREAD-LONGROLL DONE $(ts)"
