#!/usr/bin/env bash
# podread_agerule.sh -- pod-side read of part A of card PREREG_V8F_PROOF_ADD71 (AGERULE10K); CPU only. Adapted from podread_longroll.sh and podread_longroll_n3.sh (card ADD70).
# Order: control e1 first (the pod reproduces the banked dev-box read of card ADD46, same reader command); only if e1 passes, ONE call of read_agerule10k.py (the reader of card ADD70
# under notes N3 and N4 with the tables of card ADD71 and control a1 against the result of card ADD70 as re-read under note N4). Never reads twice into one folder. No rm, no mv, no kill. Starts no rollout.
# Last line: ZZ PODREAD-AGERULE DONE or ZZ PODREAD-AGERULE STOP <why>.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
LR=${LR_DIR:-$W/longroll}; TST=${LR_TEST_DIR:-$W/lr_test}; BLD=${LR_BUILD_DIR:-$W/build}
GRIDFULL=${LR_GRID_FULL:-/workspace/v8proof/hierrep/pins/eval_grid_windows.json}
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-AGERULE STOP $*"; exit 3; }
[ -f "$W/AGERULE10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
ROOTS=$W/t1
OUT=$P/out; AR=$OUT/read_agerule
[ -e "$AR/READ_AGERULE10K.json" ] && stop "ALREADY-READ out/read_agerule/READ_AGERULE10K.json exists"
mkdir -p "$OUT/e1ar" "$AR"
PL=$P/root/integ10k/pull; mkdir -p "$PL"
for d in "$W"/t1/*_s[01]; do n=$(basename "$d"); [ -e "$PL/$n" ] || ln -s "$d" "$PL/$n"; done
ADD67JSON=$P/out/read_headbound/READ_HEADBOUND10K.json
ADD70JSON=$OUT/read_longroll_n5/READ_LONGROLL10K_N5.json
HF=$OUT/read_longroll/POD_HASHES.txt
for f in "$P/root/integ10k/read_longroll10k.py" "$P/root/integ10k/read_longroll10k_n3.py" "$P/root/integ10k/read_longroll10k_n4.py" "$P/root/integ10k/read_agerule10k.py" "$P/root/integ10k/LONGROLL10K_REUSED.json" "$P/root/integ10k/strat_roll_read.py" "$HF" "$ADD67JSON" "$ADD70JSON" \
         "$LR/grid_p30.json" "$LR/grid_p60.json" "$GRIDFULL" "$TST/CHECK_PREROLL_B.txt" "$TST/v8f_t1_eval_reader_rule_rzgfhq.py" "$BLD/v8f_t1_eval_reader_rule_rzgfhq.py" "$BLD/v8f_t1_eval_reader_rule_rzgfh.py"; do
  [ -f "$f" ] || stop "MISSING-INPUT $f"
done
grep -q '"controls_all_pass": true' "$ADD70JSON" || stop "ADD70-READ-NOT-ADMITTED"
for r in RLcP30Hb20 RQ50P30Hb20 R0C10XP30Hb20 R0P30Hb20 R0KP30Hb20 RLcKF5P30Hb20 RQ50KF5P30Hb20 RLcHb20 RQ50Hb20 R0C10XHb20 R0KHb20 R0Hb20 RLcKF5Hb20 RQ50KF5Hb20 R0; do
  for s in 0 1; do [ -d "$W/t1/I10${r}_s$s" ] || stop "MISSING-RUN I10${r}_s$s"; done
done
sha256sum "$P/root/integ10k/read_longroll10k.py" "$P/root/integ10k/read_longroll10k_n3.py" "$P/root/integ10k/read_longroll10k_n4.py" "$P/root/integ10k/read_agerule10k.py" "$ADD70JSON" > "$AR/READER_HASHES.txt"
cut -c1-16,65- "$AR/READER_HASHES.txt"

# ---- control e1: the banked read of card ADD46, same command (pairs and order of read_i10k_n2.sh)
PA="I10RR:I10R0,I10RR:I10RLb,I10RLb:I10R0,I10RR0:I10R0,I10RRid:I10R0,I10RRx:I10R0"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/e1ar/hw" > "$OUT/e1ar/hw.log" 2>&1 ); r1=$?; echo "[$(ts)] e1 headway rc $r1"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/e1ar/q3" > "$OUT/e1ar/q3.log" 2>&1 ); r2=$?; echo "[$(ts)] e1 families rc $r2"
[ "$r1" = 0 ] && [ "$r2" = 0 ] || { tail -n 6 "$OUT/e1ar/hw.log" "$OUT/e1ar/q3.log" | cut -c1-300; stop "E1-READER-RC $r1 $r2"; }
"$PY" "$P/podread_envcheck.py" --bank "$P/bank" --got "$OUT/e1ar" --out "$OUT/e1ar/ENV_E1.json" || stop "E1-FAIL"

# ---- the read of part A
( cd "$P/root/integ10k" && "$PY" read_agerule10k.py --add70-n3-json "$ADD70JSON" --read-dir "$AR" --pull "$PL" --reused "$P/root/integ10k/LONGROLL10K_REUSED.json" --code "$P/code" --grid-dir "$LR" --grid-full "$GRIDFULL" \
  --wrapper "$BLD/v8f_t1_eval_reader_rule_rzgfhq.py" --parent-wrapper "$BLD/v8f_t1_eval_reader_rule_rzgfh.py" --test-wrapper "$TST/v8f_t1_eval_reader_rule_rzgfhq.py" --check-preroll "$TST/CHECK_PREROLL_B.txt" \
  --e1 "$OUT/e1ar/ENV_E1.json" --add67-json "$ADD67JSON" --add67-hw "$P/out/read_headbound/hw/RESULT_HEADWAY.json" --add67-q3 "$P/out/read_headbound/q3/RESULT_Q3.json" --pod-hashes "$HF" \
  --w "$W" --tree "$W/tree" --out "$AR/READ_AGERULE10K.json" > "$AR/read_agerule10k.log" 2>&1 )
echo "[$(ts)] read rc $?"
grep -E '^ZZ READ-AGERULE|^== a control did not pass| FAIL | NOT COMPUTABLE |Traceback|Error' "$AR/read_agerule10k.log" | cut -c1-600 | head -n 14
echo "ZZ PODREAD-AGERULE DONE $(ts)"
