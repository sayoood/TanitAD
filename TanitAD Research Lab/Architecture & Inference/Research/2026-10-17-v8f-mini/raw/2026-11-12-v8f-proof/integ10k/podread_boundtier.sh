#!/usr/bin/env bash
# podread_boundtier.sh -- pod-side read of card PREREG_V8F_PROOF_ADD68 (BOUNDTIER10K). Adapted from podread_headbound.sh (card ADD67) with the names, pairs and files of this card.
# Same readers and rules as the pod reads of cards ADD47 to ADD67.
# Order: control e1 first (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf, same reader command); only if e1 passes, the read of card ADD68.
# CPU only (CUDA_VISIBLE_DEVICES=-1). Starts no rollout and no training. Never reads twice into one folder: stops if out/read_boundtier/READ_BOUNDTIER10K.json already exists.
# Last line: ZZ PODREAD-BOUNDTIER DONE or ZZ PODREAD-BOUNDTIER STOP <why>.
# Files it needs on the pod (pushed before it runs): $P/root/integ10k/read_boundtier10k.py and $P/root/integ10k/BOUNDTIER10K_REUSED.json (sha256 of the result files, tick files and rule logs of the own reused runs);
# the queue writes the three tool-check outputs and the wrapper copy into $W/bt_test/, the two compare files and the hashes of the four runs of card ADD67 into $W/small_bt/. The pairs that need the runs of
# card ADD67 are read only if out/read_headbound/READ_HEADBOUND10K.json exists with controls_all_pass true.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-BOUNDTIER STOP $*"; exit 3; }
[ -f "$W/BOUNDTIER10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
ROOTS=$W/t1
OUT=$P/out; [ -e "$OUT/read_boundtier/READ_BOUNDTIER10K.json" ] && stop "ALREADY-READ out/read_boundtier/READ_BOUNDTIER10K.json exists"
mkdir -p "$OUT/e1bt" "$OUT/read_boundtier"
PL=$P/root/integ10k/pull; mkdir -p "$PL"
for d in "$W"/t1/*_s[01]; do n=$(basename "$d"); [ -e "$PL/$n" ] || ln -s "$d" "$PL/$n"; done
[ -e "$PL/small_bt" ] || ln -s "$W/small_bt" "$PL/small_bt"
[ -f "$PL/small_bt/CMP_R0C10HbID.txt" ] || stop "NO-CMP-FILE-G1"
[ -f "$PL/small_bt/CMP_RLcHbID.txt" ] || stop "NO-CMP-FILE-G2"
[ -f "$PL/small_bt/REUSED_CAND_TAKEN.txt" ] || stop "NO-CAND-HASH-FILE"
[ -f "$P/root/integ10k/read_boundtier10k.py" ] || stop "NO-READER-FILE"
[ -f "$P/root/integ10k/BOUNDTIER10K_REUSED.json" ] || stop "NO-REUSED-RECORD"
[ -f "$W/bt_test/CHECK_HEAD_BOUND.txt" ] || stop "NO-TOOL-CHECK-OUTPUT-HEAD"
[ -f "$W/bt_test/CHECK_RULE_EVERY_RZGFH.txt" ] || stop "NO-TOOL-CHECK-OUTPUT-RULE"
[ -f "$W/bt_test/CHECK_ACC_GAIN_RZGFH.txt" ] || stop "NO-TOOL-CHECK-OUTPUT-GAIN"

# ---- control e1: the banked read of card ADD46, same command (pairs and order of read_i10k_n2.sh)
PA="I10RR:I10R0,I10RR:I10RLb,I10RLb:I10R0,I10RR0:I10R0,I10RRid:I10R0,I10RRx:I10R0"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/e1bt/hw" > "$OUT/e1bt/hw.log" 2>&1 ); r1=$?; echo "[$(ts)] e1 headway rc $r1"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/e1bt/q3" > "$OUT/e1bt/q3.log" 2>&1 ); r2=$?; echo "[$(ts)] e1 families rc $r2"
[ "$r1" = 0 ] && [ "$r2" = 0 ] || { tail -n 6 "$OUT/e1bt/hw.log" "$OUT/e1bt/q3.log" | cut -c1-300; stop "E1-READER-RC $r1 $r2"; }
"$PY" "$P/podread_envcheck.py" --bank "$P/bank" --got "$OUT/e1bt" --out "$OUT/e1bt/ENV_E1.json" || stop "E1-FAIL"

# ---- the read of card ADD68 (pairs of the card: the four settings of the bound, ACC, READOUT-K, the two candidate pairs, the pairs beside)
PAIRS="I10R0C10Hb20:I10R0C10,I10R0C10XHb20:I10R0C10X,I10RLcHb20:I10RLc,I10RQ50Hb20:I10RQ50,I10R0C10X:I10R0C10"
ADD67JSON=$P/out/read_headbound/READ_HEADBOUND10K.json
if [ -f "$ADD67JSON" ] && grep -q '"controls_all_pass": true' "$ADD67JSON"; then PAIRS="$PAIRS,I10R0C10XHb20:I10R0KHb20,I10R0KHb20:I10R0Hb20,I10RLcKF5Hb20:I10RLcHb20,I10RQ50KF5Hb20:I10RQ50Hb20,I10R0Hb05:I10R0Hb20,I10R0Hb10:I10R0Hb20,I10R0Hb40:I10R0Hb20,I10R0C10XHb20:I10R0Hb20"; echo "[$(ts)] card ADD67 read present with controls_all_pass true: the pairs that need its runs are read"; else echo "[$(ts)] card ADD67 read absent or its controls did not pass: the pairs that need its runs are not read"; fi
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/read_boundtier/hw" > "$OUT/read_boundtier/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/read_boundtier/q3" > "$OUT/read_boundtier/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
"$PY" "$P/root/integ10k/read_boundtier10k.py" --read-dir "$OUT/read_boundtier" --pull "$PL" --reused "$P/root/integ10k/BOUNDTIER10K_REUSED.json" --wrapper "$W/build/v8f_t1_eval_reader_rule_rzgfh.py" --test-wrapper "$W/bt_test/v8f_t1_eval_reader_rule_rzgfh.py" --test-out-head "$W/bt_test/CHECK_HEAD_BOUND.txt" --test-out-rule "$W/bt_test/CHECK_RULE_EVERY_RZGFH.txt" --test-out-gain "$W/bt_test/CHECK_ACC_GAIN_RZGFH.txt" --taken "$PL/small_bt/REUSED_CAND_TAKEN.txt" --add67-json "$ADD67JSON" --pairs-read "$PAIRS" --w "$W" --tree "$W/tree" --e1 "$OUT/e1bt/ENV_E1.json" --out "$OUT/read_boundtier/READ_BOUNDTIER10K.json" > "$OUT/read_boundtier/read_boundtier10k.log" 2>&1
echo "[$(ts)] read rc $?"
tail -n 230 "$OUT/read_boundtier/read_boundtier10k.log" | cut -c1-430
echo "ZZ PODREAD-BOUNDTIER DONE $(ts)"
