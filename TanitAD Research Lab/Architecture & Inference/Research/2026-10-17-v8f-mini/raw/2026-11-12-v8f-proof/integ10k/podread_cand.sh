#!/usr/bin/env bash
# podread_cand.sh -- pod-side read of card PREREG_V8F_PROOF_ADD65 (CAND10K). Written by make_cand10k.py from podread_rulefast.sh (card ADD62) with the names, pairs and files of this card.
# Same readers and rules as the pod reads of cards ADD47 to ADD58.
# Order: control e1 first (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf, same reader command); only if e1 passes, the read of card ADD65.
# CPU only (CUDA_VISIBLE_DEVICES=-1). Starts no rollout and no training. Last line: ZZ PODREAD-CAND DONE or ZZ PODREAD-CAND STOP <why>.
# Files it needs on the pod (pushed by the lead before it runs): $P/root/integ10k/read_cand10k.py and $P/root/integ10k/CAND10K_REUSED.json (sha256 of the result files and rule logs of the
# reused runs, recorded before the queue started); the two tool-check outputs and the wrapper copy are written by the queue into $W/cand_test/, the two compare files into $W/small_cand/.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-CAND STOP $*"; exit 3; }
[ -f "$W/CAND10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
ROOTS=$W/t1
OUT=$P/out; mkdir -p "$OUT/e1cd" "$OUT/read_cand"
PL=$P/root/integ10k/pull; mkdir -p "$PL"
for d in "$W"/t1/*_s[01]; do n=$(basename "$d"); [ -e "$PL/$n" ] || ln -s "$d" "$PL/$n"; done
[ -e "$PL/small_cand" ] || ln -s "$W/small_cand" "$PL/small_cand"
[ -f "$PL/small_cand/CMP_R0KID.txt" ] || stop "NO-CMP-FILE-C1"
[ -f "$PL/small_cand/CMP_RLcKIDp.txt" ] || stop "NO-CMP-FILE-C2"
[ -f "$P/root/integ10k/read_cand10k.py" ] || stop "NO-READER-FILE"
[ -f "$P/root/integ10k/CAND10K_REUSED.json" ] || stop "NO-REUSED-RECORD"
[ -f "$W/cand_test/CHECK_RULE_EVERY_RZGF.txt" ] || stop "NO-TOOL-CHECK-OUTPUT-RULE"
[ -f "$W/cand_test/CHECK_ACC_GAIN_RZGF.txt" ] || stop "NO-TOOL-CHECK-OUTPUT-GAIN"

# ---- control e1: the banked read of card ADD46, same command (pairs and order of read_i10k_n2.sh)
PA="I10RR:I10R0,I10RR:I10RLb,I10RLb:I10R0,I10RR0:I10R0,I10RRid:I10R0,I10RRx:I10R0"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/e1cd/hw" > "$OUT/e1cd/hw.log" 2>&1 ); r1=$?; echo "[$(ts)] e1 headway rc $r1"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/e1cd/q3" > "$OUT/e1cd/q3.log" 2>&1 ); r2=$?; echo "[$(ts)] e1 families rc $r2"
[ "$r1" = 0 ] && [ "$r2" = 0 ] || { tail -n 6 "$OUT/e1cd/hw.log" "$OUT/e1cd/q3.log" | cut -c1-300; stop "E1-READER-RC $r1 $r2"; }
"$PY" "$P/podread_envcheck.py" --bank "$P/bank" --got "$OUT/e1cd" --out "$OUT/e1cd/ENV_E1.json" || stop "E1-FAIL"

# ---- the read of card ADD65 (pairs of the card: RK, FK and CC per form, the pairs beside)
PAIRS="I10RLcK:I10R0G0X,I10RQ50K:I10R0G0X,I10RLcKF5:I10RLcK,I10RQ50KF5:I10RQ50K,I10RLcKF5:I10RLc,I10RQ50KF5:I10RQ50,I10RLcK:I10RLc,I10RQ50K:I10RQ50,I10RLcKF5:I10RLcF5,I10RQ50KF5:I10RQ50F5,I10RQ50KF5:I10RLcKF5,I10RLcKF5:I10R0,I10RQ50KF5:I10R0"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/read_cand/hw" > "$OUT/read_cand/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/read_cand/q3" > "$OUT/read_cand/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
"$PY" "$P/root/integ10k/read_cand10k.py" --read-dir "$OUT/read_cand" --pull "$PL" --reused "$P/root/integ10k/CAND10K_REUSED.json" --wrapper "$W/build/v8f_t1_eval_reader_rule_rzgf.py" --test-wrapper "$W/cand_test/v8f_t1_eval_reader_rule_rzgf.py" --test-out-rule "$W/cand_test/CHECK_RULE_EVERY_RZGF.txt" --test-out-gain "$W/cand_test/CHECK_ACC_GAIN_RZGF.txt" --e1 "$OUT/e1cd/ENV_E1.json" --out "$OUT/read_cand/READ_CAND10K.json" > "$OUT/read_cand/read_cand10k.log" 2>&1
echo "[$(ts)] read rc $?"
tail -n 230 "$OUT/read_cand/read_cand10k.log" | cut -c1-430
echo "ZZ PODREAD-CAND DONE $(ts)"
