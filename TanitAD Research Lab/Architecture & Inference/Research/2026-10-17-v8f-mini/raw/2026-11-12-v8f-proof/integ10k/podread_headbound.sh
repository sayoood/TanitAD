#!/usr/bin/env bash
# podread_headbound.sh -- pod-side read of card PREREG_V8F_PROOF_ADD67 (HEADBOUND10K). Written by make_headbound10k.py from podread_cand.sh (card ADD65) with the names, pairs and files of this card.
# Same readers and rules as the pod reads of cards ADD47 to ADD58.
# Order: control e1 first (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf, same reader command); only if e1 passes, the read of card ADD67.
# CPU only (CUDA_VISIBLE_DEVICES=-1). Starts no rollout and no training. Last line: ZZ PODREAD-HEADBOUND DONE or ZZ PODREAD-HEADBOUND STOP <why>.
# Files it needs on the pod (pushed by the lead before it runs): $P/root/integ10k/read_headbound10k.py and $P/root/integ10k/HEADBOUND10K_REUSED.json (sha256 of the result files and rule logs of the
# reused runs, recorded before the queue started; the two runs of card ADD65 are hashed by the queue into $W/small_hb/REUSED_CAND_TAKEN.txt); the three tool-check outputs and the wrapper copy are written by the queue into $W/hb_test/, the two compare files into $W/small_hb/. The two pairs of the form HC are read only if the read of card ADD65 exists with controls_all_pass true.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-HEADBOUND STOP $*"; exit 3; }
[ -f "$W/HEADBOUND10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
ROOTS=$W/t1
OUT=$P/out; mkdir -p "$OUT/e1hb" "$OUT/read_headbound"
PL=$P/root/integ10k/pull; mkdir -p "$PL"
for d in "$W"/t1/*_s[01]; do n=$(basename "$d"); [ -e "$PL/$n" ] || ln -s "$d" "$PL/$n"; done
[ -e "$PL/small_hb" ] || ln -s "$W/small_hb" "$PL/small_hb"
[ -f "$PL/small_hb/CMP_R0HbID.txt" ] || stop "NO-CMP-FILE-H1"
[ -f "$PL/small_hb/CMP_R0KHbID.txt" ] || stop "NO-CMP-FILE-H2"
[ -f "$PL/small_hb/REUSED_CAND_TAKEN.txt" ] || stop "NO-CAND-HASH-FILE"
[ -f "$P/root/integ10k/read_headbound10k.py" ] || stop "NO-READER-FILE"
[ -f "$P/root/integ10k/HEADBOUND10K_REUSED.json" ] || stop "NO-REUSED-RECORD"
[ -f "$W/hb_test/CHECK_HEAD_BOUND.txt" ] || stop "NO-TOOL-CHECK-OUTPUT-HEAD"
[ -f "$W/hb_test/CHECK_RULE_EVERY_RZGFH.txt" ] || stop "NO-TOOL-CHECK-OUTPUT-RULE"
[ -f "$W/hb_test/CHECK_ACC_GAIN_RZGFH.txt" ] || stop "NO-TOOL-CHECK-OUTPUT-GAIN"

# ---- control e1: the banked read of card ADD46, same command (pairs and order of read_i10k_n2.sh)
PA="I10RR:I10R0,I10RR:I10RLb,I10RLb:I10R0,I10RR0:I10R0,I10RRid:I10R0,I10RRx:I10R0"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/e1hb/hw" > "$OUT/e1hb/hw.log" 2>&1 ); r1=$?; echo "[$(ts)] e1 headway rc $r1"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/e1hb/q3" > "$OUT/e1hb/q3.log" 2>&1 ); r2=$?; echo "[$(ts)] e1 families rc $r2"
[ "$r1" = 0 ] && [ "$r2" = 0 ] || { tail -n 6 "$OUT/e1hb/hw.log" "$OUT/e1hb/q3.log" | cut -c1-300; stop "E1-READER-RC $r1 $r2"; }
"$PY" "$P/podread_envcheck.py" --bank "$P/bank" --got "$OUT/e1hb" --out "$OUT/e1hb/ENV_E1.json" || stop "E1-FAIL"

# ---- the read of card ADD67 (pairs of the card: HP, HK and HC per form, the pairs beside)
PAIRS="I10R0Hb20:I10R0,I10R0KHb20:I10R0G0X,I10R0KHb10:I10R0KHb20,I10R0KHb20:I10R0,I10RLcKF5Hb20:I10RLc,I10RQ50KF5Hb20:I10RQ50"
CANDJSON=$P/out/read_cand/READ_CAND10K.json
if [ -f "$CANDJSON" ] && grep -q '"controls_all_pass": true' "$CANDJSON"; then PAIRS="$PAIRS,I10RLcKF5Hb20:I10RLcKF5,I10RQ50KF5Hb20:I10RQ50KF5"; echo "[$(ts)] card ADD65 read present with controls_all_pass true: the pairs HC are read"; else echo "[$(ts)] card ADD65 read absent or its controls did not pass: the pairs HC are not read"; fi
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/read_headbound/hw" > "$OUT/read_headbound/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/read_headbound/q3" > "$OUT/read_headbound/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
"$PY" "$P/root/integ10k/read_headbound10k.py" --read-dir "$OUT/read_headbound" --pull "$PL" --reused "$P/root/integ10k/HEADBOUND10K_REUSED.json" --wrapper "$W/build/v8f_t1_eval_reader_rule_rzgfh.py" --test-wrapper "$W/hb_test/v8f_t1_eval_reader_rule_rzgfh.py" --test-out-head "$W/hb_test/CHECK_HEAD_BOUND.txt" --test-out-rule "$W/hb_test/CHECK_RULE_EVERY_RZGFH.txt" --test-out-gain "$W/hb_test/CHECK_ACC_GAIN_RZGFH.txt" --taken "$PL/small_hb/REUSED_CAND_TAKEN.txt" --cand-json "$CANDJSON" --pairs-read "$PAIRS" --w "$W" --tree "$W/tree" --e1 "$OUT/e1hb/ENV_E1.json" --out "$OUT/read_headbound/READ_HEADBOUND10K.json" > "$OUT/read_headbound/read_headbound10k.log" 2>&1
echo "[$(ts)] read rc $?"
tail -n 230 "$OUT/read_headbound/read_headbound10k.log" | cut -c1-430
echo "ZZ PODREAD-HEADBOUND DONE $(ts)"
