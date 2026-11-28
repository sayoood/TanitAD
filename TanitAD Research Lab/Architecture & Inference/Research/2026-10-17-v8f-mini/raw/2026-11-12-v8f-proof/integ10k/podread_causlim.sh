#!/usr/bin/env bash
# podread_causlim.sh -- pod-side read of card PREREG_V8F_PROOF_ADD61 (CAUSLIM10K). Written from podread_accwin2s.sh (card ADD58) with the names, pairs and files of this card.
# Same readers and rules as the pod reads of cards ADD47 to ADD58.
# Order: control e1 first (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf, same reader command); only if e1 passes, the read of card ADD61.
# CPU only (CUDA_VISIBLE_DEVICES=-1). Starts no rollout and no training. Last line: ZZ PODREAD-CAUSLIM DONE or ZZ PODREAD-CAUSLIM STOP <why>.
# Files it needs on the pod (pushed by the lead before it runs): $P/root/integ10k/read_causlim10k.py and $P/root/integ10k/CAUSLIM10K_REUSED.json (sha256 of the result files of the
# reused runs, recorded before the queue started); the two tool-check outputs and the wrapper copy are written by the queue into $W/causlim_test/, the two compare files into $W/small_causlim/.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-CAUSLIM STOP $*"; exit 3; }
[ -f "$W/CAUSLIM10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
ROOTS=$W/t1
OUT=$P/out; mkdir -p "$OUT/e1cl" "$OUT/read_causlim"
PL=$P/root/integ10k/pull; mkdir -p "$PL"
for d in "$W"/t1/*_s[01]; do n=$(basename "$d"); [ -e "$PL/$n" ] || ln -s "$d" "$PL/$n"; done
[ -e "$PL/small_causlim" ] || ln -s "$W/small_causlim" "$PL/small_causlim"
[ -f "$PL/small_causlim/CMP_R0CEIDc.txt" ] || stop "NO-CMP-FILE-C1"
[ -f "$PL/small_causlim/CMP_R0CEIDe.txt" ] || stop "NO-CMP-FILE-C2"
[ -f "$P/root/integ10k/read_causlim10k.py" ] || stop "NO-READER-FILE"
[ -f "$P/root/integ10k/CAUSLIM10K_REUSED.json" ] || stop "NO-REUSED-RECORD"
[ -f "$W/causlim_test/TEST_ACCEL_LIMIT_CHORD_RZC.txt" ] || stop "NO-TOOL-TEST-OUTPUT"
[ -f "$W/causlim_test/CHECK_TICK_READOUT_RZC.txt" ] || stop "NO-TOOL-CHECK-OUTPUT"

# ---- control e1: the banked read of card ADD46, same command (pairs and order of read_i10k_n2.sh)
PA="I10RR:I10R0,I10RR:I10RLb,I10RLb:I10R0,I10RR0:I10R0,I10RRid:I10R0,I10RRx:I10R0"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/e1cl/hw" > "$OUT/e1cl/hw.log" 2>&1 ); r1=$?; echo "[$(ts)] e1 headway rc $r1"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/e1cl/q3" > "$OUT/e1cl/q3.log" 2>&1 ); r2=$?; echo "[$(ts)] e1 families rc $r2"
[ "$r1" = 0 ] && [ "$r2" = 0 ] || { tail -n 6 "$OUT/e1cl/hw.log" "$OUT/e1cl/q3.log" | cut -c1-300; stop "E1-READER-RC $r1 $r2"; }
"$PY" "$P/podread_envcheck.py" --bank "$P/bank" --got "$OUT/e1cl" --out "$OUT/e1cl/ENV_E1.json" || stop "E1-FAIL"

# ---- the read of card ADD61 (pairs of the card: SP, the two PARTS pairs, GC, PR and the pair beside PR)
PAIRS="I10R0CE5:I10R0M5,I10R0CE5:I10R0C5,I10R0CE5:I10R0E5,I10R0CE5:I10R0,I10R0CE10:I10R0,I10R0CE10:I10R0E10"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/read_causlim/hw" > "$OUT/read_causlim/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/read_causlim/q3" > "$OUT/read_causlim/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
"$PY" "$P/root/integ10k/read_causlim10k.py" --read-dir "$OUT/read_causlim" --pull "$PL" --reused "$P/root/integ10k/CAUSLIM10K_REUSED.json" --wrapper "$W/build/v8f_t1_eval_reader_rule_rzc.py" --test-wrapper "$W/causlim_test/v8f_t1_eval_reader_rule_rzc.py" --test-out-chord "$W/causlim_test/TEST_ACCEL_LIMIT_CHORD_RZC.txt" --test-out-tick "$W/causlim_test/CHECK_TICK_READOUT_RZC.txt" --e1 "$OUT/e1cl/ENV_E1.json" --out "$OUT/read_causlim/READ_CAUSLIM10K.json" > "$OUT/read_causlim/read_causlim10k.log" 2>&1
echo "[$(ts)] read rc $?"
tail -n 230 "$OUT/read_causlim/read_causlim10k.log" | cut -c1-430
echo "ZZ PODREAD-CAUSLIM DONE $(ts)"
