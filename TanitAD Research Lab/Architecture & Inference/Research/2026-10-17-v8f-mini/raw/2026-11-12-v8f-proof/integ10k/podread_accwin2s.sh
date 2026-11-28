#!/usr/bin/env bash
# podread_accwin2s.sh -- pod-side read of card PREREG_V8F_PROOF_ADD58 (ACCWIN2S10K). Written from podread_accwin.sh (card ADD55) with the names, pairs and files of this card.
# Same readers and rules as the pod reads of cards ADD47 to ADD55.
# Order: control e1 first (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf, same reader command); only if e1 passes, the read of card ADD58.
# CPU only (CUDA_VISIBLE_DEVICES=-1). Starts no rollout and no training. Last line: ZZ PODREAD-ACCWIN2S DONE or ZZ PODREAD-ACCWIN2S STOP <why>.
# Files it needs on the pod (pushed by the lead before it runs): $P/root/integ10k/read_accwin2s10k.py and $P/root/integ10k/ACCWIN2S10K_REUSED.json (sha256 of the result files of the
# reused runs, recorded before the queue started); the tool check output and the wrapper copy are written by the queue into $W/accwin2s_test/.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-ACCWIN2S STOP $*"; exit 3; }
[ -f "$W/ACCWIN2S10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
ROOTS=$W/t1
OUT=$P/out; mkdir -p "$OUT/e1v" "$OUT/read_accwin2s"
PL=$P/root/integ10k/pull; mkdir -p "$PL"
for d in "$W"/t1/*_s[01]; do n=$(basename "$d"); [ -e "$PL/$n" ] || ln -s "$d" "$PL/$n"; done
[ -e "$PL/small_accwin2s" ] || ln -s "$W/small_accwin2s" "$PL/small_accwin2s"
[ -f "$PL/small_accwin2s/CMP_R0VID.txt" ] || stop "NO-CMP-FILE"
[ -f "$P/root/integ10k/read_accwin2s10k.py" ] || stop "NO-READER-FILE"
[ -f "$P/root/integ10k/ACCWIN2S10K_REUSED.json" ] || stop "NO-REUSED-RECORD"
[ -f "$W/accwin2s_test/CHECK_TICK_READOUT_RV.txt" ] || stop "NO-TOOL-CHECK-OUTPUT"

# ---- control e1: the banked read of card ADD46, same command (pairs and order of read_i10k_n2.sh)
PA="I10RR:I10R0,I10RR:I10RLb,I10RLb:I10R0,I10RR0:I10R0,I10RRid:I10R0,I10RRx:I10R0"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/e1v/hw" > "$OUT/e1v/hw.log" 2>&1 ); r1=$?; echo "[$(ts)] e1 headway rc $r1"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/e1v/q3" > "$OUT/e1v/q3.log" 2>&1 ); r2=$?; echo "[$(ts)] e1 families rc $r2"
[ "$r1" = 0 ] && [ "$r2" = 0 ] || { tail -n 6 "$OUT/e1v/hw.log" "$OUT/e1v/q3.log" | cut -c1-300; stop "E1-READER-RC $r1 $r2"; }
"$PY" "$P/podread_envcheck.py" --bank "$P/bank" --got "$OUT/e1v" --out "$OUT/e1v/ENV_E1.json" || stop "E1-FAIL"

# ---- the read of card ADD58 (pairs of the card: SP, GC, PR, and the three beside pairs)
PAIRS="I10R0V5:I10R0M5,I10R0V5:I10R0,I10R0V10:I10R0,I10R0V5:I10R0W5,I10R0V5:I10R0Z5,I10R0V10:I10R0W10"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/read_accwin2s/hw" > "$OUT/read_accwin2s/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/read_accwin2s/q3" > "$OUT/read_accwin2s/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
"$PY" "$P/root/integ10k/read_accwin2s10k.py" --read-dir "$OUT/read_accwin2s" --pull "$PL" --reused "$P/root/integ10k/ACCWIN2S10K_REUSED.json" --wrapper "$W/build/v8f_t1_eval_reader_rule_rv.py" --test-wrapper "$W/accwin2s_test/v8f_t1_eval_reader_rule_rv.py" --test-out "$W/accwin2s_test/CHECK_TICK_READOUT_RV.txt" --out "$OUT/read_accwin2s/READ_ACCWIN2S10K.json" > "$OUT/read_accwin2s/read_accwin2s10k.log" 2>&1
echo "[$(ts)] read rc $?"
tail -n 230 "$OUT/read_accwin2s/read_accwin2s10k.log" | cut -c1-430
echo "ZZ PODREAD-ACCWIN2S DONE $(ts)"
