#!/usr/bin/env bash
# podread_cycle.sh : pod-side read of card PREREG_V8F_PROOF_ADD52 (CYCLE10K). Same readers and rules as the pod reads of cards ADD47 to ADD49
# (the dev box's commit memory is left to another session's long read; logged in TIMES_NOTE.md before this ran).
# Order: control e1 first (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf, same reader command); only if e1 passes, the read of card ADD52.
# CPU only (CUDA_VISIBLE_DEVICES=-1). Starts no rollout and no training. Last line: ZZ PODREAD-CYCLE DONE or ZZ PODREAD-CYCLE STOP <why>.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-CYCLE STOP $*"; exit 3; }
[ -f "$W/CYCLE10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
ROOTS=$W/t1
OUT=$P/out; mkdir -p "$OUT/e1y" "$OUT/read_cycle"
PL=$P/root/integ10k/pull; mkdir -p "$PL"
for d in "$W"/t1/*_s[01]; do n=$(basename "$d"); [ -e "$PL/$n" ] || ln -s "$d" "$PL/$n"; done
[ -e "$PL/small_cycle" ] || ln -s "$W/small_cycle" "$PL/small_cycle"
[ -f "$PL/small_cycle/CMP_RLcM10.txt" ] || stop "NO-CMP-FILE"

# ---- control e1: the banked read of card ADD46, same command (pairs and order of read_i10k_n2.sh)
PA="I10RR:I10R0,I10RR:I10RLb,I10RLb:I10R0,I10RR0:I10R0,I10RRid:I10R0,I10RRx:I10R0"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/e1y/hw" > "$OUT/e1y/hw.log" 2>&1 ); r1=$?; echo "[$(ts)] e1 headway rc $r1"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/e1y/q3" > "$OUT/e1y/q3.log" 2>&1 ); r2=$?; echo "[$(ts)] e1 families rc $r2"
[ "$r1" = 0 ] && [ "$r2" = 0 ] || { tail -n 6 "$OUT/e1y/hw.log" "$OUT/e1y/q3.log" | cut -c1-300; stop "E1-READER-RC $r1 $r2"; }
"$PY" "$P/podread_envcheck.py" --bank "$P/bank" --got "$OUT/e1y" --out "$OUT/e1y/ENV_E1.json" || stop "E1-FAIL"

# ---- the read of card ADD52 (pairs of the card)
PAIRS="I10R0M5:I10R0,I10RLcM5:I10RLc,I10RQ50M5:I10RQ50,I10RLcM5:I10R0M5,I10RQ50M5:I10R0M5,I10RLc:I10R0,I10RQ50:I10R0"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/read_cycle/hw" > "$OUT/read_cycle/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/read_cycle/q3" > "$OUT/read_cycle/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
"$PY" "$P/root/integ10k/read_cycle10k.py" --read-dir "$OUT/read_cycle" --pull "$PL" --reused "$P/root/integ10k/CYCLE10K_REUSED.json" --out "$OUT/read_cycle/READ_CYCLE10K.json" > "$OUT/read_cycle/read_cycle10k.log" 2>&1
echo "[$(ts)] read rc $?"
tail -n 130 "$OUT/read_cycle/read_cycle10k.log" | cut -c1-430
echo "ZZ PODREAD-CYCLE DONE $(ts)"
