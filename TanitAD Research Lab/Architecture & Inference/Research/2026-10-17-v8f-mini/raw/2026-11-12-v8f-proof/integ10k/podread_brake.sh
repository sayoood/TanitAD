#!/usr/bin/env bash
# podread_brake.sh -- written by make_brake10k.py from podread_feas.sh: the pod read of card PREREG_V8F_PROOF_ADD57 (BRAKE10K).
# podread_brake.sh : pod-side read of card PREREG_V8F_PROOF_ADD53 (BRAKE10K). Same readers and rules as the pod reads of cards ADD47 to ADD49
# (the dev box's commit memory is left to another session's long read; logged in TIMES_NOTE.md before this ran).
# Order: control e1 first (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf, same reader command); only if e1 passes, the read of card ADD57.
# CPU only (CUDA_VISIBLE_DEVICES=-1). Starts no rollout and no training. Last line: ZZ PODREAD-BRAKE DONE or ZZ PODREAD-BRAKE STOP <why>.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-BRAKE STOP $*"; exit 3; }
[ -f "$W/BRAKE10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
ROOTS=$W/t1
OUT=$P/out; mkdir -p "$OUT/e1h" "$OUT/read_brake"
PL=$P/root/integ10k/pull; mkdir -p "$PL"
for d in "$W"/t1/*_s[01]; do n=$(basename "$d"); [ -e "$PL/$n" ] || ln -s "$d" "$PL/$n"; done
[ -e "$PL/small_brake" ] || ln -s "$W/small_brake" "$PL/small_brake"
[ -f "$PL/small_brake/CMP_RLcB6.txt" ] || stop "NO-CMP-FILE"

# ---- control e1: the banked read of card ADD46, same command (pairs and order of read_i10k_n2.sh)
PA="I10RR:I10R0,I10RR:I10RLb,I10RLb:I10R0,I10RR0:I10R0,I10RRid:I10R0,I10RRx:I10R0"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/e1h/hw" > "$OUT/e1h/hw.log" 2>&1 ); r1=$?; echo "[$(ts)] e1 headway rc $r1"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/e1h/q3" > "$OUT/e1h/q3.log" 2>&1 ); r2=$?; echo "[$(ts)] e1 families rc $r2"
[ "$r1" = 0 ] && [ "$r2" = 0 ] || { tail -n 6 "$OUT/e1h/hw.log" "$OUT/e1h/q3.log" | cut -c1-300; stop "E1-READER-RC $r1 $r2"; }
"$PY" "$P/podread_envcheck.py" --bank "$P/bank" --got "$OUT/e1h" --out "$OUT/e1h/ENV_E1.json" || stop "E1-FAIL"

# ---- the read of card ADD57 (pairs of the card)
PAIRS="I10RLcB8:I10RLc,I10RQ50B8:I10RQ50,I10RLcB10:I10RLc,I10RLcB10:I10RLcB8,I10RLcB8:I10R0,I10RQ50B8:I10R0,I10RQ50B8:I10RLcB8"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/read_brake/hw" > "$OUT/read_brake/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/read_brake/q3" > "$OUT/read_brake/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
"$PY" "$P/root/integ10k/read_brake10k.py" --read-dir "$OUT/read_brake" --pull "$PL" --reused "$P/root/integ10k/TAUCURVE10K_REUSED.json" --wrapper "$W/build/v8f_t1_eval_reader_rule_bm.py" --out "$OUT/read_brake/READ_BRAKE10K.json" > "$OUT/read_brake/read_brake10k.log" 2>&1
echo "[$(ts)] read rc $?"
tail -n 230 "$OUT/read_brake/read_brake10k.log" | cut -c1-430
echo "ZZ PODREAD-BRAKE DONE $(ts)"
