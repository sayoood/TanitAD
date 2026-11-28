#!/usr/bin/env bash
# podread_taucurve.sh : pod-side read of card PREREG_V8F_PROOF_ADD51 (TAUCURVE10K). Same readers and rules as the pod reads of cards ADD47 to ADD49
# (the dev box's commit memory is left to another session's long read; logged in TIMES_NOTE.md before this ran).
# Order: control e1 first (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf, same reader command); only if e1 passes, the read of card ADD51.
# CPU only (CUDA_VISIBLE_DEVICES=-1). Starts no rollout and no training. Last line: ZZ PODREAD-TAUCURVE DONE or ZZ PODREAD-TAUCURVE STOP <why>.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-TAUCURVE STOP $*"; exit 3; }
[ -f "$W/TAUCURVE10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
ROOTS=$W/t1
OUT=$P/out; mkdir -p "$OUT/e1t" "$OUT/read_taucurve"
PL=$P/root/integ10k/pull; mkdir -p "$PL"
for d in "$W"/t1/*_s[01]; do n=$(basename "$d"); [ -e "$PL/$n" ] || ln -s "$d" "$PL/$n"; done
[ -e "$PL/small_taucurve" ] || ln -s "$W/small_taucurve" "$PL/small_taucurve"
[ -f "$PL/small_taucurve/CMP_RLcT12.txt" ] || stop "NO-CMP-FILE"

# ---- control e1: the banked read of card ADD46, same command (pairs and order of read_i10k_n2.sh)
PA="I10RR:I10R0,I10RR:I10RLb,I10RLb:I10R0,I10RR0:I10R0,I10RRid:I10R0,I10RRx:I10R0"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/e1t/hw" > "$OUT/e1t/hw.log" 2>&1 ); r1=$?; echo "[$(ts)] e1 headway rc $r1"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/e1t/q3" > "$OUT/e1t/q3.log" 2>&1 ); r2=$?; echo "[$(ts)] e1 families rc $r2"
[ "$r1" = 0 ] && [ "$r2" = 0 ] || { tail -n 6 "$OUT/e1t/hw.log" "$OUT/e1t/q3.log" | cut -c1-300; stop "E1-READER-RC $r1 $r2"; }
"$PY" "$P/podread_envcheck.py" --bank "$P/bank" --got "$OUT/e1t" --out "$OUT/e1t/ENV_E1.json" || stop "E1-FAIL"

# ---- the read of card ADD51 (pairs of the card)
PAIRS="I10RLcT08:I10RLc,I10RLcT10:I10RLc,I10RLcT15:I10RLc,I10RLcT20:I10RLc,I10RQ50T08:I10RQ50,I10RQ50T10:I10RQ50,I10RQ50T15:I10RQ50,I10RQ50T20:I10RQ50,I10RQ50T15:I10RLc,I10RQ50T20:I10RLc,I10RLcT20:I10R0,I10RQ50T20:I10R0"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/read_taucurve/hw" > "$OUT/read_taucurve/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/read_taucurve/q3" > "$OUT/read_taucurve/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
"$PY" "$P/root/integ10k/read_taucurve10k.py" --read-dir "$OUT/read_taucurve" --pull "$PL" --reused "$P/root/integ10k/TAUCURVE10K_REUSED.json" --out "$OUT/read_taucurve/READ_TAUCURVE10K.json" > "$OUT/read_taucurve/read_taucurve10k.log" 2>&1
echo "[$(ts)] read rc $?"
tail -n 130 "$OUT/read_taucurve/read_taucurve10k.log" | cut -c1-430
echo "ZZ PODREAD-TAUCURVE DONE $(ts)"
