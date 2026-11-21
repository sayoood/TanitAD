#!/usr/bin/env bash
# podread_rule1s.sh -- written from podread_accwin.sh / podread_brake.sh: pod-side read of card PREREG_V8F_PROOF_ADD59 (RULE1S10K).
# Same readers and rules as the pod reads of cards ADD47 to ADD57 (the dev box's commit memory is left to another session's long read).
# Order: control e1 first (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf, same reader command); only if e1 passes, the read of card ADD59.
# CPU only (CUDA_VISIBLE_DEVICES=-1). Starts no rollout and no training. Last line: ZZ PODREAD-RULE1S DONE or ZZ PODREAD-RULE1S STOP <why>.
# Needs in $P/root/integ10k (pushed by the lead before): read_rule1s10k.py, RULE1S10K_REUSED.json (sha256 of the 16 reused result files, recorded before the queue).
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-RULE1S STOP $*"; exit 3; }
[ -f "$W/RULE1S10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
ROOTS=$W/t1
OUT=$P/out; mkdir -p "$OUT/e1r" "$OUT/read_rule1s"
PL=$P/root/integ10k/pull; mkdir -p "$PL"
for d in "$W"/t1/*_s[01]; do n=$(basename "$d"); [ -e "$PL/$n" ] || ln -s "$d" "$PL/$n"; done
[ -e "$PL/small_rule1s" ] || ln -s "$W/small_rule1s" "$PL/small_rule1s"
[ -f "$PL/small_rule1s/CMP_R0BkID.txt" ] || stop "NO-CMP-FILE"
[ -f "$P/root/integ10k/RULE1S10K_REUSED.json" ] || stop "NO-REUSED-JSON"

# ---- control e1: the banked read of card ADD46, same command (pairs and order of read_i10k_n2.sh)
PA="I10RR:I10R0,I10RR:I10RLb,I10RLb:I10R0,I10RR0:I10R0,I10RRid:I10R0,I10RRx:I10R0"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/e1r/hw" > "$OUT/e1r/hw.log" 2>&1 ); r1=$?; echo "[$(ts)] e1 headway rc $r1"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/e1r/q3" > "$OUT/e1r/q3.log" 2>&1 ); r2=$?; echo "[$(ts)] e1 families rc $r2"
[ "$r1" = 0 ] && [ "$r2" = 0 ] || { tail -n 6 "$OUT/e1r/hw.log" "$OUT/e1r/q3.log" | cut -c1-300; stop "E1-READER-RC $r1 $r2"; }
"$PY" "$P/podread_envcheck.py" --bank "$P/bank" --got "$OUT/e1r" --out "$OUT/e1r/ENV_E1.json" || stop "E1-FAIL"

# ---- the read of card ADD59 (pairs of card section 4; A:B = A minus B)
#   CY (1.0 s minus 2.0 s, backward)   RU (rule minus no rule, backward, at 1.0 s and at 2.0 s)   GB (1.0 s backward minus banked 2.0 s pinned)
#   RB (2.0 s backward minus banked 2.0 s pinned)   ZERO beside (rule minus no rule at 1.0 s; rule at 1.0 s minus banked rule at 2.0 s)
CY="I10R0B5:I10R0Bk10,I10RLcBk5:I10RLcBk10,I10RQ50Bk5:I10RQ50Bk10"
RU="I10RLcBk5:I10R0B5,I10RQ50Bk5:I10R0B5,I10RLcBk10:I10R0Bk10,I10RQ50Bk10:I10R0Bk10"
GB="I10RLcBk5:I10RLc,I10RQ50Bk5:I10RQ50"
RB="I10R0Bk10:I10R0,I10RLcBk10:I10RLc,I10RQ50Bk10:I10RQ50"
ZR="I10RLcZ5:I10R0Z5,I10RQ50Z5:I10R0Z5,I10RLcZ5:I10RLc,I10RQ50Z5:I10RQ50"
PAIRS="$CY,$RU,$GB,$RB,$ZR"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/read_rule1s/hw" > "$OUT/read_rule1s/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/read_rule1s/q3" > "$OUT/read_rule1s/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
"$PY" "$P/root/integ10k/read_rule1s10k.py" --read-dir "$OUT/read_rule1s" --pull "$PL" --reused "$P/root/integ10k/RULE1S10K_REUSED.json" --wrapper "$W/build/v8f_t1_eval_reader_rule_rz.py" --out "$OUT/read_rule1s/READ_RULE1S10K.json" > "$OUT/read_rule1s/read_rule1s10k.log" 2>&1
echo "[$(ts)] read rc $?"
tail -n 230 "$OUT/read_rule1s/read_rule1s10k.log" | cut -c1-430
echo "ZZ PODREAD-RULE1S DONE $(ts)"
