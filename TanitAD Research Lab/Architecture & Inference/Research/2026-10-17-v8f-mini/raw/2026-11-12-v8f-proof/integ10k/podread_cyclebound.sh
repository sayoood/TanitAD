#!/usr/bin/env bash
# podread_cyclebound.sh -- pod-side read of card PREREG_V8F_PROOF_ADD69 (CYCLEBOUND10K). Adapted from podread_boundtier.sh (card ADD68) with the names, pairs and files of this card.
# Same readers and rules as the pod reads of cards ADD47 to ADD68.
# Order: control e1 first (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf, same reader command); only if e1 passes, the read of card ADD69.
# CPU only (CUDA_VISIBLE_DEVICES=-1). Starts no rollout and no training. Never reads twice into one folder: stops if out/read_cyclebound/READ_CYCLEBOUND10K.json already exists.
# Last line: ZZ PODREAD-CYCLEBOUND DONE or ZZ PODREAD-CYCLEBOUND STOP <why>.
# Files it needs on the pod (pushed before it runs): $P/root/integ10k/read_cyclebound10k.py and $P/root/integ10k/CYCLEBOUND10K_REUSED.json (sha256 of the result files, tick files and rule logs of the
# six own reused runs); the queue writes the three tool-check outputs and the wrapper copy into $W/cb_test/, the compare file and the hashes of the two runs of card ADD68 into $W/small_cb/.
# The hashes of the four runs of card ADD67 are the file $W/small_bt/REUSED_CAND_TAKEN.txt (written by the queue of card ADD68). The pairs that need the runs of card ADD67 are read only if
# out/read_headbound/READ_HEADBOUND10K.json has controls_all_pass true; the pairs that need the runs of card ADD68 only if out/read_boundtier/READ_BOUNDTIER10K.json has controls_all_pass true.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-CYCLEBOUND STOP $*"; exit 3; }
[ -f "$W/CYCLEBOUND10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
ROOTS=$W/t1
OUT=$P/out; [ -e "$OUT/read_cyclebound/READ_CYCLEBOUND10K.json" ] && stop "ALREADY-READ out/read_cyclebound/READ_CYCLEBOUND10K.json exists"
mkdir -p "$OUT/e1cb" "$OUT/read_cyclebound"
PL=$P/root/integ10k/pull; mkdir -p "$PL"
for d in "$W"/t1/*_s[01]; do n=$(basename "$d"); [ -e "$PL/$n" ] || ln -s "$d" "$PL/$n"; done
[ -e "$PL/small_cb" ] || ln -s "$W/small_cb" "$PL/small_cb"
[ -e "$PL/small_bt" ] || ln -s "$W/small_bt" "$PL/small_bt"
[ -f "$PL/small_cb/CMP_R0M5HbID.txt" ] || stop "NO-CMP-FILE-Y1"
[ -f "$PL/small_cb/REUSED_CAND_TAKEN.txt" ] || stop "NO-CAND-HASH-FILE-ADD68-RUNS"
[ -f "$PL/small_bt/REUSED_CAND_TAKEN.txt" ] || stop "NO-CAND-HASH-FILE-ADD67-RUNS"
[ -f "$P/root/integ10k/read_cyclebound10k.py" ] || stop "NO-READER-FILE"
[ -f "$P/root/integ10k/CYCLEBOUND10K_REUSED.json" ] || stop "NO-REUSED-RECORD"
[ -f "$W/cb_test/CHECK_HEAD_BOUND.txt" ] || stop "NO-TOOL-CHECK-OUTPUT-HEAD"
[ -f "$W/cb_test/CHECK_RULE_EVERY_RZGFH.txt" ] || stop "NO-TOOL-CHECK-OUTPUT-RULE"
[ -f "$W/cb_test/CHECK_ACC_GAIN_RZGFH.txt" ] || stop "NO-TOOL-CHECK-OUTPUT-GAIN"

# ---- control e1: the banked read of card ADD46, same command (pairs and order of read_i10k_n2.sh)
PA="I10RR:I10R0,I10RR:I10RLb,I10RLb:I10R0,I10RR0:I10R0,I10RRid:I10R0,I10RRx:I10R0"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/e1cb/hw" > "$OUT/e1cb/hw.log" 2>&1 ); r1=$?; echo "[$(ts)] e1 headway rc $r1"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/e1cb/q3" > "$OUT/e1cb/q3.log" 2>&1 ); r2=$?; echo "[$(ts)] e1 families rc $r2"
[ "$r1" = 0 ] && [ "$r2" = 0 ] || { tail -n 6 "$OUT/e1cb/hw.log" "$OUT/e1cb/q3.log" | cut -c1-300; stop "E1-READER-RC $r1 $r2"; }
"$PY" "$P/podread_envcheck.py" --bank "$P/bank" --got "$OUT/e1cb" --out "$OUT/e1cb/ENV_E1.json" || stop "E1-FAIL"

# ---- the read of card ADD69 (pairs of the card: the six named pairs, the six beside)
PAIRS="I10R0M5Hb20:I10R0M5,I10RLcM5Hb20:I10RLcM5,I10RQ50M5Hb20:I10RQ50M5,I10R0M5:I10R0,I10RLcM5:I10RLc,I10RQ50M5:I10RQ50"
ADD67JSON=$P/out/read_headbound/READ_HEADBOUND10K.json
ADD68JSON=$P/out/read_boundtier/READ_BOUNDTIER10K.json
if [ -f "$ADD67JSON" ] && grep -q '"controls_all_pass": true' "$ADD67JSON"; then
  PAIRS="$PAIRS,I10R0M5Hb20:I10R0Hb20,I10R0KM5Hb20:I10R0KHb20,I10RLcKM5Hb20:I10RLcKF5Hb20,I10RQ50KM5Hb20:I10RQ50KF5Hb20"; echo "[$(ts)] card ADD67 admitted: its controls_all_pass is true; its four pairs are read"
else
  echo "[$(ts)] card ADD67 NOT admitted (READ_HEADBOUND10K.json absent or controls_all_pass not true): CYC-PINNED, CYC-K, CYC-LABEL-CAND, CYC-READER-CAND are not read"
fi
if [ -f "$ADD68JSON" ] && grep -q '"controls_all_pass": true' "$ADD68JSON"; then
  PAIRS="$PAIRS,I10RLcM5Hb20:I10RLcHb20,I10RQ50M5Hb20:I10RQ50Hb20"; echo "[$(ts)] card ADD68 admitted: its controls_all_pass is true; its two pairs are read"
else
  echo "[$(ts)] card ADD68 NOT admitted (READ_BOUNDTIER10K.json absent or controls_all_pass not true): CYC-LABEL-RULE and CYC-READER-RULE are not read"
fi
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/read_cyclebound/hw" > "$OUT/read_cyclebound/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/read_cyclebound/q3" > "$OUT/read_cyclebound/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
"$PY" "$P/root/integ10k/read_cyclebound10k.py" --read-dir "$OUT/read_cyclebound" --pull "$PL" --reused "$P/root/integ10k/CYCLEBOUND10K_REUSED.json" --wrapper "$W/build/v8f_t1_eval_reader_rule_rzgfh.py" --test-wrapper "$W/cb_test/v8f_t1_eval_reader_rule_rzgfh.py" --test-out-head "$W/cb_test/CHECK_HEAD_BOUND.txt" --test-out-rule "$W/cb_test/CHECK_RULE_EVERY_RZGFH.txt" --test-out-gain "$W/cb_test/CHECK_ACC_GAIN_RZGFH.txt" --taken "$PL/small_cb/REUSED_CAND_TAKEN.txt" --taken67 "$PL/small_bt/REUSED_CAND_TAKEN.txt" --add67-json "$ADD67JSON" --add68-json "$ADD68JSON" --pairs-read "$PAIRS" --w "$W" --tree "$W/tree" --e1 "$OUT/e1cb/ENV_E1.json" --out "$OUT/read_cyclebound/READ_CYCLEBOUND10K.json" > "$OUT/read_cyclebound/read_cyclebound10k.log" 2>&1
echo "[$(ts)] read rc $?"
tail -n 230 "$OUT/read_cyclebound/read_cyclebound10k.log" | cut -c1-430
echo "ZZ PODREAD-CYCLEBOUND DONE $(ts)"
