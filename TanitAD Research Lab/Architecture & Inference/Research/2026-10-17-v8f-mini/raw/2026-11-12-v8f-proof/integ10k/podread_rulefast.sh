#!/usr/bin/env bash
# podread_rulefast.sh -- pod-side read of card PREREG_V8F_PROOF_ADD62 (RULEFAST10K). Written by make_rulefast10k.py from podread_causlim.sh (card ADD61) with the names, pairs and files of this card.
# Same readers and rules as the pod reads of cards ADD47 to ADD58.
# Order: control e1 first (the pod reproduces the banked dev-box read of card ADD46 leaf for leaf, same reader command); only if e1 passes, the read of card ADD62.
# CPU only (CUDA_VISIBLE_DEVICES=-1). Starts no rollout and no training. Last line: ZZ PODREAD-RULEFAST DONE or ZZ PODREAD-RULEFAST STOP <why>.
# Files it needs on the pod (pushed by the lead before it runs): $P/root/integ10k/read_rulefast10k.py and $P/root/integ10k/RULEFAST10K_REUSED.json (sha256 of the result files and rule logs of the
# reused runs, recorded before the queue started); the tool-check output and the wrapper copy are written by the queue into $W/rulefast_test/, the compare file into $W/small_rulefast/.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-RULEFAST STOP $*"; exit 3; }
[ -f "$W/RULEFAST10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
ROOTS=$W/t1
OUT=$P/out; mkdir -p "$OUT/e1rf" "$OUT/read_rulefast"
PL=$P/root/integ10k/pull; mkdir -p "$PL"
for d in "$W"/t1/*_s[01]; do n=$(basename "$d"); [ -e "$PL/$n" ] || ln -s "$d" "$PL/$n"; done
[ -e "$PL/small_rulefast" ] || ln -s "$W/small_rulefast" "$PL/small_rulefast"
[ -f "$PL/small_rulefast/CMP_RLcFID.txt" ] || stop "NO-CMP-FILE-K1"
[ -f "$P/root/integ10k/read_rulefast10k.py" ] || stop "NO-READER-FILE"
[ -f "$P/root/integ10k/RULEFAST10K_REUSED.json" ] || stop "NO-REUSED-RECORD"
[ -f "$W/rulefast_test/CHECK_RULE_EVERY.txt" ] || stop "NO-TOOL-CHECK-OUTPUT"

# ---- control e1: the banked read of card ADD46, same command (pairs and order of read_i10k_n2.sh)
PA="I10RR:I10R0,I10RR:I10RLb,I10RLb:I10R0,I10RR0:I10R0,I10RRid:I10R0,I10RRx:I10R0"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/e1rf/hw" > "$OUT/e1rf/hw.log" 2>&1 ); r1=$?; echo "[$(ts)] e1 headway rc $r1"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PA" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/e1rf/q3" > "$OUT/e1rf/q3.log" 2>&1 ); r2=$?; echo "[$(ts)] e1 families rc $r2"
[ "$r1" = 0 ] && [ "$r2" = 0 ] || { tail -n 6 "$OUT/e1rf/hw.log" "$OUT/e1rf/q3.log" | cut -c1-300; stop "E1-READER-RC $r1 $r2"; }
"$PY" "$P/podread_envcheck.py" --bank "$P/bank" --got "$OUT/e1rf" --out "$OUT/e1rf/ENV_E1.json" || stop "E1-FAIL"

# ---- the read of card ADD62 (pairs of the card: FR per form, the dose pairs, the pairs beside)
PAIRS="I10RLcF5:I10RLc,I10RQ50F5:I10RQ50,I10RLcF2:I10RLc,I10RLcF2:I10RLcF5,I10RLcF5:I10R0,I10RQ50F5:I10R0,I10RLcF2:I10R0,I10RQ50F5:I10RLcF5"
( cd "$P/code/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$OUT/read_rulefast/hw" > "$OUT/read_rulefast/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
( cd "$P/code/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$OUT/read_rulefast/q3" > "$OUT/read_rulefast/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
"$PY" "$P/root/integ10k/read_rulefast10k.py" --read-dir "$OUT/read_rulefast" --pull "$PL" --reused "$P/root/integ10k/RULEFAST10K_REUSED.json" --wrapper "$W/build/v8f_t1_eval_reader_rule_rzf.py" --test-wrapper "$W/rulefast_test/v8f_t1_eval_reader_rule_rzf.py" --test-out "$W/rulefast_test/CHECK_RULE_EVERY.txt" --e1 "$OUT/e1rf/ENV_E1.json" --out "$OUT/read_rulefast/READ_RULEFAST10K.json" > "$OUT/read_rulefast/read_rulefast10k.log" 2>&1
echo "[$(ts)] read rc $?"
tail -n 230 "$OUT/read_rulefast/read_rulefast10k.log" | cut -c1-430
echo "ZZ PODREAD-RULEFAST DONE $(ts)"
