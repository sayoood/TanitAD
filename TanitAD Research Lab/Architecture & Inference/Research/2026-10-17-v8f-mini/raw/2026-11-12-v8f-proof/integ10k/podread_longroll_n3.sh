#!/usr/bin/env bash
# podread_longroll_n3.sh -- the read of record of card PREREG_V8F_PROOF_ADD70 by its note N3 (pod, CPU only).
# The first read (podread_longroll.sh) stays as it is in out/read_longroll/. This script reads into the new folder out/read_longroll_n3/ with the wrapper read_longroll10k_n3.py around the
# unedited reader; control e1 is the one the first read wrote (out/e1lr/ENV_E1.json, PASS), the inputs and the command line are those of the first read.
# Never reads twice into one folder. No rm, no mv, no kill. Starts no rollout and no training.
# Last line: ZZ PODREAD-LONGROLL-N3 DONE or ZZ PODREAD-LONGROLL-N3 STOP <why>.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
LR=${LR_DIR:-$W/longroll}; TST=${LR_TEST_DIR:-$W/lr_test}; BLD=${LR_BUILD_DIR:-$W/build}
GRIDFULL=${LR_GRID_FULL:-/workspace/v8proof/hierrep/pins/eval_grid_windows.json}
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-LONGROLL-N3 STOP $*"; exit 3; }
[ -f "$W/LONGROLL10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
OUT=$P/out; N3=$OUT/read_longroll_n3
[ -e "$N3/READ_LONGROLL10K_N3.json" ] && stop "ALREADY-READ out/read_longroll_n3/READ_LONGROLL10K_N3.json exists"
[ -f "$OUT/read_longroll/READ_LONGROLL10K.json" ] || stop "FIRST-READ-ABSENT"
grep -q '"PASS": true' "$OUT/e1lr/ENV_E1.json" || stop "E1-OF-THE-FIRST-READ-NOT-PASS"
mkdir -p "$N3"
PL=$P/root/integ10k/pull
ADD67JSON=$P/out/read_headbound/READ_HEADBOUND10K.json
HF=$OUT/read_longroll/POD_HASHES.txt
for f in "$P/root/integ10k/read_longroll10k.py" "$P/root/integ10k/read_longroll10k_n3.py" "$P/root/integ10k/LONGROLL10K_REUSED.json" "$P/root/integ10k/strat_roll_read.py" "$HF" "$ADD67JSON"; do
  [ -f "$f" ] || stop "MISSING-INPUT $f"
done
sha256sum "$P/root/integ10k/read_longroll10k.py" "$P/root/integ10k/read_longroll10k_n3.py" > "$N3/READER_HASHES.txt"
cat "$N3/READER_HASHES.txt" | cut -c1-16,65-
( cd "$P/root/integ10k" && "$PY" read_longroll10k_n3.py --read-dir "$N3" --pull "$PL" --reused "$P/root/integ10k/LONGROLL10K_REUSED.json" --code "$P/code" --grid-dir "$LR" --grid-full "$GRIDFULL" \
  --wrapper "$BLD/v8f_t1_eval_reader_rule_rzgfhq.py" --parent-wrapper "$BLD/v8f_t1_eval_reader_rule_rzgfh.py" --test-wrapper "$TST/v8f_t1_eval_reader_rule_rzgfhq.py" --check-preroll "$TST/CHECK_PREROLL_B.txt" \
  --e1 "$OUT/e1lr/ENV_E1.json" --add67-json "$ADD67JSON" --add67-hw "$P/out/read_headbound/hw/RESULT_HEADWAY.json" --add67-q3 "$P/out/read_headbound/q3/RESULT_Q3.json" --pod-hashes "$HF" \
  --w "$W" --tree "$W/tree" --out "$N3/READ_LONGROLL10K_N3.json" > "$N3/read_longroll10k_n3.log" 2>&1 )
echo "[$(ts)] read rc $?"
grep -c . "$N3/read_longroll10k_n3.log"
grep -E '^ZZ READ-LONGROLL-N3|^== a control did not pass|FAIL|NOT COMPUTABLE' "$N3/read_longroll10k_n3.log" | cut -c1-300 | head -n 12
echo "ZZ PODREAD-LONGROLL-N3 DONE $(ts)"
