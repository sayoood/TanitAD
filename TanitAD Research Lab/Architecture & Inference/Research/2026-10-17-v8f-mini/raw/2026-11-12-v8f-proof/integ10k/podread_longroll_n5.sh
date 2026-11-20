#!/usr/bin/env bash
# podread_longroll_n5.sh -- re-read of card PREREG_V8F_PROOF_ADD70 by its notes N4 and N5 (own start point of a pre-rolled window; control n2 in the form of note N5); pod, CPU only.
# The reads under notes N2 and N3 stay as they are. This script reads into the new folder out/read_longroll_n5/ with read_longroll10k_n4.py (a wrapper around the reader of note N3).
# Control e1 is the file the first read wrote (out/e1lr/ENV_E1.json, PASS). Never reads twice into one folder. No rm, no mv, no kill. Starts no rollout and no training.
# Last line: ZZ PODREAD-LONGROLL-N5 DONE or ZZ PODREAD-LONGROLL-N5 STOP <why>.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
LR=${LR_DIR:-$W/longroll}; TST=${LR_TEST_DIR:-$W/lr_test}; BLD=${LR_BUILD_DIR:-$W/build}
GRIDFULL=${LR_GRID_FULL:-/workspace/v8proof/hierrep/pins/eval_grid_windows.json}
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-LONGROLL-N5 STOP $*"; exit 3; }
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
OUT=$P/out; N5=$OUT/read_longroll_n5; N3JSON=$OUT/read_longroll_n3/READ_LONGROLL10K_N3.json
[ -e "$N5/READ_LONGROLL10K_N5.json" ] && stop "ALREADY-READ out/read_longroll_n5/READ_LONGROLL10K_N5.json exists"
grep -q '"PASS": true' "$OUT/e1lr/ENV_E1.json" || stop "E1-OF-THE-FIRST-READ-NOT-PASS"
mkdir -p "$N5"
PL=$P/root/integ10k/pull
ADD67JSON=$P/out/read_headbound/READ_HEADBOUND10K.json
HF=$OUT/read_longroll/POD_HASHES.txt
for f in "$P/root/integ10k/read_longroll10k.py" "$P/root/integ10k/read_longroll10k_n3.py" "$P/root/integ10k/read_longroll10k_n4.py" "$P/root/integ10k/LONGROLL10K_REUSED.json" "$P/root/integ10k/strat_roll_read.py" "$HF" "$ADD67JSON" "$N3JSON"; do
  [ -f "$f" ] || stop "MISSING-INPUT $f"
done
sha256sum "$P/root/integ10k/read_longroll10k.py" "$P/root/integ10k/read_longroll10k_n3.py" "$P/root/integ10k/read_longroll10k_n4.py" "$P/code/headway/headway_read_r2.py" "$P/code/stageB/hierrep_read.py" > "$N5/READER_HASHES.txt"
cut -c1-16,65- "$N5/READER_HASHES.txt"
( cd "$P/root/integ10k" && "$PY" read_longroll10k_n4.py --n3-json "$N3JSON" --read-dir "$N5" --pull "$PL" --reused "$P/root/integ10k/LONGROLL10K_REUSED.json" --code "$P/code" --grid-dir "$LR" --grid-full "$GRIDFULL" \
  --wrapper "$BLD/v8f_t1_eval_reader_rule_rzgfhq.py" --parent-wrapper "$BLD/v8f_t1_eval_reader_rule_rzgfh.py" --test-wrapper "$TST/v8f_t1_eval_reader_rule_rzgfhq.py" --check-preroll "$TST/CHECK_PREROLL_B.txt" \
  --e1 "$OUT/e1lr/ENV_E1.json" --add67-json "$ADD67JSON" --add67-hw "$P/out/read_headbound/hw/RESULT_HEADWAY.json" --add67-q3 "$P/out/read_headbound/q3/RESULT_Q3.json" --pod-hashes "$HF" \
  --w "$W" --tree "$W/tree" --out "$N5/READ_LONGROLL10K_N5.json" > "$N5/read_longroll10k_n5.log" 2>&1 )
echo "[$(ts)] read rc $?"
grep -E '^ZZ READ-LONGROLL-N4|^== a control did not pass| FAIL |NOT COMPUTABLE|Traceback|Error' "$N5/read_longroll10k_n5.log" | cut -c1-600 | head -n 14
echo "ZZ PODREAD-LONGROLL-N5 DONE $(ts)"
