#!/usr/bin/env bash
# podread_handsrc_decomp.sh -- pod-side read of card PREREG_V8F_PROOF_ADD73 (HANDSRC10K) under its notes N1 and N2: which input of the state hand-over at the first scored tick carries the
# state part of the cost of age (runs I10R0KP30LgHb20, I10R0KP30SlHb20, I10R0KP30PlHb20 of the card; banked I10R0KP30Hb20, I10R0KHb20, I10R0Hb20, I10R0). CPU only, no rollout.
# Needs HANDSRC10K_DONE.txt and the re-read of card ADD70 under its note N4 with all controls passed. Never reads twice into one folder. No rm, no mv, no kill.
# Last line: ZZ PODREAD-HANDSRC-DECOMP DONE or ZZ PODREAD-HANDSRC-DECOMP STOP <why>.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
LR=${LR_DIR:-$W/longroll}
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-HANDSRC-DECOMP STOP $*"; exit 3; }
[ -f "$W/HANDSRC10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
[ -e "$W/HANDSRC10K_STOP.txt" ] && stop "QUEUE-STOPPED"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
OUT=$P/out; DC=$OUT/read_handsrc_decomp; ADD70JSON=$OUT/read_longroll_n5/READ_LONGROLL10K_N5.json
[ -e "$DC/READ_HANDSRC_DECOMP.json" ] && stop "ALREADY-READ out/read_handsrc_decomp/READ_HANDSRC_DECOMP.json exists"
PL=$P/root/integ10k/pull; mkdir -p "$PL"
for d in "$W"/t1/*_s[01]; do n=$(basename "$d"); [ -e "$PL/$n" ] || ln -s "$d" "$PL/$n"; done
for f in "$P/root/integ10k/read_longroll10k.py" "$P/root/integ10k/read_longroll10k_n3.py" "$P/root/integ10k/read_longroll10k_n4.py" "$P/root/integ10k/read_agerule_decomp.py" "$P/root/integ10k/read_agedose_decomp.py" \
         "$P/root/integ10k/read_handsrc_decomp.py" "$P/root/integ10k/strat_roll_read.py" "$LR/grid_p30.json" "$ADD70JSON" "$W/handsrc_test/CHECK_DONE_chk1n1.txt"; do
  [ -f "$f" ] || stop "MISSING-INPUT $f"
done
grep -q '"controls_all_pass": true' "$ADD70JSON" || stop "ADD70-REREAD-NOT-ADMITTED"
grep -q '^ZZ CHECK-HANDOVER-SRC PASS' "$W/handsrc_test/CHECK_DONE_chk1n1.txt" || stop "TOOL-CHECK-HANDOVER-SRC-NOT-PASSED"
for r in R0KHb20 R0KP30Hb20 R0Hb20 R0 R0KP30LgHb20 R0KP30SlHb20 R0KP30PlHb20; do for s in 0 1; do [ -d "$PL/I10${r}_s$s" ] || stop "MISSING-RUN I10${r}_s$s"; done; done
mkdir -p "$DC"
sha256sum "$P/root/integ10k/read_handsrc_decomp.py" "$P/root/integ10k/read_agedose_decomp.py" "$P/root/integ10k/read_agerule_decomp.py" "$P/root/integ10k/read_longroll10k_n4.py" "$P/root/integ10k/read_longroll10k.py" \
  "$P/root/integ10k/strat_roll_read.py" "$ADD70JSON" > "$DC/READER_HASHES.txt"
cut -c1-16,65- "$DC/READER_HASHES.txt"
( cd "$P/root/integ10k" && "$PY" read_handsrc_decomp.py --selftest > "$DC/selftest_pod.log" 2>&1 ); st=$?
tail -n 1 "$DC/selftest_pod.log" | cut -c1-120
[ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
( cd "$P/root/integ10k" && "$PY" read_handsrc_decomp.py --pull "$PL" --code "$P/code" --grid-dir "$LR" --add70-json "$ADD70JSON" --w "$W" --tree "$W/tree" --out "$DC/READ_HANDSRC_DECOMP.json" > "$DC/read_handsrc_decomp.log" 2>&1 )
echo "[$(ts)] read rc $?"
grep -E '^ZZ READ-HANDSRC|a control of part . did not pass| FAIL |Traceback|Error' "$DC/read_handsrc_decomp.log" | cut -c1-400 | head -n 14
echo "ZZ PODREAD-HANDSRC-DECOMP DONE $(ts)"
