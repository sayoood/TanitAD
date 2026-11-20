#!/usr/bin/env bash
# podread_closrc10k.sh -- pod-side read of card PREREG_V8F_PROOF_ADD75 (CLOSRC10K): which input of the lead rule carries the contact of the reader-fed form at 6 s of own driving
# (runs I10RQ50KP30Hb20, I10RQ50cKP30Hb20, I10RQ50hKP30Hb20, I10RLbKP30Hb20, I10RLcKP30Hb20 of the card; banked I10R0KP30Hb20, I10R0KHb20, I10R0Hb20, I10R0). CPU only, no rollout.
# Needs CLOSRC10K_DONE.txt and the re-read of card ADD70 under its note N4 with all controls passed. Never reads twice into one folder. No rm, no mv, no kill.
# Last line: ZZ PODREAD-CLOSRC10K DONE or ZZ PODREAD-CLOSRC10K STOP <why>.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
LR=${LR_DIR:-$W/longroll}
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-CLOSRC10K STOP $*"; exit 3; }
[ -f "$W/CLOSRC10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
[ -e "$W/CLOSRC10K_STOP.txt" ] && stop "QUEUE-STOPPED"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
OUT=$P/out; DC=$OUT/${CLOSRC_DIR:-read_closrc10k}; ADD70JSON=$OUT/read_longroll_n5/READ_LONGROLL10K_N5.json
[ -e "$DC/READ_CLOSRC10K.json" ] && stop "ALREADY-READ $DC/READ_CLOSRC10K.json exists"
PL=$P/root/integ10k/pull; mkdir -p "$PL"
for d in "$W"/t1/*_s[01]; do n=$(basename "$d"); [ -e "$PL/$n" ] || ln -s "$d" "$PL/$n"; done
for f in "$P/root/integ10k/read_longroll10k.py" "$P/root/integ10k/read_longroll10k_n3.py" "$P/root/integ10k/read_longroll10k_n4.py" "$P/root/integ10k/read_agerule_decomp.py" "$P/root/integ10k/read_agedose_decomp.py" \
         "$P/root/integ10k/read_handsrc_decomp.py" "$P/root/integ10k/read_closrc10k.py" "$P/root/integ10k/strat_roll_read.py" "$LR/grid_p30.json" "$ADD70JSON"; do
  [ -f "$f" ] || stop "MISSING-INPUT $f"
done
grep -q '"controls_all_pass": true' "$ADD70JSON" || stop "ADD70-REREAD-NOT-ADMITTED"
for r in R0KHb20 R0KP30Hb20 R0Hb20 R0 RQ50KP30Hb20 RQ50cKP30Hb20 RQ50hKP30Hb20 RLbKP30Hb20 RLcKP30Hb20; do for s in 0 1; do [ -d "$PL/I10${r}_s$s" ] || stop "MISSING-RUN I10${r}_s$s"; done; done
mkdir -p "$DC"
sha256sum "$P/root/integ10k/read_closrc10k.py" "$P/root/integ10k/read_handsrc_decomp.py" "$P/root/integ10k/read_agedose_decomp.py" "$P/root/integ10k/read_agerule_decomp.py" "$P/root/integ10k/read_longroll10k_n4.py" \
  "$P/root/integ10k/read_longroll10k.py" "$P/root/integ10k/strat_roll_read.py" "$ADD70JSON" > "$DC/READER_HASHES.txt"
cut -c1-16,65- "$DC/READER_HASHES.txt"
( cd "$P/root/integ10k" && "$PY" read_closrc10k.py --selftest > "$DC/selftest_pod.log" 2>&1 ); st=$?
tail -n 1 "$DC/selftest_pod.log" | cut -c1-120
[ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
( cd "$P/root/integ10k" && "$PY" read_closrc10k.py --pull "$PL" --code "$P/code" --grid-dir "$LR" --add70-json "$ADD70JSON" --w "$W" --tree "$W/tree" --out "$DC/READ_CLOSRC10K.json" > "$DC/read_closrc10k.log" 2>&1 )
echo "[$(ts)] read rc $?"
grep -E '^ZZ READ-CLOSRC|a control did not pass| FAIL |Traceback|Error' "$DC/read_closrc10k.log" | cut -c1-500 | head -n 14
echo "ZZ PODREAD-CLOSRC10K DONE $(ts)"
