#!/usr/bin/env bash
# podread_midlook10k.sh -- pod-side read of card PREREG_V8F_PROOF_ADD76 (MIDLOOK10K): the lead rule with a closing-speed input and a second look in the reader-fed form, at age and fresh
# (new runs I10RQ50hKF5P30Hb20, I10RQ50cKF5P30Hb20, I10RQ50hKF5Hb20, I10RQ50cKF5Hb20; banked I10RQ50KF5P30Hb20, I10RLcKF5P30Hb20, I10R0KP30Hb20, I10RQ50KF5Hb20, I10RLcKF5Hb20, I10R0KHb20, I10R0Hb20, I10R0).
# CPU only, no rollout. Needs MIDLOOK10K_DONE.txt. Never reads twice into one folder. No rm, no mv, no kill.
# Output folder: $P/out/read_midlook10k, or $P/out/$MIDLOOK_DIR when that variable is set. Banked read of card ADD71: AGERULE_JSON (default $P/out/read_agerule/READ_AGERULE10K.json).
# Last line: ZZ PODREAD-MIDLOOK10K DONE or ZZ PODREAD-MIDLOOK10K STOP <why>.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
LR=${LR_DIR:-$W/longroll}
FG=${FRESH_GRID:-/workspace/v8proof/hierrep/pins/eval_grid_windows.json}
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-MIDLOOK10K STOP $*"; exit 3; }
[ -f "$W/MIDLOOK10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
[ -e "$W/MIDLOOK10K_STOP.txt" ] && stop "QUEUE-STOPPED"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
OUT=$P/out; DC=$OUT/${MIDLOOK_DIR:-read_midlook10k}; AGERULEJSON=${AGERULE_JSON:-$OUT/read_agerule/READ_AGERULE10K.json}
[ -e "$DC/READ_MIDLOOK10K.json" ] && stop "ALREADY-READ $DC/READ_MIDLOOK10K.json exists"
PL=$P/root/integ10k/pull; mkdir -p "$PL"
for d in "$W"/t1/*_s[01]; do n=$(basename "$d"); [ -e "$PL/$n" ] || ln -s "$d" "$PL/$n"; done
for f in "$P/root/integ10k/read_longroll10k.py" "$P/root/integ10k/read_longroll10k_n3.py" "$P/root/integ10k/read_longroll10k_n4.py" "$P/root/integ10k/read_agerule_decomp.py" "$P/root/integ10k/read_agedose_decomp.py" \
         "$P/root/integ10k/read_handsrc_decomp.py" "$P/root/integ10k/read_closrc10k.py" "$P/root/integ10k/read_midlook10k.py" "$P/root/integ10k/strat_roll_read.py" "$LR/grid_p30.json" "$FG" "$AGERULEJSON"; do
  [ -f "$f" ] || stop "MISSING-INPUT $f"
done
grep -q '"controls_all_pass": true' "$AGERULEJSON" || stop "ADD71-READ-NOT-ADMITTED"
for r in R0KHb20 R0KP30Hb20 R0Hb20 R0 RQ50KF5P30Hb20 RLcKF5P30Hb20 RQ50KF5Hb20 RLcKF5Hb20 RQ50hKF5P30Hb20 RQ50cKF5P30Hb20 RQ50hKF5Hb20 RQ50cKF5Hb20; do for s in 0 1; do [ -d "$PL/I10${r}_s$s" ] || stop "MISSING-RUN I10${r}_s$s"; done; done
mkdir -p "$DC"
sha256sum "$P/root/integ10k/read_midlook10k.py" "$P/root/integ10k/read_closrc10k.py" "$P/root/integ10k/read_handsrc_decomp.py" "$P/root/integ10k/read_agedose_decomp.py" "$P/root/integ10k/read_agerule_decomp.py" \
  "$P/root/integ10k/read_longroll10k_n4.py" "$P/root/integ10k/read_longroll10k_n3.py" "$P/root/integ10k/read_longroll10k.py" "$P/root/integ10k/strat_roll_read.py" "$AGERULEJSON" "$FG" "$LR/grid_p30.json" > "$DC/READER_HASHES.txt"
cut -c1-16,65- "$DC/READER_HASHES.txt"
ls -l "$P/root/integ10k/read_midlook10k.py" "$P/root/integ10k/read_closrc10k.py" "$P/root/integ10k/read_handsrc_decomp.py" "$P/root/integ10k/read_agedose_decomp.py" "$P/root/integ10k/read_agerule_decomp.py" \
  "$P/root/integ10k/read_longroll10k_n4.py" "$P/root/integ10k/read_longroll10k_n3.py" "$P/root/integ10k/read_longroll10k.py" "$P/root/integ10k/strat_roll_read.py" >> "$DC/READER_HASHES.txt"
( cd "$P/root/integ10k" && "$PY" read_midlook10k.py --selftest > "$DC/selftest_pod.log" 2>&1 ); st=$?
tail -n 1 "$DC/selftest_pod.log" | cut -c1-120
[ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
( cd "$P/root/integ10k" && "$PY" read_midlook10k.py --pull "$PL" --code "$P/code" --grid-dir "$LR" --fresh-grid "$FG" --agerule-json "$AGERULEJSON" --w "$W" --tree "$W/tree" --out "$DC/READ_MIDLOOK10K.json" > "$DC/read_midlook10k.log" 2>&1 )
echo "[$(ts)] read rc $?"
grep -E '^ZZ READ-MIDLOOK|a control did not pass| FAIL |Traceback|Error' "$DC/read_midlook10k.log" | cut -c1-500 | head -n 14
echo "ZZ PODREAD-MIDLOOK10K DONE $(ts)"
