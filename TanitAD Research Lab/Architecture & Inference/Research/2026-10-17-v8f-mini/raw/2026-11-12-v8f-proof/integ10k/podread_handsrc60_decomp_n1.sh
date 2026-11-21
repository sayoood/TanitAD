#!/usr/bin/env bash
# podread_handsrc60_decomp_n1.sh -- the re-read of note N1 of card ADD74 (new folder out/read_handsrc60_decomp_n1; no headway name where the instrument is not qualified; control p1). pod-side read of card PREREG_V8F_PROOF_ADD74 (HANDSRC60): the read of card ADD73 (notes N1, N2) at 12 s of own driving, runs I10R0KP60LgHb20,
# I10R0KP60SlHb20, I10R0KP60PlHb20 of the card; banked I10R0KP60Hb20, I10R0KHb20, I10R0Hb20, I10R0. CPU only, no rollout. Needs HANDSRC60_DONE.txt, the re-read of card ADD70 under its
# note N4 and the read of card ADD72 Amendment 1, both with all controls passed. Never reads twice into one folder. No rm, no mv, no kill.
# Last line: ZZ PODREAD-HANDSRC60-DECOMP DONE or ZZ PODREAD-HANDSRC60-DECOMP STOP <why>.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
LR=${LR_DIR:-$W/longroll}
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-HANDSRC60-DECOMP STOP $*"; exit 3; }
[ -f "$W/HANDSRC60_DONE.txt" ] || stop "QUEUE-NOT-DONE"
[ -e "$W/HANDSRC60_STOP.txt" ] && stop "QUEUE-STOPPED"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
OUT=$P/out; DC=$OUT/read_handsrc60_decomp_n1; ADD70JSON=$OUT/read_longroll_n5/READ_LONGROLL10K_N5.json; ADD72JSON=$OUT/read_agedose_decomp/READ_AGEDOSE_DECOMP.json
[ -e "$DC/READ_HANDSRC60_DECOMP.json" ] && stop "ALREADY-READ out/read_handsrc60_decomp_n1/READ_HANDSRC60_DECOMP.json exists"
PL=$P/root/integ10k/pull; mkdir -p "$PL"
for d in "$W"/t1/*_s[01]; do n=$(basename "$d"); [ -e "$PL/$n" ] || ln -s "$d" "$PL/$n"; done
for f in "$P/root/integ10k/read_longroll10k.py" "$P/root/integ10k/read_longroll10k_n3.py" "$P/root/integ10k/read_longroll10k_n4.py" "$P/root/integ10k/read_agerule_decomp.py" "$P/root/integ10k/read_agedose_decomp.py" \
         "$P/root/integ10k/read_handsrc_decomp.py" "$P/root/integ10k/strat_roll_read.py" "$LR/grid_p60.json" "$ADD70JSON" "$ADD72JSON" "$W/handsrc_test/CHECK_DONE_chk1n1.txt"; do
  [ -f "$f" ] || stop "MISSING-INPUT $f"
done
grep -q '"controls_all_pass": true' "$ADD70JSON" || stop "ADD70-REREAD-NOT-ADMITTED"
grep -q '"controls_all_pass": true' "$ADD72JSON" || stop "ADD72-READ-NOT-ADMITTED"
for r in R0KHb20 R0KP60Hb20 R0Hb20 R0 R0KP60LgHb20 R0KP60SlHb20 R0KP60PlHb20; do for s in 0 1; do [ -d "$PL/I10${r}_s$s" ] || stop "MISSING-RUN I10${r}_s$s"; done; done
mkdir -p "$DC"
sha256sum "$P/root/integ10k/read_handsrc_decomp.py" "$P/root/integ10k/read_agedose_decomp.py" "$P/root/integ10k/read_agerule_decomp.py" "$P/root/integ10k/read_longroll10k_n4.py" "$P/root/integ10k/read_longroll10k.py" \
  "$P/root/integ10k/strat_roll_read.py" "$ADD70JSON" "$ADD72JSON" > "$DC/READER_HASHES.txt"
cut -c1-16,65- "$DC/READER_HASHES.txt"
( cd "$P/root/integ10k" && "$PY" read_handsrc_decomp.py --p 60 --pull "$PL" --code "$P/code" --grid-dir "$LR" --add70-json "$ADD70JSON" --agedose-json "$ADD72JSON" --w "$W" --tree "$W/tree" \
  --first-part1-json "$OUT/read_handsrc60_decomp/READ_HANDSRC60_DECOMP.json" --out "$DC/READ_HANDSRC60_DECOMP.json" > "$DC/read_handsrc60_decomp.log" 2>&1 )
echo "[$(ts)] read rc $?"
grep -E '^ZZ READ-HANDSRC|a control of part . did not pass| FAIL |Traceback|Error' "$DC/read_handsrc60_decomp.log" | cut -c1-400 | head -n 14
echo "ZZ PODREAD-HANDSRC60-DECOMP DONE $(ts)"
