#!/usr/bin/env bash
# podread_agerule_decomp.sh -- pod-side read of part B of card PREREG_V8F_PROOF_ADD71 with its notes N1 and N2 (carried pose against own motion state on the banked runs I10R0KNlHb20 and
# I10R0KNlP30Hb20 of card ADD70); CPU only, no rollout. Needs the re-read of card ADD70 under its note N4 with all controls passed (control b4 compares with it).
# Never reads twice into one folder. No rm, no mv, no kill. Last line: ZZ PODREAD-AGERULE-DECOMP DONE or ZZ PODREAD-AGERULE-DECOMP STOP <why>.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
LR=${LR_DIR:-$W/longroll}
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-AGERULE-DECOMP STOP $*"; exit 3; }
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
OUT=$P/out; DC=$OUT/read_agerule_decomp_n2; ADD70JSON=$OUT/read_longroll_n5/READ_LONGROLL10K_N5.json
[ -e "$DC/READ_AGERULE_DECOMP_N2.json" ] && stop "ALREADY-READ out/read_agerule_decomp_n2/READ_AGERULE_DECOMP_N2.json exists"
PL=$P/root/integ10k/pull
for f in "$P/root/integ10k/read_longroll10k.py" "$P/root/integ10k/read_longroll10k_n3.py" "$P/root/integ10k/read_longroll10k_n4.py" "$P/root/integ10k/read_agerule_decomp.py" "$LR/grid_p30.json" "$ADD70JSON"; do
  [ -f "$f" ] || stop "MISSING-INPUT $f"
done
grep -q '"controls_all_pass": true' "$ADD70JSON" || stop "ADD70-REREAD-NOT-ADMITTED"
for r in R0KNlHb20 R0KNlP30Hb20 R0Hb20; do for s in 0 1; do [ -d "$PL/I10${r}_s$s" ] || stop "MISSING-RUN I10${r}_s$s"; done; done
mkdir -p "$DC"
sha256sum "$P/root/integ10k/read_agerule_decomp.py" "$P/root/integ10k/read_longroll10k_n4.py" "$ADD70JSON" > "$DC/READER_HASHES.txt"
cut -c1-16,65- "$DC/READER_HASHES.txt"
( cd "$P/root/integ10k" && "$PY" read_agerule_decomp.py --pull "$PL" --code "$P/code" --grid-dir "$LR" --add70-json "$ADD70JSON" --out "$DC/READ_AGERULE_DECOMP_N2.json" > "$DC/read_agerule_decomp.log" 2>&1 )
echo "[$(ts)] read rc $?"
grep -E 'controls_all_pass|a control did not pass| FAIL|Traceback|Error' "$DC/read_agerule_decomp.log" | cut -c1-400 | head -n 12
echo "ZZ PODREAD-AGERULE-DECOMP DONE $(ts)"
