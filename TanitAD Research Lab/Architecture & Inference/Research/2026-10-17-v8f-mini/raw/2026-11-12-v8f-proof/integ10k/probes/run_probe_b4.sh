#!/usr/bin/env bash
# run_probe_b4.sh -- runs probe_b4.py on the pod (CPU only, controls only). Writes into podread/probe/decomp_b4/ (new folder). No rm, no mv, no kill.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
export HEADWAY_Q3DIR=$P/code/q3 HIERREP_STAGEB=$P/code/stageB TANITEVAL_ROOT=$P/code/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || { echo "ZZ PROBE-B4 STOP MEMORY"; exit 3; }
D=$P/probe/decomp_b4
[ -e "$D/OUT.json" ] && { echo "ZZ PROBE-B4 STOP ALREADY-RUN"; exit 3; }
mkdir -p "$D"
( cd "$P/root/integ10k" && "$PY" "$P/probe/probe_b4.py" --pull "$P/root/integ10k/pull" --code "$P/code" --grid-dir "$W/longroll" --add70-json "$P/out/read_longroll_n5/READ_LONGROLL10K_N5.json" --out "$D/OUT.json" --n-boot 20 > "$D/probe_b4.log" 2>&1 )
echo "rc $?"
grep -E '^NAN|^B4|Traceback|Error' "$D/probe_b4.log" | cut -c1-200 | head -n 60
echo "ZZ PROBE-B4 DONE"
