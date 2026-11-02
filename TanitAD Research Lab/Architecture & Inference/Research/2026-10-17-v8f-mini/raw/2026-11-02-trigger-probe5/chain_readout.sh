#!/bin/bash
# chain_readout.sh -- waits for raw/CHAIN_P5.done (features pulled + sha256 verified), then runs the readout stages in order (CPU, 4 threads).  Log raw/logs/readout.log; markers raw/READOUT.done / raw/READOUT.failed.
set -u
P=/e/Projects/TanitAD-artifacts/dagland/trigger_probe5
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=$P/raw/logs/readout.log
say() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" >> "$LOG"; }
rm -f "$P/raw/READOUT.done" "$P/raw/READOUT.failed"
say "readout chain start; waiting for CHAIN_P5.done"
for i in $(seq 1 900); do
  [ -f "$P/raw/CHAIN_P5.done" ] && break
  [ -f "$P/raw/CHAIN_P5.failed" ] && { say "upstream failed: $(cat $P/raw/CHAIN_P5.failed)"; echo upstream > "$P/raw/READOUT.failed"; exit 1; }
  sleep 60
done
[ -f "$P/raw/CHAIN_P5.done" ] || { say "upstream never finished"; echo timeout > "$P/raw/READOUT.failed"; exit 1; }
cd "$P" || exit 1
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for st in assemble pca lin strata eval decide; do
  say "stage $st"
  $PY -I run_p5.py $st >> "$P/raw/logs/readout_$st.log" 2>&1 || { say "stage $st FAILED (see raw/logs/readout_$st.log)"; echo "$st" > "$P/raw/READOUT.failed"; exit 1; }
done
say "readout done"
echo ok > "$P/raw/READOUT.done"
