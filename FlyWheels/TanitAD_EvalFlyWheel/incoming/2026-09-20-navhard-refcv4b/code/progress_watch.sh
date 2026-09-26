#!/usr/bin/env bash
# Samples, every 5 min, how far each scored arm has got (seam-agent calls, one JSONL line per scored
# token) and the box's available RAM — so that if the RAM guard fires a third time, WHERE and AT WHAT
# MEMORY is a measurement, not a reconstruction. Ends when the run writes bench_run.json.
set -u
RUN="$1"; OUT="$2"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
while [ ! -f "$RUN/bench_run.json" ]; do
  g=$("$PY" -c "import psutil;print('%.2f'%(psutil.virtual_memory().available/2**30))" 2>/dev/null)
  line="$(date '+%F %T') avail_gb=$g"
  for a in ECHO A1; do
    f="$RUN/raw/$a/$a.calls.jsonl"
    n=$([ -f "$f" ] && wc -l < "$f" || echo 0)
    line="$line $a=$n/5912"
  done
  echo "$line" >> "$OUT"
  sleep 300
done
echo "$(date '+%F %T') run wrote bench_run.json — watch ends" >> "$OUT"
