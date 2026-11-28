#!/usr/bin/env bash
# podread_gaprule_strat.sh -- pod-side run of read_gaprule_strat.py (card ADD95, the STRATEGIC family of the twelve pairs, an addition after the names; no name). CPU only, no rollout.
# Writes $GP/read3/GAPRULE10KB_STRATEGIC.json and read_strat.log. Never reads twice into one file. It deletes, moves and stops nothing.
# Last line: ZZ PODREAD-GAPRULE-STRAT DONE rc <n>   or   ZZ PODREAD-GAPRULE-STRAT STOP <why>.
set -u
V=/workspace/v8proof; W=$V/integ10k; P=$V/podread; RD=$P/root/integ10k; PL=$RD/pull; CODE=$P/code
GP=$V/gaprule10kb; K=$GP/kit; OUT=$GP/read3
PY=/workspace/venv/bin/python
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-GAPRULE-STRAT STOP $*"; exit 3; }
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz SR_ESTREAD_DIR=$RD
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"; [ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"; fi
for f in read_gaprule_strat.py read_gaprule10kb2.py make_planted_joins_b.py; do [ -f "$K/$f" ] || stop "MISSING-KIT-FILE $f"; done
[ -f "$GP/read2/GAPRULE10KB_CONTROLS.json" ] || stop "NO-CONTROLS-FILE"
[ -e "$OUT/GAPRULE10KB_STRATEGIC.json" ] && stop "ALREADY-READ"
mkdir -p "$OUT"
sha256sum "$K/read_gaprule_strat.py" "$K/read_gaprule10kb2.py" "$GP/read2/GAPRULE10KB_CONTROLS.json" "$RD/strat_roll_read.py" > "$OUT/READER_HASHES.txt"
( cd "$RD" && PYTHONPATH="$K" "$PY" "$K/read_gaprule_strat.py" --pull "$PL" --code "$CODE" --grid-dir "$W/longroll" --w "$W" --tree "$W/tree" \
    --controls "$GP/read2/GAPRULE10KB_CONTROLS.json" --out "$OUT/GAPRULE10KB_STRATEGIC.json" > "$OUT/read_strat.log" 2>&1 ); rc=$?
grep -E '^s_strategic|^STRATEGIC|^ZZ READ-GAPRULE-STRAT|Traceback|Error|REFUSED' "$OUT/read_strat.log" | cut -c1-200 | head -n 20
echo "ZZ PODREAD-GAPRULE-STRAT DONE rc $rc $(ts)"
