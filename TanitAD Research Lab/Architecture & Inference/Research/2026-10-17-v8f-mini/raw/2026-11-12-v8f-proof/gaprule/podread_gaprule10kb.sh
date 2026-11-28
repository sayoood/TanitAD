#!/usr/bin/env bash
# podread_gaprule10kb.sh -- pod-side read of card PREREG_V8F_PROOF_ADD95 (GAPRULE10KB), after smoothreq/podread_smoothreq10k.sh and chain_read_smoothreq10k.sh. CPU only, no rollout.
# Stage controls: needs both done markers of the queue and no stop marker; links the twelve run folders into the pull folder of the readers; runs the reader's selftest and then
#   read_gaprule10kb.py --stage controls (writes $OUT/GAPRULE10KB_CONTROLS.json; values are in the file and are not printed).
# Stage names: read_gaprule10kb.py --stage names on that controls file (the reader refuses unless every control passed), writes $OUT/GAPRULE10KB_RESULT.json.
# Never reads twice into one file. It deletes, moves and stops nothing.
# Usage: bash podread_gaprule10kb.sh controls|names
# Last line: ZZ PODREAD-GAPRULE10KB <stage> DONE rc <n>   or   ZZ PODREAD-GAPRULE10KB STOP <why>.
set -u
STAGE=${1:-}
V=/workspace/v8proof; W=$V/integ10k; P=$V/podread; RD=$P/root/integ10k; PL=$RD/pull; CODE=$P/code
GP=$V/gaprule10kb; K=$GP/kit; OUT=$GP/read; CHK=$GP/check_j2
PY=/workspace/venv/bin/python
WRAP=$W/build/v8f_t1_eval_reader_rule_rzgfhqetj.py
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-GAPRULE10KB STOP $*"; exit 3; }
case "$STAGE" in controls|names) ;; *) stop "usage: controls|names";; esac
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz SR_ESTREAD_DIR=$RD
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"; [ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"; fi
for f in read_gaprule10kb.py make_planted_joins_b.py lead_pytest_b.out; do [ -f "$K/$f" ] || stop "MISSING-KIT-FILE $f"; done
MODS="read_estclose10k_n2.py read_handsrc_decomp.py read_closrc10k.py read_agedose_decomp.py read_agerule_decomp.py read_longroll10k.py read_longroll10k_n3.py read_longroll10k_n4.py strat_roll_read.py"
for m in $MODS; do [ -f "$RD/$m" ] || stop "MISSING-INPUT $RD/$m"; done
if [ "$STAGE" = controls ]; then
  { [ -f "$GP/GAPRULE10K_DONE.txt" ] || { [ -f "$GP/GAPRULE10K_DONE_s0.txt" ] && [ -f "$GP/GAPRULE10K_DONE_s1.txt" ]; }; } || stop "QUEUE-NOT-DONE"
  for sf in "$GP"/GAPRULE10K_STOP*.txt; do [ -e "$sf" ] && stop "QUEUE-STOPPED"; done
  [ -e "$OUT/GAPRULE10KB_CONTROLS.json" ] && stop "ALREADY-READ controls"
  for f in "$W/longroll/grid_p30.json" "$GP/joins/PLANTED_JOINS_RECORD.json" "$CHK/CHECK_WRAPPER_J_DONE.txt" "$CHK/CHECK_WRAPPER_J.hashes" "$V/hierrep/pins/join_eval141.npz" "$WRAP"; do
    [ -f "$f" ] || stop "MISSING-INPUT $f"
  done
  WSHA=$(sha256sum "$WRAP" | cut -c1-16)
  for x in G0 GR1 GR2 GW05 GC10 GC20; do for s in 0 1; do
    d=$W/t1/I10RLcKF5P30Hb20j${x}_s$s; [ -f "$d/T1_DONE" ] || stop "MISSING-RUN $d"
    [ -e "$PL/$(basename "$d")" ] || ln -s "$d" "$PL/$(basename "$d")"
  done; done
  for r in RQ50KF5P30Hb20 RLcKF5P30Hb20; do for s in 0 1; do [ -d "$PL/I10${r}_s$s" ] || stop "MISSING-BANKED-RUN I10${r}_s$s"; done; done
  mkdir -p "$OUT"
  ( cd "$RD" && sha256sum $MODS; sha256sum "$K/read_gaprule10kb.py" "$K/make_planted_joins_b.py" "$K/lead_pytest_b.out" "$WRAP" "$GP/joins/PLANTED_JOINS_RECORD.json" "$GP"/joins/*.npz "$CHK/CHECK_WRAPPER_J_DONE.txt" ) > "$OUT/READER_HASHES.txt"
  ( cd "$RD" && "$PY" "$K/read_gaprule10kb.py" --selftest > "$OUT/selftest_pod.log" 2>&1 ); st=$?
  tail -n 1 "$OUT/selftest_pod.log" | cut -c1-120
  [ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
  ( cd "$RD" && "$PY" "$K/read_gaprule10kb.py" --stage controls --pull "$PL" --code "$CODE" --grid-dir "$W/longroll" --w "$W" --tree "$W/tree" --wrapper-sha "$WSHA" \
      --planted-record "$GP/joins/PLANTED_JOINS_RECORD.json" --planted-dir "$GP/joins" --eval-join "$V/hierrep/pins/join_eval141.npz" \
      --wrapper-check "$CHK/CHECK_WRAPPER_J_DONE.txt" --tests-out "$K/lead_pytest_b.out" --out "$OUT/GAPRULE10KB_CONTROLS.json" > "$OUT/read_controls.log" 2>&1 ); rc=$?
  grep -E '^(g_grid_read|j[0-4]) (PASS|FAIL)|^ZZ READ-GAPRULE10K|Traceback|Error|REFUSED|STOP' "$OUT/read_controls.log" | cut -c1-300 | head -n 16
  echo "ZZ PODREAD-GAPRULE10KB controls DONE rc $rc $(ts)"
else
  [ -f "$OUT/GAPRULE10KB_CONTROLS.json" ] || stop "NO-CONTROLS-FILE"
  [ -e "$OUT/GAPRULE10KB_RESULT.json" ] && stop "ALREADY-READ names"
  ( cd "$RD" && "$PY" "$K/read_gaprule10kb.py" --stage names --controls "$OUT/GAPRULE10KB_CONTROLS.json" --out "$OUT/GAPRULE10KB_RESULT.json" > "$OUT/read_names.log" 2>&1 ); rc=$?
  tail -n 1 "$OUT/read_names.log" | cut -c1-200
  echo "ZZ PODREAD-GAPRULE10KB names DONE rc $rc $(ts)"
fi
