#!/usr/bin/env bash
# podread_presattr10k.sh -- pod-side read of card PREREG_V8F_PROOF_ADD114 (PRESATTR10K). CPU only (CUDA_VISIBLE_DEVICES=-1 is set here), no rollout.
# Stage controls: needs the done marker of the rollout queue and no stop marker, links the three new run names (both seeds) and the banked runs of the two settings (and the spread run) into the pull folder of the
#   readers, runs the reader's selftest first, then read_presattr10k.py --stage controls (writes $OUT/PRESATTR10K_CONTROLS.json). The time arguments of control e4 are filled here with stat -c %Y (file modification
#   times of the two lead test outputs in $GE/lead_tests), the first token of the first line of PRESATTR10K_START.txt (integer seconds since the epoch; the line is `<epoch seconds> <ISO time>Z pid <pid>`) and
#   date +%s at the start of the stage.
# Stage names: read_presattr10k.py --stage names on that controls file (the reader refuses unless e0 to e7 all pass), writes $OUT/PRESATTR10K_RESULT.json, the names log $OUT/read_names.log and $OUT/read_names.out.
# Never reads twice into one file. It deletes, moves and stops nothing.
# Usage: bash podread_presattr10k.sh controls|names
# Prints only the lines e<i> PASS|FAIL, the reader's last line and: ZZ PODREAD-PRESATTR10K <stage> DONE rc <n> <time>   (or ZZ PODREAD-PRESATTR10K STOP <why>).
set -u
STAGE=${1:-}
V=/workspace/v8proof; W=$V/integ10k; P=$V/podread; RD=$P/root/integ10k; PL=$RD/pull; CODE=$P/code
GE=$V/presattr; K=$GE/kit_read; OUT=$GE/read; LT=$GE/lead_tests; PJ=$GE/joins
CS=$V/cropscale; CHK=$V/visrule10k/check_v
MODEL=$CS/kit_read/read_cropscale10k.py
GAPF=$CS/gapfiles/VGAPLE.npz
LAF=$V/fallattr/gapfiles/VGAPA_LA.npz
GREC=$V/gaperr/gapfiles/GAPERR_GAPFILES_RECORD.json
PINNED=$V/hierrep/pins/eval_grid_windows.json
JOIN=$V/hierrep/pins/join_eval141.npz
PY=/workspace/venv/bin/python
WRAP=$W/build/v8f_t1_eval_reader_rule_rzgfhqetjv.py
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-PRESATTR10K STOP $*"; exit 3; }
case "$STAGE" in controls|names) ;; *) stop "usage: controls|names";; esac
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz SR_ESTREAD_DIR=$RD
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); [ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"; fi
for f in "$K/read_presattr10k.py" "$MODEL"; do [ -f "$f" ] || stop "MISSING-KIT-FILE $f"; done
MODS="read_estclose10k_n2.py read_handsrc_decomp.py read_closrc10k.py read_agedose_decomp.py read_agerule_decomp.py read_longroll10k.py read_longroll10k_n3.py read_longroll10k_n4.py strat_roll_read.py"
for m in $MODS; do [ -f "$RD/$m" ] || stop "MISSING-INPUT $RD/$m"; done
if [ "$STAGE" = controls ]; then
  CSTART=$(date +%s)
  [ -f "$GE/PRESATTR10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
  [ -e "$GE/PRESATTR10K_STOP.txt" ] && stop "QUEUE-STOPPED"
  [ -f "$GE/PRESATTR10K_START.txt" ] || stop "NO-START-MARKER"
  [ -e "$OUT/PRESATTR10K_CONTROLS.json" ] && stop "ALREADY-READ controls"
  for f in "$LT/lead_pytest_queue.out" "$LT/lead_pytest_read.out" "$W/longroll/grid_p30.json" "$PINNED" "$GAPF" "$LAF" "$GREC" "$PJ/JLE.npz" "$PJ/JLE_RECORD.json" "$CHK/CHECK_WRAPPER_V_DONE.txt" "$CHK/CHECK_WRAPPER_V.hashes" "$JOIN" "$WRAP"; do
    [ -f "$f" ] || stop "MISSING-INPUT $f"
  done
  for x in I10RLcKF5Hb20 I10RQ50KF5Hb20aLA I10RLcKF5Hb20pLE I10RQ50KF5Hb20gGEc I10RQ50KF5Hb20 I10RLcKF5P30Hb20 I10RQ50KF5P30Hb20aLA I10RLcKF5P30Hb20pLE I10RQ50KF5P30Hb20gGEc I10RQ50KF5P30Hb20 I10R0Hb20; do
    for s in 0 1; do
      d=$W/t1/${x}_s$s; [ -f "$d/T1_DONE" ] || stop "MISSING-RUN $d"
      [ -e "$PL/$(basename "$d")" ] || ln -s "$d" "$PL/$(basename "$d")"
    done
  done
  mkdir -p "$OUT"
  MQ=$(stat -c %Y "$LT/lead_pytest_queue.out"); MR=$(stat -c %Y "$LT/lead_pytest_read.out")
  QS=$(head -n 1 "$GE/PRESATTR10K_START.txt" | awk '{print $1}'); [[ "$QS" =~ ^[0-9]+$ ]] || stop "START-MARKER-UNREADABLE"
  ( cd "$RD" && sha256sum $MODS; sha256sum "$K/read_presattr10k.py" "$MODEL" "$LT/lead_pytest_queue.out" "$LT/lead_pytest_read.out" "$WRAP" "$GAPF" "$LAF" "$GREC" "$PJ/JLE.npz" "$PJ/JLE_RECORD.json" "$CHK/CHECK_WRAPPER_V_DONE.txt" "$JOIN" "$PINNED" ) > "$OUT/READER_HASHES.txt"
  ( cd "$RD" && "$PY" "$K/read_presattr10k.py" --selftest --model-reader "$MODEL" > "$OUT/selftest_pod.log" 2>&1 ); st=$?
  tail -n 1 "$OUT/selftest_pod.log" | cut -c1-120
  [ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
  ( cd "$RD" && "$PY" "$K/read_presattr10k.py" --stage controls --model-reader "$MODEL" --pull "$PL" --code "$CODE" --grid-dir "$W/longroll" --grid-pinned "$PINNED" --w "$W" --tree "$W/tree" --wrapper-file "$WRAP" \
      --gap-file "$GAPF" --gap-record "$GREC" --la-file "$LAF" --jle-file "$PJ/JLE.npz" --jle-record "$PJ/JLE_RECORD.json" --eval-join "$JOIN" \
      --wrapper-check "$CHK/CHECK_WRAPPER_V_DONE.txt" --wrapper-hashes "$CHK/CHECK_WRAPPER_V.hashes" \
      --tests-queue "$LT/lead_pytest_queue.out" --tests-read "$LT/lead_pytest_read.out" --tests-mtime-queue "$MQ" --tests-mtime-read "$MR" --queue-start "$QS" --controls-start "$CSTART" \
      --done-marker "$GE/PRESATTR10K_DONE.txt" --out "$OUT/PRESATTR10K_CONTROLS.json" > "$OUT/read_controls.log" 2>&1 ); rc=$?
  grep -E '^e[0-7] (PASS|FAIL)|^ZZ READ-PRESATTR10K' "$OUT/read_controls.log" | cut -c1-200 | head -n 10
  echo "ZZ PODREAD-PRESATTR10K controls DONE rc $rc $(ts)"
else
  [ -f "$OUT/PRESATTR10K_CONTROLS.json" ] || stop "NO-CONTROLS-FILE"
  [ -e "$OUT/PRESATTR10K_RESULT.json" ] && stop "ALREADY-READ names"
  ( cd "$RD" && "$PY" "$K/read_presattr10k.py" --stage names --model-reader "$MODEL" --controls "$OUT/PRESATTR10K_CONTROLS.json" --out "$OUT/PRESATTR10K_RESULT.json" --log "$OUT/read_names.log" > "$OUT/read_names.out" 2>&1 ); rc=$?
  tail -n 1 "$OUT/read_names.out" | cut -c1-200
  echo "ZZ PODREAD-PRESATTR10K names DONE rc $rc $(ts)"
fi
