#!/usr/bin/env bash
# podread_visseed10k.sh -- pod-side read of card PREREG_V8F_PROOF_ADD104 (VISSEED10K), derived from podread_visrep10k.sh (card ADD103). CPU only (CUDA_VISIBLE_DEVICES=-1 is set here), no rollout.
# Stage controls: needs both done markers of the queue and no stop marker; links the six run folders of this card and the banked run folders of card ADD102 (V0, VD5c) and card ADD103 (VD5cx) into the pull
#   folder of the readers (RR and RLc must be there already); runs the reader's selftest and then read_visseed10k.py --stage controls (writes $OUT/VISSEED10K_CONTROLS.json; values are in the file and are not printed).
#   The STRATEGIC family is read inside that stage.
# Stage names: read_visseed10k.py --stage names on that controls file (the reader refuses unless every control passed), writes $OUT/VISSEED10K_RESULT.json.
# Never reads twice into one file. It deletes, moves and stops nothing.
# Usage: bash podread_visseed10k.sh controls|names
# Last line: ZZ PODREAD-VISSEED10K <stage> DONE rc <n>   or   ZZ PODREAD-VISSEED10K STOP <why>.
set -u
STAGE=${1:-}
V=/workspace/v8proof; W=$V/integ10k; P=$V/podread; RD=$P/root/integ10k; PL=$RD/pull; CODE=$P/code
GP=$V/visseed10k; K=$GP/kit; OUT=$GP/read; GF=$GP/gapfiles; FR=$GP/fitroot
A99=$V/cropread/run1/fits; CHK=$V/visrule10k/check_v
PY=/workspace/venv/bin/python
WRAP=$W/build/v8f_t1_eval_reader_rule_rzgfhqetjv.py
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-VISSEED10K STOP $*"; exit 3; }
case "$STAGE" in controls|names) ;; *) stop "usage: controls|names";; esac
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz SR_ESTREAD_DIR=$RD
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"; [ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"; fi
for f in read_visseed10k.py make_seed_gapfile.py lead_pytest.out; do [ -f "$K/$f" ] || stop "MISSING-KIT-FILE $f"; done
MODS="read_estclose10k_n2.py read_handsrc_decomp.py read_closrc10k.py read_agedose_decomp.py read_agerule_decomp.py read_longroll10k.py read_longroll10k_n3.py read_longroll10k_n4.py strat_roll_read.py"
for m in $MODS; do [ -f "$RD/$m" ] || stop "MISSING-INPUT $RD/$m"; done
if [ "$STAGE" = controls ]; then
  { [ -f "$GP/VISSEED10K_DONE.txt" ] || { [ -f "$GP/VISSEED10K_DONE_s0.txt" ] && [ -f "$GP/VISSEED10K_DONE_s1.txt" ]; }; } || stop "QUEUE-NOT-DONE"
  for sf in "$GP"/VISSEED10K_STOP*.txt; do [ -e "$sf" ] && stop "QUEUE-STOPPED"; done
  [ -e "$OUT/VISSEED10K_CONTROLS.json" ] && stop "ALREADY-READ controls"
  for f in "$W/longroll/grid_p30.json" "$GF/VISSEED_GAPFILES_RECORD.json" "$GF/VGAPF_s2.npz" "$GF/VGAPF_s3.npz" "$GF/VGAPF_s4.npz" \
           "$CHK/CHECK_WRAPPER_V_DONE.txt" "$CHK/CHECK_WRAPPER_V.hashes" "$V/hierrep/pins/join_eval141.npz" "$WRAP" "$FR/lists/eval141.txt" "$A99/D_H5_s0/FIT_RECORD.json" "$A99/D_H5_s1/FIT_RECORD.json" \
           "$FR/fits/D_H5_s2/FIT_RECORD.json" "$FR/fits/D_H5_s3/FIT_RECORD.json" "$FR/fits/D_H5_s4/FIT_RECORD.json"; do
    [ -f "$f" ] || stop "MISSING-INPUT $f"
  done
  for s in 2 3 4; do for x in score.npz meta.json; do [ -f "$FR/dump/D_H5/reader_s${s}_$x" ] || stop "MISSING-INPUT $FR/dump/D_H5/reader_s${s}_$x"; done; done
  for x in fF2 fF3 fF4 vV0 vVD5c wVD5cx; do for s in 0 1; do
    d=$W/t1/I10RQ50KF5P30Hb20${x}_s$s; [ -f "$d/T1_DONE" ] || stop "MISSING-RUN $d"
    [ -e "$PL/$(basename "$d")" ] || ln -s "$d" "$PL/$(basename "$d")"
  done; done
  for r in RQ50KF5P30Hb20 RLcKF5P30Hb20; do for s in 0 1; do [ -d "$PL/I10${r}_s$s" ] || stop "MISSING-BANKED-RUN I10${r}_s$s"; done; done
  mkdir -p "$OUT"
  ( cd "$RD" && sha256sum $MODS; sha256sum "$K/read_visseed10k.py" "$K/make_seed_gapfile.py" "$K/lead_pytest.out" "$WRAP" "$GF/VISSEED_GAPFILES_RECORD.json" "$GF"/*.npz "$CHK/CHECK_WRAPPER_V_DONE.txt" "$A99/D_H5_s0/FIT_RECORD.json" "$A99/D_H5_s1/FIT_RECORD.json" ) > "$OUT/READER_HASHES.txt"
  ( cd "$RD" && "$PY" "$K/read_visseed10k.py" --selftest > "$OUT/selftest_pod.log" 2>&1 ); st=$?
  tail -n 1 "$OUT/selftest_pod.log" | cut -c1-120
  [ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
  ( cd "$RD" && "$PY" "$K/read_visseed10k.py" --stage controls --pull "$PL" --code "$CODE" --grid-dir "$W/longroll" --w "$W" --tree "$W/tree" --wrapper-file "$WRAP" \
      --gap-record "$GF/VISSEED_GAPFILES_RECORD.json" --gap-dir "$GF" --fit-root "$FR" --add99-fits "$A99" --eval-list "$FR/lists/eval141.txt" --eval-join "$V/hierrep/pins/join_eval141.npz" \
      --wrapper-check "$CHK/CHECK_WRAPPER_V_DONE.txt" --wrapper-hashes "$CHK/CHECK_WRAPPER_V.hashes" --tests-out "$K/lead_pytest.out" --out "$OUT/VISSEED10K_CONTROLS.json" > "$OUT/read_controls.log" 2>&1 ); rc=$?
  grep -E '^(x[0-6]) (PASS|FAIL)|^ZZ READ-VISSEED10K|Traceback|Error|REFUSED|STOP' "$OUT/read_controls.log" | cut -c1-300 | head -n 16
  echo "ZZ PODREAD-VISSEED10K controls DONE rc $rc $(ts)"
else
  [ -f "$OUT/VISSEED10K_CONTROLS.json" ] || stop "NO-CONTROLS-FILE"
  [ -e "$OUT/VISSEED10K_RESULT.json" ] && stop "ALREADY-READ names"
  ( cd "$RD" && "$PY" "$K/read_visseed10k.py" --stage names --controls "$OUT/VISSEED10K_CONTROLS.json" --out "$OUT/VISSEED10K_RESULT.json" > "$OUT/read_names.log" 2>&1 ); rc=$?
  tail -n 1 "$OUT/read_names.log" | cut -c1-200
  echo "ZZ PODREAD-VISSEED10K names DONE rc $rc $(ts)"
fi
