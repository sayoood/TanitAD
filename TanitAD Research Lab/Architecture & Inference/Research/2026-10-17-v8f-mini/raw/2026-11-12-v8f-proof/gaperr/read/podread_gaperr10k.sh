#!/usr/bin/env bash
# podread_gaperr10k.sh -- pod-side read of card PREREG_V8F_PROOF_ADD111 (GAPERR10K), generated from the r1 pod driver of card ADD110 by make_read_gaperr10k.py. CPU only (CUDA_VISIBLE_DEVICES=-1 is set here), no rollout.
# Stage controls: needs the done marker of the rollout queue and no stop marker, links the eight run folders of this card, the banked GEc and LA folders and the control run V0 into the pull folder of the readers
#   (RR and RLc must be there already), runs the reader's selftest first, then read_gaperr10k.py --stage controls (writes $OUT/GAPERR10K_CONTROLS.json). The time arguments of control b4 are filled
#   here with stat -c %Y (file modification times of the two lead_pytest.out files) and the first token of the first line of GAPERR10K_START.txt (integer seconds since the epoch; the line is
#   `<epoch seconds> <ISO time> pid <pid>`). The file LA of card ADD110 is read at $FA0/gapfiles/VGAPA_LA.npz (--la-file).
# Stage names: read_gaperr10k.py --stage names on that controls file (the reader refuses unless b0 to b5 all pass), writes $OUT/GAPERR10K_RESULT.json and the names log $OUT/read_names.log.
# Never reads twice into one file. It deletes, moves and stops nothing.
# Usage: bash podread_gaperr10k.sh controls|names
# Prints only the lines b<i> PASS|FAIL, the reader's last line and: ZZ PODREAD-GAPERR10K <stage> DONE rc <n>   (or ZZ PODREAD-GAPERR10K STOP <why>).
set -u
STAGE=${1:-}
V=/workspace/v8proof; W=$V/integ10k; P=$V/podread; RD=$P/root/integ10k; PL=$RD/pull; CODE=$P/code
FA=$V/gaperr; K=$FA/kit_read; KG=$FA/kit_gap; OUT=$FA/read; GF=$FA/gapfiles
FA0=$V/fallattr; LAF=$FA0/gapfiles/VGAPA_LA.npz
CS=$V/cropscale; CHK=$V/visrule10k/check_v
MODEL=$CS/kit_read/read_cropscale10k.py
JOIN=$V/hierrep/pins/join_eval141.npz
PY=/workspace/venv/bin/python
WRAP=$W/build/v8f_t1_eval_reader_rule_rzgfhqetjv.py
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-GAPERR10K STOP $*"; exit 3; }
case "$STAGE" in controls|names) ;; *) stop "usage: controls|names";; esac
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz SR_ESTREAD_DIR=$RD
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); [ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"; fi
for f in "$K/read_gaperr10k.py" "$K/lead_pytest.out" "$KG/lead_pytest.out" "$MODEL"; do [ -f "$f" ] || stop "MISSING-KIT-FILE $f"; done
MODS="read_estclose10k_n2.py read_handsrc_decomp.py read_closrc10k.py read_agedose_decomp.py read_agerule_decomp.py read_longroll10k.py read_longroll10k_n3.py read_longroll10k_n4.py strat_roll_read.py"
for m in $MODS; do [ -f "$RD/$m" ] || stop "MISSING-INPUT $RD/$m"; done
if [ "$STAGE" = controls ]; then
  [ -f "$FA/GAPERR10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"
  [ -e "$FA/GAPERR10K_STOP.txt" ] && stop "QUEUE-STOPPED"
  [ -f "$FA/GAPERR10K_START.txt" ] || stop "NO-START-MARKER"
  [ -e "$OUT/GAPERR10K_CONTROLS.json" ] && stop "ALREADY-READ controls"
  for f in "$W/longroll/grid_p30.json" "$GF/GAPERR_GAPFILES_RECORD.json" "$GF/VGAPA_NR.npz" "$GF/VGAPA_BR.npz" "$GF/VGAPA_SM.npz" "$GF/VGAPA_SL.npz" "$LAF" "$CHK/CHECK_WRAPPER_V_DONE.txt" \
           "$CHK/CHECK_WRAPPER_V.hashes" "$JOIN" "$WRAP"; do
    [ -f "$f" ] || stop "MISSING-INPUT $f"
  done
  for x in bNR bBR bSM bSL aLA gGEc vV0; do for s in 0 1; do
    d=$W/t1/I10RQ50KF5P30Hb20${x}_s$s; [ -f "$d/T1_DONE" ] || stop "MISSING-RUN $d"
    [ -e "$PL/$(basename "$d")" ] || ln -s "$d" "$PL/$(basename "$d")"
  done; done
  for r in RQ50KF5P30Hb20 RLcKF5P30Hb20; do for s in 0 1; do [ -d "$PL/I10${r}_s$s" ] || stop "MISSING-BANKED-RUN I10${r}_s$s"; done; done
  mkdir -p "$OUT"
  MG=$(stat -c %Y "$KG/lead_pytest.out"); MR=$(stat -c %Y "$K/lead_pytest.out")
  QS=$(head -n 1 "$FA/GAPERR10K_START.txt" | awk '{print $1}'); [[ "$QS" =~ ^[0-9]+$ ]] || stop "START-MARKER-UNREADABLE"
  ( cd "$RD" && sha256sum $MODS; sha256sum "$K/read_gaperr10k.py" "$MODEL" "$K/lead_pytest.out" "$KG/lead_pytest.out" "$WRAP" "$GF/GAPERR_GAPFILES_RECORD.json" "$GF"/*.npz "$LAF" "$CHK/CHECK_WRAPPER_V_DONE.txt" "$JOIN" ) > "$OUT/READER_HASHES.txt"
  ( cd "$RD" && "$PY" "$K/read_gaperr10k.py" --selftest --model-reader "$MODEL" > "$OUT/selftest_pod.log" 2>&1 ); st=$?
  tail -n 1 "$OUT/selftest_pod.log" | cut -c1-120
  [ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
  ( cd "$RD" && "$PY" "$K/read_gaperr10k.py" --stage controls --model-reader "$MODEL" --pull "$PL" --code "$CODE" --grid-dir "$W/longroll" --w "$W" --tree "$W/tree" --wrapper-file "$WRAP" \
      --fa "$FA" --gap-dir "$GF" --gap-record "$GF/GAPERR_GAPFILES_RECORD.json" --la-file "$LAF" --eval-join "$JOIN" --wrapper-check "$CHK/CHECK_WRAPPER_V_DONE.txt" --wrapper-hashes "$CHK/CHECK_WRAPPER_V.hashes" \
      --tests-gap "$KG/lead_pytest.out" --tests-read "$K/lead_pytest.out" --tests-mtime-gap "$MG" --tests-mtime-read "$MR" --queue-start "$QS" \
      --out "$OUT/GAPERR10K_CONTROLS.json" > "$OUT/read_controls.log" 2>&1 ); rc=$?
  grep -E '^b[0-5] (PASS|FAIL)|^ZZ READ-GAPERR10K' "$OUT/read_controls.log" | cut -c1-200 | head -n 10
  echo "ZZ PODREAD-GAPERR10K controls DONE rc $rc $(ts)"
else
  [ -f "$OUT/GAPERR10K_CONTROLS.json" ] || stop "NO-CONTROLS-FILE"
  [ -e "$OUT/GAPERR10K_RESULT.json" ] && stop "ALREADY-READ names"
  ( cd "$RD" && "$PY" "$K/read_gaperr10k.py" --stage names --model-reader "$MODEL" --controls "$OUT/GAPERR10K_CONTROLS.json" --out "$OUT/GAPERR10K_RESULT.json" --log "$OUT/read_names.log" > "$OUT/read_names.out" 2>&1 ); rc=$?
  tail -n 1 "$OUT/read_names.out" | cut -c1-200
  echo "ZZ PODREAD-GAPERR10K names DONE rc $rc $(ts)"
fi
