#!/usr/bin/env bash
# podread_cropscale10k.sh -- pod-side read of card PREREG_V8F_PROOF_ADD106 (CROPSCALE10K), derived from podread_visseed10k.sh (card ADD104). CPU only (CUDA_VISIBLE_DEVICES=-1 is set here), no rollout.
# Stage controls: needs both done markers of the rollout queue (or the one for both model seeds), no stop marker of it, the done markers of the fit stages FITS and PERM, and ENC_CONTROLS.json; links the ten run folders of this
#   card and the banked run folders of card ADD102 (V0, VD5c), card ADD104 (F2, F3, F4) and card ADD105 (E5c) into the pull folder of the readers (RR and RLc must be there already); runs the reader's selftest and then
#   read_cropscale10k.py --stage controls (writes $OUT/CROPSCALE10K_CONTROLS.json; values are in the file and are not printed). The STRATEGIC family and the T0 read are computed inside that stage.
# Stage names: read_cropscale10k.py --stage names on that controls file (the reader refuses unless g0 to g10 are all present and pass), writes $OUT/CROPSCALE10K_RESULT.json and the names log $OUT/read_names.log.
# Never reads twice into one file. It deletes, moves and stops nothing.
# Usage: bash podread_cropscale10k.sh controls|names
# Last line: ZZ PODREAD-CROPSCALE10K <stage> DONE rc <n>   or   ZZ PODREAD-CROPSCALE10K STOP <why>.
set -u
STAGE=${1:-}
V=/workspace/v8proof; W=$V/integ10k; P=$V/podread; RD=$P/root/integ10k; PL=$RD/pull; CODE=$P/code
CS=$V/cropscale; K=$CS/kit_read; KE=$CS/kit_enc; KF=$CS/kit_fit; OUT=$CS/read; GF=$CS/gapfiles; FR=$CS/fitroot
R1=$V/cropread/run1; MID=$V/visseed10k/fitroot; CHK=$V/visrule10k/check_v
PY=/workspace/venv/bin/python
WRAP=$W/build/v8f_t1_eval_reader_rule_rzgfhqetjv.py
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-CROPSCALE10K STOP $*"; exit 3; }
case "$STAGE" in controls|names) ;; *) stop "usage: controls|names";; esac
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz SR_ESTREAD_DIR=$RD
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"; [ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"; fi
for f in "$K/read_cropscale10k.py" "$K/lead_pytest.out" "$KF/make_scale_gapfiles.py" "$KF/lead_pytest.out" "$KE/ctl_g0g1.py" "$KE/lead_pytest.out"; do [ -f "$f" ] || stop "MISSING-KIT-FILE $f"; done
MODS="read_estclose10k_n2.py read_handsrc_decomp.py read_closrc10k.py read_agedose_decomp.py read_agerule_decomp.py read_longroll10k.py read_longroll10k_n3.py read_longroll10k_n4.py strat_roll_read.py"
for m in $MODS; do [ -f "$RD/$m" ] || stop "MISSING-INPUT $RD/$m"; done
if [ "$STAGE" = controls ]; then
  { [ -f "$CS/CROPSCALE10K_DONE.txt" ] || { [ -f "$CS/CROPSCALE10K_DONE_s0.txt" ] && [ -f "$CS/CROPSCALE10K_DONE_s1.txt" ]; }; } || stop "QUEUE-NOT-DONE"
  for sf in "$CS"/CROPSCALE10K_STOP*.txt; do [ -e "$sf" ] && stop "QUEUE-STOPPED"; done
  for st in FITS PERM; do [ -f "$CS/DONE_$st.txt" ] || stop "FIT-STAGE-NOT-DONE $st"; done
  [ -e "$OUT/CROPSCALE10K_CONTROLS.json" ] && stop "ALREADY-READ controls"
  for f in "$W/longroll/grid_p30.json" "$CS/ENC_CONTROLS.json" "$CS/lists/train_final.txt" "$CS/lists/LISTS_LARGE_RECORD.json" "$GF/CROPSCALE_GAPFILES_RECORD.json" "$GF/VGAPL_s2.npz" "$GF/VGAPL_s3.npz" "$GF/VGAPL_s4.npz" \
           "$GF/VGAPLE.npz" "$GF/VGAPSE3.npz" "$CHK/CHECK_WRAPPER_V_DONE.txt" "$CHK/CHECK_WRAPPER_V.hashes" "$V/hierrep/pins/join_eval141.npz" "$WRAP" "$R1/lists/train_final.txt" "$R1/lists/eval141.txt" \
           "$V/cropread/enc_out/lists/train_eligible.txt" "$V/visseed10k/gapfiles/VISSEED_GAPFILES_RECORD.json" "$R1/fits/D_H5_s0/FIT_RECORD.json" "$R1/fits/D_H5_s1/FIT_RECORD.json" \
           "$FR/fits/D_H5_s2_perm/FIT_RECORD.json" "$FR/dump/D_H5_perm/reader_s2_score.npz" "$FR/dump/D_H5_perm/reader_s2_meta.json"; do
    [ -f "$f" ] || stop "MISSING-INPUT $f"
  done
  for s in 2 3 4; do
    for f in "$FR/fits/D_H5_s$s/FIT_RECORD.json" "$FR/fits/D_H5_s$s/head.pt" "$FR/dump/D_H5/reader_s${s}_score.npz" "$FR/dump/D_H5/reader_s${s}_meta.json" "$MID/fits/D_H5_s$s/FIT_RECORD.json" \
             "$MID/dump/D_H5/reader_s${s}_score.npz" "$MID/dump/D_H5/reader_s${s}_meta.json"; do
      [ -f "$f" ] || stop "MISSING-INPUT $f"
    done
  done
  for s in 0 1; do [ -f "$R1/dump/D_H5/reader_s${s}_score.npz" ] || stop "MISSING-INPUT $R1/dump/D_H5/reader_s${s}_score.npz"; done
  for x in gG2 gG3 gG4 gGEc gE3c vV0 vVD5c fF2 fF3 fF4 eE5c; do for s in 0 1; do
    d=$W/t1/I10RQ50KF5P30Hb20${x}_s$s; [ -f "$d/T1_DONE" ] || stop "MISSING-RUN $d"
    [ -e "$PL/$(basename "$d")" ] || ln -s "$d" "$PL/$(basename "$d")"
  done; done
  for r in RQ50KF5P30Hb20 RLcKF5P30Hb20; do for s in 0 1; do [ -d "$PL/I10${r}_s$s" ] || stop "MISSING-BANKED-RUN I10${r}_s$s"; done; done
  mkdir -p "$OUT"
  ( cd "$RD" && sha256sum $MODS; sha256sum "$K/read_cropscale10k.py" "$KF/make_scale_gapfiles.py" "$KE/ctl_g0g1.py" "$K/lead_pytest.out" "$KF/lead_pytest.out" "$KE/lead_pytest.out" "$WRAP" "$CS/ENC_CONTROLS.json" \
      "$CS/lists/train_final.txt" "$CS/lists/LISTS_LARGE_RECORD.json" "$GF/CROPSCALE_GAPFILES_RECORD.json" "$GF"/*.npz "$CHK/CHECK_WRAPPER_V_DONE.txt" "$R1/fits/D_H5_s0/FIT_RECORD.json" "$R1/fits/D_H5_s1/FIT_RECORD.json" ) > "$OUT/READER_HASHES.txt"
  ( cd "$RD" && "$PY" "$K/read_cropscale10k.py" --selftest > "$OUT/selftest_pod.log" 2>&1 ); st=$?
  tail -n 1 "$OUT/selftest_pod.log" | cut -c1-120
  [ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
  ( cd "$RD" && "$PY" "$K/read_cropscale10k.py" --stage controls --pull "$PL" --code "$CODE" --grid-dir "$W/longroll" --w "$W" --tree "$W/tree" --wrapper-file "$WRAP" \
      --cs-root "$CS" --pilot-run "$R1" --gap-dir "$GF" --gap-record "$GF/CROPSCALE_GAPFILES_RECORD.json" --fit-root "$FR" --large-list "$CS/lists/train_final.txt" --lists-record "$CS/lists/LISTS_LARGE_RECORD.json" \
      --eval-list "$R1/lists/eval141.txt" --pilot-list "$R1/lists/train_final.txt" --eligible "$V/cropread/enc_out/lists/train_eligible.txt" --old-fits "$R1/fits" --old-dump-root "$R1/dump" --mid-root "$MID" \
      --rec104 "$V/visseed10k/gapfiles/VISSEED_GAPFILES_RECORD.json" --enc-controls "$CS/ENC_CONTROLS.json" --eval-join "$V/hierrep/pins/join_eval141.npz" \
      --wrapper-check "$CHK/CHECK_WRAPPER_V_DONE.txt" --wrapper-hashes "$CHK/CHECK_WRAPPER_V.hashes" --tests-enc "$KE/lead_pytest.out" --tests-fit "$KF/lead_pytest.out" --tests-read "$K/lead_pytest.out" \
      --out "$OUT/CROPSCALE10K_CONTROLS.json" > "$OUT/read_controls.log" 2>&1 ); rc=$?
  grep -E '^g([0-9]|10) (PASS|FAIL)|^ZZ READ-CROPSCALE10K|Traceback|Error|REFUSED|STOP' "$OUT/read_controls.log" | cut -c1-300 | head -n 20
  echo "ZZ PODREAD-CROPSCALE10K controls DONE rc $rc $(ts)"
else
  [ -f "$OUT/CROPSCALE10K_CONTROLS.json" ] || stop "NO-CONTROLS-FILE"
  [ -e "$OUT/CROPSCALE10K_RESULT.json" ] && stop "ALREADY-READ names"
  ( cd "$RD" && "$PY" "$K/read_cropscale10k.py" --stage names --controls "$OUT/CROPSCALE10K_CONTROLS.json" --out "$OUT/CROPSCALE10K_RESULT.json" --log "$OUT/read_names.log" > "$OUT/read_names.out" 2>&1 ); rc=$?
  tail -n 1 "$OUT/read_names.out" | cut -c1-200
  echo "ZZ PODREAD-CROPSCALE10K names DONE rc $rc $(ts)"
fi
