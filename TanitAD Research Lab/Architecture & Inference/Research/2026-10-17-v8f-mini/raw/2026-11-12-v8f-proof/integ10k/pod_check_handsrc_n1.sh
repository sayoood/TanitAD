#!/usr/bin/env bash
# pod_check_handsrc_n1.sh -- note N1 of card PREREG_V8F_PROOF_ADD73: runs check_handover_src_n1.py with --analyze-only on the run folders the first pod check left in
# $W/handsrc_test/chk1 (no rollout, CPU only). Writes $W/handsrc_test/CHECK_HANDOVER_SRC_chk1n1.txt and CHECK_DONE_chk1n1.txt (the last line of the check). No rm, no mv, no kill.
set -u
W=/workspace/v8proof/integ10k; B=$W/build; TREE=$B/tree_ov; V=/workspace/v8proof; PY=/workspace/venv/bin/python
T=$W/handsrc_test; WD=$T/chk1
[ -d "$WD" ] || { echo "ZZ POD-CHECK-HANDSRC-N1 STOP no workdir"; exit 3; }
[ -e "$T/CHECK_DONE_chk1n1.txt" ] && { echo "ZZ POD-CHECK-HANDSRC-N1 STOP already run"; exit 3; }
OUT=$T/CHECK_HANDOVER_SRC_chk1n1.txt
CUDA_VISIBLE_DEVICES=-1 DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 timeout 900 "$PY" "$B/check_handover_src_n1.py" \
  --parent "$B/v8f_t1_eval_reader_rule_rzgfhq.py" --new "$B/v8f_t1_eval_reader_rule_rzgfhqs.py" --tree "$TREE" --ckpt "$W/ckpt/UE_s0/A/ckpt_last.pt" \
  --tracks "$V/data/inputs/native_tracks.pt" --eval-cache "$W/eval_cache" --eval-join "$V/hierrep/pins/join_eval141.npz" --grid "$T/grid_tiny_p30.json" \
  --workdir "$WD" --distrule-dir "$B" --python "$PY" --analyze-only > "$OUT" 2>&1
echo "rc $?"
tail -n 1 "$OUT" > "$T/CHECK_DONE_chk1n1.txt"
grep -E '^FAIL|^ZZ|Traceback' "$OUT" | cut -c1-300 | head -n 8
grep -cE '^PASS' "$OUT"
echo "ZZ POD-CHECK-HANDSRC-N1 DONE"
