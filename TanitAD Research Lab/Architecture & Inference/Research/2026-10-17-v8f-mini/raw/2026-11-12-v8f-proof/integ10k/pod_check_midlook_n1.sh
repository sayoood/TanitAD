#!/usr/bin/env bash
# pod_check_midlook_n1.sh -- form of record of the pod check of study "midlook-closing" (note MIDLOOK_CLOSING_CHECK_NOTE_N1.md): the analysis part of check_midlook_closing.py repeated on the
# run folders already written by pod_check_midlook_b.sh ($W/midlook_test/chk1), with --analyze-only and --tol-reader 1e-4. No rollout is repeated. Writes two files into $W/midlook_test only.
# No rm, no mv, no kill. Refuses to run twice. One other compute process is admitted on the GPU (never more than three GPU jobs).
# Markers: $W/midlook_test/CHECK_DONE_chk1n1.txt (last line of the check) or CHECK_STOP_n1.txt.
set -u
W=/workspace/v8proof/integ10k; B=$W/build; TREE=$B/tree_ov; V=/workspace/v8proof; PY=/workspace/venv/bin/python
T=$W/midlook_test; WD=$T/chk1; TAG=chk1n1; TINY=$W/handsrc_test/grid_tiny_p30.json
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) $*" > "$T/CHECK_STOP_n1.txt"; echo "ZZ POD-CHECK-MIDLOOK-N1 STOP $*"; exit 3; }
[ -d "$WD" ] || stop "NO-WORKDIR $WD"
[ -e "$T/CHECK_DONE_$TAG.txt" ] && stop "ALREADY-RUN $TAG"
[ -e "$T/CHECK_MIDLOOK_CLOSING_$TAG.txt" ] && stop "OUTPUT-EXISTS $TAG"
[ -f "$TINY" ] || stop "NO-TINY-GRID"
for f in v8f_t1_eval_reader_rule_rzgfhq.py v8f_t1_eval_reader_rule_rzgfhqc.py check_midlook_closing.py tok_reader.py; do [ -f "$B/$f" ] || stop "MISSING-BUILD-FILE $f"; done
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); [ "$free_gb" -ge 28 ] || stop "MEMORY free ${free_gb} GB"
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); [ "$n" -le 1 ] || stop "GPU-BUSY $n"
OUT=$T/CHECK_MIDLOOK_CLOSING_$TAG.txt
sha256sum "$B/check_midlook_closing.py" "$B/v8f_t1_eval_reader_rule_rzgfhqc.py" "$B/v8f_t1_eval_reader_rule_rzgfhq.py" | cut -c1-16,65- > "$OUT.hashes"
DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 timeout 3600 "$PY" "$B/check_midlook_closing.py" \
  --parent "$B/v8f_t1_eval_reader_rule_rzgfhq.py" --new "$B/v8f_t1_eval_reader_rule_rzgfhqc.py" --tree "$TREE" --ckpt "$W/ckpt/UE_s0/A/ckpt_last.pt" \
  --tracks "$V/data/inputs/native_tracks.pt" --eval-cache "$W/eval_cache" --eval-join "$V/hierrep/pins/join_eval141.npz" --grid "$TINY" \
  --reader "$W/lr16/TK16_s0/ckpt_last.pt" --qmap "$W/qmap_sweep/QMAP_LR16_s0_q50.json" \
  --workdir "$WD" --distrule-dir "$B" --python "$PY" --analyze-only --tol-reader 1e-4 > "$OUT" 2>&1
echo "rc $? $(ts)" >> "$OUT.rc"
tail -n 1 "$OUT" > "$T/CHECK_DONE_$TAG.txt"
echo "ZZ POD-CHECK-MIDLOOK-N1 DONE $(tail -n 1 "$OUT" | cut -c1-200)"
