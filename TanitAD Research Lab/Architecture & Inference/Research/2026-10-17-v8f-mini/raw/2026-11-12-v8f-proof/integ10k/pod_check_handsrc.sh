#!/usr/bin/env bash
# pod_check_handsrc.sh -- runs check_handover_src.py (part B) of card PREREG_V8F_PROOF_ADD73 on the pod: parent wrapper against the new wrapper on a tiny grid (3 clips, 27 windows),
# model seed 0, readout K, form R0. One GPU user at a time (stops when another compute process is on the GPU). Writes only into $W/handsrc_test. No rm, no mv, no kill.
# Markers: $W/handsrc_test/CHECK_DONE.txt (last line of the check inside) or CHECK_STOP.txt.
set -u
W=/workspace/v8proof/integ10k; B=$W/build; TREE=$B/tree_ov; V=/workspace/v8proof; PY=/workspace/venv/bin/python
T=$W/handsrc_test; WD=${1:-$T/chk1}
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) $*" > "$T/CHECK_STOP.txt"; echo "ZZ POD-CHECK-HANDSRC STOP $*"; exit 3; }
[ -e "$WD" ] && stop "WORKDIR-EXISTS $WD"
[ -f "$T/grid_tiny_p30.json" ] || stop "NO-TINY-GRID"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); [ "$free_gb" -ge 28 ] || stop "MEMORY free ${free_gb} GB"
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); [ "$n" = 0 ] || stop "GPU-BUSY $n"
OUT=$T/CHECK_HANDOVER_SRC_$(basename "$WD").txt
DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 timeout 3000 "$PY" "$B/check_handover_src.py" \
  --parent "$B/v8f_t1_eval_reader_rule_rzgfhq.py" --new "$B/v8f_t1_eval_reader_rule_rzgfhqs.py" --tree "$TREE" --ckpt "$W/ckpt/UE_s0/A/ckpt_last.pt" \
  --tracks "$V/data/inputs/native_tracks.pt" --eval-cache "$W/eval_cache" --eval-join "$V/hierrep/pins/join_eval141.npz" --grid "$T/grid_tiny_p30.json" \
  --workdir "$WD" --distrule-dir "$B" --python "$PY" ${CHECK_EXTRA:-} > "$OUT" 2>&1
echo "rc $? $(ts)" >> "$OUT.rc"
tail -n 1 "$OUT" > "$T/CHECK_DONE_$(basename "$WD").txt"
echo "ZZ POD-CHECK-HANDSRC DONE $(tail -n 1 "$OUT" | cut -c1-200)"
