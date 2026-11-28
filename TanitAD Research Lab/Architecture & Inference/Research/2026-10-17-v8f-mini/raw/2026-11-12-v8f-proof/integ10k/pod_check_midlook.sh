#!/usr/bin/env bash
# pod_check_midlook.sh -- runs check_midlook_closing.py (part B) of study "midlook-closing" on the pod: parent wrapper against the new wrapper v8f_t1_eval_reader_rule_rzgfhqc.py on the tiny grid
# (3 clips, 27 windows; handsrc_test/grid_tiny_p30.json), model seed 0, reader seed 0 (map q50), readout causal_g, acc-gain 0, path-exec, kappa 0.2, pre-roll 30; twelve small rollouts and one refusal run.
# One GPU user at a time (stops when another compute process is on the GPU). Writes only into $W/midlook_test. No rm, no mv, no kill. Refuses to run twice.
# Markers: $W/midlook_test/CHECK_DONE_chk1.txt (last line of the check) or CHECK_STOP.txt.
# Inputs to be in $B (build folder) first: v8f_t1_eval_reader_rule_rzgfhq.py, v8f_t1_eval_reader_rule_rzgfhqc.py, check_midlook_closing.py, tok_reader.py and the distrule files.
set -u
W=/workspace/v8proof/integ10k; B=$W/build; TREE=$B/tree_ov; V=/workspace/v8proof; PY=/workspace/venv/bin/python
T=$W/midlook_test; WD=${1:-$T/chk1}; TINY=$W/handsrc_test/grid_tiny_p30.json
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "$T"
stop() { echo "$(ts) $*" > "$T/CHECK_STOP.txt"; echo "ZZ POD-CHECK-MIDLOOK STOP $*"; exit 3; }
[ -e "$WD" ] && stop "WORKDIR-EXISTS $WD"
[ -e "$T/CHECK_DONE_$(basename "$WD").txt" ] && stop "ALREADY-RUN $(basename "$WD")"
[ -f "$TINY" ] || stop "NO-TINY-GRID"
for f in v8f_t1_eval_reader_rule_rzgfhq.py v8f_t1_eval_reader_rule_rzgfhqc.py check_midlook_closing.py tok_reader.py; do [ -f "$B/$f" ] || stop "MISSING-BUILD-FILE $f"; done
for f in "$W/ckpt/UE_s0/A/ckpt_last.pt" "$W/lr16/TK16_s0/ckpt_last.pt" "$W/qmap_sweep/QMAP_LR16_s0_q50.json" "$V/data/inputs/native_tracks.pt" "$V/hierrep/pins/join_eval141.npz"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); [ "$free_gb" -ge 28 ] || stop "MEMORY free ${free_gb} GB"
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); [ "$n" = 0 ] || stop "GPU-BUSY $n"
OUT=$T/CHECK_MIDLOOK_CLOSING_$(basename "$WD").txt
DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 timeout 7200 "$PY" "$B/check_midlook_closing.py" \
  --parent "$B/v8f_t1_eval_reader_rule_rzgfhq.py" --new "$B/v8f_t1_eval_reader_rule_rzgfhqc.py" --tree "$TREE" --ckpt "$W/ckpt/UE_s0/A/ckpt_last.pt" \
  --tracks "$V/data/inputs/native_tracks.pt" --eval-cache "$W/eval_cache" --eval-join "$V/hierrep/pins/join_eval141.npz" --grid "$TINY" \
  --reader "$W/lr16/TK16_s0/ckpt_last.pt" --qmap "$W/qmap_sweep/QMAP_LR16_s0_q50.json" \
  --workdir "$WD" --distrule-dir "$B" --python "$PY" ${CHECK_EXTRA:-} > "$OUT" 2>&1
echo "rc $? $(ts)" >> "$OUT.rc"
tail -n 1 "$OUT" > "$T/CHECK_DONE_$(basename "$WD").txt"
echo "ZZ POD-CHECK-MIDLOOK DONE $(tail -n 1 "$OUT" | cut -c1-200)"
