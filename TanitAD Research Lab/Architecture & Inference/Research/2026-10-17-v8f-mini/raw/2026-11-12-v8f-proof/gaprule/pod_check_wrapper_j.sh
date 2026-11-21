#!/usr/bin/env bash
# pod_check_wrapper_j.sh -- card ADD94 (GAPRULE10K), control j2: the pod check of the wrapper v8f_t1_eval_reader_rule_rzgfhqetj.py, after shufclose/pod_check_shufclose.sh.
# Runs check_wrapper_j.py: small rollouts of form RLc (model seed 0, readout K, --rule-every 5, pre-roll of 30 steps (6 s)) on the first NCLIPS clips of the pre-roll grid with
# the generation rzgfhqet, the new generation without --rule-join, the new generation with --rule-join G0.npz, and one refusal.
# Writes only into /workspace/v8proof/gaprule10k. No process listing or stopping, no file removal or move. Refuses to run twice. Never more than three GPU jobs.
# Environment (optional): NCLIPS (default 8).
# Output: $GP/check_j.out (text of the check), $GP/check_j/CHECK_WRAPPER_J_DONE.txt and CHECK_WRAPPER_J.hashes (written by the check), $GP/CHECK_J_STOP.txt on a stop.
set -u
W=/workspace/v8proof/integ10k; B=$W/build; TREE=$B/tree_ov; V=/workspace/v8proof; PY=/workspace/venv/bin/python
GP=/workspace/v8proof/gaprule10k; K=$GP/kit; WD=$GP/check_j; OUT=$GP/check_j.out
NCL=${NCLIPS:-8}
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "$(ts) $*" > "$GP/CHECK_J_STOP.txt"; echo "ZZ POD-CHECK-WRAPPER-J STOP $*"; exit 3; }
[ -d "$GP" ] || { echo "ZZ POD-CHECK-WRAPPER-J STOP NO-ROOT $GP"; exit 3; }
[ -e "$OUT" ] && stop "ALREADY-RUN $OUT"
[ -e "$WD" ] && stop "WORKDIR-EXISTS $WD"
for f in v8f_t1_eval_reader_rule_rzgfhqetj.py v8f_t1_eval_reader_rule_rzgfhqet.py closest.py tok_reader.py dist_rule.py v8f_t1_eval_rule.py v8f_t1_eval_rule_mix.py qcal_map.py; do
  [ -f "$B/$f" ] || stop "MISSING-BUILD-FILE $f"
done
[ -f "$K/check_wrapper_j.py" ] || stop "MISSING-KIT-FILE check_wrapper_j.py"
[ -d "$TREE/taniteval/taniteval" ] && [ -e "$TREE/stack/scripts/v8f_t1_eval.py" ] || stop "NO-OVERLAY-TREE $TREE"
for f in "$V/data/inputs/native_tracks.pt" "$W/longroll/grid_p30.json" "$V/hierrep/pins/join_eval141.npz" "$W/ckpt/UE_s0/A/ckpt_last.pt" "$GP/joins/G0.npz" "$GP/joins/PLANTED_JOINS_RECORD.json"; do
  [ -f "$f" ] || stop "MISSING-INPUT $f"
done
[ -f "$W/eval_cache/eval/manifest.jsonl" ] || stop "NO-EVAL-CACHE"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); [ "$free_gb" -ge 28 ] || stop "MEMORY free ${free_gb} GB"; fi
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); [ "$n" -lt 3 ] || stop "GPU-BUSY $n"
echo "[$(ts)] start, NCLIPS $NCL, gpu jobs before $n" > "$GP/check_j.start"
DISTRULE_DIR=$B CLOSEST_DIR=$B PYTHONPATH="$TREE/stack" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 timeout 7200 "$PY" "$K/check_wrapper_j.py" \
  --old "$B/v8f_t1_eval_reader_rule_rzgfhqet.py" --new "$B/v8f_t1_eval_reader_rule_rzgfhqetj.py" --tree "$TREE" --ckpt "$W/ckpt/UE_s0/A/ckpt_last.pt" \
  --tracks "$V/data/inputs/native_tracks.pt" --eval-cache "$W/eval_cache" --eval-join "$V/hierrep/pins/join_eval141.npz" --grid "$W/longroll/grid_p30.json" \
  --workdir "$WD" --g0-join "$GP/joins/G0.npz" --distrule-dir "$B" --closest-dir "$B" --python "$PY" --n-clips "$NCL" > "$OUT" 2>&1
echo "rc $? $(ts)" > "$OUT.rc"
echo "ZZ POD-CHECK-WRAPPER-J DONE $(tail -n 1 "$OUT" | cut -c1-200)"
