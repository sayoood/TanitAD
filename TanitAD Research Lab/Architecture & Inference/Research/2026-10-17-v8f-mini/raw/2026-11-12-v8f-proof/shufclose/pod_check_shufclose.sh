#!/usr/bin/env bash
# pod_check_shufclose.sh -- the pod check of the wrapper v8f_t1_eval_reader_rule_rzgfhqet.py (card ADD83, control u0: the table source is the estimate). Runs check_shufclose.py:
# four small rollouts of the new wrapper on a subset of the pinned grid (form RR; form RRe with the table of the estimate itself ident_L25_s0.npz, a table of zeros and a table of +2 m/s;
# model seed 0, readout K, --rule-every 5, q50 map), three refusals, and the comparison with the banked run I10RQ50eL25KF5Hb20_s0 of card ADD79.
# Writes only into $W/shufclose_test. No process listing or stopping, no file removal or move. Refuses to run twice with the same tag. One other GPU compute process is admitted
# (never more than three GPU jobs).
# Environment (all optional): TAG (default sk1; the marker and record names), RUNTAG (the run folder, default = TAG), ANALYZE=1 (only read the run folders already there; give a new TAG
# and the RUNTAG of the finished rollouts), TOL_BANKED (default 1e-3), NCLIPS (default 3), TABLES (default /workspace/v8proof/closeread/est_shuf).
# Markers: $W/shufclose_test/CHECK_SHUF_DONE_$TAG.txt (the last line of the check: ZZ CHECK-SHUFCLOSE PASS ... or FAIL ...) or CHECK_SHUF_STOP_$TAG.txt.
# Record: $W/shufclose_test/CHECK_SHUFCLOSE_$TAG.txt (output), .hashes (sha256 prefixes of the files), .rc.
set -u
W=/workspace/v8proof/integ10k; B=$W/build; TREE=$B/tree_ov; V=/workspace/v8proof; PY=/workspace/venv/bin/python
T=$W/shufclose_test
TABLES=${TABLES:-$V/closeread/est_shuf}
TAG=${TAG:-sk1}; RUNTAG=${RUNTAG:-$TAG}; ANALYZE=${ANALYZE:-0}; TOLB=${TOL_BANKED:-1e-3}; NCL=${NCLIPS:-3}
WD=$T/$RUNTAG
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { mkdir -p "$T"; echo "$(ts) $*" > "$T/CHECK_SHUF_STOP_$TAG.txt"; echo "ZZ POD-CHECK-SHUFCLOSE STOP $*"; exit 3; }
mkdir -p "$T" || stop "NO-DIR $T"
[ -e "$T/CHECK_SHUF_DONE_$TAG.txt" ] && stop "ALREADY-RUN $TAG"
[ -e "$T/CHECK_SHUFCLOSE_$TAG.txt" ] && stop "OUTPUT-EXISTS $TAG"
if [ "$ANALYZE" = 1 ]; then [ -d "$WD" ] || stop "NO-WORKDIR $WD"; else [ -e "$WD" ] && stop "WORKDIR-EXISTS $WD"; fi
for f in v8f_t1_eval_reader_rule_rzgfhqet.py v8f_t1_eval_reader_rule_rzgfhqe.py v8f_t1_eval_reader_rule_rzgfhqc.py check_shufclose.py closest.py tok_reader.py dist_rule.py v8f_t1_eval_rule.py v8f_t1_eval_rule_mix.py qcal_map.py; do
  [ -f "$B/$f" ] || stop "MISSING-BUILD-FILE $f"
done
[ -d "$TREE/taniteval/taniteval" ] && [ -e "$TREE/stack/scripts/v8f_t1_eval.py" ] || stop "NO-OVERLAY-TREE $TREE"
[ -f "$W/t1/I10RQ50eL25KF5Hb20_s0/T1_DONE" ] || stop "NO-BANKED-RUN I10RQ50eL25KF5Hb20_s0"
for f in "$TABLES/ident_L25_s0.npz" "$TABLES/zero_table.npz" "$TABLES/const2_table.npz" "$V/closeread/est/closest_L25_s0.npz"; do
  [ -f "$f" ] || stop "MISSING-TABLE $f"
done
for f in "$V/data/inputs/native_tracks.pt" "$V/hierrep/pins/eval_grid_windows.json" "$V/hierrep/pins/join_eval141.npz" "$W/ckpt/UE_s0/A/ckpt_last.pt" "$W/lr16/TK16_s0/ckpt_last.pt" "$W/qmap_sweep/QMAP_LR16_s0_q50.json"; do
  [ -f "$f" ] || stop "MISSING-INPUT $f"
done
[ -f "$W/eval_cache/eval/manifest.jsonl" ] || stop "NO-EVAL-CACHE"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); [ "$free_gb" -ge 28 ] || stop "MEMORY free ${free_gb} GB"; fi
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); [ "$n" -lt 3 ] || stop "GPU-BUSY $n"
OUT=$T/CHECK_SHUFCLOSE_$TAG.txt
sha256sum "$B/check_shufclose.py" "$B/v8f_t1_eval_reader_rule_rzgfhqet.py" "$B/v8f_t1_eval_reader_rule_rzgfhqe.py" "$B/closest.py" "$TABLES/ident_L25_s0.npz" "$TABLES/zero_table.npz" "$TABLES/const2_table.npz" | cut -c1-16,65- > "$OUT.hashes"
AO=""; [ "$ANALYZE" = 1 ] && AO="--analyze-only"
DISTRULE_DIR=$B CLOSEST_DIR=$B PYTHONPATH="$TREE/stack" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 timeout 7200 "$PY" "$B/check_shufclose.py" \
  --new "$B/v8f_t1_eval_reader_rule_rzgfhqet.py" --tree "$TREE" --ckpt "$W/ckpt/UE_s0/A/ckpt_last.pt" \
  --tracks "$V/data/inputs/native_tracks.pt" --eval-cache "$W/eval_cache" --eval-join "$V/hierrep/pins/join_eval141.npz" --grid "$V/hierrep/pins/eval_grid_windows.json" \
  --reader "$W/lr16/TK16_s0/ckpt_last.pt" --qmap "$W/qmap_sweep/QMAP_LR16_s0_q50.json" \
  --workdir "$WD" --distrule-dir "$B" --closest-dir "$B" --python "$PY" --n-clips "$NCL" --tol-banked "$TOLB" \
  --tables "$TABLES" --est-file "$V/closeread/est/closest_L25_s0.npz" --banked-est "$W/t1/I10RQ50eL25KF5Hb20_s0" $AO > "$OUT" 2>&1
echo "rc $? $(ts)" >> "$OUT.rc"
tail -n 1 "$OUT" > "$T/CHECK_SHUF_DONE_$TAG.txt"
echo "ZZ POD-CHECK-SHUFCLOSE DONE $(tail -n 1 "$OUT" | cut -c1-200)"
