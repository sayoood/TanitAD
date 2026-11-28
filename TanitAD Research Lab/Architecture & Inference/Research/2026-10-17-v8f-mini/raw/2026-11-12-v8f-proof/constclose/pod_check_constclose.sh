#!/usr/bin/env bash
# pod_check_constclose.sh -- the pod check of card ADD80 (CONSTCLOSE10K), control k1: the constant reaches the rule. Runs check_constclose.py: ONE small rollout of the wrapper
# v8f_t1_eval_reader_rule_rzgfhqe.py in form RRe with const_C20_H1.npz (2.0 m/s) on the first three clips of the pinned grid (model seed 0, readout K, --rule-every 5, q50 map), and the checks on its logs.
# Writes only into $W/constclose_test. No process listing or stopping, no file removal or move. Refuses to run twice with the same tag. One other GPU compute process is admitted (never more than three GPU jobs).
# Environment (all optional): TAG (default ck1; the marker and record names), RUNTAG (the run folder, default = TAG), ANALYZE=1 (only read the run folder already there; give a new TAG and the RUNTAG of the
# finished rollout), NCLIPS (default 3), CONSTDIR (default /workspace/v8proof/closeread/est_const; the files written by make_const_files.py).
# Markers: $W/constclose_test/CHECK_CONST_DONE_$TAG.txt (the last line of the check: ZZ CHECK-CONSTCLOSE PASS ... or FAIL ...) or CHECK_CONST_STOP_$TAG.txt.
# Record: $W/constclose_test/CHECK_CONSTCLOSE_$TAG.txt (output), .hashes (sha256 prefixes of the files: the check, the wrapper, closest.py and the three constant files), .rc.
set -u
W=/workspace/v8proof/integ10k; B=$W/build; TREE=$B/tree_ov; V=/workspace/v8proof; PY=/workspace/venv/bin/python
T=$W/constclose_test
TAG=${TAG:-ck1}; RUNTAG=${RUNTAG:-$TAG}; ANALYZE=${ANALYZE:-0}; NCL=${NCLIPS:-3}; CD=${CONSTDIR:-$V/closeread/est_const}
WD=$T/$RUNTAG
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { mkdir -p "$T"; echo "$(ts) $*" > "$T/CHECK_CONST_STOP_$TAG.txt"; echo "ZZ POD-CHECK-CONSTCLOSE STOP $*"; exit 3; }
mkdir -p "$T" || stop "NO-DIR $T"
[ -e "$T/CHECK_CONST_DONE_$TAG.txt" ] && stop "ALREADY-RUN $TAG"
[ -e "$T/CHECK_CONSTCLOSE_$TAG.txt" ] && stop "OUTPUT-EXISTS $TAG"
if [ "$ANALYZE" = 1 ]; then [ -d "$WD" ] || stop "NO-WORKDIR $WD"; else [ -e "$WD" ] && stop "WORKDIR-EXISTS $WD"; fi
for f in v8f_t1_eval_reader_rule_rzgfhqe.py v8f_t1_eval_reader_rule_rzgfhqc.py check_constclose.py closest.py tok_reader.py dist_rule.py v8f_t1_eval_rule.py v8f_t1_eval_rule_mix.py qcal_map.py; do
  [ -f "$B/$f" ] || stop "MISSING-BUILD-FILE $f"
done
for c in C10 C20 C40; do [ -f "$CD/const_${c}_H1.npz" ] || stop "NO-CONST-FILE $CD/const_${c}_H1.npz"; done
[ -d "$TREE/taniteval/taniteval" ] && [ -e "$TREE/stack/scripts/v8f_t1_eval.py" ] || stop "NO-OVERLAY-TREE $TREE"
for f in "$V/data/inputs/native_tracks.pt" "$V/hierrep/pins/eval_grid_windows.json" "$V/hierrep/pins/join_eval141.npz" "$W/ckpt/UE_s0/A/ckpt_last.pt" "$W/lr16/TK16_s0/ckpt_last.pt" "$W/qmap_sweep/QMAP_LR16_s0_q50.json"; do
  [ -f "$f" ] || stop "MISSING-INPUT $f"
done
[ -f "$W/eval_cache/eval/manifest.jsonl" ] || stop "NO-EVAL-CACHE"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); [ "$free_gb" -ge 28 ] || stop "MEMORY free ${free_gb} GB"; fi
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); [ "$n" -lt 3 ] || stop "GPU-BUSY $n"
OUT=$T/CHECK_CONSTCLOSE_$TAG.txt
sha256sum "$B/check_constclose.py" "$B/v8f_t1_eval_reader_rule_rzgfhqe.py" "$B/closest.py" "$CD/const_C10_H1.npz" "$CD/const_C20_H1.npz" "$CD/const_C40_H1.npz" | cut -c1-16,65- > "$OUT.hashes"
AO=""; [ "$ANALYZE" = 1 ] && AO="--analyze-only"
DISTRULE_DIR=$B CLOSEST_DIR=$B PYTHONPATH="$TREE/stack" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 timeout 3600 "$PY" "$B/check_constclose.py" \
  --new "$B/v8f_t1_eval_reader_rule_rzgfhqe.py" --tree "$TREE" --ckpt "$W/ckpt/UE_s0/A/ckpt_last.pt" \
  --tracks "$V/data/inputs/native_tracks.pt" --eval-cache "$W/eval_cache" --eval-join "$V/hierrep/pins/join_eval141.npz" --grid "$V/hierrep/pins/eval_grid_windows.json" \
  --reader "$W/lr16/TK16_s0/ckpt_last.pt" --qmap "$W/qmap_sweep/QMAP_LR16_s0_q50.json" \
  --workdir "$WD" --distrule-dir "$B" --closest-dir "$B" --python "$PY" --n-clips "$NCL" --const-file "$CD/const_C20_H1.npz" $AO > "$OUT" 2>&1
echo "rc $? $(ts)" >> "$OUT.rc"
tail -n 1 "$OUT" > "$T/CHECK_CONST_DONE_$TAG.txt"
echo "ZZ POD-CHECK-CONSTCLOSE DONE $(tail -n 1 "$OUT" | cut -c1-200)"
