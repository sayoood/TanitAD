#!/usr/bin/env bash
# post_pbn.sh -- after chain_pbn.sh: Q1 probes (CPU) -> Q2 LiDAR read (CPU) -> copy raw JSONs into the package -> assemble RESULT.
# Waits for the capture chain's end marker; refuses on a capture failure. Markers ZZPOST-<stage>-<rc>ZZ in $W/logs/post.log.
set -u
C="$(cd "$(dirname "$0")" && pwd)"
PKG="$(cd "$C/.." && pwd)"
W=${PBN_WORK:-F:/perc_bottleneck}
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export TMP=F:/c_overflow_2026-10-26/tmp TEMP=F:/c_overflow_2026-10-26/tmp PYTHONUTF8=1 OMP_NUM_THREADS=8
P="$W/logs/post.log"
echo "ZZPOST-start-0ZZ $(date +%T)" >> "$P"
until grep -q -E "ZZPBN-alldone-0ZZ|ZZPBN-cap_[a-z]+-[1-9][0-9]*ZZ|ZZPBN-lock-1ZZ" "$W/logs/chain.log" 2>/dev/null; do sleep 20; done
if ! grep -q "ZZPBN-alldone-0ZZ" "$W/logs/chain.log"; then echo "ZZPOST-capturefailed-1ZZ $(date +%T)" >> "$P"; exit 2; fi
mkdir -p "$PKG/raw" "$W/q1" "$W/q2"
"$PY" -X utf8 "$C/pbn_probe.py" --cap-train "$W/cap_train" --cap-eval "$W/cap_eval" --out "$W/q1" \
    --prereg "$PKG/PREREG_PERC_BOTTLENECK.md" --seal "$PKG/PREREG_PERC_BOTTLENECK.sha256" > "$W/logs/q1.log" 2>&1
rc=$?; echo "ZZPOST-q1-${rc}ZZ $(date +%T)" >> "$P"
"$PY" -X utf8 "$C/pbn_q2_lidar.py" --cap-eval "$W/cap_eval" --out "$W/q2/RESULT_Q2.json" > "$W/logs/q2.log" 2>&1
rc2=$?; echo "ZZPOST-q2-${rc2}ZZ $(date +%T)" >> "$P"
[ -f "$W/q1/RESULT_Q1.json" ] && cp "$W/q1/RESULT_Q1.json" "$PKG/raw/RESULT_Q1.json"
[ -f "$W/q2/RESULT_Q2.json" ] && cp "$W/q2/RESULT_Q2.json" "$PKG/raw/RESULT_Q2.json"
[ -f "$W/q3/RESULT_Q3.json" ] && cp "$W/q3/RESULT_Q3.json" "$PKG/raw/RESULT_Q3.json"
cp "$W/cap_eval/capture.json" "$PKG/raw/capture_eval.json"; cp "$W/cap_train/capture.json" "$PKG/raw/capture_train.json"
"$PY" -X utf8 "$C/pbn_assemble.py" --pkg "$PKG" > "$W/logs/assemble.log" 2>&1
rc3=$?; echo "ZZPOST-assemble-${rc3}ZZ $(date +%T)" >> "$P"
[ $rc -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ] && echo "ZZPOST-done-0ZZ $(date +%T)" >> "$P"
