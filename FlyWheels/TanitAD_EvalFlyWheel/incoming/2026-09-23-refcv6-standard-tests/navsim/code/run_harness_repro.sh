#!/usr/bin/env bash
# Deliverable 1 — THE HARNESS STILL READS ITS KNOWN VALUES.
# Re-score E2's two warmup controls through the SAME official scorer path the refcv6 arms use,
# sequentially (one scorer at a time: the box's RAM is shared), then compare every per-token cell
# with E2's banked CSVs (code/check_harness_repro.py). A FAIL is logged, never retried silently.
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
E2RAW="$P/../../2026-09-19-navsim-refcv4b-bridge/raw"
OUT="$P/raw/harness_repro"
mkdir -p "$OUT"
echo "START $(date -u +%FT%TZ)" >> "$OUT/queue.log"
"$PY" "$P/code/score_arm6.py" --arm CV_official --official-agent constant_velocity_agent \
      --out "$OUT" --exp-tag e6repro > "$OUT/CV_official.driver.txt" 2>&1
echo "CV_official rc=$? $(date -u +%FT%TZ)" >> "$OUT/queue.log"
"$PY" "$P/code/score_arm6.py" --arm STOP_zero --seam "$E2RAW/seam_STOP_zero.npz" \
      --out "$OUT" --exp-tag e6repro > "$OUT/STOP_zero.driver.txt" 2>&1
echo "STOP_zero rc=$? $(date -u +%FT%TZ)" >> "$OUT/queue.log"
echo "ZZQUEUEDONEZZ" >> "$OUT/queue.log"
