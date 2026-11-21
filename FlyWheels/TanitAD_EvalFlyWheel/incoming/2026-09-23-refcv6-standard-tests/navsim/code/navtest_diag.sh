#!/usr/bin/env bash
# navtest on W3's 200-token subset: R6_A1 + its inference-seed replicate R6_A1_s1 (the navtest seed
# floor) + SPEC §12's diagnostic arms (R6_VMAXOFF, and the
# PRIVILEGED R6_VMAXORACLE) for ONE checkpoint, through the one-command runner.
#   bash code/navtest_diag.sh <ckpt path> <out dir>
# Waits (artifact-gated, never on an exit code) for: the 416 navtest bank at 32/32 shards, the
# checkpoint file (a milestone is fetched + md5-verified by the milestone runner — its MD5SUMS line
# is required when the file is not the step-1,000 kit checkpoint), and the oracle input file.
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
CK="$1"; OUT="$2"
B=D:/Archive/devbox-C/navsim/exp/refcv6_navtest416/frame_bank
SUB=D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/A1_sub200_tokens.json
ORA="$P/raw/inputs/vmax_oracle_navtest.json"
mkdir -p "$OUT"
L="$OUT/navtest_diag.log"
echo "WAIT ckpt=$CK $(date -u +%FT%TZ)" >> "$L"
until [ "$(ls $B/shard_*.DONE.json 2>/dev/null | wc -l)" -ge 32 ] && [ -s "$ORA" ] && [ -s "$CK" ] \
      && { [ "$(basename "$CK")" = "ckpt_step1000.pt" ] || grep -q " $(basename "$CK") " "$(dirname "$CK")/MD5SUMS" 2>/dev/null; }; do
  sleep 300
done
echo "READY $(date -u +%FT%TZ)" >> "$L"
"$PY" "$P/code/run_navsim_refcv6.py" --ckpt "$CK" --splits navtest \
   --arms "navtest=R6_A1,R6_A1_s1,R6_VMAXOFF,R6_VMAXORACLE" --tokens-navtest "$SUB" \
   --device auto --gpu-wait-s 0 --out "$OUT" >> "$L" 2>&1
for f in summary_navtest.json decomposition_navtest.json MILESTONE_SUMMARY.json; do
  if [ -s "$OUT/$f" ]; then echo "ARTIFACT_OK $f" >> "$L"; else echo "ARTIFACT_MISSING $f" >> "$L"; fi
done
echo "ZZNAVTESTDIAGDONEZZ $(date -u +%FT%TZ)" >> "$L"
