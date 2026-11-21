#!/usr/bin/env bash
# RAM-gated, RETRYING official-scorer queue (one scorer at a time).
#   bash code/score_queue6.sh <split> <out_dir> <exp_tag> <arm>=<seam.npz> [<arm>=<seam.npz> ...]
#   (an <arm>=OFFICIAL:<agent> entry scores a devkit agent, e.g. CV_official=OFFICIAL:constant_velocity_agent)
# ⚠️ WHY (MEASURED 2026-09-24 00:46): other sessions drove the box to 1.9 GB free and E1's RAM guard
# (correctly) aborted a warmup scoring at 54/204. Each arm therefore waits for >= RAM_MIN_GB free on
# 3 consecutive 30 s samples, and an arm whose log carries the guard's abort token is RETRIED (up to
# RETRIES). An arm whose counts already read PASS is skipped. The artifact decides, never an rc.
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
NPY="C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
SPLIT="$1"; OUT="$2"; TAG="$3"; shift 3
RAM_MIN_GB="${RAM_MIN_GB:-4}"; RETRIES="${RETRIES:-6}"
mkdir -p "$OUT"
Q="$OUT/score_queue.log"
ram_gate() {
  local ok=0 f=0
  while [ $ok -lt 3 ]; do
    f=$("$PY" -c "import psutil; print(int(psutil.virtual_memory().available/2**30))")
    if [ "$f" -ge "$RAM_MIN_GB" ]; then ok=$((ok+1)); else ok=0; fi
    [ $ok -lt 3 ] && sleep 30
  done
  echo "RAMGATE $1 free=${f}GB $(date -u +%FT%TZ)" >> "$Q"
}
for spec in "$@"; do
  arm="${spec%%=*}"; src="${spec#*=}"
  if [ "$SPLIT" = "warmup_two_stage" ]; then t="$arm"; else t="${arm}__${SPLIT}"; fi
  cnt="$OUT/score_${t}.counts.json"
  for try in $(seq 1 "$RETRIES"); do
    if "$PY" -c "import json,sys; sys.exit(0 if json.load(open(r'$cnt')).get('status')=='PASS' else 1)" 2>/dev/null; then
      echo "PASS_ALREADY $arm $(date -u +%FT%TZ)" >> "$Q"; break
    fi
    ram_gate "$arm#$try"
    if [ "${src#OFFICIAL:}" != "$src" ]; then
      "$NPY" "$P/code/score_arm6.py" --arm "$arm" --official-agent "${src#OFFICIAL:}" --split "$SPLIT" \
         --out "$OUT" --exp-tag "$TAG" > "$OUT/${arm}.driver.txt" 2>&1
    else
      "$NPY" "$P/code/score_arm6.py" --arm "$arm" --seam "$src" --split "$SPLIT" --out "$OUT" \
         --exp-tag "$TAG" > "$OUT/${arm}.driver.txt" 2>&1
    fi
    if grep -q "E1_RAM_GUARD_ABORT" "$OUT/score_${t}.log" 2>/dev/null; then
      mkdir -p "$OUT/_ramguard_abort"
      cp "$OUT/score_${t}.log" "$OUT/_ramguard_abort/score_${t}.try${try}.log" 2>/dev/null
      echo "RAMGUARD_ABORT $arm try=$try $(date -u +%FT%TZ)" >> "$Q"
      continue
    fi
    st=$("$PY" -c "import json; print(json.load(open(r'$cnt')).get('status'))" 2>/dev/null)
    echo "SCORED $arm status=${st:-NO_COUNTS} try=$try $(date -u +%FT%TZ)" >> "$Q"
    break
  done
done
echo "ZZSCOREQUEUEDONEZZ $(date -u +%FT%TZ)" >> "$Q"
