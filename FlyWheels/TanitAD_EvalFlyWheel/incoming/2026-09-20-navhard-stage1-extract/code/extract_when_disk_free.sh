#!/usr/bin/env bash
# Agent-free waiter: run the navhard stage-1 extraction once W3's navtest chain has released D:.
#
# WHY WAIT. Both jobs stream the SAME 32 archives on the SAME exFAT external disk. MEASURED
# 2026-09-20: one shard took >10 min with W3's chain running concurrently; W7 priced an uncontended
# pass at ~1.5-2 h for all 32. Waiting is therefore faster in wall-clock than racing, and it keeps
# D: responsive for the training runs that also read it.
#
# TRIGGER. W3's chain writes `CHAIN_DONE` as the last line of its own log after `bank_finalize`.
# ⛔ The trigger is that ARTIFACT, never a process check: `CLAUDE.md` records that `pgrep -f` style
# matching self-matches and that an exit code can be overwritten by a pipeline. We grep a marker
# that the chain writes only on success, and we emit an OPAQUE token of our own so this script can
# never match its own echoed command line (the documented monitor-echo trap).
#
# SAFETY. Nothing here touches the live A1 run, W7's finisher, or the navsim data tree: the
# extractor writes ONLY to its own staging root. It is resumable and content-asserting, so an
# interrupted pass costs nothing and a rerun is free.
set -u

REPO="D:/Projects/TanitAD"
PKG="$REPO/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-20-navhard-stage1-extract"
W3LOG="$REPO/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/chain_w3/chain.log"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
INPUTS="C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json"
ARCH="D:/Archive/devbox-C/navsim/data/openscene-v1.1/openscene_sensor_test_camera"
DEST="C:/Users/Admin/tanitad-caches/navhard-stage1-frames-20260920"
LOG="$PKG/raw/extract_waiter.log"
mkdir -p "$PKG/raw/receipts"

say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

avail_mb() {
  powershell.exe -NoProfile -Command \
    "[int](Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes" 2>/dev/null \
    | tr -cd '0-9'
}

# ⛔ RE-GATED 2026-09-20T19:51Z. The original trigger was W3's chain completion. That was WRONG, and
# the reason is a priority one, not a technical one: W7's scoring retry produces the ACTUAL EPDMS
# number, while this extraction only ENABLES a better one later. MEASURED at the moment of the
# change: all four of W7's scoring arms had just been killed by `E1_RAM_GUARD_ABORT` at **1,035 MB
# available** (the scorer's OWN RSS was only 768 MB -- it was a victim, not the consumer), RAM was
# **3,574 MB**, W7's retry gates on >= 6 GB sustained, and W3 was at shard 30/32 -- i.e. this waiter
# was MINUTES from firing into exactly the contention that had just destroyed 3.12 h of paid
# inference. ⭐ A trigger must encode PRIORITY, not merely availability.
W7PKG="$REPO/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-20-navhard-refcv4b"
SUCCESS="$W7PKG/FINISH_REPORT.md"   # written ONLY when >= 1 arm reads status OK (W7's content gate)

say "waiter armed (pid $$); gated on W7 SUCCESS artifact $SUCCESS (NOT on W3 completion)"

for i in $(seq 1 720); do                     # up to 12 h
  [ -f "$SUCCESS" ] && { say "W7 success artifact present"; break; }
  sleep 60
done

if [ ! -f "$SUCCESS" ]; then
  say "W7 success artifact never appeared in 12 h -- NOT starting. ⛔ Refusing to consume the box while the primary result is still missing; re-arm by hand."
  exit 6
fi

# ⭐ Sustained headroom, not one lucky sample -- the same discipline W7's retry adopted after a
# single-sample reading let a starved box look healthy.
say "waiting for sustained RAM >= 6000 MB (5 consecutive 60 s samples)"
ok=0
for i in $(seq 1 720); do
  A=$(avail_mb)
  if [ -n "$A" ] && [ "$A" -ge 6000 ]; then ok=$((ok+1)); else ok=0; fi
  [ "$ok" -ge 5 ] && break
  sleep 60
done
if [ "$ok" -lt 5 ]; then
  say "never saw 5 consecutive samples >= 6000 MB in 12 h -- NOT starting"
  exit 7
fi

A=$(avail_mb)
say "starting extraction; avail_mb=${A:-unknown}"
t0=$(date +%s)
"$PY" "$PKG/code/extract_navhard_stage1.py" \
  --inputs "$INPUTS" --archives "$ARCH" --dest "$DEST" \
  --receipts "$PKG/raw/receipts" --shards 0-31 >> "$PKG/raw/extract_run.log" 2>&1
RC=$?
say "extraction finished rc=$RC wall_s=$(( $(date +%s) - t0 ))"

# ⭐ The admissible evidence is the VERIFY VERDICT, not the exit code above. Write it to its own
# file so it is an artifact that can be read later, not a status that scrolled past.
"$PY" "$PKG/code/extract_navhard_stage1.py" \
  --inputs "$INPUTS" --dest "$DEST" --archives "$ARCH" \
  --receipts "$PKG/raw/receipts" --verify > "$PKG/raw/VERIFY.txt" 2>&1
VRC=$?
say "verify rc=$VRC verdict=$(head -1 "$PKG/raw/VERIFY.txt" 2>/dev/null)"
say "waiter done"
