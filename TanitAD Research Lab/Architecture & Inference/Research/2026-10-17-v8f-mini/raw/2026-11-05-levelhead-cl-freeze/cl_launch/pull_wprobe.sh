#!/usr/bin/env bash
# LEVELHEAD arm CL: pull the W probe's result from box 2 (READ-ONLY on box 2: a few small files) and write cl_prereg/W_PROBE.json from level_head_probe.w_level_for_10pct_share (w_probe_read.py refuses anything it cannot verify).
# REFUSES without CL_PI_GO. Usage: bash pull_wprobe.sh          Run AFTER WPROBE.done exists (wake_cl.sh wprobe). Releases OUR BOX2_BUSY lock (moved aside, never deleted) only on full success and only when the lock text is the probe's.
# Destination: E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s0_wprobe_CL/ and cl_markers/. Test hooks (dev box): CL_DEST, CL_SCP, CL_PY, CL_LOCK, CL_PREREG_DIR. Production runs set none of them.
set -u
: "${CL_PI_GO:?set CL_PI_GO to the PI go time (HH:MM): this script refuses to contact box 2 without the go of the PI}"
CLDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEST=${CL_DEST:-/e/Projects/TanitAD-artifacts/v8f-recover}
R=D:/tanitad/runs/v8f_mini_full
SCP=${CL_SCP:-"scp -q -o BatchMode=yes -o ConnectTimeout=20"}
PY=${CL_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}
LOCK=${CL_LOCK:-/c/Users/Admin/box2_ship/BOX2_BUSY.txt}
PRE=${CL_PREREG_DIR:-$CLDIR/../cl_prereg}
d="$DEST/M2b_s0_wprobe_CL"; mkdir -p "$d" "$DEST/cl_markers/logs" "$DEST/cl_markers/pin"
$SCP -r "tanitad-box2:$R/cl/*" "$DEST/cl_markers/" 2>&1 | grep -v '^\*\*'
for f in selfcheck.json; do $SCP "tanitad-box2:$R/M2b_s0_wprobe_CL/$f" "$d/$f" 2>/dev/null || echo "  (no $f for the probe)"; done
[ -f "$DEST/cl_markers/WPROBE.done" ] && cp "$DEST/cl_markers/WPROBE.done" "$d/WPROBE.done"
[ -f "$DEST/cl_markers/logs/WPROBE_0.gate.log" ] && cp "$DEST/cl_markers/logs/WPROBE_0.gate.log" "$d/WPROBE_0.gate.log"
[ -f "$DEST/cl_markers/WPROBE.FAILED" ] && { echo "WPROBE.FAILED on box 2:"; cat "$DEST/cl_markers/WPROBE.FAILED"; }
$PY -I "$CLDIR/w_probe_read.py" "$(cygpath -m "$d" 2>/dev/null || echo "$d")" --out "$(cygpath -m "$PRE/W_PROBE.json" 2>/dev/null || echo "$PRE/W_PROBE.json")"; rc=$?
if [ "$rc" = 0 ] && [ -f "$LOCK" ] && head -c 33 "$LOCK" | grep -q '^LEVELHEAD CL W probe'; then
  mv "$LOCK" "$(dirname "$LOCK")/BOX2_BUSY.cl_wprobe_done_$(date +%Y%m%d_%H%M%S).txt"; echo "lock released (moved aside)"
elif [ "$rc" = 0 ]; then echo "lock not released: no BOX2_BUSY.txt of the probe at $LOCK"; fi
echo "pull done (rc=$rc) -> $d"; exit $rc
