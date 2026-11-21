#!/usr/bin/env bash
# LEVELHEAD arm LLR: pull the W_lead probe's result from box 2 (READ-ONLY on box 2: a few small files) and write a2_prereg/WLEAD_PROBE.json from lead_head_probe.w_lead_for_10pct_share (w_lead_probe_read.py refuses anything it cannot verify).
# REFUSES without LLR_PI_GO. Usage: bash pull_wleadprobe.sh      Run AFTER WLPROBE.done exists (wake_llr.sh wprobe). Releases OUR BOX2_BUSY lock (moved aside, never deleted) only on full success and only when the lock text is the probe's.
# Destination: E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s0_wleadprobe_LLR/ and llr_markers/. Test hooks (dev box): LLR_DEST, LLR_SCP, LLR_PY, LLR_LOCK, LLR_PREREG_DIR. Production runs set none of them.
set -u
: "${LLR_PI_GO:?set LLR_PI_GO to the PI go time (HH:MM): this script refuses to contact box 2 without the go of the PI}"
LLDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEST=${LLR_DEST:-/e/Projects/TanitAD-artifacts/v8f-recover}
R=D:/tanitad/runs/v8f_mini_full
SCP=${LLR_SCP:-"scp -q -o BatchMode=yes -o ConnectTimeout=20"}
PY=${LLR_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}
LOCK=${LLR_LOCK:-/c/Users/Admin/box2_ship/BOX2_BUSY.txt}
PRE=${LLR_PREREG_DIR:-$LLDIR/../a2_prereg}
d="$DEST/M2b_s0_wleadprobe_LLR"; mkdir -p "$d" "$DEST/llr_markers/logs" "$DEST/llr_markers/pin"
$SCP -r "tanitad-box2:$R/llr/*" "$DEST/llr_markers/" 2>&1 | grep -v '^\*\*'
for f in selfcheck.json; do $SCP "tanitad-box2:$R/M2b_s0_wleadprobe_LLR/$f" "$d/$f" 2>/dev/null || echo "  (no $f for the probe)"; done
[ -f "$DEST/llr_markers/WLPROBE.done" ] && cp "$DEST/llr_markers/WLPROBE.done" "$d/WLPROBE.done"
[ -f "$DEST/llr_markers/logs/WLPROBE_0.gate.log" ] && cp "$DEST/llr_markers/logs/WLPROBE_0.gate.log" "$d/WLPROBE_0.gate.log"
[ -f "$DEST/llr_markers/WLPROBE.FAILED" ] && { echo "WLPROBE.FAILED on box 2:"; cat "$DEST/llr_markers/WLPROBE.FAILED"; }
$PY -I "$LLDIR/w_lead_probe_read.py" "$(cygpath -m "$d" 2>/dev/null || echo "$d")" --out "$(cygpath -m "$PRE/WLEAD_PROBE.json" 2>/dev/null || echo "$PRE/WLEAD_PROBE.json")" --frozen "$(cygpath -m "$PRE/FROZEN_LLR.txt" 2>/dev/null || echo "$PRE/FROZEN_LLR.txt")"; rc=$?
if [ "$rc" = 0 ] && [ -f "$LOCK" ] && head -c 26 "$LOCK" | grep -q '^LEVELHEAD LLR W_lead probe'; then
  mv "$LOCK" "$(dirname "$LOCK")/BOX2_BUSY.llr_wprobe_done_$(date +%Y%m%d_%H%M%S).txt"; echo "lock released (moved aside)"
elif [ "$rc" = 0 ]; then echo "lock not released: no BOX2_BUSY.txt of the probe at $LOCK"; fi
echo "pull done (rc=$rc) -> $d"; exit $rc
