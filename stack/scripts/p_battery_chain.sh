#!/bin/bash
# P-BATTERY on a v6 checkpoint — the frozen-latent interpretation heads that
# decide whether S-W's world model may propagate upward (X5).
#
# ⛔ WHY THIS EXISTS. Both E-ENC arms wrote `gate_verdict: INCONCLUSIVE` because
# P1/P3/P6 are computed by EXTERNAL probe scripts and folded in via
# `--gate-probes`, which nothing was supplying. A gate that cannot reach a
# verdict is not a gate — it is a file. This chain produces the JSON that flag
# consumes.
#
# ⭐ IT RUNS ON ANY CHECKPOINT, WHICH IS THE POINT. The battery reads a FROZEN
# trunk, so it does not need the run to finish: at 18.1 s/step a 30 k run is
# ~151 h, and waiting for it to answer "is the physics right?" would be exactly
# the v5 mistake (discovering the action echo only at final eval). Run it at
# 5 k and stop a bad run at hour 25 instead of hour 151.
#
# ⚠️ NEVER run this on the pod that is training — it needs the GPU. Either pause
# the trainer at a checkpoint boundary, or run it on the other pod with the
# checkpoint pulled from HF.
#
# Usage:
#   CKPT=/workspace/experiments/v6F-SW-30k/ckpt.pt \
#   OUT=/workspace/experiments/v6F-pbattery \
#   bash scripts/p_battery_chain.sh
set -u
S="${S:-/workspace/TanitAD/stack}"
PY="${PY:-python3}"
CKPT="${CKPT:?set CKPT=/path/to/ckpt.pt}"
OUT="${OUT:-/workspace/experiments/pbattery}"
VAL="${VAL:-/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl}"
JOIN="${JOIN:-}"
LOG="$OUT/p_battery.log"

export PYTHONPATH="$S"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
mkdir -p "$OUT"
cd "$S" || { echo "PB_EXIT=NO_STACK" >> "$LOG"; exit 1; }

# ---- gate 10: the checkpoint exists and is loadable ----------------------- #
[ -f "$CKPT" ] || { echo "PB_EXIT=NO_CKPT" >> "$LOG"; exit 1; }

# ---- gate 15: real disk, never df ---------------------------------------- #
dd if=/dev/zero of="$OUT/.ddp" bs=1M count=64 >/dev/null 2>&1 || {
  echo "PB_EXIT=DISK" >> "$LOG"; rm -f "$OUT/.ddp"; exit 1; }
rm -f "$OUT/.ddp"

# ---- gate 12: REAL imports + CUDA, before any expensive work -------------- #
# The T1 lesson: greps passing while an import failed burned 11 min/arm and
# then died in analyze().
"$PY" - >> "$LOG" 2>&1 <<'PYEOF'
import sys
try:
    sys.path.insert(0, "scripts")
    import torch
    from tanitad.models.v6 import V6Stack                       # noqa: F401
    a = torch.nn.Conv2d(3, 4, 3).cuda()(torch.randn(1, 3, 8, 8).cuda())
    assert tuple(a.shape) == (1, 4, 6, 6)
    print("PB_IMPORTS_OK")
except Exception as e:                                          # noqa: BLE001
    print(f"PB_IMPORT_FAIL {type(e).__name__}: {e}")
PYEOF
grep -q PB_IMPORTS_OK "$LOG" || { echo "PB_EXIT=IMPORTS" >> "$LOG"; exit 1; }

# ---- P1 / P2 — latent readout retention (incl. lead_gap = perception) ----- #
J=""
[ -n "$JOIN" ] && J="--join-file $JOIN"
"$PY" -u scripts/probe_latent_state.py --ckpt "$CKPT" \
  --v2-val-cache "$VAL" --require-parity --out "$OUT/p1p2" \
  --ks 5,10,15,20 --episodes 40 --stride 8 $J >> "$LOG" 2>&1
echo "PB_P1_EXIT=$?" >> "$LOG"

# ---- P3 / P6 — action-response sign & gain, action-subspace dims ---------- #
"$PY" -u scripts/stage_a_probes.py --ckpt "$CKPT" \
  --v2-val-cache "$VAL" --require-parity --out "$OUT/p3p6" >> "$LOG" 2>&1
echo "PB_P3_EXIT=$?" >> "$LOG"

# ---- fold into ONE --gate-probes JSON ------------------------------------- #
"$PY" - "$OUT" >> "$LOG" 2>&1 <<'PYEOF'
import glob, json, os, sys
out = sys.argv[1]
merged, sources = {}, []
for p in sorted(glob.glob(os.path.join(out, "*", "*.json"))):
    try:
        d = json.load(open(p))
    except Exception as e:                                      # noqa: BLE001
        print(f"skip {p}: {type(e).__name__}: {e}")
        continue
    if isinstance(d, dict):
        merged |= {k: v for k, v in d.items() if not k.startswith("_")}
        sources.append(p)
merged["_sources"] = sources
merged["_note"] = ("folded by p_battery_chain.sh for --gate-probes; the gate "
                   "reads P1/P3/P6 from here")
dst = os.path.join(out, "gate_probes.json")
json.dump(merged, open(dst, "w"), indent=1)
print(f"PB_MERGED n_keys={len(merged)} from {len(sources)} files -> {dst}")
PYEOF
grep -q PB_MERGED "$LOG" || { echo "PB_EXIT=MERGE" >> "$LOG"; exit 1; }
echo "P_BATTERY_DONE" >> "$LOG"
