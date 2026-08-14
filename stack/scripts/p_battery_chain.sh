#!/bin/bash
# P-BATTERY on a v6 checkpoint — the frozen-latent interpretation heads that
# decide whether S-W's world model may propagate upward (X5).
#
# ⛔⛔ MEASURED 2026-08-14, pod4: THIS CHAIN CANNOT YET RUN ON A v6 CHECKPOINT.
# The probes are built for the v5 `WorldModel`, not merely for the v5 checkpoint
# FORMAT, and the failure walks forward one layer at a time as each surface
# assumption is fixed:
#
#   run 1  ModuleNotFoundError: train_p8_occupancy / train_v58f_unicycle_head
#          -> shipped; these are sibling imports of the probes.
#   run 2  KeyError: 'model'
#          -> v6 writes {"stack": sd, "opt", "step", "config"}; v5 wrote
#             {"model": ...}. Fixed properly in ckpt_compat.state_dict_of and
#             replay.arms.load_checkpoint_state (+ test_v6_ckpt_layout_compat).
#   run 3  KeyError: 'model' again, from a THIRD site — a bare `sd = ck["model"]`
#          in a v5-era script (e.g. eval_flagship_v4.py:332). Many such sites
#          exist; they must all route through ckpt_compat.state_dict_of.
#   run 4  (against the fp16 snapshot, which IS in v5 `model`-key layout)
#          KeyError: 'predictor.act_emb.0.weight'
#          -> `ckpt_compat._ACT_KEY`, a v5 WorldModel PARAMETER NAME. v6's
#             module tree has no such key (it is `predictor_op.*`).
#
# ⇒ **The blocker is architectural, not a path or a key.** Run 4 is the one that
# settles it: even a checkpoint in perfect v5 layout fails, because the probe
# builds a v5 WorldModel and infers action_dim from v5 parameter names. Porting
# the P-battery to v6 means giving the probes a V6Stack construction path and a
# v6 action-dim source — real work, not a shim.
#
# ⚠️ DO NOT "fix" this by relaxing the load to strict=False. That would leave the
# probe tensors random-initialised and produce NUMBERS THAT LOOK LIKE RESULTS —
# the exact failure ckpt_compat's own docstring was written to prevent.
#
# Until that port lands, this chain is correct and useful for **v5-era**
# checkpoints only, and the v6 X5 gate has NO instrument. That gap is the point:
# S-W is training now, and the gate meant to decide whether its world model may
# propagate upward cannot currently read it.
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
