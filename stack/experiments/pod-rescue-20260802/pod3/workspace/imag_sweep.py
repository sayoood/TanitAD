"""PURE-IMAGINATION DECAY CURVE — pre-registered in PREREG_imagination_horizon.md (eb27a36).

The canary encodes an 8-frame context and then rolls the operative predictor forward in LATENT
SPACE with NO further frames. That is exactly "the camera is not fed". This sweeps how far it can
imagine before the grounded trajectory falls apart.

Actions stay `expert_future` (the canary's own convention) so this isolates WORLD-MODEL fidelity
from action selection — the question is how long the WM can dream, not how well a planner chooses.
"""
from __future__ import annotations
import json, sys, time
sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
import torch

CK = sys.argv[1] if len(sys.argv) > 1 else \
    "/workspace/v4run/hfcache/hub/models--Sayood--tanitad-flagship-4b-speedjerk/snapshots/17902296dee666e27ab60bccafcf889fef0ab0a8/ckpt.pt"
VAL = "/workspace/val40cache"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/imag_sweep.json"
HORIZONS_S = [1.0, 2.0, 4.0, 8.0]          # -> K = 10, 20, 40, 80 at 10 Hz

import eval_flagship_v4 as E
from train_flagship_v4 import canary_rollout

dev = torch.device("cuda")
ck = torch.load(CK, map_location="cpu", weights_only=False)
world, grounding, step = E.load_v1_from_ck(ck, dev)[:3]
print(f"[imag] ckpt step={step}", flush=True)
ds = E.build_val_dataset(VAL) if hasattr(E, "build_val_dataset") else None
if ds is None:                                    # fall back to the harness's own loader
    ds = E.make_val(VAL) if hasattr(E, "make_val") else None
assert ds is not None, "no val-dataset builder found on eval_flagship_v4"

rows = {}
for s in HORIZONS_S:
    K = int(round(s * 10))
    hs = sorted({max(1, K // 4), max(1, K // 2), max(1, 3 * K // 4), K})
    t0 = time.time()
    out = canary_rollout(world, grounding, ds, dev, horizons=hs, k_max=K,
                         episodes=40, stride=8, batch=16, amp=True)
    ade = out.get(f"canary_ade@{s:g}s") or out.get("canary_ade@2s") or out.get("ade")
    rows[f"{s:g}s"] = {"K": K, "horizons": hs, "n": out.get("n"),
                       "raw": {k: v for k, v in out.items() if isinstance(v, (int, float))},
                       "wallclock_s": round(time.time() - t0, 1)}
    print(f"[imag] {s:g}s (K={K}) n={out.get('n')} -> "
          f"{ {k: round(v,4) for k,v in out.items() if isinstance(v,float)} }", flush=True)

json.dump({"ckpt": CK, "val": VAL, "horizons_s": HORIZONS_S,
           "note": "pure imagination: 8-frame context, then latent rollout with NO new frames; "
                   "actions=expert_future so this is WM fidelity, not action selection",
           "rows": rows}, open(OUT, "w"), indent=1)
print(f"-> {OUT}", flush=True)
