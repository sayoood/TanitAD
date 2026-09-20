#!/usr/bin/env python3
"""ECHO control ``ha0_ext`` on NavSim — POST-HOC DIAGNOSTIC (see RESULT.md; NOT a bar). TANITAD VENV.

WHY IT EXISTS. BAR-E2-1 passed, and the decomposition says A1 won on the SAFETY terms
(NC, TTC, DAC) while losing EP and EC — i.e. it drives more cautiously than CV. A model
handed (v0, a_long, curvature) can be "cautious" by ECHOING a measured deceleration, with
no perception at all. The programme's standing answer to that is the echo control
``ha0_ext`` (``stack/tanitad/eval/echo_gate.py``): constant measured a0 AND constant
measured curvature k0, integrated exactly — ``refc_v3.kinematic_goal_extrapolation``,
IMPORTED, never re-derived. It reads EXACTLY A1's declared ego inputs (v0 = hypot(vx, vy),
a0 = ax, k0 = clip(ay / max(v0, 4)^2, +-0.12) — the same ``ego_block`` the model got) and
no pixel, no command. Needing no frames, it also runs the 16 stage-1 scenes, so its
official two-stage EPDMS exists.

Output poses are the function's own (x, y, heading) at 0.5..4.0 s — no spline.
    -> raw/seam_ECHO_ha0_ext.npz  (+ .manifest.json)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tanitad_navsim_bridge as B  # noqa: E402
from tanitad.refs.refc_v3 import kinematic_goal_extrapolation  # noqa: E402

RAW = os.path.join(os.path.dirname(HERE), "raw")


def main() -> int:
    doc = json.load(open(os.path.join(RAW, "navsim_agent_inputs.json"), encoding="utf-8"))
    toks = sorted(doc["tokens"])
    rows, poses = [], []
    for t in toks:
        r = doc["tokens"][t]
        decl = B.declare(r["ego_statuses"], "A3_ego_nocmd")      # ego[t0] only, no command
        eb = B.ego_block(decl)
        v0, a0, _, k0, _ = eb["ego_state"]
        out = kinematic_goal_extrapolation(torch.tensor([v0], dtype=torch.float64),
                                           torch.tensor([a0], dtype=torch.float64),
                                           torch.tensor([k0], dtype=torch.float64),
                                           B.NAVSIM_T_S)[0].numpy()        # [8, 4]
        poses.append(out[:, :3].astype(np.float32))
        rows.append({"token": t, "stage": r["stage"], "v0": v0, "a0": a0, "k0": k0})
    np.savez(os.path.join(RAW, "seam_ECHO_ha0_ext.npz"), token=np.asarray(toks),
             fingerprint=np.asarray([doc["tokens"][t]["fingerprint"] for t in toks]),
             source=np.asarray(["precomputed"] * len(toks)), poses=np.stack(poses),
             knots=np.full((len(toks), 8, 2), np.nan, np.float32),
             sampling=np.asarray([8, 0.5]), arm=np.asarray("ECHO_ha0_ext"))
    B.json_dump({"arm": "ECHO_ha0_ext", "status": "POST-HOC DIAGNOSTIC — not a bar",
                 "declared_inputs": ["ego_velocity[t0]", "ego_acceleration[t0]"],
                 "implementation": "tanitad.refs.refc_v3.kinematic_goal_extrapolation",
                 "n": len(toks), "rows": rows},
                os.path.join(RAW, "seam_ECHO_ha0_ext.manifest.json"))
    print(f"seam_ECHO_ha0_ext: {len(toks)} rows (stage 1 + 2, no stand-ins)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
