#!/usr/bin/env python3
"""Compact the pod's R1-planner sweep dump into the repo-sized per-window dump.

Same construction as Rung 1's `tb_rung1_compact.py`: what every bar, horizon,
stratification and interval in `RUNG1_PLANNER_AND_CONTROLS.md` reads is the dense
per-window `de` [599, 185] per arm plus the two comparator-free floors, and — for
the planner arms only — the dense `fed_actions`, because the mechanism verdict is
an ACTION-amplitude statement and cannot be reconstructed from `psi`/`pred_speed`
(the kinematic reconstruction does not apply to a controller-produced action).

Everything in the report therefore recomputes from this file with NO GPU.

Usage:
    python tb_r1_planner_compact.py --dump perwindow/r1planner_perwindow_K185.pt \
        --out perwindow/r1planner_compact_K185.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

KEEP_FED = ("a_planner", "a_planner_vdec", "a_planner_gtlook",
            "a_imagination__own__roSTR", "a_imagination__hold__roSTR")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = torch.load(a.dump, map_location="cpu", weights_only=False)
    gt = d["gt"]
    de = {k: torch.linalg.norm(v - gt, dim=-1).float()
          for k, v in d["pred"].items()}
    de["d_constant_velocity"] = torch.linalg.norm(d["cv"] - gt, dim=-1).float()
    de["d2_hold_v0"] = torch.linalg.norm(d["hold_v0"] - gt, dim=-1).float()

    own = d["pred"]["a_imagination__own__roSTR"]
    hold = d["pred"]["a_imagination__hold__roSTR"]
    selftest = {}
    for k, v in d["pred"].items():
        if k.startswith("a_planner"):
            selftest[k] = {
                "max_abs_diff_vs_own_m": round(float((v - own).abs().max()), 6),
                "max_abs_diff_vs_hold_m": round(float((v - hold).abs().max()), 6)}
    selftest["ANTI_NOOP_PASS"] = bool(
        all(x["max_abs_diff_vs_own_m"] > 1e-6 and x["max_abs_diff_vs_hold_m"] > 1e-6
            for x in selftest.values() if isinstance(x, dict)))

    out = {"dense_de": de,
           "fed_actions": {k: v.float() for k, v in d.get("fed_actions", {}).items()
                           if k in KEEP_FED},
           "psi": d["psi"], "pred_speed": d["pred_speed"],
           "v_last": d["speed"], "head_deg": d["head_deg"],
           "eid": d["eid"], "t0": d["t0"], "ep_i": d["ep_i"],
           "selftest": selftest, "meta": d["meta"],
           "note": ("dense per-window de [N,K] for every R1-planner arm plus the "
                    "two comparator-free floors, and the dense fed (steer, accel, "
                    "v0) for the 5 arms whose ACTION amplitude sets the mechanism "
                    "verdict. Any bar, horizon or stratification recomputes from "
                    "this file with no GPU.")}
    torch.save(out, a.out)
    print(json.dumps(selftest, indent=1))
    print(f"[compact] {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB, "
          f"{len(de)} de arms, {len(out['fed_actions'])} fed arms)")
    return 0 if selftest["ANTI_NOOP_PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
