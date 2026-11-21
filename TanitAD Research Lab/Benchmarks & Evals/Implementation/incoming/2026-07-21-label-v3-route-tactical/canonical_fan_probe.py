"""What the CANONICAL 881 val windows can still be measured on, offline.

The canonical 40-episode val epcache (`physicalai-val-0c5f7dac3b11`) exists only
on the eval pod, and pods were off-limits for this work (pod1 training, pod3
holding the GPU lock). But the full-fan dumps `taniteval/results/fan_refc-*.pt`
carry, for all 881 canonical windows: the GT ego-frame waypoints at 0.5/1/1.5/2 s,
v0, |Δheading@2s|, a_gt, every anchor proposal and the selected index. That is
enough for two canonical measurements:

 (A) RECONSTRUCT the 5-way maneuver label from the GT waypoints and separate
     "a longitudinal decision was PRESENT" from "it SURVIVED the turn>brake>accel
     priority" — the label-side half of the 5-way collapse.
 (B) find the windows where the fan's ORACLE plan is a different LATERAL mode
     from the SELECTED plan — the windows a strategic/lateral goal could have
     disambiguated.

RECONSTRUCTION CAVEAT (do not quote these as pose-derived): the heading comes
from the final polyline tangent and the speed from the leg lengths, not from the
raw poses. Validated against the authoritative pose-derived counts in
`taniteval/results/planfan_clips_tactical_head_val.json`, printed alongside.

  python canonical_fan_probe.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

RES = Path(__file__).resolve().parents[5] / "taniteval" / "results"
CURV_TURN = 1.0 / 60.0                      # refb_labels.CURV_TURN_MAN_PER_M
DV_ACC, DV_BRK, STOP_V, MOVING_V = 1.0, -1.0, 0.3, 1.0
YAW_TURN = 0.15                             # refb_labels.YAW_TURN_RAD
NAMES = ("lane_keep", "turn_left", "turn_right", "accelerate", "brake_stop")
# the authoritative pose-derived gt_v2 row, for the reconstruction check
AUTHORITATIVE_GT_V2 = {"lane_keep": 544, "turn_left": 93, "turn_right": 49,
                       "accelerate": 105, "brake_stop": 90}


def _polyline(wp: torch.Tensor, v0: torch.Tensor):
    """[n,4,2] ego-frame waypoints -> (final tangent heading, arc, v_end)."""
    n = wp.shape[0]
    path = torch.cat([torch.zeros(n, 1, 2), wp], 1)
    d = path[:, 1:] - path[:, :-1]
    legs = d.norm(dim=-1)
    return (torch.atan2(d[:, -1, 1], d[:, -1, 0]), legs.sum(1), legs[:, -1] / 0.5)


def _man5(dyaw, arc, v0, v_end):
    """classify_maneuver_v2's rule, on the reconstructed polyline."""
    kappa = dyaw / arc.clamp_min(0.1)
    dv = v_end - v0
    cls = torch.zeros_like(dyaw, dtype=torch.long)
    acc = dv > DV_ACC
    brake = (dv < DV_BRK) | ((v_end < STOP_V) & (v0 >= MOVING_V))
    cls[acc] = 3
    cls[brake] = 4
    turn = kappa.abs() >= CURV_TURN
    cls[turn & (dyaw > 0)] = 1
    cls[turn & (dyaw < 0)] = 2
    return cls, turn, brake, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    D = {k: torch.load(RES / f"fan_refc-{k}-30k.pt", map_location="cpu",
                       weights_only=False) for k in ("base", "xl")}
    A = D["base"]
    gt, v0, eid = A["gt"], A["v0"], A["eid"]
    n = gt.shape[0]
    dyaw, arc, v_end = _polyline(gt, v0)
    cls, turn, brake, acc = _man5(dyaw, arc, v0, v_end)
    present = brake | acc
    survives = (cls == 3) | (cls == 4)
    destroyed = present & turn

    out = {
        "substrate": {k: str(RES / f"fan_refc-{k}-30k.pt") for k in D},
        "n_windows": n, "n_episodes": len(set(eid)),
        "caveat": "RECONSTRUCTED from the fan's GT waypoints (polyline tangent "
                  "+ leg speeds), NOT from raw poses. The canonical epcache is "
                  "eval-pod-only.",
        "man5_reconstructed": {NAMES[i]: int((cls == i).sum()) for i in range(5)},
        "man5_authoritative_gt_v2": AUTHORITATIVE_GT_V2,
        "longitudinal_present": int(present.sum()),
        "longitudinal_present_pct": round(100 * float(present.float().mean()), 2),
        "longitudinal_survives_priority": int(survives.sum()),
        "longitudinal_survives_pct": round(100 * float(survives.float().mean()), 2),
        "destroyed_by_turn_priority": int(destroyed.sum()),
        "destroyed_pct": round(100 * float(destroyed.float().mean()), 2),
        "lateral_disagreement": {},
    }

    for k, d in D.items():
        fan, sel = d["fan"], d["sel"]
        de = torch.linalg.norm(fan - d["gt"][:, None], dim=-1).mean(-1)
        orc = de.argmin(1)
        ar = torch.arange(fan.shape[0])
        ds, _, _ = _polyline(fan[ar, sel], v0)
        do, _, _ = _polyline(fan[ar, orc], v0)
        ms = torch.where(ds > YAW_TURN, 1, torch.where(ds < -YAW_TURN, 2, 0))
        mo = torch.where(do > YAW_TURN, 1, torch.where(do < -YAW_TURN, 2, 0))
        m = (ms != mo) & ((ds.abs() > YAW_TURN) | (do.abs() > YAW_TURN))
        idx = torch.nonzero(m).flatten().tolist()
        # window -> frame: the dump walks starts = range(0, T-28, 8) per episode
        seen, frame = {}, []
        for e in eid:
            frame.append(8 * seen.get(e, 0) + 7)
            seen[e] = seen.get(e, 0) + 1
        out["lateral_disagreement"][k] = {
            "definition": "the 3-way LATERAL mode (|final tangent heading| > "
                          "0.15 rad) of the SELECTED plan differs from the "
                          "ORACLE-in-fan plan, and at least one of them is a turn",
            "n": len(idx),
            "episodes": sorted({eid[i] for i in idx}),
            "windows": [{"w": i, "ep": eid[i], "frame": frame[i],
                         "v0": round(float(v0[i]), 2),
                         "head_deg": round(float(A["head_deg"][i]), 2),
                         "gt_longitudinal_present": bool(present[i]),
                         "gt_is_turn": bool(turn[i])} for i in idx],
            "gt_longitudinal_present_on_those": int(present[m].sum()),
            "gt_is_turn_on_those": int(turn[m].sum()),
        }

    print(json.dumps({k: v for k, v in out.items()
                      if k != "lateral_disagreement"}, indent=1))
    for k, v in out["lateral_disagreement"].items():
        print(f" {k}: n={v['n']} eps={v['episodes']} "
              f"long-present {v['gt_longitudinal_present_on_those']}/{v['n']} "
              f"turn {v['gt_is_turn_on_those']}/{v['n']}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"-> {a.json}")


if __name__ == "__main__":
    main()
