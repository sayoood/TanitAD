"""ANALYTIC CIRCLE TEST for taniteval.four_families._seq_geometry.

DECIDES BY CONSTRUCTION, not by inspection. A circular arc of radius R has
curvature EXACTLY 1/R everywhere and yaw-rate EXACTLY v/R rad/s. Feed such an
arc, sampled on the REAL v3 horizon set, through the REAL metric, and compare
the recovered value to the analytic one.

Discriminating control: the SAME arc on a UNIFORM horizon. If it recovers 1/R
there and fails on the non-uniform set, the mechanism is PROVEN, not inferred.
Null: a straight line, true curvature exactly 0.

ZERO GPU. CPU tensors only.
"""
from __future__ import annotations
import json, math, os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import four_families as ff

TICK = 0.1                                   # refc_v3.py:121 -- DT=0.1, 10 Hz
V3_HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)  # refc_v3.py:121 -- STEPS


def arc(R, v, times, dtype=torch.float64):
    """Circular arc from the origin, tangent +x, curving LEFT.
    x = R sin(vt/R), y = R(1-cos(vt/R)).  True curvature = +1/R EXACTLY."""
    t = torch.as_tensor(times, dtype=dtype)
    th = v * t / R
    return torch.stack([R * torch.sin(th), R * (1 - torch.cos(th))], -1).unsqueeze(0)


def line(v, times, dtype=torch.float64):
    t = torch.as_tensor(times, dtype=dtype)
    return torch.stack([v * t, torch.zeros_like(t)], -1).unsqueeze(0)


def probe(wp, dt, R, v, label, times):
    g = ff._seq_geometry(wp, dt)
    k = g["curvature"][0]
    pv = g["pair_valid"][0]
    yr = g["yaw_rate"][0]
    sp = g["speed"][0]
    true_k = float("inf") if R is None else 1.0 / R
    out = {
        "label": label,
        "dt_passed": dt,
        "n_slots": int(wp.shape[1]),
        "slot_times_s": [round(float(x), 4) for x in times],
        "slot_gaps_s": [round(float(times[i + 1] - times[i]), 4) for i in range(len(times) - 1)],
        "first_gap_vs_origin_s": round(float(times[0]), 4),
        "true_curvature_1pm": None if R is None else true_k,
        "recovered_curvature_per_pair": [round(float(x), 8) for x in k],
        "recovered_curvature_mean": round(float(k[pv].mean()) if pv.any() else float("nan"), 8),
        "curv_ratio_recovered_over_true": (
            None if R is None else round(float(k[pv].mean()) / true_k, 6)),
        "true_yaw_rate_rad_s": None if R is None else v / R,
        "recovered_yaw_rate_mean": round(float(yr.mean()), 8),
        "yaw_ratio": (None if R is None else round(float(yr.mean()) / (v / R), 6)),
        "true_speed_mps": v,
        "recovered_speed_mean": round(float(sp.mean()), 6),
        "speed_ratio": round(float(sp.mean()) / v, 6),
        "n_pair_valid": int(pv.sum()),
    }
    return out


def main():
    v = 10.0
    results = []

    # ---- slot-time grids -------------------------------------------------
    grids = {
        # THE REAL v3 SET: steps x 0.1 s -> 0.5,1,1.5,2,3,4,5,6  (gaps 0.5x4 then 1.0x3)
        "A_v3_NONUNIFORM": [h * TICK for h in V3_HORIZONS],
        # CONTROL 1: same 8 slots, UNIFORM 0.5 s spacing (the operative stride)
        "B_uniform_0.5s_8slot": [0.5 * (i + 1) for i in range(8)],
        # CONTROL 2: same 6.0 s SPAN, UNIFORM 0.75 s spacing, 8 slots
        "C_uniform_span6s_8slot": [0.75 * (i + 1) for i in range(8)],
        # CONTROL 3: the uniform 2 s PREFIX the gate arm used (4 slots @ 0.5 s)
        "D_uniform_2s_prefix": [0.5 * (i + 1) for i in range(4)],
        # REFERENCE: the DENSE 0.1 s grid the module constant names
        "E_dense_0.1s_60slot": [TICK * (i + 1) for i in range(60)],
    }

    for R in (20.0, 50.0, 200.0, 1000.0):
        for gname, times in grids.items():
            # dt a caller would actually pass: 0.1 (infer_dt's NON-UNIFORM fallback)
            for dt in (0.1, 0.5):
                results.append(probe(arc(R, v, times), dt, R, v,
                                     f"{gname}|R={R}|dt={dt}", times))

    # ---- NULL: straight line, true curvature exactly 0 -------------------
    nulls = []
    for gname, times in grids.items():
        g = ff._seq_geometry(line(v, times), 0.1)
        k = g["curvature"][0]
        nulls.append({"grid": gname, "max_abs_curvature": float(k.abs().max()),
                      "mean_curvature": float(k.mean()),
                      "speed_mean": float(g["speed"][0].mean())})

    # ---- fp32 (production dtype: callers do .float()) --------------------
    fp32 = []
    for R in (50.0,):
        for gname, times in grids.items():
            wp = arc(R, v, times, dtype=torch.float32)
            g = ff._seq_geometry(wp, 0.1)
            k = g["curvature"][0]
            fp32.append({"grid": gname, "R": R, "true": 1.0 / R,
                         "recovered_mean": float(k.mean()),
                         "ratio": float(k.mean()) / (1.0 / R)})

    print(json.dumps({"arc": results, "null_straight_line": nulls, "fp32": fp32},
                     indent=1))


if __name__ == "__main__":
    main()
