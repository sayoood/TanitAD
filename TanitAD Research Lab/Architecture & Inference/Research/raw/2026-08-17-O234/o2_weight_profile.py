"""Reproduce O2's per-cell weight profile in PURE PYTHON — no torch, no GPU.

WHY THIS EXISTS. O2's only content that is not already O5's is the DEVIATION of its
weight profile from 1 (see 2026-08-17-O234-DESIGN-RESEARCH.md §2.1). To say how much
content that is, the profile has to be printed. This file is a line-for-line
transliteration of two functions so the number can be audited without a GPU box:

  * ``tanitad.models.v6.readout_grid_ranges``   (v6.py:357)
  * ``tanitad.models.v6.time_to_reach_weights`` (v6.py:324) -> ``time_to_reach`` (v6.py:305)

⛔ It is a REPRODUCTION, not the source of truth. The cross-check that makes it
admissible is in the doc: the live trainer's own log recorded
``o2_w_min 0.1455 / o2_w_max 2.4681`` and ``0.1390 / 1.6884``
(.../2026-08-16-v6-stage-chain/raw/dry_ladder_default.json), both inside the
[0.11, 2.34] envelope printed here.

Run: python o2_weight_profile.py            (writes o2_weight_profile.json beside it)
"""
import json
import math
import pathlib

# --- the live geometry, from source ---------------------------------------
# ReadoutCfg default `grid=4, d_readout=128` (v6.py:2782); grid_w is None => square.
# Confirmed against the probe cache, whose `cells` tensor is [16, 128]
# (PROBE_POSITIVE_CONTROL.md §2.1).
GRID_H, GRID_W = 4, 4
NEAR_M, FAR_M = 3.0, 80.0     # readout_grid_ranges defaults
TAU_S = 2.0                   # --o2-tau-s default (train_v6_staged.py:3752)
HORIZON_S = 6.0               # PLAN_STEPS * DT (v6.py:138)
V_FLOOR = 1.0                 # time_to_reach default


def readout_grid_ranges(grid_h=GRID_H, grid_w=GRID_W, near_m=NEAR_M, far_m=FAR_M):
    """v6.py:357. row 0 = TOP = far ; row grid_h-1 = BOTTOM = near.
    Geometric spacing, because image row maps roughly to inverse depth.
    Columns SHARE a row's range -- there is no lateral depth cue without
    calibration, and the source says so."""
    if grid_h == 1:
        rows = [math.sqrt(near_m * far_m)]
    else:
        frac = [1.0 - i / (grid_h - 1) for i in range(grid_h)]
        rows = [near_m * (far_m / near_m) ** f for f in frac]
    return [r for r in rows for _ in range(grid_w)]     # row-major [grid_h*grid_w]


def time_to_reach_weights(dist_m, v_ego, tau_s=TAU_S, horizon_s=HORIZON_S,
                          v_floor=V_FLOOR, normalize=True):
    """v6.py:324. w = exp(-t_reach / tau), then MEAN-1 normalised over the cell
    axis. ⭐ The normalisation is the whole reason O2 collapses onto O5: a mean-1
    weight RE-ALLOCATES the loss instead of rescaling it, so
    O2 = (O5's step-j term) + Cov_c(w, err), exactly."""
    v = max(float(v_ego), v_floor)
    w = [math.exp(-min(max(abs(d) / v, 0.0), horizon_s) / tau_s) for d in dist_m]
    if normalize:
        m = sum(w) / len(w)
        w = [x / m for x in w]
    return w


def half_weight_distance_m(v_ego, tau_s=TAU_S, v_floor=V_FLOOR):
    """v6.py:348 -- the metre distance at which the weight halves: v * tau * ln2."""
    return max(float(v_ego), v_floor) * tau_s * math.log(2.0)


def main():
    cells = readout_grid_ranges()
    rows = [round(cells[i * GRID_W], 2) for i in range(GRID_H)]
    out = {"grid": [GRID_H, GRID_W], "tau_s": TAU_S, "horizon_s": HORIZON_S,
           "near_m": NEAR_M, "far_m": FAR_M,
           "row_ranges_m_top_to_bottom": rows, "by_speed": {}}
    print(f"row ranges (m), row0=TOP=far -> row{GRID_H-1}=BOTTOM=near: {rows}")
    print()
    print(" v(m/s) | per-row weight (all columns identical) | max/min | half-wt")
    for v in (5, 10, 15, 20, 30):
        w = time_to_reach_weights(cells, v)
        per = [round(w[i * GRID_W], 3) for i in range(GRID_H)]
        ratio, half = max(w) / min(w), half_weight_distance_m(v)
        out["by_speed"][str(v)] = {"per_row_weight": per,
                                   "max_over_min": round(ratio, 3),
                                   "half_weight_distance_m": round(half, 2)}
        print(f"  {v:4d}  | {per}   | {ratio:6.2f}x | {half:5.1f} m")

    # ⭐ the number the recommendation turns on: where the GT lead actually sits.
    # MEASURED lead cx over 2 721 windows (PROBE_POSITIVE_CONTROL.md §3.1).
    out["gt_lead_cx_m"] = {"mean": 15.53, "median": 15.05, "p10": 8.15,
                           "p90": 24.33, "n_windows": 2721,
                           "source": "PROBE_POSITIVE_CONTROL.md 3.1 (MEASURED)"}
    print()
    print("GT lead cx p10-p90 = 8.15-24.33 m  =>  falls between rows at 8.96 m "
          "and 26.78 m, i.e. rows 1-2, weighted 0.771-1.396 at 15 m/s.")
    print("The 1.703x row is the 3.0 m row -- road surface under the ego's nose.")

    p = pathlib.Path(__file__).with_suffix(".json")
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p.name}")


if __name__ == "__main__":
    main()
