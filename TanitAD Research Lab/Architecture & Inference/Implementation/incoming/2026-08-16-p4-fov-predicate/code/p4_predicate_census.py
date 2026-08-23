"""P4/FOV predicate census — is the join's `occ` flag the SAME test as `fov_mask`?

Pure geometry. No corpus, no checkpoint, no GPU, no join file. Reproduces
`raw/p4_predicate_identity.json` bit-identically.

Run::

    PYTHONUTF8=1 python p4_predicate_census.py --out ../raw/p4_predicate_identity.json

Five questions, each answered by construction rather than by reading a writeup:

  A. Are ``build_obstacle_join.visibility_occ`` and ``bev_raster.fov_mask`` the
     same predicate?  Feed the 7 680 CELL CENTRES to the AGENT-space function and
     compare, elementwise, at several half-angles.
  B. Does the DEFAULT half-angle agree BIT-EXACTLY? ``math.radians(60.0)`` vs
     ``math.radians(120.0) / 2.0`` are two different float expressions.
  C. Where does the out-of-field set live?  (The P4 "occluded" population IS this
     set, so its geometry is the population's geometry.)
  D. What would an ``_infov`` twin do to the P4 occluded arm?  Measured on the
     REAL ``rasterize`` with real footprint extents, not argued.
  E. 120 deg (join default, the SENSOR) vs 117 deg (the v5f sub-frame, what the
     ENCODER was fed): how many cells, and how much AGENT AZIMUTH, disagree.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[5]                       # …/TanitAD
_STACK = _REPO / "stack"
for _p in (str(_STACK), str(_STACK / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.data.bev_raster import (BEVGrid, GRID_DEFAULT,  # noqa: E402
                                     cell_azimuth_rad, cell_centers_xy,
                                     fov_mask, rasterize)
import build_obstacle_join as boj                                     # noqa: E402

#: the two frames that actually matter here (P8_V6_PORT.md §1.3).
HFOV_SENSOR = 120.0            # camera_front_wide_120fov — the join's default
HFOV_V5F_SUBFRAME = 117.0      # centred_subframe(…, 176, 624) — what the encoder saw
HFOV_V6F = 120.0               # v6F trains at the full 256x640


# --------------------------------------------------------------------------- A
def predicate_identity(grid: BEVGrid = GRID_DEFAULT) -> dict:
    """Run the AGENT-space `occ` test over the CELL centres and compare."""
    X, Y = cell_centers_xy(grid)
    # every cell centre, presented as a zero-extent agent row (cx, cy, yaw, l, w)
    pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size),
                    np.zeros(X.size), np.zeros(X.size)], axis=1)
    rows = []
    for hfov in (30.0, 60.0, 90.0, 117.0, 120.0, 150.0, 179.0):
        occ = boj.visibility_occ(pts, hfov_deg=hfov).reshape(X.shape)
        msk = fov_mask(grid, math.radians(hfov) / 2.0)
        rows.append({
            "hfov_deg": hfov,
            "n_cells": int(X.size),
            "n_disagree": int(np.count_nonzero((occ == 0) != msk)),
            "identical": bool(np.array_equal(occ == 0, msk)),
            "n_in_field": int(np.count_nonzero(msk)),
        })
    return {"_question": "is `occ == 0` the same set as `fov_mask == True`?",
            "occ_source": "stack/scripts/build_obstacle_join.py:210-223 "
                          "visibility_occ -> np.abs(arctan2(cy, cx)) <= "
                          "radians(hfov_deg)/2",
            "fov_source": "stack/tanitad/data/bev_raster.py:334-355 fov_mask -> "
                          "np.abs(cell_azimuth_rad) <= half_angle_rad, with "
                          "cell_azimuth_rad = arctan2(y, x) (:323-331)",
            "per_hfov": rows,
            "all_identical": all(r["identical"] for r in rows)}


# --------------------------------------------------------------------------- B
def default_half_angle_bit_equality() -> dict:
    """``fov_mask``'s default vs ``visibility_occ``'s default, as raw floats."""
    a = math.radians(60.0)                      # bev_raster.fov_mask default
    b = math.radians(120.0) / 2.0               # visibility_occ at HFOV_DEG_DEFAULT
    return {"_question": "do the two DEFAULTS agree bit-exactly, or only to "
                         "within rounding?",
            "fov_mask_default_rad": repr(a),
            "visibility_occ_default_rad": repr(b),
            "hex_fov_mask": a.hex(),
            "hex_visibility_occ": b.hex(),
            "bit_identical": a == b,
            "ulp_gap": abs(a - b)}


# --------------------------------------------------------------------------- C
def out_of_field_geometry(grid: BEVGrid = GRID_DEFAULT,
                          hfov_deg: float = HFOV_SENSOR) -> dict:
    """Where the P4 'occluded' population lives, since it IS the masked-out set."""
    X, Y = cell_centers_xy(grid)
    msk = fov_mask(grid, math.radians(hfov_deg) / 2.0)
    out = ~msk
    n = int(np.count_nonzero(out))
    return {"hfov_deg": hfov_deg,
            "n_out_of_field": n,
            "n_cells": int(X.size),
            "frac_out_of_field": round(n / X.size, 6),
            "max_x_m_of_an_out_cell": float(X[out].max()) if n else None,
            "min_abs_y_m_of_an_out_cell": float(np.abs(Y[out]).min()) if n else None,
            "boundary_rule": "|y| > x * tan(hfov/2)  =>  an out-of-field cell "
                             "exists only where x < y_half_m / tan(hfov/2)",
            "x_ceiling_m": round(grid.y_half_m / math.tan(
                math.radians(hfov_deg) / 2.0), 4),
            "_read": "the P4 occluded arm is scored ENTIRELY inside this wedge: "
                     "near, off-axis, and 8 % of the grid. The visible arm is "
                     "scored on the other 92 %. The two arms are therefore "
                     "DISJOINT REGIONS of different size and different range."}


# --------------------------------------------------------------------------- D
def what_an_infov_twin_would_do(grid: BEVGrid = GRID_DEFAULT,
                                hfov_deg: float = HFOV_SENSOR) -> dict:
    """MEASURED on the real rasteriser: mask the OCCLUDED subset raster to the
    field and count what is left.

    An occ==1 agent has its CENTRE outside the field. Its footprint cells that
    survive an in-field mask are the sliver straddling the +-hfov/2 ray.

    ⚠️ SAMPLING NOTE, and it corrected an over-claim of mine. Sampling agent
    centres UNIFORMLY OVER THE WHOLE GRID yields only ~40 occluded agents per
    4 000 draws — enough to report "0 survivors" for a sub-cell agent and be
    WRONG. Every occluded agent on this grid lives at ``x < y_half/tan(th)``, so
    the sweep samples that wedge directly and draws 20 000. The corrected
    sub-cell number is NOT zero: cell-centre quantisation alone moves a hit up
    to half a cell diagonal, which near the ego origin is enough to cross the
    azimuth boundary.
    """
    half = math.radians(hfov_deg) / 2.0
    msk = fov_mask(grid, half)
    x_ceiling = grid.y_half_m / math.tan(half)
    res = {"_sampling": {
        "agent_centres": "uniform over the OUT-OF-FIELD WEDGE "
                         f"(0 < x < {x_ceiling:.4f} m, |y| <= "
                         f"{grid.y_half_m} m), rejection-tested against the "
                         "occ predicate — not uniform over the whole grid",
        "n_draws": 20000,
        "yaw": "uniform on (-pi, pi]"}}
    for tag, (ln, wd) in (("subcell_agent_0p05", (0.05, 0.05)),
                          ("subcell_agent_0p45", (0.45, 0.45)),
                          ("automobile_4p5x2p0", (4.5, 2.0)),
                          ("heavy_truck_12x2p6", (12.0, 2.6))):
        rng = np.random.default_rng(20260816)   # same centres for every extent
        kept_cells = kept_windows = n_windows = tot_cells = 0
        surv = []
        for _ in range(20000):
            cx = float(rng.uniform(0.0, x_ceiling))
            cy = float(rng.uniform(-grid.y_half_m, grid.y_half_m))
            yaw = float(rng.uniform(-math.pi, math.pi))
            if abs(math.atan2(cy, cx)) <= half:
                continue                        # occ == 0, the VISIBLE arm
            r = rasterize([{"cx": cx, "cy": cy, "yaw": yaw, "l": ln,
                            "w": wd, "occ": 1}], grid=grid) > 0.5
            n_all = int(np.count_nonzero(r))
            if n_all == 0:
                continue                        # footprint missed every centre
            n_win = int(np.count_nonzero(r & msk))
            n_windows += 1
            tot_cells += n_all
            kept_cells += n_win
            kept_windows += int(n_win > 0)
            surv.append(n_win / n_all)
        res[tag] = {
            "agent_l_w_m": [ln, wd],
            "n_occluded_agents_sampled": n_windows,
            "cells_before_mask": tot_cells,
            "cells_after_mask": kept_cells,
            "cell_survival_frac": round(kept_cells / tot_cells, 6) if tot_cells else None,
            "agents_with_ANY_cell_left": kept_windows,
            "agents_emptied_by_the_mask": n_windows - kept_windows,
            "agent_emptied_frac": round(1.0 - kept_windows / n_windows, 6)
            if n_windows else None,
            "mean_per_agent_survival": round(float(np.mean(surv)), 6) if surv else None,
        }
    res["_read"] = (
        "cell_recall (train_p8_occupancy.py) returns NaN when the subset raster "
        "is empty, and _mean_n drops NaN — so an emptied agent drops out of n "
        "silently. An `_infov` twin on the P4 path therefore does not 'correct' "
        "the occluded arm; it DELETES most of it and rescores the remainder on "
        "footprint slivers that straddle the +-hfov/2 ray — a population "
        "ordered by agent EXTENT, not by agent visibility. The survival "
        "fraction rising monotonically with vehicle length IS that re-selection, "
        "measured.")
    res["_the_residual_is_not_zero"] = (
        "a sub-cell agent is not exactly emptied: the raster tests CELL CENTRES, "
        "so a hit can sit up to half a cell diagonal (0.354 m) from the agent "
        "centre, which near the ego origin crosses the azimuth boundary. That "
        "residual IS the agent-centre vs cell-centre granularity gap named in "
        "the stamp — it is small and it is real, and quoting it as 0 would be "
        "the same over-claim this census exists to prevent.")
    res["_evidence_class"] = ("MEASURED (ours) on the real bev_raster.rasterize; "
                              "the AGENT POSITION PRIOR is uniform-on-wedge "
                              "(synthetic), not the corpus prior, which needs the "
                              "pod-side join file.")
    return res


# --------------------------------------------------------------------------- E
def sensor_vs_encoder_frame(grid: BEVGrid = GRID_DEFAULT) -> dict:
    """120 deg (sensor / join default) vs 117 deg (the v5f sub-frame the encoder
    was actually fed). Cells AND agent-azimuth, because the join flags AGENTS."""
    m120 = fov_mask(grid, math.radians(HFOV_SENSOR) / 2.0)
    m117 = fov_mask(grid, math.radians(HFOV_V5F_SUBFRAME) / 2.0)
    X, Y = cell_centers_xy(grid)
    dis = m120 & ~m117                      # in-field at 120, out at 117
    n = int(np.count_nonzero(dis))
    az = np.degrees(np.abs(cell_azimuth_rad(grid)))
    return {
        "n_out_120": int(np.count_nonzero(~m120)),
        "n_out_117": int(np.count_nonzero(~m117)),
        "n_cells_disagreeing": n,
        "frac_of_grid": round(n / X.size, 6),
        "disagreeing_cells_are_a_superset_only": bool(
            np.count_nonzero(m117 & ~m120) == 0),
        "disagree_band_deg": [HFOV_V5F_SUBFRAME / 2.0, HFOV_SENSOR / 2.0],
        "disagree_cell_x_range_m": [float(X[dis].min()), float(X[dis].max())]
        if n else None,
        "disagree_cell_abs_y_range_m": [float(np.abs(Y[dis]).min()),
                                        float(np.abs(Y[dis]).max())] if n else None,
        "cell_azimuth_max_deg": round(float(az.max()), 4),
        "_the_population_mismatch": (
            "⛔ THE 36 CELLS ARE THE WRONG DENOMINATOR FOR THE JOIN. `fov_mask` "
            "grades CELLS; `visibility_occ` grades AGENT CENTRES. The join-side "
            "n is the number of AGENT ROWS whose centre azimuth falls in the "
            "[58.5, 60.0] deg annulus — flagged `visible` by the join, invisible "
            "to the encoder. That n is a property of the corpus, is NOT derivable "
            "from the grid, and needs the pod-side join file."),
        "_direction_of_the_bias": (
            "agents in the annulus are mislabelled VISIBLE while the encoder "
            "never saw them. They are therefore occluded-like rows sitting in "
            "the visible bucket. Since the published result is "
            "recall_occluded (0.2178) > recall_visible (0.1881), the "
            "contamination can only RAISE the visible arm and SHRINK the gap "
            "=> correcting 120 -> 117 is CONSERVATIVE for the P4 claim."),
        "lf0_corridor_overlap": _lf0_corridor_overlap(grid, dis),
    }


def _lf0_corridor_overlap(grid: BEVGrid, dis: np.ndarray) -> dict:
    """Do any of the 36 disagreeing cells lie in the set LF0 actually scans?

    LF0 walks a +-half_m ego corridor from ``--min-row`` (default 2) forward
    (BEV_CONSUMER_AUDIT.md §2.1). If the answer is zero at its run
    configuration, the 120/117 disagreement cannot move the LF0 verdict either.
    """
    _X, Y = cell_centers_xy(grid)
    nx, _ny = grid.shape
    rows = np.arange(nx)[:, None]
    out = {}
    for half_m in (1.0, 1.5, 2.0):
        for min_row in (0, 2):
            sel = (np.abs(Y) <= half_m + 0.25) & (rows >= min_row)
            out[f"pm{half_m}m_minrow{min_row}"] = {
                "n_corridor_cells": int(np.count_nonzero(sel)),
                "n_disagreeing_in_corridor": int(np.count_nonzero(sel & dis))}
    out["_read"] = ("zero at LF0's run configuration (min_row 2) => the "
                    "120/117 disagreement cannot move the banked LF0 verdict, "
                    "for the same reason the whole-field mask cannot.")
    return out


# --------------------------------------------------------------------------- F
def banked_p4_reread(repo: Path = _REPO) -> dict:
    """Re-read the BANKED P4 split across all four k — not just the quoted k=10.

    The published sentence quotes k=10 only. The artifact carries k=5/10/15/20.
    Two questions a permanence claim must survive: does the ordering hold at
    every k, and does the occluded arm DECAY with k (memory) or stay FLAT
    (a regional prior)?
    """
    base = (repo / "TanitAD Research Hub" / "Architecture & Inference"
            / "Implementation" / "incoming" / "2026-08-07-hierarchical-wm-redesign")
    out = {}
    for tag in ("1", "2"):
        p = base / f"p8_gate_attempt{tag}.json"
        if not p.exists():
            out[f"attempt{tag}"] = {"error": "artifact absent"}
            continue
        sp = json.loads(p.read_text(encoding="utf-8"))["mini_eval"][
            "visible_occluded_split"]
        ks = sorted((k for k in sp if k.isdigit()), key=int)
        rows = {}
        for k in ks:
            r = sp[k]
            rows[k] = {
                "enc_occ": r["recall_occluded_enc"],
                "enc_vis": r["recall_visible_enc"],
                "enc_occ_minus_vis": round(r["recall_occluded_enc"]
                                           - r["recall_visible_enc"], 6),
                "pred_occ": r["recall_occluded_pred"],
                "pred_vis": r["recall_visible_pred"],
                "pred_occ_minus_vis": round(r["recall_occluded_pred"]
                                            - r["recall_visible_pred"], 6),
                "n_occ": r["n_occluded_enc"], "n_vis": r["n_visible_enc"]}
        enc_d = [rows[k]["enc_occ_minus_vis"] for k in ks]
        pred_d = [rows[k]["pred_occ_minus_vis"] for k in ks]
        occ_series = [rows[k]["enc_occ"] for k in ks]
        out[f"attempt{tag}"] = {
            "ks": ks, "per_k": rows,
            "enc_occ_gt_vis_at_all_k": all(d > 0 for d in enc_d),
            "pred_occ_gt_vis_at_all_k": all(d > 0 for d in pred_d),
            "pred_k_with_positive_gap": [k for k, d in zip(ks, pred_d) if d > 0],
            "enc_occ_recall_range": [min(occ_series), max(occ_series)],
            "enc_occ_recall_spread": round(max(occ_series) - min(occ_series), 6),
            "enc_occ_recall_monotone_in_k": bool(
                all(b <= a for a, b in zip(occ_series, occ_series[1:]))
                or all(b >= a for a, b in zip(occ_series, occ_series[1:]))),
        }
    out["_read"] = (
        "attempt 2 (the published run): the ENC ordering occluded > visible "
        "holds at all four k, but the PRED ordering — the half the registry "
        "also quotes — holds at k=10 ONLY and reverses at k=5/15/20. And the "
        "occluded recall does NOT decay with k, which is what a memory claim "
        "predicts and what a fixed regional prior does not. attempt 1's "
        "decoder had collapsed to ~0 recall on BOTH arms, so it is silent on "
        "this question rather than a control.")
    out["_evidence_class"] = ("MEASURED (ours; artifacts = p8_gate_attempt"
                              "{1,2}.json, re-read, no re-run)")
    return out


def build(grid: BEVGrid = GRID_DEFAULT) -> dict:
    return {
        "_task": "P4 permanence vs the FOV mask — are they the same predicate?",
        "_date": "2026-08-16",
        "_evidence_class": "MEASURED (ours) — pure geometry, no corpus / "
                           "checkpoint / GPU / join file",
        "grid": {"x_fwd_m": grid.x_fwd_m, "y_half_m": grid.y_half_m,
                 "cell_m": grid.cell_m, "shape": list(grid.shape)},
        "A_predicate_identity": predicate_identity(grid),
        "B_default_half_angle": default_half_angle_bit_equality(),
        "C_out_of_field_geometry": out_of_field_geometry(grid),
        "D_infov_twin_would": what_an_infov_twin_would_do(grid),
        "E_sensor_vs_encoder_frame": sensor_vs_encoder_frame(grid),
        "F_banked_p4_reread": banked_p4_reread(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("p4_predicate_census")
    ap.add_argument("--out", default=str(_HERE.parent / "raw"
                                         / "p4_predicate_identity.json"))
    a = ap.parse_args(argv)
    d = build()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d, indent=1, ensure_ascii=False),
                           encoding="utf-8")
    print(json.dumps(d, indent=1, ensure_ascii=False))
    print(f"\n[p4] -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
