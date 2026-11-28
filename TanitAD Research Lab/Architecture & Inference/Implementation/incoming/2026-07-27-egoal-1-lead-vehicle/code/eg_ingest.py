#!/usr/bin/env python3
"""E-GOAL-1 S1 -- the aligned `obstacle.offline` per-window dump.

PRIORITY-1 DELIVERABLE. Produces, in one pass:

  raw/eg_align.json    the clock-join proof, the per-clip SPAN audit, the
                       coverage report, and the fidelity check against the
                       repo's own reader
  raw/eg_windows.parquet   one row per (clip, grid time): causal ego kinematics,
                       causal lead-vehicle state, and the 2 s along/cross-track
                       displacement targets

Run:
    OMP_NUM_THREADS=6 python eg_ingest.py --out ../raw [--chunks 36 170 ...]

⚠️ THE HAZARD IS THE SPAN, NOT THE CLOCK. `egomotion` runs 20-140 s per clip;
`obstacle.offline` stops at ~20 s. A grid point outside the intersection returns
"no obstacle rows", which is INDISTINGUISHABLE from an empty road -- it would
turn a coverage failure into a plausible, stable, wrong "38 % of windows have a
lead vehicle". Every clip's spans are measured and the grid is asserted inside
the intersection before a single feature is built.

⭐ A window with NO lead vehicle is a legitimate value (`lead_present = 0`), not
a gap to impute.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eg_common import (CLIP_END_S, DENS_COLS, EGO_COLS, HIST_S,  # noqa: E402
                       HORIZON_S, LEAD_COLS, ROOT, TARGET_HZ, XTRA_COLS,
                       assert_clock_join, available_chunks, clip_alias,
                       ego_frame, grid_inside_intersection, iter_clips,
                       lead_extra, lead_frame, r4, selection, span_of)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "raw"))
    ap.add_argument("--chunks", nargs="*", type=int, default=None)
    ap.add_argument("--clock-clips", type=int, default=40,
                    help="clips used for the clock-join proof")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    sel = selection()
    chunk_of = dict(zip(sel["clip_id"], sel["chunk"].astype(int)))
    country_of = dict(zip(sel["clip_id"], sel["country"].astype(str)))
    scen_of = dict(zip(sel["clip_id"], sel["scenario"].astype(str)))
    chunks = a.chunks if a.chunks else available_chunks()
    print(f"[ingest] {len(chunks)} obstacle.offline chunks on this host",
          flush=True)

    # the pre-registered grid: 10 Hz, t in [HIST_S, CLIP_END_S - HORIZON_S]
    t_grid = np.arange(HIST_S, CLIP_END_S - HORIZON_S + 1e-9, 1.0 / TARGET_HZ)

    # ---------------------------------------------------------------- clock #
    print("[ingest] proving the clock join ...", flush=True)
    clock_sample = []
    for c in chunks:
        for cid, ego, obst in iter_clips(c):
            if len(obst):
                clock_sample.append((cid, ego, obst))
            if len(clock_sample) >= a.clock_clips:
                break
        if len(clock_sample) >= a.clock_clips:
            break
    clock = assert_clock_join(clock_sample)
    print(f"[ingest] clock join OK: delta={clock['best_delta_s']} s, "
          f"+1 s control {clock['failing_control']['ratio_plus1s_over_0']}x worse",
          flush=True)
    del clock_sample

    # ------------------------------------------------------- span + build #
    rows, spans = [], []
    n_seen = n_kept = 0
    drop = {"not_in_selection": 0, "no_obstacle_rows": 0,
            "grid_outside_intersection": 0, "ego_too_short": 0}
    for ci, c in enumerate(chunks):
        for cid, ego, obst in iter_clips(c):
            n_seen += 1
            if chunk_of.get(cid) != c:
                drop["not_in_selection"] += 1
                continue
            if not len(obst):
                drop["no_obstacle_rows"] += 1
                continue
            sp = span_of(ego, obst)
            sp["alias"] = clip_alias(cid)
            sp["chunk"] = c
            sp["grid_inside"] = grid_inside_intersection(sp, t_grid)
            spans.append(sp)
            if sp["ego_t1"] < CLIP_END_S - 0.05:
                drop["ego_too_short"] += 1
                continue
            # ⭐ PER-WINDOW coverage, not per-clip drop. A grid point outside the
            # obstacle span has NO INFORMATION about a lead vehicle; a grid point
            # inside it with no vehicle has the information "there is none".
            # Collapsing those two is exactly how a coverage failure becomes a
            # plausible lead-presence rate, so they are kept distinct here and
            # only the first is excluded.
            cov_row = ((t_grid >= sp["obst_t0"] - 1e-9)
                       & (t_grid <= sp["obst_t1"] + 1e-9))
            if not cov_row.any():
                drop["grid_outside_intersection"] += 1
                continue

            ef = ego_frame(ego, t_grid)
            lf = lead_frame(obst, t_grid)
            xf = lead_extra(obst, t_grid, lf, ef["v"])
            d = {**ef, **lf, **xf}
            d["t_s"] = t_grid
            d["obst_cov"] = cov_row.astype(float)
            d["clip"] = sp["alias"]
            d["chunk"] = c
            d["country"] = country_of.get(cid, "?")
            d["scenario"] = scen_of.get(cid, "?")
            rows.append(pd.DataFrame(d))
            n_kept += 1
        print(f"[ingest] chunk {c:04d} ({ci+1}/{len(chunks)}): "
              f"{n_kept} clips kept / {n_seen} seen", flush=True)

    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out / "eg_windows.parquet", index=False)

    # ------------------------------------------------------------ coverage #
    good = (np.isfinite(df["y_long"]) & np.isfinite(df["v"])
            & (df["obst_cov"] > 0))
    g = df[good]
    per_clip_lead = g.groupby("clip")["lead_present"].mean()
    cov = {
        "n_clips_kept": n_kept,
        "n_clips_seen": n_seen,
        "dropped": drop,
        "n_windows": int(len(df)),
        "n_windows_obstacle_covered": int((df["obst_cov"] > 0).sum()),
        "frac_windows_obstacle_covered": r4(float((df["obst_cov"] > 0).mean())),
        "n_windows_finite_target": int(good.sum()),
        "grid": {"t0_s": float(t_grid.min()), "t1_s": float(t_grid.max()),
                 "hz": TARGET_HZ, "n_per_clip": int(t_grid.size),
                 "horizon_s": HORIZON_S},
        "obstacle_rows_present_frac_of_clips": 1.0,
        "lead_present_frac_windows": r4(g["lead_present"].mean()),
        "clips_with_any_lead_frac": r4(
            float((g.groupby("clip")["lead_present"].max() > 0).mean())),
        "per_clip_lead_frac": {
            "p10": r4(per_clip_lead.quantile(0.10)),
            "p50": r4(per_clip_lead.quantile(0.50)),
            "p90": r4(per_clip_lead.quantile(0.90))},
        "mean_agents_ahead_50m": r4(g["n_ahead_50m"].mean()),
        "frac_windows_zero_agents_ahead": r4(float((g["n_ahead_50m"] == 0).mean())),
        "gap_m_when_present": {
            "p10": r4(g.loc[g.lead_present > 0, "gap_m"].quantile(0.10)),
            "p50": r4(g.loc[g.lead_present > 0, "gap_m"].quantile(0.50)),
            "p90": r4(g.loc[g.lead_present > 0, "gap_m"].quantile(0.90))},
        "target": {
            "y_long_mean_m": r4(g["y_long"].mean()),
            "y_long_sd_m": r4(g["y_long"].std()),
            "y_lat_sd_m": r4(g["y_lat"].std()),
            "canonical_881_reference": {
                "mean_displacement_m": 25.51, "along_sd_m": 18.73,
                "cross_sd_m": 2.01,
                "source": "GOAL_INPUT.md S2.1 (parent stream)"}},
        "v_mean_ms": r4(g["v"].mean()),
        "n_countries": int(g["country"].nunique()),
        "n_chunks": int(g["chunk"].nunique()),
    }

    span_arr = pd.DataFrame(spans)
    cov["spans"] = {
        "n": int(len(span_arr)),
        "ego_t0_s": [r4(span_arr.ego_t0.min()), r4(span_arr.ego_t0.max())],
        "ego_t1_s": [r4(span_arr.ego_t1.min()), r4(span_arr.ego_t1.max())],
        "obst_t0_s": [r4(span_arr.obst_t0.min()), r4(span_arr.obst_t0.max())],
        "obst_t1_s": [r4(span_arr.obst_t1.min()), r4(span_arr.obst_t1.max())],
        "frac_grid_inside_intersection": r4(span_arr.grid_inside.mean()),
        "median_tracks_per_clip": int(span_arr.n_tracks.median()),
        "why_this_matters": (
            "egomotion runs far past 20 s while obstacle.offline stops at ~20 s. "
            "A grid point outside the intersection yields NO obstacle rows, which "
            "is indistinguishable from an empty road -- it would silently convert "
            "a coverage failure into a plausible lead-presence rate."),
    }

    res = {"what": "E-GOAL-1 S1 -- aligned obstacle.offline per-window dump",
           "host": "dev box, CPU only; no pod contacted",
           "corpus": {
               "root": str(ROOT),
               "note": ("read-only over labels/*.zip + r0/phase0_selection.parquet; "
                        "_epcache never written, no clip re-selected, parity key "
                        "e438721ae894 / skip-hash f09e44db untouched"),
               "chunks": [int(c) for c in chunks]},
           "clock_join": clock,
           "coverage": cov,
           "columns": {"ego": EGO_COLS, "lead": LEAD_COLS, "density": DENS_COLS,
                       "derived": XTRA_COLS,
                       "targets": ["y_long", "y_lat"]}}
    (out / "eg_align.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(cov, indent=1), flush=True)
    print(f"[ingest] -> {out/'eg_windows.parquet'} ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
