#!/usr/bin/env python3
"""P1 — establish `obstacle.offline` FROM SOURCE, on real local bytes.

Answers, each MEASURED here (not inherited from pai_label_schemas.json):
  schema · clock alignment with egomotion · sampling rate · per-cuboid vs
  per-frame timestamps · coordinate frame convention (x fwd / y left) ·
  track continuity · corpus coverage.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack"))
ROOT = Path(r"C:/Users/Admin/tanitad-data/physicalai")
OUT = Path(__file__).with_name("p1_obstacle_probe.json")


def chunks():
    o = {int(p.name.split("_")[-1].split(".")[0])
         for p in (ROOT / "labels" / "obstacle.offline").glob("*.zip")}
    e = {int(p.name.split("_")[-1].split(".")[0])
         for p in (ROOT / "labels" / "egomotion").glob("*.zip")}
    return sorted(o & e)


def main():
    res = {"_what": "MEASURED obstacle.offline facts, read from local gated bytes",
           "_root": str(ROOT), "chunks_probed": []}
    cs = chunks()
    res["n_chunks_local_with_both"] = len(cs)

    # ---------------- coverage: clips with an obstacle member ---------------- #
    sel = pd.read_parquet(ROOT / "r0" / "phase0_selection.parquet")
    sel["clip_id"] = sel["clip_id"].astype(str)
    chunk_of = dict(zip(sel["clip_id"], sel["chunk"].astype(int)))
    by_chunk = {}
    for cid, ch in chunk_of.items():
        by_chunk.setdefault(ch, set()).add(cid)

    cov_have = cov_want = 0
    per_chunk_cov = {}
    for c in cs:
        want = by_chunk.get(c, set())
        if not want:
            continue
        with zipfile.ZipFile(ROOT / "labels" / "obstacle.offline" /
                             f"obstacle.offline.chunk_{c:04d}.zip") as z:
            present = {n.split("/")[-1].split(".")[0] for n in z.namelist()
                       if n.endswith(".parquet")}
        h = len(want & present)
        cov_have += h
        cov_want += len(want)
        per_chunk_cov[str(c)] = {"selected": len(want), "with_obstacle": h}
    res["coverage_over_phase0_selection"] = {
        "n_selected_clips_in_local_chunks": cov_want,
        "n_with_obstacle_member": cov_have,
        "frac": round(cov_have / max(cov_want, 1), 4),
        "note": ("fraction of OUR phase0-selected clips that have an "
                 "obstacle.offline parquet, over the locally-held chunks only"),
        "per_chunk": per_chunk_cov,
    }

    # ---------------- deep probe on a handful of real clips ----------------- #
    deep = []
    classes = Counter()
    n_deep = 0
    for c in cs:
        if n_deep >= 12:
            break
        ozp = ROOT / "labels" / "obstacle.offline" / f"obstacle.offline.chunk_{c:04d}.zip"
        ezp = ROOT / "labels" / "egomotion" / f"egomotion.chunk_{c:04d}.zip"
        with zipfile.ZipFile(ozp) as oz, zipfile.ZipFile(ezp) as ez:
            om = {n.split("/")[-1].split(".")[0]: n for n in oz.namelist()
                  if n.endswith(".parquet")}
            em = {n.split("/")[-1].split(".")[0]: n for n in ez.namelist()
                  if n.endswith(".parquet")}
            want = sorted((by_chunk.get(c, set()) & set(om) & set(em)))[:2]
            for cid in want:
                obs = pd.read_parquet(io.BytesIO(oz.read(om[cid])))
                ego = pd.read_parquet(io.BytesIO(ez.read(em[cid])))
                classes.update(obs["label_class"].astype(str).tolist())
                to = obs["timestamp_us"].to_numpy(np.float64)
                te = ego["timestamp"].to_numpy(np.float64)
                # per-track cadence
                cad, life, nsamp = [], [], []
                for tid, g in obs.groupby("track_id"):
                    t = np.sort(g["timestamp_us"].to_numpy(np.float64)) / 1e6
                    nsamp.append(len(t))
                    if len(t) > 1:
                        cad.append(float(np.median(np.diff(t))))
                        life.append(float(t[-1] - t[0]))
                d = {
                    "chunk": c,
                    "clip_sha8": __import__("hashlib").sha256(cid.encode()).hexdigest()[:8],
                    "n_rows": int(len(obs)),
                    "n_tracks": int(obs["track_id"].nunique()),
                    "obs_t_min_us": float(to.min()), "obs_t_max_us": float(to.max()),
                    "obs_span_s": round(float((to.max() - to.min()) / 1e6), 3),
                    "ego_t_min_us": float(te.min()), "ego_t_max_us": float(te.max()),
                    "ego_span_s": round(float((te.max() - te.min()) / 1e6), 3),
                    "clock_offset_start_s": round(float((to.min() - te.min()) / 1e6), 3),
                    "obs_inside_ego_span": bool(to.min() >= te.min() - 1e3
                                                and to.max() <= te.max() + 1e3),
                    "n_unique_timestamps": int(len(np.unique(to))),
                    "rows_per_unique_ts": round(len(to) / max(len(np.unique(to)), 1), 3),
                    "median_track_cadence_s": (round(float(np.median(cad)), 4)
                                               if cad else None),
                    "median_track_life_s": (round(float(np.median(life)), 3)
                                            if life else None),
                    "median_samples_per_track": (int(np.median(nsamp)) if nsamp else 0),
                    "ego_median_dt_s": round(float(np.median(np.diff(np.sort(te)))) / 1e6, 4),
                    "reference_frame": sorted(obs["reference_frame"].astype(str).unique()),
                    "ref_ts_equals_ts": bool(np.array_equal(
                        obs["reference_frame_timestamp_us"].to_numpy(np.uint64),
                        obs["timestamp_us"].to_numpy(np.uint64))),
                    "columns": list(map(str, obs.columns)),
                }
                # frame convention: for the nearest in-corridor vehicle ahead,
                # does center_x shrink as the ego drives forward?
                veh = obs[obs["label_class"].isin(
                    ["automobile", "heavy_truck", "bus", "other_vehicle", "trailer"])]
                fwd = veh[(veh["center_x"] > 3) & (veh["center_x"] < 60)
                          & (veh["center_y"].abs() < 2.0)]
                d["n_inlane_ahead_rows"] = int(len(fwd))
                d["frac_rows_x_positive"] = round(float((veh["center_x"] > 0).mean()), 3)
                d["abs_y_p50_m"] = round(float(veh["center_y"].abs().median()), 2)
                d["abs_x_p50_m"] = round(float(veh["center_x"].abs().median()), 2)
                d["size_x_p50"] = round(float(veh["size_x"].median()), 2)
                d["size_y_p50"] = round(float(veh["size_y"].median()), 2)
                deep.append(d)
                n_deep += 1
        res["chunks_probed"].append(c)
    res["per_clip"] = deep
    res["class_counts_probed"] = dict(classes.most_common())
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1)[:2500])
    print("\n--- per clip (first 3) ---")
    print(json.dumps(deep[:3], indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
