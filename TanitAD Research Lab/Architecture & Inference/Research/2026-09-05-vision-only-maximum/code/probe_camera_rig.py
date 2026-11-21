"""Probe the PhysicalAI-AV camera rig: how many cameras, where they point, what they cost.

Two independent probes, both MEASURED at run time:

  P-A  HF blob listing of ``camera/`` + ``calibration/`` (file counts, byte totals) --
       independent of the 2026-07-26 DataFlyWheel census, which read ONE chunk and
       multiplied.
  P-B  The rig geometry itself: pull the 63 KB
       ``calibration/sensor_extrinsics/sensor_extrinsics.chunk_0000.parquet`` and
       compute, per sensor, the mount position and the BORESIGHT AZIMUTH in the vehicle
       frame from the quaternion.  This is what settles the coverage fraction -- the
       FOV names alone do not, because they do not say where each camera points.

Coverage is then computed as the union of per-camera azimuth intervals
[yaw - HFOV/2, yaw + HFOV/2] on the circle.

.. warning::
   The HFOV in each camera's NAME is the manufacturer's figure for that lens.  The
   *pinhole* formula is wrong on our canonical CYLINDRICAL frame (CLAUDE.md), but the
   name is a rig fact, not a projection fact, so it is used directly here.

Writes its JSON to ``$PROBE_OUT`` (default ``../raw/camera_rig_probe.json``).  The G:
mount raises Errno 22 mid-write, so author on local disk and copy in.  No GPU.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
import sys

import truststore

truststore.inject_into_ssl()

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "raw"
KEYS = pathlib.Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt")

# HFOV per camera, from the dataset card's own sensor names (PUBLISHED).
NAME_HFOV_DEG = {
    "camera_front_wide_120fov": 120.0,
    "camera_front_tele_30fov": 30.0,
    "camera_cross_left_120fov": 120.0,
    "camera_cross_right_120fov": 120.0,
    "camera_rear_left_70fov": 70.0,
    "camera_rear_right_70fov": 70.0,
    "camera_rear_tele_30fov": 30.0,
}


def _token() -> str:
    """Read the HF token IN PLACE; never copy it anywhere else (CLAUDE.md invariant)."""
    txt = KEYS.read_text(encoding="utf-8", errors="replace")
    m = re.findall(r"hf_[A-Za-z0-9]+", txt)
    if not m:
        raise SystemExit("no hf_ token found in Keys.txt")
    return max(m, key=len)


def quat_axes(qx: float, qy: float, qz: float, qw: float):
    """Return the sensor's +x and +z axes expressed in the rig frame.

    PhysicalAI ships extrinsics as rig<-sensor rotations.  A camera looks down its own
    +z in the usual optical convention; the rig frame is x forward, y left, z up
    (MEASURED by the parked-car experiment, ``tanitad/data/bev_raster.py``).  Both
    candidate axes are returned so a convention disagreement is VISIBLE rather than
    assumed -- the front-wide camera is the control: whichever axis puts it near 0 deg
    azimuth is the optical axis.
    """
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    r = [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]
    z_axis = [r[0][2], r[1][2], r[2][2]]
    x_axis = [r[0][0], r[1][0], r[2][0]]
    az_z = math.degrees(math.atan2(z_axis[1], z_axis[0]))
    el_z = math.degrees(math.asin(max(-1.0, min(1.0, z_axis[2]))))
    az_x = math.degrees(math.atan2(x_axis[1], x_axis[0]))
    el_x = math.degrees(math.asin(max(-1.0, min(1.0, x_axis[2]))))
    return az_z, el_z, az_x, el_x, z_axis, x_axis


def union_coverage(intervals_deg, n_bins: int = 3600) -> float:
    """Fraction of the azimuth circle covered by the union of the intervals."""
    hit = [False] * n_bins
    step = 360.0 / n_bins
    for lo, hi in intervals_deg:
        span = hi - lo
        k0 = int(math.floor((lo % 360.0) / step))
        n_steps = int(math.ceil(span / step))
        for k in range(n_steps + 1):
            hit[(k0 + k) % n_bins] = True
    return sum(hit) / n_bins


def main() -> int:
    from huggingface_hub import HfApi, hf_hub_download

    tok = _token()
    api = HfApi(token=tok)
    out: dict = {"repo": REPO, "probes": {}}

    # ---------------------------------------------------------------- P-A listing
    info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)
    per_dir: dict[str, dict] = {}
    for sib in info.siblings or []:
        parts = sib.rfilename.split("/")
        if len(parts) < 2:
            continue
        top, feat = parts[0], parts[1]
        if top not in ("camera", "calibration", "lidar", "labels"):
            continue
        key = f"{top}/{feat}"
        d = per_dir.setdefault(key, {"n_files": 0, "bytes": 0})
        d["n_files"] += 1
        d["bytes"] += int(sib.size or 0)
    out["probes"]["P_A_blob_listing"] = {
        "sha": info.sha,
        "last_modified": str(getattr(info, "last_modified", None)),
        "dirs": {k: per_dir[k] for k in sorted(per_dir)},
    }
    print("== P-A ==")
    for k in sorted(per_dir):
        d = per_dir[k]
        print(f"  {k:44s} n_files={d['n_files']:6d} bytes={d['bytes']:>16,d}")

    # ---------------------------------------------------------------- P-B rig geometry
    fp = hf_hub_download(
        REPO,
        "calibration/sensor_extrinsics/sensor_extrinsics.chunk_0000.parquet",
        repo_type="dataset",
        token=tok,
    )
    import pandas as pd

    df = pd.read_parquet(fp)
    meta = {
        "file": "calibration/sensor_extrinsics/sensor_extrinsics.chunk_0000.parquet",
        "n_rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "index_names": [str(x) for x in (df.index.names or [])],
    }
    out["probes"]["P_B_extrinsics"] = meta
    print("== P-B ==")
    print("  cols:", meta["columns"], "index:", meta["index_names"], "rows:", meta["n_rows"])

    df = df.reset_index()
    name_col = next(
        (c for c in df.columns if str(c).lower() in ("sensor_name", "camera_name", "name")), None
    )
    meta["name_col"] = str(name_col)
    if name_col is None:
        print("  !! no sensor-name column; columns after reset_index:", list(map(str, df.columns)))
        dest = pathlib.Path(os.environ.get("PROBE_OUT", str(OUT / "camera_rig_probe.json")))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        return 2

    names = sorted(set(map(str, df[name_col].tolist())))
    meta["distinct_sensors"] = names
    meta["n_distinct_sensors"] = len(names)

    rig: dict[str, dict] = {}
    for nm, g in df.groupby(name_col):
        row = {c: float(g[c].median()) for c in ("qx", "qy", "qz", "qw", "x", "y", "z") if c in g}
        if len(row) < 7:
            rig[str(nm)] = {"n_clips": int(len(g)), "cols_missing": sorted(set(
                ("qx", "qy", "qz", "qw", "x", "y", "z")) - set(row))}
            continue
        az_z, el_z, az_x, el_x, zax, xax = quat_axes(row["qx"], row["qy"], row["qz"], row["qw"])
        rig[str(nm)] = {
            "n_clips": int(len(g)),
            "pos_m": {"x": row["x"], "y": row["y"], "z": row["z"]},
            "pos_spread_p95_p05_m": {
                k: float(g[k].quantile(0.95) - g[k].quantile(0.05)) for k in ("x", "y", "z")
            },
            "az_deg_sensor_plus_z": az_z,
            "el_deg_sensor_plus_z": el_z,
            "az_deg_sensor_plus_x": az_x,
            "el_deg_sensor_plus_x": el_x,
            "sensor_z_in_rig": zax,
            "sensor_x_in_rig": xax,
            "hfov_deg_from_name": NAME_HFOV_DEG.get(str(nm)),
        }
    out["rig"] = rig

    for conv, key in (("plus_z", "az_deg_sensor_plus_z"), ("plus_x", "az_deg_sensor_plus_x")):
        ivs, ivs_all, ivs_wide = [], [], []
        for nm, r in rig.items():
            fov = r.get("hfov_deg_from_name")
            if fov is None or key not in r:
                continue
            az = r[key]
            ivs_all.append((az - fov / 2.0, az + fov / 2.0))
            if "120fov" in nm:
                ivs_wide.append((az - fov / 2.0, az + fov / 2.0))
            if nm == "camera_front_wide_120fov":
                ivs.append((az - fov / 2.0, az + fov / 2.0))
        out.setdefault("coverage", {})[conv] = {
            "all_7_cameras": union_coverage(ivs_all) if ivs_all else None,
            "three_120fov_cameras": union_coverage(ivs_wide) if ivs_wide else None,
            "front_wide_only": union_coverage(ivs) if ivs else None,
        }

    print("== rig ==")
    for nm in sorted(rig):
        r = rig[nm]
        if "pos_m" in r:
            print(
                f"  {nm:30s} n={r['n_clips']:5d} "
                f"pos=({r['pos_m']['x']:+.3f},{r['pos_m']['y']:+.3f},{r['pos_m']['z']:+.3f}) "
                f"az+z={r['az_deg_sensor_plus_z']:+8.2f} el+z={r['el_deg_sensor_plus_z']:+6.2f} "
                f"az+x={r['az_deg_sensor_plus_x']:+8.2f} el+x={r['el_deg_sensor_plus_x']:+6.2f} "
                f"fov={r['hfov_deg_from_name']}"
            )
        else:
            print(f"  {nm:30s} n={r['n_clips']:5d} MISSING {r.get('cols_missing')}")
    print("== coverage ==")
    print(json.dumps(out.get("coverage"), indent=2))

    dest = pathlib.Path(os.environ.get("PROBE_OUT", str(OUT / "camera_rig_probe.json")))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
