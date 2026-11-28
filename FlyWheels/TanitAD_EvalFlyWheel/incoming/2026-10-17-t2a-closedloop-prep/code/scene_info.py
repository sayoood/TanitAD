"""Per-scene facts the box-2 run needs, from ``rig_trajectories.json`` ALONE (no GPU, no render, no volume.nurec).

    python scene_info.py <scene dir or rig_trajectories.json> [...]  [--json out.json]

Prints / writes, per scene:
  * the clip length, the logged 10 Hz speed profile (min / max / the [2 s, 6 s] max the T1 ceiling is derived from) and the
    refcv6 ceiling BIN each definition lands in -- so a reader can see whether the oracle ceiling is robust to the
    exact definition (it is, when both land in the same {30, 50, 100, 120} km/h bin);
  * the camera mount (``T_sensor_rig``) and its distance to the nearest PhysicalAI-AV ``sensor_extrinsics`` row, when the
    eval-kit extrinsics table is reachable (the CONVENTION check behind ``extrinsics_from_T_sensor_rig``);
  * the rolling-shutter readout and the first/second frame timestamps (30 Hz -> stride 3).

The ceiling is an ORACLE of the logged ego's future (PI-authorised 2026-09-16 for the T1 sidecar) and exists in the
T2 run ONLY as the explicit arm ``--t2a-vmax-ms``; the withheld arm is the deployable one.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t2a_policies as tp  # noqa: E402  (imports closedloop_drive -> needs the tree on sys.path)

STEPS_KMH = (30, 50, 100, 120)       # refcv6_max_speed.SPEED_MAX_STEPS_KMH_V6 (containing-window quantisation)


def bin_of(v_ms: float) -> int:
    kmh = v_ms * 3.6
    for i, s in enumerate(STEPS_KMH):
        if kmh <= s:
            return i
    return len(STEPS_KMH) - 1                  # > 120 clamps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--extrinsics", default="D:/refcv6_eval_kit/data/refcv6_train_eval139_extrinsics.json")
    a = ap.parse_args()
    ext_rot = []
    if Path(a.extrinsics).exists():
        tab = json.loads(Path(a.extrinsics).read_text())
        ext_rot = [tp.quat_xyzw_to_rot(r["qx"], r["qy"], r["qz"], r["qw"]) for r in tab.values() if "qx" in r]
    out = {}
    for sc in a.scenes:
        rig = tp.read_rig_json(sc)
        d = rig["json"]
        ent = d["rig_trajectories"][0]
        key = f"{tp.cd.CAM}@{ent['sequence_id']}"
        par = d["camera_calibrations"][key]["camera_model"]["parameters"]
        Tf = ent["cameras_frame_T_rig_worlds"][key]
        n = len(Tf)
        ts = [ent["cameras_frame_timestamps_us"][key][f][1] for f in range(3)]
        stride = int(round(1e5 / (ts[1] - ts[0])))
        xy = np.array([np.array(Tf[f * stride][1], np.float64)[:2, 3] for f in range(n // stride)])
        v = np.linalg.norm(np.diff(xy, axis=0), axis=1) / 0.1
        v2_6 = v[20:60] if len(v) > 60 else v
        T = tp.front_wide_T_sensor_rig(rig["json"])
        dist = float(min(np.linalg.norm(R - T[:3, :3]) for R in ext_rot)) if ext_rot else None
        e = {"rig_json": rig["path"], "n_camera_frames": int(n), "stride_to_10hz": stride, "n_ticks_10hz": int(n // stride),
             "duration_s": float((n - 1) / (1e6 / (ts[1] - ts[0]))),
             "speed_ms": {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
                          "v0_first_tick": float(v[0]), "max_2_to_6s": float(v2_6.max())},
             "ceiling_bin_whole_clip_max": bin_of(float(v.max())),
             "ceiling_bin_2_to_6s_max": bin_of(float(v2_6.max())),
             "ceiling_kmh_whole_clip_max": STEPS_KMH[bin_of(float(v.max()))],
             "ceiling_kmh_2_to_6s_max": STEPS_KMH[bin_of(float(v2_6.max()))],
             "suggested_t2a_vmax_ms": float(v2_6.max()),
             "bin_margin_kmh": {"2_to_6s_max_kmh": float(v2_6.max() * 3.6), "bin_edges_kmh": list(STEPS_KMH)},
             "shutter": str(par["shutter_type"]), "cam_wh": [int(x) for x in par["resolution"]],
             "cam_cx_cy": [float(x) for x in par["principal_point"]],
             "T_sensor_rig_xyz_m": [float(x) for x in T[:3, 3]],
             "extrinsics_rot_frob_dist_to_nearest_physicalai_row": dist}
        out[Path(sc).name if Path(sc).is_dir() else Path(sc).parent.name] = e
        print(json.dumps({Path(sc).name: e}, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
