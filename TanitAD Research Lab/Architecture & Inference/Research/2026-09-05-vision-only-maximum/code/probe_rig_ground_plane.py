"""Where is z = 0 in the PhysicalAI rig frame? -- the metric-scale anchor question.

Monocular depth is scale-free.  The standard fix is a KNOWN CAMERA HEIGHT above the
road (SfMLearner / MonoDepth lineage: fit the ground plane, scale so the camera sits at
h).  ``sensor_extrinsics`` puts ``camera_front_wide_120fov`` at z = +1.430 m in the rig
frame -- but that is only a camera HEIGHT if the rig's z = 0 IS the road surface, and
nothing in our repo has ever asserted that.

THE CONTROL THAT MUST READ A KNOWN VALUE
----------------------------------------
``obstacle.offline`` cuboids are expressed in the same rig frame (``reference_frame ==
"rig"`` on every row).  A cuboid carries ``center_z`` and ``size_z``.  If z = 0 is the
road surface, then for cars standing on that road

    center_z  ~=  size_z / 2                (a car's centre sits at half its height)

so the ratio ``center_z / (size_z/2)`` must read ~1.0, and the residual
``center_z - size_z/2`` must read ~0 m.  If instead z = 0 were the rear-axle centre
(~0.35 m up) or the rig/roof, the residual would read that offset.  This is a control
whose correct answer is known in advance, which is the only thing that makes the
conclusion trustworthy (CLAUDE.md probe rule).

Cheap: one ``obstacle.offline`` chunk zip (~50 MB), no GPU.
Writes ``$PROBE_OUT`` (default ``../raw/rig_ground_plane_probe.json``).
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import re
import sys
import zipfile

import truststore

truststore.inject_into_ssl()

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "raw"
KEYS = pathlib.Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt")
N_CLIPS = 12


def _token() -> str:
    import time
    txt = None
    for _k in range(30):  # the G: mount flaps (CLAUDE.md); retry rather than die
        try:
            txt = KEYS.read_text(encoding="utf-8", errors="replace")
            break
        except OSError:
            time.sleep(3)
    if txt is None:
        raise SystemExit("Keys.txt unreadable after 30 tries -- mount wedged")
    m = re.findall(r"hf_[A-Za-z0-9]+", txt)
    if not m:
        raise SystemExit("no hf_ token found in Keys.txt")
    return max(m, key=len)


def main() -> int:
    from huggingface_hub import hf_hub_download
    import numpy as np
    import pandas as pd

    tok = _token()
    fp = hf_hub_download(
        REPO,
        "labels/obstacle.offline/obstacle.offline.chunk_0000.zip",
        repo_type="dataset",
        token=tok,
    )
    out: dict = {
        "source": "labels/obstacle.offline/obstacle.offline.chunk_0000.zip",
        "zip_bytes": os.path.getsize(fp),
        "n_clips_read": 0,
    }
    frames = []
    with zipfile.ZipFile(fp) as z:
        members = [n for n in z.namelist() if n.endswith(".parquet")]
        out["n_members_in_zip"] = len(members)
        for name in members[:N_CLIPS]:
            with z.open(name) as fh:
                frames.append(pd.read_parquet(io.BytesIO(fh.read())))
    df = pd.concat(frames, ignore_index=True)
    out["n_clips_read"] = min(N_CLIPS, out["n_members_in_zip"])
    out["n_rows"] = int(len(df))
    out["columns"] = list(map(str, df.columns))
    if "reference_frame" in df:
        out["reference_frames"] = {
            str(k): int(v) for k, v in df["reference_frame"].value_counts().items()
        }
    cls_col = "label_class" if "label_class" in df else None
    out["classes"] = (
        {str(k): int(v) for k, v in df[cls_col].value_counts().items()} if cls_col else None
    )

    res: dict = {}
    for cls, g in (df.groupby(cls_col) if cls_col else [("ALL", df)]):
        if len(g) < 50:
            continue
        cz = g["center_z"].to_numpy(dtype=float)
        sz = g["size_z"].to_numpy(dtype=float)
        bottom = cz - sz / 2.0
        res[str(cls)] = {
            "n": int(len(g)),
            "size_z_median_m": float(np.median(sz)),
            "center_z_median_m": float(np.median(cz)),
            "bottom_z_median_m": float(np.median(bottom)),
            "bottom_z_p25_m": float(np.percentile(bottom, 25)),
            "bottom_z_p75_m": float(np.percentile(bottom, 75)),
            "ratio_center_over_halfheight_median": float(np.median(cz / (sz / 2.0 + 1e-9))),
        }
    out["per_class"] = res

    # The verdict, computed rather than eyeballed.  Cars are the control class: their
    # bottom face is the road.  |median bottom_z| < 0.15 m  =>  z = 0 IS the road plane.
    car = res.get("Automobile") or res.get("car") or res.get("Car") or res.get("vehicle")
    if car is None:
        big = max(res.items(), key=lambda kv: kv[1]["n"])
        out["control_class_used"] = big[0]
        car = big[1]
    else:
        out["control_class_used"] = next(
            k for k, v in res.items() if v is car
        )
    out["bottom_z_median_of_control_m"] = car["bottom_z_median_m"]
    out["verdict_rig_z0_is_road_plane"] = bool(abs(car["bottom_z_median_m"]) < 0.15)
    out["camera_height_above_road_m_if_true"] = 1.430 - car["bottom_z_median_m"]

    print(json.dumps(out, indent=2)[:6000])
    dest = pathlib.Path(os.environ.get("PROBE_OUT", str(OUT / "rig_ground_plane_probe.json")))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
