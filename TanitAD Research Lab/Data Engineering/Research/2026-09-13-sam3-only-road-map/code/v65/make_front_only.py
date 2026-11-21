"""Front-camera-only arm from an all-camera v6 extraction: keep CAM_FW's class raster and evidence, drop the other six
cameras, re-lift points and ranges from CAM_FW exactly as the extractor did. Everything downstream (refine, consensus,
renderer, scorer) then runs unchanged, so the camera set is the only difference between the two arms.
Usage: make_front_only.py <c8> <src v6raw dir> <dst dir>   (SAM3MAP_ROOT = native sequence root)"""
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3paint"); sys.path.insert(0, str(HERE))
import numpy as np
import sam3_paint as P
import camera_model as CM
import ground_surface as GS
from sam3map_consensus import relift

ROOT = Path(os.environ.get("SAM3MAP_ROOT", "/home/nvidia/sam3map/native7"))
c8, src, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); dst.mkdir(parents=True, exist_ok=True)
t0 = time.time(); n = 0
for f in sorted(src.glob("[0-9][0-9][0-9].npz")):
    d = dict(np.load(f, allow_pickle=True)); tok = str(d["tok"])
    out = {k: v for k, v in d.items() if not (k.startswith(("cls_", "evid_", "ego_", "pts_", "rng_")))}
    out["cls_CAM_FW"] = d["cls_CAM_FW"]
    if "evid_CAM_FW" in d:
        out["evid_CAM_FW"] = d["evid_CAM_FW"]
    fd = ROOT / f"seq_{c8}" / tok
    c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
    grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64)); sg = GS.smooth_grid(grid, fb)
    p_, r_ = relift(d["cls_CAM_FW"], CM.Camera.from_calib(c, fr["cam_order"].index("CAM_FW")), sg, d["T_world_rig"])
    for k in range(1, 8):
        out[f"pts_{k}"] = p_.get(k, np.zeros((0, 2), np.float32)); out[f"rng_{k}"] = r_.get(k, np.zeros((0,), np.float16))
    st = json.loads(str(d["stats"])); out["stats"] = json.dumps({"CAM_FW": st.get("CAM_FW", {}), "_front_only_from": src.name})
    out["ver"] = "v6f"
    np.savez_compressed(dst / f.name, **out); n += 1
print(f"front-only frames {n}  {time.time() - t0:.0f}s"); print("ZZFRONTONLY-DONEZZ")
