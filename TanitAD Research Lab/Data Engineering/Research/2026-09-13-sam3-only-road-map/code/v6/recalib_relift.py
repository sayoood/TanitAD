"""Apply a per-camera calibration override (calib_refine.py "override") to a SAM3 map: a corrected sequence root and
the map's points / ranges re-lifted from its UNCHANGED class rasters (SAM3 masks live in the image; only the ground warp
changes).
  root   <out root>/seq_<c8>/<token>/: calib.npz with sensor2lidar_rotation[i] = R @ dR(yaw, pitch, roll) and
         sensor2lidar_translation[i] += (0, 0, dz) for each overridden camera (camera_model convention, identical to
         calib_refine.perturbed); images, lidar.npy, meta.json, frame.json symlinked; poses.json copied
  relift <dst npz dir>: every npz of <src npz dir> with pts_k / rng_k re-lifted exactly as the extractor does
CONTROL first: the same relift with an EMPTY override must reproduce the source points and ranges bit for bit on every
frame, else nothing is written.
Usage: recalib_relift.py <c8> <override json> <src npz dir> <dst npz dir> <out root>   (SAM3MAP_ROOT = source root)
"""
import json, os, shutil, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
for p_ in ("/home/nvidia/sam3paint", "/home/nvidia/sam3map", "/home/nvidia/sam3map/eval", str(HERE)):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
import numpy as np
import sam3_paint as P
import camera_model as CM
import ground_surface as GS
from calib_refine import rot
from sam3map_consensus import relift


def corrected_calib(c, cam_order, override):
    arr = {k: np.array(c[k]) for k in c.files}
    for cam, o in override.items():
        if cam not in cam_order:
            continue
        i = cam_order.index(cam)
        dR = rot("y", o["yaw_deg"]) @ rot("x", o["pitch_deg"]) @ rot("z", o["roll_deg"])
        arr["sensor2lidar_rotation"][i] = arr["sensor2lidar_rotation"][i] @ dR
        arr["sensor2lidar_translation"][i] = arr["sensor2lidar_translation"][i] + np.array([0.0, 0.0, o["dz_m"]])
    return arr


def relift_frame(d, c_arr, fr, sg, cams):
    T = d["T_world_rig"]
    accp, accr = {k: [] for k in range(1, 8)}, {k: [] for k in range(1, 8)}
    for cam in cams:
        C = CM.Camera.from_calib(c_arr, fr["cam_order"].index(cam))
        p_, r_ = relift(d[f"cls_{cam}"], C, sg, T)
        for k in p_:
            accp[k].append(p_[k]); accr[k].append(r_[k])
    cat = lambda lst, shape, dt: np.concatenate(lst) if lst else np.zeros(shape, dt)
    return ({k: cat(accp[k], (0, 2), np.float32) for k in range(1, 8)}, {k: cat(accr[k], (0,), np.float16) for k in range(1, 8)})


class _Arr(dict):
    """dict with .files so camera_model.Camera.from_calib accepts it like an NpzFile."""
    @property
    def files(self):
        return list(self.keys())


def main():
    c8, ov_path, src, dst, out_root = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
    root = Path(os.environ.get("SAM3MAP_ROOT", "/home/nvidia/sam3map/native7"))
    override = json.loads(ov_path.read_text(encoding="utf-8")).get("override", {})
    t0 = time.time()
    files = sorted(src.glob("[0-9][0-9][0-9].npz"))
    sd = root / f"seq_{c8}"
    frames = []
    for f in files:
        d = dict(np.load(f, allow_pickle=True))
        fd = sd / str(d["tok"])
        c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
        grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64))
        cams = [k[4:] for k in d if k.startswith("cls_")]
        frames.append((f, d, c, fr, GS.smooth_grid(grid, fb), cams))
    # CONTROL: empty override reproduces the source bit for bit
    bad = 0
    for f, d, c, fr, sg, cams in frames:
        pts, rng = relift_frame(d, _Arr(corrected_calib(c, fr["cam_order"], {})), fr, sg, cams)
        if not all(np.array_equal(pts[k], d[f"pts_{k}"]) and np.array_equal(rng[k], d[f"rng_{k}"]) for k in range(1, 8) if f"pts_{k}" in d):
            bad += 1
    print(f"CONTROL empty override: {len(frames) - bad}/{len(frames)} identical  {time.time() - t0:.0f}s", flush=True)
    if bad:
        print("ZZRECALIB-CONTROL-FAILEDZZ"); sys.exit(1)
    dst.mkdir(parents=True, exist_ok=True)
    so = out_root / f"seq_{c8}"; so.mkdir(parents=True, exist_ok=True)
    shutil.copy(sd / "poses.json", so / "poses.json")
    moved = {}
    for f, d, c, fr, sg, cams in frames:
        tok = str(d["tok"]); td = so / tok; td.mkdir(exist_ok=True)
        for name in ("images", "lidar.npy", "meta.json", "frame.json"):
            if not (td / name).exists():
                os.symlink(sd / tok / name, td / name)
        arr = corrected_calib(c, fr["cam_order"], override)
        np.savez(td / "calib.npz", **arr)
        pts, rng = relift_frame(d, _Arr(arr), fr, sg, cams)
        for k in range(1, 8):
            if len(pts[k]) and len(d.get(f"pts_{k}", [])) == len(pts[k]):
                moved.setdefault(k, []).append(float(np.median(np.hypot(*(pts[k] - d[f"pts_{k}"]).T))))
            d[f"pts_{k}"] = pts[k]; d[f"rng_{k}"] = rng[k]
        st = json.loads(str(d["stats"])); st["_recalib"] = override; d["stats"] = json.dumps(st)
        np.savez_compressed(dst / f.name, **d)
    rep = {"override": override, "frames": len(frames), "median_point_shift_m_where_counts_match": {k: round(float(np.median(v)), 3) for k, v in moved.items()},
           "seconds": round(time.time() - t0, 1)}
    (dst.parent / f"recalib_{dst.name}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep)); print("ZZRECALIB-DONEZZ")


if __name__ == "__main__":
    main()
