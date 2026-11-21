"""Calibration on Qwen-Drive's demo frames: what do TRUE and PREDICTED rasters score on each metric?

Arms per frame: GT (the true map/occupancy), PRED (Qwen-Drive, A0), and two no-information controls --
SHUFFLED (the GT raster of a different demo frame of the same rig) and MIRRORED (GT flipped left-right).
A metric is only admissible for our data if GT separates from both controls here.
"""
import glob, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/nvidia/qwendrive/v2/mapq")
import mapq_core as mc  # noqa: E402

DEMO = sorted(glob.glob("/home/nvidia/qwendrive/qwen-drive/data/demo/perception/*"))
CTRL = Path("/home/nvidia/qwendrive/ctrl/A0_out")
frames = []
for fd in DEMO:
    fr = json.load(open(fd + "/frame.json"))
    c = np.load(fd + "/calib.npz"); l2e = c["lidar2ego"]
    L = np.load(fd + "/lidar.npy").astype(np.float64)
    pts = L @ l2e[:3, :3].T + l2e[:3, 3]
    g = np.load(fd + "/gt.npz"); p = np.load(CTRL / (Path(fd).name + ".npz"))
    frames.append(dict(tok=Path(fd).name, typ=fr["dataset_type"], pts=pts, boxes=g["boxes"], labels=g["labels"],
                       gt_map=g["map"].astype(int), gt_occ=g["occ"], pr_map=p["map"].astype(int), pr_occ=p["occ"]))

rows = {k: [] for k in ("GT", "PRED", "SHUFFLED", "MIRRORED")}
frames = [f for f in frames if f["typ"] == "nuplan"]   # the rig v2 emulates; nuScenes' 32-beam sweep is too sparse
per_frame = []
for idx, f in enumerate(frames):
    veh = f["boxes"][f["labels"] == 0]
    same = [o for o in frames if o["typ"] == f["typ"] and o["tok"] != f["tok"]]
    sh = same[idx % len(same)]
    arms = {"GT": (f["gt_map"], f["gt_occ"]), "PRED": (f["pr_map"], f["pr_occ"]),
            "SHUFFLED": (sh["gt_map"], sh["gt_occ"]),
            "MIRRORED": (f["gt_map"][::-1, :], f["gt_occ"].reshape(200, 200, 16)[:, ::-1, :])}
    for arm, (m, o) in arms.items():
        r = mc.frame_metrics(m, o, f["pts"], f["boxes"], veh)
        rows[arm].append(r)
        per_frame.append({"tok": f["tok"], "arm": arm, **{k: round(a / n, 4) if n else None for k, (a, n) in r.items()}})
res = {arm: mc.aggregate(r) for arm, r in rows.items()}
res["frames"] = [f["tok"] + ":" + f["typ"] for f in frames]
res["per_frame"] = per_frame
for arm in rows:
    res[arm + "_median_of_frames"] = {k: round(float(np.median([pf[k] for pf in per_frame if pf["arm"] == arm and pf.get(k) is not None])), 4)
                                      for k in ("MAP_A", "MAP_A2", "MAP_C", "MAP_E", "OCC_A", "OCC_A2", "OCC_B", "OCC_C")
                                      if any(pf.get(k) is not None for pf in per_frame if pf["arm"] == arm)}
Path("/home/nvidia/qwendrive/v2/mapq/demo_calibration.json").write_text(json.dumps(res, indent=1))
keys = ["MAP_A", "MAP_A2", "MAP_C", "MAP_E", "OCC_A", "OCC_A2", "OCC_B", "OCC_C"]
print(f"{'arm':9s} " + " ".join(f"{k:>14s}" for k in keys))
for arm in ("GT", "PRED", "SHUFFLED", "MIRRORED"):
    print(f"{arm:9s} " + " ".join(f"{res[arm][k]['value']:>8} n{res[arm][k]['n']:<5d}" if k in res[arm] else f"{'-':>14s}" for k in keys))
for arm in ("GT", "PRED", "SHUFFLED", "MIRRORED"):
    print(f"{arm:9s} median-of-frames", res[arm + "_median_of_frames"])
print("ZZDEMO-CAL-DONEZZ")
