"""Pace of the FRONT-CAMERA map pipeline per frame on Thor, measured on real frames, nothing written to the sequence dirs.
For N frames of a native sequence: the v6 extractor's classify() on CAM_FW (the exact per-view work of the delivered maps: all
prompts, evidence bits, refinements), the stripe pass's two prompts on the same image, and the point lifting; model build timed
separately. Night and day clip, every 10th frame.
Usage: front_pace_probe.py <c8>[,<c8>...] <every>"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3map/eval")
import numpy as np
from PIL import Image
t_import = time.time()
import sam3map_extract_v6 as E

ROOT = Path("/home/nvidia/sam3map/native7")
clips, every = sys.argv[1].split(","), int(sys.argv[2])
t0 = time.time(); proc, _ = E.S.build(conf=0.25); t_build = time.time() - t0
rows = []
for c8 in clips:
    sd = ROOT / f"seq_{c8}"
    toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
    for j in range(0, len(toks), every):
        fd = sd / toks[j]
        c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
        ta = time.time()
        grid, fb = E.P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64)); sg = E.GS.smooth_grid(grid, fb)
        img = Image.open(fd / "images" / "CAM_FW.jpg").convert("RGB")
        C = E.CM.Camera.from_calib(c, fr["cam_order"].index("CAM_FW"), img.width, img.height)
        tb = time.time()
        cls, st, evid = E.classify(proc, img, (C, sg), True)
        tc = time.time()
        state = proc.set_image(img)
        for prompt in ("crosswalk stripe", "white stripe on road"):
            for s, m in E.instances(proc, state, prompt, 0.4):
                pass
        td = time.time()
        for k in range(1, 8):
            m = np.isin(cls, (1, 2, 3, 4, 6)) if k == 1 else (cls == k)
            if m.any():
                C.lift(m, sg, stride=2)
        te = time.time()
        rows.append({"clip": c8, "frame": j, "load_s": round(tb - ta, 2), "classify_s": round(tc - tb, 2), "stripes_s": round(td - tc, 2), "lift_s": round(te - td, 2), "total_s": round(te - ta, 2)})
        print(rows[-1], flush=True)
tot = np.array([r["total_s"] for r in rows]); cl = np.array([r["classify_s"] for r in rows]); sp = np.array([r["stripes_s"] for r in rows])
rep = {"model_build_s": round(t_build, 1), "frames": len(rows), "total_s_median": round(float(np.median(tot)), 2), "total_s_mean": round(float(tot.mean()), 2),
       "classify_s_mean": round(float(cl.mean()), 2), "stripes_s_mean": round(float(sp.mean()), 2), "rows": rows}
Path("/home/nvidia/sam3map/front_pace_probe.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
print({k: v for k, v in rep.items() if k != "rows"}); print("ZZPACE-DONEZZ")
