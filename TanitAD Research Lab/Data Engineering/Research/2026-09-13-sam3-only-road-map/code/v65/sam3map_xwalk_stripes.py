"""Crosswalk = the painted STRIPES, not the crossing area (v6s), from SAM3 stripe prompts.

MEASURED (stripe_prompt_test.py, 9 crosswalk views, day + night, front / cross / rear / tele cameras): "crosswalk stripe"
at >= 0.5 returns ~11 instances per view, 97 % of its mask is bright paint, it covers 72 % of the stripe paint and colours
6.5 % of the gaps between stripes (the area prompt "crosswalk": 70 %), with 2 % of the mask outside the crossing.
"white stripe on road" covers more stripes (77 %) but 18 % of its mask lies outside crossings (other markings), so it is
used only inside a detected crossing.
Rule per camera raster (960x540): stripes = ("crosswalk stripe" (>= 0.4) OR "white stripe on road" (>= 0.4)) AND inside the
extraction's crosswalk evidence dilated 31 px; every class-3 (area) pixel becomes drivable; stripe pixels that are
drivable, line or hatched become 3. The stripe mask is kept as xstripe_<cam>; points re-lifted exactly as the extractor did.
Usage: sam3map_xwalk_stripes.py <c8> <src raw dir> <dst raw dir>   (SAM3MAP_ROOT native root, SAM3MAP_VIEWS)
"""
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, str(HERE))
import numpy as np
import cv2
from PIL import Image
import sam3map_extract_v6 as E
from sam3map_consensus import relift

ROOT = Path(os.environ.get("SAM3MAP_ROOT", "/home/nvidia/sam3map/native7"))
VIEWS = os.environ.get("SAM3MAP_VIEWS", "CAM_FW,CAM_CL,CAM_CR,CAM_RL,CAM_RR,CAM_RT,CAM_FT").split(",")
THR = 0.4


def main():
    c8, src, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc, _ = E.S.build(conf=0.25)
    tot = {"frames": 0, "area_px_removed": 0, "stripe_px_set": 0, "stripe_instances": 0, "white_stripe_px_in_area": 0}
    for f in sorted(src.glob("[0-9][0-9][0-9].npz")):
        if (dst / f.name).exists():
            continue
        d = dict(np.load(f, allow_pickle=True)); tok = str(d["tok"]); fd = ROOT / f"seq_{c8}" / tok
        cams = [v for v in VIEWS if f"cls_{v}" in d]
        per = {}
        for cam in cams:
            img = Image.open(fd / "images" / f"{cam}.jpg").convert("RGB")
            state = proc.set_image(img)
            H, W = img.height, img.width
            xs = np.zeros((H, W), bool); nx = 0
            for s, m in E.instances(proc, state, "crosswalk stripe", THR):
                xs |= m; nx += 1
            ws = np.zeros((H, W), bool)
            for s, m in E.instances(proc, state, "white stripe on road", THR):
                ws |= m
            ev = d.get(f"evid_{cam}")
            area = np.zeros((H, W), bool)
            if ev is not None:
                area = cv2.dilate(np.repeat(np.repeat(((ev & 2) > 0).astype(np.uint8), 2, axis=0), 2, axis=1)[:H, :W], np.ones((31, 31), np.uint8)) > 0
            stripes = (xs | ws) & area                                          # both prompts gated by a detected crossing (smoke: a line bar fired)
            small = stripes[::2, ::2]                                           # the extractor's 2x nearest downscale
            cl = d[f"cls_{cam}"].copy()
            n_area = int((cl == 3).sum()); cl[cl == 3] = 1
            put = small & np.isin(cl, (1, 2, 6))                                # a zebra stored as hatched at extraction also gets its stripes
            cl[put] = 3
            d[f"cls_{cam}"] = cl; d[f"xstripe_{cam}"] = small
            per[cam] = {"area_px_removed": n_area, "stripe_px_set": int(put.sum()), "crosswalk_stripe_instances": nx}
            tot["area_px_removed"] += n_area; tot["stripe_px_set"] += int(put.sum()); tot["stripe_instances"] += nx
            tot["white_stripe_px_in_area"] += int((ws & area & ~xs).sum())
        c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
        grid, fb = E.P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64)); sg = E.GS.smooth_grid(grid, fb)
        accp, accr = {k: [] for k in range(1, 8)}, {k: [] for k in range(1, 8)}
        for cam in cams:
            p_, r_ = relift(d[f"cls_{cam}"], E.CM.Camera.from_calib(c, fr["cam_order"].index(cam)), sg, d["T_world_rig"])
            for k in p_:
                accp[k].append(p_[k]); accr[k].append(r_[k])
        for k in range(1, 8):
            d[f"pts_{k}"] = np.concatenate(accp[k]) if accp[k] else np.zeros((0, 2), np.float32)
            d[f"rng_{k}"] = np.concatenate(accr[k]) if accr[k] else np.zeros((0,), np.float16)
        st = json.loads(str(d["stats"])); st["_xwalk_stripes"] = per; d["stats"] = json.dumps(st); d["ver"] = "v6s"
        np.savez_compressed(dst / f.name, **d)
        tot["frames"] += 1
        if tot["frames"] % 10 == 1:
            print(f"  {f.name} {time.time() - t0:.0f}s {tot}", flush=True)
    (dst.parent / f"xwalk_stripes_{dst.name}.json").write_text(json.dumps({**tot, "seconds": round(time.time() - t0, 1)}, indent=1), encoding="utf-8")
    print(f"totals {tot}  {time.time() - t0:.0f}s"); print("ZZXSTRIPES-DONEZZ")


if __name__ == "__main__":
    main()
