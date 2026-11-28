"""Crosswalk stripes with SAM3 on ZOOMED CROPS (v6sz; PI 2026-09-14 on the zoom test: "84 und zoom crop seem to be good, are we
taking these?").

MEASURED first (stripe_zoom_test.py, night front camera): SAM3 "crosswalk stripe" on a crop around the crossing finds +24 % stripe
pixels at 5-12 m at the same image top-hat contrast as on the full image (14.9 vs gap 10.1), and nothing more beyond ~15 m at night.
Rule, per frame and camera: exactly sam3map_xwalk_stripes.py (full image, "crosswalk stripe" >= 0.4 OR "white stripe on road" >= 0.4,
inside the extraction's crosswalk evidence dilated 31 px; area class 3 -> drivable; stripe pixels on drivable / line / hatched -> 3),
PLUS the same two prompts on crops around the crossings in view: the crossing cells of the delivered map's vote fields (1.5 m
closing of cells with any crosswalk vote, on road) projected into the camera, ground points within ZOOM_MAX_M of the camera; one crop
per view grown 15 %, at least 640 px wide, 16:9, split in two when wider than 1400 px; crop masks OR-ed into the full-image masks.
Points re-lifted exactly as the extractor did. Output ver "v6sz"; per camera stats in stats._xwalk_stripes_zoom.
Usage: sam3map_xwalk_stripes_zoom.py <c8> <src raw dir> <dst raw dir> <fields render dir>   (SAM3MAP_ROOT, SAM3MAP_VIEWS, ZOOM_MAX_M)"""
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, str(HERE))
import numpy as np
import cv2
from PIL import Image
import sam3map_extract_v6 as E
from sam3map_consensus import relift
import gt_reproject as GR

ROOT = Path(os.environ.get("SAM3MAP_ROOT", "/home/nvidia/sam3map/native7"))
VIEWS = os.environ.get("SAM3MAP_VIEWS", "CAM_FW,CAM_CL,CAM_CR,CAM_RL,CAM_RR,CAM_RT,CAM_FT").split(",")
THR = 0.4
ZOOM_MAX_M = float(os.environ.get("ZOOM_MAX_M", "20"))


def crops_for(mask_small, W, H):
    """Crop boxes (full-resolution x0, y0, x1, y1) around the crossing pixels of a 540 x 960 mask."""
    vv, uu = np.nonzero(mask_small)
    u0, u1, v0, v1 = uu.min() * 2, (uu.max() + 1) * 2, vv.min() * 2, (vv.max() + 1) * 2
    boxes = []
    spans = [(u0, u1)] if (u1 - u0) * 1.15 <= 1400 else [(u0, (u0 + u1) // 2 + 60), ((u0 + u1) // 2 - 60, u1)]
    for a, b in spans:
        cw = max(640, int((b - a) * 1.15)); ch = max(int(cw * 9 / 16), int((v1 - v0) * 1.15)); cw = max(cw, int(ch * 16 / 9))
        cw, ch = min(cw, W), min(ch, H)
        cu, cv_ = (a + b) // 2, (v0 + v1) // 2
        x0 = int(np.clip(cu - cw // 2, 0, W - cw)); y0 = int(np.clip(cv_ - ch // 2, 0, H - ch))
        boxes.append((x0, y0, x0 + cw, y0 + ch))
    return boxes


def main():
    c8, src, dst, fdir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]); dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    F = np.load(fdir / "worldmap_fields.npz", allow_pickle=True)
    vk = {k: F[f"vk{k}"].astype(np.float32) for k in (1, 2, 3, 4, 6, 7)}
    dw = vk[1] + vk[2] + vk[3] + vk[4] + vk[6]; drv = (dw > 0) & (dw >= vk[7])
    x_any = (vk[3] > 0) | (F["vraw3"].astype(np.float32) > 0)
    cross = ((cv2.morphologyEx(x_any.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0) & drv).astype(np.uint8)
    origin, res = F["origin"], float(F["res"])
    proc, _ = E.S.build(conf=0.25)
    tot = {"frames": 0, "area_px_removed": 0, "stripe_px_set": 0, "stripe_px_full_only": 0, "stripe_px_with_zoom": 0, "crops": 0}
    for f in sorted(src.glob("[0-9][0-9][0-9].npz")):
        if (dst / f.name).exists():
            continue
        d = dict(np.load(f, allow_pickle=True)); tok = str(d["tok"]); fd = ROOT / f"seq_{c8}" / tok; T = d["T_world_rig"]
        cams = [v for v in VIEWS if f"cls_{v}" in d]
        c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
        grid, fb = E.P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64)); sg = E.GS.smooth_grid(grid, fb)
        per = {}
        for cam in cams:
            img = Image.open(fd / "images" / f"{cam}.jpg").convert("RGB")
            W, H = img.width, img.height
            C = E.CM.Camera.from_calib(c, fr["cam_order"].index(cam), W, H)
            state = proc.set_image(img)
            xs = np.zeros((H, W), bool); ws = np.zeros((H, W), bool)
            for s, m in E.instances(proc, state, "crosswalk stripe", THR):
                xs |= m
            for s, m in E.instances(proc, state, "white stripe on road", THR):
                ws |= m
            ev = d.get(f"evid_{cam}")
            area = np.zeros((H, W), bool)
            if ev is not None:
                area = cv2.dilate(np.repeat(np.repeat(((ev & 2) > 0).astype(np.uint8), 2, axis=0), 2, axis=1)[:H, :W], np.ones((31, 31), np.uint8)) > 0
            full_only = int(((xs | ws) & area).sum())
            # ---- zoomed crops around the crossings in view
            wxy, keep, hw = GR.camera_ground(C, sg, T, 2, lidar=None, ego_small=d.get(f"ego_{cam}"))
            cam_w = T[:2, :2] @ C.t[:2] + T[:2, 3]
            near = np.hypot(wxy[:, 0] - cam_w[0], wxy[:, 1] - cam_w[1]) <= ZOOM_MAX_M
            inx = (GR.lookup(wxy, keep & near, hw, cross, origin, res) > 0)
            n_crops = 0
            if int(inx.sum()) >= 300:
                for (x0, y0, x1, y1) in crops_for(inx, W, H):
                    st_c = proc.set_image(img.crop((x0, y0, x1, y1)))
                    for s, m in E.instances(proc, st_c, "crosswalk stripe", THR):
                        xs[y0:y1, x0:x1] |= m
                    for s, m in E.instances(proc, st_c, "white stripe on road", THR):
                        ws[y0:y1, x0:x1] |= m
                    n_crops += 1
            stripes = (xs | ws) & area
            small = stripes[::2, ::2]
            cl = d[f"cls_{cam}"].copy()
            n_area = int((cl == 3).sum()); cl[cl == 3] = 1
            put = small & np.isin(cl, (1, 2, 6))
            cl[put] = 3
            d[f"cls_{cam}"] = cl; d[f"xstripe_{cam}"] = small
            per[cam] = {"area_px_removed": n_area, "stripe_px_set": int(put.sum()), "crops": n_crops, "stripe_px_full_only": full_only, "stripe_px_with_zoom": int(stripes.sum())}
            tot["area_px_removed"] += n_area; tot["stripe_px_set"] += int(put.sum()); tot["crops"] += n_crops
            tot["stripe_px_full_only"] += full_only; tot["stripe_px_with_zoom"] += int(stripes.sum())
        accp, accr = {k: [] for k in range(1, 8)}, {k: [] for k in range(1, 8)}
        for cam in cams:
            p_, r_ = relift(d[f"cls_{cam}"], E.CM.Camera.from_calib(c, fr["cam_order"].index(cam)), sg, d["T_world_rig"])
            for k in p_:
                accp[k].append(p_[k]); accr[k].append(r_[k])
        for k in range(1, 8):
            d[f"pts_{k}"] = np.concatenate(accp[k]) if accp[k] else np.zeros((0, 2), np.float32)
            d[f"rng_{k}"] = np.concatenate(accr[k]) if accr[k] else np.zeros((0,), np.float16)
        st = json.loads(str(d["stats"])); st["_xwalk_stripes_zoom"] = per; d["stats"] = json.dumps(st); d["ver"] = "v6sz"
        np.savez_compressed(dst / f.name, **d)
        tot["frames"] += 1
        if tot["frames"] % 10 == 1:
            print(f"  {f.name} {time.time() - t0:.0f}s {tot}", flush=True)
    (dst.parent / f"xwalk_stripes_zoom_{dst.name}.json").write_text(json.dumps({**tot, "seconds": round(time.time() - t0, 1)}, indent=1), encoding="utf-8")
    print(f"totals {tot}  {time.time() - t0:.0f}s"); print("ZZXSTRIPESZOOM-DONEZZ")


if __name__ == "__main__":
    main()
