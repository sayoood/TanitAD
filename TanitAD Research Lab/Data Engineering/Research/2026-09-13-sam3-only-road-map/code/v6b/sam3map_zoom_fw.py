"""v6z: a 2x ZOOM TILE of the native front camera for SAM3, pasted into the v6 raw front raster (label artifact).

Why (MEASURED 2026-09-13, a2_by_camera.py): at night the native CAM_FW road mask puts 2.81 % of LiDAR tall-obstacle points
on drivable at 8-15 m, 2.0x the 1.43 % of v2's virtual 63.7-deg crop of the same camera (0-8 m and 15-30 m are equal),
and the v6 night clip vote FAILED its bar on MAP_A2 (0.1429 > 0.0664 + 0.05). The crop acted as a zoom: SAM3 saw the
far road with ~2x the pixels. Here SAM3 runs a second time per frame on the central 960x540 of CAM_FW upsampled to
1920x1080 (the extractor's own classify, with the lifting mapped back to full-image pixels); its classes replace the
full-image classes inside that region (5 % border excluded). All other cameras and the evidence bits are unchanged.

PRE-REGISTERED 2026-09-13 before its first run: v6z WORKS if (a) on the night clip the v6z clip vote MAP_A2 <= its own
SAM3_fused_1s + 0.05 (the bar v6 failed), (b) on the day clip the clip-vote MAP_A2 rises by no more than 0.01 over v6
(0.0225), and (c) MAP_B >= 0.95 on both clips. Anything else is a FAIL, reported as such. The night GT world map (v6.1
pipeline on v6z) is reported beside it.
RESULT (MEASURED, 96 frames each): FAIL -- night v6z clip vote A2 0.1430 (v6 0.1429; bar 0.0661 + 0.05), B 0.9732; day
identical to v6 to 4 decimals (A2 0.0225, B 0.9841); night GT world map A2 0.0853 (v6.1 0.0854). The zoom changed ~0.3 %
of CAM_FW pixels per frame (smoke token 60: 1,574 px) and no metric: magnification of the fisheye image is REFUTED as the
cause of the night spill. Next lever: rectification (sam3map_rectified_road.py).
Usage: sam3map_zoom_fw.py <c8> <src raw dir> <dst raw dir>   (SAM3MAP_ROOT = native sequence root)
"""
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, str(HERE))
import numpy as np
import cv2
from PIL import Image
import sam3map_extract_v6 as E                               # classify, S (sam3_smoke), P, CM, GS
from sam3map_consensus import relift

ROOT = Path(os.environ.get("SAM3MAP_ROOT", "/home/nvidia/sam3map/native7"))
X0, Y0, CW, CH = 480, 330, 960, 540                          # full-res crop box of CAM_FW (the far road sits just below the horizon)
MARGIN_X, MARGIN_Y = 24, 13                                  # 5 % of the pasted 480x270 region


class ZoomLift:
    """Lifting for masks in ZOOMED crop pixels: embed at full-image pixels, then the real camera model."""
    def __init__(self, cam):
        self.cam = cam

    def lift(self, mask, sgrid, stride=2, max_xy=(30.0, 15.0)):
        full = np.zeros((1080, 1920), bool)
        full[Y0:Y0 + CH, X0:X0 + CW] = mask[::2, ::2]
        return self.cam.lift(full, sgrid, stride=stride, max_xy=max_xy)


def main():
    c8, src, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc, _ = E.S.build(conf=0.25)
    sd = ROOT / f"seq_{c8}"
    tot = {"frames": 0, "fw_px_changed": 0, "fw_drivable_px_before": 0, "fw_drivable_px_after": 0}
    for f in sorted(src.glob("[0-9][0-9][0-9].npz")):
        if (dst / f.name).exists():
            continue
        d = dict(np.load(f, allow_pickle=True))
        tok = str(d["tok"]); fd = sd / tok; T = d["T_world_rig"]
        c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
        grid, fb = E.P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64)); sg = E.GS.smooth_grid(grid, fb)
        img = Image.open(fd / "images" / "CAM_FW.jpg").convert("RGB")
        cam = E.CM.Camera.from_calib(c, fr["cam_order"].index("CAM_FW"), img.width, img.height)
        zimg = img.crop((X0, Y0, X0 + CW, Y0 + CH)).resize((1920, 1080), Image.BICUBIC)
        cls_z, st_z, _ = E.classify(proc, zimg, (ZoomLift(cam), sg), True)
        small = cv2.resize(cls_z, (CW // 2, CH // 2), interpolation=cv2.INTER_NEAREST)       # 480x270 in stored-raster pixels
        cl = d["cls_CAM_FW"].copy(); before = cl.copy()
        ys, xs = Y0 // 2, X0 // 2
        cl[ys + MARGIN_Y: ys + CH // 2 - MARGIN_Y, xs + MARGIN_X: xs + CW // 2 - MARGIN_X] = small[MARGIN_Y: CH // 2 - MARGIN_Y, MARGIN_X: CW // 2 - MARGIN_X]
        d["cls_CAM_FW"] = cl
        drive = (1, 2, 3, 4, 6)
        tot["fw_px_changed"] += int((cl != before).sum()); tot["frames"] += 1
        tot["fw_drivable_px_before"] += int(np.isin(before, drive).sum()); tot["fw_drivable_px_after"] += int(np.isin(cl, drive).sum())
        accp, accr = {k: [] for k in range(1, 8)}, {k: [] for k in range(1, 8)}
        cams = [k[4:] for k in d if k.startswith("cls_")]
        order = [v for v in os.environ.get("SAM3MAP_VIEWS", "CAM_FW,CAM_CL,CAM_CR,CAM_RL,CAM_RR,CAM_RT,CAM_FT").split(",") if v in cams]
        for cm_ in order:                                                                  # the extractor's camera order
            C = E.CM.Camera.from_calib(c, fr["cam_order"].index(cm_))
            p_, r_ = relift(d[f"cls_{cm_}"], C, sg, T)
            for k in p_:
                accp[k].append(p_[k]); accr[k].append(r_[k])
        for k in range(1, 8):
            d[f"pts_{k}"] = np.concatenate(accp[k]) if accp[k] else np.zeros((0, 2), np.float32)
            d[f"rng_{k}"] = np.concatenate(accr[k]) if accr[k] else np.zeros((0,), np.float16)
        st = json.loads(str(d["stats"])); st["_zoom_fw"] = {"box_fullres": [X0, Y0, CW, CH], "zoom_stats": st_z}; d["stats"] = json.dumps(st)
        d["ver"] = "v6z"
        np.savez_compressed(dst / f.name, **d)
        if tot["frames"] % 20 == 1:
            print(f"  {f.name} {time.time() - t0:.0f}s {tot}", flush=True)
    (dst.parent / f"zoom_fw_{dst.name}.json").write_text(json.dumps({**tot, "seconds": round(time.time() - t0, 1)}, indent=1), encoding="utf-8")
    print(f"totals {tot}  {time.time() - t0:.0f}s"); print("ZZZOOMFW-DONEZZ")


if __name__ == "__main__":
    main()
