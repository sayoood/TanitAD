"""SAM3 road-paint PROMPT BANK: every instance of every prompt at a low score floor, so prompt subsets, confidence
thresholds and cross-class competition can be swept OFFLINE without a second SAM3 pass.

demo  the nuPlan demo frames that carry the TRUE map (classes 2 road_line, 4 crosswalk) -> per-instance lifted BEV
      rasters (0.15 m, packed), the true map and the observed-ground mask, one npz per frame
ours  chosen frames of our clips -> per-instance 960x540 masks (packed), one npz per frame
Every instance also carries the extractor's on-road statistics (share within 15 px of the road mask; share of its
13 px ring that is road), so the v1 acceptance rule can be replayed offline.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3vendor"); sys.path.insert(0, "/home/nvidia/sam3paint"); sys.path.insert(0, "/home/nvidia/sam3map")
import numpy as np
import cv2
from PIL import Image
import sam3_smoke as S
import sam3_paint as P

BANK = {
    "line": ["lane marking", "road marking", "stop line", "dashed line", "solid line", "white line on road", "yellow line on road"],
    "crosswalk": ["crosswalk", "zebra crossing", "pedestrian crossing", "crosswalk stripes"],
    "hatch": ["hatched road marking", "chevron road marking", "diagonal stripes on road", "painted traffic island"],
    "symbol": ["arrow painted on road", "road arrow", "text painted on road", "word on road surface",
               "bicycle symbol painted on road", "triangle painted on road"],
    "road": ["road"],
}
FLOOR = 0.20
OUT = Path("/home/nvidia/sam3map/promptbank")


def image_instances(proc, img):
    state = proc.set_image(img)
    inst = []
    for fam, prompts in BANK.items():
        for pr in prompts:
            o = proc.set_text_prompt(state=state, prompt=pr)
            if o.get("scores") is None:
                continue
            sc = o["scores"].float().cpu().numpy().reshape(-1)
            if not len(sc):
                continue
            ms = o["masks"].cpu().numpy().reshape(len(sc), img.height, img.width)
            for k in np.flatnonzero(sc >= FLOOR):
                if ms[k].any():
                    inst.append((fam, pr, float(sc[k]), ms[k].astype(bool)))
    road = np.zeros((img.height, img.width), bool)
    for fam, pr, s, m in inst:
        if fam == "road" and s >= 0.5:
            road |= m
    zone = cv2.dilate(road.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    rows = []
    for fam, pr, s, m in inst:
        n = int(m.sum())
        ys, xs = np.nonzero(m)
        y0, y1 = max(0, ys.min() - 8), min(img.height, ys.max() + 9); x0, x1 = max(0, xs.min() - 8), min(img.width, xs.max() + 9)
        mc = m[y0:y1, x0:x1].astype(np.uint8)
        ring = (cv2.dilate(mc, np.ones((13, 13), np.uint8)) > 0) & (mc == 0)
        rows.append({"fam": fam, "prompt": pr, "score": round(s, 4), "px": n,
                     "onroad": round(float((m & zone).sum()) / max(n, 1), 4),
                     "ring_road": round(float((ring & road[y0:y1, x0:x1]).sum()) / max(int(ring.sum()), 1), 4)})
    return inst, rows


def run_demo(proc):
    D = Path("/home/nvidia/qwendrive/qwen-drive/data/demo/perception")
    for fd in sorted(p for p in D.iterdir() if p.is_dir()):
        fr = json.loads((fd / "frame.json").read_text())
        if fr["dataset_type"] != "nuplan":
            continue
        c = np.load(fd / "calib.npz"); g = np.load(fd / "gt.npz")
        grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64))
        Ks, Rs, ts, sizes, meta, rasters = [], [], [], [], [], []
        for i, cam in enumerate(fr["cam_order"]):
            img = Image.open(fd / "images" / f"{cam}.jpg").convert("RGB")
            K, R, t = c["cam_intrinsic"][i], c["sensor2lidar_rotation"][i], c["sensor2lidar_translation"][i]
            Ks.append(K); Rs.append(R); ts.append(t); sizes.append(img.size)
            inst, rows = image_instances(proc, img)
            for (fam, pr, s, m), row in zip(inst, rows):
                row["cam"] = cam
                meta.append(row)
                rasters.append(np.packbits(P.raster(P.lift(m, K, R, t, grid, fb))))
        observed = P.observed_ground(Ks, Rs, ts, sizes, grid, fb)
        np.savez_compressed(OUT / "demo" / f"{fd.name}.npz", meta=json.dumps(meta), rasters=np.stack(rasters),
                            gt=g["map"].astype(np.int8), observed=observed)
        print(f"  demo {fd.name[:8]}: {len(meta)} instances", flush=True)


def run_ours(proc, clips):
    views = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2"]
    for c8, js in clips.items():
        sd = Path("/home/nvidia/qwendrive/v2") / f"seq_{c8}"
        toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
        for j in js:
            meta, masks = [], []
            for cam in views:
                img = Image.open(sd / toks[j] / "images" / f"{cam}.jpg").convert("RGB")
                inst, rows = image_instances(proc, img)
                for (fam, pr, s, m), row in zip(inst, rows):
                    row["cam"] = cam
                    meta.append(row)
                    masks.append(np.packbits(m[::2, ::2]))            # the exact samples P.lift takes at stride 2
            np.savez_compressed(OUT / "ours" / f"{c8}_{j:03d}.npz", meta=json.dumps(meta), masks=np.stack(masks), tok=toks[j])
            print(f"  ours {c8[:2]}.. j={j}: {len(meta)} instances", flush=True)


if __name__ == "__main__":
    (OUT / "demo").mkdir(parents=True, exist_ok=True); (OUT / "ours").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc, _ = S.build(conf=FLOOR)
    run_demo(proc)
    print(f"demo done {time.time() - t0:.0f}s", flush=True)
    run_ours(proc, {"4fbd97b6a4b7": [5, 10, 20, 35, 48, 60, 75, 95], "73495082f98b": [30, 60, 83, 89]})
    print(f"ZZBANK-DONEZZ {time.time() - t0:.0f}s")
