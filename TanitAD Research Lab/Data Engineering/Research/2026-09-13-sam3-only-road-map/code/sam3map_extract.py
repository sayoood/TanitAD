"""SAM3-ONLY road map: combined prompts per camera view -> pixel classes -> ground-lifted BEV points (world frame).

The only SEMANTIC source is SAM3. Geometry (camera calibration, ego poses, LiDAR ground height) only places the
pixels on the ground -- labels may use LiDAR; nothing here is an inference path.

Pixel classes, highest priority last-written:
  1 drivable road surface      road-area prompts, plus every accepted marking (paint lies ON the road)
  2 lane / road line           "lane marking", "road marking", "stop line"
  3 crosswalk                  "crosswalk", "zebra crossing"
  4 painted symbol             "arrow painted on road", "text painted on road"
  5 non-drivable road edge     drivable boundary that touches sidewalk / grass / curb / barrier, plus curb pixels
                               hugging the road
Combination rules (the "smart" part, each measurable later):
  * a marking instance is ACCEPTED only if >= 60 % of its pixels lie on or within 15 px of the road mask AND >= 35 %
    of the ring of pixels around it is road -- paint is surrounded by road; the anonymisation patch on the ego's
    bonnet (accepted as "road marking" by the first rule alone, 2026-09-13) is surrounded by bonnet
  * a crosswalk instance additionally needs score >= 0.5 and <= 15 % of the view's road pixels (SAM3 over-called
    crosswalks as fans/blobs on 2026-09-13 at 12.2 % of the road area)
  * edges come from geometry of the masks, not from a single "edge" prompt
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3vendor"); sys.path.insert(0, "/home/nvidia/sam3paint")
import numpy as np
import cv2
from PIL import Image
import sam3_smoke as S
import sam3_paint as P

V2 = Path("/home/nvidia/qwendrive/v2")
VIEWS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2"]      # B0 (rear tele, 18 % observed) skipped
ROAD_P = {"road": 0.50, "asphalt road surface": 0.50}
NONDRIVE_P = {"sidewalk": 0.40, "grass": 0.40, "curb": 0.40, "guardrail": 0.40}
LINE_P = {"lane marking": 0.40, "road marking": 0.40, "stop line": 0.45}
ZEBRA_P = {"crosswalk": 0.50, "zebra crossing": 0.50}
SYMBOL_P = {"arrow painted on road": 0.45, "text painted on road": 0.45}
OUT_W, OUT_H = 960, 540


def instances(proc, state, prompt, thr):
    out = proc.set_text_prompt(state=state, prompt=prompt)
    if out.get("scores") is None:
        return []
    sc = out["scores"].float().cpu().numpy().reshape(-1)
    if not len(sc):
        return []
    ms = out["masks"].cpu().numpy()
    ms = ms.reshape(len(sc), ms.shape[-2], ms.shape[-1])
    return [(float(s), ms[k]) for k, s in enumerate(sc) if s >= thr]


def classify(proc, img):
    state = proc.set_image(img)
    H, W = img.height, img.width
    road = np.zeros((H, W), bool)
    for p, t in ROAD_P.items():
        for s, m in instances(proc, state, p, t):
            road |= m
    nondrive = np.zeros((H, W), bool); curb = np.zeros((H, W), bool)
    for p, t in NONDRIVE_P.items():
        for s, m in instances(proc, state, p, t):
            nondrive |= m
            if p == "curb":
                curb |= m
    road_zone = cv2.dilate(road.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    road_px = max(int(road.sum()), 1)
    groups = {2: np.zeros((H, W), bool), 3: np.zeros((H, W), bool), 4: np.zeros((H, W), bool)}
    stats = {"accepted": {}, "rejected_offroad": 0, "rejected_zebra_cap": 0}
    for cls, table in ((2, LINE_P), (3, ZEBRA_P), (4, SYMBOL_P)):
        for p, t in table.items():
            for s, m in instances(proc, state, p, t):
                n = int(m.sum())
                if n == 0:
                    continue
                ys, xs = np.nonzero(m)
                y0, y1 = max(0, ys.min() - 8), min(H, ys.max() + 9); x0, x1 = max(0, xs.min() - 8), min(W, xs.max() + 9)
                mc = m[y0:y1, x0:x1].astype(np.uint8)
                ring = (cv2.dilate(mc, np.ones((13, 13), np.uint8)) > 0) & (mc == 0)
                ring_road = float((ring & road[y0:y1, x0:x1]).sum()) / max(int(ring.sum()), 1)
                if ring_road < 0.35 or (m & road_zone).sum() / n < 0.60:
                    stats["rejected_offroad"] += 1       # a marking is surrounded by road; a bonnet patch by bonnet
                    continue
                if cls == 3 and n > 0.15 * road_px:
                    stats["rejected_zebra_cap"] += 1
                    continue
                groups[cls] |= m
                stats["accepted"][p] = stats["accepted"].get(p, 0) + 1
    drivable = road | groups[2] | groups[3] | groups[4]
    k7 = np.ones((7, 7), np.uint8)
    ring = (cv2.dilate(drivable.astype(np.uint8), k7) > 0) & ~drivable
    near_nondrive = cv2.dilate(nondrive.astype(np.uint8), k7) > 0
    edge = (ring & near_nondrive) | (curb & road_zone)
    cls = np.zeros((H, W), np.uint8)
    cls[drivable] = 1
    cls[groups[2]] = 2
    cls[groups[3]] = 3
    cls[groups[4]] = 4
    cls[edge] = 5
    return cls, stats


def ego_body_masks(sd, toks, n=24, thr=7.0):
    """Pixels that do not change across the clip = the ego vehicle's own body (bonnet, mirrors, the anonymisation
    patch on the bonnet). Frames are spread over the whole clip so a stop cannot freeze the scene; only the lower
    40 % of the image is eligible; the mask is closed and dilated by a few pixels."""
    pick = np.linspace(0, len(toks) - 1, n).round().astype(int)
    out = {}
    for cam in VIEWS:
        stack = np.stack([np.asarray(Image.open(sd / toks[k] / "images" / f"{cam}.jpg").convert("L").resize((240, 135)), np.float32) for k in pick])
        std = stack.std(axis=0)
        m = (std < thr) & (stack.mean(axis=0) > 3)                 # black unobserved pixels are not body
        m[: int(0.6 * 135)] = False
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        out[cam] = cv2.resize(m, (1920, 1080), interpolation=cv2.INTER_NEAREST) > 0
    return out


def main():
    c8 = sys.argv[1]
    sel = sys.argv[2] if len(sys.argv) > 2 else "all"
    out = Path(f"/home/nvidia/sam3map/{c8}"); out.mkdir(parents=True, exist_ok=True)
    sd = V2 / f"seq_{c8}"
    toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
    poses = json.loads((sd / "poses.json").read_text())
    idx = range(len(toks)) if sel == "all" else [int(x) for x in sel.split(",")]
    t0 = time.time()
    proc, _ = S.build(conf=0.25)
    print("built %.1fs" % (time.time() - t0), flush=True)
    for j in idx:
        tok = toks[j]; fd = sd / tok
        if (out / f"{j:03d}.npz").exists():
            continue
        c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
        pts = np.load(fd / "lidar.npy").astype(np.float64)
        grid, fb = P.ground_grid(pts)
        T = np.array(poses[tok]["T_world_rig"])
        lifted = {k: [] for k in range(1, 6)}
        small = {}
        stats_all = {}
        for cam in VIEWS:
            i = fr["cam_order"].index(cam)
            img = Image.open(fd / "images" / f"{cam}.jpg").convert("RGB")
            cls, st = classify(proc, img)
            stats_all[cam] = st
            small[cam] = cv2.resize(cls, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)
            K, R, t = c["cam_intrinsic"][i], c["sensor2lidar_rotation"][i], c["sensor2lidar_translation"][i]
            for k in range(1, 6):
                m = cls == k
                if k == 1:
                    m = cls >= 1                         # every marking pixel is also drivable surface
                if m.any():
                    xy = P.lift(m, K, R, t, grid, fb, stride=2)
                    if len(xy):
                        lifted[k].append((np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ T.T)[:, :2].astype(np.float32))
        np.savez_compressed(out / f"{j:03d}.npz", tok=tok, T_world_rig=T, stats=json.dumps(stats_all),
                            **{f"cls_{cam}": small[cam] for cam in VIEWS},
                            **{f"pts_{k}": (np.concatenate(lifted[k]) if lifted[k] else np.zeros((0, 2), np.float32)) for k in range(1, 6)})
        acc = {cam: stats_all[cam]["accepted"] for cam in ("CAM_F0", "CAM_L2")}
        print(f"  j={j:3d} {time.time() - t0:6.0f}s  pts " + " ".join(f"{k}:{sum(len(a) for a in lifted[k])}" for k in range(1, 6)) + f"  F0/L2 accepted {acc}", flush=True)
    print("ZZSAM3MAP-DONEZZ %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
