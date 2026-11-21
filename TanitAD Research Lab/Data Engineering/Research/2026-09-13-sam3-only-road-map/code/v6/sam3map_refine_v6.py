"""v6 refinement of the SAM3-only map on ALL NATIVE cameras, derived exactly from the extractor's stored rasters.

CONTROL (unchanged contract): re-lifting the stored 960x540 rasters through camera_model + the smooth ground must
reproduce the extractor's points AND ranges bit for bit on every frame before any rule applies.
  R1 ego body   per camera, SAM3 on K = 8 frames spread over the clip: "car hood" instances reaching the bottom 10 % of
                the image (>= 20 000 px), and "car" instances touching the image border; a pixel covered in >= 5 of the 8
                frames is the ego vehicle (the native rig shows the bonnet in CAM_FW and body panels at the edges of
                CAM_RL / CAM_RR; other cars do not hold the same pixels for 20 s). Dilated 5 px (960x540) -> class 0.
  R2 ego patch  front-wide camera only, the v3 rule (paint recurring in the band above the bonnet while the ego moved)
  R3 thin edges edge pixels survive only if 4-adjacent to a drivable-or-paint pixel
Classes 1-7 (7 = sidewalk / verge). Output /home/nvidia/sam3map/<c8>_<tag>/ (tag from SAM3MAP_TAG, default v6).
"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3vendor"); sys.path.insert(0, "/home/nvidia/sam3paint"); sys.path.insert(0, "/home/nvidia/sam3map")
import numpy as np
import cv2
from PIL import Image
import sam3_paint as P
import camera_model as CM
import ground_surface as GS
from sam3map_extract import instances

ROOT = Path(os.environ.get("SAM3MAP_ROOT", "/home/nvidia/sam3map/native7"))
VIEWS = os.environ.get("SAM3MAP_VIEWS", "CAM_FW,CAM_CL,CAM_CR,CAM_RL,CAM_RR,CAM_RT,CAM_FT").split(",")
FRONT = [v for v in VIEWS if v in ("CAM_FW", "CAM_F0")][:1]
PAINT, DRIVE = (2, 3, 4, 6), (1, 2, 3, 4, 6)
K_FRAMES, VOTE, DIL = 8, 5, 5
BAND_PX, PATCH_MIN_FRAMES, PATCH_SPREAD, PATCH_COLS, PATCH_INSIDE, MOVE_M = 60, 3, 0.30, (0.40, 0.60), 0.50, 8.0


def relift(cls_small, cam, sgrid, T):
    full = np.repeat(np.repeat(cls_small, 2, axis=0), 2, axis=1)
    pts, rng = {}, {}
    for k in range(1, 8):
        m = np.isin(full, DRIVE) if k == 1 else (full == k)
        if m.any():
            xy, r = cam.lift(m, sgrid, stride=2)
            if len(xy):
                pts[k] = (np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ T.T)[:, :2].astype(np.float32)
                rng[k] = r.astype(np.float16)
    return pts, rng


def ego_masks(sd, toks):
    import sam3_smoke as S
    proc, _ = S.build(conf=0.25)
    pick = np.linspace(0, len(toks) - 1, K_FRAMES).round().astype(int)
    poses = json.loads((sd / "poses.json").read_text())
    Pp = np.array([np.array(poses[toks[j]]["T_world_rig"])[:2, 3] for j in pick])
    spread_m = float(np.max(np.hypot(Pp[:, None, 0] - Pp[None, :, 0], Pp[:, None, 1] - Pp[None, :, 1])))
    use_car = spread_m >= 15.0          # a stopped ego would make parked cars look static: then trust only "car hood"
    out, info = {"_spread_m": round(spread_m, 1)}, {"_spread_m": round(spread_m, 1), "_car_rule": use_car}
    for cam in VIEWS:
        votes = np.zeros((540, 960), np.int16)
        for j in pick:
            img = Image.open(sd / toks[j] / "images" / f"{cam}.jpg").convert("RGB")
            state = proc.set_image(img)
            H, W = img.height, img.width
            m_all = np.zeros((H, W), bool)
            for s, m in instances(proc, state, "car hood", 0.6):
                ys = np.nonzero(m.any(axis=1))[0]
                if m.sum() >= 20_000 and len(ys) and ys.max() >= 0.9 * H:
                    m_all |= m
            for s, m in (instances(proc, state, "car", 0.5) if use_car else []):
                if m[:3].any() or m[-3:].any() or m[:, :3].any() or m[:, -3:].any():
                    m_all |= m
            votes += m_all[::2, ::2]
        mask = votes >= VOTE
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((2 * DIL + 1, 2 * DIL + 1), np.uint8)) > 0
        out[cam] = mask; info[cam] = round(100 * float(mask.mean()), 2)
    del proc
    return out, info


def main():
    c8 = sys.argv[1]; tag = os.environ.get("SAM3MAP_TAG", "v6")
    src = Path(f"/home/nvidia/sam3map/{c8}_{tag}raw"); dst = Path(f"/home/nvidia/sam3map/{c8}_{tag}"); dst.mkdir(exist_ok=True)
    sd = ROOT / f"seq_{c8}"
    toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
    files = sorted(src.glob("[0-9][0-9][0-9].npz"))
    t0 = time.time()
    ego, ego_pct = ego_masks(sd, toks)
    ego.pop("_spread_m", None)
    print(f"ego body mask, % of each view: {ego_pct}", flush=True)
    zone_d = np.zeros((540, 960), bool)
    if FRONT:
        fc = FRONT[0]; em = ego[fc]
        band = np.zeros_like(em)
        for x in np.nonzero(em.any(axis=0))[0]:
            top = int(np.argmax(em[:, x])); band[max(0, top - BAND_PX): top, x] = True
        band[:, : int(PATCH_COLS[0] * 960)] = False; band[:, int(PATCH_COLS[1] * 960):] = False
        N = len(files); pos = np.zeros((N, 2))
        cnt = np.zeros((540, 960), np.int32); first = np.full((540, 960), N, np.int32); last = np.full((540, 960), -1, np.int32)
        for n, f in enumerate(files):
            d = np.load(f, allow_pickle=True); pos[n] = d["T_world_rig"][:2, 3]
            m = np.isin(d[f"cls_{fc}"], PAINT) & ~em
            cnt += m; first[m & (first == N)] = n; last[m] = n
        spread = np.where(cnt > 0, last - first, 0)
        fi, la = first.clip(0, N - 1), last.clip(0, N - 1)
        moved = np.hypot(pos[la, 0] - pos[fi, 0], pos[la, 1] - pos[fi, 1]) * (cnt > 0)
        zone = band & (cnt >= PATCH_MIN_FRAMES) & ((spread >= PATCH_SPREAD * N) | (moved >= MOVE_M))
        zone_d = cv2.dilate(zone.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        print(f"patch zone ({fc}): {int(zone.sum())} px", flush=True)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    tot = {"control_identical": 0, "control_failed": 0, "R1_px": 0, "R2_components": 0, "R2_px": 0, "R3_px": 0}
    cat = lambda lst, shape, dt: np.concatenate(lst) if lst else np.zeros(shape, dt)
    for f in files:
        d = dict(np.load(f, allow_pickle=True))
        tok = str(d["tok"]); fd = sd / tok; T = d["T_world_rig"]
        c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
        grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64))
        sgrid = GS.smooth_grid(grid, fb)
        cams = {cam: CM.Camera.from_calib(c, fr["cam_order"].index(cam)) for cam in VIEWS}
        accp, accr = {k: [] for k in range(1, 8)}, {k: [] for k in range(1, 8)}
        for cam in VIEWS:
            p_, r_ = relift(d[f"cls_{cam}"], cams[cam], sgrid, T)
            for k in p_:
                accp[k].append(p_[k]); accr[k].append(r_[k])
        if not all(np.array_equal(cat(accp[k], (0, 2), np.float32), d[f"pts_{k}"]) and np.array_equal(cat(accr[k], (0,), np.float16), d[f"rng_{k}"])
                   for k in range(1, 8)):
            tot["control_failed"] += 1; print(f"  CONTROL FAILED on {f.name}: not refined", flush=True); continue
        tot["control_identical"] += 1
        rec = {}; newp, newr = {k: [] for k in range(1, 8)}, {k: [] for k in range(1, 8)}
        for cam in VIEWS:
            cl = d[f"cls_{cam}"].copy()
            r1 = int((cl[ego[cam]] > 0).sum()); cl[ego[cam]] = 0
            r2c = r2 = 0
            if FRONT and cam == FRONT[0]:
                mk = np.isin(cl, PAINT).astype(np.uint8)
                n, lab = cv2.connectedComponents(mk, connectivity=8)
                size = np.bincount(lab.ravel(), minlength=n); inside = np.bincount(lab[zone_d].ravel(), minlength=n)
                hit = np.nonzero((inside >= PATCH_INSIDE * size) & (np.arange(n) > 0))[0]
                if len(hit):
                    kill = np.isin(lab, hit); r2c = int(len(hit)); r2 = int(kill.sum()); cl[kill] = 0
            e = cl == 5
            keep = e & (cv2.dilate(np.isin(cl, DRIVE).astype(np.uint8), cross) > 0)
            r3 = int((e & ~keep).sum()); cl[e & ~keep] = 0
            d[f"cls_{cam}"] = cl
            rec[cam] = {"R1_px": r1, "R2_components": r2c, "R2_px": r2, "R3_px": r3}
            for q, v in (("R1_px", r1), ("R2_components", r2c), ("R2_px", r2), ("R3_px", r3)):
                tot[q] += v
            p_, r_ = relift(cl, cams[cam], sgrid, T)
            for k in p_:
                newp[k].append(p_[k]); newr[k].append(r_[k])
        st = json.loads(str(d["stats"])); st["_refine_v2"] = rec
        d["stats"] = json.dumps(st)
        for cam in VIEWS:
            d[f"ego_{cam}"] = ego[cam]                 # the ego body per camera: "not seen" for the BEV / ground truth (not "background")
        for k in range(1, 8):
            d[f"pts_{k}"] = cat(newp[k], (0, 2), np.float32); d[f"rng_{k}"] = cat(newr[k], (0,), np.float16)
        np.savez_compressed(dst / f.name, **d)
    json.dump({"ego_mask_pct": ego_pct, **tot}, open(dst.parent / f"refine_{tag}_{c8}.json", "w"), indent=1)
    print(f"totals {tot}  {time.time() - t0:.0f}s", flush=True)
    print("ZZREFINE6-DONEZZ")


if __name__ == "__main__":
    main()
