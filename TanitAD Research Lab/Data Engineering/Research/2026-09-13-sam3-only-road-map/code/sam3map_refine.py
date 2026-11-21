"""v2 of the SAM3-only map, derived EXACTLY from v1's stored per-view class rasters (the road classes are not re-segmented).

Why exact: v1 stored cls_<cam> at 960x540 by nearest resize, i.e. full-res pixel (2i, 2j) -- precisely the pixel P.lift
samples at stride 2. Re-lifting the stored raster must therefore reproduce v1's BEV points bit for bit; that CONTROL is
asserted on every frame before any rule is applied, and a frame that fails it is not refined.

Rules (written down 2026-09-13 before any v2 score exists; v1 showed each defect in the rendered video):
  R1 ego bonnet     SAM3 "car hood" on K = 8 frames spread over the clip, CAM_F0 only; instances with score >= 0.6,
                    >= 20 000 px, whose lowest row lies in the bottom 10 % of the image; per-pixel majority over the K
                    frames, dilated by 5 px at 960x540 -> class 0 in every frame (v1 labelled the bonnet as road)
  R2 ego patch      CAM_F0 marking components (classes 2-4, 8-connected) overlapping the PATCH ZONE -> class 0 (v1
                    accepted the anonymisation patch on the bonnet as a lane line). Patch zone = the 60 px (960x540)
                    above the bonnet's top edge, in the CENTRAL 20 % of columns, where a marking occurs in >= 3 frames
                    whose first and last occurrence are >= 30 % of the clip apart (paint passes through the band in
                    consecutive frames; a patch fixed to the car recurs at the same pixels all clip long).
                    REVISED before any score: the first form (marking in >= 20 % of frames, whole bonnet width) was
                    INERT on the day clip -- the patch is labelled in only 10/96 frames (40-42, 60, 65, 75, 86, 90,
                    91, 95); measured: the revised zone is one 28 x 24 px blob at the patch, and the outer 80 % of
                    columns (where lane lines meet the bonnet) reach a count of only 2 / 1 and form no zone.
                    REVISED AGAIN (v2.1), after looking at what it removed on the night clip: "overlaps the zone"
                    deleted REAL crosswalk stripes and a stop line in frames 83-90 (components of 4.6-11.5k px merely
                    touching a 592 px zone). Now a component is removed only if >= 50 % of it lies inside the zone
                    dilated by 2 px -- the ego artefacts (a 28 x 24 px patch; a ~550 px reflection line in night
                    frames 0-28) sit wholly inside it, real paint does not.
  R3 thin edges     class-5 pixels survive only if 4-adjacent to a class 1-4 pixel (v1 edges became red blobs in BEV)
Output: /home/nvidia/sam3map/<c8>_v2/<jjj>.npz in v1's format, plus refine counts in `stats`.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3vendor"); sys.path.insert(0, "/home/nvidia/sam3paint"); sys.path.insert(0, "/home/nvidia/sam3map")
import numpy as np
import cv2
from PIL import Image
import sam3_paint as P
from sam3map_extract import instances, VIEWS, V2

K_FRAMES, HOOD_SCORE, HOOD_MIN_PX, HOOD_BOTTOM = 8, 0.6, 20_000, 0.90
DILATE_PX, BAND_PX, PATCH_MIN_FRAMES, PATCH_SPREAD, PATCH_COLS = 5, 60, 3, 0.30, (0.40, 0.60)
PATCH_INSIDE = 0.50      # a component is an ego artefact only if >= half of it lies in the (2 px dilated) zone


def relift(cls_small, K, R, t, grid, fb, T):
    full = np.repeat(np.repeat(cls_small, 2, axis=0), 2, axis=1)          # full[2i, 2j] == cls_small[i, j]
    out = {}
    for k in range(1, 6):
        m = (full >= 1) if k == 1 else (full == k)
        if m.any():
            xy = P.lift(m, K, R, t, grid, fb, stride=2)
            if len(xy):
                out[k] = (np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ T.T)[:, :2].astype(np.float32)
    return out


def bonnet_mask(c8, toks, sd):
    import sam3_smoke as S
    proc, _ = S.build(conf=0.25)
    pick = np.linspace(0, len(toks) - 1, K_FRAMES).round().astype(int)
    votes = np.zeros((540, 960), np.int16); found = []
    for j in pick:
        img = Image.open(sd / toks[j] / "images" / "CAM_F0.jpg").convert("RGB")
        state = proc.set_image(img)
        m_all = np.zeros((img.height, img.width), bool); best = 0.0
        for s, m in instances(proc, state, "car hood", HOOD_SCORE):
            ys = np.nonzero(m.any(axis=1))[0]
            if m.sum() >= HOOD_MIN_PX and len(ys) and ys.max() >= HOOD_BOTTOM * img.height:
                m_all |= m; best = max(best, s)
        found.append((int(j), round(best, 2), int(m_all.sum())))
        votes += m_all[::2, ::2]
    mask = votes >= (K_FRAMES + 1) // 2
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((2 * DILATE_PX + 1, 2 * DILATE_PX + 1), np.uint8)) > 0
    del proc
    return mask, found


def main():
    c8 = sys.argv[1]
    src = Path(f"/home/nvidia/sam3map/{c8}"); dst = Path(f"/home/nvidia/sam3map/{c8}_v2"); dst.mkdir(exist_ok=True)
    sd = V2 / f"seq_{c8}"
    toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
    files = sorted(src.glob("*.npz"))
    t0 = time.time()
    bonnet, found = bonnet_mask(c8, toks, sd)
    print(f"bonnet: {int(bonnet.sum())} px at 960x540 ({100 * bonnet.mean():.1f} % of F0); per sampled frame (j, score, px): {found}", flush=True)
    # patch zone from the clip's own marking frequency, after R1
    cols = np.nonzero(bonnet.any(axis=0))[0]
    band = np.zeros_like(bonnet)
    for x in cols:
        top = int(np.argmax(bonnet[:, x]))
        band[max(0, top - BAND_PX): top, x] = True
    band[:, : int(PATCH_COLS[0] * 960)] = False; band[:, int(PATCH_COLS[1] * 960):] = False
    N = len(files)
    cnt = np.zeros((540, 960), np.int32); first = np.full((540, 960), N, np.int32); last = np.full((540, 960), -1, np.int32)
    for n, f in enumerate(files):
        c = np.load(f)["cls_CAM_F0"]
        m = (c >= 2) & (c <= 4) & ~bonnet
        cnt += m; first[m & (first == N)] = n; last[m] = n
    spread = np.where(cnt > 0, last - first, 0)
    zone = band & (cnt >= PATCH_MIN_FRAMES) & (spread >= PATCH_SPREAD * N)
    zone_d = cv2.dilate(zone.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ys, xs = np.nonzero(zone)
    print(f"patch zone: {int(zone.sum())} px, bbox rows/cols {(int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())) if len(ys) else None}; "
          f"max marking count in the band {int(cnt[band].max()) if band.any() else 0}", flush=True)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    tot = {"control_identical": 0, "control_failed": 0, "R1_px": 0, "R2_components": 0, "R2_px": 0, "R3_px": 0}
    for f in files:
        d = dict(np.load(f, allow_pickle=True))
        tok = str(d["tok"]); fd = sd / tok; T = d["T_world_rig"]
        c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
        grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64))
        cams = {cam: (c["cam_intrinsic"][fr["cam_order"].index(cam)], c["sensor2lidar_rotation"][fr["cam_order"].index(cam)],
                      c["sensor2lidar_translation"][fr["cam_order"].index(cam)]) for cam in VIEWS}
        # ---- CONTROL: re-lifting v1's rasters must give v1's points exactly
        acc = {k: [] for k in range(1, 6)}
        for cam in VIEWS:
            for k, v in relift(d[f"cls_{cam}"], *cams[cam], grid, fb, T).items():
                acc[k].append(v)
        same = all(np.array_equal(np.concatenate(acc[k]) if acc[k] else np.zeros((0, 2), np.float32), d[f"pts_{k}"]) for k in range(1, 6))
        if not same:
            tot["control_failed"] += 1
            print(f"  CONTROL FAILED on {f.name}: not refined", flush=True)
            continue
        tot["control_identical"] += 1
        rec = {}
        new = {k: [] for k in range(1, 6)}
        for cam in VIEWS:
            cl = d[f"cls_{cam}"].copy()
            r1 = r2c = r2 = 0
            if cam == "CAM_F0":
                r1 = int((cl[bonnet] > 0).sum()); cl[bonnet] = 0
                mk = ((cl >= 2) & (cl <= 4)).astype(np.uint8)
                n, lab = cv2.connectedComponents(mk, connectivity=8)
                size = np.bincount(lab.ravel(), minlength=n)
                inside = np.bincount(lab[zone_d].ravel(), minlength=n)
                hit = np.nonzero((inside >= PATCH_INSIDE * size) & (np.arange(n) > 0))[0]
                if len(hit):
                    kill = np.isin(lab, hit); r2c = int(len(hit)); r2 = int(kill.sum()); cl[kill] = 0
            e = cl == 5
            keep = e & (cv2.dilate(((cl >= 1) & (cl <= 4)).astype(np.uint8), cross) > 0)
            r3 = int((e & ~keep).sum()); cl[e & ~keep] = 0
            d[f"cls_{cam}"] = cl
            rec[cam] = {"R1_px": r1, "R2_components": r2c, "R2_px": r2, "R3_px": r3}
            tot["R1_px"] += r1; tot["R2_components"] += r2c; tot["R2_px"] += r2; tot["R3_px"] += r3
            for k, v in relift(cl, *cams[cam], grid, fb, T).items():
                new[k].append(v)
        st = json.loads(str(d["stats"])); st["_refine_v2"] = rec
        d["stats"] = json.dumps(st)
        for k in range(1, 6):
            d[f"pts_{k}"] = np.concatenate(new[k]) if new[k] else np.zeros((0, 2), np.float32)
        np.savez_compressed(dst / f.name, **d)
    json.dump({"bonnet_px_960x540": int(bonnet.sum()), "bonnet_sampled": found, "patch_zone_px": int(zone.sum()), **tot,
               "rules": {"K_FRAMES": K_FRAMES, "HOOD_SCORE": HOOD_SCORE, "HOOD_MIN_PX": HOOD_MIN_PX, "HOOD_BOTTOM": HOOD_BOTTOM,
                         "DILATE_PX": DILATE_PX, "BAND_PX": BAND_PX, "PATCH_MIN_FRAMES": PATCH_MIN_FRAMES,
                         "PATCH_SPREAD": PATCH_SPREAD, "PATCH_COLS": PATCH_COLS, "PATCH_INSIDE": PATCH_INSIDE}},
              open(dst.parent / f"refine_{c8}.json", "w"), indent=1)
    print(f"totals {tot}  {time.time() - t0:.0f}s", flush=True)
    print("ZZREFINE-DONEZZ")


if __name__ == "__main__":
    main()
