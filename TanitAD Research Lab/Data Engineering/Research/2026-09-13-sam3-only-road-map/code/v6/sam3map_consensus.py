"""R4 -- clip consensus for PAINTED AREAS (v6.1): every painted region of the world gets ONE class for the whole clip.

Why (MEASURED 2026-09-13, contact sheet of the day clip token 41 against the raw images): the per-image rules gave one
white hatched gore area three semantics -- crosswalk on CAM_RL and CAM_RR, lane lines on CAM_RT -- dashed lane-line
segments on CAM_CR became hatched, and on front clip 1f1f05ca011d a row of yield triangles became hatched. The per-image
ground test could not run where SAM3's line prompt missed the stripes ("too_few" -> prompt fallback), and one oblique
view rarely sees a whole region.
  pass A  every camera of every frame is warped onto the ground (the v5 metric layer: camera model, smooth ground, MIP
          luminance, 1.5 m white top-hat) and accumulated in ONE world grid at 0.10 m: observation weight, bright paint,
          the weighted votes of the SAM3 classes (crosswalk, hatched, line) and of the prompt evidence (crosswalk prompt,
          diagonal-stripes prompt). Weight = camera factor / (1 + (range / 8 m)^2).
  pass B  regions = cells whose area-paint votes are >= 35 % of their observation weight (closed 0.5 m, >= 1 m^2). On the
          FUSED paint of all views: a row of short blobs (yield teeth) -> road line; thin fragments along the region axis
          (dashed or solid line) -> road line; else the v3.1 ground test (stripe_geometry.decide_points) -> zebra /
          hatched; if that is ambiguous, the post-rule class votes decide (front cameras x3); no votes -> unchanged.
  pass C  every raster pixel of classes 1/2/3/6 is lifted to its world cell: zebra regions turn 2/3/6 into 3, and 1 into 3
          where the cell's area share is >= 0.5; hatched likewise into 6; line regions turn 3/6 into 2. Points and ranges
          are re-lifted exactly as the extractor did, after the CONTROL that the input re-lifts bit-identically.
Usage: sam3map_consensus.py <c8> <src npz dir> <dst npz dir>   (SAM3MAP_ROOT = the sequence root)
"""
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
for p_ in ("/home/nvidia/sam3paint", "/home/nvidia/sam3map", "/home/nvidia/sam3map/eval", str(HERE)):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
import numpy as np
import cv2
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
import sam3_paint as P
import camera_model as CM
import ground_surface as GS
import stripe_geometry as SG
import sam3map_render_v5 as R5

ROOT = Path(os.environ.get("SAM3MAP_ROOT", "/home/nvidia/sam3map/native7"))
RES = R5.RES
CAM_W = {"CAM_FW": 1.0, "CAM_F0": 1.0, "CAM_FT": 0.8, "CAM_CL": 0.6, "CAM_CR": 0.6, "CAM_RL": 0.5, "CAM_RR": 0.5, "CAM_RT": 0.6}
FRONT = ("CAM_FW", "CAM_F0", "CAM_FT")
AREA_SHARE, OBS_MIN, REGION_MIN_M2, PAINT_SHARE, STRONG_SHARE, PAINT_NEAR_M = 0.35, 0.3, 1.0, 0.5, 0.5, 15.0
DRIVE = (1, 2, 3, 4, 6)


def relift(cls_small, cam, sgrid, T):
    """The extractor's lifting of a 960x540 class raster (identical to sam3map_refine_v6.relift)."""
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


def lift_uv(C, sg, u, v):
    """Full-resolution pixel centres -> rig-frame ground xy on the smooth ground (the camera_model.lift iteration)."""
    d = C.rays_rig(u, v)
    good = np.isfinite(d).all(axis=1) & (d[:, 2] < -1e-3)
    d = np.where(good[:, None], d, np.array([0.0, 0.0, -1.0]))
    z = np.full(len(u), float(np.median(sg)))
    for _ in range(3):
        s = (z - C.t[2]) / d[:, 2]
        xy = C.t[:2] + d[:, :2] * s[:, None]
        z = GS.height(xy, sg)
    return xy, good & (s > 0) & (np.hypot(xy[:, 0] - C.t[0], xy[:, 1] - C.t[1]) <= R5.R_MAX)


def blobs_of(M, x0, y0):
    lab, nb = ndi.label(M, structure=np.ones((3, 3)))
    out = []
    for b, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        ii, jj = np.nonzero(lab[sl] == b)
        if len(ii) < 5:
            continue
        X = x0 + (ii + sl[0].start + 0.5) * RES; Y = y0 + (jj + sl[1].start + 0.5) * RES
        cov = np.cov(np.c_[X, Y].T) if len(ii) > 2 else np.eye(2) * 1e-4
        ev_, evec = np.linalg.eigh(cov)
        L = float(np.sqrt(12 * max(ev_[1], 0)) + RES); W = float(np.sqrt(12 * max(ev_[0], 0)) + RES)
        out.append({"n": len(ii), "area": len(ii) * RES ** 2, "L": L, "W": W, "theta": float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])) % 180),
                    "cx": float(X.mean()), "cy": float(Y.mean())})
    return out


def fold(a):
    return float(abs((a + 90.0) % 180.0 - 90.0))


def stripe_modes(M, Rd):
    """Edge-based stripe orientation of a 0.10 m paint raster (structure tensor), valid for stripes of ANY width (the
    1 m-cell point test of stripe_geometry rejects stripes wider than ~0.5 m: they are not elongated inside a 1 m cell).
    Returns (share per 15 deg bin of STRIPE directions, total edge weight)."""
    S = ndi.gaussian_filter(M.astype(np.float32), 1.0)
    gx = ndi.sobel(S, axis=0); gy = ndi.sobel(S, axis=1)
    J11 = ndi.gaussian_filter(gx * gx, 2.0); J22 = ndi.gaussian_filter(gy * gy, 2.0); J12 = ndi.gaussian_filter(gx * gy, 2.0)
    tr = J11 + J22
    coh = np.sqrt((J11 - J22) ** 2 + 4 * J12 ** 2) / np.maximum(tr, 1e-9)
    grad_dir = 0.5 * np.degrees(np.arctan2(2 * J12, J11 - J22))           # axis 0 = world x, axis 1 = world y
    stripe = (grad_dir + 90.0) % 180.0
    w = tr * coh * Rd
    keep = w > 0.05 * w.max() if w.max() > 0 else np.zeros_like(w, bool)
    hist = np.array([w[keep & (stripe >= b) & (stripe < b + 15)].sum() for b in range(0, 180, 15)])
    tot = float(hist.sum())
    return (hist / tot if tot > 0 else hist), tot


def geometry_verdict(M, Rd, axis, elong):
    """zebra / hatched / ambiguous from the edge-based modes, with the v3.1 thresholds: two strong modes >= 40 deg apart
    -> hatched (chevron, or stripes plus a border line); one dominant mode >= 60 deg to the region axis -> zebra;
    20-55 deg -> hatched; region elongation >= 1.5 for the axis tests."""
    share, tot = stripe_modes(M, Rd)
    if tot <= 0 or M.sum() < 30:
        return "too_few", {}
    b1 = int(np.argmax(share)); m1 = b1 * 15 + 7.5
    s2 = share.copy()
    for dd in (-2, -1, 0, 1, 2):
        s2[(b1 + dd) % 12] = 0
    b2 = int(np.argmax(s2)); m2 = b2 * 15 + 7.5
    delta1 = fold(m1 - axis); sep = fold(m1 - m2)
    info = {"mode1_deg": m1, "mode1_share": round(float(share[b1]), 2), "mode2_deg": m2, "mode2_share": round(float(s2[b2]), 2),
            "delta1_deg": round(delta1, 1), "mode_sep_deg": round(sep, 1)}
    if share[b1] >= 0.25 and s2[b2] >= 0.25 and sep >= 40:
        return "hatched", info
    if elong < 1.5 or share[b1] < 0.35:
        return "ambiguous", info
    if delta1 >= 60:
        return "zebra", info
    if 20 <= delta1 <= 55:
        return "hatched", info
    return "ambiguous", info


def decide_region(M, R, x0, y0, votes, path, tang):
    """One region's class (3 zebra, 6 hatched, 2 line, 0 unchanged) and the reasons."""
    info = {"area_m2": round(float(R.sum()) * RES ** 2, 1), **{k: round(v, 2) for k, v in votes.items()}}
    ri, rj = np.nonzero(R)
    rxy = np.c_[x0 + (ri + 0.5) * RES, y0 + (rj + 0.5) * RES]
    mi, mj = np.nonzero(M)
    mxy = np.c_[x0 + (mi + 0.5) * RES, y0 + (mj + 0.5) * RES]
    info["paint_m2"] = round(len(mxy) * RES ** 2, 2)
    c = rxy.mean(axis=0); k = int(np.argmin(np.hypot(path[:, 0] - c[0], path[:, 1] - c[1])))
    dpath = float(np.hypot(*(path[k] - c)))
    road = float(np.degrees(np.arctan2(tang[k, 1], tang[k, 0])) % 180) if dpath <= 15.0 else None
    cov = np.cov(rxy.T); ev_, evec = np.linalg.eigh(cov)
    axis = float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])) % 180); elong = float(np.sqrt(max(ev_[1], 1e-9) / max(ev_[0], 1e-9)))
    info.update({"axis_deg": round(axis, 1), "elong": round(elong, 2), "road_deg": None if road is None else round(road, 1), "d_path_m": round(dpath, 1)})
    bl = blobs_of(M, x0, y0)
    sig = [b for b in bl if b["area"] >= 0.05]
    info["blobs"] = len(sig)
    # (1) yield teeth: >= 4 short blobs in a row, no long stripes
    short = [b for b in sig if b["L"] < 1.0]
    longs = [b for b in sig if b["L"] >= 1.2]
    if len(short) >= 4 and sum(b["area"] for b in longs) <= 0.3 * sum(b["area"] for b in sig):
        cxy = np.array([[b["cx"], b["cy"]] for b in short])
        e2, _ = np.linalg.eigh(np.cov(cxy.T))
        if np.sqrt(max(e2[0], 0)) <= 0.4 and np.sqrt(12 * max(e2[1], 0)) >= 2.0:
            info["rule"] = "yield_teeth_row"; return 2, info
    # (2) line fragments: thin blobs along the region axis carry >= 80 % of the paint, region elongated
    if sig and elong >= 3.0:
        thin_along = [b for b in sig if b["W"] <= 0.35 and fold(b["theta"] - axis) <= 15.0]
        if sum(b["area"] for b in thin_along) >= 0.8 * sum(b["area"] for b in sig):
            info["rule"] = "line_fragments_along_axis"; return 2, info
    # (3) the ground test on the fused paint (edge-based stripe modes, v3.1 thresholds)
    verdict, gi = geometry_verdict(M, ndi.binary_dilation(R, iterations=5), axis, elong)
    info["geometry"] = verdict; info.update({f"g_{k}": v for k, v in gi.items()})
    if verdict == "zebra":
        info["rule"] = "ground_zebra"; return 3, info
    if verdict == "hatched":
        info["rule"] = "ground_hatched"; return 6, info
    # (4) road-direction test when the region is square-ish (the axis is undefined) but stripes are clear
    if road is not None and verdict == "ambiguous" and "mode1_deg" in gi:
        a1 = fold(gi["mode1_deg"] - road)
        info["road_delta_deg"] = round(a1, 1)
        if gi["mode1_share"] >= 0.4 and (a1 <= 20 or a1 >= 70) and len(longs) >= 3:
            info["rule"] = "road_parallel_stripes"; return 3, info
        if gi["mode1_share"] >= 0.4 and 30 <= a1 <= 60 and len(longs) >= 2:
            info["rule"] = "road_diagonal_stripes"; return 6, info
    # (5) class votes, front cameras x3
    s3 = 3 * votes["f3"] + (votes["v3"] - votes["f3"]); s6 = 3 * votes["f6"] + (votes["v6"] - votes["f6"])
    if s3 + s6 >= 0.5:
        info["rule"] = "votes_front_x3"; return (3 if s3 >= s6 else 6), info
    info["rule"] = "no_decision"; return 0, info


def main():
    c8, src, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    files = sorted(src.glob("[0-9][0-9][0-9].npz"))
    frames = [dict(np.load(f, allow_pickle=True)) for f in files]
    sd = ROOT / f"seq_{c8}"
    toks = [str(f["tok"]) for f in frames]
    cams = [c for c in R5.R4.VIEWS if f"cls_{c}" in frames[0]]
    Ts = [f["T_world_rig"] for f in frames]; path = np.array([T[:2, 3] for T in Ts])
    tang = np.zeros_like(path)
    for n in range(len(path)):
        a, b = max(0, n - 2), min(len(path) - 1, n + 2)
        d = path[b] - path[a]
        tang[n] = d / np.hypot(*d) if np.hypot(*d) > 0.3 else Ts[n][:2, 0]
    wx0, wy0 = path.min(axis=0) - (R5.R_MAX + 12); wx1, wy1 = path.max(axis=0) + (R5.R_MAX + 12)
    WW, WH = int(np.ceil((wx1 - wx0) / RES)), int(np.ceil((wy1 - wy0) / RES))
    F = {k: np.zeros((WW, WH), np.float32) for k in ("obs", "pnt", "obsn", "pntn", "v3", "v6", "v2", "vx", "vd", "f3", "f6")}
    ctx = []
    # ---- pass A
    for n, f in enumerate(frames):
        fd = sd / toks[n]
        c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
        lp = fd / "lidar.npy"
        grid, fb = P.ground_grid(np.load(lp).astype(np.float64)) if lp.exists() else (np.full(2500, np.nan), 0.0)
        sg = GS.smooth_grid(grid, fb)
        cm = {}
        for cam in cams:
            i = fr["cam_order"].index(cam)
            wh = tuple(int(x) for x in np.atleast_2d(c["image_wh"])[min(i, len(np.atleast_2d(c["image_wh"])) - 1)]) if "image_wh" in c.files else (1920, 1080)
            C = CM.Camera.from_calib(c, i, *wh); cm[cam] = C
            img = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"))
            code, ev, paint, _ = R5.layer_fields(C, sg, img, f[f"cls_{cam}"], f.get(f"ego_{cam}"), f.get(f"evid_{cam}"))
            fi, fj, li, lj, dist = R5.world_cells(Ts[n], C.t, wx0, wy0, WW, WH)
            cd = code[li, lj]; seen = cd != 255
            fi, fj, li, lj, dist, cd = fi[seen], fj[seen], li[seen], lj[seen], dist[seen], cd[seen]
            w = (CAM_W.get(cam, 0.6) / (1.0 + (dist / 8.0) ** 2)).astype(np.float32)
            e = ev[li, lj]
            F["obs"][fi, fj] += w
            F["pnt"][fi, fj] += w * paint[li, lj]
            nr = dist <= PAINT_NEAR_M                                        # paint evidence only from views where paint is resolved
            F["obsn"][fi[nr], fj[nr]] += w[nr]; F["pntn"][fi[nr], fj[nr]] += w[nr] * paint[li[nr], lj[nr]]
            F["v3"][fi, fj] += w * (cd == 3); F["v6"][fi, fj] += w * (cd == 6); F["v2"][fi, fj] += w * (cd == 2)
            F["vx"][fi, fj] += w * ((e & 2) > 0); F["vd"][fi, fj] += w * ((e & 1) > 0)
            if cam in FRONT:
                F["f3"][fi, fj] += w * (cd == 3); F["f6"][fi, fj] += w * (cd == 6)
        ctx.append((sg, cm))
        if n % 20 == 0:
            print(f"  pass A frame {n}: {time.time() - t0:.0f}s", flush=True)
    # ---- pass B
    obs = np.maximum(F["obs"], 1e-6)
    share = (F["v3"] + F["v6"] + F["vx"] + F["vd"]) / obs
    cand = ndi.binary_closing((share >= AREA_SHARE) & (F["obs"] >= OBS_MIN), structure=np.ones((5, 5)))
    lab, nreg = ndi.label(cand, structure=np.ones((3, 3)))
    pshare = np.where(F["obsn"] >= 0.1, F["pntn"] / np.maximum(F["obsn"], 1e-6), F["pnt"] / obs)
    dec = np.zeros(nreg + 1, np.uint8); regions = []
    for r, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        s0 = slice(max(0, sl[0].start - 10), min(WW, sl[0].stop + 10)); s1 = slice(max(0, sl[1].start - 10), min(WH, sl[1].stop + 10))
        R = lab[s0, s1] == r
        if R.sum() * RES ** 2 < REGION_MIN_M2:
            continue
        Rd = ndi.binary_dilation(R, iterations=5)
        M = (pshare[s0, s1] >= PAINT_SHARE) & (F["obs"][s0, s1] >= OBS_MIN) & Rd
        votes = {k: float(F[k][s0, s1][R].sum()) for k in ("v3", "v6", "v2", "vx", "vd", "f3", "f6", "obs")}
        x0, y0 = wx0 + s0.start * RES, wy0 + s1.start * RES
        k_, info = decide_region(M, R, x0, y0, votes, path, tang)
        dec[r] = k_
        info.update({"id": r, "class": int(k_), "cx": round(float(x0 + np.nonzero(R)[0].mean() * RES), 1), "cy": round(float(y0 + np.nonzero(R)[1].mean() * RES), 1)})
        regions.append((info, M, R, np.where(Rd, pshare[s0, s1], 0.0)))
    print(f"pass B: {len(regions)} regions -> " + json.dumps({str(k): int(sum(1 for i, *_ in regions if i['class'] == k)) for k in (0, 2, 3, 6)})
          + f"  {time.time() - t0:.0f}s", flush=True)
    # ---- pass C
    tot = {"control_identical": 0, "control_failed": 0, "to3": 0, "to6": 0, "to2": 0, "px_changed": 0}
    per_cam = {cam: 0 for cam in cams}
    for n, (f, fpath) in enumerate(zip(frames, files)):
        sg, cm = ctx[n]; T = Ts[n]
        d = dict(f)
        accp, accr = {k: [] for k in range(1, 8)}, {k: [] for k in range(1, 8)}
        newp, newr = {k: [] for k in range(1, 8)}, {k: [] for k in range(1, 8)}
        for cam in cams:
            p_, r_ = relift(f[f"cls_{cam}"], cm[cam], sg, T)
            for k in p_:
                accp[k].append(p_[k]); accr[k].append(r_[k])
        cat = lambda lst, shape, dt: np.concatenate(lst) if lst else np.zeros(shape, dt)
        if not all(np.array_equal(cat(accp[k], (0, 2), np.float32), f[f"pts_{k}"]) and np.array_equal(cat(accr[k], (0,), np.float16), f[f"rng_{k}"])
                   for k in range(1, 8) if f"pts_{k}" in f):
            tot["control_failed"] += 1; print(f"  CONTROL FAILED on {fpath.name}: copied unchanged", flush=True)
            np.savez_compressed(dst / fpath.name, **d); continue
        tot["control_identical"] += 1
        fr_ch = {}
        for cam in cams:
            cl = f[f"cls_{cam}"].copy()
            vv, uu = np.nonzero(np.isin(cl, (1, 2, 3, 6)))
            if len(vv):
                xy, ok = lift_uv(cm[cam], sg, uu * 2 + 1.0, vv * 2 + 1.0)
                w = xy @ T[:2, :2].T + T[:2, 3]
                ci = np.floor((w[:, 0] - wx0) / RES).astype(np.int64); cj = np.floor((w[:, 1] - wy0) / RES).astype(np.int64)
                ok &= (ci >= 0) & (ci < WW) & (cj >= 0) & (cj < WH)
                rid = np.zeros(len(vv), np.int64); rid[ok] = lab[ci[ok], cj[ok]]
                dc = dec[rid]; old = cl[vv, uu]; sh = np.zeros(len(vv), np.float32); sh[ok] = share[ci[ok], cj[ok]]
                new = old.copy()
                area_px = np.isin(old, (2, 3, 6)); strong1 = (old == 1) & (sh >= STRONG_SHARE)
                new[(dc == 3) & (area_px | strong1)] = 3
                new[(dc == 6) & (area_px | strong1)] = 6
                new[(dc == 2) & np.isin(old, (3, 6))] = 2
                ch = new != old
                if ch.any():
                    cl[vv[ch], uu[ch]] = new[ch]
                    tot["to3"] += int((new[ch] == 3).sum()); tot["to6"] += int((new[ch] == 6).sum()); tot["to2"] += int((new[ch] == 2).sum())
                    tot["px_changed"] += int(ch.sum()); per_cam[cam] += int(ch.sum()); fr_ch[cam] = int(ch.sum())
            d[f"cls_{cam}"] = cl
            p_, r_ = relift(cl, cm[cam], sg, T)
            for k in p_:
                newp[k].append(p_[k]); newr[k].append(r_[k])
        for k in range(1, 8):
            d[f"pts_{k}"] = cat(newp[k], (0, 2), np.float32); d[f"rng_{k}"] = cat(newr[k], (0,), np.float16)
        st = json.loads(str(d["stats"])); st["_consensus"] = {"px_changed": fr_ch}
        d["stats"] = json.dumps(st)
        np.savez_compressed(dst / fpath.name, **d)
        if n % 20 == 0:
            print(f"  pass C frame {n}: {time.time() - t0:.0f}s", flush=True)
    rep = {"clip_c8_digest_only": "see meta.json clip_sha12", "regions": [i for i, *_ in regions], "totals": tot, "px_changed_per_camera": per_cam,
           "params": {"AREA_SHARE": AREA_SHARE, "OBS_MIN": OBS_MIN, "REGION_MIN_M2": REGION_MIN_M2, "PAINT_SHARE": PAINT_SHARE, "STRONG_SHARE": STRONG_SHARE, "CAM_W": CAM_W},
           "seconds": round(time.time() - t0, 1)}
    (dst.parent / f"consensus_{dst.name}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    atlas(regions, dst.parent / f"consensus_atlas_{dst.name}.png")
    print(f"totals {tot} per camera {per_cam}  {time.time() - t0:.0f}s", flush=True)
    print("ZZCONSENSUS-DONEZZ")


def atlas(regions, out, tile=260, cols=6):
    """One tile per region (largest first): fused paint (white), region (grey), decision + rule."""
    regions = sorted(regions, key=lambda x: -x[0]["area_m2"])[:36]
    if not regions:
        return
    rows = (len(regions) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * (tile + 44)), (10, 12, 11)); dr = ImageDraw.Draw(sheet)
    colour = {0: (150, 150, 150), 2: R5.R4.COL[2], 3: R5.R4.COL[3], 6: R5.R4.COL[6]}
    for q, (info, M, R, PS) in enumerate(regions):
        g = (np.clip(PS, 0, 1) * 200).astype(np.uint8)
        im = np.zeros(M.shape + (3,), np.uint8); im[R] = (40, 60, 90); im[..., 0] = np.maximum(im[..., 0], g); im[..., 1] = np.maximum(im[..., 1], g); im[..., 2] = np.maximum(im[..., 2], g)
        im[M] = (255, 255, 255)
        pil = Image.fromarray(im.transpose(1, 0, 2)[::-1])          # world y up, x right
        s = min(tile / pil.width, tile / pil.height); pil = pil.resize((max(1, int(pil.width * s)), max(1, int(pil.height * s))), Image.NEAREST)
        x, y = (q % cols) * tile, (q // cols) * (tile + 44)
        sheet.paste(pil, (x + (tile - pil.width) // 2, y + (tile - pil.height) // 2))
        dr.rectangle([x, y, x + tile - 1, y + tile - 1], outline=colour[info["class"]], width=4)
        dr.text((x + 4, y + tile + 2), f"#{info['id']} {info['area_m2']} m2 -> {info['class']} {info['rule']}", fill=(230, 232, 231), font=R5.R4.font(12))
        dr.text((x + 4, y + tile + 18), f"g {info.get('geometry', '-')} d1 {info.get('g_delta1_deg', '-')} el {info['elong']} v3 {info['v3']} v6 {info['v6']} f3 {info['f3']} f6 {info['f6']}",
                fill=(170, 176, 173), font=R5.R4.font(11))
    sheet.save(out)


if __name__ == "__main__":
    main()
