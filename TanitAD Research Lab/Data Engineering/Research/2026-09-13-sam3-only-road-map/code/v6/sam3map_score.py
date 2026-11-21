"""SAM3-only map vs Qwen-Drive map on the SAME frames, with the LiDAR checks calibrated on Qwen's demo this morning.

SAM3 classes -> the map-metric raster convention (x -30..30, y -15..15 at 0.15 m, map classes 1 drivable, 2 line,
3 edge, 4 crosswalk): drivable->1, lane/road line->2, painted symbol->2, crosswalk->4, non-drivable edge->3.
Arms: SAM3 single frame, SAM3 fused +-1 s (the rendered map), Qwen v2 single frame, Qwen v2 fused +-2 s.
Metrics (mapq_core; true nuPlan map in brackets): MAP_A2 tall obstacles on drivable [0.03 %], MAP_B ego path on
road, MAP_C vehicles on road [100 %], MAP_E edges on a real curb/obstacle edge [74.7 %].

NEXT-LEVER ARM, pre-registered 2026-09-13 BEFORE its first run (the +-1 s map left 13 % of the car's own path off the
road on the first 15 frames): SAM3_clipvote = an offline label rule over the WHOLE clip. A cell is class k when, among
the frames that SAW it (inside one of the 7 views, within R_OBS = 15 m of the rig, where stride-2 lifting still samples
every 0.15 m cell), at least 30 % (drivable) / 20 % (thin classes) labelled it k; >= 2 observations; the render's
fractions and closings, NOT tuned. It WORKS if, on the full clip, MAP_B >= 0.95 AND MAP_A2 <= SAM3_fused_1s + 0.05,
with its mirrored control below it on MAP_B and MAP_E. Anything else is a FAIL, reported as such.
"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

MQ = Path(r"<scratchpad>/mapq")
sys.path.insert(0, str(MQ))
import mapq_core as mc  # noqa: E402
import mapq_ours as mo  # noqa: E402
sys.path.append(str(Path(__file__).resolve().parent)); sys.path.append(str(Path(__file__).resolve().parent.parent))   # camera_model beside / above
import camera_model as CM  # noqa: E402

TO_MAP = {1: 1, 2: 2, 4: 2, 3: 4, 5: 3, 6: 2}      # v3 hatched area counts as paint (the map metrics have no hatch class)
PAINT_WRITE_ORDER = (2, 4, 6, 3, 5)                  # later classes overwrite earlier ones in the metric raster


VIEWS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2"]      # replaced in main() by the views the npz carries
CALIB_ROOT = None                                                                  # SAM3MAP_ROOT (native sequences) or mo.V2


def cameras(fd):
    """The map's own views at this token, through camera_model (pinhole or f-theta), calib from CALIB_ROOT."""
    cd = (CALIB_ROOT or mo.V2) / fd.parent.name / fd.name
    c = np.load(cd / "calib.npz"); fr = json.loads((cd / "frame.json").read_text(encoding="utf-8"))
    out = []
    for cam in VIEWS:
        i = fr["cam_order"].index(cam)
        w, h = (int(x) for x in np.atleast_2d(c["image_wh"])[min(i, len(np.atleast_2d(c["image_wh"])) - 1)]) if "image_wh" in c.files else (1920, 1080)
        out.append(CM.Camera.from_calib(c, i, w, h))
    return out


def ground_grid(pts):
    p = pts[np.hypot(pts[:, 0], pts[:, 1]) < 45]
    key = (np.floor((p[:, 0] + 50) / 2).astype(int) * 50 + np.floor((p[:, 1] + 50) / 2).astype(int))
    grid = np.full(2500, np.nan)
    order = np.argsort(key, kind="stable"); k, z = key[order], p[order, 2]
    starts = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
    for a, b in zip(starts, np.r_[starts[1:], len(k)]):
        if b - a >= 5:
            grid[k[a]] = np.percentile(z[a:b], 10)
    return grid, (float(np.nanmedian(grid)) if np.isfinite(grid).any() else 0.0)


def camera_coverage(fd, grid, fb):
    """Map cells whose ground point projects inside at least one of the 7 SAM3 views (the same views the map used)."""
    ii, jj = np.meshgrid(np.arange(200), np.arange(400), indexing="ij")
    x = -30 + (jj.ravel() + 0.5) * 0.15; y = -15 + (ii.ravel() + 0.5) * 0.15
    ci = np.clip(np.floor((x + 50) / 2).astype(int), 0, 49); cj = np.clip(np.floor((y + 50) / 2).astype(int), 0, 49)
    z = np.where(np.isfinite(grid[ci * 50 + cj]), grid[ci * 50 + cj], fb)
    Pg = np.c_[x, y, z]
    return visible(fd, Pg).reshape(200, 400)


def worldmap_raster(wm, T):
    """The renderer-v5 whole-clip world map (the GROUND-TRUTH candidate, non-causal) cut into this frame's metric raster:
    1 drivable -> 1, lane line / arrow-text / hatched -> 2, non-drivable edge -> 3, crosswalk -> 4, sidewalk-verge -> 5 walkway."""
    ii, jj = np.meshgrid(np.arange(200), np.arange(400), indexing="ij")
    xy = np.c_[-30 + (jj.ravel() + 0.5) * 0.15, -15 + (ii.ravel() + 0.5) * 0.15]
    w = xy @ T[:2, :2].T + T[:2, 3]
    cls, (x0, y0), res = wm["cls"], wm["origin"], float(wm["res"])
    i = np.floor((w[:, 0] - x0) / res).astype(np.int64); j = np.floor((w[:, 1] - y0) / res).astype(np.int64)
    ok = (i >= 0) & (i < cls.shape[0]) & (j >= 0) & (j < cls.shape[1])
    code = np.full(len(xy), 255, np.uint8); code[ok] = cls[i[ok], j[ok]]
    out = np.zeros(len(xy), np.int64)
    for k, m in ((1, 1), (2, 2), (4, 2), (6, 2), (3, 4), (5, 3), (7, 5)):
        out[code == k] = m
    return out.reshape(200, 400)


def keep_observed(xy, obs):
    if xy is None or not len(xy):
        return xy
    i = np.floor((xy[:, 1] + 15) / 0.15).astype(int); j = np.floor((xy[:, 0] + 30) / 0.15).astype(int)
    ok = (i >= 0) & (i < 200) & (j >= 0) & (j < 400)
    keep = np.zeros(len(xy), bool); keep[ok] = obs[i[ok], j[ok]]
    return xy[keep]


R_OBS = 15.0      # fx 1545 from ~1.8 m: stride-2 samples land <= ~0.15 m apart along the ground out to ~15 m
VOTE_RES = 0.15   # the metric raster's own resolution, so nothing is resampled between the vote and the score


def visible(fd, Pg):
    seen = np.zeros(len(Pg), bool)
    for C in cameras(fd):
        ok = C.project_rig(Pg)[2]
        if C.ft is None:                                   # pinhole: the banked scorer's depth cut (0.5 m), so old arms reproduce
            ok &= ((Pg - C.t) @ C.R)[:, 2] > 0.5
        seen |= ok
    return seen


def clip_vote(frames, c8, toks):
    """Whole-clip visibility-normalised vote in the world frame (see the module docstring for the committed rule)."""
    P = np.array([f["T_world_rig"][:2, 3] for f in frames])
    x0, y0 = P.min(axis=0) - 50; x1, y1 = P.max(axis=0) + 50
    W, H = int(np.ceil((x1 - x0) / VOTE_RES)), int(np.ceil((y1 - y0) / VOTE_RES))

    def flat_cells(xy):
        i = np.floor((xy[:, 1] - y0) / VOTE_RES).astype(np.int64); j = np.floor((xy[:, 0] - x0) / VOTE_RES).astype(np.int64)
        ok = (i >= 0) & (i < H) & (j >= 0) & (j < W)
        return np.unique(i[ok] * W + j[ok])                      # distinct cells: one vote per frame

    obs = np.zeros(H * W, np.int16); hits = {k: np.zeros(H * W, np.int16) for k in range(1, 7)}
    g = np.arange(-R_OBS, R_OBS, VOTE_RES / 2) + VOTE_RES / 4    # half-cell lattice, so a rotated world grid has no holes
    gx, gy = np.meshgrid(g, g, indexing="ij"); gx, gy = gx.ravel(), gy.ravel()
    keep = np.hypot(gx, gy) <= R_OBS; gx, gy = gx[keep], gy[keep]
    ci = np.clip(np.floor((gx + 50) / 2).astype(int), 0, 49) * 50 + np.clip(np.floor((gy + 50) / 2).astype(int), 0, 49)
    for n, f in enumerate(frames):
        T = f["T_world_rig"]; fd = mo.V2 / f"seq_{c8}" / toks[n]
        lid = np.load(fd / "lidar.npy").astype(np.float64)     # RIG frame, the frame calib.npz is expressed in
        grid, fb = ground_grid(lid)
        z = np.where(np.isfinite(grid[ci]), grid[ci], fb)
        s = visible(fd, np.c_[gx, gy, z])
        w = np.c_[gx[s], gy[s], np.zeros(s.sum()), np.ones(s.sum())] @ T.T
        obs[flat_cells(w[:, :2])] += 1
        o = T[:2, 3]
        for k in range(1, 7):
            p = f.get(f"pts_{k}", np.zeros((0, 2), np.float32))
            if len(p):
                hits[k][flat_cells(p[np.hypot(p[:, 0] - o[0], p[:, 1] - o[1]) <= R_OBS])] += 1
    obs = obs.reshape(H, W); hits = {k: v.reshape(H, W) for k, v in hits.items()}
    seen2 = obs >= 2
    vote = np.zeros((H, W), np.uint8)
    vote[ndi.binary_closing(seen2 & (hits[1] >= np.ceil(0.3 * obs)), structure=np.ones((5, 5)))] = 1
    for k in PAINT_WRITE_ORDER:
        vote[ndi.binary_closing(seen2 & (hits[k] >= np.ceil(0.2 * obs)), structure=np.ones((3, 3)))] = TO_MAP[k]
    return vote, (x0, y0, W, H)


def vote_raster(vote, geo, T):
    x0, y0, W, H = geo
    ii, jj = np.meshgrid(np.arange(200), np.arange(400), indexing="ij")
    xy = np.c_[-30 + (jj.ravel() + 0.5) * 0.15, -15 + (ii.ravel() + 0.5) * 0.15]
    w = np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ T.T
    i = np.floor((w[:, 1] - y0) / VOTE_RES).astype(int); j = np.floor((w[:, 0] - x0) / VOTE_RES).astype(int)
    ok = (i >= 0) & (i < H) & (j >= 0) & (j < W)
    out = np.zeros(len(xy), np.int64); out[ok] = vote[i[ok], j[ok]]
    return out.reshape(200, 400)


def sam3_raster(frames, j, half, T):
    """Same fusion rule as the render: drivable in >= 30 % of the window's frames, thin classes in >= 20 %."""
    win = frames[max(0, j - half): j + half + 1]
    need_road, need_thin = max(1, int(np.ceil(0.3 * len(win)))), max(1, int(np.ceil(0.2 * len(win))))
    hits = {k: np.zeros((200, 400), np.int16) for k in range(1, 7)}
    Tinv = np.linalg.inv(T)
    for f in win:
        for k in range(1, 7):
            p = f.get(f"pts_{k}", np.zeros((0, 2), np.float32))
            if not len(p):
                continue
            q = (np.c_[p, np.zeros(len(p)), np.ones(len(p))] @ Tinv.T)[:, :2]
            i = np.floor((q[:, 1] + 15) / 0.15).astype(int); jj = np.floor((q[:, 0] + 30) / 0.15).astype(int)
            ok = (i >= 0) & (i < 200) & (jj >= 0) & (jj < 400)
            m = np.zeros((200, 400), bool); m[i[ok], jj[ok]] = True
            hits[k] += m
    out = np.zeros((200, 400), np.int64)
    out[ndi.binary_closing(hits[1] >= need_road, structure=np.ones((5, 5)))] = 1
    for k in PAINT_WRITE_ORDER:
        out[ndi.binary_closing(hits[k] >= need_thin, structure=np.ones((3, 3)))] = TO_MAP[k]
    return out


def main():
    global VIEWS, CALIB_ROOT
    c8 = sys.argv[1]; npz_dir = Path(sys.argv[2])
    files = sorted(npz_dir.glob("[0-9][0-9][0-9].npz"))
    frames = [dict(np.load(f, allow_pickle=True)) for f in files]
    VIEWS = sorted(k[4:] for k in frames[0] if k.startswith("cls_"))
    CALIB_ROOT = Path(os.environ["SAM3MAP_ROOT"]) if os.environ.get("SAM3MAP_ROOT") else None
    print(f"views {VIEWS}; calibration from {CALIB_ROOT or mo.V2}", flush=True)
    toks = [str(f["tok"]) for f in frames]
    clip = mo.Clip(c8)
    all_toks = sorted(p.name for p in (mo.V2 / f"seq_{c8}").iterdir() if p.is_dir())
    metas = [json.loads((mo.V2 / f"seq_{c8}" / t / "meta.json").read_text(encoding="utf-8")) for t in all_toks]
    poses = [clip.pose(m["t_ref_us"]) for m in metas]
    qmaps = [np.load(mo.PRED / "v2" / f"out_seq_{c8}" / f"{t}.npz")["map"].astype(int) for t in all_toks]
    rows = {k: [] for k in ("SAM3_single", "SAM3_fused_1s", "SAM3_clipvote", "QWEN_single", "QWEN_fused_2s", "SAM3_MIRRORED",
                            "SAM3_clipvote_MIRRORED")}
    wm = dict(np.load(os.environ["SAM3MAP_WORLDMAP"], allow_pickle=True)) if os.environ.get("SAM3MAP_WORLDMAP") else None
    if wm is not None:
        rows.update({"SAM3_GT_worldmap": [], "SAM3_GT_worldmap_MIRRORED": []})
    veh_on_walkway = [0, 0]
    rows_obs, cov_share = {}, []
    vote, geo = clip_vote(frames, c8, toks)
    for n, f in enumerate(frames):
        j = all_toks.index(toks[n]); meta = metas[j]
        pts, _ = clip.sweep_ego(meta["t_ref_us"])
        g = np.load(mo.V2 / f"seq_{c8}" / toks[n] / "gt.npz")
        split = mc.split_points(pts, g["boxes"]); edge = mc.edge_evidence(split[1], split[0])
        kw = dict(ego_path_xy=clip.path_xy(meta["t_ref_us"]), split=split, edge=edge)
        T = f["T_world_rig"]
        s1 = sam3_raster(frames, n, 0, T); s5 = sam3_raster(frames, n, 5, T); sv = vote_raster(vote, geo, T)
        arms = {"SAM3_single": s1, "SAM3_fused_1s": s5, "SAM3_clipvote": sv, "QWEN_single": qmaps[j],
                "QWEN_fused_2s": mo.fuse(qmaps, poses, j, 10), "SAM3_MIRRORED": s5[::-1, :], "SAM3_clipvote_MIRRORED": sv[::-1, :]}
        if wm is not None:
            gtm = worldmap_raster(wm, T)
            arms["SAM3_GT_worldmap"] = gtm; arms["SAM3_GT_worldmap_MIRRORED"] = gtm[::-1, :]
            vb = g["boxes"][g["labels"] == 0]
            if len(vb):
                c_, _ = mc.map_cls(gtm, vb[:, :2]); veh_on_walkway[0] += int((c_ == 5).sum()); veh_on_walkway[1] += len(c_)
        for a, m in arms.items():
            rows[a].append(mc.frame_metrics(m, None, pts, g["boxes"], g["boxes"][g["labels"] == 0], **kw))
        # --- the same comparison restricted to ground SAM3's cameras actually see (both arms masked identically)
        fd = mo.V2 / f"seq_{c8}" / toks[n]
        lid = np.load(fd / "lidar.npy").astype(np.float64)     # RIG frame, the frame calib.npz is expressed in
        grid, fb = ground_grid(lid)
        obs = camera_coverage(fd, grid, fb)
        cov_share.append(float(obs.mean()))
        split_o = tuple(keep_observed(a, obs) for a in split)
        veh_o = g["boxes"][g["labels"] == 0]
        if len(veh_o):
            veh_o = veh_o[np.isin(np.arange(len(veh_o)), np.flatnonzero(keep_observed(veh_o[:, :2], obs)[:, 0:1].shape[0] * [True]))] if False else veh_o[
                np.array([bool(len(keep_observed(v[None, :2], obs))) for v in veh_o])]
        kw_o = dict(ego_path_xy=keep_observed(kw["ego_path_xy"], obs), split=split_o, edge=(edge[0], edge[1]))
        for a, m in arms.items():
            mo_ = np.where(obs, m, 0)
            rows_obs.setdefault(a, []).append(mc.frame_metrics(mo_, None, pts, g["boxes"], veh_o, **kw_o))
    res = {a: mc.aggregate(r) for a, r in rows.items()}
    res["observed_only"] = {a: mc.aggregate(r) for a, r in rows_obs.items()}
    res["camera_coverage_share_of_window_mean"] = round(float(np.mean(cov_share)), 4)
    if wm is not None:
        res["GT_worldmap_vehicle_centres_on_sidewalk_verge"] = {"value": round(veh_on_walkway[0] / max(veh_on_walkway[1], 1), 4), "n": veh_on_walkway[1]}
        res["GT_worldmap_source"] = os.environ["SAM3MAP_WORLDMAP"]
    out = npz_dir.parent / (f"sam3map_score_{npz_dir.name}" + ("_" + os.environ.get("SAM3MAP_SCORE_TAG", "gt") if wm is not None else "") + ".json")   # v1 and v2 directories score separately
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    keys = ("MAP_A2", "MAP_B", "MAP_C", "MAP_E")
    print(f"{'arm':15s} " + " ".join(f"{k:>9s}" for k in keys) + "   (n frames %d)" % len(frames))
    for a in rows:
        print(f"{a:15s} " + " ".join(f"{res[a][k]['value']:>9}" if k in res[a] else f"{'-':>9s}" for k in keys))
    print(f"--- restricted to SAM3 camera coverage ({100 * res['camera_coverage_share_of_window_mean']:.1f} % of the 60 x 30 m window on average)")
    for a in rows_obs:
        print(f"{a:15s} " + " ".join(f"{res['observed_only'][a][k]['value']:>9}" if k in res["observed_only"][a] else f"{'-':>9s}" for k in keys))
    print("ZZSAM3MAP-SCORE-DONEZZ")


if __name__ == "__main__":
    main()
