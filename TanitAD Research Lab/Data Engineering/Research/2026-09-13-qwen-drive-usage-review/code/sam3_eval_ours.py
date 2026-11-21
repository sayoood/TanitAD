"""Dev-box half: which road-paint source is right on OUR data -- Qwen (single / fused), SAM3 (accumulated), or a fusion?

REFERENCE (independent of both camera models): LiDAR ground returns with high intensity (retro-reflective paint;
works at night), accumulated over +-1 s through ego poses, restricted to a CORRIDOR within 7 m of the ego's own
driven path -- which keeps the verges (bright gravel/grass, measured 2026-09-13) out of the reference without
using either vision source to define the road.
  precision = predicted paint cells (corridor, LiDAR-covered) within 0.45 m of reference paint
  recall    = reference paint cells (corridor, camera-observed) within 0.45 m of predicted paint
Controls that must collapse: each source MIRRORED, SAM3 from the OTHER clip (SHUFFLED), and the REFERENCE mirrored
(proves the reference is spatially meaningful, not a density artefact).
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import ndimage

MQ = Path(r"<scratchpad>/mapq")
sys.path.insert(0, str(MQ))
import mapq_core as mc  # noqa: E402
import mapq_ours as mo  # noqa: E402

S3 = Path(r"C:/Users/Admin/qwenvis/v2/sam3_ours")
HALF = 5          # +-5 frames = +-1 s at 5 Hz
TOL = 3           # 0.45 m
CORRIDOR_M = 7.0


def to_raster(xy_world, T_world_rig):
    q = np.c_[xy_world, np.zeros(len(xy_world)), np.ones(len(xy_world))] @ np.linalg.inv(T_world_rig).T
    m = np.zeros((200, 400), np.int32)
    i = np.floor((q[:, 1] + 15) / 0.15).astype(int); j = np.floor((q[:, 0] + 30) / 0.15).astype(int)
    k = (i >= 0) & (i < 200) & (j >= 0) & (j < 400)
    np.add.at(m, (i[k], j[k]), 1)
    return m


def fuse_classaware(maps, poses, j, half):
    """Area classes by majority; thin classes (road_line 2, crosswalk 4) kept where >= 30 % of covering frames saw them."""
    xy, ii, jj = mo.grid_xy()
    H = np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))]
    votes = np.zeros((len(xy), 6), np.int32); cover = np.zeros(len(xy), np.int32)
    for k in range(max(0, j - half), min(len(maps), j + half + 1)):
        q = H @ (np.linalg.inv(poses[k]) @ poses[j]).T
        ci = np.floor((q[:, 1] + 15) / 0.15).astype(int); cj = np.floor((q[:, 0] + 30) / 0.15).astype(int)
        ok = (ci >= 0) & (ci < 200) & (cj >= 0) & (cj < 400)
        votes[np.flatnonzero(ok), maps[k][ci[ok], cj[ok]]] += 1; cover[ok] += 1
    area = votes[:, [0, 1, 3, 5]].argmax(axis=1); area = np.array([0, 1, 3, 5])[area]
    out = area.copy()
    frac = votes / np.maximum(cover, 1)[:, None]
    out[frac[:, 4] >= 0.3] = 4
    out[frac[:, 2] >= 0.3] = 2
    none = cover == 0
    out[none] = maps[j].ravel()[none]
    return out.reshape(200, 400)


def pr(pred, ref, covered, observed):
    near_ref = ndimage.binary_dilation(ref, iterations=TOL); near_pred = ndimage.binary_dilation(pred, iterations=TOL)
    pp = pred & covered; rr = ref & observed
    return [int((pp & near_ref).sum()), int(pp.sum()), int((rr & near_pred).sum()), int(rr.sum())]


def main():
    arms = {}
    for c8 in ("4fbd97b6a4b7", "73495082f98b"):
        files = sorted(S3.glob(f"{c8}_*.npz"))
        idx = [int(f.stem.split("_")[1]) for f in files]
        clip = mo.Clip(c8)
        toks = sorted(p.name for p in (mo.V2 / f"seq_{c8}").iterdir() if p.is_dir())
        metas = [json.loads((mo.V2 / f"seq_{c8}" / t / "meta.json").read_text(encoding="utf-8")) for t in toks]
        poses = [clip.pose(m["t_ref_us"]) for m in metas]
        maps = [np.load(mo.PRED / "v2" / f"out_seq_{c8}" / f"{t}.npz")["map"].astype(int) for t in toks]
        S = {j: np.load(f, allow_pickle=True) for j, f in zip(idx, files)}
        other = "73495082f98b" if c8 == "4fbd97b6a4b7" else "4fbd97b6a4b7"
        S_other = {int(f.stem.split("_")[1]): np.load(f, allow_pickle=True) for f in sorted(S3.glob(f"{other}_*.npz"))}
        sweep_cache = {}

        def bright_world(si):
            if si not in sweep_cache:
                t = int(clip.mid[si])
                pts, _ = clip.sweep_ego(t)
                g = np.load(mo.V2 / f"seq_{c8}" / toks[int(np.argmin([abs(m['t_ref_us'] - t) for m in metas]))] / "gt.npz")
                st, gr, ag, ab, tall = mc.split_points(pts, g["boxes"])
                bp = mo.paint_points(gr)
                T = clip.pose(t)
                w = lambda xy: (np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ T.T)[:, :2]
                sweep_cache[si] = (w(bp), w(gr[:, :2]))
            return sweep_cache[si]

        for j in idx:
            if j - HALF not in S or j + HALF not in S:
                continue
            T = poses[j]
            # --- SAM3 accumulated over +-1 s, a cell needs hits from >= 2 frames
            hits_p = sum((to_raster(S[k]["paint_world"], T) > 0).astype(int) for k in range(j - HALF, j + HALF + 1))
            hits_z = sum((to_raster(S[k]["zebra_world"], T) > 0).astype(int) for k in range(j - HALF, j + HALF + 1))
            sam3 = (hits_p >= 2) | (hits_z >= 2)
            jo = min(max(S_other), max(min(S_other), j))
            sam3_sh = sum((to_raster(S_other[k]["paint_world"], T) > 0).astype(int) for k in [jo]) > 0
            observed = S[j]["observed"].astype(bool)
            # --- reference: LiDAR bright ground within +-1 s, corridor around the ego path
            t = metas[j]["t_ref_us"]
            sis = np.flatnonzero(np.abs(clip.mid - t) <= 1_000_000)
            bw = [bright_world(si) for si in sis]
            ref = to_raster(np.concatenate([b for b, _ in bw]), T) > 0
            covered = ndimage.binary_dilation(to_raster(np.concatenate([g for _, g in bw]), T) > 0, iterations=1)
            path = clip.path_xy(t)
            corridor = np.zeros((200, 400), bool)
            pi = np.floor((path[:, 1] + 15) / 0.15).astype(int); pj = np.floor((path[:, 0] + 30) / 0.15).astype(int)
            kk = (pi >= 0) & (pi < 200) & (pj >= 0) & (pj < 400); corridor[pi[kk], pj[kk]] = True
            corridor = ndimage.distance_transform_edt(~corridor) * 0.15 <= CORRIDOR_M
            ref &= corridor; covered &= corridor; obs_c = observed & corridor
            # --- sources and fusions
            q1 = np.isin(maps[j], (2, 4)); qf_map = fuse_classaware(maps, poses, j, HALF); qf = np.isin(qf_map, (2, 4))
            roadish = ndimage.binary_dilation(np.isin(qf_map, (1, 2, 4)), iterations=4)
            near_s = ndimage.binary_dilation(sam3, iterations=TOL); near_q = ndimage.binary_dilation(qf, iterations=TOL)
            override = (qf & ~observed) | (sam3 & roadish)
            lines_only = (hits_p >= 2) & roadish; zebra_only = (hits_z >= 2) & roadish
            zebra_area = int(zebra_only.sum()); road_area = int(roadish.sum())
            arms.setdefault("_zebra_share_of_road", [0, 0, 0, 0]); arms["_zebra_share_of_road"][0] += zebra_area; arms["_zebra_share_of_road"][1] += max(road_area, 1)
            srcs = {"QWEN_single": q1, "QWEN_fused": qf, "SAM3_acc": sam3 & roadish, "SAM3_lines": lines_only, "SAM3_zebra": zebra_only,
                    "QWEN_lines": np.isin(qf_map, (2,)), "QWEN_zebra": np.isin(qf_map, (4,)), "OVERRIDE": override,
                    "UNION": qf | (sam3 & roadish), "AGREE": (qf & near_s) | (sam3 & roadish & near_q),
                    "SAM3_MIRRORED": (sam3 & roadish)[::-1, :], "QWEN_MIRRORED": qf[::-1, :], "SAM3_SHUFFLED": sam3_sh & roadish}
            for name, pm in srcs.items():
                a = pr(pm, ref, covered, obs_c)
                acc = arms.setdefault(name, [0, 0, 0, 0]); acc[:] = [x + y for x, y in zip(acc, a)]
                acc2 = arms.setdefault(f"{c8}:{name}", [0, 0, 0, 0]); acc2[:] = [x + y for x, y in zip(acc2, a)]
            a = pr(srcs["SAM3_acc"], ref[::-1, :], covered, obs_c)
            acc = arms.setdefault("SAM3_vs_REF_MIRRORED", [0, 0, 0, 0]); acc[:] = [x + y for x, y in zip(acc, a)]
            if j == idx[len(idx) // 2]:
                np.savez_compressed(S3.parent / f"sam3_fig_{c8}_{j:03d}.npz", ref=ref, corridor=corridor, qf=qf_map, sam3=sam3, q1=maps[j],
                                    union=np.where(sam3 & roadish, np.where(hits_z >= 2, 4, 2), qf_map))
        print(f"  {c8} scored", flush=True)
    res = {}
    for name, (a, n, b, m) in arms.items():
        if name.startswith("_"):
            continue
        p, r = a / max(n, 1), b / max(m, 1)
        res[name] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4), "n_pred": n, "n_ref": m}
    (S3.parent / "sam3_paint_ours.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    zs = arms["_zebra_share_of_road"]; print("SAM3 crosswalk area as share of road-ish area:", round(zs[0] / zs[1], 4))
    for name in ("QWEN_single", "QWEN_fused", "QWEN_lines", "QWEN_zebra", "SAM3_acc", "SAM3_lines", "SAM3_zebra", "OVERRIDE", "UNION", "AGREE",
                 "SAM3_MIRRORED", "QWEN_MIRRORED", "SAM3_SHUFFLED", "SAM3_vs_REF_MIRRORED"):
        d = res[name]; print(f"{name:22s} P {d['precision']:.3f}  R {d['recall']:.3f}  F1 {d['f1']:.3f}  pred {d['n_pred']}  ref {d['n_ref']}")
    for c8 in ("4fbd97b6a4b7", "73495082f98b"):
        print(c8, {k.split(":")[1]: (v["precision"], v["recall"]) for k, v in res.items() if k.startswith(c8 + ":")})
    print("ZZSAM3-EVAL-DONEZZ")


if __name__ == "__main__":
    main()
