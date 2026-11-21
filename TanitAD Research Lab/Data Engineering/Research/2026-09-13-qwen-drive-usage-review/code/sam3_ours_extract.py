"""Thor half of the our-data SAM3 paint test: SAM3 masks on all 8 v2 views -> lifted onto the LiDAR ground ->
WORLD coordinates (clip-local egomotion frame), saved per frame. Fusion and scoring run on the dev box,
where the LiDAR intensity lives. Also re-scores the demo with the UNION and AGREE fusion rules."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3vendor"); sys.path.insert(0, "/home/nvidia/sam3paint")
import numpy as np
from PIL import Image
from scipy import ndimage
import sam3_smoke as S
import sam3_paint as P

V2 = Path("/home/nvidia/qwendrive/v2")
WINDOWS = {"4fbd97b6a4b7": range(12, 40), "73495082f98b": range(34, 62)}      # 28 frames = 5.6 s each, crosswalks in view
OUT = Path("/home/nvidia/sam3paint/ours"); OUT.mkdir(exist_ok=True)


def masks_both(proc, img):
    state = proc.set_image(img)
    res = {}
    for name, prompts in (("paint", P.PAINT_PROMPTS), ("zebra", P.ZEBRA_PROMPTS)):
        u = np.zeros((img.height, img.width), bool)
        for pr in prompts:
            out = proc.set_text_prompt(state=state, prompt=pr)
            if out.get("scores") is None:
                continue
            sc = out["scores"].float().cpu().numpy().reshape(-1)
            if len(sc):
                ms = out["masks"].cpu().numpy().reshape(len(sc), img.height, img.width)
                for k in np.flatnonzero(sc >= P.SCORE_MIN):
                    u |= ms[k]
        res[name] = u
    return res


def demo_variants(frames):
    arms = {k: [0, 0, 0, 0] for k in ("QWEN", "SAM3", "OVERRIDE", "UNION", "AGREE")}
    for tok, gm, q, paint, zebra, observed in frames:
        true_paint = np.isin(gm, (2, 4)); qp = np.isin(q, (2, 4)); sp = paint | zebra
        roadish = ndimage.binary_dilation(np.isin(q, (1, 2, 4)), iterations=4)
        near_s = ndimage.binary_dilation(sp, iterations=P.TOL); near_q = ndimage.binary_dilation(qp, iterations=P.TOL)
        v = {"QWEN": qp, "SAM3": sp, "OVERRIDE": np.isin(P.fuse(q, paint, zebra, observed), (2, 4)),
             "UNION": qp | (sp & roadish), "AGREE": (qp & near_s) | (sp & roadish & near_q)}
        for arm, pm in v.items():
            a = P.pr(pm, true_paint, observed)
            for k in range(4):
                arms[arm][k] += a[k]
    out = {}
    for arm, a in arms.items():
        p, r = a[0] / max(a[1], 1), a[2] / max(a[3], 1)
        out[arm] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4)}
        print(f"  demo {arm:9s} P {p:.3f}  R {r:.3f}  F1 {out[arm]['f1']:.3f}")
    Path("sam3_demo_variants.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    t0 = time.time()
    proc, _ = S.build(conf=0.25)
    demo_variants(P.run_demo(proc))
    for c8, win in WINDOWS.items():
        sd = V2 / f"seq_{c8}"
        toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
        poses = json.loads((sd / "poses.json").read_text())
        for j in win:
            tok = toks[j]; fd = sd / tok
            c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
            pts = np.load(fd / "lidar.npy").astype(np.float64)                    # rig frame
            grid, fb = P.ground_grid(pts)
            T = np.array(poses[tok]["T_world_rig"])
            acc = {"paint": [], "zebra": []}
            Ks, Rs, ts, sizes = [], [], [], []
            for i, cam in enumerate(fr["cam_order"]):
                img = Image.open(fd / "images" / f"{cam}.jpg").convert("RGB")
                K, R, t = c["cam_intrinsic"][i], c["sensor2lidar_rotation"][i], c["sensor2lidar_translation"][i]
                Ks.append(K); Rs.append(R); ts.append(t); sizes.append(img.size)
                ms = masks_both(proc, img)
                for name in acc:
                    xy = P.lift(ms[name], K, R, t, grid, fb)
                    if len(xy):
                        w = np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ T.T
                        acc[name].append(w[:, :2])
            observed = P.observed_ground(Ks, Rs, ts, sizes, grid, fb)
            np.savez_compressed(OUT / f"{c8}_{j:03d}.npz", tok=tok,
                                paint_world=np.concatenate(acc["paint"]) if acc["paint"] else np.zeros((0, 2)),
                                zebra_world=np.concatenate(acc["zebra"]) if acc["zebra"] else np.zeros((0, 2)),
                                observed=observed, T_world_rig=T)
            print(f"  {c8} j={j} paint_pts={sum(len(a) for a in acc['paint'])} zebra_pts={sum(len(a) for a in acc['zebra'])}  {time.time() - t0:.0f}s", flush=True)
    print("ZZSAM3-OURS-EXTRACT-DONEZZ %.0fs" % (time.time() - t0))
