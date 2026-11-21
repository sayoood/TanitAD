"""One frame, four panels in the SAME ego-frame raster (forward up, left left):
front camera | LiDAR evidence (ground / static obstacles / curb steps / paint) | Qwen single-frame map | Qwen map fused +-2 s.
Clip ids are written as sha12 only."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mapq_core as mc  # noqa: E402
import mapq_ours as mo  # noqa: E402

MAP_PAL = np.array([(250, 250, 250), (196, 205, 214), (255, 193, 7), (225, 95, 65), (75, 180, 170), (139, 195, 74)], np.uint8)


def to_img(m):                           # (Y, X) raster -> forward up, left left (upstream `_map_rgb` convention)
    return MAP_PAL[np.clip(m, 0, 5)].transpose(1, 0, 2)[::-1, ::-1]


def evidence_img(split, paint):
    static, ground, agent, above, tall = split
    img = np.full((200, 400, 3), 255, np.uint8)
    def put(xy, col, cell=0.15):
        i = np.floor((xy[:, 1] + 15) / cell).astype(int); j = np.floor((xy[:, 0] + 30) / cell).astype(int)
        k = (i >= 0) & (i < 200) & (j >= 0) & (j < 400); img[i[k], j[k]] = col
    put(ground[:, :2], (196, 205, 214))
    ev, cov = mc.edge_evidence(ground, static)
    ii, jj = np.where(mc.up03(ev)); img[ii, jj] = np.maximum(img[ii, jj].astype(int) - 60, 0).astype(np.uint8)
    put(static[:, :2], (60, 60, 60))
    put(agent[:, :2], (255, 138, 0))
    put(paint, (230, 30, 30))
    return img.transpose(1, 0, 2)[::-1, ::-1]


def main(c8, j):
    V2 = mo.V2
    toks = sorted(p.name for p in (V2 / f"seq_{c8}").iterdir() if p.is_dir())
    clip = mo.Clip(c8)
    metas = [json.loads((V2 / f"seq_{c8}" / t / "meta.json").read_text(encoding="utf-8")) for t in toks]
    poses = [clip.pose(m["t_ref_us"]) for m in metas]
    maps = [np.load(mo.PRED / "v2" / f"out_seq_{c8}" / f"{t}.npz")["map"].astype(int) for t in toks]
    pts, _ = clip.sweep_ego(metas[j]["t_ref_us"])
    g = np.load(V2 / f"seq_{c8}" / toks[j] / "gt.npz")
    split = mc.split_points(pts, g["boxes"]); paint = mo.paint_points(split[1])
    fused = mo.fuse(maps, poses, j, 10)
    cam = Image.open(V2 / f"seq_{c8}" / toks[j] / "images" / "CAM_F0.jpg").convert("RGB").resize((640, 360))
    fig, ax = plt.subplots(1, 4, figsize=(17, 6.4), gridspec_kw={"width_ratios": [2.2, 1, 1, 1]})
    ax[0].imshow(cam); ax[0].set_title("front camera (virtual CAM_F0)", fontsize=10)
    ax[1].imshow(evidence_img(split, paint)); ax[1].set_title("LiDAR evidence (independent of Qwen)", fontsize=10)
    ax[2].imshow(to_img(maps[j])); ax[2].set_title("Qwen map, single frame", fontsize=10)
    ax[3].imshow(to_img(fused)); ax[3].set_title("Qwen map, fused over ±2 s", fontsize=10)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    for a in ax[1:]:
        a.plot([100], [200], marker="^", color="k", markersize=9)      # ego at raster centre, forward up
    leg = [Patch(color=np.array(c) / 255, label=l) for c, l in (((196, 205, 214), "LiDAR ground"), ((60, 60, 60), "static obstacle"),
                                                                ((255, 138, 0), "agent (GT box)"), ((230, 30, 30), "bright ground = paint"),
                                                                ((136, 145, 154), "height step / obstacle edge (darkened)"))]
    leg += [Patch(color=MAP_PAL[k] / 255, label=n) for k, n in ((1, "map: driveable"), (2, "map: road line"), (3, "map: road edge"), (4, "map: crosswalk"), (5, "map: walkway"))]
    fig.legend(handles=leg, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle(f"clip {metas[j]['clip_sha12']}  t = {(metas[j]['t_ref_us'] - metas[0]['t_ref_us']) / 1e6:.1f} s   —   60 m x 30 m map window, forward up",
                 fontsize=11)
    out = V2 / f"mapq_{metas[j]['clip_sha12']}_{j:03d}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
