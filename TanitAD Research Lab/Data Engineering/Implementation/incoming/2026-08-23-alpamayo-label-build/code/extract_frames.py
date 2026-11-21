"""Extract camera frames + the ego path for visual validation of Steer clips."""
import io, json, sys
from pathlib import Path
import torch
from PIL import Image

ROOT = Path("/home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl")
OUT = Path("/home/nvidia/gtac/frames"); OUT.mkdir(parents=True, exist_ok=True)
ids = [l.strip() for l in open("/home/nvidia/gtac/steer_ids.txt") if l.strip()]

meta = {}
for cid in ids:
    f = ROOT / f"{cid}.v2ep.pt"
    if not f.exists():
        print("MISSING", cid); continue
    d = torch.load(f, map_location="cpu", weights_only=False)
    buf, lens = d["jpeg_buf"].numpy().tobytes(), d["jpeg_len"].tolist()
    poses = d["poses"].double()
    T = len(lens)
    # frames at t=0, mid, and mid+6s (the tactical band end)
    picks = [0, T // 2, min(T // 2 + 60, T - 1)]
    off = [0]
    for L in lens: off.append(off[-1] + int(L))
    tiles = []
    for k in picks:
        img = Image.open(io.BytesIO(buf[off[k]:off[k + 1]])).convert("RGB")
        img = img.resize((480, 192))
        tiles.append(img)
    sheet = Image.new("RGB", (480, 192 * 3))
    for i, t in enumerate(tiles):
        sheet.paste(t, (0, i * 192))
    sheet.save(OUT / f"{cid[:8]}.jpg", quality=82)
    # ego path over the tactical band, in ego frame of mid
    t0 = T // 2; t1 = min(t0 + 60, T - 1)
    yaw0 = float(poses[t0, 2])
    import math
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    path = []
    for k in range(t0, t1 + 1, 5):
        dx = float(poses[k, 0] - poses[t0, 0]); dy = float(poses[k, 1] - poses[t0, 1])
        path.append([round(dx * c - dy * s, 2), round(dx * s + dy * c, 2)])
    dyaw = math.atan2(math.sin(float(poses[t1, 2]) - yaw0),
                      math.cos(float(poses[t1, 2]) - yaw0))
    arc = float((poses[t0 + 1:t1 + 1, :2] - poses[t0:t1, :2]).norm(dim=-1).sum())
    meta[cid] = {"picks": picks, "T": T, "ego_path_2_6s": path,
                 "net_dyaw_deg": round(math.degrees(dyaw), 1),
                 "arc_m": round(arc, 1),
                 "lateral_end_m": path[-1][1] if path else None,
                 "v0": round(float(poses[t0, 3]), 2)}
    print(f"{cid[:8]}  dyaw={math.degrees(dyaw):+6.1f}deg  arc={arc:6.1f}m  "
          f"lat_end={path[-1][1]:+6.2f}m  v0={float(poses[t0,3]):.1f}")
json.dump(meta, open("/home/nvidia/gtac/steer_geom.json", "w"), indent=1)
print("wrote", OUT)
