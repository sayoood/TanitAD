"""SAM3-ONLY road map, v3: instance-level paint classification (crosswalk vs hatched area vs line vs arrow/text).

Same geometry and output format as v1 (sam3map_extract.py), plus class 6 HATCHED AREA and `ver = "v3"`.
Classes: 1 drivable, 2 lane/road line, 3 crosswalk, 4 arrow/text, 5 non-drivable edge, 6 hatched area.

Rules (each earned by a measured failure, sam3_v3_marks.py + promptbank replays, 2026-09-13):
  road       road 0.50, asphalt road surface 0.50 (v1)
  on-road    a paint instance needs >= 60 % within 15 px of the road mask and >= 35 % road in its 13 px ring (v1)
  line       lane marking 0.40, road marking 0.40, stop line 0.45 (v1; the demo sweep's winners did not hold out)
  crosswalk  crosswalk / zebra crossing / pedestrian crossing 0.50; accepted if <= 15 % of the view's road pixels OR
             it holds >= 3 line stripes (the v1 area cap rejected a zebra right in front of the car); its stripes
             become crosswalk
  hatched    a crosswalk region whose stripes are >= 30 % covered by "diagonal stripes on road" (>= 0.40) is a
             hatched area (SAM3 scores "crosswalk" 0.79-0.89 on yellow hatched zones, as high as on a real zebra;
             "diagonal stripes on road" separates them: 0.46-0.53 vs 0.22); a standalone diagonal-stripes instance
             >= 0.45 on the road is hatched too
  arrow/text arrow painted on road / text painted on road 0.45; DROPPED when >= 60 % covered by "dashed line"
             (>= 0.45) or "solid line" (>= 0.85) -- measured: dash-like "text" instances carry dashed 0.43-0.83 or
             solid 0.91-0.94, the six real arrows dashed <= 0.31 and solid <= 0.51 -- or when the ground footprint
             is narrower than 0.35 m or longer than 8 m
  edge       v1 (the drivable ring touching sidewalk / grass / curb / guardrail, plus curb pixels hugging the road)
  priority   drivable < line < arrow/text < hatched < crosswalk < edge
The ego bonnet and ego-attached artefacts are removed afterwards by sam3map_refine.py (clip-level, exact).
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3vendor"); sys.path.insert(0, "/home/nvidia/sam3paint")
import numpy as np
import cv2
from PIL import Image
import sam3_smoke as S
import sam3_paint as P

import os
V2 = Path(os.environ.get("SAM3MAP_ROOT", "/home/nvidia/qwendrive/v2"))                    # front-only clips: /home/nvidia/sam3map/front
VIEWS = os.environ.get("SAM3MAP_VIEWS", "CAM_F0,CAM_L0,CAM_R0,CAM_L1,CAM_R1,CAM_L2,CAM_R2").split(",")
PROMPTS = {
    "road": {"road": 0.50, "asphalt road surface": 0.50},
    "nondrive": {"sidewalk": 0.40, "grass": 0.40, "curb": 0.40, "guardrail": 0.40},
    "line": {"lane marking": 0.40, "road marking": 0.40, "stop line": 0.45},
    "gate": {"dashed line": 0.45, "solid line": 0.85},
    "crosswalk": {"crosswalk": 0.50, "zebra crossing": 0.50, "pedestrian crossing": 0.50},
    "diag": {"diagonal stripes on road": 0.40},
    "symbol": {"arrow painted on road": 0.45, "text painted on road": 0.45},
}
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
    return [(float(s), ms[k].astype(bool)) for k, s in enumerate(sc) if s >= thr and ms[k].any()]


def onroad(m, road, zone, H, W):
    n = int(m.sum())
    ys, xs = np.nonzero(m)
    y0, y1 = max(0, ys.min() - 8), min(H, ys.max() + 9); x0, x1 = max(0, xs.min() - 8), min(W, xs.max() + 9)
    mc = m[y0:y1, x0:x1].astype(np.uint8)
    ring = (cv2.dilate(mc, np.ones((13, 13), np.uint8)) > 0) & (mc == 0)
    ring_road = float((ring & road[y0:y1, x0:x1]).sum()) / max(int(ring.sum()), 1)
    return ring_road >= 0.35 and float((m & zone).sum()) / max(n, 1) >= 0.60


def footprint(m, K, R, t, grid, fb):
    xy = P.lift(m, K, R, t, grid, fb, stride=2)
    if len(xy) < 12:
        return 0.0, 0.0
    (_, _), (w, h), _ = cv2.minAreaRect(np.asarray(xy, np.float32))
    return float(min(w, h)), float(max(w, h))


def classify(proc, img, lift_ctx):
    state = proc.set_image(img)
    H, W = img.height, img.width
    inst = {fam: [(p, s, m) for p, t in table.items() for s, m in instances(proc, state, p, t)] for fam, table in PROMPTS.items()}
    road = np.zeros((H, W), bool)
    for p, s, m in inst["road"]:
        road |= m
    nondrive = np.zeros((H, W), bool); curb = np.zeros((H, W), bool)
    for p, s, m in inst["nondrive"]:
        nondrive |= m
        if p == "curb":
            curb |= m
    zone = cv2.dilate(road.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    road_px = max(int(road.sum()), 1)
    st = {"accepted": {}, "rejected_offroad": 0, "rejected_zebra_cap": 0, "crosswalk_by_stripes": 0, "crosswalk_to_hatched": 0,
          "hatched_standalone": 0, "symbol_dropped_dashed_solid": 0, "symbol_dropped_footprint": 0}
    ok = {}
    for fam in ("line", "crosswalk", "diag", "symbol"):
        ok[fam] = []
        for p, s, m in inst[fam]:
            if onroad(m, road, zone, H, W):
                ok[fam].append((p, s, m))
            else:
                st["rejected_offroad"] += 1
    line = np.zeros((H, W), bool)
    for p, s, m in ok["line"]:
        line |= m; st["accepted"][p] = st["accepted"].get(p, 0) + 1
    n_lab, lab = cv2.connectedComponents(line.astype(np.uint8), connectivity=8)
    comp_px = np.bincount(lab.ravel(), minlength=n_lab)
    diag = np.zeros((H, W), bool)
    for p, s, m in ok["diag"]:
        diag |= m
    cross = np.zeros((H, W), bool); hatch = np.zeros((H, W), bool)
    for p, s, m in ok["crosswalk"]:
        md = cv2.dilate(m.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        inside = np.bincount(lab[md].ravel(), minlength=n_lab)
        members = np.nonzero((inside >= 0.5 * comp_px) & (np.arange(n_lab) > 0))[0]
        if m.sum() > 0.15 * road_px:
            if len(members) < 3:
                st["rejected_zebra_cap"] += 1
                continue
            st["crosswalk_by_stripes"] += 1
        stripes = np.isin(lab, members)
        if stripes.sum() and (stripes & diag).sum() >= 0.3 * stripes.sum():
            hatch |= m | stripes; st["crosswalk_to_hatched"] += 1
        else:
            cross |= m | stripes; st["accepted"][p] = st["accepted"].get(p, 0) + 1
    for p, s, m in ok["diag"]:
        if s >= 0.45 and not (m & (cross | hatch)).any():
            hatch |= m; st["hatched_standalone"] += 1
    gate = [(p, s, m) for p, s, m in inst["gate"]]
    sym = np.zeros((H, W), bool)
    for p, s, m in ok["symbol"]:
        n = int(m.sum())
        if any((m & g).sum() >= 0.6 * n for gp, gs, g in gate):
            st["symbol_dropped_dashed_solid"] += 1
            continue
        w, l = footprint(m, *lift_ctx)
        if w < 0.35 or l > 8.0:
            st["symbol_dropped_footprint"] += 1
            continue
        sym |= m; st["accepted"][p] = st["accepted"].get(p, 0) + 1
    drivable = road | line | cross | hatch | sym
    k7 = np.ones((7, 7), np.uint8)
    ring = (cv2.dilate(drivable.astype(np.uint8), k7) > 0) & ~drivable
    near_nondrive = cv2.dilate(nondrive.astype(np.uint8), k7) > 0
    edge = (ring & near_nondrive) | (curb & zone)
    cls = np.zeros((H, W), np.uint8)
    cls[drivable] = 1; cls[line] = 2; cls[sym] = 4; cls[hatch] = 6; cls[cross] = 3; cls[edge] = 5
    return cls, st


def main():
    c8 = sys.argv[1]
    sel = sys.argv[2] if len(sys.argv) > 2 else "all"
    out = Path(f"/home/nvidia/sam3map/{c8}_v3raw"); out.mkdir(parents=True, exist_ok=True)
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
        lp = fd / "lidar.npy"
        grid, fb = P.ground_grid(np.load(lp).astype(np.float64)) if lp.exists() else (np.full(2500, np.nan), 0.0)
        T = np.array(poses[tok]["T_world_rig"])
        lifted = {k: [] for k in range(1, 7)}; ranges = {k: [] for k in range(1, 7)}
        small, stats_all = {}, {}
        for cam in VIEWS:
            i = fr["cam_order"].index(cam)
            K, R, t = c["cam_intrinsic"][i], c["sensor2lidar_rotation"][i], c["sensor2lidar_translation"][i]
            img = Image.open(fd / "images" / f"{cam}.jpg").convert("RGB")
            cls, st = classify(proc, img, (K, R, t, grid, fb))
            stats_all[cam] = st
            small[cam] = cv2.resize(cls, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)
            for k in range(1, 7):
                m = np.isin(cls, (1, 2, 3, 4, 6)) if k == 1 else (cls == k)
                if m.any():
                    xy = P.lift(m, K, R, t, grid, fb, stride=2)
                    if len(xy):
                        lifted[k].append((np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ T.T)[:, :2].astype(np.float32))
                        ranges[k].append(np.hypot(xy[:, 0] - t[0], xy[:, 1] - t[1]).astype(np.float16))   # from the camera
        np.savez_compressed(out / f"{j:03d}.npz", tok=tok, T_world_rig=T, stats=json.dumps(stats_all), ver="v3",
                            **{f"cls_{cam}": small[cam] for cam in VIEWS},
                            **{f"pts_{k}": (np.concatenate(lifted[k]) if lifted[k] else np.zeros((0, 2), np.float32)) for k in range(1, 7)},
                            **{f"rng_{k}": (np.concatenate(ranges[k]) if ranges[k] else np.zeros((0,), np.float16)) for k in range(1, 7)})
        tot = {k: sum(len(a) for a in lifted[k]) for k in range(1, 7)}
        f0 = stats_all[VIEWS[0]]
        print(f"  j={j:3d} {time.time() - t0:6.0f}s pts {tot}  F0 {f0['accepted']} x2h {f0['crosswalk_to_hatched']} "
              f"symdrop {f0['symbol_dropped_dashed_solid']}+{f0['symbol_dropped_footprint']}", flush=True)
    print("ZZSAM3MAP-DONEZZ %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
