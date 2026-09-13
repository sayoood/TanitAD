"""v3 road-paint classification from SAM3 instances, and its replay on the prompt bank (v1 rule vs v3 rule).

Why v3 (PI, 2026-09-13, after the v2 videos): hatched chevron areas came out as a mix of line / crosswalk / arrow-text,
dashed lines were tagged as text, and a zebra crossing right in front of the car stayed yellow "line" stripes.
Measured causes: v1's crosswalk AREA CAP rejected the near crosswalk (2 instances); on the nuPlan demo's true map
"text painted on road" puts 49 % of its cells on true lane lines; "hatched road marking" fires on everything while
"chevron road marking" never lands on a true line or crosswalk; crosswalk confidence 0.5 halves recall (held-out
2-fold: 4 prompts at 0.25 lifts recall 0.331 -> 0.426, F1 0.471 -> 0.494; shuffled control 0.027).

v3 RULES (written before the replay):
  on-road   every paint instance: >= 60 % within 15 px of the road mask AND >= 35 % of its 13 px ring is road (v1)
  line      lane marking 0.40, road marking 0.40, stop line 0.45 (v1 -- the demo sweep's winners did not hold out)
  crosswalk crosswalk / zebra crossing / pedestrian crossing / crosswalk stripes at 0.25; an instance is accepted
            if it covers <= 15 % of the road pixels OR contains >= 3 line stripes (a stripe = a line component with
            >= 50 % of its pixels inside the instance dilated 5 px) -- stripes, not size, separate a zebra from a blob;
            stripes inside an accepted crosswalk become crosswalk
  hatch     chevron road marking at 0.45 -> class 6 HATCHED AREA (the region, with its stripes)
  symbol    arrow painted on road 0.45, text painted on road 0.45; a symbol instance is DROPPED when >= 50 % of it
            overlaps accepted line instances whose best score >= its own score - 0.05 (a dash is not a word)
  priority  line < symbol < hatched < crosswalk
Classes: 2 line, 3 crosswalk, 4 symbol, 6 hatched area.

REPLAY OF v3 ON 84 VIEWS OF OUR CLIPS (prompt bank, 12 frames x 7 views) -- FAILED, and why:
  crosswalk pixels x3.8 vs v1 (1,160 small instances accepted at 0.25): yellow hatched zones, dashes and an arrow
  became crosswalk; all 58 symbol instances were dropped because the generic "road marking" instance covers arrows
  and text too. The demo-tuned 0.25 does not transfer (4 US / SG frames, no hatched zones). Measured on the replay
  views: SAM3 scores "crosswalk" 0.79-0.89 on the hatched zones -- as high as on the real zebra (0.85) -- so NO
  crosswalk threshold separates them; "diagonal stripes on road" does (hatched 0.53 / 0.46, painted triangle 0.49,
  real zebra 0.22, give-way dashes 0.30).

v3b RULES (written before their replay):
  crosswalk crosswalk / zebra crossing / pedestrian crossing at 0.50 (v1 threshold; "crosswalk stripes" dropped: it
            scores 0.87-0.88 on single hatched stripes); area cap replaced by the >= 3 stripes test
  hatched   a crosswalk region whose inside stripes are >= 30 % covered by "diagonal stripes on road" (>= 0.40) is a
            HATCHED AREA (6); a standalone "diagonal stripes on road" instance >= 0.45 passing the on-road test is
            hatched too
  symbol    arrow / text at 0.45, lifted to the ground: kept only if the minimum-area rectangle of its ground
            footprint is >= 0.35 m wide and <= 8 m long (a dash is ~0.15 m wide)
  line      unchanged (v1)"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3paint")
import numpy as np
import cv2

V1 = {"line": {"lane marking": 0.40, "road marking": 0.40, "stop line": 0.45}, "crosswalk": {"crosswalk": 0.50, "zebra crossing": 0.50},
      "symbol": {"arrow painted on road": 0.45, "text painted on road": 0.45}}
V3 = {"line": {"lane marking": 0.40, "road marking": 0.40, "stop line": 0.45},
      "crosswalk": {"crosswalk": 0.25, "zebra crossing": 0.25, "pedestrian crossing": 0.25, "crosswalk stripes": 0.25},
      "hatch": {"chevron road marking": 0.45},
      "symbol": {"arrow painted on road": 0.45, "text painted on road": 0.45}}


V3B = {"line": {"lane marking": 0.40, "road marking": 0.40, "stop line": 0.45},
       "crosswalk": {"crosswalk": 0.50, "zebra crossing": 0.50, "pedestrian crossing": 0.50},
       "diag": {"diagonal stripes on road": 0.40},
       "symbol": {"arrow painted on road": 0.45, "text painted on road": 0.45}}


def onroad_ok(r):
    return r["onroad"] >= 0.6 and r["ring_road"] >= 0.35


def ground_footprint(m, lift_ctx):
    """(width_m, length_m, n) of a 960x540 instance mask lifted to the ground; None when it cannot be lifted."""
    if lift_ctx is None:
        return None
    import sam3_paint as P
    K, R, t, grid, fb = lift_ctx
    full = np.repeat(np.repeat(m, 2, axis=0), 2, axis=1)
    xy = P.lift(full, K, R, t, grid, fb, stride=2)
    if len(xy) < 12:
        return (0.0, 0.0, int(len(xy)))
    (_, _), (w, h), _ = cv2.minAreaRect(np.asarray(xy, np.float32))
    return (float(min(w, h)), float(max(w, h)), int(len(xy)))


def classify_v3b(rows, masks, road_px, lift_ctx):
    H, W = masks.shape[1:]
    acc = {fam: [(r, m) for r, m in zip(rows, masks) if r["prompt"] in V3B[fam] and r["score"] >= V3B[fam][r["prompt"]] and onroad_ok(r)] for fam in V3B}
    line = np.zeros((H, W), bool)
    for r, m in acc["line"]:
        line |= m
    n_lab, lab = cv2.connectedComponents(line.astype(np.uint8), connectivity=8)
    comp_px = np.bincount(lab.ravel(), minlength=n_lab)
    diag = np.zeros((H, W), bool)
    for r, m in acc["diag"]:
        diag |= m
    st = {"crosswalk_small": 0, "crosswalk_by_stripes": 0, "crosswalk_rejected_blob": 0, "crosswalk_to_hatched": 0,
          "hatched_standalone": 0, "symbols_kept": 0, "symbols_dropped_thin_or_long": 0, "symbols_unliftable": 0}
    cross = np.zeros((H, W), bool); hatch = np.zeros((H, W), bool)
    for r, m in acc["crosswalk"]:
        md = cv2.dilate(m.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        inside = np.bincount(lab[md].ravel(), minlength=n_lab)
        members = np.nonzero((inside >= 0.5 * comp_px) & (np.arange(n_lab) > 0))[0]
        if m.sum() <= 0.15 * road_px:
            st["crosswalk_small"] += 1
        elif len(members) >= 3:
            st["crosswalk_by_stripes"] += 1
        else:
            st["crosswalk_rejected_blob"] += 1
            continue
        stripes = np.isin(lab, members)
        region = m | stripes
        if stripes.sum() and (stripes & diag).sum() >= 0.3 * stripes.sum():
            hatch |= region; st["crosswalk_to_hatched"] += 1
        else:
            cross |= region
    for r, m in acc["diag"]:
        if r["score"] >= 0.45 and not (m & (cross | hatch)).any():
            hatch |= m; st["hatched_standalone"] += 1
    sym = np.zeros((H, W), bool)
    for r, m in acc["symbol"]:
        fp = ground_footprint(m, lift_ctx)
        if fp is None or fp[2] < 12:
            st["symbols_unliftable"] += 1
            continue
        if fp[0] < 0.35 or fp[1] > 8.0:
            st["symbols_dropped_thin_or_long"] += 1
            continue
        sym |= m; st["symbols_kept"] += 1
    cls = np.zeros((H, W), np.uint8)
    cls[line] = 2; cls[sym] = 4; cls[hatch] = 6; cls[cross] = 3
    return cls, st


def classify_v1(rows, masks, road_px):
    H, W = masks.shape[1:]
    cls = np.zeros((H, W), np.uint8); st = {"crosswalk_cap_rejects": 0}
    for fam, code in (("line", 2), ("crosswalk", 3), ("symbol", 4)):
        for r, m in zip(rows, masks):
            t = V1[fam].get(r["prompt"])
            if t is None or r["score"] < t or not onroad_ok(r):
                continue
            if fam == "crosswalk" and m.sum() > 0.15 * road_px:
                st["crosswalk_cap_rejects"] += 1
                continue
            cls[m] = code
    return cls, st


def classify_v3(rows, masks, road_px):
    H, W = masks.shape[1:]
    acc = {fam: [(r, m) for r, m in zip(rows, masks) if r["prompt"] in V3[fam] and r["score"] >= V3[fam][r["prompt"]] and onroad_ok(r)] for fam in V3}
    line = np.zeros((H, W), bool); line_score = np.zeros((H, W), np.float32)
    for r, m in acc["line"]:
        line |= m; line_score[m] = np.maximum(line_score[m], r["score"])
    n_lab, lab = cv2.connectedComponents(line.astype(np.uint8), connectivity=8)
    comp_px = np.bincount(lab.ravel(), minlength=n_lab)
    st = {"crosswalk_accepted_by_stripes": 0, "crosswalk_accepted_small": 0, "crosswalk_rejected_blob": 0,
          "symbols_dropped_as_line": 0, "hatch_accepted": len(acc["hatch"])}
    cross = np.zeros((H, W), bool)
    for r, m in acc["crosswalk"]:
        md = cv2.dilate(m.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        inside = np.bincount(lab[md].ravel(), minlength=n_lab)
        stripes = int(((inside >= 0.5 * comp_px) & (np.arange(n_lab) > 0)).sum())
        if m.sum() <= 0.15 * road_px:
            st["crosswalk_accepted_small"] += 1
        elif stripes >= 3:
            st["crosswalk_accepted_by_stripes"] += 1
        else:
            st["crosswalk_rejected_blob"] += 1
            continue
        cross |= m
        members = np.nonzero((inside >= 0.5 * comp_px) & (np.arange(n_lab) > 0))[0]
        cross |= np.isin(lab, members)
    hatch = np.zeros((H, W), bool)
    for r, m in acc["hatch"]:
        md = cv2.dilate(m.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        inside = np.bincount(lab[md].ravel(), minlength=n_lab)
        hatch |= m | np.isin(lab, np.nonzero((inside >= 0.5 * comp_px) & (np.arange(n_lab) > 0))[0])
    sym = np.zeros((H, W), bool)
    for r, m in acc["symbol"]:
        ov = m & line
        if ov.sum() >= 0.5 * m.sum() and float(line_score[ov].max()) >= r["score"] - 0.05:
            st["symbols_dropped_as_line"] += 1
            continue
        sym |= m
    cls = np.zeros((H, W), np.uint8)
    cls[line] = 2; cls[sym] = 4; cls[hatch] = 6; cls[cross] = 3
    return cls, st


DEBUG_KEY = ""


def classify_v31(rows, masks, road_px, lift_ctx, cam):
    """v3b with the crosswalk / hatched split decided by stripe geometry on the ground (stripe_geometry.py)."""
    import stripe_geometry as SG
    H, W = masks.shape[1:]
    acc = {fam: [(r, m) for r, m in zip(rows, masks) if r["prompt"] in V3B[fam] and r["score"] >= V3B[fam][r["prompt"]] and onroad_ok(r)] for fam in V3B}
    line = np.zeros((H, W), bool)
    for r, m in acc["line"]:
        line |= m
    n_lab, lab = cv2.connectedComponents(line.astype(np.uint8), connectivity=8)
    comp_px = np.bincount(lab.ravel(), minlength=n_lab)
    diag = np.zeros((H, W), bool)
    for r, m in acc["diag"]:
        diag |= m
    st = {"zebra": 0, "hatched_geom": 0, "fallback_prompt_F0": 0, "fallback_crosswalk": 0, "rejected_blob": 0, "hatched_standalone_F0": 0}
    cross = np.zeros((H, W), bool); hatch = np.zeros((H, W), bool)
    up = lambda x: np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)
    K, R, t, grid, fb = lift_ctx
    for r, m in acc["crosswalk"]:
        md = cv2.dilate(m.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        inside = np.bincount(lab[md].ravel(), minlength=n_lab)
        members = np.nonzero((inside >= 0.5 * comp_px) & (np.arange(n_lab) > 0))[0]
        if m.sum() > 0.15 * road_px and len(members) < 3:
            st["rejected_blob"] += 1
            continue
        stripes = np.isin(lab, members)
        verdict, info = SG.decide2(up(stripes), up(m | stripes), K, R, t, grid, fb) if len(members) else ("too_few", {})
        if DEBUG_KEY and DEBUG_KEY.endswith(cam):
            print(f"    [{DEBUG_KEY}] score {r['score']:.2f} members {len(members)} -> {verdict} {info}")
        if verdict == "zebra":
            is_hatch = False; st["zebra"] += 1
        elif verdict == "hatched":
            is_hatch = True; st["hatched_geom"] += 1
        elif cam == "CAM_F0":
            is_hatch = bool(stripes.sum()) and (stripes & diag).sum() >= 0.3 * stripes.sum(); st["fallback_prompt_F0"] += 1
        else:
            is_hatch = False; st["fallback_crosswalk"] += 1
        (hatch if is_hatch else cross)[:] |= m | stripes
    if cam == "CAM_F0":
        for r, m in acc["diag"]:
            if r["score"] >= 0.45 and not (m & (cross | hatch)).any():
                hatch |= m; st["hatched_standalone_F0"] += 1
    sym = np.zeros((H, W), bool)
    for r, m in acc["symbol"]:
        fp = ground_footprint(m, lift_ctx)
        if fp is None or fp[2] < 12 or fp[0] < 0.35 or fp[1] > 8.0:
            continue
        sym |= m
    cls = np.zeros((H, W), np.uint8)
    cls[line] = 2; cls[sym] = 4; cls[hatch] = 6; cls[cross] = 3
    return cls, st


def replay(bank_dir, out_dir, sheet_views):
    COL = {2: (255, 214, 0), 3: (0, 220, 210), 4: (255, 60, 220), 6: (255, 140, 0)}
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    tot = {"v1": {}, "v3": {}}
    tiles = []
    for f in sorted(bank_dir.glob("*.npz")):
        d = np.load(f)
        rows = json.loads(str(d["meta"]))
        masks_all = np.unpackbits(d["masks"], axis=1)[:, :540 * 960].reshape(-1, 540, 960).astype(bool)
        for cam in sorted({r["cam"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["cam"] == cam]
            R_ = [rows[i] for i in idx]; M_ = masks_all[idx]
            road = np.zeros((540, 960), bool)
            for r, m in zip(R_, M_):
                if r["fam"] == "road" and r["score"] >= 0.5:
                    road |= m
            road_px = max(int(road.sum()), 1)
            c8_, _ = f.stem.split("_")
            fd = Path("/home/nvidia/qwendrive/v2") / f"seq_{c8_}" / str(d["tok"])
            cal = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
            import sam3_paint as P
            grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64))
            ci = fr["cam_order"].index(cam)
            ctx = (cal["cam_intrinsic"][ci], cal["sensor2lidar_rotation"][ci], cal["sensor2lidar_translation"][ci], grid, fb)
            global DEBUG_KEY
            DEBUG_KEY = f"{f.stem}_{cam}" if f"{f.stem}_{cam}" in sheet_views else ""
            c1, s1 = classify_v3b(R_, M_, road_px, ctx); c3, s3 = classify_v31(R_, M_, road_px, ctx, cam)
            for tag, c, s in (("v1", c1, s1), ("v3", c3, s3)):
                for k in (2, 3, 4, 6):
                    tot[tag][f"px_class{k}"] = tot[tag].get(f"px_class{k}", 0) + int((c == k).sum())
                for k, v in s.items():
                    tot[tag][k] = tot[tag].get(k, 0) + v
            key = f"{f.stem}_{cam}"
            if key in sheet_views:
                c8, j = f.stem.split("_")
                seq = Path("/home/nvidia/qwendrive/v2") / f"seq_{c8}"
                tok = str(d["tok"])
                img = np.asarray(Image.open(seq / tok / "images" / f"{cam}.jpg").convert("RGB").resize((960, 540))).astype(np.float32)
                pair = []
                for c in (c1, c3):
                    a = img.copy()
                    for k, col in COL.items():
                        a[c == k] = 0.3 * a[c == k] + 0.7 * np.array(col, np.float32)
                    pair.append(a)
                tiles.append(np.concatenate(pair, axis=1))
    if tiles:
        sheet = np.concatenate([cv2.resize(t, (1280, 360)) for t in tiles], axis=0)
        Image.fromarray(sheet.clip(0, 255).astype(np.uint8)).save(out_dir / "v1_vs_v3.jpg", quality=88)
    (out_dir / "replay_totals.json").write_text(json.dumps(tot, indent=1))
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    views = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else set()
    replay(Path("/home/nvidia/sam3map/promptbank/ours"), Path("/home/nvidia/sam3map/promptbank/replay"), views)
    print("ZZREPLAY-DONEZZ")
