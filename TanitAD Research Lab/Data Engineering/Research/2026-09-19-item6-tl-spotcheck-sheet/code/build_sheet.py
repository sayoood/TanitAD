"""Item 6 readiness: a ~50-frame traffic-light colour spot-check sheet for the PI (~15 min).

What is checked: does the lamp the car responds to show the colour in the LABEL? The label
is the traffic-light tactical goal of the v8.1 release — an **Alpamayo-derived teacher
signal** read from the VLM's reasoning text, NOT ground truth (V8_MANIFEST's own rule).

Which frame: the LAST frame Alpamayo saw — its anchor, `ALPAMAYO_T0_S = 5.1` s into the
clip (`stack/tanitad/data/alpamayo_records.py:149`), on the raw timeline zeroed at the
first egomotion sample (camera and egomotion share a clock: the first camera frame lands
+0.063 s after it). The label's own `t_nominal_s` is a band-midpoint CONVENTION
(`time_basis: untimed`), not evidence of when the light was seen, so it is not used.

Sampling: stratified by the label colour, fixed seed. Records carrying MORE than one
traffic-light key (a transition, e.g. RED and GREEN) are excluded and counted.
Sources: the v8.1 blobs on C: (md5-verified below), the local mirror of the mp4s
(byte-identical to HF, 4,719/4,719 sha256), and the local egomotion parquets.
🔒 Every output is keyed by sha12; no clip id is written anywhere.
"""
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release/v8")
MD5 = {"s2_labels_v8_train.jsonl.gz": "b45377a1f25263b5c0f3d318c126b1ac",
       "s2_labels_v8_eval.jsonl.gz": "eefc38d1453bd1c73802d44d45affced"}
CAM = Path("C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov")
EGO = Path("C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo")
#: per-clip principal point. ⛔ LOAD-BEARING: two rigs (A cy~543, B cy~755); a zoom band
#: placed by FRAME fraction showed empty sky on a rig-B clip (first build, 2026-09-19).
CY = pd.read_parquet("D:/Projects/TanitAD-artifacts/_s2build-copy-20260919/release/"
                     "front_wide_cy.parquet").set_index("clip_id")
ALPAMAYO_T0_S = 5.1
SEED = 20260919
PLAN = {"RED": 14, "GREEN": 14, "YELLOW": 12, "colourless": 10}     # 50
KEY = {"TRAFFIC_LIGHT_REACT_RED": "RED", "TRAFFIC_LIGHT_REACT_GREEN": "GREEN",
       "TRAFFIC_LIGHT_REACT_YELLOW": "YELLOW", "TRAFFIC_LIGHT_REACT": "colourless"}
OUT = Path(sys.argv[1])


def sha12(c):
    return hashlib.sha256(c.encode()).hexdigest()[:12]


# ── 1. the labels, verified ─────────────────────────────────────────────────
recs = []
for name, want in MD5.items():
    p = REL / name
    got = hashlib.md5(p.read_bytes()).hexdigest()
    assert got == want, "%s md5 %s, expected %s" % (name, got, want)
    recs += [json.loads(l) for l in gzip.open(p, "rt", encoding="utf-8") if l.strip()]
pool, multi = {k: [] for k in PLAN}, 0
for r in recs:
    goals = (r.get("g_tac") or {}).get("goals") or {}
    tl = [KEY[k] for k in goals if k in KEY]
    if len(tl) > 1:
        multi += 1
        continue
    if tl:
        g = goals[next(k for k in goals if k in KEY)]
        pool[tl[0]].append({"clip": r["clip_id"], "label": tl[0],
                            "grounded": bool(g.get("grounded")),
                            "disputed": bool(g.get("disputed")),
                            "reason": ((r.get("tac_SIT") or {}).get("text") or "")[:220]})
print("pool:", {k: len(v) for k, v in pool.items()}, "| multi-key records excluded:", multi)

# ── 2. the stratified sample (top up from RED/GREEN if a rare stratum is short) ─
rng = random.Random(SEED)
take = dict(PLAN)
short = sum(max(0, PLAN[k] - len(pool[k])) for k in PLAN)
for k in PLAN:
    take[k] = min(PLAN[k], len(pool[k]))
for k in ("RED", "GREEN"):
    extra = min(short, len(pool[k]) - take[k])
    take[k] += extra
    short -= extra
rows = []
for k in ("RED", "GREEN", "YELLOW", "colourless"):
    pick = sorted(pool[k], key=lambda x: sha12(x["clip"]))
    rows += rng.sample(pick, take[k])
print("sample:", {k: sum(1 for r in rows if r["label"] == k) for k in PLAN}, "total", len(rows))


# ── 3. the frame Alpamayo last saw, composed as ONE image ───────────────────
def frame_at(clip):
    ts = pd.read_parquet(CAM / ("%s.timestamps.parquet" % clip))["timestamp"].to_numpy()
    e0 = pd.read_parquet(EGO / ("%s.parquet" % clip), columns=["timestamp"])["timestamp"].min()
    target = e0 + ALPAMAYO_T0_S * 1e6
    idx = int(np.argmin(np.abs(ts - target)))
    with av.open(str(CAM / ("%s.mp4" % clip))) as f:
        for i, fr in enumerate(f.decode(video=0)):
            if i == idx:
                return fr.to_image(), idx, (ts[idx] - e0) / 1e6
    raise RuntimeError("frame %d not decoded for %s" % (idx, sha12(clip)))


def compose(img, cy):
    """Full frame for context + a native-resolution band around the clip's OWN horizon
    (cy - 0.30 H .. cy + 0.05 H, central half): where distant and mid-range lamps sit."""
    W, H = img.size
    y0, y1 = max(0, int(cy - 0.30 * H)), min(H, int(cy + 0.05 * H))
    full = img.resize((960, int(960 * H / W)))
    crop = img.crop((int(0.25 * W), y0, int(0.75 * W), y1))
    crop = crop.resize((960, int(960 * crop.size[1] / crop.size[0])))
    canvas = Image.new("RGB", (960, full.size[1] + crop.size[1] + 6), (40, 40, 40))
    canvas.paste(full, (0, 0))
    canvas.paste(crop, (0, full.size[1] + 6))
    d = ImageDraw.Draw(canvas)
    sy = full.size[1] / H
    d.rectangle([240, int(y0 * sy), 720, int(y1 * sy)], outline=(255, 210, 0), width=2)
    return canvas


media = OUT / "media"
media.mkdir(parents=True, exist_ok=True)
sample = []
for n, r in enumerate(rows, 1):
    img, idx, t = frame_at(r["clip"])
    name = "%02d_%s.jpg" % (n, sha12(r["clip"]))
    cy, rig = float(CY.loc[r["clip"], "cy"]), str(CY.loc[r["clip"], "rig"])
    compose(img, cy).save(media / name, "JPEG", quality=84)
    sample.append({"n": n, "sha12": sha12(r["clip"]), "label": r["label"],
                   "grounded": r["grounded"], "disputed": r["disputed"],
                   "frame_index": idx, "frame_time_s": round(float(t), 3), "rig": rig, "cy": cy,
                   "image": "media/" + name, "reason": r["reason"]})
    print("  %02d %-10s %s  frame %d @ %.2f s" % (n, r["label"], sha12(r["clip"]), idx, t))

json.dump({"seed": SEED, "plan": PLAN, "taken": take, "pool": {k: len(v) for k, v in pool.items()},
           "multi_key_excluded": multi, "alpamayo_t0_s": ALPAMAYO_T0_S, "rows": sample},
          open(OUT / "raw" / "sample.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
with open(OUT / "sheet_template.csv", "w", encoding="utf-8", newline="") as f:
    f.write("n,sha12,label,grounded,frame_time_s,verdict,notes\n")
    for s in sample:
        f.write("%d,%s,%s,%s,%.2f,,\n" % (s["n"], s["sha12"], s["label"], s["grounded"], s["frame_time_s"]))
print("wrote %d images, raw/sample.json, sheet_template.csv" % len(sample))
