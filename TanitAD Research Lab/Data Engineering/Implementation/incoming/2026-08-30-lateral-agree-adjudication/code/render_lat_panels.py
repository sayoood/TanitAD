"""Render inspection panels for the D-LAT-AGREE cases — for the PI to judge.

Numbers cannot settle "is a 1.2 m drift a lane change or a wobble", and they
cannot settle who is right when the two label sources disagree on direction.
This puts the evidence side by side, per clip:

  * three camera frames across the label window (what the scene was)
  * a metric BEV of the ego path in the KEY-FRAME frame, with the signed
    lateral track and the +/-1.0 m NUDGE threshold drawn (what geometry measured)
  * both label claims and the Alpamayo CoT sentence (what each source said)

Follows the standing viz preference: camera projection + metric BEV together,
with the decoded manoeuvre as text overlay.
"""
import gzip
import io
import json
import math
import re
import tarfile
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release/tanitad-v7-training-corpus")
CAM = Path("C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov")
OUT = Path("C:/Users/Admin/tanitad-wt/_s2build/release/lat_panels")
OUT.mkdir(exist_ok=True)
NUDGE_LAT_M = 1.0
VERB = re.compile(r"\b(nudge|steer|swerve|merge|turn|shift|move|pull|drift|veer|bear)\w*"
                  r"\s+(?:to\s+the\s+|to\s+|slightly\s+)?(left|right)\b", re.I)

rows = [json.loads(x) for x in gzip.open(REL / "labels" / "s2_labels_v7.jsonl.gz",
                                         "rt", encoding="utf-8") if x.strip()]
ego_tar = tarfile.open(REL / "egomotion" / "egomotion_alpamayo.tar")
ego_idx = {m.name.split(".")[0]: m for m in ego_tar.getmembers() if m.isfile()}


def side_of(c):
    t = (c or "").upper()
    return "left" if t.endswith(("_L", "_LEFT")) else (
        "right" if t.endswith(("_R", "_RIGHT")) else "straight")


def poses_of(cid):
    d = pd.read_parquet(io.BytesIO(ego_tar.extractfile(ego_idx[cid]).read()))
    x, y = d.x.to_numpy(), d.y.to_numpy()
    qz, qw = d.qz.to_numpy(), d.qw.to_numpy()
    yaw = 2.0 * np.arctan2(qz, qw)                      # planar yaw
    t = d.timestamp.to_numpy() / 1e6                    # us -> s
    return x, y, yaw, t


def lateral_track(x, y, yaw, k):
    """Signed lateral offset from the key-frame heading (+ = left) — the same
    quantity `ego_manoeuvre` uses to decide NUDGE, recomputed here so the panel
    shows the DECIDING evidence rather than a proxy."""
    c, s = math.cos(-yaw[k]), math.sin(-yaw[k])
    fwd = c * (x - x[k]) - s * (y - y[k])
    lat = s * (x - x[k]) + c * (y - y[k])
    return fwd, lat


def frames_of(cid, times, t0):
    p = CAM / f"{cid}.mp4"
    want = sorted(t0 + t for t in times)
    got, wi = [], 0
    with av.open(str(p)) as c:
        st = c.streams.video[0]
        fps = float(st.average_rate or 30.0)
        for i, fr in enumerate(c.decode(video=0)):
            if wi >= len(want):
                break
            if i / fps >= want[wi]:
                got.append(fr.to_ndarray(format="rgb24"))
                wi += 1
    return got


def panel(r, tag):
    cid = r["clip_id"]
    if cid not in ego_idx or not (CAM / f"{cid}.mp4").exists():
        return None
    alp = ((r.get("alpamayo") or {}).get("lateral") or {}).get("alpamayo_side")
    geom = (r.get("a_tac") or {}).get("lat")
    ev = ((r.get("cot_tokens") or {}).get("evidence") or "").strip()
    t0 = float(r.get("t0_s") or 0.0)
    x, y, yaw, t = poses_of(cid)
    k = int(np.argmin(np.abs(t - t0)))
    hi = int(np.argmin(np.abs(t - (t0 + 6.0))))          # tactical band
    hi = max(hi, k + 2)
    fwd, lat = lateral_track(x, y, yaw, k)
    seg_f, seg_l = fwd[k:hi], lat[k:hi]
    j = int(np.argmax(np.abs(seg_l))) if len(seg_l) else 0
    peak = float(seg_l[j]) if len(seg_l) else 0.0

    imgs = frames_of(cid, [0.0, 3.0, 6.0], t0)
    fig = plt.figure(figsize=(15, 7.2), dpi=100)
    for i, im in enumerate(imgs[:3]):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(im)
        ax.set_title(f"t0 + {i*3.0:.0f} s", fontsize=9)
        ax.axis("off")
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(seg_f, seg_l, lw=2.5, color="#1f77b4", label="ego path (key-frame frame)")
    ax.axhline(0, color="#888", lw=0.8)
    for sgn, lab in ((1, "+1.0 m = NUDGE_L threshold"), (-1, "-1.0 m = NUDGE_R threshold")):
        ax.axhline(sgn * NUDGE_LAT_M, color="#d62728", ls="--", lw=1.1,
                   label=lab if sgn > 0 else None)
    ax.axhline(-NUDGE_LAT_M, color="#d62728", ls="--", lw=1.1)
    if len(seg_l):
        ax.plot([seg_f[j]], [peak], "o", ms=9, color="#d62728")
        ax.annotate(f"peak lateral {peak:+.2f} m", (seg_f[j], peak),
                    textcoords="offset points", xytext=(10, 10), fontsize=11,
                    color="#d62728", weight="bold")
    ax.set_xlabel("forward distance from key frame (m)")
    ax.set_ylabel("lateral offset (m)   + = LEFT")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    lim = max(2.0, abs(peak) * 1.4)
    ax.set_ylim(-lim, lim)

    verb = VERB.search(ev)
    vtxt = f"text verb: '{verb.group(0)}'" if verb else "text: no ego-manoeuvre verb"
    head = (f"[{tag}]  {cid[:13]}   |   ALPAMAYO side: {str(alp).upper()}   |   "
            f"GEOMETRY: {geom}   |   measured peak lateral {peak:+.2f} m   |   {vtxt}")
    fig.suptitle(head, fontsize=11.5, weight="bold", y=0.985)
    if ev:
        fig.text(0.5, 0.505, f"Alpamayo CoT: \u201c{ev[:190]}\u201d", ha="center",
                 fontsize=9.5, style="italic", color="#333")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / f"{tag}_{cid[:8]}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return {"path": str(p), "clip": cid, "tag": tag, "alp": alp, "geom": geom,
            "peak_m": round(peak, 2), "cot": ev[:300],
            "verb": verb.group(0) if verb else None}


# --- pick representative cases ----------------------------------------------
buckets = {"EXTRACTOR-WRONG": [], "GENUINE-CONTRADICTION": [],
           "ONE-SIDED-GEOM": [], "ONE-SIDED-ALPAMAYO": [], "CONTROL-AGREE": []}
for r in rows:
    alp = ((r.get("alpamayo") or {}).get("lateral") or {}).get("alpamayo_side")
    geom = (r.get("a_tac") or {}).get("lat")
    if not alp or not geom:
        continue
    g = side_of(geom)
    ev = ((r.get("cot_tokens") or {}).get("evidence") or "")
    m = VERB.search(ev)
    v = m.group(2).lower() if m else None
    if alp != "straight" and g != "straight" and alp != g:
        buckets["EXTRACTOR-WRONG" if v == g else
                ("GENUINE-CONTRADICTION" if v == alp else "GENUINE-CONTRADICTION")].append(r)
    elif alp == "straight" and g != "straight":
        buckets["ONE-SIDED-GEOM"].append(r)
    elif alp != "straight" and g == "straight":
        buckets["ONE-SIDED-ALPAMAYO"].append(r)
    elif alp == "straight" and g == "straight":
        buckets["CONTROL-AGREE"].append(r)

made = []
PER = {"EXTRACTOR-WRONG": 3, "GENUINE-CONTRADICTION": 3, "ONE-SIDED-GEOM": 3,
       "ONE-SIDED-ALPAMAYO": 2, "CONTROL-AGREE": 1}
for tag, n in PER.items():
    pool = buckets[tag]
    print(f"{tag}: {len(pool)} available", flush=True)
    for r in pool[:n * 3]:
        if len([m for m in made if m["tag"] == tag]) >= n:
            break
        try:
            got = panel(r, tag)
            if got:
                made.append(got)
                print(f"  rendered {got['path']}", flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f"  skip {r['clip_id'][:8]}: {type(e).__name__}: {str(e)[:70]}",
                  flush=True)
json.dump(made, open(OUT / "panels.json", "w"), indent=1)
print(f"\n{len(made)} panels in {OUT}")
