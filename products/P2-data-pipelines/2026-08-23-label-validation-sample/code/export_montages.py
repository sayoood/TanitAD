"""Export per-clip frame montages so the frames can be INSPECTED, not inferred.

⚠️ WHY THIS STEP EXISTS. Every check so far compared a LABEL against GEOMETRY —
and both are derived from the same ego poses. That is an internal-consistency
check, not validation against the world. It cannot answer "is there a cyclist",
"is this an intersection", "is there a traffic light" — and it was exactly such
a question (a cyclist claimed on a clip containing none) that the PI answered by
LOOKING. The frames are the only independent witness available here.
"""
from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

LBL = Path("C:/Users/Admin/tanitad-wt/_s2build/labels_v3")
OUT = Path("C:/Users/Admin/tanitad-wt/_s2build/montages")
OUT.mkdir(parents=True, exist_ok=True)
EPC = glob.glob("C:/Users/Admin/tanitad-data/physicalai/_epcache/*/ep_*.pt")

KEY, HZ = 78, 10.0
OFFSETS = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
CELL = 232

WANT = ["5b4eef8f", "c84534a9", "5aef0388", "00d05901", "4d389996",
        "1a293863", "a8a381bf", "b0499b70", "12bb97af", "e084c7c3"]


def main() -> None:
    ci = json.loads((LBL / "clip_index.json").read_text(encoding="utf-8"))["clips"]
    leg = collections.defaultdict(list)
    for u, v in ci.items():
        leg[v["episode_id_legacy"]].append(u)
    amb = {k for k, v in leg.items() if len(v) > 1}

    by_leg = collections.defaultdict(list)
    cache = {}
    for p in sorted(EPC):
        d = torch.load(p, map_location="cpu", weights_only=False)
        cache[p] = d
        by_leg[int(d["episode_id"])].append(p)
    ambc = {k for k, v in by_leg.items() if len(v) > 1}

    made = []
    for p in sorted(EPC):
        d = cache[p]
        e = int(d["episode_id"])
        if e in amb or e in ambc or e not in leg:
            continue
        uid = leg[e][0]
        if uid[:8] not in WANT:
            continue
        poses = np.asarray(d["poses"], dtype=np.float64)
        T = poses.shape[0]
        sheet = Image.new("RGB", (CELL * len(OFFSETS), CELL + 22), (16, 20, 26))
        draw = ImageDraw.Draw(sheet)
        for k, off in enumerate(OFFSETS):
            i = max(0, min(T - 1, int(round(KEY + off * HZ))))
            arr = d["frames_u8"][i][6:9].permute(1, 2, 0).numpy()
            im = Image.fromarray(arr).resize((CELL, CELL), Image.LANCZOS)
            sheet.paste(im, (k * CELL, 22))
            v = float(poses[i, 3])
            lab = f"t{off:+.0f}s  v={v:.1f}"
            if off == 0:
                lab = f"KEY t0  v={v:.1f}"
                draw.rectangle([k * CELL, 22, k * CELL + CELL - 1, CELL + 21],
                               outline=(90, 155, 255), width=3)
            draw.text((k * CELL + 5, 5), lab, fill=(210, 225, 240))
        f = OUT / f"{uid[:8]}.png"
        sheet.save(f)
        made.append(str(f))
    print(f"wrote {len(made)} montages")
    for m in made:
        print(" ", m)


if __name__ == "__main__":
    main()
