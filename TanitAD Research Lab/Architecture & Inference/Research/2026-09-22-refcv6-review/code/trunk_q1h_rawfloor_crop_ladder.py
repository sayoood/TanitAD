"""Q1h — redo Q1f's intervention on the SENSITIVE instrument, and with a crop
deep enough to actually remove the strip.

Two things Q1f got wrong, both mine:
  1. **The trunk's pooled feature was the wrong probe.** Its RAW-PIXEL floor read
     **0.984** balanced accuracy against the trunk's **0.872** — by `CLAUDE.md`'s
     own probe rule, a representation that does not beat raw input has added
     nothing *for this target*, so the trunk arms cannot resolve the question.
     The raw floor is the instrument. It is also free: pooling, no forward.
  2. **The crop was too shallow.** `NOBOTTOM` removed 32 rows while the measured
     strip is **27-37** rows, so ~20 of the 74 positive clips kept 1-5 black
     rows. This runs a LADDER — 0, 32, 48, 64, 96 rows removed — so the point
     where decodability collapses (or fails to) is visible rather than assumed.

Matched control at every rung: the same number of rows removed from the **TOP**,
which keeps the strip. Plus the constant-only control (must read exactly 0.5),
a label permutation (must fall to chance), and n / d printed.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch
import torchvision.io as tvio

CACHE = "D:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl"
LADDER = (0, 32, 48, 64, 96)


def main() -> int:
    q1d = json.load(open(os.path.join(os.path.dirname(__file__), "..", "raw",
                                      "q1d_rig_black_strip.json")))
    lab = {r["clip_id"]: int(r["n_rows_fully_black"] > 0)
           for r in q1d["per_clip"]}
    nblack = {r["clip_id"]: int(r["n_rows_fully_black"])
              for r in q1d["per_clip"]}

    feats = {}
    y, cids = [], []
    for p in sorted(glob.glob(CACHE + "/*.v2ep.pt")):
        cid = os.path.basename(p).split(".")[0]
        if cid not in lab:
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        offs = torch.cat([torch.zeros(1, dtype=torch.int64),
                          torch.cumsum(d["jpeg_len"], 0)])
        dec = tvio.decode_png if d.get("codec") == "png" else tvio.decode_jpeg
        f = dec(d["jpeg_buf"][0:int(offs[1])],
                mode=tvio.ImageReadMode.RGB).float().div(255.0)
        for k in LADDER:
            bot = f[:, :416 - k, :] if k else f
            top = f[:, k:, :] if k else f
            for tag, img in ((f"CROP_BOTTOM_{k}", bot), (f"CROP_TOP_{k}", top)):
                if k == 0 and tag != "CROP_BOTTOM_0":
                    continue
                v = torch.nn.functional.adaptive_avg_pool2d(
                    img[None], (8, 16))[0].flatten().numpy()
                feats.setdefault(tag, []).append(v)
        y.append(lab[cid])
        cids.append(cid)

    y = np.asarray(y)
    X = {k: np.stack(v) for k, v in feats.items()}
    X["CONSTANT_ONLY"] = np.ones((len(y), 1), dtype=np.float32)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)

    def score(Xa, yy, n_rep=10):
        accs = []
        for r in range(n_rep):
            idx = rng.permutation(len(yy))
            cut = len(yy) // 2
            fit, sc = idx[:cut], idx[cut:]
            gs = GridSearchCV(
                make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=3000)),
                {"logisticregression__C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]},
                cv=StratifiedKFold(4, shuffle=True, random_state=r),
                scoring="balanced_accuracy")
            gs.fit(Xa[fit], yy[fit])
            accs.append(balanced_accuracy_score(yy[sc], gs.predict(Xa[sc])))
        a = np.asarray(accs)
        return {"mean": round(float(a.mean()), 4), "sd": round(float(a.std()), 4),
                "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}

    out = {
        "instrument": "RAW PIXELS, 8x16 average pool, NO network",
        "n": int(len(y)), "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "d": int(X["CROP_BOTTOM_0"].shape[1]),
        "chance_balanced_accuracy": 0.5,
        "max_black_rows_in_corpus": max(nblack.values()),
        "positives_with_more_than_32_black_rows": int(sum(
            1 for c in cids if nblack[c] > 32)),
        "arms": {k: score(v, y) for k, v in sorted(X.items())},
    }
    out["arms"]["CROP_BOTTOM_0_LABELS_PERMUTED"] = score(
        X["CROP_BOTTOM_0"], rng.permutation(y))
    out["reading_key"] = (
        "CROP_BOTTOM_k removes the strip; CROP_TOP_k removes the same k rows "
        "from the other end and KEEPS it. If CROP_BOTTOM falls with k while "
        "CROP_TOP does not, the strip is the carrier. If both stay high, the "
        "two groups differ in SCENE CONTENT and the strip is not needed to "
        "tell them apart.")
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
