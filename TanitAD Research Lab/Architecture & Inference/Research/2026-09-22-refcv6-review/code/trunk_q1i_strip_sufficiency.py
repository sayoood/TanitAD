"""Q1i — the SUFFICIENCY control the crop ladder cannot give.

The ladder (q1h) shows decodability falls when the BOTTOM rows are removed and
does not when the same number of TOP rows are removed. ⚠️ But the two crops are
matched on SIZE, not on INFORMATION: the bottom of a forward camera is near road,
the top is sky, so a bottom/top asymmetry is partly expected on content grounds
alone. The ladder therefore shows the strip is *involved*, not that it is
*sufficient*.

⭐ THE CLEAN EXPERIMENT IS A FABRICATED LABEL ON THE CLEAN HALF. Take only the
**65 clips with NO strip** — one homogeneous group — and give a random half of
them an artificial 31-row black bottom strip (the measured mean height). The
label is now, by construction, *"I painted a strip on this one"* and nothing
else: the two sub-groups are drawn from the same population, the same campaigns,
the same scenes.

If a raw-pixel probe recovers that label, a black strip **alone** is enough to
make a group linearly identifiable at zero capacity. That is the whole of F1's
concern, decided without any rig attribution.

Controls: constant-only (must read exactly 0.5), a label permutation (must fall
to chance), and the SAME probe run on the UNPAINTED clean half with the same
random labels (must also read chance — if it does not, the split itself leaks).
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
STRIP_ROWS = 31          # the measured mean strip height (q1d)


def main() -> int:
    q1d = json.load(open(os.path.join(os.path.dirname(__file__), "..", "raw",
                                      "q1d_rig_black_strip.json")))
    clean = [r["clip_id"] for r in q1d["per_clip"]
             if r["n_rows_fully_black"] == 0]

    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, size=len(clean))
    imgs = []
    for cid in clean:
        p = os.path.join(CACHE, cid + ".v2ep.pt")
        d = torch.load(p, map_location="cpu", weights_only=False)
        offs = torch.cat([torch.zeros(1, dtype=torch.int64),
                          torch.cumsum(d["jpeg_len"], 0)])
        dec = tvio.decode_png if d.get("codec") == "png" else tvio.decode_jpeg
        imgs.append(dec(d["jpeg_buf"][0:int(offs[1])],
                        mode=tvio.ImageReadMode.RGB).float().div(255.0))

    def pool(x):
        return torch.nn.functional.adaptive_avg_pool2d(
            x[None], (8, 16))[0].flatten().numpy()

    painted, unpainted = [], []
    for i, f in enumerate(imgs):
        unpainted.append(pool(f))
        g = f.clone()
        if y[i] == 1:
            g[:, -STRIP_ROWS:, :] = 0.0          # ⛔ the ONLY difference
        painted.append(pool(g))
    X = {"PAINTED_STRIP": np.stack(painted),
         "CONTROL_UNPAINTED_same_labels": np.stack(unpainted),
         "CONSTANT_ONLY": np.ones((len(y), 1), dtype=np.float32)}

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    r2 = np.random.default_rng(0)

    def score(Xa, yy, n_rep=10):
        a = []
        for r in range(n_rep):
            idx = r2.permutation(len(yy))
            cut = len(yy) // 2
            fit, sc = idx[:cut], idx[cut:]
            gs = GridSearchCV(
                make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=5000)),
                {"logisticregression__C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]},
                cv=StratifiedKFold(4, shuffle=True, random_state=r),
                scoring="balanced_accuracy")
            gs.fit(Xa[fit], yy[fit])
            a.append(balanced_accuracy_score(yy[sc], gs.predict(Xa[sc])))
        a = np.asarray(a)
        return {"mean": round(float(a.mean()), 4), "sd": round(float(a.std()), 4),
                "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}

    out = {
        "population": "the 65 clips with NO measured strip — ONE homogeneous group",
        "instrument": "RAW PIXELS, 8x16 average pool, NO network",
        "n": int(len(y)), "n_painted": int(y.sum()),
        "n_unpainted": int((1 - y).sum()), "d": 384,
        "strip_rows_painted": STRIP_ROWS,
        "chance_balanced_accuracy": 0.5,
        "arms": {k: score(v, y) for k, v in X.items()},
    }
    out["arms"]["PAINTED_LABELS_PERMUTED"] = score(
        X["PAINTED_STRIP"], r2.permutation(y))
    out["reading_key"] = (
        "PAINTED_STRIP high and CONTROL_UNPAINTED at chance => a black strip "
        "ALONE makes a group linearly identifiable from raw pixels. Both at "
        "chance => the strip is not by itself a usable feature at this n.")
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
