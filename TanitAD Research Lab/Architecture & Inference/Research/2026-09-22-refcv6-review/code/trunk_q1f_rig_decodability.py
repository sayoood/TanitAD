"""Q1f — THE DISCRIMINATING EXPERIMENT for F1: is the rig IDENTIFIABLE from the
trunk's planner feature, and is the black STRIP the carrier?

q1e established the channel (the strip moves `pooled` by 17.9 % of the typical
between-clip distance) but NOT that anything can use it. A probe on the full
frame alone would prove nothing either: the rig correlates with location, time
of day and camera, so rig-decodability is expected whatever the strip does.

⭐ THE INTERVENTION IS A CROP, WITH A MATCHED CONTROL CROP:
  * **FULL**      416 rows — the frame as trained on;
  * **NOBOTTOM**  rows [0:384] — the strip is REMOVED (384 % 32 == 0);
  * **NOTOP**     rows [32:416] — the same 32 rows removed, from the OTHER end,
                  so the strip is RETAINED. Matched on size and on "32 rows of
                  context were lost".
If rig-decodability survives NOTOP and collapses on NOBOTTOM, the strip is the
carrier. If all three read the same, the rig is legible from the scene anyway
and the strip adds nothing — which is also a real answer.

⛔ The probe rules from `CLAUDE.md` are followed literally:
  * a **CONSTANT-ONLY control** that must read the no-information value
    (majority-class balanced accuracy = 0.5, exactly);
  * a **RAW-PIXEL floor** (downsampled pixels) — a representation that does not
    beat raw input has added nothing;
  * **n and d printed**; n = 139 << d = 512, so this is underpowered by
    construction and the regulariser is fitted on the FIT split ONLY, never on
    the scored split;
  * a **label-permutation** arm, which must fall to chance.

Labels come from `raw/q1d_rig_black_strip.json` (n_rows_fully_black > 0). ⚠️ On
FULL and NOTOP that label is partly self-fulfilling — a probe could read the
strip it is labelled by. That is the point: NOBOTTOM is the arm that cannot.
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


def main() -> int:
    from tanitad.models.timm_trunk import TimmResNetTrunk, TimmTrunkConfig

    q1d = json.load(open(os.path.join(os.path.dirname(__file__), "..", "raw",
                                      "q1d_rig_black_strip.json")))
    lab = {r["clip_id"]: int(r["n_rows_fully_black"] > 0)
           for r in q1d["per_clip"]}

    t = TimmResNetTrunk(TimmTrunkConfig(
        model_name="resnet34.a1_in1k", frames=3, mode="shared",
        image_hw=(416, 1024), pretrained=True)).eval()

    ARMS = {"FULL": slice(0, 416), "NOBOTTOM": slice(0, 384),
            "NOTOP": slice(32, 416)}
    feats = {k: [] for k in ARMS}
    pix = []
    y = []
    cids = []
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
        for name, sl in ARMS.items():
            x3 = f[:, sl, :]
            x = torch.cat([x3, x3, x3], dim=0)[None]
            with torch.no_grad():
                _s16, _s32, pooled = t.forward_features(x)
            feats[name].append(pooled[0].numpy())
        # RAW-PIXEL floor: the FULL frame, average-pooled to 8x16x3 = 384 dims
        pix.append(torch.nn.functional.adaptive_avg_pool2d(
            f[None], (8, 16))[0].flatten().numpy())
        y.append(lab[cid])
        cids.append(cid)

    y = np.asarray(y)
    X = {k: np.stack(v) for k, v in feats.items()}
    X["RAWPIXEL_FLOOR"] = np.stack(pix)
    X["CONSTANT_ONLY"] = np.ones((len(y), 1), dtype=np.float32)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)

    def score(Xa, yy, n_rep=8):
        accs = []
        for r in range(n_rep):
            idx = rng.permutation(len(yy))
            cut = len(yy) // 2
            fit, sc = idx[:cut], idx[cut:]
            # ⛔ C is selected on the FIT split only, by inner CV.
            gs = GridSearchCV(
                make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000)),
                {"logisticregression__C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]},
                cv=StratifiedKFold(4, shuffle=True, random_state=r),
                scoring="balanced_accuracy")
            try:
                gs.fit(Xa[fit], yy[fit])
                accs.append(balanced_accuracy_score(
                    yy[sc], gs.predict(Xa[sc])))
            except ValueError:
                accs.append(float("nan"))
        a = np.asarray(accs, dtype=float)
        return {"mean": round(float(np.nanmean(a)), 4),
                "sd": round(float(np.nanstd(a)), 4),
                "min": round(float(np.nanmin(a)), 4),
                "max": round(float(np.nanmax(a)), 4), "n_rep": n_rep}

    out = {"n": int(len(y)), "n_positive": int(y.sum()),
           "n_negative": int((1 - y).sum()),
           "majority_fraction": round(float(max(y.mean(), 1 - y.mean())), 4),
           "chance_balanced_accuracy": 0.5,
           "d_per_arm": {k: int(v.shape[1]) for k, v in X.items()},
           "arms": {}}
    for k, Xa in X.items():
        out["arms"][k] = score(Xa, y)
    # label-permutation control on the strongest arm
    yp = rng.permutation(y)
    out["arms"]["FULL_LABELS_PERMUTED"] = score(X["FULL"], yp)
    out["reading_key"] = (
        "NOBOTTOM removes the strip; NOTOP removes the same 32 rows from the "
        "other end and KEEPS it. FULL ~ NOTOP >> NOBOTTOM => the strip is the "
        "carrier. All three equal => the rig is legible from scene content and "
        "the strip adds nothing measurable at n=139.")
    out["clip_ids"] = cids
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
