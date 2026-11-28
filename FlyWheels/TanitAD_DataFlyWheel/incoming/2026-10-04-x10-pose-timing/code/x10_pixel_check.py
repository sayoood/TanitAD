"""X10 pixel check -- is the image stored at cache row k really camera frame searchsorted(t_frames, t_query[k])?

Everything in the timing correction rests on that premise; until now it was read from the builder (v2_compressed.py:121-123)
and from D3, never compared on pixels.  This decodes the cached row, rebuilds the canonical image of the candidate camera
frames {fi-2 .. fi+2} from the raw mp4 with the builder's OWN decode function, and reports which candidate the cached row
equals.  Rows are chosen where `delta` is large (> 20 ms), so that a "nearest frame" cache (fi-1) and the builder's
"next frame" (fi) are distinguishable -- the whole point of the test.

    PYTHONPATH=<clean tree>/stack;<tree>/stack/scripts;<tree>/taniteval  python x10_pixel_check.py --out ../raw/x10_pixel_check.json

CPU only; reads 2 eval clips' payloads (~100 MB each) and 2 raw mp4s.  Ids are sha12.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch
import torchvision.io as tvio

import v2_compressed as V
from tanitad.data import pose_sync as PS
from tanitad.data.calib import CanonicalFrame

CACHE = "D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139"
CAM = "C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clips", type=int, default=2)
    ap.add_argument("--rows-per-clip", type=int, default=5)
    a = ap.parse_args()
    man = torch.load(f"{CACHE}/_v2manifest.pt", map_location="cpu", weights_only=False)
    rng = np.random.default_rng(20261004)
    pick = sorted(rng.choice(len(man["clip_id"]), size=a.n_clips, replace=False).tolist())
    res = {"clips": []}
    for i in pick:
        cid = man["clip_id"][i]
        d = torch.load(f"{CACHE}/{cid}.v2ep.pt", map_location="cpu", weights_only=False)
        codec = d.get("codec")
        frame = CanonicalFrame.from_dict(d["frame"])
        pm = d.get("projection_mode", "ftheta_crop")
        ts = pd.read_parquet(f"{CAM}/{cid}.timestamps.parquet")
        tf = ts[next(c for c in ts.columns if "time" in c.lower())].to_numpy(np.float64)
        tq, fi, unit = PS.resample_grid(tf)
        delta_s, dt, n = PS.grid_delta(tf)
        lens = d["jpeg_len"]
        assert len(lens) == n, (len(lens), n)
        offs = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(lens, 0)])
        dec = tvio.decode_png if codec == "png" else tvio.decode_jpeg
        rows = [k for k in range(5, n - 5) if delta_s[k] * 1e3 > 20.0][:: max(1, (n // a.rows_per_clip))][: a.rows_per_clip]
        need = sorted({int(fi[k]) + o for k in rows for o in (-2, -1, 0, 1, 2)})
        mp4 = f"{CAM}/{cid}.mp4"
        cand = V._decode_cropped_selected(mp4, 256, np.array(need), frame, pm)        # the builder's own decode + remap
        cmap = {f: cand[j] for j, f in enumerate(need)}
        per = []
        for k in rows:
            cached = dec(d["jpeg_buf"][int(offs[k]):int(offs[k + 1])], mode=tvio.ImageReadMode.RGB)
            errs = {}
            for o in (-2, -1, 0, 1, 2):
                c = cmap[int(fi[k]) + o]
                errs[str(o)] = float((cached.float() - c.float()).abs().mean())
            best = min(errs, key=errs.get)
            per.append({"row": int(k), "delta_ms": round(float(delta_s[k] * 1e3), 3), "frame_idx": int(fi[k]),
                        "mean_abs_pixel_err_by_offset_from_frame_idx": errs, "best_offset": int(best),
                        "exact_equal_at_best": bool(torch.equal(cached, cmap[int(fi[k]) + int(best)]))})
            print(cid[:0], per[-1], flush=True)
        res["clips"].append({"sha12": hashlib.sha256(cid.encode()).hexdigest()[:12], "codec": codec, "projection_mode": pm,
                             "n_rows_cache": int(n), "rows": per})
    allb = [r["best_offset"] for c in res["clips"] for r in c["rows"]]
    res["summary"] = {"n_rows": len(allb), "best_offset_counts": {str(o): allb.count(o) for o in (-2, -1, 0, 1, 2)},
                      "premise_holds_if_all_best_offsets_are_0": bool(all(b == 0 for b in allb))}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["summary"]))


if __name__ == "__main__":
    main()
