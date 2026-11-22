#!/usr/bin/env python3
"""label_offaxis_diag.py -- label diagnostic with no card (definitions logged in TIMES_NOTE.md before its run): where is the label's lead when it is far off the camera axis.
Tier T0 (recorded frames, label facts only; no head output, no fit and no rollout enters); not driving performance.
Frames: every frame of the 141 evaluation clips with has_lead true in the pilot's box files (card ADD99; the join's own lead rule). Bins of the absolute azimuth of the lead's box centre:
0-8, 8-16, 16-30, 30-45, above 45 degrees (upper edge included). Per bin: frames, clips, median range (m), median absolute lateral offset of the box centre in the ego frame (m), median forward
offset (m), median ego speed (m/s), median absolute path curvature ahead (1/m), share of frames with that curvature above 0.02 1/m (a radius below 50 m), share of frames with the box centre
beyond 60 degrees (the half field of view of the front-wide camera), median angular width of the box (degrees).
Usage (pod, CPU): label_offaxis_diag.py --box-dir DIR --eval-list FILE
"""
import argparse
import os

import numpy as np

BINS = ((0, 8), (8, 16), (16, 30), (30, 45), (45, None))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--box-dir", required=True)
    ap.add_argument("--eval-list", required=True)
    a = ap.parse_args(argv)
    ids = [l.split()[0] for l in open(a.eval_list, encoding="utf-8") if l.strip()]
    cols = {k: [] for k in ("az", "rng", "lat", "fwd", "v", "kap", "angw", "clip")}
    n_frames = 0
    for i, c in enumerate(ids):
        z = np.load(os.path.join(a.box_dir, c + ".npz"))
        h = z["has_lead"].astype(bool) & np.isfinite(z["ctr_az"])
        n_frames += len(h)
        ce = np.asarray(z["ctr_ego"], np.float64)
        cols["az"].append(np.abs(z["ctr_az"][h]))
        cols["rng"].append(z["rng"][h])
        cols["lat"].append(np.abs(ce[h, 1]))
        cols["fwd"].append(ce[h, 0])
        cols["v"].append(z["v_ego"][h])
        cols["kap"].append(np.abs(z["kappa_ahead"][h]))
        cols["angw"].append(z["ang_w"][h])
        cols["clip"].append(np.full(int(h.sum()), i))
    C = {k: np.concatenate(v) for k, v in cols.items()}
    n = len(C["az"])
    print("LABEL-OFFAXIS clips %d frames %d with a lead %d (share %.3f)" % (len(ids), n_frames, n, n / max(n_frames, 1)))
    med = lambda x: float(np.nanmedian(x)) if len(x) else float("nan")
    for lo, hi in BINS:
        m = (C["az"] > lo) if lo else np.ones(n, bool)
        if hi:
            m &= C["az"] <= hi
        k = int(m.sum())
        name = ("%d-%d deg" % (lo, hi)) if hi else "above %d deg" % lo
        if not k:
            print("LABEL-OFFAXIS bin %-13s | frames 0" % name)
            continue
        print("LABEL-OFFAXIS bin %-13s | frames %5d (share %.3f) | clips %3d | range %5.1f m | lateral %5.1f m | forward %5.1f m | ego speed %5.1f m/s | curvature %.4f 1/m | share curvature above 0.02: %.2f | share beyond 60 deg: %.2f | box width %4.1f deg"
              % (name, k, k / n, len(np.unique(C["clip"][m])), med(C["rng"][m]), med(C["lat"][m]), med(C["fwd"][m]), med(C["v"][m]), med(C["kap"][m]),
                 float((C["kap"][m] > 0.02).mean()), float((C["az"][m] > 60).mean()), med(C["angw"][m])))


if __name__ == "__main__":
    main()
