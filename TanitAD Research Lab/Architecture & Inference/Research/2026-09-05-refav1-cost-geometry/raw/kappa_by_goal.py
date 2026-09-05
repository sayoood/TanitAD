#!/usr/bin/env python3
"""Realised planner curvature, split by the DECODED GOAL TOKEN and by GROUND
TRUTH, from a refav1_arm `decisions` sidecar. Zero GPU.

This is the post-hoc CONTROL for the `ccosh` hold branch: on windows whose
decoded lateral goal is LANE_KEEP the goal IS the hold field, so a planner with
a defined cost there must emit ~0 curvature. The SAME-BREATH control that must
read NON-ZERO is the TURN-decoded column: if both columns read 0.0 the arm has
simply stopped steering and the LANE_KEEP zero is not evidence of a repair.

ASCII output only (cp1252 dev box).
"""
import glob, json, os, sys
import numpy as np

# the REAL v7.0 lateral vocabulary, imported rather than transcribed: a
# hand-written list had TURN_L/TURN_R at the wrong indices on the first pass.
from tanitad.models.vocab_v7 import (TACTICAL_LAT_ACTIONS_V7,
                                     TACTICAL_LON_ACTIONS_V7)
LAT_V70 = list(TACTICAL_LAT_ACTIONS_V7)
LON_V70 = list(TACTICAL_LON_ACTIONS_V7)


def load(dumpdir):
    kap, lat, lon, src, gt = [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(dumpdir, "decisions", "*.npz"))):
        z = np.load(f, allow_pickle=True)
        kap.append(z["cl_controls"][..., 1])
        lat.append(z["goal_lat_cl"])
        lon.append(z["goal_lon_cl"])
        src.append(z["plan_source_cl"])
        # ha0_ext holds the MEASURED (a0, kappa0): its curvature channel is the
        # measured kappa at t0 and is the only per-window curvature reference
        # the sidecar carries.
        gt.append(z["ha0_ext_controls"][:, 0, 1] if "ha0_ext_controls" in z.files
                  else np.full(z["goal_lat_cl"].shape, np.nan))
    if not kap:
        return None
    return (np.concatenate(kap, 0), np.concatenate(lat), np.concatenate(lon),
            np.concatenate(src), np.concatenate(gt))


def report(dumpdir, label, turn_thresh=4e-2):
    r = load(dumpdir)
    if r is None:
        print("NO DECISIONS in %s" % dumpdir)
        return None
    kap, lat, lon, src, k0 = r
    absk = np.abs(kap).max(-1)
    mk2 = (kap ** 2).mean(-1)
    n = len(lat)
    print("==== %s   n_windows=%d" % (label, n))
    print("  by DECODED LATERAL GOAL TOKEN")
    print("    %-14s %5s %10s %10s %10s %10s" %
          ("token", "n", "med|k|max", "frac k!=0", "frac@cap", "mean k^2"))
    for t in sorted(set(int(x) for x in lat)):
        m = lat == t
        name = LAT_V70[t] if 0 <= t < len(LAT_V70) else str(t)
        print("    %-14s %5d %10.5f %10.4f %10.4f %10.6f" % (
            name, int(m.sum()), float(np.median(absk[m])),
            float((absk[m] > 1e-9).mean()), float((absk[m] >= 0.1999).mean()),
            float(mk2[m].mean())))
    print("  by GROUND TRUTH (|kappa0| from ha0_ext, threshold %.3g)" % turn_thresh)
    if np.all(np.isnan(k0)):
        print("    ha0_ext_controls absent from the sidecar - GT split UNAVAILABLE")
    else:
        for name, m in (("GT-straight", np.abs(k0) <= turn_thresh),
                        ("GT-turn", np.abs(k0) > turn_thresh)):
            if not m.any():
                print("    %-14s n=0" % name)
                continue
            same = np.sign(kap[m].mean(-1)) == np.sign(k0[m])
            print("    %-14s %5d  med|k|max %.5f  frac k!=0 %.4f  frac@cap %.4f"
                  "  dir_correct %.4f" % (
                      name, int(m.sum()), float(np.median(absk[m])),
                      float((absk[m] > 1e-9).mean()),
                      float((absk[m] >= 0.1999).mean()), float(same.mean())))
    print("  by DECODED LONGITUDINAL GOAL TOKEN")
    for t in sorted(set(int(x) for x in lon)):
        m = lon == t
        name = LON_V70[t] if 0 <= t < len(LON_V70) else str(t)
        print("    %-24s %5d" % (name, int(m.sum())))
    print("  plan_source: cem %.3f  baseline %.3f" % (
        float((src == 0).mean()), float((src != 0).mean())))
    return dict(n=n, med_absk=float(np.median(absk)),
                frac_nonzero=float((absk > 1e-9).mean()),
                frac_cap=float((absk >= 0.1999).mean()),
                mean_k2=float(mk2.mean()))


if __name__ == "__main__":
    out = {}
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        out[args[i + 1]] = report(args[i], args[i + 1])
        print()
    if os.environ.get("KBG_JSON"):
        with open(os.environ["KBG_JSON"], "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
