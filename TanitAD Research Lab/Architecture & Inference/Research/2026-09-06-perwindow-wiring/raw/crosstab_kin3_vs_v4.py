"""How much resolution does kin3 destroy?

Cross-tabulates the live trainer's runtime kin3 3x3 labels against the v4
per-window 7x6 kinematic vocabulary, on the SAME windows, from the SAME poses.
This is the measurement behind the claim "a finer vocabulary on the same
windows": it says exactly which v4 tokens kin3 cannot express.

ASCII-only output (cp1252 console).
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter

import numpy as np
import torch

WT = "C:/Users/Admin/tanitad-wt"
sys.path.insert(0, WT + "/stack")
sys.path.insert(0, WT + "/stack/scripts")

import v4_labels as V4                                # noqa: E402
from tanitad.refs import refc_tactical as tac         # noqa: E402

CACHE = ("C:/Users/Admin/tanitad-data/physicalai/_epcache/"
         "physicalai-train-14231cd29c74")
N_EPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400

LAT4 = list(V4.LAT_TOKENS)
LON4 = list(V4.LON_TOKENS)
LAT3 = list(tac.LAT_CLASSES)
LON3 = list(tac.LON_CLASSES)
H = 20


def main() -> int:
    fs = sorted(glob.glob(CACHE + "/*.pt"))[:N_EPS]
    print("EPISODES (control, must be non-zero):", len(fs))
    lat4a, lon4a, lat3a, lon3a = [], [], [], []
    bad = 0
    for p in fs:
        try:
            poses = torch.load(p, map_location="cpu",
                               weights_only=False)["poses"]
        except Exception:
            bad += 1
            continue
        T = poses.shape[0]
        n = T - V4.WINDOW - V4.MAX_HORIZON
        if n <= 0:
            continue
        Ls = [V4.WINDOW - 1 + i for i in range(n)]
        # kin3 -- exactly the trainer's call: pose_last [B,4], future [B,20,4]
        pose_last = torch.stack([poses[L] for L in Ls])
        fut = torch.stack([poses[L + 1:L + 1 + H] for L in Ls])
        l3, n3 = tac.window_factored_labels(pose_last, fut)
        lat3a.append(l3.numpy())
        lon3a.append(n3.numpy())
        lat4a.append(np.array([V4.lat_target(poses, L) for L in Ls]))
        lon4a.append(np.array([V4.lon_target(poses, L) for L in Ls]))
    print("UNREADABLE:", bad, "(must be 0)")

    lat4 = np.concatenate(lat4a); lon4 = np.concatenate(lon4a)
    lat3 = np.concatenate(lat3a); lon3 = np.concatenate(lon3a)
    n = len(lat4)
    print("WINDOWS n :", n)
    print()

    def xtab(name, fine, ftoks, coarse, ctoks):
        print("=== %s : v4 (%d tokens, rows) x kin3 (%d classes, cols) ==="
              % (name, len(ftoks), len(ctoks)))
        hdr = "  %-14s" % "" + "".join("%12s" % c for c in ctoks) + "%10s" % "row n"
        print(hdr)
        collapsed = []
        for i, ft in enumerate(ftoks):
            m = fine == i
            row = [int(((coarse == j) & m).sum()) for j in range(len(ctoks))]
            tot = int(m.sum())
            print("  %-14s" % ft + "".join("%12d" % v for v in row)
                  + "%10d" % tot)
            if tot:
                # the coarse class this fine token is MOSTLY mapped into
                top = int(np.argmax(row))
                collapsed.append((ft, ctoks[top], row[top] / tot, tot))
        print("  %-14s" % "col n" + "".join(
            "%12d" % int((coarse == j).sum()) for j in range(len(ctoks))))
        print()
        print("  Where each v4 token LANDS in kin3 (dominant class, purity, n):")
        for ft, ct, pur, tot in collapsed:
            print("    %-14s -> %-12s  purity %.3f   n=%d" % (ft, ct, pur, tot))
        # the headline: distinct v4 tokens sharing one kin3 class
        by_c = {}
        for ft, ct, pur, tot in collapsed:
            by_c.setdefault(ct, []).append((ft, tot))
        print()
        print("  COLLISIONS (v4 tokens kin3 cannot tell apart):")
        n_lost = 0
        for ct, lst in by_c.items():
            if len(lst) > 1:
                tot = sum(t for _, t in lst)
                # windows lost = all but the largest member of the group
                n_lost += tot - max(t for _, t in lst)
                print("    kin3 %-12s <- %s   (n=%d)"
                      % (ct, ", ".join("%s(%d)" % (f, t) for f, t in lst), tot))
        print("  WINDOWS whose v4 identity kin3 cannot represent: %d / %d = %.4f"
              % (n_lost, n, n_lost / max(n, 1)))
        print()
        return {"collapsed": [(f, c, p, t) for f, c, p, t in collapsed],
                "n_lost": n_lost, "n": n}

    rlat = xtab("LATERAL", lat4, LAT4, lat3, LAT3)
    rlon = xtab("LONGITUDINAL", lon4, LON4, lon3, LON3)

    with open("C:/Users/Admin/tanitad-pwwire/crosstab_kin3_vs_v4.json", "w") as f:
        json.dump({"cache": CACHE, "n_windows": n, "lat": rlat, "lon": rlon},
                  f, indent=2)
    print("WROTE crosstab_kin3_vs_v4.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
