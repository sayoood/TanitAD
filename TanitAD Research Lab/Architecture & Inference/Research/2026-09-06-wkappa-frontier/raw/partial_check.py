#!/usr/bin/env python3
"""SMOKE CHECK on the seed replicates while they run: compare the episodes already
dumped against seed 0's SAME episodes.

⛔ NOT a result and never quotable as one -- 5 windows of one episode. Its only job
is to catch a replicate that is behaving nothing like its seed-0 twin, an hour
before the record lands. The controls are what make it informative:
  * `ha0` is DETERMINISTIC (a straight line at the measured v0) and must be
    BIT-IDENTICAL between seeds -- if it is not, the corpora differ and the
    comparison is meaningless;
  * `g` (the recorded human) must likewise be bit-identical.
Only `cl` may move, because only the planner samples. ASCII only.
"""
import glob
import os
import sys

import numpy as np

P4 = r"C:\Users\Admin\refav1_margin\p4out"
OUT = r"C:\Users\Admin\wkfront\out"


def curv_mae(dumpdir, ep, arm, ref="g"):
    f = os.path.join(dumpdir, ep)
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    if arm not in z.files or ref not in z.files:
        return None

    def k(p):
        p = np.concatenate([np.zeros((p.shape[0], 1, 2)), p], axis=1)
        d = np.diff(p, axis=1)
        ds = np.linalg.norm(d, axis=-1)
        th = np.arctan2(d[..., 1], d[..., 0])
        dth = (np.diff(th, axis=1) + np.pi) % (2 * np.pi) - np.pi
        seg = 0.5 * (ds[:, :-1] + ds[:, 1:])
        m = seg >= 0.05
        return np.where(m, dth / np.maximum(seg, 1e-9), 0.0), m

    ka, ma = k(z[arm])
    kg, mg = k(z[ref])
    m = ma & mg
    n = m.sum()
    return float((np.abs(ka - kg) * m).sum() / max(n, 1)), z[arm]


def main():
    base = os.path.join(P4, "dump_wk7")
    eps = sorted(os.path.basename(p) for p in glob.glob(os.path.join(base, "ep*.npz")))
    for tag in sys.argv[1:]:
        d = os.path.join(OUT, "dump_%s" % tag)
        done = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "ep*.npz")))
        print("==== %s: %d/%d episodes dumped" % (tag, len(done), len(eps)))
        for ep in done:
            r0 = curv_mae(base, ep, "cl")
            r1 = curv_mae(d, ep, "cl")
            if r0 is None or r1 is None:
                continue
            # controls: the deterministic arms must be bit-identical
            same = []
            z0 = np.load(os.path.join(base, ep), allow_pickle=True)
            z1 = np.load(os.path.join(d, ep), allow_pickle=True)
            for key in ("ha0", "g"):
                if key in z0.files and key in z1.files:
                    same.append("%s %s" % (key, "IDENTICAL"
                                if np.array_equal(z0[key], z1[key]) else "*** DIFFERS ***"))
            print("   %s  seed0 curv %.6f  |  %s curv %.6f  |  d %+.6f   controls: %s"
                  % (ep, r0[0], tag, r1[0], r1[0] - r0[0], ", ".join(same)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
