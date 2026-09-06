#!/usr/bin/env python3
"""IS THE FRICTION-CIRCLE ZERO EARNED BY THE COST, OR STRUCTURAL IN THE VOCABULARY?

`GOAL_KAPPA_TURN = 0.08` (the shipped L1-0.08 lateral goal vocabulary) is the
planner's ONLY curvature-carrying candidate. Lateral acceleration is
`a_lat = v^2 * kappa`, so an arm that never emits |kappa| > 0.08 cannot leave the
`mu * g` circle below a BREAK-EVEN SPEED that depends on nothing but mu and the
vocabulary. This prints that speed, and the speed the panel actually reaches.

⛔ CONTROLS, both of which must read their known values:
  * the recorded human `g` must exceed the vocabulary cap (it is a real vehicle,
    not a token) -- if it does not, the kappa recovery here is wrong;
  * `ha0` is a perfectly straight line and must read max|kappa| == 0 EXACTLY.
ASCII only.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Admin\tanitad-ctg\stack")
from tanitad.refs import feasible_decode as FD                    # noqa: E402

DT = 0.2
G = 9.81
KAPPA_VOCAB = 0.08          # refa_v1.GOAL_KAPPA_TURN, the shipped L1-0.08 level
P4 = r"C:\Users\Admin\refav1_margin\p4out"


def path_kappa_v(dumpdir, key, vmin=2.0):
    ks, vs = [], []
    for f in sorted(glob.glob(dumpdir + "/ep*.npz")):
        z = np.load(f, allow_pickle=True)
        if key not in z.files:
            continue
        a = z[key]
        if "v0" in z.files and vmin > 0:
            a = a[z["v0"] >= vmin]
        if not len(a):
            continue
        p = np.concatenate([np.zeros((a.shape[0], 1, 2)), a], axis=1)
        d = np.diff(p, axis=1)
        ds = np.linalg.norm(d, axis=-1)
        th = np.arctan2(d[..., 1], d[..., 0])
        dth = (np.diff(th, axis=1) + np.pi) % (2 * np.pi) - np.pi
        seg = 0.5 * (ds[:, :-1] + ds[:, 1:])
        m = seg > 1e-6
        k = np.where(m, dth / np.maximum(seg, 1e-9), 0.0)
        v = seg / DT
        ks.append(k[m])
        vs.append(v[m])
    if not ks:
        return None, None
    return np.concatenate(ks), np.concatenate(vs)


def main():
    mu = FD.MU_KAMM
    v_break = float(np.sqrt(mu * G / KAPPA_VOCAB))
    print("mu = %.3f (feasible_decode.MU_KAMM), g = %.2f, vocabulary kappa = %.4f"
          % (mu, G, KAPPA_VOCAB))
    print("BREAK-EVEN SPEED for the vocabulary's own turn curvature:")
    print("  a_lat = v^2 * kappa = mu*g  ->  v = sqrt(%.3f*%.2f/%.4f) = %.3f m/s"
          % (mu, G, KAPPA_VOCAB, v_break))
    print("  => below %.2f m/s, an arm capped at |kappa| <= %.4f is INSIDE the"
          % (v_break, KAPPA_VOCAB))
    print("     friction circle BY CONSTRUCTION, whatever its cost weights are.")
    print()
    print("%-14s %8s %10s %10s %10s %10s" %
          ("arm", "n_steps", "max|kappa|", "max v", "max a_lat", "max g"))
    for tag in sys.argv[1:]:
        d = os.path.join(P4, "dump_%s" % tag)
        if not os.path.isdir(d):
            d = os.path.join(r"C:\Users\Admin\wkfront\out", "dump_%s" % tag)
        for key in ("cl", "g", "ha0"):
            k, v = path_kappa_v(d, key)
            if k is None:
                continue
            alat = v * v * np.abs(k)
            note = ""
            if key == "g":
                note = "  <- CONTROL: a real vehicle, must EXCEED the vocab cap"
            if key == "ha0":
                note = "  <- CONTROL: straight line, must read max|kappa| 0.0000"
            # ⭐ THE DISCRIMINATOR between "structural in the vocabulary" and
            # "co-ordinated with speed": if the arm emitted its turn curvature at
            # the panel's top speed it WOULD violate, so the zero is only
            # structural if the panel never reaches the break-even speed. It
            # does (max v > 9.27), so what matters is the speed at which large
            # curvature is actually emitted.
            big = np.abs(k) >= 0.05
            v_big = float(v[big].max()) if big.any() else float("nan")
            print("%-14s %8d %10.4f %10.3f %10.3f %10.3f   max v where|k|>=.05 %7.3f%s"
                  % ("%s:%s" % (tag, key), len(k), np.abs(k).max(), v.max(),
                     alat.max(), alat.max() / G, v_big, note))
        print()


if __name__ == "__main__":
    main()
