#!/usr/bin/env python3
"""Per-window ADE split by GT stratum, zero GPU, straight from the dumps.

Tells us WHERE refav1 has to win. Strata use |kappa0| from `ha0_ext_controls`
(the measured curvature at t0) at the programme's measured crossover 4e-2.
⛔ CONTROL that must read EXACTLY 0.0: `g` against itself. ASCII only.
"""
import glob, os, sys
import numpy as np

TH = 4e-2


def load(dumpdir):
    out = {}
    for f in sorted(glob.glob(os.path.join(dumpdir, "ep*.npz"))):
        z = np.load(f, allow_pickle=True)
        for k in ("g", "cl", "ha", "ha0", "ha0_ext", "ol", "v0"):
            if k in z.files:
                out.setdefault(k, []).append(z[k])
    d = {k: np.concatenate(v, 0) for k, v in out.items()}
    k0 = []
    for f in sorted(glob.glob(os.path.join(dumpdir, "decisions", "*.npz"))):
        z = np.load(f, allow_pickle=True)
        k0.append(z["ha0_ext_controls"][:, 0, 1])
    d["k0"] = np.concatenate(k0, 0) if k0 else None
    return d


def ade(p, g):
    return np.linalg.norm(p - g, axis=-1).mean(-1)


def main(dumpdir, label, vmin=0.0):
    d = load(dumpdir)
    if d.get("k0") is None:
        print("%s: no decisions sidecar" % label); return
    g, k0, v0 = d["g"], d["k0"], d["v0"]
    keep = v0 >= vmin
    print("==== %s   n=%d (v0 >= %.1f: %d)" % (label, len(g), vmin, int(keep.sum())))
    print("    CONTROL g-vs-g ADE = %.6f (must be exactly 0)" % float(ade(g, g).mean()))
    strata = (("ALL", np.ones(len(g), bool)),
              ("GT-straight", np.abs(k0) <= TH),
              ("GT-turn", np.abs(k0) > TH))
    print("    %-12s %5s %9s %9s %9s %9s %9s" %
          ("stratum", "n", "cl", "ha", "ha0", "ha0_ext", "ol"))
    for name, m in strata:
        m = m & keep
        if not m.any():
            print("    %-12s n=0" % name); continue
        row = [float(ade(d[a][m], g[m]).mean()) if a in d else float("nan")
               for a in ("cl", "ha", "ha0", "ha0_ext", "ol")]
        print("    %-12s %5d %9.4f %9.4f %9.4f %9.4f %9.4f" % (name, int(m.sum()), *row))
    for name, m in strata[1:]:
        m = m & keep
        if not m.any():
            continue
        dlt = ade(d["cl"][m], g[m]) - ade(d["ha0_ext"][m], g[m])
        print("    cl - ha0_ext on %-12s mean %+8.4f  median %+8.4f  frac cl better %.4f"
              % (name, float(dlt.mean()), float(np.median(dlt)), float((dlt < 0).mean())))


if __name__ == "__main__":
    vmin = float(os.environ.get("ADE_VMIN", "0"))
    for i in range(0, len(sys.argv) - 1, 2):
        main(sys.argv[1 + i], sys.argv[2 + i], vmin)
        print()
