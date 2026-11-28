#!/usr/bin/env python3
"""WHERE DOES THE PENALTY'S ADE GAIN COME FROM? Attribution by DECODED GOAL TOKEN.

⭐⭐ WHY THIS IS THE DECIDING 0-GPU MEASUREMENT. `ccos_argmax -> wk15` buys
-0.4338 m ADE and kills `turn_left`. A3 keeps the penalty on LANE_KEEP-goal
windows and removes it on TURN-goal windows, so **A3 can retain exactly the part
of that -0.4338 that is earned on LANE_KEEP windows, and gives back the part
earned on TURN windows.** This file measures that split BEFORE the A3 arms land,
which turns A3's committed prediction from a guess into an arithmetic bound.

⛔ THE COMPARISON IS PAIRED, WINDOW BY WINDOW, on the shared `ws` grid -- never a
difference of two pooled means. The GRID CONTROL is that `ha`/`ha0`/`ha0_ext`/`ol`
are bit-identical between the two arms: they do not depend on the cost lever, so
any difference means the arms are not on one grid and nothing here is admissible.

⛔ AND THE TRIVIAL-ARM CONTROL: `ol` is the RECORDED FUTURE (T0). It is printed so
the reader can see the floor the planner is being scored against, and it is NEVER
a driving number.

ASCII output only.
"""
import glob
import json
import os
import sys

import numpy as np

from tanitad.models.vocab_v7 import TACTICAL_LAT_ACTIONS_V7

LAT = list(TACTICAL_LAT_ACTIONS_V7)


def load(dumpdir):
    ep, dec = {}, {}
    for f in sorted(glob.glob(os.path.join(dumpdir, "ep*.npz"))):
        z = np.load(f, allow_pickle=True)
        eid = int(z["eid"][0])
        for w, i in zip(z["ws"].tolist(), range(len(z["ws"]))):
            ep[(eid, int(w))] = {k: z[k][i] for k in ("g", "cl", "ha", "ha0",
                                                      "ha0_ext", "ol", "v0")}
    for f in sorted(glob.glob(os.path.join(dumpdir, "decisions", "*.npz"))):
        z = np.load(f, allow_pickle=True)
        eid = int(os.path.basename(f)[2:5])
        for w, i in zip(z["ws"].tolist(), range(len(z["ws"]))):
            dec[(eid, int(w))] = int(z["goal_lat_cl"][i])
    return ep, dec


def ade(a, b):
    """mean L2 over the horizon between two [H,2] displacement paths."""
    return float(np.sqrt(((a - b) ** 2).sum(-1)).mean())


def main(argv):
    root = argv[1]
    A, B = argv[2], argv[3]
    epA, decA = load(os.path.join(root, "dump_" + A))
    epB, decB = load(os.path.join(root, "dump_" + B))
    keys = sorted(set(epA) & set(epB))
    print("=" * 104)
    print("PAIRED ADE ATTRIBUTION BY DECODED GOAL TOKEN:  %s  ->  %s" % (A, B))
    print("=" * 104)
    print("  shared windows: %d  (A had %d, B had %d)" % (len(keys), len(epA), len(epB)))

    # ⛔ GRID CONTROL -- these arms do not depend on the cost lever.
    bad = []
    for k in keys:
        for arm in ("ha", "ha0", "ha0_ext", "ol", "g"):
            if not np.array_equal(epA[k][arm], epB[k][arm]):
                bad.append((k, arm))
    print("  GRID CONTROL (ha/ha0/ha0_ext/ol/g must be bit-identical): %s"
          % ("PASSED" if not bad else "*** FAILED on %d (window, arm) pairs -- VOID ***"
             % len(bad)))
    if bad:
        return 1
    if any(decA[k] != decB[k] for k in keys):
        n = sum(decA[k] != decB[k] for k in keys)
        print("  ⚠️ decoded goal differs on %d/%d windows -- the goal head does not"
              % (n, len(keys)))
        print("     depend on the cost, so this must be 0. Attribution below is VOID.")
        return 1
    print("  decoded goal identical on all %d windows  [the split is well defined]" % len(keys))
    print()

    dA = np.array([ade(epA[k]["cl"], epA[k]["g"]) for k in keys])
    dB = np.array([ade(epB[k]["cl"], epB[k]["g"]) for k in keys])
    dOL = np.array([ade(epA[k]["ol"], epA[k]["g"]) for k in keys])
    dH0 = np.array([ade(epA[k]["ha0"], epA[k]["g"]) for k in keys])
    tok = np.array([decA[k] for k in keys])
    delta = dB - dA

    print("  %-16s %5s %10s %10s %11s %11s %9s"
          % ("goal token", "n", "ADE(A)", "ADE(B)", "mean delta", "SHARE of", "ha0"))
    print("  %-16s %5s %10s %10s %11s %11s %9s"
          % ("", "", A[:10], B[:10], "B-A", "total delta", "floor"))
    print("  " + "-" * 82)
    tot = float(delta.mean())
    for t in sorted(set(tok.tolist())):
        m = tok == t
        nm = LAT[t] if 0 <= t < len(LAT) else str(t)
        share = float(delta[m].sum()) / float(delta.sum()) if delta.sum() != 0 else float("nan")
        print("  %-16s %5d %10.4f %10.4f %11.4f %10.1f%% %9.4f"
              % (nm, int(m.sum()), dA[m].mean(), dB[m].mean(), delta[m].mean(),
                 100.0 * share, dH0[m].mean()))
    print("  " + "-" * 82)
    print("  %-16s %5d %10.4f %10.4f %11.4f %10.1f%% %9.4f"
          % ("ALL", len(keys), dA.mean(), dB.mean(), tot, 100.0, dH0.mean()))
    print()
    print("  ⛔ T0 CONTROL (never a driving number): ol (recorded future) ADE = %.4f"
          % dOL.mean())
    print("  ⛔ TRIVIAL FLOOR: ha0 (constant velocity, straight) ADE = %.4f" % dH0.mean())
    print()

    # ⭐ THE BOUND A3's PREDICTION RESTS ON.
    lk = np.isin(tok, [LAT.index("LANE_KEEP")])
    tn = np.isin(tok, [LAT.index(t) for t in LAT if t.startswith("TURN_")])
    other = ~(lk | tn)
    s_lk = float(delta[lk].sum()) / len(keys)
    s_tn = float(delta[tn].sum()) / len(keys)
    s_ot = float(delta[other].sum()) / len(keys)
    print("=" * 104)
    print("  ⭐ THE ARITHMETIC BOUND ON A3")
    print("=" * 104)
    print("  A3 keeps the penalty on LANE_KEEP-goal windows and removes it on TURN-goal")
    print("  windows, so it can retain at most the LANE_KEEP share and gives back the")
    print("  TURN share. Contributions to the total mean delta (they sum to it):")
    print("    LANE_KEEP contribution : %+.4f m   (n=%d)  <- A3 RETAINS" % (s_lk, int(lk.sum())))
    print("    TURN      contribution : %+.4f m   (n=%d)  <- A3 GIVES BACK" % (s_tn, int(tn.sum())))
    print("    other     contribution : %+.4f m   (n=%d)  <- A3 retains (shift class)"
          % (s_ot, int(other.sum())))
    print("    total                  : %+.4f m   (n=%d)" % (s_lk + s_tn + s_ot, len(keys)))
    print()
    print("  ⇒ PREDICTED A3 ADE ~ %.4f  (= %s's %.4f + the TURN give-back %+.4f)"
          % (dA.mean() + s_lk + s_ot, A, dA.mean(), -s_tn))
    print("  ⚠️ A BOUND, NOT A FORECAST. It assumes the LANE_KEEP windows behave")
    print("     identically under A3 (they do -- same weight) and that removing the")
    print("     TURN penalty returns those windows EXACTLY to A's behaviour. The second")
    print("     half is the assumption: the goal FIELD is unchanged but the search is")
    print("     stochastic, so the give-back is approximate and can overshoot either way.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
