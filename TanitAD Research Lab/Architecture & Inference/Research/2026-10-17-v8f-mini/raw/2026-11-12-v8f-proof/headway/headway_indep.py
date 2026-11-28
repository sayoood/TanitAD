"""headway_indep.py -- independent recomputation of the primary cell U for ONE arm (HIER_s0) of card ADD25.
Shares no code with headway_read.py (file loading only: numpy npz).  Plain per-window loops, a different projection (dense sampling of the logged path,
then exact refinement by the closed form on the two neighbouring segments is NOT used: the exact segment projection is done with a scalar loop).
Compares with RESULT_HEADWAY.json (levels.HIER.cells.U.per_seed[0]).  Usage: python headway_indep.py [result_json]"""
import glob
import json
import math
import sys

import numpy as np

DUMP = "F:/TanitAD-scratch/v8proof/hiergap/pull/HIER_s0/dump_HIER_s0_m10_causal_heads"
JOIN = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz"
RES = sys.argv[1] if len(sys.argv) > 1 else "F:/TanitAD-scratch/v8proof/headway/out_q12/RESULT_HEADWAY.json"


def project_arc(path, q):
    """path: list of (x, y) from the origin; arc of the orthogonal projection of q on the nearest non-degenerate segment; the path is extended past both ends along the end directions."""
    segs = []
    cum = 0.0
    for i in range(len(path) - 1):
        ax, ay = path[i]
        bx, by = path[i + 1]
        L = math.hypot(bx - ax, by - ay)
        if L > 1e-9:
            segs.append((ax, ay, bx, by, L, cum, (bx - ax) / L, (by - ay) / L))
        cum += L
    if not segs:
        segs_dir = (1.0, 0.0)
        return q[0] * segs_dir[0] + q[1] * segs_dir[1]
    best = None
    for k, (ax, ay, bx, by, L, c0, ux, uy) in enumerate(segs):
        s = (q[0] - ax) * ux + (q[1] - ay) * uy                       # signed distance along the segment from its start
        if k == 0:
            lo = -1.0e9
        else:
            lo = 0.0
        hi = L if k < len(segs) - 1 else 1.0e9
        s = min(max(s, lo), hi)
        px, py = ax + s * ux, ay + s * uy
        d = math.hypot(q[0] - px, q[1] - py)
        if best is None or d < best[0]:
            best = (d, c0 + s)
    return best[1]


def main():
    fs = sorted(glob.glob(DUMP + "/ep*.npz"))
    assert len(fs) == 141
    Z = np.load(JOIN)
    off, n_cl, lv, lg = Z["off"], Z["n"], Z["lead_valid"], Z["lead_gap"]
    n_def = n_uns = n_lead = 0
    eps = set()
    for f in fs:
        z = np.load(f)
        for w in range(len(z["ws"])):
            ci, t0 = int(z["clip_index"][w]), int(z["ws"][w])
            frames = []
            ok = True
            prev = None
            for k in range(0, 31):
                row = int(off[ci]) + t0 + k
                v = (t0 + k < int(n_cl[ci])) and int(lv[row]) == 1 and math.isfinite(float(lg[row])) and 0.0 <= float(lg[row]) <= 60.0
                gv = float(lg[row])
                if v and prev is not None and abs(gv - prev) > 3.0:
                    v = False
                if not v:
                    break
                frames.append(gv)
                prev = gv
            if len(frames) < 11:                                 # frame 0 and frames 1..10
                continue
            n_lead += 1
            eps.add(ci)
            ns = min(30, len(frames) - 1)
            gpts = [(0.0, 0.0)] + [tuple(p) for p in z["g"][w]]
            cpts = [(0.0, 0.0)] + [tuple(p) for p in z["cl"][w]]
            cum = [0.0]
            for i in range(1, 31):
                cum.append(cum[-1] + math.hypot(gpts[i][0] - gpts[i - 1][0], gpts[i][1] - gpts[i - 1][1]))
            unsafe_l = unsafe_m = False
            for k in range(1, ns + 1):
                a_m = project_arc(gpts, cpts[k])
                gap_l = frames[k]
                gap_m = gap_l + cum[k] - a_m
                vl = max(math.hypot(gpts[k][0] - gpts[k - 1][0], gpts[k][1] - gpts[k - 1][1]) / 0.2, 0.5)
                vm = max(math.hypot(cpts[k][0] - cpts[k - 1][0], cpts[k][1] - cpts[k - 1][1]) / 0.2, 0.5)
                if gap_l < 2.0 or gap_l / vl < 1.0:
                    unsafe_l = True
                if gap_m < 2.0 or gap_m / vm < 1.0:
                    unsafe_m = True
            if unsafe_l:
                continue
            n_def += 1
            n_uns += int(unsafe_m)
    u = n_uns / n_def
    print("independent HIER_s0: lead windows %d, episodes %d, windows defined for U %d, model-unsafe %d, U %.12f" % (n_lead, len(eps), n_def, n_uns, u))
    R = json.load(open(RES, encoding="utf-8"))
    ref = R["levels"]["HIER"]["cells"]["U"]["per_seed"][R["levels"]["HIER"]["seeds"].index(0)]
    print("reader HIER_s0 U %.12f ; abs difference %.3e" % (ref, abs(ref - u)))
    # window-level U is a share over windows, but the reader's per-seed value is the mean over windows within seed over finite windows: same quantity
    print("AGREE to 1e-9" if abs(ref - u) <= 1e-9 else "DIFFERENCE reported")


if __name__ == "__main__":
    main()
