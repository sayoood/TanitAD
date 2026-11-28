#!/usr/bin/env python3
"""THE CORRECTED TURNING CRITERION, applied to every BANKED arm. 0 GPU.

⛔ WHY. `turn_left` recall on this panel is unreachable: the v1 gate is
`|dyaw| > 0.15` rad and the GROUND TRUTH clears it on only 3 of 9 TURN_L-goal
windows (median dyaw 0.0431). An arm earns recall by OUT-TURNING the human, and
pays for it in ADE -- so "accurate" and "turns" are in direct conflict BY
CONSTRUCTION of the gate at this panel's speeds (v0 median 1.40 m/s).

⇒ This replaces it with three statistics that are reachable and say what they
mean, pre-registered in RESULT.md §F4 BEFORE the A4 arm landed:
   1. median |kappa_plan - kappa_gt| on TURN-goal windows  (no speed threshold)
   2. the v2 CURVATURE gate |kappa| >= 1/60, reported beside v1
   3. SIGNED dyaw error vs GT, so over- and under-turning are distinguishable
      instead of pooled into one recall number

⛔ CONTROLS, and the reading is void without them:
   * `g` (GROUND TRUTH) must read kappa error EXACTLY 0 -- it is the reference.
   * `ha0` (constant velocity, straight) is the NO-TURN floor: any arm that does
     not beat it on kappa error has added nothing over driving straight.

ASCII output only.
"""
import glob, os, sys
import numpy as np
from tanitad.models.vocab_v7 import TACTICAL_LAT_ACTIONS_V7
from tanitad.refs.refc_tactical import YAW_TURN_RAD, CURV_TURN_MAN_PER_M
LAT = list(TACTICAL_LAT_ACTIONS_V7)
R = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/refav1_margin/p4out"
DT = 0.2

def curv_of(path):
    """per-step curvature of a [H,2] displacement path: dtheta / ds."""
    d = np.diff(path, axis=0)
    s = np.hypot(d[:, 0], d[:, 1])
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    k = np.zeros(len(s) - 1)
    ok = s[1:] > 1e-3
    k[ok] = np.diff(th)[ok] / s[1:][ok]
    return k, float(th[-1] - th[0])

def load(tag):
    d = {}
    dd = os.path.join(R, "dump_" + tag)
    if not os.path.isdir(dd): return None
    for f in sorted(glob.glob(os.path.join(dd, "decisions", "*.npz"))):
        z = np.load(f, allow_pickle=True); e = int(os.path.basename(f)[2:5])
        for i, w in enumerate(z["ws"].tolist()):
            d[(e, int(w))] = dict(tok=LAT[int(z["goal_lat_cl"][i])], ctl=z["cl_controls"][i])
    for f in sorted(glob.glob(os.path.join(dd, "ep*.npz"))):
        z = np.load(f, allow_pickle=True); e = int(z["eid"][0])
        for i, w in enumerate(z["ws"].tolist()):
            k = (e, int(w))
            if k in d:
                d[k]["cl"] = z["cl"][i]; d[k]["g"] = z["g"][i]
                d[k]["ha0"] = z["ha0"][i]; d[k]["v0"] = float(z["v0"][i])
    return d

arms = [a for a in ("ccos_argmax", "kamm07", "combined", "wk15", "best",
                    "wk151", "lonshift", "bestlad") if load(a)]
base = load("ccos_argmax")
turn = [k for k in base if base[k]["tok"].startswith("TURN_")]
print("=" * 108)
print("THE CORRECTED TURNING CRITERION on the BANKED arms -- TURN-goal windows, n=%d" % len(turn))
print("  (v1 recall is NOT used: the GT clears its gate on 3/9 TURN_L windows -- see turngate.txt)")
print("=" * 108)
hdr = ("  %-13s %9s | %11s %11s | %9s %9s | %11s" %
       ("arm", "ADE_all", "kappa MAE", "vs ha0", "v2 pass", "v1 pass", "signed dyaw"))
print(hdr); print("  " + "-" * (len(hdr) - 2))

def row(tag, use="cl"):
    d = load(tag) if tag not in ("g", "ha0") else base
    ke, dye, v2, v1 = [], [], 0, 0
    ade = []
    for k in turn:
        p = d[k][use] if use in d[k] else d[k]["cl"]
        kg, dyg = curv_of(d[k]["g"])
        kp, dyp = curv_of(p)
        n = min(len(kg), len(kp))
        ke.append(float(np.median(np.abs(kp[:n] - kg[:n]))))
        dye.append(dyp - dyg)
        if float(np.abs(kp).mean()) >= CURV_TURN_MAN_PER_M: v2 += 1
        if abs(dyp) > YAW_TURN_RAD: v1 += 1
        ade.append(float(np.sqrt(((p - d[k]["g"]) ** 2).sum(-1)).mean()))
    return (np.median(ke), np.median(dye), v2, v1, np.mean(ade))

h0 = row("ccos_argmax", "ha0")
gg = row("ccos_argmax", "g")
print("  %-13s %9.4f | %11.5f %11s | %8d/%-2d %8d/%-2d | %+11.4f   <- CONTROL (must be 0)"
      % ("g  GT", gg[4], gg[0], "-", gg[2], len(turn), gg[3], len(turn), gg[1]))
print("  %-13s %9.4f | %11.5f %11s | %8d/%-2d %8d/%-2d | %+11.4f   <- FLOOR (straight)"
      % ("ha0 straight", h0[4], h0[0], "-", h0[2], len(turn), h0[3], len(turn), h0[1]))
print("  " + "-" * (len(hdr) - 2))
res = []
for a in arms:
    r = row(a)
    res.append((a, r))
    print("  %-13s %9.4f | %11.5f %+10.5f | %8d/%-2d %8d/%-2d | %+11.4f"
          % (a, r[4], r[0], r[0] - h0[0], r[2], len(turn), r[3], len(turn), r[1]))
print()
print("  ⛔ 'BEATS THE STRAIGHT-LINE FLOOR ON CURVATURE' is the reachable form of 'it turns':")
better = [a for a, r in res if r[0] < h0[0]]
print("     arms with kappa MAE BELOW ha0's %.5f : %s" % (h0[0], better if better else "NONE"))
print()
print("  ⭐ RANK ON THE PI's ACTUAL QUESTION -- accurate AND turning, both columns:")
print("     (ADE over TURN-goal windows, and kappa MAE vs the human)")
for a, r in sorted(res, key=lambda x: x[1][4])[:5]:
    verdict = "TURNS (beats straight)" if r[0] < h0[0] else "does NOT beat straight"
    print("       %-13s ADE %.4f   kappa MAE %.5f   %s" % (a, r[4], r[0], verdict))
