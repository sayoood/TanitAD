"""check_head_bound.py -- tool check h4 of card PREREG_V8F_PROOF_ADD67 (HEADBOUND10K) of the parameter --kappa-max of v8f_t1_eval_reader_rule_rzgfh.py. No model, no GPU, no data of any card.
Synthetic plans are committed through the wrapper's frames function (counted_frames of the real pinned hermite_frames), as the pinned tool's commit_plan does (knots = current knot first, m = 10).
  1  a plan that jitters by millimetres around a standing car: the readout's heading jumps (the case would say nothing otherwise); the bounded heading keeps every step within K x distance
     (1e-9 rad) and the bound acts on some frames; the output turns towards the readout's heading at every frame where the bound acts
  2  a plan along an arc of 10 m radius at K = 0.2: the output heading equals the readout's heading at every frame (bit for bit) and the bound acts on no frame
  3  the same arc at K = 0.05: the bound acts on every frame and the heading step of every frame equals K x the distance between the two frame positions (1e-12 rad), turning towards the readout
  4  a huge K (1e9): the frames equal the frames of the parent counted_frames (no --kappa-max) bit for bit on the three plans above, a standing car (all knots equal) and 200 random plans
  5  the counters of the readout log: frames_total, frames_bound_acted, the speed bins add up, max_excess_rad at most 1e-9, max_abs_diff_where_not_acted, max_abs_diff_xy and max_abs_diff_v are 0.0
  6  the parameter is refused for K <= 0, nan, inf and without --readout; absent, the wrapper's STATE has kappa None and kb None
  7  planted faults, both must be caught by checks 1 to 3: (a) the bound is not applied, (b) the bound is applied with the turn of the wrong sign
Run on the pod: CUDA_VISIBLE_DEVICES=-1 DISTRULE_DIR=<build> PYTHONPATH=<tree>/stack python check_head_bound.py <tree>
Prints the sha256 of the wrapper file it ran with. Last line: ZZ CHECK-HEAD-BOUND all hold: True/False"""
import hashlib
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("DISTRULE_DIR", os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v8f_t1_eval_rule as W  # noqa: E402
import v8f_t1_eval_reader_rule_rzgfh as RZ  # noqa: E402

T, path = W.load_pinned(sys.argv[1])
M = 10
TOL_EXC = 1e-9


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def commit(frames, k_prev, knots, psi0):
    return frames(k_prev, np.asarray(knots, np.float64), M, psi0)


def scenario_jitter():
    rng = np.random.default_rng(3)
    x0 = np.array([5.0, -2.0])
    knots = x0[None] + 0.005 * rng.standard_normal((M + 3, 2))        # five millimetres of jitter around a standing car
    knots[0] = x0
    return x0 + 0.005 * rng.standard_normal(2), knots, 0.3


def scenario_arc(R=10.0, step=1.0):
    th = np.arange(-1, M + 3) * step / R
    P = np.stack([R * np.sin(th), R * (1.0 - np.cos(th))], axis=1)    # P[0] = knot before the current one, P[1] = the current knot (angle 0)
    return P[0], P[1:], 0.0


def scenario_stand():
    return np.array([1.0, 1.0]), np.tile(np.array([[1.0, 1.0]]), (M + 3, 1)), -0.7


def scenario_random(i):
    rng = np.random.default_rng(1000 + i)
    v = rng.uniform(0.0, 12.0)
    ang = rng.uniform(-math.pi, math.pi) + np.cumsum(0.4 * rng.standard_normal(M + 3))
    mag = v * 0.2 * (1.0 + 0.3 * rng.standard_normal(M + 3)).clip(0.0, None)
    if i % 5 == 0:
        mag = mag * 0.01                                                # near a standstill
    steps = mag[:, None] * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    knots = np.cumsum(steps, axis=0)
    return knots[0] - steps[0], knots, float(rng.uniform(-math.pi, math.pi))


def run(kappa, scen, bad_bound=None):
    """the frames of the wrapper with the bound (rec = the counters), the frames of the readout, the start heading and position"""
    k_prev, knots, psi0 = scen
    RZ.STATE["kb"] = RZ.kb_new()
    f_b = RZ.counted_frames(T.hermite_frames, "pinned", None, False, kappa)
    f_0 = RZ.counted_frames(T.hermite_frames, "pinned", None, False, None)
    if bad_bound is not None:
        old = RZ.bound_heading
        RZ.bound_heading = bad_bound
    try:
        xyb, psib, vb = commit(f_b, k_prev, knots, psi0)
    finally:
        if bad_bound is not None:
            RZ.bound_heading = old
    xy0, psi_r, v0 = commit(f_0, k_prev, knots, psi0)
    return (xyb, psib, vb), (xy0, psi_r, v0), RZ.STATE["kb"], knots[0], psi0


def steps_of(psi, psi0):
    p = np.concatenate([[psi0], psi])
    return np.array([wrap(p[i + 1] - p[i]) for i in range(len(psi))])


def dists_of(xy, x0):
    p = np.vstack([x0[None], xy])
    return np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))


def toward(psib, psir, psi0):
    """at every frame where the output differs from the readout's heading, the output step has the sign of (readout minus previous output)"""
    prev = psi0
    ok = True
    for o, r in zip(psib, psir):
        d = wrap(r - prev)
        s = wrap(o - prev)
        if o != r:
            ok = ok and (d > 0) == (s > 0) and s != 0.0
        prev = o
    return bool(ok)


def checks(bad_bound=None, verbose=False):
    """checks 1 to 3 (the ones a planted fault must break); returns a dict name -> bool and the lines printed"""
    res, lines = {}, []
    # 1 jitter around a standing car
    kappa = 0.2
    (xyb, psib, vb), (xyr, psir, vr), kb, x0, psi0 = run(kappa, scenario_jitter(), bad_bound)
    d_r, d_b = np.abs(steps_of(psir, psi0)), np.abs(steps_of(psib, psi0))
    lim = kappa * dists_of(xyb, x0)
    jumps = float(d_r.max())
    within = bool((d_b <= lim + TOL_EXC).all())
    res["1_jitter"] = bool(jumps > 0.5 and within and kb["frames_bound_acted"] > 0 and toward(psib, psir, psi0))
    lines.append("1 jitter around a standing car, K 0.2: readout's largest heading step %.3f rad (above 0.5: %s) | bounded steps within K x distance: %s (largest excess %.3e rad) | frames acted %d of %d | towards the readout: %s"
                 % (jumps, jumps > 0.5, within, float((d_b - lim).max()), kb["frames_bound_acted"], kb["frames_total"], toward(psib, psir, psi0)))
    # 2 arc of 10 m at K 0.2
    (xyb, psib, vb), (xyr, psir, vr), kb, x0, psi0 = run(0.2, scenario_arc(), bad_bound)
    res["2_arc_K02_untouched"] = bool(np.array_equal(psib, psir) and kb["frames_bound_acted"] == 0 and kb["frames_total"] == 2 * M)
    lines.append("2 arc of radius 10 m at K 0.2: output heading equals the readout's at every frame (bit for bit): %s | frames acted %d of %d | largest readout step per metre %.4f (at most 0.1 x 1.1)"
                 % (np.array_equal(psib, psir), kb["frames_bound_acted"], kb["frames_total"], float((np.abs(steps_of(psir, psi0)) / dists_of(xyr, x0)).max())))
    # 3 same arc at K 0.05
    kap = 0.05
    (xyb, psib, vb), (xyr, psir, vr), kb, x0, psi0 = run(kap, scenario_arc(), bad_bound)
    st, lim = np.abs(steps_of(psib, psi0)), kap * dists_of(xyb, x0)
    err = float(np.abs(st - lim).max())
    res["3_arc_K005_bounded"] = bool(kb["frames_bound_acted"] == 2 * M and err <= 1e-12 and toward(psib, psir, psi0))
    lines.append("3 the same arc at K 0.05: frames acted %d of %d | largest |step - K x distance| %.3e rad (at most 1e-12) | towards the readout: %s"
                 % (kb["frames_bound_acted"], kb["frames_total"], err, toward(psib, psir, psi0)))
    return res, lines


ok_all = True
res, lines = checks(None)
for ln in lines:
    print(ln)
ok_all = ok_all and all(res.values())

# 4 a huge K gives the parent's frames bit for bit
scen = [("jitter", scenario_jitter()), ("arc", scenario_arc()), ("stand", scenario_stand())] + [("random%d" % i, scenario_random(i)) for i in range(200)]
nbad, nacted = 0, 0
for nm, sc in scen:
    for K_ in (1e9,):
        (xyb, psib, vb), (xyr, psir, vr), kb, x0, psi0 = run(K_, sc)
        same = np.array_equal(xyb, xyr) and np.array_equal(psib, psir) and np.array_equal(vb, vr)
        nbad += int(not same)
        nacted += kb["frames_bound_acted"]
print("4 a huge K (1e9): the frames of %d plans (jitter, arc, standing car, 200 random) equal the readout's bit for bit: %s (plans differing %d; frames acted in total %d)" % (len(scen), nbad == 0, nbad, nacted))
ok_all = ok_all and nbad == 0 and nacted == 0

# 5 the counters of the readout log, on a mixed batch at K 0.2 and 0.05
RZ.STATE["kb"] = RZ.kb_new()
kbm = RZ.STATE["kb"]
tot_frames, diffs_nacted, dxy, dv, mexc = 0, 0.0, 0.0, 0.0, -1.0
for K_ in (0.2, 0.05):
    f_b = RZ.counted_frames(T.hermite_frames, "pinned", None, False, K_)
    for nm, (k_prev, knots, psi0) in scen[:60]:
        commit(f_b, k_prev, knots, psi0)
        tot_frames += 2 * M
bins = sum(v["frames_total"] for v in kbm["by_speed"].values())
binsa = sum(v["frames_bound_acted"] for v in kbm["by_speed"].values())
c5 = bool(kbm["frames_total"] == tot_frames and bins == tot_frames and binsa == kbm["frames_bound_acted"] and kbm["max_excess_rad"] is not None and kbm["max_excess_rad"] <= TOL_EXC
          and kbm["max_abs_diff_where_not_acted"] == 0.0 and kbm["max_abs_diff_xy"] == 0.0 and kbm["max_abs_diff_v"] == 0.0 and kbm["frames_bound_acted"] > 0)
print("5 counters over %d frames: frames_total %d | speed bins add up to the total %s and to the acted frames %s | frames acted %d | max_excess_rad %.3e (at most 1e-9) | max_abs_diff_where_not_acted %s | max_abs_diff_xy %s | max_abs_diff_v %s | %s"
      % (tot_frames, kbm["frames_total"], bins == tot_frames, binsa == kbm["frames_bound_acted"], kbm["frames_bound_acted"], kbm["max_excess_rad"], kbm["max_abs_diff_where_not_acted"], kbm["max_abs_diff_xy"],
         kbm["max_abs_diff_v"], c5))
ok_all = ok_all and c5

# 6 refusals and defaults
refused = []
for argv in (["--tree", "x", "--kappa-max", "0", "--readout", "pinned"], ["--tree", "x", "--kappa-max", "-0.1", "--readout", "pinned"], ["--tree", "x", "--kappa-max", "nan", "--readout", "pinned"],
             ["--tree", "x", "--kappa-max", "inf", "--readout", "pinned"], ["--tree", "x", "--kappa-max", "0.2"]):
    try:
        RZ.main(argv)
        refused.append(False)
    except SystemExit as ex:
        refused.append(str(ex).startswith("REFUSED"))
RZ.STATE["kappa"], RZ.STATE["kb"] = None, None
c6 = bool(all(refused) and RZ.STATE["kappa"] is None and RZ.STATE["kb"] is None)
print("6 refused K 0, K -0.1, nan, inf and K without --readout: %s | absent: kappa None, kb None" % refused)
ok_all = ok_all and c6

# 7 planted faults, both must be caught by checks 1 to 3
def fault_not_applied(xy, psi, v, xy0, psi0, kappa, rec=None):
    return np.asarray(psi, np.float64).copy()


def fault_wrong_sign(xy, psi, v, xy0, psi0, kappa, rec=None):
    xy, psi = np.asarray(xy, np.float64), np.asarray(psi, np.float64)
    out = np.empty(len(psi))
    prev, pxy = float(psi0), np.asarray(xy0, np.float64)
    for f in range(len(psi)):
        lim = float(kappa) * float(np.hypot(xy[f, 0] - pxy[0], xy[f, 1] - pxy[1]))
        d = wrap(float(psi[f]) - prev)
        o = float(psi[f]) if abs(d) <= lim else wrap(prev - (1.0 if d > 0 else -1.0) * lim)
        out[f] = o
        prev, pxy = o, xy[f]
    return out


caught = {}
for nm, fn in (("the bound is not applied", fault_not_applied), ("the bound is applied with the turn of the wrong sign", fault_wrong_sign)):
    r, _ = checks(fn)
    caught[nm] = not all(r.values())
print("7 planted faults caught: %s" % "; ".join("%s: %s" % (k, v) for k, v in caught.items()) + " | caught: %s" % all(caught.values()))
ok_all = ok_all and all(caught.values())
print("pinned tool %s | wrapper rzgfh sha256 %s" % (os.path.basename(path), hashlib.sha256(open(RZ.__file__, "rb").read()).hexdigest()))
print("ZZ CHECK-HEAD-BOUND all hold: %s" % ok_all)
