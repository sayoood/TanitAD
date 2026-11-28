"""check_acc_gain_rzgfh.py (copy of check_acc_gain_rzgf.py, written by make_headbound10k.py; the wrapper name is the only change) -- tool check (control a4) of the readout causal_g of v8f_t1_eval_reader_rule_rzgfh.py (card PREREG_V8F_PROOF_ADD63, ACCGAIN10K), no model, no GPU, no data of any card.
Straight plans with given 0.2 s step speeds are committed for m = 5 steps through the wrapper's frames functions and the pinned v8f_mini_data.yaw_kin_from_track; printed are the
speed v0 and the acceleration a0 the model would read at every executed step.
  1  for the gains -0.25, 0, 0.5, 1 (each with and without --path-exec) v0 = s_i and a0 = gain x (s_i - s_(i-1)) / 0.2 s at every executed step of five speed profiles
  2  with --path-exec the positions and headings equal those of the backward readout bit for bit and the speed column equals that of the same gain without it (curved plan;
     the pinned and the backward positions or headings must differ there, else the comparison would say nothing)
  3  planted faults, both must be caught: (a) the gain is ignored (always 1.0), (b) the path is taken from the pinned call although --path-exec is on
  4  how far the general formula lies from the formula of causal (gain 1) and of causal_z (gain 0), expected at most 1e-12; the routed gains 1 and 0 equal the readouts causal and
     causal_z through the parent wrapper bit for bit; pinned, backward, causal, causal_w, causal_z of this wrapper equal those of the parent bit for bit
  5  the log field max_abs_acceleration_minus_gain_executed is present and at most 1e-6 after all calls; defaults: gain None, path_exec False, readout pinned
Run on the pod: DISTRULE_DIR=<build> PYTHONPATH=<tree>/stack python check_acc_gain.py <tree>
Prints the sha256 of the wrapper file it ran with. Last line: ZZ CHECK-ACC-GAIN all hold: True/False"""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("DISTRULE_DIR", os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v8f_t1_eval_rule as W  # noqa: E402
import v8f_t1_eval_reader_rule_rz as RZ0  # noqa: E402
import v8f_t1_eval_reader_rule_rzgfh as RZ  # noqa: E402

T, path = W.load_pinned(sys.argv[1])
D = T.D
DT = 0.2
m = 5
GAINS = (-0.25, 0.0, 0.5, 1.0)
CASES = [("steady 10", [10, 10, 10, 10, 10, 10, 10]),
         ("executed flat, the plan's next step +1.0", [10, 10, 10, 10, 10, 10, 11]),
         ("last executed step +1.0", [10, 10, 10, 10, 10, 11, 11]),
         ("executed ramp +0.5 per step", [10, 10.5, 11, 11.5, 12, 12.5, 13]),
         ("executed sawtooth", [11.5, 11.6, 11.3, 11.5, 11.7, 11.75, 12.1])]
ok_all = True


def kin_of(frames, k_prev, knots, s_before, x1):
    xy, psi, v = frames(k_prev, knots, m, 0.0)
    track = np.zeros((2 * m + 1, 4))
    track[0] = [x1, 0.0, 0.0, s_before]
    track[1:, 0:2], track[1:, 2], track[1:, 3] = xy, psi, v
    kin, om = D.yaw_kin_from_track(track)
    return kin                                                    # kin[i] = (v0, a0, kappa0) at executed step i (row 0 = NaN)


# the frames functions under test: factory(gain, path_exec) -> frames(k_prev, knots, m, psi0)
def real(g, px):
    return RZ.counted_frames(T.hermite_frames, "causal_g", g, px)


def fault_gain_ignored(g, px):
    return RZ.counted_frames(T.hermite_frames, "causal_g", 1.0, px)


def fault_path_pinned(g, px):
    return RZ.counted_frames(T.hermite_frames, "causal_g", g, False)


def gain_ok(factory, verbose=False):
    good = True
    for name, s in CASES:
        s = np.asarray(s, np.float64)
        x = np.concatenate([[0.0], np.cumsum(s * DT)])
        k_prev = np.array([x[0], 0.0])
        knots = np.stack([x[1:], np.zeros(len(x) - 1)], -1)
        row = []
        for g in GAINS:
            for px in (False, True):
                kin = kin_of(factory(g, px), k_prev, knots, s[0], x[1])
                exp = np.array([g * (s[i] - s[i - 1]) / DT for i in range(1, m + 1)])
                h = np.allclose(kin[1:, 0], s[1:m + 1], atol=1e-9) and np.allclose(kin[1:, 1], exp, atol=1e-6)
                good = good and h
                if px:
                    row.append("g %+.2f a0 %+.4f (exp %+.4f) %s" % (g, kin[-1][1], exp[-1], "holds" if h else "DIFFERENT"))
        if verbose:
            print("%-42s executed acceleration at the tick %+.3f m/s2 | with --path-exec: %s" % (name, (s[m] - s[m - 1]) / DT, " | ".join(row)))
    return good


rng = np.random.default_rng(11)
th = np.cumsum(rng.normal(0.0, 0.03, 12))
stp = 0.2 * (8.0 + rng.normal(0.0, 0.6, 12))
pts = np.cumsum(np.stack([stp * np.cos(th), stp * np.sin(th)], -1), 0)
k_prev, knots = pts[0], pts[1:]


def path_ok(factory):
    xb, pb, vb = RZ.backward_frames(T.hermite_frames, k_prev, knots, m, 0.1)
    good = True
    for g in GAINS:
        xp, pp, vp = factory(g, True)(k_prev, knots, m, 0.1)
        xn, pn, vn = factory(g, False)(k_prev, knots, m, 0.1)
        good = good and np.array_equal(xp, xb) and np.array_equal(pp, pb) and np.array_equal(vp, vn)
    return good


# 1 the gain, real wrapper
g1 = gain_ok(real, verbose=True)
print("1 acceleration read = gain x executed at every executed step (gains %s, with and without --path-exec): %s" % (list(GAINS), g1))
ok_all = ok_all and g1

# 2 the path
xo, po, vo = T.hermite_frames(k_prev, knots, m, 0.1)
xb, pb, vb = RZ.backward_frames(T.hermite_frames, k_prev, knots, m, 0.1)
dpos, dhead = float(np.abs(np.asarray(xo) - np.asarray(xb)).max()), float(np.abs(np.asarray(po) - np.asarray(pb)).max())
nonvac = dpos > 1e-6 or dhead > 1e-6
p2 = path_ok(real)
print("2 --path-exec: positions and headings equal the backward readout bit for bit, speed column equal to that of the same gain without it: %s | pinned against backward on this plan: max position difference %.6f m, max heading difference %.6f rad, comparison not vacuous: %s" % (p2, dpos, dhead, nonvac))
ok_all = ok_all and p2 and nonvac

# 3 planted faults
f_gain_caught = (not gain_ok(fault_gain_ignored)) and g1
f_path_caught = (not path_ok(fault_path_pinned)) and p2
print("3 planted fault (a) gain ignored is caught: %s | planted fault (b) path taken from the pinned call is caught: %s" % (f_gain_caught, f_path_caught))
ok_all = ok_all and f_gain_caught and f_path_caught

# 4 identities
ds = []
for gg, mode in ((1.0, "step"), (0.0, "zero")):
    a = RZ.causal_g_frames(T.hermite_frames, k_prev, knots, m, 0.1, gg, False, True)
    b = RZ.causal_frames(T.hermite_frames, k_prev, knots, m, 0.1, mode)
    ds.append(max(float(np.abs(np.asarray(u) - np.asarray(w_)).max()) for u, w_ in zip(a, b)))
gen_ok = max(ds) <= 1e-12
route_ok = True
for gg, mode in ((1.0, "causal"), (0.0, "causal_z")):
    a = RZ.counted_frames(T.hermite_frames, "causal_g", gg, False)(k_prev, knots, m, 0.1)
    b = RZ0.counted_frames(T.hermite_frames, mode)(k_prev, knots, m, 0.1)
    route_ok = route_ok and all(np.array_equal(u, w_) for u, w_ in zip(a, b))
same_rz = True
for mode in ("pinned", "backward", "causal", "causal_w", "causal_z"):
    a = RZ0.counted_frames(T.hermite_frames, mode)(k_prev, knots, m, 0.1)
    b = RZ.counted_frames(T.hermite_frames, mode)(k_prev, knots, m, 0.1)
    same_rz = same_rz and all(np.array_equal(u, w_) for u, w_ in zip(a, b))
print("4 general formula against the formula of causal (gain 1) and causal_z (gain 0): largest difference %.3e and %.3e (at most 1e-12: %s) | gains 1 and 0 routed through the parent formulas equal causal and causal_z bit for bit: %s | the five readouts of the parent through this wrapper bit for bit: %s" % (ds[0], ds[1], gen_ok, route_ok, same_rz))
ok_all = ok_all and gen_ok and route_ok and same_rz

# 5 log fields and defaults
h = RZ.STATE["hf"]
log_ok = "max_dga" in h and h["max_dga"] <= 1e-6
dflt = RZ.STATE["gain"] is None and RZ.STATE["path_exec"] is False and RZ.STATE["readout"] == "pinned"
print("5 log field max_abs_acceleration_minus_gain_executed present and at most 1e-6 (largest %.3e): %s | defaults gain None, path_exec False, readout pinned: %s" % (h.get("max_dga", float("nan")), log_ok, dflt))
ok_all = ok_all and log_ok and dflt
print("pinned tool %s | wrapper rzgfh sha256 %s" % (os.path.basename(path), hashlib.sha256(open(RZ.__file__, "rb").read()).hexdigest()))
print("ZZ CHECK-ACC-GAIN all hold: %s" % ok_all)
