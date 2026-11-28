"""check_tick_readout_rz.py -- tool check of the two readouts of v8f_t1_eval_reader_rule_rz.py (card PREREG_V8F_PROOF_ADD55), no model, no GPU, no data of any card.
A straight plan with given 0.2 s step speeds is committed for m = 5 steps through the wrapper's frames functions and the pinned v8f_mini_data.yaw_kin_from_track; printed
are the speed v0 and the acceleration a0 the model would read at the tick and at every executed step.
  causal_w  must read v0 = s_m and a0 = (s_i - s_{i-w}) / (0.2 s x w), w = min(i, 5), at executed step i
  causal_z  must read v0 = s_m and a0 = 0 at every executed step
  causal    through this wrapper must equal the causal readout of the wrapper of card ADD54 bit for bit (also pinned and backward)
Run on the pod: DISTRULE_DIR=<build> PYTHONPATH=<tree>/stack python check_tick_readout_rz.py <tree>
Last line: ZZ CHECK-TICK-READOUT-RZ all hold: True/False"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("DISTRULE_DIR", os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v8f_t1_eval_rule as W  # noqa: E402
import v8f_t1_eval_reader_rule_ro as RO  # noqa: E402
import v8f_t1_eval_reader_rule_rz as RZ  # noqa: E402

T, path = W.load_pinned(sys.argv[1])
D = T.D
DT = 0.2
m = 5
ok_all = True
CASES = [("steady 10", [10, 10, 10, 10, 10, 10, 10]),
         ("executed flat, the plan's next step +1.0", [10, 10, 10, 10, 10, 10, 11]),
         ("last executed step +1.0", [10, 10, 10, 10, 10, 11, 11]),
         ("executed ramp +0.5 per step", [10, 10.5, 11, 11.5, 12, 12.5, 13]),
         ("executed sawtooth", [11.5, 11.6, 11.3, 11.5, 11.7, 11.75, 12.1])]


def kin_of(frames, k_prev, knots, s_before, x1):
    xy, psi, v = frames(k_prev, knots, m, 0.0)
    track = np.zeros((2 * m + 1, 4))
    track[0] = [x1, 0.0, 0.0, s_before]
    track[1:, 0:2], track[1:, 2], track[1:, 3] = xy, psi, v
    kin, om = D.yaw_kin_from_track(track)
    return kin                                                    # kin[i] = (v0, a0, kappa0) at executed step i (row 0 = NaN)


for name, s in CASES:
    s = np.asarray(s, np.float64)
    x = np.concatenate([[0.0], np.cumsum(s * DT)])
    k_prev = np.array([x[0], 0.0])
    knots = np.stack([x[1:], np.zeros(len(x) - 1)], -1)
    kw = kin_of(lambda kp, kn, mm, p0: RZ.causal_frames(T.hermite_frames, kp, kn, mm, p0, "window"), k_prev, knots, s[0], x[1])
    kz = kin_of(lambda kp, kn, mm, p0: RZ.causal_frames(T.hermite_frames, kp, kn, mm, p0, "zero"), k_prev, knots, s[0], x[1])
    exp_w = np.array([(s[i] - s[i - min(i, 5)]) / (DT * min(i, 5)) for i in range(1, m + 1)])
    gw = np.allclose(kw[1:, 0], s[1:m + 1], atol=1e-9) and np.allclose(kw[1:, 1], exp_w, atol=1e-6)
    gz = np.allclose(kz[1:, 0], s[1:m + 1], atol=1e-9) and np.allclose(kz[1:, 1], 0.0, atol=1e-6)
    ok_all = ok_all and gw and gz
    print("%-42s executed acceleration at the tick %+.3f m/s2 | causal_w v0 %.4f a0 %+.4f (expected %+.4f, %s) | causal_z v0 %.4f a0 %+.4f (%s)" % (
        name, (s[m] - s[m - 1]) / DT, kw[-1][0], kw[-1][1], exp_w[-1], "all steps hold" if gw else "DIFFERENT", kz[-1][0], kz[-1][1], "all steps hold" if gz else "DIFFERENT"))

rng = np.random.default_rng(11)
th = np.cumsum(rng.normal(0.0, 0.03, 12))
stp = 0.2 * (8.0 + rng.normal(0.0, 0.6, 12))
pts = np.cumsum(np.stack([stp * np.cos(th), stp * np.sin(th)], -1), 0)
k_prev, knots = pts[0], pts[1:]
same = True
for mode in ("pinned", "backward", "causal"):
    a = RO.counted_frames(T.hermite_frames, mode)(k_prev, knots, m, 0.1)
    b = RZ.counted_frames(T.hermite_frames, mode)(k_prev, knots, m, 0.1)
    same = same and all(np.array_equal(x1, x2) for x1, x2 in zip(a, b))
xo, po, vo = T.hermite_frames(k_prev, knots, m, 0.1)
keep = True
for mode in ("causal_w", "causal_z"):
    xb, pb, vb = RZ.counted_frames(T.hermite_frames, mode)(k_prev, knots, m, 0.1)
    ch = np.linalg.norm(np.diff(np.vstack([k_prev[None], knots[:m + 1]]), axis=0), axis=-1) / DT
    keep = keep and np.array_equal(xo, xb) and np.array_equal(po, pb) and np.allclose(vb[1::2], ch[1:], atol=1e-12)
h = RZ.STATE["hf"]
print("curved random plan: pinned, backward, causal equal the wrapper of card ADD54 bit for bit: %s | the two new readouts keep positions and headings and the knot speeds: %s | log fields present: %s" % (
    same, keep, all(k in h for k in ("max_a0", "max_daw", "sum_a0"))))
ok_all = ok_all and same and keep and all(k in h for k in ("max_a0", "max_daw", "sum_a0"))
print("pinned tool %s" % os.path.basename(path))
print("ZZ CHECK-TICK-READOUT-RZ all hold: %s" % ok_all)
