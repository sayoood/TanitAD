"""check_tick_readout.py -- tool check, no model, no GPU, no data of any card: what the pinned tool's state rebuild measures at a tick. A straight plan with given
0.2 s step speeds is committed for m steps by the pinned tool's own functions (hermite_frames, then v8f_mini_data.yaw_kin_from_track on the appended slice); printed are
the ego speed v0 and the acceleration a0 the model would read at the tick, against the last executed step speed s_m and the executed acceleration (s_m - s_m-1) / 0.2 s,
for three readouts:
  pinned    the tool as it is (central tangent at the last executed knot; it uses the plan's next knot, which is never executed).
            closed form on a straight line: v0 = (s_m + s_m+1) / 2,  a0 = (0.625 (s_m+1 - s_m) - 0.125 (s_m - s_m-1)) / 0.1 s
  backward  the same function given the executed knots only (backward difference at the last executed knot)
  causal    speeds from executed steps only: speed at a knot = the step speed into it, speed at the half step = the mean of that and the step before
            (functions of v8f_t1_eval_reader_rule_ro.py if it is importable, else skipped)
Run on the pod: DISTRULE_DIR=<build> PYTHONPATH=<tree>/stack python check_tick_readout.py <tree>
Last line: ZZ CHECK-TICK-READOUT formula holds: True/False"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("DISTRULE_DIR", os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v8f_t1_eval_rule as W  # noqa: E402

try:
    import v8f_t1_eval_reader_rule_ro as HW  # noqa: E402
except Exception as e:  # noqa: BLE001
    HW = None
    print("wrapper _ro not importable (%s): the causal readout is skipped" % type(e).__name__)

T, path = W.load_pinned(sys.argv[1])
D = T.D
DT = 0.2
ok_all = True
CASES = [("steady 10", [10, 10, 10, 10, 10, 10, 10]),
         ("executed flat, the plan's next step +1.0", [10, 10, 10, 10, 10, 10, 11]),
         ("executed flat, the plan's next step +0.35", [11.7, 11.7, 11.7, 11.7, 11.7, 11.7, 12.05]),
         ("executed flat, the plan's next step -1.0", [10, 10, 10, 10, 10, 10, 9]),
         ("last executed step +1.0, the plan then flat", [10, 10, 10, 10, 10, 11, 11]),
         ("executed ramp +0.5 per step, the plan goes on", [10, 10.5, 11, 11.5, 12, 12.5, 13]),
         ("executed ramp +0.5 per step, the plan then flat", [10, 10.5, 11, 11.5, 12, 12.5, 12.5])]
m = 5


def read(frames, k_prev, knots, s_before, x1):
    xy, psi, v = frames(k_prev, knots, m, 0.0)
    track = np.zeros((2 * m + 1, 4))
    track[0] = [x1, 0.0, 0.0, s_before]
    track[1:, 0:2], track[1:, 2], track[1:, 3] = xy, psi, v
    kin, om = D.yaw_kin_from_track(track)
    return float(kin[-1][0]), float(kin[-1][1])


for name, s in CASES:
    s = np.asarray(s, np.float64)                                 # s[0] = the step before the block, s[1..5] = executed, s[6] = the plan's next step
    x = np.concatenate([[0.0], np.cumsum(s * DT)])                # x[0] = previous knot, x[1] = the current knot, x[2..6] = executed knots, x[7] = next plan knot
    k_prev = np.array([x[0], 0.0])
    knots = np.stack([x[1:], np.zeros(len(x) - 1)], -1)           # knots[0] = current knot, knots[i] = plan knot i
    v0, a0 = read(T.hermite_frames, k_prev, knots, s[0], x[1])
    v_f = 0.5 * (s[m] + s[m + 1])
    a_f = (0.625 * (s[m + 1] - s[m]) - 0.125 * (s[m] - s[m - 1])) / 0.1
    good = abs(v0 - v_f) < 1e-9 and abs(a0 - a_f) < 1e-6
    ok_all = ok_all and good
    vb, ab = read(lambda kp, kn, mm, p0: T.hermite_frames(kp, kn[:mm + 1], mm, p0), k_prev, knots, s[0], x[1])
    txt = "%-48s executed: step %.3f m/s, acceleration %+.3f m/s2 | pinned v0 %.4f a0 %+.4f (closed form %.4f %+.4f, %s) | backward v0 %.4f a0 %+.4f" % (
        name, s[m], (s[m] - s[m - 1]) / DT, v0, a0, v_f, a_f, "equal" if good else "DIFFERENT", vb, ab)
    if HW is not None:
        vc, ac = read(lambda kp, kn, mm, p0: HW.causal_frames(T.hermite_frames, kp, kn, mm, p0), k_prev, knots, s[0], x[1])
        cg = abs(vc - s[m]) < 1e-9 and abs(ac - (s[m] - s[m - 1]) / DT) < 1e-6
        ok_all = ok_all and cg
        txt += " | causal v0 %.4f a0 %+.4f (%s)" % (vc, ac, "equals the executed values" if cg else "DIFFERENT")
    print(txt)
if HW is not None:
    rng = np.random.default_rng(7)
    th = np.cumsum(rng.normal(0.0, 0.03, 12))
    stp = 0.2 * (8.0 + rng.normal(0.0, 0.6, 12))
    pts = np.cumsum(np.stack([stp * np.cos(th), stp * np.sin(th)], -1), 0)
    k_prev, knots = pts[0], pts[1:]
    xo, po, vo = T.hermite_frames(k_prev, knots, m, 0.1)
    xc, pc, vc = HW.causal_frames(T.hermite_frames, k_prev, knots, m, 0.1)
    xp, pp, vp = HW.counted_frames(T.hermite_frames, "pinned")(k_prev, knots, m, 0.1)
    xb, pb, vb = HW.counted_frames(T.hermite_frames, "backward")(k_prev, knots, m, 0.1)
    ch = np.linalg.norm(np.diff(np.vstack([k_prev[None], knots[:m + 1]]), axis=0), axis=-1) / DT
    c1 = np.array_equal(xo, xc) and np.array_equal(po, pc) and np.allclose(vc[1::2], ch[1:], atol=1e-12) and np.allclose(vc[0::2], 0.5 * (ch[:-1] + ch[1:]), atol=1e-12)
    c2 = np.array_equal(xo, xp) and np.array_equal(po, pp) and np.array_equal(vo, vp)
    c3 = np.array_equal(xo[:2 * m - 2], xb[:2 * m - 2]) and np.array_equal(vo[:2 * m - 3], vb[:2 * m - 3]) and abs(vb[-1] - ch[m]) < 1e-12 and not np.array_equal(vo, vb)
    c4 = not np.array_equal(vo, vc)
    print("curved random plan: causal keeps positions and headings and takes its speeds from executed steps: %s | pinned passes through unchanged: %s | backward changes the last interval only and ends at the last step speed: %s | the readouts differ from pinned on this plan: %s" % (c1, c2, c3, c4))
    print("counter after two counted calls: %d" % HW.STATE["hf"]["calls"])
    ok_all = ok_all and c1 and c2 and c3 and c4 and HW.STATE["hf"]["calls"] == 2
print("pinned tool %s" % os.path.basename(path))
print("ZZ CHECK-TICK-READOUT formula holds: %s" % ok_all)
