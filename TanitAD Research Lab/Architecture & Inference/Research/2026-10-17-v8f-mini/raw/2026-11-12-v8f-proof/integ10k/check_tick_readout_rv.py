"""check_tick_readout_rv.py -- tool check (control v9) of the readout causal_v of v8f_t1_eval_reader_rule_rv.py (card PREREG_V8F_PROOF_ADD58), no model, no GPU, no data of any card.
A stand-in window state holds a track in the layout of the pinned tool's WindowState (row 2q = knot q: x, y, heading, speed; the rows up to 2*j finite; the blocks are
committed through the frames function the wrapper installs and through the wrapper's own commit_plan wrapper, which sets STATE["exec"]). Plans are straight with known step
speeds. For every executed step i of every block the acceleration the model would read is compared with the hand-computed value
    a_i = (S[k] - S[k-w]) / (0.2 s x w),  k = the index of executed knot i,  w = min(10, k - 1) (the steps before S[k] in the track), S[q] = step speed into knot q,
read two ways: from the speed column of the frames (knot speed and half-step speed) and through the pinned tool's own D.yaw_kin_from_track. Cases: a logged past of 3, 10, 11 and
14 steps (shorter and longer than ten), blocks of 5 steps with two and three replans, blocks of 10 steps with two replans. Then: pinned, backward, causal, causal_w, causal_z of
the wrapper rv equal those of the wrapper rz bit for bit (curved random plan); the readout log fields; causal_v refuses to run without a window state.
Run on the pod: DISTRULE_DIR=<build> PYTHONPATH=<tree>/stack python check_tick_readout_rv.py <tree>
Last line: ZZ CHECK-TICK-READOUT-RV all hold: True/False"""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("DISTRULE_DIR", os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v8f_t1_eval_rule as W  # noqa: E402
import v8f_t1_eval_reader_rule_rz as RZ  # noqa: E402
import v8f_t1_eval_reader_rule_rv as RV  # noqa: E402

T, path = W.load_pinned(sys.argv[1])
D = T.D
DT = 0.2
ok_all = True
FRAMES = RV.counted_frames(T.hermite_frames, "causal_v")


class Stand:
    """the part of WindowState the readout reads: track, j, t0 ; commit_plan as the pinned tool's (knots from the plan, k_prev = track[2*(j-1)], the slice written)"""

    def __init__(self, hist):
        self.j = len(hist)
        self.t0 = self.j
        x = np.concatenate([[0.0], np.cumsum(np.asarray(hist, np.float64) * DT)])          # knots 0 .. j
        s = np.concatenate([[hist[0]], np.asarray(hist, np.float64)])                         # speed at knot q (the step into it; knot 0 has none)
        self.track = np.full((2 * (self.j + 40) + 1, 4), np.nan)
        for q in range(self.j + 1):
            self.track[2 * q] = [x[q], 0.0, 0.0, s[q]]
            if q:
                self.track[2 * q - 1] = [0.5 * (x[q - 1] + x[q]), 0.0, 0.0, 0.5 * (s[q - 1] + s[q])]

    def commit_plan(self, plan_steps, m):
        j = self.j
        xs = self.track[2 * j, 0] + np.concatenate([[0.0], np.cumsum(np.asarray(plan_steps, np.float64) * DT)])
        knots = np.stack([xs, np.zeros_like(xs)], -1)                                           # knots[0] = the current knot, knots[i] = the plan's knot j+i
        xy, psi, v = FRAMES(self.track[2 * (j - 1), :2], knots, m, 0.0)
        sl = slice(2 * j + 1, 2 * (j + m) + 1)
        self.track[sl, 0:2], self.track[sl, 2], self.track[sl, 3] = xy, psi, v
        self.j = j + m


Stand.commit_plan = RV.commit_plan_with_state(Stand.commit_plan)
rng = np.random.default_rng(58)
CASES = [("past 3 steps, 5-step blocks x2", 3, 5, 2), ("past 10 steps, 5-step blocks x3", 10, 5, 3), ("past 11 steps, 5-step blocks x3", 11, 5, 3),
         ("past 14 steps, 5-step blocks x3", 14, 5, 3), ("past 3 steps, 10-step blocks x2", 3, 10, 2), ("past 14 steps, 10-step blocks x2", 14, 10, 2)]
n_calls, sum_len, n_short, sum_dump, n_dump, worst = 0, 0, 0, 0.0, 0, 0.0
for name, j0, m, nb in CASES:
    hist = 8.0 + 2.0 * np.sin(0.9 * np.arange(j0)) + rng.normal(0.0, 0.4, j0)
    st = Stand(hist)
    S = {q: float(hist[q - 1]) for q in range(1, j0 + 1)}
    g_col, g_kin, g_knot = True, True, True
    for b in range(nb):
        j = st.j
        plan = np.clip(8.0 + 3.0 * rng.normal(0.0, 1.0, m + 3), 1.0, None)
        STATE_BEFORE = RV.STATE["exec"]
        st.commit_plan(plan, m)
        n_calls += 1
        g_knot = g_knot and STATE_BEFORE is None and RV.STATE["exec"] is None
        sl = st.track[2 * j:2 * (j + m) + 1]
        kin, om = D.yaw_kin_from_track(sl)
        for i in range(1, m + 1):
            S[j + i] = float(plan[i - 1])
            k = j + i
            w = max(1, min(10, k - 1))
            exp = (S[k] - S[k - w]) / (DT * w)
            v_half, v_knot = st.track[2 * (j + i - 1) + 1, 3], st.track[2 * (j + i), 3]
            a_col = (v_knot - v_half) / (0.5 * DT)
            g_col = g_col and abs(a_col - exp) <= 1e-6 and abs(v_knot - S[k]) <= 1e-9
            g_kin = g_kin and abs(kin[i][1] - exp) <= 1e-6 and abs(kin[i][0] - S[k]) <= 1e-9
            worst = max(worst, abs(a_col - exp), abs(kin[i][1] - exp))
        k = j + m
        w_m = max(1, min(10, k - 1))
        sum_len += w_m
        n_short += int(w_m < 10)
        if (j - st.t0 + m) - 10 >= 1:
            n_dump += 1
            sum_dump += abs((S[k] - S[k - w_m]) / (DT * w_m))
    ok_all = ok_all and g_col and g_kin and g_knot
    print("%-36s every executed step of every block: speed column %s | pinned yaw_kin_from_track %s | holder reset after each commit %s | window at the last knot of block 1: %d steps" % (
        name, "holds" if g_col else "DIFFERENT", "holds" if g_kin else "DIFFERENT", "holds" if g_knot else "DIFFERENT", max(1, min(10, j0 + m - 1))))

h = RV.STATE["hf"]
log_ok = (h.get("w2_calls") == n_calls and h.get("w2_sum_len") == sum_len and h.get("w2_calls") > 0 and h.get("w2_n_short") == n_short and h.get("w2_max_diff", 1.0) <= 1e-6
          and h.get("w2_dump_calls") == n_dump and abs(h.get("w2_dump_sum_abs_a0", -1.0) - sum_dump) <= 1e-6 and h["max_dv"] <= 1e-9)
ok_all = ok_all and log_ok
print("readout log: calls %s (expected %d) sum of window lengths %s (expected %d) calls below ten steps %s (expected %d) largest tick-minus-window difference %.3g, "
      "dump-reach calls %s (expected %d) their sum %.9f (expected %.9f), largest speed difference %.3g | %s" % (
          h.get("w2_calls"), n_calls, h.get("w2_sum_len"), sum_len, h.get("w2_n_short"), n_short, h.get("w2_max_diff", float("nan")), h.get("w2_dump_calls"), n_dump,
          h.get("w2_dump_sum_abs_a0", float("nan")), sum_dump, h["max_dv"], "holds" if log_ok else "DIFFERENT"))
print("largest difference between the frames and the hand-computed acceleration over all steps: %.3g" % worst)

# the five older readouts equal the wrapper rz bit for bit (curved random plan, no window state set)
r2 = np.random.default_rng(11)
th = np.cumsum(r2.normal(0.0, 0.03, 14))
stp = 0.2 * (8.0 + r2.normal(0.0, 0.6, 14))
pts = np.cumsum(np.stack([stp * np.cos(th), stp * np.sin(th)], -1), 0)
k_prev, knots = pts[0], pts[1:]
same = RV.STATE["exec"] is None
for mode in ("pinned", "backward", "causal", "causal_w", "causal_z"):
    for m in (5, 10):
        a = RZ.counted_frames(T.hermite_frames, mode)(k_prev, knots, m, 0.1)
        b = RV.counted_frames(T.hermite_frames, mode)(k_prev, knots, m, 0.1)
        same = same and all(np.array_equal(x1, x2) for x1, x2 in zip(a, b))
ok_all = ok_all and same
print("pinned, backward, causal, causal_w, causal_z (5 and 10 steps) equal the wrapper rz bit for bit: %s" % same)
try:
    RV.counted_frames(T.hermite_frames, "causal_v")(k_prev, knots, 5, 0.1)
    refuses = False
except RuntimeError:
    refuses = True
ok_all = ok_all and refuses
print("causal_v without a window state refuses to run: %s" % refuses)
print("pinned tool %s | wrapper rv sha256 %s" % (os.path.basename(path), hashlib.sha256(open(RV.__file__, "rb").read()).hexdigest()))
print("ZZ CHECK-TICK-READOUT-RV all hold: %s" % ok_all)
