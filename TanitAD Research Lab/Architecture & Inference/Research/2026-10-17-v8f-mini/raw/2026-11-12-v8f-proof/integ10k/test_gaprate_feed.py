"""test_gaprate_feed.py -- pure tests of the two new functions of v8f_t1_eval_reader_rule_d.py (card PREREG_V8F_PROOF_ADD50), no model, no GPU. Run on the pod:
  DISTRULE_DIR=<build> PYTHONPATH=<tree>/stack python test_gaprate_feed.py
Each case has an arm that must fail (a wrong gather, a wrong sign, a leak of a value where no read exists). Last line: ZZ TEST-GAPRATE-FEED PASS n/n or FAIL."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v8f_t1_eval_reader_rule_d as D  # noqa: E402


class FakeReader:
    """p = index of the FIRST frame of the 5 / 1000, gap = index of the LAST frame (the tokens carry their frame index)."""
    hist = 5

    def read(self, tokens):
        t = np.asarray(tokens, np.float32)
        assert t.ndim == 4 and t.shape[1] == 5, t.shape
        return (t[:, 0, 0, 0] / 1000.0).astype(np.float32), t[:, -1, 0, 0].astype(np.float32)


class St:
    def __init__(self, j, T=80):
        self.j = j
        self.d = {"feats": (np.arange(T, dtype=np.float16)[:, None, None] * np.ones((1, 2, 3), np.float16))}


res = []


def case(name, cond):
    res.append(bool(cond))
    print("   %-58s %s" % (name, "PASS" if cond else "FAIL"))


rd = FakeReader()
sts = [St(3), St(4), St(28), St(29), St(45), St(79)]
p, g, ok = D.prev_read(sts, rd, 25)
case("prev_read K=25: history flag (j-29 >= 0)", list(ok) == [False, False, False, True, True, True])
case("prev_read K=25: gap = frame j-25, p = frame j-29", np.allclose(g[3:], [4, 20, 54]) and np.allclose(p[3:], [0.0, 0.016, 0.050], atol=1e-6))
case("prev_read K=25: NaN where no history", bool(np.all(~np.isfinite(g[:3])) and np.all(~np.isfinite(p[:3]))))
p0, g0, ok0 = D.prev_read(sts, rd, 0)
case("prev_read K=0: the read at the tick (frames j-4 .. j)", list(ok0) == [False, True, True, True, True, True] and np.allclose(g0[1:], [4, 28, 29, 45, 79]))
pm, gm, okm = D.prev_read(sts, rd, 26)
case("regression arm: a gather one step off is seen", not np.allclose(gm[3:], g[3:], equal_nan=True))
case("prev_read: a step beyond the clip is no read", not D.prev_read([St(90, T=80)], rd, 0)[2][0])

present = np.array([True, True, True, False, True, True])
gnow = np.array([10.0, 12.0, 30.0, 20.0, 15.0, np.nan])
pprev = np.array([0.9, 0.4, 0.9, 0.9, 0.9, 0.9])
gprev = np.array([20.0, 20.0, np.nan, 20.0, 5.0, 20.0])
okp = np.array([True, True, True, True, False, True])
have, c = D.gap_rate_closing(present, gnow, pprev, gprev, okp, 0.5, 25)
case("gap_rate: have only where present at both ends and finite", list(have) == [True, False, False, False, False, False])
case("gap_rate: value = slope * (g - g_prev) / (0.2 K), closing in is negative", abs(float(c[0]) - 0.5 * (10.0 - 20.0) / 5.0) < 1e-6 and float(c[0]) < 0)
case("gap_rate: exactly 0 where no read", bool(np.all(c[1:] == 0.0)) and c.dtype == np.float32)
h2, c2 = D.gap_rate_closing(present, gnow, pprev, gprev, okp, 0.5, 5)
case("gap_rate: K enters the rate", abs(float(c2[0]) - 0.5 * (10.0 - 20.0) / 1.0) < 1e-6)
case("regression arm: the reversed sign is seen", abs(float(c[0]) - 0.5 * (20.0 - 10.0) / 5.0) > 1e-3)
case("forms: RRd is a form, the parent forms stay", D.FORMS[-1] == "RRd" and D.FORMS[:8] == ("R0", "RLb", "RR", "RRid", "RR0", "RLc", "RRc", "RRh"))
u_d = D.feed("RRd", np.array([0.9]), np.array([30.0]), np.array([True]), np.array([0.0, 60.0]), np.array([0.0, 30.0]))
u_r = D.feed("RR", np.array([0.9]), np.array([30.0]), np.array([True]), np.array([0.0, 60.0]), np.array([0.0, 30.0]))
case("feed: presence and gap of RRd equal those of RR", bool(u_d[0][0] == u_r[0][0]) and float(u_d[1][0]) == float(u_r[1][0]) == 15.0)
print("ZZ TEST-GAPRATE-FEED %s %d/%d" % ("PASS" if all(res) else "FAIL", sum(res), len(res)))
sys.exit(0 if all(res) else 1)
