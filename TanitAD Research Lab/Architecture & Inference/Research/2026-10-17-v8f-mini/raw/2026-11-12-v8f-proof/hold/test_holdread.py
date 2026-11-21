"""Unit tests for the pieces of holdread.py that can be wrong silently.  Run: python test_holdread.py"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import holdread as HR  # noqa: E402

np = HR.np


class T(unittest.TestCase):
    def test_folds_disjoint_deterministic(self):
        eid = np.repeat(np.arange(100, 129), 3)
        a = HR.fold_assign(eid, 0)
        b = HR.fold_assign(eid, 0)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 29)
        self.assertEqual(sorted(np.bincount(list(a.values())).tolist()), [5, 6, 6, 6, 6])
        self.assertNotEqual(a, HR.fold_assign(eid, 1))
        # the deal is the permutation of the sorted episodes, round-robin
        perm = np.random.default_rng(0).permutation(29)
        self.assertEqual(a[int(100 + perm[7])], 7 % 5)

    def test_crossfit_episode_disjoint(self):
        rng = np.random.default_rng(1)
        n = 120
        eid = np.repeat(np.arange(30), 4)
        g = rng.normal(size=(n, 30, 2))
        cl = rng.normal(size=(n, 30, 2))
        T_ol = rng.uniform(0, 10, n)
        rest = np.ones(n, bool)
        rest[:8] = False
        cf = HR.crossfit(cl, g, T_ol, rest, eid)
        for fit, held in cf["splits"]:
            self.assertEqual(len(fit & held), 0)
        self.assertTrue(np.all(np.isnan(cf["theta_w"][~rest])))
        self.assertTrue(np.all(np.isfinite(cf["theta_w"][rest])))
        cf2 = HR.crossfit(cl, g, T_ol, rest, eid)
        self.assertTrue(np.array_equal(cf["fold_w"], cf2["fold_w"]))

    def test_theta_and_tie(self):
        # windows: T_ol = 1.5 (ade_cl 5, ade_stand 1) and T_ol = 7 (ade_cl 1, ade_stand 5).
        T_ol = np.array([1.5, 7.0])
        a_cl = np.array([5.0, 1.0])
        a_st = np.array([1.0, 5.0])
        th, v = HR.select_theta(T_ol, a_cl, a_st)
        # theta 2 holds only the first window: mean (1+1)/2 = 1.0, the same at theta 3, 4, 6 (7 is not below 6); theta 8 holds both: 3.0
        self.assertEqual(th, 2.0)
        self.assertAlmostEqual(v, 1.0)
        # strict inequality: T_ol equal to theta is not held. T_ol = 2.0 exactly: theta 2 does not hold it, theta 3 does
        th2, _ = HR.select_theta(np.array([2.0]), np.array([5.0]), np.array([1.0]))
        self.assertEqual(th2, 3.0)
        # all-equal values: the smallest grid value
        th3, _ = HR.select_theta(np.array([1.0, 1.0]), np.array([2.0, 2.0]), np.array([2.0, 2.0]))
        self.assertEqual(th3, 0.5)

    def test_replace_rule(self):
        n = 4
        cl = np.ones((n, 30, 2)) * np.arange(1, n + 1)[:, None, None]
        T_ol = np.array([0.5, 0.5, 5.0, 0.5])
        rest = np.array([True, False, True, True])
        out, hold = HR.replace_rule(cl, T_ol, rest, 2.0)
        self.assertEqual(hold.tolist(), [True, False, False, True])
        self.assertTrue(np.all(out[0] == 0.0) and np.all(out[3] == 0.0))
        self.assertTrue(np.array_equal(out[1], cl[1]) and np.array_equal(out[2], cl[2]))       # outside REST and above theta: untouched
        out_i, hold_i = HR.replace_rule(cl, T_ol, rest, 2.0, invert=True)
        self.assertEqual(hold_i.tolist(), [False, False, True, False])
        out_0, h0 = HR.replace_rule(cl, T_ol, rest, 0.0)
        self.assertEqual(float(np.abs(out_0 - cl).max()), 0.0)
        # per-window theta
        out_w, hold_w = HR.replace_rule(cl, T_ol, rest, np.array([0.1, 9.0, 9.0, 0.1]))
        self.assertEqual(hold_w.tolist(), [False, False, True, False])

    def test_auroc(self):
        s = np.array([1.0, 2.0, 2.0, 3.0])
        lab = np.array([False, False, True, True])
        # pos {2,3} vs neg {1,2}: pairs 2>1, 2=2 (half), 3>1, 3>2 -> (1 + .5 + 1 + 1)/4
        self.assertAlmostEqual(HR.auroc(s, lab), 3.5 / 4)
        self.assertEqual(HR.auroc(np.array([1.0, 2.0]), np.array([True, False])), 0.0)
        self.assertEqual(HR.auroc(np.array([1.0, 1.0]), np.array([True, False])), 0.5)
        self.assertTrue(math.isnan(HR.auroc(np.array([1.0]), np.array([True]))))

    def test_cross_track(self):
        n = 3
        g = np.zeros((n, 30, 2))
        x = np.zeros((n, 30, 2))
        # window 0: logged along +x at 1 m per step; endpoint error (0, 0.5) -> 0.5
        g[0, :, 0] = np.arange(1, 31)
        x[0] = g[0]
        x[0, 29, 1] += 0.5
        # window 1: logged at 45 degrees, 1 m per step; endpoint error (1, 0): the orthogonal component is sin 45
        g[1, :, 0] = np.arange(1, 31) / math.sqrt(2)
        g[1, :, 1] = np.arange(1, 31) / math.sqrt(2)
        x[1] = g[1]
        x[1, 29, 0] += 1.0
        # window 2: the logged ego stays (no valid step): ego-frame lateral axis |y|; error (3, -0.7) -> 0.7
        x[2, 29] = [3.0, -0.7]
        ct, fall = HR.cross_track(x, g)
        self.assertAlmostEqual(ct[0], 0.5)
        self.assertAlmostEqual(ct[1], math.sin(math.pi / 4))
        self.assertAlmostEqual(ct[2], 0.7)
        self.assertEqual(fall.tolist(), [False, False, True])
        # error along the heading contributes nothing
        x[0, 29] = g[0, 29] + [2.0, 0.0]
        self.assertAlmostEqual(HR.cross_track(x, g)[0][0], 0.0)

    def test_hit_and_ade(self):
        g = np.zeros((2, 30, 2))
        x = np.zeros((2, 30, 2))
        x[0, 29, 0] = 2.0
        x[1, 29, 0] = 2.0001
        self.assertEqual(HR.endpoint_hit(x, g).tolist(), [1.0, 0.0])
        self.assertAlmostEqual(HR.ade6(x, g)[0], 2.0 / 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
