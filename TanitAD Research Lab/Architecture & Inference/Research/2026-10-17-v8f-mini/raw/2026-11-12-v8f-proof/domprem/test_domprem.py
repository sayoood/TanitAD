"""Unit tests for the pieces of domprem.py that can be wrong silently.  Run: python test_domprem.py"""
import itertools
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import domprem as D  # noqa: E402

np = D.np


def brute_best(P, tol, eps=1e-9):
    """exact maximum by brute force: discs centred on single points, and the two discs of radius tol through every pair of points."""
    P = np.asarray(P, float)
    n = len(P)
    cands = [P[i] for i in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        d = np.linalg.norm(P[j] - P[i])
        if d > 2 * tol + eps:
            continue
        mid = (P[i] + P[j]) / 2
        h = math.sqrt(max(tol * tol - (d / 2) ** 2, 0.0))
        if d < 1e-12:
            continue
        nrm = np.array([-(P[j] - P[i])[1], (P[j] - P[i])[0]]) / d
        cands += [mid + h * nrm, mid - h * nrm]
    return max(int((np.linalg.norm(P - c, axis=1) <= tol + eps).sum()) for c in cands)


class T(unittest.TestCase):
    def test_sweep_against_bruteforce(self):
        rng = np.random.default_rng(1)
        for trial in range(60):
            n = int(rng.integers(1, 26))
            spread = float(rng.choice([1.5, 3.0, 8.0]))
            P = rng.normal(0, spread, (n, 2))
            for tol in (1.0, 2.0, 4.0):
                got, ctr = D.best_disc(P, tol)
                self.assertEqual(got, brute_best(P, tol), (trial, tol, n))
                self.assertEqual(int((np.linalg.norm(P - np.asarray(ctr), axis=1) <= tol + 1e-9).sum()), got)

    def test_sweep_duplicates_and_boundary(self):
        P = np.array([[0, 0], [0, 0], [4.0, 0.0], [10, 10]], float)           # 0 and 4 are exactly 2 tol apart (tol 2): the disc centred at 2 holds both on its boundary
        self.assertEqual(D.best_disc(P, 2.0)[0], 3)
        self.assertEqual(D.best_disc(np.array([[0.0, 0.0], [4.0 + 1e-6, 0.0]]), 2.0)[0], 1)
        self.assertEqual(D.best_disc(np.zeros((0, 2)), 2.0)[0], 0)
        self.assertEqual(D.best_disc(np.array([[1.0, 1.0]]), 2.0)[0], 1)

    def test_d4_synthetic(self):
        two, one = D.synth_d4()
        self.assertEqual(D.best_disc(two, 2.0)[0], 60)
        self.assertEqual(D.best_disc(one, 2.0)[0], 100)
        self.assertEqual(brute_best(two, 2.0), 60)

    def test_wrap_around_arc(self):
        # a configuration whose best arcs wrap past angle 2 pi: points on a ring around the origin, tol 2 -> disc centred at the origin holds all
        t = np.linspace(0, 2 * math.pi, 13)[:-1]
        P = 1.9 * np.stack([np.cos(t), np.sin(t)], 1)
        self.assertEqual(D.best_disc(P, 2.0)[0], 12)
        self.assertEqual(brute_best(P, 2.0), 12)

    def test_context_binning_edges(self):
        v = np.array([0.0, 1.999999, 2.0, 7.999999, 8.0, 14.999999, 15.0, 30.0])
        self.assertEqual(D.ctx_index(v).tolist(), [0, 0, 1, 1, 2, 2, 3, 3])
        m = D.ctx_masks(v)
        self.assertEqual(len(m), 5)
        self.assertEqual(m[0].sum(), 8)
        self.assertEqual([int(x.sum()) for x in m[1:]], [2, 2, 2, 2])

    def test_margin_hand_example(self):
        # 10 windows, logged cells: 6 x cell 3, 3 x cell 5, 1 x cell 7.  arm right on 4 of them -> acc 0.4, p_max 0.6, m = -0.2
        jg = np.array([3] * 6 + [5] * 3 + [7])
        right = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0], float)
        sub = np.arange(10)
        self.assertAlmostEqual(D.maxshare(jg, sub, 12), 0.6)
        self.assertAlmostEqual(D.margin(right, jg, sub, 12), -0.2)
        # p_max is recomputed on the resampled selection: window 9 (cell 7) drawn three times, nothing else from cell 3 beyond one window
        sel = np.array([9, 9, 9, 0, 6])
        self.assertAlmostEqual(D.maxshare(jg, sel, 12), 3 / 5)           # cell 7 now holds 3 of 5
        self.assertAlmostEqual(D.margin(right, jg, sel, 12), (right[sel].mean()) - 0.6)

    def test_hit_rule_at_exactly_tol(self):
        g = np.zeros((3, 30, 2))
        cl = np.zeros((3, 30, 2))
        cl[0, 29] = [2.0, 0.0]               # exactly tol: a hit
        cl[1, 29] = [2.0000001, 0.0]         # just beyond: a miss
        cl[2, 29] = [0.0, -1.0]
        fde = np.linalg.norm(cl[:, 29] - g[:, 29], axis=-1)
        self.assertEqual((fde <= 2.0).tolist(), [True, False, True])

    def test_e_and_separation(self):
        right = np.array([1, 1, 0, 1.0])
        hit = np.array([1, 0, 1, 1.0])
        self.assertAlmostEqual(D.share_hit_among_right(right, hit, np.arange(4)), 2 / 3)
        g6 = np.array([[0, 0], [0, 2], [10, 0], [10, 0], [10, 2]], float)
        cell = np.array([1, 1, 2, 2, 2])
        d, rho, S = D.separation(g6, cell, np.arange(5), 2.0)
        self.assertAlmostEqual(d, math.hypot(10 - 0, (2 / 3) - 1))
        self.assertGreater(rho, 0)
        self.assertAlmostEqual(S, d / (4 + 2 * rho))

    def test_labels(self):
        self.assertEqual(D.label_premise({"point": 0.1, "lo": 0.01, "hi": 0.2}), "PREMISE-HOLDS")
        self.assertEqual(D.label_premise({"point": -0.1, "lo": -0.2, "hi": -0.01}), "PREMISE-FAILS")
        self.assertEqual(D.label_premise({"point": 0.1, "lo": -0.01, "hi": 0.2}), "UNDECIDED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
