"""Unit tests for hierwhere.py on synthetic trajectories (no dump is read).  Run: python test_hierwhere.py   (CPU only)."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import hierwhere as H  # noqa: E402

DT = 0.2


def traj(speeds, yaw_deg_total=0.0):
    """speeds [30] m/s on the 0.2 s grid; the heading turns linearly by yaw_deg_total over the 30 steps (starting at 0)."""
    sp = np.asarray(speeds, float)
    th = np.radians(np.linspace(0.0, yaw_deg_total, 30))
    d = np.stack([sp * DT * np.cos(th), sp * DT * np.sin(th)], -1)
    return np.cumsum(d, axis=0)


def const(v):
    return np.full(30, float(v))


def ramp(v0, v1):
    return np.linspace(v0, v1, 30)


class TestModes(unittest.TestCase):
    def mode(self, sp, yaw=0.0, scale=1.0):
        lon, lat = H.modes(traj(sp, yaw)[None], scale)
        return int(lon[0]), int(lat[0])

    def test_twelve_cells(self):
        # longitudinal exemplars: STOP 0, DECEL 1, ACCEL 2, KEEP 3 ; lateral: LEFT 0, RIGHT 1, STRAIGHT 2
        lon_sp = {0: np.concatenate([np.linspace(8, 0.0, 15), const(0.0)[:15]]), 1: ramp(8, 4), 2: ramp(4, 8), 3: const(6)}
        lat_yaw = {0: 40.0, 1: -40.0, 2: 0.0}
        seen = set()
        for i, sp in lon_sp.items():
            for j, yw in lat_yaw.items():
                lo, la = self.mode(sp, yw)
                self.assertEqual((lo, la), (i, j), (i, j))
                seen.add(H.joint(lo, la))
        self.assertEqual(len(seen), 12)
        self.assertEqual(H.KEEP_STRAIGHT, H.joint(3, 2))

    def test_stop_edges(self):
        base = np.concatenate([ramp(8, 1.0)[:20], const(1.0)[:10]])
        for last, want in ((0.4999, 0), (0.5001, 1)):          # speed in the last interval just below / above 0.5 m/s: strict "below"
            sp = base.copy()
            sp[29] = last
            sp[20:29] = 0.2 if want == 0 else 1.0
            lo, _ = self.mode(sp)
            self.assertEqual(lo, 0 if want == 0 else 1, (last, lo))
        # already at rest the whole 6 s: the card's literal definition calls it STOP
        self.assertEqual(self.mode(const(0.0))[0], 0)
        # touches rest inside [2, 6] and moves again at the end: not STOP
        sp = const(5.0)
        sp[12:16] = 0.1
        self.assertNotEqual(self.mode(sp)[0], 0)

    def test_decel_accel_edges_kin_level(self):
        n = 6
        sp = np.full((n, 30), 5.0)
        dv = np.array([-2.0000001, -2.0, -1.9999999, 1.9999999, 2.0, 2.0000001])
        sp[:, 0] = 5.0
        sp[:, 29] = 5.0 + dv
        lon, _ = H.classify(sp, np.zeros(n))
        self.assertEqual(lon.tolist(), [1, 1, 3, 3, 2, 2])         # DECEL if <= -2.0, ACCEL if >= +2.0

    def test_heading_edges_kin_level(self):
        sp = np.full((6, 30), 5.0)
        dh = np.array([9.9999, 10.0, 10.0001, -9.9999, -10.0, -10.0001])
        _, lat = H.classify(sp, dh)
        self.assertEqual(lat.tolist(), [2, 0, 0, 2, 1, 1])

    def test_scale(self):
        sp = np.full((2, 30), 5.0)
        sp[:, 29] = 5.0 - 1.6                                       # dv = -1.6: DECEL only at x0.75 (threshold 1.5)
        lon1, _ = H.classify(sp, np.zeros(2), 1.0)
        lon2, _ = H.classify(sp, np.zeros(2), 0.75)
        self.assertEqual((lon1[0], lon2[0]), (3, 1))
        _, la = H.classify(sp, np.array([8.0, 12.0]), 1.25)         # 10 x 1.25 = 12.5: neither LEFT
        self.assertEqual(la.tolist(), [2, 2])

    def test_kinematics_recovers_speed_and_turn(self):
        sp, dh = H.kinematics(traj(ramp(3, 9), 35.0)[None])
        self.assertTrue(np.allclose(sp[0], ramp(3, 9)))
        self.assertAlmostEqual(float(dh[0]), 35.0, places=6)


class TestDecomposition(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.n = 600
        self.eid = np.repeat(np.arange(30), 20)
        self.eD, self.eF = rng.gamma(2, 1.0, self.n), rng.gamma(2, 1.0, self.n)
        self.cD, self.cF = rng.random(self.n) < 0.7, rng.random(self.n) < 0.6

    def test_h2_identity_exact(self):
        rows, ident = H.r3_rows(self.eid, self.eD, self.eF, self.cD, self.cF, 50)
        self.assertLessEqual(ident, 1e-9)
        s = rows["WITHIN"]["point"] + rows["SELECTION"]["point"] + rows["BOTH-MISS"]["point"]
        self.assertAlmostEqual(s, rows["D"]["point"], places=5)
        self.assertAlmostEqual(rows["D"]["point"], float((self.eD - self.eF).mean()), places=5)

    def test_partition_counts(self):
        _, sets = H.decompose(self.eD, self.eF, self.cD, self.cF)
        tot = sum(m.astype(int) for m in sets.values())
        self.assertTrue((tot == 1).all())

    def test_h4_label_swap_centred_on_zero(self):
        # arms generated from one distribution, the logged-mode hit independent of the arm: swapped terms must straddle 0 on average
        pts = []
        for sd in range(40):
            eD, eF, cD, cF, nsw = H.swap_arms(self.eD, self.eF, self.cD, self.cF, sd)
            self.assertTrue(200 < nsw < 400)
            rows, ident = H.r3_rows(self.eid, eD, eF, cD, cF, 30)
            self.assertLessEqual(ident, 1e-9)
            pts.append((rows["SELECTION"]["point"], rows["WITHIN"]["point"]))
        pts = np.array(pts)
        # mean over swaps is ~0 (sd of one swapped term ~ 0.1/sqrt(40) ~ 0.02)
        self.assertLess(abs(pts[:, 0].mean()), 0.06)
        self.assertLess(abs(pts[:, 1].mean()), 0.06)

    def test_swap_flips_sign_of_difference_only(self):
        eD, eF, cD, cF, _ = H.swap_arms(self.eD, self.eF, self.cD, self.cF, 3)
        self.assertAlmostEqual(float(np.abs((eD - eF)).sum()), float(np.abs(self.eD - self.eF).sum()), places=9)
        self.assertEqual(int((cD ^ cF).sum()), int((self.cD ^ self.cF).sum()))


class TestBootstrap(unittest.TestCase):
    def test_matches_taniteval_paired(self):
        rng = np.random.default_rng(0)
        eid = np.repeat(np.arange(25), 8)
        a, b = rng.normal(1.0, 1, 200), rng.normal(0.9, 1, 200)
        pb = H.CI.paired_episode_cluster_bootstrap(a, b, eid, n_boot=300, seed=0)
        mine = H.boot(eid, {"d": H.mean_over(a - b)}, 300)["d"]
        self.assertAlmostEqual(mine["point"], pb["delta"], places=3)
        self.assertAlmostEqual(mine["lo"], pb["lo"], places=3)
        self.assertAlmostEqual(mine["hi"], pb["hi"], places=3)

    def test_clustering_resamples_episodes_not_windows(self):
        # one episode carries all the signal: with episode resampling the CI must be wide, and identical draws across statistics
        eid = np.repeat(np.arange(10), 10)
        v = np.zeros(100)
        v[:10] = 1.0
        r = H.boot(eid, {"a": H.mean_over(v), "b": H.mean_over(v * 2)}, 400)
        self.assertEqual(r["a"]["lo"], 0.0)
        self.assertAlmostEqual(r["b"]["hi"], 2 * r["a"]["hi"], places=12)     # same draws for every statistic
        self.assertGreater(r["a"]["hi"], 0.15)

    def test_mean_over_masks_nan(self):
        f = H.mean_over(np.array([1.0, np.nan, 3.0, 5.0]), np.array([True, True, True, False]))
        self.assertEqual(f(np.arange(4)), 2.0)

    def test_outcome_names(self):
        def mk(w, s, wl=None, wh=None, sl=None, sh=None):
            return {"WITHIN": {"point": w, "lo": w - 0.01 if wl is None else wl, "hi": w + 0.01 if wh is None else wh},
                    "SELECTION": {"point": s, "lo": s - 0.01 if sl is None else sl, "hi": s + 0.01 if sh is None else sh}}
        self.assertEqual(H.outcome(mk(-0.02, -0.1), mk(-0.01, -0.02))[0], "SELECTION-CARRIES")
        self.assertEqual(H.outcome(mk(-0.1, -0.02), mk(-0.01, -0.02))[0], "WITHIN-CARRIES")
        self.assertEqual(H.outcome(mk(-0.02, -0.1), mk(-0.05, 0.02))[0], "MIXED")           # s1 SELECTION > WITHIN
        self.assertEqual(H.outcome(mk(-0.02, -0.1, sl=-0.2, sh=0.01), mk(-0.01, -0.02))[0], "MIXED")   # interval includes 0


if __name__ == "__main__":
    unittest.main(verbosity=2)
