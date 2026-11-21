"""Synthetic CPU tests of pg_geom: analytic clearance / frac_out / key-time indexing / min distance / lead gap / closing / TTC / padding, and RED mutations of the geometry."""
import math
import os
import sys
import unittest

import numpy as np
import shapely
from shapely.geometry import Polygon, box

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pg_geom as G                                                                       # noqa: E402

L_EGO, W_EGO, REAR_TO_CENTER = 4.049 + 1.127, 2.297, (4.049 - 1.127) / 2.0


def ego_states(xs, ys, hs, speed):
    """rear-axle x, y, heading arrays [T] -> states [1, T, 11] and coords [1, T, 5, 2] (independent of navsim's helper)."""
    T = len(xs)
    st = np.zeros((1, T, 11))
    st[0, :, 0], st[0, :, 1], st[0, :, 2] = xs, ys, hs
    st[0, :, 3] = speed
    cx, cy = xs + REAR_TO_CENTER * np.cos(hs), ys + REAR_TO_CENTER * np.sin(hs)
    co = np.zeros((1, T, 5, 2))
    hl, hw = L_EGO / 2, W_EGO / 2
    for i, (a, b) in {G.FL: (hl, hw), G.RL: (-hl, hw), G.RR: (-hl, -hw), G.FR: (hl, -hw), G.CTR: (0, 0)}.items():
        co[0, :, i, 0] = cx + a * np.cos(hs) - b * np.sin(hs)
        co[0, :, i, 1] = cy + a * np.sin(hs) + b * np.cos(hs)
    return st, co


def straight(x0, speed, T=41, y=0.0):
    t = np.arange(T) * 0.1
    return ego_states(x0 + speed * t, np.full(T, y), np.zeros(T), np.full(T, speed))


class TestDac(unittest.TestCase):
    def setUp(self):
        self.drv = G.Drivable([box(0, -5, 20, 5)])

    def test_clearance_inside(self):
        st, co = straight(x0=3.0, speed=4.0)                       # rear axle 3 -> 19; rear corners at x - 1.127 .. ; front corner at x_rear + 4.049
        out = G.dac_scalars(co, self.drv)
        rear_min = 3.0 - 1.127 - 0.0                               # rear bumper x at t=0
        front_max = 3.0 + 4.0 * 4.0 + 4.049                        # = 23.049 > 20 -> exits through the far edge
        self.assertLess(front_max, 20.0 + 5)
        self.assertEqual(out[0, 1], (np.array([4.049 + 3.0 + 4.0 * 0.1 * t for t in range(41)]) >= 20.0).mean())   # frac_out: front corners beyond x=20
        # a step with the front corners beyond the far edge has negative clearance
        self.assertLess(out[0, 0], 0.0)

    def test_clearance_values_all_inside(self):
        st, co = straight(x0=3.0, speed=1.0)                       # front corner reaches 3 + 4 + 4.049 = 11.049 < 20; rear bumper at 1.873 at t=0
        out = G.dac_scalars(co, self.drv)
        expect_min = min(3.0 - 1.127 - 0.0, 5.0 - W_EGO / 2, 20.0 - (3.0 + 1.0 * 4.0 + 4.049))      # rear edge x=0, side edge y=+-5, far edge
        self.assertAlmostEqual(out[0, 0], min(expect_min, 5.0 - W_EGO / 2), places=9)
        self.assertEqual(out[0, 1], 0.0)
        # key times: step 0 clearance = min(rear distance 1.873, side 3.8515, front 20-7.049)
        self.assertAlmostEqual(out[0, 2], min(3.0 - 1.127, 5.0 - W_EGO / 2, 20.0 - 7.049), places=9)
        # step 40 (index 2 + 8 = 10): rear axle 7, front corner at 7 + 4.049 = 11.049 -> 8.951 to the far edge; rear bumper at 7 - 1.127 = 5.873
        self.assertAlmostEqual(out[0, 10], min(5.873, 5.0 - W_EGO / 2, 20.0 - 11.049), places=9)

    def test_outside_is_negative_and_frac(self):
        T = 41
        ys = np.where(np.arange(T) >= 30, 6.0, 0.0)                # the last 11 steps drift to y = 6 (centre outside y <= 5)
        st, co = ego_states(np.full(T, 8.0), ys, np.zeros(T), np.zeros(T))
        out = G.dac_scalars(co, self.drv)
        self.assertAlmostEqual(out[0, 1], 11 / 41)
        self.assertLess(out[0, 0], 0.0)
        self.assertAlmostEqual(out[0, 0], -(6.0 + W_EGO / 2 - 5.0), places=9)       # the left corners are 2.1485 outside
        self.assertGreater(out[0, 2], 0.0)                         # step 0 is inside
        self.assertLess(out[0, 2 + 8], 0.0)                        # step 40 is outside

    def test_cap_and_empty(self):
        st, co = straight(x0=8.0, speed=0.0)
        big = G.Drivable([box(-100, -100, 100, 100)])
        out = G.dac_scalars(co, big)
        self.assertAlmostEqual(out[0, 0], G.CAP_CLEAR)             # nothing within the cap
        self.assertEqual(out[0, 1], 0.0)
        out = G.dac_scalars(co, G.Drivable([]))
        self.assertAlmostEqual(out[0, 0], -G.CAP_CLEAR)
        self.assertEqual(out[0, 1], 1.0)

    def test_sign_is_the_label(self):
        """the registered tautology: min clearance < 0 iff frac_out > 0 (a rectangle has no seams)."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            T = 41
            ys = rng.uniform(-6, 6) + np.cumsum(rng.normal(0, 0.1, T))
            xs = rng.uniform(3, 15) + np.cumsum(rng.normal(0, 0.1, T))
            st, co = ego_states(xs, ys, rng.uniform(-0.4, 0.4) * np.ones(T), np.zeros(T))
            out = G.dac_scalars(co, self.drv)
            self.assertEqual(bool(out[0, 0] < 0.0), bool(out[0, 1] > 0.0))

    def test_unsigned_drops_label_column(self):
        g = np.arange(G.K_DAC, dtype=float)[None, :]
        u = G.unsigned(g)
        self.assertEqual(u.shape[-1], G.K_DAC - 1)
        self.assertNotIn(1.0, u.tolist()[0][:1])                   # column 1 (frac_out) is gone: u[0] = |g0|, |g2..|
        self.assertEqual(u[0].tolist(), [0.0] + [float(i) for i in range(2, G.K_DAC)])

    def test_key_steps(self):
        self.assertEqual(G.KEY_STEPS, tuple(range(0, 41, 5)))
        self.assertEqual(G.K_DAC, 11)
        self.assertEqual(G.K_NC, 4)


def static_slots(polys, n=21, vel=None):
    return [G.Slot(polys, vel) for _ in range(n)]


class TestNc(unittest.TestCase):
    def test_static_box_ahead(self):
        T = 41
        st, co = straight(x0=0.0, speed=5.0)                       # front bumper = 4.049 + 0.5 t
        obstacle = box(30.0, -1.0, 34.0, 1.0)
        out = G.nc_scalars(co, st, static_slots([obstacle]))
        gap40 = 30.0 - (4.049 + 20.0)
        self.assertAlmostEqual(out[0, 0], gap40, places=9)                              # min distance at the last step
        self.assertAlmostEqual(out[0, 1], gap40 / 5.0, places=9)                         # min time gap
        self.assertAlmostEqual(out[0, 2], 5.0)                                           # closing speed = ego speed (static lead)
        self.assertAlmostEqual(out[0, 3], gap40 / 5.0 - 1.0, places=9)                   # TTC margin

    def test_moving_lead(self):
        st, co = straight(x0=0.0, speed=5.0)
        # a box that moves at 2 m/s: slot k (0.2 s) at x = 30 + 0.4 k
        slots = [G.Slot([box(30.0 + 0.4 * k, -1.0, 34.0 + 0.4 * k, 1.0)], [[2.0, 0.0]]) for k in range(21)]
        out = G.nc_scalars(co, st, slots)
        self.assertAlmostEqual(out[0, 2], 3.0)                                           # 5 - 2
        gap40 = 30.0 + 0.4 * 20 - (4.049 + 20.0)
        self.assertAlmostEqual(out[0, 0], gap40, places=9)

    def test_no_agents_and_off_corridor(self):
        st, co = straight(x0=0.0, speed=5.0)
        out = G.nc_scalars(co, st, static_slots([]))
        self.assertEqual(out[0].tolist(), [G.D_CAP, G.T_CAP, 0.0, G.T_CAP - 1.0])
        side = box(10.0, 6.0, 14.0, 8.0)                                                  # 5 m to the side: outside the corridor
        out = G.nc_scalars(co, st, static_slots([side]))
        self.assertAlmostEqual(out[0, 0], 6.0 - W_EGO / 2, places=9)                     # distance sees it
        self.assertEqual(out[0, 1], G.T_CAP)                                              # no lead
        self.assertEqual(out[0, 2], 0.0)

    def test_overlap_is_contact(self):
        st, co = straight(x0=0.0, speed=5.0)
        out = G.nc_scalars(co, st, static_slots([box(10.0, -1.0, 14.0, 1.0)]))           # ego reaches x = 10 - ... : front 4.049 + 0.5 t >= 10 at t = 12 -> overlap
        self.assertEqual(out[0, 0], 0.0)
        self.assertAlmostEqual(out[0, 3], -1.0)                                           # TTC 0 -> margin -1
        self.assertEqual(out[0, 1], 0.0)

    def test_stopped_ego_time_gap_uses_floor(self):
        st, co = straight(x0=0.0, speed=0.0)
        out = G.nc_scalars(co, st, static_slots([box(20.0, -1.0, 24.0, 1.0)]))
        gap = 20.0 - 4.049
        self.assertAlmostEqual(out[0, 1], min(10.0, gap / 0.5), places=9)
        self.assertEqual(out[0, 2], 0.0)                                                  # ego stopped: closing 0
        self.assertEqual(out[0, 3], G.T_CAP - 1.0)                                         # closing 0 -> TTC cap

    def test_floored(self):
        g = np.array([[0.0, 0.1, 3.0, -1.0], [4.0, 3.0, 1.0, 0.5]])
        f = G.floored(g)
        self.assertEqual(f[0].tolist(), [0.5, 0.5, 3.0, 0.0])
        self.assertEqual(f[1].tolist(), [4.0, 3.0, 1.0, 0.5])

    def test_build_slots_velocity(self):
        maps = [(["a", "red_light_1"], [box(0, 0, 4, 2), box(50, 50, 51, 51)]), (["a"], [box(0.4, 0, 4.4, 2)]), (["a"], [box(0.8, 0, 4.8, 2)])]
        sl = G.build_slots(maps)
        self.assertEqual([s.n for s in sl], [1, 1, 1])                                    # the red-light box is excluded
        self.assertAlmostEqual(sl[0].vels[0, 0], 2.0)
        self.assertAlmostEqual(sl[2].vels[0, 0], 2.0)                                     # backward difference at the last slot


class TestMutations(unittest.TestCase):
    """RED: the analytic tests above must FAIL when the geometry is wrong (a test that passes on a mutated module tests nothing)."""

    def _run(self):
        s = unittest.TestSuite()
        for cls in (TestDac, TestNc):
            s.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
        r = unittest.TestResult()
        s.run(r)
        return len(r.failures) + len(r.errors)

    def test_sign_flip_is_red(self):
        orig = G.signed_clearance
        G.signed_clearance = lambda drv, pts, cap=G.CAP_CLEAR: (lambda sc_in: (-sc_in[0], sc_in[1]))(orig(drv, pts, cap))
        try:
            self.assertGreater(self._run(), 0)
        finally:
            G.signed_clearance = orig

    def test_key_steps_shift_is_red(self):
        orig = G.KEY_STEPS
        G.KEY_STEPS = tuple(s + 1 if s < 40 else 40 for s in orig)
        try:
            self.assertGreater(self._run(), 0)
        finally:
            G.KEY_STEPS = orig

    def test_front_offset_is_red(self):
        orig = G.FRONT_FROM_REAR
        G.FRONT_FROM_REAR = 3.0
        try:
            self.assertGreater(self._run(), 0)
        finally:
            G.FRONT_FROM_REAR = orig

    def test_vmin_is_red(self):
        orig = G.V_MIN
        G.V_MIN = 2.0
        try:
            self.assertGreater(self._run(), 0)
        finally:
            G.V_MIN = orig

    def test_green_baseline(self):
        self.assertEqual(self._run(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
