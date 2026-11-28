"""Synthetic-only tests for the S0-B scripts.  `python test_s0b.py` runs them; `python test_s0b.py --mutations` copies the scripts
to a temp dir, applies each mutation and REQUIRES the suite to fail (RED)."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import s0b_raster as RS  # noqa: E402
import s0b_readout as RO  # noqa: E402

SC = RO.load_sel_common(os.environ.get("S0B_SEL_DIR", str(HERE / "ref")))
FP = RO.FootParams(5.176, 2.297, 1.461, "synthetic test value")


def synth(S=40, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(S, 64, 7)).astype(np.float32)
    pdm = rng.random((S, 64, 7))
    pdm[..., 0] = rng.choice([0.0, 0.5, 1.0], size=(S, 64))
    pdm[..., 3] = rng.choice([0.0, 1.0], size=(S, 64))
    rule = SC.agg_torch(logits, lane=False)
    return logits, pdm, rule, np.argmax(rule, axis=-1)


def straight_props(P=64, v=8.0):
    t = np.arange(1, 21) * 0.2
    xy = np.zeros((P, 20, 2))
    for p in range(P):
        xy[p, :, 0] = v * t * (0.5 + p / 63.0)
        xy[p, :, 1] = 0.02 * p * t
    return xy, np.zeros((P, 20))


class T(unittest.TestCase):
    def test_global_to_ego_known_pose(self):
        x, y, h = RS.global_to_ego(10.0, 8.0, np.pi / 2 + 0.3, 10.0, 5.0, np.pi / 2)
        self.assertAlmostEqual(float(x), 3.0); self.assertAlmostEqual(float(y), 0.0); self.assertAlmostEqual(float(h), 0.3)
        x, y, _ = RS.global_to_ego(7.0, 5.0, 0.0, 10.0, 5.0, np.pi / 2)       # 3 m behind-left in global -> ego y = +3
        self.assertAlmostEqual(float(x), 0.0); self.assertAlmostEqual(float(y), 3.0)

    def test_quat_yaw(self):
        q = (np.cos(0.4), 0.0, 0.0, np.sin(0.4))
        self.assertAlmostEqual(float(RS.quat_yaw(*q)), 0.8)

    def test_footprint_cells_known_pose(self):
        r, c, v, _ = RS.box_cells(10.0, 0.0, 0.0, 2.0, 1.0)
        rr, cc = r[v], c[v]
        self.assertEqual(int(v.sum()), 32)                                     # x centres 8.25..11.75 (8) * y -0.75..0.75 (4)
        self.assertEqual((rr.min(), rr.max()), (32, 39)); self.assertEqual((cc.min(), cc.max()), (62, 65))
        r, c, v, _ = RS.box_cells(10.0, 0.0, np.pi / 2, 2.0, 1.0)              # rotated: 4 rows x 8 cols
        self.assertEqual(int(v.sum()), 32); self.assertEqual(int(np.ptp(r[v])) + 1, 4); self.assertEqual(int(np.ptp(c[v])) + 1, 8)

    def test_out_of_grid_is_free_and_partial(self):
        r, c, v, partial = RS.box_cells(71.5, 0.0, 0.0, 2.0, 1.0)              # right edge past x=72
        self.assertTrue(bool(partial[0])); self.assertLess(int(v.sum()), 32)
        self.assertEqual(int(RS.box_cells(200.0, 0.0, 0.0, 2.0, 1.0)[2].sum()), 0)

    def test_interpolation(self):
        rx = np.zeros((3, 11)); ry = np.zeros((3, 11)); ryaw = np.zeros((3, 11)); pr = np.zeros((3, 11), bool)
        rx[0, :] = np.arange(11) * 5.0; pr[0, :] = True                        # moving 10 m/s, all samples
        rx[1, 2] = 7.0; ry[1, 2] = 1.0; pr[1, 2] = True                        # one sample -> present at all steps
        rx[2, 0:4] = [0, 1, 2, 3]; pr[2, 0:4] = True                           # last sample at 1.5 s
        x, y, yaw, p = RS.interpolate_tracks(rx, ry, ryaw, pr)
        self.assertTrue(np.allclose(x[0], RS.STEP_T * 10.0)); self.assertTrue(p[0].all())
        self.assertTrue(p[1].all()); self.assertTrue((x[1] == 7.0).all())
        self.assertTrue(p[2, :16].all())                                       # last sample 1.5 s = step 15 (float-exact: 15*0.1 == 1.5)
        self.assertFalse(p[2, 16:].any())                                      # no extrapolation, as NAVSIM
        ry2 = np.zeros((1, 11)); yw = np.zeros((1, 11)); yw[0, :2] = [3.0, -3.0]; pp = np.zeros((1, 11), bool); pp[0, :2] = True
        _, _, yaw, _ = RS.interpolate_tracks(np.zeros((1, 11)), ry2, yw, pp)
        self.assertLess(abs(yaw[0, 2] - (3.0 + 0.4 * ((-3.0 + 2 * np.pi) - 3.0))), 1e-9)   # unwrapped

    def test_nc_proxy_uses_time_index(self):
        xy, yw = straight_props()
        poses = RO.proposals_10hz(xy, yw)
        ft = RO.footprint_tables(poses, FP)
        Rg = RS.reg_raster("zero")
        # an obstacle that exists ONLY at step 0.. and in a cell where proposal 63 will be at t=3.0 s only
        cx = float(poses[63, 30, 0] + FP.rear_axle_to_center); cy = float(poses[63, 30, 1])
        r, c, v, _ = RS.box_cells(cx, cy, 0.0, 0.3, 0.3)
        cell = int((r * RS.NY + c)[v][0])
        Rg[30, cell] = True
        nc, ttc, _ = RO.flags_for_raster(Rg, ft)
        self.assertTrue(nc[63])
        Rg2 = RS.reg_raster("zero"); Rg2[5, cell] = True                       # same cell, WRONG time -> no NC for 63
        nc2, _, _ = RO.flags_for_raster(Rg2, ft)
        self.assertFalse(nc2[63])

    def test_ttc_proxy_looks_ahead(self):
        t = np.arange(1, 21) * 0.2
        xy = np.zeros((64, 20, 2)); xy[..., 0] = 10.0 * np.minimum(t, 1.0)[None, :]      # 10 m/s for 1 s, then dead stop
        poses = RO.proposals_10hz(xy, np.zeros((64, 20)))
        ft = RO.footprint_tables(poses, FP)
        Rg = RS.reg_raster("zero")
        sp = float(ft.speed[0, 10]); self.assertAlmostEqual(sp, 10.0, places=6)
        cx = float(poses[0, 10, 0] + FP.rear_axle_to_center + FP.length / 2 + 0.6 * sp - 0.2)
        r, c, v, _ = RS.box_cells(cx, 0.0, 0.0, 0.3, 0.3)
        Rg[16, int((r * RS.NY + c)[v][0])] = True                              # step 10 + 6; the stopped ego never reaches it
        nc, ttc, _ = RO.flags_for_raster(Rg, ft)
        self.assertTrue(ttc[0]); self.assertFalse(nc[0])

    def test_creg_zero_and_one_are_exactly_zero(self):
        for seed in range(3):
            logits, pdm, rule, p6 = synth(40, seed)
            xy, yw = straight_props()
            ft = RO.footprint_tables(RO.proposals_10hz(xy, yw), FP)
            for kind in ("zero", "one"):
                nc, ttc, _ = RO.flags_for_raster(RS.reg_raster(kind), ft)
                self.assertEqual(bool(nc.all()), kind == "one")
                nc_s = np.tile(nc, (40, 1)); ttc_s = np.tile(ttc, (40, 1))
                g = RO.set_gains(rule, pdm[..., 6], p6, nc_s, ttc_s)
                self.assertEqual(float(g["both"].mean()), 0.0)
                self.assertEqual(bool(g["all_flagged"].all()), kind == "one")

    def test_fallback_when_all_flagged(self):
        logits, pdm, rule, p6 = synth(10, 1)
        allf = np.ones((10, 64), bool)
        pk, a = RO.veto_pick(rule, allf, p6)
        self.assertTrue((pk == p6).all() and a.all())
        some = np.zeros((10, 64), bool); some[np.arange(10), p6] = True
        pk2, a2 = RO.veto_pick(rule, some, p6)
        self.assertTrue((pk2 != p6).all() and not a2.any())

    def test_oracle_matches_independent_formula(self):
        logits, pdm, rule, p6 = synth(60, 2)
        nc, ttc, pd = pdm[..., 0], pdm[..., 3], pdm[..., 6]
        o = RO.oracle_gains(rule, pd, nc, ttc, p6)["both"]
        a = np.arange(60); ok = (nc > 0) & (ttc > 0)
        rr = np.where(ok, rule, -1.0); pk = np.where(ok.any(1), rr.argmax(1), p6)
        self.assertAlmostEqual(float(o.mean()), float(100 * (pd[a, pk] - pd[a, p6]).mean()), places=9)

    def test_derangement(self):
        logs = np.array(["a"] * 5 + ["b"] * 2 + ["c"] * 1 + ["d"] * 9)
        p = RO.shuffle_partners(logs)
        ok = p >= 0
        self.assertEqual(int((~ok).sum()), 1); self.assertEqual(int(np.nonzero(~ok)[0][0]), 7)   # single-set log excluded
        self.assertTrue((p[ok] != np.arange(len(logs))[ok]).all())
        self.assertTrue((logs[p[ok]] == logs[ok]).all())
        self.assertTrue((p == RO.shuffle_partners(logs)).all())                 # seed-deterministic
        for n in range(2, 12):
            d = RO.derangement(n, np.random.default_rng(n))
            self.assertTrue((d != np.arange(n)).all())

    def test_bootstrap_seed_reproducible(self):
        rng = np.random.default_rng(5)
        logs = np.array([f"l{i % 7}" for i in range(70)]); g = rng.normal(size=70); o = np.abs(rng.normal(size=70)) + 1
        a = RO.column_report(logs, {"X": g}, o, n_draws=500); b = RO.column_report(logs, {"X": g}, o, n_draws=500)
        self.assertEqual(a, b)
        c = RO.column_report(logs, {"X": g}, o, n_draws=500, seed=1)
        self.assertNotEqual(a["X"]["gain_ci"], c["X"]["gain_ci"])
        self.assertTrue(a["X"]["gain_ci"][0] <= a["X"]["gain"] <= a["X"]["gain_ci"][1])

    def test_raster_arms_synthetic(self):
        N = 2
        rx = np.zeros((N, 11)); ry = np.zeros((N, 11)); ryaw = np.zeros((N, 11)); pr = np.zeros((N, 11), bool)
        rx[0] = 10.0 + np.arange(11) * 2.0; ry[0] = 0.0; pr[0] = True          # a car moving 4 m/s ahead
        rx[1] = 30.0; ry[1] = 10.0; pr[1] = True                                # a static cone
        st = RS.SetTracks(rx, ry, ryaw, pr, np.array([2.0, 0.25]), np.array([1.0, 0.25]), np.array([True, False]),
                          np.array([4.0, 0.0]), np.zeros(2), (0.0, 0.0, 0.0))
        arms = RS.build_arms(st)
        self.assertEqual(set(arms), set(RS.ARMS_RASTER))
        n = {k: v[:, :RS.NXY].sum(axis=1) for k, v in arms.items()}
        self.assertTrue((n["R-t0"] == n["R-t0"][0]).all())                      # frozen
        self.assertTrue((n["R-true"] >= n["R-dyn"]).all() and (n["R-true"] - n["R-dyn"] > 0).all())
        self.assertTrue(np.array_equal(arms["R-true"], arms["R-cv"]))           # 4 m/s linear == constant velocity here
        self.assertFalse(np.array_equal(arms["R-true"][0], arms["R-true"][50]))

    def test_empty_scene_flags_nothing(self):
        z = np.zeros((1, 11)); pr = np.ones((1, 11), bool)
        st = RS.SetTracks(z + 500.0, z, z, pr, np.array([2.0]), np.array([1.0]), np.array([True]), np.zeros(1), np.zeros(1), (0, 0, 0))
        ft = RO.footprint_tables(RO.proposals_10hz(*straight_props()), FP)
        for arm, Rg in RS.build_arms(st).items():
            nc, ttc, o0 = RO.flags_for_raster(Rg, ft)
            self.assertFalse(nc.any() or ttc.any() or o0, arm)

    def test_gfid_counts(self):
        flag = np.zeros((2, 64), bool); flag[0, :3] = True
        nc = np.ones((2, 64)); ttc = np.ones((2, 64)); nc[0, 0] = 0; nc[0, 5] = 0
        rule = np.tile(np.arange(64, 0, -1.0), (2, 1))
        g = RO.gfid(flag, nc, ttc, rule)
        self.assertEqual((g["all"]["tp"], g["all"]["fp"], g["all"]["fn"]), (1, 2, 1))
        self.assertEqual((g["top5"]["tp"], g["top5"]["fp"], g["top5"]["fn"]), (1, 2, 0))


def mutate(src, old, new):
    assert old in src, f"mutation target not found: {old!r}"
    return src.replace(old, new, 1)


MUTATIONS = [
    ("remove-fallback", "s0b_readout.py", "return np.where(allf, p6_pick, pk), allf", "return pk, allf"),
    ("nc-drop-time-index", "s0b_readout.py", "Rg[t[None, :, None], ft.idx_nc[:, 1:, :]]", "Rg[0][ft.idx_nc[:, 1:, :]]"),
    ("derangement-allows-fixed-points", "s0b_readout.py", "if not (p == ident).any():", "if True:"),
    ("ttc-no-lookahead", "s0b_readout.py", "np.minimum(ts + d, T - 1)", "np.minimum(ts, T - 1)"),
    ("global-to-ego-sign", "s0b_raster.py", "return c * dx + s * dy, -s * dx + c * dy,", "return c * dx - s * dy, s * dx + c * dy,"),
    ("one-sample-track-not-everywhere", "s0b_raster.py", "x[i], y[i], yaw[i], pres[i] = raw_x[i, j], raw_y[i, j], raw_yaw[i, j], True",
     "x[i], y[i], yaw[i] = raw_x[i, j], raw_y[i, j], raw_yaw[i, j]"),
    ("pad-column-leaks", "s0b_raster.py", "out[:, :NXY] = occ.reshape(T, NXY)", "out[:, :NXY] = occ.reshape(T, NXY); out[:, NXY] = True"),
]


def run_mutations():
    bad = 0
    for name, fn, old, new in MUTATIONS:
        d = Path(tempfile.mkdtemp(prefix="s0bmut_"))
        try:
            for f in ("s0b_raster.py", "s0b_readout.py", "test_s0b.py"):
                shutil.copy(HERE / f, d / f)
            shutil.copytree(HERE / "ref", d / "ref", ignore=shutil.ignore_patterns("__pycache__"))
            p = d / fn
            p.write_text(mutate(p.read_text(encoding="utf-8"), old, new), encoding="utf-8", newline="\n")
            r = subprocess.run([sys.executable, "test_s0b.py"], cwd=d, capture_output=True, text=True, encoding="utf-8")
            red = r.returncode != 0
            print(f"MUTATION {name}: {'RED (good)' if red else 'STILL GREEN (BAD)'}")
            bad += 0 if red else 1
        finally:
            shutil.rmtree(d, ignore_errors=True)
    print("MUTATIONS", "ALL RED" if bad == 0 else f"{bad} SURVIVED")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--mutations" in sys.argv:
        sys.exit(run_mutations())
    unittest.main()
