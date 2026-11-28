#!/usr/bin/env python3
"""Tests of s1_read.py on SYNTHETIC fixtures only (no real data).  CUDA_VISIBLE_DEVICES=-1, plain unittest:  python test_s1_read.py"""
import math
import os
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1_read as R  # noqa: E402

NB = 300
CITIES = ["LV", "BOS", "PIT", "SG"]
FPL = 4                         # frames per log
LPC = 3                         # logs per city


def road_label(c):
    """int8 SDF [160,128]: a road along x, width 8 m, centre y = c (a multiple of 0.5 m); + inside, 0.1 m units."""
    yf = R.Y0 + R.FINE_RES * (np.arange(R.NYF) + 0.5)
    s = np.clip(np.rint((4.0 - np.abs(yf - c)) / 0.1), -127, 127).astype(np.int8)
    return np.broadcast_to(s[None, :], (R.NXF, R.NYF)).copy()


def mask_logit(label, shift_cols=None, shift_from_x=None):
    """a perfect head's logit [40,32] (+3 inside, -3 outside) from a label; optionally the road shifted by `shift_cols` coarse columns in y (for x-columns >= shift_from_x)."""
    frac, _ = R.block_targets(label[None])
    m = frac[0] >= 0.5
    if shift_cols:
        ms = np.roll(m, shift_cols, axis=1)
        if shift_from_x is None:
            m = ms
        else:
            m = m.copy()
            m[shift_from_x:] = ms[shift_from_x:]
    return np.where(m, 3.0, -3.0).astype(np.float32)


def proposals(c, rng):
    """[64, 20, 3]: 32 pass (drift toward the road side to 1.5 c, stay on the road), 32 fail (half drift 3.2 m to the side opposite the road, half 9 m beyond it); dac = 1 / 0.
    Built so that a y-flipped field mis-ranks them (the flipped road covers the failing side)."""
    sg = 1.0 if c >= 0 else -1.0
    s = np.arange(20) / 19.0
    P = np.zeros((64, 20, 3))
    for k in range(64):
        x = 1.25 * (np.arange(20) + 1) * (0.8 + 0.4 * rng.random())
        if k < 32:
            y = 1.5 * c * s * (0.5 + 0.5 * rng.random())
        else:
            y = (-sg * 3.2 if k % 2 else sg * 9.0) * s * (0.9 + 0.1 * rng.random())
        P[k, :, 0], P[k, :, 1] = x, y
    dac = np.r_[np.ones(32), np.zeros(32)]
    return P, dac


def fixture(centres=(-2.0, 2.0), seed=0, bos_shift=False, odd_log_shift=False, far_shift=False):
    rng = np.random.default_rng(seed)
    logs, city_of_log, labels, poses, dacs = [], {}, [], [], []
    heads = []
    for ci, city in enumerate(CITIES):
        for li in range(LPC):
            lg = f"2021.01.{ci + 1:02d}.00.00.00_veh-{ci}_{li:05d}_{li + 9:05d}"
            city_of_log[lg] = city
            for fi in range(FPL):
                c = centres[(li + fi) % len(centres)]
                lab = road_label(c)
                P, dac = proposals(c, rng)
                shift = 1 if (bos_shift and city == "BOS") or (odd_log_shift and li % 2 == 1) else 0
                logits = mask_logit(lab, shift_cols=shift, shift_from_x=24 if far_shift else None)
                if far_shift:       # only the far band is wrong
                    logits = mask_logit(lab, shift_cols=1, shift_from_x=24)
                logs.append(lg)
                labels.append(lab)
                poses.append(P)
                dacs.append(dac)
                heads.append(logits)
    N = len(logs)
    sdf = np.zeros((N, 40, 32), np.float32)
    perfect = np.stack([mask_logit(l) for l in labels])
    d = {"frame_log": np.asarray(logs), "city_of_log": city_of_log, "label": np.stack(labels), "valid": np.ones(N, bool), "poses20": np.stack(poses),
         "dac": np.stack(dacs), "set_valid": np.ones(N, bool), "all_logs": sorted(city_of_log),
         "arms": {"s0": {"logit": np.stack(heads), "sdf": sdf}, "shuf": {"logit": np.full((N, 40, 32), -3.0, np.float32), "sdf": sdf}}}
    d["_perfect"] = perfect
    return d


class T(unittest.TestCase):
    # ------------------------------------------------------------------ primitives
    def test_boot_draw_is_the_programmes(self):
        W = R.boot_weights(13, 50, 20260927)
        rng = np.random.default_rng(20260927)
        ref = np.stack([np.bincount(rng.integers(0, 13, 13), minlength=13) for _ in range(50)])
        self.assertTrue((W == ref).all())
        rng = np.random.default_rng(20260927)
        ch = rng.choice(13, size=(50, 13), replace=True)          # sel_common.log_boot_idx's scheme
        self.assertTrue((np.stack([np.bincount(r, minlength=13) for r in ch]) == W).all())

    def test_to_navsim_vec_equals_verbatim_loop(self):
        rng = np.random.default_rng(1)
        p = np.cumsum(rng.normal(size=(5, 20, 3)), axis=1) * np.array([1.0, 0.3, 0.1])
        p[:, 7, 2] = 3.1
        p[:, 8, 2] = -3.1                                         # a heading wrap inside an interval

        def loop(poses20, dt=0.2):
            t_src = dt * np.arange(1, poses20.shape[0] + 1)
            out = np.zeros((8, 3))
            for k in range(8):
                t = 0.5 * (k + 1)
                j = int(np.searchsorted(t_src, t - 1e-9))
                if j < poses20.shape[0] and abs(t_src[j] - t) < 1e-9:
                    out[k] = poses20[j]
                    continue
                w = (t - t_src[j - 1]) / (t_src[j] - t_src[j - 1])
                out[k, :2] = (1 - w) * poses20[j - 1, :2] + w * poses20[j, :2]
                out[k, 2] = R.wrap(poses20[j - 1, 2] + w * R.wrap(poses20[j, 2] - poses20[j - 1, 2]))
            return out
        v = R.to_navsim_vec(p)
        for i in range(5):
            self.assertTrue(np.allclose(v[i], loop(p[i]), atol=1e-12))

    def test_densify_hits_the_input_poses_and_starts_at_origin(self):
        p8 = np.cumsum(np.ones((8, 3)), axis=0) * np.array([2.0, 0.1, 0.0])
        d = R.densify(p8)
        self.assertEqual(d.shape, (41, 3))
        self.assertTrue(np.allclose(d[0], 0) and np.allclose(d[5], p8[0]) and np.allclose(d[40], p8[7]))
        self.assertTrue(np.allclose(d[1], p8[0] * 0.2))

    def test_sdf_from_mask_halfplane(self):
        m = np.zeros((40, 32), bool)
        m[:, 16:] = True
        f = R.sdf_from_mask(m, 2.0)
        self.assertAlmostEqual(f[5, 16], 1.0)                      # first inside cell centre: half a cell = 1 m
        self.assertAlmostEqual(f[5, 15], -1.0)
        self.assertAlmostEqual(f[5, 20], 9.0)
        self.assertTrue((R.sdf_from_mask(np.ones((4, 4), bool), 2.0) == R.BIG_M).all())

    def test_pooled_auc_matches_brute_force(self):
        rng = np.random.default_rng(3)
        sc = np.round(rng.normal(size=300), 1)                      # ties on purpose
        ps = rng.random(300) < 0.6
        cl = rng.integers(0, 7, 300)
        S, Nm = R.auc_cluster_stats(sc, ps, cl, 7)
        pos, neg = sc[ps], sc[~ps]
        brute = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
        self.assertAlmostEqual(R.auc_value(S, Nm), brute, places=12)

    def test_city_weights(self):
        self.assertEqual(sum(R.CITY_N.values()), R.CITY_TOTAL)
        self.assertAlmostEqual(sum(R.CITY_WEIGHT.values()), 1.0, places=12)

    def test_parse_a3_lines(self):
        txt = "x\n2021.06.09.14.50.36_veh-26_00832_00905  LV  11  SFT24\n2021.08.09.17.55.59_veh-28_00021_00307  BOS  15  extra-8\n```"
        self.assertEqual(R.parse_a3_cities(txt), {"2021.06.09.14.50.36_veh-26_00832_00905": "LV", "2021.08.09.17.55.59_veh-28_00021_00307": "BOS"})

    # ------------------------------------------------------------------ the verdict table, row by row
    def test_verdict_rows(self):
        v = R.verdict
        self.assertEqual(v(0.9, 0.95, True, 0.1, 0.5, 0.9, {"LV": .9, "BOS": .9}, None)[0], "S1-GO")
        self.assertEqual(v(0.9, 0.95, None, 0.1, 0.5, 0.9, {"LV": .9}, None)[0], "INCOMPLETE")
        self.assertEqual(v(0.9, 0.95, False, 0.1, 0.5, 0.9, {"LV": .9}, None)[0], "UNNAMED")
        self.assertEqual(v(0.75, 0.95, True, 0.1, 0.5, 0.85, {"LV": .85, "BOS": .6, "PIT": .8}, None)[0], "S1-COMP")
        self.assertEqual(v(0.75, 0.95, True, 0.1, 0.5, 0.85, {"LV": .85, "BOS": .75}, None)[0], "UNNAMED")
        self.assertEqual(v(0.85, 0.80, True, 0.1, 0.5, 0.9, {"LV": .9}, None)[0], "S1-BOUND")
        self.assertEqual(v(0.5, 0.6, True, 0.1, 0.5, 0.5, {"LV": .5}, None)[0], "INCOMPLETE")
        self.assertEqual(v(0.5, 0.6, True, 0.1, 0.5, 0.5, {"LV": .5}, 0.90)[0], "S1-FIT")
        self.assertEqual(v(0.5, 0.6, True, 0.1, 0.5, 0.5, {"LV": .5}, 0.80)[0], "S1-WEAK")
        self.assertEqual(v(0.9, 0.95, True, 0.5, 0.5, 0.9, {"LV": .9}, None)[0], "INVALID")        # C-SHUF at the bar: < 0.50 required
        self.assertEqual(v(0.9, 0.95, True, 0.1, 0.8, 0.9, {"LV": .9}, None)[0], "INVALID")        # C-FLIP at the bar: < 0.80 required
        self.assertEqual(v(0.9, 0.95, True, None, 0.5, 0.9, {"LV": .9}, None)[0], "INCOMPLETE")
        # boundaries: P1 exactly 0.80 and Q exactly 0.90 hold (>=)
        self.assertEqual(v(0.80, 0.90, True, 0.1, 0.5, 0.9, {"LV": .9}, None)[0], "S1-GO")

    # ------------------------------------------------------------------ the whole read on a perfect head
    def test_perfect_head_is_go(self):
        d = fixture()
        d["arms"]["s0"]["logit"] = d["_perfect"]
        d["arms"]["init0"] = {"logit": np.full_like(d["_perfect"], -3.0), "sdf": d["arms"]["s0"]["sdf"]}
        r = R.run_read(d, n_boot=NB)
        self.assertAlmostEqual(r["bars"]["S1-P1"]["value"], 1.0, places=12)
        self.assertAlmostEqual(r["bars"]["S1-Q"]["value"], 1.0, places=12)
        self.assertEqual(r["true_field_auc_2m"]["point"], 1.0)
        self.assertLess(r["controls"]["C-SHUF"]["value"], 0.5)
        self.assertLess(r["controls"]["C-FLIP"]["value"], 0.8)
        self.assertTrue(r["bars"]["S1-F"]["holds"])
        self.assertEqual(r["verdict"], "S1-GO")
        for b in ("0-20", "20-40", "40-72", "0-40"):
            self.assertAlmostEqual(r["arms"]["HA-s0"]["p1"][b]["point"], 1.0, places=12)
        self.assertEqual(r["arms"]["HA-s0"]["q"]["proposals_with_no_point_inside"], 0)

    def test_missing_initial_head_is_incomplete_never_substituted(self):
        d = fixture()
        d["arms"]["s0"]["logit"] = d["_perfect"]
        r = R.run_read(d, n_boot=NB)
        self.assertEqual(r["verdict"], "INCOMPLETE")
        self.assertIsNone(r["bars"]["S1-F"])
        self.assertTrue(any(m.startswith("MISSING-INPUT initial_head_predictions") for m in r["missing_inputs"]))
        self.assertTrue(any(m.startswith("MISSING-INPUT in_sample_iou") for m in r["missing_inputs"]))

    def test_init_equal_to_trained_is_not_a_pass(self):
        d = fixture()
        d["arms"]["s0"]["logit"] = d["_perfect"]
        d["arms"]["init0"] = {"logit": d["_perfect"].copy(), "sdf": d["arms"]["s0"]["sdf"]}
        r = R.run_read(d, n_boot=NB)
        self.assertFalse(r["bars"]["S1-F"]["holds"])
        self.assertEqual(r["verdict"], "UNNAMED")

    def test_constant_head_fails_p1_and_q(self):
        d = fixture()
        for val in (3.0, -3.0):
            d["arms"]["s0"]["logit"] = np.full_like(d["_perfect"], val)
            r = R.run_read(d, n_boot=NB)
            self.assertLess(r["bars"]["S1-P1"]["value"], R.BAR_P1)
            self.assertFalse(r["bars"]["S1-P1"]["holds"])
            self.assertFalse(r["bars"]["S1-Q"]["holds"])
            self.assertEqual(r["verdict"], "INCOMPLETE")            # both fail, in-sample IoU missing
        d["arms"]["s0"]["logit"] = np.full_like(d["_perfect"], 3.0)
        r = R.run_read(d, n_boot=NB)
        self.assertAlmostEqual(r["bars"]["S1-P1"]["value"], 0.125, places=12)      # 4 of 32 cells are road
        self.assertAlmostEqual(r["bars"]["S1-Q"]["value"], 0.5, places=12)        # every d_k ties

    # ------------------------------------------------------------------ the controls
    def test_shuffled_head_that_passes_triggers_invalid(self):
        d = fixture()
        d["arms"]["s0"]["logit"] = d["_perfect"]
        d["arms"]["init0"] = {"logit": np.full_like(d["_perfect"], -3.0), "sdf": d["arms"]["s0"]["sdf"]}
        d["arms"]["shuf"]["logit"] = d["_perfect"].copy()           # a "no-map-information" head that reads the map: the instrument cannot tell
        r = R.run_read(d, n_boot=NB)
        self.assertGreaterEqual(r["controls"]["C-SHUF"]["value"], 0.5)
        self.assertEqual(r["verdict"], "INVALID")
        d["arms"]["shuf"]["logit"] = np.roll(d["_perfect"], 3, axis=2).copy()    # a road 6 m off: a head that does not read this frame's map
        r = R.run_read(d, n_boot=NB)
        self.assertLess(r["controls"]["C-SHUF"]["value"], 0.5)
        self.assertEqual(r["verdict"], "S1-GO")

    def test_yflip_is_computed_on_the_flipped_prediction(self):
        d = fixture()
        d["arms"]["s0"]["logit"] = d["_perfect"]
        r = R.run_read(d, n_boot=NB)
        flipped = d["_perfect"][:, :, ::-1]
        pts = R.footprint_points(d["poses20"])
        dk, _ = R.dk_from_masks(flipped > 0, pts)
        grp = R.Groups(d["frame_log"], d["city_of_log"])
        W = R.boot_weights(grp.G, NB)
        q, _ = R.q_block(dk, d["dac"], d["set_valid"], grp, W, grp.idx)
        self.assertEqual(r["arms"]["HA-s0"]["q_yflip"]["point"], q["point"])
        self.assertLess(q["point"], 0.80)
        self.assertLess(q["point"], r["arms"]["HA-s0"]["q"]["point"])

    def test_yflip_blind_instrument_is_invalid(self):
        d = fixture(centres=(0.0,))                                  # a road symmetric about the ego: the flip changes nothing
        d["arms"]["s0"]["logit"] = d["_perfect"]
        d["arms"]["init0"] = {"logit": np.full_like(d["_perfect"], -3.0), "sdf": d["arms"]["s0"]["sdf"]}
        r = R.run_read(d, n_boot=NB)
        self.assertGreaterEqual(r["controls"]["C-FLIP"]["value"], 0.8)
        self.assertEqual(r["verdict"], "INVALID")

    # ------------------------------------------------------------------ city reweighting: a known value, and the mutation that must break it
    def test_city_reweighting_known_value_and_mutations_go_red(self):
        d = fixture(bos_shift=True)                                  # BOS head = road shifted one 2 m cell: inter 3, union 5 => IoU 0.6; other cities 1.0
        d["arms"]["s0"]["logit"] = d["arms"]["s0"]["logit"]
        known = 1.0 - 0.4 * 3731 / 12146
        r = R.run_read(d, n_boot=NB)
        got = r["bars"]["S1-P1"]["value"]
        self.assertAlmostEqual(got, known, places=12)
        pc = r["arms"]["HA-s0"]["p1"]["0-40"]["per_city"]
        self.assertAlmostEqual(pc["BOS"], 0.6, places=12)
        for c in ("LV", "PIT", "SG"):
            self.assertAlmostEqual(pc[c], 1.0, places=12)
        # mutation arm 1: drop the reweighting -> the known value check must fail (pooled IoU 3600/4080)
        rm = R.run_read(d, n_boot=NB, mutate="no_reweight")
        self.assertAlmostEqual(rm["bars"]["S1-P1"]["value"], 3600.0 / 4080.0, places=12)
        self.assertGreater(abs(rm["bars"]["S1-P1"]["value"] - known), 1e-3)
        # mutation arm 2: per-frame resampling instead of log-cluster -> the CI must differ (narrower on a log-clustered fixture)
        d2 = fixture(odd_log_shift=True)
        a = R.run_read(d2, n_boot=NB)["bars"]["S1-P1"]["ci95"]
        b = R.run_read(d2, n_boot=NB, mutate="frame_resample")["bars"]["S1-P1"]["ci95"]
        self.assertGreater((a[1] - a[0]), 1.25 * (b[1] - b[0]))

    def test_bands(self):
        d = fixture(far_shift=True)                                  # the head is right for x < 40 m (columns < 24) and shifted beyond
        r = R.run_read(d, n_boot=NB)
        p1 = r["arms"]["HA-s0"]["p1"]
        self.assertAlmostEqual(p1["0-20"]["point"], 1.0, places=12)
        self.assertAlmostEqual(p1["20-40"]["point"], 1.0, places=12)
        self.assertAlmostEqual(p1["0-40"]["point"], 1.0, places=12)
        self.assertAlmostEqual(p1["40-72"]["point"], 0.6, places=12)

    def test_q_is_live_the_head_moves_the_auc(self):
        d = fixture()
        d["arms"]["s0"]["logit"] = d["_perfect"]
        q_good = R.run_read(d, n_boot=NB)["bars"]["S1-Q"]["value"]
        d["arms"]["s0"]["logit"] = np.roll(d["_perfect"], 3, axis=2)          # road moved 6 m
        q_bad = R.run_read(d, n_boot=NB)["bars"]["S1-Q"]["value"]
        self.assertEqual(q_good, 1.0)
        self.assertLess(q_bad, 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
