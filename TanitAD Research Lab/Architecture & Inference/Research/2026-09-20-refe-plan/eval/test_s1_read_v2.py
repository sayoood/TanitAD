#!/usr/bin/env python3
"""Tests of s1_read_v2.py on SYNTHETIC fixtures only (no real data).  CUDA_VISIBLE_DEVICES=-1, plain unittest:  python test_s1_read_v2.py
Part 1 = the 18 frozen v1 tests, ported with unchanged bodies (v1's `import s1_read as R` -> `import s1_read_v2 as R`; v2 re-exports the frozen names). Part 2 = the v2 additions.
Needs the frozen s1_read.py (sha asserted at import): set S1_READ_V1 or run inside the repo layout."""
import math
import os
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1_read_v2 as R  # noqa: E402

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


class T(unittest.TestCase):  # PART 1: the 18 frozen v1 tests
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


# ======================================================================== PART 2: what v2 adds
def upsample_logit(lg2):
    """[N,40,32] -> [N,80,64] 1 m logits by 2x2 replication."""
    return np.repeat(np.repeat(lg2, 2, axis=1), 2, axis=2)


def perfect_1m(d):
    """a perfect 1 m head: +3 inside / -3 outside the 1 m label mask."""
    return np.where(R.block_mask(d["label"], 2), 3.0, -3.0).astype(np.float32)


def renamed(d, mapping):
    """copy of d whose arm KEYS are renamed (the arrays are shared)."""
    e = dict(d)
    e["arms"] = {mapping.get(k, k): v for k, v in d["arms"].items()}
    return e


class T2(unittest.TestCase):
    def setUp(self):
        self.d = fixture()
        self.d["arms"]["s0"]["logit"] = self.d["_perfect"]
        self.d["arms"]["init0"] = {"logit": np.full_like(self.d["_perfect"], -3.0), "sdf": self.d["arms"]["s0"]["sdf"]}

    # ---------------------------------------------------------------- the frozen reader is the one imported
    def test_frozen_reader_is_imported_with_its_sha_and_refuses_a_changed_one(self):
        self.assertEqual(R.V1_SHA256, "58aa3a77b976a6fb4b08b4bc5126d572e8be8ef1fc37926d9c9ce0983eed718e")
        raw, lf = R._sha(R.V1_PATH), R._sha(R.V1_PATH, True)
        self.assertIn(R.V1_SHA256, (raw, lf))
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s1_read.py")
            orig = open(R.V1_PATH, "rb").read()
            b = orig.replace(b"BAR_P1, BAR_Q, BAR_CSHUF, BAR_CFLIP = 0.80", b"BAR_P1, BAR_Q, BAR_CSHUF, BAR_CFLIP = 0.79", 1)
            self.assertNotEqual(b, orig)
            open(p, "wb").write(b)
            with self.assertRaises(SystemExit):
                R.load_v1(p)

    def test_v2_reuses_the_frozen_functions_not_copies(self):
        for n in ("frame_stats", "p1_block", "q_block", "verdict", "diff_block", "city_iou_read", "boot_weights", "block_targets"):
            self.assertIs(getattr(R, n), getattr(R._V1, n))

    # ---------------------------------------------------------------- (i) arm names are not hardcoded
    def test_default_table_equals_explicit_renamed_table_digit_for_digit(self):
        base = R.run_read(self.d, n_boot=NB)
        e = renamed(self.d, {"s0": "xa", "shuf": "xb", "init0": "xc"})
        spec = [{"key": "xa", "name": "HB-s0", "role": "trained", "cell": 2.0, "shuf": "xb", "init": "xc"},
                {"key": "xb", "name": "HB-SHUF", "role": "shuf", "cell": 2.0}, {"key": "xc", "name": "HB-s0-initial", "role": "init", "cell": 2.0}]
        r = R.run_read(e, n_boot=NB, arms=spec)
        ren = {"HB-s0": "HA-s0", "HB-SHUF": "HA-SHUF", "HB-s0-initial": "HA-s0-initial"}
        self.assertEqual(sorted(r["arms"]), sorted(ren))
        for k, v in r["arms"].items():
            self.assertEqual(v, base["arms"][ren[k]])
        for k in ("bars", "controls", "verdict", "S1_F", "true_field_auc_2m", "sigma_b_m", "sensitivity_all48_groups"):
            self.assertEqual(r[k], base[k], k)
        self.assertEqual(r["per_arm"]["HB-s0"]["verdict"], "S1-GO")

    def test_each_trained_arm_is_judged_by_its_own_shuf_and_its_own_flip(self):
        d = self.d
        N = len(d["frame_log"])
        d["arms"]["b"] = {"logit": d["_perfect"].copy(), "sdf": d["arms"]["s0"]["sdf"]}                 # a second trained arm (also perfect)
        d["arms"]["bshuf"] = {"logit": d["_perfect"].copy(), "sdf": d["arms"]["s0"]["sdf"]}             # ... whose own SHUF reads the map: INVALID for b only
        d["arms"]["binit"] = {"logit": np.full((N, 40, 32), -3.0, np.float32), "sdf": d["arms"]["s0"]["sdf"]}
        spec = R.default_arm_spec(d) + [{"key": "b", "name": "B", "role": "trained", "cell": 2.0, "shuf": "bshuf", "init": "binit"},
                                        {"key": "bshuf", "name": "B-SHUF", "role": "shuf", "cell": 2.0}, {"key": "binit", "name": "B-initial", "role": "init", "cell": 2.0}]
        r = R.run_read(d, n_boot=NB, arms=spec)
        self.assertEqual(r["per_arm"]["HA-s0"]["verdict"], "S1-GO")
        self.assertEqual(r["per_arm"]["B"]["verdict"], "INVALID")
        self.assertGreaterEqual(r["per_arm"]["B"]["controls"]["C-SHUF"]["value"], 0.5)
        self.assertEqual(r["verdict"], "S1-GO")                                                          # top level = the first trained arm

    def test_trained_arm_without_a_shuf_is_incomplete_not_a_pass(self):
        spec = [{"key": "s0", "name": "X", "role": "trained", "cell": 2.0, "shuf": None, "init": "init0"}, {"key": "init0", "name": "Xi", "role": "init", "cell": 2.0}]
        r = R.run_read(self.d, n_boot=NB, arms=spec)
        self.assertEqual(r["per_arm"]["X"]["verdict"], "INCOMPLETE")
        self.assertIsNone(r["per_arm"]["X"]["controls"]["C-SHUF"])
        self.assertEqual(r["per_arm"]["X"]["x_row"], "INCOMPLETE")

    def test_arm_table_refusals(self):
        d = self.d
        good = R.default_arm_spec(d)
        with self.assertRaises(ValueError):
            R.check_spec(good + [dict(good[0])], d)                                                      # duplicate key / name
        with self.assertRaises(ValueError):
            R.check_spec([{"key": "nope", "name": "N", "role": "trained", "cell": 2.0}], d)              # no arrays
        with self.assertRaises(ValueError):
            R.check_spec([{"key": "s0", "name": "N", "role": "trained", "cell": 1.0}], d)                # 2 m arrays declared as 1 m
        with self.assertRaises(ValueError):
            R.check_spec([{"key": "s0", "name": "N", "role": "weird", "cell": 2.0}], d)
        with self.assertRaises(ValueError):
            R.check_spec([{"key": "shuf", "name": "N", "role": "shuf", "cell": 2.0}], d)                 # no trained arm
        with self.assertRaises(ValueError):
            R.check_spec([{"key": "s0", "name": "N", "role": "trained", "cell": 2.0, "shuf": "zz"}], d)
        with self.assertRaises(ValueError):
            R.cell_grid(1.3)

    def test_manifest_tables(self):
        legacy = {"files": {"head_s0_a": 1, "head_s0_b": 1, "head_shuf_a": 1, "head_shuf_b": 1}}
        spec, pairs = R.arms_from_manifest(legacy)
        self.assertEqual([(a["key"], a["name"], a["role"], a["files"]) for a in spec],
                         [("s0", "HA-s0", "trained", ["head_s0_a", "head_s0_b"]), ("shuf", "HA-SHUF", "shuf", ["head_shuf_a", "head_shuf_b"])])
        self.assertIsNone(spec[0]["init"])
        self.assertEqual(pairs, [])
        legacy["files"].update({"head_init0_a": 1, "head_init0_b": 1})
        self.assertEqual(R.arms_from_manifest(legacy)[0][0]["init"], "init0")
        v2 = {"files": {}, "arms": [{"name": "HB-s0", "role": "trained", "files": ["a", "b"], "cell": 2.0, "shuf": "HB-SHUF"}, {"name": "HC-s0", "role": "trained", "files": ["c", "d"], "cell": 1}],
              "pairs": [{"x": "HB-s0", "ref": "HA-s0"}]}
        spec, pairs = R.arms_from_manifest(v2)
        self.assertEqual([a["name"] for a in spec], ["HB-s0", "HC-s0"])
        self.assertEqual(spec[1]["cell"], 1.0)
        self.assertEqual(pairs, [{"x": "HB-s0", "ref": "HA-s0"}])

    # ---------------------------------------------------------------- (ii) the paired delta
    def _pair_fixture(self, **kw):
        d = fixture(**(kw or {"bos_shift": True}))                                                                      # ref = a head with BOS shifted (IoU < 1)
        ref = d["arms"]["s0"]["logit"].copy()
        d["arms"]["x"] = {"logit": d["_perfect"], "sdf": d["arms"]["s0"]["sdf"]}                          # X = perfect
        d["arms"]["r"] = {"logit": ref, "sdf": d["arms"]["s0"]["sdf"]}
        spec = [{"key": "x", "name": "X", "role": "trained", "cell": 2.0, "shuf": "shuf"}, {"key": "r", "name": "REF", "role": "trained", "cell": 2.0, "shuf": "shuf"},
                {"key": "shuf", "name": "SH", "role": "shuf", "cell": 2.0}]
        return d, spec

    def test_paired_delta_point_ci_and_lift(self):
        d, spec = self._pair_fixture(odd_log_shift=True)                                                   # errors vary by log: the resamples change the differences
        r = R.run_read(d, n_boot=NB, arms=spec, pairs=[{"x": "X", "ref": "REF"}])
        p = r["paired"][0]
        x, ref = r["per_arm"]["X"]["bars"], r["per_arm"]["REF"]["bars"]
        self.assertEqual(p["p1"]["point"], x["S1-P1"]["value"] - ref["S1-P1"]["value"])
        self.assertEqual(p["q"]["point"], x["S1-Q"]["value"] - ref["S1-Q"]["value"])
        self.assertGreater(p["p1"]["point"], 0.05)
        # the CI is the percentile interval of the per-resample differences of the SAME resamples (recomputed independently)
        grp = R.Groups(np.asarray(d["frame_log"]).astype(str), d["city_of_log"])
        W = R.boot_weights(grp.G, NB, R.SEED)
        sx = R.frame_stats(d["arms"]["x"]["logit"], d["arms"]["x"]["sdf"], d["label"], d["valid"])
        sr = R.frame_stats(d["arms"]["r"]["logit"], d["arms"]["r"]["sdf"], d["label"], d["valid"])
        _, rx, _ = R.p1_block(sx, grp, W)
        _, rr, _ = R.p1_block(sr, grp, W)
        ci, _ = R.ci95(rx - rr)
        self.assertEqual(p["p1"]["ci95"], ci)
        self.assertTrue(p["p1"]["lower95"] > 0)

    def test_paired_delta_of_an_arm_with_itself_is_exactly_zero_and_not_a_lift(self):
        d, spec = self._pair_fixture()
        d["arms"]["x2"] = {"logit": d["arms"]["x"]["logit"].copy(), "sdf": d["arms"]["x"]["sdf"]}
        spec.append({"key": "x2", "name": "X2", "role": "trained", "cell": 2.0, "shuf": "shuf"})
        r = R.run_read(d, n_boot=NB, arms=spec, pairs=[{"x": "X2", "ref": "X"}])
        p = r["paired"][0]
        self.assertEqual(p["p1"]["point"], 0.0)
        self.assertEqual(p["p1"]["ci95"], [0.0, 0.0])
        self.assertEqual(p["q"]["ci95"], [0.0, 0.0])
        self.assertFalse(p["x_lift"])

    def test_mutation_unpaired_delta_goes_red(self):
        """the regression arm of the paired block: reading REF through DIFFERENT resamples makes the delta of two identical arms non-degenerate; the exact-zero assertion must fail on it."""
        d, spec = self._pair_fixture(odd_log_shift=True)                                                   # head errors vary by LOG, so the resamples matter
        d["arms"]["x2"] = {"logit": d["arms"]["r"]["logit"].copy(), "sdf": d["arms"]["r"]["sdf"]}          # an arm identical to REF, a different object
        spec.append({"key": "x2", "name": "X2", "role": "trained", "cell": 2.0, "shuf": "shuf"})
        ok = R.run_read(d, n_boot=NB, arms=spec, pairs=[{"x": "X2", "ref": "REF"}])["paired"][0]
        self.assertEqual(ok["p1"]["ci95"], [0.0, 0.0])
        bad = R.run_read(d, n_boot=NB, arms=spec, pairs=[{"x": "X2", "ref": "REF"}], mutate="unpaired_delta")["paired"][0]
        self.assertGreater(bad["p1"]["ci95"][1] - bad["p1"]["ci95"][0], 1e-4)
        self.assertNotEqual(bad["p1"]["ci95"], [0.0, 0.0])

    def test_x_rows_mechanical(self):
        xr = R.x_row
        c = {"LV": .9, "BOS": .9}
        self.assertEqual(xr(.9, .95, True, .1, .5, .9, c, True), "X-GO")
        self.assertEqual(xr(.9, .95, None, .1, .5, .9, c, True), "INCOMPLETE")
        self.assertEqual(xr(.9, .95, False, .1, .5, .9, c, True), "UNNAMED")
        self.assertEqual(xr(.75, .95, True, .1, .5, .85, {"LV": .85, "BOS": .6}, None), "X-COMP")
        self.assertEqual(xr(.85, .8, True, .1, .5, .9, c, None), "X-BOUND")
        self.assertEqual(xr(.5, .6, True, .1, .5, .5, c, True), "X-LIFT")
        self.assertEqual(xr(.5, .6, True, .1, .5, .5, c, False), "X-NULL")
        self.assertEqual(xr(.5, .6, True, .1, .5, .5, c, None), "INCOMPLETE")
        self.assertEqual(xr(.9, .95, True, .5, .5, .9, c, True), "INVALID")
        self.assertEqual(xr(.9, .95, True, .1, .8, .9, c, True), "INVALID")
        self.assertEqual(xr(.8, .9, True, .1, .5, .9, c, True), "X-GO")                                    # bars are >=

    def test_x_row_in_a_read_is_lift_aware(self):
        d, spec = self._pair_fixture()
        d["arms"]["x"]["logit"] = np.full_like(d["_perfect"], 3.0)                                         # X: constant-drivable (P1 0.125), fails both bars
        d["arms"]["r"]["logit"] = np.full_like(d["_perfect"], -3.0)                                        # REF: constant-empty (P1 0.0), worse on P1
        r = R.run_read(d, n_boot=NB, arms=spec, pairs=[{"x": "X", "ref": "REF"}])
        self.assertFalse(r["per_arm"]["X"]["bars"]["S1-P1"]["holds"])
        self.assertTrue(r["paired"][0]["p1"]["point"] > 0)
        self.assertEqual(r["per_arm"]["X"]["x_row"], "X-LIFT" if r["paired"][0]["x_lift"] else "X-NULL")

    # ---------------------------------------------------------------- (iv) a supplied in-sample IoU splits FIT / WEAK
    def test_supplied_in_sample_iou_splits_fit_weak(self):
        d = fixture()
        d["arms"]["s0"]["logit"] = np.full_like(d["_perfect"], 3.0)
        for val, want in ((0.85, "S1-FIT"), (0.8499999, "S1-WEAK")):
            spec = R.default_arm_spec(d)
            spec[0]["in_sample_iou"] = val
            r = R.run_read(d, n_boot=NB, arms=spec)
            self.assertEqual(r["verdict"], want)
            self.assertFalse(any(m.startswith("MISSING-INPUT in_sample_iou") for m in r["missing_inputs"]))
        r = R.run_read(d, n_boot=NB)
        self.assertEqual(r["verdict"], "INCOMPLETE")
        self.assertTrue(any(m.startswith("MISSING-INPUT in_sample_iou") for m in r["missing_inputs"]))

    # ---------------------------------------------------------------- (iii) HC's 1 m grid
    def test_cell_grid_and_block_mask(self):
        self.assertEqual(R.cell_grid(2.0), (40, 32, 4))
        self.assertEqual(R.cell_grid(1.0), (80, 64, 2))
        lab = self.d["label"]
        self.assertTrue((R.block_mask(lab, 4) == (R.block_targets(lab)[0] >= 0.5)).all())

    def test_dk_cell_2m_is_byte_identical_to_the_frozen_call_and_1m_is_finer(self):
        d = self.d
        pts = R.footprint_points(d["poses20"][:6])
        m2 = np.asarray(d["_perfect"][:6]) > 0
        a, na = R.dk_from_masks(m2, pts)
        b, nb = R.dk_from_masks_cell(m2, pts, 2.0)
        self.assertTrue((a == b).all() and (na == nb).all())
        m1 = R.block_mask(d["label"][:6], 2)
        c, _ = R.dk_from_masks_cell(m1, pts, 1.0)
        self.assertEqual(c.shape, a.shape)
        self.assertTrue(np.isfinite(c).any())

    def test_pool_rule_is_probability_space_threshold_half(self):
        lg = np.zeros((2, 80, 64), np.float32)
        lg[0, 0:2, 0:2] = [[100.0, -1.0], [-1.0, -1.0]]          # mean logit +24 (would be drivable), mean sigmoid 0.452 (not)
        lg[1, 0:2, 0:2] = [[-100.0, 1.0], [1.0, 1.0]]            # mean logit -24 (would be empty), mean sigmoid 0.548 (drivable)
        p, _ = R.pool_to_2m(lg, np.zeros_like(lg))
        self.assertLess(p[0, 0, 0], 0.0)
        self.assertGreater(p[1, 0, 0], 0.0)
        self.assertEqual(p[0, 5, 5], 0.0)                         # all-zero logits: p = 0.5 exactly -> NOT > 0.5 (strict, like logit > 0)
        self.assertEqual(p.shape, (2, 40, 32))

    def test_perfect_1m_head_reads_one_on_the_reference_grid_and_the_1m_grid(self):
        d = self.d
        N = len(d["frame_log"])
        d["arms"]["hc"] = {"logit": perfect_1m(d), "sdf": np.zeros((N, 80, 64), np.float32)}
        d["arms"]["hcs"] = {"logit": np.full((N, 80, 64), -3.0, np.float32), "sdf": np.zeros((N, 80, 64), np.float32)}
        spec = [{"key": "hc", "name": "HC-s0", "role": "trained", "cell": 1.0, "shuf": "hcs"}, {"key": "hcs", "name": "HC-SHUF", "role": "shuf", "cell": 1.0}]
        r = R.run_read(d, n_boot=NB, arms=spec)
        a = r["arms"]["HC-s0"]
        for b in ("0-20", "20-40", "40-72", "0-40"):
            self.assertAlmostEqual(a["p1"][b]["point"], 1.0, places=12)        # the 1 m road recovers the 2 m reference mask exactly
        self.assertAlmostEqual(a["p1_1m_grid"]["point"], 1.0, places=12)
        self.assertAlmostEqual(a["q"]["point"], 1.0, places=12)
        self.assertLess(a["q_yflip"]["point"], a["q"]["point"])
        self.assertEqual(r["per_arm"]["HC-s0"]["verdict"], "INCOMPLETE")        # no initial head supplied: S1-F unread
        with self.assertRaises(ValueError):                                     # a 1 m head declared as 2 m is refused
            R.run_read(d, n_boot=NB, arms=[dict(spec[0], cell=2.0), spec[1]])

    def test_1m_p1_equals_the_2m_arm_when_the_1m_head_is_an_upsampled_2m_head(self):
        d = fixture(bos_shift=True)
        N = len(d["frame_log"])
        lg2 = d["arms"]["s0"]["logit"]
        d["arms"]["u"] = {"logit": upsample_logit(lg2), "sdf": np.zeros((N, 80, 64), np.float32)}
        spec = [{"key": "s0", "name": "A2", "role": "trained", "cell": 2.0, "shuf": "shuf"}, {"key": "u", "name": "A1", "role": "trained", "cell": 1.0, "shuf": "shuf"},
                {"key": "shuf", "name": "S", "role": "shuf", "cell": 2.0}]
        r = R.run_read(d, n_boot=NB, arms=spec)
        for b in ("0-20", "20-40", "40-72", "0-40"):
            self.assertEqual(r["arms"]["A1"]["p1"][b]["point"], r["arms"]["A2"]["p1"][b]["point"])
            self.assertEqual(r["arms"]["A1"]["p1"][b]["ci95"], r["arms"]["A2"]["p1"][b]["ci95"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
