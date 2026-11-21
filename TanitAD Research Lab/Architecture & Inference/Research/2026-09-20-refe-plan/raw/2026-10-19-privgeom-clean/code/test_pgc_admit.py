"""Synthetic CPU tests of the PRIVGEOM-CLEAN admissibility check (`pgc_admit`): the planted POS / NEG controls, the RED arm (disable the check => the leak gets through), the mutation table,
the family ablation, and reachability (the fit's block reads ONLY the 9 key-step |clearance| columns).

The synthetic proposals are signed-clearance random walks (41 steps at 0.1 s, 64 proposals per set); the DAC label is the exact verdict `min_t c_t >= 0` over ALL 41 steps; the candidate block is the 9 key-step
|c_t| (steps 0, 5, ..., 40); the registered leaky block adds |c_min|.  Nothing here reads a stored label or a real scalar.
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pgc_admit as AD                                                                   # noqa: E402
import pgc_lib as C                                                                      # noqa: E402

KEY = (0, 5, 10, 15, 20, 25, 30, 35, 40)
_CACHE = {}


def synth(n_sets: int = 260, seed: int = 11):
    """-> block [N,64,9] float64, leaky10 [N,64,10], label [N,64] (0 unsafe / 1 safe), g15 [N,64,15] (the extraction layout: c_min, frac_out, 9 SIGNED key clearances, 4 NC columns).

    A world built so that its structure is KNOWN: the 9 key-step magnitudes B are a smooth positive process; the verdict is a NOISY function of proximity (P(unsafe) = sigmoid(-(min_k B_k - 0.8) / 0.5)), so NO rule over
    B can prove it (the candidate is unprovable by construction).  |c_min| obeys the true identity of the min: when SAFE, |c_min| = c_min <= every c_t = B_k (so |c_min| <= min_k B_k); when UNSAFE it is unrelated to B.
    Hence the registered 10-column block PROVES unsafe through the ORDER B_k < |c_min| (R54), while no single column does."""
    key = (n_sets, seed)
    if key in _CACHE:
        return _CACHE[key]
    rng = np.random.default_rng(seed)
    tt = np.linspace(0.0, 1.0, 9)[None, None, :]
    logw = np.log(rng.uniform(0.4, 4.0, size=(n_sets, 1, 1)))
    wave = 0.45 * np.sin(2 * np.pi * rng.uniform(0.3, 1.5, (n_sets, 64, 1)) * tt + rng.uniform(0, 6.28, (n_sets, 64, 1)))
    B = np.exp(logw + wave + rng.normal(0, 0.45, (n_sets, 64, 1)) + np.cumsum(rng.normal(0, 0.12, (n_sets, 64, 9)), -1))
    B[..., 0] = np.exp(logw[..., 0] + rng.normal(0, 0.05, (n_sets, 1))) * np.ones((1, 64))   # the start clearance is shared inside a set
    pu = 1.0 / (1.0 + np.exp((B.min(-1) - 0.8) / 0.5))
    unsafe = rng.random((n_sets, 64)) < pu
    label = np.where(unsafe, 0.0, 1.0)
    cmin_abs = np.where(unsafe, rng.exponential(0.9, (n_sets, 64)), B.min(-1) * rng.uniform(0.0, 1.0, (n_sets, 64)))
    block = B
    leaky10 = np.concatenate([cmin_abs[..., None], block], -1)
    g15 = np.zeros((n_sets, 64, 15))
    g15[..., 0] = np.where(unsafe, -cmin_abs, cmin_abs)
    g15[..., 1] = unsafe.astype(float)
    g15[..., 2:11] = B
    _CACHE[key] = (block, leaky10, label, g15)
    return _CACHE[key]


class TestPlantedControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block, cls.leaky10, cls.label, cls.g15 = synth()
        cls.res = AD.a_test(cls.block, cls.label, cls.leaky10)

    def test_data_is_a_real_test(self):
        u = float((self.label == 0).mean())
        self.assertTrue(0.08 < u < 0.6, u)                                              # both classes present in numbers that matter
        c = self.res["candidate"]
        self.assertGreater(c["n_unsafe"], 1000)

    def test_positive_controls_flagged(self):
        for name, v in self.res["controls"].items():
            if v["expected_flagged"]:
                self.assertTrue(v["flagged"], f"{name} is a verdict-bearing block and must be REFUSED: {v['best_unsafe_proof_by_family']}")

    def test_negative_controls_pass(self):
        for name, v in self.res["controls"].items():
            if not v["expected_flagged"]:
                self.assertFalse(v["flagged"], f"{name} is clean and must PASS: {v['top']}")

    def test_registered_nosign_block_is_the_R54_case(self):
        v = self.res["controls"]["POS-REGISTERED"]
        self.assertGreater(v["hits_by_family"]["P"], 0)                                 # the pair family finds |c_t| < |c_min|
        b = v["best_unsafe_proof_by_family"]["P"]
        self.assertGreaterEqual(b["precision"], AD.PREC_BAR)

    def test_order_only_plant_is_caught_by_order_not_by_value(self):
        v = self.res["controls"]["POS-ORDER"]
        self.assertEqual(v["hits_by_family"]["U"], 0)                                   # no unary VALUE rule separates it ...
        self.assertGreater(v["hits_by_family"]["P"], 0)                                 # ... its ORDER against C_ref does

    def test_linear_combination_plant_needs_a_combination_family(self):
        v = self.res["controls"]["POS-LIN"]
        self.assertEqual(v["hits_by_family"]["U"] + v["hits_by_family"]["P"] + v["hits_by_family"]["V"], 0)
        self.assertGreater(v["hits_by_family"]["LIN"] + v["hits_by_family"]["T"], 0)

    def test_candidate_admissible_and_controls_ok(self):
        self.assertTrue(self.res["controls_ok"])
        self.assertFalse(self.res["candidate"]["flagged"], self.res["candidate"]["top"])
        self.assertTrue(self.res["admissible"])


class TestRedArm(unittest.TestCase):
    """Disable the check => the planted leaks GET THROUGH and the controls read wrong (the gate would FAIL).  Every mutation of the check must do the same."""

    @classmethod
    def setUpClass(cls):
        cls.block, cls.leaky10, cls.label, cls.g15 = synth()

    def test_disabled_check_lets_every_planted_leak_through(self):
        r = AD.a_test(self.block, self.label, self.leaky10, cfg=AD.AdmitConfig(disabled=True))
        self.assertFalse(r["controls_ok"])
        leaked = sorted(k for k, v in r["controls"].items() if v["expected_flagged"] and not v["flagged"])
        self.assertEqual(leaked, sorted(k for k, v in r["controls"].items() if v["expected_flagged"]))   # ALL planted leaks pass a disabled check
        self.assertIn("POS-REGISTERED", leaked)
        self.assertIn("POS-EQUAL", leaked)

    def test_every_mutation_breaks_the_controls(self):
        for name, cfg in AD.mutations().items():
            r = AD.a_test(self.block, self.label, self.leaky10, cfg=cfg)
            self.assertFalse(r["controls_ok"], f"mutation survives: {name}")

    def test_the_mutation_table_includes_the_inverted_sense(self):
        self.assertIn("sense inverted (flag when nothing is proven)", AD.mutations())

    def test_family_ablation_each_plant_is_caught_by_some_family(self):
        ab = AD.family_ablation(self.block, self.label, self.leaky10)
        for plant in ("POS-REGISTERED", "POS-EQUAL", "POS-VALUE", "POS-ORDER", "POS-LIN"):
            self.assertTrue(any(ab[f][plant] for f in AD.FAMILIES), plant)
        print("  family ablation (rows = family alone, cols = plants flagged):")
        for f in AD.FAMILIES:
            print(f"   {f:>3}: " + " ".join(f"{p}={'Y' if v else '.'}" for p, v in ab[f].items()))


class TestReachability(unittest.TestCase):
    """The fit reads ONLY `clean_block(g)`; the sign-bearing columns can never reach it (mutate them: the block must not move; a MUTATED block that includes column 0 must be caught)."""

    def test_sign_and_nc_columns_do_not_reach_the_fit(self):
        _, _, _, g15 = synth()
        b0 = C.clean_block(g15)
        g2 = g15.copy()
        g2[..., 0] = 123.0                                                               # |c_min| / sign
        g2[..., 1] = 7.0                                                                 # frac_out
        g2[..., 11:] = -5.0                                                              # NC
        self.assertTrue(np.array_equal(b0, C.clean_block(g2)))
        self.assertEqual(b0.shape[-1], 9)
        self.assertTrue((b0 >= 0).all())                                                 # unsigned

    def test_mutated_block_that_reads_column_zero_is_detected(self):
        _, _, label, g15 = synth()
        bad = np.abs(g15[..., [0] + list(range(2, 11))])                                 # the registered nosign block = the R54 leak
        r = AD.search(bad, label)
        self.assertTrue(r["flagged"])
        good = AD.search(C.clean_block(g15), label)
        self.assertFalse(good["flagged"])

    def test_clean_block_names_and_cols(self):
        self.assertEqual(C.BLOCK_COLS, tuple(range(2, 11)))
        self.assertEqual(len(C.BLOCK_NAMES), 9)


if __name__ == "__main__":
    unittest.main(verbosity=1)
