"""Synthetic CPU tests for ep_admission_sim.py: the generator is calibrated by construction, the plant is exact, the per-log shortcut IS the library estimator, and the controls have teeth (RED arms).
Run: python -I test_ep_admission_sim.py   (B = 500 draws here; the simulation itself uses the registered B = 10,000)."""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ep_admission_sim as S                                                                  # noqa: E402

A = S.A
L, P, F = A.libs()
sys.path.insert(0, S.ADM_DIR)
import test_ep_admission as T                                                                 # noqa: E402  (the synthetic bank generator of the gate's own tests)

DRAWS = F.boot_draws(S.N_LOGS, 500, S.BOOT_SEED)


def synth(over=0.0, seed=7, **kw):
    z6_p2, z6_c, lab, li, ch = T.make_case(over=over, n=3510, n_logs=S.N_LOGS, seed=seed, **kw)
    comp = S.components(z6_p2, z6_c, lab, li)
    return comp, (z6_p2, z6_c, lab, li)


def cond(v, kind, name="x", a1=True, a2=True, gen="A", **kw):
    return {"name": name, "gen": gen, "geom": "syn", "a1": a1, "a2": a2, "kind": kind, "vec": {0: S.build_condition(v, kind, **kw), 1: S.build_condition(v, kind, **kw)}}


class TestHelpers(unittest.TestCase):
    def test_wilson(self):
        lo, hi = S.wilson(0, 200)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.0188, places=3)
        lo, hi = S.wilson(10, 200)
        self.assertAlmostEqual(lo, 0.0273, places=3)
        self.assertAlmostEqual(hi, 0.0896, places=3)
        self.assertTrue(np.isnan(S.wilson(0, 0)[0]))
        lo, hi = S.wilson(200, 200)
        self.assertAlmostEqual(hi, 1.0, places=12)

    def test_outer_draws_are_deterministic_and_distinct(self):
        self.assertTrue(np.array_equal(S.outer_idx(3), S.outer_idx(3)))
        self.assertFalse(np.array_equal(S.outer_idx(3), S.outer_idx(4)))
        self.assertEqual(S.outer_idx(0).shape, (S.N_LOGS,))
        self.assertTrue(set(np.unique(S.outer_idx(0))) <= set(range(S.N_LOGS)))
        self.assertTrue(set(np.unique(S.wild_signs(0))) <= {-1.0, 1.0})


class TestGeneratorIsCalibratedAndThePlantIsExact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comp, cls.raw = synth(over=0.13)

    def test_calibrated_by_construction(self):
        d = S.build_condition(self.comp["vec"], "cal")
        self.assertAlmostEqual(d["Yp"].sum() / d["Sp"].sum(), 1.0, places=12)
        self.assertAlmostEqual(d["Ye"].sum() / d["Se"].sum(), 1.0, places=12)
        for h in S.HEADS:
            self.assertAlmostEqual(d[f"OBS_{h}"].sum() / d[f"PRED_{h}"].sum(), 1.0, places=12)

    def test_the_real_data_it_is_built_from_is_not_calibrated(self):
        r = S.build_condition(self.comp["vec"], "raw")
        self.assertGreater(r["Yp"].sum() / r["Sp"].sum() - r["Ye"].sum() / r["Se"].sum(), 0.10)                  # the planted over=0.13 is in the raw data, and calibration removes it

    def test_the_ep_plant_sets_d_m_to_delta_exactly(self):
        for delta in (0.0, 0.05, 0.135):
            d = S.build_condition(self.comp["vec"], "plant_ep", delta=delta)
            self.assertAlmostEqual(d["Yp"].sum() / d["Sp"].sum() - d["Ye"].sum() / d["Se"].sum(), delta, places=12)
            self.assertAlmostEqual(d["Yp"].sum() / d["Sp"].sum(), 1.0, places=12)

    def test_the_a2_plant_sets_the_ratio_to_rho_on_that_head_only(self):
        d = S.build_condition(self.comp["vec"], "plant_a2", head="NC", rho=1.5)
        self.assertAlmostEqual(d["OBS_NC"].sum() / d["PRED_NC"].sum(), 1.5, places=12)
        self.assertAlmostEqual(d["OBS_DAC"].sum() / d["PRED_DAC"].sum(), 1.0, places=12)

    def test_the_split_components_add_up_to_the_direct_vectors(self):
        v = self.comp["vec"]
        for a, b in (("y_p", ("Yu", "Ypc")), ("s_p", ("Su", "Spc")), ("y_e", ("Yu", "Yec")), ("s_e", ("Su", "Sec"))):
            self.assertTrue(np.allclose(v[a], v[b[0]] + v[b[1]], rtol=0, atol=1e-12))


class TestShortcutIsTheEstimator(unittest.TestCase):
    def test_the_identity_draw_equals_the_library_pipeline_bit_for_bit(self):
        comp, (z6_p2, z6_c, lab, li) = synth(over=0.05)
        v = comp["vec"]
        St = P.Stats(li.astype(np.int64), DRAWS, F, n_logs=S.N_LOGS)
        pk_p, pk_c = P.rule_pick(z6_p2), P.rule_pick(z6_c)
        t = L.pair_terms(P.sigmoid64(z6_c[..., S.EP]), lab[..., S.EP].astype(np.float64), pk_p, pk_c)
        lib = L.core_stats(St, t)["D_M"]
        mine = F.diff_ci(v["y_p"], v["s_p"], v["y_e"], v["s_e"], DRAWS)
        self.assertTrue(S.eq_ci(lib, mine))
        rel = L.reliability_stats(St, lab, z6_c, z6_p2, pk_p, pk_c)
        for h in S.HEADS:
            self.assertTrue(S.eq_ci(rel[h]["composite_picks"]["deficit_form_sum_1my_over_sum_1msigma"], F.ratio_ci(v[f"OBS_{h}"], v[f"PRED_{h}"], DRAWS)))

    def test_a_replicate_is_the_estimator_on_the_drawn_logs(self):
        comp, _ = synth(over=0.05)
        c = cond(comp["vec"], "cal")
        rec = S.replicate(c, 5, DRAWS)
        idx = S.outer_idx(5)
        d = c["vec"][0]
        ci = F.diff_ci(d["Yp"][idx], d["Sp"][idx], d["Ye"][idx], d["Se"][idx], DRAWS)
        self.assertEqual(tuple(rec["dm"][0]), (ci["point"], ci["lo"], ci["hi"]))
        self.assertTrue(np.array_equal(rec["dm"][0], rec["dm"][1]))                      # both seeds are the same synthetic candidate here, on the SAME drawn logs

    def test_the_identity_candidate_reads_exactly_zero(self):
        comp, _ = synth()
        v = comp["vec"]
        d = {"Yp": v["y_p"], "Sp": v["s_p"], "Ye": v["y_p"], "Se": v["s_p"]}
        c = {"name": "I", "gen": "A", "geom": "syn", "a1": True, "a2": False, "kind": "identity", "vec": {0: d, 1: d}}
        for r in range(5):
            self.assertTrue((S.replicate(c, r, DRAWS)["dm"] == 0.0).all())


class TestControlsHaveTeeth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comp, _ = synth(over=0.13)
        cls.n = 60

    def reps(self, c):
        recs = [S.replicate(c, r, DRAWS) for r in range(self.n)]
        return S.summarize(c, {k: np.stack([x[k] for x in recs]) for k in ("dm", "a2", "mp", "me")})

    def test_pc1_passes_a_calibrated_generator_and_fails_an_uncalibrated_one(self):
        cal = self.reps(cond(self.comp["vec"], "cal"))
        self.assertTrue(S.pc1_calibrated(cal)["pass"], S.pc1_calibrated(cal))
        raw = self.reps(cond(self.comp["vec"], "raw"))                                    # RED arm: the raw bank carries the planted over-prediction
        self.assertFalse(S.pc1_calibrated(raw)["pass"])

    def test_pc2_separates_zero_from_delta(self):
        plant = self.reps(cond(self.comp["vec"], "plant_ep", delta=0.1, a2=False))
        self.assertTrue(S.pc2_planted(plant, 0.1)["pass"])
        self.assertFalse(S.pc1_calibrated(plant)["pass"])
        cal = self.reps(cond(self.comp["vec"], "cal"))
        self.assertFalse(S.pc2_planted(cal, 0.1)["pass"])

    def test_a_planted_over_prediction_is_refused_and_a_calibrated_candidate_mostly_is_not(self):
        plant = self.reps(cond(self.comp["vec"], "plant_ep", delta=0.1, a2=False))
        self.assertEqual(plant["A1"]["r1_any_seed"]["k"], self.n)
        cal = self.reps(cond(self.comp["vec"], "cal"))
        self.assertLess(cal["A1"]["r1_any_seed"]["rate"], 0.25)

    def test_pc4_catches_a_broken_decision_function(self):
        plant = cond(self.comp["vec"], "plant_ep", delta=0.1, a2=False)
        recs = [S.replicate(plant, r, DRAWS) for r in range(10)]
        rec = {k: np.stack([x[k] for x in recs]) for k in ("dm", "a2", "mp", "me")}
        self.assertEqual(S.summarize(plant, rec)["A1"]["pc4_inline_disagreements"], 0)
        with mock.patch.object(A, "a1_verdict_r1", lambda ci, *a, **k: "ADMIT-A1"):          # RED arm: a decision function that always admits
            self.assertGreater(S.summarize(plant, rec)["A1"]["pc4_inline_disagreements"], 0)

    def test_the_wild_generator_has_mean_calibration_and_differs_from_generator_a(self):
        calB = self.reps(cond(self.comp["vec"], "cal", gen="B", a2=False))
        self.assertTrue(S.pc1_calibrated(calB)["pass"], S.pc1_calibrated(calB))
        calA = self.reps(cond(self.comp["vec"], "cal", a2=False))
        self.assertFalse(np.array_equal(calA["A1"]["D_M_point_sd"], calB["A1"]["D_M_point_sd"]))


if __name__ == "__main__":
    unittest.main()
