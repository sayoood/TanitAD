"""Synthetic CPU-only tests for EP-RELIAB-PICKS (no bank, no GPU, no real label).  Run: python -m unittest test_epreliab  (or pytest).

Every decision function has a RED mutation that must break the branch table; the planted quantities (gain, mass form, attribution, scale gate, reliability, reach, power, null) have planted cases
that a mutated copy must fail.  Registered source: PREREG_EP_RELIAB_PICKS.md + Amendment 1.
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epreliab_lib as L                                                                  # noqa: E402

P = L.load_pickrule()
F = L.load_fastneg()


def stats_for(log_idx, n_logs, B=500, seed=11):
    return P.Stats(np.asarray(log_idx, np.int64), F.boot_draws(n_logs, B, seed), F, n_logs=n_logs)


def synth_with(sig_fn=None, y_fn=None, frac=0.6, N=600, n_logs=12, seed=3):
    """synthetic bank; returns arr, ep_col, S, sig, y, pk_p, pk_e where y / sig can be replaced by functions of (sig, y, pk_p, pk_e)."""
    arr, ep, li, ch = L.synthetic_bank(N=N, n_logs=n_logs, seed=seed, frac_changed=frac)
    zE = arr["z6"].copy()
    zE[..., L.EP] = ep
    pk_p, pk_e = P.rule_pick(arr["z6"]), P.rule_pick(zE)
    sig = P.sigmoid64(zE[..., L.EP])
    y = arr["lab"][..., L.EP].astype(np.float64)
    if y_fn is not None:
        y = y_fn(sig, y, pk_p, pk_e)
    return arr, ep, stats_for(li, n_logs), sig, y, pk_p, pk_e


class TestDecisionFunctions(unittest.TestCase):
    def test_branch_table_passes_unmutated(self):
        self.assertEqual(L.branch_table_failures(L), [])

    def test_every_decision_function_has_a_red_mutation(self):
        for old, new in L.DECISION_MUTATIONS:
            m = L.mutated_copy(old, new)
            self.assertTrue(len(m.branch_table_failures(m)) > 0, f"mutation not caught: {old!r} -> {new!r}")

    def test_mutation_report_consistent(self):
        r = L.mutation_report()
        self.assertEqual(r["unmutated_failures"], [])
        self.assertEqual(r["n_caught"], r["n_mutations"])
        self.assertEqual(r["uncaught"], [])
        self.assertGreaterEqual(r["n_mutations"], 20)

    def test_constants_are_the_registered_ones(self):
        self.assertEqual((L.SCALE_LO, L.SCALE_HI, L.PLANT), (0.9, 1.1, 0.10))
        self.assertEqual((L.BOOT_B, L.BOOT_SEED, L.N_LOGS), (10000, 20260927, 117))
        self.assertEqual((L.NULL_C_REPS, L.NULL_E_REPS, L.NULL_SALT, L.NULL_SEED_BASE, L.NULL_VOID_FIRES), (5, 200, 7919, 20261019, 2))
        self.assertEqual(L.TERMS, ("NC", "DAC", "EP", "TTC", "C"))
        self.assertEqual(L.v1_w(), (5.0, 5.0, 2.0))
        self.assertEqual((L.NC, L.DAC, L.EP, L.TTC, L.CMF), (0, 1, 2, 3, 4))


class TestPins(unittest.TestCase):
    def test_tampered_landed_module_refused(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(L.REF, "pickrule_lib.py")
            dst = os.path.join(d, "pickrule_lib.py")
            shutil.copyfile(src, dst)
            with open(dst, "ab") as f:
                f.write(b"\n# tampered\n")
            with self.assertRaises(L.EpReliabError):
                L.load_pinned("pickrule_lib_tamper_probe", [dst], L.PICKRULE_LIB_SHA256)
            sys.modules.pop("pickrule_lib_tamper_probe", None)

    def test_missing_module_refused(self):
        with self.assertRaises(L.EpReliabError):
            L.load_pinned("nothing_here", [os.path.join(HERE, "no_such_file.py")], "0" * 64)

    def test_prereg_modified_copy_refused_and_unregistered_amendment_refused(self):
        root = os.path.join(HERE, "..")
        with tempfile.TemporaryDirectory() as d:
            for n in ("PREREG_EP_RELIAB_PICKS.md", "PREREG_EP_RELIAB_PICKS_AMENDMENT1.md"):
                shutil.copyfile(os.path.join(root, n), os.path.join(d, n))
            self.assertEqual(L.require_prereg(os.path.join(d, "PREREG_EP_RELIAB_PICKS.md")), L.PREREG_SHA256)
            with open(os.path.join(d, "PREREG_EP_RELIAB_PICKS_AMENDMENT2.md"), "w", encoding="utf-8") as f:
                f.write("an amendment nobody registered")
            with self.assertRaises(L.EpReliabError):
                L.require_prereg(os.path.join(d, "PREREG_EP_RELIAB_PICKS.md"))
            os.remove(os.path.join(d, "PREREG_EP_RELIAB_PICKS_AMENDMENT2.md"))
            with open(os.path.join(d, "PREREG_EP_RELIAB_PICKS.md"), "ab") as f:
                f.write(b"\nmoved goalpost\n")
            with self.assertRaises(L.EpReliabError):
                L.require_prereg(os.path.join(d, "PREREG_EP_RELIAB_PICKS.md"))

    def test_the_real_prereg_hashes_to_the_sealed_value(self):
        self.assertEqual(L.require_prereg(), L.PREREG_SHA256)


class TestGainHandValues(unittest.TestCase):
    def setUp(self):
        # 4 sets of 3 proposals; P2 picks a, the composite picks b
        self.sig = np.array([[0.9, 0.5, 0.2], [0.8, 0.3, 0.6], [0.4, 0.7, 0.1], [0.5, 0.5, 0.9]])
        self.y = np.array([[0.7, 0.4, 0.1], [0.9, 0.2, 0.5], [0.3, 0.8, 0.2], [0.6, 0.1, 0.5]])
        self.pk_p = np.array([0, 0, 1, 2])
        self.pk_e = np.array([1, 2, 1, 0])          # set 2: no change
        self.S = stats_for([0, 0, 1, 1], 2, B=200)

    def test_gain_terms_against_a_loop(self):
        t = L.pair_terms(self.sig, self.y, self.pk_p, self.pk_e)
        for i in range(4):
            self.assertAlmostEqual(t["ghat"][i], self.sig[i, self.pk_e[i]] - self.sig[i, self.pk_p[i]], 15)
            self.assertAlmostEqual(t["g"][i], self.y[i, self.pk_e[i]] - self.y[i, self.pk_p[i]], 15)
        self.assertEqual(t["changed"].tolist(), [True, True, False, True])
        cs = L.core_stats(self.S, t)
        ghat = [0.5 - 0.9, 0.6 - 0.8, 0.0, 0.5 - 0.9]
        g = [0.4 - 0.7, 0.5 - 0.9, 0.0, 0.6 - 0.5]
        self.assertAlmostEqual(cs["delta"]["point"], (sum(ghat) - sum(g)) / 4, 12)
        # mass form by hand
        ye, se = [0.4, 0.5, 0.8, 0.6], [0.5, 0.6, 0.7, 0.5]
        yp, sp = [0.7, 0.9, 0.8, 0.5], [0.9, 0.8, 0.7, 0.9]
        self.assertAlmostEqual(cs["D_M"]["point"], sum(yp) / sum(sp) - sum(ye) / sum(se), 12)

    def test_delta_is_pred_minus_real(self):
        t = L.pair_terms(self.sig, self.y, self.pk_p, self.pk_e)
        f = L.full_gain_stats(self.S, t)
        self.assertAlmostEqual(f["delta"]["point"], f["pred_gain"]["point"] - f["real_gain"]["point"], 12)
        self.assertEqual(f["n_changed"], 3)
        self.assertAlmostEqual(f["changed_only"]["delta"]["point"], (t["ghat"].sum() - t["g"].sum()) / 3, 12)

    def test_mass_form_sign_is_invariant_to_a_global_rescale_but_delta_is_not(self):
        t1 = L.pair_terms(self.sig, self.y, self.pk_p, self.pk_e)
        c1 = L.core_stats(self.S, t1)
        for k in (0.5, 2.0, 3.7):
            tk = L.pair_terms(self.sig * k, self.y, self.pk_p, self.pk_e)
            ck = L.core_stats(self.S, tk)
            self.assertEqual(np.sign(ck["D_M"]["point"]), np.sign(c1["D_M"]["point"]))
            self.assertEqual(L.stat_class(ck["D_M"]), L.stat_class(c1["D_M"]))
            self.assertAlmostEqual(ck["D_M"]["point"], c1["D_M"]["point"] / k, 12)
        t2 = L.pair_terms(self.sig * 2.0, self.y, self.pk_p, self.pk_e)
        self.assertNotAlmostEqual(L.core_stats(self.S, t2)["delta"]["point"], c1["delta"]["point"], 6)


class TestAttribution(unittest.TestCase):
    W = (5.0, 5.0, 2.0)

    def lab_pair(self, a, b):
        return np.array([a], np.float64), np.array([b], np.float64)

    def test_single_term_goes_entirely_to_that_term(self):
        base = [1.0, 1.0, 0.6, 1.0, 0.8]
        for k, newv in enumerate([0.0, 0.0, 0.9, 0.0, 0.1]):
            b = list(base)
            b[k] = newv
            A, B = self.lab_pair(base, b)
            contrib, total = L.attribute(A, B, self.W)
            for j in range(5):
                if j != k:
                    self.assertEqual(float(contrib[0, j]), 0.0, f"term {L.TERMS[j]} must be exactly 0 when only {L.TERMS[k]} differs")
            self.assertAlmostEqual(float(contrib[0, k]), float(total[0]), 15)
            self.assertNotEqual(float(total[0]), 0.0)
            phi = L.shapley(A, B, self.W)
            self.assertAlmostEqual(float(phi[0, k]), float(total[0]), 14)
            self.assertEqual([float(phi[0, j]) for j in range(5) if j != k], [0.0] * 4)

    def test_two_term_hand_values_and_order_dependence(self):
        # A: NC 1, DAC 1, EP .5, TTC 1, C 1 ; B: NC 0, DAC 1, EP .9, TTC 1, C 1
        A, B = self.lab_pair([1, 1, 0.5, 1, 1], [0, 1, 0.9, 1, 1])
        pdA = 1 * 1 * (5 * 0.5 + 5 * 1 + 2 * 1) / 12
        c_nc = 0 * 1 * (5 * 0.5 + 5 + 2) / 12 - pdA                 # NC first: the whole of A's PDMS is lost
        c_ep = 0.0                                                   # EP is valued at NC = 0: zero
        contrib, total = L.attribute(A, B, self.W)
        self.assertAlmostEqual(float(contrib[0, 0]), c_nc, 14)
        self.assertAlmostEqual(float(contrib[0, 2]), c_ep, 14)
        self.assertAlmostEqual(float(total[0]), -pdA, 14)
        # a different order changes the split but not the total: EP first is valued at A's NC = 1
        contrib2, total2 = L.attribute(A, B, self.W, order=(2, 0, 1, 3, 4))
        self.assertAlmostEqual(float(contrib2[0, 2]), 5 * 0.4 / 12, 14)
        self.assertAlmostEqual(float(contrib2[0, 0]), -(1 * (5 * 0.9 + 5 + 2) / 12), 14)
        self.assertAlmostEqual(float(total2[0]), float(total[0]), 14)
        # Shapley is the average over orders: for two players it is the mean of the two orders
        phi = L.shapley(A, B, self.W)
        self.assertAlmostEqual(float(phi[0, 0]), 0.5 * (float(contrib[0, 0]) + float(contrib2[0, 0])), 14)
        self.assertAlmostEqual(float(phi[0, 2]), 0.5 * (float(contrib[0, 2]) + float(contrib2[0, 2])), 14)
        self.assertAlmostEqual(float(phi.sum()), float(total[0]), 14)

    def test_random_telescoping_and_shapley_efficiency(self):
        rng = np.random.default_rng(0)
        A = rng.choice([0.0, 0.5, 1.0], size=(500, 5))
        B = rng.choice([0.0, 0.5, 1.0], size=(500, 5))
        A[:, 2:] = rng.uniform(size=(500, 3))
        B[:, 2:] = rng.uniform(size=(500, 3))
        contrib, total = L.attribute(A, B, self.W)
        phi = L.shapley(A, B, self.W)
        self.assertLessEqual(float(np.abs(contrib.sum(1) - total).max()), 1e-12)
        self.assertLessEqual(float(np.abs(phi.sum(1) - total).max()), 1e-12)

    def test_attribution_total_equals_the_landed_formula_difference(self):
        arr, ep, S, sig, y, pk_p, pk_e = synth_with()
        lab = arr["lab"]
        idx = np.arange(len(pk_p))
        contrib, total = L.attribute(lab[idx, pk_p], lab[idx, pk_e])
        pd = P.pdms_from_labels(lab)
        want = P.at(pd, pk_e).astype(np.float64) - P.at(pd, pk_p).astype(np.float64)
        self.assertLessEqual(float(np.abs(total - want).max()), 1e-6)

    def test_red_mutants_of_the_attribution_are_caught(self):
        A, B = self.lab_pair([1, 1, 0.5, 0.0, 0.2], [0.5, 0, 0.9, 1.0, 0.9])
        want, tw = L.attribute(A, B, self.W)
        phiw = L.shapley(A, B, self.W)
        for old, new in [("+ w[2] * t[:, 4]) / float(sum(w))", "+ w[2] * t[:, 3]) / float(sum(w))"), ("/ float(sum(w))", "/ 10.0"), ("for k in order:", "for k in reversed(order):"),
                         ("cur[:, k] = B[:, k]\n        val = _pdms_terms(cur, w)\n        contrib[:, k] = val - prev", "cur[:, k] = B[:, k]\n        val = _pdms_terms(cur, w)\n        contrib[:, k] = prev - val")]:
            m = L.mutated_copy(old, new)
            got, tg = m.attribute(A, B, self.W)
            self.assertFalse(np.allclose(got, want) and np.allclose(tg, tw), f"mutation not caught: {old!r}")
        m = L.mutated_copy("/ math.factorial(5)", "/ math.factorial(4)")
        self.assertFalse(np.allclose(m.shapley(A, B, self.W), phiw))

    def test_dominant_term_and_shares_on_a_planted_loss(self):
        # all of the loss from the DAC multiplier: B keeps NC, loses DAC
        arr, ep, S, sig, y, pk_p, pk_e = synth_with(frac=0.6)
        lab = arr["lab"].copy()
        idx = np.arange(len(pk_p))
        lab[idx, pk_p] = np.array([1.0, 1.0, 0.5, 1.0, 1.0, 1.0], np.float32)
        lab[idx, pk_e] = np.array([1.0, 1.0, 0.5, 1.0, 1.0, 1.0], np.float32)
        lab[np.nonzero(pk_p != pk_e)[0], pk_e[pk_p != pk_e]] = np.array([1.0, 0.0, 0.5, 1.0, 1.0, 1.0], np.float32)
        at_ = L.attribution_stats(S, lab, pk_p, pk_e)
        self.assertEqual(at_["dominant_term"], "DAC")
        self.assertAlmostEqual(at_["share_of_loss"]["DAC"]["point"], 1.0, 12)
        for t in ("NC", "EP", "TTC", "C"):
            self.assertEqual(at_["sequential_x100"][t]["point"], 0.0)
        self.assertGreater(at_["dominant_bootstrap_frequency"]["DAC"], 0.99)
        self.assertLessEqual(at_["max_abs_seq_sum_minus_total_per_set"], 1e-12)

    def test_signed_ratio_with_a_negative_denominator(self):
        num = np.array([-1.0, -2.0, -3.0])
        den = np.array([-2.0, -2.0, -4.0])
        draws = np.array([[0, 1, 2], [0, 0, 1], [2, 2, 1]])
        ci = L.signed_ratio_ci(num, den, draws)
        self.assertAlmostEqual(ci["point"], 6.0 / 8.0, 14)
        v = [num[d].sum() / den[d].sum() for d in draws]
        self.assertAlmostEqual(ci["lo"], float(np.percentile(v, 2.5)), 14)
        self.assertAlmostEqual(ci["hi"], float(np.percentile(v, 97.5)), 14)
        self.assertEqual(ci["n_nan"], 0)
        z = L.signed_ratio_ci(np.zeros(3), np.zeros(3), draws)
        self.assertTrue(math.isnan(z["point"]) and z["n_nan"] == 3)


class TestScaleGate(unittest.TestCase):
    def slope(self, kind):
        x, y, sf, li, nl = L.planted_slope_bank(kind)
        return L.scale_slope(stats_for(li, nl, B=300), x, y, sf)

    def test_planted_on_and_off(self):
        self.assertTrue(L.scale_on(self.slope("exact")))
        self.assertTrue(L.scale_on(self.slope("ok_1.05")))
        self.assertFalse(L.scale_on(self.slope("half")))
        self.assertFalse(L.scale_on(self.slope("high_1.2")))
        self.assertAlmostEqual(self.slope("half")["point"], 0.5, 3)

    def test_gate_needs_both_seeds(self):
        on, off = self.slope("exact"), self.slope("half")
        self.assertTrue(L.scale_gate([on, on]))
        self.assertFalse(L.scale_gate([on, off]))
        self.assertFalse(L.scale_gate([off, on]))

    def test_global_offset_per_set_offset_and_unsafe_proposals(self):
        rng = np.random.default_rng(1)
        N, n_logs = 400, 10
        x = rng.uniform(0.05, 0.95, (N, 64))
        li = rng.integers(0, n_logs, size=N)
        S = stats_for(li, n_logs, B=300)
        safe = np.ones((N, 64), bool)
        base = L.scale_slope(S, x, x.copy(), safe)["point"]
        # a per-set offset on sigma (y = x, sigma = x + c_i): the WITHIN-set slope is untouched, a pooled slope would collapse
        c = rng.normal(0.0, 1.0, (N, 1))
        self.assertAlmostEqual(L.scale_slope(S, x + c, x.copy(), safe)["point"], base, 9)
        m = L.mutated_copy("    sc = (sig - (sig * m).sum(axis=1, keepdims=True) / n) * m\n    yc = (y - (y * m).sum(axis=1, keepdims=True) / n) * m",
                           "    sc = sig * m\n    yc = y * m")
        self.assertLess(m.scale_slope(S, x + c, x.copy(), safe)["point"], 0.5)                 # the pooled (level-dependent) mutant collapses
        # unsafe proposals carry garbage labels: they must be excluded
        unsafe = rng.random((N, 64)) < 0.3
        y_bad = np.where(unsafe, rng.uniform(0, 1, (N, 64)), x)
        self.assertAlmostEqual(L.scale_slope(S, x, y_bad, ~unsafe)["point"], 1.0, 9)
        m2 = L.mutated_copy("    m = safe.astype(np.float64)", "    m = np.ones_like(safe, dtype=np.float64)")
        self.assertLess(m2.scale_slope(S, x, y_bad, ~unsafe)["point"], 0.95)
        # a set with fewer than 2 safe proposals contributes nothing
        sf = np.zeros((N, 64), bool)
        sf[:, 0] = True
        ci = L.scale_slope(S, x, rng.uniform(0, 1, (N, 64)), sf)
        self.assertTrue(math.isnan(ci["point"]) or ci["point"] == 0.0)


class TestReliability(unittest.TestCase):
    def test_hand_values_and_a_swapped_ratio_mutant(self):
        N = 400
        rng = np.random.default_rng(2)
        li = rng.integers(0, 8, size=N)
        S = stats_for(li, 8, B=300)
        z6 = rng.normal(0, 1, (N, 64, 6)).astype(np.float32)
        lab = rng.uniform(0, 1, (N, 64, 6)).astype(np.float32)
        pk_p = rng.integers(0, 64, N)
        pk_e = rng.integers(0, 64, N)
        r = L.reliability_stats(S, lab, z6, z6, pk_p, pk_e)
        for name, col in (("NC", 0), ("DAC", 1), ("EP", 2), ("TTC", 3)):
            y = lab[np.arange(N), pk_e, col].astype(np.float64)
            s = 1 / (1 + np.exp(-z6[np.arange(N), pk_e, col].astype(np.float64)))
            self.assertAlmostEqual(r[name]["composite_picks"]["positive_form_sum_y_over_sum_sigma"]["point"], y.sum() / s.sum(), 12)
            self.assertAlmostEqual(r[name]["composite_picks"]["deficit_form_sum_1my_over_sum_1msigma"]["point"], (1 - y).sum() / (1 - s).sum(), 12)
        self.assertEqual(r["EP"]["primary"], "positive")
        self.assertEqual(r["NC"]["primary"], "deficit")
        m = L.mutated_copy('"deficit_form_sum_1my_over_sum_1msigma": S.ratio(1.0 - ye, 1.0 - se),', '"deficit_form_sum_1my_over_sum_1msigma": S.ratio(1.0 - se, 1.0 - ye),')
        rm = m.reliability_stats(S, lab, z6, z6, pk_p, pk_e)
        self.assertNotAlmostEqual(rm["NC"]["composite_picks"]["deficit_form_sum_1my_over_sum_1msigma"]["point"], r["NC"]["composite_picks"]["deficit_form_sum_1my_over_sum_1msigma"]["point"], 3)

    def test_planted_obs_over_pred(self):
        # y = 0.8 sigma at every pick -> the positive form reads 0.8 and the EP paired difference is 0 when both pick populations are scaled alike
        N = 300
        rng = np.random.default_rng(4)
        li = rng.integers(0, 6, size=N)
        S = stats_for(li, 6, B=200)
        z6 = rng.normal(0, 1, (N, 64, 6)).astype(np.float32)
        sig = 1 / (1 + np.exp(-z6.astype(np.float64)))
        lab = (0.8 * sig).astype(np.float32)
        pk_p, pk_e = rng.integers(0, 64, N), rng.integers(0, 64, N)
        r = L.reliability_stats(S, lab, z6, z6, pk_p, pk_e)
        self.assertAlmostEqual(r["EP"]["composite_picks"]["positive_form_sum_y_over_sum_sigma"]["point"], 0.8, 5)
        self.assertAlmostEqual(r["EP"]["P2_picks"]["positive_form_sum_y_over_sum_sigma"]["point"], 0.8, 5)
        self.assertAlmostEqual(r["EP"]["paired_composite_minus_P2_primary"]["point"], 0.0, 5)


class TestReachPowerNull(unittest.TestCase):
    def test_synthetic_bank_is_as_designed(self):
        arr, ep, li, ch = L.synthetic_bank()
        zE = arr["z6"].copy()
        zE[..., L.EP] = ep
        pk_p, pk_e = P.rule_pick(arr["z6"]), P.rule_pick(zE)
        self.assertTrue((pk_p == 0).all())
        self.assertTrue(np.array_equal(pk_e == 1, ch))
        self.assertTrue(0 < ch.sum() < len(ch))

    def test_reach_check_passes_and_red_mutants_fail(self):
        arr, ep, S, sig, y, pk_p, pk_e = synth_with()
        r = L.reach_check(S, sig, y, pk_p, pk_e)
        self.assertTrue(r["pass"], r)
        self.assertAlmostEqual(r["label_delta_move"], 0.07 * r["n_changed"] / r["N"], 12)
        self.assertAlmostEqual(r["sigma_delta_move"], 0.05 * r["n_changed"] / r["N"], 12)
        self.assertAlmostEqual(r["label_changed_only_delta_move"], 0.07, 12)
        for old, new in [("out[ch, pk_e[ch]] -= d", "out[ch, pk_p[ch]] -= d"), ("out[ch, pk_e[ch]] -= d", "out[ch, pk_e[ch]] += d"),
                         ("out[ch, pk_e[ch]] += s", "out[ch, pk_p[ch]] += s"), ("out[ch, pk_e[ch]] += s", "out[ch, pk_e[ch]] -= s")]:
            m = L.mutated_copy(old, new)
            self.assertFalse(m.reach_check(S, sig, y, pk_p, pk_e)["pass"], f"mutation not caught: {old!r} -> {new!r}")

    def test_p2_vs_p2_reads_exactly_zero(self):
        arr, ep, S, sig, y, pk_p, pk_e = synth_with()
        res = L.analyze_seed(arr, arr["z6"][..., L.EP], S)
        G, A = res["G"], res["attribution"]
        self.assertEqual(res["n_changed"], 0)
        for ci in [G["delta"], G["pred_gain"], G["real_gain"], G["D_M"], A["total_x100"]] + list(A["sequential_x100"].values()) + list(A["shapley_x100"].values()):
            self.assertEqual((float(ci["point"]), float(ci["lo"]), float(ci["hi"])), (0.0, 0.0, 0.0))
        self.assertEqual(G["M_e"]["point"], G["M_p"]["point"])
        # and the composite is NOT identically zero (the identity is not vacuous)
        res2 = L.analyze_seed(arr, ep, S)
        self.assertNotEqual(res2["G"]["delta"]["point"], 0.0)
        self.assertGreater(res2["n_changed"], 0)

    def test_c_pdms_and_safe_mask_equal_epr1(self):
        E = L.load_epr1()
        arr, ep, li, ch = L.synthetic_bank()
        self.assertTrue(np.array_equal(P.pdms_from_labels(arr["lab"]), E.pdms_np(arr["lab"])))
        self.assertTrue(np.array_equal(L.safe_mask(arr["lab"]), E.safe_np(arr["lab"])))

    def _neutral_power(self, mod=L):
        arr, ep, S, sig, y, pk_p, pk_e = synth_with()
        return mod.power_check(S, sig, y, pk_p, pk_e), (S, sig, y, pk_p, pk_e)

    def test_power_check_fires_on_the_neutral_bank_and_a_red_mutant_fails(self):
        r, (S, sig, y, pk_p, pk_e) = self._neutral_power()
        self.assertTrue(r["pass"], r)
        for f in ("DELTA", "MASS"):
            self.assertEqual(r[f]["plus"]["class"], "OVER")
            self.assertEqual(r[f]["minus"]["class"], "UNDER")
            self.assertEqual(r[f]["zero"]["class"], "WITHIN")
            self.assertTrue(r[f]["zero_is_exactly_0_0"])
        # the neutral bank is independent of the real labels: wrecking every label at the picks changes nothing
        y2 = y.copy()
        y2[np.arange(len(pk_p)), pk_e] = 123.0
        y2[np.arange(len(pk_p)), pk_p] = -77.0
        r2 = L.power_check(S, sig, y2, pk_p, pk_e)
        self.assertEqual(r2["DELTA"]["plus"]["ci"]["point"], r["DELTA"]["plus"]["ci"]["point"])
        self.assertEqual(r2["MASS"]["minus"]["ci"]["point"], r["MASS"]["minus"]["ci"]["point"])
        for old, new in [('if ci["lo"] > 0:\n        return "OVER"', 'if ci["lo"] > 10:\n        return "OVER"'), ('if ci["hi"] < 0:\n        return "UNDER"', 'if ci["hi"] < -10:\n        return "UNDER"'),
                         ("out[ch, pk_e[ch]] += s", "out[ch, pk_e[ch]] += 0.0 * s")]:
            m = L.mutated_copy(old, new)
            self.assertFalse(m.power_check(S, sig, y, pk_p, pk_e)["pass"], f"mutation not caught: {old!r}")
        m = L.mutated_copy("    out[idx, pk_e] = sig[idx, pk_e]\n    return out", "    return out")
        self.assertFalse(m.power_check(S, sig, y, pk_p, pk_e)["pass"], "a neutral bank that is not neutral must fail the zero arm")

    def test_shuffle_preserves_tokens_bijection_and_coincidence(self):
        arr, ep, S, sig, y, pk_p, pk_e = synth_with()
        a, b = L.shuffled_pair(pk_p, pk_e, 0)
        self.assertTrue(np.array_equal(a == b, pk_p == pk_e))
        self.assertTrue(((a >= 0) & (a < 64)).all())
        self.assertFalse(np.array_equal(a, pk_p))
        a2, b2 = L.shuffled_pair(pk_p, pk_e, 0)
        self.assertTrue(np.array_equal(a, a2) and np.array_equal(b, b2))
        a3, _ = L.shuffled_pair(pk_p, pk_e, 1)
        self.assertFalse(np.array_equal(a, a3))
        # uniform over the 64 proposals, per set
        counts = np.bincount(a, minlength=64)
        self.assertLess(counts.max(), 40)

    def test_null_is_silent_on_an_informative_bank_and_fires_when_the_link_is_not_broken(self):
        # a bank whose real picks carry a large, undelivered predicted gain: Delta is clearly OVER on the real picks
        arr, ep, S, sig, y, pk_p, pk_e = synth_with(y_fn=lambda sig, y, pp, pe: np.where(np.arange(64)[None, :] == 1, 0.0, y))
        real = L.core_stats(S, L.pair_terms(sig, y, pk_p, pk_e))
        self.assertEqual(L.stat_class(real["delta"]), "OVER")
        eps = {0: ep, 1: ep}
        null = L.run_null(arr, eps, S, "DELTA", reps_c=5, reps_e=20, real_points={0: real["delta"]["point"], 1: real["delta"]["point"]})
        # the registered rule tolerates a chance fire: VOID needs 2 or more of 5 (the rule's own false-fire rate is about 0.05 per replicate, so 0 or 1 is expected; 12 logs here, 117 on the real bank)
        self.assertLess(null["c_null"]["fires"], 2, "shuffled picks fired the registered rule on 2 or more of 5 replicates")
        self.assertFalse(null["c_null"]["void"])
        self.assertLess(null["e_null"]["empirical_any_fire_rate"], 0.2)
        self.assertLess(null["e_null"]["per_seed"]["0"]["empirical_two_sided_p_with_plus_one"], 0.1)       # the real Delta is far in the null's tail
        # a null that does NOT break the pick -> label link (the pick is left where it was) fires on every replicate and voids the arm
        m = L.mutated_copy("    return perm[idx, pk_p], perm[idx, pk_e]", "    return pk_p, pk_e")
        bad = m.run_null(arr, eps, S, "DELTA", reps_c=5, reps_e=0)
        self.assertEqual(bad["c_null"]["fires"], 5)
        self.assertTrue(bad["c_null"]["void"])

    def test_e_null_summary_keys(self):
        arr, ep, S, sig, y, pk_p, pk_e = synth_with()
        null = L.run_null(arr, {0: ep, 1: ep}, S, "MASS", reps_c=5, reps_e=10, real_points={0: 0.1, 1: 0.1})
        e = null["e_null"]
        for k in ("n", "empirical_any_fire_rate", "p_at_least_2_of_5_fire_at_that_rate", "per_seed", "other_form_any_fire_rate"):
            self.assertIn(k, e)
        self.assertIn("empirical_two_sided_p_with_plus_one", e["per_seed"]["0"])
        self.assertEqual(len(null["c_null"]["replicates"]), 5)
        self.assertEqual(null["c_null"]["replicates"][0]["perm_seed"], 20261019)
        self.assertEqual(null["c_null"]["replicates"][1]["perm_seed"], 20261019 + 7919)


class TestEndToEndDecision(unittest.TestCase):
    """The full per-seed analysis + the registered decision on planted banks where the answer is known."""

    def run_case(self, y_fn):
        arr, ep, S, sig, y, pk_p, pk_e = synth_with(y_fn=y_fn)
        arr2 = {"z6": arr["z6"], "lab": arr["lab"].astype(np.float64)}
        arr2["lab"][..., L.EP] = y
        per = {s: L.analyze_seed(arr2, ep, S) for s in (0, 1)}
        return L.decide(per), per

    def test_calibrated_bank_is_within_noise_on_scale(self):
        dec, per = self.run_case(lambda sig, y, pp, pe: sig.copy())
        self.assertTrue(dec["scale_gate_on_scale"], per[0]["scale"])
        self.assertEqual(dec["deciding_form"], "DELTA")
        self.assertEqual(dec["outcome"], "EP-WITHIN-NOISE")
        self.assertEqual(per[0]["G"]["delta"]["point"], 0.0)

    def test_undelivered_gain_is_overpredicted_in_both_forms(self):
        def y_fn(sig, y, pp, pe):
            out = sig.copy()
            ch = np.nonzero(pp != pe)[0]
            out[ch, pe[ch]] -= 0.3
            return out
        dec, per = self.run_case(y_fn)
        self.assertEqual(dec["outcomes"]["DELTA"], "EP-OVERPREDICTED")
        self.assertEqual(dec["outcomes"]["MASS"], "EP-OVERPREDICTED")
        self.assertEqual(dec["outcome"], "EP-OVERPREDICTED")
        self.assertFalse(dec["form_sensitive"])
        self.assertAlmostEqual(per[0]["G"]["delta"]["point"], 0.3 * per[0]["n_changed"] / per[0]["n_sets"], 9)

    def test_delivered_extra_gain_is_underpredicted_in_both_forms(self):
        def y_fn(sig, y, pp, pe):
            out = sig.copy()
            ch = np.nonzero(pp != pe)[0]
            out[ch, pe[ch]] += 0.2
            return out
        dec, per = self.run_case(y_fn)
        self.assertEqual(dec["outcomes"]["DELTA"], "EP-UNDERPREDICTED")
        self.assertEqual(dec["outcomes"]["MASS"], "EP-UNDERPREDICTED")

    def test_a_rescaled_row_flips_delta_but_not_the_mass_form(self):
        # the row over-states its spread by 2 (off-scale) while the realised gain equals the predicted gain / 2: Delta reads OVER, the gate says OFF-SCALE, the mass form decides
        arr, ep, S, sig, y, pk_p, pk_e = synth_with()
        y = (sig * 0.5)
        arr2 = {"z6": arr["z6"], "lab": arr["lab"].astype(np.float64)}
        arr2["lab"][..., L.EP] = y
        per = {s: L.analyze_seed(arr2, ep, S) for s in (0, 1)}
        dec = L.decide(per)
        self.assertEqual(dec["outcomes"]["DELTA"], "EP-OVERPREDICTED")
        self.assertFalse(dec["scale_gate_on_scale"])
        self.assertEqual(dec["deciding_form"], "MASS")
        self.assertEqual(dec["outcomes"]["MASS"], "EP-WITHIN-NOISE")
        self.assertEqual(dec["outcome"], "EP-WITHIN-NOISE")
        self.assertTrue(dec["form_sensitive"])

    def test_seeds_that_disagree_are_within(self):
        arr, ep, S, sig, y, pk_p, pk_e = synth_with(y_fn=lambda sig, y, pp, pe: sig.copy())
        arr2 = {"z6": arr["z6"], "lab": arr["lab"].astype(np.float64)}
        arr2["lab"][..., L.EP] = sig
        per = {0: L.analyze_seed(arr2, ep, S), 1: L.analyze_seed(arr2, ep, S)}
        per[1]["G"]["delta"] = {"lo": 0.01, "hi": 0.2, "point": 0.1}
        per[0]["G"]["delta"] = {"lo": -0.2, "hi": 0.2, "point": 0.0}
        dec = L.decide(per)
        self.assertEqual(dec["outcomes"]["DELTA"], "EP-WITHIN-NOISE")


if __name__ == "__main__":
    unittest.main()
