"""Unit tests for gatemix.py on synthetic data (no dumps read)."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import gatemix as G  # noqa: E402


def synth(n_ep=60, per=20, seed=0, informative=True):
    """hierarchy better on critical windows, flat better on bulk; tick-0 probabilities confident of a non-common class on critical windows."""
    rng = np.random.default_rng(seed)
    eid = np.repeat(np.arange(n_ep), per)
    n = len(eid)
    crit = rng.random(n) < 0.4
    eh = np.where(crit, 1.0, 2.0) + 0.1 * rng.standard_normal(n)
    ef = np.where(crit, 2.0, 1.0) + 0.1 * rng.standard_normal(n)
    pl = np.full((n, G.NLAT), 0.01)
    po = np.full((n, G.NLON), 0.01)
    pl[:, 0] = 0.9
    po[:, 0] = 0.9                                   # common case (0,0)
    if informative:
        pl[crit, 0], pl[crit, 3] = 0.05, 0.9
        po[crit, 0], po[crit, 2] = 0.05, 0.9
    else:
        pl[crit, 0] = pl[crit, 0]                      # no signal in the probabilities
    pl, po = pl / pl.sum(1, keepdims=True), po / po.sum(1, keepdims=True)
    return eid, crit, eh, ef, pl, po


class T(unittest.TestCase):
    def test_g1_identities(self):
        eid, crit, eh, ef, pl, po = synth()
        fold = G.fold_assign(eid)
        cf = G.crossfit(eh, ef, pl, po, fold)
        S = cf["S"]
        cH, cF = {"ade6_m": eh}, {"ade6_m": ef}
        self.assertEqual(float(np.abs(G.compose(S >= S.min() - 1, cH, cF)["ade6_m"] - eh).max()), 0.0)
        self.assertEqual(float(np.abs(G.compose(S >= S.max() + 1, cH, cF)["ade6_m"] - ef).max()), 0.0)
        c = G.tau_candidates(S)
        self.assertEqual(len(c), 103)
        self.assertTrue(np.isneginf(c[0]) and np.isposinf(c[-1]))

    def test_crossfit_never_sees_heldout(self):
        eid, crit, eh, ef, pl, po = synth()
        fold = G.fold_assign(eid)
        base = G.crossfit(eh, ef, pl, po, fold)
        f = 2
        ho = fold == f
        eh2, ef2, pl2, po2 = eh.copy(), ef.copy(), pl.copy(), po.copy()
        eh2[ho], ef2[ho] = 50.0, -50.0                  # destroy the held-out errors
        pl2[ho] = np.roll(pl2[ho], 1, axis=1)           # and permute its probabilities
        po2[ho] = np.roll(po2[ho], 2, axis=1)
        alt = G.crossfit(eh2, ef2, pl2, po2, fold)
        self.assertEqual(base["tau"][ho][0], alt["tau"][ho][0])        # tau of fold f unchanged
        self.assertEqual(base["common"][f], alt["common"][f])
        self.assertTrue(len(np.unique(base["tau"][ho])) == 1)
        # a change in a TRAINING fold does move something (the check can fail)
        eh3 = eh.copy()
        eh3[fold == 0] = 50.0
        alt3 = G.crossfit(eh3, ef, pl, po, fold)
        self.assertNotEqual(base["tau"][ho][0], alt3["tau"][ho][0])
        # folds are by episode
        for e in np.unique(eid):
            self.assertEqual(len(np.unique(fold[eid == e])), 1)

    def test_planted_score_beats_both_and_shuffle_does_not(self):
        eid, crit, eh, ef, pl, po = synth()
        fold = G.fold_assign(eid)
        cf = G.crossfit(eh, ef, pl, po, fold)
        comp = np.where(cf["drive"], eh, ef)
        self.assertGreater(G.auroc(cf["S"], crit), 0.99)
        rh = G.H.boot(eid, {"d": G.H.mean_over(comp - eh)}, 300)["d"]
        rf = G.H.boot(eid, {"d": G.H.mean_over(comp - ef)}, 300)["d"]
        self.assertEqual(G.outcome_cell(rh, rf), "GATE-BEATS-BOTH")
        best = eh if eh.mean() <= ef.mean() else ef
        neg = 0
        for k in range(8):
            cs = G.crossfit(eh, ef, pl, po, fold, shuffle_seed=k)
            r = G.H.boot(eid, {"d": G.H.mean_over(np.where(cs["drive"], eh, ef) - best)}, 300)["d"]
            neg += int(G.sep_neg(r))
        self.assertLessEqual(neg, 1)

    def test_oracle_identity_g4(self):
        eid, crit, eh, ef, pl, po = synth()
        orc = G.compose(crit, {"ade6_m": eh}, {"ade6_m": ef})["ade6_m"]
        implied = (crit.sum() * eh[crit].mean() + (~crit).sum() * ef[~crit].mean()) / len(crit)
        self.assertLess(abs(orc.mean() - implied), 1e-9)

    def test_outcomes(self):
        r = lambda p, lo, hi: {"point": p, "lo": lo, "hi": hi}
        self.assertEqual(G.outcome_cell(r(-1, -2, -.5), r(-1, -2, -.5)), "GATE-BEATS-BOTH")
        self.assertEqual(G.outcome_cell(r(-1, -2, -.5), r(.5, .1, .9)), "GATE-WORSE")
        self.assertEqual(G.outcome_cell(r(-1, -2, .5), r(-1, -2, -.5)), "GATE-TIES")
        b = {"P1": {0: "GATE-BEATS-BOTH", 1: "GATE-BEATS-BOTH"}, "P2": {0: "GATE-TIES", 1: "GATE-TIES"}}
        self.assertEqual(G.outcome_overall(b), "GATE-DOMINATES")
        b["P2"][1] = "GATE-WORSE"
        self.assertEqual(G.outcome_overall(b), "GATE-COSTS")
        b["P2"][1] = "GATE-TIES"
        b["P1"][1] = "GATE-TIES"
        self.assertEqual(G.outcome_overall(b), "GATE-TIES")

    def test_score_and_auroc(self):
        pl = np.array([[.7, .2, .1, 0, 0]])
        po = np.array([[.5, .5, 0, 0, 0, 0, 0]])
        J = G.joint_probs(pl, po)
        self.assertAlmostEqual(float(G.score(J, 0)[0]), 0.35)       # (0,1) = 0.35 once (0,0) is the common case
        self.assertEqual(G.auroc(np.array([1., 2., 3., 4.]), np.array([0, 0, 1, 1], bool)), 1.0)
        self.assertEqual(G.auroc(np.array([1., 1., 1., 1.]), np.array([0, 0, 1, 1], bool)), 0.5)


if __name__ == "__main__":
    unittest.main()
