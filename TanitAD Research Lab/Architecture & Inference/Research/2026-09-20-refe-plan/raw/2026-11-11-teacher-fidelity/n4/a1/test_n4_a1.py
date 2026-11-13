"""A1 unit tests: the label pipeline uses the nuPlan-DB ego (label_ego), scoring keeps the cache ego.  Run like test_n4_chain (CHAIN_N4.md sec. 6 env), from this dir."""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n4_geom as G        # noqa: E402
import n4_nt_build as BD   # noqa: E402

CACHE_EGO, DB_EGO = object(), object()


def bare(label_fn=None, sc=None):
    cx = object.__new__(G.Ctx)
    cx.sc, cx.ego, cx.label_ego_fn, cx._label_ego, cx.force_cache_label_ego, cx.label_ego_src = sc, CACHE_EGO, label_fn, None, False, None
    return cx


class Probe:
    """replaces iterate20 / tracked8 / R.solve inside n4_geom: records which ego each label stage received"""
    def __init__(self):
        self.seen = {}

    def install(self, test):
        keep = (G.iterate20, G.tracked8, G.R.solve, G.ND)
        def it(nd, t20, t8, ego, ctl, a, b):
            self.seen["iterate20"] = ego
            return (np.zeros((20, 3)),)
        def tr(nd, Pb, ego):
            self.seen["tracked8"] = ego
            return None
        def solve(t20, d, lam, track, to):
            track(None)
            return {G.LAM: {"label": np.ones((20, 3))}}
        G.iterate20, G.tracked8, G.R.solve = it, tr, solve
        test.addCleanup(lambda: (setattr(G, "iterate20", keep[0]), setattr(G, "tracked8", keep[1]), setattr(G.R, "solve", keep[2])))


class TestLabelEgo(unittest.TestCase):
    def test_labels_use_label_ego_in_iterate20_and_tracked8(self):
        p = Probe(); p.install(self)
        cx = bare(lambda: DB_EGO)
        cx.labels(np.zeros((20, 3)))
        self.assertIs(p.seen["iterate20"], DB_EGO)
        self.assertIs(p.seen["tracked8"], DB_EGO)       # the closure inside R.solve
        self.assertEqual(cx.label_ego_src, "db")

    def test_cache_route_without_provider_refuses(self):
        p = Probe(); p.install(self)
        with self.assertRaises(G.LabelEgoMissing):
            bare(None).labels(np.zeros((20, 3)))
        self.assertEqual(p.seen, {})                    # nothing was built from the cache ego

    def test_provider_returning_none_refuses(self):
        p = Probe(); p.install(self)
        with self.assertRaises(G.LabelEgoMissing):
            bare(lambda: None).labels(np.zeros((20, 3)))

    def test_db_route_uses_scenario_ego(self):
        p = Probe(); p.install(self)
        cx = bare(None, sc=object())
        cx.labels(np.zeros((20, 3)))
        self.assertIs(p.seen["iterate20"], CACHE_EGO)   # on the DB route self.ego IS the DB ego
        self.assertEqual(cx.label_ego_src, "db")

    def test_mutation_forces_cache_ego_and_is_visible(self):
        p = Probe(); p.install(self)
        cx = bare(lambda: DB_EGO)
        cx.force_cache_label_ego = True
        cx.labels(np.zeros((20, 3)))
        self.assertIs(p.seen["iterate20"], CACHE_EGO)
        self.assertEqual(cx.label_ego_src, "cache")

    def test_provider_called_once(self):
        p = Probe(); p.install(self)
        n = []
        cx = bare(lambda: n.append(1) or DB_EGO)
        cx.labels(np.zeros((20, 3))); cx.labels(np.zeros((20, 3)))
        self.assertEqual(len(n), 1)

    def test_db_ego_fn_refuses_missing_log(self):
        with self.assertRaises(G.LabelEgoMissing):
            BD.db_ego_fn({}, "nolog", "tok")()

    def test_label_mutation_is_a_known_smoke_mutation(self):
        self.assertIn("label_ego_cache", BD.MUTATIONS)


if __name__ == "__main__":
    unittest.main()
