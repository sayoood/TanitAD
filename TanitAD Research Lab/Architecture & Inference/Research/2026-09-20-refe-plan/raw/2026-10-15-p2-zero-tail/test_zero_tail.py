#!/usr/bin/env python3
"""Tests for zero_tail.py.  Run:  C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -m unittest -v test_zero_tail
(unittest, not pytest: the eval venv has no pytest and installing into it is a known trap.)

Three kinds of test:
  * KNOWN-ANSWER: a synthetic dump whose SEL / PROP / cause / TTC counts are fixed by construction.
  * RED MUTATION: swap the SEL / PROP rule; the known-answer check MUST then fail (a check that cannot fail proves nothing).
  * REAL DATA: the mean recomputed from the P2 csv equals RESULT_P2's 88.82 within 0.01 (proves the right dump was read), the
    CRLF original and the repo copy hold the same content, and RESULT.json agrees with an independent naive recomputation.
"""
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zero_tail as Z  # noqa: E402


# ------------------------------------------------------------------ the synthetic world
def make_world():
    """10 tokens x 64 proposals.  Zero picks: tokens 0-5.  SEL: 0-3 (best 0.9, 0.8, 1.0, 0.5).  PROP: 4, 5.
    Non-zero picks: tokens 6-9 (pick 0.9); TTC tokens: 6, 7, 8 (see below)."""
    n = 10
    pdms = np.zeros((n, 64))
    nc = np.ones((n, 64))
    dac = np.ones((n, 64))
    ttc = np.ones((n, 64))
    for t, v in zip(range(4), (0.9, 0.8, 1.0, 0.5)):
        pdms[t, 7] = v
    dac[4, :] = 0.0                                     # token 4: every proposal leaves the drivable area -> all_DAC0
    nc[5, :] = 0.0                                      # token 5: every proposal collides -> all_NC0
    for t in range(6, 10):
        pdms[t, 0] = 0.9
    # TTC: token 6 has a TTC-ok proposal better than the pick; token 7 a TTC-ok proposal WORSE than the pick; token 8 none
    ttc[6, :] = 0.0; ttc[6, 3] = 1.0; pdms[6, 3] = 0.95
    ttc[7, :] = 0.0; ttc[7, 3] = 1.0; pdms[7, 3] = 0.85
    ttc[8, :] = 0.0
    pick_pdms = np.array([0, 0, 0, 0, 0, 0, 0.9, 0.9, 0.9, 0.9])
    pick_nc = np.array([1, 0, 0, 0.5, 1, 0, 1, 1, 1, 1], float)      # causes of the zero picks: DAC, NC, both, DAC(nc .5), DAC, NC
    pick_dac = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1], float)
    logs = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    return dict(n=n, pdms=pdms, nc=nc, dac=dac, ttc=ttc, pick_pdms=pick_pdms, pick_nc=pick_nc, pick_dac=pick_dac, logs=logs)


def known_answer(W, sel_fn=Z.sel_any):
    """Run the q1 / cause / fan machinery on the world and ASSERT the by-construction answers."""
    fs = Z.fan_stats(W["pdms"], W["nc"], W["dac"], W["ttc"], sel_fn=sel_fn)
    zero = W["pick_pdms"] == 0.0
    boot = Z.Boot(W["logs"], 5, B=400)
    q = Z.q1_block(zero, fs["sel"], fs["best"], boot, W["n"])
    assert q["n_zero"] == 6, q
    assert q["n_SEL"] == 4, q
    assert q["n_PROP"] == 2, q
    assert abs(q["SEL_share"]["point"] - 4 / 6) < 1e-12, q
    assert abs(q["oracle_on_SEL_gain_pdms_pts"]["point"] - 100 * (0.9 + 0.8 + 1.0 + 0.5) / 10) < 1e-9, q
    assert q["verdict"] == "SELECTION", q
    return q, fs, zero, boot


class TestKnownAnswer(unittest.TestCase):
    def test_sel_prop_counts_and_gain(self):
        known_answer(make_world())

    def test_RED_swapped_sel_prop_rule_is_caught(self):
        """The mutation: SEL <-> PROP. The same known-answer check must now FAIL."""
        W = make_world()
        with self.assertRaises(AssertionError):
            known_answer(W, sel_fn=lambda p: ~Z.sel_any(p))

    def test_RED_all_instead_of_any_is_caught(self):
        """The mutation: 'every proposal > 0' instead of 'some proposal > 0'."""
        W = make_world()
        with self.assertRaises(AssertionError):
            known_answer(W, sel_fn=lambda p: (p > 0).all(axis=1))

    def test_RED_strict_vs_nonstrict_threshold_is_caught(self):
        """The mutation: '>= 0' (everything is SEL)."""
        W = make_world()
        with self.assertRaises(AssertionError):
            known_answer(W, sel_fn=lambda p: (p >= 0).any(axis=1))

    def test_counts_sum_and_cause_crosstab(self):
        W = make_world()
        q, fs, zero, boot = known_answer(W)
        self.assertEqual(q["n_SEL"] + q["n_PROP"], q["n_zero"])
        cause = Z.cause_labels(W["pick_nc"], W["pick_dac"])
        self.assertEqual([str(c) for c in cause[:6]], ["DAC_only", "NC_only", "both", "DAC_only", "DAC_only", "NC_only"])
        cb = Z.cause_block(zero, cause, fs["sel"], boot)
        self.assertEqual({c: cb["all"][c]["n"] for c in Z.CAUSES}, {"DAC_only": 3, "NC_only": 2, "both": 1, "neither": 0})
        self.assertEqual({c: cb["SEL"][c]["n"] for c in Z.CAUSES}, {"DAC_only": 2, "NC_only": 1, "both": 1, "neither": 0})
        self.assertEqual({c: cb["PROP"][c]["n"] for c in Z.CAUSES}, {"DAC_only": 1, "NC_only": 1, "both": 0, "neither": 0})
        self.assertEqual(cb["SEL"]["n_total"] + cb["PROP"]["n_total"], 6)

    def test_prop_failure_kinds(self):
        W = make_world()
        fs = Z.fan_stats(W["pdms"], W["nc"], W["dac"], W["ttc"])
        self.assertEqual(str(fs["prop_kind"][4]), "all_DAC0")
        self.assertEqual(str(fs["prop_kind"][5]), "all_NC0")
        # a mixed fan: half the proposals leave the road, half collide, none is a clean zero from one gate
        pdms = np.zeros((1, 64)); nc = np.ones((1, 64)); dac = np.ones((1, 64)); ttc = np.ones((1, 64))
        dac[0, :32] = 0.0; nc[0, 32:] = 0.0
        self.assertEqual(str(Z.fan_stats(pdms, nc, dac, ttc)["prop_kind"][0]), "mixed")
        # an 'other' fan: NC x DAC > 0 yet PDMS == 0
        nc2 = np.ones((1, 64)); dac2 = np.ones((1, 64))
        self.assertEqual(str(Z.fan_stats(pdms, nc2, dac2, ttc)["prop_kind"][0]), "other")

    def test_cleanliness_known_answer(self):
        W = make_world()
        q, fs, zero, boot = known_answer(W)
        self.assertEqual(fs["n_nonzero"][:6].tolist(), [1, 1, 1, 1, 0, 0])
        c = Z.cleanliness(zero, fs["sel"], fs["n_nonzero"])
        self.assertEqual((c["n"], c["exactly_1_of_64"], c["32_or_more_of_64"]), (4, 4, 0))
        # a fan with 40 non-zero proposals lands in the 32+ bucket
        n_nz = np.array([40, 3, 10, 1])
        c2 = Z.cleanliness(np.array([True] * 4), np.array([True] * 4), n_nz)
        self.assertEqual((c2["exactly_1_of_64"], c2["2_to_4_of_64"], c2["5_to_31_of_64"], c2["32_or_more_of_64"]), (1, 1, 1, 1))

    def test_ttc_block_known_answer(self):
        W = make_world()
        fs = Z.fan_stats(W["pdms"], W["nc"], W["dac"], W["ttc"])
        ttc_tok = np.array([False] * 6 + [True, True, True, False])
        boot = Z.Boot(W["logs"], 5, B=400)
        b = Z.ttc_block(ttc_tok, fs["sel_ttc"], fs["best_ttcok"], W["pick_pdms"], boot, W["n"])
        self.assertEqual((b["n_TTC_tokens"], b["n_SEL_TTC"], b["n_PROP_TTC"]), (3, 2, 1))
        # token 6: +0.05, token 7: floored at 0 (the TTC-ok proposal is worse than the pick), token 8: none
        self.assertAlmostEqual(b["headroom_pdms_pts"]["point"], 100 * 0.05 / 10, places=9)
        # a mutation that forgets the floor must change the answer
        head_nofloor = np.where(ttc_tok & fs["sel_ttc"], fs["best_ttcok"] - W["pick_pdms"], 0.0).sum() / 10 * 100
        self.assertNotAlmostEqual(head_nofloor, b["headroom_pdms_pts"]["point"], places=6)


class TestVerdict(unittest.TestCase):
    def test_branches(self):
        self.assertEqual(Z.verdict(60, 100, 2.0), "SELECTION")
        self.assertEqual(Z.verdict(50, 100, 2.5), "SELECTION")         # exactly 50 % and gain >= 2
        self.assertEqual(Z.verdict(60, 100, 1.99), "MIXED")            # SEL >= 50 % but gain below 2.0 -> 'otherwise'
        self.assertEqual(Z.verdict(49, 100, 9.0), "PROPOSAL")          # PROP = 51 %
        self.assertEqual(Z.verdict(40, 100, 0.1), "PROPOSAL")
        self.assertEqual(Z.verdict(0, 0, 1.0), "NO_ZERO_TOKENS")

    def test_exact_half_goes_to_selection_first_when_gain_clears(self):
        self.assertEqual(Z.verdict(5, 10, 3.0), "SELECTION")
        self.assertEqual(Z.verdict(5, 10, 1.0), "PROPOSAL")            # PROP >= 50 % also holds; SEL branch failed on the gain

    def test_RED_swapped_branches_are_caught(self):
        def mutated(n_sel, n_zero, gain):
            return Z.verdict(n_zero - n_sel, n_zero, gain)
        self.assertNotEqual(mutated(80, 100, 3.0), Z.verdict(80, 100, 3.0))


class TestBootstrapAndConcentration(unittest.TestCase):
    def test_ratio_point_and_paired_draws(self):
        logs = np.repeat(np.arange(20), 5)
        b1 = Z.Boot(logs, 20, B=500)
        b2 = Z.Boot(logs, 20, B=500)
        np.testing.assert_array_equal(b1.draws, b2.draws)               # same seed -> same log draws (paired by construction)
        num = (np.arange(100) % 2 == 0).astype(float)
        r = b1.ratio(num, np.ones(100))
        self.assertAlmostEqual(r["point"], 0.5, places=12)
        self.assertLessEqual(r["lo"], 0.5 + 1e-9)
        self.assertGreaterEqual(r["hi"], 0.5 - 1e-9)

    def test_degenerate_interval_collapses(self):
        logs = np.repeat(np.arange(10), 4)
        b = Z.Boot(logs, 10, B=300)
        r = b.ratio(np.ones(40), np.ones(40))
        self.assertEqual((r["point"], r["lo"], r["hi"]), (1.0, 1.0, 1.0))

    def test_interval_widens_when_one_log_carries_everything(self):
        logs = np.repeat(np.arange(10), 4)
        num = np.zeros(40); num[:4] = 1.0
        r = Z.Boot(logs, 10, B=2000).ratio(num, np.ones(40))
        self.assertAlmostEqual(r["point"], 0.1)
        self.assertEqual(r["lo"], 0.0)
        self.assertGreater(r["hi"], 0.2)

    def test_k50(self):
        self.assertEqual(Z.k50(np.array([10, 5, 3, 2])), 1)             # total 20, half 10
        self.assertEqual(Z.k50(np.array([4, 4, 4, 4])), 2)              # total 16, half 8
        self.assertEqual(Z.k50(np.array([3, 3, 3])), 2)                 # total 9, ceil(9/2)=5 -> 2 logs
        self.assertEqual(Z.k50(np.array([0, 0])), 0)
        self.assertEqual(Z.k50(np.array([1] * 509)), 255)

    def test_concentration_top10_and_ties_by_name(self):
        names = np.array(["a", "b", "c", "d"])
        log_idx = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        zero = np.array([True, True, True, True, True, False, False, False])
        c = Z.concentration(zero, log_idx, 4, names)
        self.assertEqual([t["log"] for t in c["top10"]][:3], ["a", "b", "c"])    # a=2, b=2 tie -> by name; c=1
        self.assertEqual(c["k50"], 2)                                            # total 5, ceil(5/2)=3 -> a+b
        self.assertEqual(c["n_logs_with_zero"], 3)

    def test_permutation_null_is_reproducible_and_bounded(self):
        log_idx = np.repeat(np.arange(30), 20)
        a = Z.permutation_null(600, 40, log_idx, 30, B=200)
        b = Z.permutation_null(600, 40, log_idx, 30, B=200)
        np.testing.assert_array_equal(a["ks"], b["ks"])
        self.assertTrue((a["ks"] >= 1).all() and (a["ks"] <= 30).all())

    def test_embed(self):
        out = Z.embed(5, np.array([1, 3]), np.array([True, True]), False)
        self.assertEqual(out.tolist(), [False, True, False, True, False])


# ------------------------------------------------------------------ real data
def _p2_original():
    p = Path(Z.DEFAULTS["p2_csv"])
    return p if p.exists() else None


class TestRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tk = json.load(open(Z.DEFAULTS["tokens"], encoding="utf-8"))
        cls.tokens = sorted(cls.tk["tokens"])
        cls.repo_csv = Path(Z.DEFAULTS["p2_csv_repo"])
        cls.res_path = HERE / "RESULT.json"

    def test_p2_mean_recomputed_equals_result_p2_88_82(self):
        """Proves the right dump was read: the mean of 12,146 token scores = RESULT_P2's 88.82 within 0.01."""
        s = Z.load_scores(self.repo_csv, self.tokens)
        mean = float(s["score"].mean() * 100)
        self.assertEqual(len(s["score"]), 12146)
        self.assertLessEqual(abs(mean - 88.82), 0.01, mean)
        self.assertAlmostEqual(mean, s["average"] * 100, places=6)       # and the csv's own average row
        self.assertEqual(int((s["score"] == 0.0).sum()), 509)             # RESULT_P2: zero-PDMS tokens 509
        self.assertTrue(s["valid"].all())

    def test_p2_original_is_the_banked_file_and_the_repo_copy_has_the_same_content(self):
        orig = _p2_original()
        if orig is None:
            self.skipTest("D:/E: original not mounted")
        a = orig.read_bytes()
        self.assertEqual(hashlib.sha256(a).hexdigest(), Z.EXPECT["p2_csv_sha256"])
        self.assertEqual(hashlib.md5(a).hexdigest(), "bed5687fb75e884d45cf991c2ff391c9")          # MD5SUMS.txt in the repo
        self.assertEqual(a.replace(b"\r\n", b"\n"), self.repo_csv.read_bytes())                  # repo copy = LF-normalised original

    def test_u400_table_at_pick_reproduces_its_csv_and_mean(self):
        u4 = Z.load_scores(Z.DEFAULTS["u400_csv_repo"], self.tokens)
        self.assertLessEqual(abs(float(u4["score"].mean() * 100) - 88.25), 0.01)
        self.assertEqual(int((u4["score"] == 0).sum()), 529)
        T = np.load(Z.DEFAULTS["c3_table"], allow_pickle=True)
        pos = {t: i for i, t in enumerate(self.tokens)}
        idx = np.array([pos[str(t)] for t in T["token"]])
        at_pick = T["pdms"][np.arange(len(idx)), T["pick"]]
        self.assertEqual(float(np.max(np.abs(at_pick - u4["score"][idx]))), 0.0)

    def test_RED_swapped_rule_on_real_u400_table_changes_the_headline(self):
        T = np.load(Z.DEFAULTS["c3_table"], allow_pickle=True)
        u4 = Z.load_scores(Z.DEFAULTS["u400_csv_repo"], self.tokens)
        pos = {t: i for i, t in enumerate(self.tokens)}
        idx = np.array([pos[str(t)] for t in T["token"]])
        zero = (u4["score"][idx] == 0.0)
        sel = Z.sel_any(T["pdms"].astype(float))
        self.assertEqual((int((zero & sel).sum()), int(zero.sum())), (455, 529))                   # = C3's 455 / 529
        self.assertEqual(int((zero & ~sel).sum()), 74)
        self.assertNotEqual(int((zero & ~Z.sel_any(T["pdms"].astype(float))).sum()), 455)           # the swapped rule reads 74, not 455

    def test_result_json_agrees_with_a_naive_loop(self):
        if not self.res_path.exists():
            self.skipTest("RESULT.json not produced yet")
        R = json.load(open(self.res_path, encoding="utf-8"))
        self.assertNotIn("STOP", R)
        self.assertTrue(all(g["pass"] for g in R["gates"].values()))
        # independent recomputation of u400's SEL / PROP by a plain python loop over the C3 table
        T = np.load(Z.DEFAULTS["c3_table"], allow_pickle=True)
        u4 = Z.load_scores(Z.DEFAULTS["u400_csv_repo"], self.tokens)
        pos = {t: i for i, t in enumerate(self.tokens)}
        n_sel = n_prop = 0
        gain = 0.0
        for r, t in enumerate(T["token"]):
            if u4["score"][pos[str(t)]] != 0.0:
                continue
            row = [float(x) for x in T["pdms"][r]]
            if max(row) > 0:
                n_sel += 1
                gain += max(row)
            else:
                n_prop += 1
        q = R["Q1a_u400_own"]["SEL_def"]
        self.assertEqual((q["n_SEL"], q["n_PROP"]), (n_sel, n_prop))
        self.assertAlmostEqual(q["oracle_on_SEL_gain_pdms_pts"]["point"], gain / 12146 * 100, places=9)
        # and P2 V1 by a naive per-token hierarchy loop
        p2 = Z.load_scores(self.repo_csv, self.tokens)
        census = Z.load_census(Z.DEFAULTS["census"], self.tokens)
        in_u = {str(t): r for r, t in enumerate(T["token"])}
        v1_sel = 0
        for i, t in enumerate(self.tokens):
            if p2["score"][i] != 0.0:
                continue
            fan = T["pdms"][in_u[t]] if t in in_u else census["pdms"][i]
            v1_sel += int(max(float(x) for x in fan) > 0)
        self.assertEqual(R["Q1c_P2_bridged"]["variants"]["V1_hierarchy"]["SEL_def"]["n_SEL"], v1_sel)

    def test_result_json_internal_identities(self):
        if not self.res_path.exists():
            self.skipTest("RESULT.json not produced yet")
        R = json.load(open(self.res_path, encoding="utf-8"))
        for v in R["Q1c_P2_bridged"]["variants"].values():
            s = v["SEL_def"]
            self.assertEqual(s["n_SEL"] + s["n_PROP"], s["n_zero"])
            self.assertEqual(s["n_zero"], 509)
            c = v["Q2_cause"]
            self.assertEqual(sum(c["all"][k]["n"] for k in Z.CAUSES), 509)
            self.assertEqual(c["SEL"]["n_total"] + c["PROP"]["n_total"], 509)
        c = R["Q2_P2_pick_level"]["cause"]
        self.assertEqual(sum(c[k]["n"] for k in Z.CAUSES), 509)
        self.assertEqual(R["Q3_concentration"]["P2"]["n_zero"], 509)
        self.assertEqual(R["bridge"]["persistence"]["P2z_and_u400z"] + R["bridge"]["persistence"]["P2z_only"], 509)


if __name__ == "__main__":
    unittest.main(verbosity=2)
