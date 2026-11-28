"""CPU tests of sh5_lib / sh5_admit / sh5_gpulock.  RED mutations: every decision function, the sign-clean target rule, the stacker identity, the permutation controls and the lock rules each have a mutated
copy that MUST make a check fail (a test that cannot fail proves nothing).   python -B test_sh5_lib.py"""
import json
import os
import sys
import tempfile
import time
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sh5_lib as L                                                                                    # noqa: E402
import sh5_admit as AD                                                                                 # noqa: E402
import sh5_gpulock as GL                                                                               # noqa: E402
import sh5_synth as SY                                                                                 # noqa: E402


def mutated(module, old, new):
    """a copy of `module` whose source has `old` replaced by `new` (must occur) - the RED-mutation helper"""
    src = open(module.__file__, encoding="utf-8").read()
    assert old in src, f"mutation target not found: {old!r}"
    m = types.ModuleType(module.__name__ + "_mut")
    m.__file__ = module.__file__
    exec(compile(src.replace(old, new, 1), module.__file__, "exec"), m.__dict__)
    return m


class TestKeypoints(unittest.TestCase):
    def test_straight_path(self):
        t = 0.2 * (np.arange(20) + 1)
        tr = np.stack([5.0 * t, 0 * t, 0 * t], -1)
        kp = L.keypoints(tr)
        self.assertEqual(kp.shape, (9, 3))
        np.testing.assert_allclose(kp[:, 0], 5.0 * np.array(L.KEY_TIMES_S), atol=1e-5)
        np.testing.assert_allclose(kp[:, 1:], 0.0, atol=1e-6)

    def test_t0_is_origin_and_t4_is_last_pose_with_repaired_heading(self):
        tr = np.zeros((20, 3))
        tr[:, 0] = np.arange(20)
        tr[:, 2] = np.linspace(0.1, 1.0, 20)
        tr[-1, 2] = 2.9                                              # the corrupted last heading (A7)
        kp = L.keypoints(tr)
        np.testing.assert_allclose(kp[0], 0.0, atol=1e-7)
        self.assertAlmostEqual(float(kp[-1, 0]), 19.0, places=5)
        self.assertAlmostEqual(float(kp[-1, 2]), float(tr[-2, 2]), places=5)          # repaired: previous pose's heading
        kp_raw = L.keypoints(tr, repair=False)
        self.assertAlmostEqual(float(kp_raw[-1, 2]), 2.9, places=5)                    # RED: without the repair the corrupted heading survives

    def test_heading_interpolates_across_the_wrap(self):
        tr = np.zeros((20, 3))
        tr[:, 2] = np.pi - 0.02
        tr[3::2, 2] = -np.pi + 0.02                                    # alternates across +-pi
        kp = L.keypoints(tr)
        self.assertTrue((np.abs(np.abs(kp[1:-1, 2]) - np.pi) < 0.05).all())             # no wrap artefact (a linear interpolation would give ~0)

    def test_batch_shape(self):
        self.assertEqual(L.keypoints(np.zeros((3, 64, 20, 3))).shape, (3, 64, 9, 3))


class TestTargets(unittest.TestCase):
    def test_sign_clean_targets_ignore_cmin_and_fracout(self):
        rng = np.random.default_rng(0)
        g = rng.normal(0, 3, (4, 64, 15))
        t = L.clearance_targets(g)
        self.assertEqual(t.shape, (4, 64, 9))
        np.testing.assert_allclose(t, np.minimum(np.abs(g[..., 2:11]), 10.0).astype(np.float32))
        g2 = g.copy()
        g2[..., 0] = 123.0                                             # |dac_clear_min|
        g2[..., 1] = 0.77                                              # dac_frac_out
        np.testing.assert_array_equal(L.clearance_targets(g2), t)      # the targets do not move when the verdict-bearing columns move
        self.assertEqual(L.N_REG, 9)

    def test_red_target_that_includes_cmin_changes(self):
        m = mutated(L, "return np.minimum(np.abs(g[..., 2:11]), CLEAR_CAP).astype(np.float32)", "return np.minimum(np.abs(g[..., [0] + list(range(2, 11))]), CLEAR_CAP).astype(np.float32)")
        g = np.random.default_rng(1).normal(0, 3, (2, 64, 15))
        g2 = g.copy()
        g2[..., 0] = 55.0
        self.assertFalse(np.array_equal(m.clearance_targets(g), m.clearance_targets(g2)))          # RED: the mutated rule IS sensitive to c_min: the invariance test above would fail on it


class TestAdmit(unittest.TestCase):
    def _synth_block(self, leak):
        rng = np.random.default_rng(3)
        N = 200
        c0 = rng.uniform(1.0, 6.0, (N, 64, 1))                                 # starts inside
        slope = rng.normal(-1.2, 1.1, (N, 64, 1))
        curv = np.abs(rng.normal(0.35, 0.3, (N, 64, 1)))                       # approach-and-retreat paths exist: a GRAZE (min just above 0) and a CROSSING (min just below) look alike in |c|
        tt = np.arange(41) * 0.1
        sc = c0 + slope * tt + curv * tt ** 2 * 1.5 + 0.8 * np.cumsum(rng.normal(size=(N, 64, 41)), -1) * 0.3        # an irregular boundary: crossings between key steps and grazes exist
        cmin = sc.min(-1)
        lab = (cmin >= 0).astype(np.float64)
        key = np.abs(sc[..., ::5][..., :9])
        blk = np.concatenate([np.abs(cmin)[..., None], key], -1) if leak else key
        return blk, lab

    def test_leaky_block_flagged_and_clean_not(self):
        blk, lab = self._synth_block(True)
        self.assertTrue(AD.search(blk, lab)["flagged"])                        # |c_t| < |c_min| proves unsafe
        blk, lab = self._synth_block(False)
        res = AD.a_test(blk, lab, known_leaky=self._synth_block(True)[0])
        self.assertTrue(res["controls_ok"], res)
        self.assertFalse(res["candidate"]["flagged"], res["candidate"]["top"])
        self.assertTrue(res["admissible"])

    def test_red_test_that_cannot_see_a_planted_column_fails_its_control(self):
        blk, lab = self._synth_block(False)
        m = mutated(AD, "if p >= prec_bar and r >= recall_min:", "if p >= 2.0 and r >= recall_min:")
        res = m.a_test(blk, lab)
        self.assertFalse(res["POS-2"]["flagged"])                               # RED: a search that never fires misses the planted channel ...
        self.assertFalse(res["controls_ok"])                                    # ... so the controls refuse to certify it

    def test_red_always_flagging_test_fails_the_negative_control(self):
        blk, lab = self._synth_block(False)
        m = mutated(AD, "if p >= prec_bar and r >= recall_min:", "if True:")
        res = m.a_test(blk, lab)
        self.assertTrue(res["NEG"]["flagged"])
        self.assertFalse(res["controls_ok"])


class TestStacker(unittest.TestCase):
    def test_recovers_weights(self):
        rng = np.random.default_rng(0)
        n = 60000
        zl, zs = rng.normal(0, 1.5, n), rng.normal(0, 1.5, n)
        y = (rng.random(n) < L._sig(0.7 * zl + 1.3 * zs)).astype(float)
        a, b, ok = L.fit_stacker(zl, zs, y)
        self.assertTrue(ok)
        self.assertAlmostEqual(a, 0.7, delta=0.06)
        self.assertAlmostEqual(b, 1.3, delta=0.08)

    def test_zero_branch_gives_L_exactly(self):
        rng = np.random.default_rng(1)
        zl = rng.normal(0, 1, (30, 64)).astype(np.float32)
        y = (rng.random((30, 64)) < 0.5).astype(np.float32)
        fold = np.arange(30) % 5
        out, par = L.oof_stack(zl, np.zeros_like(zl), y, fold)
        self.assertTrue(np.array_equal(out, zl))                                # bit for bit (C-ID)
        self.assertTrue(all(p[0] == 1.0 and p[1] == 0.0 for p in par))

    def test_red_identity_breaks_without_the_degenerate_rule(self):
        m = mutated(L, "if zs.size == 0 or float(zs.std()) < 1e-9:", "if False:")
        rng = np.random.default_rng(2)
        zl = rng.normal(0, 2.0, 20000)
        y = (rng.random(20000) < L._sig(1.8 * zl)).astype(float)
        a, b, ok = m.fit_stacker(zl, np.zeros_like(zl), y)
        self.assertNotEqual(a, 1.0)                                              # RED: the scale refits and L+S != L

    def test_leave_fold_out(self):
        rng = np.random.default_rng(3)
        zl, zs = rng.normal(0, 1, (50, 64)), rng.normal(0, 1, (50, 64))
        y = (rng.random((50, 64)) < L._sig(zl + zs)).astype(float)
        fold = np.arange(50) % 5
        out, par = L.oof_stack(zl, zs, y, fold)
        a, b, _ = L.fit_stacker(zl[fold != 2], zs[fold != 2], y[fold != 2])
        self.assertEqual((a, b), (par[2][0], par[2][1]))                         # fold 2's weights come from the OTHER folds only
        np.testing.assert_allclose(out[fold == 2], (a * zl[fold == 2] + b * zs[fold == 2]).astype(np.float32))


class TestPermutations(unittest.TestCase):
    def test_within_token(self):
        D, _ = SY.make_synth(T=12, seed=1)
        tgt = D["tgt"]
        p = L.permute_within_token(tgt, D["token_idx"], 7)
        self.assertFalse(np.array_equal(p, tgt))
        for t in range(D["T"]):
            rows = D["token_idx"] == t
            a = np.sort(tgt[rows].reshape(-1, 9), axis=0)
            b = np.sort(p[rows].reshape(-1, 9), axis=0)
            self.assertTrue(np.allclose(a[:, 0], b[:, 0]))                       # per-token marginal of column 0 unchanged
        np.testing.assert_array_equal(p, L.permute_within_token(tgt, D["token_idx"], 7))        # deterministic
        self.assertFalse(np.array_equal(p, L.permute_within_token(tgt, D["token_idx"], 8)))
        # the K-vector moves as a unit (its internal structure is kept) but the proposal <-> clearance MATCH is destroyed: the link to the proposal's own geometry collapses
        own = np.minimum(np.abs(D["sc_key"]), 10.0)                                  # the targets ARE the |clearance| of each proposal's own geometry ...
        self.assertTrue(np.allclose(own, tgt))
        still_own = np.isclose(p, own).all(-1).mean()                                # ... and after the permutation almost no proposal still carries its own 9-vector
        self.assertLess(still_own, 0.05)
        self.assertGreater(np.isclose(tgt, own).all(-1).mean(), 0.99)
        self.assertAlmostEqual(np.corrcoef(tgt[..., 0].ravel(), tgt[..., 8].ravel())[0, 1], np.corrcoef(p[..., 0].ravel(), p[..., 8].ravel())[0, 1], places=6)

    def test_within_set(self):
        a = np.arange(3 * 64).reshape(3, 64).astype(float)
        p = L.permute_within_set(a, 5)
        self.assertTrue(all(sorted(p[i]) == sorted(a[i]) for i in range(3)))
        self.assertFalse(np.array_equal(p, a))

    def test_plant_noise_is_token_stable(self):
        a = L.plant_noise("abc", (3, 64, 9), 1.0)
        self.assertTrue(np.array_equal(a, L.plant_noise("abc", (3, 64, 9), 1.0)))
        self.assertFalse(np.array_equal(a, L.plant_noise("abd", (3, 64, 9), 1.0)))


class TestAnalyzer(unittest.TestCase):
    def test_perfect_and_random_and_equal_to_privgeom(self):
        D, _ = SY.make_synth(T=40, seed=2)
        lab = D["lab"][..., L.DAC_COL]
        an = L.Analyzer(lab, D["code"], D["log_idx"], D["n_logs"], B=500)
        perfect = np.where(lab == 1.0, 1.0, -1.0) + 0.0
        self.assertAlmostEqual(an.auc(an.stats(perfect), "Q123")["point"], 1.0, places=9)
        rnd = np.random.default_rng(0).normal(size=lab.shape)
        self.assertAlmostEqual(an.auc(an.stats(rnd), "Q123")["point"], 0.5, delta=0.05)
        # C-ID-L analogue: identical numbers to the PRIVGEOM-SAFE analyzer on identical inputs
        pg = "C:/Users/Admin/par18/privgeom/code"
        if os.path.isdir(pg):
            sys.path.insert(0, pg)
            import pg_lib as P
            lab6 = D["lab"]
            a2 = P.Analyzer(lab6, D["code"], D["log_idx"], D["n_logs"], B=500)
            z = D["z6"][..., L.DAC_COL]
            s1, s2 = an.stats(z), a2.per_log_stats(z, "dac")
            for k in ("Q4", "Q123"):
                self.assertEqual(an.auc(s1, k), a2.auc(s2, k))

    def test_diff_of_diffs_matches_direct(self):
        D, _ = SY.make_synth(T=40, seed=4)
        lab = D["lab"][..., L.DAC_COL]
        an = L.Analyzer(lab, D["code"], D["log_idx"], D["n_logs"], B=300)
        rng = np.random.default_rng(1)
        z = [rng.normal(size=lab.shape) + (lab == 1.0) * k for k in (0.2, 0.5, 0.9, 0.1)]
        s = [an.stats(x) for x in z]
        dd = an.diff_of_diffs(s[0], s[1], s[2], s[3])
        d_a, d_b = an.diff(s[0], s[1]), an.diff(s[2], s[3])
        self.assertAlmostEqual(dd["point"], d_a["point"] - d_b["point"], places=9)


class TestDecisionTable(unittest.TestCase):
    def test_table_passes(self):
        self.assertEqual(L.branch_table_failures(), [])

    def test_every_mutation_breaks_the_table(self):
        NLF = 'all(seed_call(c) == "REACH" for c in cis_by_seed)'                                       # the Amendment-3 C-NULL-LABEL fire rule (one definition: it IS seed_call)
        rule = 'if ci["lo"] > REACH_LO_MIN and ci["point"] >= REACH_POINT_MIN:'                          # the Amendment-2 CTX-REACHABLE per-seed rule
        pins = "REACH_LO_MIN, REACH_POINT_MIN = 0.0, 0.02"
        muts = [(rule, 'if ci["lo"] >= REACH_LO_MIN and ci["point"] >= REACH_POINT_MIN:'),             # lower-CI comparator flipped
                (rule, 'if ci["lo"] > REACH_LO_MIN and ci["point"] > REACH_POINT_MIN:'),               # point comparator flipped
                (rule, 'if ci["point"] >= REACH_POINT_MIN:'),                                          # RED: drop the lower-CI conjunct
                (rule, 'if ci["lo"] > REACH_LO_MIN:'),                                                 # RED: drop the point conjunct
                (rule, 'if ci["lo"] > REACH_LO_MIN or ci["point"] >= REACH_POINT_MIN:'),               # and -> or
                (rule, 'if ci["lo"] > INFO_MARGIN:'),                                                  # RED: the PRE-AMENDMENT (sealed) rule is caught
                (pins, "REACH_LO_MIN, REACH_POINT_MIN = 0.0, 0.03"),                                   # point threshold changed
                (pins, "REACH_LO_MIN, REACH_POINT_MIN = 0.0, 0.01"),
                (pins, "REACH_LO_MIN, REACH_POINT_MIN = 0.01, 0.02"),                                  # lower-CI threshold changed
                ("INFO_MARGIN, NOTREACH_MARGIN = 0.02, 0.01", "INFO_MARGIN, NOTREACH_MARGIN = 0.02, 0.02"),   # NOT threshold changed (INFO_MARGIN is HISTORICAL since Amendment 3: no decision function reads it, so moving it is not a decision change)
                # ---- Amendment 3: C-NULL-LABEL fires iff the replicate is CTX-REACHABLE by the Amendment-2 rule (`seed_call(c) == "REACH"`)
                (NLF, 'all(c["lo"] > INFO_MARGIN for c in cis_by_seed)'),                              # RED: revert to the SEALED rule (misses a planted +0.025 replicate with lower CI > 0)
                (NLF, 'all(c["lo"] > REACH_LO_MIN for c in cis_by_seed)'),                             # RED: a second copy of the rule that DROPS the point conjunct
                (NLF, 'all(c["point"] >= REACH_POINT_MIN for c in cis_by_seed)'),                      # RED: a second copy of the rule that DROPS the lower-CI conjunct
                (NLF, 'all(c["lo"] > REACH_LO_MIN or c["point"] >= REACH_POINT_MIN for c in cis_by_seed)'),   # RED: and -> or
                (NLF, 'all(c["lo"] >= REACH_LO_MIN and c["point"] >= REACH_POINT_MIN for c in cis_by_seed)'),   # RED: strict comparator weakened
                (NLF, 'all(c["lo"] > REACH_LO_MIN and c["point"] > REACH_POINT_MIN for c in cis_by_seed)'),     # RED: inclusive comparator hardened
                (NLF, 'any(seed_call(c) == "REACH" for c in cis_by_seed)'),                            # RED: every seed -> any seed
                (NLF, 'all(seed_call(c) != "NOT" for c in cis_by_seed)'),                              # RED: INC fires too
                ('return bool(len(cis_by_seed) > 0 and all(seed_call(c) == "REACH" for c in cis_by_seed))', 'return bool(all(seed_call(c) == "REACH" for c in cis_by_seed))'),   # RED: no seed read fires (vacuous truth)
                ("if ci[\"hi\"] < NOTREACH_MARGIN:", "if ci[\"hi\"] <= NOTREACH_MARGIN:"),
                ("all(c == \"REACH\" for c in calls)", "any(c == \"REACH\" for c in calls)"),            # seeds: all -> any
                ("all(c == \"NOT\" for c in calls)", "any(c == \"NOT\" for c in calls)"),
                ("ddac_ci[\"hi\"] <= 0.0", "ddac_ci[\"hi\"] < 0.0"),
                ("dpdms_ci[\"lo\"] > 0.0", "dpdms_ci[\"lo\"] >= 0.0"),
                ("return bool(any(fires))", "return bool(sum(fires) >= 2)"),
                ("c[\"lo\"] > 0.0 and POWER_POINT_LO <= c[\"point\"] <= POWER_POINT_HI", "POWER_POINT_LO <= c[\"point\"] <= POWER_POINT_HI"),
                ("if raw == \"NOT-REACHABLE\" and not power:", "if raw == \"NOT-REACHABLE\" and False:"),
                ("if raw == \"CTX-REACHABLE\" and shuf_gain:", "if False:"),
                ("return \"INTEGRATE-SH5\" if pick_gain else \"STOP-AT-DIAGNOSTIC\"", "return \"INTEGRATE-SH5\""),
                ("if not creach_ok:\n        return \"NO-CALL(C-REACH)\"", "if False:\n        return \"NO-CALL(C-REACH)\"")]
        for old, new in muts:
            m = mutated(L, old, new)
            self.assertTrue(len(L.branch_table_failures(m)) > 0, f"mutation did not break the table: {old}")


class TestAmendment2Rule(unittest.TestCase):
    """AMENDMENT 2 (MM ruling 2026-10-19): CTX-REACHABLE per seed iff Delta_F lower CI > 0 AND point >= +0.02, on both seeds; NOT-REACHABLE unchanged; GAIN-NOT-FROM-CTX reads S-shuf with the NEW rule;
    VOID (C-NULL-LABEL) kept its sealed numeric rule in Amendment 2 and takes the new one in Amendment 3 (`TestAmendment3Rule`).  Each conjunct: its boundary, and a RED mutation that drops it and MUST pass a planted case it should reject."""
    cp = staticmethod(lambda point, lo, hi: {"point": point, "lo": lo, "hi": hi})
    SRC = 'if ci["lo"] > REACH_LO_MIN and ci["point"] >= REACH_POINT_MIN:'

    def test_boundaries_of_each_conjunct(self):
        c = self.cp
        self.assertEqual(L.seed_call(c(0.020, 0.001, 0.040)), "REACH")                    # point == +0.02 passes (>=)
        self.assertEqual(L.seed_call(c(0.0199, 0.001, 0.040)), "INC")                     # just below: materiality fails
        self.assertEqual(L.seed_call(c(0.030, 1e-9, 0.060)), "REACH")                     # lower CI barely above 0 passes
        self.assertEqual(L.seed_call(c(0.030, 0.0, 0.060)), "INC")                        # lower CI == 0: significance fails (strict)
        self.assertEqual(L.seed_call(c(0.030, -1e-9, 0.060)), "INC")
        self.assertEqual(L.seed_call(c(0.019, 0.006, 0.032)), "INC")                      # planted +0.019, significant
        self.assertEqual(L.seed_call(c(0.005, 0.001, 0.012)), "INC")                      # planted +0.005, significant
        self.assertEqual(L.seed_call(c(0.005, 0.001, 0.009)), "NOT")                      # ... whose upper CI < +0.01 is NOT-REACHABLE (unchanged rule)
        self.assertEqual(L.seed_call(c(0.025, -0.003, 0.053)), "INC")                     # planted +0.025, not significant
        self.assertEqual(L.seed_call(c(0.0262, 0.0201, 0.0600)), "REACH")                 # the sealed rule's region is inside the new one
        self.assertEqual(L.seed_call(c(0.0262, 0.0200, 0.0600)), "REACH")                 # the sealed edge lo == +0.02 (INC before Amendment 2) is REACH now
        self.assertEqual(L.seed_call({"point": float("nan"), "lo": 0.01, "hi": 0.05}), "INC")
        self.assertEqual(L.seed_call({"point": 0.03, "lo": float("nan"), "hi": 0.05}), "INC")

    def test_red_dropping_the_lower_ci_conjunct_passes_a_planted_insignificant_case(self):
        m = mutated(L, self.SRC, 'if ci["point"] >= REACH_POINT_MIN:')
        planted = self.cp(0.025, -0.003, 0.053)                                             # material but NOT significant
        self.assertEqual(L.seed_call(planted), "INC")
        self.assertEqual(m.seed_call(planted), "REACH")                                     # the mutated rule wrongly passes it
        fails = L.branch_table_failures(m)
        self.assertTrue(any("NOT significant" in f for f in fails), fails)

    def test_red_dropping_the_point_conjunct_passes_planted_immaterial_cases(self):
        m = mutated(L, self.SRC, 'if ci["lo"] > REACH_LO_MIN:')
        for planted in (self.cp(0.019, 0.006, 0.032), self.cp(0.005, 0.001, 0.012)):        # "+0.019, significant" and "+0.005, significant"
            self.assertEqual(L.seed_call(planted), "INC")
            self.assertEqual(m.seed_call(planted), "REACH")
        fails = L.branch_table_failures(m)
        self.assertTrue(any("+0.019" in f for f in fails) and any("+0.005" in f for f in fails), fails)

    def test_the_pre_amendment_rule_cannot_pass_the_table(self):
        m = mutated(L, self.SRC, 'if ci["lo"] > INFO_MARGIN:')
        self.assertTrue(len(L.branch_table_failures(m)) > 0)
        self.assertEqual(m.seed_call(self.cp(0.030, 0.010, 0.050)), "INC")                  # sealed rule: lower CI 0.010 < 0.02
        self.assertEqual(L.seed_call(self.cp(0.030, 0.010, 0.050)), "REACH")

    def test_both_seeds_and_the_other_calls_are_unchanged(self):
        c = self.cp
        s = [L.seed_call(c(0.03, 0.01, 0.05)), L.seed_call(c(0.03, 0.01, 0.05))]
        self.assertEqual(L.reach_call(s), "CTX-REACHABLE")
        self.assertEqual(L.reach_call([L.seed_call(c(0.03, 0.01, 0.05)), L.seed_call(c(0.03, -0.01, 0.07))]), "INCONCLUSIVE")        # one seed not significant
        self.assertEqual(L.reach_call([L.seed_call(c(0.03, 0.01, 0.05)), L.seed_call(c(0.015, 0.002, 0.028))]), "INCONCLUSIVE")        # one seed not material
        self.assertEqual(L.reach_call([L.seed_call(c(0.001, -0.004, 0.009))] * 2), "NOT-REACHABLE")
        self.assertEqual(L.final_outcome("CTX-REACHABLE", True, True, False, False), "CTX-REACHABLE")                                 # no power needed to stand
        self.assertEqual(L.final_outcome("NOT-REACHABLE", True, True, False, False), "UNDERPOWERED")
        self.assertTrue(L.power_ok([c(0.03, 0.001, 0.06)] * 2))                             # C-POWER unchanged: lower CI > 0 and point in [0.015, 0.045]
        self.assertFalse(L.power_ok([c(0.05, 0.001, 0.09)] * 2))

    def test_gain_not_from_ctx_reads_the_shuffled_arm_with_the_new_rule(self):
        c = self.cp
        shuf = [L.seed_call(c(0.025, 0.002, 0.050))] * 2                                      # REACH under Amendment 2 (the sealed rule said INC)
        self.assertEqual(shuf, ["REACH", "REACH"])
        self.assertEqual(L.final_outcome("CTX-REACHABLE", True, True, False, True, all(x == "REACH" for x in shuf)), "GAIN-NOT-FROM-CTX")
        shuf1 = [L.seed_call(c(0.025, 0.002, 0.050)), L.seed_call(c(0.025, -0.002, 0.050))]
        self.assertEqual(L.final_outcome("CTX-REACHABLE", True, True, False, True, all(x == "REACH" for x in shuf1)), "CTX-REACHABLE")
        # REACHABILITY from the caller (the "built, tested, unreachable" class): analyze() must derive BOTH the raw call and the S-shuf call from L.seed_call, and the conversion from "REACH" on every seed
        src = open(os.path.join(HERE, "sh5_train_eval.py"), encoding="utf-8").read()
        self.assertIn('r["call"] = L.seed_call(r["delta_F"])', src)
        self.assertIn('r["call_shuf"] = L.seed_call(r["delta_F_shuf"])', src)
        self.assertIn('R["shuf_gain"] = bool(all(R["seeds"][f"s{s}"].get("call_shuf") == "REACH" for s in L.SEEDS))', src)
        self.assertIn('L.final_outcome(R["raw_call"], gates_ok, R["c_reach_ok"], R["null_label_void"], R["c_power_ok"], R["shuf_gain"])', src)

    def test_the_void_count_rule_is_unchanged_by_amendment_2_and_3(self):
        self.assertTrue(L.null_label_void([False, True, False]))
        self.assertFalse(L.null_label_void([False, False, False]))
        self.assertFalse(L.null_label_void([]))


class TestAmendment3Rule(unittest.TestCase):
    """AMENDMENT 3 (MM ruling 2026-10-19 s5): a C-NULL-LABEL replicate FIRES iff it satisfies the Amendment-2 CTX-REACHABLE rule (lower CI > 0 AND point >= +0.02) on the seed it is read on; the VOID COUNT rule (any of 3
    replicates fires) is unchanged.  The fire rule is `seed_call(c) == "REACH"`: ONE definition, no second copy of the thresholds."""
    cp = staticmethod(lambda point, lo, hi: {"point": point, "lo": lo, "hi": hi})
    NLF = 'all(seed_call(c) == "REACH" for c in cis_by_seed)'

    def test_planted_null_replicates(self):
        c = self.cp
        self.assertTrue(L.null_label_fires([c(0.025, 0.004, 0.046)]))                         # "+0.025, lower CI > 0": FIRES now (the sealed rule, lower CI > +0.02, said no)
        self.assertEqual(L.seed_call(c(0.025, 0.004, 0.046)), "REACH")                        # ... and the amended call reads the SAME replicate CTX-REACHABLE: the null and the claim agree
        self.assertFalse(L.null_label_fires([c(0.015, 0.004, 0.026)]))                        # "+0.015, significant": point < +0.02 -> does not fire
        self.assertFalse(L.null_label_fires([c(0.05, -0.002, 0.102)]))                        # "+0.05, lower CI < 0": not significant -> does not fire
        self.assertEqual(L.seed_call(c(0.015, 0.004, 0.026)), "INC")
        self.assertEqual(L.seed_call(c(0.05, -0.002, 0.102)), "INC")

    def test_boundaries_of_each_conjunct(self):
        c = self.cp
        self.assertTrue(L.null_label_fires([c(0.020, 0.001, 0.040)]))                         # point == +0.02 fires (>=)
        self.assertFalse(L.null_label_fires([c(0.0199, 0.001, 0.040)]))
        self.assertTrue(L.null_label_fires([c(0.030, 1e-9, 0.060)]))                          # lower CI barely above 0 fires
        self.assertFalse(L.null_label_fires([c(0.030, 0.0, 0.060)]))                          # lower CI == 0: strict
        self.assertFalse(L.null_label_fires([c(0.030, -1e-9, 0.060)]))
        self.assertTrue(L.null_label_fires([c(0.060, 0.020, 0.100)]))                         # the sealed edge lo == +0.02 (no fire before Amendment 3) fires
        self.assertFalse(L.null_label_fires([{"point": float("nan"), "lo": 0.01, "hi": 0.05}]))
        self.assertFalse(L.null_label_fires([{"point": 0.03, "lo": float("nan"), "hi": 0.05}]))
        self.assertFalse(L.null_label_fires([]))                                               # no seed read, no fire
        self.assertFalse(L.null_label_fires([c(0.03, 0.01, 0.05), c(0.03, -0.01, 0.07)]))     # several seeds: EVERY seed must be CTX-REACHABLE
        self.assertTrue(L.null_label_fires([c(0.03, 0.01, 0.05), c(0.025, 0.002, 0.05)]))

    def test_the_fire_rule_is_the_ctx_reachable_rule_and_nothing_else(self):
        import inspect
        src = inspect.getsource(L.null_label_fires)
        self.assertIn(self.NLF, src)                                                           # it calls seed_call ...
        body = src.split('"' * 3)[-1]
        for forbidden in ("INFO_MARGIN", "REACH_LO_MIN", "REACH_POINT_MIN", '["lo"]', '["point"]'):
            self.assertNotIn(forbidden, body)                                                  # ... and holds no second copy of a threshold or a field read
        rng = np.random.default_rng(3)
        for _ in range(400):                                                                   # on random intervals the two agree exactly
            p = float(rng.normal(0.02, 0.03))
            lo, hi = p - abs(float(rng.normal(0.02, 0.02))), p + abs(float(rng.normal(0.02, 0.02)))
            ci = {"point": p, "lo": lo, "hi": hi}
            self.assertEqual(L.null_label_fires([ci]), L.seed_call(ci) == "REACH")

    def test_a_mutation_of_seed_call_moves_the_machinery_null_with_it(self):
        """reuse, not a copy: change the CTX-REACHABLE rule and the C-NULL-LABEL fire follows, and the table fails on the NULL-LABEL checks as well as on the call's."""
        c = self.cp
        old = 'if ci["lo"] > REACH_LO_MIN and ci["point"] >= REACH_POINT_MIN:'
        for new, planted in (('if ci["lo"] > REACH_LO_MIN:', c(0.015, 0.004, 0.026)),                                  # drop the point conjunct
                             ('if ci["point"] >= REACH_POINT_MIN:', c(0.05, -0.002, 0.102))):                            # drop the lower-CI conjunct
            m = mutated(L, old, new)
            self.assertFalse(L.null_label_fires([planted]))
            self.assertTrue(m.null_label_fires([planted]))                                     # the mutated rule fires on a planted replicate the real one rejects: the VOID reads seed_call
            self.assertTrue(any(f.startswith("A3 planted null replicate") for f in L.branch_table_failures(m)), L.branch_table_failures(m))

    def test_red_reverting_to_the_sealed_rule_misses_the_planted_025_replicate(self):
        m = mutated(L, self.NLF, 'all(c["lo"] > INFO_MARGIN for c in cis_by_seed)')
        planted = self.cp(0.025, 0.004, 0.046)
        self.assertTrue(L.null_label_fires([planted]))
        self.assertFalse(m.null_label_fires([planted]))                                        # the sealed rule does not fire on it: it would NOT void a machinery that manufactures +0.025
        fails = L.branch_table_failures(m)
        self.assertTrue(any("+0.025, lower CI > 0" in f for f in fails), fails)

    def test_red_dropping_either_conjunct_passes_a_planted_case_it_should_reject(self):
        m_pt = mutated(L, self.NLF, 'all(c["lo"] > REACH_LO_MIN for c in cis_by_seed)')       # drops the point conjunct
        m_lo = mutated(L, self.NLF, 'all(c["point"] >= REACH_POINT_MIN for c in cis_by_seed)')   # drops the lower-CI conjunct
        imm, ins = self.cp(0.015, 0.004, 0.026), self.cp(0.05, -0.002, 0.102)
        self.assertFalse(L.null_label_fires([imm]))
        self.assertTrue(m_pt.null_label_fires([imm]))
        self.assertFalse(L.null_label_fires([ins]))
        self.assertTrue(m_lo.null_label_fires([ins]))
        self.assertTrue(any("+0.015" in f for f in L.branch_table_failures(m_pt)))
        self.assertTrue(any("+0.05, lower CI < 0" in f for f in L.branch_table_failures(m_lo)))

    def test_void_follows_the_fires_and_the_amended_call_cannot_pass_a_machinery_that_voids(self):
        c = self.cp
        gain = c(0.03, 0.008, 0.052)                                                           # a machinery that manufactures +0.03: the amended call reads it CTX-REACHABLE ...
        self.assertEqual(L.seed_call(gain), "REACH")
        void = L.null_label_void([L.null_label_fires([gain]), False, False])
        self.assertTrue(void)                                                                  # ... and C-NULL-LABEL now VOIDs it (under the sealed rule, lower CI 0.008 < 0.02, it did not)
        self.assertEqual(L.final_outcome("CTX-REACHABLE", True, True, void, True), "VOID(C-NULL-LABEL)")
        self.assertFalse(L.null_label_void([L.null_label_fires([c(0.001, -0.004, 0.006)])] * 3))   # a clean replicate does not

    def test_the_count_rule_and_the_other_calls_are_unchanged(self):
        self.assertTrue(L.null_label_void([False, False, True]))
        self.assertFalse(L.null_label_void([False, False, False]))
        self.assertEqual(L.final_outcome("NOT-REACHABLE", True, True, False, True), "NOT-REACHABLE")
        self.assertEqual(L.final_outcome("INCONCLUSIVE", True, True, False, True), "INCONCLUSIVE")
        self.assertEqual(L.final_outcome("CTX-REACHABLE", True, True, False, True, True), "GAIN-NOT-FROM-CTX")
        self.assertEqual(L.NULL_LABEL_REPS, 3)

    def test_the_wiring_in_analyze_reaches_the_fire_rule(self):
        """REACHABILITY from the caller: analyze() builds `fires` through L.null_label_fires and the run's VOID from them (the "built, tested, unreachable" class)."""
        src = open(os.path.join(HERE, "sh5_train_eval.py"), encoding="utf-8").read()
        self.assertIn('row["fires"] = L.null_label_fires([row["delta_F"]])', src)
        self.assertIn('R["null_label_void"] = L.null_label_void([r["fires"] for r in nulls["nulllabel"]])', src)
        self.assertIn('for rep in range(L.NULL_REPS if arm == "nullprox" else L.NULL_LABEL_REPS):', src)

    def test_banked_cpu_false_fire_rows_read_by_the_real_function(self):
        path = os.path.join(HERE, "..", "work", "falsefire_sim.json")
        if not os.path.exists(path):
            self.skipTest("work/falsefire_sim.json is not in this tree")
        d = json.load(open(path, encoding="utf-8"))
        self.assertEqual(len(d["rows"]), 16)
        self.assertEqual(sum(L.null_label_fires([r["delta_F"]]) for r in d["rows"]), 0)         # 16 noise-trained replicates: none fires under the new rule (the false-fire control is not lost)
        self.assertEqual(len(d["pos_rows"]), 4)
        self.assertEqual(sum(L.null_label_fires([r["delta_F"]]) for r in d["pos_rows"]), 4)     # 4 replicates whose branch reads the ctx: all fire


class TestAmend2PowerArithmetic(unittest.TestCase):
    """the arithmetic of `sh5_amend2_power.py` (not pinned: the checker of the amendment's power table)"""

    def test_normal_tail_and_bivariate_limits(self):
        import sh5_amend2_power as P
        self.assertAlmostEqual(P.Q(1.96), 0.0249979, places=6)
        for a in (-1.0, 0.0, 0.66, 2.5):
            self.assertAlmostEqual(P.bvn_both_above(a, 0.0), P.Q(a) ** 2, places=12)           # independent
            self.assertAlmostEqual(P.bvn_both_above(a, 1.0), P.Q(a), places=12)                # identical
            v = [P.bvn_both_above(a, r) for r in (0.0, 0.3, 0.6, 0.9, 1.0)]
            self.assertTrue(all(v[i] <= v[i + 1] + 1e-12 for i in range(len(v) - 1)), v)       # monotone in the correlation
        self.assertAlmostEqual(P.bvn_both_above(0.0, 0.5), 0.25 + np.arcsin(0.5) / (2 * np.pi), places=6)   # the closed form at a = 0

    def test_thresholds_name_the_binding_conjunct(self):
        import sh5_amend2_power as P
        full = P.thresholds(0.00765)
        loc = P.thresholds(0.01543)
        self.assertAlmostEqual(full["sealed_point_needed"], 0.02 + 1.96 * 0.00765, places=9)
        self.assertEqual(full["binding_conjunct_of_new_rule"], "point >= +0.02")              # FULL: 1.96 se = 0.015 < 0.02
        self.assertAlmostEqual(full["new_point_needed"], 0.02, places=12)
        self.assertEqual(loc["binding_conjunct_of_new_rule"], "lower CI > 0")                 # LOCAL: 1.96 se = 0.030 > 0.02
        self.assertAlmostEqual(loc["new_point_needed"], 1.96 * 0.01543, places=9)

    def test_new_rule_is_never_less_powerful_than_the_sealed_rule(self):
        import sh5_amend2_power as P
        for se in (0.003, 0.00765, 0.01543, 0.03):
            th = P.thresholds(se)
            self.assertLessEqual(th["new_point_needed"], th["sealed_point_needed"] + 1e-15)
            for eff in (0.0, 0.01, 0.03, 0.05):
                a = P.p_pass(eff, se, th["sealed_point_needed"], 1.0)["per_seed"]
                b = P.p_pass(eff, se, th["new_point_needed"], 1.0)["per_seed"]
                self.assertGreaterEqual(b, a - 1e-15)


class TestAmend3PowerArithmetic(unittest.TestCase):
    """the arithmetic of `sh5_amend3_power.py` (not pinned: the checker of the amendment's power table)"""

    def test_threshold_found_on_the_real_function_equals_the_closed_form(self):
        import sh5_amend3_power as P
        for se in (0.003, 0.00765, 0.01543, 0.03):
            t = P.bisect_threshold(L.null_label_fires, se)
            self.assertAlmostEqual(t, max(1.96 * se, 0.02), places=9)                            # the power arithmetic IS the library's rule
            ci = lambda p: {"point": p, "lo": p - 1.96 * se, "hi": p + 1.96 * se}                # noqa: E731
            self.assertFalse(L.null_label_fires([ci(t - 1e-7)]))
            self.assertTrue(L.null_label_fires([ci(t + 1e-7)]))
            self.assertAlmostEqual(P.bisect_threshold(lambda cs: all(c["lo"] > 0.02 for c in cs), se), 0.02 + 1.96 * se, places=9)   # the sealed rule

    def test_void_bounds_and_the_new_rule_never_less_sensitive(self):
        import sh5_amend3_power as P
        self.assertEqual(P.void_bounds(0.0), [0.0, 0.0])
        self.assertEqual(P.void_bounds(1.0), [1.0, 1.0])
        lo, hi = P.void_bounds(0.1)
        self.assertAlmostEqual(lo, 0.1)
        self.assertAlmostEqual(hi, 1 - 0.9 ** 3)
        for se in (0.00765, 0.01543):
            tn, ts = max(1.96 * se, 0.02), 0.02 + 1.96 * se
            tab = P.rule_table(se, ts, tn, se)
            for e, row in tab.items():
                self.assertGreaterEqual(row["new"]["per_replicate_fires"], row["sealed"]["per_replicate_fires"] - 1e-15)       # the new rule fires whenever the sealed one does
                self.assertGreaterEqual(row["new"]["P_VOID_bounds_identical_to_independent"][0], row["sealed"]["P_VOID_bounds_identical_to_independent"][0] - 1e-15)

    def test_void_power_of_the_new_rule_equals_the_per_seed_call_power(self):
        """the null is read by the claim's own test: at every effect the lower bound of P(VOID) IS the per-seed CTX-REACHABLE power (Amendment 2's table, same inputs)"""
        import sh5_amend2_power as P2
        import sh5_amend3_power as P
        for se in (0.00765, 0.01543):
            th2 = P2.thresholds(se)
            tab = P.rule_table(se, th2["sealed_point_needed"], th2["new_point_needed"], se)
            for eff in (0.0, 0.02, 0.03, 0.04):
                self.assertAlmostEqual(tab[f"{eff:+.3f}"]["new"]["P_VOID_bounds_identical_to_independent"][0], P2.p_pass(eff, se, th2["new_point_needed"], 1.0)["per_seed"], places=12)


class TestPickLevel(unittest.TestCase):
    def test_identity_when_the_dac_row_is_unchanged(self):
        D, _ = SY.make_synth(T=40, seed=5)
        FL, PK = L.refs()
        an = L.Analyzer(D["lab"][..., L.DAC_COL], D["code"], D["log_idx"], D["n_logs"], B=300)
        stats = PK.Stats(D["log_idx"], an.draws, FL, n_logs=D["n_logs"])
        pl = L.pick_level(D["z6"], D["lab"], D["z6"][..., L.DAC_COL], D["ep_s0"], stats)
        for k in ("rule[S]", "EP[S]"):
            d = pl[k]["vs_P2"] if k == "rule[S]" else pl["EP[S] minus EP"]
            self.assertEqual(d["d_dac0"]["point"], 0.0 if k == "rule[S]" else 0.0)
        self.assertEqual(pl["rule[S]"]["vs_P2"]["d_pdms_x100"]["point"], 0.0)            # C-ID at the pick level

    def test_a_perfect_dac_row_moves_the_picks(self):
        D, _ = SY.make_synth(T=40, seed=6)
        FL, PK = L.refs()
        lab_dac = D["lab"][..., L.DAC_COL]
        an = L.Analyzer(lab_dac, D["code"], D["log_idx"], D["n_logs"], B=300)
        stats = PK.Stats(D["log_idx"], an.draws, FL, n_logs=D["n_logs"])
        z = np.where(lab_dac == 1.0, 8.0, -8.0)
        pl = L.pick_level(D["z6"], D["lab"], z, None, stats)
        self.assertEqual(pl["rule[S]"]["dac0_rate"]["point"], 0.0)

    def test_reliability_is_one_when_calibrated(self):
        rng = np.random.default_rng(0)
        N = 6000
        z6 = rng.normal(0, 1, (N, 64, 6))
        p_safe = L._sig(rng.normal(1.0, 1.5, (N, 64)))
        lab = (rng.random((N, 64)) < p_safe).astype(np.float64)
        z6[..., L.DAC_COL] = np.log(p_safe / (1 - p_safe))
        FL, PK = L.refs()
        li = np.arange(N) % 20
        draws = FL.boot_draws(20, 300)
        stats = PK.Stats(li, draws, FL, n_logs=20)
        # unconditional picks (rule on random columns) leave the pick independent of the DAC label: ratio ~ 1 for a calibrated row
        r = L.reliability_at_picks(z6, z6[..., L.DAC_COL], lab, stats)
        self.assertAlmostEqual(r["all"]["point"], 1.0, delta=0.12)


class TestGpuLock(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(dir=os.environ.get("SH5_TMP"))
        self.lock, self.mark = os.path.join(self.d, "devbox_gpu.lock"), os.path.join(self.d, "devbox_gpu.chain_active")
        self.kw = dict(lock=self.lock, mark=self.mark, used_fn=lambda: 1500, alive_fn=lambda p: False, sleep_fn=lambda s: None, compute_fn=lambda: [])

    def test_takes_and_releases_only_its_own_body(self):
        body, why = GL.try_lock(**self.kw)
        self.assertEqual(why, "ok")
        self.assertTrue(os.path.exists(self.lock))
        self.assertTrue(GL.release(body, self.lock))
        self.assertFalse(os.path.exists(self.lock))

    def test_foreign_lock_is_never_removed(self):
        for holder in ({"job": "x", "pid": 1}, {"owner": "l1_stepA", "pid": 2}, "not json"):
            open(self.lock, "w").write(json.dumps(holder) if isinstance(holder, dict) else holder)
            os.utime(self.lock, (time.time() - 86400, time.time() - 86400))          # even a day old
            before = open(self.lock).read()
            body, why = GL.try_lock(**self.kw)
            self.assertIsNone(body)
            self.assertIn("FOREIGN", why)
            self.assertEqual(open(self.lock).read(), before)                          # byte-identical
            os.remove(self.lock)

    def test_own_stale_lock_replaced_only_if_dead_and_old(self):
        open(self.lock, "w").write(json.dumps({"owner": GL.OWNER, "pid": 99999999, "job": "j"}))
        body, why = GL.try_lock(**self.kw)
        self.assertIsNone(body)                                                       # dead pid but fresh: refused
        os.utime(self.lock, (time.time() - 3600, time.time() - 3600))
        body, why = GL.try_lock(**self.kw)
        self.assertEqual(why, "ok")                                                   # dead and an hour old: replaced
        GL.release(body, self.lock)

    def test_chain_active_waits(self):
        open(self.mark, "w").write("x")
        body, why = GL.try_lock(**self.kw)
        self.assertIsNone(body)
        self.assertIn("chain_active", why)

    def test_compute_process_gives_the_lock_back_and_cg_is_ignored(self):
        kw = dict(self.kw, compute_fn=lambda: [4242])
        body, why = GL.try_lock(**kw)
        self.assertIsNone(body)
        self.assertFalse(os.path.exists(self.lock))                                   # given back
        table = ("| Processor:  GPU PID Type Process name |\n|    0   N/A  N/A     1234    C+G   ...explorer.exe  N/A |\n"
                 "|    0   N/A  N/A     5678      C   ...python.exe   N/A |")
        self.assertEqual(GL.compute_pids(table), [5678])                              # C+G ignored, C detected
        self.assertEqual(GL.compute_pids("|    0   N/A  N/A     1234    C+G   ...explorer.exe  N/A |"), [])

    def test_red_mutated_compute_check_would_not_refuse(self):
        m = mutated(GL, 'if len(t) >= 6 and t[0].isdigit() and t[3].isdigit() and t[4] == "C":', 'if len(t) >= 6 and t[0].isdigit() and t[3].isdigit() and "C" in t[4]:')
        self.assertEqual(m.compute_pids("|    0   N/A  N/A     1234    C+G   ...explorer.exe  N/A |"), [1234])      # RED: the mutant counts a desktop app as a compute job
        self.assertEqual(GL.compute_pids("|    0   N/A  N/A     1234    C+G   ...explorer.exe  N/A |"), [])

    def test_unreadable_nvidia_smi_fails_closed(self):
        import subprocess as sp
        orig = sp.run
        sp.run = lambda *a, **k: (_ for _ in ()).throw(OSError("no nvidia-smi"))
        try:
            self.assertEqual(GL.compute_pids(), [-1])
            self.assertGreater(GL.used_mib(), 10 ** 6)
        finally:
            sp.run = orig

    def test_ram_gate_before_the_lock(self):
        logs = []
        n = {"i": 0}

        def avail():
            n["i"] += 1
            return 2.0 if n["i"] < 3 else 20.0
        body = GL.acquire(10, log=logs.append, avail_fn=avail, min_avail_gb=8.0, poll_s=0, **self.kw)
        self.assertIsNotNone(body)
        self.assertTrue(any("RAM available" in l for l in logs))
        self.assertTrue(n["i"] >= 3)
        GL.release(body, self.lock)

    def test_max_wait_expires_without_running(self):
        open(self.lock, "w").write(json.dumps({"job": "other"}))
        t = {"v": 0.0}
        body = GL.acquire(5, log=lambda m: None, avail_fn=lambda: 99, now_fn=lambda: t.__setitem__("v", t["v"] + 10) or t["v"], poll_s=0, **self.kw)
        self.assertIsNone(body)

    def test_require_locked(self):
        os.environ.pop("SH5_GPU_LOCKED", None)
        with self.assertRaises(SystemExit) as c:
            GL.require_locked("cuda")
        self.assertEqual(c.exception.code, 4)
        GL.require_locked("cpu")


if __name__ == "__main__":
    unittest.main(verbosity=1)
