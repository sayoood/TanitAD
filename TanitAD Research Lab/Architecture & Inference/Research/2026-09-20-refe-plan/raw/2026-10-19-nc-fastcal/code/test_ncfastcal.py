"""Synthetic CPU-only tests for NC-FASTCAL (no bank, no GPU, no real label).  Run: python -m unittest -v test_ncfastcal

Every decision function has a RED mutation that must break the branch table; the calibration code path (per-stratum, per-fold, training-folds-only, the NC column) has planted cases that a mutated
copy must fail; the C-NULL control has a planted case in which it CAN fire (power, RETRACTION_LOG R51) and one in which it must stay silent.  Reach checks are stated on PLANTED inputs only.
"""
from __future__ import annotations

import os
import sys
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ncfastcal_lib as L                                                                 # noqa: E402

LIB = os.path.join(HERE, "ncfastcal_lib.py")
LIB_SRC = open(LIB, "r", encoding="utf-8").read()
P = L.load_pickrule()
F = L.load_fastneg()


def mutated(old: str, new: str):
    """A copy of ncfastcal_lib with the FIRST occurrence of `old` replaced by `new` (the occurrence must exist)."""
    assert old in LIB_SRC, f"mutation anchor not found: {old!r}"
    ns = types.ModuleType("ncfastcal_lib_mut")
    ns.__file__ = LIB
    exec(compile(LIB_SRC.replace(old, new, 1), LIB, "exec"), ns.__dict__)
    return ns


def sig(u):
    return 1.0 / (1.0 + np.exp(-np.asarray(u, np.float64)))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the planted bank: speed-conditional miscalibration of the NC head (kind "speed") or uniform miscalibration (kind "global")
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def planted_bank(kind: str, N: int = 3000, n_logs: int = 60, seed: int = 5):
    """64 proposals per set, strata Q1..Q4 uniform per proposal.  The raw NC logit is u.  kind 'speed': the truth is P(safe) = sigmoid(u) in Q1..Q3 and sigmoid(0.25 u - 2.5) in Q4 (the head is
    overconfident only in the fast stratum).  kind 'global': the truth is sigmoid(0.3 u - 1.5) everywhere.  The composite's EP column favours Q4 by +2 logits (fast looks like progress); the
    EP LABEL is higher in Q4.  P2's EP column carries no speed bonus."""
    rng = np.random.default_rng(seed)
    code = rng.integers(1, 5, size=(N, 64)).astype(np.uint8)
    log_idx = rng.integers(0, n_logs, size=N).astype(np.int64)
    fold = (log_idx % L.N_FOLDS).astype(np.int64)
    u = rng.normal(0.0, 1.5, size=(N, 64))
    if kind == "speed":
        q = np.where(code == 4, sig(0.25 * u - 2.5), sig(u))
    elif kind == "global":
        q = sig(0.3 * u - 1.5)
    else:
        raise ValueError(kind)
    nc_lab = (rng.random((N, 64)) < q).astype(np.float32)
    e_p2 = rng.normal(0.0, 1.0, size=(N, 64))
    z6 = np.zeros((N, 64, 6), np.float32)
    z6[..., L.NC], z6[..., L.DAC], z6[..., L.EP], z6[..., L.TTC], z6[..., L.CMF], z6[..., L.DDC] = u, 6.0, e_p2, 4.0, 4.0, 0.0
    lab = np.ones((N, 64, 6), np.float32)
    lab[..., L.NC] = nc_lab
    lab[..., L.EP] = (0.4 + 0.5 * (code == 4)).astype(np.float32)
    ep_comp = (e_p2 + 2.0 * (code == 4)).astype(np.float32)
    tokens = np.array([f"t{i // 2}" for i in range(N)])
    return {"z6": z6, "lab": lab, "code": code, "fold": fold, "log_idx": log_idx, "ep": ep_comp, "n_logs": n_logs, "tokens": tokens}


def stats_for(bank, B: int = 1500, seed: int = 11):
    return P.Stats(bank["log_idx"], F.boot_draws(bank["n_logs"], B, seed), F, n_logs=bank["n_logs"])


def evaluate(bank, mod=L, code=None, S=None):
    """The shared decision path on a planted bank, with the calibration code of `mod` (this module or a mutated copy)."""
    code = bank["code"] if code is None else code
    S = S or stats_for(bank)
    pdms = P.pdms_from_labels(bank["lab"])
    pk_p = P.rule_pick(bank["z6"])
    x = bank["z6"][..., L.NC].astype(np.float64)
    params, _ = mod.fit_calibration(x, bank["lab"][..., L.NC], code, bank["fold"])
    nc = mod.apply_calibration(x, code, bank["fold"], params)
    d = L.decide(P, bank["z6"], bank["lab"], {0: bank["ep"], 1: bank["ep"]}, nc, pdms, pk_p, S)
    return d, params


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestPlatt(unittest.TestCase):
    def test_recovers_planted_parameters(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 2.0, size=200000)
        t = (rng.random(200000) < sig(0.6 * x - 1.2)).astype(np.float64)
        r = L.platt_fit(x, t)
        self.assertTrue(r["ok"], r)
        self.assertLess(abs(r["a"] - 0.6), 0.03)
        self.assertLess(abs(r["b"] + 1.2), 0.03)
        self.assertLess(r["grad_inf"], 1e-8)

    def test_matches_lbfgs_on_soft_targets(self):
        from scipy.optimize import minimize
        rng = np.random.default_rng(2)
        x = rng.normal(0, 2.0, size=50000)
        t = np.clip(sig(0.8 * x - 0.4) + rng.normal(0, 0.15, 50000), 0.0, 1.0)
        t[::7] = 0.5
        r = L.platt_fit(x, t)
        f = lambda ab: float(np.mean(np.logaddexp(0.0, ab[0] * x + ab[1]) - t * (ab[0] * x + ab[1])))              # noqa: E731
        g = lambda ab: np.array([np.mean((sig(ab[0] * x + ab[1]) - t) * x), np.mean(sig(ab[0] * x + ab[1]) - t)])    # noqa: E731
        o = minimize(f, np.array([1.0, 0.0]), jac=g, method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 500})
        self.assertLess(abs(r["a"] - o.x[0]), 1e-6)
        self.assertLess(abs(r["b"] - o.x[1]), 1e-6)

    def test_identity_is_a_fixed_point_when_the_head_is_calibrated(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 2.0, size=20000)
        r = L.platt_fit(x, sig(x))                                    # soft target = the head's own probability
        self.assertTrue(r["ok"])
        self.assertLess(abs(r["a"] - 1.0), 1e-6)
        self.assertLess(abs(r["b"]), 1e-6)

    def test_deterministic_bit_for_bit(self):
        rng = np.random.default_rng(4)
        x = rng.normal(0, 2.0, size=30000)
        t = (rng.random(30000) < sig(0.5 * x)).astype(np.float64)
        r1, r2 = L.platt_fit(x, t), L.platt_fit(x, t)
        self.assertEqual((r1["a"], r1["b"], r1["iters"]), (r2["a"], r2["b"], r2["iters"]))

    def test_failure_rules_fall_back_to_identity(self):
        rng = np.random.default_rng(5)
        x = rng.normal(0, 2.0, size=5000)
        few = L.platt_fit(x[:999], np.ones(999))
        self.assertFalse(few["ok"])
        self.assertEqual((few["a"], few["b"]), (1.0, 0.0))
        allsafe = L.platt_fit(x, np.ones(5000))                       # no finite optimum (b -> +inf): the REGISTERED rule accepts the point where the gradient reaches 1e-8, which is far from the identity
        self.assertGreater(allsafe["b"], 10.0)                       # documented edge of the registered rule (|param| <= 50); the result lists every fitted parameter
        self.assertLessEqual(abs(allsafe["b"]), L.MAX_ABS_PARAM)
        sep = L.platt_fit(x, (x > 0).astype(np.float64))              # perfectly separable: a -> +inf
        self.assertFalse(sep["ok"])
        self.assertEqual((sep["a"], sep["b"]), (1.0, 0.0))
        const = L.platt_fit(np.full(5000, 2.0), (rng.random(5000) < 0.3).astype(np.float64))      # constant input: singular Hessian
        self.assertFalse(const["ok"])
        self.assertEqual((const["a"], const["b"]), (1.0, 0.0))


class TestCalibrationCodePath(unittest.TestCase):
    def test_reach_and_fold_table_passes_unmutated(self):
        self.assertEqual(L.reach_table_failures(L), [])

    def test_every_calibration_mutation_is_caught(self):
        for old, new, why in [
            ("A = params[fi, ci, 0]", "A = params[fi, 0, 0]", "stratum ignored in a"),
            ("B = params[fi, ci, 1]", "B = params[fi, 0, 1]", "stratum ignored in b"),
            ("A = params[fi, ci, 0]", "A = params[0, ci, 0]", "fold ignored in a"),
            ("B = params[fi, ci, 1]", "B = params[0, ci, 1]", "fold ignored in b"),
            ("return A * np.asarray(x, np.float64) + B", "return A * np.asarray(x, np.float64)", "offset dropped"),
            ("return A * np.asarray(x, np.float64) + B", "return np.asarray(x, np.float64) + B", "scale dropped"),
            ("z[..., NC] = nc", "z[..., DAC] = nc", "wrong column written"),
            ("tr = fold != k", "tr = fold >= 0", "fits on ALL sets (reads the held-out labels)"),
            ("tr = fold != k", "tr = fold == k", "fits on the HELD-OUT fold"),
        ]:
            m = mutated(old, new)
            self.assertNotEqual(m.reach_table_failures(m), [], f"mutation not caught: {why}")

    def test_the_local_rule_agrees_with_the_pinned_one(self):
        rng = np.random.default_rng(7)
        z = rng.normal(0, 2.0, size=(500, 64, 6))
        self.assertTrue(np.array_equal(L._rule_pick(z), P.rule_pick(z)))

    def test_float32_to_float64_identity_is_exact(self):
        rng = np.random.default_rng(8)
        z6 = rng.normal(0, 3.0, size=(100, 64, 6)).astype(np.float32)
        code = rng.integers(0, 5, size=(100, 64)).astype(np.uint8)
        fold = (np.arange(100) % 5).astype(np.int64)
        nc = L.apply_calibration(z6[..., 0], code, fold, L.identity_params())
        self.assertTrue(np.array_equal(P.rule_pick(L.composite(z6, nc=nc)), P.rule_pick(z6)))
        self.assertTrue(np.array_equal(L.composite(z6, nc=nc), z6.astype(np.float64)))

    def test_calibration_improves_heldout_bce_only_where_the_head_is_miscalibrated(self):
        b = planted_bank("speed")
        x = b["z6"][..., L.NC].astype(np.float64)
        params, _ = L.fit_calibration(x, b["lab"][..., L.NC], b["code"], b["fold"])
        r = L.heldout_bce(x, b["lab"][..., L.NC], b["code"], b["fold"], params)
        self.assertGreater(r["Q4"]["improvement"], 0.2)
        self.assertLess(abs(r["Q1"]["improvement"]), 0.01)
        self.assertLess(params[:, 4, 0].mean(), 0.5)                  # Q4 slope near 0.25
        self.assertGreater(params[:, 1, 0].mean(), 0.9)              # Q1 slope near 1

    def test_a_fit_that_ignores_the_strata_is_caught_by_the_planted_bank(self):
        m = mutated("m = cs == s", "m = cs >= 0")
        b = planted_bank("speed")
        x = b["z6"][..., L.NC].astype(np.float64)
        pm, _ = m.fit_calibration(x, b["lab"][..., L.NC], b["code"], b["fold"])
        self.assertLess(abs(pm[:, 4, 0].mean() - pm[:, 1, 0].mean()), 1e-9)            # the mutant gives every stratum the same fit
        pg, _ = L.fit_calibration(x, b["lab"][..., L.NC], b["code"], b["fold"])
        self.assertGreater(pg[:, 1, 0].mean() - pg[:, 4, 0].mean(), 0.4)               # the real fit separates them


class TestDecision(unittest.TestCase):
    def test_branch_table_passes_unmutated(self):
        self.assertEqual(L.branch_table_failures(L), [])

    def test_every_decision_function_has_a_red_mutation(self):
        muts = [
            ('return bool(d_nc_ci["hi"] < NC_UB)', 'return bool(d_nc_ci["hi"] <= NC_UB)', "(i) strict < relaxed to <="),
            ('return bool(d_nc_ci["hi"] < NC_UB)', 'return bool(d_nc_ci["lo"] < NC_UB)', "(i) reads the lower bound"),
            ('return bool(d_nc_ci["hi"] < NC_UB)', 'return bool(not (d_nc_ci["hi"] >= NC_UB))', "(i) NaN becomes True"),
            ("NC_UB = 0.001 ", "NC_UB = 0.0011 ", "(i) bound loosened"),
            ("NC_UB = 0.001 ", "NC_UB = 0.0009 ", "(i) bound tightened"),
            ('return bool(d_pdms_ci["lo"] >= PDMS_LB)', 'return bool(d_pdms_ci["lo"] > PDMS_LB)', "(ii) >= tightened to >"),
            ('return bool(d_pdms_ci["lo"] >= PDMS_LB)', 'return bool(d_pdms_ci["hi"] >= PDMS_LB)', "(ii) reads the upper bound"),
            ('return bool(d_pdms_ci["lo"] >= PDMS_LB)', 'return bool(not (d_pdms_ci["lo"] < PDMS_LB))', "(ii) NaN becomes True"),
            ("PDMS_LB = 0.0 ", "PDMS_LB = 0.5 ", "(ii) bound raised"),
            ("return cond_i(d_nc_ci) and cond_ii(d_pdms_ci)", "return cond_i(d_nc_ci) or cond_ii(d_pdms_ci)", "and -> or"),
            ("return cond_i(d_nc_ci) and cond_ii(d_pdms_ci)", "return cond_i(d_nc_ci)", "(ii) dropped"),
            ("all(seed_ok(a, b) for a, b in per_seed)", "any(seed_ok(a, b) for a, b in per_seed)", "both seeds -> either seed"),
            ("len(per_seed) > 0 and all(seed_ok", "all(seed_ok", "empty list recovers"),
            ("all(cond_i(a) for a, _ in per_seed)", "any(cond_i(a) for a, _ in per_seed)", "cond_i_only: all -> any"),
            ("NULL_VOID_FIRES = 2", "NULL_VOID_FIRES = 3", "void threshold 2 -> 3"),
            ("sum(1 for f in fires if f) >= NULL_VOID_FIRES", "sum(1 for f in fires if f) > NULL_VOID_FIRES", "void >= -> >"),
            ("    if void:\n", "    if False:\n", "void never wins"),
            ('"NO-RECOVER": "NC calibration closes;', '"NO-RECOVER": "a 0-train pick-rule registration;', "next arm of NO-RECOVER swapped"),
        ]
        for old, new, why in muts:
            m = mutated(old, new)
            self.assertNotEqual(m.branch_table_failures(m), [], f"mutation not caught: {why}")


class TestPermutation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9)
        self.N = 400
        self.code = rng.integers(0, 5, size=(self.N, 64)).astype(np.uint8)
        self.tokens = np.array([f"t{i // 3}" if i % 7 else f"solo{i}" for i in range(self.N)])      # groups of 3 plus singletons
        self.groups = L.token_groups(self.tokens)

    def props_fail(self, mod):
        bad = []
        cp = mod.permute_codes_within_token(self.code, self.groups, 123)
        if np.array_equal(cp, self.code):
            bad.append("no change")
        for g in self.groups:
            if not np.array_equal(np.sort(cp[g].ravel()), np.sort(self.code[g].ravel())):
                bad.append("token multiset not preserved")
                break
        if not np.array_equal(cp, mod.permute_codes_within_token(self.code, self.groups, 123)):
            bad.append("not deterministic")
        if np.array_equal(cp, mod.permute_codes_within_token(self.code, self.groups, 124)):
            bad.append("seed has no effect")
        return bad

    def test_properties_hold_and_mutants_fail(self):
        self.assertEqual(self.props_fail(L), [])
        self.assertTrue(any(len(g) == 1 for g in self.groups) and any(len(g) == 3 for g in self.groups))
        self.assertEqual(self.props_fail(mutated("perm = rng.permutation(pooled.size)", "perm = np.arange(pooled.size)")) != [], True)
        # cross-token pooling (everything is one token) breaks the per-token multiset
        m = mutated("    for idx in groups:\n        pooled", "    for idx in [np.concatenate(groups)]:\n        pooled")
        self.assertNotEqual(self.props_fail(m), [])

    def test_the_permutation_breaks_the_proposal_level_link(self):
        cp = L.permute_codes_within_token(self.code, self.groups, 5)
        self.assertLess(float((cp == self.code).mean()), 0.3)         # chance level for 5 strata is 0.2


class TestStrataJoin(unittest.TestCase):
    def build(self):
        rng = np.random.default_rng(10)
        N = 60
        v = np.abs(rng.normal(4.0, 2.0, size=(N, 64)))
        pick = rng.integers(0, 64, size=N)
        v_b = v[np.arange(N), pick]
        v_b[:5] = 0.1
        r, cuts, code = F.build_strata(v, v_b)
        counts = {n: int((code == i).sum()) for i, n in enumerate(L.STRATA_NAMES)}
        sj = {"cut_points": {"c25": float(cuts[0]), "c50": float(cuts[1]), "c75": float(cuts[2])}, "rows_per_stratum": counts}
        names = np.array([f"L{i % 7}" for i in range(7)])
        log_idx = np.arange(N) % 7
        arr = {"log_names": names, "log_idx": log_idx, "top8": np.tile(pick[:, None], (1, 8)), "z6": np.zeros((N, 64, 6), np.float32)}
        speeds = {"key_log": names[log_idx], "pick": pick, "r": r, "v_b": v_b, "code": code}
        return arr, speeds, sj

    def test_consistent_join_passes_and_each_tamper_fails(self):
        arr, speeds, sj = self.build()
        self.assertTrue(L.strata_join_checks(arr, speeds, sj)["pass"])
        bad = dict(speeds, key_log=np.roll(speeds["key_log"], 1))
        self.assertFalse(L.strata_join_checks(arr, bad, sj)["pass"])
        bad = dict(speeds, pick=(speeds["pick"] + 1) % 64)
        self.assertFalse(L.strata_join_checks(arr, bad, sj)["pass"])
        c2 = speeds["code"].copy()
        c2[10, 3] = (c2[10, 3] % 4) + 1
        self.assertFalse(L.strata_join_checks(arr, dict(speeds, code=c2), sj)["pass"])
        sj2 = dict(sj, cut_points=dict(sj["cut_points"], c50=sj["cut_points"]["c50"] + 0.05))
        self.assertFalse(L.strata_join_checks(arr, speeds, sj2)["pass"])


class TestLandedComparison(unittest.TestCase):
    def test_compare_landed(self):
        row = {f: {"point": 0.1, "lo": 0.05, "hi": 0.2} for f in L.LANDED_FIELDS}
        row["n_changed_vs_p2"] = 5
        self.assertTrue(L.compare_landed(row, row)["pass"])
        other = {f: dict(v) if isinstance(v, dict) else v for f, v in row.items()}
        other["d_pdms_x100"]["lo"] = 0.05 + 1e-9
        self.assertFalse(L.compare_landed(row, other)["pass"])


class TestPlantedPowerOfTheControls(unittest.TestCase):
    """The control must be able to fire (power) and stay silent where it should: planted banks, the real decision path, the real null."""

    def test_speed_conditional_miscalibration_recovers_on_true_strata_and_the_null_stays_silent(self):
        b = planted_bank("speed")
        S = stats_for(b)
        d, params = evaluate(b, S=S)
        self.assertTrue(d["recovers"], {sd: (v["d_nc"], v["d_pdms_x100"]) for sd, v in d["per_seed"].items()})
        pdms, pk_p = P.pdms_from_labels(b["lab"]), P.rule_pick(b["z6"])
        groups = L.token_groups(b["tokens"])
        null = L.run_null(P, b["z6"], b["lab"], {0: b["ep"], 1: b["ep"]}, b["code"], b["fold"], groups, pdms, pk_p, S)
        self.assertEqual(null["of"], 5)
        self.assertLess(null["fires"], L.NULL_VOID_FIRES, null)
        self.assertFalse(null["void"])
        # the uncalibrated composite does NOT recover (the lever is the calibration)
        zU = L.composite(b["z6"], ep=b["ep"])
        du = L.arm_decision(P, zU, b["lab"], pdms, pk_p, S)
        self.assertFalse(du["cond_i"] and du["cond_ii"])

    def test_globally_miscalibrated_bank_makes_the_null_fire_and_the_label_void(self):
        b = planted_bank("global")
        S = stats_for(b)
        d, _ = evaluate(b, S=S)
        self.assertTrue(d["recovers"], {sd: (v["d_nc"], v["d_pdms_x100"]) for sd, v in d["per_seed"].items()})
        pdms, pk_p = P.pdms_from_labels(b["lab"]), P.rule_pick(b["z6"])
        null = L.run_null(P, b["z6"], b["lab"], {0: b["ep"], 1: b["ep"]}, b["code"], b["fold"], L.token_groups(b["tokens"]), pdms, pk_p, S)
        self.assertGreaterEqual(null["fires"], L.NULL_VOID_FIRES, null)
        self.assertTrue(null["void"])
        self.assertEqual(L.final_label(d["recovers"], null["void"]), "VOID")

    def test_a_calibration_that_ignores_the_strata_cannot_pass_the_speed_bank(self):
        m = mutated("m = cs == s", "m = cs >= 0")
        b = planted_bank("speed")
        d, _ = evaluate(b, mod=m)
        d_true, _ = evaluate(b)
        self.assertTrue(d_true["recovers"])
        self.assertFalse(d["recovers"], "a stratum-blind fit recovered the speed-conditional bank")


if __name__ == "__main__":
    unittest.main(verbosity=2)
