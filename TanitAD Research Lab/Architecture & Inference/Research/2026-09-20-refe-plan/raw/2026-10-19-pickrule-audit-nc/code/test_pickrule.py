"""Synthetic CPU-only tests for PICKRULE-AUDIT-NC (no bank, no GPU, no real label).  Run: python -m unittest test_pickrule  (or pytest).

Every decision function has a RED mutation that must break the branch table; the rule, the veto, Q1 and the PDMS have planted cases that a mutated copy must fail; the controls have a planted case
in which they CAN fire (power, RETRACTION_LOG R51).
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pickrule_lib as P                                                                  # noqa: E402

LIB = os.path.join(HERE, "pickrule_lib.py")
LIB_SRC = open(LIB, "r", encoding="utf-8").read()
PLANNER = f"{P.REF}/planner.py"


def mutated(old: str, new: str):
    """A copy of pickrule_lib with the FIRST occurrence of `old` replaced by `new` (the occurrence must exist)."""
    assert old in LIB_SRC, f"mutation anchor not found: {old!r}"
    ns = types.ModuleType("pickrule_lib_mut")
    ns.__file__ = LIB
    exec(compile(LIB_SRC.replace(old, new, 1), LIB, "exec"), ns.__dict__)
    return ns


def logit(p):
    p = np.asarray(p, np.float64)
    return np.log(p / (1.0 - p))


def planted(kind: str, N: int = 2000, n_logs: int = 40, seed: int = 1):
    """P2's pick is proposal 0 in every set; the EP-led pick is proposal 1 (the EP column is the only thing that moves).  Proposal 1 is labelled NC = 0 in 40 % of the sets; kind sets its NC sigmoid.
    Every proposal >= 2 is hopeless (never picked); 5 of them carry NC = 0 labels so a label permutation has realistic unsafe density."""
    rng = np.random.default_rng(seed)
    z6 = np.full((N, 64, 6), -9.0, np.float32)
    lab = np.ones((N, 64, 6), np.float32)
    z6[:, 0, :] = logit([0.95, 0.95, 0.30, 0.05, 0.05, 0.9])
    n_unsafe = int(0.4 * N)
    order = rng.permutation(N)
    unsafe = np.zeros(N, bool)
    unsafe[order[:n_unsafe]] = True
    half = np.zeros(N, bool)
    half[order[:n_unsafe // 2]] = True                       # first half of the unsafe sets (for "mixed")
    if kind == "composition":
        nc_sig = np.where(unsafe, 0.45, 0.95)
    elif kind == "rowmiss":
        nc_sig = np.where(unsafe, 0.85, 0.95)
    elif kind == "mixed":
        nc_sig = np.where(unsafe, np.where(half, 0.45, 0.85), 0.95)
    elif kind == "structural":
        nc_sig = np.full(N, 0.45)
    else:
        raise ValueError(kind)
    z6[:, 1, 0] = logit(nc_sig)
    z6[:, 1, 1:] = logit([0.95, 0.0067, 0.05, 0.05, 0.9])
    ep = z6[:, :, 2].copy()
    ep[:, 1] = float(logit(0.99))
    lab[unsafe, 1, 0] = 0.0
    for j in range(2, 7):
        lab[:, j, 0] = 0.0
    log_idx = rng.integers(0, n_logs, size=N)
    return {"z6": z6, "lab": lab}, ep, log_idx, n_logs, unsafe


def stats_for(log_idx, n_logs, B=2000, seed=11):
    F = P.load_fastneg()
    return P.Stats(log_idx, F.boot_draws(n_logs, B, seed), F, n_logs=n_logs)


def eps_all(ep):
    return {(s, sd): ep for s in P.SOURCES for sd in P.SEEDS}


class TestRule(unittest.TestCase):
    def test_aggregate_matches_a_hand_loop(self):
        rng = np.random.default_rng(0)
        z = rng.normal(0, 2.5, size=(7, 64, 6)).astype(np.float32)
        agg = P.rule_aggregate(z)
        for i in range(7):
            for j in range(64):
                p = [1.0 / (1.0 + np.exp(-float(v))) for v in z[i, j]]
                want = p[0] * p[1] * (p[2] * 5.0 + p[3] * 5.0 + p[4] * 2.0) / 12.0
                self.assertEqual(agg[i, j], want)

    def test_formula_text_is_the_planners_return_expression(self):
        src = open(PLANNER, "r", encoding="utf-8").read().split("\n")
        ret = src[560].strip()                                                              # planner.py:561
        self.assertTrue(ret.startswith("return p[..., 0] * p[..., 1] * (p[..., 2] * w1[0]"))
        expr = ret[len("return "):]
        self.assertIn("return " + expr, open(LIB, "r", encoding="utf-8").read())

    def test_ties_go_to_the_lowest_index(self):
        z = np.zeros((3, 64, 6), np.float32)
        z[:, :, :] = 1.0
        self.assertTrue((P.rule_pick(z) == 0).all())
        z[1, 5, :] = 3.0
        z[1, 9, :] = 3.0
        self.assertEqual(int(P.rule_pick(z)[1]), 5)

    def test_red_mutations_of_the_rule(self):
        rng = np.random.default_rng(3)
        z = rng.normal(0, 2.0, size=(400, 64, 6)).astype(np.float32)
        want = P.rule_pick(z)
        for old, new in [("V1_W = (5.0, 5.0, 2.0)  ", "V1_W = (5.0, 5.0, 4.0)  "), ("return p[..., 0] * p[..., 1] * (p[..., 2] * w1[0] + p[..., 3] * w1[1] + p[..., 4] * w1[2])", "return p[..., 0] * p[..., 1] * (p[..., 2] * w1[0] + p[..., 3] * w1[1] + p[..., 5] * w1[2])"),
                         ("return p[..., 0] * p[..., 1] * (", "return p[..., 0] * (")]:
            m = mutated(old, new)
            self.assertFalse(np.array_equal(m.rule_pick(z), want), f"mutation {old!r} was not caught")
        m = mutated("return np.argmax(a, axis=1)", "return a.shape[1] - 1 - np.argmax(a[:, ::-1], axis=1)")        # last index on ties
        tie = np.ones((2, 64, 6), np.float32)
        self.assertEqual(int(P.rule_pick(tie)[0]), 0)
        self.assertNotEqual(int(m.rule_pick(tie)[0]), 0)

    def test_float32_diagnostic_runs_and_mostly_agrees(self):
        rng = np.random.default_rng(4)
        z = rng.normal(0, 2.0, size=(300, 64, 6)).astype(np.float32)
        self.assertGreater(float((P.rule_pick(z, np.float32) == P.rule_pick(z)).mean()), 0.97)

    def test_planner_pin(self):
        r = P.check_planner_pin(PLANNER)
        self.assertEqual(r["status"], "PASS", r)
        self.assertEqual(P.check_planner_pin(f"{P.REF}/does_not_exist.py")["status"], "NOT_RUN")
        lines = open(PLANNER, "rb").read().split(b"\n")
        lines[560] = lines[560].replace(b"/ w1.sum()", b"/ w1.sum() * 1.0")
        with tempfile.TemporaryDirectory() as d:
            tp = os.path.join(d, "planner.py")
            open(tp, "wb").write(b"\n".join(lines))
            self.assertEqual(P.check_planner_pin(tp)["status"], "FAIL")
            lines2 = open(PLANNER, "rb").read().split(b"\n")
            k = P.PLANNER_V1W_LINE - 1
            lines2[k] = lines2[k].replace(b"(5.0, 5.0, 2.0)", b"(5.0, 5.0, 4.0)")
            open(tp, "wb").write(b"\n".join(lines2))
            self.assertEqual(P.check_planner_pin(tp)["status"], "FAIL")


class TestVeto(unittest.TestCase):
    def test_planted_veto_and_boundary(self):
        z = np.full((1, 64, 6), -9.0, np.float32)
        z[0, 0] = logit([0.95, 0.95, 0.3, 0.05, 0.05, 0.9])
        z[0, 1] = logit([0.45, 0.95, 0.99, 0.05, 0.05, 0.9])
        self.assertEqual(int(P.rule_pick(z)[0]), 1)
        self.assertEqual(int(P.veto_pick(z, 0.3)[0][0]), 1)           # 0.45 >= 0.3: eligible
        self.assertEqual(int(P.veto_pick(z, 0.5)[0][0]), 0)           # 0.45 < 0.5: excluded
        z[0, 1, 0] = 0.0                                               # sigmoid exactly 0.5
        self.assertEqual(int(P.veto_pick(z, 0.5)[0][0]), 1)           # eligible at tau = 0.5 (>=)
        m = mutated("elig = sigmoid64(z6[..., NC]) >= tau", "elig = sigmoid64(z6[..., NC]) > tau")
        self.assertEqual(int(m.veto_pick(z, 0.5)[0][0]), 0)           # the mutant excludes the boundary: caught

    def test_identity_and_fallback(self):
        rng = np.random.default_rng(5)
        z = rng.normal(0, 2.0, size=(500, 64, 6)).astype(np.float32)
        base = P.rule_pick(z)
        self.assertGreater(int((base != 0).sum()), 10)
        p0, f0 = P.veto_pick(z, 0.0)
        p1, f1 = P.veto_pick(z, 1.01)
        self.assertTrue(np.array_equal(p0, base) and not f0.any())
        self.assertTrue(np.array_equal(p1, base) and f1.all())
        p5, f5 = P.veto_pick(z, 0.5)
        self.assertTrue(0 < int((p5 != base).sum()) < len(z))
        sig = P.sigmoid64(z[..., 0])
        self.assertTrue((sig[np.arange(len(z)), p5][~f5] >= 0.5).all())
        m = mutated("pick = np.where(has, np.argmax(masked, axis=1), np.argmax(agg, axis=1))", "pick = np.argmax(masked, axis=1)")
        pm, _ = m.veto_pick(z, 1.01)
        self.assertFalse(np.array_equal(pm, base))                    # no fallback -> index 0 on every all-excluded set: caught


class TestPdms(unittest.TestCase):
    def test_analytic_and_epr1_identity(self):
        lb = np.ones((2, 64, 6), np.float32)
        self.assertTrue((P.pdms_from_labels(lb) == 1.0).all())
        lb[0, 3, 0] = 0.5
        self.assertAlmostEqual(float(P.pdms_from_labels(lb)[0, 3]), 0.5, places=6)
        lb[0, 4, 1] = 0.0
        self.assertEqual(float(P.pdms_from_labels(lb)[0, 4]), 0.0)
        lb[1, 7, 2:5] = [0.0, 1.0, 0.0]                                  # EP 0, TTC 1, C 0 -> 5/12
        self.assertAlmostEqual(float(P.pdms_from_labels(lb)[1, 7]), 5.0 / 12.0, places=6)
        E = P.load_epr1()
        rng = np.random.default_rng(6)
        r = rng.random((50, 64, 6)).astype(np.float32)
        r[..., :2] = rng.choice([0.0, 0.5, 1.0], size=(50, 64, 2)).astype(np.float32)
        self.assertTrue(np.array_equal(P.pdms_from_labels(r), E.pdms_np(r)))
        m = mutated("V1_W = (5.0, 5.0, 2.0)  ", "V1_W = (5.0, 5.0, 4.0)  ")
        self.assertFalse(np.array_equal(m.pdms_from_labels(r), E.pdms_np(r)))


class TestDecisionFunctions(unittest.TestCase):
    def test_branch_table_passes_unmutated(self):
        self.assertEqual(P.branch_table_failures(P), [])

    def test_every_decision_function_has_a_red_mutation(self):
        muts = [
            ('if ci["lo"] > 0.5:\n        return "COMPOSITION"', 'if ci["lo"] >= 0.5:\n        return "COMPOSITION"'),
            ('if ci["hi"] < 0.5:\n        return "ROW-MISS"', 'if ci["hi"] <= 0.5:\n        return "ROW-MISS"'),
            ('if ci["lo"] > 0.5:\n        return "COMPOSITION"', 'if ci["hi"] > 0.5:\n        return "COMPOSITION"'),
            ('if ci["hi"] < 0.5:\n        return "ROW-MISS"', 'if ci["lo"] < 0.5:\n        return "ROW-MISS"'),
            ('if all(c == "COMPOSITION" for c in classes):', 'if any(c == "COMPOSITION" for c in classes):'),
            ('if all(c == "ROW-MISS" for c in classes):', 'if any(c == "ROW-MISS" for c in classes):'),
            ("return all((r <= p2_hi) and (lo >= 0.0) for r, lo in zip(rate_points, dpdms_los))", "return any((r <= p2_hi) and (lo >= 0.0) for r, lo in zip(rate_points, dpdms_los))"),
            ("return all((r <= p2_hi) and (lo >= 0.0) for r, lo in zip(rate_points, dpdms_los))", "return all((r < p2_hi) and (lo >= 0.0) for r, lo in zip(rate_points, dpdms_los))"),
            ("return all((r <= p2_hi) and (lo >= 0.0) for r, lo in zip(rate_points, dpdms_los))", "return all((r <= p2_hi) and (lo > 0.0) for r, lo in zip(rate_points, dpdms_los))"),
            ("return all((r <= p2_hi) and (lo >= 0.0) for r, lo in zip(rate_points, dpdms_los))", "return all((r <= p2_hi) or (lo >= 0.0) for r, lo in zip(rate_points, dpdms_los))"),
            ("return any(bool(v) for v in flags_by_tau.values())", "return all(bool(v) for v in flags_by_tau.values())"),
            ('("COMPOSITION", True): "a 0-train', '("COMPOSITION", True): "a soft gating rule (x) a 0-train'),
            ('("ROW-MISS", False): "NC calibration', '("ROW-MISS", False): "report both shares and stop; NC calibration'),
            ('("MIXED", False): "report both shares and stop"', '("MIXED", False): "NC calibration on the fast band"'),
            ('return bool(ci["lo"] > 0.5)', 'return bool(ci["lo"] >= 0.5)'),
            ('return bool(ci["lo"] > 0.5)', 'return bool(ci.get("point", ci["hi"]) > 0.5)'),
            ("return any(sum(1 for f in fires if f) >= NULL_VOID_FIRES for fires in fires_by_key.values())", "return any(sum(1 for f in fires if f) > NULL_VOID_FIRES for fires in fires_by_key.values())"),
            ("return any(sum(1 for f in fires if f) >= NULL_VOID_FIRES for fires in fires_by_key.values())", "return all(sum(1 for f in fires if f) >= NULL_VOID_FIRES for fires in fires_by_key.values())"),
        ]
        for old, new in muts:
            m = mutated(old, new)
            self.assertTrue(len(m.branch_table_failures(m)) > 0, f"mutation not caught: {old!r} -> {new!r}")


class TestPlantedQuantities(unittest.TestCase):
    def run_kind(self, kind):
        arr, ep, log_idx, n_logs, unsafe = planted(kind)
        zE = arr["z6"].copy()
        zE[..., P.EP] = ep
        self.assertTrue((P.rule_pick(arr["z6"]) == 0).all(), "P2 must pick proposal 0 everywhere")
        self.assertTrue((P.rule_pick(zE) == 1).all(), "the EP-led composite must pick proposal 1 everywhere")
        S = stats_for(log_idx, n_logs)
        res = P.run_analysis(arr, eps_all(ep), S)
        null = P.run_null(arr, eps_all(ep), S)
        return res, null, S, arr, ep

    def test_composition(self):
        res, null, *_ = self.run_kind("composition")
        q = res["composites"]["OOF_s0"]["q"]
        self.assertEqual(q["class"], "COMPOSITION")
        self.assertAlmostEqual(q["q1"]["point"], 1.0, places=9)
        self.assertEqual(res["by_source"]["OOF"]["outcome"], "COMPOSITION")
        self.assertFalse(null["void"])
        self.assertEqual(sum(v["fires"] for v in null["by_key"].values()), 0)
        flags = res["by_source"]["OOF"]["veto_flags"]
        self.assertFalse(flags["0.3"]["recovers"])                    # 0.45 >= 0.3: the flagged pick survives
        for t in ("0.5", "0.7", "0.9"):
            self.assertTrue(flags[t]["recovers"], t)
        self.assertTrue(res["by_source"]["OOF"]["veto_recovers"])
        self.assertTrue(res["by_source"]["OOF"]["next_arm"].startswith("a 0-train pick-rule registration"))
        # Q2 reference: P2 never picks an NC-unsafe proposal here
        self.assertEqual(q["n_B"], 0)

    def test_rowmiss(self):
        res, null, *_ = self.run_kind("rowmiss")
        q = res["composites"]["OOF_s0"]["q"]
        self.assertEqual(q["class"], "ROW-MISS")
        self.assertAlmostEqual(q["q1"]["point"], 0.0, places=9)
        self.assertEqual(res["by_source"]["OOF"]["outcome"], "ROW-MISS")
        self.assertFalse(null["void"])
        self.assertTrue(res["by_source"]["OOF"]["next_arm"].startswith("NC calibration"))
        flags = res["by_source"]["OOF"]["veto_flags"]
        self.assertFalse(flags["0.7"]["recovers"])                    # 0.85 >= 0.7
        self.assertTrue(flags["0.9"]["recovers"])                     # a veto above the row's sigmoid CAN recover even under ROW-MISS (amendment A1.1)

    def test_mixed(self):
        res, *_ = self.run_kind("mixed")
        q = res["composites"]["OOF_s0"]["q"]
        self.assertEqual(q["class"], "MIXED")
        self.assertAlmostEqual(q["q1"]["point"], 0.5, places=9)
        self.assertEqual(res["by_source"]["OOF"]["outcome"], "MIXED")

    def test_structural_artifact_fires_the_null(self):
        res, null, *_ = self.run_kind("structural")
        self.assertEqual(res["composites"]["OOF_s0"]["q"]["class"], "COMPOSITION")
        self.assertTrue(null["void"])
        self.assertTrue(all(v["fires"] == P.NULL_REPS for v in null["by_key"].values()))        # the control CAN fire (power)

    def test_strict_threshold_in_q1(self):
        arr, ep, log_idx, n_logs, unsafe = planted("composition", N=300)
        arr["z6"][:, 1, 0] = np.where(unsafe, 0.0, arr["z6"][:, 1, 0])        # sigmoid exactly 0.5 on the unsafe-labelled EP-led picks
        zE = arr["z6"].copy()
        zE[..., P.EP] = ep
        pk_e, pk_p = P.rule_pick(zE), P.rule_pick(arr["z6"])
        S = stats_for(log_idx, n_logs)
        q = P.q1_q2(arr["lab"][..., 0], P.sigmoid64(arr["z6"][..., 0]), pk_e, pk_p, S)
        self.assertEqual(q["n_over"], 0)
        m = mutated("over = A & (se < thr)", "over = A & (se <= thr)")
        qm = m.q1_q2(arr["lab"][..., 0], P.sigmoid64(arr["z6"][..., 0]), pk_e, pk_p, m.Stats(log_idx, S.draws, S.F, n_logs=n_logs))
        self.assertEqual(qm["n_over"], qm["n_A"])                              # the mutant counts the boundary as flagged: caught

    def test_unsafe_definition_is_label_zero(self):
        arr, ep, log_idx, n_logs, unsafe = planted("composition", N=300)
        arr["lab"][unsafe, 1, 0] = 0.5                                         # NC = 0.5 is NOT NC-unsafe
        zE = arr["z6"].copy()
        zE[..., P.EP] = ep
        S = stats_for(log_idx, n_logs)
        q = P.q1_q2(arr["lab"][..., 0], P.sigmoid64(arr["z6"][..., 0]), P.rule_pick(zE), P.rule_pick(arr["z6"]), S)
        self.assertEqual(q["n_A"], 0)
        ql = P.q1_q2(arr["lab"][..., 0], P.sigmoid64(arr["z6"][..., 0]), P.rule_pick(zE), P.rule_pick(arr["z6"]), S, unsafe_lt1=True)
        self.assertGreater(ql["n_A"], 0)                                       # the descriptive `< 1` sensitivity does count it
        m = mutated("unsafe = (lab_nc < 1.0) if unsafe_lt1 else (lab_nc == 0.0)", "unsafe = (lab_nc < 1.0)")
        qm = m.q1_q2(arr["lab"][..., 0], P.sigmoid64(arr["z6"][..., 0]), P.rule_pick(zE), P.rule_pick(arr["z6"]), m.Stats(log_idx, S.draws, S.F, n_logs=n_logs))
        self.assertGreater(qm["n_A"], 0)                                       # the mutant would have counted NC = 0.5: caught


class TestEstimator(unittest.TestCase):
    def test_bootstrap_identity_and_point(self):
        F, E = P.load_fastneg(), P.load_epr1()
        self.assertTrue(np.array_equal(F.boot_draws(117, 10000, 20260927), E.boot_draws(117, 10000, 20260927)))
        rng = np.random.default_rng(8)
        log_idx = rng.integers(0, 30, size=500)
        flag = rng.random(500) < 0.2
        S = P.Stats(log_idx, F.boot_draws(30, 2000, 5), F, n_logs=30)
        ci = S.rate(flag)
        self.assertAlmostEqual(ci["point"], float(flag.mean()), places=12)
        self.assertLess(ci["lo"], ci["point"])
        self.assertGreater(ci["hi"], ci["point"])
        d = S.rate_diff(flag, flag)
        self.assertEqual(d["point"], 0.0)
        self.assertEqual(d["lo"], 0.0)
        self.assertEqual(d["hi"], 0.0)


class TestGate(unittest.TestCase):
    def synth(self):
        rng = np.random.default_rng(9)
        N = 600
        z6 = rng.normal(0, 2.0, size=(N, 64, 6)).astype(np.float32)
        lab = np.concatenate([rng.choice([0.0, 0.5, 1.0], size=(N, 64, 2), p=[0.1, 0.05, 0.85]), rng.random((N, 64, 4))], axis=2).astype(np.float32)
        top8 = np.argsort(-P.rule_aggregate(z6), axis=1, kind="stable")[:, :8].astype(np.int64)
        ep = (z6[..., 2] + rng.normal(0, 1.5, size=(N, 64))).astype(np.float32)
        return {"z6": z6, "lab": lab, "top8": top8}, ep

    def ref(self, E, arr, ep):
        zE = arr["z6"].copy()
        zE[..., 2] = ep
        pk_e, pk_p = E.picks_np(zE), E.picks_np(arr["z6"])
        safe, pd = E.safe_np(arr["lab"]), E.pdms_np(arr["lab"])
        rows, N = np.arange(len(pk_e)), len(pk_e)
        return {"new_pdms_x100": 100.0 * pd[rows, pk_e].astype(np.float64).sum() / N, "p2_pdms_x100": 100.0 * pd[rows, pk_p].astype(np.float64).sum() / N,
                "unsafe_rate_new": float((~safe[rows, pk_e]).sum() / N), "unsafe_rate_p2": float((~safe[rows, pk_p]).sum() / N), "oof_picks_changed": int((pk_e != pk_p).sum())}

    def test_gate_passes_and_each_red_arm_fails(self):
        E = P.load_epr1()
        arr, ep = self.synth()
        ep_cols = {(s, sd): ep for s in P.SOURCES for sd in P.SEEDS}
        ref = {sd: self.ref(E, arr, ep) for sd in P.SEEDS}
        g = P.gate_checks(arr, ep_cols, E, ref, PLANNER)
        self.assertTrue(g["ok"], {k: v for k, v in g["checks"].items() if not v["pass"]})
        # C-ID1: a wrong banked pick
        bad = dict(arr)
        bad["top8"] = arr["top8"].copy()
        bad["top8"][:5, 0] = (bad["top8"][:5, 0] + 1) % 64
        self.assertFalse(P.gate_checks(bad, ep_cols, E, ref, PLANNER)["checks"]["C-ID1"]["pass"])
        # C-ID1: a tampered planner copy
        lines = open(PLANNER, "rb").read().split(b"\n")
        lines[560] = lines[560].replace(b"p[..., 2] * w1[0]", b"p[..., 2] * w1[1]")
        with tempfile.TemporaryDirectory() as d:
            tp = os.path.join(d, "planner.py")
            open(tp, "wb").write(b"\n".join(lines))
            self.assertFalse(P.gate_checks(arr, ep_cols, E, ref, tp)["checks"]["C-ID1"]["pass"])
        # C-ID2: a banked number off by 1e-6, and a wrong changed-pick count
        ref2 = {sd: dict(ref[sd]) for sd in P.SEEDS}
        ref2[0]["unsafe_rate_new"] += 1e-6
        self.assertFalse(P.gate_checks(arr, ep_cols, E, ref2, PLANNER)["checks"]["C-ID2[seed0]"]["pass"])
        ref3 = {sd: dict(ref[sd]) for sd in P.SEEDS}
        ref3[1]["oof_picks_changed"] += 1
        self.assertFalse(P.gate_checks(arr, ep_cols, E, ref3, PLANNER)["checks"]["C-ID2[seed1]"]["pass"])
        # C-COL: an EP column that also leaks into another column cannot be built by the caller, but an EP column equal to P2's must fail ("column 2 differs on >= 1 proposal")
        same = {(s, sd): arr["z6"][..., 2].copy() for s in P.SOURCES for sd in P.SEEDS}
        self.assertFalse(P.gate_checks(arr, same, E, ref, PLANNER)["checks"]["C-COL[OOF_s0]"]["pass"])
        # C-ID3b (Amendment 2): the veto's reach under a planted label-free shift gates; a no-op veto and a switch each fail it; the registered clause is informational
        self.assertTrue(g["checks"]["C-ID3b"]["pass"])
        self.assertFalse(g["checks"]["C-ID3-registered-reach-clause"]["gating"])
        noop = mutated("pick = np.where(has, np.argmax(masked, axis=1), np.argmax(agg, axis=1))", "pick = np.argmax(agg, axis=1)")
        gn = noop.gate_checks(arr, ep_cols, E, ref, PLANNER)
        self.assertFalse(gn["checks"]["C-ID3b"]["pass"])
        self.assertFalse(gn["ok"])
        switch = mutated("pick = np.where(has, np.argmax(masked, axis=1), np.argmax(agg, axis=1))", "pick = (np.argmax(agg, axis=1) + 1) % 64")
        gs = switch.gate_checks(arr, ep_cols, E, ref, PLANNER)
        self.assertFalse(gs["checks"]["C-ID3b"]["pass"])
        self.assertFalse(gs["checks"]["C-ID3"]["pass"])
        # an informational row that fails must NOT fail the gate: P2 rows whose picks all sit above NC sigmoid 0.5 (the real situation)
        arr2 = dict(arr)
        arr2["z6"] = arr["z6"].copy()
        arr2["z6"][..., 0] = np.abs(arr2["z6"][..., 0]) + 1.0                                  # every NC sigmoid > 0.73: the veto at 0.5 cannot move a P2 pick
        arr2["top8"] = np.argsort(-P.rule_aggregate(arr2["z6"]), axis=1, kind="stable")[:, :8].astype(np.int64)
        zE2 = arr2["z6"].copy()
        zE2[..., 2] = ep
        ref_b = {sd: self.ref(E, arr2, ep) for sd in P.SEEDS}
        gb = P.gate_checks(arr2, ep_cols, E, ref_b, PLANNER)
        self.assertFalse(gb["checks"]["C-ID3-registered-reach-clause"]["pass"])
        self.assertTrue(gb["checks"]["C-ID3b"]["pass"])
        self.assertTrue(gb["ok"], {k: v for k, v in gb["checks"].items() if not v["pass"] and v["gating"]})
        # a missing planner copy is NOT RUN, not PASS
        g2 = P.gate_checks(arr, ep_cols, E, ref, f"{P.REF}/nope.py")
        self.assertEqual(g2["checks"]["C-ID1"]["detail"]["planner_pin"]["status"], "NOT_RUN")

    def test_apply_row_changes_only_the_ep_column(self):
        import torch
        E = P.load_epr1()
        rng = np.random.default_rng(10)
        feat = torch.from_numpy(rng.normal(0, 1, size=(4, 64, 256)).astype(np.float16))
        z6 = torch.from_numpy(rng.normal(0, 1, size=(4, 64, 6)).astype(np.float32))
        w0, b0 = rng.normal(0, 0.1, 256).astype(np.float32), np.array([0.3], np.float32)
        same = E.apply_row(z6, feat, w0, b0, w0, b0)
        self.assertTrue(torch.equal(same, z6))                                 # the P2 row reproduces the stored logits bit for bit
        w1 = w0 + 0.05
        new = E.apply_row(z6, feat, w1, b0, w0, b0)
        self.assertTrue(torch.equal(new[..., [0, 1, 3, 4, 5]], z6[..., [0, 1, 3, 4, 5]]))
        self.assertFalse(torch.equal(new[..., 2], z6[..., 2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
