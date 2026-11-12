"""unittest for fam_ci.py.  CUDA_VISIBLE_DEVICES=-1 python -m unittest test_fam_ci -v   (CPU only; synthetic fixtures)."""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fam_ci as F  # noqa: E402

NB = 2000  # fewer draws in tests; the estimator is identical


def make(values, n_logs=6, per_log=10, label="X"):
    """synthetic families JSON with per-token headway_min_m = values (len n_logs*per_log)."""
    n = n_logs * per_log
    toks = [f"t{i:04d}" for i in range(n)]
    logs = [f"log{i // per_log}" for i in range(n)]
    v = list(map(float, values))
    pw = {k: list(v) for k in F.PER_TOKEN_LONG}
    blk = {"longitudinal": {"distance_keeping": {"_per_window": pw},
                            "ci": {"components": {"speed_mae_mps": {"mean": 1.0, "lo": 0.9, "hi": 1.1}}}},
           "lateral": {"ci": {"components": {"heading_mae_deg": {"mean": 2.0, "lo": 1.0, "hi": 3.0}}}},
           "tactical": {"lateral_decision": {"accuracy": 0.9}}}
    j = {"_label": label, "n": n, "n_logs": n_logs, "lead_states": {"LEAD": n},
         "frames": {fr: {"role": "PRIMARY", "block": copy.deepcopy(blk)} for fr in F.FRAMES}}
    return j, (toks, logs)


def rng_vals(n, seed=1):
    return np.random.default_rng(seed).normal(10, 2, n)


class T(unittest.TestCase):
    def run_cmp(self, a, b, tok):
        return F.compare(a, b, tok, n_boot=NB)

    def test_self_is_exact_zero(self):
        a, tok = make(rng_vals(60))
        r = self.run_cmp(a, copy.deepcopy(a), tok)
        for fr in F.FRAMES:
            for k, e in r["frames"][fr]["families"]["longitudinal"].items():
                if e["status"] == "OK":
                    d = e["diff_B_minus_A"]
                    self.assertEqual((d["point"], d["lo"], d["hi"]), (0.0, 0.0, 0.0), k)
                    self.assertFalse(e["separated_from_0"])

    def test_constant_shift_reads_c(self):
        v = rng_vals(60)
        a, tok = make(v)
        b, _ = make(v + 0.75)
        r = self.run_cmp(a, b, tok)
        e = r["frames"]["executed_tracked"]["families"]["longitudinal"]["headway_min_m"]["diff_B_minus_A"]
        for x in (e["point"], e["lo"], e["hi"]):
            self.assertAlmostEqual(x, 0.75, places=12)

    def test_only_shifted_metric_moves(self):
        v = rng_vals(60)
        a, tok = make(v)
        b, _ = make(v)
        for fr in F.FRAMES:
            b["frames"][fr]["block"]["longitudinal"]["distance_keeping"]["_per_window"]["time_gap_min_s"] = list(v + 2.0)
        r = self.run_cmp(a, b, tok)["frames"]["executed_tracked"]["families"]["longitudinal"]
        self.assertAlmostEqual(r["time_gap_min_s"]["diff_B_minus_A"]["point"], 2.0, places=12)
        self.assertEqual(r["headway_min_m"]["diff_B_minus_A"]["point"], 0.0)

    def test_token_mismatch_refused(self):
        a, (tk, lg) = make(rng_vals(60))
        b, _ = make(rng_vals(60))
        b["tokens"], b["logs"] = tk[:-1] + ["tXXXX"], lg
        a["tokens"], a["logs"] = tk, lg
        with self.assertRaises(F.TokenMismatch) as c:
            F.compare(a, b, n_boot=NB)
        self.assertIn("60", str(c.exception))
        a2, tok = make(rng_vals(60))
        b2, _ = make(rng_vals(54), n_logs=6, per_log=9)
        with self.assertRaises(F.TokenMismatch):
            F.compare(a2, b2, tok, n_boot=NB)  # n differs from tokens supplied
        b3, _ = make(rng_vals(60))
        b3["lead_states"] = {"LEAD": 59, "NO_LEAD": 1}
        with self.assertRaises(F.TokenMismatch):
            F.compare(a2, b3, tok, n_boot=NB)

    def test_missing_tokens_refused(self):
        a, _ = make(rng_vals(60))
        with self.assertRaises(F.TokenMismatch):
            F.compare(a, copy.deepcopy(a), None, n_boot=NB)

    def test_intersection_and_nan(self):
        v = rng_vals(60)
        a, tok = make(v)
        b, _ = make(v + 1.0)
        for fr in F.FRAMES:
            pa = a["frames"][fr]["block"]["longitudinal"]["distance_keeping"]["_per_window"]["min_ttc_s"]
            pb = b["frames"][fr]["block"]["longitudinal"]["distance_keeping"]["_per_window"]["min_ttc_s"]
            for i in range(0, 60, 2):
                pa[i] = float("nan")
            for i in range(0, 60, 3):
                pb[i] = None
        e = self.run_cmp(a, b, tok)["frames"]["executed_tracked"]["families"]["longitudinal"]["min_ttc_s"]
        want = sum(1 for i in range(60) if i % 2 and i % 3)
        self.assertEqual(e["n_tokens"], want)
        self.assertEqual(e["n_tokens_defined_A"], 30)
        self.assertEqual(e["n_tokens_defined_B"], 40)
        self.assertAlmostEqual(e["diff_B_minus_A"]["point"], 1.0, places=12)

    def test_no_per_token_and_strategic(self):
        a, tok = make(rng_vals(60))
        r = self.run_cmp(a, copy.deepcopy(a), tok)["frames"]["executed_tracked"]["families"]
        self.assertEqual(r["longitudinal"]["speed_mae_mps"]["status"], "NO-PER-TOKEN-DATA")
        self.assertEqual(r["lateral"]["heading_mae_deg"]["status"], "NO-PER-TOKEN-DATA")
        self.assertEqual(r["tactical"]["lateral_decision.accuracy"]["status"], "NO-PER-TOKEN-DATA")
        self.assertEqual(r["strategic"]["status"], "PAUSED")

    def test_seed_bit_for_bit(self):
        v = rng_vals(60)
        a, tok = make(v)
        b, _ = make(v + rng_vals(60, 5) * 0.1)
        r1 = self.run_cmp(a, b, tok)
        r2 = self.run_cmp(a, b, tok)
        self.assertEqual(r1, r2)
        r3 = F.compare(a, b, tok, n_boot=NB, seed=1)
        self.assertNotEqual(r1["frames"]["executed_tracked"]["families"]["longitudinal"]["headway_min_m"]["diff_B_minus_A"],
                            r3["frames"]["executed_tracked"]["families"]["longitudinal"]["headway_min_m"]["diff_B_minus_A"])

    def test_order_realigned(self):
        v = rng_vals(60)
        a, (tk, lg) = make(v)
        b, _ = make(v + 0.5)
        a["tokens"], a["logs"] = tk, lg
        p = np.random.default_rng(3).permutation(60)
        b["tokens"], b["logs"] = [tk[i] for i in p], [lg[i] for i in p]
        for fr in F.FRAMES:
            pw = b["frames"][fr]["block"]["longitudinal"]["distance_keeping"]["_per_window"]
            for k in pw:
                pw[k] = [pw[k][i] for i in p]
        e = F.compare(a, b, n_boot=NB)["frames"]["executed_tracked"]["families"]["longitudinal"]["headway_min_m"]["diff_B_minus_A"]
        self.assertAlmostEqual(e["point"], 0.5, places=12)

    # ---- mutation arm: the cluster resample is what makes the CI wide when within-log values are perfectly correlated
    def fixture_correlated(self):
        # 8 logs x 25 tokens; value constant within a log (perfect within-log correlation), differs between logs
        lv = np.array([-3.0, -1.5, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0])
        v = np.repeat(lv, 25)
        return v, [f"log{i // 25}" for i in range(200)]

    def test_cluster_wider_than_pertoken_known_factor(self):
        v, logs = self.fixture_correlated()
        _, lo_c, hi_c, nl = F.cluster_boot(v, logs, 5000)
        _, lo_t, hi_t, _ = F.pertoken_boot(v, 5000)
        self.assertEqual(nl, 8)
        ratio = (hi_c - lo_c) / (hi_t - lo_t)
        # SE ratio = sqrt(design effect) = sqrt(25) = 5 (up to bootstrap-width noise); a per-token resample would read ~1
        self.assertGreater(ratio, 4.0)
        self.assertLess(ratio, 6.5)

    def test_compare_uses_cluster_resample_MUTATION(self):
        """the production path must give the CLUSTER interval. Swap in the per-token resample and this test goes RED."""
        v, _ = self.fixture_correlated()
        a, tok = make(np.zeros(200), n_logs=8, per_log=25)
        b, _ = make(v, n_logs=8, per_log=25)
        e = F.compare(a, b, tok, n_boot=5000)["frames"]["executed_tracked"]["families"]["longitudinal"]["headway_min_m"]["diff_B_minus_A"]
        _, lo_t, hi_t, _ = F.pertoken_boot(v, 5000)
        self.assertGreater(e["hi"] - e["lo"], 4.0 * (hi_t - lo_t))

    def test_mutation_detected(self):
        """run the mutation for real: monkeypatch cluster_boot to the per-token resample; the check above must fail."""
        orig = F.cluster_boot
        F.cluster_boot = lambda v, logs, n_boot=F.B_DRAWS, seed=F.SEED: F.pertoken_boot(v, n_boot, seed)
        try:
            with self.assertRaises(AssertionError):
                self.test_compare_uses_cluster_resample_MUTATION()
        finally:
            F.cluster_boot = orig

    def test_matches_tfn2_boot_convention(self):
        """independent re-implementation of tfn2_analyze.boot (verbatim logic) must agree bit-for-bit."""
        v = np.random.default_rng(9).normal(size=90)
        logs = [f"l{i // 9}" for i in range(90)]
        ul = sorted(set(logs))
        idx = {l: np.array([i for i, x in enumerate(logs) if x == l]) for l in ul}
        s = np.array([v[idx[l]].sum() for l in ul])
        n = np.array([len(idx[l]) for l in ul])
        d = np.random.default_rng(F.SEED).integers(0, len(ul), (F.B_DRAWS, len(ul)))
        m = s[d].sum(1) / n[d].sum(1)
        want = (float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))
        got = F.cluster_boot(v, logs)[:3]
        self.assertEqual(want, got)


if __name__ == "__main__":
    unittest.main()
