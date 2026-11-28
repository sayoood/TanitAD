"""unittest for the per-window path of fam_ci.py (fam_pertoken).  CUDA_VISIBLE_DEVICES=-1 python -m unittest test_fam_ci_pertoken -v  (CPU, synthetic)."""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fam_ci as F  # noqa: E402

NB = 2000
NL, PL = 6, 20
N = NL * PL
TOK = [f"t{i:04d}" for i in range(N)]
LOGS = [f"log{i // PL}" for i in range(N)]
CLS = ["a", "b", "c"]


def pw(seed=0, steps=None):
    """a synthetic _per_window block: one mean, one rms, one ratio (micro mean over valid steps), one median, tactical decisions + goal."""
    r = np.random.default_rng(seed)
    steps = r.integers(0, 9, N).astype(float) if steps is None else steps
    err_sum = steps * r.gamma(2.0, 1.0, N)                       # per-token SUM of step errors over `steps` valid steps
    gt = np.random.default_rng(999).integers(0, 3, N)                 # the HUMAN's manoeuvre ids: identical for every arm
    pred = np.where(r.random(N) < 0.7, gt, r.integers(0, 3, N))
    rows = np.flatnonzero(r.random(N) < 0.9)
    return {"schema": "taniteval.per_window/1", "n": N, "tokens": list(TOK), "eid": list(LOGS), "status": "OK",
            "longitudinal": {"speed_mae_mps": {"kind": "mean", "value": r.gamma(2, 1, N), "dp": 4},
                             "speed_rmse_mps": {"kind": "rms", "value": r.gamma(2, 1, N), "dp": 4},
                             "ego_progress.progress_ratio_median": {"kind": "median", "value": r.gamma(9, 0.12, rows.size), "rows": rows, "dp": 4},
                             "ego_progress.gt_progress_mean_m": {"kind": "ratio", "num": r.random(N), "den": np.ones(N), "scale": 1.0, "dp": 4, "human_only": True}},
            "lateral": {"heading_mae_deg": {"kind": "ratio", "num": err_sum, "den": steps, "scale": 1.0, "dp": 4}},
            "tactical": {"status": "OK",
                         "lateral_decision": {"classes": CLS, "gt": gt, "pred": pred},
                         "longitudinal_decision": {"classes": CLS, "gt": gt, "pred": pred},
                         "maneuver_5way_collapsed": {"classes": CLS, "gt": gt, "pred": pred},
                         "goal_setting": {"goal_point_error_m": {"kind": "mean", "value": r.gamma(2, 1, N), "dp": 4}}},
            "unavailable": {}}


def tolist(o):
    if isinstance(o, dict):
        return {k: tolist(v) for k, v in o.items()}
    return o.tolist() if hasattr(o, "tolist") else o


def make(pwb, label="X"):
    blk = {"longitudinal": {"ci": {"components": {}}}, "lateral": {"ci": {"components": {}}}, "tactical": {}, "_per_window": tolist(pwb)}
    return {"_label": label, "n": N, "n_logs": NL, "tokens": list(TOK), "logs": list(LOGS), "lead_states": {"NO_LEAD": N},
            "frames": {fr: {"role": "PRIMARY", "block": copy.deepcopy(blk)} for fr in F.FRAMES}}


def ent(r, fam, k, fr="executed_tracked"):
    return r["frames"][fr]["families"][fam][k]


class T(unittest.TestCase):
    def cmp(self, a, b):
        return F.compare(a, b, None, n_boot=NB)

    def test_identity_zero_on_every_family(self):
        a = make(pw(1))
        r = self.cmp(a, copy.deepcopy(a))
        seen = {"longitudinal": 0, "lateral": 0, "tactical": 0}
        for fr in F.FRAMES:
            for fam in seen:
                for k, e in r["frames"][fr]["families"][fam].items():
                    if e["status"] == "OK":
                        d = e["diff_B_minus_A"]
                        self.assertEqual((d["point"], d["lo"], d["hi"]), (0.0, 0.0, 0.0), (fam, k))
                        self.assertFalse(e["separated_from_0"])
                        seen[fam] += 1
        self.assertTrue(all(v >= 3 for v in seen.values()), seen)

    def test_ratio_of_sums_known_value(self):
        """two tokens-per-log hand fixture: A steps/err chosen so ratio-of-sums != mean-of-token-means; the point is checked by hand."""
        steps = np.tile([1.0, 9.0], N // 2)                            # every log: 10 tokens of 1 step, 10 of 9 steps
        pa, pb = pw(2, steps), pw(3, steps)
        # token errors: 1-step tokens err 10 per step, 9-step tokens err 2 per step (A); B = A + 1 per step everywhere
        per = np.tile([10.0, 2.0], N // 2)
        pa["lateral"]["heading_mae_deg"]["num"] = steps * per
        pb["lateral"]["heading_mae_deg"]["num"] = steps * (per + 1.0)
        e = ent(self.cmp(make(pa), make(pb)), "lateral", "heading_mae_deg")
        want_a = (10 * (1 * 10 + 9 * 2)) / (10 * (1 + 9))             # sum(err)/sum(steps) = 2.8 ; the mean of token means would be (10+2)/2 = 6
        self.assertAlmostEqual(e["A"]["point"], want_a, places=12)
        self.assertAlmostEqual(e["A"]["point"], 2.8, places=12)
        self.assertAlmostEqual(e["B"]["point"], 3.8, places=12)
        self.assertAlmostEqual(e["diff_B_minus_A"]["point"], 1.0, places=12)
        self.assertAlmostEqual(e["diff_B_minus_A"]["lo"], 1.0, places=12)   # +1 per step everywhere -> every draw reads exactly +1
        self.assertAlmostEqual(e["diff_B_minus_A"]["hi"], 1.0, places=12)
        self.assertEqual(e["den_total_A"], float(steps.sum()))

    def test_mutation_mean_of_means_goes_red(self):
        """swap ratio-of-sums for a mean of per-token means: the known-value check above must FAIL."""
        steps = np.tile([1.0, 9.0], N // 2)
        per = np.tile([10.0, 2.0], N // 2)
        pa, pb = pw(2, steps), pw(3, steps)
        pa["lateral"]["heading_mae_deg"]["num"] = steps * per
        pb["lateral"]["heading_mae_deg"]["num"] = steps * (per + 1.0)
        real = F._sufficient

        def mutated(ra, rb):
            SA, SB, fn, dA, dB, kind = real(ra, rb)
            if kind != "ratio":
                return SA, SB, fn, dA, dB, kind

            def mom(S):                                                   # per-token mean (num/den), then averaged over the drawn tokens
                ok = S[:, 1] > 0
                return np.stack([np.where(ok, S[:, 0] / np.where(ok, S[:, 1], 1), 0.0), ok.astype(float)], 1)
            return mom(SA), mom(SB), fn, dA, dB, kind

        F._sufficient = mutated
        try:
            e = ent(self.cmp(make(pa), make(pb)), "lateral", "heading_mae_deg")
        finally:
            F._sufficient = real
        self.assertGreater(abs(e["A"]["point"] - 2.8), 1.0)               # the mutant reads 6.0, not 2.8: the known-value assertion would be RED
        self.assertAlmostEqual(e["A"]["point"], 6.0, places=12)

    def test_constant_shift_all_kinds(self):
        a, b = pw(4), pw(4)
        for k in ("speed_mae_mps", "speed_rmse_mps"):
            b["longitudinal"][k]["value"] = b["longitudinal"][k]["value"] + 0.5
        b["lateral"]["heading_mae_deg"]["num"] = b["lateral"]["heading_mae_deg"]["num"] + 0.25 * b["lateral"]["heading_mae_deg"]["den"]
        b["longitudinal"]["ego_progress.progress_ratio_median"]["value"] = b["longitudinal"]["ego_progress.progress_ratio_median"]["value"] + 0.1
        r = self.cmp(make(a), make(b))
        self.assertAlmostEqual(ent(r, "longitudinal", "speed_mae_mps")["diff_B_minus_A"]["lo"], 0.5, places=12)
        self.assertAlmostEqual(ent(r, "lateral", "heading_mae_deg")["diff_B_minus_A"]["hi"], 0.25, places=12)
        m = ent(r, "longitudinal", "ego_progress.progress_ratio_median")["diff_B_minus_A"]
        self.assertAlmostEqual(m["lo"], 0.1, places=12)
        self.assertAlmostEqual(m["hi"], 0.1, places=12)
        rm = ent(r, "longitudinal", "speed_rmse_mps")                    # rms shifts by a non-constant amount: only check it moved and is finite
        self.assertGreater(rm["diff_B_minus_A"]["point"], 0)

    def test_only_changed_metric_moves(self):
        a, b = pw(5), pw(5)
        b["longitudinal"]["speed_mae_mps"]["value"] = b["longitudinal"]["speed_mae_mps"]["value"] + 1.0
        r = self.cmp(make(a), make(b))
        for fam, k in (("lateral", "heading_mae_deg"), ("tactical", "goal_setting.goal_point_error_m"), ("tactical", "lateral_decision.accuracy")):
            self.assertEqual(ent(r, fam, k)["diff_B_minus_A"]["point"], 0.0)
        self.assertNotEqual(ent(r, "longitudinal", "speed_mae_mps")["diff_B_minus_A"]["point"], 0.0)

    def test_categorical_accuracy_and_per_class(self):
        a, b = pw(6), pw(6)
        gt = np.asarray(a["tactical"]["lateral_decision"]["gt"])
        pb = np.asarray(b["tactical"]["lateral_decision"]["pred"]).copy()
        pb[:] = gt                                                        # B is perfect
        b["tactical"]["lateral_decision"]["pred"] = pb
        r = self.cmp(make(a), make(b))
        acc = ent(r, "tactical", "lateral_decision.accuracy")
        pa = np.asarray(a["tactical"]["lateral_decision"]["pred"])
        self.assertAlmostEqual(acc["A"]["point"], float((pa == gt).mean()), places=12)
        self.assertEqual(acc["B"]["point"], 1.0)
        self.assertAlmostEqual(acc["diff_B_minus_A"]["point"], 1.0 - float((pa == gt).mean()), places=12)
        self.assertEqual(ent(r, "tactical", "lateral_decision.kappa")["B"]["point"], 1.0)
        rec = ent(r, "tactical", "lateral_decision.per_class.a.recall")
        self.assertEqual(rec["B"]["point"], 1.0)
        self.assertEqual(ent(r, "tactical", "lateral_decision.per_class.a.precision")["B"]["point"], 1.0)
        self.assertEqual(rec["n_tokens_defined_A"], int((gt == 0).sum()))
        # untouched decisions did not move
        self.assertEqual(ent(r, "tactical", "longitudinal_decision.accuracy")["diff_B_minus_A"]["point"], 0.0)

    def test_human_only_not_compared(self):
        e = ent(self.cmp(make(pw(7)), make(pw(8))), "longitudinal", "ego_progress.gt_progress_mean_m")
        self.assertEqual(e["status"], "HUMAN-ONLY")

    def test_token_id_pairing_realigns_and_refuses(self):
        a, b = make(pw(9)), make(pw(10))
        base = self.cmp(a, b)
        # B stored in another token order, arrays permuted consistently -> identical result
        perm = np.random.default_rng(0).permutation(N)
        bp = copy.deepcopy(b)
        bp["tokens"] = [b["tokens"][i] for i in perm]
        bp["logs"] = [b["logs"][i] for i in perm]
        for fr in F.FRAMES:
            w = bp["frames"][fr]["block"]["_per_window"]
            w["tokens"] = list(bp["tokens"])
            w["eid"] = list(bp["logs"])
            for fam in ("longitudinal", "lateral"):
                for k, c in w[fam].items():
                    if c["kind"] == "median":
                        full = np.full(N, np.nan)
                        full[np.asarray(c["rows"], int)] = c["value"]
                        full = full[perm]
                        c["rows"] = np.flatnonzero(np.isfinite(full)).tolist()
                        c["value"] = full[np.isfinite(full)].tolist()
                    else:
                        for kk in ("value", "num", "den"):
                            if kk in c:
                                c[kk] = np.asarray(c[kk])[perm].tolist()
            for sub in ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed"):
                for kk in ("gt", "pred"):
                    w["tactical"][sub][kk] = np.asarray(w["tactical"][sub][kk])[perm].tolist()
            w["tactical"]["goal_setting"]["goal_point_error_m"]["value"] = np.asarray(w["tactical"]["goal_setting"]["goal_point_error_m"]["value"])[perm].tolist()
        r2 = self.cmp(a, bp)
        for fam, k in (("lateral", "heading_mae_deg"), ("longitudinal", "speed_mae_mps"), ("tactical", "lateral_decision.accuracy"),
                       ("longitudinal", "ego_progress.progress_ratio_median"), ("tactical", "goal_setting.goal_point_error_m")):
            self.assertEqual(ent(base, fam, k)["diff_B_minus_A"], ent(r2, fam, k)["diff_B_minus_A"], k)
        # refusals
        c = copy.deepcopy(b)
        c["tokens"][3] = "zzzz"
        with self.assertRaises(F.TokenMismatch):
            self.cmp(a, c)
        d = copy.deepcopy(b)
        d["logs"][0] = "other"
        with self.assertRaises(F.TokenMismatch):
            self.cmp(a, d)
        e = copy.deepcopy(b)
        e["frames"]["executed_tracked"]["block"]["_per_window"]["tokens"][0] = "zzzz"
        with self.assertRaises(F.TokenMismatch):
            self.cmp(a, e)

    def test_mean_kind_matches_cluster_boot(self):
        """the new path for a `mean` record must equal the legacy cluster_boot on the per-token differences (same draws; to 1e-12: sum(B)-sum(A) vs sum(B-A) differ at 1e-16)."""
        a, b = pw(11), pw(12)
        e = ent(self.cmp(make(a), make(b)), "longitudinal", "speed_mae_mps")["diff_B_minus_A"]
        d = np.asarray(b["longitudinal"]["speed_mae_mps"]["value"]) - np.asarray(a["longitudinal"]["speed_mae_mps"]["value"])
        p, lo, hi, _ = F.cluster_boot(d, LOGS, NB, F.SEED)
        self.assertAlmostEqual(e["lo"], lo, places=12)
        self.assertAlmostEqual(e["hi"], hi, places=12)
        self.assertAlmostEqual(e["point"], p, places=12)

    def test_default_json_without_per_window_unchanged(self):
        a = make(pw(13))
        for fr in F.FRAMES:
            a["frames"][fr]["block"].pop("_per_window")
        r = F.compare(a, copy.deepcopy(a), None, n_boot=NB)
        for fr in F.FRAMES:
            self.assertNotIn("speed_mae_mps", r["frames"][fr]["families"]["longitudinal"])


if __name__ == "__main__":
    unittest.main()
