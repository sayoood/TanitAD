"""Synthetic CPU tests of pg_lib: the branch table and its RED mutations, C-ID-L (new L arm == landed fit_rows), C-REACH (planted label), C-POWER (weak plant), C-NULL (permuted block),
C-CONV (convergence vs an oracle), standardisation and permutation helpers.  The fits use the REGISTERED recipe (20 epochs, batch 256 sets, lr 1e-3 / 1e-2) on ~4,000 training sets so that the
step count matches the real subset."""
import os
import sys
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pg_lib as P                                                                       # noqa: E402

FL = P.FL
import torch
torch.set_num_threads(8)
N, N_LOGS = 4000, 40
_CACHE = {}


def synth():
    if "d" in _CACHE:
        return _CACHE["d"]
    rng = np.random.default_rng(7)
    feat = (rng.normal(size=(N, 64, 256)) * 0.5).astype(np.float16)
    w_true = rng.normal(size=256) * 0.15
    s = feat.astype(np.float32) @ w_true.astype(np.float32)
    h = rng.normal(size=(N, 64)) * 1.2
    risk = s + h
    unsafe = (risk + 0.8 * rng.logistic(size=(N, 64))) > 2.0
    y = (~unsafe).astype(np.float32)
    p2_w = (-0.6 * w_true).astype(np.float32)
    p2_b = 2.0
    z_st = (feat.astype(np.float32) @ p2_w + p2_b).astype(np.float32)
    g = (h + 0.2 * rng.normal(size=(N, 64)))[..., None].astype(np.float32)
    log_idx = np.repeat(np.arange(N_LOGS), N // N_LOGS)
    fold = log_idx % 5
    code = rng.integers(1, 5, size=(N, 64)).astype(np.uint8)
    lab = np.stack([y, y], axis=-1)
    d = dict(feat=feat, z_st=z_st, y=y, p2_w=p2_w, p2_b=p2_b, g=g, log_idx=log_idx, fold=fold, code=code, lab=lab, oracle=(-(s + h)).astype(np.float32))
    _CACHE["d"] = d
    return d


def analyzer(d, B=300):
    return P.Analyzer(d["lab"], d["code"], d["log_idx"], N_LOGS, B=B)


def f_auc(an, z):
    return an.auc(an.per_log_stats(z, "nc"), "Q4")["point"]


class TestBranchTable(unittest.TestCase):
    def test_green(self):
        self.assertEqual(P.branch_table_failures(), [])

    def _ns(self, **over):
        ns = types.SimpleNamespace(**{k: getattr(P, k) for k in ("info_call", "no_jump_call", "seed_call", "row_call", "null_fires", "null_void", "power_ok", "final_call")})
        ns.__dict__.update(over)
        return ns

    def test_red_mutations(self):
        ci_lo = lambda ci: ci["lo"]                                                          # noqa: E731
        muts = {
            "info comparator flipped": dict(info_call=lambda ci: ci["lo"] < 0.02),
            "info margin 0": dict(info_call=lambda ci: ci["lo"] > 0.0),
            "info uses hi": dict(info_call=lambda ci: ci["hi"] > 0.02),
            "info >= at bound": dict(info_call=lambda ci: ci["lo"] >= 0.02),
            "nojump uses lo": dict(no_jump_call=lambda ci: ci["lo"] < 0.01),
            "nojump bound 0.02": dict(no_jump_call=lambda ci: ci["hi"] < 0.02),
            "row any instead of all (info)": dict(row_call=lambda calls: "INFO-UPSTREAM" if any(c == "INFO-UPSTREAM" for c in calls) else ("NO-JUMP" if all(c == "NO-JUMP" for c in calls) and calls else "INCONCLUSIVE")),
            "row any instead of all (nojump)": dict(row_call=lambda calls: "INFO-UPSTREAM" if calls and all(c == "INFO-UPSTREAM" for c in calls) else ("NO-JUMP" if any(c == "NO-JUMP" for c in calls) else "INCONCLUSIVE")),
            "null fires any": dict(null_fires=lambda cis: any(P.info_call(c) for c in cis)),
            "null void on 1": dict(null_void=lambda f: sum(1 for x in f if x) >= 1),
            "null void on 3": dict(null_void=lambda f: sum(1 for x in f if x) >= 3),
            "power any": dict(power_ok=lambda cis: any(P.info_call(c) for c in cis)),
            "final no conversion": dict(final_call=lambda call, vn, vi, pw: call),
            "final null over inputs": dict(final_call=lambda call, vn, vi, pw: "VOID" if vn else ("VOID-INPUTS" if vi else ("UNDERPOWERED" if call == "NO-JUMP" and not pw else call))),
            "final unpowered converts info": dict(final_call=lambda call, vn, vi, pw: "VOID-INPUTS" if vi else ("VOID" if vn else ("UNDERPOWERED" if not pw else call))),
        }
        for name, over in muts.items():
            self.assertTrue(len(P.branch_table_failures(self._ns(**over))) > 0, f"mutation not RED: {name}")


class TestHelpers(unittest.TestCase):
    def test_standardise_uses_training_stats(self):
        rng = np.random.default_rng(0)
        g = rng.normal(5.0, 2.0, size=(50, 64, 3)).astype(np.float32)
        g[..., 2] = 1.0                                                       # constant column -> 0
        tr = np.arange(0, 40)
        s = P.standardise(g, tr)
        flat = s[tr].reshape(-1, 3)
        self.assertTrue(np.allclose(flat[:, :2].mean(0), 0, atol=1e-5) and np.allclose(flat[:, :2].std(0), 1, atol=1e-4))
        self.assertTrue(np.all(s[..., 2] == 0))

    def test_permute_within_set(self):
        g = np.arange(5 * 64 * 2, dtype=np.float32).reshape(5, 64, 2)
        p = P.permute_within_set(g, 3)
        for i in range(5):
            self.assertEqual(sorted(p[i, :, 0].tolist()), sorted(g[i, :, 0].tolist()))
            self.assertTrue(np.array_equal(p[i, :, 1] - p[i, :, 0], np.full(64, 1.0)))          # the K-vector moves together
        self.assertFalse(np.array_equal(p, g))
        self.assertTrue(np.array_equal(P.permute_within_set(g, 3), p))

    def test_weak_plant(self):
        lab = np.zeros((4, 64), np.float32)
        a, b = P.weak_plant(lab, 0), P.weak_plant(lab, 1)
        self.assertEqual(a.shape, (4, 64, 1))
        self.assertFalse(np.array_equal(a, b))


class TestFits(unittest.TestCase):
    def test_id_l_matches_landed_fit_rows(self):
        d = synth()
        tr = np.arange(0, 1200)
        prm = P.fit_head("L", d["feat"], d["z_st"], d["y"], None, tr, d["p2_w"], d["p2_b"], seed=0)
        w, b = FL.fit_rows(d["feat"], d["z_st"][..., None], d["y"][..., None], tr, d["p2_w"][None, :], np.array([d["p2_b"]], np.float32), 0, cols=(0,))
        self.assertTrue(np.allclose(prm["w"], w[0], atol=1e-5), float(np.abs(prm["w"] - w[0]).max()))
        self.assertTrue(np.allclose(prm["b"], b, atol=1e-5))

    def test_reach_planted_label(self):
        """C-REACH on the code path: G := the label gives F-AUC >= 0.99 for L+G and G-only."""
        d = synth()
        an = analyzer(d)
        plant = d["y"][..., None].astype(np.float32)
        for arm in ("LG", "G"):
            z = P.oof_logits(arm, d["feat"], d["z_st"], d["y"], plant, d["fold"], 0, d["p2_w"], d["p2_b"])
            self.assertGreaterEqual(f_auc(an, z), P.REACH_MIN_AUC, arm)

    def test_reach_is_red_with_a_mute_g_block(self):
        """a mutation: the G block at lr 1e-3 (mute) must FAIL the same reach bar -- otherwise lr 1e-2 is untested."""
        d = synth()
        an = analyzer(d)
        orig_head = P.Head.__init__

        def mute_init(self, *a, **k):
            k["lr_g"] = 1e-3
            orig_head(self, *a, **k)
        P.Head.__init__ = mute_init
        try:
            z = P.oof_logits("LG", d["feat"], d["z_st"], d["y"], d["y"][..., None].astype(np.float32), d["fold"], 0, d["p2_w"], d["p2_b"])
        finally:
            P.Head.__init__ = orig_head
        self.assertLess(f_auc(an, z), P.REACH_MIN_AUC)

    def test_power_null_and_convergence(self):
        d = synth()
        an = analyzer(d)
        zL = P.oof_logits("L", d["feat"], d["z_st"], d["y"], None, d["fold"], 0, d["p2_w"], d["p2_b"])
        zG = P.oof_logits("LG", d["feat"], d["z_st"], d["y"], d["g"], d["fold"], 0, d["p2_w"], d["p2_b"])
        aL, aG, aO = f_auc(an, zL), f_auc(an, zG), f_auc(an, d["oracle"])
        print(f"\n  synthetic F-AUC: L {aL:.4f}  L+G(true hidden) {aG:.4f}  oracle {aO:.4f}")
        self.assertGreater(aG - aL, 0.02)                                       # information that exists is found
        self.assertGreaterEqual(aG, aO - 0.01)                                  # C-CONV: within 0.01 of the oracle
        fires = 0
        for rep in range(3):                                                    # C-NULL on pure noise: a permuted block must not fire the rule
            gp = P.permute_within_set(d["g"], P.NULL_SEED_BASE + P.NULL_SALT * rep)
            zp = P.oof_logits("LG", d["feat"], d["z_st"], d["y"], gp, d["fold"], 0, d["p2_w"], d["p2_b"])
            ci = an.diff(an.per_log_stats(zp, "nc"), an.per_log_stats(zL, "nc"), "Q4")
            fires += int(P.info_call(ci))
            print(f"  null rep {rep}: Delta {ci['point']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}]")
        print(f"  empirical false-fire (3 replicates, seed 0, synthetic): {fires}/3")
        self.assertEqual(fires, 0)
        wp = P.weak_plant(d["y"], 0)                                            # C-POWER: a weak planted signal is found
        zw = P.oof_logits("LG", d["feat"], d["z_st"], d["y"], wp, d["fold"], 0, d["p2_w"], d["p2_b"])
        ci = an.diff(an.per_log_stats(zw, "nc"), an.per_log_stats(zL, "nc"), "Q4")
        print(f"  weak plant: Delta {ci['point']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}]")
        self.assertTrue(P.info_call(ci))


if __name__ == "__main__":
    unittest.main(verbosity=1)
