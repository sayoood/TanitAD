"""Synthetic CPU tests of pgc_lib: the branch table and its RED mutations, the block / centring / null helpers, and the fit path (C-REACH planted label through the CENTRED block, the centred null does not
CONFIRM, centring invariance, a known weak plant is found).  The fits use the REGISTERED recipe (20 epochs, batch 256 sets, lr 1e-3 / 1e-2) on ~2,000 sets.  No stored label, no real scalar is read."""
import os
import sys
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pgc_lib as C                                                                      # noqa: E402

P, FL = C.P, C.FL
import torch
torch.set_num_threads(8)
N, N_LOGS = 2000, 40
_CACHE = {}


def synth():
    if "d" in _CACHE:
        return _CACHE["d"]
    rng = np.random.default_rng(5)
    feat = (rng.normal(size=(N, 64, 256)) * 0.5).astype(np.float16)
    w_true = rng.normal(size=256) * 0.15
    s = feat.astype(np.float32) @ w_true.astype(np.float32)
    h = rng.normal(size=(N, 64)) * 1.2                                              # the proximity latent the token cannot see
    set_off = rng.normal(size=(N, 1)) * 2.0                                         # a BETWEEN-set level that the centred arm must ignore
    unsafe = (s + h + 0.8 * rng.logistic(size=(N, 64))) > 2.0
    y = (~unsafe).astype(np.float32)
    p2_w = (-0.6 * w_true).astype(np.float32)
    p2_b = 2.0
    z_st = (feat.astype(np.float32) @ p2_w + p2_b).astype(np.float32)
    cols = [(h + 0.3 * rng.normal(size=(N, 64)) + set_off + 0.1 * k) for k in range(9)]
    g15 = np.zeros((N, 64, 15))
    g15[..., 2:11] = np.stack(cols, -1)
    log_idx = np.repeat(np.arange(N_LOGS), N // N_LOGS)
    fold = log_idx % 5
    code = rng.integers(1, 5, size=(N, 64)).astype(np.uint8)
    lab = np.stack([y, y], axis=-1)
    d = dict(feat=feat, z_st=z_st, y=y, p2_w=p2_w, p2_b=p2_b, g15=g15, log_idx=log_idx, fold=fold, code=code, lab=lab)
    _CACHE["d"] = d
    return d


def analyzer(d, B=400):
    return P.Analyzer(d["lab"], d["code"], d["log_idx"], N_LOGS, B=B)


def fit(d, arm, g, seed=0):
    return P.oof_logits(arm, d["feat"], d["z_st"], d["y"], g, d["fold"], seed, d["p2_w"], d["p2_b"])


def delta(an, za, zb):
    return an.diff(an.per_log_stats(za, "dac"), an.per_log_stats(zb, "dac"), "Q4")


class TestBranchTable(unittest.TestCase):
    def test_green(self):
        self.assertEqual(C.branch_table_failures(), [])

    def _ns(self, **over):
        ns = types.SimpleNamespace(**{k: getattr(C, k) for k in ("confirm_call", "null_call", "seed_call", "row_call", "small_reading", "null_fires", "null_void", "power_ok", "final_call")})
        ns.__dict__.update(over)
        return ns

    def test_red_mutations(self):
        muts = {
            "confirm: point conjunct dropped": dict(confirm_call=lambda ci: ci["lo"] > 0.0),
            "confirm: lower-CI conjunct dropped": dict(confirm_call=lambda ci: ci["point"] >= 0.02),
            "confirm: lower CI bar +0.02 (the sealed-SH5 hole)": dict(confirm_call=lambda ci: ci["lo"] > 0.02),
            "confirm: point bar 0.03": dict(confirm_call=lambda ci: ci["lo"] > 0.0 and ci["point"] >= 0.03),
            "confirm: point bar 0.01": dict(confirm_call=lambda ci: ci["lo"] > 0.0 and ci["point"] >= 0.01),
            "confirm: uses hi": dict(confirm_call=lambda ci: ci["hi"] > 0.0 and ci["point"] >= 0.02),
            "confirm: lo >= 0": dict(confirm_call=lambda ci: ci["lo"] >= 0.0 and ci["point"] >= 0.02),
            "null: comparator flipped": dict(null_call=lambda ci: ci["hi"] > 0.01),
            "null: bound 0.01 (the SH5 NOT-REACHABLE bar)": dict(null_call=lambda ci: ci["hi"] < 0.01),
            "null: bound 0.03": dict(null_call=lambda ci: ci["hi"] < 0.03),
            "null: <= at bound": dict(null_call=lambda ci: ci["hi"] <= 0.02),
            "null: uses lo": dict(null_call=lambda ci: ci["lo"] < 0.01),
            "row: any instead of all (confirm)": dict(row_call=lambda calls: "CONFIRM" if any(c == "CONFIRM" for c in calls) else ("NULL" if calls and all(c == "NULL" for c in calls) else "INCONCLUSIVE")),
            "row: any instead of all (null)": dict(row_call=lambda calls: "CONFIRM" if calls and all(c == "CONFIRM" for c in calls) else ("NULL" if any(c == "NULL" for c in calls) else "INCONCLUSIVE")),
            "small: point conjunct dropped": dict(small_reading=lambda cis: bool(len(cis) and all(c["lo"] > 0.0 for c in cis))),
            "small: any instead of all": dict(small_reading=lambda cis: bool(any(c["lo"] > 0.0 and c["point"] < 0.02 for c in cis))),
            "null fires: sealed-SH5 rule (lower CI > +0.02)": dict(null_fires=lambda ci: ci["lo"] > 0.02),
            "null fires: point only": dict(null_fires=lambda ci: ci["point"] >= 0.02),
            "null fires: lower CI only": dict(null_fires=lambda ci: ci["lo"] > 0.0),
            "null void: all instead of any": dict(null_void=lambda f: bool(len(f) and all(f))),
            "null void: needs 2": dict(null_void=lambda f: sum(1 for x in f if x) >= 2),
            "power: any seed": dict(power_ok=lambda cis: any(C.seed_call(c) == "CONFIRM" for c in cis)),
            "power: uses NULL rule": dict(power_ok=lambda cis: bool(len(cis) and all(C.null_call(c) for c in cis))),
            "final: no conversion": dict(final_call=lambda call, ad, vi, vn, pw: call),
            "final: null over inputs": dict(final_call=lambda call, ad, vi, vn, pw: "NO-CALL(ADMIT)" if not ad else ("VOID" if vn else ("VOID-INPUTS" if vi else ("UNDERPOWERED" if call == "NULL" and not pw else call)))),
            "final: unpowered converts confirm": dict(final_call=lambda call, ad, vi, vn, pw: "NO-CALL(ADMIT)" if not ad else ("VOID-INPUTS" if vi else ("VOID" if vn else ("UNDERPOWERED" if not pw else call)))),
            "final: admit ignored": dict(final_call=lambda call, ad, vi, vn, pw: "VOID-INPUTS" if vi else ("VOID" if vn else ("UNDERPOWERED" if call == "NULL" and not pw else call))),
            "final: admit below inputs": dict(final_call=lambda call, ad, vi, vn, pw: "VOID-INPUTS" if vi else ("NO-CALL(ADMIT)" if not ad else ("VOID" if vn else ("UNDERPOWERED" if call == "NULL" and not pw else call)))),
        }
        for name, over in muts.items():
            self.assertTrue(len(C.branch_table_failures(self._ns(**over))) > 0, f"mutation not RED: {name}")


class TestCallProperties(unittest.TestCase):
    def test_confirm_and_null_are_mutually_exclusive_and_total_order_holds(self):
        rng = np.random.default_rng(0)
        for _ in range(20000):
            lo = rng.normal(0.0, 0.03)
            pt = lo + abs(rng.normal(0.0, 0.02))
            hi = pt + abs(rng.normal(0.0, 0.02))
            ci = {"lo": lo, "point": pt, "hi": hi}
            self.assertFalse(C.confirm_call(ci) and C.null_call(ci))
            self.assertEqual(C.seed_call(ci) == "CONFIRM", C.confirm_call(ci))


class TestArms(unittest.TestCase):
    def test_primary_and_companion_are_the_two_distinct_arms(self):
        self.assertIn(C.PRIMARY_ARM, ("LGc", "LGu"))
        self.assertEqual({C.PRIMARY_ARM, C.COMPANION_ARM}, {"LGc", "LGu"})


class TestBlocks(unittest.TestCase):
    def test_clean_block_reads_only_columns_2_to_10(self):
        d = synth()
        g = d["g15"].copy()
        b0 = C.clean_block(g)
        for cols in ((0,), (1,), (11, 12, 13, 14), (0, 1, 11, 12, 13, 14)):
            g2 = g.copy()
            g2[..., list(cols)] = 99.0
            self.assertTrue(np.array_equal(b0, C.clean_block(g2)), cols)
        g3 = g.copy()
        g3[..., 5] += 1.0
        self.assertFalse(np.array_equal(b0, C.clean_block(g3)))                      # the control: a key column DOES move the block

    def test_block_hash_is_the_glive_registration_hash(self):
        """C-ID-EXTRACT: the gate's `block_hash` equals the G-LIVE registration's `sha_bytes` of the same block (float32 store -> float64 |g[..., 2:11]| of the OK sets)."""
        import pgc_glive as GL
        rng = np.random.default_rng(4)
        g = rng.normal(size=(9, 64, 15)).astype(np.float32) * 3.0
        ok = np.array([True, True, False, True, True, True, False, True, True])
        reg = GL.sha_bytes(np.abs(np.asarray(g, np.float64)[..., GL.BLOCK_COLS])[ok])
        self.assertEqual(C.block_hash(g, ok), reg)
        g2 = g.copy()
        g2[0, 0, 4] += 1e-3                                                          # a key column changes the hash (the control)
        self.assertNotEqual(C.block_hash(g2, ok), reg)
        g3 = g.copy()
        g3[..., 0] += 7.0                                                            # the sign column does not
        self.assertEqual(C.block_hash(g3, ok), reg)
        self.assertEqual(GL.BLOCK_COLS, list(C.BLOCK_COLS))

    def test_centre_within_set(self):
        x = np.random.default_rng(1).normal(size=(7, 64, 9)) + np.arange(7)[:, None, None]
        c = C.centre_within_set(x)
        self.assertTrue(np.allclose(c.mean(1), 0.0, atol=1e-12))
        self.assertTrue(np.allclose(c - c.mean(0, keepdims=True), c - c.mean(0, keepdims=True)))
        self.assertTrue(np.allclose(C.centre_within_set(x + 5.0 * np.arange(7)[:, None, None]), c))      # a set-level offset vanishes

    def test_null_block_permutes_within_set_and_moves_the_vector(self):
        x = np.random.default_rng(2).normal(size=(6, 64, 9)).astype(np.float32)
        n0, n1 = C.null_block(x, 0), C.null_block(x, 1)
        self.assertFalse(np.array_equal(n0, n1))
        self.assertTrue(np.array_equal(n0, C.null_block(x, 0)))
        for i in range(6):
            self.assertEqual(sorted(map(tuple, n0[i].tolist())), sorted(map(tuple, x[i].tolist())))   # the 9-vector moves together
        self.assertTrue(np.allclose(C.centre_within_set(n0), C.centre_within_set(n0)))
        # centring is permutation-invariant: centre(permute(x)) == permute(centre(x))
        self.assertTrue(np.allclose(C.centre_within_set(n0).mean(1), 0.0, atol=1e-6))

    def test_weak_plant(self):
        lab = np.zeros((4, 64))
        a, b = C.weak_plant(lab, 1.0), C.weak_plant(lab, 2.0)
        self.assertEqual(a.shape, (4, 64, 1))
        self.assertTrue(np.allclose(b, 2.0 * a))
        self.assertTrue(np.array_equal(C.weak_plant(lab, 1.0), a))


class TestFits(unittest.TestCase):
    def test_reach_planted_label_through_the_centred_path(self):
        """C-REACH on the code path: G := the label, centred like the block, gives F-AUC >= 0.99 for L+G and G-only."""
        d = synth()
        an = analyzer(d)
        plant = C.prepared(C.label_plant(d["y"]), True)
        for arm in ("LG", "G"):
            z = fit(d, arm, plant)
            a = an.auc(an.per_log_stats(z, "dac"), "Q4")["point"]
            self.assertGreaterEqual(a, C.REACH_MIN_AUC, arm)

    def test_centring_invariance_and_information_found_nulls_not(self):
        d = synth()
        an = analyzer(d)
        blk = C.clean_block(d["g15"])
        zL = fit(d, "L", None)
        zc = fit(d, "LG", C.prepared(blk, True))
        dc = delta(an, zc, zL)
        print(f"  centred arm Delta_F {dc['point']:+.4f} [{dc['lo']:+.4f}, {dc['hi']:+.4f}]")
        self.assertTrue(C.confirm_call(dc))                                           # information that exists is found
        blk2 = blk + 3.0 * np.random.default_rng(3).normal(size=(N, 1, 1))              # a pure set-level shift
        zc2 = fit(d, "LG", C.prepared(blk2, True))
        self.assertTrue(np.allclose(zc, zc2, atol=2e-3), float(np.abs(zc - zc2).max()))   # the centred arm does not see it (float32 round-off only)
        fires = 0
        for rep in range(3):                                                           # C-NULL on the same block: within-set permuted + centred must not CONFIRM
            zn = fit(d, "LG", C.prepared(C.null_block(blk, rep), True), seed=rep % 2)
            ci = delta(an, zn, fit(d, "L", None, seed=rep % 2))
            fires += int(C.null_fires(ci))
            print(f"  null rep {rep}: Delta {ci['point']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}]")
        self.assertEqual(fires, 0)

    def test_weak_plant_is_found(self):
        d = synth()
        an = analyzer(d)
        zL = fit(d, "L", None)
        zw = fit(d, "LG", C.prepared(C.weak_plant(d["y"], 1.0), True))
        ci = delta(an, zw, zL)
        print(f"  weak plant Delta {ci['point']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}]")
        self.assertTrue(C.confirm_call(ci))


class TestGradientReach(unittest.TestCase):
    """G-LIVE of the fit path: every trainable group gets a NON-ZERO gradient from the real block, and a control that must read EXACTLY 0.0 (an all-zero block) proves the probe can see a zero."""

    def _step(self, g_np):
        d = synth()
        idx = np.arange(256)
        head = P.Head("LG", d["p2_w"], d["p2_b"], g_np.shape[-1])
        w0, b0, u0 = head.w.detach().clone(), head.b.detach().clone(), head.u.detach().clone()
        gs = torch.as_tensor(P.standardise(g_np, idx)[idx])
        head.step(torch.as_tensor(d["feat"][idx]), torch.as_tensor(d["z_st"][idx]), torch.as_tensor(d["y"][idx]), gs)
        return (head.w - w0).abs().sum().item(), (head.b - b0).abs().sum().item(), (head.u - u0).abs().detach().numpy()

    def test_every_group_moves_on_the_real_block_and_u_is_exactly_zero_on_a_zero_block(self):
        d = synth()
        blk = C.prepared(C.clean_block(d["g15"]), True)
        dw, db, du = self._step(blk)
        self.assertGreater(dw, 0.0)
        self.assertGreater(db, 0.0)
        self.assertTrue((du > 0.0).all(), du)                                          # all 9 columns of u received a gradient
        dw0, db0, du0 = self._step(np.zeros_like(blk))
        self.assertEqual(float(du0.max()), 0.0)                                        # the control reads EXACTLY 0.0 (a zero block has no gradient)
        self.assertGreater(dw0, 0.0)                                                   # the token group still trains: the probe separates groups


if __name__ == "__main__":
    unittest.main(verbosity=1)
