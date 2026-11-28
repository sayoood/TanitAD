"""Synthetic CPU-only tests for headrow_probe.py (no bank, no GPU).  One deliberate-regression (mutation) arm per check: the mutated code MUST go RED.
Run:  CUDA_VISIBLE_DEVICES=-1 python -I test_headrow_probe.py
"""
from __future__ import annotations

import dataclasses
import io
import math
import os
import sys
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np          # noqa: E402
import torch                # noqa: E402

import headrow_probe as H   # noqa: E402

E = H.E
torch.set_num_threads(2)


def _stack(net, feat, z, z6s, col):
    """The column assembly of the landed forward (probe leaf rows feed the non-target columns), shared by the mutated forwards."""
    if net.probe:
        z_oth = torch.nn.functional.linear(feat, net.frozen_W6, net.frozen_b6)
        return torch.stack([z if c == col else z_oth[..., c] for c in range(6)], -1)
    return torch.stack([z if c == col else z6s[..., c] for c in range(6)], -1)


def make_syn(seed: int = 0, n_logs: int = 10, per_log: int = 6, planted: bool = False):
    """A small synthetic bank: random features, a P2-like head, labels in the real value sets, stored logits = feat . W + b + noise."""
    rng = np.random.default_rng(seed)
    N = n_logs * per_log
    feat = torch.from_numpy(rng.standard_normal((N, 64, 256)).astype(np.float16))
    W6 = (rng.standard_normal((6, 256)) * 0.05).astype(np.float32)
    b6 = (rng.standard_normal(6) * 0.1).astype(np.float32)
    z6 = (feat.float().numpy() @ W6.T + b6 + 0.01 * rng.standard_normal((N, 64, 6))).astype(np.float32)
    lb = np.zeros((N, 64, 6), np.float32)
    lb[..., 0] = rng.choice([0.0, 0.5, 1.0], size=(N, 64), p=[0.1, 0.1, 0.8])
    lb[..., 1] = rng.choice([0.0, 1.0], size=(N, 64), p=[0.25, 0.75])
    safe = (lb[..., 0] >= 1.0) & (lb[..., 1] >= 1.0)
    lb[..., 2] = np.where(safe, rng.random((N, 64)), 0.0)
    lb[..., 3] = rng.choice([0.0, 1.0], size=(N, 64), p=[0.2, 0.8])
    lb[..., 4] = 1.0
    lb[..., 5] = 1.0
    if planted:   # make the TTC label a nonlinear function of the features so a hidden branch can find it
        g = (feat.float().numpy()[..., 0] * feat.float().numpy()[..., 1])
        lb[..., 3] = (g > 0).astype(np.float32)
    zt = torch.from_numpy(z6)
    top8 = torch.from_numpy(np.stack([E.top8_np(z6[i]) for i in range(N)]).astype(np.int64))
    log_idx = np.repeat(np.arange(n_logs), per_log)
    fold = log_idx % 5
    return H.make_pd(feat, zt, lb, top8, log_idx, fold, n_logs), W6, b6


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pd, cls.W6, cls.b6 = make_syn(0)
        cls.draws = E.boot_draws(cls.pd.n_logs, B=400)

    # ------------------------------------------------------------------ concordance, Brier, pick level
    def _brute_conc(self, Z, y, v, idx8=None):
        num = den = 0.0
        for i in range(len(Z)):
            ids = list(range(64)) if idx8 is None else [int(j) for j in idx8[i]]
            for a in ids:
                for b in ids:
                    if v[i, a] and v[i, b] and y[i, a] > y[i, b]:
                        den += 1
                        num += 1.0 if Z[i, a] > Z[i, b] else (0.5 if Z[i, a] == Z[i, b] else 0.0)
        return num, den

    def test_conc_matches_bruteforce_all_rows(self):
        rng = np.random.default_rng(1)
        Z = rng.standard_normal((self.pd.n_sets, 64)).astype(np.float32)
        Z[:, :5] = 0.0                                                              # forced ties
        for r in H.ROWS:
            c = H.ROW_COL[r]
            y, v = self.pd.y[c], self.pd.valid[c]
            n, d = H.conc_per_set(Z, y, v)
            n8, d8 = H.conc_per_set(Z, y, v, self.pd.top8)
            bn, bd = self._brute_conc(Z, y.numpy(), v.numpy())
            bn8, bd8 = self._brute_conc(Z, y.numpy(), v.numpy(), self.pd.top8.numpy())
            self.assertAlmostEqual(n.sum(), bn, places=6)
            self.assertEqual(d.sum(), bd)
            self.assertAlmostEqual(n8.sum(), bn8, places=6)
            self.assertEqual(d8.sum(), bd8)

    def test_mutation_tie_convention_goes_red(self):
        Z = np.zeros((self.pd.n_sets, 64), np.float32)                              # all ties -> 0.5 exactly
        c = H.ROW_COL["ttc"]
        n, d = H.conc_per_set(Z, self.pd.y[c], self.pd.valid[c])
        self.assertAlmostEqual(n.sum() / d.sum(), 0.5, places=12)
        n_bad, _ = H.conc_per_set(Z, self.pd.y[c], self.pd.valid[c], tie_half=False)
        self.assertNotAlmostEqual(n_bad.sum() / d.sum(), 0.5, places=3)             # the mutated tie rule is caught

    def test_ep_pairs_need_both_safe(self):
        c = H.ROW_COL["ep"]
        Z = np.random.default_rng(2).standard_normal((self.pd.n_sets, 64)).astype(np.float32)
        _, d = H.conc_per_set(Z, self.pd.y[c], self.pd.valid[c])
        y = self.pd.y[c].numpy()
        s = self.pd.safe
        manual = sum(1 for i in range(self.pd.n_sets) for a in range(64) for b in range(64) if s[i, a] and s[i, b] and y[i, a] > y[i, b])
        self.assertEqual(d.sum(), manual)
        _, d_all = H.conc_per_set(Z, self.pd.y[c], torch.ones_like(self.pd.valid[c]))
        self.assertGreater(d_all.sum(), d.sum())                                    # dropping the safe mask changes the count: the mask is live

    def test_brier(self):
        rng = np.random.default_rng(3)
        Z = rng.standard_normal((self.pd.n_sets, 64)).astype(np.float32)
        c = H.ROW_COL["ep"]
        s, n = H.brier_per_set(Z, self.pd.y[c], self.pd.valid[c])
        p = 1 / (1 + np.exp(-Z.astype(np.float64)))
        m = self.pd.valid[c].numpy()
        self.assertAlmostEqual(s.sum(), (((p - self.pd.y[c].numpy()) ** 2) * m).sum(), places=8)
        self.assertEqual(n.sum(), m.sum())

    def test_pick_level_replaces_only_that_column(self):
        rng = np.random.default_rng(4)
        Z = rng.standard_normal((self.pd.n_sets, 64)).astype(np.float32) * 3
        a = H.cell_logs(self.pd, "dac", Z)
        z6 = self.pd.z6.numpy().copy()
        z6[..., 1] = Z
        pk = E.picks_np(z6)
        r = np.arange(len(pk))
        self.assertAlmostEqual(a["pdn"].sum(), self.pd.pdms[r, pk].sum(), places=5)
        z6_wrong = self.pd.z6.numpy().copy()
        z6_wrong[..., 0] = Z                                                         # mutation m10: the wrong column
        pk_w = E.picks_np(z6_wrong)
        self.assertFalse(np.array_equal(pk, pk_w))                                   # the column test can tell them apart
        ref = H.p2_logs(self.pd, "dac")
        self.assertEqual(ref["changed"], 0.0)

    # ------------------------------------------------------------------ the estimator
    def test_bootstrap_unit_is_the_log(self):
        per_set = np.ones(self.pd.n_sets)
        with self.assertRaises(H.ProbeError):
            H.ratio_ci(per_set, per_set, self.draws)                                 # mutation m8: resampling SETS is refused
        ok = np.ones(self.pd.n_logs)
        ci = H.ratio_ci(ok, ok * 2, self.draws)
        self.assertAlmostEqual(ci["point"], 0.5)
        self.assertAlmostEqual(ci["lo"], 0.5)

    def test_wilson(self):
        lo, hi = H.wilson(0, 100)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)

    # ------------------------------------------------------------------ nets: identity, init
    def test_identity_all_rows_arms(self):
        for r in H.ROWS:
            for arm in H.ARMS:
                res = H.identity_check(self.pd, self.W6, self.b6, r, arm)
                self.assertTrue(res["ok"], res)
                self.assertTrue(res["z_equal_stored_all_sets"])

    def test_mutations_of_identity_go_red(self):
        col = H.ROW_COL["ep"]

        def zero_skip():                                                             # m6: skip initialised to zeros instead of P2's row (but ref = P2): z != stored
            n = H.ProbeNet("M", self.W6, self.b6, col, 0, 0)
            with torch.no_grad():
                n.skip_w.zero_()
            return n

        def nonzero_w2():                                                            # m7: non-zero output layer
            n = H.ProbeNet("M", self.W6, self.b6, col, 0, 0)
            with torch.no_grad():
                n.W2 += 0.01
            return n
        self.assertFalse(H.identity_check(self.pd, self.W6, self.b6, "ep", "M", net_factory=zero_skip)["ok"])
        self.assertFalse(H.identity_check(self.pd, self.W6, self.b6, "ep", "M", net_factory=nonzero_w2)["ok"])

    def test_m_has_hidden256_gelu_no_dropout_and_l_has_none(self):
        m = H.ProbeNet("M", self.W6, self.b6, 0, 0, 0)
        l = H.ProbeNet("L", self.W6, self.b6, 0, 0, 0)
        self.assertEqual(tuple(m.W1.shape), (256, 256))
        self.assertEqual(tuple(m.W2.shape), (1, 256))
        self.assertFalse(any(isinstance(x, torch.nn.Dropout) for x in m.modules()))
        self.assertFalse(hasattr(l, "W1"))
        self.assertEqual(len(m.trainable()), 6)
        self.assertEqual(len(l.trainable()), 2)

    def test_cold_start_ignores_stored_logits(self):
        net = H.ProbeNet("M", self.W6, self.b6, 3, 0, 0, cold=True)
        with torch.no_grad():
            net.W2 += 0.02
        a = H.forward_z6(net, self.pd.feat[:8], self.pd.z6[:8], 3)[..., 3]
        b = H.forward_z6(net, self.pd.feat[:8], self.pd.z6[:8] + 5.0, 3)[..., 3]
        self.assertTrue(torch.equal(a, b))
        warm = H.ProbeNet("M", self.W6, self.b6, 3, 0, 0, cold=False)
        a = H.forward_z6(warm, self.pd.feat[:8], self.pd.z6[:8], 3)[..., 3]
        b = H.forward_z6(warm, self.pd.feat[:8], self.pd.z6[:8] + 5.0, 3)[..., 3]
        self.assertFalse(torch.equal(a, b))

    def test_only_target_column_changes(self):
        net = H.ProbeNet("M", self.W6, self.b6, 1, 0, 0, perturb=True)
        z6 = H.forward_z6(net, self.pd.feat[:8], self.pd.z6[:8], 1)
        for c in range(6):
            same = torch.equal(z6[..., c], self.pd.z6[:8][..., c])
            self.assertEqual(same, c != 1)

    # ------------------------------------------------------------------ reach proof + mutations
    def test_reach_proof_green_every_row_and_arm(self):
        for r in H.ROWS:
            for arm in H.ARMS:
                rep = H.reach_proof(self.pd, self.W6, self.b6, r, arm, n=64)
                self.assertEqual(rep["zero_control"]["frozen_rows_w"], 0.0)
                self.assertGreater(rep["full"]["skip_w"], 0.0)
                if arm == "M":
                    self.assertEqual(rep["zero_output_layer"]["W1"], 0.0)
                    self.assertGreater(rep["zero_output_layer"]["W2"], 0.0)
                    self.assertGreater(rep["full"]["W1"], 0.0)

    def test_mutation_no_detach_goes_red(self):
        def fwd(net, feat, z6s, col):                                                # m1: the features are NOT detached
            col = net.col if col is None else col
            feat = feat.float()
            base = torch.zeros_like(z6s[..., col]) if net.cold else z6s[..., col]
            z = base + feat @ (net.skip_w - net.ref_w) + (net.skip_b - net.ref_b)
            if net.arm == "M":
                h = torch.nn.functional.gelu(feat @ net.W1.t() + net.b1)
                z = z + (h @ net.W2.t() + net.b2).squeeze(-1)
            return _stack(net, feat, z, z6s, col)
        for arm in H.ARMS:
            with self.assertRaises(H.ReachError):
                H.reach_proof(self.pd, self.W6, self.b6, "ttc", arm, n=64, forward_fn=fwd)

    def test_mutation_leak_of_another_column_goes_red(self):
        def loss(z6, y, v, col, w=1.0):                                              # m2: the loss also trains column col+1
            return H.row_loss(z6, y, v, col, w) + H.row_loss(z6, y, v, (col + 1) % 6, w)
        with self.assertRaises(H.ReachError):
            H.reach_proof(self.pd, self.W6, self.b6, "nc", "L", n=64, loss_fn=loss)

    def test_mutation_drop_ep_safe_mask_goes_red(self):
        def loss(z6, y, v, col, w=1.0):                                              # m3: no valid mask
            return H.row_loss(z6, y, torch.ones_like(v), col, w)
        with self.assertRaises(H.ReachError):
            H.reach_proof(self.pd, self.W6, self.b6, "ep", "L", n=64, loss_fn=loss)

    def test_mutation_stop_gradient_into_hidden_goes_red(self):
        def fwd(net, feat, z6s, col):                                                # m4: the hidden branch is cut from the graph
            col = net.col if col is None else col
            feat = feat.detach().float()
            base = z6s[..., col]
            z = base + feat @ (net.skip_w - net.ref_w) + (net.skip_b - net.ref_b)
            if net.arm == "M":
                h = torch.nn.functional.gelu(feat @ net.W1.t() + net.b1).detach()
                z = z + (h @ net.W2.t() + net.b2).squeeze(-1)
            return _stack(net, feat, z, z6s, col)
        with self.assertRaises(H.ReachError):
            H.reach_proof(self.pd, self.W6, self.b6, "dac", "M", n=64, forward_fn=fwd)

    def test_mutation_zero_loss_weight_goes_red(self):
        def loss(z6, y, v, col, w=1.0):                                              # m5: the BCE term has weight 0
            return H.row_loss(z6, y, v, col, 0.0)
        with self.assertRaises(H.ReachError):
            H.reach_proof(self.pd, self.W6, self.b6, "ttc", "L", n=64, loss_fn=loss)

    # ------------------------------------------------------------------ fit / OOF
    def test_same_batches_for_L_M_and_all_rows(self):
        cfg = H.ProbeCfg(epochs=2, batch_sets=16)
        tr = np.nonzero(self.pd.fold != 0)[0]
        seqs = {}
        for arm in H.ARMS:
            for r in ("nc", "ep"):
                rec = []
                H.fit_arm(self.pd, tr, cfg, arm, r, 0, 0, self.W6, self.b6, batch_hook=lambda b, rec=rec: rec.append(b.tolist()))
                seqs[(arm, r)] = rec
        ref = seqs[("L", "nc")]
        for k, v in seqs.items():
            self.assertEqual(v, ref, k)
        rec1 = []
        H.fit_arm(self.pd, tr, cfg, "L", "nc", 1, 0, self.W6, self.b6, batch_hook=lambda b: rec1.append(b.tolist()))
        self.assertNotEqual(rec1, ref)                                               # the seed changes the batches

    def test_fit_step_count_and_hyperparameters(self):
        cfg = H.ProbeCfg(epochs=3, batch_sets=16)
        steps = []
        H.fit_arm(self.pd, np.arange(40), cfg, "L", "ttc", 0, 0, self.W6, self.b6, batch_hook=lambda b: steps.append(len(b)))
        self.assertEqual(len(steps), 3 * math.ceil(40 / 16))
        d = torch.optim.Adam(H.ProbeNet("L", self.W6, self.b6, 3, 0, 0).trainable(), lr=H.LR, betas=H.BETAS, eps=H.ADAM_EPS, weight_decay=H.WEIGHT_DECAY).defaults
        self.assertEqual((d["lr"], d["weight_decay"], H.EPOCHS, H.BATCH_SETS), (1e-3, 1e-4, 20, 256))

    def test_oof_scores_every_set_once_and_never_trains_on_its_fold(self):
        seen_oof = np.zeros(self.pd.n_sets, int)
        trains = []

        def spy(pd, tr, cfg, arm, row, seed, fold, W6, b6, y_train=None, cold=False):
            oof = np.nonzero(pd.fold == fold)[0]
            self.assertEqual(len(np.intersect1d(oof, tr)), 0)
            self.assertEqual(len(set(pd.log_idx[oof]) & set(pd.log_idx[tr])), 0)       # grouped by LOG
            seen_oof[oof] += 1
            trains.append(len(tr))
            return H.fit_arm(pd, tr[:16], dataclasses.replace(cfg, epochs=1), arm, row, seed, fold, W6, b6, y_train=y_train, cold=cold)
        res = H.oof_logits(self.pd, H.ProbeCfg(epochs=1), "L", "ttc", 0, self.W6, self.b6, fit_fn=spy)
        self.assertTrue((seen_oof == 1).all())
        self.assertEqual(res["Z"].shape, (self.pd.n_sets, 64))
        self.assertTrue(np.isfinite(res["Z"]).all())

    def test_M_can_find_a_nonlinear_signal_L_cannot(self):
        pd, W6, b6 = make_syn(5, n_logs=20, per_log=10, planted=True)
        cfg = H.ProbeCfg(epochs=12, batch_sets=32, lr=3e-3)
        out = {}
        for arm in H.ARMS:
            out[arm] = H.oof_logits(pd, cfg, arm, "ttc", 0, W6, b6, cold=True)["Z"]
        c = H.ROW_COL["ttc"]
        cl = np.mean([H.conc_per_set(out["L"], pd.y[c], pd.valid[c])[0].sum() / H.conc_per_set(out["L"], pd.y[c], pd.valid[c])[1].sum()])
        nm, dm = H.conc_per_set(out["M"], pd.y[c], pd.valid[c])
        self.assertLess(cl, 0.58)
        self.assertGreater(nm.sum() / dm.sum(), cl + 0.05)

    # ------------------------------------------------------------------ decisions and controls rules
    def _dec(self, ml, l=0.9, p2=0.9, c2=False, cpos=True):
        ps = {s: {"ml": {"lo": ml[s][0], "hi": ml[s][1]}, "l_conc": l, "p2_conc": p2} for s in (0, 1)}
        return H.decide_row(ps, c2, cpos)

    def test_decision_rule_table(self):
        self.assertEqual(self._dec({0: (0.012, 0.03), 1: (0.011, 0.02)})["call"], "CAPACITY-LIMITED")
        self.assertEqual(self._dec({0: (0.010, 0.03), 1: (0.02, 0.03)})["call"], "INCONCLUSIVE")           # lower CI == 0.01 is not > 0.01
        self.assertEqual(self._dec({0: (0.012, 0.03), 1: (-0.01, 0.02)})["call"], "INCONCLUSIVE")          # one seed only
        self.assertEqual(self._dec({0: (-0.01, 0.005), 1: (-0.02, 0.009)})["call"], "FEATURE-LIMITED")
        self.assertEqual(self._dec({0: (-0.01, 0.010), 1: (-0.02, 0.009)})["call"], "INCONCLUSIVE")        # upper CI == 0.01 is not < 0.01
        self.assertEqual(self._dec({0: (-0.01, 0.005), 1: (-0.02, 0.009)}, l=0.85, p2=0.9)["call"], "INCONCLUSIVE")   # L not within 0.01 of P2
        self.assertEqual(self._dec({0: (-0.01, 0.005), 1: (-0.02, 0.009)}, l=0.905, p2=0.9)["call"], "FEATURE-LIMITED")
        self.assertEqual(self._dec({0: (-0.01, 0.005), 1: (-0.02, 0.009)}, l=0.92, p2=0.9)["call"], "INCONCLUSIVE")    # two-sided

    def test_vetoes(self):
        cap = {0: (0.012, 0.03), 1: (0.011, 0.02)}
        feat = {0: (-0.01, 0.005), 1: (-0.02, 0.009)}
        self.assertEqual(self._dec(cap, c2=True)["call"], "INCONCLUSIVE")
        self.assertEqual(self._dec(feat, c2=True)["call"], "FEATURE-LIMITED")        # a RED C2 does not void FEATURE
        self.assertEqual(self._dec(feat, cpos=False)["call"], "INCONCLUSIVE")
        self.assertEqual(self._dec(cap, cpos=False)["call"], "CAPACITY-LIMITED")     # a RED C-POS does not void CAPACITY

    def test_c2_rule_any_replicate_lower_ci(self):
        green = [{"lo": 0.49 - 0.01 * i, "point": 0.5, "hi": 0.52} for i in range(5)]
        self.assertFalse(H.c2_cell_red(green))
        red = list(green)
        red[3] = {"lo": 0.5001, "point": 0.51, "hi": 0.52}
        self.assertTrue(H.c2_cell_red(red))                                          # one replicate suffices
        pointonly = [{"lo": 0.45, "point": 0.51, "hi": 0.55}] * 5
        self.assertFalse(H.c2_cell_red(pointonly))                                   # a per-seed point rule is NOT used
        self.assertTrue(H.c2_dec_red([{"lo": 0.0101}]))
        self.assertFalse(H.c2_dec_red([{"lo": 0.0100}]))

    def test_cleak_flag(self):
        self.assertEqual(H.cleak_flag({0: {"lo": 0.001, "hi": 0.01}, 1: {"lo": 0.002, "hi": 0.01}}), "LEAK")
        self.assertEqual(H.cleak_flag({0: {"lo": -0.01, "hi": 0.0}, 1: {"lo": -0.01, "hi": -0.001}}), "CLEAN")
        self.assertEqual(H.cleak_flag({0: {"lo": 0.001, "hi": 0.01}, 1: {"lo": -0.002, "hi": 0.01}}), "AMBIGUOUS")

    def test_combine_outcome(self):
        mk = lambda c: {"call": c}                                                      # noqa: E731
        allf = H.combine_outcome({r: mk("FEATURE-LIMITED") for r in H.ROWS}, True)
        self.assertEqual(allf["label"], "ALL-FEATURE")
        mixed = H.combine_outcome({"nc": mk("CAPACITY-LIMITED"), "dac": mk("INCONCLUSIVE"), "ep": mk("FEATURE-LIMITED"), "ttc": mk("CAPACITY-LIMITED")}, True)
        arms = {x["row"]: x["arm"] for x in mixed["next_arms"]}
        self.assertIn("safety-row MLP", arms["nc"])
        self.assertIn("f = 25", arms["dac"])
        self.assertIn("nonlinear-head", arms["ttc"])
        self.assertNotIn("ep", arms)

    # ------------------------------------------------------------------ controls: shuffles, null, noise, planted
    def test_shuffle_valid_preserves_multiset_and_mask(self):
        for r in H.ROWS:
            c = H.ROW_COL[r]
            y, v = self.pd.y[c], self.pd.valid[c]
            ys = H.shuffle_valid(y, v, H.c2_seed(1, r))
            self.assertTrue(torch.equal(ys[~v], y[~v]))                              # invalid entries untouched (EP: unsafe)
            for i in range(0, self.pd.n_sets, 7):
                self.assertTrue(np.array_equal(np.sort(ys[i][v[i]].numpy()), np.sort(y[i][v[i]].numpy())))
            self.assertFalse(torch.equal(ys, y))
            self.assertTrue(torch.equal(ys, H.shuffle_valid(y, v, H.c2_seed(1, r))))     # deterministic
            self.assertFalse(torch.equal(ys, H.shuffle_valid(y, v, H.c2_seed(2, r))))

    def test_null_trial_structure_and_size(self):
        tr = H.null_trial(self.pd, "ttc", self.draws, 0)
        self.assertEqual(len(tr["lo_minus_half"]), 5)
        self.assertIn(tr["red"], (True, False))
        # iid scores: the bootstrap point is near 0.5 (so the rule is a calibrated one-sided test, not a point rule)
        self.assertTrue(all(abs(x) < 0.2 for x in tr["lo_minus_half"]))

    def test_noise_cell_zero_sigma_is_identity(self):
        pk0 = E.picks_np(self.pd.z6.numpy())
        c = H.noise_cell(self.pd, "dac", 0.0, 0, 0, self.draws, pk0)
        self.assertEqual(c["changed_pick_fraction"], 0.0)
        self.assertEqual(c["d_unsafe"]["point"], 0.0)
        big = H.noise_cell(self.pd, "dac", 1.0, 0, 4, self.draws, pk0)
        self.assertGreater(big["changed_pick_fraction"], 0.0)
        m = H.match_noise(0.5, [c, big])
        self.assertIn(m["sigma"], (0.0, 1.0))

    def test_planted_labels_have_no_linear_part_and_are_deterministic(self):
        feat = self.pd.feat
        y1, info = H.planted_labels(feat)
        y2, _ = H.planted_labels(feat)
        self.assertTrue(torch.equal(y1, y2))
        self.assertTrue(((y1 > 0) & (y1 < 1)).all())
        x = feat.reshape(-1, 256).float().numpy()
        x = (x - x.mean(0)) / x.std(0)
        g = np.log(y1.numpy().reshape(-1) / (1 - y1.numpy().reshape(-1)))
        beta = np.linalg.lstsq(np.c_[x, np.ones(len(x))], g, rcond=None)[0]
        resid = g - np.c_[x, np.ones(len(x))] @ beta
        self.assertLess(1 - resid.var() / g.var(), 1e-3)                              # R^2 of the best linear fit ~ 0

    # ------------------------------------------------------------------ A-RES
    def _ares_inputs(self):
        rng = np.random.default_rng(9)
        pk_old = E.picks_np(self.pd.z6.numpy())
        pk_new = np.where(rng.random(self.pd.n_sets) < 0.6, rng.integers(0, 64, self.pd.n_sets), pk_old)
        return pk_new, pk_old

    def test_ares_matches_bruteforce(self):
        pk_new, pk_old = self._ares_inputs()
        st = H.ares_stats(self.pd, pk_new, pk_old, self.draws)
        idx = np.arange(self.pd.n_sets)
        bf = H.ares_bruteforce(self.pd, pk_new, pk_old, idx)
        self.assertEqual(st["n_V"], bf["n_V"])
        self.assertEqual(int(round(st["frac_V_m_ge0"]["point"] * st["n_V"])), bf["n_V_m_ge0"])
        a = st["auc_all"]["point"]
        self.assertAlmostEqual(a, bf["auc_num"] / bf["auc_den"], places=9)

    def test_ares_mutation_arms_go_red(self):
        pk_new, pk_old = self._ares_inputs()
        base = H.ares_stats(self.pd, pk_new, pk_old, self.draws)
        flip = H.ares_stats(self.pd, pk_new, pk_old, self.draws, flip_sign=True)
        self.assertNotEqual(base["frac_V_m_ge0"]["point"], flip["frac_V_m_ge0"]["point"])             # sign of m
        allsafe = H.ares_stats(self.pd, pk_new, pk_old, self.draws, unsafe_mask_fn=lambda s: np.ones_like(s))
        self.assertEqual(allsafe["n_V"], 0)                                                          # unsafe mask
        self.assertTrue(allsafe["auc_all"].get("empty"))
        self.assertGreater(base["n_V"], 0)
        shifted = np.roll(self.pd.top8.numpy(), 1, axis=0)
        wrong8 = H.ares_stats(self.pd, pk_new, pk_old, self.draws, top8_override=shifted)
        self.assertNotEqual(base["auc_all"]["point"], wrong8["auc_all"]["point"])                    # top-8 membership
        self.assertNotEqual(base["share_V_new_is_best_pnc_pdac"]["point"], H.ares_stats(self.pd, pk_new, pk_old, self.draws, top8_override=shifted)["share_V_new_is_best_pnc_pdac"]["point"])

    def test_ares_identity_c0(self):
        pk_old = E.picks_np(self.pd.z6.numpy())
        st = H.ares_stats(self.pd, pk_old, pk_old, self.draws)
        self.assertEqual((st["n_V"], st["n_S"], st["n_D"]), (0, 0, 0))

    def test_ares_flag_table(self):
        mk = lambda n, fr, au: {"n_V": n, "frac_V_m_ge0": {"point": fr}, "auc_V": {"point": au}}      # noqa: E731
        self.assertEqual(H.ares_flag({"0": mk(500, 0.6, 0.55), "1": mk(500, 0.55, 0.6)})["flag"], "R-RES")
        self.assertEqual(H.ares_flag({"0": mk(500, 0.4, 0.75), "1": mk(500, 0.3, 0.7)})["flag"], "R-OVER")
        self.assertEqual(H.ares_flag({"0": mk(500, 0.6, 0.55), "1": mk(500, 0.4, 0.75)})["flag"], "R-MIXED")
        self.assertEqual(H.ares_flag({"0": mk(99, 0.6, 0.55), "1": mk(500, 0.55, 0.6)})["flag"], "R-NONE")
        self.assertEqual(H.ares_flag({"0": mk(500, 0.6, 0.65), "1": mk(500, 0.55, 0.6)})["flag"], "R-MIXED")

    def test_apply_row_chunked_equals_landed(self):
        w = (self.W6[2] + 0.01).astype(np.float32)
        b = (self.b6[2:3] + 0.1).astype(np.float32)
        a = H.apply_row_chunked(self.pd.z6, self.pd.feat, w, b, self.W6[2], self.b6[2:3], 2, chunk=7)
        c = E.apply_row(self.pd.z6, self.pd.feat, w, b, self.W6[2], self.b6[2:3], col=2)
        self.assertTrue(torch.equal(a, c))

    # ------------------------------------------------------------------ guard and parent pin
    def test_pathguard_refuses_forbidden_tokens_and_outside_paths(self):
        with tempfile.TemporaryDirectory() as d:
            g = E.PathGuard(read_dirs=[d], read_files=[], write_dirs=[d])
            with open(os.path.join(d, "ok.txt"), "wb") as f:
                f.write(b"x")
            self.assertEqual(E._io_read_bytes(g, os.path.join(d, "ok.txt")), b"x")
            os.makedirs(os.path.join(d, "navtest_tables"))
            with open(os.path.join(d, "navtest_tables", "t.txt"), "wb") as f:
                f.write(b"x")
            with self.assertRaises(E.PathGuardError):
                E._io_read_bytes(g, os.path.join(d, "navtest_tables", "t.txt"))
            with self.assertRaises(E.PathGuardError):
                E._io_read_bytes(g, os.path.join(os.path.dirname(d), "elsewhere.txt"))
            with self.assertRaises(E.PathGuardError):
                E._io_write_bytes(g, os.path.join(os.path.dirname(d), "w.txt"), b"x")

    def test_parent_is_the_landed_epr1(self):
        import hashlib
        with open(os.path.join(HERE, "epr1.py"), "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), H.PARENT_EPR1_SHA256)

    def test_cfg_sha_changes_with_every_lever(self):
        base = H.ProbeCfg().sha256()
        for k, v in (("epochs", 19), ("batch_sets", 128), ("lr", 2e-3), ("weight_decay", 0.0), ("hidden", 128), ("bce_w", 0.5), ("c2_reps", 4)):
            self.assertNotEqual(dataclasses.replace(H.ProbeCfg(), **{k: v}).sha256(), base, k)


if __name__ == "__main__":
    unittest.main(verbosity=1)
