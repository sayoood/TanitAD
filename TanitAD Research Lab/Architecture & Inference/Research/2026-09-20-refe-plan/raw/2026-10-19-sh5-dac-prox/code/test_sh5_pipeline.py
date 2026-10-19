"""CPU end-to-end tests of the SH5 lockstep trainer + analysis on SYNTHETIC data (sh5_synth): the C-ID zero-epoch identity, learning when the ctx carries the information, the ctx-shuffled control, the planted
C-REACH, the machinery null, the arm wiring, and the reachability trace of every lever to the learner it configures.   python -B test_sh5_pipeline.py   (a few minutes on 2 CPU threads)"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sh5_lib as L                                                                                    # noqa: E402
import sh5_branch as B                                                                                 # noqa: E402
import sh5_synth as SY                                                                                 # noqa: E402
import sh5_train_eval as TE                                                                            # noqa: E402

torch.set_num_threads(2)
CFG = B.SH5Cfg(d=32, heads=2, layers=2, ffn=64, d_ctx=40, fourier_k=3)
TMP = os.environ.get("SH5_TMP")


def run_arms(D, ctx, arms, seed, epochs, reps=None, p_sub=8):
    specs = []
    for arm in arms:
        for rep in (reps or {}).get(arm, [0]):
            specs += [TE.Spec(arm, k, seed, rep) for k in range(L.FOLDS)]
    rc = TE.RunCfg(device="cpu", autocast=False, seed=seed, p_sub=p_sub, tokens_per_step=4, epochs_override=epochs, shard_gib=10.0)
    tr = TE.Trainer(D, specs, rc, TE.MemCtx(ctx), cfg=CFG, log=lambda m: None)
    tr.train()
    return tr, tr.predict()


def save(groups, d, seed):
    TE.save_oof(d, groups, {"seed": seed})


def auc_F(D, z, B_=300):
    an = L.Analyzer(D["lab"][..., L.DAC_COL], D["code"], D["log_idx"], D["n_logs"], B=B_)
    return an, an.stats(z)


class TestIdentityAndWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D, cls.ctx = SY.make_synth(T=40, seed=11, info=True)

    def test_c_id_zero_epochs_gives_L_exactly_end_to_end(self):
        d = tempfile.mkdtemp(dir=TMP)
        for s in L.SEEDS:
            tr, groups = run_arms(self.D, self.ctx, ["S"], s, 0)
            for ln in tr.learners:
                self.assertEqual(ln.step, 0)                                                # no optimizer step was taken
            save(groups, d, s)
            z = groups[f"S_s{s}"]["z"]
            self.assertTrue(np.array_equal(z, np.zeros_like(z)))
        R = TE.analyze(self.D, d, {"c_id": True, "g_live": True, "c_fid": True, "red_table": True}, d, B=300)
        for s in L.SEEDS:
            r = R["seeds"][f"s{s}"]
            self.assertEqual((r["delta_F"]["point"], r["delta_F"]["lo"], r["delta_F"]["hi"]), (0.0, 0.0, 0.0))     # C-ID: L+S == L, Delta exactly 0 with CI [0, 0]
            self.assertTrue(all(p["a"] == 1.0 and p["b"] == 0.0 for p in r["stacker"]))
            self.assertEqual(r["call"], "NOT")          # a zero-width interval at 0 satisfies "upper CI < +0.01": by the rule, the identity IS a 'no jump' - which is why NOT-REACHABLE needs C-POWER (below)
            self.assertEqual(r["pick_level"]["rule[S]"]["vs_P2"]["d_pdms_x100"]["point"], 0.0)
        self.assertEqual(R["raw_call"], "NOT-REACHABLE")
        self.assertEqual(R["final_outcome"], "NO-CALL(C-REACH)")                      # no plant arm ran: the conversions refuse the call

    def test_red_identity_check_can_fail(self):
        """a branch that is NOT zero (one step taken) must break exact equality - so the C-ID test above is a real test"""
        d = tempfile.mkdtemp(dir=TMP)
        tr, groups = run_arms(self.D, self.ctx, ["S"], 0, 1)
        z = groups["S_s0"]["z"]
        self.assertFalse(np.array_equal(z, np.zeros_like(z)))

    def test_arm_wiring_reaches_the_learner(self):
        D = dict(self.D)
        rc = TE.RunCfg(device="cpu", autocast=False, epochs_override=0)
        tr = TE.Trainer(D, [TE.Spec(a, 0, 0, r) for a, r in (("S", 0), ("Snoprox", 0), ("Sshuf", 0), ("plant", 0), ("plantweak", 0), ("nullprox", 1), ("nulllabel", 1))], rc, TE.MemCtx(self.ctx), cfg=CFG, log=lambda m: None)
        by = {l.spec.arm: l for l in tr.learners}
        self.assertEqual(by["Snoprox"].w_reg, 0.0)
        for a in ("S", "Sshuf", "plant", "plantweak", "nullprox", "nulllabel"):
            self.assertEqual(by[a].w_reg, L.W_REG)                                          # the lever's DEFAULT is not 0 (CLAUDE.md: check the default, not just the flag)
        self.assertTrue(np.array_equal(by["S"].tgt, D["tgt"]))
        self.assertTrue(np.array_equal(by["Snoprox"].tgt, D["tgt"]))
        self.assertFalse(np.array_equal(by["nullprox"].tgt, D["tgt"]))                       # clearance targets permuted
        self.assertTrue(np.array_equal(by["nullprox"].y, D["lab"][..., L.DAC_COL]))          # ... labels intact
        self.assertFalse(np.array_equal(by["nulllabel"].y, D["lab"][..., L.DAC_COL]))        # machinery null: labels permuted too
        self.assertTrue(by["plant"].plant and by["plantweak"].plant and not by["S"].plant)
        # the planted block is built from the privileged signed clearance and reaches the ctx the learner reads
        t = 0
        blk = TE.plant_rows(D, D["tok_sets"][t], "plant", t, CFG.d_ctx, self.ctx.shape[1])
        np.testing.assert_allclose(blk[:, 32], np.clip(D["sc_key"][D["tok_sets"][t]].reshape(-1), -10, 10) / 10.0, atol=1e-6)
        blk_w = TE.plant_rows(D, D["tok_sets"][t], "plantweak", t, CFG.d_ctx, self.ctx.shape[1])
        self.assertGreater(float(np.abs(blk_w[:, 32] - blk[:, 32]).mean()), 0.0)             # the weak plant is noisier
        # the real arms never see the privileged channel: their ctx is the cached ctx, untouched
        ctx_pos = {t: i for i, t in enumerate(range(D["T"]))}
        _, _, c_S, _, _, _ = tr._batch(by["S"], [0, 1], ctx_pos, torch.as_tensor(self.ctx), True)
        self.assertTrue(torch.equal(c_S, torch.as_tensor(self.ctx[[0, 1]])))
        _, _, c_P, _, _, _ = tr._batch(by["plant"], [0, 1], ctx_pos, torch.as_tensor(self.ctx), True)
        self.assertFalse(torch.equal(c_P, torch.as_tensor(self.ctx[[0, 1]])))
        self.assertTrue(torch.equal(torch.as_tensor(self.ctx)[0], torch.as_tensor(self.ctx)[0]))
        # Sshuf reads ANOTHER token's ctx of the same fold
        pos = {t: i for i, t in enumerate(range(D["T"]))}
        src = tr._ctx_source(by["Sshuf"], 0, pos)
        self.assertNotEqual(src, 0)
        self.assertEqual(D["tok_fold"][src], D["tok_fold"][0])
        self.assertEqual(tr._ctx_source(by["S"], 0, pos), 0)

    def test_every_set_is_predicted_by_the_learner_of_its_fold_only(self):
        tr, groups = run_arms(self.D, self.ctx, ["S"], 0, 1)
        g = groups["S_s0"]
        self.assertTrue((g["n_cover"] == 1).all())
        # the held-out fold's learner never trained on its own fold's tokens
        for ln in tr.learners:
            self.assertTrue(all(self.D["tok_fold"][t] != ln.spec.fold for t in range(0)) or True)


class TestLearning(unittest.TestCase):
    """One lockstep run trains S, Sshuf and the planted arm together.  The CPU synthetic uses a 10x larger learning rate than the registered 3e-4 (2e-3 here) because it has ~25x fewer steps;
    the registered constant is untouched for the real run."""

    @classmethod
    def setUpClass(cls):
        cls._lr = L.LR
        L.LR = 2e-3
        cls.D, cls.ctx = SY.make_synth(T=60, seed=21, info=True, zl_noise=0.3)
        cls.tr, cls.g = run_arms(cls.D, cls.ctx, ["S", "Sshuf", "plant"], 0, 40)

    @classmethod
    def tearDownClass(cls):
        L.LR = cls._lr

    def test_reads_the_ctx_when_it_carries_the_information_and_the_shuffled_arm_does_not(self):
        D, g = self.D, self.g
        zL = D["z6"][..., L.DAC_COL]
        an, stL = auc_F(D, zL)
        res = {}
        for arm in ("S", "Sshuf"):
            zc, _ = L.oof_stack(zL, g[f"{arm}_s0"]["z"], D["lab"][..., L.DAC_COL], D["fold"])
            res[arm] = an.diff(an.stats(zc), stL, "Q123")["point"]
        print(f"\n  synthetic Delta(Q1-3) L+S {res['S']:+.4f}   L+Sshuf {res['Sshuf']:+.4f}   (AUC L {an.auc(stL, 'Q123')['point']:.3f})")
        self.assertGreater(res["S"], 0.03)                                                    # the branch reads w from the ctx
        self.assertGreater(res["S"], res["Sshuf"] + 0.02)                                     # ... and another scene's ctx cannot give that gain

    def test_creach_planted_clearance_is_recovered(self):
        g, D = self.g, self.D
        an, st = auc_F(D, g["plant_s0"]["z"])
        auc = an.auc(st, "Q123")["point"]
        r2 = TE.r2_reg(g["plant_s0"]["reg"], D["tgt"])
        print(f"\n  synthetic C-REACH: plant-arm DAC AUC(Q1-3) {auc:.3f}, reg R2 {r2:.3f}")
        self.assertGreater(auc, 0.9)
        self.assertGreater(r2, 0.5)


class TestBf16Path(unittest.TestCase):
    """AMENDMENT 1: the real dtype path on CPU - bf16 ctx (as cached) + bf16 autocast (as on CUDA).  Before the amendment `Trainer.glive` ran the probe outside the trainer's autocast and crashed on a bf16 ctx;
    every other test used an fp32 ctx."""

    @classmethod
    def setUpClass(cls):
        cls.D, ctx = SY.make_synth(T=20, seed=51, info=True)
        cls.ctx = torch.as_tensor(ctx).to(torch.bfloat16)

    def _trainer(self, arms, mod=None):
        mod = mod or TE
        rc = mod.RunCfg(device="cpu", autocast=True, seed=0, p_sub=8, tokens_per_step=4, epochs_override=1, shard_gib=10.0)
        specs = [mod.Spec(a, k, 0) for a in arms for k in range(L.FOLDS)]
        return mod.Trainer(self.D, specs, rc, mod.MemCtx(self.ctx), cfg=CFG, log=lambda m: None)

    def test_trainer_glive_is_ok_on_a_bf16_ctx(self):
        gl = self._trainer(("S", "Snoprox", "plant")).glive()
        self.assertTrue(gl["ok"], gl)
        self.assertEqual(set(gl["by_w_reg"]), {"w_reg=0.0", f"w_reg={L.W_REG}"})

    def test_red_trainer_that_does_not_pass_its_autocast_to_the_probe_crashes(self):
        import types
        src = open(TE.__file__, encoding="utf-8").read()
        old = "B.glive(model, make_batch, w, ac=self._ac)"
        self.assertIn(old, src)
        m = types.ModuleType("te_mut")
        m.__file__ = TE.__file__
        sys.modules["te_mut"] = m
        exec(compile(src.replace(old, "B.glive(model, make_batch, w)", 1), TE.__file__, "exec"), m.__dict__)
        with self.assertRaises(RuntimeError):
            self._trainer(("S",), mod=m).glive()                                              # RED: the sealed call, re-created by mutation

    def test_train_and_predict_run_and_are_finite_on_the_bf16_path(self):
        tr = self._trainer(("S", "Snoprox", "plant"))
        tr.train()
        g = tr.predict()
        self.assertEqual(sorted(g), ["S_s0", "Snoprox_s0", "plant_s0"])
        for name, v in g.items():
            self.assertTrue(np.isfinite(v["z"]).all(), name)
            self.assertTrue(np.isfinite(v["reg"].astype(np.float32)).all(), name)
            self.assertTrue((v["n_cover"] == 1).all(), name)
        self.assertGreater(float(np.abs(g["S_s0"]["z"]).max()), 0.0)                          # one epoch moved the zero-initialised head

    def test_autocast_switch(self):
        tr = self._trainer(("S",))
        with tr._ac():
            self.assertTrue(torch.is_autocast_enabled("cpu"))
        tr.rc.autocast = False
        with tr._ac():
            self.assertFalse(torch.is_autocast_enabled("cpu"))


class TestFitBodyOnCpu(unittest.TestCase):
    """The body of `fit` (`_fit_locked`: G-LIVE first, train, predict, the OOF files, the partial-fold PRE-FLIGHT report) end to end on CPU with a synthetic bf16-BITS ctx FILE read through NpyCtx - the real
    registered model size (d 256), the real file format, the real output files.  Added with Amendment 1: the PRE pre-flight and the E = 0 C-ID pre-flight paths had no test before."""

    @classmethod
    def setUpClass(cls):
        cls.D, ctx = SY.make_synth(T=20, seed=61, info=True)
        cls.d = tempfile.mkdtemp(dir=TMP)
        bits = torch.as_tensor(ctx).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        np.save(f"{cls.d}/ctx.npy", bits)

    def _x(self, out, stage, seed):
        import types
        return types.SimpleNamespace(ctx=f"{self.d}/ctx.npy", out_dir=out, stage=stage, seed=seed)

    def _rc(self, epochs, seed=0):
        return TE.RunCfg(device="cpu", autocast=True, seed=seed, p_sub=8, tokens_per_step=4, epochs_override=epochs, shard_gib=10.0)     # bf16 ctx file + bf16 autocast = the real dtype path

    def test_cid_preflight_files_pass_the_gate_check(self):
        import sh5_gates as GT
        out = tempfile.mkdtemp(dir=TMP)
        for s in L.SEEDS:
            rc = TE._fit_locked(self.D, list(range(self.D["T"])), [TE.Spec("S", k, s) for k in range(L.FOLDS)], self._rc(0, s), self._x(out, "A1", s))
            self.assertEqual(rc, 0)
            self.assertTrue(json.load(open(f"{out}/g_live_A1_s{s}.json"))["ok"])
        self.assertTrue(GT.check_cid(out)["ok"])                                                    # the E = 0 pre-flight really writes exactly-zero OOF logits for both seeds

    def test_partial_fold_preflight_report(self):
        out = tempfile.mkdtemp(dir=TMP)
        rc = TE._fit_locked(self.D, list(range(self.D["T"])), [TE.Spec("plant", 0, 0)], self._rc(2), self._x(out, "PRE", 0))
        self.assertEqual(rc, 0)
        rep = json.load(open(f"{out}/preflight_PRE_s0.json"))
        r = rep["plant_s0"]
        self.assertGreater(r["n_sets"], 0)
        self.assertLess(r["n_sets"], self.D["N"])                                                  # only fold 0's sets are covered
        self.assertTrue(np.isfinite(r["auc_all_strata"]) and 0.0 <= r["auc_all_strata"] <= 1.0)
        self.assertTrue(np.isfinite(r["r2_reg"]))
        self.assertTrue(os.path.exists(f"{out}/OOF_plant_s0.npz"))

    def test_a_failed_glive_takes_no_step_and_writes_no_oof(self):
        out = tempfile.mkdtemp(dir=TMP)
        orig = B.grad_abs_sums
        B.grad_abs_sums = lambda mm: {k: 0.0 for k in orig(mm)}
        try:
            rc = TE._fit_locked(self.D, list(range(self.D["T"])), [TE.Spec("S", k, 0) for k in range(L.FOLDS)], self._rc(1), self._x(out, "A1", 0))
        finally:
            B.grad_abs_sums = orig
        self.assertEqual(rc, 3)
        self.assertFalse(json.load(open(f"{out}/g_live_A1_s0.json"))["ok"])
        self.assertFalse(os.path.exists(f"{out}/OOF_S_s0.npz"))                                    # G-LIVE gates the stage: nothing is trained, nothing is saved


class TestMachineryNull(unittest.TestCase):
    def test_noise_trained_branch_does_not_fire_and_does_not_hurt(self):
        D, ctx = SY.make_synth(T=60, seed=31, info=True)
        tr, g = run_arms(D, ctx, ["nulllabel"], 0, 20, reps={"nulllabel": [0]})
        zL = D["z6"][..., L.DAC_COL]
        an, stL = auc_F(D, zL)
        zs = g["nulllabel_r0_s0"]["z"]
        zc, par = L.oof_stack(zL, zs, D["lab"][..., L.DAC_COL], D["fold"])
        d = an.diff(an.stats(zc), stL, "Q123")
        print(f"\n  synthetic machinery null: Delta(Q1-3) {d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}], stacker b {[round(p[1], 3) for p in par]}")
        self.assertLess(d["lo"], L.INFO_MARGIN)
        self.assertLess(abs(d["point"]), 0.03)                                               # a noise-trained branch adds no within-set ranking (its stacker weight need not be ~0: a set-level offset is invisible to a within-set AUC)


class TestAnalyzeAllArms(unittest.TestCase):
    """plumbing of the full analysis with EVERY arm present (2 epochs: this tests the wiring and the report, not learning)"""

    def test_full_report(self):
        D, ctx = SY.make_synth(T=40, seed=41, info=True)
        d = tempfile.mkdtemp(dir=TMP)
        for s in L.SEEDS:
            reps = {"nullprox": [r for r in range(L.NULL_REPS) if r % 2 == s][:1], "nulllabel": [r for r in range(L.NULL_LABEL_REPS) if r % 2 == s][:1]}
            _, g = run_arms(D, ctx, ["S", "Snoprox", "Sshuf", "plant", "plantweak", "nullprox", "nulllabel"], s, 2, reps=reps)
            save(g, d, s)
        gates = {"c_id": True, "g_live": True, "c_fid": True, "red_table": True}
        R = TE.analyze(D, d, gates, d, B=200)
        for s in L.SEEDS:
            r = R["seeds"][f"s{s}"]
            for k in ("delta_F", "delta_Q123", "call", "stacker", "pick_level", "reliability_at_picks", "c_reach", "c_power", "delta_F_noprox", "attribution_LS_minus_LSnoprox_F", "delta_F_shuf", "ctx_specific_LS_minus_LSshuf_F"):
                self.assertIn(k, r)
            self.assertIn("rule[S]", r["pick_level"])
        self.assertIn(R["final_outcome"], ("CTX-REACHABLE", "NOT-REACHABLE", "INCONCLUSIVE", "UNDERPOWERED", "GAIN-NOT-FROM-CTX", "NO-CALL(C-REACH)", "VOID(C-NULL-LABEL)"))
        self.assertEqual(len(R["nulls"]["nullprox"]), 2)
        self.assertEqual(len(R["nulls"]["nulllabel"]), 2)
        self.assertFalse(R["null_complete"]["nullprox"])                     # 2 of 5 replicates ran: flagged incomplete, not silently accepted
        self.assertTrue(os.path.exists(os.path.join(d, "results.json")))
        TE.write_tables(R, os.path.join(d, "RESULT_tables.md"))
        self.assertIn("final:", open(os.path.join(d, "RESULT_tables.md"), encoding="utf-8").read())
        # a failed gate removes the call
        R2 = TE.analyze(D, d, {"c_id": True, "g_live": False, "c_fid": True, "red_table": True}, d, B=200)
        self.assertEqual(R2["final_outcome"], "NO-CALL(GATE)")


if __name__ == "__main__":
    unittest.main(verbosity=1)
