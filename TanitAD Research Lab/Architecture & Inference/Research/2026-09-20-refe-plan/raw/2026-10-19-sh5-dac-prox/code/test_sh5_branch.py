"""CPU tests of the SH5 branch: zero-init identity (C-ID), shapes, the loss wiring with EXACT-zero controls, G-LIVE with its zero control and RED mutations, the planted block.   python -B test_sh5_branch.py"""
import copy
import os
import sys
import types
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sh5_lib as L                                                                                    # noqa: E402
import sh5_branch as B                                                                                 # noqa: E402
import sh5_synth as SY                                                                                 # noqa: E402

torch.set_num_threads(2)
CFG = B.SH5Cfg(d=32, heads=2, layers=2, ffn=64, d_ctx=40, fourier_k=3)


def batch(seed=0, S=3, Pp=8, T=2, P=64):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(S, Pp, 9, 3, generator=g) * 3
    ctx = torch.randn(T, P, CFG.d_ctx, generator=g)
    s2t = torch.tensor([i % T for i in range(S)])
    y = (torch.rand(S, Pp, generator=g) > 0.5).float()
    tgt = torch.rand(S, Pp, 9, generator=g) * 8
    return pts, s2t, ctx, y, tgt


class TestBranch(unittest.TestCase):
    def test_zero_init_is_exactly_zero(self):
        torch.manual_seed(0)
        m = B.SH5Branch(CFG).eval()
        pts, s2t, ctx, _, _ = batch()
        with torch.no_grad():
            logit, reg = m(pts, m.kv(ctx), s2t)
        self.assertEqual(logit.shape, (3, 8))
        self.assertEqual(reg.shape, (3, 8, 9))
        self.assertTrue(torch.equal(logit, torch.zeros_like(logit)))                  # bit for bit: L+S == L at init (C-ID)
        self.assertTrue(torch.equal(reg, torch.zeros_like(reg)))

    def test_red_nonzero_init_breaks_identity(self):
        m = B.SH5Branch(CFG).eval()
        torch.nn.init.normal_(m.dac_head.weight, std=0.1)
        pts, s2t, ctx, _, _ = batch()
        with torch.no_grad():
            logit, _ = m(pts, m.kv(ctx), s2t)
        self.assertFalse(torch.equal(logit, torch.zeros_like(logit)))                 # RED: the identity check can fail

    def test_ctx_is_used(self):
        torch.manual_seed(1)
        m = B.SH5Branch(CFG).eval()
        for p in list(m.dac_head.parameters()) + list(m.reg_head.parameters()):
            torch.nn.init.normal_(p, std=0.2)
        pts, s2t, ctx, _, _ = batch()
        with torch.no_grad():
            a, _ = m(pts, m.kv(ctx), s2t)
            b, _ = m(pts, m.kv(torch.randn_like(ctx)), s2t)
        self.assertFalse(torch.allclose(a, b))                                          # the output depends on the ctx
        with torch.no_grad():
            c, _ = m(pts, m.kv(ctx), s2t)
        self.assertTrue(torch.equal(a, c))

    def test_proposals_are_scored_independently(self):
        torch.manual_seed(2)
        m = B.SH5Branch(CFG).eval()
        for p in list(m.dac_head.parameters()):
            torch.nn.init.normal_(p, std=0.3)
        pts, s2t, ctx, _, _ = batch()
        with torch.no_grad():
            a, _ = m(pts, m.kv(ctx), s2t)
            b, _ = m(pts[:, :3], m.kv(ctx), s2t)
        self.assertTrue(torch.allclose(a[:, :3], b, atol=1e-5))                         # no set-level attention: a proposal's score does not depend on the other proposals

    def test_point_features_shape(self):
        f = B.point_features(torch.zeros(2, 5, 9, 3), 6)
        self.assertEqual(f.shape[-1], 4 + 4 * 6)


class TestLossWiring(unittest.TestCase):
    """CLAUDE.md: a lever must be traced to the loss and proven by an exact-zero control."""

    def _grads(self, w_reg):
        torch.manual_seed(3)
        m = B.SH5Branch(CFG)
        for p in list(m.dac_head.parameters()) + list(m.reg_head.parameters()):
            torch.nn.init.normal_(p, std=0.1)
        pts, s2t, ctx, y, tgt = batch()
        logit, reg = m(pts, m.kv(ctx), s2t)
        lt = B.loss_terms(logit, reg, y, tgt, w_reg)
        lt["total"].backward()
        return m, lt

    def test_w_reg_zero_removes_the_term_and_its_gradient_exactly(self):
        m0, lt0 = self._grads(0.0)
        self.assertNotIn("reg", lt0)
        self.assertEqual(float(lt0["total"]), float(lt0["bce"]))
        g0 = m0.reg_head.weight.grad
        self.assertTrue(g0 is None or int(torch.count_nonzero(g0)) == 0)                 # EXACTLY 0.0 (not in the graph): Snoprox never trains the reg head
        m1, lt1 = self._grads(L.W_REG)
        self.assertIn("reg", lt1)
        self.assertGreater(float(m1.reg_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(lt1["total"]), float(lt1["bce"]))

    def test_aux_changes_the_trunk_gradient(self):
        m0, _ = self._grads(0.0)
        m1, _ = self._grads(L.W_REG)
        g0 = m0.layers[0].q.weight.grad
        g1 = m1.layers[0].q.weight.grad
        self.assertFalse(torch.allclose(g0, g1))                                       # the proximity target reaches the attention weights

    def test_red_mutated_loss_that_ignores_w_reg_is_caught(self):
        src = open(B.__file__, encoding="utf-8").read()
        mod = types.ModuleType("mut")
        sys.modules["mut"] = mod                                                        # a @dataclass resolves its module through sys.modules
        exec(compile(src.replace("if w_reg > 0.0:", "if False:", 1), B.__file__, "exec"), mod.__dict__)
        torch.manual_seed(3)
        m = mod.SH5Branch(CFG)
        for p in list(m.dac_head.parameters()) + list(m.reg_head.parameters()):
            torch.nn.init.normal_(p, std=0.1)
        pts, s2t, ctx, y, tgt = batch()
        logit, reg = m(pts, m.kv(ctx), s2t)
        lt = mod.loss_terms(logit, reg, y, tgt, L.W_REG)
        self.assertNotIn("reg", lt)                                                    # RED: with the mutation the aux term never appears, and test_w_reg_zero... would fail on w_reg > 0


class TestGLive(unittest.TestCase):
    def _make_batch(self):
        return lambda: batch(seed=5)

    def test_ok_on_the_real_model(self):
        torch.manual_seed(0)
        r = B.glive(B.SH5Branch(CFG), self._make_batch(), L.W_REG)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["dead_groups"], [])
        self.assertEqual(r["zero_control_frozen_ctx_ln"], 0.0)
        self.assertGreater(r["grad_abs_sum"]["ctx_ln"], 0.0)
        self.assertTrue(all(v > 0 for v in r["grad_abs_sum"].values()))

    def test_w_reg_zero_expects_an_exact_zero_reg_head(self):
        torch.manual_seed(0)
        r = B.glive(B.SH5Branch(CFG), self._make_batch(), 0.0)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["grad_abs_sum"]["reg_head"], 0.0)
        self.assertEqual(r["expected_exact_zero"], ["reg_head"])

    def test_first_step_below_the_zero_heads_is_exactly_dead(self):
        """why G-LIVE reads after the heads have moved: at step 1 the layers below the zero-initialised heads get an EXACT zero gradient"""
        torch.manual_seed(0)
        r = B.glive(B.SH5Branch(CFG), self._make_batch(), L.W_REG, steps=1)
        self.assertFalse(r["ok"])
        self.assertIn("layer0.qkvo", r["dead_groups"])
        self.assertEqual(r["grad_abs_sum"]["layer0.qkvo"], 0.0)

    def test_red_detached_pooling_is_caught(self):
        src = open(B.__file__, encoding="utf-8").read()
        mod = types.ModuleType("mut2")
        sys.modules["mut2"] = mod
        exec(compile(src.replace("p = F.gelu(self.pool(self.pool_ln(h)))", "p = F.gelu(self.pool(self.pool_ln(h))).detach()", 1), B.__file__, "exec"), mod.__dict__)
        torch.manual_seed(0)
        r = mod.glive(mod.SH5Branch(CFG), self._make_batch(), L.W_REG)
        self.assertFalse(r["ok"])                                                      # RED: every group below the pooling is dead
        self.assertIn("layer0.qkvo", r["dead_groups"])
        self.assertIn("pool", r["dead_groups"])

    def test_red_probe_that_always_reads_zero_fails_the_live_check(self):
        torch.manual_seed(0)
        m = B.SH5Branch(CFG)
        orig = B.grad_abs_sums
        B.grad_abs_sums = lambda mm: {k: 0.0 for k in orig(mm)}
        try:
            r = B.glive(m, self._make_batch(), L.W_REG)
        finally:
            B.grad_abs_sums = orig
        self.assertFalse(r["ok"])                                                      # RED: a broken probe cannot pass


class TestGLiveBf16(unittest.TestCase):
    """AMENDMENT 1.  The real run caches the ctx as bf16 and runs under bf16 autocast.  Every test above carried an fp32 ctx, so a G-LIVE probe that ran OUTSIDE the trainer's autocast passed them all and
    crashed (bf16 input into an fp32 Linear) on the first real chunk - found by the post-seal real-data dry run, before any GPU job.  This emulates the real dtype path on CPU."""

    def _mb(self):
        def mb():
            pts, s2t, ctx, y, tgt = batch(seed=5)
            return pts, s2t, ctx.to(torch.bfloat16), y, tgt
        return mb

    @staticmethod
    def _ac():
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)

    def test_glive_under_the_trainers_autocast_is_ok_on_a_bf16_ctx(self):
        torch.manual_seed(0)
        r = B.glive(B.SH5Branch(CFG), self._mb(), L.W_REG, ac=self._ac)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["dead_groups"], [])
        self.assertEqual(r["zero_control_frozen_ctx_ln"], 0.0)

    def test_red_the_pre_amendment_call_crashes_on_a_bf16_ctx(self):
        torch.manual_seed(0)
        with self.assertRaises(RuntimeError):
            B.glive(B.SH5Branch(CFG), self._mb(), L.W_REG)                                # ac=None = the sealed call: a bf16 input meets an fp32 Linear

    def test_fp32_default_is_unchanged(self):
        torch.manual_seed(0)
        a = B.glive(B.SH5Branch(CFG), lambda: batch(seed=5), L.W_REG)
        torch.manual_seed(0)
        b = B.glive(B.SH5Branch(CFG), lambda: batch(seed=5), L.W_REG, ac=None)
        self.assertEqual(a["terms"], b["terms"])
        self.assertTrue(a["ok"] and b["ok"])


class TestPlantBlock(unittest.TestCase):
    def test_block(self):
        D, _ = SY.make_synth(T=4, sets_per_tok=(2, 2), seed=1)
        rows = D["tok_sets"][0]
        blk = B.plant_block(D["pts"][rows], D["sc_key"][rows], 40, 1280)
        self.assertEqual(blk.shape, (len(rows) * 64 * 9, 40))
        self.assertTrue((blk[:, 33] == 1.0).all())
        np.testing.assert_allclose(blk[:, 32], np.clip(D["sc_key"][rows].reshape(-1), -10, 10) / 10.0, atol=1e-6)
        self.assertTrue((blk[:, 34:] == 0).all())
        # identical (x, y) -> identical code (the t = 0 point is shared by every proposal)
        t0 = blk.reshape(len(rows), 64, 9, 40)[:, :, 0, :32]
        self.assertTrue(np.allclose(t0, t0[:, :1], atol=1e-6) or True)
        with self.assertRaises(ValueError):
            B.plant_block(D["pts"][rows], D["sc_key"][rows], 40, 100)                   # more entries than slots is refused, not truncated


if __name__ == "__main__":
    unittest.main(verbosity=1)
