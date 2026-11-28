"""CPU tests of the ctx-build plumbing that does NOT need the planner or the GPU: the fidelity gate (with RED mutations), the lock-wrapped runner (never runs the job without the lock; never touches a
foreign lock), the bf16-bits ctx reader, the evidence-built gates (C-ID from the arrays), and the argv surface.   python -B test_sh5_ctx.py"""
import json
import os
import sys
import tempfile
import time
import types
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sh5_lib as L                                                                                    # noqa: E402
import sh5_ctx_build as CB                                                                             # noqa: E402
import sh5_gpulock as GL                                                                               # noqa: E402
import sh5_gates as GT                                                                                 # noqa: E402
import sh5_train_eval as TE                                                                            # noqa: E402

TMP = os.environ.get("SH5_TMP")


def mutated(module, old, new):
    src = open(module.__file__, encoding="utf-8").read()
    assert old in src, old
    m = types.ModuleType(module.__name__ + "_mut")
    m.__file__ = module.__file__
    sys.modules[m.__name__] = m
    exec(compile(src.replace(old, new, 1), module.__file__, "exec"), m.__dict__)
    return m


class TestFidelityGate(unittest.TestCase):
    def test_pass_and_fail(self):
        self.assertTrue(CB.fid_gate(np.zeros(100))["pass"])
        g = CB.fid_gate(np.concatenate([np.zeros(98), np.full(2, 0.5)]))            # 98 % within tol < 99 %
        self.assertFalse(g["pass"])
        g = CB.fid_gate(np.concatenate([np.zeros(99), np.full(1, 0.5)]))            # exactly 99 %
        self.assertTrue(g["pass"])
        self.assertFalse(CB.fid_gate(np.array([]))["pass"])                           # nothing measured is not a pass
        self.assertFalse(CB.fid_gate(np.full(100, 2e-3))["pass"])                     # every set just outside the 1e-3 tolerance

    def test_red_mutations_break_it(self):
        m = mutated(CB, "frac >= L.FID_MIN_FRAC", "frac >= 0.0")
        self.assertTrue(m.fid_gate(np.full(100, 9.0))["pass"])                        # RED: the mutant passes a hopeless ctx, the real gate does not
        self.assertFalse(CB.fid_gate(np.full(100, 9.0))["pass"])
        m = mutated(CB, "(maxabs <= L.FID_TOL)", "(maxabs <= 1.0)")
        self.assertTrue(m.fid_gate(np.full(100, 0.5))["pass"])
        self.assertFalse(CB.fid_gate(np.full(100, 0.5))["pass"])

    def test_row(self):
        a = np.zeros((64, 6), np.float32)
        b = a.copy()
        b[3, 2] = 0.25
        self.assertAlmostEqual(CB.fidelity_row(a, b), 0.25)
        self.assertEqual(CB.fidelity_row(a, a), 0.0)


class TestLockedRunner(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(dir=TMP)
        self.lock, self.mark = os.path.join(self.d, "l.lock"), os.path.join(self.d, "l.mark")
        self.kw = dict(lock=self.lock, mark=self.mark, used_fn=lambda: 1500, alive_fn=lambda p: False, sleep_fn=lambda s: None, compute_fn=lambda: [], avail_fn=lambda: 99.0, poll_s=0)

    def test_runs_under_the_lock_and_releases(self):
        seen = {}

        def job():
            seen["lock_held"] = os.path.exists(self.lock)
            seen["env"] = os.environ.get("SH5_GPU_LOCKED")
            return 0
        rc = GL.run_locked_inprocess(job, 10, min_avail_gb=3.0, log=lambda m: None, **self.kw)
        self.assertEqual(rc, 0)
        self.assertTrue(seen["lock_held"])
        self.assertEqual(seen["env"], "1")
        self.assertFalse(os.path.exists(self.lock))

    def test_a_foreign_lock_means_the_job_never_runs_and_the_lock_is_untouched(self):
        open(self.lock, "w").write(json.dumps({"job": "refcv8-chain", "pid": 123}))
        ran = []
        t = {"v": 0.0}
        rc = GL.run_locked_inprocess(lambda: ran.append(1) or 0, 5, min_avail_gb=3.0, log=lambda m: None, now_fn=lambda: t.__setitem__("v", t["v"] + 10) or t["v"], **self.kw)
        self.assertEqual(rc, GL.EX_BUSY)
        self.assertEqual(ran, [])
        self.assertEqual(json.loads(open(self.lock).read()), {"job": "refcv8-chain", "pid": 123})

    def test_release_even_if_the_job_raises_and_never_a_changed_lock(self):
        def boom():
            raise RuntimeError("x")
        with self.assertRaises(RuntimeError):
            GL.run_locked_inprocess(boom, 10, min_avail_gb=3.0, log=lambda m: None, **self.kw)
        self.assertFalse(os.path.exists(self.lock))

        def swap():
            os.remove(self.lock)
            open(self.lock, "w").write(json.dumps({"job": "someone-else"}))
            return 0
        GL.run_locked_inprocess(swap, 10, min_avail_gb=3.0, log=lambda m: None, **self.kw)
        self.assertEqual(json.loads(open(self.lock).read()), {"job": "someone-else"})        # we did not remove a lock whose bytes are not ours

    def test_ram_gate_blocks_before_the_lock_is_taken(self):
        kw = dict(self.kw, avail_fn=lambda: 1.0)
        t = {"v": 0.0}
        ran = []
        rc = GL.run_locked_inprocess(lambda: ran.append(1) or 0, 5, min_avail_gb=8.0, log=lambda m: None, now_fn=lambda: t.__setitem__("v", t["v"] + 10) or t["v"], **kw)
        self.assertEqual(rc, GL.EX_BUSY)
        self.assertFalse(os.path.exists(self.lock))                                          # the lock was never even taken while RAM-starved
        self.assertEqual(ran, [])


class TestNpyCtx(unittest.TestCase):
    def test_bf16_bits_roundtrip(self):
        d = tempfile.mkdtemp(dir=TMP)
        x = torch.randn(5, 16, 8).to(torch.bfloat16)
        bits = x.view(torch.int16).numpy().view(np.uint16)
        p = os.path.join(d, "ctx.npy")
        np.save(p, bits)
        prov = TE.NpyCtx(p, [4, 0, 2], device="cpu")
        got = prov.load([0, 1, 2])
        self.assertTrue(torch.equal(got, x[[4, 0, 2]]))                                        # token t reads file row ctx_rows[t]
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertEqual(prov.bytes_per_token, 16 * 8 * 2)
        import unittest.mock as mock
        with mock.patch("torch.stack", side_effect=AssertionError("load() must not stack the whole shard on the host")):
            again = prov.load([0, 1, 2])                                                       # Amendment 1: preallocated and filled per token, never a host-side stack of the shard
        self.assertTrue(torch.equal(again, got))
        self.assertTrue(torch.equal(prov.load([2]), x[[2]]))
        np.save(os.path.join(d, "bad.npy"), bits.astype(np.float32))
        with self.assertRaises(L.Sh5Error):
            TE.NpyCtx(os.path.join(d, "bad.npy"), [0], device="cpu")                           # a non-uint16 file is refused, not reinterpreted


class TestGates(unittest.TestCase):
    def _cid(self, d, nonzero=False, epochs=0):
        for s in L.SEEDS:
            z = np.zeros((20, 64), np.float32)
            if nonzero:
                z[3, 5] = 1e-7
            np.savez(f"{d}/OOF_S_s{s}.npz", z=z, reg=np.zeros((20, 64, 9), np.float16))
            json.dump({"epochs": epochs}, open(f"{d}/OOF_S_s{s}.json", "w"))

    def test_cid_from_the_arrays(self):
        d = tempfile.mkdtemp(dir=TMP)
        self._cid(d)
        self.assertTrue(GT.check_cid(d)["ok"])
        d2 = tempfile.mkdtemp(dir=TMP)
        self._cid(d2, nonzero=True)
        self.assertFalse(GT.check_cid(d2)["ok"])                                                # RED: a single non-zero logit fails the identity
        d3 = tempfile.mkdtemp(dir=TMP)
        self._cid(d3, epochs=3)
        self.assertFalse(GT.check_cid(d3)["ok"])                                                # an arm that trained is not the E=0 pre-flight
        self.assertFalse(GT.check_cid(tempfile.mkdtemp(dir=TMP))["ok"])                          # missing evidence is not a pass

    def test_build_reads_evidence_not_flags(self):
        d = tempfile.mkdtemp(dir=TMP)
        self._cid(d)
        json.dump({"ok": True}, open(f"{d}/gl.json", "w"))
        json.dump({"c_fid": {"pass": True}}, open(f"{d}/man.json", "w"))
        json.dump({"all_passed": True}, open(f"{d}/ts.json", "w"))
        g = GT.build(d, [f"{d}/gl.json"], f"{d}/man.json", f"{d}/ts.json")
        self.assertTrue(all(g[k] for k in ("c_id", "g_live", "c_fid", "red_table")))
        json.dump({"ok": False}, open(f"{d}/gl.json", "w"))
        self.assertFalse(GT.build(d, [f"{d}/gl.json"], f"{d}/man.json", f"{d}/ts.json")["g_live"])
        json.dump({"c_fid": {"pass": False}}, open(f"{d}/man.json", "w"))
        self.assertFalse(GT.build(d, [f"{d}/gl.json"], f"{d}/man.json", f"{d}/ts.json")["c_fid"])
        json.dump({"all_passed": False}, open(f"{d}/ts.json", "w"))
        self.assertFalse(GT.build(d, [f"{d}/gl.json"], f"{d}/man.json", f"{d}/ts.json")["red_table"])


class TestCtxKeep(unittest.TestCase):
    """Amendment 1: `fit` and `analyze` share ONE definition of the analysis set - the tokens whose ctx was built (the manifest's done flags)."""

    def test_done_flags_define_the_set_and_the_rows(self):
        d = tempfile.mkdtemp(dir=TMP)
        np.save(f"{d}/done.npy", np.array([True, False, True, True]))
        json.dump({"tokens": ["a", "b", "c", "d"], "done_file": f"{d}/done.npy"}, open(f"{d}/man.json", "w"))
        keep, row_of = TE.ctx_keep(np.array(["d", "a", "x", "b", "c"]), f"{d}/man.json")
        self.assertEqual(keep.tolist(), [True, True, False, False, True])                      # 'b' was never built; 'x' is not in the manifest
        self.assertEqual(row_of, {"a": 0, "c": 2, "d": 3})                                      # the ctx-file row of each built token (file order, not the inputs' order)

    def test_no_done_file_means_all_manifest_tokens(self):
        d = tempfile.mkdtemp(dir=TMP)
        json.dump({"tokens": ["a", "b"]}, open(f"{d}/man.json", "w"))
        keep, row_of = TE.ctx_keep(np.array(["b", "a", "z"]), f"{d}/man.json")
        self.assertEqual(keep.tolist(), [True, True, False])
        self.assertEqual(row_of, {"a": 0, "b": 1})


class TestSurface(unittest.TestCase):
    def test_stage_table_covers_the_registered_arms(self):
        flat = [a for a, _ in TE.STAGES["A1"]] + [a for a, _ in TE.STAGES["A2"]]
        self.assertEqual(sorted(flat), sorted(["S", "Snoprox", "Sshuf", "plant", "plantweak"]))
        self.assertEqual(TE.STAGES["B"], "null")
        for a in flat + ["nullprox", "nulllabel"]:
            self.assertIn(a, TE.ARM_ORDER)

    def test_spec_names_are_unique(self):
        names = {TE.Spec(a, k, s, r).name for a in TE.ARM_ORDER for k in range(5) for s in (0, 1) for r in range(5)}
        self.assertTrue(len(names) >= 5 * 2 * 5 * 2)

    def test_fit_stage_B_assigns_replicates_to_seeds(self):
        got = {s: [r for r in range(L.NULL_REPS) if r % 2 == s] for s in (0, 1)}
        self.assertEqual(got[0], [0, 2, 4])
        self.assertEqual(got[1], [1, 3])
        self.assertEqual(sorted(got[0] + got[1]), list(range(L.NULL_REPS)))                    # all 5 replicates run exactly once

    def test_cli_help_parses(self):
        for mod, argv in ((TE, ["fit", "--help"]), (TE, ["analyze", "--help"]), (CB, ["run", "--help"]), (CB, ["plan", "--help"])):
            with self.assertRaises(SystemExit) as c:
                mod.main(argv)
            self.assertEqual(c.exception.code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
