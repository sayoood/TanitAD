"""Tests for epr1ub_gate.py -- each check has a deliberate-regression arm that must make the gate FAIL.  CPU only, stdlib unittest:

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I test_epr1ub_gate.py -v

G-DVB     the registered argv passes; every lever changed (u grid, lr grid, anchor, criterion, epochs, seeds, batch sets, threads, fraction), a missing / duplicated / unknown flag, a path flag, and a MUTATED
          epr1ub.py (target, selection `any`, grid order, the criterion list) each FAIL.
G-COMMIT  passes on the deployed code dir; a tampered landed parent, a bogus commit, a missing staged file FAIL; a staged file is reported `landed: false`.
G-LIVE    green on a synthetic bank; a u term that is deleted / has an inverted mask / a wrong target / a fit path that drops u or ignores lr each FAIL.
G-SUITE   a passing suite passes; a failing, a skipped and a too-small suite FAIL.
TOKEN     mint -> verify round trip; a changed argv, an edited code file, a changed input, an edited token, another key, the wrong profile, a FAILing check each REFUSE; a FAIL demotes a stale PASS.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epr1ub_gate as G   # noqa: E402
import epr1ub as U   # noqa: E402
import test_epr1 as T   # noqa: E402
import test_epr1ub as TU   # noqa: E402

E = U.E
COMMIT = "1afc54f6fa5ee865f2966e3c0a7b3141f24ef0c1"
GIT = G.DEFAULT_GIT_DIR


def cfg(**kw) -> G.GateConfig:
    base = dict(commit=COMMIT, fraction=10, stage="run", code_dir=HERE)
    base.update(kw)
    return G.GateConfig(**base)


def dvb(c: G.GateConfig, mod=None, strict=True, argv=None):
    mod, _ = (mod, None) if mod is not None else G.load_verified_ub(HERE)
    return G.check_dvb(mod, argv or c.argv(), stage=c.stage, strict=strict)[0]


def load_gate_mut(tmp, name, edits):
    return TU.load_mut(tmp, name, edits)


class TestDVB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="epr1ub_gate_t_")
        cls.mod, _ = G.load_verified_ub(HERE)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_dvb_registered_argv_passes_strict(self):
        r = dvb(cfg(), self.mod)
        self.assertEqual((r["status"], r["reasons"]), ("PASS", []))
        self.assertEqual(r["details"]["n_configs_built"], 9)
        self.assertEqual(dvb(cfg(stage="c2"), self.mod)["status"], "PASS")

    def test_dvb_every_lever_changed_fails(self):
        for kw in ({"u_grid": "0.03,0.1,0.5"}, {"u_grid": "0.03,0.1"}, {"u_grid": "0.3,0.1,0.03"}, {"lr_grid": "0.0001,0.001,0.01"}, {"lr_grid": "0.001,0.003"}, {"anchor": "0.3"},
                   {"criterion": "pooled_oof"}, {"epochs": 11}, {"seeds": "0,1,2"}, {"seeds": "1,0"}, {"batch_sets": 128}, {"threads": 0}, {"fraction": 25}):
            r = dvb(cfg(**kw), self.mod)
            self.assertEqual(r["status"], "FAIL", kw)

    def test_dvb_flag_hygiene(self):
        base = cfg().argv()
        i = base.index("--u-grid")
        missing = base[:i] + base[i + 2:]
        self.assertTrue(any("not NAMED" in x for x in dvb(cfg(), self.mod, argv=missing)["reasons"]))
        self.assertEqual(dvb(cfg(), self.mod, argv=base + ["--anchor", "0.1"])["status"], "FAIL")           # duplicated
        self.assertEqual(dvb(cfg(), self.mod, argv=base + ["--bogus", "1"])["status"], "FAIL")              # unknown
        self.assertEqual(dvb(cfg(), self.mod, argv=base + ["--bank-dir", "C:/elsewhere/bank"])["status"], "FAIL")      # a path other than the registered one
        self.assertEqual(dvb(cfg(), self.mod, argv=base + ["--f10-dir", "C:/Users/Admin/epr1/run/f25"])["status"], "FAIL")
        self.assertEqual(dvb(cfg(), self.mod, argv=["python", "-I", "epr1.py", "run"] + base[4:])["status"], "FAIL")       # the wrong script
        self.assertEqual(dvb(cfg(), self.mod, argv=["python", "-I", "/x/epr1ub.py", "lc"] + base[4:])["status"], "FAIL")   # the wrong sub-command

    def test_dvb_mutated_module_fails(self):
        muts = {
            "unsafe_target_1": [("UNSAFE_TARGET = 0.0 ", "UNSAFE_TARGET = 1.0 ")],
            "anchor_0.3": [("UB_ANCHOR = 0.1 ", "UB_ANCHOR = 0.3 ")],
            "sel_any": [("return all(h <= SEL_BOUND for h in his)", "return any(h <= SEL_BOUND for h in his)")],
            "sel_lt": [("return all(h <= SEL_BOUND for h in his)", "return all(h < SEL_BOUND for h in his)")],
            "sel_bound": [("SEL_BOUND = 0.0 ", "SEL_BOUND = 0.01 ")],
            "grid_order": [("for lr in lr_grid for u in u_grid]", "for u in u_grid for lr in lr_grid]")],
            "criterion_list": [('CRITERIA = (CRITERION_REG, "pooled_oof")', 'CRITERIA = (CRITERION_REG, "pooled_oof", "x")')],
            "norm_changed": [('UNSAFE_NORM = "mean_over_unsafe_entries"', 'UNSAFE_NORM = "mean_over_all_entries"')],
            "tie_last": [("j = int(np.argmax(sc))", "j = len(sc) - 1 - int(np.argmax(sc[::-1]))")],
            "ident_u": [("IDENT_LR, IDENT_U = 1e-3, 0.0", "IDENT_LR, IDENT_U = 1e-3, 0.1")],
            "f10_pin": [('F10_RESULT_SHA256 = "eb5734', 'F10_RESULT_SHA256 = "ab5734')],
        }
        for i, (nm, edits) in enumerate(muts.items()):
            mod = load_gate_mut(self.tmp, f"gmut_dvb_{i}_{nm.replace('.', '_')}", edits)
            self.assertEqual(dvb(cfg(), mod)["status"], "FAIL", nm)

    def test_dvb_non_strict_allows_synthetic_paths_only_with_an_override(self):
        r = dvb(cfg(), self.mod, strict=False)
        self.assertEqual(r["status"], "PASS")


class TestCommit(unittest.TestCase):
    def test_commit_passes_on_the_deployed_code_and_reports_staged_files_as_not_landed(self):
        r = G.check_commit(commit=COMMIT, git_dir=GIT, repo_code_path=G.DEFAULT_REPO_CODE_PATH, code_dir=HERE)
        self.assertEqual((r["status"], r["reasons"]), ("PASS", []))
        self.assertEqual(r["details"]["landed_files"]["epr1.py"], U.PARENT_EPR1_SHA256)
        self.assertEqual(set(r["details"]["staged_files"]), set(G.STAGED_FILES))
        for f, v in r["details"]["staged_files"].items():
            self.assertFalse(v["landed"], f)

    def test_commit_regressions(self):
        tmp = tempfile.mkdtemp(prefix="epr1ub_commit_")
        try:
            for f in os.listdir(HERE):
                if os.path.isfile(os.path.join(HERE, f)):
                    shutil.copy(os.path.join(HERE, f), tmp)
            ok = G.check_commit(commit=COMMIT, git_dir=GIT, repo_code_path=G.DEFAULT_REPO_CODE_PATH, code_dir=tmp)
            self.assertEqual(ok["status"], "PASS")
            for f in ("epr1.py", "zero_tail.py", "test_epr1.py", "epr1_dev.py", "epr1_gate.py"):
                p = os.path.join(tmp, f)
                keep = T._rb(p)
                T._wb(p, keep + b"\n# edited\n")
                self.assertEqual(G.check_commit(commit=COMMIT, git_dir=GIT, repo_code_path=G.DEFAULT_REPO_CODE_PATH, code_dir=tmp)["status"], "FAIL", f)
                T._wb(p, keep)
            self.assertEqual(G.check_commit(commit="0" * 40, git_dir=GIT, repo_code_path=G.DEFAULT_REPO_CODE_PATH, code_dir=tmp)["status"], "FAIL")
            self.assertEqual(G.check_commit(commit="not-a-commit", git_dir=GIT, repo_code_path=G.DEFAULT_REPO_CODE_PATH, code_dir=tmp)["status"], "FAIL")
            os.remove(os.path.join(tmp, "epr1ub.py"))
            self.assertEqual(G.check_commit(commit=COMMIT, git_dir=GIT, repo_code_path=G.DEFAULT_REPO_CODE_PATH, code_dir=tmp)["status"], "FAIL")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_tampered_epr1_is_refused_when_the_ub_module_is_loaded(self):
        tmp = tempfile.mkdtemp(prefix="epr1ub_tamper2_")
        try:
            shutil.copy(os.path.join(HERE, "epr1ub.py"), tmp)
            shutil.copy(os.path.join(HERE, "epr1_gate.py"), tmp)
            T._wb(os.path.join(tmp, "epr1.py"), T._rb(os.path.join(HERE, "epr1.py")) + b"\n#x\n")
            with self.assertRaises(Exception):
                G.load_verified_ub(tmp)
        finally:
            sys.modules.pop("epr1ub__gate_dummy", None)
            shutil.rmtree(tmp, ignore_errors=True)


class TestLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="epr1ub_live_t_")
        cls.w = T.make_world(os.path.join(cls.tmp, "w"))
        cls.mod, _ = G.load_verified_ub(HERE)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def opts(self, mod=None):
        return TU.opts_ub(self.w, os.path.join(self.tmp, "o"), mod=mod)

    def test_live_green_on_a_synthetic_bank(self):
        r = G.check_live_ub(self.mod, self.opts(), {})
        self.assertEqual((r["status"], r["reasons"]), ("PASS", []))
        d = r["details"]
        for k in ("u_only_nonzero", "other_rows_and_features_exactly_zero", "zero_control_exactly_zero", "leak_control_reads_nonzero", "u0_bit_identical_to_landed", "label_invariant",
                  "safe_logit_invariant", "unsafe_logit_sensitive", "no_unsafe_entry_exactly_zero", "u_reaches_optimiser"):
            self.assertTrue(d[k], k)
        self.assertEqual(d["concordance_computed"], False)

    def test_live_regressions_each_fail_the_gate(self):
        muts = {
            "u_term_deleted": [("loss = loss + u * un", "loss = loss")],
            "mask_inverted": [("m = (~safe).to(bce.dtype)", "m = safe.to(bce.dtype)")],
            "mask_dropped": [("m = (~safe).to(bce.dtype)", "m = torch.ones_like(bce)")],
            "wrong_target": [("torch.full_like(z, UNSAFE_TARGET)", "torch.full_like(z, 1.0)")],
            "weight_ignored": [("loss = loss + u * un", "loss = loss + 1.0 * un")],
            "fit_drops_u": [("loss_fn=make_loss_fn(cfg.u)", "loss_fn=None")],
            "fit_ignores_lr": [("E.fit_row(data, train_idx, cfg, seed, W6, b6, loss_fn", "E.fit_row(data, train_idx, dataclasses.replace(cfg, lr=1e-3), seed, W6, b6, loss_fn")],
        }
        for i, (nm, edits) in enumerate(muts.items()):
            mod = TU.load_mut(self.tmp, f"gmut_live_{i}_{nm}", edits)
            r = G.check_live_ub(mod, self.opts(mod), {})
            self.assertEqual(r["status"], "FAIL", nm)


class TestSuite(unittest.TestCase):
    def run_fake(self, body: str, floor: int) -> dict:
        tmp = tempfile.mkdtemp(prefix="epr1ub_suite_t_")
        try:
            T._wt(os.path.join(tmp, "test_fake.py"), body)
            return G.check_suite(commit=COMMIT, git_dir=GIT, repo_code_path=G.DEFAULT_REPO_CODE_PATH, code_dir=tmp, python=sys.executable, suite_tests=(("test_fake.py", floor),),
                                 timeout_s=120, staged=("test_fake.py",))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_suite_pass_and_the_three_regressions(self):
        ok = "import unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertTrue(True)\nif __name__ == '__main__':\n    unittest.main()\n"
        self.assertEqual(self.run_fake(ok, 1)["status"], "PASS")
        bad = ok.replace("assertTrue(True)", "assertTrue(False)")
        self.assertEqual(self.run_fake(bad, 1)["status"], "FAIL")
        skip = ok.replace("    def test_a(self):", "    @unittest.skip('x')\n    def test_a(self):")
        self.assertEqual(self.run_fake(skip, 1)["status"], "FAIL")
        self.assertEqual(self.run_fake(ok, 5)["status"], "FAIL")                                         # fewer tests than the floor

    def test_registered_floors(self):
        self.assertEqual(G.REG_MIN_TESTS, {"test_epr1.py": 46, "test_epr1ub.py": 39})


class TestToken(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="epr1ub_tok_t_")
        cls.w = T.make_world(os.path.join(cls.tmp, "w"))
        cls.key = os.path.join(cls.tmp, "key", "launch_gate.key")
        cls.mod, _ = G.load_verified_ub(HERE)
        cls.commit_res = G.check_commit(commit=COMMIT, git_dir=GIT, repo_code_path=G.DEFAULT_REPO_CODE_PATH, code_dir=HERE)
        assert cls.commit_res["status"] == "PASS"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def fake_run(self, c: G.GateConfig, fail: str | None = None):
        o = TU.opts_ub(self.w, os.path.join(self.tmp, "o"))
        checks = {k: G._res("PASS", [], {}) for k in G.CHECKS}
        checks["G-COMMIT"] = self.commit_res
        if fail:
            checks[fail] = G._res("FAIL", [f"{fail} deliberately red"], {})
        return lambda cfg_, overrides=None, log=print: {"checks": checks, "mod_sha": "x", "opts": o, "aux": {}, "argv": cfg_.argv()}

    def mint(self, name="run", fail=None, **kw):
        c = cfg(token_path=os.path.join(self.tmp, "gate", name, G.TOKEN_NAME), key_file=self.key, **kw)
        real = G.run_checks
        G.run_checks = self.fake_run(c, fail)
        try:
            res = G.mint(c, overrides={"threads": 2}, log=lambda *a: None)
        finally:
            G.run_checks = real
        return c, res

    def verify(self, c, argv=None, code_dir=HERE, key=None, profile=G.PROFILE_SYNTHETIC):
        return G.verify_token(c.resolved_token_path(), argv or c.argv(), code_dir=code_dir, key=key or G.EG.load_key(self.key, create=False), git_dir=GIT, profile=profile)

    def test_token_round_trip_and_the_binding(self):
        c, res = self.mint("rt")
        self.assertEqual(res["verdict"], "PASS")
        ok, reasons, info = self.verify(c)
        self.assertEqual((ok, reasons), (True, []))
        b = res["token"]["binding"]
        self.assertEqual(b["epr1_sha256"], U.PARENT_EPR1_SHA256)
        self.assertEqual(b["epr1ub_sha256"], G.sha256_file(os.path.join(HERE, "epr1ub.py")))
        self.assertTrue(all(v["landed"] is False for v in b["staged_files"].values()))
        self.assertIn("deviation", b)
        for k in ("f10_result", "f10_row_seed0", "f10_row_seed1", "bank_manifest", "p2_ckpt"):
            self.assertIn(k, b["data"])

    def test_token_refusals(self):
        c, res = self.mint("ref")
        argv = c.argv()
        argv2 = list(argv)
        argv2[argv2.index("--threads") + 1] = "4"
        self.assertFalse(self.verify(c, argv=argv2)[0])
        argv3 = list(argv)
        argv3[argv3.index("--u-grid") + 1] = "0.03,0.1,0.5"
        self.assertFalse(self.verify(c, argv=argv3)[0])
        self.assertFalse(self.verify(c, profile=G.PROFILE)[0])                                          # a synthetic token never verifies for the real profile
        # edited token / another key
        p = c.resolved_token_path()
        tok = G.EG.read_json(p)
        tok["binding"]["fraction"] = 25
        G.EG.write_json(p, tok)
        self.assertFalse(self.verify(c)[0])
        c2, _ = self.mint("ref2")
        other = os.path.join(self.tmp, "key2", "k.key")
        self.assertFalse(self.verify(c2, key=G.EG.load_key(other, create=True))[0])
        # an edited code file / an edited staged file / a changed input
        tmp = tempfile.mkdtemp(prefix="epr1ub_tokcode_")
        try:
            for f in os.listdir(HERE):
                if os.path.isfile(os.path.join(HERE, f)):
                    shutil.copy(os.path.join(HERE, f), tmp)
            self.assertTrue(self.verify(c2, code_dir=tmp)[0])
            for f in ("epr1ub.py", "epr1ub_gate.py", "epr1.py", "test_epr1ub.py"):
                keep = T._rb(os.path.join(tmp, f))
                T._wb(os.path.join(tmp, f), keep + b"\n# x\n")
                self.assertFalse(self.verify(c2, code_dir=tmp)[0], f)
                T._wb(os.path.join(tmp, f), keep)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        T._wt(self.w["manifest"] + ".bak", "x")
        tok2 = G.EG.read_json(c2.resolved_token_path())
        self.assertTrue(self.verify(c2)[0])
        keep = T._rb(self.w["manifest"])
        T._wb(self.w["manifest"], keep + b" ")
        try:
            self.assertFalse(self.verify(c2)[0])
        finally:
            T._wb(self.w["manifest"], keep)
        self.assertEqual(tok2["verdict"], "PASS")

    def test_a_failing_check_mints_no_token_and_demotes_a_stale_pass(self):
        c, res = self.mint("dem")
        self.assertTrue(os.path.exists(c.resolved_token_path()))
        for chk in G.CHECKS:
            c3, res3 = self.mint("dem", fail=chk)
            self.assertEqual(res3["verdict"], "FAIL", chk)
            self.assertFalse(os.path.exists(c3.resolved_token_path()), chk)
            self.assertTrue(os.path.exists(os.path.join(os.path.dirname(c3.resolved_token_path()), G.FAIL_NAME)))
            self.assertFalse(self.verify(c3)[0])
            # restore a PASS for the next loop iteration
            self.mint("dem")

    def test_cli_refuses_a_missing_commit_and_verify_without_a_token_is_refused(self):
        out = []
        self.assertEqual(G.main(["--fraction", "10"], _out=out.append), 2)
        out2 = []
        rc = G.main(["verify", "--fraction", "10", "--token", os.path.join(self.tmp, "nope", G.TOKEN_NAME), "--key-file", self.key], _out=out2.append)
        self.assertEqual(rc, 3)
        self.assertFalse(json.loads(out2[-1])["ok"])
        out3 = []
        self.assertEqual(G.main(["argv", "--fraction", "10"], _out=out3.append), 0)
        argv = json.loads(out3[-1])["argv"]
        for flag in G.REQUIRED_EXPLICIT:
            self.assertIn(flag, argv)

    def test_the_landed_parent_gate_and_epr1_are_the_pinned_bytes(self):
        self.assertEqual(G.sha256_file(os.path.join(HERE, "epr1_gate.py")), G.PARENT_GATE_SHA256)
        self.assertEqual(G.sha256_file(os.path.join(HERE, "epr1.py")), G.PARENT_EPR1_SHA256)
        # the signature primitive is the landed one: a token signed here verifies under the landed function
        key = G.EG.load_key(self.key, create=True)
        body = {"a": 1}
        tok = {**body, "hmac_sha256": G.EG.sign(body, key)}
        self.assertTrue(G.EG.signature_ok(tok, key))


if __name__ == "__main__":
    unittest.main(verbosity=2)
