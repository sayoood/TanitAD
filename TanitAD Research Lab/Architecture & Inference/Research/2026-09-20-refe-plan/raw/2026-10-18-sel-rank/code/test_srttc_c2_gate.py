"""Tests for srttc_c2_gate.py and srttc_c2_run.py (the SR-TTC C2 launch-gate stage, MM R5) on SYNTHETIC worlds only (never the real bank).  stdlib unittest (also collected by pytest):

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I test_srttc_c2_gate.py -v

Every check has a deliberate-regression arm that MUST make the gate FAIL, each paired with a same-breath control that PASSes:
  G-DVB    --no-gate-stage-ack / the lc sub-command / epochs 11 / a wrong entry name / an entry in another dir / a wrong registered r1_result sha / an edited r1_result.json / another fraction / c* outside the
           grid / a c* whose rebuilt config sha differs / r1 written by other code / overridden pins / replacements mismatch / a C2 module with another shuffle seed, result name or row format /
           a C2 module whose R is not the verified srttc_r1 / C2 outputs that already exist / a non-registered path (strict)
  G-LIVE   C2 fits on the REAL labels / identity shuffle / shuffle across sets / shuffle in place / seed ignored / a fit that bypasses R1's loss / a detached R1 row (the frozen part) / the controls themselves
  G-BANK   a corrupted unit byte / r1_result.json fitted on another bank manifest
  G-PATH   a navtest path in the out dir
  G-SUITE  a mutated test file / an emptied one / a skipped one / the second spec failing alone / no test file at the commit
  G-COMMIT a deployed file that differs from its blob / an unlanded landed-file / a new file that differs from its landed copy / the frozen gate edited / an unknown commit
  token    argv X used with argv Y / an edited token (HMAC) / another key / the synthetic profile used for a real launch / srttc_c2.py edited after the mint / the manifest changed / r1_result.json changed /
           an R1 `run` token used for C2 / a FAIL report / a stale PASS is demoted
  ENTRY    srttc_c2_run.py REFUSES with no token (exit 3, nothing written), with --no-gate-stage-ack (exit 2), with the lc sub-command (exit 2), with a token for another argv, with srttc_c2.py edited after the mint;
           with a valid token it writes the C2 rows + result + attestation, bound to the R1 run; a second run writes nothing; the attestation refuses an edited result
  MUTATION the enforcement code itself: each marked line of srttc_c2_gate.py is mutated and a scenario that is GREEN on the original must go RED
  LOADERS  C2-GATE-SIBLING-LEAK: every load execs the hashed bytes into a module of its own: a second code dir (a different srttc_r1, identical bytes, or the first dir DELETED) runs ITS files,
           never a sys.modules / sys.path / earlier-load module; a load that raises leaves nothing behind; restoring each process-global lookup must turn the arms RED
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import srttc_c2_gate as CG   # noqa: E402
import test_srttc_r1 as T    # noqa: E402   (make_world / opts_for; a module binding, so its TestCases are NOT collected here)
import test_srttc_r1_gate as RG   # noqa: E402  (make_repo / add_summary / clone_world helpers; same reason)

G = CG.G
R = None            # srttc_r1 as imported for building the base R1 run (the real module file; the gate loads its own verified copy)
quiet = lambda *a, **k: None   # noqa: E731
rb, wb = RG.rb, RG.wb
REPO = RG.REPO_CODE_PATH
FAKE_OK = RG.FAKE_TESTS_OK
SPEC_FAKE = (("test_srttc_r1.py", (), 3), ("test_srttc_dev.py", (), 3))


def append(p, data=b"\n# x\n"):
    with open(p, "ab") as f:
        f.write(data)


def sha_f(p):
    return hashlib.sha256(rb(p)).hexdigest()


def src_of(name):
    return rb(os.path.join(HERE, name)).decode("utf-8")


ROOT = BASE_WORLD = BASE_OUT = BASE_R1 = None


def setUpModule():
    global ROOT, BASE_WORLD, BASE_OUT, BASE_R1, R
    import srttc_r1 as R_
    R = R_
    ROOT = tempfile.mkdtemp(prefix="srttc_c2gate_t_")
    BASE_WORLD = T.make_world(os.path.join(ROOT, "base"))
    RG.add_summary(BASE_WORLD)
    BASE_OUT = os.path.join(ROOT, "base_out")
    BASE_R1 = R.run_r1(T.opts_for(BASE_WORLD, BASE_OUT, epochs=10, batch_sets=256))        # the REAL R1 on the synthetic bank: r1_result.json + the two rows c* produced


def tearDownModule():
    shutil.rmtree(ROOT, ignore_errors=True)


class Fx:
    """A synthetic world + its R1 run + a deployed code dir + a scratch repo whose HEAD holds exactly the landed code files (+ fake suite files)."""

    def __init__(self, name: str, *, replace: dict | None = None, new_replace: dict | None = None, tests: dict | None = None, commit_replace: dict | None = None, omit: tuple = ()):
        self.root = os.path.join(ROOT, name)
        self.w = RG.clone_world(BASE_WORLD, os.path.join(self.root, "world"))
        self.out = os.path.join(self.root, "out_r1")
        shutil.copytree(BASE_OUT, self.out)
        self.code_dir = os.path.join(self.root, "code")
        os.makedirs(self.code_dir, exist_ok=True)
        files = {}
        for f in CG.LANDED_FILES + CG.NEW_FILES:
            data = (replace or {}).get(f, rb(os.path.join(HERE, f)))
            wb(os.path.join(self.code_dir, f), data if isinstance(data, bytes) else data.encode("utf-8"))
        for f in CG.LANDED_FILES:
            if f not in omit:
                files[f] = (commit_replace or {}).get(f, rb(os.path.join(self.code_dir, f)))
        for f in CG.NEW_FILES:
            if f in (new_replace or {}):                   # a NEW file that the commit already has (maybe different from the deployed copy)
                files[f] = new_replace[f]
        for tf, body in (tests or {"test_srttc_r1.py": FAKE_OK, "test_srttc_dev.py": FAKE_OK}).items():
            files[tf] = body.encode("utf-8") if isinstance(body, str) else body
        self.git_dir, self.commit = RG.make_repo(os.path.join(self.root, "repo"), files)
        self.key_file = os.path.join(self.root, "keys", "launch_gate.key")
        self.n = 0
        self.r1_sha = sha_f(os.path.join(self.out, "r1_result.json"))
        self.overrides = dict(bank_dir=self.w["bank"], manifest=self.w["manifest"], ranked_logs=self.w["tsv"], p2_ckpt=self.w["ckpt"], s0_marker=self.w["marker"],
                              ranked_sha256=self.w["tsv_sha"], expected_n_logs=None, min_logs_per_fold=2, reach_n_sets=64, threads=2)

    def cfg(self, **kw) -> CG.C2Config:
        self.n += 1
        base = dict(commit=self.commit, fraction=10, python=sys.executable, code_dir=self.code_dir, out_dir=self.out, threads=2, git_dir=self.git_dir,
                    repo_code_path=REPO, r1_result_sha256=self.r1_sha, token_path=os.path.join(self.root, "tok", f"t{self.n}", CG.TOKEN_NAME), key_file=self.key_file,
                    suite_specs=SPEC_FAKE)
        base.update(kw)
        return CG.C2Config(**base)

    def mint(self, overrides=None, **kw) -> dict:
        cfg = self.cfg(**kw)
        res = CG.mint(cfg, overrides=self.overrides if overrides is None else overrides, log=quiet)
        res["cfg"] = cfg
        return res

    def verify(self, res_or_cfg, argv=None, **kw):
        cfg = res_or_cfg["cfg"] if isinstance(res_or_cfg, dict) else res_or_cfg
        key = G.load_key(kw.pop("key_file", self.key_file), create=False)
        return CG.verify_token(cfg.resolved_token_path(), cfg.argv() if argv is None else argv, code_dir=kw.pop("code_dir", self.code_dir), key=key, git_dir=self.git_dir,
                               profile=kw.pop("profile", CG.PROFILE_SYNTHETIC))

    def mod_pair(self):
        files = {f: sha_f(os.path.join(self.code_dir, f)) for f in ("srttc_r1.py", "srttc_c2.py")}
        return CG.load_verified_pair(self.code_dir, files)

    def opts_aux(self, cfg=None, **kw):
        cfg = cfg or self.cfg()
        mod, c2mod, _ = self.mod_pair()
        res, opts = CG.check_dvb(mod, c2mod, cfg.argv(), code_dir=self.code_dir, r1_sha256=self.r1_sha, mod_sha=sha_f(os.path.join(self.code_dir, "srttc_r1.py")), strict=False)
        assert opts is not None, res["reasons"]
        opts = dataclasses.replace(opts, **{**self.overrides, **kw})
        bank, aux = CG.check_bank(mod, opts)
        return mod, c2mod, opts, aux, bank


def blocked(res, check=None, contains=None):
    ok = res["verdict"] == "FAIL" and res["token_path"] is None
    if check:
        ok = ok and res["checks"][check]["status"] == "FAIL"
    if contains:
        ok = ok and any(contains in r for r in res["reasons"])
    return ok


def c2_mutant_source(old: str, new: str, base: str | None = None) -> str:
    s = base if base is not None else src_of("srttc_c2.py")
    assert s.count(old) == 1, f"mutation anchor must occur exactly once: {old!r} x{s.count(old)}"
    return s.replace(old, new)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestDVB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = Fx("dvb")

    def dvb(self, fx=None, **kw):
        fx = fx or self.fx
        cfg = fx.cfg(**kw)
        mod, c2mod, _ = fx.mod_pair()
        res, opts = CG.check_dvb(mod, c2mod, cfg.argv(), code_dir=fx.code_dir, r1_sha256=kw.get("r1_result_sha256", fx.r1_sha), mod_sha=sha_f(os.path.join(fx.code_dir, "srttc_r1.py")), strict=False)
        return res, opts

    def test_registered_command_passes_and_records_c_star(self):
        res, opts = self.dvb()
        self.assertEqual(res["status"], "PASS", res["reasons"])
        self.assertIsNotNone(opts)
        self.assertEqual(res["details"]["c2_shuffle_seed"], 20261018)
        self.assertEqual(res["details"]["c_star"]["config_sha256"], BASE_R1["c_star"]["config_sha256"])
        self.assertEqual(res["details"]["r1_result_sha256"], self.fx.r1_sha)

    def test_no_gate_stage_ack_is_not_accepted(self):
        res, _ = self.dvb(extra_flags=("--no-gate-stage-ack",))
        self.assertEqual(res["status"], "FAIL")
        self.assertTrue(any("NOT accepted" in r and "no-gate-stage-ack" in r for r in res["reasons"]), res["reasons"])

    def test_lc_subcommand_and_a_wrong_entry_name_are_refused(self):
        cfg = self.fx.cfg()
        mod, c2mod, _ = self.fx.mod_pair()
        mp = sha_f(os.path.join(self.fx.code_dir, "srttc_r1.py"))
        a = cfg.argv()
        a[3] = "lc"
        r, _ = CG.check_dvb(mod, c2mod, a, code_dir=self.fx.code_dir, r1_sha256=self.fx.r1_sha, mod_sha=mp, strict=False)
        self.assertTrue(r["status"] == "FAIL" and any("sub-command" in x for x in r["reasons"]), r["reasons"])
        a = cfg.argv()
        a[2] = a[2].replace("srttc_c2_run.py", "srttc_c2.py")
        r, _ = CG.check_dvb(mod, c2mod, a, code_dir=self.fx.code_dir, r1_sha256=self.fx.r1_sha, mod_sha=mp, strict=False)
        self.assertTrue(r["status"] == "FAIL" and any("must start" in x for x in r["reasons"]), r["reasons"])
        a = cfg.argv()
        a[2] = os.path.join(self.fx.root, "elsewhere", "srttc_c2_run.py")
        r, _ = CG.check_dvb(mod, c2mod, a, code_dir=self.fx.code_dir, r1_sha256=self.fx.r1_sha, mod_sha=mp, strict=False)
        self.assertTrue(r["status"] == "FAIL" and any("not in the gated code dir" in x for x in r["reasons"]), r["reasons"])

    def test_the_frozen_declared_vs_built_check_is_reached_through_the_translation(self):
        res, _ = self.dvb(epochs=11)
        self.assertEqual(res["status"], "FAIL")
        self.assertTrue(any("epochs" in r for r in res["reasons"]), res["reasons"])
        res, _ = self.dvb(seeds="0,1,2")
        self.assertEqual(res["status"], "FAIL")
        res, _ = self.dvb(criterion="fold_mean")
        self.assertEqual(res["status"], "FAIL")
        self.assertEqual(self.dvb()[0]["status"], "PASS")                                      # control

    def test_strict_mode_refuses_a_non_registered_path(self):
        cfg = self.fx.cfg()
        mod, c2mod, _ = self.fx.mod_pair()
        res, _ = CG.check_dvb(mod, c2mod, cfg.argv(), code_dir=self.fx.code_dir, r1_sha256=self.fx.r1_sha, mod_sha=sha_f(os.path.join(self.fx.code_dir, "srttc_r1.py")), strict=True)
        self.assertEqual(res["status"], "FAIL")
        self.assertTrue(any("registered" in r for r in res["reasons"]))

    def test_c_star_must_be_the_registered_one(self):
        res, _ = self.dvb(r1_result_sha256="0" * 64)
        self.assertTrue(res["status"] == "FAIL" and any("REGISTERED" in r for r in res["reasons"]), res["reasons"])
        res, _ = self.dvb(r1_result_sha256="")
        self.assertTrue(res["status"] == "FAIL" and any("not given" in r for r in res["reasons"]), res["reasons"])
        for name, edit, needle in (
                ("fraction", lambda d: d.__setitem__("fraction_pct", 25), "is for f ="),
                ("stage", lambda d: d.__setitem__("stage", "LC"), "schema/stage"),
                ("grid", lambda d: d["c_star"].__setitem__("lr", 0.5), "registered 9"),
                ("cfgsha", lambda d: d["c_star"].__setitem__("config_sha256", "0" * 64), "rebuilt"),
                ("code", lambda d: d.__setitem__("code_sha256", "1" * 64), "written by srttc_r1.py"),
                ("pins", lambda d: d["inputs"].__setitem__("pins_overridden", ["row_tol"]), "overrode registered pins"),
                ("repl", lambda d: d["inputs"].__setitem__("replacements", {"dropped": ["x"], "added": ["y"]}), "replacements"),
                ("params", lambda d: d["params"].__setitem__("batch_sets", 64), "params")):
            fx = Fx("dvb_r1_" + name)
            p = os.path.join(fx.out, "r1_result.json")
            d = json.loads(rb(p))
            edit(d)
            wb(p, json.dumps(d, indent=1).encode("utf-8"))
            sha = sha_f(p)                                                    # the edited file IS 'registered', so only its content can refuse it
            cfg = fx.cfg(r1_result_sha256=sha)
            mod, c2mod, _ = fx.mod_pair()
            res, _o = CG.check_dvb(mod, c2mod, cfg.argv(), code_dir=fx.code_dir, r1_sha256=sha, mod_sha=sha_f(os.path.join(fx.code_dir, "srttc_r1.py")), strict=(name == "pins"))
            self.assertTrue(res["status"] == "FAIL" and any(needle in r for r in res["reasons"]), (name, res["reasons"]))
        # the control for the strict 'pins' arm: the same strict call with an R1 result that overrode nothing does NOT report overridden pins
        fx = Fx("dvb_r1_pins_ctl")
        p = os.path.join(fx.out, "r1_result.json")
        d = json.loads(rb(p))
        d["inputs"]["pins_overridden"] = []
        wb(p, json.dumps(d, indent=1).encode("utf-8"))
        sha = sha_f(p)
        mod, c2mod, _ = fx.mod_pair()
        res, _o = CG.check_dvb(mod, c2mod, fx.cfg(r1_result_sha256=sha).argv(), code_dir=fx.code_dir, r1_sha256=sha, mod_sha=sha_f(os.path.join(fx.code_dir, "srttc_r1.py")), strict=True)
        self.assertFalse(any("overrode registered pins" in r for r in res["reasons"]), res["reasons"])

    def test_a_c2_module_that_builds_another_object_fails(self):
        for name, old, new, needle in (
                ("seed", "C2_SHUFFLE_SEED = 20261018 ", "C2_SHUFFLE_SEED = 7 ", "C2_SHUFFLE_SEED"),
                ("result", 'RESULT_NAME = "r1_c2_result.json"', 'RESULT_NAME = "c2.json"', "RESULT_NAME"),
                ("rowfmt", 'ROW_FMT = "ttc_row_c2shuffle_seed{sd}.npy"', 'ROW_FMT = "c2_{sd}.npy"', "ROW_FMT")):
            fx = Fx("dvb_c2_" + name, replace={"srttc_c2.py": c2_mutant_source(old, new)})
            res, _ = self.dvb(fx)
            self.assertTrue(res["status"] == "FAIL" and any(needle in r for r in res["reasons"]), (name, res["reasons"]))
        mod, c2mod, _ = self.fx.mod_pair()
        other = types.SimpleNamespace(**{k: getattr(c2mod, k) for k in dir(c2mod) if not k.startswith("__")})
        other.R = types.SimpleNamespace()
        res, _ = CG.check_dvb(mod, other, self.fx.cfg().argv(), code_dir=self.fx.code_dir, r1_sha256=self.fx.r1_sha, mod_sha=sha_f(os.path.join(self.fx.code_dir, "srttc_r1.py")), strict=False)
        self.assertTrue(res["status"] == "FAIL" and any("not THE verified srttc_r1" in r for r in res["reasons"]), res["reasons"])

    def test_existing_c2_outputs_fail_because_the_rows_are_written_once(self):
        fx = Fx("dvb_exists")
        wb(os.path.join(fx.out, CG.REG_C2_RESULT), b"{}")
        res, _ = self.dvb(fx)
        self.assertTrue(res["status"] == "FAIL" and any("already exists" in r for r in res["reasons"]), res["reasons"])
        fx2 = Fx("dvb_exists_row")
        wb(os.path.join(fx2.out, CG.REG_C2_ROW_FMT.format(sd=1)), b"x")
        res, _ = self.dvb(fx2)
        self.assertEqual(res["status"], "FAIL")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestPathAndBank(unittest.TestCase):
    def test_pass_and_controls(self):
        fx = Fx("pb_pass")
        mod, c2mod, opts, aux, bank = fx.opts_aux()
        self.assertEqual(bank["status"], "PASS", bank["reasons"])
        self.assertEqual(CG.check_path(mod, opts, fx.cfg().argv())["status"], "PASS")
        self.assertIn("r1_result_bank_manifest_sha256", bank["details"])

    def test_a_navtest_path_in_the_launch_fails(self):
        fx = Fx("pb_navtest")
        mod, c2mod, opts, aux, bank = fx.opts_aux()
        argv = fx.cfg().argv()
        argv[2] = argv[2].replace("code", "navtest_code")
        res = CG.check_path(mod, opts, argv)
        self.assertTrue(res["status"] == "FAIL" and any("navtest" in r for r in res["reasons"]), res["reasons"])
        argv = fx.cfg().argv()
        argv[argv.index("--out-dir") + 1] = os.path.join(fx.root, "navtest_out")
        res = CG.check_path(mod, dataclasses.replace(opts, out_dir=os.path.join(fx.root, "navtest_out")), argv)
        self.assertEqual(res["status"], "FAIL")

    def test_one_corrupted_unit_byte_fails_the_bank(self):
        fx = Fx("pb_corrupt")
        u = fx.w["units"][0]
        b = bytearray(rb(u))
        b[len(b) // 2] ^= 0xFF
        wb(u, bytes(b))
        res = fx.mint()
        self.assertTrue(blocked(res, "G-BANK"), res["reasons"][:3])

    def test_r1_result_fitted_on_another_manifest_fails_the_bank(self):
        fx = Fx("pb_manifest")
        RG.edit_manifest(fx.w, lambda files: files.__setitem__("run/extra.txt", {"path": "x", "sha256": "0" * 64, "bytes": 1}))
        mod, c2mod, opts, aux, bank = fx.opts_aux()
        self.assertEqual(bank["status"], "FAIL")
        self.assertTrue(any("SAME bank" in r for r in bank["reasons"]), bank["reasons"])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = Fx("live")
        cls.mod, cls.c2mod, cls.opts, cls.aux, cls.bank = cls.fx.opts_aux()

    def live_for(self, name, c2_src=None, r1_src=None):
        """G-LIVE with a mutated srttc_c2.py / srttc_r1.py (each in its own code dir whose files ARE what the gate loads)."""
        d = os.path.join(ROOT, "live_" + name)
        os.makedirs(d, exist_ok=True)
        wb(os.path.join(d, "srttc_r1.py"), (r1_src or src_of("srttc_r1.py")).encode("utf-8"))
        wb(os.path.join(d, "srttc_c2.py"), (c2_src or src_of("srttc_c2.py")).encode("utf-8"))
        mod, c2mod, _ = CG.load_verified_pair(d, {})
        return CG.check_live(mod, c2mod, self.opts, self.aux)

    def test_pass_and_recorded_facts(self):
        res = CG.check_live(self.mod, self.c2mod, self.opts, self.aux)
        self.assertEqual(res["status"], "PASS", res["reasons"])
        d = res["details"]
        self.assertGreater(d["c2_fraction_of_labels_moved"], 0.0)
        self.assertTrue(d["c2_shuffle_keeps_label_multiset"] and d["c2_loss_received_shuffled_labels"] and d["c2_fit_moved_ttc_row"])
        self.assertEqual((d["c2_loss_calls"], d["c2_grad_same_labels_l1_must_be_0"]), (1, 0.0))
        self.assertGreater(d["c2_grad_shuffled_vs_real_l1"], 0.0)
        self.assertFalse(d["concordance_computed"])
        self.assertTrue(d["other_rows_and_features_exactly_zero"] and d["zero_control_exactly_zero"] and d["leak_control_reads_nonzero"])    # the frozen part is part of this check

    def test_a_c2_fit_on_the_real_labels_fails(self):
        res = self.live_for("reallabels", c2_src=c2_mutant_source("tr = dataclasses.replace(data, y3=y_shuf)", "tr = data"))
        self.assertEqual(res["status"], "FAIL")
        self.assertTrue(any("REAL labels" in r or "did not receive the SHUFFLED" in r for r in res["reasons"]), res["reasons"])

    def test_an_identity_shuffle_fails(self):
        res = self.live_for("identity", c2_src=c2_mutant_source("yn[i] = yn[i, rng.permutation(n_prop)]", "yn[i] = yn[i, np.arange(n_prop)]"))
        self.assertEqual(res["status"], "FAIL")
        self.assertTrue(any("moved" in r or "seed" in r for r in res["reasons"]), res["reasons"])

    def test_a_shuffle_across_sets_fails(self):
        res = self.live_for("acrosssets", c2_src=c2_mutant_source("    return torch.from_numpy(yn)", "    return torch.from_numpy(yn[rng.permutation(len(yn))])"))
        self.assertEqual(res["status"], "FAIL")
        self.assertTrue(any("multiset" in r for r in res["reasons"]), res["reasons"])

    def test_a_shuffle_in_place_fails(self):
        res = self.live_for("inplace", c2_src=c2_mutant_source("yn = y3.numpy().copy()", "yn = y3.numpy()"))
        self.assertEqual(res["status"], "FAIL")
        self.assertTrue(any("in place" in r for r in res["reasons"]), res["reasons"])

    def test_a_shuffle_that_ignores_its_seed_fails(self):
        res = self.live_for("noseed", c2_src=c2_mutant_source("rng = np.random.default_rng(seed)", "rng = np.random.default_rng(12345)"))
        self.assertEqual(res["status"], "FAIL")
        self.assertTrue(any("depend on its seed" in r for r in res["reasons"]), res["reasons"])

    def test_a_fit_that_bypasses_r1s_loss_fails(self):
        res = self.live_for("bypass", c2_src=c2_mutant_source("fit = fit or R.fit_row", "fit = fit or (lambda tr, idx, cfg, sd, W6_, b6_: (np.asarray(W6_)[R.TTC_COL].copy(), np.asarray(b6_)[R.TTC_COL:R.TTC_COL + 1].copy()))"))
        self.assertEqual(res["status"], "FAIL")
        self.assertTrue(any("called the loss 0 times" in r for r in res["reasons"]) and any("moved the TTC row by 0.0" in r for r in res["reasons"]), res["reasons"])

    def test_a_detached_r1_row_fails_in_the_frozen_part(self):
        r1 = src_of("srttc_r1.py")
        self.assertEqual(r1.count("feat = feat.detach().float()"), 1)
        res = self.live_for("r1mut", r1_src=r1.replace("feat = feat.detach().float()", "feat = feat.float()"))
        self.assertEqual(res["status"], "FAIL")
        self.assertTrue(any("features received a gradient" in r for r in res["reasons"]), res["reasons"])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestSuiteAndCommit(unittest.TestCase):
    def suite(self, fx, specs=SPEC_FAKE):
        return CG.check_suite(commit=fx.commit, git_dir=fx.git_dir, repo_code_path=REPO, python=sys.executable, specs=specs)

    def test_pass_then_each_regression_arm_fails(self):
        fx = Fx("su_ok")
        r = self.suite(fx)
        self.assertEqual(r["status"], "PASS", r["reasons"])
        self.assertEqual([x["n_tests"] for x in r["details"]["runs"]], [3, 3])
        bad = Fx("su_mut", tests={"test_srttc_r1.py": FAKE_OK, "test_srttc_dev.py": RG.FAKE_TESTS_MUTATED})
        r = self.suite(bad)
        self.assertTrue(r["status"] == "FAIL" and any("test_srttc_dev.py" in x for x in r["reasons"]), r["reasons"])     # the SECOND spec failing alone is enough
        for nm, body in (("empty", RG.FAKE_TESTS_EMPTY), ("skip", RG.FAKE_TESTS_SKIP)):
            r = self.suite(Fx("su_" + nm, tests={"test_srttc_r1.py": body, "test_srttc_dev.py": FAKE_OK}))
            self.assertEqual(r["status"], "FAIL", nm)
        r = self.suite(Fx("su_notest", tests={"test_srttc_r1.py": FAKE_OK}))
        self.assertTrue(r["status"] == "FAIL" and any("not in the export" in x for x in r["reasons"]), r["reasons"])

    def test_the_default_specs_name_r1s_suite_and_the_c2_tests(self):
        self.assertEqual(CG.SUITE_REAL, (("test_srttc_r1.py", (), 33), ("test_srttc_dev.py", ("C2Fit",), 5)))

    def test_commit_check_and_each_regression_arm(self):
        fx = Fx("co_ok")
        r = CG.check_commit(commit=fx.commit, git_dir=fx.git_dir, repo_code_path=REPO, code_dir=fx.code_dir, suite_files=("test_srttc_r1.py",), frozen_gate_sha=CG.FROZEN_GATE_SHA)
        self.assertEqual(r["status"], "PASS", r["reasons"])
        self.assertEqual({f: r["details"]["landed"][f] for f in CG.BOUND_FILES}, {"srttc_c2.py": True, "srttc_r1.py": True, "srttc_r1_gate.py": True, "srttc_c2_gate.py": False, "srttc_c2_run.py": False})
        chk = lambda f, **kw: CG.check_commit(commit=kw.pop("commit", f.commit), git_dir=f.git_dir, repo_code_path=REPO, code_dir=kw.pop("code_dir", f.code_dir), suite_files=kw.pop("suite_files", ()),    # noqa: E731
                                              frozen_gate_sha=CG.FROZEN_GATE_SHA)
        # a deployed landed file that differs from its blob
        fx2 = Fx("co_diff")
        with open(os.path.join(fx2.code_dir, "srttc_c2.py"), "ab") as f:
            f.write(b"\n# edited\n")
        r = chk(fx2)
        self.assertTrue(r["status"] == "FAIL" and any("differs from its blob" in x for x in r["reasons"]), r["reasons"])
        # a landed file that is not in the commit
        fx3 = Fx("co_unlanded", omit=("srttc_c2.py",))
        r = chk(fx3)
        self.assertTrue(r["status"] == "FAIL" and any("not landed" in x for x in r["reasons"]), r["reasons"])
        fx3a = Fx("co_notdeployed")
        os.remove(os.path.join(fx3a.code_dir, "srttc_c2.py"))
        self.assertTrue(any("not deployed" in x for x in chk(fx3a)["reasons"]))
        fx3b = Fx("co_notinrepo")
        git_dir = fx3b.git_dir
        wb(os.path.join(fx3b.code_dir, "srttc_c2.py"), (src_of("srttc_c2.py") + "\n# local only\n").encode("utf-8"))
        self.assertEqual(chk(fx3b)["status"], "FAIL")
        # a NEW file that the commit has but differs from the deployed one
        fx4 = Fx("co_newdiff", new_replace={"srttc_c2_run.py": b"# a different landed version\n"})
        r = chk(fx4)
        self.assertTrue(r["status"] == "FAIL" and any("srttc_c2_run.py is in commit" in x for x in r["reasons"]), r["reasons"])
        # a NEW file that is not deployed
        fx5 = Fx("co_newmissing")
        os.remove(os.path.join(fx5.code_dir, "srttc_c2_gate.py"))
        self.assertTrue(any("not deployed" in x for x in chk(fx5)["reasons"]))
        # the frozen gate edited after this process imported it
        fx6 = Fx("co_frozen", replace={"srttc_r1_gate.py": src_of("srttc_r1_gate.py") + "\n# edited\n"})
        r = chk(fx6)
        self.assertTrue(r["status"] == "FAIL" and any("frozen gate" in x for x in r["reasons"]), r["reasons"])
        self.assertEqual(CG.check_commit(commit="0" * 40, git_dir=git_dir, repo_code_path=REPO, code_dir=fx.code_dir)["status"], "FAIL")
        r = chk(fx, suite_files=("test_not_there.py",))
        self.assertTrue(r["status"] == "FAIL" and any("nothing to run" in x for x in r["reasons"]), r["reasons"])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestMintAndVerify(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = Fx("mv")
        cls.res = cls.fx.mint()

    def test_full_gate_passes_end_to_end_and_the_token_binds_what_it_says(self):
        res = self.res
        self.assertEqual(res["verdict"], "PASS", res["reasons"][:4])
        tok = res["token"]
        b = tok["binding"]
        self.assertEqual((tok["kind"], tok["schema"], b["profile"], b["stage"], b["fraction"], b["commit"]), ("launch-gate-token", "tanitad.launch_gate/1", CG.PROFILE_SYNTHETIC, "c2", 10, self.fx.commit))
        self.assertEqual(b["bank_manifest_sha256"], sha_f(self.fx.w["manifest"]))
        self.assertEqual(b["r1_result_sha256"], self.fx.r1_sha)
        self.assertEqual(b["argv_sha256"], G.argv_sha256(res["cfg"].argv()))
        self.assertEqual(b["c_star"]["config_sha256"], BASE_R1["c_star"]["config_sha256"])
        self.assertEqual(b["shuffle_seed"], 20261018)
        self.assertEqual(set(b["code_files"]), set(CG.BOUND_FILES))
        self.assertEqual({k: v for k, v in b["landed"].items() if not v}, {"srttc_c2_gate.py": False, "srttc_c2_run.py": False})
        self.assertEqual([c for c in CG.CHECKS if tok["checks"][c]["status"] != "PASS"], [])
        self.assertEqual(G.read_json(res["token_path"])["hmac_sha256"], tok["hmac_sha256"])
        ok, reasons, info = self.fx.verify(res)
        self.assertTrue(ok, reasons)
        self.assertEqual(info["unlanded_files"], ["srttc_c2_gate.py", "srttc_c2_run.py"])

    def test_every_check_has_a_regression_arm_that_blocks_the_token(self):
        # G-COMMIT: an edited deployed file
        fx = Fx("mv_commit")
        with open(os.path.join(fx.code_dir, "srttc_c2.py"), "ab") as f:
            f.write(b"\n# edited after landing\n")
        r = fx.mint()
        self.assertTrue(blocked(r, "G-COMMIT") and all(r["checks"][c]["status"] == "NOT-RUN" for c in ("G-DVB", "G-PATH", "G-BANK", "G-LIVE", "G-SUITE")), r["reasons"][:3])
        # G-DVB: a wrong lever
        r = Fx("mv_dvb").mint(epochs=11)
        self.assertTrue(blocked(r, "G-DVB", "epochs"), r["reasons"][:3])
        # G-LIVE: C2 on the real labels
        bad = c2_mutant_source("tr = dataclasses.replace(data, y3=y_shuf)", "tr = data")
        r = Fx("mv_live", replace={"srttc_c2.py": bad}).mint()
        self.assertTrue(blocked(r, "G-LIVE"), r["reasons"][:3])
        # G-SUITE
        r = Fx("mv_suite", tests={"test_srttc_r1.py": RG.FAKE_TESTS_MUTATED, "test_srttc_dev.py": FAKE_OK}).mint()
        self.assertTrue(blocked(r, "G-SUITE"), r["reasons"][:3])
        # the control: the same world, nothing broken
        self.assertEqual(Fx("mv_ctl").mint()["verdict"], "PASS")

    def test_token_for_argv_x_used_with_argv_y_fails_verify(self):
        cfg = self.res["cfg"]
        ok, reasons, _ = self.fx.verify(self.res, argv=dataclasses.replace(cfg, epochs=3).argv())
        self.assertFalse(ok)
        self.assertTrue(any("argv differs" in r for r in reasons), reasons)
        ok, reasons, _ = self.fx.verify(self.res, argv=dataclasses.replace(cfg, extra_flags=("--no-gate-stage-ack",)).argv())
        self.assertFalse(ok)
        self.assertTrue(self.fx.verify(self.res)[0])                                          # control

    def test_edited_token_wrong_key_and_wrong_profile_fail(self):
        fx = Fx("mv_tok")
        res = fx.mint()
        tok = G.read_json(res["token_path"])
        tok["binding"]["fraction"] = 25
        G.write_json(res["token_path"], tok)
        ok, reasons, _ = fx.verify(res)
        self.assertFalse(ok)
        self.assertTrue(any("HMAC" in r for r in reasons), reasons)
        fx2 = Fx("mv_key")
        res2 = fx2.mint()
        other = os.path.join(fx2.root, "keys2", "k")
        G.load_key(other, create=True)
        ok, reasons, _ = fx2.verify(res2, key_file=other)
        self.assertFalse(ok)
        self.assertTrue(any("HMAC" in r for r in reasons), reasons)
        ok, reasons, _ = fx2.verify(res2, profile=CG.PROFILE)
        self.assertFalse(ok)
        self.assertTrue(any("profile" in r for r in reasons), reasons)

    def test_changes_after_the_mint_fail_verify(self):
        for name, mutate, needle in (
                ("c2code", lambda fx: append(os.path.join(fx.code_dir, "srttc_c2.py")), "srttc_c2.py"),
                ("r1code", lambda fx: append(os.path.join(fx.code_dir, "srttc_r1.py")), "srttc_r1.py"),
                ("entry", lambda fx: append(os.path.join(fx.code_dir, "srttc_c2_run.py")), "srttc_c2_run.py"),
                ("gate", lambda fx: append(os.path.join(fx.code_dir, "srttc_c2_gate.py")), "srttc_c2_gate.py"),
                ("manifest", lambda fx: RG.edit_manifest(fx.w, lambda files: files.__setitem__("x", {"path": "x", "sha256": "0" * 64, "bytes": 1})), "inputs differ"),
                ("marker", lambda fx: wb(fx.w["marker"], b"changed\n"), "inputs differ"),
                ("r1result", lambda fx: wb(os.path.join(fx.out, "r1_result.json"), rb(os.path.join(fx.out, "r1_result.json")) + b" "), "inputs differ")):
            fx = Fx("mv_after_" + name)
            res = fx.mint()
            self.assertEqual(res["verdict"], "PASS", name)
            self.assertTrue(fx.verify(res)[0], name)
            mutate(fx)
            ok, reasons, _ = fx.verify(res)
            self.assertFalse(ok, name)
            self.assertTrue(any(needle in r for r in reasons), (name, reasons))

    def test_an_r1_run_token_and_an_lc_token_cannot_open_c2(self):
        fx = Fx("mv_r1tok")
        res = fx.mint()
        key = G.load_key(fx.key_file, create=False)
        tok = G.read_json(res["token_path"])
        tok["binding"]["profile"] = CG.PROFILE_SYNTHETIC      # keep the profile the verifier expects, change only the stage
        tok["binding"]["stage"] = "run"
        tok.pop("hmac_sha256")
        tok["hmac_sha256"] = G.sign(tok, key)
        G.write_json(res["token_path"], tok)
        ok, reasons, _ = fx.verify(res)
        self.assertFalse(ok)
        self.assertTrue(any("cannot open C2" in r for r in reasons), reasons)
        tok["binding"]["stage"] = "c2"
        tok["binding"]["profile"] = "srttc-r1"
        tok.pop("hmac_sha256")
        tok["hmac_sha256"] = G.sign(tok, key)
        G.write_json(res["token_path"], tok)
        ok, reasons, _ = fx.verify(res, profile="srttc-c2")
        self.assertFalse(ok)
        self.assertTrue(any("profile" in r for r in reasons), reasons)

    def test_a_fail_report_is_never_accepted_and_a_stale_pass_is_demoted(self):
        fx = Fx("mv_stale")
        ok_res = fx.mint(token_path=os.path.join(fx.root, "tok", "same", CG.TOKEN_NAME))
        self.assertEqual(ok_res["verdict"], "PASS")
        p = ok_res["token_path"]
        bad = CG.mint(fx.cfg(token_path=p, epochs=11), overrides=fx.overrides, log=quiet)
        self.assertEqual(bad["verdict"], "FAIL")
        self.assertFalse(os.path.exists(p), "a failing run demotes the stale PASS")
        self.assertTrue(os.path.exists(os.path.join(os.path.dirname(p), CG.FAIL_NAME)))
        self.assertTrue(os.path.isdir(os.path.join(os.path.dirname(p), "superseded")))
        ok, reasons, _ = CG.verify_token(os.path.join(os.path.dirname(p), CG.FAIL_NAME), ok_res["cfg"].argv(), code_dir=fx.code_dir, key=G.load_key(fx.key_file, create=False),
                                          git_dir=fx.git_dir, profile=CG.PROFILE_SYNTHETIC)
        self.assertFalse(ok)
        self.assertTrue(any("not PASS" in r for r in reasons), reasons)

    def test_missing_token_missing_key_and_missing_commit_refuse(self):
        fx = Fx("mv_missing")
        cfg = fx.cfg()
        key = G.load_key(fx.key_file, create=True)
        ok, reasons, _ = CG.verify_token(cfg.resolved_token_path(), cfg.argv(), code_dir=fx.code_dir, key=key, git_dir=fx.git_dir, profile=CG.PROFILE_SYNTHETIC)
        self.assertFalse(ok)
        self.assertTrue(any("unreadable" in r for r in reasons), reasons)
        out = []
        rc = CG.main(["--fraction", "10", "--r1-result-sha256", "0" * 64], _out=out.append)
        self.assertEqual(rc, 2)
        rc = CG.main(["--fraction", "10", "--commit", "0" * 40], _out=out.append)
        self.assertEqual(rc, 2)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestEntry(unittest.TestCase):
    """srttc_c2_run.py (-> srttc_c2_gate.run_main): the C2 fit REFUSES without a matching PASS token, and writes + attests with one."""

    def run_entry(self, fx, argv, **kw):
        out = []
        rc = CG.run_main(argv, os.path.join(fx.code_dir, "srttc_c2_run.py"), _key_file=fx.key_file, _git_dir=fx.git_dir, _profile=CG.PROFILE_SYNTHETIC,
                         _opts_overrides=fx.overrides, _out=out.append, **kw)
        return rc, "\n".join(out)

    def cli_args(self, cfg):
        return cfg.argv()[3:]

    def written(self, fx):
        return sorted(f for f in os.listdir(fx.out) if f.startswith("ttc_row_c2") or f in (CG.REG_C2_RESULT, CG.ATTEST_NAME))

    def test_refuses_without_a_token_and_writes_nothing(self):
        fx = Fx("en_notoken")
        G.load_key(fx.key_file, create=True)
        rc, msg = self.run_entry(fx, self.cli_args(fx.cfg()), _token_path=os.path.join(fx.root, "tok", "none", CG.TOKEN_NAME))
        self.assertEqual(rc, 3, msg)
        self.assertIn("REFUSED by the launch gate", msg)
        self.assertEqual(self.written(fx), [])

    def test_no_gate_stage_ack_and_the_lc_subcommand_are_refused_even_with_a_valid_token(self):
        fx = Fx("en_ack")
        res = fx.mint()
        a = self.cli_args(res["cfg"])
        rc, msg = self.run_entry(fx, a + ["--no-gate-stage-ack"], _token_path=res["token_path"])
        self.assertEqual(rc, 2, msg)
        self.assertIn("NOT accepted", msg)
        rc, msg = self.run_entry(fx, ["lc"] + a[1:], _token_path=res["token_path"])
        self.assertEqual(rc, 2, msg)
        rc, msg = self.run_entry(fx, a[1:], _token_path=res["token_path"])            # no sub-command at all
        self.assertEqual(rc, 2, msg)
        self.assertEqual(self.written(fx), [])

    def test_a_token_for_another_argv_or_edited_code_refuses_and_writes_nothing(self):
        fx = Fx("en_other")
        res = fx.mint()
        a = self.cli_args(dataclasses.replace(res["cfg"], epochs=3))
        rc, msg = self.run_entry(fx, a, _token_path=res["token_path"])
        self.assertEqual(rc, 3, msg)
        self.assertIn("argv differs", msg)
        with open(os.path.join(fx.code_dir, "srttc_c2.py"), "ab") as f:
            f.write(b"\n# edited after the mint\n")
        rc, msg = self.run_entry(fx, self.cli_args(res["cfg"]), _token_path=res["token_path"])
        self.assertEqual(rc, 3, msg)
        self.assertIn("srttc_c2.py", msg)
        self.assertEqual(self.written(fx), [])

    def test_a_valid_token_writes_the_c2_rows_result_and_attestation_bound_to_the_r1_run(self):
        fx = Fx("en_ok")
        res = fx.mint()
        rc, msg = self.run_entry(fx, self.cli_args(res["cfg"]), _token_path=res["token_path"])
        self.assertEqual(rc, 0, msg)
        self.assertEqual(self.written(fx), sorted([CG.REG_C2_RESULT, CG.ATTEST_NAME, "ttc_row_c2shuffle_seed0.npy", "ttc_row_c2shuffle_seed1.npy"]))
        c2 = json.loads(rb(os.path.join(fx.out, CG.REG_C2_RESULT)))
        self.assertEqual((c2["stage"], c2["shuffle_seed"], c2["c_star"], c2["bank_manifest_sha256"]), ("C2", 20261018, BASE_R1["c_star"], BASE_R1["inputs"]["bank_manifest_sha256"]))
        for sd in ("0", "1"):
            self.assertTrue(c2["final_rows"][sd]["differs_from_real_label_row"])
        att = json.loads(rb(os.path.join(fx.out, CG.ATTEST_NAME)))
        self.assertEqual((att["token_sha256"], att["r1_result_sha256"], att["c2_result_sha256"]), (sha_f(res["token_path"]), fx.r1_sha, sha_f(os.path.join(fx.out, CG.REG_C2_RESULT))))
        self.assertEqual(att["unlanded_files"], ["srttc_c2_gate.py", "srttc_c2_run.py"])
        key = G.load_key(fx.key_file, create=False)
        ok, reasons, _ = CG.verify_attestation(fx.out, key=key, token_path=res["token_path"], profile=CG.PROFILE_SYNTHETIC)
        self.assertTrue(ok, reasons)
        # a second run writes nothing new (the rows are written once) and the result is untouched
        before = {f: sha_f(os.path.join(fx.out, f)) for f in self.written(fx)}
        rc, msg = self.run_entry(fx, self.cli_args(res["cfg"]), _token_path=res["token_path"])
        self.assertNotEqual(rc, 0, msg)
        self.assertEqual({f: sha_f(os.path.join(fx.out, f)) for f in self.written(fx)}, before)
        # the attestation refuses an edited result, an edited row and an edited R1 result
        for name, mutate in (("result", lambda: wb(os.path.join(fx.out, CG.REG_C2_RESULT), rb(os.path.join(fx.out, CG.REG_C2_RESULT)) + b" ")),
                             ("row", lambda: wb(os.path.join(fx.out, "ttc_row_c2shuffle_seed1.npy"), rb(os.path.join(fx.out, "ttc_row_c2shuffle_seed1.npy"))[:-1] + b"\x00"))):
            fx2 = Fx("en_att_" + name)
            res2 = fx2.mint()
            self.assertEqual(self.run_entry(fx2, self.cli_args(res2["cfg"]), _token_path=res2["token_path"])[0], 0)
            k2 = G.load_key(fx2.key_file, create=False)
            self.assertTrue(CG.verify_attestation(fx2.out, key=k2, token_path=res2["token_path"], profile=CG.PROFILE_SYNTHETIC)[0])
            if name == "result":
                wb(os.path.join(fx2.out, CG.REG_C2_RESULT), rb(os.path.join(fx2.out, CG.REG_C2_RESULT)) + b" ")
            else:
                p = os.path.join(fx2.out, "ttc_row_c2shuffle_seed1.npy")
                wb(p, rb(p)[:-1] + bytes([rb(p)[-1] ^ 1]))
            ok, reasons, _ = CG.verify_attestation(fx2.out, key=k2, token_path=res2["token_path"], profile=CG.PROFILE_SYNTHETIC)
            self.assertFalse(ok, name)
        fx3 = Fx("en_att_nofile")
        self.assertFalse(CG.verify_attestation(fx3.out, key=G.load_key(fx3.key_file, create=True))[0])

    def test_the_real_command_line_refuses_without_a_token_as_a_subprocess(self):
        fx = Fx("en_cli")
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "TANITAD_GATE_KEY_FILE": fx.key_file})
        G.load_key(fx.key_file, create=True)
        cp = subprocess.run([sys.executable, "-I", os.path.join(fx.code_dir, "srttc_c2_run.py"), *self.cli_args(fx.cfg())], capture_output=True, env=env, timeout=300)
        self.assertEqual(cp.returncode, 3, (cp.stdout + cp.stderr).decode("utf-8", "replace")[-500:])
        self.assertIn("REFUSED by the launch gate", cp.stdout.decode("utf-8", "replace"))
        self.assertEqual(self.written(fx), [])
        cp = subprocess.run([sys.executable, "-I", os.path.join(fx.code_dir, "srttc_c2_run.py"), *self.cli_args(fx.cfg()), "--no-gate-stage-ack"], capture_output=True, env=env, timeout=300)
        self.assertEqual(cp.returncode, 2)

    def test_the_real_cli_cannot_reach_the_synthetic_hooks(self):
        import inspect
        sig = inspect.signature(CG.run_main)
        hooks = [n for n in sig.parameters if n.startswith("_") and n != "_out"]
        self.assertTrue(hooks and all(sig.parameters[n].kind == inspect.Parameter.KEYWORD_ONLY for n in hooks))
        ap = CG.make_parser()
        known = {s for a in ap._actions for s in a.option_strings}
        self.assertFalse({h.replace("_", "-") for h in hooks} & known)
        src = rb(os.path.join(HERE, "srttc_c2_run.py")).decode("utf-8")
        self.assertNotIn("_opts_overrides", src)
        self.assertNotIn("_profile", src)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
_N = [0]


class TestMutationOfTheEnforcement(unittest.TestCase):
    """Each marked line of srttc_c2_gate.py is mutated; a scenario that is GREEN on the original must go RED on the mutant.  A check that no test can turn red is not a check."""

    GATE_SRC = rb(os.path.join(HERE, "srttc_c2_gate.py")).decode("utf-8")

    def load_mutant(self, old, new, name):
        """`old`/`new` are strings, or equally long tuples of strings (several lines mutated together: a check that has redundant layers is only 'caught' when every layer is gone)."""
        olds, news = (old, new) if isinstance(old, tuple) else ((old,), (new,))
        src = self.GATE_SRC
        for o, n in zip(olds, news):
            self.assertEqual(self.GATE_SRC.count(o), 1, f"anchor {o!r} x{self.GATE_SRC.count(o)}")
            src = src.replace(o, n)
        modname = f"srttc_c2_gate_mutant_{name}"
        m = types.ModuleType(modname)
        m.__file__ = os.path.join(HERE, "srttc_c2_gate.py")
        sys.modules[modname] = m
        exec(compile(src, m.__file__, "exec", dont_inherit=True), m.__dict__)
        return m

    def world(self, kind):
        """A minted world with the scenario's damage applied.  -> (fx, argv, token path)."""
        _N[0] += 1
        fx = Fx(f"mu_{kind}_{_N[0]}")
        res = fx.mint()
        cfg = res["cfg"]
        a = cfg.argv()[3:]
        tok = res["token_path"]
        if kind == "notoken":
            tok = os.path.join(fx.root, "tok", "none", CG.TOKEN_NAME)
        elif kind == "ack":
            a = a + ["--no-gate-stage-ack"]
        elif kind == "otherargv":
            a = dataclasses.replace(cfg, epochs=3).argv()[3:]
        elif kind == "editedtoken":
            t = G.read_json(tok)
            t["binding"]["fraction"] = 25
            G.write_json(tok, t)
        elif kind == "editedcode":
            with open(os.path.join(fx.code_dir, "srttc_c2.py"), "ab") as f:
                f.write(b"\n# e\n")
        elif kind == "r1changed":
            wb(os.path.join(fx.out, "r1_result.json"), rb(os.path.join(fx.out, "r1_result.json")) + b" ")
        elif kind == "stage":
            k = G.load_key(fx.key_file, create=False)
            t = G.read_json(tok)
            t["binding"]["stage"] = "run"
            t.pop("hmac_sha256")
            t["hmac_sha256"] = G.sign(t, k)
            G.write_json(tok, t)
        return fx, a, tok

    def entry_refuses(self, gate_mod, world):
        """-> True iff the enforcing entry of `gate_mod` refuses in this world (exit 2/3, no C2 result written)."""
        fx, a, tok = world
        out = []
        try:
            rc = gate_mod.run_main(a, os.path.join(fx.code_dir, "srttc_c2_run.py"), _token_path=tok, _key_file=fx.key_file, _git_dir=fx.git_dir, _profile=CG.PROFILE_SYNTHETIC,
                                   _opts_overrides=fx.overrides, _out=out.append)
        except Exception:                                                              # noqa: BLE001
            return False
        self.last_msg = " | ".join(out)
        return rc in (2, 3) and not os.path.exists(os.path.join(fx.out, CG.REG_C2_RESULT))

    CASES = (
        ("notoken", "verify_skipped", "    if not ok:\n        _out(\"REFUSED by the launch gate", "    if False:\n        _out(\"REFUSED by the launch gate"),
        ("ack", "ack_allowed", "    if any(str(t).split(\"=\")[0] == \"--no-gate-stage-ack\" for t in argv):\n        _out(\"REFUSED: --no-gate-stage-ack is NOT accepted (Master Mind R5",
         "    if False:\n        _out(\"REFUSED: --no-gate-stage-ack is NOT accepted (Master Mind R5"),
        ("otherargv", "argv_any", "    if not isinstance(token_argv, list) or len(token_argv) != len(argv):\n        return False", "    return True"),
        ("editedtoken", "hmac_skipped", "    if not G.signature_ok(tok, key):", "    if False:"),
        ("stage", "stage_skipped", "    if b.get(\"stage\") != STAGE:", "    if False:"),
        # the deployed code's sha256 is re-measured in FOUR layers (per file, tree digest, the trainer fields, the blob at the commit): the mutant removes all four
        ("editedcode", "code_not_remeasured",
         ("        if now[f] is None or now[f] != code.get(f):", "    if len(reasons) == n_before and G.tree_digest(now) != b.get(\"tree_sha256\"):",
          "        if b.get(key_) != now.get(f):", "                        if blob is None or sha256_bytes(blob) != now.get(f):"),
         ("        if False:", "    if False:", "        if False:", "                        if False:")),
        ("r1changed", "inputs_not_remeasured", "    if d_bad:", "    if False:"),
    )

    def test_every_enforcement_marker_is_caught_and_the_original_refuses_each_scenario(self):
        for kind, name, old, new in self.CASES:
            w = self.world(kind)
            self.assertTrue(self.entry_refuses(CG, w), f"{name}: the original must refuse the '{kind}' scenario (the GREEN side)")
            m = self.load_mutant(old, new, name)
            if kind == "ack":
                # --no-gate-stage-ack is refused by THREE layers (the explicit message, the usage parser, and the token's argv).  Removing the explicit one must change the message, and the flag must STILL not reach the fit.
                self.assertIn("NOT accepted", self.last_msg)
                self.assertTrue(self.entry_refuses(m, w), "defence in depth: the other layers still refuse the flag")
                self.assertNotIn("NOT accepted", self.last_msg, "the mutant lost the explicit refusal (its message changed)")
                continue
            self.assertFalse(self.entry_refuses(m, w), f"{name}: the mutant must be caught (it still refused the '{kind}' scenario)")

    def test_gate_side_markers_are_caught(self):
        fx = Fx("mu_gate_side")
        mod, c2mod, _ = fx.mod_pair()
        mp = sha_f(os.path.join(fx.code_dir, "srttc_r1.py"))
        # G-DVB: the registered-sha comparison of r1_result.json
        m = self.load_mutant("                if r1_sha256 and sha != r1_sha256:", "                if False:", "dvb_sha")
        args = dict(code_dir=fx.code_dir, r1_sha256="0" * 64, mod_sha=mp, strict=False)
        self.assertEqual(CG.check_dvb(mod, c2mod, fx.cfg().argv(), **args)[0]["status"], "FAIL")
        self.assertEqual(m.check_dvb(mod, c2mod, fx.cfg().argv(), **args)[0]["status"], "PASS", "the mutant accepts an unregistered c*")
        # G-DVB: --no-gate-stage-ack
        m = self.load_mutant("        if any(str(t).split(\"=\")[0] == \"--no-gate-stage-ack\" for t in argv):\n            reasons.append(\"--no-gate-stage-ack is NOT accepted",
                             "        if False:\n            reasons.append(\"--no-gate-stage-ack is NOT accepted", "dvb_ack")
        a = fx.cfg(extra_flags=("--no-gate-stage-ack",)).argv()
        ok_args = dict(code_dir=fx.code_dir, r1_sha256=fx.r1_sha, mod_sha=mp, strict=False)
        self.assertTrue(any("NOT accepted" in x for x in CG.check_dvb(mod, c2mod, a, **ok_args)[0]["reasons"]))
        self.assertFalse(any("NOT accepted" in x for x in m.check_dvb(mod, c2mod, a, **ok_args)[0]["reasons"]))
        # G-LIVE: the 'loss received the shuffled labels' decision
        m = self.load_mutant("            if not got_shuffled:\n                reasons.append(\"the loss did not receive the SHUFFLED labels", "            if False:\n                reasons.append(\"the loss did not receive the SHUFFLED labels", "live_recv")
        _mod, _c2, opts, aux, _b = fx.opts_aux()
        bad_dir = os.path.join(ROOT, "mu_live_bad")
        os.makedirs(bad_dir, exist_ok=True)
        wb(os.path.join(bad_dir, "srttc_r1.py"), rb(os.path.join(HERE, "srttc_r1.py")))
        wb(os.path.join(bad_dir, "srttc_c2.py"), c2_mutant_source("tr = dataclasses.replace(data, y3=y_shuf)", "tr = data").encode("utf-8"))
        modb, c2b, _ = CG.load_verified_pair(bad_dir, {})
        orig = CG.check_live(modb, c2b, opts, aux)
        mut = m.check_live(modb, c2b, opts, aux)
        self.assertTrue(any("REAL labels" in r for r in orig["reasons"]), orig["reasons"])
        self.assertFalse(any("did not receive the SHUFFLED" in r for r in mut["reasons"]), "the mutated decision no longer fires")
        self.assertTrue(any("REAL labels" in r for r in mut["reasons"]), "the 'received the REAL labels' arm still catches the real-label fit")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class _FakeFitData:
    """The two fields `srttc_c2.fit_c2_rows` reads (`y3`, `n_sets`) and `dataclasses.replace`s: enough to run the module's own function against a stand-in `srttc_r1.fit_row` that returns a sentinel."""
    y3: torch.Tensor
    n_sets: int


class TestEachLoadUsesExactlyTheVerifiedFiles(unittest.TestCase):
    """C2-GATE-SIBLING-LEAK (BACKLOG, fix before any reuse of the C2 chain).  The loaders registered the module in `sys.modules` under the sha256 of its bytes and served it to every later caller with the
    same bytes, and srttc_c2.py's `import srttc_r1 as R` was answered by a process-global `sys.modules['srttc_r1']`.  A second code dir therefore ran the FIRST dir's module (its `__file__`, its state): once that
    dir was gone `run_c2` raised FileNotFoundError.  Every arm asserts on a SENTINEL value the module under test returns, not on a path string, and each arm is paired with the mutant that restores the lookup."""

    GATE_SRC = rb(os.path.join(HERE, "srttc_c2_gate.py")).decode("utf-8")
    FILES = ("srttc_r1_gate.py", "srttc_r1.py", "srttc_c2.py")
    PREFIXES = ("srttc_r1__c2g_", "srttc_c2__c2g_", "srttc_r1_gate__c2dep_")

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="c2g_sibling_")

    @classmethod
    def tearDownClass(cls):
        for k in [k for k in sys.modules if k.startswith("srttc_c2_gate_sibling_mutant_")]:
            sys.modules.pop(k, None)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # -- helpers --------------------------------------------------------------------------------------------------------------------------------------------------
    def mk(self, name, tag, uid, **boom):
        """A code dir with the real three files.  srttc_r1.py gets `SENTINEL = tag` and a `fit_row` that returns ('fit_row from', tag) appended (its sha256 therefore depends on `tag` and `uid`: the same tag and
        uid is the SAME bytes in another dir).  A file named in `boom` gets a `raise RuntimeError(msg)` appended."""
        d = os.path.join(self.tmp, f"{name}_{uid}")
        os.makedirs(d)
        for f in self.FILES:
            data = rb(os.path.join(HERE, f))
            if f == "srttc_r1.py":
                data += (f"\n\nSENTINEL = {tag!r}\n\n\ndef fit_row(*a, **k):\n    return ('fit_row from', SENTINEL)\n\n# uid {uid}\n").encode("utf-8")
            if f in boom:
                data += ("\nraise RuntimeError(%r)\n" % boom[f]).encode("utf-8")
            wb(os.path.join(d, f), data)
        return d

    def snap(self):
        return {k for k in sys.modules if k.startswith(self.PREFIXES)}

    def run_fit(self, c2):
        """`srttc_c2.fit_c2_rows` as the module ITSELF runs it: its default `fit` is `R.fit_row`, so the row it returns is whatever the srttc_r1 it was bound to returns."""
        rows, _ = c2.fit_c2_rows(_FakeFitData(torch.zeros(3, 8), 3), None, (0,), None, None)
        return rows["0"]

    def mutant(self, marker, new, name):
        """srttc_c2_gate.py with the ONE line that carries `marker` replaced by `new` (the lines of `new` are indented like the original)."""
        lines = self.GATE_SRC.split("\n")
        idx = [i for i, ln in enumerate(lines) if marker in ln]
        self.assertEqual(len(idx), 1, (marker, len(idx)))
        ln = lines[idx[0]]
        indent = ln[:len(ln) - len(ln.lstrip())]
        lines[idx[0]] = "\n".join(indent + x for x in new.split("\n"))
        modname = f"srttc_c2_gate_sibling_mutant_{name}"
        m = types.ModuleType(modname)
        m.__file__ = os.path.join(HERE, "srttc_c2_gate.py")
        sys.modules[modname] = m
        exec(compile("\n".join(lines), m.__file__, "exec", dont_inherit=True), m.__dict__)
        return m

    # -- the arms: each takes a gate module and raises AssertionError when that gate hands a load the wrong module ---------------------------------------------------
    def arm_poison(self, gm):
        """Code dirs A, B (a DIFFERENT srttc_r1) and A2 (the SAME bytes as A, in another dir) are loaded in a process that already has some other `srttc_r1` registered under that name.  Each load must run ITS
        dir's srttc_r1: asserted by the sentinel its `fit_row` returns through `srttc_c2.fit_c2_rows`."""
        uid = os.urandom(6).hex()
        dirs = [self.mk(f"pa_{n}", tag, uid) for n, tag in (("A", "A"), ("B", "B"), ("A2", "A"))]
        decoy = types.ModuleType("srttc_r1")
        decoy.SENTINEL = "GLOBAL"
        decoy.fit_row = lambda *a, **k: ("fit_row from", "GLOBAL")
        saved, path0 = sys.modules.get("srttc_r1"), list(sys.path)
        sys.modules["srttc_r1"] = decoy                                           # a process that already imported another srttc_r1 (every test run does: setUpModule)
        try:
            loads = [gm.load_verified_pair(d, {}) for d in dirs]
            untouched = sys.modules.get("srttc_r1") is decoy
            path_same = sys.path == path0
        finally:
            if saved is None:
                sys.modules.pop("srttc_r1", None)
            else:
                sys.modules["srttc_r1"] = saved
            sys.path[:] = path0
        for tag, (r1, c2, _shas) in zip(("A", "B", "A"), loads):
            self.assertIs(c2.R, r1, f"{tag}: srttc_c2's R is not the srttc_r1 that was loaded with it")
            self.assertEqual(r1.SENTINEL, tag, f"{tag}: the srttc_r1 module of this load carries another dir's sentinel")
            self.assertEqual(self.run_fit(c2), ("fit_row from", tag), f"{tag}: the C2 fit ran another module's srttc_r1 code")
        self.assertIsNot(loads[2][0], loads[0][0], "identical bytes in a second dir were served the first dir's srttc_r1 module")
        self.assertIsNot(loads[2][1], loads[0][1], "identical bytes in a second dir were served the first dir's srttc_c2 module")
        self.assertTrue(untouched, "a load replaced or removed sys.modules['srttc_r1'] (process-global state)")
        self.assertTrue(path_same, "a load left an entry on sys.path")

    def arm_gone(self, gm):
        """A is loaded and then DELETED; B (identical bytes) is loaded next.  B must not raise, must not carry A's state, and the file its module points at (what `run_c2` re-reads at srttc_c2.py:110) must
        be readable and be the bytes that were verified."""
        uid = os.urandom(6).hex()
        dA = self.mk("pb_A", "same", uid)
        r1A, _c2A, _ = gm.load_verified_pair(dA, {})
        r1A.LEFT_BY_A = "state of A"
        shutil.rmtree(dA)
        dB = self.mk("pb_B", "same", uid)
        r1B, c2B, shas = gm.load_verified_pair(dB, {})
        self.assertFalse(hasattr(r1B, "LEFT_BY_A"), "B was handed A's module object (its state is visible)")
        self.assertIsNot(r1B, r1A)
        with open(os.path.abspath(c2B.R.__file__), "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), shas["srttc_r1.py"], "the file the loaded module points at is not the file whose sha256 was verified")

    def arm_failed_load(self, gm):
        """A load whose exec raises leaves no module in sys.modules, restores sys.path and never touches sys.modules['srttc_r1']; the same bytes raise again on the next call."""
        uid = os.urandom(6).hex()
        cases = (("frozen gate", {"srttc_r1_gate.py": "boom frozen"}, lambda d: gm._load_frozen_gate(d)),
                 ("exec_verified_module", {"srttc_r1.py": "boom evm"}, lambda d: gm.exec_verified_module(d, "srttc_r1.py", "srttc_r1")),
                 ("pair, srttc_c2.py raises", {"srttc_c2.py": "boom c2"}, lambda d: gm.load_verified_pair(d, {})),
                 ("pair, srttc_r1.py raises", {"srttc_r1.py": "boom r1"}, lambda d: gm.load_verified_pair(d, {})))
        for i, (label, broken, call) in enumerate(cases):
            d = self.mk(f"pf{i}", "boom", uid, **broken)
            before, r1_before, path0 = self.snap(), sys.modules.get("srttc_r1"), list(sys.path)
            for attempt in (1, 2):
                with self.assertRaises(RuntimeError) as cm:
                    call(d)
                self.assertIn(next(iter(broken.values())), str(cm.exception), f"{label}, attempt {attempt}")
            self.assertEqual(self.snap() - before, set(), f"{label}: a FAILED load left modules registered in sys.modules")
            self.assertIs(sys.modules.get("srttc_r1"), r1_before, f"{label}: sys.modules['srttc_r1'] changed")
            self.assertEqual(sys.path, path0, f"{label}: sys.path changed")

    # -- the tests ------------------------------------------------------------------------------------------------------------------------------------------------
    def test_a_second_code_dir_runs_its_own_srttc_r1_poison_arm(self):
        self.arm_poison(CG)

    def test_a_deleted_first_code_dir_is_never_touched_by_a_later_load(self):
        self.arm_gone(CG)

    def test_a_failed_load_leaves_nothing_behind_and_fails_again(self):
        self.arm_failed_load(CG)

    def test_restoring_each_process_global_lookup_turns_the_arms_red(self):
        reuse = self.mutant("MUTATION-TARGET-SIBLING-NAME", "name = stem\nif name in sys.modules:\n    return sys.modules[name]", "reuse")     # the sha-keyed sys.modules lookup of the old loaders
        nobind = self.mutant("MUTATION-TARGET-SIBLING-BIND", "if False:", "nobind")                                                        # `import srttc_r1` answered by sys.modules / sys.path again
        noguard = self.mutant("MUTATION-TARGET-POISON-GUARD", "pass", "noguard")                                                           # a failed load stays registered
        for arm in (self.arm_poison, self.arm_gone, self.arm_failed_load):
            arm(CG)                                                                    # the GREEN side, in the same breath
        for name, gm, arms in (("sha-keyed sys.modules lookup", reuse, (self.arm_poison, self.arm_gone)), ("global srttc_r1 binding", nobind, (self.arm_poison,)),
                               ("no poison guard", noguard, (self.arm_failed_load,))):
            for arm in arms:
                with self.assertRaises(AssertionError, msg=f"{name}: the mutant must turn {arm.__name__} RED"):
                    arm(gm)

    def test_the_sibling_import_is_answered_by_the_verified_module_only(self):
        import builtins
        process_import = builtins.__import__
        sentinel = types.ModuleType("verified_srttc_r1")
        ns = CG._bound_builtins({"srttc_r1": sentinel})
        imp = ns["__import__"]
        self.assertIs(imp("srttc_r1"), sentinel)
        self.assertIs(imp("srttc_r1", None, None, ("Cfg",), 0), sentinel)
        self.assertIs(imp("json"), json)                                                # everything else goes to the normal machinery
        with self.assertRaises(ImportError):
            imp("srttc_r1.sub")                                                         # a submodule would reach past the verified module
        self.assertIs(builtins.__import__, process_import)                              # the process-wide __import__ is untouched
        self.assertIsNot(ns["__import__"], process_import)

    def test_every_load_is_a_module_of_its_own_with_its_own_file(self):
        uid = os.urandom(6).hex()
        d1, d2 = self.mk("pe1", "same", uid), self.mk("pe2", "same", uid)
        (a1, b1, _), (a2, b2, _) = CG.load_verified_pair(d1, {}), CG.load_verified_pair(d2, {})
        self.assertEqual((os.path.dirname(a1.__file__), os.path.dirname(a2.__file__)), (d1, d2))
        self.assertEqual((os.path.dirname(b1.__file__), os.path.dirname(b2.__file__)), (d1, d2))
        self.assertEqual(len({a1.__name__, a2.__name__, b1.__name__, b2.__name__}), 4)
        g1, g2 = CG._load_frozen_gate(d1)[0], CG._load_frozen_gate(d2)[0]
        self.assertIsNot(g1, g2)
        self.assertIsNot(g1, CG.G)


if __name__ == "__main__":
    unittest.main(verbosity=2)
