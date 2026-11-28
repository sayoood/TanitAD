"""Tests for srttc_r1_gate.py and launch_r1.ps1 on SYNTHETIC banks only (never the real bank).  stdlib unittest (also collected by pytest):

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I test_srttc_r1_gate.py -v

Every check has a deliberate-regression arm that MUST make the gate FAIL, each paired with a same-breath control that PASSes:
  G-DVB    epochs 11 / batch-sets 128 / criterion fold_mean / seeds 0,1,2 / both-rule rising / fraction 7 / a lever not named / a duplicated flag / an unknown flag / a 4th anchor (grid of 12) /
           a grid() that builds 10 configs under a registered argv (the BUILT object is inspected, not the argv text) / Adam betas changed / a non-registered path
  G-LIVE   TTC row detached / pairwise term deleted / optimiser step cut / features not detached / a loss that leaks into another row / a wrong P2 checkpoint / a corrupted first unit
  G-BANK   one corrupted unit byte / no S0_DONE / S0_FAILED beside it / 3 of 40 logs without a token (7.5 %; 2 of 40 = 5 % passes) / unit not in the manifest / manifest unit not on disk /
           a manifest R1's own reader rejects / a selected log with no set in the bank / S0_SUMMARY.json edited after the manifest
  G-PATH   a navtest bank path / a dev-gate out-dir / a disabled guard / a forbidden-token list that lost "navtest"
  G-SUITE  a mutated test file / an emptied test file (below the floor) / a skipped test / no test file at the commit
  G-COMMIT a deployed file that differs from its blob / an unknown commit / the gate file edited
  token    argv X used with argv Y / an edited token (HMAC) / another key / srttc_r1.py edited after the mint / the manifest changed / the S0 marker changed / the synthetic profile used for a real launch /
           a FAIL report / a stale PASS is demoted by a failing run / a run-stage token used for the lc stage
  launcher launch_r1.ps1 refuses with no S0_DONE, with no token, with a gate that refuses (exit 3, ok=false, ok=true with exit 3), and builds the held WMI command line from the gate's argv
  END TO END  the full gate PASSes on a synthetic world (fake suite) and again with the REAL test_srttc_r1.py as G-SUITE on a clean `git archive` export.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import srttc_r1_gate as G   # noqa: E402
import test_srttc_r1 as T   # noqa: E402   (make_world: the synthetic bank builder; a module binding, so its TestCases are NOT collected here)

REPO_CODE_PATH = "arm/code"
FAKE_TESTS_OK = ("import unittest\n\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertTrue(True)\n\n    def test_b(self):\n        self.assertEqual(1, 1)\n\n"
                 "    def test_c(self):\n        self.assertIn(1, [1])\n\n\nif __name__ == '__main__':\n    unittest.main()\n")
FAKE_TESTS_MUTATED = FAKE_TESTS_OK.replace("self.assertEqual(1, 1)", "self.assertEqual(1, 2)")
FAKE_TESTS_EMPTY = "import unittest\n\nif __name__ == '__main__':\n    unittest.main()\n"
FAKE_TESTS_SKIP = FAKE_TESTS_OK.replace("    def test_c(self):", "    @unittest.skip('x')\n    def test_c(self):")
quiet = lambda *a, **k: None   # noqa: E731


def types_ns(**kw):
    import types
    return types.SimpleNamespace(**kw)


def rb(p):
    with open(p, "rb") as f:
        return f.read()


def wb(p, data):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)


def git(root, *args):
    r = subprocess.run([G._git_exe(), "-C", root, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false", "-c", "commit.gpgsign=false", *args], capture_output=True, env=G._clean_env())
    if r.returncode != 0:
        raise RuntimeError(f"git {args}: {r.stderr.decode('utf-8', 'replace')}")
    return r.stdout.decode().strip()


def make_repo(root, files: dict) -> tuple:
    """A SCRATCH repo (never the programme's): -> (git_dir, commit).  files: name -> bytes, committed under REPO_CODE_PATH."""
    os.makedirs(root, exist_ok=True)
    git(root, "init", "-q")
    for name, data in files.items():
        wb(os.path.join(root, REPO_CODE_PATH, name), data)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "landing")
    return os.path.join(root, ".git"), git(root, "rev-parse", "HEAD")


def add_summary(w: dict, n_empty: int = 0, n_logs: int = 40, tamper_after: bool = False) -> None:
    """S0_SUMMARY.json next to S0_DONE, listed in the manifest (as srttc_s0_post.py does)."""
    p = os.path.join(os.path.dirname(w["marker"]), "S0_SUMMARY.json")
    doc = {"n_logs": n_logs, "n_logs_without_any_token": n_empty, "logs_without_any_token": [f"nolog{i}" for i in range(n_empty)], "PARTIAL": False,
           "replacement_rule_exceeded": bool(n_empty > 0.05 * n_logs)}
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
    edit_manifest(w, lambda files: files.__setitem__("run/S0_SUMMARY.json", {"path": p.replace("\\", "/"), "sha256": hashlib.sha256(rb(p)).hexdigest(), "bytes": os.path.getsize(p)}))
    if tamper_after:
        doc["n_logs_without_any_token"] = 0
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump({**doc, "PARTIAL": True}, f, indent=1)


def edit_manifest(w: dict, fn) -> None:
    with open(w["manifest"], encoding="utf-8") as f:
        doc = json.load(f)
    fn(doc["files"])
    with open(w["manifest"], "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)


def clone_world(w: dict, dst: str) -> dict:
    src = w["root"]
    shutil.copytree(src, dst)

    c = dict(w)
    for k in ("root", "bank", "manifest", "tsv", "ckpt", "marker"):
        c[k] = os.path.normpath(os.path.join(dst, os.path.relpath(w[k], src)))
    c["units"] = [os.path.join(c["bank"], os.path.basename(u)) for u in w["units"]]
    return c


class Fixture:
    """A synthetic world + a deployed code dir + a scratch repo whose HEAD holds exactly those code files (+ the chosen test file)."""

    def __init__(self, root: str, world: dict, tests: str = FAKE_TESTS_OK, test_bytes: bytes | None = None):
        self.root = root
        self.w = world
        self.code_dir = os.path.join(root, "code")
        os.makedirs(self.code_dir, exist_ok=True)
        for f in G.BOUND_CODE_FILES:
            shutil.copyfile(os.path.join(HERE, f), os.path.join(self.code_dir, f))
        files = {f: rb(os.path.join(HERE, f)) for f in G.BOUND_CODE_FILES}
        files[G.SUITE_TEST_FILE] = test_bytes if test_bytes is not None else tests.encode()
        files[G.GATE_TEST_FILE] = rb(os.path.join(HERE, G.GATE_TEST_FILE))
        self.git_dir, self.commit = make_repo(os.path.join(root, "repo"), files)
        self.key_file = os.path.join(root, "keys", "launch_gate.key")
        self.n = 0
        self.overrides = dict(bank_dir=world["bank"], manifest=world["manifest"], ranked_logs=world["tsv"], p2_ckpt=world["ckpt"], s0_marker=world["marker"],
                              ranked_sha256=world["tsv_sha"], expected_n_logs=None, min_logs_per_fold=2, reach_n_sets=64, threads=2)

    def cfg(self, **kw) -> G.GateConfig:
        self.n += 1
        base = dict(commit=self.commit, fraction=10, stage="run", python=sys.executable, code_dir=self.code_dir, out_dir=os.path.join(self.root, "out_r1"), threads=2,
                    git_dir=self.git_dir, repo_code_path=REPO_CODE_PATH, token_path=os.path.join(self.root, "tok", f"t{self.n}", G.TOKEN_NAME), key_file=self.key_file,
                    suite_min_tests=3)
        base.update(kw)
        return G.GateConfig(**base)

    def mint(self, overrides=None, **kw) -> dict:
        cfg = self.cfg(**kw)
        res = G.mint(cfg, overrides=self.overrides if overrides is None else overrides, log=quiet)
        res["cfg"] = cfg
        return res

    def verify(self, res_or_cfg, argv=None, **kw):
        cfg = res_or_cfg["cfg"] if isinstance(res_or_cfg, dict) else res_or_cfg
        key = G.load_key(kw.pop("key_file", self.key_file), create=False)
        return G.verify_token(cfg.resolved_token_path(), cfg.argv() if argv is None else argv, code_dir=kw.pop("code_dir", self.code_dir), key=key, git_dir=self.git_dir,
                              profile=kw.pop("profile", G.PROFILE_SYNTHETIC))


ROOT = None
BASE = None          # the base synthetic world
STD = None           # the standard fixture (fake suite)
MOD = None           # srttc_r1 loaded from the VERIFIED bytes, exactly as the gate loads it


def setUpModule():
    global ROOT, BASE, STD, MOD
    ROOT = tempfile.mkdtemp(prefix="srttc_gate_t_")
    BASE = T.make_world(os.path.join(ROOT, "base"))
    add_summary(BASE)
    STD = Fixture(os.path.join(ROOT, "std"), clone_world(BASE, os.path.join(ROOT, "std", "world")))
    MOD, _ = G.load_verified_module(STD.code_dir)


def tearDownModule():
    shutil.rmtree(ROOT, ignore_errors=True)


def new_fixture(name: str, **kw) -> Fixture:
    w = clone_world(BASE, os.path.join(ROOT, name, "world"))
    return Fixture(os.path.join(ROOT, name), w, **kw)


def opts_for(fx: Fixture, **kw):
    argv = fx.cfg().argv()
    _res, opts = G.check_dvb(MOD, argv, stage="run", strict=False)
    return dataclasses.replace(opts, **{**fx.overrides, **kw})


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestPrimitives(unittest.TestCase):
    """The token format IS the programme's: known answers computed ONCE from refe/gate_token.py (blob 0e258774f3ff481224d14eb1c10584b73aaa4f6d at 58bd819) and pinned here."""

    KEY = bytes(range(32))

    def test_known_answer_vectors_equal_refe(self):
        self.assertEqual(G.canonical_json({"b": [1, "\u00e9", None], "a": 1.5, "c": {"z": 1, "y": 2}}).hex(), "7b2261223a312e352c2262223a5b312c22c3a9222c6e756c6c5d2c2263223a7b2279223a322c227a223a317d7d")
        self.assertEqual(G.argv_sha256(["a", "--b", "c d", "0,1"]), "0947ed7843c7bb65140c1ceae4d9520f4c1aadf6dc782ae29821fbd128d7627c")
        self.assertEqual(G.key_id(self.KEY), "2484cebe7ad1199f")
        body = {"schema": G.GATE_SCHEMA, "kind": "launch-gate-token", "verdict": "PASS", "binding": {"profile": "p", "argv": ["a", "--b", "c d"], "n": 3, "f": 0.1, "u": "\u00e9"},
                "created_utc": "2026-10-18T00:00:00Z"}
        self.assertEqual(G.sign(body, self.KEY), "82702d9f5ca96d7a875c763d8a11fc0ebf8728fcabc45e10b3f300bfa37f16a0")
        self.assertEqual(G.tree_digest({"srttc_r1.py": "ab" * 32, "srttc_r1_gate.py": "cd" * 32}), "de34e817262e3237580684e9c481d4d823b56ad60bf2f0391d7ebaf6081b204e")
        p = os.path.join(ROOT, "abc.bin")
        wb(p, b"abc")
        self.assertEqual(G.fingerprint_file(p), {"kind": "file", "bytes": 3, "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "method": "sha256-v1"})
        self.assertEqual(G.GATE_SCHEMA, "tanitad.launch_gate/1")

    def test_hmac_refuses_an_edit_and_another_key(self):
        tok = {"schema": G.GATE_SCHEMA, "verdict": "PASS", "binding": {"x": 1}}
        tok["hmac_sha256"] = G.sign(tok, self.KEY)
        self.assertTrue(G.signature_ok(tok, self.KEY))                                   # control
        self.assertFalse(G.signature_ok({**tok, "verdict": "FAIL"}, self.KEY))
        self.assertFalse(G.signature_ok({**tok, "binding": {"x": 2}}, self.KEY))
        self.assertFalse(G.signature_ok(tok, bytes(reversed(range(32)))))

    def test_key_file_cannot_live_inside_the_code_tree(self):
        with self.assertRaises(G.GateError):
            G.load_key(os.path.join(STD.code_dir, "k.key"), create=True, forbid_under=[STD.code_dir])
        k = G.load_key(os.path.join(ROOT, "okkeys", "k.key"), create=True, forbid_under=[STD.code_dir])     # control
        self.assertEqual(len(k), 32)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestDVB(unittest.TestCase):
    def dvb(self, argv=None, stage="run", strict=True, **kw):
        return G.check_dvb(MOD, argv if argv is not None else STD.cfg(stage=stage, **kw).argv(), stage=stage, strict=strict)[0]

    def test_registered_command_passes_strict(self):
        r = self.dvb()
        self.assertEqual(r["status"], "PASS", r["reasons"])
        self.assertEqual(r["details"]["n_configs_built"], 9)
        self.assertTrue(r["details"]["strict"])
        self.assertEqual(self.dvb(stage="lc")["status"], "PASS")    # the learning-curve stage is gated the same way

    def test_each_wrong_lever_fails(self):
        for kw, needle in (({"epochs": 11}, "epochs"), ({"batch_sets": 128}, "batch_sets"), ({"criterion": "fold_mean"}, "criterion"), ({"seeds": "0,1,2"}, "seeds"),
                           ({"both_rule": "rising"}, "both_rule"), ({"seeds": "0"}, "seeds")):
            r = self.dvb(**kw)
            self.assertEqual(r["status"], "FAIL", kw)
            self.assertTrue(any(needle in x for x in r["reasons"]), (kw, r["reasons"]))

    def test_fraction_outside_the_registered_sizes_fails(self):
        argv = STD.cfg().argv()
        argv[argv.index("--fraction") + 1] = "7"
        r = self.dvb(argv)
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("parser refuses" in x for x in r["reasons"]), r["reasons"])
        for f in G.REG_FRACTIONS:                                                         # control: every registered size passes
            self.assertEqual(self.dvb(fraction=f)["status"], "PASS", f)

    def test_a_lever_must_be_named(self):
        argv = STD.cfg().argv()
        i = argv.index("--criterion")
        r = self.dvb(argv[:i] + argv[i + 2:])                                              # the default happens to be right, but it is not DECLARED
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("not NAMED" in x for x in r["reasons"]), r["reasons"])

    def test_duplicate_and_unknown_flags_fail(self):
        self.assertEqual(self.dvb(STD.cfg().argv() + ["--epochs", "10"])["status"], "FAIL")
        r = self.dvb(STD.cfg().argv() + ["--lr", "0.1"])
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("unknown flag" in x or "parser refuses" in x for x in r["reasons"]), r["reasons"])

    def test_argv_shape_and_stage(self):
        argv = STD.cfg().argv()
        bad = list(argv)
        bad[1] = "-X"
        self.assertEqual(self.dvb(bad)["status"], "FAIL")
        self.assertEqual(self.dvb(argv, stage="lc")["status"], "FAIL")                    # an R1 argv is not an LC argv

    def test_grid_with_one_extra_config_fails(self):
        with mock.patch.object(MOD, "GRID_ANCHOR", (0.03, 0.1, 0.3, 1.0)):
            r = self.dvb()
        self.assertEqual(r["status"], "FAIL")
        self.assertEqual(r["details"]["n_configs_built"], 12)
        self.assertTrue(any("12 configurations" in x for x in r["reasons"]), r["reasons"])
        self.assertEqual(self.dvb()["status"], "PASS")                                     # control: restored

    def test_built_object_is_inspected_not_the_argv_text(self):
        orig = MOD.grid
        with mock.patch.object(MOD, "grid", lambda e=10, b=256: orig(e, b) + [orig(e, b)[0]]):
            r = self.dvb()                                                                 # the argv text is the registered one
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("10 configurations" in x for x in r["reasons"]), r["reasons"])

    def test_grid_reordered_or_an_axis_changed_fails(self):
        with mock.patch.object(MOD, "GRID_LR", (1e-3, 3e-4, 1e-4)):
            self.assertEqual(self.dvb()["status"], "FAIL")                                 # same 9 configs, different order: ties would go elsewhere
        with mock.patch.object(MOD, "GRID_LR", (1e-4, 3e-4, 3e-3)):
            self.assertEqual(self.dvb()["status"], "FAIL")

    def test_optimiser_constants_are_pinned(self):
        with mock.patch.object(MOD, "ADAM_BETAS", (0.9, 0.99)):
            r = self.dvb()
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("ADAM_BETAS" in x for x in r["reasons"]) or any("config sha256" in x for x in r["reasons"]), r["reasons"])

    def test_a_non_registered_path_or_pin_fails_strict_only(self):
        r = self.dvb(STD.cfg().argv() + ["--s0-marker", os.path.join(ROOT, "S0_DONE")])
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("s0_marker" in x for x in r["reasons"]), r["reasons"])
        self.assertEqual(self.dvb(STD.cfg().argv() + ["--s0-marker", os.path.join(ROOT, "S0_DONE")], strict=False)["status"], "PASS")   # the API-level synthetic route


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestPath(unittest.TestCase):
    def test_pass_and_controls(self):
        cfg = STD.cfg()
        r = G.check_path(MOD, opts_for(STD), cfg.argv())
        self.assertEqual(r["status"], "PASS", r["reasons"])
        self.assertEqual(r["details"]["n_tokens_refused_for_the_token"], len(MOD.FORBIDDEN_TOKENS))

    def test_a_navtest_bank_path_fails(self):
        cfg = STD.cfg()
        argv = cfg.argv() + ["--bank-dir", "C:/data/navtest/bank"]
        r = G.check_path(MOD, dataclasses.replace(opts_for(STD), bank_dir="C:/data/navtest/bank"), argv)
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("navtest" in x for x in r["reasons"]), r["reasons"])

    def test_a_devgate_out_dir_fails(self):
        argv = STD.cfg(out_dir=os.path.join(ROOT, "runs", "dev-gate")).argv()
        r = G.check_path(MOD, opts_for(STD, out_dir=os.path.join(ROOT, "runs", "dev-gate")), argv)
        self.assertEqual(r["status"], "FAIL")

    def test_a_forbidden_token_only_in_the_argv_fails(self):
        argv = STD.cfg().argv() + ["--replacements-file", "C:/data/navtest/replacements.json"]
        r = G.check_path(MOD, opts_for(STD), argv)                                         # the Options are the clean synthetic ones
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("names a dev-gate / navtest token" in x for x in r["reasons"]), r["reasons"])
        self.assertEqual(G.check_path(MOD, opts_for(STD), STD.cfg().argv())["status"], "PASS")     # control

    def test_a_disabled_guard_fails(self):
        with mock.patch.object(MOD.PathGuard, "forbidden", staticmethod(lambda p: [])):
            r = G.check_path(MOD, opts_for(STD), STD.cfg().argv())
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("did NOT refuse" in x for x in r["reasons"]), r["reasons"])

    def test_a_token_list_that_lost_navtest_fails(self):
        with mock.patch.object(MOD, "FORBIDDEN_TOKENS", tuple(t for t in MOD.FORBIDDEN_TOKENS if t != "navtest")):
            r = G.check_path(MOD, opts_for(STD), STD.cfg().argv())
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("lost 'navtest'" in x for x in r["reasons"]), r["reasons"])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestBank(unittest.TestCase):
    n = 0

    def world(self, **summary):
        type(self).n += 1
        w = clone_world(BASE, os.path.join(ROOT, f"bk{type(self).n}"))
        if summary:
            add_summary(w, **summary)
        return w

    def bank(self, w):
        ov = dict(bank_dir=w["bank"], manifest=w["manifest"], ranked_logs=w["tsv"], p2_ckpt=w["ckpt"], s0_marker=w["marker"], ranked_sha256=w["tsv_sha"], expected_n_logs=None,
                  min_logs_per_fold=2, reach_n_sets=64, threads=2)
        _r, opts = G.check_dvb(MOD, STD.cfg().argv(), stage="run", strict=False)
        return G.check_bank(MOD, dataclasses.replace(opts, **ov))

    def test_pass(self):
        r, aux = self.bank(self.world())
        self.assertEqual(r["status"], "PASS", r["reasons"])
        self.assertTrue(r["details"]["r1_reads_manifest"])
        self.assertEqual(r["details"]["n_units_verified"], r["details"]["n_units_on_disk"])
        self.assertEqual(r["details"]["n_selected_logs_missing_from_bank"], 0)
        self.assertEqual(len(aux["verified_units"]), r["details"]["n_units_on_disk"])

    def test_one_corrupted_unit_byte_fails(self):
        w = self.world()
        u = w["units"][3]
        data = bytearray(rb(u))
        data[len(data) // 2] ^= 0xFF
        wb(u, bytes(data))
        r, _ = self.bank(w)
        self.assertEqual(r["status"], "FAIL")
        self.assertEqual(r["details"]["n_units_sha_mismatch"], 1)
        self.assertTrue(any("sha256" in x and os.path.basename(u) in x for x in r["reasons"]), r["reasons"])

    def test_no_s0_done_fails_and_s0_failed_beside_it_fails(self):
        w = self.world()
        os.remove(w["marker"])
        r, _ = self.bank(w)
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("S0 is not done" in x for x in r["reasons"]), r["reasons"])
        w2 = self.world()
        wb(os.path.join(os.path.dirname(w2["marker"]), "S0_FAILED"), b"rc=5\n")
        self.assertEqual(self.bank(w2)[0]["status"], "FAIL")

    def test_logs_without_any_token_threshold_is_five_percent(self):
        self.assertEqual(self.bank(self.world(n_empty=2))[0]["status"], "PASS")           # 2 of 40 = 5.0 %: allowed
        r, _ = self.bank(self.world(n_empty=3))                                            # 3 of 40 = 7.5 %
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("logs_without_any_token" in x for x in r["reasons"]), r["reasons"])

    def test_unit_not_in_manifest_and_manifest_unit_not_on_disk_fail(self):
        w = self.world()
        extra = os.path.join(w["bank"], "Lextra000001.sets.npz")
        shutil.copyfile(w["units"][0], extra)
        r, _ = self.bank(w)
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("not in the manifest" in x for x in r["reasons"]), r["reasons"])
        w2 = self.world()
        os.remove(w2["units"][1])
        r2, _ = self.bank(w2)
        self.assertEqual(r2["status"], "FAIL")
        self.assertTrue(any("not on disk" in x for x in r2["reasons"]), r2["reasons"])

    def test_a_selected_log_with_no_set_in_the_bank_fails(self):
        w = self.world()
        first = w["units"][0]                                                              # unit 0 holds rank 0..2, all selected at f = 10 %
        os.remove(first)
        edit_manifest(w, lambda files: [files.pop(k) for k in [k for k in files if os.path.basename(k) == os.path.basename(first)]])
        r, _ = self.bank(w)
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("no set in the bank" in x for x in r["reasons"]), r["reasons"])

    def test_s0_summary_edited_after_the_manifest_fails(self):
        w = self.world()
        add_summary(w, tamper_after=True)
        r, _ = self.bank(w)
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("S0_SUMMARY.json sha256" in x for x in r["reasons"]), r["reasons"])
        w2 = self.world()
        os.remove(os.path.join(os.path.dirname(w2["marker"]), "S0_SUMMARY.json"))
        self.assertEqual(self.bank(w2)[0]["status"], "FAIL")

    def test_a_manifest_that_r1s_own_reader_rejects_fails_the_gate(self):
        """The REAL S0 manifest lists run/batches/<unit>/audit.json (15 distinct contents) and R1's load_manifest keys by basename and raises on that.  The gate must agree with R1's reader."""
        w = self.world()
        add = {f"run/batches/L{i:02d}/audit.json": {"path": f"C:/x/run/batches/L{i:02d}/audit.json", "sha256": hashlib.sha256(f"a{i}".encode()).hexdigest(), "bytes": 3} for i in range(3)}
        edit_manifest(w, lambda files: files.update(add))
        opts = dataclasses.replace(G.check_dvb(MOD, STD.cfg().argv(), stage="run", strict=False)[1], bank_dir=w["bank"], manifest=w["manifest"], ranked_logs=w["tsv"], p2_ckpt=w["ckpt"],
                                   s0_marker=w["marker"], ranked_sha256=w["tsv_sha"], expected_n_logs=None, min_logs_per_fold=2)
        try:
            MOD.load_manifest(opts.guard(), opts.manifest)
            accepts = True
        except MOD.BankError:
            accepts = False
        r, _ = G.check_bank(MOD, opts)
        self.assertEqual(r["status"] == "PASS", accepts, r["reasons"])                    # the gate says exactly what R1's reader says
        self.assertEqual(r["details"]["r1_reads_manifest"], accepts)
        with mock.patch.object(MOD, "load_manifest", side_effect=MOD.BankError("lists audit.json twice")):    # a reader that rejects: deterministic regression arm
            r2, _ = G.check_bank(MOD, opts)
        self.assertEqual(r2["status"], "FAIL")
        self.assertTrue(any("REJECTS this manifest" in x for x in r2["reasons"]), r2["reasons"])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestLive(unittest.TestCase):
    def live(self, opts=None):
        opts = opts or opts_for(STD)
        _r, aux = G.check_bank(MOD, opts)
        return G.check_live(MOD, opts, aux)

    def test_pass_and_recorded_facts(self):
        r = self.live()
        self.assertEqual(r["status"], "PASS", r["reasons"])
        d = r["details"]
        self.assertTrue(d["loss_finite"] and d["other_rows_and_features_exactly_zero"] and d["zero_control_exactly_zero"] and d["leak_control_reads_nonzero"] and d["one_step_moved_the_ttc_row"])
        self.assertTrue(all(d["ttc_row_gradient_nonzero"].values()))
        self.assertFalse(d["concordance_computed"])
        self.assertTrue(d["unit"].endswith(".sets.npz"))
        banned = {"concordance", "pairs", "n_pairs", "base_rate", "label_mean"}
        self.assertFalse(banned & {k for k in d if k != "concordance_computed"}, "no label statistic may be recorded")

    def test_first_unit_in_key_order_is_the_one_holding_the_lowest_ranked_selected_log(self):
        r = self.live()
        guard = opts_for(STD).guard()
        rows, _ = MOD.load_ranked_logs(guard, STD.w["tsv"], STD.w["tsv_sha"], None)
        want_log = rows[0]["log"]
        z = np.load(os.path.join(STD.w["bank"], r["details"]["unit"]), allow_pickle=False)
        self.assertIn(want_log, {str(x) for x in z["log"]})
        self.assertEqual(r["details"]["first_key_rank"], 0)

    def test_ttc_row_detached_fails(self):
        orig = MOD.forward_z6

        def detached(head, feat, z6s):
            h = types_ns(row_w=head.row_w.detach(), row_b=head.row_b.detach(), p2_w=head.p2_w, p2_b=head.p2_b, probe=head.probe,
                         frozen_W6=getattr(head, "frozen_W6", None), frozen_b6=getattr(head, "frozen_b6", None))
            return orig(h, feat, z6s)
        with mock.patch.object(MOD, "forward_z6", detached):
            r = self.live()
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("gradient on the TTC row" in x or "moved the TTC row" in x for x in r["reasons"]), r["reasons"])

    def test_pairwise_term_deleted_fails(self):
        with mock.patch.object(MOD, "pairwise_term", lambda z6, y3, idx8: z6[..., MOD.TTC_COL].sum() * 0.0):
            r = self.live()
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any(x.startswith("pair_only") for x in r["reasons"]), r["reasons"])

    def test_optimiser_step_cut_fails(self):
        with mock.patch.object(torch.optim.Adam, "step", lambda self, *a, **k: None):
            r = self.live()
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("moved the TTC row" in x for x in r["reasons"]), r["reasons"])

    def test_features_not_detached_fails(self):
        def attached(head, feat, z6s):
            feat = feat.float()                                                            # MUTATION: the detach is gone
            delta = feat @ (head.row_w - head.p2_w) + (head.row_b - head.p2_b)
            z_ttc = z6s[..., MOD.TTC_COL] + delta
            z_oth = torch.nn.functional.linear(feat, head.frozen_W6, head.frozen_b6) if head.probe else z6s
            return torch.stack([z_ttc if c == MOD.TTC_COL else z_oth[..., c] for c in range(MOD.N_COMP)], dim=-1)
        with mock.patch.object(MOD, "forward_z6", attached):
            r = self.live()
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("features received a gradient" in x for x in r["reasons"]), r["reasons"])

    def test_a_loss_that_leaks_into_another_row_fails(self):
        orig = MOD.total_loss

        def leaky_total(z6, y3, idx8, aw, pw=1.0):
            loss, p, a = orig(z6, y3, idx8, aw, pw)
            return loss + 0.1 * z6[..., 2].pow(2).mean(), p, a
        with mock.patch.object(MOD, "total_loss", leaky_total):
            r = self.live()
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("NON-TTC row" in x for x in r["reasons"]), r["reasons"])

    def test_wrong_p2_checkpoint_fails(self):
        rng = np.random.default_rng(99)
        w = clone_world(BASE, os.path.join(ROOT, "wrongp2", "world"))
        W6, b6 = STD.w["W6"].copy(), STD.w["b6"].copy()
        W6[3] = W6[3] + rng.normal(size=W6[3].shape).astype(np.float32)
        torch.save({"format": "refe-full-v1", "model": {"score_head.weight": torch.from_numpy(W6), "score_head.bias": torch.from_numpy(b6)}, "meta": {}}, w["ckpt"])
        opts = opts_for(STD, p2_ckpt=w["ckpt"])
        _r, aux = G.check_bank(MOD, opts)
        r = G.check_live(MOD, opts, aux)
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("does not reproduce the stored TTC logits" in x for x in r["reasons"]), r["reasons"])

    def test_a_blind_probe_fails_its_own_positive_control(self):
        """If the probe could not read a leak, every 'exactly 0.0' above would mean nothing.  A forward that routes the other columns around the probe parameters is blind: G-LIVE must say so."""
        def blind(head, feat, z6s):
            feat = feat.detach().float()
            delta = feat @ (head.row_w - head.p2_w) + (head.row_b - head.p2_b)
            return torch.stack([z6s[..., c] + (delta if c == MOD.TTC_COL else 0.0) for c in range(MOD.N_COMP)], dim=-1)
        with mock.patch.object(MOD, "forward_z6", blind):
            r = self.live()
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("positive control" in x for x in r["reasons"]), r["reasons"])
        self.assertEqual(self.live()["status"], "PASS")                                    # control: the real forward is not blind

    def test_a_corrupted_first_unit_fails_before_it_is_read(self):
        w = clone_world(BASE, os.path.join(ROOT, "corruptlive", "world"))
        r0 = self.live()
        u = os.path.join(w["bank"], r0["details"]["unit"])
        data = bytearray(rb(u))
        data[len(data) // 3] ^= 0x01
        wb(u, bytes(data))
        opts = dataclasses.replace(opts_for(STD), bank_dir=w["bank"], manifest=w["manifest"])
        r = G.check_live(MOD, opts, {})
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("sha256" in x for x in r["reasons"]), r["reasons"])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestSuiteAndCommit(unittest.TestCase):
    n = 0

    def fx(self, **kw):
        type(self).n += 1
        return new_fixture(f"su{type(self).n}", **kw)

    def suite(self, fx, **kw):
        return G.check_suite(commit=fx.commit, git_dir=fx.git_dir, repo_code_path=REPO_CODE_PATH, python=sys.executable, min_tests=3, **kw)

    def test_pass_then_each_regression_arm_fails(self):
        ok = self.suite(self.fx())
        self.assertEqual(ok["status"], "PASS", ok["reasons"])
        self.assertEqual(ok["details"]["n_tests"], 3)
        mut = self.suite(self.fx(tests=FAKE_TESTS_MUTATED))
        self.assertEqual(mut["status"], "FAIL")
        self.assertTrue(any("exited" in x or "FAILED" in x for x in mut["reasons"]), mut["reasons"])
        emp = self.suite(self.fx(tests=FAKE_TESTS_EMPTY))
        self.assertEqual(emp["status"], "FAIL")
        self.assertTrue(any("floor" in x for x in emp["reasons"]), emp["reasons"])
        skp = self.suite(self.fx(tests=FAKE_TESTS_SKIP))
        self.assertEqual(skp["status"], "FAIL")
        self.assertTrue(any("'OK'" in x for x in skp["reasons"]), skp["reasons"])

    def test_the_suite_runs_at_the_commit_not_in_the_working_dir(self):
        fx = self.fx()
        wb(os.path.join(fx.code_dir, G.SUITE_TEST_FILE), FAKE_TESTS_MUTATED.encode())      # an edited deployed copy must not matter: the clean export is what runs
        self.assertEqual(self.suite(fx)["status"], "PASS")

    def test_no_test_file_at_the_commit_fails(self):
        fx = self.fx()
        r = G.check_suite(commit=fx.commit, git_dir=fx.git_dir, repo_code_path=REPO_CODE_PATH, python=sys.executable, test_file="test_nothing.py", min_tests=3)
        self.assertEqual(r["status"], "FAIL")

    def test_commit_check(self):
        fx = self.fx()
        kw = dict(commit=fx.commit, git_dir=fx.git_dir, repo_code_path=REPO_CODE_PATH, code_dir=fx.code_dir)
        self.assertEqual(G.check_commit(**kw)["status"], "PASS")                           # control
        wb(os.path.join(fx.code_dir, "srttc_r1.py"), rb(os.path.join(fx.code_dir, "srttc_r1.py")) + b"\n# edited\n")
        r = G.check_commit(**kw)
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("differs from its blob" in x for x in r["reasons"]), r["reasons"])
        fx2 = self.fx()
        wb(os.path.join(fx2.code_dir, "srttc_r1_gate.py"), rb(os.path.join(fx2.code_dir, "srttc_r1_gate.py")) + b"\n")
        self.assertEqual(G.check_commit(commit=fx2.commit, git_dir=fx2.git_dir, repo_code_path=REPO_CODE_PATH, code_dir=fx2.code_dir)["status"], "FAIL")
        for bad in ("0" * 40, "abc", "", fx.commit[:12]):
            self.assertEqual(G.check_commit(**{**kw, "commit": bad})["status"], "FAIL", bad)

    def test_unlanded_file_fails(self):
        fx = self.fx()
        shutil.copyfile(os.path.join(HERE, "srttc_r1_gate.py"), os.path.join(fx.code_dir, "srttc_r1_gate.py"))
        git(os.path.join(fx.root, "repo"), "rm", "-q", "--cached", "-f", f"{REPO_CODE_PATH}/srttc_r1_gate.py")
        git(os.path.join(fx.root, "repo"), "commit", "-q", "-m", "drop")
        c2 = git(os.path.join(fx.root, "repo"), "rev-parse", "HEAD")
        r = G.check_commit(commit=c2, git_dir=fx.git_dir, repo_code_path=REPO_CODE_PATH, code_dir=fx.code_dir)
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("not landed" in x for x in r["reasons"]), r["reasons"])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestMintAndVerify(unittest.TestCase):
    n = 0

    def fx(self):
        type(self).n += 1
        return new_fixture(f"mv{type(self).n}")

    def test_full_gate_passes_end_to_end_on_a_synthetic_world(self):
        fx = self.fx()
        res = fx.mint()
        self.assertEqual(res["verdict"], "PASS", res["reasons"])
        self.assertTrue(os.path.basename(res["token_path"]) == "r1_gate_token.json" and os.path.exists(res["token_path"]))
        for c in G.CHECKS:
            self.assertEqual(res["checks"][c]["status"], "PASS", (c, res["checks"][c]["reasons"]))
        tok = G.read_json(res["token_path"])
        b = tok["binding"]
        self.assertEqual(b["profile"], G.PROFILE_SYNTHETIC)                                  # a synthetic world can never mint a REAL-profile token
        self.assertEqual(b["commit"], fx.commit)
        self.assertEqual(b["argv_sha256"], G.argv_sha256(res["cfg"].argv()))
        self.assertEqual(b["srttc_r1_sha256"], hashlib.sha256(rb(os.path.join(fx.code_dir, "srttc_r1.py"))).hexdigest())
        self.assertEqual(b["bank_manifest_sha256"], hashlib.sha256(rb(fx.w["manifest"])).hexdigest())
        self.assertEqual(tok["schema"], "tanitad.launch_gate/1")
        ok, reasons, info = fx.verify(res)
        self.assertTrue(ok, reasons)

    def assert_no_token(self, res, check):
        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["checks"][check]["status"], "FAIL", res["checks"][check]["reasons"])
        self.assertIsNone(res["token_path"])
        self.assertFalse(os.path.exists(res["cfg"].resolved_token_path()), "no r1_gate_token.json may exist after a failing run")
        self.assertTrue(os.path.exists(os.path.join(os.path.dirname(res["cfg"].resolved_token_path()), G.FAIL_NAME)))

    def test_every_check_has_a_regression_arm_that_blocks_the_token(self):
        fx = self.fx()
        self.assertEqual(fx.mint()["verdict"], "PASS")                                       # the control: same fixture, no regression
        self.assert_no_token(fx.mint(epochs=11), "G-DVB")                                    # epochs 11
        with mock.patch.object(MOD, "GRID_ANCHOR", (0.03, 0.1, 0.3, 1.0)):                  # a grid with one extra config
            self.assert_no_token(fx.mint(), "G-DVB")
        orig = MOD.forward_z6
        with mock.patch.object(MOD, "forward_z6", lambda head, feat, z6s: orig(types_ns(row_w=head.row_w.detach(), row_b=head.row_b.detach(), p2_w=head.p2_w, p2_b=head.p2_b,
                                                                                         probe=head.probe, frozen_W6=getattr(head, "frozen_W6", None),
                                                                                         frozen_b6=getattr(head, "frozen_b6", None)), feat, z6s)):
            self.assert_no_token(fx.mint(), "G-LIVE")                                        # TTC row detached
        with mock.patch.object(MOD.PathGuard, "forbidden", staticmethod(lambda p: [])):      # a disabled dev-gate / navtest guard
            self.assert_no_token(fx.mint(), "G-PATH")
        nav = dict(fx.overrides, bank_dir="C:/data/navtest/bank")                            # a navtest path
        self.assert_no_token(fx.mint(overrides=nav), "G-PATH")

    def test_corrupted_unit_byte_blocks_the_token(self):
        fx = self.fx()
        u = fx.w["units"][2]
        data = bytearray(rb(u))
        data[len(data) // 2] ^= 0xFF
        wb(u, bytes(data))
        res = fx.mint()
        self.assert_no_token(res, "G-BANK")
        self.assertEqual(res["checks"]["G-LIVE"]["status"], "NOT-RUN")                       # nothing is read from a bank that is not verified

    def test_mutated_test_file_blocks_the_token(self):
        fx = new_fixture("mv_mut", tests=FAKE_TESTS_MUTATED)
        self.assert_no_token(fx.mint(), "G-SUITE")

    def test_deployed_file_differing_from_its_blob_blocks_the_token_and_runs_nothing(self):
        fx = self.fx()
        wb(os.path.join(fx.code_dir, "srttc_r1.py"), rb(os.path.join(fx.code_dir, "srttc_r1.py")) + b"\n# edit\n")
        res = fx.mint()
        self.assert_no_token(res, "G-COMMIT")
        for c in ("G-DVB", "G-PATH", "G-BANK", "G-LIVE", "G-SUITE"):
            self.assertEqual(res["checks"][c]["status"], "NOT-RUN", c)                       # no unverified byte is imported or run

    def test_token_for_argv_x_used_with_argv_y_fails_verify(self):
        fx = self.fx()
        res = fx.mint()
        self.assertTrue(fx.verify(res)[0])                                                   # control
        for kw in ({"epochs": 11}, {"threads": 4}, {"fraction": 25}, {"batch_sets": 128}, {"seeds": "0,1,2"}, {"stage": "lc"}):
            ok, reasons, _ = fx.verify(res, argv=fx.cfg(**kw).argv())
            self.assertFalse(ok, kw)
            self.assertTrue(any("argv differs" in r for r in reasons), (kw, reasons))
        ok, reasons, _ = fx.verify(res, argv=res["cfg"].argv() + ["--replacements-file", "x.json"])
        self.assertFalse(ok)

    def test_edited_token_wrong_key_and_wrong_profile_fail(self):
        fx = self.fx()
        res = fx.mint()
        p = res["token_path"]
        tok = G.read_json(p)
        edited = dict(tok, created_utc="2030-01-01T00:00:00Z")
        G.write_json(p, edited)
        ok, reasons, _ = fx.verify(res)
        self.assertFalse(ok)
        self.assertTrue(any("HMAC" in r for r in reasons), reasons)
        G.write_json(p, tok)
        self.assertTrue(fx.verify(res)[0])                                                   # control: the original bytes verify again
        other = os.path.join(ROOT, "otherkey", "k.key")
        G.load_key(other, create=True)
        ok, reasons, _ = fx.verify(res, key_file=other)
        self.assertFalse(ok)
        self.assertTrue(any("HMAC" in r for r in reasons), reasons)
        ok, reasons, _ = fx.verify(res, profile=G.PROFILE)                                   # a synthetic-world token can never launch the REAL R1
        self.assertFalse(ok)
        self.assertTrue(any("profile" in r for r in reasons), reasons)

    def test_code_manifest_and_marker_changes_after_the_mint_fail_verify(self):
        fx = self.fx()
        res = fx.mint()
        self.assertTrue(fx.verify(res)[0])
        man = rb(fx.w["manifest"])
        wb(fx.w["manifest"], man + b"\n")
        ok, reasons, _ = fx.verify(res)
        self.assertFalse(ok)
        self.assertTrue(any("inputs differ" in r and "bank_manifest" in r for r in reasons), reasons)
        wb(fx.w["manifest"], man)
        self.assertTrue(fx.verify(res)[0])
        marker = rb(fx.w["marker"])
        wb(fx.w["marker"], marker + b"resumed\n")
        ok, reasons, _ = fx.verify(res)
        self.assertFalse(ok)
        self.assertTrue(any("s0_marker" in r for r in reasons), reasons)
        wb(fx.w["marker"], marker)
        src = rb(os.path.join(fx.code_dir, "srttc_r1.py"))
        wb(os.path.join(fx.code_dir, "srttc_r1.py"), src + b"\n")
        ok, reasons, _ = fx.verify(res)
        self.assertFalse(ok)
        self.assertTrue(any("srttc_r1.py" in r for r in reasons), reasons)

    def test_a_fail_report_is_never_accepted_and_a_stale_pass_is_demoted(self):
        fx = self.fx()
        ok_res = fx.mint(token_path=os.path.join(fx.root, "tok", "same", G.TOKEN_NAME))
        self.assertEqual(ok_res["verdict"], "PASS")
        bad = fx.mint(token_path=os.path.join(fx.root, "tok", "same", G.TOKEN_NAME), epochs=11)
        self.assertEqual(bad["verdict"], "FAIL")
        d = os.path.join(fx.root, "tok", "same")
        self.assertFalse(os.path.exists(os.path.join(d, G.TOKEN_NAME)), "the stale PASS must not survive a failing run")
        self.assertTrue(os.listdir(os.path.join(d, "superseded")))
        key = G.load_key(fx.key_file, create=False)
        ok, reasons, _ = G.verify_token(os.path.join(d, G.FAIL_NAME), bad["cfg"].argv(), code_dir=fx.code_dir, key=key, git_dir=fx.git_dir, profile=G.PROFILE_SYNTHETIC)
        self.assertFalse(ok)
        self.assertTrue(any("not PASS" in r for r in reasons), reasons)
        again = fx.mint(token_path=os.path.join(d, G.TOKEN_NAME))
        self.assertEqual(again["verdict"], "PASS")
        self.assertFalse(os.path.exists(os.path.join(d, G.FAIL_NAME)))                       # a later PASS demotes the FAIL report

    def test_a_run_stage_token_cannot_launch_the_lc_stage_and_lc_has_its_own_token(self):
        fx = self.fx()
        run = fx.mint()
        lc = fx.mint(stage="lc")
        self.assertEqual((run["verdict"], lc["verdict"]), ("PASS", "PASS"))
        self.assertFalse(fx.verify(run, argv=fx.cfg(stage="lc").argv())[0])
        self.assertTrue(fx.verify(lc)[0])

    def test_missing_token_missing_key_and_missing_commit_refuse(self):
        fx = self.fx()
        cfg = fx.cfg()
        key = G.load_key(fx.key_file, create=True)
        ok, reasons, _ = G.verify_token(cfg.resolved_token_path(), cfg.argv(), code_dir=fx.code_dir, key=key, git_dir=fx.git_dir, profile=G.PROFILE_SYNTHETIC)
        self.assertFalse(ok)
        self.assertTrue(any("unreadable" in r for r in reasons), reasons)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestCLI(unittest.TestCase):
    def args(self, fx, cfg, *extra, sub=None):
        a = ([sub] if sub else []) + ["--fraction", "10", "--commit", fx.commit, "--python", sys.executable, "--code-dir", fx.code_dir, "--out-dir", cfg.out_dir, "--threads", "2",
                                      "--git-dir", fx.git_dir, "--repo-code-path", REPO_CODE_PATH, "--token", cfg.token_path, "--key-file", fx.key_file]
        return a + list(extra)

    def run_main(self, argv, **kw):
        buf = []
        rc = G.main(argv, _out=lambda *a, **k: buf.append(" ".join(str(x) for x in a)), **kw)
        return rc, buf

    def test_default_command_is_mint_and_verify_prints_one_json_line(self):
        fx = new_fixture("cli1")
        cfg = fx.cfg()
        rc, out = self.run_main(self.args(fx, cfg), _test_overrides=fx.overrides, _test_suite={"suite_min_tests": 3})
        self.assertEqual(rc, 0, out)                                                         # `srttc_r1_gate.py --fraction 10 --commit ...` with no sub-command mints
        rc, out = self.run_main(self.args(fx, cfg, sub="verify"), _test_overrides=fx.overrides)
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(out), 1)
        v = json.loads(out[0])
        self.assertTrue(v["ok"])
        self.assertEqual(v["argv"][3:7], ["run", "--fraction", "10", "--out-dir"])
        rc, out = self.run_main(self.args(fx, cfg, "--epochs", "11", sub="verify"), _test_overrides=fx.overrides)
        self.assertEqual(rc, 3)
        v = json.loads(out[0])
        self.assertFalse(v["ok"])
        self.assertTrue(any("argv differs" in r for r in v["reasons"]), v["reasons"])
        rc, out = self.run_main(self.args(fx, cfg, sub="verify"))                                  # without the synthetic hook the profile is the REAL one: refused
        self.assertEqual(rc, 3)
        self.assertTrue(any("profile" in r for r in json.loads(out[0])["reasons"]))

    def test_cli_refuses_epochs_11_and_a_missing_commit(self):
        fx = new_fixture("cli2")
        cfg = fx.cfg()
        rc, out = self.run_main(self.args(fx, cfg, "--epochs", "11"), _test_overrides=fx.overrides, _test_suite={"suite_min_tests": 3})
        self.assertEqual(rc, 1, out)
        self.assertFalse(os.path.exists(cfg.token_path))
        rc, out = self.run_main(["--fraction", "10"])
        self.assertEqual(rc, 2)

    def test_argv_command_prints_the_registered_command(self):
        rc, out = self.run_main(["argv", "--fraction", "10"])
        self.assertEqual(rc, 0)
        argv = json.loads(out[0])["argv"]
        self.assertEqual(argv[:2], [G.DEFAULT_PYTHON, "-I"])
        self.assertEqual(argv[2], r"C:\Users\Admin\srttc_r1\code\srttc_r1.py")
        i = argv.index("--epochs")
        self.assertEqual(argv[i:i + 2], ["--epochs", "10"])
        for flag, val in (("--batch-sets", "256"), ("--seeds", "0,1"), ("--criterion", "pooled_oof"), ("--both-rule", "flat"), ("--fraction", "10"), ("--out-dir", r"C:\Users\Admin\srttc_r1\run\f10")):
            self.assertEqual(argv[argv.index(flag) + 1], val, flag)

    def test_the_real_cli_cannot_reach_the_synthetic_hooks(self):
        self.assertNotIn("test", " ".join(a.dest for a in G.make_parser()._actions))
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "srttc_r1_gate.py"), "argv", "--fraction", "10", "--bank-dir", "x"], capture_output=True)
        self.assertNotEqual(r.returncode, 0)                                                 # no --bank-dir / --manifest / --p2-ckpt flag exists on the gate's command line


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
LAUNCHER = os.path.join(HERE, "launch_r1.ps1")


def ps(command: str, timeout: int = 120) -> tuple:
    r = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True, timeout=timeout)
    return r.returncode, (r.stdout + b"\n" + r.stderr).decode("utf-8", "replace")


def run_launcher(*args: str) -> tuple:
    """The 'paste the body' route: the script text is compiled into a script block (execution policy is Restricted on this box; -ExecutionPolicy Bypass is never used)."""
    cmd = ("try { & ([scriptblock]::Create((Get-Content -Raw -LiteralPath '" + LAUNCHER.replace("'", "''") + "'))) " + " ".join(args) +
           " } catch { 'CAUGHT: ' + $_.Exception.Message; exit 1 }")
    return ps(cmd)


STUB_GATE = ("import json, sys\n"
             "mode = open(__file__.replace('srttc_r1_gate.py', 'stub_mode.txt')).read().strip()\n"
             "argv = ['C:\\\\Users\\\\Admin\\\\venvs\\\\driverl-eval\\\\Scripts\\\\python.exe', '-I', __file__.replace('srttc_r1_gate.py', 'srttc_r1.py'), 'run', '--fraction', '10', '--out-dir', 'C:\\\\x\\\\run\\\\f10',\n"
             "        '--threads', '8', '--epochs', '10', '--batch-sets', '256', '--seeds', '0,1', '--criterion', 'pooled_oof', '--both-rule', 'flat']\n"
             "open(__file__.replace('srttc_r1_gate.py', 'seen.json'), 'w').write(json.dumps(sys.argv[1:]))\n"
             "ok = mode in ('ok', 'ok_but_exit3')\n"
             "print(json.dumps({'ok': ok, 'reasons': [] if ok else ['stub refuses'], 'argv': argv}))\n"
             "sys.exit(0 if mode in ('ok', 'false_but_exit0') else 3)\n")


class TestLauncher(unittest.TestCase):
    def scratch(self, name, mode=None, s0_done=True, manifest=True):
        d = os.path.join(ROOT, name)
        code, s0, bank = os.path.join(d, "code"), os.path.join(d, "s0run"), os.path.join(d, "bank")
        for p in (code, s0, bank):
            os.makedirs(p, exist_ok=True)
        wb(os.path.join(code, "srttc_r1.py"), b"# stub\n")
        if mode is not None:
            wb(os.path.join(code, "srttc_r1_gate.py"), STUB_GATE.encode())
            wb(os.path.join(code, "stub_mode.txt"), mode.encode())
        if s0_done:
            wb(os.path.join(s0, "S0_DONE"), b"{}")
        if manifest:
            wb(os.path.join(bank, "MANIFEST.sha256.json"), b"{}")
        args = [f"-Fraction 10 -DryRun -CodeDir '{code}' -S0Run '{s0}' -Bank '{bank}' -OutRoot '{d}\\run' -GateRoot '{d}\\gate'"]
        return d, " ".join(args)

    def test_script_parses_and_keeps_the_wmi_pattern_and_never_bypasses_execution_policy(self):
        with open(LAUNCHER, encoding="utf-8") as f:
            text = f.read()
        rc, out = ps("$e = $null; [void][System.Management.Automation.Language.Parser]::ParseFile('" + LAUNCHER.replace("'", "''") + "', [ref]$null, [ref]$e); "
                     "if ($e.Count -gt 0) { $e | ForEach-Object { $_.Message }; exit 1 } else { 'PARSE_OK' }")
        self.assertIn("PARSE_OK", out)
        code_lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        code = "\n".join(code_lines)
        for needle in ("Win32_Process", "Win32_ProcessStartup", "0x08000200", "ProcessStartupInformation", "' < NUL >> \"'", "cmd.exe /c", "Invoke-CimMethod"):
            self.assertIn(needle, code, needle)
        self.assertNotIn("Bypass", code)
        self.assertNotIn("-ExecutionPolicy", code)
        self.assertIn("S0_DONE", code)                                                       # the held refusal is kept
        self.assertIn("srttc_r1_gate.py", code)

    def test_refuses_without_s0_done_and_creates_nothing(self):
        d, a = self.scratch("pl1", mode="ok", s0_done=False)
        rc, out = run_launcher(a)
        self.assertNotEqual(rc, 0)
        self.assertIn("REFUSED", out)
        self.assertIn("S0_DONE", out)
        self.assertFalse(os.path.exists(os.path.join(d, "run")))

    def test_refuses_without_the_gate_script_or_without_a_token(self):
        d, a = self.scratch("pl2", mode=None)                                               # no srttc_r1_gate.py deployed
        rc, out = run_launcher(a)
        self.assertNotEqual(rc, 0)
        self.assertIn("srttc_r1_gate.py is not deployed", out)
        d2 = os.path.join(ROOT, "pl3")
        code = os.path.join(d2, "code")
        os.makedirs(code, exist_ok=True)
        for f in G.BOUND_CODE_FILES:
            shutil.copyfile(os.path.join(HERE, f), os.path.join(code, f))                   # the REAL gate, no token minted
        for p in ("s0run", "bank"):
            os.makedirs(os.path.join(d2, p), exist_ok=True)
        wb(os.path.join(d2, "s0run", "S0_DONE"), b"{}")
        wb(os.path.join(d2, "bank", "MANIFEST.sha256.json"), b"{}")
        rc, out = run_launcher(f"-Fraction 10 -DryRun -CodeDir '{code}' -S0Run '{d2}\\s0run' -Bank '{d2}\\bank' -OutRoot '{d2}\\run' -GateRoot '{d2}\\gate'")
        self.assertNotEqual(rc, 0)
        self.assertIn("REFUSED by the launch gate", out)                                    # no token / no key: the gate refuses and the launcher stops
        self.assertFalse(os.path.exists(os.path.join(d2, "run")))

    def test_every_gate_refusal_shape_stops_the_launch(self):
        for mode in ("refuse", "false_but_exit0", "ok_but_exit3"):
            d, a = self.scratch(f"pl_{mode}", mode=mode)
            rc, out = run_launcher(a)
            self.assertNotEqual(rc, 0, mode)
            self.assertIn("REFUSED", out, mode)
            self.assertNotIn("DRY RUN", out, mode)
            self.assertFalse(os.path.exists(os.path.join(d, "run")), mode)

    def test_a_verified_token_launches_exactly_the_argv_the_gate_printed_in_the_held_wmi_form(self):
        d, a = self.scratch("pl_ok", mode="ok")
        rc, out = run_launcher(a)
        self.assertEqual(rc, 0, out)
        line = [ln for ln in out.splitlines() if ln.startswith("DRY RUN")]
        self.assertEqual(len(line), 1, out)
        py = r"C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe"
        code = os.path.join(d, "code")
        argv = [py, "-I", os.path.join(code, "srttc_r1.py"), "run", "--fraction", "10", "--out-dir", r"C:\x\run\f10", "--threads", "8", "--epochs", "10", "--batch-sets", "256",
                "--seeds", "0,1", "--criterion", "pooled_oof", "--both-rule", "flat"]
        quoted = " ".join(f'"{x}"' if re.search(r"[\s\\/]", x) else x for x in argv)
        log = os.path.join(d, "run", "f10") + "\\r1.log"
        want = f'cmd.exe /c "{quoted} < NUL >> "{log}" 2>&1"'
        self.assertEqual(line[0], "DRY RUN (gate verified; not launched): " + want)
        self.assertFalse(os.path.exists(os.path.join(d, "run")), "a dry run launches nothing and creates nothing")
        # the launcher handed the gate EXACTLY its own parameters (the gate rebuilds the command from them and verifies the token against it)
        seen = G.read_json(os.path.join(code, "seen.json"))
        self.assertEqual(seen, ["verify", "--stage", "run", "--fraction", "10", "--python", py, "--code-dir", code, "--out-dir", os.path.join(d, "run", "f10"), "--threads", "8",
                                "--epochs", "10", "--batch-sets", "256", "--token", os.path.join(d, "gate", "f10_run", "r1_gate_token.json"), "--git-dir", "C:/Users/Admin/tanitad-push/.git"])

    def test_lc_stage_uses_its_own_token_path_and_needs_r1_result(self):
        d, a = self.scratch("pl_lc", mode="ok")
        rc, out = run_launcher(a + " -Lc")
        self.assertNotEqual(rc, 0)
        self.assertIn("r1_result.json does not exist", out)
        wb(os.path.join(d, "run", "f10", "r1_result.json"), b"{}")
        rc, out = run_launcher(a + " -Lc")
        self.assertEqual(rc, 0, out)
        self.assertIn("lc.log", out)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestEndToEndRealSuite(unittest.TestCase):
    def test_full_gate_with_the_real_test_suite_on_a_clean_git_archive_export(self):
        """G-SUITE = the REAL test_srttc_r1.py (all 33 tests) in full form on a clean export; every other check on the synthetic world; the token verifies."""
        fx = new_fixture("e2e_real", test_bytes=rb(os.path.join(HERE, G.SUITE_TEST_FILE)))
        res = fx.mint(suite_min_tests=G.REG_MIN_SUITE_TESTS)
        self.assertEqual(res["verdict"], "PASS", res["reasons"])
        s = res["checks"]["G-SUITE"]["details"]
        self.assertGreaterEqual(s["n_tests"], G.REG_MIN_SUITE_TESTS)
        self.assertEqual(s["status_line"], "OK")
        self.assertEqual(s["returncode"], 0)
        self.assertTrue(fx.verify(res)[0])
        print(f"\n  [real G-SUITE] {s['n_tests']} tests, {s['seconds']} s, status {s['status_line']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
