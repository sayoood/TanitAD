"""Tests of build_candidate_bank.py that need NO GPU and NO pod: pins, refusals, the override plumbing of the landed S0 driver, the set-store label check, the launch text.
Run under the driverl-eval interpreter on the dev box (it imports the landed driver):  C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I test_build_candidate_bank.py
Tests that need S0's files (bank, plan, code) are SKIPPED when this is not the dev box.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_candidate_bank as BB                                                          # noqa: E402

HAVE_S0 = os.path.exists(f"{BB.S0_ROOT}/work/bank/MANIFEST.sha256.json") and os.path.exists(f"{BB.S0_CODE}/srttc_s0_driver.py") and os.path.exists(BB.P2_CKPT)
WORK = "D:/Projects/TanitAD-artifacts/_p6prep_unit_root"


def s0_listing_hash() -> str:
    h = hashlib.sha256()
    for d, _dirs, files in os.walk(BB.S0_ROOT):
        for f in sorted(files):
            p = os.path.join(d, f)
            st = os.stat(p)
            h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


@unittest.skipUnless(HAVE_S0, "needs the dev box's S0 directory")
class TestBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists(WORK):
            shutil.rmtree(WORK)
        cls.s0_before = s0_listing_hash()
        ns = argparse.Namespace(root=WORK, ckpt=BB.P2_CKPT, ckpt_sha256=BB.P2_CKPT_SHA256, candidate_id="unit", s0_root=BB.S0_ROOT, s0_code=BB.S0_CODE)
        assert BB.cmd_prepare(ns) == 0
        cls.cand = BB.read_candidate(WORK)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(WORK, ignore_errors=True)

    def test_zz_s0_directory_is_untouched(self):
        self.assertEqual(s0_listing_hash(), self.s0_before)                                      # (named zz_ so it runs last)

    def test_pins_are_the_landed_files(self):
        self.assertEqual(len(BB.check_s0_pins()), len(BB.S0_PINS))

    def test_a_changed_landed_file_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            for rel in BB.S0_PINS:
                os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True)
                shutil.copyfile(os.path.join(BB.S0_CODE, rel), os.path.join(d, rel))
            self.assertEqual(len(BB.check_s0_pins(d)), len(BB.S0_PINS))
            with open(os.path.join(d, "srttc_s0_dump.py"), "ab") as f:
                f.write(b"\n# changed\n")
            with self.assertRaises(BB.BuildError):
                BB.check_s0_pins(d)

    def test_prepare_refuses_c_drive_s0_root_and_a_wrong_checkpoint_hash(self):
        base = dict(ckpt=BB.P2_CKPT, ckpt_sha256=BB.P2_CKPT_SHA256, candidate_id="x", s0_root=BB.S0_ROOT, s0_code=BB.S0_CODE)
        with self.assertRaises(BB.BuildError):
            BB.cmd_prepare(argparse.Namespace(root="C:/Users/Admin/p6bad_root", **base))
        with self.assertRaises(BB.BuildError):
            BB.cmd_prepare(argparse.Namespace(root=f"{BB.S0_ROOT}/inside", **base))
        base["ckpt_sha256"] = "0" * 64
        with self.assertRaises(BB.BuildError):
            BB.cmd_prepare(argparse.Namespace(root=WORK + "_other", **base))
        self.assertFalse(os.path.exists(f"{BB.S0_ROOT}/inside"))

    def test_the_plan_copied_is_the_registered_sample(self):
        mp = json.load(open(f"{WORK}/run/plan/master_plan.json", encoding="utf-8"))
        self.assertEqual((mp["n_logs"], mp["n_tokens"], mp["n_sets"]), (BB.N_LOGS, BB.N_TOKENS, BB.N_SETS))
        self.assertEqual(BB.sha256_file(f"{WORK}/run/plan/ranked_fit_logs.tsv"), BB.RANKED_LOGS_SHA256)

    def test_dump_refuses_until_the_sets_exist_and_are_verified(self):
        with self.assertRaises(BB.BuildError):
            BB.cmd_dump(argparse.Namespace(root=WORK, device="cpu", min_free_ram_mb=4500, no_stage=False, dry_run=True))

    def test_the_driver_override_routes_everything_to_the_new_root(self):
        """Fabricate empty per-log set stores + a provenance file + verify_sets PASS in the TEMP root (the driver only checks that they exist), import the LANDED driver with the four overrides and run ITS
        preflight: it must read the plan from the temp root, hash the candidate checkpoint, and write preflight.json under the temp root, not under S0's."""
        mp = json.load(open(f"{WORK}/run/plan/master_plan.json", encoding="utf-8"))
        for r in mp["logs"]:
            np.savez(f"{WORK}/work/sets/{r['log']}.npz", stub=np.zeros(1))
        json.dump({"per_log_stores": {"sets": 15808}, "checks": {"served_missing_from_bank": 0, "ckpt_step_differs_from_served_keys": 0}}, open(f"{WORK}/run/plan/podsets_provenance.json", "w"))
        json.dump({"PASS": True}, open(f"{WORK}/run/verify_sets.json", "w"))
        D = BB.load_s0_driver(self.cand)
        self.assertEqual((D.SRC, D.CKPT, D.CKPT_SHA256), (self.cand["root"], self.cand["ckpt"], self.cand["ckpt_sha256"]))
        self.assertTrue(D.C.OWNER.startswith("p6bank_unit"))
        ns = argparse.Namespace(tag="main", device="cpu", smoke_logs="", min_free_ram_mb=4500, no_stage=True)
        ctx = D.Ctx(ns)
        self.assertEqual(ctx.wd.replace("\\", "/"), f"{WORK}/run")
        mp2 = D.stage_preflight(ctx)
        self.assertEqual(mp2["n_sets"], BB.N_SETS)
        pf = json.load(open(f"{WORK}/run/preflight.json", encoding="utf-8"))
        self.assertTrue(pf["PASS"])
        self.assertEqual(pf["ckpt_sha256"], BB.P2_CKPT_SHA256)
        self.assertFalse(os.path.exists(f"{BB.S0_ROOT}/run/preflight.json.tmp"))
        # a checkpoint whose sha256 differs from the one `prepare` recorded is refused by the landed preflight (rc1)
        D.CKPT_SHA256 = "0" * 64
        with self.assertRaises(D.Stop):
            D.stage_preflight(ctx)
        # and `dump --dry-run` now passes the wrapper's own gates and imports the driver with the overrides
        self.assertEqual(BB.cmd_dump(argparse.Namespace(root=WORK, device="cpu", min_free_ram_mb=4500, no_stage=True, dry_run=True)), 0)

    def test_manifest_refuses_without_a_finished_dump(self):
        with self.assertRaises(BB.BuildError):
            BB.cmd_manifest(argparse.Namespace(root=WORK))

    def test_verify_sets_flags_a_label_that_changed_on_the_pod(self):
        import ep_admission as A
        rows, _ = A.read_ranked(f"{WORK}/run/plan/ranked_fit_logs.tsv", expected_sha256=BB.RANKED_LOGS_SHA256)
        p2 = A.load_bank_units(BB.P2_BANK, rows)
        lg = rows[0]["log"]
        idx = [i for i, k in enumerate(p2["keys"]) if k[0] == lg]
        tok = np.array([p2["keys"][i][1] for i in idx])
        store = dict(token=tok, step=np.array([p2["keys"][i][2] for i in idx], np.int32), rank=np.array([p2["keys"][i][3] for i in idx], np.int32), ckpt_step=p2["ckpt_step"][idx], labels6=p2["lab"][idx].copy())
        for f in os.listdir(f"{WORK}/work/sets"):
            os.remove(f"{WORK}/work/sets/{f}")
        np.savez(f"{WORK}/work/sets/{lg}.npz", **store)
        rep_ok = self._verify_sets()
        self.assertEqual(rep_ok["sets_equal_to_S0"], len(idx))
        self.assertEqual(rep_ok["sets_differ_or_unknown"], 0)
        self.assertFalse(rep_ok["PASS"])                                                         # the other 116 logs have no store
        store["labels6"][0, 3, 2] = 0.5 if store["labels6"][0, 3, 2] != 0.5 else 0.25            # a label that changed
        np.savez(f"{WORK}/work/sets/{lg}.npz", **store)
        rep_bad = self._verify_sets()
        self.assertEqual(rep_bad["sets_differ_or_unknown"], 1)
        self.assertEqual(rep_bad["sets_equal_to_S0"], len(idx) - 1)

    def _verify_sets(self):
        BB.cmd_verify_sets(argparse.Namespace(root=WORK, p2_bank=BB.P2_BANK))
        return json.load(open(f"{WORK}/run/verify_sets.json", encoding="utf-8"))

    def test_launch_text_is_the_wmi_detached_pattern_and_never_bypasses_policy(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            BB.cmd_launch(argparse.Namespace(root=WORK))
        t = buf.getvalue()
        self.assertIn("Win32_Process", t)
        self.assertIn("0x08000200", t)
        self.assertIn("< NUL", t)
        self.assertIn("dump --root", t)
        self.assertIn("--device cuda", t)
        self.assertNotIn("-ExecutionPolicy", t.replace("-ExecutionPolicy Bypass is never used", ""))


class TestPure(unittest.TestCase):
    def test_pick_margin_is_top1_minus_top2(self):
        z = np.full((64, 6), -5.0)
        z[:, :5] = np.array([2.0, 2.0, 0.0, 0.0, 0.0])[None, :]
        z[7, 2] = 3.0
        z[9, 2] = 1.0
        p = 1 / (1 + np.exp(-z))
        agg = p[:, 0] * p[:, 1] * (5 * p[:, 2] + 5 * p[:, 3] + 2 * p[:, 4]) / 12
        s = np.sort(agg)[::-1]
        self.assertAlmostEqual(BB._pick_margin(z), float(s[0] - s[1]), places=12)

    def test_the_registered_constants(self):
        self.assertEqual((BB.N_LOGS, BB.N_TOKENS, BB.N_SETS), (117, 8922, 15808))
        self.assertEqual(BB.P2_CKPT_SHA256, "ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51")
        self.assertEqual(BB.SMOKE_LOGIT_TOL, 5e-3)


if __name__ == "__main__":
    unittest.main()
