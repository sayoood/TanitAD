"""Tests for confirm_read.py (the confirm READ entry point: preflight -> navtest C2 picks -> the landed read -> C2 score -> four families -> final report) on SYNTHETIC worlds only.  stdlib unittest
(also collected by pytest):

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I test_confirm_read.py -v

The world is the C2 tests' `C2World` (the landed confirm tests' NavWorld + the C2 files) with PLAUSIBLE proposal geometry (smooth trajectories) and a matching navtest-inputs export (a logged human future),
so the REAL `families6.py` + NavSim adapter run in this process on the seams the LANDED `stage_read` writes.

(a) ORDER: the C2 picks are made while the ledger is ABSENT, the C2 score and the families only while it EXISTS; a RED C2-pass control or an unrunnable families tool refuses BEFORE the look is spent.
(b) AFTER THE LOOK: a failure raises PostLedgerError ("THE LOOK IS SPENT"), `post` resumes, `verify` lists what is missing; the final report is created once and is bound to the files it summarises.
(c) THE VERDICT is the registered bar applied mechanically; INCOMPLETE > CONTROL_RED > CLEARS > DOES_NOT_CLEAR, each with its regression arm.
(d) EVERY CHECK HAS A DELIBERATE-REGRESSION ARM (`MUTATION-TARGET-CRD-<X>`) with a coverage test.
(e) THE PRODUCTION RUNNER: one test runs the real tool through the SUBPROCESS runner (the call stage3_full makes); it is skipped unless SRDEV_SUBPROCESS=1 so the default run stays one Python process.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refe_head_confirm as C    # noqa: E402
import epr1_dev as D             # noqa: E402
import navtest_c2 as N           # noqa: E402
import confirm_families as F     # noqa: E402
import confirm_read as K         # noqa: E402
import test_navtest_c2 as TN     # noqa: E402
from test_epr1_dev import sha_f, wj   # noqa: E402

SRC = os.path.join(HERE, "confirm_read.py")
NP_ = D.R.N_PROPOSALS
SK = D.SEED_KEYS
MUTATED = set()


def rj(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_mutant(*edits, name="confirm_read_mutant"):
    with open(SRC, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for marker, old, new in edits:
        pat = re.compile(rf"MUTATION-TARGET-CRD-{re.escape(marker)}(?![A-Z0-9-])")
        idx = [i for i, ln in enumerate(lines) if pat.search(ln)]
        assert len(idx) == 1, f"marker {marker} sits on {len(idx)} lines"
        assert lines[idx[0]].count(old) == 1, f"{old!r} occurs {lines[idx[0]].count(old)} times on the {marker} line"
        lines[idx[0]] = lines[idx[0]].replace(old, new)
    mod = types.ModuleType(name)
    mod.__file__ = SRC
    sys.modules[name] = mod
    try:
        exec(compile("\n".join(lines), SRC, "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    return mod


def list_markers():
    with open(SRC, encoding="utf-8") as f:
        return sorted(set(re.findall(r"MUTATION-TARGET-CRD-([A-Z0-9-]+?)(?![A-Z0-9-])", f.read())))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the world
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class PWorld(TN.C2World):
    """C2World with smooth proposal trajectories (so families6 sees plausible poses) and the navtest-inputs export for its tokens."""

    def __init__(self, root, **kw):
        super().__init__(root, **kw)
        w = self.w
        rng = np.random.default_rng(11)
        n = self.N
        v0 = rng.uniform(1.0, 12.0, size=n)
        sf = rng.uniform(0.6, 1.4, size=(n, NP_))
        kap = rng.normal(0.0, 0.02, size=(n, NP_))
        t = 0.5 * np.arange(1, 9)
        x = (v0[:, None, None] * sf[:, :, None]) * t[None, None, :]
        props = np.stack([x, kap[:, :, None] * x ** 2 / 2.0, kap[:, :, None] * x], -1).astype(np.float32)
        w.props = props
        self.write_table()
        self.write_bank()
        ar = np.arange(n)
        np.savez(self.seam, token=w.token, fingerprint=self.fp, poses=props[ar, w.pick], sampling=np.array([8, 0.5]), arm=np.array("p2"))
        exp = {}
        for i, tok in enumerate(w.token):
            a, k0, v, xx, pts = rng.normal(0, 1.5), rng.choice([0.0, 0.0, 0.04, -0.04]) + rng.normal(0, 0.003), v0[i], 0.0, []
            for _ in range(8):
                v = max(0.0, v + a * 0.5)
                xx += v * 0.5
                pts.append([xx, k0 * xx * xx / 2.0, k0 * xx])
            exp[str(tok)] = {"stage": 1, "human_future_poses": pts, "ego_statuses": [{"ego_velocity": [float(v0[i]), 0.0]}], "log_name": str(w.log[i]),
                             "cv_poses": [[float(v0[i]) * 0.5 * k, 0.0, 0.0] for k in range(1, 9)]}
        self.export = os.path.join(root, "navtest_inputs.json.gz")
        with gzip.open(self.export, "wt", encoding="utf-8") as f:
            json.dump({"tokens": exp}, f)

    def cfg(self, **kw):
        base = dict(inputs=self.export, runner="inprocess", n_boot=200)
        base.update(kw)
        return F.FamiliesConfig(**base)


def tool_available() -> bool:
    try:
        F.resolve_tool(F.FamiliesConfig(inputs=__file__, runner="inprocess"))
    except F.FamiliesRefusal:
        return False
    return os.path.exists("D:/Projects/TanitAD/taniteval/adapters/navsim.py")


class KBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="crd_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self._k = 0

    def world(self, **kw):
        self._k += 1
        return PWorld(os.path.join(self.tmp, f"pw{self._k}"), **kw)

    def needs_tool(self):
        if not tool_available():
            self.skipTest("families6.py or the NavSim adapter is not on this machine (SKIPPED, not passed)")

    def mut(self, marker, old, new, scenario):
        MUTATED.add(marker)
        self.assertTrue(scenario(K), f"{marker}: the scenario must fire on the UNMUTATED module")
        mod = load_mutant((marker, old, new))
        self.assertFalse(scenario(mod), f"{marker}: the one-line mutant must make the check stop firing")


def hand_inputs(landed=True, fam_clears=True, c2_pass=True, complete=True, arm="EP-R1", fam="EPR1"):
    seeds = ("0", "1")
    res = {"primaries": {arm: {"both_seeds_clear_all_clauses_except_families6": landed, "per_seed": {sd: {"clears_without_families6": landed} for sd in seeds}, "other_primary_beside_it": {"x": 1}}},
           "readings_open": ["P2", fam], "readings": {}, "look": {"spent": True}}
    c2 = {"arms": {arm: {"pass": c2_pass, "vacuous_seeds": []}}, "c2_pass_all_arms": c2_pass, "red_arms": [] if c2_pass else [arm], "consequence": {}}
    names = ["P2"] + [f"{fam}_s{sd}" for sd in seeds]
    famrep = {"arms": {arm: {"both_seeds_clear_families6_clause": fam_clears, "per_seed_clears": {sd: fam_clears for sd in seeds}}},
              "readings": {n: {"status": "COMPUTED", "complete": complete, "longitudinal": {"complete": complete}, "lateral": {"complete": complete}, "tactical": {"complete": complete},
                               "strategic": {"status": "PAUSED"}, "vs_p2": {"clears": fam_clears}} for n in names},
              "all_readings_reported": True, "all_complete": complete, "not_computed": []}
    return res, c2, famrep


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (c) the verdict
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Verdict(KBase):
    def v(self, mod=K, **kw):
        res, c2, fam = hand_inputs(**kw)
        return mod.arm_verdict("EP-R1", "EPR1", res, c2, fam)["candidate_verdict"]

    def test_the_matrix(self):
        self.assertEqual(self.v(), "CLEARS_ALL_REGISTERED_CLAUSES")
        self.assertEqual(self.v(landed=False), "DOES_NOT_CLEAR")
        self.assertEqual(self.v(fam_clears=False), "DOES_NOT_CLEAR")
        self.assertEqual(self.v(landed=False, fam_clears=False), "DOES_NOT_CLEAR")
        self.assertEqual(self.v(c2_pass=False), "CONTROL_RED_RULING_NEEDED")
        self.assertEqual(self.v(c2_pass=False, landed=False), "CONTROL_RED_RULING_NEEDED", "a RED control outranks the clauses: the numbers are not quotable until the ruling")
        self.assertEqual(self.v(complete=False), "INCOMPLETE")
        self.assertEqual(self.v(complete=False, c2_pass=False), "INCOMPLETE", "a missing family outranks everything")
        res, c2, fam = hand_inputs()
        self.assertEqual(K.arm_verdict("EP-R1", "EPR1", res, None, fam)["candidate_verdict"], "INCOMPLETE")
        self.assertEqual(K.arm_verdict("EP-R1", "EPR1", res, c2, None)["candidate_verdict"], "INCOMPLETE")
        self.assertEqual(K.arm_verdict("SR-TTC", "SRTTC", res, c2, fam)["candidate_verdict"], "INCOMPLETE", "an arm not in the look")
        d = K.arm_verdict("EP-R1", "EPR1", res, c2, fam)
        self.assertEqual(d["parts"]["navtest_c2_pass"], True)
        self.assertIn(d["candidate_verdict"], K.VERDICTS)

    def test_reasons_name_the_failing_clause(self):
        res, c2, fam = hand_inputs(landed=False, fam_clears=False)
        r = K.arm_verdict("EP-R1", "EPR1", res, c2, fam)["reasons"]
        self.assertEqual(len(r), 2)
        self.assertTrue("landed clauses" in r[0] and "EP component" in r[0] and "families6 clause" in r[1])
        res, c2, fam = hand_inputs(landed=False, fam_clears=True, arm="SR-TTC", fam="SRTTC")
        r = K.arm_verdict("SR-TTC", "SRTTC", res, c2, fam)["reasons"]
        self.assertEqual(len(r), 1)
        self.assertNotIn("EP component", r[0])

    def test_each_branch_has_a_regression_arm(self):
        self.mut("V-CLEARS", "if not fails:", "if True:", lambda m: self.v(m, fam_clears=False) == "DOES_NOT_CLEAR")
        self.mut("V-C2RED", "if not ca[\"pass\"]:", "if False:", lambda m: self.v(m, c2_pass=False) == "CONTROL_RED_RULING_NEEDED")
        self.mut("V-INCOMPLETE", "if famrep is None or fa is None or inc:", "if False:", lambda m: self.v(m, complete=False) == "INCOMPLETE")

    def test_assemble_final_completeness_and_the_pieces(self):
        res, c2, fam = hand_inputs()
        res["readings"] = {"JOINT_s0": {"gain_x100": {"point": 1.0}, "label": "SECONDARY"}}
        f = K.assemble_final(res, c2, fam, {"h": "x"}, {"n": 1})
        self.assertTrue(f["completeness"]["complete"])
        self.assertEqual(f["strategic"]["status"], "PAUSED")
        self.assertEqual(f["arms"]["EP-R1"]["candidate_verdict"], "CLEARS_ALL_REGISTERED_CLAUSES")
        self.assertEqual(f["arms"]["EP-R1"]["other_primary_beside_it"], {"x": 1})
        self.assertEqual(f["joint_secondary"]["JOINT_s0"]["label"], "SECONDARY")
        self.assertEqual((f["notes"], f["hashes"]), ({"n": 1}, {"h": "x"}))
        self.assertIn("CANDIDATE", f["verdict_note"])
        self.assertFalse(K.assemble_final(res, None, fam, {})["completeness"]["complete"])
        self.assertFalse(K.assemble_final(res, c2, None, {})["completeness"]["complete"])
        fam["readings"]["P2"]["strategic"] = {"status": "OK"}
        self.assertFalse(K.assemble_final(res, c2, fam, {})["completeness"]["strategic_printed_paused_on_every_reading"])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (a)(b) the run
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Spy:
    """records the order of the steps and, at each call, whether the ledger exists"""

    def __init__(self, nw):
        self.nw, self.events = nw, []

    def wrap(self, mod, name, label):
        orig = getattr(mod, name)

        def f(*a, **k):
            self.events.append((label, os.path.exists(self.nw.ledger)))
            return orig(*a, **k)
        return mock.patch.object(mod, name, f)


class Run(KBase):
    def test_the_full_run_in_order_and_the_final_report_is_complete_and_bound(self):
        self.needs_tool()
        nw = self.world()
        o = nw.opts()
        spy = Spy(nw)
        with spy.wrap(N, "run_c2_picks", "c2_picks"), spy.wrap(C, "stage_read", "read"), spy.wrap(N, "run_c2_score", "c2_score"), spy.wrap(F, "run_families", "families"):
            final = K.run_confirm_read(o, nw.cfg())
        self.assertEqual(spy.events, [("c2_picks", False), ("read", False), ("c2_score", True), ("families", True)], "C2 picks BEFORE the ledger; C2 score and families only AFTER it")
        for f in (nw.ledger, os.path.join(o.out_dir, "confirm_read_result.json"), os.path.join(o.out_dir, N.SUBDIR, N.PICKS_REPORT), os.path.join(o.out_dir, N.SUBDIR, N.SCORE_RESULT),
                  os.path.join(o.out_dir, F.FAMILIES_DIR, F.REPORT_NAME), os.path.join(o.out_dir, K.FINAL_NAME)):
            self.assertTrue(os.path.exists(f), f)
        self.assertTrue(final["completeness"]["complete"], final["completeness"])
        self.assertEqual(sorted(final["arms"]), ["EP-R1", "SR-TTC"])
        for arm, a in final["arms"].items():
            self.assertIn(a["candidate_verdict"], K.VERDICTS, arm)
            self.assertIsNotNone(a["parts"]["navtest_c2_pass"], arm)
            self.assertIn("other_primary_beside_it", a)
        self.assertTrue(final["navtest_c2"]["pass_all_arms"])
        self.assertEqual(final["notes"], {"c2_picks_made_after_the_look": False})
        self.assertEqual(sorted(final["families"]), ["EPR1_s0", "EPR1_s1", "JOINT_s0", "JOINT_s1", "P2", "SRTTC_s0", "SRTTC_s1"])
        self.assertTrue(all(v["strategic"] == "PAUSED" for v in final["families"].values()))
        v = K.verify(o)
        self.assertTrue(v["complete"], v)
        self.assertEqual(rj(os.path.join(o.out_dir, K.FINAL_NAME))["hashes"], K._hashes(o, None))
        # the landed result still carries what it carried (the wrapper added, it did not replace)
        res = rj(os.path.join(o.out_dir, "confirm_read_result.json"))
        self.assertTrue(res["look"]["spent"])
        self.assertIn("families6", json.dumps(res["primaries"]["EP-R1"]["families6"]))
        # the family-6 clause of the final report equals the one in the families report
        fam = rj(os.path.join(o.out_dir, F.FAMILIES_DIR, F.REPORT_NAME))
        for arm in ("EP-R1", "SR-TTC"):
            self.assertEqual(final["arms"][arm]["parts"]["families6_clause_both_seeds"], fam["arms"][arm]["both_seeds_clear_families6_clause"])
            self.assertEqual(final["arms"][arm]["parts"]["landed_clauses_both_seeds"], res["primaries"][arm]["both_seeds_clear_all_clauses_except_families6"])

    def test_a_red_c2_in_the_final_report_is_control_red_not_a_clear(self):
        self.needs_tool()
        nw = self.world(ep_c2=(2.5, 2.5))
        final = K.run_confirm_read(nw.opts(), nw.cfg())
        self.assertEqual(final["arms"]["EP-R1"]["candidate_verdict"], "CONTROL_RED_RULING_NEEDED")
        self.assertNotEqual(final["arms"]["SR-TTC"]["candidate_verdict"], "CONTROL_RED_RULING_NEEDED")
        self.assertEqual(final["navtest_c2"]["red_arms"], ["EP-R1"])

    # ---- before the look
    def test_an_unrunnable_families_tool_refuses_before_the_look_is_spent(self):
        nw = self.world()
        bad = nw.cfg(families6=os.path.join(nw.root, "nope.py"))

        def scen(mod):
            reads = []
            with mock.patch.object(C, "stage_read", lambda *a, **k: reads.append(1)):
                try:
                    mod.run_confirm_read(nw.opts(), bad)
                except C.ConfirmRefusal as ex:
                    return "families6.py" in str(ex) and not os.path.exists(nw.ledger) and not reads
                except Exception:                                                                 # noqa: BLE001
                    return False
            return False
        self.mut("PF-TOOL", "tool = F.resolve_tool(cfg)", "tool = None", scen)

    def test_a_spent_look_refuses_the_read_and_points_at_post_and_verify(self):
        nw = self.world()
        os.makedirs(os.path.dirname(nw.ledger), exist_ok=True)
        wj(nw.ledger, {"schema": C.SCHEMA})
        with self.assertRaises(C.ConfirmRefusal) as cm:
            K.preflight(nw.opts(), nw.cfg())
        self.assertIn("SPENT", str(cm.exception))
        self.assertIn("post", str(cm.exception))
        self.mut("PF-SPENT", "if C._io_exists(guard, opts.look_ledger):", "if False:", lambda m: _preflight_refuses(m, nw))

    def test_preflight_needs_a_ledger_path_and_a_fresh_out_dir(self):
        nw = self.world()
        with self.assertRaises(C.ConfirmRefusal):
            K.preflight(nw.opts(look_ledger=None), nw.cfg())
        o = nw.opts()
        os.makedirs(o.out_dir)
        wj(os.path.join(o.out_dir, K.FINAL_NAME), {})
        with self.assertRaises(C.ConfirmRefusal) as cm:
            K.preflight(o, nw.cfg())
        self.assertIn("fresh output directory", str(cm.exception))

    def test_a_red_c2_control_refuses_before_the_look_and_the_landed_read_never_runs(self):
        nw = self.world(ep_c2=(0.0, 0.0), ttc_c2=(0.0, 0.0))                          # identity C2 rows: the reach control is RED
        reads = []
        with mock.patch.object(C, "stage_read", lambda *a, **k: reads.append(1)):
            with self.assertRaises(N.C2Refusal) as cm:
                K.run_confirm_read(nw.opts(), nw.cfg(families6=SRC))                  # SRC exists, so the preflight's tool check passes and the C2 control is what refuses
        self.assertEqual(reads, [])
        self.assertFalse(os.path.exists(nw.ledger))
        self.assertIn("UNSPENT", str(cm.exception))

    def test_the_c2_picks_are_what_puts_c2_in_the_final_report(self):
        self.needs_tool()

        def scen(mod):
            nw2 = self.world()
            try:
                f = mod.run_confirm_read(nw2.opts(), nw2.cfg())
            except Exception:                                                                    # noqa: BLE001
                return False
            return f["notes"]["c2_picks_made_after_the_look"] is False
        self.mut("C2PICKS", "N.run_c2_picks(opts, c2_out)", "None", scen)

    # ---- after the look
    def test_a_failure_after_the_look_is_loud_resumable_and_the_final_report_is_created_once(self):
        self.needs_tool()
        nw = self.world()
        o = nw.opts()
        with mock.patch.object(F, "run_families", side_effect=RuntimeError("boom")):
            with self.assertRaises(K.PostLedgerError) as cm:
                K.run_confirm_read(o, nw.cfg())
        self.assertIn("THE LOOK IS SPENT", str(cm.exception))
        self.assertIn("post", str(cm.exception))
        self.assertTrue(os.path.exists(nw.ledger))
        self.assertFalse(os.path.exists(os.path.join(o.out_dir, K.FINAL_NAME)))
        v = K.verify(o)
        self.assertFalse(v["complete"])
        self.assertIn("families_report", v["missing"])
        self.assertIn("final_report", v["missing"])
        self.assertIn("post", v["next"])
        final = K.post(o, nw.cfg())
        self.assertTrue(final["completeness"]["complete"])
        self.assertEqual(final["notes"], {"c2_picks_made_after_the_look": False})
        self.assertTrue(K.verify(o)["complete"])
        with self.assertRaises(C.ConfirmRefusal) as cm2:
            K.post(o, nw.cfg())
        self.assertIn("created once", str(cm2.exception))
        self.mut("POST-ONCE", "if C._io_exists(guard, os.path.join(opts.out_dir, FINAL_NAME)):", "if False:", lambda m: _post_refuses(m, o, nw.cfg()))

    def test_the_final_report_is_created_exclusively_even_if_the_pre_check_is_raced(self):
        self.needs_tool()
        nw = self.world()
        o = nw.opts()
        with mock.patch.object(F, "run_families", side_effect=RuntimeError("boom")):
            with self.assertRaises(K.PostLedgerError):
                K.run_confirm_read(o, nw.cfg())
        real = C._io_exists
        final_path = os.path.join(o.out_dir, K.FINAL_NAME)

        def scen(mod):
            wj(final_path, {"squatter": True})
            try:
                with mock.patch.object(C, "_io_exists", lambda g, p: False if p.replace("\\", "/").endswith(K.FINAL_NAME) else real(g, p)):
                    mod.post(o, nw.cfg())
            except FileExistsError:
                return True
            except Exception:                                                                     # noqa: BLE001
                return False
            finally:
                os.remove(final_path)
            return False
        self.mut("FINAL-EXCL", "exclusive=True)", "exclusive=False)", scen)

    def test_the_landed_cli_run_directly_leaves_a_look_that_post_can_finish_and_discloses_it(self):
        self.needs_tool()
        nw = self.world()
        o = nw.opts()
        C.stage_read(o)                                                                          # the bypass: nobody ran the C2 picks first
        v = K.verify(o)
        self.assertFalse(v["complete"])
        self.assertEqual(set(v["missing"]), {"navtest_c2_picks_report", "navtest_c2_result", "families_report", "final_report"})
        final = K.post(o, nw.cfg())
        self.assertTrue(final["notes"]["c2_picks_made_after_the_look"], "disclosed: made after the look (no outcome is involved in the picks)")
        self.assertTrue(K.verify(o)["complete"])

    def test_verify_notices_a_piece_edited_after_the_final_report(self):
        self.needs_tool()
        nw = self.world()
        o = nw.opts()
        K.run_confirm_read(o, nw.cfg())
        p = os.path.join(o.out_dir, N.SUBDIR, N.SCORE_RESULT)
        d = rj(p)
        d["c2_pass_all_arms"] = not d["c2_pass_all_arms"]
        wj(p, d)

        def scen(mod):
            v = mod.verify(o)
            return v["complete"] is False and v["final_bound_to_these_files"]["navtest_c2_result_sha256"] is False
        self.mut("VERIFY", "bool(not missing and bound and all(bound.values()))", "bool(not missing)", scen)

    def test_verify_before_anything_says_to_run_read(self):
        nw = self.world()
        v = K.verify(nw.opts())
        self.assertFalse(v["complete"])
        self.assertIn("read", v["next"])
        self.assertEqual(sorted(v["missing"]), ["families_report", "final_report", "landed_read_result", "ledger", "navtest_c2_picks_report", "navtest_c2_result"])


def _preflight_refuses(mod, nw) -> bool:
    try:
        mod.preflight(nw.opts(), nw.cfg())
    except C.ConfirmRefusal as ex:
        return "SPENT" in str(ex)
    except Exception:                                                                             # noqa: BLE001
        return False
    return False


def _post_refuses(mod, o, cfg) -> bool:
    try:
        mod.post(o, cfg)
    except C.ConfirmRefusal as ex:
        return "created once" in str(ex)
    except FileExistsError:
        return False
    except Exception:                                                                             # noqa: BLE001
        return False
    return False


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (e) the CLI and the production runner
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Cli(KBase):
    def flags(self, nw):
        o = nw.opts()
        return ["--bank-dir", nw.bank_dir, "--out-dir", o.out_dir, "--ep-run-dir", nw.w.run_dir, "--ep-registered-result-sha256", nw.w.r1_sha(), "--ep-dev-result", nw.ep_dev_path,
                "--ttc-run-dir", nw.w.ttc_dir, "--ttc-registered-result-sha256", nw.ttc_sha, "--ttc-dev-result", nw.ttc_dev_path, "--table", nw.table, "--p2-ckpt", nw.w.ckpt,
                "--banked-p2-seam", nw.seam, "--look-ledger", nw.ledger, "--inputs", nw.export, "--runner", "inprocess", "--n-boot", "200"]

    def test_every_flag_reaches_its_config(self):
        ap = K.build_parser()
        a = ap.parse_args(["read", "--bank-dir", "B", "--look-ledger", "L", "--c2-out", "X", "--families6", "F", "--inputs", "I", "--python", "P", "--n-boot", "77", "--runner", "inprocess"])
        self.assertEqual((a.cmd, a.c2_out, a.families6, a.inputs, a.python, a.n_boot, a.runner), ("read", "X", "F", "I", "P", 77, "inprocess"))
        for cmd in ("post", "verify"):
            self.assertEqual(ap.parse_args([cmd, "--bank-dir", "B"]).cmd, cmd)
        with self.assertRaises(SystemExit):
            ap.parse_args(["picks", "--bank-dir", "B"])

    def test_exit_codes_read_refused_post_ledger_failure_and_verify(self):
        self.needs_tool()
        nw = self.world()
        real = C.ConfirmOptions

        def patched(**kw):
            kw.update(table_sha256=sha_f(nw.table), expect_tokens=nw.N, expect_logs=nw.n_logs)
            return real(**kw)
        fl = self.flags(nw)
        with mock.patch.object(N.C, "ConfirmOptions", patched):
            self.assertEqual(K.main(["verify"] + fl), 1, "nothing exists yet")
            self.assertEqual(K.main(["post"] + fl), 2, "post on an unspent look is refused")
            with mock.patch.object(F, "run_families", side_effect=RuntimeError("boom")):
                self.assertEqual(K.main(["read"] + fl), 3, "a failure after the look was spent is exit 3")
            self.assertEqual(K.main(["read"] + fl), 2, "a second read is refused: the look is spent")
            self.assertEqual(K.main(["verify"] + fl), 1)
            self.assertEqual(K.main(["post"] + fl), 0)
            self.assertEqual(K.main(["verify"] + fl), 0)
            self.assertEqual(K.main(["post"] + fl), 2, "the final report is created once")

    @unittest.skipUnless(os.environ.get("SRDEV_SUBPROCESS") == "1", "set SRDEV_SUBPROCESS=1 to run the real tool through the production SUBPROCESS runner (adds a second live process)")
    def test_the_production_subprocess_runner_runs_the_real_tool_and_agrees_with_the_inprocess_runner(self):
        self.needs_tool()
        nw = self.world()
        o_a, o_b = nw.opts(), None
        K.run_confirm_read(o_a, nw.cfg(runner="inprocess"))
        fam_a = rj(os.path.join(o_a.out_dir, F.FAMILIES_DIR, F.REPORT_NAME))
        fdir = os.path.join(o_a.out_dir, "families_subprocess")
        rep_b = F.run_families(o_a, nw.cfg(runner="subprocess"), out_dir=fdir)
        for nm in fam_a["readings"]:
            self.assertEqual(json.dumps(fam_a["readings"][nm], sort_keys=True, default=str), json.dumps(rep_b["readings"][nm], sort_keys=True, default=str), nm)
        self.assertTrue(rep_b["all_complete"])


class ZZCoverage(unittest.TestCase):
    def test_every_marker_in_the_source_was_mutated_by_a_test_in_this_run(self):
        missing = sorted(set(list_markers()) - MUTATED)
        self.assertEqual(missing, [], f"markers with no mutation test: {missing}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
