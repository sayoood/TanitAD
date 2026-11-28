"""Tests for confirm_families.py (the four metric families on the confirm seams) on SYNTHETIC seams only (never the real navtest seams).  stdlib unittest (also collected by pytest):

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I test_confirm_families.py -v

The REAL tool runs here: `families6.py` (FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/code/) with the REAL NavSim adapter, in this process (`runner="inprocess"`: one Python
process; the production runner is the subprocess the landed `stage3_full.compute_families` uses), on seams built here from a synthetic export with a logged human future.  When the tool or the adapter is not
on the machine those tests report SKIPPED with the reason; they never pass silently.

(a) EQUIVALENCE: the ported `fam_adverse` / `fam_direction` equal `proxy_eval.py`'s, extracted from ITS SOURCE, on every ordered pair of the banked families JSONs and on hand-built pairs of every direction.
(b) KNOWN VALUES: hand-built interval pairs; the report of the banked P2 families JSON (a control: complete, strategic PAUSED, distance-keeping UNAVAILABLE with its reason and n = 0).
(c) EVERY CHECK HAS A DELIBERATE-REGRESSION ARM (`MUTATION-TARGET-FAM-<X>`), with a coverage test.
(d) COMPLETENESS: every reading of the look gets a report; a reading the tool did not produce is NOT COMPUTED and never a pass; a missing metric makes the reading incomplete.
(e) REACHABILITY: the argv equals the one `stage3_full.compute_families` builds (from its source); the tool is invoked with the recorded seam and label; every CLI flag reaches its config; no file-system call outside `_io_*`.
"""
from __future__ import annotations

import ast
import copy
import glob
import gzip
import itertools
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
import confirm_families as F     # noqa: E402
from test_epr1_dev import FS_CALLS, _dotted, sha_f, wj   # noqa: E402

SRC = os.path.join(HERE, "confirm_families.py")
REFE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
PROXY_EVAL = os.path.join(REFE, "eval", "proxy_eval.py")
STAGE3 = os.path.join(REFE, "eval", "stage3_full.py")
BANKED = os.path.join(REFE, "raw", "2026-10-18-sel-rank", "raw", "families")
MUTATED = set()


def rj(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_mutant(*edits, name="confirm_families_mutant"):
    with open(SRC, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for marker, old, new in edits:
        pat = re.compile(rf"MUTATION-TARGET-FAM-{re.escape(marker)}(?![A-Z0-9-])")
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
        return sorted(set(re.findall(r"MUTATION-TARGET-FAM-([A-Z0-9-]+?)(?![A-Z0-9-])", f.read())))


def banked(name):
    p = os.path.join(BANKED, name)
    return rj(p) if os.path.exists(p) else None


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# a synthetic export + seams (the human future is logged; the seams are the human future plus noise of a chosen size)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def make_scene(n=120, n_logs=10, seed=3):
    rng = np.random.default_rng(seed)
    toks = [f"tok{i:05d}" for i in range(n)]
    logs = [f"2021.05.{(i % n_logs) + 1:02d}.10.00.00_veh-{i % n_logs}_00001_00200" for i in range(n)]
    human, exp = [], {}
    for i, t in enumerate(toks):
        v0 = rng.uniform(1.0, 12.0)
        a = rng.normal(0, 1.5)
        kap = rng.choice([0.0, 0.0, 0.0, 0.04, -0.04]) + rng.normal(0, 0.003)
        x, v, pts = 0.0, v0, []
        for k in range(1, 9):
            v = max(0.0, v + a * 0.5)
            x += v * 0.5
            pts.append([x, kap * x * x / 2.0, kap * x])
        human.append(pts)
        exp[t] = {"stage": 1, "human_future_poses": pts, "ego_statuses": [{"ego_velocity": [v0, 0.0]}], "log_name": logs[i], "cv_poses": [[v0 * 0.5 * k, 0.0, 0.0] for k in range(1, 9)]}
    return toks, np.array(human, np.float64), exp


class Scene:
    """export + the read-out directory the landed `stage_read` leaves behind: confirm_read_result.json (with `seams`), the ledger, and seams/seam_<name>.npz."""

    def __init__(self, root, noise=None, n=120, seed=3):
        self.root = root
        self.toks, self.human, exp = make_scene(n=n, seed=seed)
        self.n = n
        os.makedirs(root, exist_ok=True)
        self.export = os.path.join(root, "navtest_inputs.json.gz")
        with gzip.open(self.export, "wt", encoding="utf-8") as f:
            json.dump({"tokens": exp}, f)
        self.out = os.path.join(root, "read_out")
        self.seam_dir = os.path.join(self.out, "seams")
        os.makedirs(self.seam_dir, exist_ok=True)
        self.ledger = os.path.join(root, "look", "NAVTEST_LOOK_SPENT.json")
        os.makedirs(os.path.dirname(self.ledger), exist_ok=True)
        # the noise per reading: (sx, sy, sh).  EPR1_s1 is deliberately bad laterally and longitudinally.
        self.noise = noise or {"P2": (0.05, 0.05, 0.01), "EPR1_s0": (0.05, 0.05, 0.01), "EPR1_s1": (0.05, 1.5, 0.2), "SRTTC_s0": (0.05, 0.05, 0.01), "SRTTC_s1": (0.05, 0.05, 0.01),
                               "JOINT_s0": (0.05, 0.05, 0.01), "JOINT_s1": (0.05, 0.05, 0.01)}
        self.names = list(self.noise)
        rng = np.random.default_rng(5)
        self.seams = {}
        for nm, (sx, sy, sh) in self.noise.items():
            p = (self.human + rng.normal(size=self.human.shape) * np.array([sx, sy, sh])).astype(np.float32)
            path = os.path.join(self.seam_dir, f"seam_{nm}.npz").replace("\\", "/")
            np.savez(path, token=np.array(self.toks), fingerprint=np.array(self.toks), poses=p, sampling=np.array([8, 0.5]), arm=np.array(f"REFE_HEAD_{nm}"))
            self.seams[nm] = {"path": path, "sha256": sha_f(path), "label": "refe_head_" + nm.lower()}
        self.write_read_result()
        self.write_ledger()

    def write_read_result(self, drop=()):
        seams = {k: v for k, v in self.seams.items() if k not in drop}
        wj(os.path.join(self.out, "confirm_read_result.json"), {"schema": C.SCHEMA, "stage": "READ", "seams": seams, "estimator": {"n_tokens": self.n}, "p2_pick": {"pdms_x100": 0.0}})

    def write_ledger(self):
        wj(self.ledger, {"schema": C.SCHEMA, "spent": "2026-10-19T00:00:00+0200", "readings": ["P2", "EPR1", "SRTTC", "JOINT"], "picks_sha256": {nm: "0" * 64 for nm in self.names}})

    def opts(self, **kw):
        base = dict(out_dir=self.out, bank_dir=os.path.join(self.root, "no_bank"), look_ledger=self.ledger)
        base.update(kw)
        return C.ConfirmOptions(**base)

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


class FBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cfam_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self._k = 0

    def scene(self, **kw):
        self._k += 1
        return Scene(os.path.join(self.tmp, f"s{self._k}"), **kw)

    def needs_tool(self):
        if not tool_available():
            self.skipTest("families6.py or the NavSim adapter is not on this machine (SKIPPED, not passed)")

    def mut(self, marker, old, new, scenario):
        MUTATED.add(marker)
        self.assertTrue(scenario(F), f"{marker}: the scenario must fire on the UNMUTATED module")
        mod = load_mutant((marker, old, new))
        self.assertFalse(scenario(mod), f"{marker}: the one-line mutant must make the check stop firing")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (a) equivalence with proxy_eval's source
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def proxy_namespace():
    """FAM_SKIP, fam_direction and fam_adverse exactly as PROXY_EVAL's source defines them (extracted by AST; the module itself is not imported: it imports the model stack)."""
    with open(PROXY_EVAL, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in ("fam_direction", "fam_adverse"))
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FAM_SKIP" for t in n.targets))]
    assert [getattr(n, "name", "FAM_SKIP") for n in keep].count("fam_adverse") == 1
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), PROXY_EVAL, "exec"), ns)
    return ns


def comp(lo, hi, mean=None):
    return {"mean": (lo + hi) / 2 if mean is None else mean, "lo": lo, "hi": hi}


def fj(long=None, lat=None, tac=None):
    return {"families": {F.MODEL_KEY: {"longitudinal": {"ci": {"components": long or {}}}, "lateral": {"ci": {"components": lat or {}}}, "tactical": tac or {}}}}


class Equivalence(FBase):
    def setUp(self):
        super().setUp()
        if not os.path.exists(PROXY_EVAL):
            self.skipTest("proxy_eval.py not found beside the package (SKIPPED, not passed)")
        self.ns = proxy_namespace()

    def test_fam_direction_equals_the_originals_on_every_metric_name_the_banked_blocks_carry(self):
        names = set()
        for p in glob.glob(os.path.join(BANKED, "*_families.json")):
            blk = rj(p)["families"][F.MODEL_KEY]
            for fam in ("longitudinal", "lateral"):
                names |= set(blk[fam]["ci"]["components"])
        names |= {"accel_mae_mps2", "target_speed_acc.within_1.0_mps", "speed_bias_mps", "ego_progress.progress_ratio_mean", "accuracy", "kappa", "x_error", "x_rmse"}
        self.assertGreater(len(names), 20)
        for m in sorted(names):
            self.assertEqual(F.fam_direction(m), self.ns["fam_direction"](m), m)
        self.assertEqual(F.fam_direction("accel_mae_mps2"), "lower", "the 2026-09-28 substring bug: a LOWER acceleration error is better")
        self.assertEqual(F.FAM_SKIP, self.ns["FAM_SKIP"])

    def test_fam_adverse_equals_the_original_on_every_ordered_pair_of_banked_families_jsons(self):
        files = sorted(glob.glob(os.path.join(BANKED, "*_families.json")))
        if len(files) < 3:
            self.skipTest("the banked families JSONs are not beside the package (SKIPPED, not passed)")
        docs = {os.path.basename(p): rj(p) for p in files}
        n = 0
        for a, b in itertools.permutations(docs, 2):
            self.assertEqual(F.fam_adverse(docs[a], docs[b]), self.ns["fam_adverse"](docs[a], docs[b]), (a, b))
            n += 1
        self.assertGreaterEqual(n, 12)
        self.assertTrue(any(F.fam_adverse(docs[a], docs[b])["adverse"] for a, b in itertools.permutations(docs, 2)), "the pairs must exercise the adverse branch")

    def test_fam_adverse_equals_the_original_on_hand_built_pairs_of_every_direction(self):
        metrics = ["speed_mae_mps", "target_speed_acc.within_1.0_mps", "speed_bias_mps", "ego_progress.progress_ratio_mean", "ego_progress.gt_progress_mean_m"]
        rng = np.random.default_rng(0)
        for _ in range(200):
            def block():
                out = {}
                for m in metrics:
                    c = float(rng.normal(0, 1))
                    w = float(abs(rng.normal(0, 0.5))) + 0.01
                    out[m] = comp(c - w, c + w)
                return out
            a, b = fj(long=block()), fj(long=block())
            self.assertEqual(F.fam_adverse(a, b), self.ns["fam_adverse"](a, b))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (b) known values
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class KnownValues(FBase):
    def test_each_direction_branch_by_hand(self):
        W = fj(long={"speed_mae_mps": comp(1.0, 2.0), "target_speed_acc.within_1.0_mps": comp(0.6, 0.7), "speed_bias_mps": comp(0.1, 0.2), "ego_progress.progress_ratio_mean": comp(1.1, 1.2)})
        ok = fj(long={"speed_mae_mps": comp(1.5, 2.5), "target_speed_acc.within_1.0_mps": comp(0.55, 0.65), "speed_bias_mps": comp(0.05, 0.25), "ego_progress.progress_ratio_mean": comp(1.0, 1.3)})
        self.assertEqual(F.fam_adverse(W, ok)["adverse"], [])
        self.assertEqual(F.fam_adverse(W, ok)["components_compared"], 4)
        worse = fj(long={"speed_mae_mps": comp(2.1, 3.0), "target_speed_acc.within_1.0_mps": comp(0.40, 0.55), "speed_bias_mps": comp(0.3, 0.5), "ego_progress.progress_ratio_mean": comp(1.3, 1.5)})
        got = {a["metric"]: a["direction"] for a in F.fam_adverse(W, worse)["adverse"]}
        self.assertEqual(got, {"speed_mae_mps": "lower", "target_speed_acc.within_1.0_mps": "higher", "speed_bias_mps": "magnitude", "ego_progress.progress_ratio_mean": "closer_to_1"})
        better = fj(long={"speed_mae_mps": comp(0.1, 0.5), "target_speed_acc.within_1.0_mps": comp(0.8, 0.9)})
        self.assertEqual(F.fam_adverse(W, better)["adverse"], [], "separation in the GOOD direction is not adverse")
        skip = fj(long={"ego_progress.gt_progress_mean_m": comp(1.0, 2.0)})
        self.assertEqual(F.fam_adverse(skip, fj(long={"ego_progress.gt_progress_mean_m": comp(5.0, 6.0)}))["components_compared"], 0, "properties of the human future are skipped")

    def test_vs_p2_clears_only_when_something_was_compared_and_nothing_is_adverse(self):
        W = fj(long={"speed_mae_mps": comp(1.0, 2.0)})
        self.assertTrue(F.vs_p2(W, fj(long={"speed_mae_mps": comp(1.2, 1.9)}))["clears"])
        self.assertFalse(F.vs_p2(W, fj(long={"speed_mae_mps": comp(2.5, 3.0)}))["clears"])
        empty = F.vs_p2(fj(), fj())
        self.assertEqual((empty["components_compared"], empty["evaluated"], empty["clears"]), (0, False, False), "zero components compared = NOT evaluated, never a clear")
        self.assertFalse(F.vs_p2(None, W)["clears"])
        self.assertFalse(F.vs_p2(W, None)["clears"])
        zero_compared = lambda m: m.vs_p2(fj(), fj())["clears"] is False
        self.mut("CLEARS", "r[\"evaluated\"] and not r[\"adverse\"]", "not r[\"adverse\"]", zero_compared)

    def test_the_banked_p2_families_json_is_a_complete_four_family_report(self):
        d = banked("P2_pick_families.json")
        if d is None:
            self.skipTest("banked P2 families JSON not found (SKIPPED, not passed)")
        rep = F.four_family_report("P2", d, None, 12146, baseline=True)
        self.assertEqual(rep["status"], "COMPUTED")
        self.assertTrue(rep["complete"] and rep["longitudinal"]["complete"] and rep["lateral"]["complete"] and rep["tactical"]["complete"])
        self.assertEqual(rep["strategic"]["status"], "PAUSED")
        self.assertFalse(rep["strategic"]["is_a_pass"])
        self.assertIn("item 25", rep["strategic"]["ruling"])
        dk = rep["longitudinal"]["items"]["distance_keeping"]
        self.assertEqual((dk["status"], dk["n"]), ("UNAVAILABLE", 0), "distance-keeping is UNAVAILABLE with its reason and its n")
        self.assertIn("lead-agent", dk["reason"])
        ts = rep["longitudinal"]["items"]["target_speed_accuracy"]
        self.assertEqual(ts["status"], "COMPUTED")
        self.assertEqual(ts["components"]["target_speed_acc.within_1.0_mps"]["value"], 0.7214)
        self.assertEqual(ts["components"]["target_speed_acc.within_1.0_mps"]["ci95"], [0.7078, 0.7348])
        self.assertEqual(rep["lateral"]["items"]["heading_mae_deg"]["components"]["heading_mae_deg"]["value"], 2.4839)
        self.assertEqual(rep["lateral"]["items"]["cross_mae_m"]["components"]["cross_mae_m"]["n_episodes"], 136)
        self.assertEqual(rep["tactical"]["items"]["lateral_decision"]["accuracy"], 0.9237)
        self.assertEqual(rep["tactical"]["items"]["maneuver_5way_collapsed"]["kappa"], 0.7145)
        self.assertEqual(rep["tactical"]["items"]["goal_setting"]["status"], "COMPUTED")
        self.assertEqual((rep["n_scored"], rep["all_tokens_scored"]), (12146, True))
        self.assertFalse(F.four_family_report("P2", d, None, 12147, baseline=True)["all_tokens_scored"], "a reading that scored fewer tokens than the look has is disclosed")
        self.assertEqual(rep["vs_p2"], {"baseline": True})
        self.assertEqual(rep["strategic"]["families6_own_status"], "UNAVAILABLE")
        self.mut("STRATEGIC", "\"status\": \"PAUSED\", \"ruling\"", "\"status\": \"OK\", \"ruling\"", lambda m: m.four_family_report("P2", d, None, 12146, baseline=True)["strategic"]["status"] == "PAUSED")
        self.mut("DK", "\"reason\": str(dk.get(\"reason\")), \"n\": dk.get(\"n\")}", "\"reason\": str(dk.get(\"reason\")), \"n\": None}", lambda m: m.four_family_report("P2", d, None, 12146, baseline=True)["complete"])

    def test_a_banked_arm_is_compared_against_the_banked_p2(self):
        p2, ar = banked("P2_pick_families.json"), banked("ORACLE_best64_families.json")
        if p2 is None or ar is None:
            self.skipTest("banked families JSONs not found (SKIPPED, not passed)")
        rep = F.four_family_report("ORACLE_best64", ar, p2, 12146)
        self.assertEqual(rep["vs_p2"]["adverse"], F.fam_adverse(p2, ar)["adverse"])
        self.assertGreater(rep["vs_p2"]["components_compared"], 15)
        self.assertEqual(rep["vs_p2"]["clears"], bool(rep["vs_p2"]["adverse"] == []))
        self.assertIn("tactical_no_interval", rep["vs_p2"])

    def test_a_metric_the_block_lacks_without_a_reason_is_MISSING_and_makes_the_reading_incomplete(self):
        d = banked("P2_pick_families.json")
        if d is None:
            self.skipTest("banked P2 families JSON not found (SKIPPED, not passed)")
        d2 = copy.deepcopy(d)
        del d2["families"][F.MODEL_KEY]["lateral"]["ci"]["components"]["heading_mae_deg"]
        rep = F.four_family_report("X", d2, None, 12146, baseline=True)
        self.assertEqual(rep["lateral"]["items"]["heading_mae_deg"]["status"], "MISSING")
        self.assertFalse(rep["lateral"]["complete"])
        self.assertFalse(rep["complete"])
        self.assertTrue(rep["longitudinal"]["complete"])
        self.mut("MISSING", "return {\"status\": \"MISSING\", \"reason\": f\"components", "return {\"status\": \"COMPUTED\", \"reason\": f\"components", lambda m: not m.four_family_report("X", d2, None, 12146, baseline=True)["complete"])
        # an item that is unavailable WITH a reason but without an n is also incomplete
        d3 = copy.deepcopy(d)
        d3["families"][F.MODEL_KEY]["longitudinal"]["distance_keeping"].pop("n")
        self.assertFalse(F.four_family_report("X", d3, None, 12146, baseline=True)["complete"])
        d4 = copy.deepcopy(d)
        del d4["families"][F.MODEL_KEY]["longitudinal"]["distance_keeping"]
        self.assertEqual(F.four_family_report("X", d4, None, 12146, baseline=True)["longitudinal"]["items"]["distance_keeping"]["status"], "MISSING")

    def test_the_tactical_family_is_part_of_completeness(self):
        d = banked("P2_pick_families.json")
        if d is None:
            self.skipTest("banked P2 families JSON not found (SKIPPED, not passed)")
        d2 = copy.deepcopy(d)
        del d2["families"][F.MODEL_KEY]["tactical"]["lateral_decision"]
        scen = lambda m: not m.four_family_report("X", d2, None, 12146, baseline=True)["complete"]
        self.assertEqual(F.four_family_report("X", d2, None, 12146, baseline=True)["tactical"]["items"]["lateral_decision"]["status"], "MISSING")
        self.mut("COMPLETE", "and rep[\"tactical\"][\"complete\"])", ")", scen)
        # a tactical block that is unavailable with a reason and n is complete (reported, not hidden)
        d3 = copy.deepcopy(d)
        d3["families"][F.MODEL_KEY]["tactical"] = {"status": "UNAVAILABLE", "reason": "no maneuver labels", "n": 12146}
        rep = F.four_family_report("X", d3, None, 12146, baseline=True)
        self.assertTrue(rep["tactical"]["complete"])
        self.assertTrue(all(v["status"] == "UNAVAILABLE" and v["n"] == 12146 and v["reason"] for v in rep["tactical"]["items"].values()))

    def test_an_undefined_kappa_with_a_reason_is_reported_not_missing(self):
        blk = {"status": "OK", "accuracy": 0.9, "kappa": None, "kappa_undefined_reason": "only one class present", "n": 5, "classes": ["a"]}
        it = F._decision_item(blk)
        self.assertEqual((it["status"], it["kappa"], it["kappa_undefined_reason"]), ("COMPUTED", None, "only one class present"))
        self.assertEqual(F._decision_item({"status": "OK", "accuracy": 0.9, "kappa": None, "n": 5})["status"], "MISSING")

    def test_a_reading_the_tool_did_not_produce_is_not_computed_and_never_a_pass(self):
        rep = F.four_family_report("EPR1_s0", None, {"families": {}}, 100)
        self.assertEqual(rep["status"], "NOT_COMPUTED")
        self.assertFalse(rep["complete"])
        self.assertFalse(rep["vs_p2"]["clears"])
        self.assertEqual(rep["strategic"]["status"], "PAUSED", "STRATEGIC is printed even when nothing else is")
        self.assertIn("no output", rep["reason"])
        rep = F.four_family_report("EPR1_s0", {"families": {F.MODEL_KEY: {"status": "RAISED", "error": "boom"}}}, None, 100)
        self.assertEqual(rep["status"], "NOT_COMPUTED")
        self.assertIn("boom", rep["reason"])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the REAL tool on synthetic seams
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class RealTool(FBase):
    def test_every_reading_of_the_look_gets_a_complete_four_family_report_from_the_real_tool(self):
        self.needs_tool()
        sc = self.scene()
        rep = F.run_families(sc.opts(), sc.cfg())
        self.assertEqual(sorted(rep["readings"]), sorted(sc.names))
        self.assertEqual(rep["readings_expected"][0], "P2")
        self.assertTrue(rep["all_readings_reported"] and rep["all_complete"], [n for n, r in rep["readings"].items() if not r["complete"]])
        self.assertEqual(rep["not_computed"], [])
        for nm, r in rep["readings"].items():
            self.assertEqual(r["strategic"]["status"], "PAUSED", nm)
            self.assertEqual(r["longitudinal"]["items"]["distance_keeping"]["status"], "UNAVAILABLE", nm)
            self.assertGreater(r["longitudinal"]["items"]["distance_keeping"]["reason"].__len__(), 20)
            self.assertEqual(r["longitudinal"]["items"]["distance_keeping"]["n"], 0)
            self.assertEqual(r["n_scored"], sc.n)
            self.assertTrue(r["all_tokens_scored"])
        self.assertEqual(rep["readings"]["P2"]["vs_p2"], {"baseline": True})
        fdir = os.path.join(sc.out, F.FAMILIES_DIR)
        for nm in sc.names:
            self.assertTrue(os.path.exists(os.path.join(fdir, f"confirm_{nm}_families.json")) and os.path.exists(os.path.join(fdir, f"confirm_{nm}_families.log")), nm)
        self.assertTrue(os.path.exists(os.path.join(fdir, F.REPORT_NAME)))
        self.assertEqual(rep["tool"]["families6_sha256"], F.FAMILIES6_SHA256)
        self.assertTrue(rep["tool"]["tool_is_the_pinned_one"])

    def test_a_reading_with_a_bad_lateral_path_separates_adverse_and_the_arm_clause_needs_both_seeds(self):
        self.needs_tool()
        sc = self.scene()
        rep = F.run_families(sc.opts(), sc.cfg())
        bad, good = rep["readings"]["EPR1_s1"]["vs_p2"], rep["readings"]["EPR1_s0"]["vs_p2"]
        self.assertFalse(bad["clears"])
        self.assertTrue(any(a["family"] == "lateral" and a["metric"] == "cross_mae_m" for a in bad["adverse"]), [a["metric"] for a in bad["adverse"]])
        self.assertTrue(good["clears"], good["adverse"])
        self.assertGreater(bad["components_compared"], 15)
        self.assertEqual(rep["arms"]["EP-R1"]["per_seed_clears"], {"0": True, "1": False})
        self.assertFalse(rep["arms"]["EP-R1"]["both_seeds_clear_families6_clause"])
        self.assertTrue(rep["arms"]["SR-TTC"]["both_seeds_clear_families6_clause"], rep["arms"]["SR-TTC"])
        # the same comparison through the ported function, from the files the tool wrote
        fdir = os.path.join(sc.out, F.FAMILIES_DIR)
        p2, e1 = rj(os.path.join(fdir, "confirm_P2_families.json")), rj(os.path.join(fdir, "confirm_EPR1_s1_families.json"))
        self.assertEqual(F.fam_adverse(p2, e1)["adverse"], bad["adverse"])
        self.mut("BOTH", "all(rs[sd][\"vs_p2\"].get(\"clears\") for sd in C.SEED_KEYS)", "any(rs[sd][\"vs_p2\"].get(\"clears\") for sd in C.SEED_KEYS)", lambda m: not m.run_families(sc.opts(), sc.cfg())["arms"]["EP-R1"]["both_seeds_clear_families6_clause"])

    def test_the_tool_is_invoked_with_the_recorded_seam_the_confirm_label_stage_1_and_the_n_boot(self):
        self.needs_tool()
        sc = self.scene()
        calls = []
        real = F._io_run_inprocess

        def spy(argv, log, cwd):
            calls.append((list(argv), cwd))
            return real(argv, log, cwd)
        with mock.patch.dict(F.RUNNERS, {"inprocess": spy}):
            F.run_families(sc.opts(), sc.cfg(n_boot=150))
        self.assertEqual(len(calls), len(sc.names))
        self.assertEqual(calls[0][0][1:], [calls[0][0][1], "--seam", sc.seams["P2"]["path"], "--inputs", os.path.abspath(sc.export), "--stage", "1", "--label", "confirm_P2", "--out",
                                           os.path.join(sc.out, F.FAMILIES_DIR, "confirm_P2_families.json"), "--n-boot", "150"], "P2 first: it is the paired baseline of every clause")
        self.assertEqual([c[0][c[0].index("--label") + 1] for c in calls], [f"confirm_{n}" for n in ["P2"] + sorted(x for x in sc.names if x != "P2")])
        self.assertTrue(all(c[1] == os.path.dirname(c[0][1]) for c in calls), "cwd is families6's own directory, as stage3_full does")

    def test_a_reading_for_which_the_tool_writes_nothing_is_not_computed_and_its_arm_does_not_clear(self):
        self.needs_tool()
        sc = self.scene()
        real = F._io_run_inprocess

        def flaky(argv, log, cwd):
            if "confirm_SRTTC_s1" in argv:
                with open(log, "w", encoding="utf-8") as f:
                    f.write("simulated crash\n")
                return 1
            return real(argv, log, cwd)
        with mock.patch.dict(F.RUNNERS, {"inprocess": flaky}):
            rep = F.run_families(sc.opts(), sc.cfg())
        self.assertEqual(rep["not_computed"], ["SRTTC_s1"])
        self.assertFalse(rep["all_complete"])
        self.assertEqual(rep["readings"]["SRTTC_s1"]["status"], "NOT_COMPUTED")
        self.assertFalse(rep["readings"]["SRTTC_s1"]["vs_p2"]["clears"])
        self.assertFalse(rep["arms"]["SR-TTC"]["both_seeds_clear_families6_clause"], "an arm with a reading the tool did not produce never clears")
        self.assertEqual(rep["tool_runs"]["SRTTC_s1"]["produced_output"], False)
        self.assertEqual(rep["tool_runs"]["SRTTC_s1"]["rc"], 1)
        self.assertTrue(rep["tool_runs"]["P2"]["produced_output"])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# refusals: before the tool runs
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Refusals(FBase):
    def runs(self, mod, sc, **kw):
        calls = []
        with mock.patch.dict(F.RUNNERS, {"inprocess": lambda a, l, c: calls.append(a) or 1}), mock.patch.dict(mod.RUNNERS, {"inprocess": lambda a, l, c: calls.append(a) or 1}):
            try:
                mod.run_families(sc.opts(**kw.get("opts", {})), sc.cfg(**kw.get("cfg", {})))
            except mod.FamiliesRefusal as ex:
                return str(ex), calls
            except Exception as ex:                                                              # noqa: BLE001
                return f"OTHER {type(ex).__name__}: {ex}", calls
        return None, calls

    def test_no_ledger_no_families_and_the_tool_never_runs(self):
        self.needs_tool()
        sc = self.scene()
        os.remove(sc.ledger)

        def scen(mod):
            msg, calls = self.runs(mod, sc)
            return bool(msg and "not spent" in msg and not calls)
        self.mut("LEDGER", "if not _io_exists(guard, opts.look_ledger):", "if False:", scen)

    def test_a_reading_of_the_look_without_a_seam_in_the_read_result_refuses_the_whole_step(self):
        self.needs_tool()
        sc = self.scene()
        sc.write_read_result(drop=("JOINT_s1",))

        def scen(mod):
            msg, calls = self.runs(mod, sc)
            return bool(msg and "JOINT_s1" in msg and not calls)
        self.mut("COMPLETENESS", "if missing:", "if False:", scen)

    def test_a_seam_edited_after_the_look_is_refused_before_the_tool_runs(self):
        self.needs_tool()
        sc = self.scene()
        p = sc.seams["EPR1_s0"]["path"]
        z = np.load(p)
        poses = z["poses"].copy()
        poses[0, 0, 0] += 1.0
        np.savez(p, token=z["token"], fingerprint=z["fingerprint"], poses=poses, sampling=z["sampling"], arm=z["arm"])

        def scen(mod):
            msg, calls = self.runs(mod, sc)
            return bool(msg and "EPR1_s0" in msg and "differs" in msg)
        self.mut("SEAMSHA", "if _io_sha256(guard, seam) != seams[nm][\"sha256\"]:", "if False:", scen)

    def test_an_unrunnable_tool_refuses_naming_what_is_missing(self):
        sc = self.scene()
        with self.assertRaises(F.FamiliesRefusal) as cm:
            F.resolve_tool(F.FamiliesConfig(families6=os.path.join(sc.root, "nope.py"), inputs=os.path.join(sc.root, "nope.json.gz"), runner="inprocess"))
        self.assertIn("families6.py", str(cm.exception))
        self.assertIn("navtest inputs export", str(cm.exception))
        with self.assertRaises(F.FamiliesRefusal) as cm:
            F.resolve_tool(F.FamiliesConfig(families6=SRC, inputs=sc.export, python=os.path.join(sc.root, "no_python.exe"), runner="subprocess"))
        self.assertIn("python interpreter", str(cm.exception))
        with self.assertRaises(F.FamiliesRefusal) as cm:
            F.resolve_tool(F.FamiliesConfig(families6=SRC, inputs=sc.export, runner="telepathy"))
        self.assertIn("runner", str(cm.exception))
        ok = F.resolve_tool(F.FamiliesConfig(families6=SRC, inputs=sc.export, runner="inprocess"))
        self.assertEqual(ok["families6"], os.path.abspath(SRC))

    def test_families_need_a_ledger_path(self):
        sc = self.scene()
        with self.assertRaises(F.FamiliesRefusal):
            F.run_families(sc.opts(look_ledger=None), sc.cfg())


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (e) reachability
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Reach(FBase):
    def test_the_argv_equals_the_one_stage3_full_compute_families_builds(self):
        if not os.path.exists(STAGE3):
            self.skipTest("stage3_full.py not found beside the package (SKIPPED, not passed)")
        with open(STAGE3, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "compute_families")
        call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and _dotted(n.func) == "subprocess.call")
        theirs = ["<v>" if not isinstance(e, ast.Constant) else e.value for e in call.args[0].elts]
        mine = F.families6_argv({"python": "PY", "families6": "F6", "inputs": "IN"}, F.FamiliesConfig(), "SEAM", "LBL", "OUT")
        norm = ["<v>" if m in ("PY", "F6", "IN", "SEAM", "LBL", "OUT") else m for m in mine]
        self.assertEqual(norm, theirs)
        self.assertEqual(mine[mine.index("--n-boot") + 1], "2000")
        self.assertEqual(F.N_BOOT_DEFAULT, 2000)

    def test_no_file_system_call_outside_the_io_functions(self):
        with open(SRC, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        bad = []
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            if fn.name.startswith("_io_"):
                continue
            for n in ast.walk(fn):
                if isinstance(n, ast.Call) and (_dotted(n.func) in FS_CALLS or _dotted(n.func).startswith("subprocess.")):
                    bad.append((fn.name, _dotted(n.func)))
        self.assertEqual(bad, [])

    def test_every_cli_flag_reaches_its_config_and_a_missing_ledger_exits_2(self):
        ap = F.build_parser()
        a = ap.parse_args(["run", "--out-dir", "O", "--look-ledger", "L", "--families6", "F", "--inputs", "I", "--python", "P", "--n-boot", "77", "--runner", "inprocess"])
        self.assertEqual({act.dest for act in ap._actions if act.dest not in ("help", "cmd")}, {"out_dir", "look_ledger", "families6", "inputs", "python", "n_boot", "runner"})
        seen = {}
        with mock.patch.object(F, "run_families", lambda o, c, *x, **k: seen.update(o=o, c=c)):
            self.assertEqual(F.main(["run", "--out-dir", "O", "--look-ledger", "L", "--families6", "F", "--inputs", "I", "--python", "P", "--n-boot", "77", "--runner", "inprocess"]), 0)
        self.assertEqual((seen["o"].out_dir, seen["o"].look_ledger), ("O", "L"))
        self.assertEqual((seen["c"].families6, seen["c"].inputs, seen["c"].python, seen["c"].n_boot, seen["c"].runner), ("F", "I", "P", 77, "inprocess"))
        sc = self.scene()
        os.remove(sc.ledger)
        self.assertEqual(F.main(["run", "--out-dir", sc.out, "--look-ledger", sc.ledger, "--inputs", sc.export, "--runner", "inprocess"]), 2)

    def test_the_tool_candidates_include_the_repo_relative_path(self):
        rel = F.FAMILIES6_CANDIDATES[1]
        self.assertTrue(rel.replace("\\", "/").endswith("FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/code/families6.py"))
        self.assertEqual(os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(rel)))), "2026-09-23-refcv6-standard-tests")
        self.assertEqual(F.FAMILIES6_CANDIDATES[0], os.environ.get("REFE_FAMILIES6", ""))


class ZZCoverage(unittest.TestCase):
    def test_every_marker_in_the_source_was_mutated_by_a_test_in_this_run(self):
        missing = sorted(set(list_markers()) - MUTATED)
        self.assertEqual(missing, [], f"markers with no mutation test: {missing}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
