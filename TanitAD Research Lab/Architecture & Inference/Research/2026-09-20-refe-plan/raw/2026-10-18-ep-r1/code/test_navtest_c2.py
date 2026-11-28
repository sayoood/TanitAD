"""Tests for navtest_c2.py (the navtest shuffled-label negative control) on SYNTHETIC worlds only (never the real S0 bank, never the real navtest outcomes).  stdlib unittest (also collected by pytest):

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I test_navtest_c2.py -v

The world is the landed confirm tests' `NavWorld` (same P2 head, same EP-R1 / SR-TTC run dirs, same informative-EP label) plus the C2 files the dev gates recorded: EP-R1's `epr1_c2_result.json` /
`ep_row_c2shuffle_seed*.npy` and SR-TTC's `r1_c2_result.json` / `ttc_row_c2shuffle_seed*.npy`, and the `c2_result_sha256` / `c2_rows_sha256` fields of both dev results.

(a) KNOWN VALUES: every C2 reading equals an independent float64 recomputation; the score equals a numpy value.
(b) IDENTITY ARM: C2 rows equal to P2's give P2's pick on every token and the reach control goes RED BEFORE the look.
(c) REFUSALS fire before any outcome is opened: the table of every `picks` test has NO pdms / sub member at all.  The `score` step refuses, without touching the outcomes, when the ledger is absent.
(d) EVERY CHECK HAS A DELIBERATE-REGRESSION ARM: navtest_c2.py is mutated at ONE marked line (`MUTATION-TARGET-NTC2-<X>`) and the check must go RED while the unmutated module is GREEN; a coverage test requires
    every marker in the source to be mutated.
(e) REACHABILITY: every CLI flag reaches ConfirmOptions; the CLI runs picks -> the landed read -> score; an AST scan proves no file-system call lives outside an `_io_*` function.
"""
from __future__ import annotations

import ast
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
import epr1 as R                 # noqa: E402
import epr1_dev as D             # noqa: E402
import refe_head_confirm as C    # noqa: E402
import navtest_c2 as N           # noqa: E402
import test_refe_head_confirm as TC   # noqa: E402
from test_epr1_dev import World, sha_b, sha_f, wj, FS_CALLS, _dotted   # noqa: E402

SRC = os.path.join(HERE, "navtest_c2.py")
SK = D.SEED_KEYS
NP_ = R.N_PROPOSALS
MUTATED = set()


def rj(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_mutant(*edits, name="navtest_c2_mutant"):
    """navtest_c2.py with ONE marked line edited per (marker, old, new)."""
    with open(SRC, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for marker, old, new in edits:
        pat = re.compile(rf"MUTATION-TARGET-NTC2-{re.escape(marker)}(?![A-Z0-9-])")
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


def near_tie_free(z, picks_mod, picks_ref):
    """The module applies in float32 and the oracle in float64: a pick may differ only where the two top scores are within 1e-5."""
    bad = []
    for i in np.nonzero(picks_mod != picks_ref)[0]:
        p = 1.0 / (1.0 + np.exp(-z[i]))
        s = p[:, 0] * p[:, 1] * (5 * p[:, 2] + 5 * p[:, 3] + 2 * p[:, 4]) / 12.0
        if abs(s[picks_mod[i]] - s[picks_ref[i]]) > 1e-5:
            bad.append(int(i))
    return bad


def list_markers():
    with open(SRC, encoding="utf-8") as f:
        return sorted(set(re.findall(r"MUTATION-TARGET-NTC2-([A-Z0-9-]+?)(?![A-Z0-9-])", f.read())))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the world: NavWorld + the C2 files and the dev-result fields that bind them
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class C2World(TC.NavWorld):
    def __init__(self, root, ep_c2=(-3.0, -3.0), ttc_c2=(-3.0, -3.0), dev_records_c2=True, **kw):
        self._ep_c2, self._ttc_c2, self._rec = ep_c2, ttc_c2, dev_records_c2
        self._ready = False
        self._v = {}
        super().__init__(root, **kw)

    # ---- C2 files
    def _row(self, col, sc, sd):
        w = self.w
        pert = (np.float32(1e-3 * sc * int(sd)) * np.cos(np.arange(R.FEAT_DIM))).astype(np.float32)
        return (w.W6[col] + np.float32(sc) * w.u + pert, w.b6[col:col + 1].copy())

    def set_ep_c2(self, scales=None, rows=None):
        w = self.w
        final = {}
        for sd, sc in zip(SK, scales or self._ep_c2):
            row = rows[sd] if rows else self._row(2, sc, sd)
            final[sd] = World._write_row(w.run_dir, f"ep_row_c2shuffle_seed{sd}.npy", row, w.W6[2], w.b6[2:3])
        w.c2["final_rows"] = final
        w.write_c2()

    def set_ttc_c2(self, scales=None, rows=None):
        w = self.w
        final = {}
        for sd, sc in zip(SK, scales or self._ttc_c2):
            row = rows[sd] if rows else self._row(3, sc, sd)
            final[sd] = World._write_row(w.ttc_dir, f"ttc_row_c2shuffle_seed{sd}.npy", row, w.W6[3], w.b6[3:4])
        doc = {"schema": "srttc-r1-v1", "stage": "C2", "c_star": w.ttc_r1["c_star"], "bank_manifest_sha256": w.ttc_r1["inputs"]["bank_manifest_sha256"], "final_rows": final}
        self.ttc_c2_doc = doc
        wj(os.path.join(w.ttc_dir, "r1_c2_result.json"), doc)

    def ensure(self):
        if not self._ready:
            self._ready = True
            self.set_ep_c2()
            self.set_ttc_c2()

    def c2_fields(self, arm):
        w = self.w
        if arm == "EP-R1":
            return {"c2_result_sha256": sha_f(os.path.join(w.run_dir, "epr1_c2_result.json")), "c2_rows_sha256": {sd: w.c2["final_rows"][sd]["sha256"] for sd in SK}}
        d = rj(os.path.join(w.ttc_dir, "r1_c2_result.json"))
        return {"c2_result_sha256": sha_f(os.path.join(w.ttc_dir, "r1_c2_result.json")), "c2_rows_sha256": {sd: d["final_rows"][sd]["sha256"] for sd in SK}}

    def ep_dev_doc(self, verdict, rows=None, result_sha=None):
        self.ensure()
        doc = super().ep_dev_doc(verdict, rows, result_sha)
        if self._rec:
            doc["inputs"].update(self.c2_fields("EP-R1"))
        self._v["ep"] = verdict
        return doc

    def ttc_dev_doc(self, verdict, rows=None, result_sha=None):
        self.ensure()
        doc = super().ttc_dev_doc(verdict, rows, result_sha)
        if self._rec:
            doc["inputs"].update(self.c2_fields("SR-TTC"))
        self._v["ttc"] = verdict
        return doc

    def refresh_dev(self):
        """re-record the dev results' C2 fields after the C2 files were edited (the dev gate 'saw' the edited files)"""
        if self._v.get("ep"):
            self.write_ep_dev(self._v["ep"])
        if self._v.get("ttc"):
            self.write_ttc_dev(self._v["ttc"])

    def edit_dev_c2(self, arm, **fields):
        p = self.ep_dev_path if arm == "EP-R1" else self.ttc_dev_path
        d = rj(p)
        d["inputs"].update(fields)
        wj(p, d)

    def ep_real(self):
        return {sd: (self.w.rows[sd][0], self.w.rows[sd][1]) for sd in SK}

    def ttc_real(self):
        return {sd: (self.w.ttc_rows[sd][0], self.w.ttc_rows[sd][1]) for sd in SK}

    def ep_c2_rows(self):
        return {sd: self._row(2, sc, sd) for sd, sc in zip(SK, self._ep_c2)}


class NBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ntc2_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self._k = 0

    def world(self, **kw):
        self._k += 1
        return C2World(os.path.join(self.tmp, f"nw{self._k}"), **kw)

    def read_look(self, nw, **kw):
        """spend the look with the LANDED read (this is what the wrapper does between picks and score)"""
        o = nw.opts(**kw)
        C.stage_read(o)
        return o

    def mut(self, marker, old, new, scenario):
        """scenario(mod) -> True iff the check FIRES.  GREEN on the unmutated module, and the one-line mutant must make it stop firing."""
        MUTATED.add(marker)
        self.assertTrue(scenario(N), f"{marker}: the scenario must fire on the UNMUTATED module")
        mod = load_mutant((marker, old, new))
        self.assertFalse(scenario(mod), f"{marker}: the one-line mutant must make the check stop firing")


def picks_refused(mod, nw, fragment=None, **kw) -> bool:
    try:
        mod.run_c2_picks(nw.opts(**kw))
    except mod.C2Refusal as ex:
        return fragment is None or fragment in str(ex)
    except Exception:                                                     # noqa: BLE001
        return False
    return False


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# step 1 on a table with NO outcome members: happy path, bindings, gating
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Picks(NBase):
    def test_happy_path_writes_picks_and_a_report_without_touching_the_outcomes_or_the_ledger(self):
        nw = self.world(outcomes=False)
        o = nw.opts()
        rep = N.run_c2_picks(o)
        out = os.path.join(o.out_dir, N.SUBDIR)
        self.assertTrue(os.path.exists(os.path.join(out, N.PICKS_NPZ)) and os.path.exists(os.path.join(out, N.PICKS_REPORT)))
        self.assertFalse(os.path.exists(nw.ledger), "picks never spends the look")
        self.assertFalse(rep["look"]["spent"])
        self.assertEqual(sorted(rep["readings"]), ["C2EPR1_s0", "C2EPR1_s1", "C2SRTTC_s0", "C2SRTTC_s1"])
        self.assertEqual(rep["arms"], ["EP-R1", "SR-TTC"])
        for k in ("integrity_featcheck", "C0_identity", "C1_reach", "C4_fan_frozen"):
            self.assertTrue(rep["controls"][k]["pass"], k)
        z = np.load(os.path.join(out, N.PICKS_NPZ))
        self.assertEqual(sorted(z.files), ["C2EPR1_s0", "C2EPR1_s1", "C2SRTTC_s0", "C2SRTTC_s1", "P2", "token"])
        self.assertEqual(rep["picks_file"]["sha256"], sha_f(os.path.join(out, N.PICKS_NPZ)))
        for nm, ent in rep["readings"].items():
            self.assertEqual(ent["picks_sha256"], sha_b(z[nm].tobytes()))
            self.assertGreater(ent["entries_changed_in_named_column"], 0)
            self.assertGreater(ent["changed_picks_vs_p2"], 0, f"{nm}: a shuffled-label row that moves no pick tests nothing")

    def test_the_binding_records_the_files_dev_applied_and_the_real_rows_the_look_will_read(self):
        nw = self.world(outcomes=False)
        rep = N.run_c2_picks(nw.opts())
        b = rep["binding"]
        self.assertEqual(b["EP-R1"]["dev_c2_rows_sha256"], nw.c2_fields("EP-R1")["c2_rows_sha256"])
        self.assertEqual(b["SR-TTC"]["dev_c2_result_sha256"], nw.c2_fields("SR-TTC")["c2_result_sha256"])
        self.assertEqual(b["EP-R1"]["c2_rows_sha256"], b["EP-R1"]["dev_c2_rows_sha256"])
        self.assertEqual(b["EP-R1"]["real_rows_sha256"], {sd: nw.w.r1["final_rows"][sd]["sha256"] for sd in SK})
        self.assertEqual(b["SR-TTC"]["real_rows_sha256"], {sd: nw.w.ttc_r1["final_rows"][sd]["sha256"] for sd in SK})
        self.assertNotEqual(b["EP-R1"]["c2_rows_sha256"], b["EP-R1"]["real_rows_sha256"])

    def test_every_c2_reading_equals_an_independent_float64_recomputation(self):
        nw = self.world(outcomes=False)
        o = nw.opts()
        N.run_c2_picks(o)
        z = np.load(os.path.join(o.out_dir, N.SUBDIR, N.PICKS_NPZ))
        tie = near_tie_free
        for sd in SK:
            for nm, named in ((f"C2EPR1_s{sd}", {2: nw.ep_c2_rows()[sd]}), (f"C2SRTTC_s{sd}", {3: nw._row(3, nw._ttc_c2[int(sd)], sd)})):
                ref, lg = nw.oracle_picks(named)
                self.assertEqual(tie(lg, z[nm].astype(np.int64), ref), [], nm)
        ref_p2, _ = nw.oracle_picks({})
        self.assertTrue(np.array_equal(ref_p2, z["P2"].astype(np.int64)))

    def test_the_picks_step_is_exclusive_and_the_look_gating_is_the_landed_one(self):
        nw = self.world(outcomes=False)
        o = nw.opts()
        N.run_c2_picks(o)
        with self.assertRaises(N.C2Refusal):
            N.run_c2_picks(nw.opts(out_dir=o.out_dir))
        nw2 = self.world(outcomes=False, ep_dev=None, ttc_dev=None)
        with self.assertRaises(C.LookNotOpen):
            N.run_c2_picks(nw2.opts())

    def test_only_the_arms_whose_primary_reading_is_open_get_a_c2_reading(self):
        nw = self.world(outcomes=False, ep_dev="FAIL")
        rep = N.run_c2_picks(nw.opts())
        self.assertEqual(rep["arms"], ["SR-TTC"])
        self.assertEqual(sorted(rep["readings"]), ["C2SRTTC_s0", "C2SRTTC_s1"])
        nw = self.world(outcomes=False, ttc_dev=None)
        ab = nw.abandon("SR-TTC")
        rep = N.run_c2_picks(nw.opts(ttc_abandon=ab, ttc_dev_result=None))
        self.assertEqual(rep["arms"], ["EP-R1"])
        self.assertEqual(sorted(rep["readings"]), ["C2EPR1_s0", "C2EPR1_s1"])

    # ---- the registered chain, each link with a deliberate-regression arm
    def test_a_dev_result_that_does_not_record_the_c2_files_cannot_bind_navtest_c2(self):
        nw = self.world(outcomes=False, dev_records_c2=False)
        with self.assertRaises(N.C2Refusal) as cm:
            N.run_c2_picks(nw.opts())
        self.assertIn("does not record", str(cm.exception))
        nw = self.world(outcomes=False)
        d = rj(nw.ep_dev_path)
        del d["inputs"]["c2_rows_sha256"]["1"]
        wj(nw.ep_dev_path, d)
        self.assertTrue(picks_refused(N, nw, "does not record"))

    def test_the_c2_result_must_be_the_file_dev_read(self):
        nw = self.world(outcomes=False)
        nw.edit_dev_c2("EP-R1", c2_result_sha256="0" * 64)
        self.mut("BIND-RESULT", "if sha != binding[\"c2_result_sha256\"]:", "if False:", lambda m: picks_refused(m, nw, "these are not the C2 rows dev tested"))

    def test_each_row_must_be_the_row_dev_applied(self):
        nw = self.world(outcomes=False)
        rows = nw.c2_fields("SR-TTC")["c2_rows_sha256"]
        nw.edit_dev_c2("SR-TTC", c2_rows_sha256={"0": rows["0"], "1": "1" * 64})
        self.mut("BIND-ROWS", "if rsha != binding[\"c2_rows_sha256\"][sd]:", "if False:", lambda m: picks_refused(m, nw, "the dev gate applied"))

    def test_the_c2_fit_must_belong_to_this_cstar(self):
        nw = self.world(outcomes=False)
        d = rj(os.path.join(nw.w.run_dir, "epr1_c2_result.json"))
        d["c_star"] = dict(d["c_star"], config_sha256="0" * 64)
        wj(os.path.join(nw.w.run_dir, "epr1_c2_result.json"), d)
        nw.edit_dev_c2("EP-R1", c2_result_sha256=sha_f(os.path.join(nw.w.run_dir, "epr1_c2_result.json")))
        self.mut("BIND-CSTAR", "if (doc.get(\"c_star\") or {}).get(\"config_sha256\") != arm_doc[\"c_star\"][\"config_sha256\"]:", "if False:", lambda m: picks_refused(m, nw, "different c*"))

    def test_the_c2_fit_must_belong_to_this_bank(self):
        nw = self.world(outcomes=False)
        d = rj(os.path.join(nw.w.ttc_dir, "r1_c2_result.json"))
        d["bank_manifest_sha256"] = "0" * 64
        wj(os.path.join(nw.w.ttc_dir, "r1_c2_result.json"), d)
        nw.edit_dev_c2("SR-TTC", c2_result_sha256=sha_f(os.path.join(nw.w.ttc_dir, "r1_c2_result.json")))
        self.mut("BIND-BANK", "if doc.get(\"bank_manifest_sha256\") != (arm_doc.get(\"inputs\") or {}).get(\"bank_manifest_sha256\"):", "if False:", lambda m: picks_refused(m, nw, "different S0 bank"))

    def test_a_c2_row_that_is_the_real_row_tests_nothing_and_is_refused(self):
        nw = self.world(outcomes=False)
        nw.set_ep_c2(rows=nw.ep_real())                                                    # the "shuffled" rows ARE the real fit
        nw.refresh_dev()
        self.mut("DIFFERS", "if rsha == real_rows_sha[sd]:", "if False:", lambda m: picks_refused(m, nw, "IS the real row"))

    def test_a_c2_result_missing_or_not_a_c2_stage_is_refused(self):
        nw = self.world(outcomes=False)
        os.remove(os.path.join(nw.w.ttc_dir, "r1_c2_result.json"))
        self.assertTrue(picks_refused(N, nw, "does not exist"))
        nw = self.world(outcomes=False)
        d = rj(os.path.join(nw.w.ttc_dir, "r1_c2_result.json"))
        d["stage"] = "R1"
        wj(os.path.join(nw.w.ttc_dir, "r1_c2_result.json"), d)
        nw.refresh_dev()
        self.assertTrue(picks_refused(N, nw, "not a C2 stage"))

    # ---- (b) identity: C2 rows equal to P2's row test nothing -> RED before the look
    def test_identity_arm_c2_rows_equal_to_p2s_give_p2s_pick_everywhere_and_the_reach_control_refuses_before_the_look(self):
        nw = self.world(outcomes=False, ep_c2=(0.0, 0.0), ttc_c2=(0.0, 0.0))
        for sd in SK:
            self.assertTrue(np.array_equal(R.row_bytes(*nw.ep_c2_rows()[sd]), R.row_bytes(nw.w.W6[2], nw.w.b6[2:3])), "scale 0 is exactly P2's row")
        with self.assertRaises(N.C2Refusal) as cm:
            N.run_c2_picks(nw.opts())
        self.assertIn("control(s) RED", str(cm.exception))
        self.assertIn("UNSPENT", str(cm.exception))
        self.assertFalse(os.path.exists(nw.ledger))
        # the mutant (control gate removed) completes, and its picks ARE P2's on every token: the identity the gate exists to refuse
        MUTATED.add("CTLGATE")
        m = load_mutant(("CTLGATE", "if not C.controls_pass(ctl):", "if False:"))
        o = nw.opts()
        rep = m.run_c2_picks(o)
        z = np.load(os.path.join(o.out_dir, N.SUBDIR, N.PICKS_NPZ))
        for nm in rep["readings"]:
            self.assertTrue(np.array_equal(z[nm], z["P2"]), nm)
        self.assertFalse(rep["controls"]["C1_reach"]["pass"])

    def test_a_red_integrity_control_refuses_the_picks_step_too(self):
        nw = self.world(outcomes=False)
        pr = nw.w.props.copy()
        pr[0, 0, 0, 0] += 1e-3
        nw.write_bank(props=pr)                                                            # C4: the dump differs from the table
        with self.assertRaises(N.C2Refusal) as cm:
            N.run_c2_picks(nw.opts())
        self.assertIn("C4_fan_frozen", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(nw.opts().out_dir, N.SUBDIR, N.PICKS_NPZ)), "a refused pass leaves nothing behind")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# step 2: after the ledger
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Score(NBase):
    def spent(self, nw, **kw):
        """picks -> the landed read (spends the look) -> opts for score"""
        o0 = nw.opts(**kw)
        N.run_c2_picks(o0)
        C.stage_read(nw.opts(out_dir=os.path.join(nw.root, "read_out"), **kw))
        return o0

    def test_score_equals_a_numpy_value_and_passes_on_a_shuffled_world(self):
        nw = self.world()
        o = self.spent(nw)
        res = N.run_c2_score(o)
        self.assertTrue(res["c2_pass_all_arms"])
        self.assertEqual(res["red_arms"], [])
        w = nw.w
        ar = np.arange(nw.N)
        base = nw.oracle_picks({})[0]
        for arm, col in (("EP-R1", 2), ("SR-TTC", 3)):
            for sd in SK:
                row = nw.ep_c2_rows()[sd] if arm == "EP-R1" else nw._row(3, nw._ttc_c2[int(sd)], sd)
                new = nw.oracle_picks({col: row})[0]
                want = 100.0 * float(w.pdms[ar, new].mean() - w.pdms[ar, base].mean())
                got = res["arms"][arm]["per_seed"][sd]["gain_x100"]["point"]
                self.assertAlmostEqual(got, want, places=8, msg=f"{arm} s{sd}")
                self.assertLessEqual(got, 0.0)
                self.assertEqual(res["arms"][arm]["per_seed"][sd]["changed_picks"], int((new != base).sum()))
        self.assertEqual(res["look"]["path"], nw.ledger)
        self.assertFalse(res["consequence"]["registered"])
        self.assertEqual(set(res["gaps"]), {"GN1", "GN2"})
        self.assertEqual(res["estimator"]["B"], 10000)

    def test_the_c2_intervals_use_the_same_draws_as_the_real_readings(self):
        nw = self.world()
        o = self.spent(nw)
        res = N.run_c2_score(o)
        real = rj(os.path.join(nw.root, "read_out", "confirm_read_result.json"))
        self.assertEqual(real["estimator"]["seed"], res["estimator"]["seed"])
        self.assertEqual(real["estimator"]["n_logs"], res["estimator"]["n_logs"])
        # the P2 pick the C2 readings are paired against is the one the look was spent on
        self.assertEqual(rj(o.look_ledger)["picks_sha256"]["P2"], rj(os.path.join(o.out_dir, N.SUBDIR, N.PICKS_REPORT))["p2_picks_sha256"])

    def test_a_c2_row_that_gains_makes_that_arm_red_and_only_that_arm(self):
        nw = self.world(ep_c2=(2.5, 2.5))                                                  # the EP "shuffled" rows move along +u: the instrument reads a gain from a null
        o = self.spent(nw)
        res = N.run_c2_score(o)
        self.assertFalse(res["c2_pass_all_arms"])
        self.assertEqual(res["red_arms"], ["EP-R1"])
        self.assertTrue(all(res["arms"]["EP-R1"]["per_seed"][sd]["gain_x100"]["point"] > 0 for sd in SK))
        self.assertTrue(res["arms"]["SR-TTC"]["pass"])
        self.assertIn("Master Mind rules", res["consequence"]["note"])
        MUTATED.add("RULE")
        m = load_mutant(("RULE", "bool(pt <= 0.0)", "bool(pt <= 1e9)"))
        self.assertTrue(m.run_c2_score(o)["c2_pass_all_arms"], "the mutant reads the RED arm as green")

    def test_both_seeds_rule_one_red_seed_makes_the_arm_red(self):
        nw = self.world(ttc_c2=(-3.0, 3.0))
        o = self.spent(nw)
        res = N.run_c2_score(o)
        self.assertEqual(res["red_arms"], ["SR-TTC"])
        self.assertEqual({sd: v["meets_rule"] for sd, v in res["arms"]["SR-TTC"]["per_seed"].items()}, {"0": True, "1": False})
        MUTATED.add("BOTHSEEDS")
        m = load_mutant(("BOTHSEEDS", "all(v[\"meets_rule\"] for v in seeds.values())", "any(v[\"meets_rule\"] for v in seeds.values())"))
        self.assertTrue(m.run_c2_score(o)["arms"]["SR-TTC"]["pass"], "any-seed would pass a one-seed-red arm")

    def test_the_rule_boundary_is_exact_zero_with_no_tolerance(self):
        mk = lambda pt: {"gain_x100": {"point": pt, "lo": pt - 1, "hi": pt + 1}, "changed_picks": 5}
        self.assertTrue(N.c2_verdict({"0": mk(0.0), "1": mk(-1e-12)})["pass"])
        self.assertFalse(N.c2_verdict({"0": mk(0.0), "1": mk(1e-12)})["pass"], "no tolerance (MM D7)")

    def test_a_c2_that_moved_no_pick_reads_exactly_zero_and_is_flagged_vacuous(self):
        nw = self.world(ep_c2=(1e-6, 1e-6))
        o = self.spent(nw)
        res = N.run_c2_score(o)
        v = res["arms"]["EP-R1"]
        self.assertEqual(v["vacuous_seeds"], ["0", "1"])
        self.assertTrue(all(s["gain_x100"]["point"] == 0.0 and s["changed_picks"] == 0 for s in v["per_seed"].values()))
        self.assertTrue(v["pass"], "<= 0 holds")
        self.assertTrue(all(s["vacuous"] for s in v["per_seed"].values()), "but the pass is reported as vacuous, never as a pass of substance")

    # ---- (c) the ledger gates the outcomes
    def test_score_before_the_look_is_spent_refuses_without_opening_the_outcomes(self):
        nw = self.world()
        o = nw.opts()
        N.run_c2_picks(o)
        self.assertFalse(os.path.exists(nw.ledger))

        def scen(mod):
            calls = []
            with mock.patch.object(D, "read_outcomes", side_effect=lambda *a, **k: calls.append(1)):
                try:
                    mod.run_c2_score(o)
                except mod.C2Refusal as ex:
                    return "not spent" in str(ex) and not calls
                except Exception:                                                          # noqa: BLE001
                    return False
            return False
        self.mut("LEDGER", "if not C._io_exists(guard, opts.look_ledger):", "if False:", scen)

    def test_a_ledger_of_another_look_is_refused(self):
        nw = self.world()
        o = self.spent(nw)
        led = rj(o.look_ledger)
        led["table_sha256"] = "0" * 64
        wj(o.look_ledger, led)
        self.mut("LEDGER-BIND", "if led.get(\"table_sha256\") != rep[\"inputs\"][\"navtest_table_sha256\"] or", "if False and", lambda m: _refuses(m, o, "does not describe the look"))
        led = rj(o.look_ledger)
        led["table_sha256"] = rj(os.path.join(o.out_dir, N.SUBDIR, N.PICKS_REPORT))["inputs"]["navtest_table_sha256"]
        led["picks_sha256"] = dict(led["picks_sha256"], P2="0" * 64)
        wj(o.look_ledger, led)
        self.assertTrue(_refuses(N, o, "P2's pick"))
        led["picks_sha256"] = dict(led["picks_sha256"], P2=rj(os.path.join(o.out_dir, N.SUBDIR, N.PICKS_REPORT))["p2_picks_sha256"])
        led["rows_sha256"] = {"EP-R1": {"0": "0" * 64, "1": "1" * 64}, "SR-TTC": led["rows_sha256"]["SR-TTC"]}
        wj(o.look_ledger, led)
        self.assertTrue(_refuses(N, o, "real rows"))

    def test_picks_edited_after_the_pass_are_refused(self):
        nw = self.world()
        o = self.spent(nw)
        pk = os.path.join(o.out_dir, N.SUBDIR, N.PICKS_NPZ)
        z = np.load(pk)
        arr = {k: z[k] for k in z.files}
        arr["C2EPR1_s0"] = arr["C2EPR1_s0"].copy()
        arr["C2EPR1_s0"][0] = (arr["C2EPR1_s0"][0] + 1) % NP_
        np.savez(pk, **arr)
        self.mut("PICKSHA", "if C._io_sha256(guard, pk_path) != rep[\"picks_file\"][\"sha256\"]:", "if False:", lambda m: _refuses(m, o, "edited after the pass"))

    def test_score_needs_the_picks_report_of_this_look_and_a_ledger_path(self):
        nw = self.world()
        with self.assertRaises(N.C2Refusal):
            N.run_c2_score(nw.opts(look_ledger=None))
        o = self.spent(nw)
        os.remove(os.path.join(o.out_dir, N.SUBDIR, N.PICKS_REPORT))
        with self.assertRaises(Exception):
            N.run_c2_score(o)

    def test_score_is_deterministic_and_rerunnable(self):
        nw = self.world()
        o = self.spent(nw)
        a = N.run_c2_score(o)
        b = N.run_c2_score(o)
        for k in ("arms", "c2_pass_all_arms", "estimator"):
            self.assertEqual(json.dumps(D._jsonable(a[k]), sort_keys=True), json.dumps(D._jsonable(b[k]), sort_keys=True), k)


def _refuses(mod, o, fragment) -> bool:
    try:
        mod.run_c2_score(o)
    except mod.C2Refusal as ex:
        return fragment in str(ex)
    except Exception:                                                                      # noqa: BLE001
        return False
    return False


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (e) reachability
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Reach(NBase):
    def test_no_file_system_call_outside_the_io_functions(self):
        with open(SRC, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        bad = []
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            if fn.name.startswith("_io_"):
                continue
            for n in ast.walk(fn):
                if isinstance(n, ast.Call) and _dotted(n.func) in FS_CALLS:
                    bad.append((fn.name, _dotted(n.func)))
        self.assertEqual(bad, [])

    def test_every_cli_flag_reaches_confirm_options_and_the_cmd_choices_are_picks_and_score(self):
        ap = N.build_parser()
        flags = ["--bank-dir", "B", "--out-dir", "O", "--ep-run-dir", "E", "--ep-registered-result-sha256", "e", "--ep-dev-result", "ed", "--ep-abandon", "ea", "--ttc-run-dir", "T",
                 "--ttc-registered-result-sha256", "t", "--ttc-dev-result", "td", "--ttc-abandon", "ta", "--table", "TB", "--p2-ckpt", "CK", "--banked-p2-seam", "S", "--look-ledger", "L", "--c2-out", "X"]
        a = ap.parse_args(["picks"] + flags)
        dests = {act.dest for act in ap._actions if act.dest not in ("help", "cmd")}
        self.assertEqual(dests, {"bank_dir", "out_dir", "ep_run_dir", "ep_registered_result_sha256", "ep_dev_result", "ep_abandon", "ttc_run_dir", "ttc_registered_result_sha256", "ttc_dev_result",
                                 "ttc_abandon", "table", "p2_ckpt", "banked_p2_seam", "look_ledger", "c2_out"})
        o = N.options_from_args(a)
        self.assertEqual((o.bank_dir, o.out_dir, o.ep_run_dir, o.ep_registered_result_sha256, o.ep_dev_result, o.ep_abandon, o.ttc_run_dir, o.ttc_registered_result_sha256, o.ttc_dev_result,
                          o.ttc_abandon, o.table, o.p2_ckpt, o.banked_p2_seam, o.look_ledger), ("B", "O", "E", "e", "ed", "ea", "T", "t", "td", "ta", "TB", "CK", "S", "L"))
        self.assertEqual(a.c2_out, "X")
        self.assertEqual(o.pins_overridden(), [], "the CLI never changes a pin")
        with self.assertRaises(SystemExit):
            ap.parse_args(["read"] + flags)

    def test_the_cli_runs_picks_then_the_landed_read_then_score(self):
        nw = self.world()
        o = nw.opts()
        base = ["--bank-dir", nw.bank_dir, "--out-dir", o.out_dir, "--ep-run-dir", nw.w.run_dir, "--ep-registered-result-sha256", nw.w.r1_sha(), "--ep-dev-result", nw.ep_dev_path,
                "--ttc-run-dir", nw.w.ttc_dir, "--ttc-registered-result-sha256", nw.ttc_sha, "--ttc-dev-result", nw.ttc_dev_path, "--table", nw.table, "--p2-ckpt", nw.w.ckpt,
                "--banked-p2-seam", nw.seam, "--look-ledger", nw.ledger]
        # the CLI's ConfirmOptions carry the real pins, so the synthetic world is told its own (the same override the landed CLI tests use)
        real = C.ConfirmOptions

        def patched(**kw):
            kw.update(table_sha256=sha_f(nw.table), expect_tokens=nw.N, expect_logs=nw.n_logs)
            return real(**kw)
        with mock.patch.object(N.C, "ConfirmOptions", patched):
            self.assertEqual(N.main(["picks"] + base), 0)
            self.assertEqual(N.main(["score"] + base), 2, "the look is not spent: score refuses (exit 2), nothing was opened")
            self.assertFalse(os.path.exists(os.path.join(o.out_dir, N.SUBDIR, N.SCORE_RESULT)))
            C.stage_read(nw.opts(out_dir=os.path.join(nw.root, "read_out")))
            self.assertEqual(N.main(["score"] + base), 0)
        self.assertTrue(os.path.exists(os.path.join(o.out_dir, N.SUBDIR, N.SCORE_RESULT)))
        self.assertEqual(N.main(["picks"] + base), 2, "a second picks into the same C2 directory is refused")


class ZZCoverage(unittest.TestCase):
    def test_every_marker_in_the_source_was_mutated_by_a_test_in_this_run(self):
        missing = sorted(set(list_markers()) - MUTATED)
        self.assertEqual(missing, [], f"markers with no mutation test: {missing}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
