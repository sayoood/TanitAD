"""Tests for refe_head_confirm.py on SYNTHETIC worlds only (never the real S0 bank, never the real navtest outcomes).  stdlib unittest (also collected by pytest):

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I test_refe_head_confirm.py -v

(a) IDENTITY ARM: rows equal to P2's give P2's pick on every token for every reading (bit for bit), the P2 seam equals the banked P2 seam, and the reach control goes RED for such a no-op row.
(b) KNOWN VALUES: the new picks equal an independent float64 / pure-python recomputation (one column replaced; two for the JOINT); the gain equals a numpy value; the bar clauses and the PREREG s8
    triggers on hand-built records.
(c) REFUSALS fire BEFORE any outcome is opened and before the table / bank are opened: the synthetic table of every refusal test has NO pdms / sub / valid / sub_names member, and the no-dev-result
    refusals run with a table and a bank that do not exist at all.
(d) THE LOOK IS SPENT ONCE: the ledger is created exclusively BEFORE the outcome arrays are opened (a crashed first read still consumed it); RED controls leave it unspent.
(e) EVERY CHECK HAS A DELIBERATE-REGRESSION ARM: refe_head_confirm.py is mutated at ONE marked line (`MUTATION-TARGET-CONFIRM-<X>`) and the check must go RED while the unmutated module is GREEN;
    a coverage test requires every marker in the source to be mutated at least once here.
(f) REACHABILITY: every CLI flag is traced into ConfirmOptions; the real navtest path names (`refe_navtest`, `table_p2_full`) pass the guard; an AST scan proves no file-system call lives
    outside an `_io_*` function.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
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
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epr1 as R              # noqa: E402
import epr1_dev as D          # noqa: E402
import refe_head_confirm as C  # noqa: E402
import zero_tail as ZT        # noqa: E402
from test_epr1_dev import World, ref_pick, sha_b, sha_f, wj, _rt, _rb, FS_CALLS, _dotted   # noqa: E402

SRC = os.path.join(HERE, "refe_head_confirm.py")
NP_ = R.N_PROPOSALS
SK = D.SEED_KEYS


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# mutation harness
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_cmutant(*edits):
    """refe_head_confirm.py with ONE marked line edited per (marker, old, new): `MUTATION-TARGET-CONFIRM-<marker>` must sit on exactly one line and `old` must occur exactly once ON THAT LINE."""
    with open(SRC, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for marker, old, new in edits:
        pat = re.compile(rf"MUTATION-TARGET-CONFIRM-{re.escape(marker)}(?![A-Z0-9-])")
        idx = [i for i, ln in enumerate(lines) if pat.search(ln)]
        assert len(idx) == 1, f"marker {marker} sits on {len(idx)} lines"
        assert lines[idx[0]].count(old) == 1, f"{old!r} occurs {lines[idx[0]].count(old)} times on the {marker} line"
        lines[idx[0]] = lines[idx[0]].replace(old, new)
    return _exec_module("\n".join(lines))


def _exec_module(src):
    name = "refe_head_confirm_mutant"
    mod = types.ModuleType(name)
    mod.__file__ = SRC
    sys.modules[name] = mod
    try:
        exec(compile(src, SRC, "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    return mod


def mopts(mod, opts):
    return mod.ConfirmOptions(**{**opts.__dict__})


def list_markers(src):
    return sorted(set(re.findall(r"MUTATION-TARGET-CONFIRM-([A-Z0-9-]+?)(?![A-Z0-9-])", src)))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# synthetic navtest world (built on the dev tests' World: same P2 head, same EP-R1 / SR-TTC run dirs, same informative-EP label)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class NavWorld:
    def __init__(self, root, outcomes=True, ep_scales=(3.0, 3.0), ttc_scales=(2.0, 2.0), ep_dev="PASS", ttc_dev="PASS", n_chunks=3, seed=0, clean_subs=False):
        self.root = root
        self.w = w = World(os.path.join(root, "w"), outcomes=outcomes, scales=ep_scales, ttc=True, ttc_scales=ttc_scales, seed=seed)
        if clean_subs:                       # TTC / C / DDC always satisfied: no sub-score can be adverse, so a seed that gains on EP clears every clause (the formula still holds exactly)
            w.sub = w.sub.copy()
            w.sub[..., 3:6] = 1.0
            w.pdms = w.sub[..., 0] * w.sub[..., 1] * (5.0 * w.sub[..., 2] + 5.0 * w.sub[..., 3] + 2.0 * w.sub[..., 4]) / 12.0
        self.N, self.n_logs, self.outcomes = w.N, w.n_logs, outcomes
        self.nav_dir = os.path.join(root, "refe_navtest")                       # the REAL names: the dev guard would refuse them, this module's guard must not
        self.bank_dir = os.path.join(self.nav_dir, "bank")
        self.table = os.path.join(self.nav_dir, "table_p2_full.npz")
        self.seam = os.path.join(self.nav_dir, "seams", "stage2_p2_full.npz")
        self.ledger = os.path.join(root, "look", C.LOOK_FILE)
        self.dev_dir = os.path.join(root, "dev_results")
        for d in (self.bank_dir, os.path.dirname(self.seam), self.dev_dir):
            os.makedirs(d, exist_ok=True)
        self.fp = np.array([f"fp-{t}" for t in w.token])
        self.write_table()
        ar = np.arange(self.N)
        np.savez(self.seam, token=w.token, fingerprint=self.fp, poses=w.props[ar, w.pick], sampling=np.array([8, 0.5]), arm=np.array("p2"))
        rng = np.random.default_rng(7)
        self.chunks = [np.sort(c) for c in np.array_split(rng.permutation(self.N), n_chunks)]          # bank order != table order
        self.write_bank()
        self.ttc_sha = sha_f(os.path.join(w.ttc_dir, "r1_result.json"))
        self.ep_dev_path = os.path.join(self.dev_dir, "epr1_dev_result.json")
        self.ttc_dev_path = os.path.join(self.dev_dir, "srttc_dev_result.json")
        if ep_dev:
            self.write_ep_dev(ep_dev)
        if ttc_dev:
            self.write_ttc_dev(ttc_dev)
        self._n = 0

    # ---- files
    def write_table(self, pick=None, outcomes=None):
        w = self.w
        T = {"token": w.token, "log": w.log, "fingerprint": self.fp, "logits": w.logits, "proposals": w.props, "pick": w.pick if pick is None else pick}
        if self.outcomes if outcomes is None else outcomes:
            T.update(pdms=w.pdms, sub=w.sub, valid=np.ones((self.N, NP_), bool), sub_names=np.array(D.SUB_NAMES))
        np.savez(self.table, **T)

    def write_bank(self, feat=None, props=None, pick=None, logits=None, skip=(), extra_dup=None):
        w = self.w
        feat = w.feat if feat is None else feat
        props = w.props if props is None else props
        pick = w.pick if pick is None else pick
        logits = w.logits if logits is None else logits
        for f in os.listdir(self.bank_dir):
            os.remove(os.path.join(self.bank_dir, f))
        chunks = [c for i, c in enumerate(self.chunks) if i not in skip]
        if extra_dup is not None:
            chunks = chunks + [self.chunks[extra_dup]]
        for i, idx in enumerate(chunks):
            np.savez(os.path.join(self.bank_dir, f"c{i:02d}_feat.npz"), token=w.token[idx], log=w.log[idx], prehead_features=feat[idx])
            np.savez(os.path.join(self.bank_dir, f"c{i:02d}_dump.npz"), token=w.token[idx], fingerprint=self.fp[idx], proposals=props[idx], logits=logits[idx], pick=pick[idx], log=w.log[idx])
        self.write_manifest()

    def write_manifest(self):
        g = C.ConfirmGuard(read_dirs=[self.bank_dir], write_dirs=[self.bank_dir])
        man = C.make_bank_manifest(g, self.bank_dir)
        mp = os.path.join(self.bank_dir, C.BANK_MANIFEST)
        if os.path.exists(mp):
            os.remove(mp)
        C._io_write_bytes(g, mp, C.canon(man), exclusive=True)

    def ep_dev_doc(self, verdict, rows=None, result_sha=None):
        w = self.w
        return {"schema": D.SCHEMA, "stage": D.STAGE, "dev_gate": {"verdict": verdict},
                "inputs": {"epr1_result_sha256": result_sha or w.r1_sha(), "rows_sha256": rows or {sd: w.r1["final_rows"][sd]["sha256"] for sd in SK}}}

    def ttc_dev_doc(self, verdict, rows=None, result_sha=None):
        w = self.w
        return {"schema": "srttc-dev-v1", "dev_gate": {"verdict": verdict},
                "inputs": {"r1_result_sha256": result_sha or self.ttc_sha, "rows_sha256": rows or {sd: w.ttc_r1["final_rows"][sd]["sha256"] for sd in SK}}}

    def write_ep_dev(self, verdict, **kw):
        wj(self.ep_dev_path, self.ep_dev_doc(verdict, **kw))

    def write_ttc_dev(self, verdict, **kw):
        wj(self.ttc_dev_path, self.ttc_dev_doc(verdict, **kw))

    def abandon(self, which, **over):
        doc = {"arm": which, "abandoned": True, "register_row": "R-TEST-1: formally abandoned for the test", "reason": "synthetic", "decided_by": "Master Mind (test)"}
        doc.update(over)
        p = os.path.join(self.dev_dir, f"{which}_abandon.json")
        wj(p, doc)
        return p

    # ---- options
    def opts(self, **kw):
        self._n += 1
        w = self.w
        base = dict(out_dir=os.path.join(self.root, f"out{self._n}"), bank_dir=self.bank_dir, ep_run_dir=w.run_dir, ep_registered_result_sha256=w.r1_sha(), ep_dev_result=self.ep_dev_path,
                    ttc_run_dir=w.ttc_dir, ttc_registered_result_sha256=self.ttc_sha, ttc_dev_result=self.ttc_dev_path, table=self.table, table_sha256=sha_f(self.table), p2_ckpt=w.ckpt,
                    banked_p2_seam=self.seam, look_ledger=self.ledger, expect_tokens=self.N, expect_logs=self.n_logs)
        base.update(kw)
        return C.ConfirmOptions(**base)

    def p2_logits(self):
        return self.w.logits

    # ---- an independent float64 recomputation of the planner on logits with rows replaced
    def oracle_logits(self, named):
        z = self.w.logits.astype(np.float64).copy()
        f = self.w.feat.astype(np.float64)
        for col, (wt, b) in named.items():
            z[..., col] = z[..., col] + f @ (wt.astype(np.float64) - self.w.W6[col].astype(np.float64)) + (float(b[0]) - float(self.w.b6[col]))
        return z

    def oracle_picks(self, named):
        z = self.oracle_logits(named)
        return np.array([ref_pick(z[i]) for i in range(self.N)]), z


class CBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="refeconf_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self._k = 0

    def nav(self, **kw):
        self._k += 1
        return NavWorld(os.path.join(self.tmp, f"nw{self._k}"), **kw)

    def picks(self, nw, **kw):
        o = nw.opts(**kw)
        rep = C.stage_picks(o)
        z = np.load(os.path.join(o.out_dir, "picks.npz"))
        return rep, {k: z[k] for k in z.files}, o


def seeds_rows(nw, arm):
    w = nw.w
    if arm == "EP-R1":
        return {sd: (w.rows[sd][0], w.rows[sd][1]) for sd in SK}
    return {sd: (w.ttc_rows[sd][0], w.ttc_rows[sd][1]) for sd in SK}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (a) identity
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Identity(CBase):
    def test_p2_reading_is_the_table_pick_bit_for_bit_and_its_seam_is_the_banked_seam(self):
        nw = self.nav()
        rep, pk, o = self.picks(nw)
        self.assertTrue(np.array_equal(pk["P2"].astype(np.int64), nw.w.pick))
        self.assertEqual(rep["controls"]["C0_identity"]["pass"], True)
        self.assertEqual(rep["controls"]["banked_p2_seam"]["max_abs_pose_m"], 0.0)
        self.assertTrue(rep["controls"]["banked_p2_seam"]["fingerprints_equal"])
        z = np.load(rep["seams"]["P2"]["path"])
        self.assertTrue(np.array_equal(z["poses"], np.load(nw.seam)["poses"]))
        self.assertEqual(str(z["arm"]), "REFE_HEAD_P2")
        self.assertEqual(z["poses"].dtype, np.float32)
        self.assertEqual(z["poses"].shape, (nw.N, 8, 3))

    def test_rows_equal_to_p2s_give_p2s_pick_on_every_token_for_every_reading_and_c1_goes_red(self):
        nw = self.nav(ep_scales=(0.0, 0.0), ttc_scales=(0.0, 0.0))
        w = nw.w
        for sd in SK:                                                                  # the rows ARE P2's
            self.assertTrue(np.array_equal(w.rows[sd][0], w.W6[2]))
            self.assertTrue(np.array_equal(w.ttc_rows[sd][0], w.W6[3]))
        o = nw.opts()
        prep = C.prepare(o)
        self.assertEqual(prep["families"], ["P2", "EPR1", "SRTTC", "JOINT"])
        core = C.run_picks_core(o, prep)
        P = core["stats"]["picks"]
        self.assertEqual(sorted(P), sorted(prep["plan"]))
        for nm in P:
            self.assertTrue(np.array_equal(P[nm].astype(np.int64), w.pick), nm)
        self.assertTrue(core["ctl"]["C0_identity"]["pass"])
        self.assertFalse(core["ctl"]["C1_reach"]["pass"], "a row that equals P2's changes nothing: reach must read RED, not green")
        for nm, v in core["ctl"]["C1_reach"]["per_reading"].items():
            self.assertEqual(v["entries_changed_in_named_columns"], 0, nm)
        with self.assertRaises(C.ConfirmRefusal) as cm:                                 # ...and the stage therefore refuses and writes no seam
            C.stage_picks(nw.opts())
        self.assertIn("C1_reach", str(cm.exception))

    def test_a_real_row_changes_picks_and_only_the_named_columns(self):
        nw = self.nav()
        o = nw.opts()
        core = C.run_picks_core(o, C.prepare(o))
        c1 = core["ctl"]["C1_reach"]
        self.assertTrue(c1["pass"])
        for nm, v in c1["per_reading"].items():
            self.assertGreater(v["entries_changed_in_named_columns"], 0, nm)
            self.assertTrue(v["other_columns_bit_identical"], nm)
        self.assertGreater(c1["positive_control_negated_p2_ep_row_changed_picks"], 0)
        P = core["stats"]["picks"]
        self.assertGreater(int((P["EPR1_s0"] != P["P2"]).sum()), 0)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (b) known values
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class KnownValues(CBase):
    def _near_tie_free(self, z, picks_mod, picks_ref):
        """The module applies in float32 and the oracle in float64: a pick may differ only where the two top scores are within 1e-5."""
        bad = []
        for i in np.nonzero(picks_mod != picks_ref)[0]:
            p = 1.0 / (1.0 + np.exp(-z[i]))
            s = p[:, 0] * p[:, 1] * (5 * p[:, 2] + 5 * p[:, 3] + 2 * p[:, 4]) / 12.0
            if abs(s[picks_mod[i]] - s[picks_ref[i]]) > 1e-5:
                bad.append(int(i))
        return bad

    def test_every_reading_equals_an_independent_recomputation(self):
        nw = self.nav()
        rep, pk, o = self.picks(nw)
        ep, ttc = seeds_rows(nw, "EP-R1"), seeds_rows(nw, "SR-TTC")
        for sd in SK:
            for nm, named in ((f"EPR1_s{sd}", {2: ep[sd]}), (f"SRTTC_s{sd}", {3: ttc[sd]}), (f"JOINT_s{sd}", {2: ep[sd], 3: ttc[sd]})):
                ref, z = nw.oracle_picks(named)
                self.assertEqual(self._near_tie_free(z, pk[nm].astype(np.int64), ref), [], nm)
        ref_p2, z = nw.oracle_picks({})
        self.assertTrue(np.array_equal(ref_p2, pk["P2"].astype(np.int64)))
        self.assertGreater(int((pk["EPR1_s0"] != pk["P2"]).sum()), 5, "the synthetic EP row must actually move picks, or the oracle comparison proves nothing")
        self.assertGreater(int((pk["JOINT_s0"] != pk["EPR1_s0"]).sum()) + int((pk["JOINT_s0"] != pk["SRTTC_s0"]).sum()), 0, "the JOINT must differ from at least one single replacement")

    def test_the_ep_only_reading_is_not_the_joint_and_the_ttc_row_does_not_leak_into_it(self):
        nw = self.nav(ttc_scales=(6.0, 6.0))
        rep, pk, o = self.picks(nw)
        ep = seeds_rows(nw, "EP-R1")
        ref, z = nw.oracle_picks({2: ep["0"]})
        self.assertEqual(self._near_tie_free(z, pk["EPR1_s0"].astype(np.int64), ref), [])
        self.assertGreater(int((pk["JOINT_s0"] != pk["EPR1_s0"]).sum()), 0)

    def test_gain_equals_a_numpy_value_for_every_reading(self):
        nw = self.nav()
        o = nw.opts()
        res = C.stage_read(o)
        ar = np.arange(nw.N)
        self.assertFalse(os.path.exists(os.path.join(o.out_dir, "picks.npz")), "`read` writes its result and its seams, not the picks stage's picks.npz")
        led = json.loads(_rt(nw.ledger))
        self.assertEqual(sorted(led["picks_sha256"]), sorted(res["readings"].keys() | {"P2"}))
        w = nw.w
        ep, ttc = seeds_rows(nw, "EP-R1"), seeds_rows(nw, "SR-TTC")
        base = nw.oracle_picks({})[0]
        for sd in SK:
            for nm, named in ((f"EPR1_s{sd}", {2: ep[sd]}), (f"SRTTC_s{sd}", {3: ttc[sd]}), (f"JOINT_s{sd}", {2: ep[sd], 3: ttc[sd]})):
                new = nw.oracle_picks(named)[0]
                want = 100.0 * float(w.pdms[ar, new].mean() - w.pdms[ar, base].mean())
                got = res["readings"][nm]["gain_x100"]["point"]
                self.assertAlmostEqual(got, want, places=8, msg=nm)
                self.assertEqual(res["readings"][nm]["changed_picks"], int((new != base).sum()), nm)
        for sd in SK:
            self.assertGreater(res["readings"][f"EPR1_s{sd}"]["gain_x100"]["point"], 5.0, "the EP row along the label direction must gain on the synthetic world")

    def test_interval_is_the_paired_log_cluster_bootstrap_of_zero_tail(self):
        nw = self.nav()
        o = nw.opts()
        res = C.stage_read(o)
        tab = C.load_table_inputs(o.guard(), o)
        w = nw.w
        ar = np.arange(nw.N)
        ep = seeds_rows(nw, "EP-R1")
        new, base = nw.oracle_picks({2: ep["0"]})[0], nw.oracle_picks({})[0]
        boot = ZT.Boot(tab.log_idx, len(tab.log_names))
        ref = boot.ratio(w.pdms[ar, new] - w.pdms[ar, base], np.ones(nw.N), scale=100.0)
        got = res["readings"]["EPR1_s0"]["gain_x100"]
        for k in ("point", "lo", "hi"):
            self.assertAlmostEqual(got[k], ref[k], places=8, msg=k)
        self.assertEqual(got["B"], 10000)
        self.assertEqual(res["estimator"]["seed"], 20260927)
        self.assertEqual(res["estimator"]["n_logs"], nw.n_logs)

    def test_p2_pick_is_the_paired_baseline_and_the_p2_reading_is_not_a_row(self):
        nw = self.nav()
        res = C.stage_read(nw.opts())
        ar = np.arange(nw.N)
        self.assertAlmostEqual(res["p2_pick"]["pdms_x100"], 100.0 * float(nw.w.pdms[ar, nw.w.pick].mean()), places=8)
        self.assertNotIn("P2", res["readings"])


# bar clauses and PREREG s8 triggers on hand-built records ---------------------------------------------------------------------------------------------------------
def rec(point, lo, hi, **comp):
    comps = {nm: {"delta_x100": {"point": 0.0, "lo": -0.5, "hi": 0.5}} for nm in D.COMPS}
    for nm, (p, l, h) in comp.items():
        comps[nm] = {"delta_x100": {"point": p, "lo": l, "hi": h}}
    return {"gain_x100": {"point": point, "lo": lo, "hi": hi}, "components": comps}


class BarClauses(CBase):
    def test_clears(self):
        c = C.bar_clauses(rec(0.35, 0.05, 0.7, ep=(1.0, 0.2, 1.8)), "EPR1")
        self.assertTrue(c["clears_without_families6"])
        self.assertEqual(c["adverse_subscores"], [])

    def test_point_below_the_bar_does_not_clear(self):
        c = C.bar_clauses(rec(0.29, 0.05, 0.7), "EPR1")
        self.assertFalse(c["total_point_ge_bar"])
        self.assertFalse(c["clears_without_families6"])
        self.assertTrue(C.bar_clauses(rec(0.30, 0.05, 0.7), "EPR1")["total_point_ge_bar"], "the bar is >= +0.30")

    def test_lower_bound_not_above_zero_does_not_clear(self):
        c = C.bar_clauses(rec(0.5, 0.0, 1.0), "EPR1")
        self.assertFalse(c["total_lower_gt_0"])
        self.assertFalse(c["clears_without_families6"])

    def test_a_subscore_wholly_below_zero_blocks_epr1_and_srttc(self):
        for fam in ("EPR1", "SRTTC"):
            c = C.bar_clauses(rec(0.5, 0.1, 0.9, ttc=(-1.0, -1.5, -0.2)), fam)
            self.assertEqual(c["adverse_subscores"], ["ttc"])
            self.assertFalse(c["no_subscore_wholly_below_0"], fam)
            self.assertFalse(c["clears_without_families6"], fam)

    def test_the_ep_component_wholly_below_zero_blocks_epr1_only_through_its_own_clause(self):
        c = C.bar_clauses(rec(0.5, 0.1, 0.9, ep=(-1.0, -1.5, -0.2)), "EPR1")
        self.assertTrue(c["no_subscore_wholly_below_0"], "EPR1's five-component list excludes ep; ep has its own clause")
        self.assertFalse(c["ep_component_not_wholly_below_0"])
        self.assertFalse(c["clears_without_families6"])
        c2 = C.bar_clauses(rec(0.5, 0.1, 0.9, ep=(-1.0, -1.5, -0.2)), "SRTTC")
        self.assertFalse(c2["no_subscore_wholly_below_0"], "SR-TTC: none of the six sub-scores may be wholly below 0")
        self.assertNotIn("ep_component_not_wholly_below_0", c2)

    def test_epr1_rows_fired(self):
        self.assertEqual(C.navtest_rows_ep(rec(0.2, 0.05, 0.4, ep=(1.0, 0.2, 1.8))), ["E-SMALL"])
        self.assertEqual(C.navtest_rows_ep(rec(0.2, -0.1, 0.5, ep=(1.0, 0.2, 1.8))), ["E-NOTPROVEN"])
        self.assertIn("E-TRADE", C.navtest_rows_ep(rec(0.5, 0.1, 0.9, ep=(1.0, 0.2, 1.8), ttc=(-1.0, -2.0, -0.1))))
        self.assertIn("E-MECH", C.navtest_rows_ep(rec(-0.1, -0.4, 0.2)))
        fired = C.navtest_rows_ep(rec(-0.5, -0.9, -0.1))
        self.assertIn("E-REFUTED", fired)
        self.assertNotIn("E-SMALL", fired)
        self.assertEqual(C.navtest_rows_ep(rec(0.5, 0.1, 0.9, ep=(1.0, 0.2, 1.8))), [])

    def test_bar_mutants_are_red(self):
        low = rec(0.20, 0.05, 0.4)
        self.assertFalse(C.bar_clauses(low, "EPR1")["total_point_ge_bar"])
        mut = load_cmutant(("BAR", 'g["point"] >= BAR_X100', 'g["point"] >= -1.0'))
        self.assertTrue(mut.bar_clauses(low, "EPR1")["total_point_ge_bar"], "the mutant is the one that lets a +0.20 point clear a +0.30 bar")
        zero = rec(0.5, -0.1, 1.0)
        self.assertFalse(C.bar_clauses(zero, "EPR1")["total_lower_gt_0"])
        mut = load_cmutant(("BAR", 'g["lo"] > 0.0', 'g["lo"] > -9.0'))
        self.assertTrue(mut.bar_clauses(zero, "EPR1")["total_lower_gt_0"])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (c) the look opens only after both dev gates are read (or one arm is formally abandoned) -- refusals precede every other open
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class LookGating(CBase):
    def test_no_dev_result_at_all_refuses_before_the_table_or_the_bank_is_opened(self):
        nw = self.nav(outcomes=False, ep_dev=None, ttc_dev=None)
        o = nw.opts(table=os.path.join(self.tmp, "does_not_exist", "table_p2_full.npz"), bank_dir=os.path.join(self.tmp, "no_bank"))
        for stage in (C.stage_picks, lambda x: C.stage_read(x)):
            with self.assertRaises(C.LookNotOpen):
                stage(o)
        self.assertFalse(os.path.exists(o.out_dir), "nothing was written")
        self.assertFalse(os.path.exists(nw.ledger))

    def test_only_epr1_dev_result_exists_refuses_and_names_srttc(self):
        nw = self.nav(outcomes=False, ttc_dev=None)
        with self.assertRaises(C.LookNotOpen) as cm:
            C.stage_picks(nw.opts())
        self.assertIn("SR-TTC", str(cm.exception))

    def test_only_srttc_dev_result_exists_refuses_and_names_epr1(self):
        nw = self.nav(outcomes=False, ep_dev=None)
        with self.assertRaises(C.LookNotOpen) as cm:
            C.stage_picks(nw.opts())
        self.assertIn("EP-R1", str(cm.exception))

    def test_both_dev_results_open_the_look_and_all_readings_are_built_with_no_outcome_in_the_table(self):
        nw = self.nav(outcomes=False)
        rep, pk, o = self.picks(nw)
        self.assertEqual(rep["readings_open"], ["P2", "EPR1", "SRTTC", "JOINT"])
        self.assertEqual(sorted(pk), sorted(["token"] + [f"{f}_s{s}" for f in ("EPR1", "SRTTC", "JOINT") for s in SK] + ["P2"]))
        self.assertFalse(rep["look"]["spent"])
        self.assertFalse(os.path.exists(nw.ledger), "the picks stage never spends the look")
        self.assertEqual(sorted(os.listdir(os.path.join(o.out_dir, "seams"))), sorted(f"seam_{k}.npz" for k in pk if k != "token"))

    def test_epr1_dev_pass_and_srttc_formally_abandoned_reads_epr1_only(self):
        nw = self.nav(outcomes=False, ttc_dev=None)
        rep, pk, o = self.picks(nw, ttc_abandon=nw.abandon("SR-TTC"))
        self.assertEqual(rep["readings_open"], ["P2", "EPR1"])
        self.assertIn("SRTTC", rep["readings_not_open"])
        self.assertIn("JOINT", rep["readings_not_open"])
        self.assertEqual(sorted(k for k in pk if k != "token"), ["EPR1_s0", "EPR1_s1", "P2"])
        self.assertEqual(rep["inputs"]["ttc_dev"]["status"], "ABANDONED")
        self.assertIn("R-TEST-1", rep["inputs"]["ttc_dev"]["register_row"])

    def test_srttc_dev_pass_and_epr1_formally_abandoned_reads_srttc_only(self):
        nw = self.nav(outcomes=False, ep_dev=None)
        rep, pk, o = self.picks(nw, ep_abandon=nw.abandon("EP-R1"))
        self.assertEqual(rep["readings_open"], ["P2", "SRTTC"])
        self.assertEqual(sorted(k for k in pk if k != "token"), ["P2", "SRTTC_s0", "SRTTC_s1"])

    def test_both_arms_abandoned_refuses(self):
        nw = self.nav(outcomes=False, ep_dev=None, ttc_dev=None)
        with self.assertRaises(C.LookNotOpen) as cm:
            C.stage_picks(nw.opts(ep_abandon=nw.abandon("EP-R1"), ttc_abandon=nw.abandon("SR-TTC")))
        self.assertIn("nothing to read", str(cm.exception))

    def test_bad_abandonment_files_are_refused(self):
        nw = self.nav(outcomes=False, ttc_dev=None)
        for over in ({"register_row": ""}, {"register_row": None}, {"abandoned": False}, {"decided_by": " "}, {"reason": ""}, {"arm": "EP-R1"}):
            p = nw.abandon("SR-TTC", **over)
            with self.assertRaises(C.ConfirmRefusal, msg=str(over)):
                C.stage_picks(nw.opts(ttc_abandon=p))

    def test_a_dev_result_and_an_abandonment_for_the_same_arm_is_ambiguous_and_refused(self):
        nw = self.nav(outcomes=False)
        with self.assertRaises(C.ConfirmRefusal) as cm:
            C.stage_picks(nw.opts(ttc_abandon=nw.abandon("SR-TTC")))
        self.assertIn("BOTH", str(cm.exception))

    def test_epr1_fail_or_void_is_not_read_on_navtest_but_srttc_follows_its_own_dev_gate(self):
        for v in ("FAIL", "VOID"):
            nw = self.nav(outcomes=False, ep_dev=v)
            rep, pk, o = self.picks(nw)
            self.assertEqual(rep["readings_open"], ["P2", "SRTTC"], v)
            self.assertIn("EPR1", rep["readings_not_open"])
            self.assertIn("JOINT", rep["readings_not_open"], "the JOINT is EP-R1's secondary: no EP-R1 pass, no joint")
            self.assertNotIn("EPR1_s0", pk)

    def test_srttc_fail_with_epr1_pass_reads_epr1_and_the_joint(self):
        nw = self.nav(outcomes=False, ttc_dev="FAIL")
        rep, pk, o = self.picks(nw)
        self.assertEqual(rep["readings_open"], ["P2", "EPR1", "JOINT"])
        self.assertIn("SRTTC", rep["readings_not_open"])
        self.assertNotIn("SRTTC_s0", pk)

    def test_srttc_abandoned_with_epr1_pass_has_no_joint(self):
        nw = self.nav(outcomes=False, ttc_dev=None)
        rep, pk, o = self.picks(nw, ttc_abandon=nw.abandon("SR-TTC"))
        self.assertNotIn("JOINT_s0", pk)
        self.assertIn("abandoned", rep["readings_not_open"]["JOINT"])

    def test_no_arm_passed_leaves_the_look_unspent(self):
        for ev, tv in (("FAIL", "FAIL"), ("VOID", "FAIL"), ("FAIL", "VOID")):
            nw = self.nav(ep_dev=ev, ttc_dev=tv)
            with self.assertRaises(C.LookNotOpen):
                C.stage_read(nw.opts())
            self.assertFalse(os.path.exists(nw.ledger), (ev, tv))

    def test_malformed_dev_results_are_refused(self):
        nw = self.nav(outcomes=False)
        good = nw.ep_dev_doc("PASS")
        for name, doc in (("schema", {**good, "schema": "other"}), ("stage", {**good, "stage": "R1"}), ("verdict", {**good, "dev_gate": {"verdict": "MAYBE"}}),
                          ("seeds", {**good, "inputs": {**good["inputs"], "rows_sha256": {"0": "x"}}}), ("result_sha", {**good, "inputs": {"rows_sha256": good["inputs"]["rows_sha256"]}})):
            wj(nw.ep_dev_path, doc)
            with self.assertRaises(C.ConfirmRefusal, msg=name):
                C.stage_picks(nw.opts())

    def test_eligibility_table(self):
        P, F, V, A = ({"status": "READ", "verdict": "PASS"}, {"status": "READ", "verdict": "FAIL"}, {"status": "READ", "verdict": "VOID"}, {"status": "ABANDONED"})
        cases = [(P, P, ["P2", "EPR1", "SRTTC", "JOINT"]), (P, F, ["P2", "EPR1", "JOINT"]), (P, V, ["P2", "EPR1", "JOINT"]), (P, A, ["P2", "EPR1"]), (F, P, ["P2", "SRTTC"]),
                 (V, P, ["P2", "SRTTC"]), (A, P, ["P2", "SRTTC"])]
        for ep, ttc, want in cases:
            self.assertEqual(C.eligible_readings(ep, ttc)[0], want, (ep, ttc))
        for ep, ttc in ((F, F), (F, A), (A, F), (V, V)):
            with self.assertRaises(C.LookNotOpen, msg=(ep, ttc)):
                C.eligible_readings(ep, ttc)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# registered objects: the sha chain from the registered c* through the dev result to the rows applied now
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class ShaChain(CBase):
    def test_unregistered_or_wrong_registered_sha_is_refused(self):
        nw = self.nav(outcomes=False)
        for sha in (None, "0" * 64):
            with self.assertRaises(C.ConfirmRefusal, msg=str(sha)):
                C.prepare(nw.opts(ep_registered_result_sha256=sha))
            with self.assertRaises(C.ConfirmRefusal, msg=str(sha)):
                C.prepare(nw.opts(ttc_registered_result_sha256=sha))

    def test_result_edited_after_the_dev_gate_read_it_is_refused_even_if_re_registered(self):
        nw = self.nav(outcomes=False)
        d = dict(nw.w.r1)
        d["note"] = "edited after the dev gate"
        wj(os.path.join(nw.w.run_dir, "epr1_result.json"), d)
        new_sha = nw.w.r1_sha()
        with self.assertRaises(C.ConfirmRefusal) as cm:
            C.prepare(nw.opts(ep_registered_result_sha256=new_sha))
        self.assertIn("dev-gate result recorded", str(cm.exception))

    def test_a_row_file_tampered_after_the_fit_is_refused(self):
        nw = self.nav(outcomes=False)
        p = os.path.join(nw.w.run_dir, "ep_row_seed1.npy")
        blob = bytearray(_rb(p))
        blob[-1] ^= 0x01
        with open(p, "wb") as f:
            f.write(bytes(blob))
        with self.assertRaises((C.ConfirmRefusal, D.DevRefusal)):
            C.prepare(nw.opts())

    def test_rows_that_are_not_the_dev_gated_rows_are_refused(self):
        nw = self.nav(outcomes=False)
        rows = {sd: nw.w.r1["final_rows"][sd]["sha256"] for sd in SK}
        rows["1"] = "f" * 64
        nw.write_ep_dev("PASS", rows=rows)
        with self.assertRaises(C.ConfirmRefusal) as cm:
            C.prepare(nw.opts())
        self.assertIn("dev gate applied", str(cm.exception))

    def test_rows_fitted_from_another_p2_are_refused(self):
        nw = self.nav(outcomes=False)
        sd_ = torch.load(nw.w.ckpt, weights_only=True)
        sd_["model"]["score_head.weight"][2] += 0.01
        torch.save(sd_, nw.w.ckpt)
        with self.assertRaises(C.ConfirmRefusal) as cm:
            C.prepare(nw.opts())
        self.assertIn("different P2", str(cm.exception))

    def test_code_pin_table_pin_and_shape_pins(self):
        nw = self.nav(outcomes=False)
        with self.assertRaises(C.ConfirmRefusal):
            C.prepare(nw.opts(epr1_sha256="0" * 64))
        with self.assertRaises(C.ConfirmRefusal):
            C.stage_picks(nw.opts(table_sha256="0" * 64))
        with self.assertRaises(C.ConfirmRefusal):
            C.stage_picks(nw.opts(expect_tokens=nw.N + 1))
        with self.assertRaises(C.ConfirmRefusal):
            C.stage_picks(nw.opts(expect_logs=nw.n_logs + 1))
        self.assertEqual(set(nw.opts().pins_overridden()), {"table_sha256", "expect_tokens", "expect_logs"}, "a run that overrides a pin says so in its report")

    def test_pins_overridden_is_empty_for_the_real_defaults(self):
        o = C.ConfirmOptions(out_dir="x", bank_dir="y")
        self.assertEqual(o.pins_overridden(), [])

    def test_joint_is_dropped_loudly_when_the_two_rows_were_not_fitted_on_the_same_logs(self):
        nw = self.nav(outcomes=False)
        d = dict(nw.w.ttc_r1)
        d["fraction_pct"] = 25
        wj(os.path.join(nw.w.ttc_dir, "r1_result.json"), d)
        sha = sha_f(os.path.join(nw.w.ttc_dir, "r1_result.json"))
        nw.ttc_sha = sha
        nw.write_ttc_dev("PASS", result_sha=sha)
        prep = C.prepare(nw.opts(ttc_registered_result_sha256=sha))
        self.assertEqual(prep["families"], ["P2", "EPR1", "SRTTC"])
        self.assertIn("fraction_pct", prep["why_not"]["JOINT"])
        self.assertIn("EPR1_s0", prep["plan"])
        self.assertNotIn("JOINT_s0", prep["plan"])

    def test_joint_compat_reasons(self):
        a = {"fraction_pct": 10, "inputs": {"bank_manifest_sha256": "m", "ranked_fit_logs_sha256": "r", "replacements": {"n": 1}}}
        self.assertEqual(C.joint_compat(a, json.loads(json.dumps(a))), [])
        for k, v in (("fraction_pct", 25),):
            b = json.loads(json.dumps(a))
            b[k] = v
            self.assertEqual(len(C.joint_compat(a, b)), 1)
        for k in ("bank_manifest_sha256", "ranked_fit_logs_sha256", "replacements"):
            b = json.loads(json.dumps(a))
            b["inputs"][k] = "other"
            self.assertEqual(len(C.joint_compat(a, b)), 1, k)
            del b["inputs"][k]
            self.assertEqual(len(C.joint_compat(a, b)), 1, k + " missing")

    def test_real_epr1_dev_result_feeds_the_chain_end_to_end(self):
        """The EP-R1 dev gate's OWN output file (D.run_dev on a synthetic dev world) is accepted by this module's reader: the contract is not just self-consistent."""
        nw = self.nav(outcomes=True, ep_dev=None)
        w = nw.w
        res = D.run_dev(w.opts())
        self.assertEqual(res["dev_gate"]["verdict"], "PASS")
        path = os.path.join(w.opts().resolved_out_dir(), D.RESULT_NAME)
        st = C.read_dev_result(C.ConfirmGuard(read_dirs=[os.path.dirname(path)]), path, "EP-R1")
        self.assertEqual(st["verdict"], "PASS")
        self.assertEqual(st["result_sha256"], w.r1_sha())
        self.assertEqual(st["rows_sha256"], {sd: w.r1["final_rows"][sd]["sha256"] for sd in SK})
        prep = C.prepare(nw.opts(ep_dev_result=path))
        self.assertIn("EPR1", prep["families"])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the feature bank
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Bank(CBase):
    def assertRefusedNoSeams(self, nw, contains=None, **kw):
        o = nw.opts(**kw)
        with self.assertRaises((C.ConfirmRefusal, D.DevRefusal)) as cm:
            C.stage_picks(o)
        if contains:
            self.assertIn(contains, str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(o.out_dir, "seams")), "a refused run writes no seam")

    def test_missing_manifest_is_refused(self):
        nw = self.nav(outcomes=False)
        os.remove(os.path.join(nw.bank_dir, C.BANK_MANIFEST))
        self.assertRefusedNoSeams(nw, "MANIFEST")

    def test_a_chunk_that_is_not_in_the_manifest_is_refused(self):
        nw = self.nav(outcomes=False)
        shutil.copy(os.path.join(nw.bank_dir, "c00_feat.npz"), os.path.join(nw.bank_dir, "c09_feat.npz"))
        shutil.copy(os.path.join(nw.bank_dir, "c00_dump.npz"), os.path.join(nw.bank_dir, "c09_dump.npz"))
        self.assertRefusedNoSeams(nw, "not in the manifest")

    def test_a_tampered_chunk_is_refused(self):
        nw = self.nav(outcomes=False)
        p = os.path.join(nw.bank_dir, "c01_feat.npz")
        z = np.load(p)
        d = {k: z[k] for k in z.files}
        z.close()
        d["note"] = np.array("same arrays, other bytes")
        np.savez(p, **d)
        self.assertRefusedNoSeams(nw, "manifest")

    def test_an_incomplete_bank_is_refused(self):
        nw = self.nav(outcomes=False)
        nw.write_bank(skip=(2,))
        self.assertRefusedNoSeams(nw, "INCOMPLETE")

    def test_a_token_in_two_chunks_is_refused(self):
        nw = self.nav(outcomes=False)
        nw.write_bank(extra_dup=1)
        self.assertRefusedNoSeams(nw, "already appeared")

    def test_a_token_outside_the_table_is_refused(self):
        nw = self.nav(outcomes=False)
        p = os.path.join(nw.bank_dir, "c00_feat.npz")
        z = np.load(p)
        d = {k: z[k] for k in z.files}
        z.close()
        d["token"] = d["token"].copy()
        d["token"][0] = "ffffffffffffffff"
        np.savez(p, **d)
        nw.write_manifest()
        self.assertRefusedNoSeams(nw, "not navtest-table tokens")

    def test_a_missing_dump_is_refused_when_c4_needs_it(self):
        nw = self.nav(outcomes=False)
        os.remove(os.path.join(nw.bank_dir, "c00_dump.npz"))
        nw.write_manifest()
        self.assertRefusedNoSeams(nw, "c00_dump")

    def test_log_names_that_differ_from_the_tables_are_refused(self):
        nw = self.nav(outcomes=False)
        p = os.path.join(nw.bank_dir, "c00_feat.npz")
        z = np.load(p)
        d = {k: z[k] for k in z.files}
        z.close()
        d["log"] = d["log"][::-1].copy()
        np.savez(p, **d)
        nw.write_manifest()
        self.assertRefusedNoSeams(nw, "log names differ")

    def test_features_that_do_not_reproduce_the_stored_logits_make_the_integrity_control_red(self):
        nw = self.nav(outcomes=False)
        bad = nw.w.feat.copy()
        bad[:, :, 10] = (bad[:, :, 10].astype(np.float32) + 0.5).astype(np.float16)
        nw.write_bank(feat=bad)
        o = nw.opts()
        core = C.run_picks_core(o, C.prepare(o))
        self.assertFalse(core["ctl"]["integrity_featcheck"]["pass"])
        self.assertGreater(core["ctl"]["integrity_featcheck"]["max_abs_featcheck"], C.FEATCHECK_TOL)
        self.assertRefusedNoSeams(nw, "integrity_featcheck")

    def test_featcheck_negative_control_reads_far_above_the_tolerance_on_the_good_bank(self):
        nw = self.nav(outcomes=False)
        o = nw.opts()
        core = C.run_picks_core(o, C.prepare(o))
        f = core["ctl"]["integrity_featcheck"]
        self.assertTrue(f["pass"])
        self.assertLess(f["max_abs_featcheck"], 1e-4)
        self.assertGreater(f["negative_control_next_token_min_max_abs"], 10 * C.FEATCHECK_TOL)

    def test_c4_red_on_a_one_millimetre_dump_perturbation(self):
        nw = self.nav(outcomes=False)
        pr = nw.w.props.copy()
        pr[5, 7, 3, 1] += 1e-3
        nw.write_bank(props=pr)
        o = nw.opts()
        core = C.run_picks_core(o, C.prepare(o))
        self.assertFalse(core["ctl"]["C4_fan_frozen"]["pass"])
        self.assertAlmostEqual(core["ctl"]["C4_fan_frozen"]["proposals_max_abs_m"], 1e-3, places=6)
        self.assertRefusedNoSeams(nw, "C4_fan_frozen")

    def test_c4_red_when_the_dump_pick_differs_from_the_tables(self):
        nw = self.nav(outcomes=False)
        pk = nw.w.pick.copy()
        pk[3] = (pk[3] + 1) % NP_
        nw.write_bank(pick=pk)
        o = nw.opts()
        core = C.run_picks_core(o, C.prepare(o))
        self.assertFalse(core["ctl"]["C4_fan_frozen"]["pass"])
        self.assertFalse(core["ctl"]["C4_fan_frozen"]["picks_equal"])

    def test_c4_red_when_the_dump_logits_differ(self):
        nw = self.nav(outcomes=False)
        lg = nw.w.logits.copy()
        lg[2, 2, 2] += 0.01
        nw.write_bank(logits=lg)
        o = nw.opts()
        core = C.run_picks_core(o, C.prepare(o))
        self.assertFalse(core["ctl"]["C4_fan_frozen"]["pass"])

    def test_c0_red_when_the_banked_pick_is_not_the_rule_pick(self):
        nw = self.nav(outcomes=False)
        pk = nw.w.pick.copy()
        pk[4] = (pk[4] + 1) % NP_
        nw.write_table(pick=pk)
        nw.write_bank(pick=pk)
        o = nw.opts()
        core = C.run_picks_core(o, C.prepare(o))
        self.assertFalse(core["ctl"]["C0_identity"]["pass"])
        self.assertTrue(core["ctl"]["C4_fan_frozen"]["pass"])

    def test_the_banked_p2_seam_must_match_when_given(self):
        nw = self.nav(outcomes=False)
        z = np.load(nw.seam)
        d = {k: z[k] for k in z.files}
        z.close()
        d["poses"] = d["poses"].copy()
        d["poses"][0, 0, 0] += 0.01
        np.savez(nw.seam, **d)
        o = nw.opts()
        core = C.run_picks_core(o, C.prepare(o))
        self.assertFalse(core["ctl"]["banked_p2_seam"]["pass"])
        self.assertAlmostEqual(core["ctl"]["banked_p2_seam"]["max_abs_pose_m"], 0.01, places=5)

    def test_the_bank_is_mapped_by_token_not_by_order(self):
        nw = self.nav(outcomes=False)
        order = np.concatenate(nw.chunks)
        self.assertFalse(np.array_equal(order, np.arange(nw.N)), "the synthetic bank must be in another order than the table")
        rep, pk, o = self.picks(nw)
        self.assertTrue(np.array_equal(pk["P2"].astype(np.int64), nw.w.pick))
        self.assertTrue(np.array_equal(pk["token"], nw.w.token))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (d) the look is spent once
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class LookSpent(CBase):
    def test_second_read_is_refused_and_the_first_wrote_the_ledger(self):
        nw = self.nav()
        res = C.stage_read(nw.opts())
        self.assertTrue(os.path.exists(nw.ledger))
        self.assertTrue(res["look"]["spent"])
        with self.assertRaises(C.ConfirmRefusal) as cm:
            C.stage_read(nw.opts())
        self.assertIn("SPENT", str(cm.exception))

    def test_a_read_that_crashes_after_the_ledger_still_consumed_the_look(self):
        nw = self.nav(outcomes=False)                                       # no pdms / sub: the outcome open raises
        with self.assertRaises(KeyError):
            C.stage_read(nw.opts())
        self.assertTrue(os.path.exists(nw.ledger), "the ledger was written BEFORE the outcome arrays were touched")
        nw.write_table(outcomes=True)
        with self.assertRaises(C.ConfirmRefusal):
            C.stage_read(nw.opts(table_sha256=sha_f(nw.table)))

    def test_red_control_leaves_the_look_unspent(self):
        nw = self.nav()
        pr = nw.w.props.copy()
        pr[0, 0, 0, 0] += 1e-3
        nw.write_bank(props=pr)
        with self.assertRaises(C.ConfirmRefusal) as cm:
            C.stage_read(nw.opts())
        self.assertIn("UNSPENT", str(cm.exception))
        self.assertFalse(os.path.exists(nw.ledger))
        nw.write_bank()                                                      # repaired
        C.stage_read(nw.opts())
        self.assertTrue(os.path.exists(nw.ledger))

    def test_the_picks_stage_can_be_repeated_on_fresh_dirs_and_never_spends(self):
        nw = self.nav(outcomes=False)
        r1, p1, _ = self.picks(nw)
        r2, p2, _ = self.picks(nw)
        for k in p1:
            self.assertTrue(np.array_equal(p1[k], p2[k]), k)
        self.assertFalse(os.path.exists(nw.ledger))

    def test_picks_refuses_an_out_dir_that_already_holds_a_report(self):
        nw = self.nav(outcomes=False)
        o = nw.opts()
        C.stage_picks(o)
        with self.assertRaises(C.ConfirmRefusal):
            C.stage_picks(nw.opts(out_dir=o.out_dir))

    def test_read_needs_a_ledger_path(self):
        nw = self.nav()
        with self.assertRaises(C.ConfirmRefusal) as cm:
            C.stage_read(nw.opts(look_ledger=None))
        self.assertIn("look-ledger", str(cm.exception))

    def test_read_is_deterministic(self):
        a, b = self.nav(), self.nav()
        ra, rb = C.stage_read(a.opts()), C.stage_read(b.opts())
        for k in ("readings", "primaries", "p2_pick", "estimator"):
            self.assertEqual(json.dumps(D._jsonable(ra[k]), sort_keys=True), json.dumps(D._jsonable(rb[k]), sort_keys=True), k)

    def test_the_ledger_records_the_rows_picks_and_inputs_it_spent_the_look_on(self):
        nw = self.nav()
        res = C.stage_read(nw.opts())
        led = json.loads(_rt(nw.ledger))
        self.assertEqual(led["readings"], ["P2", "EPR1", "SRTTC", "JOINT"])
        self.assertEqual(led["table_sha256"], sha_f(nw.table))
        self.assertEqual(sorted(led["rows_sha256"]), ["EP-R1", "SR-TTC"])
        self.assertEqual(res["look"]["ledger"], nw.ledger)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# what the read reports
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class ReadResult(CBase):
    def test_structure_labels_and_disclosures(self):
        nw = self.nav()
        res = C.stage_read(nw.opts())
        self.assertEqual(sorted(res["readings"]), sorted(f"{f}_s{s}" for f in ("EPR1", "SRTTC", "JOINT") for s in SK))
        self.assertIn("SECONDARY", res["readings"]["JOINT_s0"]["label"])
        self.assertIn("vs_ep_r1_alone", res["readings"]["JOINT_s0"])
        self.assertIn("vs_ttc_alone", res["readings"]["JOINT_s0"])
        self.assertIn("bar_clauses_informational", res["readings"]["JOINT_s0"])
        self.assertNotIn("bar_clauses", res["readings"]["JOINT_s0"], "the JOINT has no primary bar")
        self.assertIn("no family-wise correction", res["multiplicity_disclosure"])
        for arm in ("EP-R1", "SR-TTC"):
            p = res["primaries"][arm]
            self.assertIn("PENDING", p["families6"])
            self.assertIn("PAUSED", p["strategic"])
            self.assertIn("UNAVAILABLE", p["distance_keeping"])
        self.assertIsInstance(res["readings"]["EPR1_s0"]["navtest_rows_fired"], list)
        self.assertTrue(all(res["controls"][k]["pass"] for k in res["controls"]))
        self.assertIn("T1", res["tier"])

    def test_an_adverse_subscore_blocks_the_primary_end_to_end(self):
        """On the default synthetic world seed 0 gains +13 with a separated interval, yet its TTC component is wholly below 0 (a selection side effect): the clause must block it."""
        nw = self.nav()
        res = C.stage_read(nw.opts())
        c = res["readings"]["EPR1_s0"]["bar_clauses"]
        self.assertTrue(c["total_point_ge_bar"] and c["total_lower_gt_0"])
        self.assertTrue(c["adverse_subscores"], "the synthetic world must exhibit an adverse sub-score for this test to prove anything")
        self.assertFalse(c["clears_without_families6"])
        self.assertFalse(res["primaries"]["EP-R1"]["both_seeds_clear_all_clauses_except_families6"])

    def test_each_primary_states_the_other_reading_beside_it(self):
        nw = self.nav(clean_subs=True)
        res = C.stage_read(nw.opts())
        self.assertEqual(set(res["primaries"]["EP-R1"]["other_primary_beside_it"]), {"SR-TTC"})
        self.assertEqual(res["primaries"]["EP-R1"]["other_primary_beside_it"]["SR-TTC"], res["primaries"]["SR-TTC"]["both_seeds_clear_all_clauses_except_families6"])
        self.assertEqual(res["primaries"]["SR-TTC"]["other_primary_beside_it"]["EP-R1"], res["primaries"]["EP-R1"]["both_seeds_clear_all_clauses_except_families6"])
        nw2 = self.nav(ttc_dev="FAIL")
        res2 = C.stage_read(nw2.opts())
        self.assertIn("not read", res2["primaries"]["EP-R1"]["other_primary_beside_it"]["SR-TTC"])
        self.assertIn("FAIL", res2["primaries"]["EP-R1"]["other_primary_beside_it"]["SR-TTC"])

    def test_joint_cannot_decide_a_primary(self):
        """Only the EP-R1 / SR-TTC readings feed `primaries`; a JOINT that clears every clause moves neither verdict."""
        nw = self.nav(ep_scales=(1e-6, 1e-6))
        res = C.stage_read(nw.opts())
        self.assertFalse(res["primaries"]["EP-R1"]["both_seeds_clear_all_clauses_except_families6"])
        self.assertEqual(set(res["primaries"]), {"EP-R1", "SR-TTC"})

    def test_both_seeds_rule_one_seed_clearing_is_not_enough(self):
        nw = self.nav(ep_scales=(3.0, 1e-6), clean_subs=True)
        res = C.stage_read(nw.opts())
        c0 = res["readings"]["EPR1_s0"]["bar_clauses"]
        c1 = res["readings"]["EPR1_s1"]["bar_clauses"]
        self.assertTrue(c0["clears_without_families6"])
        self.assertFalse(c1["total_point_ge_bar"])
        self.assertFalse(res["primaries"]["EP-R1"]["both_seeds_clear_all_clauses_except_families6"])
        self.assertEqual(res["primaries"]["EP-R1"]["per_seed"]["0"], c0)

    def test_both_seeds_clearing_is_reported_as_a_candidate_pending_families6(self):
        nw = self.nav(ep_scales=(3.0, 3.0), clean_subs=True)
        res = C.stage_read(nw.opts())
        for sd in SK:
            self.assertTrue(res["readings"][f"EPR1_s{sd}"]["bar_clauses"]["clears_without_families6"], sd)
        self.assertTrue(res["primaries"]["EP-R1"]["both_seeds_clear_all_clauses_except_families6"])
        self.assertIn("CANDIDATE", res["primaries"]["EP-R1"]["families6"])
        self.assertEqual(res["readings"]["EPR1_s0"]["navtest_rows_fired"], [], "no trigger fires on a clean clearing seed")

    def test_read_with_one_arm_abandoned_has_only_the_open_arms_primaries(self):
        nw = self.nav(ttc_dev=None)
        res = C.stage_read(nw.opts(ttc_abandon=nw.abandon("SR-TTC")))
        self.assertEqual(set(res["primaries"]), {"EP-R1"})
        self.assertEqual(sorted(res["readings"]), ["EPR1_s0", "EPR1_s1"])

    def test_the_read_writes_its_seams_before_the_look_and_they_are_loadable_w3_seams(self):
        nw = self.nav()
        o = nw.opts()
        res = C.stage_read(o)
        ready = json.loads(_rt(os.path.join(o.out_dir, "ready_for_score.json")))
        self.assertEqual(len(ready["readings"]), 7)
        self.assertIn("AFTER", ready["WARNING"])
        self.assertIn("--label", ready["readings"][0]["command"])
        self.assertIn("--seam", ready["readings"][0]["command"])
        for r in ready["readings"]:
            self.assertTrue(r["label"].startswith("refe"), "score_navtest_refe.py refuses a label that does not start with 'refe'")
            z = np.load(r["seam"])
            self.assertEqual(sorted(z.files), ["arm", "fingerprint", "poses", "sampling", "token"])
            self.assertEqual(list(z["sampling"]), [8.0, 0.5])
            self.assertTrue(np.array_equal(z["token"], nw.w.token))
            self.assertTrue(np.array_equal(z["fingerprint"], nw.fp))
            self.assertIn(r["reading"], str(z["arm"]))
        self.assertEqual(res["seams"]["EPR1_s0"]["changed_picks_vs_p2"], res["readings"]["EPR1_s0"]["changed_picks"])
        # a seam's poses are exactly the table's executed pose of the reading's pick
        z = np.load(res["seams"]["EPR1_s0"]["path"])
        ep = seeds_rows(nw, "EP-R1")
        new = nw.oracle_picks({2: ep["0"]})[0]
        agree = np.all(z["poses"] == nw.w.props[np.arange(nw.N), new], axis=(1, 2))
        self.assertGreaterEqual(float(agree.mean()), 0.99)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# applier regressions: each control must go RED when the applier is wrong (monkeypatched D.apply_rows)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class ApplierRegressions(CBase):
    def _core(self, nw):
        o = nw.opts()
        return C.run_picks_core(o, C.prepare(o))["ctl"]

    def test_c0_red_when_the_applier_shifts_logits_even_for_p2s_own_rows(self):
        nw = self.nav(outcomes=False)
        orig = D.apply_rows

        def shifted(z6, feat, rows, W6, b6, chunk=512):
            return orig(z6, feat, rows, W6, b6, chunk) + np.float32(1e-6)
        with mock.patch.object(D, "apply_rows", shifted):
            ctl = self._core(nw)
        self.assertFalse(ctl["C0_identity"]["pass"])
        self.assertFalse(ctl["C0_identity"]["logits_bit_identical_with_p2_rows"])
        self.assertTrue(ctl["integrity_featcheck"]["pass"] and ctl["C4_fan_frozen"]["pass"])

    def test_c1_red_when_the_row_never_reaches_the_logits(self):
        nw = self.nav(outcomes=False)
        with mock.patch.object(D, "apply_rows", lambda z6, feat, rows, W6, b6, chunk=512: z6.numpy().copy()):
            ctl = self._core(nw)
        self.assertFalse(ctl["C1_reach"]["pass"])
        self.assertTrue(ctl["C0_identity"]["pass"])

    def test_c1_red_when_the_row_lands_in_the_wrong_column(self):
        nw = self.nav(outcomes=False)
        orig = D.apply_rows

        def wrong(z6, feat, rows, W6, b6, chunk=512):
            return orig(z6, feat, {({2: 4, 3: 5}.get(c, c)): v for c, v in rows.items()}, W6, b6, chunk)
        with mock.patch.object(D, "apply_rows", wrong):
            ctl = self._core(nw)
        self.assertFalse(ctl["C1_reach"]["pass"])

    def test_c1_red_when_picks_would_ignore_the_new_logits(self):
        nw = self.nav(outcomes=False)
        o = nw.opts()
        with mock.patch.object(D, "pick_logits", lambda z6: np.full(z6.shape[0], 5, np.int64)):
            core = C.run_picks_core(o, C.prepare(o))
        P = core["stats"]["picks"]
        self.assertTrue(all(np.array_equal(P[k], P["P2"]) for k in P))
        self.assertEqual(core["ctl"]["C1_reach"]["positive_control_negated_p2_ep_row_changed_picks"], 0)
        self.assertFalse(core["ctl"]["C1_reach"]["pass"])
        self.assertTrue(core["ctl"]["C1_reach"]["per_reading"]["EPR1_s0"]["entries_changed_in_named_columns"] > 0, "the LOGITS changed: it is the picks that ignore them")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (e) deliberate-regression arms on the module's own gates
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Mutants(CBase):
    def test_dev_result_missing_mutant(self):
        nw = self.nav(outcomes=False, ep_dev=None)
        o = nw.opts()
        with self.assertRaises(C.LookNotOpen):
            C.prepare(o)
        mut = load_cmutant(("DVMISSING", "if not dv:", "if False:"))
        with self.assertRaises(Exception) as cm:
            mut.prepare(mopts(mut, o))
        self.assertNotIsInstance(cm.exception, (mut.LookNotOpen, C.LookNotOpen), "the mutant must be the one that tries to read a dev result that does not exist")

    def test_epr1_eligibility_mutant(self):
        F, P = {"status": "READ", "verdict": "FAIL"}, {"status": "READ", "verdict": "PASS"}
        self.assertEqual(C.eligible_readings(F, P)[0], ["P2", "SRTTC"])
        mut = load_cmutant(("EPELIG", 'and ep["verdict"] == "PASS"', ""))
        self.assertIn("EPR1", mut.eligible_readings(F, P)[0])

    def test_srttc_eligibility_mutant(self):
        F, P = {"status": "READ", "verdict": "FAIL"}, {"status": "READ", "verdict": "PASS"}
        self.assertEqual(C.eligible_readings(P, F)[0], ["P2", "EPR1", "JOINT"])
        mut = load_cmutant(("TTCELIG", 'and ttc["verdict"] == "PASS"', ""))
        self.assertIn("SRTTC", mut.eligible_readings(P, F)[0])

    def test_joint_eligibility_mutant(self):
        P, A = {"status": "READ", "verdict": "PASS"}, {"status": "ABANDONED"}
        self.assertNotIn("JOINT", C.eligible_readings(P, A)[0])
        mut = load_cmutant(("JOINTELIG", 'ttc["status"] == "READ"', "True"))
        self.assertIn("JOINT", mut.eligible_readings(P, A)[0])

    def test_registered_sha_mutant(self):
        nw = self.nav(outcomes=False)
        o = nw.opts(ep_registered_result_sha256="0" * 64)
        with self.assertRaises(C.ConfirmRefusal):
            C.prepare(o)
        mut = load_cmutant(("REGSHA", "sha != registered_sha", "False"))
        self.assertIn("EPR1", mut.prepare(mopts(mut, o))["families"])

    def test_dev_chain_mutant(self):
        nw = self.nav(outcomes=False)
        d = dict(nw.w.r1)
        d["note"] = "edited after the dev gate"
        wj(os.path.join(nw.w.run_dir, "epr1_result.json"), d)
        o = nw.opts(ep_registered_result_sha256=nw.w.r1_sha())
        with self.assertRaises(C.ConfirmRefusal):
            C.prepare(o)
        mut = load_cmutant(("DEVCHAIN", 'sha != state["result_sha256"]', "False"))
        self.assertIn("EPR1", mut.prepare(mopts(mut, o))["families"])

    def test_row_sha_mutant(self):
        nw = self.nav(outcomes=False)
        rows = {sd: nw.w.r1["final_rows"][sd]["sha256"] for sd in SK}
        rows["0"] = "e" * 64
        nw.write_ep_dev("PASS", rows=rows)
        o = nw.opts()
        with self.assertRaises(C.ConfirmRefusal):
            C.prepare(o)
        mut = load_cmutant(("ROWSHA", 'rsha != state["rows_sha256"][sd]', "False"))
        self.assertIn("EPR1", mut.prepare(mopts(mut, o))["families"])

    def test_p2_row_mutant(self):
        nw = self.nav(outcomes=False)
        sd_ = torch.load(nw.w.ckpt, weights_only=True)
        sd_["model"]["score_head.weight"][3] += 0.01
        torch.save(sd_, nw.w.ckpt)
        o = nw.opts()
        with self.assertRaises(C.ConfirmRefusal):
            C.prepare(o)
        mut = load_cmutant(("P2ROW", '!= (doc.get("inputs") or {}).get("p2_row_sha256")', "== None"))
        self.assertIn("SRTTC", mut.prepare(mopts(mut, o))["families"])

    def test_table_sha_mutant(self):
        nw = self.nav(outcomes=False)
        o = nw.opts(table_sha256="0" * 64)
        with self.assertRaises(C.ConfirmRefusal):
            C.load_table_inputs(o.guard(), o)
        mut = load_cmutant(("TABLESHA", "sha != opts.table_sha256", "False"))
        self.assertEqual(mut.load_table_inputs(mopts(mut, o).guard(), mopts(mut, o)).n, nw.N)

    def test_bank_sha_mutant(self):
        nw = self.nav(outcomes=False)
        p = os.path.join(nw.bank_dir, "c01_feat.npz")
        z = np.load(p)
        d = {k: z[k] for k in z.files}
        z.close()
        d["note"] = np.array("same arrays, other bytes")
        np.savez(p, **d)
        o = nw.opts()
        with self.assertRaises(C.ConfirmRefusal):
            C.run_picks_core(o, C.prepare(o))
        mut = load_cmutant(("BANKSHA", 'sha != man["files"][fn]', "False"))
        mo = mopts(mut, o)
        core = mut.run_picks_core(mo, mut.prepare(mo))
        self.assertTrue(core["ctl"]["C0_identity"]["pass"])

    def test_duplicate_token_mutant(self):
        nw = self.nav(outcomes=False)
        nw.write_bank(extra_dup=1)
        o = nw.opts()
        with self.assertRaises(C.ConfirmRefusal):
            C.run_picks_core(o, C.prepare(o))
        mut = load_cmutant(("DUPTOKEN", "if seen[rows].any():", "if False:"))
        mo = mopts(mut, o)
        core = mut.run_picks_core(mo, mut.prepare(mo))
        self.assertEqual(core["stats"]["n_tokens_covered"], nw.N, "the mutant is the one that accepts a token twice")

    def test_incomplete_bank_mutant(self):
        nw = self.nav(outcomes=False)
        nw.write_bank(skip=(2,))
        o = nw.opts()
        with self.assertRaises(C.ConfirmRefusal) as cm:
            C.run_picks_core(o, C.prepare(o))
        self.assertIn("INCOMPLETE", str(cm.exception))
        mut = load_cmutant(("INCOMPLETE", "if not seen.all():", "if False:"))
        mo = mopts(mut, o)
        core = mut.run_picks_core(mo, mut.prepare(mo))
        self.assertTrue((core["stats"]["picks"]["P2"] == -1).any(), "the mutant leaves tokens with no pick")
        self.assertFalse(core["ctl"]["C0_identity"]["pass"], "...and C0 is the only thing that still notices")

    def test_c0_mutant(self):
        nw = self.nav(outcomes=False)
        pk = nw.w.pick.copy()
        pk[4] = (pk[4] + 1) % NP_
        nw.write_table(pick=pk)
        nw.write_bank(pick=pk)
        o = nw.opts()
        self.assertFalse(C.run_picks_core(o, C.prepare(o))["ctl"]["C0_identity"]["pass"])
        mut = load_cmutant(("C0", "bool(c0_rule and", "bool(True or"))
        mo = mopts(mut, o)
        self.assertTrue(mut.run_picks_core(mo, mut.prepare(mo))["ctl"]["C0_identity"]["pass"])

    def test_c1_mutant(self):
        nw = self.nav(outcomes=False, ep_scales=(0.0, 0.0), ttc_scales=(0.0, 0.0))
        o = nw.opts()
        self.assertFalse(C.run_picks_core(o, C.prepare(o))["ctl"]["C1_reach"]["pass"])
        mut = load_cmutant(("C1", "bool(all(", "bool(True or all("))
        mo = mopts(mut, o)
        self.assertTrue(mut.run_picks_core(mo, mut.prepare(mo))["ctl"]["C1_reach"]["pass"])

    def test_c4_mutant_with_a_tolerance_misses_the_perturbation(self):
        nw = self.nav(outcomes=False)
        pr = nw.w.props.copy()
        pr[5, 7, 3, 1] += 1e-3
        nw.write_bank(props=pr)
        o = nw.opts()
        self.assertFalse(C.run_picks_core(o, C.prepare(o))["ctl"]["C4_fan_frozen"]["pass"])
        mut = load_cmutant(("C4", 'stats["c4_max_abs"] == 0.0', 'stats["c4_max_abs"] < 1.0'))
        mo = mopts(mut, o)
        self.assertTrue(mut.run_picks_core(mo, mut.prepare(mo))["ctl"]["C4_fan_frozen"]["pass"])

    def test_control_gate_mutant_writes_seams_through_a_red_control(self):
        nw = self.nav(outcomes=False)
        pr = nw.w.props.copy()
        pr[5, 7, 3, 1] += 1e-3
        nw.write_bank(props=pr)
        o = nw.opts()
        with self.assertRaises(C.ConfirmRefusal):
            C.stage_picks(o)
        self.assertFalse(os.path.exists(os.path.join(o.out_dir, "seams")))
        mut = load_cmutant(("CTLGATE", "if not controls_pass(core[\"ctl\"]):", "if False:"))
        mo = mopts(mut, nw.opts())
        rep = mut.stage_picks(mo)
        self.assertTrue(os.path.exists(os.path.join(mo.out_dir, "seams")), "the mutant is the one that ships seams from a RED instrument")
        self.assertFalse(rep["controls"]["C4_fan_frozen"]["pass"])

    def test_control_gate_mutant_on_read_spends_the_look_through_a_red_control(self):
        nw = self.nav()
        pr = nw.w.props.copy()
        pr[5, 7, 3, 1] += 1e-3
        nw.write_bank(props=pr)
        with self.assertRaises(C.ConfirmRefusal):
            C.stage_read(nw.opts())
        self.assertFalse(os.path.exists(nw.ledger))
        mut = load_cmutant(("CTLGATE-READ", "if not controls_pass(core[\"ctl\"]):", "if False:"))
        res = mut.stage_read(mopts(mut, nw.opts()))
        self.assertTrue(os.path.exists(nw.ledger), "the mutant is the one that spends the look on a broken instrument")
        self.assertFalse(res["controls"]["C4_fan_frozen"]["pass"])

    def test_ledger_mutant_lets_a_second_read_through(self):
        nw = self.nav()
        C.stage_read(nw.opts())
        with self.assertRaises(C.ConfirmRefusal):
            C.stage_read(nw.opts())
        nw2 = self.nav()
        mut = load_cmutant(("LEDGER", "opts.look_ledger, canon(ledger)", 'os.path.join(opts.out_dir, "ledger_elsewhere.json"), canon(ledger)'))
        mut.stage_read(mopts(mut, nw2.opts()))
        self.assertFalse(os.path.exists(nw2.ledger), "the mutant never writes the ledger")
        res = mut.stage_read(mopts(mut, nw2.opts()))
        self.assertTrue(res["look"]["spent"], "...so a second read runs: the look was opened twice")

    def test_both_seeds_mutant(self):
        nw = self.nav(ep_scales=(3.0, 1e-6), clean_subs=True)
        res = C.stage_read(nw.opts())
        self.assertFalse(res["primaries"]["EP-R1"]["both_seeds_clear_all_clauses_except_families6"])
        self.assertTrue(res["readings"]["EPR1_s0"]["bar_clauses"]["clears_without_families6"], "seed 0 alone must clear for the mutant to differ")
        nw2 = self.nav(ep_scales=(3.0, 1e-6), clean_subs=True)
        mut = load_cmutant(("BOTH", "both = all(", "both = any("))
        res2 = mut.stage_read(mopts(mut, nw2.opts()))
        self.assertTrue(res2["primaries"]["EP-R1"]["both_seeds_clear_all_clauses_except_families6"], "the mutant is the one that promotes a one-seed clearing")

    def test_joint_compat_mutant(self):
        nw = self.nav(outcomes=False)
        d = dict(nw.w.ttc_r1)
        d["fraction_pct"] = 25
        wj(os.path.join(nw.w.ttc_dir, "r1_result.json"), d)
        sha = sha_f(os.path.join(nw.w.ttc_dir, "r1_result.json"))
        nw.write_ttc_dev("PASS", result_sha=sha)
        o = nw.opts(ttc_registered_result_sha256=sha)
        self.assertNotIn("JOINT", C.prepare(o)["families"])
        mut = load_cmutant(("COMPAT", "if bad:", "if False:"))
        self.assertIn("JOINT", mut.prepare(mopts(mut, o))["families"])

    def test_forbidden_tokens_mutant_makes_the_real_navtest_names_unreachable(self):
        nw = self.nav(outcomes=False)
        o = nw.opts()
        self.assertIn("refe_navtest", o.table.replace("\\", "/"))
        self.assertIn("table_p2_full", o.table)
        self.assertEqual(C.load_table_inputs(o.guard(), o).n, nw.N)
        with self.assertRaises(D.DevGuardError):
            D.DevGuard(read_files=[o.table]).check_read(o.table)                    # the dev guard WOULD refuse these names: reuse of it would be a built-and-unreachable module
        mut = load_cmutant(("FORBID", '".sets.npz"', '"navtest"'))
        with self.assertRaises(D.DevGuardError):
            mut.load_table_inputs(mopts(mut, o).guard(), mopts(mut, o))

    def test_every_marker_in_the_source_is_mutated_by_a_test_here(self):
        markers = list_markers(_rt(SRC))
        self.assertGreaterEqual(len(markers), 20, markers)
        me = _rt(__file__)
        missing = [m for m in markers if f'"{m}"' not in me]
        self.assertEqual(missing, [], f"markers with no mutant test: {missing}")
        for m in markers:                                                          # and each sits on exactly one line (load_cmutant asserts it; this lists the offenders)
            n = len([ln for ln in _rt(SRC).split("\n") if re.search(rf"MUTATION-TARGET-CONFIRM-{re.escape(m)}(?![A-Z0-9-])", ln)])
            self.assertEqual(n, 1, m)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (f) guard, reachability of every flag, no file-system call outside _io_*
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
FS_ALL = FS_CALLS | {"os.path.getsize", "glob.glob", "os.path.getmtime"}


def fs_violations(src):
    bad = []
    for fn in [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)]:
        if fn.name.startswith("_io_"):
            continue
        for n in ast.walk(fn):
            if isinstance(n, ast.Call):
                name = _dotted(n.func)
                if name in FS_ALL and not (fn.name == "npz_bytes" and name == "np.savez"):          # np.savez into a BytesIO: no file
                    bad.append((fn.name, name))
    return bad


class Guard(CBase):
    def test_guard_refuses_the_fit_bank_and_unlisted_paths_but_allows_the_navtest_names(self):
        g = C.ConfirmGuard(read_dirs=["C:/x"], write_dirs=["C:/x"])
        for p in ("C:/Users/Admin/srttc_s0/work/bank/L1.sets.npz", "C:/x/a.sets.npz", "C:/x/s1_read_cache/a.npz", "C:/x/../x/ranked_fit_logs.tsv", "C:/x/srttc_s0/run/x.json"):
            with self.assertRaises(D.DevGuardError, msg=p):
                g.check_read(p)
        with self.assertRaises(D.DevGuardError):
            g.check_read("C:/elsewhere/file.npy")
        with self.assertRaises(D.DevGuardError):
            g.check_write("C:/elsewhere/out.json")
        g.check_read("C:/x/refe_navtest/table_p2_full.npz")
        g.check_write("C:/x/refe_navtest/seams/s.npz")

    def test_the_real_default_paths_pass_the_guard(self):
        o = C.ConfirmOptions(out_dir="C:/Users/Admin/par18/x/out", bank_dir="D:/Projects/TanitAD/data/refe_navtest/bank", ep_run_dir="C:/Users/Admin/epr1/run/f10",
                             ttc_run_dir="C:/Users/Admin/srttc_r1/run/f10", look_ledger="D:/Projects/TanitAD/data/refe_navtest/NAVTEST_LOOK_SPENT.json", banked_p2_seam=C.NAVTEST_BANKED_SEAM)
        g = o.guard()
        for p in (o.table, o.banked_p2_seam, o.p2_ckpt, o.bank_dir + "/c00_feat.npz", o.ttc_run_dir + "/r1_result.json", o.ep_run_dir + "/epr1_result.json"):
            g.check_read(p)
        g.check_write(o.out_dir + "/seams/seam_P2.npz")
        g.check_write(o.look_ledger)

    def test_no_filesystem_call_outside_an_io_function(self):
        bad = fs_violations(_rt(SRC))
        self.assertEqual(bad, [], f"file-system access outside _io_*: {bad}")

    def test_ast_scan_has_teeth(self):
        base = _rt(SRC)
        self.assertEqual(fs_violations(base), [])
        for plant in ("np.load(p)", "open(p, 'rb')", "os.listdir(p)", "torch.load(p)", "os.path.exists(p)", "np.savez(p)", "os.path.getsize(p)"):
            planted = base + f"\n\ndef sneaky(p):\n    return {plant}\n"
            self.assertEqual(fs_violations(planted), [("sneaky", plant.split("(")[0])], plant)

    def test_the_picks_stage_opens_no_outcome_member_of_the_table(self):
        """Wrap np.load so any access to pdms / sub / valid / sub_names of the table in the picks stage fails loudly."""
        nw = self.nav(outcomes=True)
        real_load = np.load
        touched = []

        class Spy:
            def __init__(self, z):
                self.z = z
                self.files = z.files

            def __getitem__(self, k):
                if k in ("pdms", "sub", "valid", "sub_names", "pdms_at_pick"):
                    touched.append(k)
                return self.z[k]

            def close(self):
                self.z.close()
        def spy_load(*a, **k):
            z = real_load(*a, **k)
            return Spy(z) if hasattr(z, "files") else z
        with mock.patch.object(np, "load", spy_load):
            C.stage_picks(nw.opts())
        self.assertEqual(touched, [])
        touched.clear()
        with mock.patch.object(np, "load", spy_load):
            C.stage_read(nw.opts())
        self.assertIn("pdms", touched, "the spy can see an outcome read when there is one (a probe that cannot read a defect is not a probe)")


class Cli(CBase):
    def test_every_flag_reaches_the_options_the_stage_receives(self):
        got = {}

        def stub(opts):
            got["o"] = opts
            return {}
        argv = ["picks", "--bank-dir", "B", "--out-dir", "O", "--ep-run-dir", "ER", "--ep-registered-result-sha256", "a" * 64, "--ep-dev-result", "EDR", "--ep-abandon", "EAB",
                "--ttc-run-dir", "TR", "--ttc-registered-result-sha256", "b" * 64, "--ttc-dev-result", "TDR", "--ttc-abandon", "TAB", "--table", "TBL", "--p2-ckpt", "CK",
                "--banked-p2-seam", "BS", "--look-ledger", "LL"]
        with mock.patch.object(C, "stage_picks", stub):
            self.assertEqual(C.main(argv), 0)
        o = got["o"]
        want = dict(bank_dir="B", out_dir="O", ep_run_dir="ER", ep_registered_result_sha256="a" * 64, ep_dev_result="EDR", ep_abandon="EAB", ttc_run_dir="TR",
                    ttc_registered_result_sha256="b" * 64, ttc_dev_result="TDR", ttc_abandon="TAB", table="TBL", p2_ckpt="CK", banked_p2_seam="BS", look_ledger="LL")
        for k, v in want.items():
            self.assertEqual(getattr(o, k), v, k)
        got.clear()
        with mock.patch.object(C, "stage_read", stub):
            self.assertEqual(C.main(["read"] + argv[1:]), 0)
        self.assertEqual(got["o"].look_ledger, "LL")

    def test_every_option_field_is_either_a_flag_or_a_registered_pin(self):
        flags = {a.dest for a in C.build_parser()._actions}
        pins = {"table_sha256", "expect_tokens", "expect_logs", "featcheck_tol", "dump_logit_tol", "require_dump", "epr1_sha256"}
        fields = {f.name for f in dataclasses.fields(C.ConfirmOptions)}
        flag_fields = {f.replace("-", "_") for f in flags}
        self.assertEqual(sorted(fields - flag_fields - pins), [], "a ConfirmOptions field that is neither a CLI flag nor a registered pin is unreachable from the caller")

    def test_main_returns_2_on_a_refusal_and_writes_nothing(self):
        nw = self.nav(outcomes=False, ep_dev=None, ttc_dev=None)
        o = nw.opts()
        rc = C.main(["picks", "--bank-dir", nw.bank_dir, "--out-dir", o.out_dir, "--ep-run-dir", nw.w.run_dir, "--ep-registered-result-sha256", nw.w.r1_sha(), "--ep-dev-result", nw.ep_dev_path,
                     "--ttc-run-dir", nw.w.ttc_dir, "--ttc-registered-result-sha256", nw.ttc_sha, "--ttc-dev-result", nw.ttc_dev_path, "--table", nw.table, "--p2-ckpt", nw.w.ckpt,
                     "--banked-p2-seam", nw.seam])
        self.assertEqual(rc, 2)
        self.assertFalse(os.path.exists(o.out_dir))

    def test_manifest_subcommand_hashes_every_npz_and_the_applier_accepts_it(self):
        nw = self.nav(outcomes=False)
        os.remove(os.path.join(nw.bank_dir, C.BANK_MANIFEST))
        self.assertEqual(C.main(["manifest", "--bank-dir", nw.bank_dir]), 0)
        man = json.loads(_rt(os.path.join(nw.bank_dir, C.BANK_MANIFEST)))
        self.assertEqual(man["n_files"], 2 * len(nw.chunks))
        for fn, ent in man["files"].items():
            self.assertEqual(ent["sha256"], sha_f(os.path.join(nw.bank_dir, fn)))
        rep, pk, o = self.picks(nw)
        self.assertEqual(rep["inputs"]["n_bank_chunks"], len(nw.chunks))

    def test_default_pins_match_the_landed_p2_full_table_manifest(self):
        self.assertEqual(C.NAVTEST_TABLE_SHA256, "33441082b7044c951e3cef5fb898cac50797039aed93560b2712e3ff4c34906e")
        self.assertEqual((C.N_TOKENS, C.N_LOGS), (12146, 136))
        self.assertEqual(sha_f(os.path.join(HERE, "epr1.py")), C.D.EPR1_SHA256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
