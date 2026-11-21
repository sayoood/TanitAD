"""Tests for p2tab_driver.same_pick_proof (known-answer world + DELIBERATE-REGRESSION arms that must make the proof FAIL) and the SEL/PROP rule on a world
with a known answer.  Run:  C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I test_p2tab.py   (CPU, no model, no harness, a few seconds)."""
import csv
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p2tab_driver as D  # noqa: E402
import c3_run_p2 as C      # noqa: E402
import zero_tail as ZT     # noqa: E402

SUBN = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance"]
N, K = 5, 64


def world(tmp, *, pose_shift=0.0, pdms_shift=0.0, nonzero_at_pick=False, dump_pick_shift=0):
    rng = np.random.default_rng(7)
    toks = [f"t{i:02d}" for i in range(N)]
    props = rng.normal(size=(N, K, 8, 3)).astype(np.float32)
    pick = np.arange(N) * 3 % K
    logits = rng.normal(size=(N, K, 7)).astype(np.float32)
    pdms = rng.uniform(0.0, 1.0, size=(N, K))
    sub = rng.uniform(0.0, 1.0, size=(N, K, 6))
    pdms[np.arange(N), pick] = 0.0                      # P2's zero tokens: the pick scores exactly 0
    sub[np.arange(N), pick, 1] = 0.0                    # DAC = 0 at the pick
    banked_pose = props[np.arange(N), pick].copy()
    csv_sub = sub[np.arange(N), pick].copy()
    props_dump = props.copy()
    props_dump[np.arange(N), pick, 0, 0] += pose_shift
    pdms_t = pdms.copy()
    pdms_t[np.arange(N), pick] += pdms_shift
    if nonzero_at_pick:
        pdms_t[0, pick[0]] = 0.3
    dpick = (pick + dump_pick_shift) % K                  # a dump whose planner picked a DIFFERENT proposal than the banked P2 run
    np.savez(os.path.join(tmp, "proposals.npz"), token=np.array(toks), proposals=props_dump, logits=logits, pick=dpick)
    np.savez(os.path.join(tmp, "table.npz"), token=np.array(toks), pdms=pdms_t, sub=sub, valid=np.ones((N, K), bool), pick=dpick, sub_names=np.array(SUBN))
    np.savez(os.path.join(tmp, "banked_seam.npz"), token=np.array(toks), poses=banked_pose)
    with open(os.path.join(tmp, "banked.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + ["token", "valid"] + SUBN + ["score"])
        for i, t in enumerate(toks):
            w.writerow([i, t, True] + [float(x) for x in csv_sub[i]] + [0.0])
    C.BANKED_SEAM, C.BANKED_CSV = os.path.join(tmp, "banked_seam.npz"), os.path.join(tmp, "banked.csv")
    return toks, pick


class SamePickProof(unittest.TestCase):
    def run_proof(self, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            world(tmp, **kw)
            return D.same_pick_proof(tmp, os.path.join(tmp, "proof.json"))

    def test_known_answer_world_passes_and_counts_bit_identical(self):
        R = self.run_proof()
        self.assertTrue(R["PASS"], R)
        self.assertEqual(R["pose"]["n_bit_identical"], N)
        self.assertEqual(R["pdms_at_pick_vs_banked_csv"]["n_exact"], N)
        self.assertEqual(R["pdms_at_pick_vs_banked_csv"]["n_table_pdms_at_pick_exactly_0"], N)

    def test_REGRESSION_pose_off_by_2mm_fails(self):
        self.assertFalse(self.run_proof(pose_shift=2e-3)["PASS"])

    def test_pose_off_by_half_mm_is_within_the_registered_bar(self):          # the 1e-3 m bar is the C3 fidelity bar, not tighter
        R = self.run_proof(pose_shift=5e-4)
        self.assertTrue(R["PASS"])
        self.assertEqual(R["pose"]["n_bit_identical"], 0)          # the shift is applied to every token's pick

    def test_REGRESSION_pdms_at_pick_off_by_1e3_fails(self):
        self.assertFalse(self.run_proof(pdms_shift=1e-3)["PASS"])

    def test_pdms_off_by_rounding_quantum_would_still_fail_the_zero_status(self):  # a zero token must read EXACTLY 0.0 at the pick
        R = self.run_proof(pdms_shift=4e-5)
        self.assertFalse(R["PASS"])

    def test_REGRESSION_a_nonzero_pick_in_the_table_fails(self):
        self.assertFalse(self.run_proof(nonzero_at_pick=True)["PASS"])

    def test_REGRESSION_a_different_pick_than_the_banked_P2_run_fails(self):
        R = self.run_proof(dump_pick_shift=1)
        self.assertFalse(R["PASS"])
        self.assertGreater(R["pose"]["n_gt_tol"], 0)


class Aggregate(unittest.TestCase):
    def test_navsim_v1_aggregate_known_answer(self):
        lg = np.full((2, 3, 7), -20.0)                      # sigmoid ~ 0 everywhere
        lg[0, 1, :5] = 20.0                                 # proposal 1: all five terms ~1
        lg[1, 2, :5] = 20.0
        lg[1, 2, 1] = -20.0                                 # but DAC ~ 0 -> the multiplicative gate kills it
        lg[1, 0, :5] = 20.0
        lg[1, 0, 2] = 0.0                                   # proposal 0: EP = 0.5
        self.assertEqual(D.pick_from_logits(lg).tolist(), [1, 0])


class SelProp(unittest.TestCase):
    def test_sel_prop_counts_and_swap_regression(self):
        pdms = np.zeros((4, 64))
        pdms[1, 5] = 0.7                                    # token 1: SEL (best 0.7); tokens 0, 2, 3: PROP
        pdms[2, 9] = 0.0
        sub = np.ones((4, 64, 6))
        fs = ZT.fan_stats(pdms, sub[..., 0], sub[..., 1], sub[..., 3])
        zero = np.ones(4, bool)
        boot = ZT.Boot(np.array([0, 0, 1, 1]), 2, B=200, seed=1)
        q = ZT.q1_block(zero, fs["sel"], fs["best"], boot, 4)
        self.assertEqual((q["n_SEL"], q["n_PROP"]), (1, 3))
        self.assertAlmostEqual(q["oracle_on_SEL_gain_pdms_pts"]["point"], 0.7 / 4 * 100)
        sw = ZT.q1_block(zero, ~fs["sel"], fs["best"], boot, 4)           # DELIBERATE REGRESSION: swapped rule must move the counts
        self.assertNotEqual((sw["n_SEL"], sw["n_PROP"]), (q["n_SEL"], q["n_PROP"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
