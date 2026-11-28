"""Tests of hierrep_read.py on SYNTHETIC dumps only (no real arm output exists).  Run on Windows:
  set CUDA_VISIBLE_DEVICES=-1 & set PYTHONDONTWRITEBYTECODE=1 & python test_hierrep_read.py
(taniteval is imported from TANITEVAL_ROOT, default the pinned tree used by HIERWHERE; PYTHONPATH entries are joined with ';' on Windows.)"""
import json
import os
import shutil
import sys
import tempfile
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
os.environ.setdefault("TANITEVAL_ROOT", "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/taniteval")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import hierrep_read as H  # noqa: E402

N_EP, N_PER = 12, 20
NB = 200


def make_g(rng):
    n = N_EP * N_PER
    v0 = rng.uniform(0, 12, n)
    a = rng.uniform(-3, 2, n)
    w = rng.choice([0.0, 0.0, 0.0, 0.15, -0.15], n)
    t = (np.arange(30) + 1) * 0.2
    v = np.maximum(0.0, v0[:, None] + a[:, None] * t[None, :])
    hd = np.cumsum(np.broadcast_to(w[:, None] * 0.2, (n, 30)), axis=1)
    p = np.cumsum(np.stack([v * 0.2 * np.cos(hd), v * 0.2 * np.sin(hd)], -1), axis=1)
    eid = np.repeat(np.arange(N_EP), N_PER)
    return p, eid, v0


def write_world(root, plan, seeds=range(6), rng_seed=0, steps_ok=True):
    """plan(arm, crit) -> error magnitude scale array.  Returns crit mask."""
    g, eid, v0 = make_g(np.random.default_rng(rng_seed))
    crit = H.classify(*H.kinematics(g))
    crit = (crit[0] * 3 + crit[1]) != H.KEEP_STRAIGHT
    for arm in H.ARMS:
        for s in seeds:
            r = np.random.default_rng(10000 + 100 * s + H.ARMS.index(arm))
            mag = r.uniform(0.5, 1.5, len(g)) * plan(arm, crit)
            phi = r.uniform(0, 2 * np.pi, len(g))
            cl = g + mag[:, None, None] * np.stack([np.cos(phi), np.sin(phi)], -1)[:, None, :]
            d = f"{root}/t1/{arm}_s{s}/dump_{arm}_s{s}_m10_causal_heads"
            os.makedirs(d, exist_ok=True)
            for e in range(N_EP):
                m = eid == e
                np.savez(f"{d}/ep{e:04d}.npz", g=g[m], cl=cl[m], ws=np.arange(m.sum()), eid=eid[m], clip_index=eid[m], v0=v0[m])
            rd = f"{root}/runs/{arm}_s{s}"
            os.makedirs(rd, exist_ok=True)
            with open(rd + "/summary.json", "w", encoding="utf-8") as f:
                json.dump({"done": True, "steps": H.EXPECT_STEPS[arm] if steps_ok else 5, "levers": {"n_params_trainable": H.EXPECT_PARAMS}}, f)
    return crit


def run_read(root, seeds="0,1,2,3,4,5"):
    out = root + "/out"
    rc = H.main(["read", "--t1-root", root + "/t1", "--runs-root", root + "/runs", "--out-dir", out, "--n-boot", str(NB), "--seeds", seeds,
                 "--no-card-counts", "--expect-files", str(N_EP)])
    with open(out + "/RESULT_HIERREP.json", encoding="utf-8") as f:
        return rc, json.load(f), out


HIERS = ("HIER", "DAG")


class T(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="hierrep_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_planted_effect_reads_hier(self):
        write_world(self.root, lambda arm, crit: np.where(crit & (arm in HIERS), 0.25, 1.0))
        rc, res, out = run_read(self.root)
        c2 = res["stages"]["stage2"]
        self.assertEqual(c2["ade6"]["critical"]["label"], "HIER")
        self.assertTrue(all(x < 0 for x in c2["ade6"]["critical"]["d_s"]))
        self.assertEqual(c2["ade6"]["bulk"]["label"], "OPEN")
        self.assertIn(res["PRIMARY_OUTCOME"], ("HIER-DOMINATES", "HIER-ON-CRITICAL"))
        self.assertEqual(res["stages"]["stage1"]["ade6"]["critical"]["label"], "HIER")
        self.assertTrue(res["controls"]["r4_label_swap"]["PASS"], res["controls"]["r4_label_swap"])
        self.assertEqual(rc, 0)
        txt = open(out + "/VERDICT_HIERREP.md", encoding="utf-8").read()
        self.assertIn("STRATEGIC: PAUSED", txt)
        self.assertIn("T1-kin", txt)
        self.assertNotIn("closed" + " loop", txt)

    def test_null_reads_open(self):
        write_world(self.root, lambda arm, crit: np.ones(len(crit)))
        rc, res, _ = run_read(self.root)
        self.assertEqual(res["stages"]["stage2"]["ade6"]["critical"]["label"], "OPEN")
        self.assertEqual(res["PRIMARY_OUTCOME"], "NOT-ESTABLISHED")
        self.assertEqual(res["stages"]["stage1"]["outcome"], "NOT-ESTABLISHED")

    def test_mirrored_effect_reads_flat(self):
        write_world(self.root, lambda arm, crit: np.where(crit & (arm not in HIERS), 0.25, 1.0))
        rc, res, _ = run_read(self.root)
        self.assertEqual(res["stages"]["stage2"]["ade6"]["critical"]["label"], "FLAT")
        self.assertEqual(res["PRIMARY_OUTCOME"], "FLAT-AHEAD")

    def test_incomplete_seed_excluded_and_nothing_passes(self):
        write_world(self.root, lambda arm, crit: np.where(crit & (arm in HIERS), 0.25, 1.0))
        os.remove(f"{self.root}/t1/DAG_s3/dump_DAG_s3_m10_causal_heads/ep0005.npz")        # a dump cut short
        with open(f"{self.root}/runs/FLAT_s4/summary.json", "w", encoding="utf-8") as f:    # a run that stopped before its last step
            json.dump({"done": False, "steps": 100}, f)
        rc, res, _ = run_read(self.root)
        self.assertEqual(res["seeds_read"], [0, 1, 2, 5])
        self.assertEqual(sorted(res["seeds_excluded"]), [3, 4])
        c = res["stages"]["stage2"]["ade6"]["critical"]
        self.assertEqual(c["n_seeds"], 4)
        self.assertEqual(c["label"], "OPEN")
        self.assertIn("INCOMPLETE", c["flags"])
        self.assertEqual(res["PRIMARY_OUTCOME"], "NOT-ESTABLISHED")

    def test_param_mismatch_fails_control(self):
        write_world(self.root, lambda arm, crit: np.ones(len(crit)))
        with open(f"{self.root}/runs/FLAT_s0/summary.json", "w", encoding="utf-8") as f:
            json.dump({"done": True, "steps": 19164, "levers": {"n_params_trainable": 7000000}}, f)
        rc, res, out = run_read(self.root)
        self.assertEqual(rc, 2)
        self.assertFalse(res["controls"]["r3_param_count"]["PASS"])
        self.assertIn("READ NOT VALID", open(out + "/VERDICT_HIERREP.md", encoding="utf-8").read())

    def test_swap_control_on_a_planted_effect_is_mostly_open(self):
        g, eid, _ = make_g(np.random.default_rng(0))
        uniq, inv = H.episode_units(eid)
        rng = np.random.default_rng(5)
        sE = rng.normal(-1.0, 0.05, (6, N_EP)) * 10.0
        cE = np.full((6, N_EP), 10.0)
        self.assertEqual(H.read_cell(sE, cE, NB)["label"], "HIER")
        r4 = H.r4_swap_control(sE, cE, NB)
        self.assertTrue(r4["PASS"])
        self.assertEqual(len(r4["draws"]), 20)
        # a swap that happens to flip nothing must read the planted label (the control is not vacuous): all-False swap = original
        self.assertEqual(H.label_of(sE.sum(1) / cE.sum(1), -2.0, -0.5), ("HIER", []))
        self.assertEqual(H.label_of(-sE.sum(1) / cE.sum(1), 0.5, 2.0), ("FLAT", []))

    def test_signflip_exact_and_t(self):
        p, n = H.sign_flip_p([-1, -2, -3, -4, -5, -6])
        self.assertEqual(n, 64)
        self.assertAlmostEqual(p, 2 / 64)
        p2, _ = H.sign_flip_p([-1, 2, -3, 4, -5, 6])
        self.assertGreater(p2, 0.5)
        lo, hi = H.t_interval([1, 2, 3, 4, 5, 6])
        self.assertAlmostEqual((lo + hi) / 2, 3.5)
        self.assertEqual(H.outcome("HIER", "OPEN"), "HIER-ON-CRITICAL")
        self.assertEqual(H.outcome("OPEN", "HIER"), "NOT-ESTABLISHED")

    def test_two_level_bootstrap_has_seed_variance(self):
        """seeds with opposite-sign differences and tiny episode noise: an episode-only interval would exclude 0, the two-level one must not."""
        E = 12
        d = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        sE = np.repeat(d[:, None], E, 1) * 5.0
        cE = np.full((6, E), 5.0)
        lo, hi, _ = H.boot_two_level(sE, cE, 500)
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_r1(self):
        write_world(self.root, lambda arm, crit: np.ones(len(crit)))
        d = f"{self.root}/t1/HIER_s0/dump_HIER_s0_m10_causal_heads"
        args = ["r1", "--dump", d, "--expect-files", str(N_EP)]
        self.assertEqual(H.main(args + ["--ref", self.root + "/absent.json"]), 4)
        D = H.load_dump(d, N_EP)
        ade = float(np.nanmean(H.comps(D["cl"], D["g"])["ade6_m"]))
        ref = self.root + "/ref.json"
        json.dump({"ade6_all_windows_m": ade + 0.002, "n_windows": len(D["g"])}, open(ref, "w"))
        self.assertEqual(H.main(args + ["--ref", ref]), 0)
        json.dump({"ade6_all_windows_m": ade + 0.02, "n_windows": len(D["g"])}, open(ref, "w"))
        self.assertEqual(H.main(args + ["--ref", ref]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
