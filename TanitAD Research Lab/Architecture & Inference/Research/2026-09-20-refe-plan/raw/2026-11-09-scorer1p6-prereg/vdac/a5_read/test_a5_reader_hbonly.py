#!/usr/bin/env python3
"""Synthetic-fixture tests for the A5 read set-up (no real data, CPU).  They do NOT test or edit the frozen reader; they show how it behaves on the A5 manifests' arm tables.
Needs: S1_READ_V1 = the frozen s1_read.py; EVAL_DIR = the folder holding s1_read_v2.py and test_s1_read_v2.py (its fixture builder).  Run: python test_a5_reader_hbonly.py"""
import json, os, re, sys, tempfile, unittest
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.environ.get("EVAL_DIR") or os.path.join(HERE, "..", "..", "..", "..", "eval")
sys.path.insert(0, EVAL); sys.path.insert(0, HERE)
import s1_read_v2 as R            # noqa: E402
import test_s1_read_v2 as F       # noqa: E402  (fixture builder)
import a5_hc_s1f_subset as W      # noqa: E402
import fill_manifest_sha as FS    # noqa: E402
import make_manifests_a5 as MM    # noqa: E402
NB = 300


def arms_into(d, spec, perfect, shift_names=()):
    N = len(d["frame_log"]); sdf2 = np.zeros((N, 40, 32), np.float32)
    for a in spec:
        one = abs(a["cell"] - 1.0) < 1e-9
        if a["role"] == "trained":
            lg = d["arms"]["s0"]["logit"].copy() if a["name"] == "HA-s0" else perfect.copy()
        else:
            lg = np.full((N, 40, 32), -3.0, np.float32)
        if one:
            lg = np.repeat(np.repeat(lg, 2, axis=1), 2, axis=2)
        d["arms"][a["name"]] = {"logit": lg.astype(np.float32), "sdf": np.zeros((N, 80, 64) if one else (N, 40, 32), np.float32)}


class A5Tables(unittest.TestCase):
    def table(self, which):
        man = MM.build(which)
        spec, pairs = R.arms_from_manifest(man)
        for a in spec:
            for fk in a["files"]:
                self.assertIn(fk, man["files"], (which, a["name"], fk))
        return man, spec, pairs

    def test_hb_only_manifest_reads_with_its_pair_and_no_hc(self):
        man, spec, pairs = self.table("hb")
        self.assertEqual(pairs, [{"x": "HB-s0", "ref": "HA-s0"}])
        self.assertFalse(any(a["name"].startswith("HC") for a in spec))
        d = F.fixture(bos_shift=True); P = d.pop("_perfect"); arms_into(d, spec, P)
        r = R.run_read(d, n_boot=NB, arms=spec, pairs=pairs)
        self.assertIn("HB-s0", r["per_arm"]); self.assertEqual(len(r["paired"]), 1)
        self.assertEqual(r["per_arm"]["HB-s0"]["verdict"], "S1-GO")        # own SHUF + own initial head + perfect fixture head
        self.assertGreater(r["paired"][0]["p1"]["lower95"], 0)             # HB (perfect) minus HA-s0 (BOS shifted): P1 delta separated (the fixture's Q delta is ~0, so x_lift is not asserted)
        self.assertIsNotNone(r["per_arm"]["HB-s0"]["S1_F"])
        self.assertEqual(man["files"]["hb_init_b"]["sha256"], man["files"]["head_init0_b"]["sha256"])   # HB init_b is HA's file

    def test_pair_naming_an_arm_that_is_not_in_the_table_is_refused_so_no_hc_pair_in_the_hb_manifest(self):
        _, spec, _ = self.table("hb")
        d = F.fixture(); P = d.pop("_perfect"); arms_into(d, spec, P)
        with self.assertRaises(ValueError):
            R.run_read(d, n_boot=NB, arms=spec, pairs=[{"x": "HC-s0", "ref": "HA-s0"}])

    def test_hc_manifest_without_init_arm_is_read_but_s1f_is_missing_and_verdict_incomplete(self):
        _, spec, pairs = self.table("hc")
        self.assertEqual(pairs, [{"x": "HC-s0", "ref": "HA-s0"}])
        self.assertIsNone(next(a for a in spec if a["name"] == "HC-s0").get("init"))
        d = F.fixture(bos_shift=True); P = d.pop("_perfect"); arms_into(d, spec, P)
        r = R.run_read(d, n_boot=NB, arms=spec, pairs=pairs)
        self.assertIsNone(r["per_arm"]["HC-s0"]["S1_F"]); self.assertEqual(r["per_arm"]["HC-s0"]["verdict"], "INCOMPLETE")
        self.assertTrue(r["per_arm"]["HC-s0"]["verdict_missing"])

    def test_load_real_refuses_a_partial_init_arm_source_level(self):
        src = open(os.path.join(EVAL, "s1_read_v2.py"), encoding="utf-8").read()
        m = re.search(r"if not got\.all\(\):\s*\n\s*raise SystemExit\(f\"REFUSED: arm \{a\['name'\]\} does not cover the export's", src)
        self.assertIsNotNone(m)      # an init arm with only the 1,705 held-out frames is a REFUSAL in the frozen loader (UNVERIFIED claim of PREREG 7.5 -> answered: it does NOT accept it)

    def test_arm_rows_match_prereg_7_5(self):
        man = MM.build("both")
        by = {a["name"]: a for a in man["arms"]}
        self.assertEqual((by["HB-s0"]["cell"], by["HC-s0"]["cell"]), (2.0, 1.0))
        self.assertEqual(by["HB-s0"]["init"], "HB-s0-initial"); self.assertNotIn("init", by["HC-s0"])
        self.assertEqual(by["HB-s0-initial"]["files"], ["hb_init_a", "hb_init_b"])
        self.assertEqual(man["pairs"], [{"x": "HB-s0", "ref": "HA-s0"}, {"x": "HC-s0", "ref": "HA-s0"}])
        self.assertNotIn("hc_init_b", man["files"])
        self.assertTrue(all(man["files"][k]["sha256"] == "PENDING" for k in man["files"] if k.startswith(("hb_s0", "hb_shuf", "hb_init_a", "hc_"))))


class SubsetS1F(unittest.TestCase):
    def test_s1f_on_subset_trained_perfect_vs_constant_init_holds_and_uses_only_the_subset_logs(self):
        d = F.fixture(bos_shift=True); P = d.pop("_perfect"); N = len(P)
        d["arms"]["t"] = {"logit": np.repeat(np.repeat(P, 2, 1), 2, 2), "sdf": np.zeros((N, 80, 64), np.float32)}
        d["arms"]["i"] = {"logit": np.full((N, 80, 64), -3.0, np.float32), "sdf": np.zeros((N, 80, 64), np.float32)}
        logs = d["frame_log"].astype(str); ul = sorted(set(logs))
        pick = [l for c in F.CITIES for l in [x for x in ul if d["city_of_log"][x] == c][:2]]       # 2 logs per city = 8
        keep = np.isin(logs, pick)
        r = W.s1f_on_subset(d, keep, "t", "i", 1.0, n_boot=NB)
        self.assertEqual(r["n_groups"], 8); self.assertEqual(r["n_frames"], int(keep.sum()))
        self.assertTrue(r["S1_F"]["hold"]); self.assertTrue(r["p1_defined"])
        # a subset that misses a city: the city-reweighted P1 is undefined (NaN), flagged, not a fail
        keep2 = np.isin(logs, ul[:6]); r2 = W.s1f_on_subset(d, keep2, "t", "i", 1.0, n_boot=NB)
        self.assertFalse(r2["p1_defined"]); self.assertTrue(r2["cities_absent"]); self.assertFalse(r2["S1_F"]["hold"])
        # an init equal to the trained head is not S1-F (delta exactly 0)
        d["arms"]["i"] = {"logit": d["arms"]["t"]["logit"].copy(), "sdf": d["arms"]["t"]["sdf"]}
        self.assertFalse(W.s1f_on_subset(d, keep, "t", "i", 1.0, n_boot=NB)["S1_F"]["hold"])


class FillSha(unittest.TestCase):
    def test_fill_accepts_matching_and_refuses_mismatch_and_missing(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "x.npz"); open(p, "wb").close(); open(p, "ab").write(b"abc")
            h = FS.sha(p)
            man = {"files": {"k": {"path": p, "sha256": "PENDING"}, "fixed": {"path": p, "sha256": "f" * 64}}}
            self.assertEqual(FS.fill(man, {"k": h})["files"]["k"]["sha256"], h)
            self.assertEqual(man["files"]["k"]["sha256"], "PENDING")      # input untouched
            with self.assertRaises(SystemExit): FS.fill(man, {"k": "0" * 64})
            with self.assertRaises(SystemExit): FS.fill(man, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
