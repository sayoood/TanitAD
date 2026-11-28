"""Refusal tests for tf_n1_a1_chain.py (run with CUDA_VISIBLE_DEVICES=-1; stdlib only, `python -m pytest` or `python -m unittest`).
Every test builds a tiny fake run tree, flips ONE thing, and requires the chain's guards to REFUSE (rc != 0, 'REFUSING' on stderr) -- and the
unmodified fixture to PASS, so a refusal cannot be an artefact of a broken fixture (the control that must read PASS)."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CHAIN = os.path.join(HERE, "tf_n1_a1_chain.py")
K = 42
HARNESS_FILES = ["w/pk/eval/harness/HARNESS_PINS.json", "w/pk/eval/harness/score_e.py"]
A1_FILES = HARNESS_FILES + ["w/pk/eval/score_p4_a1.py", "n1/a1/tf_n1_analyze_a1.py", "n1/a1/tf_n1_a1_chain.py",
                            "build/out/refe_tfn1_tc20s.npz", "build/out/refe_tfn1_tw.npz", "build/out/refe_tfn1_m.npz",
                            "inputs/refe_p4_teacher_navtest.npz"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


class Fx:
    def __init__(self, root):
        self.run = os.path.join(root, "run")
        for f in A1_FILES + ["n1/tf_n1_chain.py"]:
            w(os.path.join(self.run, f), f"content of {f}\n")
        for x in ("N1C", "N1T", "N1W", "CERT", "human"):
            w(os.path.join(self.run, "fam", f"fam_{x}.json"), "{}")
        self.write_pins()
        self.amendment = os.path.join(root, "PREREG_TF_N1_A1.md")
        w(self.amendment, "the amendment\n")
        self.sha = sha(self.amendment)
        self.ledger = os.path.join(root, "ledger_K.json")
        self.write_ledger()

    def write_pins(self):
        w(os.path.join(self.run, "n1", "N1_PINS.json"),
          json.dumps({"files": {"n1/tf_n1_chain.py": sha(os.path.join(self.run, "n1/tf_n1_chain.py"))}}))
        w(os.path.join(self.run, "n1", "a1", "A1_PINS.json"),
          json.dumps({"files": {f: sha(os.path.join(self.run, f)) for f in A1_FILES}}))
        self.pins_sha = sha(os.path.join(self.run, "n1", "a1", "A1_PINS.json"))

    def write_ledger(self, mutate=None):
        reads = [{"key": "X:1:E3", "n": 40}, {"key": "TF-N1:aaaa:E3", "n": 41},
                 {"key": f"TF-N1-A1:{self.sha}:E3", "n": K}]
        if mutate:
            reads = mutate(reads)
        w(self.ledger, json.dumps({"reads": reads}))

    def run_chain(self, **over):
        a = dict(run_dir=self.run, amendment=self.amendment, amendment_sha256=self.sha, a1_pins_sha256=self.pins_sha,
                 ledger=self.ledger, procs="2")
        a.update(over)
        argv = [sys.executable, CHAIN, "--check-only"] + [x for k, v in a.items() for x in (f"--{k.replace('_', '-')}", str(v))]
        return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL,
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))


class ChainRefusals(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tfn1a1_", dir=os.environ.get("TFN1_A1_TMP"))
        self.fx = Fx(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def refused(self, r, needle):
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("REFUSING", r.stderr + r.stdout)
        self.assertIn(needle, r.stderr + r.stdout)

    def test_control_good_fixture_passes(self):
        r = self.fx.run_chain()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ZZN1A1 GUARDS PASS", r.stdout)

    def test_wrong_amendment_sha(self):
        self.refused(self.fx.run_chain(amendment_sha256="0" * 64), "amendment-sha256")

    def test_edited_amendment(self):
        with open(self.fx.amendment, "a", encoding="utf-8") as f:
            f.write("one more line after the hash\n")
        self.refused(self.fx.run_chain(), "amendment-sha256")

    def test_ledger_entry_missing(self):
        self.fx.write_ledger(lambda r: [x for x in r if not x["key"].startswith("TF-N1-A1")])
        self.refused(self.fx.run_chain(), "no entry")

    def test_ledger_entry_for_a_different_sha(self):
        self.fx.write_ledger(lambda r: [dict(x, key=x["key"].replace(self.fx.sha, "f" * 64)) for x in r])
        self.refused(self.fx.run_chain(), "no entry")

    def test_ledger_n_differs_from_k(self):
        self.fx.write_ledger(lambda r: [dict(x, n=43) if x["key"].startswith("TF-N1-A1") else x for x in r])
        self.refused(self.fx.run_chain(), "n = 43")

    def test_a_later_read_landed(self):
        self.fx.write_ledger(lambda r: r + [{"key": "OTHER:1:E3", "n": 43}])
        self.refused(self.fx.run_chain(), "another read landed")

    def test_void_entry_missing(self):
        self.fx.write_ledger(lambda r: [x for x in r if not x["key"].startswith("TF-N1:")])
        self.refused(self.fx.run_chain(), "void TF-N1 entry")

    def test_procs_over_two(self):
        self.refused(self.fx.run_chain(procs=3), "at most 2 processes")

    def test_n1_pin_mismatch(self):
        w(os.path.join(self.fx.run, "n1/tf_n1_chain.py"), "edited\n")
        self.refused(self.fx.run_chain(), "N1 pinned files differ")

    def test_a1_pins_sha_mismatch(self):
        self.refused(self.fx.run_chain(a1_pins_sha256="1" * 64), "A1_PINS.json missing or its sha256")

    def test_a1_pinned_file_edited_after_pinning(self):
        w(os.path.join(self.fx.run, "w/pk/eval/score_p4_a1.py"), "edited after the pins\n")
        self.refused(self.fx.run_chain(), "A1 pinned files differ")

    def test_harness_file_edited(self):
        w(os.path.join(self.fx.run, "w/pk/eval/harness/score_e.py"), "edited harness\n")
        self.refused(self.fx.run_chain(), "A1 pinned files differ")

    def test_unpinned_file_in_harness_tree(self):
        w(os.path.join(self.fx.run, "w/pk/eval/harness/w3e/code/sneaky.py"), "x = 1\n")
        self.refused(self.fx.run_chain(), "unpinned files under the harness tree")

    def test_a_plan_seam_swapped(self):
        w(os.path.join(self.fx.run, "build/out/refe_tfn1_tc20s.npz"), "another candidate\n")
        self.refused(self.fx.run_chain(), "A1 pinned files differ")

    def test_pins_file_missing_a_required_entry(self):
        p = os.path.join(self.fx.run, "n1", "a1", "A1_PINS.json")
        d = json.load(open(p, encoding="utf-8"))
        del d["files"]["build/out/refe_tfn1_m.npz"]
        w(p, json.dumps(d))
        self.refused(self.fx.run_chain(a1_pins_sha256=sha(p)), "does not pin build/out/refe_tfn1_m.npz")

    def test_family_file_missing(self):
        os.remove(os.path.join(self.fx.run, "fam", "fam_N1C.json"))
        self.refused(self.fx.run_chain(), "fam_N1C.json")


class EarlyCidh(unittest.TestCase):
    """the early C-ID-H gate: a 1e-6 breach on one token of one metric, or one missing token, must FAIL it"""

    def _csvs(self, tmp, n=12146, bump=None, drop=False):
        head = ",token,valid,no_at_fault_collisions,drivable_area_compliance,driving_direction_compliance,ego_progress," \
               "time_to_collision_within_bound,comfort,score\n"
        a, b = [head], [head]
        for i in range(n):
            row = f"{i},tok{i},True,1.0,1.0,1.0,0.5,1.0,1.0,0.8\n"
            b.append(row)
            if drop and i == 0:
                continue
            a.append(row.replace(",0.5,", f",{0.5 + bump},") if (bump and i == 7) else row)
        pa, pb = os.path.join(tmp, "a.csv"), os.path.join(tmp, "b.csv")
        w(pa, "".join(a))
        w(pb, "".join(b))
        return pa, pb

    def setUp(self):
        sys.path.insert(0, HERE)
        import tf_n1_a1_chain as C
        self.C = C
        self.tmp = tempfile.mkdtemp(prefix="tfn1a1c_", dir=os.environ.get("TFN1_A1_TMP"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_identical_passes(self):
        self.assertTrue(self.C.early_cidh(*self._csvs(self.tmp))["pass"])

    def test_breach_just_over_tolerance_fails(self):
        r = self.C.early_cidh(*self._csvs(self.tmp, bump=2e-6))
        self.assertFalse(r["pass"])
        self.assertEqual(r["tokens_over_tol"], 1)

    def test_breach_just_under_tolerance_passes(self):
        self.assertTrue(self.C.early_cidh(*self._csvs(self.tmp, bump=5e-7))["pass"])

    def test_missing_token_fails(self):
        self.assertFalse(self.C.early_cidh(*self._csvs(self.tmp, drop=True))["pass"])

    def test_short_set_fails_even_if_equal(self):
        self.assertFalse(self.C.early_cidh(*self._csvs(self.tmp, n=100))["pass"])


if __name__ == "__main__":
    unittest.main()
