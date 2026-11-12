"""C-THR tests (PREREG_TF_N2_A1): the chain rolls at 2 threads and refuses a roll whose shard logs are not at OMP_NUM_THREADS=2."""
import importlib.util, os, sys, tempfile, types, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("chain_a1", os.path.join(HERE, "tfn2_nt_chain.py")); C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)


class FakeR:
    def __init__(self, root, procs=2):
        self.root, self.a = root, types.SimpleNamespace(procs=procs)
    def p(self, *x):
        return os.path.join(self.root, *x)


def write_logs(root, heads):
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    for i, h in enumerate(heads):
        open(os.path.join(root, "logs", f"p4_R0_s{i}.log"), "w", encoding="utf-8", newline=chr(10)).write(h + chr(10) + "# cmd" + chr(10))


class CThr(unittest.TestCase):
    def test_constant_and_argv(self):
        self.assertEqual(C.ROLL_THREADS, 2)
        src = open(os.path.join(HERE, "tfn2_nt_chain.py"), encoding="utf-8").read()
        self.assertIn('"--threads", str(ROLL_THREADS)', src)
        self.assertNotIn('"--threads", "1"', src)

    def test_pass_at_2(self):
        with tempfile.TemporaryDirectory() as d:
            write_logs(d, ["# REFE_SIM_HZ=10 OMP_NUM_THREADS=2"] * 2)
            self.assertTrue(C.cthr(FakeR(d), "R0")["PASS"])

    def test_fail_at_1_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            write_logs(d, ["# REFE_SIM_HZ=10 OMP_NUM_THREADS=2", "# REFE_SIM_HZ=10 OMP_NUM_THREADS=1"])
            self.assertFalse(C.cthr(FakeR(d), "R0")["PASS"])

    def test_fail_missing_shard(self):
        with tempfile.TemporaryDirectory() as d:
            write_logs(d, ["# REFE_SIM_HZ=10 OMP_NUM_THREADS=2"])
            self.assertFalse(C.cthr(FakeR(d), "R0")["PASS"])

    def test_fail_no_logs(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(C.cthr(FakeR(d), "R0")["PASS"])

    def test_fail_substring_trap(self):
        with tempfile.TemporaryDirectory() as d:
            write_logs(d, ["# REFE_SIM_HZ=10 OMP_NUM_THREADS=24"] * 2)
            self.assertFalse(C.cthr(FakeR(d), "R0")["PASS"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
