#!/usr/bin/env python3
"""Offline tests for the RJ5 navtrain chain (no nuplan, no DB). Run: python test_rj5.py  (needs numpy: the driverl-eval or box-2 venv).
Each control has a mutation arm that must go RED. The end-to-end reach test (RJ5 rows differ from R0 rows, R0 reproduces the bank) is the box-2 smoke."""
import gzip
import hashlib
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rj5_chain as C  # noqa: E402
import rj5_merge as M  # noqa: E402
import rj5_roll as R  # noqa: E402


def traj(seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return (np.cumsum(rng.normal(1.0, 0.1, (20, 3)), axis=0) * scale).tolist()


def write_gz(p, rows):
    with gzip.open(p, "wt", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class T(unittest.TestCase):
    def test_knobs_registered_and_arms(self):
        self.assertEqual(R.expected_knobs("R0"), R.REGISTERED)
        e = R.expected_knobs("RJ5")
        self.assertEqual(e["positive_jerk_limit_when_nonnegative_acc"], 5.0)
        self.assertEqual({k: v for k, v in e.items() if k != "positive_jerk_limit_when_nonnegative_acc"},
                         {k: v for k, v in R.REGISTERED.items() if k != "positive_jerk_limit_when_nonnegative_acc"})
        self.assertEqual(sorted(R.ARM_JERK), ["R0", "RJ5"])      # nothing else is reachable from the CLI

    def test_knob_mismatch_mutation(self):
        built = dict(R.REGISTERED)
        self.assertEqual(R.knob_mismatch(built, R.expected_knobs("R0")), [])
        self.assertTrue(R.knob_mismatch(built, R.expected_knobs("RJ5")))          # the dropped-override mutation must be RED
        self.assertTrue(R.knob_mismatch({}, R.expected_knobs("R0")))              # a knob that was never read back is a mismatch

    def test_source_rebinds_overrides_before_build(self):
        src = open(os.path.join(HERE, "rj5_roll.py"), encoding="utf-8").read()
        i, j = src.index("BTR.TEACHER_OVERRIDES = overrides_for(arm)"), src.index('BTR.build_planner("cpu")')
        self.assertLess(i, j)
        self.assertIn("KNOB-NOT-REACHED", src)

    def test_shards_partition_exactly(self):
        by = {f"log{i:03d}": {"t": []} for i in range(37)}
        got = []
        for s in range(5):
            got += R.pick_logs(by, None, f"{s}/5", 0)
        self.assertEqual(sorted(got), sorted(by))
        self.assertEqual(len(got), len(set(got)))
        with self.assertRaises(SystemExit):
            R.pick_logs(by, None, "5/5", 0)

    def test_resume_ignores_torn_and_error_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "rj5_x.jsonl")
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"log_name": "a", "token": "t1", "rank": 0, "arm": "RJ5", "traj": [[0, 0, 0]]}) + "\n")
                f.write(json.dumps({"log_name": "a", "token": "t2", "rank": 0, "arm": "RJ5", "error": "x"}) + "\n")
                f.write('{"log_name": "a", "tok')                    # torn last line
            self.assertEqual(R.done_keys(p), {("a", "t1", 0, "RJ5")})

    def test_traj_sha_is_float64_le_bytes(self):
        t = traj(1)
        self.assertEqual(M.traj_sha(t), R.traj_sha(t))
        self.assertEqual(M.traj_sha(t), hashlib.sha256(np.asarray(t, np.float64).tobytes()).hexdigest())

    def _fixture(self, d):
        bank = [{"log_name": "L1", "token": f"t{i}", "rank": 0, "traj": traj(i)} for i in range(6)] + \
               [{"log_name": "L2", "token": "u0", "rank": 0, "traj": traj(99)}]
        write_gz(os.path.join(d, "bank.jsonl.gz"), bank)
        rows = []
        for i in range(5):                                           # t5 and u0 have no RJ5 row -> fallbacks
            b = bank[i]["traj"]
            moved = (np.asarray(b) + (0.5 if i == 0 else 0.0)).tolist()
            rows.append({"log_name": "L1", "token": f"t{i}", "rank": 0, "arm": "RJ5", "traj": moved})
        rows.append({"log_name": "L1", "token": "t0", "rank": 0, "arm": "R0", "traj": bank[0]["traj"], "bank_dmax": 0.0})
        rows.append({"log_name": "L1", "token": "t1", "rank": 0, "arm": "R0", "traj": bank[1]["traj"], "bank_dmax": 0.0})
        rows.append({"log_name": "L1", "token": "t5", "rank": 0, "arm": "RJ5", "error": "boom"})
        os.makedirs(os.path.join(d, "rows"), exist_ok=True)
        with open(os.path.join(d, "rows", "rj5_a.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return bank, rows

    def test_merge_coverage_fallback_control(self):
        with tempfile.TemporaryDirectory() as d:
            bank, rows = self._fixture(d)
            files = [os.path.join(d, "rows", "rj5_a.jsonl")]
            out, rep, ok = M.merge(os.path.join(d, "bank.jsonl.gz"), files, 1.0)
            self.assertFalse(ok)                                      # 2 of 7 keys fall back: coverage < 1.0 REFUSES
            self.assertEqual(rep["n_fallback"], 2)
            self.assertEqual(rep["n_error_rows"], 1)
            self.assertEqual(rep["control_R0"]["n"], 2)
            self.assertEqual(rep["control_R0"]["max_abs"], 0.0)
            self.assertEqual(rep["rj5_movement"]["n_moved_gt_1e-6_m"], 1)
            self.assertEqual({r["source"] for r in out}, {"rj5", "raw_fallback"})
            fb = [r for r in out if r["source"] == "raw_fallback"]
            self.assertTrue(all(r["traj_sha256"] == r["bank_traj_sha256"] and r["dmax_vs_bank"] == 0.0 for r in fb))
            _, rep2, ok2 = M.merge(os.path.join(d, "bank.jsonl.gz"), files, 0.7)
            self.assertTrue(ok2)

    def test_merge_conflicting_duplicate_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            bank, rows = self._fixture(d)
            dup = dict(rows[0])
            dup["traj"] = (np.asarray(dup["traj"]) + 1e-3).tolist()
            with open(os.path.join(d, "rows", "rj5_b.jsonl"), "w", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(dup) + "\n")
            files = [os.path.join(d, "rows", "rj5_a.jsonl"), os.path.join(d, "rows", "rj5_b.jsonl")]
            _, rep, ok = M.merge(os.path.join(d, "bank.jsonl.gz"), files, 0.0)
            self.assertEqual(rep["n_conflicting_duplicates"], 1)
            self.assertFalse(ok)

    def test_merge_main_writes_deterministic_output(self):
        with tempfile.TemporaryDirectory() as d:
            self._fixture(d)
            a = ["--bank", os.path.join(d, "bank.jsonl.gz"), "--rows-dir", os.path.join(d, "rows"), "--min-coverage", "0.0"]
            self.assertEqual(M.main(a + ["--out-dir", os.path.join(d, "o1")]), 0)
            self.assertEqual(M.main(a + ["--out-dir", os.path.join(d, "o2")]), 0)
            r1 = json.load(open(os.path.join(d, "o1", "rj5_merge_report.json"), encoding="utf-8"))
            r2 = json.load(open(os.path.join(d, "o2", "rj5_merge_report.json"), encoding="utf-8"))
            self.assertEqual(r1["output_sha256"], r2["output_sha256"])

    def test_chain_helpers(self):
        self.assertEqual(C.split_batches(list("abcde"), 2), [["a", "b"], ["c", "d"], ["e"]])
        self.assertEqual(C.pick_control_logs(list("abcdef"), 3), ["a", "d"])
        self.assertEqual(C.pick_control_logs(list("abcdef"), 0), [])
        # global index: batches of 2 with every=4 -> controls only in batches 0 and 2
        logs = list("abcdefgh")
        got = [C.pick_control_logs(b, 4, bi * 2) for bi, b in enumerate(C.split_batches(logs, 2))]
        self.assertEqual(got, [["a"], [], ["e"], []])
        self.assertNotEqual(got, [C.pick_control_logs(b, 4) for b in C.split_batches(logs, 2)])  # old per-batch rule differs


    def _guard_main(self, free_gb, mutate=False):
        """Drive main() with --fetch through a stub fetch script that leaves a marker; disk_usage reports free_gb."""
        from collections import namedtuple
        d = tempfile.mkdtemp()
        stub = os.path.join(d, "stub_fetch.py")
        marker = os.path.join(d, "FETCH_RAN")
        with open(stub, "w", encoding="utf-8") as f:
            f.write("open(%r, 'w').write('x')\n" % marker)
        lf = os.path.join(d, "logs.txt")
        with open(lf, "w", encoding="utf-8") as f:
            f.write("logA\nlogB\n")
        DU = namedtuple("DU", "total used free")
        real_du, real_guard = C.shutil.disk_usage, C.disk_guard
        C.shutil.disk_usage = lambda p: DU(0, 0, int(free_gb * 2**30))
        if mutate:
            C.disk_guard = lambda *a: (True, 0.0)
        try:
            rc = C.main(["--bank", os.path.join(d, "bank.jsonl.gz"), "--logs-file", lf, "--work", os.path.join(d, "w"),
                         "--db-dir", d, "--fetch", "--fetch-script", stub, "--procs", "1", "--batch-logs", "2",
                         "--control-every", "0", "--min-free-gb", "20", "--est-gb-per-log", "0.3", "--max-batches", "1"])
        finally:
            C.shutil.disk_usage, C.disk_guard = real_du, real_guard
        return rc, os.path.exists(marker)

    def test_disk_guard_reachable_from_main(self):
        self.assertEqual(C.disk_guard(21.0, 2, 0.3, 20.0)[0], True)
        self.assertEqual(C.disk_guard(20.5, 2, 0.3, 20.0)[0], False)
        rc, ran = self._guard_main(20.5)
        self.assertEqual((rc, ran), (6, False))   # refused before any fetch

    def test_disk_guard_mutation_goes_red(self):
        rc, ran = self._guard_main(20.5, mutate=True)
        self.assertTrue(ran)                     # with the guard disabled the fetch runs: the test above discriminates
        self.assertEqual(rc, 0)                  # free 20.5 >= 20 after the (empty) fetch, so the chain completes
        rc2, ran2 = self._guard_main(19.0, mutate=True)
        self.assertEqual((rc2, ran2), (6, True))  # post-fetch check stops it even when the pre-fetch guard is bypassed


if __name__ == "__main__":
    unittest.main(verbosity=2)
