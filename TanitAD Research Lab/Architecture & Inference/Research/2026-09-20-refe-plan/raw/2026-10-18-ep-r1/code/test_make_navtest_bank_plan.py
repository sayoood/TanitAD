"""Tests for make_navtest_bank_plan.py on SYNTHETIC chunks / tables (CPU, no GPU, never the real navtest outcomes).

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I test_make_navtest_bank_plan.py -v

The plan shape is checked against the shape of the REAL DG-DUMP plan (`srttc_s0/work/dgdump/dgdump_plan.json`, 4 navtest tokens), and the file names the commands write are checked against the names
`refe_head_confirm.py` reads (a converter that writes a bank the applier cannot find is the built-and-unreachable defect class).
"""
from __future__ import annotations

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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epr1_dev as D            # noqa: E402
import make_navtest_bank_plan as P   # noqa: E402
import refe_head_confirm as C   # noqa: E402
from test_refe_head_confirm import NavWorld   # noqa: E402

SRC = os.path.join(HERE, "make_navtest_bank_plan.py")
DGDUMP_PLAN_KEYS = {"tokens": list, "token_log": dict, "token_root": dict, "db": dict}       # the shape of the real dgdump_plan.json


def _rt(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def load_pmutant(marker, old, new):
    lines = _rt(SRC).split("\n")
    idx = [i for i, ln in enumerate(lines) if re.search(rf"MUTATION-TARGET-PLAN-{marker}(?![A-Z0-9-])", ln)]
    assert len(idx) == 1, (marker, idx)
    assert lines[idx[0]].count(old) == 1, (marker, old)
    lines[idx[0]] = lines[idx[0]].replace(old, new)
    mod = types.ModuleType("make_navtest_bank_plan_mutant")
    mod.__file__ = SRC
    sys.modules[mod.__name__] = mod
    try:
        exec(compile("\n".join(lines), SRC, "exec"), mod.__dict__)
    finally:
        sys.modules.pop(mod.__name__, None)
    return mod


class Plan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="refeplan_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.n_logs, self.per_log, self.n_chunks = 6, 4, 3
        self.logs = [f"2021.05.25.14.16.{i:02d}_veh-35_{i:05d}_{i + 9:05d}" for i in range(self.n_logs)]
        self.tokens = [f"{i:016x}" for i in range(self.n_logs * self.per_log)]
        self.tlog = {t: self.logs[i // self.per_log] for i, t in enumerate(self.tokens)}
        self.chunks_dir = os.path.join(self.tmp, "chunks")
        self.db_dir = os.path.join(self.tmp, "db")
        os.makedirs(self.chunks_dir)
        os.makedirs(self.db_dir)
        for lg in self.logs:
            open(os.path.join(self.db_dir, f"{lg}.db"), "wb").close()
        self.table = os.path.join(self.tmp, "table_p2_full.npz")
        np.savez(self.table, token=np.array(self.tokens), log=np.array([self.tlog[t] for t in self.tokens]))        # NO outcome member at all
        self.write_chunks()

    def write_chunks(self, groups=None, mutate=None):
        groups = groups or [self.tokens[:8], self.tokens[8:16], self.tokens[16:]]
        ents = []
        for i, g in enumerate(groups):
            doc = {"rule": f"chunk {i}", "tokens": g, "token_log": {t: self.tlog[t] for t in g}}
            if mutate:
                mutate(i, doc)
            raw = json.dumps(doc, indent=0).encode("utf-8")
            with open(os.path.join(self.chunks_dir, f"c{i:02d}_tokens.json"), "wb") as f:
                f.write(raw)
            ents.append({"id": i, "n": len(g), "n_logs": len({self.tlog[t] for t in g}), "tokens_file": f"c{i:02d}_tokens.json", "tokens_sha256": hashlib.sha256(raw).hexdigest()})
        with open(os.path.join(self.chunks_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump({"n_tokens": len(self.tokens), "chunks": ents}, f)

    def build(self, mod=P, **kw):
        out = os.path.join(self.tmp, f"plans{len(os.listdir(self.tmp))}")
        args = dict(chunks_dir=self.chunks_dir, table=self.table, out_dir=out, bank_dir="D:/bank", table_sha256=None, db_dir=self.db_dir, expect_tokens=len(self.tokens),
                    expect_logs=self.n_logs, expect_chunks=self.n_chunks, frames_root="D:/frames")
        args.update(kw)
        return mod.build(**args), out

    # ---- happy path
    def test_plan_files_have_the_shape_of_the_real_dgdump_plan(self):
        man, out = self.build()
        self.assertEqual(man["n_chunks"], 3)
        seen = []
        for i in range(3):
            plan = json.loads(_rt(os.path.join(out, f"plan_c{i:02d}.json")))
            self.assertEqual(set(plan), set(DGDUMP_PLAN_KEYS))
            for k, typ in DGDUMP_PLAN_KEYS.items():
                self.assertIsInstance(plan[k], typ, k)
            self.assertEqual(sorted(plan["token_log"]), sorted(plan["tokens"]))
            self.assertEqual(set(plan["token_root"].values()), {"D:/frames"})
            self.assertEqual(sorted(plan["db"]), sorted(set(plan["token_log"].values())))
            for lg, dbp in plan["db"].items():
                self.assertEqual(dbp, f"{self.db_dir}/{lg}.db")
            self.assertTrue(all(plan["token_log"][t] == self.tlog[t] for t in plan["tokens"]))
            seen += plan["tokens"]
        self.assertEqual(sorted(seen), sorted(self.tokens), "every table token is in exactly one plan")

    def test_the_real_dgdump_plan_has_the_same_shape_when_present(self):
        p = "C:/Users/Admin/srttc_s0/work/dgdump/dgdump_plan.json"
        if not os.path.exists(p):
            self.skipTest("the DG-DUMP plan is not on this box")
        real = json.loads(_rt(p))
        self.assertEqual({k: type(v) for k, v in real.items()}, DGDUMP_PLAN_KEYS)

    def test_commands_run_the_fan_path_with_features_and_write_the_names_the_applier_reads(self):
        man, out = self.build(bank_dir="D:/bank")
        cmds = _rt(os.path.join(out, "bank_commands.txt")).strip().split("\n")
        self.assertEqual(len(cmds), 3)
        for i, c in enumerate(cmds):
            self.assertIn("srttc_s0_dump.py", c)
            self.assertIn("--feat-out", c)
            self.assertIn("--device cuda", c)
            self.assertNotIn("--lean", c, "lean drops the features: the bank would be empty")
            self.assertNotIn("--sets-dir", c, "the fan path, not the S0 fit-set path")
            self.assertIn(f"--out D:/bank/c{i:02d}{C.DUMP_SUFFIX}", c)
            self.assertIn(f"--feat-out D:/bank/c{i:02d}{C.FEAT_SUFFIX}", c)
            self.assertIn("model_final.pt", c)

    def test_command_names_are_exactly_the_names_a_navworld_bank_has(self):
        nw_root = os.path.join(self.tmp, "nw")
        nw = NavWorld(nw_root, outcomes=False)
        names = sorted(os.listdir(nw.bank_dir))
        man, out = self.build(bank_dir=nw.bank_dir.replace("\\", "/"))
        cmds = _rt(os.path.join(out, "bank_commands.txt")).strip().split("\n")
        written = sorted(os.path.basename(m) for c in cmds for m in re.findall(r"--(?:feat-)?out (\S+)", c))
        self.assertEqual(written, [n for n in names if n.endswith(".npz")], "the converter and the applier disagree about the bank's file names")
        self.assertEqual([stem for stem in C.list_bank_chunks(nw.opts().guard(), nw.bank_dir)], [f"c{i:02d}" for i in range(3)])

    def test_only_token_and_log_are_opened_from_the_table(self):
        real_load = np.load
        touched = []

        class Spy:
            def __init__(self, z):
                self.z, self.files = z, z.files

            def __getitem__(self, k):
                touched.append(k)
                return self.z[k]

            def close(self):
                self.z.close()
        with mock.patch.object(np, "load", lambda *a, **k: Spy(real_load(*a, **k)) if str(a[0]).endswith(".npz") else real_load(*a, **k)):
            self.build()
        self.assertEqual(sorted(set(touched)), ["log", "token"])

    # ---- refusals
    def test_tampered_token_file_is_refused(self):
        with open(os.path.join(self.chunks_dir, "c01_tokens.json"), "ab") as f:
            f.write(b" ")
        with self.assertRaises(P.PlanRefusal) as cm:
            self.build()
        self.assertIn("sha256", str(cm.exception))

    def test_a_table_token_in_no_chunk_is_refused(self):
        self.write_chunks(groups=[self.tokens[:8], self.tokens[8:16], self.tokens[16:-1]])
        with self.assertRaises(P.PlanRefusal) as cm:
            self.build()
        self.assertIn("in no chunk", str(cm.exception))

    def test_a_token_in_two_chunks_is_refused(self):
        self.write_chunks(groups=[self.tokens[:8], self.tokens[7:16], self.tokens[16:]])
        with self.assertRaises(P.PlanRefusal) as cm:
            self.build()
        self.assertIn("is in chunks", str(cm.exception))

    def test_a_chunk_whose_token_log_disagrees_with_the_table_is_refused(self):
        def bad(i, doc):
            if i == 0:
                t = doc["tokens"][0]
                doc["token_log"][t] = self.logs[-1]
        self.write_chunks(mutate=bad)
        with self.assertRaises(P.PlanRefusal) as cm:
            self.build()
        self.assertIn("log", str(cm.exception))

    def test_a_missing_nuplan_db_is_refused_and_can_be_waived_explicitly(self):
        os.remove(os.path.join(self.db_dir, f"{self.logs[2]}.db"))
        with self.assertRaises(P.PlanRefusal) as cm:
            self.build()
        self.assertIn("DB", str(cm.exception))
        man, out = self.build(check_db=False)
        self.assertEqual(man["n_chunks"], 3)

    def test_pins(self):
        with self.assertRaises(P.PlanRefusal):
            self.build(table_sha256="0" * 64)
        with self.assertRaises(P.PlanRefusal):
            self.build(expect_tokens=len(self.tokens) + 1)
        with self.assertRaises(P.PlanRefusal):
            self.build(expect_logs=self.n_logs + 1)
        with self.assertRaises(P.PlanRefusal):
            self.build(expect_chunks=4)

    def test_defaults_are_the_landed_navtest_constants(self):
        self.assertEqual((P.N_TOKENS, P.N_LOGS, P.N_CHUNKS), (12146, 136, 14))
        self.assertEqual(C.NAVTEST_TABLE_SHA256, "33441082b7044c951e3cef5fb898cac50797039aed93560b2712e3ff4c34906e")

    def test_main_returns_2_on_refusal(self):
        rc = P.main(["--chunks-dir", self.chunks_dir, "--out-dir", os.path.join(self.tmp, "o"), "--bank-dir", "D:/bank", "--table", self.table])
        self.assertEqual(rc, 2, "the real pins (12,146 tokens) must refuse a 24-token synthetic table")

    # ---- mutants
    def test_sha_mutant(self):
        with open(os.path.join(self.chunks_dir, "c01_tokens.json"), "ab") as f:
            f.write(b" ")
        with self.assertRaises(P.PlanRefusal):
            self.build()
        mut = load_pmutant("SHA", 'sha != e["tokens_sha256"]', "False")
        man, _ = self.build(mod=mut)
        self.assertEqual(man["n_chunks"], 3)

    def test_missing_mutant(self):
        self.write_chunks(groups=[self.tokens[:8], self.tokens[8:16], self.tokens[16:-1]])
        mut = load_pmutant("MISSING", "if missing:", "if False:")
        man, _ = self.build(mod=mut)
        self.assertEqual(man["n_chunks"], 3, "the mutant is the one that plans a bank with a token missing")

    def test_dup_mutant(self):
        self.write_chunks(groups=[self.tokens[:8], self.tokens[7:16], self.tokens[16:]])
        mut = load_pmutant("DUP", "if t in seen:", "if False:")
        man, _ = self.build(mod=mut)
        self.assertEqual(man["n_chunks"], 3)

    def test_log_mutant(self):
        def bad(i, doc):
            if i == 0:
                doc["token_log"][doc["tokens"][0]] = self.logs[-1]
        self.write_chunks(mutate=bad)
        mut = load_pmutant("LOG", "if tl[t] != tlog[t]:", "if False:")
        man, _ = self.build(mod=mut)
        self.assertEqual(man["n_chunks"], 3)

    def test_every_marker_is_mutated_here(self):
        src, me = _rt(SRC), _rt(__file__)
        markers = sorted(set(re.findall(r"MUTATION-TARGET-PLAN-([A-Z0-9-]+?)(?![A-Z0-9-])", src)))
        self.assertGreaterEqual(len(markers), 4)
        self.assertEqual([m for m in markers if f'load_pmutant("{m}"' not in me], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
