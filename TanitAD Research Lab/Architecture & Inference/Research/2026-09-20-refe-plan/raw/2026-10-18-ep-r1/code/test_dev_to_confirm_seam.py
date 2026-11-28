"""THE SEAM between the two dev appliers and the confirm's navtest C2: the REAL dev results feed `navtest_c2`, not hand-written look-alikes.  stdlib unittest (also collected by pytest):

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I test_dev_to_confirm_seam.py -v

"Built, tested and unreachable from its caller" is a known defect class: `test_navtest_c2.py` builds its dev results by hand, so a renamed field in an applier would pass every one of its tests and break the
first real read.  Here the dev results are written by `epr1_dev.run_dev` (landed) and `srttc_dev.run_dev` (this package) on their own synthetic worlds, and the CONFIRM side must accept them:
  * `refe_head_confirm.read_dev_result` reads each result,
  * `navtest_c2.dev_c2_binding` finds the C2 result sha256 and the two C2 row sha256s the dev gate recorded,
  * `navtest_c2.load_c2_rows` loads the very C2 files the dev gate applied, bound to the arm's c* and bank, and refuses them once an edit makes them differ from what dev recorded.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SELRANK = os.path.normpath(os.path.join(HERE, "..", "..", "2026-10-18-sel-rank", "code"))
sys.path.insert(0, HERE)
if SELRANK not in sys.path:
    sys.path.append(SELRANK)          # AFTER the ep-r1 dir: the only shared module name (zero_tail.py) is byte-identical in both
import refe_head_confirm as C    # noqa: E402
import epr1_dev as D             # noqa: E402
import navtest_c2 as N           # noqa: E402
import test_epr1_dev as TE       # noqa: E402
import test_srttc_dev as TS      # noqa: E402
from test_epr1_dev import sha_f, wj   # noqa: E402

SK = D.SEED_KEYS


class Seam(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="seam_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def check_chain(self, arm, run_dir, out_dir, result_path, r1_name, real_rows_sha):
        g = C.ConfirmGuard(read_dirs=[run_dir, out_dir], write_dirs=[])
        state = C.read_dev_result(g, result_path, arm)                                         # the landed contract reader accepts the applier's output
        binding = N.dev_c2_binding(g, arm, state)                                              # and the C2 fields are where navtest_c2 looks for them
        spec = N.ARM_SPEC[arm]
        self.assertEqual(binding["c2_result_sha256"], sha_f(os.path.join(run_dir, spec["c2_result"])))
        self.assertEqual(set(binding["c2_rows_sha256"]), set(SK))
        arm_doc, _ = D.load_json(g, os.path.join(run_dir, r1_name))
        c2 = N.load_c2_rows(g, arm, run_dir, binding, arm_doc, real_rows_sha)                  # the dev gate's C2 files load, bound to this c* and this bank
        self.assertEqual({sd: v[2] for sd, v in c2["rows"].items()}, binding["c2_rows_sha256"])
        return g, state, binding, arm_doc, c2

    def test_epr1_dev_result_feeds_navtest_c2(self):
        w = TE.World(os.path.join(self.tmp, "ep"))
        o = w.opts()
        res = D.run_dev(o)
        self.assertEqual(res["inputs"]["c2_result_sha256"], sha_f(os.path.join(w.run_dir, "epr1_c2_result.json")))
        real = {sd: w.r1["final_rows"][sd]["sha256"] for sd in SK}
        g, state, binding, arm_doc, c2 = self.check_chain("EP-R1", w.run_dir, o.resolved_out_dir(), os.path.join(o.resolved_out_dir(), D.RESULT_NAME), "epr1_result.json", real)
        self.assertEqual(binding["c2_rows_sha256"], res["inputs"]["c2_rows_sha256"])
        self.assertEqual(state["verdict"], res["dev_gate"]["verdict"])
        # a C2 row edited after the dev gate read it is refused by the confirm, even though the file still loads
        edited = dict(binding, c2_rows_sha256={"0": binding["c2_rows_sha256"]["0"], "1": "0" * 64})
        with self.assertRaises(N.C2Refusal):
            N.load_c2_rows(g, "EP-R1", w.run_dir, edited, arm_doc, real)

    def test_srttc_dev_result_feeds_navtest_c2(self):
        w = TS.DW(os.path.join(self.tmp, "ttc"))
        o = w.opts()
        res = TS.D.run_dev(o)
        self.assertEqual(res["inputs"]["c2_result_sha256"], sha_f(os.path.join(w.run, "r1_c2_result.json")))
        real = {sd: w.r1doc()["final_rows"][sd]["sha256"] for sd in SK}
        g, state, binding, arm_doc, c2 = self.check_chain("SR-TTC", w.run, o.resolved_out_dir(), os.path.join(o.resolved_out_dir(), TS.D.RESULT_NAME), "r1_result.json", real)
        self.assertEqual(binding["c2_rows_sha256"], res["inputs"]["c2_rows_sha256"])
        self.assertEqual(state["verdict"], res["dev_gate"]["verdict"])
        self.assertEqual(state["rows_sha256"], real, "the rows the confirm will read are the rows the dev gate applied")
        edited = dict(binding, c2_result_sha256="0" * 64)
        with self.assertRaises(N.C2Refusal):
            N.load_c2_rows(g, "SR-TTC", w.run, edited, arm_doc, real)

    def test_the_srttc_c2_files_are_the_ones_srttc_c2_writes_not_a_hand_made_copy(self):
        """The C2 files in the DW world come from the real chain (srttc_r1.run_r1 -> srttc_c2.run_c2): their schema is what navtest_c2 binds against."""
        b = TS.base()
        self.assertEqual(b["c2"]["stage"], "C2")
        self.assertEqual(b["c2"]["c_star"]["config_sha256"], b["r1"]["c_star"]["config_sha256"])
        self.assertEqual(b["c2"]["bank_manifest_sha256"], b["r1"]["inputs"]["bank_manifest_sha256"])
        self.assertEqual(sorted(b["c2"]["final_rows"]), sorted(SK))
        self.assertTrue(all(e["file"].startswith("ttc_row_c2shuffle_seed") for e in b["c2"]["final_rows"].values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
