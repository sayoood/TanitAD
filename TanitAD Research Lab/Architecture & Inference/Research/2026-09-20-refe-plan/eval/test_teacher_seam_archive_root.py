"""teacher_navtest_seam.resolve_archive_file: the navsim archive is found by the FILE, never by the root directory.

Why: on 2026-11-12 a one-file copy at E:/Archive/devbox-C/navsim/exp/... (made because this module hardcoded E:) made the E:
root "exist", and the pinned harness, which takes the first existing root DIRECTORY, then failed (TF-N2 A1 score_T).
The mutation arm swaps the file test for the old directory test and requires the shadowing test to go RED.
Run: CUDA_VISIBLE_DEVICES=-1 python -m unittest test_teacher_seam_archive_root  (from eval/)
"""
import os
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import teacher_navtest_seam as T  # noqa: E402

REL = "exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"


def _touch(root, rel):
    p = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").close()


class ArchiveRoot(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.a = os.path.join(self.td.name, "a").replace("\\", "/")
        self.b = os.path.join(self.td.name, "b").replace("\\", "/")

    def tearDown(self):
        self.td.cleanup()

    def shadow_case(self, fn):
        # root a EXISTS and even holds part of the tree, but not the file; root b holds the file
        os.makedirs(os.path.join(self.a, "exp", "w3_navtest_v1"))
        _touch(self.b, REL)
        return fn(REL, env={}, roots=[self.a, self.b])

    def test_partial_root_does_not_shadow(self):
        self.assertEqual(self.shadow_case(T.resolve_archive_file), f"{self.b}/{REL}")

    def test_first_root_holding_the_file_wins(self):
        _touch(self.a, REL)
        _touch(self.b, REL)
        self.assertEqual(T.resolve_archive_file(REL, env={}, roots=[self.a, self.b]), f"{self.a}/{REL}")

    def test_env_var_overrides(self):
        got = T.resolve_archive_file(REL, env={"TANITAD_NAVSIM_ROOT": self.b.replace("/", "\\") + "\\"}, roots=[self.a])
        self.assertEqual(got, f"{self.b}/{REL}")

    def test_none_holds_it_returns_first_root(self):
        self.assertEqual(T.resolve_archive_file(REL, env={}, roots=[self.a, self.b]), f"{self.a}/{REL}")

    def test_mutation_directory_test_goes_red(self):
        with open(T.__file__, encoding="utf-8") as fh:
            src = fh.read()
        start = src.index("def resolve_archive_file")
        end = src.index("\n\n\nEXPORT = ")
        body = src[start:end]
        self.assertEqual(body.count("if os.path.isfile(p):"), 1)
        mutant_src = body.replace("if os.path.isfile(p):", "if os.path.isdir(root):")
        ns = types.ModuleType("mutant")
        ns.__dict__["os"] = os
        exec("from __future__ import annotations\n" + mutant_src, ns.__dict__)
        self.assertNotEqual(self.shadow_case(ns.resolve_archive_file), f"{self.b}/{REL}",
                            "the mutation (directory test) must reproduce the 2026-11-12 shadowing")

    def test_module_export_is_a_file_on_this_box(self):
        roots = [f"{d}:/Archive/devbox-C/navsim" for d in "EFD"]
        if not any(os.path.isfile(f"{r}/{REL}") for r in roots) and not os.environ.get("TANITAD_NAVSIM_ROOT"):
            self.skipTest("no navsim archive on this machine")
        self.assertTrue(os.path.isfile(T.EXPORT), T.EXPORT)


if __name__ == "__main__":
    unittest.main()
