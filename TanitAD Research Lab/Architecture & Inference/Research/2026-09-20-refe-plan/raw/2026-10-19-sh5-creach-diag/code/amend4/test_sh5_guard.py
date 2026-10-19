"""Tests of the seal guard and the Amendment chain (a checker's own tests; not pinned).   python -B test_sh5_guard.py"""
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sh5_guard as G  # noqa: E402

TMP = os.environ.get("SH5_TMP")


def w(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def make_root():
    r = tempfile.mkdtemp(dir=TMP)
    w(f"{r}/PREREG_SH5_DAC_PROX.md", "prereg v1\n")
    w(f"{r}/PREREG_SH5_DAC_PROX.sha256", G.sha256_file(f"{r}/PREREG_SH5_DAC_PROX.md") + "  PREREG_SH5_DAC_PROX.md\n")
    w(f"{r}/code/a.py", "a = 1\n")
    w(f"{r}/code/b.py", "b = 1\n")
    json.dump({"files": {"code/a.py": G.sha256_file(f"{r}/code/a.py"), "code/b.py": G.sha256_file(f"{r}/code/b.py")}}, open(f"{r}/CODE_PINS.json", "w"))
    return r


def amend(r, k, changed, text="amendment\n", old=None):
    pins = json.load(open(f"{r}/CODE_PINS.json"))["files"]
    for j in range(1, k):
        pins.update(json.load(open(f"{r}/CODE_PINS_AMENDMENT{j}.json"))["to"])
    frm = {n: pins[n] for n in changed}
    to = {n: G.sha256_file(f"{r}/{n}") for n in changed}
    w(f"{r}/PREREG_SH5_DAC_PROX_AMENDMENT{k}.md", text)
    w(f"{r}/PREREG_SH5_DAC_PROX_AMENDMENT{k}.sha256", G.sha256_file(f"{r}/PREREG_SH5_DAC_PROX_AMENDMENT{k}.md") + f"  x\n")
    json.dump({"from": old or frm, "to": to}, open(f"{r}/CODE_PINS_AMENDMENT{k}.json", "w"))


class TestGuard(unittest.TestCase):
    def test_sealed_ok_and_edits_refuse(self):
        r = make_root()
        self.assertEqual(G.require_sealed(r)["pins"], 2)
        w(f"{r}/code/a.py", "a = 2\n")
        with self.assertRaises(G.NotSealed):
            G.require_sealed(r)                                                    # a pinned file edited after the seal refuses
        w(f"{r}/code/a.py", "a = 1\n")
        self.assertEqual(G.require_sealed(r)["pins"], 2)
        w(f"{r}/PREREG_SH5_DAC_PROX.md", "prereg v2\n")
        with self.assertRaises(G.NotSealed):
            G.require_sealed(r)                                                    # an edited prereg refuses
        shutil.rmtree(r)
        with self.assertRaises(Exception):
            G.require_sealed(r)                                                    # no seal, no number

    def test_amendment_moves_exactly_the_listed_file(self):
        r = make_root()
        w(f"{r}/code/a.py", "a = 2\n")
        amend(r, 1, ["code/a.py"])
        self.assertEqual(G.require_sealed(r)["amendments"], [1])
        w(f"{r}/code/b.py", "b = 2\n")
        with self.assertRaises(G.NotSealed):
            G.require_sealed(r)                                                    # RED: a file the amendment did not list still refuses
        w(f"{r}/code/b.py", "b = 1\n")
        w(f"{r}/code/a.py", "a = 3\n")
        with self.assertRaises(G.NotSealed):
            G.require_sealed(r)                                                    # RED: the amended file edited again after the amendment refuses

    def test_amendment_chain_and_tampering(self):
        r = make_root()
        w(f"{r}/code/a.py", "a = 2\n")
        amend(r, 1, ["code/a.py"])
        w(f"{r}/code/a.py", "a = 3\n")
        amend(r, 2, ["code/a.py"])
        self.assertEqual(G.require_sealed(r)["amendments"], [1, 2])
        # tamper the amendment text: its record no longer matches
        w(f"{r}/PREREG_SH5_DAC_PROX_AMENDMENT2.md", "edited\n")
        with self.assertRaises(G.NotSealed):
            G.require_sealed(r)
        # a broken chain: amendment 2 claims to start from the SEALED pin, not from amendment 1's
        r2 = make_root()
        w(f"{r2}/code/a.py", "a = 2\n")
        amend(r2, 1, ["code/a.py"])
        w(f"{r2}/code/a.py", "a = 3\n")
        amend(r2, 2, ["code/a.py"], old={"code/a.py": json.load(open(f"{r2}/CODE_PINS.json"))["files"]["code/a.py"]})
        with self.assertRaises(G.NotSealed):
            G.require_sealed(r2)

    def test_amendment_cannot_pin_a_new_file_or_lack_its_files(self):
        r = make_root()
        w(f"{r}/code/c.py", "c = 1\n")
        w(f"{r}/PREREG_SH5_DAC_PROX_AMENDMENT1.md", "x\n")
        w(f"{r}/PREREG_SH5_DAC_PROX_AMENDMENT1.sha256", G.sha256_file(f"{r}/PREREG_SH5_DAC_PROX_AMENDMENT1.md") + "  x\n")
        json.dump({"from": {"code/c.py": "0" * 64}, "to": {"code/c.py": G.sha256_file(f"{r}/code/c.py")}}, open(f"{r}/CODE_PINS_AMENDMENT1.json", "w"))
        with self.assertRaises(G.NotSealed):
            G.require_sealed(r)                                                    # an amendment re-pins only files that were already pinned
        r2 = make_root()
        w(f"{r2}/PREREG_SH5_DAC_PROX_AMENDMENT1.md", "x\n")
        with self.assertRaises(G.NotSealed):
            G.require_sealed(r2)                                                   # an amendment text without its hash and pins does not count


if __name__ == "__main__":
    unittest.main(verbosity=1)
