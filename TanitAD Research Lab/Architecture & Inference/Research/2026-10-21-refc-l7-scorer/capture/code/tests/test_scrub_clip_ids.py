"""scrub_clip_ids.py: the clip-id rule's enforcement, with literal expectations and a deliberate-regression arm for each way it could read clean while dirty.

    python tests/test_scrub_clip_ids.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
TOOL = str(HERE / "scrub_clip_ids.py")
ID = "-".join(["deadbeef", "1234", "4abc", "9def", "0123456789ab"])   # SYNTHETIC id, built from parts so this file holds no UUID-shaped literal; the expected sha12 below is a literal

LITERAL_SHA12 = "clip-sha12-d4103be19f24"


def run(*args):
    r = subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)
    return r.returncode, r.stdout


def test_full_id_and_prefix_rewritten_and_clean():
    with tempfile.TemporaryDirectory() as t:
        (Path(t) / "manifest.json").write_text(json.dumps({"clip_id": ID}), encoding="utf-8")
        (Path(t) / "roll.log").write_text("  [1/1] deadbeef-123 3 windows 116s\n", encoding="utf-8")
        rc, out = run("--root", t, "--json", str(Path(t) / "rep.json"))
        assert rc == 0 and "ZZL7CSCRUBOKZZ" in out, out
        assert json.loads((Path(t) / "manifest.json").read_text())["clip_id"] == LITERAL_SHA12
        assert (Path(t) / "roll.log").read_text() == "  [1/1] " + LITERAL_SHA12 + " 3 windows 116s\n"
        assert json.loads((Path(t) / "rep.json").read_text())["tokens_rewritten"] == 2


def test_check_only_on_a_dirty_tree_is_BAD_and_does_not_rewrite():
    with tempfile.TemporaryDirectory() as t:
        (Path(t) / "manifest.json").write_text(json.dumps({"clip_id": ID}), encoding="utf-8")
        rc, out = run("--root", t, "--check-only")
        assert rc == 3 and "ZZL7CSCRUBBADZZ" in out, out
        assert ID in (Path(t) / "manifest.json").read_text()


def test_empty_scan_is_inconclusive_not_clean():
    with tempfile.TemporaryDirectory() as t:
        rc, out = run("--root", t)
        assert rc == 3 and "ZZL7CSCRUBBADZZ" in out, "an empty scan passed"


def test_npz_string_array_with_an_id_is_BAD():
    with tempfile.TemporaryDirectory() as t:
        (Path(t) / "ok.txt").write_text("nothing here\n", encoding="utf-8")
        (Path(t) / "ep.npz").write_bytes(b"")
        np.savez(Path(t) / "ep.npz", clip_id=np.asarray([ID]))
        # the id is only learnable from a text file or --known-ids: supply it the way the Thor run does
        (Path(t) / "known.txt").write_text(ID + "\n", encoding="utf-8")
        rc, out = run("--root", t, "--known-ids", str(Path(t) / "known.txt"))
        assert rc == 3 and "ZZL7CSCRUBBADZZ" in out and "npz-string" in out, out


def test_unrelated_hex_is_untouched():
    with tempfile.TemporaryDirectory() as t:
        (Path(t) / "a.json").write_text(json.dumps({"clip_id": ID, "md5": "83e30bd3791f0767707eade5f4671759", "skip": "f09e44db"}), encoding="utf-8")
        rc, _ = run("--root", t)
        d = json.loads((Path(t) / "a.json").read_text())
        assert rc == 0 and d["md5"] == "83e30bd3791f0767707eade5f4671759" and d["skip"] == "f09e44db" and d["clip_id"] == LITERAL_SHA12


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("PASS", name)
    print("ZZL7CSCRUBTESTSPASS", n)
