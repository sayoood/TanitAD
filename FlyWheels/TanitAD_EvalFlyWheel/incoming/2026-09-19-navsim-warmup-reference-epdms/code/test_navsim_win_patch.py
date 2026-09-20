#!/usr/bin/env python3
"""Pins the ONE behaviour patch in navsim_win.py.

Expectations are LITERALS (never an expression over the code under test), and the
original devkit expression is exercised alongside, so the test shows the defect it
exists to fix instead of asserting agreement with itself.

Run:  C:/Users/Admin/navsim/venv/Scripts/python.exe -m pytest -q code/test_navsim_win_patch.py
"""
import importlib.util
import pathlib
import tempfile

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("navsim_win", _HERE / "navsim_win.py")
nw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nw)

LINUX = "/root/exp/metric_cache/2021.08.16.14.23.37_veh-45_00015_00132/unknown/4cac9f6cd85a5b47/metric_cache.pkl"
WIN = r"C:\Users\Admin\navsim\exp\metric_cache_warmup_two_stage\2021.08.16.14.23.37_veh-45_00015_00132\unknown\5b733b329372ed8c8\metric_cache.pkl"
MIXED = "C:/Users/Admin/navsim/exp/mc\\2021.08.16.14.23.37_veh-45_00015_00132\\unknown\\0dc54a8c8203567b\\metric_cache.pkl"


def original(p: str) -> str:                     # navsim/common/dataloader.py:316, verbatim expression
    return p.split("/")[-2]


def test_linux_path_patched_equals_literal():
    assert nw.token_from_cache_path(LINUX) == "4cac9f6cd85a5b47"


def test_linux_path_patched_equals_original():   # identical wherever the original works
    assert nw.token_from_cache_path(LINUX) == original(LINUX)


def test_windows_path_patched_equals_literal():
    assert nw.token_from_cache_path(WIN) == "5b733b329372ed8c8"


def test_mixed_separators_patched_equals_literal():
    assert nw.token_from_cache_path(MIXED) == "0dc54a8c8203567b"


def test_original_FAILS_on_windows_path():       # the defect the patch exists for
    with pytest.raises(IndexError):
        original(WIN)


def test_patched_loader_reads_a_backslash_metadata_csv():
    """End-to-end on the real function body: a metadata CSV written the way nuPlan's
    save_cache_metadata writes it on Windows (header + str(WindowsPath) rows)."""
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        (root / "metadata").mkdir()
        rows = [WIN, WIN.replace("5b733b329372ed8c8", "4cac9f6cd85a5b47")]
        (root / "metadata" / "x_metadata_node_0.csv").write_text("file_name\n" + "\n".join(rows) + "\n")
        got = nw._load_metric_cache_paths_patched(None, root)
        assert sorted(got) == ["4cac9f6cd85a5b47", "5b733b329372ed8c8"]
        assert got["5b733b329372ed8c8"] == WIN
