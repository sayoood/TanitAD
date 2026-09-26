"""Pins W3's use of E1's loader patch on NAVSIM **v1.1** (run in the NAVSIM venv).

    PYTHONPATH=<v1.1 tree> C:/Users/Admin/navsim-crun/venv/Scripts/python.exe -m pytest -q tests/test_v1_loader_patch.py

E1's replacement was written for v2's ``dataloader.py:306-317``. W3 applies it to v1.1's
``dataloader.py:170-181``, which is only admissible if the two bodies are the SAME CODE except the
token expression. Expectations are LITERALS, never an expression over the code under test.
"""
import importlib.util
import inspect
import os
import sys
import textwrap

import pytest

V11 = "D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
E1 = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
      "2026-09-19-navsim-warmup-reference-epdms/code/navsim_win.py")
sys.path.insert(0, V11)

from navsim.common import dataloader as dl  # noqa: E402

_spec = importlib.util.spec_from_file_location("e1w", E1)
e1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e1)

#: v1.1 @ 3e8291b, navsim/common/dataloader.py:176-181, as READ 2026-09-19 (code lines only).
V11_BODY = [
    'metadata_dir = cache_path / "metadata"',
    'metadata_file = [file for file in metadata_dir.iterdir() if ".csv" in str(file)][0]',
    'with open(str(metadata_file), "r") as f:',
    'cache_paths = f.read().splitlines()[1:]',
    'metric_cache_dict = {cache_path.split("/")[-2]: cache_path for cache_path in cache_paths}',
    'return metric_cache_dict',
]
E1_TOKEN_LINE = ('metric_cache_dict = {token_from_cache_path(cache_path): cache_path '
                 'for cache_path in cache_paths}')


def _code_lines(fn):
    """Code lines of a function body: docstring, blank and comment lines dropped."""
    src = textwrap.dedent(inspect.getsource(fn)).splitlines()[1:]      # drop the def line
    out, in_doc = [], False
    for ln in src:
        s = ln.strip()
        if not in_doc and s.startswith('"""'):
            if s.count('"""') >= 2 and len(s) > 3:
                continue                                                  # one-line docstring
            in_doc = True
            continue
        if in_doc:
            if '"""' in s:
                in_doc = False
            continue
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def test_v11_body_is_the_expected_literal():
    assert _code_lines(dl.MetricCacheLoader._load_metric_cache_paths) == V11_BODY


def test_e1_body_equals_v11_except_the_token_line():
    got = _code_lines(e1._load_metric_cache_paths_patched)
    want = V11_BODY[:4] + [E1_TOKEN_LINE] + V11_BODY[5:]
    assert got == want


WIN = (r"D:\Archive\devbox-C\navsim\exp\w3_navtest_v1\metric_cache_smoke20"
       r"\2021.06.03.12.02.06_veh-35_01100_01227\unknown\0f206a62842b59b2\metric_cache.pkl")


def test_patched_token_on_a_v11_windows_path():
    assert e1.token_from_cache_path(WIN) == "0f206a62842b59b2"


def test_original_expression_fails_on_it():          # the defect, shown (C7 shows it on a real cache)
    with pytest.raises(IndexError):
        WIN.split("/")[-2]


def test_linux_path_unchanged():
    p = "/x/metric_cache/2021.06.03.12.02.06_veh-35_01100_01227/unknown/0f206a62842b59b2/metric_cache.pkl"
    assert e1.token_from_cache_path(p) == p.split("/")[-2] == "0f206a62842b59b2"
