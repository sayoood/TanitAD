"""refcv9 B3 / E1-BOX-OWN-RECORD: the overfit record of the B3 head. Same inherited protocol literals, a NEW schema string only."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

G = pytest.importorskip("g_box_overfit")
SRC = (ROOT / "scripts" / "g_box_overfit.py").read_text(encoding="utf-8")


def test_no_b3_token_leaves_the_record_untouched():
    assert G.b3_record_fields(["--w-box3d", "1.0", "--trunk-compile"]) == {}


def test_b3_tokens_stamp_the_schema_and_the_flags_and_no_new_literal():
    r = G.b3_record_fields(["--w-box3d", "1.0", "--r9-box-quality", "--r9-box-dn", "--r9-box-class-sqrt"])
    assert r["schema"] == "tanitad.g_box_overfit_record/2-b3" == G.B3_SCHEMA
    assert r["b3"]["flags"] == ["--r9-box-quality", "--r9-box-dn", "--r9-box-class-sqrt"]
    assert "literals" in r["b3"] and isinstance(r["b3"]["literals"], str)           # a pointer to rec["literals"], not a second literal set


def test_unknown_r9_box_token_is_refused_and_the_build_argv_keeps_every_b3_flag():
    with pytest.raises(SystemExit):
        G.b3_record_fields(["--r9-box-typo"])
    argv = ["--r9-box-dn", "--trunk-compile", "--r9-box-quality"]
    assert G.harness_build_argv(argv)[0] == ["--r9-box-dn", "--r9-box-quality"]       # only --trunk-compile is dropped


def test_main_stamps_the_b3_fields_from_the_launch_argv_RED_if_the_call_is_removed():
    assert SRC.count("rec.update(b3_record_fields(launch_argv))") == 1
    assert "rec.update(b3_record_fields(launch_argv))" not in SRC.replace("rec.update(b3_record_fields(launch_argv))", "pass")
