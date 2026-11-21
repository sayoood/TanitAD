"""Tests of run_dumps_obd.py: planted fake banks (no checkpoint is loaded, no model run, subprocess is forbidden), the tag table and the job file.
Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_run_dumps_obd.py"""
import hashlib
import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, HERE)
import run_dumps_obd as D  # noqa: E402


def bank(tmp_path, tag, commit=D.COMMIT, man_tag=None, bytes_=b"fake-ckpt", sha=None, manifest=True, ckpt=True):
    b = tmp_path / f"obd_{tag}"
    b.mkdir(parents=True, exist_ok=True)
    if ckpt:
        (b / "ckpt_last.pt").write_bytes(bytes_)
    if manifest:
        m = {"tag": man_tag or tag, "commit": commit, "files": {"ckpt_last.pt": {"sha256": sha or hashlib.sha256(bytes_).hexdigest(), "bytes": len(bytes_)}}}
        (b / "BANK_MANIFEST.json").write_text(json.dumps(m))
    return b


@pytest.fixture
def arms(tmp_path, monkeypatch):
    for t in D.OBD_TAGS:
        monkeypatch.setitem(D.ARMS, t, (str(tmp_path / f"obd_{t}" / "ckpt_last.pt"), ("bank", str(tmp_path / f"obd_{t}" / "BANK_MANIFEST.json"))))
    monkeypatch.setattr(D.RD.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a refused arm must launch nothing")))
    return tmp_path


def test_tag_table_covers_the_cards_arms_and_joins_the_otband_table():
    assert set(D.OBD_TAGS) == {"OBD-NULL_s0", "OBD-NULL_s1", "OBD-NULL_rep", "OBD-CTL_s0", "OBD-CTL_s1", "OBD-BAND_s0", "OBD-BAND_s1", "OBD-BAND_rep"}
    assert D.OBD_TAGS[:7] == ("OBD-NULL_s0", "OBD-NULL_rep", "OBD-CTL_s0", "OBD-BAND_s0", "OBD-CTL_s1", "OBD-BAND_s1", "OBD-BAND_rep")      # the chain's order; NULL_s1 conditional, last
    assert all(t in D.RD.ARMS for t in D.OBD_TAGS) and "OB-CTL_s0" in D.RD.ARMS and "PARITY_DAGOACC_s0" in D.RD.ARMS     # extended, not replaced
    assert D.RD.OUT == D.OUT and D.RD.TREE == D.TREE
    for t in D.OBD_TAGS:
        assert D.ARMS[t][0].endswith(f"obd_{t}/ckpt_last.pt") and D.ARMS[t][1][0] == "bank"


def test_a_good_bank_passes(arms):
    bank(arms, "OBD-CTL_s0")
    assert D.refusal("OBD-CTL_s0") is None


def test_refusals(arms):
    assert "unknown tag" in D.refusal("OBD-FOO_s0")
    assert "checkpoint missing" in D.refusal("OBD-CTL_s0")                                  # nothing banked
    bank(arms, "OBD-CTL_s0", manifest=False)
    assert "BANK_MANIFEST.json missing" in D.refusal("OBD-CTL_s0")
    bank(arms, "OBD-CTL_s1", sha="f" * 64)
    assert "sha256 differs" in D.refusal("OBD-CTL_s1")
    bank(arms, "OBD-BAND_s0", commit="202849da013b5d74c0b0086dd9865fcda6836018")
    assert "code of record" in D.refusal("OBD-BAND_s0")
    bank(arms, "OBD-BAND_s1", man_tag="OBD-BAND_s0")
    assert "names tag" in D.refusal("OBD-BAND_s1")
    b = bank(arms, "OBD-NULL_s0")
    (b / "BANK_MANIFEST.json").write_text(json.dumps({"tag": "OBD-NULL_s0", "commit": D.COMMIT, "files": {}}))
    assert "no ckpt_last.pt sha256" in D.refusal("OBD-NULL_s0")


def test_run_returns_a_refusal_and_launches_nothing(arms):
    r = D.run("OBD-BAND_rep")
    assert r["status"].startswith("REFUSED") and "rc" not in r


def test_every_good_bank_passes_for_all_eight_tags(arms):
    for t in D.OBD_TAGS:
        bank(arms, t)
        assert D.refusal(t) is None, t


def test_tree_pins_checked_against_the_otband_pins(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "TREE", str(tmp_path / "nothing"))
    assert "tree file missing" in D.tree_problem()
    if os.path.isdir("F:/TanitAD-scratch/otband_read/tree"):
        monkeypatch.setattr(D, "TREE", "F:/TanitAD-scratch/otband_read/tree")
        assert D.tree_problem() is None


def test_job_file_lists_the_seven_planned_arms_in_order():
    t = open(os.path.join(HERE, "job_dumps.cmd"), encoding="utf-8").read()
    m = re.search(r"run_dumps_obd\.py ([^>\r\n]+) >>", t)
    assert tuple(m.group(1).split()) == D.OBD_TAGS[:7]
    assert "OBD-NULL_s1" not in t                                                            # conditional: launched by hand once NULL_PARITY.json says so
    assert "obd\\read" in t and "-I" in t
