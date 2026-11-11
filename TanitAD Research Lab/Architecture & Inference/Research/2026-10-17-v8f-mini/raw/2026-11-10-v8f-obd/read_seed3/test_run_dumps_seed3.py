"""Tests of run_dumps_seed3.py and seed3_common.py: planted fake banks (no checkpoint is loaded, no model run, subprocess is forbidden), the tag table, the argv check, the skip of
finished arms, and the job file.  Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_run_dumps_seed3.py"""
import hashlib
import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, HERE)
import run_dumps_seed3 as S  # noqa: E402
import seed3_common as C  # noqa: E402

BASE_ARGV = ("--arm M2b --seed {seed} --grid pool4x10 --out D:\\tanitad\\runs\\v8f_obd\\{tag} --steps 3000 --lr 1e-4 --dagger-bank-sha256 197a68bb --dagger-seed 0 "
             "--detach-upper --w-op 0.3333333 --w-tac-traj 0.6666667")


def bank(tmp_path, tag, commit=S.COMMIT, man_tag=None, bytes_=b"fake-ckpt", sha=None, manifest=True, ckpt=True, argv=True, seed=None):
    b = tmp_path / f"obd_{tag}"
    b.mkdir(parents=True, exist_ok=True)
    if ckpt:
        (b / "ckpt_last.pt").write_bytes(bytes_)
    if manifest:
        (b / "BANK_MANIFEST.json").write_text(json.dumps({"tag": man_tag or tag, "commit": commit, "files": {"ckpt_last.pt": {"sha256": sha or hashlib.sha256(bytes_).hexdigest()}}}))
    if argv is True:
        sd = seed if seed is not None else (C.NEW[tag][1] if tag in C.NEW else int(tag.split("_s")[1]))
        (b / "argv.txt").write_text(BASE_ARGV.format(seed=sd, tag=tag) + "\n")
    elif argv:
        (b / "argv.txt").write_text(argv)
    return b


def good_pair(tmp_path, tag):
    bank(tmp_path, C.NEW[tag][0])        # the banked base arm
    return bank(tmp_path, tag)


@pytest.fixture
def world(tmp_path, monkeypatch):
    for t in C.NEW_TAGS:
        monkeypatch.setitem(S.D.ARMS, t, (str(tmp_path / f"obd_{t}" / "ckpt_last.pt"), ("bank", str(tmp_path / f"obd_{t}" / "BANK_MANIFEST.json"))))
        monkeypatch.setitem(S.RD.ARMS, t, S.D.ARMS[t])
    monkeypatch.setattr(S, "PULL", str(tmp_path))
    monkeypatch.setattr(S.RD, "OUT", str(tmp_path / "t1"))
    (tmp_path / "t1").mkdir()
    monkeypatch.setattr(S.RD.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a refused or finished arm must launch nothing")))
    return tmp_path


def test_tag_table_is_the_amendments_four_arms_and_extends_the_tables():
    assert S.NEW_TAGS == ("OBD-CTL_s10", "OBD-BAND_s10", "OBD-CTL_s11", "OBD-BAND_s11")
    assert C.NEW == {"OBD-CTL_s10": ("OBD-CTL_s0", 10, 0), "OBD-BAND_s10": ("OBD-BAND_s0", 10, 0), "OBD-CTL_s11": ("OBD-CTL_s1", 11, 1), "OBD-BAND_s11": ("OBD-BAND_s1", 11, 1)}
    assert all(t in S.RD.ARMS and t in S.D.ARMS for t in S.NEW_TAGS)
    assert "OBD-CTL_s0" in S.RD.ARMS and "PARITY_DAGOACC_s0" in S.RD.ARMS and "OB-CTL_s0" in S.RD.ARMS          # extended, not replaced
    assert S.RD.TREE == S.TREE and S.RD.OUT == S.OUT or os.environ.get("OBD_DUMPS")
    for t in S.NEW_TAGS:
        assert S.D.ARMS[t][0].endswith(f"obd_{t}/ckpt_last.pt") and S.D.ARMS[t][1][0] == "bank"
    assert S.COMMIT == "cdbff971fa6dad4d20b937fe8abda6cb9f622adb"


def test_init_of_tag_reads_s11_as_init_one():
    assert [C.init_of_tag(t) for t in S.NEW_TAGS] == [0, 0, 1, 1]
    assert not "OBD-CTL_s11".endswith("_s1") and C.init_of_tag("OBD-CTL_s11") == 1


def test_a_good_bank_passes(world):
    for t in S.NEW_TAGS:
        good_pair(world, t)
        assert S.refusal(t) is None, t


def test_refusals_from_the_existing_driver_apply(world):
    assert "unknown tag" in S.refusal("OBD-CTL_s0")
    assert "checkpoint missing" in S.refusal("OBD-CTL_s10")
    good_pair(world, "OBD-CTL_s10")
    (world / "obd_OBD-CTL_s10" / "BANK_MANIFEST.json").unlink()
    assert "BANK_MANIFEST.json missing" in S.refusal("OBD-CTL_s10")
    good_pair(world, "OBD-BAND_s10")
    bank(world, "OBD-BAND_s10", sha="f" * 64)
    assert "sha256 differs" in S.refusal("OBD-BAND_s10")
    good_pair(world, "OBD-CTL_s11")
    bank(world, "OBD-CTL_s11", commit="202849da013b5d74c0b0086dd9865fcda6836018")
    assert "code of record" in S.refusal("OBD-CTL_s11")
    good_pair(world, "OBD-BAND_s11")
    bank(world, "OBD-BAND_s11", man_tag="OBD-BAND_s10")
    assert "names tag" in S.refusal("OBD-BAND_s11")


def test_argv_refusals(world):
    good_pair(world, "OBD-CTL_s10")
    assert S.refusal("OBD-CTL_s10") is None
    bank(world, "OBD-CTL_s10", argv=False)
    (world / "obd_OBD-CTL_s10" / "argv.txt").unlink()
    assert "argv.txt missing" in S.refusal("OBD-CTL_s10")
    bank(world, "OBD-CTL_s10", seed=0)                                                  # the base seed was not changed
    assert "--seed 10" in S.refusal("OBD-CTL_s10")
    bank(world, "OBD-CTL_s10", argv=BASE_ARGV.format(seed=10, tag="OBD-CTL_s10").replace("--lr 1e-4", "--lr 2e-4"))     # a second lever moved
    assert "more than --seed and --out" in S.refusal("OBD-CTL_s10")
    bank(world, "OBD-CTL_s10", argv=BASE_ARGV.format(seed=10, tag="elsewhere"))        # only --out differs: allowed
    assert S.refusal("OBD-CTL_s10") is None
    (world / "obd_OBD-CTL_s0" / "argv.txt").unlink()
    assert "argv.txt missing" in S.refusal("OBD-CTL_s10")


def test_run_returns_a_refusal_and_launches_nothing(world):
    r = S.run("OBD-BAND_s11")
    assert r["status"].startswith("REFUSED") and "rc" not in r
    good_pair(world, "OBD-BAND_s11")
    bank(world, "OBD-BAND_s11", seed=1)
    assert S.run("OBD-BAND_s11")["status"].startswith("REFUSED: ")


def test_a_finished_arm_is_skipped_before_any_check(world):
    (world / "t1" / "done_OBD-CTL_s10_m10_causal_heads.txt").write_text("{}")
    r = S.run("OBD-CTL_s10")                       # no bank exists: skipped, not refused
    assert r["status"].startswith("skip")


def test_job_file_lists_the_four_arms_in_order_and_writes_its_marker():
    t = open(os.path.join(HERE, "job_dumps_seed3.cmd"), encoding="utf-8").read()
    m = re.search(r"run_dumps_seed3\.py ([^>\r\n]+) >>", t)
    assert tuple(m.group(1).split()) == S.NEW_TAGS
    assert "obd\\read_seed3" in t and "-I" in t and "DUMPS_SEED3_RUN.done" in t and "set \"RC=%ERRORLEVEL%\"" in t


def test_launch_script_uses_the_wmi_pattern():
    t = open(os.path.join(HERE, "launch_wmi.ps1"), encoding="utf-8").read()
    assert "Win32_Process" in t and "< NUL" in t and "0x08000200" in t and "job_dumps_seed3.cmd" in t and "obd\\read_seed3" in t
