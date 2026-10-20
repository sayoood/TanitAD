"""Pins for pf_lane.py: the ONE replaced function (check_deploy_pf) and the re-pointing.  LITERAL expectations; mutations must go RED.

The clip-fed frozen table (battery package, sha256 b208c9c1...) is the known-value input: its 8 knots are copied here as LITERALS from the landed
deploy table (configs/refcv8_retime_deploy_gains.json) and test_deploy_gains_r8.py, never re-derived from the code under test.
    python -m pytest -q tests/test_pf_lane.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import pf_lane as PL  # noqa: E402

R8 = Path("C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery")
CLIPFED_FILE = R8 / "raw" / "retime_trainfit" / "FROZEN_GAINS_RETIME_train.json"
CLIPFED_SHA = "b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee"
KNOTS_LIT = [0.9673903790217738, 0.6192916835387219, 0.5511341845460037, 0.530539399347386, 0.6546139969520486, 0.6994940271226132, 0.7222699235551424,
             0.7453601677670005]
S6_TRAIN_LIT = [0.6393548580383511, 0.5969720822621201, 0.6546139969520486, 0.6994940271226132, 0.7222699235551424, 0.7453601677670005]


@pytest.fixture(scope="module")
def M():
    return PL.mods()


@pytest.fixture(scope="module")
def fz():
    assert hashlib.sha256(CLIPFED_FILE.read_bytes()).hexdigest() == CLIPFED_SHA
    return json.load(open(CLIPFED_FILE, encoding="utf-8"))


def other_table(fz):
    """A synthetic re-fit: every RETIME gain moved by a literal offset (what a per-frame re-fit looks like to this code)."""
    f = copy.deepcopy(fz)
    for s in ("S2", "S6"):
        f["surfaces"][s]["all"]["RETIME"] = [g + 0.0125 for g in f["surfaces"][s]["all"]["RETIME"]]
    return f


def test_knots_of_the_clip_fed_table_are_the_literal_deployed_knots(fz):
    assert PL.knots_of(fz) == KNOTS_LIT
    assert fz["surfaces"]["S6"]["all"]["RETIME"] == S6_TRAIN_LIT


def test_check_deploy_pf_agrees_with_the_package_check_on_the_clip_fed_table(M, fz):
    d = M["BD"].build(fz)
    assert M["BD"].check_deploy(d) == [] and PL.check_deploy_pf(d, fz) == []


def test_a_refit_table_passes_pf_check_and_fails_the_clip_fed_literals(M, fz):
    f2 = other_table(fz)
    d2 = M["BD"].build(f2)
    assert PL.check_deploy_pf(d2, f2) == []
    assert M["BD"].check_deploy(d2) != []                       # the package check carries the clip-fed literals: it would refuse ANY re-fit


@pytest.mark.parametrize("mut", ["gains_from_other_table", "knot_time", "s2_not_identity", "s6_substitution", "changed_instants"])
def test_mutation_check_deploy_pf_goes_red(M, fz, mut):
    d = M["BD"].build(fz)
    ref = fz
    if mut == "gains_from_other_table":
        ref = other_table(fz)                                     # deploy built from table A, checked against registered table B
    elif mut == "knot_time":
        d["knots"]["t_s"] = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.5]
    elif mut == "s2_not_identity":
        d["surfaces"]["S2"]["RETIME_DEPLOY"] = [g * 1.01 for g in d["surfaces"]["S2"]["RETIME_DEPLOY"]]
    elif mut == "s6_substitution":
        d["surfaces"]["S6"]["RETIME_DEPLOY"][0] = d["surfaces"]["S6"]["RETIME_TRAIN"][0]
    elif mut == "changed_instants":
        d["surfaces"]["S6"]["changed_vs_train"] = d["surfaces"]["S6"]["changed_vs_train"][:1]
    assert PL.check_deploy_pf(d, ref) != []


def test_configure_repoints_every_output_literally(M, tmp_path):
    cfg = PL.configure(M, tmp_path, "a" * 64)
    raw = tmp_path / "raw"
    assert Path(M["CP"].OUT) == raw / "clamp_posthoc" and Path(M["CR"].OUT) == raw / "compliant_reread"
    assert Path(M["RT"].BANKED).parent == raw and Path(M["TFM"].TF) == raw / "retime_trainfit"
    assert Path(M["TFM"].FROZEN_TRAIN) == raw / "retime_trainfit" / "FROZEN_GAINS_RETIME_train.json"
    assert Path(M["BD"].OUT) == raw / "retime_deploy" and M["BD"].SOURCE_SHA256 == "a" * 64 and M["D8"].FROZEN_SHA == "a" * 64
    assert Path(M["D8"].RS.DEPLOY_TABLE_PATH) == raw / "retime_deploy" / "refcv8_retime_deploy_gains_pf.json"
    assert Path(M["DF"].DEP) == raw / "retime_deploy" and Path(M["TAC"].DEP) == raw / "retime_deploy" and Path(M["TAC"].PKG) == tmp_path
    assert Path(M["PP"].DEP) == raw / "retime_deploy"
    assert str(M["CP"].ARCHIVE) == "C:/Users/Admin/refcv8_eval_archive/battery"        # the battery TREE is read where the package reads it
    assert cfg["registered_sha256"] == "a" * 64


def test_install_refuses_a_table_that_is_not_the_registered_one(tmp_path):
    reg = tmp_path / "REG.sha256"
    reg.write_text("0" * 64 + "\n")
    with pytest.raises(SystemExit):
        PL.cmd_install(tmp_path / "pkg", CLIPFED_FILE, reg)
    reg.write_text("\n")                                          # an empty registration is never a match
    with pytest.raises(SystemExit):
        PL.cmd_install(tmp_path / "pkg", CLIPFED_FILE, reg)
    assert not (tmp_path / "pkg" / "raw" / "retime_trainfit" / "FROZEN_GAINS_RETIME_train.json").exists()


def test_verify_installed_is_inconclusive_on_empty_strings(tmp_path):
    (tmp_path / "raw" / "retime_trainfit").mkdir(parents=True)
    with pytest.raises(SystemExit) as e:
        PL.verify_installed(tmp_path, "")
    assert "INCONCLUSIVE" in str(e.value)


@pytest.mark.parametrize("raw,want", [(b"\xef\xbb\xbf" + b"ab" * 32 + b"\r\n", "ab" * 32), (("AB" * 32 + "  x.json\n").encode(), "ab" * 32),
                                      (b"ab" * 31 + b"a\n", ""), (b"", "")])
def test_read_registered_forms(tmp_path, raw, want):
    f = tmp_path / "REG.sha256"
    f.write_bytes(raw)
    assert PL.read_registered(f) == want
