"""⛔ SPEC_REFCV7 §7 (A2, the PI's E1 ruling 2026-09-26): the nav-compliance τ is the BANKED
TRAIN-split derivation, and "its sha256 is recorded in config.json".

`--nav-compliance-tau-file <json>`: `train()` reads the file, REFUSES a missing or unreadable one
and a `--nav-compliance-tau-rad` that differs from its `tau` by more than 1e-12 (the launch gate's
own tolerance), and stamps `{path, sha256, tau}` into config.json (`seams.nav_compliance_tau_file`).
The pin records the path (a DECLARED `RefCV3Config` field -- G-HYG) and never OPENS the file: an
eval rebuild re-runs the pin from the recorded argv on boxes where the file does not exist
(`taniteval/tools/refcv3_arm.rebuild_config`), so the float stays what the model is built from and
the file is its verifier. G-DVB reads the BUILT decoder's tolerance against the file's τ.

Expectations are LITERALS: the banked τ below is copied from
`…/2026-09-26-declared-vs-built/raw/nav_compliance_tau_train.json`; a sha256 is recomputed with
hashlib from the bytes the test itself wrote (independent of the code under test).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refc_v3_train as T  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.train import declared_vs_built as dvb  # noqa: E402

TAU = 0.18063741505146028          # the banked τ (rad), a LITERAL
TAU_ARG = "0.18063741505146028"
NAVC = ["--graft-nav-compliance", "--nav-compliance-tau-rad", TAU_ARG]
SMOKE = ["--arm", "hier", "--smoke", "--synth-episodes", "2", "--steps", "2", "--batch", "2",
         "--device", "cpu", "--save-every", "100", "--log-every", "1"]


def _file(tmp_path, body=None, name="nav_compliance_tau_train.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps({"tau": TAU, "status": "OK"} if body is None else body),
                 encoding="utf-8")
    return p


def _args(*extra):
    return T.build_parser().parse_args(["--arm", "hier", "--out", "z", *extra])


def _pin(*extra):
    return T._pin_trainer_cfg(v3.refc_v3_smoke_config(True), _args(*extra))


# ---- the verifier (train() only) ------------------------------------------------------------ #
def test_a_MATCH_returns_path_sha256_tau(tmp_path):
    p = _file(tmp_path)
    got = T._verify_navc_tau_file(_args(*NAVC, "--nav-compliance-tau-file", str(p)))
    assert got == {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                   "tau": 0.18063741505146028}


def test_a_MISMATCH_REFUSES_at_the_gate_tolerance(tmp_path):
    p = _file(tmp_path)
    with pytest.raises(SystemExit, match="differs from the banked tau"):
        T._verify_navc_tau_file(_args("--graft-nav-compliance", "--nav-compliance-tau-rad",
                                      "0.1806", "--nav-compliance-tau-file", str(p)))
    # the tolerance is 1e-12, written as literals on both sides of it
    with pytest.raises(SystemExit, match="differs from the banked tau"):
        T._verify_navc_tau_file(_args("--graft-nav-compliance", "--nav-compliance-tau-rad",
                                      "0.180637415053460", "--nav-compliance-tau-file", str(p)))
    assert T._verify_navc_tau_file(_args(
        "--graft-nav-compliance", "--nav-compliance-tau-rad", "0.1806374150519603",
        "--nav-compliance-tau-file", str(p)))["tau"] == TAU        # |d| = 5e-13: admitted


def test_a_MISSING_file_REFUSES(tmp_path):
    with pytest.raises(SystemExit, match="no such file"):
        T._verify_navc_tau_file(_args(*NAVC, "--nav-compliance-tau-file",
                                      str(tmp_path / "absent.json")))


@pytest.mark.parametrize("body,needle", [("not json at all", "not a tau file"),
                                          ({"tau_rad": TAU}, "not a tau file"),
                                          ({"tau": -0.1}, "not a positive finite")])
def test_an_UNREADABLE_or_INVALID_file_REFUSES(tmp_path, body, needle):
    p = tmp_path / "bad.json"
    p.write_text(body if isinstance(body, str) else json.dumps(body), encoding="utf-8")
    with pytest.raises(SystemExit, match=needle):
        T._verify_navc_tau_file(_args(*NAVC, "--nav-compliance-tau-file", str(p)))


def test_no_file_flag_means_NO_verification_and_NO_stamp():
    assert T._verify_navc_tau_file(_args(*NAVC)) is None


# ---- the pin (never opens the file) --------------------------------------------------------- #
def test_the_pin_REFUSES_a_file_without_its_term_or_without_its_float(tmp_path):
    p = str(_file(tmp_path))
    with pytest.raises(SystemExit, match="without --graft-nav-compliance"):
        _pin("--nav-compliance-tau-file", p)
    with pytest.raises(SystemExit, match="needs --nav-compliance-tau-rad as well"):
        _pin("--graft-nav-compliance", "--nav-compliance-tau-file", p)


def test_an_EVAL_REBUILD_on_a_box_WITHOUT_the_file_still_builds_the_same_model(tmp_path):
    absent = str(tmp_path / "on_another_box" / "nav_compliance_tau_train.json")
    cfg = _pin(*NAVC, "--nav-compliance-tau-file", absent)
    assert cfg.core.graft_nav_compliance is True
    assert cfg.core.nav_compliance_tau_rad == 0.18063741505146028
    assert cfg.nav_compliance_tau_file == absent


def test_the_provenance_fields_are_DECLARED_so_G_HYG_admits_them():
    names = {f.name for f in dataclasses.fields(v3.RefCV3Config)}
    assert {"nav_compliance_tau_file", "nav_compliance_tau_sha256"} <= names
    cfg = v3.refc_v3_smoke_config(True)
    cfg.nav_compliance_tau_sha256 = "0" * 64                    # declared: no refusal
    assert dataclasses.replace(cfg).nav_compliance_tau_sha256 == "0" * 64   # survives a rebuild


# ---- through the REAL train() --------------------------------------------------------------- #
def test_config_json_STAMPS_path_sha256_tau_through_the_REAL_train(tmp_path):
    p = _file(tmp_path)
    out = tmp_path / "run"
    T.train(T.build_parser().parse_args(
        ["--out", str(out), *SMOKE, *NAVC, "--nav-compliance-tau-file", str(p)]))
    seams = json.loads((out / "config.json").read_text(encoding="utf-8"))["seams"]
    assert seams["nav_compliance_tau_file"] == {
        "path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "tau": 0.18063741505146028}
    assert seams["nav_compliance_tau_rad"] == 0.18063741505146028


def test_RED_a_mismatched_float_REFUSES_in_train_BEFORE_config_json(tmp_path):
    p = _file(tmp_path)
    out = tmp_path / "run_red"
    with pytest.raises(SystemExit, match="differs from the banked tau"):
        T.train(T.build_parser().parse_args(
            ["--out", str(out), *SMOKE, "--graft-nav-compliance", "--nav-compliance-tau-rad",
             "0.1806", "--nav-compliance-tau-file", str(p)]))
    assert not (out / "config.json").exists()


def test_a_run_WITHOUT_the_file_stamps_None():
    st = T._seam_stamp(_pin(*NAVC), _args(*NAVC))
    assert st["nav_compliance_tau_file"] is None


# ---- G-DVB: the BUILT tolerance against the file -------------------------------------------- #
def test_GDVB_reads_the_BUILT_tolerance_against_the_FILE(tmp_path):
    p = _file(tmp_path)
    a = _args(*NAVC, "--nav-compliance-tau-file", str(p))
    m = v3.RefCV3Model(_pin(*NAVC, "--nav-compliance-tau-file", str(p)))
    entry = dvb.REGISTRY["nav_compliance_tau_file"]
    assert entry.kind == "built"
    assert entry.check(m, a) == []
    # RED: the banked file says something else than the model was built with
    _file(tmp_path, body={"tau": 0.2})
    got = [(x.lever, x.declared, x.built, x.read_from) for x in entry.check(m, a)]
    assert got == [("--nav-compliance-tau-file", 0.2, 0.18063741505146028,
                    "core.decoder.navc_tau_rad")]
    # RED: the file is gone
    p.unlink()
    got = [(x.lever, x.why) for x in entry.check(m, a)]
    assert got == [("--nav-compliance-tau-file", "unreadable (FileNotFoundError)")]
