"""Tests of code/clr_wiring.py (the O-sensitivity of CLR -> the wiring of LLR, PREREG_LEVELHEAD_A2.md) on constructed readouts and on A1's real CL read-out (both seeds O-INSENSITIVE).
Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -q tests/test_clr_wiring.py --basetemp F:/TanitAD-scratch/motion_c/clr_read/pt/basetemp_wiring     (env CLR_WIRING_PATH aims the mutation arms at a copy)"""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
A1_READOUT = "E:/Projects/TanitAD-artifacts/dagland/motion_c/cl_readout/raw/CL_READOUT.json"


def _load(path):
    spec = importlib.util.spec_from_file_location("clr_wiring_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _load(os.environ.get("CLR_WIRING_PATH", f"{PK}/code/clr_wiring.py"))


def readout(change0, change1, label0=None, label1=None, fail=False):
    def seed(ch, lab):
        return {"labels": {"sensitivity": lab or W.sens_label(abs(ch)), "tracking": "PARTIAL"}, "control_failures": ["x"] if fail else [],
                "tracking": {"r2_all": {"point": 0.78}, "band_mae": {"point": 1.9}},
                "sensitivity": {"plus3": {"mean_signed_change_ms": {"point": ch}, "mean_abs_change_ms": {"point": abs(ch)}}, "label_alt_mean_abs_reading": W.sens_label(abs(ch)),
                                "e_level_over_ego_norm": {"mean_of_per_window_ratio": {"point": 0.004}}}}
    return {"per_seed": {"s0": seed(change0, label0), "s1": seed(change1, label1)}}


def run(tmp_path, d):
    p = tmp_path / "ro.json"
    json.dump(d, open(p, "w", encoding="utf-8"))
    rc = W.main(["--readout", str(p), "--raw", str(tmp_path), "--md", str(tmp_path / "W.md")])
    return rc, json.load(open(tmp_path / "CLR_WIRING.json", encoding="utf-8"))


def test_thresholds_are_A1s_and_inclusive_at_the_sensitive_edge():
    assert (W.SENS_HI, W.SENS_LO) == (0.5, 0.1)
    assert W.sens_label(0.5) == "O-SENSITIVE" and W.sens_label(0.4999) == "WEAK"
    assert W.sens_label(0.1) == "WEAK" and W.sens_label(0.0999) == "O-INSENSITIVE"


def test_wiring_for_both_seeds_sensitive_and_insensitive():
    a = W.wiring({"s0": "O-SENSITIVE", "s1": "O-SENSITIVE"})
    assert a["decision"] == "O-SENSITIVE" and "e_lead added to the ego token" in a["wiring"] and "zero-init" in a["wiring"]
    b = W.wiring({"s0": "O-INSENSITIVE", "s1": "O-INSENSITIVE"})
    assert b["decision"] == "O-INSENSITIVE" and "separate conditioning token" in b["wiring"] and "--aux-cond token" in b["wiring"] and "non-zero init" in b["wiring"]


@pytest.mark.parametrize("labels", [{"s0": "O-SENSITIVE", "s1": "O-INSENSITIVE"}, {"s0": "WEAK", "s1": "WEAK"}, {"s0": "O-SENSITIVE", "s1": "WEAK"}, {"s0": "O-INSENSITIVE"}])
def test_a_mixed_or_weak_reading_is_not_a_wiring_the_card_names(labels):
    a = W.wiring(labels)
    assert a["decision"] == "NOT-NAMED-BY-THE-CARD" and a["wiring"] is None and "Master Mind decides" in a["note"]


def test_main_on_constructed_readouts(tmp_path):
    rc, r = run(tmp_path, readout(0.8, -0.6))
    assert rc == 0 and r["WIRING"]["decision"] == "O-SENSITIVE" and r["controls_ok"] == {"s0": True, "s1": True}
    assert "Decision: O-SENSITIVE" in open(tmp_path / "W.md", encoding="utf-8").read()
    rc, r = run(tmp_path, readout(0.0004, -0.0005))
    assert rc == 0 and r["WIRING"]["decision"] == "O-INSENSITIVE"
    rc, r = run(tmp_path, readout(0.8, 0.0004))
    assert rc == 0 and r["WIRING"]["decision"] == "NOT-NAMED-BY-THE-CARD"


def test_main_refuses_when_the_readouts_own_controls_failed_or_its_label_disagrees(tmp_path):
    rc, r = run(tmp_path, readout(0.8, 0.8, fail=True))
    assert rc == 3 and r["WIRING"]["decision"] == "REFUSED"
    rc, r = run(tmp_path, readout(0.8, 0.8, label0="O-INSENSITIVE"))            # the stored label contradicts the recomputed one
    assert rc == 3 and r["WIRING"]["decision"] == "REFUSED"


def test_a1s_real_cl_readout_reads_insensitive_at_both_seeds_and_the_md_names_the_wiring(tmp_path):
    rc, r = run(tmp_path, json.load(open(A1_READOUT, encoding="utf-8")))
    assert rc == 0 and r["WIRING"]["decision"] == "O-INSENSITIVE"
    assert {k: v["sensitivity_label_A1"] for k, v in r["per_seed"].items()} == {"s0": "O-INSENSITIVE", "s1": "O-INSENSITIVE"}
    md = open(tmp_path / "W.md", encoding="utf-8").read().lower()
    for w in ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn", "closed loop"):
        assert w not in md, w
