"""The dose verdict's `--arm K2DC02cl` (PREREG_K2DC02CL_2026-11-22): CPU only, no scipy, no GPU, no ssh.

* K2DC02cl reads its dose settings from stack/ops/runs.d/r9e2_K2DC02cl.argv.json (target 0.02 as K2DC02; bounds 2.5,3.5; the DOSE band stays 0.5x..1.5x the target = [0.01, 0.03],
  PREREG_K2DC02 section 2's rule UNCHANGED);
* NEW, K2DC02cl only: the DOSE-MEAN check. The mean of the probe shares at step >= 3000 must lie in [0.01, 0.035] else the outcome reads DOSE-OFF-TARGET (no claim about the
  0.02 dose). A run that already fails the >= 2-probe rule stays DOSE-VOID. K2DC02 and K2DC02s1 documents get NO mean keys (their documents are unchanged);
* its document carries ITS pre-registration sentence; K2DC02's / K2DC02s1's sentences and the byte-golden of K2DC02's document (test_r9_e2_dose_verdict_k2sg.py) are untouched;
* an arm that is not listed is still refused.
Mutants (each must turn a test RED): the K2DC02cl entry removed from ARM_ARGV; the mean band removed from ARM_MEAN_BAND.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOSE = ROOT / "scripts" / "r9_e2_dose_verdict.py"
ARGVCL = ROOT / "ops" / "runs.d" / "r9e2_K2DC02cl.argv.json"


@pytest.fixture()
def D():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("_dose_k2dc02cl", DOSE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _nums():
    fam = {f: {} for f in ("LONGITUDINAL", "LATERAL", "TACTICAL")}
    return {s: {"dc_vs_k0": {"LAT_cross_mae_m": {"delta": 0.001, "lo": 0, "hi": 0.002, "separated": False}, "LAT_cross_abs_final_m": {"delta": 0.0, "lo": 0, "hi": 0, "separated": False}},
                "dc_vs_k2": {"LAT_cross_mae_m": {"delta": -0.004, "lo": -0.005, "hi": -0.003, "separated": True}, "LAT_cross_abs_final_m": {"delta": -0.004, "lo": -0.005, "hi": -0.003, "separated": True}},
                "jitter_dlag_dc_vs_k2": {"delta": -0.0005, "lo": -0.001, "hi": 0.0, "separated": False},
                "dc_vs_k2_by_family": fam, "dc_vs_k0_by_family": fam} for s in ("0", "1")}


def _metrics(tmp_path, name, rows):
    m = tmp_path / name
    m.write_text("\n".join(json.dumps({"step": s, "dc_share_read": v, "dc_w_eff": 3.0}) for s, v in rows) + "\n", encoding="utf-8")
    return m


def _doc(D, arm, m):
    a = D.arm_dose(arm)
    mb = D.ARM_MEAN_BAND.get(arm)
    return D.build(_nums(), D.void_check(m, tuple(a["share_band"]), tuple(a["bounds"]), mb), None, a)


def test_k2dc02cl_reads_its_own_argv_json(D):
    argv = json.loads(ARGVCL.read_text(encoding="utf-8"))["argv"]
    a = D.arm_dose("K2DC02cl")
    assert a["arm"] == "K2DC02cl" and Path(a["argv_json"]).name == "r9e2_K2DC02cl.argv.json"
    assert a["target"] == float(argv[argv.index("--map-hires-share-target") + 1]) == 0.02
    assert a["bounds"] == [2.5, 3.5] and a["share_band"] == [0.01, 0.03]            # the DOSE rule band is PREREG_K2DC02 section 2's, unchanged
    assert argv[argv.index("--seed") + 1] == "0"
    assert D.ARM_MEAN_BAND == {"K2DC02cl": (0.01, 0.035)} and D.EXIT["DOSE-OFF-TARGET"] == 3


def test_document_carries_its_prereg_and_the_other_arms_sentences_are_unchanged(D, tmp_path):
    assert sorted(D.ARM_PREREG_NOTE) == sorted(D.ARM_ARGV) == ["K2DC02", "K2DC02cl", "K2DC02s1"]
    m = _metrics(tmp_path, "ok.jsonl", ((3000, 0.02), (3250, 0.025)))
    doc = _doc(D, "K2DC02cl", m)
    assert doc["arm"]["arm"] == "K2DC02cl" and doc["outcome"] == "SUPPORTED" and "PREREG_K2DC02CL_2026-11-22" in doc["arm"]["note"] and "DOSE-MEAN" in doc["arm"]["note"]
    assert doc["thresholds"]["share_band"] == [0.01, 0.03]
    for arm, want in (("K2DC02s1", "PREREG_K2DC02S1_2026-11-20"), ("K2DC02", "PREREG_K2DC02_2026-11-18")):
        assert want in _doc(D, arm, m)["arm"]["note"]
        assert "mean_band" not in _doc(D, arm, m)["dose_validity"]                    # no mean check, no new keys: those documents are unchanged


def test_dose_mean_inside_the_band_is_held(D, tmp_path):
    doc = _doc(D, "K2DC02cl", _metrics(tmp_path, "a.jsonl", ((3000, 0.02), (3250, 0.026), (3500, 0.015), (500, 0.9))))      # the step < 3000 probe never counts
    v = doc["dose_validity"]
    assert v["status"] == "HELD" and v["valid"] is True and v["mean_in_band"] is True and v["mean_band"] == [0.01, 0.035]
    assert v["n_mean_probes"] == 3 and abs(v["mean_share_step_ge_3000"] - 0.020333) < 1e-6
    assert doc["outcome"] == "SUPPORTED"


def test_dose_mean_outside_the_band_reads_off_target_even_with_two_probes_inside(D, tmp_path):
    rows = ((3000, 0.012), (3250, 0.012), (3500, 0.012), (3750, 0.09), (4000, 0.09), (4250, 0.09))                         # 3 probes inside [0.01, 0.03], mean 0.051
    doc = _doc(D, "K2DC02cl", _metrics(tmp_path, "b.jsonl", rows))
    v = doc["dose_validity"]
    assert v["n_inside_band"] == 3 and v["mean_in_band"] is False and v["status"] == "OFF-TARGET" and v["valid"] is False
    assert doc["outcome"] == "DOSE-OFF-TARGET" and doc["outcome_if_valid"] == "SUPPORTED"                                   # the three outcomes are still computed, never promoted
    assert D.EXIT[doc["outcome"].split(" ")[0]] == 3
    for arm in ("K2DC02", "K2DC02s1"):                                                                                     # the same log under the old arms: HELD, no mean rule
        assert _doc(D, arm, _metrics(tmp_path, f"b_{arm}.jsonl", rows))["outcome"] == "SUPPORTED"


def test_the_two_probe_rule_still_decides_dose_void_first(D, tmp_path):
    rows = ((3000, 0.034), (3250, 0.034), (3500, 0.034))                                                                   # mean 0.034 is inside the mean band, but no probe is inside [0.01, 0.03]
    doc = _doc(D, "K2DC02cl", _metrics(tmp_path, "c.jsonl", rows))
    assert doc["dose_validity"]["status"] == "VOID" and doc["dose_validity"]["mean_in_band"] is True and doc["outcome"] == "DOSE-VOID"


def test_no_probe_at_step_3000_or_later_is_void_not_off_target(D, tmp_path):
    doc = _doc(D, "K2DC02cl", _metrics(tmp_path, "d.jsonl", ((250, 0.02), (500, 0.02))))
    v = doc["dose_validity"]
    assert v["status"] == "VOID" and v["mean_share_step_ge_3000"] is None and v["mean_in_band"] is False and doc["outcome"] == "DOSE-VOID"


def test_markdown_renders_the_mean_line(D, tmp_path):
    doc = _doc(D, "K2DC02cl", _metrics(tmp_path, "e.jsonl", ((3000, 0.012), (3250, 0.012), (3500, 0.09), (3750, 0.09))))
    text = D.md(doc)
    assert "DOSE-OFF-TARGET" in text and "DOSE-MEAN: mean late share 0.051 vs mean band [0.01, 0.035]: OUTSIDE." in text
    assert "DOSE-MEAN" not in D.md(_doc(D, "K2DC02", _metrics(tmp_path, "f.jsonl", ((3000, 0.02), (3250, 0.02)))))


def test_an_unlisted_arm_is_still_refused(D):
    with pytest.raises(SystemExit):
        D.arm_dose("K2DC02zz")


def test_MUTANT_without_the_argv_entry_or_without_the_mean_band_the_checks_go_RED(D, tmp_path, monkeypatch):
    rows = ((3000, 0.012), (3250, 0.012), (3500, 0.012), (3750, 0.09), (4000, 0.09), (4250, 0.09))
    m = _metrics(tmp_path, "g.jsonl", rows)
    monkeypatch.setitem(D.ARM_MEAN_BAND, "K2DC02cl", None)                                                                  # the mean band removed: the off-target log slips through as SUPPORTED
    assert _doc(D, "K2DC02cl", m)["outcome"] == "SUPPORTED"                                                                # (the test above asserts DOSE-OFF-TARGET and would fail)
    monkeypatch.delitem(D.ARM_ARGV, "K2DC02cl")                                                                            # the arm entry removed: the arm is refused
    with pytest.raises(SystemExit):
        D.arm_dose("K2DC02cl")
