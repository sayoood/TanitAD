"""The dose verdict's `--arm K2DC02s1` (MM 4a3d05, 2026-11-20): the PREREG_K2DC02S1_2026-11-20 replicate. CPU only, no scipy, no GPU, no ssh.

* K2DC02s1 reads its dose settings from stack/ops/runs.d/r9e2_K2DC02s1.argv.json (the same 0.02 target and 0.1,25 bounds as K2DC02: the replicate moves --seed only);
* its document carries ITS pre-registration sentence (PREREG_K2DC02S1, the section 4 floor named as a noise reading);
* K2DC02's sentence is byte-identical to the text the tool carried before this arm existed (the K2DC02 golden sha256 in test_r9_e2_dose_verdict_k2sg.py pins its whole document);
* an arm that is not listed is still refused.
Mutant (must turn a test RED): the K2DC02s1 entry removed from ARM_ARGV.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOSE = ROOT / "scripts" / "r9_e2_dose_verdict.py"
ARGVS1 = ROOT / "ops" / "runs.d" / "r9e2_K2DC02s1.argv.json"
K2DC02_NOTE = ("Pre-registered for K2DC02 in e2/e2raw/PREREG_K2DC02_2026-11-18.md (landed 75d9ca52, before any K2DC02 training step): K2DC is the comparator "
               "in K2's slot (--k2-read = E2_READ_K2DC's read); `reason` text naming next arms is K2DC's, and that PREREG's section 4 replaces it.")


@pytest.fixture()
def D():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("_dose_k2dc02s1", DOSE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _nums():
    return {s: {"dc_vs_k0": {"LAT_cross_mae_m": {"delta": 0.001, "lo": 0, "hi": 0.002, "separated": False}, "LAT_cross_abs_final_m": {"delta": 0.0, "lo": 0, "hi": 0, "separated": False}},
                "dc_vs_k2": {"LAT_cross_mae_m": {"delta": -0.004, "lo": -0.005, "hi": -0.003, "separated": True}, "LAT_cross_abs_final_m": {"delta": -0.004, "lo": -0.005, "hi": -0.003, "separated": True}},
                "jitter_dlag_dc_vs_k2": {"delta": -0.0005, "lo": -0.001, "hi": 0.0, "separated": False}} for s in ("0", "1")}


def test_k2dc02s1_reads_its_own_argv_json(D):
    argv = json.loads(ARGVS1.read_text(encoding="utf-8"))["argv"]
    a = D.arm_dose("K2DC02s1")
    assert a["arm"] == "K2DC02s1" and Path(a["argv_json"]).name == "r9e2_K2DC02s1.argv.json"
    assert a["target"] == float(argv[argv.index("--map-hires-share-target") + 1]) == 0.02
    assert a["bounds"] == [0.1, 25.0] and a["share_band"] == [0.01, 0.03]
    assert argv[argv.index("--seed") + 1] == "1"


def test_k2dc02s1_document_carries_its_prereg_and_k2dc02_text_is_unchanged(D, tmp_path):
    assert D.ARM_PREREG_NOTE["K2DC02"] == K2DC02_NOTE
    assert sorted(D.ARM_PREREG_NOTE) == sorted(D.ARM_ARGV)                 # every derived arm has its sentence
    m = tmp_path / "m.jsonl"
    m.write_text("\n".join(json.dumps({"step": s, "dc_share_read": v, "dc_w_eff": 3.0}) for s, v in ((3000, 0.02), (3250, 0.025))) + "\n", encoding="utf-8")
    for arm, want in (("K2DC02s1", "PREREG_K2DC02S1_2026-11-20"), ("K2DC02", "PREREG_K2DC02_2026-11-18")):
        a = D.arm_dose(arm)
        doc = D.build(_nums(), D.void_check(m, tuple(a["share_band"]), tuple(a["bounds"])), None, a)
        assert doc["arm"]["arm"] == arm and doc["outcome"] == "SUPPORTED" and want in doc["arm"]["note"]
        assert doc["thresholds"]["share_band"] == [0.01, 0.03]
    s1 = D.build(_nums(), {"valid": True, "status": "HELD"}, None, D.arm_dose("K2DC02s1"))["arm"]["note"]
    assert "FLOOR" in s1 and "never a lever verdict" in s1 and "PREREG_K2DC02_2026-11-18" not in s1
    k2 = D.build(_nums(), {"valid": True, "status": "HELD"}, None, D.arm_dose("K2DC02"))["arm"]["note"]
    assert k2.endswith(K2DC02_NOTE)


def test_unlisted_arms_are_still_refused(D):
    for arm in ("K2DC02s2", "K2DC03", "k2dc02s1"):
        with pytest.raises(SystemExit):
            D.arm_dose(arm)


def test_MUTANT_without_the_entry_the_arm_is_refused(D, monkeypatch):
    monkeypatch.delitem(D.ARM_ARGV, "K2DC02s1")
    with pytest.raises(SystemExit):
        D.arm_dose("K2DC02s1")
