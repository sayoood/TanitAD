"""LEAD-VIOL must SURVIVE the refcv3 analysis path (SPEC_REFCV8_DRAFT section 8.7, X7).

WHAT BROKE (MEASURED 2026-10-11 by the r7base baseline agent). `taniteval/tools/refcv3_arm.py::
_distance_keeping` REPLACED `four_families.longitudinal.distance_keeping` with a block built from
`lead_metrics.distance_keeping`. `lead_metrics.lead_viol` is wired only in
`four_families._distance_keeping`, which that path overrides, so a refcv3 analysis (the refcv8
battery path) NEVER carried LEAD-VIOL and nothing failed. The fix computes LEAD-VIOL on the same
common-grid join and MERGES the block into the existing one.

THE FIXTURE is the hand-derived PR table of `test_lead_viol.py` (K = 4, dt = 0.5 s, v0 = 10 m/s,
stationary lead, human drives x = 5,10,15,20): 6 lead windows in 2 episodes, 5 at-risk, 1 violation
-> share_of_at_risk = 0.2, progress ratio mean(1.0, 0.8, 1.0, 0.6, 1.6) = 1.0; the STOP arm reads
progress 0. It goes through the REAL B1 block loader, the REAL common-grid selection and the REAL
`join_lead_block` (clip/frame join + the label-free speed proof) - only the model rollout is absent.

THE DELIBERATE REGRESSIONS re-introduce the old behaviour at the two seams the fix touches:
the REPLACE (`_merge_distance_keeping` -> `dict(new)`) and the dropped computation
(`_arm_lead_viol` -> nothing). Each must turn the literals RED.

ASCII-ONLY BY CHOICE (assertion text reaches a cp1252 console on this dev box).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "refcv3_arm.py"
K, DT, V0 = 4, 0.5, 10.0
NAN = float("nan")
SENTINEL = "kept-from-the-family-block"


def _load_tool():
    if not TOOL.exists():
        pytest.skip(f"tool not present: {TOOL}")
    try:
        spec = importlib.util.spec_from_file_location("_refcv3_arm_leadviol_ut", TOOL)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_refcv3_arm_leadviol_ut"] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as ex:                       # pragma: no cover
        pytest.skip(f"refcv3_arm is not importable here: {ex!r}")


def _xs(vals):
    return np.stack([np.asarray(vals, dtype=float), np.zeros(len(vals))], axis=1)


H4 = [5, 10, 15, 20]
PLANS = np.stack([_xs(H4), _xs([4, 8, 12, 16]), _xs(H4), _xs([3, 6, 9, 12]),
                  _xs([8, 16, 24, 32]), _xs([0, 0, 0, 0])])
HUMANS = np.stack([_xs(H4)] * 6)
LEADS = np.stack([np.stack([np.full(K, 40.0), np.zeros(K)], axis=1)] * 5
                 + [np.stack([np.full(K, 25.0), np.zeros(K)], axis=1)])
EID_W = ["ep000"] * 3 + ["ep001"] * 3


def _fixture(tmp_path):
    from taniteval.four_families import all_families
    files = []
    for fi in range(2):
        f = tmp_path / f"ep{fi:03d}.npz"
        np.savez(f, ws=np.arange(3), v0=np.full(3, V0, dtype=np.float32))
        files.append(str(f))
    manifest = {"episodes": [{"file_index": 0, "clip_id": "clipA", "name": "ep000"},
                             {"file_index": 1, "clip_id": "clipB", "name": "ep001"}]}
    block = tmp_path / "lead_block.npz"
    np.savez(block,
             clip_id=np.array(["clipA"] * 3 + ["clipB"] * 3),
             frame=np.array([0, 1, 2, 0, 1, 2]),
             ts_rel_s=np.array([0.5, 1.0, 1.5, 2.0]),
             leads=LEADS.copy(), lead_lens=np.full(6, 4.0), speeds=np.full(6, V0),
             state=np.array(["LEAD"] * 6), eid=np.array(["ep000"] * 3 + ["ep001"] * 3))
    P_cat = {"os": PLANS.copy(), "stop": np.zeros_like(PLANS)}
    rec = {"arms": {}}
    for arm, plans in P_cat.items():
        fam = all_families({"pred": plans.copy(), "gt": HUMANS.copy(),
                            "wp_steps": list(range(1, K + 1)), "dt_s": DT})
        dk = fam["longitudinal"]["distance_keeping"]
        assert dk["status"] == "UNAVAILABLE"                      # the family block before the patch
        dk["_sentinel"] = SENTINEL
        rec["arms"][arm] = {"four_families": fam}
    return rec, files, manifest, str(block), P_cat


def _run(mod, tmp_path):
    rec, files, manifest, block, P_cat = _fixture(tmp_path)
    out = mod._distance_keeping(
        rec, files, manifest, block, ["os", "stop"], P_cat, HUMANS.copy(),
        [("stop", "os", "paired_os_minus_stop")], EID_W, 200, 0,
        {"os": "T1", "stop": "T1"}, DT, K, 0)
    return rec, out


def _check(mod, tmp_path):
    """The literals. Green on the real tool; RED under each deliberate regression."""
    rec, out = _run(mod, tmp_path)
    assert out["status"] == "PRESENT", out.get("reason")
    for arm in ("os", "stop"):
        dk = rec["arms"][arm]["four_families"]["longitudinal"]["distance_keeping"]
        # (1) MERGED, not replaced: the family block's own key survives ...
        assert dk.get("_sentinel") == SENTINEL, sorted(dk)
        # ... and the common-grid values replaced its UNAVAILABLE status
        assert dk["status"] == "OK" and dk["n"] == 6, (dk["status"], dk.get("n"))
        # (2) LEAD-VIOL is PRESENT in the record the battery reads
        assert "lead_viol" in dk, sorted(dk)
        assert "lead_viol" in out["per_arm"][arm]
    lv = rec["arms"]["os"]["four_families"]["longitudinal"]["distance_keeping"]["lead_viol"]
    assert lv["n_lead_windows"] == 6 and lv["n_at_risk"] == 5 and lv["n_viol"] == 1, lv
    assert lv["share_of_at_risk"] == pytest.approx(0.2)
    assert lv["share_of_at_risk_ci"]["n_episodes"] == 2           # w0-w2 | w3-w4  (w5 is not at risk)
    assert lv["lead_progress_ratio"] == pytest.approx(1.0, abs=1e-12)
    # the decision reads the CLIPPED mean (MM 2026-10-11): this fixture overshoots one window (1.6) and
    # crawls two (0.8, 0.6), so (1 + .8 + 1 + .6 + 1) / 5 = 0.88 < 0.90 -> CRAWL, whatever the 1.0 mean says
    assert lv["lead_progress_ratio_clipped"] == pytest.approx(0.88)
    assert lv["verdict"]["verdict"] == "INADMISSIBLE-CRAWL"
    st = rec["arms"]["stop"]["four_families"]["longitudinal"]["distance_keeping"]["lead_viol"]
    assert st["n_viol"] == 0 and st["lead_progress_ratio"] == 0.0
    assert st["verdict"]["verdict"] == "INADMISSIBLE-CRAWL"
    # (3) the paired read, with the at-risk bar key
    pr = out["paired_lead_viol"]["paired_os_minus_stop"]
    assert pr["at_risk"]["n_at_risk"] == 5 and pr["at_risk"]["delta"] == pytest.approx(0.2)
    assert pr["x7_direction_met_at_risk"] is False                # os violates more than STOP
    return rec, out


def test_lead_viol_survives_the_refcv3_analysis_path(tmp_path):
    mod = _load_tool()
    rec, out = _check(mod, tmp_path)
    json.dumps(out["per_arm"], default=str)
    # the interval-coverage audit of the patched family block stays complete and flags nothing new
    cov = rec["arms"]["os"]["four_families"]["_ci_coverage"]["longitudinal"]
    assert not [u for u in cov.get("undeclared", []) if "lead_viol" in u], cov.get("undeclared")
    # no numpy arrays leaked into the record
    lv = rec["arms"]["os"]["four_families"]["longitudinal"]["distance_keeping"]["lead_viol"]
    assert "_per_window" not in lv
    json.dumps(lv)


def test_mutation_the_old_replace_goes_red(tmp_path, monkeypatch):
    """The literal old behaviour: the common-grid block REPLACES the family block."""
    mod = _load_tool()
    monkeypatch.setattr(mod, "_merge_distance_keeping", lambda old, new: dict(new))
    with pytest.raises(AssertionError):
        _check(mod, tmp_path)


def test_mutation_lead_viol_never_computed_goes_red(tmp_path, monkeypatch):
    """The old path computed no LEAD-VIOL at all (it is wired only in four_families)."""
    mod = _load_tool()
    monkeypatch.setattr(mod, "_arm_lead_viol", lambda *a, **k: (None, None))
    with pytest.raises(AssertionError):
        _check(mod, tmp_path)


def _call(mod, tmp_path, pairs):
    rec, files, manifest, block, P_cat = _fixture(tmp_path)
    out = mod._distance_keeping(rec, files, manifest, block, ["os", "stop"], P_cat, HUMANS.copy(),
                                pairs, EID_W, 50, 0, {"os": "T1", "stop": "T1"}, DT, K, 0)
    return rec, out


def test_a_refused_lead_viol_is_recorded_not_dropped(tmp_path, monkeypatch):
    """A ValueError (inputs inconsistent) is an honest refusal: it is RECORDED with its reason and
    the rest of the analysis survives - never silently absent."""
    mod = _load_tool()
    from taniteval import lead_metrics as lm

    def bad_value(*a, **k):
        raise ValueError("eid has 5 entries for 6 windows")
    monkeypatch.setattr(lm, "lead_viol", bad_value)
    rec, out = _call(mod, tmp_path, [("stop", "os", "paired_os_minus_stop")])
    lv = rec["arms"]["os"]["four_families"]["longitudinal"]["distance_keeping"]["lead_viol"]
    assert lv["status"] == "REFUSED" and "eid has 5 entries" in lv["reason"]
    assert out["paired_lead_viol"]["paired_os_minus_stop"]["status"] == "REFUSED"
    assert out["status"] == "PRESENT"                              # the other metrics are intact


def test_a_defect_in_lead_viol_propagates_instead_of_becoming_a_refusal(tmp_path, monkeypatch):
    mod = _load_tool()
    from taniteval import lead_metrics as lm

    def defect(*a, **k):
        raise TypeError("a bug in our code")
    monkeypatch.setattr(lm, "lead_viol", defect)
    with pytest.raises(TypeError):
        _call(mod, tmp_path, [])
