"""The pilot cannot escalate. Run: CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q tests/test_pilot_chain.py"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import pytest
import chain_gpu as G


def _fake(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "POUT", str(tmp_path / "pilot"))
    monkeypatch.setattr(G, "OUT", str(tmp_path))
    seen = {"clips": [], "cmds": []}

    def job_retry(stage, clips, out, budget=30, limit=0, max_wait_s=0):
        seen["clips"].append(clips)
        for c in json.load(open(clips))["clips"]:
            os.makedirs(f"{out}/ctx{stage}", exist_ok=True)
            open(f"{out}/ctx{stage}/{c}.npz", "w").write("x")
        return 0, "log"

    class R:
        returncode = 0
        stdout = ""
        stderr = ""

    def run(cmd, **kw):
        seen["cmds"].append(cmd)
        return R()
    monkeypatch.setattr(G, "job_retry", job_retry)
    monkeypatch.setattr(G.subprocess, "run", run)
    monkeypatch.setattr(G, "run_full", lambda skip: pytest.fail("full run entered from a pilot"))
    return seen


def test_pilot_default_ends_at_pilot_done_and_never_touches_full_list(tmp_path, monkeypatch):
    seen = _fake(tmp_path, monkeypatch)
    assert G.main([]) == 0
    assert (tmp_path / "pilot" / "ZZ_PILOT_DONE").exists()
    assert seen["clips"] and all(c == G.PILOT for c in seen["clips"])
    assert not any("clips_all" in str(x) for x in seen["clips"] + seen["cmds"])
    assert not (tmp_path / "ZZ_CHAIN_DONE").exists()


def test_full_flag_without_finished_pilot_is_refused(tmp_path, monkeypatch):
    _fake(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        G.main([G.FULL_FLAG])


def test_full_flag_with_finished_pilot_reaches_full(tmp_path, monkeypatch):
    _fake(tmp_path, monkeypatch)
    os.makedirs(tmp_path / "pilot")
    (tmp_path / "pilot" / "ZZ_PILOT_DONE").write_text("x")
    monkeypatch.setattr(G, "run_full", lambda skip: "FULL")
    assert G.main([G.FULL_FLAG]) == "FULL"


def test_regression_pilot_that_escalates_goes_red(tmp_path, monkeypatch):
    """Deliberate regression: a main() that also calls run_full after the pilot must trip the guard installed by _fake."""
    _fake(tmp_path, monkeypatch)

    def bad_main(argv):
        G.run_pilot()
        return G.run_full(False)
    with pytest.raises(pytest.fail.Exception):
        bad_main([])


def test_pilot_clip_list_has_smoke_and_is_200():
    p = json.load(open(HERE + "/splits/pilot_clips.json"))["clips"]
    s = json.load(open(HERE + "/smoke/smoke20.json"))["clips"]
    assert len(p) == 200 and len(set(p)) == 200 and set(s) <= set(p)
