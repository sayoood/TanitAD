"""Tests for p5c_families.py (B9): summary reads the tracked frame as the only guard, polarity, strategic PAUSED, token file order.
Run: CUDA_VISIBLE_DEVICES=-1 python -m pytest test_p5c_families.py -q"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5c_families as F       # noqa: E402
from p5c_read import fam_direction    # noqa: E402
from test_p5c_read import _fam, ent    # noqa: E402


def test_summary_marks_only_the_tracked_frame_as_guard():
    bad = {"speed_mae_mps": ent(0.1, 0.3)}
    s = F.summarize(_fam(entries_tracked=bad, entries_transp=bad))
    assert s["frames"]["executed_tracked"]["guard"] is True and s["frames"]["executed_transparent"]["guard"] is False
    assert [a["component"] for a in s["frames"]["executed_tracked"]["separated_adverse"]] == ["speed_mae_mps"]
    assert "NEVER A GUARD" in s["frames"]["executed_transparent"]["role"]
    assert s["strategic"]["status"] == "PAUSED"


def test_RED_polarity_of_dotted_names_keeps_the_decimal_point():
    """regression of the bug found while building B7: 'target_speed_acc.within_1.0_mps' was split at its decimal point and read as LOWER-is-better."""
    s = F.summarize(_fam(entries_tracked={"target_speed_acc.within_1.0_mps": ent(0.1, 0.3)}))
    assert s["frames"]["executed_tracked"]["separated_adverse"] == []             # higher accuracy for C1 is NOT adverse
    s = F.summarize(_fam(entries_tracked={"target_speed_acc.within_1.0_mps": ent(-0.3, -0.1)}))
    assert len(s["frames"]["executed_tracked"]["separated_adverse"]) == 1


def test_missing_frame_is_reported_absent():
    s = F.summarize({"frames": {"executed_tracked": {"families": {}}}})
    assert s["frames"]["executed_transparent"]["status"] == "ABSENT"


def test_tokens_file_is_sorted_by_token_and_has_log_prefix(tmp_path):
    rows = [{"log": "logB", "token": "bbb", "arms": {}}, {"log": "logA", "token": "aaa", "arms": {}}, {"log": "logC", "token": "ccc", "error": "x"}]
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "tokens.txt"
    assert F.write_tokens(str(p), str(out)) == 2
    assert out.read_text().splitlines() == ["logA:aaa", "logB:bbb"]


def test_pair_direction_is_x_minus_y(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(F, "run_cmd", lambda argv, log: seen.update(argv=argv) or 1)
    with pytest.raises(SystemExit):
        F.pair_ci("fam_C1.json", "fam_C0.json", "C1", "C0", "tok.txt", str(tmp_path), "py")
    a = seen["argv"]
    assert a[a.index("--a") + 1] == "fam_C0.json" and a[a.index("--b") + 1] == "fam_C1.json"      # diff = B - A = C1 - C0


def test_polarity_rule_matches_proxy_eval_if_present():
    pe = os.path.join(HERE, "..", "proxy_eval.py")
    if not os.path.exists(pe):
        pytest.skip("proxy_eval not in tree")
    import importlib.util
    sp = importlib.util.spec_from_file_location("pe_t", pe)
    m = importlib.util.module_from_spec(sp)
    try:
        sp.loader.exec_module(m)
    except Exception:
        pytest.skip("proxy_eval not importable here")
    for k in ("speed_mae_mps", "accel_mae_mps2", "target_speed_acc.within_1.0_mps", "heading_bias_deg", "lead_progress_ratio", "kappa"):
        assert fam_direction(k) == m.fam_direction(k)


def test_RED_raised_frames_are_detected():
    """a families_run frame whose block RAISED must stop B9 (else fam_ci writes empty families and G2 reads a vacuous PASS)."""
    ok = {"frames": {"executed_tracked": {"role": "PRIMARY", "block": {"status": "OK", "longitudinal": {}}}}}
    bad = {"frames": {"executed_tracked": {"role": "PRIMARY", "block": {"status": "RAISED", "error": "TypeError: per_window"}}, "reference": {"block": {"status": "OK"}}}}
    assert F.raised_frames(ok) == [] and F.raised_frames(bad) == [("executed_tracked", "TypeError: per_window")]
