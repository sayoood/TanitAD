"""Tests for p5c_render.py (B8): refuses a number without its twin, refuses banned wording, one qualifier per verdict; RED arms for each.
Run: CUDA_VISIBLE_DEVICES=-1 python -m pytest test_p5c_render.py -q"""
import copy
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5c_read as R          # noqa: E402
import p5c_render as X        # noqa: E402
from test_p5c_read import mk_rows, arms_with_shift, _fam   # noqa: E402


def good_result(d_t=0.03, d_p=0.0, c0r=False):
    arms = arms_with_shift(d_t, d_p)
    if c0r:
        arms["C0r"] = copy.deepcopy(arms["C0"])
    return R.run(mk_rows(arms), "C1", "C0", "base", c0r="C0r" if c0r else None, fam_ci=_fam(), controls=[], rehearsal="stand-in arms")


def test_renders_a_complete_result():
    txt = X.render(good_result())
    assert "tracked" in txt and "transparent" in txt and "QUALIFIER" in txt and "STRATEGIC family: PAUSED" in txt
    assert txt.count("QUALIFIER ") == 1


def test_every_number_line_is_printed_twice_or_labelled():
    for ln in X.render(good_result()).splitlines():
        if ln.startswith(("C1 ", "C0 ", "base ", "C1-", "C0-")) and ": " in ln:
            assert ("tracked" in ln and "transparent" in ln) or "no tracker in this number" in ln or "both scorers are inside the definition" in ln, ln


def test_RED_dropping_the_transparent_twin_is_refused():
    r = good_result()
    del r["levels"]["C1"]["pdms"]["transparent"]
    with pytest.raises(X.RenderRefused, match="twin"):
        X.render(r)


def test_RED_dropping_the_tracked_twin_is_refused():
    r = good_result()
    del r["contrasts"]["C1-C0"]["ttc"]["tracked"]
    with pytest.raises(X.RenderRefused):
        X.render(r)


def test_no_tracker_label_is_accepted_and_printed():
    assert "no tracker in this number" in X.render_number("label distance", {"no_tracker": True, "value": 0.12})
    with pytest.raises(X.RenderRefused):
        X.render_number("x", {"no_tracker": True})


def test_bare_number_refused():
    with pytest.raises(X.RenderRefused):
        X.render_number("x", {"point": 1.0, "lo": 0, "hi": 2})
    with pytest.raises(X.RenderRefused):
        X.render_number("x", 3.0)


BANNED_SAMPLES = ["C1 drives better than C0", "the student drives better", "C1 is safer", "a safer planner", "better driving", "improved driving",
                  "drives more safely", "this improves driving", "improves safety", "C1 is a better driver", "superior driving", "drive well",
                  "driving quality improvement", "safety improvement", "C1 drives smoother", "it drives like a human"]


@pytest.mark.parametrize("phrase", BANNED_SAMPLES)
def test_RED_banned_wording_is_refused_anywhere(phrase):
    with pytest.raises(X.RenderRefused, match="banned"):
        X.render(good_result(), notes=phrase)


def test_RED_banned_wording_refused_under_TI():
    r = good_result(0.03, 0.03)                   # transparent gain too -> TI
    assert r["qualifier"]["code"] == "TI"
    with pytest.raises(X.RenderRefused, match="banned"):
        X.render(r, notes="TI means C1 drives better")


@pytest.mark.parametrize("phrase", ["C1's plans survive the benchmark's tracker (NAVSIM v1.1 LQR) better than C0's", "drivable area compliance", "DAC drivable",
                                    "better scores", "driving command adherence", "safety layer is out of scope"])
def test_allowed_wording_passes(phrase):
    assert X.banned_hits(phrase) == []


def test_lever_claim_sentence_is_the_cards():
    r = good_result(0.03, 0.0)
    r["verdict"] = dict(r["verdict"], verdict="LEVER", wording="C1's plans survive the benchmark's tracker better than C0's")
    txt = X.render(r)
    assert "CLAIM: C1's plans survive the benchmark's tracker (NAVSIM v1.1 LQR) better than C0's: tracked +" in txt
    assert "tracker dependence TD" in txt and "qualifier TM" in txt


def test_non_positive_verdict_prints_no_claim():
    r = good_result(0.0, 0.0)
    txt = X.render(r)
    assert "CLAIM:" not in txt and "DESCRIPTIVE ONLY" in txt


def test_exactly_one_qualifier_and_it_must_match_the_cis():
    r = good_result()
    r2 = copy.deepcopy(r)
    r2["qualifier"] = {"code": "TI", "text": "x"}            # d_T is ~0: TI's condition does not hold
    with pytest.raises(X.RenderRefused, match="does not match"):
        X.render(r2)
    r3 = copy.deepcopy(r)
    del r3["qualifier"]
    with pytest.raises(X.RenderRefused, match="qualifier"):
        X.render(r3)
    r4 = copy.deepcopy(r)
    r4["qualifier"] = {"code": "TM-ish", "text": "x"}
    with pytest.raises(X.RenderRefused):
        X.render(r4)


def test_TI_carries_the_not_closed_loop_note():
    assert "not a closed-loop result" in X.render(good_result(0.03, 0.03))


def test_deterministic_and_cli(tmp_path):
    r = good_result()
    assert X.render(r) == X.render(r)
    p, o = tmp_path / "r.json", tmp_path / "o.md"
    p.write_text(json.dumps(r))
    sys.argv = ["x", "--read", str(p), "--out", str(o)]
    assert X.main() == 0 and o.read_text(encoding="utf-8") == X.render(r)
    sys.argv = ["x", "--read", str(p), "--out", str(tmp_path / "no.md"), "--notes", "safer"]
    assert X.main() == 2 and not (tmp_path / "no.md").exists()


def test_rehearsal_banner_is_first_line():
    assert X.render(good_result()).splitlines()[0].startswith("*** stand-in arms")
