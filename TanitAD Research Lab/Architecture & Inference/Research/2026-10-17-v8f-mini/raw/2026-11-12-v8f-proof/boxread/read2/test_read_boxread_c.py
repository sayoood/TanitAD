"""tests of read_boxread_c.py (card PREREG_V8F_PROOF_ADD128). Tiny synthetic worlds of the kit of card ADD127 (its test module is imported for the world builder).
Run one file at a time. BOXREAD_KIT_READ names the folder of the kit of card ADD127 (default: ../read beside this folder, or the pod path)."""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.environ.get("BOXREAD_KIT_READ"), os.path.join(os.path.dirname(HERE), "read"), "/workspace/v8proof/boxread/kit/read"):
    if cand and os.path.isfile(os.path.join(cand, "read_boxread.py")):
        KIT = cand
        break
else:
    raise RuntimeError("the kit of card ADD127 was not found")
for p in (KIT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import test_read_boxread as T  # noqa: E402
import read_boxread_c as RC  # noqa: E402
from test_read_boxread import fast_render  # noqa: E402,F401

RB = RC.RB


@pytest.fixture
def small_floor(monkeypatch):
    monkeypatch.setattr(RC, "C2_MIN_FRAMES", 50)
    monkeypatch.setattr(RC, "C2_MIN_CLIPS", 10)


def world(tmp_path, capsys, **opt):
    """-> (argv builder with the prior controls file and the locator hash, info)"""
    argv, info = T.make_world(tmp_path / "w", **opt)
    prior_out = str(tmp_path / "prior")
    RB.main(argv("controls", out=prior_out))
    capsys.readouterr()
    prior = os.path.join(prior_out, "BOXREAD_CONTROLS.json")
    sha = RB.sha_of(os.path.join(info["dirs"]["loc"], "LOC_s0.npz"))

    def av(mode, extra=(), sha_chars=16, with_prior=True):
        e = list(extra)
        if with_prior:
            e += ["--prior-controls", prior]
        if sha_chars:
            e += ["--locator-sha256", sha[:sha_chars]]
        return argv(mode, extra=e)
    info["prior"] = prior
    return av, info


def run(argv, capsys):
    rc = RC.main(argv)
    return rc, capsys.readouterr().out


def controls_json(info):
    return json.load(open(os.path.join(info["dirs"]["out"], "BOXREAD_CONTROLS.json"), encoding="utf-8"))


def rewrite_train_wide(info, fn):
    """write the wide detector files of the train clips again; fn(az, el, w, h) -> list of (box, score, label)"""
    d = info["dirs"]
    for t in info["tr"]:
        z = np.load(os.path.join(d["box"], t + ".npz"))
        frames = {f: fn(float(z["ctr_az"][f]), float(z["ctr_el"][f]), float(z["ang_w"][f]), float(z["ang_h"][f])) for f in range(T.NFR)}
        p = os.path.join(d["det/wide"], t + ".npz")
        os.replace(p, p + ".first")
        T.write_det(p, "wide", T.NFR, frames, T.WIDE, info["wsha"])
        os.remove(p + ".first")


def test_reader_identity_is_checked(monkeypatch):
    assert RC._sha(RB.__file__).startswith(RC.RB_PREFIX)
    monkeypatch.setattr(RC, "RB_PREFIX", "0" * 16)
    with pytest.raises(SystemExit):
        RC.load_reader()


def test_clean_world_all_pass_and_the_imported_reader_is_left_as_it_was(tmp_path, capsys, fast_render, small_floor):
    av, info = world(tmp_path, capsys)
    rc, out = run(av("controls"), capsys)
    lines = [x for x in out.splitlines() if x.strip()]
    assert rc == 0 and lines[-1] == "ZZ BOXREAD CONTROLS ALL-PASS", out
    ok = re.compile(r"^(ctl b[0-7] PASS|b5 share 0\.\d{4}|ZZ BOXREAD CONTROLS ALL-PASS)$")
    assert all(ok.match(x) for x in lines), lines
    cj = controls_json(info)
    assert cj["card"] == "PREREG_V8F_PROOF_ADD128" and cj["all_pass"] is True
    b2 = cj["controls"]["b2"]
    assert b2["mirrored_share_decides"] is False and b2["n_frames_displaced"] >= 50
    assert all(v <= 0.10 for v in b2["displaced_same_share"].values())
    assert cj["controls"]["b1"]["inside_share_decides"] is False
    assert cj["controls"]["b0"]["detector_listing_as_card_add127"] == {"wide": True, "tele": True}
    assert cj["inputs"]["read_boxread_c.py"] == RC._sha(RC.__file__) and cj["inputs"]["prior_controls"] == RC._sha(info["prior"])
    assert RB.CARD == "PREREG_V8F_PROOF_ADD127" and RB.ctl_b2 is RC.ORIG["ctl_b2"] and RB.build_parser is RC.ORIG["build_parser"]


def test_c2_fails_with_too_few_frames_at_the_card_floor(tmp_path, capsys, fast_render):
    av, info = world(tmp_path, capsys)
    rc, out = run(av("controls"), capsys)
    assert rc == 2 and re.search(r"^ctl b2 FAIL the displacement clause has \d+ frames of \d+ clips", out, re.M), out


def test_c2_fails_when_the_boxes_cover_the_window(tmp_path, capsys, fast_render, small_floor):
    av, info = world(tmp_path, capsys)
    px = float(T.WIDE[1])
    rewrite_train_wide(info, lambda az, el, w, h: [([0.0, 0.0, px, px], 0.9, 3)])
    rc, out = run(av("controls", with_prior=False), capsys)
    assert rc == 2 and re.search(r"^ctl b2 FAIL with the direction displaced (left|right|up) the same detection is picked on \d\.\d{4}", out, re.M), out
    sh = controls_json(info)["controls"]["b2"]["displaced_same_share"]
    assert min(sh.values()) > 0.5


def test_c2_fails_when_the_azimuth_sign_of_the_boxes_is_turned(tmp_path, capsys, fast_render, small_floor):
    av, info = world(tmp_path, capsys)
    rewrite_train_wide(info, lambda az, el, w, h: [(T.box_for(-az, el, w, h, T.WIDE), 0.9, 3)])
    rc, out = run(av("controls", with_prior=False), capsys)
    assert rc == 2 and re.search(r"^ctl b2 FAIL \S", out, re.M), out


def test_leads_at_the_image_centre_fail_the_mirror_of_card_add127_and_pass_c2(tmp_path, capsys, fast_render, small_floor):
    av, info = world(tmp_path, capsys, train_az=0.0)
    pj = json.load(open(info["prior"], encoding="utf-8"))
    assert pj["card"] == "PREREG_V8F_PROOF_ADD127" and pj["controls"]["b2"]["pass"] is False
    assert any(f.startswith(RC.MIRROR_LINE) for f in pj["controls"]["b2"]["failures"])
    rc, out = run(av("controls"), capsys)
    assert "ctl b2 PASS" in out.splitlines(), out
    b2 = controls_json(info)["controls"]["b2"]
    assert b2["share_mirrored"] > 0.5 * b2["share"] and b2["pass"] is True


def test_c1_stores_the_inside_share_beside_and_does_not_fail_on_it(tmp_path, capsys, fast_render, small_floor):
    av, info = world(tmp_path, capsys, train_az=40.0)
    pj = json.load(open(info["prior"], encoding="utf-8"))
    assert pj["controls"]["b1"]["pass"] is False
    rc, out = run(av("controls"), capsys)
    assert "ctl b1 PASS" in out.splitlines(), out
    b1 = controls_json(info)["controls"]["b1"]
    assert b1["inside_share"] < 0.90 and b1["inside_share_decides"] is False


def test_c1_still_fails_when_the_render_tool_test_fails(tmp_path, capsys, monkeypatch, small_floor):
    monkeypatch.setattr(RB.BG, "load_enc_common", lambda path=None: object())
    monkeypatch.setattr(RB.BG, "render_roundtrip", lambda enc, win, az, el, w, h, **k: (az + 0.5, el, w, h))
    av, info = world(tmp_path, capsys)
    rc, out = run(av("controls"), capsys)
    assert rc == 2 and re.search(r"^ctl b1 FAIL \S", out, re.M), out


@pytest.mark.parametrize("chars,good", [(16, True), (64, True), (15, False), (0, False)])
def test_c4_takes_a_beginning_of_at_least_16_characters(tmp_path, capsys, fast_render, small_floor, chars, good):
    av, info = world(tmp_path, capsys)
    rc, out = run(av("controls", sha_chars=chars), capsys)
    assert ("ctl b4 PASS" in out.splitlines()) is good, out


def test_c4_fails_on_other_characters(tmp_path, capsys, fast_render, small_floor):
    av, info = world(tmp_path, capsys)
    rc, out = run(av("controls", sha_chars=0, extra=["--locator-sha256", "0" * 16]), capsys)
    assert rc == 2 and re.search(r"^ctl b4 FAIL the sha256 of the locator file does not begin", out, re.M), out


def test_c0_needs_the_controls_file_of_card_add127_and_the_same_detector_files(tmp_path, capsys, fast_render, small_floor):
    av, info = world(tmp_path, capsys)
    rc, out = run(av("controls", with_prior=False), capsys)
    assert rc == 2 and re.search(r"^ctl b0 FAIL no controls file of card ADD127 given", out, re.M), out
    extra = os.path.join(info["dirs"]["det/wide"], "zz_one_more.npz")
    T.write_det(extra, "wide", T.NFR, {}, T.WIDE, info["wsha"])
    rc, out = run(av("controls"), capsys)
    assert rc == 2 and re.search(r"^ctl b0 FAIL the wide detector files are not the ones", out, re.M), out


def test_names_open_only_after_the_controls_of_this_card(tmp_path, capsys, fast_render, small_floor):
    av, info = world(tmp_path, capsys)
    rc, out = run(av("names"), capsys)
    assert rc == 3 and "REFUSED" in out
    rc, out = run(av("controls"), capsys)
    assert rc == 0, out
    rc, out = run(av("names"), capsys)
    names = [x.split()[-1] for x in out.splitlines() if x.startswith("ZZ BOXREAD NAME ")]
    assert rc == 0 and len(names) >= 2 and all(n.startswith("BOXREAD-T0-") for n in names), out
    d = info["dirs"]["out"]
    rj = json.load(open(os.path.join(d, "BOXREAD_RESULT.json"), encoding="utf-8"))
    assert rj["card"] == "PREREG_V8F_PROOF_ADD128"
    txt = open(os.path.join(d, "BOXREAD_VERDICT.md"), encoding="utf-8").read()
    assert "ADD128" in txt and "changed after the controls run of card ADD127" in txt


def test_a_controls_file_of_card_add127_does_not_open_the_names_of_this_card(tmp_path, capsys, fast_render, small_floor):
    av, info = world(tmp_path, capsys)
    out_dir = info["dirs"]["out"]
    RB.main(av("controls", with_prior=False, sha_chars=64)[: -0 or None])
    capsys.readouterr()
    cj = controls_json(info)
    assert cj["card"] == "PREREG_V8F_PROOF_ADD127"
    rc, out = run(av("names"), capsys)
    assert rc == 3 and "REFUSED" in out and not os.path.exists(os.path.join(out_dir, "BOXREAD_RESULT.json"))
