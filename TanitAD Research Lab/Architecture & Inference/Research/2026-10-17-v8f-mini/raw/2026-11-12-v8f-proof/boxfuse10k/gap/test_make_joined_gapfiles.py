"""tests of make_joined_gapfiles.py (card PREREG_V8F_PROOF_ADD131). Tiny synthetic worlds of the kits of cards ADD127 to ADD129 (their test modules are imported).
Run one file at a time. BOXREAD_KIT_F names the folder of the kit of card ADD129 (default: ../../boxread/fuse, or the pod path)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.environ.get("BOXREAD_KIT_F"), os.path.join(os.path.dirname(os.path.dirname(HERE)), "boxread", "fuse"), "/workspace/v8proof/boxread/kit_f"):
    if cand and os.path.isfile(os.path.join(cand, "read_boxfuse.py")):
        KITF = cand
        break
else:
    raise RuntimeError("the kit of card ADD129 was not found")
for p in (KITF, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import test_read_boxfuse as TF  # noqa: E402
import make_joined_gapfiles as MJ  # noqa: E402
from test_read_boxfuse import fast_render, small_floor  # noqa: E402,F401

RF, RB, BS, T = MJ.RF, MJ.RB, MJ.BS, TF.T


def world(tmp_path, capsys, monkeypatch, all_frames=True, call_extra=""):
    """the files of cards ADD128 and ADD129, a base gap file, a detector folder of a pass over every frame, the call record, the argument file"""
    monkeypatch.setattr(RF, "F3_MIN_SHARE", 0.05)
    fa, info = TF.world(tmp_path, capsys)
    assert RF.main(fa("controls")) == 0 and RF.main(fa("names")) == 0
    capsys.readouterr()
    d = info["dirs"]
    root = info["root"]
    J = BS.load_npz(os.path.join(root, "join.npz"))
    L = BS.lay_dump(J, BS.load_npz(os.path.join(d["dump"], "D_0.npz")))
    ev = [str(c) for c in J["clip"]]
    order = ev[::-1]
    n, off = np.asarray(J["n"], np.int64), np.asarray(J["off"], np.int64)
    ci, fr, gap = [], [], []
    for k, c in enumerate(order):
        j = ev.index(c)
        ci += [k] * int(n[j])
        fr += list(range(int(n[j])))
        gap += list(np.asarray(L["gap"], np.float64)[int(off[j]):int(off[j] + n[j])])
    gap = np.asarray(gap, np.float64)
    fr = np.asarray(fr, np.int64)
    has = np.isfinite(gap) & (fr >= 4)
    base = os.path.join(root, "VGAPLE.npz")
    np.savez(base, clip_index=np.asarray(ci, np.int64), frame=fr, gap=np.where(has, gap, np.nan), has_out=has, clips=np.asarray(order), seed=np.int64(-1))
    new = os.path.join(root, "det_new")
    os.makedirs(os.path.join(new, "wide"))
    for c in ev:
        z = np.load(os.path.join(d["box"], c + ".npz"))
        frames = {}
        for f in range(T.NFR):
            lead = bool(z["has_lead"][f])
            if lead:
                frames[f] = T.frame_dets(float(z["ctr_az"][f]), float(z["ctr_el"][f]), float(z["ang_w"][f]), float(z["ang_h"][f]), T.WIDE)
            elif all_frames:
                frames[f] = []
        T.write_det(os.path.join(new, "wide", c + ".npz"), "wide", T.NFR, frames, T.WIDE, info["wsha"])
    BS.wj(os.path.join(new, "wide", "BOXDET_RECORD.json"), {"weights_sha256": info["wsha"], "camera": "wide"})
    call = os.path.join(new, "DET_CALL.txt")
    BS.wt(call, "python box_detect.py --camera wide --list eval.txt --root stage --out-dir det_new" + call_extra + "\n")
    argv = fa("controls")
    toks = argv[1:]
    i = toks.index("--out-dir")
    del toks[i:i + 2]
    af = os.path.join(root, "rb_args.txt")
    BS.wt(af, "\n".join(toks) + "\n")
    info.update({"base": base, "new": new, "call": call, "af": af, "r129": os.path.join(info["fout"], "FUSE_RESULT.json"), "gf": str(tmp_path / "gapfiles")})
    return info


def build(info, capsys, **kw):
    a = MJ.rb_args(info["af"], info["gf"] + "_scratch")
    rec = MJ.build(info["base"], info["new"], info["call"], kw.pop("r129", info["r129"]), info["gf"], a, **kw)
    ok = MJ.print_record(rec)
    return rec, ok, capsys.readouterr().out


def test_join_rule_and_second_routine():
    base = np.array([10.0, 20.0, 30.0, np.nan, 12.0, 15.0])
    has = np.array([True, True, True, False, True, True])
    gbox = np.array([14.0, 31.0, np.nan, 9.0, 12.0, 15.5])
    pick = np.array([True, True, False, True, True, False])
    out, g = MJ.join_gaps(base, has, gbox, pick, 1.5)
    assert g.tolist() == [True, False, False, False, True, False]
    assert abs(out[0] - np.sqrt(140.0)) < 1e-12 and out[1] == 20.0 and out[2] == 30.0 and np.isnan(out[3]) and out[4] == 12.0 and out[5] == 15.0
    assert np.allclose(out, MJ.join_rows(base, has, gbox, pick, 1.5), equal_nan=True)
    one, g1 = MJ.join_gaps(base, has, gbox, pick, 1.0)
    assert np.array_equal(one, np.where(has, base, np.nan), equal_nan=True)


def test_clean_world_all_controls_pass_and_no_error_is_printed(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info = world(tmp_path, capsys, monkeypatch)
    rec, ok, out = build(info, capsys)
    assert ok and out.strip().splitlines()[-1] == "ZZ JOINED-GAPFILES ALL-PASS", out
    assert all(rec["pass_j%d" % i] is True for i in range(5)) and rec["card"] == "PREREG_V8F_PROOF_ADD131"
    B, S, S9 = MJ.load(info["base"]), MJ.load(os.path.join(info["gf"], "VGAPJE.npz")), MJ.load(os.path.join(info["gf"], "VGAPJE90.npz"))
    has = B["has_out"].astype(bool)
    assert np.array_equal(np.isfinite(S["gap"]), has) and (S["gap"][has] != B["gap"][has]).any()
    assert np.allclose(S9["gap"][has], 0.9 * S["gap"][has]) and np.array_equal(S["clips"], B["clips"])
    assert rec["files"]["VGAPJE"]["sha256"] == MJ.sha_file(os.path.join(info["gf"], "VGAPJE.npz"))
    assert rec["counts"]["n_rows_gated"] > 0 and rec["counts"]["n_rows_with_output_whose_frame_was_not_run"] == 0
    assert set(rec["controls"]["j2"]["reproduced"].values()) == {True}
    r129 = json.load(open(info["r129"], encoding="utf-8"))
    for v in r129["measures"]["E"].values():
        assert ("%.3f" % v) not in out


def test_second_build_into_the_same_folder_refuses(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info = world(tmp_path, capsys, monkeypatch)
    build(info, capsys)
    with pytest.raises(SystemExit):
        build(info, capsys)


def test_j0_fails_for_a_pass_that_selected_lead_frames(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info = world(tmp_path, capsys, monkeypatch, call_extra=" --box-dir boxes --only-lead-frames")
    rec, ok, out = build(info, capsys)
    assert not ok and rec["pass_j0"] is False and "ctl j0 FAIL the pass was called with a selection" in out


def test_j0_fails_when_decodable_frames_were_not_run(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info = world(tmp_path, capsys, monkeypatch, all_frames=False)
    rec, ok, out = build(info, capsys)
    assert not ok and rec["pass_j0"] is False and "decodable frames on which the detector was not run" in out


@pytest.mark.parametrize("fault", ["shift", "filled"])
def test_j1_fails_on_a_planted_fault_of_the_file(tmp_path, capsys, fast_render, small_floor, monkeypatch, fault):
    info = world(tmp_path, capsys, monkeypatch)
    rec, ok, out = build(info, capsys, fault=fault)
    assert not ok and rec["pass_j1"] is False


def test_j2_fails_when_row_198_is_not_reproduced(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info = world(tmp_path, capsys, monkeypatch)
    rj = json.load(open(info["r129"], encoding="utf-8"))
    rj["measures"]["E"]["FG_2"] += 0.01
    bad = os.path.join(info["root"], "r129_changed.json")
    BS.wt(bad, json.dumps(rj))
    rec, ok, out = build(info, capsys, r129=bad)
    assert not ok and rec["pass_j2"] is False and "ctl j2 FAIL FG_2" in out


def test_j3_fails_when_the_new_pass_has_other_boxes(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info = world(tmp_path, capsys, monkeypatch)
    d = info["dirs"]
    for c in info["ev"]:
        z = np.load(os.path.join(d["box"], c + ".npz"))
        frames = {f: (T.frame_dets(float(z["ctr_az"][f]), float(z["ctr_el"][f]), 1.3 * float(z["ang_w"][f]), 1.3 * float(z["ang_h"][f]), T.WIDE) if z["has_lead"][f] else [])
                  for f in range(T.NFR)}
        p = os.path.join(info["new"], "wide", c + ".npz")
        os.replace(p, p + ".first")
        T.write_det(p, "wide", T.NFR, frames, T.WIDE, info["wsha"])
        os.remove(p + ".first")
    rec, ok, out = build(info, capsys)
    assert not ok and rec["pass_j3"] is False and "ctl j3 FAIL" in out


def test_j4_fails_for_a_gate_that_lets_every_row_pass(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info = world(tmp_path, capsys, monkeypatch)
    monkeypatch.setattr(MJ, "GATE_FACTOR", 1e9)
    rec, ok, out = build(info, capsys)
    assert not ok and rec["pass_j4"] is False and "ctl j4 FAIL with the base gaps permuted" in out


def test_t0_refuses_before_an_all_pass_record_and_names_after_it(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info = world(tmp_path, capsys, monkeypatch)
    a = MJ.rb_args(info["af"], str(tmp_path / "t0"))
    assert MJ.t0_main(info["base"], info["gf"], str(tmp_path / "t0"), a) == 3
    capsys.readouterr()
    build(info, capsys)
    rc = MJ.t0_main(info["base"], info["gf"], str(tmp_path / "t0"), a)
    out = capsys.readouterr().out
    assert rc == 0 and "ZZ BOXFUSE10K T0 NAME BOXFUSE10K-T0-JOINED-" in out, out
    rj = json.load(open(os.path.join(str(tmp_path / "t0"), "BOXFUSE10K_T0_RESULT.json"), encoding="utf-8"))
    assert rj["card"] == "PREREG_V8F_PROOF_ADD131" and rj["n_rows"] > 0 and len(rj["interval"]) == 2
