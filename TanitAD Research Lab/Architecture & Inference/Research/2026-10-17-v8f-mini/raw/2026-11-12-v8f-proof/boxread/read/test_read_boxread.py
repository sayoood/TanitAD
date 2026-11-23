"""tests of read_boxread.py and box_geom.py on tiny synthetic data (card PREREG_V8F_PROOF_ADD127). No result, video, image, token or detector file of the project is opened.
Portable: the interpreter is sys.executable; BOXSIZE_READER, BOXREAD_ENC_COMMON, TANITAD_STACK and BOXREAD_TELE_ENCODE name the imported tools; BOXREAD_TEST_CWD sets the cwd of child processes."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import box_geom as BG  # noqa: E402
import read_boxread as RB  # noqa: E402

BS = RB.BS
WIDE, TELE = RB.WIDE_WIN, RB.TELE_WIN
NE, NT, NFR = BS.SYN_NE, BS.SYN_NT, BS.SYN_NFR


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def _enc_path():
    for p in (os.environ.get("BOXREAD_ENC_COMMON"), "/workspace/v8proof/cropread/kit/enc/enc_common.py", "F:/TanitAD-scratch/v8proof/cropread/enc/enc_common.py"):
        if p and os.path.isfile(p):
            return p
    return None


IMPORTED = {"read_boxsize.py": BS.__boxread_path__, "enc_common.py": _enc_path()}
HASH_AT_IMPORT = {k: _sha(v) for k, v in IMPORTED.items() if v}


def _have_render():
    try:
        BG.load_enc_common(None).setup_paths()
        import torch  # noqa: F401
        from tanitad.data.calib import FThetaIntrinsics  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


HAVE_RENDER = _have_render()


# ============================================================================================ geometry
@pytest.mark.parametrize("win", [WIDE, TELE, (8.0, 256.0, 1.0, -2.0)])
def test_pixel_direction_round_trip(win):
    rng = np.random.RandomState(1)
    px = win[1]
    c, r = rng.uniform(0, px, 40), rng.uniform(0, px, 40)
    az, el = BG.pix_to_dir(c, r, win)
    c2, r2, ahead = BG.dir_to_pix(az, el, win)
    assert ahead.all() and np.abs(c2 - c).max() < 1e-6 and np.abs(r2 - r).max() < 1e-6
    cc, rr, _ = BG.dir_to_pix(win[2], win[3], win)                      # the window centre direction is the centre pixel
    assert abs(cc - px / 2) < 1e-6 and abs(rr - px / 2) < 1e-6


def test_azimuth_left_and_elevation_up_signs():
    c0, r0, _ = BG.dir_to_pix(0.0, -6.0, WIDE)
    cl, _, _ = BG.dir_to_pix(4.0, -6.0, WIDE)
    _, ru, _ = BG.dir_to_pix(0.0, -2.0, WIDE)
    assert cl < c0 and ru < r0                                          # left is a smaller column, up is a smaller row
    assert abs((c0 - cl) / 4.0 - 24.0) < 0.3                            # about 24 pixels per degree (the cylinder is tilted by 6 degrees: not exact off the axis)
    t0, _, _ = BG.dir_to_pix(0.0, 0.0, TELE)
    t1, _, _ = BG.dir_to_pix(2.0, 0.0, TELE)
    assert abs((t0 - t1) / 2.0 - 64.0) < 1e-6                           # exactly 64 pixels per degree on an untilted window


def test_box_angles_of_a_planted_rectangle():
    for win, az, el, w, h in BG.RENDER_CASES:
        c, r, _ = BG.dir_to_pix([az - w / 2, az + w / 2, az - w / 2, az + w / 2], [el - h / 2, el - h / 2, el + h / 2, el + h / 2], win)
        a = BG.box_angles([c.min(), r.min(), c.max(), r.max()], win)
        assert max(abs(a[0] - az), abs(a[1] - el), abs(a[2] - w), abs(a[3] - h)) < 0.1


@pytest.mark.skipif(not HAVE_RENDER, reason="enc_common, torch or the stack tree not importable here")
def test_rectangle_through_window_grid_comes_back_within_0p1_degree():
    """control b1's tool test: a rectangle drawn at a known direction and angular size into a synthetic camera frame, rendered by enc_common.window_grid."""
    enc = BG.load_enc_common(None)
    for win, az, el, w, h in BG.RENDER_CASES:
        r = BG.render_roundtrip(enc, win, az, el, w, h)
        assert r is not None
        assert max(abs(r[0] - az), abs(r[1] - el), abs(r[2] - w), abs(r[3] - h)) < 0.1, (win, r)


def test_contains_and_edge_distance():
    box = [300.0, 300.0, 420.0, 400.0]
    c, r = 360.0, 350.0
    az, el = BG.pix_to_dir(c, r, WIDE)
    assert BG.contains(box, az, el, WIDE) and BG.edge_distance_deg(box, az, el, WIDE) == 0.0
    az2, el2 = BG.pix_to_dir(420.0 + 24.0, 350.0, WIDE)                  # 1.0 degree right of the right edge
    assert not BG.contains(box, az2, el2, WIDE)
    assert abs(BG.edge_distance_deg(box, az2, el2, WIDE) - 1.0) < 0.02
    az3, el3 = BG.pix_to_dir(420.0 + 48.0, 350.0, WIDE)
    assert abs(BG.edge_distance_deg(box, az3, el3, WIDE) - 2.0) < 0.03
    assert BG.edge_distance_deg(box, 170.0, 0.0, WIDE) == np.inf         # behind the window axis


def test_pick_rule():
    big = [200.0, 200.0, 500.0, 500.0]
    small = [330.0, 330.0, 390.0, 380.0]
    far = [600.0, 300.0, 650.0, 360.0]
    az, el = BG.pix_to_dir(360.0, 350.0, WIDE)
    assert BG.pick_one([big], az, el, WIDE) == 0                         # contains
    assert BG.pick_one([big, small, far], az, el, WIDE) == 1             # several contain it: the smallest
    az_o, el_o = BG.pix_to_dir(330.0 - 24.0 * 1.2, 350.0, WIDE)          # 1.2 degrees left of the small box, outside the big one's reach
    assert BG.pick_one([small, far], az_o, el_o, WIDE) == 0              # nearest edge within 1.5 degrees
    az_p, el_p = BG.pix_to_dir(330.0 - 24.0 * 1.7, 350.0, WIDE)
    assert BG.pick_one([small, far], az_p, el_p, WIDE) == -1             # 1.7 degrees: none
    assert BG.pick_one([], az, el, WIDE) == -1 and BG.pick_one([small], np.nan, 0.0, WIDE) == -1


def test_point_in_other_cam_identity_and_source():
    az, el = np.array([3.0, -2.0]), np.array([-1.0, 4.0])
    a, e = BG.point_in_other_cam(np.eye(3), np.eye(3), az, el)
    assert np.allclose(a, az) and np.allclose(e, el)
    R = BG.rot_cam_to_vehicle(0.0, 0.01, 0.02, 1.0)
    p = os.environ.get("BOXREAD_TELE_ENCODE") or "F:/TanitAD-scratch/v8proof/telegap/enc/tele_encode.py"
    if os.path.isfile(p) and HAVE_RENDER:
        import importlib.util
        try:
            spec = importlib.util.spec_from_file_location("tele_encode_probe", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
        except Exception:  # noqa: BLE001
            pytest.skip("tele_encode.py not importable here")
        a2, e2 = BG.point_in_other_cam(np.eye(3), R, az, el)
        for i in range(2):
            ref = m.point_in_other_cam(np.eye(3), R, float(az[i]), float(el[i]))
            assert abs(ref[0] - a2[i]) < 1e-9 and abs(ref[1] - e2[i]) < 1e-9


# ============================================================================================ a tiny world
AZ0, EL0 = 6.0, -2.0


def az_of(f, base=AZ0):
    return base + 0.05 * np.asarray(f, np.float64)


def box_for(az, el, w, h, win):
    c, r, _ = BG.dir_to_pix([az - w / 2, az + w / 2, az - w / 2, az + w / 2], [el - h / 2, el - h / 2, el + h / 2, el + h / 2], win)
    return [float(c.min()), float(r.min()), float(c.max()), float(r.max())]


def write_det(path, cam, n, frames, win, weights, run=None):
    """frames: {frame: [(box, score, label), ...]}; run: bool [n] (default: the frames that have a list)."""
    det_frame, boxes, sc, lb = [], [], [], []
    for f in sorted(frames):
        for b, s, l in frames[f]:
            det_frame.append(f)
            boxes.append(b)
            sc.append(s)
            lb.append(l)
    fr_run = np.zeros(n, bool)
    for f in frames:
        fr_run[f] = True
    if run is not None:
        fr_run = np.asarray(run, bool)
    np.savez(path, n_frames=np.int64(n), frame_ok=np.ones(n, bool), frame_run=fr_run, det_frame=np.asarray(det_frame, np.int32), det_box=np.asarray(boxes, np.float32).reshape(-1, 4),
             det_score=np.asarray(sc, np.float32), det_label=np.asarray(lb, np.int16), win=np.asarray(win, np.float64), camera=np.asarray(cam), weights_sha256=np.asarray(weights))


def frame_dets(az, el, w, h, win, mode="exact"):
    exact = (box_for(az, el, w, h, win), 0.9, 3)
    person = (box_for(az, el, w * 2.0, h * 2.0, win), 0.99, 1)           # not a vehicle: must be ignored
    weak = (box_for(az, el, w * 0.5, h * 0.5, win), 0.4, 3)              # below the score bar: must be ignored
    return [weak, person] if mode == "none" else [exact, person, weak]


def make_world(root, **o):
    """a tiny pod-like tree. options: see the body. -> (argv builder, info)"""
    root = str(root)
    J, dumps, preds, boxes, tbox = BS.syn_world("cuts")
    d = {k: os.path.join(root, k) for k in ("box", "det/wide", "det/tele", "lists", "dump", "loc", "tests", "out", "cal")}
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    wsha = hashlib.sha256(b"weights-of-the-test").hexdigest()
    rng = np.random.RandomState(11)
    ev = [str(c) for c in J["clip"]]
    nrow = len(ev) * NFR
    notrun_clip = ev[0]
    for c in ev:
        b = dict(boxes[c])
        b["ctr_az"], b["ctr_el"] = az_of(np.arange(NFR)), np.full(NFR, EL0)
        np.savez(os.path.join(d["box"], c + ".npz"), **b)
        run_frames = {}
        tele_frames = {}
        for f in range(NFR):
            if not b["has_lead"][f]:
                continue
            w, h = float(b["ang_w"][f]), float(b["ang_h"][f])
            ws, hs = w * o.get("eval_scale", 1.0), h * o.get("eval_scale", 1.0)
            if o.get("wide_noise"):
                k = float(np.exp(o["wide_noise"] * rng.randn()))
                ws, hs = ws * k, hs * k
            mode = "none" if (o.get("eval_none") or (o.get("eval_sparse") and f % 20 != 5)) else "exact"
            run_frames[f] = frame_dets(float(az_of(f)), EL0, ws, hs, WIDE, mode)
            tele_frames[f] = frame_dets(float(az_of(f)), EL0, w, h, TELE)
        run = None
        if o.get("eval_notrun") and c == notrun_clip:
            run = np.zeros(NFR, bool)
            for f in run_frames:
                run[f] = f < 10
            run_frames = {f: v for f, v in run_frames.items() if f < 10}
        write_det(os.path.join(d["det/wide"], c + ".npz"), "wide", NFR, run_frames, WIDE, wsha, run)
        if o.get("tele"):
            trun = None
            if o.get("tele_notrun"):
                trun = np.zeros(NFR, bool)
                tele_frames = {}
            write_det(os.path.join(d["det/tele"], c + ".npz"), "tele", NFR, tele_frames, TELE, wsha, trun)
    tr = sorted(tbox)
    for t in tr:
        b = dict(tbox[t])
        b["ctr_az"], b["ctr_el"] = az_of(np.arange(NFR), o.get("train_az", AZ0)), np.full(NFR, EL0)
        np.savez(os.path.join(d["box"], t + ".npz"), **b)
        wf, tf = {}, {}
        run = np.ones(NFR, bool)
        if o.get("train_notrun") and int(t[1:]) % 2 == 0:
            run[: NFR // 2] = False
        for f in range(NFR):
            w, h = float(b["ang_w"][f]), float(b["ang_h"][f])
            if o.get("train_const"):
                w, h = 4.0, 3.0
            mode = "none" if o.get("train_dead") else "exact"
            if run[f]:
                wf[f] = frame_dets(float(b["ctr_az"][f]), EL0, w, h, WIDE, mode)
                tf[f] = frame_dets(float(b["ctr_az"][f]), EL0, float(b["ang_w"][f]), float(b["ang_h"][f]), TELE, mode)
        write_det(os.path.join(d["det/wide"], t + ".npz"), "wide", NFR, wf, WIDE, wsha, run)
        if o.get("tele"):
            write_det(os.path.join(d["det/tele"], t + ".npz"), "tele", NFR, tf, TELE, wsha, run)
    for cam in ("wide", "tele"):
        BS.wj(os.path.join(d["det/" + cam], "BOXDET_RECORD.json"), {"weights_sha256": wsha, "camera": cam})
    BS.wt(os.path.join(d["lists"], "train.txt"), "".join("%s fit\n" % c for c in tr))
    BS.wt(os.path.join(d["lists"], "eval.txt"), "".join(c + "\n" for c in ev))
    th, eh = BS.sha_file(os.path.join(d["lists"], "train.txt")), BS.sha_file(os.path.join(d["lists"], "eval.txt"))
    BS.wj(os.path.join(root, "list_record.json"), {"train_list_sha256": th, "eval_list_sha256": eh})
    np.savez(os.path.join(root, "join.npz"), **J)
    dp = {}
    for k, dd in dumps.items():
        dd = dict(dd)
        if o.get("d_bad") and k == "D_1":
            dd["gap"] = dd["gap"] + 1.0
        p = os.path.join(d["dump"], k + ".npz")
        np.savez(p, **dd)
        dp[k] = p
    t3 = BS.t3_rows({k: BS.lay_dump(J, x) for k, x in dumps.items()})
    ref = {k: round(BS.mean_abs_err(BS.lay_dump(J, dumps[k])["gap"], BS.lay_dump(J, dumps[k])["lab_gap"], t3), 3) for k in BS.ARM_D}
    # locator's file
    ci = np.repeat(np.arange(len(ev)), NFR)
    fr = np.tile(np.arange(NFR), len(ev))
    off = o.get("loc_offset", 0.0)
    az = az_of(fr) + 0.2 * np.sin(fr) + off
    el = np.full(nrow, EL0)
    has = fr >= 4
    lp = os.path.join(d["loc"], "LOC_s0.npz")
    if o.get("loc_label"):
        np.savez(lp, **dict(np.load(os.path.join(d["box"], ev[0] + ".npz"))))
    else:
        np.savez(lp, clip_index=ci, frame=fr, az_pred=np.where(has, az, np.nan), el_pred=np.where(has, el, np.nan), has_out=has, clips=np.asarray(ev))
    BS.wj(os.path.join(d["loc"], "LOC_s0_meta.json"), {"pred_sha256": BS.sha_file(lp)})
    tag = wsha[:8] if not o.get("weights_bad") else "deadbeef"
    BS.wj(os.path.join(root, "WEIGHTS_RECORD.json"), {"sha256": wsha, "name_tag": tag, "sha256_begins_with_name_tag": True})
    for cam in ("wide", "tele"):
        rows = "clip_id,qx,qy,qz,qw,x,y,z\n" + "".join("%s,0,0,0,1,0,0,0\n" % c for c in ev + tr)
        BS.wt(os.path.join(d["cal"], "front_%s_extrinsics.csv" % cam), rows)
    now = time.time()
    for k in ("det", "read"):
        p = os.path.join(d["tests"], k + ".out")
        BS.wt(p, "1 failed, 17 passed in 1s\n" if (o.get("tests_bad") and k == "read") else "....\n18 passed in 1.0s\n")
        os.utime(p, (now - 500, now - 500))
    info = {"root": root, "dirs": d, "ref": ref, "n_rows": int(t3.sum()), "ev": ev, "tr": tr, "J": J, "dumps": dumps, "wsha": wsha}

    def argv(mode, out=None, extra=()):
        a = [mode, "--eval-join", os.path.join(root, "join.npz"), "--box-dir", d["box"], "--list-record", os.path.join(root, "list_record.json"),
             "--list-file", "train_list=" + os.path.join(d["lists"], "train.txt"), "--list-file", "eval_list=" + os.path.join(d["lists"], "eval.txt"),
             "--det-dir", os.path.join(root, "det"), "--locator-dir", d["loc"], "--weights-record", os.path.join(root, "WEIGHTS_RECORD.json"),
             "--row191-values", ",".join("%s=%s" % kv for kv in ref.items()), "--wide-extr", os.path.join(d["cal"], "front_wide_extrinsics.csv"),
             "--tele-extr", os.path.join(d["cal"], "front_tele_extrinsics.csv"), "--out-dir", out or d["out"], "--B", "300", "--expect-rows", str(info["n_rows"]),
             "--expect-clips", str(o.get("expect_clips", len(ev))),
             "--test-out", "det=" + os.path.join(d["tests"], "det.out"), "--test-out", "read=" + os.path.join(d["tests"], "read.out"),
             "--test-before", "det=%f" % (now - 100), "--test-before", "read=%f" % (now - 100)]
        for k, p in dp.items():
            a += ["--d-pred", "%s=%s" % (k, p)]
        return a + list(extra)
    info["argv"] = argv
    return argv, info


@pytest.fixture
def fast_render(monkeypatch):
    """the real render is tested once above; the scenarios below use an exact stand-in for it to stay quick."""
    monkeypatch.setattr(RB.BG, "load_enc_common", lambda path=None: object())
    monkeypatch.setattr(RB.BG, "render_roundtrip", lambda enc, win, az, el, w, h, **k: (az, el, w, h))


def run_main(argv, capsys):
    rc = RB.main(argv)
    return rc, capsys.readouterr().out


# ============================================================================================ controls
def test_clean_world_all_pass_real_render_and_no_measure_printed(tmp_path, capsys):
    if not HAVE_RENDER:
        pytest.skip("render needs enc_common, torch and the stack tree")
    argv, info = make_world(tmp_path / "w")
    rc, out = run_main(argv("controls"), capsys)
    lines = [x for x in out.splitlines() if x.strip()]
    assert rc == 0 and lines[-1] == "ZZ BOXREAD CONTROLS ALL-PASS", out
    for k in RB.CONTROLS:
        assert "ctl %s PASS" % k in lines
    ok = re.compile(r"^(ctl b[0-7] PASS|b5 share 0\.\d{4}|ZZ BOXREAD CONTROLS ALL-PASS)$")
    assert all(ok.match(x) for x in lines), lines
    d = info["dirs"]["out"]
    cj = json.load(open(os.path.join(d, "BOXREAD_CONTROLS.json"), encoding="utf-8"))
    assert cj["all_pass"] is True and sorted(cj["controls"]) == list(RB.CONTROLS)
    assert set(cj["controls"]["b0"]["arm_d_within_tolerance"].values()) == {True}
    txt = json.dumps(cj)
    for v in info["ref"].values():                                        # no error value of an evaluation arm is stored
        assert ("%.3f" % v) not in txt
    assert not os.path.exists(os.path.join(d, "BOXREAD_RESULT.json")) and not os.path.exists(os.path.join(d, "BOXREAD_VERDICT.md"))


FAULTS = [("b0", {"weights_bad": True}), ("b0", {"d_bad": True}), ("b1", {"train_az": 40.0}), ("b2", {"train_dead": True}), ("b3", {"train_const": True}),
          ("b4", {"loc_label": True}), ("b5", {"loc_offset": 30.0}), ("b6", {"tests_bad": True}), ("b7", {"eval_notrun": True}), ("b7", {"expect_clips": 140})]


@pytest.mark.parametrize("ctl,opt", FAULTS, ids=["%s-%s" % (c, "-".join(o)) for c, o in FAULTS])
def test_each_control_fails_on_its_planted_defect(tmp_path, capsys, fast_render, ctl, opt):
    argv, info = make_world(tmp_path / "w", **opt)
    rc, out = run_main(argv("controls"), capsys)
    assert rc == 2 and out.strip().splitlines()[-1] == "ZZ BOXREAD CONTROLS FAILED"
    assert re.search(r"^ctl %s FAIL \S" % ctl, out, re.M), out
    cj = json.load(open(os.path.join(info["dirs"]["out"], "BOXREAD_CONTROLS.json"), encoding="utf-8"))
    assert cj["all_pass"] is False and cj["controls"][ctl]["pass"] is False


def test_b1_fails_when_the_render_tool_test_fails(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(RB.BG, "load_enc_common", lambda path=None: object())
    monkeypatch.setattr(RB.BG, "render_roundtrip", lambda enc, win, az, el, w, h, **k: (az + 0.5, el, w, h))
    argv, _ = make_world(tmp_path / "w")
    rc, out = run_main(argv("controls"), capsys)
    assert rc == 2 and re.search(r"^ctl b1 FAIL", out, re.M)


def test_b7_names_the_count_of_wide_rows_whose_frame_was_not_run(tmp_path, capsys, fast_render):
    argv, info = make_world(tmp_path / "w", eval_notrun=True)
    run_main(argv("controls"), capsys)
    cj = json.load(open(os.path.join(info["dirs"]["out"], "BOXREAD_CONTROLS.json"), encoding="utf-8"))
    b7 = cj["controls"]["b7"]
    n = b7["n_rows_wide_frame_not_run"]
    assert n > 0 and any(("%d evaluation rows whose frame has frame_run false" % n) in x for x in b7["failures"])
    assert cj["controls"]["b5"]["pass"] is True                             # a not-run frame is not read as a no-pick by b5 either way, the failure is b7's


def test_clean_world_with_train_frames_not_run_passes_and_counts_them(tmp_path, capsys, fast_render):
    argv, info = make_world(tmp_path / "w", train_notrun=True)
    rc, _ = run_main(argv("controls"), capsys)
    cj = json.load(open(os.path.join(info["dirs"]["out"], "BOXREAD_CONTROLS.json"), encoding="utf-8"))
    assert rc == 0
    assert cj["controls"]["b1"]["n_lead_frames_not_run_left_out"] > 0 and cj["controls"]["b2"]["n_frames_not_run_left_out"] > 0


def test_train_frames_not_run_are_left_out_of_the_map(tmp_path):
    argv, info = make_world(tmp_path / "w", train_notrun=True)
    store = RB.DetStore(os.path.join(info["root"], "det"))
    reads = RB.train_reads(info["dirs"]["box"], store, info["tr"], "wide")
    lnh, lnw, gp, ci = RB.map_arrays(reads)
    n_all = sum(int(r["lead"].sum()) for r in reads)
    assert 0 < len(gp) < n_all
    assert all(r["run"][r["lead"] & np.isfinite(r["gap"]) & r["pick"]].all() for r in reads)


def test_second_controls_call_writes_a_new_file(tmp_path, capsys, fast_render):
    argv, info = make_world(tmp_path / "w")
    run_main(argv("controls"), capsys)
    run_main(argv("controls"), capsys)
    d = info["dirs"]["out"]
    assert os.path.exists(os.path.join(d, "BOXREAD_CONTROLS.json")) and os.path.exists(os.path.join(d, "BOXREAD_CONTROLS_2.json"))


# ============================================================================================ vision only (b4)
def test_loader_refuses_a_label_file_in_the_place_of_the_locator(tmp_path):
    argv, info = make_world(tmp_path / "w")
    with pytest.raises(RB.LabelFileInLocatorPlace):
        RB.load_locator(os.path.join(info["dirs"]["box"], info["ev"][0] + ".npz"))
    RB.load_locator(os.path.join(info["dirs"]["loc"], "LOC_s0.npz"))
    assert not ({"box_dir", "lab", "labels", "label", "box"} & set(RB.inspect.signature(RB.eval_pick_locator).parameters))


def test_arms_bw_bt_read_the_locator_direction_not_the_label(tmp_path):
    """moving the label direction changes BL and leaves BW alone; moving the locator's direction changes BW."""
    argv, info = make_world(tmp_path / "w")
    J, store = info["J"], RB.DetStore(os.path.join(info["root"], "det"))
    az = np.full(len(info["ev"]) * NFR, np.nan)
    el = np.full_like(az, np.nan)
    got = RB.eval_pick_locator(J, az, el, lambda cam, c: store.get(cam, c), lambda c: None, False)
    assert not got["wide"]["pick"].any()                                    # no locator output: no pick, whatever the labels say
    az2 = az_of(np.tile(np.arange(NFR), len(info["ev"])))
    got2 = RB.eval_pick_locator(J, az2, np.full_like(az2, EL0), lambda cam, c: store.get(cam, c), lambda c: None, False)
    assert got2["wide"]["pick"].any()


# ============================================================================================ the arms, with no pick and tele fallback
def _small_world():
    n = np.array([6, 6])
    J = {"clip": np.array(["a1", "b2"]), "n": n, "off": np.array([0, 6])}
    gap = np.full(12, 20.0)
    lab = np.full(12, 22.0)
    dumps = {k: {"gap": gap + i, "lab_gap": lab, "lab_present": np.ones(12, bool)} for i, k in enumerate(BS.ARM_D)}
    return J, dumps


def test_frames_with_no_pick_take_arm_d_and_bt_falls_back_to_bw():
    J, L = _small_world()
    G = BS.MapG(np.log(np.array([1.0, 2.0, 3.0, 4.0])), np.log(np.array([2.0, 3.0, 4.0, 5.0])), np.array([10.0, 20.0, 30.0, 40.0]), k=2)
    z = np.zeros(12, bool)
    base = {"pick": z.copy(), "run": np.ones(12, bool), "w": np.full(12, np.nan), "h": np.full(12, np.nan)}
    reads = {"wide": dict(base), "tele": dict(base), "BL": dict(base), "lab_w": np.full(12, 3.0), "lab_h": np.full(12, 2.0)}
    P = RB.compute_values(J, L, reads, G, G, 50)
    for i, k in enumerate(BS.ARM_D):
        assert P["E"]["BW_%d" % i] == pytest.approx(P["E"][k]) and P["E"]["BT_%d" % i] == pytest.approx(P["E"][k]) and P["E"]["BL_%d" % i] == pytest.approx(P["E"][k])
    wide = {"pick": np.arange(12) % 2 == 0, "run": np.ones(12, bool), "w": np.full(12, 3.0), "h": np.full(12, 2.0)}
    reads2 = dict(reads, wide=wide)
    P2 = RB.compute_values(J, L, reads2, G, G, 50)
    assert P2["E"]["BW_0"] != pytest.approx(P2["E"]["D_0"])                 # picked rows take the map's gap
    for i in range(3):
        assert P2["E"]["BT_%d" % i] == pytest.approx(P2["E"]["BW_%d" % i])  # no tele pick: BT is BW
    assert P2["beside"]["share_picked"]["BW"]["all"]["share"] == pytest.approx(0.5)


def test_frame_run_true_without_a_qualifying_detection_is_no_pick():
    n = 5
    d = {"n_frames": n, "frame_ok": np.ones(n, bool), "frame_run": np.array([True, True, False, True, True]), "det_frame": np.array([0, 1, 2, 3], np.int32),
         "det_box": np.array([box_for(6.0, EL0, 3.0, 2.0, WIDE)] * 4, np.float32), "det_score": np.array([0.9, 0.4, 0.9, 0.9], np.float32),
         "det_label": np.array([3, 3, 3, 1], np.int16), "win": np.asarray(WIDE)}
    pk = RB.pick_clip(d, np.full(n, 6.0), np.full(n, EL0), n)
    assert pk.tolist() == [0, -1, -1, -1, -1]                                 # frame 1 score 0.4, frame 2 not run, frame 3 a person, frame 4 nothing
    assert RB.run_of(d, n).tolist() == [True, True, False, True, True]


def test_tele_row_with_frame_run_false_has_no_tele_pick_and_falls_back(tmp_path, capsys, fast_render):
    argv, info = make_world(tmp_path / "w", tele=True, tele_notrun=True)
    store = RB.DetStore(os.path.join(info["root"], "det"))
    rot = RB.Rotations(os.path.join(info["dirs"]["cal"], "front_wide_extrinsics.csv"), os.path.join(info["dirs"]["cal"], "front_tele_extrinsics.csv"))
    az = az_of(np.tile(np.arange(NFR), len(info["ev"])))
    got = RB.eval_pick_locator(info["J"], az, np.full_like(az, EL0), lambda cam, c: store.get(cam, c), rot.rel, True)
    assert got["wide"]["pick"].any() and not got["tele"]["pick"].any() and not got["tele"]["run"].any()


# ============================================================================================ names reached on planted data
def controls_then_names(tmp_path, capsys, **opt):
    argv, info = make_world(tmp_path / "w", **opt)
    rc, _ = run_main(argv("controls"), capsys)
    assert rc == 0, "controls must all pass in this scenario"
    rc, out = run_main(argv("names"), capsys)
    res = json.load(open(os.path.join(info["dirs"]["out"], "BOXREAD_RESULT.json"), encoding="utf-8"))
    return rc, out, res, info


def names_of(out):
    return re.findall(r"^ZZ BOXREAD NAME (\S+)$", out, re.M)


def test_names_better_less_short_and_tele_does_not_add(tmp_path, capsys, fast_render):
    rc, out, res, info = controls_then_names(tmp_path, capsys)
    assert rc == 0
    nm = names_of(out)
    assert nm == res["names"] and len(nm) == 4
    assert "BOXREAD-T0-WIDE-BETTER" in nm and "BOXREAD-T0-FAR-LEADS-LESS-SHORT" in nm and "BOXREAD-T0-TELE-DOES-NOT-ADD" in nm and "BOXREAD-T0-TELE-BETTER" in nm   # no tele file: BT is BW
    for k in ("measures", "beside", "counts", "input_hashes", "names"):
        assert k in res
    v = open(os.path.join(info["dirs"]["out"], "BOXREAD_VERDICT.md"), encoding="utf-8").read()
    assert "recorded frames, no rollout; a diagnostic of a perception read and not driving performance" in v
    for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
        assert fam in v
    assert "R = " in v and "## Limits of the card" in v and "## Beside, naming nothing" in v


def test_names_worse_and_far_leads_stay_short(tmp_path, capsys, fast_render):
    rc, out, res, _ = controls_then_names(tmp_path, capsys, eval_scale=0.5)
    nm = names_of(out)
    assert rc == 0 and "BOXREAD-T0-WIDE-WORSE" in nm and "BOXREAD-T0-FAR-LEADS-STAY-SHORT" in nm

class _IdentityMap:
    """a stand-in map whose gap is the 'width' it is given: the crafted reads below choose the gap each arm reads."""

    def predict(self, w, h):
        return np.asarray(w, np.float64).copy()


def _crafted(bw_err):
    """40 clips x 10 rows, label gap 30 m, arm D reads 34 m on every row (error 4); BW reads label + bw_err[clip], all rows picked."""
    nc, nr = 40, 10
    J = {"clip": np.array(["k%02d" % i for i in range(nc)]), "n": np.full(nc, nr), "off": np.arange(nc) * nr}
    lab = np.full(nc * nr, 30.0)
    L = {k: {"gap": lab + 4.0, "lab_gap": lab, "lab_present": np.ones(nc * nr, bool)} for k in BS.ARM_D}
    err = np.repeat(np.asarray(bw_err, np.float64), nr)
    one = {"pick": np.ones(nc * nr, bool), "run": np.ones(nc * nr, bool), "w": lab + err, "h": np.full(nc * nr, 2.0)}
    none = {"pick": np.zeros(nc * nr, bool), "run": np.ones(nc * nr, bool), "w": np.full(nc * nr, np.nan), "h": np.full(nc * nr, np.nan)}
    reads = {"wide": one, "tele": none, "BL": none, "lab_w": np.full(nc * nr, 3.0), "lab_h": np.full(nc * nr, 2.0)}
    return J, L, reads


def test_names_small_gain_and_not_separated_on_crafted_reads():
    J, L, reads = _crafted(np.full(40, 3.7))                                  # every clip: BW error 3.7 against D's 4.0 -> R 0.925, interval degenerate below 1
    P = RB.compute_values(J, L, reads, _IdentityMap(), None, 200)
    assert P["R"]["BW"] == pytest.approx(0.925) and "BOXREAD-T0-WIDE-SMALL-GAIN" in RB.derive_names(P)
    J, L, reads = _crafted(np.where(np.arange(40) % 2 == 0, 2.0, 6.0))        # half the clips better, half worse: R 1.0, interval spans 1
    P = RB.compute_values(J, L, reads, _IdentityMap(), None, 200)
    assert P["R"]["BW"] == pytest.approx(1.0) and "BOXREAD-T0-WIDE-NOT-SEPARATED" in RB.derive_names(P)


def test_names_tele_better_and_adds_where_the_wide_read_is_noisy(tmp_path, capsys, fast_render):
    rc, out, res, _ = controls_then_names(tmp_path, capsys, tele=True, wide_noise=0.5)
    nm = names_of(out)
    assert rc == 0 and "BOXREAD-T0-TELE-ADDS" in nm and "BOXREAD-T0-TELE-BETTER" in nm, (nm, res["measures"]["R"], res["measures"]["iv"])
    assert res["counts"]["n_eval_rows_with_tele_pick"] > 0


def test_every_name_of_the_card_is_reachable_from_the_rule():
    cases = {("WIDE", 0.5, (0.4, 0.6)): "BOXREAD-T0-WIDE-BETTER", ("WIDE", 0.95, (0.9, 0.99)): "BOXREAD-T0-WIDE-SMALL-GAIN", ("WIDE", 1.2, (1.1, 1.3)): "BOXREAD-T0-WIDE-WORSE",
             ("WIDE", 1.0, (0.9, 1.1)): "BOXREAD-T0-WIDE-NOT-SEPARATED", ("TELE", 0.5, (0.4, 0.6)): "BOXREAD-T0-TELE-BETTER", ("TELE", 0.95, (0.9, 0.99)): "BOXREAD-T0-TELE-SMALL-GAIN",
             ("TELE", 1.2, (1.1, 1.3)): "BOXREAD-T0-TELE-WORSE", ("TELE", 1.0, (0.9, 1.1)): "BOXREAD-T0-TELE-NOT-SEPARATED"}
    for (tag, R, iv), want in cases.items():
        assert RB.name_ratio(tag, iv, R) == want
    assert RB.name_ratio("WIDE", [0.5, 0.9], 0.90) == "BOXREAD-T0-WIDE-BETTER" and RB.name_ratio("WIDE", [0.5, 0.9], 0.9001) == "BOXREAD-T0-WIDE-SMALL-GAIN"
    assert RB.name_far(-1.0, -6.25, [0.5, 2.0]) == "BOXREAD-T0-FAR-LEADS-LESS-SHORT"
    assert RB.name_far(-3.2, -6.25, [0.5, 2.0]) == "BOXREAD-T0-FAR-LEADS-STAY-SHORT"     # more than half of D's size
    assert RB.name_far(-1.0, -6.25, [-0.5, 2.0]) == "BOXREAD-T0-FAR-LEADS-STAY-SHORT"    # interval not above 0


# ============================================================================================ refusals and the files
def test_names_refuses_without_a_controls_file_and_without_an_all_pass_file(tmp_path, capsys, fast_render):
    argv, info = make_world(tmp_path / "w")
    rc, out = run_main(argv("names"), capsys)
    assert rc == 3 and "ZZ BOXREAD NAME" not in out
    argv2, info2 = make_world(tmp_path / "w2", train_dead=True)
    run_main(argv2("controls"), capsys)
    rc, out = run_main(argv2("names"), capsys)
    assert rc == 3 and "ZZ BOXREAD NAME" not in out
    assert not os.path.exists(os.path.join(info2["dirs"]["out"], "BOXREAD_RESULT.json"))


def test_names_refuses_when_an_input_changed_after_the_controls(tmp_path, capsys, fast_render):
    argv, info = make_world(tmp_path / "w")
    run_main(argv("controls"), capsys)
    p = os.path.join(info["dirs"]["det/wide"], info["ev"][0] + ".npz")
    BS.wt(os.path.join(info["dirs"]["det/wide"], "extra.npz"), "x")           # a changed listing of the detector folder
    rc, out = run_main(argv("names"), capsys)
    assert rc == 3 and "inputs differ" in out and "ZZ BOXREAD NAME" not in out


def test_second_names_call_writes_a_new_file(tmp_path, capsys, fast_render):
    rc, out, res, info = controls_then_names(tmp_path, capsys)
    rc2, out2 = run_main(info["argv"]("names"), capsys)
    d = info["dirs"]["out"]
    assert rc2 == 0 and os.path.exists(os.path.join(d, "BOXREAD_RESULT_2.json")) and os.path.exists(os.path.join(d, "BOXREAD_VERDICT_2.md"))


def test_cli_refusal_exit_code_and_lines_in_a_child_process(tmp_path):
    argv, info = make_world(tmp_path / "w")
    cwd = os.environ.get("BOXREAD_TEST_CWD") or str(tmp_path)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1", BOXSIZE_READER=BS.__boxread_path__)
    p = subprocess.run([sys.executable, os.path.join(HERE, "read_boxread.py")] + argv("names"), cwd=cwd, env=env, capture_output=True, encoding="utf-8")
    assert p.returncode == 3 and "REFUSED" in p.stdout and "ZZ BOXREAD NAME" not in p.stdout


# ============================================================================================ files of the kit
def test_driver_and_defs_are_present_and_clean():
    sh = open(os.path.join(HERE, "podread_boxread.sh"), encoding="utf-8").read()
    assert "ZZ POD-BOXREAD" in sh and "read_args.txt" in sh
    for bad in (r"\brm\b", r"\bmv\b", r"\bkill\b", r"\bpkill\b"):
        assert not re.search(bad, sh.split("\n", 1)[1]), bad
    banned = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lon" + "ger", "closed " + "loop"]
    for fn in ("box_geom.py", "read_boxread.py", "test_read_boxread.py", "podread_boxread.sh", "BOXREAD_READ_DEFS.md", "read_args.txt"):
        t = open(os.path.join(HERE, fn), encoding="utf-8").read().lower()
        for w in banned:
            assert w not in t, (fn, w)
        assert len(fn) < 100 and not re.match(r"^(aux|con|prn|nul)\b", fn)


def test_zz_imported_readers_are_byte_unchanged_after_the_tests():
    assert BS.sha_file(BS.__boxread_path__).startswith(RB.BS_PREFIX)
    for k, p in IMPORTED.items():
        if p:
            assert _sha(p) == HASH_AT_IMPORT[k], k
