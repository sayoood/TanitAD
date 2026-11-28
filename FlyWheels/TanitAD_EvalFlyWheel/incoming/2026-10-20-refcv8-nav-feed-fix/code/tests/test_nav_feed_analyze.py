"""CPU tests for nav_feed_analyze.py: the committed rule (pure function) and an end-to-end run on SYNTHETIC dumps with PLANTED effects.

    cd <hold>/code/tests
    CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 NF_ANALYZE_TEST=1 \
      PYTHONPATH='<hold>/code/overlay;<hold>/code;C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' \
      C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q -p no:cacheprovider test_nav_feed_analyze.py

A planted effect is the discriminating control: the analysis must (1) say MATTERS when -0.5 m is planted on the mismatch windows of the full arm,
(2) say NULL when nothing is planted, (3) refuse to call a label admissible when the token arm moves the CONTROL windows, and (4) refuse an unsealed
prereg.  Literal expectations; no value is derived from the code under test.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import nav_feed_analyze as NA

W_SLOTS = [5, 10, 15, 20, 30, 40, 50, 60]
V = 8.0                                   # m/s, straight line


# =============================================================================================== the rule (pure)
def cell(delta, lo, hi):
    return {"delta": delta, "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0)}


FLOOR = {"F": 0.02}
CTRL_OK = {0: {"abs_delta": 0.0}, 1: {"abs_delta": 0.03}}          # 0.03 <= 2 * 0.02
CTRL_BAD = {0: {"abs_delta": 0.0}, 1: {"abs_delta": 0.0401}}       # 0.0401 > 0.04


@pytest.mark.parametrize("prim,ctrl,label,admissible", [
    ({0: cell(-0.5, -0.8, -0.2), 1: cell(-0.45, -0.7, -0.1)}, CTRL_OK, "FEED-FIX-MATTERS", True),
    ({0: cell(-0.5, -0.8, -0.2), 1: cell(-0.45, -0.7, -0.1)}, CTRL_BAD, "MIXED", False),                # MATTERS pattern, control failed
    ({0: cell(-0.05, -0.2, 0.1), 1: cell(0.02, -0.1, 0.15)}, CTRL_OK, "FEED-FIX-NULL", True),
    ({0: cell(-0.05, -0.2, 0.1), 1: cell(0.02, -0.1, 0.15)}, CTRL_BAD, "FEED-FIX-NULL", False),           # literal rule; flagged NOT admissible
    ({0: cell(-0.5, -0.8, -0.2), 1: cell(-0.1, -0.3, 0.1)}, CTRL_OK, "MIXED", True),                     # separated at one seed only
    ({0: cell(0.5, 0.2, 0.8), 1: cell(0.45, 0.1, 0.7)}, CTRL_OK, "MIXED", True),                         # separated WORSE at both: not NULL, not MATTERS
    ({0: cell(-0.5, -0.8, -0.2), 1: cell(0.45, 0.1, 0.7)}, CTRL_OK, "MIXED", True),                      # opposite signs
    ({0: cell(-0.015, -0.03, -0.001), 1: cell(-0.5, -0.8, -0.2)}, CTRL_OK, "MIXED", True),               # separated, but |delta| 0.015 <= F 0.02 at seed 0: no effect
    ({0: cell(-0.02, -0.03, -0.001), 1: cell(-0.02, -0.03, -0.001)}, CTRL_OK, "MIXED", True),            # |delta| == F is NOT above the floor (strict)
    ({0: cell(-0.0201, -0.03, -0.001), 1: cell(-0.0201, -0.03, -0.001)}, CTRL_OK, "FEED-FIX-MATTERS", True),   # just above F, both seeds
])
def test_decide_truth_table(prim, ctrl, label, admissible):
    v = NA.decide(prim, FLOOR, ctrl)
    assert v["label"] == label and v["label_admissible"] is admissible


def test_decide_not_evaluable_when_a_primary_cell_is_absent():
    assert NA.decide({0: cell(-0.5, -0.8, -0.2), 1: None}, FLOOR, CTRL_OK)["label"] == "NOT-EVALUABLE"
    assert NA.decide({0: cell(-0.5, -0.8, -0.2), 1: {"note": "too few"}}, FLOOR, CTRL_OK)["label"] == "NOT-EVALUABLE"


def test_decide_boundary_is_strict_hi_below_zero():
    assert NA.decide({0: cell(-0.1, -0.3, 0.0), 1: cell(-0.1, -0.3, 0.0)}, FLOOR, CTRL_OK)["separated_negative_by_seed"] == [False, False]
    assert NA.decide({0: cell(-0.1, -0.3, -1e-9), 1: cell(-0.1, -0.3, -1e-9)}, FLOOR, CTRL_OK)["label"] == "FEED-FIX-MATTERS"


def test_the_rule_constants_are_the_committed_ones():
    assert (NA.PRIMARY_METRIC, NA.FLOOR_MULT, NA.N_BOOT, NA.BOOT_SEED, NA.SEEDS) == ("ade_6s", 2.0, 2000, 0, (0, 1))
    assert NA.ARMS == ("clip", "perframe_token", "perframe_full")
    assert NA.SLOT_STEPS == [5, 10, 15, 20, 30, 40, 50, 60] and NA.S6_SLOTS == [1, 3, 4, 5, 6, 7] and NA.S6_GT_IDX == [9, 19, 29, 39, 49, 59]


def test_gt_class_and_plan_class_literals():
    v0 = np.array([8.0, 8.0, 8.0, 0.5, 8.0])
    v6 = np.array([8.5, 4.0, 11.0, 3.0, 0.5])
    ok = np.ones(5, bool)
    assert NA.gt_class(v0, v6, ok).tolist() == ["cruising", "decelerating", "accelerating", "start_from_stop", "stopping"]
    P6 = np.zeros((2, 6, 2))
    P6[0, -1, 0], P6[0, -2, 0] = 20.0, 12.0          # last-second speed 8 m/s vs v0 8 -> cruise
    P6[1, -1, 0], P6[1, -2, 0] = 20.0, 17.0          # 3 m/s vs v0 8 -> decel
    assert NA.plan_class(P6, np.array([8.0, 8.0])).tolist() == ["cruise", "decel"]


# =============================================================================================== synthetic dumps
def write_dump(root: Path, name: str, seed: int, rows: list, plan_shift: dict, arm: str, rng_noise=0.0, nav_clip=None, nav_tok=None, known=None):
    """rows: [(episode_index, ws)] in order.  plan_shift[(ep, ws)] = (dx, dy) added to the GT-aligned plan of every slot."""
    d = root / name
    (d / "decisions").mkdir(parents=True, exist_ok=True)
    by_ep = {}
    for r in rows:
        by_ep.setdefault(r[0], []).append(r)
    eps = []
    rs = np.random.default_rng(1000 + seed)
    for fi, (ei, rr) in enumerate(sorted(by_ep.items())):
        n = len(rr)
        ws = np.array([w for _e, w in rr], dtype=np.int64)
        t = np.arange(1, 61) * 0.1
        gt = np.zeros((n, 60, 4))
        gt[..., 0] = V * t
        gt[..., 3] = V
        pose = np.zeros((n, 4))
        pose[:, 3] = V
        gt_ego = np.stack([V * t[np.array([h - 1 for h in W_SLOTS])]] * n)          # [n, 8]
        plan = np.zeros((n, 8, 2))
        plan[..., 0] = gt_ego
        for i, (_e, w) in enumerate(rr):
            dx, dy = plan_shift.get((ei, w), (0.0, 0.0))
            plan[i, :, 0] += dx
            plan[i, :, 1] += dy
        plan += rs.normal(0.0, rng_noise, plan.shape)
        g2 = np.zeros((n, 4, 2))
        g2[..., 0] = gt_ego[:, :4]
        np.savez_compressed(d / f"ep{fi:03d}.npz", ws=ws, eid=np.array([fi]), clip_index=np.array([ei]), v0=np.full(n, V, np.float32),
                            g=g2.astype(np.float32), os=plan[:, :4].astype(np.float32))
        dec = {"ws": ws, "plan_full_nav_true": plan[:, None].astype(np.float32).reshape(n, 8, 2), "gt_future_ext": gt.astype(np.float32),
               "gt_future_valid_ext": np.ones((n, 60), np.float32), "pose_last": pose.astype(np.float32),
               "nav_cmd": np.array([(nav_tok if arm != "clip" else nav_clip)[(ei, int(w))] for w in ws], dtype=np.int64),
               "lat_pred_nav_true": np.zeros(n, np.int64), "lon_pred_nav_true": np.zeros(n, np.int64), "route_pred_nav_true": np.zeros(n, np.int64),
               "lat_label": np.zeros(n, np.int64), "lon_label": np.zeros(n, np.int64), "route_label": np.zeros(n, np.int64)}
        if arm != "clip":
            dec["nav_cmd_clip"] = np.array([nav_clip[(ei, int(w))] for w in ws], dtype=np.int64)
            dec["r8_nav_known"] = np.array([known[(ei, int(w))] for w in ws], dtype=bool)
            fed = np.zeros((n, 6), np.float32)
            if arm == "perframe_full":
                fed[dec["r8_nav_known"]] = 0.5
            dec["r8_nav_fed"] = fed
        np.savez_compressed(d / "decisions" / f"ep{fi:03d}.npz", **dec)
        eps.append({"file_index": fi, "episode_index": ei, "clip_id": "synthetic", "episode_id": ei, "n_windows": n})
    man = {"episodes": eps, "inference_seed": {"seed": seed}, "model": {"step": 40000, "ckpt": "synthetic"}}
    if arm != "clip":
        man["nav_feed"] = {"mode": arm}
    json.dump(man, open(d / "manifest.json", "w", encoding="utf-8"))


def build_world(tmp: Path, *, effect_full=-0.5, ctrl_shift=0.0, noise=0.01, full_effect_seed1=None):
    """8 episodes x 8 windows: ws 10..45; mismatch = ws in {10,15,20,25}, control = ws in {30,35,40,45}."""
    rows, mism_ws, ctrl_ws = [], {}, {}
    for ei in range(8):
        for k in range(8):
            ws = 10 + 5 * k
            rows.append((ei, ws))
            (mism_ws if k < 4 else ctrl_ws).setdefault(ei, []).append(ws)
    nav_clip = {(ei, ws): (2 if ws in mism_ws[ei] else 0) for ei, ws in rows}
    nav_tok = {(ei, ws): (0 if ws in mism_ws[ei] else 0) for ei, ws in rows}
    known = {(ei, ws): bool(ws % 10 == 0) for ei, ws in rows}
    win = {"episodes": [{"episode_index": ei, "sha12": f"{ei:012x}", "mismatch_ws": mism_ws[ei], "control_ws": ctrl_ws[ei],
                         "mismatch_fed": [2] * 4, "mismatch_v9": [0] * 4, "control_v9": [0] * 4} for ei in range(8)]}
    wp = tmp / "windows.json"
    json.dump(win, open(wp, "w"))
    out = tmp / "out"
    out.mkdir()
    for seed in (0, 1):
        eff = effect_full if (seed == 0 or full_effect_seed1 is None) else full_effect_seed1
        shift_full = {(ei, ws): (eff, 0.0) for ei in range(8) for ws in mism_ws[ei]}
        shift_tok = {(ei, ws): (ctrl_shift, 0.0) for ei in range(8) for ws in ctrl_ws[ei]}
        write_dump(out, f"dump_clip_s{seed}", seed, rows, {}, "clip", noise, nav_clip, nav_tok, known)
        write_dump(out, f"dump_perframe_full_s{seed}", seed, rows, {**shift_full, **shift_tok}, "perframe_full", noise, nav_clip, nav_tok, known)
        write_dump(out, f"dump_perframe_token_s{seed}", seed, rows, shift_tok, "perframe_token", noise, nav_clip, nav_tok, known)
    return win, wp, out


def run_analysis(tmp, wp, out, monkeypatch, seal=True):
    pre = tmp / "PREREG_NAV_FEED.md"
    pre.write_text("synthetic prereg", encoding="utf-8")
    (tmp / "PREREG_NAV_FEED.md.sha256").write_text(hashlib.sha256(pre.read_bytes()).hexdigest() + "  PREREG_NAV_FEED.md\n", encoding="utf-8")
    if not seal:
        (tmp / "PREREG_NAV_FEED.md.sha256").write_text("0" * 64 + "  PREREG_NAV_FEED.md\n", encoding="utf-8")
        monkeypatch.delenv("NF_ANALYZE_TEST", raising=False)
    NA.main(["--out-dir", str(out), "--windows", str(wp), "--prereg", str(pre), "--archive", str(tmp / "no_archive"), "--n-boot", "300"])
    return json.load(open(out / "NAV_FEED.json", encoding="utf-8"))


def test_planted_worsening_is_read_as_MIXED_never_MATTERS(tmp_path, monkeypatch):
    """The full arm's plan is moved 0.5 m OFF the GT on 32 mismatch windows: perframe_full - clip = +0.5 (WORSE).  Separated, but the wrong way:
    the committed rule must not call that MATTERS (and not NULL either, because it IS separated)."""
    monkeypatch.setenv("NF_ANALYZE_TEST", "1")
    _w, wp, out = build_world(tmp_path, effect_full=-0.5)
    r = run_analysis(tmp_path, wp, out, monkeypatch)
    v = r["committed_label"]
    assert v["label"] == "MIXED" and v["label_admissible"] is True and v["separated_negative_by_seed"] == [False, False]
    assert v["separated_any_by_seed"] == [True, True]
    for s in ("s0", "s1"):
        c = r["primary_cells_perframe_full_minus_clip_ade6s_mismatch"][s]
        assert c["n_windows"] == 32 and c["n_episodes"] == 8 and c["delta"] == pytest.approx(0.5, abs=0.05) and c["separated"] is True
    assert r["controls"]["windows_identical_across_dumps_and_equal_to_the_sealed_union"] is True


def test_planted_improvement_sign_convention(tmp_path, monkeypatch):
    """Make the CLIP arm the wrong one (it is 0.5 m off), the full arm exact: perframe_full - clip must be NEGATIVE and read MATTERS."""
    monkeypatch.setenv("NF_ANALYZE_TEST", "1")
    _w, wp, out = build_world(tmp_path, effect_full=0.0)
    # rewrite the clip dumps with a +0.5 m error on the mismatch windows
    win = json.load(open(wp))
    rows = [(e["episode_index"], w) for e in win["episodes"] for w in sorted(e["mismatch_ws"] + e["control_ws"])]
    nav_clip = {(e["episode_index"], w): (2 if w in e["mismatch_ws"] else 0) for e in win["episodes"] for w in e["mismatch_ws"] + e["control_ws"]}
    nav_tok = {k: 0 for k in nav_clip}
    known = {k: bool(k[1] % 10 == 0) for k in nav_clip}
    bad = {(e["episode_index"], w): (0.5, 0.0) for e in win["episodes"] for w in e["mismatch_ws"]}
    for s in (0, 1):
        write_dump(out, f"dump_clip_s{s}", s, rows, bad, "clip", 0.01, nav_clip, nav_tok, known)
    r = run_analysis(tmp_path, wp, out, monkeypatch)
    for s in ("s0", "s1"):
        c = r["primary_cells_perframe_full_minus_clip_ade6s_mismatch"][s]
        assert c["delta"] < -0.4 and c["separated"] is True
    assert r["committed_label"]["label"] == "FEED-FIX-MATTERS"


def test_no_effect_is_read_as_NULL(tmp_path, monkeypatch):
    monkeypatch.setenv("NF_ANALYZE_TEST", "1")
    _w, wp, out = build_world(tmp_path, effect_full=0.0, noise=0.05)
    r = run_analysis(tmp_path, wp, out, monkeypatch)
    assert r["committed_label"]["label"] == "FEED-FIX-NULL"


def test_a_control_that_moves_is_flagged_not_admissible(tmp_path, monkeypatch):
    """Nothing but a +0.3 m shift of the token arm on the CONTROL windows (where the tokens are identical): the instrument is broken."""
    monkeypatch.setenv("NF_ANALYZE_TEST", "1")
    _w, wp, out = build_world(tmp_path, effect_full=-0.5, ctrl_shift=0.3)
    r = run_analysis(tmp_path, wp, out, monkeypatch)
    v = r["committed_label"]
    assert v["control_ok"] is False and v["label_admissible"] is False and "NOT an admissible result" in v["admissibility_note"]
    assert v["label"] != "FEED-FIX-MATTERS"


def test_unsealed_prereg_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("NF_ANALYZE_TEST", "1")
    _w, wp, out = build_world(tmp_path)
    with pytest.raises(SystemExit):
        run_analysis(tmp_path, wp, out, monkeypatch, seal=False)


def test_missing_dump_is_reported_not_silently_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv("NF_ANALYZE_TEST", "1")
    _w, wp, out = build_world(tmp_path)
    import shutil
    shutil.rmtree(out / "dump_perframe_full_s1")
    r = run_analysis(tmp_path, wp, out, monkeypatch)
    assert "perframe_full_s1" in r["controls"]["missing_dumps"]
    assert r["committed_label"]["label"] == "NOT-EVALUABLE"


def test_frame_check_and_token_controls_read_true_on_clean_dumps(tmp_path, monkeypatch):
    monkeypatch.setenv("NF_ANALYZE_TEST", "1")
    _w, wp, out = build_world(tmp_path)
    r = run_analysis(tmp_path, wp, out, monkeypatch)
    for nm, c3 in r["controls"]["C3_gt_and_dump_consistency"].items():
        assert c3["gt_ego_frame_vs_banked_g_max_abs_m"] < 1e-4 and c3["os_2s_vs_plan_full_first4_max_abs_m"] < 1e-4
    t = r["controls"]["token_feed_as_designed"]
    assert t["clip_s0"]["nav_cmd_equals_the_banked_clip_token"] is True
    assert t["perframe_full_s0"]["nav_cmd_equals_the_v9_per_frame_token"] is True
    assert t["perframe_full_s0"]["r8_nav_fed_nonzero_rows_equal_known_rows"] is True
    assert t["perframe_token_s0"]["r8_nav_fed_nonzero_rows_equal_known_rows"] is True        # token arm: nothing fed
