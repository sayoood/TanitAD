"""TanitResim tests — exporter bundle schema + FastAPI single-port server.

CPU-only, no model checkpoints: synthetic :class:`TimestepRecord` streams
(main-like + refb-like arm outputs, random uint8 camera frames) exercise the
full export path, then :class:`fastapi.testclient.TestClient` drives the
server against the written bundle. Pins:

(a) exporter writes a portable bundle — session.json schema + one JPEG per
    step, arm/episode summaries, branded colors;
(b) portability — no absolute paths leak into session.json (checkpoints are
    stored basename-only, frame refs are relative);
(c) fail-loud — empty stream and frameless records raise;
(d) frame downscale — wide frames are capped at max_w;
(e) server — index served, sessions list, session fetch (+404), frame fetch
    (+404 guards), static SPA asset served.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.replay.engine import WAYPOINT_STEPS, ArmOutput, TimestepRecord
from tanitad.resim.export import RESIM_COLORS, export_bundle, resim_color

MANEUVERS = ("lane_keep", "turn_left", "turn_right", "accelerate", "brake_stop")


# ---------- synthetic records -------------------------------------------------

def _frame(step: int, h: int = 48, w: int = 64) -> np.ndarray:
    g = np.random.default_rng(step)
    return g.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def _main_out(scale: float) -> ArmOutput:
    wp = np.array([[k * 0.8 * scale, 0.1 * k] for k in WAYPOINT_STEPS],
                  dtype=np.float64)
    return ArmOutput(latency_ms=12.0, waypoints=wp,
                     action=np.array([0.12, -0.1]),
                     imag_rel={1: 0.5, 4: 0.9}, sigma=0.3,
                     imag_traj={1: np.array([0.8, 0.0])})


def _refb_out(scale: float) -> ArmOutput:
    wp = np.array([[k * 0.8 * scale, -0.05 * k] for k in WAYPOINT_STEPS],
                  dtype=np.float64)
    return ArmOutput(latency_ms=5.0, waypoints=wp,
                     action=np.array([0.05, 0.2]),
                     action_seq=np.zeros((3, 2)),
                     maneuver_probs=np.array([0.6, 0.1, 0.1, 0.1, 0.1]),
                     maneuver_gt=0, nav_cmd=1, conf=0.7, ood=0.2)


def _rec(step, ep, t, hw=(48, 64)) -> TimestepRecord:
    gt = np.array([[k * 0.8, 0.05 * k] for k in WAYPOINT_STEPS],
                  dtype=np.float64)
    sc = 1.0 + 0.1 * t
    return TimestepRecord(
        step=step, corpus="toy-val", episode_id=ep, ep_index=ep, t=t,
        gt_waypoints=gt, gt_action=np.array([0.1, -0.2]),
        speed=8.0, yaw_rate=0.05,
        arms={"main": _main_out(sc), "refb": _refb_out(sc)},
        frame=_frame(step, *hw))


def _records(n_ep=2, n_step=5):
    recs, step = [], 0
    for ep in range(n_ep):
        for j in range(n_step):
            recs.append(_rec(step, ep, 3 + j))
            step += 1
    return recs


def _build(tmp_path, name="sess1", **kw):
    bundle = tmp_path / name
    session = export_bundle(
        _records(), bundle, "Demo Session",
        corpora=["toy-val"],
        arm_ckpts={"main": "/workspace/exp/main.pt",
                   "refb": r"C:\ckpts\refb.pt"},
        maneuver_classes=MANEUVERS, **kw)
    return bundle, session


# ---------- (a) exporter bundle schema ---------------------------------------

def test_export_builds_valid_bundle(tmp_path):
    bundle, session = _build(tmp_path)

    assert (bundle / "session.json").is_file()
    frames_dir = bundle / "frames"
    assert frames_dir.is_dir()
    n_steps = sum(len(ep["steps"]) for ep in session["episodes"])
    assert n_steps == 10
    assert len(list(frames_dir.glob("*.jpg"))) == n_steps

    meta = session["meta"]
    assert meta["session_name"] == "Demo Session"
    assert meta["corpora"] == ["toy-val"]
    assert meta["waypoint_steps"] == list(WAYPOINT_STEPS)
    assert list(meta["maneuver_classes"]) == list(MANEUVERS)

    names = [a["name"] for a in meta["arms"]]
    assert names == ["main", "refb"]
    for a in meta["arms"]:
        assert a["color"] == RESIM_COLORS[a["name"]]
        assert a["ade"] is not None and a["fde"] is not None
        assert a["latency_p50"] > 0
        assert a["ckpt"] in ("main.pt", "refb.pt")     # basename only

    assert len(meta["episodes"]) == 2
    for em in meta["episodes"]:
        assert set(em) >= {"idx", "corpus_tag", "n_steps", "per_arm_ade",
                           "worst_step", "worst_ade", "thumb"}
        assert em["n_steps"] == 5
        assert set(em["per_arm_ade"]) == {"main", "refb"}
        assert 0 <= em["worst_step"] < em["n_steps"]
        assert (bundle / "frames" / em["thumb"]).is_file()


def test_export_step_schema_and_heads(tmp_path):
    bundle, session = _build(tmp_path)
    step0 = session["episodes"][0]["steps"][0]

    assert (bundle / "frames" / step0["frame"]).is_file()
    # GT paths are origin-prefixed (len == waypoints + 1), 2-D points.
    assert len(step0["gt_wp_bev"]) == len(WAYPOINT_STEPS) + 1
    assert len(step0["gt_wp_img"]) == len(WAYPOINT_STEPS) + 1
    assert all(len(p) == 2 for p in step0["gt_wp_img"])
    assert set(step0["gt_action"]) == {"steer", "accel"}
    assert set(step0["ego"]) == {"speed", "yaw_rate"}

    main = step0["arms"]["main"]
    assert main["wp_img"] is not None and main["wp_bev"] is not None
    assert main["ade"] is not None
    assert "imag_rel" in main["heads"] and "sigma" in main["heads"]
    assert set(main["heads"]["imag_rel"]) == {"1", "4"}   # str keys, JSON-safe

    refb = step0["arms"]["refb"]["heads"]
    assert refb["conf"] == 0.7 and refb["ood"] == 0.2
    assert len(refb["maneuver_probs"]) == len(MANEUVERS)
    assert refb["maneuver_gt"] == 0 and refb["nav_cmd"] == 1


# ---------- (a2) formal-gate panel data (compare_arms integration) -----------

def test_export_attaches_formal_gates(tmp_path):
    """arm_gates + gates_summary (from compare_arms.compact_gate_blocks) flow
    into the bundle so the UI gate panel + GO banner have their data."""
    arm_gates = {
        "main": {"D1": "PASS", "d1_ade_0_2s": 0.42,
                 "oracle_ceiling_ade_0_2s": 0.30, "D2": "PASS",
                 "d2_dir_acc": 0.88, "D3": "PASS", "d3_ratio": 1.1,
                 "grounded_ade_0_2s": 0.55, "grounded_beats_cv": True},
        "refb": {"D1": "FAIL", "d1_ade_0_2s": 1.9,
                 "oracle_ceiling_ade_0_2s": 0.31, "D2": "N/A",
                 "d2_dir_acc": None, "D3": "N/A", "d3_ratio": None,
                 "grounded_ade_0_2s": 2.1, "grounded_beats_cv": False},
    }
    gates_summary = {
        "baselines": {"constant_velocity": 0.28, "go_straight": 0.31,
                      "constant_yaw_rate": 0.30},
        "verdict": {"per_metric": {"d1_decode_ade_0_2s":
                                   {"winner_lowest": "main"}}},
        "n_val_episodes": 12, "n_windows": 240,
        "camera_ade_max_m": 1.0, "oracle_ceiling_target_m": 1.65}
    bundle, session = _build(tmp_path, "gated", arm_gates=arm_gates,
                             gates_summary=gates_summary)
    meta = session["meta"]
    assert meta["gates"] == gates_summary
    by = {a["name"]: a for a in meta["arms"]}
    assert by["main"]["gates"]["D1"] == "PASS"
    assert by["main"]["gates"]["grounded_beats_cv"] is True
    assert by["refb"]["gates"]["D2"] == "N/A"


def test_export_gates_optional(tmp_path):
    """A bundle exported without gates is still valid (overlays-only)."""
    bundle, session = _build(tmp_path, "nogates")
    assert session["meta"]["gates"] is None
    for a in session["meta"]["arms"]:
        assert a["gates"] is None


def test_worst_step_tracks_largest_error(tmp_path):
    # ADE grows with t (scale 1+0.1t), so the last step of each episode is worst.
    _, session = _build(tmp_path)
    for em in session["meta"]["episodes"]:
        assert em["worst_step"] == em["n_steps"] - 1


# ---------- (h) kinematic maneuver labels from ego poses ----------------------
# The bundle carries a per-window maneuver class (scripts/refb_labels) so the
# SPA can draw the maneuver badge + timeline strip. Computed from GROUND-TRUTH
# ego poses at export time (arm-independent), null when no poses are supplied.

def _poses(T: int, yaw_rate: float = 0.0, v: float = 8.0) -> np.ndarray:
    """Synthetic ego poses [T, 4] = (x, y, yaw, v). Only yaw (col 2) and v
    (col 3) drive the maneuver class; positions are left at origin."""
    yaw = yaw_rate * 0.1 * np.arange(T)          # dt = 0.1 s (10 Hz contract)
    return np.stack([np.zeros(T), np.zeros(T), yaw, np.full(T, v)],
                    axis=1).astype(np.float64)


def test_export_maneuvers_from_ego_poses(tmp_path):
    import refb_labels as rl

    # ep0 yaws +0.4 rad over the 2 s horizon (> 0.15) -> turn_left throughout;
    # ep1 drives straight at constant speed -> lane_keep throughout. _records()
    # anchors both episodes' 5 steps at t = 3..7.
    ego_poses = {0: _poses(40, yaw_rate=0.2), 1: _poses(40, yaw_rate=0.0)}
    session = export_bundle(
        _records(), tmp_path / "man", "Maneuver Session",
        corpora=["toy-val"], maneuver_classes=MANEUVERS, ego_poses=ego_poses)

    assert list(session["meta"]["maneuver_classes"]) == list(MANEUVERS)
    ep0, ep1 = session["episodes"]
    assert [s["maneuver"] for s in ep0["steps"]] == [rl.TURN_LEFT] * 5
    assert [s["maneuver"] for s in ep1["steps"]] == [rl.LANE_KEEP] * 5

    # Exact match to refb_labels.maneuver_labels indexed by the anchor t (=3+j)
    # — the SAME kinematic label REF-B targets, computed once for display.
    import torch
    labs = rl.maneuver_labels(torch.as_tensor(ego_poses[0]), rl.LABEL_HORIZON)
    for j, s in enumerate(ep0["steps"]):
        assert s["maneuver"] == int(labs[3 + j])

    # Per-episode histogram (name-keyed, canonical order) drives the home-card
    # maneuver ribbon.
    m0, m1 = session["meta"]["episodes"]
    assert m0["maneuver_counts"] == {"turn_left": 5}
    assert m1["maneuver_counts"] == {"lane_keep": 5}


def test_export_maneuver_null_without_poses(tmp_path):
    # No ego_poses -> maneuver stays null and the strip is hidden by the SPA.
    _, session = _build(tmp_path)
    for ep in session["episodes"]:
        assert all(s["maneuver"] is None for s in ep["steps"])
    for em in session["meta"]["episodes"]:
        assert em["maneuver_counts"] == {}


def test_export_maneuver_null_when_poses_too_short(tmp_path):
    # Poses shorter than the label horizon cannot be labelled -> null, no crash.
    ego_poses = {0: _poses(10), 1: _poses(10)}       # T=10 <= horizon (20)
    session = export_bundle(
        _records(), tmp_path / "short", "x",
        maneuver_classes=MANEUVERS, ego_poses=ego_poses)
    for ep in session["episodes"]:
        assert all(s["maneuver"] is None for s in ep["steps"])


# ---------- (i) viz-standard: nav_commands + BEV-only fallback ----------------
# THE STANDARD (taniteval.corpus_overlay) needs the decoded strategic route/goal
# label and a BEV-only fallback when camera calibration is unrecoverable.

def test_meta_carries_nav_commands(tmp_path):
    """meta.nav_commands maps ArmOutput.nav_cmd -> strategic route/goal label in
    the canonical refb.NAV_COMMANDS order, so the SPA labels routes from DATA
    (not the old hard-coded ["straight","left","right"] that mislabelled 0/3)."""
    from tanitad.resim.export import NAV_COMMANDS
    _, session = _build(tmp_path)
    assert session["meta"]["nav_commands"] == list(NAV_COMMANDS)
    assert session["meta"]["nav_commands"][0] == "follow"      # NOT "straight"
    assert session["meta"]["nav_commands"][3] == "straight"    # NOT undefined


def test_uncalibrated_corpus_nulls_image_paths_keeps_bev(tmp_path):
    """A corpus flagged uncalibrated -> every step's gt_wp_img / arm.wp_img is
    null (camera overlay off) while wp_bev + the frame JPEG survive: the SPA's
    BEV-only fallback (raw frame + "see BEV" note, BEV carries the comparison)."""
    bundle, session = _build(tmp_path, "uncal", uncalibrated_corpora=["toy-val"])
    assert session["meta"]["uncalibrated_corpora"] == ["toy-val"]
    for ep in session["episodes"]:
        for st in ep["steps"]:
            assert st["gt_wp_img"] is None          # camera overlay disabled
            assert st["gt_wp_bev"] is not None       # BEV survives
            for a in st["arms"].values():
                assert a["wp_img"] is None
                assert a["wp_bev"] is not None        # BEV comparison intact
    n_steps = sum(len(ep["steps"]) for ep in session["episodes"])
    assert len(list((bundle / "frames").glob("*.jpg"))) == n_steps  # frames kept


def test_calibrated_default_projects_image_paths(tmp_path):
    """Default (no uncalibrated_corpora) still projects the camera fan."""
    _, session = _build(tmp_path, "cal")
    assert session["meta"]["uncalibrated_corpora"] == []
    st = session["episodes"][0]["steps"][0]
    assert st["gt_wp_img"] is not None
    assert st["arms"]["main"]["wp_img"] is not None


# ---------- (j) synthetic sample bundle (the pod-free demo) -------------------

def test_sample_bundle_exercises_full_standard(tmp_path):
    """tanitad.resim.sample builds a valid, demoable bundle covering every
    viz-standard element: 3 arms each with a decoded maneuver (argmax) + a nav
    route, varied maneuvers across episodes, formal gates, and one BEV-only
    fallback episode."""
    from tanitad.resim.sample import make_sample_bundle
    session = make_sample_bundle(tmp_path / "demo")
    meta = session["meta"]
    assert [a["name"] for a in meta["arms"]] == ["main", "refa", "refb"]
    assert meta["nav_commands"][1] == "left"
    assert meta["gates"] is not None and all(a["gates"] for a in meta["arms"])
    # exactly one BEV-only-fallback corpus, present among the episodes
    assert "cosmos-ood-demo" in meta["uncalibrated_corpora"]
    seen_uncal = ({ep["corpus_tag"] for ep in meta["episodes"]}
                  & set(meta["uncalibrated_corpora"]))
    assert seen_uncal == {"cosmos-ood-demo"}
    for ep in session["episodes"]:
        uncal = ep["corpus_tag"] in meta["uncalibrated_corpora"]
        for st in ep["steps"]:
            for a in st["arms"].values():
                probs = a["heads"].get("maneuver_probs")
                assert probs and len(probs) == len(MANEUVERS)
                assert 0 <= int(np.argmax(probs)) < len(MANEUVERS)   # decoded man
                assert a["heads"].get("nav_cmd") is not None         # route/goal
                assert (a["wp_img"] is None) == uncal   # calib eps draw the fan
    # >=3 distinct maneuver mixes across the episodes (variety, not monotone)
    mixes = {tuple(sorted(em["maneuver_counts"])) for em in meta["episodes"]}
    assert len(mixes) >= 3


def test_sample_bundle_shows_both_strategic_branches(tmp_path):
    """The demo must exhibit the fix, not just satisfy it: REF-B fills the
    strategic PREDICTION slot from its route head, main/refa declare it
    unavailable, and all three still show the logged input separately."""
    from tanitad.resim.sample import make_sample_bundle

    session = make_sample_bundle(tmp_path / "demo")
    st = session["episodes"][0]["steps"][0]
    assert _viz(st, "refb", "strategic")["state"] == "present"
    assert _viz(st, "refb", "strategic")["kind"] == "model_output"
    for arm in ("main", "refa"):
        strat = _viz(st, arm, "strategic")
        assert strat["state"] == "unavailable" and strat["reason"]
    for arm in ("main", "refa", "refb"):
        inp = _viz(st, arm, "strategic_input")
        assert inp["kind"] == "given_input" and inp["state"] == "present"
    # the demo's route head is NOT a bijection of its own input -- an exact
    # input->output bijection is the nav-echo defect the programme measured at
    # 369/369, and a demo bundle must not model one.
    pairs = {(_viz(s, "refb", "strategic_input")["value"],
              _viz(s, "refb", "strategic")["value"])
             for ep in session["episodes"] for s in ep["steps"]}
    assert len({p for _, p in pairs}) < len({i for i, _ in pairs})


def test_sample_bundle_serves_over_fastapi(tmp_path):
    """The synthetic bundle is servable end-to-end: build_app lists it, returns
    its session.json, and serves a frame from an uncalibrated (fallback) step."""
    from fastapi.testclient import TestClient
    import resim_app
    from tanitad.resim.sample import make_sample_bundle
    make_sample_bundle(tmp_path / "demo")
    client = TestClient(resim_app.build_app(tmp_path))
    ids = {s["id"] for s in client.get("/api/sessions").json()}
    assert ids == {"demo"}
    body = client.get("/api/session/demo").json()
    # a frame from the BEV-only-fallback episode is still fetchable (raw camera)
    uncal_ep = next(ep for ep in body["episodes"]
                    if ep["corpus_tag"] in body["meta"]["uncalibrated_corpora"])
    frame = uncal_ep["steps"][0]["frame"]
    r = client.get(f"/frames/demo/{frame}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert uncal_ep["steps"][0]["gt_wp_img"] is None        # overlay off


def test_cli_requires_sessions_root_or_demo():
    """The CLI fails loud (SystemExit) when neither --sessions-root nor --demo
    is given — no silent serve of an accidental empty temp dir."""
    import resim_app
    with pytest.raises(SystemExit):
        resim_app.main(["--port", "9999"])


# ---------- (b) portability ---------------------------------------------------

def _walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)


ABS = re.compile(r"^[A-Za-z]:[\\/]|^/|[A-Za-z]:\\")


def test_bundle_is_portable_no_abs_paths(tmp_path):
    bundle, session = _build(tmp_path)
    text = (bundle / "session.json").read_text(encoding="utf-8")
    assert "/workspace/exp" not in text
    assert "C:\\ckpts" not in text and "C:/ckpts" not in text
    for s in _walk_strings(session):
        assert not ABS.search(s), f"absolute-looking path leaked: {s!r}"


# ---------- (c) fail-loud -----------------------------------------------------

def test_export_fails_loud_on_empty(tmp_path):
    with pytest.raises(ValueError, match="zero records"):
        export_bundle([], tmp_path / "empty", "x")


def test_export_fails_loud_on_frameless_record(tmp_path):
    rec = _rec(0, 0, 3)
    rec.frame = None
    with pytest.raises(ValueError, match="emit_frames"):
        export_bundle([rec], tmp_path / "nf", "x")


def test_resim_color_deterministic_fallback():
    assert resim_color("main") == RESIM_COLORS["main"]
    c = resim_color("mystery_arm")
    assert c == resim_color("mystery_arm") and re.match(r"^#[0-9a-f]{6}$", c)


# ---------- (d) frame downscale ----------------------------------------------

def test_wide_frames_are_downscaled(tmp_path):
    from PIL import Image
    rec = _rec(0, 0, 3, hw=(120, 900))       # 900 px wide > default 640
    export_bundle([rec], tmp_path / "wide", "x", max_w=640)
    jpg = next((tmp_path / "wide" / "frames").glob("*.jpg"))
    assert Image.open(jpg).width == 640


# ---------- (f) image-plane projection + camera letterbox (fan-draw pins) -----
# Regression pins for the TanitResim camera panel. Two invariants the display
# relies on, whose violation produced Sayed's urban-session defects:
#   1. the exporter projects forward waypoints ON/BELOW the horizon (v >= h/2) —
#      a fan drawn ABOVE the horizon can only come from the display layer;
#   2. the front-end letterbox map is ISOTROPIC (one scale for both axes) and
#      keeps forward points inside the frame rect — the stretched-frame bug was
#      an anisotropic canvas map (sx != sy) dragging the fan sideways/up.

from tanitad.replay.rr_log import to_image_plane


def _img_path(wp):
    """Origin-prefixed ego path, exactly as the exporter builds it."""
    return np.vstack([[0.0, 0.0], np.asarray(wp, dtype=np.float64)])


def test_straight_ahead_projects_centered_and_below_horizon():
    h = w = 256
    wp = np.array([[d, 0.0] for d in (5, 10, 15, 20)], dtype=np.float64)
    u, v = to_image_plane(_img_path(wp), h, w).T
    # dead-centre column for a straight-ahead trajectory
    assert np.allclose(u, w / 2, atol=1e-6)
    # every projected point sits on/below the horizon (v >= h/2)
    assert np.all(v >= h / 2 - 1e-9)
    # forward distance grows => v decreases toward the horizon: the stored
    # polyline is monotonic (equivalently: runs monotonically down to the ego).
    assert np.all(np.diff(v) < 0)


@pytest.mark.parametrize("wp", [
    [[5, 0], [10, 0], [15, 0], [20, 0]],         # straight
    [[5, 1], [10, 3], [15, 6], [20, 10]],        # hard left  (+y)
    [[5, -1], [10, -3], [15, -6], [20, -10]],    # hard right (-y)
    [[2, 4], [3, 8], [4, 12], [5, 16]],          # near + extreme lateral
])
def test_forward_trajectory_never_above_horizon(wp):
    h = w = 256
    v = to_image_plane(_img_path(np.array(wp, float)), h, w)[:, 1]
    assert np.all(v >= h / 2 - 1e-9), "projected fan rose above the horizon"


# -- letterbox canvas map (mirrors static/app.js containRect + toPx) -----------

def _contain_rect(cw, ch, fw, fh):
    scale = min(cw / fw, ch / fh)
    dw, dh = fw * scale, fh * scale
    return scale, (cw - dw) / 2.0, (ch - dh) / 2.0, dw, dh


@pytest.mark.parametrize("cw,ch", [(460, 460), (709, 420), (300, 300), (1189, 420)])
def test_letterbox_is_isotropic_and_keeps_forward_points_in_frame(cw, ch):
    fw = fh = 256
    scale, ox, oy, dw, dh = _contain_rect(cw, ch, fw, fh)
    # isotropic: ONE scale for both axes => a square frame is never stretched.
    assert scale == min(cw / fw, ch / fh)
    # frame rect is centered and never exceeds the panel.
    assert ox >= -1e-9 and oy >= -1e-9
    assert dw <= cw + 1e-9 and dh <= ch + 1e-9
    # forward, on-frame waypoints land inside the drawn frame rect.
    wp = np.array([[5, 0.5], [10, 1.0], [15, -1.5], [20, 2.0]], dtype=np.float64)
    for u, v in to_image_plane(_img_path(wp), fh, fw):
        if 0 <= v <= fh:
            cx, cy = ox + u * scale, oy + v * scale
            assert ox - 1e-6 <= cx <= ox + dw + 1e-6
            assert oy - 1e-6 <= cy <= oy + dh + 1e-6


def test_export_wp_img_never_above_horizon(tmp_path):
    # In a written bundle, every exported gt_wp_img point stays within the JPEG
    # horizontally and never rises above that frame's horizon (v >= h/2).
    from PIL import Image
    bundle, session = _build(tmp_path)
    for ep in session["episodes"]:
        for st in ep["steps"]:
            w, h = Image.open(bundle / "frames" / st["frame"]).size
            for u, v in st["gt_wp_img"]:
                assert 0 <= u <= w, f"gt_wp_img u off-frame: {u} (w={w})"
                assert v >= h / 2 - 1e-6, f"gt_wp_img above horizon: {(u, v)}"


# ---------- (g) per-corpus camera projection (D-016 extrinsics/intrinsics) ----
# comma2k19's EON cam is the verified reference (unchanged); physicalai's
# front-wide is an f-theta lens whose real canonical focal is ~444 px (not the
# 266 the nominal-rectilinear crop assumes) at height 1.43 m. The overlay defect
# was the fan drawn ~2x too close to the horizon ("sky"), fixed by the corpus's
# own calibrated focal+height. The horizon itself is NOT offset (pitch ~0).

from tanitad.replay.rr_log import (COMMA_CAM, PHYSICALAI_CAM, CamProjection,
                                   cam_for_corpus, to_image_plane as _tip)


def test_comma_projection_unchanged_regression():
    # default (no cam) and the comma cam both reproduce the OLD hardcoded
    # formula f=266, H=1.22, horizon=h/2 exactly — comma must not move.
    h = w = 256
    wp = np.array([[5.0, 1.0], [10.0, -2.0], [15.0, 0.5], [20.0, 3.0]])
    path = _img_path(wp)
    f, x = 266.0, np.clip(path[:, 0], 2.0, None)
    exp_u = w / 2 - f * (path[:, 1] / x)
    exp_v = h / 2 + f * (1.22 / x)
    for got in (_tip(path, h, w), _tip(path, h, w, cam=COMMA_CAM),
                _tip(path, h, w, cam=cam_for_corpus("comma2k19-val-61c46fca8f7f"))):
        assert np.allclose(got[:, 0], exp_u)
        assert np.allclose(got[:, 1], exp_v)


def test_cam_for_corpus_resolves_tags():
    assert cam_for_corpus("physicalai-val-8c0d3047924e") is PHYSICALAI_CAM
    assert cam_for_corpus("physicalai-train-eeabeca35fe1") is PHYSICALAI_CAM
    assert cam_for_corpus("comma2k19-val-61c46fca8f7f") is COMMA_CAM
    assert cam_for_corpus("toy-val") is COMMA_CAM        # unknown -> reference
    assert cam_for_corpus(None) is COMMA_CAM


def test_physicalai_cam_pushes_fan_down_onto_road():
    # The fix: physicalai's larger focal+height push forward waypoints LOWER
    # (larger v, onto the road) instead of bunching them up near the horizon.
    h = w = 256
    assert PHYSICALAI_CAM.f_eff_256 > COMMA_CAM.f_eff_256      # 444 vs 266
    assert PHYSICALAI_CAM.cam_h > COMMA_CAM.cam_h              # 1.43 vs 1.22
    wp = np.array([[8.0, 0.0]])                               # 8 m ahead
    v_comma = _tip(wp, h, w, cam=COMMA_CAM)[0, 1]
    v_phys = _tip(wp, h, w, cam=PHYSICALAI_CAM)[0, 1]
    assert v_phys > v_comma + 10                             # clearly lower
    # straight-ahead still dead-centre horizontally for both.
    assert abs(_tip(wp, h, w, cam=PHYSICALAI_CAM)[0, 0] - w / 2) < 1e-6


def test_physicalai_horizon_is_not_offset():
    # Calibration finding: physicalai pitch ~0 -> horizon on h/2, NOT offset.
    h = 256
    assert abs(PHYSICALAI_CAM.horizon_row(h) - h / 2) < 1e-6
    far = _tip(np.array([[500.0, 0.0]]), h, h, cam=PHYSICALAI_CAM)[0, 1]
    assert abs(far - h / 2) < 2.0                            # approaches h/2


def test_nonzero_pitch_moves_horizon_off_center():
    # The pitch->horizon machinery: a pitched-down cam raises the horizon above
    # h/2 and forward points approach THAT horizon, not h/2.
    h = w = 256
    cam = CamProjection(f_eff_256=266.0, cam_h=1.22, pitch_deg=6.0)
    hz = cam.horizon_row(h)
    assert hz < h / 2 - 5                                    # horizon risen
    far = _tip(np.array([[400.0, 0.0]]), h, w, cam=cam)[0, 1]
    assert abs(far - hz) < 2.0 and far < h / 2 - 3           # tracks the horizon


# ---------- (e) FastAPI server -----------------------------------------------

@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient
    import resim_app
    _build(tmp_path, name="sess1")
    _build(tmp_path, name="sess2")
    app = resim_app.build_app(tmp_path)
    return TestClient(app)


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "TanitResim" in r.text


def test_static_asset_served(client):
    r = client.get("/static/app.js")
    assert r.status_code == 200 and "TanitResim" in r.text


def test_sessions_list(client):
    r = client.get("/api/sessions")
    assert r.status_code == 200
    ids = {s["id"] for s in r.json()}
    assert ids == {"sess1", "sess2"}
    one = r.json()[0]
    assert one["n_episodes"] == 2 and len(one["arms"]) == 2


def test_session_fetch_and_404(client):
    r = client.get("/api/session/sess1")
    assert r.status_code == 200
    body = r.json()
    assert "meta" in body and "episodes" in body
    assert client.get("/api/session/does_not_exist").status_code == 404


def test_frame_fetch_and_guards(client):
    r = client.get("/api/session/sess1")
    frame = r.json()["episodes"][0]["steps"][0]["frame"]
    ok = client.get(f"/frames/sess1/{frame}")
    assert ok.status_code == 200
    assert ok.headers["content-type"] == "image/jpeg"
    assert client.get("/frames/sess1/missing.jpg").status_code == 404
    assert client.get("/frames/nope/x.jpg").status_code == 404


# ---------- (k) the viz-standard CONTRACT (tanitad.viz_standard) --------------
# THE STANDARD used to be a docstring claim and a README table. These pin it as
# a data structure that travels with the artifact: five declared elements per
# rendered frame, `source` + `kind` mandatory when present, `reason` mandatory
# when unavailable, and -- the defect class in SPEC section 3.1 -- a PREDICTION
# slot that refuses a ground-truth-derived input.


def test_viz_element_requires_source_and_kind_when_present():
    from tanitad.viz_standard import VizElement, VizStandardError

    ok = VizElement.present("tactical", "turn_left",
                            source="ArmOutput.maneuver_probs.argmax(-1)",
                            kind="model_output")
    assert ok.state == "present" and ok.kind == "model_output"
    # a present element with no declared source is exactly how 3.1 hid
    with pytest.raises(VizStandardError, match="source"):
        VizElement(element="tactical", state="present", value="turn_left",
                   kind="model_output")
    with pytest.raises(VizStandardError, match="kind"):
        VizElement(element="tactical", state="present", value="turn_left",
                   source="somewhere")


def test_viz_element_requires_reason_when_unavailable():
    from tanitad.viz_standard import VizElement, VizStandardError

    el = VizElement.unavailable("strategic", "arm 'main' has no route head")
    assert el.state == "unavailable" and "route head" in el.reason
    assert "unavailable" in el.hud_text() and "no route head" in el.hud_text()
    with pytest.raises(VizStandardError, match="reason"):
        VizElement(element="strategic", state="unavailable")


def test_check_frame_refuses_a_silently_missing_element():
    """The contract's whole point: a frame may not be rendered with an element
    quietly absent. Every missing standard element is named."""
    from tanitad.viz_standard import (STANDARD_ELEMENTS, VizElement,
                                      VizStandardError, check_frame)

    full = [VizElement.present(e, "x", source="src", kind="derived")
            for e in STANDARD_ELEMENTS]
    assert len(check_frame(full, where="unit")) == len(STANDARD_ELEMENTS)

    partial = [e for e in full if e.element not in ("strategic", "bev")]
    with pytest.raises(VizStandardError) as ei:
        check_frame(partial, where="unit")
    msg = str(ei.value)
    assert "strategic" in msg and "bev" in msg and "unit" in msg


def test_check_frame_accepts_a_declared_unavailable_element():
    from tanitad.viz_standard import (STANDARD_ELEMENTS, VizElement,
                                      check_frame, hud_lines)

    els = [VizElement.present(e, "x", source="src", kind="derived")
           for e in STANDARD_ELEMENTS if e != "strategic"]
    els.append(VizElement.unavailable(
        "strategic", "arm 'refa' has no route-prediction head"))
    got = check_frame(els, where="unit")
    assert len(got) == len(STANDARD_ELEMENTS)
    # the reason is DRAWN, never blank
    line = [ln for ln in hud_lines(got) if ln.startswith("strategic")][0]
    assert "unavailable" in line and "no route-prediction head" in line


def test_prediction_slot_refuses_a_given_input():
    """*** THE SPEC section 3.1 DEFECT CLASS, refused at the type level. ***

    `strategic` and `tactical` are PREDICTION slots. A value whose kind is
    `given_input` / `gt_label` cannot be put in one -- which is precisely what
    the SPA did with the GT-derived nav command."""
    from tanitad.viz_standard import (PREDICTION_ELEMENTS, VizElement,
                                      VizStandardError, check_frame)

    assert set(PREDICTION_ELEMENTS) == {"tactical", "strategic"}
    els = [VizElement.present(e, "x", source="src", kind="derived")
           for e in ("camera", "bev", "ade", "tactical")]
    els.append(VizElement.present(
        "strategic", "left",
        source="refb_labels.nav_command(episode.poses, t)",
        kind="given_input"))                      # <- the nav-echo defect
    with pytest.raises(VizStandardError, match="given_input"):
        check_frame(els, where="unit")
    # ... and the same value is fine in the aux INPUT slot
    els[-1] = VizElement.present(
        "strategic_input", "left",
        source="refb_labels.nav_command(episode.poses, t)", kind="given_input")
    els.append(VizElement.unavailable("strategic", "no route head"))
    assert len(check_frame(els, where="unit")) == 6


def test_privileged_and_conditioned_values_are_marked_on_the_frame():
    from tanitad.viz_standard import VizElement, hud_lines

    given = VizElement.present(
        "strategic_input", "left",
        source="refb_labels.nav_command(episode.poses, t)", kind="given_input")
    assert given.privileged
    pred = VizElement.present(
        "strategic", "route_straight", source="route_logits.argmax(-1)",
        kind="model_output", conditioned_on=("nav_cmd",))
    lines = hud_lines([given, pred])
    assert any("given" in ln.lower() for ln in lines)      # privileged marker
    assert any("nav_cmd" in ln for ln in lines)            # conditioning shown


def test_viz_elements_round_trip_through_json():
    from tanitad.viz_standard import VizElement, from_json, to_json

    els = (VizElement.present("ade", "0.42 m", source="mean L2", kind="derived",
                              n=4),
           VizElement.unavailable("camera", "calibration unverified"))
    assert from_json(to_json(els)) == els
    assert json.loads(json.dumps(to_json(els)))[0]["kind"] == "derived"


# ---------- (l) SPEC section 3.1 REGRESSION: nav_cmd is an INPUT --------------
# `nav_cmd` is derived from the episode's OWN FUTURE POSES and FED to the model
# (replay/arms.py:526). Rendering it as the strategic PREDICTION is the nav-echo
# defect the programme binds against. The bundle must carry the two apart, and
# an arm with no route head must say `unavailable` with a reason -- never fall
# back to the input.


def _viz(step_dict, arm, element):
    """The one viz record for (arm, element) in an exported step."""
    rows = step_dict["arms"][arm]["viz"]
    hit = [r for r in rows if r["element"] == element]
    assert len(hit) == 1, f"{arm}/{element}: expected 1 record, got {len(hit)}"
    return hit[0]


def test_export_declares_the_five_standard_elements_per_arm(tmp_path):
    from tanitad.viz_standard import STANDARD_ELEMENTS

    _, session = _build(tmp_path)
    for ep in session["episodes"]:
        for st in ep["steps"]:
            for name in ("main", "refb"):
                got = {r["element"] for r in st["arms"][name]["viz"]}
                assert got >= set(STANDARD_ELEMENTS), \
                    f"{name} missing {set(STANDARD_ELEMENTS) - got}"


def test_export_strategic_slot_is_unavailable_without_a_route_head(tmp_path):
    """*** The regression pin. *** refb carries nav_cmd=1 but NO route
    prediction: the strategic slot must be `unavailable` with a reason that
    names the input -- not the input's value wearing a prediction's label."""
    _, session = _build(tmp_path)
    st = session["episodes"][0]["steps"][0]
    strat = _viz(st, "refb", "strategic")
    assert strat["state"] == "unavailable"
    assert strat["source"] is None and strat["kind"] is None
    assert "nav_cmd" in strat["reason"] and "input" in strat["reason"].lower()
    # main has neither -> still declared, still with a reason
    assert _viz(st, "main", "strategic")["state"] == "unavailable"


def test_export_declares_the_gt_derived_input_in_its_own_slot(tmp_path):
    _, session = _build(tmp_path)
    st = session["episodes"][0]["steps"][0]
    inp = _viz(st, "refb", "strategic_input")
    assert inp["state"] == "present" and inp["kind"] == "given_input"
    assert inp["value"] == "left"                         # nav_cmd=1 -> "left"
    assert "nav_command" in inp["source"] and "poses" in inp["source"]
    # main emits no nav_cmd at all -> no input record is fabricated for it
    assert not [r for r in st["arms"]["main"]["viz"]
                if r["element"] == "strategic_input"]


def test_export_strategic_slot_carries_a_real_route_prediction(tmp_path):
    """When an arm DOES emit a route prediction it fills the slot as a
    model_output -- and declares that it was computed under the nav input."""
    rec = _rec(0, 0, 3)
    rec.arms["refb"].route_pred = 1                    # route_straight
    session = export_bundle([rec], tmp_path / "routed", "x",
                            maneuver_classes=MANEUVERS)
    st = session["episodes"][0]["steps"][0]
    strat = _viz(st, "refb", "strategic")
    assert strat["state"] == "present" and strat["kind"] == "model_output"
    assert strat["value"] == "route_straight"
    assert "route_logits" in strat["source"]
    assert any("nav_cmd" in c for c in strat["conditioned_on"]), \
        "a route head FiLM-conditioned on the nav input must declare it"
    assert st["arms"]["refb"]["heads"]["route_pred"] == 1
    # the input is STILL carried separately, never merged into the prediction
    assert _viz(st, "refb", "strategic_input")["value"] == "left"


def test_meta_carries_route_classes_and_the_gt_input_registry(tmp_path):
    from tanitad.resim.export import GT_DERIVED_INPUT_HEADS, ROUTE_CLASSES

    _, session = _build(tmp_path)
    assert session["meta"]["route_classes"] == list(ROUTE_CLASSES)
    assert session["meta"]["gt_derived_input_heads"] == \
        sorted(GT_DERIVED_INPUT_HEADS)
    assert "nav_cmd" in session["meta"]["gt_derived_input_heads"]


# -- the front-end half: a source contract over the SPA ------------------------
# There is no JS test runner here, so the SPA's half of the fix is pinned as a
# SOURCE contract: the GT-derived input key may be read in exactly ONE declared
# accessor, and the model-route accessor may never touch it. Grepping is the
# instrument, but the rule it enforces is structural -- the old app.js read
# `heads.nav_cmd` straight into a row labelled `strategic: route <goal>`.

_ACC_OPEN = "// >>> GT-DERIVED-INPUT ACCESSOR"
_ACC_CLOSE = "// <<< GT-DERIVED-INPUT ACCESSOR"
_GT_INPUT_KEYS = ("nav_cmd",)


def _app_js() -> str:
    from tanitad.resim.export import static_dir
    return (static_dir() / "app.js").read_text(encoding="utf-8")


def _js_function_body(src: str, name: str) -> str:
    """Source of ``function <name>(...) { ... }`` by brace counting."""
    i = src.index("function " + name + "(")
    j = src.index("{", i)
    depth, k = 0, j
    while k < len(src):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[i:k + 1]
        k += 1
    raise AssertionError(f"unbalanced braces reading {name}()")


def test_spa_reads_the_gt_derived_input_only_in_its_declared_accessor():
    """*** THE SPA-side regression pin. *** Every read of a GT-derived input
    key lives inside the one accessor whose name says it is an input."""
    lines = _app_js().splitlines()
    starts = [i for i, l in enumerate(lines) if _ACC_OPEN in l]
    ends = [i for i, l in enumerate(lines) if _ACC_CLOSE in l]
    assert len(starts) == 1 and len(ends) == 1 and starts[0] < ends[0], (
        "app.js must declare exactly one GT-derived-input accessor block "
        f"delimited by {_ACC_OPEN!r} / {_ACC_CLOSE!r}")
    lo, hi = starts[0], ends[0]
    offenders = []
    for i, line in enumerate(lines):
        if not any(k in line for k in _GT_INPUT_KEYS):
            continue
        if lo <= i <= hi:
            continue
        s = line.strip()
        if s.startswith("//") or s.startswith("*") or s.startswith("/*"):
            continue                                    # prose may name it
        offenders.append(f"  app.js:{i + 1}: {s}")
    assert not offenders, (
        "a ground-truth-derived INPUT is read outside the declared accessor "
        "-- this is the nav-echo defect:\n" + "\n".join(offenders))


def test_spa_model_route_accessor_never_touches_the_input():
    body = _js_function_body(_app_js(), "routePred")
    for key in _GT_INPUT_KEYS:
        assert key not in body, (
            f"routePred() references {key!r}: the model-route field must never "
            "fall back to the logged input")
    assert "reason" in body, "an unavailable route must carry its reason"


def test_spa_labels_the_prediction_and_the_input_distinctly():
    src = _app_js()
    assert 'var LBL_ROUTE_PRED = "route (model)";' in src
    assert 'var LBL_ROUTE_INPUT = "route (logged input)";' in src
    hud = _js_function_body(src, "updateCamHud")
    assert "LBL_ROUTE_PRED" in hud and "LBL_ROUTE_INPUT" in hud, \
        "the camera HUD must render BOTH fields, never one merged row"
    # an absent prediction is DRAWN with its reason, not skipped: the HUD sends
    # that branch through naSpan(), whose text says so.
    assert "naSpan(rp.reason)" in hud, \
        "an absent route prediction must render its reason, not vanish"
    assert "unavailable" in _js_function_body(src, "naSpan")
