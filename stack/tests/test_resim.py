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


def test_worst_step_tracks_largest_error(tmp_path):
    # ADE grows with t (scale 1+0.1t), so the last step of each episode is worst.
    _, session = _build(tmp_path)
    for em in session["meta"]["episodes"]:
        assert em["worst_step"] == em["n_steps"] - 1


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
