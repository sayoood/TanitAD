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
