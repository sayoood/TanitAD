"""CPU tests for the PH0 pilot package (ph0_pilot / ph0_overlay_video /
ph0_sample_clips).

No GPU, no transformers, no cv2/ffmpeg: the modules keep pod-side imports
lazy, so importing them and exercising the pure cores (schema validation,
fusion gate, JSON extraction, RLE, Engine A geometry, frame composition,
sampler determinism) must work on the dev box with numpy+torch+PIL+pandas.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
STACK = REPO / "stack"
sys.path.insert(0, str(STACK / "scripts"))
sys.path.insert(0, str(STACK))

import ph0_overlay_video as pov                                 # noqa: E402
import ph0_pilot as pp                                          # noqa: E402
import ph0_sample_clips as psc                                  # noqa: E402


# =============================================================================
# synthetic record helpers
# =============================================================================

def make_sign(grounded=True, kind="nav", text="LYON", applies=True):
    s = {"kind": kind, "state": None, "text_ocr": text,
         "applies_to_ego": applies}
    if grounded:
        s["bbox"] = [100.0, 40.0, 180.0, 90.0]
        s["frame_idx"] = 3
    return s


def make_action(verb="prepare_lane_change", direction="left",
                within_m=500.0, source="path", sign_idx=None):
    a = {"verb": verb, "direction": direction,
         "constraints": {"within_m": within_m},
         "reason": "test", "source": source, "confidence": 0.8,
         "evidence": {}}
    if sign_idx is not None:
        a["evidence"]["sign_idx"] = sign_idx
    return a


def make_record(signs=None, actions=None, goal=None, agents=None):
    return {
        "schema_version": pp.SCHEMA_VERSION,
        "clip_id": "clip_0001",
        "t0_us": 8_000_000,
        "scenario": {"illumination": "day", "weather": "clear",
                     "daynight": "day",
                     "road": {"type": "highway", "lanes_visible": 3,
                              "lane_ego": 2},
                     "agents": agents if agents is not None else [],
                     "ego_behaviour": "cruising"},
        "domain": {"class": "highway", "confidence": 0.9},
        "signs": signs if signs is not None else [],
        "strategic": {
            "goal": goal or {"kind": "follow_main_road", "target_text": None,
                             "source": "path", "confidence": 0.7,
                             "evidence": {}},
            "actions": actions if actions is not None else []},
        "_provenance": {"model_id": "test/arm", "prompt_hash": "0" * 64},
    }


def engine_a_with(lc_token=None, lc_arc=120.0, lon_token=None,
                  route_token="follow", route_valid=True, route_dist=None,
                  net_dv=0.0, stops=False):
    lc = ([{"token": lc_token, "t_start_s": 2.0, "t_end_s": 4.0,
            "lat_m": 3.2, "arc_from_t0_m": lc_arc}] if lc_token else [])
    lon = ([{"token": lon_token, "t_start_s": 1.0, "t_end_s": 3.0,
             "stop_dist_m": 30.0, "dv": -3.0, "arc_from_t0_m": 25.0}]
           if lon_token else [])
    return {"t0_idx": 80, "duration_s": 20.0,
            "polyline_xy": [[0.0, 0.0], [10.0, 0.5], [20.0, 1.0]],
            "route": {"token": route_token, "token_valid": route_valid,
                      "reason": "road_following", "dist_m": route_dist,
                      "dist_band": "d_none", "maneuver_dyaw_rad": 0.0,
                      "graded_route": 0.0, "arc_m": 180.0},
            "lane_change_events": lc, "speed_events": lon,
            "speed_profile": {"v_t0_ms": 12.0, "v_min_future_ms": 8.0,
                              "v_max_future_ms": 14.0, "net_dv_ms": net_dv,
                              "stops": stops},
            "peak_kappa_per_m": 0.002}


# =============================================================================
# schema validation
# =============================================================================

class TestSchema:
    def test_valid_synthetic_record(self):
        rec = make_record(signs=[make_sign()],
                          actions=[make_action()])
        rec["strategic"]["actions"][0]["geometric_consistency"] = "pass"
        assert pp.validate_record(rec) == []

    def test_missing_required_fields_reported(self):
        rec = make_record()
        del rec["scenario"]["road"]
        del rec["domain"]["confidence"]
        rec["_provenance"].pop("prompt_hash")
        errs = pp.validate_record(rec)
        joined = "\n".join(errs)
        assert "scenario.road" in joined
        assert "domain.confidence" in joined
        assert "_provenance.prompt_hash" in joined

    def test_bad_vocab_rejected(self):
        rec = make_record(
            signs=[dict(make_sign(), kind="banana")],
            actions=[make_action()],
            goal={"kind": "teleport", "source": "path"})
        rec["strategic"]["actions"][0]["geometric_consistency"] = "maybe"
        rec["strategic"]["actions"][0]["constraints"]["parsecs"] = 12
        errs = "\n".join(pp.validate_record(rec))
        assert "signs[0].kind" in errs
        assert "goal.kind" in errs
        assert "not pass|disputed" in errs
        assert "unknown slot" in errs

    def test_not_a_dict(self):
        assert pp.validate_record([]) == ["record is not an object"]


# =============================================================================
# fusion gate
# =============================================================================

class TestFusionGate:
    def test_consistent_lane_change_passes(self):
        rec = make_record(actions=[make_action("prepare_lane_change", "left")])
        ea = engine_a_with(lc_token="lc_left", lc_arc=120.0)
        pp.fusion_gate(rec, ea)
        a = rec["strategic"]["actions"][0]
        assert a["geometric_consistency"] == "pass"
        assert a["fusion_reasons"] == []
        assert rec["fusion"]["actions_pass"] == 1

    def test_wrong_direction_disputed(self):
        rec = make_record(actions=[make_action("prepare_lane_change",
                                               "right")])
        ea = engine_a_with(lc_token="lc_left")
        pp.fusion_gate(rec, ea)
        a = rec["strategic"]["actions"][0]
        assert a["geometric_consistency"] == "disputed"
        assert rec["fusion"]["actions_disputed"] == 1

    def test_envelope_violation_disputed(self):
        # event exists but OUTSIDE the stated within_m envelope
        rec = make_record(actions=[make_action("prepare_lane_change", "left",
                                               within_m=50.0)])
        ea = engine_a_with(lc_token="lc_left", lc_arc=120.0)
        pp.fusion_gate(rec, ea)
        assert rec["strategic"]["actions"][0][
            "geometric_consistency"] == "disputed"

    def test_stop_action_needs_stop_evidence(self):
        rec = make_record(actions=[make_action("prepare_stop", None,
                                               within_m=None)])
        pp.fusion_gate(rec, engine_a_with())          # no stop anywhere
        assert rec["strategic"]["actions"][0][
            "geometric_consistency"] == "disputed"
        rec2 = make_record(actions=[make_action("prepare_stop", None,
                                                within_m=None)])
        pp.fusion_gate(rec2, engine_a_with(lon_token="stop_at_point"))
        assert rec2["strategic"]["actions"][0][
            "geometric_consistency"] == "pass"

    def test_hold_corridor_disputed_by_junction_event(self):
        rec = make_record(actions=[make_action("hold_corridor", None,
                                               within_m=None)])
        pp.fusion_gate(rec, engine_a_with(route_token="turn_left",
                                          route_dist=40.0))
        assert rec["strategic"]["actions"][0][
            "geometric_consistency"] == "disputed"

    def test_unknown_verb_disputed(self):
        rec = make_record(actions=[make_action("do_a_barrel_roll")])
        pp.fusion_gate(rec, engine_a_with())
        a = rec["strategic"]["actions"][0]
        assert a["geometric_consistency"] == "disputed"
        assert any("vocabulary" in r for r in a["fusion_reasons"])

    def test_ungrounded_sign_auto_disputed(self):
        rec = make_record(signs=[make_sign(grounded=False)])
        pp.fusion_gate(rec, engine_a_with())
        s = rec["signs"][0]
        assert s["grounded"] is False and s["disputed"] is True
        assert rec["fusion"]["ungrounded_disputed"] == [
            {"kind": "sign", "index": 0,
             "reason": "ungrounded (missing bbox/frame_idx)"}]

    def test_ungrounded_agent_auto_disputed(self):
        rec = make_record(agents=[{"class": "car", "position_rel": "ahead",
                                   "behaviour": "braking"}])
        pp.fusion_gate(rec, engine_a_with())
        assert rec["scenario"]["agents"][0]["disputed"] is True

    def test_route_to_goal_requires_grounded_nav_sign(self):
        goal = {"kind": "route_to", "target_text": "LYON",
                "source": "signage", "confidence": 0.9,
                "evidence": {"sign_idx": 0}}
        ok = make_record(signs=[make_sign()], goal=dict(goal))
        pp.fusion_gate(ok, engine_a_with())
        assert ok["strategic"]["goal"]["fusion"] == "pass"
        bad = make_record(signs=[make_sign(grounded=False)], goal=dict(goal))
        pp.fusion_gate(bad, engine_a_with())
        assert bad["strategic"]["goal"]["fusion"] == "disputed"

    def test_engine_a_absent_path_claim_disputed_signage_stands(self):
        sign = make_sign(kind="speed", text="80")
        path_a = make_action("reduce_to", None, within_m=None, source="path")
        sig_a = make_action("reduce_to", None, within_m=None,
                            source="signage", sign_idx=0)
        rec = make_record(signs=[sign], actions=[path_a, sig_a])
        pp.fusion_gate(rec, None)
        acts = rec["strategic"]["actions"]
        assert acts[0]["geometric_consistency"] == "disputed"
        assert any("engine_a_absent" in r for r in acts[0]["fusion_reasons"])
        assert acts[1]["geometric_consistency"] == "pass"
        assert rec["fusion"]["engine_a_available"] is False


# =============================================================================
# pass-2 verdicts, JSON extraction, RLE
# =============================================================================

class TestProtocolPieces:
    def test_apply_verdicts_retracts(self):
        rec = make_record(signs=[make_sign(), make_sign(text="PARIS")],
                          actions=[make_action(), make_action("hold_corridor",
                                                              None, None)])
        claims = pp.enumerate_claims(rec)
        ids = [c["claim_id"] for c in claims]
        assert ids == ["sign_0", "sign_1", "goal", "action_0", "action_1"]
        pp.apply_verdicts(rec, [
            {"claim_id": "sign_1", "verdict": "RETRACT", "reason": "unread"},
            {"claim_id": "action_0", "verdict": "RETRACT", "reason": "n/a"},
            {"claim_id": "goal", "verdict": "CONFIRM"}])
        assert rec["signs"][1]["retracted"] is True
        assert "retracted" not in rec["signs"][0]
        assert len(rec["strategic"]["actions"]) == 1
        assert rec["strategic"]["retracted_actions"][0]["verb"] == \
            "prepare_lane_change"

    def test_retracted_sign_cannot_ground_route_to(self):
        goal = {"kind": "route_to", "target_text": "LYON",
                "source": "signage", "confidence": 0.9,
                "evidence": {"sign_idx": 0}}
        rec = make_record(signs=[make_sign()], goal=goal)
        pp.apply_verdicts(rec, [{"claim_id": "sign_0", "verdict": "RETRACT"}])
        pp.fusion_gate(rec, engine_a_with())
        assert rec["strategic"]["goal"]["fusion"] == "disputed"

    def test_extract_json_variants(self):
        obj = {"a": 1, "b": {"c": [1, 2]}}
        js = json.dumps(obj)
        assert pp.extract_json(js) == obj
        assert pp.extract_json(f"Sure! Here it is:\n```json\n{js}\n```") == obj
        assert pp.extract_json(f"prefix {js} trailing text") == obj
        assert pp.extract_json('{"s": "brace in \\" string }"}') == \
            {"s": 'brace in " string }'}
        with pytest.raises(ValueError):
            pp.extract_json("no json here")
        with pytest.raises(ValueError):
            pp.extract_json("")

    def test_rle_roundtrip_and_bbox(self):
        rng = np.random.default_rng(0)
        mask = rng.random((37, 53)) > 0.7
        rle = pp.rle_encode(mask)
        assert rle["size"] == [37, 53]
        assert np.array_equal(pp.rle_decode(rle), mask)
        empty = np.zeros((5, 5), bool)
        assert pp.mask_bbox(empty) is None
        m = np.zeros((10, 10), bool)
        m[2:5, 3:8] = True
        assert pp.mask_bbox(m) == [3, 2, 8, 5]

    def test_prompt_hash_stable_and_hexish(self):
        h = pp.prompt_hash()
        assert h == pp.prompt_hash() and len(h) == 64
        int(h, 16)


# =============================================================================
# Engine A on synthetic trajectories (torch, CPU)
# =============================================================================

def _straight_then_lc(T=200, v=10.0, dt=0.1):
    """Straight 10 m/s track with a +3.5 m (left) lateral shift mid-clip."""
    x = np.arange(T) * v * dt
    y = np.zeros(T)
    y[100:140] = np.linspace(0, 3.5, 40)
    y[140:] = 3.5
    yaw = np.zeros(T)
    yaw[1:] = np.arctan2(np.diff(y), np.diff(x))
    poses = np.stack([x, y, yaw, np.full(T, v)], axis=1)
    return torch.as_tensor(poses, dtype=torch.float32)


class TestEngineA:
    def test_summary_shape_and_speed(self):
        poses = _straight_then_lc()
        ea = pp.engine_a_summary(poses, t0_idx=80)
        assert ea["t0_idx"] == 80
        assert len(ea["polyline_xy"]) == 200
        assert ea["speed_profile"]["v_t0_ms"] == pytest.approx(10.0, abs=0.1)
        assert ea["speed_profile"]["stops"] is False
        assert json.loads(json.dumps(ea))              # JSON-serializable

    def test_lane_change_event_detected_left(self):
        ea = pp.engine_a_summary(_straight_then_lc(), t0_idx=80)
        toks = [e["token"] for e in ea["lane_change_events"]]
        assert "lc_left" in toks
        ev = next(e for e in ea["lane_change_events"]
                  if e["token"] == "lc_left")
        assert ev["arc_from_t0_m"] >= 0.0

    def test_prompt_view_drops_polyline(self):
        ea = pp.engine_a_summary(_straight_then_lc(), t0_idx=80)
        v = pp.engine_a_for_prompt(ea)
        assert "polyline_xy" not in v and "route" in v

    def test_fusion_on_real_engine_a(self):
        """End-to-end pure path: synthetic geometry disposes a VLM claim."""
        ea = pp.engine_a_summary(_straight_then_lc(), t0_idx=80)
        good = make_record(actions=[make_action("prepare_lane_change",
                                                "left", within_m=None)])
        pp.fusion_gate(good, ea)
        assert good["strategic"]["actions"][0][
            "geometric_consistency"] == "pass"
        bad = make_record(actions=[make_action("prepare_lane_change",
                                               "right", within_m=None)])
        pp.fusion_gate(bad, ea)
        assert bad["strategic"]["actions"][0][
            "geometric_consistency"] == "disputed"

    def test_missing_ego_returns_none(self, tmp_path):
        assert pp.load_ego_poses("nope", str(tmp_path)) is None
        assert pp.load_ego_poses("nope", None) is None

    def test_ego_loader_npz(self, tmp_path):
        poses = _straight_then_lc().numpy()
        np.savez(tmp_path / "clipX.npz", poses=poses)
        got = pp.load_ego_poses("clipX", str(tmp_path))
        assert got is not None and tuple(got.shape) == (200, 4)


# =============================================================================
# sampler determinism
# =============================================================================

def _toy_records(tmp_path, n_clips=40):
    import pandas as pd
    classes = ["urban", "highway", "intersection", "unstructured"]
    rows = []
    for i in range(n_clips):
        cid = f"clip_{i:04d}"
        rc = classes[i % 4] if i % 7 else classes[0]
        for task in ("meta_action", "vqa"):     # clip x task rows, like DESIGN
            rows.append({"clip_id": cid, "task": task, "road_class": rc,
                         "parsed": f"p{i}", "raw_output": f"r{i}"})
    p = tmp_path / "records.parquet"
    pd.DataFrame(rows).to_parquet(p)
    return p


class TestSampler:
    def test_seed0_twice_identical(self, tmp_path):
        rec = _toy_records(tmp_path)
        out1, out2 = tmp_path / "a.json", tmp_path / "b.json"
        psc.main(["--records", str(rec), "--out", str(out1), "--n", "12"])
        psc.main(["--records", str(rec), "--out", str(out2), "--n", "12"])
        a = json.loads(out1.read_text())
        b = json.loads(out2.read_text())
        assert a["clips"] == b["clips"]
        assert a["seed"] == 0 and a["road_col"] == "road_class"
        assert "sampler_line" in a and "--records" in a["sampler_line"]
        assert len(a["clips"]) == 12
        assert sum(a["strata_counts"].values()) == 12

    def test_every_stratum_represented(self, tmp_path):
        rec = _toy_records(tmp_path)
        out = tmp_path / "c.json"
        psc.main(["--records", str(rec), "--out", str(out), "--mini", "8"])
        d = json.loads(out.read_text())
        assert d["n"] == 8 and d["mini"] is True
        assert set(d["strata_counts"]) == {"urban", "highway",
                                           "intersection", "unstructured"}
        assert all(v >= 1 for v in d["strata_counts"].values())
        # clip ids unique, classes truthful
        ids = [c["clip_id"] for c in d["clips"]]
        assert len(set(ids)) == len(ids) == 8

    def test_different_seed_differs(self):
        clip_to_class = {f"c{i:03d}": ("a" if i % 2 else "b")
                         for i in range(60)}
        s0 = psc.stratified_sample(clip_to_class, 10, seed=0)
        s0b = psc.stratified_sample(clip_to_class, 10, seed=0)
        s1 = psc.stratified_sample(clip_to_class, 10, seed=1)
        assert s0 == s0b and s0 != s1

    def test_n_larger_than_corpus_clamps(self):
        s = psc.stratified_sample({"c1": "a", "c2": "b"}, 50, seed=0)
        assert len(s) == 2

    def test_missing_road_col_fails_loudly(self, tmp_path):
        import pandas as pd
        p = tmp_path / "r.parquet"
        pd.DataFrame([{"clip_id": "c", "task": "vqa"}]).to_parquet(p)
        with pytest.raises(SystemExit):
            psc.main(["--records", str(p), "--out",
                      str(tmp_path / "x.json")])


# =============================================================================
# renderer smoke (frame-compose only — no ffmpeg, no video files)
# =============================================================================

class TestRenderer:
    def _full_record(self):
        sign = make_sign()                              # frame_idx 3
        agent = {"class": "car", "position_rel": "ahead",
                 "behaviour": "braking", "bbox": [200.0, 120.0, 280.0, 180.0],
                 "frame_idx": 3, "bev_xy": [22.0, -1.5]}
        rec = make_record(signs=[sign], agents=[agent],
                          actions=[make_action()])
        rec["engine_a"] = engine_a_with(lc_token="lc_left", lc_arc=30.0,
                                        lon_token="stop_at_point")
        m = np.zeros((252, 448), bool)
        m[130:170, 210:270] = True
        rec["sam"] = {"skipped": False, "keyframes": [3],
                      "instances": [{"mask_ref": "sam_agent_0_f3",
                                     "class_prompt": "agent",
                                     "source_index": 0, "frame_idx": 3,
                                     "score": 0.9,
                                     "bbox": pp.mask_bbox(m),
                                     "rle": pp.rle_encode(m)}]}
        rec["alpamayo"] = {"n_rows": 2, "tasks": {"meta_action": [{}]},
                           "meta_actions": ["slow down, keep lane"]}
        pp.fusion_gate(rec, rec["engine_a"])
        return rec

    def test_compose_frame_shapes(self):
        rec = self._full_record()
        frames = [np.full((252, 448, 3), 90, np.uint8) for _ in range(2)]
        outs = [pov.compose_frame(f, rec, i + 2, n_past=16, fps=2.0)
                for i, f in enumerate(frames)]
        for out in outs:
            assert out.dtype == np.uint8
            assert out.ndim == 3 and out.shape[2] == 3
            assert out.shape[1] == pov.CAM_W + pov.TEXT_W
            assert out.shape[0] >= 640
        assert outs[0].shape == outs[1].shape

    def test_compose_frame_draws_overlays_on_claim_frame(self):
        """The frame carrying the claims (idx 3) must differ from a claim-free
        frame in the camera pane region — i.e. overlays were drawn."""
        rec = self._full_record()
        base = np.full((252, 448, 3), 90, np.uint8)
        with_claims = pov.compose_frame(base, rec, 3, n_past=16)
        without = pov.compose_frame(base, rec, 4, n_past=16)
        cam_h = with_claims.shape[0] - pov.BEV_H
        # compare below the HUD strip, above the BEV pane
        assert (with_claims[30:cam_h, :pov.CAM_W] !=
                without[30:cam_h, :pov.CAM_W]).any()

    def test_compose_frame_engine_a_null(self):
        rec = make_record(actions=[make_action()])
        rec["engine_a"] = None
        rec["alpamayo"] = None
        rec["sam"] = None
        pp.fusion_gate(rec, None)
        out = pov.compose_frame(np.zeros((252, 448, 3), np.uint8),
                                rec, 0, n_past=16)
        assert out.dtype == np.uint8 and out.ndim == 3

    def test_disputed_action_renders_red(self):
        rec = make_record(actions=[make_action("prepare_lane_change",
                                               "right")])
        rec["engine_a"] = engine_a_with(lc_token="lc_left")
        rec["alpamayo"] = None
        rec["sam"] = None
        pp.fusion_gate(rec, rec["engine_a"])
        out = pov.compose_frame(np.zeros((252, 448, 3), np.uint8),
                                rec, 0, n_past=16)
        panel = out[:, pov.CAM_W:]
        red = pov.C_DISPUTED
        hits = ((panel[:, :, 0] == red[0]) & (panel[:, :, 1] == red[1])
                & (panel[:, :, 2] == red[2]))
        assert hits.any()

    def test_mask_contour(self):
        m = np.zeros((20, 20), bool)
        m[5:15, 5:15] = True
        pts = pov._mask_contour(m)
        assert pts.shape[0] == 36                       # 10x10 square boundary
        assert pov._mask_contour(np.zeros((5, 5), bool)).shape == (0, 2)


# =============================================================================
# module hygiene — pod-only deps stay lazy
# =============================================================================

class TestLazyImports:
    def test_no_heavy_modules_required(self):
        # importing the three modules must not have pulled in pod-only deps
        for mod in ("transformers", "sam2", "cv2", "imageio_ffmpeg"):
            assert mod not in sys.modules, f"{mod} imported at module scope"

    def test_resolve_clip_forms(self):
        assert pp._resolve_clip("abc", "/root") == ("abc", "/root/abc.mp4")
        assert pp._resolve_clip("/x/y/v123.mp4", None) == \
            ("v123", "/x/y/v123.mp4")
        assert pp._resolve_clip({"clip_id": "c9", "video": "/v/c9.mp4"},
                                None) == ("c9", "/v/c9.mp4")
        assert pp._resolve_clip("abc", None) == ("abc", None)


# --------------------------------------------------------------------------- #
# --no-vlm: engine B disabled, the other three engines still run               #
# --------------------------------------------------------------------------- #
def test_null_vlm_returns_empty_extraction_and_is_labelled():
    """The failure this guards: on 2026-08-12 every VLM arm was unusable for a
    reason unrelated to the pipeline (text-only model, OOM, broken w4a16), and
    the whole pilot reported 0/8 ok — indistinguishable from a pipeline bug."""
    import ph0_pilot as P
    v = P.NullVLM()
    obj, valid = v.chat_json([], "anything")
    assert obj == {} and valid is False
    assert "no-vlm" in v.model_id, "provenance must say engine B was disabled"


def test_null_vlm_record_cannot_be_mistaken_for_an_extraction():
    """An empty record must carry NO scenario/sign content — a --no-vlm artifact
    is pipeline validation, never a vocabulary-extraction result."""
    import ph0_pilot as P
    obj, _ = P.NullVLM().chat_json([], "p")
    for k in ("scenario", "domain", "signs", "strategic"):
        assert not (obj.get(k) or {}), f"{k} must be empty under --no-vlm"


def test_no_vlm_flag_exists_and_defaults_off():
    import ph0_pilot as P
    ap = P.build_parser() if hasattr(P, "build_parser") else None
    if ap is None:
        import inspect
        src = inspect.getsource(P.main)
        assert '"--no-vlm"' in src and "action=\"store_true\"" in src
    else:
        assert ap.parse_args(["--clips", "c", "--out", "o"]).no_vlm is False


# --------------------------------------------------------------------------- #
# decoder thread pinning — the swscaler EAGAIN fix                             #
# --------------------------------------------------------------------------- #
def test_decoder_threads_defaults_to_one():
    """MEASURED on pod4: every VLM arm died with
    `BlockingIOError: [Errno 11] ... [swscaler] Failed initializing scaling
    graph`. EAGAIN there is thread-creation failure inside libswscale — nproc
    reports the HOST's 96 CPUs, so ffmpeg sizes its pool to 96 and collides
    with the container's real allowance. Video decode here is a handful of
    frames at 2 fps, so threads buy nothing and cost the entire run."""
    import ph0_pilot
    assert ph0_pilot._decoder_threads() == 1


def test_decoder_threads_is_overridable(monkeypatch):
    import ph0_pilot
    monkeypatch.setenv("PH0_DECODER_THREADS", "4")
    assert ph0_pilot._decoder_threads() == 4


def test_pyav_path_pins_thread_count_before_decoding():
    """A source guard: the pin must be set on the stream BEFORE any decode call,
    otherwise libswscale has already allocated its pool."""
    import inspect

    import ph0_pilot
    src = inspect.getsource(ph0_pilot._read_frames_at)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "thread_count = _decoder_threads()" in code
    assert code.index("thread_count") < code.index("container.decode("), \
        "thread_count must be pinned before the first decode"
    assert "cv2.setNumThreads" in code, "the cv2 path needs the same pin"


# --------------------------------------------------------------------------- #
# engine B auto-class selection — the "text-only" misdiagnosis                 #
# --------------------------------------------------------------------------- #
def test_vlm_auto_classes_try_vision_before_text_only():
    """MEASURED on pod4: loading a VLM through AutoModelForCausalLM SUCCEEDS
    (weights load, 16.68 GB VRAM, no traceback) and then generate() rejects
    ['mm_token_type_ids','pixel_values_videos','video_grid_thw'] because the
    text-only class has no vision tower. 8/8 clips failed in 0.6 s while the run
    exited 0 — so a capable checkpoint presented as a model-availability
    problem. Text-only must be the LAST resort, never the first choice."""
    from ph0_pilot import VLM_AUTO_CLASSES
    assert VLM_AUTO_CLASSES[-1] == "AutoModelForCausalLM"
    assert "AutoModelForImageTextToText" in VLM_AUTO_CLASSES
    assert VLM_AUTO_CLASSES.index("AutoModelForImageTextToText") < \
        VLM_AUTO_CLASSES.index("AutoModelForCausalLM")


def test_vlm_engine_records_which_auto_class_loaded(monkeypatch):
    """The chosen class must be recorded, not just used — otherwise the next
    reader cannot tell a video run from a text-only one."""
    import types

    import ph0_pilot

    class _M:
        def eval(self): return self
    fake = types.SimpleNamespace(
        __version__="5.15.0",
        AutoProcessor=types.SimpleNamespace(
            from_pretrained=staticmethod(lambda *a, **k: object())),
        AutoModelForImageTextToText=types.SimpleNamespace(
            from_pretrained=staticmethod(lambda *a, **k: _M())),
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake)
    monkeypatch.setitem(__import__("sys").modules, "torch",
                        types.SimpleNamespace())
    eng = ph0_pilot.VLMEngine("some/model")
    assert eng.auto_class == "AutoModelForImageTextToText"


def test_vlm_engine_raises_when_no_auto_class_works(monkeypatch):
    import types

    import ph0_pilot
    fake = types.SimpleNamespace(
        __version__="5.15.0",
        AutoProcessor=types.SimpleNamespace(
            from_pretrained=staticmethod(lambda *a, **k: object())))
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake)
    monkeypatch.setitem(__import__("sys").modules, "torch",
                        types.SimpleNamespace())
    with pytest.raises(RuntimeError, match="no usable auto-class"):
        ph0_pilot.VLMEngine("some/model")
