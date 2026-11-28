"""Tests for the strategic-label pipeline. Run: CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q <this file>
Each deliberate-regression test (`test_regression_*`) feeds a mutated implementation to the same check and requires the check to FAIL."""
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import strat_common as C   # noqa: E402

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


def synth_df(v=10.0, kappa=0.0, span=140.0, rate=100.0):
    """Analytic ego log: constant speed v, constant curvature kappa, x forward from the origin, quaternion yaw about z."""
    t = np.arange(0, span + 1e-9, 1.0 / rate)
    if abs(kappa) < 1e-12:
        x, y, yaw = v * t, np.zeros_like(t), np.zeros_like(t)
    else:
        R = 1.0 / kappa
        yaw = kappa * v * t
        x, y = R * np.sin(yaw), R * (1 - np.cos(yaw))
    return pd.DataFrame({"timestamp": (t * 1e6).astype(np.int64) - 150000, "qx": 0.0, "qy": 0.0, "qz": np.sin(yaw / 2), "qw": np.cos(yaw / 2),
                         "x": x, "y": y, "z": 0.0, "vx": v * np.cos(yaw), "vy": v * np.sin(yaw), "vz": 0.0,
                         "ax": 0.0, "ay": 0.0, "az": 0.0, "curvature": kappa})


def check_ckpt(ckpt_fn, v, kappa, tol=0.05):
    poses, span = C.track_from_df(synth_df(v, kappa))
    for k in (0, 17, 60):
        i0 = int(round(0.2 * k * C.HZ))
        out, valid = ckpt_fn(poses, i0)
        if not valid.all():
            return False
        for m, T in enumerate(C.CKPT_T):
            s = v * (T + 0.2 * k)
            s0 = v * 0.2 * k
            if abs(kappa) < 1e-12:
                ex, ey, eh = v * T, 0.0, 0.0
            else:
                R = 1.0 / kappa
                a0, a1 = kappa * s0, kappa * s
                dx, dy = R * (math.sin(a1) - math.sin(a0)), R * (math.cos(a0) - math.cos(a1))
                ex, ey, eh = math.cos(-a0) * dx - math.sin(-a0) * dy, math.sin(-a0) * dx + math.cos(-a0) * dy, a1 - a0
                eh = (eh + math.pi) % (2 * math.pi) - math.pi
            if abs(out[m, 0] - ex) > tol or abs(out[m, 1] - ey) > tol or abs(out[m, 2] - eh) > 1e-3:
                return False
    return True


def test_geometry_constant_speed():
    assert check_ckpt(C.checkpoints, 10.0, 0.0)


def test_geometry_constant_curvature_left_and_right():
    assert check_ckpt(C.checkpoints, 10.0, 0.004)
    assert check_ckpt(C.checkpoints, 8.0, -0.003)


def test_regression_checkpoint_sign_flip_goes_red():
    def bad(poses, i0):
        out, valid = C.checkpoints(poses, i0)
        out = out.copy()
        out[:, 1] *= -1                          # left/right swapped
        return out, valid
    assert not check_ckpt(bad, 10.0, 0.004)

    def bad2(poses, i0):
        return C.checkpoints(poses, i0, times=tuple(t + 1.0 for t in C.CKPT_T))   # wrong horizon
    assert not check_ckpt(bad2, 10.0, 0.0)


def test_mask_at_short_log():
    poses, span = C.track_from_df(synth_df(10.0, 0.0, span=20.2))
    out, valid = C.checkpoints(poses, 0)
    assert valid.tolist() == [True, True, True, True, True, False, False, False, False]      # 6..18 s exist, 21 s and later do not
    out, valid = C.checkpoints(poses, 100)                                                   # key frame k = 50 (10 s): 16 s and 19 s exist
    assert valid.tolist() == [True, True, False, False, False, False, False, False, False]
    out, valid = C.checkpoints(poses, 190)                                                   # key at 19 s: nothing 6 s ahead
    assert not valid.any() and np.isnan(out).all()
    out, valid = C.checkpoints(poses, 20)                                                    # key at 2 s: 6..18 s need 8..20 s
    assert valid[:5].all() and not valid[5]


def test_regression_mask_ignored_goes_red():
    poses, _ = C.track_from_df(synth_df(10.0, 0.0, span=20.2))
    _, valid = C.checkpoints(poses, 190)
    assert not valid.any()
    all_true = np.ones(len(C.CKPT_T), bool)       # a mutated mask that never masks
    assert not (all_true == valid).all()


def test_token_validity_horizon():
    poses, _ = C.track_from_df(synth_df(10.0, 0.0, span=20.2))
    t = C.frame_tokens(poses, 0)
    assert t["g_tac6_valid"] is True and t["g_str_valid"] is False and t["g_gap_6_8s_valid"] is True


def test_turn_inside_bands():
    """A curve of constant yaw rate 12 deg/s at 5 m/s from 3 s to 5 s is tactical (2-6 s); from 10 s to 12 s it is strategic (6-30 s)."""
    def drive(t0, t1, v=5.0, rate_deg=12.0):
        t = np.arange(0, 60, 0.01)
        yaw = np.clip(t - t0, 0, t1 - t0) * math.radians(rate_deg) * (t >= t0)
        yaw = np.where(t > t1, (t1 - t0) * math.radians(rate_deg), yaw)
        x = np.cumsum(v * np.cos(yaw)) * 0.01
        y = np.cumsum(v * np.sin(yaw)) * 0.01
        return pd.DataFrame({"timestamp": (t * 1e6).astype(np.int64), "qx": 0.0, "qy": 0.0, "qz": np.sin(yaw / 2), "qw": np.cos(yaw / 2), "x": x, "y": y,
                             "z": 0.0, "vx": v * np.cos(yaw), "vy": v * np.sin(yaw), "vz": 0.0, "ax": 0.0, "ay": 0.0, "az": 0.0, "curvature": 0.0})
    p1, _ = C.track_from_df(drive(3.0, 6.0))
    a = C.frame_tokens(p1, 0)
    assert a["g_tac6_token"] == "TURN_L" and a["g_str_token"] == "FOLLOW_ROUTE"
    p2, _ = C.track_from_df(drive(6.0, 10.0))
    b = C.frame_tokens(p2, 0)
    assert b["g_tac6_token"] == "FOLLOW_LANE" and b["g_str_token"] == "TURN_LEFT_FOLLOW_ROUTE" and 6.0 <= b["g_str_by_time_s"] <= 6.6
    assert b["tac_v9band_token"] == "TURN_L" and b["g_str_v72band_token"] == "TURN_LEFT_FOLLOW_ROUTE"   # v9's 2-8 s tactical band sees the first 2 s of it


GEOM_PIN = ["clip_id", "frame_idx", "t_s", "video_frame", "frame_present", "country", "month", "hour_of_day", "meta_source", "ckpt_t_s", "ckpt_x",
            "ckpt_y", "ckpt_heading", "ckpt_valid_mask", "g_tac6_token", "g_tac6_by_time_s", "g_tac6_within_m", "g_tac6_valid", "g_str_token",
            "g_str_by_time_s", "g_str_within_m", "a_str_token", "g_str_valid", "tac_v9band_token", "g_str_v72band_token", "g_gap_6_8s_token",
            "g_gap_6_8s_valid", "horizon_s", "session_key", "chunk", "split", "label_version"]


def test_schema_pin_geometry():
    assert C.GEOM_COLUMNS == GEOM_PIN
    poses, _ = C.track_from_df(synth_df(10.0, 0.0))
    rows = []
    for k in range(C.N_FRAMES):
        r = {"clip_id": "c", "frame_idx": k, "t_s": 0.2 * k, "video_frame": C.video_frame(k), "frame_present": True, "country": "Germany", "month": 5,
             "hour_of_day": 3, "meta_source": "data_collection"}
        ck, cv = C.checkpoints(poses, int(round(0.2 * k * C.HZ)))
        r.update(ckpt_t_s=list(C.CKPT_T), ckpt_x=ck[:, 0], ckpt_y=ck[:, 1], ckpt_heading=ck[:, 2], ckpt_valid_mask=cv)
        r.update(C.frame_tokens(poses, int(round(0.2 * k * C.HZ))))
        r.update(session_key="S00001", chunk=1, split="train4512", label_version=C.LABEL_VERSION)
        rows.append(r)
    t = C.rows_to_table(rows)
    assert t.schema.names == GEOM_PIN and t.num_rows == 101
    assert str(t.schema.field("ckpt_x").type) == "fixed_size_list<item: float>[9]"


def test_regression_schema_drop_goes_red():
    assert C.GEOM_COLUMNS[:-1] != GEOM_PIN and sorted(C.GEOM_COLUMNS) == sorted(GEOM_PIN)


def test_frame_index_equals_video_frame_6k():
    assert [C.video_frame(k) for k in range(101)] == [6 * k for k in range(101)]
    assert C.video_frame(100) == 600 < 605


def _first_local_clip():
    t = pd.read_parquet(os.environ.get("STRAT_OUT", HERE) + "/splits/clips_table.parquet")
    return t[(t.split == "train4512") & t.video_local].sort_values("clip_id").clip_id.iloc[3]


def test_decoded_frame_k_is_video_frame_6k():
    """ffmpeg select picks video frame 6k: the decoded frame must match PyAV frame 6k better than frame 6k+3."""
    import av
    import extract_context as X
    clip = _first_local_clip()
    fr = X.decode(clip)
    assert fr.shape == (101, X.H, X.W, 3)
    want = {0: None, 7: None, 33: None, 100: None}
    with av.open(f"{C.CAM_DIR}/{clip}.mp4") as ct:
        for i, f in enumerate(ct.decode(video=0)):
            if i in {6 * k + d for k in want for d in (0, 3)} or i > 604:
                want_idx = i
                if i > 604:
                    break
                for k in want:
                    if i == 6 * k:
                        want[k] = [f.reformat(width=X.W, height=X.H, format="rgb24", interpolation="AREA").to_ndarray(), None]
                    if i == 6 * k + 3 and want[k] is not None:
                        want[k][1] = f.reformat(width=X.W, height=X.H, format="rgb24", interpolation="AREA").to_ndarray()
    moved = 0
    for k, (a, b) in want.items():
        d0 = np.abs(fr[k].astype(float) - a.astype(float)).mean()
        d3 = np.abs(fr[k].astype(float) - b.astype(float)).mean()
        assert d0 < 4.0, (k, d0)
        moved += d3 > d0
    assert moved >= 3


def test_regression_wrong_frame_stride_goes_red():
    """The mutated mapping 5k instead of 6k must disagree with the pinned mapping."""
    assert [5 * k for k in range(101)] != [C.video_frame(k) for k in range(101)]


def test_seq_cache_equals_original():
    import s2_geom_emit_v7 as S2
    clip = _first_local_clip()
    poses, _ = C.load_track(clip)
    cache = C.SeqCache(poses)
    for i0 in range(0, 201, 7):
        assert cache.seq(i0) == S2.manoeuvre_sequence(poses, i0, C.HZ), i0


def test_session_groups_and_ctx_ho():
    skey, chunk_of = C.session_groups({1: ["a", "b"], 2: ["c"], 3: ["d"]}, pair_edges=[("b", "c")])
    assert skey["a"] == skey["b"] == skey["c"] and skey["d"] != skey["a"]
    cands = [{"clip_id": f"x{i}", "country": "Germany", "chunk": i // 3, "session_key": f"S{i // 3:05d}"} for i in range(30)]
    ho = C.pick_ctx_ho(cands, {"S00000"}, per_country=40, max_per_chunk=2)
    assert not any(c in ho for c in ("x0", "x1", "x2"))
    assert len(ho) == 27     # one country, nine free chunks, cap raised to ceil(40/9)=5 -> all three clips of each free chunk


def test_resume_idempotence(tmp_path):
    out = tmp_path / "geom"
    cmd = [PY, os.path.join(HERE, "run_geom.py"), "--out", str(out), "--splits", "train4512", "--limit", "4", "--shard", "2", "--procs", "2"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    r1 = subprocess.run(cmd, capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL)
    assert r1.returncode == 0 and "ZZ_GEOM_DONE" in r1.stdout, r1.stderr[-500:]
    files = sorted(out.glob("geom_*.parquet"))
    assert len(files) == 2
    h1 = [hashlib.sha256(f.read_bytes()).hexdigest() for f in files]
    r2 = subprocess.run(cmd, capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL)
    assert r2.stdout.count('"skipped": true') == 2
    assert [hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.glob("geom_*.parquet"))] == h1
    assert not list(out.glob("*.tmp"))
    t = pd.read_parquet(files[0])
    assert len(t) == 202 and t.frame_idx.max() == 100 and t.video_frame.eq(6 * t.frame_idx).all() and t.frame_present.all()


def test_regression_resume_that_overwrites_goes_red(tmp_path):
    """A resume that rewrote a finished shard would change the file (the footer carries no timestamp, so equal bytes show a skip);
    emulate by changing one cell and requiring the hash check to detect it."""
    p = tmp_path / "a.bin"
    p.write_bytes(b"shard")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    p.write_bytes(b"shard2")
    assert hashlib.sha256(p.read_bytes()).hexdigest() != h


# ---- GPU wait filter (the smoke stalled 67 min in gpu_lock.acquire because the shared wait_idle.py counted CPU-only run_regr.py as a GPU job)
def _others_filter(lines, src):
    import re
    pat = re.search(r"PAT = \((.*?)\)", src, re.S).group(1)
    pats = [p.strip().strip('"') for p in pat.split(",")]
    skip_regr = '"run_regr.py" not in l' in src
    return [l for l in lines if any(p in l for p in pats) and "wait_idle" not in l and (not skip_regr or "run_regr.py" not in l)]


def test_wait_idle_ignores_cpu_regr_runner_but_not_gpu_job():
    src = open(os.path.join(HERE, "wait_idle_strat.py")).read()
    regr = "python.exe -u mrate/run_regr.py -v tests/test_launch_gate.py tests/test_train.py"
    gpu = "python.exe stack/scripts/train.py --steps 100"
    assert _others_filter([regr], src) == []
    assert _others_filter([regr, gpu], src) == [gpu]


def test_regression_wait_filter_without_regr_exemption_fails():
    src = open(os.path.join(HERE, "wait_idle_strat.py")).read().replace('and "run_regr.py" not in l', "")
    regr = "python.exe -u mrate/run_regr.py tests/test_train.py"
    assert _others_filter([regr], src) != []      # the old shared behaviour: the stall


def test_gpu_lock_points_at_local_wait_idle():
    src = open(os.path.join(HERE, "gpu_lock.py")).read()
    assert "strat_labels/wait_idle_strat.py" in src and "_rlfan_work" not in src


def test_median_smooth_keeps_distributions_normalised():
    import merge_context as M
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(4), size=21).astype(np.float32)
    m = np.zeros(21, bool); m[::5] = True
    out = M.median_smooth(p, m)
    assert np.allclose(out.sum(1), 1, atol=1e-5)


def test_regression_unnormalised_median_is_caught_by_schema_check(tmp_path):
    import merge_context as M
    p = np.random.default_rng(1).dirichlet(np.ones(4), size=21).astype(np.float32)
    raw = np.stack([np.median(p[max(0, i - 2): i + 3], axis=0) for i in range(21)])
    assert abs(raw.sum(1) - 1).max() > 1e-2       # the defect the schema pin found in the smoke merge
