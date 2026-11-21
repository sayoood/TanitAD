"""Tests for the band-label stream (A). Run with the env of ``run_tests.cmd`` (CPU, PYTHONPATH = ladder export)."""
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_strategic_geom as S  # noqa: E402
import build_v9_verbatim as B  # noqa: E402
import strategic_ck as CK  # noqa: E402

OUT = os.environ.get("BAND_OUT", "F:/strat_out/band_labels")
CACHE = "E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/"
H = np.array(CK.HORIZONS_S)


def straight(v=10.0, t_end=140.0, hz=100.0, gap=None):
    ts = np.arange(0.0, t_end + 1e-9, 1.0 / hz)
    if gap:
        ts = ts[(ts < gap[0]) | (ts > gap[1])]
    z = np.zeros_like(ts)
    return B.log_from_arrays(ts, v * ts, z, z, np.full_like(ts, v))


# ------------------------------------------------------------------------------------------------ closed form
def test_straight_constant_speed_closed_form():
    v = 10.0
    log = straight(v)
    nows = np.array([0.0, 5.0, 40.0, 100.0])
    xyh, valid, why = CK.checkpoints(log, nows)
    exp = v * H[None, :] * np.ones((len(nows), 1))
    assert valid[:3].all(), why[:3]
    assert np.allclose(xyh[:3, :, 0], exp[:3], atol=1e-3)
    assert np.allclose(xyh[:3, :, 1], 0.0, atol=1e-6) and np.allclose(xyh[:3, :, 2], 0.0, atol=1e-6)
    assert valid[3].all()                       # 100 + 30 = 130 s, arc 1300 m < 1400 - 24 m


def _ramp_log():
    ts = np.arange(0.0, 140.0 + 1e-9, 0.01)
    z = np.zeros_like(ts)
    return B.log_from_arrays(ts, ts ** 2 / 4.0, z, z, ts / 2.0)       # v = t/2, x = t^2/4


def test_mutation_time_offset_goes_red():
    """Closed form on a ramp-speed log: x(NOW + h) - x(NOW). A NOW read 0.5 s late (the mutation) must break it."""
    log = _ramp_log()
    now = 30.0
    exp = ((now + H) ** 2 - now ** 2) / 4.0
    xyh, _, _ = CK.checkpoints(log, np.array([now]))
    assert np.allclose(xyh[0, :, 0], exp, atol=1e-2)                  # control: the true clock is green
    mut, _, _ = CK.checkpoints(log, np.array([now + 0.5]))
    assert not np.allclose(mut[0, :, 0], exp, atol=1e-2)             # mutated clock is red
    assert np.abs(mut[0, :, 0] - exp).max() > 1.0


def test_ramp_speed_time_index():
    """v(t) = 2 t: x(t) = t^2. The checkpoint at NOW + h is x(NOW+h) - x(NOW) in the (straight) ego frame."""
    ts = np.arange(0.0, 140.0 + 1e-9, 0.01)
    z = np.zeros_like(ts)
    log = B.log_from_arrays(ts, ts ** 2 / 4.0, z, z, ts / 2.0)       # v = t/2, x = t^2/4
    now = 30.0
    xyh, valid, why = CK.checkpoints(log, np.array([now]))
    assert valid.all(), why
    exp = ((now + H) ** 2 - now ** 2) / 4.0
    assert np.allclose(xyh[0, :, 0], exp, atol=1e-2)


def test_circle_heading_and_side():
    """Constant-yaw-rate left turn: heading increases with h and y > 0."""
    R, v = 300.0, 10.0
    ts = np.arange(0.0, 140.0 + 1e-9, 0.01)
    th = v * ts / R
    log = B.log_from_arrays(ts, R * np.sin(th), R * (1 - np.cos(th)), th, np.full_like(ts, v))
    xyh, valid, _ = CK.checkpoints(log, np.array([10.0]), tr=None)
    assert valid[0].all()
    assert (xyh[0, :, 1] > 0).all() and np.all(np.diff(xyh[0, :, 2]) > 0)
    assert np.allclose(xyh[0, :, 2], v * H / R, atol=0.02)          # heading = arc / R (smoothing keeps the tangent)


# ------------------------------------------------------------------------------------------------ masks
def test_mask_log_end_and_kernel_edge():
    log = straight(10.0, t_end=140.0)
    xyh, valid, why = CK.checkpoints(log, np.array([112.0, 108.0, 100.0]))
    assert why[0, -1] == 2 and not valid[0, -1]                       # 112 + 30 > 140: beyond log
    assert why[1, -1] == 4 and not valid[1, -1]                       # 108 + 30 = 138: arc 1380 > 1400 - 24
    assert valid[2].all()
    assert np.isnan(xyh[~valid]).all() and np.isfinite(xyh[valid]).all()


def test_mask_gap():
    log = straight(10.0, gap=(50.0, 53.0))
    xyh, valid, why = CK.checkpoints(log, np.array([40.0, 54.0]))
    # NOW = 40: h = 6, 9 -> T = 46, 49 (before the gap at 50) valid; h >= 12 spans the gap
    assert valid[0, :2].all() and not valid[0, 2:].any() and (why[0, 2:] == 3).all()
    assert valid[1].all()                                              # after the gap: nothing overlaps


def test_mask_now_before_log():
    log = straight(10.0)
    xyh, valid, why = CK.checkpoints(log, np.array([-0.1, 0.0]))
    assert (why[0] == 1).all() and not valid[0].any()
    assert valid[1].all()


def test_static_clip_no_path():
    ts = np.arange(0.0, 60.0 + 1e-9, 0.01)
    z = np.zeros_like(ts)
    xyh, valid, why = CK.checkpoints(B.log_from_arrays(ts, z, z, z, z), np.array([0.0]))
    assert not valid.any() and (why == 6).all()


# ------------------------------------------------------------------------------------------------ real clips
def _real_clips(n=14):
    return [u for u in S.clip_universe("eval") if u[6]][:n]


def test_clock_and_join_on_real_clips():
    """The cache's ``pose_grid[j]`` (absolute log pose at v2ep frame 2 j) equals the ego log read at the v9 clock
    ``grid_start + 2 j dt``; a 0.2 s clock offset is far outside (mutation)."""
    import torch
    err_ok, err_mut = [], []
    for cid, g0, dt, src, n, inm, inc in _real_clips():
        pg = torch.load(CACHE + f"clips/{cid}.pt", weights_only=True)["pose_grid"].double().numpy()
        assert len(pg) == n == 101
        log = B.load_log(S.EGO.format(cid))
        now = g0 + 2 * np.arange(len(pg)) * dt
        for off, bucket in ((0.0, err_ok), (0.2, err_mut)):
            x, y, yaw, v = B.sample(log, now + off)
            bucket.append(float(np.abs(v - pg[:, 3]).mean()))
            if off == 0.0:
                assert np.hypot(x - pg[:, 0], y - pg[:, 1]).max() < 0.05
    assert max(err_ok) < 0.01
    assert min(err_mut) > 0.02 and np.mean(err_mut) > 10 * np.mean(err_ok)       # mutated clock goes red


def test_join_key_k_equals_2j():
    """v9 rows are addressed by (sid, k) with k the raw frame; our step j reads k = 2 j (``v8f_mini_data.py:9-12``)."""
    z = np.load(f"{OUT}/strategic_geom_eval.npz")
    j = np.concatenate([np.arange(n) for n in z["clip__n"]])
    assert (z["k"] == 2 * j).all()
    v9 = np.load("E:/Projects/TanitAD-artifacts/v9labels/v9_labels_eval139.npz")
    sid_v9 = {int(s): (int(o), int(n)) for s, o, n in zip(v9["clip__sid"], v9["clip__row0"], v9["clip__n_rows"])}
    hit = 0
    for i, sid in enumerate(z["clip__sid"]):
        if int(sid) in sid_v9:
            o9, n9 = sid_v9[int(sid)]
            k9 = v9["row__k"][o9:o9 + n9]
            o, n = int(z["clip__off"][i]), int(z["clip__n"][i])
            assert set(z["k"][o + 1:o + n].tolist()) <= set(k9.tolist())       # steps 1.. have v9 rows (k = 2 .. 200, every frame)
            # same clock: v9 now_s at k == ours
            a = dict(zip(k9.tolist(), v9["row__now_s"][o9:o9 + n9].tolist()))
            assert all(abs(a[int(kk)] - nn) < 1e-9 for kk, nn in zip(z["k"][o + 1:o + n], z["now_s"][o + 1:o + n]))
            hit += 1
    assert hit >= 100


def test_vectorised_gap_and_sagitta_match_originals():
    for cid, g0, dt, *_ in _real_clips(6):
        log = B.load_log(S.EGO.format(cid))
        tr = B.track10(log)
        s_nat = B.s_at(tr, log.ts)
        sg = CK.chord_sagitta_array(log)
        nows = g0 + 2 * np.arange(0, 101, 7) * dt
        T = nows[:, None] + H[None, :]
        gk = CK._gap_ok(log.ts, nows, T, 1.0)
        for a in range(len(nows)):
            for b in range(len(H)):
                assert gk[a, b] == (B.max_gap(log, nows[a], T[a, b]) <= 1.0)
        s_t = B.s_at(tr, T)
        lo, hi = s_t - 24.0, s_t + 24.0
        ok = CK._sagitta_footprint_ok(sg, s_nat, lo, hi, 0.25)
        for a in range(len(nows)):
            for b in range(len(H)):
                assert ok[a, b] == (B.chord_sagitta_max(tr, log, lo[a, b], hi[a, b], s_nat) <= 0.25)


# ------------------------------------------------------------------------------------------------ schema pin
STEP_KEYS = {"k": ("int16", ()), "now_s": ("float64", ()), "ck_xyh": ("float32", (9, 3)), "ck_valid": ("bool", (9,)),
             "ck_why": ("uint8", (9,)),
             "ck_valid_span": ("bool", (9,)), "ck_gap_s": ("float32", (9,))}
CLIP_KEYS = {"clip__sid": "int64", "clip__off": "int64", "clip__n": "int64", "clip__grid_start_s": "float64",
             "clip__dt_s": "float64", "clip__clock_src": "int64", "clip__log_end_s": "float64",
             "clip__in_v9_manifest": "int64", "clip__in_cache141": "int64"}


@pytest.mark.parametrize("split", ["eval", "train"])
def test_schema_pin(split):
    p = f"{OUT}/strategic_geom_{split}.npz"
    if not os.path.exists(p):
        pytest.skip("not built yet")
    z = np.load(p)
    assert set(z.files) == {"clip_ids", "horizons_s", *STEP_KEYS, *CLIP_KEYS}
    N = int(z["clip__n"].sum())
    for k, (dt, shp) in STEP_KEYS.items():
        assert z[k].dtype.name == dt and z[k].shape == (N, *shp), k
    for k, dt in CLIP_KEYS.items():
        assert z[k].dtype.name == dt and z[k].shape == (len(z["clip_ids"]),), k
    assert np.allclose(z["horizons_s"], [6, 9, 12, 15, 18, 21, 24, 27, 30])
    assert (z["clip__off"] == np.concatenate([[0], np.cumsum(z["clip__n"])[:-1]])).all()
    v = z["ck_valid"]
    assert np.isfinite(z["ck_xyh"][v]).all() and np.isnan(z["ck_xyh"][~v]).all()
    assert ((z["ck_why"] == 0) == v).all()
    vs = z["ck_valid_span"]
    assert (vs | ~v).all()                                            # strict implies span-only
    g = z["ck_gap_s"]
    assert ((g <= 1.0) | np.isnan(g))[v].all() and (g[vs & ~v] > 1.0).all()      # the gap rule is the only difference
    man = json.load(open(f"{OUT}/strategic_geom_{split}.manifest.json", encoding="utf-8"))
    assert man["schema"] == "tanitad.band_labels.strategic_geom/1" and man["n_rows"] == N
    assert B.file_md5(p) == man["npz_md5"]


def test_span_mask_and_gap_matrix_on_synthetic_gap():
    log = straight(10.0, gap=(50.0, 53.0))
    nows = np.array([40.0, 54.0])
    xyh, valid, why = CK.checkpoints(log, nows)
    _, vspan, _ = CK.checkpoints(log, nows, g_max=float("inf"))
    gap = CK.max_gap_matrix(log.ts, nows)
    assert vspan[0].all() and not valid[0].all()
    assert np.allclose(gap[0, 2:], 3.01, atol=0.02) and np.allclose(gap[0, :2], 0.01, atol=0.005)
    assert np.allclose(gap[1], 0.01, atol=0.005)
