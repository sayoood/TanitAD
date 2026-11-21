import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import merge_context as M
import merge_full as F


def part_a(n=101):
    rng = np.random.default_rng(0)
    z = lambda k: rng.dirichlet(np.ones(k), n).astype(np.float32)
    return {"emb": rng.standard_normal((n, 768)).astype(np.float16), **{f"t1_{k}": rng.random(n).astype(np.float32) for k in ("luma_mean", "luma_p05", "luma_p95", "sky_luma", "sat_mean", "headlight_frac", "lit_frac")},
            "z_illum": z(5), "z_weather": z(4), "z_surface": z(3), "z_viewq": z(5), "n_frames": np.int32(n)}


def part_b(n=21):
    rng = np.random.default_rng(1)
    z = lambda k: rng.dirichlet(np.ones(k), n).astype(np.float32)
    return {"v_weather": z(4), "v_surface": z(3), "v_visibility": z(3), "n_frames": np.int32(n)}


META = {"country": "Germany", "month": 6, "hour_of_day": 12}
SW = {"weather": 0.05, "surface": 0.01, "visibility": 0.002, "illumination": 0.05}


def test_sticky_is_a_distribution_and_tracks_evidence():
    p = np.tile([[0.9, 0.1]], (40, 1))
    p[20:] = [0.05, 0.95]
    q = M.sticky_posterior(p, 0.01)
    assert abs(q.sum(1) - 1).max() < 1e-5
    assert q.argmax(1).tolist() == [0] * 20 + [1] * 20


def test_sticky_removes_a_single_frame_flip():
    p = np.tile([[0.8, 0.2]], (40, 1))
    p[20] = [0.3, 0.7]
    assert M.sticky_posterior(p, 0.01).argmax(1).sum() == 0
    assert p.argmax(1).sum() == 1


def test_glare_is_folded_into_day():
    z = np.array([[0.1, 0.1, 0.1, 0.3, 0.4]], np.float32)
    o = F.fold_glare(z)
    assert o.shape == (1, 4) and abs(o[0, 3] - 0.7) < 1e-6 and abs(o.sum() - 1) < 1e-6


def test_clip_without_qwen_has_no_vlm_frame():
    r = F.clip_context("c", part_a(), None, {}, META, SW)
    assert not r["vlm_clip"].any() and not r["vlm_frame"].any()
    for f in ("weather", "surface", "illumination"):
        assert set(r[f + "_src"]) == {"zeroshot"}
    assert set(r["visibility_src"]) == {"none"} and r["illumination_p"].shape == (101, 4)


def test_clip_with_qwen_marks_only_the_1hz_frames():
    r = F.clip_context("c", part_a(), part_b(), {}, META, SW)
    assert r["vlm_clip"].all()
    assert np.where(r["weather_src"] == "vlm")[0].tolist() == list(range(0, 101, 5))
    assert set(r["illumination_src"]) == {"zeroshot"}      # no v_illum in this part


def test_short_video_keeps_the_vlm_grid_inside_the_clip():
    r = F.clip_context("c", part_a(100), part_b(20), {}, META, SW)
    assert np.where(r["weather_src"] == "vlm")[0].tolist() == list(range(0, 100, 5))


def test_probe_training_ignores_clips_without_qwen():
    data = {f"c{i}": (part_a(), part_b() if i < 6 else None) for i in range(9)}
    p = F.fit_probes(data, list(data))
    assert p["weather"][1] is not None and set(p["weather"][0]) == {f"c{i}" for i in range(6)}
    assert "illumination" not in p


def test_a_non_finite_qwen_answer_is_no_answer():
    b = part_b()
    for k in ("v_weather", "v_surface", "v_visibility"):
        b[k][3] = np.nan
    r = F.clip_context("c", part_a(), b, {}, META, SW)
    for f in ("weather", "surface", "visibility"):
        assert np.isfinite(r[f + "_p"]).all(), f
        assert r[f + "_src"][15] != "vlm" and r[f + "_src"][10] == "vlm", f


def test_probe_training_skips_a_non_finite_answer(monkeypatch):
    import sklearn.linear_model as L
    seen = []

    class Spy(L.LogisticRegression):
        def fit(self, X, y, sample_weight=None):
            seen.append(len(X))
            return super().fit(X, y, sample_weight=sample_weight)
    monkeypatch.setattr(L, "LogisticRegression", Spy)
    data = {f"c{i}": (part_a(), part_b()) for i in range(4)}
    data["c0"][1]["v_weather"][3] = np.nan
    F.fit_probes(data, list(data))
    assert seen[0] == 4 * 21 - 1, seen[:3]
