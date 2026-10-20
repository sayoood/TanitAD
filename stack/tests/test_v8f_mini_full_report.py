"""v8F-mini full RESULT renderer (code/v8f_mini_full_report.py). CPU only, < 60 s.

Run: PYTHONPATH=<repo>/stack;<repo>/taniteval pytest -q stack/tests/test_v8f_mini_full_report.py

What is pinned, and the control that must go red for each:
  * the bars are COPIED, never re-decided: the K-1 / WM-1 PASS words in the output are the verdict's;
  * the paired deltas are the paired episode-cluster bootstrap on the SAME windows, and M1 - M1 reads exactly +0 [0, 0];
  * a run on another grid (eid), a run whose saved ade6 is not ||pred - gt|| (wrong ground truth), a verdict whose WM-1 or K-1 point
    does not re-derive from the files, P10 predictions that do not reproduce floors_perwindow.npz, and a missing run are each REFUSED;
  * the default ground-truth rebuild over the repo's eval inputs is the 2,399-window, 141-episode eval-141 grid.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
REPORT = PKG / "code" / "v8f_mini_full_report.py"
pytestmark = pytest.mark.skipif(not REPORT.exists(), reason="renderer not in this checkout")
N_EP, W, H = 20, 10, 30


def _load():
    spec = importlib.util.spec_from_file_location("v8f_mini_full_report", REPORT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _paths(rng, n):
    """Smooth ego-frame paths: speed 5-15 m/s, a gentle constant curvature per window, 0.2 s steps."""
    t = np.arange(1, H + 1) * 0.2
    v = rng.uniform(5, 15, n)[:, None]
    k = rng.normal(0, 0.01, n)[:, None]
    s = v * t
    return np.stack([np.sin(k * s + 1e-9) / (k + 1e-9), (1 - np.cos(k * s)) / (k + 1e-9)], -1)


def _world(tmp, wm_off=0.0, k1_off=0.0, floors_off=0.0, bad_eid=None, bad_ade=None, drop=None, floors=True):
    rng = np.random.default_rng(3)
    n = N_EP * W
    eid = np.repeat(np.arange(N_EP), W)
    gt = _paths(rng, n)
    ev = tmp / "eval"
    ev.mkdir(parents=True)
    np.savez(tmp / "gt.npz", gt=gt, eid=eid, dt=0.2)
    p10 = gt + rng.normal(0, 0.5, gt.shape)
    np.save(tmp / "p10.npy", p10)
    ade = lambda p: np.linalg.norm(p - gt, axis=-1).mean(1)          # noqa: E731
    raw = {"tier": "T0-test", "families": {"lateral": {"ci": {"components": {"cross_mae_m": {"mean": 0.5, "lo": 0.4, "hi": 0.6}}}}}}
    means = {}
    for arm, sd in (("M1", 0.4), ("M2b", 0.3), ("M1-navshuf", 0.8)):
        for s in (0, 1):
            tag = f"{arm}_s{s}"
            if drop == tag:
                continue
            pred = gt + rng.normal(0, sd + 0.01 * s, gt.shape)
            a = ade(pred)
            means[tag] = a.mean()
            e = eid.copy()
            if bad_eid == tag:
                e = e[::-1].copy()
            if bad_ade == tag:
                a = a + 0.01
            np.savez(ev / f"eval_{tag}.npz", pred=pred, ade6=a, eid=e)
            (ev / f"eval_{tag}.json").write_text(json.dumps({"raw": raw}), encoding="utf-8")
    pm = ade(p10).mean()
    if floors:
        np.savez(ev / "floors_perwindow.npz", P10_refit__ade6=ade(p10) + floors_off)
        (ev / "floors_report.json").write_text(json.dumps({"floors": {"P10_refit": raw}}), encoding="utf-8")
    k1 = lambda arm: [{"mean": round(float(means.get(f"{arm}_s{s}", 0) - pm + k1_off), 4), "lo": -1, "hi": -0.5,  # noqa: E731
                       "separated": True} for s in (0, 1)]                 # rounded to 4 dp, as taniteval.ci rounds a paired delta
    v = {"tier": "T1-class open loop", "estimator": "e", "provenance": {"n_clips": 4512}, "summary": "K-1 PASS (M1 False, M2b True); WM-1 FAIL",
         "K1": {"per_arm": {a: {"K1_ade6_minus_P10": k1(a), "seed_floor_F_m": 0.01, "K1_pass": a == "M2b"} for a in ("M1", "M2b")},
                "pass": True, "multiplicity": "2 arms read"},
         "WM1": {"per_seed": [{"seed": s, "mean": round(float(means.get(f"M2b_s{s}", 0) - means.get(f"M1_s{s}", 0) + wm_off), 4), "lo": -0.2,
                               "hi": -0.05, "separated": True} for s in (0, 1)],
                 "seed_floor_F_m": {"M1": 0.01, "M2b": 0.02, "max": 0.02}, "pass": False, "n_windows": n, "n_episodes": N_EP},
         "next": ["K-1 PASS => next"]}
    (tmp / "verdict.json").write_text(json.dumps(v), encoding="utf-8")
    return tmp


def _main(tmp, capsys=None):
    out = tmp / "RESULT.md"
    rc = _load().main(["--verdict", str(tmp / "verdict.json"), "--eval-dir", str(tmp / "eval"), "--out", str(out), "--json",
                       str(tmp / "r.json"), "--gt-npz", str(tmp / "gt.npz"), "--p10-pred", str(tmp / "p10.npy"), "--n-boot", "200"])
    return rc, out.read_text(encoding="utf-8"), json.loads((tmp / "r.json").read_text(encoding="utf-8"))


def test_bars_are_copied_and_the_paired_deltas_are_paired_on_the_same_windows(tmp_path):
    rc, md, r = _main(_world(tmp_path))
    assert rc == 0
    assert "| M2b |" in md and "**PASS** |" in md and "| M1 |" in md and "**FAIL** |" in md   # the verdict's words, not re-decided
    assert "WM-1: **FAIL**" in md and "K-1 overall: **PASS**" in md
    assert r["control"]["m1_minus_m1_ok"] is True and r["control"]["n_components"] == 7
    z = {t: np.load(tmp_path / "eval" / f"eval_{t}.npz") for t in ("M1_s0", "M2b_s0")}
    gt = np.load(tmp_path / "gt.npz")["gt"]
    blk = r["paired"]["M2b_s0 - M1_s0"]["lateral"]["cross_mae_m"]
    cross = lambda p: np.abs(p[..., 1] - gt[..., 1])                    # noqa: E731
    assert blk["n_episodes"] == N_EP and blk["delta"] < 0 and blk["separated"] is True       # M2b's noise is smaller
    assert np.isfinite(blk["delta_exact"]) and abs(blk["delta"] - blk["delta_exact"]) < 1e-4    # taniteval rounds the interval to 4 dp
    assert set(r["paired"]) >= {"M2b_s0 - M1_s0", "M2b_s1 - M1_s1", "M1_s0 - P10", "M2b_s1 - P10", "M1-navshuf_s0 - P10"}
    assert cross(z["M2b_s0"]["pred"]).mean() < cross(z["M1_s0"]["pred"]).mean()
    assert "| LATERAL | cross-track MAE (m) | 0.500 [0.400, 0.600] |" in md                      # P10's column from floors_report


@pytest.mark.parametrize("kw, msg", [
    (dict(bad_eid="M2b_s1"), "is not on the rebuilt eval grid"),
    (dict(bad_ade="M1_s0"), "the ground truth is not its grid"),
    (dict(wm_off=0.001), "verdict WM-1 seed 0"),
    (dict(k1_off=0.001), "verdict K-1 M1 seed 0"),
    (dict(floors_off=0.01), "do not reproduce floors_perwindow.npz"),
    (dict(drop="M2b_s1"), "eval_M2b_s1.npz is missing"),
])
def test_every_guard_refuses_by_name(tmp_path, capsys, kw, msg):
    with pytest.raises(SystemExit) as e:
        _main(_world(tmp_path, **kw))
    assert e.value.code == 2 and msg in capsys.readouterr().err


def test_a_missing_control_run_is_left_out_not_refused(tmp_path):
    rc, md, r = _main(_world(tmp_path, drop="M1-navshuf_s1", floors=False))
    assert rc == 0 and "M1-navshuf_s1" not in r["paired"] and "M1-navshuf_s0 - P10" in r["paired"]
    assert "floors_perwindow.npz absent" in md


def test_the_default_ground_truth_is_the_eval141_grid():
    raw = PKG / "raw"
    if not (raw / "eval141_kin.npz").exists():
        pytest.skip("eval inputs not in this checkout")
    m = _load()

    class A:
        gt_npz = None
        eval_grid, eval_kin, eval_join = str(raw / "eval_grid_windows.json"), str(raw / "eval141_kin.npz"), str(raw / "join_eval141.npz")
    gt, eid, dt, prov = m.ground_truth(A)
    assert gt.shape == (2399, 30, 2) and len(np.unique(eid)) == 141 and dt == pytest.approx(0.2)
    assert set(prov["files"]) == {"eval_grid_windows.json", "eval141_kin.npz", "join_eval141.npz"}
    p10 = np.load(m.P10_PRED)                                          # P10-yaw-4512: its eval-141 ADE@6s is the registered 2.4870
    assert np.linalg.norm(p10 - gt, axis=-1).mean() == pytest.approx(2.4870, abs=5e-5)
