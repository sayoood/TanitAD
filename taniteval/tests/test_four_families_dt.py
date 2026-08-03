"""Regression tests for the 2026-08-03 dt defect in ``taniteval.four_families``.

⛔ THE DEFECT. ``all_families`` read ``win["pred"]``, which for ``rollout.collect`` and
``refc_eval.collect`` is the SPARSE 4-waypoint view at ``WP_STEPS = (5, 10, 15, 20)`` — a **0.5 s**
grid — while ``_seq_geometry`` divided every derivative by the hard-coded ``DT_S = 0.1``. Result:
every published ``speed_*`` was **5x**, every ``accel_*`` **25x**, every ``yaw_rate_*`` **5x**.
Positions, heading and curvature were dt-invariant and therefore correct.

It went unseen because nothing ever compared a derived rate against a physical quantity the
episode already carried. MEASURED on Thor over 859 real held-out windows: the ego's own recorded
speed ``poses[:, 3]`` averaged **12.4565 m/s** while ``_seq_geometry(gt)["speed"]`` returned
**62.9789 m/s** — ratio **5.0559**.

These tests pin the three things that must never regress:
  1. a constant-velocity path must report ITS OWN speed, on any grid;
  2. the grid must be DERIVED from the window contract, not assumed;
  3. dt-invariant metrics must stay invariant, so a future dt change cannot silently move them.
"""
import math

import pytest
import torch

from taniteval import four_families as FF


def _straight(n, horizon, v_mps, dt):
    """A perfectly straight constant-velocity path sampled on a dt-second grid."""
    k = torch.arange(1, horizon + 1, dtype=torch.float32)
    x = (v_mps * dt) * k
    wp = torch.zeros(n, horizon, 2)
    wp[:, :, 0] = x
    return wp


# --------------------------------------------------------------- 1. physical ground truth
@pytest.mark.parametrize("v,dt,horizon", [(12.0, 0.5, 4), (12.0, 0.1, 20), (3.0, 0.5, 4)])
def test_constant_velocity_path_reports_its_own_speed(v, dt, horizon):
    """THE test the defect would have failed: a 12 m/s path must read 12 m/s, not 60."""
    wp = _straight(8, horizon, v, dt)
    g = FF._seq_geometry(wp, dt)
    assert g["speed"].mean().item() == pytest.approx(v, rel=1e-5)
    # and a constant-velocity path has zero acceleration on the correct grid
    assert g["accel"].abs().max().item() == pytest.approx(0.0, abs=1e-3)


def test_wrong_dt_inflates_by_the_documented_powers():
    """Pins the CORRECTION FACTORS, so every pre-fix number in the hub can be repaired."""
    wp = _straight(4, 4, 12.0, 0.5)                     # a real 0.5 s grid
    right = FF._seq_geometry(wp, 0.5)
    wrong = FF._seq_geometry(wp, 0.1)                   # the historical bug
    assert wrong["speed"].mean() / right["speed"].mean() == pytest.approx(5.0, rel=1e-6)
    # accel is 0 for constant velocity, so use a curved/accelerating path for the 1/dt^2 check
    k = torch.arange(1, 5, dtype=torch.float32)
    acc = torch.zeros(4, 4, 2)
    acc[:, :, 0] = 0.5 * 2.0 * (k * 0.5) ** 2           # 2 m/s^2 from rest, on a 0.5 s grid
    r2, w2 = FF._seq_geometry(acc, 0.5), FF._seq_geometry(acc, 0.1)
    assert r2["accel"].abs().mean().item() == pytest.approx(2.0, rel=1e-5)
    assert (w2["accel"].abs().mean() / r2["accel"].abs().mean()).item() == \
        pytest.approx(25.0, rel=1e-6)


# --------------------------------------------------------------- 2. the grid is derived
def test_infer_dt_reads_the_window_contract():
    dt, prov = FF.infer_dt({"wp_steps": [5, 10, 15, 20], "dt_s": 0.1})
    assert dt == pytest.approx(0.5)
    assert "wp_steps" in prov
    dt, _ = FF.infer_dt({"wp_steps": list(range(1, 21)), "dt_s": 0.1})
    assert dt == pytest.approx(0.1)


def test_infer_dt_never_guesses_silently():
    dt, prov = FF.infer_dt({})
    assert dt == FF.DT_S and "NO wp_steps" in prov
    dt, prov = FF.infer_dt({"wp_steps": [1, 5, 20], "dt_s": 0.1})
    assert "NON-UNIFORM" in prov


def test_all_families_uses_the_sparse_grid_when_told_the_contract():
    """The end-to-end path: a 12 m/s straight line scored through all_families reads 12 m/s."""
    wp = _straight(6, 4, 12.0, 0.5)
    win = {"pred": wp, "gt": wp.clone(), "wp_steps": [5, 10, 15, 20], "dt_s": 0.1}
    fam = FF.all_families(win, prefer_dense=False)
    assert fam["_grid"]["used"] == "sparse"
    assert fam["_grid"]["dt_s"] == pytest.approx(0.5)
    assert fam["longitudinal"]["dt_s"] == pytest.approx(0.5)
    # pred == gt so every error is 0; the point is that the GRID is right, checked via geometry
    g = FF._seq_geometry(wp, fam["_grid"]["dt_s"])
    assert g["speed"].mean().item() == pytest.approx(12.0, rel=1e-5)


def test_all_families_prefers_the_dense_path_when_present():
    dense = _straight(5, 20, 9.0, 0.1)
    sparse = dense[:, [4, 9, 14, 19]]
    win = {"pred": sparse, "gt": sparse.clone(),
           "pred_dense": dense, "gt_dense": dense.clone(),
           "wp_steps": [5, 10, 15, 20], "dt_s": 0.1}
    fam = FF.all_families(win)
    assert fam["_grid"]["used"] == "dense"
    assert fam["_grid"]["dt_s"] == pytest.approx(0.1)
    assert fam["_grid"]["horizon_steps"] == 20


# --------------------------------------------------------------- 3. invariants stay invariant
def test_dt_invariant_metrics_do_not_move_with_dt():
    """heading, curvature and cross-track are geometric — a dt change must not touch them.

    This is what makes the correction factors safe to apply retroactively to published numbers:
    only speed/accel/yaw_rate need repairing.
    """
    torch.manual_seed(0)
    wp = torch.cumsum(torch.randn(7, 6, 2) * 0.4 + torch.tensor([2.0, 0.0]), dim=1)
    a = FF._seq_geometry(wp, 0.1)
    b = FF._seq_geometry(wp, 0.5)
    assert torch.allclose(a["heading"], b["heading"])
    assert torch.allclose(a["curvature"], b["curvature"])
    assert torch.allclose(a["along"], b["along"])
    assert torch.allclose(a["cross"], b["cross"])
    # ...and the rate-like ones DO move, by exactly 1/dt and 1/dt^2
    assert torch.allclose(a["speed"], b["speed"] * 5.0, rtol=1e-5)
    assert torch.allclose(a["yaw_rate"], b["yaw_rate"] * 5.0, rtol=1e-5)
    assert torch.allclose(a["accel"], b["accel"] * 25.0, rtol=1e-5)


def test_min_ds_gate_scales_with_dt():
    """A fixed 0.05 m gate on a 0.5 s grid excludes essentially nothing — a crawling window would
    silently enter the curvature statistic. The gate is a SPEED now, so it scales."""
    crawl = _straight(3, 4, 0.2, 0.5)          # 0.2 m/s -> 0.1 m per 0.5 s step
    g_fixed = FF._seq_geometry(crawl, 0.1)     # historical: min_ds 0.05 m -> everything "valid"
    g_scaled = FF._seq_geometry(crawl, 0.5)    # correct: min_ds 0.25 m -> excluded
    assert bool(g_fixed["valid"].all())
    assert not bool(g_scaled["valid"].any())
    assert g_scaled["min_ds_m"] == pytest.approx(0.25)


def test_lateral_reports_the_grid_it_used():
    """A rate that travels without its grid is exactly how the defect propagated for months."""
    wp = _straight(4, 4, 8.0, 0.5)
    lat = FF.lateral(wp, wp.clone(), dt=0.5)
    assert lat["dt_s"] == pytest.approx(0.5)
    assert lat["min_ds_m"] == pytest.approx(0.25)
    lon = FF.longitudinal(wp, wp.clone(), dt=0.5)
    assert lon["dt_s"] == pytest.approx(0.5)
    assert "1/dt" in lon["rate_scaling_note"]


def test_heading_is_degrees_and_sane():
    """Guards the unit, because a rad/deg slip looks exactly like a model regression."""
    n, h = 5, 4
    wp = torch.zeros(n, h, 2)
    ang = math.radians(30.0)
    k = torch.arange(1, h + 1, dtype=torch.float32)
    wp[:, :, 0] = 6.0 * k * math.cos(ang)
    wp[:, :, 1] = 6.0 * k * math.sin(ang)
    straight = _straight(n, h, 12.0, 0.5)
    lat = FF.lateral(wp, straight, dt=0.5)
    assert lat["heading_mae_deg"] == pytest.approx(30.0, abs=1e-3)
