"""Pins for ``taniteval.lateral`` — the M1 lateral/longitudinal decomposition.

The finding this module implements is MEASURED and structural: ADE is **98.6 %
longitudinal by squared-error energy**, so the lateral axis — the one that
causes lane departures — receives ~1.4 % of the reported signal; and lateral
error **compounds 4.4-5.9x faster** than longitudinal (replicated on 2 clips).
See ``LATERAL_VS_LONGITUDINAL_ANALYSIS.md``.

These tests are written so that the *properties the finding depends on* cannot
regress silently:

* the split is **exact** (``along² + cross² == ||r||²``) — otherwise the energy
  share is meaningless;
* the **axis convention is checked, not assumed** — a transposed dump would
  invert every number in the module and turn the least-served axis into the
  headline;
* a **pure speed error produces zero cross-track** and a **pure lateral offset
  produces zero along-track** — the two failure modes the aggregate ADE conflates;
* **planted** energy shares and growth ratios are recovered to the decimal, so
  "98.6 %" and "x4.4" are computed the way the analysis computed them;
* the **tail** statistics are the gate surface, and the mean-vs-p90 gap that
  motivated them is reproducible on synthetic data with a known heavy tail.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from taniteval import driving as drv
from taniteval import lateral as L


# --------------------------------------------------------------------------- #
# fixtures                                                                      #
# --------------------------------------------------------------------------- #
def _straight_gt(n=40, K=20, v=10.0, dt=L.DT):
    """GT: straight ahead at ``v`` m/s, on the ego x axis."""
    x = torch.arange(1, K + 1, dtype=torch.float32) * v * dt
    return torch.stack([torch.stack([x, torch.zeros(K)], -1)] * n)


def _win(n=60, K=20, lat_sigma=0.4, lon_scale=1.05, v=10.0, seed=0):
    rng = np.random.default_rng(seed)
    gt = _straight_gt(n, K, v)
    pred = gt.clone()
    pred[..., 0] *= lon_scale
    pred[..., 1] += torch.as_tensor(
        rng.normal(0, lat_sigma, (n, 1))).float() * torch.linspace(0, 1, K)
    return {"eid": [str(i // 3) for i in range(n)],
            "pred_dense": pred, "gt_dense": gt,
            "speed": torch.full((n,), float(v)),
            "head_deg": torch.as_tensor(rng.normal(0, 8, n)).float(),
            "dense_steps": list(range(1, K + 1)), "dt_s": L.DT}


# ========================================================================== #
# 1. The split is EXACT — everything else depends on it                        #
# ========================================================================== #
@pytest.mark.parametrize("mode", ["ego", "frenet"])
def test_decomposition_is_orthonormal(mode):
    rng = np.random.default_rng(1)
    gt = _straight_gt(30, 25)
    pred = gt + torch.as_tensor(rng.normal(0, 1.5, (30, 25, 2))).float()
    al, cr = L.decompose(pred, gt, mode)
    de2 = torch.linalg.norm(pred - gt, dim=-1) ** 2
    assert torch.allclose(al ** 2 + cr ** 2, de2, atol=1e-3)


def test_frenet_and_ego_agree_on_a_straight_path():
    """They coincide only where the GT is straight and x-aligned — pin that."""
    gt = _straight_gt(10, 20)
    pred = gt.clone()
    pred[..., 1] += 0.8
    pred[..., 0] += 2.0
    ea, ec = L.decompose(pred, gt, "ego")
    fa, fc = L.decompose(pred, gt, "frenet")
    assert torch.allclose(ea, fa, atol=1e-3)
    assert torch.allclose(ec, fc, atol=1e-3)


def test_frenet_and_ego_DIVERGE_in_a_turn():
    """...and in a turn they must NOT agree — that is why both are emitted."""
    th = torch.linspace(0.0, 1.2, 20)
    gt = torch.stack([torch.stack([10 * torch.sin(th),
                                   10 * (1 - torch.cos(th))], -1)] * 8)
    pred = gt.clone()
    pred[..., 1] += 1.0                      # 1 m in EGO y, not path-normal
    ea, ec = L.decompose(pred, gt, "ego")
    fa, fc = L.decompose(pred, gt, "frenet")
    assert not torch.allclose(ec, fc, atol=0.05)
    assert torch.allclose(ec[:, -1], torch.full((8,), 1.0), atol=1e-3)
    assert float(fc[:, -1].abs().mean()) < 0.95, \
        "in a turn the path-normal component of a pure ego-y offset shrinks"


def test_pure_speed_error_has_zero_cross_track():
    """The conflation the aggregate ADE makes, stated as a test."""
    gt = _straight_gt(12, 30)
    pred = gt.clone()
    pred[..., 0] *= 1.4                      # 40 % too fast, perfectly on-line
    al, cr = L.decompose(pred, gt)
    assert float(cr.abs().max()) < 1e-4
    assert float(al.abs().mean()) > 5.0
    es = L.energy_share(al.numpy(), cr.numpy())
    assert es["longitudinal_share_of_squared_error"] == 1.0


def test_pure_lateral_offset_has_zero_along_track():
    gt = _straight_gt(12, 30)
    pred = gt.clone()
    pred[..., 1] += 1.5
    al, cr = L.decompose(pred, gt)
    assert float(al.abs().max()) < 1e-4
    assert np.allclose(cr.numpy(), 1.5, atol=1e-4)
    es = L.energy_share(al.numpy(), cr.numpy())
    assert es["lateral_share_of_squared_error"] == 1.0


# ========================================================================== #
# 2. The axis convention is VERIFIED, not assumed                              #
# ========================================================================== #
def test_axis_convention_passes_on_a_well_formed_dump():
    ev = L.assert_axis_convention(_straight_gt(20, 20, v=10.15), speed=[10.15] * 20)
    assert ev["verified"] is True
    # the source analysis' own check: ~22.1 m of along-track at 2 s @10.15 m/s
    assert ev["mean_abs_along_final_m"] == pytest.approx(20.3, abs=0.1)
    assert ev["mean_abs_cross_final_m"] == 0.0


def test_axis_convention_rejects_a_transposed_dump():
    gt = _straight_gt(20, 20)
    flipped = gt.flip(-1)                     # (y, x) instead of (x, y)
    with pytest.raises(ValueError, match="axis convention violated"):
        L.assert_axis_convention(flipped)


def test_axis_convention_rejects_a_speed_inconsistent_dump():
    with pytest.raises(ValueError, match="axis convention suspect"):
        L.assert_axis_convention(_straight_gt(20, 20, v=10.0), speed=[30.0] * 20)


def test_block_runs_the_axis_check():
    out = L.block(_win(), n_boot=100)
    assert out["axis_check"]["verified"] is True
    assert L.ALONG_AXIS == 0 and L.CROSS_AXIS == 1


# ========================================================================== #
# 3. Energy share — the 98.6 % finding, recomputed the way it was computed      #
# ========================================================================== #
@pytest.mark.parametrize("share", [0.986, 0.846, 0.5])
def test_energy_share_recovers_a_planted_value(share):
    """Plant a known longitudinal share and require it back to 3 decimals."""
    n, K = 200, 20
    lon = np.full((n, K), np.sqrt(share))
    lat = np.full((n, K), np.sqrt(1.0 - share))
    got = L.energy_share(lon, lat)
    assert got["longitudinal_share_of_squared_error"] == pytest.approx(
        share, abs=1e-3)
    assert (got["longitudinal_share_of_squared_error"]
            + got["lateral_share_of_squared_error"]) == pytest.approx(1.0, abs=1e-3)


def test_energy_share_by_step_has_one_entry_per_step():
    win = _win(K=20)
    al, cr = L.decompose(win["pred_dense"], win["gt_dense"])
    es = L.energy_share(al.numpy(), cr.numpy())
    assert len(es["longitudinal_share_by_step"]) == 20
    assert all(0.0 <= x <= 1.0 for x in es["longitudinal_share_by_step"])


def test_lateral_share_falls_as_longitudinal_error_grows():
    """Sanity on the direction of the finding."""
    a = L.block(_win(lon_scale=1.02, lat_sigma=0.4, seed=2), n_boot=50)
    b = L.block(_win(lon_scale=1.30, lat_sigma=0.4, seed=2), n_boot=50)
    assert (b["energy_share"]["longitudinal_share_of_squared_error"]
            > a["energy_share"]["longitudinal_share_of_squared_error"])


# ========================================================================== #
# 4. The compounding law                                                       #
# ========================================================================== #
def test_growth_recovers_a_planted_compounding_ratio():
    """lateral x4 vs longitudinal x2 over the window -> cross faster by x2."""
    n, K = 50, 20
    step = np.arange(1, K + 1, dtype=float)
    ref = 5                                   # the 0.5 s reference step
    lon = np.tile(1.0 + (step - 1) / (K - 1) * 1.0, (n, 1))   # linear-ish
    lat = np.tile(step / step[ref - 1], (n, 1))
    g = L.growth(lon, lat, ref_step=ref)
    assert g["ref_step"] == ref and g["ref_s"] == 0.5
    assert g["cross_growth_final"] == pytest.approx(K / ref, abs=1e-3)
    assert g["cross_grows_faster_by"] == pytest.approx(
        g["cross_growth_final"] / g["along_growth_final"], abs=1e-3)


def test_growth_defaults_to_the_half_second_reference():
    win = _win(K=20)
    al, cr = L.decompose(win["pred_dense"], win["gt_dense"])
    g = L.growth(al.numpy(), cr.numpy())
    assert g["ref_step"] == 5 and g["ref_s"] == 0.5
    assert len(g["cross_growth_by_step"]) == 20
    assert g["cross_growth_by_step"][4] == pytest.approx(1.0, abs=1e-6)


def test_growth_detects_the_compounding_asymmetry_on_a_realistic_window():
    """A bounded scale error + a compounding lateral drift -> cross faster."""
    n, K = 80, 20
    gt = _straight_gt(n, K)
    pred = gt.clone()
    pred[..., 0] *= 1.18                                   # bounded, ~18 %
    pred[..., 1] += torch.linspace(0, 1, K) ** 2 * 1.2     # compounding
    al, cr = L.decompose(pred, gt)
    g = L.growth(al.numpy(), cr.numpy())
    assert g["cross_grows_faster_by"] > 3.0


def test_growth_rejects_an_out_of_range_reference():
    with pytest.raises(ValueError, match="outside 1"):
        L.growth(np.ones((4, 10)), np.ones((4, 10)), ref_step=11)


# ========================================================================== #
# 5. TAIL statistics — the gate surface (§M2)                                   #
# ========================================================================== #
def test_tail_stats_reproduce_the_mean_vs_p90_gap():
    """Mostly-small errors with a heavy tail: mean stays low while p90 does not.

    This is the concealment mechanism the analysis MEASURED (mean 0.25 m,
    p90 1.40 m on the same windows)."""
    v = np.concatenate([np.full(850, 0.10), np.full(150, 1.60)])
    t = L.tail_stats(v)
    assert t["mean"] < 0.35
    assert t["p90"] > 1.5
    assert t["mean_to_p90_ratio"] > 4
    assert t["frac_beyond_m"]["1"] == pytest.approx(0.15, abs=1e-3)
    assert t["frac_beyond_m"]["1.75"] == 0.0


def test_tail_stats_emit_every_required_quantile():
    t = L.tail_stats(np.linspace(0, 10, 1001))
    for k in ("p50", "p75", "p90", "p95", "p99", "max", "mean", "n"):
        assert k in t, k
    assert t["p50"] == pytest.approx(5.0, abs=0.02)
    assert t["max"] == 10.0
    assert set(t["frac_beyond_m"]) == {"0.5", "1", "1.75"}


def test_tail_stats_on_empty_input_does_not_invent_numbers():
    assert L.tail_stats(np.array([])) == {"n": 0}


def test_block_gates_on_the_tail_not_the_mean():
    out = L.block(_win(lat_sigma=1.2, seed=4), n_boot=100)
    agg = out["dense_aggregate"]
    assert agg["cross_peak_p90"]["mean"] >= agg["cross_peak"]["mean"]
    assert "frac_beyond_m" in agg["cross_peak_tail"]
    assert "NOT the mean" in agg["cross_peak_tail"]["_read"]


# ========================================================================== #
# 6. The M1 contract: no L2 without its split, everywhere, with intervals       #
# ========================================================================== #
def test_every_horizon_reports_de_along_and_cross_together():
    out = L.block(_win(), n_boot=100)
    assert out["by_horizon"], "the block must report per-horizon rows"
    for tag, row in out["by_horizon"].items():
        for k in ("de", "along_abs", "cross_abs", "cross_p90", "cross_tail",
                  "energy_share"):
            assert k in row, f"{tag} missing {k}"
        assert row["orthogonality_max_abs_residual"] < 1e-3, tag


def test_block_intervals_are_the_decision_grade_estimator():
    out = L.block(_win(), n_boot=100)
    drv.assert_no_deprecated_estimator(out, _path="lateral")
    row = next(iter(out["by_horizon"].values()))
    assert row["de"]["estimator"] == "episode_cluster_bootstrap"
    assert row["de"]["n_episodes"] == out["n_episodes"] == 20


def test_decompose_metric_is_a_drop_in_for_any_ade_row():
    win = _win()
    m = L.decompose_metric(win["pred_dense"], win["gt_dense"], win["eid"],
                           n_boot=100)
    assert set(("de", "along_abs", "cross_abs", "cross_p90", "cross_tail",
                "energy_share_at_step")) <= set(m)
    # the split must add up to the L2 at that step, in the point estimates
    al, cr = L.decompose(win["pred_dense"], win["gt_dense"])
    de = torch.linalg.norm(win["pred_dense"] - win["gt_dense"], dim=-1)
    assert m["de"]["mean"] == pytest.approx(float(de[:, -1].mean()), abs=1e-3)
    assert m["along_abs"]["mean"] == pytest.approx(
        float(al[:, -1].abs().mean()), abs=1e-3)
    assert m["cross_abs"]["mean"] == pytest.approx(
        float(cr[:, -1].abs().mean()), abs=1e-3)


def test_per_window_covers_every_dense_step_by_default():
    win = _win(K=20)
    comp = L.per_window(win["pred_dense"], win["gt_dense"])
    assert "de@2s" in comp and "cross_abs@0.1s" in comp
    assert comp["ade_dense"].shape == (60,)
    assert np.all(comp["cross_peak"] >= comp["cross_abs_dense"])


def test_dense_is_required_and_the_refusal_is_explicit():
    win = _win()
    del win["pred_dense"]
    out = L.block(win, n_boot=50)
    assert out["dense_surface_available"] is False
    assert "compounding law" in out["skipped"]


def test_mode_must_be_valid():
    with pytest.raises(ValueError, match="mode must be one of"):
        L.decompose(_straight_gt(2, 5), _straight_gt(2, 5), mode="polar")


def test_verdict_states_both_findings():
    v = L.block(_win(), n_boot=100)["verdict"]
    assert "squared error" in v and "grows" in v and "beyond" in v


# ========================================================================== #
# 7. Paired cross-track — the channel HP-2 / HP-3 are measured in               #
# ========================================================================== #
def test_paired_cross_track_is_paired_and_oriented():
    win = _win(seed=5)
    gt = win["gt_dense"]
    good = gt.clone()
    good[..., 1] += 0.1
    bad = gt.clone()
    bad[..., 1] += 2.0
    d = L.paired_cross_track(good, bad, gt, win["eid"], n_boot=300)
    assert d["estimator"] == "paired_episode_cluster_bootstrap"
    assert d["delta"] == pytest.approx(1.9, abs=1e-2)
    assert d["separated"] is True
    assert "`a` wins" in d["_orientation"]


def test_paired_cross_track_can_test_the_tail():
    win = _win(seed=6)
    gt = win["gt_dense"]
    a = gt.clone()
    a[..., 1] += 0.2
    b = gt.clone()
    b[..., 1] += 0.2
    # 12 of 60 windows = the worst 20 %, so the tail sits ABOVE p90 while the
    # mean dilutes it by 5x. This is the concealment §M2 gates against.
    b[:12, :, 1] += 4.0
    d_mean = L.paired_cross_track(a, b, gt, win["eid"], n_boot=300)
    d_p90 = L.paired_cross_track(a, b, gt, win["eid"], n_boot=300, reduce="p90")
    assert d_mean["delta"] == pytest.approx(4.0 * 12 / 60, abs=1e-2)
    assert d_p90["delta"] == pytest.approx(4.0, abs=1e-2)
    assert d_p90["delta"] > 4 * d_mean["delta"], \
        "the tail test must see a tail the mean test dilutes"


def test_paired_cross_track_is_blind_to_a_pure_speed_difference():
    """Two arms differing only in speed must tie on the LATERAL channel."""
    win = _win(seed=7)
    gt = win["gt_dense"]
    a, b = gt.clone(), gt.clone()
    a[..., 0] *= 1.2
    b[..., 0] *= 0.8
    d = L.paired_cross_track(a, b, gt, win["eid"], n_boot=200)
    assert abs(d["delta"]) < 1e-3
    assert d["separated"] is False


# ========================================================================== #
# 8. REGISTRY PINS on the committed flagship-30k dump — the strongest check     #
#                                                                              #
# The 4-waypoint path runs on real committed data with no GPU, so this module   #
# can be checked against numbers the program already publishes. If these drift, #
# either the geometry convention or the artifact changed — fail loud rather     #
# than publish a decomposition that disagrees with the leaderboard.             #
# ========================================================================== #
_DUMP = Path(__file__).resolve().parents[1] / "results" / "windows_flagship-30k.pt"
_DRIVING = (Path(__file__).resolve().parents[1] / "results"
            / "driving_flagship-30k.json")
_needs_dump = pytest.mark.skipif(
    not (_DUMP.exists() and _DRIVING.exists()),
    reason="committed flagship-30k window dump / driving block absent")


@pytest.fixture(scope="module")
def flagship_windows():
    return torch.load(_DUMP, map_location="cpu", weights_only=False)


@_needs_dump
def test_frenet_dense_is_byte_identical_to_driving_frenet(flagship_windows):
    """One geometry, two modules. A divergence here is a silent metric fork."""
    w = flagship_windows
    a1, c1 = drv.frenet(w["pred"].float(), w["gt"].float())
    a2, c2 = L.decompose(w["pred"].float(), w["gt"].float(), "frenet")
    assert float((a1 - a2).abs().max()) == 0.0
    assert float((c1 - c2).abs().max()) == 0.0


@_needs_dump
def test_sparse_block_reproduces_the_registry_ade_and_driving_split(
        flagship_windows):
    """MODEL_REGISTRY §1.2 / driving.tier0 pins, reproduced through this module.

    ``ade_dense`` over the 4 knots IS ADE@2s = 0.4271 [0.3675, 0.4871], and the
    frenet split reproduces ``long_abs_2s_m`` / ``lat_abs_2s_m`` to the emitted
    decimal *including the CI bounds* — same estimator, same seed, same
    episodes."""
    ref = json.loads(_DRIVING.read_text(encoding="utf-8"))["headline"]
    out = L.from_sparse_windows(flagship_windows, mode="frenet", n_boot=2000)
    ade = out["dense_aggregate"]["ade_dense"]
    assert (ade["mean"], ade["lo"], ade["hi"]) == (0.4271, 0.3675, 0.4871)
    assert (ade["mean"], ade["lo"], ade["hi"]) == (
        ref["ade_0_2s"]["mean"], ref["ade_0_2s"]["lo"], ref["ade_0_2s"]["hi"])
    r2 = out["by_horizon"]["2s"]
    assert (r2["de"]["mean"], r2["de"]["lo"], r2["de"]["hi"]) == (
        ref["fde_2s"]["mean"], ref["fde_2s"]["lo"], ref["fde_2s"]["hi"])
    for mine, theirs in (("along_abs", "long_abs_2s_m"),
                         ("cross_abs", "lat_abs_2s_m")):
        assert (r2[mine]["mean"], r2[mine]["lo"], r2[mine]["hi"]) == (
            ref[theirs]["mean"], ref[theirs]["lo"], ref[theirs]["hi"]), mine


@_needs_dump
def test_axis_convention_verified_on_the_real_val_set(flagship_windows):
    """The convention check, on 881 real windows / 40 episodes — not a fixture."""
    ev = L.assert_axis_convention(flagship_windows["gt"].float(),
                                  flagship_windows["speed"], dt=0.5)
    assert ev["verified"] is True
    assert ev["mean_abs_along_final_m"] > 20 * ev["mean_abs_cross_final_m"]
    assert ev["rel_err"] < 0.01


@_needs_dump
def test_measured_tail_gap_on_the_real_val_set(flagship_windows):
    """The concealment, MEASURED on the deployed arm rather than argued.

    Mean |XTE|@2s is well under half a metre while the tail is metres — which is
    the whole case for gating on p90/p95/max (§M2)."""
    out = L.from_sparse_windows(flagship_windows, n_boot=500)
    t = out["by_horizon"]["2s"]["cross_tail"]
    assert t["mean"] < 0.35
    assert t["p90"] > 2 * t["mean"]
    assert t["max"] > 2.0
    assert t["mean_to_p90_ratio"] > 2.0


@_needs_dump
def test_lateral_compounds_faster_than_longitudinal_on_the_real_val_set(
        flagship_windows):
    """The replicated compounding law, on 881 windows instead of 2 clips."""
    out = L.from_sparse_windows(flagship_windows, n_boot=200)
    g = out["growth"]
    assert g["ref_step"] == 1, "knot 1 is 0.5 s on the sparse surface"
    assert g["cross_grows_faster_by"] > 1.0
    assert g["cross_growth_final"] > g["along_growth_final"]
