"""Pins for ``taniteval.corridor`` against E1a's COMMITTED result JSONs.

WHY THESE TESTS ARE SHAPED THIS WAY
-----------------------------------
``corridor_departure_rate`` is slated to become the **gate co-primary**,
replacing the horizon-blind ADE@2s (``01_EXECUTION_PLAN.md`` B.2 T1-1). Until
2026-07-25 it existed only in five ``incoming/`` one-off scripts and in **zero
files** under ``taniteval/taniteval/`` (HPP-0 audit §3.2). Promoting a metric
into the library is the moment its definition can silently drift from the
artifacts that made its case — so these tests do not check that the code "looks
right", they check that it **reproduces E1a's own committed numbers**.

Two independent reproduction routes, because one is not a check:

1. **Exact count reconstruction.** At K=20 the committed common-start block is
   small enough that its numbers determine the underlying per-window counts
   uniquely: ``CDR@1.75 = 0.0035`` over 43 windows x 20 steps is *exactly 3
   departed steps*, and ``window_departure = 0.0233`` is *exactly 1 window*.
   Rebuilding that matrix and running it through :mod:`taniteval.corridor` must
   return the committed values to the emitted decimal. If the definition drifts
   — a ``>=`` for a ``>``, a mean over steps instead of windows, a different
   normaliser — this test goes red.
2. **An identity that holds inside every committed block**:
   ``mean(mean_abs_xte_by_step_m) == mean_xte_m.mean``. It is checked on E1a's
   own artifact AND on our implementation, so the two agree on what "mean XTE"
   means at every horizon and stratum, not just at the one we reconstructed.

Plus the ordering invariants that any correct implementation must satisfy, again
verified against the committed artifact rather than asserted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from taniteval import corridor as C
from taniteval import driving as drv

# --------------------------------------------------------------------------- #
# The committed E1a artifacts                                                   #
# --------------------------------------------------------------------------- #
_REPO = Path(__file__).resolve().parents[2]
_E1A_DIR = (_REPO / "TanitAD Research Hub" / "Architecture & Inference"
            / "Implementation" / "incoming"
            / "2026-07-25-closedloop-horizon-and-shift")
_K185 = _E1A_DIR / "e1a_horizon_heldout44_K185.json"
_HELDOUT44 = _E1A_DIR / "e1a_horizon_heldout44.json"

# `mean_abs_xte_by_step_m` is emitted rounded to 4 dp, so the mean of K of them
# carries up to 5e-5 of accumulated rounding, and `mean_xte_m.mean` carries its
# own 5e-5. 1.2e-4 is that bound, not a fudge: any real redefinition of the
# metric moves these by orders of magnitude more (MEASURED worst residual across
# both committed artifacts: 5.5e-5, at all_windows[20].longitudinal).
_ROUND_TOL = 1.2e-4

_needs_e1a = pytest.mark.skipif(
    not _K185.exists(),
    reason=f"E1a committed artifact not present at {_K185}")


@pytest.fixture(scope="module")
def e1a():
    return json.loads(_K185.read_text(encoding="utf-8"))


def _all_blocks(doc):
    """Every stratum block in an E1a result doc, as ``(label, block)``."""
    for section in ("all_windows", "paired_common_start"):
        for K, strata in (doc.get(section) or {}).items():
            if not isinstance(strata, dict):
                continue
            for name, blk in strata.items():
                if isinstance(blk, dict) and "corridor_departure_rate" in blk:
                    yield f"{section}[{K}].{name}", blk


# ========================================================================== #
# 1. EXACT reconstruction of E1a's committed K=20 common-start numbers          #
# ========================================================================== #
def _reconstructed_k20():
    """The unique per-window |XTE| matrix behind E1a's committed K=20 block.

    Determined, not chosen, by the committed rates:
      >1.75 m : 0.0035 * 43 * 20 = 3 steps, in 0.0233 * 43 = 1 window
      >1.0 m  : 0.0093 * 43 * 20 = 8 steps, in 0.0465 * 43 = 2 windows
      >2.5 m  : none
    Window 0 carries the 3 corridor departures (2.0 m, inside the 1.75/2.5 band)
    plus 3 more in the 1.0/1.75 band; window 1 carries the remaining 2. Every
    other step is well inside the corridor."""
    lat = np.full((43, 20), 0.10)
    lat[0, :3] = 2.00        # > 1.75, < 2.5
    lat[0, 3:6] = 1.50       # > 1.0,  < 1.75
    lat[1, :2] = 1.20        # > 1.0,  < 1.75
    eid = [str(i) for i in range(43)]
    return lat, eid


@_needs_e1a
def test_reconstructs_e1a_committed_k20_common_start(e1a):
    ref = e1a["paired_common_start"]["20"]["overall"]
    lat, eid = _reconstructed_k20()
    assert lat.shape == (ref["n_windows"], ref["horizon_K"])

    got = C.corridor_block(lat, eid, thresholds=e1a["corridor_thresholds_m"],
                           primary=e1a["corridor_primary_m"], n_boot=200,
                           surface="closed_loop")
    assert got["corridor_departure_rate"]["mean"] == \
        ref["corridor_departure_rate"]["mean"] == 0.0035
    assert got["window_departure_rate"]["mean"] == \
        ref["window_departure_rate"]["mean"] == 0.0233
    for t in ("1", "1.75", "2.5"):
        assert (got["corridor_departure_rate_by_threshold_m"][t]["mean"]
                == ref["corridor_departure_rate_by_threshold_m"][t]["mean"]), t
        assert (got["window_departure_rate_by_threshold_m"][t]["mean"]
                == ref["window_departure_rate_by_threshold_m"][t]["mean"]), t


@_needs_e1a
def test_reconstructs_e1a_committed_k20_junction_stratum(e1a):
    """The same two departing windows are junction windows — E1a says so."""
    ref = e1a["paired_common_start"]["20"]["junction"]
    lat, eid = _reconstructed_k20()
    got = C.corridor_block(lat[:ref["n_windows"]], eid[:ref["n_windows"]],
                           thresholds=e1a["corridor_thresholds_m"],
                           primary=e1a["corridor_primary_m"], n_boot=200,
                           surface="closed_loop")
    assert got["n_windows"] == 6
    assert got["corridor_departure_rate"]["mean"] == \
        ref["corridor_departure_rate"]["mean"] == 0.025
    assert got["window_departure_rate"]["mean"] == \
        ref["window_departure_rate"]["mean"] == 0.1667
    assert (got["corridor_departure_rate_by_threshold_m"]["1"]["mean"]
            == ref["corridor_departure_rate_by_threshold_m"]["1"]["mean"])


# ========================================================================== #
# 2. The mean-XTE identity, on the artifact AND on our implementation           #
# ========================================================================== #
@_needs_e1a
def test_mean_xte_identity_holds_in_every_committed_block(e1a):
    """``mean(mean_abs_xte_by_step_m) == mean_xte_m.mean`` — E1a's own artifact.

    True only if ``mean_xte`` is the mean over windows of the mean over steps,
    which is the definition we lifted. It is checked here so a future change to
    either side fails loud instead of quietly redefining the metric."""
    n = 0
    for label, blk in _all_blocks(e1a):
        by_step = blk.get("mean_abs_xte_by_step_m")
        if not by_step:
            continue
        assert float(np.mean(by_step)) == pytest.approx(
            blk["mean_xte_m"]["mean"], abs=_ROUND_TOL), label
        n += 1
    assert n >= 8, f"expected many committed blocks, checked only {n}"


def test_our_mean_xte_satisfies_the_same_identity():
    rng = np.random.default_rng(0)
    lat = np.abs(rng.normal(0.4, 0.6, (60, 37)))
    blk = C.corridor_block(lat, [str(i // 3) for i in range(60)], n_boot=200)
    assert float(np.mean(blk["mean_abs_xte_by_step_m"])) == pytest.approx(
        blk["mean_xte_m"]["mean"], abs=_ROUND_TOL)
    assert float(np.mean(C.mean_xte(lat))) == pytest.approx(
        blk["mean_xte_m"]["mean"], abs=_ROUND_TOL)


# ========================================================================== #
# 3. Ordering invariants — verified on the artifact, enforced on our code       #
# ========================================================================== #
@_needs_e1a
def test_committed_blocks_satisfy_the_ordering_invariants(e1a):
    for label, blk in _all_blocks(e1a):
        cdr = blk["corridor_departure_rate_by_threshold_m"]
        wdr = blk["window_departure_rate_by_threshold_m"]
        ts = sorted(cdr, key=float)
        for a, b in zip(ts, ts[1:]):
            assert cdr[a]["mean"] >= cdr[b]["mean"] - 1e-9, f"{label} {a}->{b}"
            assert wdr[a]["mean"] >= wdr[b]["mean"] - 1e-9, f"{label} {a}->{b}"
        for t in ts:
            assert wdr[t]["mean"] >= cdr[t]["mean"] - 1e-9, f"{label} @{t}"
        assert blk["peak_xte_m"]["mean"] >= blk["mean_xte_m"]["mean"] - 1e-9, label


def test_our_block_satisfies_the_ordering_invariants():
    rng = np.random.default_rng(1)
    lat = np.abs(rng.normal(1.0, 1.2, (80, 40)))
    blk = C.corridor_block(lat, [str(i // 4) for i in range(80)], n_boot=200)
    cdr = blk["corridor_departure_rate_by_threshold_m"]
    wdr = blk["window_departure_rate_by_threshold_m"]
    ts = sorted(cdr, key=float)
    for a, b in zip(ts, ts[1:]):
        assert cdr[a]["mean"] >= cdr[b]["mean"]
        assert wdr[a]["mean"] >= wdr[b]["mean"]
    for t in ts:
        assert wdr[t]["mean"] >= cdr[t]["mean"]
    assert blk["peak_xte_m"]["mean"] >= blk["mean_xte_m"]["mean"]


def test_window_departure_dominates_corridor_departure_per_window():
    rng = np.random.default_rng(2)
    lat = np.abs(rng.normal(1.5, 1.0, (50, 25)))
    assert np.all(C.window_departure(lat) >= C.corridor_departure(lat))
    # and the fraction is 0 exactly when the window never leaves
    never = C.window_departure(lat) == 0
    assert np.all(C.corridor_departure(lat)[never] == 0)


# ========================================================================== #
# 4. The headline E1a numbers this module's docstring cites                    #
# ========================================================================== #
@_needs_e1a
def test_docstring_headline_numbers_match_the_artifact(e1a):
    """Guards the doc-drift class: a cited number whose artifact moved."""
    pcs = e1a["paired_common_start"]
    assert pcs["20"]["overall"]["corridor_departure_rate"]["mean"] == 0.0035
    assert pcs["185"]["overall"]["corridor_departure_rate"]["mean"] == 0.5877
    assert pcs["185"]["junction"]["corridor_departure_rate"]["mean"] == 0.8414
    assert pcs["185"]["overall"]["peak_xte_m"]["mean"] == 38.9445
    assert pcs["20"]["overall"]["peak_xte_m"]["mean"] == 0.3489
    assert e1a["corridor_primary_m"] == C.CORRIDOR_HALFWIDTH_M
    assert tuple(e1a["corridor_thresholds_m"]) == C.CORRIDOR_GRID_M
    assert e1a["junction_deg"] == C.JUNCTION_DEG
    # the 168x horizon effect, recomputed rather than quoted
    ratio = (pcs["185"]["overall"]["corridor_departure_rate"]["mean"]
             / pcs["20"]["overall"]["corridor_departure_rate"]["mean"])
    assert ratio > 100, ratio


@_needs_e1a
def test_e1a_estimator_is_the_decision_grade_one(e1a):
    assert "episode_cluster_bootstrap" in e1a["_estimator"]
    assert C.DEPRECATED_ESTIMATOR in e1a["_estimator"]  # named as refused
    for _label, blk in _all_blocks(e1a):
        assert blk["corridor_departure_rate"]["estimator"] == \
            "episode_cluster_bootstrap"


@pytest.mark.skipif(not _HELDOUT44.exists(), reason="heldout44 artifact absent")
def test_second_committed_artifact_also_reproduces_the_identity():
    """CLAUDE.md rule: absence — or presence — at ONE location is not enough."""
    doc = json.loads(_HELDOUT44.read_text(encoding="utf-8"))
    n = 0
    for label, blk in _all_blocks(doc):
        by_step = blk.get("mean_abs_xte_by_step_m")
        if not by_step:
            continue
        assert float(np.mean(by_step)) == pytest.approx(
            blk["mean_xte_m"]["mean"], abs=_ROUND_TOL), label
        n += 1
    assert n >= 8


# ========================================================================== #
# 5. Arbitrary horizon K + the junction stratum                                 #
# ========================================================================== #
@pytest.mark.parametrize("K", [20, 60, 120, 185])
def test_arbitrary_horizon_K(K):
    """HP-1 sweeps K in {20, 60, 120, 185}; the block must take all of them."""
    rng = np.random.default_rng(K)
    lat = np.abs(rng.normal(0.5, 0.5, (40, K)))
    blk = C.corridor_block(lat, [str(i) for i in range(40)], n_boot=100)
    assert blk["horizon_K"] == K
    assert blk["horizon_s"] == round(K * 0.1, 2)
    assert len(blk["mean_abs_xte_by_step_m"]) == K


def test_junction_mask_is_the_e1a_definition():
    hd = np.array([0.0, 9.99, 10.0, 10.01, -12.0, -9.0, 45.0])
    got = C.junction_mask(hd)
    assert got.tolist() == [False, False, True, True, True, False, True]


def test_strata_partition_exactly_once():
    rng = np.random.default_rng(5)
    hd = rng.normal(0, 15, 200)
    spd = rng.uniform(0, 20, 200)
    st = C.strata(hd, spd)
    cover = st["junction"].astype(int) + st["longitudinal"].astype(int) \
        + st["other"].astype(int)
    assert (cover == 1).all(), "the three strata must partition the windows"
    assert st["overall"].all()


def test_stratified_emits_every_stratum_and_passes_the_guard():
    rng = np.random.default_rng(6)
    n, K = 120, 30
    lat = np.abs(rng.normal(0.6, 0.8, (n, K)))
    eid = [str(i // 3) for i in range(n)]
    out = C.stratified(lat, eid, rng.normal(0, 20, n), rng.uniform(1, 20, n),
                       n_boot=100)
    for name in ("overall", "junction", "longitudinal", "other"):
        assert name in out
    assert out["overall"]["n_windows"] == n
    drv.assert_no_deprecated_estimator(out, _path="corridor.stratified")


def test_tiny_stratum_returns_none_not_a_nan_interval():
    lat = np.abs(np.random.default_rng(7).normal(0.5, 0.3, (1, 20)))
    assert C.corridor_block(lat, ["0"], n_boot=50) is None
    lat2 = np.abs(np.random.default_rng(7).normal(0.5, 0.3, (4, 20)))
    assert C.corridor_block(lat2, ["0"] * 4, n_boot=50) is None, \
        "a single episode cannot support an episode-cluster bootstrap"


# ========================================================================== #
# 6. Cross-track from the persisted dense path                                  #
# ========================================================================== #
def test_cross_track_of_an_identical_path_is_zero():
    gt = torch.stack([torch.stack([torch.arange(1, 21).float(),
                                   torch.zeros(20)], -1)] * 5)
    assert np.allclose(C.cross_track_from_paths(gt.clone(), gt), 0.0, atol=1e-5)


@pytest.mark.parametrize("offset", [0.5, 1.75, -2.5])
def test_cross_track_recovers_a_constant_lateral_offset(offset):
    """A straight GT with the prediction shifted sideways by d -> |XTE| == |d|."""
    x = torch.arange(1, 41).float()
    gt = torch.stack([torch.stack([x, torch.zeros(40)], -1)] * 4)
    pred = gt.clone()
    pred[..., 1] += offset
    lat = C.cross_track_from_paths(pred, gt)
    assert lat.shape == (4, 40)
    assert np.allclose(lat, abs(offset), atol=1e-3)


def test_cross_track_ignores_a_pure_longitudinal_error():
    """The point of decomposing: a speed error must not appear as a departure.

    A prediction that runs ahead along a straight GT has a large ADE and ZERO
    cross-track error. This is the 98.6 %-longitudinal problem
    (LATERAL_VS_LONGITUDINAL_ANALYSIS.md §1.1) stated as a test."""
    x = torch.arange(1, 31).float()
    gt = torch.stack([torch.stack([x, torch.zeros(30)], -1)] * 3)
    pred = gt.clone()
    pred[..., 0] *= 1.4                       # 40 % too fast, perfectly on-line
    lat = C.cross_track_from_paths(pred, gt)
    assert lat.max() < 1e-3
    ade = float(torch.linalg.norm(pred - gt, dim=-1).mean())
    assert ade > 5.0, "the same windows carry a large ADE"


def test_cross_track_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match=r"\[N,K,2\]"):
        C.cross_track_from_paths(torch.zeros(3, 10, 2), torch.zeros(3, 9, 2))


# ========================================================================== #
# 7. Wiring, refusals and honest limits                                         #
# ========================================================================== #
def _win(n=60, K=20, dense=True):
    rng = np.random.default_rng(11)
    x = torch.arange(1, K + 1).float()
    gt = torch.stack([torch.stack([x * (0.5 + i % 3), torch.zeros(K)], -1)
                      for i in range(n)])
    pred = gt.clone()
    pred[..., 1] += torch.as_tensor(rng.normal(0, 0.8, (n, 1))).float()
    win = {"eid": [i // 3 for i in range(n)],
           "speed": torch.as_tensor(rng.uniform(1, 20, n)).float(),
           "head_deg": torch.as_tensor(rng.normal(0, 20, n)).float(),
           "wp_steps": [5, 10, 15, 20], "dt_s": 0.1}
    if dense:
        win.update({"pred_dense": pred, "gt_dense": gt,
                    "dense_steps": list(range(1, K + 1))})
    return win


def test_from_windows_needs_the_dense_path_and_says_so():
    out = C.from_windows(_win(dense=False), n_boot=50)
    assert out["dense_surface_available"] is False
    assert "pred_dense" in out["skipped"]


def test_from_windows_emits_a_guarded_stratified_block():
    out = C.from_windows(_win(), n_boot=100)
    assert out["dense_surface_available"] is True
    assert out["overall"]["surface"] == "open_loop_dense"
    assert "must never be pooled" in out["_surface_warning"]
    drv.assert_no_deprecated_estimator(out, _path="corridor.from_windows")


def test_surface_must_be_declared_and_valid():
    lat = np.abs(np.random.default_rng(3).normal(0.5, 0.3, (20, 20)))
    with pytest.raises(ValueError, match="surface must be one of"):
        C.corridor_block(lat, [str(i) for i in range(20)], surface="whatever",
                         n_boot=50)


def test_primary_must_be_inside_the_emitted_grid():
    """A verdict that survives at only one half-width is a knife-edge."""
    lat = np.abs(np.random.default_rng(4).normal(0.5, 0.3, (20, 20)))
    with pytest.raises(ValueError, match="knife-edge"):
        C.corridor_block(lat, [str(i) for i in range(20)], primary=1.6,
                         n_boot=50)


def test_horizon_ceiling_matches_the_e1a_note():
    """PhysicalAI clips are 190-199 frames -> K=200 is structurally impossible."""
    assert C.horizon_ceiling(199) == 190
    assert C.horizon_ceiling(190) == 181
    assert C.horizon_seconds(190) == 19.0
    assert C.horizon_ceiling(199) < 200


def test_paired_delta_is_paired_and_oriented():
    rng = np.random.default_rng(9)
    n, K = 60, 40
    lat_a = np.abs(rng.normal(0.4, 0.3, (n, K)))          # the better arm
    lat_b = lat_a + 2.0                                   # departs far more
    eid = [str(i // 3) for i in range(n)]
    d = C.paired_stratum_delta(lat_a, lat_b, eid, n_boot=300)
    assert d["estimator"] == "paired_episode_cluster_bootstrap"
    assert d["delta"] > 0 and d["separated"] is True
    assert "POSITIVE = arm `a` departs the corridor less" in d["_orientation"]


def test_extrapolation_flags_are_emitted():
    lat = np.full((20, 20), 0.2)
    lat[0, :5] = 5.0                        # outside the measured P1 envelope
    yaw = np.full((20, 20), 1.0)
    yaw[1, :2] = 30.0
    blk = C.corridor_block(lat, [str(i) for i in range(20)],
                           yaw_abs_deg=yaw, n_boot=100, surface="closed_loop")
    assert blk["EXTRAPOLATION_frac_steps_lat_over_3m"] == round(5 / 400, 5)
    assert blk["EXTRAPOLATION_frac_steps_yaw_over_12deg"] == round(2 / 400, 5)
    assert blk["EXTRAPOLATION_frac_windows_any_step_out_of_envelope"] == 0.1
