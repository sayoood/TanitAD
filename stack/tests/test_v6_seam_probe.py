"""F-16 — the X2 BAND-SEAM instrument (``taniteval/taniteval/seam.py`` +
``taniteval/tools/seam_probe.py``). **Verify, never repair.**

⛔ WHAT THIS FILE PROTECTS AND PROVES.

1. **THE GEOMETRY IS DERIVED FROM v6, NEVER GUESSED.** The probed boundary is
   ``V6Config.band_slice("op").stop == band_slice("tac").start == 20``, and the
   instrument's own ``seam_boundary_of`` recomputes it from the band spec and
   REFUSES a gap or an overlap (which is itself the stitched-trajectory defect).
   An instrument probing the wrong index would report "no seam" for the best
   possible reason and the worst.
2. **THE INSTRUMENT FIRES ON AN INJECTED SEAM** — two independently-rolled bands
   concatenated, in BOTH of its failure modes (independent CONTROL sequences,
   which leave position continuous; and independent ROLLOUTS re-based, which
   jump in position). An instrument never shown to detect the defect it hunts
   is not validated.
3. **IT DOES NOT FIRE ON A GENUINE SINGLE ROLLOUT**, and the null it returns is
   ``NO_MATERIAL_SEAM`` — a POSITIVE, well-powered result (the 80 %-power MDE is
   at or below the materiality floor), not a silence.
4. **ALL FOUR VERDICTS ARE REACHABLE** (SEAM / NO_MATERIAL_SEAM / INCONCLUSIVE /
   DEGENERATE). A guard that can only ever confirm the architecture is the C13
   family and is worthless here.
5. **THE ESTIMATOR IS taniteval's, BY IDENTITY.** The intervals come from
   ``taniteval.ci.paired_episode_cluster_bootstrap`` /
   ``episode_cluster_bootstrap`` — the functions themselves, patched here to
   prove they are CALLED and not re-implemented — and the string
   ``overlapping_holdout_se`` appears NOWHERE in either source file.
6. **THE H0-BIAS CORRECTION IS PINNED.** The null reference is the MEAN of the
   within-band boundaries, not the median. The median reference is positively
   biased under H0 (``E[median of n draws] < E[draw]`` for the right-skewed
   ``|Δ^m x|``) — this file recomputes both on exchangeable data and asserts the
   median version IS biased, so nobody can "improve" the code back to it.
7. **VERIFY, NEVER REPAIR, AS A SOURCE PROPERTY**: the module imports no torch,
   defines nothing named like a loss/penalty/repair, and touches no gradient.
8. **THE REAL ``V6Stack.emit`` PATH** is exercised on CPU at the tiny geometry:
   its 60-step plan is ONE rollout, ``split_bands`` returns VIEWS of it, and the
   probe finds no seam in the emitted waypoints.

Every literal here is MEASURED on this box (2026-08-16, torch CPU, numpy
seeds stated inline) — recompute on drift, never inherit.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
_REPO = _STACK.parent
for _p in (str(_STACK), str(_STACK / "scripts"), str(_REPO / "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from taniteval import ci as _ci                              # noqa: E402
from taniteval import seam                                   # noqa: E402
from tanitad.config import (                                 # noqa: E402
    EncoderConfig, PredictorConfig, ReadoutConfig)
from tanitad.models import v6 as v6mod                       # noqa: E402
from tanitad.models.v6 import V6Config, V6Stack              # noqa: E402
from train_v58f_unicycle_head import unicycle_rollout        # noqa: E402

TOOL = _REPO / "taniteval" / "tools" / "seam_probe.py"
MODULE_SRC = (_REPO / "taniteval" / "taniteval" / "seam.py").read_text(
    encoding="utf-8")
TOOL_SRC = TOOL.read_text(encoding="utf-8")

_spec = importlib.util.spec_from_file_location("seam_probe_under_test", TOOL)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

#: keep every bootstrap in this file cheap; the ESTIMATOR is what is under test,
#: not its Monte-Carlo resolution (which is pinned in taniteval/tests/test_ci.py).
NB = 200


# --------------------------------------------------------------------------- #
# helpers — synthetic arms with a KNOWN answer                                 #
# --------------------------------------------------------------------------- #
def ar1(rng, n, t, rho, sigma):
    """Temporally COHERENT series — what a trained planner emits, and what the
    instrument needs in order to have any power in control space at all."""
    e = rng.normal(0.0, sigma, size=(n, t))
    x = np.empty((n, t))
    x[:, 0] = e[:, 0]
    for j in range(1, t):
        x[:, j] = rho * x[:, j - 1] + e[:, j]
    return x


def eids(n_ep, per_ep):
    return [f"ep{e:03d}" for e in range(n_ep) for _ in range(per_ep)]


def roll(a, k, v0, dt=seam.DT):
    """The PROGRAMME's integrator (train_v58f_unicycle_head.unicycle_rollout),
    imported and called — never re-implemented, so the SE(2) geometry here is
    the one v6 actually emits through."""
    wp, _ = unicycle_rollout(torch.tensor(a)[:, None, :],
                             torch.tensor(k)[:, None, :],
                             torch.tensor(v0), dt=dt)
    return wp[:, 0].numpy()


def genuine_and_stitched(n_ep=12, per_ep=12, t=seam.PLAN_STEPS,
                         s=seam.SEAM_BOUNDARY, seed=0):
    rng = np.random.default_rng(seed)
    n = n_ep * per_ep
    v0 = np.clip(rng.normal(12.0, 4.0, size=n), 0.5, None)
    a1, k1 = ar1(rng, n, t, 0.92, 0.30), ar1(rng, n, t, 0.95, 0.02)
    a2, k2 = ar1(rng, n, t, 0.92, 0.30), ar1(rng, n, t, 0.95, 0.02)
    wp_g = roll(a1, k1, v0)
    a_s = np.concatenate([a1[:, :s], a2[:, s:]], axis=1)
    k_s = np.concatenate([k1[:, :s], k2[:, s:]], axis=1)
    wp_b = roll(a2, k2, v0)
    return {
        "eid": eids(n_ep, per_ep),
        "genuine": {"controls": np.stack([a1, k1], -1), "waypoints": wp_g},
        "stitch_controls": {"controls": np.stack([a_s, k_s], -1),
                            "waypoints": roll(a_s, k_s, v0)},
        # two INDEPENDENT rollouts, the second re-based at its own origin
        "stitch_rollout": {"controls": np.stack([a1, k1], -1),
                           "waypoints": np.concatenate(
                               [wp_g[:, :s], wp_b[:, :t - s]], axis=1)},
    }


def tiny_cfg(**kw) -> V6Config:
    """test_v6_staged.py's tiny geometry. ``plan_steps`` stays 60 because §4b
    — the ONE 60-step rollout — is exactly what is under test."""
    base = dict(
        encoder=EncoderConfig(in_channels=3, image_size=32, image_width=32,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                  horizons=(1, 2), action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32,
        f_hidden_tac=32, f_hidden_str=32, d_plan_feat=16, emission_hidden=16,
        n_candidates=3, aux_hidden=16, sigreg_slices=8)
    base.update(kw)
    return V6Config(**base)


# =========================================================================== #
# 1. the geometry is DERIVED from v6, never guessed                           #
# =========================================================================== #
def test_seam_boundary_is_the_v6_band_edge_not_a_literal():
    cfg = tiny_cfg()
    op, tac = cfg.band_slice("op"), cfg.band_slice("tac")
    assert op.stop == tac.start, "the bands must MEET — a gap IS the defect"
    assert seam.SEAM_BOUNDARY == op.stop == tac.start == 20
    assert seam.PLAN_STEPS == v6mod.PLAN_STEPS == cfg.plan_steps == 60
    assert seam.DT == v6mod.DT == cfg.dt
    assert tuple(seam.OP_BAND_S) == tuple(v6mod.OP_BAND_S)
    assert tuple(seam.TAC_BAND_S) == tuple(v6mod.TAC_BAND_S)
    # and the instrument RE-DERIVES it from the band spec
    assert seam.seam_boundary_of() == 20
    assert seam.seam_boundary_of((0.0, 3.0), (3.0, 6.0), 0.1) == 30


def test_seam_boundary_of_refuses_a_gap_or_an_overlap():
    """The same refusal ``V6Config.__post_init__`` makes: a gap or an overlap
    here IS the stitched-trajectory defect, so the probe must not silently
    measure across it."""
    with pytest.raises(ValueError, match="gap/overlap"):
        seam.seam_boundary_of((0.0, 2.0), (2.5, 6.0), 0.1)
    with pytest.raises(ValueError, match="gap/overlap"):
        seam.seam_boundary_of((0.0, 2.0), (1.5, 6.0), 0.1)
    with pytest.raises(ValueError, match="start at 0"):
        seam.seam_boundary_of((0.5, 2.0), (2.0, 6.0), 0.1)


# =========================================================================== #
# 2. the stencil                                                              #
# =========================================================================== #
def test_boundary_diffs_are_the_documented_finite_differences():
    x = np.array([[0.0, 1.0, 3.0, 6.0, 10.0]])          # 2nd diff constant 1
    d1, i1 = seam.boundary_diffs(x, 1)
    assert i1.tolist() == [1, 2, 3, 4]
    np.testing.assert_allclose(d1[0], [1.0, 2.0, 3.0, 4.0])
    d2, i2 = seam.boundary_diffs(x, 2)
    assert i2.tolist() == [1, 2, 3]
    np.testing.assert_allclose(d2[0], [1.0, 1.0, 1.0])
    d3, i3 = seam.boundary_diffs(x, 3)
    assert i3.tolist() == [1, 2]
    np.testing.assert_allclose(d3[0], [0.0, 0.0])
    # D_m(i) reads x[i-1 .. i+m-1] — the documented anchoring
    i = 2
    assert d2[0][i - 1] == pytest.approx(abs(x[0, i + 1] - 2 * x[0, i]
                                             + x[0, i - 1]))
    with pytest.raises(ValueError, match="order-4"):
        seam.boundary_diffs(np.zeros((2, 4)), 4)


def test_seam_and_null_refuses_a_boundary_that_is_not_valid_at_this_order():
    d, idx = seam.boundary_diffs(np.zeros((3, 22)), 3)     # valid i in [1, 19]
    with pytest.raises(ValueError, match="not among"):
        seam.seam_and_null(d, idx, seam=20)


def test_local_null_is_a_symmetric_window_excluding_the_seam():
    d, idx = seam.boundary_diffs(np.zeros((2, seam.PLAN_STEPS)), 1)
    _, _, nidx = seam.seam_and_null(d, idx, 20, halfwidth=3)
    assert nidx.tolist() == [17, 18, 19, 21, 22, 23]
    _, _, gidx = seam.seam_and_null(d, idx, 20, halfwidth=None)
    assert 20 not in gidx.tolist() and len(gidx) == 58


# =========================================================================== #
# 3. THE ESTIMATOR IS taniteval's, BY IDENTITY                                #
# =========================================================================== #
def test_the_interval_is_the_episode_cluster_bootstrap_and_it_is_CALLED():
    """Not "looks like": the ci functions are monkeypatched and the counters
    must move. A re-implementation would leave them at zero."""
    rng = np.random.default_rng(3)
    x = ar1(rng, 60, seam.PLAN_STEPS, 0.9, 0.2)
    e = eids(10, 6)
    calls = {"paired": 0, "single": 0}
    real_p, real_s = _ci.paired_episode_cluster_bootstrap, \
        _ci.episode_cluster_bootstrap

    def wrap_p(*a, **k):
        calls["paired"] += 1
        return real_p(*a, **k)

    def wrap_s(*a, **k):
        calls["single"] += 1
        return real_s(*a, **k)

    seam._ci.paired_episode_cluster_bootstrap = wrap_p
    seam._ci.episode_cluster_bootstrap = wrap_s
    try:
        r = seam.continuity(x, e, n_boot=NB)
    finally:
        seam._ci.paired_episode_cluster_bootstrap = real_p
        seam._ci.episode_cluster_bootstrap = real_s
    assert calls["paired"] >= 1 and calls["single"] >= 1
    assert r["excess_ci"]["estimator"] == "paired_episode_cluster_bootstrap"
    assert r["rank_ci"]["estimator"] == "episode_cluster_bootstrap"
    assert r["excess_ci"]["n_episodes"] == 10


def test_overlapping_holdout_se_appears_nowhere_in_the_instrument():
    """⛔ the deprecated estimator narrows 1.107-3.100x AND biases the point
    estimate — up to a SIGN FLIP on paired deltas, which is exactly the shape
    of the seam contrast. It must not be reachable from here at all."""
    for name, src in (("seam.py", MODULE_SRC), ("seam_probe.py", TOOL_SRC)):
        for bad in ("overlapping_holdout_se(", "from .ci import overlapping",
                    "heldout"):
            assert bad not in src, f"{name} references {bad!r}"
    assert "overlapping_holdout_se" not in seam.__all__


def test_paired_and_single_arm_agree_on_the_point_estimate():
    """``paired(a, b)`` reduces to ``single(a - b)`` for reduce='mean' — the
    module relies on that to take ``se`` from one and the verdict from the
    other, so it is pinned rather than assumed."""
    rng = np.random.default_rng(11)
    a, b = rng.normal(size=90) + 0.3, rng.normal(size=90)
    e = eids(9, 10)
    p = _ci.paired_episode_cluster_bootstrap(a, b, e, n_boot=NB)
    s = _ci.episode_cluster_bootstrap(a - b, e, reduce="mean", n_boot=NB)
    assert p["delta"] == pytest.approx(s["mean"], abs=1e-4)
    assert p["lo"] == pytest.approx(s["lo"], abs=1e-4)


# =========================================================================== #
# 4. VERIFY, NEVER REPAIR — as a source property                              #
# =========================================================================== #
def test_the_instrument_defines_no_loss_no_gradient_no_repair():
    """⛔ X2 is 'seam metrics VERIFY, never repair' (the binding diagram cell).
    The prohibition is checked on the SOURCE, not promised in prose: nothing
    here can become a training term without this test failing."""
    low = MODULE_SRC.lower()
    assert "import torch" not in MODULE_SRC, \
        "seam.py must stay numpy-only so the statistics are testable anywhere"
    for tok in ("requires_grad", "backward(", "optim", "loss_", "add_loss",
                "nn.module", "penalt"):
        assert tok not in low, f"seam.py contains {tok!r}"
    for fn in seam.__all__:
        assert "loss" not in fn.lower() and "repair" not in fn.lower()
    # the mandate travels IN the artifact, so a reader of the JSON cannot miss it
    assert "VERIFY, NEVER REPAIR" in seam._READ.upper()
    x = np.zeros((10, 30)) + np.arange(30)
    panel = seam.continuity_panel({"x": x}, eids(5, 2), seam=10, orders=(1,),
                                  n_boot=50)
    assert "VERIFY, NEVER REPAIR" in seam.seam_report(panel).upper()
    assert all(r["verdict"] in seam.VERDICTS for r in panel["rows"])


# =========================================================================== #
# 5. H0 behaviour + the MEASURED median-reference bias (the regression pin)   #
# =========================================================================== #
def test_under_H0_the_excess_is_unbiased_and_the_rank_is_one_half():
    """Exchangeable data: i.i.d. across steps, so EVERY boundary is drawn from
    the same distribution and there is no seam by construction."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(40 * 12, seam.PLAN_STEPS))
    e = eids(40, 12)
    r = seam.continuity(x, e, n_boot=NB, seed=0)
    assert r["excess_rel_ci"]["lo"] < 0.0 < r["excess_rel_ci"]["hi"], \
        f"H0 excess interval must cover 0, got {r['excess_rel_ci']}"
    assert r["rank_mean"] == pytest.approx(0.5, abs=0.05)
    assert r["top1_rate"] == pytest.approx(r["top1_h0"], abs=0.02)
    assert r["verdict"] == "NO_MATERIAL_SEAM"


def test_the_MEDIAN_reference_is_BIASED_under_H0_which_is_why_it_is_not_used():
    """⛔ REGRESSION PIN. The median looks like the robust reference and is not
    admissible: for the right-skewed |Δ^m x| distribution ``E[median of n
    draws] < E[draw]``, so a median-referenced excess is POSITIVE under H0 and
    manufactures seams in seamless rollouts. MEASURED here on exchangeable
    data; do not "improve" seam.py back to a median reference."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40 * 12, seam.PLAN_STEPS))
    e = eids(40, 12)
    d, idx = seam.boundary_diffs(x, 1)
    d_seam, d_null, _ = seam.seam_and_null(d, idx, seam.SEAM_BOUNDARY)
    med_excess = _ci.paired_episode_cluster_bootstrap(
        d_seam, np.median(d_null, axis=-1), e, n_boot=NB)
    mean_excess = _ci.paired_episode_cluster_bootstrap(
        d_seam, d_null.mean(axis=-1), e, n_boot=NB)
    assert med_excess["separated"] and med_excess["delta"] > 0, (
        "the median reference must be shown BIASED here, otherwise this pin "
        f"has stopped protecting anything: {med_excess}")
    assert not mean_excess["separated"], (
        f"the MEAN reference must be unbiased under H0: {mean_excess}")
    # and the shipped code uses the mean
    assert "MEAN, not median" in MODULE_SRC
    r = seam.continuity(x, e, n_boot=NB)
    assert r["null_scale"] == pytest.approx(float(d_null.mean()), abs=1e-3)


# =========================================================================== #
# 6. IT FIRES ON AN INJECTED SEAM (the validation the brief requires)         #
# =========================================================================== #
@pytest.mark.parametrize("arm,channels", [
    ("stitch_controls", {"a", "kappa"}),
    ("stitch_rollout", {"wp_x", "wp_y"}),
])
def test_the_probe_FIRES_on_two_independently_rolled_bands_concatenated(
        arm, channels):
    d = genuine_and_stitched()
    ch = seam.control_channels(controls=d[arm]["controls"],
                               waypoints=d[arm]["waypoints"])
    panel = seam.continuity_panel(ch["channels"], d["eid"],
                                 units=ch["units"], n_boot=NB)
    assert panel["headline"] == "SEAM", panel["headline"]
    fired = {t.split("/")[0] for t in
             panel["seam_rows_confirmed_both_nulls"]}
    assert fired & channels, (
        f"{arm}: expected a CONFIRMED seam on {channels}, got "
        f"{panel['seam_rows_confirmed_both_nulls']}")


def test_the_boundary_scan_puts_the_injected_seam_at_the_top_with_a_low_FPR():
    """⭐ THE C13 DEFENCE, measured: the same rule applied at all 59 boundaries
    must (a) rank the injected seam FIRST and (b) fire on few or no others.
    A rule that fires everywhere has detected nothing when it fires at 20."""
    d = genuine_and_stitched()
    sc = seam.boundary_scan(d["stitch_controls"]["controls"][..., 0],
                            d["eid"], channel="a", n_boot=100)
    assert sc["argmax_boundary"] == seam.SEAM_BOUNDARY
    assert sc["seam_rank_among_boundaries"] == 0
    assert sc["seam_is_argmax"] is True
    assert sc["false_positive_rate"] <= 0.05, sc["false_positive_boundaries"]


def test_the_scan_on_a_genuine_rollout_finds_no_hotspot_at_the_band_edge():
    d = genuine_and_stitched()
    sc = seam.boundary_scan(d["genuine"]["controls"][..., 0], d["eid"],
                            channel="a", n_boot=100)
    assert sc["seam_is_argmax"] is False
    assert sc["false_positive_rate"] <= 0.05, sc["false_positive_boundaries"]


# =========================================================================== #
# 7. IT DOES NOT FIRE ON A GENUINE SINGLE ROLLOUT — and says so with power    #
# =========================================================================== #
def test_a_genuine_single_rollout_returns_a_WELL_POWERED_null():
    d = genuine_and_stitched()
    ch = seam.control_channels(controls=d["genuine"]["controls"],
                               waypoints=d["genuine"]["waypoints"])
    panel = seam.continuity_panel(ch["channels"], d["eid"],
                                  units=ch["units"], n_boot=NB)
    assert panel["seam_rows_confirmed_both_nulls"] == []
    assert panel["headline"] == "NO_MATERIAL_SEAM"
    # ⭐ the POSITIVE half: the test COULD have seen a material seam.
    for row in panel["rows"]:
        p = row["power"]
        assert p["powered_for_material_seam"] is True, (
            f"{row['channel']}/d{row['order']}/{row['null']} is under-powered: "
            f"MDE {p['mde_power80_rel']} vs floor {p['materiality_floor_rel']}")
        assert p["mde_power80_rel"] < 1.0
        assert p["n_episodes"] == 12 and p["n_windows"] == 144


# =========================================================================== #
# 8. ALL FOUR VERDICTS ARE REACHABLE                                          #
# =========================================================================== #
def test_every_verdict_in_the_space_is_reachable_none_is_decorative():
    rng = np.random.default_rng(1)
    got = set()
    # SEAM
    d = genuine_and_stitched()
    got.add(seam.continuity(d["stitch_controls"]["controls"][..., 0],
                            d["eid"], n_boot=NB)["verdict"])
    # NO_MATERIAL_SEAM
    got.add(seam.continuity(d["genuine"]["controls"][..., 0], d["eid"],
                            n_boot=NB)["verdict"])
    # INCONCLUSIVE — 8 episodes x 1 window cannot resolve a 0.25x-scale bar
    got.add(seam.continuity(rng.normal(size=(8, seam.PLAN_STEPS)),
                            eids(8, 1), n_boot=NB,
                            materiality_k=0.25)["verdict"])
    # DEGENERATE — exactly the zero-init emission head's output (a = k = 0)
    got.add(seam.continuity(np.zeros((20, seam.PLAN_STEPS)), eids(5, 4),
                            n_boot=NB)["verdict"])
    assert got == set(seam.VERDICTS), f"unreachable verdicts: "\
        f"{set(seam.VERDICTS) - got}"


def test_a_zero_init_emission_head_reads_DEGENERATE_not_PASS():
    """The v6 emission's final layer is zero-init, so at step 0 every control is
    exactly (0, 0) and the plan is the CV straight rollout. That has no
    discontinuities to rank — the honest reading is DEGENERATE, and a
    'no seam detected' there would be a vacuous pass."""
    r = seam.continuity(np.zeros((30, seam.PLAN_STEPS)), eids(6, 5),
                        n_boot=NB)
    assert r["verdict"] == "DEGENERATE"
    assert any("IDENTICALLY ZERO" in n for n in r["notes"])
    assert r["rank_mean"] == pytest.approx(0.5), \
        "mid-ranks must make an all-equal signal read exactly H0"


def test_an_underpowered_null_is_INCONCLUSIVE_not_a_clean_bill():
    """8 episodes x 1 window against a strict 0.25x materiality bar: the data
    simply cannot resolve the effect the reader asked about."""
    r = seam.continuity(np.random.default_rng(2).normal(size=(8, 60)),
                        eids(8, 1), n_boot=NB, materiality_k=0.25)
    assert r["verdict"] == "INCONCLUSIVE"
    assert r["power"]["powered_for_material_seam"] is False
    assert r["power"]["enough_episode_clusters"] is True   # n is not the issue
    assert any("UNDER-POWERED" in n for n in r["notes"])


def test_too_few_episode_clusters_can_NEVER_produce_a_clean_bill():
    """⛔ MEASURED: at 2 episodes x 2 windows the cluster bootstrap's SE
    collapses to 0.019x the within-band scale — an 80 %-power MDE of 0.054x,
    which would license a 'well-powered null' off FOUR windows. A cluster
    bootstrap over n episodes has only C(2n-1, n) distinct resamples (3 at
    n=2), so its 2.5th percentile is arithmetic, not evidence. Refuse."""
    r = seam.continuity(np.random.default_rng(2).normal(size=(4, 60)),
                        eids(2, 2), n_boot=NB)
    assert r["power"]["mde_power80_rel"] < 0.1, \
        "the pathological SE this guard exists for has changed — re-measure"
    assert r["verdict"] == "INCONCLUSIVE"
    assert r["power"]["enough_episode_clusters"] is False
    assert r["power"]["powered_for_material_seam"] is False
    assert any("TOO FEW EPISODE CLUSTERS" in n for n in r["notes"])
    assert seam.MIN_EPISODES_FOR_CLEAN_BILL == 8


# =========================================================================== #
# 9. bands — per band, never pooled; paired across arms                       #
# =========================================================================== #
def test_band_errors_are_reported_separately_with_their_own_estimator():
    rng = np.random.default_rng(4)
    gt = np.cumsum(rng.normal(size=(120, seam.PLAN_STEPS, 2)), axis=1)
    pred = gt + rng.normal(scale=0.1, size=gt.shape) \
        * np.linspace(1.0, 4.0, seam.PLAN_STEPS)[None, :, None]
    e = eids(12, 10)
    b = seam.band_errors(pred, gt, e, n_boot=NB, tier="T1", arm="x")
    assert set(b["bands"]) == {"ade_0_2s", "ade_2_6s", "ade_0_6s_pooled"}
    for k in ("ade_0_2s", "ade_2_6s"):
        assert b["bands"][k]["estimator"] == "episode_cluster_bootstrap"
        assert b["bands"][k]["n_episodes"] == 12
    assert b["band_delta_tac_minus_op"]["estimator"] == \
        "paired_episode_cluster_bootstrap"
    assert b["tier"] == "T1"
    assert "never pooled" in b["_read"].lower()
    # the bands really are the v6 slices
    assert b["seam_boundary"] == 20 and b["plan_steps"] == 60


def test_band_errors_refuse_to_be_stamped_with_no_tier():
    rng = np.random.default_rng(9)
    gt = rng.normal(size=(20, 60, 2))
    b = seam.band_errors(gt + 0.1, gt, eids(5, 4), n_boot=50)
    assert b["tier"] == "UNSTAMPED", "an omitted tier must be VISIBLE"


def test_paired_band_deltas_use_the_paired_estimator_on_identical_windows():
    rng = np.random.default_rng(6)
    gt = rng.normal(size=(60, 60, 2))
    a = gt + rng.normal(scale=0.2, size=gt.shape)
    b = gt + rng.normal(scale=0.5, size=gt.shape)
    r = seam.band_errors_paired(a, b, gt, eids(10, 6), n_boot=NB)
    for k in ("ade_0_2s", "ade_2_6s"):
        assert r["paired_delta_a_minus_b"][k]["estimator"] == \
            "paired_episode_cluster_bootstrap"
        assert r["paired_delta_a_minus_b"][k]["delta"] < 0   # a is the better
    assert "quadrature" in r["_read"]


# =========================================================================== #
# 10. THE REAL V6Stack.emit PATH                                              #
# =========================================================================== #
def test_the_real_emit_produces_ONE_rollout_whose_bands_are_VIEWS():
    """The construction claim is a claim about CODE, so it is checked on the
    code: ``emit`` returns 60 steps, ``split_bands`` returns views that
    reassemble to the identical tensor, and the two bands MEET at step 20."""
    torch.manual_seed(0)
    cfg = tiny_cfg()
    s = V6Stack(copy.deepcopy(cfg))
    z = torch.randn(6, cfg.d_op)
    g = torch.randn(6, cfg.d_goal_embed)
    v0 = torch.rand(6) * 20.0
    out = s.emit(z, g, v0)
    wp = out["waypoints"]
    assert wp.shape == (6, cfg.n_candidates, 60, 2)
    assert out["controls"].shape == (6, cfg.n_candidates, 60, 2)
    op, tac = cfg.split_bands(wp, dim=-2)
    assert op.shape[-2] == 20 and tac.shape[-2] == 40
    assert torch.equal(torch.cat([op, tac], dim=-2), wp), \
        "the bands must reassemble EXACTLY — they are slices of one rollout"
    assert op.data_ptr() == wp.data_ptr(), "split_bands must return a VIEW"


def test_the_real_emit_path_carries_no_seam_at_the_band_edge():
    """⭐ THE ARCHITECTURAL CHECK. Weights are irrelevant to a BY-CONSTRUCTION
    claim, so this runs the real ``V6Stack.emit`` with the emission head
    perturbed off its zero-init (which would be DEGENERATE by construction) and
    asserts the probe finds no CONFIRMED seam in the emitted waypoints.

    ⚠️ HONEST LIMIT, recorded rather than hidden: at random init the emission
    MLP's last layer maps one feature to all 120 outputs through independent
    rows, so the emitted controls are WHITE across steps. The within-band null
    is then as large as any control-space stitch could make it and the CONTROL
    channels have very little power — which the power block reports. The
    POSITION channels retain power, and they are the ones that see a stitched
    rollout. This test therefore asserts on the waypoint channels."""
    torch.manual_seed(0)
    cfg = tiny_cfg(n_candidates=1)
    s = V6Stack(copy.deepcopy(cfg))
    with torch.no_grad():                      # off the CV warm start
        torch.nn.init.normal_(s.emission.net[-1].weight, std=0.05)
        torch.nn.init.normal_(s.emission.net[-1].bias, std=0.05)
    n_ep, per_ep = 10, 8
    z = torch.randn(n_ep * per_ep, cfg.d_op)
    g = torch.randn(n_ep * per_ep, cfg.d_goal_embed)
    v0 = torch.rand(n_ep * per_ep) * 20.0 + 1.0
    out = s.emit(z, g, v0)
    ch = seam.control_channels(controls=out["controls"][:, 0].detach(),
                               waypoints=out["waypoints"][:, 0].detach())
    panel = seam.continuity_panel(ch["channels"], eids(n_ep, per_ep),
                                  units=ch["units"], orders=(1, 2),
                                  n_boot=NB)
    wp_rows = [t for t in panel["seam_rows_confirmed_both_nulls"]
               if t.startswith("wp_")]
    assert wp_rows == [], f"the real emit path shows a POSITION seam: {wp_rows}"


# =========================================================================== #
# 11. candidate handling                                                      #
# =========================================================================== #
def test_the_winner_is_the_probed_candidate_and_a_fan_without_sel_is_refused():
    fan = np.arange(2 * 3 * 60 * 2, dtype=np.float64).reshape(2, 3, 60, 2)
    sel = np.array([2, 0])
    got, mode = probe.pick_candidate(fan, sel, "winner", "controls")
    assert mode == "winner"
    np.testing.assert_allclose(got[0], fan[0, 2])
    np.testing.assert_allclose(got[1], fan[1, 0])
    with pytest.raises(SystemExit, match="no 'sel'"):
        probe.pick_candidate(fan, None, "winner", "controls")
    allc, mode = probe.pick_candidate(fan, sel, "all", "controls")
    assert mode == "all(3)" and allc.shape == (6, 60, 2)


def test_candidates_of_one_window_keep_that_windows_episode_id():
    """⚠️ candidates are NOT independent draws — treating them as new episodes
    would shrink the interval by ~sqrt(C) for free."""
    e = probe.expand_eid(np.array(["a", "b"]), 3)
    assert e == ["a", "a", "a", "b", "b", "b"]
    assert probe.expand_eid(np.array(["a", "b"]), 1) == ["a", "b"]


# =========================================================================== #
# 12. the CLI                                                                 #
# =========================================================================== #
def _run(args, cwd=None):
    return subprocess.run([sys.executable, str(TOOL), *args],
                          capture_output=True, text=True, cwd=cwd)


def test_cli_self_test_validates_the_instrument_end_to_end():
    """``--self-test`` IS the seam-injection validation, and it exits non-zero
    when the instrument fails to detect an injected seam."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "selftest.json"
        r = _run(["--self-test", "--quiet", "--n-boot", "150",
                  "--self-test-episodes", "10", "--self-test-per-episode", "8",
                  "--out", str(out)])
        assert r.returncode == 0, r.stdout + r.stderr
        rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["validated"] is True
    names = {c["name"]: c["pass"] for c in rec["checks"]}
    assert all(names.values()), names
    assert "injected_control_seam_FIRES_on_a_or_kappa" in names
    assert "injected_rollout_seam_FIRES_on_wp" in names
    assert rec["arms"]["genuine"]["headline"] == "NO_MATERIAL_SEAM"
    assert rec["arms"]["stitch_controls"]["headline"] == "SEAM"
    assert rec["arms"]["stitch_rollout"]["headline"] == "SEAM"


def test_cli_refuses_a_dump_with_no_tier():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "d.npz"
        rng = np.random.default_rng(0)
        np.savez(p, eid=np.array(eids(4, 4)),
                 controls=rng.normal(size=(16, 60, 2)))
        r = _run(["--dump", str(p), "--quiet"])
        assert r.returncode != 0
        assert "no admissible tier" in (r.stdout + r.stderr)
        r2 = _run(["--dump", str(p), "--tier", "T1", "--quiet",
                   "--n-boot", "100", "--no-scan", "--orders", "1"])
        assert r2.returncode == 0, r2.stdout + r2.stderr


def test_cli_refuses_a_dump_with_no_episode_ids():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "d.npz"
        np.savez(p, controls=np.zeros((8, 60, 2)), tier="T1")
        r = _run(["--dump", str(p), "--quiet"])
        assert r.returncode != 0
        assert "eid" in (r.stdout + r.stderr)


def test_cli_writes_a_record_carrying_the_estimator_the_tier_and_the_mandate():
    d = genuine_and_stitched(n_ep=8, per_ep=6)
    with tempfile.TemporaryDirectory() as td:
        p, o = Path(td) / "g.npz", Path(td) / "g.json"
        np.savez(p, eid=np.array(d["eid"]), tier="T1", arm="unit",
                 plan_steps=60, dt=0.1, op_band_s=np.array([0.0, 2.0]),
                 tac_band_s=np.array([2.0, 6.0]),
                 **d["genuine"])
        r = _run(["--dump", str(p), "--quiet", "--n-boot", "150",
                  "--scan-n-boot", "80", "--orders", "1,2", "--out", str(o)])
        assert r.returncode == 0, r.stdout + r.stderr
        rec = json.loads(o.read_text(encoding="utf-8"))
    assert rec["tier"] == "T1" and rec["arm"] == "unit"
    assert rec["seam_boundary"] == 20 and rec["seam_boundary_derived"] == 20
    assert rec["seam_boundary_overridden"] is False
    assert "episode_cluster_bootstrap" in rec["estimator"]
    assert "VERIFY, NEVER REPAIR" in rec["mandate"]
    assert rec["continuity"]["headline"] == "NO_MATERIAL_SEAM"
    assert rec["boundary_scan"]["seam_is_argmax"] is False
    assert rec["continuity"]["tier_invariant"] is True


def test_cli_paired_arms_show_WHY_the_bands_are_never_pooled():
    """⭐ THE POINT OF PER-BAND REPORTING, demonstrated end to end. Arm B is arm
    A with ONLY the tactical band replaced, so the operative band is
    BIT-IDENTICAL (delta exactly 0, not separated) while the tactical band is
    metres apart. A pooled 0-6 s delta blurs the two into one number that
    describes neither."""
    d = genuine_and_stitched(n_ep=10, per_ep=8)
    gt = d["genuine"]["waypoints"] + 0.3
    with tempfile.TemporaryDirectory() as td:
        pa, pb, o = Path(td) / "a.npz", Path(td) / "b.npz", Path(td) / "ab.json"
        for p, arm in ((pa, "genuine"), (pb, "stitch_rollout")):
            np.savez(p, eid=np.array(d["eid"]), tier="T1", arm=arm, gt=gt,
                     **d[arm])
        r = _run(["--dump", str(pa), "--dump-b", str(pb), "--quiet",
                  "--n-boot", "150", "--no-scan", "--orders", "1",
                  "--out", str(o)])
        assert r.returncode == 0, r.stdout + r.stderr
        rec = json.loads(o.read_text(encoding="utf-8"))
    bp = rec["bands_paired"]["paired_delta_a_minus_b"]
    assert bp["ade_0_2s"]["delta"] == 0.0 and not bp["ade_0_2s"]["separated"]
    assert bp["ade_2_6s"]["separated"] and bp["ade_2_6s"]["delta"] < 0
    for k in ("ade_0_2s", "ade_2_6s"):
        assert bp[k]["estimator"] == "paired_episode_cluster_bootstrap"
    # and the single-arm bands each carry their OWN interval
    assert rec["bands"]["bands"]["ade_0_2s"]["estimator"] == \
        "episode_cluster_bootstrap"


def test_cli_refuses_paired_bands_when_an_arm_has_no_waypoints():
    d = genuine_and_stitched(n_ep=8, per_ep=4)
    with tempfile.TemporaryDirectory() as td:
        pa, pb = Path(td) / "a.npz", Path(td) / "b.npz"
        np.savez(pa, eid=np.array(d["eid"]), tier="T1",
                 controls=d["genuine"]["controls"])
        np.savez(pb, eid=np.array(d["eid"]), tier="T1",
                 **d["stitch_rollout"])
        r = _run(["--dump", str(pa), "--dump-b", str(pb), "--quiet",
                  "--n-boot", "50", "--no-scan", "--orders", "1"])
        assert r.returncode != 0
        assert "waypoints" in (r.stdout + r.stderr)


def test_cli_refuses_a_dump_whose_declared_geometry_contradicts_its_arrays():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "d.npz"
        np.savez(p, eid=np.array(eids(4, 4)), tier="T1", plan_steps=40,
                 controls=np.zeros((16, 60, 2)))
        r = _run(["--dump", str(p), "--quiet"])
        assert r.returncode != 0
        assert "plan_steps" in (r.stdout + r.stderr)
