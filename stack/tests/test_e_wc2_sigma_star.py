"""E-WC2 CPU tests — the σ* estimator, on SYNTHETIC latents with a PLANTED σ.

The instrument (`scripts/e_wc2_sigma_star.py`) decides whether SEL-1 is funded or
refused before a GPU-hour is spent, so every load-bearing property is pinned here
against data whose answer is known by construction:

  * **the ridge recovers the planted σ** — y = Xw + N(0, s²) per axis, and the
    LOEO out-of-fold σ_perax comes back at s (within Monte-Carlo tolerance). The
    per-axis / radial unit relation (√2) is pinned too, because reading σ in the
    radial unit against a threshold stated in the per-axis unit would inflate it
    by 1.414 and could flip the verdict on arithmetic alone.
  * **it is the P1/P2 ridge, not a second one** — ``ridge_oof_predict``'s pooled
    OOF R² is pinned EQUAL to ``probe_latent_state.ridge_probe_cv``'s.
  * **LOEO folds are EPISODE-disjoint, not window-disjoint** — the REF-A I-JEPA
    defect (~80 % of val inside train). One test asserts the partition; a second
    plants an episode-level nuisance the leaky scheme can memorise and shows the
    leak reports a SMALLER σ, i.e. a downward bias on exactly this number.
  * **the refusal path** — short n, short episodes, relaxed guards, a missing 6 s
    horizon and an ECHO feature block all yield ``NO_VERDICT``, never a softened
    FUNDED/REFUSED.
  * **the §5.3 REDERIVE flag** — fires above 3×, does not below, and NEVER emits
    a scaled 6 s threshold in either branch.
  * **the verdict bands** at the exact pre-registered boundaries (1.7 / 3.0).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import e_wc2_sigma_star as E  # noqa: E402
from probe_latent_state import RIDGE_LAMBDAS, ridge_probe_cv  # noqa: E402

N_EP = E.PREREG["min_episodes"]          # 40 — the pre-registered surface
N_WIN = E.PREREG["min_windows"]          # 881


# ============================================================================
# synthetic surfaces
# ============================================================================
def _eids(n_win=N_WIN, n_ep=N_EP):
    """Contiguous per-episode runs, like the real stride-8 grid."""
    base, extra = divmod(n_win, n_ep)
    out = []
    for e in range(n_ep):
        out += [e] * (base + (1 if e < extra else 0))
    return out


def make_dump(*, sigma2=0.8, sigma6=1.6, n_win=N_WIN, n_ep=N_EP, d=8,
              sel_ade=0.4714, oracle_ade=0.1639, seed=0, with_6s=True,
              valid6=None, echo=False):
    """A dump conformant with ``DUMP_CONTRACT`` whose σ is PLANTED.

    The endpoint at each horizon is an exactly-linear function of ``pooled`` plus
    isotropic N(0, σ²) per axis, so the OOF ridge σ must return σ. The fan is
    constructed so ``oracle_ade``/``sel_ade`` come out at the requested values,
    which lets a test drive σ/ADE to a chosen ratio and check the band.
    """
    rng = np.random.default_rng(seed)
    eid = _eids(n_win, n_ep)
    n = len(eid)
    pooled = rng.normal(size=(n, d))
    w2 = rng.normal(size=(d, 2))
    w6 = rng.normal(size=(d, 2))
    e2 = pooled @ w2 + rng.normal(0.0, sigma2, size=(n, 2))
    e6 = pooled @ w6 + rng.normal(0.0, sigma6, size=(n, 2))

    steps, tgt = [20], [e2]
    if with_6s:
        steps, tgt = [20, 60], [e2, e6]
    gt_endpoint = np.stack(tgt, axis=1)                       # [n, He, 2]
    val = np.ones((n, len(steps)), dtype=bool)
    if with_6s and valid6 is not None:
        val[:, 1] = np.asarray(valid6, dtype=bool)

    # a 2-candidate fan at 4 waypoints whose last step is 20 (2.0 s): candidate 0
    # sits `oracle_ade` from GT, candidate 1 sits `sel_ade`; sel picks index 1.
    T, C = 4, 2
    gt = np.zeros((n, T, 2))
    fan = np.zeros((n, C, T, 2))
    fan[:, 0, :, 0] = oracle_ade
    fan[:, 1, :, 0] = sel_ade
    d_out = {
        "eid": eid,
        "pooled": torch.tensor(pooled, dtype=torch.float32),
        "ctx": torch.zeros(n, 0),
        "gt_endpoint": torch.tensor(gt_endpoint, dtype=torch.float64),
        "endpoint_steps": steps,
        "endpoint_valid": torch.tensor(val),
        "fan": torch.tensor(fan, dtype=torch.float64),
        "gt": torch.tensor(gt, dtype=torch.float64),
        "sel": torch.ones(n, dtype=torch.long),
        "wp_steps": [5, 10, 15, 20],
        "ckpt": "synthetic", "ckpt_step": 30000, "nav_mode": "follow_constant",
    }
    if echo:
        d_out["measurement"] = torch.tensor(rng.normal(size=(n, 4)),
                                            dtype=torch.float32)
    return d_out


def _run(dump, **kw):
    kw.setdefault("features", ["pooled"])
    kw.setdefault("n_boot", 0)
    return E.run(dump, **kw)


# ============================================================================
# the ridge recovers the planted sigma
# ============================================================================
@pytest.mark.parametrize("planted", [0.4, 0.8, 2.0])
def test_ridge_recovers_planted_sigma(planted):
    d = make_dump(sigma2=planted, sigma6=planted, seed=3)
    res = _run(d)
    got = res["sigma"]["2s"]["sigma_perax_m"]
    assert abs(got - planted) / planted < 0.06, (planted, got)


def test_perax_and_radial_differ_by_sqrt2():
    """⛔ The unit that decides the verdict. §3.1 injects N(0, s) PER AXIS
    (sel_winners_curse_law.py:221), so the radial RMS is √2 larger and the 1.7
    threshold is NOT defined against it."""
    res = _run(make_dump(sigma2=0.8, seed=5))
    row = res["sigma"]["2s"]
    assert row["sigma_radial_rms_m"] == pytest.approx(
        row["sigma_perax_m"] * np.sqrt(2), rel=1e-9)
    assert "PER-AXIS" in row["_unit_note"]


def test_sigma_from_residuals_hand_case():
    """res = [[3,4]] repeated: |e|² = 25, so radial RMS = 5 and per-axis
    = 5/√2 = 3.5355; long = 3, lat = 4."""
    s = E.sigma_from_residuals(np.tile(np.array([[3.0, 4.0]]), (7, 1)))
    assert s["sigma_radial_rms_m"] == pytest.approx(5.0)
    assert s["sigma_perax_m"] == pytest.approx(5.0 / np.sqrt(2))
    assert s["sigma_long_m"] == pytest.approx(3.0)
    assert s["sigma_lat_m"] == pytest.approx(4.0)
    assert s["n"] == 7


def test_sigma_from_residuals_refuses_wrong_shape():
    with pytest.raises(ValueError, match=r"\[n, 2\]"):
        E.sigma_from_residuals(np.zeros((5, 3)))


def test_four_family_axes_are_reported():
    res = _run(make_dump(seed=7))
    row = res["sigma"]["2s"]
    assert row["sigma_long_m"] > 0 and row["sigma_lat_m"] > 0
    assert set(res["four_families"]) == {"LONGITUDINAL", "LATERAL", "TACTICAL",
                                         "STRATEGIC"}


# ============================================================================
# it is the P1/P2 ridge, not a second one
# ============================================================================
def test_ridge_oof_matches_probe_battery():
    """``ridge_oof_predict`` must be ``ridge_probe_cv`` + the predictions."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(240, 6))
    y = X @ rng.normal(size=6) + rng.normal(0.0, 0.3, size=240)
    folds = E.loeo_folds(_eids(240, 12))
    a = E.ridge_oof_predict(X, y, folds, RIDGE_LAMBDAS)
    b = ridge_probe_cv(X, y, folds, RIDGE_LAMBDAS)
    assert a["r2"] == pytest.approx(b["r2"], rel=1e-12)
    assert a["n"] == b["n"]
    assert a["lambda_by_fold"] == b["lambda_by_fold"]
    for u, v in zip(a["per_fold_r2"], b["per_fold_r2"]):
        assert u == pytest.approx(v, rel=1e-12)


# ============================================================================
# LOEO folds — EPISODE-disjoint, not window-disjoint
# ============================================================================
def test_loeo_is_one_episode_per_fold():
    eid = np.asarray(_eids(N_WIN, N_EP))
    folds = E.loeo_folds(eid)
    assert np.unique(folds).size == N_EP
    for f in np.unique(folds):
        assert np.unique(eid[folds == f]).size == 1


def test_loeo_no_episode_straddles_a_fold():
    """The REF-A I-JEPA rule, asserted directly: every window of an episode is
    on the same side of every split."""
    eid = np.asarray(_eids(N_WIN, N_EP))
    folds = E.loeo_folds(eid)
    for e in np.unique(eid):
        assert np.unique(folds[eid == e]).size == 1, f"episode {e} straddles"


def test_loeo_refuses_below_two_episodes():
    with pytest.raises(ValueError, match="LOEO needs"):
        E.loeo_folds([7] * 50)


def test_window_disjoint_folds_leak_and_understate_sigma():
    """⛔ THE LEAK, MEASURED. An episode-level nuisance feature that is constant
    within an episode and random across episodes is memorisable by a
    WINDOW-random split (the episode is in train) and unavailable to LOEO (the
    episode is held out). The leaky scheme therefore reports a SMALLER σ — a
    downward bias on the exact number that decides SEL-1.

    MEASURED on this fixture: LOEO **8.195** vs window-random **0.201** — the
    leaky split recovers the 0.2 noise floor exactly (it memorised every
    episode's offset) and understates σ by **40×**."""
    rng = np.random.default_rng(17)
    n_ep, per_ep, d, d_style = 24, 30, 4, 30
    eid = np.repeat(np.arange(n_ep), per_ep)
    n = eid.size
    # a per-episode CODE wide enough (d_style > n_ep) that a linear map can
    # memorise an arbitrary per-episode offset from the episodes it has seen —
    # and cannot generalise to an episode it has not.
    style = rng.normal(size=(n_ep, d_style))[eid]   # constant WITHIN an episode
    obs = rng.normal(size=(n, d))
    X = np.concatenate([obs, style], axis=1)
    offset = rng.normal(0.0, 4.0, size=(n_ep, 2))[eid]   # arbitrary per episode
    y = obs @ rng.normal(size=(d, 2)) + offset + rng.normal(0.0, 0.2, size=(n, 2))

    def sig(folds):
        res = np.stack([y[:, j] - E.ridge_oof_predict(X, y[:, j], folds)["yhat"]
                        for j in range(2)], axis=1)
        return E.sigma_from_residuals(res)["sigma_perax_m"]

    s_loeo = sig(E.loeo_folds(eid))
    s_leak = sig(E.window_random_folds(n, 5, seed=1))
    assert s_leak < s_loeo * 0.1, (s_leak, s_loeo)


# ============================================================================
# the verdict bands
# ============================================================================
@pytest.mark.parametrize("ratio,want", [
    (0.5, "FUNDED"), (1.6999, "FUNDED"), (1.7, "FUNDED"),
    (1.7001, "INCONCLUSIVE"), (2.4, "INCONCLUSIVE"), (2.9999, "INCONCLUSIVE"),
    (3.0, "REFUSED"), (9.0, "REFUSED"),
])
def test_decide_bands_at_the_prereg_boundaries(ratio, want):
    assert E.decide(ratio, [])["verdict"] == want


def test_decide_guards_beat_any_ratio():
    """A guard failure can only produce NO_VERDICT — never a softened FUNDED."""
    out = E.decide(0.1, ["n short"])
    assert out["verdict"] == "NO_VERDICT"
    assert out["refusal_reasons"] == ["n short"]
    assert "NOT the same as REFUSED" in out["meaning"]


def test_no_verdict_is_a_distinct_token_from_refused():
    assert "NO_VERDICT" in E.VERDICTS and "REFUSED" in E.VERDICTS
    assert E.VERDICTS["NO_VERDICT"] != E.VERDICTS["REFUSED"]


def test_end_to_end_funded_band():
    """σ ≈ 0.55 against sel_ade 0.4714 ⇒ ratio ≈ 1.17 ⇒ FUNDED."""
    res = _run(make_dump(sigma2=0.55, sigma6=0.9, seed=21))
    assert res["decision"]["verdict"] == "FUNDED"
    r = res["references_and_ratios"]
    # the emitted ratios are the UNROUNDED quotient, rounded once at the end —
    # so they agree with the published 4-dp components to within that rounding
    assert r["sigma_over_ade"] == pytest.approx(
        r["sigma_perax_2s_m"] / r["sel_ade_incumbent"], abs=5e-4)
    assert r["sigma_over_oracle"] == pytest.approx(
        r["sigma_perax_2s_m"] / r["oracle_ade"], abs=5e-4)


def test_end_to_end_refused_band():
    """σ ≈ 1.6 against sel_ade 0.4714 ⇒ ratio ≈ 3.4 ⇒ REFUSED (SEL-1 refused)."""
    res = _run(make_dump(sigma2=1.6, sigma6=2.5, seed=23))
    assert res["decision"]["verdict"] == "REFUSED"
    assert "ANCHOR_GOAL" in res["decision"]["meaning"]


def test_end_to_end_inconclusive_band():
    res = _run(make_dump(sigma2=1.0, sigma6=1.4, seed=27))
    assert res["decision"]["verdict"] == "INCONCLUSIVE"
    assert "capacity control" in res["decision"]["meaning"]


def test_reference_ratios_from_3_1_are_carried():
    r = _run(make_dump(seed=29))["references_and_ratios"]
    assert r["reference_from_3_1"]["sigma_star_m"] == 0.8
    assert r["reference_from_3_1"]["sigma_over_ade"] == 1.7
    assert r["reference_from_3_1"]["sigma_over_oracle"] == 4.9


def test_published_3_1_numbers_reproduce_the_published_ratios():
    """Pins the UNIT: 0.8 m per-axis / 0.4714 = 1.70 and / 0.1639 = 4.88 are
    §3.1's published ratios. A radial-RMS reading would give 2.40 and 6.90."""
    assert 0.8 / 0.4714 == pytest.approx(E.PREREG["reference_ratio_vs_ade"],
                                         abs=0.01)
    assert 0.8 / 0.1639 == pytest.approx(E.PREREG["reference_ratio_vs_oracle"],
                                         abs=0.02)


# ============================================================================
# refusal paths
# ============================================================================
def test_refuses_when_episodes_short():
    res = _run(make_dump(n_win=200, n_ep=10, seed=31), min_episodes=10,
               min_windows=200)
    assert res["decision"]["verdict"] == "NO_VERDICT"
    joined = " ".join(res["decision"]["refusal_reasons"])
    assert "RELAXED" in joined


def test_refuses_when_windows_short_without_relaxing():
    res = _run(make_dump(n_win=300, n_ep=N_EP, seed=33))
    assert res["decision"]["verdict"] == "NO_VERDICT"
    joined = " ".join(res["decision"]["refusal_reasons"])
    assert "300 windows < the pre-registered 881" in joined


def test_refuses_when_6s_horizon_absent():
    """§5.2 requires σ at 2 s AND 6 s — a 2 s-only dump gets NO_VERDICT even
    though its 2 s ratio is comfortably inside the FUNDED band."""
    res = _run(make_dump(sigma2=0.4, with_6s=False, seed=35))
    assert res["decision"]["verdict"] == "NO_VERDICT"
    joined = " ".join(res["decision"]["refusal_reasons"])
    assert "6 s" in joined
    assert res["sigma"]["2s"]["available"] is True     # still inspectable
    assert res["sigma"]["6s"]["available"] is False


def test_echo_feature_block_is_refused_by_default():
    """The binding vision-only rule, enforced. `measurement` is the ego+nav
    embedding (refc_dump_latents.py:28) and is not deployable at inference."""
    with pytest.raises(PermissionError, match="ECHO"):
        _run(make_dump(echo=True, seed=37), features=["pooled", "measurement"])


def test_echo_feature_allowed_only_as_a_labelled_control_and_forces_no_verdict():
    res = _run(make_dump(sigma2=0.3, echo=True, seed=39),
               features=["pooled", "measurement"], allow_echo=True)
    assert res["features"]["any_echo"] is True
    assert res["decision"]["verdict"] == "NO_VERDICT"
    assert any("INADMISSIBLE" in r for r in res["decision"]["refusal_reasons"])


def test_undeclared_feature_block_is_refused_until_declared():
    d = make_dump(seed=41)
    d["mystery"] = torch.zeros(len(d["eid"]), 3)
    with pytest.raises(PermissionError, match="admissibility class"):
        _run(d, features=["pooled", "mystery"])
    res = _run(d, features=["pooled", "mystery"],
               declared={"mystery": "VISION_ONLY"})
    assert any(b["block"] == "mystery" and b["used"] for b in
               res["features"]["blocks"])


def test_zero_width_block_is_dropped_with_a_reason():
    """`ctx` is [n, 0] on a non-hierarchy arm — dropped, stated, not a crash."""
    res = _run(make_dump(seed=43), features=["pooled", "ctx"])
    ctx = [b for b in res["features"]["blocks"] if b["block"] == "ctx"][0]
    assert ctx["used"] is False and ctx["dim"] == 0
    assert "zero-width" in ctx["reason"]


def test_producer_instrument_fail_blocks_the_verdict():
    d = make_dump(sigma2=0.4, seed=45)
    d["instrument_fail"] = ["fan_bit_identical FAILED"]
    res = _run(d)
    assert res["decision"]["verdict"] == "NO_VERDICT"
    assert any("instrument_fail" in r for r in res["decision"]["refusal_reasons"])


# ============================================================================
# §5.3 — the REDERIVE flag
# ============================================================================
def test_rederive_fires_above_three_times():
    res = _run(make_dump(sigma2=0.5, sigma6=2.0, seed=47))     # 4x
    rd = res["rederive_check_5_3"]
    assert rd["rederive_required"] is True
    assert rd["threshold_transfer"] == "REDERIVE"
    assert rd["multiple"] > 3.0
    assert rd["threshold_6s"] is None
    assert "RE-DERIVED" in rd["reason"] and "NOT scaled" in rd["reason"]


def test_rederive_does_not_fire_below_three_times():
    res = _run(make_dump(sigma2=0.5, sigma6=1.0, seed=49))     # 2x
    rd = res["rederive_check_5_3"]
    assert rd["rederive_required"] is False
    assert rd["threshold_transfer"] == "RATIO_FORM_HOLDS"
    assert rd["multiple"] < 3.0


def test_no_branch_ever_emits_a_scaled_6s_threshold():
    """⛔ The §5.3 instruction is 're-derived, NOT scaled'. Neither branch may
    hand back a 6 s threshold."""
    for s6 in (0.6, 1.0, 2.0, 5.0):
        rd = E.rederive_check(0.5, s6)
        assert rd["threshold_6s"] is None
    assert E.rederive_check(0.5, None)["threshold_transfer"] == "NOT_TESTABLE"
    assert E.rederive_check(None, 0.5)["rederive_required"] is None


def test_rederive_uses_matched_windows_not_the_full_grid():
    """The 6 s endpoint is missing for the last windows of every episode. The 3×
    comparison must re-fit σ(2 s) on exactly the 6 s-valid subset — comparing a
    881-window σ(2 s) against a truncated σ(6 s) would compare different windows."""
    n_win = N_WIN
    eid = np.asarray(_eids(n_win, N_EP))
    valid6 = np.ones(n_win, dtype=bool)
    for e in np.unique(eid):                     # drop the tail of each episode
        pos = np.flatnonzero(eid == e)
        valid6[pos[-5:]] = False
    res = _run(make_dump(sigma2=0.5, sigma6=1.0, valid6=valid6, seed=51))
    rd = res["rederive_check_5_3"]
    assert rd["matched_windows"] is True
    assert "valid at BOTH horizons" in rd["matched_window_note"]
    assert res["sigma"]["6s"]["n_windows"] == int(valid6.sum())
    assert res["sigma"]["6s"]["n_excluded"] == int((~valid6).sum())
    assert "never imputed" in res["sigma"]["6s"]["excluded_reason"]
    # the 2 s headline still uses the FULL grid
    assert res["sigma"]["2s"]["n_windows"] == n_win


# ============================================================================
# the dump contract
# ============================================================================
def test_contract_validates_a_conformant_dump():
    assert E.validate_dump(make_dump(seed=53)) == []


def test_contract_reports_every_missing_piece():
    problems = E.validate_dump({"eid": [0, 1]})
    joined = " | ".join(problems)
    assert "gt_endpoint" in joined
    assert "fan" in joined and "sel" in joined
    assert "feature block" in joined


def test_contract_requires_both_20_and_60_steps():
    d = make_dump(seed=55)
    d["endpoint_steps"] = [20, 40]
    d["gt_endpoint"] = d["gt_endpoint"][:, :2]
    joined = " | ".join(E.validate_dump(d))
    assert "no 60" in joined


def test_contract_requires_endpoint_valid_mask():
    d = make_dump(seed=57)
    del d["endpoint_valid"]
    assert any("endpoint_valid" in p for p in E.validate_dump(d))


def test_print_contract_names_the_producer_command():
    txt = json.dumps(E.DUMP_CONTRACT)
    assert "refc_dump_latents.py --endpoint-steps 20,60" in txt
    assert "K_MAX" in txt


# ============================================================================
# the PRODUCER half of the contract — scripts/refc_dump_latents.py
# ============================================================================
def test_producer_endpoint_mask_marks_out_of_range_horizons():
    """A straight-line episode of 30 poses at 1 m/step. From `last`, the 2 s
    (step 20) endpoint exists only while last+20 < 30; the 6 s (step 60) endpoint
    never does. Out-of-range entries are NaN AND masked False — a consumer that
    ignores the mask fails loudly rather than reading 'the ego stopped'."""
    import refc_dump_latents as R
    poses = torch.zeros(30, 4)
    poses[:, 0] = torch.arange(30.0)
    last = torch.tensor([0, 5, 9, 10, 25])
    ep, val = R.gt_endpoints_masked(poses, last, [20, 60])
    assert ep.shape == (5, 2, 2) and val.shape == (5, 2)
    assert val[:, 0].tolist() == [True, True, True, False, False]
    assert val[:, 1].tolist() == [False] * 5
    assert torch.allclose(ep[0, 0], torch.tensor([20.0, 0.0]))
    assert bool(torch.isnan(ep[~val]).all())
    assert bool(torch.isfinite(ep[val]).all())


def test_producer_endpoint_matches_the_fan_gt_where_horizons_coincide():
    """The self-control the producer asserts: at a horizon that COINCIDES with a
    fan waypoint, the endpoint block must be bit-identical to `gt` — same rows,
    same `last`, same ego-frame convention."""
    import driving_diagnostic as dd
    import refc_dump_latents as R
    g = torch.Generator().manual_seed(5)
    poses = torch.zeros(120, 4)
    poses[:, :2] = torch.cumsum(torch.rand(120, 2, generator=g), dim=0)
    poses[:, 2] = torch.linspace(0.0, 1.3, 120)
    last = torch.arange(8, 60, 8)
    ep, val = R.gt_endpoints_masked(poses, last, [20, 60])
    gt = dd.gt_ego_waypoints(poses, last)                  # wp_steps 5,10,15,20
    assert bool(val.all())
    assert torch.equal(ep[:, 0], gt[:, list(dd.WP_STEPS).index(20)])


def test_producer_does_not_widen_the_window_grid():
    """⛔ PARITY. K_MAX must stay at max(WP_STEPS) = 20; widening it to 60 would
    re-select windows, shrink the 881-window grid and break the fan bit-identity
    gate. The producer's grid constant is pinned here."""
    import driving_diagnostic as dd
    import refc_dump_latents as R
    src = Path(R.__file__).read_text(encoding="utf-8")
    assert "K_MAX = max(dd.WP_STEPS)" in src
    assert max(dd.WP_STEPS) == 20
    assert (R.WINDOW, R.STRIDE) == (8, 8)


class _Ep:
    """The minimal episode interface ``backfill_endpoints`` needs."""

    def __init__(self, poses, episode_id, n_frames=None):
        self.poses, self.episode_id = poses, episode_id
        if n_frames is not None:
            self.n_frames = n_frames


def _synth_episodes(n_ep=4, n_frames=200, seed=3):
    g = torch.Generator().manual_seed(seed)
    eps = []
    for e in range(n_ep):
        poses = torch.zeros(n_frames, 4)
        poses[:, :2] = torch.cumsum(torch.rand(n_frames, 2, generator=g), dim=0)
        poses[:, 2] = torch.linspace(0.0, 0.9 + 0.1 * e, n_frames)
        poses[:, 3] = 8.0
        eps.append(_Ep(poses, e))
    return eps


def _banked_from(eps):
    """A banked-dump stand-in built on the SAME grid the producer builds."""
    import driving_diagnostic as dd
    import refc_dump_latents as R
    GT, EID = [], []
    for ep in eps:
        starts = R.window_starts(ep.poses.shape[0])
        last = torch.tensor([t + R.WINDOW - 1 for t in starts])
        GT.append(dd.gt_ego_waypoints(ep.poses, last))
        EID.extend([ep.episode_id] * len(starts))
    gt = torch.cat(GT).float()
    return {"eid": EID, "gt": gt, "wp_steps": list(dd.WP_STEPS),
            "pooled": torch.zeros(len(EID), 4),
            "fan": torch.zeros(len(EID), 2, 4, 2),
            "sel": torch.zeros(len(EID), dtype=torch.long)}


def test_backfill_adds_endpoints_with_no_model_and_passes_the_alignment_gate():
    """⭐ The 0-GPU path: endpoints are GROUND TRUTH, so a banked dump missing
    them needs poses, not a re-inference pass."""
    import refc_dump_latents as R
    eps = _synth_episodes()
    out = R.backfill_endpoints(_banked_from(eps), eps, [20, 60])
    ctl = out["endpoint_backfill_controls"]
    assert ctl["eid_match"] is True
    assert ctl["endpoint_20_matches_gt"] is True     # the per-row fingerprint
    assert ctl["fails"] == []
    assert out["endpoint_steps"] == [20, 60]
    assert out["gt_endpoint"].shape == (len(out["eid"]), 2, 2)
    assert 0.0 < ctl["valid_frac"]["60"] < 1.0       # the tail is genuinely lost
    assert ctl["valid_frac"]["20"] == 1.0
    assert E.validate_dump(out) == []
    assert "no model, no GPU" in out["endpoint_provenance"]


def test_backfill_refuses_a_misaligned_grid():
    """⛔ THE GATE. If the rebuilt grid does not match the banked one, every
    latent would be regressed onto a NEIGHBOUR's endpoint — σ inflated, and the
    wrong answer would look exactly like a measurement."""
    import refc_dump_latents as R
    eps = _synth_episodes()
    banked = _banked_from(eps)
    with pytest.raises(AssertionError, match="does not match the banked eid"):
        R.backfill_endpoints(banked, eps[:-1], [20, 60])   # one episode short


def test_backfill_refuses_when_the_pose_source_differs():
    """Same window count, DIFFERENT poses — caught by the bit-identity check at
    the coinciding 2 s waypoint, which the eid check alone cannot see."""
    import refc_dump_latents as R
    eps = _synth_episodes()
    banked = _banked_from(eps)
    other = _synth_episodes(seed=99)                  # same shapes, other poses
    with pytest.raises(AssertionError, match="not bit-identical"):
        R.backfill_endpoints(banked, other, [20, 60])


def test_backfill_no_strict_records_the_failure_instead_of_raising():
    import refc_dump_latents as R
    eps = _synth_episodes()
    out = R.backfill_endpoints(_banked_from(eps), _synth_episodes(seed=99),
                               [20, 60], strict=False)
    assert out["endpoint_backfill_controls"]["fails"]
    assert E.validate_dump(out) == []      # shape-conformant but gate-failed
    # …and E-WC2 still refuses, because the producer's gate is not E-WC2's job
    # to re-litigate — a caller must not be able to launder a failed alignment.


def test_backfill_tolerates_a_last_bit_libm_difference():
    """⭐ The dumps were produced on Thor (aarch64) and the backfill runs on x86;
    `cos/sin` disagree in the LAST BIT, so a literal `torch.equal` refuses a
    CORRECT backfill. One ULP of jitter on the banked column must still pass."""
    import refc_dump_latents as R
    eps = _synth_episodes()
    banked = _banked_from(eps)
    gt = banked["gt"].clone()
    col = gt[:, 3].clone().reshape(-1)               # the 2 s waypoint column
    mask = torch.zeros_like(col, dtype=torch.bool)
    mask[::7] = True                                 # perturb ~1/7 of the entries
    nudged = torch.nextafter(col, torch.full_like(col, float("inf")))
    col[mask] = nudged[mask]
    gt[:, 3] = col.reshape(gt[:, 3].shape)
    banked["gt"] = gt
    out = R.backfill_endpoints(banked, eps, [20, 60])          # must NOT raise
    agr = out["endpoint_backfill_controls"]["endpoint_20_agreement"]
    assert agr["ok"] is True
    assert agr["bit_identical"] is False              # it really was perturbed
    assert agr["max_row_ulps"] <= R.ENDPOINT_ULP_TOL
    assert agr["separation_factor"] >= R.ENDPOINT_SHIFT_MARGIN


def test_backfill_still_refuses_a_one_row_shift():
    """⛔ The failure the gate exists for, stated as the thing the tolerance must
    NOT admit: a single-row roll of the banked column is ~1e5 ULPs, five orders
    of magnitude above the libm noise the tolerance allows."""
    import refc_dump_latents as R
    eps = _synth_episodes()
    banked = _banked_from(eps)
    banked["gt"] = torch.roll(banked["gt"], shifts=1, dims=0)
    with pytest.raises(AssertionError, match="not bit-identical"):
        R.backfill_endpoints(banked, eps, [20, 60])


def test_endpoint_agreement_reports_the_shift_control_not_just_a_boolean():
    """The gate publishes its evidence: a bare `ok` cannot be audited, and the
    ±1-row control is the half that a `torch.equal` never had."""
    import refc_dump_latents as R
    a = torch.stack([torch.arange(1.0, 51.0), torch.zeros(50)], dim=-1)
    same = R.endpoint_agreement(a, a.clone())
    assert same["ok"] and same["bit_identical"]
    assert same["separation_factor"] == float("inf")
    assert same["rows_bit_identical"] == same["n_rows"] == 50
    off = R.endpoint_agreement(a, torch.roll(a, 1, 0))
    assert off["ok"] is False
    assert off["max_row_ulps"] > R.ENDPOINT_ULP_TOL


def test_endpoint_agreement_row_ulp_is_not_per_component():
    """⚠️ A rotation spreads the LONGITUDINAL magnitude's last bit into the tiny
    LATERAL component. MEASURED on the real dumps: per-component ULPs read 256,
    row-magnitude ULPs read 1.118 — the per-component reading would refuse a
    correct backfill."""
    import refc_dump_latents as R
    a = torch.tensor([[71.826546, 0.51576555]])
    b = torch.tensor([[71.82655, 0.51576567]])
    err, ulps = R._row_ulps(a, b)
    assert float(ulps.max()) < 4.0
    per_component = (a - b).abs() / torch.finfo(torch.float32).eps / a.abs().min()
    assert float(per_component.max()) > 4.0          # the reading that misleads


def test_backfill_gate_is_not_vacuous_on_a_degenerate_block():
    """⛔ The check a bare `torch.equal` could never make: on an all-zero (parked
    ego) block every ±1-row shift ALSO matches bit-for-bit, so bit-identity is
    satisfied while carrying no evidence about alignment at all. The shift
    control refuses it."""
    import refc_dump_latents as R
    z = torch.zeros(50, 2)
    agr = R.endpoint_agreement(z, z.clone())
    assert agr["bit_identical"] is True              # vacuously
    assert agr["shift1_median_abs_m"] == 0.0
    assert agr["ok"] is False                        # …and refused anyway


def test_backfill_grid_is_the_producers_grid():
    import driving_diagnostic as dd
    import refc_dump_latents as R
    assert R.K_MAX_GRID == max(dd.WP_STEPS) == 20
    assert R.window_starts(200) == list(range(0, 200 - R.WINDOW - R.K_MAX_GRID,
                                              R.STRIDE))


def test_producer_output_satisfies_the_e_wc2_contract():
    """An end-to-end shape check: a dump assembled the way the producer assembles
    it must pass `validate_dump` with no problems."""
    import refc_dump_latents as R
    poses = torch.zeros(400, 4)
    poses[:, 0] = torch.arange(400.0) * 0.5
    last = torch.arange(8, 300, 8)
    ep, val = R.gt_endpoints_masked(poses, last, [20, 60])
    n = last.numel()
    d = {"eid": [0] * n, "pooled": torch.zeros(n, 4),
         "gt_endpoint": ep, "endpoint_steps": [20, 60], "endpoint_valid": val,
         "fan": torch.zeros(n, 2, 4, 2), "gt": torch.zeros(n, 4, 2),
         "sel": torch.zeros(n, dtype=torch.long), "wp_steps": [5, 10, 15, 20]}
    assert E.validate_dump(d) == []


# ============================================================================
# references / mismatch guards
# ============================================================================
def test_ratios_absent_when_the_dump_has_no_fan():
    d = make_dump(seed=59)
    del d["fan"]
    res = _run(d)
    assert res["references_and_ratios"]["available"] is False
    assert res["decision"]["verdict"] == "NO_VERDICT"


def test_ratios_refused_when_the_fan_horizon_is_not_the_verdict_horizon():
    """⛔ A σ at 2 s over an ADE at 1 s is not a quantity."""
    d = make_dump(seed=61)
    d["wp_steps"] = [2, 4, 6, 10]                 # last waypoint = 1.0 s
    res = _run(d)
    assert "MISMATCH" in res["references_and_ratios"]
    assert res["decision"]["verdict"] == "NO_VERDICT"


def test_stamps_travel_in_the_output():
    res = _run(make_dump(seed=63))
    assert "MEASURED (ours)" in res["_evidence_class"]
    assert "T0-DIAGNOSTIC" in res["_tier"]
    assert "overlapping_holdout_se is used NOWHERE" in res["_estimator"]
    assert res["_prereg"]["fund_at_or_below"] == 1.7
    assert res["_prereg"]["refuse_at_or_above"] == 3.0


# ============================================================================
# CLI
# ============================================================================
def _py(args, cwd):
    env = dict(**__import__("os").environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return subprocess.run([sys.executable, str(Path(__file__).resolve()
                                               .parents[1] / "scripts"
                                               / "e_wc2_sigma_star.py")] + args,
                          capture_output=True, text=True, cwd=str(cwd), env=env)


def test_cli_print_contract():
    p = _py(["--print-contract"], Path(__file__).parent)
    assert p.returncode == 0, p.stderr
    assert "gt_endpoint" in p.stdout


def test_cli_validate_only_exits_3_on_a_bad_dump(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"eid": [0, 1]}, bad)
    p = _py(["--dump", str(bad), "--validate-only"], tmp_path)
    assert p.returncode == 3, p.stdout + p.stderr
    assert '"conformant": false' in p.stdout


def test_cli_end_to_end_writes_json_and_exits_0(tmp_path):
    dump = tmp_path / "lat.pt"
    torch.save(make_dump(sigma2=0.5, sigma6=0.9, seed=67), dump)
    out = tmp_path / "ewc2.json"
    p = _py(["--dump", str(dump), "--features", "pooled", "--n-boot", "0",
             "--out", str(out)], tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["decision"]["verdict"] == "FUNDED"
    assert "verdict: FUNDED" in p.stdout


def test_cli_exits_4_when_it_refuses_a_verdict(tmp_path):
    dump = tmp_path / "lat.pt"
    torch.save(make_dump(sigma2=0.5, with_6s=False, seed=69), dump)
    p = _py(["--dump", str(dump), "--features", "pooled", "--n-boot", "0"],
            tmp_path)
    assert p.returncode == 4, p.stdout + p.stderr
    assert "NO_VERDICT" in p.stdout


# ============================================================================
# the interval — episode-cluster bootstrap, and NOTHING else
# ============================================================================
def test_bootstrap_ci_is_the_episode_cluster_estimator():
    res = _run(make_dump(sigma2=0.8, sigma6=1.2, seed=71), n_boot=200)
    ci = res["sigma"]["2s"]["sigma_perax_ci"]
    if "unavailable" in ci:
        pytest.skip("taniteval/ci.py not resolvable from this checkout")
    assert "episode" in ci["estimator"].lower()
    assert "overlapping" not in ci["estimator"].lower()
    assert ci["reducer"] == "rms"
    assert ci["n_episodes"] == N_EP
    assert ci["lo"] <= res["sigma"]["2s"]["sigma_perax_m"] <= ci["hi"]


def test_ratio_ci_shares_one_episode_draw():
    res = _run(make_dump(sigma2=0.55, sigma6=0.9, seed=73), n_boot=200)
    ci = res["references_and_ratios"].get("sigma_over_ade_ci")
    if ci is None or "unavailable" in ci:
        pytest.skip("taniteval/ci.py not resolvable from this checkout")
    assert "ONE shared episode draw" in ci["estimator"]
    assert ci["lo"] <= ci["ratio"] <= ci["hi"]
