"""P1/P2 CPU tests — ridge closed form, folds, R2, gate branches, lead-gap join.

Covers the off-pod-runnable part of the P1/P2 probe harness
(WM_PHYSICS_PROOF.md; scripts/probe_latent_state.py):
  * ridge closed form pinned against an sklearn-free HAND case (1-D, algebra
    done in the comment) and the three algebraically-identical solve paths
    (svd / primal / dual) pinned equal;
  * episode-disjoint fold builder: NO episode straddles folds (asserted), all
    folds used, refusal when episodes < folds; the within-episode BLOCK folds
    (the documented episode-ID deviation): every episode in every fold,
    blocks contiguous;
  * R2 computation on hand cases (0.8 case, perfect, constant-y -> None);
  * both probes recover synthetic linear/separable structure out-of-fold;
  * P1/P2 gate logic — every branch (pass / retention-fail / cliff-fail /
    not-computable x3; decay-pass / no-fall / ratio-fail / at-chance / missing);
  * lead-gap extraction: pure corridor logic + through the P8 JoinFileReader
    on a synthetic jsonl (NO_LABEL vs labelled-clear vs lead).

The full harness (GPU + checkpoint + v2 corpus + join) is POD-SIDE only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from probe_latent_state import (DRIVING_TARGETS, LEAD_LAT_M,  # noqa: E402
                                LEAD_MAX_GAP_M, MIN_N_PER_TARGET, RidgeSVD,
                                episode_disjoint_folds, lead_gap_from_agents,
                                local_curvature, logistic_probe_cv,
                                p1_gate_dict, p2_gate_dict, r2_score,
                                ridge_fit, ridge_probe_cv,
                                within_episode_block_folds, yaw_rate_at_k)
from train_p8_occupancy import JoinFileReader  # noqa: E402


# ============================================================================
# ridge closed form
# ============================================================================
def test_ridge_hand_case_1d():
    """X = [0,1,2,3], y = [0,1,2,3], lam = 2. Centered: Sxx = 5, Sxy = 5, so
    w = Sxy / (Sxx + lam) = 5/7 and b = ybar - xbar*w = 1.5 - 1.5*5/7 = 3/7 —
    computed BY HAND, no library."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    w, b = ridge_fit(X, y, lam=2.0)
    assert w.shape == (1,)
    assert abs(w[0] - 5.0 / 7.0) < 1e-12
    assert abs(b - 3.0 / 7.0) < 1e-12


def test_ridge_three_solve_paths_identical():
    rng = np.random.default_rng(0)
    for n, d in ((9, 4), (4, 9)):          # both the n>d and n<d regimes
        X = rng.normal(size=(n, d))
        y = rng.normal(size=n)
        for lam in (0.1, 1.0, 50.0):
            w_svd, b_svd = ridge_fit(X, y, lam, mode="svd")
            w_pri, b_pri = ridge_fit(X, y, lam, mode="primal")
            w_dua, b_dua = ridge_fit(X, y, lam, mode="dual")
            np.testing.assert_allclose(w_svd, w_pri, atol=1e-9)
            np.testing.assert_allclose(w_svd, w_dua, atol=1e-9)
            assert abs(b_svd - b_pri) < 1e-9 and abs(b_svd - b_dua) < 1e-9


def test_ridge_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ridge_fit(np.zeros((4, 2)), np.zeros(4), lam=0.0)     # lam must be > 0
    with pytest.raises(ValueError):
        RidgeSVD(np.zeros(4))                                  # not [n, d]
    with pytest.raises(ValueError):
        ridge_fit(np.zeros((4, 2)), np.zeros(4), 1.0, mode="nope")


def test_ridge_gcv_lambda_from_grid():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 6))
    y = X @ rng.normal(size=6) + 0.01 * rng.normal(size=40)
    solver = RidgeSVD(X)
    lam = solver.best_lambda(y, (1e-2, 1.0, 1e3))
    assert lam in (1e-2, 1.0, 1e3)
    # near-noiseless linear target: heavy shrinkage must not win
    assert lam != 1e3
    assert np.isfinite(solver.gcv(y, lam))


# ============================================================================
# R2
# ============================================================================
def test_r2_hand_cases():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    # ss_res = 1, ss_tot = 5 -> R2 = 0.8 (hand)
    assert abs(r2_score(y, np.array([1.0, 2.0, 3.0, 5.0])) - 0.8) < 1e-12
    assert r2_score(y, y) == 1.0
    assert r2_score(np.full(4, 2.0), np.zeros(4)) is None      # no variance
    # worse than the mean -> negative, not clipped
    assert r2_score(y, y[::-1]) < 0.0


# ============================================================================
# folds
# ============================================================================
def test_episode_disjoint_folds_no_straddle():
    rng = np.random.default_rng(2)
    uids = np.repeat(np.arange(100, 110), rng.integers(5, 40, size=10))
    folds = episode_disjoint_folds(uids, n_folds=5)
    assert folds.shape == uids.shape
    for u in np.unique(uids):
        assert np.unique(folds[uids == u]).size == 1, \
            f"episode {u} straddles folds"
    assert np.unique(folds).size == 5
    # deterministic
    np.testing.assert_array_equal(folds, episode_disjoint_folds(uids, 5))


def test_episode_disjoint_folds_refuses_too_few_episodes():
    with pytest.raises(ValueError):
        episode_disjoint_folds(np.array([1, 1, 2, 2, 3, 3]), n_folds=5)


def test_within_episode_block_folds_every_episode_every_fold():
    uids = np.repeat([7, 9], 10)
    folds = within_episode_block_folds(uids, n_folds=5)
    for u in (7, 9):
        assert np.unique(folds[uids == u]).size == 5
        # blocks are CONTIGUOUS: fold ids non-decreasing along the episode
        f = folds[uids == u]
        assert (np.diff(f) >= 0).all()


# ============================================================================
# probes recover synthetic structure out-of-fold
# ============================================================================
def test_ridge_probe_cv_recovers_linear_target():
    rng = np.random.default_rng(3)
    uids = np.repeat(np.arange(20), 10)                        # 20 eps x 10 win
    X = rng.normal(size=(200, 8))
    w_true = rng.normal(size=8)
    y = X @ w_true + 0.05 * rng.normal(size=200)
    folds = episode_disjoint_folds(uids, 5)
    res = ridge_probe_cv(X, y, folds)
    assert res["n"] == 200
    assert len(res["per_fold_r2"]) == 5
    assert res["r2"] > 0.9


def test_logistic_probe_cv_recovers_separable_classes():
    rng = np.random.default_rng(4)
    uids = np.repeat([11, 22, 33, 44], 20)                     # 4 "episodes"
    onehot = np.eye(4)[np.repeat(np.arange(4), 20)]
    X = np.hstack([5.0 * onehot, 0.1 * rng.normal(size=(80, 4))])
    folds = within_episode_block_folds(uids, 5)
    res = logistic_probe_cv(X, uids, folds)
    assert res["n_classes"] == 4
    assert abs(res["chance"] - 0.25) < 1e-12
    assert res["n"] == 80
    assert res["accuracy"] > 0.9


# ============================================================================
# P1 gate — every branch
# ============================================================================
def _p1_rows(r2_pred_by_k, r2_enc=0.9, n=500):
    return {k: {"r2_enc": r2_enc, "r2_pred": v, "n": n}
            for k, v in r2_pred_by_k.items()}


def test_p1_gate_pass_branch():
    table = {"speed": _p1_rows({5: 0.90, 10: 0.85, 15: 0.80, 20: 0.75})}
    g = p1_gate_dict(table, (5, 10, 15, 20), targets=("speed",))
    v = g["per_target"]["speed"]
    assert v["computable"] and v["retention_ok"] and v["cliff_ok"]
    assert v["pass"] is True and g["PASS"] is True
    json.dumps(g)                                              # JSON-safe


def test_p1_gate_retention_fail():
    # 0.5 / 0.9 = 0.556 < 0.85 -> retention fails; curve still smooth
    table = {"speed": _p1_rows({5: 0.55, 10: 0.50, 15: 0.45, 20: 0.40})}
    g = p1_gate_dict(table, (5, 10, 15, 20), targets=("speed",))
    v = g["per_target"]["speed"]
    assert v["computable"] and not v["retention_ok"] and v["cliff_ok"]
    assert g["PASS"] is False


def test_p1_gate_cliff_fail():
    # retention at k=10 is fine (0.88/0.9), but 15 -> 20 drops 0.60 >= 0.25
    table = {"speed": _p1_rows({5: 0.90, 10: 0.88, 15: 0.80, 20: 0.20})}
    g = p1_gate_dict(table, (5, 10, 15, 20), targets=("speed",))
    v = g["per_target"]["speed"]
    assert v["retention_ok"] and not v["cliff_ok"]
    assert g["PASS"] is False


def test_p1_gate_not_computable_branches():
    ks = (5, 10, 15, 20)
    # (i) encoded probe failed (r2_enc <= 0)
    t1 = {"speed": _p1_rows({5: 0.5, 10: 0.5, 15: 0.5, 20: 0.5}, r2_enc=-0.1)}
    v1 = p1_gate_dict(t1, ks, targets=("speed",))["per_target"]["speed"]
    assert v1["computable"] is False and v1["pass"] is None
    assert "R2(enc) <= 0" in v1["reason"]
    # (ii) too few labelled windows
    t2 = {"speed": _p1_rows({5: 0.9, 10: 0.9, 15: 0.9, 20: 0.9},
                            n=MIN_N_PER_TARGET - 1)}
    v2 = p1_gate_dict(t2, ks, targets=("speed",))["per_target"]["speed"]
    assert v2["computable"] is False and "too few" in v2["reason"]
    # (iii) gate horizon absent entirely
    t3 = {"speed": {5: {"r2_enc": 0.9, "r2_pred": 0.9, "n": 500}}}
    v3 = p1_gate_dict(t3, ks, targets=("speed",))["per_target"]["speed"]
    assert v3["computable"] is False and "not evaluated" in v3["reason"]
    # zero computable targets -> overall None, never a fake verdict
    assert p1_gate_dict(t1, ks, targets=("speed",))["PASS"] is None


def test_p1_gate_mixed_targets_and_missing_curve_point():
    ks = (5, 10, 15, 20)
    table = {
        "speed": _p1_rows({5: 0.90, 10: 0.85, 15: 0.80, 20: 0.75}),
        # hole at k=15 -> smoothness cannot be certified -> target fails
        "yaw_rate": {5: {"r2_enc": 0.9, "r2_pred": 0.9, "n": 500},
                     10: {"r2_enc": 0.9, "r2_pred": 0.85, "n": 500},
                     20: {"r2_enc": 0.9, "r2_pred": 0.8, "n": 500}},
        "curvature": _p1_rows({5: 0.9, 10: 0.85, 15: 0.8, 20: 0.75}),
        "lead_gap": _p1_rows({5: 0.9, 10: 0.85, 15: 0.8, 20: 0.75}, n=10),
    }
    g = p1_gate_dict(table, ks, targets=DRIVING_TARGETS)
    assert g["per_target"]["speed"]["pass"] is True
    assert g["per_target"]["yaw_rate"]["cliff_ok"] is False
    assert g["per_target"]["lead_gap"]["computable"] is False   # n too small
    assert g["n_computable"] == 3
    assert g["PASS"] is False                                   # yaw_rate fails


# ============================================================================
# P2 gate — every branch
# ============================================================================
CHANCE = 1.0 / 40.0


def test_p2_gate_decay_pass():
    g = p2_gate_dict({5: 0.60, 10: 0.4, 15: 0.3, 20: 0.20}, CHANCE)
    assert g["gate"]["branch"] == "decay_test"
    assert g["gate"]["falls"] is True
    assert abs(g["gate"]["ratio"] - 0.20 / 0.60) < 1e-5   # gate rounds to 6 dp
    assert g["PASS"] is True
    json.dumps(g)


def test_p2_gate_no_fall_fail():
    g = p2_gate_dict({5: 0.30, 20: 0.35}, CHANCE)
    assert g["gate"]["falls"] is False and g["PASS"] is False


def test_p2_gate_falls_but_ratio_fail():
    # falls, but 0.45/0.60 = 0.75 >= 0.5 -> appearance retained too long
    g = p2_gate_dict({5: 0.60, 20: 0.45}, CHANCE)
    assert g["gate"]["falls"] is True and g["PASS"] is False


def test_p2_gate_at_chance_branch():
    g = p2_gate_dict({5: CHANCE * 1.05, 20: CHANCE}, CHANCE)
    assert g["gate"]["branch"] == "at_chance_from_start"
    assert g["PASS"] is True


def test_p2_gate_missing_k_not_computable():
    g = p2_gate_dict({5: 0.6}, CHANCE)
    assert g["PASS"] is None and g["gate"]["computable"] is False


# ============================================================================
# lead-gap extraction
# ============================================================================
def _agent(cx, cy, l=4.0, w=2.0, yaw=0.0, occ=-1.0):
    return [cx, cy, yaw, l, w, occ]


def test_lead_gap_corridor_logic():
    # ahead, in corridor: gap = cx - l/2 = 10 - 2 = 8
    assert lead_gap_from_agents(np.array([_agent(10.0, 0.5)])) == 8.0
    # lateral 3 m > LEAD_LAT_M=2 -> no lead
    assert LEAD_LAT_M == 2.0
    assert lead_gap_from_agents(np.array([_agent(10.0, 3.0)])) is None
    # behind the ego (gap < 0) -> no lead
    assert lead_gap_from_agents(np.array([_agent(-5.0, 0.0)])) is None
    # beyond the 80 m corridor -> no lead
    assert LEAD_MAX_GAP_M == 80.0
    assert lead_gap_from_agents(np.array([_agent(90.0, 0.0)])) is None
    # nearest of two candidates wins
    two = np.array([_agent(30.0, 0.0), _agent(12.0, -1.0)])
    assert lead_gap_from_agents(two) == 10.0
    # NO_LABEL and labelled-clear both yield None (excluded per-target)
    assert lead_gap_from_agents(None) is None
    assert lead_gap_from_agents(np.zeros((0, 6))) is None


def test_lead_gap_through_join_reader(tmp_path):
    """Synthetic jsonl -> JoinFileReader -> lead_gap_from_agents, end to end:
    the exact reader the pod harness uses (I1a/P8 schema)."""
    from train_p8_occupancy import episode_uid_of_clip
    p = tmp_path / "agents.jsonl"
    recs = [
        {"clip_id": "clipA", "frame_idx": 3, "agents": [
            {"cx": 20.0, "cy": 0.2, "yaw": 0.0, "l": 4.0, "w": 2.0},
            {"cx": 6.0, "cy": 1.5, "yaw": 0.0, "l": 4.0, "w": 2.0},
            {"cx": 8.0, "cy": 5.0, "yaw": 0.0, "l": 4.0, "w": 2.0}]},
        {"clip_id": "clipA", "frame_idx": 4, "agents": []},     # labelled clear
    ]
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    join = JoinFileReader(p)
    uid = episode_uid_of_clip("clipA")
    # nearest in-corridor candidate: cx=6, l=4 -> gap 4 (the 20 m car is
    # farther; the cy=5 car is outside the corridor)
    assert lead_gap_from_agents(join.lookup(uid, 3)) == 4.0
    # labelled clear -> agents [0,6] -> no lead (excluded, NOT an error)
    ag_clear = join.lookup(uid, 4)
    assert ag_clear is not None and ag_clear.shape[0] == 0
    assert lead_gap_from_agents(ag_clear) is None
    # absent frame -> NO_LABEL -> None from the reader itself
    assert join.lookup(uid, 99) is None
    assert lead_gap_from_agents(join.lookup(uid, 99)) is None


# ============================================================================
# kinematic targets
# ============================================================================
def test_local_curvature_circle():
    """Poses on a radius-20 circle -> mean local curvature ~ 1/20 = 0.05
    (+left turn, the repo sign convention)."""
    R = 20.0
    theta = torch.arange(30, dtype=torch.float64) * 0.05
    poses = torch.stack([R * torch.sin(theta), R * (1 - torch.cos(theta)),
                         theta, torch.full_like(theta, 10.0)], dim=1).float()
    k = local_curvature(poses, center=15, half_steps=5)
    assert k is not None and abs(k - 1.0 / R) < 1e-3
    # too-short clip -> None (excluded, n reported), never a fake 0
    assert local_curvature(poses[:2], center=0) is None


def test_yaw_rate_at_k():
    H = 20
    fp = torch.zeros(1, H, 4)
    fp[0, :, 2] = torch.arange(H) * 0.02                       # 0.02 rad/step
    yr = yaw_rate_at_k(fp, k=10)
    assert abs(float(yr[0]) - 0.2) < 1e-6                      # 0.02 / 0.1 s
    # wrap: a pi-crossing must not read as a 60 rad/s spin
    fp2 = torch.zeros(1, H, 4)
    fp2[0, 8, 2] = 3.13
    fp2[0, 9, 2] = -3.13
    assert abs(float(yaw_rate_at_k(fp2, k=10)[0])) < 1.0
    with pytest.raises(ValueError):
        yaw_rate_at_k(fp, k=1)


# ============================================================================
# lead-gap CLASS FILTER (the 2026-08-11 instrument fix: a pedestrian in the
# corridor is not a lead VEHICLE)
# ============================================================================
def test_lead_gap_class_filter_drops_non_vehicles():
    from probe_latent_state import VEHICLE_CLASSES
    ag = np.array([_agent(6.0, 0.0), _agent(20.0, 0.2)])
    # class-agnostic (old behaviour, classes=None): pedestrian at 6 m wins
    assert lead_gap_from_agents(ag) == 4.0
    # with classes: the pedestrian is excluded, the automobile at 20 m leads
    cls = np.array(["person", "automobile"], dtype=object)
    assert lead_gap_from_agents(ag, classes=cls) == 18.0
    # corridor full of non-vehicles -> labelled, but NO lead vehicle
    assert lead_gap_from_agents(ag, classes=np.array(
        ["person", "cyclist"], dtype=object)) is None
    assert "automobile" in VEHICLE_CLASSES


def test_lead_gap_class_filter_misalignment_raises():
    ag = np.array([_agent(6.0, 0.0), _agent(20.0, 0.2)])
    with pytest.raises(ValueError):
        lead_gap_from_agents(ag, classes=np.array(["automobile"], dtype=object))


def test_join_reader_classes_roundtrip(tmp_path):
    """cls survives the reader aligned with lookup's rows; a class-less file
    stays loadable with has_classes False (old joins keep working)."""
    from train_p8_occupancy import episode_uid_of_clip
    p = tmp_path / "agents_cls.jsonl"
    recs = [
        {"clip_id": "clipB", "frame_idx": 2, "agents": [
            {"cx": 6.0, "cy": 0.0, "yaw": 0.0, "l": 4.0, "w": 2.0,
             "cls": "person"},
            {"cx": 20.0, "cy": 0.2, "yaw": 0.0, "l": 4.0, "w": 2.0,
             "cls": "automobile"}]},
    ]
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    join = JoinFileReader(p)
    uid = episode_uid_of_clip("clipB")
    assert join.has_classes is True
    cls = join.lookup_classes(uid, 2)
    assert list(cls) == ["person", "automobile"]
    # the wired path: reader agents + reader classes -> vehicle-pure gap
    assert lead_gap_from_agents(join.lookup(uid, 2), classes=cls) == 18.0
    # class-less legacy file: loadable, has_classes False, lookup_classes None
    q = tmp_path / "agents_nocls.jsonl"
    q.write_text(json.dumps({"clip_id": "clipB", "frame_idx": 2, "agents": [
        {"cx": 6.0, "cy": 0.0, "yaw": 0.0, "l": 4.0, "w": 2.0}]}),
        encoding="utf-8")
    legacy = JoinFileReader(q)
    assert legacy.has_classes is False
    assert legacy.lookup_classes(episode_uid_of_clip("clipB"), 2) is None
