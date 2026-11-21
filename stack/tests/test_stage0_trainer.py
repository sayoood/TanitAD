"""E4.4 ``train_tactical_stage0`` pure-helper smoke — CPU-only, no checkpoint,
no corpus. The full path (frozen v5f trunk + v2 corpora) is pod-side and NOT
exercised here.

What is pinned:
  * the CV-extrapolated goal baseline matches ``refb_labels.goal_tac_targets``
    to float precision on a synthetic constant-velocity case (FDE 0, heading
    hold, speed hold) — at yaw 0 AND at a fixed non-zero heading — and errs on
    a circular arc (it must: that is what the gate compares against);
  * the 1 Hz z_op window builder: exact indices at the ideal stride, the
    documented degraded tail at the v5f window=8, clamp-repeat below ``slots``
    frames, and shape/monotonicity/tail invariants over a range of lengths;
  * the E4.4 gate-JSON writer schema: pass/fail/None logic on the FDE@4s gate,
    macro_f1 null when aux is disabled, sel_gap as a threshold-free baseline
    row, and the estimator note naming the pod-side bootstrap rescore;
  * macro-F1 from a confusion matrix (support-weighted class exclusion).
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import refb_labels  # noqa: E402
from train_tactical_stage0 import (DT, GOAL_TAUS,  # noqa: E402
                                   build_e44_gate, cv_goal_baseline,
                                   macro_f1_from_confusion, tau_name,
                                   zop_window_indices)


# ---------------------------------------------------------------------------
# CV-extrapolated goal baseline
# ---------------------------------------------------------------------------
def _straight_poses(v: float, yaw: float, T: int = 100) -> torch.Tensor:
    t = torch.arange(T, dtype=torch.float32)
    return torch.stack([v * t * DT * math.cos(yaw),
                        v * t * DT * math.sin(yaw),
                        torch.full((T,), yaw),
                        torch.full((T,), v)], dim=-1)


def test_cv_baseline_zero_fde_on_constant_velocity():
    v = 5.0
    for yaw in (0.0, 0.7):          # heading-hold must work off-axis too
        poses = _straight_poses(v, yaw)
        g, valid = refb_labels.goal_tac_targets(poses, 10)
        assert bool(valid.all())
        cv = cv_goal_baseline(torch.tensor([v]))[0]         # [K, 4]
        fde = (cv[:, :2] - g[:, :2]).norm(dim=-1)
        assert float(fde.max()) < 1e-4                      # position: FDE 0
        assert float((cv[:, 2] - g[:, 2]).abs().max()) < 1e-5   # heading hold
        assert float((cv[:, 3] - g[:, 3]).abs().max()) < 1e-4   # speed hold


def test_cv_baseline_errs_on_a_turn():
    # circular arc at constant speed: straight extrapolation must miss.
    T, v, w = 100, 5.0, 0.3
    t = torch.arange(T, dtype=torch.float32) * DT
    R = v / w
    poses = torch.stack([R * torch.sin(w * t), R * (1 - torch.cos(w * t)),
                         w * t, torch.full((T,), v)], dim=-1)
    g, valid = refb_labels.goal_tac_targets(poses, 0)
    assert bool(valid.all())
    cv = cv_goal_baseline(torch.tensor([v]))[0]
    fde = (cv[:, :2] - g[:, :2]).norm(dim=-1)
    assert float(fde.min()) > 0.1


def test_cv_baseline_shapes_and_validation():
    out = cv_goal_baseline(torch.tensor([3.0, 0.0]))
    assert out.shape == (2, len(GOAL_TAUS), 4)
    assert torch.equal(out[1], torch.zeros(len(GOAL_TAUS), 4))  # v0=0 -> origin
    try:
        cv_goal_baseline(torch.zeros(2, 2))
        raise AssertionError("expected ValueError on non-[B] v0")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 1 Hz z_op window builder
# ---------------------------------------------------------------------------
def test_zop_window_indices_exact_cases():
    assert zop_window_indices(120, 4, 10) == [89, 99, 109, 119]  # true 1 Hz
    assert zop_window_indices(31, 4, 10) == [0, 10, 20, 30]      # exact fit
    # the v5f geometry: window=8 -> the documented degraded stride tail
    assert zop_window_indices(8, 4, 10) == [1, 3, 5, 7]
    assert zop_window_indices(2, 4, 10) == [0, 0, 0, 1]          # clamp-repeat
    assert zop_window_indices(1, 4, 10) == [0, 0, 0, 0]
    assert zop_window_indices(9, 1, 10) == [8]                   # single slot


def test_zop_window_indices_invariants():
    for n in range(1, 45):
        idx = zop_window_indices(n, 4, 10)
        assert len(idx) == 4
        assert idx[-1] == n - 1                      # newest is always 'now'
        assert all(0 <= i < n for i in idx)
        assert all(b >= a for a, b in zip(idx, idx[1:]))   # causal order
    for bad in ((0, 4, 10), (8, 0, 10), (8, 4, 0)):
        try:
            zop_window_indices(*bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# gate-JSON writer
# ---------------------------------------------------------------------------
def _mini(sel4=1.0, cv4=2.0, n4=10, macro=None):
    names = tuple(tau_name(t) for t in GOAL_TAUS)

    def row(x):
        return {k: x for k in names}

    return {"n_windows": 30,
            "grid": {"episodes": 40, "stride": 8, "batch": 16},
            "n_per_tau": {names[0]: 30, names[1]: n4, names[2]: 5},
            "goal_fde_m": {
                "selected": {names[0]: 0.5, names[1]: sel4, names[2]: 3.0},
                "cv_baseline": {names[0]: 0.6, names[1]: cv4, names[2]: 4.0},
                "fan_oracle": row(0.4), "fan_min": row(0.3)},
            "goal_heading_err_rad": {"selected": row(0.1)},
            "goal_speed_err_ms": {"selected": row(0.2)},
            "sel_gap_tac": {"selected_err": 1.0, "oracle_err": 0.8,
                            "gap": 0.2, "n": 30},
            "macro_f1": macro,
            "wallclock_s": 1.0}


def test_gate_schema_and_pass_logic():
    g = build_e44_gate(_mini(), aux_enabled=False)
    # gate 1: FDE@4s point-estimate comparison
    assert g["gate_goal_fde"]["pass"] is True
    assert g["gate_goal_fde"]["selected_fde_4s_m"] == 1.0
    assert g["gate_goal_fde"]["cv_fde_4s_m"] == 2.0
    assert "bootstrap" in g["gate_goal_fde"]["estimator"]      # CI is pod-side
    assert "POD-SIDE" in g["gate_goal_fde"]["estimator"]
    # gate 2: macro_f1 null when aux disabled (the E4.4 brief)
    assert g["gate_maneuver_f1"]["macro_f1_3axis"] is None
    assert g["gate_maneuver_f1"]["pass"] is None
    # gate 3: sel_gap baseline row, no threshold
    assert g["sel_gap_tac"]["gap"] == 0.2
    assert "no threshold" in g["sel_gap_tac"]["rule"]
    # four-families coverage statement + evidence class
    assert set(g["families"]) == {"TACTICAL", "LATERAL", "LONGITUDINAL",
                                  "STRATEGIC"}
    assert g["_evidence_class"].startswith("MEASURED")
    assert g["mini_eval"]["n_windows"] == 30


def test_gate_fail_and_not_judgeable():
    g_fail = build_e44_gate(_mini(sel4=3.0), aux_enabled=False)
    assert g_fail["gate_goal_fde"]["pass"] is False
    g_null = build_e44_gate(_mini(n4=0, sel4=None, cv4=None),
                            aux_enabled=False)
    assert g_null["gate_goal_fde"]["pass"] is None
    assert "n windows" in g_null["gate_goal_fde"]["reason_if_null"]


def test_gate_aux_enabled_reports_f1_but_does_not_judge():
    m = {"lat": 0.5, "lon": 0.4, "lane": 0.6, "mean3": 0.5}
    g = build_e44_gate(_mini(macro=m), aux_enabled=True)
    assert g["gate_maneuver_f1"]["macro_f1_3axis"] == m
    # the 5-way remapped reference is a pod-side rescore -> never judged here
    assert g["gate_maneuver_f1"]["reference_5way_remapped_f1"] is None
    assert g["gate_maneuver_f1"]["pass"] is None


# ---------------------------------------------------------------------------
# macro-F1
# ---------------------------------------------------------------------------
def test_macro_f1_from_confusion():
    perfect = torch.tensor([[5, 0], [0, 5]])
    assert abs(macro_f1_from_confusion(perfect) - 1.0) < 1e-9
    # class 1 has no support -> excluded; class 0: prec 1.0, rec 0.5 -> 2/3
    skew = torch.tensor([[5, 5], [0, 0]])
    assert abs(macro_f1_from_confusion(skew) - 2.0 / 3.0) < 1e-9
    assert math.isnan(macro_f1_from_confusion(torch.zeros(3, 3)))
    try:
        macro_f1_from_confusion(torch.zeros(2, 3))
        raise AssertionError("expected ValueError on non-square confusion")
    except ValueError:
        pass
