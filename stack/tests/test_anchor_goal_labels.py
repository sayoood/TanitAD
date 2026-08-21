"""Tests for the `ANCHOR_GOAL` label deriver.

Every property here is one whose failure would emit a *label* rather than an
error — the only failure mode that reaches a trained model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.data import anchor_goal as AG                          # noqa: E402


def _vocab(k=8, horizons=(5, 10, 15, 20), seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(k, len(horizons), 2, generator=g) * 8.0, list(horizons)


# --------------------------------------------------------------------------- #
# the contract with the goal vocabulary                                        #
# --------------------------------------------------------------------------- #
def test_arg_slots_match_the_v6_vocabulary():
    """Mirrored, not imported — so a vocabulary change fires HERE instead of
    drifting silently into a label file nobody re-reads."""
    from tanitad.models.v6 import GOAL_ARG_SLOTS, TACTICAL_GOAL_TOKENS, TAC_BAND_S
    assert AG.ARG_SLOTS == GOAL_ARG_SLOTS
    assert AG.TOKEN in TACTICAL_GOAL_TOKENS
    assert AG.TAC_BAND_S == tuple(float(x) for x in TAC_BAND_S)


def test_arg_layout_is_the_vocabularys_declared_order():
    assert AG.ARG_LAYOUT == {"anchor_id": 0, "t_reach_s": 1}


# --------------------------------------------------------------------------- #
# the two refusals — both are FINDINGS, and must stay loud                     #
# --------------------------------------------------------------------------- #
def test_refuses_a_2s_vocabulary_as_a_tactical_label():
    """⛔ No shipped vocabulary reaches the tactical band. A 2 s point emitted as
    an `ANCHOR_GOAL` would be an operative-band point wearing a tactical name."""
    A, H = _vocab()
    with pytest.raises(ValueError, match="outside the tactical band"):
        AG.anchor_goal_labels(torch.zeros(3, 2), torch.ones(3, dtype=torch.bool),
                              A, H, step=20)


def test_the_operative_band_override_stamps_itself():
    A, H = _vocab()
    out = AG.anchor_goal_labels(torch.zeros(3, 2), torch.ones(3, dtype=torch.bool),
                                A, H, step=20, allow_operative_band=True)
    assert out["provenance"]["off_band"] is True
    assert "NOT a tactical goal label" in out["provenance"]["off_band_stamp"]


def test_a_60_step_label_is_accepted_when_the_vocabulary_reaches_it():
    A, H = _vocab(horizons=(20, 40, 60))
    out = AG.anchor_goal_labels(torch.zeros(3, 2), torch.ones(3, dtype=torch.bool),
                                A, H, step=60)
    assert out["t_reach_s"] == pytest.approx(6.0)
    assert out["provenance"]["off_band"] is False
    assert out["provenance"]["off_band_stamp"] is None


def test_anchor_endpoints_refuses_a_horizon_it_does_not_have():
    A, H = _vocab()
    with pytest.raises(ValueError, match="does NOT contain step 60"):
        AG.anchor_endpoints(A, H, step=60)


def test_anchor_endpoints_reads_the_named_horizon_not_the_last():
    A = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
    assert torch.equal(AG.anchor_endpoints(A, [20, 40, 60], step=20),
                       torch.tensor([[1.0, 0.0]]))


# --------------------------------------------------------------------------- #
# ⛔ the categorical-arg gap must stay VISIBLE, not be papered over             #
# --------------------------------------------------------------------------- #
def test_anchor_id_is_never_written_into_a_physical_units_slot():
    A, H = _vocab(horizons=(20, 40, 60))
    out = AG.anchor_goal_labels(torch.randn(6, 2) * 5, torch.ones(6, dtype=torch.bool),
                                A, H, step=60)
    a0 = AG.ARG_LAYOUT["anchor_id"]
    assert torch.isnan(out["args"][:, a0]).all()
    assert float(out["arg_mask"][:, a0].sum()) == 0.0
    assert out["anchor_id"].dtype == torch.long and (out["anchor_id"] >= 0).all()
    assert "type error" in out["provenance"]["arg0_is_unset_because"]


def test_t_reach_slot_is_set_and_masked_only_where_valid():
    A, H = _vocab(horizons=(20, 40, 60))
    valid = torch.tensor([True, False, True, True])
    out = AG.anchor_goal_labels(torch.randn(4, 2), valid, A, H, step=60)
    t = AG.ARG_LAYOUT["t_reach_s"]
    assert torch.equal(out["arg_mask"][:, t] > 0, valid)


# --------------------------------------------------------------------------- #
# validity: excluded with a reason, never imputed                              #
# --------------------------------------------------------------------------- #
def test_invalid_rows_get_id_minus_one_and_nan_residual_not_a_plausible_value():
    A, H = _vocab(horizons=(20, 40, 60))
    e = torch.tensor([[10.0, 1.0], [float("nan"), 0.0], [3.0, 0.0]])
    out = AG.anchor_goal_labels(e, torch.tensor([True, True, False]), A, H, step=60)
    assert out["n_valid"] == 1 and out["n"] == 3
    assert int(out["anchor_id"][0]) >= 0
    assert int(out["anchor_id"][1]) == -1 and int(out["anchor_id"][2]) == -1
    assert torch.isnan(out["residual"][1]).all()
    assert torch.isnan(out["residual"][2]).all()
    assert "NEVER imputed" in out["invalid_reason"]


def test_all_invalid_yields_no_sigma_rather_than_a_zero():
    A, H = _vocab(horizons=(20, 40, 60))
    out = AG.anchor_goal_labels(torch.zeros(3, 2), torch.zeros(3, dtype=torch.bool),
                                A, H, step=60)
    assert out["n_valid"] == 0 and out["sigma_perax_m"] is None


# --------------------------------------------------------------------------- #
# the assignment itself                                                        #
# --------------------------------------------------------------------------- #
def test_assignment_picks_the_nearest_and_the_runner_up_is_never_closer():
    A = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    a = AG.assign_anchor(torch.tensor([[9.0, 0.0], [1.0, 0.0]]), A)
    assert a["ids"].tolist() == [1, 0]
    d0 = (a["residual"] ** 2).sum(-1)
    d1 = (a["residual_second"] ** 2).sum(-1)
    assert bool((d1 >= d0 - 1e-12).all())


def test_sigma_is_the_per_axis_form_the_thresholds_are_stated_in():
    """σ_perax = sqrt(mean(|e|²)/2) — the unit §3.1's requirement curve uses, NOT
    the radial RMS, which is √2 larger and would silently miss the bar."""
    A = torch.tensor([[[0.0, 0.0]], [[100.0, 100.0]]])
    e = torch.tensor([[3.0, 4.0], [-3.0, -4.0]])
    out = AG.anchor_goal_labels(e, torch.ones(2, dtype=torch.bool), A, [60], step=60)
    assert out["sigma_perax_m"] == pytest.approx((25.0 / 2.0) ** 0.5)


def test_frame_convention_is_stated_in_the_provenance():
    A, H = _vocab(horizons=(20, 40, 60))
    out = AG.anchor_goal_labels(torch.zeros(2, 2), torch.ones(2, dtype=torch.bool),
                                A, H, step=60)
    assert "x forward, y left" in out["provenance"]["frame"]


def test_endpoint_contract_matches_driving_diagnostic_gt_ego_waypoints():
    """The documented input contract, pinned against the programme's own
    producer — so "ego-frame displacement" means one thing here and there."""
    import driving_diagnostic as dd
    g = torch.Generator().manual_seed(3)
    T = 90
    poses = torch.zeros(T, 4)
    poses[:, 0] = torch.cumsum(torch.rand(T, generator=g) + 1.0, 0)
    poses[:, 1] = torch.cumsum(torch.randn(T, generator=g) * 0.05, 0)
    poses[:, 2] = torch.cumsum(torch.randn(T, generator=g) * 0.01, 0)
    last = torch.tensor([10, 20])
    gt = dd.gt_ego_waypoints(poses, last, wp_steps=[60])[:, 0]      # [2, 2]
    A = gt.reshape(2, 1, 2).double()
    out = AG.anchor_goal_labels(gt, torch.ones(2, dtype=torch.bool), A, [60], step=60)
    assert out["sigma_perax_m"] == pytest.approx(0.0, abs=1e-9)


def _imported_module_names(path) -> set[str]:
    """Every module name this file IMPORTS, from its AST — including
    function-local and lazily-deferred imports (``ast.walk`` sees the whole tree).

    ⛔ WHY AN AST AND NOT A SUBSTRING SCAN. Both modules under test DOCUMENT the
    rule in prose — their docstrings contain the literal text
    ``tanitad.data.situations`` in the sentence saying they never read it, and
    ``e_ag1_anchor_floor.run()`` emits it in a provenance string. A grep-style
    guard would match its OWN DOCUMENTATION and fire on a clean module. An AST
    walk reads only real `import` statements, so it cannot.
    """
    import ast
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.add(base)
            names.update(f"{base}.{a.name}" if base else a.name
                         for a in node.names)
    return names


def _reaches_situations(names) -> list[str]:
    return sorted(n for n in names
                  if n == "situations" or n.endswith(".situations")
                  or ".situations." in n)


# --------------------------------------------------------------------------- #
# ⛔ the binding disjointness rule                                              #
# --------------------------------------------------------------------------- #
def test_module_has_no_situation_classifier_path():
    src = Path(AG.__file__).read_text(encoding="utf-8")
    for token in ("detect_lane_change", "detect_intersection", "detect_roundabout",
                  "situations_from_poses"):
        assert token not in src, f"{token} reached the ANCHOR_GOAL label deriver"
    assert _reaches_situations(_imported_module_names(AG.__file__)) == []
    # ⚠️ TRANSITIVELY, in a SUBPROCESS. An in-process `sys.modules` check reports
    # the SESSION's imports — it passes alone and fails in the full suite, which
    # is a check on the wrong thing.
    import subprocess
    root = str(Path(AG.__file__).resolve().parents[3])
    code = ("import sys; sys.path.insert(0, r'%s'); "
            "from tanitad.data import anchor_goal; "
            "print('LEAK' if 'tanitad.data.situations' in sys.modules else 'CLEAN')"
            % root)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "CLEAN" in r.stdout, r.stdout + r.stderr[-2000:]
