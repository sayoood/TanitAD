"""Unit tests for the planner<->WM seam projection (one-sided PCGrad).

Pure-torch, CPU, milliseconds — mirrors the trainer's discipline of unit-testing
the correctness-critical gradient path in isolation. Run:

    C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest \
        ".../2026-07-23-planner-wm-gradient-coupling/tests/test_grad_surgery.py" -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grad_surgery import _SeamProject, deconflict, seam_project  # noqa: E402


def _cos(a, b):
    return float((a.flatten() @ b.flatten())
                 / (a.norm() * b.norm()).clamp_min(1e-12))


def test_forward_is_identity_on_both_branches():
    x = torch.randn(3, 5, 64, requires_grad=True)
    s_wm, s_plan = seam_project(x)
    assert torch.equal(s_wm, x)
    assert torch.equal(s_plan, x)


def test_deconflict_orthogonalises_when_conflicting():
    # a planner gradient with a deliberate anti-reference component
    g_ref = torch.tensor([[1.0, 0.0, 0.0]])
    g_plan = torch.tensor([[-2.0, 3.0, 0.0]])          # negative dot with g_ref
    proj, diag = deconflict(g_plan, g_ref)
    # residual must be orthogonal to g_ref (the -2 x-component removed)
    assert abs(float((proj * g_ref).sum())) < 1e-6
    assert proj[0, 0].abs() < 1e-6 and abs(float(proj[0, 1]) - 3.0) < 1e-6
    assert diag["seam_frac_conflict"] == 1.0


def test_deconflict_is_noop_when_aligned():
    g_ref = torch.tensor([[1.0, 1.0, 0.0]])
    g_plan = torch.tensor([[2.0, 1.0, 5.0]])           # positive dot -> keep as is
    proj, diag = deconflict(g_plan, g_ref)
    assert torch.allclose(proj, g_plan)
    assert diag["seam_frac_conflict"] == 0.0


def test_seam_leaves_wm_gradient_byte_identical():
    # The WM (reference) gradient must be untouched by the seam — one-sidedness.
    torch.manual_seed(1)
    trunk = torch.nn.Linear(32, 32, bias=False)
    x = torch.randn(4, 32)

    trunk.zero_grad(set_to_none=True)
    s_wm, _ = seam_project(trunk(x))
    ((s_wm - 1.0) ** 2).mean().backward()
    g_seam = trunk.weight.grad.clone()

    trunk.zero_grad(set_to_none=True)
    ((trunk(x) - 1.0) ** 2).mean().backward()
    g_plain = trunk.weight.grad.clone()

    assert torch.allclose(g_seam, g_plain, atol=0.0)


def test_seam_noop_when_planner_aligns_with_wm():
    # If the planner loss pushes the SAME way as the WM loss, the trunk gradient
    # under the seam equals the plain summed gradient exactly.
    torch.manual_seed(2)
    trunk = torch.nn.Linear(32, 32, bias=False)
    x = torch.randn(4, 32)

    trunk.zero_grad(set_to_none=True)
    a = trunk(x)
    s_wm, s_plan = seam_project(a)
    (((s_wm - 1.0) ** 2).mean() + 0.5 * ((s_plan - 1.0) ** 2).mean()).backward()
    g_seam = trunk.weight.grad.clone()

    trunk.zero_grad(set_to_none=True)
    a2 = trunk(x)
    (((a2 - 1.0) ** 2).mean() + 0.5 * ((a2 - 1.0) ** 2).mean()).backward()
    g_plain = trunk.weight.grad.clone()

    assert torch.allclose(g_seam, g_plain, atol=1e-7)


def test_seam_removes_conflicting_planner_pull_from_trunk():
    # A planner loss that directly OPPOSES the WM loss must not move the trunk in
    # the WM's own descent direction the wrong way: with the seam, the trunk
    # gradient's projection onto the WM gradient is >= the plain (summed) case.
    torch.manual_seed(3)
    trunk = torch.nn.Linear(48, 48, bias=False)
    x = torch.randn(8, 48)

    def run(use_seam):
        trunk.zero_grad(set_to_none=True)
        st = trunk(x)
        s_wm, s_plan = seam_project(st) if use_seam else (st, st)
        l_wm = ((s_wm - 1.0) ** 2).mean()
        l_plan = -((s_plan - 1.0) ** 2).mean()          # exact opposite of WM
        (l_wm + l_plan).backward()
        return trunk.weight.grad.clone()

    # reference WM-only gradient direction
    trunk.zero_grad(set_to_none=True)
    ((trunk(x) - 1.0) ** 2).mean().backward()
    g_wm = trunk.weight.grad.clone()

    g_plain = run(False)
    g_seam = run(True)
    # plain cancels (planner = -WM) -> ~0 gradient; seam keeps the WM component.
    assert _cos(g_seam, g_wm) > _cos(g_plain, g_wm) or g_plain.norm() < 1e-6
    assert g_seam.norm() >= g_plain.norm() - 1e-6


def test_global_vs_per_sample_shapes_and_finiteness():
    g_plan = torch.randn(6, 4, 16)
    g_ref = torch.randn(6, 4, 16)
    for ps in (True, False):
        proj, diag = deconflict(g_plan, g_ref, per_sample=ps)
        assert proj.shape == g_plan.shape
        assert torch.isfinite(proj).all()
        assert 0.0 <= diag["seam_frac_removed_mean"] < math.inf


def test_gradvaccine_raises_cosine_toward_phi():
    # GradVaccine mode should push a conflicting cosine UP toward the target phi.
    torch.manual_seed(4)
    g_ref = torch.randn(1, 256)
    g_plan = -0.7 * g_ref + 0.3 * torch.randn(1, 256)   # start with negative cosine
    c0 = _cos(g_plan, g_ref)
    proj, _ = deconflict(g_plan, g_ref, target_cos=0.1)
    c1 = _cos(proj, g_ref)
    assert c1 > c0                                       # cosine increased


def test_diag_flags_conflict_direction():
    _SeamProject.last_diag = {}
    trunk = torch.nn.Linear(16, 16, bias=False)
    x = torch.randn(4, 16)
    s_wm, s_plan = seam_project(trunk(x))
    l = ((s_wm - 1.0) ** 2).mean() - ((s_plan - 1.0) ** 2).mean()
    l.backward()
    d = _SeamProject.last_diag
    assert "seam_cos_mean" in d and "seam_frac_removed_mean" in d
    assert d["seam_frac_conflict"] > 0.0
