"""O7 / O8 — the EXTERNAL-TARGET terms (E-DEC-9 / E-DEC-10).

WHY THEY EXIST. Every pre-existing term has a SELF-GENERATED target: O5 predicts
our own next latent, O3 our own masked readout cells, O6 asks only for isotropy,
O1 only for per-action difference. The model therefore chooses BOTH what to
represent and what to predict, and "ego motion + noise" satisfies all of it with
ZERO scene content -- which is what was MEASURED: every arm sat BELOW a constant
predictor on agent count, while frozen DINOv3 read +0.2754 on data it had never
trained on (E-DEC-7).

An encoder trained on nothing but distillation into frozen DINOv3 went from
-1.0407 to +0.3274 on `n_agents` (t 12.63, 24/24 episodes) -- clearing zero AND
the raw-pixel floor for the first time in the programme -- while ego speed ROSE
+0.2830 -> +0.3940 (E-DEC-8). O7 brings that target into the objective; O8 is its
TEACHER-FREE counterpart, targeting raw pixels so the pipeline stays
self-contained.

⛔ The defaults are 0.0 and the heads are not constructed at all when off. That
is load-bearing: a live 30k run and every existing checkpoint must keep a
bit-identical loss, RNG stream and state_dict.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_v6_staged import O7Distill, O8Pixel, build_parser  # noqa: E402

GH, GW, N_CELLS, D_READ = 4, 8, 32, 64


def _o8() -> O8Pixel:
    torch.manual_seed(0)
    return O8Pixel(d_readout=D_READ, n_cells=N_CELLS, grid_hw=(GH, GW))


def test_both_terms_default_to_off():
    """⛔ Load-bearing: on by default would change every existing run."""
    a = build_parser().parse_args(["--stage", "S-W", "--out", "x"])
    assert float(a.w_o7_distill) == 0.0
    assert float(a.w_o8_pixel) == 0.0


def test_o8_target_shape_matches_the_cell_grid():
    o8 = _o8()
    t = o8.target(torch.rand(2, 3, 256, 640))
    assert t.shape == (2, N_CELLS, 3 * o8.ph * o8.pw)


def test_o8_target_is_NOT_a_per_cell_mean_colour():
    """The degeneracy this term must not have.

    A mean colour is trivially predictable and carries no layout, so the term
    could be satisfied without representing anything -- the same failure mode
    E-DEC-7 describes. Two images with the SAME per-cell mean but different
    layout must produce DIFFERENT targets.

    The split is HORIZONTAL at the half-cell mark, i.e. structure the 8x10
    per-cell patch can actually represent. See the companion test below for the
    scale at which it provably cannot.
    """
    o8 = _o8()
    flat = torch.full((1, 3, 256, 640), 0.5)
    split = flat.clone()
    cell_w = 640 // GW
    for c in range(GW):                      # dark left half / bright right half
        x0 = c * cell_w
        split[:, :, :, x0:x0 + cell_w // 2] = 0.0
        split[:, :, :, x0 + cell_w // 2:x0 + cell_w] = 1.0
    a, b = o8.target(flat), o8.target(split)
    assert torch.allclose(a.mean(-1), b.mean(-1), atol=1e-5), (
        "the two test images were supposed to share a per-cell MEAN")
    assert not torch.allclose(a, b), (
        "O8's target collapsed to a per-cell mean colour — it would be "
        "satisfiable without representing layout, which is the whole defect "
        "these external targets exist to remove")


def test_o8_target_IS_LOW_PASS_and_that_is_a_real_limitation():
    """⚠️ Documented, not hidden: O8 averages away structure finer than its patch.

    `adaptive_avg_pool2d` to gh*ph x gw*pw means each target value averages an
    (H/(gh*ph)) x (W/(gw*pw)) = 8 x 8 pixel block. Alternating single-pixel rows
    therefore land on EXACTLY the per-cell mean -- caught by an earlier version
    of the test above, which is how this limitation was found.

    ⇒ O8 cannot supervise detail below ~8 px, which is roughly a distant vehicle.
    That is the concrete respect in which the teacher-free target is WEAKER than
    O7's DINOv3 features, and it is the thing to look at first if E-DEC-10's arm
    underperforms E-DEC-9's.
    """
    o8 = _o8()
    flat = torch.full((1, 3, 256, 640), 0.5)
    striped = flat.clone()
    striped[:, :, ::2, :] = 0.0
    striped[:, :, 1::2, :] = 1.0
    assert torch.allclose(o8.target(flat), o8.target(striped), atol=1e-6), (
        "sub-8px structure is expected to average out; if this now differs the "
        "pooling changed and the limitation note above must be re-derived")


def test_o8_gradient_reaches_the_readout_input():
    o8 = _o8()
    z = torch.randn(2, N_CELLS * D_READ, requires_grad=True)
    loss = o8(z, torch.rand(2, 3, 256, 640))
    loss.backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0, (
        "O8 produced no gradient into the latent — the term would be inert")


def test_o8_target_carries_NO_gradient():
    """A target that is differentiable is not a target — the model could move it."""
    o8 = _o8()
    t = o8.target(torch.rand(2, 3, 256, 640))
    assert not t.requires_grad, (
        "O8's target must be detached; a target the model can influence "
        "re-opens exactly the self-generated-target degeneracy (E-DEC-7)")


def test_o7_head_is_registered_but_the_teacher_is_NOT():
    """The frozen teacher must never enter our checkpoint or our optimiser."""
    o7 = O7Distill(d_readout=D_READ, n_cells=N_CELLS, grid_hw=(GH, GW))
    keys = list(o7.state_dict())
    assert any(k.startswith("head.") for k in keys), "the trainable head is missing"
    assert not any("teacher" in k for k in keys), (
        "the frozen teacher leaked into the state_dict — it would be saved into "
        "every checkpoint and, if it ever required grad, optimised")
    assert all(p.requires_grad for p in o7.head.parameters())


def test_o7_and_o8_present_the_same_interface():
    """They are a MATCHED PAIR (E-DEC-10): same shapes, only the target differs."""
    o7 = O7Distill(d_readout=D_READ, n_cells=N_CELLS, grid_hw=(GH, GW))
    o8 = _o8()
    for m in (o7, o8):
        assert m.n_cells == N_CELLS and tuple(m.grid_hw) == (GH, GW)
        assert hasattr(m, "target") and hasattr(m, "head")


@pytest.mark.parametrize("flag,attr", [("--w-o7-distill", "w_o7_distill"),
                                       ("--w-o8-pixel", "w_o8_pixel")])
def test_each_weight_is_settable_from_the_cli(flag, attr):
    a = build_parser().parse_args(["--stage", "S-W", "--out", "x", flag, "0.25"])
    assert float(getattr(a, attr)) == 0.25
