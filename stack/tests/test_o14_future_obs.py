"""O14 — the future-observation auxiliary (R2, PI-approved; PREREG_O14_FUTURE_OBS).

⛔ THE TWO LOAD-BEARING TESTS:
  * ``test_the_default_is_bit_identical_to_the_incumbent`` — w_o14 absent/0 must
    construct NOTHING: same parameter names, same count. Every arm comparison in
    the prereg rests on this.
  * ``test_the_gradient_reaches_the_encoder`` — INERT is a pre-registered outcome
    whose first audit line is "did the aux ever touch the encoder"; this pins the
    path structurally so an INERT read means the OBJECTIVE failed, not the wiring.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from train_v6_staged import (  # noqa: E402
    O14_PIX_H, O14_PIX_W, _o14_pixel_target, build_stack_from_args,
)


def _args(**kw):
    """Args from the trainer's OWN parser — defaults stay authoritative, so a
    new required field never silently diverges from what launches actually get
    (three rounds of hand-guessed namespaces died on hidden fields before this)."""
    from train_v6_staged import build_parser
    argv = ["--out", "UNUSED-o14-test", "--stage", "S-W",
            "--frame-h", "64", "--frame-w", "160", "--enc-dim", "32",
            "--enc-depth", "1", "--enc-heads", "2", "--readout-grid", "2",
            "--readout-grid-w", "4", "--readout-dim", "16", "--pred-dim", "32",
            "--pred-depth", "1", "--pred-heads", "2", "--window", "4",
            "--d-tac", "16", "--d-str", "8"]
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, str(v)]
    return build_parser().parse_args(argv)


def _frames(b=3, w=4, c=9, h=64, ww=160, k=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(b, w, c, h, ww, generator=g),
            torch.rand(b, k, c, h, ww, generator=g))


# --------------------------------------------------------------------------- #
# bit-identity                                                                  #
# --------------------------------------------------------------------------- #
def test_the_default_is_bit_identical_to_the_incumbent():
    """No --w-o14 ⇒ no module, no state_dict key, no parameter — at all."""
    plain = build_stack_from_args(_args())
    zero = build_stack_from_args(_args(w_o14=0.0))
    on = build_stack_from_args(_args(w_o14=1.0))
    p_plain = [n for n, _ in plain.named_parameters()]
    p_zero = [n for n, _ in zero.named_parameters()]
    p_on = [n for n, _ in on.named_parameters()]
    assert p_plain == p_zero
    assert not any("o14" in n for n in p_plain)
    assert any(n.startswith("o14_head.") for n in p_on)


def test_the_head_belongs_to_aux_so_S_W_trains_it():
    """`group_of` must map every o14 parameter to "aux" — the group whose X3
    contract permits encoder backprop for label-free (data-target) losses."""
    on = build_stack_from_args(_args(w_o14=1.0))
    for n, _ in on.named_parameters():
        if n.startswith("o14_head."):
            assert on.group_of(n) == "aux"


# --------------------------------------------------------------------------- #
# the target builder                                                            #
# --------------------------------------------------------------------------- #
def test_fut_and_rec_pick_different_frames_and_diff_is_their_difference():
    fr, ff = _frames()
    fut = _o14_pixel_target(fr, ff, mode="fut", k=4)
    rec = _o14_pixel_target(fr, ff, mode="rec", k=4)
    dif = _o14_pixel_target(fr, ff, mode="fut_diff", k=4)
    assert fut.shape == rec.shape == (3, O14_PIX_H * O14_PIX_W)
    assert not torch.allclose(fut, rec)
    assert torch.allclose(dif, fut - rec, atol=1e-6)


def test_k_selects_the_kth_future_frame():
    fr, ff = _frames()
    t1 = _o14_pixel_target(fr, ff, mode="fut", k=1)
    t4 = _o14_pixel_target(fr, ff, mode="fut", k=4)
    assert not torch.allclose(t1, t4)


def test_k_beyond_the_horizon_refuses_loudly():
    fr, ff = _frames(k=3)
    with pytest.raises(ValueError, match="o14_k"):
        _o14_pixel_target(fr, ff, mode="fut", k=4)


def test_newest_frame_only_layout_is_also_correct():
    """C=3 frames: the [-3:] slice must be the whole (only) frame."""
    fr, ff = _frames(c=3)
    t = _o14_pixel_target(fr, ff, mode="fut", k=2)
    assert t.shape == (3, O14_PIX_H * O14_PIX_W)


def test_the_shuffle_is_a_derangement_not_a_randperm():
    """Cyclic shift: NO row keeps its own target (the o11 lesson — randperm
    fixes points with p~1/B and silently un-shuffles that row)."""
    fr, ff = _frames(b=5)
    plain = _o14_pixel_target(fr, ff, mode="fut", k=2)
    shuf = _o14_pixel_target(fr, ff, mode="fut", k=2, shuffle=True)
    for i in range(5):
        assert not torch.allclose(plain[i], shuf[i]), f"row {i} kept its target"
    assert torch.allclose(shuf[1], plain[0])          # roll(+1) exactly


# --------------------------------------------------------------------------- #
# the gradient path                                                             #
# --------------------------------------------------------------------------- #
def test_the_gradient_reaches_the_encoder():
    """The INERT audit line, pinned structurally: L1(o14_head([z_t, a3]), tgt)
    must produce a NONZERO gradient on an encoder parameter."""
    stack = build_stack_from_args(_args(w_o14=1.0))
    fr, ff = _frames(b=2)
    flat = fr.reshape(2 * 4, *fr.shape[2:])
    states = stack.readout(stack.encoder(flat)).reshape(2, 4, -1)
    a3 = torch.zeros(2, 3)
    pred = stack.o14_head(torch.cat([states[:, -1], a3], dim=-1))
    tgt = _o14_pixel_target(fr, ff, mode="fut", k=4)
    torch.nn.functional.l1_loss(pred, tgt).backward()
    enc_grads = [p.grad for n, p in stack.named_parameters()
                 if n.startswith("encoder.") and p.grad is not None]
    assert enc_grads, "no encoder parameter received a gradient"
    assert any(float(g.abs().max()) > 0 for g in enc_grads)


def test_the_target_carries_no_gradient():
    """The target is DATA. At the loss site it arrives as a leaf batch tensor;
    the builder must not attach it to any graph when its inputs are leaves —
    otherwise the loss could optimise the pathway of its own supervision."""
    fr, ff = _frames()
    assert not fr.requires_grad and not ff.requires_grad     # the real case
    t = _o14_pixel_target(fr, ff, mode="fut", k=2)
    assert t.requires_grad is False and t.grad_fn is None
