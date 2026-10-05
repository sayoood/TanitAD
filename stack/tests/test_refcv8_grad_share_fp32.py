"""refcv8 WP-B: the gradient-share instrument's fp32 REPLAY under a bf16 trunk (2026-10-05).

MEASURED on the Thor L3 smoke (`refcv8-wpb-smoke-L3`, `--trunk-bf16`): `gs_trunk_lin_rel_err` = 5.9e-3 against the
1e-4 bar, while the fp32 groups read 9.3e-5 and 6.0e-5. A bf16 backward cannot meet the linearity control, so the
statistic is now read on an fp32 replay of the SAME step. Pinned here:
* the replay meets the bar on a bf16 net whose direct reading misses it. The RED arm is the direct reading;
* everything the replay could disturb is restored: the global RNG, the dedicated generators, the buffers and the
  levers. The training forward that follows is BIT-identical to one taken without the replay;
* `.grad` is untouched;
* the trainer takes the replay BEFORE its training forward whenever a bf16 lever is on.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _refcv8_rig as R  # noqa: E402
from tanitad.refs import refcv8_conditioning as r8c  # noqa: E402
from tanitad.train import grad_share as GS  # noqa: E402


class _Trunk(torch.nn.Module):
    """A conv trunk with the timm trunk's lever contract: ``memory_levers['bf16']`` -> bf16 autocast, fp32 out."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.BatchNorm2d(16),
                                       torch.nn.ReLU(), torch.nn.Conv2d(16, 16, 3, padding=1), torch.nn.ReLU(),
                                       torch.nn.Conv2d(16, 8, 3, padding=1))
        self.memory_levers = {"bf16": True}

    def forward(self, x):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=bool(self.memory_levers["bf16"])):
            y = self.net(x)
        return y.float()


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.encoder = _Trunk()
        self.h1 = torch.nn.Linear(8, 4)
        self.h2 = torch.nn.Linear(8, 4)
        self.gen = r8c.R8Generator(5)

    def losses(self, x):
        f = self.encoder(x).mean((2, 3))
        noise = torch.randn(f.shape) * 0.1 + self.gen.randn(f.shape, "cpu") * 0.1      # global + dedicated draws
        a = (self.h1(f + noise) ** 2).mean()
        b = (self.h2(f) - 1.0).abs().mean()
        return {"traj": a, "agent": b, "loss": a + 0.7 * b}


def _setup():
    net = _Net().train()
    x = torch.randn(4, 3, 32, 32, generator=torch.Generator().manual_seed(3))
    groups = {"trunk": list(net.encoder.parameters())}
    weights = {"traj": 1.0, "agent": 0.7}
    return net, x, groups, weights


def test_RED_ARM_a_bf16_reading_misses_the_linearity_bar_and_the_replay_meets_it():
    net, x, groups, weights = _setup()
    torch.manual_seed(11)
    lo = net.losses(x)
    direct = GS.measure(GS.term_tensors(lo, weights), lo["loss"], groups)
    rep = GS.fp32_replay(net, lambda: net.losses(x), groups, weights)
    assert direct["gs_trunk_lin_rel_err"] > 1e-4                 # the Thor failure, reproduced on CPU bf16
    assert rep["gs_trunk_lin_rel_err"] <= 1e-4
    assert rep["gs_fp32_replay"] == 1.0 and rep["gs_fp32_replay_levers"] == 1.0
    assert sum(v for k, v in rep.items() if k.startswith("gs_trunk_proj_")) == pytest.approx(1.0, abs=1e-6)


def test_the_replay_leaves_the_training_step_bit_identical():
    net, x, groups, weights = _setup()
    sd0 = {k: v.clone() for k, v in net.state_dict().items()}
    torch.manual_seed(21)
    ref = net.losses(x)["loss"]
    ref.backward()
    g_ref = [p.grad.clone() for p in net.parameters()]
    net.load_state_dict(sd0)
    net.zero_grad(set_to_none=True)
    net.gen = r8c.R8Generator(5)
    torch.manual_seed(21)
    GS.fp32_replay(net, lambda: net.losses(x), groups, weights)
    assert all(p.grad is None for p in net.parameters())         # .grad never touched
    assert net.encoder.memory_levers["bf16"] is True              # lever restored
    got = net.losses(x)["loss"]
    got.backward()
    assert torch.equal(got, ref)
    assert all(torch.equal(p.grad, q) for p, q in zip(net.parameters(), g_ref))


def test_buffers_are_restored_after_the_replay():
    net, x, groups, weights = _setup()
    before = {n: b.clone() for n, b in net.named_buffers()}
    GS.fp32_replay(net, lambda: net.losses(x), groups, weights)
    assert all(torch.equal(b, before[n]) for n, b in net.named_buffers())


def test_the_real_timm_trunk_exposes_the_lever_the_replay_switches():
    T = R.trainer()
    _cfg, m = R.build(T, refcv8=False)
    trunks = [mm for mm in m.modules() if isinstance(getattr(mm, "memory_levers", None), dict)]
    assert trunks, "the rig's timm trunk carries no memory_levers dict -- the replay would switch nothing"
    trunks[0].memory_levers["bf16"] = True
    assert GS.bf16_levers(m) == [trunks[0]]


def test_the_trainer_replays_before_its_training_forward():
    T = R.trainer()
    src = inspect.getsource(T.train)
    i_rep = src.index("_gshare.fp32_replay(")
    i_fwd = src.index("losses = compute_losses_v3(model, batch, device, mode=args.mode,\n")
    assert i_rep < i_fwd and "if _gs_due and _gshare.bf16_levers(model):" in src


def test_the_replay_runs_with_TF32_OFF_and_restores_every_switch():
    """L5 (MEASURED on Thor, L4 G-SMOKE: trunk linearity 4.2e-4 with bf16 off but cuDNN TF32 on). Inside the replay both
    TF32 switches are off and matmul precision is 'highest'; afterwards every switch is back to its configured value."""
    net, x, groups, weights = _setup()
    seen = {}

    def run():
        seen.update(mm=torch.backends.cuda.matmul.allow_tf32, cd=torch.backends.cudnn.allow_tf32,
                    prec=torch.get_float32_matmul_precision())
        return net.losses(x)
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
              torch.get_float32_matmul_precision())
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        row = GS.fp32_replay(net, run, groups, weights)
        assert seen == {"mm": False, "cd": False, "prec": "highest"}
        assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.get_float32_matmul_precision()) == (True, True, "high")
        assert row["gs_fp32_replay_tf32_off"] == 1.0
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before[0], before[1]
        torch.set_float32_matmul_precision(before[2])
