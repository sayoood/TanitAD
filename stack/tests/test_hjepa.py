"""Tests for ``tanitad.models.hjepa`` -- the H-JEPA module (arXiv 2610.06805), CPU-only and fast.

What each group pins (and why it is not decoration):

* SIGReg: sits at its analytic null floor on Gaussian data, grows on collapsed data (linearly in
  n, exactly), and is EXACTLY invariant to data-parallel sharding via the ECF all-reduce
  (simulated in-process, plus a real 2-process gloo run that also pins the gradient scaling).
* Prop. 1 / Prop. 3: a background-only encoder + identity predictor reaches L_pred = 0 while
  SIGReg reads its floor (so the marginal regulariser cannot see the collapse), and the IDM term
  then costs ~1 on unit-variance actions (with a non-collapsed control that must read ~0).
* ``action_sensitivity``: three controls that must read their known values (self-swap 0.0
  exactly, identity predictor 0.0, a linear-action stub > 0) plus channel isolation.
* Reachability (the CLAUDE.md "built, tested, unreachable from its caller" class): after one
  backward through ``loss()`` EVERY parameter of EVERY level is live, a deliberately unused
  control parameter reads exactly 0.0, and the mutation gamma = 0 makes exactly the IDM heads
  dead -- while the positive control (gamma > 0) makes none dead.
* The planner reduces the goal cost on a toy linear system with a known optimum, respects
  bounds, and refines a seed plan.
"""

from __future__ import annotations

import math
import os
import pathlib
import tempfile

import pytest
import torch
from torch import nn

from tanitad.models.hjepa import (
    CausalPredictor, HJEPAStack, IDMHead, LevelConfig, MacroActionEncoder, SIGReg,
    dead_parameters, donor_indices, monotonicity, param_grad_report, rho_within_episode,
)

pytestmark = pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad:UserWarning")


@pytest.fixture(scope="module", autouse=True)
def _few_threads():
    """Tiny tensors: many BLAS threads only add contention (and this box is shared). Restored."""
    old = torch.get_num_threads()
    torch.set_num_threads(max(1, min(2, old)))
    yield
    torch.set_num_threads(old)


# ---------------------------------------------------------------------------------------------
# small helpers / stubs
# ---------------------------------------------------------------------------------------------

def _tiny_cfgs(d1=8, a=2, d2=6, m=3, s=2, w=1, ctx=3, **kw):
    l1 = LevelConfig(latent_dim=d1, action_dim=a, pred_width=16, pred_depth=1, pred_heads=2,
                     context=ctx, idm_hidden=16, **kw)
    l2 = LevelConfig(latent_dim=d2, action_dim=m, stride=s, window=w, enc_hidden=16,
                     macro_hidden=8, pred_width=16, pred_depth=1, pred_heads=2, context=ctx,
                     idm_hidden=16, **kw)
    return [l1, l2]


class IdentityPredictor(nn.Module):
    context = 3

    def forward(self, z, a):
        return z


class LinearActionPredictor(nn.Module):
    """z_{t+1} = z_t + a_t W -- 'a predictor stub that adds the action embedding linearly'."""
    context = 3

    def __init__(self, e, d, zero_rows=(), seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(e, d, generator=g) * 0.4
        for r in zero_rows:
            w[r] = 0.0
        self.W = nn.Parameter(w)

    def forward(self, z, a):
        return z + a @ self.W


class _SumPool(nn.Module):
    stride = 2

    def forward(self, e):               # (..., s, E) -> (..., E)
        return e.sum(dim=-2)


class _SumWindow(nn.Module):
    """Encoder stub for window 2: z_t = (first state) + (second state) of the window."""
    window = 2

    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, x):
        return x[..., :self.d] + x[..., self.d:]


def _stack(**kw):
    torch.manual_seed(0)
    return HJEPAStack(_tiny_cfgs(**kw), sigreg_projections=32)


def _batch(b=8, t1=7, d1=8, a=2, seed=1, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(b, t1, d1, generator=g)
    act = torch.randn(b, t1 - 1, a, generator=g)
    return z.requires_grad_(requires_grad), act


# ---------------------------------------------------------------------------------------------
# SIGReg
# ---------------------------------------------------------------------------------------------

def test_sigreg_quadrature_is_the_stated_one():
    sg = SIGReg(8)
    assert sg.t.numel() == 17 and float(sg.t[0]) == 0.0 and abs(float(sg.t[-1]) - 3.0) < 1e-12
    # window-free weights integrate a constant over [-3, 3] (even integrand => 2x half-domain)
    w = sg.qweights / torch.exp(-sg.t.square() / 2)
    assert abs(float(w.sum()) - 6.0) < 1e-12
    # the analytic null floor ~1.05 (int (1 - e^{-t^2}) e^{-t^2/2} dt over [-3, 3])
    assert 1.04 < sg.null_floor() < 1.07


@pytest.mark.parametrize("n", [32, 256, 1024])
def test_sigreg_sits_at_its_analytic_floor_on_gaussian_data(n):
    torch.manual_seed(0)
    sg = SIGReg(512)
    x = torch.randn(4, n, 32)                    # (T, N, D): one test per timestep over N sequences
    v = float(sg(x, sync=False))
    assert abs(v - sg.null_floor()) < 0.12 * sg.null_floor(), (n, v)
    # paper K.3: the floor is 1.04-1.37 over n = 32..2048
    assert 0.9 < v < 1.4


def test_sigreg_grows_on_collapsed_data_and_linearly_in_n():
    sg = SIGReg(64)
    u = sg.sample_directions(8, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    base = torch.randn(1, 1, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    stats = {}
    for n in (64, 256):
        x = base.expand(1, n, 8).contiguous()    # exact collapse: every sample identical
        stats[n] = float(sg(x, directions=u, sync=False))
    assert stats[64] > 10 * sg.null_floor()
    assert abs(stats[256] / stats[64] - 4.0) < 1e-9          # prefactor n: exactly linear in n


def test_sigreg_sees_scale_and_rank_collapse_but_not_the_background_collapse():
    torch.manual_seed(1)
    sg = SIGReg(256)
    floor = sg.null_floor()
    g = torch.randn(2, 512, 16)
    assert float(sg(g, sync=False)) < 1.25 * floor
    assert float(sg(0.1 * g, sync=False)) > 20 * floor               # shrunken
    rank1 = torch.zeros_like(g)
    rank1[..., 0] = g[..., 0]
    assert float(sg(rank1, sync=False)) > 20 * floor                 # rank-1
    # Prop. 1 / Prop. 2: episode-constant latents with a (T, N, D) grouping read the FLOOR
    ep = torch.randn(1, 512, 16)
    const_in_time = ep.expand(5, 512, 16).contiguous()               # z_t = b for every t
    assert float(sg(const_in_time, sync=False)) < 1.25 * floor


def test_sigreg_sharding_invariance_float64():
    """K.3 / Table 11: one fixed batch, W shards, in float64 -> the SAME number."""
    torch.manual_seed(0)
    sg = SIGReg(64)
    x = torch.randn(256, 32, dtype=torch.float64)
    u = sg.sample_directions(32, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    ref = sg(x.unsqueeze(0), directions=u, sync=False)
    for world in (1, 2, 4, 8):
        shards = [s.unsqueeze(0) for s in x.chunk(world, dim=0)]
        ecfs = [sg.local_ecf(s, u) for s in shards]
        cos_g = sum(e[0] for e in ecfs) / world
        sin_g = sum(e[1] for e in ecfs) / world

        def reduce_fn(c, s, _c=cos_g, _s=sin_g, _w=world):
            return _c, _s, _w                              # what the all-reduce returns on every rank

        vals = [sg(s, directions=u, reduce_fn=reduce_fn) for s in shards]
        for v in vals:
            assert abs(float(v) - float(ref)) < 1e-12 * max(1.0, float(ref)), (world, float(v), float(ref))
    # the failure mode the all-reduce removes: averaging W small local statistics is NOT invariant
    shards = [s.unsqueeze(0) for s in x.chunk(4, dim=0)]
    naive = sum(float(sg(s, directions=u, sync=False)) for s in shards) / 4
    assert abs(naive - float(ref)) > 1e-3 * float(ref)


def test_sigreg_generator_makes_the_direction_draw_reproducible():
    sg = SIGReg(64)
    x = torch.randn(1, 64, 8)
    a = sg(x, generator=torch.Generator().manual_seed(7), sync=False)
    b = sg(x, generator=torch.Generator().manual_seed(7), sync=False)
    c = sg(x, generator=torch.Generator().manual_seed(8), sync=False)
    assert float(a) == float(b) and float(a) != float(c)


def test_sigreg_gradient_is_finite_nonzero_and_bf16_safe():
    sg = SIGReg(64)
    x = (0.5 * torch.randn(2, 64, 8)).requires_grad_(True)
    sg(x, sync=False).backward()
    assert torch.isfinite(x.grad).all() and float(x.grad.abs().sum()) > 0
    xb = (0.5 * torch.randn(2, 64, 8)).to(torch.bfloat16).requires_grad_(True)
    v = sg(xb, sync=False)
    v.backward()
    assert v.dtype == torch.float32 and torch.isfinite(xb.grad.float()).all()


def _gloo_worker(rank, world, init_file, out_dir):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=pathlib.Path(init_file).as_uri(),
                            rank=rank, world_size=world)
    torch.manual_seed(0)
    x = torch.randn(256, 32, dtype=torch.float64)
    xs = x[rank * 128:(rank + 1) * 128].clone().requires_grad_(True)
    sg = SIGReg(64)
    val = sg(xs, generator=torch.Generator().manual_seed(123))
    val.backward()
    torch.save({"val": val.detach(), "grad": xs.grad}, os.path.join(out_dir, f"r{rank}.pt"))
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                    reason="gloo not available")
def test_sigreg_gloo_two_process():
    """A REAL process group: value == single-process reference; grad == W x reference shard grad
    (the identity-backward convention that makes DDP's gradient average equal the global one)."""
    import torch.multiprocessing as mp
    with tempfile.TemporaryDirectory() as td:
        init_file = os.path.join(td, "init")
        mp.spawn(_gloo_worker, args=(2, init_file, td), nprocs=2, join=True)
        r = [torch.load(os.path.join(td, f"r{i}.pt")) for i in range(2)]
    torch.manual_seed(0)
    x = torch.randn(256, 32, dtype=torch.float64).requires_grad_(True)
    ref = SIGReg(64)(x, generator=torch.Generator().manual_seed(123), sync=False)
    ref.backward()
    for i in range(2):
        assert abs(float(r[i]["val"]) - float(ref.detach())) < 1e-12 * max(1.0, float(ref.detach()))
        want = 2.0 * x.grad[i * 128:(i + 1) * 128]
        assert torch.allclose(r[i]["grad"], want, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------------------------
# IDM, Prop. 1 and Prop. 3
# ---------------------------------------------------------------------------------------------

def _train_idm(sample, d, da, steps=200, seed=0):
    torch.manual_seed(seed)
    head = IDMHead(d, da, hidden=32)
    opt = torch.optim.Adam(head.parameters(), lr=5e-3)
    for _ in range(steps):
        z, a = sample(256)
        loss = head.loss(z, a)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z, a = sample(4096)
        return float(head.loss(z, a))


def test_idm_loss_is_eq8_and_zero_head_reads_the_action_second_moment():
    head = IDMHead(4, 3, hidden=8)
    with torch.no_grad():
        head.net[-1].weight.zero_()
        head.net[-1].bias.zero_()
    torch.manual_seed(0)
    z = torch.randn(64, 5, 4)
    a = torch.randn(64, 4, 3)
    assert abs(float(head.loss(z, a).detach()) - float(a.pow(2).mean())) < 1e-6
    # stop-gradient on the target: no gradient reaches `a`
    a.requires_grad_(True)
    head.loss(z, a).backward()
    assert a.grad is None


def test_prop1_background_encoder_with_identity_predictor_reaches_zero_loss():
    """Proposition 1: E reads only the episode background b, F = id => L_pred = 0, and a
    marginal regulariser (SIGReg, (T,N,D) grouping) reads its floor -- it cannot see this."""
    torch.manual_seed(0)
    stack = HJEPAStack(_tiny_cfgs(d1=32), sigreg_projections=256,
                       predictor_overrides={1: IdentityPredictor(), 2: IdentityPredictor()}).double()
    b = torch.randn(1024, 1, 32, dtype=torch.double)               # background ~ N(0, I)
    z1 = b.expand(1024, 7, 32).contiguous()                        # constant within every episode
    a = torch.randn(1024, 6, 2, dtype=torch.double)                # unit-variance, independent of b
    out = stack.loss(z1, a)
    assert float(out["pred/L1"]) == 0.0
    assert float(out["pred/L2"]) < 1e-12
    floor = stack.sigreg.null_floor()
    assert abs(float(out["sigreg/L1"]) - floor) < 0.3 * floor      # SIGReg is blind to it ...
    healthy = torch.randn(1024, 7, 32, dtype=torch.double)         # ... exactly as it is to a healthy z
    v_healthy = float(stack.sigreg(healthy.transpose(0, 1), sync=False))
    assert abs(float(out["sigreg/L1"]) - v_healthy) < 0.3 * floor
    assert rho_within_episode(z1) < 1e-20                          # Definition 1: exact collapse
    assert rho_within_episode(healthy) > 0.8


def test_prop3_idm_floor_is_one_under_collapse_and_vanishes_without_it():
    d, da = 8, 2

    def collapsed(n):             # z_t = z_{t+1} = b (episode background); a independent
        b = torch.randn(n, 1, d)
        return b.expand(n, 2, d).contiguous(), torch.randn(n, 1, da)

    def informative(n):           # z_{t+1} = z_t + a_t (the action IS recoverable)
        z0 = torch.randn(n, 1, d)
        a = torch.randn(n, 1, da)
        z1 = z0.clone()
        z1[..., :da] = z1[..., :da] + a
        return torch.cat([z0, z1], dim=1), a

    floor = _train_idm(collapsed, d, da)
    assert 0.95 < floor < 1.06, floor                  # Prop. 3: >= 1 for unit-variance actions
    control = _train_idm(informative, d, da)
    assert control < 0.1, control                      # the same head CAN fit when z carries a_t


# ---------------------------------------------------------------------------------------------
# structure (Eq. 1), predictor, losses
# ---------------------------------------------------------------------------------------------

def test_eq1_stride_window_and_macro_action_indexing():
    # w = 1, s = 2
    stack = _stack(d1=4, a=3, d2=4, m=3, s=2, w=1)
    stack.encoders[0] = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        stack.encoders[0].weight.copy_(torch.eye(4))
    stack.macro_encoders[0] = _SumPool()
    z1, a = _batch(b=2, t1=8, d1=4, a=3)
    enc = stack.encode(z1, a)
    assert enc.z[1].shape == (2, 4, 4)                               # (8 - 1)//2 + 1
    assert torch.equal(enc.z[1], z1[:, 0:7:2])
    for t in range(3):
        assert torch.allclose(enc.act[1][:, t], a[:, 2 * t:2 * t + 2].sum(1))
    # w = 2, s = 3: z_t from states [3t, 3t+2); macro-action t from actions [3t+1, 3t+4)
    stack = _stack(d1=4, a=3, d2=4, m=3, s=3, w=2)
    stack.encoders[0] = _SumWindow(4)
    sp = _SumPool()
    sp.stride = 3
    stack.macro_encoders[0] = sp
    z1, a = _batch(b=2, t1=11, d1=4, a=3)
    enc = stack.encode(z1, a)
    assert enc.z[1].shape == (2, 4, 4)                               # (11 - 2)//3 + 1
    for t in range(4):
        assert torch.allclose(enc.z[1][:, t], z1[:, 3 * t] + z1[:, 3 * t + 1])
    assert enc.act[1].shape == (2, 3, 3)
    for t in range(3):
        assert torch.allclose(enc.act[1][:, t], a[:, 3 * t + 1:3 * t + 4].sum(1))


def test_encode_fails_loudly_on_short_or_mismatched_clips():
    stack = _stack()
    z1, a = _batch(t1=2)
    with pytest.raises(ValueError):
        stack.encode(z1, a)                                          # level 2 would have 1 state
    z1, a = _batch(t1=7)
    with pytest.raises(ValueError):
        stack.encode(z1, a[:, :-1])


@pytest.mark.parametrize("mode", ["mlp", "transformer"])
def test_macro_action_encoder_shape_and_final_layernorm(mode):
    enc = MacroActionEncoder(embed_dim=5, stride=3, out_dim=7, hidden=8, mode=mode)
    y = enc(torch.randn(4, 6, 3, 5))
    assert y.shape == (4, 6, 7)
    assert float(y.mean(-1).abs().max()) < 1e-5 and abs(float(y.var(-1, unbiased=False).mean()) - 1) < 0.1


def test_causal_predictor_is_causal_and_action_conditioned():
    torch.manual_seed(0)
    p = CausalPredictor(8, 2, width=16, depth=2, heads=2, context=4).eval()
    z, a = torch.randn(3, 4, 8), torch.randn(3, 4, 2)
    base = p(z, a)
    z2, a2 = z.clone(), a.clone()
    z2[:, 3] += 1.0
    a2[:, 3] += 1.0
    pert = p(z2, a2)
    assert torch.allclose(base[:, :3], pert[:, :3], atol=1e-6)         # causality
    assert float((base[:, 3] - pert[:, 3]).abs().max()) > 1e-4
    # actions move the output at init (identity-gate AdaLN)
    assert float((p(z, a) - p(z, a.flip(0))).abs().max()) > 1e-5
    # DiT adaLN-Zero: blocks start as identity => the output is action-INDEPENDENT at init
    pz = CausalPredictor(8, 2, width=16, depth=2, heads=2, context=4, adaln_init="zero").eval()
    assert float((pz(z, a) - pz(z, a.flip(0))).abs().max()) == 0.0


def test_loss_dict_terms_and_weighting():
    stack = _stack()
    z1, a = _batch()
    out = stack.loss(z1, a, generator=torch.Generator().manual_seed(0))
    for k in ("pred/L1", "sigreg/L1", "idm/L1", "pred/L2", "sigreg/L2", "idm/L2", "act_sigreg/L2", "total"):
        assert k in out and torch.isfinite(out[k]), k
    c1, c2 = stack.cfg
    want = (out["pred/L1"] + c1.beta * out["sigreg/L1"] + c1.gamma * out["idm/L1"]
            + out["pred/L2"] + c2.beta * out["sigreg/L2"] + c2.gamma * out["idm/L2"]
            + c2.act_sigreg * out["act_sigreg/L2"])
    assert abs(float(out["total"]) - float(want)) < 1e-4 * max(1.0, float(want))
    assert "act_sigreg/L1" not in out                                  # level 1 has no action term
    # sum reduction = D x mean reduction for the same predictor
    s2 = HJEPAStack(_tiny_cfgs(), sigreg_projections=32, pred_reduction="sum")
    s2.load_state_dict(stack.state_dict())
    o2 = s2.loss(z1, a, generator=torch.Generator().manual_seed(0))
    assert abs(float(o2["pred/L1"]) - 8 * float(out["pred/L1"])) < 1e-4 * float(o2["pred/L1"])


def test_pred_loss_target_is_the_next_state_not_the_current_one():
    """Eq. (2): position i predicts z_{i+1}. Data generated by the stub's own dynamics => exactly 0;
    a time-shifted or time-shuffled target must NOT read 0 (the alpha-3x deliberate regression)."""
    torch.manual_seed(0)
    stub = LinearActionPredictor(2, 8, seed=3)
    stack = HJEPAStack(_tiny_cfgs(d1=8, a=2), sigreg_projections=16,
                       predictor_overrides={1: stub}).double()
    stub.double()
    a = torch.randn(16, 6, 2, dtype=torch.double)
    z = [torch.randn(16, 8, dtype=torch.double)]
    for t in range(6):
        z.append(z[-1] + a[:, t] @ stub.W.detach())
    z1 = torch.stack(z, 1)
    assert float(stack.loss(z1, a)["pred/L1"]) < 1e-20                   # chunked (3 + 3) too
    shuffled = z1[:, torch.randperm(7, generator=torch.Generator().manual_seed(0))]
    assert float(stack.loss(shuffled, a)["pred/L1"]) > 1e-2              # time-shuffled targets
    # the reverse-time sequence is not generated by the same actions either
    assert float(stack.loss(z1.flip(1), a)["pred/L1"]) > 1e-2


def test_pred_loss_chunks_clips_longer_than_the_context():
    stack = _stack(ctx=3)
    z1, a = _batch(t1=13)                                              # 12 transitions = 4 chunks of 3
    assert math.isfinite(float(stack.loss(z1, a)["pred/L1"]))


@pytest.mark.parametrize("macro_mode", ["mlp", "transformer"])
@pytest.mark.parametrize("act_embed", [None, 5])
def test_reachability_every_parameter_of_every_level_is_live(macro_mode, act_embed):
    torch.manual_seed(0)
    cfgs = _tiny_cfgs(macro_mode=macro_mode)
    cfgs[0].act_embed_dim = act_embed
    stack = HJEPAStack(cfgs, sigreg_projections=32)
    stack.control = nn.Parameter(torch.zeros(3))                       # deliberately unused
    z1, a = _batch(t1=7)                                               # level 1: 6 = 2 x c ; level 2: 4 states
    stack.loss(z1, a)["total"].backward()
    rep = param_grad_report(stack)
    assert rep["control"] == 0.0                                       # the probe CAN read an exact zero
    assert dead_parameters(stack, rep) == ["control"]
    assert all(v > 0.0 for n, v in rep.items() if n != "control")


def test_mutation_gamma_zero_kills_exactly_the_idm_heads():
    torch.manual_seed(0)
    cfgs = _tiny_cfgs()
    for c in cfgs:
        c.gamma = 0.0
    stack = HJEPAStack(cfgs, sigreg_projections=32)
    z1, a = _batch(t1=7)
    stack.loss(z1, a)["total"].backward()
    dead = dead_parameters(stack)
    idm = sorted(n for n, _ in stack.named_parameters() if n.startswith("idm_heads."))
    assert dead == idm and len(idm) > 0                                # IDM heads exactly 0.0 ...
    assert all(v == 0.0 for n, v in param_grad_report(stack).items() if n.startswith("idm_heads."))
    # ... and the positive control: gamma > 0 revives them
    stack2 = HJEPAStack(_tiny_cfgs(), sigreg_projections=32)
    stack2.loss(z1, a)["total"].backward()
    assert dead_parameters(stack2) == []
    # built_levers reads what the loss reads: a gamma = 0 level reports idm_live False
    assert all(not row["idm_live"] for row in stack.built_levers().values())


def test_zero_init_adaln_is_glive_unsafe_which_is_why_it_is_not_the_default():
    torch.manual_seed(0)
    stack = HJEPAStack(_tiny_cfgs(adaln_init="zero"), sigreg_projections=32)
    z1, a = _batch(t1=7)
    stack.loss(z1, a)["total"].backward()
    dead = dead_parameters(stack)
    assert any(n.endswith("blocks.0.qkv.weight") for n in dead)       # gate 0 => block weights get 0 grad


def test_upper_level_objectives_backpropagate_into_the_level1_latents():
    """E4: no stop-gradient between levels -- the level-2 loss alone reaches z1."""
    stack = _stack()
    z1, a = _batch(t1=7, requires_grad=True)
    stack.loss(z1, a)["level/L2"].backward()
    assert float(z1.grad.abs().sum()) > 0
    z1b, a = _batch(t1=7, requires_grad=True)
    stack.loss(z1b, a, detach_target=True)["total"].backward()       # targets detached: still live
    assert float(z1b.grad.abs().sum()) > 0


def test_idm_target_is_detached_the_macro_encoder_learns_only_through_the_predictor():
    stack = _stack()
    z1, a = _batch(t1=7)
    enc = stack.encode(z1, a)
    idm = stack.idm_heads[1].loss(enc.z[1], enc.idm_target[1])
    idm.backward()
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in stack.macro_encoders.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in stack.encoders.parameters())


def test_training_reduces_prediction_and_idm_losses_on_structured_data():
    torch.manual_seed(0)
    d, da = 6, 2
    cfgs = _tiny_cfgs(d1=d, a=da, d2=4, m=2)
    for c in cfgs:
        c.beta, c.gamma = 0.05, 5.0
    stack = HJEPAStack(cfgs, sigreg_projections=32)
    B = torch.randn(da, d)
    opt = torch.optim.AdamW(stack.param_groups(3e-3, upper_lr_mult=1.0))

    def batch(n=32):
        a = torch.randn(n, 6, da)
        z = [torch.randn(n, d)]
        for t in range(6):
            z.append(0.9 * z[-1] + a[:, t] @ B)
        return torch.stack(z, 1), a

    first = None
    for step in range(120):
        z, a = batch()
        out = stack.loss(z, a)
        opt.zero_grad()
        out["total"].backward()
        opt.step()
        if step == 0:
            first = {k: float(v) for k, v in out.items()}
    last = {k: float(v) for k, v in out.items()}
    assert last["idm/L1"] < 0.5 * first["idm/L1"]
    assert last["pred/L1"] < first["pred/L1"]


def test_param_groups_partition_every_parameter_once_with_upper_lr_scaled():
    stack = _stack()
    groups = stack.param_groups(1e-3, upper_lr_mult=0.2)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(stack.parameters()))
    assert abs(groups[0]["lr"] - 1e-3) < 1e-12 and abs(groups[1]["lr"] - 2e-4) < 1e-12


# ---------------------------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------------------------

def _sens_stack(pred_cls=None, zero_rows=()):
    torch.manual_seed(0)
    over = None
    if pred_cls == "identity":
        over = {1: IdentityPredictor(), 2: IdentityPredictor()}
    elif pred_cls == "linear":
        over = {1: LinearActionPredictor(3, 8, zero_rows=zero_rows),
                2: LinearActionPredictor(3, 6)}
    cfgs = _tiny_cfgs(a=3, d1=8, d2=6, m=3)
    return HJEPAStack(cfgs, sigreg_projections=16, predictor_overrides=over)


def _windows_for_sens(b=12, h=10, a=3):
    g = torch.Generator().manual_seed(2)
    return torch.randn(b, 8, generator=g), torch.randn(b, h, a, generator=g)


def test_action_sensitivity_controls_read_their_known_values():
    z0, acts = _windows_for_sens()
    ks = (1, 5, 10)
    # (1) swapping with the window's OWN actions: exactly 0.0, even for a trained-looking predictor
    lin = _sens_stack("linear")
    own = lin.action_sensitivity(z0, acts, ks=ks, perm=torch.arange(12))
    assert all(v == 0.0 for v in own.values())
    # (2) identity predictor: exactly 0.0 (0 / eps), never NaN
    ident = _sens_stack("identity")
    r = ident.action_sensitivity(z0, acts, ks=ks)
    assert all(v == 0.0 for v in r.values())
    # (3) a predictor that adds the action embedding linearly: > 0 (the probe CAN see a non-zero)
    r = lin.action_sensitivity(z0, acts, ks=ks)
    assert all(r[f"sens/all/k{k}"] > 0.05 for k in ks)
    # a real (randomly initialised, identity-gate) predictor is also > 0
    real = _sens_stack()
    r = real.action_sensitivity(z0, acts, ks=ks)
    assert all(r[f"sens/all/k{k}"] > 0 for k in ks)


def test_action_sensitivity_is_per_channel_and_isolates_a_deaf_channel():
    z0, acts = _windows_for_sens()
    stack = _sens_stack("linear", zero_rows=(1,))          # channel 1 is IGNORED by the predictor
    r = stack.action_sensitivity(z0, acts, ks=(5,), channels={"a": [0], "steer": [1], "v": [2], "all": [0, 1, 2]})
    assert r["sens/steer/k5"] == 0.0                       # a deaf channel reads exactly 0
    assert r["sens/a/k5"] > 0 and r["sens/v/k5"] > 0
    assert r["sens/all/k5"] >= max(r["sens/a/k5"], r["sens/v/k5"]) * 0.99


def test_action_sensitivity_level2_uses_macro_action_space_and_restores_mode():
    stack = _sens_stack("linear")
    stack.train()
    z2 = torch.randn(5, 6)
    m = torch.randn(5, 4, 3)
    r = stack.action_sensitivity(z2, m, level=2, ks=(1, 4))
    assert r["sens/all/k4"] > 0 and stack.training


def test_donor_indices_never_self_and_cross_episode():
    p = donor_indices(10, seed=0)
    assert sorted(p.tolist()) == list(range(10)) and (p != torch.arange(10)).all()
    assert torch.equal(p, donor_indices(10, seed=0)) and not torch.equal(p, donor_indices(10, seed=1))
    ep = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    d = donor_indices(10, episode_ids=ep, seed=0)
    assert (ep[d] != ep).all()
    assert torch.equal(d, donor_indices(10, episode_ids=ep, seed=0))


def test_rho_within_episode_eq10():
    torch.manual_seed(0)
    E, T, D = 40, 30, 6
    b = torch.randn(E, 1, D)
    assert rho_within_episode(b.expand(E, T, D).contiguous()) < 1e-20          # exact collapse -> 0
    iid = torch.randn(E, T, D)
    assert abs(rho_within_episode(iid) - (T - 1) / T) < 0.05                   # no episode structure
    mixed = b + 0.1 * torch.randn(E, T, D)
    rho = rho_within_episode(mixed)
    assert 0.002 < rho < 0.03                                                  # mostly background
    # law of total variance: rho + between-episode share == 1 (population variances)
    zd = mixed.double()
    between = zd.mean(1).var(0, unbiased=False).sum() / zd.reshape(-1, D).var(0, unbiased=False).sum()
    assert abs(rho + float(between) - 1.0) < 1e-12
    # the flat / ragged form agrees with the (E, T, D) form
    ids = torch.arange(E).repeat_interleave(T)
    assert abs(rho_within_episode(mixed.reshape(-1, D), episode_ids=ids) - rho) < 1e-12


def test_monotonicity_is_negative_spearman():
    t = torch.arange(10.0)
    assert float(monotonicity(-t)) == pytest.approx(1.0)
    assert float(monotonicity(t)) == pytest.approx(-1.0)
    assert monotonicity(torch.stack([-t, t])).tolist() == pytest.approx([1.0, -1.0])


# ---------------------------------------------------------------------------------------------
# planner and scorer on a toy linear system with a known optimum
# ---------------------------------------------------------------------------------------------

class _LinDyn(nn.Module):
    """z_{t+1} = z_t + a_t B (c = 1 or more, no state mixing) -- exact, known optimum."""
    context = 3

    def __init__(self, B):
        super().__init__()
        self.register_buffer("B", B)

    def forward(self, z, a):
        return z + a @ self.B


def _toy(d=3, da=2, seed=0, two_level=True):
    g = torch.Generator().manual_seed(seed)
    B = torch.randn(da, d, generator=g)
    over = {1: _LinDyn(B)}
    cfgs = [LevelConfig(latent_dim=d, action_dim=da, context=3)]
    if two_level:
        over[2] = _LinDyn(B)                                  # C = B so level-1 can track level-2's subgoal
        cfgs.append(LevelConfig(latent_dim=d, action_dim=da, stride=2, window=1, enc_hidden=4, macro_hidden=4))
    stack = HJEPAStack(cfgs, sigreg_projections=8, predictor_overrides=over)
    if two_level:
        stack.encoders[0] = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            stack.encoders[0].weight.copy_(torch.eye(d))
        stack.encoders[0].requires_grad_(False)
    return stack, B, g


def test_plan_topdown_reduces_cost_on_a_toy_linear_system_single_level():
    stack, B, g = _toy(two_level=False)
    z0 = torch.randn(4, 3, generator=g)
    a_true = torch.randn(4, 4, 2, generator=g)
    goal = z0 + (a_true.sum(1)) @ B                           # reachable in H = 4 steps
    res = stack.plan_topdown(z0, goal, top_horizon=4, num_candidates=8, n_iters=400, lr=0.05,
                             last_k=1, generator=torch.Generator().manual_seed(0))
    assert res.actions.shape == (4, 4, 2)
    assert (res.costs[1] < 1e-3 * res.init_costs[1].clamp_min(1e-9)).all()
    assert float(res.costs[1].max()) < 1e-3
    # the returned plan really reaches the goal when replayed through the dynamics
    end = z0 + res.actions.sum(1) @ B
    assert float((end - goal).norm(dim=-1).max()) < 5e-2
    # best-so-far cost is non-increasing by construction
    h = res.history[1]
    assert all(h[i + 1] <= h[i] + 1e-9 for i in range(len(h) - 1))
    # the model's parameters were never touched by the planner
    assert all(p.grad is None for p in stack.parameters())


def test_plan_topdown_two_levels_subgoal_tracking():
    stack, B, g = _toy(two_level=True)
    z0 = torch.randn(3, 3, generator=g)
    m_true = torch.randn(3, 3, 2, generator=g)
    goal2 = z0 + m_true.sum(1) @ B                            # level-2 goal, reachable in 3 macro-steps
    res = stack.plan_topdown(z0, goal2, top_horizon=3, num_candidates=8, n_iters=500, lr=0.05,
                             generator=torch.Generator().manual_seed(1))
    assert res.level_actions[2].shape == (3, 3, 2) and res.level_actions[1].shape == (3, 2, 2)  # H1 = K*s + w - 1 = 2
    assert float(res.costs[2].max()) < 1e-3
    assert float(res.costs[1].max()) < 1e-3                   # level 1 tracks the first level-2 subgoal
    # the level-1 plan lands on the level-2 subgoal z^2_1
    sub = res.level_rollouts[2][:, 0]
    end1 = z0 + res.actions.sum(1) @ B
    assert float((end1 - sub).norm(dim=-1).max()) < 5e-2
    assert (res.costs[2] < 1e-3 * res.init_costs[2].clamp_min(1e-9)).all()


def test_plan_topdown_respects_box_bounds_and_hits_the_analytic_optimum():
    stack = HJEPAStack([LevelConfig(latent_dim=2, action_dim=2, context=3)], sigreg_projections=8,
                       predictor_overrides={1: _LinDyn(torch.eye(2))})
    z0 = torch.zeros(1, 2)
    goal = torch.tensor([[3.0, -3.0]])
    res = stack.plan_topdown(z0, goal, top_horizon=2, num_candidates=4, n_iters=300, lr=0.1,
                             last_k=1, bounds={1: (-0.5, 0.5)}, generator=torch.Generator().manual_seed(0))
    assert float(res.actions.abs().max()) <= 0.5 + 1e-6
    # max reachable displacement is (1, -1) => remaining (2, -2) => cost 8 exactly
    assert abs(float(res.costs[1]) - 8.0) < 1e-3
    # a non-box constraint through project_fn (clip the L1 norm of every step)
    res2 = stack.plan_topdown(z0, goal, top_horizon=2, num_candidates=4, n_iters=200, lr=0.1,
                              last_k=1, project_fn=lambda lv, a: a.clamp(-0.25, 0.25),
                              generator=torch.Generator().manual_seed(0))
    assert float(res2.actions.abs().max()) <= 0.25 + 1e-6 and float(res2.costs[1]) > float(res.costs[1])


def test_plan_topdown_refines_a_seed_plan_without_making_it_worse():
    stack, B, g = _toy(two_level=False)
    z0 = torch.randn(2, 3, generator=g)
    goal = z0 + torch.randn(2, 3, generator=g)
    seed_plan = 0.3 * torch.randn(2, 4, 2, generator=g)
    res = stack.plan_topdown(z0, goal, top_horizon=4, num_candidates=1, n_iters=60, lr=0.02, last_k=1,
                             init_actions={1: seed_plan})
    seed_cost = stack.goal_cost(stack.rollout(1, z0, seed_plan), goal, 1, last_k=1)
    assert (res.costs[1] <= seed_cost + 1e-9).all() and (res.costs[1] < seed_cost).any()
    assert torch.allclose(res.init_costs[1], seed_cost, atol=1e-6)


def test_plan_topdown_early_stopping_and_mode_restore():
    stack, B, g = _toy(two_level=False)
    stack.train()
    z0 = torch.randn(1, 3, generator=g)
    goal = z0.clone()                                           # already at the goal
    res = stack.plan_topdown(z0, goal, top_horizon=2, num_candidates=1, n_iters=90, patience=3, last_k=1)
    assert res.iters[1] < 90 and stack.training


def test_score_candidates_ranks_by_the_lifted_goal_distance():
    stack, B, g = _toy(two_level=True)
    z0 = torch.randn(2, 3, generator=g)
    target = torch.randn(2, 2, generator=g)                     # desired TOTAL action over H1 = 2 steps
    goal = z0 + target @ B
    cands = 0.5 * torch.randn(2, 5, 2, 2, generator=g)
    cands[:, 3] = target.unsqueeze(1) / 2                       # candidate 3 hits the goal exactly
    cost = stack.score_candidates(z0, cands, goal, level=2)
    assert cost.shape == (2, 5)
    assert cost.argmin(dim=1).tolist() == [3, 3] and float(cost[:, 3].max()) < 1e-10
    cost2, det = stack.score_candidates(z0, cands, goal, level=2, return_details=True)
    assert det["lifted"].shape == (2, 5, 2, 3) and torch.equal(cost, cost2)
    # misaligned horizon (H1 + 1 - w not a multiple of the stride) fails loud, never silently
    with pytest.raises(ValueError):
        stack.score_candidates(z0, torch.randn(2, 5, 3, 2), goal, level=2)


def test_built_levers_reads_the_built_modules():
    stack = _stack(s=3, w=1)
    lev = stack.built_levers()
    assert lev[2]["stride"] == 3 and lev[2]["window"] == 1 and lev[1]["context"] == 3
    assert lev[1]["gamma"] == 100.0 and lev[2]["act_sigreg"] == 0.005
    assert stack.parameter_counts()["total"] == sum(p.numel() for p in stack.parameters())


def test_action_prior_fit_sets_percentile_box():
    stack = _stack()
    acts = torch.randn(5000, 3)
    stack.fit_action_prior(2, acts)
    lo, hi = stack.prior_lo_2, stack.prior_hi_2
    assert (lo < -1.5).all() and (hi > 1.5).all() and (lo > -3).all()
    assert abs(float(stack.prior_std_2.mean()) - 1.0) < 0.1
