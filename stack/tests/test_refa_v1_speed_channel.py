"""The speed channel (#5) for refav1, under the PI ruling of 2026-09-02:

    "It is allowed to use the velocity as initial measured state at its cycle
     time. It is not allowed to use the future dynamic information from the
     ground truth."

REF-A's longitudinal blindness (3.73 m) was repaired by a speed input (0.83 m
— `MODEL_REGISTRY.md`); refav1's predictor saw (a, kappa) only. The admissible
form is a THIRD predictor input v_k/SPEED_SCALE_MPS with v_k INTEGRATED from
the measured anchor speed and the action sequence — never indexed from a
future GT speed — while the planner's CONTROL space stays (a, kappa).

⭐ THE LOAD-BEARING TEST feeds one v0 with two action sequences and reads the
channel the predictor actually received: it must differ, follow
v0 + Σ a·dt exactly, and start at the measured v0 in both.

⛔ THE OFF-PATH IDENTITY TEST pins that `speed_channel=False` (and
`tmix_groups=None`) leaves the state_dict keys and the fixed-seed forward loss
byte-identical to the pre-edit code — the live Thor run's checkpoint must load.
"""
import hashlib
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.refs.refa_v1 import SPEED_SCALE_MPS, RefAV1, RefAV1Config
from tanitad.refs.refa_v1_plan import PlanConfig


def _cfg(**kw) -> RefAV1Config:
    base = dict(d_enc=16, d_state=16, n_tokens=8,
                op_dt=0.2, op_steps=30, op_layers=1, op_heads=2, op_window=2,
                tac_dt=0.6, tac_steps=10, tac_queries=4, tac_layers=1,
                str_dt=3.0, str_steps=2, str_dim=8, str_layers=1)
    base.update(kw)
    return RefAV1Config(**base)


def _hier_cfg(**kw) -> RefAV1Config:
    """The trainer's --smoke configuration (`refa_v1_train.build_model`)."""
    c = RefAV1Config(strategic_cfg=StrategicPolicyConfig(d_model=32, depth=1,
                                                         n_heads=2, d_ctx=16,
                                                         d_cmd=8),
                     tactical_cfg=TacticalPolicyConfig(d_model=32, depth=1,
                                                       n_heads=2, d_intent=16),
                     **kw)
    c.d_enc, c.n_tokens, c.d_state = 32, 8, 32
    c.op_layers, c.op_heads, c.tac_layers = 1, 2, 1
    c.tac_queries, c.str_dim, c.str_layers = 4, 16, 1
    return c


def _inputs(c: RefAV1Config, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(2, c.op_window, c.n_tokens, c.d_enc, generator=g)
    a = torch.randn(2, c.op_steps, c.a_dim, generator=g)
    fut = torch.randn(2, c.op_steps, c.n_tokens, c.d_enc, generator=g)
    fit = torch.randn(256, c.d_enc, generator=g)
    return f, a, fut, fit


def _expected_v(a: torch.Tensor, v0: torch.Tensor, dt: float) -> torch.Tensor:
    """v_k = v0 + sum_{j<k} a_j dt, written independently of the model."""
    out = torch.empty_like(a)
    for b in range(a.shape[0]):
        v = float(v0[b])
        for k in range(a.shape[1]):
            out[b, k] = v
            v = v + float(a[b, k]) * dt
    return out


# --------------------------------------------------------- the a_dim split --
def test_a_dim_stays_the_control_width_and_a_in_dim_is_the_predictor_width():
    off, on = _cfg(), _cfg(speed_channel=True)
    assert (off.a_dim, off.a_in_dim) == (2, 2)
    assert (on.a_dim, on.a_in_dim) == (2, 3)
    assert RefAV1Config().speed_channel is False
    m = RefAV1(on)
    assert m.operative.act[0].in_features == 3          # predictor INPUT
    assert m.tactical.act[0].in_features == 3
    assert m.strategic.act.in_features == 2             # DECISION: strategic stays (a, kappa)
    assert m.proposal[-1].out_features == on.proposal_k * on.plan_steps * 2   # CONTROL space


# ------------------------------------------------------------- (1) shapes --
def test_shapes_with_the_speed_channel_on_and_the_refusals():
    c = _hier_cfg(speed_channel=True)
    m = RefAV1(c)
    f, a, fut, fit = _inputs(c)
    m.std.fit(fit)
    v0 = torch.tensor([3.0, 25.0])
    out = m(f, a, future_feats=fut, v0=v0)
    assert out["op_pred"].shape == (2, c.op_steps, c.n_tokens, c.d_state)
    assert out["tac_pred"].shape == (2, c.tac_steps, c.tac_queries, c.d_state)
    assert out["str_pred"].shape == (2, c.str_steps, c.str_dim)
    assert out["proposal"].shape == (2, c.proposal_k, c.plan_steps, 2)
    assert torch.isfinite(out["loss"].detach())
    out["loss"].backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in m.operative.act.parameters())
    with pytest.raises(ValueError, match="v0"):            # no anchor speed
        m(f, a, future_feats=fut)
    with pytest.raises(ValueError, match="derived"):       # pre-widened action
        m(f, torch.randn(2, c.op_steps, 3), future_feats=fut, v0=v0)
    with pytest.raises(ValueError, match="rows"):          # wrong batch
        m(f, a, future_feats=fut, v0=torch.tensor([1.0]))


def test_the_ext_and_cf_paths_run_with_the_channel_on():
    """Strategic extension (2-wide opening actions) and the counterfactual
    term (negatives re-integrated from THIS row's v0) both stay live."""
    c = _cfg(speed_channel=True, w_cf=0.5, cf_negs=1)
    m = RefAV1(c)
    f, a, fut, fit = _inputs(c)
    m.std.fit(fit)
    k_ext = c.str_ext_steps
    out = m(f, a, future_feats=fut, v0=torch.tensor([4.0, 9.0]),
            str_ext_targets=torch.randn(2, k_ext, c.n_tokens, c.d_enc),
            str_ext_actions=torch.randn(2, k_ext, c.a_dim))
    assert torch.isfinite(out["loss"].detach())
    assert "loss_feat_str_ext" in out and "cf_loss" in out
    assert out["str_pred_ext"].shape == (2, k_ext, c.str_dim)


# ------------------------------------------------ (2) THE LOAD-BEARING ONE --
def test_THE_LOAD_BEARING_ONE_speed_is_integrated_from_v0_and_actions_not_read_from_gt():
    c = _cfg(speed_channel=True)
    m = RefAV1(c)
    f, a1, fut, fit = _inputs(c)
    m.std.fit(fit)
    a2 = a1.clone()
    a2[..., 0] = -a1[..., 0]                 # same kappa, opposite accelerations
    v0 = torch.tensor([10.0, 20.0])
    seen = []
    orig = m.operative.rollout

    def spy(field, actions, intent=None, last_only=False):
        seen.append(actions.detach().clone())
        return orig(field, actions, intent=intent, last_only=last_only)

    m.operative.rollout = spy
    with torch.no_grad():
        m(f, a1, future_feats=fut, v0=v0)
        m(f, a2, future_feats=fut, v0=v0)
    x1, x2 = seen
    assert x1.shape[-1] == 3 and x2.shape[-1] == 3
    assert torch.equal(x1[..., :2], a1) and torch.equal(x2[..., :2], a2)
    v1, v2 = x1[..., 2] * SPEED_SCALE_MPS, x2[..., 2] * SPEED_SCALE_MPS
    # k = 0 is the MEASURED state — identical for both sequences
    assert torch.allclose(v1[:, 0], v0) and torch.allclose(v2[:, 0], v0)
    # from k = 1 the channel follows the ACTIONS ...
    assert not torch.allclose(v1[:, 1:], v2[:, 1:])
    # ... by exactly v0 + sum_{j<k} a_j dt, against an independent integrator
    assert torch.allclose(v1, _expected_v(a1[..., 0], v0, c.op_dt), atol=1e-4)
    assert torch.allclose(v2, _expected_v(a2[..., 0], v0, c.op_dt), atol=1e-4)
    # opposite accelerations integrate symmetrically about v0
    assert torch.allclose(v1 + v2, 2 * v0[:, None].expand_as(v1), atol=1e-4)
    # and a different v0 with the same actions shifts every step by that constant
    x3 = m.augment_actions(a1, v0 + 5.0)[..., 2] * SPEED_SCALE_MPS
    assert torch.allclose(x3 - v1, torch.full_like(v1, 5.0), atol=1e-4)
    # the tactical rollout consumes the SAME channel at its window openings
    tac_seen = []
    m.tactical.rollout = (lambda field, actions, intent=None, last_only=False,
                          _o=m.tactical.rollout:
                          (tac_seen.append(actions.detach().clone()),
                           _o(field, actions, intent=intent,
                              last_only=last_only))[1])
    with torch.no_grad():
        m(f, a1, future_feats=fut, v0=v0)
    stride = int(round(c.tac_dt / c.op_dt))
    assert torch.allclose(tac_seen[0][..., 2], x1[:, ::stride][:, :c.tac_steps, 2])


def test_teacher_forcing_telescopes_to_the_grid_speeds_and_T1_runs_the_same_code():
    """With a = dv/dt (the loader's definition) the integrated channel equals
    the grid speeds — the T0 definition. The SAME function then runs on
    arbitrary (model-generated) actions where no grid speed exists."""
    c = _cfg(speed_channel=True)
    m = RefAV1(c)
    v_grid = 5.0 + torch.cumsum(torch.rand(1, c.op_steps + 1), dim=1)
    a = ((v_grid[:, 1:] - v_grid[:, :-1]) / c.op_dt)[..., None]
    a = torch.cat([a, torch.zeros_like(a)], dim=-1)
    x = m.augment_actions(a, v_grid[:, 0])
    assert torch.allclose(x[..., 2] * SPEED_SCALE_MPS, v_grid[:, :-1], atol=1e-4)
    # T1-shaped call: the model's own (arbitrary) actions, same code path
    own = torch.randn(1, c.op_steps, 2)
    y = m.augment_actions(own, v_grid[:, 0])
    assert y.shape == (1, c.op_steps, 3)
    assert torch.allclose(y[..., 2] * SPEED_SCALE_MPS,
                          _expected_v(own[..., 0], v_grid[:, 0], c.op_dt),
                          atol=1e-4)


# -------------------------------------------------------------- (3) plan() --
def test_plan_rolls_3_wide_inputs_while_the_candidates_stay_2_wide():
    c = _hier_cfg(speed_channel=True)
    m = RefAV1(c)
    m.std.fit(torch.randn(256, c.d_enc))
    widths, firsts = [], []
    for pred in (m.tactical, m.operative):
        def spy(field, actions, intent=None, last_only=False, _o=pred.rollout):
            widths.append(int(actions.shape[-1]))
            firsts.append(actions[:, 0, 2].detach().clone())
            return _o(field, actions, intent=intent, last_only=last_only)
        pred.rollout = spy
    v0 = 12.0
    res = m.plan(torch.randn(1, c.op_window, c.n_tokens, c.d_enc), v0=v0,
                 target_speed=10.0,
                 plan_cfg=PlanConfig(n_samples=16, n_iters=2, n_elites=4,
                                     horizon=c.plan_steps, dt=c.op_dt, seed=0))
    assert widths and set(widths) == {3}                 # every rollout saw v
    assert all(torch.allclose(fv * SPEED_SCALE_MPS, torch.full_like(fv, v0))
               for fv in firsts)                         # from the tick's v0
    assert res.controls.shape == (c.plan_steps, 2)        # candidates 2-wide
    assert res.elites.shape[-1] == 2
    assert math.isfinite(res.cost)
    assert res.cost <= min(res.baseline_costs.values()) + 1e-6
    assert res.fine_costs and "plan" in res.fine_costs    # coarse-to-fine ran


# -------------------------------------------------- (4) OFF-path identity --
# Computed 2026-09-02 by `scratchpad/baseline_identity.py` against the PRE-EDIT
# refa_v1.py (commit 48c3e58 tree), torch 2.11.0+cu128, CPU float32, and
# reproduced bit-identically on a second run. Same constants as
# `test_refa_v1_ema_targets.py`; both flags default off, so both files must
# agree with the same pre-edit numbers.
_IDENTITY = {
    "nohier": dict(make=_cfg, n_keys=101,
                   sha="36a932a582ed230c80cb917c66a875e8a97aa0c8138df7224e596ece5a653f94",
                   loss=1.3071329593658447),
    "hier": dict(make=_hier_cfg, n_keys=167,
                 sha="ef0c9e81a7b7b2ecb94ecfc8e1ed353476502b28ace3bc2d354fe4b247454f10",
                 loss=1.3248074054718018),
}


@pytest.mark.parametrize("name", sorted(_IDENTITY))
def test_OFF_path_identity_state_dict_keys_and_fixed_seed_loss_unchanged(name):
    spec = _IDENTITY[name]
    torch.manual_seed(0)
    m = RefAV1(spec["make"]())
    assert m.cfg.speed_channel is False and m.cfg.tmix_groups is None
    assert m.cfg.a_in_dim == 2
    keys = sorted(m.state_dict())
    assert len(keys) == spec["n_keys"]
    assert hashlib.sha256("\n".join(keys).encode()).hexdigest() == spec["sha"]
    f, a, fut, fit = _inputs(m.cfg, seed=0)
    m.std.fit(fit)
    with torch.no_grad():
        out = m(f, a, future_feats=fut)                 # no v0: ignored when off
        out_v = m(f, a, future_feats=fut, v0=torch.tensor([7.0, 8.0]))
    assert float(out["loss"]) == pytest.approx(spec["loss"], rel=1e-6, abs=0.0)
    assert float(out_v["loss"]) == float(out["loss"])   # v0 is inert when off


# ------------------------------------------------------ (5) trainer flags --
def _trainer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "refa_v1_train.py"
    spec = importlib.util.spec_from_file_location("refa_v1_train_under_test",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_trainer_flags_reach_the_config_and_the_smoke_path_runs(tmp_path):
    tr = _trainer()
    common = ["--smoke", "--steps", "2", "--bs", "2", "--log-every", "1",
              "--save-every", "2", "--device", "cpu"]
    assert tr.main(common + ["--speed-channel", "--tmix-groups", "1",
                             "--out", str(tmp_path / "on")]) == 0
    ck = torch.load(tmp_path / "on" / "ckpt.pt", map_location="cpu",
                    weights_only=False)
    assert ck["cfg"]["speed_channel"] is True
    assert ck["cfg"]["tmix_groups"] == 1
    assert ck["model"]["operative.act.0.weight"].shape[1] == 3
    assert ck["model"]["tactical.act.0.weight"].shape[1] == 3
    assert ck["model"]["strategic.act.weight"].shape[1] == 2
    assert tuple(ck["model"]["adapter.tmix.weight"].shape) == (32, 32, 3)
    # the OFF default builds the shipped shapes
    assert tr.main(common + ["--out", str(tmp_path / "off")]) == 0
    ck = torch.load(tmp_path / "off" / "ckpt.pt", map_location="cpu",
                    weights_only=False)
    assert ck["cfg"]["speed_channel"] is False
    assert ck["cfg"]["tmix_groups"] is None
    assert ck["model"]["operative.act.0.weight"].shape[1] == 2
    assert tuple(ck["model"]["adapter.tmix.weight"].shape) == (32, 1, 3)
    assert not any(k.startswith("ema.") for k in ck["model"])


# ---------------------------------------------------------- tmix_groups --
def test_tmix_groups_default_is_depthwise_and_groups_1_builds_and_runs():
    torch.manual_seed(0)
    base = RefAV1(_cfg())
    torch.manual_seed(0)
    explicit = RefAV1(_cfg(tmix_groups=None))
    assert base.adapter.tmix.groups == 16 and explicit.adapter.tmix.groups == 16
    sb, se = base.state_dict(), explicit.state_dict()
    assert list(sb) == list(se)
    assert all(torch.equal(sb[k], se[k]) for k in sb)
    full = RefAV1(_cfg(tmix_groups=1))
    assert full.adapter.tmix.groups == 1
    assert tuple(full.adapter.tmix.weight.shape) == (16, 16, 3)
    f, a, fut, fit = _inputs(full.cfg)
    full.std.fit(fit)
    out = full(f, a, future_feats=fut)
    assert torch.isfinite(out["loss"].detach())
    out["loss"].backward()
    assert float(full.adapter.tmix.weight.grad.abs().sum()) > 0
    with pytest.raises(ValueError, match="tmix_groups"):
        _cfg(tmix_groups=5).sanity()                      # 16 % 5 != 0
    with pytest.raises(ValueError, match="tmix_groups"):
        _cfg(tmix_groups=0).sanity()


# ------------------------------------------------------------- the loader --
N_TOK, D, T_EP, ACCEL = 8, 16, 201, 0.7


def _fixture(tmp_path, n_eps=2):
    """`test_refav1_loader._fixture`, trimmed: features carry their own cache
    index, speed is linear so every grid speed is analytic."""
    cache, eps = tmp_path / "cache", tmp_path / "eps"
    cache.mkdir(), eps.mkdir()
    t_c = math.ceil(T_EP / 2)
    for i in range(n_eps):
        nm = f"ep{i:02d}"
        f = torch.arange(t_c, dtype=torch.float16)[:, None, None].expand(
            t_c, N_TOK, D).contiguous()
        torch.save(f, cache / f"{nm}.pt")
        poses = torch.zeros(T_EP, 4)
        poses[:, 3] = 5.0 + ACCEL * 0.1 * torch.arange(T_EP)
        actions = torch.zeros(T_EP, 2)
        actions[:, 0] = 0.01 * (i + 1)
        torch.save({"poses": poses, "actions": actions, "episode_id": nm},
                   eps / f"{nm}.v2ep.pt")
    return cache, eps


def test_the_loader_emits_v0_measured_at_the_anchor_and_no_future_speed(tmp_path):
    from tanitad.data.refav1_loader import RefAV1Windows
    cache, eps = _fixture(tmp_path)
    ld = RefAV1Windows(cache, eps, op_window=2, op_steps=30, str_dt=3.0,
                       str_ext_steps=2, seed=0)
    b = ld.batch(4)
    t_last = b["feats"][:, -1, 0, 0].float()             # the anchor index t
    assert b["v0"].shape == (4,)
    assert torch.allclose(b["v0"], 5.0 + ACCEL * 0.1 * (2 * t_last), atol=1e-4)
    assert [k for k in b if k.startswith("v")] == ["v0"]  # no future speed key
    # the model reproduces the fixture's grid speeds from (v0, a) ALONE:
    # the loader's a = dv/dt telescopes under teacher forcing (T0 definition)
    m = RefAV1(_cfg(speed_channel=True))
    x = m.augment_actions(b["actions"], b["v0"])
    frames = 2 * (t_last[:, None] + torch.arange(30)[None].float())
    assert torch.allclose(x[..., 2] * SPEED_SCALE_MPS,
                          5.0 + ACCEL * 0.1 * frames, atol=1e-3)
