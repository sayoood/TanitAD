"""The refav1 stage-2 loader — values asserted against analytic kinematics.

The fixture builds episodes with LINEAR speed (constant accel) and a KNOWN
kappa channel, so the loader's (a, kappa) can be checked to numerical
precision instead of "shapes look right".
"""
import math

import pytest
import torch

from tanitad.data.refav1_loader import RefAV1Windows, split_episodes

N_TOK, D = 8, 16
T_EP = 201                     # v2ep frames at 10 Hz
T_C = math.ceil(T_EP / 2)      # 101 cache steps at 0.2 s
ACCEL = 0.7                    # m/s^2, exact on every grid


def _fixture(tmp_path, n_eps=2, t_ep=T_EP):
    cache = tmp_path / "cache"
    eps = tmp_path / "eps"
    cache.mkdir(exist_ok=True), eps.mkdir(exist_ok=True)
    for i in range(n_eps):
        nm = f"ep{i:02d}"
        t_c = math.ceil(t_ep / 2)
        # features: frame index broadcast, so a window's identity is readable
        f = torch.arange(t_c, dtype=torch.float16)[:, None, None].expand(
            t_c, N_TOK, D).contiguous()
        torch.save(f, cache / f"{nm}.pt")
        v = 5.0 + ACCEL * 0.1 * torch.arange(t_ep)          # linear speed
        poses = torch.zeros(t_ep, 4)
        poses[:, 3] = v
        actions = torch.zeros(t_ep, 2)
        actions[:, 0] = 0.01 * (i + 1)                       # kappa channel
        actions[:, 1] = -99.0                                # decoy: NOT kappa
        torch.save({"poses": poses, "actions": actions,
                    "episode_id": nm}, eps / f"{nm}.v2ep.pt")
    return cache, eps


def _loader(tmp_path, **kw):
    cache, eps = _fixture(tmp_path)
    base = dict(op_window=2, op_steps=30, str_dt=3.0, str_ext_steps=2, seed=0)
    base.update(kw)
    return RefAV1Windows(cache, eps, **base)


def test_window_count_is_exact(tmp_path):
    ld = _loader(tmp_path)
    # reach = (6 + 2*3)/0.2 = 60; t in [W-1, T_C - 60 - 2] -> 101-60-2-1+1 = 39
    per_ep = (T_C - 60 - 1) - (2 - 1)
    assert len(ld) == 2 * per_ep


def test_shapes_and_the_ext_contract(tmp_path):
    ld = _loader(tmp_path)
    b = ld.batch(3)
    assert b["feats"].shape == (3, 2, N_TOK, D)
    assert b["future_feats"].shape == (3, 30, N_TOK, D)
    assert b["actions"].shape == (3, 30, 2)
    assert b["str_ext_targets"].shape == (3, 2, N_TOK, D)
    assert b["str_ext_actions"].shape == (3, 2, 2)
    assert b["lat_label"] is None and b["nav_cmd"] is None


def test_ACTION_VALUES_match_analytic_kinematics(tmp_path):
    """⭐ a must equal the constant accel EXACTLY (linear speed), and kappa
    must be the MEASURED channel-0 — the decoy in channel 1 (-99) proves the
    (kappa, a)->(a, kappa) swap happened; consuming the stored order verbatim
    would put -99 in every kappa."""
    ld = _loader(tmp_path)
    b = ld.batch(4)
    assert torch.allclose(b["actions"][..., 0],
                          torch.full_like(b["actions"][..., 0], ACCEL),
                          atol=1e-5)
    assert float(b["actions"][..., 1].min()) > 0.0          # kappa, not -99
    assert float(b["str_ext_actions"][..., 0].mean()) == pytest.approx(ACCEL,
                                                                       abs=1e-5)


def test_future_and_ext_targets_sit_at_the_advertised_times(tmp_path):
    """Features carry their own cache index, so WHERE a target came from is
    read off the tensor — the D-REFAV1-LADDER lesson applied to the loader."""
    ld = _loader(tmp_path, seed=3)
    b = ld.batch(2)
    t_last = b["feats"][:, -1, 0, 0]                 # the window's last index
    fut0 = b["future_feats"][:, 0, 0, 0]
    assert torch.allclose(fut0, t_last + 1)          # first future = t+1
    # ext ticks close at 6+3=9 s and 6+6=12 s -> cache offsets 45 and 60
    assert torch.allclose(b["str_ext_targets"][:, 0, 0, 0], t_last + 45)
    assert torch.allclose(b["str_ext_targets"][:, 1, 0, 0], t_last + 60)


def test_a_mis_gridded_cache_is_REFUSED(tmp_path):
    cache, eps = _fixture(tmp_path)
    bad = torch.zeros(80, N_TOK, D, dtype=torch.float16)     # not ceil(201/2)
    torch.save(bad, cache / "ep00.pt")
    with pytest.raises(ValueError, match="time-warp"):
        RefAV1Windows(cache, eps, op_window=2, op_steps=30)


def test_missing_v2ep_is_REFUSED_by_name(tmp_path):
    cache, eps = _fixture(tmp_path)
    (eps / "ep01.v2ep.pt").unlink()
    with pytest.raises(FileNotFoundError, match="ep01"):
        RefAV1Windows(cache, eps, op_window=2, op_steps=30)


def test_too_short_episodes_are_a_loud_zero_not_a_silent_skip(tmp_path):
    cache, eps = _fixture(tmp_path, t_ep=61)                 # T_c=31 < reach
    with pytest.raises(ValueError, match="0 windows"):
        RefAV1Windows(cache, eps, op_window=2, op_steps=30)


def test_determinism_and_the_split(tmp_path):
    a = _loader(tmp_path, seed=7).batch(4)
    b = _loader(tmp_path, seed=7).batch(4)
    assert torch.equal(a["feats"], b["feats"])
    tr, va = split_episodes([f"ep{i}" for i in range(10)], val_frac=0.2, seed=1)
    assert len(va) == 2 and not set(tr) & set(va)
    tr2, va2 = split_episodes([f"ep{i}" for i in range(10)], val_frac=0.2,
                              seed=1)
    assert (tr, va) == (tr2, va2)
