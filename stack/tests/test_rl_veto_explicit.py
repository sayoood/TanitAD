"""The safety VETO is keyed on the CONFIG, never on the reward's KEY SET.

⛔⛔ THE DEFECT THIS FILE PINS (MEASURED 2026-09-05,
`TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-refc-rl-readiness/RESULT.md`
§12.1, RETRACTION #24).

`posttrain.rl_objective` used to compose the veto as::

    veto = None
    if "collision" in spec.weights:      # KEY MEMBERSHIP -- true at weight 0.0
        veto = COMPONENTS["collision"](traj, ctx) < 0
    ttc = ttc_violation(...)             # read no config at ALL
    veto = ttc if veto is None else (veto | ttc)

and `advantage.truncated_inter_anchor_advantage` pins a vetoed candidate at
``veto_value = -1.0`` **outside** the group-relative centring. So the panel's
"constant reward" control -- every weight set to 0.0, whose whole purpose was to
read the no-information value -- did **not** produce a zero advantage. It
produced a **veto-only advantage at FULL STRENGTH**: MEASURED `veto_rate_mean`
**0.0897**, `final_loss` **-1.863**, and 14 of 57 fan-safety metrics separated on
an arm that was supposed to move nothing.

⚠️ A zero-weight control that is not a null is the false-green class wearing a
control's clothes: it fired the pre-registered VOID gate and the whole `rl`-vs-base
attribution had to be withdrawn.

⭐ The repair is not "remember to check the weights". It is that the constraint
channel is now three booleans somebody has to TYPE (`veto_enabled`,
`veto_collision`, `veto_ttc`), recorded by `to_dict()` into the run's own
`config.json`. This file pins BOTH directions -- off means EXACTLY 0.0, and on
means the veto fires on a violating candidate regardless of what the reward
weights say.

No GPU, no corpus: hand-built [B, N, G, S, 2] trajectories against a hand-placed
lead, so the contact and TTC geometry is exact rather than sampled.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from tanitad.rl import rewards as R
from tanitad.rl.config import PostTrainConfig
from tanitad.rl.posttrain import rl_objective, veto_mask

#: The arm that discovered the defect, kept as a documented constant. It is a
#: CORPUS number (120 fixed eval windows of the B1 v7.2 fit set), so no unit test
#: can reproduce it -- `raw/verify_veto_measured.py` in the veto-only WP asserts it
#: against the banked `ctrl_const/arm_summary.json`. Quoted here so the number and
#: the mechanism live in the same place.
CTRL_CONST_VETO_RATE_MEAN = 0.0897

DT = 0.5


def _straight(v, *, b=1, n=2, g=2, s=5, lat=0.0):
    """[b, n, g, s, 2] constant-speed straight paths at ``v`` m/s on the 0.5 s grid."""
    t = torch.arange(s, dtype=torch.float32) * DT
    p = torch.stack([v * t, torch.full_like(t, lat)], dim=-1)
    return p.expand(b, n, g, s, 2).clone()


def _lead(x0, v, *, s=5):
    """[1, 1, 1, s, 2] lead track starting at ``x0`` and moving at ``v`` m/s."""
    t = torch.arange(s, dtype=torch.float32) * DT
    return torch.stack([x0 + v * t, torch.zeros_like(t)], dim=-1).reshape(1, 1, 1, s, 2)


def _ctx(lead_x0=6.0, lead_v=0.0):
    return {"dt": DT, "lead_len_m": 4.5, "lead_path": _lead(lead_x0, lead_v),
            "v0": torch.tensor([[[10.0]]])}


def _cfg(**kw):
    base = dict(method="grpo", group_size=2, normalize="none", dt=DT,
                reward_weights={k: 0.0 for k in R.DEFAULT_WEIGHTS},
                w_anchor=0.0, w_imitation=0.0, steps=1, batch=1, lr=0.0)
    base.update(kw)
    return PostTrainConfig(**base)


# --------------------------------------------------------------------------- #
# 1. the mechanism is gone from the source                                      #
# --------------------------------------------------------------------------- #
def test_veto_is_not_keyed_on_the_reward_key_set():
    """⛔ The literal defect: `rl_objective` must not consult `spec.weights` for the veto."""
    src = inspect.getsource(rl_objective)
    assert '"collision" in spec.weights' not in src, (
        "rl_objective still keys the veto on KEY MEMBERSHIP in spec.weights, which is "
        "TRUE at weight 0.0 -- the RESULT.md 12.1 defect. Key it on cfg.veto_enabled.")
    assert "veto_mask(" in src, "rl_objective must compose the veto through veto_mask()"


def test_config_records_the_three_channels():
    """A run record that does not say whether the constraint was on is not a record."""
    d = _cfg(veto_enabled=True, veto_collision=True, veto_ttc=False).to_dict()
    for k in ("veto_enabled", "veto_collision", "veto_ttc"):
        assert k in d, f"{k} missing from PostTrainConfig.to_dict()"
    assert d["veto_ttc"] is False


# --------------------------------------------------------------------------- #
# 2. OFF means EXACTLY zero -- the null that `ctrl_const` was supposed to be     #
# --------------------------------------------------------------------------- #
def test_zero_weight_spec_with_veto_off_is_an_actual_null():
    """⭐ THE HEADLINE PIN: zero weights + veto off => veto_rate EXACTLY 0.0.

    A candidate that is unambiguously in contact (a 10 m/s path into a lead
    standing 6 m ahead) must still read a zero veto rate, because the channel is
    switched off. Before the fix this configuration read 0.0897.
    """
    traj = _straight(10.0)
    ctx = _ctx(lead_x0=6.0, lead_v=0.0)
    cfg = _cfg(veto_enabled=False)
    spec = R.RewardSpec(weights=dict(cfg.reward_weights), dt=DT)

    # the scene really does contain a violation -- otherwise this test would pass
    # for the wrong reason (an absence proved by an empty population).
    assert bool((R.COMPONENTS["collision"](traj, ctx) < 0).any()), \
        "control failed: the constructed scene has no contact to veto"

    logp = torch.zeros(traj.shape[:-2], requires_grad=True)
    out = rl_objective(traj, logp, ctx, cfg, spec)
    assert float(out["veto_rate"]) == 0.0
    assert torch.equal(out["advantage"], torch.zeros_like(out["advantage"]))
    assert float(out["reward"].abs().max()) == 0.0


def test_zero_weight_spec_with_veto_on_is_a_veto_only_arm():
    """The other direction: the SAME zero weights with the veto on pin candidates at -1."""
    traj = _straight(10.0)
    ctx = _ctx(lead_x0=6.0, lead_v=0.0)
    cfg = _cfg(veto_enabled=True)
    spec = R.RewardSpec(weights=dict(cfg.reward_weights), dt=DT)
    logp = torch.zeros(traj.shape[:-2], requires_grad=True)
    out = rl_objective(traj, logp, ctx, cfg, spec)
    assert float(out["veto_rate"]) > 0.0
    assert float(out["reward"].abs().max()) == 0.0, "the reward must still be identically 0"
    assert float(out["advantage"].min()) == pytest.approx(cfg.veto_value), (
        "a vetoed candidate must be PINNED at veto_value, not merely ranked lower -- "
        "that pinning outside the centring is why a constant reward was not a null")


# --------------------------------------------------------------------------- #
# 3. the channel no longer depends on the reward carrying a "collision" key      #
# --------------------------------------------------------------------------- #
def test_collision_veto_fires_without_a_collision_weight():
    """⛔ The converse of the defect: no `collision` key, veto still fires."""
    traj = _straight(10.0)
    ctx = _ctx(lead_x0=6.0, lead_v=0.0)
    cfg = _cfg(reward_weights={"progress": 1.0}, veto_enabled=True,
               veto_collision=True, veto_ttc=False)
    assert bool(veto_mask(traj, ctx, cfg).any()), (
        "the collision channel must be keyed on cfg.veto_collision, not on the reward "
        "carrying a 'collision' key")


def test_ttc_channel_is_switchable():
    """The TTC half used to read no config at all -- it must now be switchable, and its
    OWN firing must be provable (a control that reads True with it on)."""
    # a 12 m/s path closing on a lead 20 m ahead at 2 m/s: no contact in 2 s, but the
    # gap closes fast enough for TTC to fall under 1.5 s.
    traj = _straight(12.0)
    ctx = _ctx(lead_x0=20.0, lead_v=2.0)
    on = _cfg(veto_enabled=True, veto_collision=False, veto_ttc=True)
    off = _cfg(veto_enabled=True, veto_collision=False, veto_ttc=False)
    assert bool(R.ttc_violation(traj, {**ctx, "ttc_min_s": on.ttc_min_s}).any()), \
        "control failed: the constructed scene has no TTC violation to veto"
    assert bool(veto_mask(traj, ctx, on).any())
    assert not bool(veto_mask(traj, ctx, off).any())


def test_veto_mask_returns_a_tensor_not_none_when_off():
    """A channel that reports 0.0 is evidence; a channel that reports nothing is not."""
    traj = _straight(3.0)
    ctx = _ctx(lead_x0=200.0, lead_v=0.0)
    m = veto_mask(traj, ctx, _cfg(veto_enabled=False))
    assert isinstance(m, torch.Tensor) and m.dtype == torch.bool
    assert m.shape == traj.shape[:-2]
    assert not bool(m.any())


# --------------------------------------------------------------------------- #
# 4. the arm table says what it does                                            #
# --------------------------------------------------------------------------- #
def test_driver_arm_table_declares_the_veto():
    """The veto-only arms are PRODUCTS, and `ctrl_null` is the null `ctrl_const` was not."""
    import importlib.util
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "scripts", "rl_refcv3_min.py")
    if not os.path.exists(path):
        pytest.skip("rl_refcv3_min.py not present in this tree")
    spec_ = importlib.util.spec_from_file_location("_rl_refcv3_min_armtable", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    arms = mod.ARMS
    for name in ("veto200", "veto2k"):
        assert arms[name]["veto_enabled"] is True
        assert set(arms[name]["weights"].values()) == {0.0}, \
            f"{name} must carry NO ranking signal -- the veto is the whole arm"
    assert arms["ctrl_null"]["veto_enabled"] is False
    assert set(arms["ctrl_null"]["weights"].values()) == {0.0}
    # the banked arm must still reproduce: the old key-membership veto gave reg_echo
    # the TTC channel only.
    assert arms["reg_echo"]["veto_collision"] is False
    assert arms["reg_echo"]["veto_ttc"] is True
