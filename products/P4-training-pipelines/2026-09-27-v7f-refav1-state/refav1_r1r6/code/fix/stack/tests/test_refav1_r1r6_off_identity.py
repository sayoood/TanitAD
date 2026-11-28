"""PI 2026-09-27 (R1/R3/R5/R6): every new flag DEFAULT OFF, and OFF is BIT-IDENTICAL.

The cross-version proof (the unmodified c36b6ddd tree vs this one: 6,088 tensors bitwise --
model init / forward / grads / an AdamW step on 9 configs, plan() on 3 variants, the loader on
real eval episodes, and three `--smoke` trainer runs incl. their checkpoints) is banked in the
package's `raw/bitident/`. These tests pin, INSIDE one tree, the mechanism that makes it hold:

  * with every flag off NOTHING is built and NO random number is drawn -- a model built with
    the R1-R6 builder replaced by the pre-directive "set None" stub is bit-identical (state
    dict, the RNG stream after the build, and forward outputs incl. the loss);
  * every new forward input is REFUSED while its flag is off (never silently unused);
  * the trainer's new flags default off and translate to NO config kwarg and NO loader kwarg;
  * the loader emits exactly the pre-directive keys with its new options off.
Each identity carries a same-breath control that must DIFFER.
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.refs.refa_v1 import RefAV1, RefAV1Config

_TRAIN = Path(__file__).resolve().parents[1] / "scripts" / "refa_v1_train.py"


def _cfg(**kw) -> RefAV1Config:
    base = dict(d_enc=16, d_state=16, n_tokens=8, op_layers=1, op_heads=2, op_window=2,
                tac_queries=4, tac_layers=1, str_dim=8, str_layers=1,
                strategic_cfg=StrategicPolicyConfig(d_model=16, depth=1, n_heads=2, d_ctx=8,
                                                    d_cmd=8),
                tactical_cfg=TacticalPolicyConfig(d_model=16, depth=1, n_heads=2, d_intent=8))
    base.update(kw)
    return RefAV1Config(**base)


def _batch(c, seed=3):
    g = torch.Generator().manual_seed(seed)
    return dict(feats=torch.randn(2, c.op_window, c.n_tokens, c.d_enc, generator=g),
                actions=torch.randn(2, c.op_steps, c.a_dim, generator=g) * 0.1,
                future_feats=torch.randn(2, c.op_steps, c.n_tokens, c.d_enc, generator=g),
                v0=torch.rand(2, generator=g) * 20, nav_cmd=torch.tensor([0, 2]),
                lat_label=torch.tensor([1, -100]), lon_label=torch.tensor([0, 3]),
                route_label=torch.tensor([1, 0]))


def _stub_build(self, cfg, intent_dim):
    """The PRE-DIRECTIVE behaviour: no new module, no RNG draw -- only the None attributes the
    rest of the class reads."""
    self.vmax_to_ctx = self.vmax_to_intent = None
    self.goal_head = self.speed_band_head = self.traj_head = None


def _build(monkeypatch=None, stub=False, **kw):
    if stub:
        monkeypatch.setattr(RefAV1, "_build_r1r6", _stub_build)
    torch.manual_seed(11)
    m = RefAV1(_cfg(**kw))
    rng_after = torch.get_rng_state().clone()
    if stub:
        monkeypatch.undo()
    return m, rng_after


@pytest.mark.parametrize("variant", [{}, dict(speed_channel=True), dict(ema_targets=True),
                                     dict(strategic_cfg=None, tactical_cfg=None)])
def test_a_off_builds_nothing_draws_nothing_and_forwards_identically(monkeypatch, variant):
    m_new, rng_new = _build(**variant)
    m_old, rng_old = _build(monkeypatch, stub=True, **variant)
    assert torch.equal(rng_new, rng_old), "the OFF build drew random numbers"
    sd_new, sd_old = m_new.state_dict(), m_old.state_dict()
    assert list(sd_new) == list(sd_old)
    assert all(torch.equal(sd_new[k], sd_old[k]) for k in sd_new)
    for m in (m_new, m_old):
        with torch.no_grad():
            m.std.fit(torch.randn(32, 16, generator=torch.Generator().manual_seed(5)))
    b = _batch(m_new.cfg)
    if m_new.tactical_policy is None:
        for k in ("nav_cmd", "lat_label", "lon_label", "route_label"):
            b.pop(k)
    f, a_, fut = b.pop("feats"), b.pop("actions"), b.pop("future_feats")
    o_new = m_new(f, a_, future_feats=fut, **b)
    o_old = m_old(f, a_, future_feats=fut, **b)
    assert set(o_new) == set(o_old)
    for k, v in o_new.items():
        if torch.is_tensor(v):
            assert torch.equal(v, o_old[k]), k
        else:
            assert v == o_old[k], k
    # same-breath control: an ON flag DOES build and DOES draw, so both probes can see a build
    _, rng_default = _build()
    m_on, rng_on = _build(w_traj=1.0, traj_grid=(2, 4), traj_regions=(1, 2))
    assert not torch.equal(rng_on, rng_default)
    assert any(k.startswith("traj_head.") for k in m_on.state_dict())


def test_b_off_has_no_new_key_and_every_new_attribute_is_None():
    m = RefAV1(_cfg())
    for attr in ("vmax_to_ctx", "vmax_to_intent", "goal_head", "speed_band_head",
                 "traj_head"):
        assert getattr(m, attr) is None, attr
    bad = [k for k in m.state_dict()
           if k.startswith(("vmax_to", "goal_", "speed_band", "traj_"))]
    assert bad == []
    # control: the strategic layer IS still built with strategic_off at its default
    assert m.strategic_policy is not None and m.strategic is not None


@pytest.mark.parametrize("kw", [
    dict(speed_max_ms=torch.tensor([10.0, 20.0])),
    dict(goal_y=torch.zeros(2, 22), goal_w=torch.ones(2, 22)),
    dict(speed_band=torch.zeros(2, 2), speed_band_mask=torch.ones(2, dtype=torch.bool)),
    dict(traj_gt=torch.zeros(2, 30, 2), traj_mask=torch.ones(2, 30, dtype=torch.bool)),
    dict(a0=torch.zeros(2)), dict(kappa0=torch.zeros(2))])
def test_c_every_new_input_is_REFUSED_while_its_flag_is_off(kw):
    m = RefAV1(_cfg())
    with torch.no_grad():
        m.std.fit(torch.randn(32, 16))
    b = _batch(m.cfg)
    f, a_, fut = b.pop("feats"), b.pop("actions"), b.pop("future_feats")
    with pytest.raises(ValueError, match="silently unused|is 0 \\(no head\\)"):
        m(f, a_, future_feats=fut, **b, **kw)
    m(f, a_, future_feats=fut, **b)          # control: the same call without it runs


def _trainer():
    spec = importlib.util.spec_from_file_location("_refa_v1_train_offid", _TRAIN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_d_trainer_defaults_are_off_and_add_no_kwarg():
    T = _trainer()
    a = T.build_parser().parse_args([])
    assert (a.strategic_off, a.vmax_input, a.w_goal, a.w_speed_band, a.w_traj,
            a.goal_negatives, a.cot_negative_sidecar, a.r5_prior, a.r5_kappa_source) == (
        False, False, 0.0, 0.0, 0.0, "measured", None, "kdx", "steer_t0")
    assert T.r1r6_cfg_kwargs(a) == {}
    assert T.r1r6_loader_kwargs(a, RefAV1Config()) == {}
    # a pre-directive Namespace (no new attribute at all) builds the same config
    old = argparse.Namespace(**{k: v for k, v in vars(a).items()
                                if k not in ("strategic_off", "vmax_input", "w_goal",
                                             "w_speed_band", "w_traj", "r5_prior",
                                             "r5_kappa_source", "goal_negatives",
                                             "cot_negative_sidecar")})
    assert T.r1r6_cfg_kwargs(old) == {}
    # control: ON flags DO produce kwargs
    on = T.build_parser().parse_args(["--w-traj", "1", "--r5-prior", "cv", "--strategic-off"])
    kw = T.r1r6_cfg_kwargs(on)
    assert kw["w_traj"] == 1.0 and kw["traj_prior"] == "cv" and kw["strategic_off"] is True
    assert kw["w_feat_str"] == 0.0 and kw["w_str_label"] == 0.0


#: the dev-box eval kit, overridable -- an absolute home path only as an env default
_EVAL = Path(os.environ.get("TANITAD_REFAV1_EVAL_ROOT",
                            "C:/Users/Admin/tanitad-data/refav1-eval141"))
_LBL = Path(os.environ.get("TANITAD_V72_EVAL_LABELS",
                           "C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz"))


@pytest.mark.skipif(not (_EVAL / "refav1-fp8-eval").is_dir() or not _LBL.is_file(),
                    reason="the dev-box refav1 eval cache is not on this host")
def test_e_loader_off_emits_exactly_the_pre_directive_keys():
    from tanitad.data.refav1_loader import RefAV1Windows
    C, E = _EVAL / "refav1-fp8-eval", _EVAL / "eps"
    names = sorted(p.stem for p in C.glob("*.pt") if p.stem != "index")[:2]
    kw = dict(op_window=4, op_steps=30, str_ext_steps=2, lru=2, seed=1, episodes=names,
              labels_path=_LBL, nav_path=_LBL)
    off = RefAV1Windows(C, E, **kw).batch(3)
    assert list(off) == ["feats", "future_feats", "actions", "lat_label", "lon_label",
                         "route_label", "nav_cmd", "nav_valid", "v0", "str_ext_targets",
                         "str_ext_actions"]
    on = RefAV1Windows(C, E, goal_targets=True, speed_band=True, speed_max=True,
                       traj_steps=30, **kw).batch(3)
    for k in off:                         # the shared keys are bit-identical
        v = off[k]
        assert torch.equal(v, on[k]) if torch.is_tensor(v) else v == on[k], k
    assert set(on) - set(off) == {"goal_y", "goal_w", "speed_band", "speed_band_mask",
                                  "speed_max_ms", "speed_max_valid", "traj_gt", "traj_mask",
                                  "a0", "kappa0"}
