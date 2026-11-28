"""refcv7 wired into RefCV3Model — a REAL forward on the refcv6 build.

Builder copied from ``test_refcv6_bev_tactical_wiring._model`` (timm resnet18 at 64x64,
pretrained=False: no download; a one-parameter stand-in perception branch that honours
the real branch's contract).
"""
from __future__ import annotations

import dataclasses as _dc
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.refs import refc_v3 as v3                          # noqa: E402
from tanitad.refs import refcv6_tactical as v6tac               # noqa: E402
from tanitad.refs import refcv7_heads as r7h                    # noqa: E402
from tanitad.refs import refcv7_oracle as r7o                   # noqa: E402

D_BEV = 8


class _FakeBranch(torch.nn.Module):
    def __init__(self, d_bev: int):
        super().__init__()
        self.lift = None
        self.map_branch = None
        self.box_mem = None
        self.box_dec = None
        self.d_bev = int(d_bev)
        self.proj = torch.nn.Linear(1, self.d_bev)

    def forward(self, fmap_s16, grid=None, valid=None):
        b = fmap_s16.shape[0]
        pooled = fmap_s16.mean(dim=(1, 2, 3)).reshape(b, 1, 1)
        feats = self.proj(pooled).expand(b, 6, self.d_bev)
        return {"bev_feats": feats.transpose(1, 2).reshape(b, self.d_bev, 2, 3),
                "bev_tokens": feats}


def _cfg(refcv7: bool, tac: bool = True):
    from tanitad.refs.refc_agents import AgentSeamConfig
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.tac_vocab_version = "v7.0"
    cfg.core.encoder = _dc.replace(
        cfg.core.encoder, trunk="timm", trunk_name="resnet18.a1_in1k",
        trunk_pretrained=False, in_channels=3, image_size=64, image_width=None)
    cfg.core.agents = AgentSeamConfig(enable=True)
    cfg.core.decoder.cross_agent = True
    cfg.tac_decoder_v6 = tac
    cfg.tac_decoder_cfg = v6tac.TacticalDecoderConfig(
        d_model=64, n_layers=1, n_heads=4, ff_mult=2,
        d_agent=int(cfg.core.decoder.d), d_bev=D_BEV, sources=("agent", "bev"))
    cfg.refcv7 = refcv7
    cfg.refcv7_head_cfg = r7h.Refcv7HeadConfig(d_model=32, n_layers=1, n_heads=4,
                                               n_queries=8)
    return cfg


def _model(refcv7: bool = True, seed: int = 0):
    torch.manual_seed(seed)
    m = v3.RefCV3Model(_cfg(refcv7))
    m._perception = _FakeBranch(D_BEV)
    return m


def _fwd(model, b: int = 2, nav: int = 0):
    enc = model.cfg.core.encoder
    h, w = enc.image_hw()
    g = torch.Generator().manual_seed(1)
    frames = torch.rand(b, int(model.cfg.core.window), int(enc.in_channels), h, w,
                        generator=g)
    return model(frames, nav_cmd=torch.full((b,), nav, dtype=torch.long),
                 v0=torch.tensor([4.0] * b))


def test_off_build_has_no_refcv7_modules_or_ledger_lines():
    m = _model(refcv7=False)
    assert m.refcv7_wta is None and m.refcv7_scorer is None
    assert not any(k.startswith("refcv7_") for k in m.state_dict())
    assert "refcv7_wta" not in v3.param_breakdown_v3(m)


def test_building_refcv7_does_not_move_any_existing_parameter():
    """⛔ refcv7 is built LAST: every pre-existing parameter must draw the SAME
    values at the same seed with refcv7 on or off (bit-identity of the OFF arm)."""
    off = _model(refcv7=False, seed=0).state_dict()
    on = _model(refcv7=True, seed=0).state_dict()
    shared = [k for k in off if not k.startswith("_perception")]
    assert shared and all(torch.equal(off[k], on[k]) for k in shared)


def test_refcv7_without_the_scene_hook_REFUSES():
    with pytest.raises(ValueError, match="SCENE HOOK"):
        v3.RefCV3Model(_cfg(refcv7=True, tac=False))


def test_forward_emits_candidates_scores_and_the_deployed_pick():
    m = _model().eval()
    with torch.no_grad():
        out = _fwd(m)
    n = out["anchor_traj"].shape[1]
    q = m.cfg.refcv7_head_cfg.n_queries
    assert out["r7_candidates"].shape == (2, n + q, 8, 2)
    assert set(out["r7_logits"]) == set(r7o.SUBSCORES)
    assert torch.equal(out["traj"], out["traj_r7"])
    assert "traj_v3" in out                      # the refcv6 pick survives for comparison
    idx = out["r7_sel_idx"]
    assert torch.equal(out["traj"], out["r7_candidates"][torch.arange(2), idx])


def test_select_off_keeps_the_refcv6_pick():
    torch.manual_seed(0)
    cfg = _cfg(True)
    cfg.refcv7_select = False
    m = v3.RefCV3Model(cfg)
    m._perception = _FakeBranch(D_BEV)
    with torch.no_grad():
        out = _fwd(m.eval())
    assert "traj_v3" not in out and "traj_r7" in out


def test_scorer_gradient_never_reaches_the_proposal_head_but_reaches_the_trunk():
    m = _model().train()
    out = _fwd(m)
    loss = sum(v.float().mean() for v in out["r7_logits"].values())
    loss.backward()
    wta_g = sum(float(p.grad.abs().sum()) for p in m.refcv7_wta.parameters()
                if p.grad is not None)
    assert wta_g == 0.0                           # disentangled (stop-gradient)
    trunk_g = sum(float(p.grad.abs().sum()) for p in m.core.encoder.parameters()
                  if p.grad is not None)
    assert trunk_g > 0.0                          # the scorer shapes the shared trunk


def test_wta_gradient_reaches_its_head_and_the_trunk():
    m = _model().train()
    out = _fwd(m)
    out["r7_wta"].abs().mean().backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in m.refcv7_wta.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in m.core.encoder.parameters())


def test_ledger_lines_sum_to_the_total():
    m = _model()
    br = v3.param_breakdown_v3(m)
    assert br["refcv7_wta"] > 0 and br["refcv7_scorer"] > 0
    lines = sum(v for k, v in br.items() if k != "total")
    # the attached stand-in branch is not part of the v3 ledger
    pb = sum(p.numel() for p in m._perception.parameters())
    assert lines == br["total"] - pb


def _toad_model():
    torch.manual_seed(0)
    cfg = _cfg(True)
    cfg.refcv7_toad = True
    cfg.refcv7_toad_iters = 2
    cfg.refcv7_toad_samples = 8
    cfg.refcv7_toad_fit_steps = 20
    m = v3.RefCV3Model(cfg)
    m._perception = _FakeBranch(D_BEV)
    return m


def test_toad_runs_at_eval_and_never_scores_below_the_base():
    m = _toad_model().eval()
    with torch.no_grad():
        out = _fwd(m)
    assert "traj_toad" in out
    assert torch.equal(out["traj"], out["traj_toad"])     # it is the deployed pick
    # its own guarantee: the returned plan never scores below the base
    # the guarantee is against the ACTUAL pick's reward
    assert torch.all(out["r7_toad_reward"] >= out["r7_toad_base_reward"] - 1e-6)
    # ⚠️ the control-space inversion of an UNTRAINED, free-form proposal need not be
    # exact (measured ~2 m here) — which is why the comparison above is the guarantee
    # and this number is REPORTED, not asserted small
    assert float(out["r7_toad_fit_rms_m"]) >= 0.0


def test_toad_is_SKIPPED_in_training_mode():
    """⛔ A search inside the training loop would optimise the plan against a
    scorer still being fitted to it."""
    m = _toad_model().train()
    out = _fwd(m)
    assert "traj_toad" not in out


def test_toad_is_deterministic():
    """Same model, same input, same seed -> the same plan. (That a DIFFERENT seed
    moves the plan is asserted in `test_refcv7_toad.py`, where the reward is
    analytic and the search always fires; with an untrained scorer here the search
    usually returns the base, so a seed difference need not show.)"""
    m = _toad_model().eval()
    with torch.no_grad():
        a = _fwd(m)["traj_toad"]
        b = _fwd(m)["traj_toad"]
    assert torch.equal(a, b)
