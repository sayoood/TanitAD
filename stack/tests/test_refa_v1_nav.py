"""Nav injection into the tactical policy and the operative layer (PI 2026-09-01).

The PI's decision after the echo concern was raised: nav reaches the tactical
policy DIRECTLY (added to its FiLM cond) and the operative/tactical FIELD
predictors (added to the intent conditioning them). The strategic subspace
predictor stays nav-free. The chain path (nav → StrategicPolicy → ctx) already
existed — so every test that wants to see the NEW path must first SILENCE the
old one, or it measures the chain and calls it the injection.
"""
import pytest
import torch

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.refs.refa_v1 import RefAV1, RefAV1Config


def _cfg(**kw) -> RefAV1Config:
    base = dict(tac_vocab_version="v6.0", d_enc=16, d_state=16, n_tokens=8,
                op_dt=0.2, op_steps=30, op_layers=1, op_heads=2, op_window=2,
                tac_dt=0.6, tac_steps=10, tac_queries=4, tac_layers=1,
                str_dt=3.0, str_steps=2, str_dim=8, str_layers=1,
                strategic_cfg=StrategicPolicyConfig(d_model=16, depth=1,
                                                    n_heads=2, d_ctx=8,
                                                    d_cmd=8),
                tactical_cfg=TacticalPolicyConfig(d_model=16, depth=1,
                                                  n_heads=2, d_intent=8))
    base.update(kw)
    return RefAV1Config(**base)


def _silence_chain_nav(m: RefAV1) -> None:
    """Zero the strategic policy's OWN nav embedding, so the only remaining
    nav-sensitive path is the direct injection under test."""
    with torch.no_grad():
        m.strategic_policy.nav_emb.weight.zero_()


def _outs(m: RefAV1, nav: int, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = m.cfg
    f = torch.randn(2, c.op_window, c.n_tokens, c.d_enc, generator=g)
    a = torch.randn(2, c.op_steps, c.a_dim, generator=g)
    with torch.no_grad():
        return m(f, a, nav_cmd=torch.full((2,), nav, dtype=torch.long))


def test_with_the_chain_silenced_nav_still_reaches_operative_and_tactical():
    """⭐ The isolating test: chain nav zeroed. nav_inject=False → nav changes
    NOTHING (pinning that the chain was the only old path). nav_inject=True →
    nav still moves the operative and tactical rollouts, through the injection
    alone. To make the near-zero-init injection visible at test time, its
    projections are re-scaled to O(1) first — the test is about ROUTING, not
    about init magnitude (that has its own test)."""
    base = RefAV1(_cfg(nav_inject=False)).eval()
    _silence_chain_nav(base)
    a0, a1 = _outs(base, 0), _outs(base, 1)
    for k in ("op_pred", "tac_pred"):
        assert torch.equal(a0[k], a1[k]), f"{k}: a nav path exists besides " \
                                          "the chain and the injection"
    m = RefAV1(_cfg(nav_inject=True)).eval()
    _silence_chain_nav(m)
    with torch.no_grad():
        for lin in (m.nav_to_ctx, m.nav_to_intent):
            lin.weight.data.normal_(0, 0.5)
    b0, b1 = _outs(m, 0), _outs(m, 1)
    for k in ("op_pred", "tac_pred"):
        assert not torch.equal(b0[k], b1[k]), k


def test_the_strategic_subspace_predictor_stays_nav_free():
    """⛔ Deliberate: the strategic prediction stays independently falsifiable.
    Same silenced-chain setup, injection live — str_pred must not move."""
    m = RefAV1(_cfg(nav_inject=True)).eval()
    _silence_chain_nav(m)
    with torch.no_grad():
        for lin in (m.nav_to_ctx, m.nav_to_intent):
            lin.weight.data.normal_(0, 0.5)
    b0, b1 = _outs(m, 0), _outs(m, 1)
    assert torch.equal(b0["str_pred"], b1["str_pred"])


def test_injection_params_exist_only_with_brains_and_flag():
    assert RefAV1(_cfg(nav_inject=True)).nav_inj_emb is not None
    assert RefAV1(_cfg(nav_inject=False)).nav_inj_emb is None
    assert RefAV1(_cfg(nav_inject=True, strategic_cfg=None,
                       tactical_cfg=None)).nav_inj_emb is None


def test_the_injection_starts_near_zero():
    """Down-scaled init: the decision arms start ≈ pre-decision behaviour, so
    a trained difference is attributable to training, not an init shock."""
    m = RefAV1(_cfg(nav_inject=True)).eval()
    _silence_chain_nav(m)
    b0, b1 = _outs(m, 0), _outs(m, 1)
    rel = float((b0["op_pred"] - b1["op_pred"]).norm()
                / b0["op_pred"].norm().clamp_min(1e-9))
    assert rel < 0.05, f"injection moves the rollout by {rel:.3f} at init"


def test_forward_and_plan_share_one_brains_site():
    """The conditioning chain lives in `_run_brains` and ONLY there — the
    defect this prevents is a nav change applying at training and silently
    not at deployment (or vice versa)."""
    import ast
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "tanitad" / "refs" / "refa_v1.py").read_text(encoding="utf-8")
    cls = next(n for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.ClassDef) and n.name == "RefAV1")
    fns = {n.name: ast.unparse(n) for n in cls.body
           if isinstance(n, ast.FunctionDef)}
    for name in ("forward", "plan"):
        assert "_run_brains" in fns[name], name
        assert "self.strategic_policy(" not in fns[name], (
            f"{name} runs the brains inline — the single-site rule is broken")
        assert "self.tactical_policy(" not in fns[name]


def test_emitted_ctx_is_the_augmented_cond_the_tactical_policy_consumed():
    """`out["ctx"]` must be what FiLM actually saw. With injection scaled up
    and the chain silenced, ctx must differ across nav values."""
    m = RefAV1(_cfg(nav_inject=True)).eval()
    _silence_chain_nav(m)
    with torch.no_grad():
        m.nav_to_ctx.weight.data.normal_(0, 0.5)
    b0, b1 = _outs(m, 0), _outs(m, 1)
    assert not torch.equal(b0["ctx"], b1["ctx"])
