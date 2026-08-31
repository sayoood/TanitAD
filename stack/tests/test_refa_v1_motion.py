"""Observed-motion injection (PI 2026-08-31: "implement and try 1").

⚠️ THE PREMISE WAS CORRECTED BY ITS OWN CONTROL TEST. The first draft claimed
the rollout state is "Markovian on one static frame"; the pin-the-defect test
FAILED against that claim, because ``WideAdapter.tmix`` is a temporal conv —
history does reach the state. The measured defect is narrower and still real:
tmix is DEPTHWISE, so each channel mixes only its own past, and cross-channel
temporal structure (parallax, an edge moving between patch channels — MOTION)
has no route in. The injection adds exactly that: a full cross-channel
projection of z_t − z_{t−1}. Hypothesis H-REFAV1-MOTION; the arm decides.
"""
import pytest
import torch

from tanitad.refs.refa_v1 import RefAV1, RefAV1Config


def _cfg(**kw) -> RefAV1Config:
    base = dict(d_enc=16, d_state=16, n_tokens=8,
                op_dt=0.2, op_steps=30, op_layers=1, op_heads=2, op_window=2,
                tac_dt=0.6, tac_steps=10, tac_queries=4, tac_layers=1,
                str_dt=3.0, str_steps=2, str_dim=8, str_layers=1)
    base.update(kw)
    return RefAV1Config(**base)


def _two_histories(c: RefAV1Config, seed=0):
    """Same LAST frame, different PREVIOUS frame — the discriminating input."""
    g = torch.Generator().manual_seed(seed)
    last = torch.randn(1, 1, c.n_tokens, c.d_enc, generator=g)
    prev_a = torch.randn(1, 1, c.n_tokens, c.d_enc, generator=g)
    prev_b = torch.randn(1, 1, c.n_tokens, c.d_enc, generator=g)
    return (torch.cat([prev_a, last], dim=1),
            torch.cat([prev_b, last], dim=1))


def test_the_DEFECT_history_reaches_the_state_only_DEPTHWISE():
    """⚠️ SELF-CORRECTION, kept on the record. The first draft of this test
    asserted the baseline ignores history entirely ("Markovian on one static
    frame") — and FAILED, because `WideAdapter.tmix` is a temporal conv: the
    control test caught the author's claim being too strong. The measured
    defect is narrower: tmix is DEPTHWISE (groups == d_state), so each channel
    mixes only its own past. Cross-channel temporal structure — parallax, an
    edge moving between patch channels, i.e. MOTION — has no route into the
    state. Both halves asserted."""
    c = _cfg()
    m = RefAV1(c).eval()
    # half 1: history DOES reach the state (tmix exists and is temporal)
    fa, fb = _two_histories(c)
    a_seq = torch.zeros(1, c.op_steps, c.a_dim)
    with torch.no_grad():
        pa = m(fa, a_seq)["op_pred"]
        pb = m(fb, a_seq)["op_pred"]
    assert not torch.equal(pa, pb), "tmix vanished — re-derive this file"
    # half 2: but that route is per-channel only — no cross-channel mixing
    assert m.adapter.tmix.groups == c.d_state, (
        "tmix grew cross-channel reach; the motion_inject premise is stale")
    assert m.adapter.tmix.kernel_size == (3,)


def _kill_tmix(m: RefAV1) -> None:
    """Silence the depthwise path so the injection is the ONLY route history
    can take — without this, every cross-history difference is confounded by
    tmix and the tests below would pass for the wrong reason."""
    with torch.no_grad():
        m.adapter.tmix.weight.zero_()
        if m.adapter.tmix.bias is not None:
            m.adapter.tmix.bias.zero_()


def test_with_tmix_silenced_ONLY_the_injection_carries_history():
    """⭐ The isolating pair: tmix zeroed in both arms. Baseline becomes
    history-blind (pinning that tmix was its only route); the motion arm still
    distinguishes histories — through the injection alone."""
    c_off, c_on = _cfg(), _cfg(motion_inject=True)
    a_seq = torch.zeros(1, c_off.op_steps, c_off.a_dim)
    base = RefAV1(c_off).eval()
    _kill_tmix(base)
    fa, fb = _two_histories(c_off)
    with torch.no_grad():
        assert torch.equal(base(fa, a_seq)["op_pred"],
                           base(fb, a_seq)["op_pred"])
    mot = RefAV1(c_on).eval()
    _kill_tmix(mot)
    with torch.no_grad():
        pa, pb = mot(fa, a_seq)["op_pred"], mot(fb, a_seq)["op_pred"]
    assert not torch.equal(pa, pb)


def test_all_three_levels_inherit_the_injected_state():
    """The injection lands BEFORE `_tac_field` and `subspace`, so tactical and
    strategic read the same motion-aware state — one state, three readers.
    tmix silenced so the injection is the only cross-history route."""
    c = _cfg(motion_inject=True)
    m = RefAV1(c).eval()
    _kill_tmix(m)
    fa, fb = _two_histories(c)
    a_seq = torch.zeros(1, c.op_steps, c.a_dim)
    with torch.no_grad():
        oa, ob = m(fa, a_seq), m(fb, a_seq)
    for k in ("tac_pred", "str_pred"):
        assert not torch.equal(oa[k], ob[k]), k


def test_default_off_adds_no_parameters_and_matches_old_state_dict():
    n_off = sum(p.numel() for p in RefAV1(_cfg()).parameters())
    n_on = sum(p.numel() for p in RefAV1(_cfg(motion_inject=True)).parameters())
    assert n_on > n_off                       # gated params exist only when on
    assert "motion_in.1.weight" not in RefAV1(_cfg()).state_dict()


def test_the_injection_starts_NEAR_zero_for_comparability():
    """Down-scaled init: at step 0 the motion arm is near-indistinguishable
    from baseline, so a training difference is attributable to what training
    MADE of the channel, not to an init-scale shock (the 535x residual-head
    lesson, applied in reverse)."""
    c = _cfg(motion_inject=True)
    m = RefAV1(c).eval()
    fa, _ = _two_histories(c)
    with torch.no_grad():
        field = m.encode(fa)
        inj = m._last_state(field) - field[:, -1]
        rel = float(inj.norm() / field[:, -1].norm())
    assert rel < 0.05, f"injection is {rel:.3f} of the state at init"


def test_plan_and_forward_share_the_injected_state():
    """`_last_state` is the single site — if plan() re-derived `field[:, -1]`
    itself, training and deployment would disagree about what the current
    state IS. Asserted on the AST (comments deliberately mention the old form,
    so a raw substring count would trip on its own documentation)."""
    import ast
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "tanitad" / "refs" / "refa_v1.py").read_text(encoding="utf-8")
    cls = next(n for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.ClassDef) and n.name == "RefAV1")
    fns = {n.name: ast.unparse(n) for n in cls.body
           if isinstance(n, ast.FunctionDef)
           and n.name in ("forward", "plan", "_last_state")}
    assert "field[:, -1]" in fns["_last_state"]
    for name in ("forward", "plan"):
        assert "field[:, -1]" not in fns[name], (
            f"{name} re-derives the last state instead of calling _last_state")
        assert "_last_state" in fns[name]


def test_motion_with_a_one_frame_window_is_REFUSED():
    with pytest.raises(ValueError, match="op_window >= 2"):
        _cfg(motion_inject=True, op_window=1).sanity()
