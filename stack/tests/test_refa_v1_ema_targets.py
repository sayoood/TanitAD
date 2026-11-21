"""EMA teacher for refav1's tactical/strategic targets (BYOL 2006.07733,
I-JEPA 2301.08243 — both banked in `TanitAD Research Lab/Library/`).

SPEC E-ARCH-TSC-1 §3 / RESULT R3 (2026-09-02): ``--target-space frozen`` pins
the OPERATIVE target only. ``tq = _tac_field(tgt)`` and ``st =
strategic.subspace(tgt)`` are derived from the TRAINED adapter in BOTH target
spaces, and ``detach_aux_targets`` stops their gradient but not their
shrinkage — a shrinking student still shrinks them. ``ema_targets`` derives
those two targets from a slow EMA copy of the whole target path instead.

⭐ THE LOAD-BEARING TEST does not argue this — it MEASURES it: collapse the
student and read ``tgt_std_tac`` / ``tgt_std_str`` in the same forward. With
the teacher on they do not move; with it off (the deliberate regression) the
same collapse drives both to exactly zero.

⛔ THE OFF-PATH IDENTITY TEST pins that a model built with ``ema_targets=False``
has the SAME state_dict keys and the SAME fixed-seed forward loss as the code
had BEFORE the teacher existed (values computed 2026-09-02 on the pre-edit
`refa_v1.py`, torch 2.11.0+cu128, CPU) — the live Thor run trains from a
frozen copy of that code and its checkpoint must load later.
"""
import hashlib

import pytest
import torch

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.refs.refa_v1 import RefAV1, RefAV1Config


def _cfg(**kw) -> RefAV1Config:
    """Same small no-hierarchy config as `test_refa_v1_target_space._cfg`."""
    base = dict(d_enc=16, d_state=16, n_tokens=8,
                op_dt=0.2, op_steps=30, op_layers=1, op_heads=2, op_window=2,
                tac_dt=0.6, tac_steps=10, tac_queries=4, tac_layers=1,
                str_dt=3.0, str_steps=2, str_dim=8, str_layers=1)
    base.update(kw)
    return RefAV1Config(**base)


def _hier_cfg() -> RefAV1Config:
    """EXACTLY the trainer's --smoke configuration (`refa_v1_train.build_model`),
    so the pinned value below is the value that path produces."""
    c = RefAV1Config(strategic_cfg=StrategicPolicyConfig(d_model=32, depth=1,
                                                         n_heads=2, d_ctx=16,
                                                         d_cmd=8),
                     tactical_cfg=TacticalPolicyConfig(d_model=32, depth=1,
                                                       n_heads=2, d_intent=16))
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


def _forward(m: RefAV1, f, a, fut, fit) -> dict:
    if not bool(m.std.fitted):
        m.std.fit(fit)
    with torch.no_grad():
        return m(f, a, future_feats=fut)


def _collapse_the_adapter(m: RefAV1) -> None:
    """`test_refa_v1_target_space._collapse_the_adapter`, verbatim: every
    trained path into the field zeroed, plus the residual delta heads."""
    with torch.no_grad():
        for p in m.adapter.parameters():
            p.zero_()
        for pred in (m.operative, m.tactical):
            for p in pred.head.parameters():
                p.zero_()


def _collapse_the_target_path(m: RefAV1) -> None:
    """Stronger: ALSO zero the other trained modules on the target side
    (tac_queries, tac_pool, strategic.read) — the teacher must cover them
    too, or the target still follows the student through the pool."""
    _collapse_the_adapter(m)
    with torch.no_grad():
        m.tac_queries.zero_()
        for p in m.tac_pool.parameters():
            p.zero_()
        for p in m.strategic.read.parameters():
            p.zero_()


# --------------------------------------------------------------- (1) frozen --
def test_ema_params_are_frozen_and_outside_the_trainable_set():
    m = RefAV1(_cfg(ema_targets=True))
    ema = list(m.ema.parameters())
    assert ema, "the teacher holds no parameters"
    assert all(p.requires_grad is False for p in ema)
    ema_ids = {id(p) for p in ema}
    trainable = [p for p in m.parameters() if p.requires_grad]
    assert not any(id(p) in ema_ids for p in trainable)
    assert m.trainable_parameters() == sum(p.numel() for p in trainable)


def test_the_teacher_covers_the_WHOLE_target_path_and_nothing_else():
    m = RefAV1(_cfg(ema_targets=True))
    names = {n for n, _ in m.ema.named_parameters()}
    assert any(n.startswith("adapter.") for n in names)
    assert "tac_queries" in names
    assert any(n.startswith("tac_pool.") for n in names)
    assert any(n.startswith("str_read.") for n in names)
    # every teacher parameter is paired with exactly one student parameter
    pairs = m.ema.pairs(m)
    assert len(pairs) == len(list(m.ema.parameters()))
    assert all(p_e.shape == p_s.shape for p_e, p_s in pairs)
    assert all(p_e is not p_s for p_e, p_s in pairs)
    # the predictors are NOT copied — the rollout-vs-target asymmetry stays
    assert not hasattr(m.ema, "operative") and not hasattr(m.ema, "tactical")


# ------------------------------------------------- (2) THE LOAD-BEARING ONE --
def test_THE_LOAD_BEARING_ONE_collapsing_the_student_does_not_move_the_targets():
    """⭐⭐ Same forward, same inputs, student zeroed in between. With the
    teacher the target-scale instruments read IDENTICAL values; without it
    (the deliberate regression) the same collapse takes both to zero."""
    c = _cfg(ema_targets=True)
    torch.manual_seed(0)
    m = RefAV1(c)
    f, a, fut, fit = _inputs(c)
    before = _forward(m, f, a, fut, fit)
    assert before["tgt_std_tac"] > 1e-3 and before["tgt_std_str"] > 1e-3
    _collapse_the_target_path(m)
    with torch.no_grad():                                # the student DID collapse
        assert float(m.encode(f).abs().max()) == 0.0
    after = _forward(m, f, a, fut, fit)
    assert after["tgt_std_tac"] == before["tgt_std_tac"]
    assert after["tgt_std_str"] == before["tgt_std_str"]
    assert after["tgt_std_op"] == before["tgt_std_op"]   # frozen, as before

    # --- deliberate regression: teacher OFF, identical collapse ------------
    c0 = _cfg(ema_targets=False)
    torch.manual_seed(0)
    m0 = RefAV1(c0)
    b0 = _forward(m0, f, a, fut, fit)
    assert b0["tgt_std_tac"] > 1e-3 and b0["tgt_std_str"] > 1e-3
    _collapse_the_target_path(m0)
    a0 = _forward(m0, f, a, fut, fit)
    assert a0["tgt_std_tac"] == 0.0 and a0["tgt_std_str"] == 0.0


def test_the_ema_targets_carry_no_graph_even_with_detach_off():
    """The teacher is a stop-gradient by construction (requires_grad=False on
    every parameter, targets built under no_grad) — `detach_aux_targets` is
    not what keeps gradient off it."""
    c = _cfg(ema_targets=True, detach_aux_targets=False)
    m = RefAV1(c)
    f, a, fut, fit = _inputs(c)
    m.std.fit(fit)
    out = m(f, a, future_feats=fut)
    out["loss"].backward()
    assert all(p.grad is None for p in m.ema.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in m.adapter.parameters())
    assert torch.isfinite(out["loss"].detach())


# ---------------------------------------------------------- (3) the update --
def test_one_ema_update_moves_each_teacher_param_by_exactly_one_minus_decay():
    m = RefAV1(_cfg(ema_targets=True, ema_decay=0.9, ema_decay_end=0.9))
    pairs = m.ema.pairs(m)
    with torch.no_grad():                      # open a gap on every student param
        for _, p_s in pairs:
            p_s.add_(torch.randn_like(p_s))
    before = [p_e.clone() for p_e, _ in pairs]
    students = [p_s.clone() for _, p_s in pairs]
    decay = m.ema_update(step=1, total_steps=10)
    assert decay == pytest.approx(0.9)
    for (p_e, p_s), b, s in zip(pairs, before, students):
        assert torch.allclose(p_e, b + (1.0 - 0.9) * (s - b), atol=1e-6)
        assert torch.equal(p_s, s)             # the student is never touched
        assert p_e.requires_grad is False


# -------------------------------------------------------- (4) the schedule --
def test_the_schedule_is_linear_and_reaches_decay_end_on_the_last_step():
    m = RefAV1(_cfg(ema_targets=True, ema_decay=0.996, ema_decay_end=0.999))
    assert m.ema_decay_at(0, 1000) == pytest.approx(0.996)
    assert m.ema_decay_at(500, 1000) == pytest.approx(0.9975)
    assert m.ema_decay_at(1000, 1000) == pytest.approx(0.999)
    ds = [m.ema_decay_at(s, 1000) for s in range(0, 1001, 50)]
    assert ds == sorted(ds)
    # the trainer's last call is (steps, steps): returns decay_end exactly
    assert m.ema_update(1000, 1000) == pytest.approx(0.999)
    assert m.ema_decay_at(5, 0) == pytest.approx(0.996)     # degenerate run


def test_ema_update_refuses_when_the_teacher_is_off():
    m = RefAV1(_cfg())
    assert m.ema is None
    with pytest.raises(RuntimeError, match="ema_targets=False"):
        m.ema_update(1, 10)


def test_a_backwards_schedule_is_refused_and_the_default_is_off():
    with pytest.raises(ValueError, match="ema_decay"):
        _cfg(ema_targets=True, ema_decay=0.999, ema_decay_end=0.9).sanity()
    assert RefAV1Config().ema_targets is False


def test_the_teacher_lives_in_the_state_dict_when_on():
    """A resumed run must restore its teacher, not re-copy a stale student."""
    m = RefAV1(_cfg(ema_targets=True))
    keys = list(m.state_dict())
    assert any(k.startswith("ema.adapter.") for k in keys)
    assert "ema.tac_queries" in keys
    assert any(k.startswith("ema.tac_pool.") for k in keys)
    assert any(k.startswith("ema.str_read.") for k in keys)


# ------------------------------------------------- (5) OFF-path identity --
# Computed 2026-09-02 by `scratchpad/baseline_identity.py` against the PRE-EDIT
# refa_v1.py (commit 48c3e58 tree), torch 2.11.0+cu128, CPU float32, and
# reproduced bit-identically on a second run. `sha` is sha256 over the sorted
# state_dict key names joined by newline.
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
    assert m.ema is None
    keys = sorted(m.state_dict())
    assert not any("ema" in k for k in keys)
    assert len(keys) == spec["n_keys"]
    assert hashlib.sha256("\n".join(keys).encode()).hexdigest() == spec["sha"]
    f, a, fut, fit = _inputs(m.cfg, seed=0)
    out = _forward(m, f, a, fut, fit)
    assert float(out["loss"]) == pytest.approx(spec["loss"], rel=1e-6, abs=0.0)
