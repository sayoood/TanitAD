"""R2 (PI 2026-09-27): the navigation command reaches the tactical AND operative layers at training — and a real
``--nav-cond`` v7F run must be able to START. Two defects were MEASURED at tip ``c36b6ddd`` by the R1/R4 stream
(``products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_r1r4/RESULT.md`` s7):

1. BUILD: ``V6Stack.synthetic_batch`` carried no nav keys, and ``assert_isolation`` — which every trainer build
   runs on that batch — therefore raised ``NavTokenMissing`` whenever ``nav_cond`` was on.
2. STEP 1: ``train()``'s batch dict splatted the NavEmitter output and THEN wrote ``"nav_token": b.get(...)``;
   in a dict literal the LATER key wins, so every emitted token was replaced by ``None``.

Every expectation below is a literal or a known value; each fix has a regression arm that must go RED.
"""
from __future__ import annotations

import copy
import inspect
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.models.nav_conditioning import NavTokenMissing  # noqa: E402
from tanitad.models.v6 import V6Config, V6Stack  # noqa: E402


def _cfg(**kw) -> V6Config:
    base = dict(
        encoder=EncoderConfig(in_channels=3, image_size=32, image_width=32,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                  horizons=(1,), action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32,
        f_hidden_tac=32, f_hidden_str=32, d_plan_feat=16, emission_hidden=16,
        n_candidates=4, aux_hidden=16, sigreg_slices=8)
    base.update(kw)
    return V6Config(**base)


def _build(cfg: V6Config, seed: int = 0) -> V6Stack:
    torch.manual_seed(seed)
    return V6Stack(copy.deepcopy(cfg))


# --------------------------------------------------------------------------- #
# 1. BUILD — the synthetic contract carries the mandatory nav channel
# --------------------------------------------------------------------------- #

def test_nav_stack_builds_and_passes_isolation():
    """The fixed build: a nav-conditioned stack's own isolation probe runs to completion."""
    s = _build(_cfg(nav_cond=True))
    assert s.nav is not None
    b = s.synthetic_batch(batch=2, seed=0)
    assert b["nav_token"].dtype == torch.int64 and tuple(b["nav_token"].shape) == (2,)
    assert tuple(b["nav_args"].shape) == (2, 2)
    assert int(b["nav_token"].min()) >= 0 and int(b["nav_token"].max()) < len(s.nav.tokens)
    s.assert_isolation(b)                      # raised NavTokenMissing at tip c36b6ddd


def test_REGRESSION_ARM_without_nav_keys_the_build_probe_refuses():
    """The defect, reproduced: the same probe on a batch WITHOUT the nav keys must raise by name."""
    s = _build(_cfg(nav_cond=True))
    b = s.synthetic_batch(batch=2, seed=0)
    b.pop("nav_token"); b.pop("nav_args")
    with pytest.raises(NavTokenMissing):
        s.assert_isolation(b)


def test_nav_keys_are_drawn_LAST_so_every_existing_batch_is_byte_identical():
    """nav on vs off: every pre-existing key is bit-identical (same seed, same RNG stream up to the nav draw),
    and a stack WITHOUT nav gets no nav key at all."""
    on = _build(_cfg(nav_cond=True)).synthetic_batch(batch=3, seed=5)
    off = _build(_cfg(nav_cond=False)).synthetic_batch(batch=3, seed=5)
    assert "nav_token" not in off and "nav_args" not in off
    assert set(on) - set(off) == {"nav_token", "nav_args"}
    for k in off:
        assert torch.equal(on[k], off[k]), k


# --------------------------------------------------------------------------- #
# 2. STEP 1 — the emitted nav token must survive the batch dict literal
# --------------------------------------------------------------------------- #

def _train_src() -> str:
    import train_v6_staged as T                                    # noqa: E402
    return inspect.getsource(T.train)


def test_the_nav_emitter_splat_comes_AFTER_the_get_forwarding():
    src = _train_src()
    # rfind: the LAST `.get` forwarding must precede the emitter -- a duplicate `.get` line re-added AFTER the
    # splat recreates the defect, and a first-occurrence check would pass it.
    i_get = src.rfind('"nav_token": b.get("nav_token")')
    i_emit = src.find('nav_emitter(b["ep_idx"]')
    assert i_get != -1 and i_emit != -1, "both nav sites must still exist in train()"
    assert i_get < i_emit, ("the NavEmitter splat sits ABOVE the `.get` forwarding: in a dict literal the later "
                            "key wins, so every emitted nav token is overwritten by None (the step-1 defect)")


def test_REGRESSION_ARM_the_old_order_really_drops_the_emitted_token():
    """Known value of the Python rule the fix relies on, in the exact shape of the trainer's literal."""
    b = {}                                      # a dataset item with no nav token, as on the real corpus
    emitted = {"nav_token": torch.tensor([2]), "nav_args": torch.zeros(1, 2)}
    old = {**emitted, "nav_token": b.get("nav_token"), "nav_args": b.get("nav_args")}
    new = {"nav_token": b.get("nav_token"), "nav_args": b.get("nav_args"), **emitted}
    assert old["nav_token"] is None             # the defect
    assert int(new["nav_token"]) == 2           # the fix


# --------------------------------------------------------------------------- #
# 3. BUILD — every nav parameter belongs to a stage group (it raised in group_of)
# --------------------------------------------------------------------------- #

EXPECTED_NAV_GROUPS = {
    "nav.embed.weight": "predictor_op", "nav.arg_proj.weight": "predictor_op",
    "nav.arg_proj.bias": "predictor_op",
    "nav.layer_proj.operative.weight": "predictor_op", "nav.layer_proj.operative.bias": "predictor_op",
    "nav.gate.operative": "predictor_op",
    "nav.layer_proj.tactical.weight": "layer_tac", "nav.layer_proj.tactical.bias": "layer_tac",
    "nav.gate.tactical": "layer_tac",
    "nav.layer_proj.strategic.weight": "layer_str", "nav.layer_proj.strategic.bias": "layer_str",
    "nav.gate.strategic": "layer_str",
}


def test_every_nav_parameter_has_the_designed_group():
    s = _build(_cfg(nav_cond=True))
    got = {n: s.group_of(n) for n, _ in s.named_parameters() if n.startswith("nav.")}
    assert got == EXPECTED_NAV_GROUPS


def test_REGRESSION_ARM_an_ungrouped_nav_parameter_still_raises(monkeypatch):
    s = _build(_cfg(nav_cond=True))
    kept = tuple(e for e in V6Stack._GROUP_PREFIXES if not e[0].startswith("nav.embed."))
    monkeypatch.setattr(V6Stack, "_GROUP_PREFIXES", kept)
    with pytest.raises(KeyError):
        s.group_of("nav.embed.weight")


# --------------------------------------------------------------------------- #
# 4. THE REAL LAUNCH PATH -- argv -> build_parser -> build_stack_from_args
# --------------------------------------------------------------------------- #
# MEASURED 2026-09-27 by the v7f launch-gate profile (G-DVB): `--nav-cond` passed preflight but
# build_stack_from_args never mapped it into V6Config, so every v7-line launch built a stack with NO
# nav conditioner -- and sections 1-3 above could not see it, because they build configs DIRECTLY.
# These tests go through the trainer's own parser and builder (production default geometry; the
# builder runs assert_isolation on synthetic_batch, so fixes 1-3 are exercised on this path too).

_BASE_ARGV = ["--stage", "S-W", "--out", "x", "--v2-cache", "y"]


def _real_build(extra):
    import train_v6_staged as T                                    # noqa: E402
    a = T.build_parser().parse_args(_BASE_ARGV + extra)
    return T.build_stack_from_args(a)


def test_REAL_LAUNCH_nav_cond_builds_the_conditioner():
    s = _real_build(["--nav-cond", "--nav-labels", "z"])
    assert s.cfg.nav_cond is True
    assert s.nav is not None
    assert {n for n, _ in s.named_parameters() if n.startswith("nav.")} == set(EXPECTED_NAV_GROUPS)


def test_REAL_LAUNCH_without_the_flag_there_is_no_conditioner():
    s = _real_build([])
    assert s.cfg.nav_cond is False and s.nav is None


# --------------------------------------------------------------------------- #
# 5. F7 + F8 -- found by the v7f launch gate once nav was really built
# --------------------------------------------------------------------------- #
# F7: the S-W world-model losses roll predictor_op through SHARED helpers (stage_a_losses,
# rollout_transitions) that never forwarded nav, so the operative nav tensors got NO gradient in S-W
# (and S-T freezes their group). F8: synthetic_train_batch had no nav keys, so --dry-run crashed.

_OPERATIVE_NAV = ("nav.embed.weight", "nav.arg_proj.weight", "nav.arg_proj.bias",
                  "nav.layer_proj.operative.weight", "nav.layer_proj.operative.bias", "nav.gate.operative")


def _sw_loss(stack):
    import train_v6_staged as T                                    # noqa: E402
    b = T.synthetic_train_batch(stack, batch=2, k=12, seed=1)
    b["gt_wp"] = torch.randn(2, 10, 2, generator=torch.Generator().manual_seed(1))
    torch.manual_seed(3)
    return T.v6_loss_step(stack, b, stage="S-W", weights=T.V6LossWeights(), o1_k=10, o5_k=12,
                          generator=torch.Generator().manual_seed(11))["loss"]


def _wake(stack):
    """FiLM (`predictor_op.*.film.to_scale_shift`) and the nav projection are ZERO-init by design, so at step 0 NO
    conditioning term -- actions included -- receives gradient. Waking both (small random, as a few real steps would)
    is what makes the nav path's gradient observable; with nav NOT forwarded it stays dead regardless."""
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for n, p in stack.named_parameters():
            if ".film.to_scale_shift." in n or n.startswith("nav.layer_proj.operative."):
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return stack


def _reached(stack, loss, names):
    params = dict(stack.named_parameters())
    grads = torch.autograd.grad(loss, [params[n] for n in names], allow_unused=True)
    return {n for n, g in zip(names, grads) if g is not None and float(g.abs().sum()) > 0.0}


def test_F8_the_dry_run_batch_carries_nav_when_the_stack_has_it():
    import train_v6_staged as T                                    # noqa: E402
    on = T.synthetic_train_batch(_build(_cfg(nav_cond=True)), batch=2, k=12, seed=1)
    off = T.synthetic_train_batch(_build(_cfg()), batch=2, k=12, seed=1)
    assert on["nav_token"].dtype == torch.int64 and tuple(on["nav_args"].shape) == (2, 2)
    assert "nav_token" not in off
    for k in off:                                                  # drawn LAST: pre-existing keys unchanged
        a, b = off[k], on[k]
        assert (torch.equal(a, b) if isinstance(a, torch.Tensor) else
                all(torch.equal(x, y) for x, y in zip(a, b))), k


def test_F7_the_S_W_loss_trains_the_OPERATIVE_nav_path():
    s = _wake(_build(_cfg(nav_cond=True)))
    assert _reached(s, _sw_loss(s), _OPERATIVE_NAV) == set(_OPERATIVE_NAV)


def test_F7_REGRESSION_ARM_without_the_binding_the_operative_nav_path_is_dead(monkeypatch):
    import train_v6_staged as T                                    # noqa: E402

    class _Unbound(T._NavBoundPredictor):                          # the tip behaviour: nav never forwarded
        def __call__(self, *a, **kw):
            return self._p(*a, **kw)
    monkeypatch.setattr(T, "_NavBoundPredictor", _Unbound)
    s = _wake(_build(_cfg(nav_cond=True)))
    reached = _reached(s, _sw_loss(s), _OPERATIVE_NAV)
    assert reached == set(), reached
