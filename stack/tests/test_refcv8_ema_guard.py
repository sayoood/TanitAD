"""refcv8 R8-WARMUP -- the weight EMA (``--ema-decay``) and the B1 guard (``--b1-guard-ref-ap2m`` / ``--b1-guard-step``).

THE RULING (``Project Steering/SPEC_REFCV8_DRAFT.md`` R8-WARMUP + the B1 entry rule): re-warm both LR groups, cosine
over the run, **EMA 0.9998 evaluated at every in-run eval and saved**, and the guard *"if the EMA box3d AP@2 m on the
in-run eval set at refcv8 step 1,000 is below refcv7-50,400's value on the same set, the head re-warm is cut to the
trunk's 5e-5."* ``grep -n ema`` found nothing in the trainer at the tip; this is the implementation and its pins.

Every pin has a DELIBERATE-REGRESSION arm that must go RED (a named ``test_DELIBERATE_REGRESSION_*`` or an in-test
mutant): the EMA update skipped, a swap that does not restore, a guard with the comparison flipped, a pin deleted.
Expectations are LITERALS (the analytic values are written out, never recomputed from the code under test).

  * ``WeightEMA`` -- analytic decay on a constant parameter (exact in binary at d = 0.5; 0.9998 / 1000 steps within
    5e-6), what is and is not shadowed, the memory it costs, the swap restoring live weights BIT FOR BIT (fp32, a bf16
    parameter, a tied weight, an exception inside the block), refusals while swapped in, the save / resume round trip.
  * the guard -- the decision on literal values (strict ``<``), the cut on a REAL AdamW pair of groups through the
    trainer's own ``apply_lr_schedule``, decided once, refusal when the metric is missing.
  * the trainer, on the 1-step synthetic ``train()`` rig of ``test_refc_v3_save_before_eval``: ckpt / model-only EMA /
    metrics keys, the live weights bit-identical with and without the EMA (so an EMA eval never perturbs training),
    the EMA row equal to the live row when the EMA IS the live weights, resume refusals both ways, the guard end to end
    (cut / keep / decided once / survives a resume), the startup pin refusals, and ``--ema-decay 0`` = the tip's
    checkpoint, rows and summary key for key.
CPU-only, synthetic data; nothing touches a pod.
"""
from __future__ import annotations

import copy
import inspect
import json
import math
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _refcv8_rig  # noqa: E402,F401  (puts scripts/ on sys.path)
import test_refc_v3_save_before_eval as RIG  # noqa: E402  the synthetic 1-step train() rig
from tanitad.train import weight_ema as W  # noqa: E402

T = RIG.T


# ====================================================================== toy models
class _One(nn.Module):
    def __init__(self, v=(1.0, -2.0)):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(list(v)))


class _Toy(nn.Module):
    """7 shadowed entries: lin.weight(6) lin.bias(2) bn.weight(2) bn.bias(2) bn.running_mean(2) bn.running_var(2)
    const_buf(2) = 18 values. NOT shadowed: the frozen parameter and the integer bn.num_batches_tracked."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 2)
        self.bn = nn.BatchNorm1d(2)
        self.frozen = nn.Parameter(torch.ones(2), requires_grad=False)
        self.register_buffer("const_buf", torch.full((2,), 0.1))

    def forward(self, x):
        return self.bn(self.lin(x)) + self.frozen + self.const_buf


TOY_SHADOWED = {"lin.weight", "lin.bias", "bn.weight", "bn.bias", "bn.running_mean", "bn.running_var", "const_buf"}


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """dtype, shape and every BIT equal (so -0.0 != 0.0 and NaN == NaN with the same payload)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(a.detach().reshape(-1).contiguous().view(torch.uint8),
                       b.detach().reshape(-1).contiguous().view(torch.uint8))


def _sd_bits_equal(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(_bits_equal(a[k], b[k]) for k in a)


def _snap(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _set(model, value: float) -> None:
    with torch.no_grad():
        model.p.fill_(value)


# ====================================================================== WeightEMA: analytic decay
def _after(n: int, decay: float, live_value, e0, *, skip_last: bool = False) -> torch.Tensor:
    m = _One(e0)
    ema = W.WeightEMA(m, decay)
    with torch.no_grad():
        m.p.copy_(torch.tensor(live_value))
    for i in range(n):
        if skip_last and i == n - 1:
            continue                                  # the DELIBERATE REGRESSION: one update lost
        ema.update(m)
    return ema.shadow["p"].clone()


def test_analytic_decay_exact_in_binary_at_d_half():
    """p + (e0 - p) * d**n with d = 1/2, n = 5: 3 + (1-3)/32 = 2.9375 and 5 + (-2-5)/32 = 4.78125 - both exact in fp32."""
    got = _after(5, 0.5, [3.0, 5.0], [1.0, -2.0])
    assert torch.equal(got, torch.tensor([2.9375, 4.78125]))


def test_analytic_decay_at_the_ruling_value_0p9998_over_1000_steps():
    """3 + (1-3) * 0.9998**1000 = 1.362571247113765 and -2 + (0.5+2) * 0.9998**1000 = 0.04678594110779377 (float64
    literals); the fp32 recursion must land within 5e-6."""
    got = _after(1000, 0.9998, [3.0, -2.0], [1.0, 0.5])
    assert abs(float(got[0]) - 1.362571247113765) < 5e-6
    # second element: live -2, e0 0.5 -> -2 + 2.5 * 0.8187143764431175 = 0.04678594110779377
    assert abs(float(got[1]) - 0.04678594110779377) < 5e-6


def test_DELIBERATE_REGRESSION_a_skipped_update_goes_RED():
    ok = _after(1000, 0.9998, [3.0, -2.0], [1.0, 0.5])
    bad = _after(1000, 0.9998, [3.0, -2.0], [1.0, 0.5], skip_last=True)
    target = 1.362571247113765
    assert abs(float(ok[0]) - target) < 5e-6
    assert abs(float(bad[0]) - target) > 2e-4          # one lost update moves the EMA by ~3.3e-4: far outside the bar


def test_a_constant_tensor_stays_EXACTLY_constant():
    """lerp(x, x, w) == x bit for bit. d*x + (1-d)*x can move by an ulp; 50k updates would drift buffers that never change."""
    m = _Toy()
    ema = W.WeightEMA(m, 0.9998)
    before = {k: v.clone() for k, v in ema.shadow.items()}
    for _ in range(300):
        ema.update(m)
    assert all(_bits_equal(before[k], ema.shadow[k]) for k in before)
    assert ema.num_updates == 300


def test_what_is_shadowed_and_what_it_costs():
    m = _Toy()
    ema = W.WeightEMA(m, 0.5)
    assert set(ema.shadow) == TOY_SHADOWED                      # no frozen param, no int buffer
    assert ema.numel == 18 and ema.nbytes == 72                # 18 values x 4 B
    assert all(s.dtype == torch.float32 for s in ema.shadow.values())


def test_the_shadow_is_fp32_even_for_a_bf16_parameter():
    m = _One()
    m.to(torch.bfloat16)
    ema = W.WeightEMA(m, 0.5)
    assert ema.shadow["p"].dtype == torch.float32 and ema.nbytes == 8


def test_a_tied_weight_is_shadowed_once():
    m = nn.Module()
    m.a = nn.Linear(2, 2, bias=False)
    m.b = nn.Linear(2, 2, bias=False)
    m.b.weight = m.a.weight                                     # tied: one object under two names
    ema = W.WeightEMA(m, 0.5)
    assert list(ema.shadow) == ["a.weight"] and ema.numel == 4


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_decay_outside_the_open_unit_interval_refuses(bad):
    with pytest.raises(W.EMAStateError, match="must be in"):
        W.WeightEMA(_One(), bad)


def test_update_refuses_when_the_trainable_set_changed():
    m = _Toy()
    ema = W.WeightEMA(m, 0.5)
    m.frozen.requires_grad_(True)                                # a param was UNFROZEN after the EMA was built
    with pytest.raises(W.EMAStateError, match="not the ones the EMA was built from"):
        ema.update(m)


# ====================================================================== the swap
def _randomised_toy(seed=0):
    torch.manual_seed(seed)
    m = _Toy()
    m.train()
    m(torch.randn(8, 3))                                         # moves the BN running stats and num_batches_tracked
    return m


def _drive(ema, m, n=7):
    for i in range(n):
        with torch.no_grad():
            for p in m.parameters():
                if p.requires_grad:
                    p.add_(0.01 * (i + 1))
        ema.update(m)


def test_swap_puts_the_EMA_in_and_restores_the_live_weights_BIT_FOR_BIT():
    m = _randomised_toy()
    ema = W.WeightEMA(m, 0.9)
    _drive(ema, m)
    live = _snap(m)
    shadow = {k: v.clone() for k, v in ema.shadow.items()}
    with ema.swapped_in(m):
        assert ema.swapped
        inside = _snap(m)
        for k in TOY_SHADOWED:                                   # EMA values inside ...
            assert _bits_equal(inside[k], shadow[k]), k
        assert _bits_equal(inside["frozen"], live["frozen"])     # ... frozen and integer entries untouched
        assert _bits_equal(inside["bn.num_batches_tracked"], live["bn.num_batches_tracked"])
        assert any(not _bits_equal(inside[k], live[k]) for k in TOY_SHADOWED)        # and they really differ
    assert not ema.swapped
    assert _sd_bits_equal(_snap(m), live)                        # live weights back, every entry
    assert all(_bits_equal(ema.shadow[k], shadow[k]) for k in shadow)               # and the EMA is still the EMA


def test_DELIBERATE_REGRESSION_a_swap_that_does_not_restore_goes_RED(monkeypatch):
    m = _randomised_toy()
    ema = W.WeightEMA(m, 0.9)
    _drive(ema, m)
    live = _snap(m)

    def _forget(self, model):                                    # clears the flag, restores nothing
        self._stash, self._swapped = {}, False

    monkeypatch.setattr(W.WeightEMA, "swap_out", _forget)
    with ema.swapped_in(m):
        pass
    assert not _sd_bits_equal(_snap(m), live)                    # the pin above would FAIL on this mutant


def test_an_exception_inside_the_block_still_restores():
    m = _randomised_toy()
    ema = W.WeightEMA(m, 0.9)
    _drive(ema, m)
    live = _snap(m)
    with pytest.raises(RuntimeError, match="boom"):
        with ema.swapped_in(m):
            raise RuntimeError("boom")
    assert _sd_bits_equal(_snap(m), live) and not ema.swapped


def test_a_bf16_parameter_is_stashed_not_exchanged_so_the_fp32_shadow_is_never_rounded():
    """e0 = 2.0, live = 1.0, d = 0.7, 3 updates -> 1 + (2-1) * 0.343 = 1.343 (not representable in bf16)."""
    m = _One((2.0, 2.0))
    m.to(torch.bfloat16)
    ema = W.WeightEMA(m, 0.7)
    with torch.no_grad():
        m.p.fill_(1.0)
    for _ in range(3):
        ema.update(m)
    shadow_before = ema.shadow["p"].clone()
    assert abs(float(shadow_before[0]) - 1.343) < 1e-6 and shadow_before.dtype == torch.float32
    live = _snap(m)
    with ema.swapped_in(m):
        assert m.p.dtype == torch.bfloat16
        assert float(m.p[0].detach()) == float(torch.tensor(1.343).to(torch.bfloat16))      # rounded INSIDE the eval only
    assert _sd_bits_equal(_snap(m), live)                        # live back exactly (1.0 in bf16)
    assert _bits_equal(ema.shadow["p"], shadow_before)           # the fp32 EMA was NOT overwritten by the rounded copy


def test_a_tied_weight_round_trips():
    m = nn.Module()
    m.a = nn.Linear(2, 2, bias=False)
    m.b = nn.Linear(2, 2, bias=False)
    m.b.weight = m.a.weight
    with torch.no_grad():
        m.a.weight.zero_()
    ema = W.WeightEMA(m, 0.5)                                    # e0 = 0
    with torch.no_grad():
        m.a.weight.add_(1.0)                                     # live = 1
    ema.update(m)                                                # EMA = 0 + 0.5 * (1 - 0) = 0.5
    live = _snap(m)
    assert set(live) == {"a.weight", "b.weight"}
    with ema.swapped_in(m):
        assert m.a.weight is m.b.weight                          # one tensor under both names ...
        assert torch.equal(m.a.weight, torch.full((2, 2), 0.5))  # ... holding the EMA
    assert torch.equal(m.a.weight, torch.ones(2, 2)) and torch.equal(m.b.weight, torch.ones(2, 2))
    sd = ema.model_state_dict(m)
    assert torch.equal(sd["a.weight"], torch.full((2, 2), 0.5)) and torch.equal(sd["b.weight"], torch.full((2, 2), 0.5))


def test_update_and_state_dict_refuse_while_swapped_in():
    m = _Toy()
    ema = W.WeightEMA(m, 0.5)
    with ema.swapped_in(m):
        with pytest.raises(W.EMAStateError, match="swapped in"):
            ema.update(m)
        with pytest.raises(W.EMAStateError, match="swapped in"):
            ema.state_dict()
        with pytest.raises(W.EMAStateError, match="already swapped"):
            ema.swap_in(m)
    with pytest.raises(W.EMAStateError, match="without a swap_in"):
        ema.swap_out(m)


# ====================================================================== save / resume round trip
def test_state_dict_round_trip_through_torch_save(tmp_path):
    m = _randomised_toy()
    ema = W.WeightEMA(m, 0.9)
    _drive(ema, m)
    torch.save(ema.state_dict(), tmp_path / "e.pt")
    sd = torch.load(tmp_path / "e.pt", weights_only=False)
    assert sd["schema"] == "weight_ema/1" and sd["decay"] == 0.9 and sd["num_updates"] == 7
    m2 = _randomised_toy(seed=5)
    ema2 = W.WeightEMA(m2, 0.9)
    assert not _sd_bits_equal(ema2.shadow, ema.shadow)
    ema2.load_state_dict(sd, m2)
    assert _sd_bits_equal(ema2.shadow, ema.shadow) and ema2.num_updates == 7


def test_load_refuses_a_different_decay_keys_or_shape():
    m = _Toy()
    ema = W.WeightEMA(m, 0.9)
    sd = ema.state_dict()
    with pytest.raises(W.EMAStateError, match="decay"):
        W.WeightEMA(_Toy(), 0.8).load_state_dict(sd)
    other = nn.Module()
    other.q = nn.Parameter(torch.zeros(2))
    with pytest.raises(W.EMAStateError, match="covers"):
        W.WeightEMA(other, 0.9).load_state_dict(sd)
    bad = copy.deepcopy(sd)
    bad["shadow"]["lin.weight"] = torch.zeros(5)
    with pytest.raises(W.EMAStateError, match="shape"):
        W.WeightEMA(_Toy(), 0.9).load_state_dict(bad)
    with pytest.raises(W.EMAStateError, match="schema"):
        W.WeightEMA(_Toy(), 0.9).load_state_dict({"decay": 0.9})


def test_model_state_dict_substitutes_the_EMA_and_loads_strictly():
    m = _randomised_toy()
    ema = W.WeightEMA(m, 0.9)
    _drive(ema, m)
    live = _snap(m)
    sd = ema.model_state_dict(m)
    assert set(sd) == set(live)                                   # the full key set: it loads into a bare model
    for k in TOY_SHADOWED:
        assert torch.equal(sd[k], ema.shadow[k]), k
    assert torch.equal(sd["frozen"], live["frozen"]) and torch.equal(sd["bn.num_batches_tracked"], live["bn.num_batches_tracked"])
    fresh = _Toy()
    fresh.load_state_dict(sd)                                     # strict
    assert _sd_bits_equal(_snap(m), live)                         # exporting never touched the live model


# ====================================================================== _resume_ema_guard (literal states)
_ARGS = types.SimpleNamespace


def test_resume_refusals_literal():
    ema_sd = {"decay": 0.9998, "num_updates": 7}
    with pytest.raises(SystemExit, match="carries no EMA state"):
        T._resume_ema_guard({"step": 5}, _ARGS(ema_decay=0.9998))
    with pytest.raises(SystemExit, match="--ema-decay is 0"):
        T._resume_ema_guard({"step": 5, "ema": ema_sd}, _ARGS(ema_decay=0.0))
    with pytest.raises(SystemExit, match="carries B1-guard state but"):
        T._resume_ema_guard({"step": 5, "ema": ema_sd, "b1_guard": {"ref_ap2m": 0.2}}, _ARGS(ema_decay=0.9998))
    with pytest.raises(SystemExit, match="carries no guard state"):
        T._resume_ema_guard({"step": 5, "ema": ema_sd}, _ARGS(ema_decay=0.9998, b1_guard_ref_ap2m=0.22867))
    with pytest.raises(SystemExit, match="is not the argv's"):
        T._resume_ema_guard({"step": 5, "ema": ema_sd, "b1_guard": {"ref_ap2m": 0.2, "guard_step": 1000}},
                            _ARGS(ema_decay=0.9998, b1_guard_ref_ap2m=0.22867))
    b1 = {"ref_ap2m": 0.22867, "guard_step": 1000}
    got = T._resume_ema_guard({"step": 5, "ema": ema_sd, "b1_guard": b1},
                              _ARGS(ema_decay=0.9998, b1_guard_ref_ap2m=0.22867, b1_guard_step=None))
    assert got == (ema_sd, b1)
    assert T._resume_ema_guard({"step": 5}, _ARGS(ema_decay=0.0)) == (None, None)                 # the tip's checkpoint


# ====================================================================== the guard on literal values
def test_decide_is_STRICT_less_than_on_literals():
    assert W.b1_guard_decide(0.22866, 0.22867) == "cut"
    assert W.b1_guard_decide(0.22867, 0.22867) == "keep"          # equal is not "below"
    assert W.b1_guard_decide(0.30000, 0.22867) == "keep"
    assert W.b1_guard_decide(0.0, 0.22867) == "cut"


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_decide_refuses_a_value_it_cannot_apply_the_rule_to(bad):
    with pytest.raises(W.B1GuardError, match="will not guess"):
        W.b1_guard_decide(bad, 0.22867)


def test_DELIBERATE_REGRESSION_a_flipped_comparison_goes_RED(monkeypatch):
    def _flipped(ema, ref):
        return "cut" if float(ema) <= float(ref) else "keep"           # the off-by-equality mutant
    monkeypatch.setattr(W, "b1_guard_decide", _flipped)
    assert W.b1_guard_decide(0.22867, 0.22867) == "cut"                 # the literal pin above reads "keep": RED


def _dd_opt(head_lr=1e-4, enc_lr=5e-5):
    enc, head = nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(2))
    return torch.optim.AdamW([{"name": "encoder", "params": [enc], "lr": enc_lr},
                              {"name": "head", "params": [head], "lr": head_lr}], lr=head_lr)


def _lrs(opt):
    return {g["name"]: g["lr"] for g in opt.param_groups}


def test_the_cut_on_a_real_AdamW_pair_continues_the_cosine_from_the_trunk_peak():
    opt = _dd_opt()
    args = types.SimpleNamespace(lr=1e-4)
    T.apply_lr_schedule(opt, args, lambda s: 1.0, 0)                  # latches initial_lr: head 1e-4, encoder 5e-5
    st = W.b1_guard_state(0.22867, 1000)
    row = W.b1_guard_apply(st, 1000, {"ema_eval_box3d_ap2m": 0.20000}, opt)
    assert row == {"step": 1000, "b1_guard": "cut", "b1_cut": True, "b1_ema_box3d_ap2m": 0.2,
                   "b1_ref_box3d_ap2m": 0.22867, "b1_guard_step": 1000,
                   "b1_head_peak_lr_before": 1e-4, "b1_head_peak_lr_after": 5e-5}
    assert st["decided"] is True and st["decision"] == "cut" and st["decided_at_step"] == 1000
    T.apply_lr_schedule(opt, args, lambda s: 0.5, 1000)               # the cosine at half height
    assert _lrs(opt) == {"encoder": 2.5e-5, "head": 2.5e-5}            # head follows the trunk's peak
    assert W.b1_guard_apply(st, 1500, {"ema_eval_box3d_ap2m": 0.0}, opt) is None        # decided ONCE


def test_keep_leaves_the_peaks_alone_and_is_also_decided_once():
    opt = _dd_opt()
    args = types.SimpleNamespace(lr=1e-4)
    T.apply_lr_schedule(opt, args, lambda s: 1.0, 0)
    st = W.b1_guard_state(0.22867, 1000)
    row = W.b1_guard_apply(st, 1000, {"ema_eval_box3d_ap2m": 0.23100}, opt)
    assert row["b1_guard"] == "keep" and row["b1_cut"] is False and row["b1_head_peak_lr_after"] is None
    T.apply_lr_schedule(opt, args, lambda s: 0.5, 1000)
    assert _lrs(opt) == {"encoder": 2.5e-5, "head": 5e-5}
    assert W.b1_guard_apply(st, 1500, {"ema_eval_box3d_ap2m": 0.0}, opt) is None        # a later low value cannot cut


def test_the_guard_waits_for_its_step_and_for_a_successful_ema_eval_and_refuses_without_the_metric():
    opt = _dd_opt()
    T.apply_lr_schedule(opt, types.SimpleNamespace(lr=1e-4), lambda s: 1.0, 0)
    st = W.b1_guard_state(0.22867, 1000)
    assert W.b1_guard_apply(st, 500, {"ema_eval_box3d_ap2m": 0.0}, opt) is None              # before the step
    assert W.b1_guard_apply(st, 1000, {"ema_eval_error": "OOM"}, opt) is None                # this eval failed: wait
    assert st["decided"] is False
    with pytest.raises(W.B1GuardError, match="is not being measured"):
        W.b1_guard_apply(st, 1000, {"eval_loss": 1.0}, opt)                                  # armed, never measured
    with pytest.raises(W.B1GuardError, match="will not guess"):
        W.b1_guard_apply(st, 1000, {"ema_eval_box3d_ap2m": None}, opt)                       # NaN is written as null
    assert W.b1_guard_apply(None, 1000, {}, opt) is None                                     # not armed


def test_head_cut_refuses_without_the_dd_groups_or_a_latched_peak():
    plain = torch.optim.Adam([nn.Parameter(torch.zeros(1))], lr=1e-4)
    with pytest.raises(W.B1GuardError, match="'encoder' and 'head' groups"):
        W.b1_head_cut(plain)
    with pytest.raises(W.B1GuardError, match="no latched initial_lr"):
        W.b1_head_cut(_dd_opt())                                       # apply_lr_schedule has not run
    # a trunk peak ABOVE the head peak must never RAISE the head: min()
    opt = _dd_opt(head_lr=1e-4, enc_lr=2e-4)
    T.apply_lr_schedule(opt, types.SimpleNamespace(lr=1e-4), lambda s: 1.0, 0)
    assert W.b1_head_cut(opt) == (1e-4, 1e-4)


@pytest.mark.parametrize("ref", [0.0, -0.1, 1.5, float("nan")])
def test_guard_state_refuses_a_reference_that_is_not_an_AP(ref):
    with pytest.raises(W.B1GuardError, match="an AP is in"):
        W.b1_guard_state(ref, 1000)


# ====================================================================== the startup pins
_GOOD = ["--arm", "hier", "--out", "X", "--steps", "3000", "--ema-decay", "0.9998", "--b1-guard-ref-ap2m", "0.22867",
         "--opt", "dd", "--eval-cache", "E", "--eval-every", "500", "--slot-vis1", "--w-box3d", "1.0"]


def _pin(extra):
    from tanitad.refs import refcv8_conditioning as r8c
    args = T.build_parser().parse_args(extra)
    T._pin_refcv8(types.SimpleNamespace(refcv8=r8c.R8Config()), args)
    return args


def _without(argv, *flags):
    out, skip = [], 0
    for i, a in enumerate(argv):
        if skip:
            skip -= 1
            continue
        if a in flags:
            skip = 0 if a in ("--slot-vis1",) else 1
            continue
        out.append(a)
    return out


def test_the_pin_accepts_the_ruling_command_line_and_the_off_default():
    a = _pin(_GOOD)
    assert (a.ema_decay, a.b1_guard_ref_ap2m, a.b1_guard_step) == (0.9998, 0.22867, None)
    d = _pin(["--arm", "hier", "--out", "X", "--steps", "3000"])
    assert (d.ema_decay, d.b1_guard_ref_ap2m, d.b1_guard_step) == (0.0, None, None)          # off = the tip


@pytest.mark.parametrize("argv,match", [
    (["--arm", "hier", "--out", "X", "--ema-decay", "1.0"], "--ema-decay 1.0 must be 0"),
    (["--arm", "hier", "--out", "X", "--ema-decay", "-0.5"], "must be 0 .off. or in"),
    (["--arm", "hier", "--out", "X", "--b1-guard-step", "1000"], "--b1-guard-step without --b1-guard-ref-ap2m"),
    (_without(_GOOD, "--ema-decay"), "--ema-decay 0: the guard reads"),
    (_without(_GOOD, "--eval-cache"), "no in-run eval"),
    (_without(_GOOD, "--eval-every"), "no in-run eval"),
    (_without(_GOOD, "--slot-vis1"), "--slot-vis1 is off"),
    (_without(_GOOD, "--w-box3d"), "--w-box3d is 0"),
    (_without(_GOOD, "--opt"), "--opt is not dd"),
    (_GOOD + ["--encoder-lr-mult", "1.0"], "--encoder-lr-mult >= 1"),
    (_GOOD + ["--b1-guard-step", "3001"], "could never fire"),
    (_GOOD + ["--b1-guard-step", "0"], "must be >= 1"),
    (_without(_GOOD, "--b1-guard-ref-ap2m") + ["--b1-guard-ref-ap2m", "0"], "an AP is in"),
])
def test_the_pin_refuses_each_bad_combination_before_config_json(argv, match):
    with pytest.raises(SystemExit, match=match):
        _pin(argv)


def test_DELIBERATE_REGRESSION_with_the_pin_removed_nothing_refuses(monkeypatch):
    monkeypatch.setattr(T, "_pin_ema_guard", lambda args: None)
    for argv in (_without(_GOOD, "--ema-decay"), _without(_GOOD, "--slot-vis1"), _without(_GOOD, "--opt")):
        _pin(argv)                                                      # no SystemExit: the refusals above are the pin's


# ====================================================================== the trainer on the synthetic rig
def _train(out, extra=(), steps=3, patches=None):
    mp = pytest.MonkeyPatch()
    RIG._rig(mp)
    try:
        for tgt, name, val in (patches or ()):
            mp.setattr(tgt, name, val)
        args = T.build_parser().parse_args(RIG._argv(out) + ["--steps", str(steps)] + list(extra))
        assert T.train(args) == {"step": steps}
    finally:
        mp.undo()
    return out


def _rows(out):
    return RIG._rows(Path(out))


def _ck(out, name="ckpt.pt"):
    return torch.load(Path(out) / name, map_location="cpu", weights_only=False)


def test_ema_on_writes_ckpt_state_model_only_file_eval_keys_and_summary(tmp_path):
    out = _train(tmp_path / "on", ["--ema-decay", "0.5"], steps=3)
    ck = _ck(out)
    assert set(ck) == {"model", "opt", "step", "data_pos", "ema"}
    assert ck["ema"]["schema"] == "weight_ema/1" and ck["ema"]["decay"] == 0.5 and ck["ema"]["num_updates"] == 3
    assert len(ck["ema"]["shadow"]) == 179                                 # measured on this rig: 179 tensors
    assert sum(v.numel() for v in ck["ema"]["shadow"].values()) == 131899     # and 131,899 values = 527,596 B fp32
    # the model-only file: same layout as ckpt_<milestone>.pt, EMA values in the shadowed slots
    me = _ck(out, "model_ema.pt")
    assert set(me) == {"model", "step", "ema_decay", "num_updates", "kind"}
    assert me["step"] == 3 and me["ema_decay"] == 0.5 and me["num_updates"] == 3 and me["kind"] == "ema_model_only"
    assert set(me["model"]) == set(ck["model"])
    assert all(torch.equal(me["model"][k], v) for k, v in ck["ema"]["shadow"].items())
    assert any(not torch.equal(me["model"][k], ck["model"][k]) for k in ck["ema"]["shadow"])      # it IS different
    assert (Path(out) / "ckpt_ema_1.pt").exists()                          # MILESTONES == (1,) in the rig
    evals = [r for r in _rows(out) if "eval_loss" in r]
    assert len(evals) == 3
    for r in evals:
        live = {k for k in r if k.startswith("eval_") and k not in ("eval_batches", "eval_windows")}
        assert len(live) == 21 and all(f"ema_{k}" in r for k in live), sorted(live)
    s = json.loads((Path(out) / "summary.json").read_text())
    assert s["ema_decay"] == 0.5 and s["ema_num_updates"] == 3 and s["done"] is True
    cfg = json.loads((Path(out) / "config.json").read_text())
    assert cfg["ema"]["decay"] == 0.5 and cfg["ema"]["n_tensors"] == 179 and cfg["ema"]["shadow_bytes"] == 527596
    assert cfg["ema"]["init"] == "the weights at step 0"


def test_ema_off_is_the_tips_checkpoint_rows_and_summary_key_for_key(tmp_path):
    out = _train(tmp_path / "off", [], steps=2)
    assert set(_ck(out)) == {"model", "opt", "step", "data_pos"}
    assert sorted(p.name for p in Path(out).iterdir()) == ["ckpt.pt", "ckpt_1.pt", "config.json", "metrics.jsonl", "summary.json"]
    assert not any(k.startswith(("ema_", "b1_")) for r in _rows(out) for k in r)
    assert set(json.loads((Path(out) / "summary.json").read_text())) == {"done", "step", "arm", "seed", "wallclock_s"}
    assert "ema" not in json.loads((Path(out) / "config.json").read_text())


def test_the_live_weights_are_BIT_IDENTICAL_with_and_without_the_EMA(tmp_path):
    """EMA eval swaps weights in and out every step and runs a second eval pass; none of it may move what training
    reads: the weights, the optimiser state and the RNG stream. Two off runs also agree (the rig is deterministic)."""
    off1 = _ck(_train(tmp_path / "off1", [], steps=3))
    off2 = _ck(_train(tmp_path / "off2", [], steps=3))
    on = _ck(_train(tmp_path / "on", ["--ema-decay", "0.9"], steps=3))
    assert _sd_bits_equal(off1["model"], off2["model"])
    assert _sd_bits_equal(off1["model"], on["model"])
    assert off1["opt"]["state"].keys() == on["opt"]["state"].keys()
    for k, st in off1["opt"]["state"].items():
        assert all(torch.equal(st[n], on["opt"]["state"][k][n]) for n in st)


def test_DELIBERATE_REGRESSION_a_swap_that_does_not_restore_changes_the_run(tmp_path, monkeypatch):
    off = _ck(_train(tmp_path / "off", [], steps=3))

    def _forget(self, model):
        self._stash, self._swapped = {}, False

    on = _ck(_train(tmp_path / "bad", ["--ema-decay", "0.9"], steps=3, patches=[(W.WeightEMA, "swap_out", _forget)]))
    assert not _sd_bits_equal(off["model"], on["model"])                   # the equality pin above would be RED


def test_a_swap_out_that_fails_is_fatal_not_survived(tmp_path):
    def _stuck(self, model):                                               # leaves the model holding the EMA weights
        return None
    with pytest.raises(SystemExit, match="swap-out did not complete"):
        _train(tmp_path / "stuck", ["--ema-decay", "0.9"], steps=1, patches=[(W.WeightEMA, "swap_out", _stuck)])


def test_the_EMA_row_equals_the_live_row_when_the_EMA_IS_the_live_weights(tmp_path):
    """decay 1e-12 -> lerp weight rounds to 1.0 -> ema == live after every update. Then EVERY ema_eval_* key must equal
    its eval_* twin (same 21 metrics through the second eval pass), which pins the duplicated aggregate loop against the
    live one. The mutant (update skipped, weights moving) must NOT satisfy it."""
    def _twin_gap(out):
        r = [x for x in _rows(out) if "eval_loss" in x][-1]
        ks = [k for k in r if k.startswith("eval_") and k not in ("eval_batches", "eval_windows")]
        assert len(ks) == 21
        return [k for k in ks if r[f"ema_{k}"] != r[k]]

    ok = _train(tmp_path / "ok", ["--ema-decay", "1e-12", "--lr", "1e-3"], steps=3)
    assert _twin_gap(ok) == []
    bad = _train(tmp_path / "bad", ["--ema-decay", "1e-12", "--lr", "1e-3"], steps=3,
                 patches=[(W.WeightEMA, "update", lambda self, model: None)])
    assert "eval_loss" in _twin_gap(bad)


def test_the_EMA_row_really_evaluates_the_shadow_not_the_live_weights(tmp_path, monkeypatch):
    """The converse of the twin test: with a real decay and a moving model the EMA row must DIFFER from the live row
    (an eval pass that forgot the swap would read the live weights under an ema_ label). Pinned two ways: the values
    differ, and the swap really happened - ``swap_in`` is entered once per in-run eval, with the shadow's weights."""
    entered = []
    real = W.WeightEMA.swap_in

    def _spy(self, model):
        real(self, model)
        entered.append(self.num_updates)
    out = _train(tmp_path / "d", ["--ema-decay", "0.9", "--lr", "1e-3"], steps=3, patches=[(W.WeightEMA, "swap_in", _spy)])
    assert entered == [1, 2, 3]                                          # once per eval, after 1, 2, 3 updates
    last = [r for r in _rows(out) if "eval_loss" in r][-1]
    assert last["ema_eval_loss"] != last["eval_loss"] and last["ema_eval_traj"] != last["eval_traj"]


def test_DELIBERATE_REGRESSION_a_skipped_update_is_visible_in_the_checkpoint(tmp_path):
    ok = _ck(_train(tmp_path / "ok", ["--ema-decay", "0.9"], steps=3))
    assert ok["ema"]["num_updates"] == 3
    bad = _ck(_train(tmp_path / "bad", ["--ema-decay", "0.9"], steps=3,
                     patches=[(W.WeightEMA, "update", lambda self, model: None)]))
    assert bad["ema"]["num_updates"] != 3                                  # == 0: the literal 3 above would be RED


def test_resume_refuses_both_ways_and_continues_the_same_ema(tmp_path, monkeypatch):
    out = tmp_path / "r"
    _train(out, ["--ema-decay", "0.5"], steps=2)
    assert _ck(out)["ema"]["num_updates"] == 2
    # (1) the flag was dropped from the launch line
    with pytest.raises(SystemExit, match="--ema-decay is 0"):
        _train(out, [], steps=4)
    # (2) a different decay is a different experiment
    with pytest.raises(SystemExit, match="EMA does not fit this run"):
        _train(out, ["--ema-decay", "0.7"], steps=4)
    assert _ck(out)["step"] == 2                                          # neither refusal touched the checkpoint
    # (3) the same decay resumes the SAME average: the loaded state is the saved one (2 updates) and it goes on to 4
    seen = {}
    real = W.WeightEMA.load_state_dict

    def _spy(self, sd, model=None):
        seen["n"] = sd["num_updates"]
        seen["shadow"] = {k: v.clone() for k, v in sd["shadow"].items()}
        return real(self, sd, model)
    saved = _ck(out)["ema"]["shadow"]
    _train(out, ["--ema-decay", "0.5"], steps=4, patches=[(W.WeightEMA, "load_state_dict", _spy)])
    assert seen["n"] == 2 and all(torch.equal(seen["shadow"][k], saved[k]) for k in saved)
    assert _ck(out)["ema"]["num_updates"] == 4 and _ck(out)["step"] == 4
    assert json.loads((out / "config.json").read_text())["ema"]["init"] == "resumed ckpt.pt['ema']"
    # (4) a checkpoint WITHOUT an EMA cannot be resumed under --ema-decay > 0 (a fresh shadow averages another run)
    out2 = tmp_path / "r2"
    _train(out2, [], steps=2)
    with pytest.raises(SystemExit, match="carries no EMA state"):
        _train(out2, ["--ema-decay", "0.5"], steps=4)


# ---------------------------------------------------------------------------- the guard, end to end
class _Died(BaseException):
    """A process death inside the step-N eval (the C-REFCV3-EVAL-DEATH shape): nothing in train() may catch it."""


def _guard_run(out, aps, *, ref=0.5, guard_step=2, steps=4, decide=None, extra=(), die_at_call=None):
    """The trainer with the guard armed. The synthetic rig has no box head, so (1) the startup pin - which would demand
    --slot-vis1 / --w-box3d, pinned separately above - is stubbed, and (2) ``ema_eval_box3d_ap2m`` is INJECTED into the
    EMA eval row from ``aps`` (one value per in-run eval). Everything downstream of the row is the real code: the
    guard, the real ``--opt dd`` AdamW pair of groups, the schedule, the checkpoint, the resume."""
    calls = {"n": 0}
    real = T._ema_eval_row

    def _inject(*a, **k):
        if die_at_call is not None and calls["n"] + 1 == die_at_call:
            calls["n"] += 1
            raise _Died()
        row = real(*a, **k)
        row[W.B1_METRIC_KEY] = aps[min(calls["n"], len(aps) - 1)]
        calls["n"] += 1
        return row
    patches = [(T, "_pin_ema_guard", lambda args: None), (T, "_ema_eval_row", _inject)]
    if decide is not None:
        patches.append((W, "b1_guard_decide", decide))
    return _train(out, ["--ema-decay", "0.5", "--b1-guard-ref-ap2m", str(ref), "--b1-guard-step", str(guard_step),
                        "--opt", "dd", "--warmup", "1"] + list(extra), steps=steps, patches=patches)


def _groups(out, name="ckpt.pt"):
    return {g["name"]: g for g in _ck(out, name)["opt"]["param_groups"]}


def test_guard_cut_end_to_end(tmp_path):
    # evals at steps 1..4 read 0.9 (before the guard step: ignored), 0.4 (< 0.5: CUT), 0.1, 0.1 (decided once)
    out = _guard_run(tmp_path / "cut", [0.9, 0.4, 0.1, 0.1])
    rows = _rows(out)
    g = [r for r in rows if "b1_guard" in r]
    assert len(g) == 1 and g[0] == {"step": 2, "b1_guard": "cut", "b1_cut": True, "b1_ema_box3d_ap2m": 0.4,
                                    "b1_ref_box3d_ap2m": 0.5, "b1_guard_step": 2,
                                    "b1_head_peak_lr_before": 1e-4, "b1_head_peak_lr_after": 5e-5}
    grp = _groups(out)
    assert grp["head"]["initial_lr"] == 5e-5 and grp["encoder"]["initial_lr"] == 5e-5
    # last step index 3, warmup 1, steps 4: sched = 0.5 * (1 + cos(pi * 2/3)) = 0.25 -> BOTH groups at 1.25e-5
    assert abs(grp["head"]["lr"] - 1.25e-5) < 1e-12 and abs(grp["encoder"]["lr"] - 1.25e-5) < 1e-12
    ck = _ck(out)
    assert ck["b1_guard"]["decided"] is True and ck["b1_guard"]["decision"] == "cut" and ck["b1_guard"]["decided_at_step"] == 2
    assert json.loads((Path(out) / "summary.json").read_text())["b1_guard"]["decision"] == "cut"
    cfg = json.loads((Path(out) / "config.json").read_text())["ema"]["b1_guard"]
    assert cfg["armed"] is True and cfg["ref_ap2m"] == 0.5 and cfg["guard_step"] == 2 and cfg["decided"] is False


def test_guard_keep_end_to_end_and_a_finished_runs_decision_is_in_its_checkpoint(tmp_path):
    """The step-2 checkpoint is written BEFORE the step-2 eval, so it first carries the guard PENDING; the guard then
    re-saves it with the decision. Without that, continuing a finished run would decide a second time."""
    out = tmp_path / "keep"
    _guard_run(out, [0.9, 0.6, 0.1, 0.1], steps=2)                         # decided at step 2: 0.6 >= 0.5 -> KEEP
    assert _ck(out)["b1_guard"]["decision"] == "keep" and _ck(out)["b1_guard"]["decided_at_step"] == 2
    # the RE-SAVED checkpoint is the loop's own payload plus the decision: same keys, same data position, same step
    plain = _ck(_train(tmp_path / "plain", ["--ema-decay", "0.5", "--opt", "dd", "--warmup", "1"], steps=2))
    resaved = _ck(out)
    assert set(resaved) == set(plain) | {"b1_guard"} == {"model", "opt", "step", "data_pos", "ema", "b1_guard"}
    assert resaved["data_pos"] == plain["data_pos"] and resaved["step"] == plain["step"] == 2
    assert set(resaved["opt"]["param_groups"][0]) == set(plain["opt"]["param_groups"][0])
    _guard_run(out, [0.1, 0.1], steps=4)                                   # continued: a low value must NOT cut now
    assert [r["b1_guard"] for r in _rows(out) if "b1_guard" in r] == ["keep"]
    grp = _groups(out)
    assert grp["head"]["initial_lr"] == 1e-4 and grp["encoder"]["initial_lr"] == 5e-5
    # head 1e-4 * 0.25, encoder 5e-5 * 0.25 at the last step
    assert abs(grp["head"]["lr"] - 2.5e-5) < 1e-12 and abs(grp["encoder"]["lr"] - 1.25e-5) < 1e-12


def test_a_cut_decision_survives_a_resume_in_the_optimizer_state(tmp_path):
    out = tmp_path / "cutresume"
    _guard_run(out, [0.9, 0.4, 0.1, 0.1], steps=3)                         # CUT at step 2
    assert _groups(out)["head"]["initial_lr"] == 5e-5
    _guard_run(out, [0.9, 0.9], steps=4)                                   # resumed: the cut is in opt.param_groups
    assert [r["b1_guard"] for r in _rows(out) if "b1_guard" in r] == ["cut"]
    assert _groups(out)["head"]["initial_lr"] == 5e-5 and _ck(out)["b1_guard"]["decision"] == "cut"


def test_a_death_between_the_guard_steps_checkpoint_and_its_eval_is_caught_up_on_resume(tmp_path):
    """The ckpt at the guard step is saved before the eval; the process dies INSIDE that eval. The resume must decide
    from those very weights BEFORE the next training step - not at the next scheduled eval, after more steps at the
    uncut head rate."""
    out = tmp_path / "died"
    with pytest.raises(_Died):
        _guard_run(out, [0.9, 0.4], steps=2, die_at_call=2)               # call 2 = the step-2 eval
    ck = _ck(out)
    assert ck["step"] == 2 and ck["b1_guard"]["decided"] is False          # pending in the checkpoint
    assert [r for r in _rows(out) if "b1_guard" in r] == []
    _guard_run(out, [0.4, 0.1, 0.1], steps=4)                              # resume: catch-up reads 0.4 < 0.5 -> CUT
    g = [r for r in _rows(out) if "b1_guard" in r]
    assert len(g) == 1 and g[0]["step"] == 2 and g[0]["b1_guard"] == "cut" and g[0]["b1_catch_up_on_resume"] is True
    assert _ck(out)["b1_guard"]["decided_at_step"] == 2
    assert _groups(out)["head"]["initial_lr"] == 5e-5


def test_DELIBERATE_REGRESSION_a_flipped_guard_does_not_cut_and_goes_RED(tmp_path):
    out = _guard_run(tmp_path / "flip", [0.9, 0.4, 0.1, 0.1], decide=lambda e, r: "keep")
    assert _groups(out)["head"]["initial_lr"] == 1e-4                     # the cut test reads 5e-5: RED on the mutant


def test_guard_with_the_metric_missing_refuses_the_run(tmp_path):
    with pytest.raises(SystemExit, match="B1 guard: .*is not being measured"):
        _train(tmp_path / "nometric", ["--ema-decay", "0.5", "--b1-guard-ref-ap2m", "0.5", "--b1-guard-step", "2",
                                       "--opt", "dd", "--warmup", "1"], steps=3,
               patches=[(T, "_pin_ema_guard", lambda args: None)])      # no box head, nothing injected


def test_guard_waits_through_a_failed_ema_eval(tmp_path):
    real = T._ema_eval_row
    n = {"i": 0}

    def _flaky(*a, **k):
        n["i"] += 1
        if n["i"] == 2:                                                    # the EMA eval AT the guard step dies
            return {"ema_eval_error": "RuntimeError: synthetic", "ema_eval_batches_done": 0}
        row = real(*a, **k)
        row[W.B1_METRIC_KEY] = 0.4
        return row
    out = _train(tmp_path / "wait", ["--ema-decay", "0.5", "--b1-guard-ref-ap2m", "0.5", "--b1-guard-step", "2",
                                     "--opt", "dd", "--warmup", "1"], steps=4,
                 patches=[(T, "_pin_ema_guard", lambda args: None), (T, "_ema_eval_row", _flaky)])
    g = [r for r in _rows(out) if "b1_guard" in r]
    assert len(g) == 1 and g[0]["step"] == 3 and g[0]["b1_guard"] == "cut"   # the first eval that CARRIED the metric


# ---------------------------------------------------------------------------- source pins (with mutants)
def _train_src():
    src = inspect.getsource(T).replace("\r\n", "\n")
    return src[src.index("\ndef train("):]


def test_the_update_follows_the_optimizer_step_and_the_resume_check_precedes_the_load():
    tr = _train_src()
    assert "        opt.step()\n        if _ema is not None:" in tr and "            _ema.update(model)\n        step += 1" in tr
    mut = tr.replace("        opt.step()\n        if _ema is not None:", "        if _ema is not None:")
    assert "        opt.step()\n        if _ema is not None:" not in mut                    # the mutant breaks the pin
    i_check = tr.index("_ema_resume, _b1_resume = _resume_ema_guard(state, args)")
    i_load = tr.index('model.load_state_dict(state["model"])')
    assert i_check < i_load                                                                 # refuse BEFORE any state loads
    assert tr.index("_ema = _b1 = None") < tr.index("while step < args.steps:")             # built before the loop


def test_the_swap_wraps_only_the_ema_eval_and_the_live_eval_block_is_untouched():
    src = inspect.getsource(T).replace("\r\n", "\n")
    fn = src[src.index("\ndef _ema_eval_row("):src.index("\ndef _resume_ema_guard(")]
    assert "with ema.swapped_in(model), _RngIsolated(device, None), torch.no_grad():" in fn
    tr = _train_src()
    live = tr[tr.index("            model.eval()\n            acc, nb_e, eval_err = {}, 0, None"):
              tr.index("            model.train()\n            if eval_err is not None:")]
    assert "swapped_in" not in live and "_ema" not in live                                 # the live pass is the tip's
