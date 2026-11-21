"""refcv6 -- the GRADIENT-CONFLICT DETECTOR, and the four things that make it an
instrument rather than a number.

``SPEC_REFCV6_V2.md`` §6 and ``PREREG_BEV_CAPACITY_COMPETITION.md`` §4.

1. **Three analytic controls read their known values EXACTLY** -- ``+1`` / the
   detached ``0`` / ``-1``. Not to a tolerance: ``==``.
2. ⛔ **The 30x mutation**, and what it MEASURES about the pre-registered
   statistic. Read :func:`test_mutation_table` before quoting any ``cd_cos``.
3. **Per parameter group**, because "the trunk is in conflict" and "the last
   stage is in conflict" imply different fixes.
4. **The flag off is bit-identical** -- and so, here, is the flag ON, which is a
   stronger claim and is proven the same way.

⛔ **The guards are proven by MUTATION.** Every refusal branch in
``grad_conflict.controls`` is REACHED here by re-introducing the defect it
exists to catch (a lossy denominator; an un-detached "detached" control), so a
green run means the branch works, not that it was never visited.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from tanitad.train import grad_conflict as gc
from tanitad.train.grad_conflict import (
    ConflictConfig, ControlFailure, GradientConflictDetector, MODE_PROBE,
    MODE_REUSE, cosine_stats, default_group_of, enabled_for_arm)

SEED = 20260917


# --------------------------------------------------------------------------- #
#  the tiny two-head smoke: ONE trunk, a planner head and a perception head
# --------------------------------------------------------------------------- #
class _Trunk(nn.Module):
    """Named the way both real refcv6 trunks are: a ``stem`` then ``layer*``
    stages (timm ResNet) -- ``refc.ResNetEncoder`` uses ``stem``/``stages.N``
    and both are exercised by :func:`test_grouping_matches_both_real_trunks`."""

    def __init__(self, c: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, c, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.layer1 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(c, 2 * c, 3, stride=2, padding=1),
                                    nn.ReLU())

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        return self.layer2(self.layer1(x))


class TwoHead(nn.Module):
    """``encoder`` (shared) + ``traj_head`` (planning) + ``aux_head`` (perception).

    ``detach_aux`` is the REAL detached control: the perception head reads
    ``feat.detach()``, so no perception gradient exists on theta_trunk at all --
    the thing the surrogate in ``controls()`` stands in for.
    """

    def __init__(self, detach_aux: bool = False, c: int = 8):
        super().__init__()
        self.encoder = _Trunk(c)
        self.traj_head = nn.Linear(2 * c, 4)
        self.aux_head = nn.Linear(2 * c, 3)
        self.detach_aux = bool(detach_aux)

    def forward(self, x):
        f = self.encoder(x).mean(dim=(2, 3))
        return self.traj_head(f), self.aux_head(f.detach() if self.detach_aux else f)


def _batch(n: int = 4, seed: int = SEED):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 3, 32, 32, generator=g),
            torch.randn(n, 4, generator=g), torch.randn(n, 3, generator=g))


def _losses(model, batch):
    x, yt, ya = batch
    pt, pa = model(x)
    return ((pt - yt) ** 2).mean(), ((pa - ya) ** 2).mean()


def _fresh(detach_aux: bool = False, seed: int = SEED):
    torch.manual_seed(seed)
    return TwoHead(detach_aux=detach_aux)


def _det(model, **kw):
    cfg = ConflictConfig(enabled=True, **kw)
    return GradientConflictDetector.for_model(model, cfg)


# ==========================================================================  #
#  1. THE THREE ANALYTIC CONTROLS -- exactly, not to a tolerance
# ==========================================================================  #
def test_control_self_cosine_is_exactly_plus_one():
    """⛔ ``cos(g, g) == +1.0``. An IDENTITY, derived independently of this code.

    ⭐ It is exact, not 1 - 1e-16, because the denominator is
    ``sqrt(a*b)`` and never ``sqrt(a)*sqrt(b)`` -- see
    :func:`test_mutation_lossy_denominator_is_CAUGHT`, which re-introduces the
    other spelling and watches the control refuse it.
    """
    m = _fresh()
    d = _det(m)
    lt, la = _losses(m, _batch())
    res = d.controls(lt, la)
    assert res.self_cos == 1.0, f"cos(g,g) read {res.self_cos!r}"
    assert abs(res.self_cos - 1.0) <= 0.0


def test_control_negated_cosine_is_exactly_minus_one():
    """⛔ ``cos(g, -g) == -1.0``. Fixes the SIGN CONVENTION -- the
    ``R-2026-09-08-wpa-mirror`` family is a sign nobody asserted."""
    m = _fresh()
    d = _det(m)
    lt, la = _losses(m, _batch())
    res = d.controls(lt, la)
    assert res.negated_cos == -1.0, f"cos(g,-g) read {res.negated_cos!r}"


def test_control_detached_aux_reads_zero_WITH_ITS_RECEIPT():
    """⛔ A detached aux reads a conflict of **exactly 0.0** -- and the raw
    quotient is **NaN**, so "orthogonal" can never be confused with "absent".

    The prereg asks for both ("exactly 0.0" *and* "report NaN -> refuse, never 0
    by accident"). Both are delivered: ``cos`` is the raw 0/0, ``conflict`` is
    the reported 0.0, and ``degenerate`` / ``norm_aux`` are logged beside it.
    """
    m = _fresh()
    d = _det(m)
    lt, la = _losses(m, _batch())
    res = d.controls(lt, la)
    assert math.isnan(res.detached_cos)
    assert res.detached_conflict == 0.0
    assert res.detached_norm_aux == 0.0
    assert res.detached_degenerate is True
    assert res.ok and res.failures == ()


def test_detached_control_holds_on_a_REAL_detached_head():
    """⭐ The surrogate in ``controls()`` is CHECKED, not trusted: the same three
    readings come out of a model whose perception head actually reads
    ``feat.detach()``."""
    m = _fresh(detach_aux=True)
    d = _det(m)
    lt, la = _losses(m, _batch())
    r = d.measure(lt, la)
    assert math.isnan(r.pooled.cos)
    assert r.pooled.conflict == 0.0
    assert r.pooled.norm_aux == 0.0
    assert r.pooled.degenerate is True
    # and the planning gradient is emphatically NOT zero -- otherwise the row
    # above would be the vacuous "nothing trained" reading wearing the control's
    # clothes.
    assert r.pooled.norm_traj > 0.0
    for g in r.groups.values():
        if g.name == gc.HEADS_NAME:
            continue
        assert g.norm_aux == 0.0, f"group {g.name} still receives aux gradient"


def test_no_epsilon_anywhere_so_absent_and_orthogonal_never_collide():
    """⛔ A zero side must be NaN. An eps in the denominator would make a
    detached head and a perfectly orthogonal one read the same 0.0, which is the
    one confusion this control exists to prevent."""
    z = torch.zeros(5)
    st = cosine_stats([torch.ones(5)], [z])
    assert math.isnan(st["cos"]) and st["degenerate"] is True
    assert st["norm_aux"] == 0.0
    # a genuinely orthogonal pair, both non-zero: 0.0 and NOT degenerate
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 2.0, 0.0])
    so = cosine_stats([a], [b])
    assert so["cos"] == 0.0 and so["degenerate"] is False and so["norm_aux"] == 2.0


# ==========================================================================  #
#  2. ⛔ THE 30x MUTATION -- and the finding it produces
# ==========================================================================  #
def test_mutation_table():
    """⛔⛔ **THE MUTATION ARM, AND WHAT IT MEASURES ABOUT THE PRE-REGISTERED
    STATISTIC.**

    ``E-DEC-18``'s MEASURED mechanism was SCALE: the aux term sat 10-30x above
    the objective. The prereg re-introduces it as "scale the aux loss by 30x and
    the conflict detector must fire".

    ⛔ **The cosine CANNOT fire on it.** ``cos`` is invariant under positive
    scaling *by construction* -- that is the transformation a cosine quotients
    out -- so no implementation of ``cos(g_traj, g_aux)`` can see a 30x aux. This
    test proves it two ways and then shows the channels that DO fire:

    | channel | 1x | 30x | verdict |
    |---|---|---|---|
    | ``cos``   | unchanged to float32 rounding | **BLIND** |
    | ``ratio`` | x30 | **FIRES** |
    | ``proj``  | x30, crossing -1 | **FIRES** |

    ``proj = <g_traj, g_aux> / |g_traj|^2`` is the signed multiple of the
    planner's own gradient that the aux adds along it; ``proj <= -1`` means the
    aux gradient more than cancels the planning gradient. THAT is the reading
    that goes red.
    """
    m = _fresh()
    d = _det(m)
    lt, la = _losses(m, _batch())
    r1 = d.measure(lt, la)
    r30 = d.measure(lt, la * 30.0)
    # -- the cosine is blind ------------------------------------------------ #
    # 30 is not a binary-exact scale, so the float32 gradient it produces is
    # 30*g rounded: the cosine moves by ROUNDING, not by signal.
    assert abs(r30.pooled.cos - r1.pooled.cos) < 1e-6
    rel = abs(r30.pooled.cos - r1.pooled.cos) / max(abs(r1.pooled.cos), 1e-30)
    assert rel < 1e-4, f"cos moved {rel:.3g} relative -- unexpected, re-derive"
    # ⭐ and with a BINARY-EXACT scale the invariance is bit-exact, which proves
    # the residual above is rounding and not sensitivity.
    r32 = d.measure(lt, la * 32.0)
    assert r32.pooled.cos == r1.pooled.cos, (
        "a power-of-two rescale changed the cosine; the invariance argument is "
        "wrong and the whole reading needs re-deriving")
    # -- the magnitude channels fire --------------------------------------- #
    assert r30.pooled.ratio == pytest.approx(30.0 * r1.pooled.ratio, rel=1e-5)
    assert r30.pooled.proj == pytest.approx(30.0 * r1.pooled.proj, rel=1e-5)
    assert r32.pooled.ratio == pytest.approx(32.0 * r1.pooled.ratio, rel=1e-6)
    # -- and the arm this batch produces is one where 30x GOES RED ---------- #
    # (the sign of `proj` is the batch's own; what is asserted is the 30x jump
    # in MAGNITUDE, which is the defect's signature, plus that 30x crosses the
    # "the aux now dominates the planner's descent direction" line.)
    assert abs(r30.pooled.proj) > 1.0 > abs(r1.pooled.proj), (
        f"|proj| went {abs(r1.pooled.proj):.4g} -> {abs(r30.pooled.proj):.4g}; "
        f"the mutation must cross 1.0 to be the E-DEC-18 regime")
    assert r30.pooled.ratio > 10.0, (
        f"|g_aux|/|g_traj| read {r30.pooled.ratio:.4g} at 30x; E-DEC-18's "
        f"measured regime is 10-30x and the detector must land in it")


def test_cosine_is_scale_invariant_analytically():
    """The blindness above is not an artefact of one batch: it is algebra, and
    it holds for any positive scale on either side."""
    g = torch.randn(64, generator=torch.Generator().manual_seed(1))
    h = torch.randn(64, generator=torch.Generator().manual_seed(2))
    base = cosine_stats([g], [h])["cos"]
    for s in (2.0, 4.0, 1024.0, 0.125):
        assert cosine_stats([g], [h * s])["cos"] == base
        assert cosine_stats([g * s], [h])["cos"] == base
    # ...and NEGATIVE scale flips it, exactly.
    assert cosine_stats([g], [h * -1.0])["cos"] == -base


def test_mutation_moves_the_cosine_only_by_moving_the_PARAMETERS():
    """⭐ The worked example. Over a short run the 30x arm's cosine DOES diverge
    from the 1x arm's -- but only because the 30x aux has dragged the trunk
    somewhere else. At step 0, on identical parameters, the two cosines are
    identical (previous test); by step N they are not. The cosine is a lagging
    indicator of a scale defect; ``ratio`` / ``proj`` are immediate ones."""
    def run(scale: float, steps: int = 12):
        m = _fresh()
        d = _det(m)
        opt = torch.optim.SGD(m.parameters(), lr=0.05)
        cos, ratio = [], []
        for i in range(steps):
            lt, la = _losses(m, _batch(seed=SEED + i))
            r = d.measure(lt * 1.0, la * scale)
            cos.append(r.pooled.cos)
            ratio.append(r.pooled.ratio)
            opt.zero_grad(set_to_none=True)
            (lt + scale * la).backward()
            opt.step()
        return cos, ratio

    c1, q1 = run(1.0)
    c30, q30 = run(30.0)
    assert abs(c30[0] - c1[0]) < 1e-6          # step 0: identical parameters
    assert q30[0] == pytest.approx(30.0 * q1[0], rel=1e-5)
    assert abs(c30[-1] - c1[-1]) > 1e-3, (     # step N: the trunk has moved
        "the 30x arm's cosine never separated from the 1x arm's even after the "
        "parameters diverged -- the detector has no trajectory signal either")
    # the magnitude channel separated from step 0 and stayed separated
    assert q30[-1] > q1[-1]


# ==========================================================================  #
#  3. PER PARAMETER GROUP
# ==========================================================================  #
def test_groups_are_reported_beside_the_pooled_number():
    """⛔ "The trunk is in conflict" and "the last stage is in conflict" imply
    different fixes, so the pooled number is never the only one logged."""
    m = _fresh()
    d = _det(m)
    lt, la = _losses(m, _batch())
    r = d.measure(lt, la)
    assert set(d.group_index) == {"stem", "stage_layer1", "stage_layer2"}
    assert r.groups[gc.POOLED_NAME] is r.pooled
    for name in ("stem", "stage_layer1", "stage_layer2", gc.HEADS_NAME):
        assert name in r.groups, f"{name} missing from the reading"
        assert r.groups[name].n_params > 0
    row = r.row()
    for name in ("stem", "stage_layer1", "stage_layer2"):
        for f in ("cos", "ratio", "proj", "gn_traj", "gn_aux"):
            assert f"cd_{name}_{f}" in row
    # the pooled parameter count is the sum of the group counts, so no
    # parameter is double-counted or silently dropped from the pooled number.
    assert r.pooled.n_params == sum(
        r.groups[g].n_params for g in d.group_index)


def test_heads_group_is_a_STRUCTURAL_control_reading_exactly_zero():
    """⭐ A control nobody has to configure. The planning loss and the perception
    loss touch DISJOINT head parameters, so their dot product there is exactly
    0.0 while both norms are non-zero -- a genuine orthogonality, and a standing
    check that the two losses are the two losses we think they are."""
    m = _fresh()
    d = _det(m)
    lt, la = _losses(m, _batch())
    h = d.measure(lt, la).groups[gc.HEADS_NAME]
    assert h.cos == 0.0 and h.degenerate is False
    assert h.norm_traj > 0.0 and h.norm_aux > 0.0
    assert h.dot == 0.0


def test_grouping_matches_both_real_trunks():
    """The group names come from the backbone's OWN naming, for both trunks
    refcv6 can build. ⚠️ Positional, never substring: ``layer1.0.conv1.weight``
    contains ``conv1`` and a substring rule files half of every ResNet stage
    under ``stem``."""
    assert default_group_of("conv1.weight") == "stem"
    assert default_group_of("bn1.bias") == "stem"
    assert default_group_of("layer1.0.conv1.weight") == "stage_layer1"
    assert default_group_of("net.layer4.2.bn3.bias") == "stage_layer4"
    assert default_group_of("stem.0.weight") == "stem"
    assert default_group_of("stages.2.0.conv1.weight") == "stage_stages2"
    assert default_group_of("fuse16.proj.weight") == "fuse"


def test_grouping_on_the_real_refc_encoder():
    """The in-repo REF-C trunk, built for real, groups into stem + 4 stages."""
    from tanitad.refs import refc
    torch.manual_seed(0)
    m = refc.RefCModel(refc.refc_smoke_config())
    assert gc.resolve_trunk_prefixes(m) == ("encoder.",)
    d = _det(m)
    assert set(d.group_index) == {
        "stem", "stage_stages0", "stage_stages1", "stage_stages2", "stage_stages3"}
    assert sum(len(v) for v in d.group_index.values()) == len(d.trunk_params)


def test_the_trunk_prefix_is_RESOLVED_because_the_two_models_differ():
    """⛔ MEASURED 2026-09-17, and the reason ``resolve_trunk_prefixes`` exists:
    ``RefCModel`` holds the trunk at ``encoder.`` and the trainer's own
    ``RefCV3Model`` WRAPS it at ``core.encoder.``. A hard-coded ``encoder.``
    selects ZERO parameters on the v3 model -- a detector that would have logged
    ``NaN`` for an entire run while ``config.json`` said it was on."""
    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = TwoHead()

    w = Wrapped()
    assert gc.resolve_trunk_prefixes(w) == ("core.encoder.",)
    # the two candidates are DISJOINT, so exactly one ever matches
    assert not "core.encoder.x".startswith("encoder.")
    d = GradientConflictDetector(w.named_parameters(), ConflictConfig(
        enabled=True, trunk_prefixes=gc.resolve_trunk_prefixes(w)))
    assert d.trunk_names and all(n.startswith("core.encoder.")
                                 for n in d.trunk_names)
    assert set(d.group_index) == {"stem", "stage_layer1", "stage_layer2"}


# ==========================================================================  #
#  4. THE FLAG -- off is bit-identical, and so is on
# ==========================================================================  #
def _train_steps(model, *, detector, steps: int = 5, scale: float = 1.0):
    """The trainer's own sequence: zero_grad -> [probe] -> backward -> clip -> step."""
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    rows = []
    for i in range(steps):
        lt, la = _losses(model, _batch(seed=SEED + i))
        total = lt + scale * la
        opt.zero_grad(set_to_none=True)
        if detector is not None:
            rows.append(detector.measure(lt, scale * la, step=i).row())
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
    return rows


def _state_bytes(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _assert_bitwise(a, b, what):
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), (
            f"{what}: `{k}` differs -- max |d| = "
            f"{(a[k].double() - b[k].double()).abs().max().item():.3e}")


def test_flag_off_constructs_NOTHING():
    """⛔ With the flag off, ``for_model`` returns ``None``: no module, no
    parameter list, no RNG draw, no ``metrics.jsonl`` key. The off path cannot
    differ from the pre-detector trainer even in an allocation."""
    m = _fresh()
    assert GradientConflictDetector.for_model(m, ConflictConfig(enabled=False)) is None
    assert _train_steps(_fresh(), detector=None) == []


def test_flag_off_is_BIT_IDENTICAL_on_a_fixed_seed():
    """⛔ The requirement, proven on every tensor of the state dict."""
    a = _fresh()
    _train_steps(a, detector=None)
    b = _fresh()
    _train_steps(b, detector=GradientConflictDetector.for_model(
        b, ConflictConfig(enabled=False)))
    _assert_bitwise(_state_bytes(a), _state_bytes(b), "flag OFF")


def test_flag_ON_in_probe_mode_is_ALSO_bit_identical():
    """⭐ Stronger than the requirement, and the reason this instrument can never
    be blamed for an arm's result: ``autograd.grad`` does not accumulate into
    ``.grad``, so the caller's own ``backward()`` writes exactly the bytes it
    wrote before. An arm with the detector ON and an arm with it off are the
    same arm."""
    a = _fresh()
    _train_steps(a, detector=None)
    b = _fresh()
    d = _det(b, mode=MODE_PROBE)
    rows = _train_steps(b, detector=d)
    _assert_bitwise(_state_bytes(a), _state_bytes(b), "flag ON (probe)")
    assert len(rows) == 5 and all("cd_cos" in r for r in rows)


def test_bit_identity_check_can_FAIL_so_it_is_not_vacuous():
    """⛔ MUTATION of the check itself. A comparison that cannot fail proves
    nothing, so here is a run that must differ -- and does."""
    a = _fresh()
    _train_steps(a, detector=None)
    b = _fresh()
    _train_steps(b, detector=None, scale=1.0001)
    with pytest.raises(AssertionError, match="differs"):
        _assert_bitwise(_state_bytes(a), _state_bytes(b), "mutated")


def test_reuse_mode_is_the_prereg_cost_and_is_NOT_bit_identical():
    """:data:`MODE_REUSE` is the prereg's "one extra backward over the trunk":
    the two partial gradients ARE the training gradient. It reproduces a fused
    ``backward()`` to float tolerance but NOT bitwise -- the two partial sums are
    rounded before they are added -- which is exactly why it is opt-in and
    ``probe`` is the default."""
    ref = _fresh()
    lt, la = _losses(ref, _batch())
    (lt + la).backward()
    want = {n: p.grad.clone() for n, p in ref.named_parameters() if p.grad is not None}

    got_m = _fresh()
    d = _det(got_m, mode=MODE_REUSE)
    lt2, la2 = _losses(got_m, _batch())
    d.measure(lt2, la2, retain_graph=False)
    n = d.accumulate_()
    assert n == len(want)
    got = {nm: p.grad for nm, p in got_m.named_parameters() if p.grad is not None}
    assert set(got) == set(want)
    for k in want:
        assert torch.allclose(got[k], want[k], rtol=1e-5, atol=1e-7), k
    assert not all(torch.equal(got[k], want[k]) for k in want), (
        "reuse mode came out bitwise equal on this batch; the claim in the "
        "docstring is then wrong in the user's favour, which is still wrong")


def test_subtract_mode_is_bit_identical_AND_recovers_the_plan_gradient():
    """⭐ :data:`MODE_SUBTRACT` -- ONE extra backward and the training step still
    bit-identical, because it READS ``.grad`` instead of writing it.

    Two claims, both checked: the step is unchanged, and ``g_plan = .grad -
    g_aux`` really is the gradient of ``total - aux`` (linearity), matched here
    against a ``probe`` reading of exactly that loss.
    """
    # -- the reference: a probe reading whose planning side IS `total - aux` -- #
    ref = _fresh()
    dref = _det(ref, mode=MODE_PROBE)
    lt, la = _losses(ref, _batch())
    want = dref.measure(lt, la).pooled            # `lt` IS total-minus-aux here

    got_m = _fresh()
    d = _det(got_m, mode=gc.MODE_SUBTRACT)
    lt2, la2 = _losses(got_m, _batch())
    (lt2 + la2).backward(retain_graph=True)
    r = d.measure_after_backward(la2)
    assert r.plan_side == "total_minus_aux"
    assert r.row()["cd_plan_side"] == "total_minus_aux"
    assert r.pooled.cos == pytest.approx(want.cos, rel=1e-4, abs=1e-6)
    assert r.pooled.ratio == pytest.approx(want.ratio, rel=1e-4)
    assert r.pooled.proj == pytest.approx(want.proj, rel=1e-4, abs=1e-6)
    # -- and the step it ran inside is bit-identical to the detector-free one -- #
    a = _fresh()
    _train_steps(a, detector=None)
    b = _fresh()
    db = _det(b, mode=gc.MODE_SUBTRACT)
    opt = torch.optim.SGD(b.parameters(), lr=0.05, momentum=0.9)
    for i in range(5):
        p, q = _losses(b, _batch(seed=SEED + i))
        opt.zero_grad(set_to_none=True)
        (p + q).backward(retain_graph=True)
        db.measure_after_backward(q, step=i)
        torch.nn.utils.clip_grad_norm_(b.parameters(), 10.0)
        opt.step()
    _assert_bitwise(_state_bytes(a), _state_bytes(b), "subtract mode")


def test_subtract_mode_refuses_before_the_backward_and_from_the_wrong_mode():
    """⛔ Called before the backward, every ``.grad`` is ``None`` and the
    planning side would be the zero vector -- a full run of degenerate NaNs that
    looks like a measurement. It refuses instead."""
    m = _fresh()
    d = _det(m, mode=gc.MODE_SUBTRACT)
    lt, la = _losses(m, _batch())
    with pytest.raises(RuntimeError, match="before the training backward"):
        d.measure_after_backward(la)
    with pytest.raises(RuntimeError, match="subtract"):
        _det(_fresh(), mode=MODE_PROBE).measure_after_backward(la)


def test_plan_side_is_stamped_on_every_row_so_the_two_are_never_compared():
    """⛔ ``probe`` and ``subtract`` answer against DIFFERENT planning losses.
    A row that does not say which would be silently comparable with the other."""
    m = _fresh()
    lt, la = _losses(m, _batch())
    assert _det(m, mode=MODE_PROBE).measure(lt, la).row()["cd_plan_side"] == "traj"
    assert gc.PLAN_SIDE[gc.MODE_SUBTRACT] == "total_minus_aux"
    assert set(gc.PLAN_SIDE) == {MODE_PROBE, MODE_REUSE, gc.MODE_SUBTRACT}


def test_reuse_mode_refuses_from_a_probe_detector_and_before_measure():
    m = _fresh()
    with pytest.raises(RuntimeError, match="probe"):
        _det(m, mode=MODE_PROBE).accumulate_()
    with pytest.raises(RuntimeError, match="before measure"):
        _det(m, mode=MODE_REUSE).accumulate_()


def test_flag_default_is_ON_for_refcv6_arms_and_OFF_otherwise():
    """⛔ ``SPEC_REFCV6_V2.md`` §6: the detector "runs in every arm" -- every
    *refcv6* arm. ``refc_v3_train`` stamps ``refcv6: null`` for the baseline
    precisely so that absence and baseline stay distinguishable."""
    flags = object()                       # any non-None refcv6 block
    assert enabled_for_arm(flags) is True
    assert enabled_for_arm(None) is False
    assert enabled_for_arm(flags, aux_present=False) is False   # no 2nd gradient
    assert enabled_for_arm(flags, override=False) is False      # --no-...
    assert enabled_for_arm(None, override=True) is True


# ==========================================================================  #
#  ⛔ MUTATION PROOFS OF THE GUARDS THEMSELVES
# ==========================================================================  #
def _lossy_cosine_stats(ga, gb, *, dtype=torch.float64):
    """THE DEFECT: ``sqrt(a) * sqrt(b)`` instead of ``sqrt(a*b)``. MEASURED over
    200,000 uniform draws, ``sqrt(S)*sqrt(S) != S`` for **46.8 %** of ``S``, and
    the quotient then misses ``1.0`` at the same rate -- so this is a defect that
    a single-seed control passes better than half the time."""
    # `cosine_stats` here is the name bound at import: the ORIGINAL function, so
    # monkeypatching the module attribute cannot make this recurse.
    st = cosine_stats(ga, gb, dtype=dtype)
    na, nb = st["norm_traj"], st["norm_aux"]
    st["cos"] = (st["dot"] / (na * nb)) if na and nb else float("nan")
    return st


def test_mutation_lossy_denominator_is_CAUGHT(monkeypatch):
    """⛔ Re-introduce the tempting spelling and watch the ``+1`` control refuse
    it. Without this, "the control passed" would only mean "nobody wrote the
    other formula yet".

    ⚠️ **The mutation is run over SEVERAL seeds, deliberately.** The lossy
    spelling is exact on ~53 % of batches, so a single-seed mutation test would
    itself be a coin flip -- it would report "the guard works" on a batch where
    the defect never appeared. The exact spelling must pass on **every** seed and
    the lossy one must fail on **at least one**; both halves are asserted.
    """
    seeds = [SEED + 100 * i for i in range(12)]
    exact_ok, lossy_failed = 0, []
    for s in seeds:
        m = _fresh(seed=s)
        d = _det(m)
        lt, la = _losses(m, _batch(seed=s))
        if d.controls(lt, la).ok:                       # the real implementation
            exact_ok += 1
        monkeypatch.setattr(gc, "cosine_stats", _lossy_cosine_stats)
        try:
            res = d.controls(lt, la)
            if not res.ok:
                lossy_failed.append((s, res))
        finally:
            monkeypatch.undo()
    assert exact_ok == len(seeds), (
        f"the shipped `sqrt(a*b)` spelling missed +1.0 on "
        f"{len(seeds) - exact_ok}/{len(seeds)} seeds -- the identity claim is wrong")
    assert lossy_failed, (
        "the lossy denominator passed on all 12 seeds; the mutation did not "
        "reintroduce the defect and this guard is unproven")
    s, res = lossy_failed[0]
    assert any("exactly +1.0" in f for f in res.failures), res.failures
    assert res.self_cos != 1.0
    # ...and the REFUSAL, not just the report
    m = _fresh(seed=s)
    d = _det(m)
    lt, la = _losses(m, _batch(seed=s))
    monkeypatch.setattr(gc, "cosine_stats", _lossy_cosine_stats)
    with pytest.raises(ControlFailure, match=r"did not read their known values"):
        d.self_check(lt, la)


def test_mutation_an_UNDETACHED_detached_control_is_CAUGHT():
    """⛔ The detached control's failure branch, REACHED. A subclass builds the
    control from the LIVE aux loss instead of a detached one; every one of the
    three detached assertions must fire."""
    class Broken(GradientConflictDetector):
        def controls(self, loss_traj, loss_aux=None, *, retain_graph=True):
            # the defect: "detached" is the live loss, so g_aux is NOT zero
            return super().controls(loss_traj, loss_aux, retain_graph=retain_graph)

    m = _fresh()
    d = Broken(m.named_parameters(), ConflictConfig(enabled=True))
    lt, la = _losses(m, _batch())
    # patch the one line that makes the control a control
    orig = torch.Tensor.detach
    try:
        torch.Tensor.detach = lambda self: self          # type: ignore[assignment]
        res = d.controls(lt, la)
    finally:
        torch.Tensor.detach = orig                       # type: ignore[assignment]
    assert res.ok is False
    joined = " | ".join(res.failures)
    assert "not detached" in joined or "must be\nexactly 0.0" in joined or \
           "must be exactly 0.0" in joined, res.failures
    assert res.detached_norm_aux > 0.0
    assert not math.isnan(res.detached_cos)


def test_probe_REFUSES_an_empty_or_frozen_trunk():
    """⛔ A detector over the empty set logs NaN forever and looks like a
    measurement. It refuses instead, and says which case it is."""
    m = _fresh()
    # (a) the prefix matches nothing -- the wiring bug that bit the v3 model
    with pytest.raises(ValueError, match="NO parameter matches the prefix"):
        GradientConflictDetector(m.named_parameters(),
                                 ConflictConfig(enabled=True,
                                                trunk_prefixes=("nosuch.",)))
    with pytest.raises(ValueError, match="none of the trunk prefixes"):
        gc.resolve_trunk_prefixes(m, candidates=("nosuch.",))
    # (b) the prefix matches, but the trunk is frozen -- a different diagnosis
    for p in m.encoder.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="match the prefix but are FROZEN"):
        _det(m)


def test_probe_REFUSES_a_loss_with_no_graph():
    """A loss that does not require grad would read a vacuous NaN forever."""
    m = _fresh()
    d = _det(m)
    lt, la = _losses(m, _batch())
    with pytest.raises(ValueError, match="does not require grad"):
        d.measure(lt, la.detach())


def test_config_refuses_a_nonsense_setting():
    with pytest.raises(ValueError, match="mode must be"):
        ConflictConfig(enabled=True, mode="clip")
    with pytest.raises(ValueError, match="every must be"):
        ConflictConfig(enabled=True, every=0)
    with pytest.raises(ValueError, match="no trunk prefix"):
        ConflictConfig(enabled=True, trunk_prefixes=())


# ==========================================================================  #
#  the log row
# ==========================================================================  #
def test_row_is_unrounded_and_carries_the_receipts():
    """⛔ NOT rounded by the detector: a real 1e-8 reading rounded to 5 dp is
    0.0, which is the exact signature of the defect this measures. The trainer
    merges this AFTER its own rounding pass, the way ``_grad_probe_row`` is."""
    m = _fresh()
    d = _det(m)
    lt, la = _losses(m, _batch())
    row = d.measure(lt, la * 1e-9, step=7).row()
    for k in ("cd_cos", "cd_conflict", "cd_ratio", "cd_proj", "cd_gn_traj",
              "cd_gn_aux", "cd_degenerate", "cd_n_params", "cd_ms"):
        assert k in row
    assert 0.0 < row["cd_gn_aux"] < 1e-5
    assert round(row["cd_gn_aux"], 5) == 0.0, (
        "the caller's 5-dp rounding would have erased this gradient; that is "
        "why the detector's row is merged after it")
    assert row["cd_degenerate"] == 0.0


def test_provenance_names_the_trunk_it_measured():
    m = _fresh()
    d = _det(m)
    lt, la = _losses(m, _batch())
    d.self_check(lt, la)
    p = d.provenance()
    assert p["config"]["enabled"] is True and p["config"]["mode"] == MODE_PROBE
    assert p["n_trunk_params"] == sum(x.numel() for x in m.encoder.parameters())
    assert p["n_head_params"] == sum(
        x.numel() for n, x in m.named_parameters() if not n.startswith("encoder."))
    assert p["controls"]["ok"] is True and p["controls"]["self_cos"] == 1.0
    assert sum(g["n_params"] for g in p["groups"].values()) == p["n_trunk_params"]


# --------------------------------------------------------------------------- #
# ⛔⛔ THE DEVICE ARM. Every `cosine_stats` test above runs on the CPU — 12 of
# them, zero CUDA references — and that is exactly why this defect shipped.
#
# MEASURED 2026-09-17, on the FIRST GPU run of the refcv6 chain (the §10.6
# pipeline validation): `cosine_stats` built its three accumulators with
# `torch.zeros((), dtype=dtype)` and no `device=`, so they landed on the CPU and
# `saq += av.dot(av)` raised
#
#     RuntimeError: Expected all tensors to be on the same device,
#                   but found at least two devices, cuda:0 and cpu!
#
# at step 0, inside the detector's OWN self-check. The detector had been
# mutation-proven, benchmarked and shipped — all on CPU. A guard exercised only
# on one device is a guard that has never met the other one.
#
# ⚠️ A SKIP HERE IS INCONCLUSIVE, NEVER A PASS. On a box with no CUDA this arm
# reports nothing and the defect would be invisible again; that is stated rather
# than hidden behind a green run.
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="no CUDA on this box — this arm is INCONCLUSIVE here, "
                           "not passing; the defect it guards is device-specific")
def test_REGRESSION_cosine_stats_accumulates_on_the_GRADIENTS_device():
    """⛔ The exact call that died. Before the fix this RAISED; it must now run.

    Mixed ``None``/tensor on purpose: the ``None`` branches are where the
    accumulator is touched without a matching partner, which is where the
    device mismatch first surfaces.
    """
    ga = [torch.randn(64, device="cuda"), None, torch.randn(32, device="cuda")]
    gb = [torch.randn(64, device="cuda"), torch.randn(16, device="cuda"), None]
    r = cosine_stats(ga, gb)
    assert r["n_tensors"] == 3 and r["n_params"] == 112
    assert math.isfinite(r["cos"])


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="no CUDA on this box — INCONCLUSIVE, not passing")
def test_the_plus_and_minus_one_identities_hold_EXACTLY_on_cuda_too():
    """⭐ The docstring promises the controls are IDENTITIES, not tolerances.

    That promise is what the float64 accumulation is for, and it has to survive
    the move onto the device — a fix that made the error go away while turning
    ``1.0`` into ``0.9999997`` would have broken the thing the module exists to
    report.
    """
    g = [torch.randn(128, device="cuda"), torch.randn(7, 5, device="cuda")]
    assert cosine_stats(g, g)["cos"] == 1.0
    assert cosine_stats(g, [-t for t in g])["cos"] == -1.0


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="no CUDA on this box — INCONCLUSIVE, not passing")
def test_the_cpu_path_is_UNCHANGED_by_the_device_fix():
    """⛔ The control that keeps the fix honest. Deriving the device from the
    gradients must not quietly move CPU work onto the GPU, or every existing
    test above would be measuring something new."""
    c = [torch.randn(50), torch.randn(4, 4)]
    assert cosine_stats(c, c)["cos"] == 1.0
    assert cosine_stats(c, [-t for t in c])["cos"] == -1.0


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="no CUDA on this box — INCONCLUSIVE, not passing")
def test_an_ALL_NONE_side_still_degenerates_rather_than_crashing_on_cuda():
    """⚠️ ``dev`` is derived from the first non-``None`` tensor across BOTH
    lists, so a side that is entirely ``None`` must still find a device from the
    other one — and the zero-norm side must read ``degenerate``, never a
    tolerated epsilon."""
    g = [torch.randn(16, device="cuda")]
    r = cosine_stats([None], g)
    assert r["degenerate"] is True and math.isnan(r["cos"])
